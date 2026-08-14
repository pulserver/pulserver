/**
 * @file autolabel.cpp
 * @brief Deriving encoding counters from a trajectory.  See autolabel.hpp.
 */

#include "pulseq/autolabel.hpp"

#include "pulseq/sequence.hpp"
#include "pulseq/shape.hpp"

/* The RF spectrum, at the C library's ordinary precision.  Safe here because
 * this translation unit never sees raw64.hpp -- see the warning on that file. */
#include "pulseq_rf.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulseq
{
    namespace
    {
        using Vec3 = std::array<double, 3>;

        /**
         * mirror, then reflect, then reorder -- MATLAB's order, and none of
         * the three commutes with the others.
         *
         * @p mirror is what separates an encoding quantity from a slice one.
         * `mirror_fourier` negates the Fourier directions and leaves slice
         * ordering alone, so it applies to k and to readout gradients but not
         * to slice positions or slice-select gradients; `reflect` and
         * `reorder` apply to all of them.  Hence the flag rather than one
         * function: the two groups differ in this and in nothing else.
         */
        Vec3 orient(const Vec3& v, const AutoLabelOptions& o, bool mirror)
        {
            Vec3 reflected = v;
            if (mirror && o.mirror_fourier)
            {
                for (int a = 0; a < 3; ++a)
                    reflected[static_cast<size_t>(a)] = -reflected[static_cast<size_t>(a)];
            }
            for (int a = 0; a < 3; ++a)
            {
                if (o.reflect[static_cast<size_t>(a)])
                    reflected[static_cast<size_t>(a)] = -reflected[static_cast<size_t>(a)];
            }
            Vec3 out{{0.0, 0.0, 0.0}};
            for (int a = 0; a < 3; ++a)
            {
                const int src = o.reorder[static_cast<size_t>(a)];
                out[static_cast<size_t>(a)] = reflected[static_cast<size_t>(src)];
            }
            return out;
        }

        /** k, and the gradients that encode it: `mirror_fourier` applies. */
        Vec3 orient_encoding(const Vec3& v, const AutoLabelOptions& o)
        {
            return orient(v, o, true);
        }

        /** Slice positions and slice-select gradients: it does not. */
        Vec3 orient_slice(const Vec3& v, const AutoLabelOptions& o)
        {
            return orient(v, o, false);
        }

        /**
         * A three-number definition, per axis and oriented like k.
         *
         * Zero where the definition is absent, too short, or non-positive on
         * that axis -- every caller reads zero as "nothing stated", so a file
         * that gives only some of the numbers is no worse than one that gives
         * none.  Only `reorder` applies: these are lengths and counts, so a
         * reflected axis is the same axis.
         */
        std::array<double, 3> oriented_definition(
            const Sequence& seq, const char* key, const AutoLabelOptions& o)
        {
            std::array<double, 3> out{{0.0, 0.0, 0.0}};
            const Definition* def = seq.definition(key);
            if (def == nullptr || def->numbers().size() < 3)
                return out;
            for (int a = 0; a < 3; ++a)
            {
                const double v =
                    def->numbers()[static_cast<size_t>(o.reorder[static_cast<size_t>(a)])];
                out[static_cast<size_t>(a)] = (v > 0.0) ? v : 0.0;
            }
            return out;
        }

        double norm3(const Vec3& v)
        {
            return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        }

        double dot3(const Vec3& a, const Vec3& b)
        {
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        }

        /** Index of the largest component by magnitude. */
        int dominant(const Vec3& v)
        {
            int best = 0;
            for (int a = 1; a < 3; ++a)
            {
                if (std::fabs(v[static_cast<size_t>(a)]) > std::fabs(v[static_cast<size_t>(best)]))
                    best = a;
            }
            return best;
        }

        /**
         * The unit normal a slice-select gradient defines.
         *
         * Signed so its dominant component is positive, which is what makes
         * the offset of a slice independent of the sign the gradient happened
         * to be played with -- a refocused or reversed slice select selects the
         * same slice.
         */
        Vec3 slice_normal(const Vec3& g)
        {
            const double n = norm3(g);
            Vec3 out{{0.0, 0.0, 0.0}};
            if (n <= 0.0)
                return out;
            const double sign = (g[static_cast<size_t>(dominant(g))] < 0.0) ? -1.0 : 1.0;
            for (int a = 0; a < 3; ++a)
                out[static_cast<size_t>(a)] = sign * g[static_cast<size_t>(a)] / n;
            return out;
        }

        double median_of(std::vector<double> v)
        {
            if (v.empty())
                return 0.0;
            const size_t mid = v.size() / 2;
            std::nth_element(v.begin(), v.begin() + static_cast<long>(mid), v.end());
            const double hi = v[mid];
            if (v.size() % 2 == 1)
                return hi;
            const double lo = *std::max_element(v.begin(), v.begin() + static_cast<long>(mid));
            return 0.5 * (lo + hi);
        }

        /**
         * The spectral bandwidth of the RF pulse in block @p block_index, Hz.
         *
         * Rebuilds the complex pulse from the file's own three shapes --
         * magnitude, phase and, when present, an explicit time base -- and
         * hands it to the shared estimator in `csrc/src/pulseq/pulseq_rf.c`,
         * the same one pulseg's RF statistics use.  The amplitude scalar is
         * left out: the width is measured relative to the pulse's own peak, so
         * scaling it cannot move the answer.
         *
         * Returns 0 when the block has no pulse or the spectrum is not
         * measurable -- never an analytic stand-in, which at this layer would
         * be indistinguishable from a measurement.
         */
        double rf_bandwidth(const Sequence& seq, int block_index)
        {
            const Block block = seq.get_block(block_index);
            if (block.rf == 0)
                return 0.0;

            const double* row = seq.rf_library().row(block.rf);
            const int mag_id = static_cast<int>(row[1]);
            const int phase_id = static_cast<int>(row[2]);
            const int time_id = static_cast<int>(row[3]);
            const double center = row[4];
            if (mag_id <= 0)
                return 0.0;

            const ShapeLibrary& shapes = seq.shape_library();
            const std::vector<double> magnitude = decompress_shape(
                shapes.samples(mag_id), shapes.num_compressed(mag_id),
                shapes.num_uncompressed(mag_id));
            const int n = static_cast<int>(magnitude.size());
            if (n < 2)
                return 0.0;

            std::vector<double> phase;
            if (phase_id > 0)
            {
                phase = decompress_shape(shapes.samples(phase_id),
                                         shapes.num_compressed(phase_id),
                                         shapes.num_uncompressed(phase_id));
                if (static_cast<int>(phase.size()) != n)
                    phase.clear();
            }

            /* Times in microseconds, which is what the estimator takes.  With
             * no time shape the samples sit on the RF raster; with one, the
             * shape holds raster counts, as the file format defines it. */
            const double raster_us = seq.rf_raster_time() * 1e6;
            std::vector<float> t(static_cast<size_t>(n));
            if (time_id > 0)
            {
                const std::vector<double> tt = decompress_shape(
                    shapes.samples(time_id), shapes.num_compressed(time_id),
                    shapes.num_uncompressed(time_id));
                if (static_cast<int>(tt.size()) != n)
                    return 0.0;
                for (int i = 0; i < n; ++i)
                    t[static_cast<size_t>(i)] = static_cast<float>(tt[static_cast<size_t>(i)] *
                                                                   raster_us);
            }
            else
            {
                for (int i = 0; i < n; ++i)
                    t[static_cast<size_t>(i)] = static_cast<float>((i + 0.5) * raster_us);
            }

            /* Phase is carried in turns, so a full circle is 1. */
            std::vector<float> re(static_cast<size_t>(n));
            std::vector<float> im(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i)
            {
                const double m = magnitude[static_cast<size_t>(i)];
                const double p =
                    phase.empty() ? 0.0 : 2.0 * M_PI * phase[static_cast<size_t>(i)];
                re[static_cast<size_t>(i)] = static_cast<float>(m * std::cos(p));
                im[static_cast<size_t>(i)] = static_cast<float>(m * std::sin(p));
            }

            pulseq_rf_spectrum* plan = nullptr;
            if (pulseq_rf_spectrum_create(&plan, static_cast<float>(raster_us),
                                          static_cast<float>(PULSEQ_RF_DEFAULT_RESOLUTION_HZ)) < 0)
                return 0.0;

            double bandwidth = 0.0;
            if (pulseq_rf_spectrum_run(plan, re.data(), im.data(), t.data(), n,
                                       static_cast<float>(center * 1e6)) >= 0)
                bandwidth = pulseq_rf_bandwidth(
                    plan, static_cast<float>(PULSEQ_RF_DEFAULT_CUTOFF), nullptr);
            pulseq_rf_spectrum_free(plan);
            return bandwidth;
        }

        /**
         * The shape of the repeat nest, read from when each k-space position
         * came back.
         *
         * @param visits  per k-space position, the acquisition indices that
         *                landed on it, in order.
         * @param levels  how many dimensions the caller named.
         * @return        their sizes, outermost first, with product equal to
         *                the number of visits.
         *
         * ### What is being read
         *
         * A dimension nested *inside* the k-space loop brings a position back
         * almost immediately -- the two echoes of a multi-echo train are
         * consecutive acquisitions.  One *outside* it brings the position back
         * only after a whole pass over k-space.  So the gaps between a
         * position's successive visits fall into as many distinct sizes as
         * there are levels, and the run lengths between the large ones give
         * the sizes: `1, 1, L, 1, 1, L, ...` is two echoes inside, three
         * frames outside.
         *
         * The gaps are compared by *rank*, not by value.  What identifies a
         * level is that its gap is larger than the one below, and turning the
         * gaps into rank indices makes the comparison across positions
         * meaningful without assuming the passes are equally long.
         *
         * ### What it refuses, and why it refuses so much
         *
         * A wrong split is not a degraded answer.  It puts two different
         * acquisitions into one slot, which a reconstruction then averages
         * together -- and every label around it still looks ordinary.  So this
         * only answers when the evidence is unambiguous:
         *
         *  - every position visited the same number of times.  Three
         *    navigators on one line of an EPI are not a dimension: 127
         *    positions seen once and one seen four times is a ragged shape
         *    with no nest in it;
         *  - every position showing the *same* rank pattern, so the nest is
         *    one nest rather than a different one per line;
         *  - the run lengths dividing the visits exactly, and there being as
         *    many levels as were named.
         *
         * Anything else throws.  A caller who knows better than the evidence
         * can still say so by giving the sizes outright.
         */
        std::vector<int> infer_repeat_sizes(const std::vector<std::vector<int>>& visits,
                                            size_t levels)
        {
            if (levels == 0)
                return std::vector<int>();

            /*
             * One name takes the whole count, and there is nothing to read --
             * so nothing to refuse either.  This is deliberately ahead of the
             * rectangle check: an EPI's three navigators on one line are not a
             * nest, but calling them `SET` is still a perfectly good thing to
             * ask for, and it is the same number `REP` would have carried.
             */
            if (levels == 1)
            {
                size_t most = 0;
                for (size_t p = 0; p < visits.size(); ++p)
                    most = std::max(most, visits[p].size());
                return std::vector<int>(1, static_cast<int>(std::max<size_t>(most, 1)));
            }

            size_t count = 0;
            for (size_t p = 0; p < visits.size(); ++p)
            {
                if (p == 0)
                    count = visits[p].size();
                else if (visits[p].size() != count)
                    throw std::runtime_error(
                        "auto_label: the repeats are not a rectangle -- one k-space position "
                        "was acquired " + std::to_string(count) + " time(s) and another " +
                        std::to_string(visits[p].size()) +
                        ", so there is no nest of dimensions to read out of them. Give the "
                        "sizes explicitly if you know what they are.");
            }
            if (count == 0 || count == 1)
                return std::vector<int>(levels, 1);

            /* Gaps as rank indices, and the same ranks for every position. */
            std::vector<int> ranks;
            for (size_t p = 0; p < visits.size(); ++p)
            {
                std::vector<long> gaps;
                gaps.reserve(count - 1);
                for (size_t j = 1; j < visits[p].size(); ++j)
                    gaps.push_back(static_cast<long>(visits[p][j]) -
                                   static_cast<long>(visits[p][j - 1]));

                std::vector<long> distinct = gaps;
                std::sort(distinct.begin(), distinct.end());
                distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());

                std::vector<int> here(gaps.size(), 0);
                for (size_t j = 0; j < gaps.size(); ++j)
                    here[j] = static_cast<int>(
                        std::lower_bound(distinct.begin(), distinct.end(), gaps[j]) -
                        distinct.begin());

                if (p == 0)
                    ranks = here;
                else if (here != ranks)
                    throw std::runtime_error(
                        "auto_label: the k-space positions are not all revisited in the same "
                        "pattern, so the repeats do not form one nest of dimensions. Give the "
                        "sizes explicitly if you know what they are.");
            }

            /*
             * Peel the nest from the inside out.  The innermost dimension is
             * the leading run of rank-0 gaps plus one; dropping those visits
             * leaves the gaps at the next level up, and the same rule applies
             * again.
             */
            std::vector<int> inner_first;
            size_t remaining = count;
            std::vector<int> level = ranks;
            for (size_t d = 0; d < levels; ++d)
            {
                if (d + 1 == levels)
                {
                    inner_first.push_back(static_cast<int>(remaining));
                    remaining = 1;
                    break;
                }

                size_t run = 0;
                while (run < level.size() && level[run] == level[0])
                    ++run;
                const size_t size = run + 1;

                if (size < 1 || remaining % size != 0)
                    throw std::runtime_error(
                        "auto_label: the repeat pattern does not divide into the " +
                        std::to_string(levels) + " dimensions named -- read a size of " +
                        std::to_string(size) + " out of " + std::to_string(remaining) +
                        " visits, which is not a whole number of them.");

                inner_first.push_back(static_cast<int>(size));
                remaining /= size;

                /* The gaps that survive are the ones between groups. */
                std::vector<int> next;
                for (size_t j = size - 1; j < level.size(); j += size)
                    next.push_back(level[j]);
                level = next;
            }

            if (remaining != 1)
                throw std::runtime_error(
                    "auto_label: the " + std::to_string(levels) +
                    " dimensions named account for fewer repeats than were found.");

            std::vector<int> out(inner_first.rbegin(), inner_first.rend());
            long product = 1;
            for (size_t d = 0; d < out.size(); ++d)
                product *= out[d];
            if (product != static_cast<long>(count))
                throw std::runtime_error(
                    "auto_label: read repeat dimensions whose product is " +
                    std::to_string(product) + ", but each k-space position was acquired " +
                    std::to_string(count) + " times.");
            return out;
        }

        /** Whether the caller asked for this counter to be left alone. */
        bool skipped(const AutoLabelOptions& options, const char* name)
        {
            for (size_t i = 0; i < options.skip.size(); ++i)
            {
                if (options.skip[i] == name)
                    return true;
            }
            return false;
        }

        /** MATLAB's `any(x ~= 0)`: whether a counter is ever set. */
        bool any_nonzero(const std::vector<int>& v)
        {
            for (size_t i = 0; i < v.size(); ++i)
            {
                if (v[i] != 0)
                    return true;
            }
            return false;
        }

    }  // namespace

    std::vector<std::pair<std::string, const std::vector<int>*>> AutoLabels::present() const
    {
        std::vector<std::pair<std::string, const std::vector<int>*>> out;
        /* The order counters are emitted in.  Fixed rather than incidental so
         * two runs produce the same extension library and a diff of two
         * labelled files is about the labels. */
        if (!noise.empty())
            out.emplace_back("NOISE", &noise);
        if (!slc.empty())
            out.emplace_back("SLC", &slc);
        if (!rev.empty())
            out.emplace_back("REV", &rev);
        if (!lin.empty())
            out.emplace_back("LIN", &lin);
        if (!par.empty())
            out.emplace_back("PAR", &par);
        if (!rep.empty())
            out.emplace_back("REP", &rep);
        /* Declaration order, after the derived counters: the caller chose it,
         * and it is the order the dimensions vary in. */
        for (size_t i = 0; i < named.size(); ++i)
        {
            if (!named[i].second.empty())
                out.emplace_back(named[i].first, &named[i].second);
        }
        return out;
    }

    /* ================================================================== */
    /*  Detection                                                         */
    /* ================================================================== */

    AutoLabelResult detect_labels(const KSpace& ks, const Sequence& seq,
                                  const AutoLabelOptions& options)
    {
        /* A reorder that is not a permutation would silently duplicate one
         * axis and drop another, and every number below would still look
         * plausible. */
        {
            std::array<bool, 3> seen{{false, false, false}};
            for (int a = 0; a < 3; ++a)
            {
                const int src = options.reorder[static_cast<size_t>(a)];
                if (src < 0 || src > 2 || seen[static_cast<size_t>(src)])
                    throw std::runtime_error("auto_label: reorder is not a permutation of 0,1,2");
                seen[static_cast<size_t>(src)] = true;
            }
        }

        /* The counters to leave alone.  A name that is not one of them cannot
         * be honoured and is far more likely a typo than a request, so it is
         * refused rather than quietly ignored. */
        {
            for (size_t s = 0; s < options.skip.size(); ++s)
            {
                const std::string& name = options.skip[s];
                if (name != "NOISE" && name != "SLC" && name != "REV" && name != "LIN" &&
                    name != "PAR" && name != "REP")
                    throw std::runtime_error(
                        "auto_label: " + name +
                        " is not a counter this derives, so there is nothing to skip. It is "
                        "left alone already -- only NOISE, SLC, REV, LIN, PAR and REP are "
                        "written here.");
            }
        }

        /* The declared dimensions, checked before anything is computed: a
         * declaration that cannot be honoured is the caller's mistake, and
         * saying so before the work is done is cheaper for both of us. */
        {
            const std::vector<RepeatDim>& dims = options.repeat_dims;
            for (size_t d = 0; d < dims.size(); ++d)
            {
                const std::string& name = dims[d].name;
                if (name.empty())
                    throw std::runtime_error("auto_label: a repeat dimension has no name");
                if (name == "NOISE" || name == "SLC" || name == "REV" || name == "LIN" ||
                    name == "PAR")
                    throw std::runtime_error(
                        "auto_label: " + name +
                        " is derived from the trajectory, so it cannot also be a repeat "
                        "dimension");
                for (size_t e = 0; e < d; ++e)
                {
                    if (dims[e].name == name)
                        throw std::runtime_error("auto_label: repeat dimension " + name +
                                                 " is declared twice");
                }
                /* 0 means "read it from the acquisition order"; a negative
                 * number is not a size at all. */
                if (dims[d].size < 0)
                    throw std::runtime_error("auto_label: repeat dimension " + name +
                                             " has a negative size");
            }
        }

        const bool skip_noise = skipped(options, "NOISE");
        const bool skip_slc = skipped(options, "SLC");
        const bool skip_rev = skipped(options, "REV");
        const bool skip_lin = skipped(options, "LIN");
        const bool skip_par = skipped(options, "PAR");
        const bool skip_rep = skipped(options, "REP");

        AutoLabelResult result;
        result.key_groups = ks.key_groups;
        result.num_readouts = static_cast<int>(ks.readouts.size());

        const int n_adc = static_cast<int>(ks.readouts.size());
        if (n_adc == 0)
            return result;

        result.labels.adc_block.reserve(static_cast<size_t>(n_adc));
        for (const Readout& r : ks.readouts)
            result.labels.adc_block.push_back(r.block_index);

        /* -- slice offsets --------------------------------------------- */
        /*
         * An excitation's frequency offset over its gradient is a position
         * along the gradient direction; projecting the per-axis positions on
         * the signed unit normal collapses the three into the one number that
         * identifies the slice.
         */
        const int n_exc = static_cast<int>(ks.excitations.size());
        std::vector<double> slice_offset(static_cast<size_t>(n_exc), 0.0);
        for (int i = 0; i < n_exc; ++i)
        {
            const Vec3 g = orient_slice(ks.excitations[static_cast<size_t>(i)].g, options);
            const Vec3 pos = orient_slice(ks.slice_pos[static_cast<size_t>(i)], options);
            const Vec3 n = slice_normal(g);
            const double offset = dot3(pos, n);
            slice_offset[static_cast<size_t>(i)] = std::isfinite(offset) ? offset : 0.0;
        }

        /* -- noise scans ------------------------------------------------ */
        /*
         * An ADC before the first excitation has not sampled a signal, so it
         * is not part of the encoding and is excluded from every counter
         * derived below.  With no excitation at all there is no signal to
         * label and the whole scan is noise.
         */
        int first_signal = n_adc;
        if (n_exc > 0)
        {
            const double t_first = ks.excitations[0].t;
            for (int i = 0; i < n_adc; ++i)
            {
                if (ks.readouts[static_cast<size_t>(i)].t0 > t_first)
                {
                    first_signal = i;
                    break;
                }
            }
        }
        const int n_signal = n_adc - first_signal;

        std::vector<int> noise(static_cast<size_t>(n_adc), 0);
        for (int i = 0; i < first_signal; ++i)
            noise[static_cast<size_t>(i)] = 1;
        if (first_signal > 0 && !skip_noise)
            result.labels.noise = noise;
        if (n_signal <= 0)
            return result;

        /* -- slice counter per ADC -------------------------------------- */
        /*
         * The excitation an ADC belongs to is the last one before it.  Both
         * lists are in time order, so one merge walk does it.
         *
         * Two things about the table this builds.
         *
         * **It covers the slices that were acquired**, not every excitation in
         * the file.  A scan that plays dummy TRs before acquiring -- five of
         * them in gre_2d_3sl_3avg -- would otherwise have those dummies claim
         * slice indices, and a dummy at a position that is never acquired
         * would put a slice in `SlicePositions` that the recon allocates and
         * never fills.  `autoLabel.m:122` uniques over all of them and does
         * carry that entry.
         *
         * **It is ordered by position, not by acquisition.**  SLC is a
         * geometric index -- it says where a slice is, and `SlicePositions`
         * beside it says where that is in metres -- so the counter has to
         * follow the prescription and not the order the scanner chose to visit
         * it in.  An interleaved acquisition (0, 2, 4, 1, 3) is the case that
         * separates the two: indexing by first occurrence would number those
         * five slices 0..4 in the order they arrive and hand the recon a stack
         * shuffled into acquisition order.  Nothing about the prescription is
         * assumed here -- not that the slices are evenly spaced, not that
         * there are no gaps, not that they are contiguous.  The positions come
         * from the pulses' own frequency offsets and are sorted; whatever
         * spacing that produces is the answer.
         *
         * MATLAB computes both orderings (`sliceCountersAcquisitionOrder` and
         * `sliceCountersSorted`, lines 122 and 128) and labels with the first.
         *
         * @warning The offsets are read from the RF frequency offsets *as
         * authored*.  `TransformFOV` carries a translation as phase and leaves
         * them alone, but a scale multiplies the slice-select gradient without
         * touching the offset, so a slice that was at 5 mm reads as being
         * somewhere else afterwards.  Label first, transform second.
         *
         * The tolerance replaces MATLAB's exact `unique`, which would split
         * one slice in two over a last-bit difference between two excitations
         * of it.  It is relative to the spread, so it scales with the scan;
         * nanometres against millimetre slice spacing cannot merge two real
         * slices.
         */
        std::vector<double> unique_slices;
        /** The same positions, always ascending: the slice gap is measured on
         *  these whatever indexing @ref SliceSorting asked for. */
        std::vector<double> ascending_slices;
        std::vector<int> slice_of_adc(static_cast<size_t>(n_adc), 0);
        {
            double widest = 0.0;
            for (int i = 0; i < n_exc; ++i)
                widest = std::max(widest, std::fabs(slice_offset[static_cast<size_t>(i)]));
            const double tol = (widest > 0.0) ? 1e-6 * widest : 1e-12;

            /* Pass one: which distinct positions were acquired, and which one
             * each ADC belongs to.  Grouped in arrival order for now -- the
             * ranking is a second pass, because the last slice to arrive can
             * still be the first one in space. */
            std::vector<int> group_of_adc(static_cast<size_t>(n_adc), -1);
            int e = 0;
            for (int i = first_signal; i < n_adc; ++i)
            {
                const double t = ks.readouts[static_cast<size_t>(i)].t0;
                while (e + 1 < n_exc && ks.excitations[static_cast<size_t>(e + 1)].t < t)
                    ++e;
                if (n_exc == 0)
                    continue;

                const double v = slice_offset[static_cast<size_t>(e)];
                int found = -1;
                for (size_t s = 0; s < unique_slices.size(); ++s)
                {
                    if (std::fabs(unique_slices[s] - v) <= tol)
                    {
                        found = static_cast<int>(s);
                        break;
                    }
                }
                if (found < 0)
                {
                    found = static_cast<int>(unique_slices.size());
                    unique_slices.push_back(v);
                }
                group_of_adc[static_cast<size_t>(i)] = found;
            }

            /*
             * Pass two: decide which slice gets which index.  `rank[g]` is
             * where arrival-order group g ends up, and `unique_slices` is
             * reordered to match, so `unique_slices[SLC]` is the position of
             * slice `SLC` under every sorting.
             *
             * The ascending order is computed regardless, because the slice
             * gap is the closest spacing between adjacent positions and that
             * is a geometric fact rather than an indexing choice -- MATLAB
             * likewise measures it before it applies `descending`.
             */
            std::vector<int> order(unique_slices.size());
            for (size_t s = 0; s < order.size(); ++s)
                order[s] = static_cast<int>(s);
            std::sort(order.begin(), order.end(), [&unique_slices](int a, int b) {
                return unique_slices[static_cast<size_t>(a)] <
                       unique_slices[static_cast<size_t>(b)];
            });

            ascending_slices.resize(order.size());
            for (size_t s = 0; s < order.size(); ++s)
                ascending_slices[s] = unique_slices[static_cast<size_t>(order[s])];

            if (options.sort_slices == SliceSorting::Descending)
                std::reverse(order.begin(), order.end());

            std::vector<int> rank(unique_slices.size(), 0);
            if (options.sort_slices == SliceSorting::Acquisition)
            {
                /* Arrival order: the group index is already the answer, and
                 * the positions stay as they were met. */
                for (size_t s = 0; s < rank.size(); ++s)
                    rank[s] = static_cast<int>(s);
            }
            else
            {
                std::vector<double> sorted_positions(unique_slices.size(), 0.0);
                for (size_t s = 0; s < order.size(); ++s)
                {
                    rank[static_cast<size_t>(order[s])] = static_cast<int>(s);
                    sorted_positions[s] = unique_slices[static_cast<size_t>(order[s])];
                }
                unique_slices.swap(sorted_positions);
            }

            for (int i = first_signal; i < n_adc; ++i)
            {
                const int g = group_of_adc[static_cast<size_t>(i)];
                if (g >= 0)
                    slice_of_adc[static_cast<size_t>(i)] = rank[static_cast<size_t>(g)];
            }
        }

        /* -- readout direction, polarity and the Cartesian test ---------- */
        std::vector<Vec3> g_echo(static_cast<size_t>(n_adc));
        std::vector<Vec3> k_echo(static_cast<size_t>(n_adc));
        std::vector<int> sign_readout(static_cast<size_t>(n_adc), 0);

        for (int i = 0; i < n_adc; ++i)
        {
            g_echo[static_cast<size_t>(i)] =
                orient_encoding(ks.readouts[static_cast<size_t>(i)].g_echo, options);
            k_echo[static_cast<size_t>(i)] =
                orient_encoding(ks.readouts[static_cast<size_t>(i)].k_echo, options);
        }

        const Vec3& g_first = g_echo[static_cast<size_t>(first_signal)];
        const double s_first =
            (g_first[static_cast<size_t>(dominant(g_first))] < 0.0) ? -1.0 : 1.0;

        for (int i = first_signal; i < n_adc; ++i)
        {
            const Vec3& g = g_echo[static_cast<size_t>(i)];
            const double s = (g[static_cast<size_t>(dominant(g))] < 0.0) ? -1.0 : 1.0;
            sign_readout[static_cast<size_t>(i)] = (s < 0.0) ? -1 : 1;

            double diff = 0.0;
            for (int a = 0; a < 3; ++a)
            {
                const double d = g[static_cast<size_t>(a)] * s -
                                 g_first[static_cast<size_t>(a)] * s_first;
                diff += d * d;
            }
            if (std::sqrt(diff) > options.cartesian_tolerance)
            {
                throw std::runtime_error(
                    "auto_label: the readouts do not share a direction, so this sequence has no "
                    "Cartesian encoding counters to derive (readout at block " +
                    std::to_string(ks.readouts[static_cast<size_t>(i)].block_index) + ")");
            }
        }

        /* -- the central readout, and the sampling step it defines -------- */
        const int central = ks.central_readout;
        if (central < 0 || central >= n_adc)
            throw std::runtime_error("auto_label: no readout passes through the k-space centre");
        if (ks.readouts[static_cast<size_t>(central)].center_sample < 0)
            throw std::runtime_error(
                "auto_label: the central readout has no echo, so the k-space centre is not a "
                "sample of it");

        const Readout& central_ro = ks.readouts[static_cast<size_t>(central)];
        if (static_cast<int>(ks.k_central.size()) != 3 * central_ro.num_samples)
            throw std::runtime_error("auto_label: the central readout's samples are missing");

        /* Its samples projected on its own direction: one monotone coordinate
         * standing in for the three, which is what makes a spacing a scalar. */
        std::vector<double> projection(static_cast<size_t>(central_ro.num_samples), 0.0);
        {
            Vec3 dir = g_echo[static_cast<size_t>(central)];
            const double n = norm3(dir);
            if (n <= 0.0)
                throw std::runtime_error("auto_label: the central readout has no gradient");
            for (int a = 0; a < 3; ++a)
                dir[static_cast<size_t>(a)] /= n;

            for (int j = 0; j < central_ro.num_samples; ++j)
            {
                Vec3 k{{0.0, 0.0, 0.0}};
                for (int a = 0; a < 3; ++a)
                    k[static_cast<size_t>(a)] =
                        ks.k_central[static_cast<size_t>(a) *
                                         static_cast<size_t>(central_ro.num_samples) +
                                     static_cast<size_t>(j)];
                projection[static_cast<size_t>(j)] = dot3(orient_encoding(k, options), dir);
            }
        }

        double dk_readout = 0.0;
        {
            std::vector<double> steps;
            steps.reserve(projection.size());
            for (size_t j = 1; j < projection.size(); ++j)
                steps.push_back(projection[j] - projection[j - 1]);
            dk_readout = median_of(steps);
        }
        if (dk_readout == 0.0)
            throw std::runtime_error("auto_label: the central readout does not move through k");

        /* -- ramp sampling ---------------------------------------------- */
        /*
         * A readout sampled on a trapezoid's ramps has a curved k, and a
         * gridder cannot undo that without the ramp geometry.  The test is the
         * second difference of the projection, scaled so it reads as a
         * fraction of the readout: uniform sampling gives zero.
         */
        {
            bool curved = false;
            for (size_t j = 2; j < projection.size(); ++j)
            {
                const double second =
                    projection[j] - 2.0 * projection[j - 1] + projection[j - 2];
                if (std::fabs(second / dk_readout * static_cast<double>(projection.size())) > 0.1)
                {
                    curved = true;
                    break;
                }
            }
            if (curved)
            {
                const Block block = seq.get_block(central_ro.block_index);
                /* The readout axis is the dominant one, and reorder would move
                 * it -- MATLAB warns and gives up there, so do the same. */
                const int axis = dominant(g_echo[static_cast<size_t>(central)]);
                const int32_t grad_ids[3] = {block.gx, block.gy, block.gz};
                const int32_t grad = grad_ids[axis];
                if (options.reorder[static_cast<size_t>(axis)] == axis && grad != 0 &&
                    seq.grad_kind(grad) == GradKind::Trap && block.adc != 0)
                {
                    const double* trap = seq.trap_library().row(seq.grad_row(grad));
                    const double* adc = seq.adc_library().row(block.adc);
                    result.aux.has_gridding = true;
                    result.aux.trapezoid_gridding[0] = trap[1]; /* rise  */
                    result.aux.trapezoid_gridding[1] = trap[2]; /* flat  */
                    result.aux.trapezoid_gridding[2] = trap[3]; /* fall  */
                    result.aux.trapezoid_gridding[3] = adc[2] - trap[4];
                    result.aux.trapezoid_gridding[4] = adc[0] * adc[1];
                    result.aux.target_gridded_samples = static_cast<int>(adc[0]);
                }
            }
        }

        /* -- reduce to one point per readout ----------------------------- */
        /*
         * From here the trajectory is the echo positions and nothing else.
         * The readout axis vanishes with them -- every echo sits at the same
         * place along it -- which is exactly why the surviving axes are the
         * phase-encoded ones and why LIN comes off the first of them.
         */
        const double k_threshold = std::fabs(dk_readout / 50.0);
        const Vec3 k_center = orient_encoding(ks.k_center, options);

        std::array<bool, 3> encoded{{false, false, false}};
        for (int a = 0; a < 3; ++a)
        {
            double extent = 0.0;
            for (int i = first_signal; i < n_adc; ++i)
                extent = std::max(extent, std::fabs(k_echo[static_cast<size_t>(i)]
                                                        [static_cast<size_t>(a)] -
                                                    k_center[static_cast<size_t>(a)]));
            encoded[static_cast<size_t>(a)] = extent >= k_threshold;
        }

        std::vector<int> axes;
        for (int a = 0; a < 3; ++a)
        {
            if (encoded[static_cast<size_t>(a)])
                axes.push_back(a);
        }

        std::vector<int> lin(static_cast<size_t>(n_adc), 0);
        std::vector<int> par(static_cast<size_t>(n_adc), 0);
        std::vector<std::vector<int>> index(axes.size());

        /* The prescription, if the file states one -- see the grid branch. */
        const std::array<double, 3> fov = oriented_definition(seq, "FOV", options);
        const std::array<double, 3> matrix = oriented_definition(seq, "Matrix", options);

        for (size_t ax = 0; ax < axes.size(); ++ax)
        {
            const int a = axes[ax];

            /* The encoding step: the smallest spacing between distinct
             * positions, averaged over every spacing that agrees with it.
             * Averaging rather than taking the minimum outright is what keeps
             * one noisy pair from setting the step for the whole scan. */
            std::vector<double> sorted;
            sorted.reserve(static_cast<size_t>(n_signal));
            for (int i = first_signal; i < n_adc; ++i)
                sorted.push_back(k_echo[static_cast<size_t>(i)][static_cast<size_t>(a)] -
                                 k_center[static_cast<size_t>(a)]);
            std::sort(sorted.begin(), sorted.end());

            double dk_min = 0.0;
            bool have_min = false;
            for (size_t j = 1; j < sorted.size(); ++j)
            {
                const double d = sorted[j] - sorted[j - 1];
                if (d < k_threshold)
                    continue;
                if (!have_min || d < dk_min)
                {
                    dk_min = d;
                    have_min = true;
                }
            }

            double dk = 0.0;
            if (have_min)
            {
                double total = 0.0;
                int count = 0;
                for (size_t j = 1; j < sorted.size(); ++j)
                {
                    const double d = sorted[j] - sorted[j - 1];
                    if (d < k_threshold || d - dk_min > k_threshold)
                        continue;
                    total += d;
                    ++count;
                }
                dk = (count > 0) ? total / count : 0.0;
            }

            index[ax].assign(static_cast<size_t>(n_adc), 0);

            /*
             * The prescribed grid, when `FOV` and `Matrix` are both stated:
             * the step is 1/FOV and the counter is a position on a matrix of
             * that size, so index 0 is the edge of k-space whether or not
             * anything was sampled there.
             *
             * Inferring the step from the sampled positions instead cannot
             * see past the sampling.  The smallest gap in an accelerated
             * scan *is* the accelerated step, and the lowest position sampled
             * becomes index zero, so R = 2 counts 0, 1, 2 where it should
             * count 0, 2, 4, and a partial-Fourier scan starts at zero
             * wherever it really starts.  Both hand a reconstruction a
             * counter that no longer says which line of the prescribed
             * matrix this is.
             */
            const double dk_grid =
                (fov[static_cast<size_t>(a)] > 0.0) ? 1.0 / fov[static_cast<size_t>(a)] : 0.0;
            const int n_grid = static_cast<int>(matrix[static_cast<size_t>(a)]);
            if (dk_grid > 0.0 && n_grid > 0)
            {
                std::vector<int> grid(static_cast<size_t>(n_adc), 0);
                bool fits = true;
                for (int i = first_signal; i < n_adc && fits; ++i)
                {
                    const double steps =
                        k_echo[static_cast<size_t>(i)][static_cast<size_t>(a)] / dk_grid;
                    const double rounded = std::floor(steps + 0.5);
                    const int idx = static_cast<int>(rounded) + n_grid / 2;
                    /* A readout that lands between grid points, or outside
                     * the matrix, means the definitions describe something
                     * other than what is being sampled -- a non-Cartesian
                     * trajectory above all.  Trust the trajectory. */
                    if (std::fabs(steps - rounded) * dk_grid > k_threshold || idx < 0 ||
                        idx >= n_grid)
                        fits = false;
                    grid[static_cast<size_t>(i)] = idx;
                }
                if (fits)
                {
                    index[ax] = grid;
                    continue;
                }
            }

            if (dk == 0.0)
                continue;

            /*
             * Indices are counted from the readout nearest the origin among
             * the surviving axes -- not from `k_center`, which is a sample of
             * the central readout and therefore carries whatever offset the
             * readout axis has.  MATLAB does the same, and it is what makes
             * the centre line an integer rather than a half-step.
             */
            int origin = first_signal;
            double best = 0.0;
            for (int i = first_signal; i < n_adc; ++i)
            {
                double d = 0.0;
                for (size_t b = 0; b < axes.size(); ++b)
                {
                    const double v =
                        k_echo[static_cast<size_t>(i)][static_cast<size_t>(axes[b])];
                    d += v * v;
                }
                if (i == first_signal || d < best)
                {
                    best = d;
                    origin = i;
                }
            }

            const double base = k_echo[static_cast<size_t>(origin)][static_cast<size_t>(a)];
            int lowest = 0;
            for (int i = first_signal; i < n_adc; ++i)
            {
                const double v = k_echo[static_cast<size_t>(i)][static_cast<size_t>(a)] - base;
                const int idx = static_cast<int>(std::floor(v / dk + 0.5));
                index[ax][static_cast<size_t>(i)] = idx;
                if (i == first_signal || idx < lowest)
                    lowest = idx;
            }
            for (int i = first_signal; i < n_adc; ++i)
                index[ax][static_cast<size_t>(i)] -= lowest;
        }

        if (!axes.empty())
        {
            for (int i = 0; i < n_adc; ++i)
                lin[static_cast<size_t>(i)] = index[0][static_cast<size_t>(i)];
        }
        if (axes.size() > 1)
        {
            for (int i = 0; i < n_adc; ++i)
                par[static_cast<size_t>(i)] = index[1][static_cast<size_t>(i)];
        }

        /* -- repetitions -------------------------------------------------- */
        /*
         * How many times this (slice, line, partition, ...) has already been
         * acquired.  A map rather than a dense array over the index bounding
         * box: an undersampled or non-rectangular 3D scan has a bounding box
         * far larger than the number of readouts in it, and the count is a
         * property of the readouts.
         */
        std::vector<int> rep(static_cast<size_t>(n_adc), 0);
        /* The visits themselves, kept because reading the shape of the repeat
         * nest needs to know *when* a position came back, not just how often;
         * see infer_repeat_sizes(). */
        std::vector<std::vector<int>> visits;
        {
            std::map<std::vector<int>, int> seen;
            std::vector<int> key(axes.size() + 1, 0);
            for (int i = first_signal; i < n_adc; ++i)
            {
                key[0] = slice_of_adc[static_cast<size_t>(i)];
                for (size_t ax = 0; ax < axes.size(); ++ax)
                    key[ax + 1] = index[ax][static_cast<size_t>(i)];

                std::map<std::vector<int>, int>::iterator it = seen.find(key);
                if (it == seen.end())
                {
                    it = seen.emplace(key, static_cast<int>(visits.size())).first;
                    visits.push_back(std::vector<int>());
                }
                rep[static_cast<size_t>(i)] = static_cast<int>(visits[static_cast<size_t>(
                                                                   it->second)]
                                                                   .size());
                visits[static_cast<size_t>(it->second)].push_back(i);
            }
        }

        /* -- assemble ---------------------------------------------------- */
        std::vector<int> slc(static_cast<size_t>(n_adc), 0);
        for (int i = first_signal; i < n_adc; ++i)
            slc[static_cast<size_t>(i)] = slice_of_adc[static_cast<size_t>(i)];

        std::vector<int> rev(static_cast<size_t>(n_adc), 0);
        for (int i = first_signal; i < n_adc; ++i)
            rev[static_cast<size_t>(i)] = (sign_readout[static_cast<size_t>(i)] < 0) ? 1 : 0;

        if (any_nonzero(slc) && !skip_slc)
            result.labels.slc = slc;
        if (any_nonzero(rev) && !skip_rev)
            result.labels.rev = rev;
        if (any_nonzero(lin) && !skip_lin)
            result.labels.lin = lin;
        if (any_nonzero(par) && !skip_par)
            result.labels.par = par;

        /* -- splitting the repeat count into the declared dimensions ------- */
        /*
         * A mixed-radix digit extraction, fastest dimension first, which is
         * the order the repeats arrived in.
         *
         * Everything about this is the caller's declaration rather than a
         * measurement, so the only thing worth checking is that the two are
         * consistent: a repeat count the declared sizes cannot represent means
         * either the scan is not what was declared or the declaration missed a
         * dimension.  Wrapping it round with a modulo would produce counters
         * that look perfectly ordinary and put two different acquisitions in
         * the same slot, so it raises instead.
         */
        if (skip_rep)
        {
            /* Nothing derived and nothing named: the sequence said it already
             * and whatever it said is better than a bare repeat count. */
            if (!options.repeat_dims.empty())
                throw std::runtime_error(
                    "auto_label: REP cannot be both skipped and named in repeat_dims -- "
                    "the repeat count is either yours or mine.");
        }
        else if (options.repeat_dims.empty())
        {
            if (any_nonzero(rep))
                result.labels.rep = rep;
        }
        else
        {
            const std::vector<RepeatDim>& dims = options.repeat_dims;

            /*
             * Sizes, outermost first.  Read from the acquisition order unless
             * every one was given; a size that was given is *checked* against
             * what was read rather than trusted, because the two disagreeing
             * means one of them is wrong and picking either in silence is the
             * failure this is here to avoid.
             */
            bool all_given = true;
            for (size_t d = 0; d < dims.size(); ++d)
            {
                if (dims[d].size <= 0)
                    all_given = false;
            }

            std::vector<int> sizes;
            if (all_given)
            {
                for (size_t d = 0; d < dims.size(); ++d)
                    sizes.push_back(dims[d].size);
            }
            else
            {
                sizes = infer_repeat_sizes(visits, dims.size());
                for (size_t d = 0; d < dims.size(); ++d)
                {
                    if (dims[d].size > 0 && dims[d].size != sizes[d])
                        throw std::runtime_error(
                            "auto_label: " + dims[d].name + " was declared with size " +
                            std::to_string(dims[d].size) +
                            ", but the acquisition order says it is " +
                            std::to_string(sizes[d]) + ".");
                }
            }

            long capacity = 1;
            for (size_t d = 0; d < sizes.size(); ++d)
                capacity *= (sizes[d] > 0) ? sizes[d] : 1;

            std::vector<std::vector<int>> split(dims.size(),
                                                std::vector<int>(static_cast<size_t>(n_adc), 0));

            /*
             * Mixed-radix digits, extracted from the innermost dimension --
             * the last of the list, since the caller writes the outermost
             * first -- because that is the one that varies between one repeat
             * and the next.
             */
            for (int i = first_signal; i < n_adc; ++i)
            {
                int remaining = rep[static_cast<size_t>(i)];
                if (remaining >= capacity)
                    throw std::runtime_error(
                        "auto_label: the repeat dimensions hold " + std::to_string(capacity) +
                        " acquisitions of one k-space position, but " +
                        std::to_string(remaining + 1) + " were found (readout at block " +
                        std::to_string(ks.readouts[static_cast<size_t>(i)].block_index) + ")");

                for (size_t d = dims.size(); d-- > 0;)
                {
                    const int size = (sizes[d] > 0) ? sizes[d] : 1;
                    split[d][static_cast<size_t>(i)] = remaining % size;
                    remaining /= size;
                }
            }

            for (size_t d = 0; d < dims.size(); ++d)
            {
                if (any_nonzero(split[d]))
                    result.labels.named.emplace_back(dims[d].name, split[d]);
            }
        }

        /* -- aux ---------------------------------------------------------- */
        /*
         * Indexed by the same readout the counters are, which is the bug noted
         * in the header: MATLAB indexes a front-padded vector with a counter
         * that starts at the first signal readout.
         */
        if (!result.labels.lin.empty())
        {
            result.aux.has_center_line = true;
            result.aux.center_line = lin[static_cast<size_t>(central)];
        }
        if (!result.labels.par.empty())
        {
            result.aux.has_center_partition = true;
            result.aux.center_partition = par[static_cast<size_t>(central)];
        }
        /*
         * The echo index, quoted in the frame a reconstruction reads it in:
         * after REV has been honoured.
         *
         * `Readout::center_sample` is an index into the readout as acquired,
         * and on a bipolar train the two polarities differ by one -- 64 and 63
         * on 128 samples -- because they reach the same point in k from
         * opposite ends.  A single `kSpaceCenterSample` for the whole scan can
         * only be true of one of those frames, and the useful one is the frame
         * every line lands in once the reversed ones have been mirrored, which
         * is where N/2 is the answer for a fully sampled readout.
         *
         * So a reverse central readout is mirrored here.  MATLAB quotes the
         * raw index of whichever readout it happened to find first
         * (`autoLabel.m:392`), which on an EPI beginning with a reverse
         * navigator is 63 -- a number that is correct for that one navigator
         * and wrong for the image lines it is quoted alongside.
         */
        result.aux.has_center_sample = true;
        result.aux.center_sample =
            (rev[static_cast<size_t>(central)] != 0)
                ? (central_ro.num_samples - 1 - central_ro.center_sample)
                : central_ro.center_sample;
        if (unique_slices.size() > 1)
            result.aux.slice_positions = unique_slices;

        /* -- slice thickness and gap -------------------------------------- */
        /*
         * A slice-selective pulse excites a slab of bandwidth over gradient:
         * the frequency offset that moves the slice is the gradient times the
         * distance, so the spread of frequencies the pulse contains is the
         * spread of positions it excites.
         *
         * Taken from the excitation that produced the first acquired readout,
         * which is the same one the slice table is anchored on.  An excitation
         * with no gradient is not slice-selective and has no thickness to
         * report -- and a bandwidth the estimator could not measure is left
         * unreported rather than replaced by an analytic guess.
         */
        {
            int e = 0;
            for (int i = 0; i < n_exc; ++i)
            {
                if (ks.excitations[static_cast<size_t>(i)].t <
                    ks.readouts[static_cast<size_t>(first_signal)].t0)
                    e = i;
            }
            if (n_exc > 0)
            {
                const Vec3 g = orient_slice(ks.excitations[static_cast<size_t>(e)].g, options);
                const double amplitude = norm3(g);
                const double bandwidth =
                    rf_bandwidth(seq, ks.excitations[static_cast<size_t>(e)].block_index);
                if (amplitude > 0.0 && bandwidth > 0.0)
                {
                    result.aux.has_slice_thickness = true;
                    result.aux.slice_thickness = bandwidth / amplitude;

                    /*
                     * The gap is what lies between two slices: their spacing
                     * less one thickness.  Measured on the closest pair, which
                     * is a neighbouring pair because the table is in position
                     * order.  A negative result is a real answer --
                     * overlapping slices -- and is reported as one.
                     */
                    if (ascending_slices.size() > 1)
                    {
                        double closest = ascending_slices[1] - ascending_slices[0];
                        for (size_t s = 2; s < ascending_slices.size(); ++s)
                            closest = std::min(closest,
                                               ascending_slices[s] - ascending_slices[s - 1]);
                        result.aux.has_slice_gap = true;
                        result.aux.slice_gap = closest - result.aux.slice_thickness;
                    }
                }
            }
        }

        return result;
    }

    /* ================================================================== */
    /*  Application                                                       */
    /* ================================================================== */

    void apply_labels(Sequence& seq, const AutoLabels& labels, const AutoLabelAux& aux)
    {
        const std::vector<std::pair<std::string, const std::vector<int>*>> present =
            labels.present();

        if (!present.empty())
        {
            const int labelset = seq.extension_type_id("LABELSET");

            /* The ids of the counters being written, so an earlier run's
             * LABELSET for the same counter can be dropped rather than
             * shadowed by a second one on the same block. */
            std::vector<int> managed;
            managed.reserve(present.size());
            for (size_t f = 0; f < present.size(); ++f)
                managed.push_back(seq.label_id(present[f].first));

            /*
             * One LABELSET row per distinct (value, counter), reused.
             *
             * `register_label_set` appends unconditionally -- the right default
             * for building a scan, where deduplication happens once at the end
             * -- but here the row count is bounded by the number of *values* a
             * counter takes, not by how many readouts take them.  A 256-line
             * scan wants 256 rows, not one per acquisition.  Seeded from what
             * the sequence already holds, so applying twice reuses the first
             * run's rows instead of laying down a second set.
             */
            std::map<std::pair<int32_t, int32_t>, int> label_rows;
            for (int id = 1; id <= seq.label_set_library().size(); ++id)
            {
                const int32_t* row = seq.label_set_library().row(id);
                label_rows.emplace(std::make_pair(row[0], row[1]), id);
            }

            for (size_t i = 0; i < labels.adc_block.size(); ++i)
            {
                const int block_index = labels.adc_block[i];
                Block block = seq.get_block(block_index);

                /* Rebuild the chain without the links this owns, keeping
                 * everything else -- a rotation extension on a readout block
                 * is not ours to discard.  Collected head-first, then relinked
                 * in reverse so the surviving order is unchanged. */
                std::vector<std::pair<int32_t, int32_t>> keep;
                for (int32_t link = block.ext; link != 0;)
                {
                    const int32_t* row = seq.extensions_library().row(link);
                    const int32_t type = row[0];
                    const int32_t ref = row[1];
                    const int32_t next = row[2];
                    bool mine = false;
                    if (type == labelset && ref != 0)
                    {
                        const int32_t label = seq.label_set_library().row(ref)[1];
                        for (size_t m = 0; m < managed.size(); ++m)
                        {
                            if (managed[m] == label)
                            {
                                mine = true;
                                break;
                            }
                        }
                    }
                    if (!mine)
                        keep.emplace_back(type, ref);
                    link = next;
                }

                int32_t head = 0;
                for (size_t k = keep.size(); k-- > 0;)
                    head = static_cast<int32_t>(
                        seq.chain_extension(keep[k].first, keep[k].second, head));

                /* Only where the value changes: that is what SET means, and a
                 * scan that re-stated every counter on every readout would
                 * carry one extension row per acquisition. */
                for (size_t f = present.size(); f-- > 0;)
                {
                    const std::vector<int>& values = *present[f].second;
                    const int value = values[i];
                    const int previous = (i == 0) ? 0 : values[i - 1];
                    if (value == previous)
                        continue;
                    const std::pair<int32_t, int32_t> key(static_cast<int32_t>(value),
                                                          static_cast<int32_t>(managed[f]));
                    std::map<std::pair<int32_t, int32_t>, int>::iterator it =
                        label_rows.find(key);
                    if (it == label_rows.end())
                        it = label_rows
                                 .emplace(key, seq.register_label_set(key.first, key.second))
                                 .first;
                    head = static_cast<int32_t>(
                        seq.chain_extension(labelset, static_cast<int32_t>(it->second), head));
                }

                if (head != block.ext)
                {
                    block.ext = head;
                    seq.set_block(block_index, block);
                }
            }
        }

        if (aux.has_center_line)
            seq.set_definition("kSpaceCenterLine",
                               Definition::integers({static_cast<double>(aux.center_line)}));
        if (aux.has_center_partition)
            seq.set_definition("kSpaceCenterPartition",
                               Definition::integers({static_cast<double>(aux.center_partition)}));
        if (aux.has_center_sample)
            seq.set_definition("kSpaceCenterSample",
                               Definition::integers({static_cast<double>(aux.center_sample)}));
        if (!aux.slice_positions.empty())
            seq.set_definition("SlicePositions", Definition(aux.slice_positions));
        if (aux.has_slice_thickness)
            seq.set_definition("SliceThickness", Definition(aux.slice_thickness));
        if (aux.has_slice_gap)
            seq.set_definition("SliceGap", Definition(aux.slice_gap));
        if (aux.has_gridding)
        {
            seq.set_definition("TrapezoidGriddingParameters",
                               Definition(std::vector<double>(aux.trapezoid_gridding.begin(),
                                                              aux.trapezoid_gridding.end())));
            seq.set_definition(
                "TargetGriddedSamples",
                Definition::integers({static_cast<double>(aux.target_gridded_samples)}));
        }
    }

    /* ================================================================== */

    AutoLabelResult auto_label(Sequence& seq, const AutoLabelOptions& options, bool apply)
    {
        KSpaceOptions ko;
        ko.first_block = options.first_block;
        ko.last_block = options.last_block;
        ko.trajectory_delay = options.trajectory_delay;
        ko.derive_center_sample = true;
        /* One point per readout is the whole input; see the header. */
        ko.materialize_samples = false;

        const KSpace ks = calculate_kspace(seq, ko);
        AutoLabelResult result = detect_labels(ks, seq, options);
        if (apply)
            apply_labels(seq, result.labels, result.aux);
        return result;
    }

}  // namespace pulseq
