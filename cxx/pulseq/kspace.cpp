/**
 * @file kspace.cpp
 * @brief The Sequence-level surface over the trajectory core.  See kspace.hpp.
 *
 * The arithmetic is in ktraj.cpp; this marshals its results into @ref KSpace
 * and adds the two things only a host caller wants -- the dense per-breakpoint
 * trajectory and the slice positions -- plus the MRD layout helpers.
 */

#include "pulseq/kspace.hpp"

#include "ktraj.hpp"
#include "pulseq/sequence.hpp"

#include <cmath>
#include <vector>

namespace pulseq
{
    namespace
    {
        detail::KTrajOptions core_options(const KSpaceOptions& options)
        {
            detail::KTrajOptions out;
            out.trajectory_delay = options.trajectory_delay;
            out.gradient_offset = options.gradient_offset;
            out.first_block = options.first_block;
            out.last_block = options.last_block;
            out.apply_rotation = options.apply_rotation;
            out.sample_window_average = options.sample_window_average;
            out.derive_center_sample = options.derive_center_sample;
            return out;
        }

        void copy_rf(const detail::KTrajRf& src, RfEventTiming& dst)
        {
            dst.block_index = src.block_index;
            dst.t = src.t;
            dst.freq_offset = src.freq_offset;
            dst.phase_offset = src.phase_offset;
            dst.g = src.g;
        }

    }  // namespace

    KSpace calculate_kspace(const Sequence& seq, const KSpaceOptions& options)
    {
        const detail::KTrajPlan plan(seq, core_options(options));
        const std::vector<detail::KTrajReadout>& found = plan.readouts();
        const int num_readouts = static_cast<int>(found.size());

        KSpace out;
        out.readouts.resize(found.size());
        out.key_groups = plan.key_groups();
        out.total_duration = plan.total_duration();
        out.k_center = plan.k_center();

        int total = 0;
        for (int i = 0; i < num_readouts; ++i)
        {
            const detail::KTrajReadout& info = found[static_cast<size_t>(i)];
            Readout& r = out.readouts[static_cast<size_t>(i)];
            r.block_index = info.block_index;
            r.num_samples = info.num_samples;
            r.t0 = info.t0;
            r.dwell = info.dwell;
            r.center_sample = info.center_sample;
            r.rotation_id = info.rotation_id;
            r.g_echo = info.g_echo;
            r.k_echo = info.k_echo;
            for (int a = 0; a < 3; ++a)
                r.constant_axes[static_cast<size_t>(a)] =
                    (info.constant_axes & (1u << a)) != 0u;
            r.sample_offset = total;
            total += (info.num_samples > 0) ? info.num_samples : 0;
        }
        out.total_samples = total;

        /* One rotation shared by every readout is a change of frame; more than
         * one is an encoding.  See KSpace::rotations_vary. */
        for (int i = 1; i < num_readouts; ++i)
        {
            if (out.readouts[static_cast<size_t>(i)].rotation_id != out.readouts[0].rotation_id)
            {
                out.rotations_vary = true;
                break;
            }
        }

        if (options.materialize_samples)
        {
            out.k_adc.resize(static_cast<size_t>(total) * 3);
            out.t_adc.resize(static_cast<size_t>(total));
            for (int i = 0; i < num_readouts; ++i)
            {
                const Readout& r = out.readouts[static_cast<size_t>(i)];
                if (r.num_samples <= 0)
                    continue;
                plan.readout_k(i, out.k_adc.data() + static_cast<size_t>(r.sample_offset) * 3);
                for (int j = 0; j < r.num_samples; ++j)
                    out.t_adc[static_cast<size_t>(r.sample_offset + j)] =
                        r.t0 + static_cast<double>(j) * r.dwell;
            }
        }

        if (options.block_origins)
        {
            const int first = (options.first_block > 0) ? options.first_block : 1;
            const int last = (options.last_block > 0) ? options.last_block : seq.num_blocks();
            out.block_origins.resize(
                static_cast<size_t>((last >= first) ? (last - first + 1) : 0));
            for (int b = first; b <= last; ++b)
                out.block_origins[static_cast<size_t>(b - first)] = plan.block_origin(b);
        }

        /* The reference profile, always -- one readout, and the only one an
         * analysis of the sampling pattern has to look inside. */
        out.central_readout = plan.central_readout();
        if (out.central_readout >= 0 && out.central_readout < num_readouts)
        {
            const Readout& r = out.readouts[static_cast<size_t>(out.central_readout)];
            if (r.num_samples > 0)
            {
                out.k_central.resize(static_cast<size_t>(r.num_samples) * 3);
                plan.readout_k(out.central_readout, out.k_central.data());
            }
        }

        out.excitations.resize(plan.excitations().size());
        for (size_t i = 0; i < plan.excitations().size(); ++i)
            copy_rf(plan.excitations()[i], out.excitations[i]);

        out.refocusings.resize(plan.refocusings().size());
        for (size_t i = 0; i < plan.refocusings().size(); ++i)
            copy_rf(plan.refocusings()[i], out.refocusings[i]);

        /* Slice positions.
         *
         * offset / gradient is a length: the frequency offset that selects a
         * slice is the gradient times the distance to it.  An axis with no
         * gradient gives no information about position, and Pulseq's own
         * calculateKspacePP reports 0 there rather than a NaN, so this does
         * too. */
        out.slice_pos.resize(out.excitations.size());
        out.t_slice_pos.resize(out.excitations.size());
        for (size_t i = 0; i < out.excitations.size(); ++i)
        {
            const RfEventTiming& rf = out.excitations[i];
            out.t_slice_pos[i] = rf.t;
            for (int a = 0; a < 3; ++a)
            {
                const double g = rf.g[static_cast<size_t>(a)];
                out.slice_pos[i][static_cast<size_t>(a)] = (g != 0.0) ? rf.freq_offset / g : 0.0;
            }
        }

        if (options.dense)
        {
            /*
             * The dense grid is for drawing, and the echo search is the
             * expensive half of a plan -- there is nothing here that wants it.
             *
             * Deliberately not on the gradient raster: between two breakpoints
             * k is one quadratic, so the breakpoints are the smallest grid
             * that loses nothing structural.  A five-millisecond flat top
             * costs two points rather than five hundred.
             */
            detail::KTrajOptions dense_options = core_options(options);
            dense_options.derive_center_sample = false;
            const detail::KTrajPlan dense(seq, dense_options);

            out.t_ktraj = dense.breakpoints();
            out.k_ktraj.resize(out.t_ktraj.size() * 3);
            dense.at_times(out.t_ktraj.data(), static_cast<int>(out.t_ktraj.size()),
                           out.k_ktraj.data());
        }

        return out;
    }

    /* ================================================================== */
    /*  MRD                                                               */
    /* ================================================================== */

    std::array<bool, 3> active_axes(const KSpace& ks, double tolerance)
    {
        std::array<bool, 3> active{{false, false, false}};

        for (const Readout& r : ks.readouts)
        {
            if (r.num_samples <= 0)
                continue;
            for (int a = 0; a < 3; ++a)
            {
                if (active[static_cast<size_t>(a)])
                    continue;
                const double* k = ks.k_adc.data() + static_cast<size_t>(r.sample_offset) * 3 +
                                  static_cast<size_t>(a) * static_cast<size_t>(r.num_samples);
                for (int j = 0; j < r.num_samples; ++j)
                {
                    if (std::fabs(k[j]) > tolerance)
                    {
                        active[static_cast<size_t>(a)] = true;
                        break;
                    }
                }
            }
        }
        return active;
    }

    int interleave_readout(const KSpace& ks, const std::array<bool, 3>& axes, int readout,
                           float* out)
    {
        if (readout < 0 || readout >= static_cast<int>(ks.readouts.size()) || out == nullptr)
            return 0;

        const Readout& r = ks.readouts[static_cast<size_t>(readout)];
        if (r.num_samples <= 0)
            return 0;

        int ndim = 0;
        for (int a = 0; a < 3; ++a)
            ndim += axes[static_cast<size_t>(a)] ? 1 : 0;
        if (ndim == 0)
            return 0;

        const double* base = ks.k_adc.data() + static_cast<size_t>(r.sample_offset) * 3;
        int written = 0;
        for (int j = 0; j < r.num_samples; ++j)
        {
            for (int a = 0; a < 3; ++a)
            {
                if (!axes[static_cast<size_t>(a)])
                    continue;
                out[written++] = static_cast<float>(
                    base[static_cast<size_t>(a) * static_cast<size_t>(r.num_samples) +
                         static_cast<size_t>(j)]);
            }
        }
        return written;
    }

}  // namespace pulseq
