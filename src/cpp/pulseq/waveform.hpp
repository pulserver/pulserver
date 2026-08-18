/**
 * @file waveform.hpp
 * @brief A gradient as breakpoints, which is the form every integral wants.
 *
 * Library-private, and deliberately so: this is the one header in
 * `src/cpp/pulseq/` that is not installed.  `Piecewise` is an implementation
 * detail two translation units happen to share -- `trajectory.cpp` and
 * `moments.cpp` -- and promoting it to `src/cpp/include/pulseq/` would make a
 * scratch type part of the public surface for no caller's benefit.
 *
 * Both gradient kinds reduce to the same thing: a piecewise-linear
 * **normalised** waveform on a grid of breakpoints.  A trapezoid is four
 * breakpoints written down; an arbitrary gradient is its samples with
 * `first`/`last` closing the ends.  Once in that form every integral over it
 * is exact segment by segment, so nothing is resampled onto a raster it was
 * not already on -- which is the whole reason the b-tensor here does not
 * depend on a step size.
 *
 * Normalised is the important word.  The amplitude stays outside, exactly as
 * the file stores it (see events.hpp), so one decompressed waveform serves
 * every instance of that gradient and scaling an instance is one multiply.
 */

#ifndef PULSEQ_CXX_WAVEFORM_HPP
#define PULSEQ_CXX_WAVEFORM_HPP

#include "pulseq/sequence.hpp"
#include "pulseq/shape.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace pulseq
{
    namespace detail
    {

        /**
         * A normalised waveform as breakpoints: value `v[i]` at time `t[i]`,
         * linear in between, zero outside.  `c[i]` is the integral from `t[0]`
         * to `t[i]`, so evaluating the cumulative anywhere is a lookup plus the
         * area of one partial trapezoid.
         */
        struct Piecewise
        {
            std::vector<double> t;
            std::vector<double> v;
            std::vector<double> c;

            bool empty() const
            {
                return t.size() < 2;
            }

            void integrate()
            {
                c.assign(t.size(), 0.0);
                for (size_t i = 1; i < t.size(); ++i)
                    c[i] = c[i - 1] + 0.5 * (v[i] + v[i - 1]) * (t[i] - t[i - 1]);
            }

            double total() const
            {
                return c.empty() ? 0.0 : c.back();
            }

            /** The waveform at @p x, holding the end values outside. */
            double value_at(double x) const
            {
                if (empty())
                    return 0.0;
                if (x <= t.front())
                    return v.front();
                if (x >= t.back())
                    return v.back();
                const size_t hi =
                    static_cast<size_t>(std::upper_bound(t.begin(), t.end(), x) - t.begin());
                const size_t lo = hi - 1;
                const double span = t[hi] - t[lo];
                if (span <= 0.0)
                    return v[hi];
                return v[lo] + (v[hi] - v[lo]) * (x - t[lo]) / span;
            }

            /** The integral from `t[0]` to @p x. */
            double cumulative_at(double x) const
            {
                if (empty())
                    return 0.0;
                if (x <= t.front())
                    return 0.0;
                if (x >= t.back())
                    return c.back();
                const size_t hi =
                    static_cast<size_t>(std::upper_bound(t.begin(), t.end(), x) - t.begin());
                const size_t lo = hi - 1;
                const double span = t[hi] - t[lo];
                if (span <= 0.0)
                    return c[hi];
                const double frac = (x - t[lo]) / span;
                const double v_x = v[lo] + (v[hi] - v[lo]) * frac;
                return c[lo] + 0.5 * (v[lo] + v_x) * (x - t[lo]);
            }
        };

        /* ============================================================== */
        /*  Hashing, and collapsing a library onto its distinct rows       */
        /* ============================================================== */

        constexpr size_t HASH_SEED = 2166136261u;
        constexpr uint64_t HASH_PRIME = 1099511628211ull;

        /** One 64-bit word mixed in: splitmix64's finalizer, then FNV. */
        inline void hash_word(size_t& h, uint64_t x)
        {
            x ^= x >> 33;
            x *= 0xff51afd7ed558ccdull;
            x ^= x >> 33;
            h = static_cast<size_t>((static_cast<uint64_t>(h) ^ x) * HASH_PRIME);
        }

        /**
         * A double mixed in, with -0.0 folded onto +0.0.
         *
         * Hashed as one word rather than byte by byte: these hashes run over
         * every row of a library, which on an undeduplicated protocol-scale
         * scan is millions of rows, and eight dependent steps per double is
         * the difference between the pass being noticeable and not.
         */
        inline void hash_double(size_t& h, double v)
        {
            if (v == 0.0)
                v = 0.0;
            uint64_t bits = 0;
            std::memcpy(&bits, &v, sizeof(bits));
            hash_word(h, bits);
        }

        inline void hash_int(size_t& h, long long v)
        {
            hash_word(h, static_cast<uint64_t>(v));
        }

        /**
         * Map each of @p count 1-based ids onto the first id that means the
         * same thing, as @p hash and @p equal define it.  Entry 0 is unused.
         *
         * The libraries are not deduplicated until a sequence is written, so a
         * scan that plays one readout a million times holds a million
         * identical rows under a million ids.  Everything that wants to do
         * work once per *distinct* event -- integrating a gradient, planning a
         * shift, keying a memo -- needs this first, and keying on the id alone
         * would find nothing shared where there is nothing but sharing.
         *
         * Open addressing over one flat array, sized to twice the count and
         * rounded up to a power of two so the table never exceeds half load
         * and probes stay short.  Node-based hashing costs an allocation per
         * row, which at these sizes is more than the work being saved.
         */
        template <typename Hash, typename Equal>
        inline std::vector<int> canonical_ids(int count, Hash hash, Equal equal)
        {
            std::vector<int> canon(static_cast<size_t>(count) + 1, 0);
            if (count <= 0)
                return canon;

            int capacity = 16;
            while (capacity < count * 2)
            {
                if (capacity > (1 << 29))
                    break;
                capacity <<= 1;
            }
            const size_t mask = static_cast<size_t>(capacity) - 1;
            std::vector<int> table(static_cast<size_t>(capacity), -1);

            for (int id = 1; id <= count; ++id)
            {
                size_t slot = hash(id) & mask;
                canon[static_cast<size_t>(id)] = id;
                for (;;)
                {
                    const int occupant = table[slot];
                    if (occupant < 0)
                    {
                        table[slot] = id;
                        break;
                    }
                    if (equal(id, occupant))
                    {
                        canon[static_cast<size_t>(id)] = canon[static_cast<size_t>(occupant)];
                        break;
                    }
                    slot = (slot + 1) & mask;
                }
            }
            return canon;
        }

        /**
         * Flat across [t0, t1]?
         *
         * Checked at the window edges and at every breakpoint strictly inside
         * it, which is exact for a piecewise-linear waveform: a segment can
         * only depart from a constant at its own ends.  The tolerance is
         * relative to the waveform's own scale, so a normalised ramp of 1e-12
         * does not read as structure.
         */
        inline bool is_flat(const Piecewise& p, double t0, double t1)
        {
            if (p.empty())
                return true;

            double lo = p.value_at(t0);
            double hi = lo;
            const double edge = p.value_at(t1);
            lo = std::min(lo, edge);
            hi = std::max(hi, edge);

            double scale = 0.0;
            for (size_t i = 0; i < p.t.size(); ++i)
            {
                if (p.t[i] > t0 && p.t[i] < t1)
                {
                    lo = std::min(lo, p.v[i]);
                    hi = std::max(hi, p.v[i]);
                }
                scale = std::max(scale, std::fabs(p.v[i]));
            }

            return (hi - lo) <= 1e-9 * std::max(scale, 1.0);
        }

        /* A trapezoid at unit amplitude: flat at 1 between its ramps. */
        inline Piecewise unit_trapezoid(const double* row)
        {
            const double rise = row[1];
            const double flat = row[2];
            const double fall = row[3];
            const double delay = row[4];

            Piecewise p;
            p.t.reserve(4);
            p.v.reserve(4);
            double at = delay;
            p.t.push_back(at);
            p.v.push_back(0.0);
            at += rise;
            p.t.push_back(at);
            p.v.push_back(1.0);
            if (flat > 0.0)
            {
                at += flat;
                p.t.push_back(at);
                p.v.push_back(1.0);
            }
            at += fall;
            p.t.push_back(at);
            p.v.push_back(0.0);

            p.integrate();
            return p;
        }

        /**
         * An arbitrary gradient decoded into samples and their times.
         *
         * The shape library holds the waveform already normalised, so the
         * samples are used as they stand; `first` and `last` are absolute
         * (Hz/m) and are divided back down by the amplitude to join them.
         * With `time_shape_id == 0` the samples sit at raster centres, which
         * is the convention the file's own header states.
         */
        struct ArbSamples
        {
            const double* w = nullptr;
            std::vector<double> decoded; /**< owns `w` when it was compressed */
            std::vector<double> tt;      /**< block-relative sample times, s  */
            double first = 0.0;          /**< normalised                      */
            double last = 0.0;
            double delay = 0.0;
            double raster = 0.0;
            double shape_end = 0.0;
            bool uniform = false;
            bool valid = false;
        };

        inline ArbSamples arb_samples(const Sequence& seq, const double* row)
        {
            ArbSamples out;

            const double amplitude = row[0];
            const int shape_id = static_cast<int>(row[3]);
            const int time_shape_id = static_cast<int>(row[4]);
            out.delay = row[5];
            out.raster = seq.grad_raster_time();
            out.uniform = time_shape_id <= 0;

            if (shape_id <= 0 || shape_id > seq.shape_library().size())
                return out;

            /* A sequence read from a file holds its shapes encoded, and one
             * built in memory may not have been through compress_shapes() yet.
             * Both are normal, so decode on demand rather than demanding a
             * particular state. */
            const ShapeLibrary& shapes = seq.shape_library();
            const int n = shapes.num_uncompressed(shape_id);
            if (n <= 0)
                return out;

            out.w = shapes.samples(shape_id);
            if (shapes.is_compressed(shape_id))
            {
                out.decoded = decompress_shape(out.w, shapes.num_compressed(shape_id), n);
                if (static_cast<int>(out.decoded.size()) != n)
                    return out;
                out.w = out.decoded.data();
            }

            out.tt.reserve(static_cast<size_t>(n));
            if (time_shape_id > 0 && time_shape_id <= shapes.size())
            {
                const int nt = shapes.num_uncompressed(time_shape_id);
                const double* ts = shapes.samples(time_shape_id);
                for (int i = 0; i < std::min(n, nt); ++i)
                    out.tt.push_back(out.delay + ts[i] * out.raster);
            }
            else
            {
                for (int i = 0; i < n; ++i)
                    out.tt.push_back(out.delay + (static_cast<double>(i) + 0.5) * out.raster);
            }
            if (out.tt.empty())
                return out;

            const double scale = (amplitude != 0.0) ? 1.0 / amplitude : 0.0;
            out.first = row[1] * scale;
            out.last = row[2] * scale;
            out.shape_end =
                out.uniform ? out.delay + static_cast<double>(n) * out.raster : out.tt.back();
            out.valid = true;
            return out;
        }

        /** An arbitrary gradient at unit amplitude, as the file stores it. */
        inline Piecewise unit_arbitrary(const Sequence& seq, const double* row)
        {
            Piecewise p;
            const ArbSamples s = arb_samples(seq, row);
            if (!s.valid)
                return p;

            p.t.reserve(s.tt.size() + 2);
            p.v.reserve(s.tt.size() + 2);
            p.t.push_back(s.delay);
            p.v.push_back(s.first);
            for (size_t i = 0; i < s.tt.size(); ++i)
            {
                p.t.push_back(s.tt[i]);
                p.v.push_back(s.w[i]);
            }
            if (s.shape_end > p.t.back())
            {
                p.t.push_back(s.shape_end);
                p.v.push_back(s.last);
            }

            p.integrate();
            return p;
        }

        /* ============================================================== */
        /*  The analysis form                                             */
        /* ============================================================== */
        /*
         * Everything below answers a different question from the one above.
         *
         * `unit_arbitrary` reports the gradient the way the file stores it,
         * which is what the trajectory wants: k at an ADC sample has to be the
         * number the scanner and the reconstruction also arrive at, and they
         * arrive at it from the stored samples.
         *
         * An analysis of what the *body* experienced wants the waveform that
         * was really played, and those differ.  A Pulseq arbitrary gradient is
         * sampled at the raster interval *centres*, so a trapezoid converted
         * to a shape comes back with its corners rounded over one raster
         * interval.  The edge samples are recoverable -- a centre is the mean
         * of its two edges -- and MATLAB recovers them in
         * `restoreAdditionalShapeSamples.m`.  PyPulseq carries a `TODO:
         * Implement restoreAdditionalShapeSamples` where that would go; this
         * project ported it, and `Sequence.waveforms()` already uses it.  The
         * b-tensor uses it too, so the two agree to the bit.
         */

        /** MATLAB's threshold on the reconstruction, with MATLAB's own
         *  `% what's the reasonable threshold?` -- not ours to tune. */
        constexpr double RESTORE_TOLERANCE = 2e-5;
        /** MATLAB's `1e-8` on the second difference of the doubled waveform. */
        constexpr double COLLINEAR_TOLERANCE = 1e-8;

        /**
         * A centres-raster gradient put back on the raster edges.
         *
         * Falls back to the stored samples bracketed by `first` and `last` --
         * which is what PyPulseq returns always -- when the reconstruction
         * drifts away from the recorded `last`, the sign that the shape was
         * never piecewise linear on the raster.  A spiral takes that branch.
         */
        inline Piecewise restore_edges(const ArbSamples& s)
        {
            const size_t n = s.tt.size();
            Piecewise p;

            double peak = 0.0;
            for (size_t i = 0; i < n; ++i)
                peak = std::max(peak, std::fabs(s.w[i]));

            /* e[0] = first, e[k] = 2 w[k-1] - e[k-1]. */
            std::vector<double> restored(n + 1);
            restored[0] = s.first;
            for (size_t k = 1; k <= n; ++k)
                restored[k] = 2.0 * s.w[k - 1] - restored[k - 1];

            if (std::fabs(restored[n] - s.last) > RESTORE_TOLERANCE * peak)
            {
                p.t.push_back(s.delay);
                p.v.push_back(s.first);
                for (size_t i = 0; i < n; ++i)
                {
                    p.t.push_back(s.tt[i]);
                    p.v.push_back(s.w[i]);
                }
                p.t.push_back(s.tt[n - 1] + 0.5 * s.raster);
                p.v.push_back(s.last);
                p.integrate();
                return p;
            }

            /* Where the reconstruction agrees with plain interpolation, prefer
             * the interpolation: the alternating sum accumulates error along
             * the shape and the interpolation does not. */
            const double slack = std::numeric_limits<double>::epsilon() + RESTORE_TOLERANCE * peak;
            std::vector<double> edges(n + 1);
            for (size_t k = 0; k <= n; ++k)
            {
                const double interpolated = (k == 0) ? s.first
                    : (k == n)                       ? s.last
                                                     : 0.5 * (s.w[k - 1] + s.w[k]);
                edges[k] =
                    (std::fabs(restored[k] - interpolated) <= slack) ? interpolated : restored[k];
            }

            /* Edges and centres interleaved onto the half raster. */
            std::vector<double> dense(2 * n + 1);
            for (size_t k = 0; k <= n; ++k)
                dense[2 * k] = edges[k];
            for (size_t i = 0; i < n; ++i)
                dense[2 * i + 1] = s.w[i];

            /* Keep only what bends: on a straight segment the second
             * difference vanishes and the interior samples say nothing. */
            for (size_t i = 0; i < dense.size(); ++i)
            {
                const bool endpoint = (i == 0 || i + 1 == dense.size());
                const double curvature =
                    endpoint ? 1.0 : dense[i + 1] - 2.0 * dense[i] + dense[i - 1];
                if (std::fabs(curvature) > COLLINEAR_TOLERANCE)
                {
                    p.t.push_back(s.delay + static_cast<double>(i) * 0.5 * s.raster);
                    p.v.push_back(dense[i]);
                }
            }
            p.integrate();
            return p;
        }

        /**
         * The waveform an analysis should integrate, at unit amplitude.
         *
         * Mirrors `_gradient_piece` in `src/python/pulserver/pypulseq/_pulseqpp.py`
         * case for case, which is what `Sequence.waveforms()` hands upstream.
         */
        inline Piecewise unit_arbitrary_analysis(const Sequence& seq, const double* row)
        {
            const ArbSamples s = arb_samples(seq, row);
            if (!s.valid)
                return Piecewise{};

            const size_t n = s.tt.size();
            const double half = 0.5 * s.raster;

            bool on_centres = true;
            for (size_t i = 0; i < n && on_centres; ++i)
            {
                const double expected = s.delay + (static_cast<double>(i) + 0.5) * s.raster;
                on_centres = std::fabs(s.tt[i] - expected) < 1e-12 * std::max(1.0, s.raster);
            }
            if (on_centres)
                return restore_edges(s);

            Piecewise p;
            if (std::fabs(s.tt[0] - (s.delay + half)) < 1e-12 * std::max(1.0, s.raster))
            {
                /* Oversampled but rasterised: the first sample sits on the
                 * half raster, so the stored shape is missing its endpoints.
                 * MATLAB puts them back; upstream's else-branch does not. */
                p.t.push_back(s.delay);
                p.v.push_back(s.first);
                for (size_t i = 0; i < n; ++i)
                {
                    p.t.push_back(s.tt[i]);
                    p.v.push_back(s.w[i]);
                }
                p.t.push_back(s.shape_end);
                p.v.push_back(s.last);
                p.integrate();
                return p;
            }

            /* An extended trapezoid, already on the raster edges. */
            for (size_t i = 0; i < n; ++i)
            {
                p.t.push_back(s.tt[i]);
                p.v.push_back(s.w[i]);
            }
            p.integrate();
            return p;
        }

        /** The unit waveform behind a gradient id, empty when there is none. */
        inline Piecewise unit_waveform(const Sequence& seq, int32_t grad_id, double* out_amplitude)
        {
            if (out_amplitude)
                *out_amplitude = 0.0;
            if (grad_id <= 0)
                return Piecewise{};

            const int row = seq.grad_row(static_cast<int>(grad_id));
            switch (seq.grad_kind(static_cast<int>(grad_id)))
            {
            case GradKind::Trap:
            {
                const double* d = seq.trap_library().row(row);
                if (out_amplitude)
                    *out_amplitude = d[0];
                return unit_trapezoid(d);
            }
            case GradKind::Arbitrary:
            {
                const double* d = seq.arb_library().row(row);
                if (out_amplitude)
                    *out_amplitude = d[0];
                return unit_arbitrary(seq, d);
            }
            default:
                return Piecewise{};
            }
        }

        /** @ref unit_waveform in the analysis form; see above for which. */
        inline Piecewise unit_waveform_analysis(
            const Sequence& seq,
            int32_t grad_id,
            double* out_amplitude)
        {
            if (out_amplitude)
                *out_amplitude = 0.0;
            if (grad_id <= 0)
                return Piecewise{};

            const int row = seq.grad_row(static_cast<int>(grad_id));
            switch (seq.grad_kind(static_cast<int>(grad_id)))
            {
            case GradKind::Trap:
            {
                const double* d = seq.trap_library().row(row);
                if (out_amplitude)
                    *out_amplitude = d[0];
                return unit_trapezoid(d);
            }
            case GradKind::Arbitrary:
            {
                const double* d = seq.arb_library().row(row);
                if (out_amplitude)
                    *out_amplitude = d[0];
                return unit_arbitrary_analysis(seq, d);
            }
            default:
                return Piecewise{};
            }
        }

    } // namespace detail
} // namespace pulseq

#endif /* PULSEQ_CXX_WAVEFORM_HPP */
