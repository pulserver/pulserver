/**
 * @file trajectory.cpp
 * @brief Base k-space for a block's ADC window.  See trajectory.hpp.
 *
 * Both gradient kinds reduce to the same thing: a piecewise-linear normalised
 * waveform on a grid of breakpoints.  A trapezoid is four breakpoints written
 * down; an arbitrary gradient is its samples with `first`/`last` closing the
 * ends.  Once in that form the cumulative integral is exact segment by segment
 * and evaluating it at an ADC sample centre is one interpolation, so nothing
 * is resampled onto a raster it was not already on.
 */

#include "pulseq/trajectory.hpp"

#include "pulseq/shape.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pulseq
{

    namespace
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

            bool empty() const { return t.size() < 2; }

            void integrate()
            {
                c.assign(t.size(), 0.0);
                for (size_t i = 1; i < t.size(); ++i)
                    c[i] = c[i - 1] + 0.5 * (v[i] + v[i - 1]) * (t[i] - t[i - 1]);
            }

            double total() const { return c.empty() ? 0.0 : c.back(); }

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

        /* A trapezoid at unit amplitude: flat at 1 between its ramps. */
        Piecewise unit_trapezoid(const double* row)
        {
            const double rise = row[1];
            const double flat = row[2];
            const double fall = row[3];
            const double delay = row[4];

            Piecewise p;
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

        /*
         * An arbitrary gradient at unit amplitude.
         *
         * The shape library holds the waveform already normalised, so the
         * samples are used as they stand.  `first` and `last` are absolute
         * (Hz/m) and close the ends at the block's own boundaries; they are
         * divided back down by the amplitude to stay in the same units as the
         * samples.  With `time_shape_id == 0` the samples sit at raster
         * centres, which is the convention the file's own header states.
         */
        Piecewise unit_arbitrary(const Sequence& seq, const double* row)
        {
            const double amplitude = row[0];
            const double first = row[1];
            const double last = row[2];
            const int shape_id = static_cast<int>(row[3]);
            const int time_shape_id = static_cast<int>(row[4]);
            const double delay = row[5];
            const double raster = seq.grad_raster_time();

            Piecewise p;
            if (shape_id <= 0 || shape_id > seq.shape_library().size())
                return p;

            /* A sequence read from a file holds its shapes encoded, and one
             * built in memory may not have been through compress_shapes() yet.
             * Both are normal, so decode on demand rather than demanding a
             * particular state. */
            const ShapeLibrary& shapes = seq.shape_library();
            const int n = shapes.num_uncompressed(shape_id);
            if (n <= 0)
                return p;

            std::vector<double> decoded;
            const double* w = shapes.samples(shape_id);
            if (shapes.is_compressed(shape_id))
            {
                decoded = decompress_shape(w, shapes.num_compressed(shape_id), n);
                if (static_cast<int>(decoded.size()) != n)
                    return p;
                w = decoded.data();
            }

            std::vector<double> tt;
            tt.reserve(static_cast<size_t>(n));
            if (time_shape_id > 0 && time_shape_id <= shapes.size())
            {
                const int nt = shapes.num_uncompressed(time_shape_id);
                const double* ts = shapes.samples(time_shape_id);
                for (int i = 0; i < std::min(n, nt); ++i)
                    tt.push_back(delay + ts[i] * raster);
            }
            else
            {
                for (int i = 0; i < n; ++i)
                    tt.push_back(delay + (static_cast<double>(i) + 0.5) * raster);
            }
            if (tt.empty())
                return p;

            const double scale = (amplitude != 0.0) ? 1.0 / amplitude : 0.0;

            p.t.push_back(delay);
            p.v.push_back(first * scale);
            for (size_t i = 0; i < tt.size(); ++i)
            {
                p.t.push_back(tt[i]);
                p.v.push_back(w[i]);
            }
            const double end = (time_shape_id > 0) ? tt.back() : delay + static_cast<double>(n) * raster;
            if (end > p.t.back())
            {
                p.t.push_back(end);
                p.v.push_back(last * scale);
            }

            p.integrate();
            return p;
        }

        /** The unit waveform behind a gradient id, empty when there is none. */
        Piecewise unit_waveform(const Sequence& seq, int32_t grad_id, double* out_amplitude)
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

        /*
         * Flat across [t0, t1]?
         *
         * Checked at the window edges and at every breakpoint inside it, which
         * is exact for a piecewise-linear waveform: a segment can only depart
         * from a constant at its own ends.  The tolerance is relative to the
         * waveform's own scale so a normalised ramp of 1e-12 does not read as
         * structure.
         */
        bool is_flat(const Piecewise& p, double t0, double t1)
        {
            if (p.empty())
                return true;

            double lo = p.value_at(t0);
            double hi = lo;
            const double edge = p.value_at(t1);
            lo = std::min(lo, edge);
            hi = std::max(hi, edge);

            for (size_t i = 0; i < p.t.size(); ++i)
            {
                if (p.t[i] <= t0 || p.t[i] >= t1)
                    continue;
                lo = std::min(lo, p.v[i]);
                hi = std::max(hi, p.v[i]);
            }

            double scale = 0.0;
            for (size_t i = 0; i < p.v.size(); ++i)
                scale = std::max(scale, std::fabs(p.v[i]));
            const double tol = 1e-9 * std::max(scale, 1.0);
            return (hi - lo) <= tol;
        }

    }  // namespace

    double gradient_moment(const Sequence& seq, int32_t grad_id)
    {
        const Piecewise p = unit_waveform(seq, grad_id, nullptr);
        return p.total();
    }

    BaseTrajectory base_trajectory(const Sequence& seq, int block_index)
    {
        if (block_index < 1 || block_index > seq.num_blocks())
            throw std::out_of_range("base_trajectory: block index out of range");

        const Block block = seq.get_block(block_index);

        BaseTrajectory out;
        const int32_t grads[3] = {block.gx, block.gy, block.gz};

        Piecewise wave[3];
        for (int a = 0; a < 3; ++a)
        {
            double amplitude = 0.0;
            wave[a] = unit_waveform(seq, grads[a], &amplitude);
            out.axis[a].amplitude = amplitude;
            out.axis[a].present = grads[a] > 0;
            out.axis[a].moment = wave[a].total();
        }

        if (block.adc <= 0)
            return out;

        const double* adc = seq.adc_library().row(static_cast<int>(block.adc));
        const int num_samples = static_cast<int>(adc[0]);
        const double dwell = adc[1];
        const double delay = adc[2];
        if (num_samples <= 0 || dwell <= 0.0)
            return out;

        out.has_adc = true;
        out.num_samples = num_samples;
        out.dwell = dwell;
        out.delay = delay;

        /* Sample centres, matching how the readout is actually sampled. */
        const double t_first = delay + 0.5 * dwell;
        const double t_last = delay + (static_cast<double>(num_samples) - 0.5) * dwell;

        for (int a = 0; a < 3; ++a)
        {
            AxisTrajectory& ax = out.axis[a];
            ax.constant = is_flat(wave[a], t_first, t_last);
            if (!ax.present || wave[a].empty())
                continue;

            ax.base.resize(static_cast<size_t>(num_samples));
            for (int i = 0; i < num_samples; ++i)
            {
                const double t = delay + (static_cast<double>(i) + 0.5) * dwell;
                ax.base[static_cast<size_t>(i)] = wave[a].cumulative_at(t);
            }
        }

        return out;
    }

    /* ================================================================== */
    /*  Absolute k                                                        */
    /* ================================================================== */

    std::vector<std::array<double, 3>> block_k_origins(const Sequence& seq)
    {
        const int blocks = seq.num_blocks();
        std::vector<std::array<double, 3>> out(static_cast<size_t>(blocks) + 1,
                                               std::array<double, 3>{0.0, 0.0, 0.0});

        std::array<double, 3> k{0.0, 0.0, 0.0};
        for (int b = 1; b <= blocks; ++b)
        {
            out[static_cast<size_t>(b)] = k;

            const Block block = seq.get_block(b);
            const int32_t grads[3] = {block.gx, block.gy, block.gz};

            Piecewise wave[3];
            double amplitude[3] = {0.0, 0.0, 0.0};
            for (int a = 0; a < 3; ++a)
                wave[a] = unit_waveform(seq, grads[a], &amplitude[a]);

            char use = 0;
            double t_center = 0.0;
            if (block.rf > 0 && block.rf <= seq.rf_library().size())
            {
                const double* rf = seq.rf_library().row(static_cast<int>(block.rf));
                t_center = rf[5] + rf[4];  // delay + center
                const std::vector<char>& uses = seq.rf_uses();
                const size_t idx = static_cast<size_t>(block.rf) - 1;
                if (idx < uses.size())
                    use = uses[idx];
            }

            if (use == 'e' || use == 'r')
            {
                /* Split at the pulse centre so crushers on either side land on
                 * the correct side of the reset or the sign flip. */
                for (int a = 0; a < 3; ++a)
                {
                    const double before = wave[a].cumulative_at(t_center);
                    double mid = k[a] + amplitude[a] * before;
                    mid = (use == 'e') ? 0.0 : -mid;
                    k[a] = mid + amplitude[a] * (wave[a].total() - before);
                }
            }
            else
            {
                for (int a = 0; a < 3; ++a)
                    k[a] += amplitude[a] * wave[a].total();
            }
        }

        return out;
    }

    std::array<std::vector<double>, 3> absolute_trajectory(
        const Sequence& seq, int block_index, const std::array<double, 3>& origin)
    {
        std::array<std::vector<double>, 3> out;

        const BaseTrajectory t = base_trajectory(seq, block_index);
        if (!t.has_adc)
            return out;

        for (int a = 0; a < 3; ++a)
        {
            const AxisTrajectory& ax = t.axis[a];
            if (ax.base.empty())
                continue;
            out[static_cast<size_t>(a)].resize(ax.base.size());
            for (size_t i = 0; i < ax.base.size(); ++i)
                out[static_cast<size_t>(a)][i] =
                    origin[static_cast<size_t>(a)] + ax.amplitude * ax.base[i];
        }

        return out;
    }

    /* ================================================================== */
    /*  FOV shift                                                         */
    /* ================================================================== */

    namespace
    {
        /** The gradient state of one block, ready to evaluate at any time. */
        struct BlockGradients
        {
            Piecewise wave[3];
            double amplitude[3] = {0.0, 0.0, 0.0};

            /** k at @p t, in 1/m, given the block's starting k. */
            double k_at(int axis, double t, double origin) const
            {
                return origin + amplitude[axis] * wave[axis].cumulative_at(t);
            }

            /** The gradient itself at @p t, in Hz/m. */
            double g_at(int axis, double t) const
            {
                return amplitude[axis] * wave[axis].value_at(t);
            }
        };

        BlockGradients block_gradients(const Sequence& seq, const Block& block)
        {
            BlockGradients out;
            const int32_t grads[3] = {block.gx, block.gy, block.gz};
            for (int a = 0; a < 3; ++a)
                out.wave[a] = unit_waveform(seq, grads[a], &out.amplitude[a]);
            return out;
        }

        /** The shift's phase, in cycles, at @p t. */
        double shift_cycles(const BlockGradients& g, const std::array<double, 3>& shift,
                            const std::array<double, 3>& origin, double t)
        {
            double cycles = 0.0;
            for (int a = 0; a < 3; ++a)
            {
                if (shift[static_cast<size_t>(a)] == 0.0)
                    continue;
                cycles += shift[static_cast<size_t>(a)] *
                          g.k_at(a, t, origin[static_cast<size_t>(a)]);
            }
            return cycles;
        }

        /** The shift's frequency, in Hz, at @p t -- the instantaneous slope. */
        double shift_frequency(const BlockGradients& g, const std::array<double, 3>& shift,
                               double t)
        {
            double hz = 0.0;
            for (int a = 0; a < 3; ++a)
            {
                if (shift[static_cast<size_t>(a)] == 0.0)
                    continue;
                hz += shift[static_cast<size_t>(a)] * g.g_at(a, t);
            }
            return hz;
        }

        constexpr double kTwoPi = 6.283185307179586476925286766559;

    }  // namespace

    void apply_fov_shift(Sequence& seq, const std::array<double, 3>& shift_m,
                         FovShiftScope scope, int first, int last)
    {
        if (shift_m[0] == 0.0 && shift_m[1] == 0.0 && shift_m[2] == 0.0)
            return;

        /* Absolute k is accumulated over the whole sequence even when only
         * part of it is being shifted -- see the header. */
        const std::vector<std::array<double, 3>> origins = block_k_origins(seq);
        const int blocks = seq.num_blocks();

        const int from = std::max(first, 1);
        const int to = (last <= 0) ? blocks : std::min(last, blocks);

        for (int b = from; b <= to; ++b)
        {
            Block block = seq.get_block(b);
            if (block.rf <= 0 && block.adc <= 0)
                continue;

            const BlockGradients g = block_gradients(seq, block);
            const std::array<double, 3>& origin = origins[static_cast<size_t>(b)];

            /* -- RF: bake into the phase shape, referenced to its centre -- */
            if (block.rf > 0 && block.rf <= seq.rf_library().size())
            {
                double row[RF_WIDTH];
                const double* existing = seq.rf_library().row(static_cast<int>(block.rf));
                for (int c = 0; c < RF_WIDTH; ++c)
                    row[c] = existing[c];

                const int phase_id = static_cast<int>(row[2]);
                const int time_id = static_cast<int>(row[3]);
                const double center = row[4];
                const double delay = row[5];

                if (phase_id > 0 && phase_id <= seq.shape_library().size())
                {
                    const ShapeLibrary& shapes = seq.shape_library();
                    const int n = shapes.num_uncompressed(phase_id);

                    std::vector<double> phase;
                    if (shapes.is_compressed(phase_id))
                        phase = decompress_shape(shapes.samples(phase_id),
                                                 shapes.num_compressed(phase_id), n);
                    else
                        phase.assign(shapes.samples(phase_id), shapes.samples(phase_id) + n);

                    std::vector<double> times;
                    times.reserve(static_cast<size_t>(n));
                    const double raster = seq.rf_raster_time();
                    if (time_id > 0 && time_id <= shapes.size())
                    {
                        const int nt = shapes.num_uncompressed(time_id);
                        std::vector<double> ticks;
                        if (shapes.is_compressed(time_id))
                            ticks = decompress_shape(shapes.samples(time_id),
                                                     shapes.num_compressed(time_id), nt);
                        else
                            ticks.assign(shapes.samples(time_id),
                                         shapes.samples(time_id) + nt);
                        for (int i = 0; i < n && i < nt; ++i)
                            times.push_back(delay + ticks[static_cast<size_t>(i)] * raster);
                    }
                    else
                    {
                        for (int i = 0; i < n; ++i)
                            times.push_back(delay + (static_cast<double>(i) + 0.5) * raster);
                    }

                    if (static_cast<int>(phase.size()) == n &&
                        static_cast<int>(times.size()) == n)
                    {
                        const double at_center =
                            shift_cycles(g, shift_m, origin, delay + center);
                        for (int i = 0; i < n; ++i)
                            phase[static_cast<size_t>(i)] +=
                                shift_cycles(g, shift_m, origin,
                                             times[static_cast<size_t>(i)]) -
                                at_center;

                        row[2] = static_cast<double>(
                            seq.register_raw_shape(phase.data(), n));
                        block.rf = seq.register_rf(row, seq.rf_uses()[
                            static_cast<size_t>(block.rf) - 1]);
                        seq.set_block(b, block);
                    }
                }
            }

            /* -- ADC: frequency, phase, and whatever is left over -------- */
            if (scope == FovShiftScope::RfAndAdc && block.adc > 0 &&
                block.adc <= seq.adc_library().size())
            {
                double row[ADC_WIDTH];
                const double* existing = seq.adc_library().row(static_cast<int>(block.adc));
                for (int c = 0; c < ADC_WIDTH; ++c)
                    row[c] = existing[c];

                const int num_samples = static_cast<int>(row[0]);
                const double dwell = row[1];
                const double delay = row[2];
                if (num_samples > 0 && dwell > 0.0)
                {
                    const double t_center =
                        delay + 0.5 * static_cast<double>(num_samples) * dwell;

                    const double freq_hz = shift_frequency(g, shift_m, t_center);
                    const double cycles_center = shift_cycles(g, shift_m, origin, t_center);
                    const double phase_rad =
                        kTwoPi * (cycles_center - freq_hz * (t_center - delay));

                    /* Whatever the frequency and phase offsets cannot express.
                     * Identically zero under a constant gradient, which is why
                     * a Cartesian readout stays two scalars. */
                    std::vector<double> residual(static_cast<size_t>(num_samples), 0.0);
                    double worst = 0.0;
                    for (int i = 0; i < num_samples; ++i)
                    {
                        const double t = delay + (static_cast<double>(i) + 0.5) * dwell;
                        const double phi = kTwoPi * shift_cycles(g, shift_m, origin, t);
                        residual[static_cast<size_t>(i)] =
                            phi - kTwoPi * freq_hz * (t - delay) - phase_rad;
                        worst = std::max(worst,
                                         std::fabs(residual[static_cast<size_t>(i)]));
                    }

                    row[5] += freq_hz;    // freq_offset
                    row[6] += phase_rad;  // phase_offset
                    if (worst > 1e-12)
                        row[7] = static_cast<double>(
                            seq.register_raw_shape(residual.data(), num_samples));

                    block.adc = seq.register_adc(row);
                    seq.set_block(b, block);
                }
            }
        }
    }

    /* ================================================================== */
    /*  FOV scale                                                         */
    /* ================================================================== */

    void apply_fov_scale(Sequence& seq, const std::array<double, 3>& scale, int first, int last)
    {
        if (scale[0] == 1.0 && scale[1] == 1.0 && scale[2] == 1.0)
            return;

        const int blocks = seq.num_blocks();
        const int from = std::max(first, 1);
        const int to = (last <= 0) ? blocks : std::min(last, blocks);

        for (int b = from; b <= to; ++b)
        {
            Block block = seq.get_block(b);
            int32_t* axes[3] = {&block.gx, &block.gy, &block.gz};

            bool changed = false;
            for (int a = 0; a < 3; ++a)
            {
                const double factor = scale[static_cast<size_t>(a)];
                const int32_t id = *axes[a];
                if (factor == 1.0 || id <= 0)
                    continue;

                const int row_index = seq.grad_row(static_cast<int>(id));
                switch (seq.grad_kind(static_cast<int>(id)))
                {
                case GradKind::Trap:
                {
                    double row[TRAP_WIDTH];
                    const double* existing = seq.trap_library().row(row_index);
                    for (int c = 0; c < TRAP_WIDTH; ++c)
                        row[c] = existing[c];
                    row[0] *= factor;  // amplitude; the timings are untouched
                    *axes[a] = static_cast<int32_t>(seq.register_trap(row));
                    changed = true;
                    break;
                }
                case GradKind::Arbitrary:
                {
                    double row[ARB_WIDTH];
                    const double* existing = seq.arb_library().row(row_index);
                    for (int c = 0; c < ARB_WIDTH; ++c)
                        row[c] = existing[c];
                    /* Amplitude and the two endpoint values, which are stored
                     * absolute rather than normalised.  The shape ids in
                     * columns 3 and 4 are carried over untouched -- that is
                     * the whole point of scaling here. */
                    row[0] *= factor;
                    row[1] *= factor;
                    row[2] *= factor;
                    *axes[a] = static_cast<int32_t>(seq.register_arbitrary(row));
                    changed = true;
                    break;
                }
                case GradKind::None:
                    break;
                }
            }

            if (changed)
                seq.set_block(b, block);
        }
    }

    /* ================================================================== */
    /*  Carrying the base trajectory in a .seq                            */
    /* ================================================================== */

    void attach_base_trajectory(Sequence& seq)
    {
        seq.set_definition(PHASE_MODULATION_MODE_KEY,
                           Definition(std::string(PHASE_MODULATION_MODE_BASE_TRAJECTORY)));

        const int blocks = seq.num_blocks();

        /* Which shape each block wants, and which ADC row it currently uses.
         * Collected first because whether a row can simply be rewritten
         * depends on what its *other* users want, which is not known until
         * every block has been visited. */
        std::vector<int32_t> wanted(static_cast<size_t>(blocks) + 1, 0);
        std::vector<int32_t> adc_of(static_cast<size_t>(blocks) + 1, 0);

        for (int b = 1; b <= blocks; ++b)
        {
            const Block block = seq.get_block(b);
            if (block.adc <= 0)
                continue;
            adc_of[static_cast<size_t>(b)] = block.adc;

            const BaseTrajectory t = base_trajectory(seq, b);
            if (!t.has_adc)
                continue;

            const double window = static_cast<double>(t.num_samples) * t.dwell;
            if (!(window > 0.0))
                continue;

            std::vector<double> packed(static_cast<size_t>(3 * t.num_samples), 0.0);
            for (int a = 0; a < 3; ++a)
            {
                const AxisTrajectory& ax = t.axis[a];
                if (ax.base.empty())
                    continue;
                double* dst = packed.data() + static_cast<size_t>(a) * t.num_samples;
                for (int i = 0; i < t.num_samples; ++i)
                    dst[i] = ax.base[static_cast<size_t>(i)] / window;
            }

            wanted[static_cast<size_t>(b)] = static_cast<int32_t>(
                seq.register_raw_shape(packed.data(), static_cast<int>(packed.size())));
        }

        /* An ADC row all of whose users want the same shape is rewritten where
         * it stands.  Otherwise the row has to be split, and the users that
         * disagree get rows of their own.
         *
         * The distinction is worth drawing because `remove_duplicates`
         * collapses equal rows but does not collect unreferenced ones: always
         * appending would leave one orphaned ADC row per original behind in
         * the written file.  In the common case -- a sequence built block by
         * block, where every block registered its own row -- each row has a
         * single user and nothing is ever split.
         */
        const int adc_rows = seq.adc_library().size();
        std::vector<int32_t> agreed(static_cast<size_t>(adc_rows) + 1, -1);
        std::vector<char> conflicted(static_cast<size_t>(adc_rows) + 1, 0);

        for (int b = 1; b <= blocks; ++b)
        {
            const int32_t id = adc_of[static_cast<size_t>(b)];
            if (id <= 0 || id > adc_rows)
                continue;
            int32_t& seen = agreed[static_cast<size_t>(id)];
            if (seen == -1)
                seen = wanted[static_cast<size_t>(b)];
            else if (seen != wanted[static_cast<size_t>(b)])
                conflicted[static_cast<size_t>(id)] = 1;
        }

        for (int id = 1; id <= adc_rows; ++id)
        {
            if (conflicted[static_cast<size_t>(id)] || agreed[static_cast<size_t>(id)] <= 0)
                continue;
            seq.adc_library().row(id)[7] =
                static_cast<double>(agreed[static_cast<size_t>(id)]);
        }

        for (int b = 1; b <= blocks; ++b)
        {
            const int32_t id = adc_of[static_cast<size_t>(b)];
            const int32_t shape = wanted[static_cast<size_t>(b)];
            if (id <= 0 || id > adc_rows || shape <= 0)
                continue;
            if (!conflicted[static_cast<size_t>(id)])
                continue;

            double row[ADC_WIDTH];
            const double* existing = seq.adc_library().row(static_cast<int>(id));
            for (int c = 0; c < ADC_WIDTH; ++c)
                row[c] = existing[c];
            row[7] = static_cast<double>(shape);

            Block block = seq.get_block(b);
            block.adc = seq.register_adc(row);
            seq.set_block(b, block);
        }
    }

    bool has_base_trajectory(const Sequence& seq)
    {
        const Definition* mode = seq.definition(PHASE_MODULATION_MODE_KEY);
        return mode != nullptr && mode->kind() == Definition::Kind::Text &&
               mode->text() == PHASE_MODULATION_MODE_BASE_TRAJECTORY;
    }

    BaseTrajectory read_base_trajectory(const Sequence& seq, int block_index)
    {
        if (block_index < 1 || block_index > seq.num_blocks())
            throw std::out_of_range("read_base_trajectory: block index out of range");

        const Block block = seq.get_block(block_index);

        BaseTrajectory out;
        const int32_t grads[3] = {block.gx, block.gy, block.gz};
        for (int a = 0; a < 3; ++a)
        {
            out.axis[a].present = grads[a] > 0;
            if (grads[a] > 0)
            {
                double amplitude = 0.0;
                const Piecewise p = unit_waveform(seq, grads[a], &amplitude);
                out.axis[a].amplitude = amplitude;
                out.axis[a].moment = p.total();
            }
        }

        if (block.adc <= 0)
            return out;

        const double* adc = seq.adc_library().row(static_cast<int>(block.adc));
        const int num_samples = static_cast<int>(adc[0]);
        const double dwell = adc[1];
        const int shape_id = static_cast<int>(adc[7]);
        if (num_samples <= 0 || dwell <= 0.0 || shape_id <= 0 ||
            shape_id > seq.shape_library().size())
            return out;

        const ShapeLibrary& shapes = seq.shape_library();
        const int stored = shapes.num_uncompressed(shape_id);
        if (stored != 3 * num_samples)
            return out;

        std::vector<double> decoded;
        const double* samples = shapes.samples(shape_id);
        if (shapes.is_compressed(shape_id))
        {
            decoded = decompress_shape(samples, shapes.num_compressed(shape_id), stored);
            if (static_cast<int>(decoded.size()) != stored)
                return out;
            samples = decoded.data();
        }

        out.has_adc = true;
        out.num_samples = num_samples;
        out.dwell = dwell;
        out.delay = adc[2];

        const double window = static_cast<double>(num_samples) * dwell;
        for (int a = 0; a < 3; ++a)
        {
            const double* src = samples + static_cast<size_t>(a) * num_samples;

            /* Flatness is read back off the samples rather than carried, which
             * is what storing every axis buys: an axis that never varies is
             * one a consumer takes the DC of from the encoding counters. */
            double lo = src[0];
            double hi = src[0];
            for (int i = 1; i < num_samples; ++i)
            {
                lo = std::min(lo, src[i]);
                hi = std::max(hi, src[i]);
            }
            out.axis[a].constant = (hi - lo) <= 1e-12;

            if (!out.axis[a].present)
                continue;

            out.axis[a].base.resize(static_cast<size_t>(num_samples));
            for (int i = 0; i < num_samples; ++i)
                out.axis[a].base[static_cast<size_t>(i)] = src[i] * window;
        }

        return out;
    }

}  // namespace pulseq
