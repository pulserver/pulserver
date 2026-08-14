/**
 * @file trajectory.cpp
 * @brief Base k-space for a block's ADC window.  See trajectory.hpp.
 *
 * Both gradient kinds reduce to the same thing: a piecewise-linear normalised
 * waveform on a grid of breakpoints -- see waveform.hpp, which holds that
 * reduction and is shared with moments.cpp.  Once in that form the cumulative
 * integral is exact segment by segment and evaluating it at an ADC sample
 * centre is one interpolation, so nothing is resampled onto a raster it was
 * not already on.
 */

#include "pulseq/trajectory.hpp"

#include "pulseq/kspace.hpp"
#include "pulseq/shape.hpp"
#include "waveform.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pulseq
{

    namespace
    {
        using detail::Piecewise;
        using detail::unit_waveform;

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

    /*
     * One line of glue over the C89 core.  See the header for why there is no
     * second implementation here any more.
     *
     * The core reports origins as a dense array indexed from `first_block`;
     * this returns Pulseq's 1-based layout with an unused `[0]`, because that
     * is what `absolute_trajectory` and `apply_fov_shift` index with.
     */
    std::vector<std::array<double, 3>> block_k_origins(Sequence& seq)
    {
        KSpaceOptions options;
        /* Logical frame: a rotation belongs to the block that carries it, and
         * `dr . k` is invariant when both are rotated anyway. */
        options.apply_rotation = false;
        options.block_origins = true;
        /* Nothing here reads a sample or an echo, and both are the expensive
         * half of a plan. */
        options.materialize_samples = false;
        options.derive_center_sample = false;

        const KSpace ks = calculate_kspace(seq, options);

        std::vector<std::array<double, 3>> out(ks.block_origins.size() + 1,
                                               std::array<double, 3>{0.0, 0.0, 0.0});
        for (size_t b = 0; b < ks.block_origins.size(); ++b)
            out[b + 1] = ks.block_origins[b];
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

        /*
         * Whether a readout's share of a shift must be left to the consumer.
         *
         * A block carrying a ROTATIONS extension plays its gradients at an
         * arbitrary physical angle, so the shift phase is orientation-
         * dependent and cannot be pre-baked -- and the consumer needs the
         * base trajectory for its metadata regardless.  A gradient that
         * varies across the ADC window cannot be expressed as a frequency at
         * all.  Everything else -- Cartesian, unrotated -- is two scalars,
         * and deferring it would store a trajectory nobody needs.
         */
        bool adc_defers_to_consumer(const Sequence& seq, const Block& block)
        {
            if (block.adc <= 0 || block.adc > seq.adc_library().size())
                return false;

            const int rotations = seq.find_extension_type_id("ROTATIONS");
            if (rotations > 0)
            {
                int32_t node = block.ext;
                while (node > 0 && node <= seq.extensions_library().size())
                {
                    const int32_t* link = seq.extensions_library().row(static_cast<int>(node));
                    if (link[0] == rotations)
                        return true;
                    node = link[2];
                }
            }

            const double* adc = seq.adc_library().row(static_cast<int>(block.adc));
            const int num_samples = static_cast<int>(adc[0]);
            const double dwell = adc[1];
            const double delay = adc[2];
            if (num_samples <= 0 || dwell <= 0.0)
                return false;

            const double t_first = delay + 0.5 * dwell;
            const double t_last =
                delay + (static_cast<double>(num_samples) - 0.5) * dwell;

            const BlockGradients g = block_gradients(seq, block);
            for (int a = 0; a < 3; ++a)
            {
                if (!is_flat(g.wave[a], t_first, t_last))
                    return true;
            }
            return false;
        }

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

            /* -- RF: a constant gradient is a frequency; anything else is
             *       baked into the phase shape, referenced to its centre -- */
            if (block.rf > 0 && block.rf <= seq.rf_library().size())
            {
                double row[RF_WIDTH];
                const double* existing = seq.rf_library().row(static_cast<int>(block.rf));
                for (int c = 0; c < RF_WIDTH; ++c)
                    row[c] = existing[c];

                const int mag_id = static_cast<int>(row[1]);
                const int phase_id = static_cast<int>(row[2]);
                const int time_id = static_cast<int>(row[3]);
                const double center = row[4];
                const double delay = row[5];

                const ShapeLibrary& shapes = seq.shape_library();
                int n = 0;
                if (phase_id > 0 && phase_id <= shapes.size())
                    n = shapes.num_uncompressed(phase_id);
                else if (mag_id > 0 && mag_id <= shapes.size())
                    n = shapes.num_uncompressed(mag_id);

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

                if (n > 0 && static_cast<int>(times.size()) == n)
                {
                    const double t_center = delay + center;
                    const double at_center = shift_cycles(g, shift_m, origin, t_center);
                    const double freq_hz = shift_frequency(g, shift_m, t_center);

                    /* How far the added phase departs from a straight line at
                     * the centre's slope, in radians -- the same guard the ADC
                     * branch puts on its residual. */
                    double worst = 0.0;
                    for (int i = 0; i < n; ++i)
                    {
                        const double t = times[static_cast<size_t>(i)];
                        const double linear = freq_hz * (t - t_center);
                        const double cycles =
                            shift_cycles(g, shift_m, origin, t) - at_center;
                        worst = std::max(worst, std::fabs(kTwoPi * (cycles - linear)));
                    }

                    if (worst <= 1e-12)
                    {
                        /* Constant gradient across the pulse: the shift is a
                         * linear phase, and the row's own frequency and phase
                         * columns carry that exactly.  No shape is registered,
                         * and -- because the centre's share is subtracted --
                         * the pair is independent of the block's k origin, so
                         * every block sharing this pulse and gradient dedups
                         * onto one row.  Consumers read the pair as
                         * `phase + 2*pi*freq*t` with t from the shape's start,
                         * so zero-at-centre means backing out the centre. */
                        row[8] += freq_hz;
                        row[9] -= kTwoPi * freq_hz * center;
                        block.rf = seq.register_rf(row, seq.rf_uses()[
                            static_cast<size_t>(block.rf) - 1]);
                        seq.set_block(b, block);
                    }
                    else
                    {
                        std::vector<double> phase;
                        if (phase_id > 0 && phase_id <= shapes.size())
                        {
                            if (shapes.is_compressed(phase_id))
                                phase = decompress_shape(shapes.samples(phase_id),
                                                         shapes.num_compressed(phase_id),
                                                         n);
                            else
                                phase.assign(shapes.samples(phase_id),
                                             shapes.samples(phase_id) + n);
                        }
                        else
                        {
                            phase.assign(static_cast<size_t>(n), 0.0);
                        }

                        if (static_cast<int>(phase.size()) == n)
                        {
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
            }

            /* -- ADC: frequency, phase, and whatever is left over -------- */
            const bool bake_adc =
                scope == FovShiftScope::RfAndAdc ||
                (scope == FovShiftScope::Server && !adc_defers_to_consumer(seq, block));
            if (bake_adc && block.adc > 0 && block.adc <= seq.adc_library().size())
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

            /* A Cartesian, unrotated readout takes its shift as two scalars
             * and its k from the encoding counters -- storing a trajectory
             * for it would be pure weight.  Only the readouts the consumer
             * actually has to finish get one. */
            if (!adc_defers_to_consumer(seq, block))
                continue;

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

        /* The marker says "phase_modulation holds a base trajectory"; a
         * sequence in which no readout stores one must not carry it, or a
         * fully Cartesian server-mode file would read as something other than
         * the native file it is byte-for-byte. */
        bool attached_any = false;
        for (int b = 1; b <= blocks; ++b)
            if (wanted[static_cast<size_t>(b)] > 0)
            {
                attached_any = true;
                break;
            }
        if (attached_any)
            seq.set_definition(PHASE_MODULATION_MODE_KEY,
                               Definition(std::string(PHASE_MODULATION_MODE_BASE_TRAJECTORY)));

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
