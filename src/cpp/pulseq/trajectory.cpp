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

#include "ktraj.hpp"
#include "pulseq/shape.hpp"
#include "waveform.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace pulseq
{

    namespace
    {
        using detail::is_flat;
        using detail::Piecewise;
        using detail::unit_waveform;

        /*
         * Constant across [t0, t1] to the bit?
         *
         * Sampled where @ref is_flat samples, and exact where it tolerates:
         * this answers whether the accumulated phase is a straight line, and
         * a waveform that merely nearly holds still bends it by a little.
         * How much bending is worth carrying is a threshold the caller
         * already applies to the residual itself; the point of asking here is
         * that a gradient which does not move at all has no residual to
         * compute, so the sample loop can be skipped rather than run and
         * found to be zero.
         */
        bool is_exactly_constant(const Piecewise& p, double t0, double t1)
        {
            if (p.empty())
                return true;

            const double held = p.value_at(t0);
            if (p.value_at(t1) != held)
                return false;

            for (size_t i = 0; i < p.t.size(); ++i)
            {
                if (p.t[i] <= t0 || p.t[i] >= t1)
                    continue;
                if (p.v[i] != held)
                    return false;
            }
            return true;
        }

    } // namespace

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
     * One line of glue over the core, which reports origins as a flat array
     * indexed from its own first block; this returns Pulseq's 1-based layout
     * with an unused `[0]`, because that is what `absolute_trajectory` and
     * `apply_fov_shift` index with.
     *
     * Driving the plan directly rather than going through `calculate_kspace`
     * saves a copy of the whole array -- fifty megabytes on a two-million-block
     * scan, for a caller that reads it once.
     */
    std::vector<std::array<double, 3>> block_k_origins(const Sequence& seq)
    {
        detail::KTrajOptions options;
        /* Logical frame: a rotation belongs to the block that carries it, and
         * `dr . k` is invariant when both are rotated anyway. */
        options.apply_rotation = false;
        /* Nothing here reads a sample or an echo, and deriving the echo is the
         * expensive half of a plan. */
        options.derive_center_sample = false;

        const detail::KTrajPlan plan(seq, options);
        const std::vector<double>& origins = plan.block_origins();
        const size_t count = origins.size() / 3;

        std::vector<std::array<double, 3>> out(count + 1, std::array<double, 3>{0.0, 0.0, 0.0});
        for (size_t b = 0; b < count; ++b)
        {
            out[b + 1] = {{origins[b * 3], origins[b * 3 + 1], origins[b * 3 + 2]}};
        }
        return out;
    }

    std::array<std::vector<double>, 3> absolute_trajectory(
        const Sequence& seq,
        int block_index,
        const std::array<double, 3>& origin)
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

        /**
         * The shift's phase, in cycles, at @p t, with the block's k origin
         * left out -- everything the block's own gradients sweep.
         *
         * The split is what makes a readout's share of a shift memoizable.
         * Writing the origin's contribution as `dr . k0`, the phase is
         * `dr . k0 + swept(t)`, and both the residual the ADC carries and the
         * shape the RF carries are *differences* of that phase against its
         * value at a reference time -- so `dr . k0` cancels out of them
         * exactly.  What is left depends only on the gradient rows and the
         * event timings, which is to say on what the block plays and not on
         * where in k-space it plays it.  Every readout of a phase-encode
         * table therefore shares one residual.
         *
         * Subtracting the two whole phases instead is the same number in
         * exact arithmetic and a worse one in floating point: `dr . k0`
         * reaches hundreds of cycles at the edge of k-space while the
         * residual is of order a milliradian, so cancelling them costs about
         * as many digits as the answer has.
         */
        double swept_cycles(const BlockGradients& g, const std::array<double, 3>& shift, double t)
        {
            double cycles = 0.0;
            for (int a = 0; a < 3; ++a)
            {
                if (shift[static_cast<size_t>(a)] == 0.0)
                    continue;
                cycles +=
                    shift[static_cast<size_t>(a)] * g.amplitude[a] * g.wave[a].cumulative_at(t);
            }
            return cycles;
        }

        /** `dr . k0`: what the block's entry point contributes, in cycles. */
        double origin_cycles(
            const std::array<double, 3>& shift,
            const std::array<double, 3>& origin)
        {
            double cycles = 0.0;
            for (int a = 0; a < 3; ++a)
                cycles += shift[static_cast<size_t>(a)] * origin[static_cast<size_t>(a)];
            return cycles;
        }

        /** The shift's frequency, in Hz, at @p t -- the instantaneous slope. */
        double shift_frequency(
            const BlockGradients& g,
            const std::array<double, 3>& shift,
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

        /**
         * Is the shift's phase a straight line across [t0, t1]?
         *
         * Only the axes the shift actually has a component along are asked:
         * an axis contributing no phase cannot bend it, however it moves.
         */
        bool shift_is_linear(
            const BlockGradients& g,
            const std::array<double, 3>& shift,
            double t0,
            double t1)
        {
            for (int a = 0; a < 3; ++a)
            {
                if (shift[static_cast<size_t>(a)] == 0.0)
                    continue;
                if (!is_exactly_constant(g.wave[a], t0, t1))
                    return false;
            }
            return true;
        }

        /**
         * What a block's base trajectory is a function of, as bytes.
         *
         * The gradient rows as they stand, and the ADC's sample count, dwell
         * and delay -- the whole of what `base_trajectory` reads.  Deliberately
         * the rows rather than the ids that name them: the library is not
         * deduplicated until the sequence is written, so a scan that plays one
         * arm ten thousand times holds ten thousand identical rows, and an id
         * would find nothing shared where there is nothing but sharing.  The
         * ADC's frequency and phase are left out because the samples do not
         * depend on them and RF spoiling moves them every shot; the rotation
         * is left out because composing it is the consumer's job, which is
         * what lets one stored arm serve every shot that turns it.
         */
        void played_key_into(
            const Sequence& seq,
            const Block& block,
            const double* adc,
            std::string& key)
        {
            const auto push = [&key](const void* bytes, size_t count)
            { key.append(static_cast<const char*>(bytes), count); };

            const int32_t grads[3] = {block.gx, block.gy, block.gz};
            for (int a = 0; a < 3; ++a)
            {
                const GradKind kind =
                    (grads[a] > 0) ? seq.grad_kind(static_cast<int>(grads[a])) : GradKind::None;
                const int tag = static_cast<int>(kind);
                push(&tag, sizeof(tag));

                const int row =
                    (kind == GradKind::None) ? 0 : seq.grad_row(static_cast<int>(grads[a]));
                switch (kind)
                {
                case GradKind::Trap:
                    push(seq.trap_library().row(row), TRAP_WIDTH * sizeof(double));
                    break;
                case GradKind::Arbitrary:
                    push(seq.arb_library().row(row), ARB_WIDTH * sizeof(double));
                    break;
                case GradKind::None:
                    break;
                }
            }

            push(adc, 3 * sizeof(double)); // num_samples, dwell, delay
        }

        constexpr double kTwoPi = 6.283185307179586476925286766559;

        /*
         * Whether a readout's share of a shift must be left to the consumer.
         *
         * Not a statement that a rotated readout's phase is unknowable here.
         * With `R` the block's extension, the shift phase is
         * `dr_logical . (R k_logical)`: the prescribed FOV rotation cancels
         * against the logical shift, but `R` does not, and `R` is in the file.
         * What defers it is narrower -- `swept_cycles` and `block_k_origins`
         * both work in the UNROTATED logical frame, so what this function can
         * compute for a rotated block is `dr . k_logical`, which is the wrong
         * number.  The guard keeps that number out of the file; the consumer,
         * which composes `R` when it builds the trajectory, gets it right.
         *
         * A gradient that varies across the ADC window cannot be expressed as
         * a frequency at all.  Everything else -- Cartesian, unrotated -- is
         * two scalars, and deferring it would store a trajectory nobody needs.
         *
         * The question is about the readout, not about any one shift: the
         * consumer that has to finish a rotated or swept readout needs its
         * base trajectory for the k-space it reports either way, which is why
         * `attach_base_trajectory` -- where no shift exists at all -- asks
         * exactly this.
         */
        /** Does @p block carry a ROTATIONS extension? */
        bool block_is_rotated(const Sequence& seq, const Block& block)
        {
            const int rotations = seq.find_extension_type_id("ROTATIONS");
            if (rotations <= 0)
                return false;

            int32_t node = block.ext;
            while (node > 0 && node <= seq.extensions_library().size())
            {
                const int32_t* link = seq.extensions_library().row(static_cast<int>(node));
                if (link[0] == rotations)
                    return true;
                node = link[2];
            }
            return false;
        }

        /**
         * Does a gradient move across the ADC window?
         *
         * The half of the deferral question that depends only on which events
         * the block plays, and so answers the same for every block playing
         * them -- unlike the rotation, which is the block's own.
         */
        bool adc_gradients_vary(const Sequence& seq, const Block& block, const BlockGradients& g)
        {
            const double* adc = seq.adc_library().row(static_cast<int>(block.adc));
            const int num_samples = static_cast<int>(adc[0]);
            const double dwell = adc[1];
            const double delay = adc[2];
            if (num_samples <= 0 || dwell <= 0.0)
                return false;

            const double t_first = delay + 0.5 * dwell;
            const double t_last = delay + (static_cast<double>(num_samples) - 0.5) * dwell;

            for (int a = 0; a < 3; ++a)
            {
                if (!is_flat(g.wave[a], t_first, t_last))
                    return true;
            }
            return false;
        }

        /* ============================================================== */
        /*  What a shift does to a block, given only what it plays         */
        /* ============================================================== */

        /**
         * What a block plays, as canonical event ids: its three gradients and
         * one of its RF or ADC.
         *
         * Canonical rather than raw because the libraries are not deduplicated
         * until the sequence is written -- see `detail::canonical_ids`.  Ids
         * rather than the rows themselves because this is built and hashed
         * once per block of the scan, and twenty bytes of integers is an order
         * of magnitude less work than a hundred and sixty of doubles.
         */
        struct PlanKey
        {
            int grad[3] = {0, 0, 0};
            int event = 0;

            bool operator==(const PlanKey& other) const
            {
                return grad[0] == other.grad[0] && grad[1] == other.grad[1] &&
                    grad[2] == other.grad[2] && event == other.event;
            }
        };

        struct PlanKeyHash
        {
            size_t operator()(const PlanKey& key) const
            {
                size_t h = detail::HASH_SEED;
                detail::hash_int(h, key.grad[0]);
                detail::hash_int(h, key.grad[1]);
                detail::hash_int(h, key.grad[2]);
                detail::hash_int(h, key.event);
                return h;
            }
        };

        /**
         * The RF side of a shift, which is origin-free in full.
         *
         * Under a gradient that is constant across the pulse the shift is a
         * frequency and a phase on the row itself; otherwise it is a phase
         * shape.  Either way the centre's share is subtracted, and with it the
         * `dr . k0` term -- so the resulting row is the same for every block
         * playing this pulse under these gradients, and one registration
         * serves all of them.
         */
        struct RfPlan
        {
            /** The row to point the block at, or 0 to leave it alone. */
            int32_t rf = 0;
        };

        /** The readout side of a shift, less the block's own entry point. */
        struct AdcPlan
        {
            double freq_hz = 0.0;
            /** The shift's phase at the window centre, origin excluded. */
            double cycles_center = 0.0;
            double t_center = 0.0;
            /** Whether the shift's phase is a straight line across the window. */
            bool linear = true;
            /** Whether any gradient moves across the ADC window. */
            bool varies = false;
            /** False for a degenerate ADC row, which is left untouched. */
            bool valid = false;
            /**
             * The residual phase shape, built on first use rather than with
             * the rest of the plan: a server-mode readout that defers to the
             * consumer never carries one, and registering it anyway would
             * leave a shape in the file that nothing references.
             */
            int32_t residual = 0;
            bool residual_known = false;
        };

        RfPlan plan_rf(
            Sequence& seq,
            const BlockGradients& g,
            const std::array<double, 3>& shift,
            int rf_id,
            char use)
        {
            RfPlan plan;

            double row[RF_WIDTH];
            const double* existing = seq.rf_library().row(rf_id);
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

            /* The pulse's own span, and the sample times only if they are
             * needed: a gradient that holds still across the span writes two
             * scalars, and reading every sample to discover that costs more
             * than the rest of the block put together. */
            const double raster = seq.rf_raster_time();
            std::vector<double> ticks;
            bool sampled = n > 0;
            double t_first = delay + 0.5 * raster;
            double t_last = delay + (static_cast<double>(n) - 0.5) * raster;
            if (time_id > 0 && time_id <= shapes.size())
            {
                const int nt = shapes.num_uncompressed(time_id);
                sampled = sampled && nt >= n;
                if (sampled)
                {
                    if (shapes.is_compressed(time_id))
                        ticks = decompress_shape(
                            shapes.samples(time_id),
                            shapes.num_compressed(time_id),
                            nt);
                    else
                        ticks.assign(shapes.samples(time_id), shapes.samples(time_id) + nt);
                    t_first = delay + ticks.front() * raster;
                    t_last = delay + ticks[static_cast<size_t>(n) - 1] * raster;
                }
            }

            if (!sampled)
                return plan;

            const double t_center = delay + center;
            const double at_center = swept_cycles(g, shift, t_center);
            const double freq_hz = shift_frequency(g, shift, t_center);

            /* How far the added phase departs from a straight line at the
             * centre's slope, in radians -- the same guard the ADC branch puts
             * on its residual.  Zero by construction when the gradient does not
             * move, so it is only measured when it does. */
            std::vector<double> times;
            double worst = 0.0;
            if (!shift_is_linear(g, shift, std::min(t_first, t_center), std::max(t_last, t_center)))
            {
                times.reserve(static_cast<size_t>(n));
                for (int i = 0; i < n; ++i)
                    times.push_back(
                        ticks.empty() ? delay + (static_cast<double>(i) + 0.5) * raster
                                      : delay + ticks[static_cast<size_t>(i)] * raster);

                for (int i = 0; i < n; ++i)
                {
                    const double t = times[static_cast<size_t>(i)];
                    const double linear = freq_hz * (t - t_center);
                    const double cycles = swept_cycles(g, shift, t) - at_center;
                    worst = std::max(worst, std::fabs(kTwoPi * (cycles - linear)));
                }
            }

            if (worst <= 1e-12)
            {
                /* Constant gradient across the pulse: the shift is a linear
                 * phase, and the row's own frequency and phase columns carry
                 * that exactly.  No shape is registered.  Consumers read the
                 * pair as `phase + 2*pi*freq*t` with t from the shape's start,
                 * so zero-at-centre means backing out the centre. */
                row[8] += freq_hz;
                row[9] -= kTwoPi * freq_hz * center;
                plan.rf = static_cast<int32_t>(seq.register_rf(row, use));
                return plan;
            }

            std::vector<double> phase;
            if (phase_id > 0 && phase_id <= shapes.size())
            {
                if (shapes.is_compressed(phase_id))
                    phase = decompress_shape(
                        shapes.samples(phase_id),
                        shapes.num_compressed(phase_id),
                        n);
                else
                    phase.assign(shapes.samples(phase_id), shapes.samples(phase_id) + n);
            }
            else
            {
                phase.assign(static_cast<size_t>(n), 0.0);
            }

            if (static_cast<int>(phase.size()) != n)
                return plan;

            for (int i = 0; i < n; ++i)
                phase[static_cast<size_t>(i)] +=
                    swept_cycles(g, shift, times[static_cast<size_t>(i)]) - at_center;

            row[2] = static_cast<double>(seq.register_raw_shape(phase.data(), n));
            plan.rf = static_cast<int32_t>(seq.register_rf(row, use));
            return plan;
        }

        AdcPlan plan_adc(
            Sequence& seq,
            const BlockGradients& g,
            const std::array<double, 3>& shift,
            int adc_id)
        {
            AdcPlan plan;

            const double* adc = seq.adc_library().row(adc_id);
            const int num_samples = static_cast<int>(adc[0]);
            const double dwell = adc[1];
            const double delay = adc[2];
            if (num_samples <= 0 || dwell <= 0.0)
                return plan;

            plan.valid = true;
            plan.t_center = delay + 0.5 * static_cast<double>(num_samples) * dwell;
            plan.freq_hz = shift_frequency(g, shift, plan.t_center);
            plan.cycles_center = swept_cycles(g, shift, plan.t_center);

            const double t_first = delay + 0.5 * dwell;
            const double t_last = delay + (static_cast<double>(num_samples) - 0.5) * dwell;

            for (int a = 0; a < 3; ++a)
            {
                if (!is_flat(g.wave[a], t_first, t_last))
                {
                    plan.varies = true;
                    break;
                }
            }

            /* Whether the frequency and phase offsets can express the whole
             * shift.  Identically so under a constant gradient, which is why a
             * Cartesian readout stays two scalars -- and why a residual is not
             * built sample by sample only to come out flat. */
            plan.linear = shift_is_linear(g, shift, t_first, t_last);
            return plan;
        }

        /** The residual the frequency and phase offsets cannot express. */
        void build_residual(
            Sequence& seq,
            const BlockGradients& g,
            const std::array<double, 3>& shift,
            int adc_id,
            AdcPlan& plan)
        {
            plan.residual_known = true;

            const double* adc = seq.adc_library().row(adc_id);
            const int num_samples = static_cast<int>(adc[0]);
            const double dwell = adc[1];
            const double delay = adc[2];

            std::vector<double> residual(static_cast<size_t>(num_samples), 0.0);
            double worst = 0.0;
            for (int i = 0; i < num_samples; ++i)
            {
                const double t = delay + (static_cast<double>(i) + 0.5) * dwell;
                residual[static_cast<size_t>(i)] = kTwoPi *
                    (swept_cycles(g, shift, t) - plan.cycles_center -
                     plan.freq_hz * (t - plan.t_center));
                worst = std::max(worst, std::fabs(residual[static_cast<size_t>(i)]));
            }

            if (worst > 1e-12)
                plan.residual =
                    static_cast<int32_t>(seq.register_raw_shape(residual.data(), num_samples));
        }

    } // namespace

    void apply_fov_shift(
        Sequence& seq,
        const std::array<double, 3>& shift_m,
        FovShiftScope scope,
        int first,
        int last)
    {
        if (shift_m[0] == 0.0 && shift_m[1] == 0.0 && shift_m[2] == 0.0)
            return;

        /* Absolute k is accumulated over the whole sequence even when only
         * part of it is being shifted -- see the header. */
        /*
         * One plan, for two things: where k stands at the start of every
         * block, and which event rows mean the same thing.  Absolute k is
         * accumulated over the whole sequence even when only part of it is
         * being shifted -- see the header.
         */
        detail::KTrajOptions plan_options;
        plan_options.apply_rotation = false;
        plan_options.derive_center_sample = false;
        const detail::KTrajPlan plan(seq, plan_options);

        const std::vector<double>& origins = plan.block_origins();
        const std::vector<int>& grad_canon = plan.grad_canon();
        const std::vector<int>& adc_canon = plan.adc_canon();

        /*
         * RF rows collapse the same way, and have to be collapsed here: the
         * trajectory never looks at a pulse's shape, so the plan has no reason
         * to have done it.
         */
        const std::vector<int> rf_canon = detail::canonical_ids(
            seq.rf_library().size(),
            [&seq](int id)
            {
                const double* row = seq.rf_library().row(id);
                size_t h = detail::HASH_SEED;
                for (int c = 0; c < RF_WIDTH; ++c)
                    detail::hash_double(h, row[c]);
                detail::hash_int(h, seq.rf_uses()[static_cast<size_t>(id) - 1]);
                return h;
            },
            [&seq](int a, int b)
            {
                if (seq.rf_uses()[static_cast<size_t>(a) - 1] !=
                    seq.rf_uses()[static_cast<size_t>(b) - 1])
                    return false;
                const double* ra = seq.rf_library().row(a);
                const double* rb = seq.rf_library().row(b);
                for (int c = 0; c < RF_WIDTH; ++c)
                    if (ra[c] != rb[c])
                        return false;
                return true;
            });

        const int blocks = seq.num_blocks();

        const int from = std::max(first, 1);
        const int to = (last <= 0) ? blocks : std::min(last, blocks);

        /*
         * What the shift does to a block, worked out once per distinct thing
         * a block plays.
         *
         * Everything below the origin is memoizable, and that is not a
         * coincidence: see @ref swept_cycles.  A phase-encode table plays one
         * readout ten thousand times at ten thousand different entry points,
         * and without this each one rebuilds the same waveforms, re-derives
         * the same residual sample by sample, and registers its own copy of
         * the same shape -- which the writer then deduplicates back down to
         * the one row it always was.
         */
        std::unordered_map<PlanKey, RfPlan, PlanKeyHash> rf_plans;
        std::unordered_map<PlanKey, AdcPlan, PlanKeyHash> adc_plans;

        for (int b = from; b <= to; ++b)
        {
            Block block = seq.get_block(b);
            if (block.rf <= 0 && block.adc <= 0)
                continue;

            const size_t slot = static_cast<size_t>(b - 1) * 3;
            const std::array<double, 3> origin{
                {origins[slot], origins[slot + 1], origins[slot + 2]}};

            /* What the block plays, as canonical ids: three gradients, and
             * then whichever event this key is for. */
            PlanKey key{};
            const int32_t grads[3] = {block.gx, block.gy, block.gz};
            for (int a = 0; a < 3; ++a)
                key.grad[a] = (grads[a] > 0) ? grad_canon[static_cast<size_t>(grads[a])] : 0;

            /*
             * The waveforms, built at most once per block and only when a plan
             * actually has to be worked out.  On a scan whose readouts repeat
             * -- which is every scan this is slow on -- that is a handful of
             * times rather than once per block, and decompressing a spiral arm
             * two million times is exactly what it saves.
             */
            bool have_gradients = false;
            BlockGradients g;
            const auto gradients = [&]() -> const BlockGradients&
            {
                if (!have_gradients)
                {
                    g = block_gradients(seq, block);
                    have_gradients = true;
                }
                return g;
            };

            /* -- RF: a constant gradient is a frequency; anything else is
             *       baked into the phase shape, referenced to its centre -- */
            if (block.rf > 0 && block.rf <= seq.rf_library().size())
            {
                const char use = seq.rf_uses()[static_cast<size_t>(block.rf) - 1];
                key.event = rf_canon[static_cast<size_t>(block.rf)];

                auto found = rf_plans.find(key);
                if (found == rf_plans.end())
                    found =
                        rf_plans
                            .emplace(
                                key,
                                plan_rf(seq, gradients(), shift_m, static_cast<int>(block.rf), use))
                            .first;

                if (found->second.rf > 0)
                {
                    block.rf = found->second.rf;
                    seq.set_block(b, block);
                }
            }

            /* -- ADC: frequency, phase, and whatever is left over -------- */
            if (block.adc > 0 && block.adc <= seq.adc_library().size())
            {
                key.event = adc_canon[static_cast<size_t>(block.adc)];

                auto found = adc_plans.find(key);
                if (found == adc_plans.end())
                    found =
                        adc_plans
                            .emplace(
                                key,
                                plan_adc(seq, gradients(), shift_m, static_cast<int>(block.adc)))
                            .first;
                AdcPlan& adc_plan = found->second;

                /* The rotation is the block's own, so whether a server-mode
                 * readout defers cannot be memoized with the rest. */
                const bool bake = scope == FovShiftScope::RfAndAdc ||
                    (scope == FovShiftScope::Server &&
                     !(adc_plan.varies || block_is_rotated(seq, block)));

                if (bake && adc_plan.valid)
                {
                    if (!adc_plan.linear && !adc_plan.residual_known)
                        build_residual(
                            seq,
                            gradients(),
                            shift_m,
                            static_cast<int>(block.adc),
                            adc_plan);

                    double row[ADC_WIDTH];
                    const double* existing = seq.adc_library().row(static_cast<int>(block.adc));
                    for (int c = 0; c < ADC_WIDTH; ++c)
                        row[c] = existing[c];

                    /* The only part the block's own entry point reaches: the
                     * phase the shift already has at the window's centre. */
                    const double phase_rad = kTwoPi *
                        (origin_cycles(shift_m, origin) + adc_plan.cycles_center -
                         adc_plan.freq_hz * (adc_plan.t_center - row[2]));

                    row[5] += adc_plan.freq_hz; // freq_offset
                    row[6] += phase_rad;        // phase_offset
                    if (adc_plan.residual > 0)
                        row[7] = static_cast<double>(adc_plan.residual);

                    block.adc = seq.register_adc(row);
                    seq.set_block(b, block);
                }
            }
        }
    }

    std::vector<std::vector<std::pair<int, int>>> label_gate_runs(
        const Sequence& seq,
        const std::vector<std::string>& labels,
        int first,
        int last)
    {
        const size_t n = labels.size();
        std::vector<std::vector<std::pair<int, int>>> runs(n);

        const int blocks = seq.num_blocks();
        const int from = std::max(first, 1);
        const int to = (last <= 0) ? blocks : std::min(last, blocks);
        if (from > to)
            return runs;

        /* A label the sequence never registered cannot appear in any LABELSET
         * row, so it exempts nothing and `find_label_id` reporting 0 is the
         * answer rather than a lookup failure. */
        std::vector<int32_t> ids(n);
        for (size_t i = 0; i < n; ++i)
            ids[i] = static_cast<int32_t>(seq.find_label_id(labels[i]));

        const int32_t labelset = static_cast<int32_t>(seq.find_extension_type_id("LABELSET"));
        const int32_t links = seq.extensions_library().size();
        const int32_t sets = seq.label_set_library().size();
        const int32_t* events = seq.block_events();

        std::vector<char> flagged(n, 0);
        std::vector<int> start(n, -1);

        for (int b = from; b <= to; ++b)
        {
            if (labelset != 0)
            {
                int32_t link = events[static_cast<size_t>(b - 1) * BLOCK_WIDTH + 5];
                while (link > 0 && link <= links)
                {
                    const int32_t* row = seq.extensions_library().row(link);
                    if (row[0] == labelset && row[1] > 0 && row[1] <= sets)
                    {
                        const int32_t* entry = seq.label_set_library().row(row[1]);
                        for (size_t i = 0; i < n; ++i)
                        {
                            if (ids[i] != 0 && entry[1] == ids[i])
                                flagged[i] = entry[0] != 0;
                        }
                    }
                    link = row[2];
                }
            }

            for (size_t i = 0; i < n; ++i)
            {
                if (flagged[i])
                {
                    if (start[i] >= 0)
                    {
                        runs[i].emplace_back(start[i], b - 1);
                        start[i] = -1;
                    }
                }
                else if (start[i] < 0)
                {
                    start[i] = b;
                }
            }
        }

        for (size_t i = 0; i < n; ++i)
        {
            if (start[i] >= 0)
                runs[i].emplace_back(start[i], to);
        }
        return runs;
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
                    row[0] *= factor; // amplitude; the timings are untouched
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

        /*
         * What every block playing these events wants, worked out once.
         *
         * One entry rather than two maps keyed alike: whether the gradients
         * sweep and which shape they produce are the same question asked
         * twice, and the key is the expensive part of asking it.  The key
         * itself is built into a buffer that outlives the iteration, so a
         * scan with half a million readouts does not allocate half a million
         * strings to look up the one answer they share.
         */
        struct Attachment
        {
            /** Do the gradients sweep across the ADC window? */
            bool varies = false;
            bool varies_known = false;
            /** The stored shape, 0 when this readout stores none. */
            int32_t shape = 0;
            bool shape_known = false;
        };
        std::unordered_map<std::string, Attachment> played;
        std::string key;
        key.reserve(160);

        const int32_t* events = seq.block_events();

        for (int b = 1; b <= blocks; ++b)
        {
            const int32_t* row = events + static_cast<size_t>(b - 1) * BLOCK_WIDTH;
            Block block;
            block.rf = row[0];
            block.gx = row[1];
            block.gy = row[2];
            block.gz = row[3];
            block.adc = row[4];
            block.ext = row[5];

            if (block.adc <= 0 || block.adc > seq.adc_library().size())
                continue;
            adc_of[static_cast<size_t>(b)] = block.adc;

            key.clear();
            played_key_into(seq, block, seq.adc_library().row(static_cast<int>(block.adc)), key);
            Attachment& shared = played[key];

            /* A Cartesian, unrotated readout takes its shift as two scalars
             * and its k from the encoding counters -- storing a trajectory
             * for it would be pure weight.  Only the readouts the consumer
             * actually has to finish get one.  The rotation is asked of the
             * block and the sweep of the events, because only the second is
             * shared by everything that plays them. */
            if (!shared.varies_known)
            {
                shared.varies = adc_gradients_vary(seq, block, block_gradients(seq, block));
                shared.varies_known = true;
            }
            if (!shared.varies && !block_is_rotated(seq, block))
                continue;

            if (!shared.shape_known)
            {
                shared.shape_known = true;

                const BaseTrajectory t = base_trajectory(seq, b);
                const double window = static_cast<double>(t.num_samples) * t.dwell;
                if (t.has_adc && window > 0.0)
                {
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
                    shared.shape = static_cast<int32_t>(
                        seq.register_raw_shape(packed.data(), static_cast<int>(packed.size())));
                }
            }

            wanted[static_cast<size_t>(b)] = shared.shape;
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
            seq.set_definition(
                PHASE_MODULATION_MODE_KEY,
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
            seq.adc_library().row(id)[7] = static_cast<double>(agreed[static_cast<size_t>(id)]);
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

} // namespace pulseq
