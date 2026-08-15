/**
 * @file trajectory.hpp
 * @brief The k-space a block's ADC window traverses, with the amplitude left
 *        outside it.
 *
 * ### Why the amplitude stays out
 *
 * A gradient is held here the way the file holds it: a normalised waveform and
 * one scalar (see events.hpp).  Integrating the *normalised* waveform gives a
 * trajectory that is the same object for every instance of that gradient --
 * a phase-encode step, a spiral arm at a different amplitude, a variable-flip
 * train -- so the library that stores it deduplicates to one row per distinct
 * shape rather than one row per readout.  Multiplying the amplitude in first
 * would produce a numerically distinct array per instance, which is exactly
 * the explosion this exists to avoid.
 *
 * The consumer reconstitutes k for an instance as
 *
 *     k_axis(t) = amplitude_axis * base_axis(t)          [Hz*s/m == 1/m]
 *
 * and then applies any rotation.  Because `dr . k` is invariant under a
 * rotation applied to both, a shift expressed in the *logical* frame needs no
 * knowledge of the rotation at all.
 *
 * ### What this does not know
 *
 * `base` covers the ADC window of one block.  Absolute k also carries whatever
 * moment accumulated since the excitation reset, which lives in *earlier*
 * blocks -- typically a prewinder whose amplitude varies per line.  That DC is
 * per-instance and is not recoverable from a single block, so it is not
 * invented here.  `AxisTrajectory::moment` gives the normalised whole-block
 * integral, so a caller that walks a TR can accumulate the DC itself as
 * `sum(amplitude_e * moment_e)`; a caller that would rather read it off the
 * encoding counters can do that instead.
 *
 * For the same reason there is no `center_sample`: the sample at which k
 * crosses zero is a property of the absolute trajectory, not of the base, and
 * for a segment that never passes through the origin it does not exist.  It is
 * a discovered property of the composed result, never an imposed one.
 */

#ifndef PULSEQ_CXX_TRAJECTORY_HPP
#define PULSEQ_CXX_TRAJECTORY_HPP

#include <array>
#include <vector>

#include "pulseq/sequence.hpp"

namespace pulseq
{

    /** One gradient axis over one block's ADC window. */
    struct AxisTrajectory
    {
        /**
         * Normalised cumulative moment at the ADC sample centres, in seconds.
         * Multiply by `amplitude` for k in 1/m.  Empty when the block has no
         * gradient on this axis.
         */
        std::vector<double> base;

        /** The event's amplitude scalar, Hz/m. */
        double amplitude = 0.0;

        /**
         * Normalised integral of the whole block's waveform, in seconds --
         * what this block contributes to the DC of every later block.
         */
        double moment = 0.0;

        /** Whether the block drives this axis at all. */
        bool present = false;

        /**
         * Whether the waveform is flat across the ADC window.  A flat axis
         * carries no shape information, so a consumer stores no base for it
         * and takes its k from the encoding counters instead -- the `-1`
         * trivial-shot case in the cache's trajectory table.
         *
         * @warning This is the **logical** frame, like everything else here,
         * and flat is therefore not on its own a licence to drop the
         * trajectory.  A radial spoke is a flat gradient plus a per-shot
         * rotation extension: logically as constant as a Cartesian line,
         * physically a line at an arbitrary angle, and with no encoding
         * counter that locates it.  A consumer taking the trivial-shot route
         * must carry the block's rotation with the amplitude and apply it.
         * `pulseq_ktraj_readout::constant_axes` answers the same question in
         * the frame its k is reported in, and says physically-not-constant
         * for the same spoke.
         */
        bool constant = true;
    };

    /** A block's ADC window, per axis. */
    struct BaseTrajectory
    {
        /** False when the block has no ADC; every axis is then left empty. */
        bool has_adc = false;

        int num_samples = 0;
        /** Seconds. */
        double dwell = 0.0;
        /** ADC delay from the start of the block, seconds. */
        double delay = 0.0;

        /** Indexed by Axis (X, Y, Z). */
        AxisTrajectory axis[3];
    };

    /**
     * The base trajectory of block @p block_index (1-based).
     *
     * Throws std::out_of_range if the index is not a block of @p seq.  A block
     * without an ADC returns `has_adc == false` rather than throwing: walking a
     * sequence and asking every block is the expected use.
     */
    BaseTrajectory base_trajectory(const Sequence& seq, int block_index);

    /**
     * The normalised integral of one gradient event over its whole extent, in
     * seconds -- `AxisTrajectory::moment` without building an ADC window.
     *
     * For accumulating the DC across the blocks of a TR, where most blocks
     * have no ADC and only their total moment matters.  @p grad_id is a
     * gradient id as a block carries it; 0 returns 0.
     */
    double gradient_moment(const Sequence& seq, int32_t grad_id);

    /* ================================================================== */
    /*  Absolute k                                                        */
    /* ================================================================== */

    /**
     * Where k stands at the start of every block, per axis, in 1/m.
     *
     * Indexed by block number, so `[b]` belongs to block `b` and `[0]` is
     * unused; the returned vector is `num_blocks() + 1` long.
     *
     * This is the DC that `base_trajectory` deliberately does not invent.  A
     * readout's absolute trajectory is the origin of its block plus the base
     * scaled by the amplitude, which is what @ref absolute_trajectory does.
     *
     * ### What moves k
     *
     * Gradients accumulate it, and RF resets or reverses it:
     *
     * - an **excitation** (`use` "excitation") sets k to zero
     * - a **refocusing** pulse negates it -- this is what makes a spin echo
     *   walk back towards the origin, and leaving it out would put every
     *   echo of a train in the wrong half of k-space
     * - every other use, inversion and saturation included, leaves it alone
     * - a pulse that declares **no** use raises, because guessing between
     *   those is not a small error: read an excitation as neutral and k never
     *   restarts, so it accumulates across the whole scan
     *
     * The reset and the flip happen at the pulse's centre (`delay + center`),
     * not at the block boundary, so a block whose gradients straddle its RF --
     * a refocusing pulse between two crushers, the ordinary case -- is split
     * there and each side accumulates on the correct side of the sign flip.
     *
     * ### The limit
     *
     * A rotation extension is not applied: origins come back in the logical
     * frame, matching `base_trajectory`.  Composing rotation is the consumer's
     * job precisely because `dr . k` is invariant when both are rotated, which
     * is what lets a logical-frame shift ignore rotation entirely.
     *
     * ### Where the arithmetic happens
     *
     * In `csrc/src/pulseq/pulseq_ktraj.c`, which is the only place it happens.
     * This used to be a second implementation -- a `Piecewise` per block,
     * written before the C89 core existed -- and the two agreed over every
     * fixture in the repository to 4e-16 relative, which is what made it safe
     * to delete one of them.  What survives is this name, because
     * `TransformFOV` is built on it and "k at the start of each block" is the
     * right thing for it to ask for.
     *
     * @note Non-const because reaching the core means serialising the
     * sequence, and `write_binary` stamps `TotalDuration` on the way past.
     */
    std::vector<std::array<double, 3>> block_k_origins(Sequence& seq);

    /**
     * A readout's absolute k, per axis, in 1/m -- origin plus amplitude times
     * base.
     *
     * @p origin is that block's entry from @ref block_k_origins.  Axes the
     * block does not drive come back empty; an axis that is flat across the
     * window still gets its samples, since a constant gradient still sweeps k.
     *
     * Returns three empty vectors for a block with no ADC.
     */
    std::array<std::vector<double>, 3> absolute_trajectory(
        const Sequence& seq, int block_index, const std::array<double, 3>& origin);

    /* ================================================================== */
    /*  FOV shift                                                         */
    /* ================================================================== */

    /** What @ref apply_fov_shift is allowed to touch. */
    enum class FovShiftScope
    {
        /**
         * RF only -- every readout's ADC side is left to a consumer that
         * will apply it from the base trajectory.
         */
        RfOnly,
        /**
         * RF and ADC both, so the file needs no help from anything
         * downstream.  Native mode, and what a `.seq` shared with another
         * toolbox has to be.
         */
        RfAndAdc,
        /**
         * Server mode: RF always; the ADC side is baked as two scalars for
         * readouts where that is exact -- constant gradient across the
         * window, no ROTATIONS extension -- and left to the consumer for the
         * rest, which keep the base-trajectory path (the consumer applies
         * the centering phase modulation *and* wants the trajectory for its
         * metadata).  @ref attach_base_trajectory draws the same line, so a
         * fully Cartesian sequence in server mode equals its native twin.
         */
        Server
    };

    /**
     * Shift the field of view by @p shift_m, expressed in **logical**
     * coordinates (metres).
     *
     * ### Why logical, and why that makes rotation vanish
     *
     * The phase a shift imposes is `dr . k`.  Rotate both and the dot product
     * is unchanged, so a shift written in the same frame as the gradients
     * needs no knowledge of any rotation -- FOV rotation, a rotation
     * extension, PMC -- at all.  Expressing the shift in the physical frame is
     * what forces an interpreter to undo rotations before it can compute a
     * modulation, and that undoing is the entire complexity this replaces.
     *
     * The caller converts a prescribed physical offset once, with
     * `dr_logical = R^T dr_physical`.
     *
     * ### What it writes
     *
     * `k` is in 1/m and the shift in m, so their dot product is in **cycles**
     * with no factor of 2*pi anywhere -- which is also why the RF phase shape,
     * which the file stores in turns, takes it unscaled.
     *
     * - **RF**: always applied -- there is no downstream consumer that could
     *   do it instead.  Under a gradient that is constant across the pulse
     *   (the slice-select flat top, i.e. nearly every pulse) the shift is a
     *   scalar frequency and phase offset on the RF row itself: no shape is
     *   registered, and the pair is independent of the block's k origin, so
     *   repeated (pulse, gradient) contexts dedup onto one row.  Otherwise
     *   the phase shape gains `dr . k(t)`, referenced to the pulse's own
     *   centre so its effective phase is untouched.
     * - **ADC** (`RfAndAdc`, and `Server` where exact): the shift is split
     *   the way Pulseq splits it -- a frequency offset, a phase offset, and
     *   whatever residual is left over into `phase_modulation`.  Under a
     *   constant gradient the residual is identically zero and a Cartesian
     *   readout costs two scalars, which is what keeps native mode compact.
     *   `Server` bakes exactly the readouts for which no residual is
     *   possible and no ROTATIONS extension applies, and defers the rest to
     *   the consumer of the base trajectory.
     *
     * Blocks flagged `NOROT` are not handled here; a rotation-exempt block
     * needs its gradients counter-rotated at design time, which is a
     * transform on the block, not on the shift.
     *
     * ### The block range
     *
     * @p first and @p last are 1-based and inclusive, and bound only which
     * blocks are *modified*; `0` for @p last means "to the end".  Absolute k
     * is still accumulated from block 1 regardless, because where k stands at
     * the start of the range is a fact about everything played before it.  A
     * caller shifting part of a scan -- the label-gated sub-ranges a
     * `NOPOS` flag carves out, a module with its own prescription -- gets the
     * same phase for those blocks as it would have from shifting the whole
     * thing, which is the only way a partial shift is meaningful.
     */
    void apply_fov_shift(Sequence& seq, const std::array<double, 3>& shift_m,
                         FovShiftScope scope, int first = 1, int last = 0);

    /* ================================================================== */
    /*  FOV scale                                                         */
    /* ================================================================== */

    /**
     * Scale the gradients of blocks @p first..@p last (1-based, inclusive;
     * `0` for @p last means "to the end") by @p scale, per logical axis.
     *
     * The vector scales the *gradient*, so the field of view along an axis is
     * divided by it -- Pulseq's convention, and `mr.scaleGrad`'s.  A factor of
     * zero flattens that axis, which is how a prescription disables an
     * encoding direction.
     *
     * Cheap by construction: a gradient is stored as a normalised shape beside
     * a scalar amplitude (see the file comment above), so this multiplies the
     * amplitude and leaves the waveform alone.  No shape is registered, no
     * samples are touched, and two gradients that differed only in amplitude
     * before still share their shape afterwards.
     *
     * Unlike the shift this needs no notion of absolute k, so blocks outside
     * the range are not read at all.
     */
    void apply_fov_scale(Sequence& seq, const std::array<double, 3>& scale,
                         int first = 1, int last = 0);

    /* ================================================================== */
    /*  Carrying the base trajectory in a .seq                            */
    /* ================================================================== */

    /**
     * The `[DEFINITIONS]` entry that says an ADC's `phase_modulation` holds a
     * base trajectory rather than a phase.
     *
     * The overload is deliberate -- it is the only field of the right shape,
     * and a consumer of ours is the only thing that reads it -- but a file
     * that carries it has to *say* so.  Without the marker a server-mode
     * `.seq` is indistinguishable from a native one, and handing it to a
     * toolbox that applies `phase_modulation` literally produces a plausible
     * wrong image rather than an error.  Unknown definitions are ignorable, so
     * the marker costs nothing to anyone who does not care.
     */
    inline constexpr const char* PHASE_MODULATION_MODE_KEY = "PhaseModulationMode";
    /** The value @ref attach_base_trajectory writes. */
    inline constexpr const char* PHASE_MODULATION_MODE_BASE_TRAJECTORY = "base_trajectory";

    /**
     * Store the base trajectory of each readout that needs one in its ADC's
     * `phase_modulation`, and stamp the sequence as carrying them.
     *
     * ### Which readouts get one
     *
     * The ones a consumer cannot finish from scalars and encoding counters:
     * a block carrying a ROTATIONS extension, or a gradient that varies
     * across the ADC window.  A Cartesian, unrotated readout stores nothing
     * -- its shift is two scalars on the ADC row and its k comes from the
     * labels -- and a sequence in which *no* readout stores a trajectory is
     * not stamped at all.  The line is drawn by the same predicate
     * `FovShiftScope::Server` bakes by, so the two cannot disagree.
     *
     * ### The layout
     *
     * Three axes, always, laid out axis-major: all of x, then all of y, then
     * all of z, `num_samples` each.  Always three because then nothing has to
     * be inferred from a length -- an axis that is flat across the window is
     * stored flat, and a consumer reads "this axis needs its DC from the
     * encoding counters" off the samples instead of off a convention that
     * cannot distinguish (x, y) from (x, z).  Axis-major rather than
     * interleaved because the shape codec run-length encodes a *derivative*:
     * a flat axis then costs three numbers, where interleaving three axes
     * would make a sequence that alternates every sample and compresses to
     * nothing.  That compression is the entire reason storing three axes is
     * affordable.
     *
     * ### The scaling
     *
     * Samples are the normalised moment divided by the ADC window duration
     * (`num_samples * dwell`), which puts them at order 1.  Not cosmetic:
     * `compress_shape` quantises onto an *absolute* 1e-7 grid, so leaving the
     * base in seconds -- order 1e-4, stepping by order 1e-5 -- would leave
     * about a hundred levels across a readout.  Both factors are already in
     * the ADC row, so a consumer rescales with what it has:
     *
     *     k_axis(t) = amplitude_axis * num_samples * dwell * stored_axis(t)
     *
     * ### What it does to the libraries
     *
     * A phase-modulation shape lives in the ADC row, so two readouts that
     * share an ADC definition but traverse different k-space now need
     * different rows.  This registers them unconditionally and leaves the
     * collapsing to `remove_duplicates`, which is this library's usual
     * bargain -- the surviving count is one row per distinct base trajectory,
     * which is the number that was always going to be needed.
     *
     * Idempotent in effect but not free: calling it twice registers a second
     * set of identical shapes for deduplication to remove.
     */
    void attach_base_trajectory(Sequence& seq);

    /** Whether @p seq carries base trajectories, by the marker above. */
    bool has_base_trajectory(const Sequence& seq);

    /**
     * Recover what @ref attach_base_trajectory stored for block @p block_index.
     *
     * The inverse of the layout and scaling documented above, so `base` comes
     * back in seconds -- the same units `base_trajectory` produces, which is
     * what makes the two comparable.  `constant` is *derived* from the stored
     * samples rather than carried, which is the point of storing all three
     * axes.
     *
     * Returns `has_adc == false` for a block with no ADC or no stored
     * modulation.
     */
    BaseTrajectory read_base_trajectory(const Sequence& seq, int block_index);

}  // namespace pulseq

#endif /* PULSEQ_CXX_TRAJECTORY_HPP */
