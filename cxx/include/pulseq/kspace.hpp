/**
 * @file kspace.hpp
 * @brief Where every ADC sample of a `pulseq::Sequence` sits in k-space.
 *
 * ### One calculator, not two
 *
 * The arithmetic lives in `csrc/src/pulseq/pulseq_ktraj.c`, in C89, because
 * the interpreter links it directly.  Rather than reimplement it here, this
 * runs the same code: `raw64.cpp` compiles that file a second time in double
 * precision, and a `Sequence` reaches it by being written to Pulseq's binary
 * format in memory and parsed back.  The round trip costs one serialise and
 * one parse -- tens of milliseconds on a large scan.
 *
 * It carries amplitudes as float64 and times as integer picoseconds, so those
 * survive exactly.  **Shape samples do not**: the binary format writes them as
 * float32 (write_binary.cpp), and since a shape is stored run-length encoded
 * as a *derivative*, decompressing it sums those errors -- measured at 3e-8
 * per sample, reaching about 1e-5 in k across a five-hundred-sample readout.
 * Small, but not nothing, and not something to leave to chance in a number
 * whose whole purpose is to be more accurate than the alternatives.  So the
 * shapes are handed across separately, in double, and written back over the
 * parsed ones; see `restore_shapes` in kspace.cpp.  With that, the bridge is
 * exact.
 *
 * The alternative was an in-memory `Sequence` to `raw64::pulseq_file`
 * converter, which is a few hundred lines that would have to stay in lockstep
 * with read.cpp's unit conversions forever, for a saving that does not matter
 * to an offline call.
 *
 * ### What this adds on top of the core
 *
 * The core answers about ADC samples.  Two things only the host wants are
 * built here:
 *
 *  - the **dense trajectory**, k at every gradient breakpoint across the
 *    scan, which is what upstream's `calculate_kspace` returns as `k_traj`
 *    and what a plot needs.  It is sized by the number of breakpoints, not by
 *    the raster, and can still be bounded with a block range.
 *  - **slice positions**, from each excitation's frequency offset divided by
 *    the gradient it sits on.
 */

#ifndef PULSEQ_CXX_KSPACE_HPP
#define PULSEQ_CXX_KSPACE_HPP

#include <array>
#include <cstdint>
#include <vector>

namespace pulseq
{

    class Sequence;

    /** How the trajectory is sampled and in which frame. */
    struct KSpaceOptions
    {
        /** Per-axis gradient timing compensation, seconds. */
        std::array<double, 3> trajectory_delay{{0.0, 0.0, 0.0}};
        /** Per-axis background gradient, Hz/m. */
        std::array<double, 3> gradient_offset{{0.0, 0.0, 0.0}};

        /** 1-based inclusive; 0 for `last_block` means "to the end". */
        int first_block = 1;
        int last_block = 0;

        /** Resolve rotation extensions into k (the physical frame). */
        bool apply_rotation = true;

        /**
         * Give each sample the k averaged over its dwell rather than k at the
         * midpoint.  Off by default -- PyPulseq, MRpro and mri-nufft all use
         * the midpoint, and matching them is what makes the answer
         * comparable.  See `pulseq_ktraj_opts::sample_window_average`.
         */
        bool sample_window_average = false;

        /** Derive each readout's echo sample. */
        bool derive_center_sample = true;

        /**
         * Also build the dense per-breakpoint trajectory.  Off by default
         * because only plotting wants it and it is the one output whose size
         * grows with the scan rather than with the acquisition.
         */
        bool dense = false;

        /**
         * Fill `KSpace::k_adc` and `t_adc`.
         *
         * On by default -- it is what a trajectory is usually wanted for.
         * Turn it off for an analysis that reads one point per readout:
         * `Readout::k_echo` and `KSpace::k_central` are filled either way, and
         * they are the whole input to `auto_label`, which on a 16-million-
         * sample scan is the difference between three doubles per readout and
         * four hundred megabytes.
         */
        bool materialize_samples = true;

        /**
         * Also report k at the start of every block in range.
         *
         * Off by default because it is three doubles per *block*, which on a
         * million-block scan is larger than the acquisition it describes.
         * `pulseq::block_k_origins()` is this with Pulseq's 1-based indexing,
         * and is what `TransformFOV` reads.
         */
        bool block_origins = false;
    };

    /** One ADC event: where it is, and which sample is its echo. */
    struct Readout
    {
        int block_index = 0;  /**< 1-based                        */
        int num_samples = 0;
        double t0 = 0.0;      /**< first sample centre, s         */
        double dwell = 0.0;
        /** Index of the echo, or -1 when the readout has none. */
        int center_sample = -1;
        /** Gradient at the echo, Hz/m; zero when there is no echo. */
        std::array<double, 3> g_echo{{0.0, 0.0, 0.0}};
        /**
         * k at the echo, 1/m; zero when there is no echo.
         *
         * One point per readout, and for an encoding-label analysis the only
         * point that matters -- which is what lets `auto_label` run without
         * `k_adc`.
         */
        std::array<double, 3> k_echo{{0.0, 0.0, 0.0}};
        /** Offset of this readout's first sample within `k_adc`. */
        int sample_offset = 0;

        /**
         * Axes whose k moves at a constant rate across the ADC window, in the
         * frame `k_adc` is in -- so with `apply_rotation` on, the physical
         * one.
         *
         * A constant axis can be stored as a line rather than as samples.
         * The distinction that makes this worth carrying is **radial**: a
         * spoke is a flat gradient plus a per-shot rotation, so it is
         * logically as constant as a Cartesian line while physically sweeping
         * a line at an arbitrary angle.  A consumer must not read "constant
         * gradient" as "needs no trajectory" -- that holds for Cartesian,
         * where the encoding counters and a frequency offset locate the line,
         * and not for a spoke, which has no counter to locate.
         *
         * To store a spoke compactly, use the logical gradient amplitude with
         * @ref rotation_id, which is what pulseg's trajectory cache does.
         */
        std::array<bool, 3> constant_axes{{false, false, false}};

        /**
         * Rotation library row for this readout's block, or -1 for none.
         *
         * With `constant_axes` all set, this is what still distinguishes one
         * readout from another: a radial spoke is a constant gradient whose
         * *direction* the rotation supplies.  See @ref KSpace::rotations_vary.
         */
        int rotation_id = -1;
    };

    /** An RF pulse that moves k. */
    struct RfEventTiming
    {
        int block_index = 0;
        double t = 0.0;
        double freq_offset = 0.0;
        double phase_offset = 0.0;
        std::array<double, 3> g{{0.0, 0.0, 0.0}};
    };

    /** Everything @ref calculate_kspace produces. */
    struct KSpace
    {
        std::vector<Readout> readouts;

        /**
         * Absolute k for every acquired sample, 1/m.
         *
         * Axis-major within a readout -- all of x, then y, then z -- matching
         * what `attach_base_trajectory` stores and what the shape codec
         * compresses well.  MRD wants the transpose; see @ref
         * interleave_readout.
         */
        std::vector<double> k_adc;
        int total_samples = 0;

        /** Sample times, one per entry of a readout, seconds. */
        std::vector<double> t_adc;

        std::vector<RfEventTiming> excitations;
        std::vector<RfEventTiming> refocusings;

        /** Empty unless `KSpaceOptions::dense`. */
        std::vector<double> t_ktraj;
        /** `3 * t_ktraj.size()`, axis-major.  Empty unless `dense`. */
        std::vector<double> k_ktraj;

        /**
         * Slice position per excitation, metres, and when it was played.
         *
         * Undefined entries -- an excitation with no gradient on an axis --
         * come back as 0, which is what Pulseq's own `calculateKspacePP`
         * does.
         */
        std::vector<std::array<double, 3>> slice_pos;
        std::vector<double> t_slice_pos;

        /** The scan's closest approach to k = 0, 1/m. */
        std::array<double, 3> k_center{{0.0, 0.0, 0.0}};

        /**
         * Index into `readouts` of the readout achieving `k_center`, or -1.
         *
         * The scan's reference profile: the one readout that provably passes
         * through the centre, so its sample spacing is the encoding step and
         * its curvature is what says whether sampling is uniform.
         */
        int central_readout = -1;

        /**
         * Whether the readouts do not all carry the same rotation.
         *
         * The discriminator that separates a Cartesian scan from a radial
         * one, and the reason it is not enough to look at whether a gradient
         * is flat.
         *
         * A constant gradient during the ADC normally means the readout is a
         * straight line the reconstruction can place from an encoding counter
         * and a frequency offset, with no trajectory stored at all.  **A
         * radial spoke is also a constant gradient** -- rotating a constant
         * gradient leaves it constant -- and there is no counter that places
         * it, because what changes between shots is the direction, carried in
         * the rotation.  So a scan is representable that way only when every
         * axis is constant *and* this is false; one rotation shared by the
         * whole scan is just a fixed change of frame and can be folded in
         * once, which is why the test is variation rather than presence.
         *
         * When it is true, a trajectory (or the equivalent phase modulation)
         * has to be produced even though every gradient is flat.
         */
        bool rotations_vary = false;

        /**
         * That readout's samples, axis-major, `3 * num_samples`.
         *
         * Always filled, even with `materialize_samples` off -- it is one
         * readout, and every analysis of the sampling pattern needs exactly
         * this one.
         */
        std::vector<double> k_central;

        /**
         * k at the start of each block in range, 1/m, **always logical** --
         * a rotation belongs to the block that carries it, and an origin is
         * where the previous block left off.  Indexed from the first block of
         * the range; empty unless `KSpaceOptions::block_origins`.
         *
         * `pulseq::block_k_origins()` is this re-indexed from 1, and is what
         * `TransformFOV` builds its phase on.  It used to be a second
         * implementation; the two were shown equal to 4e-16 over every fixture
         * in the repository and then there was one.
         */
        std::vector<std::array<double, 3>> block_origins;

        double total_duration = 0.0;

        /**
         * Distinct repeat keys among the readouts.
         *
         * `1 - key_groups / readouts.size()` is how much of the echo search
         * the memo answered without looking.  Exposed because a scan that
         * ought to repeat and does not is a wrong key rather than a slow run.
         */
        int key_groups = 0;
    };

    /**
     * The k-space trajectory of @p seq.
     *
     * Takes the sequence by non-const reference because the binary writer
     * records `TotalDuration` in `[DEFINITIONS]` before serialising, exactly
     * as `write_binary` does.
     */
    KSpace calculate_kspace(Sequence& seq, const KSpaceOptions& options = {});

    /* ================================================================== */
    /*  MRD                                                               */
    /* ================================================================== */

    /**
     * Which axes any readout of @p ks actually moves along.
     *
     * Decided over **every** readout, not one: a readout whose rotation
     * leaves an axis silent would otherwise pack one axis fewer and shift
     * every sample after it.  This is the same rule
     * `trajectory_cache_reader.h` states for `PrecomputedTrajectory::
     * axis_active`, and it exists so the two paths pack identically.
     */
    std::array<bool, 3> active_axes(const KSpace& ks, double tolerance = 1e-6);

    /**
     * Copy one readout out in MRD's layout: `ndim * num_samples` **interleaved**
     * values, with inactive axes dropped.
     *
     * The core stores axis-major because that is what compresses; ISMRMRD's
     * `Acquisition` trajectory field is interleaved. The transpose lives here
     * rather than in the core so the `phase_modulation` path, which wants
     * axis-major, does not have to undo it.
     *
     * @return the number of values written, or 0 when no axis is active.
     */
    int interleave_readout(const KSpace& ks, const std::array<bool, 3>& axes, int readout,
                           float* out);

}  // namespace pulseq

#endif /* PULSEQ_CXX_KSPACE_HPP */
