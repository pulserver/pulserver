/**
 * @file pulseq_ktraj.h
 * @brief The k-space every ADC sample sits at, straight off a parsed .seq.
 *
 * ### What this is for
 *
 * Non-Cartesian recon needs an absolute k for each acquired sample, and it
 * needs to know which sample of each readout is the echo.  Neither is
 * written into a `.seq`: nothing in the format marks the echo sample, and
 * PyPulseq's `calculate_kspace` does not compute an echo position at all.
 *
 * So this derives both, from nothing but the file.
 *
 * ### Why it is not just cumulative-sum-the-waveform
 *
 * A `.seq` stores a gradient as a **normalised shape beside a scalar
 * amplitude**, and a scan replays a handful of distinct shapes a great many
 * times.  Integrating the shape library once -- rather than integrating a
 * materialised waveform per block -- makes the cost scale with the number of
 * distinct gradients instead of with the length of the scan.  That is the
 * whole performance argument, and it is why this exists beside
 * `compute_kspace_trajectory()` in pulseg_structure.c, which integrates a
 * uniform-raster waveform because that is the form the PSD already has.
 *
 * On top of that, blocks repeat: see @ref pulseq_ktraj_opts for the repeat key.
 *
 * ### Precision
 *
 * Every accumulation inside is `double` regardless of `PULSEQ_REAL`.  k is a
 * running integral over an entire scan, so a float32 accumulator drifts in a
 * way that a float32 *result* does not -- the storage type is a memory
 * decision, the accumulator type is a correctness one.
 *
 * ### Frames
 *
 * By default a rotation extension is resolved into the samples, so k comes
 * back in the **physical** frame -- what a reconstruction needs, and what
 * PyPulseq and Pulseq's MATLAB `calculateKspacePP` both return.  Set
 * `apply_rotation` to 0 for the logical frame.  Note that
 * `pulseq::block_k_origins()` in cxx/pulseq/trajectory.cpp deliberately does
 * the opposite and is always logical: the FOV-shift phase `dr . k` is
 * invariant when both are rotated, so that path must never see a resolved
 * rotation.  The two are different contracts, not an inconsistency.
 */

#ifndef PULSEQ_KTRAJ_H
#define PULSEQ_KTRAJ_H

#include "pulseq_config.h"
#include "pulseq_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /* ============================================================== */
    /*  Options                                                       */
    /* ============================================================== */

    /** @brief What to compute, and in which frame. */
    typedef struct pulseq_ktraj_opts
    {
        /**
         * Per-axis gradient timing compensation, seconds.  Shifts the
         * gradient time base only -- not the ADC sample times and not the RF
         * event times, which are intrinsically synchronised with each other.
         * Applying it to the ADC instead is the mistake Pulseq's own
         * `calculateKspacePP` documents having made and reverted.
         */
        PULSEQ_REAL trajectory_delay[3];

        /** Per-axis background gradient, Hz/m, added before integration. */
        PULSEQ_REAL gradient_offset[3];

        /** 1-based inclusive block range; 0 for @c last_block means "to the end". */
        int first_block;
        int last_block;

        /** 1 = resolve rotation extensions into k (physical frame). */
        int apply_rotation;

        /**
         * 1 = give each sample the k averaged over its dwell window, rather
         * than k at the window's midpoint.
         *
         * An ADC sample integrates signal for a whole dwell, so the average is
         * the coordinate it physically belongs to; the midpoint is a
         * first-order stand-in for it.  The two differ by
         * `(dwell^2 / 24) * dg/dt`, which is **exactly zero while the gradient
         * is flat** -- so an ordinary Cartesian readout is unaffected and a
         * ramp-sampled one is not.  At a 2 us dwell and 200 T/m/s the offset
         * is about 0.35% of one sample step.
         *
         * Off by default, and deliberately so: PyPulseq, MRpro and mri-nufft
         * all sample at the midpoint, so leaving this off is what makes our
         * answer comparable with theirs.  Turn it on when the trajectory is
         * going to a gridder rather than to a comparison.
         */
        int sample_window_average;

        /**
         * 1 = derive each readout's `center_sample`.  Costs one extra sweep
         * over the readouts, so a caller that only wants k can decline it.
         */
        int derive_center_sample;

        /**
         * Optional hint: the number of blocks in one TR, if the caller
         * happens to know it.  Purely an allocation hint for the repeat-key
         * memo -- never a correctness input.  TR *detection* is not safe to
         * rely on here (a zero-duration counter block has collapsed a
         * sequence to "one TR" before now), and nothing below depends on it.
         * 0 means "no idea", which is always fine.
         */
        int tr_period_blocks;
    } pulseq_ktraj_opts;

    /** @brief Fill @p opts with the defaults documented above. */
    void pulseq_ktraj_opts_init(pulseq_ktraj_opts *opts);

    /* ============================================================== */
    /*  Results                                                       */
    /* ============================================================== */

    /** @brief One ADC event and where it sits in k-space. */
    typedef struct pulseq_ktraj_readout
    {
        int block_index;       /**< 1-based block carrying the ADC          */
        int num_samples;       /**< ADC samples                             */
        PULSEQ_REAL t0;        /**< time of the first sample centre, s      */
        PULSEQ_REAL dwell;     /**< s                                       */
        PULSEQ_REAL k0[3];     /**< k at the block start, 1/m               */
        PULSEQ_REAL g_echo[3]; /**< gradient at the echo sample, Hz/m       */
        /**
         * Absolute k at the echo sample, 1/m, in the output frame; zero when
         * there is no echo.
         *
         * Carried per readout because it is what an encoding-label analysis
         * actually reads: reducing a scan to one point per readout is the step
         * that makes such an analysis cost O(readouts) instead of O(samples),
         * and asking for it through @ref pulseq_ktraj_readout_k would mean
         * materialising every sample to keep one of them.  It is three doubles
         * the search already had in hand.
         */
        PULSEQ_REAL k_echo[3];
        /**
         * Index of the sample closest to the k-space centre, or -1 when it
         * was not derived.  See @ref pulseq_ktraj_plan_create for the rule.
         *
         * It is an index into *this* readout as acquired, so on a bipolar
         * train the two polarities report different numbers -- 64 and 63 on a
         * 128-sample readout -- because their nearest samples are the same
         * point in k reached from opposite ends.  Mirroring the reverse lines,
         * which is what their REV label asks for, brings both to 64.  A caller
         * that wants one number for the whole scan has to say which frame it
         * means; `pulseq::detect_labels` quotes the mirrored one.
         */
        int center_sample;
        /**
         * Bit i set when axis i's gradient is flat across the ADC window.
         *
         * A flat axis still sweeps k; it just does so at a constant rate, so
         * a consumer may store it as a straight line -- an amplitude and an
         * endpoint -- rather than as samples.
         *
         * **In the frame @c k is reported in.**  With `apply_rotation` set
         * this is the physical answer, which is not the logical one: a radial
         * spoke is a flat gradient on one logical axis plus a per-shot
         * rotation, so it is logically as constant as a Cartesian line while
         * physically sweeping a line at an arbitrary angle.  Handing back the
         * logical answer beside a rotated k would invite the shortcut that a
         * constant gradient needs no trajectory at all -- true for Cartesian,
         * where the recon indexes the line with an encoding counter and a
         * frequency offset, and false for radial, which has no such counter.
         *
         * The compact form for a spoke is still available, and is the one
         * pulseg's trajectory cache uses: a flat *logical* gradient plus
         * @c rotation_id, with the consumer applying the rotation.  Take that
         * route from @c rotation_id and the gradient amplitudes, not from
         * this mask.
         */
        unsigned int constant_axes;
        /**
         * 0-based row of the rotation library this readout's block carries,
         * or -1 for none.
         *
         * Present so a spoke can be stored as a line plus an angle rather
         * than as samples; see @c constant_axes.
         */
        int rotation_id;
    } pulseq_ktraj_readout;

    /** @brief One RF pulse that moves k, and the gradient it sits on. */
    typedef struct pulseq_ktraj_rf
    {
        int block_index;           /**< 1-based                              */
        PULSEQ_REAL t;             /**< pulse centre, s from the range start */
        PULSEQ_REAL freq_offset;   /**< Hz, including the ppm term           */
        PULSEQ_REAL phase_offset;  /**< rad, referenced as PyPulseq does     */
        PULSEQ_REAL g[3];          /**< gradient at the pulse centre, Hz/m   */
    } pulseq_ktraj_rf;

    /** @brief Everything @ref pulseq_ktraj_all materialises. */
    typedef struct pulseq_ktraj
    {
        int num_readouts;
        pulseq_ktraj_readout *readouts;

        /**
         * Absolute k for every sample of every readout, 1/m, concatenated in
         * readout order and **axis-major within a readout**: all of x, then
         * all of y, then all of z, `num_samples` each.  Readout i begins at
         * `sample_offset[i]`, which counts samples, not values -- so its x
         * axis is `k[3 * sample_offset[i] + 0 * num_samples]`.
         *
         * Axis-major rather than interleaved because that is the layout
         * `attach_base_trajectory()` already uses and the one a shape codec
         * compresses; keeping the two the same avoids a transpose at every
         * boundary between them.
         */
        PULSEQ_REAL *k;
        int *sample_offset;
        int total_samples;

        int num_excitations;
        pulseq_ktraj_rf *excitations;
        int num_refocusings;
        pulseq_ktraj_rf *refocusings;

        /** The sample the whole scan comes closest to k=0 at, 1/m. */
        PULSEQ_REAL k_center[3];
        /** Sum of the block durations in range, s. */
        PULSEQ_REAL total_duration;
    } pulseq_ktraj;

    /** @brief Release everything a @ref pulseq_ktraj owns. */
    void pulseq_ktraj_free(pulseq_ktraj *traj);

    /* ============================================================== */
    /*  Incremental interface                                         */
    /* ============================================================== */

    /**
     * @brief The integrated shape library plus per-block k origins.
     *
     * Opaque, and the expensive half of the work: creating one walks every
     * block once and integrates every distinct gradient once.  Reading
     * samples out of it afterwards is cheap and needs no further allocation.
     */
    typedef struct pulseq_ktraj_plan pulseq_ktraj_plan;

    /**
     * @brief Build a plan over @p seq.
     *
     * With `derive_center_sample` set, this also sweeps the readouts twice:
     * once to find the sample of the whole scan closest to the origin, and
     * once to set each readout's `center_sample` to its own sample closest to
     * *that* point.  Referencing the global closest approach rather than a
     * hard zero is what keeps the answer meaningful for a readout that never
     * crosses the origin -- partial Fourier, an off-centre spiral interleaf --
     * where the nearest-to-zero sample would just be an endpoint.  It is the
     * rule Pulseq's `autoLabel.m` uses, for the same reason.
     *
     * Both sweeps memoize on the block's **repeat key**: the event-id tuple
     * (gx, gy, gz, adc, rotation) plus the entry k rounded to a fraction of
     * the sampling step.  Two blocks agreeing on all of it produce the same
     * samples, so the echo search runs once per distinct readout rather than
     * once per TR.  The rotation id has to be in the key: gradient ids alone
     * are shape-and-amplitude only, and two blocks that differ solely by a
     * per-slice rotation would otherwise collide.
     *
     * @return PULSEQ_SUCCESS, or a negative error code.
     */
    int pulseq_ktraj_plan_create(pulseq_ktraj_plan **out, const pulseq_file *seq,
                                 const pulseq_ktraj_opts *opts);

    /** @brief Release a plan. */
    void pulseq_ktraj_plan_free(pulseq_ktraj_plan *plan);

    /** @brief Number of ADC events in the plan's block range. */
    int pulseq_ktraj_num_readouts(const pulseq_ktraj_plan *plan);

    /** @brief Copy readout @p index's descriptor out.  0-based. */
    int pulseq_ktraj_readout_info(const pulseq_ktraj_plan *plan, int index,
                                  pulseq_ktraj_readout *out);

    /**
     * @brief Absolute k for readout @p index, into a caller-supplied buffer.
     *
     * @p out_k must hold `3 * num_samples` values, and is written
     * axis-major -- x, then y, then z.  This is the streaming entry point: a
     * caller walking a million readouts pays for one readout at a time.
     */
    int pulseq_ktraj_readout_k(const pulseq_ktraj_plan *plan, int index,
                               PULSEQ_REAL *out_k);

    /** @brief Excitation count, and descriptor by 0-based index. */
    int pulseq_ktraj_num_excitations(const pulseq_ktraj_plan *plan);
    int pulseq_ktraj_excitation_info(const pulseq_ktraj_plan *plan, int index,
                                     pulseq_ktraj_rf *out);

    /** @brief Refocusing count, and descriptor by 0-based index. */
    int pulseq_ktraj_num_refocusings(const pulseq_ktraj_plan *plan);
    int pulseq_ktraj_refocusing_info(const pulseq_ktraj_plan *plan, int index,
                                     pulseq_ktraj_rf *out);

    /**
     * @brief k at the start of block @p block_index (1-based), 1/m.
     *
     * Always the logical frame -- a rotation belongs to the block that
     * carries it, and an origin is where the previous block left off.
     */
    int pulseq_ktraj_block_origin(const pulseq_ktraj_plan *plan, int block_index,
                                  PULSEQ_REAL out_k[3]);

    /** @brief Sum of block durations over the plan's range, seconds. */
    PULSEQ_REAL pulseq_ktraj_total_duration(const pulseq_ktraj_plan *plan);

    /**
     * @brief Distinct repeat keys among the readouts, or 0 if centres were
     *        not derived.
     *
     * `1 - num_key_groups / num_readouts` is the fraction of readouts the memo
     * answered without a search.  Worth looking at: a scan that ought to
     * repeat and reports one group per readout has a key that is wrong for it,
     * which is a correctness smell rather than merely a slow run.
     */
    int pulseq_ktraj_num_key_groups(const pulseq_ktraj_plan *plan);

    /** @brief The scan's closest approach to k=0, 1/m. */
    int pulseq_ktraj_center(const pulseq_ktraj_plan *plan, PULSEQ_REAL out_k[3]);

    /**
     * @brief Index of the readout that achieves that closest approach, or -1.
     *
     * The scan's reference profile: the one readout guaranteed to pass through
     * the k-space centre, so its sample spacing defines the encoding step and
     * its shape says whether sampling is uniform.  A caller wanting the
     * trajectory of exactly one readout wants this one.
     */
    int pulseq_ktraj_central_readout(const pulseq_ktraj_plan *plan);

    /* ============================================================== */
    /*  Sampling k anywhere                                           */
    /* ============================================================== */

    /**
     * @brief Absolute k at arbitrary times, 1/m, axis-major (`3 * count`).
     *
     * Times are seconds from the start of the plan's block range and need not
     * be sorted or lie on any raster.  A time outside the range clamps to the
     * nearest end, which is what makes a plot's closing sample well defined.
     *
     * This exists so the dense whole-sequence trajectory -- what PyPulseq
     * returns as `k_traj` and what a plot draws -- comes out of the same
     * accumulator as the ADC samples rather than a second one written beside
     * it.  A rotation is resolved per block, exactly as for a readout.
     */
    int pulseq_ktraj_at_times(const pulseq_ktraj_plan *plan, const PULSEQ_REAL *times,
                              int count, PULSEQ_REAL *out_k);

    /**
     * @brief Every time at which the trajectory can change slope.
     *
     * The sorted, de-duplicated union of block boundaries and gradient
     * breakpoints over the plan's range.  Between two consecutive entries k is
     * a single quadratic, so a consumer that samples here and interpolates
     * linearly draws the trajectory with no more error than the drawing
     * itself introduces -- which is why the dense grid is this and not the
     * gradient raster.  A ramp of any length costs two points.
     *
     * Pass NULL for @p out_t to ask only for the count.  Otherwise at most
     * @p max entries are written.
     *
     * @return the number of breakpoints, or a negative error code.
     */
    int pulseq_ktraj_breakpoints(const pulseq_ktraj_plan *plan, PULSEQ_REAL *out_t, int max);

    /* ============================================================== */
    /*  One-shot interface                                            */
    /* ============================================================== */

    /**
     * @brief Plan, materialise every readout, and free the plan.
     *
     * For the offline/host path, where the whole trajectory is wanted at
     * once.  Memory is `3 * total_adc_samples` values, which is bounded by
     * the acquisition rather than by the scan duration -- but on a very long
     * scan it is still large, and that is when the incremental interface
     * above is the right one.
     */
    int pulseq_ktraj_all(const pulseq_file *seq, const pulseq_ktraj_opts *opts,
                         pulseq_ktraj *out);

#ifdef __cplusplus
}
#endif

#endif /* PULSEQ_KTRAJ_H */
