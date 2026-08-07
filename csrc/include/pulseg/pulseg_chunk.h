/**
 * @file pulseg_chunk.h
 * @brief Splitting a scan into chunks of gradient waveforms that can be
 *        materialised while the previous chunk plays.
 *
 * ### The problem
 *
 * A rotation event rotates the trajectory inside the logical frame, so it is a
 * property of the waveform rather than of the prescription.  The `.seq` and
 * the `.pge` cache both keep it as an event; whoever loads the cache
 * materialises it.  For an interpreter that means the waveform a block plays
 * is per instance, and a scan can hold far more distinct waveforms than the
 * sequencer has memory for -- a radial acquisition with a thousand spokes is a
 * thousand rotations of one shape.
 *
 * The answer is not to hold them all.  It is to hold two chunks' worth: play
 * chunk k out of one buffer while chunk k+1 is built into the other.  This
 * header plans that split.
 *
 * ### Most scans should not stream at all
 *
 * The usual sequence is a handful of base waveforms replayed at different
 * amplitudes and rotations.  Every distinct waveform then fits in waveform
 * memory at once, and the right answer is to put them there at pulsegen and
 * never touch `movewaveimm` again during the scan -- the playout loop is left
 * issuing nothing but `setwave`/`setiamp`.  That is @ref PULSEG_WAVE_RESIDENT,
 * and the planner reports it whenever it holds.
 *
 * Streaming is for what does not fit: individually optimised trajectories,
 * where the waveform library is measured in gigabytes.  Only then does
 * anything have to be loaded mid-scan.
 *
 * ### Streaming works one segment ahead
 *
 * A segment is the playout unit -- the interpreter points the hardware at its
 * waveforms and issues one `startseq` -- so waveforms cannot be swapped inside
 * one, and a segment is the natural granularity for the work.  While segment N
 * plays, the interpreter materialises segment N+1 (loading shapes, applying
 * rotations, `movewaveimm` into the free half of the buffer) and issues
 * segment N's own instruction-set commands.
 *
 * That fixes the time budget at **one segment's playout**, which is the real
 * constraint: a short segment leaves little room, and rotation materialisation
 * in particular is not cheap.  Looking further ahead would not help -- the work
 * for segment N+1 still has to be done inside segment N -- so the plan does not
 * try, and the check is simply that each segment can be built inside its
 * predecessor.
 *
 * When it cannot, no arrangement rescues it, and @ref pulseg_plan_chunks says
 * so at download rather than letting the scanner discover it as an EOS.  That
 * is the whole reason this is planned on the host.
 */

#ifndef PULSEG_CHUNK_H
#define PULSEG_CHUNK_H

#include "pulseg_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief What makes two gradient waveforms the same waveform.
     *
     * Not just the shape and the rotation.  A rotated axis is a combination of
     * all three,
     *
     *     gx' = R00*ax*wx + R01*ay*wy + R02*az*wz
     *
     * so the amplitudes only leave the waveform when the three axes share one
     * shape.  In general the result depends on their *ratios* too, and the key
     * has to carry them: two instances match when they play the same three
     * shapes under the same rotation with the same ratios, at which point they
     * differ by one scalar and `setiamp` covers the difference.
     *
     * Ratios are normalised by the largest magnitude of the triple.  The
     * quantised copy decides identity -- instances agreeing to within the
     * tolerance share a waveform rather than each minting their own -- while
     * the unquantised one is what gets combined.  Both are kept because
     * recovering the second from the first would silently depend on the
     * tolerance the planner happened to run with.
     */
    typedef struct pulseg_wave_key
    {
        int shape_id[3];  /**< per axis; 0 = trapezoid or no gradient      */
        int grad_def[3];  /**< per axis definition id; -1 = no gradient    */
        int rotation_id;  /**< -1 when the block carries no rotation       */
        int ratio_q[3];   /**< quantised ratios; identity only            */
        float ratio[3];   /**< the ratios themselves, for materialising    */
        int num_points;   /**< corner points per axis, an upper bound     */
        int num_axes;     /**< how many axes it actually materialises      */
    } pulseg_wave_key;

    /** How the waveforms reach the hardware. */
    typedef enum
    {
        /** Every distinct waveform fits at once: load them all at pulsegen
         *  and upload nothing during the scan. */
        PULSEG_WAVE_RESIDENT = 0,
        /** They do not fit: materialise each segment while its predecessor
         *  plays, into the free half of a two-segment buffer. */
        PULSEG_WAVE_STREAMED = 1
    } pulseg_wave_mode;

    /** One chunk: a segment instance, and the waves it needs resident. */
    typedef struct pulseg_chunk
    {
        int first_position; /**< exec-stream index of the first block      */
        int num_positions;
        int first_wave;     /**< index into pulseg_chunk_plan::chunk_waves */
        int num_waves;
        long wave_samples;  /**< total samples to materialise              */
        float playout_us;   /**< how long this chunk plays                 */
        float build_us;     /**< estimated cost to materialise it          */
    } pulseg_chunk;

    /**
     * @brief What the interpreter can afford.
     *
     * `build_us_per_sample` is the interpreter's measured materialisation and
     * upload rate.  It belongs to the caller because it is a property of the
     * hardware, not of the sequence, and a wrong value here is the difference
     * between a plan that keeps up and one that reports EOS on the scanner.
     */
    typedef struct pulseg_chunk_budget
    {
        long max_wave_samples;     /**< per chunk; the buffer holds two    */
        float build_us_per_sample; /**< materialise + upload, per sample   */
        float headroom;            /**< fraction of playout time the build
                                        may occupy; 0.5 leaves half spare  */
        float ratio_tolerance;     /**< amplitude-ratio quantisation step  */
    } pulseg_chunk_budget;

/** Defaults; `max_wave_samples` has none, the interpreter must supply it. */
#define PULSEG_CHUNK_BUDGET_INIT {0, 0.0f, 0.5f, 1.0e-4f}

    /** A whole scan's plan. */
    typedef struct pulseg_chunk_plan
    {
        /** RESIDENT or STREAMED; see pulseg_wave_mode.  In RESIDENT mode the
         *  chunk list is a single entry covering the scan and exists only so
         *  consumers have one shape to read. */
        pulseg_wave_mode mode;

        int num_chunks;
        pulseg_chunk *chunks;

        /** Every distinct waveform in the scan, once.  A chunk names the ones
         *  it needs by index (see chunk_waves) rather than holding copies, so
         *  a wave two segments share costs one entry. */
        int num_waves;
        pulseg_wave_key *waves;

        /** Per-chunk index lists into `waves`, concatenated; a chunk's slice
         *  is [first_wave, first_wave + num_waves). */
        int *chunk_waves;
        int num_chunk_waves;

        /** Points across every distinct wave -- what RESIDENT mode needs to
         *  hold, and what deciding the mode compares against the budget. */
        long total_wave_points;

        long peak_wave_samples; /**< largest chunk; sizes the ping-pong  */

        /**
         * Points in the longest single wave.
         *
         * A hardware waveform slot is reserved at a fixed size, so the pool
         * has to be cut to the largest thing that will ever be swapped into
         * it.  `peak_wave_samples` is a whole chunk and answers a different
         * question -- how much memory the pool needs in total.
         */
        int max_wave_points;

        /**
         * Which wave each exec-stream position plays, indexed by position;
         * `position_count` long.  -1 where the block drives no gradient.
         *
         * Precomputed rather than derived at playout: recovering it means
         * rebuilding the key and searching the chunk, and the scan loop is
         * the one place in this system with no time to spare.
         */
        int position_count;
        int *position_wave;
    } pulseg_chunk_plan;

/* clang-format off */
#define PULSEG_CHUNK_PLAN_INIT \
    {PULSEG_WAVE_RESIDENT, 0, NULL, 0, NULL, NULL, 0, 0, 0, 0, 0, NULL}
    /* clang-format on */

    /**
     * @brief Plan how subsequence @p subseq_idx gets its waveforms to hardware.
     *
     * One pass collects every distinct waveform.  If they all fit
     * `max_wave_samples` the plan is RESIDENT and there is nothing more to
     * decide -- nothing is uploaded mid-scan, so no rate can be missed.
     *
     * Otherwise the plan is STREAMED: one chunk per segment instance, each
     * naming the waves it needs, and each checked against its predecessor's
     * playout times `headroom`.  Chunk 0 is exempt, being materialised before
     * the scan starts.
     *
     * @return PULSEG_SUCCESS, or PULSEG_ERR_CHUNK_INFEASIBLE when a segment
     *         cannot be built inside the one before it.  @p diag names the
     *         chunk and both numbers.  Nothing rescues that case: the work for
     *         segment N+1 has to happen during segment N whatever the plan
     *         says, so it is reported rather than worked around.
     *
     * The caller frees the plan with @ref pulseg_free_chunk_plan.
     */
    int pulseg_plan_chunks(
        const pulseg_collection *coll,
        int subseq_idx,
        const pulseg_chunk_budget *budget,
        pulseg_chunk_plan *out,
        pulseg_diagnostic *diag);

    /** Release a plan.  Safe on a zeroed or already-freed plan. */
    void pulseg_free_chunk_plan(pulseg_chunk_plan *plan);

    /* ================================================================== */
    /*  Materialisation                                                   */
    /* ================================================================== */

    /**
     * @brief Build the waveform one wave key plays on one physical axis.
     *
     * ### What comes out
     *
     * The waveform **normalised to unit peak**, plus that peak in `*out_peak`
     * (Hz/m).  That split is what the hardware wants -- a normalised wave and
     * a scalar amplitude -- and it is also what lets one materialised wave
     * serve every instance sharing the key: they differ by the scalar alone.
     *
     * ### The common grid, and why there are two of them
     *
     * Where the axes agree on an *edge-aligned* description -- trapezoids, and
     * arbitrary gradients carrying an explicit time shape -- the output lives
     * on the union of their corner times.  Each axis is piecewise linear, so
     * evaluating all of them at the union of their breakpoints reproduces
     * every one exactly and the rotated combination is piecewise linear on
     * that same grid.  Nothing is interpolated, which is both exact and cheap:
     * a rotated trapezoid stays four points wide however long it is.  This is
     * what Pulseq's `rotate3D` does.
     *
     * One axis breaks that.  A *uniformly sampled* arbitrary gradient is
     * stored as samples at raster CENTRES and nothing else -- the hardware
     * object carries no explicit first or last sample, because those sit on
     * the edges -- so any combination involving one can only be expressed on
     * that centre grid.  As soon as a single involved axis is of that kind,
     * every axis is evaluated at raster centres instead, spanning from the
     * earliest start to the latest end.
     *
     * Whoever builds the hardware slot must make the same choice, or the RSP
     * swaps in a waveform sampled differently from the one the slot was built
     * for.
     *
     * The result comes back as times *and* amplitudes either way.
     *
     * ### Rotation
     *
     * Applied when the key carries one, as `g_out = sum_d R[out_axis][d] *
     * ratio[d] * w_d(t)`.  This is why the ratios are part of the key: the
     * amplitudes only leave the combination when the three axes share a
     * shape.
     *
     * ### Every axis needs a slot
     *
     * A rotated block puts a non-zero component on axes it never drove, so a
     * caller that reserves hardware waveform slots must reserve all three
     * before it can upload any of them.  `pulseg_wave_key::num_axes` is 3
     * exactly when that is the case.
     *
     * @param out_time_us  breakpoint times, us from the block start; may be
     *                     NULL together with @p out_amp to ask only for the
     *                     point count.
     * @param out_amp      normalised amplitudes at those times.
     * @return PULSEG_SUCCESS, PULSEG_ERR_INDEX when the buffers are too small
     *         (with `*out_num_points` set to what is required), or an error.
     */
    int pulseg_materialize_wave(
        const pulseg_collection *coll,
        int subseq_idx,
        const pulseg_wave_key *key,
        int out_axis,
        float *out_time_us,
        float *out_amp,
        int max_points,
        int *out_num_points,
        float *out_peak);

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_CHUNK_H */
