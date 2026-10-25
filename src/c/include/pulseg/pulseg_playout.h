/**
 * @file pulseg_playout.h
 * @brief The two stages of a segmented playout, driven over a backend.
 *
 * A playout prepares every segment's events once, before the scan, and then
 * plays the scan one segment instance at a time, setting only the registers
 * of each block.  pulseg_playout_prepare() is the first stage and
 * pulseg_playout_scan() the second.  Both call the backend a playout
 * provides, in an order and with values that are the library's, so every
 * playout built on them plays a collection alike; what each call does to the
 * hardware is the backend's.
 */

#ifndef PULSEG_PLAYOUT_H
#define PULSEG_PLAYOUT_H

#include "pulseg_config.h"
#include "pulseg_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief One segment instance the scan loop plays.
     */
    /**
     * @brief What play_instance() returns to ask for the instance again.
     *
     * Positive, so the walk does not read it as a failure, and distinct from
     * PULSEG_SUCCESS, which is itself 1: a backend that returns success must
     * not be asking for anything.
     */
#define PULSEG_PLAYOUT_REPLAY 2

    typedef struct pulseg_playout_segment
    {
        int subsequence;    /**< subsequence it plays in                   */
        int segment;        /**< global segment index                      */
        int instance;       /**< instances of the segment played before it
                                 in this stage                             */
        int first_position; /**< execution-stream position of its first
                                 block                                     */
        int num_blocks;     /**< its blocks                                */
        int rotate;         /**< 1 where the prescription rotation turns its
                                 gradients, 0 where its blocks carry NOROT;
                                 the segment's own flags say only that one
                                 of its instances does                     */
        int await_trigger;  /**< 1 where one of its blocks waits for a
                                 physiological trigger input; the segment's
                                 own trigger says only that one of its
                                 instances does                            */
        int slot;           /**< the slot of its positions' rings it plays,
                                 @c instance modulo the plan's slots; -1
                                 unless the waves are streamed             */
    } pulseg_playout_segment;

    /**
     * @brief One block of a segment instance, as the scan loop sets it.
     */
    typedef struct pulseg_playout_block
    {
        int position;                   /**< its index in the segment      */
        pulseg_block_instance instance; /**< its registers as played; the
                                             prescan's changes applied     */
        const pulseg_wave_region *wave; /**< where its wave plays from, at
                                             instance.wave_amp_hz_per_m;
                                             NULL where it plays none      */
    } pulseg_playout_block;

    /**
     * @brief Samples to load into waveform memory.
     *
     * One axis of wave @c wave of subsequence @c subsequence at the centres of
     * @c count raster intervals, as pulseg_wave_samples() converts it: the
     * region it is loaded into, from @c offset in that axis's memory.
     */
    typedef struct pulseg_wave_load
    {
        int subsequence;
        int wave;
        int axis;
        long offset;
        long count;
        const PULSEG_WAVE_SAMPLE *samples;
    } pulseg_wave_load;

    /**
     * @brief Values normalised to unit peak, as waveform-memory samples.
     *
     * Each is scaled by PULSEG_WAVE_FULL_SCALE, clamped to
     * [-PULSEG_WAVE_FULL_SCALE, PULSEG_WAVE_FULL_SCALE] and quantized with
     * PULSEG_WAVE_QUANTIZE.  For a wave, an arbitrary gradient shape or an RF
     * magnitude, whose physical scale is the amplitude a block plays it at.
     */
    void pulseg_wave_samples(const float *values, long count, PULSEG_WAVE_SAMPLE *samples);

    /**
     * @brief Phases as waveform-memory samples.
     *
     * Each phase, times @p radians_per_unit, is wrapped into [-pi, pi), scaled
     * by PULSEG_WAVE_FULL_SCALE / PULSEG_WAVE_PHASE_FULL_SCALE, clamped and
     * quantized as pulseg_wave_samples() does.  An RF phase shape holds
     * cycles, 2 pi radians per unit; an ADC's phase modulation, radians.
     */
    void pulseg_phase_samples(
        const float *phase,
        long count,
        double radians_per_unit,
        PULSEG_WAVE_SAMPLE *samples);

    /**
     * @brief A playout's side of the two stages.
     *
     * A NULL entry is skipped.  Each returns PULSEG_SUCCESS or a negative
     * error code, which ends the stage and is its result.
     */
    typedef struct pulseg_playout_backend
    {
        void *ctx; /**< handed to every entry */

        /** First stage: the waveform memory the waves take, per axis
         *  plan->samples, before anything is loaded into it. */
        int (*reserve_waves)(void *ctx, const pulseg_wave_plan *plan);

        /** First stage: a segment, before its positions. */
        int (*prepare_segment)(void *ctx, int segment, const pulseg_segment_info *info);

        /** First stage: a position of @p segment.  @p slots is NULL where
         *  the position plays no wave.  Otherwise it holds one entry per slot
         *  of the plan, each with the span every wave the position plays
         *  covers: where the waves are streamed, the ring of slots they play
         *  from; where they are resident, offsets of -1, each wave's own
         *  region saying where it is held. */
        int (*prepare_block)(
            void *ctx,
            int segment,
            int position,
            const pulseg_block_info *info,
            const pulseg_wave_region *slots);

        /** Both stages: samples into waveform memory.  Resident waves are
         *  loaded in the first stage; streamed ones in the second, into the
         *  slot the instance plays, once the instance before it has been
         *  handed over.  That slot was last read by the instance of its
         *  segment the plan's number of slots earlier, which the backend lets
         *  end before it writes. */
        int (*load_wave)(void *ctx, const pulseg_wave_load *load);

        /** Second stage: a segment instance, before its blocks. */
        int (*begin_instance)(void *ctx, const pulseg_playout_segment *segment);

        /** Second stage: a block of the instance, in order.  The collection's
         *  cursor is on the block, so the cursor getters answer for it. */
        int (*set_block)(
            void *ctx,
            const pulseg_playout_segment *segment,
            const pulseg_playout_block *block);

        /** Second stage: start the instance, once its blocks are set.
         *
         *  PULSEG_PLAYOUT_REPLAY asks for the same instance again, from its
         *  first block, and it is not counted among the instances played
         *  before the next one. A scan that judges a measurement only once it
         *  has made it -- a tracker that rejects a repetition the subject
         *  moved during -- says so here, where the decision is already made.
         */
        int (*play_instance)(void *ctx, const pulseg_playout_segment *segment);
    } pulseg_playout_backend;

    /**
     * @brief How the scan loop plays a collection.
     */
    typedef struct pulseg_playout_options
    {
        int prescan_subsequence; /**< -1 plays the scan, which plays no
                                      prescan-only subsequence; otherwise the
                                      receive-gain calibration prescan on
                                      this subsequence                     */
        int prescan_readouts;    /**< readouts after which the prescan ends,
                                      with the instance that completes them;
                                      0 takes the subsequence's
                                      num_gain_cal_readouts, at least 1    */
    } pulseg_playout_options;

/* clang-format off */
#define PULSEG_PLAYOUT_OPTIONS_INIT {-1, 0}
/* clang-format on */

    /**
     * @brief The receive-gain calibration a prescan plays.
     *
     * The subsequence the host generated for it, which only a prescan plays
     * (pulseg_subseq_info.prescan_only), where the cache holds one that
     * reads out.  Otherwise, where exactly one subsequence reads out, that
     * subsequence from its start, after the generated subsequence where the
     * cache holds one that does not read out: a pTx tour, every channel
     * driven at the scan's largest before the calibration.
     */
    typedef struct pulseg_receive_calibration
    {
        int generated;   /**< 1 for the generated, prescan-only subsequence */
        int lead;        /**< the prescan-only subsequence played before
                              @c subsequence, -1 for none                  */
        int subsequence; /**< subsequence a prescan calibrates on, -1 where
                              neither holds                               */
        int readouts;    /**< readouts it calibrates on                    */
    } pulseg_receive_calibration;

    /**
     * @brief Find the receive-gain calibration of a collection.
     *
     * @return PULSEG_SUCCESS, or a negative error code.
     */
    int pulseg_get_receive_calibration(const pulseg_collection *coll, pulseg_receive_calibration *out);

    /**
     * @brief The largest drive each transmit channel receives in the scan, or
     *        in a prescan.
     *
     * A drive is the RF amplitude times the channel's waveform peak, times
     * the channel's shim magnitude under a shim; in a parallel-transmit
     * collection a single waveform with no shim drives every channel.  With
     * @p options NULL or naming no prescan subsequence, over every block of
     * the scan, prescan-only subsequences left out; otherwise over what that
     * prescan plays: a prescan-only subsequence it plays first, then the
     * calibration from the subsequence's start to the instance completing
     * its readouts.  Comparing the two says
     * whether a prescan drives every channel as hard as the scan does.
     *
     * @param[out] drive_hz  Per channel, in Hz; @p max_channels entries are
     *                       written, 0 beyond the collection's channels.
     * @return The collection's transmit channels (1 without
     *         parallel-transmit RF), or a negative error code.
     */
    int pulseg_get_channel_drives(
        const pulseg_collection *coll,
        const pulseg_playout_options *options,
        float *drive_hz,
        int max_channels);

    /**
     * @brief The first stage: prepare every segment and load resident waves.
     *
     * Reserves the memory @p plan holds the waves in, hands the backend every
     * global segment and each of its positions in order, and, where the
     * waves are resident, loads every wave of every subsequence into its
     * region, on the plan's raster.  Needs the definitions alone, as the
     * pulse-generation cache holds them.
     *
     * @param[in] plan  The layout, as pulseg_get_wave_plan() copies it.
     * @return PULSEG_SUCCESS, or the first negative code the backend returns.
     */
    int pulseg_playout_prepare(
        const pulseg_collection *coll,
        const pulseg_wave_plan *plan,
        const pulseg_playout_backend *backend);

    /**
     * @brief The second stage: play the execution stream, instance by instance.
     *
     * For each segment instance, in play order: begin_instance(); for each
     * of its blocks, load its wave where the waves are streamed, then
     * set_block(); then play_instance().  The n-th instance of a segment
     * plays slot n % slots of its positions' rings, so its waves are loaded
     * while the instances before it play.
     *
     * The prescan plays one subsequence from its start and ends with the
     * instance that completes its readouts, after the whole of a
     * prescan-only subsequence other than it, once, under the same rules,
     * whose readouts are NOPRESCAN.  The scan plays no prescan-only
     * subsequence.  It waits
     * for no trigger, drives
     * no digital output, plays at zero every gradient whose amplitude varies
     * across repetitions, and a wave any of whose logical axes varies, and
     * neither acquires nor counts a readout its block's NOPRESCAN label
     * marks.
     *
     * @param[in,out] coll     Collection with its execution stream loaded;
     *                         its cursor is reset and moved.
     * @param[in]     plan     The layout the first stage prepared.
     * @param[in]     options  NULL plays the scan.
     * @return PULSEG_SUCCESS; PULSEG_ERR_INVALID_ARGUMENT, with @p diag
     *         filled, when the execution stream is not loaded; or the first
     *         negative code the backend returns.
     */
    int pulseg_playout_scan(
        pulseg_collection *coll,
        const pulseg_wave_plan *plan,
        const pulseg_playout_backend *backend,
        const pulseg_playout_options *options,
        pulseg_diagnostic *diag);

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_PLAYOUT_H */
