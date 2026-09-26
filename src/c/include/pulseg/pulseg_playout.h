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
        int half;           /**< the half of its positions' wave slots it
                                 plays; -1 unless the waves are streamed   */
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
     * One axis of wave @c wave of subsequence @c subsequence, normalised to
     * unit peak, at the centres of @c count raster intervals: the region it
     * is loaded into, from @c offset in that axis's memory.
     */
    typedef struct pulseg_wave_load
    {
        int subsequence;
        int wave;
        int axis;
        long offset;
        long count;
        const float *samples;
    } pulseg_wave_load;

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

        /** First stage: a position of @p segment.  @p slot is NULL where the
         *  position plays no wave.  Otherwise both entries hold the span every
         *  wave it plays covers, and, where the waves are streamed, the two
         *  slots they play from; where they are resident, the offsets are -1
         *  and each wave's own region says where it is held. */
        int (*prepare_block)(
            void *ctx,
            int segment,
            int position,
            const pulseg_block_info *info,
            const pulseg_wave_region *slot);

        /** Both stages: samples into waveform memory.  Resident waves are
         *  loaded in the first stage; streamed ones in the second, into the
         *  half the instance plays, after the instance before it has
         *  started. */
        int (*load_wave)(void *ctx, const pulseg_wave_load *load);

        /** Second stage: a segment instance, before its blocks. */
        int (*begin_instance)(void *ctx, const pulseg_playout_segment *segment);

        /** Second stage: a block of the instance, in order.  The collection's
         *  cursor is on the block, so the cursor getters answer for it. */
        int (*set_block)(
            void *ctx,
            const pulseg_playout_segment *segment,
            const pulseg_playout_block *block);

        /** Second stage: start the instance, once its blocks are set. */
        int (*play_instance)(void *ctx, const pulseg_playout_segment *segment);
    } pulseg_playout_backend;

    /**
     * @brief How the scan loop plays a collection.
     */
    typedef struct pulseg_playout_options
    {
        int prescan_subsequence; /**< -1 plays the scan; otherwise the
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
     * @brief The first stage: prepare every segment and load resident waves.
     *
     * Lays out the waves with pulseg_plan_waves() on @p budget, reserves
     * their memory, hands the backend every global segment and each of its
     * positions in order, and, where the waves are resident, loads every
     * wave of every subsequence into its region.  Needs the definitions
     * alone, as the pulse-generation cache holds them.
     *
     * @return PULSEG_SUCCESS; a pulseg_plan_waves() error, with @p diag
     *         filled; or the first negative code the backend returns.
     */
    int pulseg_playout_prepare(
        const pulseg_collection *coll,
        const pulseg_wave_budget *budget,
        const pulseg_playout_backend *backend,
        pulseg_diagnostic *diag);

    /**
     * @brief The second stage: play the execution stream, instance by instance.
     *
     * For each segment instance, in play order: begin_instance(); for each
     * of its blocks, load its wave where the waves are streamed, then
     * set_block(); then play_instance().  The n-th instance of a segment
     * plays half n % 2 of the streamed slots, so its waves are loaded while
     * the instance before it plays.  The wave layout is recomputed from
     * @p budget as the first stage computed it, and a streamed layout that
     * cannot be loaded in time is refused before anything plays.
     *
     * The prescan plays one subsequence from its start and ends with the
     * instance that completes its readouts.  It waits for no trigger, drives
     * no digital output, and plays at zero every gradient whose amplitude
     * varies across repetitions, and a wave any of whose logical axes
     * varies.
     *
     * @param[in,out] coll     Collection with its execution stream loaded;
     *                         its cursor is reset and moved.
     * @param[in]     options  NULL plays the scan.
     * @return PULSEG_SUCCESS; a pulseg_plan_waves() error, with @p diag
     *         filled; or the first negative code the backend returns.
     */
    int pulseg_playout_scan(
        pulseg_collection *coll,
        const pulseg_wave_budget *budget,
        const pulseg_playout_backend *backend,
        const pulseg_playout_options *options,
        pulseg_diagnostic *diag);

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_PLAYOUT_H */
