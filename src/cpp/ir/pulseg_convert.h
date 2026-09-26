/**
 * @file pulseg_convert.h
 * @brief Raw pulseq files -> pulseg collection, and the layout of its waves,
 *        on the host.
 *
 * pulseg_convert_collection() is the seam between the two modules: it takes
 * pulseq_file structures the caller filled and produces the deduplicated,
 * segmented pulseg intermediate representation.  pulseg_plan_waves() lays
 * out its waves for a playout's budget.  Both are implemented by the C++
 * passes beside this header and are not part of the library a scanner links.
 */

#ifndef PULSEG_CONVERT_H
#define PULSEG_CONVERT_H

#include "pulseg_config.h"
#include "pulseg_types.h"
#include "pulseg_io.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /* ================================================================== */
    /*  Raw pulseq model -> collection                    */
    /* ================================================================== */

    /**
     * @brief Convert @p n already-parsed pulseq files into a loaded
     * collection: unique-block dedup, TR/segmentation detection, scan-table
     * expansion, freq-mod flags, label table, and cross-subsequence
     * consistency checks.
     *
     * @param[out] coll          Caller-allocated collection to populate,
     *                           from pulseg_collection_alloc().
     * @param[out] diag          Optional diagnostic (NULL uses a local one).
     * @param[in]  files         Array of @p n already-parsed pulseq files.
     * @param[in]  n             Number of entries in @p files (>= 1).
     * @param[in]  opts          Scanner limits, rasters and vendor hooks.
     * @param[in]  parse_labels  1 to also build the ADC label table.
     * @return Number of subsequences converted on success (== @p n),
     *         0 on failure (diag->code holds the negative error code).
     */
    int pulseg_convert_collection(
        pulseg_collection *coll,
        pulseg_diagnostic *diag,
        const pulseq_file *files,
        int n,
        const pulseg_opts *opts,
        int parse_labels);

    /* ================================================================== */
    /*  Waveform memory for waves                                         */
    /* ================================================================== */

    /**
     * @brief Lay out the waves of a collection in a playout's waveform
     * memory, as pulseg_get_wave_plan() describes the layout.
     *
     * @param[in]  coll    Collection with its execution stream.
     * @param[out] plan    Overwritten; release with pulseg_free_wave_plan(),
     *                     whatever the result.
     * @param[out] diag    States the shortfall on failure; may be NULL.
     * @return PULSEG_SUCCESS; PULSEG_ERR_INVALID_ARGUMENT for a budget with a
     *         raster or headroom that is not positive, a negative memory or
     *         load rate, or fewer than two slots; PULSEG_ERR_WAVE_MEMORY when
     *         neither layout fits, with the sizes filled;
     *         PULSEG_ERR_WAVE_LOADING when an instance cannot be loaded in
     *         time, with the plan filled and the instance named; or another
     *         negative error code.
     */
    int pulseg_plan_waves(
        const pulseg_collection *coll,
        const pulseg_wave_budget *budget,
        pulseg_wave_plan *plan,
        pulseg_diagnostic *diag);

    /**
     * @brief Lay out the waves of a converted collection with
     * pulseg_plan_waves() and keep the layout, which its cache then carries.
     */
    int pulseg_store_wave_plan(
        pulseg_collection *coll,
        const pulseg_wave_budget *budget,
        pulseg_diagnostic *diag);

    /* ================================================================== */
    /*  Gradients of the heaviest repetition                              */
    /* ================================================================== */

    /**
     * @brief Find the heaviest repetition of every subsequence of a converted
     * collection and keep its gradients, which its cache then carries, as
     * pulseg_get_tr_corner_points() describes them.
     */
    int pulseg_store_repetitions(pulseg_collection *coll);

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_CONVERT_H */
