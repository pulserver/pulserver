/**
 * @file pulseg_convert.h
 * @brief Raw pulseq files -> pulseg collection, and the derived sequence
 *        description.
 *
 * pulseg_convert_collection() is the seam between the two modules: it takes
 * already-parsed pulseq_file structures and produces the deduplicated,
 * segmented pulseg intermediate representation. pulseg_read() composes it
 * with the pulseq reader for callers that just want a file path.
 *
 * The sequence description is the human/metadata view of a loaded
 * collection -- the event list, RF shape tuples and shim definitions that
 * the recon side and the analysis tooling consume.
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
     * This is the "convert" half of the former one-shot @c pulseg_read() /
     * @c pulseg_read_from_buffers() loaders -- those now compose
     * @c pulseq_read() (or @c _from_buffer / @c
     * pulseq_file_set_read()) with this function. Each element of
     * @p files must already carry its own populated @c opts (set at
     * pulseq_file_init() / read time); there is no separate top-level
     * opts parameter -- passing one alongside per-file opts would be
     * ambiguous about which wins, so callers rely on the per-file copy.
     *
     * @param[out] coll          Caller-allocated collection to populate
     *                           (typically freshly heap-allocated and
     *                           zeroed, as pulseg_read() does).
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
    /*  Sequence description (SEQDESC section)                            */
    /* ================================================================== */

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_CONVERT_H */
