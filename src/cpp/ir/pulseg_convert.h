/**
 * @file pulseg_convert.h
 * @brief Raw pulseq files -> pulseg collection, on the host.
 *
 * pulseg_convert_collection() is the seam between the two modules: it takes
 * pulseq_file structures the caller filled and produces the deduplicated,
 * segmented pulseg intermediate representation. It is implemented by the C++
 * passes beside this header and is not part of the library a scanner links.
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

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_CONVERT_H */
