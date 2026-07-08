/**
 * @file pulseg_convert.h
 * @brief Sequence-description (SEQDESC) derivation entry points.
 *
 * Split out of the former pulseg_methods.h (Stage 1 layout normalization).
 * All functions use the pulseg_ prefix and are declared extern "C" when
 * compiled with a C++ compiler.
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
    /*  Raw pulseq model -> collection (Stage 3 Step 2)                    */
    /* ================================================================== */

    /**
     * @brief Convert @p n already-parsed pulseq files into a loaded
     * collection: unique-block dedup, TR/segmentation detection, scan-table
     * expansion, freq-mod flags, label table, and cross-subsequence
     * consistency checks.
     *
     * This is the "convert" half of the former one-shot @c pulseg_read() /
     * @c pulseg_read_from_buffers() loaders -- those now compose
     * @c pulseg_pulseq_file_read() (or @c _from_buffer / @c
     * pulseg_pulseq_file_set_read()) with this function. Each element of
     * @p files must already carry its own populated @c opts (set at
     * pulseg_pulseq_file_init() / read time); there is no separate top-level
     * opts parameter -- passing one alongside per-file opts would be
     * ambiguous about which wins, so callers rely on the per-file copy.
     *
     * @param[out] coll          Caller-allocated collection to populate
     *                           (typically freshly heap-allocated and
     *                           zeroed, as pulseg_read() does).
     * @param[out] diag          Optional diagnostic (NULL uses a local one).
     * @param[in]  files         Array of @p n already-parsed pulseq files.
     * @param[in]  n             Number of entries in @p files (>= 1).
     * @param[in]  parse_labels  1 to also build the ADC label table.
     * @param[in]  num_averages  Number of averages (>= 1; see
     *                           pulseg_read()'s parameter of the same name).
     * @return Number of subsequences converted on success (== @p n),
     *         0 on failure (diag->code holds the negative error code).
     */
    int pulseg_convert_collection(
        pulseg_collection *coll,
        pulseg_diagnostic *diag,
        const pulseg_pulseq_file *files,
        int n,
        int parse_labels,
        int num_averages);

    /* ================================================================== */
    /*  Sequence description (SEQDESC section)                            */
    /* ================================================================== */

    /**
     * @brief Build a per-subsequence sequence description from a loaded collection.
     *
     * Allocates and populates @p out with the event list, RF shape tuples, shim
     * definitions, and composite RF group annotations for @p subseq_idx.
     * Call pulseg_sequence_description_free() when done.
     *
     * @param[out] out        Caller-allocated descriptor to fill.
     * @param[in]  coll       Loaded pulseg collection.
     * @param[in]  subseq_idx Subsequence index.
     * @return PULSEG_SUCCESS or negative error code.
     */
    int pulseg_get_sequence_description(pulseg_sequence_description *out,
                                           const pulseg_collection *coll,
                                           int subseq_idx);

    /**
     * @brief Free all heap allocations inside a pulseg_sequence_description.
     *
     * Does NOT free @p desc itself.
     *
     * @param[in,out] desc  Descriptor whose inner pointers to free.
     */
    void pulseg_sequence_description_free(pulseg_sequence_description *desc);

    /**
     * @brief Compute scan-global sequence parameters from all loaded subsequences.
     *
     * @param[out] out   Caller-allocated output struct.
     * @param[in]  coll  Loaded pulseg collection.
     * @return PULSEG_SUCCESS or negative error code.
     */
    int pulseg_get_sequence_parameters(pulseg_sequence_parameters *out,
                                          const pulseg_collection *coll);

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_CONVERT_H */
