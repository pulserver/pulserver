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

#ifdef __cplusplus
extern "C"
{
#endif

    /* ================================================================== */
    /*  Sequence description (SEQDESC section)                            */
    /* ================================================================== */

    /**
     * @brief Build a per-subsequence sequence description from a loaded collection.
     *
     * Allocates and populates @p out with the event list, RF shape tuples, shim
     * definitions, and composite RF group annotations for @p subseq_idx.
     * Call pulseg_free_sequence_description() when done.
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
    void pulseg_free_sequence_description(pulseg_sequence_description *desc);

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
