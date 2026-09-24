/**
 * @file pulseq.h
 * @brief The raw Pulseq file model the conversion is fed, and its shape codec.
 *
 * This module is self-contained: it depends on no other library in this
 * repository. It owns the model (pulseq_types.h) and the run-length shape
 * codec the intermediate representation keeps its waveforms in. The host
 * fills, walks and frees a pulseq_file through src/cpp/ir/pulseq_file.h.
 */

#ifndef PULSEQ_H
#define PULSEQ_H

#include <stdio.h>

#include "pulseq_config.h"
#include "pulseq_types.h"

#if defined(__cplusplus) && !defined(PULSEQ_NO_EXTERN_C)
extern "C"
{
#endif

    /* ============================================================== */
    /*  Shape codec                                                   */
    /* ============================================================== */

    /**
     * @brief Decompress a run-length-encoded SHAPES library entry.
     * @param[out] result   Receives the decompressed samples (caller frees
     *                      result->samples with PULSEQ_FREE).
     * @param[in]  encoded  Raw (possibly RLE-compressed) shape.
     * @param[in]  scale    Multiplier applied to every decompressed sample.
     * @return 1 on success, 0 on failure.
     */
    int pulseq_decompress_shape(
        pulseq_shape *result,
        const pulseq_shape *encoded,
        PULSEQ_REAL scale);

#if defined(__cplusplus) && !defined(PULSEQ_NO_EXTERN_C)
}
#endif

#endif /* PULSEQ_H */
