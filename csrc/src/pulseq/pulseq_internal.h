/* pulseq_internal.h -- types shared between the pulseq module's own sources.
 *
 * Included by pulseq implementation (.c) files only.  NOT part of the
 * public API; nothing outside csrc/src/pulseq/ may include it.
 */

#ifndef PULSEQ_INTERNAL_H
#define PULSEQ_INTERNAL_H

#include "pulseq.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/** @brief One (name, id) row in a name-lookup table. */
typedef struct pulseq_table_entry
{
    const char *name;
    int value;
} pulseq_table_entry;

/** @brief Per-library scale factors applied while reading library rows. */
typedef struct pulseq_scale
{
    int size;
    float *values;
} pulseq_scale;

/**
 * @brief Extract the reserved (well-known) keys out of an already-populated
 * generic definitions library.
 *
 * Shared by the text reader (pulseq_parse.c) and the binary reader
 * (pulseq_binary.c): both build definitions_library as name plus string
 * values, so both need the same second step to fill
 * reserved_definitions_library.
 */
void pulseq__read_definitions(pulseq_file *seq);

/**
 * @brief Free every library inside @p seq and return it to its initial state,
 * keeping file_path and design_raster (unlike pulseq_file_free()).
 */
void pulseq__file_reset(pulseq_file *seq);

#endif /* PULSEQ_INTERNAL_H */
