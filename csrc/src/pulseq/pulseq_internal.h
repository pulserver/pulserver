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

#endif /* PULSEQ_INTERNAL_H */
