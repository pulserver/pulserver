/**
 * @file pulseg_repetition.c
 * @brief The gradients of the repetition of a subsequence that carries the
 *        most gradient energy, as the cache carries them, and their samples
 *        on a raster.
 *
 * See pulseg_get_tr_corner_points().
 */

#include <stdlib.h>
#include <string.h>

#include "pulseg_internal.h"
#include "pulseg.h"

static float *copy_floats(const float *from, int count)
{
    float *to = (float *)PULSEG_ALLOC((size_t)count * sizeof(float));

    if (to)
        memcpy(to, from, (size_t)count * sizeof(float));
    return to;
}

void pulseg_corner_point_stream_free(pulseg_corner_point_stream *s)
{
    static const pulseg_corner_point_stream empty = PULSEG_CORNER_POINT_STREAM_INIT;

    if (!s)
        return;
    if (s->time_us)
        PULSEG_FREE(s->time_us);
    if (s->gx_hz_per_m)
        PULSEG_FREE(s->gx_hz_per_m);
    if (s->gy_hz_per_m)
        PULSEG_FREE(s->gy_hz_per_m);
    if (s->gz_hz_per_m)
        PULSEG_FREE(s->gz_hz_per_m);
    *s = empty;
}

static int copy_stream(const pulseg_corner_point_stream *from, pulseg_corner_point_stream *to)
{
    *to = *from;
    to->time_us = copy_floats(from->time_us, from->num_points);
    to->gx_hz_per_m = copy_floats(from->gx_hz_per_m, from->num_points);
    to->gy_hz_per_m = copy_floats(from->gy_hz_per_m, from->num_points);
    to->gz_hz_per_m = copy_floats(from->gz_hz_per_m, from->num_points);
    if (to->time_us && to->gx_hz_per_m && to->gy_hz_per_m && to->gz_hz_per_m)
        return PULSEG_SUCCESS;
    pulseg_corner_point_stream_free(to);
    return PULSEG_ERR_ALLOC_FAILED;
}

int pulseg_get_tr_corner_points(
    const pulseg_collection *coll,
    pulseg_corner_point_stream *out,
    pulseg_diagnostic *diag,
    int subseq_idx)
{
    static const pulseg_corner_point_stream empty = PULSEG_CORNER_POINT_STREAM_INIT;
    const pulseg_corner_point_stream *stored;

    if (!coll || !out)
        return PULSEG_ERR_NULL_POINTER;
    *out = empty;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;
    stored = &coll->descriptors[subseq_idx].repetition;
    if (stored->num_points <= 0)
    {
        if (diag)
            pulseg__diag_printf(
                diag, "subsequence %d: the collection carries no repetition gradients", subseq_idx);
        return PULSEG_ERR_INVALID_ARGUMENT;
    }
    return copy_stream(stored, out);
}

int pulseg_sample_corner_points(
    const pulseg_corner_point_stream *s,
    int axis,
    float raster_us,
    long num_samples,
    float *out)
{
    const float *v;
    long i;

    if (!s || !out)
        return PULSEG_ERR_NULL_POINTER;
    if (axis < PULSEG_GRAD_AXIS_X || axis > PULSEG_GRAD_AXIS_Z || raster_us <= 0.0f ||
        num_samples < 0)
        return PULSEG_ERR_INVALID_ARGUMENT;
    v = (axis == PULSEG_GRAD_AXIS_X) ? s->gx_hz_per_m
        : (axis == PULSEG_GRAD_AXIS_Y) ? s->gy_hz_per_m
                                       : s->gz_hz_per_m;
    for (i = 0; i < num_samples; ++i)
        out[i] = pulseg__linear_at(
            s->time_us, v, s->num_points, ((float)i + 0.5f) * raster_us);
    return PULSEG_SUCCESS;
}
