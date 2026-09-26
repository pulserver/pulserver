/**
 * @file pulseg_repetition.c
 * @brief The gradients of the repetition of a subsequence that carries the
 *        most gradient energy, joined over its blocks.
 *
 * See pulseg_get_tr_corner_points().
 */

#include <stdlib.h>
#include <string.h>

#include "pulseg_internal.h"
#include "pulseg.h"

/* The integral over block-table entry @p block of its squared gradient summed
 * over the axes, in (Hz/m)^2 s; its rotation leaves it unchanged. */
static float block_energy(const pulseg_sequence_descriptor *desc, int block)
{
    const pulseg_block_table_element *bte = &desc->block_table[block];
    const pulseg_grad_table_element *element;
    float energy = 0.0f;
    int ids[3];
    int d;

    ids[0] = bte->gx_id;
    ids[1] = bte->gy_id;
    ids[2] = bte->gz_id;
    for (d = 0; d < 3; ++d)
    {
        if (ids[d] < 0 || ids[d] >= desc->grad_table_size)
            continue;
        element = &desc->grad_table[ids[d]];
        if (element->id < 0 || element->id >= desc->num_unique_grads)
            continue;
        energy += element->amplitude * element->amplitude *
            pulseg__grad_instance_energy(
                desc, &desc->grad_definitions[element->id], element->shape_id);
    }
    return energy;
}

/* The execution-stream positions per repetition; the whole stream where the
 * subsequence does not repeat. */
static int repetition_size(const pulseg_sequence_descriptor *desc)
{
    const int size = desc->tr_descriptor.tr_size;
    return (size > 0 && size < desc->exec_stream_len) ? size : desc->exec_stream_len;
}

static float repetition_energy(const pulseg_sequence_descriptor *desc, int first, int count)
{
    float energy = 0.0f;
    int n;

    for (n = first; n < first + count && n < desc->exec_stream_len; ++n)
    {
        const int block = pulseg__exec_block_idx(desc, n);
        if (block >= 0 && block < desc->num_blocks)
            energy += block_energy(desc, block);
    }
    return energy;
}

/* The first execution-stream position of the repetition of most gradient
 * energy, the earliest of those that tie, and that energy. */
static int heaviest_repetition(const pulseg_sequence_descriptor *desc, int size, float *energy)
{
    int first, heaviest = 0;

    *energy = -1.0f;
    for (first = 0; first < desc->exec_stream_len; first += size)
    {
        const float e = repetition_energy(desc, first, size);
        if (e > *energy)
        {
            *energy = e;
            heaviest = first;
        }
    }
    return heaviest;
}

/* The stream being joined, and the room its arrays have. */
typedef struct stream_builder
{
    pulseg_corner_point_stream *out;
    int capacity;
} stream_builder;

static float *grown(float *old, int used, int capacity)
{
    float *made = (float *)PULSEG_ALLOC((size_t)capacity * sizeof(float));

    if (made && old && used > 0)
        memcpy(made, old, (size_t)used * sizeof(float));
    if (old)
        PULSEG_FREE(old);
    return made;
}

static int builder_room(stream_builder *b, int need)
{
    pulseg_corner_point_stream *s = b->out;
    const int capacity = (need > 2 * b->capacity) ? need : 2 * b->capacity;

    if (need <= b->capacity)
        return PULSEG_SUCCESS;
    s->time_us = grown(s->time_us, s->num_points, capacity);
    s->gx_hz_per_m = grown(s->gx_hz_per_m, s->num_points, capacity);
    s->gy_hz_per_m = grown(s->gy_hz_per_m, s->num_points, capacity);
    s->gz_hz_per_m = grown(s->gz_hz_per_m, s->num_points, capacity);
    if (!s->time_us || !s->gx_hz_per_m || !s->gy_hz_per_m || !s->gz_hz_per_m)
        return PULSEG_ERR_ALLOC_FAILED;
    b->capacity = capacity;
    return PULSEG_SUCCESS;
}

/* Append a corner.  One at the time of the last replaces it: where one
 * block's gradient meets the next's, the later block's corner stands. */
static int builder_point(stream_builder *b, float t, float gx, float gy, float gz)
{
    pulseg_corner_point_stream *s = b->out;
    int i;

    if (s->num_points > 0 && t <= s->time_us[s->num_points - 1] + 1e-3f)
        i = s->num_points - 1;
    else if (PULSEG_FAILED(builder_room(b, s->num_points + 1)))
        return PULSEG_ERR_ALLOC_FAILED;
    else
        i = s->num_points++;
    s->time_us[i] = t;
    s->gx_hz_per_m[i] = gx;
    s->gy_hz_per_m[i] = gy;
    s->gz_hz_per_m[i] = gz;
    return PULSEG_SUCCESS;
}

/* Append the corners block-table entry @p block plays, from @p start_us. */
static int builder_block(
    stream_builder *b,
    const pulseg_sequence_descriptor *desc,
    int block,
    float start_us)
{
    float *time_us = NULL;
    float *gradient[3];
    int i, n, a;
    int rc = pulseg__block_gradients(desc, block, &time_us, gradient, &n);

    for (i = 0; i < n && PULSEG_SUCCEEDED(rc); ++i)
        rc = builder_point(
            b, start_us + time_us[i], gradient[0][i], gradient[1][i], gradient[2][i]);
    if (time_us)
        PULSEG_FREE(time_us);
    for (a = 0; a < 3; ++a)
        if (gradient[a])
            PULSEG_FREE(gradient[a]);
    return rc;
}

/* Join the blocks of the repetition from execution-stream position @p first,
 * from zero at its start, where no gradient plays there, to zero at its end,
 * where none plays up to it. */
static int join_repetition(
    stream_builder *b,
    const pulseg_sequence_descriptor *desc,
    int first,
    int size)
{
    float start_us = 0.0f;
    int n, rc;

    rc = builder_point(b, 0.0f, 0.0f, 0.0f, 0.0f);
    for (n = first; n < first + size && n < desc->exec_stream_len && PULSEG_SUCCEEDED(rc); ++n)
    {
        const int block = pulseg__exec_block_idx(desc, n);
        if (block < 0 || block >= desc->num_blocks)
            continue;
        rc = builder_block(b, desc, block, start_us);
        start_us += (float)pulseg__played_duration_us(desc, block);
    }
    if (PULSEG_SUCCEEDED(rc) && b->out->time_us[b->out->num_points - 1] < start_us - 1e-3f)
        rc = builder_point(b, start_us, 0.0f, 0.0f, 0.0f);
    b->out->duration_us = start_us;
    return rc;
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

int pulseg_get_tr_corner_points(
    const pulseg_collection *coll,
    pulseg_corner_point_stream *out,
    pulseg_diagnostic *diag,
    int subseq_idx)
{
    static const pulseg_corner_point_stream empty = PULSEG_CORNER_POINT_STREAM_INIT;
    const pulseg_sequence_descriptor *desc;
    stream_builder builder;
    int size, rc;

    if (!coll || !out)
        return PULSEG_ERR_NULL_POINTER;
    *out = empty;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;
    desc = &coll->descriptors[subseq_idx];
    if (desc->exec_stream_len <= 0 || !desc->block_table)
    {
        if (diag)
            pulseg__diag_printf(diag, "subsequence %d: no execution stream loaded", subseq_idx);
        return PULSEG_ERR_INVALID_ARGUMENT;
    }

    size = repetition_size(desc);
    out->first_position = heaviest_repetition(desc, size, &out->energy);
    builder.out = out;
    builder.capacity = 0;
    rc = join_repetition(&builder, desc, out->first_position, size);
    if (PULSEG_FAILED(rc))
        pulseg_corner_point_stream_free(out);
    return rc;
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
