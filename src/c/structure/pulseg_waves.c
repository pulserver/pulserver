/**
 * @file pulseg_waves.c
 * @brief Rotated waves: the waveform each axis of a rotated block plays.
 *
 * See pulseg_wave in pulseg_internal.h for what a wave is, and
 * pulseg_materialize_wave() for how it is sampled.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "pulseg_internal.h"
#include "pulseg.h"

/* One logical axis of a block, normalised: value v[i] at time t[i] from the
 * block's start, linear in between, zero outside. */
typedef struct axis_corners
{
    float *t;
    float *v;
    int n;
} axis_corners;

static void axis_corners_free(axis_corners *c)
{
    if (c->t)
        PULSEG_FREE(c->t);
    if (c->v)
        PULSEG_FREE(c->v);
    c->t = NULL;
    c->v = NULL;
    c->n = 0;
}

/* 1 on success (an undriven axis has no corners), 0 when out of memory. */
static int axis_corners_build(
    const pulseg_sequence_descriptor *desc,
    int grad_def_id,
    int shape_id,
    axis_corners *out)
{
    const pulseg_grad_definition *gd;
    pulseq_shape samples;
    pulseq_shape times;
    float delay, raster;
    int i, k, ns, has_time;

    out->t = NULL;
    out->v = NULL;
    out->n = 0;
    if (grad_def_id < 0 || grad_def_id >= desc->num_unique_grads)
        return 1;
    gd = &desc->grad_definitions[grad_def_id];
    delay = (float)gd->delay;

    if (gd->type == 0)
    {
        k = (gd->flat_time_or_unused > 0) ? 4 : 3;
        out->t = (float *)PULSEG_ALLOC((size_t)k * sizeof(float));
        out->v = (float *)PULSEG_ALLOC((size_t)k * sizeof(float));
        if (!out->t || !out->v)
        {
            axis_corners_free(out);
            return 0;
        }
        out->t[0] = delay;
        out->v[0] = 0.0f;
        out->t[1] = delay + (float)gd->rise_time_or_unused;
        out->v[1] = 1.0f;
        if (k == 4)
        {
            out->t[2] = out->t[1] + (float)gd->flat_time_or_unused;
            out->v[2] = 1.0f;
        }
        out->t[k - 1] = out->t[k - 2] + (float)gd->fall_time_or_num_uncompressed_samples;
        out->v[k - 1] = 0.0f;
        out->n = k;
        return 1;
    }

    if (shape_id < 1 || shape_id > desc->num_shapes)
        return 1;
    samples.samples = NULL;
    samples.num_samples = 0;
    samples.num_uncompressed_samples = 0;
    if (!pulseq_decompress_shape(&samples, &desc->shapes[shape_id - 1], 1.0f))
        return 0;
    ns = samples.num_samples;
    if (ns <= 0)
    {
        if (samples.samples)
            PULSEG_FREE(samples.samples);
        return 1;
    }

    raster = desc->grad_raster_us;
    times.samples = NULL;
    times.num_samples = 0;
    times.num_uncompressed_samples = 0;
    has_time = 0;
    if (gd->unused_or_time_shape_id > 0 && gd->unused_or_time_shape_id <= desc->num_shapes)
    {
        if (!pulseq_decompress_shape(&times, &desc->shapes[gd->unused_or_time_shape_id - 1], raster))
        {
            PULSEG_FREE(samples.samples);
            if (times.samples)
                PULSEG_FREE(times.samples);
            return 0;
        }
        has_time = times.num_samples == ns;
    }

    out->t = (float *)PULSEG_ALLOC((size_t)(ns + 2) * sizeof(float));
    out->v = (float *)PULSEG_ALLOC((size_t)(ns + 2) * sizeof(float));
    if (!out->t || !out->v)
    {
        axis_corners_free(out);
        PULSEG_FREE(samples.samples);
        if (times.samples)
            PULSEG_FREE(times.samples);
        return 0;
    }

    /* On a time shape the first and last samples are the event's edges.  On
     * the raster the samples sit at interval centres, and the event holds
     * its end samples over the half intervals at its edges, which keeps the
     * area the samples give. */
    k = 0;
    if (!has_time)
    {
        out->t[k] = delay;
        out->v[k] = samples.samples[0];
        k++;
    }
    for (i = 0; i < ns; ++i)
    {
        out->t[k] = delay + (has_time ? times.samples[i] : ((float)i + 0.5f) * raster);
        out->v[k] = samples.samples[i];
        k++;
    }
    if (!has_time)
    {
        out->t[k] = delay + (float)ns * raster;
        out->v[k] = samples.samples[ns - 1];
        k++;
    }
    out->n = k;

    PULSEG_FREE(samples.samples);
    if (times.samples)
        PULSEG_FREE(times.samples);
    return 1;
}

static float axis_value_at(const axis_corners *c, float x)
{
    int lo, hi, mid;
    float span;

    if (c->n < 2 || x < c->t[0] || x > c->t[c->n - 1])
        return 0.0f;
    lo = 0;
    hi = c->n - 1;
    while (hi - lo > 1)
    {
        mid = (lo + hi) / 2;
        if (c->t[mid] <= x)
            lo = mid;
        else
            hi = mid;
    }
    span = c->t[hi] - c->t[lo];
    if (span <= 0.0f)
        return c->v[hi];
    return c->v[lo] + (c->v[hi] - c->v[lo]) * (x - c->t[lo]) / span;
}

/* 1 when one driven axis is an arbitrary gradient on the raster: its samples
 * are known only at raster centres, so the combination is too. */
static int wave_on_raster_centres(const pulseg_sequence_descriptor *desc, const pulseg_wave *wave)
{
    const pulseg_grad_definition *gd;
    int d;

    for (d = 0; d < 3; ++d)
    {
        if (wave->grad_def[d] < 0 || wave->grad_def[d] >= desc->num_unique_grads)
            continue;
        gd = &desc->grad_definitions[wave->grad_def[d]];
        if (gd->type != 0 && gd->unused_or_time_shape_id <= 0)
            return 1;
    }
    return 0;
}

static int float_cmp(const void *a, const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa < fb) ? -1 : ((fa > fb) ? 1 : 0);
}

int pulseg__wave_materialize(
    const pulseg_sequence_descriptor *desc,
    const pulseg_wave *wave,
    int out_axis,
    float *out_time_us,
    float *out_amp,
    int max_points,
    int *out_num_points,
    float *out_peak)
{
    axis_corners axis[3];
    float *grid = NULL;
    float peak, v, x, lo, hi, raster;
    int total, n, i, d, have, rc;
    int centred = 0;

    if (!desc || !wave || !out_num_points)
        return PULSEG_ERR_NULL_POINTER;
    if (out_axis < PULSEG_GRAD_AXIS_X || out_axis > PULSEG_GRAD_AXIS_Z)
        return PULSEG_ERR_INVALID_ARGUMENT;
    *out_num_points = 0;
    if (out_peak)
        *out_peak = 0.0f;

    for (d = 0; d < 3; ++d)
    {
        axis[d].t = NULL;
        axis[d].v = NULL;
        axis[d].n = 0;
    }
    rc = PULSEG_SUCCESS;
    total = 0;
    for (d = 0; d < 3; ++d)
    {
        if (!axis_corners_build(desc, wave->grad_def[d], wave->shape_id[d], &axis[d]))
        {
            rc = PULSEG_ERR_ALLOC_FAILED;
            goto done;
        }
        total += axis[d].n;
    }
    if (total == 0)
        goto done;

    if (wave_on_raster_centres(desc, wave))
    {
        /* Centres of the raster from the earliest start, rounded down onto
         * it, to the latest end. */
        raster = desc->grad_raster_us;
        lo = 0.0f;
        hi = 0.0f;
        have = 0;
        for (d = 0; d < 3; ++d)
        {
            if (axis[d].n < 1)
                continue;
            if (!have || axis[d].t[0] < lo)
                lo = axis[d].t[0];
            if (!have || axis[d].t[axis[d].n - 1] > hi)
                hi = axis[d].t[axis[d].n - 1];
            have = 1;
        }
        if (!have || hi <= lo || raster <= 0.0f)
            goto done;
        lo = (float)floor((double)(lo / raster) + 1e-6) * raster;
        n = (int)ceil((double)((hi - lo) / raster) - 1e-6);
        if (n < 1)
            n = 1;
        /* The centres, and the two edges the wave holds its end values to. */
        grid = (float *)PULSEG_ALLOC((size_t)(n + 2) * sizeof(float));
        if (!grid)
        {
            rc = PULSEG_ERR_ALLOC_FAILED;
            goto done;
        }
        grid[0] = lo;
        for (i = 0; i < n; ++i)
            grid[i + 1] = lo + ((float)i + 0.5f) * raster;
        grid[n + 1] = lo + (float)n * raster;
        n += 2;
        centred = 1;
    }
    else
    {
        /* Each axis is linear between its corners, so the union of the
         * corner times carries the combination exactly. */
        grid = (float *)PULSEG_ALLOC((size_t)total * sizeof(float));
        if (!grid)
        {
            rc = PULSEG_ERR_ALLOC_FAILED;
            goto done;
        }
        n = 0;
        for (d = 0; d < 3; ++d)
            for (i = 0; i < axis[d].n; ++i)
                grid[n++] = axis[d].t[i];
        qsort(grid, (size_t)n, sizeof(float), float_cmp);
        total = 0;
        for (i = 0; i < n; ++i)
            if (i == 0 || grid[i] > grid[total - 1])
                grid[total++] = grid[i];
        n = total;
    }

    *out_num_points = n;
    peak = 0.0f;
    for (i = 0; i < n; ++i)
    {
        /* On raster centres the edges hold the first and last centres. */
        x = grid[i];
        if (centred && i == 0)
            x = grid[1];
        else if (centred && i == n - 1)
            x = grid[n - 2];
        v = 0.0f;
        for (d = 0; d < 3; ++d)
            v += wave->rotation[out_axis * 3 + d] * wave->ratio[d] * axis_value_at(&axis[d], x);
        if ((float)fabs((double)v) > peak)
            peak = (float)fabs((double)v);
        if (out_time_us && out_amp && i < max_points)
        {
            out_time_us[i] = grid[i];
            out_amp[i] = v;
        }
    }
    if (out_peak)
        *out_peak = peak;
    if (!out_time_us || !out_amp)
        goto done;
    if (max_points < n)
    {
        rc = PULSEG_ERR_INDEX;
        goto done;
    }
    if (peak > 0.0f)
        for (i = 0; i < n; ++i)
            out_amp[i] /= peak;

done:
    if (grid)
        PULSEG_FREE(grid);
    for (d = 0; d < 3; ++d)
        axis_corners_free(&axis[d]);
    return rc;
}

int pulseg_get_num_waves(const pulseg_collection *coll, int subseq_idx)
{
    if (!coll)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;
    return coll->descriptors[subseq_idx].num_waves;
}

int pulseg_materialize_wave(
    const pulseg_collection *coll,
    int subseq_idx,
    int wave_idx,
    int out_axis,
    float *out_time_us,
    float *out_amp,
    int max_points,
    int *out_num_points,
    float *out_peak)
{
    const pulseg_sequence_descriptor *desc;

    if (!coll || !out_num_points)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;
    desc = &coll->descriptors[subseq_idx];
    if (wave_idx < 0 || wave_idx >= desc->num_waves || !desc->waves)
        return PULSEG_ERR_INVALID_ARGUMENT;
    return pulseg__wave_materialize(
        desc,
        &desc->waves[wave_idx],
        out_axis,
        out_time_us,
        out_amp,
        max_points,
        out_num_points,
        out_peak);
}

float pulseg__wave_scale(
    const pulseg_sequence_descriptor *desc,
    const pulseg_block_table_element *bte)
{
    const pulseg_grad_table_element *element;
    int ids[3];
    float scale;
    int d;

    ids[0] = bte->gx_id;
    ids[1] = bte->gy_id;
    ids[2] = bte->gz_id;
    scale = 0.0f;
    for (d = 0; d < 3; ++d)
    {
        if (ids[d] < 0 || ids[d] >= desc->grad_table_size)
            continue;
        element = &desc->grad_table[ids[d]];
        if (element->id < 0 || element->id >= desc->num_unique_grads)
            continue;
        if (fabs((double)element->amplitude) > fabs((double)scale))
            scale = element->amplitude;
    }
    return scale;
}

void pulseg__block_wave(
    const pulseg_sequence_descriptor *desc,
    int block_idx,
    int *wave_id,
    float amp_hz_per_m[3])
{
    const pulseg_wave *wave;
    float scale;
    int d, w;

    *wave_id = -1;
    amp_hz_per_m[0] = 0.0f;
    amp_hz_per_m[1] = 0.0f;
    amp_hz_per_m[2] = 0.0f;
    if (!desc->block_wave || !desc->waves || block_idx < 0 || block_idx >= desc->num_blocks)
        return;
    w = desc->block_wave[block_idx];
    if (w < 0 || w >= desc->num_waves)
        return;

    scale = pulseg__wave_scale(desc, &desc->block_table[block_idx]);
    wave = &desc->waves[w];
    for (d = 0; d < 3; ++d)
        amp_hz_per_m[d] = scale * wave->peak[d];
    *wave_id = w;
}
