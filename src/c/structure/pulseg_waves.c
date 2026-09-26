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

static int axis_corners_alloc(axis_corners *c, int n)
{
    c->t = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    c->v = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    if (c->t && c->v)
    {
        c->n = n;
        return 1;
    }
    axis_corners_free(c);
    return 0;
}

static int trapezoid_corners(const pulseg_grad_definition *gd, axis_corners *out)
{
    const float delay = (float)gd->delay;
    const int k = (gd->flat_time_or_unused > 0) ? 4 : 3;

    if (!axis_corners_alloc(out, k))
        return 0;
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
    return 1;
}

/* On a time shape of as many samples the first and last samples are the
 * event's edges.  On the raster the samples sit at interval centres, and the
 * event holds its end samples over the half intervals at its edges, which
 * keeps the area the samples give. */
static int arbitrary_corners(
    const pulseg_grad_definition *gd,
    const pulseq_shape *samples,
    const pulseq_shape *times,
    float raster,
    axis_corners *out)
{
    const float delay = (float)gd->delay;
    const int ns = samples->num_samples;
    const int has_time = times->num_samples == ns;
    int i, k;

    if (!axis_corners_alloc(out, has_time ? ns : ns + 2))
        return 0;
    k = 0;
    if (!has_time)
    {
        out->t[k] = delay;
        out->v[k++] = samples->samples[0];
    }
    for (i = 0; i < ns; ++i)
    {
        out->t[k] = delay + (has_time ? times->samples[i] : ((float)i + 0.5f) * raster);
        out->v[k++] = samples->samples[i];
    }
    if (!has_time)
    {
        out->t[k] = delay + (float)ns * raster;
        out->v[k] = samples->samples[ns - 1];
    }
    return 1;
}

/* Shape @p id (1-based) decompressed at @p scale; 1 on success, an absent
 * shape leaving @p out empty. */
static int decompressed(
    const pulseg_sequence_descriptor *desc,
    int id,
    float scale,
    pulseq_shape *out)
{
    if (id < 1 || id > desc->num_shapes)
        return 1;
    return pulseq_decompress_shape(out, &desc->shapes[id - 1], scale);
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
    int ok;

    out->t = NULL;
    out->v = NULL;
    out->n = 0;
    if (grad_def_id < 0 || grad_def_id >= desc->num_unique_grads)
        return 1;
    gd = &desc->grad_definitions[grad_def_id];
    if (gd->type == 0)
        return trapezoid_corners(gd, out);

    memset(&samples, 0, sizeof(samples));
    memset(&times, 0, sizeof(times));
    ok = decompressed(desc, shape_id, 1.0f, &samples) &&
        decompressed(desc, gd->unused_or_time_shape_id, desc->grad_raster_us, &times);
    if (ok && samples.num_samples > 0)
        ok = arbitrary_corners(gd, &samples, &times, desc->grad_raster_us, out);
    if (samples.samples)
        PULSEG_FREE(samples.samples);
    if (times.samples)
        PULSEG_FREE(times.samples);
    return ok;
}

static float axis_value_at(const axis_corners *c, float x)
{
    int lo, hi;
    float span;

    if (c->n < 2 || x < c->t[0] || x > c->t[c->n - 1])
        return 0.0f;
    lo = 0;
    hi = c->n - 1;
    while (hi - lo > 1)
    {
        const int mid = (lo + hi) / 2;
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

/* The points a wave is materialised at, over the three axes it combines. */
typedef struct wave_grid
{
    axis_corners axis[3];
    float *t;
    int n;
    int centred; /* raster centres between two held edges */
} wave_grid;

static void wave_grid_free(wave_grid *g)
{
    int d;

    for (d = 0; d < 3; ++d)
        axis_corners_free(&g->axis[d]);
    if (g->t)
        PULSEG_FREE(g->t);
    g->t = NULL;
    g->n = 0;
}

/* 1 and the earliest start and latest end of the driven axes; 0 when none is. */
static int axes_span(const axis_corners axis[3], float *lo, float *hi)
{
    int d, have = 0;

    for (d = 0; d < 3; ++d)
    {
        if (axis[d].n < 1)
            continue;
        if (!have || axis[d].t[0] < *lo)
            *lo = axis[d].t[0];
        if (!have || axis[d].t[axis[d].n - 1] > *hi)
            *hi = axis[d].t[axis[d].n - 1];
        have = 1;
    }
    return have;
}

/* The centres of the raster from the earliest start, rounded down onto it,
 * to the latest end, and the two edges the wave holds its end values to. */
static int centre_grid(wave_grid *g, float raster)
{
    float lo = 0.0f;
    float hi = 0.0f;
    int i, m;

    if (!axes_span(g->axis, &lo, &hi) || hi <= lo || raster <= 0.0f)
        return PULSEG_SUCCESS;
    lo = (float)floor((double)(lo / raster) + 1e-6) * raster;
    m = (int)ceil((double)((hi - lo) / raster) - 1e-6);
    if (m < 1)
        m = 1;
    g->t = (float *)PULSEG_ALLOC((size_t)(m + 2) * sizeof(float));
    if (!g->t)
        return PULSEG_ERR_ALLOC_FAILED;
    g->t[0] = lo;
    for (i = 0; i < m; ++i)
        g->t[i + 1] = lo + ((float)i + 0.5f) * raster;
    g->t[m + 1] = lo + (float)m * raster;
    g->n = m + 2;
    return PULSEG_SUCCESS;
}

/* Each axis is linear between its corners, so the union of the corner times
 * carries the combination exactly. */
static int corner_grid(wave_grid *g)
{
    const int total = g->axis[0].n + g->axis[1].n + g->axis[2].n;
    int d, i, k;

    if (total == 0)
        return PULSEG_SUCCESS;
    g->t = (float *)PULSEG_ALLOC((size_t)total * sizeof(float));
    if (!g->t)
        return PULSEG_ERR_ALLOC_FAILED;
    k = 0;
    for (d = 0; d < 3; ++d)
        for (i = 0; i < g->axis[d].n; ++i)
            g->t[k++] = g->axis[d].t[i];
    qsort(g->t, (size_t)total, sizeof(float), float_cmp);
    k = 0;
    for (i = 0; i < total; ++i)
        if (k == 0 || g->t[i] > g->t[k - 1])
            g->t[k++] = g->t[i];
    g->n = k;
    return PULSEG_SUCCESS;
}

static int wave_grid_build(
    const pulseg_sequence_descriptor *desc,
    const pulseg_wave *wave,
    wave_grid *g)
{
    int d, rc;

    memset(g, 0, sizeof(*g));
    rc = PULSEG_SUCCESS;
    for (d = 0; d < 3 && PULSEG_SUCCEEDED(rc); ++d)
        if (!axis_corners_build(desc, wave->grad_def[d], wave->shape_id[d], &g->axis[d]))
            rc = PULSEG_ERR_ALLOC_FAILED;
    if (PULSEG_SUCCEEDED(rc))
    {
        g->centred = wave_on_raster_centres(desc, wave);
        rc = g->centred ? centre_grid(g, desc->grad_raster_us) : corner_grid(g);
    }
    if (PULSEG_FAILED(rc))
        wave_grid_free(g);
    return rc;
}

/* The combination on output axis @p out_axis at point @p i; on raster
 * centres the edges hold the first and last centres. */
static float wave_grid_value(const pulseg_wave *wave, const wave_grid *g, int out_axis, int i)
{
    float x = g->t[i];
    float v = 0.0f;
    int d;

    if (g->centred && i == 0)
        x = g->t[1];
    else if (g->centred && i == g->n - 1)
        x = g->t[g->n - 2];
    for (d = 0; d < 3; ++d)
        v += wave->rotation[out_axis * 3 + d] * wave->ratio[d] * axis_value_at(&g->axis[d], x);
    return v;
}

/* The combination at every point of @p g, written to the arrays when they
 * are given, for as many points as they hold; the largest magnitude. */
static float wave_grid_fill(
    const pulseg_wave *wave,
    const wave_grid *g,
    int out_axis,
    float *out_time_us,
    float *out_amp,
    int max_points)
{
    float peak = 0.0f;
    int i;

    for (i = 0; i < g->n; ++i)
    {
        const float v = wave_grid_value(wave, g, out_axis, i);
        if ((float)fabs((double)v) > peak)
            peak = (float)fabs((double)v);
        if (out_time_us && out_amp && i < max_points)
        {
            out_time_us[i] = g->t[i];
            out_amp[i] = v;
        }
    }
    return peak;
}

static int normalised(float *amp, int n, float peak)
{
    if (peak > 0.0f)
    {
        int i;
        for (i = 0; i < n; ++i)
            amp[i] /= peak;
    }
    return PULSEG_SUCCESS;
}

static int materialize_arguments(
    const pulseg_sequence_descriptor *desc,
    const pulseg_wave *wave,
    int out_axis,
    const int *out_num_points)
{
    if (!desc || !wave || !out_num_points)
        return PULSEG_ERR_NULL_POINTER;
    if (out_axis < PULSEG_GRAD_AXIS_X || out_axis > PULSEG_GRAD_AXIS_Z)
        return PULSEG_ERR_INVALID_ARGUMENT;
    return PULSEG_SUCCESS;
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
    wave_grid g;
    float peak;
    int rc;

    rc = materialize_arguments(desc, wave, out_axis, out_num_points);
    if (PULSEG_FAILED(rc))
        return rc;
    rc = wave_grid_build(desc, wave, &g);
    if (PULSEG_FAILED(rc))
        return rc;

    peak = wave_grid_fill(wave, &g, out_axis, out_time_us, out_amp, max_points);
    *out_num_points = g.n;
    if (out_peak)
        *out_peak = peak;
    if (out_time_us && out_amp)
        rc = (max_points < g.n) ? PULSEG_ERR_INDEX : normalised(out_amp, g.n, peak);
    wave_grid_free(&g);
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
