/**
 * @file pulseg_waves.c
 * @brief Waves: the waveform each axis plays of a block at a position that
 *        plays waves, and where a playout holds them.
 *
 * See pulseg_wave in pulseg_internal.h for what a wave is, and
 * pulseg_materialize_wave() for how it is sampled.
 */

#include <float.h>
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

float pulseg__linear_at(const float *t, const float *v, int n, float x)
{
    int lo, hi;
    float span;

    if (n < 2 || x < t[0] || x > t[n - 1])
        return 0.0f;
    lo = 0;
    hi = n - 1;
    while (hi - lo > 1)
    {
        const int mid = (lo + hi) / 2;
        if (t[mid] <= x)
            lo = mid;
        else
            hi = mid;
    }
    span = t[hi] - t[lo];
    if (span <= 0.0f)
        return v[hi];
    return v[lo] + (v[hi] - v[lo]) * (x - t[lo]) / span;
}

static float axis_value_at(const axis_corners *c, float x)
{
    return pulseg__linear_at(c->t, c->v, c->n, x);
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

/* The amplitude of largest magnitude, with its sign, among the gradient
 * events of @p bte, the first axis on a tie; 0 when it drives none.  What a
 * wave is scaled by. */
static float wave_scale(
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

/* The grid of @p g and the combination on it, times @p scale, into arrays of
 * its own. */
static int grid_copy(
    const pulseg_wave *wave,
    const wave_grid *g,
    float scale,
    float **time_us,
    float *gradient[3])
{
    int a, i;

    *time_us = (float *)PULSEG_ALLOC((size_t)g->n * sizeof(float));
    for (a = 0; a < 3; ++a)
        gradient[a] = (float *)PULSEG_ALLOC((size_t)g->n * sizeof(float));
    if (!*time_us || !gradient[0] || !gradient[1] || !gradient[2])
        return PULSEG_ERR_ALLOC_FAILED;
    for (i = 0; i < g->n; ++i)
    {
        (*time_us)[i] = g->t[i];
        for (a = 0; a < 3; ++a)
            gradient[a][i] = scale * wave_grid_value(wave, g, a, i);
    }
    return PULSEG_SUCCESS;
}

int pulseg__block_gradients(
    const pulseg_sequence_descriptor *desc,
    int block_idx,
    float **time_us,
    float *gradient[3],
    int *n)
{
    const pulseg_block_table_element *bte;
    pulseg_wave wave;
    wave_grid g;
    int rc;

    *time_us = NULL;
    gradient[0] = gradient[1] = gradient[2] = NULL;
    *n = 0;
    if (block_idx < 0 || block_idx >= desc->num_blocks)
        return PULSEG_ERR_INVALID_ARGUMENT;
    bte = &desc->block_table[block_idx];
    if (!pulseg__block_combination(desc, bte, &wave))
        return PULSEG_SUCCESS;
    rc = wave_grid_build(desc, &wave, &g);
    if (PULSEG_FAILED(rc))
        return rc;
    if (g.n > 0)
        rc = grid_copy(&wave, &g, wave_scale(desc, bte), time_us, gradient);
    if (PULSEG_SUCCEEDED(rc))
        *n = g.n;
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

/* The descriptor of subsequence @p subseq_idx where it holds wave
 * @p wave_idx; NULL otherwise. */
static const pulseg_sequence_descriptor *wave_owner(
    const pulseg_collection *coll,
    int subseq_idx,
    int wave_idx)
{
    const pulseg_sequence_descriptor *desc;

    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return NULL;
    desc = &coll->descriptors[subseq_idx];
    return (desc->waves && wave_idx >= 0 && wave_idx < desc->num_waves) ? desc : NULL;
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
    desc = wave_owner(coll, subseq_idx, wave_idx);
    if (!desc)
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

/* Each axis's definition, shape and amplitude, into @p wave and
 * @p amplitude; 0 when the block drives none. */
static int block_axes(
    const pulseg_sequence_descriptor *desc,
    const pulseg_block_table_element *bte,
    pulseg_wave *wave,
    float amplitude[3])
{
    const pulseg_grad_table_element *element;
    int ids[3];
    int d, driven = 0;

    ids[0] = bte->gx_id;
    ids[1] = bte->gy_id;
    ids[2] = bte->gz_id;
    for (d = 0; d < 3; ++d)
    {
        wave->grad_def[d] = -1;
        wave->shape_id[d] = 0;
        amplitude[d] = 0.0f;
        if (ids[d] < 0 || ids[d] >= desc->grad_table_size)
            continue;
        element = &desc->grad_table[ids[d]];
        if (element->id < 0 || element->id >= desc->num_unique_grads)
            continue;
        wave->grad_def[d] = element->id;
        wave->shape_id[d] = element->shape_id;
        amplitude[d] = element->amplitude;
        driven = 1;
    }
    return driven;
}

/* The rotation of @p bte, the identity without one. */
static void block_rotation(
    const pulseg_sequence_descriptor *desc,
    const pulseg_block_table_element *bte,
    pulseg_wave *wave)
{
    const int rotated = bte->rotation_id >= 0 && bte->rotation_id < desc->num_rotations &&
        desc->rotation_matrices != NULL;
    int i;

    wave->rotation_id = rotated ? bte->rotation_id : -1;
    for (i = 0; i < 9; ++i)
        wave->rotation[i] = rotated ? desc->rotation_matrices[bte->rotation_id][i]
                                    : ((i % 4 == 0) ? 1.0f : 0.0f);
}

int pulseg__block_combination(
    const pulseg_sequence_descriptor *desc,
    const pulseg_block_table_element *bte,
    pulseg_wave *wave)
{
    float amplitude[3];
    float scale;
    int d;

    wave->num_points = 0;
    wave->start_us = 0.0f;
    wave->end_us = 0.0f;
    for (d = 0; d < 3; ++d)
    {
        wave->ratio[d] = 0.0f;
        wave->peak[d] = 0.0f;
    }
    block_rotation(desc, bte, wave);
    if (!block_axes(desc, bte, wave, amplitude))
        return 0;
    scale = wave_scale(desc, bte);
    if (scale == 0.0f)
        return 0;
    for (d = 0; d < 3; ++d)
        wave->ratio[d] = amplitude[d] / scale;
    return 1;
}

/* The wave block-table entry @p block_idx plays, -1 where it plays none. */
static int block_wave_index(const pulseg_sequence_descriptor *desc, int block_idx)
{
    int w;

    if (!desc->block_wave || block_idx < 0 || block_idx >= desc->num_blocks)
        return -1;
    w = desc->block_wave[block_idx];
    return (desc->waves && w < desc->num_waves) ? w : -1;
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
    w = block_wave_index(desc, block_idx);
    if (w < 0)
        return;

    scale = wave_scale(desc, &desc->block_table[block_idx]);
    wave = &desc->waves[w];
    for (d = 0; d < 3; ++d)
        amp_hz_per_m[d] = scale * wave->peak[d];
    *wave_id = w;
}

/* One axis of a wave's points, normalised, in arrays of its own. */
static int wave_points(
    const pulseg_sequence_descriptor *desc,
    int wave_idx,
    int out_axis,
    axis_corners *out)
{
    const pulseg_wave *wave = &desc->waves[wave_idx];
    int n = 0;
    int rc = pulseg__wave_materialize(desc, wave, out_axis, NULL, NULL, 0, &n, NULL);

    if (PULSEG_FAILED(rc) || n < 1)
        return rc;
    if (!axis_corners_alloc(out, n))
        return PULSEG_ERR_ALLOC_FAILED;
    rc = pulseg__wave_materialize(desc, wave, out_axis, out->t, out->v, n, &n, NULL);
    if (PULSEG_FAILED(rc))
        axis_corners_free(out);
    return rc;
}

int pulseg_sample_wave(
    const pulseg_collection *coll,
    int subseq_idx,
    int wave_idx,
    int out_axis,
    float start_us,
    float raster_us,
    long num_samples,
    float *out)
{
    const pulseg_sequence_descriptor *desc;
    axis_corners points;
    long i;
    int rc;

    if (!coll || !out)
        return PULSEG_ERR_NULL_POINTER;
    desc = wave_owner(coll, subseq_idx, wave_idx);
    if (!desc || raster_us <= 0.0f || num_samples < 0)
        return PULSEG_ERR_INVALID_ARGUMENT;
    memset(&points, 0, sizeof(points));
    rc = wave_points(desc, wave_idx, out_axis, &points);
    if (PULSEG_FAILED(rc))
        return rc;
    for (i = 0; i < num_samples; ++i)
        out[i] = axis_value_at(&points, start_us + ((float)i + 0.5f) * raster_us);
    axis_corners_free(&points);
    return PULSEG_SUCCESS;
}

void pulseg__wave_cover(float start_us, float end_us, float raster_us, pulseg_wave_region *region)
{
    const double first = floor((double)start_us / (double)raster_us + 1e-4) * (double)raster_us;
    const long n = (long)ceil(((double)end_us - first) / (double)raster_us - 1e-4);

    region->start_us = (float)first;
    region->samples = n > 0 ? n : 1;
}

/* Give @p region an offset on each axis of @p axes, after what @p used
 * holds there, and -1 on the others. */
static void place(pulseg_wave_region *region, int axes, long used[3])
{
    int a;

    for (a = 0; a < 3; ++a)
    {
        region->offset[a] = -1;
        if (axes & (1 << a))
        {
            region->offset[a] = used[a];
            used[a] += region->samples;
        }
    }
}

static int peak_axes(const pulseg_wave *wave)
{
    int a, axes = 0;

    for (a = 0; a < 3; ++a)
        if (wave->peak[a] > 0.0f)
            axes |= 1 << a;
    return axes;
}

static int driven_axes(const pulseg_wave_region *region)
{
    return (region->offset[0] >= 0) + (region->offset[1] >= 0) + (region->offset[2] >= 0);
}

static void free_regions(pulseg_wave_region **regions, int count)
{
    int i;

    if (!regions)
        return;
    for (i = 0; i < count; ++i)
        if (regions[i])
            PULSEG_FREE(regions[i]);
    PULSEG_FREE(regions);
}

void pulseg_free_wave_plan(pulseg_wave_plan *plan)
{
    static const pulseg_wave_plan empty = PULSEG_WAVE_PLAN_INIT;

    if (!plan)
        return;
    free_regions(plan->waves, plan->num_subsequences);
    free_regions(plan->slots, plan->num_segments);
    if (plan->num_waves)
        PULSEG_FREE(plan->num_waves);
    if (plan->num_positions)
        PULSEG_FREE(plan->num_positions);
    *plan = empty;
}

/* @p count empty region tables and their lengths; 1 on success. */
static int alloc_tables(int count, int **lengths, pulseg_wave_region ***tables)
{
    int i;

    if (count <= 0)
        return 1;
    *lengths = (int *)PULSEG_ALLOC((size_t)count * sizeof(int));
    *tables = (pulseg_wave_region **)PULSEG_ALLOC((size_t)count * sizeof(pulseg_wave_region *));
    if (!*lengths || !*tables)
        return 0;
    for (i = 0; i < count; ++i)
    {
        (*lengths)[i] = 0;
        (*tables)[i] = NULL;
    }
    return 1;
}

static pulseg_wave_region *alloc_regions(int count)
{
    return (pulseg_wave_region *)PULSEG_ALLOC((size_t)count * sizeof(pulseg_wave_region));
}

/* RESIDENT: a region per wave of each subsequence, laid out in order. */
static int lay_out_waves(const pulseg_collection *coll, float raster_us, pulseg_wave_plan *plan)
{
    int s, w;

    plan->num_subsequences = coll->num_subsequences;
    if (!alloc_tables(coll->num_subsequences, &plan->num_waves, &plan->waves))
        return PULSEG_ERR_ALLOC_FAILED;
    for (s = 0; s < coll->num_subsequences; ++s)
    {
        const pulseg_sequence_descriptor *desc = &coll->descriptors[s];
        if (desc->num_waves <= 0 || !desc->waves)
            continue;
        plan->waves[s] = alloc_regions(desc->num_waves);
        if (!plan->waves[s])
            return PULSEG_ERR_ALLOC_FAILED;
        plan->num_waves[s] = desc->num_waves;
        for (w = 0; w < desc->num_waves; ++w)
        {
            pulseg__wave_cover(
                desc->waves[w].start_us, desc->waves[w].end_us, raster_us, &plan->waves[s][w]);
            place(&plan->waves[s][w], peak_axes(&desc->waves[w]), plan->resident_samples);
        }
    }
    return PULSEG_SUCCESS;
}

/* The two slots of position @p b of segment @p g, empty where it plays no
 * wave. */
static int lay_out_position(
    const pulseg_collection *coll,
    int g,
    int b,
    float raster_us,
    pulseg_wave_region slot[2],
    long used[3])
{
    pulseg_block_info block = PULSEG_BLOCK_INFO_INIT;
    int h;
    const int rc = pulseg_get_block_info(coll, &block, g, b);

    if (PULSEG_FAILED(rc))
        return rc;
    for (h = 0; h < 2; ++h)
    {
        slot[h].samples = 0;
        slot[h].start_us = 0.0f;
        if (block.wave_points > 0)
            pulseg__wave_cover(block.wave_start_us, block.wave_end_us, raster_us, &slot[h]);
        place(&slot[h], block.wave_points > 0 ? block.wave_axes : 0, used);
    }
    return PULSEG_SUCCESS;
}

/* STREAMED: two slots per position of each segment that plays waves. */
static int lay_out_slots(const pulseg_collection *coll, float raster_us, pulseg_wave_plan *plan)
{
    pulseg_segment_info seg = PULSEG_SEGMENT_INFO_INIT;
    int g, b;

    plan->num_segments = coll->total_unique_segments;
    if (!alloc_tables(plan->num_segments, &plan->num_positions, &plan->slots))
        return PULSEG_ERR_ALLOC_FAILED;
    for (g = 0; g < plan->num_segments; ++g)
    {
        int rc = pulseg_get_segment_info(coll, &seg, g);
        if (PULSEG_FAILED(rc))
            return rc;
        if (seg.num_blocks <= 0)
            continue;
        plan->slots[g] = alloc_regions(2 * seg.num_blocks);
        if (!plan->slots[g])
            return PULSEG_ERR_ALLOC_FAILED;
        plan->num_positions[g] = seg.num_blocks;
        for (b = 0; b < seg.num_blocks && PULSEG_SUCCEEDED(rc); ++b)
            rc = lay_out_position(
                coll, g, b, raster_us, &plan->slots[g][2 * b], plan->streamed_samples);
        if (PULSEG_FAILED(rc))
            return rc;
    }
    return PULSEG_SUCCESS;
}

/* How long loading a segment's waves into its slots takes. */
static float segment_load_us(const pulseg_wave_plan *plan, int g, float load_us_per_sample)
{
    float total = 0.0f;
    int b;

    if (g < 0 || g >= plan->num_segments || !plan->slots[g])
        return 0.0f;
    for (b = 0; b < plan->num_positions[g]; ++b)
        total += (float)plan->slots[g][2 * b].samples *
            (float)driven_axes(&plan->slots[g][2 * b]) * load_us_per_sample;
    return total;
}

/* The scan's segment instances, walked entry by entry across the chain. */
typedef struct instance_walk
{
    int last;          /* segment of the previous entry, -1 at a subsequence's start */
    int local;         /* segment of the current entry */
    int position;      /* the current entry's position in its instance */
    int started;       /* 1 once an instance has begun */
    int has_previous;  /* 1 once an instance has ended */
    float previous_us; /* duration of the instance before the current one */
    float current_us;  /* duration of the current instance so far */
} instance_walk;

/* Account entry @p n of @p desc; 1 when it starts a segment instance. */
static int instance_starts(instance_walk *walk, const pulseg_sequence_descriptor *desc, int n)
{
    const int local = pulseg__exec_seg_id(desc, n);
    const int block = pulseg__exec_block_idx(desc, n);
    int blocks;

    if (local < 0 || local >= desc->num_unique_segments)
    {
        walk->last = -1;
        return 0;
    }
    blocks = desc->segment_definitions[local].num_blocks;
    walk->position = (local == walk->last && blocks > 0) ? (walk->position + 1) % blocks : 0;
    walk->last = local;
    walk->local = local;
    if (walk->position == 0)
    {
        walk->has_previous = walk->started;
        walk->previous_us = walk->current_us;
        walk->started = 1;
        walk->current_us = 0.0f;
    }
    if (block >= 0 && block < desc->num_blocks)
        walk->current_us += (float)pulseg__played_duration_us(desc, block);
    return walk->position == 0;
}

/* Record the spare time loading segment @p g leaves while the instance
 * before it plays, where it is the least yet. */
static void note_spare(
    pulseg_wave_plan *plan,
    const pulseg_wave_budget *budget,
    const instance_walk *walk,
    int g,
    int s,
    int n)
{
    float spare;

    if (!walk->has_previous)
        return;
    spare = budget->headroom * walk->previous_us -
        segment_load_us(plan, g, budget->load_us_per_sample);
    if (spare < plan->least_spare_us)
    {
        plan->least_spare_us = spare;
        plan->tightest_subseq = s;
        plan->tightest_position = n;
    }
}

int pulseg__global_segment(const pulseg_collection *coll, int s, int local)
{
    if (!coll->seg_local_to_global)
        return local;
    return coll->seg_local_to_global[coll->subsequence_info[s].segment_id_offset + local];
}

static int scan_loop_loaded(const pulseg_collection *coll)
{
    int s;

    for (s = 0; s < coll->num_subsequences; ++s)
        if (coll->descriptors[s].exec_stream_len <= 0 || coll->descriptors[s].num_exec_runs <= 0)
            return 0;
    return 1;
}

/* Walk the scan's segment instances, each loaded while the one before it
 * plays, for the least spare time.  0 when the execution stream is not
 * loaded. */
static int check_loading(
    const pulseg_collection *coll,
    const pulseg_wave_budget *budget,
    pulseg_wave_plan *plan)
{
    instance_walk walk;
    int s, n;

    if (!scan_loop_loaded(coll))
        return 0;
    memset(&walk, 0, sizeof(walk));
    plan->least_spare_us = FLT_MAX;
    for (s = 0; s < coll->num_subsequences; ++s)
    {
        const pulseg_sequence_descriptor *desc = &coll->descriptors[s];
        walk.last = -1;
        for (n = 0; n < desc->exec_stream_len; ++n)
            if (instance_starts(&walk, desc, n))
                note_spare(plan, budget, &walk, pulseg__global_segment(coll, s, walk.local), s, n);
    }
    if (plan->tightest_subseq < 0)
        plan->least_spare_us = 0.0f;
    return 1;
}

/* PULSEG_WAVES_* the budget affords, or -1 when neither layout fits. */
static int affordable_mode(const pulseg_wave_plan *plan, long max_samples)
{
    int a, any = 0, resident = 1, streamed = 1;

    for (a = 0; a < 3; ++a)
    {
        any |= plan->resident_samples[a] > 0;
        resident &= plan->resident_samples[a] <= max_samples;
        streamed &= plan->streamed_samples[a] <= max_samples;
    }
    if (!any)
        return PULSEG_WAVES_NONE;
    if (resident)
        return PULSEG_WAVES_RESIDENT;
    return streamed ? PULSEG_WAVES_STREAMED : -1;
}

static void adopt(pulseg_wave_plan *plan, int mode)
{
    const long *held = (mode == PULSEG_WAVES_STREAMED) ? plan->streamed_samples
                                                       : plan->resident_samples;
    int a;

    plan->mode = mode;
    for (a = 0; a < 3; ++a)
        plan->samples[a] = held[a];
}

static int wave_memory_shortfall(
    const pulseg_wave_plan *plan,
    const pulseg_wave_budget *budget,
    pulseg_diagnostic *diag)
{
    if (diag)
    {
        diag->code = PULSEG_ERR_WAVE_MEMORY;
        pulseg__diag_printf(
            diag,
            "waves take %ld, %ld, %ld samples at once and %ld, %ld, %ld in two slots "
            "per position, against %ld per axis",
            plan->resident_samples[0], plan->resident_samples[1], plan->resident_samples[2],
            plan->streamed_samples[0], plan->streamed_samples[1], plan->streamed_samples[2],
            budget->max_samples);
    }
    return PULSEG_ERR_WAVE_MEMORY;
}

static int wave_loading_shortfall(const pulseg_wave_plan *plan, pulseg_diagnostic *diag)
{
    if (diag)
    {
        diag->code = PULSEG_ERR_WAVE_LOADING;
        pulseg__diag_printf(
            diag,
            "subsequence %d, execution-stream position %d: loading its waves "
            "overruns the playout before it by %.1f us",
            plan->tightest_subseq, plan->tightest_position, (double)(-plan->least_spare_us));
    }
    return PULSEG_ERR_WAVE_LOADING;
}

static int plan_arguments(
    const pulseg_collection *coll,
    const pulseg_wave_budget *budget,
    pulseg_wave_plan *plan)
{
    static const pulseg_wave_plan empty = PULSEG_WAVE_PLAN_INIT;

    if (!coll || !budget || !plan)
        return PULSEG_ERR_NULL_POINTER;
    *plan = empty;
    if (budget->raster_us <= 0.0f || budget->max_samples < 0 || budget->load_us_per_sample < 0.0f ||
        budget->headroom <= 0.0f)
        return PULSEG_ERR_INVALID_ARGUMENT;
    return PULSEG_SUCCESS;
}

int pulseg_plan_waves(
    const pulseg_collection *coll,
    const pulseg_wave_budget *budget,
    pulseg_wave_plan *plan,
    pulseg_diagnostic *diag)
{
    int mode;
    int rc = plan_arguments(coll, budget, plan);

    if (PULSEG_SUCCEEDED(rc))
        rc = lay_out_waves(coll, budget->raster_us, plan);
    if (PULSEG_SUCCEEDED(rc))
        rc = lay_out_slots(coll, budget->raster_us, plan);
    if (PULSEG_FAILED(rc))
        return rc;

    mode = affordable_mode(plan, budget->max_samples);
    if (mode < 0)
        return wave_memory_shortfall(plan, budget, diag);
    adopt(plan, mode);
    if (mode == PULSEG_WAVES_STREAMED && budget->load_us_per_sample > 0.0f)
        plan->loading_checked = check_loading(coll, budget, plan);
    if (plan->loading_checked && plan->least_spare_us < 0.0f)
        return wave_loading_shortfall(plan, diag);
    return PULSEG_SUCCESS;
}
