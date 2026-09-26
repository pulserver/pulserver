/**
 * @file pulseg_waves.c
 * @brief Waves: the waveform each axis plays of a block at a position that
 *        plays waves, and where a playout holds them.
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

/* @p count region tables, each a copy of @p from's, with @p lengths[i]
 * times @p per regions; NULL on failure. */
static pulseg_wave_region **copy_regions(
    pulseg_wave_region *const *from,
    const int *lengths,
    int count,
    int per)
{
    pulseg_wave_region **to;
    int i, ok = 1;

    if (!from || !lengths || count <= 0)
        return NULL;
    to = (pulseg_wave_region **)PULSEG_ALLOC((size_t)count * sizeof(pulseg_wave_region *));
    if (!to)
        return NULL;
    for (i = 0; i < count; ++i)
    {
        const size_t n = (size_t)lengths[i] * (size_t)per;
        to[i] = NULL;
        if (n == 0 || !from[i])
            continue;
        to[i] = (pulseg_wave_region *)PULSEG_ALLOC(n * sizeof(pulseg_wave_region));
        if (!to[i])
            ok = 0;
        else
            memcpy(to[i], from[i], n * sizeof(pulseg_wave_region));
    }
    if (!ok)
    {
        free_regions(to, count);
        return NULL;
    }
    return to;
}

static int *copy_ints(const int *from, int count)
{
    int *to;

    if (count <= 0 || !from)
        return NULL;
    to = (int *)PULSEG_ALLOC((size_t)count * sizeof(int));
    if (to)
        memcpy(to, from, (size_t)count * sizeof(int));
    return to;
}

/* 1 when @p to holds every table @p from does. */
static int copied(const pulseg_wave_plan *from, const pulseg_wave_plan *to)
{
    return (!from->num_waves || to->num_waves) && (!from->num_positions || to->num_positions) &&
        (!from->waves || to->waves) && (!from->slots || to->slots);
}

/* A deep copy of @p from into @p to, which is overwritten; released with
 * pulseg_free_wave_plan(). */
static int copy_wave_plan(const pulseg_wave_plan *from, pulseg_wave_plan *to)
{
    *to = *from;
    to->num_waves = copy_ints(from->num_waves, from->num_subsequences);
    to->num_positions = copy_ints(from->num_positions, from->num_segments);
    to->waves = copy_regions(from->waves, to->num_waves, from->num_subsequences, 1);
    to->slots =
        copy_regions(from->slots, to->num_positions, from->num_segments, from->budget.slots);
    if (copied(from, to))
        return PULSEG_SUCCESS;
    pulseg_free_wave_plan(to);
    return PULSEG_ERR_ALLOC_FAILED;
}

static int same_value(float a, float b)
{
    const double scale = fabs((double)a) > 1.0 ? fabs((double)a) : 1.0;

    return fabs((double)a - (double)b) <= 1e-6 * scale;
}

static int same_budget(const pulseg_wave_budget *a, const pulseg_wave_budget *b)
{
    return a->max_samples == b->max_samples && a->slots == b->slots &&
        same_value(a->raster_us, b->raster_us) &&
        same_value(a->load_us_per_sample, b->load_us_per_sample) &&
        same_value(a->headroom, b->headroom);
}

static int another_budget(const pulseg_wave_budget *made, const pulseg_wave_budget *given,
                          pulseg_diagnostic *diag)
{
    if (diag)
    {
        diag->code = PULSEG_ERR_WAVE_BUDGET;
        pulseg__diag_printf(
            diag,
            "laid out for %ld samples, %g us raster, %g us per sample, headroom %g, %d slots; "
            "given %ld samples, %g us raster, %g us per sample, headroom %g, %d slots",
            made->max_samples, (double)made->raster_us, (double)made->load_us_per_sample,
            (double)made->headroom, made->slots, given->max_samples, (double)given->raster_us,
            (double)given->load_us_per_sample, (double)given->headroom, given->slots);
    }
    return PULSEG_ERR_WAVE_BUDGET;
}

int pulseg_get_wave_plan(
    const pulseg_collection *coll,
    const pulseg_wave_budget *budget,
    pulseg_wave_plan *plan,
    pulseg_diagnostic *diag)
{
    static const pulseg_wave_plan empty = PULSEG_WAVE_PLAN_INIT;

    if (!coll || !plan)
        return PULSEG_ERR_NULL_POINTER;
    *plan = empty;
    if (budget && coll->wave_plan.mode != PULSEG_WAVES_NONE &&
        !same_budget(&coll->wave_plan.budget, budget))
        return another_budget(&coll->wave_plan.budget, budget, diag);
    return copy_wave_plan(&coll->wave_plan, plan);
}

int pulseg__global_segment(const pulseg_collection *coll, int s, int local)
{
    if (!coll->seg_local_to_global)
        return local;
    return coll->seg_local_to_global[coll->subsequence_info[s].segment_id_offset + local];
}
