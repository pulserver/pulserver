/**
 * @file pulseg_playout.c
 * @brief The two stages of a segmented playout over a backend; see
 *        pulseg_playout.h.
 */

#include <string.h>

#include "pulseg_internal.h"
#include "pulseg.h"

/* Loading waves into waveform memory: on which raster, through which
 * backend, with room for the longest region the layout holds. */
typedef struct wave_loader
{
    const pulseg_collection *coll;
    const pulseg_playout_backend *backend;
    float raster_us;
    float *scratch;
} wave_loader;

static long longest_region(const pulseg_wave_plan *plan)
{
    long longest = 1;
    int i, j;

    for (i = 0; i < plan->num_subsequences; ++i)
        for (j = 0; j < plan->num_waves[i]; ++j)
            if (plan->waves[i][j].samples > longest)
                longest = plan->waves[i][j].samples;
    for (i = 0; i < plan->num_segments; ++i)
        for (j = 0; j < 2 * plan->num_positions[i]; ++j)
            if (plan->slots[i][j].samples > longest)
                longest = plan->slots[i][j].samples;
    return longest;
}

static int loader_open(
    wave_loader *l,
    const pulseg_collection *coll,
    const pulseg_playout_backend *backend,
    float raster_us,
    const pulseg_wave_plan *plan)
{
    l->coll = coll;
    l->backend = backend;
    l->raster_us = raster_us;
    l->scratch = (float *)PULSEG_ALLOC((size_t)longest_region(plan) * sizeof(float));
    return l->scratch ? PULSEG_SUCCESS : PULSEG_ERR_ALLOC_FAILED;
}

static void loader_close(wave_loader *l)
{
    if (l->scratch)
        PULSEG_FREE(l->scratch);
    l->scratch = NULL;
}

/* Sample wave @p w of subsequence @p s over @p region and load each axis it
 * drives where the region holds it. */
static int load_wave(const wave_loader *l, int s, int w, const pulseg_wave_region *region)
{
    pulseg_wave_load load;
    int a, rc = PULSEG_SUCCESS;

    load.subsequence = s;
    load.wave = w;
    load.count = region->samples;
    load.samples = l->scratch;
    for (a = 0; a < 3 && PULSEG_SUCCEEDED(rc); ++a)
    {
        if (region->offset[a] < 0)
            continue;
        load.axis = a;
        load.offset = region->offset[a];
        rc = pulseg_sample_wave(
            l->coll, s, w, a, region->start_us, l->raster_us, region->samples, l->scratch);
        if (PULSEG_SUCCEEDED(rc) && l->backend->load_wave)
            rc = l->backend->load_wave(l->backend->ctx, &load);
    }
    return rc;
}

static int playout_arguments(
    const pulseg_collection *coll,
    const pulseg_wave_budget *budget,
    const pulseg_playout_backend *backend)
{
    return (coll && budget && backend) ? PULSEG_SUCCESS : PULSEG_ERR_NULL_POINTER;
}

/* ================================================================== */
/*  First stage                                                       */
/* ================================================================== */

/* The span every wave of a resident layout at position @p info covers, in
 * both entries of @p slot, held nowhere. */
static void resident_span(const pulseg_block_info *info, float raster_us, pulseg_wave_region slot[2])
{
    int h, a;

    for (h = 0; h < 2; ++h)
    {
        pulseg__wave_cover(info->wave_start_us, info->wave_end_us, raster_us, &slot[h]);
        for (a = 0; a < 3; ++a)
            slot[h].offset[a] = -1;
    }
}

static int prepare_positions(
    const wave_loader *l,
    const pulseg_wave_plan *plan,
    int g,
    int num_blocks)
{
    pulseg_wave_region span[2];
    int b, rc = PULSEG_SUCCESS;

    for (b = 0; b < num_blocks && PULSEG_SUCCEEDED(rc); ++b)
    {
        pulseg_block_info info = PULSEG_BLOCK_INFO_INIT;
        const pulseg_wave_region *slot = NULL;

        rc = pulseg_get_block_info(l->coll, &info, g, b);
        if (PULSEG_FAILED(rc))
            break;
        if (info.wave_points > 0 && plan->mode == PULSEG_WAVES_STREAMED)
            slot = &plan->slots[g][2 * b];
        else if (info.wave_points > 0)
        {
            resident_span(&info, l->raster_us, span);
            slot = span;
        }
        if (l->backend->prepare_block)
            rc = l->backend->prepare_block(l->backend->ctx, g, b, &info, slot);
    }
    return rc;
}

static int prepare_segments(const wave_loader *l, const pulseg_wave_plan *plan)
{
    int g, rc = PULSEG_SUCCESS;

    for (g = 0; g < l->coll->total_unique_segments && PULSEG_SUCCEEDED(rc); ++g)
    {
        pulseg_segment_info info = PULSEG_SEGMENT_INFO_INIT;

        rc = pulseg_get_segment_info(l->coll, &info, g);
        if (PULSEG_SUCCEEDED(rc) && l->backend->prepare_segment)
            rc = l->backend->prepare_segment(l->backend->ctx, g, &info);
        if (PULSEG_SUCCEEDED(rc))
            rc = prepare_positions(l, plan, g, info.num_blocks);
    }
    return rc;
}

static int load_resident(const wave_loader *l, const pulseg_wave_plan *plan)
{
    int s, w, rc = PULSEG_SUCCESS;

    for (s = 0; s < plan->num_subsequences && PULSEG_SUCCEEDED(rc); ++s)
        for (w = 0; w < plan->num_waves[s] && PULSEG_SUCCEEDED(rc); ++w)
            rc = load_wave(l, s, w, &plan->waves[s][w]);
    return rc;
}

int pulseg_playout_prepare(
    const pulseg_collection *coll,
    const pulseg_wave_budget *budget,
    const pulseg_playout_backend *backend,
    pulseg_diagnostic *diag)
{
    pulseg_wave_plan plan = PULSEG_WAVE_PLAN_INIT;
    wave_loader loader;
    int rc = playout_arguments(coll, budget, backend);

    memset(&loader, 0, sizeof(loader));
    if (PULSEG_SUCCEEDED(rc))
        rc = pulseg_plan_waves(coll, budget, &plan, diag);
    if (PULSEG_SUCCEEDED(rc))
        rc = loader_open(&loader, coll, backend, budget->raster_us, &plan);
    if (PULSEG_SUCCEEDED(rc) && backend->reserve_waves)
        rc = backend->reserve_waves(backend->ctx, &plan);
    if (PULSEG_SUCCEEDED(rc))
        rc = prepare_segments(&loader, &plan);
    if (PULSEG_SUCCEEDED(rc) && plan.mode == PULSEG_WAVES_RESIDENT)
        rc = load_resident(&loader, &plan);
    loader_close(&loader);
    pulseg_free_wave_plan(&plan);
    return rc;
}

/* ================================================================== */
/*  Second stage                                                      */
/* ================================================================== */

/* The scan loop's walk of the execution stream. */
typedef struct scan_walk
{
    const pulseg_wave_plan *plan;
    wave_loader loader;
    int prescan;       /* the subsequence the prescan plays, -1 in the scan */
    int readouts_left; /* before the prescan ends                          */
    int done;          /* 1 once the prescan has its readouts              */
    int *instances;    /* [global segment] instances played so far         */
    int subsequence;   /* of the previous entry                            */
    int local;         /* local segment of the previous entry, -1 at a
                          subsequence's start                              */
    int position;      /* the current entry's position in its instance     */
    pulseg_playout_segment segment; /* the instance being played           */
} scan_walk;

static int stream_loaded(const pulseg_collection *coll, pulseg_diagnostic *diag)
{
    int s;

    for (s = 0; s < coll->num_subsequences; ++s)
        if (coll->descriptors[s].exec_stream_len <= 0)
        {
            if (diag)
            {
                diag->code = PULSEG_ERR_INVALID_ARGUMENT;
                pulseg__diag_printf(diag, "subsequence %d: no execution stream loaded", s);
            }
            return PULSEG_ERR_INVALID_ARGUMENT;
        }
    return PULSEG_SUCCESS;
}

/* The readouts a prescan of subsequence @p s acquires before it ends. */
static int prescan_readouts(
    const pulseg_collection *coll,
    int s,
    const pulseg_playout_options *options)
{
    const int declared = coll->descriptors[s].num_gain_cal_readouts;

    if (options->prescan_readouts > 0)
        return options->prescan_readouts;
    return declared > 1 ? declared : 1;
}

static int walk_open(
    scan_walk *w,
    const pulseg_collection *coll,
    const pulseg_wave_plan *plan,
    const pulseg_playout_options *options)
{
    const int segments = coll->total_unique_segments > 0 ? coll->total_unique_segments : 1;

    w->plan = plan;
    w->prescan = -1;
    w->subsequence = -1;
    w->local = -1;
    if (options && options->prescan_subsequence >= 0)
    {
        if (options->prescan_subsequence >= coll->num_subsequences)
            return PULSEG_ERR_INVALID_ARGUMENT;
        w->prescan = options->prescan_subsequence;
        w->readouts_left = prescan_readouts(coll, w->prescan, options);
    }
    w->instances = (int *)PULSEG_ALLOC((size_t)segments * sizeof(int));
    if (!w->instances)
        return PULSEG_ERR_ALLOC_FAILED;
    memset(w->instances, 0, (size_t)segments * sizeof(int));
    return PULSEG_SUCCESS;
}

static void walk_close(scan_walk *w)
{
    if (w->instances)
        PULSEG_FREE(w->instances);
    w->instances = NULL;
    loader_close(&w->loader);
}

/* Account entry @p n of subsequence @p s; 0 when it plays in no segment. */
static int track_position(scan_walk *w, const pulseg_sequence_descriptor *desc, int s, int n)
{
    const int local = pulseg__exec_seg_id(desc, n);
    int blocks;

    if (s != w->subsequence)
    {
        w->subsequence = s;
        w->local = -1;
    }
    if (local < 0 || local >= desc->num_unique_segments)
    {
        w->local = -1;
        return 0;
    }
    blocks = desc->segment_definitions[local].num_blocks;
    w->position = (local == w->local && blocks > 0) ? (w->position + 1) % blocks : 0;
    w->local = local;
    return 1;
}

/* 1 when entry @p n, at @p position of an instance of @p blocks, is its last. */
static int instance_ends(const pulseg_sequence_descriptor *desc, int n, int position, int blocks)
{
    return position >= blocks - 1 || n + 1 >= desc->exec_stream_len ||
        pulseg__exec_seg_id(desc, n + 1) != pulseg__exec_seg_id(desc, n);
}

/* 1 where a block of the instance of local segment @p local from entry @p n
 * waits for a physiological trigger input.  The segment's own trigger_id
 * says only that one of its instances does: a trigger delay and a plain
 * delay share a definition. */
static int instance_awaits_trigger(const pulseg_sequence_descriptor *desc, int n, int local)
{
    const int blocks = desc->segment_definitions[local].num_blocks;
    int k;

    for (k = 0; k < blocks && n + k < desc->exec_stream_len; ++k)
    {
        const int block = pulseg__exec_block_idx(desc, n + k);
        const int t = (block >= 0 && block < desc->num_blocks)
            ? desc->block_table[block].digitalout_id
            : -1;
        if (pulseg__exec_seg_id(desc, n + k) != local)
            break;
        if (t >= 0 && t < desc->num_triggers &&
            desc->trigger_events[t].trigger_type == PULSEG_TRIGGER_TYPE_INPUT)
            return 1;
    }
    return 0;
}

/* 1 where the prescription rotation turns the instance from entry @p n: its
 * blocks do not carry NOROT.  Segments break where NOROT changes, so its
 * first block speaks for all; the segment's own flags join its instances. */
static int instance_rotates(const pulseg_sequence_descriptor *desc, int n)
{
    const int block = pulseg__exec_block_idx(desc, n);

    return !(block >= 0 && block < desc->num_blocks && desc->block_table[block].norot_flag);
}

static int begin_instance(scan_walk *w, const pulseg_collection *coll, int s, int n)
{
    const pulseg_sequence_descriptor *desc = &coll->descriptors[s];
    const pulseg_virtual_segment *seg = &desc->segment_definitions[w->local];
    const pulseg_playout_backend *backend = w->loader.backend;
    pulseg_playout_segment *p = &w->segment;

    p->subsequence = s;
    p->segment = pulseg__global_segment(coll, s, w->local);
    if (p->segment < 0 || p->segment >= coll->total_unique_segments)
        return PULSEG_ERR_INDEX;
    p->instance = w->instances[p->segment];
    p->first_position = n;
    p->num_blocks = seg->num_blocks;
    p->rotate = instance_rotates(desc, n);
    p->await_trigger = w->prescan < 0 && instance_awaits_trigger(desc, n, w->local);
    p->half = (w->plan->mode == PULSEG_WAVES_STREAMED) ? p->instance % 2 : -1;
    return backend->begin_instance ? backend->begin_instance(backend->ctx, p) : PULSEG_SUCCESS;
}

/* The receive-gain calibration plays no digital output, and no gradient
 * whose amplitude varies across repetitions. */
static void prescan_registers(pulseg_block_instance *b)
{
    b->digitalout_flag = 0;
    if (b->gx_variable)
        b->gx_amp_hz_per_m = 0.0f;
    if (b->gy_variable)
        b->gy_amp_hz_per_m = 0.0f;
    if (b->gz_variable)
        b->gz_amp_hz_per_m = 0.0f;
    if (b->gx_variable || b->gy_variable || b->gz_variable)
    {
        b->wave_amp_hz_per_m[0] = 0.0f;
        b->wave_amp_hz_per_m[1] = 0.0f;
        b->wave_amp_hz_per_m[2] = 0.0f;
    }
}

/* Where the block plays its wave from, loaded there first when the waves
 * are streamed. */
static int place_wave(const scan_walk *w, int s, pulseg_playout_block *block)
{
    const pulseg_wave_plan *plan = w->plan;
    const int id = block->instance.wave_id;
    const int g = w->segment.segment;

    if (plan->mode == PULSEG_WAVES_RESIDENT && s < plan->num_subsequences &&
        id < plan->num_waves[s])
    {
        block->wave = &plan->waves[s][id];
        return PULSEG_SUCCESS;
    }
    if (plan->mode != PULSEG_WAVES_STREAMED || g >= plan->num_segments ||
        block->position >= plan->num_positions[g])
        return PULSEG_ERR_INDEX;
    block->wave = &plan->slots[g][2 * block->position + w->segment.half];
    return load_wave(&w->loader, s, id, block->wave);
}

static int set_block(scan_walk *w, const pulseg_collection *coll, int s)
{
    const pulseg_playout_backend *backend = w->loader.backend;
    pulseg_playout_block block;
    int rc;

    block.position = w->position;
    block.wave = NULL;
    rc = pulseg_get_block_instance(coll, &block.instance);
    if (PULSEG_FAILED(rc))
        return rc;
    if (w->prescan >= 0)
        prescan_registers(&block.instance);
    if (block.instance.wave_id >= 0)
        rc = place_wave(w, s, &block);
    if (PULSEG_SUCCEEDED(rc) && backend->set_block)
        rc = backend->set_block(backend->ctx, &w->segment, &block);
    if (block.instance.adc_flag)
        w->readouts_left -= 1;
    return rc;
}

static int end_instance(scan_walk *w)
{
    const pulseg_playout_backend *backend = w->loader.backend;
    const int rc =
        backend->play_instance ? backend->play_instance(backend->ctx, &w->segment) : PULSEG_SUCCESS;

    w->instances[w->segment.segment] += 1;
    w->done = w->prescan >= 0 && w->readouts_left <= 0;
    return rc;
}

/* Play the entry at the cursor. */
static int play_entry(scan_walk *w, const pulseg_collection *coll)
{
    const int s = coll->block_cursor.sequence_index;
    const int n = coll->block_cursor.exec_stream_position;
    const pulseg_sequence_descriptor *desc = &coll->descriptors[s];
    int rc = PULSEG_SUCCESS;

    if (w->prescan >= 0 && s > w->prescan)
    {
        w->done = 1;
        return rc;
    }
    if (!track_position(w, desc, s, n) || (w->prescan >= 0 && s != w->prescan))
        return rc;
    if (w->position == 0)
        rc = begin_instance(w, coll, s, n);
    if (PULSEG_SUCCEEDED(rc))
        rc = set_block(w, coll, s);
    if (PULSEG_SUCCEEDED(rc) && instance_ends(desc, n, w->position, w->segment.num_blocks))
        rc = end_instance(w);
    return rc;
}

int pulseg_playout_scan(
    pulseg_collection *coll,
    const pulseg_wave_budget *budget,
    const pulseg_playout_backend *backend,
    const pulseg_playout_options *options,
    pulseg_diagnostic *diag)
{
    pulseg_wave_plan plan = PULSEG_WAVE_PLAN_INIT;
    scan_walk walk;
    int rc = playout_arguments(coll, budget, backend);

    memset(&walk, 0, sizeof(walk));
    if (PULSEG_SUCCEEDED(rc))
        rc = stream_loaded(coll, diag);
    if (PULSEG_SUCCEEDED(rc))
        rc = pulseg_plan_waves(coll, budget, &plan, diag);
    if (PULSEG_SUCCEEDED(rc))
        rc = loader_open(&walk.loader, coll, backend, budget->raster_us, &plan);
    if (PULSEG_SUCCEEDED(rc))
        rc = walk_open(&walk, coll, &plan, options);
    if (PULSEG_SUCCEEDED(rc))
        pulseg_cursor_reset(coll);
    while (PULSEG_SUCCEEDED(rc) && !walk.done && pulseg_cursor_next(coll) == PULSEG_CURSOR_BLOCK)
        rc = play_entry(&walk, coll);
    walk_close(&walk);
    pulseg_free_wave_plan(&plan);
    return rc;
}
