/**
 * @file pulseg_playout.c
 * @brief The two stages of a segmented playout over a backend; see
 *        pulseg_playout.h.
 */

#include <math.h>
#include <string.h>

#include "pulseg_internal.h"
#include "pulseg.h"

/* ================================================================== */
/*  Waveform-memory samples                                           */
/* ================================================================== */

static PULSEG_WAVE_SAMPLE to_sample(double scaled)
{
    const double full = (double)PULSEG_WAVE_FULL_SCALE;
    const double x = scaled > full ? full : scaled < -full ? -full : scaled;
    return PULSEG_WAVE_QUANTIZE(x);
}

void pulseg_wave_samples(const float *values, long count, PULSEG_WAVE_SAMPLE *samples)
{
    long i;

    for (i = 0; i < count; ++i)
        samples[i] = to_sample((double)values[i] * (double)PULSEG_WAVE_FULL_SCALE);
}

void pulseg_phase_samples(
    const float *phase,
    long count,
    double radians_per_unit,
    PULSEG_WAVE_SAMPLE *samples)
{
    const double turn = 2.0 * M_PI;
    long i;

    for (i = 0; i < count; ++i)
    {
        double rad = (double)phase[i] * radians_per_unit;
        rad -= turn * floor((rad + M_PI) / turn);
        samples[i] = to_sample(
            rad * ((double)PULSEG_WAVE_FULL_SCALE / (double)PULSEG_WAVE_PHASE_FULL_SCALE));
    }
}

/* ================================================================== */
/*  Loading waves                                                     */
/* ================================================================== */

/* Loading waves into waveform memory: on which raster, through which
 * backend, with room for the longest region the layout holds, sampled and
 * converted, and for the span a resident position hands the backend in each
 * of its slots. */
typedef struct wave_loader
{
    const pulseg_collection *coll;
    const pulseg_playout_backend *backend;
    float raster_us;
    int slots;
    long longest;    /* samples of the longest region     */
    float *scratch;  /* one longest region for each axis  */
    PULSEG_WAVE_SAMPLE *samples;
    pulseg_wave_region *span;
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
        for (j = 0; j < plan->budget.slots * plan->num_positions[i]; ++j)
            if (plan->slots[i][j].samples > longest)
                longest = plan->slots[i][j].samples;
    return longest;
}

static int loader_open(
    wave_loader *l,
    const pulseg_collection *coll,
    const pulseg_playout_backend *backend,
    const pulseg_wave_plan *plan)
{
    const size_t longest = (size_t)longest_region(plan);

    l->coll = coll;
    l->backend = backend;
    l->raster_us = plan->budget.raster_us;
    l->slots = plan->budget.slots;
    l->longest = (long)longest;
    l->scratch = (float *)PULSEG_ALLOC(3 * longest * sizeof(float));
    l->samples = (PULSEG_WAVE_SAMPLE *)PULSEG_ALLOC(longest * sizeof(PULSEG_WAVE_SAMPLE));
    l->span = (pulseg_wave_region *)PULSEG_ALLOC((size_t)l->slots * sizeof(pulseg_wave_region));
    return (l->scratch && l->samples && l->span) ? PULSEG_SUCCESS : PULSEG_ERR_ALLOC_FAILED;
}

static void loader_close(wave_loader *l)
{
    if (l->scratch)
        PULSEG_FREE(l->scratch);
    if (l->samples)
        PULSEG_FREE(l->samples);
    if (l->span)
        PULSEG_FREE(l->span);
    l->scratch = NULL;
    l->samples = NULL;
    l->span = NULL;
}

/* Sample wave @p w of subsequence @p s over @p region and load each axis it
 * drives where the region holds it. */
static int load_wave(const wave_loader *l, int s, int w, const pulseg_wave_region *region)
{
    pulseg_wave_load load;
    float *axes[3];
    int a, rc;

    for (a = 0; a < 3; ++a)
        axes[a] = region->offset[a] < 0 ? NULL : l->scratch + (long)a * l->longest;
    rc = pulseg__sample_wave_axes(
        l->coll, s, w, region->start_us, l->raster_us, region->samples, axes);
    load.subsequence = s;
    load.wave = w;
    load.count = region->samples;
    load.samples = l->samples;
    for (a = 0; a < 3 && PULSEG_SUCCEEDED(rc); ++a)
    {
        if (!axes[a])
            continue;
        load.axis = a;
        load.offset = region->offset[a];
        pulseg_wave_samples(axes[a], region->samples, l->samples);
        if (l->backend->load_wave)
            rc = l->backend->load_wave(l->backend->ctx, &load);
    }
    return rc;
}

static int playout_arguments(
    const pulseg_collection *coll,
    const pulseg_wave_plan *plan,
    const pulseg_playout_backend *backend)
{
    if (!coll || !plan || !backend)
        return PULSEG_ERR_NULL_POINTER;
    return (plan->budget.raster_us > 0.0f && plan->budget.slots >= 1) ? PULSEG_SUCCESS
                                                                       : PULSEG_ERR_INVALID_ARGUMENT;
}

/* ================================================================== */
/*  First stage                                                       */
/* ================================================================== */

/* The span every wave of a resident layout at position @p info covers, in
 * each of the loader's slots, held nowhere. */
static const pulseg_wave_region *resident_span(const wave_loader *l, const pulseg_block_info *info)
{
    int k, a;

    for (k = 0; k < l->slots; ++k)
    {
        pulseg__wave_cover(info->wave_start_us, info->wave_end_us, l->raster_us, &l->span[k]);
        for (a = 0; a < 3; ++a)
            l->span[k].offset[a] = -1;
    }
    return l->span;
}

static int prepare_positions(
    const wave_loader *l,
    const pulseg_wave_plan *plan,
    int g,
    int num_blocks)
{
    int b, rc = PULSEG_SUCCESS;

    for (b = 0; b < num_blocks && PULSEG_SUCCEEDED(rc); ++b)
    {
        pulseg_block_info info = PULSEG_BLOCK_INFO_INIT;
        const pulseg_wave_region *slots = NULL;

        rc = pulseg_get_block_info(l->coll, &info, g, b);
        if (PULSEG_FAILED(rc))
            break;
        if (info.wave_points > 0 && plan->mode == PULSEG_WAVES_STREAMED)
            slots = &plan->slots[g][l->slots * b];
        else if (info.wave_points > 0)
            slots = resident_span(l, &info);
        if (l->backend->prepare_block)
            rc = l->backend->prepare_block(l->backend->ctx, g, b, &info, slots);
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
    const pulseg_wave_plan *plan,
    const pulseg_playout_backend *backend)
{
    wave_loader loader;
    int rc = playout_arguments(coll, plan, backend);

    memset(&loader, 0, sizeof(loader));
    if (PULSEG_SUCCEEDED(rc))
        rc = loader_open(&loader, coll, backend, plan);
    if (PULSEG_SUCCEEDED(rc) && backend->reserve_waves)
        rc = backend->reserve_waves(backend->ctx, plan);
    if (PULSEG_SUCCEEDED(rc))
        rc = prepare_segments(&loader, plan);
    if (PULSEG_SUCCEEDED(rc) && plan->mode == PULSEG_WAVES_RESIDENT)
        rc = load_resident(&loader, plan);
    loader_close(&loader);
    return rc;
}

/* ================================================================== */
/*  Transmit channels                                                 */
/* ================================================================== */

/* The channels one RF occurrence drives, and the shim it plays under, NULL
 * for none: its shim's, or its waveform's.  In a parallel-transmit collection
 * of @p ptx_channels channels, a single waveform with no shim plays on every
 * channel at the instance's amplitude. */
static int occurrence_channels(
    const pulseg_sequence_descriptor *desc,
    const pulseg_block_table_element *bte,
    const pulseg_rf_definition *def,
    int ptx_channels,
    const pulseg_rf_shim_definition **shim)
{
    int channels;

    *shim = (bte->rf_shim_id >= 0 && bte->rf_shim_id < desc->num_rf_shims)
        ? &desc->rf_shim_definitions[bte->rf_shim_id]
        : NULL;
    channels = *shim ? (*shim)->num_channels : def->num_channels;
    if (!*shim && channels <= 1)
        channels = ptx_channels;
    return channels > PULSEG_MAX_RF_SHIM_CHANNELS ? PULSEG_MAX_RF_SHIM_CHANNELS : channels;
}

/* The transmit channels of a collection: the most any RF definition carries a
 * waveform for or any RF shim weights, 1 where none has more than one. */
static int collection_ptx_channels(const pulseg_collection *coll)
{
    int s, i, channels = 1;

    for (s = 0; s < coll->num_subsequences; ++s)
    {
        const pulseg_sequence_descriptor *desc = &coll->descriptors[s];

        for (i = 0; i < desc->num_unique_rfs; ++i)
            if (desc->rf_definitions[i].num_channels > channels)
                channels = desc->rf_definitions[i].num_channels;
        for (i = 0; i < desc->num_rf_shims; ++i)
            if (desc->rf_shim_definitions[i].num_channels > channels)
                channels = desc->rf_shim_definitions[i].num_channels;
    }
    return channels;
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
    int lead;          /* 1 while the prescan-only lead plays              */
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
    p->slot = (w->plan->mode == PULSEG_WAVES_STREAMED) ? p->instance % w->plan->budget.slots : -1;
    return backend->begin_instance ? backend->begin_instance(backend->ctx, p) : PULSEG_SUCCESS;
}

/* The receive-gain calibration plays no digital output, no gradient whose
 * amplitude varies across repetitions, and acquires no readout NOPRESCAN
 * marks.  An RF pulse whose frequency offset varies across repetitions, the
 * slices of a stack, plays at their mean: the centre of the stack. */
static void prescan_registers(pulseg_block_instance *b)
{
    if (b->rf_freq_variable)
        b->rf_freq_hz = b->rf_freq_mean_hz;
    b->digitalout_flag = 0;
    if (b->noprescan_flag)
        b->adc_flag = 0;
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
    block->wave = &plan->slots[g][plan->budget.slots * block->position + w->segment.slot];
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

static int end_instance(scan_walk *w, pulseg_collection *coll)
{
    const pulseg_playout_backend *backend = w->loader.backend;
    int rc =
        backend->play_instance ? backend->play_instance(backend->ctx, &w->segment) : PULSEG_SUCCESS;

    if (rc == PULSEG_PLAYOUT_REPLAY && w->lead)
        rc = PULSEG_SUCCESS; /* the lead is played once */
    if (rc == PULSEG_PLAYOUT_REPLAY)
    {
        /* The instance is asked for again, so it has not been played: the
         * cursor goes back to the block it was marked at, and the count it
         * would have joined is left alone.
         *
         * One before it, because the walk reads the block the cursor moves
         * onto: landing on the first block would play the instance from its
         * second. */
        pulseg_cursor_rewind(coll);
        coll->block_cursor.exec_stream_position -= 1;
        w->position = 0;
        w->local = -1;
        return PULSEG_SUCCESS;
    }

    w->instances[w->segment.segment] += 1;
    w->done = w->prescan >= 0 && !w->lead && w->readouts_left <= 0;
    return rc;
}

/* Play the entry at the cursor. */
static int play_entry(scan_walk *w, pulseg_collection *coll)
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
    if (!track_position(w, desc, s, n) || (w->prescan >= 0 && s != w->prescan) ||
        (w->prescan < 0 && desc->prescan_only))
        return rc;
    if (w->position == 0)
    {
        /* Where a replay returns to. */
        pulseg_cursor_mark(coll);
        rc = begin_instance(w, coll, s, n);
    }
    if (PULSEG_SUCCEEDED(rc))
        rc = set_block(w, coll, s);
    if (PULSEG_SUCCEEDED(rc) && instance_ends(desc, n, w->position, w->segment.num_blocks))
        rc = end_instance(w, coll);
    return rc;
}

/* The prescan-only subsequence a prescan of subsequence @p prescan plays
 * first, -1 for none: the generated one, where @p prescan is another. */
static int prescan_lead(const pulseg_collection *coll, int prescan)
{
    int s;

    for (s = 0; s < coll->num_subsequences; ++s)
        if (coll->descriptors[s].prescan_only && s != prescan)
            return s;
    return -1;
}

/* Play subsequence @p lead whole, before the calibration: its readouts are
 * NOPRESCAN, and not counted. */
static int play_lead(scan_walk *w, pulseg_collection *coll, int lead)
{
    const pulseg_sequence_descriptor *desc = &coll->descriptors[lead];
    int n, rc = pulseg_cursor_seek(coll, lead, 0);

    w->subsequence = -1;
    w->lead = 1;
    for (n = 0; n < desc->exec_stream_len && PULSEG_SUCCEEDED(rc); ++n)
    {
        if (pulseg_cursor_next(coll) != PULSEG_CURSOR_BLOCK)
            return PULSEG_ERR_INDEX;
        if (!track_position(w, desc, lead, n))
            continue;
        if (w->position == 0)
            rc = begin_instance(w, coll, lead, n);
        if (PULSEG_SUCCEEDED(rc))
            rc = set_block(w, coll, lead);
        if (PULSEG_SUCCEEDED(rc) && instance_ends(desc, n, w->position, w->segment.num_blocks))
            rc = end_instance(w, coll);
    }
    w->lead = 0;
    w->subsequence = -1;
    w->local = -1;
    return rc;
}

int pulseg_playout_scan(
    pulseg_collection *coll,
    const pulseg_wave_plan *plan,
    const pulseg_playout_backend *backend,
    const pulseg_playout_options *options,
    pulseg_diagnostic *diag)
{
    scan_walk walk;
    int rc = playout_arguments(coll, plan, backend);

    memset(&walk, 0, sizeof(walk));
    if (PULSEG_SUCCEEDED(rc))
        rc = stream_loaded(coll, diag);
    if (PULSEG_SUCCEEDED(rc))
        rc = loader_open(&walk.loader, coll, backend, plan);
    if (PULSEG_SUCCEEDED(rc))
        rc = walk_open(&walk, coll, plan, options);
    if (PULSEG_SUCCEEDED(rc) && walk.prescan >= 0 && prescan_lead(coll, walk.prescan) >= 0)
        rc = play_lead(&walk, coll, prescan_lead(coll, walk.prescan));
    if (PULSEG_SUCCEEDED(rc))
        pulseg_cursor_reset(coll);
    while (PULSEG_SUCCEEDED(rc) && !walk.done && pulseg_cursor_next(coll) == PULSEG_CURSOR_BLOCK)
        rc = play_entry(&walk, coll);
    walk_close(&walk);
    return rc;
}

/* ================================================================== */
/*  Channel drives                                                    */
/* ================================================================== */

/* The per-channel waveform peaks of every RF definition of a subsequence,
 * PULSEG_MAX_RF_SHIM_CHANNELS to a definition, read once. */
static float *channel_peaks(const pulseg_sequence_descriptor *desc)
{
    const int n = desc->num_unique_rfs > 0 ? desc->num_unique_rfs : 1;
    float *peaks = (float *)PULSEG_ALLOC((size_t)n * PULSEG_MAX_RF_SHIM_CHANNELS * sizeof(float));
    int i;

    if (!peaks)
        return NULL;
    for (i = 0; i < n * PULSEG_MAX_RF_SHIM_CHANNELS; ++i)
        peaks[i] = 0.0f;
    for (i = 0; i < desc->num_unique_rfs; ++i)
        pulseg__rf_channel_peaks(
            desc, &desc->rf_definitions[i], &peaks[i * PULSEG_MAX_RF_SHIM_CHANNELS],
            PULSEG_MAX_RF_SHIM_CHANNELS);
    return peaks;
}

/* Raise @p drive_hz to what entry @p n of @p desc drives each channel at:
 * the RF amplitude times the channel's waveform peak, times its shim
 * magnitude under a shim.  A single waveform plays on every channel of a
 * parallel-transmit collection. */
static void entry_drives(
    const pulseg_sequence_descriptor *desc,
    const float *peaks,
    int n,
    int ptx_channels,
    float *drive_hz,
    int max_channels)
{
    const int block = pulseg__exec_block_idx(desc, n);
    const pulseg_block_table_element *bte;
    const pulseg_rf_table_element *rte;
    const pulseg_rf_definition *def;
    const pulseg_rf_shim_definition *shim;
    float amplitude;
    int channels, k;

    if (block < 0 || block >= desc->num_blocks)
        return;
    bte = &desc->block_table[block];
    if (bte->rf_id < 0 || bte->rf_id >= desc->rf_table_size)
        return;
    rte = &desc->rf_table[bte->rf_id];
    if (rte->id < 0 || rte->id >= desc->num_unique_rfs)
        return;
    def = &desc->rf_definitions[rte->id];
    channels = occurrence_channels(desc, bte, def, ptx_channels, &shim);
    amplitude = (float)fabs((double)rte->amplitude);
    for (k = 0; k < channels && k < max_channels; ++k)
    {
        const float peak = peaks[rte->id * PULSEG_MAX_RF_SHIM_CHANNELS + (def->num_channels > 1 ? k : 0)];
        const float drive = amplitude * peak * (shim ? shim->magnitudes[k] : 1.0f);

        if (drive > drive_hz[k])
            drive_hz[k] = drive;
    }
}

/* Raise @p drive_hz over the blocks the receive-gain calibration of
 * subsequence @p s plays: from its start to the instance that completes
 * @p readouts readouts NOPRESCAN does not exempt. */
static void calibration_drives(
    const pulseg_sequence_descriptor *desc,
    const float *peaks,
    int readouts,
    int ptx_channels,
    float *drive_hz,
    int max_channels)
{
    int n, local = -1, position = 0;

    for (n = 0; n < desc->exec_stream_len; ++n)
    {
        const int seg = pulseg__exec_seg_id(desc, n);
        const int block = pulseg__exec_block_idx(desc, n);
        int blocks;

        if (seg < 0 || seg >= desc->num_unique_segments)
        {
            local = -1;
            continue;
        }
        blocks = desc->segment_definitions[seg].num_blocks;
        position = (seg == local && blocks > 0) ? (position + 1) % blocks : 0;
        local = seg;
        entry_drives(desc, peaks, n, ptx_channels, drive_hz, max_channels);
        if (block >= 0 && block < desc->num_blocks && desc->block_table[block].adc_id >= 0 &&
            !desc->block_table[block].noprescan_flag)
            readouts -= 1;
        if (readouts <= 0 && instance_ends(desc, n, position, blocks))
            break;
    }
}

int pulseg_get_channel_drives(
    const pulseg_collection *coll,
    const pulseg_playout_options *options,
    float *drive_hz,
    int max_channels)
{
    const int prescan = (options && options->prescan_subsequence >= 0) ? options->prescan_subsequence : -1;
    float **peaks;
    int s, n, k, ptx_channels, rc = PULSEG_SUCCESS;

    if (!coll || !drive_hz)
        return PULSEG_ERR_NULL_POINTER;
    if (prescan >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;
    ptx_channels = collection_ptx_channels(coll);
    for (k = 0; k < max_channels; ++k)
        drive_hz[k] = 0.0f;
    peaks = (float **)PULSEG_ALLOC((size_t)(coll->num_subsequences > 0 ? coll->num_subsequences : 1) * sizeof(float *));
    if (!peaks)
        return PULSEG_ERR_ALLOC_FAILED;
    for (s = 0; s < coll->num_subsequences; ++s)
        peaks[s] = NULL;
    for (s = 0; s < coll->num_subsequences && PULSEG_SUCCEEDED(rc); ++s)
        if (!(peaks[s] = channel_peaks(&coll->descriptors[s])))
            rc = PULSEG_ERR_ALLOC_FAILED;

    if (PULSEG_SUCCEEDED(rc) && prescan < 0)
        for (s = 0; s < coll->num_subsequences; ++s)
        {
            if (coll->descriptors[s].prescan_only)
                continue;
            for (n = 0; n < coll->descriptors[s].exec_stream_len; ++n)
                entry_drives(&coll->descriptors[s], peaks[s], n, ptx_channels, drive_hz, max_channels);
        }
    if (PULSEG_SUCCEEDED(rc) && prescan >= 0)
    {
        const int lead = prescan_lead(coll, prescan);

        if (lead >= 0)
            for (n = 0; n < coll->descriptors[lead].exec_stream_len; ++n)
                entry_drives(&coll->descriptors[lead], peaks[lead], n, ptx_channels, drive_hz, max_channels);
        if (PULSEG_SUCCEEDED(rc))
            calibration_drives(&coll->descriptors[prescan], peaks[prescan],
                               prescan_readouts(coll, prescan, options), ptx_channels, drive_hz, max_channels);
    }

    for (s = 0; s < coll->num_subsequences; ++s)
        if (peaks[s])
            PULSEG_FREE(peaks[s]);
    PULSEG_FREE(peaks);
    return PULSEG_SUCCEEDED(rc) ? ptx_channels : rc;
}

/* ================================================================== */
/*  Receive calibration                                               */
/* ================================================================== */

static int entry_reads(const pulseg_sequence_descriptor *desc, int n)
{
    const int block = pulseg__exec_block_idx(desc, n);

    return block >= 0 && block < desc->num_blocks && desc->block_table[block].adc_id >= 0 &&
        !desc->block_table[block].noprescan_flag;
}

static int subsequence_reads(const pulseg_sequence_descriptor *desc)
{
    int n;

    for (n = 0; n < desc->exec_stream_len; ++n)
        if (entry_reads(desc, n))
            return 1;
    return 0;
}

int pulseg_get_receive_calibration(const pulseg_collection *coll, pulseg_receive_calibration *out)
{
    int s, reading = -1, count = 0;

    if (!coll || !out)
        return PULSEG_ERR_NULL_POINTER;
    out->generated = 0;
    out->lead = -1;
    out->subsequence = -1;
    out->readouts = 0;
    for (s = 0; s < coll->num_subsequences; ++s)
    {
        if (!coll->descriptors[s].prescan_only)
            continue;
        if (subsequence_reads(&coll->descriptors[s]))
        {
            out->generated = 1;
            out->subsequence = s;
            out->readouts = coll->descriptors[s].num_gain_cal_readouts;
            return PULSEG_SUCCESS;
        }
        out->lead = s;
    }
    for (s = 0; s < coll->num_subsequences; ++s)
        if (!coll->descriptors[s].prescan_only && subsequence_reads(&coll->descriptors[s]))
        {
            reading = s;
            ++count;
        }
    if (count == 1)
    {
        out->subsequence = reading;
        out->readouts = coll->descriptors[reading].num_gain_cal_readouts;
    }
    return PULSEG_SUCCESS;
}
