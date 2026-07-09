/* pulseg_accessors.c -- collection-level accessor functions
 *
 * Public functions:
 *   pulseg_get_max_adc_samples    pulseg_get_adc_dwell_ns
 *   pulseg_get_adc_num_samples    pulseg_get_num_segments
 *   pulseg_is_segment_pure_delay  pulseg_get_segment_num_blocks
 *   pulseg_get_block_start_time_us   pulseg_get_block_duration_us
 *   pulseg_block_has_rf           pulseg_block_rf_has_uniform_raster
 *   pulseg_block_rf_is_complex    pulseg_get_rf_num_samples
 *   pulseg_get_rf_num_channels    pulseg_get_rf_delay_us
 *   pulseg_get_rf_magnitude       pulseg_get_rf_phase
 *   pulseg_get_rf_time_us
 *   pulseg_block_has_grad         pulseg_block_grad_is_trapezoid
 *   pulseg_get_grad_num_samples   pulseg_get_grad_num_shots
 *   pulseg_get_grad_delay_us         pulseg_get_grad_amplitude
 *   pulseg_get_grad_initial_amplitude_hz_per_m
 *   pulseg_get_grad_initial_shot_id
 *   pulseg_get_grad_time_us
 *   pulseg_block_has_adc          pulseg_get_adc_delay_us
 *   pulseg_get_adc_library_index
 *   pulseg_block_has_digitalout   pulseg_get_digitalout_delay_us
 *   pulseg_segment_has_trigger    pulseg_get_segment_trigger_delay_us
 *   pulseg_segment_is_nav
 *   pulseg_block_has_rotation
 *   pulseg_block_has_norot        pulseg_block_has_nopos
 *
 * Internal (pulseg__) functions:
 *   pulseg__resolve_segment       pulseg__resolve_block
 */

#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "pulseg_internal.h"
#include "pulseg.h"

/* ================================================================== */
/*  Resolve helpers                                                   */
/* ================================================================== */

int pulseg__resolve_segment(
    const pulseg_collection *coll,
    const pulseg_sequence_descriptor **out_desc,
    int *out_local_seg,
    int seg_idx)
{
    int i, num_segs, global_idx;

    if (!coll || seg_idx < 0 || seg_idx >= coll->total_unique_segments)
        return 0;

    global_idx = 0;
    for (i = 0; i < coll->num_subsequences; ++i)
    {
        num_segs = coll->descriptors[i].num_unique_segments;
        if (seg_idx < global_idx + num_segs)
        {
            if (out_desc)
                *out_desc = &coll->descriptors[i];
            if (out_local_seg)
                *out_local_seg = seg_idx - global_idx;
            return 1;
        }
        global_idx += num_segs;
    }
    return 0;
}

int pulseg__resolve_block(
    const pulseg_collection *coll,
    const pulseg_sequence_descriptor **out_desc,
    const pulseg_tr_segment **out_seg,
    int *out_local_blk,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg;
    const pulseg_tr_segment *seg;

    desc = NULL;
    if (!pulseg__resolve_segment(coll, &desc, &local_seg, seg_idx))
        return 0;

    seg = &desc->segment_definitions[local_seg];
    if (blk_idx < 0 || blk_idx >= seg->num_blocks)
        return 0;

    if (out_desc)
        *out_desc = desc;
    if (out_seg)
        *out_seg = seg;
    if (out_local_blk)
        *out_local_blk = blk_idx;
    return 1;
}

/* ================================================================== */
/*  Axis helper                                                       */
/* ================================================================== */

static int get_grad_id_by_axis(const pulseg_block_definition *bdef, int axis)
{
    switch (axis)
    {
    case PULSEG_GRAD_AXIS_X:
        return bdef->gx_id;
    case PULSEG_GRAD_AXIS_Y:
        return bdef->gy_id;
    case PULSEG_GRAD_AXIS_Z:
        return bdef->gz_id;
    default:
        return -1;
    }
}

static int get_grad_event_id_by_axis(const pulseg_block_table_element *bte, int axis)
{
    switch (axis)
    {
    case PULSEG_GRAD_AXIS_X:
        return bte->gx_id;
    case PULSEG_GRAD_AXIS_Y:
        return bte->gy_id;
    case PULSEG_GRAD_AXIS_Z:
        return bte->gz_id;
    default:
        return -1;
    }
}

static const pulseg_block_table_element *resolve_block_table_via_max_energy(
    const pulseg_sequence_descriptor *desc,
    const pulseg_tr_segment *seg,
    int local_blk)
{
    int scan_idx, block_table_idx;

    if (!desc || !seg)
        return NULL;
    if (seg->max_energy_start_block < 0)
        return NULL;

    scan_idx = seg->max_energy_start_block + local_blk;
    if (scan_idx < 0 || scan_idx >= desc->scan_table_len)
        return NULL;

    block_table_idx = desc->scan_table_block_idx[scan_idx];
    if (block_table_idx < 0 || block_table_idx >= desc->num_blocks)
        return NULL;

    return &desc->block_table[block_table_idx];
}

static int resolve_grad_def_via_max_energy_instance(
    const pulseg_sequence_descriptor *desc,
    const pulseg_tr_segment *seg,
    int local_blk, int axis)
{
    const pulseg_block_table_element *bte;
    int grad_event_id;
    int grad_def_id;

    if (!desc || !seg)
        return -1;
    if (local_blk < 0 || local_blk >= seg->num_blocks)
        return -1;

    bte = resolve_block_table_via_max_energy(desc, seg, local_blk);
    if (!bte)
        return -1;

    grad_event_id = get_grad_event_id_by_axis(bte, axis);
    if (grad_event_id < 0 || grad_event_id >= desc->grad_table_size)
        return -1;

    grad_def_id = desc->grad_table[grad_event_id].id;
    if (grad_def_id < 0 || grad_def_id >= desc->num_unique_grads)
        return -1;

    return grad_def_id;
}

/* ================================================================== */
/*  Subsequence accessors (internal helpers for batch getters)         */
/* ================================================================== */

static int pulseg__get_num_subsequences(
    const pulseg_collection *coll)
{
    if (!coll)
        return 0;
    return coll->num_subsequences;
}

static float pulseg__get_tr_duration_us(
    const pulseg_collection *coll,
    int subseq_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_descriptor *tr;

    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0.0f;

    desc = &coll->descriptors[subseq_idx];
    tr = &desc->tr_descriptor;
    return tr->tr_duration_us;
}

static int pulseg__get_num_trs(
    const pulseg_collection *coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].tr_descriptor.num_trs;
}

static int pulseg__get_tr_size(
    const pulseg_collection *coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].tr_descriptor.tr_size;
}

static int pulseg__get_num_prep_blocks(
    const pulseg_collection *coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].tr_descriptor.num_prep_blocks;
}

static int pulseg__get_num_cooldown_blocks(
    const pulseg_collection *coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].tr_descriptor.num_cooldown_blocks;
}

static int pulseg__get_degenerate_prep(
    const pulseg_collection *coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].tr_descriptor.degenerate_prep;
}

static int pulseg__get_degenerate_cooldown(
    const pulseg_collection *coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].tr_descriptor.degenerate_cooldown;
}

static int pulseg__get_num_prep_trs(
    const pulseg_collection *coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].tr_descriptor.num_prep_trs;
}

static int pulseg__get_num_cooldown_trs(
    const pulseg_collection *coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].tr_descriptor.num_cooldown_trs;
}

static int pulseg__get_num_unique_adcs(
    const pulseg_collection *coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].num_unique_adcs;
}

static int pulseg__is_pmc_enabled(
    const pulseg_collection *coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].enable_pmc;
}

static int pulseg__get_subseq_segment_offset(
    const pulseg_collection *coll,
    int subseq_idx)
{
    int i, offset = 0;
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    for (i = 0; i < subseq_idx; ++i)
        offset += coll->descriptors[i].num_unique_segments;
    return offset;
}

static float pulseg__get_total_duration_us(
    const pulseg_collection *coll)
{
    if (!coll)
        return 0.0f;
    return coll->total_duration_us;
}

int pulseg_get_scan_time(const pulseg_collection *coll,
                            pulseg_scan_time_info *info,
                            int num_reps)
{
    int i, n, bt_idx;
    const pulseg_sequence_descriptor *desc;
    const pulseg_block_table_element *bte;
    const pulseg_block_definition *bdef;
    int prev_seg, cur_seg;

    if (!coll || !info)
        return PULSEG_ERR_NULL_POINTER;
    if (num_reps < 1)
        return PULSEG_ERR_INVALID_ARGUMENT;
    if (coll->num_subsequences <= 0)
        return PULSEG_ERR_COLLECTION_EMPTY;

    (void)num_reps; /* averages already baked into scan table */

    info->total_duration_us = 0.0f;
    info->total_segment_boundaries = 0;

    for (i = 0; i < coll->num_subsequences; ++i)
    {
        desc = &coll->descriptors[i];
        prev_seg = -1;

        for (n = 0; n < desc->scan_table_len; ++n)
        {
            bt_idx = desc->scan_table_block_idx[n];
            bte = &desc->block_table[bt_idx];
            bdef = &desc->block_definitions[bte->id];

            /* Duration: pure delay uses instance value, normal uses definition */
            info->total_duration_us += (bte->duration_us >= 0)
                                           ? (float)bte->duration_us
                                           : (float)bdef->duration_us;

            /* Count segment boundaries (transitions) */
            cur_seg = desc->scan_table_seg_id[n];
            if (cur_seg >= 0 && cur_seg != prev_seg)
                info->total_segment_boundaries += 1;
            prev_seg = cur_seg;
        }
    }

    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  RF accessors                                                      */
/* ================================================================== */

static int pulseg__get_num_unique_rf(
    const pulseg_collection *coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].num_unique_rfs;
}

int pulseg_get_rf_stats(
    const pulseg_collection *coll,
    pulseg_rf_stats *stats,
    int subseq_idx, int rf_idx)
{
    const pulseg_sequence_descriptor *desc;

    if (!coll || !stats)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;
    desc = &coll->descriptors[subseq_idx];
    if (rf_idx < 0 || rf_idx >= desc->num_unique_rfs)
        return PULSEG_ERR_INVALID_ARGUMENT;

    *stats = desc->rf_definitions[rf_idx].stats;
    return PULSEG_SUCCESS;
}

/*
 * pulseg_get_tr_rf_ids --
 *   Return an array of RF definition IDs for each block position
 *   within the first main TR.  Blocks without RF get -1.
 *
 *   out_rf_ids must point to a pre-allocated array of tr_size ints.
 *   Returns tr_size on success, negative error code on failure.
 */
int pulseg_get_tr_rf_ids(
    const pulseg_collection *coll,
    int *out_rf_ids,
    int subseq_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_descriptor *trd;
    const pulseg_block_table_element *bte;
    int i, block_idx, tr_size;

    if (!coll || !out_rf_ids)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[subseq_idx];
    trd = &desc->tr_descriptor;
    tr_size = trd->tr_size;

    for (i = 0; i < tr_size; ++i)
    {
        block_idx = trd->num_prep_blocks + i;
        bte = &desc->block_table[block_idx];
        if (bte->rf_id >= 0 && bte->rf_id < desc->rf_table_size)
            out_rf_ids[i] = desc->rf_table[bte->rf_id].id;
        else
            out_rf_ids[i] = -1;
    }

    return tr_size;
}

/* ================================================================== */
/*  pulseg_get_rf_array --                                         */
/*    Build an ordered array of RF stats for a TR region.             */
/*    Each entry gets the base rf_stats patched with the actual       */
/*    amplitude from the rf_table and the repetition count for        */
/*    that region.  The library allocates; caller must free().        */
/* ================================================================== */
int pulseg_get_rf_array(
    const pulseg_collection *coll,
    pulseg_rf_stats **out_pulses,
    int subseq_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_descriptor *trd;
    const pulseg_block_table_element *bte;
    const pulseg_rf_definition *rfdef;
    int start, count, num_instances;
    int use_scan_table;
    int num_passes, pass_size;
    int i, n, num_rf;

    if (!coll || !out_pulses)
        return PULSEG_ERR_NULL_POINTER;
    *out_pulses = NULL;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[subseq_idx];
    trd = &desc->tr_descriptor;

    use_scan_table = 0;
    if ((!trd->degenerate_prep || !trd->degenerate_cooldown) &&
        (trd->num_prep_blocks > 0 || trd->num_cooldown_blocks > 0))
    {
        num_passes = (desc->num_passes > 1) ? desc->num_passes : 1;
        pass_size = (num_passes > 0) ? (desc->scan_table_len / num_passes) : 0;

        start = 0;
        count = pass_size;
        num_instances = num_passes;
        use_scan_table = 1;
    }
    else
    {
        int num_avgs = (desc->num_averages > 1) ? desc->num_averages : 1;
        start = trd->num_prep_blocks + trd->imaging_tr_start;
        count = trd->tr_size;
        /* Total TR instances: imaging TRs replicated by NEX,
         * plus degenerate prep/cooldown TRs (played once each). */
        num_instances = num_avgs * trd->num_trs + trd->num_prep_trs + trd->num_cooldown_trs;
        if (num_instances < 0)
            num_instances = 0;
    }

    /* Clamp to available block range */
    if (use_scan_table)
    {
        if (start + count > desc->scan_table_len)
            count = desc->scan_table_len - start;
    }
    else if (start + count > desc->num_blocks)
    {
        count = desc->num_blocks - start;
    }
    if (count < 0)
        count = 0;

    /* Pass 1: count RF-bearing blocks */
    num_rf = 0;
    for (i = 0; i < count; ++i)
    {
        int blk_idx = use_scan_table
                          ? desc->scan_table_block_idx[start + i]
                          : (start + i);
        bte = &desc->block_table[blk_idx];
        if (bte->rf_id >= 0 && bte->rf_id < desc->rf_table_size)
        {
            int id = desc->rf_table[bte->rf_id].id;
            if (id >= 0 && id < desc->num_unique_rfs)
                num_rf++;
        }
    }

    if (num_rf == 0)
        return 0;

    /* Allocate output array (caller frees with PULSEG_FREE) */
    *out_pulses = (pulseg_rf_stats *)PULSEG_ALLOC(
        (size_t)num_rf * sizeof(pulseg_rf_stats));
    if (!*out_pulses)
        return PULSEG_ERR_ALLOC_FAILED;

    /* Pass 2: fill entries */
    n = 0;
    for (i = 0; i < count; ++i)
    {
        int blk_idx = use_scan_table ? desc->scan_table_block_idx[start + i] : (start + i);
        int rf_def_id;
        float act_amp;

        bte = &desc->block_table[blk_idx];
        if (bte->rf_id < 0 || bte->rf_id >= desc->rf_table_size)
            continue;

        rf_def_id = desc->rf_table[bte->rf_id].id;
        if (rf_def_id < 0 || rf_def_id >= desc->num_unique_rfs)
            continue;

        rfdef = &desc->rf_definitions[rf_def_id];

        /* Hard-copy base stats */
        (*out_pulses)[n] = rfdef->stats;

        /* Patch event-specific amplitude-dependent stats from rf_table. */
        act_amp = desc->rf_table[bte->rf_id].amplitude;
        (*out_pulses)[n].act_amplitude_hz = (act_amp >= 0.0f) ? act_amp : -act_amp;

        /* base_amplitude_hz retains definition-level nominal amplitude
         * from the hard-copy above (rfdef->stats.base_amplitude_hz). */
        (*out_pulses)[n].flip_angle_rad = rfdef->stats.flip_angle_rad;

        /* Set repetition count */
        (*out_pulses)[n].num_instances = num_instances;

        n++;
    }

    return n;
}

/* ================================================================== */
/*  pulseg_get_rf_event_array --                                      */
/*    Build an ordered array of RF event identities for a TR region.  */
/*    Walk logic is index-aligned with pulseg_get_rf_array() above    */
/*    (same use_scan_table/start/count selection, same clamps, same   */
/*    skip conditions) -- keep the two walks in sync.                 */
/* ================================================================== */
int pulseg_get_rf_event_array(
    const pulseg_collection *coll,
    pulseg_rf_event **out_events,
    int subseq_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_descriptor *trd;
    const pulseg_block_table_element *bte;
    const pulseg_rf_definition *rfdef;
    int start, count, num_instances;
    int use_scan_table;
    int num_passes, pass_size;
    int i, n, num_rf;

    if (!coll || !out_events)
        return PULSEG_ERR_NULL_POINTER;
    *out_events = NULL;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[subseq_idx];
    trd = &desc->tr_descriptor;

    use_scan_table = 0;
    if ((!trd->degenerate_prep || !trd->degenerate_cooldown) &&
        (trd->num_prep_blocks > 0 || trd->num_cooldown_blocks > 0))
    {
        num_passes = (desc->num_passes > 1) ? desc->num_passes : 1;
        pass_size = (num_passes > 0) ? (desc->scan_table_len / num_passes) : 0;

        start = 0;
        count = pass_size;
        num_instances = num_passes;
        use_scan_table = 1;
    }
    else
    {
        int num_avgs = (desc->num_averages > 1) ? desc->num_averages : 1;
        start = trd->num_prep_blocks + trd->imaging_tr_start;
        count = trd->tr_size;
        /* Total TR instances: imaging TRs replicated by NEX,
         * plus degenerate prep/cooldown TRs (played once each). */
        num_instances = num_avgs * trd->num_trs + trd->num_prep_trs + trd->num_cooldown_trs;
        if (num_instances < 0)
            num_instances = 0;
    }

    (void)num_instances; /* not part of the event identity */

    /* Clamp to available block range */
    if (use_scan_table)
    {
        if (start + count > desc->scan_table_len)
            count = desc->scan_table_len - start;
    }
    else if (start + count > desc->num_blocks)
    {
        count = desc->num_blocks - start;
    }
    if (count < 0)
        count = 0;

    /* Pass 1: count RF-bearing blocks */
    num_rf = 0;
    for (i = 0; i < count; ++i)
    {
        int blk_idx = use_scan_table
                          ? desc->scan_table_block_idx[start + i]
                          : (start + i);
        bte = &desc->block_table[blk_idx];
        if (bte->rf_id >= 0 && bte->rf_id < desc->rf_table_size)
        {
            int id = desc->rf_table[bte->rf_id].id;
            if (id >= 0 && id < desc->num_unique_rfs)
                num_rf++;
        }
    }

    if (num_rf == 0)
        return 0;

    /* Allocate output array (caller frees with PULSEG_FREE) */
    *out_events = (pulseg_rf_event *)PULSEG_ALLOC(
        (size_t)num_rf * sizeof(pulseg_rf_event));
    if (!*out_events)
        return PULSEG_ERR_ALLOC_FAILED;

    /* Pass 2: fill entries */
    n = 0;
    for (i = 0; i < count; ++i)
    {
        int blk_idx = use_scan_table ? desc->scan_table_block_idx[start + i] : (start + i);
        int rf_def_id;
        float act_amp;

        bte = &desc->block_table[blk_idx];
        if (bte->rf_id < 0 || bte->rf_id >= desc->rf_table_size)
            continue;

        rf_def_id = desc->rf_table[bte->rf_id].id;
        if (rf_def_id < 0 || rf_def_id >= desc->num_unique_rfs)
            continue;

        rfdef = &desc->rf_definitions[rf_def_id];

        act_amp = desc->rf_table[bte->rf_id].amplitude;

        (*out_events)[n].rf_def_id = rf_def_id;
        (*out_events)[n].amplitude_hz = (act_amp >= 0.0f) ? act_amp : -act_amp;
        (*out_events)[n].rf_shim_id = bte->rf_shim_id;
        (*out_events)[n].num_channels = (rfdef->num_channels > 1) ? rfdef->num_channels : 1;

        n++;
    }

    return n;
}

/* ================================================================== */
/*  ADC collection accessors                                          */
/* ================================================================== */

static int pulseg__get_total_readouts(
    const pulseg_collection *coll)
{
    int i, n, total;

    if (!coll)
        return 0;

    total = 0;
    for (i = 0; i < coll->num_subsequences; ++i)
    {
        const pulseg_sequence_descriptor *desc = &coll->descriptors[i];
        if (!desc->scan_table_block_idx || desc->scan_table_len <= 0)
            continue;

        for (n = 0; n < desc->scan_table_len; ++n)
        {
            int bt_idx = desc->scan_table_block_idx[n];
            if (bt_idx >= 0 && bt_idx < desc->num_blocks &&
                desc->block_table[bt_idx].adc_id >= 0)
            {
                total++;
            }
        }
    }
    return total;
}

static int pulseg__get_max_adc_samples(
    const pulseg_collection *coll)
{
    int i, j, max_samples;

    if (!coll)
        return 0;

    max_samples = 0;
    for (i = 0; i < coll->num_subsequences; ++i)
    {
        for (j = 0; j < coll->descriptors[i].num_unique_adcs; ++j)
        {
            if (coll->descriptors[i].adc_definitions[j].num_samples > max_samples)
                max_samples = coll->descriptors[i].adc_definitions[j].num_samples;
        }
    }
    return max_samples;
}

static int pulseg__get_adc_dwell_ns(
    const pulseg_collection *coll, int adc_idx)
{
    int i, global_idx, num_adcs, local;

    if (!coll || adc_idx < 0 || adc_idx >= coll->total_unique_adcs)
        return 0;

    global_idx = 0;
    for (i = 0; i < coll->num_subsequences; ++i)
    {
        num_adcs = coll->descriptors[i].num_unique_adcs;
        if (adc_idx < global_idx + num_adcs)
        {
            local = adc_idx - global_idx;
            return coll->descriptors[i].adc_definitions[local].dwell_time;
        }
        global_idx += num_adcs;
    }
    return 0;
}

static int pulseg__get_adc_num_samples(
    const pulseg_collection *coll, int adc_idx)
{
    int i, global_idx, num_adcs, local;

    if (!coll || adc_idx < 0 || adc_idx >= coll->total_unique_adcs)
        return 0;

    global_idx = 0;
    for (i = 0; i < coll->num_subsequences; ++i)
    {
        num_adcs = coll->descriptors[i].num_unique_adcs;
        if (adc_idx < global_idx + num_adcs)
        {
            local = adc_idx - global_idx;
            return coll->descriptors[i].adc_definitions[local].num_samples;
        }
        global_idx += num_adcs;
    }
    return 0;
}

/* ================================================================== */
/*  Segment accessors (internal helpers for batch getters)             */
/* ================================================================== */

static int pulseg__get_num_segments(
    const pulseg_collection *coll)
{
    if (!coll)
        return 0;
    return coll->total_unique_segments;
}

static int pulseg__get_segment_duration_us(
    const pulseg_collection *coll, int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg, k, total;
    const pulseg_tr_segment *seg;
    const pulseg_block_definition *bdef;

    if (!pulseg__resolve_segment(coll, &desc, &local_seg, seg_idx))
        return -1;

    seg = &desc->segment_definitions[local_seg];

    /* Segment-definition pure delays are canonicalized to one block raster;
     * scan-loop instances still expose per-instance delay from block table. */
    if (seg->num_blocks == 1)
    {
        bdef = &desc->block_definitions[seg->unique_block_indices[0]];
        if (bdef->rf_id == -1 && bdef->gx_id == -1 && bdef->gy_id == -1 &&
            bdef->gz_id == -1 && bdef->adc_id == -1)
        {
            if (desc->block_raster_us > 0.0f)
                return (int)(desc->block_raster_us);
            return 0;
        }
    }

    total = 0;
    for (k = 0; k < seg->num_blocks; ++k)
        total += desc->block_definitions[seg->unique_block_indices[k]].duration_us;

    return total;
}

static int pulseg__is_segment_pure_delay(
    const pulseg_collection *coll, int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg;
    const pulseg_tr_segment *seg;
    const pulseg_block_definition *bdef;

    if (!pulseg__resolve_segment(coll, &desc, &local_seg, seg_idx))
        return -1;

    seg = &desc->segment_definitions[local_seg];
    if (seg->num_blocks == 1)
    {
        bdef = &desc->block_definitions[seg->unique_block_indices[0]];
        if (bdef->rf_id == -1 && bdef->gx_id == -1 &&
            bdef->gy_id == -1 && bdef->gz_id == -1 &&
            bdef->adc_id == -1 && !seg->has_digitalout[0])
            return 1;
    }
    return 0;
}

static int pulseg__get_segment_num_blocks(
    const pulseg_collection *coll, int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg;

    if (!pulseg__resolve_segment(coll, &desc, &local_seg, seg_idx))
        return -1;

    return desc->segment_definitions[local_seg].num_blocks;
}

static int pulseg__get_segment_start_block(
    const pulseg_collection *coll, int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg;

    if (!pulseg__resolve_segment(coll, &desc, &local_seg, seg_idx))
        return -1;

    return desc->segment_definitions[local_seg].start_block;
}

/* ================================================================== */
/*  Segment table queries                                             */
/* ================================================================== */

static int pulseg__get_num_prep_segments(
    const pulseg_collection *coll, int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].segment_table.num_prep_segments;
}

static int pulseg__get_num_main_segments(
    const pulseg_collection *coll, int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].segment_table.num_main_segments;
}

static int pulseg__get_num_cooldown_segments(
    const pulseg_collection *coll, int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].segment_table.num_cooldown_segments;
}

int pulseg_get_prep_segment_table(
    const pulseg_collection *coll, int *out_ids, int subseq_idx)
{
    const pulseg_sequence_descriptor *desc;
    int n;
    if (!coll || !out_ids)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;
    desc = &coll->descriptors[subseq_idx];
    n = desc->segment_table.num_prep_segments;
    if (n > 0 && desc->segment_table.prep_segment_table)
        memcpy(out_ids, desc->segment_table.prep_segment_table, n * sizeof(int));
    return n;
}

int pulseg_get_main_segment_table(
    const pulseg_collection *coll, int *out_ids, int subseq_idx)
{
    const pulseg_sequence_descriptor *desc;
    int n;
    if (!coll || !out_ids)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;
    desc = &coll->descriptors[subseq_idx];
    n = desc->segment_table.num_main_segments;
    if (n > 0 && desc->segment_table.main_segment_table)
        memcpy(out_ids, desc->segment_table.main_segment_table, n * sizeof(int));
    return n;
}

int pulseg_get_cooldown_segment_table(
    const pulseg_collection *coll, int *out_ids, int subseq_idx)
{
    const pulseg_sequence_descriptor *desc;
    int n;
    if (!coll || !out_ids)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;
    desc = &coll->descriptors[subseq_idx];
    n = desc->segment_table.num_cooldown_segments;
    if (n > 0 && desc->segment_table.cooldown_segment_table)
        memcpy(out_ids, desc->segment_table.cooldown_segment_table, n * sizeof(int));
    return n;
}

int pulseg_get_canonical_segment_sequence(
    const pulseg_collection *coll, int *out_ids, int subseq_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_descriptor *trd;
    int n_prep, n_main, n_cool, num_passes;
    int has_nd_prep, has_nd_cool;
    int total, w, p, i;

    if (!coll)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[subseq_idx];
    trd = &desc->tr_descriptor;

    n_prep = desc->segment_table.num_prep_segments;
    n_main = desc->segment_table.num_main_segments;
    n_cool = desc->segment_table.num_cooldown_segments;

    has_nd_prep = (trd->num_prep_blocks > 0 && !trd->degenerate_prep);
    has_nd_cool = (trd->num_cooldown_blocks > 0 && !trd->degenerate_cooldown);

    if (has_nd_prep || has_nd_cool)
    {
        num_passes = (desc->num_passes > 1) ? desc->num_passes : 1;
        total = n_prep + n_main * num_passes + n_cool;
    }
    else
    {
        num_passes = 1;
        total = n_main;
    }

    if (!out_ids)
        return total;

    w = 0;
    if (has_nd_prep || has_nd_cool)
    {
        for (i = 0; i < n_prep; ++i)
            out_ids[w++] = desc->segment_table.prep_segment_table[i];

        for (p = 0; p < num_passes; ++p)
            for (i = 0; i < n_main; ++i)
                out_ids[w++] = desc->segment_table.main_segment_table[i];

        for (i = 0; i < n_cool; ++i)
            out_ids[w++] = desc->segment_table.cooldown_segment_table[i];
    }
    else
    {
        for (i = 0; i < n_main; ++i)
            out_ids[w++] = desc->segment_table.main_segment_table[i];
    }

    return w;
}

/* ================================================================== */
/*  Segment timing queries                                            */
/* ================================================================== */

static int pulseg__get_segment_num_kzero_crossings(
    const pulseg_collection *coll, int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg;

    if (!pulseg__resolve_segment(coll, &desc, &local_seg, seg_idx))
        return 0;

    return desc->segment_definitions[local_seg].timing.num_kzero_crossings;
}

static int pulseg__get_segment_rf_adc_gap_us(
    const pulseg_collection *coll, int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg;
    const pulseg_segment_timing *tm;
    int r, a;

    if (!pulseg__resolve_segment(coll, &desc, &local_seg, seg_idx))
        return -1;

    tm = &desc->segment_definitions[local_seg].timing;

    /* For each RF anchor (in order), find the first ADC anchor whose
     * start is after the RF end.  Return the smallest such gap. */
    {
        int best = -1;
        for (r = 0; r < tm->num_rf_anchors; ++r)
        {
            int rf_end = tm->rf_anchors[r].end_us;
            for (a = 0; a < tm->num_adc_anchors; ++a)
            {
                int adc_start = tm->adc_anchors[a].start_us;
                if (adc_start >= rf_end)
                {
                    int gap = adc_start - rf_end;
                    if (best < 0 || gap < best)
                        best = gap;
                    break; /* first matching ADC for this RF */
                }
            }
        }
        return best;
    }
}

static int pulseg__get_segment_adc_adc_gap_us(
    const pulseg_collection *coll, int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg;
    const pulseg_segment_timing *tm;
    int a, r, best;

    if (!pulseg__resolve_segment(coll, &desc, &local_seg, seg_idx))
        return -1;

    tm = &desc->segment_definitions[local_seg].timing;
    if (tm->num_adc_anchors < 2)
        return -1;

    best = -1;
    for (a = 1; a < tm->num_adc_anchors; ++a)
    {
        int adc_prev_end = (int)tm->adc_anchors[a - 1].end_us;
        int adc_curr_start = (int)tm->adc_anchors[a].start_us;

        /* skip this ADC pair if any RF event starts between them */
        int interleaved = 0;
        for (r = 0; r < tm->num_rf_anchors; ++r)
        {
            int rf_start = (int)tm->rf_anchors[r].start_us;
            if (rf_start > adc_prev_end && rf_start < adc_curr_start)
            {
                interleaved = 1;
                break;
            }
        }
        if (interleaved)
            continue;

        {
            int gap = adc_curr_start - adc_prev_end;
            if (best < 0 || gap < best)
                best = gap;
        }
    }
    return best;
}

/* ------------------------------------------------------------------ */
/*  Per-block RF isocenter and ADC k-zero from segment timing anchors */
/* ------------------------------------------------------------------ */

float pulseg_get_rf_isocenter_us(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    const pulseg_block_definition *bdef;
    const pulseg_rf_definition *rdef;
    int local_blk, k;
    float start_us = 0.0f;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1.0f;

    for (k = 0; k < local_blk; ++k)
        start_us += (float)desc->block_definitions[seg->unique_block_indices[k]].duration_us;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id < 0 || bdef->rf_id >= desc->num_unique_rfs)
        return -1.0f;

    rdef = &desc->rf_definitions[bdef->rf_id];
    return start_us + (float)rdef->delay + (float)rdef->stats.isodelay_us;
}

float pulseg_get_adc_kzero_us(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    const pulseg_block_definition *bdef;
    const pulseg_adc_definition *adef;
    int local_blk, k;
    float start_us = 0.0f;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1.0f;

    for (k = 0; k < local_blk; ++k)
        start_us += (float)desc->block_definitions[seg->unique_block_indices[k]].duration_us;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->adc_id < 0 || bdef->adc_id >= desc->num_unique_adcs)
        return -1.0f;

    /* If calc_segment_timing has already computed a k=0 anchor for this block
     * (which handles both Cartesian N/2 and non-Cartesian kRSS-minimum),
     * return its segment-relative kzero_us directly. */
    if (seg->timing.adc_anchors && seg->timing.num_adc_anchors > 0)
    {
        for (k = 0; k < seg->timing.num_adc_anchors; ++k)
        {
            if (seg->timing.adc_anchors[k].block_offset == local_blk)
                return seg->timing.adc_anchors[k].kzero_us;
        }
    }

    /* Fallback: midpoint (Cartesian convention). */
    adef = &desc->adc_definitions[bdef->adc_id];
    return start_us + (float)adef->delay +
           (float)(adef->num_samples / 2) * (float)adef->dwell_time * 1e-3f;
}

/* ================================================================== */
/*  Block-level queries (internal helpers for batch getter)            */
/* ================================================================== */

static int pulseg__get_block_start_time_us(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk, k, start_time;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    start_time = 0;
    for (k = 0; k < local_blk; ++k)
        start_time += desc->block_definitions[seg->unique_block_indices[k]].duration_us;

    return start_time;
}

static int pulseg__get_block_duration_us(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk;
    const pulseg_block_definition *bdef;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    if (seg->num_blocks == 1 && local_blk == 0)
    {
        bdef = &desc->block_definitions[seg->unique_block_indices[0]];
        if (bdef->rf_id == -1 && bdef->gx_id == -1 && bdef->gy_id == -1 &&
            bdef->gz_id == -1 && bdef->adc_id == -1)
        {
            if (desc->block_raster_us > 0.0f)
                return (int)(desc->block_raster_us);
            return 0;
        }
    }

    return desc->block_definitions[seg->unique_block_indices[local_blk]].duration_us;
}

/* ================================================================== */
/*  RF queries (internal helpers + public waveform getters)            */
/* ================================================================== */

static int pulseg__block_has_rf(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk;
    const pulseg_block_definition *bdef;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    return (bdef->rf_id != -1) ? 1 : 0;
}

static int pulseg__block_rf_has_uniform_raster(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk;
    const pulseg_block_definition *bdef;
    const pulseg_rf_definition *rdef;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1)
        return -1;

    rdef = &desc->rf_definitions[bdef->rf_id];
    return (rdef->time_shape_id == 0) ? 1 : 0;
}

static float *pulseg__alloc_uniform_time_us(int num_samples, float raster_us)
{
    float *t;
    int i;

    if (num_samples <= 0 || raster_us <= 0.0f)
        return NULL;

    t = (float *)PULSEG_ALLOC((size_t)num_samples * sizeof(float));
    if (!t)
        return NULL;

    for (i = 0; i < num_samples; ++i)
        t[i] = ((float)i + 0.5f) * raster_us;

    return t;
}

static int pulseg__block_rf_is_complex(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk;
    const pulseg_block_definition *bdef;
    const pulseg_rf_definition *rdef;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1)
        return -1;

    rdef = &desc->rf_definitions[bdef->rf_id];
    return (rdef->phase_shape_id != 0) ? 1 : 0;
}

static int pulseg__get_rf_num_samples(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk, shape_idx, total, nch;
    const pulseg_block_definition *bdef;
    const pulseg_rf_definition *rdef;
    const pulseg_shape_arbitrary *shape;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1)
        return -1;

    rdef = &desc->rf_definitions[bdef->rf_id];
    total = -1;

    /* try mag, then phase, then time shape */
    if (rdef->mag_shape_id > 0)
    {
        shape_idx = rdef->mag_shape_id - 1;
        if (shape_idx >= 0 && shape_idx < desc->num_shapes)
        {
            shape = &desc->shapes[shape_idx];
            if (shape->num_uncompressed_samples > 0)
                total = shape->num_uncompressed_samples;
        }
    }
    if (total < 0 && rdef->phase_shape_id > 0)
    {
        shape_idx = rdef->phase_shape_id - 1;
        if (shape_idx >= 0 && shape_idx < desc->num_shapes)
        {
            shape = &desc->shapes[shape_idx];
            if (shape->num_uncompressed_samples > 0)
                total = shape->num_uncompressed_samples;
        }
    }
    if (total < 0 && rdef->time_shape_id > 0)
    {
        shape_idx = rdef->time_shape_id - 1;
        if (shape_idx >= 0 && shape_idx < desc->num_shapes)
        {
            shape = &desc->shapes[shape_idx];
            if (shape->num_uncompressed_samples > 0)
                total = shape->num_uncompressed_samples;
        }
    }
    if (total < 0)
        return -1;

    nch = (rdef->num_channels > 1) ? rdef->num_channels : 1;
    return total / nch;
}

static int pulseg__get_rf_delay_us(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk;
    const pulseg_block_definition *bdef;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1)
        return -1;

    return desc->rf_definitions[bdef->rf_id].delay;
}

static int pulseg__get_rf_duration_us(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk;
    const pulseg_block_definition *bdef;
    const pulseg_rf_definition *rdef;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1)
        return -1;

    rdef = &desc->rf_definitions[bdef->rf_id];
    /* stats.duration_us is the last time-shape sample time (accounts for
     * custom non-uniform time shapes). Round to nearest integer µs. */
    return (int)(rdef->stats.duration_us);
}

static int pulseg__get_rf_num_channels(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk;
    const pulseg_block_definition *bdef;
    const pulseg_rf_definition *rdef;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1)
        return -1;

    rdef = &desc->rf_definitions[bdef->rf_id];
    return (rdef->num_channels > 1) ? rdef->num_channels : 1;
}

#define PULSEG__RF_SHAPE_MAG   0
#define PULSEG__RF_SHAPE_PHASE 1

/* Shared decompress+split body for magnitude/phase RF waveform getters,
 * keyed off an already-resolved rf definition. Returns a malloc'd
 * (PULSEG_ALLOC) array of *num_channels row pointers, each *num_samples
 * floats, or NULL (shape absent / decompress failure / alloc failure). */
static float **pulseg__get_rf_def_shape(
    const pulseg_sequence_descriptor *desc,
    const pulseg_rf_definition *rdef,
    int which_shape,
    int *num_channels,
    int *num_samples)
{
    int shape_id, shape_idx, nch, npts, ch;
    pulseg_shape_arbitrary decompressed;
    float *flat;
    float **result;

    *num_channels = 0;
    *num_samples = 0;

    shape_id = (which_shape == PULSEG__RF_SHAPE_MAG) ? rdef->mag_shape_id : rdef->phase_shape_id;
    if (shape_id <= 0)
        return NULL;

    shape_idx = shape_id - 1;
    if (shape_idx < 0 || shape_idx >= desc->num_shapes)
        return NULL;

    decompressed.num_samples = 0;
    decompressed.num_uncompressed_samples = 0;
    decompressed.samples = NULL;

    if (!pulseg_pulseq_decompress_shape(&decompressed, &desc->shapes[shape_idx], 1.0f))
        return NULL;

    flat = decompressed.samples;
    nch = (rdef->num_channels > 1) ? rdef->num_channels : 1;
    npts = decompressed.num_samples / nch;

    /* allocate channel-pointer array */
    result = (float **)PULSEG_ALLOC((size_t)nch * sizeof(float *));
    if (!result)
    {
        PULSEG_FREE(flat);
        return NULL;
    }

    /* split tiled flat array into per-channel rows */
    for (ch = 0; ch < nch; ++ch)
    {
        result[ch] = (float *)PULSEG_ALLOC((size_t)npts * sizeof(float));
        if (!result[ch])
        {
            int k;
            for (k = 0; k < ch; ++k)
                PULSEG_FREE(result[k]);
            PULSEG_FREE(result);
            PULSEG_FREE(flat);
            return NULL;
        }
        memcpy(result[ch], flat + ch * npts, (size_t)npts * sizeof(float));
    }

    PULSEG_FREE(flat);
    *num_channels = nch;
    *num_samples = npts;
    return result;
}

float **pulseg_get_rf_magnitude(const pulseg_collection *coll,
                                   int *num_channels,
                                   int *num_samples,
                                   int seg_idx,
                                   int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk;
    const pulseg_block_definition *bdef;
    const pulseg_rf_definition *rdef;

    if (!num_channels || !num_samples)
        return NULL;
    *num_channels = 0;
    *num_samples = 0;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return NULL;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1)
        return NULL;

    rdef = &desc->rf_definitions[bdef->rf_id];
    return pulseg__get_rf_def_shape(desc, rdef, PULSEG__RF_SHAPE_MAG, num_channels, num_samples);
}

float **pulseg_get_rf_phase(const pulseg_collection *coll,
                               int *num_channels,
                               int *num_samples,
                               int seg_idx,
                               int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk;
    const pulseg_block_definition *bdef;
    const pulseg_rf_definition *rdef;

    if (!num_channels || !num_samples)
        return NULL;
    *num_channels = 0;
    *num_samples = 0;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return NULL;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1)
        return NULL;

    rdef = &desc->rf_definitions[bdef->rf_id];
    return pulseg__get_rf_def_shape(desc, rdef, PULSEG__RF_SHAPE_PHASE, num_channels, num_samples);
}

float *pulseg_get_rf_time_us(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk, shape_idx, nch, npts;
    const pulseg_block_definition *bdef;
    const pulseg_rf_definition *rdef;
    pulseg_shape_arbitrary decompressed;
    float *result;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return NULL;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1)
        return NULL;

    rdef = &desc->rf_definitions[bdef->rf_id];
    if (rdef->time_shape_id <= 0)
    {
        npts = pulseg__get_rf_num_samples(coll, seg_idx, blk_idx);
        return pulseg__alloc_uniform_time_us(npts, desc->rf_raster_us);
    }

    shape_idx = rdef->time_shape_id - 1;
    if (shape_idx < 0 || shape_idx >= desc->num_shapes)
        return NULL;

    decompressed.num_samples = 0;
    decompressed.num_uncompressed_samples = 0;
    decompressed.samples = NULL;

    if (!pulseg_pulseq_decompress_shape(&decompressed, &desc->shapes[shape_idx],
                                     desc->rf_raster_us))
        return NULL;

    nch = (rdef->num_channels > 1) ? rdef->num_channels : 1;
    npts = decompressed.num_samples / nch;

    if (nch > 1)
    {
        /* return only first channel's time (all channels share time base) */
        result = (float *)PULSEG_ALLOC((size_t)npts * sizeof(float));
        if (!result)
        {
            PULSEG_FREE(decompressed.samples);
            return NULL;
        }
        memcpy(result, decompressed.samples, (size_t)npts * sizeof(float));
        PULSEG_FREE(decompressed.samples);
    }
    else
    {
        result = decompressed.samples;
    }

    return result;
}

/* ================================================================== */
/*  Definition-keyed RF waveform getters (pTx SAR: keyed by subseq_idx +   */
/*  rf_def_id, not seg/blk -- one lookup per unique RF definition,        */
/*  independent of where it is played in the sequence).                   */
/* ================================================================== */

float **pulseg_get_rf_def_magnitude(const pulseg_collection *coll,
                                    int *num_channels, int *num_samples,
                                    int subseq_idx, int rf_def_id)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_rf_definition *rdef;

    if (!num_channels || !num_samples)
        return NULL;
    *num_channels = 0;
    *num_samples = 0;

    if (!coll)
        return NULL;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return NULL;

    desc = &coll->descriptors[subseq_idx];
    if (rf_def_id < 0 || rf_def_id >= desc->num_unique_rfs)
        return NULL;

    rdef = &desc->rf_definitions[rf_def_id];
    return pulseg__get_rf_def_shape(desc, rdef, PULSEG__RF_SHAPE_MAG, num_channels, num_samples);
}

float **pulseg_get_rf_def_phase(const pulseg_collection *coll,
                                int *num_channels, int *num_samples,
                                int subseq_idx, int rf_def_id)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_rf_definition *rdef;

    if (!num_channels || !num_samples)
        return NULL;
    *num_channels = 0;
    *num_samples = 0;

    if (!coll)
        return NULL;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return NULL;

    desc = &coll->descriptors[subseq_idx];
    if (rf_def_id < 0 || rf_def_id >= desc->num_unique_rfs)
        return NULL;

    rdef = &desc->rf_definitions[rf_def_id];
    return pulseg__get_rf_def_shape(desc, rdef, PULSEG__RF_SHAPE_PHASE, num_channels, num_samples);
}

float *pulseg_get_rf_def_time(const pulseg_collection *coll,
                              int *num_samples,
                              int subseq_idx, int rf_def_id)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_rf_definition *rdef;
    int shape_idx, nch, npts;
    pulseg_shape_arbitrary decompressed;
    float *result;

    if (!num_samples)
        return NULL;
    *num_samples = 0;

    if (!coll)
        return NULL;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return NULL;

    desc = &coll->descriptors[subseq_idx];
    if (rf_def_id < 0 || rf_def_id >= desc->num_unique_rfs)
        return NULL;

    rdef = &desc->rf_definitions[rf_def_id];
    if (rdef->time_shape_id <= 0)
        return NULL; /* no time shape -- caller falls back to uniform raster */

    shape_idx = rdef->time_shape_id - 1;
    if (shape_idx < 0 || shape_idx >= desc->num_shapes)
        return NULL;

    decompressed.num_samples = 0;
    decompressed.num_uncompressed_samples = 0;
    decompressed.samples = NULL;

    if (!pulseg_pulseq_decompress_shape(&decompressed, &desc->shapes[shape_idx],
                                     desc->rf_raster_us))
        return NULL;

    nch = (rdef->num_channels > 1) ? rdef->num_channels : 1;
    npts = decompressed.num_samples / nch;

    if (nch > 1)
    {
        /* return only first channel's time (all channels share time base) */
        result = (float *)PULSEG_ALLOC((size_t)npts * sizeof(float));
        if (!result)
        {
            PULSEG_FREE(decompressed.samples);
            return NULL;
        }
        memcpy(result, decompressed.samples, (size_t)npts * sizeof(float));
        PULSEG_FREE(decompressed.samples);
    }
    else
    {
        result = decompressed.samples;
    }

    *num_samples = npts;
    return result;
}

float pulseg_get_rf_initial_amplitude_hz(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk, rf_event_id;
    const pulseg_block_table_element *bte;
    const pulseg_block_definition *bdef;
    const pulseg_rf_definition *rdef;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return 0.0f;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1)
        return 0.0f;

    rdef = &desc->rf_definitions[bdef->rf_id];

    bte = resolve_block_table_via_max_energy(desc, seg, local_blk);
    if (!bte)
        return rdef->stats.base_amplitude_hz;

    rf_event_id = bte->rf_id;
    if (rf_event_id >= 0 && rf_event_id < desc->rf_table_size)
        return desc->rf_table[rf_event_id].amplitude;

    return rdef->stats.base_amplitude_hz;
}

float pulseg_get_rf_max_amplitude_hz(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk, rf_event_id;
    const pulseg_block_table_element *bte;
    const pulseg_block_definition *bdef;
    const pulseg_rf_definition *rdef;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return 0.0f;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1)
        return 0.0f;

    rdef = &desc->rf_definitions[bdef->rf_id];

    bte = resolve_block_table_via_max_energy(desc, seg, local_blk);
    if (!bte)
        return (float)fabs(rdef->stats.base_amplitude_hz);

    rf_event_id = bte->rf_id;
    if (rf_event_id >= 0 && rf_event_id < desc->rf_table_size)
        return (float)fabs(desc->rf_table[rf_event_id].amplitude);

    return (float)fabs(rdef->stats.base_amplitude_hz);
}

/* ================================================================== */
/*  Gradient queries (internal helpers + public waveform getters)      */
/* ================================================================== */

static int pulseg__block_has_grad(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx, int axis)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk, grad_id;

    if (axis < PULSEG_GRAD_AXIS_X || axis > PULSEG_GRAD_AXIS_Z)
        return -1;
    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    grad_id = resolve_grad_def_via_max_energy_instance(desc, seg, local_blk, axis);
    return (grad_id != -1) ? 1 : 0;
}

static int pulseg__block_grad_is_trapezoid(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx, int axis)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk, grad_id;
    const pulseg_grad_definition *gdef;

    if (axis < PULSEG_GRAD_AXIS_X || axis > PULSEG_GRAD_AXIS_Z)
        return -1;
    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    grad_id = resolve_grad_def_via_max_energy_instance(desc, seg, local_blk, axis);
    if (grad_id == -1)
        return -1;

    gdef = &desc->grad_definitions[grad_id];
    if (gdef->type == 0)
        return 1;
    if (gdef->unused_or_time_shape_id > 0)
        return 1;
    return 0;
}

static int pulseg__get_grad_num_samples(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx, int axis)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk, grad_id, shape_idx;
    const pulseg_grad_definition *gdef;
    const pulseg_shape_arbitrary *shape;

    if (axis < PULSEG_GRAD_AXIS_X || axis > PULSEG_GRAD_AXIS_Z)
        return -1;
    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    grad_id = resolve_grad_def_via_max_energy_instance(desc, seg, local_blk, axis);
    if (grad_id == -1)
        return -1;

    gdef = &desc->grad_definitions[grad_id];

    if (gdef->type == 0)
    {
        return (gdef->flat_time_or_unused > 0) ? 4 : 3;
    }

    /* arbitrary: get from first shot shape */
    if (gdef->num_shots > 0 && gdef->shot_shape_ids[0] > 0)
    {
        shape_idx = gdef->shot_shape_ids[0] - 1;
        if (shape_idx >= 0 && shape_idx < desc->num_shapes)
        {
            shape = &desc->shapes[shape_idx];
            if (shape->num_uncompressed_samples > 0)
                return shape->num_uncompressed_samples;
        }
    }
    return -1;
}

static int pulseg__get_grad_num_shots(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx, int axis)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk, grad_id;

    if (axis < PULSEG_GRAD_AXIS_X || axis > PULSEG_GRAD_AXIS_Z)
        return -1;
    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    grad_id = resolve_grad_def_via_max_energy_instance(desc, seg, local_blk, axis);
    if (grad_id == -1)
        return -1;

    return desc->grad_definitions[grad_id].num_shots;
}

static int pulseg__get_grad_delay_us(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx, int axis)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk, grad_id;

    if (axis < PULSEG_GRAD_AXIS_X || axis > PULSEG_GRAD_AXIS_Z)
        return -1;
    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    grad_id = resolve_grad_def_via_max_energy_instance(desc, seg, local_blk, axis);
    if (grad_id == -1)
        return -1;

    return desc->grad_definitions[grad_id].delay;
}

float **pulseg_get_grad_amplitude(const pulseg_collection *coll,
                                     int *num_shots,
                                     int *num_samples,
                                     int seg_idx,
                                     int blk_idx,
                                     int axis)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk, grad_id, shot, k, shape_idx;
    int raw_grad_id;
    const pulseg_block_table_element *bte;
    const pulseg_grad_definition *gdef;
    float **waveforms;
    float *trap_waveform;
    int samples_per_shot;
    int flat_time;
    pulseg_shape_arbitrary decompressed;

    if (!num_shots || !num_samples)
    {
        if (num_shots)
            *num_shots = 0;
        if (num_samples)
            *num_samples = 0;
        return NULL;
    }
    *num_shots = 0;
    *num_samples = 0;

    if (axis < PULSEG_GRAD_AXIS_X || axis > PULSEG_GRAD_AXIS_Z)
        return NULL;
    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return NULL;

    /* Resolve through the max-energy block_table entry so that the
     * grad_definition (and thus the set of shot waveforms) matches the
     * actual physical block at the representative instance.  This is
     * necessary when different segment instances have gradients with
     * different time_shape_ids, which places them in separate
     * grad_definitions despite occupying the same segment position. */
    bte = resolve_block_table_via_max_energy(desc, seg, local_blk);
    if (!bte)
        return NULL;
    switch (axis)
    {
    case PULSEG_GRAD_AXIS_X:
        raw_grad_id = bte->gx_id;
        break;
    case PULSEG_GRAD_AXIS_Y:
        raw_grad_id = bte->gy_id;
        break;
    case PULSEG_GRAD_AXIS_Z:
        raw_grad_id = bte->gz_id;
        break;
    default:
        raw_grad_id = -1;
        break;
    }
    if (raw_grad_id < 0 || raw_grad_id >= desc->grad_table_size)
        return NULL;
    grad_id = desc->grad_table[raw_grad_id].id;
    gdef = &desc->grad_definitions[grad_id];

    waveforms = (float **)PULSEG_ALLOC(gdef->num_shots * sizeof(float *));
    if (!waveforms)
        return NULL;

    *num_shots = gdef->num_shots;

    if (gdef->type == 0)
    {
        flat_time = gdef->flat_time_or_unused;
        samples_per_shot = (flat_time > 0) ? 4 : 3;
        *num_samples = samples_per_shot;

        for (shot = 0; shot < gdef->num_shots; ++shot)
        {
            trap_waveform = (float *)PULSEG_ALLOC(samples_per_shot * sizeof(float));
            if (!trap_waveform)
            {
                for (k = 0; k < shot; ++k)
                    PULSEG_FREE(waveforms[k]);
                PULSEG_FREE(waveforms);
                *num_shots = 0;
                *num_samples = 0;
                return NULL;
            }

            trap_waveform[0] = 0.0f;
            trap_waveform[1] = 1.0f;
            if (flat_time > 0)
            {
                trap_waveform[2] = 1.0f;
                trap_waveform[3] = 0.0f;
            }
            else
            {
                trap_waveform[2] = 0.0f;
            }

            waveforms[shot] = trap_waveform;
        }
    }
    else
    {
        for (shot = 0; shot < gdef->num_shots; ++shot)
        {
            if (gdef->shot_shape_ids[shot] <= 0)
            {
                waveforms[shot] = NULL;
                continue;
            }

            shape_idx = gdef->shot_shape_ids[shot] - 1;
            if (shape_idx < 0 || shape_idx >= desc->num_shapes)
            {
                waveforms[shot] = NULL;
                continue;
            }

            decompressed.num_samples = 0;
            decompressed.num_uncompressed_samples = 0;
            decompressed.samples = NULL;

            if (!pulseg_pulseq_decompress_shape(&decompressed, &desc->shapes[shape_idx],
                                             1.0f))
            {
                waveforms[shot] = NULL;
                continue;
            }

            waveforms[shot] = decompressed.samples;
            if (*num_samples == 0)
                *num_samples = decompressed.num_samples;
        }
    }

    (void)raw_grad_id; /* initial shot reported via pulseg_get_grad_initial_shot_id */

    return waveforms;
}

float pulseg_get_grad_initial_amplitude_hz_per_m(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx, int axis)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk, grad_event_id;
    const pulseg_block_table_element *bte;

    if (axis < PULSEG_GRAD_AXIS_X || axis > PULSEG_GRAD_AXIS_Z)
        return 1.0f;
    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return 1.0f;

    bte = resolve_block_table_via_max_energy(desc, seg, local_blk);
    if (!bte)
        return 1.0f;

    switch (axis)
    {
    case PULSEG_GRAD_AXIS_X:
        grad_event_id = bte->gx_id;
        break;
    case PULSEG_GRAD_AXIS_Y:
        grad_event_id = bte->gy_id;
        break;
    case PULSEG_GRAD_AXIS_Z:
        grad_event_id = bte->gz_id;
        break;
    default:
        return 1.0f;
    }

    if (grad_event_id < 0 || grad_event_id >= desc->grad_table_size)
        return 1.0f;

    return desc->grad_table[grad_event_id].amplitude;
}

int pulseg_get_grad_initial_shot_id(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx, int axis)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk, grad_event_id;
    const pulseg_block_table_element *bte;

    if (axis < PULSEG_GRAD_AXIS_X || axis > PULSEG_GRAD_AXIS_Z)
        return 0;
    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return 0;

    bte = resolve_block_table_via_max_energy(desc, seg, local_blk);
    if (!bte)
        return 0;

    switch (axis)
    {
    case PULSEG_GRAD_AXIS_X:
        grad_event_id = bte->gx_id;
        break;
    case PULSEG_GRAD_AXIS_Y:
        grad_event_id = bte->gy_id;
        break;
    case PULSEG_GRAD_AXIS_Z:
        grad_event_id = bte->gz_id;
        break;
    default:
        return 0;
    }

    if (grad_event_id < 0 || grad_event_id >= desc->grad_table_size)
        return 0;

    return desc->grad_table[grad_event_id].shot_index;
}

float pulseg_get_grad_max_amplitude_hz_per_m(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx, int axis)
{
    float init_amp;

    if (axis < PULSEG_GRAD_AXIS_X || axis > PULSEG_GRAD_AXIS_Z)
        return 0.0f;
    init_amp = pulseg_get_grad_initial_amplitude_hz_per_m(
        coll, seg_idx, blk_idx, axis);
    return (init_amp >= 0.0f) ? init_amp : -init_amp;
}

float *pulseg_get_grad_time_us(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx, int axis)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk, grad_id, shape_idx;
    int raw_grad_id;
    const pulseg_block_table_element *bte;
    const pulseg_grad_definition *gdef;
    float *time_waveform;
    float accum;
    int rise_time, flat_time, fall_time, ns;
    pulseg_shape_arbitrary decompressed;

    if (axis < PULSEG_GRAD_AXIS_X || axis > PULSEG_GRAD_AXIS_Z)
        return NULL;
    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return NULL;

    /* Resolve through actual max-energy block_table entry (see comment
     * in pulseg_get_grad_amplitude for rationale). */
    bte = resolve_block_table_via_max_energy(desc, seg, local_blk);
    if (!bte)
        return NULL;
    switch (axis)
    {
    case PULSEG_GRAD_AXIS_X:
        raw_grad_id = bte->gx_id;
        break;
    case PULSEG_GRAD_AXIS_Y:
        raw_grad_id = bte->gy_id;
        break;
    case PULSEG_GRAD_AXIS_Z:
        raw_grad_id = bte->gz_id;
        break;
    default:
        raw_grad_id = -1;
        break;
    }
    if (raw_grad_id < 0 || raw_grad_id >= desc->grad_table_size)
        return NULL;
    grad_id = desc->grad_table[raw_grad_id].id;
    gdef = &desc->grad_definitions[grad_id];

    if (gdef->type == 0)
    {
        rise_time = gdef->rise_time_or_unused;
        flat_time = gdef->flat_time_or_unused;
        fall_time = gdef->fall_time_or_num_uncompressed_samples;

        ns = (flat_time > 0) ? 4 : 3;

        time_waveform = (float *)PULSEG_ALLOC((size_t)ns * sizeof(float));
        if (!time_waveform)
            return NULL;

        accum = 0.0f;
        time_waveform[0] = accum;
        accum += (float)rise_time;
        time_waveform[1] = accum;
        if (flat_time > 0)
        {
            accum += (float)flat_time;
            time_waveform[2] = accum;
            accum += (float)fall_time;
            time_waveform[3] = accum;
        }
        else
        {
            accum += (float)fall_time;
            time_waveform[2] = accum;
        }

        return time_waveform;
    }

    /* arbitrary: decompress time shape */
    if (gdef->unused_or_time_shape_id <= 0)
    {
        ns = pulseg__get_grad_num_samples(coll, seg_idx, blk_idx, axis);
        return pulseg__alloc_uniform_time_us(ns, desc->grad_raster_us);
    }

    shape_idx = gdef->unused_or_time_shape_id - 1;
    if (shape_idx < 0 || shape_idx >= desc->num_shapes)
        return NULL;

    decompressed.num_samples = 0;
    decompressed.num_uncompressed_samples = 0;
    decompressed.samples = NULL;

    if (!pulseg_pulseq_decompress_shape(&decompressed, &desc->shapes[shape_idx],
                                     desc->grad_raster_us))
        return NULL;

    return decompressed.samples;
}

/* ================================================================== */
/*  ADC block queries (internal helpers)                               */
/* ================================================================== */

static int pulseg__block_has_adc(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    /* Use OR-reduced flag: true if at least one segment instance has an ADC
     * event at this block position (not just the canonical block def). */
    if (seg->has_adc && seg->has_adc[local_blk])
        return 1;

    /* Fallback to canonical block definition (pre-11d sequences / cache) */
    {
        const pulseg_block_definition *bdef =
            &desc->block_definitions[seg->unique_block_indices[local_blk]];
        return (bdef->adc_id != -1) ? 1 : 0;
    }
}

static int pulseg__get_adc_delay_us(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk, adc_id;
    const pulseg_block_definition *bdef;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    adc_id = bdef->adc_id;
    if (adc_id < 0 || adc_id >= desc->num_unique_adcs)
        return -1;

    return desc->adc_definitions[adc_id].delay;
}

static int pulseg__get_adc_library_index(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk, adc_id, global_adc_idx, i;
    const pulseg_block_definition *bdef;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    adc_id = bdef->adc_id;
    if (adc_id < 0 || adc_id >= desc->num_unique_adcs)
        return -1;

    /* compute global index: sum ADC counts from prior subsequences */
    global_adc_idx = 0;
    for (i = 0; i < coll->num_subsequences; ++i)
    {
        if (&coll->descriptors[i] == desc)
            break;
        global_adc_idx += coll->descriptors[i].num_unique_adcs;
    }
    return global_adc_idx + adc_id;
}

/* ================================================================== */
/*  Digital output / trigger / flag queries (internal helpers)         */
/* ================================================================== */

static int pulseg__block_has_digitalout(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    return seg->has_digitalout[local_blk];
}

static int pulseg__get_digitalout_delay_us(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk, digitalout_id;
    const pulseg_block_table_element *bte;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    if (!seg->has_digitalout[local_blk])
        return -1;

    bte = &desc->block_table[seg->start_block + local_blk];
    digitalout_id = bte->digitalout_id;
    if (digitalout_id == -1 || digitalout_id >= desc->num_triggers)
        return -1;

    return (int)desc->trigger_events[digitalout_id].delay;
}

static int pulseg__get_digitalout_duration_us(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk, digitalout_id;
    const pulseg_block_table_element *bte;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    if (!seg->has_digitalout[local_blk])
        return -1;

    bte = &desc->block_table[seg->start_block + local_blk];
    digitalout_id = bte->digitalout_id;
    if (digitalout_id == -1 || digitalout_id >= desc->num_triggers)
        return -1;

    return (int)desc->trigger_events[digitalout_id].duration;
}

static int pulseg__get_digitalout_channel(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk, digitalout_id;
    const pulseg_block_table_element *bte;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    if (!seg->has_digitalout[local_blk])
        return -1;

    bte = &desc->block_table[seg->start_block + local_blk];
    digitalout_id = bte->digitalout_id;
    if (digitalout_id == -1 || digitalout_id >= desc->num_triggers)
        return -1;

    return desc->trigger_events[digitalout_id].trigger_channel;
}

/* ---- Segment-level physio trigger queries ------------------------ */

static int pulseg__segment_has_trigger(
    const pulseg_collection *coll,
    int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg;

    if (!pulseg__resolve_segment(coll, &desc, &local_seg, seg_idx))
        return 0;

    return (desc->segment_definitions[local_seg].trigger_id >= 0) ? 1 : 0;
}

static int pulseg__get_segment_trigger_delay_us(
    const pulseg_collection *coll,
    int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg, tid;

    if (!pulseg__resolve_segment(coll, &desc, &local_seg, seg_idx))
        return -1;

    tid = desc->segment_definitions[local_seg].trigger_id;
    if (tid < 0 || tid >= desc->num_triggers)
        return -1;

    return (int)desc->trigger_events[tid].delay;
}

static int pulseg__get_segment_trigger_duration_us(
    const pulseg_collection *coll,
    int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg, tid;

    if (!pulseg__resolve_segment(coll, &desc, &local_seg, seg_idx))
        return -1;

    tid = desc->segment_definitions[local_seg].trigger_id;
    if (tid < 0 || tid >= desc->num_triggers)
        return -1;

    return (int)desc->trigger_events[tid].duration;
}

static int pulseg__get_segment_trigger_type(
    const pulseg_collection *coll,
    int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg, tid;

    if (!pulseg__resolve_segment(coll, &desc, &local_seg, seg_idx))
        return 0;

    tid = desc->segment_definitions[local_seg].trigger_id;
    if (tid < 0 || tid >= desc->num_triggers)
        return 0;

    return desc->trigger_events[tid].trigger_type;
}

/* ---- Navigator flag query ---------------------------------------- */

static int pulseg__segment_is_nav(
    const pulseg_collection *coll,
    int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg;

    if (!pulseg__resolve_segment(coll, &desc, &local_seg, seg_idx))
        return 0;

    return desc->segment_definitions[local_seg].is_nav;
}

static int pulseg__block_has_freq_mod(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return 0;

    return seg->has_freq_mod[local_blk];
}

static int pulseg__block_has_rotation(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    return seg->has_rotation[local_blk];
}

static int pulseg__block_has_norot(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    return seg->norot_flag[local_blk];
}

static int pulseg__block_has_nopos(
    const pulseg_collection *coll,
    int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    return seg->nopos_flag[local_blk];
}

/* ------------------------------------------------------------------ */
/*  pulseg_block_needs_freq_mod — precise overlap + nopos check    */
/* ------------------------------------------------------------------ */

/*
 * Check whether a trapezoid gradient's flat region overlaps [win_start, win_end].
 * All times in us, relative to block start.
 */
static int trap_overlaps_window(
    const pulseg_grad_definition *gdef,
    float win_start, float win_end)
{
    float flat_start = (float)gdef->delay +
                       (float)gdef->rise_time_or_unused;
    float flat_end = flat_start + (float)gdef->flat_time_or_unused;
    return (flat_start < win_end) && (flat_end > win_start);
}

/*
 * Check whether an arbitrary gradient waveform has any nonzero sample
 * within [win_start, win_end].  Times in us, relative to block start.
 */
static int arb_nonzero_in_window(
    const pulseg_sequence_descriptor *desc,
    const pulseg_grad_definition *gdef,
    float win_start, float win_end)
{
    int shape_idx, i, ns;
    float raster, local_start, local_end;
    int idx_lo, idx_hi;
    pulseg_shape_arbitrary decomp;

    if (gdef->num_shots < 1 || gdef->shot_shape_ids[0] <= 0)
        return 0;

    shape_idx = gdef->shot_shape_ids[0] - 1;
    if (shape_idx < 0 || shape_idx >= desc->num_shapes)
        return 0;

    decomp.num_samples = 0;
    decomp.num_uncompressed_samples = 0;
    decomp.samples = NULL;
    if (!pulseg_pulseq_decompress_shape(&decomp, &desc->shapes[shape_idx], 1.0f))
        return 0;

    ns = decomp.num_samples;

    /* If the gradient has a non-uniform time shape, use actual sample times
     * instead of assuming uniform raster.  Uniform-raster index arithmetic
     * fails for compressed time-shaped gradients: the amplitude shape may
     * have far fewer samples than block_duration / grad_raster, so the
     * index computed from win_start/raster exceeds ns-1 and the loop body
     * never executes.
     *
     * Each time-shaped sample i represents a step (hold) from t[i] to
     * t[i+1] (or to the end of the last defined sample for i=ns-1).
     * A sample i with nonzero amplitude contributes within the window if
     * its hold interval [t[i], t[i+1]) overlaps [win_start, win_end].    */
    if (gdef->unused_or_time_shape_id > 0)
    {
        int ts_idx = gdef->unused_or_time_shape_id - 1;
        if (ts_idx >= 0 && ts_idx < desc->num_shapes)
        {
            pulseg_shape_arbitrary decomp_time;
            decomp_time.num_samples = 0;
            decomp_time.num_uncompressed_samples = 0;
            decomp_time.samples = NULL;
            if (pulseg_pulseq_decompress_shape(&decomp_time,
                                            &desc->shapes[ts_idx],
                                            desc->grad_raster_us))
            {
                int found = 0;
                int tns = decomp_time.num_samples;
                for (i = 0; i < ns && i < tns; ++i)
                {
                    float t_start = (float)gdef->delay + decomp_time.samples[i];
                    /* Each sample holds until the next sample (or is the last) */
                    float t_end = (i + 1 < tns)
                                      ? ((float)gdef->delay + decomp_time.samples[i + 1])
                                      : t_start + desc->grad_raster_us;
                    /* Interval [t_start, t_end) overlaps (win_start, win_end] */
                    if (t_end > win_start && t_start <= win_end && decomp.samples[i] != 0.0f)
                    {
                        found = 1;
                        break;
                    }
                }
                PULSEG_FREE(decomp_time.samples);
                PULSEG_FREE(decomp.samples);
                return found;
            }
        }
    }

    /* Uniform raster: convert window to sample-index range */
    raster = desc->grad_raster_us;
    local_start = win_start - (float)gdef->delay;
    local_end = win_end - (float)gdef->delay;

    idx_lo = (int)(local_start / raster);
    if (idx_lo < 0)
        idx_lo = 0;
    idx_hi = (int)(local_end / raster);
    if (idx_hi >= ns)
        idx_hi = ns - 1;

    for (i = idx_lo; i <= idx_hi; ++i)
    {
        if (decomp.samples[i] != 0.0f)
        {
            PULSEG_FREE(decomp.samples);
            return 1;
        }
    }

    PULSEG_FREE(decomp.samples);
    return 0;
}

/*
 * Check whether any gradient axis has nonzero amplitude within the
 * given temporal window (us, relative to block start).
 */
static int any_grad_overlaps_window(
    const pulseg_sequence_descriptor *desc,
    const pulseg_block_definition *bdef,
    float win_start, float win_end)
{
    int axis, grad_id;

    for (axis = 0; axis < 3; ++axis)
    {
        grad_id = get_grad_id_by_axis(bdef, axis);
        if (grad_id < 0)
            continue;

        if (desc->grad_definitions[grad_id].type == 0)
        {
            /* trapezoid */
            if (trap_overlaps_window(&desc->grad_definitions[grad_id],
                                     win_start, win_end))
                return 1;
        }
        else
        {
            /* arbitrary */
            if (arb_nonzero_in_window(desc, &desc->grad_definitions[grad_id],
                                      win_start, win_end))
                return 1;
        }
    }
    return 0;
}

int pulseg_block_needs_freq_mod(const pulseg_collection *coll,
                                   int *num_samples,
                                   int seg_idx,
                                   int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_blk;
    const pulseg_block_definition *bdef;
    int has_rf, has_adc, nopos;

    if (num_samples)
        *num_samples = 0;

    if (!pulseg__resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return 0;

    nopos = seg->nopos_flag[local_blk];
    if (nopos)
        return 0;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    has_rf = (bdef->rf_id >= 0);
    /* Use OR-reduced has_adc: true if any instance at this position has ADC */
    has_adc = (seg->has_adc && seg->has_adc[local_blk]) ? 1 : (bdef->adc_id >= 0);
    if (!has_rf && !has_adc)
        return 0;

    /* Check RF window overlap */
    if (has_rf)
    {
        const pulseg_rf_definition *rdef = &desc->rf_definitions[bdef->rf_id];
        float rf_start = (float)rdef->delay;
        float rf_end = rf_start + rdef->stats.duration_us;

        if (any_grad_overlaps_window(desc, bdef, rf_start, rf_end))
        {
            if (num_samples)
                *num_samples = (int)((float)bdef->duration_us /
                                     desc->rf_raster_us);
            return 1;
        }
    }

    /* Check ADC window overlap */
    if (has_adc)
    {
        const pulseg_adc_definition *adef = &desc->adc_definitions[bdef->adc_id];
        float adc_start = (float)adef->delay;
        float adc_end = adc_start +
                        (float)adef->num_samples *
                            (float)adef->dwell_time * 1e-3f;

        if (any_grad_overlaps_window(desc, bdef, adc_start, adc_end))
        {
            if (num_samples)
                *num_samples = (int)((float)bdef->duration_us /
                                     desc->adc_raster_us);
            return 1;
        }
    }

    return 0;
}

int pulseg_cursor_next(pulseg_collection *coll)
{
    pulseg_block_cursor *cursor;
    const pulseg_sequence_descriptor *desc;
    int next_pos;

    cursor = &coll->block_cursor;

    if (cursor->sequence_index >= coll->num_subsequences)
        return PULSEG_CURSOR_DONE;

    desc = &coll->descriptors[cursor->sequence_index];
    next_pos = cursor->scan_table_position + 1;

    /* Past end of scan table: advance to next subsequence */
    if (next_pos >= desc->scan_table_len)
    {
        cursor->sequence_index += 1;
        cursor->scan_table_position = 0;
        cursor->from_last_reset = 0;
        if (cursor->sequence_index >= coll->num_subsequences)
            return PULSEG_CURSOR_DONE;
        return PULSEG_CURSOR_BLOCK;
    }

    /* Normal advance */
    cursor->scan_table_position = next_pos;
    cursor->from_last_reset += 1;
    return PULSEG_CURSOR_BLOCK;
}

void pulseg_cursor_rewind(pulseg_collection *coll)
{
    pulseg_block_cursor *cursor;

    cursor = &coll->block_cursor;

    /* Go back by the number of blocks advanced since the last mark */
    cursor->scan_table_position -= cursor->from_last_reset;
    cursor->from_last_reset = 0;
}

void pulseg_cursor_reset(pulseg_collection *coll)
{
    if (!coll)
        return;

    /* Full rewind to the absolute start of the collection (pristine load
     * state {0, -1, 0}).  Unlike pulseg_cursor_rewind(), this also clears
     * sequence_index, so a collection whose cursor has already run to its
     * terminal PULSEG_CURSOR_DONE state (sequence_index == num_subsequences)
     * can be replayed from the top.  Required when one loaded collection is
     * traversed by more than one RSP entry point (e.g. aps2 then scan, which
     * reuse s_sc_coll because the SIM framework does not call psdcleanup
     * between entry points). */
    coll->block_cursor.sequence_index = 0;
    coll->block_cursor.scan_table_position = -1; /* -1 = before first block */
    coll->block_cursor.from_last_reset = 0;
}

void pulseg_cursor_mark(pulseg_collection *coll)
{
    if (!coll)
        return;
    coll->block_cursor.from_last_reset = 0;
}

int pulseg_cursor_get_info(
    const pulseg_collection *coll,
    pulseg_cursor_info *info)
{
    const pulseg_block_cursor *cursor;
    const pulseg_sequence_descriptor *desc;
    int pos, seg_id;

    if (!coll || !info)
        return PULSEG_ERR_NULL_POINTER;

    cursor = &coll->block_cursor;
    if (cursor->sequence_index < 0 ||
        cursor->sequence_index >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[cursor->sequence_index];
    pos = cursor->scan_table_position;
    if (pos < 0 || pos >= desc->scan_table_len)
        return PULSEG_ERR_INVALID_ARGUMENT;

    seg_id = desc->scan_table_seg_id[pos];

    info->subseq_idx = cursor->sequence_index;
    info->scan_pos = pos;
    info->segment_id = seg_id + coll->subsequence_info[cursor->sequence_index].segment_id_offset;
    /* segment_start fires at the first block of every segment instance,
     * including consecutive instances of the same segment (e.g. the ny
     * phase-encoding readout lines in MPRAGE within one TR).
     * A new instance begins when the seg_id changes, or when enough
     * blocks have elapsed to complete one full instance of the segment
     * (detected by counting back to the start of the same-seg_id run). */
    {
        int new_inst;
        if (pos == 0 || desc->scan_table_seg_id[pos] != desc->scan_table_seg_id[pos - 1])
        {
            new_inst = 1;
        }
        else if (seg_id >= 0 && seg_id < desc->num_unique_segments)
        {
            int nb = desc->segment_definitions[seg_id].num_blocks;
            int run_start = pos;
            if (nb <= 0)
                nb = 1;
            while (run_start > 0 && desc->scan_table_seg_id[run_start - 1] == seg_id)
                run_start--;
            new_inst = (((pos - run_start) % nb) == 0) ? 1 : 0;
        }
        else
        {
            new_inst = 0;
        }
        info->segment_start = new_inst;
    }
    {
        int last_inst;
        if (pos == desc->scan_table_len - 1 ||
            desc->scan_table_seg_id[pos] != desc->scan_table_seg_id[pos + 1])
        {
            last_inst = 1;
        }
        else if (seg_id >= 0 && seg_id < desc->num_unique_segments)
        {
            int nb = desc->segment_definitions[seg_id].num_blocks;
            int run_start = pos;
            if (nb <= 0)
                nb = 1;
            while (run_start > 0 && desc->scan_table_seg_id[run_start - 1] == seg_id)
                run_start--;
            last_inst = ((((pos - run_start) + 1) % nb) == 0) ? 1 : 0;
        }
        else
        {
            last_inst = 0;
        }
        info->segment_end = last_inst;
    }
    info->tr_start = desc->scan_table_tr_start
                         ? desc->scan_table_tr_start[pos]
                         : 0;
    info->pmc = desc->enable_pmc;

    /* Segment properties via local segment index */
    {
        int local_seg = seg_id; /* seg_id in scan table is local (before offset) */
        if (local_seg >= 0 && local_seg < desc->num_unique_segments)
        {
            info->is_nav = desc->segment_definitions[local_seg].is_nav;
            info->has_trigger = (desc->segment_definitions[local_seg].trigger_id >= 0) ? 1 : 0;
        }
        else
        {
            info->is_nav = 0;
            info->has_trigger = 0;
        }
    }

    return PULSEG_SUCCESS;
}

int pulseg_get_block_instance(
    const pulseg_collection *coll,
    pulseg_block_instance *inst)
{
    const pulseg_block_cursor *cursor;
    const pulseg_sequence_descriptor *desc;
    const pulseg_block_table_element *bte;
    const pulseg_block_definition *bdef;
    int idx, i;

    if (!coll || !inst)
        return PULSEG_ERR_NULL_POINTER;

    cursor = &coll->block_cursor;
    if (cursor->sequence_index >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[cursor->sequence_index];
    if (cursor->scan_table_position < 0 ||
        cursor->scan_table_position >= desc->scan_table_len)
        return PULSEG_ERR_INVALID_ARGUMENT;

    idx = desc->scan_table_block_idx[cursor->scan_table_position];
    if (idx < 0 || idx >= desc->num_blocks)
        return PULSEG_ERR_INVALID_ARGUMENT;

    bte = &desc->block_table[idx];
    bdef = &desc->block_definitions[bte->id];

    /* Duration: pure delay uses instance value, normal block uses definition */
    inst->duration_us = (bte->duration_us >= 0)
                            ? bte->duration_us
                            : bdef->duration_us;

    /* RF */
    if (bte->rf_id >= 0 && bte->rf_id < desc->rf_table_size)
    {
        inst->rf_amp_hz = desc->rf_table[bte->rf_id].amplitude;
        inst->rf_freq_hz = desc->rf_table[bte->rf_id].freq_offset;
        inst->rf_phase_rad = desc->rf_table[bte->rf_id].phase_offset;
    }
    else
    {
        inst->rf_amp_hz = 0.0f;
        inst->rf_freq_hz = 0.0f;
        inst->rf_phase_rad = 0.0f;
    }

    /* Gradients */
    if (bte->gx_id >= 0 && bte->gx_id < desc->grad_table_size)
    {
        inst->gx_amp_hz_per_m = desc->grad_table[bte->gx_id].amplitude;
        inst->gx_shot_idx = desc->grad_table[bte->gx_id].shot_index;
    }
    else
    {
        inst->gx_amp_hz_per_m = 0.0f;
        inst->gx_shot_idx = 0;
    }
    if (bte->gy_id >= 0 && bte->gy_id < desc->grad_table_size)
    {
        inst->gy_amp_hz_per_m = desc->grad_table[bte->gy_id].amplitude;
        inst->gy_shot_idx = desc->grad_table[bte->gy_id].shot_index;
    }
    else
    {
        inst->gy_amp_hz_per_m = 0.0f;
        inst->gy_shot_idx = 0;
    }
    if (bte->gz_id >= 0 && bte->gz_id < desc->grad_table_size)
    {
        inst->gz_amp_hz_per_m = desc->grad_table[bte->gz_id].amplitude;
        inst->gz_shot_idx = desc->grad_table[bte->gz_id].shot_index;
    }
    else
    {
        inst->gz_amp_hz_per_m = 0.0f;
        inst->gz_shot_idx = 0;
    }

    /* Rotation */
    if (bte->rotation_id >= 0 && bte->rotation_id < desc->num_rotations)
    {
        for (i = 0; i < 9; ++i)
            inst->rotmat[i] = desc->rotation_matrices[bte->rotation_id][i];
    }
    else
    {
        inst->rotmat[0] = 1.0f;
        inst->rotmat[1] = 0.0f;
        inst->rotmat[2] = 0.0f;
        inst->rotmat[3] = 0.0f;
        inst->rotmat[4] = 1.0f;
        inst->rotmat[5] = 0.0f;
        inst->rotmat[6] = 0.0f;
        inst->rotmat[7] = 0.0f;
        inst->rotmat[8] = 1.0f;
    }

    /* Flags */
    inst->norot_flag = bte->norot_flag;
    inst->nopos_flag = bte->nopos_flag;

    /* Digital output */
    inst->digitalout_flag = (bte->digitalout_id >= 0) ? 1 : 0;
    if (inst->digitalout_flag && bte->digitalout_id < desc->num_triggers)
    {
        inst->digitalout_channel = desc->trigger_events[bte->digitalout_id].trigger_channel;
    }
    else
    {
        inst->digitalout_channel = -1;
    }

    /* ADC */
    if (bte->adc_id >= 0 && bte->adc_id < desc->adc_table_size)
    {
        inst->adc_flag = 1;
        inst->adc_freq_hz = desc->adc_table[bte->adc_id].freq_offset;
        inst->adc_phase_rad = desc->adc_table[bte->adc_id].phase_offset;
    }
    else
    {
        inst->adc_flag = 0;
        inst->adc_freq_hz = 0.0f;
        inst->adc_phase_rad = 0.0f;
    }

    /* RF shimming */
    inst->rf_shim_id = bte->rf_shim_id;

    /* Variable-amplitude flags (TR-level) */
    {
        int vg_tr_size = desc->tr_descriptor.tr_size;
        int vg_pos;
        if (vg_tr_size > 0 && desc->variable_grad_flags)
        {
            vg_pos = cursor->scan_table_position % vg_tr_size;
            inst->gx_variable = desc->variable_grad_flags[vg_pos * 3 + 0];
            inst->gy_variable = desc->variable_grad_flags[vg_pos * 3 + 1];
            inst->gz_variable = desc->variable_grad_flags[vg_pos * 3 + 2];
        }
        else
        {
            inst->gx_variable = 0;
            inst->gy_variable = 0;
            inst->gz_variable = 0;
        }
    }

    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Frequency modulation plan                                         */
/* ================================================================== */

/* --- helper: block range for a TR region --- */
static int get_block_range(
    const pulseg_sequence_descriptor *desc,
    int tr_type, int tr_index,
    int *out_start, int *out_count)
{
    const pulseg_tr_descriptor *tr = &desc->tr_descriptor;

    switch (tr_type)
    {
    case PULSEG_TR_REGION_PREP:
        *out_start = 0;
        *out_count = desc->num_prep_blocks;
        return 1;
    case PULSEG_TR_REGION_MAIN:
        if (tr->tr_size <= 0 || tr_index < 0 || tr_index >= tr->num_trs)
            return 0;
        *out_start = desc->num_prep_blocks + tr_index * tr->tr_size;
        *out_count = tr->tr_size;
        return 1;
    case PULSEG_TR_REGION_COOLDOWN:
        *out_start = desc->pass_len - desc->num_cooldown_blocks;
        *out_count = desc->num_cooldown_blocks;
        return 1;
    case PULSEG_TR_REGION_ALL:
        *out_start = 0;
        *out_count = desc->pass_len;
        return 1;
    default:
        return 0;
    }
}

/* --- helper: count RF+ADC events in a block range --- */
static int count_fm_events_range(
    const pulseg_sequence_descriptor *desc,
    int blk_start, int blk_count)
{
    int n, count;
    count = 0;
    for (n = blk_start; n < blk_start + blk_count && n < desc->num_blocks; ++n)
    {
        if (desc->block_table[n].freq_mod_id >= 0)
            ++count;
    }
    return count;
}

int pulseg_get_freq_mod_count(
    const pulseg_collection *coll)
{
    int i, total;
    if (!coll)
        return 0;
    total = 0;
    for (i = 0; i < coll->num_subsequences; ++i)
        total += count_fm_events_range(&coll->descriptors[i],
                                       0, coll->descriptors[i].num_blocks);
    return total;
}

int pulseg_get_freq_mod_count_tr(
    const pulseg_collection *coll,
    int tr_type, int tr_index)
{
    int blk_start, blk_count;
    const pulseg_sequence_descriptor *desc;

    if (!coll || coll->num_subsequences < 1)
        return 0;
    desc = &coll->descriptors[0];
    if (!get_block_range(desc, tr_type, tr_index, &blk_start, &blk_count))
        return 0;
    return count_fm_events_range(desc, blk_start, blk_count);
}

/* ================================================================== */
/*  Label getters                                                     */
/* ================================================================== */

int pulseg_get_label_limits(const pulseg_collection *coll,
                               pulseg_label_limits *limits,
                               int subseq_idx)
{
    if (!coll || !limits)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;
    *limits = coll->descriptors[subseq_idx].label_limits;
    return PULSEG_SUCCESS;
}

static int pulseg__get_num_adc_occurrences(const pulseg_collection *coll,
                                              int subseq_idx)
{
    if (!coll)
        return 0;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].label_num_entries;
}

static int pulseg__get_num_label_columns(const pulseg_collection *coll,
                                            int subseq_idx)
{
    if (!coll)
        return 0;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].label_num_columns;
}

int pulseg_get_adc_label(const pulseg_collection *coll,
                            int *out_values,
                            int subseq_idx,
                            int occurrence_idx)
{
    const pulseg_sequence_descriptor *desc;
    int ncols, row_start, c;

    if (!coll || !out_values)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[subseq_idx];
    if (!desc->label_table || desc->label_num_entries == 0)
        return PULSEG_ERR_INVALID_ARGUMENT;
    if (occurrence_idx < 0 || occurrence_idx >= desc->label_num_entries)
        return PULSEG_ERR_INVALID_ARGUMENT;

    ncols = desc->label_num_columns;
    row_start = occurrence_idx * ncols;
    for (c = 0; c < ncols; ++c)
        out_values[c] = desc->label_table[row_start + c];

    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Definition getters                                                */
/* ================================================================== */

/* Static scratch space for the read-only view returned to callers.
 * Thread-safety note: not thread-safe; matches existing patterns. */
static pulseg_definition_entry *s_def_view = NULL;
static int s_def_view_cap = 0;

int pulseg_get_definitions(const pulseg_collection *coll,
                              const pulseg_definition_entry **out,
                              int *num_entries,
                              int subseq_idx)
{
    const pulseg_sequence_descriptor *desc;
    int i;

    if (!coll || !out || !num_entries)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[subseq_idx];
    *num_entries = desc->num_definitions;

    if (desc->num_definitions == 0)
    {
        *out = NULL;
        return PULSEG_SUCCESS;
    }

    /* Re-allocate scratch if needed */
    if (desc->num_definitions > s_def_view_cap)
    {
        if (s_def_view)
            PULSEG_FREE(s_def_view);
        s_def_view = (pulseg_definition_entry *)PULSEG_ALLOC(
            (size_t)desc->num_definitions * sizeof(pulseg_definition_entry));
        if (!s_def_view)
        {
            s_def_view_cap = 0;
            return PULSEG_ERR_ALLOC_FAILED;
        }
        s_def_view_cap = desc->num_definitions;
    }

    for (i = 0; i < desc->num_definitions; ++i)
    {
        s_def_view[i].name = desc->definitions[i].name;
        s_def_view[i].num_values = desc->definitions[i].value_size;
        s_def_view[i].values = (const char *const *)desc->definitions[i].value;
    }

    *out = s_def_view;
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Batch getters (public API)                                        */
/* ================================================================== */

int pulseg_get_collection_info(
    const pulseg_collection *coll,
    pulseg_collection_info *info)
{
    if (!coll || !info)
        return PULSEG_ERR_NULL_POINTER;

    info->num_subsequences = pulseg__get_num_subsequences(coll);
    info->num_segments = pulseg__get_num_segments(coll);
    info->max_adc_samples = pulseg__get_max_adc_samples(coll);
    info->total_readouts = pulseg__get_total_readouts(coll);
    info->total_duration_us = pulseg__get_total_duration_us(coll);

    return PULSEG_SUCCESS;
}

int pulseg_get_subseq_info(const pulseg_collection *coll,
                              pulseg_subseq_info *info,
                              int subseq_idx)
{
    if (!coll || !info)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;

    info->tr_duration_us = pulseg__get_tr_duration_us(coll, subseq_idx);
    info->num_trs = pulseg__get_num_trs(coll, subseq_idx);
    info->tr_size = pulseg__get_tr_size(coll, subseq_idx);
    info->num_prep_blocks = pulseg__get_num_prep_blocks(coll, subseq_idx);
    info->num_cooldown_blocks = pulseg__get_num_cooldown_blocks(coll, subseq_idx);
    info->num_prep_trs = pulseg__get_num_prep_trs(coll, subseq_idx);
    info->num_cooldown_trs = pulseg__get_num_cooldown_trs(coll, subseq_idx);
    info->degenerate_prep = pulseg__get_degenerate_prep(coll, subseq_idx);
    info->degenerate_cooldown = pulseg__get_degenerate_cooldown(coll, subseq_idx);
    info->num_unique_adcs = pulseg__get_num_unique_adcs(coll, subseq_idx);
    info->num_unique_rf = pulseg__get_num_unique_rf(coll, subseq_idx);
    info->pmc_enabled = pulseg__is_pmc_enabled(coll, subseq_idx);
    info->segment_offset = pulseg__get_subseq_segment_offset(coll, subseq_idx);
    info->num_prep_segments = pulseg__get_num_prep_segments(coll, subseq_idx);
    info->num_main_segments = pulseg__get_num_main_segments(coll, subseq_idx);
    info->num_cooldown_segments = pulseg__get_num_cooldown_segments(coll, subseq_idx);
    info->num_adc_occurrences = pulseg__get_num_adc_occurrences(coll, subseq_idx);
    info->num_label_columns = pulseg__get_num_label_columns(coll, subseq_idx);
    info->num_passes = coll->descriptors[subseq_idx].num_passes;
    info->num_averages = coll->descriptors[subseq_idx].num_averages;
    info->num_gain_cal_readouts = coll->descriptors[subseq_idx].num_gain_cal_readouts;

    /* Compute num_canonical_trs (unique shot-ID combinations). */
    {
        const pulseg_sequence_descriptor *d = &coll->descriptors[subseq_idx];
        const pulseg_tr_descriptor *trd = &d->tr_descriptor;
        int has_nd_p = (trd->num_prep_blocks > 0 && !trd->degenerate_prep);
        int has_nd_c = (trd->num_cooldown_blocks > 0 && !trd->degenerate_cooldown);
        int *can_idx = NULL;
        int *can_lbl = NULL;
        int nc;
        if (has_nd_p || has_nd_c)
        {
            nc = pulseg__find_unique_shot_passes(d, &can_idx, &can_lbl);
        }
        else
        {
            nc = pulseg__find_unique_shot_trs(d, &can_idx, &can_lbl);
        }
        info->num_canonical_trs = (nc > 0) ? nc : 1;
        if (can_idx)
            PULSEG_FREE(can_idx);
        if (can_lbl)
            PULSEG_FREE(can_lbl);
    }

    return PULSEG_SUCCESS;
}

int pulseg_get_segment_info(const pulseg_collection *coll,
                               pulseg_segment_info *info,
                               int seg_idx)
{
    if (!coll || !info)
        return PULSEG_ERR_NULL_POINTER;

    info->duration_us = pulseg__get_segment_duration_us(coll, seg_idx);
    info->num_blocks = pulseg__get_segment_num_blocks(coll, seg_idx);
    info->start_block = pulseg__get_segment_start_block(coll, seg_idx);
    info->pure_delay = pulseg__is_segment_pure_delay(coll, seg_idx);
    info->has_trigger = pulseg__segment_has_trigger(coll, seg_idx);
    info->trigger_type = pulseg__get_segment_trigger_type(coll, seg_idx);
    info->trigger_delay_us = pulseg__get_segment_trigger_delay_us(coll, seg_idx);
    info->trigger_duration_us = pulseg__get_segment_trigger_duration_us(coll, seg_idx);
    info->is_nav = pulseg__segment_is_nav(coll, seg_idx);
    info->num_kzero_crossings = pulseg__get_segment_num_kzero_crossings(coll, seg_idx);
    info->rf_adc_gap_us = pulseg__get_segment_rf_adc_gap_us(coll, seg_idx);
    info->adc_adc_gap_us = pulseg__get_segment_adc_adc_gap_us(coll, seg_idx);

    return PULSEG_SUCCESS;
}

int pulseg_segment_has_grad(const pulseg_collection *coll,
                               int seg_idx)
{
    pulseg_segment_info si = PULSEG_SEGMENT_INFO_INIT;
    pulseg_block_info bi = PULSEG_BLOCK_INFO_INIT;
    int blk;

    if (!coll)
        return 0;
    if (!PULSEG_SUCCEEDED(pulseg_get_segment_info(coll, &si, seg_idx)))
        return 0;

    for (blk = 0; blk < si.num_blocks; ++blk)
    {
        if (!PULSEG_SUCCEEDED(pulseg_get_block_info(coll, &bi, seg_idx, blk)))
            continue;

        if (bi.has_grad[0] || bi.has_grad[1] || bi.has_grad[2])
            return 1;
    }

    return 0;
}

int pulseg_get_block_info(const pulseg_collection *coll,
                             pulseg_block_info *info,
                             int seg_idx,
                             int blk_idx)
{
    int axis;

    if (!coll || !info)
        return PULSEG_ERR_NULL_POINTER;

    info->duration_us = pulseg__get_block_duration_us(coll, seg_idx, blk_idx);
    info->start_time_us = pulseg__get_block_start_time_us(coll, seg_idx, blk_idx);

    /* Gradient (per axis) */
    for (axis = 0; axis < 3; ++axis)
    {
        info->has_grad[axis] = pulseg__block_has_grad(coll, seg_idx, blk_idx, axis);
        info->grad_is_trapezoid[axis] = info->has_grad[axis] ? pulseg__block_grad_is_trapezoid(coll, seg_idx, blk_idx, axis) : 0;
        info->grad_delay_us[axis] = info->has_grad[axis] ? pulseg__get_grad_delay_us(coll, seg_idx, blk_idx, axis) : -1;
        info->grad_num_shots[axis] = info->has_grad[axis] ? pulseg__get_grad_num_shots(coll, seg_idx, blk_idx, axis) : -1;
        info->grad_num_samples[axis] = info->has_grad[axis] ? pulseg__get_grad_num_samples(coll, seg_idx, blk_idx, axis) : -1;
    }

    /* RF */
    info->has_rf = pulseg__block_has_rf(coll, seg_idx, blk_idx);
    info->rf_delay_us = info->has_rf ? pulseg__get_rf_delay_us(coll, seg_idx, blk_idx) : -1;
    info->rf_num_channels = info->has_rf ? pulseg__get_rf_num_channels(coll, seg_idx, blk_idx) : -1;
    info->rf_num_samples = info->has_rf ? pulseg__get_rf_num_samples(coll, seg_idx, blk_idx) : -1;
    info->rf_duration_us = info->has_rf ? pulseg__get_rf_duration_us(coll, seg_idx, blk_idx) : -1;
    info->rf_is_complex = info->has_rf ? pulseg__block_rf_is_complex(coll, seg_idx, blk_idx) : 0;
    info->rf_uniform_raster = info->has_rf ? pulseg__block_rf_has_uniform_raster(coll, seg_idx, blk_idx) : 0;

    /* ADC */
    info->has_adc = pulseg__block_has_adc(coll, seg_idx, blk_idx);
    info->adc_delay_us = info->has_adc ? pulseg__get_adc_delay_us(coll, seg_idx, blk_idx) : -1;
    info->adc_def_id = info->has_adc ? pulseg__get_adc_library_index(coll, seg_idx, blk_idx) : -1;

    /* Digital output */
    info->has_digitalout = pulseg__block_has_digitalout(coll, seg_idx, blk_idx);
    info->digitalout_delay_us = info->has_digitalout ? pulseg__get_digitalout_delay_us(coll, seg_idx, blk_idx) : -1;
    info->digitalout_duration_us = info->has_digitalout ? pulseg__get_digitalout_duration_us(coll, seg_idx, blk_idx) : -1;
    info->digitalout_channel = info->has_digitalout ? pulseg__get_digitalout_channel(coll, seg_idx, blk_idx) : -1;

    /* Flags */
    info->has_rotation = pulseg__block_has_rotation(coll, seg_idx, blk_idx);
    info->norot_flag = pulseg__block_has_norot(coll, seg_idx, blk_idx);
    info->nopos_flag = pulseg__block_has_nopos(coll, seg_idx, blk_idx);
    info->has_freq_mod = pulseg__block_has_freq_mod(coll, seg_idx, blk_idx);

    return PULSEG_SUCCESS;
}

int pulseg_get_adc_def(const pulseg_collection *coll,
                          pulseg_adc_def *def,
                          int adc_idx)
{
    if (!coll || !def)
        return PULSEG_ERR_NULL_POINTER;

    def->dwell_ns = pulseg__get_adc_dwell_ns(coll, adc_idx);
    def->num_samples = pulseg__get_adc_num_samples(coll, adc_idx);

    return PULSEG_SUCCESS;
}

int pulseg_get_rf_shim_def(const pulseg_collection *coll,
                              pulseg_rf_shim_def *def,
                              int subseq_idx,
                              int shim_idx)
{
    const pulseg_sequence_descriptor *desc;
    int i;
    const pulseg_rf_shim_definition *src;

    if (!coll || !def)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INDEX;

    desc = &coll->descriptors[subseq_idx];

    /* shim_idx is LOCAL to its subsequence (same convention as rf_id, gx_id, etc.).
     * Each subseq stores its own rf_shim_definitions[] starting at index 0. */
    if (shim_idx < 0 || shim_idx >= desc->num_rf_shims)
        return PULSEG_ERR_INDEX;

    src = &desc->rf_shim_definitions[shim_idx];
    def->num_channels = src->num_channels;
    for (i = 0; i < src->num_channels && i < PULSEG_MAX_RF_SHIM_CHANNELS; i++)
    {
        def->magnitudes[i] = src->magnitudes[i];
        def->phases[i] = src->phases[i];
    }
    return PULSEG_SUCCESS;
}

int pulseg_get_num_rf_shims(const pulseg_collection *coll,
                               int subseq_idx)
{
    if (!coll)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INDEX;
    return coll->descriptors[subseq_idx].num_rf_shims;
}

/* ================================================================== */
/*  Unique-block and segment-block getters                            */
/* ================================================================== */

int pulseg_get_num_unique_blocks(
    const pulseg_collection *coll,
    int subseq_idx)
{
    if (!coll)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INDEX;
    return coll->descriptors[subseq_idx].num_unique_blocks;
}

int pulseg_get_unique_block_id(
    const pulseg_collection *coll,
    int subseq_idx,
    int blk_def_idx)
{
    const pulseg_sequence_descriptor *desc;
    if (!coll)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INDEX;
    desc = &coll->descriptors[subseq_idx];
    if (blk_def_idx < 0 || blk_def_idx >= desc->num_unique_blocks)
        return PULSEG_ERR_INDEX;
    /* block_definitions[].id is the 1-based .seq block index */
    return desc->block_definitions[blk_def_idx].id;
}

int pulseg_get_segment_block_def_indices(
    const pulseg_collection *coll,
    int *out_ids,
    int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_segment *seg;
    int local_seg, i;

    if (!coll || !out_ids)
        return PULSEG_ERR_NULL_POINTER;
    if (!pulseg__resolve_segment(coll, &desc, &local_seg, seg_idx))
        return PULSEG_ERR_INDEX;

    seg = &desc->segment_definitions[local_seg];
    for (i = 0; i < seg->num_blocks; ++i)
        out_ids[i] = seg->unique_block_indices[i];
    return seg->num_blocks;
}
