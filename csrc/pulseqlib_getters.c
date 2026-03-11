/* pulseqlib_accessors.c -- collection-level accessor functions
 *
 * Public functions:
 *   pulseqlib_get_max_adc_samples    pulseqlib_get_adc_dwell_ns
 *   pulseqlib_get_adc_num_samples    pulseqlib_get_num_segments
 *   pulseqlib_is_segment_pure_delay  pulseqlib_get_segment_num_blocks
 *   pulseqlib_get_block_start_time_us   pulseqlib_get_block_duration_us
 *   pulseqlib_block_has_rf           pulseqlib_block_rf_has_uniform_raster
 *   pulseqlib_block_rf_is_complex    pulseqlib_get_rf_num_samples
 *   pulseqlib_get_rf_num_channels    pulseqlib_get_rf_delay_us
 *   pulseqlib_get_rf_magnitude       pulseqlib_get_rf_phase
 *   pulseqlib_get_rf_time_us
 *   pulseqlib_block_has_grad         pulseqlib_block_grad_is_trapezoid
 *   pulseqlib_get_grad_num_samples   pulseqlib_get_grad_num_shots
 *   pulseqlib_get_grad_delay_us         pulseqlib_get_grad_amplitude
 *   pulseqlib_get_grad_initial_amplitude_hz_per_m
 *   pulseqlib_get_grad_initial_shot_id
 *   pulseqlib_get_grad_time_us
 *   pulseqlib_block_has_adc          pulseqlib_get_adc_delay_us
 *   pulseqlib_get_adc_library_index
 *   pulseqlib_block_has_digitalout   pulseqlib_get_digitalout_delay_us
 *   pulseqlib_segment_has_trigger    pulseqlib_get_segment_trigger_delay_us
 *   pulseqlib_segment_is_nav
 *   pulseqlib_block_has_rotation
 *   pulseqlib_block_has_norot        pulseqlib_block_has_nopos
 *
 * Internal (pulseqlib__) functions:
 *   pulseqlib__resolve_segment       pulseqlib__resolve_block
 */

#include <string.h>
#include <stdlib.h>

#include "pulseqlib_internal.h"
#include "pulseqlib_methods.h"

/* ================================================================== */
/*  Resolve helpers                                                   */
/* ================================================================== */

int pulseqlib__resolve_segment(
    const pulseqlib_sequence_descriptor** out_desc,
    int* out_local_seg,
    const pulseqlib_collection* coll,
    int seg_idx)
{
    int i, num_segs, global_idx;

    if (!coll || seg_idx < 0 || seg_idx >= coll->total_unique_segments)
        return 0;

    global_idx = 0;
    for (i = 0; i < coll->num_subsequences; ++i) {
        num_segs = coll->descriptors[i].num_unique_segments;
        if (seg_idx < global_idx + num_segs) {
            if (out_desc)      *out_desc      = &coll->descriptors[i];
            if (out_local_seg) *out_local_seg  = seg_idx - global_idx;
            return 1;
        }
        global_idx += num_segs;
    }
    return 0;
}

int pulseqlib__resolve_block(
    const pulseqlib_sequence_descriptor** out_desc,
    const pulseqlib_tr_segment** out_seg,
    int* out_local_blk,
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    int local_seg;
    const pulseqlib_tr_segment* seg;

    desc = NULL;
    if (!pulseqlib__resolve_segment(&desc, &local_seg, coll, seg_idx))
        return 0;

    seg = &desc->segment_definitions[local_seg];
    if (blk_idx < 0 || blk_idx >= seg->num_blocks)
        return 0;

    if (out_desc)      *out_desc      = desc;
    if (out_seg)       *out_seg       = seg;
    if (out_local_blk) *out_local_blk = blk_idx;
    return 1;
}

/* ================================================================== */
/*  Axis helper                                                       */
/* ================================================================== */

static int get_grad_id_by_axis(const pulseqlib_block_definition* bdef, int axis)
{
    switch (axis) {
        case PULSEQLIB_GRAD_AXIS_X: return bdef->gx_id;
        case PULSEQLIB_GRAD_AXIS_Y: return bdef->gy_id;
        case PULSEQLIB_GRAD_AXIS_Z: return bdef->gz_id;
        default: return -1;
    }
}

static int get_grad_event_id_by_axis(const pulseqlib_block_table_element* bte, int axis)
{
    switch (axis) {
        case PULSEQLIB_GRAD_AXIS_X: return bte->gx_id;
        case PULSEQLIB_GRAD_AXIS_Y: return bte->gy_id;
        case PULSEQLIB_GRAD_AXIS_Z: return bte->gz_id;
        default: return -1;
    }
}

/* ================================================================== */
/*  Subsequence accessors (internal helpers for batch getters)         */
/* ================================================================== */

static int pulseqlib__get_num_subsequences(
    const pulseqlib_collection* coll)
{
    if (!coll) return 0;
    return coll->num_subsequences;
}

static float pulseqlib__get_tr_duration_us(
    const pulseqlib_collection* coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0.0f;
    return coll->descriptors[subseq_idx].tr_descriptor.tr_duration_us;
}

static int pulseqlib__get_num_trs(
    const pulseqlib_collection* coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].tr_descriptor.num_trs;
}

static int pulseqlib__get_tr_size(
    const pulseqlib_collection* coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].tr_descriptor.tr_size;
}

static int pulseqlib__get_num_prep_blocks(
    const pulseqlib_collection* coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].tr_descriptor.num_prep_blocks;
}

static int pulseqlib__get_num_cooldown_blocks(
    const pulseqlib_collection* coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].tr_descriptor.num_cooldown_blocks;
}

static int pulseqlib__get_degenerate_prep(
    const pulseqlib_collection* coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].tr_descriptor.degenerate_prep;
}

static int pulseqlib__get_degenerate_cooldown(
    const pulseqlib_collection* coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].tr_descriptor.degenerate_cooldown;
}

static int pulseqlib__get_num_prep_trs(
    const pulseqlib_collection* coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].tr_descriptor.num_prep_trs;
}

static int pulseqlib__get_num_cooldown_trs(
    const pulseqlib_collection* coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].tr_descriptor.num_cooldown_trs;
}

static int pulseqlib__get_num_unique_adcs(
    const pulseqlib_collection* coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].num_unique_adcs;
}

static int pulseqlib__is_pmc_enabled(
    const pulseqlib_collection* coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].enable_pmc;
}

static int pulseqlib__get_subseq_segment_offset(
    const pulseqlib_collection* coll,
    int subseq_idx)
{
    int i, offset = 0;
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    for (i = 0; i < subseq_idx; ++i)
        offset += coll->descriptors[i].num_unique_segments;
    return offset;
}

static float pulseqlib__get_total_duration_us(
    const pulseqlib_collection* coll)
{
    if (!coll) return 0.0f;
    return coll->total_duration_us;
}

int pulseqlib_get_scan_time(
    const pulseqlib_collection* coll,
    int                        num_reps,
    pulseqlib_scan_time_info*  info)
{
    int i, n, bt_idx;
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_block_table_element* bte;
    const pulseqlib_block_definition* bdef;
    int prev_seg, cur_seg;

    if (!coll || !info) return PULSEQLIB_ERR_NULL_POINTER;
    if (num_reps < 1) return PULSEQLIB_ERR_INVALID_ARGUMENT;
    if (coll->num_subsequences <= 0) return PULSEQLIB_ERR_COLLECTION_EMPTY;

    (void)num_reps;  /* averages already baked into scan table */

    info->total_duration_us        = 0.0f;
    info->total_segment_boundaries = 0;

    for (i = 0; i < coll->num_subsequences; ++i) {
        desc = &coll->descriptors[i];
        prev_seg = -1;

        for (n = 0; n < desc->scan_table_len; ++n) {
            bt_idx = desc->scan_table_block_idx[n];
            bte  = &desc->block_table[bt_idx];
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

    return PULSEQLIB_SUCCESS;
}

/* ================================================================== */
/*  RF accessors                                                      */
/* ================================================================== */

static int pulseqlib__get_num_unique_rf(
    const pulseqlib_collection* coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].num_unique_rfs;
}

int pulseqlib_get_rf_stats(
    const pulseqlib_collection* coll,
    pulseqlib_rf_stats* stats,
    int subseq_idx, int rf_idx)
{
    const pulseqlib_sequence_descriptor* desc;

    if (!coll || !stats) return PULSEQLIB_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;
    desc = &coll->descriptors[subseq_idx];
    if (rf_idx < 0 || rf_idx >= desc->num_unique_rfs)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;

    *stats = desc->rf_definitions[rf_idx].stats;
    return PULSEQLIB_SUCCESS;
}

/*
 * pulseqlib_get_tr_rf_ids --
 *   Return an array of RF definition IDs for each block position
 *   within the first main TR.  Blocks without RF get -1.
 *
 *   out_rf_ids must point to a pre-allocated array of tr_size ints.
 *   Returns tr_size on success, negative error code on failure.
 */
int pulseqlib_get_tr_rf_ids(
    const pulseqlib_collection* coll,
    int* out_rf_ids,
    int subseq_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_descriptor* trd;
    const pulseqlib_block_table_element* bte;
    int i, block_idx, tr_size;

    if (!coll || !out_rf_ids) return PULSEQLIB_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[subseq_idx];
    trd  = &desc->tr_descriptor;
    tr_size = trd->tr_size;

    for (i = 0; i < tr_size; ++i) {
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
/*  pulseqlib_get_rf_array --                                         */
/*    Build an ordered array of RF stats for a TR region.             */
/*    Each entry gets the base rf_stats patched with the actual       */
/*    amplitude from the rf_table and the repetition count for        */
/*    that region.  The library allocates; caller must free().        */
/* ================================================================== */
int pulseqlib_get_rf_array(
    const pulseqlib_collection*  coll,
    pulseqlib_rf_stats**         out_pulses,
    int                          subseq_idx,
    int                          region)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_descriptor* trd;
    const pulseqlib_block_table_element* bte;
    const pulseqlib_rf_definition* rfdef;
    int start, count, num_instances;
    int i, n, num_rf;

    if (!coll || !out_pulses)
        return PULSEQLIB_ERR_NULL_POINTER;
    *out_pulses = NULL;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[subseq_idx];
    trd  = &desc->tr_descriptor;

    /* Determine block range and instance count for the region */
    switch (region) {
    case PULSEQLIB_TR_REGION_PREP:
        start = 0;
        count = trd->num_prep_blocks + trd->tr_size;
        num_instances = 1;
        break;

    case PULSEQLIB_TR_REGION_MAIN:
        start = trd->num_prep_blocks;
        count = trd->tr_size;
        num_instances = trd->num_trs;
        if (!trd->degenerate_prep)    num_instances--;
        if (!trd->degenerate_cooldown) num_instances--;
        if (num_instances < 0) num_instances = 0;
        break;

    case PULSEQLIB_TR_REGION_COOLDOWN:
        start = trd->num_prep_blocks +
                (trd->num_trs > 0 ? (trd->num_trs - 1) * trd->tr_size : 0);
        count = trd->tr_size + trd->num_cooldown_blocks;
        num_instances = 1;
        break;

    default:
        return PULSEQLIB_ERR_INVALID_ARGUMENT;
    }

    /* Clamp to available block range */
    if (start + count > desc->num_blocks)
        count = desc->num_blocks - start;
    if (count < 0) count = 0;

    /* Pass 1: count RF-bearing blocks */
    num_rf = 0;
    for (i = 0; i < count; ++i) {
        bte = &desc->block_table[start + i];
        if (bte->rf_id >= 0 && bte->rf_id < desc->rf_table_size) {
            int id = desc->rf_table[bte->rf_id].id;
            if (id >= 0 && id < desc->num_unique_rfs)
                num_rf++;
        }
    }

    if (num_rf == 0)
        return 0;

    /* Allocate output array */
    *out_pulses = (pulseqlib_rf_stats*)malloc(
        (size_t)num_rf * sizeof(pulseqlib_rf_stats));
    if (!*out_pulses)
        return PULSEQLIB_ERR_ALLOC_FAILED;

    /* Pass 2: fill entries */
    n = 0;
    for (i = 0; i < count; ++i) {
        int blk_idx = start + i;
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

        /* Patch actual amplitude from rf_table */
        act_amp = desc->rf_table[bte->rf_id].amplitude;
        (*out_pulses)[n].act_amplitude_hz = (act_amp >= 0.0f)
            ? act_amp : -act_amp;

        /* Set repetition count */
        (*out_pulses)[n].num_instances = num_instances;

        n++;
    }

    return n;
}

/* ================================================================== */
/*  ADC collection accessors                                          */
/* ================================================================== */

static int pulseqlib__get_total_readouts(
    const pulseqlib_collection* coll)
{
    int i, b, total;

    if (!coll) return 0;

    total = 0;
    for (i = 0; i < coll->num_subsequences; ++i) {
        const pulseqlib_sequence_descriptor* desc = &coll->descriptors[i];
        const pulseqlib_tr_descriptor* trd = &desc->tr_descriptor;
        int adc_per_tr = 0;
        int adc_prep   = 0;
        int adc_cool   = 0;
        int main_trs;

        /* Count ADC blocks in one main TR */
        for (b = 0; b < trd->tr_size; ++b) {
            int blk_idx = trd->num_prep_blocks + b;
            if (blk_idx < desc->num_blocks &&
                desc->block_table[blk_idx].adc_id >= 0)
                adc_per_tr++;
        }

        /* Count ADC blocks in prep region */
        for (b = 0; b < trd->num_prep_blocks; ++b) {
            if (b < desc->num_blocks &&
                desc->block_table[b].adc_id >= 0)
                adc_prep++;
        }

        /* Count ADC blocks in cooldown region */
        {
            int cool_start = trd->num_prep_blocks +
                             trd->num_trs * trd->tr_size;
            for (b = 0; b < trd->num_cooldown_blocks; ++b) {
                int idx = cool_start + b;
                if (idx < desc->num_blocks &&
                    desc->block_table[idx].adc_id >= 0)
                    adc_cool++;
            }
        }

        main_trs = trd->num_trs;
        total += adc_per_tr * main_trs + adc_prep + adc_cool;
    }
    return total;
}

static int pulseqlib__get_max_adc_samples(
    const pulseqlib_collection* coll)
{
    int i, j, max_samples;

    if (!coll) return 0;

    max_samples = 0;
    for (i = 0; i < coll->num_subsequences; ++i) {
        for (j = 0; j < coll->descriptors[i].num_unique_adcs; ++j) {
            if (coll->descriptors[i].adc_definitions[j].num_samples > max_samples)
                max_samples = coll->descriptors[i].adc_definitions[j].num_samples;
        }
    }
    return max_samples;
}

static int pulseqlib__get_adc_dwell_ns(
    const pulseqlib_collection* coll, int adc_idx)
{
    int i, global_idx, num_adcs, local;

    if (!coll || adc_idx < 0 || adc_idx >= coll->total_unique_adcs)
        return 0;

    global_idx = 0;
    for (i = 0; i < coll->num_subsequences; ++i) {
        num_adcs = coll->descriptors[i].num_unique_adcs;
        if (adc_idx < global_idx + num_adcs) {
            local = adc_idx - global_idx;
            return coll->descriptors[i].adc_definitions[local].dwell_time;
        }
        global_idx += num_adcs;
    }
    return 0;
}

static int pulseqlib__get_adc_num_samples(
    const pulseqlib_collection* coll, int adc_idx)
{
    int i, global_idx, num_adcs, local;

    if (!coll || adc_idx < 0 || adc_idx >= coll->total_unique_adcs)
        return 0;

    global_idx = 0;
    for (i = 0; i < coll->num_subsequences; ++i) {
        num_adcs = coll->descriptors[i].num_unique_adcs;
        if (adc_idx < global_idx + num_adcs) {
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

static int pulseqlib__get_num_segments(
    const pulseqlib_collection* coll)
{
    if (!coll) return 0;
    return coll->total_unique_segments;
}

static int pulseqlib__get_segment_duration_us(
    const pulseqlib_collection* coll, int seg_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    int local_seg, k, total;
    const pulseqlib_tr_segment* seg;
    const pulseqlib_block_definition* bdef;

    if (!pulseqlib__resolve_segment(&desc, &local_seg, coll, seg_idx))
        return -1;

    seg = &desc->segment_definitions[local_seg];

    /* Segment-definition pure delays are canonicalized to one block raster;
     * scan-loop instances still expose per-instance delay from block table. */
    if (seg->num_blocks == 1) {
        bdef = &desc->block_definitions[seg->unique_block_indices[0]];
        if (bdef->rf_id == -1 && bdef->gx_id == -1 && bdef->gy_id == -1 &&
            bdef->gz_id == -1 && bdef->adc_id == -1) {
            if (desc->block_raster_us > 0.0f)
                return (int)(desc->block_raster_us + 0.5f);
            return 0;
        }
    }

    total = 0;
    for (k = 0; k < seg->num_blocks; ++k)
        total += desc->block_definitions[seg->unique_block_indices[k]].duration_us;

    return total;
}

static int pulseqlib__is_segment_pure_delay(
    const pulseqlib_collection* coll, int seg_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    int local_seg;
    const pulseqlib_tr_segment* seg;
    const pulseqlib_block_definition* bdef;

    if (!pulseqlib__resolve_segment(&desc, &local_seg, coll, seg_idx))
        return -1;

    seg = &desc->segment_definitions[local_seg];
    if (seg->num_blocks == 1) {
        bdef = &desc->block_definitions[seg->unique_block_indices[0]];
        if (bdef->rf_id == -1 && bdef->gx_id == -1 &&
            bdef->gy_id == -1 && bdef->gz_id == -1 &&
            bdef->adc_id == -1)
            return 1;
    }
    return 0;
}

static int pulseqlib__get_segment_num_blocks(
    const pulseqlib_collection* coll, int seg_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    int local_seg;

    if (!pulseqlib__resolve_segment(&desc, &local_seg, coll, seg_idx))
        return -1;

    return desc->segment_definitions[local_seg].num_blocks;
}

static int pulseqlib__get_segment_start_block(
    const pulseqlib_collection* coll, int seg_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    int local_seg;

    if (!pulseqlib__resolve_segment(&desc, &local_seg, coll, seg_idx))
        return -1;

    return desc->segment_definitions[local_seg].start_block;
}

/* ================================================================== */
/*  Segment table queries                                             */
/* ================================================================== */

static int pulseqlib__get_num_prep_segments(
    const pulseqlib_collection* coll, int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].segment_table.num_prep_segments;
}

static int pulseqlib__get_num_main_segments(
    const pulseqlib_collection* coll, int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].segment_table.num_main_segments;
}

static int pulseqlib__get_num_cooldown_segments(
    const pulseqlib_collection* coll, int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].segment_table.num_cooldown_segments;
}

int pulseqlib_get_prep_segment_table(
    const pulseqlib_collection* coll, int subseq_idx, int* out_ids)
{
    const pulseqlib_sequence_descriptor* desc;
    int n;
    if (!coll || !out_ids) return PULSEQLIB_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;
    desc = &coll->descriptors[subseq_idx];
    n = desc->segment_table.num_prep_segments;
    if (n > 0 && desc->segment_table.prep_segment_table)
        memcpy(out_ids, desc->segment_table.prep_segment_table, n * sizeof(int));
    return n;
}

int pulseqlib_get_main_segment_table(
    const pulseqlib_collection* coll, int subseq_idx, int* out_ids)
{
    const pulseqlib_sequence_descriptor* desc;
    int n;
    if (!coll || !out_ids) return PULSEQLIB_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;
    desc = &coll->descriptors[subseq_idx];
    n = desc->segment_table.num_main_segments;
    if (n > 0 && desc->segment_table.main_segment_table)
        memcpy(out_ids, desc->segment_table.main_segment_table, n * sizeof(int));
    return n;
}

int pulseqlib_get_cooldown_segment_table(
    const pulseqlib_collection* coll, int subseq_idx, int* out_ids)
{
    const pulseqlib_sequence_descriptor* desc;
    int n;
    if (!coll || !out_ids) return PULSEQLIB_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;
    desc = &coll->descriptors[subseq_idx];
    n = desc->segment_table.num_cooldown_segments;
    if (n > 0 && desc->segment_table.cooldown_segment_table)
        memcpy(out_ids, desc->segment_table.cooldown_segment_table, n * sizeof(int));
    return n;
}

/* ================================================================== */
/*  Segment timing queries                                            */
/* ================================================================== */

static int pulseqlib__get_segment_num_kzero_crossings(
    const pulseqlib_collection* coll, int seg_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    int local_seg;

    if (!pulseqlib__resolve_segment(&desc, &local_seg, coll, seg_idx))
        return 0;

    return desc->segment_definitions[local_seg].timing.num_kzero_crossings;
}

static int pulseqlib__get_segment_rf_adc_gap_us(
    const pulseqlib_collection* coll, int seg_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    int local_seg;
    const pulseqlib_segment_timing* tm;
    int r, a;

    if (!pulseqlib__resolve_segment(&desc, &local_seg, coll, seg_idx))
        return -1;

    tm = &desc->segment_definitions[local_seg].timing;

    /* For each RF anchor (in order), find the first ADC anchor whose
     * start is after the RF end.  Return the smallest such gap. */
    {
        int best = -1;
        for (r = 0; r < tm->num_rf_anchors; ++r) {
            int rf_end = tm->rf_anchors[r].end_us;
            for (a = 0; a < tm->num_adc_anchors; ++a) {
                int adc_start = tm->adc_anchors[a].start_us;
                if (adc_start >= rf_end) {
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

static int pulseqlib__get_segment_adc_adc_gap_us(
    const pulseqlib_collection* coll, int seg_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    int local_seg;
    const pulseqlib_segment_timing* tm;
    int a, best;

    if (!pulseqlib__resolve_segment(&desc, &local_seg, coll, seg_idx))
        return -1;

    tm = &desc->segment_definitions[local_seg].timing;
    if (tm->num_adc_anchors < 2)
        return -1;

    best = -1;
    for (a = 1; a < tm->num_adc_anchors; ++a) {
        int gap = (int)(tm->adc_anchors[a].start_us -
                        tm->adc_anchors[a - 1].end_us);
        if (best < 0 || gap < best)
            best = gap;
    }
    return best;
}

/* ------------------------------------------------------------------ */
/*  Per-block RF isocenter and ADC k-zero from segment timing anchors */
/* ------------------------------------------------------------------ */

float pulseqlib_get_rf_isocenter_us(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_rf_definition* rdef;
    int local_blk, k;
    float start_us = 0.0f;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1.0f;

    for (k = 0; k < local_blk; ++k)
        start_us += (float)desc->block_definitions[
            seg->unique_block_indices[k]
        ].duration_us;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id < 0 || bdef->rf_id >= desc->num_unique_rfs)
        return -1.0f;

    rdef = &desc->rf_definitions[bdef->rf_id];
    return start_us + (float)rdef->delay + (float)rdef->stats.isodelay_us;
}

float pulseqlib_get_adc_kzero_us(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_adc_definition* adef;
    int local_blk, k;
    float start_us = 0.0f;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1.0f;

    for (k = 0; k < local_blk; ++k)
        start_us += (float)desc->block_definitions[
            seg->unique_block_indices[k]
        ].duration_us;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->adc_id < 0 || bdef->adc_id >= desc->num_unique_adcs)
        return -1.0f;

    adef = &desc->adc_definitions[bdef->adc_id];
    return start_us + (float)adef->delay +
        (float)(adef->num_samples / 2) * (float)adef->dwell_time * 1e-3f;
}

/* ================================================================== */
/*  Block-level queries (internal helpers for batch getter)            */
/* ================================================================== */

static int pulseqlib__get_block_start_time_us(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, k, start_time;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    start_time = 0;
    for (k = 0; k < local_blk; ++k)
        start_time += desc->block_definitions[seg->unique_block_indices[k]].duration_us;

    return start_time;
}

static int pulseqlib__get_block_duration_us(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk;
    const pulseqlib_block_definition* bdef;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    if (seg->num_blocks == 1 && local_blk == 0) {
        bdef = &desc->block_definitions[seg->unique_block_indices[0]];
        if (bdef->rf_id == -1 && bdef->gx_id == -1 && bdef->gy_id == -1 &&
            bdef->gz_id == -1 && bdef->adc_id == -1) {
            if (desc->block_raster_us > 0.0f)
                return (int)(desc->block_raster_us + 0.5f);
            return 0;
        }
    }

    return desc->block_definitions[seg->unique_block_indices[local_blk]].duration_us;
}

/* ================================================================== */
/*  RF queries (internal helpers + public waveform getters)            */
/* ================================================================== */

static int pulseqlib__block_has_rf(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk;
    const pulseqlib_block_definition* bdef;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    return (bdef->rf_id != -1) ? 1 : 0;
}

static int pulseqlib__block_rf_has_uniform_raster(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_rf_definition* rdef;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1) return -1;

    rdef = &desc->rf_definitions[bdef->rf_id];
    return (rdef->time_shape_id != 0) ? 1 : 0;
}

static int pulseqlib__block_rf_is_complex(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_rf_definition* rdef;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1) return -1;

    rdef = &desc->rf_definitions[bdef->rf_id];
    return (rdef->phase_shape_id != 0) ? 1 : 0;
}

static int pulseqlib__get_rf_num_samples(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, shape_idx, total, nch;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_rf_definition* rdef;
    const pulseqlib_shape_arbitrary* shape;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1) return -1;

    rdef = &desc->rf_definitions[bdef->rf_id];
    total = -1;

    /* try mag, then phase, then time shape */
    if (rdef->mag_shape_id > 0) {
        shape_idx = rdef->mag_shape_id - 1;
        if (shape_idx >= 0 && shape_idx < desc->num_shapes) {
            shape = &desc->shapes[shape_idx];
            if (shape->num_uncompressed_samples > 0)
                total = shape->num_uncompressed_samples;
        }
    }
    if (total < 0 && rdef->phase_shape_id > 0) {
        shape_idx = rdef->phase_shape_id - 1;
        if (shape_idx >= 0 && shape_idx < desc->num_shapes) {
            shape = &desc->shapes[shape_idx];
            if (shape->num_uncompressed_samples > 0)
                total = shape->num_uncompressed_samples;
        }
    }
    if (total < 0 && rdef->time_shape_id > 0) {
        shape_idx = rdef->time_shape_id - 1;
        if (shape_idx >= 0 && shape_idx < desc->num_shapes) {
            shape = &desc->shapes[shape_idx];
            if (shape->num_uncompressed_samples > 0)
                total = shape->num_uncompressed_samples;
        }
    }
    if (total < 0) return -1;

    nch = (rdef->num_channels > 1) ? rdef->num_channels : 1;
    return total / nch;
}

static int pulseqlib__get_rf_delay_us(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk;
    const pulseqlib_block_definition* bdef;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1) return -1;

    return desc->rf_definitions[bdef->rf_id].delay;
}

static int pulseqlib__get_rf_num_channels(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_rf_definition* rdef;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1) return -1;

    rdef = &desc->rf_definitions[bdef->rf_id];
    return (rdef->num_channels > 1) ? rdef->num_channels : 1;
}

float** pulseqlib_get_rf_magnitude(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx,
    int* num_channels, int* num_samples)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, shape_idx, nch, npts, ch;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_rf_definition* rdef;
    pulseqlib_shape_arbitrary decompressed;
    float* flat;
    float** result;

    if (!num_channels || !num_samples) return NULL;
    *num_channels = 0;
    *num_samples = 0;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return NULL;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1) return NULL;

    rdef = &desc->rf_definitions[bdef->rf_id];
    if (rdef->mag_shape_id <= 0) return NULL;

    shape_idx = rdef->mag_shape_id - 1;
    if (shape_idx < 0 || shape_idx >= desc->num_shapes) return NULL;

    decompressed.num_samples = 0;
    decompressed.num_uncompressed_samples = 0;
    decompressed.samples = NULL;

    if (!pulseqlib__decompress_shape(&decompressed, &desc->shapes[shape_idx],
                                     1.0f))
        return NULL;

    flat = decompressed.samples;
    nch  = (rdef->num_channels > 1) ? rdef->num_channels : 1;
    npts = decompressed.num_samples / nch;

    /* allocate channel-pointer array */
    result = (float**)PULSEQLIB_ALLOC((size_t)nch * sizeof(float*));
    if (!result) { PULSEQLIB_FREE(flat); return NULL; }

    /* split tiled flat array into per-channel rows */
    for (ch = 0; ch < nch; ++ch) {
        result[ch] = (float*)PULSEQLIB_ALLOC((size_t)npts * sizeof(float));
        if (!result[ch]) {
            int k;
            for (k = 0; k < ch; ++k) PULSEQLIB_FREE(result[k]);
            PULSEQLIB_FREE(result);
            PULSEQLIB_FREE(flat);
            return NULL;
        }
        memcpy(result[ch], flat + ch * npts, (size_t)npts * sizeof(float));
    }

    PULSEQLIB_FREE(flat);
    *num_channels = nch;
    *num_samples  = npts;
    return result;
}

float** pulseqlib_get_rf_phase(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx,
    int* num_channels, int* num_samples)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, shape_idx, nch, npts, ch;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_rf_definition* rdef;
    pulseqlib_shape_arbitrary decompressed;
    float* flat;
    float** result;

    if (!num_channels || !num_samples) return NULL;
    *num_channels = 0;
    *num_samples = 0;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return NULL;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1) return NULL;

    rdef = &desc->rf_definitions[bdef->rf_id];
    if (rdef->phase_shape_id <= 0) return NULL;

    shape_idx = rdef->phase_shape_id - 1;
    if (shape_idx < 0 || shape_idx >= desc->num_shapes) return NULL;

    decompressed.num_samples = 0;
    decompressed.num_uncompressed_samples = 0;
    decompressed.samples = NULL;

    if (!pulseqlib__decompress_shape(&decompressed, &desc->shapes[shape_idx], 1.0f))
        return NULL;

    flat = decompressed.samples;
    nch  = (rdef->num_channels > 1) ? rdef->num_channels : 1;
    npts = decompressed.num_samples / nch;

    /* allocate channel-pointer array */
    result = (float**)PULSEQLIB_ALLOC((size_t)nch * sizeof(float*));
    if (!result) { PULSEQLIB_FREE(flat); return NULL; }

    /* split tiled flat array into per-channel rows */
    for (ch = 0; ch < nch; ++ch) {
        result[ch] = (float*)PULSEQLIB_ALLOC((size_t)npts * sizeof(float));
        if (!result[ch]) {
            int k;
            for (k = 0; k < ch; ++k) PULSEQLIB_FREE(result[k]);
            PULSEQLIB_FREE(result);
            PULSEQLIB_FREE(flat);
            return NULL;
        }
        memcpy(result[ch], flat + ch * npts, (size_t)npts * sizeof(float));
    }

    PULSEQLIB_FREE(flat);
    *num_channels = nch;
    *num_samples  = npts;
    return result;
}

float* pulseqlib_get_rf_time_us(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, shape_idx, nch, npts;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_rf_definition* rdef;
    pulseqlib_shape_arbitrary decompressed;
    float* result;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return NULL;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1) return NULL;

    rdef = &desc->rf_definitions[bdef->rf_id];
    if (rdef->time_shape_id <= 0) return NULL;

    shape_idx = rdef->time_shape_id - 1;
    if (shape_idx < 0 || shape_idx >= desc->num_shapes) return NULL;

    decompressed.num_samples = 0;
    decompressed.num_uncompressed_samples = 0;
    decompressed.samples = NULL;

    if (!pulseqlib__decompress_shape(&decompressed, &desc->shapes[shape_idx],
                                     desc->rf_raster_us))
        return NULL;

    nch  = (rdef->num_channels > 1) ? rdef->num_channels : 1;
    npts = decompressed.num_samples / nch;

    if (nch > 1) {
        /* return only first channel's time (all channels share time base) */
        result = (float*)PULSEQLIB_ALLOC((size_t)npts * sizeof(float));
        if (!result) { PULSEQLIB_FREE(decompressed.samples); return NULL; }
        memcpy(result, decompressed.samples, (size_t)npts * sizeof(float));
        PULSEQLIB_FREE(decompressed.samples);
    } else {
        result = decompressed.samples;
    }

    return result;
}

float pulseqlib_get_rf_initial_amplitude_hz(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, bt_idx, rf_event_id;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_rf_definition* rdef;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return 0.0f;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1) return 0.0f;

    rdef = &desc->rf_definitions[bdef->rf_id];

    bt_idx = seg->max_energy_start_block + local_blk;
    rf_event_id = desc->block_table[bt_idx].rf_id;
    if (rf_event_id >= 0 && rf_event_id < desc->rf_table_size)
        return desc->rf_table[rf_event_id].amplitude;

    return rdef->stats.base_amplitude_hz;
}

float pulseqlib_get_rf_max_amplitude_hz(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_rf_definition* rdef;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return 0.0f;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1) return 0.0f;

    rdef = &desc->rf_definitions[bdef->rf_id];
    return rdef->stats.base_amplitude_hz;
}

/* ================================================================== */
/*  Gradient queries (internal helpers + public waveform getters)      */
/* ================================================================== */

static int pulseqlib__block_has_grad(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx, int axis)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, grad_id;
    const pulseqlib_block_definition* bdef;

    if (axis < PULSEQLIB_GRAD_AXIS_X || axis > PULSEQLIB_GRAD_AXIS_Z) return -1;
    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    grad_id = get_grad_id_by_axis(bdef, axis);
    return (grad_id != -1) ? 1 : 0;
}

static int pulseqlib__block_grad_is_trapezoid(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx, int axis)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, grad_id;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_grad_definition* gdef;

    if (axis < PULSEQLIB_GRAD_AXIS_X || axis > PULSEQLIB_GRAD_AXIS_Z) return -1;
    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    grad_id = get_grad_id_by_axis(bdef, axis);
    if (grad_id == -1) return -1;

    gdef = &desc->grad_definitions[grad_id];
    if (gdef->type == 0) return 1;
    if (gdef->unused_or_time_shape_id > 0) return 1;
    return 0;
}

static int pulseqlib__get_grad_num_samples(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx, int axis)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, grad_id, shape_idx;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_grad_definition* gdef;
    const pulseqlib_shape_arbitrary* shape;

    if (axis < PULSEQLIB_GRAD_AXIS_X || axis > PULSEQLIB_GRAD_AXIS_Z) return -1;
    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    grad_id = get_grad_id_by_axis(bdef, axis);
    if (grad_id == -1) return -1;

    gdef = &desc->grad_definitions[grad_id];

    if (gdef->type == 0) {
        return (gdef->flat_time_or_unused > 0) ? 4 : 3;
    }

    /* arbitrary: get from first shot shape */
    if (gdef->num_shots > 0 && gdef->shot_shape_ids[0] > 0) {
        shape_idx = gdef->shot_shape_ids[0] - 1;
        if (shape_idx >= 0 && shape_idx < desc->num_shapes) {
            shape = &desc->shapes[shape_idx];
            if (shape->num_uncompressed_samples > 0)
                return shape->num_uncompressed_samples;
        }
    }
    return -1;
}

static int pulseqlib__get_grad_num_shots(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx, int axis)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, grad_id;
    const pulseqlib_block_definition* bdef;

    if (axis < PULSEQLIB_GRAD_AXIS_X || axis > PULSEQLIB_GRAD_AXIS_Z) return -1;
    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    grad_id = get_grad_id_by_axis(bdef, axis);
    if (grad_id == -1) return -1;

    return desc->grad_definitions[grad_id].num_shots;
}

static int pulseqlib__get_grad_delay_us(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx, int axis)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, grad_id;
    const pulseqlib_block_definition* bdef;

    if (axis < PULSEQLIB_GRAD_AXIS_X || axis > PULSEQLIB_GRAD_AXIS_Z) return -1;
    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    grad_id = get_grad_id_by_axis(bdef, axis);
    if (grad_id == -1) return -1;

    return desc->grad_definitions[grad_id].delay;
}

float** pulseqlib_get_grad_amplitude(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx, int axis,
    int* num_shots, int* num_samples)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, grad_id, shot, k, shape_idx;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_grad_definition* gdef;
    float** waveforms;
    float* trap_waveform;
    int samples_per_shot;
    int flat_time;
    pulseqlib_shape_arbitrary decompressed;

    if (!num_shots || !num_samples) {
        if (num_shots) *num_shots = 0;
        if (num_samples) *num_samples = 0;
        return NULL;
    }
    *num_shots = 0;
    *num_samples = 0;

    if (axis < PULSEQLIB_GRAD_AXIS_X || axis > PULSEQLIB_GRAD_AXIS_Z) return NULL;
    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return NULL;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    grad_id = get_grad_id_by_axis(bdef, axis);
    if (grad_id == -1) return NULL;

    gdef = &desc->grad_definitions[grad_id];

    waveforms = (float**)PULSEQLIB_ALLOC(gdef->num_shots * sizeof(float*));
    if (!waveforms) return NULL;

    *num_shots = gdef->num_shots;

    if (gdef->type == 0) {
        flat_time = gdef->flat_time_or_unused;
        samples_per_shot = (flat_time > 0) ? 4 : 3;
        *num_samples = samples_per_shot;

        for (shot = 0; shot < gdef->num_shots; ++shot) {
            trap_waveform = (float*)PULSEQLIB_ALLOC(samples_per_shot * sizeof(float));
            if (!trap_waveform) {
                for (k = 0; k < shot; ++k) PULSEQLIB_FREE(waveforms[k]);
                PULSEQLIB_FREE(waveforms);
                *num_shots = 0;
                *num_samples = 0;
                return NULL;
            }

            trap_waveform[0] = 0.0f;
            trap_waveform[1] = 1.0f;
            if (flat_time > 0) {
                trap_waveform[2] = 1.0f;
                trap_waveform[3] = 0.0f;
            } else {
                trap_waveform[2] = 0.0f;
            }

            waveforms[shot] = trap_waveform;
        }
    } else {
        for (shot = 0; shot < gdef->num_shots; ++shot) {
            if (gdef->shot_shape_ids[shot] <= 0) {
                waveforms[shot] = NULL;
                continue;
            }

            shape_idx = gdef->shot_shape_ids[shot] - 1;
            if (shape_idx < 0 || shape_idx >= desc->num_shapes) {
                waveforms[shot] = NULL;
                continue;
            }

            decompressed.num_samples = 0;
            decompressed.num_uncompressed_samples = 0;
            decompressed.samples = NULL;

            if (!pulseqlib__decompress_shape(&decompressed, &desc->shapes[shape_idx],
                                             1.0f)) {
                waveforms[shot] = NULL;
                continue;
            }

            waveforms[shot] = decompressed.samples;
            if (*num_samples == 0)
                *num_samples = decompressed.num_samples;
        }
    }

    return waveforms;
}

float pulseqlib_get_grad_initial_amplitude_hz_per_m(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx, int axis)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, block_table_idx, grad_event_id;
    const pulseqlib_block_table_element* bte;

    if (axis < PULSEQLIB_GRAD_AXIS_X || axis > PULSEQLIB_GRAD_AXIS_Z) return 1.0f;
    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return 1.0f;

    block_table_idx = seg->max_energy_start_block + local_blk;
    bte = &desc->block_table[block_table_idx];

    switch (axis) {
        case PULSEQLIB_GRAD_AXIS_X: grad_event_id = bte->gx_id; break;
        case PULSEQLIB_GRAD_AXIS_Y: grad_event_id = bte->gy_id; break;
        case PULSEQLIB_GRAD_AXIS_Z: grad_event_id = bte->gz_id; break;
        default: return 1.0f;
    }

    if (grad_event_id < 0 || grad_event_id >= desc->grad_table_size)
        return 1.0f;

    return desc->grad_table[grad_event_id].amplitude;
}

int pulseqlib_get_grad_initial_shot_id(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx, int axis)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, block_table_idx, grad_event_id;
    const pulseqlib_block_table_element* bte;

    if (axis < PULSEQLIB_GRAD_AXIS_X || axis > PULSEQLIB_GRAD_AXIS_Z) return 0;
    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return 0;

    block_table_idx = seg->max_energy_start_block + local_blk;
    bte = &desc->block_table[block_table_idx];

    switch (axis) {
        case PULSEQLIB_GRAD_AXIS_X: grad_event_id = bte->gx_id; break;
        case PULSEQLIB_GRAD_AXIS_Y: grad_event_id = bte->gy_id; break;
        case PULSEQLIB_GRAD_AXIS_Z: grad_event_id = bte->gz_id; break;
        default: return 0;
    }

    if (grad_event_id < 0 || grad_event_id >= desc->grad_table_size)
        return 0;

    return desc->grad_table[grad_event_id].shot_index;
}

float pulseqlib_get_grad_max_amplitude_hz_per_m(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx, int axis)
{
    float init_amp;

    if (axis < PULSEQLIB_GRAD_AXIS_X || axis > PULSEQLIB_GRAD_AXIS_Z)
        return 0.0f;
    init_amp = pulseqlib_get_grad_initial_amplitude_hz_per_m(
        coll, seg_idx, blk_idx, axis);
    return (init_amp >= 0.0f) ? init_amp : -init_amp;
}

float* pulseqlib_get_grad_time_us(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx, int axis)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, grad_id, shape_idx;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_grad_definition* gdef;
    float* time_waveform;
    float accum;
    int rise_time, flat_time, fall_time, ns;
    pulseqlib_shape_arbitrary decompressed;

    if (axis < PULSEQLIB_GRAD_AXIS_X || axis > PULSEQLIB_GRAD_AXIS_Z) return NULL;
    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return NULL;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    grad_id = get_grad_id_by_axis(bdef, axis);
    if (grad_id == -1) return NULL;

    gdef = &desc->grad_definitions[grad_id];

    if (gdef->type == 0) {
        rise_time = gdef->rise_time_or_unused;
        flat_time = gdef->flat_time_or_unused;
        fall_time = gdef->fall_time_or_num_uncompressed_samples;

        ns = (flat_time > 0) ? 4 : 3;

        time_waveform = (float*)PULSEQLIB_ALLOC((size_t)ns * sizeof(float));
        if (!time_waveform) return NULL;

        accum = 0.0f;
        time_waveform[0] = accum;
        accum += (float)rise_time;
        time_waveform[1] = accum;
        if (flat_time > 0) {
            accum += (float)flat_time;
            time_waveform[2] = accum;
            accum += (float)fall_time;
            time_waveform[3] = accum;
        } else {
            accum += (float)fall_time;
            time_waveform[2] = accum;
        }

        return time_waveform;
    }

    /* arbitrary: decompress time shape */
    if (gdef->unused_or_time_shape_id <= 0) return NULL;

    shape_idx = gdef->unused_or_time_shape_id - 1;
    if (shape_idx < 0 || shape_idx >= desc->num_shapes) return NULL;

    decompressed.num_samples = 0;
    decompressed.num_uncompressed_samples = 0;
    decompressed.samples = NULL;

    if (!pulseqlib__decompress_shape(&decompressed, &desc->shapes[shape_idx],
                                     desc->grad_raster_us))
        return NULL;

    return decompressed.samples;
}

/* ================================================================== */
/*  ADC block queries (internal helpers)                               */
/* ================================================================== */

static int pulseqlib__block_has_adc(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk;
    const pulseqlib_block_definition* bdef;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];

    return (bdef->adc_id != -1) ? 1 : 0;
}

static int pulseqlib__get_adc_delay_us(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, adc_id;
    const pulseqlib_block_definition* bdef;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    adc_id = bdef->adc_id;
    if (adc_id < 0 || adc_id >= desc->num_unique_adcs) return -1;

    return desc->adc_definitions[adc_id].delay;
}

static int pulseqlib__get_adc_library_index(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, adc_id, global_adc_idx, i;
    const pulseqlib_block_definition* bdef;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    adc_id = bdef->adc_id;
    if (adc_id < 0 || adc_id >= desc->num_unique_adcs) return -1;

    /* compute global index: sum ADC counts from prior subsequences */
    global_adc_idx = 0;
    for (i = 0; i < coll->num_subsequences; ++i) {
        if (&coll->descriptors[i] == desc) break;
        global_adc_idx += coll->descriptors[i].num_unique_adcs;
    }
    return global_adc_idx + adc_id;
}

/* ================================================================== */
/*  Digital output / trigger / flag queries (internal helpers)         */
/* ================================================================== */

static int pulseqlib__block_has_digitalout(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    return seg->has_digitalout[local_blk];
}

static int pulseqlib__get_digitalout_delay_us(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, digitalout_id;
    const pulseqlib_block_table_element* bte;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    if (!seg->has_digitalout[local_blk]) return -1;

    bte = &desc->block_table[seg->start_block + local_blk];
    digitalout_id = bte->digitalout_id;
    if (digitalout_id == -1 || digitalout_id >= desc->num_triggers) return -1;

    return (int)desc->trigger_events[digitalout_id].delay;
}

static int pulseqlib__get_digitalout_duration_us(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, digitalout_id;
    const pulseqlib_block_table_element* bte;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    if (!seg->has_digitalout[local_blk]) return -1;

    bte = &desc->block_table[seg->start_block + local_blk];
    digitalout_id = bte->digitalout_id;
    if (digitalout_id == -1 || digitalout_id >= desc->num_triggers) return -1;

    return (int)desc->trigger_events[digitalout_id].duration;
}

/* ---- Segment-level physio trigger queries ------------------------ */

static int pulseqlib__segment_has_trigger(
    const pulseqlib_collection* coll,
    int seg_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    int local_seg;

    if (!pulseqlib__resolve_segment(&desc, &local_seg, coll, seg_idx))
        return 0;

    return (desc->segment_definitions[local_seg].trigger_id >= 0) ? 1 : 0;
}

static int pulseqlib__get_segment_trigger_delay_us(
    const pulseqlib_collection* coll,
    int seg_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    int local_seg, tid;

    if (!pulseqlib__resolve_segment(&desc, &local_seg, coll, seg_idx))
        return -1;

    tid = desc->segment_definitions[local_seg].trigger_id;
    if (tid < 0 || tid >= desc->num_triggers) return -1;

    return (int)desc->trigger_events[tid].delay;
}

static int pulseqlib__get_segment_trigger_duration_us(
    const pulseqlib_collection* coll,
    int seg_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    int local_seg, tid;

    if (!pulseqlib__resolve_segment(&desc, &local_seg, coll, seg_idx))
        return -1;

    tid = desc->segment_definitions[local_seg].trigger_id;
    if (tid < 0 || tid >= desc->num_triggers) return -1;

    return (int)desc->trigger_events[tid].duration;
}

/* ---- Navigator flag query ---------------------------------------- */

static int pulseqlib__segment_is_nav(
    const pulseqlib_collection* coll,
    int seg_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    int local_seg;

    if (!pulseqlib__resolve_segment(&desc, &local_seg, coll, seg_idx))
        return 0;

    return desc->segment_definitions[local_seg].is_nav;
}

static int pulseqlib__block_has_freq_mod(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return 0;

    return seg->has_freq_mod[local_blk];
}

static int pulseqlib__block_has_rotation(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    return seg->has_rotation[local_blk];
}

static int pulseqlib__block_has_norot(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    return seg->norot_flag[local_blk];
}

static int pulseqlib__block_has_nopos(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    return seg->nopos_flag[local_blk];
}

/* ------------------------------------------------------------------ */
/*  pulseqlib_block_needs_freq_mod — precise overlap + nopos check    */
/* ------------------------------------------------------------------ */

/*
 * Check whether a trapezoid gradient's flat region overlaps [win_start, win_end].
 * All times in us, relative to block start.
 */
static int trap_overlaps_window(
    const pulseqlib_grad_definition* gdef,
    float win_start, float win_end)
{
    float flat_start = (float)gdef->delay +
                       (float)gdef->rise_time_or_unused;
    float flat_end   = flat_start + (float)gdef->flat_time_or_unused;
    return (flat_start < win_end) && (flat_end > win_start);
}

/*
 * Check whether an arbitrary gradient waveform has any nonzero sample
 * within [win_start, win_end].  Times in us, relative to block start.
 */
static int arb_nonzero_in_window(
    const pulseqlib_sequence_descriptor* desc,
    const pulseqlib_grad_definition* gdef,
    float win_start, float win_end)
{
    int shape_idx, i, ns;
    float raster, local_start, local_end;
    int idx_lo, idx_hi;
    pulseqlib_shape_arbitrary decomp;

    if (gdef->num_shots < 1 || gdef->shot_shape_ids[0] <= 0)
        return 0;

    shape_idx = gdef->shot_shape_ids[0] - 1;
    if (shape_idx < 0 || shape_idx >= desc->num_shapes)
        return 0;

    decomp.num_samples = 0;
    decomp.num_uncompressed_samples = 0;
    decomp.samples = NULL;
    if (!pulseqlib__decompress_shape(&decomp, &desc->shapes[shape_idx], 1.0f))
        return 0;

    ns = decomp.num_samples;
    raster = desc->grad_raster_us;
    local_start = win_start - (float)gdef->delay;
    local_end   = win_end   - (float)gdef->delay;

    idx_lo = (int)(local_start / raster);
    if (idx_lo < 0) idx_lo = 0;
    idx_hi = (int)(local_end / raster);
    if (idx_hi >= ns) idx_hi = ns - 1;

    for (i = idx_lo; i <= idx_hi; ++i) {
        if (decomp.samples[i] != 0.0f) {
            PULSEQLIB_FREE(decomp.samples);
            return 1;
        }
    }

    PULSEQLIB_FREE(decomp.samples);
    return 0;
}

/*
 * Check whether any gradient axis has nonzero amplitude within the
 * given temporal window (us, relative to block start).
 */
static int any_grad_overlaps_window(
    const pulseqlib_sequence_descriptor* desc,
    const pulseqlib_block_definition* bdef,
    float win_start, float win_end)
{
    int axis, grad_id;

    for (axis = 0; axis < 3; ++axis) {
        grad_id = get_grad_id_by_axis(bdef, axis);
        if (grad_id < 0) continue;

        if (desc->grad_definitions[grad_id].type == 0) {
            /* trapezoid */
            if (trap_overlaps_window(&desc->grad_definitions[grad_id],
                                     win_start, win_end))
                return 1;
        } else {
            /* arbitrary */
            if (arb_nonzero_in_window(desc, &desc->grad_definitions[grad_id],
                                      win_start, win_end))
                return 1;
        }
    }
    return 0;
}

int pulseqlib_block_needs_freq_mod(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx,
    int* num_samples)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk;
    const pulseqlib_block_definition* bdef;
    int has_rf, has_adc, nopos;

    if (num_samples) *num_samples = 0;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return 0;

    nopos = seg->nopos_flag[local_blk];
    if (nopos) return 0;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    has_rf  = (bdef->rf_id >= 0);
    has_adc = (bdef->adc_id >= 0);
    if (!has_rf && !has_adc) return 0;

    /* Check RF window overlap */
    if (has_rf) {
        const pulseqlib_rf_definition* rdef = &desc->rf_definitions[bdef->rf_id];
        float rf_start = (float)rdef->delay;
        float rf_end   = rf_start + rdef->stats.duration_us;

        if (any_grad_overlaps_window(desc, bdef, rf_start, rf_end)) {
            if (num_samples)
                *num_samples = (int)((float)bdef->duration_us /
                                     desc->rf_raster_us);
            return 1;
        }
    }

    /* Check ADC window overlap */
    if (has_adc) {
        const pulseqlib_adc_definition* adef = &desc->adc_definitions[bdef->adc_id];
        float adc_start = (float)adef->delay;
        float adc_end   = adc_start +
                          (float)adef->num_samples *
                          (float)adef->dwell_time * 1e-3f;

        if (any_grad_overlaps_window(desc, bdef, adc_start, adc_end)) {
            if (num_samples)
                *num_samples = (int)((float)bdef->duration_us /
                                     desc->adc_raster_us);
            return 1;
        }
    }

    return 0;
}

int pulseqlib_cursor_next(pulseqlib_collection* coll)
{
    pulseqlib_block_cursor* cursor;
    const pulseqlib_sequence_descriptor* desc;
    int next_pos;

    cursor = &coll->block_cursor;

    if (cursor->sequence_index >= coll->num_subsequences)
        return PULSEQLIB_CURSOR_DONE;

    desc = &coll->descriptors[cursor->sequence_index];
    next_pos = cursor->scan_table_position + 1;

    /* Past end of scan table: advance to next subsequence */
    if (next_pos >= desc->scan_table_len) {
        cursor->sequence_index += 1;
        cursor->scan_table_position = 0;
        cursor->from_last_reset = 0;
        if (cursor->sequence_index >= coll->num_subsequences)
            return PULSEQLIB_CURSOR_DONE;
        return PULSEQLIB_CURSOR_BLOCK;
    }

    /* Normal advance */
    cursor->scan_table_position = next_pos;
    cursor->from_last_reset += 1;
    return PULSEQLIB_CURSOR_BLOCK;
}

void pulseqlib_cursor_reset(pulseqlib_collection* coll)
{
    pulseqlib_block_cursor* cursor;

    cursor = &coll->block_cursor;

    /* Go back by the number of blocks advanced since the last mark */
    cursor->scan_table_position -= cursor->from_last_reset;
    cursor->from_last_reset = 0;
}

void pulseqlib_cursor_mark(pulseqlib_collection* coll)
{
    if (!coll) return;
    coll->block_cursor.from_last_reset = 0;
}

int pulseqlib_cursor_get_info(
    const pulseqlib_collection* coll,
    pulseqlib_cursor_info* info)
{
    const pulseqlib_block_cursor* cursor;
    const pulseqlib_sequence_descriptor* desc;
    int pos, seg_id;

    if (!coll || !info) return PULSEQLIB_ERR_NULL_POINTER;

    cursor = &coll->block_cursor;
    if (cursor->sequence_index < 0 ||
        cursor->sequence_index >= coll->num_subsequences)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[cursor->sequence_index];
    pos  = cursor->scan_table_position;
    if (pos < 0 || pos >= desc->scan_table_len)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;

    seg_id = desc->scan_table_seg_id[pos];

    info->subseq_idx    = cursor->sequence_index;
    info->scan_pos      = pos;
    info->segment_id    = seg_id + coll->subsequence_info[cursor->sequence_index].segment_id_offset;
    info->segment_start = (pos == 0 || desc->scan_table_seg_id[pos] !=
                                       desc->scan_table_seg_id[pos - 1]) ? 1 : 0;
    info->segment_end   = (pos == desc->scan_table_len - 1 ||
                           desc->scan_table_seg_id[pos] !=
                           desc->scan_table_seg_id[pos + 1]) ? 1 : 0;
    info->tr_start      = desc->scan_table_tr_start
                          ? desc->scan_table_tr_start[pos] : 0;
    info->pmc           = desc->enable_pmc;

    /* Segment properties via local segment index */
    {
        int local_seg = seg_id;   /* seg_id in scan table is local (before offset) */
        if (local_seg >= 0 && local_seg < desc->num_unique_segments) {
            info->is_nav      = desc->segment_definitions[local_seg].is_nav;
            info->has_trigger = (desc->segment_definitions[local_seg].trigger_id >= 0) ? 1 : 0;
        } else {
            info->is_nav      = 0;
            info->has_trigger = 0;
        }
    }

    return PULSEQLIB_SUCCESS;
}

int pulseqlib_get_block_instance(
    const pulseqlib_collection* coll,
    pulseqlib_block_instance* inst)
{
    const pulseqlib_block_cursor* cursor;
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_block_table_element* bte;
    const pulseqlib_block_definition* bdef;
    int idx, i;

    if (!coll || !inst) return PULSEQLIB_ERR_NULL_POINTER;

    cursor = &coll->block_cursor;
    if (cursor->sequence_index >= coll->num_subsequences)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[cursor->sequence_index];
    if (cursor->scan_table_position < 0 ||
        cursor->scan_table_position >= desc->scan_table_len)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;

    idx  = desc->scan_table_block_idx[cursor->scan_table_position];
    if (idx < 0 || idx >= desc->num_blocks)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;

    bte  = &desc->block_table[idx];
    bdef = &desc->block_definitions[bte->id];

    /* Duration: pure delay uses instance value, normal block uses definition */
    inst->duration_us = (bte->duration_us >= 0)
                        ? bte->duration_us
                        : bdef->duration_us;

    /* RF */
    if (bte->rf_id >= 0 && bte->rf_id < desc->rf_table_size) {
        inst->rf_amp_hz   = desc->rf_table[bte->rf_id].amplitude;
        inst->rf_freq_hz  = desc->rf_table[bte->rf_id].freq_offset;
        inst->rf_phase_rad = desc->rf_table[bte->rf_id].phase_offset;
    } else {
        inst->rf_amp_hz   = 0.0f;
        inst->rf_freq_hz  = 0.0f;
        inst->rf_phase_rad = 0.0f;
    }

    /* Gradients */
    if (bte->gx_id >= 0 && bte->gx_id < desc->grad_table_size) {
        inst->gx_amp_hz_per_m      = desc->grad_table[bte->gx_id].amplitude;
        inst->gx_shot_idx = desc->grad_table[bte->gx_id].shot_index;
    } else {
        inst->gx_amp_hz_per_m = 0.0f;
        inst->gx_shot_idx = 0;
    }
    if (bte->gy_id >= 0 && bte->gy_id < desc->grad_table_size) {
        inst->gy_amp_hz_per_m      = desc->grad_table[bte->gy_id].amplitude;
        inst->gy_shot_idx = desc->grad_table[bte->gy_id].shot_index;
    } else {
        inst->gy_amp_hz_per_m = 0.0f;
        inst->gy_shot_idx = 0;
    }
    if (bte->gz_id >= 0 && bte->gz_id < desc->grad_table_size) {
        inst->gz_amp_hz_per_m      = desc->grad_table[bte->gz_id].amplitude;
        inst->gz_shot_idx = desc->grad_table[bte->gz_id].shot_index;
    } else {
        inst->gz_amp_hz_per_m = 0.0f;
        inst->gz_shot_idx = 0;
    }

    /* Rotation */
    if (bte->rotation_id >= 0 && bte->rotation_id < desc->num_rotations) {
        for (i = 0; i < 9; ++i)
            inst->rotmat[i] = desc->rotation_matrices[bte->rotation_id][i];
    } else {
        inst->rotmat[0] = 1.0f; inst->rotmat[1] = 0.0f; inst->rotmat[2] = 0.0f;
        inst->rotmat[3] = 0.0f; inst->rotmat[4] = 1.0f; inst->rotmat[5] = 0.0f;
        inst->rotmat[6] = 0.0f; inst->rotmat[7] = 0.0f; inst->rotmat[8] = 1.0f;
    }

    /* Flags */
    inst->norot_flag = bte->norot_flag;
    inst->nopos_flag = bte->nopos_flag;

    /* Digital output */
    inst->digitalout_flag = (bte->digitalout_id >= 0) ? 1 : 0;

    /* ADC */
    if (bte->adc_id >= 0 && bte->adc_id < desc->adc_table_size) {
        inst->adc_flag  = 1;
        inst->adc_freq_hz  = desc->adc_table[bte->adc_id].freq_offset;
        inst->adc_phase_rad = desc->adc_table[bte->adc_id].phase_offset;
    } else {
        inst->adc_flag  = 0;
        inst->adc_freq_hz  = 0.0f;
        inst->adc_phase_rad = 0.0f;
    }

    /* RF shimming */
    inst->rf_shim_id = bte->rf_shim_id;

    return PULSEQLIB_SUCCESS;
}

/* ================================================================== */
/*  Frequency modulation plan                                         */
/* ================================================================== */

/* --- helper: block range for a TR region --- */
static int get_block_range(
    const pulseqlib_sequence_descriptor* desc,
    int tr_type, int tr_index,
    int* out_start, int* out_count)
{
    const pulseqlib_tr_descriptor* tr = &desc->tr_descriptor;

    switch (tr_type) {
    case PULSEQLIB_TR_REGION_PREP:
        *out_start = 0;
        *out_count = desc->num_prep_blocks;
        return 1;
    case PULSEQLIB_TR_REGION_MAIN:
        if (tr->tr_size <= 0 || tr_index < 0 || tr_index >= tr->num_trs)
            return 0;
        *out_start = desc->num_prep_blocks + tr_index * tr->tr_size;
        *out_count = tr->tr_size;
        return 1;
    case PULSEQLIB_TR_REGION_COOLDOWN:
        *out_start = desc->pass_len - desc->num_cooldown_blocks;
        *out_count = desc->num_cooldown_blocks;
        return 1;
    case PULSEQLIB_TR_REGION_ALL:
        *out_start = 0;
        *out_count = desc->pass_len;
        return 1;
    default:
        return 0;
    }
}

/* --- helper: count RF+ADC events in a block range --- */
static int count_fm_events_range(
    const pulseqlib_sequence_descriptor* desc,
    int blk_start, int blk_count)
{
    int n, count;
    count = 0;
    for (n = blk_start; n < blk_start + blk_count && n < desc->num_blocks; ++n) {
        if (desc->block_table[n].freq_mod_id >= 0)
            ++count;
    }
    return count;
}

int pulseqlib_get_freq_mod_count(
    const pulseqlib_collection* coll)
{
    int i, total;
    if (!coll) return 0;
    total = 0;
    for (i = 0; i < coll->num_subsequences; ++i)
        total += count_fm_events_range(&coll->descriptors[i],
                                       0, coll->descriptors[i].num_blocks);
    return total;
}

int pulseqlib_get_freq_mod_count_tr(
    const pulseqlib_collection* coll,
    int tr_type, int tr_index)
{
    int blk_start, blk_count;
    const pulseqlib_sequence_descriptor* desc;

    if (!coll || coll->num_subsequences < 1) return 0;
    desc = &coll->descriptors[0];
    if (!get_block_range(desc, tr_type, tr_index, &blk_start, &blk_count))
        return 0;
    return count_fm_events_range(desc, blk_start, blk_count);
}

/* ================================================================== */
/*  Label getters                                                     */
/* ================================================================== */

int pulseqlib_get_label_limits(const pulseqlib_collection* coll,
                               int subseq_idx,
                               pulseqlib_label_limits* limits)
{
    if (!coll || !limits) return PULSEQLIB_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;
    *limits = coll->descriptors[subseq_idx].label_limits;
    return PULSEQLIB_SUCCESS;
}

static int pulseqlib__get_num_adc_occurrences(const pulseqlib_collection* coll,
                                      int subseq_idx)
{
    if (!coll) return 0;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences) return 0;
    return coll->descriptors[subseq_idx].label_num_entries;
}

static int pulseqlib__get_num_label_columns(const pulseqlib_collection* coll,
                                    int subseq_idx)
{
    if (!coll) return 0;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences) return 0;
    return coll->descriptors[subseq_idx].label_num_columns;
}

int pulseqlib_get_adc_label(const pulseqlib_collection* coll,
                            int subseq_idx,
                            int occurrence_idx,
                            int* out_values)
{
    const pulseqlib_sequence_descriptor* desc;
    int ncols, row_start, c;

    if (!coll || !out_values) return PULSEQLIB_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[subseq_idx];
    if (!desc->label_table || desc->label_num_entries == 0)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;
    if (occurrence_idx < 0 || occurrence_idx >= desc->label_num_entries)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;

    ncols     = desc->label_num_columns;
    row_start = occurrence_idx * ncols;
    for (c = 0; c < ncols; ++c)
        out_values[c] = desc->label_table[row_start + c];

    return PULSEQLIB_SUCCESS;
}

/* ================================================================== */
/*  Batch getters (public API)                                        */
/* ================================================================== */

int pulseqlib_get_collection_info(
    const pulseqlib_collection* coll,
    pulseqlib_collection_info*  info)
{
    if (!coll || !info) return PULSEQLIB_ERR_NULL_POINTER;

    info->num_subsequences = pulseqlib__get_num_subsequences(coll);
    info->num_segments     = pulseqlib__get_num_segments(coll);
    info->max_adc_samples  = pulseqlib__get_max_adc_samples(coll);
    info->total_readouts   = pulseqlib__get_total_readouts(coll);
    info->total_duration_us = pulseqlib__get_total_duration_us(coll);

    return PULSEQLIB_SUCCESS;
}

int pulseqlib_get_subseq_info(
    const pulseqlib_collection* coll,
    int                         subseq_idx,
    pulseqlib_subseq_info*      info)
{
    if (!coll || !info) return PULSEQLIB_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;

    info->tr_duration_us       = pulseqlib__get_tr_duration_us(coll, subseq_idx);
    info->num_trs              = pulseqlib__get_num_trs(coll, subseq_idx);
    info->tr_size              = pulseqlib__get_tr_size(coll, subseq_idx);
    info->num_prep_blocks      = pulseqlib__get_num_prep_blocks(coll, subseq_idx);
    info->num_cooldown_blocks  = pulseqlib__get_num_cooldown_blocks(coll, subseq_idx);
    info->num_prep_trs         = pulseqlib__get_num_prep_trs(coll, subseq_idx);
    info->num_cooldown_trs     = pulseqlib__get_num_cooldown_trs(coll, subseq_idx);
    info->degenerate_prep      = pulseqlib__get_degenerate_prep(coll, subseq_idx);
    info->degenerate_cooldown  = pulseqlib__get_degenerate_cooldown(coll, subseq_idx);
    info->num_unique_adcs      = pulseqlib__get_num_unique_adcs(coll, subseq_idx);
    info->num_unique_rf        = pulseqlib__get_num_unique_rf(coll, subseq_idx);
    info->pmc_enabled          = pulseqlib__is_pmc_enabled(coll, subseq_idx);
    info->segment_offset       = pulseqlib__get_subseq_segment_offset(coll, subseq_idx);
    info->num_prep_segments    = pulseqlib__get_num_prep_segments(coll, subseq_idx);
    info->num_main_segments    = pulseqlib__get_num_main_segments(coll, subseq_idx);
    info->num_cooldown_segments = pulseqlib__get_num_cooldown_segments(coll, subseq_idx);
    info->num_adc_occurrences  = pulseqlib__get_num_adc_occurrences(coll, subseq_idx);
    info->num_label_columns    = pulseqlib__get_num_label_columns(coll, subseq_idx);
    info->num_passes           = coll->descriptors[subseq_idx].num_passes;

    return PULSEQLIB_SUCCESS;
}

int pulseqlib_get_segment_info(
    const pulseqlib_collection* coll,
    int                         seg_idx,
    pulseqlib_segment_info*     info)
{
    if (!coll || !info) return PULSEQLIB_ERR_NULL_POINTER;

    info->duration_us          = pulseqlib__get_segment_duration_us(coll, seg_idx);
    info->num_blocks           = pulseqlib__get_segment_num_blocks(coll, seg_idx);
    info->start_block          = pulseqlib__get_segment_start_block(coll, seg_idx);
    info->pure_delay           = pulseqlib__is_segment_pure_delay(coll, seg_idx);
    info->has_trigger          = pulseqlib__segment_has_trigger(coll, seg_idx);
    info->trigger_delay_us     = pulseqlib__get_segment_trigger_delay_us(coll, seg_idx);
    info->trigger_duration_us  = pulseqlib__get_segment_trigger_duration_us(coll, seg_idx);
    info->is_nav               = pulseqlib__segment_is_nav(coll, seg_idx);
    info->num_kzero_crossings  = pulseqlib__get_segment_num_kzero_crossings(coll, seg_idx);
    info->rf_adc_gap_us        = pulseqlib__get_segment_rf_adc_gap_us(coll, seg_idx);
    info->adc_adc_gap_us       = pulseqlib__get_segment_adc_adc_gap_us(coll, seg_idx);

    return PULSEQLIB_SUCCESS;
}

int pulseqlib_get_block_info(
    const pulseqlib_collection* coll,
    int                         seg_idx,
    int                         blk_idx,
    pulseqlib_block_info*       info)
{
    int axis;

    if (!coll || !info) return PULSEQLIB_ERR_NULL_POINTER;

    info->duration_us   = pulseqlib__get_block_duration_us(coll, seg_idx, blk_idx);
    info->start_time_us = pulseqlib__get_block_start_time_us(coll, seg_idx, blk_idx);

    /* Gradient (per axis) */
    for (axis = 0; axis < 3; ++axis) {
        info->has_grad[axis]           = pulseqlib__block_has_grad(coll, seg_idx, blk_idx, axis);
        info->grad_is_trapezoid[axis]  = info->has_grad[axis] ?
            pulseqlib__block_grad_is_trapezoid(coll, seg_idx, blk_idx, axis) : 0;
        info->grad_delay_us[axis]      = info->has_grad[axis] ?
            pulseqlib__get_grad_delay_us(coll, seg_idx, blk_idx, axis) : -1;
        info->grad_num_shots[axis]     = info->has_grad[axis] ?
            pulseqlib__get_grad_num_shots(coll, seg_idx, blk_idx, axis) : -1;
        info->grad_num_samples[axis]   = info->has_grad[axis] ?
            pulseqlib__get_grad_num_samples(coll, seg_idx, blk_idx, axis) : -1;
    }

    /* RF */
    info->has_rf            = pulseqlib__block_has_rf(coll, seg_idx, blk_idx);
    info->rf_delay_us       = info->has_rf ?
        pulseqlib__get_rf_delay_us(coll, seg_idx, blk_idx) : -1;
    info->rf_num_channels   = info->has_rf ?
        pulseqlib__get_rf_num_channels(coll, seg_idx, blk_idx) : -1;
    info->rf_num_samples    = info->has_rf ?
        pulseqlib__get_rf_num_samples(coll, seg_idx, blk_idx) : -1;
    info->rf_is_complex     = info->has_rf ?
        pulseqlib__block_rf_is_complex(coll, seg_idx, blk_idx) : 0;
    info->rf_uniform_raster = info->has_rf ?
        pulseqlib__block_rf_has_uniform_raster(coll, seg_idx, blk_idx) : 0;

    /* ADC */
    info->has_adc     = pulseqlib__block_has_adc(coll, seg_idx, blk_idx);
    info->adc_delay_us = info->has_adc ?
        pulseqlib__get_adc_delay_us(coll, seg_idx, blk_idx) : -1;
    info->adc_def_id  = info->has_adc ?
        pulseqlib__get_adc_library_index(coll, seg_idx, blk_idx) : -1;

    /* Digital output */
    info->has_digitalout      = pulseqlib__block_has_digitalout(coll, seg_idx, blk_idx);
    info->digitalout_delay_us = info->has_digitalout ?
        pulseqlib__get_digitalout_delay_us(coll, seg_idx, blk_idx) : -1;
    info->digitalout_duration_us = info->has_digitalout ?
        pulseqlib__get_digitalout_duration_us(coll, seg_idx, blk_idx) : -1;

    /* Flags */
    info->has_rotation = pulseqlib__block_has_rotation(coll, seg_idx, blk_idx);
    info->norot_flag   = pulseqlib__block_has_norot(coll, seg_idx, blk_idx);
    info->nopos_flag   = pulseqlib__block_has_nopos(coll, seg_idx, blk_idx);
    info->has_freq_mod = pulseqlib__block_has_freq_mod(coll, seg_idx, blk_idx);

    return PULSEQLIB_SUCCESS;
}

int pulseqlib_get_adc_def(
    const pulseqlib_collection* coll,
    int                         adc_idx,
    pulseqlib_adc_def*          def)
{
    if (!coll || !def) return PULSEQLIB_ERR_NULL_POINTER;

    def->dwell_ns    = pulseqlib__get_adc_dwell_ns(coll, adc_idx);
    def->num_samples = pulseqlib__get_adc_num_samples(coll, adc_idx);

    return PULSEQLIB_SUCCESS;
}

/* ================================================================== */
/*  Unique-block and segment-block getters                            */
/* ================================================================== */

int pulseqlib_get_num_unique_blocks(
    const pulseqlib_collection* coll,
    int                         subseq_idx)
{
    if (!coll) return PULSEQLIB_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEQLIB_ERR_INDEX;
    return coll->descriptors[subseq_idx].num_unique_blocks;
}

int pulseqlib_get_unique_block_id(
    const pulseqlib_collection* coll,
    int                         subseq_idx,
    int                         blk_def_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    if (!coll) return PULSEQLIB_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEQLIB_ERR_INDEX;
    desc = &coll->descriptors[subseq_idx];
    if (blk_def_idx < 0 || blk_def_idx >= desc->num_unique_blocks)
        return PULSEQLIB_ERR_INDEX;
    /* block_definitions[].id is the 1-based .seq block index */
    return desc->block_definitions[blk_def_idx].id;
}

int pulseqlib_get_segment_block_def_indices(
    const pulseqlib_collection* coll,
    int                         seg_idx,
    int*                        out_ids)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_seg, i;

    if (!coll || !out_ids) return PULSEQLIB_ERR_NULL_POINTER;
    if (!pulseqlib__resolve_segment(&desc, &local_seg, coll, seg_idx))
        return PULSEQLIB_ERR_INDEX;

    seg = &desc->segment_definitions[local_seg];
    for (i = 0; i < seg->num_blocks; ++i)
        out_ids[i] = seg->unique_block_indices[i];
    return seg->num_blocks;
}