/* pulseqlib_accessors.c -- collection-level accessor functions
 *
 * Public functions:
 *   pulseqlib_get_max_adc_samples    pulseqlib_get_adc_dwell_us
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
 *   pulseqlib_block_has_trigger      pulseqlib_get_trigger_delay_us
 *   pulseqlib_block_has_rotation
 *   pulseqlib_block_has_norot        pulseqlib_block_has_nopos
 *
 * Internal (pulseqlib__) functions:
 *   pulseqlib__resolve_segment       pulseqlib__resolve_block
 */

#include <string.h>

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

/* ================================================================== */
/*  Subsequence accessors                                             */
/* ================================================================== */

int pulseqlib_get_num_subsequences(
    const pulseqlib_collection* coll)
{
    if (!coll) return 0;
    return coll->num_subsequences;
}

float pulseqlib_get_tr_duration_us(
    const pulseqlib_collection* coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0.0f;
    return coll->descriptors[subseq_idx].tr_descriptor.tr_duration_us;
}

int pulseqlib_get_num_trs(
    const pulseqlib_collection* coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].tr_descriptor.num_trs;
}

int pulseqlib_get_tr_size(
    const pulseqlib_collection* coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].tr_descriptor.tr_size;
}

int pulseqlib_get_num_unique_adcs(
    const pulseqlib_collection* coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].num_unique_adcs;
}

float pulseqlib_get_total_duration_us(
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
    int i, j;

    if (!coll || !info) return PULSEQLIB_ERR_NULL_POINTER;
    if (num_reps < 1) return PULSEQLIB_ERR_INVALID_ARGUMENT;
    if (coll->num_subsequences <= 0) return PULSEQLIB_ERR_COLLECTION_EMPTY;

    info->total_duration_us        = 0.0f;
    info->total_segment_boundaries = 0;

    for (i = 0; i < coll->num_subsequences; ++i) {
        const pulseqlib_sequence_descriptor* desc = &coll->descriptors[i];
        const pulseqlib_tr_descriptor*       trd  = &desc->tr_descriptor;
        const pulseqlib_segment_table_result* stab = &desc->segment_table;

        float prep_dur    = 0.0f;
        float cooldown_dur = 0.0f;
        int   cooldown_blk_start;
        int   N;     /* total TR count including degenerate extras */
        int   navg;
        int   prep_segs, cool_segs, main_segs;

        /* --- sum prep block durations (0 when degenerate) --- */
        for (j = 0; j < trd->num_prep_blocks; ++j)
            prep_dur += (float)desc->block_definitions[
                desc->block_table[j].id].duration_us;

        /* --- sum cooldown block durations (0 when degenerate) --- */
        cooldown_blk_start = desc->num_blocks - trd->num_cooldown_blocks;
        for (j = 0; j < trd->num_cooldown_blocks; ++j)
            cooldown_dur += (float)desc->block_definitions[
                desc->block_table[cooldown_blk_start + j].id].duration_us;

        /* --- total TR count ----------------------------------- */
        N = trd->num_trs;
        if (desc->num_prep_blocks > 0 && trd->degenerate_prep)
            N += trd->num_prep_trs;
        if (desc->num_cooldown_blocks > 0 && trd->degenerate_cooldown)
            N += trd->num_cooldown_trs;

        /* --- averages ----------------------------------------- */
        navg = desc->ignore_averages ? 1 : num_reps;

        /* --- duration ----------------------------------------- */
        if (N >= 2) {
            info->total_duration_us +=
                (prep_dur + trd->tr_duration_us)
                + (float)navg * (float)(N - 2) * trd->tr_duration_us
                + (cooldown_dur + trd->tr_duration_us);
        } else if (N == 1) {
            info->total_duration_us +=
                trd->tr_duration_us + prep_dur + cooldown_dur;
        }
        /* N == 0: no contribution */

        /* --- segment boundaries ------------------------------- */
        main_segs = stab->num_main_segments;
        prep_segs = (trd->degenerate_prep)
                        ? main_segs : stab->num_prep_segments;
        cool_segs = (trd->degenerate_cooldown)
                        ? main_segs : stab->num_cooldown_segments;

        if (N >= 2) {
            info->total_segment_boundaries +=
                prep_segs + navg * (N - 2) * main_segs + cool_segs;
        } else if (N == 1) {
            info->total_segment_boundaries += main_segs;
        }
    }

    return PULSEQLIB_OK;
}

/* ================================================================== */
/*  RF accessors                                                      */
/* ================================================================== */

int pulseqlib_get_num_unique_rf(
    const pulseqlib_collection* coll,
    int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].num_unique_rfs;
}

#if PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_GEHC
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
    return PULSEQLIB_OK;
}

float pulseqlib_get_rf_base_amplitude_hz(
    const pulseqlib_collection* coll,
    int subseq_idx, int rf_idx)
{
    const pulseqlib_sequence_descriptor* desc;

    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0.0f;
    desc = &coll->descriptors[subseq_idx];
    if (rf_idx < 0 || rf_idx >= desc->num_unique_rfs)
        return 0.0f;

    return desc->rf_definitions[rf_idx].stats.max_amplitude_hz;
}
#endif

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
/*  ADC collection accessors                                          */
/* ================================================================== */

int pulseqlib_get_max_adc_samples(
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

int pulseqlib_get_adc_dwell_us(
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

int pulseqlib_get_adc_num_samples(
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
/*  Segment accessors                                                 */
/* ================================================================== */

int pulseqlib_get_num_segments(
    const pulseqlib_collection* coll)
{
    if (!coll) return 0;
    return coll->total_unique_segments;
}

int pulseqlib_get_segment_duration_us(
    const pulseqlib_collection* coll, int seg_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    int local_seg, k, total;
    const pulseqlib_tr_segment* seg;

    if (!pulseqlib__resolve_segment(&desc, &local_seg, coll, seg_idx))
        return -1;

    seg = &desc->segment_definitions[local_seg];
    total = 0;
    for (k = 0; k < seg->num_blocks; ++k)
        total += desc->block_definitions[seg->unique_block_indices[k]].duration_us;

    return total;
}

int pulseqlib_is_segment_pure_delay(
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
            bdef->gy_id == -1 && bdef->gz_id == -1)
            return 1;
    }
    return 0;
}

int pulseqlib_get_segment_num_blocks(
    const pulseqlib_collection* coll, int seg_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    int local_seg;

    if (!pulseqlib__resolve_segment(&desc, &local_seg, coll, seg_idx))
        return -1;

    return desc->segment_definitions[local_seg].num_blocks;
}

/* ================================================================== */
/*  Segment timing queries                                            */
/* ================================================================== */

int pulseqlib_get_segment_num_kzero_crossings(
    const pulseqlib_collection* coll, int seg_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    int local_seg;

    if (!pulseqlib__resolve_segment(&desc, &local_seg, coll, seg_idx))
        return 0;

    return desc->segment_definitions[local_seg].timing.num_kzero_crossings;
}

/* ================================================================== */
/*  Block-level queries                                               */
/* ================================================================== */

int pulseqlib_get_block_start_time_us(
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

int pulseqlib_get_block_duration_us(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    return desc->block_definitions[seg->unique_block_indices[local_blk]].duration_us;
}

/* ================================================================== */
/*  RF queries                                                        */
/* ================================================================== */

int pulseqlib_block_has_rf(
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

int pulseqlib_block_rf_has_uniform_raster(
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

int pulseqlib_block_rf_is_complex(
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

int pulseqlib_get_rf_num_samples(
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

int pulseqlib_get_rf_delay_us(
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

int pulseqlib_get_rf_num_channels(
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

#if PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_GEHC
    if (!pulseqlib__decompress_shape(&decompressed, &desc->shapes[shape_idx],
                                     rdef->stats.max_amplitude_hz))
        return NULL;
#else
    if (!pulseqlib__decompress_shape(&decompressed, &desc->shapes[shape_idx], 1.0f))
        return NULL;
#endif

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
    int seg_idx, int blk_idx, int* num_samples)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, shape_idx, nch, npts;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_rf_definition* rdef;
    pulseqlib_shape_arbitrary decompressed;
    float* result;

    if (!num_samples) return NULL;
    *num_samples = 0;

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

    *num_samples = npts;
    return result;
}

/* ================================================================== */
/*  Gradient queries                                                  */
/* ================================================================== */

int pulseqlib_block_has_grad(
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

int pulseqlib_block_grad_is_trapezoid(
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

int pulseqlib_get_grad_num_samples(
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

int pulseqlib_get_grad_num_shots(
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

int pulseqlib_get_grad_delay_us(
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
    int* num_shots, int** num_samples_per_shot)
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

    if (!num_shots || !num_samples_per_shot) {
        if (num_shots) *num_shots = 0;
        if (num_samples_per_shot) *num_samples_per_shot = NULL;
        return NULL;
    }
    *num_shots = 0;
    *num_samples_per_shot = NULL;

    if (axis < PULSEQLIB_GRAD_AXIS_X || axis > PULSEQLIB_GRAD_AXIS_Z) return NULL;
    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return NULL;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    grad_id = get_grad_id_by_axis(bdef, axis);
    if (grad_id == -1) return NULL;

    gdef = &desc->grad_definitions[grad_id];

    *num_samples_per_shot = (int*)PULSEQLIB_ALLOC(gdef->num_shots * sizeof(int));
    if (!*num_samples_per_shot) return NULL;

    waveforms = (float**)PULSEQLIB_ALLOC(gdef->num_shots * sizeof(float*));
    if (!waveforms) {
        PULSEQLIB_FREE(*num_samples_per_shot);
        *num_samples_per_shot = NULL;
        return NULL;
    }

    *num_shots = gdef->num_shots;

    if (gdef->type == 0) {
        flat_time = gdef->flat_time_or_unused;
        samples_per_shot = (flat_time > 0) ? 4 : 3;

        for (shot = 0; shot < gdef->num_shots; ++shot) {
            trap_waveform = (float*)PULSEQLIB_ALLOC(samples_per_shot * sizeof(float));
            if (!trap_waveform) {
                for (k = 0; k < shot; ++k) PULSEQLIB_FREE(waveforms[k]);
                PULSEQLIB_FREE(waveforms);
                PULSEQLIB_FREE(*num_samples_per_shot);
                *num_samples_per_shot = NULL;
                *num_shots = 0;
                return NULL;
            }

            trap_waveform[0] = 0.0f;
            trap_waveform[1] = gdef->max_amplitude[shot];
            if (flat_time > 0) {
                trap_waveform[2] = gdef->max_amplitude[shot];
                trap_waveform[3] = 0.0f;
            } else {
                trap_waveform[2] = 0.0f;
            }

            waveforms[shot] = trap_waveform;
            (*num_samples_per_shot)[shot] = samples_per_shot;
        }
    } else {
        for (shot = 0; shot < gdef->num_shots; ++shot) {
            if (gdef->shot_shape_ids[shot] <= 0) {
                waveforms[shot] = NULL;
                (*num_samples_per_shot)[shot] = 0;
                continue;
            }

            shape_idx = gdef->shot_shape_ids[shot] - 1;
            if (shape_idx < 0 || shape_idx >= desc->num_shapes) {
                waveforms[shot] = NULL;
                (*num_samples_per_shot)[shot] = 0;
                continue;
            }

            decompressed.num_samples = 0;
            decompressed.num_uncompressed_samples = 0;
            decompressed.samples = NULL;

            if (!pulseqlib__decompress_shape(&decompressed, &desc->shapes[shape_idx],
                                             gdef->max_amplitude[shot])) {
                waveforms[shot] = NULL;
                (*num_samples_per_shot)[shot] = 0;
                continue;
            }

            waveforms[shot] = decompressed.samples;
            (*num_samples_per_shot)[shot] = decompressed.num_samples;
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

float* pulseqlib_get_grad_time_us(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx, int axis, int* num_samples)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, grad_id, shape_idx;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_grad_definition* gdef;
    float* time_waveform;
    float accum;
    int rise_time, flat_time, fall_time;
    pulseqlib_shape_arbitrary decompressed;

    if (!num_samples) return NULL;
    *num_samples = 0;

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

        *num_samples = (flat_time > 0) ? 4 : 3;

        time_waveform = (float*)PULSEQLIB_ALLOC((*num_samples) * sizeof(float));
        if (!time_waveform) { *num_samples = 0; return NULL; }

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

    *num_samples = decompressed.num_samples;
    return decompressed.samples;
}

/* ================================================================== */
/*  ADC block queries                                                 */
/* ================================================================== */

int pulseqlib_block_has_adc(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_block_table_element* bte;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    bte  = &desc->block_table[bdef->id];

    return (bte->adc_id != -1) ? 1 : 0;
}

int pulseqlib_get_adc_delay_us(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, adc_id;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_block_table_element* bte;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    bte  = &desc->block_table[bdef->id];

    if (bte->adc_id == -1) return -1;

    adc_id = bte->adc_id;
    if (adc_id < 0 || adc_id >= desc->num_unique_adcs) return -1;

    return desc->adc_definitions[adc_id].delay;
}

int pulseqlib_get_adc_library_index(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, adc_id, global_adc_idx, i;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_block_table_element* bte;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    bte  = &desc->block_table[bdef->id];

    if (bte->adc_id == -1) return -1;

    adc_id = bte->adc_id;
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
/*  Trigger / rotation / flag queries                                 */
/* ================================================================== */

int pulseqlib_block_has_trigger(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    return seg->has_trigger[local_blk];
}

int pulseqlib_get_trigger_delay_us(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, trigger_id;
    const pulseqlib_block_table_element* bte;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    if (!seg->has_trigger[local_blk]) return -1;

    bte = &desc->block_table[seg->start_block + local_blk];
    trigger_id = bte->trigger_id;
    if (trigger_id == -1 || trigger_id >= desc->num_triggers) return -1;

    return (int)desc->trigger_events[trigger_id].delay;
}

int pulseqlib_block_has_rotation(
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

int pulseqlib_block_has_norot(
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

int pulseqlib_block_has_nopos(
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

int pulseqlib_cursor_next(pulseqlib_collection* coll)
{
    pulseqlib_block_cursor* cursor;
    const pulseqlib_sequence_descriptor* desc;
    int imaging_start;
    int cooldown_start;
    int next_idx;

    cursor = &coll->block_cursor;

    if (cursor->sequence_index >= coll->num_subsequences)
        return PULSEQLIB_CURSOR_DONE;

    desc = &coll->descriptors[cursor->sequence_index];
    imaging_start  = desc->num_prep_blocks;
    cooldown_start = desc->num_blocks - desc->num_cooldown_blocks;

    next_idx = cursor->within_sequence_block_index + 1;

    /* Hit cooldown boundary on non-last rep: wrap to imaging start */
    if (next_idx == cooldown_start &&
        cursor->current_repetition < coll->num_repetitions - 1) {
        cursor->current_repetition += 1;
        cursor->within_sequence_block_index = imaging_start;
        cursor->from_last_reset = 0;
        return PULSEQLIB_CURSOR_BLOCK;
    }

    /* Past end of sequence: advance to next subsequence */
    if (next_idx >= desc->num_blocks) {
        cursor->sequence_index += 1;
        cursor->current_repetition = 0;
        cursor->within_sequence_block_index = 0;
        cursor->from_last_reset = 0;
        if (cursor->sequence_index >= coll->num_subsequences)
            return PULSEQLIB_CURSOR_DONE;
        return PULSEQLIB_CURSOR_BLOCK;
    }

    /* Normal advance */
    cursor->within_sequence_block_index = next_idx;
    cursor->from_last_reset += 1;
    return PULSEQLIB_CURSOR_BLOCK;
}

void pulseqlib_cursor_reset(pulseqlib_collection* coll)
{
    pulseqlib_block_cursor* cursor;

    cursor = &coll->block_cursor;

    /* Go back by the number of blocks advanced since the last reset */
    cursor->within_sequence_block_index -= cursor->from_last_reset;
    cursor->from_last_reset = 0;
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
    idx  = cursor->within_sequence_block_index;
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

    /* Trigger */
    inst->trigon_flag = (bte->trigger_id >= 0) ? 1 : 0;

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

    return PULSEQLIB_OK;
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
        *out_start = desc->num_blocks - desc->num_cooldown_blocks;
        *out_count = desc->num_cooldown_blocks;
        return 1;
    case PULSEQLIB_TR_REGION_ALL:
        *out_start = 0;
        *out_count = desc->num_blocks;
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

/* --- compute waveforms & phase offsets for an existing plan --- */
static void compute_fm_waveforms(
    pulseqlib_freq_mod_plan* plan,
    const pulseqlib_sequence_descriptor* desc,
    const float* shift)
{
    int n, inst_idx, fmid, ch, s;
    const pulseqlib_block_table_element* bte;
    const pulseqlib_freq_mod_definition* fm;
    float amp[3], scaled[3], act_shift[3];
    const float* R;
    float identity[9];
    int max_s;

    max_s = plan->max_samples;

    /* Zero the backing store */
    memset(plan->_waveform_data, 0,
           (size_t)plan->num_instances * (size_t)max_s * sizeof(float));

    /* Identity matrix for blocks without rotation */
    identity[0] = 1.0f; identity[1] = 0.0f; identity[2] = 0.0f;
    identity[3] = 0.0f; identity[4] = 1.0f; identity[5] = 0.0f;
    identity[6] = 0.0f; identity[7] = 0.0f; identity[8] = 1.0f;

    inst_idx = 0;
    for (n = 0; n < desc->num_blocks; ++n) {
        if (plan->block_to_instance[n] < 0) continue;

        bte = &desc->block_table[n];
        fmid = bte->freq_mod_id;
        fm = &desc->freq_mod_definitions[fmid];

        /* Gradient amplitudes (Hz/m) for each axis */
        amp[0] = (bte->gx_id >= 0 && bte->gx_id < desc->grad_table_size)
                 ? desc->grad_table[bte->gx_id].amplitude : 0.0f;
        amp[1] = (bte->gy_id >= 0 && bte->gy_id < desc->grad_table_size)
                 ? desc->grad_table[bte->gy_id].amplitude : 0.0f;
        amp[2] = (bte->gz_id >= 0 && bte->gz_id < desc->grad_table_size)
                 ? desc->grad_table[bte->gz_id].amplitude : 0.0f;

        /* scaled = amplitude .* shift  (Hz) */
        scaled[0] = amp[0] * shift[0];
        scaled[1] = amp[1] * shift[1];
        scaled[2] = amp[2] * shift[2];

        /* Undo rotation: act_shift = R^T @ scaled */
        if (bte->rotation_id >= 0 && bte->rotation_id < desc->num_rotations
            && !bte->norot_flag)
            R = desc->rotation_matrices[bte->rotation_id];
        else
            R = identity;
        pulseqlib__apply_rotation(act_shift, R, scaled, 1); /* transpose=1 */

        /* freq_mod waveform: sum_ch(base_ch[s] * act_shift[ch]) */
        {
            float* row = plan->waveforms[inst_idx];
            for (s = 0; s < fm->num_samples; ++s) {
                row[s] = 0.0f;
                if (fm->waveform_gx) row[s] += fm->waveform_gx[s] * act_shift[0];
                if (fm->waveform_gy) row[s] += fm->waveform_gy[s] * act_shift[1];
                if (fm->waveform_gz) row[s] += fm->waveform_gz[s] * act_shift[2];
            }
        }

        /* phase_offset: sum_ch(ref_integral[ch] * act_shift[ch])  [rad] */
        plan->phase_offset[inst_idx] = 0.0f;
        for (ch = 0; ch < 3; ++ch)
            plan->phase_offset[inst_idx] += fm->ref_integral[ch] * act_shift[ch];

        ++inst_idx;
    }
}

/* --- allocate plan + build block_to_instance map + compute waveforms --- */
static int build_fm_plan_range(
    pulseqlib_freq_mod_plan* plan,
    const pulseqlib_sequence_descriptor* desc,
    const float* shift,
    int blk_start, int blk_count)
{
    int n, fmid;
    int max_s, count, inst_idx;

    memset(plan, 0, sizeof(*plan));

    /* ---------- Pass 1: count instances & find max_samples ---------- */
    count = 0;
    max_s = 0;
    for (n = blk_start; n < blk_start + blk_count && n < desc->num_blocks; ++n) {
        fmid = desc->block_table[n].freq_mod_id;
        if (fmid < 0 || fmid >= desc->num_freq_mod_defs) continue;
        ++count;
        if (desc->freq_mod_definitions[fmid].num_samples > max_s)
            max_s = desc->freq_mod_definitions[fmid].num_samples;
    }

    if (count == 0) return PULSEQLIB_OK;

    plan->num_instances = count;
    plan->max_samples   = max_s;
    plan->num_blocks    = desc->num_blocks;
    plan->raster_us     = desc->freq_mod_definitions[0].raster_us;
    plan->_desc         = (const void*)desc;

    /* Allocate output arrays */
    plan->_waveform_data = (float*)PULSEQLIB_ALLOC(
        (size_t)count * (size_t)max_s * sizeof(float));
    plan->waveforms    = (float**)PULSEQLIB_ALLOC(
        (size_t)count * sizeof(float*));
    plan->num_samples  = (int*)PULSEQLIB_ALLOC((size_t)count * sizeof(int));
    plan->phase_offset = (float*)PULSEQLIB_ALLOC((size_t)count * sizeof(float));
    plan->block_to_instance = (int*)PULSEQLIB_ALLOC(
        (size_t)desc->num_blocks * sizeof(int));
    if (!plan->_waveform_data || !plan->waveforms ||
        !plan->num_samples || !plan->phase_offset ||
        !plan->block_to_instance) {
        pulseqlib_freq_mod_plan_free(plan);
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }

    /* Set up row pointers into the flat backing store */
    {
        int r;
        for (r = 0; r < count; ++r)
            plan->waveforms[r] = plan->_waveform_data + (size_t)r * (size_t)max_s;
    }

    /* Build block_to_instance: all -1, then fill for freq_mod blocks */
    {
        int k;
        for (k = 0; k < desc->num_blocks; ++k)
            plan->block_to_instance[k] = -1;
    }
    inst_idx = 0;
    for (n = blk_start; n < blk_start + blk_count && n < desc->num_blocks; ++n) {
        fmid = desc->block_table[n].freq_mod_id;
        if (fmid < 0 || fmid >= desc->num_freq_mod_defs) continue;
        plan->block_to_instance[n] = inst_idx;
        plan->num_samples[inst_idx] = desc->freq_mod_definitions[fmid].num_samples;
        ++inst_idx;
    }

    /* Compute waveforms and phase offsets */
    compute_fm_waveforms(plan, desc, shift);

    return PULSEQLIB_OK;
}

int pulseqlib_build_freq_mod_plan(
    pulseqlib_freq_mod_plan** plan,
    const pulseqlib_collection* coll,
    const float* shift_m,
    int tr_type, int tr_index)
{
    int blk_start, blk_count, result;
    const pulseqlib_sequence_descriptor* desc;
    pulseqlib_freq_mod_plan* p;

    if (!plan || !coll || !shift_m)
        return PULSEQLIB_ERR_NULL_POINTER;
    *plan = NULL;
    if (coll->num_subsequences < 1)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[0];
    if (!get_block_range(desc, tr_type, tr_index, &blk_start, &blk_count))
        return PULSEQLIB_ERR_INVALID_ARGUMENT;

    p = (pulseqlib_freq_mod_plan*)PULSEQLIB_ALLOC(sizeof(*p));
    if (!p) return PULSEQLIB_ERR_ALLOC_FAILED;
    memset(p, 0, sizeof(*p));

    result = build_fm_plan_range(p, desc, shift_m, blk_start, blk_count);
    if (PULSEQLIB_FAILED(result)) {
        pulseqlib_freq_mod_plan_free(p);
        PULSEQLIB_FREE(p);
        return result;
    }
    *plan = p;
    return PULSEQLIB_OK;
}

int pulseqlib_update_freq_mod_plan(
    pulseqlib_freq_mod_plan* plan,
    const float* shift_m)
{
    const pulseqlib_sequence_descriptor* desc;

    if (!plan || !shift_m)
        return PULSEQLIB_ERR_NULL_POINTER;
    if (plan->num_instances == 0 || !plan->_desc)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;

    desc = (const pulseqlib_sequence_descriptor*)plan->_desc;
    compute_fm_waveforms(plan, desc, shift_m);
    return PULSEQLIB_OK;
}

void pulseqlib_freq_mod_plan_free(pulseqlib_freq_mod_plan* plan)
{
    if (!plan) return;
    if (plan->_waveform_data)    { PULSEQLIB_FREE(plan->_waveform_data);    plan->_waveform_data = NULL; }
    if (plan->waveforms)         { PULSEQLIB_FREE(plan->waveforms);         plan->waveforms = NULL; }
    if (plan->num_samples)       { PULSEQLIB_FREE(plan->num_samples);       plan->num_samples = NULL; }
    if (plan->phase_offset)      { PULSEQLIB_FREE(plan->phase_offset);      plan->phase_offset = NULL; }
    if (plan->block_to_instance) { PULSEQLIB_FREE(plan->block_to_instance); plan->block_to_instance = NULL; }
    plan->num_instances = 0;
    plan->max_samples   = 0;
    plan->num_blocks    = 0;
    plan->_desc         = NULL;
    PULSEQLIB_FREE(plan);
}

int pulseqlib_get_freq_mod_waveform(
    const pulseqlib_freq_mod_plan* plan,
    int block_idx,
    const float** out_waveform,
    int* out_num_samples,
    float* out_phase_rad)
{
    int inst;
    if (!plan || !out_waveform || !out_num_samples || !out_phase_rad)
        return 0;
    if (block_idx < 0 || block_idx >= plan->num_blocks)
        return 0;
    inst = plan->block_to_instance[block_idx];
    if (inst < 0)
        return 0;
    *out_waveform    = plan->waveforms[inst];
    *out_num_samples = plan->num_samples[inst];
    *out_phase_rad   = plan->phase_offset[inst];
    return 1;
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
    return PULSEQLIB_OK;
}

int pulseqlib_get_num_adc_occurrences(const pulseqlib_collection* coll,
                                      int subseq_idx)
{
    if (!coll) return 0;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences) return 0;
    return coll->descriptors[subseq_idx].label_num_entries;
}

int pulseqlib_get_num_label_columns(const pulseqlib_collection* coll,
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

    return PULSEQLIB_OK;
}