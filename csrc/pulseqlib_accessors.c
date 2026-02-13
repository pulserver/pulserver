/* pulseqlib_accessors.c -- collection-level accessor functions
 *
 * Public functions:
 *   pulseqlib_get_max_adc_samples    pulseqlib_get_adc_dwell
 *   pulseqlib_get_adc_num_samples    pulseqlib_get_num_segments
 *   pulseqlib_is_segment_pure_delay  pulseqlib_get_segment_num_blocks
 *   pulseqlib_get_block_start_time   pulseqlib_get_block_duration
 *   pulseqlib_block_has_rf           pulseqlib_block_rf_has_uniform_raster
 *   pulseqlib_block_rf_is_complex    pulseqlib_get_rf_num_samples
 *   pulseqlib_get_rf_delay           pulseqlib_get_rf_magnitude
 *   pulseqlib_get_rf_phase           pulseqlib_get_rf_time
 *   pulseqlib_block_has_grad         pulseqlib_block_grad_is_trapezoid
 *   pulseqlib_get_grad_num_samples   pulseqlib_get_grad_num_shots
 *   pulseqlib_get_grad_delay         pulseqlib_get_grad_amplitude
 *   pulseqlib_get_grad_initial_amplitude
 *   pulseqlib_get_grad_initial_shot_id
 *   pulseqlib_get_grad_time
 *   pulseqlib_block_has_adc          pulseqlib_get_adc_delay
 *   pulseqlib_get_adc_library_index
 *   pulseqlib_block_has_trigger      pulseqlib_get_trigger_delay
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
    const pulseqlib_sequence_descriptor_collection* coll,
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
    const pulseqlib_sequence_descriptor_collection* coll,
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
/*  ADC collection accessors                                          */
/* ================================================================== */

int pulseqlib_get_max_adc_samples(
    const pulseqlib_sequence_descriptor_collection* coll)
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

int pulseqlib_get_adc_dwell(
    const pulseqlib_sequence_descriptor_collection* coll, int adc_idx)
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
    const pulseqlib_sequence_descriptor_collection* coll, int adc_idx)
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
    const pulseqlib_sequence_descriptor_collection* coll)
{
    if (!coll) return 0;
    return coll->total_unique_segments;
}

int pulseqlib_is_segment_pure_delay(
    const pulseqlib_sequence_descriptor_collection* coll, int seg_idx)
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
    const pulseqlib_sequence_descriptor_collection* coll, int seg_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    int local_seg;

    if (!pulseqlib__resolve_segment(&desc, &local_seg, coll, seg_idx))
        return -1;

    return desc->segment_definitions[local_seg].num_blocks;
}

/* ================================================================== */
/*  Block-level queries                                               */
/* ================================================================== */

int pulseqlib_get_block_start_time(
    const pulseqlib_sequence_descriptor_collection* coll,
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

int pulseqlib_get_block_duration(
    const pulseqlib_sequence_descriptor_collection* coll,
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
    const pulseqlib_sequence_descriptor_collection* coll,
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
    const pulseqlib_sequence_descriptor_collection* coll,
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
    const pulseqlib_sequence_descriptor_collection* coll,
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
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, shape_idx;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_rf_definition* rdef;
    const pulseqlib_shape_arbitrary* shape;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    bdef = &desc->block_definitions[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1) return -1;

    rdef = &desc->rf_definitions[bdef->rf_id];

    /* try mag, then phase, then time shape */
    if (rdef->mag_shape_id > 0) {
        shape_idx = rdef->mag_shape_id - 1;
        if (shape_idx >= 0 && shape_idx < desc->num_shapes) {
            shape = &desc->shapes[shape_idx];
            if (shape->num_uncompressed_samples > 0)
                return shape->num_uncompressed_samples;
        }
    }
    if (rdef->phase_shape_id > 0) {
        shape_idx = rdef->phase_shape_id - 1;
        if (shape_idx >= 0 && shape_idx < desc->num_shapes) {
            shape = &desc->shapes[shape_idx];
            if (shape->num_uncompressed_samples > 0)
                return shape->num_uncompressed_samples;
        }
    }
    if (rdef->time_shape_id > 0) {
        shape_idx = rdef->time_shape_id - 1;
        if (shape_idx >= 0 && shape_idx < desc->num_shapes) {
            shape = &desc->shapes[shape_idx];
            if (shape->num_uncompressed_samples > 0)
                return shape->num_uncompressed_samples;
        }
    }
    return -1;
}

int pulseqlib_get_rf_delay(
    const pulseqlib_sequence_descriptor_collection* coll,
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

float* pulseqlib_get_rf_magnitude(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx, int* num_samples)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, shape_idx;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_rf_definition* rdef;
    pulseqlib_shape_arbitrary decompressed;

    if (!num_samples) return NULL;
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
                                     rdef->max_amplitude))
        return NULL;
#else
    if (!pulseqlib__decompress_shape(&decompressed, &desc->shapes[shape_idx], 1.0f))
        return NULL;
#endif

    *num_samples = decompressed.num_samples;
    return decompressed.samples;
}

float* pulseqlib_get_rf_phase(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx, int* num_samples)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, shape_idx;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_rf_definition* rdef;
    pulseqlib_shape_arbitrary decompressed;

    if (!num_samples) return NULL;
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

    *num_samples = decompressed.num_samples;
    return decompressed.samples;
}

float* pulseqlib_get_rf_time(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx, int* num_samples)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, shape_idx;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_rf_definition* rdef;
    pulseqlib_shape_arbitrary decompressed;

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
                                     desc->rf_raster_time_us))
        return NULL;

    *num_samples = decompressed.num_samples;
    return decompressed.samples;
}

/* ================================================================== */
/*  Gradient queries                                                  */
/* ================================================================== */

int pulseqlib_block_has_grad(
    const pulseqlib_sequence_descriptor_collection* coll,
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
    const pulseqlib_sequence_descriptor_collection* coll,
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
    const pulseqlib_sequence_descriptor_collection* coll,
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
    const pulseqlib_sequence_descriptor_collection* coll,
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

int pulseqlib_get_grad_delay(
    const pulseqlib_sequence_descriptor_collection* coll,
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
    const pulseqlib_sequence_descriptor_collection* coll,
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

float pulseqlib_get_grad_initial_amplitude(
    const pulseqlib_sequence_descriptor_collection* coll,
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
    const pulseqlib_sequence_descriptor_collection* coll,
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

float* pulseqlib_get_grad_time(
    const pulseqlib_sequence_descriptor_collection* coll,
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
                                     desc->grad_raster_time_us))
        return NULL;

    *num_samples = decompressed.num_samples;
    return decompressed.samples;
}

/* ================================================================== */
/*  ADC block queries                                                 */
/* ================================================================== */

int pulseqlib_block_has_adc(
    const pulseqlib_sequence_descriptor_collection* coll,
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

int pulseqlib_get_adc_delay(
    const pulseqlib_sequence_descriptor_collection* coll,
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
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk, adc_id, k, global_adc_idx, i;
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
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    return seg->has_trigger[local_blk];
}

int pulseqlib_get_trigger_delay(
    const pulseqlib_sequence_descriptor_collection* coll,
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
    const pulseqlib_sequence_descriptor_collection* coll,
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
    const pulseqlib_sequence_descriptor_collection* coll,
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
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_segment* seg;
    int local_blk;

    if (!pulseqlib__resolve_block(&desc, &seg, &local_blk, coll, seg_idx, blk_idx))
        return -1;

    return seg->nopos_flag[local_blk];
}