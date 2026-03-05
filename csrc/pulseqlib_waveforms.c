/* pulseqlib_waveforms.c -- gradient, RF, ADC waveform extraction
 *
 * Extracts native-timing and uniform-raster gradient waveforms,
 * RF magnitude/phase, and ADC events from the block playback table.
 *
 * Public functions:
 *   pulseqlib_get_tr_gradient_waveforms / _free
 *   pulseqlib_get_tr_waveforms          / _free
 *
 * Internal (called by pulseqlib_safety.c):
 *   pulseqlib__uniform_grad_waveforms_free
 *   pulseqlib__get_gradient_waveforms_range
 *   pulseqlib__find_unique_shot_trs
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "pulseqlib_internal.h"
#include "pulseqlib_methods.h"

/*  Gradient waveform free                                            */
/* ================================================================== */

void pulseqlib_tr_gradient_waveforms_free(pulseqlib_tr_gradient_waveforms* w)
{
    if (!w) return;
    if (w->gx.time_us)            PULSEQLIB_FREE(w->gx.time_us);
    if (w->gx.amplitude_hz_per_m) PULSEQLIB_FREE(w->gx.amplitude_hz_per_m);
    if (w->gx.seg_label)          PULSEQLIB_FREE(w->gx.seg_label);
    if (w->gy.time_us)            PULSEQLIB_FREE(w->gy.time_us);
    if (w->gy.amplitude_hz_per_m) PULSEQLIB_FREE(w->gy.amplitude_hz_per_m);
    if (w->gy.seg_label)          PULSEQLIB_FREE(w->gy.seg_label);
    if (w->gz.time_us)            PULSEQLIB_FREE(w->gz.time_us);
    if (w->gz.amplitude_hz_per_m) PULSEQLIB_FREE(w->gz.amplitude_hz_per_m);
    if (w->gz.seg_label)          PULSEQLIB_FREE(w->gz.seg_label);
    memset(w, 0, sizeof(*w));
}

/* ================================================================== */
/*  Internal uniform gradient waveform free                           */
/* ================================================================== */

void pulseqlib__uniform_grad_waveforms_free(pulseqlib__uniform_grad_waveforms* w)
{
    if (!w) return;
    if (w->gx) PULSEQLIB_FREE(w->gx);
    if (w->gy) PULSEQLIB_FREE(w->gy);
    if (w->gz) PULSEQLIB_FREE(w->gz);
    memset(w, 0, sizeof(*w));
}

/* ================================================================== */
/*  Gradient sample counting                                           */
/* ================================================================== */

static int count_grad_samples_for_block(
    const pulseqlib_sequence_descriptor* desc,
    const pulseqlib_grad_definition* gdef,
    float block_duration_us)
{
    int count;
    int num_samples;
    float delay_us, rise_us, flat_us, fall_us, duration_us;
    float grad_raster_us;
    pulseqlib_shape_arbitrary decomp_time;

    if (!gdef) return 2;

    count = 0;
    decomp_time.samples = NULL;
    decomp_time.num_uncompressed_samples = 0;

    grad_raster_us  = desc->grad_raster_us;
    num_samples     = gdef->fall_time_or_num_uncompressed_samples;
    delay_us        = (float)gdef->delay;

    if (delay_us > 0.0f) count++;

    if (gdef->type == 0) {
        rise_us = (float)gdef->rise_time_or_unused;
        flat_us = (float)gdef->flat_time_or_unused;
        fall_us = (float)gdef->fall_time_or_num_uncompressed_samples;
        duration_us = delay_us + rise_us + flat_us + fall_us;
        count += (flat_us > 0) ? 4 : 3;
    } else {
        if (gdef->unused_or_time_shape_id > 0 &&
            gdef->unused_or_time_shape_id <= desc->num_shapes &&
            pulseqlib__decompress_shape(&decomp_time,
                &desc->shapes[gdef->unused_or_time_shape_id - 1],
                grad_raster_us)) {
            duration_us = delay_us +
                decomp_time.samples[decomp_time.num_uncompressed_samples - 1];
        } else {
            duration_us = delay_us + 0.5f * grad_raster_us +
                          grad_raster_us * (float)(num_samples - 1);
        }
        if (decomp_time.samples) PULSEQLIB_FREE(decomp_time.samples);
        count += num_samples;
    }

    if (duration_us < block_duration_us) count++;
    return count;
}

/* ================================================================== */
/*  Position-specific max amplitudes (filtered by TR group)           */
/* ================================================================== */

/*
 * Computes per-position worst-case |amplitude| for each shot index,
 * considering only TR instances whose group label matches target_group.
 * If tr_group_labels is NULL, all TRs are included (unfiltered).
 *
 * Output arrays must be pre-allocated to tr_size * PULSEQLIB_MAX_GRAD_SHOTS.
 */
static int compute_position_max_amplitudes_filtered(
    const pulseqlib_sequence_descriptor* desc,
    float* pos_max_gx, float* pos_max_gy, float* pos_max_gz,
    const int* tr_group_labels, int target_group)
{
    const pulseqlib_tr_descriptor* tr;
    int tr_start, tr_size, num_trs;
    int tr_idx, pos, block_idx;
    const pulseqlib_block_table_element* bte;
    const pulseqlib_grad_table_element* gte;
    float abs_amp;
    int raw_id, shot_idx, arr_idx, n;

    tr       = &desc->tr_descriptor;
    tr_size  = tr->tr_size;
    num_trs  = tr->num_trs;

    for (n = 0; n < tr_size * PULSEQLIB_MAX_GRAD_SHOTS; ++n) {
        pos_max_gx[n] = 0.0f;
        pos_max_gy[n] = 0.0f;
        pos_max_gz[n] = 0.0f;
    }

    for (tr_idx = tr->imaging_tr_start / tr_size;
         tr_idx < tr->imaging_tr_start / tr_size + num_trs;
         ++tr_idx) {
        /* skip TRs not in the target group */
        if (tr_group_labels && tr_group_labels[tr_idx] != target_group)
            continue;

        tr_start = tr->num_prep_blocks + tr_idx * tr_size;
        for (pos = 0; pos < tr_size; ++pos) {
            block_idx = tr_start + pos;
            bte = &desc->block_table[block_idx];

            /* Gx */
            raw_id = bte->gx_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size) {
                gte = &desc->grad_table[raw_id];
                shot_idx = gte->shot_index;
                if (shot_idx >= 0 && shot_idx < PULSEQLIB_MAX_GRAD_SHOTS) {
                    abs_amp = gte->amplitude;
                    if (abs_amp < 0.0f) abs_amp = -abs_amp;
                    arr_idx = pos * PULSEQLIB_MAX_GRAD_SHOTS + shot_idx;
                    if (abs_amp > pos_max_gx[arr_idx])
                        pos_max_gx[arr_idx] = abs_amp;
                }
            }

            /* Gy */
            raw_id = bte->gy_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size) {
                gte = &desc->grad_table[raw_id];
                shot_idx = gte->shot_index;
                if (shot_idx >= 0 && shot_idx < PULSEQLIB_MAX_GRAD_SHOTS) {
                    abs_amp = gte->amplitude;
                    if (abs_amp < 0.0f) abs_amp = -abs_amp;
                    arr_idx = pos * PULSEQLIB_MAX_GRAD_SHOTS + shot_idx;
                    if (abs_amp > pos_max_gy[arr_idx])
                        pos_max_gy[arr_idx] = abs_amp;
                }
            }

            /* Gz */
            raw_id = bte->gz_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size) {
                gte = &desc->grad_table[raw_id];
                shot_idx = gte->shot_index;
                if (shot_idx >= 0 && shot_idx < PULSEQLIB_MAX_GRAD_SHOTS) {
                    abs_amp = gte->amplitude;
                    if (abs_amp < 0.0f) abs_amp = -abs_amp;
                    arr_idx = pos * PULSEQLIB_MAX_GRAD_SHOTS + shot_idx;
                    if (abs_amp > pos_max_gz[arr_idx])
                        pos_max_gz[arr_idx] = abs_amp;
                }
            }
        }
    }
    return PULSEQLIB_SUCCESS;
}

/* ================================================================== */
/*  Compute per-position variable-gradient flags (ZERO_VAR mode)      */
/* ================================================================== */

/**
 * For each (position, axis) within the canonical TR, determine whether
 * the gradient amplitude varies across TR instances.  A flag of 1 means
 * variable (will be zeroed in ZERO_VAR mode); 0 means constant (keeps
 * its actual amplitude).
 *
 * The test is simple: record the amplitude from the first TR instance
 * that has a non-null gradient at (pos, axis), then check all subsequent
 * TRs.  If any differ, set the flag.
 */
int pulseqlib__compute_variable_grad_flags(pulseqlib_sequence_descriptor* desc)
{
    const pulseqlib_tr_descriptor* tr;
    int tr_size, num_trs, n, pos, block_idx, raw_id, tr_idx;
    const pulseqlib_block_table_element* bte;
    const pulseqlib_grad_table_element* gte;
    float first_amp[3];  /* per axis: amplitude from first TR */
    int   seen[3];       /* per axis: 1 if first TR recorded */

    if (!desc) return PULSEQLIB_ERR_NULL_POINTER;

    tr      = &desc->tr_descriptor;
    tr_size = tr->tr_size;
    num_trs = tr->num_trs;

    /* free any prior allocation */
    if (desc->variable_grad_flags) {
        PULSEQLIB_FREE(desc->variable_grad_flags);
        desc->variable_grad_flags = NULL;
    }

    if (tr_size <= 0 || num_trs <= 0)
        return PULSEQLIB_SUCCESS;

    n = tr_size * 3;
    desc->variable_grad_flags = (int*)PULSEQLIB_ALLOC((size_t)n * sizeof(int));
    if (!desc->variable_grad_flags)
        return PULSEQLIB_ERR_ALLOC_FAILED;
    for (pos = 0; pos < n; ++pos)
        desc->variable_grad_flags[pos] = 0;

    for (pos = 0; pos < tr_size; ++pos) {
        first_amp[0] = 0.0f; first_amp[1] = 0.0f; first_amp[2] = 0.0f;
        seen[0] = 0; seen[1] = 0; seen[2] = 0;

        for (tr_idx = 0; tr_idx < num_trs; ++tr_idx) {
            block_idx = tr->num_prep_blocks + tr_idx * tr_size + pos;
            if (block_idx < 0 || block_idx >= desc->num_blocks) continue;
            bte = &desc->block_table[block_idx];

            /* axis 0 = gx, 1 = gy, 2 = gz */
            {
                int axis;
                int raw_ids[3];
                raw_ids[0] = bte->gx_id;
                raw_ids[1] = bte->gy_id;
                raw_ids[2] = bte->gz_id;

                for (axis = 0; axis < 3; ++axis) {
                    raw_id = raw_ids[axis];
                    if (raw_id >= 0 && raw_id < desc->grad_table_size) {
                        gte = &desc->grad_table[raw_id];
                        if (!seen[axis]) {
                            first_amp[axis] = gte->amplitude;
                            seen[axis] = 1;
                        } else {
                            if (gte->amplitude != first_amp[axis]) {
                                desc->variable_grad_flags[pos * 3 + axis] = 1;
                            }
                        }
                    }
                }
            }
        }
    }
    return PULSEQLIB_SUCCESS;
}

/* ================================================================== */
/*  Find unique shot-index TR variants                                */
/* ================================================================== */

/*
 * For multi-shot sequences, different TR instances may use different
 * shot indices, producing different waveform shapes.  This function
 * identifies the unique shot-index fingerprints across TR instances
 * and returns:
 *   - return value: number of unique patterns (0 on failure)
 *   - *out_unique_tr_indices: representative TR index per group
 *   - *out_tr_group_labels:  group label (0..num_unique-1) per TR
 *
 * Caller must free *out_unique_tr_indices and *out_tr_group_labels.
 */
int pulseqlib__find_unique_shot_trs(
    const pulseqlib_sequence_descriptor* desc,
    int** out_unique_tr_indices,
    int** out_tr_group_labels)
{
    const pulseqlib_tr_descriptor* tr;
    int tr_size, num_trs, num_cols;
    int* int_rows;
    int* unique_defs;
    int* event_table;
    int* result_indices;
    int num_unique;
    int tr_idx, pos, col, block_idx, raw_id;
    const pulseqlib_block_table_element* bte;
    const pulseqlib_grad_table_element* gte;

    *out_unique_tr_indices = NULL;
    *out_tr_group_labels   = NULL;

    tr      = &desc->tr_descriptor;
    tr_size = tr->tr_size;
    num_trs = tr->num_trs;

    if (num_trs <= 0 || tr_size <= 0) return 0;

    /* Each row: tr_size * 3 ints = (gx_shot, gy_shot, gz_shot) per position */
    num_cols = tr_size * 3;

    int_rows    = (int*)PULSEQLIB_ALLOC((size_t)num_trs * (size_t)num_cols * sizeof(int));
    unique_defs = (int*)PULSEQLIB_ALLOC((size_t)num_trs * sizeof(int));
    event_table = (int*)PULSEQLIB_ALLOC((size_t)num_trs * sizeof(int));
    if (!int_rows || !unique_defs || !event_table) {
        if (int_rows)    PULSEQLIB_FREE(int_rows);
        if (unique_defs) PULSEQLIB_FREE(unique_defs);
        if (event_table) PULSEQLIB_FREE(event_table);
        return 0;
    }

    /* Build the fingerprint matrix */
    for (tr_idx = 0; tr_idx < num_trs; ++tr_idx) {
        col = 0;
        for (pos = 0; pos < tr_size; ++pos) {
            block_idx = tr->num_prep_blocks + tr_idx * tr_size + pos;
            bte = &desc->block_table[block_idx];

            /* Gx shot index */
            raw_id = bte->gx_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size) {
                gte = &desc->grad_table[raw_id];
                int_rows[tr_idx * num_cols + col] = gte->shot_index;
            } else {
                int_rows[tr_idx * num_cols + col] = -1;
            }
            col++;

            /* Gy shot index */
            raw_id = bte->gy_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size) {
                gte = &desc->grad_table[raw_id];
                int_rows[tr_idx * num_cols + col] = gte->shot_index;
            } else {
                int_rows[tr_idx * num_cols + col] = -1;
            }
            col++;

            /* Gz shot index */
            raw_id = bte->gz_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size) {
                gte = &desc->grad_table[raw_id];
                int_rows[tr_idx * num_cols + col] = gte->shot_index;
            } else {
                int_rows[tr_idx * num_cols + col] = -1;
            }
            col++;
        }
    }

    /* Deduplicate */
    num_unique = pulseqlib__deduplicate_int_rows(
        unique_defs, event_table, int_rows, num_trs, num_cols);

    PULSEQLIB_FREE(int_rows);

    if (num_unique <= 0) {
        PULSEQLIB_FREE(unique_defs);
        PULSEQLIB_FREE(event_table);
        return 0;
    }

    /* Copy representative TR indices into right-sized array */
    result_indices = (int*)PULSEQLIB_ALLOC((size_t)num_unique * sizeof(int));
    if (!result_indices) {
        PULSEQLIB_FREE(unique_defs);
        PULSEQLIB_FREE(event_table);
        return 0;
    }
    for (tr_idx = 0; tr_idx < num_unique; ++tr_idx) {
        result_indices[tr_idx] = unique_defs[tr_idx];
    }
    PULSEQLIB_FREE(unique_defs);

    *out_unique_tr_indices = result_indices;
    *out_tr_group_labels   = event_table;
    return num_unique;
}

/* ================================================================== */
/*  Fill waveform for a single block                                  */
/* ================================================================== */

static int fill_grad_waveform_for_block(
    const pulseqlib_sequence_descriptor* desc,
    float* time, float* waveform, int start_idx,
    const pulseqlib_grad_definition* gdef,
    const pulseqlib_grad_table_element* gte,
    float t0,
    const float* pos_max_amp,
    float block_duration_us)
{
    int i, idx;
    float sign, max_amp;
    float delay_us, t_sample, last_written;
    int shape_id, time_shape_id, shot_idx, num_samples;
    float rise_us, flat_us, fall_us;
    float grad_raster_us, block_end_us;
    pulseqlib_shape_arbitrary decomp_wave, decomp_time;
    int has_time_shape;

    idx = start_idx;
    grad_raster_us = desc->grad_raster_us;
    block_end_us   = t0 + block_duration_us;
    decomp_wave.samples = NULL;
    decomp_time.samples = NULL;

    if (!gdef || !gte) {
        time[idx]     = t0;
        waveform[idx] = 0.0f;
        idx++;
        time[idx]     = block_end_us;
        waveform[idx] = 0.0f;
        idx++;
        return idx - start_idx;
    }

    last_written = t0;
    sign     = (gte->amplitude >= 0.0f) ? 1.0f : -1.0f;
    shot_idx = gte->shot_index;
    max_amp  = pos_max_amp[shot_idx];
    delay_us = (float)gdef->delay;

    if (delay_us > 0.0f) {
        t_sample = t0;
        time[idx]     = t_sample;
        waveform[idx] = 0.0f;
        last_written  = t_sample;
        idx++;
    }

    if (gdef->type == 0) {
        rise_us = (float)gdef->rise_time_or_unused;
        flat_us = (float)gdef->flat_time_or_unused;
        fall_us = (float)gdef->fall_time_or_num_uncompressed_samples;

        if (flat_us > 0) {
            t_sample = t0 + delay_us;
            time[idx] = t_sample; waveform[idx] = 0.0f;
            last_written = t_sample; idx++;

            t_sample = t0 + delay_us + rise_us;
            time[idx] = t_sample; waveform[idx] = sign * max_amp;
            last_written = t_sample; idx++;

            t_sample = t0 + delay_us + rise_us + flat_us;
            time[idx] = t_sample; waveform[idx] = sign * max_amp;
            last_written = t_sample; idx++;

            t_sample = t0 + delay_us + rise_us + flat_us + fall_us;
            time[idx] = t_sample; waveform[idx] = 0.0f;
            last_written = t_sample; idx++;
        } else {
            t_sample = t0 + delay_us;
            time[idx] = t_sample; waveform[idx] = 0.0f;
            last_written = t_sample; idx++;

            t_sample = t0 + delay_us + rise_us;
            time[idx] = t_sample; waveform[idx] = sign * max_amp;
            last_written = t_sample; idx++;

            t_sample = t0 + delay_us + rise_us + fall_us;
            time[idx] = t_sample; waveform[idx] = 0.0f;
            last_written = t_sample; idx++;
        }
    } else {
        num_samples   = gdef->fall_time_or_num_uncompressed_samples;
        time_shape_id = gdef->unused_or_time_shape_id;
        shape_id      = gdef->shot_shape_ids[shot_idx];

        if (shape_id <= 0 || shape_id > desc->num_shapes) return 0;
        if (!pulseqlib__decompress_shape(&decomp_wave,
                &desc->shapes[shape_id - 1], 1.0f))
            return 0;

        has_time_shape = 0;
        if (time_shape_id > 0 && time_shape_id <= desc->num_shapes) {
            if (pulseqlib__decompress_shape(&decomp_time,
                    &desc->shapes[time_shape_id - 1], grad_raster_us))
                has_time_shape = 1;
        }

        if (has_time_shape) {
            for (i = 0; i < num_samples; ++i) {
                t_sample = t0 + delay_us + decomp_time.samples[i];
                time[idx]     = t_sample;
                waveform[idx] = sign * max_amp * decomp_wave.samples[i];
                last_written  = t_sample;
                idx++;
            }
        } else {
            for (i = 0; i < num_samples; ++i) {
                t_sample = t0 + delay_us + 0.5f * grad_raster_us +
                           (float)i * grad_raster_us;
                time[idx]     = t_sample;
                waveform[idx] = sign * max_amp * decomp_wave.samples[i];
                last_written  = t_sample;
                idx++;
            }
        }

        if (decomp_wave.samples) PULSEQLIB_FREE(decomp_wave.samples);
        if (decomp_time.samples) PULSEQLIB_FREE(decomp_time.samples);
    }

    if (block_end_us > last_written) {
        time[idx]     = block_end_us;
        waveform[idx] = 0.0f;
        idx++;
    }

    return idx - start_idx;
}

/* ================================================================== */
/*  Interpolate to uniform raster                                     */
/* ================================================================== */

static int interpolate_to_uniform(
    float** time, float** waveform, int* num_samples,
    float target_raster_us)
{
    float* t_in;
    float* w_in;
    float* t_out = NULL;
    float* w_out = NULL;
    int n_in, n_out, i;
    float t_start, t_end, duration;

    t_out = NULL;
    w_out = NULL;

    if (!time || !waveform || !num_samples || *num_samples <= 0)
        return PULSEQLIB_SUCCESS;

    t_in = *time;
    w_in = *waveform;
    n_in = *num_samples;

    t_start  = t_in[0];
    t_end    = t_in[n_in - 1];
    duration = t_end - t_start;
    if (duration <= 0.0f) return PULSEQLIB_SUCCESS;

    n_out = (int)(duration / target_raster_us) + 1;

    t_out = (float*)PULSEQLIB_ALLOC(n_out * sizeof(float));
    w_out = (float*)PULSEQLIB_ALLOC(n_out * sizeof(float));
    if (!t_out || !w_out) {
        if (t_out) PULSEQLIB_FREE(t_out);
        if (w_out) PULSEQLIB_FREE(w_out);
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }

    for (i = 0; i < n_out; ++i)
        t_out[i] = t_start + (float)i * target_raster_us;

    pulseqlib__interp1_linear(w_out, t_out, n_out, t_in, w_in, n_in);

    PULSEQLIB_FREE(t_in);
    PULSEQLIB_FREE(w_in);

    *time        = t_out;
    *waveform    = w_out;
    *num_samples = n_out;
    return PULSEQLIB_SUCCESS;
}

/* ================================================================== */
/*  Gradient waveforms for an arbitrary block range                   */
/* ================================================================== */

/*  amplitude_mode (uses PULSEQLIB_AMP_* defines from pulseqlib_types.h):
 *    PULSEQLIB_AMP_MAX_POS  (0) = position-max (worst-case safety)
 *    PULSEQLIB_AMP_ZERO_VAR (1) = zero variable grads, keep constant (k-space)
 *    PULSEQLIB_AMP_ACTUAL   (2) = actual block amplitude (single-TR)
 */
int pulseqlib__get_gradient_waveforms_range(
    const pulseqlib_sequence_descriptor* desc,
    pulseqlib__uniform_grad_waveforms* out,
    pulseqlib_diagnostic* diag,
    int block_start,
    int block_count,
    int amplitude_mode,
    const int* tr_group_labels,
    int target_group)
{
    pulseqlib_diagnostic local_diag;
    int n, block_idx;
    int total_gx, total_gy, total_gz;
    int idx_gx, idx_gy, idx_gz;
    int num_gx, num_gy, num_gz;
    int result;
    float t0, block_dur_us, target_raster_us;
    int block_def_id;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_block_table_element* bte;
    int gx_raw, gy_raw, gz_raw;
    const pulseqlib_grad_definition* gx_def;
    const pulseqlib_grad_definition* gy_def;
    const pulseqlib_grad_definition* gz_def;
    const pulseqlib_grad_table_element* gx_tab;
    const pulseqlib_grad_table_element* gy_tab;
    const pulseqlib_grad_table_element* gz_tab;
    float* pos_max_gx;
    float* pos_max_gy;
    float* pos_max_gz;
    float actual_amp[PULSEQLIB_MAX_GRAD_SHOTS];
    int k;
    float* time_gx;
    float* time_gy;
    float* time_gz;
    float* wf_gx;
    float* wf_gy;
    float* wf_gz;

    pos_max_gx = NULL;
    pos_max_gy = NULL;
    pos_max_gz = NULL;
    time_gx = NULL;
    time_gy = NULL;
    time_gz = NULL;
    wf_gx = NULL;
    wf_gy = NULL;
    wf_gz = NULL;

    if (!diag) { pulseqlib_diagnostic_init(&local_diag); diag = &local_diag; }
    else       { pulseqlib_diagnostic_init(diag); }

    if (!desc || !out) {
        diag->code = PULSEQLIB_ERR_NULL_POINTER;
        return diag->code;
    }

    memset(out, 0, sizeof(*out));

    if (block_count <= 0) {
        diag->code = PULSEQLIB_ERR_TR_NO_BLOCKS;
        return diag->code;
    }
    if (block_start < 0 || block_start + block_count > desc->num_blocks) {
        diag->code = PULSEQLIB_ERR_INVALID_ARGUMENT;
        return diag->code;
    }

    /* position-max amplitudes (only for worst-case main-TR mode) */
    if (amplitude_mode == PULSEQLIB_AMP_MAX_POS ||
        amplitude_mode == PULSEQLIB_AMP_ZERO_VAR) {
        pos_max_gx = (float*)PULSEQLIB_ALLOC(
            (size_t)block_count * PULSEQLIB_MAX_GRAD_SHOTS * sizeof(float));
        pos_max_gy = (float*)PULSEQLIB_ALLOC(
            (size_t)block_count * PULSEQLIB_MAX_GRAD_SHOTS * sizeof(float));
        pos_max_gz = (float*)PULSEQLIB_ALLOC(
            (size_t)block_count * PULSEQLIB_MAX_GRAD_SHOTS * sizeof(float));
        if (!pos_max_gx || !pos_max_gy || !pos_max_gz) {
            if (pos_max_gx) PULSEQLIB_FREE(pos_max_gx);
            if (pos_max_gy) PULSEQLIB_FREE(pos_max_gy);
            if (pos_max_gz) PULSEQLIB_FREE(pos_max_gz);
            diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
            return diag->code;
        }
        compute_position_max_amplitudes_filtered(desc,
            pos_max_gx, pos_max_gy, pos_max_gz,
            tr_group_labels, target_group);

        /* ZERO_VAR: zero out positions whose gradients vary across TRs */
        if (amplitude_mode == PULSEQLIB_AMP_ZERO_VAR &&
            desc->variable_grad_flags) {
            int vp;
            for (vp = 0; vp < block_count && vp < desc->tr_descriptor.tr_size; ++vp) {
                if (desc->variable_grad_flags[vp * 3 + 0])
                    for (k = 0; k < PULSEQLIB_MAX_GRAD_SHOTS; ++k)
                        pos_max_gx[vp * PULSEQLIB_MAX_GRAD_SHOTS + k] = 0.0f;
                if (desc->variable_grad_flags[vp * 3 + 1])
                    for (k = 0; k < PULSEQLIB_MAX_GRAD_SHOTS; ++k)
                        pos_max_gy[vp * PULSEQLIB_MAX_GRAD_SHOTS + k] = 0.0f;
                if (desc->variable_grad_flags[vp * 3 + 2])
                    for (k = 0; k < PULSEQLIB_MAX_GRAD_SHOTS; ++k)
                        pos_max_gz[vp * PULSEQLIB_MAX_GRAD_SHOTS + k] = 0.0f;
            }
        }
    }

    /* ---- pass 1: count samples ---- */
    total_gx = 0; total_gy = 0; total_gz = 0;
    for (n = 0; n < block_count; ++n) {
        block_idx    = block_start + n;
        bte          = &desc->block_table[block_idx];
        block_def_id = bte->id;
        bdef         = &desc->block_definitions[block_def_id];
        block_dur_us = (bte->duration_us >= 0) ? (float)bte->duration_us
                                               : (float)bdef->duration_us;

        gx_raw = bte->gx_id; gy_raw = bte->gy_id; gz_raw = bte->gz_id;
        gx_def = (gx_raw >= 0 && gx_raw < desc->grad_table_size &&
                  desc->grad_table[gx_raw].id >= 0 &&
                  desc->grad_table[gx_raw].id < desc->num_unique_grads)
                 ? &desc->grad_definitions[desc->grad_table[gx_raw].id] : NULL;
        gy_def = (gy_raw >= 0 && gy_raw < desc->grad_table_size &&
                  desc->grad_table[gy_raw].id >= 0 &&
                  desc->grad_table[gy_raw].id < desc->num_unique_grads)
                 ? &desc->grad_definitions[desc->grad_table[gy_raw].id] : NULL;
        gz_def = (gz_raw >= 0 && gz_raw < desc->grad_table_size &&
                  desc->grad_table[gz_raw].id >= 0 &&
                  desc->grad_table[gz_raw].id < desc->num_unique_grads)
                 ? &desc->grad_definitions[desc->grad_table[gz_raw].id] : NULL;

        total_gx += count_grad_samples_for_block(desc, gx_def, block_dur_us);
        total_gy += count_grad_samples_for_block(desc, gy_def, block_dur_us);
        total_gz += count_grad_samples_for_block(desc, gz_def, block_dur_us);
    }

    /* ---- allocate (local time arrays + output waveform arrays) ---- */
    time_gx = (float*)PULSEQLIB_ALLOC((size_t)total_gx * sizeof(float));
    wf_gx   = (float*)PULSEQLIB_ALLOC((size_t)total_gx * sizeof(float));
    time_gy = (float*)PULSEQLIB_ALLOC((size_t)total_gy * sizeof(float));
    wf_gy   = (float*)PULSEQLIB_ALLOC((size_t)total_gy * sizeof(float));
    time_gz = (float*)PULSEQLIB_ALLOC((size_t)total_gz * sizeof(float));
    wf_gz   = (float*)PULSEQLIB_ALLOC((size_t)total_gz * sizeof(float));
    if (!time_gx || !wf_gx ||
        !time_gy || !wf_gy ||
        !time_gz || !wf_gz) {
        if (pos_max_gx) PULSEQLIB_FREE(pos_max_gx);
        if (pos_max_gy) PULSEQLIB_FREE(pos_max_gy);
        if (pos_max_gz) PULSEQLIB_FREE(pos_max_gz);
        if (time_gx) PULSEQLIB_FREE(time_gx);
        if (time_gy) PULSEQLIB_FREE(time_gy);
        if (time_gz) PULSEQLIB_FREE(time_gz);
        if (wf_gx) PULSEQLIB_FREE(wf_gx);
        if (wf_gy) PULSEQLIB_FREE(wf_gy);
        if (wf_gz) PULSEQLIB_FREE(wf_gz);
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        return diag->code;
    }

    /* ---- pass 2: fill ---- */
    t0 = 0.0f; idx_gx = 0; idx_gy = 0; idx_gz = 0;
    for (n = 0; n < block_count; ++n) {
        block_idx    = block_start + n;
        bte          = &desc->block_table[block_idx];
        block_def_id = bte->id;
        bdef         = &desc->block_definitions[block_def_id];
        block_dur_us = (bte->duration_us >= 0) ? (float)bte->duration_us
                                               : (float)bdef->duration_us;

        gx_raw = bte->gx_id; gy_raw = bte->gy_id; gz_raw = bte->gz_id;

        gx_tab = (gx_raw >= 0 && gx_raw < desc->grad_table_size)
                 ? &desc->grad_table[gx_raw] : NULL;
        gy_tab = (gy_raw >= 0 && gy_raw < desc->grad_table_size)
                 ? &desc->grad_table[gy_raw] : NULL;
        gz_tab = (gz_raw >= 0 && gz_raw < desc->grad_table_size)
                 ? &desc->grad_table[gz_raw] : NULL;

        gx_def = (gx_tab && gx_tab->id >= 0 && gx_tab->id < desc->num_unique_grads)
                 ? &desc->grad_definitions[gx_tab->id] : NULL;
        gy_def = (gy_tab && gy_tab->id >= 0 && gy_tab->id < desc->num_unique_grads)
                 ? &desc->grad_definitions[gy_tab->id] : NULL;
        gz_def = (gz_tab && gz_tab->id >= 0 && gz_tab->id < desc->num_unique_grads)
                 ? &desc->grad_definitions[gz_tab->id] : NULL;

        if (amplitude_mode == PULSEQLIB_AMP_MAX_POS ||
            amplitude_mode == PULSEQLIB_AMP_ZERO_VAR) {
            idx_gx += fill_grad_waveform_for_block(desc,
                time_gx, wf_gx, idx_gx,
                gx_def, gx_tab, t0,
                &pos_max_gx[n * PULSEQLIB_MAX_GRAD_SHOTS], block_dur_us);
            idx_gy += fill_grad_waveform_for_block(desc,
                time_gy, wf_gy, idx_gy,
                gy_def, gy_tab, t0,
                &pos_max_gy[n * PULSEQLIB_MAX_GRAD_SHOTS], block_dur_us);
            idx_gz += fill_grad_waveform_for_block(desc,
                time_gz, wf_gz, idx_gz,
                gz_def, gz_tab, t0,
                &pos_max_gz[n * PULSEQLIB_MAX_GRAD_SHOTS], block_dur_us);
        } else {
            for (k = 0; k < PULSEQLIB_MAX_GRAD_SHOTS; ++k) actual_amp[k] = 0.0f;
            if (gx_tab) {
                k = gx_tab->shot_index;
                if (k >= 0 && k < PULSEQLIB_MAX_GRAD_SHOTS) {
                    actual_amp[k] = gx_tab->amplitude;
                    if (actual_amp[k] < 0.0f) actual_amp[k] = -actual_amp[k];
                }
            }
            idx_gx += fill_grad_waveform_for_block(desc,
                time_gx, wf_gx, idx_gx,
                gx_def, gx_tab, t0, actual_amp, block_dur_us);

            for (k = 0; k < PULSEQLIB_MAX_GRAD_SHOTS; ++k) actual_amp[k] = 0.0f;
            if (gy_tab) {
                k = gy_tab->shot_index;
                if (k >= 0 && k < PULSEQLIB_MAX_GRAD_SHOTS) {
                    actual_amp[k] = gy_tab->amplitude;
                    if (actual_amp[k] < 0.0f) actual_amp[k] = -actual_amp[k];
                }
            }
            idx_gy += fill_grad_waveform_for_block(desc,
                time_gy, wf_gy, idx_gy,
                gy_def, gy_tab, t0, actual_amp, block_dur_us);

            for (k = 0; k < PULSEQLIB_MAX_GRAD_SHOTS; ++k) actual_amp[k] = 0.0f;
            if (gz_tab) {
                k = gz_tab->shot_index;
                if (k >= 0 && k < PULSEQLIB_MAX_GRAD_SHOTS) {
                    actual_amp[k] = gz_tab->amplitude;
                    if (actual_amp[k] < 0.0f) actual_amp[k] = -actual_amp[k];
                }
            }
            idx_gz += fill_grad_waveform_for_block(desc,
                time_gz, wf_gz, idx_gz,
                gz_def, gz_tab, t0, actual_amp, block_dur_us);
        }

        t0 += block_dur_us;
    }

    if (pos_max_gx) PULSEQLIB_FREE(pos_max_gx);
    if (pos_max_gy) PULSEQLIB_FREE(pos_max_gy);
    if (pos_max_gz) PULSEQLIB_FREE(pos_max_gz);

    num_gx = idx_gx;
    num_gy = idx_gy;
    num_gz = idx_gz;

    /* interpolate each axis to uniform raster (half gradient raster) */
    target_raster_us = 0.5f * desc->grad_raster_us;

    result = interpolate_to_uniform(
        &time_gx, &wf_gx,
        &num_gx, target_raster_us);
    if (PULSEQLIB_FAILED(result)) {
        if (time_gx) PULSEQLIB_FREE(time_gx);
        if (time_gy) PULSEQLIB_FREE(time_gy);
        if (time_gz) PULSEQLIB_FREE(time_gz);
        if (wf_gx) PULSEQLIB_FREE(wf_gx);
        if (wf_gy) PULSEQLIB_FREE(wf_gy);
        if (wf_gz) PULSEQLIB_FREE(wf_gz);
        diag->code = result; return result;
    }
    result = interpolate_to_uniform(
        &time_gy, &wf_gy,
        &num_gy, target_raster_us);
    if (PULSEQLIB_FAILED(result)) {
        if (time_gx) PULSEQLIB_FREE(time_gx);
        if (time_gy) PULSEQLIB_FREE(time_gy);
        if (time_gz) PULSEQLIB_FREE(time_gz);
        if (wf_gx) PULSEQLIB_FREE(wf_gx);
        if (wf_gy) PULSEQLIB_FREE(wf_gy);
        if (wf_gz) PULSEQLIB_FREE(wf_gz);
        diag->code = result; return result;
    }
    result = interpolate_to_uniform(
        &time_gz, &wf_gz,
        &num_gz, target_raster_us);
    if (PULSEQLIB_FAILED(result)) {
        if (time_gx) PULSEQLIB_FREE(time_gx);
        if (time_gy) PULSEQLIB_FREE(time_gy);
        if (time_gz) PULSEQLIB_FREE(time_gz);
        if (wf_gx) PULSEQLIB_FREE(wf_gx);
        if (wf_gy) PULSEQLIB_FREE(wf_gy);
        if (wf_gz) PULSEQLIB_FREE(wf_gz);
        diag->code = result; return result;
    }

    /* Post-interpolation: all axes share the same uniform raster. */
    out->gx = wf_gx;
    out->gy = wf_gy;
    out->gz = wf_gz;
    out->num_samples = num_gx;
    out->raster_us = target_raster_us;
    PULSEQLIB_FREE(time_gx);
    PULSEQLIB_FREE(time_gy);
    PULSEQLIB_FREE(time_gz);

    diag->code = PULSEQLIB_SUCCESS;
    return PULSEQLIB_SUCCESS;
}

/* ================================================================== */
/*  get_tr_gradient_waveforms                                         */
/* ================================================================== */

int pulseqlib_get_tr_gradient_waveforms(
    const pulseqlib_collection* coll,
    int subseq_idx,
    pulseqlib_tr_gradient_waveforms* waveforms,
    pulseqlib_diagnostic* diag)
{
    const pulseqlib_sequence_descriptor* desc;
    pulseqlib__uniform_grad_waveforms uw;
    int rc, i;
    float* time_arr;

    memset(&uw, 0, sizeof(uw));
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences) {
        if (diag) { pulseqlib_diagnostic_init(diag); diag->code = PULSEQLIB_ERR_INVALID_ARGUMENT; }
        return PULSEQLIB_ERR_INVALID_ARGUMENT;
    }
    desc = &coll->descriptors[subseq_idx];
    rc = pulseqlib__get_gradient_waveforms_range(desc, &uw, diag,
        desc->tr_descriptor.num_prep_blocks
            + desc->tr_descriptor.imaging_tr_start,
        desc->tr_descriptor.tr_size,
        PULSEQLIB_AMP_MAX_POS, NULL, 0);
    if (PULSEQLIB_FAILED(rc)) return rc;
    if (!waveforms) { pulseqlib__uniform_grad_waveforms_free(&uw); return PULSEQLIB_ERR_NULL_POINTER; }
    memset(waveforms, 0, sizeof(*waveforms));
    /* build common time array */
    time_arr = (float*)PULSEQLIB_ALLOC((size_t)uw.num_samples * sizeof(float));
    if (!time_arr) { pulseqlib__uniform_grad_waveforms_free(&uw); return PULSEQLIB_ERR_ALLOC_FAILED; }
    for (i = 0; i < uw.num_samples; ++i) time_arr[i] = (float)i * uw.raster_us;
    /* gx */
    waveforms->gx.num_samples = uw.num_samples;
    waveforms->gx.amplitude_hz_per_m = uw.gx; uw.gx = NULL;
    waveforms->gx.time_us = time_arr;
    waveforms->gx.seg_label = NULL;
    /* gy */
    time_arr = (float*)PULSEQLIB_ALLOC((size_t)uw.num_samples * sizeof(float));
    if (!time_arr) { pulseqlib__uniform_grad_waveforms_free(&uw); pulseqlib_tr_gradient_waveforms_free(waveforms); return PULSEQLIB_ERR_ALLOC_FAILED; }
    for (i = 0; i < uw.num_samples; ++i) time_arr[i] = (float)i * uw.raster_us;
    waveforms->gy.num_samples = uw.num_samples;
    waveforms->gy.amplitude_hz_per_m = uw.gy; uw.gy = NULL;
    waveforms->gy.time_us = time_arr;
    waveforms->gy.seg_label = NULL;
    /* gz */
    time_arr = (float*)PULSEQLIB_ALLOC((size_t)uw.num_samples * sizeof(float));
    if (!time_arr) { pulseqlib__uniform_grad_waveforms_free(&uw); pulseqlib_tr_gradient_waveforms_free(waveforms); return PULSEQLIB_ERR_ALLOC_FAILED; }
    for (i = 0; i < uw.num_samples; ++i) time_arr[i] = (float)i * uw.raster_us;
    waveforms->gz.num_samples = uw.num_samples;
    waveforms->gz.amplitude_hz_per_m = uw.gz; uw.gz = NULL;
    waveforms->gz.time_us = time_arr;
    waveforms->gz.seg_label = NULL;
    return PULSEQLIB_SUCCESS;
}

/* ================================================================== */
/*  Native-timing TR waveforms (for plotting)                        */
/* ================================================================== */

/*
 * Count RF samples for a single block (flat block index).
 * Returns 0 if block has no RF.
 */
static int count_rf_samples_for_flat_block(
    const pulseqlib_sequence_descriptor* desc,
    int block_idx)
{
    const pulseqlib_block_table_element* bte;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_rf_definition* rdef;
    const pulseqlib_shape_arbitrary* shape;
    int shape_idx, nch;

    bte = &desc->block_table[block_idx];
    bdef = &desc->block_definitions[bte->id];
    if (bdef->rf_id < 0) return 0;
    rdef = &desc->rf_definitions[bdef->rf_id];
    if (rdef->mag_shape_id <= 0) return 0;
    shape_idx = rdef->mag_shape_id - 1;
    if (shape_idx < 0 || shape_idx >= desc->num_shapes) return 0;
    shape = &desc->shapes[shape_idx];
    nch = (rdef->num_channels > 1) ? rdef->num_channels : 1;
    return shape->num_uncompressed_samples / nch;
}

/*
 * Fill RF waveform for a single block (flat block index).
 * Writes into time_mag[], mag[], phase[] at start_idx.
 * Returns number of samples written.
 */
static int fill_rf_waveform_for_flat_block(
    const pulseqlib_sequence_descriptor* desc,
    int block_idx,
    float* time_mag, float* mag, float* phase,
    int start_idx, float t0)
{
    const pulseqlib_block_table_element* bte;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_rf_definition* rdef;
    const pulseqlib_rf_table_element* rtab;
    pulseqlib_shape_arbitrary decomp_mag, decomp_phase, decomp_time;
    float rf_raster_us, delay_us, amp;
    int nch, npts, i, idx;
    int has_time_shape;

    bte = &desc->block_table[block_idx];
    bdef = &desc->block_definitions[bte->id];
    if (bdef->rf_id < 0) return 0;
    rdef = &desc->rf_definitions[bdef->rf_id];
    if (rdef->mag_shape_id <= 0) return 0;

    /* instance-level params */
    if (bte->rf_id < 0 || bte->rf_id >= desc->rf_table_size) return 0;
    rtab = &desc->rf_table[bte->rf_id];
    amp  = rtab->amplitude;         /* Hz */

    rf_raster_us = desc->rf_raster_us;
    delay_us     = (float)rdef->delay;
    nch = (rdef->num_channels > 1) ? rdef->num_channels : 1;

    /* decompress magnitude shape */
    decomp_mag.samples = NULL;
    decomp_mag.num_uncompressed_samples = 0;
    if (!pulseqlib__decompress_shape(&decomp_mag,
            &desc->shapes[rdef->mag_shape_id - 1], 1.0f))
        return 0;
    npts = decomp_mag.num_uncompressed_samples / nch;

    /* decompress phase shape */
    decomp_phase.samples = NULL;
    decomp_phase.num_uncompressed_samples = 0;
    if (rdef->phase_shape_id > 0 &&
        rdef->phase_shape_id <= desc->num_shapes) {
        pulseqlib__decompress_shape(&decomp_phase,
            &desc->shapes[rdef->phase_shape_id - 1], 1.0f);
    }

    /* decompress time shape */
    decomp_time.samples = NULL;
    decomp_time.num_uncompressed_samples = 0;
    has_time_shape = 0;
    if (rdef->time_shape_id > 0 &&
        rdef->time_shape_id <= desc->num_shapes) {
        if (pulseqlib__decompress_shape(&decomp_time,
                &desc->shapes[rdef->time_shape_id - 1], rf_raster_us))
            has_time_shape = 1;
    }

    /* fill arrays (channel 0 only) */
    idx = start_idx;
    for (i = 0; i < npts; ++i) {
        float t_sample;
        if (has_time_shape && i < decomp_time.num_uncompressed_samples)
            t_sample = t0 + delay_us + decomp_time.samples[i];
        else
            t_sample = t0 + delay_us + 0.5f * rf_raster_us
                       + (float)i * rf_raster_us;

        time_mag[idx] = t_sample;
        mag[idx]      = amp * decomp_mag.samples[i];    /* Hz */
        phase[idx]    = (decomp_phase.samples && i < decomp_phase.num_uncompressed_samples)
                        ? decomp_phase.samples[i] + rtab->phase_offset
                        : rtab->phase_offset;           /* rad */
        idx++;
    }

    if (decomp_mag.samples)   PULSEQLIB_FREE(decomp_mag.samples);
    if (decomp_phase.samples) PULSEQLIB_FREE(decomp_phase.samples);
    if (decomp_time.samples)  PULSEQLIB_FREE(decomp_time.samples);
    return idx - start_idx;
}

/*
 * Find which segment a block at position pos_in_tr belongs to.
 * Returns segment index (into segment_definitions), or -1.
 */
static int find_segment_for_block_pos(
    const pulseqlib_tr_segment* segments,
    int num_segments,
    int pos_in_tr)
{
    int s;
    for (s = 0; s < num_segments; ++s) {
        if (pos_in_tr >= segments[s].start_block &&
            pos_in_tr <  segments[s].start_block + segments[s].num_blocks)
            return s;
    }
    return -1;
}

/* ---- public: get_tr_waveforms ---- */
int pulseqlib_get_tr_waveforms(
    const pulseqlib_collection* coll,
    int subseq_idx,
    int amplitude_mode,
    int tr_index,
    int include_prep,
    int include_cooldown,
    int collapse_delays,
    pulseqlib_tr_waveforms* out,
    pulseqlib_diagnostic* diag)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_descriptor* tr;
    pulseqlib_diagnostic local_diag;
    int block_start, block_count, tr_block_start;
    int n, block_idx, k;
    int total_gx, total_gy, total_gz, total_rf, total_adc;
    int idx_gx, idx_gy, idx_gz, idx_rf, idx_adc;
    float t0, block_dur_us;
    const pulseqlib_block_table_element* bte;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_grad_definition* gx_def;
    const pulseqlib_grad_definition* gy_def;
    const pulseqlib_grad_definition* gz_def;
    const pulseqlib_grad_table_element* gx_tab;
    const pulseqlib_grad_table_element* gy_tab;
    const pulseqlib_grad_table_element* gz_tab;
    int gx_raw, gy_raw, gz_raw;
    float* pos_max_gx;
    float* pos_max_gy;
    float* pos_max_gz;
    float actual_amp[PULSEQLIB_MAX_GRAD_SHOTS];

    pos_max_gx = NULL;
    pos_max_gy = NULL;
    pos_max_gz = NULL;

    if (!diag) { pulseqlib_diagnostic_init(&local_diag); diag = &local_diag; }
    else       { pulseqlib_diagnostic_init(diag); }

    if (!coll || !out) {
        diag->code = PULSEQLIB_ERR_NULL_POINTER;
        return diag->code;
    }
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences) {
        diag->code = PULSEQLIB_ERR_INVALID_ARGUMENT;
        return diag->code;
    }
    memset(out, 0, sizeof(*out));

    desc = &coll->descriptors[subseq_idx];
    tr   = &desc->tr_descriptor;

    if (tr->tr_size <= 0) {
        diag->code = PULSEQLIB_ERR_TR_NO_BLOCKS;
        return diag->code;
    }

    /* ---- determine block range ---- */
    if (amplitude_mode == PULSEQLIB_AMP_ACTUAL) {
        if (tr_index < 0 || tr_index >= tr->num_trs) {
            diag->code = PULSEQLIB_ERR_INVALID_ARGUMENT;
            return diag->code;
        }
        tr_block_start = tr->num_prep_blocks + tr_index * tr->tr_size;
    } else {
        /* canonical main TR (first imaging instance) */
        tr_block_start = tr->num_prep_blocks + tr->imaging_tr_start;
    }

    block_start = tr_block_start;
    block_count = tr->tr_size;

    if (include_prep && (amplitude_mode != PULSEQLIB_AMP_ACTUAL || tr_index == 0)) {
        block_start = 0;
        block_count = tr_block_start + tr->tr_size - block_start;
    }
    if (include_cooldown &&
        (amplitude_mode != PULSEQLIB_AMP_ACTUAL || tr_index == tr->num_trs - 1)) {
        block_count = (desc->num_blocks - desc->num_cooldown_blocks
                       + desc->num_cooldown_blocks) - block_start;
        /* simpler: */
        block_count = desc->num_blocks - block_start;
    }

    if (block_start < 0 || block_start + block_count > desc->num_blocks) {
        diag->code = PULSEQLIB_ERR_INVALID_ARGUMENT;
        return diag->code;
    }

    /* ---- precompute position-max if needed ---- */
    if (amplitude_mode == PULSEQLIB_AMP_MAX_POS ||
        amplitude_mode == PULSEQLIB_AMP_ZERO_VAR) {
        pos_max_gx = (float*)PULSEQLIB_ALLOC(
            (size_t)tr->tr_size * PULSEQLIB_MAX_GRAD_SHOTS * sizeof(float));
        pos_max_gy = (float*)PULSEQLIB_ALLOC(
            (size_t)tr->tr_size * PULSEQLIB_MAX_GRAD_SHOTS * sizeof(float));
        pos_max_gz = (float*)PULSEQLIB_ALLOC(
            (size_t)tr->tr_size * PULSEQLIB_MAX_GRAD_SHOTS * sizeof(float));
        if (!pos_max_gx || !pos_max_gy || !pos_max_gz) goto alloc_fail;
        compute_position_max_amplitudes_filtered(desc,
            pos_max_gx, pos_max_gy, pos_max_gz, NULL, 0);

        /* ZERO_VAR: zero out positions whose gradients vary across TRs */
        if (amplitude_mode == PULSEQLIB_AMP_ZERO_VAR &&
            desc->variable_grad_flags) {
            int vp;
            for (vp = 0; vp < tr->tr_size; ++vp) {
                if (desc->variable_grad_flags[vp * 3 + 0])
                    for (k = 0; k < PULSEQLIB_MAX_GRAD_SHOTS; ++k)
                        pos_max_gx[vp * PULSEQLIB_MAX_GRAD_SHOTS + k] = 0.0f;
                if (desc->variable_grad_flags[vp * 3 + 1])
                    for (k = 0; k < PULSEQLIB_MAX_GRAD_SHOTS; ++k)
                        pos_max_gy[vp * PULSEQLIB_MAX_GRAD_SHOTS + k] = 0.0f;
                if (desc->variable_grad_flags[vp * 3 + 2])
                    for (k = 0; k < PULSEQLIB_MAX_GRAD_SHOTS; ++k)
                        pos_max_gz[vp * PULSEQLIB_MAX_GRAD_SHOTS + k] = 0.0f;
            }
        }
    }

    /* ---- PASS 1: count samples ---- */
    total_gx = 0; total_gy = 0; total_gz = 0; total_rf = 0; total_adc = 0;
    for (n = 0; n < block_count; ++n) {
        block_idx = block_start + n;
        bte  = &desc->block_table[block_idx];
        bdef = &desc->block_definitions[bte->id];
        block_dur_us = (bte->duration_us >= 0) ? (float)bte->duration_us
                                               : (float)bdef->duration_us;

        gx_raw = bte->gx_id; gy_raw = bte->gy_id; gz_raw = bte->gz_id;
        gx_def = (gx_raw >= 0 && gx_raw < desc->grad_table_size &&
                  desc->grad_table[gx_raw].id >= 0 &&
                  desc->grad_table[gx_raw].id < desc->num_unique_grads)
                 ? &desc->grad_definitions[desc->grad_table[gx_raw].id] : NULL;
        gy_def = (gy_raw >= 0 && gy_raw < desc->grad_table_size &&
                  desc->grad_table[gy_raw].id >= 0 &&
                  desc->grad_table[gy_raw].id < desc->num_unique_grads)
                 ? &desc->grad_definitions[desc->grad_table[gy_raw].id] : NULL;
        gz_def = (gz_raw >= 0 && gz_raw < desc->grad_table_size &&
                  desc->grad_table[gz_raw].id >= 0 &&
                  desc->grad_table[gz_raw].id < desc->num_unique_grads)
                 ? &desc->grad_definitions[desc->grad_table[gz_raw].id] : NULL;

        total_gx += count_grad_samples_for_block(desc, gx_def, block_dur_us);
        total_gy += count_grad_samples_for_block(desc, gy_def, block_dur_us);
        total_gz += count_grad_samples_for_block(desc, gz_def, block_dur_us);
        total_rf += count_rf_samples_for_flat_block(desc, block_idx);

        if (amplitude_mode == PULSEQLIB_AMP_ACTUAL) {
            if (bte->adc_id >= 0 && bte->adc_id < desc->adc_table_size)
                total_adc++;
        } else {
            if (bdef->adc_id >= 0 && bdef->adc_id < desc->num_unique_adcs)
                total_adc++;
        }
    }

    /* ---- allocate ---- */
    out->gx.time_us    = (float*)PULSEQLIB_ALLOC((size_t)total_gx * sizeof(float));
    out->gx.amplitude  = (float*)PULSEQLIB_ALLOC((size_t)total_gx * sizeof(float));
    out->gy.time_us    = (float*)PULSEQLIB_ALLOC((size_t)total_gy * sizeof(float));
    out->gy.amplitude  = (float*)PULSEQLIB_ALLOC((size_t)total_gy * sizeof(float));
    out->gz.time_us    = (float*)PULSEQLIB_ALLOC((size_t)total_gz * sizeof(float));
    out->gz.amplitude  = (float*)PULSEQLIB_ALLOC((size_t)total_gz * sizeof(float));

    if (total_rf > 0) {
        out->rf_mag.time_us   = (float*)PULSEQLIB_ALLOC((size_t)total_rf * sizeof(float));
        out->rf_mag.amplitude = (float*)PULSEQLIB_ALLOC((size_t)total_rf * sizeof(float));
        out->rf_phase.time_us = (float*)PULSEQLIB_ALLOC((size_t)total_rf * sizeof(float));
        out->rf_phase.amplitude = (float*)PULSEQLIB_ALLOC((size_t)total_rf * sizeof(float));
    }
    if (total_adc > 0) {
        out->adc_events = (pulseqlib_adc_event*)PULSEQLIB_ALLOC(
            (size_t)total_adc * sizeof(pulseqlib_adc_event));
    }
    out->blocks = (pulseqlib_tr_block_descriptor*)PULSEQLIB_ALLOC(
        (size_t)block_count * sizeof(pulseqlib_tr_block_descriptor));

    /* check allocations (simplified: check critical ones) */
    if (!out->gx.time_us || !out->gx.amplitude ||
        !out->gy.time_us || !out->gy.amplitude ||
        !out->gz.time_us || !out->gz.amplitude ||
        !out->blocks ||
        (total_rf > 0 && (!out->rf_mag.time_us || !out->rf_mag.amplitude ||
                          !out->rf_phase.time_us || !out->rf_phase.amplitude)) ||
        (total_adc > 0 && !out->adc_events))
        goto alloc_fail;

    /* ---- PASS 2: fill ---- */
    t0 = 0.0f;
    idx_gx = 0; idx_gy = 0; idx_gz = 0; idx_rf = 0; idx_adc = 0;

    for (n = 0; n < block_count; ++n) {
        int pos_in_tr;
        block_idx = block_start + n;
        bte  = &desc->block_table[block_idx];
        bdef = &desc->block_definitions[bte->id];
        block_dur_us = (bte->duration_us >= 0) ? (float)bte->duration_us
                                               : (float)bdef->duration_us;

        /* Pure-delay shrinkage: if collapse_delays is set and this block
         * has no RF, no gradients, and no ADC, clamp its display duration
         * to a short dummy value (100 µs = 0.1 ms). */
        if (collapse_delays &&
            bdef->rf_id < 0 &&
            bte->gx_id < 0 && bte->gy_id < 0 && bte->gz_id < 0 &&
            bdef->adc_id < 0 &&
            block_dur_us > 100.0f)
        {
            block_dur_us = 100.0f;  /* 0.1 ms dummy delay */
        }

        /* block metadata */
        out->blocks[n].start_us    = t0;
        out->blocks[n].duration_us = block_dur_us;

        /* segment assignment: only for blocks within the main TR range */
        pos_in_tr = block_idx - tr_block_start;
        if (pos_in_tr >= 0 && pos_in_tr < tr->tr_size) {
            out->blocks[n].segment_idx = find_segment_for_block_pos(
                desc->segment_definitions,
                desc->segment_table.num_unique_segments,
                pos_in_tr);
        } else {
            out->blocks[n].segment_idx = -1;
        }

        /* ---- gradients ---- */
        gx_raw = bte->gx_id; gy_raw = bte->gy_id; gz_raw = bte->gz_id;
        gx_tab = (gx_raw >= 0 && gx_raw < desc->grad_table_size)
                 ? &desc->grad_table[gx_raw] : NULL;
        gy_tab = (gy_raw >= 0 && gy_raw < desc->grad_table_size)
                 ? &desc->grad_table[gy_raw] : NULL;
        gz_tab = (gz_raw >= 0 && gz_raw < desc->grad_table_size)
                 ? &desc->grad_table[gz_raw] : NULL;

        gx_def = (gx_tab && gx_tab->id >= 0 && gx_tab->id < desc->num_unique_grads)
                 ? &desc->grad_definitions[gx_tab->id] : NULL;
        gy_def = (gy_tab && gy_tab->id >= 0 && gy_tab->id < desc->num_unique_grads)
                 ? &desc->grad_definitions[gy_tab->id] : NULL;
        gz_def = (gz_tab && gz_tab->id >= 0 && gz_tab->id < desc->num_unique_grads)
                 ? &desc->grad_definitions[gz_tab->id] : NULL;

        if ((amplitude_mode == PULSEQLIB_AMP_MAX_POS ||
             amplitude_mode == PULSEQLIB_AMP_ZERO_VAR) &&
            pos_in_tr >= 0 && pos_in_tr < tr->tr_size) {
            idx_gx += fill_grad_waveform_for_block(desc,
                out->gx.time_us, out->gx.amplitude, idx_gx,
                gx_def, gx_tab, t0,
                &pos_max_gx[pos_in_tr * PULSEQLIB_MAX_GRAD_SHOTS], block_dur_us);
            idx_gy += fill_grad_waveform_for_block(desc,
                out->gy.time_us, out->gy.amplitude, idx_gy,
                gy_def, gy_tab, t0,
                &pos_max_gy[pos_in_tr * PULSEQLIB_MAX_GRAD_SHOTS], block_dur_us);
            idx_gz += fill_grad_waveform_for_block(desc,
                out->gz.time_us, out->gz.amplitude, idx_gz,
                gz_def, gz_tab, t0,
                &pos_max_gz[pos_in_tr * PULSEQLIB_MAX_GRAD_SHOTS], block_dur_us);
        } else {
            /* PULSEQLIB_AMP_ACTUAL, or MAX_POS/ZERO_VAR for prep/cooldown blocks */
            for (k = 0; k < PULSEQLIB_MAX_GRAD_SHOTS; ++k) actual_amp[k] = 0.0f;
            if (gx_tab) {
                k = gx_tab->shot_index;
                if (k >= 0 && k < PULSEQLIB_MAX_GRAD_SHOTS) {
                    actual_amp[k] = gx_tab->amplitude;
                    if (actual_amp[k] < 0.0f) actual_amp[k] = -actual_amp[k];
                }
            }
            idx_gx += fill_grad_waveform_for_block(desc, out->gx.time_us, out->gx.amplitude, idx_gx, gx_def, gx_tab, t0, actual_amp, block_dur_us);

            for (k = 0; k < PULSEQLIB_MAX_GRAD_SHOTS; ++k) actual_amp[k] = 0.0f;
            if (gy_tab) {
                k = gy_tab->shot_index;
                if (k >= 0 && k < PULSEQLIB_MAX_GRAD_SHOTS) {
                    actual_amp[k] = gy_tab->amplitude;
                    if (actual_amp[k] < 0.0f) actual_amp[k] = -actual_amp[k];
                }
            }
            idx_gy += fill_grad_waveform_for_block(desc, out->gy.time_us, out->gy.amplitude, idx_gy, gy_def, gy_tab, t0, actual_amp, block_dur_us);

            for (k = 0; k < PULSEQLIB_MAX_GRAD_SHOTS; ++k) actual_amp[k] = 0.0f;
            if (gz_tab) {
                k = gz_tab->shot_index;
                if (k >= 0 && k < PULSEQLIB_MAX_GRAD_SHOTS) {
                    actual_amp[k] = gz_tab->amplitude;
                    if (actual_amp[k] < 0.0f) actual_amp[k] = -actual_amp[k];
                }
            }
            idx_gz += fill_grad_waveform_for_block(desc, out->gz.time_us, out->gz.amplitude, idx_gz, gz_def, gz_tab, t0, actual_amp, block_dur_us);
        }

        /* ---- RF ---- */
        idx_rf += fill_rf_waveform_for_flat_block(desc, block_idx,
            out->rf_mag.time_us, out->rf_mag.amplitude,
            out->rf_phase.amplitude, idx_rf, t0);

        /* ---- ADC ---- */
        if (amplitude_mode == PULSEQLIB_AMP_ACTUAL) {
            /* ACTUAL mode: per-instance ADC from block table */
            if (bte->adc_id >= 0 && bte->adc_id < desc->adc_table_size) {
                int adc_def_id = desc->adc_table[bte->adc_id].id;
                if (adc_def_id >= 0 && adc_def_id < desc->num_unique_adcs) {
                    const pulseqlib_adc_definition* adef = &desc->adc_definitions[adc_def_id];
                    pulseqlib_adc_event* ev = &out->adc_events[idx_adc];
                    ev->onset_us         = t0 + (float)adef->delay;
                    ev->duration_us      = (float)adef->num_samples * (float)adef->dwell_time * 1e-3f;
                    ev->num_samples      = adef->num_samples;
                    ev->freq_offset_hz   = desc->adc_table[bte->adc_id].freq_offset;
                    ev->phase_offset_rad = desc->adc_table[bte->adc_id].phase_offset;
                    idx_adc++;
                }
            }
        } else {
            /* MAX_POS / ZERO_VAR: canonical ADC from block definition */
            if (bdef->adc_id >= 0 && bdef->adc_id < desc->num_unique_adcs) {
                const pulseqlib_adc_definition* adef = &desc->adc_definitions[bdef->adc_id];
                pulseqlib_adc_event* ev = &out->adc_events[idx_adc];
                ev->onset_us         = t0 + (float)adef->delay;
                ev->duration_us      = (float)adef->num_samples * (float)adef->dwell_time * 1e-3f;
                ev->num_samples      = adef->num_samples;
                ev->freq_offset_hz   = 0.0f;
                ev->phase_offset_rad = 0.0f;
                idx_adc++;
            }
        }

        t0 += block_dur_us;
    }

    if (pos_max_gx) PULSEQLIB_FREE(pos_max_gx);
    if (pos_max_gy) PULSEQLIB_FREE(pos_max_gy);
    if (pos_max_gz) PULSEQLIB_FREE(pos_max_gz);

    /* rf_phase shares time_us with rf_mag */
    if (total_rf > 0) {
        memcpy(out->rf_phase.time_us, out->rf_mag.time_us,
               (size_t)idx_rf * sizeof(float));
    }

    out->gx.num_samples       = idx_gx;
    out->gy.num_samples       = idx_gy;
    out->gz.num_samples       = idx_gz;
    out->rf_mag.num_samples   = idx_rf;
    out->rf_phase.num_samples = idx_rf;
    out->num_adc_events       = idx_adc;
    out->num_blocks           = block_count;
    out->total_duration_us    = t0;

    diag->code = PULSEQLIB_SUCCESS;
    return PULSEQLIB_SUCCESS;

alloc_fail:
    if (pos_max_gx) PULSEQLIB_FREE(pos_max_gx);
    if (pos_max_gy) PULSEQLIB_FREE(pos_max_gy);
    if (pos_max_gz) PULSEQLIB_FREE(pos_max_gz);
    pulseqlib_tr_waveforms_free(out);
    diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
    return diag->code;
}

void pulseqlib_tr_waveforms_free(pulseqlib_tr_waveforms* w)
{
    if (!w) return;
    if (w->gx.time_us)         PULSEQLIB_FREE(w->gx.time_us);
    if (w->gx.amplitude)       PULSEQLIB_FREE(w->gx.amplitude);
    if (w->gy.time_us)         PULSEQLIB_FREE(w->gy.time_us);
    if (w->gy.amplitude)       PULSEQLIB_FREE(w->gy.amplitude);
    if (w->gz.time_us)         PULSEQLIB_FREE(w->gz.time_us);
    if (w->gz.amplitude)       PULSEQLIB_FREE(w->gz.amplitude);
    if (w->rf_mag.time_us)     PULSEQLIB_FREE(w->rf_mag.time_us);
    if (w->rf_mag.amplitude)   PULSEQLIB_FREE(w->rf_mag.amplitude);
    if (w->rf_phase.time_us)   PULSEQLIB_FREE(w->rf_phase.time_us);
    if (w->rf_phase.amplitude) PULSEQLIB_FREE(w->rf_phase.amplitude);
    if (w->adc_events)         PULSEQLIB_FREE(w->adc_events);
    if (w->blocks)             PULSEQLIB_FREE(w->blocks);
    memset(w, 0, sizeof(*w));
}
