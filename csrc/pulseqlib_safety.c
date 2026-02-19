/* pulseqlib_safety.c -- gradient waveform extraction, acoustic analysis, PNS
 *
 * Public functions:
 *   pulseqlib_get_tr_gradient_waveforms / _free
 *   pulseqlib_get_tr_acoustic_spectra   / _free
 *   pulseqlib_compute_pns               / _free
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "pulseqlib_internal.h"
#include "pulseqlib_methods.h"

#if PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_GEHC
#include "external_kiss_fft.h"
#include "external_kiss_fftr.h"
#endif

/* ================================================================== */
/*  File-scope constants                                               */
/* ================================================================== */
#define PEAK_LOG10_THRESHOLD    2.25f
#define PEAK_NORM_SCALE         10.0f
#define PEAK_EPS                1e-30f

#define PNS_KERNEL_DURATION_FACTOR 20.0f

/* ================================================================== */
/*  Gradient waveform free                                            */
/* ================================================================== */

void pulseqlib_tr_gradient_waveforms_free(pulseqlib_tr_gradient_waveforms* w)
{
    if (!w) return;
    if (w->time)       PULSEQLIB_FREE(w->time);
    if (w->waveform_gx) PULSEQLIB_FREE(w->waveform_gx);
    if (w->waveform_gy) PULSEQLIB_FREE(w->waveform_gy);
    if (w->waveform_gz) PULSEQLIB_FREE(w->waveform_gz);
    w->time        = NULL;
    w->waveform_gx = NULL;
    w->waveform_gy = NULL;
    w->waveform_gz = NULL;
    w->num_samples = 0;
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

    grad_raster_us  = desc->grad_raster_time_us;
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

    for (tr_idx = 0; tr_idx < num_trs; ++tr_idx) {
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
    return PULSEQLIB_OK;
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
static int find_unique_shot_trs(
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
    grad_raster_us = desc->grad_raster_time_us;
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
        return PULSEQLIB_OK;

    t_in = *time;
    w_in = *waveform;
    n_in = *num_samples;

    t_start  = t_in[0];
    t_end    = t_in[n_in - 1];
    duration = t_end - t_start;
    if (duration <= 0.0f) return PULSEQLIB_OK;

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
    return PULSEQLIB_OK;
}

/* ================================================================== */
/*  Gradient waveforms for an arbitrary block range                   */
/* ================================================================== */

/*  amplitude_mode:
 *    0 = actual block amplitude (single-TR)
 *    1 = position-max (worst-case safety)
 *    2 = definition-min (best-case k-space)
 */
static int get_gradient_waveforms_range(
    const pulseqlib_sequence_descriptor* desc,
    pulseqlib_tr_gradient_waveforms* waveforms,
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

    pos_max_gx = NULL;
    pos_max_gy = NULL;
    pos_max_gz = NULL;
    time_gx = NULL;
    time_gy = NULL;
    time_gz = NULL;

    if (!diag) { pulseqlib_diagnostic_init(&local_diag); diag = &local_diag; }
    else       { pulseqlib_diagnostic_init(diag); }

    if (!desc || !waveforms) {
        diag->code = PULSEQLIB_ERR_NULL_POINTER;
        return diag->code;
    }

    memset(waveforms, 0, sizeof(*waveforms));

    if (block_count <= 0) {
        diag->code = PULSEQLIB_ERR_TR_NO_BLOCKS;
        return diag->code;
    }
    if (block_start < 0 || block_start + block_count > desc->num_blocks) {
        diag->code = PULSEQLIB_ERR_INVALID_ARGUMENT;
        return diag->code;
    }

    /* position-max amplitudes (only for worst-case main-TR mode) */
    if (amplitude_mode == 1) {
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
    time_gx            = (float*)PULSEQLIB_ALLOC((size_t)total_gx * sizeof(float));
    waveforms->waveform_gx = (float*)PULSEQLIB_ALLOC((size_t)total_gx * sizeof(float));
    time_gy            = (float*)PULSEQLIB_ALLOC((size_t)total_gy * sizeof(float));
    waveforms->waveform_gy = (float*)PULSEQLIB_ALLOC((size_t)total_gy * sizeof(float));
    time_gz            = (float*)PULSEQLIB_ALLOC((size_t)total_gz * sizeof(float));
    waveforms->waveform_gz = (float*)PULSEQLIB_ALLOC((size_t)total_gz * sizeof(float));
    if (!time_gx || !waveforms->waveform_gx ||
        !time_gy || !waveforms->waveform_gy ||
        !time_gz || !waveforms->waveform_gz) {
        if (pos_max_gx) PULSEQLIB_FREE(pos_max_gx);
        if (pos_max_gy) PULSEQLIB_FREE(pos_max_gy);
        if (pos_max_gz) PULSEQLIB_FREE(pos_max_gz);
        if (time_gx) PULSEQLIB_FREE(time_gx);
        if (time_gy) PULSEQLIB_FREE(time_gy);
        if (time_gz) PULSEQLIB_FREE(time_gz);
        pulseqlib_tr_gradient_waveforms_free(waveforms);
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

        if (amplitude_mode == 1) {
            idx_gx += fill_grad_waveform_for_block(desc,
                time_gx, waveforms->waveform_gx, idx_gx,
                gx_def, gx_tab, t0,
                &pos_max_gx[n * PULSEQLIB_MAX_GRAD_SHOTS], block_dur_us);
            idx_gy += fill_grad_waveform_for_block(desc,
                time_gy, waveforms->waveform_gy, idx_gy,
                gy_def, gy_tab, t0,
                &pos_max_gy[n * PULSEQLIB_MAX_GRAD_SHOTS], block_dur_us);
            idx_gz += fill_grad_waveform_for_block(desc,
                time_gz, waveforms->waveform_gz, idx_gz,
                gz_def, gz_tab, t0,
                &pos_max_gz[n * PULSEQLIB_MAX_GRAD_SHOTS], block_dur_us);
        } else if (amplitude_mode == 2) {
            /* definition-min mode: use gd->min_amplitude */
            for (k = 0; k < PULSEQLIB_MAX_GRAD_SHOTS; ++k) actual_amp[k] = 0.0f;
            if (gx_def) {
                for (k = 0; k < PULSEQLIB_MAX_GRAD_SHOTS; ++k)
                    actual_amp[k] = gx_def->min_amplitude[k];
            }
            idx_gx += fill_grad_waveform_for_block(desc,
                time_gx, waveforms->waveform_gx, idx_gx,
                gx_def, gx_tab, t0, actual_amp, block_dur_us);

            for (k = 0; k < PULSEQLIB_MAX_GRAD_SHOTS; ++k) actual_amp[k] = 0.0f;
            if (gy_def) {
                for (k = 0; k < PULSEQLIB_MAX_GRAD_SHOTS; ++k)
                    actual_amp[k] = gy_def->min_amplitude[k];
            }
            idx_gy += fill_grad_waveform_for_block(desc,
                time_gy, waveforms->waveform_gy, idx_gy,
                gy_def, gy_tab, t0, actual_amp, block_dur_us);

            for (k = 0; k < PULSEQLIB_MAX_GRAD_SHOTS; ++k) actual_amp[k] = 0.0f;
            if (gz_def) {
                for (k = 0; k < PULSEQLIB_MAX_GRAD_SHOTS; ++k)
                    actual_amp[k] = gz_def->min_amplitude[k];
            }
            idx_gz += fill_grad_waveform_for_block(desc,
                time_gz, waveforms->waveform_gz, idx_gz,
                gz_def, gz_tab, t0, actual_amp, block_dur_us);
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
                time_gx, waveforms->waveform_gx, idx_gx,
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
                time_gy, waveforms->waveform_gy, idx_gy,
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
                time_gz, waveforms->waveform_gz, idx_gz,
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
    target_raster_us = 0.5f * desc->grad_raster_time_us;

    result = interpolate_to_uniform(
        &time_gx, &waveforms->waveform_gx,
        &num_gx, target_raster_us);
    if (PULSEQLIB_FAILED(result)) {
        if (time_gx) PULSEQLIB_FREE(time_gx);
        if (time_gy) PULSEQLIB_FREE(time_gy);
        if (time_gz) PULSEQLIB_FREE(time_gz);
        pulseqlib_tr_gradient_waveforms_free(waveforms);
        diag->code = result; return result;
    }
    result = interpolate_to_uniform(
        &time_gy, &waveforms->waveform_gy,
        &num_gy, target_raster_us);
    if (PULSEQLIB_FAILED(result)) {
        if (time_gx) PULSEQLIB_FREE(time_gx);
        if (time_gy) PULSEQLIB_FREE(time_gy);
        if (time_gz) PULSEQLIB_FREE(time_gz);
        pulseqlib_tr_gradient_waveforms_free(waveforms);
        diag->code = result; return result;
    }
    result = interpolate_to_uniform(
        &time_gz, &waveforms->waveform_gz,
        &num_gz, target_raster_us);
    if (PULSEQLIB_FAILED(result)) {
        if (time_gx) PULSEQLIB_FREE(time_gx);
        if (time_gy) PULSEQLIB_FREE(time_gy);
        if (time_gz) PULSEQLIB_FREE(time_gz);
        pulseqlib_tr_gradient_waveforms_free(waveforms);
        diag->code = result; return result;
    }

    /* Post-interpolation: all axes share the same uniform raster.
     * Keep time_gx as the shared time vector, discard time_gy/time_gz. */
    waveforms->time        = time_gx;
    waveforms->num_samples = num_gx;
    PULSEQLIB_FREE(time_gy);
    PULSEQLIB_FREE(time_gz);

    diag->code = PULSEQLIB_OK;
    return PULSEQLIB_OK;
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
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences) {
        if (diag) { pulseqlib_diagnostic_init(diag); diag->code = PULSEQLIB_ERR_INVALID_ARGUMENT; }
        return PULSEQLIB_ERR_INVALID_ARGUMENT;
    }
    desc = &coll->descriptors[subseq_idx];
    return get_gradient_waveforms_range(desc, waveforms, diag,
        desc->tr_descriptor.num_prep_blocks,
        desc->tr_descriptor.tr_size,
        1, NULL, 0);
}

/* ================================================================== */
/*  Min-amplitude TR gradient waveforms (for k-space trajectory)      */
/* ================================================================== */

int pulseqlib_get_tr_gradient_waveforms_min(
    const pulseqlib_sequence_descriptor* desc,
    pulseqlib_tr_gradient_waveforms* waveforms,
    pulseqlib_diagnostic* diag)
{
    if (!desc) {
        if (diag) { pulseqlib_diagnostic_init(diag); diag->code = PULSEQLIB_ERR_NULL_POINTER; }
        return PULSEQLIB_ERR_NULL_POINTER;
    }
    return get_gradient_waveforms_range(desc, waveforms, diag,
        desc->tr_descriptor.num_prep_blocks,
        desc->tr_descriptor.tr_size,
        2, NULL, 0);
}

/* ================================================================== */
/*  K-space trajectory from uniform gradient waveforms                */
/* ================================================================== */

/*
 * Computes k-space trajectory by cumulative trapezoidal integration of
 * gradient waveforms (already on uniform raster).
 *
 * Output arrays (kx, ky, kz, krss) must be caller-allocated with at
 * least waveforms->num_samples elements.  dt_us is returned for
 * convenience.
 */
static int compute_kspace_trajectory(
    const pulseqlib_tr_gradient_waveforms* waveforms,
    float* kx, float* ky, float* kz, float* krss,
    float* dt_us)
{
    int i, n;
    float dt_s;
    float cum_x, cum_y, cum_z;
    float v;

    if (!waveforms || !waveforms->time || waveforms->num_samples < 2)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;

    n = waveforms->num_samples;
    *dt_us = waveforms->time[1] - waveforms->time[0];
    dt_s   = (*dt_us) * 1e-6f;

    /* cumulative trapezoidal integration */
    cum_x = 0.0f; cum_y = 0.0f; cum_z = 0.0f;
    kx[0] = 0.0f; ky[0] = 0.0f; kz[0] = 0.0f;

    for (i = 1; i < n; ++i) {
        cum_x += 0.5f * (waveforms->waveform_gx[i - 1] + waveforms->waveform_gx[i]) * dt_s;
        cum_y += 0.5f * (waveforms->waveform_gy[i - 1] + waveforms->waveform_gy[i]) * dt_s;
        cum_z += 0.5f * (waveforms->waveform_gz[i - 1] + waveforms->waveform_gz[i]) * dt_s;
        kx[i] = cum_x;
        ky[i] = cum_y;
        kz[i] = cum_z;
    }

    /* RSS magnitude */
    for (i = 0; i < n; ++i) {
        v = kx[i] * kx[i] + ky[i] * ky[i] + kz[i] * kz[i];
        krss[i] = (v > 0.0f) ? (float)sqrt((double)v) : 0.0f;
    }

    return PULSEQLIB_OK;
}

/* ================================================================== */
/*  Find k-space zero crossings (k=0 passages)                       */
/* ================================================================== */

/*
 * A zero crossing is a local minimum of krss that is <= threshold.
 * Floor convention: for symmetric plateaus the leftmost sample is kept.
 *
 * Two-pass protocol:
 *   find_kspace_zero_crossings(krss, n, thr, NULL, &cnt, 1);
 *   indices = PULSEQLIB_ALLOC(cnt * sizeof(int));
 *   find_kspace_zero_crossings(krss, n, thr, indices, &cnt, 0);
 */
static int find_kspace_zero_crossings(
    const float* krss, int n, float threshold,
    int* zero_indices, int* out_count, int count_only)
{
    int i, cnt;
    float prev, curr, next;

    if (!krss || n < 2 || !out_count)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;

    cnt = 0;

    for (i = 0; i < n; ++i) {
        curr = krss[i];
        if (curr > threshold) continue;

        prev = (i > 0)     ? krss[i - 1] : curr + 1.0f;
        next = (i < n - 1) ? krss[i + 1] : curr + 1.0f;

        /* curr <= both neighbors => local minimum */
        if (curr <= prev && curr <= next) {
            /* plateau dedup: skip if left neighbor equals curr and
             * was itself a local min (already recorded) */
            if (i > 0 && krss[i - 1] == curr) {
                float pprev = (i > 1) ? krss[i - 2] : curr + 1.0f;
                if (krss[i - 1] <= pprev)
                    continue;
            }
            if (!count_only && zero_indices)
                zero_indices[cnt] = i;
            cnt++;
        }
    }

    *out_count = cnt;
    return PULSEQLIB_OK;
}

/* ================================================================== */
/*  Compute segment timing anchors                                    */
/* ================================================================== */

int pulseqlib__compute_segment_timing(
    pulseqlib_sequence_descriptor* desc,
    pulseqlib_diagnostic* diag)
{
    pulseqlib_diagnostic local_diag;
    int seg_idx, blk, block_idx, result;
    const pulseqlib_tr_segment* seg;
    const pulseqlib_block_table_element* bte;
    const pulseqlib_block_definition* bdef;
    int rf_count, adc_count;
    float t_accum, block_dur_us;
    int rf_raw, adc_raw;
    const pulseqlib_rf_definition* rdef;
    const pulseqlib_adc_definition* adef;
    const pulseqlib_rf_table_element* rte;
    pulseqlib_segment_rf_anchor* rf_arr;
    pulseqlib_segment_adc_anchor* adc_arr;
    int rf_def_id, adc_def_id;
    float adc_dur_us;

    /* k-space trajectory variables */
    pulseqlib_tr_gradient_waveforms min_waveforms;
    float *kx, *ky, *kz, *krss;
    int *kzero_indices;
    int num_kzero, n_samples;
    float dt_us, k_threshold;

    /* ADC-to-kzero mapping variables */
    int a, s, closest_idx, zi, kz_sample;
    float seg_time_offset, adc_mid_us, best_dist, dist;
    float kzero_time_us, kzero_in_adc;
    int pos_in_tr, num_prep;
    int tr_size;

    int has_kspace;

    if (!diag) { pulseqlib_diagnostic_init(&local_diag); diag = &local_diag; }

    memset(&min_waveforms, 0, sizeof(min_waveforms));
    kx = NULL; ky = NULL; kz = NULL; krss = NULL;
    kzero_indices = NULL;
    num_kzero = 0;
    n_samples = 0;
    dt_us = 0.0f;
    has_kspace = 0;

    if (!desc || desc->num_unique_segments <= 0)
        return PULSEQLIB_OK;

    num_prep = desc->tr_descriptor.num_prep_blocks;
    tr_size  = desc->tr_descriptor.tr_size;

    /* ---- Step A: build min-amplitude k-space trajectory ---- */
    if (tr_size > 0) {
        result = get_gradient_waveforms_range(desc, &min_waveforms, diag,
            num_prep, tr_size, 2, NULL, 0);

        if (!PULSEQLIB_FAILED(result) && min_waveforms.num_samples >= 2) {
            n_samples = min_waveforms.num_samples;
            kx   = (float*)PULSEQLIB_ALLOC((size_t)n_samples * sizeof(float));
            ky   = (float*)PULSEQLIB_ALLOC((size_t)n_samples * sizeof(float));
            kz   = (float*)PULSEQLIB_ALLOC((size_t)n_samples * sizeof(float));
            krss = (float*)PULSEQLIB_ALLOC((size_t)n_samples * sizeof(float));
            if (kx && ky && kz && krss) {
                result = compute_kspace_trajectory(&min_waveforms,
                    kx, ky, kz, krss, &dt_us);
                if (!PULSEQLIB_FAILED(result)) {
                    /* threshold = 1% of max |k| */
                    k_threshold = 0.0f;
                    for (a = 0; a < n_samples; ++a)
                        if (krss[a] > k_threshold) k_threshold = krss[a];
                    k_threshold *= 0.01f;
                    if (k_threshold < 1e-10f) k_threshold = 1e-10f;

                    find_kspace_zero_crossings(krss, n_samples, k_threshold,
                                               NULL, &num_kzero, 1);
                    if (num_kzero > 0) {
                        kzero_indices = (int*)PULSEQLIB_ALLOC(
                            (size_t)num_kzero * sizeof(int));
                        if (kzero_indices) {
                            find_kspace_zero_crossings(krss, n_samples,
                                k_threshold, kzero_indices, &num_kzero, 0);
                            has_kspace = 1;
                        }
                    } else {
                        has_kspace = 1;  /* valid trajectory, just no crossings */
                    }
                }
            }
        }
    }

    /* ---- Step B: for each segment, collect RF and ADC anchors ---- */
    for (seg_idx = 0; seg_idx < desc->num_unique_segments; ++seg_idx) {
        seg = &desc->segment_definitions[seg_idx];

        /* count RF and ADC events */
        rf_count  = 0;
        adc_count = 0;
        for (blk = 0; blk < seg->num_blocks; ++blk) {
            block_idx = seg->start_block + blk;
            if (block_idx < 0 || block_idx >= desc->num_blocks) continue;
            bte = &desc->block_table[block_idx];
            if (bte->rf_id >= 0)  rf_count++;
            if (bte->adc_id >= 0) adc_count++;
        }

        /* allocate anchor arrays */
        rf_arr  = NULL;
        adc_arr = NULL;
        if (rf_count > 0) {
            rf_arr = (pulseqlib_segment_rf_anchor*)PULSEQLIB_ALLOC(
                (size_t)rf_count * sizeof(pulseqlib_segment_rf_anchor));
            if (!rf_arr) goto timing_fail;
        }
        if (adc_count > 0) {
            adc_arr = (pulseqlib_segment_adc_anchor*)PULSEQLIB_ALLOC(
                (size_t)adc_count * sizeof(pulseqlib_segment_adc_anchor));
            if (!adc_arr) {
                if (rf_arr) PULSEQLIB_FREE(rf_arr);
                goto timing_fail;
            }
        }

        /* compute segment start time within TR (for kzero mapping) */
        seg_time_offset = 0.0f;
        if (has_kspace && seg->start_block >= num_prep &&
            seg->start_block < num_prep + tr_size) {
            pos_in_tr = seg->start_block - num_prep;
            for (s = 0; s < pos_in_tr; ++s) {
                bte = &desc->block_table[num_prep + s];
                bdef = &desc->block_definitions[bte->id];
                seg_time_offset += (bte->duration_us >= 0)
                    ? (float)bte->duration_us
                    : (float)bdef->duration_us;
            }
        }

        /* fill anchors */
        rf_count  = 0;
        adc_count = 0;
        t_accum   = 0.0f;

        for (blk = 0; blk < seg->num_blocks; ++blk) {
            block_idx = seg->start_block + blk;
            if (block_idx < 0 || block_idx >= desc->num_blocks) continue;
            bte  = &desc->block_table[block_idx];
            bdef = &desc->block_definitions[bte->id];
            block_dur_us = (bte->duration_us >= 0)
                ? (float)bte->duration_us
                : (float)bdef->duration_us;

            /* RF anchor */
            rf_raw = bte->rf_id;
            if (rf_raw >= 0 && rf_raw < desc->rf_table_size) {
                rte = &desc->rf_table[rf_raw];
                rf_def_id = rte->id;
                if (rf_def_id >= 0 && rf_def_id < desc->num_unique_rfs) {
                    rdef = &desc->rf_definitions[rf_def_id];
                    rf_arr[rf_count].block_offset = blk;
                    rf_arr[rf_count].start_us     = t_accum + (float)rdef->delay;
#if PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_GEHC
                    rf_arr[rf_count].end_us        = t_accum + (float)rdef->delay +
                                                     rdef->stats.duration_us;
                    rf_arr[rf_count].isocenter_us   = t_accum + (float)rdef->delay +
                                                      (float)rdef->stats.isodelay_us;
#else
                    rf_arr[rf_count].end_us         = t_accum + block_dur_us;
                    rf_arr[rf_count].isocenter_us    = t_accum + (float)rdef->delay;
#endif
                    rf_arr[rf_count].base_amplitude = rte->amplitude;
                    rf_count++;
                }
            }

            /* ADC anchor */
            adc_raw = bte->adc_id;
            if (adc_raw >= 0 && adc_raw < desc->adc_table_size) {
                adc_def_id = desc->adc_table[adc_raw].id;
                if (adc_def_id >= 0 && adc_def_id < desc->num_unique_adcs) {
                    adef = &desc->adc_definitions[adc_def_id];
                    adc_dur_us = (float)adef->num_samples *
                                 (float)adef->dwell_time * 1e-3f;

                    adc_arr[adc_count].block_offset = blk;
                    adc_arr[adc_count].start_us = t_accum + (float)adef->delay;
                    adc_arr[adc_count].end_us   = t_accum + (float)adef->delay +
                                                  adc_dur_us;

                    /* default: N/2 */
                    adc_arr[adc_count].kzero_index = adef->num_samples / 2;
                    adc_arr[adc_count].kzero_us    =
                        adc_arr[adc_count].start_us +
                        (float)(adef->num_samples / 2) *
                        (float)adef->dwell_time * 1e-3f;

                    /* refine kzero via k-space trajectory */
                    if (has_kspace && num_kzero > 0 &&
                        seg->start_block >= num_prep &&
                        seg->start_block < num_prep + tr_size) {

                        adc_mid_us = seg_time_offset +
                            adc_arr[adc_count].start_us +
                            0.5f * adc_dur_us;

                        /* find closest k=0 crossing */
                        closest_idx = 0;
                        best_dist   = 1e30f;
                        for (a = 0; a < num_kzero; ++a) {
                            zi = kzero_indices[a];
                            dist = (float)zi * dt_us +
                                   min_waveforms.time[0] - adc_mid_us;
                            if (dist < 0.0f) dist = -dist;
                            if (dist < best_dist) {
                                best_dist   = dist;
                                closest_idx = a;
                            }
                        }

                        /* convert to ADC sample index */
                        kzero_time_us = (float)kzero_indices[closest_idx] *
                                        dt_us + min_waveforms.time[0];
                        kzero_in_adc  = kzero_time_us -
                            (seg_time_offset + adc_arr[adc_count].start_us);
                        kz_sample = (int)(kzero_in_adc /
                            ((float)adef->dwell_time * 1e-3f));
                        if (kz_sample < 0) kz_sample = 0;
                        if (kz_sample >= adef->num_samples)
                            kz_sample = adef->num_samples - 1;
                        adc_arr[adc_count].kzero_index = kz_sample;
                        adc_arr[adc_count].kzero_us    =
                            kzero_time_us - seg_time_offset;
                    }

                    adc_count++;
                }
            }

            t_accum += block_dur_us;
        }

        /* store timing */
        ((pulseqlib_tr_segment*)seg)->timing.num_rf_anchors  = rf_count;
        ((pulseqlib_tr_segment*)seg)->timing.rf_anchors      = rf_arr;
        ((pulseqlib_tr_segment*)seg)->timing.num_adc_anchors = adc_count;
        ((pulseqlib_tr_segment*)seg)->timing.adc_anchors     = adc_arr;
        ((pulseqlib_tr_segment*)seg)->timing.num_kzero_crossings = num_kzero;
        ((pulseqlib_tr_segment*)seg)->timing.kzero_crossing_indices = NULL;

        if (num_kzero > 0 && kzero_indices) {
            int* copy = (int*)PULSEQLIB_ALLOC((size_t)num_kzero * sizeof(int));
            if (copy) {
                int ci;
                for (ci = 0; ci < num_kzero; ++ci) copy[ci] = kzero_indices[ci];
                ((pulseqlib_tr_segment*)seg)->timing.kzero_crossing_indices = copy;
            }
        }
    }

    /* cleanup */
    if (kx)   PULSEQLIB_FREE(kx);
    if (ky)   PULSEQLIB_FREE(ky);
    if (kz)   PULSEQLIB_FREE(kz);
    if (krss) PULSEQLIB_FREE(krss);
    if (kzero_indices) PULSEQLIB_FREE(kzero_indices);
    pulseqlib_tr_gradient_waveforms_free(&min_waveforms);

    return PULSEQLIB_OK;

timing_fail:
    if (kx)   PULSEQLIB_FREE(kx);
    if (ky)   PULSEQLIB_FREE(ky);
    if (kz)   PULSEQLIB_FREE(kz);
    if (krss) PULSEQLIB_FREE(krss);
    if (kzero_indices) PULSEQLIB_FREE(kzero_indices);
    pulseqlib_tr_gradient_waveforms_free(&min_waveforms);
    return PULSEQLIB_ERR_ALLOC_FAILED;
}

/* ================================================================== */
/*  Acoustic spectra free                                             */
/* ================================================================== */

void pulseqlib_tr_acoustic_spectra_free(pulseqlib_tr_acoustic_spectra* s)
{
    if (!s) return;

    if (s->frequencies)      PULSEQLIB_FREE(s->frequencies);
    if (s->spectra_gx)       PULSEQLIB_FREE(s->spectra_gx);
    if (s->spectra_gy)       PULSEQLIB_FREE(s->spectra_gy);
    if (s->spectra_gz)       PULSEQLIB_FREE(s->spectra_gz);
    if (s->max_envelope_gx)  PULSEQLIB_FREE(s->max_envelope_gx);
    if (s->max_envelope_gy)  PULSEQLIB_FREE(s->max_envelope_gy);
    if (s->max_envelope_gz)  PULSEQLIB_FREE(s->max_envelope_gz);
    if (s->peaks_gx)         PULSEQLIB_FREE(s->peaks_gx);
    if (s->peaks_gy)         PULSEQLIB_FREE(s->peaks_gy);
    if (s->peaks_gz)         PULSEQLIB_FREE(s->peaks_gz);
    if (s->frequencies_full) PULSEQLIB_FREE(s->frequencies_full);
    if (s->spectra_gx_full)  PULSEQLIB_FREE(s->spectra_gx_full);
    if (s->spectra_gy_full)  PULSEQLIB_FREE(s->spectra_gy_full);
    if (s->spectra_gz_full)  PULSEQLIB_FREE(s->spectra_gz_full);
    if (s->frequencies_seq)  PULSEQLIB_FREE(s->frequencies_seq);
    if (s->spectra_gx_seq)   PULSEQLIB_FREE(s->spectra_gx_seq);
    if (s->spectra_gy_seq)   PULSEQLIB_FREE(s->spectra_gy_seq);
    if (s->spectra_gz_seq)   PULSEQLIB_FREE(s->spectra_gz_seq);
    if (s->peaks_gx_seq)     PULSEQLIB_FREE(s->peaks_gx_seq);
    if (s->peaks_gy_seq)     PULSEQLIB_FREE(s->peaks_gy_seq);
    if (s->peaks_gz_seq)     PULSEQLIB_FREE(s->peaks_gz_seq);

    memset(s, 0, sizeof(*s));
}

/* ================================================================== */
/*  Acoustic support structure                                        */
/* ================================================================== */

#if PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_GEHC

typedef struct {
    int   nwin;
    int   nfft;
    int   nfreq;
    int   output_freq_bins;
    int   num_windows;
    int   hop_size;
    float grad_raster_us;
    float freq_resolution;
    float max_frequency_hz;
    float* cos_window;
    float* work_buffer;
    kiss_fftr_cfg fft_cfg;
    kiss_fft_cpx* fft_out;
} acoustic_support;

typedef struct {
    int    num_samples;
    int    num_samples_original;
    float* samples;
    int    owns_memory;
} acoustic_waveform;

static int acoustic_support_init(
    acoustic_support* sup,
    int num_samples, int target_window_size,
    float target_spectral_resolution_hz,
    float grad_raster_us, float max_frequency_hz)
{
    int nwin, nfft, nfreq, output_freq_bins;
    int hop_size, num_windows, padded_len, min_nfft, max_idx, i;
    float freq_res;
    float* cos_win  = NULL;
    float* work     = NULL;
    kiss_fft_cpx* fft_out = NULL;
    kiss_fftr_cfg cfg = NULL;

    if (!sup || num_samples <= 0 || target_window_size <= 0 ||
        target_spectral_resolution_hz <= 0.0f)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;

    memset(sup, 0, sizeof(*sup));

    nwin = (num_samples >= target_window_size) ? target_window_size : num_samples;

    min_nfft = (int)ceil((double)(1.0e6f / (grad_raster_us * target_spectral_resolution_hz)));
    nfft = (min_nfft < nwin) ? nwin : (int)pulseqlib__next_pow2((size_t)min_nfft);
    if (nfft < nwin) nfft = (int)pulseqlib__next_pow2((size_t)nwin);

    nfreq = nfft / 2 + 1;
    freq_res = (float)(1.0e6 / (grad_raster_us * (double)nfft));

    if (max_frequency_hz < 0.0f) {
        output_freq_bins = nfreq;
    } else {
        max_idx = (int)(max_frequency_hz / freq_res + 0.5f);
        if (max_idx >= nfreq) output_freq_bins = nfreq;
        else if (max_idx < 1) output_freq_bins = 1;
        else output_freq_bins = max_idx + 1;
    }

    hop_size = nwin / 2;
    if (hop_size < 1) hop_size = 1;

    if (num_samples <= nwin) {
        num_windows = 1;
    } else {
        padded_len  = ((num_samples + nwin - 1) / nwin) * nwin;
        num_windows = (padded_len - nwin) / hop_size + 1;
    }

    cos_win = (float*)PULSEQLIB_ALLOC(nwin * sizeof(float));
    work    = (float*)PULSEQLIB_ALLOC(nfft * sizeof(float));
    fft_out = (kiss_fft_cpx*)PULSEQLIB_ALLOC(nfreq * sizeof(kiss_fft_cpx));
    if (!cos_win || !work || !fft_out) goto fail;

    cfg = kiss_fftr_alloc(nfft, 0, NULL, NULL);
    if (!cfg) goto fail;

    for (i = 0; i < nwin; ++i)
        cos_win[i] = 0.5f * (1.0f - (float)cos(2.0 * M_PI * (double)(i + 1) / (double)nwin));

    sup->nwin             = nwin;
    sup->nfft             = nfft;
    sup->nfreq            = nfreq;
    sup->output_freq_bins = output_freq_bins;
    sup->num_windows      = num_windows;
    sup->hop_size         = hop_size;
    sup->grad_raster_us   = grad_raster_us;
    sup->freq_resolution  = freq_res;
    sup->max_frequency_hz = max_frequency_hz;
    sup->cos_window       = cos_win;
    sup->work_buffer      = work;
    sup->fft_cfg          = cfg;
    sup->fft_out          = fft_out;
    return PULSEQLIB_OK;

fail:
    if (cos_win) PULSEQLIB_FREE(cos_win);
    if (work)    PULSEQLIB_FREE(work);
    if (fft_out) PULSEQLIB_FREE(fft_out);
    if (cfg)     kiss_fftr_free(cfg);
    return PULSEQLIB_ERR_ALLOC_FAILED;
}

static void acoustic_support_free(acoustic_support* sup)
{
    if (!sup) return;
    if (sup->cos_window)   PULSEQLIB_FREE(sup->cos_window);
    if (sup->work_buffer)  PULSEQLIB_FREE(sup->work_buffer);
    if (sup->fft_out)      PULSEQLIB_FREE(sup->fft_out);
    if (sup->fft_cfg)      kiss_fftr_free(sup->fft_cfg);
    memset(sup, 0, sizeof(*sup));
}

static int acoustic_waveform_init(
    acoustic_waveform* aw,
    const acoustic_support* sup,
    const float* waveform, int num_samples, int padded_len)
{
    float* buf;
    int i;

    buf = NULL;

    if (!aw || !sup || !waveform || num_samples <= 0)
        return PULSEQLIB_ERR_NULL_POINTER;

    memset(aw, 0, sizeof(*aw));
    aw->num_samples_original = num_samples;

    buf = (float*)PULSEQLIB_ALLOC(padded_len * sizeof(float));
    if (!buf) return PULSEQLIB_ERR_ALLOC_FAILED;

    for (i = 0; i < num_samples; ++i) buf[i] = waveform[i];
    for (i = num_samples; i < padded_len; ++i) buf[i] = 0.0f;

    aw->num_samples = padded_len;
    aw->samples     = buf;
    aw->owns_memory = 1;
    return PULSEQLIB_OK;
}

static void acoustic_waveform_free(acoustic_waveform* aw)
{
    if (!aw) return;
    if (aw->owns_memory && aw->samples) PULSEQLIB_FREE(aw->samples);
    memset(aw, 0, sizeof(*aw));
}

/* ================================================================== */
/*  Single window spectrum                                            */
/* ================================================================== */

static int compute_window_spectrum(
    acoustic_support* sup, float* spectrum,
    const acoustic_waveform* aw, int window_index)
{
    int i, start_idx;
    int nwin, nfft, nfreq, out_bins;
    float* work;
    float* cos_win;
    const float* samples;
    float mean, fft_norm;
    kiss_fft_cpx* fft_out;

    if (!spectrum || !sup || !aw || !aw->samples)
        return PULSEQLIB_ERR_NULL_POINTER;
    if (window_index < 0 || window_index >= sup->num_windows)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;

    nwin     = sup->nwin;
    nfft     = sup->nfft;
    nfreq    = sup->nfreq;
    out_bins = sup->output_freq_bins;
    work     = sup->work_buffer;
    cos_win  = sup->cos_window;
    fft_out  = sup->fft_out;
    samples  = aw->samples;

    start_idx = window_index * sup->hop_size;
    if (start_idx + nwin > aw->num_samples)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;

    for (i = 0; i < nwin; ++i)  work[i] = samples[start_idx + i];
    for (i = nwin; i < nfft; ++i) work[i] = 0.0f;

    mean = 0.0f;
    for (i = 0; i < nwin; ++i) mean += work[i];
    mean /= (float)nwin;
    for (i = 0; i < nwin; ++i) work[i] -= mean;

    for (i = 0; i < nwin; ++i) work[i] *= cos_win[i];

    kiss_fftr(sup->fft_cfg, work, fft_out);

    fft_norm = 1.0f / (float)nfft;
    for (i = 0; i < nfreq; ++i) { fft_out[i].r *= fft_norm; fft_out[i].i *= fft_norm; }

    for (i = 0; i < out_bins; ++i)
        spectrum[i] = (float)sqrt((double)(fft_out[i].r * fft_out[i].r +
                                           fft_out[i].i * fft_out[i].i));
    return PULSEQLIB_OK;
}

/* ================================================================== */
/*  Resonance peak detection                                          */
/* ================================================================== */

/* FIX: output first */
static void detect_resonances(int* peaks, const float* mag, int n)
{
    int i;
    float max_val, sum_log, mean_log, norm, log_val;

    if (n <= 0 || !mag || !peaks) return;
    for (i = 0; i < n; ++i) peaks[i] = 0;
    if (n < 3) return;

    max_val = 0.0f;
    for (i = 0; i < n; ++i) if (mag[i] > max_val) max_val = mag[i];
    if (max_val <= 0.0f) return;

    sum_log = 0.0f;
    for (i = 0; i < n; ++i) {
        norm = (mag[i] / max_val + PEAK_EPS) * PEAK_NORM_SCALE;
        sum_log += (float)log10((double)norm);
    }
    mean_log = sum_log / (float)n;

    for (i = 1; i < n - 1; ++i) {
        if (mag[i] > mag[i-1] && mag[i] > mag[i+1]) {
            norm    = (mag[i] / max_val + PEAK_EPS) * PEAK_NORM_SCALE;
            log_val = (float)log10((double)norm);
            if (log_val - mean_log > PEAK_LOG10_THRESHOLD)
                peaks[i] = 1;
        }
    }
}

/* ================================================================== */
/*  Acoustic violation check                                          */
/* ================================================================== */

static int check_acoustic_violations(
    pulseqlib_acoustic_violation* violation, int** out_peaks,
    const float* spectrum, const float* frequencies, int num_freq_bins,
    float max_envelope,
    const pulseqlib_forbidden_band* bands, int num_bands)
{
    int* peaks;
    int i, b;
    float freq, peak_mag, worst_freq, worst_mag;
    int worst_band;

    peaks = NULL;

    violation->detected          = 0;
    violation->band_index        = -1;
    violation->peak_frequency_hz = 0.0f;
    violation->max_amplitude     = max_envelope;
    violation->allowed_amplitude = 0.0f;

    if (num_bands <= 0 || !bands) {
        if (out_peaks) *out_peaks = NULL;
        return PULSEQLIB_OK;
    }

    peaks = (int*)PULSEQLIB_ALLOC((size_t)num_freq_bins * sizeof(int));
    if (!peaks) return PULSEQLIB_ERR_ALLOC_FAILED;

    detect_resonances(peaks, spectrum, num_freq_bins);

    worst_freq = 0.0f; worst_band = -1; worst_mag = 0.0f;
    for (i = 0; i < num_freq_bins; ++i) {
        if (!peaks[i]) continue;
        freq     = frequencies[i];
        peak_mag = spectrum[i];
        for (b = 0; b < num_bands; ++b) {
            if (freq >= bands[b].freq_min_hz && freq <= bands[b].freq_max_hz) {
                if (peak_mag > worst_mag) {
                    worst_mag  = peak_mag;
                    worst_freq = freq;
                    worst_band = b;
                }
                break;
            }
        }
    }

    if (out_peaks) *out_peaks = peaks;
    else           PULSEQLIB_FREE(peaks);

    if (worst_band >= 0) {
        violation->peak_frequency_hz = worst_freq;
        violation->band_index        = worst_band;
        violation->allowed_amplitude = bands[worst_band].max_amplitude;
        if (max_envelope > bands[worst_band].max_amplitude)
            violation->detected = 1;
    }
    return PULSEQLIB_OK;
}

/* ================================================================== */
/*  Sliding window spectra                                            */
/* ================================================================== */

/* FIX: outputs grouped before inputs, out_peaks + out_max_envelope moved up */
static int compute_sliding_window_spectra(
    acoustic_support* sup,
    float* spectra_out,
    float* out_max_envelope,
    int* out_peaks,
    const float* waveform, const float* frequencies,
    int num_samples, int padded_len, int combined,
    const pulseqlib_forbidden_band* bands, int num_bands)
{
    acoustic_waveform aw;
    int w, i, result, start_idx, window_len;
    float* win_spectrum;
    float max_env_win, abs_val, max_env_overall;
    pulseqlib_acoustic_violation viol;
    int* win_peaks;

    win_spectrum = NULL;
    memset(&aw, 0, sizeof(aw));
    max_env_overall = 0.0f;

    if (combined) {
        win_spectrum = (float*)PULSEQLIB_ALLOC(sup->output_freq_bins * sizeof(float));
        if (!win_spectrum) return PULSEQLIB_ERR_ALLOC_FAILED;
        for (i = 0; i < sup->output_freq_bins; ++i) spectra_out[i] = 0.0f;
    }

    result = acoustic_waveform_init(&aw, sup, waveform, num_samples, padded_len);
    if (PULSEQLIB_FAILED(result)) {
        if (win_spectrum) PULSEQLIB_FREE(win_spectrum);
        return result;
    }

    for (w = 0; w < sup->num_windows; ++w) {
        start_idx  = w * sup->hop_size;
        window_len = sup->nwin;
        if (start_idx + window_len > aw.num_samples)
            window_len = aw.num_samples - start_idx;

        max_env_win = 0.0f;
        for (i = start_idx; i < start_idx + window_len; ++i) {
            abs_val = (aw.samples[i] >= 0.0f) ? aw.samples[i] : -aw.samples[i];
            if (abs_val > max_env_win) max_env_win = abs_val;
        }

        if (combined) {
            if (max_env_win > max_env_overall) max_env_overall = max_env_win;
            result = compute_window_spectrum(sup, win_spectrum, &aw, w);
            if (PULSEQLIB_FAILED(result)) {
                PULSEQLIB_FREE(win_spectrum); acoustic_waveform_free(&aw); return result;
            }
            for (i = 0; i < sup->output_freq_bins; ++i)
                if (win_spectrum[i] > spectra_out[i]) spectra_out[i] = win_spectrum[i];
        } else {
            if (out_max_envelope) out_max_envelope[w] = max_env_win;
            result = compute_window_spectrum(sup,
                &spectra_out[w * sup->output_freq_bins], &aw, w);
            if (PULSEQLIB_FAILED(result)) {
                if (win_spectrum) PULSEQLIB_FREE(win_spectrum);
                acoustic_waveform_free(&aw); return result;
            }

            if (num_bands > 0 && bands) {
                win_peaks = NULL;
                result = check_acoustic_violations(&viol, out_peaks ? &win_peaks : NULL,
                    &spectra_out[w * sup->output_freq_bins], frequencies,
                    sup->output_freq_bins, max_env_win, bands, num_bands);
                if (PULSEQLIB_FAILED(result)) {
                    if (win_spectrum) PULSEQLIB_FREE(win_spectrum);
                    acoustic_waveform_free(&aw); return result;
                }
                if (win_peaks && out_peaks) {
                    memcpy(&out_peaks[w * sup->output_freq_bins], win_peaks,
                           sup->output_freq_bins * sizeof(int));
                    PULSEQLIB_FREE(win_peaks);
                }
            } else if (out_peaks) {
                detect_resonances(
                    &out_peaks[w * sup->output_freq_bins],
                    &spectra_out[w * sup->output_freq_bins],
                    sup->output_freq_bins);
            }
        }
    }

    if (combined && out_max_envelope) out_max_envelope[0] = max_env_overall;

    if (win_spectrum) PULSEQLIB_FREE(win_spectrum);
    acoustic_waveform_free(&aw);
    return PULSEQLIB_OK;
}

/* ================================================================== */
/*  Sequence spectrum (full-TR FFT + harmonic sampling)               */
/* ================================================================== */

/* FIX: outputs grouped before inputs */
static int compute_sequence_spectrum(
    float* full_spectrum,
    float** seq_spectrum, float** seq_frequencies,
    int* out_num_freq_bins_full, float* out_freq_res_full,
    int* out_num_picked, float* out_max_envelope,
    int** out_seq_peaks,
    const float* waveform, int num_samples,
    float grad_raster_us, float target_spectral_res_hz,
    float max_frequency, float fundamental_freq, int num_trs,
    const pulseqlib_forbidden_band* bands, int num_bands)
{
    int nfft, nfreq, output_bins_full, num_picked;
    int min_nfft, max_idx;
    float freq_res, max_freq, max_env, abs_val;
    float* work;
    float* cos_win;
    kiss_fft_cpx* fft_out;
    kiss_fftr_cfg cfg;
    float* picked_mag;
    float* picked_freq;
    float mean, fft_norm, norm_factor;
    int i, k, freq_idx;
    float freq, freq_low, freq_high, t;
    float re_low, im_low, re_high, im_high, re_interp, im_interp;
    pulseqlib_acoustic_violation viol;
    int* seq_peaks;
    int result;

    result      = PULSEQLIB_OK;
    work        = NULL;
    cos_win     = NULL;
    fft_out     = NULL;
    cfg         = NULL;
    picked_mag  = NULL;
    picked_freq = NULL;

    if (!waveform || num_samples <= 0) return PULSEQLIB_ERR_INVALID_ARGUMENT;

    max_env = 0.0f;
    for (i = 0; i < num_samples; ++i) {
        abs_val = (waveform[i] >= 0.0f) ? waveform[i] : -waveform[i];
        if (abs_val > max_env) max_env = abs_val;
    }
    if (out_max_envelope) *out_max_envelope = max_env;

    min_nfft = (int)ceil((double)(1.0e6 / (grad_raster_us * target_spectral_res_hz)));
    nfft = (min_nfft < num_samples) ? (int)pulseqlib__next_pow2((size_t)num_samples)
                                    : (int)pulseqlib__next_pow2((size_t)min_nfft);
    nfreq = nfft / 2 + 1;
    freq_res = (float)(1.0e6 / (grad_raster_us * (double)nfft));
    max_freq = (max_frequency > 0.0f) ? max_frequency : (float)(5.0e5 / grad_raster_us);
    max_idx  = (int)(max_freq / freq_res + 0.5f);
    if (max_idx >= nfreq) output_bins_full = nfreq;
    else if (max_idx < 1) output_bins_full = 1;
    else                  output_bins_full = max_idx + 1;

    work    = (float*)PULSEQLIB_ALLOC((size_t)nfft * sizeof(float));
    cos_win = (float*)PULSEQLIB_ALLOC((size_t)num_samples * sizeof(float));
    fft_out = (kiss_fft_cpx*)PULSEQLIB_ALLOC((size_t)nfreq * sizeof(kiss_fft_cpx));
    if (!work || !cos_win || !fft_out) { result = PULSEQLIB_ERR_ALLOC_FAILED; goto fail; }

    cfg = kiss_fftr_alloc(nfft, 0, NULL, NULL);
    if (!cfg) { result = PULSEQLIB_ERR_ALLOC_FAILED; goto fail; }

    for (i = 0; i < num_samples; ++i)
        cos_win[i] = (float)(0.5 * (1.0 - cos(2.0 * M_PI * (double)(i + 1) / (double)num_samples)));

    for (i = 0; i < num_samples; ++i) work[i] = waveform[i];
    for (i = num_samples; i < nfft; ++i) work[i] = 0.0f;

    mean = 0.0f;
    for (i = 0; i < num_samples; ++i) mean += work[i];
    mean /= (float)num_samples;
    for (i = 0; i < num_samples; ++i) work[i] -= mean;
    for (i = 0; i < num_samples; ++i) work[i] *= cos_win[i];

    kiss_fftr(cfg, work, fft_out);
    fft_norm = 1.0f / (float)nfft;
    for (i = 0; i < nfreq; ++i) { fft_out[i].r *= fft_norm; fft_out[i].i *= fft_norm; }

    if (full_spectrum) {
        for (i = 0; i < output_bins_full; ++i)
            full_spectrum[i] = (float)sqrt((double)(fft_out[i].r * fft_out[i].r +
                                                    fft_out[i].i * fft_out[i].i));
    }

    if (fundamental_freq > 0.0f && seq_spectrum && num_trs > 0) {
        num_picked = (int)(max_freq / fundamental_freq) + 1;
        picked_mag = (float*)PULSEQLIB_ALLOC((size_t)num_picked * sizeof(float));
        if (!picked_mag) { result = PULSEQLIB_ERR_ALLOC_FAILED; goto fail; }

        if (seq_frequencies) {
            picked_freq = (float*)PULSEQLIB_ALLOC((size_t)num_picked * sizeof(float));
            if (!picked_freq) { result = PULSEQLIB_ERR_ALLOC_FAILED; goto fail; }
        }

        norm_factor = (num_trs > 1) ? (1.0f / (float)num_trs) : 1.0f;

        for (k = 0; k < num_picked; ++k) {
            freq = (float)k * fundamental_freq;
            if (picked_freq) picked_freq[k] = freq;

            freq_idx = (int)(freq / freq_res);
            if (freq_idx >= nfreq - 1) {
                picked_mag[k] = 0.0f;
            } else if (freq_idx == 0) {
                picked_mag[k] = (float)sqrt((double)(fft_out[0].r * fft_out[0].r +
                                                     fft_out[0].i * fft_out[0].i)) * norm_factor;
            } else {
                freq_low  = (float)freq_idx * freq_res;
                freq_high = (float)(freq_idx + 1) * freq_res;
                re_low  = fft_out[freq_idx].r;     im_low  = fft_out[freq_idx].i;
                re_high = fft_out[freq_idx + 1].r;  im_high = fft_out[freq_idx + 1].i;
                t = (freq - freq_low) / (freq_high - freq_low);
                re_interp = re_low * (1.0f - t) + re_high * t;
                im_interp = im_low * (1.0f - t) + im_high * t;
                picked_mag[k] = (float)sqrt((double)(re_interp * re_interp +
                                                     im_interp * im_interp)) * norm_factor;
            }
        }

        *seq_spectrum = picked_mag;   picked_mag = NULL;
        if (seq_frequencies) { *seq_frequencies = picked_freq; picked_freq = NULL; }
        if (out_num_picked) *out_num_picked = num_picked;
    }

    if (out_num_freq_bins_full) *out_num_freq_bins_full = output_bins_full;
    if (out_freq_res_full)      *out_freq_res_full      = freq_res;

    if (num_bands > 0 && bands && seq_spectrum && *seq_spectrum) {
        num_picked = out_num_picked ? *out_num_picked : 0;
        result = check_acoustic_violations(&viol, out_seq_peaks,
            *seq_spectrum, (seq_frequencies && *seq_frequencies) ? *seq_frequencies : NULL,
            num_picked, max_env, bands, num_bands);
        if (PULSEQLIB_FAILED(result)) goto fail;
    } else if (out_seq_peaks && seq_spectrum && *seq_spectrum) {
        num_picked = out_num_picked ? *out_num_picked : 0;
        seq_peaks = (int*)PULSEQLIB_ALLOC((size_t)num_picked * sizeof(int));
        if (!seq_peaks) { result = PULSEQLIB_ERR_ALLOC_FAILED; goto fail; }
        detect_resonances(seq_peaks, *seq_spectrum, num_picked);
        *out_seq_peaks = seq_peaks;
    }

fail:
    if (work)        PULSEQLIB_FREE(work);
    if (cos_win)     PULSEQLIB_FREE(cos_win);
    if (fft_out)     PULSEQLIB_FREE(fft_out);
    if (cfg)         kiss_fftr_free(cfg);
    if (picked_mag)  PULSEQLIB_FREE(picked_mag);
    if (picked_freq) PULSEQLIB_FREE(picked_freq);
    return result;
}

/* ================================================================== */
/*  get_tr_acoustic_spectra                                           */
/* ================================================================== */

int pulseqlib_get_tr_acoustic_spectra(
    pulseqlib_tr_acoustic_spectra* spectra,
    pulseqlib_diagnostic* diag,
    const pulseqlib_tr_gradient_waveforms* waveforms,
    float grad_raster_time_us,
    int target_window_size,
    float target_spectral_resolution_hz,
    float max_frequency_hz,
    int combined,
    int num_trs,
    float tr_duration_us,
    int num_forbidden_bands,
    const pulseqlib_forbidden_band* forbidden_bands,
    int store_results)
{
    acoustic_support sup;
    pulseqlib_diagnostic local_diag;
    int max_samples, result, output_size, padded_len, i;
    float fundamental_freq;
    int num_freq_bins_full, num_freq_bins_seq;
    float freq_res_full;
    float* seq_spec_gx;
    float* seq_spec_gy;
    float* seq_spec_gz;
    float* seq_freqs;

    seq_spec_gx = NULL;
    seq_spec_gy = NULL;
    seq_spec_gz = NULL;
    seq_freqs   = NULL;

    if (!diag) { pulseqlib_diagnostic_init(&local_diag); diag = &local_diag; }
    else       { pulseqlib_diagnostic_init(diag); }

    if (!waveforms || !spectra) {
        diag->code = PULSEQLIB_ERR_NULL_POINTER; return diag->code;
    }

    memset(spectra, 0, sizeof(*spectra));
    memset(&sup, 0, sizeof(sup));

    max_samples = waveforms->num_samples;
    if (max_samples <= 0) { diag->code = PULSEQLIB_ERR_INVALID_ARGUMENT; return diag->code; }

    result = acoustic_support_init(&sup, max_samples, target_window_size,
                                   target_spectral_resolution_hz,
                                   grad_raster_time_us, max_frequency_hz);
    if (PULSEQLIB_FAILED(result)) { diag->code = result; return result; }

    spectra->combined       = combined;
    spectra->num_windows    = combined ? 1 : sup.num_windows;
    spectra->num_freq_bins  = sup.output_freq_bins;
    spectra->freq_resolution = sup.freq_resolution;

    output_size = combined ? sup.output_freq_bins : sup.num_windows * sup.output_freq_bins;

    if (store_results) {
        spectra->spectra_gx      = (float*)PULSEQLIB_ALLOC((size_t)output_size * sizeof(float));
        spectra->spectra_gy      = (float*)PULSEQLIB_ALLOC((size_t)output_size * sizeof(float));
        spectra->spectra_gz      = (float*)PULSEQLIB_ALLOC((size_t)output_size * sizeof(float));
        spectra->frequencies     = (float*)PULSEQLIB_ALLOC((size_t)sup.output_freq_bins * sizeof(float));
        spectra->max_envelope_gx = (float*)PULSEQLIB_ALLOC((size_t)spectra->num_windows * sizeof(float));
        spectra->max_envelope_gy = (float*)PULSEQLIB_ALLOC((size_t)spectra->num_windows * sizeof(float));
        spectra->max_envelope_gz = (float*)PULSEQLIB_ALLOC((size_t)spectra->num_windows * sizeof(float));

        if (!combined) {
            spectra->peaks_gx = (int*)PULSEQLIB_ALLOC((size_t)output_size * sizeof(int));
            spectra->peaks_gy = (int*)PULSEQLIB_ALLOC((size_t)output_size * sizeof(int));
            spectra->peaks_gz = (int*)PULSEQLIB_ALLOC((size_t)output_size * sizeof(int));
            if (!spectra->peaks_gx || !spectra->peaks_gy || !spectra->peaks_gz) {
                pulseqlib_tr_acoustic_spectra_free(spectra);
                acoustic_support_free(&sup);
                diag->code = PULSEQLIB_ERR_ALLOC_FAILED; return diag->code;
            }
        }

        if (!spectra->spectra_gx || !spectra->spectra_gy || !spectra->spectra_gz ||
            !spectra->frequencies ||
            !spectra->max_envelope_gx || !spectra->max_envelope_gy || !spectra->max_envelope_gz) {
            pulseqlib_tr_acoustic_spectra_free(spectra);
            acoustic_support_free(&sup);
            diag->code = PULSEQLIB_ERR_ALLOC_FAILED; return diag->code;
        }

        memset(spectra->spectra_gx, 0, (size_t)output_size * sizeof(float));
        memset(spectra->spectra_gy, 0, (size_t)output_size * sizeof(float));
        memset(spectra->spectra_gz, 0, (size_t)output_size * sizeof(float));
        memset(spectra->max_envelope_gx, 0, (size_t)spectra->num_windows * sizeof(float));
        memset(spectra->max_envelope_gy, 0, (size_t)spectra->num_windows * sizeof(float));
        memset(spectra->max_envelope_gz, 0, (size_t)spectra->num_windows * sizeof(float));
        if (!combined) {
            memset(spectra->peaks_gx, 0, (size_t)output_size * sizeof(int));
            memset(spectra->peaks_gy, 0, (size_t)output_size * sizeof(int));
            memset(spectra->peaks_gz, 0, (size_t)output_size * sizeof(int));
        }

        for (i = 0; i < sup.output_freq_bins; ++i)
            spectra->frequencies[i] = (float)i * sup.freq_resolution;
    }

    /* padded length */
    padded_len = (max_samples <= sup.nwin) ? max_samples
                 : ((max_samples + sup.nwin - 1) / sup.nwin) * sup.nwin;

    /* sliding window: Gx */
    if (waveforms->num_samples > 0) {
        result = compute_sliding_window_spectra(&sup,
            store_results ? spectra->spectra_gx : NULL,
            store_results ? spectra->max_envelope_gx : NULL,
            store_results ? spectra->peaks_gx : NULL,
            waveforms->waveform_gx, spectra->frequencies,
            waveforms->num_samples, padded_len, combined,
            forbidden_bands, num_forbidden_bands);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_tr_acoustic_spectra_free(spectra);
            acoustic_support_free(&sup);
            diag->code = result; return result;
        }
    }
    /* sliding window: Gy */
    if (waveforms->num_samples > 0) {
        result = compute_sliding_window_spectra(&sup,
            store_results ? spectra->spectra_gy : NULL,
            store_results ? spectra->max_envelope_gy : NULL,
            store_results ? spectra->peaks_gy : NULL,
            waveforms->waveform_gy, spectra->frequencies,
            waveforms->num_samples, padded_len, combined,
            forbidden_bands, num_forbidden_bands);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_tr_acoustic_spectra_free(spectra);
            acoustic_support_free(&sup);
            diag->code = result; return result;
        }
    }
    /* sliding window: Gz */
    if (waveforms->num_samples > 0) {
        result = compute_sliding_window_spectra(&sup,
            store_results ? spectra->spectra_gz : NULL,
            store_results ? spectra->max_envelope_gz : NULL,
            store_results ? spectra->peaks_gz : NULL,
            waveforms->waveform_gz, spectra->frequencies,
            waveforms->num_samples, padded_len, combined,
            forbidden_bands, num_forbidden_bands);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_tr_acoustic_spectra_free(spectra);
            acoustic_support_free(&sup);
            diag->code = result; return result;
        }
    }

    acoustic_support_free(&sup);

    /* ---- sequence spectra ---- */
    if (num_trs > 1 && tr_duration_us > 0.0f) {
        fundamental_freq = 1.0e6f / tr_duration_us;
        spectra->num_trs        = num_trs;
        spectra->tr_duration_us = tr_duration_us;
        spectra->fundamental_freq = fundamental_freq;
    } else {
        fundamental_freq = 0.0f;
        spectra->num_trs        = 1;
        spectra->tr_duration_us = tr_duration_us;
        spectra->fundamental_freq = 0.0f;
    }

    /* Gx (first call: determines sizes) */
    if (waveforms->num_samples > 0) {
        result = compute_sequence_spectrum(NULL,
            (store_results && fundamental_freq > 0.0f) ? &seq_spec_gx : NULL,
            (store_results && fundamental_freq > 0.0f) ? &seq_freqs   : NULL,
            &num_freq_bins_full, &freq_res_full, &num_freq_bins_seq, NULL,
            (store_results && fundamental_freq > 0.0f) ? &spectra->peaks_gx_seq : NULL,
            waveforms->waveform_gx, waveforms->num_samples,
            grad_raster_time_us, target_spectral_resolution_hz,
            max_frequency_hz, fundamental_freq, num_trs,
            forbidden_bands, num_forbidden_bands);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_tr_acoustic_spectra_free(spectra);
            diag->code = result; return result;
        }
    } else {
        result = compute_sequence_spectrum(NULL, NULL, NULL,
            &num_freq_bins_full, &freq_res_full, NULL, NULL, NULL,
            waveforms->waveform_gy ? waveforms->waveform_gy : waveforms->waveform_gz,
            max_samples, grad_raster_time_us, target_spectral_resolution_hz,
            max_frequency_hz, 0.0f, num_trs,
            forbidden_bands, num_forbidden_bands);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_tr_acoustic_spectra_free(spectra);
            diag->code = result; return result;
        }
    }

    spectra->num_freq_bins_full  = num_freq_bins_full;
    spectra->freq_resolution_full = freq_res_full;

    if (store_results) {
        spectra->spectra_gx_full  = (float*)PULSEQLIB_ALLOC((size_t)num_freq_bins_full * sizeof(float));
        spectra->spectra_gy_full  = (float*)PULSEQLIB_ALLOC((size_t)num_freq_bins_full * sizeof(float));
        spectra->spectra_gz_full  = (float*)PULSEQLIB_ALLOC((size_t)num_freq_bins_full * sizeof(float));
        spectra->frequencies_full = (float*)PULSEQLIB_ALLOC((size_t)num_freq_bins_full * sizeof(float));
        if (!spectra->spectra_gx_full || !spectra->spectra_gy_full ||
            !spectra->spectra_gz_full || !spectra->frequencies_full) {
            if (seq_spec_gx) PULSEQLIB_FREE(seq_spec_gx);
            if (seq_freqs)   PULSEQLIB_FREE(seq_freqs);
            pulseqlib_tr_acoustic_spectra_free(spectra);
            diag->code = PULSEQLIB_ERR_ALLOC_FAILED; return diag->code;
        }
        for (i = 0; i < num_freq_bins_full; ++i)
            spectra->frequencies_full[i] = (float)i * freq_res_full;

        if (fundamental_freq > 0.0f && num_freq_bins_seq > 0) {
            spectra->num_freq_bins_seq = num_freq_bins_seq;
            spectra->spectra_gx_seq = (float*)PULSEQLIB_ALLOC((size_t)num_freq_bins_seq * sizeof(float));
            spectra->spectra_gy_seq = (float*)PULSEQLIB_ALLOC((size_t)num_freq_bins_seq * sizeof(float));
            spectra->spectra_gz_seq = (float*)PULSEQLIB_ALLOC((size_t)num_freq_bins_seq * sizeof(float));
            spectra->frequencies_seq = seq_freqs; seq_freqs = NULL;
            if (!spectra->spectra_gx_seq || !spectra->spectra_gy_seq || !spectra->spectra_gz_seq) {
                if (seq_spec_gx) PULSEQLIB_FREE(seq_spec_gx);
                pulseqlib_tr_acoustic_spectra_free(spectra);
                diag->code = PULSEQLIB_ERR_ALLOC_FAILED; return diag->code;
            }
            if (seq_spec_gx && waveforms->num_samples > 0) {
                memcpy(spectra->spectra_gx_seq, seq_spec_gx, (size_t)num_freq_bins_seq * sizeof(float));
                PULSEQLIB_FREE(seq_spec_gx); seq_spec_gx = NULL;
            } else {
                memset(spectra->spectra_gx_seq, 0, (size_t)num_freq_bins_seq * sizeof(float));
            }
        }
    } else {
        if (seq_spec_gx) { PULSEQLIB_FREE(seq_spec_gx); seq_spec_gx = NULL; }
        if (seq_freqs)   { PULSEQLIB_FREE(seq_freqs);   seq_freqs = NULL; }
    }

    /* full-TR spectrum: Gx */
    if (waveforms->num_samples > 0) {
        result = compute_sequence_spectrum(
            store_results ? spectra->spectra_gx_full : NULL, NULL, NULL,
            NULL, NULL, NULL, &spectra->max_envelope_gx_full, NULL,
            waveforms->waveform_gx, waveforms->num_samples,
            grad_raster_time_us, target_spectral_resolution_hz,
            max_frequency_hz, 0.0f, num_trs,
            forbidden_bands, num_forbidden_bands);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_tr_acoustic_spectra_free(spectra);
            diag->code = result; return result;
        }
    } else if (store_results) {
        memset(spectra->spectra_gx_full, 0, (size_t)num_freq_bins_full * sizeof(float));
    }

    /* Gy */
    if (waveforms->num_samples > 0) {
        result = compute_sequence_spectrum(
            store_results ? spectra->spectra_gy_full : NULL,
            (store_results && fundamental_freq > 0.0f) ? &seq_spec_gy : NULL,
            NULL,
            NULL, NULL, NULL, &spectra->max_envelope_gy_full,
            (store_results && fundamental_freq > 0.0f) ? &spectra->peaks_gy_seq : NULL,
            waveforms->waveform_gy, waveforms->num_samples,
            grad_raster_time_us, target_spectral_resolution_hz,
            max_frequency_hz, fundamental_freq, num_trs,
            forbidden_bands, num_forbidden_bands);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_tr_acoustic_spectra_free(spectra);
            diag->code = result; return result;
        }
        if (store_results && seq_spec_gy && spectra->spectra_gy_seq) {
            memcpy(spectra->spectra_gy_seq, seq_spec_gy, (size_t)num_freq_bins_seq * sizeof(float));
            PULSEQLIB_FREE(seq_spec_gy); seq_spec_gy = NULL;
        } else if (seq_spec_gy) { PULSEQLIB_FREE(seq_spec_gy); seq_spec_gy = NULL; }
    } else if (store_results) {
        memset(spectra->spectra_gy_full, 0, (size_t)num_freq_bins_full * sizeof(float));
        if (spectra->spectra_gy_seq)
            memset(spectra->spectra_gy_seq, 0, (size_t)num_freq_bins_seq * sizeof(float));
        if (spectra->peaks_gy_seq)
            memset(spectra->peaks_gy_seq, 0, (size_t)num_freq_bins_seq * sizeof(int));
    }

    /* Gz */
    if (waveforms->num_samples > 0) {
        result = compute_sequence_spectrum(
            store_results ? spectra->spectra_gz_full : NULL,
            (store_results && fundamental_freq > 0.0f) ? &seq_spec_gz : NULL,
            NULL,
            NULL, NULL, NULL, &spectra->max_envelope_gz_full,
            (store_results && fundamental_freq > 0.0f) ? &spectra->peaks_gz_seq : NULL,
            waveforms->waveform_gz, waveforms->num_samples,
            grad_raster_time_us, target_spectral_resolution_hz,
            max_frequency_hz, fundamental_freq, num_trs,
            forbidden_bands, num_forbidden_bands);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_tr_acoustic_spectra_free(spectra);
            diag->code = result; return result;
        }
        if (store_results && seq_spec_gz && spectra->spectra_gz_seq) {
            memcpy(spectra->spectra_gz_seq, seq_spec_gz, (size_t)num_freq_bins_seq * sizeof(float));
            PULSEQLIB_FREE(seq_spec_gz); seq_spec_gz = NULL;
        } else if (seq_spec_gz) { PULSEQLIB_FREE(seq_spec_gz); seq_spec_gz = NULL; }
    } else if (store_results) {
        memset(spectra->spectra_gz_full, 0, (size_t)num_freq_bins_full * sizeof(float));
        if (spectra->spectra_gz_seq)
            memset(spectra->spectra_gz_seq, 0, (size_t)num_freq_bins_seq * sizeof(float));
        if (spectra->peaks_gz_seq)
            memset(spectra->peaks_gz_seq, 0, (size_t)num_freq_bins_seq * sizeof(int));
    }

    diag->code = PULSEQLIB_OK;
    return PULSEQLIB_OK;
}

#else /* !PULSEQLIB_VENDOR_GEHC */

int pulseqlib_get_tr_acoustic_spectra(
    pulseqlib_tr_acoustic_spectra* spectra,
    pulseqlib_diagnostic* diag,
    const pulseqlib_tr_gradient_waveforms* waveforms,
    float grad_raster_time_us,
    int target_window_size,
    float target_spectral_resolution_hz,
    float max_frequency_hz,
    int combined,
    int num_trs,
    float tr_duration_us,
    int num_forbidden_bands,
    const pulseqlib_forbidden_band* forbidden_bands,
    int store_results)
{
    (void)spectra; (void)waveforms; (void)grad_raster_time_us;
    (void)target_window_size; (void)target_spectral_resolution_hz;
    (void)max_frequency_hz; (void)combined; (void)num_trs;
    (void)tr_duration_us; (void)num_forbidden_bands;
    (void)forbidden_bands; (void)store_results;
    if (diag) { pulseqlib_diagnostic_init(diag); diag->code = PULSEQLIB_ERR_NOT_IMPLEMENTED; }
    return PULSEQLIB_ERR_NOT_IMPLEMENTED;
}

#endif /* PULSEQLIB_VENDOR_GEHC */

/* ================================================================== */
/*  PNS                                                               */
/* ================================================================== */

#if PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_GEHC

/* FIX: outputs before inputs */
static int build_pns_kernel(
    float** kernel, int* kernel_len,
    float dt_us, const pulseqlib_pns_params* params)
{
    int n, i;
    float c_s, dt_s, s_min, tau, denom;
    float* k;

    if (params->chronaxie_us <= 0.0f) return PULSEQLIB_ERR_PNS_INVALID_CHRONAXIE;
    if (params->rheobase     <= 0.0f) return PULSEQLIB_ERR_PNS_INVALID_RHEOBASE;
    if (params->alpha        <= 0.0f) return PULSEQLIB_ERR_PNS_INVALID_PARAMS;

    c_s  = params->chronaxie_us * 1e-6f;
    dt_s = dt_us * 1e-6f;
    s_min = params->rheobase / params->alpha;

    n = (int)(PNS_KERNEL_DURATION_FACTOR * c_s / dt_s) + 1;
    k = (float*)PULSEQLIB_ALLOC((size_t)n * sizeof(float));
    if (!k) return PULSEQLIB_ERR_ALLOC_FAILED;

    for (i = 0; i < n; ++i) {
        tau   = (float)i * dt_s;
        denom = (c_s + tau) * (c_s + tau);
        k[i]  = (dt_s / s_min) * (c_s / denom);
    }

    *kernel     = k;
    *kernel_len = n;
    return PULSEQLIB_OK;
}

/* FIX: output before inputs, C89-compliant declarations */
static void compute_slew_rate(
    float* slew_out,
    const float* waveform, int num_samples,
    float dt_us, float gamma_hz_per_tesla)
{
    int i;
    float dt_s;
    float inv_g;

    dt_s = dt_us * 1e-6f;
    inv_g = 1.0f / gamma_hz_per_tesla;

    for (i = 0; i < num_samples - 1; ++i)
        slew_out[i] = ((waveform[i + 1] - waveform[i]) * inv_g) / dt_s;
}

/* FIX: outputs then in-out then scratch then inputs */
static int process_pns_axis_circular(
    float* pns_axis, float* pns_total,
    float* pns_store,
    float* padded_waveform, float* slew_rate, float* pns_conv,
    const float* waveform, int num_samples,
    const float* kernel, int kernel_len,
    float grad_raster_us, float gamma_hz_per_tesla,
    int full_output_len)
{
    int i, padded_len, slew_len, rc;

    (void)full_output_len;

    if (num_samples <= 0 || !waveform) return PULSEQLIB_OK;

    padded_len = num_samples + kernel_len;
    slew_len   = padded_len - 1;

    for (i = 0; i < num_samples; ++i)
        padded_waveform[i] = waveform[i];
    for (i = 0; i < kernel_len; ++i)
        padded_waveform[num_samples + i] = waveform[i % num_samples];

    compute_slew_rate(slew_rate, padded_waveform, padded_len, grad_raster_us, gamma_hz_per_tesla);

    rc = pulseqlib__convolve_fft(pns_conv, slew_rate, slew_len, kernel, kernel_len);
    if (PULSEQLIB_FAILED(rc)) return rc;

    for (i = 0; i < slew_len; ++i) {
        pns_axis[i]   = pns_conv[i] * 100.0f;
        pns_total[i] += pns_conv[i] * pns_conv[i];
    }

    if (pns_store) {
        for (i = 0; i < slew_len; ++i) pns_store[i] = pns_axis[i];
    }
    return PULSEQLIB_OK;
}

int pulseqlib_compute_pns(
    pulseqlib_pns_result* result,
    pulseqlib_diagnostic* diag,
    float gamma_hz_per_tesla,
    float pns_threshold,
    const pulseqlib_tr_gradient_waveforms* waveforms,
    float grad_raster_time_us,
    const pulseqlib_pns_params* params,
    int store_waveforms)
{
    pulseqlib_diagnostic local_diag;
    int max_samples, padded_len, slew_len, full_output_len;
    int kernel_len, i;
    float* kernel;
    float* padded;
    float* slew;
    float* conv;
    float* axis;
    float* pns_x;
    float* pns_y;
    float* pns_z;
    float* pns_tot;
    float max_pns;
    int max_idx;
    int rc;

    kernel  = NULL;
    padded  = NULL;
    slew    = NULL;
    conv    = NULL;
    axis    = NULL;
    pns_x   = NULL;
    pns_y   = NULL;
    pns_z   = NULL;
    pns_tot = NULL;
    rc      = PULSEQLIB_OK;

    if (!diag) { pulseqlib_diagnostic_init(&local_diag); diag = &local_diag; }
    else       { pulseqlib_diagnostic_init(diag); }

    if (!waveforms || !params || !result) {
        diag->code = PULSEQLIB_ERR_NULL_POINTER; return diag->code;
    }

    memset(result, 0, sizeof(*result));

    max_samples = waveforms->num_samples;
    if (max_samples <= 1) { diag->code = PULSEQLIB_ERR_PNS_NO_WAVEFORM; return diag->code; }

    rc = build_pns_kernel(&kernel, &kernel_len, grad_raster_time_us, params);
    if (PULSEQLIB_FAILED(rc)) { diag->code = rc; return rc; }

    padded_len      = max_samples + kernel_len;
    slew_len        = padded_len - 1;
    full_output_len = slew_len;

    padded  = (float*)PULSEQLIB_ALLOC((size_t)padded_len * sizeof(float));
    slew    = (float*)PULSEQLIB_ALLOC((size_t)slew_len * sizeof(float));
    conv    = (float*)PULSEQLIB_ALLOC((size_t)slew_len * sizeof(float));
    axis    = (float*)PULSEQLIB_ALLOC((size_t)full_output_len * sizeof(float));
    pns_tot = (float*)PULSEQLIB_ALLOC((size_t)full_output_len * sizeof(float));
    if (!padded || !slew || !conv || !axis || !pns_tot) {
        rc = PULSEQLIB_ERR_ALLOC_FAILED; goto fail;
    }
    for (i = 0; i < full_output_len; ++i) pns_tot[i] = 0.0f;

    if (store_waveforms) {
        pns_x = (float*)PULSEQLIB_ALLOC((size_t)full_output_len * sizeof(float));
        pns_y = (float*)PULSEQLIB_ALLOC((size_t)full_output_len * sizeof(float));
        pns_z = (float*)PULSEQLIB_ALLOC((size_t)full_output_len * sizeof(float));
        if (!pns_x || !pns_y || !pns_z) { rc = PULSEQLIB_ERR_ALLOC_FAILED; goto fail; }
        for (i = 0; i < full_output_len; ++i) { pns_x[i] = 0.0f; pns_y[i] = 0.0f; pns_z[i] = 0.0f; }
    }

    /* X */
    rc = process_pns_axis_circular(axis, pns_tot,
        store_waveforms ? pns_x : NULL,
        padded, slew, conv,
        waveforms->waveform_gx, waveforms->num_samples,
        kernel, kernel_len,
        grad_raster_time_us, gamma_hz_per_tesla,
        full_output_len);
    if (PULSEQLIB_FAILED(rc)) goto fail;

    /* Y */
    rc = process_pns_axis_circular(axis, pns_tot,
        store_waveforms ? pns_y : NULL,
        padded, slew, conv,
        waveforms->waveform_gy, waveforms->num_samples,
        kernel, kernel_len,
        grad_raster_time_us, gamma_hz_per_tesla,
        full_output_len);
    if (PULSEQLIB_FAILED(rc)) goto fail;

    /* Z */
    rc = process_pns_axis_circular(axis, pns_tot,
        store_waveforms ? pns_z : NULL,
        padded, slew, conv,
        waveforms->waveform_gz, waveforms->num_samples,
        kernel, kernel_len,
        grad_raster_time_us, gamma_hz_per_tesla,
        full_output_len);
    if (PULSEQLIB_FAILED(rc)) goto fail;

    max_pns = 0.0f; max_idx = 0;
    for (i = 0; i < full_output_len; ++i) {
        pns_tot[i] = 100.0f * (float)sqrt((double)pns_tot[i]);
        if (pns_tot[i] > max_pns) { max_pns = pns_tot[i]; max_idx = i; }
    }

    result->num_samples    = full_output_len;
    result->pns_total      = pns_tot; pns_tot = NULL;
    result->max_pns        = max_pns;
    result->max_pns_index  = max_idx;
    result->max_pns_time_us = (float)max_idx * grad_raster_time_us;

    if (store_waveforms) {
        result->pns_x = pns_x; pns_x = NULL;
        result->pns_y = pns_y; pns_y = NULL;
        result->pns_z = pns_z; pns_z = NULL;
    }

    if (!store_waveforms && max_pns > pns_threshold)
        rc = PULSEQLIB_ERR_PNS_THRESHOLD_EXCEEDED;
    else
        rc = PULSEQLIB_OK;
    diag->code = rc;

fail:
    if (kernel)  PULSEQLIB_FREE(kernel);
    if (padded)  PULSEQLIB_FREE(padded);
    if (slew)    PULSEQLIB_FREE(slew);
    if (conv)    PULSEQLIB_FREE(conv);
    if (axis)    PULSEQLIB_FREE(axis);
    if (pns_tot) PULSEQLIB_FREE(pns_tot);
    if (pns_x)   PULSEQLIB_FREE(pns_x);
    if (pns_y)   PULSEQLIB_FREE(pns_y);
    if (pns_z)   PULSEQLIB_FREE(pns_z);
    return rc;
}

#else /* !PULSEQLIB_VENDOR_GEHC */

int pulseqlib_compute_pns(
    pulseqlib_pns_result* result,
    pulseqlib_diagnostic* diag,
    float gamma_hz_per_tesla,
    float pns_threshold,
    const pulseqlib_tr_gradient_waveforms* waveforms,
    float grad_raster_time_us,
    const pulseqlib_pns_params* params,
    int store_waveforms)
{
    (void)result; (void)gamma_hz_per_tesla; (void)pns_threshold;
    (void)waveforms; (void)grad_raster_time_us; (void)params; (void)store_waveforms;
    if (diag) { pulseqlib_diagnostic_init(diag); diag->code = PULSEQLIB_ERR_NOT_IMPLEMENTED; }
    return PULSEQLIB_ERR_NOT_IMPLEMENTED;
}

#endif /* PULSEQLIB_VENDOR_GEHC */

/* ================================================================== */
/*  PNS result free                                                   */
/* ================================================================== */

void pulseqlib_pns_result_free(pulseqlib_pns_result* r)
{
    if (!r) return;
    if (r->pns_x)     { PULSEQLIB_FREE(r->pns_x);     r->pns_x = NULL; }
    if (r->pns_y)     { PULSEQLIB_FREE(r->pns_y);     r->pns_y = NULL; }
    if (r->pns_z)     { PULSEQLIB_FREE(r->pns_z);     r->pns_z = NULL; }
    if (r->pns_total) { PULSEQLIB_FREE(r->pns_total); r->pns_total = NULL; }
    r->num_samples    = 0;
    r->max_pns        = 0.0f;
    r->max_pns_index  = 0;
    r->max_pns_time_us = 0.0f;
}

/* ================================================================== */
/*  Collection-level safety check                                     */
/* ================================================================== */

int check_max_grad(
    const pulseqlib_collection* coll,
    pulseqlib_diagnostic* diag,
    const pulseqlib_opts* opts
) {
    int s, b, raw_id;
    int worst_subseq, worst_block;
    float gx_amp, gy_amp, gz_amp, gsos, gsos_max, limit_sq, hz_per_mt;
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_block_table_element* bte;

    if (!coll || !opts) {
        if (diag) { pulseqlib_diagnostic_init(diag); diag->code = PULSEQLIB_ERR_NULL_POINTER; }
        return PULSEQLIB_ERR_NULL_POINTER;
    }
    if (diag) pulseqlib_diagnostic_init(diag);

    /* ---- max gradient amplitude (GSOS) check ---- */
    gsos_max     = 0.0f;
    limit_sq     = opts->max_grad * opts->max_grad;
    worst_subseq = 0;
    worst_block  = 0;

    for (s = 0; s < coll->num_subsequences; ++s) {
        desc = &coll->descriptors[s];
        for (b = 0; b < desc->num_blocks; ++b) {
            bte = &desc->block_table[b];

            gx_amp = 0.0f;
            raw_id = bte->gx_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size)
                gx_amp = desc->grad_table[raw_id].amplitude;

            gy_amp = 0.0f;
            raw_id = bte->gy_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size)
                gy_amp = desc->grad_table[raw_id].amplitude;

            gz_amp = 0.0f;
            raw_id = bte->gz_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size)
                gz_amp = desc->grad_table[raw_id].amplitude;

            gsos = gx_amp * gx_amp + gy_amp * gy_amp + gz_amp * gz_amp;
            if (gsos > gsos_max) {
                gsos_max     = gsos;
                worst_subseq = s;
                worst_block  = b;
            }
        }
    }

    if (limit_sq > 0.0f && gsos_max > limit_sq) {
        hz_per_mt = opts->gamma * 0.001f;
        if (diag) {
            diag->code                = PULSEQLIB_ERR_MAX_GRAD_EXCEEDED;
            diag->channel             = worst_subseq;
            diag->block_index         = worst_block;
            diag->gradient_amplitude  = (float)sqrt((double)gsos_max) / hz_per_mt;
            diag->max_allowed_amplitude = opts->max_grad / hz_per_mt;
        }
        return PULSEQLIB_ERR_MAX_GRAD_EXCEEDED;
    }

    return PULSEQLIB_OK;
}

/* ================================================================== */
/*  Gradient continuity check (cursor dry-run over n repetitions)     */
/* ================================================================== */

int check_grad_continuity(
    pulseqlib_collection* coll,
    pulseqlib_diagnostic* diag,
    const pulseqlib_opts* opts)
{
    pulseqlib_block_cursor saved_cursor;
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_block_table_element* bte;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_grad_definition* gdef;
    const pulseqlib_grad_table_element* gte;
    int n, raw_id, rot_id, status, cur_seq;
    int grad_def_ids[3];
    int shot_idx[3];
    float amp[3], first_val[3], last_val[3];
    float first_phys[3], last_phys[3], prev_phys[3];
    float max_allowed, grad_raster_s, step, hz_per_mt;

    if (!coll || !opts) {
        if (diag) { pulseqlib_diagnostic_init(diag); diag->code = PULSEQLIB_ERR_NULL_POINTER; }
        return PULSEQLIB_ERR_NULL_POINTER;
    }
    if (diag) pulseqlib_diagnostic_init(diag);

    /* save cursor state */
    saved_cursor = coll->block_cursor;
    coll->block_cursor.current_repetition = 0;
    coll->block_cursor.sequence_index = 0;
    coll->block_cursor.within_sequence_block_index = 0;
    coll->block_cursor.from_last_reset = 0;

    prev_phys[0] = 0.0f;
    prev_phys[1] = 0.0f;
    prev_phys[2] = 0.0f;
    cur_seq = 0;
    status  = PULSEQLIB_CURSOR_BLOCK;

    desc = &coll->descriptors[0];
    grad_raster_s = desc->grad_raster_time_us * 1e-6f;
    max_allowed   = opts->max_slew * grad_raster_s;

    while (status != PULSEQLIB_CURSOR_DONE) {
        /* detect subsequence change */
        if (coll->block_cursor.sequence_index != cur_seq) {
            /* end-of-subsequence: prev must ramp to zero */
            for (n = 0; n < 3; ++n) {
                step = prev_phys[n];
                if (step < 0.0f) step = -step;
                if (step > max_allowed) {
                    hz_per_mt = opts->gamma * 0.001f;
                    if (diag) {
                        diag->code                  = PULSEQLIB_ERR_GRAD_DISCONTINUITY;
                        diag->channel               = n;
                        diag->block_index           = -1;
                        diag->gradient_amplitude    = step / hz_per_mt;
                        diag->max_allowed_amplitude = max_allowed / hz_per_mt;
                    }
                    coll->block_cursor = saved_cursor;
                    return PULSEQLIB_ERR_GRAD_DISCONTINUITY;
                }
            }

            cur_seq = coll->block_cursor.sequence_index;
            prev_phys[0] = 0.0f;
            prev_phys[1] = 0.0f;
            prev_phys[2] = 0.0f;

            desc = &coll->descriptors[cur_seq];
            grad_raster_s = desc->grad_raster_time_us * 1e-6f;
            max_allowed   = opts->max_slew * grad_raster_s;
        }

        /* read current block */
        bte  = &desc->block_table[coll->block_cursor.within_sequence_block_index];
        bdef = &desc->block_definitions[bte->id];

        /* grad table: amplitude + shot_index */
        grad_def_ids[0] = bdef->gx_id;
        grad_def_ids[1] = bdef->gy_id;
        grad_def_ids[2] = bdef->gz_id;

        raw_id = bte->gx_id;
        if (raw_id >= 0 && raw_id < desc->grad_table_size) {
            gte = &desc->grad_table[raw_id];
            amp[0] = gte->amplitude; shot_idx[0] = gte->shot_index;
        } else { amp[0] = 0.0f; shot_idx[0] = 0; }

        raw_id = bte->gy_id;
        if (raw_id >= 0 && raw_id < desc->grad_table_size) {
            gte = &desc->grad_table[raw_id];
            amp[1] = gte->amplitude; shot_idx[1] = gte->shot_index;
        } else { amp[1] = 0.0f; shot_idx[1] = 0; }

        raw_id = bte->gz_id;
        if (raw_id >= 0 && raw_id < desc->grad_table_size) {
            gte = &desc->grad_table[raw_id];
            amp[2] = gte->amplitude; shot_idx[2] = gte->shot_index;
        } else { amp[2] = 0.0f; shot_idx[2] = 0; }

        /* first_value / last_value from grad definitions, scaled by amplitude */
        for (n = 0; n < 3; ++n) {
            if (grad_def_ids[n] >= 0 && grad_def_ids[n] < desc->num_unique_grads) {
                gdef = &desc->grad_definitions[grad_def_ids[n]];
                first_val[n] = gdef->first_value[shot_idx[n]] * amp[n];
                last_val[n]  = gdef->last_value[shot_idx[n]]  * amp[n];
            } else {
                first_val[n] = 0.0f;
                last_val[n]  = 0.0f;
            }
        }

        /* transform logical -> physical */
        rot_id = bte->rotation_id;
        if (rot_id >= 0 && rot_id < desc->num_rotations) {
            pulseqlib__apply_rotation(first_phys, desc->rotation_matrices[rot_id], first_val, 1);
            pulseqlib__apply_rotation(last_phys,  desc->rotation_matrices[rot_id], last_val,  1);
        } else {
            first_phys[0] = first_val[0]; first_phys[1] = first_val[1]; first_phys[2] = first_val[2];
            last_phys[0]  = last_val[0];  last_phys[1]  = last_val[1];  last_phys[2]  = last_val[2];
        }

        /* continuity check */
        for (n = 0; n < 3; ++n) {
            step = first_phys[n] - prev_phys[n];
            if (step < 0.0f) step = -step;
            if (step > max_allowed) {
                hz_per_mt = opts->gamma * 0.001f;
                if (diag) {
                    diag->code                  = PULSEQLIB_ERR_GRAD_DISCONTINUITY;
                    diag->channel               = n;
                    diag->block_index           = coll->block_cursor.within_sequence_block_index;
                    diag->gradient_amplitude    = step / hz_per_mt;
                    diag->max_allowed_amplitude = max_allowed / hz_per_mt;
                }
                coll->block_cursor = saved_cursor;
                return PULSEQLIB_ERR_GRAD_DISCONTINUITY;
            }
        }

        prev_phys[0] = last_phys[0];
        prev_phys[1] = last_phys[1];
        prev_phys[2] = last_phys[2];

        /* advance cursor */
        status = pulseqlib_cursor_next(coll);
    }

    /* final subsequence trailing edge */
    for (n = 0; n < 3; ++n) {
        step = prev_phys[n];
        if (step < 0.0f) step = -step;
        if (step > max_allowed) {
            hz_per_mt = opts->gamma * 0.001f;
            if (diag) {
                diag->code                  = PULSEQLIB_ERR_GRAD_DISCONTINUITY;
                diag->channel               = n;
                diag->block_index           = -1;
                diag->gradient_amplitude    = step / hz_per_mt;
                diag->max_allowed_amplitude = max_allowed / hz_per_mt;
            }
            coll->block_cursor = saved_cursor;
            return PULSEQLIB_ERR_GRAD_DISCONTINUITY;
        }
    }

    coll->block_cursor = saved_cursor;
    return PULSEQLIB_OK;
}

/* ================================================================== */
/*  Max slew rate check (per unique block definition)                 */
/* ================================================================== */

int check_max_slew(
    const pulseqlib_collection* coll,
    pulseqlib_diagnostic* diag,
    const pulseqlib_opts* opts)
{
    int s, d, n, grad_id, shot;
    float slew_limit, slew_phys;
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_block_definition* bdef;
    const pulseqlib_grad_definition* gdef;
    int grad_ids[3];

    if (!coll || !opts) {
        if (diag) { pulseqlib_diagnostic_init(diag); diag->code = PULSEQLIB_ERR_NULL_POINTER; }
        return PULSEQLIB_ERR_NULL_POINTER;
    }
    if (diag) pulseqlib_diagnostic_init(diag);

    slew_limit = opts->max_slew / (float)sqrt(3.0);

    for (s = 0; s < coll->num_subsequences; ++s) {
        desc = &coll->descriptors[s];

        for (d = 0; d < desc->num_unique_blocks; ++d) {
            bdef = &desc->block_definitions[d];
            grad_ids[0] = bdef->gx_id;
            grad_ids[1] = bdef->gy_id;
            grad_ids[2] = bdef->gz_id;

            for (n = 0; n < 3; ++n) {
                grad_id = grad_ids[n];
                if (grad_id < 0 || grad_id >= desc->num_unique_grads)
                    continue;

                gdef = &desc->grad_definitions[grad_id];
                for (shot = 0; shot < gdef->num_shots; ++shot) {
                    slew_phys = gdef->slew_rate[shot] * gdef->max_amplitude[shot];
                    if (slew_phys > slew_limit) {
                        if (diag) {
                            diag->code                  = PULSEQLIB_ERR_MAX_SLEW_EXCEEDED;
                            diag->channel               = n;
                            diag->block_index           = d;
                            diag->gradient_amplitude    = slew_phys;
                            diag->max_allowed_amplitude = slew_limit;
                        }
                        return PULSEQLIB_ERR_MAX_SLEW_EXCEEDED;
                    }
                }
            }
        }
    }

    return PULSEQLIB_OK;
}

/* ================================================================== */
/*  Safety check                                                      */
/* ================================================================== */
int pulseqlib_check_safety(
    pulseqlib_collection* coll,
    pulseqlib_diagnostic* diag,
    const pulseqlib_opts* opts,
    int num_forbidden_bands,
    const int* band_freq_min_hz,
    const int* band_freq_max_hz,
    const float* band_amp_max_tesla_m,
    int pns_decay_time_us,
    float pns_stim_threshold_tesla_m_s)
{
    int rc, s, u;
    int num_unique_trs;
    int* unique_tr_indices;
    int* tr_group_labels;
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_descriptor* trd;
    pulseqlib_tr_gradient_waveforms waveforms;
    pulseqlib_tr_acoustic_spectra spectra;
    pulseqlib_pns_result pns_result;
    pulseqlib_forbidden_band* bands;
    pulseqlib_pns_params pns_params;
    int b;
    int cd_start, cd_size;

    if (!coll || !opts) {
        if (diag) { pulseqlib_diagnostic_init(diag); diag->code = PULSEQLIB_ERR_NULL_POINTER; }
        return PULSEQLIB_ERR_NULL_POINTER;
    }
    if (diag) pulseqlib_diagnostic_init(diag);

    /* ---- 1. max gradient amplitude ---- */
    rc = check_max_grad(coll, diag, opts);
    if (PULSEQLIB_FAILED(rc)) return rc;

    /* ---- 2. gradient continuity ---- */
    rc = check_grad_continuity(coll, diag, opts);
    if (PULSEQLIB_FAILED(rc)) return rc;

    /* ---- 3. max slew rate ---- */
    rc = check_max_slew(coll, diag, opts);
    if (PULSEQLIB_FAILED(rc)) return rc;

    /* ---- 4. build forbidden bands array ---- */
    bands = NULL;
    if (num_forbidden_bands > 0 && band_freq_min_hz &&
        band_freq_max_hz && band_amp_max_tesla_m) {
        bands = (pulseqlib_forbidden_band*)PULSEQLIB_ALLOC(
            (size_t)num_forbidden_bands * sizeof(pulseqlib_forbidden_band));
        if (!bands) {
            if (diag) diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
            return PULSEQLIB_ERR_ALLOC_FAILED;
        }
        for (b = 0; b < num_forbidden_bands; ++b) {
            bands[b].freq_min_hz   = (float)band_freq_min_hz[b];
            bands[b].freq_max_hz   = (float)band_freq_max_hz[b];
            bands[b].max_amplitude = band_amp_max_tesla_m[b];
        }
    }

    /* ---- 5. build PNS params ---- */
#if PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_GEHC
    pns_params.chronaxie_us = (float)pns_decay_time_us;
    pns_params.rheobase     = pns_stim_threshold_tesla_m_s;
    pns_params.alpha        = 1.0f;
#endif

    /* ---- 6. per-subsequence acoustic + PNS ---- */
    for (s = 0; s < coll->num_subsequences; ++s) {
        desc = &coll->descriptors[s];
        trd  = &desc->tr_descriptor;

        /* --- 6a. prep TR (if not degenerate) --- */
        if (trd->num_prep_blocks > 0 && !trd->degenerate_prep) {
            memset(&waveforms, 0, sizeof(waveforms));
            rc = get_gradient_waveforms_range(desc, &waveforms, diag,
                0, trd->num_prep_blocks + trd->tr_size, 0,
                NULL, 0);
            if (PULSEQLIB_FAILED(rc)) goto fail;

            if (num_forbidden_bands > 0) {
                memset(&spectra, 0, sizeof(spectra));
                rc = pulseqlib_get_tr_acoustic_spectra(
                    &spectra, diag, &waveforms,
                    desc->grad_raster_time_us,
                    0, 0.0f, 0.0f, 1,
                    1, 0.0f,
                    num_forbidden_bands, bands, 0);
                pulseqlib_tr_acoustic_spectra_free(&spectra);
                if (PULSEQLIB_FAILED(rc)) {
                    pulseqlib_tr_gradient_waveforms_free(&waveforms);
                    goto fail;
                }
            }

#if PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_GEHC
            memset(&pns_result, 0, sizeof(pns_result));
            rc = pulseqlib_compute_pns(
                &pns_result, diag, opts->gamma,
                pns_stim_threshold_tesla_m_s,
                &waveforms, desc->grad_raster_time_us,
                &pns_params, 0);
            pulseqlib_pns_result_free(&pns_result);
            if (PULSEQLIB_FAILED(rc)) {
                pulseqlib_tr_gradient_waveforms_free(&waveforms);
                goto fail;
            }
#endif
            pulseqlib_tr_gradient_waveforms_free(&waveforms);
        }

        /* --- 6b. main TR (unique shot-index variants) --- */
        unique_tr_indices = NULL;
        tr_group_labels   = NULL;
        num_unique_trs = find_unique_shot_trs(desc,
            &unique_tr_indices, &tr_group_labels);

        /* fallback: if dedup failed or single-shot, do one unfiltered pass */
        if (num_unique_trs <= 0) num_unique_trs = 1;

        for (u = 0; u < num_unique_trs; ++u) {
            memset(&waveforms, 0, sizeof(waveforms));
            rc = get_gradient_waveforms_range(desc, &waveforms, diag,
                trd->num_prep_blocks, trd->tr_size, 1,
                tr_group_labels, u);
            if (PULSEQLIB_FAILED(rc)) {
                if (unique_tr_indices) PULSEQLIB_FREE(unique_tr_indices);
                if (tr_group_labels)   PULSEQLIB_FREE(tr_group_labels);
                goto fail;
            }

            if (num_forbidden_bands > 0) {
                memset(&spectra, 0, sizeof(spectra));
                rc = pulseqlib_get_tr_acoustic_spectra(
                    &spectra, diag, &waveforms,
                    desc->grad_raster_time_us,
                    0, 0.0f, 0.0f, 1,
                    trd->num_trs, trd->tr_duration_us,
                    num_forbidden_bands, bands, 0);
                pulseqlib_tr_acoustic_spectra_free(&spectra);
                if (PULSEQLIB_FAILED(rc)) {
                    pulseqlib_tr_gradient_waveforms_free(&waveforms);
                    if (unique_tr_indices) PULSEQLIB_FREE(unique_tr_indices);
                    if (tr_group_labels)   PULSEQLIB_FREE(tr_group_labels);
                    goto fail;
                }
            }

#if PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_GEHC
            memset(&pns_result, 0, sizeof(pns_result));
            rc = pulseqlib_compute_pns(
                &pns_result, diag, opts->gamma,
                pns_stim_threshold_tesla_m_s,
                &waveforms, desc->grad_raster_time_us,
                &pns_params, 0);
            pulseqlib_pns_result_free(&pns_result);
            if (PULSEQLIB_FAILED(rc)) {
                pulseqlib_tr_gradient_waveforms_free(&waveforms);
                if (unique_tr_indices) PULSEQLIB_FREE(unique_tr_indices);
                if (tr_group_labels)   PULSEQLIB_FREE(tr_group_labels);
                goto fail;
            }
#endif
            pulseqlib_tr_gradient_waveforms_free(&waveforms);
        }

        if (unique_tr_indices) PULSEQLIB_FREE(unique_tr_indices);
        if (tr_group_labels)   PULSEQLIB_FREE(tr_group_labels);

        /* --- 6c. cooldown TR (if not degenerate) --- */
        if (trd->num_cooldown_blocks > 0 && !trd->degenerate_cooldown) {
            cd_size  = trd->tr_size + trd->num_cooldown_blocks;
            cd_start = desc->num_blocks - cd_size;

            memset(&waveforms, 0, sizeof(waveforms));
            rc = get_gradient_waveforms_range(desc, &waveforms, diag,
                cd_start, cd_size, 0,
                NULL, 0);
            if (PULSEQLIB_FAILED(rc)) goto fail;

            if (num_forbidden_bands > 0) {
                memset(&spectra, 0, sizeof(spectra));
                rc = pulseqlib_get_tr_acoustic_spectra(
                    &spectra, diag, &waveforms,
                    desc->grad_raster_time_us,
                    0, 0.0f, 0.0f, 1,
                    1, 0.0f,
                    num_forbidden_bands, bands, 0);
                pulseqlib_tr_acoustic_spectra_free(&spectra);
                if (PULSEQLIB_FAILED(rc)) {
                    pulseqlib_tr_gradient_waveforms_free(&waveforms);
                    goto fail;
                }
            }

#if PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_GEHC
            memset(&pns_result, 0, sizeof(pns_result));
            rc = pulseqlib_compute_pns(
                &pns_result, diag, opts->gamma,
                pns_stim_threshold_tesla_m_s,
                &waveforms, desc->grad_raster_time_us,
                &pns_params, 0);
            pulseqlib_pns_result_free(&pns_result);
            if (PULSEQLIB_FAILED(rc)) {
                pulseqlib_tr_gradient_waveforms_free(&waveforms);
                goto fail;
            }
#endif
            pulseqlib_tr_gradient_waveforms_free(&waveforms);
        }
    }

    if (bands) PULSEQLIB_FREE(bands);
    return PULSEQLIB_OK;

fail:
    if (bands) PULSEQLIB_FREE(bands);
    return rc;
}