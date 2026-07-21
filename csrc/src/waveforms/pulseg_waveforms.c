/**
 * @file pulseg_waveforms.c
 * @brief Flattening one TR onto a uniform raster.
 *
 * Consumers that need a waveform rather than a block list -- safety analysis,
 * trajectory integration, wrapper-side plotting -- go through here. The
 * amplitude mode selects which TR is being asked about: the safety worst case
 * (PULSEG_AMP_MAX_POS), the structural skeleton with variable gradients
 * zeroed (PULSEG_AMP_ZERO_VAR), or one concrete TR (PULSEG_AMP_ACTUAL).
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pulseg_internal.h"
#include "pulseg.h"

/* ================================================================== */
/*  Gradient waveform free                                            */
/* ================================================================== */

void pulseg_tr_gradient_waveforms_free(pulseg_tr_gradient_waveforms *w)
{
    if (!w)
        return;
    if (w->gx.time_us)
        PULSEG_FREE(w->gx.time_us);
    if (w->gx.amplitude_hz_per_m)
        PULSEG_FREE(w->gx.amplitude_hz_per_m);
    if (w->gx.seg_label)
        PULSEG_FREE(w->gx.seg_label);
    if (w->gy.time_us)
        PULSEG_FREE(w->gy.time_us);
    if (w->gy.amplitude_hz_per_m)
        PULSEG_FREE(w->gy.amplitude_hz_per_m);
    if (w->gy.seg_label)
        PULSEG_FREE(w->gy.seg_label);
    if (w->gz.time_us)
        PULSEG_FREE(w->gz.time_us);
    if (w->gz.amplitude_hz_per_m)
        PULSEG_FREE(w->gz.amplitude_hz_per_m);
    if (w->gz.seg_label)
        PULSEG_FREE(w->gz.seg_label);
    memset(w, 0, sizeof(*w));
}

/* ================================================================== */
/*  Internal uniform gradient waveform free                           */
/* ================================================================== */

void pulseg__uniform_grad_waveforms_free(pulseg__uniform_grad_waveforms *w)
{
    if (!w)
        return;
    if (w->gx)
        PULSEG_FREE(w->gx);
    if (w->gy)
        PULSEG_FREE(w->gy);
    if (w->gz)
        PULSEG_FREE(w->gz);
    memset(w, 0, sizeof(*w));
}

/* ================================================================== */
/*  Gradient sample counting                                           */
/* ================================================================== */

static int count_grad_samples_for_block(
    const pulseg_sequence_descriptor *desc,
    const pulseg_grad_definition *gdef,
    float block_duration_us)
{
    int count;
    int num_samples;
    float delay_us, rise_us, flat_us, fall_us, duration_us;
    float grad_raster_us;
    pulseq_shape decomp_time;

    if (!gdef)
        return 2;

    count = 0;
    decomp_time.samples = NULL;
    decomp_time.num_uncompressed_samples = 0;

    grad_raster_us = desc->grad_raster_us;
    num_samples = gdef->fall_time_or_num_uncompressed_samples;
    delay_us = (float)gdef->delay;

    if (delay_us > 0.0f)
        count++;

    if (gdef->type == 0)
    {
        rise_us = (float)gdef->rise_time_or_unused;
        flat_us = (float)gdef->flat_time_or_unused;
        fall_us = (float)gdef->fall_time_or_num_uncompressed_samples;
        duration_us = delay_us + rise_us + flat_us + fall_us;
        count += (flat_us > 0) ? 4 : 3;
    }
    else
    {
        if (gdef->unused_or_time_shape_id > 0 &&
            gdef->unused_or_time_shape_id <= desc->num_shapes &&
            pulseq_decompress_shape(
                &decomp_time,
                &desc->shapes[gdef->unused_or_time_shape_id - 1],
                grad_raster_us))
        {
            duration_us = delay_us + decomp_time.samples[decomp_time.num_uncompressed_samples - 1];
        }
        else
        {
            duration_us =
                delay_us + 0.5f * grad_raster_us + grad_raster_us * (float)(num_samples - 1);
        }
        if (decomp_time.samples)
            PULSEG_FREE(decomp_time.samples);
        count += num_samples;
    }

    if (duration_us < block_duration_us)
        count++;
    return count;
}

/* ================================================================== */
/*  Position-specific max amplitudes (filtered by TR group)           */
/* ================================================================== */

/*
 * Computes per-position worst-case amplitude for each shot index,
 * preserving the sign of the instance with the largest absolute value.
 * Considering only TR instances whose group label matches target_group.
 * If tr_group_labels is NULL, all TRs are included (unfiltered).
 *
 * Output arrays must be pre-allocated to tr_size * PULSEG_MAX_GRAD_SHOTS.
 */
static int compute_position_max_amplitudes_filtered(
    const pulseg_sequence_descriptor *desc,
    float *pos_max_gx,
    float *pos_max_gy,
    float *pos_max_gz,
    int block_start,
    int block_count,
    const int *tr_group_labels,
    int target_group)
{
    const pulseg_tr_descriptor *tr;
    int tr_start, tr_size, num_trs;
    int tr_idx, pos, block_idx;
    int use_full_pass_layout, num_passes, pass_size, pass_idx, st_pos;
    const pulseg_block_table_element *bte;
    const pulseg_grad_table_element *gte;
    float abs_amp;
    int raw_id, shot_idx, arr_idx, n;

    tr = &desc->tr_descriptor;
    tr_size = tr->tr_size;
    num_trs = tr->num_trs;
    use_full_pass_layout =
        !(block_start == (tr->num_prep_blocks + tr->imaging_tr_start) && block_count == tr_size);
    num_passes = (desc->num_passes > 1) ? desc->num_passes : 1;
    pass_size = (num_passes > 0) ? (desc->exec_stream_len / num_passes) : 0;

    for (n = 0; n < block_count * PULSEG_MAX_GRAD_SHOTS; ++n)
    {
        pos_max_gx[n] = 0.0f;
        pos_max_gy[n] = 0.0f;
        pos_max_gz[n] = 0.0f;
    }

    if (use_full_pass_layout && pass_size > 0 && desc->exec_stream_block_idx)
    {
        for (pass_idx = 0; pass_idx < num_passes; ++pass_idx)
        {
            if (tr_group_labels && tr_group_labels[pass_idx] != target_group)
                continue;

            for (pos = 0; pos < block_count; ++pos)
            {
                st_pos = pass_idx * pass_size + block_start + pos;
                if (st_pos < 0 || st_pos >= desc->exec_stream_len)
                    continue;

                block_idx = desc->exec_stream_block_idx[st_pos];
                if (block_idx < 0 || block_idx >= desc->num_blocks)
                    continue;
                bte = &desc->block_table[block_idx];

                raw_id = bte->gx_id;
                if (raw_id >= 0 && raw_id < desc->grad_table_size)
                {
                    gte = &desc->grad_table[raw_id];
                    shot_idx = gte->shot_index;
                    if (shot_idx >= 0 && shot_idx < PULSEG_MAX_GRAD_SHOTS)
                    {
                        abs_amp = gte->amplitude;
                        if (abs_amp < 0.0f)
                            abs_amp = -abs_amp;
                        arr_idx = pos * PULSEG_MAX_GRAD_SHOTS + shot_idx;
                        if (abs_amp > (float)fabs((double)pos_max_gx[arr_idx]))
                            pos_max_gx[arr_idx] = gte->amplitude;
                    }
                }

                raw_id = bte->gy_id;
                if (raw_id >= 0 && raw_id < desc->grad_table_size)
                {
                    gte = &desc->grad_table[raw_id];
                    shot_idx = gte->shot_index;
                    if (shot_idx >= 0 && shot_idx < PULSEG_MAX_GRAD_SHOTS)
                    {
                        abs_amp = gte->amplitude;
                        if (abs_amp < 0.0f)
                            abs_amp = -abs_amp;
                        arr_idx = pos * PULSEG_MAX_GRAD_SHOTS + shot_idx;
                        if (abs_amp > (float)fabs((double)pos_max_gy[arr_idx]))
                            pos_max_gy[arr_idx] = gte->amplitude;
                    }
                }

                raw_id = bte->gz_id;
                if (raw_id >= 0 && raw_id < desc->grad_table_size)
                {
                    gte = &desc->grad_table[raw_id];
                    shot_idx = gte->shot_index;
                    if (shot_idx >= 0 && shot_idx < PULSEG_MAX_GRAD_SHOTS)
                    {
                        abs_amp = gte->amplitude;
                        if (abs_amp < 0.0f)
                            abs_amp = -abs_amp;
                        arr_idx = pos * PULSEG_MAX_GRAD_SHOTS + shot_idx;
                        if (abs_amp > (float)fabs((double)pos_max_gz[arr_idx]))
                            pos_max_gz[arr_idx] = gte->amplitude;
                    }
                }
            }
        }
        return PULSEG_SUCCESS;
    }

    for (tr_idx = 0; tr_idx < num_trs; ++tr_idx)
    {
        /* skip TRs not in the target group */
        if (tr_group_labels && tr_group_labels[tr_idx] != target_group)
            continue;

        tr_start = tr->num_prep_blocks + tr->imaging_tr_start + tr_idx * tr_size;
        for (pos = 0; pos < tr_size; ++pos)
        {
            block_idx = tr_start + pos;
            bte = &desc->block_table[block_idx];

            /* Gx */
            raw_id = bte->gx_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size)
            {
                gte = &desc->grad_table[raw_id];
                shot_idx = gte->shot_index;
                if (shot_idx >= 0 && shot_idx < PULSEG_MAX_GRAD_SHOTS)
                {
                    abs_amp = gte->amplitude;
                    if (abs_amp < 0.0f)
                        abs_amp = -abs_amp;
                    arr_idx = pos * PULSEG_MAX_GRAD_SHOTS + shot_idx;
                    if (abs_amp > (float)fabs((double)pos_max_gx[arr_idx]))
                        pos_max_gx[arr_idx] = gte->amplitude;
                }
            }

            /* Gy */
            raw_id = bte->gy_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size)
            {
                gte = &desc->grad_table[raw_id];
                shot_idx = gte->shot_index;
                if (shot_idx >= 0 && shot_idx < PULSEG_MAX_GRAD_SHOTS)
                {
                    abs_amp = gte->amplitude;
                    if (abs_amp < 0.0f)
                        abs_amp = -abs_amp;
                    arr_idx = pos * PULSEG_MAX_GRAD_SHOTS + shot_idx;
                    if (abs_amp > (float)fabs((double)pos_max_gy[arr_idx]))
                        pos_max_gy[arr_idx] = gte->amplitude;
                }
            }

            /* Gz */
            raw_id = bte->gz_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size)
            {
                gte = &desc->grad_table[raw_id];
                shot_idx = gte->shot_index;
                if (shot_idx >= 0 && shot_idx < PULSEG_MAX_GRAD_SHOTS)
                {
                    abs_amp = gte->amplitude;
                    if (abs_amp < 0.0f)
                        abs_amp = -abs_amp;
                    arr_idx = pos * PULSEG_MAX_GRAD_SHOTS + shot_idx;
                    if (abs_amp > (float)fabs((double)pos_max_gz[arr_idx]))
                        pos_max_gz[arr_idx] = gte->amplitude;
                }
            }
        }
    }
    return PULSEG_SUCCESS;
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
int pulseg__compute_variable_grad_flags(pulseg_sequence_descriptor *desc)
{
    int tr_size, n, pos, si, raw_id, tr_pos;
    const pulseg_block_table_element *bte;
    const pulseg_grad_table_element *gte;

    if (!desc)
        return PULSEG_ERR_NULL_POINTER;

    tr_size = desc->tr_descriptor.tr_size;

    /* free any prior allocation */
    if (desc->variable_grad_flags)
    {
        PULSEG_FREE(desc->variable_grad_flags);
        desc->variable_grad_flags = NULL;
    }

    if (tr_size <= 0 || desc->exec_stream_len <= 0)
        return PULSEG_SUCCESS;

    n = tr_size * 3;
    desc->variable_grad_flags = (int *)PULSEG_ALLOC((size_t)n * sizeof(int));
    if (!desc->variable_grad_flags)
        return PULSEG_ERR_ALLOC_FAILED;
    for (pos = 0; pos < n; ++pos)
        desc->variable_grad_flags[pos] = 0;

    {
        /* Per-position tracking arrays (stack-allocated for tr_size <= 64,
         * heap otherwise).  For typical sequences tr_size is small (< 32). */
        float fa[3 * 64]; /* first_amp[pos*3 + axis] */
        int sv[3 * 64];   /* seen[pos*3 + axis]      */
        float *pfa = fa;
        int *psv = sv;
        int heap = 0;

        if (tr_size > 64)
        {
            pfa = (float *)PULSEG_ALLOC((size_t)(tr_size * 3) * sizeof(float));
            psv = (int *)PULSEG_ALLOC((size_t)(tr_size * 3) * sizeof(int));
            if (!pfa || !psv)
            {
                PULSEG_FREE(pfa);
                PULSEG_FREE(psv);
                PULSEG_FREE(desc->variable_grad_flags);
                desc->variable_grad_flags = NULL;
                return PULSEG_ERR_ALLOC_FAILED;
            }
            heap = 1;
        }

        for (pos = 0; pos < tr_size * 3; ++pos)
        {
            pfa[pos] = 0.0f;
            psv[pos] = 0;
        }

        /* Walk the scan table.  exec_stream_tr_start[si] == 1 marks the first
         * block of a new TR; we use this to reset the within-TR position.
         * This uses the full expanded scan table rather than the deduplicated
         * block table, so that per-TR gradient amplitude variation (e.g. phase
         * encoding steps) is correctly detected even after deduplication. */
        tr_pos = 0;
        for (si = 0; si < desc->exec_stream_len; ++si)
        {
            /* Reset position counter at the start of each new TR */
            if (desc->exec_stream_tr_start && desc->exec_stream_tr_start[si])
                tr_pos = 0;

            {
                int bt_idx = desc->exec_stream_block_idx[si];
                if (tr_pos < tr_size && bt_idx >= 0 && bt_idx < desc->num_blocks)
                {
                    int axis;
                    int raw_ids[3];
                    bte = &desc->block_table[bt_idx];
                    raw_ids[0] = bte->gx_id;
                    raw_ids[1] = bte->gy_id;
                    raw_ids[2] = bte->gz_id;

                    for (axis = 0; axis < 3; ++axis)
                    {
                        raw_id = raw_ids[axis];
                        if (raw_id >= 0 && raw_id < desc->grad_table_size)
                        {
                            int idx = tr_pos * 3 + axis;
                            gte = &desc->grad_table[raw_id];
                            if (!psv[idx])
                            {
                                pfa[idx] = gte->amplitude;
                                psv[idx] = 1;
                            }
                            else if (gte->amplitude != pfa[idx])
                            {
                                desc->variable_grad_flags[tr_pos * 3 + axis] = 1;
                            }
                        }
                    }
                }
            }

            tr_pos++;
            if (tr_pos >= tr_size)
                tr_pos = 0;
        }

        if (heap)
        {
            PULSEG_FREE(pfa);
            PULSEG_FREE(psv);
        }
    }

    return PULSEG_SUCCESS;
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
int pulseg__find_unique_shot_trs(
    const pulseg_sequence_descriptor *desc,
    int **out_unique_tr_indices,
    int **out_tr_group_labels)
{
    const pulseg_tr_descriptor *tr;
    int tr_size, num_trs, num_cols;
    int *int_rows;
    int *unique_defs;
    int *event_table;
    int *result_indices;
    int num_unique;
    int tr_idx, pos, col, block_idx, raw_id;
    const pulseg_block_table_element *bte;
    const pulseg_grad_table_element *gte;

    *out_unique_tr_indices = NULL;
    *out_tr_group_labels = NULL;

    tr = &desc->tr_descriptor;
    tr_size = tr->tr_size;
    num_trs = tr->num_trs;

    if (num_trs <= 0 || tr_size <= 0)
        return 0;

    /* Each row: tr_size * 3 ints = (gx_shot, gy_shot, gz_shot) per position */
    num_cols = tr_size * 3;

    int_rows = (int *)PULSEG_ALLOC((size_t)num_trs * (size_t)num_cols * sizeof(int));
    unique_defs = (int *)PULSEG_ALLOC((size_t)num_trs * sizeof(int));
    event_table = (int *)PULSEG_ALLOC((size_t)num_trs * sizeof(int));
    if (!int_rows || !unique_defs || !event_table)
    {
        if (int_rows)
            PULSEG_FREE(int_rows);
        if (unique_defs)
            PULSEG_FREE(unique_defs);
        if (event_table)
            PULSEG_FREE(event_table);
        return 0;
    }

    /* Build the fingerprint matrix */
    for (tr_idx = 0; tr_idx < num_trs; ++tr_idx)
    {
        col = 0;
        for (pos = 0; pos < tr_size; ++pos)
        {
            block_idx = tr->num_prep_blocks + tr->imaging_tr_start + tr_idx * tr_size + pos;
            bte = &desc->block_table[block_idx];

            /* Gx shot index */
            raw_id = bte->gx_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size)
            {
                gte = &desc->grad_table[raw_id];
                int_rows[tr_idx * num_cols + col] = gte->shot_index;
            }
            else
            {
                int_rows[tr_idx * num_cols + col] = -1;
            }
            col++;

            /* Gy shot index */
            raw_id = bte->gy_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size)
            {
                gte = &desc->grad_table[raw_id];
                int_rows[tr_idx * num_cols + col] = gte->shot_index;
            }
            else
            {
                int_rows[tr_idx * num_cols + col] = -1;
            }
            col++;

            /* Gz shot index */
            raw_id = bte->gz_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size)
            {
                gte = &desc->grad_table[raw_id];
                int_rows[tr_idx * num_cols + col] = gte->shot_index;
            }
            else
            {
                int_rows[tr_idx * num_cols + col] = -1;
            }
            col++;
        }
    }

    /* Deduplicate */
    num_unique =
        pulseg__deduplicate_int_rows(unique_defs, event_table, int_rows, num_trs, num_cols);

    PULSEG_FREE(int_rows);

    if (num_unique <= 0)
    {
        PULSEG_FREE(unique_defs);
        PULSEG_FREE(event_table);
        return 0;
    }

    /* Copy representative TR indices into right-sized array */
    result_indices = (int *)PULSEG_ALLOC((size_t)num_unique * sizeof(int));
    if (!result_indices)
    {
        PULSEG_FREE(unique_defs);
        PULSEG_FREE(event_table);
        return 0;
    }
    for (tr_idx = 0; tr_idx < num_unique; ++tr_idx)
    {
        result_indices[tr_idx] = unique_defs[tr_idx];
    }
    PULSEG_FREE(unique_defs);

    *out_unique_tr_indices = result_indices;
    *out_tr_group_labels = event_table;
    return num_unique;
}

/* ================================================================== */
/*  Fill waveform for a single block                                  */
/* ================================================================== */

static int fill_grad_waveform_for_block(
    const pulseg_sequence_descriptor *desc,
    float *time,
    float *waveform,
    int start_idx,
    const pulseg_grad_definition *gdef,
    const pulseg_grad_table_element *gte,
    float t0,
    const float *pos_max_amp,
    float block_duration_us)
{
    int i, idx;
    float sign, max_amp;
    float delay_us, t_sample, last_written;
    int shape_id, time_shape_id, shot_idx, num_samples;
    float rise_us, flat_us, fall_us;
    float grad_raster_us, block_end_us;
    pulseq_shape decomp_wave, decomp_time;
    int has_time_shape;

    idx = start_idx;
    grad_raster_us = desc->grad_raster_us;
    block_end_us = t0 + block_duration_us;
    decomp_wave.samples = NULL;
    decomp_time.samples = NULL;

    if (!gdef || !gte)
    {
        time[idx] = t0;
        waveform[idx] = 0.0f;
        idx++;
        time[idx] = block_end_us;
        waveform[idx] = 0.0f;
        idx++;
        return idx - start_idx;
    }

    last_written = t0;
    shot_idx = gte->shot_index;
    max_amp = pos_max_amp[shot_idx];
    sign = (max_amp >= 0.0f) ? 1.0f : -1.0f;
    if (max_amp < 0.0f)
        max_amp = -max_amp;
    delay_us = (float)gdef->delay;

    if (delay_us > 0.0f)
    {
        t_sample = t0;
        time[idx] = t_sample;
        waveform[idx] = 0.0f;
        last_written = t_sample;
        idx++;
    }

    if (gdef->type == 0)
    {
        rise_us = (float)gdef->rise_time_or_unused;
        flat_us = (float)gdef->flat_time_or_unused;
        fall_us = (float)gdef->fall_time_or_num_uncompressed_samples;

        if (flat_us > 0)
        {
            t_sample = t0 + delay_us;
            time[idx] = t_sample;
            waveform[idx] = 0.0f;
            last_written = t_sample;
            idx++;

            t_sample = t0 + delay_us + rise_us;
            time[idx] = t_sample;
            waveform[idx] = sign * max_amp;
            last_written = t_sample;
            idx++;

            t_sample = t0 + delay_us + rise_us + flat_us;
            time[idx] = t_sample;
            waveform[idx] = sign * max_amp;
            last_written = t_sample;
            idx++;

            t_sample = t0 + delay_us + rise_us + flat_us + fall_us;
            time[idx] = t_sample;
            waveform[idx] = 0.0f;
            last_written = t_sample;
            idx++;
        }
        else
        {
            t_sample = t0 + delay_us;
            time[idx] = t_sample;
            waveform[idx] = 0.0f;
            last_written = t_sample;
            idx++;

            t_sample = t0 + delay_us + rise_us;
            time[idx] = t_sample;
            waveform[idx] = sign * max_amp;
            last_written = t_sample;
            idx++;

            t_sample = t0 + delay_us + rise_us + fall_us;
            time[idx] = t_sample;
            waveform[idx] = 0.0f;
            last_written = t_sample;
            idx++;
        }
    }
    else
    {
        num_samples = gdef->fall_time_or_num_uncompressed_samples;
        time_shape_id = gdef->unused_or_time_shape_id;
        shape_id = gdef->shot_shape_ids[shot_idx];

        if (shape_id <= 0 || shape_id > desc->num_shapes)
            return 0;
        if (!pulseq_decompress_shape(&decomp_wave, &desc->shapes[shape_id - 1], 1.0f))
            return 0;

        has_time_shape = 0;
        if (time_shape_id > 0 && time_shape_id <= desc->num_shapes)
        {
            if (pulseq_decompress_shape(
                    &decomp_time,
                    &desc->shapes[time_shape_id - 1],
                    grad_raster_us))
                has_time_shape = 1;
        }

        if (has_time_shape)
        {
            for (i = 0; i < num_samples; ++i)
            {
                t_sample = t0 + delay_us + decomp_time.samples[i];
                time[idx] = t_sample;
                waveform[idx] = sign * max_amp * decomp_wave.samples[i];
                last_written = t_sample;
                idx++;
            }
        }
        else
        {
            for (i = 0; i < num_samples; ++i)
            {
                t_sample = t0 + delay_us + 0.5f * grad_raster_us + (float)i * grad_raster_us;
                time[idx] = t_sample;
                waveform[idx] = sign * max_amp * decomp_wave.samples[i];
                last_written = t_sample;
                idx++;
            }
        }

        if (decomp_wave.samples)
            PULSEG_FREE(decomp_wave.samples);
        if (decomp_time.samples)
            PULSEG_FREE(decomp_time.samples);
    }

    if (block_end_us > last_written)
    {
        float tail_amp = 0.0f;
        if (gdef->type == 1 && idx > start_idx)
            tail_amp = waveform[idx - 1];
        time[idx] = block_end_us;
        waveform[idx] = tail_amp;
        idx++;
    }

    return idx - start_idx;
}

/* ================================================================== */
/*  Interpolate to uniform raster                                     */
/* ================================================================== */

static int interpolate_to_uniform(
    float **time,
    float **waveform,
    int *num_samples,
    float target_raster_us)
{
    float *t_in;
    float *w_in;
    float *t_out = NULL;
    float *w_out = NULL;
    int n_in, n_out, i;
    float t_start, t_end, duration;

    t_out = NULL;
    w_out = NULL;

    if (!time || !waveform || !num_samples || *num_samples <= 0)
        return PULSEG_SUCCESS;

    t_in = *time;
    w_in = *waveform;
    n_in = *num_samples;

    t_start = t_in[0];
    t_end = t_in[n_in - 1];
    duration = t_end - t_start;
    if (duration <= 0.0f)
        return PULSEG_SUCCESS;

    n_out = (int)(duration / target_raster_us) + 1;

    t_out = (float *)PULSEG_ALLOC(n_out * sizeof(float));
    w_out = (float *)PULSEG_ALLOC(n_out * sizeof(float));
    if (!t_out || !w_out)
    {
        if (t_out)
            PULSEG_FREE(t_out);
        if (w_out)
            PULSEG_FREE(w_out);
        return PULSEG_ERR_ALLOC_FAILED;
    }

    for (i = 0; i < n_out; ++i)
        t_out[i] = t_start + (float)i * target_raster_us;

    pulseg__interp1_linear(w_out, t_out, n_out, t_in, w_in, n_in);

    PULSEG_FREE(t_in);
    PULSEG_FREE(w_in);

    *time = t_out;
    *waveform = w_out;
    *num_samples = n_out;
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Gradient waveforms for an arbitrary block range                   */
/* ================================================================== */

/*  amplitude_mode (uses PULSEG_AMP_* defines from pulseg_types.h):
 *    PULSEG_AMP_MAX_POS  (0) = position-max (worst-case safety)
 *    PULSEG_AMP_ZERO_VAR (1) = zero variable grads, keep constant (k-space)
 *    PULSEG_AMP_ACTUAL   (2) = actual block amplitude (single-TR)
 */
int pulseg__get_gradient_waveforms_range(
    const pulseg_sequence_descriptor *desc,
    pulseg__uniform_grad_waveforms *out,
    pulseg_diagnostic *diag,
    int block_start,
    int block_count,
    int amplitude_mode,
    const int *tr_group_labels,
    int target_group,
    const int *block_order)
{
    pulseg_diagnostic local_diag;
    int n, block_idx;
    int total_gx, total_gy, total_gz;
    int idx_gx, idx_gy, idx_gz;
    int num_gx, num_gy, num_gz;
    int result;
    float t0, block_dur_us, target_raster_us;
    int block_def_id;
    const pulseg_base_block *bdef;
    const pulseg_block_table_element *bte;
    int gx_raw, gy_raw, gz_raw;
    const pulseg_grad_definition *gx_def;
    const pulseg_grad_definition *gy_def;
    const pulseg_grad_definition *gz_def;
    const pulseg_grad_table_element *gx_tab;
    const pulseg_grad_table_element *gy_tab;
    const pulseg_grad_table_element *gz_tab;
    float *pos_max_gx;
    float *pos_max_gy;
    float *pos_max_gz;
    float actual_amp[PULSEG_MAX_GRAD_SHOTS];
    int k;
    float *time_gx;
    float *time_gy;
    float *time_gz;
    float *wf_gx;
    float *wf_gy;
    float *wf_gz;

    pos_max_gx = NULL;
    pos_max_gy = NULL;
    pos_max_gz = NULL;
    time_gx = NULL;
    time_gy = NULL;
    time_gz = NULL;
    wf_gx = NULL;
    wf_gy = NULL;
    wf_gz = NULL;

    if (!diag)
    {
        pulseg_diagnostic_init(&local_diag);
        diag = &local_diag;
    }
    else
    {
        pulseg_diagnostic_init(diag);
    }

    if (!desc || !out)
    {
        diag->code = PULSEG_ERR_NULL_POINTER;
        return diag->code;
    }

    memset(out, 0, sizeof(*out));

    if (block_count <= 0)
    {
        diag->code = PULSEG_ERR_TR_NO_BLOCKS;
        return diag->code;
    }
    if (!block_order && (block_start < 0 || block_start + block_count > desc->num_blocks))
    {
        diag->code = PULSEG_ERR_INVALID_ARGUMENT;
        return diag->code;
    }
    if (block_order)
    {
        for (n = 0; n < block_count; ++n)
        {
            if (block_order[n] < 0 || block_order[n] >= desc->num_blocks)
            {
                diag->code = PULSEG_ERR_INVALID_ARGUMENT;
                return diag->code;
            }
        }
    }

    /* position-max amplitudes (only for worst-case main-TR mode) */
    if (amplitude_mode == PULSEG_AMP_MAX_POS || amplitude_mode == PULSEG_AMP_ZERO_VAR)
    {
        pos_max_gx =
            (float *)PULSEG_ALLOC((size_t)block_count * PULSEG_MAX_GRAD_SHOTS * sizeof(float));
        pos_max_gy =
            (float *)PULSEG_ALLOC((size_t)block_count * PULSEG_MAX_GRAD_SHOTS * sizeof(float));
        pos_max_gz =
            (float *)PULSEG_ALLOC((size_t)block_count * PULSEG_MAX_GRAD_SHOTS * sizeof(float));
        if (!pos_max_gx || !pos_max_gy || !pos_max_gz)
        {
            if (pos_max_gx)
                PULSEG_FREE(pos_max_gx);
            if (pos_max_gy)
                PULSEG_FREE(pos_max_gy);
            if (pos_max_gz)
                PULSEG_FREE(pos_max_gz);
            diag->code = PULSEG_ERR_ALLOC_FAILED;
            return diag->code;
        }
        compute_position_max_amplitudes_filtered(
            desc,
            pos_max_gx,
            pos_max_gy,
            pos_max_gz,
            block_start,
            block_count,
            tr_group_labels,
            target_group);

        /* ZERO_VAR: zero out positions whose gradients vary across TRs */
        if (amplitude_mode == PULSEG_AMP_ZERO_VAR && desc->variable_grad_flags)
        {
            int vp;
            for (vp = 0; vp < block_count && vp < desc->tr_descriptor.tr_size; ++vp)
            {
                if (desc->variable_grad_flags[vp * 3 + 0])
                    for (k = 0; k < PULSEG_MAX_GRAD_SHOTS; ++k)
                        pos_max_gx[vp * PULSEG_MAX_GRAD_SHOTS + k] = 0.0f;
                if (desc->variable_grad_flags[vp * 3 + 1])
                    for (k = 0; k < PULSEG_MAX_GRAD_SHOTS; ++k)
                        pos_max_gy[vp * PULSEG_MAX_GRAD_SHOTS + k] = 0.0f;
                if (desc->variable_grad_flags[vp * 3 + 2])
                    for (k = 0; k < PULSEG_MAX_GRAD_SHOTS; ++k)
                        pos_max_gz[vp * PULSEG_MAX_GRAD_SHOTS + k] = 0.0f;
            }
        }
    }

    /* ---- pass 1: count samples ---- */
    total_gx = 0;
    total_gy = 0;
    total_gz = 0;
    for (n = 0; n < block_count; ++n)
    {
        block_idx = block_order ? block_order[n] : block_start + n;
        bte = &desc->block_table[block_idx];
        block_def_id = bte->id;
        bdef = &desc->base_blocks[block_def_id];
        block_dur_us = (bte->duration_us >= 0) ? (float)bte->duration_us : (float)bdef->duration_us;

        gx_raw = bte->gx_id;
        gy_raw = bte->gy_id;
        gz_raw = bte->gz_id;
        gx_def =
            (gx_raw >= 0 && gx_raw < desc->grad_table_size && desc->grad_table[gx_raw].id >= 0 &&
             desc->grad_table[gx_raw].id < desc->num_unique_grads)
            ? &desc->grad_definitions[desc->grad_table[gx_raw].id]
            : NULL;
        gy_def =
            (gy_raw >= 0 && gy_raw < desc->grad_table_size && desc->grad_table[gy_raw].id >= 0 &&
             desc->grad_table[gy_raw].id < desc->num_unique_grads)
            ? &desc->grad_definitions[desc->grad_table[gy_raw].id]
            : NULL;
        gz_def =
            (gz_raw >= 0 && gz_raw < desc->grad_table_size && desc->grad_table[gz_raw].id >= 0 &&
             desc->grad_table[gz_raw].id < desc->num_unique_grads)
            ? &desc->grad_definitions[desc->grad_table[gz_raw].id]
            : NULL;

        total_gx += count_grad_samples_for_block(desc, gx_def, block_dur_us);
        total_gy += count_grad_samples_for_block(desc, gy_def, block_dur_us);
        total_gz += count_grad_samples_for_block(desc, gz_def, block_dur_us);
    }

    /* ---- allocate (local time arrays + output waveform arrays) ---- */
    time_gx = (float *)PULSEG_ALLOC((size_t)total_gx * sizeof(float));
    wf_gx = (float *)PULSEG_ALLOC((size_t)total_gx * sizeof(float));
    time_gy = (float *)PULSEG_ALLOC((size_t)total_gy * sizeof(float));
    wf_gy = (float *)PULSEG_ALLOC((size_t)total_gy * sizeof(float));
    time_gz = (float *)PULSEG_ALLOC((size_t)total_gz * sizeof(float));
    wf_gz = (float *)PULSEG_ALLOC((size_t)total_gz * sizeof(float));
    if (!time_gx || !wf_gx || !time_gy || !wf_gy || !time_gz || !wf_gz)
    {
        if (pos_max_gx)
            PULSEG_FREE(pos_max_gx);
        if (pos_max_gy)
            PULSEG_FREE(pos_max_gy);
        if (pos_max_gz)
            PULSEG_FREE(pos_max_gz);
        if (time_gx)
            PULSEG_FREE(time_gx);
        if (time_gy)
            PULSEG_FREE(time_gy);
        if (time_gz)
            PULSEG_FREE(time_gz);
        if (wf_gx)
            PULSEG_FREE(wf_gx);
        if (wf_gy)
            PULSEG_FREE(wf_gy);
        if (wf_gz)
            PULSEG_FREE(wf_gz);
        diag->code = PULSEG_ERR_ALLOC_FAILED;
        return diag->code;
    }

    /* ---- pass 2: fill ---- */
    t0 = 0.0f;
    idx_gx = 0;
    idx_gy = 0;
    idx_gz = 0;
    for (n = 0; n < block_count; ++n)
    {
        block_idx = block_order ? block_order[n] : block_start + n;
        bte = &desc->block_table[block_idx];
        block_def_id = bte->id;
        bdef = &desc->base_blocks[block_def_id];
        block_dur_us = (bte->duration_us >= 0) ? (float)bte->duration_us : (float)bdef->duration_us;

        gx_raw = bte->gx_id;
        gy_raw = bte->gy_id;
        gz_raw = bte->gz_id;

        gx_tab = (gx_raw >= 0 && gx_raw < desc->grad_table_size) ? &desc->grad_table[gx_raw] : NULL;
        gy_tab = (gy_raw >= 0 && gy_raw < desc->grad_table_size) ? &desc->grad_table[gy_raw] : NULL;
        gz_tab = (gz_raw >= 0 && gz_raw < desc->grad_table_size) ? &desc->grad_table[gz_raw] : NULL;

        gx_def = (gx_tab && gx_tab->id >= 0 && gx_tab->id < desc->num_unique_grads)
            ? &desc->grad_definitions[gx_tab->id]
            : NULL;
        gy_def = (gy_tab && gy_tab->id >= 0 && gy_tab->id < desc->num_unique_grads)
            ? &desc->grad_definitions[gy_tab->id]
            : NULL;
        gz_def = (gz_tab && gz_tab->id >= 0 && gz_tab->id < desc->num_unique_grads)
            ? &desc->grad_definitions[gz_tab->id]
            : NULL;

        if (amplitude_mode == PULSEG_AMP_MAX_POS || amplitude_mode == PULSEG_AMP_ZERO_VAR)
        {
            idx_gx += fill_grad_waveform_for_block(
                desc,
                time_gx,
                wf_gx,
                idx_gx,
                gx_def,
                gx_tab,
                t0,
                &pos_max_gx[n * PULSEG_MAX_GRAD_SHOTS],
                block_dur_us);
            idx_gy += fill_grad_waveform_for_block(
                desc,
                time_gy,
                wf_gy,
                idx_gy,
                gy_def,
                gy_tab,
                t0,
                &pos_max_gy[n * PULSEG_MAX_GRAD_SHOTS],
                block_dur_us);
            idx_gz += fill_grad_waveform_for_block(
                desc,
                time_gz,
                wf_gz,
                idx_gz,
                gz_def,
                gz_tab,
                t0,
                &pos_max_gz[n * PULSEG_MAX_GRAD_SHOTS],
                block_dur_us);
        }
        else
        {
            for (k = 0; k < PULSEG_MAX_GRAD_SHOTS; ++k)
                actual_amp[k] = 0.0f;
            if (gx_tab)
            {
                k = gx_tab->shot_index;
                if (k >= 0 && k < PULSEG_MAX_GRAD_SHOTS)
                {
                    actual_amp[k] = gx_tab->amplitude;
                }
            }
            idx_gx += fill_grad_waveform_for_block(
                desc,
                time_gx,
                wf_gx,
                idx_gx,
                gx_def,
                gx_tab,
                t0,
                actual_amp,
                block_dur_us);

            for (k = 0; k < PULSEG_MAX_GRAD_SHOTS; ++k)
                actual_amp[k] = 0.0f;
            if (gy_tab)
            {
                k = gy_tab->shot_index;
                if (k >= 0 && k < PULSEG_MAX_GRAD_SHOTS)
                {
                    actual_amp[k] = gy_tab->amplitude;
                }
            }
            idx_gy += fill_grad_waveform_for_block(
                desc,
                time_gy,
                wf_gy,
                idx_gy,
                gy_def,
                gy_tab,
                t0,
                actual_amp,
                block_dur_us);

            for (k = 0; k < PULSEG_MAX_GRAD_SHOTS; ++k)
                actual_amp[k] = 0.0f;
            if (gz_tab)
            {
                k = gz_tab->shot_index;
                if (k >= 0 && k < PULSEG_MAX_GRAD_SHOTS)
                {
                    actual_amp[k] = gz_tab->amplitude;
                }
            }
            idx_gz += fill_grad_waveform_for_block(
                desc,
                time_gz,
                wf_gz,
                idx_gz,
                gz_def,
                gz_tab,
                t0,
                actual_amp,
                block_dur_us);
        }

        t0 += block_dur_us;
    }

    if (pos_max_gx)
        PULSEG_FREE(pos_max_gx);
    if (pos_max_gy)
        PULSEG_FREE(pos_max_gy);
    if (pos_max_gz)
        PULSEG_FREE(pos_max_gz);

    num_gx = idx_gx;
    num_gy = idx_gy;
    num_gz = idx_gz;

    /* interpolate each axis to uniform raster (half gradient raster) */
    target_raster_us = 0.5f * desc->grad_raster_us;

    result = interpolate_to_uniform(&time_gx, &wf_gx, &num_gx, target_raster_us);
    if (PULSEG_FAILED(result))
    {
        if (time_gx)
            PULSEG_FREE(time_gx);
        if (time_gy)
            PULSEG_FREE(time_gy);
        if (time_gz)
            PULSEG_FREE(time_gz);
        if (wf_gx)
            PULSEG_FREE(wf_gx);
        if (wf_gy)
            PULSEG_FREE(wf_gy);
        if (wf_gz)
            PULSEG_FREE(wf_gz);
        diag->code = result;
        return result;
    }
    result = interpolate_to_uniform(&time_gy, &wf_gy, &num_gy, target_raster_us);
    if (PULSEG_FAILED(result))
    {
        if (time_gx)
            PULSEG_FREE(time_gx);
        if (time_gy)
            PULSEG_FREE(time_gy);
        if (time_gz)
            PULSEG_FREE(time_gz);
        if (wf_gx)
            PULSEG_FREE(wf_gx);
        if (wf_gy)
            PULSEG_FREE(wf_gy);
        if (wf_gz)
            PULSEG_FREE(wf_gz);
        diag->code = result;
        return result;
    }
    result = interpolate_to_uniform(&time_gz, &wf_gz, &num_gz, target_raster_us);
    if (PULSEG_FAILED(result))
    {
        if (time_gx)
            PULSEG_FREE(time_gx);
        if (time_gy)
            PULSEG_FREE(time_gy);
        if (time_gz)
            PULSEG_FREE(time_gz);
        if (wf_gx)
            PULSEG_FREE(wf_gx);
        if (wf_gy)
            PULSEG_FREE(wf_gy);
        if (wf_gz)
            PULSEG_FREE(wf_gz);
        diag->code = result;
        return result;
    }

    /* Post-interpolation: all axes share the same uniform raster. */
    out->gx = wf_gx;
    out->gy = wf_gy;
    out->gz = wf_gz;
    out->num_samples = num_gx;
    out->raster_us = target_raster_us;
    PULSEG_FREE(time_gx);
    PULSEG_FREE(time_gy);
    PULSEG_FREE(time_gz);

    diag->code = PULSEG_SUCCESS;
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  get_tr_gradient_waveforms                                         */
/* ================================================================== */

int pulseg_get_tr_gradient_waveforms(
    const pulseg_collection *coll,
    pulseg_tr_gradient_waveforms *waveforms,
    pulseg_diagnostic *diag,
    int subseq_idx,
    int canonical_tr_idx)
{
    const pulseg_sequence_descriptor *desc;
    pulseg__uniform_grad_waveforms uw;
    int *unique_indices = NULL;
    int *group_labels = NULL;
    int num_unique;
    int has_nd_prep, has_nd_cool;
    int rep_idx;
    int start_block, block_count;
    int rc, i;
    int *block_order;
    float *time_arr;

    memset(&uw, 0, sizeof(uw));
    block_order = NULL;
    if (!coll || canonical_tr_idx < 0 || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
    {
        if (diag)
        {
            pulseg_diagnostic_init(diag);
            diag->code = PULSEG_ERR_INVALID_ARGUMENT;
        }
        return PULSEG_ERR_INVALID_ARGUMENT;
    }

    desc = &coll->descriptors[subseq_idx];

    has_nd_prep = (desc->tr_descriptor.num_prep_blocks > 0 && !desc->tr_descriptor.degenerate_prep);
    has_nd_cool =
        (desc->tr_descriptor.num_cooldown_blocks > 0 && !desc->tr_descriptor.degenerate_cooldown);

    if (has_nd_prep || has_nd_cool)
    {
        /* Non-degenerate (e.g. MPRAGE): one canonical TR per unique pass
         * pattern.  Use find_unique_shot_passes to discover how many
         * distinct passes exist and pick the representative for this index. */
        num_unique = pulseg__find_unique_shot_passes(desc, &unique_indices, &group_labels);
        if (num_unique <= 0)
        {
            num_unique = 1;
            rep_idx = 0;
        }
        else
        {
            if (canonical_tr_idx >= num_unique)
            {
                PULSEG_FREE(unique_indices);
                PULSEG_FREE(group_labels);
                if (diag)
                {
                    pulseg_diagnostic_init(diag);
                    diag->code = PULSEG_ERR_INVALID_ARGUMENT;
                }
                return PULSEG_ERR_INVALID_ARGUMENT;
            }
            rep_idx = unique_indices[canonical_tr_idx];
            PULSEG_FREE(unique_indices);
            PULSEG_FREE(group_labels);
        }
        /* Render the representative pass with average expansion:
         * prep + num_averages * imaging + cooldown. */
        {
            int pass_base = rep_idx * desc->pass_len;
            int prep_blk = desc->tr_descriptor.num_prep_blocks;
            int img_len = desc->tr_descriptor.num_trs * desc->tr_descriptor.tr_size;
            int cool_blk = desc->tr_descriptor.num_cooldown_blocks;
            int num_avgs = (desc->num_averages > 0) ? desc->num_averages : 1;
            int exp_count = prep_blk + num_avgs * img_len + cool_blk;
            int avg_i;
            int pos_i;

            block_order = (int *)PULSEG_ALLOC((size_t)exp_count * sizeof(int));
            if (!block_order)
            {
                if (diag)
                {
                    pulseg_diagnostic_init(diag);
                    diag->code = PULSEG_ERR_ALLOC_FAILED;
                }
                return PULSEG_ERR_ALLOC_FAILED;
            }

            block_count = exp_count;
            i = 0;
            for (pos_i = 0; pos_i < prep_blk; ++pos_i)
                block_order[i++] = pass_base + pos_i;
            for (avg_i = 0; avg_i < num_avgs; ++avg_i)
                for (pos_i = 0; pos_i < img_len; ++pos_i)
                    block_order[i++] = pass_base + prep_blk + pos_i;
            for (pos_i = 0; pos_i < cool_blk; ++pos_i)
                block_order[i++] = pass_base + prep_blk + img_len + pos_i;
            start_block = 0;
        }
    }
    else
    {
        /* Degenerate (e.g. GRE): one canonical TR per unique shot pattern
         * within the imaging region. */
        num_unique = pulseg__find_unique_shot_trs(desc, &unique_indices, &group_labels);
        if (num_unique <= 0)
        {
            num_unique = 1;
            rep_idx = 0;
        }
        else
        {
            if (canonical_tr_idx >= num_unique)
            {
                PULSEG_FREE(unique_indices);
                PULSEG_FREE(group_labels);
                if (diag)
                {
                    pulseg_diagnostic_init(diag);
                    diag->code = PULSEG_ERR_INVALID_ARGUMENT;
                }
                return PULSEG_ERR_INVALID_ARGUMENT;
            }
            rep_idx = unique_indices[canonical_tr_idx];
            PULSEG_FREE(unique_indices);
            PULSEG_FREE(group_labels);
        }
        start_block = desc->tr_descriptor.num_prep_blocks + desc->tr_descriptor.imaging_tr_start +
            rep_idx * desc->tr_descriptor.tr_size;
        block_count = desc->tr_descriptor.tr_size;
    }

    rc = pulseg__get_gradient_waveforms_range(
        desc,
        &uw,
        diag,
        start_block,
        block_count,
        PULSEG_AMP_ACTUAL,
        NULL,
        0,
        block_order);
    if (PULSEG_FAILED(rc))
    {
        if (block_order)
            PULSEG_FREE(block_order);
        return rc;
    }
    if (!waveforms)
    {
        pulseg__uniform_grad_waveforms_free(&uw);
        if (block_order)
            PULSEG_FREE(block_order);
        return PULSEG_ERR_NULL_POINTER;
    }
    memset(waveforms, 0, sizeof(*waveforms));
    /* build common time array */
    time_arr = (float *)PULSEG_ALLOC((size_t)uw.num_samples * sizeof(float));
    if (!time_arr)
    {
        pulseg__uniform_grad_waveforms_free(&uw);
        if (block_order)
            PULSEG_FREE(block_order);
        return PULSEG_ERR_ALLOC_FAILED;
    }
    for (i = 0; i < uw.num_samples; ++i)
        time_arr[i] = (float)i * uw.raster_us;
    /* gx */
    waveforms->gx.num_samples = uw.num_samples;
    waveforms->gx.amplitude_hz_per_m = uw.gx;
    uw.gx = NULL;
    waveforms->gx.time_us = time_arr;
    waveforms->gx.seg_label = NULL;
    /* gy */
    time_arr = (float *)PULSEG_ALLOC((size_t)uw.num_samples * sizeof(float));
    if (!time_arr)
    {
        pulseg__uniform_grad_waveforms_free(&uw);
        pulseg_tr_gradient_waveforms_free(waveforms);
        if (block_order)
            PULSEG_FREE(block_order);
        return PULSEG_ERR_ALLOC_FAILED;
    }
    for (i = 0; i < uw.num_samples; ++i)
        time_arr[i] = (float)i * uw.raster_us;
    waveforms->gy.num_samples = uw.num_samples;
    waveforms->gy.amplitude_hz_per_m = uw.gy;
    uw.gy = NULL;
    waveforms->gy.time_us = time_arr;
    waveforms->gy.seg_label = NULL;
    /* gz */
    time_arr = (float *)PULSEG_ALLOC((size_t)uw.num_samples * sizeof(float));
    if (!time_arr)
    {
        pulseg__uniform_grad_waveforms_free(&uw);
        pulseg_tr_gradient_waveforms_free(waveforms);
        if (block_order)
            PULSEG_FREE(block_order);
        return PULSEG_ERR_ALLOC_FAILED;
    }
    for (i = 0; i < uw.num_samples; ++i)
        time_arr[i] = (float)i * uw.raster_us;
    waveforms->gz.num_samples = uw.num_samples;
    waveforms->gz.amplitude_hz_per_m = uw.gz;
    uw.gz = NULL;
    waveforms->gz.time_us = time_arr;
    waveforms->gz.seg_label = NULL;
    if (block_order)
        PULSEG_FREE(block_order);
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Native-timing TR waveforms (for plotting)                        */
/* ================================================================== */

/*
 * Count RF samples for a single block (flat block index).
 * Returns 0 if block has no RF.
 */
static int count_rf_samples_for_flat_block(const pulseg_sequence_descriptor *desc, int block_idx)
{
    const pulseg_block_table_element *bte;
    const pulseg_base_block *bdef;
    const pulseg_rf_definition *rdef;
    const pulseq_shape *shape;
    int shape_idx, nch;

    bte = &desc->block_table[block_idx];
    bdef = &desc->base_blocks[bte->id];
    if (bdef->rf_id < 0)
        return 0;
    rdef = &desc->rf_definitions[bdef->rf_id];
    if (rdef->mag_shape_id <= 0)
        return 0;
    shape_idx = rdef->mag_shape_id - 1;
    if (shape_idx < 0 || shape_idx >= desc->num_shapes)
        return 0;
    shape = &desc->shapes[shape_idx];
    nch = (rdef->num_channels > 1) ? rdef->num_channels : 1;
    /* samples + 2 zero-pad boundary samples per channel */
    return shape->num_uncompressed_samples + 2 * nch;
}

/*
 * Fill RF waveform for a single block (flat block index).
 * Writes into time_mag[], mag[], phase[] at start_idx.
 * Returns number of samples written.
 */
static int fill_rf_waveform_for_flat_block(
    const pulseg_sequence_descriptor *desc,
    float *time_mag,
    float *mag,
    float *phase,
    int *out_nch,
    int block_idx,
    int start_idx,
    float t0)
{
    const pulseg_block_table_element *bte;
    const pulseg_base_block *bdef;
    const pulseg_rf_definition *rdef;
    const pulseg_rf_table_element *rtab;
    pulseq_shape decomp_mag, decomp_phase, decomp_time;
    float rf_raster_us, delay_us, amp;
    int nch, npts, i, idx, c, src_i;
    int has_time_shape;

    bte = &desc->block_table[block_idx];
    bdef = &desc->base_blocks[bte->id];
    if (bdef->rf_id < 0)
        return 0;
    rdef = &desc->rf_definitions[bdef->rf_id];
    if (rdef->mag_shape_id <= 0)
        return 0;

    /* instance-level params */
    if (bte->rf_id < 0 || bte->rf_id >= desc->rf_table_size)
        return 0;
    rtab = &desc->rf_table[bte->rf_id];
    amp = rtab->amplitude; /* Hz */

    rf_raster_us = desc->rf_raster_us;
    delay_us = (float)rdef->delay;
    nch = (rdef->num_channels > 1) ? rdef->num_channels : 1;

    /* decompress magnitude shape */
    decomp_mag.samples = NULL;
    decomp_mag.num_uncompressed_samples = 0;
    if (!pulseq_decompress_shape(&decomp_mag, &desc->shapes[rdef->mag_shape_id - 1], 1.0f))
        return 0;
    npts = decomp_mag.num_uncompressed_samples / nch;

    /* decompress phase shape */
    decomp_phase.samples = NULL;
    decomp_phase.num_uncompressed_samples = 0;
    if (rdef->phase_shape_id > 0 && rdef->phase_shape_id <= desc->num_shapes)
    {
        pulseq_decompress_shape(
            &decomp_phase,
            &desc->shapes[rdef->phase_shape_id - 1],
            (float)PULSEG__TWO_PI);
    }

    /* decompress time shape */
    decomp_time.samples = NULL;
    decomp_time.num_uncompressed_samples = 0;
    has_time_shape = 0;
    if (rdef->time_shape_id > 0 && rdef->time_shape_id <= desc->num_shapes)
    {
        if (pulseq_decompress_shape(
                &decomp_time,
                &desc->shapes[rdef->time_shape_id - 1],
                rf_raster_us))
            has_time_shape = 1;
    }

    /* fill all channels (channel-major order: ch0[0..npts-1], ch1[0..npts-1], ...)
     * Each channel is bracketed by zero-pad samples at ±0.5*rf_raster_us
     * from the first/last RF sample to prevent interpolation artifacts
     * across inter-block gaps (e.g. MPRAGE inversion → sinc). */
    idx = start_idx;
    for (c = 0; c < nch; ++c)
    {
        float first_t_local, last_t_local;
        /* Compute first and last t_local for boundary padding */
        if (has_time_shape && decomp_time.num_uncompressed_samples > 0)
            first_t_local = decomp_time.samples[0];
        else
            first_t_local = 0.5f * rf_raster_us;
        if (has_time_shape && npts > 0 && (npts - 1) < decomp_time.num_uncompressed_samples)
            last_t_local = decomp_time.samples[npts - 1];
        else
            last_t_local = 0.5f * rf_raster_us + (float)(npts - 1) * rf_raster_us;

        /* Pre-pad zero */
        time_mag[idx] = t0 + delay_us + first_t_local - 0.5f * rf_raster_us;
        mag[idx] = 0.0f;
        phase[idx] = 0.0f;
        idx++;

        for (i = 0; i < npts; ++i)
        {
            float t_local; /* time from RF pulse start, µs – matches pypulseq rf.t */
            float freq_term;
            src_i = c * npts + i;
            if (has_time_shape && i < decomp_time.num_uncompressed_samples)
                t_local = decomp_time.samples[i];
            else
                t_local = 0.5f * rf_raster_us + (float)i * rf_raster_us;
            freq_term = 2.0f * (float)M_PI * rtab->freq_offset * (t_local * 1e-6f);

            time_mag[idx] = t0 + delay_us + t_local;
            mag[idx] = amp * decomp_mag.samples[src_i]; /* Hz */
            phase[idx] = (decomp_phase.samples && src_i < decomp_phase.num_uncompressed_samples)
                ? decomp_phase.samples[src_i] + rtab->phase_offset + freq_term
                : rtab->phase_offset + freq_term; /* rad */
            idx++;
        }

        /* Post-pad zero */
        time_mag[idx] = t0 + delay_us + last_t_local + 0.5f * rf_raster_us;
        mag[idx] = 0.0f;
        phase[idx] = 0.0f;
        idx++;
    }
    if (out_nch)
        *out_nch = nch;

    if (decomp_mag.samples)
        PULSEG_FREE(decomp_mag.samples);
    if (decomp_phase.samples)
        PULSEG_FREE(decomp_phase.samples);
    if (decomp_time.samples)
        PULSEG_FREE(decomp_time.samples);
    return idx - start_idx;
}

/*
 * Find which segment a block at position pos_in_tr belongs to.
 * Returns segment index (into segment_definitions), or -1.
 */
static int find_segment_for_block_pos(
    const pulseg_virtual_segment *segments,
    int num_segments,
    int pos_in_tr)
{
    int s;
    for (s = 0; s < num_segments; ++s)
    {
        if (pos_in_tr >= segments[s].start_block &&
            pos_in_tr < segments[s].start_block + segments[s].num_blocks)
            return s;
    }
    return -1;
}

/* ---- public: get_tr_waveforms ---- */
int pulseg_get_tr_waveforms(
    const pulseg_collection *coll,
    pulseg_tr_waveforms *out,
    pulseg_diagnostic *diag,
    int subseq_idx,
    int amplitude_mode,
    int tr_index,
    int collapse_delays,
    int num_averages)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_descriptor *tr;
    pulseg_diagnostic local_diag;
    int block_start, block_count, tr_block_start;
    int has_nd_prep, has_nd_cool;
    int main_region_start, main_region_end;
    int n, block_idx, k;
    int total_gx, total_gy, total_gz, total_rf, total_adc;
    int idx_gx, idx_gy, idx_gz, idx_rf, idx_adc, rf_nch, this_nch;
    float t0, block_dur_us;
    const pulseg_block_table_element *bte;
    const pulseg_base_block *bdef;
    const pulseg_grad_definition *gx_def;
    const pulseg_grad_definition *gy_def;
    const pulseg_grad_definition *gz_def;
    const pulseg_grad_table_element *gx_tab;
    const pulseg_grad_table_element *gy_tab;
    const pulseg_grad_table_element *gz_tab;
    int gx_raw, gy_raw, gz_raw;
    float *pos_max_gx;
    float *pos_max_gy;
    float *pos_max_gz;
    float actual_amp[PULSEG_MAX_GRAD_SHOTS];
    /* rotation post-pass variables */
    int interp_result, n_uniform, blk_n, rot_id, s;
    float target_raster_us, blk_end, t_sample_rot, vec[3], rot_out[3];
    const float *R;
    /* average-expansion variables */
    int *block_order;
    int pass_base;
    int pass_scan_start; /* scan-table offset for output slot 0 of this pass */
    int eff_num_averages;
    /* occurrence-position map: 0-based position within each segment occurrence */
    int *scan_blk_in_occ;

    pos_max_gx = NULL;
    pos_max_gy = NULL;
    pos_max_gz = NULL;
    block_order = NULL;
    scan_blk_in_occ = NULL;
    pass_base = 0;
    pass_scan_start = -1;

    if (!diag)
    {
        pulseg_diagnostic_init(&local_diag);
        diag = &local_diag;
    }
    else
    {
        pulseg_diagnostic_init(diag);
    }

    if (!coll || !out)
    {
        diag->code = PULSEG_ERR_NULL_POINTER;
        return diag->code;
    }
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
    {
        diag->code = PULSEG_ERR_INVALID_ARGUMENT;
        return diag->code;
    }
    memset(out, 0, sizeof(*out));

    desc = &coll->descriptors[subseq_idx];
    tr = &desc->tr_descriptor;
    eff_num_averages = (num_averages > 0) ? num_averages : desc->num_averages;
    has_nd_prep = (tr->num_prep_blocks > 0 && !tr->degenerate_prep);
    has_nd_cool = (tr->num_cooldown_blocks > 0 && !tr->degenerate_cooldown);

    if (tr->tr_size <= 0)
    {
        diag->code = PULSEG_ERR_TR_NO_BLOCKS;
        return diag->code;
    }

    /* ---- determine block range ---- */
    if (amplitude_mode == PULSEG_AMP_ACTUAL)
    {
        if (has_nd_prep || has_nd_cool)
        {
            /* Non-degenerate: tr_index selects a pass (0..num_passes-1).
             * Return the full pass with average expansion. */
            int num_avgs = eff_num_averages;
            if (tr_index < 0 || tr_index >= desc->num_passes)
            {
                diag->code = PULSEG_ERR_INVALID_ARGUMENT;
                return diag->code;
            }
            pass_base = tr_index * desc->pass_len;
            tr_block_start = pass_base + tr->num_prep_blocks + tr->imaging_tr_start;
            if (num_avgs > 1)
            {
                int prep_blk = tr->num_prep_blocks;
                int img_len = tr->num_trs * tr->tr_size;
                int cool_blk = tr->num_cooldown_blocks;
                int exp_count = prep_blk + num_avgs * img_len + cool_blk;
                int avg_i;
                block_order = (int *)PULSEG_ALLOC((size_t)exp_count * sizeof(int));
                if (!block_order)
                    goto alloc_fail;
                n = 0;
                for (k = 0; k < prep_blk; ++k)
                    block_order[n++] = pass_base + k;
                for (avg_i = 0; avg_i < num_avgs; ++avg_i)
                    for (k = 0; k < img_len; ++k)
                        block_order[n++] = pass_base + prep_blk + k;
                for (k = 0; k < cool_blk; ++k)
                    block_order[n++] = pass_base + prep_blk + img_len + k;
                block_start = 0;
                block_count = exp_count;
                /* scan-table position for output slot 0 of this pass:
                 * each pass occupies exp_count consecutive scan entries. */
                pass_scan_start = (pass_base / desc->pass_len) * block_count;
            }
            else
            {
                block_start = pass_base;
                block_count = desc->pass_len;
            }
        }
        else
        {
            /* Degenerate / no prep: flat TR index with average expansion.
             * Imaging TRs wrap modulo num_trs so repeated averages map
             * back to canonical block positions. */
            int num_avgs = eff_num_averages;
            int total_actual_trs = tr->num_prep_trs + num_avgs * tr->num_trs + tr->num_cooldown_trs;
            int canonical_idx;
            if (tr_index < 0 || tr_index >= total_actual_trs)
            {
                diag->code = PULSEG_ERR_INVALID_ARGUMENT;
                return diag->code;
            }
            if (tr_index < tr->num_prep_trs)
            {
                canonical_idx = tr_index;
            }
            else if (tr_index < tr->num_prep_trs + num_avgs * tr->num_trs)
            {
                canonical_idx = tr->num_prep_trs + (tr_index - tr->num_prep_trs) % tr->num_trs;
            }
            else
            {
                canonical_idx = tr->num_prep_trs + tr->num_trs +
                    (tr_index - tr->num_prep_trs - num_avgs * tr->num_trs);
            }
            tr_block_start = canonical_idx * tr->tr_size;
            block_start = tr_block_start;
            block_count = tr->tr_size;
        }
    }
    else
    {
        int *can_unique_indices = NULL;
        int *can_group_labels = NULL;
        int num_canonical = 0;
        int rep_idx = 0;

        /*
         * Canonical waveform extraction used by MAX_POS / ZERO_VAR:
         * select the representative instance for the requested canonical
         * group so geometry (RF/ADC placement, delays, etc.) matches the
         * same canonical TR used for amplitude filtering.
         */
        if (has_nd_prep || has_nd_cool)
        {
            num_canonical =
                pulseg__find_unique_shot_passes(desc, &can_unique_indices, &can_group_labels);
        }
        else
        {
            num_canonical =
                pulseg__find_unique_shot_trs(desc, &can_unique_indices, &can_group_labels);
        }

        if (num_canonical > 0 && tr_index >= 0 && tr_index < num_canonical && can_unique_indices)
        {
            rep_idx = can_unique_indices[tr_index];
        }

        if (has_nd_prep || has_nd_cool)
        {
            int num_avgs_c = eff_num_averages;
            pass_base = rep_idx * desc->pass_len;
            tr_block_start = pass_base + tr->num_prep_blocks + tr->imaging_tr_start;

            if (num_avgs_c > 1)
            {
                /* Average-expanded canonical pass:
                 * prep + num_avgs × imaging + cooldown. */
                int prep_blk_c = tr->num_prep_blocks;
                int img_len_c = tr->num_trs * tr->tr_size;
                int cool_blk_c = tr->num_cooldown_blocks;
                int exp_count_c = prep_blk_c + num_avgs_c * img_len_c + cool_blk_c;
                int avg_i_c;
                block_order = (int *)PULSEG_ALLOC((size_t)exp_count_c * sizeof(int));
                if (!block_order)
                {
                    if (can_group_labels)
                        PULSEG_FREE(can_group_labels);
                    if (can_unique_indices)
                        PULSEG_FREE(can_unique_indices);
                    goto alloc_fail;
                }
                n = 0;
                for (k = 0; k < prep_blk_c; ++k)
                    block_order[n++] = pass_base + k;
                for (avg_i_c = 0; avg_i_c < num_avgs_c; ++avg_i_c)
                    for (k = 0; k < img_len_c; ++k)
                        block_order[n++] = pass_base + prep_blk_c + k;
                for (k = 0; k < cool_blk_c; ++k)
                    block_order[n++] = pass_base + prep_blk_c + img_len_c + k;
                block_start = 0;
                block_count = exp_count_c;
                /* scan-table position for output slot 0 of this canonical pass */
                pass_scan_start = (pass_base / desc->pass_len) * block_count;
            }
            else
            {
                block_order = (int *)PULSEG_ALLOC((size_t)desc->pass_len * sizeof(int));
                if (!block_order)
                {
                    if (can_group_labels)
                        PULSEG_FREE(can_group_labels);
                    if (can_unique_indices)
                        PULSEG_FREE(can_unique_indices);
                    goto alloc_fail;
                }
                for (n = 0; n < desc->pass_len; ++n)
                    block_order[n] = pass_base + n;
                block_start = 0;
                block_count = desc->pass_len;
            }
        }
        else
        {
            int base_start = tr->num_prep_blocks + tr->imaging_tr_start;
            tr_block_start = base_start;
            block_start = base_start;
            block_count = tr->tr_size;

            if (rep_idx > 0)
            {
                block_order = (int *)PULSEG_ALLOC((size_t)tr->tr_size * sizeof(int));
                if (!block_order)
                {
                    if (can_group_labels)
                        PULSEG_FREE(can_group_labels);
                    if (can_unique_indices)
                        PULSEG_FREE(can_unique_indices);
                    goto alloc_fail;
                }
                for (n = 0; n < tr->tr_size; ++n)
                    block_order[n] = base_start + rep_idx * tr->tr_size + n;
            }
        }

        if (can_group_labels)
            PULSEG_FREE(can_group_labels);
        if (can_unique_indices)
            PULSEG_FREE(can_unique_indices);
    }

    if (!block_order && (block_start < 0 || block_start + block_count > desc->num_blocks))
    {
        diag->code = PULSEG_ERR_INVALID_ARGUMENT;
        return diag->code;
    }

    /* ---- precompute position-max if needed ---- */
    if (amplitude_mode == PULSEG_AMP_MAX_POS || amplitude_mode == PULSEG_AMP_ZERO_VAR)
    {
        int *can_group_labels = NULL;
        int *can_unique_indices = NULL;
        int num_canonical = 0;

        pos_max_gx =
            (float *)PULSEG_ALLOC((size_t)block_count * PULSEG_MAX_GRAD_SHOTS * sizeof(float));
        pos_max_gy =
            (float *)PULSEG_ALLOC((size_t)block_count * PULSEG_MAX_GRAD_SHOTS * sizeof(float));
        pos_max_gz =
            (float *)PULSEG_ALLOC((size_t)block_count * PULSEG_MAX_GRAD_SHOTS * sizeof(float));
        if (!pos_max_gx || !pos_max_gy || !pos_max_gz)
            goto alloc_fail;

        /* If tr_index > 0, filter max-pos to that canonical TR group. */
        if (tr_index >= 0)
        {
            if (has_nd_prep || has_nd_cool)
            {
                num_canonical =
                    pulseg__find_unique_shot_passes(desc, &can_unique_indices, &can_group_labels);
            }
            else
            {
                num_canonical =
                    pulseg__find_unique_shot_trs(desc, &can_unique_indices, &can_group_labels);
            }
        }

        if (num_canonical > 1 && tr_index >= 0 && tr_index < num_canonical)
        {
            compute_position_max_amplitudes_filtered(
                desc,
                pos_max_gx,
                pos_max_gy,
                pos_max_gz,
                block_start,
                block_count,
                can_group_labels,
                tr_index);
        }
        else
        {
            compute_position_max_amplitudes_filtered(
                desc,
                pos_max_gx,
                pos_max_gy,
                pos_max_gz,
                block_start,
                block_count,
                NULL,
                0);
        }

        if (can_group_labels)
            PULSEG_FREE(can_group_labels);
        if (can_unique_indices)
            PULSEG_FREE(can_unique_indices);

        /* ZERO_VAR: zero out positions whose gradients vary across TRs.
         * For non-degenerate passes, iterate over ALL imaging positions
         * in the (possibly average-expanded) canonical layout and map
         * each back to its TR-relative position for the flag lookup. */
        if (amplitude_mode == PULSEG_AMP_ZERO_VAR && desc->variable_grad_flags)
        {
            int vp, local_pos, abs_pos;
            int zv_prep = (has_nd_prep || has_nd_cool) ? tr->num_prep_blocks : 0;
            int zv_cool = (has_nd_prep || has_nd_cool) ? tr->num_cooldown_blocks : 0;
            int zv_img = block_count - zv_prep - zv_cool;
            for (vp = 0; vp < zv_img; ++vp)
            {
                local_pos = vp % tr->tr_size;
                abs_pos = zv_prep + vp;
                if (desc->variable_grad_flags[local_pos * 3 + 0])
                    for (k = 0; k < PULSEG_MAX_GRAD_SHOTS; ++k)
                        pos_max_gx[abs_pos * PULSEG_MAX_GRAD_SHOTS + k] = 0.0f;
                if (desc->variable_grad_flags[local_pos * 3 + 1])
                    for (k = 0; k < PULSEG_MAX_GRAD_SHOTS; ++k)
                        pos_max_gy[abs_pos * PULSEG_MAX_GRAD_SHOTS + k] = 0.0f;
                if (desc->variable_grad_flags[local_pos * 3 + 2])
                    for (k = 0; k < PULSEG_MAX_GRAD_SHOTS; ++k)
                        pos_max_gz[abs_pos * PULSEG_MAX_GRAD_SHOTS + k] = 0.0f;
            }
        }
    }

    /* ---- PASS 1: count samples ---- */
    total_gx = 0;
    total_gy = 0;
    total_gz = 0;
    total_rf = 0;
    total_adc = 0;
    for (n = 0; n < block_count; ++n)
    {
        block_idx = block_order ? block_order[n] : block_start + n;
        bte = &desc->block_table[block_idx];
        bdef = &desc->base_blocks[bte->id];
        block_dur_us = (bte->duration_us >= 0) ? (float)bte->duration_us : (float)bdef->duration_us;

        gx_raw = bte->gx_id;
        gy_raw = bte->gy_id;
        gz_raw = bte->gz_id;
        gx_def =
            (gx_raw >= 0 && gx_raw < desc->grad_table_size && desc->grad_table[gx_raw].id >= 0 &&
             desc->grad_table[gx_raw].id < desc->num_unique_grads)
            ? &desc->grad_definitions[desc->grad_table[gx_raw].id]
            : NULL;
        gy_def =
            (gy_raw >= 0 && gy_raw < desc->grad_table_size && desc->grad_table[gy_raw].id >= 0 &&
             desc->grad_table[gy_raw].id < desc->num_unique_grads)
            ? &desc->grad_definitions[desc->grad_table[gy_raw].id]
            : NULL;
        gz_def =
            (gz_raw >= 0 && gz_raw < desc->grad_table_size && desc->grad_table[gz_raw].id >= 0 &&
             desc->grad_table[gz_raw].id < desc->num_unique_grads)
            ? &desc->grad_definitions[desc->grad_table[gz_raw].id]
            : NULL;

        total_gx += count_grad_samples_for_block(desc, gx_def, block_dur_us);
        total_gy += count_grad_samples_for_block(desc, gy_def, block_dur_us);
        total_gz += count_grad_samples_for_block(desc, gz_def, block_dur_us);
        total_rf += count_rf_samples_for_flat_block(desc, block_idx);

        if (amplitude_mode == PULSEG_AMP_ACTUAL)
        {
            if (bte->adc_id >= 0 && bte->adc_id < desc->adc_table_size)
                total_adc++;
        }
        else
        {
            if (bdef->adc_id >= 0 && bdef->adc_id < desc->num_unique_adcs)
                total_adc++;
        }
    }

    /* ---- allocate ---- */
    out->gx.time_us = (float *)PULSEG_ALLOC((size_t)total_gx * sizeof(float));
    out->gx.amplitude = (float *)PULSEG_ALLOC((size_t)total_gx * sizeof(float));
    out->gy.time_us = (float *)PULSEG_ALLOC((size_t)total_gy * sizeof(float));
    out->gy.amplitude = (float *)PULSEG_ALLOC((size_t)total_gy * sizeof(float));
    out->gz.time_us = (float *)PULSEG_ALLOC((size_t)total_gz * sizeof(float));
    out->gz.amplitude = (float *)PULSEG_ALLOC((size_t)total_gz * sizeof(float));

    if (total_rf > 0)
    {
        out->rf_mag.time_us = (float *)PULSEG_ALLOC((size_t)total_rf * sizeof(float));
        out->rf_mag.amplitude = (float *)PULSEG_ALLOC((size_t)total_rf * sizeof(float));
        out->rf_phase.time_us = (float *)PULSEG_ALLOC((size_t)total_rf * sizeof(float));
        out->rf_phase.amplitude = (float *)PULSEG_ALLOC((size_t)total_rf * sizeof(float));
    }
    if (total_adc > 0)
    {
        out->adc_events =
            (pulseg_adc_event *)PULSEG_ALLOC((size_t)total_adc * sizeof(pulseg_adc_event));
    }
    out->blocks = (pulseg_tr_block_descriptor *)PULSEG_ALLOC(
        (size_t)block_count * sizeof(pulseg_tr_block_descriptor));

    /* check allocations (simplified: check critical ones) */
    if (!out->gx.time_us || !out->gx.amplitude || !out->gy.time_us || !out->gy.amplitude ||
        !out->gz.time_us || !out->gz.amplitude || !out->blocks ||
        (total_rf > 0 &&
         (!out->rf_mag.time_us || !out->rf_mag.amplitude || !out->rf_phase.time_us ||
          !out->rf_phase.amplitude)) ||
        (total_adc > 0 && !out->adc_events))
        goto alloc_fail;

    /* ---- scan_blk_in_occ: 0-based position within current segment occurrence ----
     * exec_stream_seg_id is populated for ALL block positions (including 2nd, 3rd, …
     * occurrences of the same unique segment), unlike segment_definitions[s].start_block
     * which only records the first occurrence.  We iterate the scan table once to
     * compute the intra-occurrence offset for every scan-table position, resetting
     * the counter whenever the segment changes OR we've filled num_blocks slots. */
    if (desc->exec_stream_seg_id && desc->exec_stream_len > 0)
    {
        int occ_pos = 0, prev_seg = -1, si;
        scan_blk_in_occ = (int *)PULSEG_ALLOC((size_t)desc->exec_stream_len * sizeof(int));
        if (!scan_blk_in_occ)
            goto alloc_fail;
        for (si = 0; si < desc->exec_stream_len; ++si)
        {
            int curr_seg = desc->exec_stream_seg_id[si];
            int seg_nblk = (curr_seg >= 0 && curr_seg < desc->segment_table.num_unique_segments)
                ? desc->segment_definitions[curr_seg].num_blocks
                : 1;
            if (curr_seg != prev_seg || occ_pos >= seg_nblk)
                occ_pos = 0;
            scan_blk_in_occ[si] = occ_pos++;
            prev_seg = curr_seg;
        }
    }

    /* ---- PASS 2: fill ---- */
    t0 = 0.0f;
    idx_gx = 0;
    idx_gy = 0;
    idx_gz = 0;
    idx_rf = 0;
    idx_adc = 0;
    rf_nch = 1;

    (void)main_region_start;
    (void)main_region_end;

    for (n = 0; n < block_count; ++n)
    {
        block_idx = block_order ? block_order[n] : block_start + n;
        bte = &desc->block_table[block_idx];
        bdef = &desc->base_blocks[bte->id];
        block_dur_us = (bte->duration_us >= 0) ? (float)bte->duration_us : (float)bdef->duration_us;

        /* Pure-delay clamping: if collapse_delays is set and this block
         * has no RF, no gradients, and no ADC, force its display duration
         * to exactly 5000 µs (5 ms) — both expand short delays and shrink
         * long ones so every delay is visible at a uniform width. */
        if (collapse_delays && bdef->rf_id < 0 && bte->gx_id < 0 && bte->gy_id < 0 &&
            bte->gz_id < 0 && bdef->adc_id < 0)
        {
            block_dur_us = 5000.0f; /* 5 ms display duration */
        }

        /* block metadata */
        out->blocks[n].start_us = t0;
        out->blocks[n].duration_us = block_dur_us;

        /* Segment assignment: use exec_stream_seg_id at the correct scan-table
         * position.  For degenerate / single-pass paths block_idx equals the
         * scan-table position (identity mapping, avg=0).  For average-expanded
         * non-degenerate passes pass_scan_start is set to the first scan-table
         * slot for this pass so pass_scan_start+n gives the exact NEX-aware
         * position even when block_order[n] repeats block-table indices across
         * multiple averages. */
        {
            int seg_id = -1;
            int scan_pos = (pass_scan_start >= 0) ? (pass_scan_start + n) : block_idx;
            if (desc->exec_stream_seg_id && scan_pos >= 0 && scan_pos < desc->exec_stream_len)
                seg_id = desc->exec_stream_seg_id[scan_pos];
            else
                seg_id = find_segment_for_block_pos(
                    desc->segment_definitions,
                    desc->segment_table.num_unique_segments,
                    n);
            out->blocks[n].segment_idx = seg_id;
        }

        /* ---- RF isocenter anchor ---- */
        out->blocks[n].rf_isocenter_us = -1.0f;
        if (bdef->rf_id >= 0 && bdef->rf_id < desc->num_unique_rfs)
        {
            const pulseg_rf_definition *rdef = &desc->rf_definitions[bdef->rf_id];
            out->blocks[n].rf_isocenter_us =
                t0 + (float)rdef->delay + (float)rdef->stats.isodelay_us;
        }

        /* ---- ADC k=0 anchor ---- */
        out->blocks[n].adc_kzero_us = -1.0f;
        {
            int has_adc_here = 0;
            const pulseg_adc_definition *adef_anchor = NULL;
            if (amplitude_mode == PULSEG_AMP_ACTUAL)
            {
                if (bte->adc_id >= 0 && bte->adc_id < desc->adc_table_size)
                {
                    int adc_def_id = desc->adc_table[bte->adc_id].id;
                    if (adc_def_id >= 0 && adc_def_id < desc->num_unique_adcs)
                    {
                        adef_anchor = &desc->adc_definitions[adc_def_id];
                        has_adc_here = 1;
                    }
                }
            }
            else
            {
                if (bdef->adc_id >= 0 && bdef->adc_id < desc->num_unique_adcs)
                {
                    adef_anchor = &desc->adc_definitions[bdef->adc_id];
                    has_adc_here = 1;
                }
            }
            if (has_adc_here && adef_anchor)
            {
                int seg_i = out->blocks[n].segment_idx;
                int found_anchor = 0;
                /* Try precomputed segment adc_anchors first. */
                if (seg_i >= 0 && seg_i < desc->segment_table.num_unique_segments)
                {
                    const pulseg_virtual_segment *seg = &desc->segment_definitions[seg_i];
                    if (seg->timing.adc_anchors && seg->timing.num_adc_anchors > 0)
                    {
                        /* blk_in_seg: position of this block within the current
                         * segment occurrence.  Use scan_blk_in_occ indexed at
                         * the same NEX-aware scan position used for segment
                         * assignment above.  For degenerate paths scan_pos ==
                         * block_idx; for average-expanded non-degenerate passes
                         * scan_pos = pass_scan_start + n correctly tracks the
                         * position within the expanded occurrence. */
                        int scan_pos2 = (pass_scan_start >= 0) ? (pass_scan_start + n) : block_idx;
                        int blk_in_seg =
                            (scan_blk_in_occ && scan_pos2 >= 0 && scan_pos2 < desc->exec_stream_len)
                            ? scan_blk_in_occ[scan_pos2]
                            : (n - seg->start_block);
                        int ai;
                        /* Compute segment start in TR-relative time. */
                        float seg_start_tr = t0;
                        {
                            int bi;
                            for (bi = 0; bi < blk_in_seg; ++bi)
                            {
                                seg_start_tr -=
                                    (float)desc->base_blocks[seg->unique_block_indices[bi]]
                                        .duration_us;
                            }
                        }
                        for (ai = 0; ai < seg->timing.num_adc_anchors; ++ai)
                        {
                            if (seg->timing.adc_anchors[ai].block_offset == blk_in_seg)
                            {
                                out->blocks[n].adc_kzero_us =
                                    seg_start_tr + seg->timing.adc_anchors[ai].kzero_us;
                                found_anchor = 1;
                                break;
                            }
                        }
                    }
                }
                /* No fallback: segment anchors must be correct. */
                if (!found_anchor)
                {
                    out->blocks[n].adc_kzero_us = -1.0f;
                    if (diag)
                    {
                        diag->code = PULSEG_ERR_INVALID_ARGUMENT;
                        sprintf(diag->message, "ADC anchor not found for block %d", n);
                    }
                    return PULSEG_ERR_INVALID_ARGUMENT;
                }
            }
        }

        /* ---- gradients ---- */
        gx_raw = bte->gx_id;
        gy_raw = bte->gy_id;
        gz_raw = bte->gz_id;
        gx_tab = (gx_raw >= 0 && gx_raw < desc->grad_table_size) ? &desc->grad_table[gx_raw] : NULL;
        gy_tab = (gy_raw >= 0 && gy_raw < desc->grad_table_size) ? &desc->grad_table[gy_raw] : NULL;
        gz_tab = (gz_raw >= 0 && gz_raw < desc->grad_table_size) ? &desc->grad_table[gz_raw] : NULL;

        gx_def = (gx_tab && gx_tab->id >= 0 && gx_tab->id < desc->num_unique_grads)
            ? &desc->grad_definitions[gx_tab->id]
            : NULL;
        gy_def = (gy_tab && gy_tab->id >= 0 && gy_tab->id < desc->num_unique_grads)
            ? &desc->grad_definitions[gy_tab->id]
            : NULL;
        gz_def = (gz_tab && gz_tab->id >= 0 && gz_tab->id < desc->num_unique_grads)
            ? &desc->grad_definitions[gz_tab->id]
            : NULL;

        if (pos_max_gx)
        {
            idx_gx += fill_grad_waveform_for_block(
                desc,
                out->gx.time_us,
                out->gx.amplitude,
                idx_gx,
                gx_def,
                gx_tab,
                t0,
                &pos_max_gx[n * PULSEG_MAX_GRAD_SHOTS],
                block_dur_us);
            idx_gy += fill_grad_waveform_for_block(
                desc,
                out->gy.time_us,
                out->gy.amplitude,
                idx_gy,
                gy_def,
                gy_tab,
                t0,
                &pos_max_gy[n * PULSEG_MAX_GRAD_SHOTS],
                block_dur_us);
            idx_gz += fill_grad_waveform_for_block(
                desc,
                out->gz.time_us,
                out->gz.amplitude,
                idx_gz,
                gz_def,
                gz_tab,
                t0,
                &pos_max_gz[n * PULSEG_MAX_GRAD_SHOTS],
                block_dur_us);
        }
        else
        {
            /* PULSEG_AMP_ACTUAL: use per-instance amplitude */
            for (k = 0; k < PULSEG_MAX_GRAD_SHOTS; ++k)
                actual_amp[k] = 0.0f;
            if (gx_tab)
            {
                k = gx_tab->shot_index;
                if (k >= 0 && k < PULSEG_MAX_GRAD_SHOTS)
                {
                    actual_amp[k] = gx_tab->amplitude;
                }
            }
            idx_gx += fill_grad_waveform_for_block(
                desc,
                out->gx.time_us,
                out->gx.amplitude,
                idx_gx,
                gx_def,
                gx_tab,
                t0,
                actual_amp,
                block_dur_us);

            for (k = 0; k < PULSEG_MAX_GRAD_SHOTS; ++k)
                actual_amp[k] = 0.0f;
            if (gy_tab)
            {
                k = gy_tab->shot_index;
                if (k >= 0 && k < PULSEG_MAX_GRAD_SHOTS)
                {
                    actual_amp[k] = gy_tab->amplitude;
                }
            }
            idx_gy += fill_grad_waveform_for_block(
                desc,
                out->gy.time_us,
                out->gy.amplitude,
                idx_gy,
                gy_def,
                gy_tab,
                t0,
                actual_amp,
                block_dur_us);

            for (k = 0; k < PULSEG_MAX_GRAD_SHOTS; ++k)
                actual_amp[k] = 0.0f;
            if (gz_tab)
            {
                k = gz_tab->shot_index;
                if (k >= 0 && k < PULSEG_MAX_GRAD_SHOTS)
                {
                    actual_amp[k] = gz_tab->amplitude;
                }
            }
            idx_gz += fill_grad_waveform_for_block(
                desc,
                out->gz.time_us,
                out->gz.amplitude,
                idx_gz,
                gz_def,
                gz_tab,
                t0,
                actual_amp,
                block_dur_us);
        }

        /* ---- RF ---- */
        this_nch = 1;
        idx_rf += fill_rf_waveform_for_flat_block(
            desc,
            out->rf_mag.time_us,
            out->rf_mag.amplitude,
            out->rf_phase.amplitude,
            &this_nch,
            block_idx,
            idx_rf,
            t0);
        if (this_nch > rf_nch)
            rf_nch = this_nch;

        /* ---- ADC ---- */
        if (amplitude_mode == PULSEG_AMP_ACTUAL)
        {
            /* ACTUAL mode: per-instance ADC from block table */
            if (bte->adc_id >= 0 && bte->adc_id < desc->adc_table_size)
            {
                int adc_def_id = desc->adc_table[bte->adc_id].id;
                if (adc_def_id >= 0 && adc_def_id < desc->num_unique_adcs)
                {
                    const pulseg_adc_definition *adef = &desc->adc_definitions[adc_def_id];
                    pulseg_adc_event *ev = &out->adc_events[idx_adc];
                    ev->onset_us = t0 + (float)adef->delay;
                    ev->duration_us = (float)adef->num_samples * (float)adef->dwell_time * 1e-3f;
                    ev->num_samples = adef->num_samples;
                    ev->freq_offset_hz = desc->adc_table[bte->adc_id].freq_offset;
                    ev->phase_offset_rad = desc->adc_table[bte->adc_id].phase_offset;
                    idx_adc++;
                }
            }
        }
        else
        {
            /* MAX_POS / ZERO_VAR: canonical ADC from block definition */
            if (bdef->adc_id >= 0 && bdef->adc_id < desc->num_unique_adcs)
            {
                const pulseg_adc_definition *adef = &desc->adc_definitions[bdef->adc_id];
                pulseg_adc_event *ev = &out->adc_events[idx_adc];
                ev->onset_us = t0 + (float)adef->delay;
                ev->duration_us = (float)adef->num_samples * (float)adef->dwell_time * 1e-3f;
                ev->num_samples = adef->num_samples;
                ev->freq_offset_hz = 0.0f;
                ev->phase_offset_rad = 0.0f;
                idx_adc++;
            }
        }

        t0 += block_dur_us;
    }

    if (pos_max_gx)
        PULSEG_FREE(pos_max_gx);
    if (pos_max_gy)
        PULSEG_FREE(pos_max_gy);
    if (pos_max_gz)
        PULSEG_FREE(pos_max_gz);
    pos_max_gx = NULL;
    pos_max_gy = NULL;
    pos_max_gz = NULL;

    /* ---- Interpolate gradients to uniform 0.5 grad raster ----
     * After this, all three axes share the same time base, which
     * makes the rotation post-pass trivial. */
    {
        int num_gx = idx_gx, num_gy = idx_gy, num_gz = idx_gz;
        target_raster_us = 0.5f * desc->grad_raster_us;

        interp_result =
            interpolate_to_uniform(&out->gx.time_us, &out->gx.amplitude, &num_gx, target_raster_us);
        if (PULSEG_FAILED(interp_result))
        {
            diag->code = interp_result;
            return interp_result;
        }
        interp_result =
            interpolate_to_uniform(&out->gy.time_us, &out->gy.amplitude, &num_gy, target_raster_us);
        if (PULSEG_FAILED(interp_result))
        {
            diag->code = interp_result;
            return interp_result;
        }
        interp_result =
            interpolate_to_uniform(&out->gz.time_us, &out->gz.amplitude, &num_gz, target_raster_us);
        if (PULSEG_FAILED(interp_result))
        {
            diag->code = interp_result;
            return interp_result;
        }

        /* All three axes should have the same sample count; use min
         * as a safety clamp against float rounding. */
        n_uniform = num_gx;
        if (num_gy < n_uniform)
            n_uniform = num_gy;
        if (num_gz < n_uniform)
            n_uniform = num_gz;
        idx_gx = n_uniform;
        idx_gy = n_uniform;
        idx_gz = n_uniform;
    }

    /* ---- Rotation post-pass ----
     * All three grad axes now share the same uniform time base.
     * Walk through samples, find each sample's block, and apply R^T
     * if that block has a rotation. */
    blk_n = 0;
    blk_end = out->blocks[0].start_us + out->blocks[0].duration_us;
    for (s = 0; s < n_uniform; ++s)
    {
        t_sample_rot = out->gx.time_us[s];
        while (blk_n + 1 < block_count && t_sample_rot >= blk_end)
        {
            blk_n++;
            blk_end = out->blocks[blk_n].start_us + out->blocks[blk_n].duration_us;
        }
        bte = &desc->block_table[block_order ? block_order[blk_n] : block_start + blk_n];
        rot_id = bte->rotation_id;
        if (rot_id < 0 || rot_id >= desc->num_rotations)
            continue;
        if (bte->norot_flag)
            continue;
        R = desc->rotation_matrices[rot_id];
        vec[0] = out->gx.amplitude[s];
        vec[1] = out->gy.amplitude[s];
        vec[2] = out->gz.amplitude[s];
        pulseg__apply_rotation(rot_out, R, vec, 1);
        out->gx.amplitude[s] = rot_out[0];
        out->gy.amplitude[s] = rot_out[1];
        out->gz.amplitude[s] = rot_out[2];
    }

    /* rf_phase shares time_us with rf_mag */
    if (total_rf > 0)
    {
        memcpy(out->rf_phase.time_us, out->rf_mag.time_us, (size_t)idx_rf * sizeof(float));
    }

    out->gx.num_samples = idx_gx;
    out->gy.num_samples = idx_gy;
    out->gz.num_samples = idx_gz;
    out->rf_mag.num_samples = idx_rf;
    out->rf_phase.num_samples = idx_rf;
    out->num_rf_channels = rf_nch;
    out->num_adc_events = idx_adc;
    out->num_blocks = block_count;
    out->total_duration_us = t0;

    if (block_order)
        PULSEG_FREE(block_order);
    if (scan_blk_in_occ)
        PULSEG_FREE(scan_blk_in_occ);
    diag->code = PULSEG_SUCCESS;
    return PULSEG_SUCCESS;

alloc_fail:
    if (block_order)
        PULSEG_FREE(block_order);
    if (scan_blk_in_occ)
        PULSEG_FREE(scan_blk_in_occ);
    if (pos_max_gx)
        PULSEG_FREE(pos_max_gx);
    if (pos_max_gy)
        PULSEG_FREE(pos_max_gy);
    if (pos_max_gz)
        PULSEG_FREE(pos_max_gz);
    pulseg_tr_waveforms_free(out);
    diag->code = PULSEG_ERR_ALLOC_FAILED;
    return diag->code;
}

void pulseg_tr_waveforms_free(pulseg_tr_waveforms *w)
{
    if (!w)
        return;
    if (w->gx.time_us)
        PULSEG_FREE(w->gx.time_us);
    if (w->gx.amplitude)
        PULSEG_FREE(w->gx.amplitude);
    if (w->gy.time_us)
        PULSEG_FREE(w->gy.time_us);
    if (w->gy.amplitude)
        PULSEG_FREE(w->gy.amplitude);
    if (w->gz.time_us)
        PULSEG_FREE(w->gz.time_us);
    if (w->gz.amplitude)
        PULSEG_FREE(w->gz.amplitude);
    if (w->rf_mag.time_us)
        PULSEG_FREE(w->rf_mag.time_us);
    if (w->rf_mag.amplitude)
        PULSEG_FREE(w->rf_mag.amplitude);
    if (w->rf_phase.time_us)
        PULSEG_FREE(w->rf_phase.time_us);
    if (w->rf_phase.amplitude)
        PULSEG_FREE(w->rf_phase.amplitude);
    if (w->adc_events)
        PULSEG_FREE(w->adc_events);
    if (w->blocks)
        PULSEG_FREE(w->blocks);
    memset(w, 0, sizeof(*w));
}
