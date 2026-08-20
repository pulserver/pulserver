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

int pulseg__count_grad_samples_for_block(
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
 * Computes the per-position worst-case amplitude, preserving the sign of the
 * instance with the largest absolute value.  Considering only TR instances
 * whose group label matches target_group; if tr_group_labels is NULL, all TRs
 * are included.
 *
 * One value per position, not one per shape.  Which waveform that amplitude
 * belongs to is the definition's business -- it names its own worst instance
 * (pulseg_grad_representative) -- and a position plays one definition.
 *
 * Output arrays must be pre-allocated to block_count entries.
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
    int use_full_pass_layout, st_pos;
    const pulseg_block_table_element *bte;
    const pulseg_grad_table_element *gte;
    int raw_id, n;

    tr = &desc->tr_descriptor;
    tr_size = tr->tr_size;
    num_trs = tr->num_trs;
    use_full_pass_layout = !(block_start == 0 && block_count == tr_size);

    for (n = 0; n < block_count; ++n)
    {
        pos_max_gx[n] = 0.0f;
        pos_max_gy[n] = 0.0f;
        pos_max_gz[n] = 0.0f;
    }

    if (use_full_pass_layout && desc->exec_stream_len > 0 && desc->exec_runs)
    {
        if (!tr_group_labels || tr_group_labels[0] == target_group)
        {
            for (pos = 0; pos < block_count; ++pos)
            {
                st_pos = block_start + pos;
                if (st_pos < 0 || st_pos >= desc->exec_stream_len)
                    continue;

                block_idx = pulseg__exec_block_idx(desc, st_pos);
                if (block_idx < 0 || block_idx >= desc->num_blocks)
                    continue;
                bte = &desc->block_table[block_idx];

                raw_id = bte->gx_id;
                if (raw_id >= 0 && raw_id < desc->grad_table_size)
                {
                    gte = &desc->grad_table[raw_id];
                    if ((float)fabs((double)gte->amplitude) > (float)fabs((double)pos_max_gx[pos]))
                        pos_max_gx[pos] = gte->amplitude;
                }

                raw_id = bte->gy_id;
                if (raw_id >= 0 && raw_id < desc->grad_table_size)
                {
                    gte = &desc->grad_table[raw_id];
                    if ((float)fabs((double)gte->amplitude) > (float)fabs((double)pos_max_gy[pos]))
                        pos_max_gy[pos] = gte->amplitude;
                }

                raw_id = bte->gz_id;
                if (raw_id >= 0 && raw_id < desc->grad_table_size)
                {
                    gte = &desc->grad_table[raw_id];
                    if ((float)fabs((double)gte->amplitude) > (float)fabs((double)pos_max_gz[pos]))
                        pos_max_gz[pos] = gte->amplitude;
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

        tr_start = tr_idx * tr_size;
        for (pos = 0; pos < tr_size; ++pos)
        {
            block_idx = tr_start + pos;
            bte = &desc->block_table[block_idx];

            /* Gx */
            raw_id = bte->gx_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size)
            {
                gte = &desc->grad_table[raw_id];
                if ((float)fabs((double)gte->amplitude) > (float)fabs((double)pos_max_gx[pos]))
                    pos_max_gx[pos] = gte->amplitude;
            }

            /* Gy */
            raw_id = bte->gy_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size)
            {
                gte = &desc->grad_table[raw_id];
                if ((float)fabs((double)gte->amplitude) > (float)fabs((double)pos_max_gy[pos]))
                    pos_max_gy[pos] = gte->amplitude;
            }

            /* Gz */
            raw_id = bte->gz_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size)
            {
                gte = &desc->grad_table[raw_id];
                if ((float)fabs((double)gte->amplitude) > (float)fabs((double)pos_max_gz[pos]))
                    pos_max_gz[pos] = gte->amplitude;
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

        /* Walk the scan pulseg__exec_tr_start(table, si) == 1 marks the first
         * block of a new TR; we use this to reset the within-TR position.
         * This uses the full expanded scan table rather than the deduplicated
         * block table, so that per-TR gradient amplitude variation (e.g. phase
         * encoding steps) is correctly detected even after deduplication. */
        tr_pos = 0;
        for (si = 0; si < desc->exec_stream_len; ++si)
        {
            /* Reset position counter at the start of each new TR */
            if (pulseg__exec_tr_start(desc, si))
                tr_pos = 0;

            {
                int bt_idx = pulseg__exec_block_idx(desc, si);
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

/* ================================================================== */
/*  Fill waveform for a single block                                  */
/* ================================================================== */

int pulseg__fill_grad_waveform_for_block(
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
    float delay_us, t_sample;
    int shape_id, time_shape_id, num_samples;
    float rise_us, flat_us, fall_us;
    float grad_raster_us, block_end_us;
    float rel_end_us;
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

    max_amp = *pos_max_amp;
    sign = (max_amp >= 0.0f) ? 1.0f : -1.0f;
    if (max_amp < 0.0f)
        max_amp = -max_amp;
    delay_us = (float)gdef->delay;

    if (delay_us > 0.0f)
    {
        t_sample = t0;
        time[idx] = t_sample;
        waveform[idx] = 0.0f;
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
            idx++;

            t_sample = t0 + delay_us + rise_us;
            time[idx] = t_sample;
            waveform[idx] = sign * max_amp;
            idx++;

            t_sample = t0 + delay_us + rise_us + flat_us;
            time[idx] = t_sample;
            waveform[idx] = sign * max_amp;
            idx++;

            t_sample = t0 + delay_us + rise_us + flat_us + fall_us;
            time[idx] = t_sample;
            waveform[idx] = 0.0f;
            idx++;
        }
        else
        {
            t_sample = t0 + delay_us;
            time[idx] = t_sample;
            waveform[idx] = 0.0f;
            idx++;

            t_sample = t0 + delay_us + rise_us;
            time[idx] = t_sample;
            waveform[idx] = sign * max_amp;
            idx++;

            t_sample = t0 + delay_us + rise_us + fall_us;
            time[idx] = t_sample;
            waveform[idx] = 0.0f;
            idx++;
        }

        rel_end_us = delay_us + rise_us + flat_us + fall_us;
    }
    else
    {
        num_samples = gdef->fall_time_or_num_uncompressed_samples;
        time_shape_id = gdef->unused_or_time_shape_id;
        /* The shape THIS instance plays.  Identical to the old
         * shot_shape_ids[gte->shot_index] -- the ordinal was only ever a
         * per-definition alias for the id the grad table now carries
         * directly. */
        shape_id = gte->shape_id;

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
                idx++;
            }
            rel_end_us = delay_us + decomp_time.samples[num_samples - 1];
        }
        else
        {
            for (i = 0; i < num_samples; ++i)
            {
                t_sample = t0 + delay_us + 0.5f * grad_raster_us + (float)i * grad_raster_us;
                time[idx] = t_sample;
                waveform[idx] = sign * max_amp * decomp_wave.samples[i];
                idx++;
            }
            rel_end_us =
                delay_us + 0.5f * grad_raster_us + grad_raster_us * (float)(num_samples - 1);
        }

        if (decomp_wave.samples)
            PULSEG_FREE(decomp_wave.samples);
        if (decomp_time.samples)
            PULSEG_FREE(decomp_time.samples);
    }

    /* Tail decision must match count_grad_samples_for_block's arithmetic
     * exactly (same small, t0-independent terms). Comparing t0-embedded
     * absolute times instead loses float32 precision once t0 grows into
     * the block's tail (up to ~1e8 us on long protocols), so the two
     * functions can disagree on whether a trailing sample is needed --
     * overflowing the array count() sized for it. */
    if (rel_end_us < block_duration_us)
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

int pulseg__interpolate_to_uniform(
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
    float actual_amp[1];
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
        pos_max_gx = (float *)PULSEG_ALLOC((size_t)block_count * sizeof(float));
        pos_max_gy = (float *)PULSEG_ALLOC((size_t)block_count * sizeof(float));
        pos_max_gz = (float *)PULSEG_ALLOC((size_t)block_count * sizeof(float));
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
                    pos_max_gx[vp] = 0.0f;
                if (desc->variable_grad_flags[vp * 3 + 1])
                    pos_max_gy[vp] = 0.0f;
                if (desc->variable_grad_flags[vp * 3 + 2])
                    pos_max_gz[vp] = 0.0f;
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

        total_gx += pulseg__count_grad_samples_for_block(desc, gx_def, block_dur_us);
        total_gy += pulseg__count_grad_samples_for_block(desc, gy_def, block_dur_us);
        total_gz += pulseg__count_grad_samples_for_block(desc, gz_def, block_dur_us);
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
            idx_gx += pulseg__fill_grad_waveform_for_block(
                desc,
                time_gx,
                wf_gx,
                idx_gx,
                gx_def,
                gx_tab,
                t0,
                &pos_max_gx[n],
                block_dur_us);
            idx_gy += pulseg__fill_grad_waveform_for_block(
                desc,
                time_gy,
                wf_gy,
                idx_gy,
                gy_def,
                gy_tab,
                t0,
                &pos_max_gy[n],
                block_dur_us);
            idx_gz += pulseg__fill_grad_waveform_for_block(
                desc,
                time_gz,
                wf_gz,
                idx_gz,
                gz_def,
                gz_tab,
                t0,
                &pos_max_gz[n],
                block_dur_us);
        }
        else
        {
            actual_amp[0] = gx_tab ? gx_tab->amplitude : 0.0f;
            idx_gx += pulseg__fill_grad_waveform_for_block(
                desc,
                time_gx,
                wf_gx,
                idx_gx,
                gx_def,
                gx_tab,
                t0,
                actual_amp,
                block_dur_us);

            actual_amp[0] = gy_tab ? gy_tab->amplitude : 0.0f;
            idx_gy += pulseg__fill_grad_waveform_for_block(
                desc,
                time_gy,
                wf_gy,
                idx_gy,
                gy_def,
                gy_tab,
                t0,
                actual_amp,
                block_dur_us);

            actual_amp[0] = gz_tab ? gz_tab->amplitude : 0.0f;
            idx_gz += pulseg__fill_grad_waveform_for_block(
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

    result = pulseg__interpolate_to_uniform(&time_gx, &wf_gx, &num_gx, target_raster_us);
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
    result = pulseg__interpolate_to_uniform(&time_gy, &wf_gy, &num_gy, target_raster_us);
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
    result = pulseg__interpolate_to_uniform(&time_gz, &wf_gz, &num_gz, target_raster_us);
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
    int num_unique;
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

    /* Exactly one canonical TR.  A definition names its own worst
     * instance (pulseg_grad_representative), so shot combinations are no
     * longer enumerated and there is nothing to index past. */
    num_unique = 1;
    if (canonical_tr_idx >= num_unique)
    {
        if (diag)
        {
            pulseg_diagnostic_init(diag);
            diag->code = PULSEG_ERR_INVALID_ARGUMENT;
        }
        return PULSEG_ERR_INVALID_ARGUMENT;
    }
    rep_idx = 0;
    start_block = rep_idx * desc->tr_descriptor.tr_size;
    block_count = desc->tr_descriptor.tr_size;

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
    int main_region_start, main_region_end;
    int n, block_idx;
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
    float actual_amp[1];
    /* rotation post-pass variables */
    int interp_result, n_uniform, blk_n, rot_id, s;
    float target_raster_us, blk_end, t_sample_rot, vec[3], rot_out[3];
    const float *R;
    /* average-expansion variables */
    int *block_order;
    int pass_scan_start; /* scan-table offset for output slot 0 of this pass */
    int eff_num_averages;

    pos_max_gx = NULL;
    pos_max_gy = NULL;
    pos_max_gz = NULL;
    block_order = NULL;
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

    if (tr->tr_size <= 0)
    {
        diag->code = PULSEG_ERR_TR_NO_BLOCKS;
        return diag->code;
    }

    /* ---- determine block range ---- */
    if (amplitude_mode == PULSEG_AMP_ACTUAL)
    {
        /* Flat TR index with average expansion: TRs wrap modulo num_trs
         * so repeated averages map back to canonical block positions. */
        int num_avgs = eff_num_averages;
        int total_actual_trs = num_avgs * tr->num_trs;
        int canonical_idx;
        if (tr_index < 0 || tr_index >= total_actual_trs)
        {
            diag->code = PULSEG_ERR_INVALID_ARGUMENT;
            return diag->code;
        }
        canonical_idx = tr_index % tr->num_trs;
        tr_block_start = canonical_idx * tr->tr_size;
        block_start = tr_block_start;
        block_count = tr->tr_size;
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
        num_canonical = 0 /* one canonical TR; representatives carry the worst case */;

        if (num_canonical > 0 && tr_index >= 0 && tr_index < num_canonical && can_unique_indices)
        {
            rep_idx = can_unique_indices[tr_index];
        }

        tr_block_start = 0;
        block_start = 0;
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
                block_order[n] = rep_idx * tr->tr_size + n;
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

        pos_max_gx = (float *)PULSEG_ALLOC((size_t)block_count * sizeof(float));
        pos_max_gy = (float *)PULSEG_ALLOC((size_t)block_count * sizeof(float));
        pos_max_gz = (float *)PULSEG_ALLOC((size_t)block_count * sizeof(float));
        if (!pos_max_gx || !pos_max_gy || !pos_max_gz)
            goto alloc_fail;

        /* If tr_index > 0, filter max-pos to that canonical TR group. */
        if (tr_index >= 0)
        {
            num_canonical = 0 /* one canonical TR; representatives carry the worst case */;
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
            int vp, local_pos;
            for (vp = 0; vp < block_count; ++vp)
            {
                local_pos = vp % tr->tr_size;
                if (desc->variable_grad_flags[local_pos * 3 + 0])
                    pos_max_gx[vp] = 0.0f;
                if (desc->variable_grad_flags[local_pos * 3 + 1])
                    pos_max_gy[vp] = 0.0f;
                if (desc->variable_grad_flags[local_pos * 3 + 2])
                    pos_max_gz[vp] = 0.0f;
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

        total_gx += pulseg__count_grad_samples_for_block(desc, gx_def, block_dur_us);
        total_gy += pulseg__count_grad_samples_for_block(desc, gy_def, block_dur_us);
        total_gz += pulseg__count_grad_samples_for_block(desc, gz_def, block_dur_us);
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
            if (desc->seg_run_id && scan_pos >= 0 && scan_pos < desc->exec_stream_len)
                seg_id = pulseg__exec_seg_id(desc, scan_pos);
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
            idx_gx += pulseg__fill_grad_waveform_for_block(
                desc,
                out->gx.time_us,
                out->gx.amplitude,
                idx_gx,
                gx_def,
                gx_tab,
                t0,
                &pos_max_gx[n],
                block_dur_us);
            idx_gy += pulseg__fill_grad_waveform_for_block(
                desc,
                out->gy.time_us,
                out->gy.amplitude,
                idx_gy,
                gy_def,
                gy_tab,
                t0,
                &pos_max_gy[n],
                block_dur_us);
            idx_gz += pulseg__fill_grad_waveform_for_block(
                desc,
                out->gz.time_us,
                out->gz.amplitude,
                idx_gz,
                gz_def,
                gz_tab,
                t0,
                &pos_max_gz[n],
                block_dur_us);
        }
        else
        {
            /* PULSEG_AMP_ACTUAL: use per-instance amplitude */
            actual_amp[0] = gx_tab ? gx_tab->amplitude : 0.0f;
            idx_gx += pulseg__fill_grad_waveform_for_block(
                desc,
                out->gx.time_us,
                out->gx.amplitude,
                idx_gx,
                gx_def,
                gx_tab,
                t0,
                actual_amp,
                block_dur_us);

            actual_amp[0] = gy_tab ? gy_tab->amplitude : 0.0f;
            idx_gy += pulseg__fill_grad_waveform_for_block(
                desc,
                out->gy.time_us,
                out->gy.amplitude,
                idx_gy,
                gy_def,
                gy_tab,
                t0,
                actual_amp,
                block_dur_us);

            actual_amp[0] = gz_tab ? gz_tab->amplitude : 0.0f;
            idx_gz += pulseg__fill_grad_waveform_for_block(
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

        interp_result = pulseg__interpolate_to_uniform(
            &out->gx.time_us,
            &out->gx.amplitude,
            &num_gx,
            target_raster_us);
        if (PULSEG_FAILED(interp_result))
        {
            diag->code = interp_result;
            return interp_result;
        }
        interp_result = pulseg__interpolate_to_uniform(
            &out->gy.time_us,
            &out->gy.amplitude,
            &num_gy,
            target_raster_us);
        if (PULSEG_FAILED(interp_result))
        {
            diag->code = interp_result;
            return interp_result;
        }
        interp_result = pulseg__interpolate_to_uniform(
            &out->gz.time_us,
            &out->gz.amplitude,
            &num_gz,
            target_raster_us);
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
     * Walk through samples, find each sample's block, and turn the logical
     * vector into the physical one it plays as. */
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
        pulseg__apply_rotation(rot_out, R, vec, 0);
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
    diag->code = PULSEG_SUCCESS;
    return PULSEG_SUCCESS;

alloc_fail:
    if (block_order)
        PULSEG_FREE(block_order);
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
