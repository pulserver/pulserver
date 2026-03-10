/*
 * pulseqlib_freqmod.c -- Frequency modulation collection.
 *
 * Builds deduped amplitude-scaled 3-channel gradient modulators and
 * shift-resolved 1D plan waveforms for all subsequences.  Supports
 * PMC re-computation and binary cache for fast reload.
 *
 * Public entry points (collection-level):
 *   pulseqlib_build_freq_mod_collection    -- build all subsequences
 *   pulseqlib_update_freq_mod_collection   -- recompute one subseq
 *   pulseqlib_freq_mod_collection_get      -- look up by subseq + pos
 *   pulseqlib_freq_mod_collection_write_cache / _read_cache
 *   pulseqlib_freq_mod_collection_free
 *
 * Copyright (c) 2024, see LICENSE.txt
 */

#include "pulseqlib_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Forward declarations (used in error paths). */
static void freq_mod_library_free(pulseqlib_freq_mod_library* lib);
void pulseqlib_freq_mod_collection_free(pulseqlib_freq_mod_collection* fmc);

/* ================================================================== */
/*  Helper: generic byte-key deduplication (O(n*u))                   */
/* ================================================================== */

/*
 * Deduplicate packed byte keys.
 *
 * unique_first[u] : index of first occurrence for unique key u.
 * map[n]          : unique index for event n.
 * Returns number of unique keys.
 */
static int dedup_keys(int* unique_first, int* map,
                      const char* keys, int nkeys, int key_bytes)
{
    int n, u, nu = 0;
    for (n = 0; n < nkeys; ++n) {
        int found = -1;
        for (u = 0; u < nu; ++u) {
            if (memcmp(keys + (size_t)n * key_bytes,
                       keys + (size_t)unique_first[u] * key_bytes,
                       (size_t)key_bytes) == 0) {
                found = u;
                break;
            }
        }
        if (found >= 0) {
            map[n] = found;
        } else {
            unique_first[nu] = n;
            map[n] = nu;
            ++nu;
        }
    }
    return nu;
}

/* ================================================================== */
/*  Helper: 3x3 matrix multiply C = A^T @ B  (row-major)             */
/* ================================================================== */

static void matmul_AtB(float* C, const float* A, const float* B)
{
    int i, j;
    for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j)
            C[i * 3 + j] = A[0 * 3 + i] * B[0 * 3 + j]
                          + A[1 * 3 + i] * B[1 * 3 + j]
                          + A[2 * 3 + i] * B[2 * 3 + j];
}

/* ================================================================== */
/*  Helper: check if 3x3 matrix is identity                           */
/* ================================================================== */

static int is_identity3(const float* M)
{
    static const float I[9] = {1,0,0, 0,1,0, 0,0,1};
    int i;
    for (i = 0; i < 9; ++i)
        if (fabsf(M[i] - I[i]) > 1e-7f) return 0;
    return 1;
}

/* ================================================================== */
/*  Helper: compare two 3x3 matrices with tolerance                   */
/* ================================================================== */

static int rotmat_equal(const float* A, const float* B)
{
    int i;
    for (i = 0; i < 9; ++i)
        if (fabsf(A[i] - B[i]) > 1e-7f) return 0;
    return 1;
}

/* ================================================================== */
/*  Helper: check if waveform has any nonzero sample                  */
/* ================================================================== */

static int axis_is_active(const float* waveform, int num_samples)
{
    int i;
    for (i = 0; i < num_samples; ++i)
        if (waveform[i] != 0.0f) return 1;
    return 0;
}

/* ================================================================== */
/*  Helper: build peak-normalized grad waveform within active window  */
/* ================================================================== */

/*
 * Identical to the former static build_freq_mod_for_block() in
 * pulseqlib_structure.c.  Builds per-axis peak-normalized gradient
 * waveforms in the active event region [active_start_us, active_end_us].
 *
 * Caller must free fmod->waveform_gx/gy/gz.
 */
static int build_freq_mod_for_block(
    const pulseqlib_sequence_descriptor* desc,
    int bdef_idx,
    float active_start_us, float active_end_us,
    float ref_time_us,
    float target_raster_us,
    pulseqlib_freq_mod_definition* fmod)
{
    float grad_raster_us = desc->grad_raster_us;
    const pulseqlib_block_definition* bdef = &desc->block_definitions[bdef_idx];
    int grad_ids[3], axis;
    const pulseqlib_grad_definition* gdef;
    int shape_id, time_shape_id, num_samples, has_time_shape;
    pulseqlib_shape_arbitrary decomp_wave, decomp_time;
    float* raw_time = NULL;
    float* raw_wave = NULL;
    int raw_n, idx, i, j;
    float delay_us, rise_us, flat_us, fall_us;
    float active_dur_us;
    float* uniform_t = NULL;

    active_dur_us = active_end_us - active_start_us;
    if (active_dur_us <= 0.0f || grad_raster_us <= 0.0f)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;

    grad_ids[0] = bdef->gx_id;
    grad_ids[1] = bdef->gy_id;
    grad_ids[2] = bdef->gz_id;

    fmod->raster_us   = grad_raster_us;
    fmod->duration_us = active_dur_us;
    fmod->num_samples = (int)(active_dur_us / grad_raster_us) + 1;
    if (fmod->num_samples < 2) fmod->num_samples = 2;

    fmod->waveform_gx = (float*)PULSEQLIB_ALLOC((size_t)fmod->num_samples * sizeof(float));
    fmod->waveform_gy = (float*)PULSEQLIB_ALLOC((size_t)fmod->num_samples * sizeof(float));
    fmod->waveform_gz = (float*)PULSEQLIB_ALLOC((size_t)fmod->num_samples * sizeof(float));
    if (!fmod->waveform_gx || !fmod->waveform_gy || !fmod->waveform_gz)
        goto fmod_fail;

    raw_time  = (float*)PULSEQLIB_ALLOC((size_t)(fmod->num_samples + 512) * sizeof(float));
    raw_wave  = (float*)PULSEQLIB_ALLOC((size_t)(fmod->num_samples + 512) * sizeof(float));
    uniform_t = (float*)PULSEQLIB_ALLOC((size_t)fmod->num_samples * sizeof(float));
    if (!raw_time || !raw_wave || !uniform_t)
        goto fmod_fail;

    for (i = 0; i < fmod->num_samples; ++i)
        uniform_t[i] = (float)i * grad_raster_us;

    for (axis = 0; axis < 3; ++axis) {
        float* out_wave = (axis == 0) ? fmod->waveform_gx :
                          (axis == 1) ? fmod->waveform_gy : fmod->waveform_gz;

        if (grad_ids[axis] < 0 || grad_ids[axis] >= desc->num_unique_grads) {
            for (i = 0; i < fmod->num_samples; ++i) out_wave[i] = 0.0f;
            fmod->ref_integral[axis] = 0.0f;
            continue;
        }

        gdef = &desc->grad_definitions[grad_ids[axis]];
        decomp_wave.samples = NULL;
        decomp_time.samples = NULL;
        idx = 0;
        delay_us = (float)gdef->delay;

        if (gdef->type == 0) {
            rise_us = (float)gdef->rise_time_or_unused;
            flat_us = (float)gdef->flat_time_or_unused;
            fall_us = (float)gdef->fall_time_or_num_uncompressed_samples;

            if (flat_us > 0.0f) {
                raw_time[idx] = delay_us - active_start_us;
                raw_wave[idx] = 0.0f; idx++;
                raw_time[idx] = delay_us + rise_us - active_start_us;
                raw_wave[idx] = 1.0f; idx++;
                raw_time[idx] = delay_us + rise_us + flat_us - active_start_us;
                raw_wave[idx] = 1.0f; idx++;
                raw_time[idx] = delay_us + rise_us + flat_us + fall_us - active_start_us;
                raw_wave[idx] = 0.0f; idx++;
            } else {
                raw_time[idx] = delay_us - active_start_us;
                raw_wave[idx] = 0.0f; idx++;
                raw_time[idx] = delay_us + rise_us - active_start_us;
                raw_wave[idx] = 1.0f; idx++;
                raw_time[idx] = delay_us + rise_us + fall_us - active_start_us;
                raw_wave[idx] = 0.0f; idx++;
            }
        } else {
            num_samples   = gdef->fall_time_or_num_uncompressed_samples;
            time_shape_id = gdef->unused_or_time_shape_id;
            shape_id      = gdef->shot_shape_ids[0];

            if (shape_id <= 0 || shape_id > desc->num_shapes) {
                for (i = 0; i < fmod->num_samples; ++i) out_wave[i] = 0.0f;
                fmod->ref_integral[axis] = 0.0f;
                continue;
            }

            if (!pulseqlib__decompress_shape(&decomp_wave,
                    &desc->shapes[shape_id - 1], 1.0f)) {
                for (i = 0; i < fmod->num_samples; ++i) out_wave[i] = 0.0f;
                fmod->ref_integral[axis] = 0.0f;
                continue;
            }

            has_time_shape = 0;
            if (time_shape_id > 0 && time_shape_id <= desc->num_shapes) {
                if (pulseqlib__decompress_shape(&decomp_time,
                        &desc->shapes[time_shape_id - 1], grad_raster_us))
                    has_time_shape = 1;
            }

            if (has_time_shape) {
                for (i = 0; i < num_samples && i < decomp_wave.num_uncompressed_samples; ++i) {
                    raw_time[idx] = delay_us + decomp_time.samples[i] - active_start_us;
                    raw_wave[idx] = decomp_wave.samples[i];
                    idx++;
                }
            } else {
                for (i = 0; i < num_samples && i < decomp_wave.num_uncompressed_samples; ++i) {
                    raw_time[idx] = delay_us + 0.5f * grad_raster_us +
                                    (float)i * grad_raster_us - active_start_us;
                    raw_wave[idx] = decomp_wave.samples[i];
                    idx++;
                }
            }

            if (decomp_wave.samples) PULSEQLIB_FREE(decomp_wave.samples);
            if (decomp_time.samples) PULSEQLIB_FREE(decomp_time.samples);
        }

        raw_n = idx;

        if (raw_n > 0 && raw_time[0] > 0.0f) {
            for (j = raw_n; j > 0; --j) {
                raw_time[j] = raw_time[j - 1];
                raw_wave[j] = raw_wave[j - 1];
            }
            raw_time[0] = 0.0f;
            raw_wave[0] = 0.0f;
            raw_n++;
        } else if (raw_n == 0) {
            raw_time[0] = 0.0f;
            raw_wave[0] = 0.0f;
            raw_n = 1;
        }
        if (raw_time[raw_n - 1] < active_dur_us) {
            raw_time[raw_n] = active_dur_us;
            raw_wave[raw_n] = 0.0f;
            raw_n++;
        }

        pulseqlib__interp1_linear(out_wave, uniform_t, fmod->num_samples,
                                  raw_time, raw_wave, raw_n);

        {
            int ref_sample = (int)(ref_time_us / grad_raster_us);
            if (ref_sample < 0) ref_sample = 0;
            if (ref_sample >= fmod->num_samples) ref_sample = fmod->num_samples - 1;
            fmod->ref_integral[axis] = (float)(PULSEQLIB__TWO_PI * 1e-6) *
                pulseqlib__trapz_real_uniform(
                out_wave, ref_sample + 1, grad_raster_us);
        }
    }

    fmod->ref_time_us = ref_time_us;

    /* ZOH upsample from grad raster to target (RF/ADC) raster */
    if (target_raster_us > 0.0f && target_raster_us < grad_raster_us - 0.001f) {
        int fine_num;
        float *fine_gx, *fine_gy, *fine_gz;

        fine_num = (int)(active_dur_us / target_raster_us) + 1;
        if (fine_num < 2) fine_num = 2;

        fine_gx = (float*)PULSEQLIB_ALLOC((size_t)fine_num * sizeof(float));
        fine_gy = (float*)PULSEQLIB_ALLOC((size_t)fine_num * sizeof(float));
        fine_gz = (float*)PULSEQLIB_ALLOC((size_t)fine_num * sizeof(float));
        if (!fine_gx || !fine_gy || !fine_gz) {
            if (fine_gx) PULSEQLIB_FREE(fine_gx);
            if (fine_gy) PULSEQLIB_FREE(fine_gy);
            if (fine_gz) PULSEQLIB_FREE(fine_gz);
            goto fmod_fail;
        }

        for (i = 0; i < fine_num; ++i) {
            int orig_idx = (int)((float)i * target_raster_us / grad_raster_us);
            if (orig_idx >= fmod->num_samples) orig_idx = fmod->num_samples - 1;
            fine_gx[i] = fmod->waveform_gx[orig_idx];
            fine_gy[i] = fmod->waveform_gy[orig_idx];
            fine_gz[i] = fmod->waveform_gz[orig_idx];
        }

        PULSEQLIB_FREE(fmod->waveform_gx);
        PULSEQLIB_FREE(fmod->waveform_gy);
        PULSEQLIB_FREE(fmod->waveform_gz);
        fmod->waveform_gx = fine_gx;
        fmod->waveform_gy = fine_gy;
        fmod->waveform_gz = fine_gz;
        fmod->num_samples = fine_num;
        fmod->raster_us   = target_raster_us;

        /* Recompute ref_integral on fine grid */
        for (axis = 0; axis < 3; ++axis) {
            float* w = (axis == 0) ? fmod->waveform_gx :
                       (axis == 1) ? fmod->waveform_gy : fmod->waveform_gz;
            int ref_sample = (int)(ref_time_us / target_raster_us);
            if (ref_sample < 0) ref_sample = 0;
            if (ref_sample >= fine_num) ref_sample = fine_num - 1;
            if (ref_sample > 0)
                fmod->ref_integral[axis] = (float)(PULSEQLIB__TWO_PI * 1e-6) *
                    pulseqlib__trapz_real_uniform(w, ref_sample + 1,
                                                 target_raster_us);
            else
                fmod->ref_integral[axis] = 0.0f;
        }
    }

    PULSEQLIB_FREE(raw_time);
    PULSEQLIB_FREE(raw_wave);
    PULSEQLIB_FREE(uniform_t);
    return PULSEQLIB_SUCCESS;

fmod_fail:
    if (fmod->waveform_gx) { PULSEQLIB_FREE(fmod->waveform_gx); fmod->waveform_gx = NULL; }
    if (fmod->waveform_gy) { PULSEQLIB_FREE(fmod->waveform_gy); fmod->waveform_gy = NULL; }
    if (fmod->waveform_gz) { PULSEQLIB_FREE(fmod->waveform_gz); fmod->waveform_gz = NULL; }
    if (raw_time)  PULSEQLIB_FREE(raw_time);
    if (raw_wave)  PULSEQLIB_FREE(raw_wave);
    if (uniform_t) PULSEQLIB_FREE(uniform_t);
    return PULSEQLIB_ERR_ALLOC_FAILED;
}

/* ================================================================== */
/*  Helper: compute plan waveforms from 3ch entries + shift           */
/* ================================================================== */

/*
 * Physical gradient at position d:
 *   G_phys(t) = R * diag(A) * g(t)
 *   f(t) = G_phys(t) . d = g(t)^T * diag(A) * R^T * d
 *
 * With amplitude-scaled entry  e_j(t) = A_j * g_j(t):
 *   f(t) = e(t) . (R^T * d)
 *
 * This factoring moves amplitudes into the entry (shift-independent)
 * and keeps only the rotation + shift in the per-call computation.
 */
static void compute_plan_waveforms(
    pulseqlib_freq_mod_library* lib,
    const float* shift_m)
{
    int p, s, ch;
    float identity[9] = {1,0,0, 0,1,0, 0,0,1};

    for (p = 0; p < lib->num_plan_instances; ++p) {
        int eidx = lib->pi_entry_idx[p];
        int ridx = lib->pi_rotation_idx[p];
        int ns   = lib->plan_num_samples[p];
        float* row = lib->plan_waveforms[p];
        const float* R;
        float u[3];     /* u = R^T @ shift_m */

        R = (ridx >= 0 && ridx < lib->num_rotations)
            ? (const float*)lib->rotations[ridx] : identity;
        pulseqlib__apply_rotation(u, R, shift_m, 1);   /* transpose */

        /* freq[s] = sum_ch entry_3ch[ch][s] * u[ch] */
        for (s = 0; s < ns; ++s) {
            float val = 0.0f;
            for (ch = 0; ch < 3; ++ch)
                val += lib->entry_waveform_3ch[
                    (size_t)eidx * lib->max_samples * 3
                    + (size_t)ch * lib->max_samples + s
                ] * u[ch];
            row[s] = val;
        }
        /* zero-pad remainder */
        for (s = ns; s < lib->max_samples; ++s)
            row[s] = 0.0f;

        /* phase = sum_ch entry_ref_3ch[ch] * u[ch] */
        lib->plan_phase[p] = 0.0f;
        for (ch = 0; ch < 3; ++ch)
            lib->plan_phase[p] += lib->entry_ref_3ch[eidx * 3 + ch] * u[ch];
    }
}

/* ================================================================== */
/*  Helper: allocate plan arrays                                      */
/* ================================================================== */

static int alloc_plan(pulseqlib_freq_mod_library* lib)
{
    size_t np = (size_t)lib->num_plan_instances;
    size_t ms = (size_t)lib->max_samples;
    int r;

    lib->plan_waveform_data = (float*)PULSEQLIB_ALLOC(np * ms * sizeof(float));
    lib->plan_waveforms     = (float**)PULSEQLIB_ALLOC(np * sizeof(float*));
    lib->plan_num_samples   = (int*)PULSEQLIB_ALLOC(np * sizeof(int));
    lib->plan_phase         = (float*)PULSEQLIB_ALLOC(np * sizeof(float));

    if (!lib->plan_waveform_data || !lib->plan_waveforms ||
        !lib->plan_num_samples   || !lib->plan_phase)
        return PULSEQLIB_ERR_ALLOC_FAILED;

    for (r = 0; r < lib->num_plan_instances; ++r)
        lib->plan_waveforms[r] = lib->plan_waveform_data + (size_t)r * ms;

    return PULSEQLIB_SUCCESS;
}

/* ================================================================== */
/*  Helper: free base definitions array                               */
/* ================================================================== */

static void free_base_defs(pulseqlib_freq_mod_definition* defs, int n)
{
    int i;
    if (!defs) return;
    for (i = 0; i < n; ++i) {
        if (defs[i].waveform_gx) PULSEQLIB_FREE(defs[i].waveform_gx);
        if (defs[i].waveform_gy) PULSEQLIB_FREE(defs[i].waveform_gy);
        if (defs[i].waveform_gz) PULSEQLIB_FREE(defs[i].waveform_gz);
    }
    PULSEQLIB_FREE(defs);
}

/* ================================================================== */
/*  Dedup key types                                                   */
/* ================================================================== */

#define FREQ_MOD_BASE_COLS  5   /* rf_def_id, adc_def_id, gx, gy, gz */

typedef struct { int base_idx; float amp[3]; } entry_key_t;
typedef struct { int entry_idx; int rot_idx; } plan_key_t;

/* ================================================================== */
/*  Build                                                             */
/* ================================================================== */

static int build_freq_mod_library(
    pulseqlib_freq_mod_library** out_lib,
    const pulseqlib_collection* coll,
    int subseq_idx,
    const float* shift_m,
    const float* fov_rotation)
{
    const pulseqlib_sequence_descriptor* desc;
    pulseqlib_freq_mod_library* lib = NULL;
    int n, count, result;

    /* Per-event working arrays (sized to count) */
    int* block_indices   = NULL;   /* [count] block_table indices            */
    int (*base_rows)[FREQ_MOD_BASE_COLS] = NULL;
    int* base_unique     = NULL;
    int* base_map        = NULL;   /* event -> base_idx                      */
    int  num_base;

    entry_key_t* entry_keys  = NULL;
    int* entry_unique        = NULL;
    int* entry_map           = NULL;   /* event -> entry_idx              */
    int  num_entries;

    plan_key_t* plan_keys    = NULL;
    int* plan_unique         = NULL;
    int* plan_map            = NULL;   /* event -> plan_instance_idx      */
    int  num_plan;

    int* block_rotation      = NULL;   /* [count] rotation_idx per event  */
    int* block_norot         = NULL;   /* [count] norot flag per event    */
    pulseqlib_freq_mod_definition* base_defs = NULL;  /* [num_base] temp */
    int* base_active_mask    = NULL;   /* [num_base * 3] per-axis flag    */
    int  max_samples;

    /* ---- Validate arguments ---- */
    if (!out_lib || !coll || !shift_m)
        return PULSEQLIB_ERR_NULL_POINTER;
    *out_lib = NULL;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[subseq_idx];

    /* ---- Count blocks with freq_mod flag ---- */
    count = 0;
    for (n = 0; n < desc->num_blocks; ++n)
        if (desc->block_table[n].freq_mod_id >= 0)
            count++;

    /* ---- Allocate library shell ---- */
    lib = (pulseqlib_freq_mod_library*)PULSEQLIB_ALLOC(sizeof(*lib));
    if (!lib) return PULSEQLIB_ERR_ALLOC_FAILED;
    memset(lib, 0, sizeof(*lib));

    /* Build scan_to_plan (all -1 initially) */
    lib->scan_table_len = desc->scan_table_len;
    if (lib->scan_table_len > 0) {
        lib->scan_to_plan = (int*)PULSEQLIB_ALLOC(
            (size_t)lib->scan_table_len * sizeof(int));
        if (!lib->scan_to_plan) goto build_fail;
        for (n = 0; n < lib->scan_table_len; ++n)
            lib->scan_to_plan[n] = -1;
    }

    if (count == 0) {
        /* No freq-mod blocks: empty library */
        *out_lib = lib;
        return PULSEQLIB_SUCCESS;
    }

    /* ---- Allocate working arrays ---- */
    block_indices  = (int*)PULSEQLIB_ALLOC((size_t)count * sizeof(int));
    base_rows      = PULSEQLIB_ALLOC((size_t)count * sizeof(*base_rows));
    base_unique    = (int*)PULSEQLIB_ALLOC((size_t)count * sizeof(int));
    base_map       = (int*)PULSEQLIB_ALLOC((size_t)count * sizeof(int));
    entry_keys     = (entry_key_t*)PULSEQLIB_ALLOC((size_t)count * sizeof(entry_key_t));
    entry_unique   = (int*)PULSEQLIB_ALLOC((size_t)count * sizeof(int));
    entry_map      = (int*)PULSEQLIB_ALLOC((size_t)count * sizeof(int));
    plan_keys      = (plan_key_t*)PULSEQLIB_ALLOC((size_t)count * sizeof(plan_key_t));
    plan_unique    = (int*)PULSEQLIB_ALLOC((size_t)count * sizeof(int));
    plan_map       = (int*)PULSEQLIB_ALLOC((size_t)count * sizeof(int));
    block_rotation = (int*)PULSEQLIB_ALLOC((size_t)count * sizeof(int));
    block_norot    = (int*)PULSEQLIB_ALLOC((size_t)count * sizeof(int));

    if (!block_indices || !base_rows || !base_unique || !base_map ||
        !entry_keys || !entry_unique || !entry_map ||
        !plan_keys || !plan_unique || !plan_map || !block_rotation ||
        !block_norot)
        goto build_fail;

    /* ==== Pass 1: gather per-event info ==== */
    {
        int idx = 0;
        for (n = 0; n < desc->num_blocks; ++n) {
            const pulseqlib_block_table_element* bte = &desc->block_table[n];
            const pulseqlib_block_definition* bdef = &desc->block_definitions[bte->id];
            int has_adc, adc_def_id;

            if (bte->freq_mod_id < 0) continue;

            block_indices[idx] = n;

            /* Base dedup key */
            base_rows[idx][0] = bdef->rf_id;
            has_adc = (bte->adc_id >= 0 && bte->adc_id < desc->adc_table_size);
            adc_def_id = has_adc ? desc->adc_table[bte->adc_id].id : -1;
            base_rows[idx][1] = adc_def_id;
            base_rows[idx][2] = bdef->gx_id;
            base_rows[idx][3] = bdef->gy_id;
            base_rows[idx][4] = bdef->gz_id;

            /* Rotation: record both rotation_id and norot_flag */
            block_rotation[idx] = (bte->rotation_id >= 0)
                ? bte->rotation_id : -1;
            block_norot[idx] = bte->norot_flag;

            idx++;
        }
    }

    /* ==== Dedup base waveforms ==== */
    num_base = pulseqlib__deduplicate_int_rows(
        base_unique, base_map,
        (const int*)base_rows, count, FREQ_MOD_BASE_COLS);

    /* ==== Build base definitions and active masks ==== */
    base_defs = (pulseqlib_freq_mod_definition*)PULSEQLIB_ALLOC(
        (size_t)num_base * sizeof(pulseqlib_freq_mod_definition));
    base_active_mask = (int*)PULSEQLIB_ALLOC((size_t)num_base * 3 * sizeof(int));
    if (!base_defs || !base_active_mask) goto build_fail;
    memset(base_defs, 0, (size_t)num_base * sizeof(pulseqlib_freq_mod_definition));

    max_samples = 0;
    for (n = 0; n < num_base; ++n) {
        int row_idx = base_unique[n];
        int blk_idx = block_indices[row_idx];
        const pulseqlib_block_table_element* bte = &desc->block_table[blk_idx];
        const pulseqlib_block_definition* bdef = &desc->block_definitions[bte->id];
        int has_rf, has_adc, adc_def_id_local;
        float active_start_us, active_end_us, ref_time_us;
        float target_raster_us;

        has_rf  = (bdef->rf_id >= 0);
        has_adc = (bte->adc_id >= 0 && bte->adc_id < desc->adc_table_size);

        /* ---- Determine active region ---- */
        if (has_rf && bdef->rf_id < desc->num_unique_rfs) {
            const pulseqlib_rf_definition* rdef = &desc->rf_definitions[bdef->rf_id];
            active_start_us = (float)rdef->delay;
            if (desc->vendor == PULSEQLIB_VENDOR_GEHC) {
                active_end_us = active_start_us + rdef->stats.duration_us;
                ref_time_us   = (float)rdef->stats.isodelay_us;
            } else {
                active_end_us = (float)bdef->duration_us;
                ref_time_us   = 0.0f;
            }
        } else if (has_adc) {
            adc_def_id_local = desc->adc_table[bte->adc_id].id;
            if (adc_def_id_local >= 0 && adc_def_id_local < desc->num_unique_adcs) {
                const pulseqlib_adc_definition* adef =
                    &desc->adc_definitions[adc_def_id_local];
                float adc_dur_us = (float)adef->num_samples *
                                   (float)adef->dwell_time * 1e-3f;
                active_start_us = (float)adef->delay;
                active_end_us   = active_start_us + adc_dur_us;
                ref_time_us     = adc_dur_us * 0.5f;
            } else {
                base_active_mask[n * 3 + 0] = 0;
                base_active_mask[n * 3 + 1] = 0;
                base_active_mask[n * 3 + 2] = 0;
                continue;
            }
        } else {
            base_active_mask[n * 3 + 0] = 0;
            base_active_mask[n * 3 + 1] = 0;
            base_active_mask[n * 3 + 2] = 0;
            continue;
        }

        /* Clamp */
        if (active_start_us < 0.0f) active_start_us = 0.0f;
        if (active_end_us > (float)bdef->duration_us)
            active_end_us = (float)bdef->duration_us;
        if (ref_time_us < 0.0f) ref_time_us = 0.0f;
        if (ref_time_us > (active_end_us - active_start_us))
            ref_time_us = active_end_us - active_start_us;

        target_raster_us = (has_rf && bdef->rf_id < desc->num_unique_rfs)
                           ? desc->rf_raster_us : desc->adc_raster_us;

        result = build_freq_mod_for_block(desc, bte->id,
            active_start_us, active_end_us, ref_time_us,
            target_raster_us, &base_defs[n]);
        if (PULSEQLIB_FAILED(result)) {
            base_active_mask[n * 3 + 0] = 0;
            base_active_mask[n * 3 + 1] = 0;
            base_active_mask[n * 3 + 2] = 0;
            continue;
        }

        /* Determine which axes are active */
        base_active_mask[n * 3 + 0] = axis_is_active(
            base_defs[n].waveform_gx, base_defs[n].num_samples);
        base_active_mask[n * 3 + 1] = axis_is_active(
            base_defs[n].waveform_gy, base_defs[n].num_samples);
        base_active_mask[n * 3 + 2] = axis_is_active(
            base_defs[n].waveform_gz, base_defs[n].num_samples);

        if (base_defs[n].num_samples > max_samples)
            max_samples = base_defs[n].num_samples;
    }

    /* ==== Pass 2: build entry dedup keys ==== */
    for (n = 0; n < count; ++n) {
        int blk_idx = block_indices[n];
        const pulseqlib_block_table_element* bte = &desc->block_table[blk_idx];
        int bi = base_map[n];
        float amp[3];

        amp[0] = (bte->gx_id >= 0 && bte->gx_id < desc->grad_table_size)
                 ? desc->grad_table[bte->gx_id].amplitude : 0.0f;
        amp[1] = (bte->gy_id >= 0 && bte->gy_id < desc->grad_table_size)
                 ? desc->grad_table[bte->gy_id].amplitude : 0.0f;
        amp[2] = (bte->gz_id >= 0 && bte->gz_id < desc->grad_table_size)
                 ? desc->grad_table[bte->gz_id].amplitude : 0.0f;

        /* Mask inactive axes to zero */
        if (!base_active_mask[bi * 3 + 0]) amp[0] = 0.0f;
        if (!base_active_mask[bi * 3 + 1]) amp[1] = 0.0f;
        if (!base_active_mask[bi * 3 + 2]) amp[2] = 0.0f;

        memset(&entry_keys[n], 0, sizeof(entry_keys[n]));
        entry_keys[n].base_idx = bi;
        entry_keys[n].amp[0]   = amp[0];
        entry_keys[n].amp[1]   = amp[1];
        entry_keys[n].amp[2]   = amp[2];
    }

    /* ==== Dedup entries ==== */
    num_entries = dedup_keys(entry_unique, entry_map,
        (const char*)entry_keys, count, (int)sizeof(entry_key_t));

    /* ==== Build amplitude-scaled 3-channel entries ==== */
    if (max_samples <= 0) max_samples = 1;  /* safety */

    lib->num_entries = num_entries;
    lib->max_samples = max_samples;
    lib->raster_us   = (num_base > 0 && base_defs[0].raster_us > 0.0f)
                        ? base_defs[0].raster_us : desc->grad_raster_us;

    lib->entry_num_samples = (int*)PULSEQLIB_ALLOC(
        (size_t)num_entries * sizeof(int));
    lib->entry_waveform_3ch = (float*)PULSEQLIB_ALLOC(
        (size_t)num_entries * max_samples * 3 * sizeof(float));
    lib->entry_ref_3ch = (float*)PULSEQLIB_ALLOC(
        (size_t)num_entries * 3 * sizeof(float));
    if (!lib->entry_num_samples || !lib->entry_waveform_3ch || !lib->entry_ref_3ch)
        goto build_fail;

    memset(lib->entry_waveform_3ch, 0,
           (size_t)num_entries * max_samples * 3 * sizeof(float));

    for (n = 0; n < num_entries; ++n) {
        int ev       = entry_unique[n];
        int bi       = base_map[ev];
        const pulseqlib_freq_mod_definition* bd = &base_defs[bi];
        float amp[3];
        int s, ch;

        /* Recover effective amplitudes from the key */
        amp[0] = entry_keys[ev].amp[0];
        amp[1] = entry_keys[ev].amp[1];
        amp[2] = entry_keys[ev].amp[2];

        lib->entry_num_samples[n] = bd->num_samples;

        /* Scale base waveforms by amplitude: e_j(t) = A_j * g_j(t) */
        {
            const float* src[3];
            src[0] = bd->waveform_gx;
            src[1] = bd->waveform_gy;
            src[2] = bd->waveform_gz;
            for (ch = 0; ch < 3; ++ch) {
                float* dst = lib->entry_waveform_3ch
                    + (size_t)n * max_samples * 3
                    + (size_t)ch * max_samples;
                if (src[ch]) {
                    for (s = 0; s < bd->num_samples; ++s)
                        dst[s] = src[ch][s] * amp[ch];
                }
            }
        }

        /* Scale reference integrals */
        for (ch = 0; ch < 3; ++ch)
            lib->entry_ref_3ch[n * 3 + ch] = bd->ref_integral[ch] * amp[ch];
    }

    /* ==== Deep-copy rotations ==== */
    lib->num_rotations = desc->num_rotations;
    if (desc->num_rotations > 0 && desc->rotation_matrices) {
        lib->rotations = PULSEQLIB_ALLOC(
            (size_t)desc->num_rotations * sizeof(float[9]));
        if (!lib->rotations) goto build_fail;
        memcpy(lib->rotations, desc->rotation_matrices,
               (size_t)desc->num_rotations * sizeof(float[9]));
    }

    /* ==== Handle rotation for frequency modulation ====
     *
     * compute_plan_waveforms() computes  u = R^T @ shift_m  where R is
     * the per-plan-instance rotation.  The four relevant cases are:
     *
     *   norot=0, rot event R_ext:  R = R_ext   → u = R_ext^T @ shift
     *       The rotation event is "undone" so that the gradient rotation
     *       is applied in the prescribed FOV orientation (e.g. for
     *       consistent diffusion direction independent of prescribed FOV).
     *
     *   norot=0, no rot event:     R = I       → u = shift
     *       No rotation to undo.
     *
     * Both norot=0 cases are handled by keeping block_rotation at its
     * original value (rotation_id or -1); they are NOT modified below.
     *
     * For blocks with norot=1, the scanner does NOT apply R_prescription
     * (the FOV rotation) to the gradients — the rotation event is applied
     * in axial (physical) orientation rather than the prescribed FOV.
     * The shift vector, however, implicitly includes R_prescription.
     * We therefore compute effective rotations:
     *
     *   norot=1, rot event R_ext:  R_eff = R_presc^T @ R_ext
     *       → u = R_ext^T @ R_presc @ shift
     *
     *   norot=1, no rot event:     R_eff = R_presc^T
     *       → u = R_presc @ shift
     *
     * The effective rotation is appended to the rotation library so that
     * compute_plan_waveforms() can use it transparently.
     */
    {
        int has_norot = 0;
        for (n = 0; n < count; ++n)
            if (block_norot[n]) { has_norot = 1; break; }

        if (has_norot && fov_rotation && !is_identity3(fov_rotation)) {
            /* Non-identity FOV rotation: compute R_eff for norot blocks */
            int cap = lib->num_rotations + count;  /* upper bound */
            float (*expanded)[9] = PULSEQLIB_ALLOC((size_t)cap * sizeof(float[9]));
            int num_exp;

            if (!expanded) goto build_fail;

            /* Copy existing rotations */
            if (lib->num_rotations > 0 && lib->rotations)
                memcpy(expanded, lib->rotations,
                       (size_t)lib->num_rotations * sizeof(float[9]));
            num_exp = lib->num_rotations;

            for (n = 0; n < count; ++n) {
                float R_eff[9];
                int found, u;

                if (!block_norot[n]) continue;

                /* Compute R_eff = R_presc^T @ R_ext (or R_presc^T if no
                 * rotation event). */
                if (block_rotation[n] >= 0 &&
                    block_rotation[n] < desc->num_rotations) {
                    matmul_AtB(R_eff, fov_rotation,
                               desc->rotation_matrices[block_rotation[n]]);
                } else {
                    /* No rotation event: R_eff = R_presc^T */
                    int i, j;
                    for (i = 0; i < 3; ++i)
                        for (j = 0; j < 3; ++j)
                            R_eff[i * 3 + j] = fov_rotation[j * 3 + i];
                }

                /* Dedup: search for R_eff among existing + new entries */
                found = -1;
                for (u = 0; u < num_exp; ++u) {
                    if (rotmat_equal(R_eff, expanded[u])) {
                        found = u;
                        break;
                    }
                }
                if (found < 0) {
                    memcpy(expanded[num_exp], R_eff, sizeof(float[9]));
                    found = num_exp++;
                }
                block_rotation[n] = found;
            }

            /* Replace rotation library with expanded version */
            if (lib->rotations) PULSEQLIB_FREE(lib->rotations);
            lib->rotations = expanded;
            lib->num_rotations = num_exp;
        } else if (has_norot) {
            /* FOV rotation is identity (or NULL): for norot blocks the
             * effective rotation equals the original rotation event
             * (which may be -1 = identity).  No expansion needed, but
             * block_rotation already holds the correct value. */
        }
    }

    /* ==== Pass 3: build plan dedup keys ==== */
    for (n = 0; n < count; ++n) {
        memset(&plan_keys[n], 0, sizeof(plan_keys[n]));
        plan_keys[n].entry_idx = entry_map[n];
        plan_keys[n].rot_idx   = block_rotation[n];
    }

    num_plan = dedup_keys(plan_unique, plan_map,
        (const char*)plan_keys, count, (int)sizeof(plan_key_t));

    /* ==== Build plan instance metadata ==== */
    lib->num_plan_instances = num_plan;
    lib->pi_entry_idx    = (int*)PULSEQLIB_ALLOC((size_t)num_plan * sizeof(int));
    lib->pi_rotation_idx = (int*)PULSEQLIB_ALLOC((size_t)num_plan * sizeof(int));
    if (!lib->pi_entry_idx || !lib->pi_rotation_idx) goto build_fail;

    for (n = 0; n < num_plan; ++n) {
        int ev = plan_unique[n];
        lib->pi_entry_idx[n]    = entry_map[ev];
        lib->pi_rotation_idx[n] = block_rotation[ev];
    }

    /* ==== Allocate plan arrays ==== */
    result = alloc_plan(lib);
    if (PULSEQLIB_FAILED(result)) goto build_fail;

    /* Fill plan_num_samples from entries */
    for (n = 0; n < num_plan; ++n)
        lib->plan_num_samples[n] =
            lib->entry_num_samples[lib->pi_entry_idx[n]];

    /* ==== Build scan_to_plan ==== */
    {
        /* Temp: block_table_idx -> plan_instance */
        int* blk_to_plan = (int*)PULSEQLIB_ALLOC(
            (size_t)desc->num_blocks * sizeof(int));
        int i;
        if (!blk_to_plan) goto build_fail;

        for (i = 0; i < desc->num_blocks; ++i) blk_to_plan[i] = -1;
        for (i = 0; i < count; ++i)
            blk_to_plan[block_indices[i]] = plan_map[i];

        for (i = 0; i < desc->scan_table_len; ++i) {
            int bti = desc->scan_table_block_idx[i];
            lib->scan_to_plan[i] = (bti >= 0 && bti < desc->num_blocks)
                                   ? blk_to_plan[bti] : -1;
        }
        PULSEQLIB_FREE(blk_to_plan);
    }

    /* ==== Compute plan waveforms ==== */
    compute_plan_waveforms(lib, shift_m);

    /* ==== For non-PMC: discard 3-channel data ==== */
    if (!desc->enable_pmc) {
        PULSEQLIB_FREE(lib->entry_waveform_3ch);
        lib->entry_waveform_3ch = NULL;
        PULSEQLIB_FREE(lib->entry_ref_3ch);
        lib->entry_ref_3ch = NULL;
    }

    /* ==== Free temporary working arrays ==== */
    free_base_defs(base_defs, num_base);
    PULSEQLIB_FREE(base_active_mask);
    PULSEQLIB_FREE(block_indices);
    PULSEQLIB_FREE(base_rows);
    PULSEQLIB_FREE(base_unique);
    PULSEQLIB_FREE(base_map);
    PULSEQLIB_FREE(entry_keys);
    PULSEQLIB_FREE(entry_unique);
    PULSEQLIB_FREE(entry_map);
    PULSEQLIB_FREE(plan_keys);
    PULSEQLIB_FREE(plan_unique);
    PULSEQLIB_FREE(plan_map);
    PULSEQLIB_FREE(block_rotation);
    PULSEQLIB_FREE(block_norot);

    *out_lib = lib;
    return PULSEQLIB_SUCCESS;

build_fail:
    free_base_defs(base_defs, num_base);
    if (base_active_mask) PULSEQLIB_FREE(base_active_mask);
    if (block_indices)  PULSEQLIB_FREE(block_indices);
    if (base_rows)      PULSEQLIB_FREE(base_rows);
    if (base_unique)    PULSEQLIB_FREE(base_unique);
    if (base_map)       PULSEQLIB_FREE(base_map);
    if (entry_keys)     PULSEQLIB_FREE(entry_keys);
    if (entry_unique)   PULSEQLIB_FREE(entry_unique);
    if (entry_map)      PULSEQLIB_FREE(entry_map);
    if (plan_keys)      PULSEQLIB_FREE(plan_keys);
    if (plan_unique)    PULSEQLIB_FREE(plan_unique);
    if (plan_map)       PULSEQLIB_FREE(plan_map);
    if (block_rotation) PULSEQLIB_FREE(block_rotation);
    if (block_norot)    PULSEQLIB_FREE(block_norot);
    if (lib)            { freq_mod_library_free(lib); }
    return PULSEQLIB_ERR_ALLOC_FAILED;
}

/* ================================================================== */
/*  Update                                                            */
/* ================================================================== */

static int update_freq_mod_library(
    pulseqlib_freq_mod_library* lib,
    const float* shift_m)
{
    if (!lib || !shift_m)
        return PULSEQLIB_ERR_NULL_POINTER;
    if (!lib->entry_waveform_3ch || !lib->entry_ref_3ch)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;  /* 3ch data already freed */
    if (lib->num_plan_instances == 0)
        return PULSEQLIB_SUCCESS;

    compute_plan_waveforms(lib, shift_m);
    return PULSEQLIB_SUCCESS;
}

/* ================================================================== */
/*  Accessor                                                          */
/* ================================================================== */

static int freq_mod_library_get(
    const pulseqlib_freq_mod_library* lib,
    int scan_table_pos,
    const float** out_waveform,
    int* out_num_samples,
    float* out_phase_rad)
{
    int pi;
    if (!lib || !out_waveform || !out_num_samples || !out_phase_rad)
        return 0;
    if (scan_table_pos < 0 || scan_table_pos >= lib->scan_table_len)
        return 0;
    pi = lib->scan_to_plan[scan_table_pos];
    if (pi < 0)
        return 0;
    *out_waveform    = lib->plan_waveforms[pi];
    *out_num_samples = lib->plan_num_samples[pi];
    *out_phase_rad   = lib->plan_phase[pi];
    return 1;
}

/* ================================================================== */
/*  Cache write                                                       */
/* ================================================================== */

#define FMOD_CACHE_MAGIC   0x464D4F44   /* "FMOD" */
#define FMOD_CACHE_VERSION 1

static int freq_mod_library_write_cache(
    const pulseqlib_freq_mod_library* lib,
    FILE* f)
{
    int has_3ch;

    if (!lib || !f)
        return PULSEQLIB_ERR_NULL_POINTER;

    has_3ch = (lib->entry_waveform_3ch != NULL && lib->entry_ref_3ch != NULL);

    /* Entry metadata */
    if (fwrite(&lib->num_entries,  sizeof(int),   1, f) != 1) goto write_fail;
    if (fwrite(&lib->max_samples,  sizeof(int),   1, f) != 1) goto write_fail;
    if (fwrite(&lib->raster_us,    sizeof(float), 1, f) != 1) goto write_fail;
    if (fwrite(&has_3ch,           sizeof(int),   1, f) != 1) goto write_fail;

    if (lib->num_entries > 0) {
        if (fwrite(lib->entry_num_samples,
                   sizeof(int), (size_t)lib->num_entries, f)
            != (size_t)lib->num_entries) goto write_fail;
    }

    /* 3-channel waveforms (PMC-enabled only) */
    if (has_3ch && lib->num_entries > 0) {
        size_t total = (size_t)lib->num_entries * lib->max_samples * 3;
        if (fwrite(lib->entry_waveform_3ch, sizeof(float), total, f) != total)
            goto write_fail;

        if (fwrite(lib->entry_ref_3ch, sizeof(float),
                   (size_t)lib->num_entries * 3, f)
            != (size_t)lib->num_entries * 3) goto write_fail;
    }

    /* Rotations */
    if (fwrite(&lib->num_rotations, sizeof(int), 1, f) != 1) goto write_fail;
    if (lib->num_rotations > 0) {
        if (fwrite(lib->rotations, sizeof(float) * 9,
                   (size_t)lib->num_rotations, f)
            != (size_t)lib->num_rotations) goto write_fail;
    }

    /* Plan instance metadata */
    if (fwrite(&lib->num_plan_instances, sizeof(int), 1, f) != 1) goto write_fail;
    if (lib->num_plan_instances > 0) {
        if (fwrite(lib->pi_entry_idx, sizeof(int),
                   (size_t)lib->num_plan_instances, f)
            != (size_t)lib->num_plan_instances) goto write_fail;
        if (fwrite(lib->pi_rotation_idx, sizeof(int),
                   (size_t)lib->num_plan_instances, f)
            != (size_t)lib->num_plan_instances) goto write_fail;
        if (fwrite(lib->plan_num_samples, sizeof(int),
                   (size_t)lib->num_plan_instances, f)
            != (size_t)lib->num_plan_instances) goto write_fail;

        /* Pre-computed plan waveforms (shift-dependent) */
        {
            size_t total = (size_t)lib->num_plan_instances * lib->max_samples;
            if (fwrite(lib->plan_waveform_data, sizeof(float), total, f)
                != total) goto write_fail;
        }
        if (fwrite(lib->plan_phase, sizeof(float),
                   (size_t)lib->num_plan_instances, f)
            != (size_t)lib->num_plan_instances) goto write_fail;
    }

    /* Scan-table mapping */
    if (fwrite(&lib->scan_table_len, sizeof(int), 1, f) != 1) goto write_fail;
    if (lib->scan_table_len > 0) {
        if (fwrite(lib->scan_to_plan, sizeof(int),
                   (size_t)lib->scan_table_len, f)
            != (size_t)lib->scan_table_len) goto write_fail;
    }

    return PULSEQLIB_SUCCESS;

write_fail:
    return PULSEQLIB_ERR_FILE_READ_FAILED;
}

/* ================================================================== */
/*  Cache read                                                        */
/* ================================================================== */

static int freq_mod_library_read_cache(
    pulseqlib_freq_mod_library** out_lib,
    FILE* f,
    const float* shift_m,
    int pmc_enabled)
{
    pulseqlib_freq_mod_library* lib = NULL;
    int result, has_3ch;

    if (!out_lib || !f || !shift_m)
        return PULSEQLIB_ERR_NULL_POINTER;
    *out_lib = NULL;

    lib = (pulseqlib_freq_mod_library*)PULSEQLIB_ALLOC(sizeof(*lib));
    if (!lib) return PULSEQLIB_ERR_ALLOC_FAILED;
    memset(lib, 0, sizeof(*lib));

    /* Entry metadata */
    if (fread(&lib->num_entries,  sizeof(int),   1, f) != 1) goto read_fail;
    if (fread(&lib->max_samples,  sizeof(int),   1, f) != 1) goto read_fail;
    if (fread(&lib->raster_us,    sizeof(float), 1, f) != 1) goto read_fail;
    if (fread(&has_3ch,           sizeof(int),   1, f) != 1) goto read_fail;

    if (lib->num_entries > 0) {
        lib->entry_num_samples = (int*)PULSEQLIB_ALLOC(
            (size_t)lib->num_entries * sizeof(int));
        if (!lib->entry_num_samples) goto read_fail;
        if (fread(lib->entry_num_samples, sizeof(int),
                  (size_t)lib->num_entries, f)
            != (size_t)lib->num_entries) goto read_fail;
    }

    /* 3-channel waveforms (present only if written with has_3ch) */
    if (has_3ch && lib->num_entries > 0 && lib->max_samples > 0) {
        size_t total = (size_t)lib->num_entries * lib->max_samples * 3;
        lib->entry_waveform_3ch = (float*)PULSEQLIB_ALLOC(
            total * sizeof(float));
        if (!lib->entry_waveform_3ch) goto read_fail;
        if (fread(lib->entry_waveform_3ch, sizeof(float), total, f) != total)
            goto read_fail;

        {
            size_t reftotal = (size_t)lib->num_entries * 3;
            lib->entry_ref_3ch = (float*)PULSEQLIB_ALLOC(
                reftotal * sizeof(float));
            if (!lib->entry_ref_3ch) goto read_fail;
            if (fread(lib->entry_ref_3ch, sizeof(float), reftotal, f)
                != reftotal) goto read_fail;
        }
    }

    /* Rotations */
    if (fread(&lib->num_rotations, sizeof(int), 1, f) != 1) goto read_fail;
    if (lib->num_rotations > 0) {
        lib->rotations = PULSEQLIB_ALLOC(
            (size_t)lib->num_rotations * sizeof(float[9]));
        if (!lib->rotations) goto read_fail;
        if (fread(lib->rotations, sizeof(float) * 9,
                  (size_t)lib->num_rotations, f)
            != (size_t)lib->num_rotations) goto read_fail;
    }

    /* Plan instance metadata */
    if (fread(&lib->num_plan_instances, sizeof(int), 1, f) != 1) goto read_fail;
    if (lib->num_plan_instances > 0) {
        lib->pi_entry_idx    = (int*)PULSEQLIB_ALLOC(
            (size_t)lib->num_plan_instances * sizeof(int));
        lib->pi_rotation_idx = (int*)PULSEQLIB_ALLOC(
            (size_t)lib->num_plan_instances * sizeof(int));
        if (!lib->pi_entry_idx || !lib->pi_rotation_idx) goto read_fail;

        if (fread(lib->pi_entry_idx, sizeof(int),
                  (size_t)lib->num_plan_instances, f)
            != (size_t)lib->num_plan_instances) goto read_fail;
        if (fread(lib->pi_rotation_idx, sizeof(int),
                  (size_t)lib->num_plan_instances, f)
            != (size_t)lib->num_plan_instances) goto read_fail;

        /* Allocate plan arrays */
        result = alloc_plan(lib);
        if (PULSEQLIB_FAILED(result)) goto read_fail;
        if (fread(lib->plan_num_samples, sizeof(int),
                  (size_t)lib->num_plan_instances, f)
            != (size_t)lib->num_plan_instances) goto read_fail;

        /* Pre-computed plan waveforms */
        {
            size_t total = (size_t)lib->num_plan_instances * lib->max_samples;
            if (fread(lib->plan_waveform_data, sizeof(float), total, f)
                != total) goto read_fail;
        }
        if (fread(lib->plan_phase, sizeof(float),
                  (size_t)lib->num_plan_instances, f)
            != (size_t)lib->num_plan_instances) goto read_fail;

        /* Set up row pointers */
        {
            int n;
            for (n = 0; n < lib->num_plan_instances; ++n)
                lib->plan_waveforms[n] = lib->plan_waveform_data
                    + (size_t)n * lib->max_samples;
        }
    }

    /* Scan-table mapping */
    if (fread(&lib->scan_table_len, sizeof(int), 1, f) != 1) goto read_fail;
    if (lib->scan_table_len > 0) {
        lib->scan_to_plan = (int*)PULSEQLIB_ALLOC(
            (size_t)lib->scan_table_len * sizeof(int));
        if (!lib->scan_to_plan) goto read_fail;
        if (fread(lib->scan_to_plan, sizeof(int),
                  (size_t)lib->scan_table_len, f)
            != (size_t)lib->scan_table_len) goto read_fail;
    }

    /* If 3ch data present and PMC enabled: recompute plan for new shift.
     * Otherwise plan waveforms from cache are used as-is. */
    if (has_3ch && lib->num_plan_instances > 0 && lib->entry_waveform_3ch)
        compute_plan_waveforms(lib, shift_m);

    /* If not PMC-enabled: discard 3-channel data */
    if (!pmc_enabled) {
        if (lib->entry_waveform_3ch) {
            PULSEQLIB_FREE(lib->entry_waveform_3ch);
            lib->entry_waveform_3ch = NULL;
        }
        if (lib->entry_ref_3ch) {
            PULSEQLIB_FREE(lib->entry_ref_3ch);
            lib->entry_ref_3ch = NULL;
        }
    }

    *out_lib = lib;
    return PULSEQLIB_SUCCESS;

read_fail:
    if (lib) freq_mod_library_free(lib);
    return PULSEQLIB_ERR_FILE_READ_FAILED;
}

/* ================================================================== */
/*  Free                                                              */
/* ================================================================== */

static void freq_mod_library_free(pulseqlib_freq_mod_library* lib)
{
    if (!lib) return;

    if (lib->entry_num_samples)   PULSEQLIB_FREE(lib->entry_num_samples);
    if (lib->entry_waveform_3ch)  PULSEQLIB_FREE(lib->entry_waveform_3ch);
    if (lib->entry_ref_3ch)       PULSEQLIB_FREE(lib->entry_ref_3ch);
    if (lib->rotations)           PULSEQLIB_FREE(lib->rotations);

    if (lib->pi_entry_idx)        PULSEQLIB_FREE(lib->pi_entry_idx);
    if (lib->pi_rotation_idx)     PULSEQLIB_FREE(lib->pi_rotation_idx);
    if (lib->plan_waveform_data)  PULSEQLIB_FREE(lib->plan_waveform_data);
    if (lib->plan_waveforms)      PULSEQLIB_FREE(lib->plan_waveforms);
    if (lib->plan_num_samples)    PULSEQLIB_FREE(lib->plan_num_samples);
    if (lib->plan_phase)          PULSEQLIB_FREE(lib->plan_phase);

    if (lib->scan_to_plan)        PULSEQLIB_FREE(lib->scan_to_plan);

    PULSEQLIB_FREE(lib);
}

/* ================================================================== */
/*  Public: collection build                                          */
/* ================================================================== */

int pulseqlib_build_freq_mod_collection(
    pulseqlib_freq_mod_collection** out_fmc,
    const pulseqlib_collection* coll,
    const float* shift_m,
    const float* fov_rotation)
{
    pulseqlib_freq_mod_collection* fmc = NULL;
    int s, nsub;

    if (!out_fmc || !coll || !shift_m)
        return PULSEQLIB_ERR_NULL_POINTER;
    *out_fmc = NULL;

    nsub = coll->num_subsequences;

    fmc = (pulseqlib_freq_mod_collection*)PULSEQLIB_ALLOC(sizeof(*fmc));
    if (!fmc) return PULSEQLIB_ERR_ALLOC_FAILED;
    memset(fmc, 0, sizeof(*fmc));

    fmc->num_subsequences = nsub;
    fmc->libs = (pulseqlib_freq_mod_library**)PULSEQLIB_ALLOC(
        (size_t)nsub * sizeof(*fmc->libs));
    if (!fmc->libs) {
        PULSEQLIB_FREE(fmc);
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }
    memset(fmc->libs, 0, (size_t)nsub * sizeof(*fmc->libs));

    for (s = 0; s < nsub; ++s) {
        int rc = build_freq_mod_library(&fmc->libs[s], coll, s, shift_m,
                                        fov_rotation);
        if (PULSEQLIB_FAILED(rc)) {
            pulseqlib_freq_mod_collection_free(fmc);
            return rc;
        }
    }

    *out_fmc = fmc;
    return PULSEQLIB_SUCCESS;
}

/* ================================================================== */
/*  Public: collection update (one subsequence)                       */
/* ================================================================== */

int pulseqlib_update_freq_mod_collection(
    pulseqlib_freq_mod_collection* fmc,
    int subseq_idx,
    const float* shift_m)
{
    if (!fmc || !shift_m)
        return PULSEQLIB_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= fmc->num_subsequences)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;
    return update_freq_mod_library(fmc->libs[subseq_idx], shift_m);
}

/* ================================================================== */
/*  Public: collection accessor                                       */
/* ================================================================== */

int pulseqlib_freq_mod_collection_get(
    const pulseqlib_freq_mod_collection* fmc,
    int subseq_idx,
    int scan_table_pos,
    const float** out_waveform,
    int* out_num_samples,
    float* out_phase_rad)
{
    if (!fmc || subseq_idx < 0 || subseq_idx >= fmc->num_subsequences)
        return 0;
    return freq_mod_library_get(fmc->libs[subseq_idx], scan_table_pos,
                                out_waveform, out_num_samples, out_phase_rad);
}

/* ================================================================== */
/*  Public: collection cache write (single file)                      */
/* ================================================================== */

#define FMCOL_CACHE_MAGIC   0x464D434F  /* "FMCO" */
#define FMCOL_CACHE_VERSION 1

int pulseqlib_freq_mod_collection_write_cache(
    const pulseqlib_freq_mod_collection* fmc,
    const char* path)
{
    FILE* f;
    int magic, version, s;

    if (!fmc || !path)
        return PULSEQLIB_ERR_NULL_POINTER;

    f = fopen(path, "wb");
    if (!f) return PULSEQLIB_ERR_FILE_READ_FAILED;

    magic   = FMCOL_CACHE_MAGIC;
    version = FMCOL_CACHE_VERSION;

    if (fwrite(&magic,   sizeof(int), 1, f) != 1) goto col_write_fail;
    if (fwrite(&version, sizeof(int), 1, f) != 1) goto col_write_fail;
    if (fwrite(&fmc->num_subsequences, sizeof(int), 1, f) != 1)
        goto col_write_fail;

    for (s = 0; s < fmc->num_subsequences; ++s) {
        int rc = freq_mod_library_write_cache(fmc->libs[s], f);
        if (PULSEQLIB_FAILED(rc)) goto col_write_fail;
    }

    fclose(f);
    return PULSEQLIB_SUCCESS;

col_write_fail:
    fclose(f);
    return PULSEQLIB_ERR_FILE_READ_FAILED;
}

/* ================================================================== */
/*  Public: collection cache read (single file)                       */
/* ================================================================== */

int pulseqlib_freq_mod_collection_read_cache(
    pulseqlib_freq_mod_collection** out_fmc,
    const char* path,
    const pulseqlib_collection* coll,
    const float* shift_m)
{
    FILE* f;
    pulseqlib_freq_mod_collection* fmc = NULL;
    int magic, version, nsub, s;

    if (!out_fmc || !path || !coll || !shift_m)
        return PULSEQLIB_ERR_NULL_POINTER;
    *out_fmc = NULL;

    f = fopen(path, "rb");
    if (!f) return PULSEQLIB_ERR_FILE_READ_FAILED;

    if (fread(&magic, sizeof(int), 1, f) != 1 || magic != FMCOL_CACHE_MAGIC)
        { fclose(f); return PULSEQLIB_ERR_FILE_READ_FAILED; }
    if (fread(&version, sizeof(int), 1, f) != 1 || version != FMCOL_CACHE_VERSION)
        { fclose(f); return PULSEQLIB_ERR_FILE_READ_FAILED; }
    if (fread(&nsub, sizeof(int), 1, f) != 1 || nsub != coll->num_subsequences)
        { fclose(f); return PULSEQLIB_ERR_FILE_READ_FAILED; }

    fmc = (pulseqlib_freq_mod_collection*)PULSEQLIB_ALLOC(sizeof(*fmc));
    if (!fmc) { fclose(f); return PULSEQLIB_ERR_ALLOC_FAILED; }
    memset(fmc, 0, sizeof(*fmc));

    fmc->num_subsequences = nsub;
    fmc->libs = (pulseqlib_freq_mod_library**)PULSEQLIB_ALLOC(
        (size_t)nsub * sizeof(*fmc->libs));
    if (!fmc->libs) {
        PULSEQLIB_FREE(fmc);
        fclose(f);
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }
    memset(fmc->libs, 0, (size_t)nsub * sizeof(*fmc->libs));

    for (s = 0; s < nsub; ++s) {
        int pmc = coll->descriptors[s].enable_pmc;
        int rc = freq_mod_library_read_cache(&fmc->libs[s], f, shift_m, pmc);
        if (PULSEQLIB_FAILED(rc)) {
            fclose(f);
            pulseqlib_freq_mod_collection_free(fmc);
            return rc;
        }
    }

    fclose(f);
    *out_fmc = fmc;
    return PULSEQLIB_SUCCESS;
}

/* ================================================================== */
/*  Public: collection free                                           */
/* ================================================================== */

void pulseqlib_freq_mod_collection_free(pulseqlib_freq_mod_collection* fmc)
{
    if (!fmc) return;
    if (fmc->libs) {
        int s;
        for (s = 0; s < fmc->num_subsequences; ++s)
            if (fmc->libs[s]) freq_mod_library_free(fmc->libs[s]);
        PULSEQLIB_FREE(fmc->libs);
    }
    PULSEQLIB_FREE(fmc);
}
