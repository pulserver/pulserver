/**
 * @file pulseg_freqmod.c
 * @brief Frequency-modulation waveforms for FOV shift and motion correction.
 *
 * Two layers:
 *   - the shift-INDEPENDENT base: deduped, amplitude-scaled 3-channel
 *     gradient modulators, built once and persisted in the cache;
 *   - the shift-DEPENDENT plans: 1-D hardware waveforms and phase offsets
 *     resolved against a particular spatial shift, rebuilt whenever the
 *     shift changes (per subsequence, or per TR under PMC).
 *
 * Only PMC-enabled subsequences retain the 3-channel data after build; for
 * everything else it is freed once the plans exist, since the shift is fixed
 * for the scan.
 */

#include "pulseg_internal.h"
#include "pulseg.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* OMEGA DAC conversion: Hz → short DAC units.
 * PULSEG_HW_FREQ_CONVERSION and PULSEG_HW_WAVEFORM_END are defined
 * in pulseg_config.h (vendor-overridable).  The OMEGA board uses
 * 4× oversampled DAC:
 *   dac_value = freq_hz / (4 * PULSEG_HW_FREQ_CONVERSION)
 * PULSEG_HW_WAVEFORM_END marks the last waveform sample. */
#define PULSEG_OMEGA_SCALE (1.0f / (4.0f * PULSEG_HW_FREQ_CONVERSION))

/* Forward declarations (used in error paths). */
static void freq_mod_library_free(pulseg_freq_mod_library *lib);
static void convert_plan_to_hw(pulseg_freq_mod_library *lib);
void pulseg_freq_mod_collection_free(pulseg_freq_mod_collection *fmc);

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
static int dedup_keys(int *unique_first, int *map, const char *keys, int nkeys, int key_bytes)
{
    int n, u, nu = 0;
    for (n = 0; n < nkeys; ++n)
    {
        int found = -1;
        for (u = 0; u < nu; ++u)
        {
            if (memcmp(
                    keys + (size_t)n * key_bytes,
                    keys + (size_t)unique_first[u] * key_bytes,
                    (size_t)key_bytes) == 0)
            {
                found = u;
                break;
            }
        }
        if (found >= 0)
        {
            map[n] = found;
        }
        else
        {
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

static void matmul_AtB(float *C, const float *A, const float *B)
{
    int i, j;
    for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j)
            C[i * 3 + j] = A[0 * 3 + i] * B[0 * 3 + j] + A[1 * 3 + i] * B[1 * 3 + j] +
                A[2 * 3 + i] * B[2 * 3 + j];
}

/* ================================================================== */
/*  Helper: compare two 3x3 matrices with tolerance                   */
/* ================================================================== */

static int rotmat_equal(const float *A, const float *B)
{
    int i;
    for (i = 0; i < 9; ++i)
        if ((float)fabs(A[i] - B[i]) > 1e-7f)
            return 0;
    return 1;
}

/* ================================================================== */
/*  Helper: check if waveform has any nonzero sample                  */
/* ================================================================== */

static int axis_is_active(const float *waveform, int num_samples)
{
    int i;
    for (i = 0; i < num_samples; ++i)
        if ((float)fabs(waveform[i]) > 1e-8f)
            return 1;
    return 0;
}

/*
 * Identical to the former static build_freq_mod_for_block() in
 * pulseg_structure.c.  Builds per-axis peak-normalized gradient
 * waveforms in the active event region [active_start_us, active_end_us].
 *
 * Caller must free fmod->waveform_gx/gy/gz.
 */
static int build_freq_mod_for_block(
    const pulseg_sequence_descriptor *desc,
    pulseg_freq_mod_definition *fmod,
    const pulseg_block_table_element *bte,
    int bdef_idx,
    float active_start_us,
    float active_end_us,
    float ref_time_us,
    float target_raster_us)
{
    float grad_raster_us = desc->grad_raster_us;
    const pulseg_base_block *bdef = &desc->base_blocks[bdef_idx];
    int grad_ids[3], axis;
    const pulseg_grad_definition *gdef;
    int shape_id, time_shape_id, num_samples, has_time_shape;
    pulseq_shape decomp_wave, decomp_time;
    float *raw_time = NULL;
    float *raw_wave = NULL;
    int raw_n, idx, i, j;
    float delay_us, rise_us, flat_us, fall_us;
    float active_dur_us;
    float *uniform_t = NULL;

    active_dur_us = active_end_us - active_start_us;
    if (active_dur_us <= 0.0f || grad_raster_us <= 0.0f)
        return PULSEG_ERR_INVALID_ARGUMENT;

    grad_ids[0] = bdef->gx_id;
    grad_ids[1] = bdef->gy_id;
    grad_ids[2] = bdef->gz_id;

    fmod->raster_us = grad_raster_us;
    fmod->duration_us = active_dur_us;
    fmod->num_samples = (int)(active_dur_us / grad_raster_us) + 1;
    if (fmod->num_samples < 2)
        fmod->num_samples = 2;

    fmod->waveform_gx = (float *)PULSEG_ALLOC((size_t)fmod->num_samples * sizeof(float));
    fmod->waveform_gy = (float *)PULSEG_ALLOC((size_t)fmod->num_samples * sizeof(float));
    fmod->waveform_gz = (float *)PULSEG_ALLOC((size_t)fmod->num_samples * sizeof(float));
    if (!fmod->waveform_gx || !fmod->waveform_gy || !fmod->waveform_gz)
        goto fmod_fail;

    raw_time = (float *)PULSEG_ALLOC((size_t)(fmod->num_samples + 512) * sizeof(float));
    raw_wave = (float *)PULSEG_ALLOC((size_t)(fmod->num_samples + 512) * sizeof(float));
    uniform_t = (float *)PULSEG_ALLOC((size_t)fmod->num_samples * sizeof(float));
    if (!raw_time || !raw_wave || !uniform_t)
        goto fmod_fail;

    for (i = 0; i < fmod->num_samples; ++i)
        uniform_t[i] = (float)i * grad_raster_us;

    for (axis = 0; axis < 3; ++axis)
    {
        float *out_wave = (axis == 0) ? fmod->waveform_gx
            : (axis == 1)             ? fmod->waveform_gy
                                      : fmod->waveform_gz;

        if (grad_ids[axis] < 0 || grad_ids[axis] >= desc->num_unique_grads)
        {
            for (i = 0; i < fmod->num_samples; ++i)
                out_wave[i] = 0.0f;
            fmod->ref_integral[axis] = 0.0f;
            continue;
        }

        gdef = &desc->grad_definitions[grad_ids[axis]];
        decomp_wave.samples = NULL;
        decomp_time.samples = NULL;
        idx = 0;
        delay_us = (float)gdef->delay;

        if (gdef->type == 0)
        {
            rise_us = (float)gdef->rise_time_or_unused;
            flat_us = (float)gdef->flat_time_or_unused;
            fall_us = (float)gdef->fall_time_or_num_uncompressed_samples;

            if (flat_us > 0.0f)
            {
                raw_time[idx] = delay_us - active_start_us;
                raw_wave[idx] = 0.0f;
                idx++;
                raw_time[idx] = delay_us + rise_us - active_start_us;
                raw_wave[idx] = 1.0f;
                idx++;
                raw_time[idx] = delay_us + rise_us + flat_us - active_start_us;
                raw_wave[idx] = 1.0f;
                idx++;
                raw_time[idx] = delay_us + rise_us + flat_us + fall_us - active_start_us;
                raw_wave[idx] = 0.0f;
                idx++;
            }
            else
            {
                raw_time[idx] = delay_us - active_start_us;
                raw_wave[idx] = 0.0f;
                idx++;
                raw_time[idx] = delay_us + rise_us - active_start_us;
                raw_wave[idx] = 1.0f;
                idx++;
                raw_time[idx] = delay_us + rise_us + fall_us - active_start_us;
                raw_wave[idx] = 0.0f;
                idx++;
            }
        }
        else
        {
            int grad_event_id;
            int shot_idx;

            grad_event_id = (axis == 0) ? bte->gx_id : (axis == 1) ? bte->gy_id : bte->gz_id;
            shot_idx = (grad_event_id >= 0 && grad_event_id < desc->grad_table_size)
                ? desc->grad_table[grad_event_id].shot_index
                : 0;
            if (shot_idx < 0 || shot_idx >= PULSEG_MAX_GRAD_SHOTS)
                shot_idx = 0;

            time_shape_id = gdef->unused_or_time_shape_id;
            shape_id = gdef->shot_shape_ids[shot_idx];

            if (shape_id <= 0 || shape_id > desc->num_shapes)
            {
                for (i = 0; i < fmod->num_samples; ++i)
                    out_wave[i] = 0.0f;
                fmod->ref_integral[axis] = 0.0f;
                continue;
            }

            if (!pulseq_decompress_shape(&decomp_wave, &desc->shapes[shape_id - 1], 1.0f))
            {
                for (i = 0; i < fmod->num_samples; ++i)
                    out_wave[i] = 0.0f;
                fmod->ref_integral[axis] = 0.0f;
                continue;
            }

            num_samples = decomp_wave.num_uncompressed_samples;

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
                for (i = 0; i < num_samples && i < decomp_wave.num_uncompressed_samples; ++i)
                {
                    raw_time[idx] = delay_us + decomp_time.samples[i] - active_start_us;
                    raw_wave[idx] = decomp_wave.samples[i];
                    idx++;
                }
            }
            else
            {
                for (i = 0; i < num_samples && i < decomp_wave.num_uncompressed_samples; ++i)
                {
                    raw_time[idx] = delay_us + 0.5f * grad_raster_us + (float)i * grad_raster_us -
                        active_start_us;
                    raw_wave[idx] = decomp_wave.samples[i];
                    idx++;
                }
            }

            if (decomp_wave.samples)
                PULSEG_FREE(decomp_wave.samples);
            if (decomp_time.samples)
                PULSEG_FREE(decomp_time.samples);
        }

        raw_n = idx;

        if (raw_n > 0 && raw_time[0] > 0.0f)
        {
            for (j = raw_n; j > 0; --j)
            {
                raw_time[j] = raw_time[j - 1];
                raw_wave[j] = raw_wave[j - 1];
            }
            raw_time[0] = 0.0f;
            raw_wave[0] = 0.0f;
            raw_n++;
        }
        else if (raw_n == 0)
        {
            raw_time[0] = 0.0f;
            raw_wave[0] = 0.0f;
            raw_n = 1;
        }
        if (raw_time[raw_n - 1] < active_dur_us)
        {
            raw_time[raw_n] = active_dur_us;
            raw_wave[raw_n] = 0.0f;
            raw_n++;
        }

        pulseg__interp1_linear(out_wave, uniform_t, fmod->num_samples, raw_time, raw_wave, raw_n);

        {
            int ref_sample = (int)(ref_time_us / grad_raster_us);
            if (ref_sample < 0)
                ref_sample = 0;
            if (ref_sample >= fmod->num_samples)
                ref_sample = fmod->num_samples - 1;
            fmod->ref_integral[axis] = (float)(PULSEG__TWO_PI * 1e-6) *
                pulseg__trapz_real_uniform(out_wave, ref_sample + 1, grad_raster_us);
        }
    }

    fmod->ref_time_us = ref_time_us;

    /* ZOH upsample from grad raster to target (RF/ADC) raster */
    if (target_raster_us > 0.0f && target_raster_us < grad_raster_us - 0.001f)
    {
        int fine_num;
        float *fine_gx, *fine_gy, *fine_gz;

        fine_num = (int)(active_dur_us / target_raster_us) + 1;
        if (fine_num < 2)
            fine_num = 2;

        fine_gx = (float *)PULSEG_ALLOC((size_t)fine_num * sizeof(float));
        fine_gy = (float *)PULSEG_ALLOC((size_t)fine_num * sizeof(float));
        fine_gz = (float *)PULSEG_ALLOC((size_t)fine_num * sizeof(float));
        if (!fine_gx || !fine_gy || !fine_gz)
        {
            if (fine_gx)
                PULSEG_FREE(fine_gx);
            if (fine_gy)
                PULSEG_FREE(fine_gy);
            if (fine_gz)
                PULSEG_FREE(fine_gz);
            goto fmod_fail;
        }

        for (i = 0; i < fine_num; ++i)
        {
            int orig_idx = (int)((float)i * target_raster_us / grad_raster_us);
            if (orig_idx >= fmod->num_samples)
                orig_idx = fmod->num_samples - 1;
            fine_gx[i] = fmod->waveform_gx[orig_idx];
            fine_gy[i] = fmod->waveform_gy[orig_idx];
            fine_gz[i] = fmod->waveform_gz[orig_idx];
        }

        PULSEG_FREE(fmod->waveform_gx);
        PULSEG_FREE(fmod->waveform_gy);
        PULSEG_FREE(fmod->waveform_gz);
        fmod->waveform_gx = fine_gx;
        fmod->waveform_gy = fine_gy;
        fmod->waveform_gz = fine_gz;
        fmod->num_samples = fine_num;
        fmod->raster_us = target_raster_us;

        /* Recompute ref_integral on fine grid */
        for (axis = 0; axis < 3; ++axis)
        {
            float *w = (axis == 0) ? fmod->waveform_gx
                : (axis == 1)      ? fmod->waveform_gy
                                   : fmod->waveform_gz;
            int ref_sample = (int)(ref_time_us / target_raster_us);
            if (ref_sample < 0)
                ref_sample = 0;
            if (ref_sample >= fine_num)
                ref_sample = fine_num - 1;
            if (ref_sample > 0)
                fmod->ref_integral[axis] = (float)(PULSEG__TWO_PI * 1e-6) *
                    pulseg__trapz_real_uniform(w, ref_sample + 1, target_raster_us);
            else
                fmod->ref_integral[axis] = 0.0f;
        }
    }

    PULSEG_FREE(raw_time);
    PULSEG_FREE(raw_wave);
    PULSEG_FREE(uniform_t);
    return PULSEG_SUCCESS;

fmod_fail:
    if (fmod->waveform_gx)
    {
        PULSEG_FREE(fmod->waveform_gx);
        fmod->waveform_gx = NULL;
    }
    if (fmod->waveform_gy)
    {
        PULSEG_FREE(fmod->waveform_gy);
        fmod->waveform_gy = NULL;
    }
    if (fmod->waveform_gz)
    {
        PULSEG_FREE(fmod->waveform_gz);
        fmod->waveform_gz = NULL;
    }
    if (raw_time)
        PULSEG_FREE(raw_time);
    if (raw_wave)
        PULSEG_FREE(raw_wave);
    if (uniform_t)
        PULSEG_FREE(uniform_t);
    return PULSEG_ERR_ALLOC_FAILED;
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
static void compute_plan_waveforms(pulseg_freq_mod_library *lib, const float *shift_m)
{
    int p, s, ch;
    float identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    for (p = 0; p < lib->num_plan_instances; ++p)
    {
        int eidx = lib->pi_entry_idx[p];
        int ridx = lib->pi_rotation_idx[p];
        int ns = lib->plan_num_samples[p];
        float *row = lib->plan_waveforms[p];
        const float *R;
        float u[3]; /* u = R^T @ shift_m */

        R = (ridx >= 0 && ridx < lib->num_rotations) ? (const float *)lib->rotations[ridx]
                                                     : identity;
        pulseg__apply_rotation(u, R, shift_m, 1); /* transpose */

        /* freq[s] = sum_ch entry_3ch[ch][s] * u[ch] */
        for (s = 0; s < ns; ++s)
        {
            float val = 0.0f;
            for (ch = 0; ch < 3; ++ch)
                val +=
                    lib->entry_waveform_3ch
                        [(size_t)eidx * lib->max_samples * 3 + (size_t)ch * lib->max_samples + s] *
                    u[ch];
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

    /* Convert float Hz waveforms → short DAC units for hardware DMA */
    convert_plan_to_hw(lib);
}

/* Convert plan_waveforms (float Hz) → plan_hw_waveforms (short DAC).
 * Sets WEOS_BIT on the last sample of each plan instance. */
static void convert_plan_to_hw(pulseg_freq_mod_library *lib)
{
    int p, s;
    for (p = 0; p < lib->num_plan_instances; ++p)
    {
        int ns = lib->plan_num_samples[p];
        const float *src = lib->plan_waveforms[p];
        short *dst = lib->plan_hw_waveforms[p];
        for (s = 0; s < ns; ++s)
        {
            int v = (int)(src[s] * PULSEG_OMEGA_SCALE);
            if (v > 32767)
                v = 32767;
            if (v < -32767)
                v = -32767;
            dst[s] = (short)(v & ~PULSEG_HW_WAVEFORM_END);
        }
        if (ns > 0)
            dst[ns - 1] |= PULSEG_HW_WAVEFORM_END;
        /* zero-pad remainder */
        for (s = ns; s < lib->max_samples; ++s)
            dst[s] = 0;
    }
}

/* Recompute per-scan extra phase from inactive-axis areas. */
static void compute_scan_phase_extra(pulseg_freq_mod_library *lib, const float *shift_m)
{
    int pos;
    float identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    if (!lib || !shift_m || !lib->scan_phase_extra || !lib->scan_inactive_area_3ch)
        return;

    for (pos = 0; pos < lib->exec_stream_len; ++pos)
    {
        int pi = lib->scan_to_plan[pos];
        int ridx = -1;
        const float *R;
        float u[3];
        const float *ia = lib->scan_inactive_area_3ch + (size_t)pos * 3;

        if (pi >= 0 && pi < lib->num_plan_instances)
            ridx = lib->pi_rotation_idx[pi];

        R = (ridx >= 0 && ridx < lib->num_rotations) ? (const float *)lib->rotations[ridx]
                                                     : identity;
        pulseg__apply_rotation(u, R, shift_m, 1); /* R^T @ shift */

        lib->scan_phase_extra[pos] = ia[0] * u[0] + ia[1] * u[1] + ia[2] * u[2];
    }
}

/* ================================================================== */
/*  Helper: allocate plan arrays                                      */
/* ================================================================== */

static int alloc_plan(pulseg_freq_mod_library *lib)
{
    size_t np = (size_t)lib->num_plan_instances;
    size_t ms = (size_t)lib->max_samples;
    int r;

    lib->plan_waveform_data = (float *)PULSEG_ALLOC(np * ms * sizeof(float));
    lib->plan_waveforms = (float **)PULSEG_ALLOC(np * sizeof(float *));
    lib->plan_hw_data = (short *)PULSEG_ALLOC(np * ms * sizeof(short));
    lib->plan_hw_waveforms = (short **)PULSEG_ALLOC(np * sizeof(short *));
    lib->plan_num_samples = (int *)PULSEG_ALLOC(np * sizeof(int));
    lib->plan_phase = (float *)PULSEG_ALLOC(np * sizeof(float));

    if (!lib->plan_waveform_data || !lib->plan_waveforms || !lib->plan_hw_data ||
        !lib->plan_hw_waveforms || !lib->plan_num_samples || !lib->plan_phase)
        return PULSEG_ERR_ALLOC_FAILED;

    for (r = 0; r < lib->num_plan_instances; ++r)
    {
        lib->plan_waveforms[r] = lib->plan_waveform_data + (size_t)r * ms;
        lib->plan_hw_waveforms[r] = lib->plan_hw_data + (size_t)r * ms;
    }

    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Helper: free base definitions array                               */
/* ================================================================== */

static void free_base_defs(pulseg_freq_mod_definition *defs, int n)
{
    int i;
    if (!defs)
        return;
    for (i = 0; i < n; ++i)
    {
        if (defs[i].waveform_gx)
            PULSEG_FREE(defs[i].waveform_gx);
        if (defs[i].waveform_gy)
            PULSEG_FREE(defs[i].waveform_gy);
        if (defs[i].waveform_gz)
            PULSEG_FREE(defs[i].waveform_gz);
    }
    PULSEG_FREE(defs);
}

/* ================================================================== */
/*  Dedup key types                                                   */
/* ================================================================== */

#define FREQ_MOD_BASE_COLS \
    7 /* freq_mod_id, active_start_us, active_dur_us, gx_def_shot, gy_def_shot, gz_def_shot, effective_duration_us */

typedef struct
{
    int base_idx;
    float amp[3];
} entry_key_t;
typedef struct
{
    int entry_idx;
    int rot_idx;
    int tr_scope_id;
    float inactive_ref_3ch[3];
} plan_key_t;

/* ================================================================== */
/*  Build                                                             */
/* ================================================================== */

static int build_freq_mod_library(
    pulseg_freq_mod_library **out_lib,
    const pulseg_collection *coll,
    int subseq_idx,
    const float *shift_m,
    const float *fov_rotation)
{
    const pulseg_sequence_descriptor *desc;
    pulseg_freq_mod_library *lib = NULL;
    int n, count, result;
    int fail_code =
        PULSEG_ERR_ALLOC_FAILED; /* build_fail return; overridden for specific failures */

    /* Per-event working arrays (sized to count) */
    int *block_indices = NULL; /* [count] block_table indices            */
    int(*base_rows)[FREQ_MOD_BASE_COLS] = NULL;
    int *base_unique = NULL;
    int *base_map = NULL; /* event -> base_idx                      */
    int num_base = 0;

    entry_key_t *entry_keys = NULL;
    int *entry_unique = NULL;
    int *entry_map = NULL; /* event -> entry_idx              */
    int num_entries;

    plan_key_t *plan_keys = NULL;
    int *plan_unique = NULL;
    int *plan_map = NULL; /* event -> plan_instance_idx      */
    int num_plan;

    int *block_rotation = NULL;                   /* [count] rotation_idx per event  */
    int *block_norot = NULL;                      /* [count] norot flag per event    */
    int *base_active_mask = NULL;                 /* [num_base * 3], allocated after dedup */
    pulseg_freq_mod_definition *base_defs = NULL; /* [num_base] temp */
    int max_samples;

    /* ---- Validate arguments ---- */
    if (!out_lib || !coll || !shift_m)
        return PULSEG_ERR_NULL_POINTER;
    *out_lib = NULL;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[subseq_idx];

    /* ---- Count blocks with freq_mod flag ---- */
    count = 0;
    for (n = 0; n < desc->num_blocks; ++n)
        if (desc->block_table[n].freq_mod_id >= 0)
            count++;

    /* ---- Allocate library shell ---- */
    lib = (pulseg_freq_mod_library *)PULSEG_ALLOC(sizeof(*lib));
    if (!lib)
        return PULSEG_ERR_ALLOC_FAILED;
    memset(lib, 0, sizeof(*lib));

    /* Build scan_to_plan (all -1 initially) */
    lib->exec_stream_len = desc->exec_stream_len;
    if (lib->exec_stream_len > 0)
    {
        lib->scan_to_plan = (int *)PULSEG_ALLOC((size_t)lib->exec_stream_len * sizeof(int));
        if (!lib->scan_to_plan)
            goto build_fail;
        for (n = 0; n < lib->exec_stream_len; ++n)
            lib->scan_to_plan[n] = -1;
    }

    if (count == 0)
    {
        /* No freq-mod blocks: empty library */
        *out_lib = lib;
        return PULSEG_SUCCESS;
    }

    /* ---- Allocate working arrays ---- */
    block_indices = (int *)PULSEG_ALLOC((size_t)count * sizeof(int));
    base_rows = PULSEG_ALLOC((size_t)count * sizeof(*base_rows));
    base_unique = (int *)PULSEG_ALLOC((size_t)count * sizeof(int));
    base_map = (int *)PULSEG_ALLOC((size_t)count * sizeof(int));
    entry_keys = (entry_key_t *)PULSEG_ALLOC((size_t)count * sizeof(entry_key_t));
    entry_unique = (int *)PULSEG_ALLOC((size_t)count * sizeof(int));
    entry_map = (int *)PULSEG_ALLOC((size_t)count * sizeof(int));
    plan_keys = (plan_key_t *)PULSEG_ALLOC((size_t)count * sizeof(plan_key_t));
    plan_unique = (int *)PULSEG_ALLOC((size_t)count * sizeof(int));
    plan_map = (int *)PULSEG_ALLOC((size_t)count * sizeof(int));
    block_rotation = (int *)PULSEG_ALLOC((size_t)count * sizeof(int));
    block_norot = (int *)PULSEG_ALLOC((size_t)count * sizeof(int));

    if (!block_indices || !base_rows || !base_unique || !base_map || !entry_keys || !entry_unique ||
        !entry_map || !plan_keys || !plan_unique || !plan_map || !block_rotation || !block_norot)
        goto build_fail;

    /* ==== Pass 1: gather per-event info ==== */
    {
        int idx = 0;
        for (n = 0; n < desc->num_blocks; ++n)
        {
            const pulseg_block_table_element *bte = &desc->block_table[n];
            const pulseg_base_block *bdef = &desc->base_blocks[bte->id];
            int has_rf, has_adc, adc_def_id_local;
            int effective_duration_us;
            int gx_key, gy_key, gz_key;
            int active_start_key, active_dur_key;

            if (bte->freq_mod_id < 0)
                continue;

            block_indices[idx] = n;
            effective_duration_us = bdef->duration_us;

            /* Compute active window timing for the base key.  Two blocks
             * that use different RF definitions but have the same active
             * window and gradient shapes will produce identical base
             * waveforms and should share the same base index. */
            has_rf = (bdef->rf_id >= 0 && bdef->rf_id < desc->num_unique_rfs);
            has_adc = (bte->adc_id >= 0 && bte->adc_id < desc->adc_table_size);
            active_start_key = 0;
            active_dur_key = 0;

            if (has_rf)
            {
                const pulseg_rf_definition *rdef = &desc->rf_definitions[bdef->rf_id];
                float rf_end_us = (float)rdef->delay + rdef->stats.duration_us;
                float a_start = (float)rdef->delay;
                float a_end;
                adc_def_id_local = has_adc ? desc->adc_table[bte->adc_id].id : -1;
                if (!has_adc && rf_end_us < 0.75f * (float)effective_duration_us)
                    a_end = (float)effective_duration_us;
                else
                    a_end = rf_end_us;
                if (a_start < 0.0f)
                    a_start = 0.0f;
                if (a_end > (float)effective_duration_us)
                    a_end = (float)effective_duration_us;
                active_start_key = (int)(a_start);
                active_dur_key = (int)(a_end - a_start);
            }
            else if (has_adc)
            {
                adc_def_id_local = desc->adc_table[bte->adc_id].id;
                if (adc_def_id_local >= 0 && adc_def_id_local < desc->num_unique_adcs)
                {
                    const pulseg_adc_definition *adef = &desc->adc_definitions[adc_def_id_local];
                    float a_start = (float)adef->delay;
                    float a_end =
                        a_start + (float)adef->num_samples * (float)adef->dwell_time * 1e-3f;
                    if (a_start < 0.0f)
                        a_start = 0.0f;
                    if (a_end > (float)effective_duration_us)
                        a_end = (float)effective_duration_us;
                    active_start_key = (int)a_start;
                    active_dur_key = (int)(a_end - a_start);
                }
            }

            /* Encode gradient base key as (def_id, shot_index) pair so that
             * blocks sharing the same gradient *shape* but differing only in
             * amplitude map to the same base waveform.  Amplitude differences
             * are handled at the entry dedup level.  */
            {
                int ax_id[3];
                int ax_key[3];
                int a;
                ax_id[0] = bte->gx_id;
                ax_id[1] = bte->gy_id;
                ax_id[2] = bte->gz_id;
                for (a = 0; a < 3; ++a)
                {
                    if (ax_id[a] >= 0 && ax_id[a] < desc->grad_table_size)
                    {
                        int g_def = desc->grad_table[ax_id[a]].id;
                        int shot = desc->grad_table[ax_id[a]].shot_index;
                        ax_key[a] = g_def * PULSEG_MAX_GRAD_SHOTS + shot;
                    }
                    else
                    {
                        ax_key[a] = -1;
                    }
                }
                gx_key = ax_key[0];
                gy_key = ax_key[1];
                gz_key = ax_key[2];
            }

            /* Base dedup key: active-window timing + gradient shape IDs.
             * Using active window (start, duration) instead of rf_id/adc_id
             * merges blocks whose RF definitions differ only in amplitude. */
            base_rows[idx][0] = bte->freq_mod_id;
            base_rows[idx][1] = active_start_key;
            base_rows[idx][2] = active_dur_key;
            base_rows[idx][3] = gx_key;
            base_rows[idx][4] = gy_key;
            base_rows[idx][5] = gz_key;
            base_rows[idx][6] = effective_duration_us;

            /* Rotation: record both rotation_id and norot_flag */
            block_rotation[idx] = (bte->rotation_id >= 0) ? bte->rotation_id : -1;
            block_norot[idx] = bte->norot_flag;

            idx++;
        }
    }

    /* ==== Dedup base waveforms ==== */
    num_base = pulseg__deduplicate_int_rows(
        base_unique,
        base_map,
        (const int *)base_rows,
        count,
        FREQ_MOD_BASE_COLS);

    /* ==== Build base definitions ==== */
    base_defs = (pulseg_freq_mod_definition *)PULSEG_ALLOC(
        (size_t)num_base * sizeof(pulseg_freq_mod_definition));
    base_active_mask = (int *)PULSEG_ALLOC((size_t)num_base * 3 * sizeof(int));
    if (!base_defs)
        goto build_fail;
    memset(base_defs, 0, (size_t)num_base * sizeof(pulseg_freq_mod_definition));
    if (!base_active_mask)
        goto build_fail;
    memset(base_active_mask, 0, (size_t)num_base * 3 * sizeof(int));

    max_samples = 0;
    for (n = 0; n < num_base; ++n)
    {
        int row_idx = base_unique[n];
        int blk_idx = block_indices[row_idx];
        const pulseg_block_table_element *bte = &desc->block_table[blk_idx];
        const pulseg_base_block *bdef = &desc->base_blocks[bte->id];
        int has_rf, has_adc, adc_def_id_local;
        float active_start_us, active_end_us, ref_time_us;
        float target_raster_us;
        int effective_duration_us;

        has_rf = (bdef->rf_id >= 0);
        has_adc = (bte->adc_id >= 0 && bte->adc_id < desc->adc_table_size);
        effective_duration_us = bdef->duration_us;

        /* ---- Determine active region ---- */
        if (has_rf && bdef->rf_id < desc->num_unique_rfs)
        {
            const pulseg_rf_definition *rdef = &desc->rf_definitions[bdef->rf_id];
            float rf_end_us;
            active_start_us = (float)rdef->delay;
            rf_end_us = active_start_us + rdef->stats.duration_us;

            /* Navigator-like RF blocks can have gradients that extend far
               beyond the RF pulse; in that case use the full effective
               block duration for freq-mod synthesis. */
            if (!has_adc && rf_end_us < 0.75f * (float)effective_duration_us)
            {
                active_end_us = (float)effective_duration_us;
            }
            else
            {
                active_end_us = rf_end_us;
            }
            /* RF zero-phase reference = segment RF-anchor isocenter. REQUIRED:
             * the former `rdef->stats.isodelay_us` fallback is deceiving (it
             * silently substitutes a different reference when anchors are absent
             * on the cache path), so a missing anchor is a HARD FAIL. With the
             * timing anchors serialized in the cache (write/read_descriptor),
             * every load path carries them, so this never triggers in practice. */
            {
                int scan_p, found = 0;
                for (scan_p = 0; scan_p < desc->exec_stream_len; ++scan_p)
                {
                    if (desc->exec_stream_block_idx[scan_p] == blk_idx)
                    {
                        int seg_id =
                            (desc->exec_stream_seg_id) ? desc->exec_stream_seg_id[scan_p] : -1;
                        if (seg_id >= 0 && seg_id < desc->num_unique_segments)
                        {
                            const pulseg_virtual_segment *seg = &desc->segment_definitions[seg_id];
                            int kb, ka, blk_offset = -1;
                            for (kb = 0; kb < seg->num_blocks; ++kb)
                            {
                                if (seg->unique_block_indices[kb] == bte->id)
                                {
                                    blk_offset = kb;
                                    break;
                                }
                            }
                            if (blk_offset >= 0 && seg->timing.rf_anchors &&
                                seg->timing.num_rf_anchors > 0)
                            {
                                for (ka = 0; ka < seg->timing.num_rf_anchors; ++ka)
                                {
                                    if (seg->timing.rf_anchors[ka].block_offset == blk_offset)
                                    {
                                        float iso = seg->timing.rf_anchors[ka].isocenter_us;
                                        float as = seg->timing.rf_anchors[ka].start_us;
                                        float iso_rel = iso - as;
                                        if (iso_rel < 0.0f)
                                            iso_rel = 0.0f;
                                        if (iso_rel > (active_end_us - active_start_us))
                                            iso_rel = active_end_us - active_start_us;
                                        ref_time_us = iso_rel;
                                        found = 1;
                                        break;
                                    }
                                }
                            }
                        }
                        break;
                    }
                }
                if (!found)
                {
                    fail_code = PULSEG_ERR_MISSING_KSPACE_ANCHOR;
                    goto build_fail;
                }
            }
        }
        else if (has_adc)
        {
            adc_def_id_local = desc->adc_table[bte->adc_id].id;
            if (adc_def_id_local >= 0 && adc_def_id_local < desc->num_unique_adcs)
            {
                const pulseg_adc_definition *adef = &desc->adc_definitions[adc_def_id_local];
                float adc_dur_us = (float)adef->num_samples * (float)adef->dwell_time * 1e-3f;
                active_start_us = (float)adef->delay;
                active_end_us = active_start_us + adc_dur_us;

                /* ADC zero-phase reference = segment ADC-anchor kzero (echo).
                 * REQUIRED: the former `adc_dur*0.5` midpoint fallback is a guess
                 * (the true echo/kzero needs the k-space trajectory and cannot be
                 * reconstructed here), so a missing anchor is a HARD FAIL. The
                 * serialized cache anchors make this unreachable in practice. */
                {
                    int scan_p, found = 0;
                    for (scan_p = 0; scan_p < desc->exec_stream_len; ++scan_p)
                    {
                        if (desc->exec_stream_block_idx[scan_p] == blk_idx)
                        {
                            int seg_id = desc->exec_stream_seg_id[scan_p];
                            if (seg_id >= 0 && seg_id < desc->num_unique_segments)
                            {
                                const pulseg_virtual_segment *seg =
                                    &desc->segment_definitions[seg_id];
                                int kb, ka, blk_offset = -1;
                                for (kb = 0; kb < seg->num_blocks; ++kb)
                                {
                                    if (seg->unique_block_indices[kb] == bte->id)
                                    {
                                        blk_offset = kb;
                                        break;
                                    }
                                }
                                if (blk_offset >= 0 && seg->timing.adc_anchors &&
                                    seg->timing.num_adc_anchors > 0)
                                {
                                    for (ka = 0; ka < seg->timing.num_adc_anchors; ++ka)
                                    {
                                        if (seg->timing.adc_anchors[ka].block_offset == blk_offset)
                                        {
                                            float kz = seg->timing.adc_anchors[ka].kzero_us;
                                            float as = seg->timing.adc_anchors[ka].start_us;
                                            float kz_rel = kz - as;
                                            if (kz_rel < 0.0f)
                                                kz_rel = 0.0f;
                                            if (kz_rel > adc_dur_us)
                                                kz_rel = adc_dur_us;
                                            ref_time_us = kz_rel;
                                            found = 1;
                                            break;
                                        }
                                    }
                                }
                            }
                            break;
                        }
                    }
                    if (!found)
                    {
                        fail_code = PULSEG_ERR_MISSING_KSPACE_ANCHOR;
                        goto build_fail;
                    }
                }
            }
            else
            {
                continue;
            }
        }
        else
        {
            continue;
        }

        /* Clamp */
        if (active_start_us < 0.0f)
            active_start_us = 0.0f;
        if (active_end_us > (float)effective_duration_us)
            active_end_us = (float)effective_duration_us;
        if (ref_time_us < 0.0f)
            ref_time_us = 0.0f;
        if (ref_time_us > (active_end_us - active_start_us))
            ref_time_us = active_end_us - active_start_us;

        target_raster_us = (has_rf && bdef->rf_id < desc->num_unique_rfs) ? desc->rf_raster_us
                                                                          : desc->adc_raster_us;

        result = build_freq_mod_for_block(
            desc,
            &base_defs[n],
            bte,
            bte->id,
            active_start_us,
            active_end_us,
            ref_time_us,
            target_raster_us);
        if (PULSEG_FAILED(result))
        {
            continue;
        }

        base_active_mask[n * 3 + 0] =
            axis_is_active(base_defs[n].waveform_gx, base_defs[n].num_samples);
        base_active_mask[n * 3 + 1] =
            axis_is_active(base_defs[n].waveform_gy, base_defs[n].num_samples);
        base_active_mask[n * 3 + 2] =
            axis_is_active(base_defs[n].waveform_gz, base_defs[n].num_samples);

        if (base_defs[n].num_samples > max_samples)
            max_samples = base_defs[n].num_samples;
    }

    /* ==== Merge bases with identical waveforms ====
     * The integer base key may over-distinguish bases (e.g. different gradient
     * event IDs for channels that are all-zero within the active window).
     * Remap base_map so that waveform-identical bases share one index.  */
    {
        int *base_remap = (int *)PULSEG_ALLOC((size_t)num_base * sizeof(int));
        if (base_remap)
        {
            int b, b2, ch2, s2;
            for (b = 0; b < num_base; ++b)
                base_remap[b] = b;

            for (b = 1; b < num_base; ++b)
            {
                for (b2 = 0; b2 < b; ++b2)
                {
                    const pulseg_freq_mod_definition *da = &base_defs[b];
                    const pulseg_freq_mod_definition *db = &base_defs[b2];
                    int identical = 1;
                    if (da->num_samples != db->num_samples || da->raster_us != db->raster_us)
                    {
                        identical = 0;
                    }
                    else
                    {
                        const float *wa[3];
                        const float *wb[3];
                        wa[0] = da->waveform_gx;
                        wa[1] = da->waveform_gy;
                        wa[2] = da->waveform_gz;
                        wb[0] = db->waveform_gx;
                        wb[1] = db->waveform_gy;
                        wb[2] = db->waveform_gz;
                        for (ch2 = 0; ch2 < 3 && identical; ++ch2)
                        {
                            if (!wa[ch2] && !wb[ch2])
                                continue;
                            if (!wa[ch2] || !wb[ch2])
                            {
                                identical = 0;
                                break;
                            }
                            for (s2 = 0; s2 < da->num_samples; ++s2)
                            {
                                if (wa[ch2][s2] != wb[ch2][s2])
                                {
                                    identical = 0;
                                    break;
                                }
                            }
                        }
                        /* Also compare ref_integral and ref_time */
                        if (identical)
                        {
                            for (ch2 = 0; ch2 < 3; ++ch2)
                            {
                                if (da->ref_integral[ch2] != db->ref_integral[ch2])
                                {
                                    identical = 0;
                                    break;
                                }
                            }
                        }
                    }
                    if (identical)
                    {
                        base_remap[b] = base_remap[b2];
                        break;
                    }
                }
            }

            /* Apply remap to base_map */
            for (b = 0; b < count; ++b)
                base_map[b] = base_remap[base_map[b]];

            PULSEG_FREE(base_remap);
        }
    }

    /* ==== Pass 2: build entry dedup keys ==== */
    for (n = 0; n < count; ++n)
    {
        int blk_idx = block_indices[n];
        const pulseg_block_table_element *bte = &desc->block_table[blk_idx];
        int bi = base_map[n];
        float amp[3];

        amp[0] = (bte->gx_id >= 0 && bte->gx_id < desc->grad_table_size)
            ? desc->grad_table[bte->gx_id].amplitude
            : 0.0f;
        amp[1] = (bte->gy_id >= 0 && bte->gy_id < desc->grad_table_size)
            ? desc->grad_table[bte->gy_id].amplitude
            : 0.0f;
        amp[2] = (bte->gz_id >= 0 && bte->gz_id < desc->grad_table_size)
            ? desc->grad_table[bte->gz_id].amplitude
            : 0.0f;

        /* Zero out amplitudes for channels inactive in the active window. */
        if (bi >= 0 && bi < num_base)
        {
            if (!base_active_mask[bi * 3 + 0])
                amp[0] = 0.0f;
            if (!base_active_mask[bi * 3 + 1])
                amp[1] = 0.0f;
            if (!base_active_mask[bi * 3 + 2])
                amp[2] = 0.0f;
        }

        memset(&entry_keys[n], 0, sizeof(entry_keys[n]));
        entry_keys[n].base_idx = bi;
        entry_keys[n].amp[0] = amp[0];
        entry_keys[n].amp[1] = amp[1];
        entry_keys[n].amp[2] = amp[2];
    }

    /* ==== Dedup entries ==== */
    num_entries = dedup_keys(
        entry_unique,
        entry_map,
        (const char *)entry_keys,
        count,
        (int)sizeof(entry_key_t));

    /* ==== Build amplitude-scaled 3-channel entries ==== */
    if (max_samples <= 0)
        max_samples = 1; /* safety */

    lib->num_entries = num_entries;
    lib->max_samples = max_samples;
    lib->raster_us = (num_base > 0 && base_defs[0].raster_us > 0.0f) ? base_defs[0].raster_us
                                                                     : desc->grad_raster_us;

    lib->entry_num_samples = (int *)PULSEG_ALLOC((size_t)num_entries * sizeof(int));
    lib->entry_waveform_3ch =
        (float *)PULSEG_ALLOC((size_t)num_entries * max_samples * 3 * sizeof(float));
    lib->entry_ref_3ch = (float *)PULSEG_ALLOC((size_t)num_entries * 3 * sizeof(float));
    if (!lib->entry_num_samples || !lib->entry_waveform_3ch || !lib->entry_ref_3ch)
        goto build_fail;

    memset(lib->entry_waveform_3ch, 0, (size_t)num_entries * max_samples * 3 * sizeof(float));

    for (n = 0; n < num_entries; ++n)
    {
        int ev = entry_unique[n];
        int bi = base_map[ev];
        const pulseg_freq_mod_definition *bd = &base_defs[bi];
        float amp[3];
        int s, ch;

        /* Recover effective amplitudes from the key */
        amp[0] = entry_keys[ev].amp[0];
        amp[1] = entry_keys[ev].amp[1];
        amp[2] = entry_keys[ev].amp[2];

        lib->entry_num_samples[n] = bd->num_samples;

        /* Scale base waveforms by amplitude: e_j(t) = A_j * g_j(t) */
        {
            const float *src[3];
            src[0] = bd->waveform_gx;
            src[1] = bd->waveform_gy;
            src[2] = bd->waveform_gz;
            for (ch = 0; ch < 3; ++ch)
            {
                float *dst = lib->entry_waveform_3ch + (size_t)n * max_samples * 3 +
                    (size_t)ch * max_samples;
                if (src[ch])
                {
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
    if (desc->num_rotations > 0 && desc->rotation_matrices)
    {
        lib->rotations = PULSEG_ALLOC((size_t)desc->num_rotations * sizeof(float[9]));
        if (!lib->rotations)
            goto build_fail;
        memcpy(
            lib->rotations,
            desc->rotation_matrices,
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
            if (block_norot[n])
            {
                has_norot = 1;
                break;
            }

        if (has_norot && fov_rotation && !pulseg__is_identity3(fov_rotation))
        {
            /* Non-identity FOV rotation: compute R_eff for norot blocks */
            int cap = lib->num_rotations + count; /* upper bound */
            float(*expanded)[9] = PULSEG_ALLOC((size_t)cap * sizeof(float[9]));
            int num_exp;

            if (!expanded)
                goto build_fail;

            /* Copy existing rotations */
            if (lib->num_rotations > 0 && lib->rotations)
                memcpy(expanded, lib->rotations, (size_t)lib->num_rotations * sizeof(float[9]));
            num_exp = lib->num_rotations;

            for (n = 0; n < count; ++n)
            {
                float R_eff[9];
                int found, u;

                if (!block_norot[n])
                    continue;

                /* Compute R_eff = R_presc^T @ R_ext (or R_presc^T if no
                 * rotation event). */
                if (block_rotation[n] >= 0 && block_rotation[n] < desc->num_rotations)
                {
                    matmul_AtB(R_eff, fov_rotation, desc->rotation_matrices[block_rotation[n]]);
                }
                else
                {
                    /* No rotation event: R_eff = R_presc^T */
                    int i, j;
                    for (i = 0; i < 3; ++i)
                        for (j = 0; j < 3; ++j)
                            R_eff[i * 3 + j] = fov_rotation[j * 3 + i];
                }

                /* Dedup: search for R_eff among existing + new entries */
                found = -1;
                for (u = 0; u < num_exp; ++u)
                {
                    if (rotmat_equal(R_eff, expanded[u]))
                    {
                        found = u;
                        break;
                    }
                }
                if (found < 0)
                {
                    memcpy(expanded[num_exp], R_eff, sizeof(float[9]));
                    found = num_exp++;
                }
                block_rotation[n] = found;
            }

            /* Replace rotation library with expanded version */
            if (lib->rotations)
                PULSEG_FREE(lib->rotations);
            lib->rotations = expanded;
            lib->num_rotations = num_exp;
        }
        else if (has_norot)
        {
            /* FOV rotation is identity (or NULL): for norot blocks the
             * effective rotation equals the original rotation event
             * (which may be -1 = identity).  No expansion needed, but
             * block_rotation already holds the correct value. */
        }
    }

    /* ==== Pass 3: plan dedup + scan_to_plan ====
     *
     * Auto-detect tr_scoped mode: if freq-mod blocks appear in more
     * than one distinct segment across the scan table, we must dedup
     * at the scan-table level (each TR gets its own plan instances).
     * Otherwise the block-level dedup suffices.
     */
    {
        int use_tr_scope = 0;
        int *blk_to_entry = NULL;
        int *blk_to_rotation_map = NULL;
        int *blk_to_base = NULL;
        int *blk_to_active_mask = NULL;
        int scan_count, sc, i;
        plan_key_t *scan_plan_keys = NULL;
        int *scan_plan_unique = NULL;
        int *scan_plan_map = NULL;
        int tr_size = desc->tr_descriptor.tr_size;
        int tr_scope_scan_id = -1;
        int tr_start_scan_pos = -1;

        /* Build block -> entry/rotation maps (needed by both paths) */
        blk_to_entry = (int *)PULSEG_ALLOC((size_t)desc->num_blocks * sizeof(int));
        blk_to_rotation_map = (int *)PULSEG_ALLOC((size_t)desc->num_blocks * sizeof(int));
        blk_to_base = (int *)PULSEG_ALLOC((size_t)desc->num_blocks * sizeof(int));
        blk_to_active_mask = (int *)PULSEG_ALLOC((size_t)desc->num_blocks * 3 * sizeof(int));
        if (!blk_to_entry || !blk_to_rotation_map || !blk_to_base || !blk_to_active_mask)
        {
            if (blk_to_entry)
                PULSEG_FREE(blk_to_entry);
            if (blk_to_rotation_map)
                PULSEG_FREE(blk_to_rotation_map);
            if (blk_to_base)
                PULSEG_FREE(blk_to_base);
            if (blk_to_active_mask)
                PULSEG_FREE(blk_to_active_mask);
            goto build_fail;
        }
        for (n = 0; n < desc->num_blocks; ++n)
        {
            blk_to_entry[n] = -1;
            blk_to_rotation_map[n] = -1;
            blk_to_base[n] = -1;
            blk_to_active_mask[n * 3 + 0] = 0;
            blk_to_active_mask[n * 3 + 1] = 0;
            blk_to_active_mask[n * 3 + 2] = 0;
        }
        for (n = 0; n < count; ++n)
        {
            int bi = base_map[n];
            blk_to_entry[block_indices[n]] = entry_map[n];
            blk_to_rotation_map[block_indices[n]] = block_rotation[n];
            blk_to_base[block_indices[n]] = bi;
            if (bi >= 0 && bi < num_base)
            {
                blk_to_active_mask[block_indices[n] * 3 + 0] = base_active_mask[bi * 3 + 0];
                blk_to_active_mask[block_indices[n] * 3 + 1] = base_active_mask[bi * 3 + 1];
                blk_to_active_mask[block_indices[n] * 3 + 2] = base_active_mask[bi * 3 + 2];
            }
        }

        lib->scan_inactive_area_3ch =
            (float *)PULSEG_ALLOC((size_t)lib->exec_stream_len * 3 * sizeof(float));
        lib->scan_phase_extra = (float *)PULSEG_ALLOC((size_t)lib->exec_stream_len * sizeof(float));
        if (!lib->scan_inactive_area_3ch || !lib->scan_phase_extra)
        {
            PULSEG_FREE(blk_to_entry);
            PULSEG_FREE(blk_to_rotation_map);
            PULSEG_FREE(blk_to_base);
            PULSEG_FREE(blk_to_active_mask);
            goto build_fail;
        }
        memset(lib->scan_inactive_area_3ch, 0, (size_t)lib->exec_stream_len * 3 * sizeof(float));
        memset(lib->scan_phase_extra, 0, (size_t)lib->exec_stream_len * sizeof(float));

        /* Detect multi-segment freq_mod: check how many distinct segments
         * contain freq_mod blocks in the scan table. */
        if (desc->exec_stream_seg_id && desc->exec_stream_len > 0)
        {
            int first_seg = -1;
            int i;
            for (i = 0; i < desc->exec_stream_len; ++i)
            {
                int bti = desc->exec_stream_block_idx[i];
                if (bti >= 0 && bti < desc->num_blocks && blk_to_entry[bti] >= 0)
                {
                    int sid = desc->exec_stream_seg_id[i];
                    if (first_seg < 0)
                        first_seg = sid;
                    else if (sid != first_seg)
                    {
                        use_tr_scope = 1;
                        break;
                    }
                }
            }
        }

        scan_count = 0;
        for (i = 0; i < desc->exec_stream_len; ++i)
        {
            int bti = desc->exec_stream_block_idx[i];
            if (bti >= 0 && bti < desc->num_blocks && blk_to_entry[bti] >= 0)
                scan_count++;
        }

        scan_plan_keys = (plan_key_t *)PULSEG_ALLOC((size_t)scan_count * sizeof(plan_key_t));
        scan_plan_unique = (int *)PULSEG_ALLOC((size_t)scan_count * sizeof(int));
        scan_plan_map = (int *)PULSEG_ALLOC((size_t)scan_count * sizeof(int));
        if (!scan_plan_keys || !scan_plan_unique || !scan_plan_map)
        {
            if (scan_plan_keys)
                PULSEG_FREE(scan_plan_keys);
            if (scan_plan_unique)
                PULSEG_FREE(scan_plan_unique);
            if (scan_plan_map)
                PULSEG_FREE(scan_plan_map);
            PULSEG_FREE(blk_to_entry);
            PULSEG_FREE(blk_to_rotation_map);
            PULSEG_FREE(blk_to_base);
            PULSEG_FREE(blk_to_active_mask);
            goto build_fail;
        }

        sc = 0;
        for (i = 0; i < desc->exec_stream_len; ++i)
        {
            int bti = desc->exec_stream_block_idx[i];

            if (desc->exec_stream_tr_start && desc->exec_stream_tr_start[i])
            {
                tr_scope_scan_id++;
                tr_start_scan_pos = i;
            }
            if (tr_scope_scan_id < 0)
            {
                tr_scope_scan_id = 0;
                tr_start_scan_pos = i;
            }

            if (!(bti >= 0 && bti < desc->num_blocks && blk_to_entry[bti] >= 0))
            {
                lib->scan_to_plan[i] = -1;
                continue;
            }

            {
                const pulseg_block_table_element *bte = &desc->block_table[bti];
                const pulseg_base_block *bdef = &desc->base_blocks[bte->id];
                int has_rf = (bdef->rf_id >= 0 && bdef->rf_id < desc->num_unique_rfs);
                int has_adc = (bte->adc_id >= 0 && bte->adc_id < desc->adc_table_size);
                float ref_abs_us = 0.0f;
                int tr_start_blk_local = -1;
                int ch;
                int entry_idx_local = blk_to_entry[bti];

                memset(&scan_plan_keys[sc], 0, sizeof(scan_plan_keys[sc]));
                scan_plan_keys[sc].entry_idx = blk_to_entry[bti];
                scan_plan_keys[sc].rot_idx = blk_to_rotation_map[bti];
                scan_plan_keys[sc].tr_scope_id = use_tr_scope ? tr_scope_scan_id : 0;

                if (tr_start_scan_pos >= 0 && tr_start_scan_pos < desc->exec_stream_len)
                {
                    tr_start_blk_local = desc->exec_stream_block_idx[tr_start_scan_pos];
                }
                if (tr_start_blk_local < 0)
                {
                    tr_start_blk_local = (tr_size > 0) ? (bti / tr_size) * tr_size : bti;
                }

                if (has_rf)
                {
                    const pulseg_rf_definition *rdef = &desc->rf_definitions[bdef->rf_id];
                    ref_abs_us = (float)rdef->delay + (float)rdef->stats.isodelay_us;

                    /* Prefer segment RF-anchor isocenter when available. */
                    {
                        int seg_id = (desc->exec_stream_seg_id) ? desc->exec_stream_seg_id[i] : -1;
                        if (seg_id >= 0 && seg_id < desc->num_unique_segments)
                        {
                            const pulseg_virtual_segment *seg = &desc->segment_definitions[seg_id];
                            int kb, ka, blk_offset = -1;
                            for (kb = 0; kb < seg->num_blocks; ++kb)
                            {
                                if (seg->unique_block_indices[kb] == bte->id)
                                {
                                    blk_offset = kb;
                                    break;
                                }
                            }
                            if (blk_offset >= 0 && seg->timing.rf_anchors &&
                                seg->timing.num_rf_anchors > 0)
                            {
                                for (ka = 0; ka < seg->timing.num_rf_anchors; ++ka)
                                {
                                    if (seg->timing.rf_anchors[ka].block_offset == blk_offset)
                                    {
                                        float iso = seg->timing.rf_anchors[ka].isocenter_us;
                                        float as = seg->timing.rf_anchors[ka].start_us;
                                        float iso_rel = iso - as;
                                        if (iso_rel < 0.0f)
                                            iso_rel = 0.0f;
                                        ref_abs_us = (float)rdef->delay + iso_rel;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                else if (has_adc)
                {
                    int adc_def_id_local = desc->adc_table[bte->adc_id].id;
                    if (adc_def_id_local >= 0 && adc_def_id_local < desc->num_unique_adcs)
                    {
                        const pulseg_adc_definition *adef =
                            &desc->adc_definitions[adc_def_id_local];
                        float adc_dur_us =
                            (float)adef->num_samples * (float)adef->dwell_time * 1e-3f;
                        float ref_time_rel_us = adc_dur_us * 0.5f;
                        int seg_id = (desc->exec_stream_seg_id) ? desc->exec_stream_seg_id[i] : -1;

                        if (seg_id >= 0 && seg_id < desc->num_unique_segments)
                        {
                            const pulseg_virtual_segment *seg = &desc->segment_definitions[seg_id];
                            int kb, ka, blk_offset = -1;
                            for (kb = 0; kb < seg->num_blocks; ++kb)
                            {
                                if (seg->unique_block_indices[kb] == bte->id)
                                {
                                    blk_offset = kb;
                                    break;
                                }
                            }
                            if (blk_offset >= 0 && seg->timing.adc_anchors &&
                                seg->timing.num_adc_anchors > 0)
                            {
                                for (ka = 0; ka < seg->timing.num_adc_anchors; ++ka)
                                {
                                    if (seg->timing.adc_anchors[ka].block_offset == blk_offset)
                                    {
                                        float kz = seg->timing.adc_anchors[ka].kzero_us;
                                        float as = seg->timing.adc_anchors[ka].start_us;
                                        float kz_rel = kz - as;
                                        if (kz_rel < 0.0f)
                                            kz_rel = 0.0f;
                                        if (kz_rel > adc_dur_us)
                                            kz_rel = adc_dur_us;
                                        ref_time_rel_us = kz_rel;
                                        break;
                                    }
                                }
                            }
                        }

                        ref_abs_us = (float)adef->delay + ref_time_rel_us;
                    }
                }

                if (ref_abs_us < 0.0f)
                    ref_abs_us = 0.0f;
                if (ref_abs_us > (float)bdef->duration_us)
                    ref_abs_us = (float)bdef->duration_us;

                if (tr_start_blk_local < 0)
                    tr_start_blk_local = 0;
                if (tr_start_blk_local > bti)
                    tr_start_blk_local = bti;

                for (ch = 0; ch < 3; ++ch)
                {
                    float area_sum = 0.0f;
                    int axis_active = 0;

                    if (!use_tr_scope)
                    {
                        scan_plan_keys[sc].inactive_ref_3ch[ch] = 0.0f;
                        continue;
                    }

                    if (entry_idx_local >= 0 && entry_idx_local < lib->num_entries)
                    {
                        int s;
                        const float *ew = lib->entry_waveform_3ch +
                            (size_t)entry_idx_local * lib->max_samples * 3 +
                            (size_t)ch * lib->max_samples;
                        int ens = lib->entry_num_samples[entry_idx_local];
                        for (s = 0; s < ens; ++s)
                        {
                            if ((float)fabs(ew[s]) > 1e-8f)
                            {
                                axis_active = 1;
                                break;
                            }
                        }
                    }

                    if (axis_active)
                    {
                        scan_plan_keys[sc].inactive_ref_3ch[ch] = 0.0f;
                        continue;
                    }

                    {
                        int bi = blk_to_base[bti];
                        int gt_id;
                        float amp = 0.0f;

                        if (ch == 0)
                            gt_id = bte->gx_id;
                        else if (ch == 1)
                            gt_id = bte->gy_id;
                        else
                            gt_id = bte->gz_id;

                        if (gt_id >= 0 && gt_id < desc->grad_table_size)
                            amp = desc->grad_table[gt_id].amplitude;

                        if (bi >= 0 && bi < num_base)
                            area_sum = base_defs[bi].ref_integral[ch] * amp;
                    }
                    scan_plan_keys[sc].inactive_ref_3ch[ch] = area_sum;
                    lib->scan_inactive_area_3ch[i * 3 + ch] =
                        scan_plan_keys[sc].inactive_ref_3ch[ch];
                }

                sc++;
            }
        }

        num_plan = dedup_keys(
            scan_plan_unique,
            scan_plan_map,
            (const char *)scan_plan_keys,
            scan_count,
            (int)sizeof(plan_key_t));

        lib->num_plan_instances = num_plan;
        lib->pi_entry_idx = (int *)PULSEG_ALLOC((size_t)num_plan * sizeof(int));
        lib->pi_rotation_idx = (int *)PULSEG_ALLOC((size_t)num_plan * sizeof(int));
        if (!lib->pi_entry_idx || !lib->pi_rotation_idx)
        {
            PULSEG_FREE(scan_plan_keys);
            PULSEG_FREE(scan_plan_unique);
            PULSEG_FREE(scan_plan_map);
            PULSEG_FREE(blk_to_entry);
            PULSEG_FREE(blk_to_rotation_map);
            PULSEG_FREE(blk_to_base);
            PULSEG_FREE(blk_to_active_mask);
            goto build_fail;
        }
        for (n = 0; n < num_plan; ++n)
        {
            int ev = scan_plan_unique[n];
            lib->pi_entry_idx[n] = scan_plan_keys[ev].entry_idx;
            lib->pi_rotation_idx[n] = scan_plan_keys[ev].rot_idx;
        }

        result = alloc_plan(lib);
        if (PULSEG_FAILED(result))
        {
            PULSEG_FREE(scan_plan_keys);
            PULSEG_FREE(scan_plan_unique);
            PULSEG_FREE(scan_plan_map);
            PULSEG_FREE(blk_to_entry);
            PULSEG_FREE(blk_to_rotation_map);
            PULSEG_FREE(blk_to_base);
            PULSEG_FREE(blk_to_active_mask);
            goto build_fail;
        }
        for (n = 0; n < num_plan; ++n)
            lib->plan_num_samples[n] = lib->entry_num_samples[lib->pi_entry_idx[n]];

        sc = 0;
        for (i = 0; i < desc->exec_stream_len; ++i)
        {
            int bti = desc->exec_stream_block_idx[i];
            if (bti >= 0 && bti < desc->num_blocks && blk_to_entry[bti] >= 0)
            {
                lib->scan_to_plan[i] = scan_plan_map[sc];
                sc++;
            }
            else
            {
                lib->scan_to_plan[i] = -1;
            }
        }

        PULSEG_FREE(scan_plan_keys);
        PULSEG_FREE(scan_plan_unique);
        PULSEG_FREE(scan_plan_map);

        PULSEG_FREE(blk_to_entry);
        PULSEG_FREE(blk_to_rotation_map);
        PULSEG_FREE(blk_to_base);
        PULSEG_FREE(blk_to_active_mask);
    }

    /* ==== Compute plan waveforms ==== */
    compute_plan_waveforms(lib, shift_m);
    compute_scan_phase_extra(lib, shift_m);

    /* Keep accessor-only semantics for phase: all channel contributions
     * are precomputed in plan_phase from the 3-channel definition itself. */

    /* ==== Free temporary working arrays ==== */
    free_base_defs(base_defs, num_base);
    PULSEG_FREE(block_indices);
    PULSEG_FREE(base_rows);
    PULSEG_FREE(base_unique);
    PULSEG_FREE(base_map);
    PULSEG_FREE(entry_keys);
    PULSEG_FREE(entry_unique);
    PULSEG_FREE(entry_map);
    PULSEG_FREE(plan_keys);
    PULSEG_FREE(plan_unique);
    PULSEG_FREE(plan_map);
    PULSEG_FREE(block_rotation);
    PULSEG_FREE(block_norot);
    PULSEG_FREE(base_active_mask);

    *out_lib = lib;
    return PULSEG_SUCCESS;

build_fail:
    free_base_defs(base_defs, num_base);
    if (block_indices)
        PULSEG_FREE(block_indices);
    if (base_rows)
        PULSEG_FREE(base_rows);
    if (base_unique)
        PULSEG_FREE(base_unique);
    if (base_map)
        PULSEG_FREE(base_map);
    if (entry_keys)
        PULSEG_FREE(entry_keys);
    if (entry_unique)
        PULSEG_FREE(entry_unique);
    if (entry_map)
        PULSEG_FREE(entry_map);
    if (plan_keys)
        PULSEG_FREE(plan_keys);
    if (plan_unique)
        PULSEG_FREE(plan_unique);
    if (plan_map)
        PULSEG_FREE(plan_map);
    if (block_rotation)
        PULSEG_FREE(block_rotation);
    if (block_norot)
        PULSEG_FREE(block_norot);
    if (base_active_mask)
        PULSEG_FREE(base_active_mask);
    if (lib)
    {
        freq_mod_library_free(lib);
    }
    return fail_code;
}

/* ================================================================== */
/*  Update                                                            */
/* ================================================================== */

static int update_freq_mod_library(pulseg_freq_mod_library *lib, const float *shift_m)
{
    if (!lib || !shift_m)
        return PULSEG_ERR_NULL_POINTER;
    if (!lib->entry_waveform_3ch || !lib->entry_ref_3ch)
        return PULSEG_ERR_INVALID_ARGUMENT; /* 3ch data already freed */
    if (lib->num_plan_instances == 0)
        return PULSEG_SUCCESS;

    compute_plan_waveforms(lib, shift_m);
    compute_scan_phase_extra(lib, shift_m);
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Accessor                                                          */
/* ================================================================== */

static int freq_mod_library_get(
    const pulseg_freq_mod_library *lib,
    const short **out_hw_waveform,
    int *out_num_samples,
    float *out_phase_rad,
    int exec_stream_pos)
{
    int pi;
    if (!lib || !out_hw_waveform || !out_num_samples || !out_phase_rad)
        return 0;
    if (exec_stream_pos < 0 || exec_stream_pos >= lib->exec_stream_len)
        return 0;
    pi = lib->scan_to_plan[exec_stream_pos];
    if (pi < 0)
        return 0;
    *out_hw_waveform = lib->plan_hw_waveforms[pi];
    *out_num_samples = lib->plan_num_samples[pi];
    *out_phase_rad = lib->plan_phase[pi] +
        ((lib->scan_phase_extra && exec_stream_pos < lib->exec_stream_len)
             ? lib->scan_phase_extra[exec_stream_pos]
             : 0.0f);
    return 1;
}

/* ================================================================== */
/*  Cache write                                                       */
/* ================================================================== */

static int freq_mod_library_write_cache(const pulseg_freq_mod_library *lib, FILE *f)
{
    if (!lib || !f)
        return PULSEG_ERR_NULL_POINTER;

    /* Entry metadata */
    if (fwrite(&lib->num_entries, sizeof(int), 1, f) != 1)
        goto write_fail;
    if (fwrite(&lib->max_samples, sizeof(int), 1, f) != 1)
        goto write_fail;
    if (fwrite(&lib->raster_us, sizeof(float), 1, f) != 1)
        goto write_fail;

    if (lib->num_entries > 0)
    {
        if (fwrite(lib->entry_num_samples, sizeof(int), (size_t)lib->num_entries, f) !=
            (size_t)lib->num_entries)
            goto write_fail;
    }

    /* 3-channel waveforms (always present) */
    if (lib->num_entries > 0 && lib->max_samples > 0)
    {
        size_t total = (size_t)lib->num_entries * lib->max_samples * 3;
        if (fwrite(lib->entry_waveform_3ch, sizeof(float), total, f) != total)
            goto write_fail;

        if (fwrite(lib->entry_ref_3ch, sizeof(float), (size_t)lib->num_entries * 3, f) !=
            (size_t)lib->num_entries * 3)
            goto write_fail;
    }

    /* Rotations */
    if (fwrite(&lib->num_rotations, sizeof(int), 1, f) != 1)
        goto write_fail;
    if (lib->num_rotations > 0)
    {
        if (fwrite(lib->rotations, sizeof(float) * 9, (size_t)lib->num_rotations, f) !=
            (size_t)lib->num_rotations)
            goto write_fail;
    }

    /* Plan instance metadata */
    if (fwrite(&lib->num_plan_instances, sizeof(int), 1, f) != 1)
        goto write_fail;
    if (lib->num_plan_instances > 0)
    {
        if (fwrite(lib->pi_entry_idx, sizeof(int), (size_t)lib->num_plan_instances, f) !=
            (size_t)lib->num_plan_instances)
            goto write_fail;
        if (fwrite(lib->pi_rotation_idx, sizeof(int), (size_t)lib->num_plan_instances, f) !=
            (size_t)lib->num_plan_instances)
            goto write_fail;
        if (fwrite(lib->plan_num_samples, sizeof(int), (size_t)lib->num_plan_instances, f) !=
            (size_t)lib->num_plan_instances)
            goto write_fail;

        /* plan_waveform_data / plan_phase are shift-dependent and recomputed
         * from the base on load (compute_plan_waveforms); they are NOT cached. */
    }

    /* Scan-table mapping */
    if (fwrite(&lib->exec_stream_len, sizeof(int), 1, f) != 1)
        goto write_fail;
    if (lib->exec_stream_len > 0)
    {
        if (fwrite(lib->scan_to_plan, sizeof(int), (size_t)lib->exec_stream_len, f) !=
            (size_t)lib->exec_stream_len)
            goto write_fail;

        /* Inactive-axis gradient areas for phase tracking */
        {
            int has_ia = (lib->scan_inactive_area_3ch != NULL);
            if (fwrite(&has_ia, sizeof(int), 1, f) != 1)
                goto write_fail;
            if (has_ia)
            {
                size_t ia_total = (size_t)lib->exec_stream_len * 3;
                if (fwrite(lib->scan_inactive_area_3ch, sizeof(float), ia_total, f) != ia_total)
                    goto write_fail;
            }
        }
    }

    return PULSEG_SUCCESS;

write_fail:
    return PULSEG_ERR_FILE_READ_FAILED;
}

/* ================================================================== */
/*  Cache read                                                        */
/* ================================================================== */

/* The unified cache is written in the writer's native byte order; the outer
 * header carries an endian marker and the COMMON/ROTATIONS/SHAPES/SCANLOOP
 * payload readers already swap on mismatch. This payload is read on the
 * @rsp/@pg target board (big-endian PowerPC on LX scanners) from a file
 * written by the little-endian host, so it needs the same treatment via the
 * shared pulseg__swap4 helpers (pulseg_internal.h). */

static int freq_mod_library_read_cache(
    pulseg_freq_mod_library **out_lib,
    FILE *f,
    const float *shift_m,
    int do_swap)
{
    pulseg_freq_mod_library *lib = NULL;
    int result;

    if (!out_lib || !f || !shift_m)
        return PULSEG_ERR_NULL_POINTER;
    *out_lib = NULL;

    lib = (pulseg_freq_mod_library *)PULSEG_ALLOC(sizeof(*lib));
    if (!lib)
        return PULSEG_ERR_ALLOC_FAILED;
    memset(lib, 0, sizeof(*lib));

    /* Entry metadata */
    if (fread(&lib->num_entries, sizeof(int), 1, f) != 1)
        goto read_fail;
    if (fread(&lib->max_samples, sizeof(int), 1, f) != 1)
        goto read_fail;
    if (fread(&lib->raster_us, sizeof(float), 1, f) != 1)
        goto read_fail;
    if (do_swap)
    {
        pulseg__swap4(&lib->num_entries);
        pulseg__swap4(&lib->max_samples);
        pulseg__swap4(&lib->raster_us);
    }

    if (lib->num_entries > 0)
    {
        lib->entry_num_samples = (int *)PULSEG_ALLOC((size_t)lib->num_entries * sizeof(int));
        if (!lib->entry_num_samples)
            goto read_fail;
        if (fread(lib->entry_num_samples, sizeof(int), (size_t)lib->num_entries, f) !=
            (size_t)lib->num_entries)
            goto read_fail;
        if (do_swap)
            pulseg__swap4_array(lib->entry_num_samples, (size_t)lib->num_entries);
    }

    /* 3-channel waveforms (always present) */
    if (lib->num_entries > 0 && lib->max_samples > 0)
    {
        size_t total = (size_t)lib->num_entries * lib->max_samples * 3;
        lib->entry_waveform_3ch = (float *)PULSEG_ALLOC(total * sizeof(float));
        if (!lib->entry_waveform_3ch)
            goto read_fail;
        if (fread(lib->entry_waveform_3ch, sizeof(float), total, f) != total)
            goto read_fail;
        if (do_swap)
            pulseg__swap4_array(lib->entry_waveform_3ch, total);

        {
            size_t reftotal = (size_t)lib->num_entries * 3;
            lib->entry_ref_3ch = (float *)PULSEG_ALLOC(reftotal * sizeof(float));
            if (!lib->entry_ref_3ch)
                goto read_fail;
            if (fread(lib->entry_ref_3ch, sizeof(float), reftotal, f) != reftotal)
                goto read_fail;
            if (do_swap)
                pulseg__swap4_array(lib->entry_ref_3ch, reftotal);
        }
    }

    /* Rotations */
    if (fread(&lib->num_rotations, sizeof(int), 1, f) != 1)
        goto read_fail;
    if (do_swap)
        pulseg__swap4(&lib->num_rotations);
    if (lib->num_rotations > 0)
    {
        lib->rotations = PULSEG_ALLOC((size_t)lib->num_rotations * sizeof(float[9]));
        if (!lib->rotations)
            goto read_fail;
        if (fread(lib->rotations, sizeof(float) * 9, (size_t)lib->num_rotations, f) !=
            (size_t)lib->num_rotations)
            goto read_fail;
        if (do_swap)
            pulseg__swap4_array(lib->rotations, (size_t)lib->num_rotations * 9);
    }

    /* Plan instance metadata */
    if (fread(&lib->num_plan_instances, sizeof(int), 1, f) != 1)
        goto read_fail;
    if (do_swap)
        pulseg__swap4(&lib->num_plan_instances);
    if (lib->num_plan_instances > 0)
    {
        lib->pi_entry_idx = (int *)PULSEG_ALLOC((size_t)lib->num_plan_instances * sizeof(int));
        lib->pi_rotation_idx = (int *)PULSEG_ALLOC((size_t)lib->num_plan_instances * sizeof(int));
        if (!lib->pi_entry_idx || !lib->pi_rotation_idx)
            goto read_fail;

        if (fread(lib->pi_entry_idx, sizeof(int), (size_t)lib->num_plan_instances, f) !=
            (size_t)lib->num_plan_instances)
            goto read_fail;
        if (fread(lib->pi_rotation_idx, sizeof(int), (size_t)lib->num_plan_instances, f) !=
            (size_t)lib->num_plan_instances)
            goto read_fail;
        if (do_swap)
        {
            pulseg__swap4_array(lib->pi_entry_idx, (size_t)lib->num_plan_instances);
            pulseg__swap4_array(lib->pi_rotation_idx, (size_t)lib->num_plan_instances);
        }

        /* Allocate plan arrays */
        result = alloc_plan(lib);
        if (PULSEG_FAILED(result))
            goto read_fail;
        if (fread(lib->plan_num_samples, sizeof(int), (size_t)lib->num_plan_instances, f) !=
            (size_t)lib->num_plan_instances)
            goto read_fail;
        if (do_swap)
            pulseg__swap4_array(lib->plan_num_samples, (size_t)lib->num_plan_instances);

        /* plan_waveform_data / plan_phase are NOT cached (shift-dependent).
         * alloc_plan already set up the plan_waveforms row pointers, and
         * compute_plan_waveforms (below) fills them from the base for the
         * current shift. */
    }

    /* Scan-table mapping */
    if (fread(&lib->exec_stream_len, sizeof(int), 1, f) != 1)
        goto read_fail;
    if (do_swap)
        pulseg__swap4(&lib->exec_stream_len);
    if (lib->exec_stream_len > 0)
    {
        lib->scan_to_plan = (int *)PULSEG_ALLOC((size_t)lib->exec_stream_len * sizeof(int));
        if (!lib->scan_to_plan)
            goto read_fail;
        if (fread(lib->scan_to_plan, sizeof(int), (size_t)lib->exec_stream_len, f) !=
            (size_t)lib->exec_stream_len)
            goto read_fail;
        if (do_swap)
            pulseg__swap4_array(lib->scan_to_plan, (size_t)lib->exec_stream_len);

        /* Inactive-axis gradient areas */
        {
            int has_ia = 0;
            if (fread(&has_ia, sizeof(int), 1, f) != 1)
                goto read_fail;
            if (do_swap)
                pulseg__swap4(&has_ia);
            if (has_ia)
            {
                size_t ia_total = (size_t)lib->exec_stream_len * 3;
                lib->scan_inactive_area_3ch = (float *)PULSEG_ALLOC(ia_total * sizeof(float));
                if (!lib->scan_inactive_area_3ch)
                    goto read_fail;
                if (fread(lib->scan_inactive_area_3ch, sizeof(float), ia_total, f) != ia_total)
                    goto read_fail;
                if (do_swap)
                    pulseg__swap4_array(lib->scan_inactive_area_3ch, ia_total);
                lib->scan_phase_extra =
                    (float *)PULSEG_ALLOC((size_t)lib->exec_stream_len * sizeof(float));
                if (!lib->scan_phase_extra)
                    goto read_fail;
                memset(lib->scan_phase_extra, 0, (size_t)lib->exec_stream_len * sizeof(float));
            }
        }
    }

    /* Recompute plan waveforms from 3-channel data for current shift */
    if (lib->num_plan_instances > 0 && lib->entry_waveform_3ch)
        compute_plan_waveforms(lib, shift_m);

    *out_lib = lib;
    return PULSEG_SUCCESS;

read_fail:
    if (lib)
        freq_mod_library_free(lib);
    return PULSEG_ERR_FILE_READ_FAILED;
}

/* ================================================================== */
/*  Free                                                              */
/* ================================================================== */

static void freq_mod_library_free(pulseg_freq_mod_library *lib)
{
    if (!lib)
        return;

    if (lib->entry_num_samples)
        PULSEG_FREE(lib->entry_num_samples);
    if (lib->entry_waveform_3ch)
        PULSEG_FREE(lib->entry_waveform_3ch);
    if (lib->entry_ref_3ch)
        PULSEG_FREE(lib->entry_ref_3ch);
    if (lib->rotations)
        PULSEG_FREE(lib->rotations);

    if (lib->pi_entry_idx)
        PULSEG_FREE(lib->pi_entry_idx);
    if (lib->pi_rotation_idx)
        PULSEG_FREE(lib->pi_rotation_idx);
    if (lib->plan_waveform_data)
        PULSEG_FREE(lib->plan_waveform_data);
    if (lib->plan_waveforms)
        PULSEG_FREE(lib->plan_waveforms);
    if (lib->plan_hw_data)
        PULSEG_FREE(lib->plan_hw_data);
    if (lib->plan_hw_waveforms)
        PULSEG_FREE(lib->plan_hw_waveforms);
    if (lib->plan_num_samples)
        PULSEG_FREE(lib->plan_num_samples);
    if (lib->plan_phase)
        PULSEG_FREE(lib->plan_phase);

    if (lib->scan_to_plan)
        PULSEG_FREE(lib->scan_to_plan);
    if (lib->scan_inactive_area_3ch)
        PULSEG_FREE(lib->scan_inactive_area_3ch);
    if (lib->scan_phase_extra)
        PULSEG_FREE(lib->scan_phase_extra);

    PULSEG_FREE(lib);
}

/* ================================================================== */
/*  Public: collection build                                          */
/* ================================================================== */

int pulseg_build_freq_mod_collection(
    pulseg_freq_mod_collection **out_fmc,
    const pulseg_collection *coll,
    const float *shift_m,
    const float *fov_rotation)
{
    pulseg_freq_mod_collection *fmc = NULL;
    int s, nsub;

    if (!out_fmc || !coll || !shift_m)
        return PULSEG_ERR_NULL_POINTER;
    *out_fmc = NULL;

    nsub = coll->num_subsequences;

    fmc = (pulseg_freq_mod_collection *)PULSEG_ALLOC(sizeof(*fmc));
    if (!fmc)
        return PULSEG_ERR_ALLOC_FAILED;
    memset(fmc, 0, sizeof(*fmc));

    fmc->num_subsequences = nsub;
    fmc->libs = (pulseg_freq_mod_library **)PULSEG_ALLOC((size_t)nsub * sizeof(*fmc->libs));
    if (!fmc->libs)
    {
        PULSEG_FREE(fmc);
        return PULSEG_ERR_ALLOC_FAILED;
    }
    memset(fmc->libs, 0, (size_t)nsub * sizeof(*fmc->libs));

    for (s = 0; s < nsub; ++s)
    {
        int rc = build_freq_mod_library(&fmc->libs[s], coll, s, shift_m, fov_rotation);
        if (PULSEG_FAILED(rc))
        {
            pulseg_freq_mod_collection_free(fmc);
            return rc;
        }
    }

    *out_fmc = fmc;
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Public: collection update (one subsequence)                       */
/* ================================================================== */

static int pulseg_update_freq_mod_collection(
    pulseg_freq_mod_collection *fmc,
    int subseq_idx,
    const float *shift_m)
{
    if (!fmc || !shift_m)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= fmc->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;
    return update_freq_mod_library(fmc->libs[subseq_idx], shift_m);
}

/* ================================================================== */
/*  Public: collection accessor                                       */
/* ================================================================== */

int pulseg_freq_mod_collection_get(
    const pulseg_freq_mod_collection *fmc,
    const short **out_hw_waveform,
    int *out_num_samples,
    float *out_phase_rad,
    int subseq_idx,
    int exec_stream_pos)
{
    if (!fmc || subseq_idx < 0 || subseq_idx >= fmc->num_subsequences)
        return 0;
    return freq_mod_library_get(
        fmc->libs[subseq_idx],
        out_hw_waveform,
        out_num_samples,
        out_phase_rad,
        exec_stream_pos);
}

/* ================================================================== */
/*  Public: collection cache write (FILE* variant)                    */
/* ================================================================== */

int pulseg__freq_mod_collection_write_f(const pulseg_freq_mod_collection *fmc, FILE *f)
{
    int s;

    if (!fmc || !f)
        return PULSEG_ERR_NULL_POINTER;

    if (fwrite(&fmc->num_subsequences, sizeof(int), 1, f) != 1)
        return PULSEG_ERR_FILE_READ_FAILED;

    for (s = 0; s < fmc->num_subsequences; ++s)
    {
        int rc = freq_mod_library_write_cache(fmc->libs[s], f);
        if (PULSEG_FAILED(rc))
            return rc;
    }

    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Public: collection cache write (path-based wrapper)               */
/* ================================================================== */

int pulseg__freq_mod_collection_read_f(
    pulseg_freq_mod_collection **out_fmc,
    FILE *f,
    const pulseg_collection *coll,
    const float *shift_m,
    int do_swap)
{
    pulseg_freq_mod_collection *fmc = NULL;
    int nsub, s;

    if (!out_fmc || !f || !coll || !shift_m)
        return PULSEG_ERR_NULL_POINTER;
    *out_fmc = NULL;

    if (fread(&nsub, sizeof(int), 1, f) != 1)
        return PULSEG_ERR_FILE_READ_FAILED;
    if (do_swap)
        pulseg__swap4(&nsub);
    if (nsub != coll->num_subsequences)
        return PULSEG_ERR_FILE_READ_FAILED;

    fmc = (pulseg_freq_mod_collection *)PULSEG_ALLOC(sizeof(*fmc));
    if (!fmc)
        return PULSEG_ERR_ALLOC_FAILED;
    memset(fmc, 0, sizeof(*fmc));

    fmc->num_subsequences = nsub;
    fmc->libs = (pulseg_freq_mod_library **)PULSEG_ALLOC((size_t)nsub * sizeof(*fmc->libs));
    if (!fmc->libs)
    {
        PULSEG_FREE(fmc);
        return PULSEG_ERR_ALLOC_FAILED;
    }
    memset(fmc->libs, 0, (size_t)nsub * sizeof(*fmc->libs));

    for (s = 0; s < nsub; ++s)
    {
        int rc = freq_mod_library_read_cache(&fmc->libs[s], f, shift_m, do_swap);
        if (PULSEG_FAILED(rc))
        {
            pulseg_freq_mod_collection_free(fmc);
            return rc;
        }
    }

    *out_fmc = fmc;
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Public: collection cache read (path-based wrapper)                */
/* ================================================================== */

/* ================================================================== */
/*  Public: collection free                                           */
/* ================================================================== */

void pulseg_freq_mod_collection_free(pulseg_freq_mod_collection *fmc)
{
    if (!fmc)
        return;
    if (fmc->libs)
    {
        int s;
        for (s = 0; s < fmc->num_subsequences; ++s)
            if (fmc->libs[s])
                freq_mod_library_free(fmc->libs[s]);
        PULSEG_FREE(fmc->libs);
    }
    PULSEG_FREE(fmc);
}

/* ================================================================== */
/*  Collection-level wrappers (operate on coll->freq_mod)             */
/* ================================================================== */

static int pulseg_build_freq_mod(
    pulseg_collection *coll,
    const float *shift_m,
    const float *fov_rotation)
{
    if (!coll)
        return PULSEG_ERR_NULL_POINTER;
    if (coll->freq_mod)
    {
        pulseg_freq_mod_collection_free(coll->freq_mod);
        coll->freq_mod = NULL;
    }
    return pulseg_build_freq_mod_collection(&coll->freq_mod, coll, shift_m, fov_rotation);
}

int pulseg__save_freq_mod_cache_section(pulseg_collection *coll, const char *seq_path)
{
    float zero_shift[3];
    float identity[9];
    int rc;

    if (!coll || !seq_path)
        return PULSEG_ERR_NULL_POINTER;

    zero_shift[0] = 0.0f;
    zero_shift[1] = 0.0f;
    zero_shift[2] = 0.0f;
    identity[0] = 1.0f;
    identity[1] = 0.0f;
    identity[2] = 0.0f;
    identity[3] = 0.0f;
    identity[4] = 1.0f;
    identity[5] = 0.0f;
    identity[6] = 0.0f;
    identity[7] = 0.0f;
    identity[8] = 1.0f;

    /* Only the shift-independent base is cached, so build at zero shift /
     * identity rotation; scan rematerializes the shift-dependent plans. */
    rc = pulseg_build_freq_mod(coll, zero_shift, identity);
    if (PULSEG_FAILED(rc))
        return rc;

    return pulseg__save_freq_mod_section(coll, seq_path);
}

int pulseg_update_freq_mod(pulseg_collection *coll, int subseq_idx, const float *shift_m)
{
    if (!coll || !coll->freq_mod)
        return PULSEG_ERR_NULL_POINTER;
    return pulseg_update_freq_mod_collection(coll->freq_mod, subseq_idx, shift_m);
}

int pulseg_get_freq_mod(
    const pulseg_collection *coll,
    const short **out_hw_waveform,
    int *out_num_samples,
    float *out_phase_rad,
    int subseq_idx,
    int exec_stream_pos)
{
    if (!coll || !coll->freq_mod)
        return 0;
    return pulseg_freq_mod_collection_get(
        coll->freq_mod,
        out_hw_waveform,
        out_num_samples,
        out_phase_rad,
        subseq_idx,
        exec_stream_pos);
}

/* ================================================================== */
/*  Per-TR targeted update (PMC mode)                                 */
/*                                                                     */
/*  Recomputes plan waveforms + phase ONLY for plan instances that    */
/*  are referenced by scan-table positions in a given TR range.       */
/*  Cost: O(plans_in_TR × max_samples) instead of full library.      */
/* ================================================================== */

int pulseg_update_freq_mod_for_tr(
    pulseg_collection *coll,
    int subseq_idx,
    int current_scan_pos,
    const float *shift_m)
{
    pulseg_freq_mod_library *lib;
    int tr_scan_start, tr_scan_count;
    int pos, pi, p, s, ch, ns;
    int *dirty; /* boolean flag per plan instance */
    float identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    if (!coll || !coll->freq_mod || !shift_m)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->freq_mod->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;

    /* Derive TR scan-table range from current position + TR size */
    {
        int tr_size = coll->descriptors[subseq_idx].tr_descriptor.tr_size;
        if (tr_size <= 0)
            return PULSEG_ERR_INVALID_ARGUMENT;
        tr_scan_start = (current_scan_pos / tr_size) * tr_size;
        tr_scan_count = tr_size;
    }

    lib = coll->freq_mod->libs[subseq_idx];
    if (!lib)
        return PULSEG_ERR_NULL_POINTER;
    if (!lib->entry_waveform_3ch || !lib->entry_ref_3ch)
        return PULSEG_ERR_INVALID_ARGUMENT; /* 3ch data not retained */
    if (lib->num_plan_instances == 0)
        return PULSEG_SUCCESS;

    /* Mark which plan instances are used by the current TR */
    dirty = (int *)PULSEG_ALLOC((size_t)lib->num_plan_instances * sizeof(int));
    if (!dirty)
        return PULSEG_ERR_ALLOC_FAILED;
    for (p = 0; p < lib->num_plan_instances; ++p)
        dirty[p] = 0;

    for (pos = tr_scan_start; pos < tr_scan_start + tr_scan_count && pos < lib->exec_stream_len;
         ++pos)
    {
        if (pos < 0)
            continue;
        pi = lib->scan_to_plan[pos];
        if (pi >= 0 && pi < lib->num_plan_instances)
            dirty[pi] = 1;
    }

    /* Recompute only dirty plan waveforms */
    for (p = 0; p < lib->num_plan_instances; ++p)
    {
        int eidx, ridx;
        float *row;
        const float *R;
        float u[3];

        if (!dirty[p])
            continue;

        eidx = lib->pi_entry_idx[p];
        ridx = lib->pi_rotation_idx[p];
        ns = lib->plan_num_samples[p];
        row = lib->plan_waveforms[p];

        R = (ridx >= 0 && ridx < lib->num_rotations) ? (const float *)lib->rotations[ridx]
                                                     : identity;
        pulseg__apply_rotation(u, R, shift_m, 1);

        for (s = 0; s < ns; ++s)
        {
            float val = 0.0f;
            for (ch = 0; ch < 3; ++ch)
                val +=
                    lib->entry_waveform_3ch
                        [(size_t)eidx * lib->max_samples * 3 + (size_t)ch * lib->max_samples + s] *
                    u[ch];
            row[s] = val;
        }
        for (s = ns; s < lib->max_samples; ++s)
            row[s] = 0.0f;

        lib->plan_phase[p] = 0.0f;
        for (ch = 0; ch < 3; ++ch)
            lib->plan_phase[p] += lib->entry_ref_3ch[eidx * 3 + ch] * u[ch];

        /* Convert this dirty plan instance to hw format */
        {
            short *hw = lib->plan_hw_waveforms[p];
            for (s = 0; s < ns; ++s)
            {
                int v = (int)(row[s] * PULSEG_OMEGA_SCALE);
                if (v > 32767)
                    v = 32767;
                if (v < -32767)
                    v = -32767;
                hw[s] = (short)(v & ~PULSEG_HW_WAVEFORM_END);
            }
            if (ns > 0)
                hw[ns - 1] |= PULSEG_HW_WAVEFORM_END;
            for (s = ns; s < lib->max_samples; ++s)
                hw[s] = 0;
        }
    }

    /* Recompute scan_phase_extra only for the TR range */
    if (lib->scan_phase_extra && lib->scan_inactive_area_3ch)
    {
        for (pos = tr_scan_start; pos < tr_scan_start + tr_scan_count && pos < lib->exec_stream_len;
             ++pos)
        {
            int ridx = -1;
            const float *R;
            float u[3];
            const float *ia;
            if (pos < 0)
                continue;
            pi = lib->scan_to_plan[pos];
            if (pi >= 0 && pi < lib->num_plan_instances)
                ridx = lib->pi_rotation_idx[pi];
            R = (ridx >= 0 && ridx < lib->num_rotations) ? (const float *)lib->rotations[ridx]
                                                         : identity;
            pulseg__apply_rotation(u, R, shift_m, 1);
            ia = lib->scan_inactive_area_3ch + (size_t)pos * 3;
            lib->scan_phase_extra[pos] = ia[0] * u[0] + ia[1] * u[1] + ia[2] * u[2];
        }
    }

    PULSEG_FREE(dirty);
    return PULSEG_SUCCESS;
}
