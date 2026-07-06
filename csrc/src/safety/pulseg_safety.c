
#include "pulseg_internal.h"

/* Helper: select the canonical TR window for a given canonical_tr_idx */
static void pulseg__select_canonical_tr_window_idx(
    const struct pulseg_sequence_descriptor *desc,
    int canonical_tr_idx,
    int *start_block,
    int *block_count,
    int *amplitude_mode,
    int *num_instances,
    float *tr_duration_us)
{
    const struct pulseg_tr_descriptor *trd = &desc->tr_descriptor;
    int has_nd_prep = (trd->num_prep_blocks > 0 && !trd->degenerate_prep);
    int has_nd_cool = (trd->num_cooldown_blocks > 0 && !trd->degenerate_cooldown);

    if (has_nd_prep || has_nd_cool)
    {
        /* Non-degenerate: canonical_tr_idx selects pass.
         * Pass-expanded waveform already bakes in num_averages via
         * _build_pass_expanded_block_order; num_instances counts how
         * many times this pass-expanded waveform repeats. */
        int pass_len = desc->pass_len;
        *start_block = canonical_tr_idx * pass_len;
        *block_count = pass_len;
        *amplitude_mode = PULSEG_AMP_MAX_POS;
        *num_instances = (desc->num_passes > 1) ? desc->num_passes : 1;
        *tr_duration_us = 0.0f;
        {
            int n;
            for (n = 0; n < pass_len; ++n)
            {
                int idx;
                const struct pulseg_block_table_element *bte;
                const struct pulseg_block_definition *bdef;
                idx = *start_block + n;
                bte = &desc->block_table[idx];
                bdef = &desc->block_definitions[bte->id];
                *tr_duration_us += (bte->duration_us >= 0)
                                       ? (float)bte->duration_us
                                       : (float)bdef->duration_us;
            }
        }
        return;
    }

    /* Degenerate: canonical_tr_idx selects imaging TR.
     * No pass expansion occurs, so averages must be accounted for here. */
    {
        int num_avgs = (desc->num_averages > 1) ? desc->num_averages : 1;
        *start_block = trd->num_prep_blocks + trd->imaging_tr_start + canonical_tr_idx * trd->tr_size;
        *block_count = trd->tr_size;
        *amplitude_mode = PULSEG_AMP_MAX_POS;
        *num_instances = trd->num_trs * num_avgs;
        *tr_duration_us = trd->tr_duration_us;
    }
}

static int pulseg__build_pass_expanded_block_order(
    const struct pulseg_sequence_descriptor *desc,
    int pass_base,
    int **out_block_order,
    int *out_block_count,
    float *out_duration_us)
{
    const struct pulseg_tr_descriptor *trd;
    int prep_blk, img_len, cool_blk, num_avgs, exp_count;
    int *block_order;
    int avg_i, pos_i, n;

    if (!desc || !out_block_order || !out_block_count)
    {
        return PULSEG_ERR_INVALID_ARGUMENT;
    }

    trd = &desc->tr_descriptor;
    prep_blk = trd->num_prep_blocks;
    img_len = trd->num_trs * trd->tr_size;
    cool_blk = trd->num_cooldown_blocks;
    num_avgs = (desc->num_averages > 0) ? desc->num_averages : 1;
    exp_count = prep_blk + num_avgs * img_len + cool_blk;

    if (exp_count <= 0)
    {
        return PULSEG_ERR_TR_NO_BLOCKS;
    }

    block_order = (int *)PULSEG_ALLOC((size_t)exp_count * sizeof(int));
    if (!block_order)
    {
        return PULSEG_ERR_ALLOC_FAILED;
    }

    n = 0;
    for (pos_i = 0; pos_i < prep_blk; ++pos_i)
        block_order[n++] = pass_base + pos_i;
    for (avg_i = 0; avg_i < num_avgs; ++avg_i)
        for (pos_i = 0; pos_i < img_len; ++pos_i)
            block_order[n++] = pass_base + prep_blk + pos_i;
    for (pos_i = 0; pos_i < cool_blk; ++pos_i)
        block_order[n++] = pass_base + prep_blk + img_len + pos_i;

    if (out_duration_us)
    {
        *out_duration_us = 0.0f;
        for (pos_i = 0; pos_i < exp_count; ++pos_i)
        {
            int blk_idx;
            const struct pulseg_block_table_element *bte;
            const struct pulseg_block_definition *bdef;

            blk_idx = block_order[pos_i];
            if (blk_idx < 0 || blk_idx >= desc->num_blocks)
            {
                PULSEG_FREE(block_order);
                return PULSEG_ERR_INVALID_ARGUMENT;
            }
            bte = &desc->block_table[blk_idx];
            bdef = &desc->block_definitions[bte->id];
            *out_duration_us += (bte->duration_us >= 0)
                                    ? (float)bte->duration_us
                                    : (float)bdef->duration_us;
        }
    }

    *out_block_order = block_order;
    *out_block_count = exp_count;
    return PULSEG_SUCCESS;
}

/* pulseg_safety.c -- safety checks, mechanical resonance analysis, PNS
 *
 * Public functions:
 *   pulseg_check_safety
 *   pulseg_calc_mech_resonances   / _free
 *   pulseg_calc_pns               / _free
 *
 * Internal:
 *   check_max_grad / check_grad_continuity / check_max_slew
 *
 * (pulseg__calc_segment_timing -- parse-time timing derivation -- now
 *  lives in pulseg_structure.c.)
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "pulseg_internal.h"
#include "pulseg.h"

#include "external_kiss_fft.h"
#include "external_kiss_fftr.h"

/* ================================================================== */
/*  File-scope constants                                               */
/* ================================================================== */
#define PNS_KERNEL_DURATION_FACTOR 20.0f


/* ================================================================== */
/*  Acoustic spectra free                                             */
/* ================================================================== */

void pulseg_mech_resonances_spectra_free(pulseg_mech_resonances_spectra *s)
{
    if (!s)
        return;

    if (s->spectrum_full_gx)
        PULSEG_FREE(s->spectrum_full_gx);
    if (s->spectrum_full_gy)
        PULSEG_FREE(s->spectrum_full_gy);
    if (s->spectrum_full_gz)
        PULSEG_FREE(s->spectrum_full_gz);

    if (s->analytical_peak_freqs)
        PULSEG_FREE(s->analytical_peak_freqs);
    if (s->analytical_peak_amp_gx)
        PULSEG_FREE(s->analytical_peak_amp_gx);
    if (s->analytical_peak_amp_gy)
        PULSEG_FREE(s->analytical_peak_amp_gy);
    if (s->analytical_peak_amp_gz)
        PULSEG_FREE(s->analytical_peak_amp_gz);
    if (s->analytical_peak_phase_gx)
        PULSEG_FREE(s->analytical_peak_phase_gx);
    if (s->analytical_peak_phase_gy)
        PULSEG_FREE(s->analytical_peak_phase_gy);
    if (s->analytical_peak_phase_gz)
        PULSEG_FREE(s->analytical_peak_phase_gz);
    if (s->analytical_peak_widths_hz)
        PULSEG_FREE(s->analytical_peak_widths_hz);
    if (s->candidate_freqs)
        PULSEG_FREE(s->candidate_freqs);
    if (s->candidate_amps_gx)
        PULSEG_FREE(s->candidate_amps_gx);
    if (s->candidate_amps_gy)
        PULSEG_FREE(s->candidate_amps_gy);
    if (s->candidate_amps_gz)
        PULSEG_FREE(s->candidate_amps_gz);
    if (s->candidate_grad_amps)
        PULSEG_FREE(s->candidate_grad_amps);
    if (s->candidate_grad_amps_gx)
        PULSEG_FREE(s->candidate_grad_amps_gx);
    if (s->candidate_grad_amps_gy)
        PULSEG_FREE(s->candidate_grad_amps_gy);
    if (s->candidate_grad_amps_gz)
        PULSEG_FREE(s->candidate_grad_amps_gz);
    if (s->candidate_violations)
        PULSEG_FREE(s->candidate_violations);
    if (s->component_freqs_hz)
        PULSEG_FREE(s->component_freqs_hz);
    if (s->component_amps)
        PULSEG_FREE(s->component_amps);
    if (s->component_phases_rad)
        PULSEG_FREE(s->component_phases_rad);
    if (s->component_widths_hz)
        PULSEG_FREE(s->component_widths_hz);
    if (s->component_axes)
        PULSEG_FREE(s->component_axes);
    if (s->component_def_ids)
        PULSEG_FREE(s->component_def_ids);
    if (s->component_contrib_ids)
        PULSEG_FREE(s->component_contrib_ids);
    if (s->component_run_ids)
        PULSEG_FREE(s->component_run_ids);
    if (s->surviving_freqs_hz)
        PULSEG_FREE(s->surviving_freqs_hz);

    memset(s, 0, sizeof(*s));
}

/* ================================================================== */
/*  Structural acoustic analysis — data types                         */
/* ================================================================== */

/** Max vertices in piecewise-linear waveform envelope. */
#define SA_MAX_PWL_VERTICES 16

/** Minimum sub-period repetitions for decomposition. */
#define SA_MIN_SUB_PERIOD_REPS 3

/** Autocorrelation peak threshold (normalised 0..1). */
#define SA_AUTOCORR_THRESHOLD 0.5f

/** Coherence ratio threshold for Tier 1 candidate detection. */
#define SA_COHERENCE_RATIO_THRESHOLD 2.0f

/** Absolute amplitude floor for Tier 1 (Hz/m). */
#define SA_AMPLITUDE_FLOOR 1.0f

/* NOTE: FFT power gate was removed from structural candidate selection.
 * The analytical coherence-ratio and amplitude-floor tiers provide
 * principled stationary-limit criteria.  The dense FFT is computed on the
 * finite expanded waveform, so gating structural candidates against it
 * introduced a spurious dependence on num_averages (longer waveform →
 * narrower FFT lobes → candidates rejected even though the infinite-
 * repetition regime excites the same resonances).
 *
 * Replaced by an analytical power gate: candidates whose cross-axis RSS²
 * of the coherent inner-TR magnitude is below SA_ANALYTICAL_GATE_FRAC of
 * the peak RSS² are rejected.  This is computed entirely from the event
 * model and does not depend on num_averages. */

/** Analytical power gate: reject TR harmonics where the cross-axis
 *  RSS² of |H(f)| is below this fraction of the peak RSS².  Same
 *  concept as the former FFT gate, but derived from the analytical
 *  inner-TR spectrum so it is independent of waveform length. */
#define SA_ANALYTICAL_GATE_FRAC 0.05f

/** FFT peak promotion: a dense-FFT local maximum must exceed this fraction
 *  of the peak RSS power to be promoted into the evaluation set when it
 *  falls between TR harmonics.  Higher than the gate to avoid adding many
 *  minor spectral features as candidates. */
#define SA_FFT_PROMOTION_POWER_FRAC 0.15f

/** Absolute amplitude safety threshold for single-event Tier 2 (Hz/m).
 * Only applies when an axis has exactly one event (coherence ratio is
 * undefined for N=1).  Value is in the same units as the coherent
 * magnitude — roughly amp_Hz_per_m × h(f) × N_events. */
#define SA_AMPLITUDE_SAFETY 1.0e6f

/**
 * A gradient event within the canonical TR.
 * Each event is a single waveform (or sub-event from decomposed arbitrary)
 * played at a specific time with a specific amplitude.
 */
typedef struct
{
    int def_id;                              /**< gradient definition id                */
    int def_index;                           /**< index into grad_definitions[]         */
    float start_time_us;                     /**< event start time within the TR (us)   */
    float amplitude;                         /**< worst-case positional amplitude (Hz/m), signed */
    int pwl_num_vertices;                    /**< >0 -> use piecewise-linear W(f)       */
    float pwl_times_us[SA_MAX_PWL_VERTICES]; /**< vertex times (us from event start) */
    float pwl_values[SA_MAX_PWL_VERTICES];   /**< vertex amplitudes (normalised)     */
    int num_reps;                            /**< repetition count (1=once, N=imaging repeat) */
    float rep_period_us;                     /**< repetition period in us (0 if num_reps==1) */
    /* FFT-based response model (for many-sample arb without sub-period) */
    float *fft_magnitudes;     /**< normalised |F[k]|/|F[0]|, k=0..fft_num_bins-1 */
    int fft_num_bins;          /**< nfft/2+1 frequency bins (0 = not used)        */
    float fft_freq_spacing_hz; /**< 1 / (nfft * raster_s)                         */
} sa_event;

/** Per-axis event list. */
typedef struct
{
    int num_events;
    sa_event *events; /**< allocated [num_events] */
} sa_axis_events;

/** Structural analysis: event lists for all three axes. */
typedef struct
{
    sa_axis_events axes[3]; /**< 0=gx, 1=gy, 2=gz */
} sa_structural_events;

static void sa_free_structural_events(sa_structural_events *se)
{
    int ax, k;
    if (!se)
        return;
    for (ax = 0; ax < 3; ++ax)
    {
        if (se->axes[ax].events)
        {
            for (k = 0; k < se->axes[ax].num_events; ++k)
            {
                if (se->axes[ax].events[k].fft_magnitudes)
                    PULSEG_FREE(se->axes[ax].events[k].fft_magnitudes);
            }
            PULSEG_FREE(se->axes[ax].events);
        }
        se->axes[ax].events = NULL;
        se->axes[ax].num_events = 0;
    }
}

/* ================================================================== */
/*  Structural acoustic analysis — occurrence extraction              */
/* ================================================================== */

/**
 * Extract grad-def occurrences for one axis within the canonical TR.
 * Collects (def_id, def_index, start_time_us, amplitude) for each block
 * that has a gradient event on the given axis.
 * @param axis  0=gx, 1=gy, 2=gz
 * Returns number of occurrences, or -1 on allocation failure.
 * Caller must free *out_events.
 */
typedef struct
{
    int def_id;
    int def_index;
    float start_time_us;
    float amplitude;
} sa_raw_occurrence;

static int sa_extract_raw_occurrences(
    sa_raw_occurrence **out_occ,
    const struct pulseg_sequence_descriptor *desc,
    int start_block, int block_count, int axis)
{
    int i, n, cap;
    float time_us;
    sa_raw_occurrence *occ;

    cap = (block_count > 0) ? block_count : 16;
    occ = (sa_raw_occurrence *)PULSEG_ALLOC((size_t)cap * sizeof(sa_raw_occurrence));
    if (!occ)
        return -1;
    n = 0;
    time_us = 0.0f;

    for (i = 0; i < block_count; ++i)
    {
        const struct pulseg_block_table_element *bte =
            &desc->block_table[start_block + i];
        const struct pulseg_block_definition *bdef =
            &desc->block_definitions[bte->id];
        int grad_table_idx;
        float blk_dur;

        switch (axis)
        {
        case 0:
            grad_table_idx = bte->gx_id;
            break;
        case 1:
            grad_table_idx = bte->gy_id;
            break;
        case 2:
            grad_table_idx = bte->gz_id;
            break;
        default:
            grad_table_idx = -1;
            break;
        }

        if (grad_table_idx >= 0)
        {
            const struct pulseg_grad_table_element *gte =
                &desc->grad_table[grad_table_idx];
            if (n >= cap)
            {
                sa_raw_occurrence *tmp;
                cap *= 2;
                tmp = (sa_raw_occurrence *)PULSEG_ALLOC((size_t)cap * sizeof(sa_raw_occurrence));
                if (!tmp)
                {
                    PULSEG_FREE(occ);
                    return -1;
                }
                memcpy(tmp, occ, (size_t)n * sizeof(sa_raw_occurrence));
                PULSEG_FREE(occ);
                occ = tmp;
            }
            occ[n].def_id = gte->id;
            occ[n].def_index = gte->id;
            occ[n].start_time_us = time_us;
            occ[n].amplitude = gte->amplitude;
            ++n;
        }

        blk_dur = (bte->duration_us >= 0) ? (float)bte->duration_us
                                          : (float)bdef->duration_us;
        time_us += blk_dur;
    }

    *out_occ = occ;
    return n;
}

/* ================================================================== */
/*  Structural acoustic analysis — sub-period detection               */
/* ================================================================== */

/**
 * Detect dominant sub-period in an arbitrary waveform via autocorrelation
 * of the rectified signal.  Returns detected period in samples, or 0 if
 * no periodicity found.
 *
 * @param wave      Decompressed waveform samples.
 * @param n         Number of samples.
 * @param out_period  Detected period in samples (0 if none).
 * @param out_reps    Number of complete repetitions.
 * @return 1 if sub-period found, 0 otherwise.
 */
static int sa_detect_sub_period(
    const float *wave, int n,
    int *out_period, int *out_reps)
{
    int tau, max_lag, best_tau, reps, k;
    float mean_val, sum_r0, sum_rt, r_norm, best_r;
    float *buf;

    *out_period = 0;
    *out_reps = 0;

    if (n < 6)
        return 0; /* Too short for periodicity */

    buf = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    if (!buf)
        return 0;

    /* Remove mean (no rectification — preserve sign for bipolar detection) */
    mean_val = 0.0f;
    for (k = 0; k < n; ++k)
        mean_val += wave[k];
    mean_val /= (float)n;
    for (k = 0; k < n; ++k)
        buf[k] = wave[k] - mean_val;

    /* Compute R[0] for normalisation */
    sum_r0 = 0.0f;
    for (k = 0; k < n; ++k)
        sum_r0 += buf[k] * buf[k];
    if (sum_r0 < 1.0e-20f)
    {
        PULSEG_FREE(buf);
        return 0;
    }

    /* Search for first significant peak in normalised autocorrelation.
     * Skip lag 0 (trivially 1.0).  Start from lag 2 to avoid edge effects. */
    max_lag = n / 2;
    best_tau = 0;
    best_r = SA_AUTOCORR_THRESHOLD; /* Minimum acceptable peak height */

    /* First, find the first valley (autocorrelation must dip below threshold
     * before we accept a peak, to avoid the initial decay region). */
    {
        int found_valley = 0;
        float prev_r = 1.0f;
        for (tau = 2; tau < max_lag; ++tau)
        {
            sum_rt = 0.0f;
            for (k = 0; k < n - tau; ++k)
                sum_rt += buf[k] * buf[k + tau];
            r_norm = sum_rt / sum_r0;
            if (r_norm < SA_AUTOCORR_THRESHOLD * 0.5f)
            {
                found_valley = 1;
            }
            if (found_valley && r_norm > best_r &&
                r_norm > prev_r)
            {
                /* Check it's a local max: look one step ahead */
                float r_next = 0.0f;
                int tau_next = tau + 1;
                if (tau_next < max_lag)
                {
                    int kk;
                    for (kk = 0; kk < n - tau_next; ++kk)
                        r_next += buf[kk] * buf[kk + tau_next];
                    r_next /= sum_r0;
                }
                if (r_norm >= r_next)
                {
                    best_tau = tau;
                    best_r = r_norm;
                    break; /* First significant peak after valley */
                }
            }
            prev_r = r_norm;
        }
    }

    PULSEG_FREE(buf);

    if (best_tau < 2)
        return 0;

    /* Validate: check that R[2*tau] is also a peak (periodic confirmation) */
    {
        float r2;
        int tau2 = 2 * best_tau;
        if (tau2 < max_lag)
        {
            float *buf2 = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
            if (buf2)
            {
                for (k = 0; k < n; ++k)
                    buf2[k] = wave[k] - mean_val;
                sum_rt = 0.0f;
                for (k = 0; k < n - tau2; ++k)
                    sum_rt += buf2[k] * buf2[k + tau2];
                r2 = sum_rt / sum_r0;
                PULSEG_FREE(buf2);
                if (r2 < SA_AUTOCORR_THRESHOLD * 0.5f)
                    return 0; /* Second harmonic too weak — not periodic */
            }
        }
    }

    reps = n / best_tau;
    if (reps < SA_MIN_SUB_PERIOD_REPS)
        return 0;

    *out_period = best_tau;
    *out_reps = reps;
    return 1;
}

/* ================================================================== */
/*  Structural acoustic analysis — FFT-based response model           */
/* ================================================================== */

/**
 * Compute the normalised FFT magnitude spectrum of a waveform.
 *
 * 1. If wave_time_us is non-NULL (non-uniform sampling), interpolate
 *    onto a uniform grid at the given raster_us spacing.
 * 2. Zero-pad to next power of 2.
 * 3. Compute real FFT via kiss_fftr.
 * 4. Normalise magnitudes by |F[0]| (DC).  If DC ≈ 0, normalise by peak.
 *
 * @param wave_vals      Waveform sample values (normalised shape, not physical amplitude).
 * @param wave_time_us   Per-sample times in µs (may be NULL for uniform raster).
 * @param num_samp       Number of samples.
 * @param raster_us      Gradient raster in µs (used for uniform grid).
 * @param out_mags       Receives allocated normalised magnitude array [nfft/2+1].
 * @param out_num_bins   Receives nfft/2+1.
 * @param out_freq_spacing_hz  Receives 1/(nfft*raster_s).
 * @return 1 on success, 0 on failure (allocation or FFT).
 */
static int sa_compute_fft_response(
    const float *wave_vals,
    const float *wave_time_us,
    int num_samp,
    float raster_us,
    float **out_mags,
    int *out_num_bins,
    float *out_freq_spacing_hz)
{
    int nfft, nfreq, i, n_uniform;
    float *uniform_vals = NULL;
    float *pad = NULL;
    kiss_fft_cpx *spectrum = NULL;
    kiss_fftr_cfg cfg = NULL;
    float *mags = NULL;
    float norm_val;

    *out_mags = NULL;
    *out_num_bins = 0;
    *out_freq_spacing_hz = 0.0f;

    if (num_samp < 2 || raster_us <= 0.0f)
        return 0;

    /* Step 1: interpolate to uniform raster if time shape is present */
    if (wave_time_us)
    {
        float duration_us = wave_time_us[num_samp - 1] - wave_time_us[0];
        float *uniform_t;
        if (duration_us <= 0.0f)
            return 0;
        n_uniform = (int)(duration_us / raster_us) + 1;
        if (n_uniform < 2)
            return 0;
        uniform_t = (float *)PULSEG_ALLOC((size_t)n_uniform * sizeof(float));
        uniform_vals = (float *)PULSEG_ALLOC((size_t)n_uniform * sizeof(float));
        if (!uniform_t || !uniform_vals)
        {
            if (uniform_t)
                PULSEG_FREE(uniform_t);
            if (uniform_vals)
                PULSEG_FREE(uniform_vals);
            return 0;
        }
        for (i = 0; i < n_uniform; ++i)
            uniform_t[i] = wave_time_us[0] + (float)i * raster_us;
        pulseg__interp1_linear(uniform_vals, uniform_t, n_uniform,
                                  wave_time_us, wave_vals, num_samp);
        PULSEG_FREE(uniform_t);
    }
    else
    {
        n_uniform = num_samp;
        uniform_vals = (float *)PULSEG_ALLOC((size_t)n_uniform * sizeof(float));
        if (!uniform_vals)
            return 0;
        memcpy(uniform_vals, wave_vals, (size_t)n_uniform * sizeof(float));
    }

    /* Step 2: zero-pad to next power of 2 */
    nfft = (int)pulseg__next_pow2((size_t)n_uniform);
    if (nfft < n_uniform)
        nfft = n_uniform;
    nfreq = nfft / 2 + 1;

    pad = (float *)PULSEG_ALLOC((size_t)nfft * sizeof(float));
    spectrum = (kiss_fft_cpx *)PULSEG_ALLOC((size_t)nfreq * sizeof(kiss_fft_cpx));
    mags = (float *)PULSEG_ALLOC((size_t)nfreq * sizeof(float));
    if (!pad || !spectrum || !mags)
        goto fail;

    memcpy(pad, uniform_vals, (size_t)n_uniform * sizeof(float));
    for (i = n_uniform; i < nfft; ++i)
        pad[i] = 0.0f;

    /* Step 3: FFT */
    cfg = kiss_fftr_alloc(nfft, 0, NULL, NULL);
    if (!cfg)
        goto fail;
    kiss_fftr(cfg, pad, spectrum);

    /* Step 4: compute magnitudes and normalise */
    for (i = 0; i < nfreq; ++i)
    {
        float r = spectrum[i].r;
        float im = spectrum[i].i;
        mags[i] = (float)sqrt((double)(r * r + im * im));
    }

    /* Normalise by DC; fall back to peak if DC ≈ 0 */
    norm_val = mags[0];
    if (norm_val < 1.0e-20f)
    {
        norm_val = 0.0f;
        for (i = 0; i < nfreq; ++i)
            if (mags[i] > norm_val)
                norm_val = mags[i];
    }
    if (norm_val > 1.0e-20f)
    {
        for (i = 0; i < nfreq; ++i)
            mags[i] /= norm_val;
    }
    else
    {
        /* Essentially zero waveform */
        for (i = 0; i < nfreq; ++i)
            mags[i] = 1.0f;
    }

    *out_mags = mags;
    mags = NULL;
    *out_num_bins = nfreq;
    *out_freq_spacing_hz = 1.0e6f / ((float)nfft * raster_us);

    PULSEG_FREE(uniform_vals);
    PULSEG_FREE(pad);
    PULSEG_FREE(spectrum);
    kiss_fftr_free(cfg);
    return 1;

fail:
    if (uniform_vals)
        PULSEG_FREE(uniform_vals);
    if (pad)
        PULSEG_FREE(pad);
    if (spectrum)
        PULSEG_FREE(spectrum);
    if (mags)
        PULSEG_FREE(mags);
    if (cfg)
        kiss_fftr_free(cfg);
    return 0;
}

/* ================================================================== */
/*  Structural acoustic analysis — build events per axis              */
/* ================================================================== */

static int sa_compare_int(const void *a, const void *b)
{
    return (*(const int *)a) - (*(const int *)b);
}

/**
 * Build event list for one axis from raw occurrences.
 *
 * For each unique def_id:
 *   - Trapezoid: one event per occurrence, PWL from rise/flat/fall.
 *   - Arbitrary with few samples: one event per occurrence, PWL approximation.
 *   - Arbitrary with many samples + detected sub-period: M sub-events per
 *     occurrence, each with PWL of one period.
 *   - Arbitrary with many samples + no sub-period: one event per occurrence,
 *     PWL approximation of full waveform (keeping first/last + zero crossings).
 */
static int sa_build_axis_events(
    sa_axis_events *ae,
    const sa_raw_occurrence *occ, int num_occ,
    const struct pulseg_sequence_descriptor *desc)
{
    int i, j, n_events, cap, did, idx, s_idx, nv;
    int *unique_ids;
    int num_unique;
    float raster;

    ae->num_events = 0;
    ae->events = NULL;
    if (num_occ <= 0)
        return PULSEG_SUCCESS;

    raster = desc->grad_raster_us;

    /* Collect unique def_ids */
    unique_ids = (int *)PULSEG_ALLOC((size_t)num_occ * sizeof(int));
    if (!unique_ids)
        return PULSEG_ERR_ALLOC_FAILED;
    for (i = 0; i < num_occ; ++i)
        unique_ids[i] = occ[i].def_id;
    qsort(unique_ids, (size_t)num_occ, sizeof(int), sa_compare_int);
    num_unique = 1;
    for (i = 1; i < num_occ; ++i)
        if (unique_ids[i] != unique_ids[i - 1])
            unique_ids[num_unique++] = unique_ids[i];

    /* Initial allocation — may grow for decomposed arbitraries */
    cap = num_occ * 2;
    ae->events = (sa_event *)PULSEG_ALLOC((size_t)cap * sizeof(sa_event));
    if (!ae->events)
    {
        PULSEG_FREE(unique_ids);
        return PULSEG_ERR_ALLOC_FAILED;
    }
    n_events = 0;

    for (idx = 0; idx < num_unique; ++idx)
    {
        const struct pulseg_grad_definition *gdef;
        /* Shared PWL template and sub-event decomposition info */
        int pwl_nv;
        float pwl_t[SA_MAX_PWL_VERTICES];
        float pwl_v[SA_MAX_PWL_VERTICES];
        int decompose;
        int sub_reps;
        int sub_pwl_nv;
        float sub_pwl_t[SA_MAX_PWL_VERTICES];
        float sub_pwl_v[SA_MAX_PWL_VERTICES];
        float sub_period_us;
        /* Shared FFT template (for many-sample arb without sub-period) */
        float *shared_fft_mags;
        int shared_fft_nbins;
        float shared_fft_spacing;
        int use_fft;

        did = unique_ids[idx];
        gdef = NULL;
        pwl_nv = 0;
        decompose = 0;
        sub_reps = 0;
        sub_pwl_nv = 0;
        sub_period_us = 0.0f;
        shared_fft_mags = NULL;
        shared_fft_nbins = 0;
        shared_fft_spacing = 0.0f;
        use_fft = 0;

        /* Find first occurrence of this def to get gdef */
        for (j = 0; j < num_occ; ++j)
        {
            if (occ[j].def_id == did)
            {
                gdef = &desc->grad_definitions[occ[j].def_index];
                break;
            }
        }
        if (!gdef)
            continue;

        /* --- Compute shared PWL parameters for this def --- */

        if (gdef->type == 0)
        {
            /* Trapezoid: 4-vertex PWL */
            float tr = (float)gdef->rise_time_or_unused;
            float tf = (float)gdef->flat_time_or_unused;
            float td = (float)gdef->fall_time_or_num_uncompressed_samples;
            pwl_nv = 4;
            pwl_t[0] = 0.0f;
            pwl_v[0] = 0.0f;
            pwl_t[1] = tr;
            pwl_v[1] = 1.0f;
            pwl_t[2] = tr + tf;
            pwl_v[2] = 1.0f;
            pwl_t[3] = tr + tf + td;
            pwl_v[3] = 0.0f;
        }
        else if (gdef->type == 1)
        {
            /* Arbitrary waveform */
            int shape_id = gdef->shot_shape_ids[0];
            int time_shape_id = gdef->unused_or_time_shape_id;

            if (shape_id > 0 && shape_id <= desc->num_shapes)
            {
                pulseg_shape_arbitrary decomp_wave;
                decomp_wave.samples = NULL;
                decomp_wave.num_samples = 0;
                decomp_wave.num_uncompressed_samples = 0;

                if (pulseg__decompress_shape(&decomp_wave,
                                                &desc->shapes[shape_id - 1], 1.0f) &&
                    decomp_wave.num_uncompressed_samples > 0)
                {
                    int num_samp = decomp_wave.num_uncompressed_samples;
                    float *wave_samp = decomp_wave.samples;

                    /* Decompress time shape if present */
                    pulseg_shape_arbitrary decomp_time;
                    float *time_us = NULL;
                    int has_time = 0;
                    decomp_time.samples = NULL;
                    decomp_time.num_samples = 0;
                    decomp_time.num_uncompressed_samples = 0;

                    if (time_shape_id > 0 && time_shape_id <= desc->num_shapes)
                    {
                        if (pulseg__decompress_shape(&decomp_time,
                                                        &desc->shapes[time_shape_id - 1], raster) &&
                            decomp_time.num_uncompressed_samples > 0)
                        {
                            has_time = 1;
                            time_us = decomp_time.samples;
                        }
                    }

                    if (num_samp < PULSEG_MIN_ARBITRARY_SAMPLES)
                    {
                        /* Few-sample arb: PWL from vertices */
                        nv = num_samp;
                        if (nv > SA_MAX_PWL_VERTICES)
                            nv = SA_MAX_PWL_VERTICES;
                        pwl_nv = nv;
                        for (s_idx = 0; s_idx < nv; ++s_idx)
                        {
                            if (has_time && time_us)
                                pwl_t[s_idx] = time_us[s_idx];
                            else
                                pwl_t[s_idx] = 0.5f * raster + (float)s_idx * raster;
                            pwl_v[s_idx] = wave_samp[s_idx];
                        }
                    }
                    else
                    {
                        /* Many-sample arb: try sub-period detection */
                        int sp, sr;
                        if (sa_detect_sub_period(wave_samp, num_samp, &sp, &sr))
                        {
                            /* Sub-period found: build PWL for one period */
                            decompose = 1;
                            sub_reps = sr;
                            sub_period_us = (has_time && time_us && sp < num_samp)
                                                ? (time_us[sp] - time_us[0])
                                                : (float)sp * raster;

                            /* Build PWL for one sub-period */
                            nv = sp;
                            if (nv > SA_MAX_PWL_VERTICES)
                                nv = SA_MAX_PWL_VERTICES;
                            sub_pwl_nv = nv;
                            if (nv == sp)
                            {
                                /* Use all samples */
                                for (s_idx = 0; s_idx < nv; ++s_idx)
                                {
                                    if (has_time && time_us)
                                        sub_pwl_t[s_idx] = time_us[s_idx] - time_us[0];
                                    else
                                        sub_pwl_t[s_idx] = (float)s_idx * raster;
                                    sub_pwl_v[s_idx] = wave_samp[s_idx];
                                }
                            }
                            else
                            {
                                /* Subsample uniformly */
                                for (s_idx = 0; s_idx < nv; ++s_idx)
                                {
                                    int si = (s_idx * (sp - 1)) / (nv - 1);
                                    if (has_time && time_us)
                                        sub_pwl_t[s_idx] = time_us[si] - time_us[0];
                                    else
                                        sub_pwl_t[s_idx] = (float)si * raster;
                                    sub_pwl_v[s_idx] = wave_samp[si];
                                }
                            }
                        }
                        else
                        {
                            /* No sub-period: compute per-event FFT response */
                            if (sa_compute_fft_response(
                                    wave_samp,
                                    has_time ? time_us : NULL,
                                    num_samp, raster,
                                    &shared_fft_mags,
                                    &shared_fft_nbins,
                                    &shared_fft_spacing))
                            {
                                use_fft = 1;
                                pwl_nv = 0;
                            }
                            else
                            {
                                /* FFT failed: fall back to coarse PWL */
                                nv = SA_MAX_PWL_VERTICES;
                                if (nv > num_samp)
                                    nv = num_samp;
                                pwl_nv = nv;
                                for (s_idx = 0; s_idx < nv; ++s_idx)
                                {
                                    int si = (nv > 1) ? (s_idx * (num_samp - 1)) / (nv - 1) : 0;
                                    if (has_time && time_us)
                                        pwl_t[s_idx] = time_us[si];
                                    else
                                        pwl_t[s_idx] = 0.5f * raster + (float)si * raster;
                                    pwl_v[s_idx] = wave_samp[si];
                                }
                            }
                        }
                    }

                    if (decomp_time.samples)
                        PULSEG_FREE(decomp_time.samples);
                }
                if (decomp_wave.samples)
                    PULSEG_FREE(decomp_wave.samples);
            }
        }

        /* --- Emit events for each occurrence of this def --- */
        for (j = 0; j < num_occ; ++j)
        {
            if (occ[j].def_id != did)
                continue;

            if (decompose && sub_reps > 0)
            {
                /* Decompose into sub-events */
                int m;
                for (m = 0; m < sub_reps; ++m)
                {
                    if (n_events >= cap)
                    {
                        sa_event *tmp;
                        cap *= 2;
                        tmp = (sa_event *)PULSEG_ALLOC((size_t)cap * sizeof(sa_event));
                        if (!tmp)
                        {
                            if (shared_fft_mags)
                                PULSEG_FREE(shared_fft_mags);
                            PULSEG_FREE(unique_ids);
                            PULSEG_FREE(ae->events);
                            ae->events = NULL;
                            return PULSEG_ERR_ALLOC_FAILED;
                        }
                        memcpy(tmp, ae->events, (size_t)n_events * sizeof(sa_event));
                        PULSEG_FREE(ae->events);
                        ae->events = tmp;
                    }
                    ae->events[n_events].def_id = did;
                    ae->events[n_events].def_index = occ[j].def_index;
                    ae->events[n_events].start_time_us = occ[j].start_time_us + (float)m * sub_period_us + (float)gdef->delay;
                    ae->events[n_events].amplitude = occ[j].amplitude;
                    ae->events[n_events].pwl_num_vertices = sub_pwl_nv;
                    memcpy(ae->events[n_events].pwl_times_us, sub_pwl_t,
                           (size_t)sub_pwl_nv * sizeof(float));
                    memcpy(ae->events[n_events].pwl_values, sub_pwl_v,
                           (size_t)sub_pwl_nv * sizeof(float));
                    ae->events[n_events].fft_magnitudes = NULL;
                    ae->events[n_events].fft_num_bins = 0;
                    ae->events[n_events].fft_freq_spacing_hz = 0.0f;
                    n_events++;
                }
            }
            else
            {
                /* Single event */
                if (n_events >= cap)
                {
                    sa_event *tmp;
                    cap *= 2;
                    tmp = (sa_event *)PULSEG_ALLOC((size_t)cap * sizeof(sa_event));
                    if (!tmp)
                    {
                        if (shared_fft_mags)
                            PULSEG_FREE(shared_fft_mags);
                        PULSEG_FREE(unique_ids);
                        PULSEG_FREE(ae->events);
                        ae->events = NULL;
                        return PULSEG_ERR_ALLOC_FAILED;
                    }
                    memcpy(tmp, ae->events, (size_t)n_events * sizeof(sa_event));
                    PULSEG_FREE(ae->events);
                    ae->events = tmp;
                }
                ae->events[n_events].def_id = did;
                ae->events[n_events].def_index = occ[j].def_index;
                ae->events[n_events].start_time_us = occ[j].start_time_us + (float)gdef->delay;
                ae->events[n_events].amplitude = occ[j].amplitude;
                if (use_fft && shared_fft_mags)
                {
                    float *fft_copy = (float *)PULSEG_ALLOC(
                        (size_t)shared_fft_nbins * sizeof(float));
                    if (!fft_copy)
                    {
                        PULSEG_FREE(shared_fft_mags);
                        PULSEG_FREE(unique_ids);
                        PULSEG_FREE(ae->events);
                        ae->events = NULL;
                        return PULSEG_ERR_ALLOC_FAILED;
                    }
                    memcpy(fft_copy, shared_fft_mags,
                           (size_t)shared_fft_nbins * sizeof(float));
                    ae->events[n_events].fft_magnitudes = fft_copy;
                    ae->events[n_events].fft_num_bins = shared_fft_nbins;
                    ae->events[n_events].fft_freq_spacing_hz = shared_fft_spacing;
                    ae->events[n_events].pwl_num_vertices = 0;
                }
                else
                {
                    ae->events[n_events].pwl_num_vertices = pwl_nv;
                    if (pwl_nv > 0)
                    {
                        memcpy(ae->events[n_events].pwl_times_us, pwl_t,
                               (size_t)pwl_nv * sizeof(float));
                        memcpy(ae->events[n_events].pwl_values, pwl_v,
                               (size_t)pwl_nv * sizeof(float));
                    }
                    ae->events[n_events].fft_magnitudes = NULL;
                    ae->events[n_events].fft_num_bins = 0;
                    ae->events[n_events].fft_freq_spacing_hz = 0.0f;
                }
                n_events++;
            }
        }
        /* Free shared FFT template for this def_id */
        if (shared_fft_mags)
        {
            PULSEG_FREE(shared_fft_mags);
            shared_fft_mags = NULL;
        }
    }

    ae->num_events = n_events;
    PULSEG_FREE(unique_ids);
    return PULSEG_SUCCESS;
}

/**
 * Build event lists for all three axes.
 */
static int sa_build_structural_events(
    sa_structural_events *se,
    const struct pulseg_sequence_descriptor *desc,
    int start_block, int block_count)
{
    int ax, result;
    sa_raw_occurrence *occ;
    int num_occ;

    memset(se, 0, sizeof(*se));

    for (ax = 0; ax < 3; ++ax)
    {
        occ = NULL;
        num_occ = sa_extract_raw_occurrences(&occ, desc, start_block, block_count, ax);
        if (num_occ < 0)
        {
            sa_free_structural_events(se);
            return PULSEG_ERR_ALLOC_FAILED;
        }
        result = sa_build_axis_events(&se->axes[ax], occ, num_occ, desc);
        PULSEG_FREE(occ);
        if (PULSEG_FAILED(result))
        {
            sa_free_structural_events(se);
            return result;
        }
    }
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Structural acoustic analysis — analytical spectrum evaluation     */
/* ================================================================== */

/**
 * Evaluate |G(f)| / |G(0)| for a piecewise-linear waveform defined by
 * n_vtx vertices (t_us[i], v[i]).  Times are in microseconds.
 *
 * For each linear segment [t_k, t_{k+1}] with values [v_k, v_{k+1}]:
 *   G_k(f) = e^{-j*omega*t_k} * [a*I_0 + b*I_1]
 * where a = v_k, b = (v_{k+1}-v_k)/T_k, omega = 2*pi*f,
 *   I_0 = integral_0^T e^{-j*omega*tau} dtau
 *   I_1 = integral_0^T tau * e^{-j*omega*tau} dtau
 *
 * Returns |G(f)| / |G(0)|.
 */
static float sa_eval_pwl_response(
    float f_hz, const float *t_us, const float *v, int n_vtx)
{
    double omega, g0, g_re, g_im, mag;
    int k;

    if (n_vtx < 2)
        return 1.0f;

    g0 = 0.0;
    for (k = 0; k < n_vtx - 1; ++k)
    {
        double dt = (double)(t_us[k + 1] - t_us[k]);
        g0 += 0.5 * ((double)v[k] + (double)v[k + 1]) * dt;
    }
    if (g0 < 1.0e-30 && g0 > -1.0e-30)
        return 1.0f;
    g0 = fabs(g0);

    omega = 2.0 * M_PI * (double)f_hz * 1.0e-6;
    if (omega < 1.0e-12 && omega > -1.0e-12)
        return 1.0f;

    g_re = 0.0;
    g_im = 0.0;

    for (k = 0; k < n_vtx - 1; ++k)
    {
        double tk = (double)t_us[k];
        double Tk = (double)(t_us[k + 1] - t_us[k]);
        double ak = (double)v[k];
        double bk = ((double)v[k + 1] - ak) / Tk;
        double wTk = omega * Tk;
        double cos_wTk = cos(wTk);
        double sin_wTk = sin(wTk);
        double cos_ph = cos(omega * tk);
        double sin_ph = sin(omega * tk);
        double c0_re, c0_im, c1_re, c1_im;
        double x_re, x_im, seg_re, seg_im;

        if (Tk < 1.0e-12)
            continue;

        c0_re = sin_wTk / omega;
        c0_im = (cos_wTk - 1.0) / omega;

        {
            double I0mT_re = c0_re - Tk * cos_wTk;
            double I0mT_im = c0_im + Tk * sin_wTk;
            c1_re = I0mT_im / omega;
            c1_im = -I0mT_re / omega;
        }

        x_re = ak * c0_re + bk * c1_re;
        x_im = ak * c0_im + bk * c1_im;

        seg_re = x_re * cos_ph + x_im * sin_ph;
        seg_im = x_im * cos_ph - x_re * sin_ph;

        g_re += seg_re;
        g_im += seg_im;
    }

    mag = sqrt(g_re * g_re + g_im * g_im);
    return (float)(mag / g0);
}

/**
 * Evaluate the waveform response W_k(f) for a single event.
 * Returns real-valued |W(f)| / |W(0)| (normalised).
 */
static float sa_eval_event_response(const sa_event *ev, float f_hz)
{
    /* FFT-based response: interpolate into stored magnitude spectrum */
    if (ev->fft_num_bins > 0 && ev->fft_magnitudes)
    {
        float bin_f = f_hz / ev->fft_freq_spacing_hz;
        int bin_lo = (int)bin_f;
        float frac;
        if (bin_lo < 0)
            return 1.0f;
        if (bin_lo >= ev->fft_num_bins - 1)
            return ev->fft_magnitudes[ev->fft_num_bins - 1];
        frac = bin_f - (float)bin_lo;
        return ev->fft_magnitudes[bin_lo] * (1.0f - frac) + ev->fft_magnitudes[bin_lo + 1] * frac;
    }
    /* PWL-based response */
    if (ev->pwl_num_vertices >= 2)
        return sa_eval_pwl_response(f_hz, ev->pwl_times_us, ev->pwl_values,
                                    ev->pwl_num_vertices);
    return 1.0f; /* Broadband fallback */
}

/**
 * Evaluate the complex spectral contribution of one event at frequency f:
 *   a_k(f) = A_k * W_k(f) * e^{-j 2 pi f t_k}
 */
static void sa_eval_event_spectrum(
    float f_hz, const sa_event *ev,
    float *out_re, float *out_im)
{
    float amp, h;
    double phase, cos_ph, sin_ph;

    amp = ev->amplitude;
    h = sa_eval_event_response(ev, f_hz);

    phase = -2.0 * M_PI * (double)f_hz * (double)ev->start_time_us * 1.0e-6;
    cos_ph = cos(phase);
    sin_ph = sin(phase);

    *out_re = amp * h * (float)cos_ph;
    *out_im = amp * h * (float)sin_ph;
}

/**
 * Evaluate the inner TR spectrum magnitude for one axis at frequency f:
 *   |H(f)| = |sum_k a_k(f) * D_{N_k}(f)|
 * where D_N is the Dirichlet kernel for events repeated N times with
 * period T:  D_N(f,T) = sum_{n=0}^{N-1} e^{-j 2 pi f n T}.
 *
 * Also returns the incoherent sum sqrt(sum N_k * |a_k|^2) for coherence ratio.
 */
static void sa_eval_inner_spectrum(
    float f_hz,
    const sa_axis_events *ae,
    float *out_coherent_mag,
    float *out_incoherent_mag)
{
    float sum_re, sum_im;
    float inc_sum;
    int k;

    sum_re = 0.0f;
    sum_im = 0.0f;
    inc_sum = 0.0f;

    for (k = 0; k < ae->num_events; ++k)
    {
        float d_re, d_im, d_mag2;
        int N = ae->events[k].num_reps;
        if (N < 1)
            N = 1;

        sa_eval_event_spectrum(f_hz, &ae->events[k], &d_re, &d_im);
        d_mag2 = d_re * d_re + d_im * d_im;

        if (N > 1 && ae->events[k].rep_period_us > 0.0f)
        {
            /* Dirichlet kernel: D_N = e^{-j(N-1)phi/2} sin(N phi/2)/sin(phi/2) */
            double phi = -2.0 * M_PI * (double)f_hz * (double)ae->events[k].rep_period_us * 1.0e-6;
            double half_phi = phi * 0.5;
            double sin_half = sin(half_phi);
            double dk_re, dk_im;
            double new_re, new_im;

            if (fabs(sin_half) < 1.0e-12)
            {
                dk_re = (double)N;
                dk_im = 0.0;
            }
            else
            {
                double ratio = sin((double)N * half_phi) / sin_half;
                double center = (double)(N - 1) * half_phi;
                dk_re = ratio * cos(center);
                dk_im = ratio * sin(center);
            }

            new_re = (double)d_re * dk_re - (double)d_im * dk_im;
            new_im = (double)d_re * dk_im + (double)d_im * dk_re;
            sum_re += (float)new_re;
            sum_im += (float)new_im;
            inc_sum += (float)N * d_mag2;
        }
        else
        {
            sum_re += d_re;
            sum_im += d_im;
            inc_sum += d_mag2;
        }
    }

    *out_coherent_mag = (float)sqrt((double)(sum_re * sum_re + sum_im * sum_im));
    *out_incoherent_mag = (float)sqrt((double)inc_sum);
}

/**
 * Compute spectral-weighted effective gradient amplitude G_eff at frequency f:
 *   G_eff(f) = sum_k |A_k| * |a_k(f)| * |D_{N_k}(f)| / sum_k |a_k(f)| * |D_{N_k}(f)|
 */
static float sa_eval_geff(
    float f_hz,
    const sa_axis_events *ae)
{
    float w_sum, w_amp_sum;
    int k;

    w_sum = 0.0f;
    w_amp_sum = 0.0f;

    for (k = 0; k < ae->num_events; ++k)
    {
        float d_re, d_im, d_mag;
        float abs_amp;
        int N = ae->events[k].num_reps;
        if (N < 1)
            N = 1;

        sa_eval_event_spectrum(f_hz, &ae->events[k], &d_re, &d_im);
        d_mag = (float)sqrt((double)(d_re * d_re + d_im * d_im));

        if (N > 1 && ae->events[k].rep_period_us > 0.0f)
        {
            double phi = -2.0 * M_PI * (double)f_hz * (double)ae->events[k].rep_period_us * 1.0e-6;
            double half_phi = phi * 0.5;
            double sin_half = sin(half_phi);
            float dk_mag;
            if (fabs(sin_half) < 1.0e-12)
                dk_mag = (float)N;
            else
                dk_mag = (float)fabs(sin((double)N * half_phi) / sin_half);
            d_mag *= dk_mag;
        }

        abs_amp = ae->events[k].amplitude;
        if (abs_amp < 0.0f)
            abs_amp = -abs_amp;
        w_sum += d_mag;
        w_amp_sum += d_mag * abs_amp;
    }

    if (w_sum <= 1.0e-20f)
        return 0.0f;
    return w_amp_sum / w_sum;
}

/* ================================================================== */
/*  Structural acoustic analysis — top-level violation check          */
/* ================================================================== */

/**
 * Run full structural acoustic analysis with event-level analytical framework.
 *
 * For each axis:
 *   1. Extract block-level occurrences from the base pass (not expanded),
 *      decompose arbitrary waveforms via autocorrelation if sub-periodicity
 *      detected.
 *   2. Tag imaging events with repetition count (num_avgs) and period, and
 *      shift cooldown event times to their expanded-pass positions.
 *   3. Build candidate frequency grid (TR harmonics of expanded pass; K is
 *      clamped to min 2 so the Fourier comb works for single-pass sequences).
 *   4. Evaluate inner TR spectrum H(f) at each candidate via coherent sum
 *      of per-event contributions with Dirichlet kernel for repeated events.
 *   5. Analytical power gate + two-tier candidate selection:
 *      - Tier 1: coherence ratio > threshold AND |H| > floor
 *      - Tier 2: |H| > safety threshold
 *   6. For each candidate: compute G_eff(f) and check forbidden bands.
 */
static int sa_check_structural_violations(
    pulseg_mech_resonances_spectra *spectra,
    const struct pulseg_sequence_descriptor *desc,
    int start_block, int block_count,
    int num_instances, float tr_duration_us,
    int num_forbidden_bands,
    const pulseg_forbidden_band *forbidden_bands,
    float peak_log10_threshold, float peak_norm_scale,
    float peak_eps, float peak_prominence,
    int num_avgs)
{
    sa_structural_events se;
    int result, ax, i, b, ci;
    int num_freqs, num_candidates;
    float f_hz, freq_max;
    float *eval_freqs;
    float *coherent_mag[3];
    float *incoherent_mag[3];
    int *is_candidate_ax[3];
    float max_fft_sq[3], fft_promo_sq[3];
    float max_ana_sq[3], ana_gate_sq[3];
    int has_fft_data;
    int n_tr_harmonics;
    int *fft_bypass_set_ax[3];
    float *cand_freqs;
    float *cand_amps[3];
    float *cand_grad_amps;
    float *cand_grad_amps_ax[3];
    int *cand_violations;
    float *ana_peak_freqs;
    float *ana_peak_amps[3];
    float *ana_peak_phases[3];
    float *ana_peak_widths;
    float *surviving_freqs_hz;

    /* Component-level export arrays */
    int max_component_terms;
    int num_component_terms;
    float *component_freqs_hz;
    float *component_amps;
    float *component_phases_rad;
    float *component_widths_hz;
    int *component_axes;
    int *component_def_ids;
    int *component_contrib_ids;
    int *component_run_ids;

    (void)peak_log10_threshold;
    (void)peak_norm_scale;
    (void)peak_eps;
    (void)peak_prominence;

    memset(&se, 0, sizeof(se));
    eval_freqs = NULL;
    fft_bypass_set_ax[0] = fft_bypass_set_ax[1] = fft_bypass_set_ax[2] = NULL;
    cand_freqs = NULL;
    n_tr_harmonics = 0;
    cand_grad_amps = NULL;
    cand_violations = NULL;
    ana_peak_freqs = NULL;
    ana_peak_widths = NULL;
    surviving_freqs_hz = NULL;
    num_component_terms = 0;
    max_component_terms = 0;
    component_freqs_hz = NULL;
    component_amps = NULL;
    component_phases_rad = NULL;
    component_widths_hz = NULL;
    component_axes = NULL;
    component_def_ids = NULL;
    component_contrib_ids = NULL;
    component_run_ids = NULL;
    for (ax = 0; ax < 3; ++ax)
    {
        coherent_mag[ax] = NULL;
        incoherent_mag[ax] = NULL;
        is_candidate_ax[ax] = NULL;
        cand_amps[ax] = NULL;
        cand_grad_amps_ax[ax] = NULL;
        ana_peak_amps[ax] = NULL;
        ana_peak_phases[ax] = NULL;
    }

    result = sa_build_structural_events(&se, desc, start_block, block_count);
    if (PULSEG_FAILED(result))
        return result;

    /* --- Tag imaging events with repetition info ---
     * For non-degenerate sequences (num_avgs > 1), imaging events repeat
     * num_avgs times with period = imaging pass duration.  Cooldown events
     * are shifted to their position in the expanded pass.  This lets the
     * Dirichlet kernel handle repetition analytically (O(1) per event per
     * frequency) instead of enumerating all N copies (O(N)). */
    if (num_avgs > 1)
    {
        const struct pulseg_tr_descriptor *trd = &desc->tr_descriptor;
        int prep_blk = trd->num_prep_blocks;
        int img_len = trd->num_trs * trd->tr_size;
        float prep_dur_us = 0.0f;
        float img_dur_us = 0.0f;
        float img_end_us;
        int bi;

        for (bi = 0; bi < prep_blk; ++bi)
        {
            int idx = start_block + bi;
            const struct pulseg_block_table_element *bte = &desc->block_table[idx];
            const struct pulseg_block_definition *bdef = &desc->block_definitions[bte->id];
            prep_dur_us += (bte->duration_us >= 0) ? (float)bte->duration_us : (float)bdef->duration_us;
        }
        for (bi = 0; bi < img_len; ++bi)
        {
            int idx = start_block + prep_blk + bi;
            const struct pulseg_block_table_element *bte = &desc->block_table[idx];
            const struct pulseg_block_definition *bdef = &desc->block_definitions[bte->id];
            img_dur_us += (bte->duration_us >= 0) ? (float)bte->duration_us : (float)bdef->duration_us;
        }
        img_end_us = prep_dur_us + img_dur_us;

        for (ax = 0; ax < 3; ++ax)
        {
            int k;
            for (k = 0; k < se.axes[ax].num_events; ++k)
            {
                sa_event *ev = &se.axes[ax].events[k];
                if (ev->start_time_us >= prep_dur_us && ev->start_time_us < img_end_us)
                {
                    ev->num_reps = num_avgs;
                    ev->rep_period_us = img_dur_us;
                }
                else if (ev->start_time_us >= img_end_us)
                {
                    ev->start_time_us += (float)(num_avgs - 1) * img_dur_us;
                    ev->num_reps = 1;
                    ev->rep_period_us = 0.0f;
                }
                else
                {
                    ev->num_reps = 1;
                    ev->rep_period_us = 0.0f;
                }
            }
        }
    }
    else
    {
        for (ax = 0; ax < 3; ++ax)
        {
            int k;
            for (k = 0; k < se.axes[ax].num_events; ++k)
            {
                se.axes[ax].events[k].num_reps = 1;
                se.axes[ax].events[k].rep_period_us = 0.0f;
            }
        }
    }

    /* Compute evaluation frequency upper bound */
    if (spectra->num_freq_bins <= 0)
    {
        sa_free_structural_events(&se);
        return PULSEG_SUCCESS;
    }
    freq_max = spectra->freq_min_hz +
               (float)(spectra->num_freq_bins - 1) * spectra->freq_spacing_hz;

    /* --- Compute per-axis FFT power gate threshold ---
     * Find peak magnitude per axis of the dense FFT and set per-axis
     * gates at SA_FFT_PROMOTION_POWER_FRAC of each axis's peak.
     * Work in squared magnitudes to avoid sqrt in the inner loops. */
    max_fft_sq[0] = max_fft_sq[1] = max_fft_sq[2] = 0.0f;
    has_fft_data = (spectra->spectrum_full_gx && spectra->spectrum_full_gy &&
                    spectra->spectrum_full_gz && spectra->num_freq_bins > 0 &&
                    spectra->freq_spacing_hz > 0.0f);
    if (has_fft_data)
    {
        for (i = 0; i < spectra->num_freq_bins; ++i)
        {
            float s[3];
            s[0] = spectra->spectrum_full_gx[i];
            s[1] = spectra->spectrum_full_gy[i];
            s[2] = spectra->spectrum_full_gz[i];
            for (ax = 0; ax < 3; ++ax)
            {
                float sq = s[ax] * s[ax];
                if (sq > max_fft_sq[ax])
                    max_fft_sq[ax] = sq;
            }
        }
    }
    for (ax = 0; ax < 3; ++ax)
        fft_promo_sq[ax] = SA_FFT_PROMOTION_POWER_FRAC * max_fft_sq[ax];

    /* --- Build evaluation frequency list ---
     * TR harmonics form the primary grid.  Additionally, any local maximum
     * of the dense FFT RSS spectrum that exceeds the gate threshold and
     * falls between TR harmonics is promoted into the evaluation set.
     * This catches spectral features (e.g. the bipolar-readout fundamental
     * in bSSFP) that land between adjacent TR harmonics.
     *
     * For non-degenerate sequences the caller passes the full-pass duration
     * as tr_duration_us and num_instances is the pass count; the Dirichlet
     * kernel on imaging events naturally produces peaks at the imaging-TR
     * harmonic spacing within the fine pass-harmonic grid. */
    if (num_instances < 2)
        num_instances = 2;
    {
        float tr_f1 = 1.0e6f / tr_duration_us;
        int m_max = (int)(freq_max / tr_f1);
        int max_eval, m, fi;
        if (m_max < 1)
            m_max = 0;

        /* Upper bound on eval_freqs size */
        max_eval = m_max + (has_fft_data ? spectra->num_freq_bins : 0);
        if (max_eval < 1)
        {
            sa_free_structural_events(&se);
            return PULSEG_SUCCESS;
        }
        eval_freqs = (float *)PULSEG_ALLOC((size_t)max_eval * sizeof(float));
        if (!eval_freqs)
            goto alloc_fail;

        /* 1) TR harmonics */
        num_freqs = 0;
        for (m = 0; m < m_max; ++m)
            eval_freqs[num_freqs++] = (float)(m + 1) * tr_f1;
        n_tr_harmonics = num_freqs;

        /* 2) FFT local maxima between TR harmonics (peak promotion)
         * Per-axis: promote a frequency if ANY axis has a local max
         * above its own per-axis threshold. */
        if (has_fft_data && m_max > 0)
        {
            const float *spec_ax[3];
            spec_ax[0] = spectra->spectrum_full_gx;
            spec_ax[1] = spectra->spectrum_full_gy;
            spec_ax[2] = spectra->spectrum_full_gz;
            for (fi = 1; fi < spectra->num_freq_bins - 1; ++fi)
            {
                float f_peak, nearest, dist;
                int any_local_max = 0;
                for (ax = 0; ax < 3; ++ax)
                {
                    float cur = spec_ax[ax][fi];
                    float cur_sq = cur * cur;
                    float prev_sq, next_sq;
                    if (cur_sq < fft_promo_sq[ax])
                        continue;
                    prev_sq = spec_ax[ax][fi - 1] * spec_ax[ax][fi - 1];
                    next_sq = spec_ax[ax][fi + 1] * spec_ax[ax][fi + 1];
                    if (cur_sq > prev_sq && cur_sq > next_sq)
                    {
                        any_local_max = 1;
                        break;
                    }
                }
                if (!any_local_max)
                    continue;

                /* Not near a TR harmonic? (> 25% of harmonic spacing) */
                f_peak = spectra->freq_min_hz + (float)fi * spectra->freq_spacing_hz;
                if (f_peak <= 0.0f || f_peak > freq_max)
                    continue;
                nearest = (float)floor((double)(f_peak / tr_f1)) * tr_f1;
                dist = f_peak - nearest;
                if (dist < 0.0f)
                    dist = -dist;
                if (dist <= tr_f1 * 0.25f)
                    continue;

                eval_freqs[num_freqs++] = f_peak;
            }
        }

        if (num_freqs == 0)
        {
            PULSEG_FREE(eval_freqs);
            eval_freqs = NULL;
            sa_free_structural_events(&se);
            return PULSEG_SUCCESS;
        }
    }

    /* --- Evaluate inner spectrum at each frequency for each axis --- */
    for (ax = 0; ax < 3; ++ax)
    {
        coherent_mag[ax] = (float *)PULSEG_ALLOC((size_t)num_freqs * sizeof(float));
        incoherent_mag[ax] = (float *)PULSEG_ALLOC((size_t)num_freqs * sizeof(float));
        if (!coherent_mag[ax] || !incoherent_mag[ax])
            goto alloc_fail;
    }

    for (i = 0; i < num_freqs; ++i)
    {
        f_hz = eval_freqs[i];
        for (ax = 0; ax < 3; ++ax)
        {
            if (se.axes[ax].num_events == 0)
            {
                coherent_mag[ax][i] = 0.0f;
                incoherent_mag[ax][i] = 0.0f;
            }
            else
            {
                sa_eval_inner_spectrum(f_hz, &se.axes[ax],
                                       &coherent_mag[ax][i], &incoherent_mag[ax][i]);
            }
        }
    }

    /* --- Per-axis analytical power gate threshold ---
     * Compute peak coherent magnitude per axis across all evaluation
     * frequencies.  Gate at SA_ANALYTICAL_GATE_FRAC of each axis's own
     * peak to reject harmonics with negligible spectral weight. */
    max_ana_sq[0] = max_ana_sq[1] = max_ana_sq[2] = 0.0f;
    for (i = 0; i < num_freqs; ++i)
    {
        for (ax = 0; ax < 3; ++ax)
        {
            float sq = coherent_mag[ax][i] * coherent_mag[ax][i];
            if (sq > max_ana_sq[ax])
                max_ana_sq[ax] = sq;
        }
    }
    for (ax = 0; ax < 3; ++ax)
        ana_gate_sq[ax] = SA_ANALYTICAL_GATE_FRAC * max_ana_sq[ax];

    /* --- Per-axis candidate selection --- */
    for (ax = 0; ax < 3; ++ax)
    {
        is_candidate_ax[ax] = (int *)PULSEG_ALLOC((size_t)num_freqs * sizeof(int));
        if (!is_candidate_ax[ax])
            goto alloc_fail;
        for (i = 0; i < num_freqs; ++i)
            is_candidate_ax[ax][i] = 0;
    }

    /* --- Pre-build per-axis FFT bypass sets ---
     * For each axis independently, find FFT local maxima above the
     * per-axis promotion threshold and mark the nearest TR harmonic
     * for gate bypass on that axis.  This catches high-order harmonics
     * where the PWL spectral model underestimates the true amplitude. */
    fft_bypass_set_ax[0] = fft_bypass_set_ax[1] = fft_bypass_set_ax[2] = NULL;
    if (has_fft_data && n_tr_harmonics > 0)
    {
        float tr_f1_byp = eval_freqs[0]; /* first TR harmonic = f1 */
        const float *spec_bypass[3];
        spec_bypass[0] = spectra->spectrum_full_gx;
        spec_bypass[1] = spectra->spectrum_full_gy;
        spec_bypass[2] = spectra->spectrum_full_gz;
        for (ax = 0; ax < 3; ++ax)
        {
            fft_bypass_set_ax[ax] = (int *)PULSEG_ALLOC(
                (size_t)num_freqs * sizeof(int));
            if (fft_bypass_set_ax[ax])
            {
                int fi;
                for (i = 0; i < num_freqs; ++i)
                    fft_bypass_set_ax[ax][i] = 0;
                for (fi = 1; fi < spectra->num_freq_bins - 1; ++fi)
                {
                    float cur = spec_bypass[ax][fi];
                    float cur_sq = cur * cur;
                    float prev_sq, next_sq;
                    float f_peak;
                    int m_nearest;

                    if (cur_sq < fft_promo_sq[ax])
                        continue;

                    prev_sq = spec_bypass[ax][fi - 1] * spec_bypass[ax][fi - 1];
                    next_sq = spec_bypass[ax][fi + 1] * spec_bypass[ax][fi + 1];
                    if (cur_sq <= prev_sq || cur_sq <= next_sq)
                        continue;

                    /* FFT local max — find nearest TR harmonic index */
                    f_peak = spectra->freq_min_hz + (float)fi * spectra->freq_spacing_hz;
                    m_nearest = (int)(f_peak / tr_f1_byp);
                    if (m_nearest >= 0 && m_nearest < n_tr_harmonics)
                        fft_bypass_set_ax[ax][m_nearest] = 1;
                }
            }
        }
    }

    for (i = 0; i < num_freqs; ++i)
    {
        for (ax = 0; ax < 3; ++ax)
        {
            float coh = coherent_mag[ax][i];
            float inc = incoherent_mag[ax][i];
            int is_bypassed = (fft_bypass_set_ax[ax] &&
                               fft_bypass_set_ax[ax][i]);

            /* Per-axis analytical power gate: reject this axis at this
             * frequency if its coherent magnitude is below the per-axis
             * gate threshold, unless FFT bypass is active for this axis. */
            if (coh * coh < ana_gate_sq[ax])
            {
                if (!is_bypassed)
                    continue;
            }

            if (se.axes[ax].num_events == 0)
                continue;

            /* Tier 3: FFT-promoted peaks — per-axis if analytical signal
             * is above the amplitude floor. */
            if (i >= n_tr_harmonics)
            {
                if (coh > SA_AMPLITUDE_FLOOR)
                    is_candidate_ax[ax][i] = 1;
                continue;
            }

            /* FFT-bypassed TR harmonics: the dense FFT already confirms a
             * real spectral peak at this frequency, so use the relaxed
             * amplitude-floor check instead of requiring high coherence. */
            if (is_bypassed)
            {
                if (coh > SA_AMPLITUDE_FLOOR)
                    is_candidate_ax[ax][i] = 1;
                continue;
            }

            /* Tier 1: coherence-detected (N_events >= 2) */
            if (se.axes[ax].num_events >= 2 && inc > 1.0e-20f)
            {
                float rho = coh / inc;
                if (rho > SA_COHERENCE_RATIO_THRESHOLD && coh > SA_AMPLITUDE_FLOOR)
                {
                    is_candidate_ax[ax][i] = 1;
                    continue;
                }
            }

            /* Tier 2: absolute amplitude threshold (single-event axes) */
            if (se.axes[ax].num_events == 1 && coh > SA_AMPLITUDE_SAFETY)
            {
                is_candidate_ax[ax][i] = 1;
            }
        }
    }

    for (ax = 0; ax < 3; ++ax)
    {
        if (fft_bypass_set_ax[ax])
            PULSEG_FREE(fft_bypass_set_ax[ax]);
    }

    /* Count candidates: a frequency is a candidate if any axis qualifies */
    num_candidates = 0;
    for (i = 0; i < num_freqs; ++i)
    {
        int any = 0;
        for (ax = 0; ax < 3; ++ax)
            if (is_candidate_ax[ax][i])
                any = 1;
        if (any)
            num_candidates++;
    }

    /* --- Estimate max component terms for export --- */
    max_component_terms = 0;
    for (ax = 0; ax < 3; ++ax)
        max_component_terms += se.axes[ax].num_events;
    /* Each event contributes one component term per candidate frequency,
     * but we export terms only for candidate frequencies.  Upper bound: */
    if (num_candidates > 0 && max_component_terms > 0)
        max_component_terms = max_component_terms * num_candidates;
    if (max_component_terms > 100000)
        max_component_terms = 100000;

    if (max_component_terms > 0)
    {
        component_freqs_hz = (float *)PULSEG_ALLOC((size_t)max_component_terms * sizeof(float));
        component_amps = (float *)PULSEG_ALLOC((size_t)max_component_terms * sizeof(float));
        component_phases_rad = (float *)PULSEG_ALLOC((size_t)max_component_terms * sizeof(float));
        component_widths_hz = (float *)PULSEG_ALLOC((size_t)max_component_terms * sizeof(float));
        component_axes = (int *)PULSEG_ALLOC((size_t)max_component_terms * sizeof(int));
        component_def_ids = (int *)PULSEG_ALLOC((size_t)max_component_terms * sizeof(int));
        component_contrib_ids = (int *)PULSEG_ALLOC((size_t)max_component_terms * sizeof(int));
        component_run_ids = (int *)PULSEG_ALLOC((size_t)max_component_terms * sizeof(int));
        if (!component_freqs_hz || !component_amps || !component_phases_rad ||
            !component_widths_hz || !component_axes || !component_def_ids ||
            !component_contrib_ids || !component_run_ids)
            goto alloc_fail;
    }

    /* --- Allocate candidate arrays --- */
    if (num_candidates > 0)
    {
        cand_freqs = (float *)PULSEG_ALLOC((size_t)num_candidates * sizeof(float));
        cand_grad_amps = (float *)PULSEG_ALLOC((size_t)num_candidates * sizeof(float));
        cand_violations = (int *)PULSEG_ALLOC((size_t)num_candidates * sizeof(int));
        ana_peak_freqs = (float *)PULSEG_ALLOC((size_t)num_candidates * sizeof(float));
        ana_peak_widths = (float *)PULSEG_ALLOC((size_t)num_candidates * sizeof(float));
        surviving_freqs_hz = (float *)PULSEG_ALLOC((size_t)num_candidates * sizeof(float));
        if (!cand_freqs || !cand_grad_amps || !cand_violations ||
            !ana_peak_freqs || !ana_peak_widths || !surviving_freqs_hz)
            goto alloc_fail;
        for (ax = 0; ax < 3; ++ax)
        {
            cand_amps[ax] = (float *)PULSEG_ALLOC((size_t)num_candidates * sizeof(float));
            cand_grad_amps_ax[ax] = (float *)PULSEG_ALLOC((size_t)num_candidates * sizeof(float));
            ana_peak_amps[ax] = (float *)PULSEG_ALLOC((size_t)num_candidates * sizeof(float));
            ana_peak_phases[ax] = (float *)PULSEG_ALLOC((size_t)num_candidates * sizeof(float));
            if (!cand_amps[ax] || !cand_grad_amps_ax[ax] ||
                !ana_peak_amps[ax] || !ana_peak_phases[ax])
                goto alloc_fail;
        }

        /* --- Fill candidate arrays --- */
        ci = 0;
        for (i = 0; i < num_freqs; ++i)
        {
            float max_ga;
            int any_ax = 0;
            for (ax = 0; ax < 3; ++ax)
                if (is_candidate_ax[ax][i])
                    any_ax = 1;
            if (!any_ax)
                continue;
            f_hz = eval_freqs[i];
            cand_freqs[ci] = f_hz;
            surviving_freqs_hz[ci] = f_hz;
            ana_peak_freqs[ci] = f_hz;

            /* FWHM estimate */
            if (num_instances > 1 && tr_duration_us > 0.0f)
            {
                float f_rep = 1.0e6f / tr_duration_us;
                ana_peak_widths[ci] = 1.2067091288032284f * (f_rep / (float)num_instances);
            }
            else
            {
                ana_peak_widths[ci] = 1.0f; /* Placeholder for K=1 */
            }

            /* Per-axis analytical amplitudes and phases */
            max_ga = 0.0f;
            for (ax = 0; ax < 3; ++ax)
            {
                float coh = coherent_mag[ax][i];
                cand_amps[ax][ci] = coh;
                ana_peak_amps[ax][ci] = coh;

                /* Phase: compute from coherent sum (with Dirichlet kernel) */
                {
                    float sum_re = 0.0f, sum_im = 0.0f;
                    int k;
                    for (k = 0; k < se.axes[ax].num_events; ++k)
                    {
                        float d_re, d_im;
                        int N = se.axes[ax].events[k].num_reps;
                        if (N < 1)
                            N = 1;
                        sa_eval_event_spectrum(f_hz, &se.axes[ax].events[k], &d_re, &d_im);
                        if (N > 1 && se.axes[ax].events[k].rep_period_us > 0.0f)
                        {
                            double phi = -2.0 * M_PI * (double)f_hz * (double)se.axes[ax].events[k].rep_period_us * 1.0e-6;
                            double half_phi = phi * 0.5;
                            double sin_half = sin(half_phi);
                            double dk_re, dk_im, new_re, new_im;
                            if (fabs(sin_half) < 1.0e-12)
                            {
                                dk_re = (double)N;
                                dk_im = 0.0;
                            }
                            else
                            {
                                double ratio = sin((double)N * half_phi) / sin_half;
                                double center = (double)(N - 1) * half_phi;
                                dk_re = ratio * cos(center);
                                dk_im = ratio * sin(center);
                            }
                            new_re = (double)d_re * dk_re - (double)d_im * dk_im;
                            new_im = (double)d_re * dk_im + (double)d_im * dk_re;
                            d_re = (float)new_re;
                            d_im = (float)new_im;
                        }
                        sum_re += d_re;
                        sum_im += d_im;
                    }
                    ana_peak_phases[ax][ci] = (float)atan2((double)sum_im, (double)sum_re);
                }

                /* G_eff per axis — only for qualifying axes */
                if (se.axes[ax].num_events > 0 && is_candidate_ax[ax][i])
                {
                    float ga = sa_eval_geff(f_hz, &se.axes[ax]);
                    cand_grad_amps_ax[ax][ci] = ga;
                    if (ga > max_ga)
                        max_ga = ga;
                }
                else
                {
                    cand_grad_amps_ax[ax][ci] = 0.0f;
                }
            }

            /* Per-axis max G_eff for the combined output field */
            cand_grad_amps[ci] = max_ga;

            /* Check against forbidden bands — per-axis independently.
             * A violation occurs if ANY single axis exceeds the band limit. */
            cand_violations[ci] = 0;
            for (b = 0; b < num_forbidden_bands; ++b)
            {
                if (f_hz >= forbidden_bands[b].freq_min_hz &&
                    f_hz <= forbidden_bands[b].freq_max_hz)
                {
                    for (ax = 0; ax < 3; ++ax)
                    {
                        if (cand_grad_amps_ax[ax][ci] > forbidden_bands[b].max_amplitude_hz_per_m)
                        {
                            cand_violations[ci] = 1;
                            break;
                        }
                    }
                }
            }

            /* Export component terms for this candidate frequency */
            for (ax = 0; ax < 3; ++ax)
            {
                int k;
                for (k = 0; k < se.axes[ax].num_events; ++k)
                {
                    float d_re, d_im, d_mag;
                    int N = se.axes[ax].events[k].num_reps;
                    if (N < 1)
                        N = 1;
                    if (num_component_terms >= max_component_terms)
                        break;
                    sa_eval_event_spectrum(f_hz, &se.axes[ax].events[k], &d_re, &d_im);
                    if (N > 1 && se.axes[ax].events[k].rep_period_us > 0.0f)
                    {
                        double phi = -2.0 * M_PI * (double)f_hz * (double)se.axes[ax].events[k].rep_period_us * 1.0e-6;
                        double half_phi = phi * 0.5;
                        double sin_half = sin(half_phi);
                        double dk_re, dk_im, new_re, new_im;
                        if (fabs(sin_half) < 1.0e-12)
                        {
                            dk_re = (double)N;
                            dk_im = 0.0;
                        }
                        else
                        {
                            double ratio = sin((double)N * half_phi) / sin_half;
                            double center = (double)(N - 1) * half_phi;
                            dk_re = ratio * cos(center);
                            dk_im = ratio * sin(center);
                        }
                        new_re = (double)d_re * dk_re - (double)d_im * dk_im;
                        new_im = (double)d_re * dk_im + (double)d_im * dk_re;
                        d_re = (float)new_re;
                        d_im = (float)new_im;
                    }
                    d_mag = (float)sqrt((double)(d_re * d_re + d_im * d_im));
                    component_freqs_hz[num_component_terms] = f_hz;
                    component_amps[num_component_terms] = d_mag;
                    component_phases_rad[num_component_terms] = (float)atan2((double)d_im, (double)d_re);
                    component_widths_hz[num_component_terms] = ana_peak_widths[ci];
                    component_axes[num_component_terms] = ax;
                    component_def_ids[num_component_terms] = se.axes[ax].events[k].def_id;
                    component_contrib_ids[num_component_terms] = k;
                    component_run_ids[num_component_terms] = 0;
                    num_component_terms++;
                }
            }

            ci++;
        }
    }

    /* --- Assign to output spectra struct (ownership transfer) --- */
    spectra->num_analytical_peaks = num_candidates;
    spectra->analytical_peak_freqs = ana_peak_freqs;
    ana_peak_freqs = NULL;
    spectra->analytical_peak_amp_gx = ana_peak_amps[0];
    ana_peak_amps[0] = NULL;
    spectra->analytical_peak_amp_gy = ana_peak_amps[1];
    ana_peak_amps[1] = NULL;
    spectra->analytical_peak_amp_gz = ana_peak_amps[2];
    ana_peak_amps[2] = NULL;
    spectra->analytical_peak_phase_gx = ana_peak_phases[0];
    ana_peak_phases[0] = NULL;
    spectra->analytical_peak_phase_gy = ana_peak_phases[1];
    ana_peak_phases[1] = NULL;
    spectra->analytical_peak_phase_gz = ana_peak_phases[2];
    ana_peak_phases[2] = NULL;
    spectra->analytical_peak_widths_hz = ana_peak_widths;
    ana_peak_widths = NULL;

    spectra->num_candidates = num_candidates;
    spectra->candidate_freqs = cand_freqs;
    cand_freqs = NULL;
    spectra->candidate_amps_gx = cand_amps[0];
    cand_amps[0] = NULL;
    spectra->candidate_amps_gy = cand_amps[1];
    cand_amps[1] = NULL;
    spectra->candidate_amps_gz = cand_amps[2];
    cand_amps[2] = NULL;
    spectra->candidate_grad_amps = cand_grad_amps;
    cand_grad_amps = NULL;
    spectra->candidate_grad_amps_gx = cand_grad_amps_ax[0];
    cand_grad_amps_ax[0] = NULL;
    spectra->candidate_grad_amps_gy = cand_grad_amps_ax[1];
    cand_grad_amps_ax[1] = NULL;
    spectra->candidate_grad_amps_gz = cand_grad_amps_ax[2];
    cand_grad_amps_ax[2] = NULL;
    spectra->candidate_violations = cand_violations;
    cand_violations = NULL;
    spectra->num_component_terms = num_component_terms;
    spectra->component_freqs_hz = component_freqs_hz;
    component_freqs_hz = NULL;
    spectra->component_amps = component_amps;
    component_amps = NULL;
    spectra->component_phases_rad = component_phases_rad;
    component_phases_rad = NULL;
    spectra->component_widths_hz = component_widths_hz;
    component_widths_hz = NULL;
    spectra->component_axes = component_axes;
    component_axes = NULL;
    spectra->component_def_ids = component_def_ids;
    component_def_ids = NULL;
    spectra->component_contrib_ids = component_contrib_ids;
    component_contrib_ids = NULL;
    spectra->component_run_ids = component_run_ids;
    component_run_ids = NULL;
    spectra->num_surviving_freqs = num_candidates;
    spectra->surviving_freqs_hz = surviving_freqs_hz;
    surviving_freqs_hz = NULL;

    /* Cleanup */
    if (eval_freqs)
        PULSEG_FREE(eval_freqs);
    for (ax = 0; ax < 3; ++ax)
    {
        if (is_candidate_ax[ax])
            PULSEG_FREE(is_candidate_ax[ax]);
        if (coherent_mag[ax])
            PULSEG_FREE(coherent_mag[ax]);
        if (incoherent_mag[ax])
            PULSEG_FREE(incoherent_mag[ax]);
    }
    sa_free_structural_events(&se);
    return PULSEG_SUCCESS;

alloc_fail:
    for (ax = 0; ax < 3; ++ax)
    {
        if (is_candidate_ax[ax])
            PULSEG_FREE(is_candidate_ax[ax]);
        if (coherent_mag[ax])
            PULSEG_FREE(coherent_mag[ax]);
        if (incoherent_mag[ax])
            PULSEG_FREE(incoherent_mag[ax]);
        if (cand_amps[ax])
            PULSEG_FREE(cand_amps[ax]);
        if (cand_grad_amps_ax[ax])
            PULSEG_FREE(cand_grad_amps_ax[ax]);
        if (ana_peak_amps[ax])
            PULSEG_FREE(ana_peak_amps[ax]);
        if (ana_peak_phases[ax])
            PULSEG_FREE(ana_peak_phases[ax]);
    }
    if (ana_peak_freqs)
        PULSEG_FREE(ana_peak_freqs);
    if (ana_peak_widths)
        PULSEG_FREE(ana_peak_widths);
    if (eval_freqs)
        PULSEG_FREE(eval_freqs);
    for (ax = 0; ax < 3; ++ax)
    {
        if (fft_bypass_set_ax[ax])
            PULSEG_FREE(fft_bypass_set_ax[ax]);
    }
    if (cand_freqs)
        PULSEG_FREE(cand_freqs);
    if (cand_grad_amps)
        PULSEG_FREE(cand_grad_amps);
    if (cand_violations)
        PULSEG_FREE(cand_violations);
    if (component_freqs_hz)
        PULSEG_FREE(component_freqs_hz);
    if (component_amps)
        PULSEG_FREE(component_amps);
    if (component_phases_rad)
        PULSEG_FREE(component_phases_rad);
    if (component_widths_hz)
        PULSEG_FREE(component_widths_hz);
    if (component_axes)
        PULSEG_FREE(component_axes);
    if (component_def_ids)
        PULSEG_FREE(component_def_ids);
    if (component_contrib_ids)
        PULSEG_FREE(component_contrib_ids);
    if (component_run_ids)
        PULSEG_FREE(component_run_ids);
    if (surviving_freqs_hz)
        PULSEG_FREE(surviving_freqs_hz);
    sa_free_structural_events(&se);
    return PULSEG_ERR_ALLOC_FAILED;
}

/* ================================================================== */
/*  Sequence spectrum (full-TR FFT)                                   */
/* ================================================================== */

/* Computes the full-TR magnitude spectrum (display-only).  The
 * canonical mechanical-resonance verdict comes from
 * sa_check_structural_violations. */
static int compute_sequence_spectrum(
    float *full_spectrum,
    int *out_num_freq_bins_full,
    float *out_freq_res_full,
    const float *waveform, int num_samples,
    float grad_raster_us, float target_spectral_res_hz,
    float max_frequency)
{
    int nfft, nfreq, output_bins_full, max_idx;
    float freq_res, max_freq, mean, fft_norm;
    float *work;
    float *cos_win;
    kiss_fft_cpx *fft_out;
    kiss_fftr_cfg cfg;
    int min_nfft, i;
    int result;

    result = PULSEG_SUCCESS;
    work = NULL;
    cos_win = NULL;
    fft_out = NULL;
    cfg = NULL;

    if (!waveform || num_samples <= 0)
        return PULSEG_ERR_INVALID_ARGUMENT;
    if (target_spectral_res_hz <= 0.0f)
        return PULSEG_ERR_INVALID_ARGUMENT;
    if (grad_raster_us <= 0.0f)
        return PULSEG_ERR_INVALID_ARGUMENT;

    min_nfft = (int)ceil((double)(1.0e6 / (grad_raster_us * target_spectral_res_hz)));
    nfft = (min_nfft < num_samples) ? (int)pulseg__next_pow2((size_t)num_samples)
                                    : (int)pulseg__next_pow2((size_t)min_nfft);
    nfreq = nfft / 2 + 1;
    freq_res = (float)(1.0e6 / (grad_raster_us * (double)nfft));
    max_freq = (max_frequency > 0.0f) ? max_frequency : (float)(5.0e5 / grad_raster_us);
    max_idx = (int)(max_freq / freq_res);
    if (max_idx >= nfreq)
        output_bins_full = nfreq;
    else if (max_idx < 1)
        output_bins_full = 1;
    else
        output_bins_full = max_idx + 1;

    work = (float *)PULSEG_ALLOC((size_t)nfft * sizeof(float));
    cos_win = (float *)PULSEG_ALLOC((size_t)num_samples * sizeof(float));
    fft_out = (kiss_fft_cpx *)PULSEG_ALLOC((size_t)nfreq * sizeof(kiss_fft_cpx));
    if (!work || !cos_win || !fft_out)
    {
        result = PULSEG_ERR_ALLOC_FAILED;
        goto fail;
    }

    cfg = kiss_fftr_alloc(nfft, 0, NULL, NULL);
    if (!cfg)
    {
        result = PULSEG_ERR_ALLOC_FAILED;
        goto fail;
    }

    for (i = 0; i < num_samples; ++i)
        cos_win[i] = (float)(0.5 * (1.0 - cos(2.0 * M_PI * (double)(i + 1) / (double)num_samples)));

    for (i = 0; i < num_samples; ++i)
        work[i] = waveform[i];
    for (i = num_samples; i < nfft; ++i)
        work[i] = 0.0f;

    mean = 0.0f;
    for (i = 0; i < num_samples; ++i)
        mean += work[i];
    mean /= (float)num_samples;
    for (i = 0; i < num_samples; ++i)
        work[i] -= mean;
    for (i = 0; i < num_samples; ++i)
        work[i] *= cos_win[i];

    kiss_fftr(cfg, work, fft_out);
    fft_norm = 1.0f / (float)nfft;
    for (i = 0; i < nfreq; ++i)
    {
        fft_out[i].r *= fft_norm;
        fft_out[i].i *= fft_norm;
    }

    if (full_spectrum)
    {
        for (i = 0; i < output_bins_full; ++i)
            full_spectrum[i] = (float)sqrt((double)(fft_out[i].r * fft_out[i].r +
                                                    fft_out[i].i * fft_out[i].i));
    }

    if (out_num_freq_bins_full)
        *out_num_freq_bins_full = output_bins_full;
    if (out_freq_res_full)
        *out_freq_res_full = freq_res;

fail:
    if (work)
        PULSEG_FREE(work);
    if (cos_win)
        PULSEG_FREE(cos_win);
    if (fft_out)
        PULSEG_FREE(fft_out);
    if (cfg)
        kiss_fftr_free(cfg);
    return result;
}

/* ================================================================== */
/*  Mechanical resonance spectra (static helper from uniform waveforms)           */
/* ================================================================== */

static int calc_mech_resonances_from_uniform(
    pulseg_mech_resonances_spectra *spectra,
    pulseg_diagnostic *diag,
    const pulseg__uniform_grad_waveforms *waveforms,
    float target_spectral_resolution_hz,
    float max_frequency_hz,
    int num_trs,
    float tr_duration_us,
    int num_forbidden_bands,
    const pulseg_forbidden_band *forbidden_bands,
    float peak_log10_threshold,
    float peak_norm_scale,
    float peak_eps,
    float peak_prominence,
    const struct pulseg_sequence_descriptor *desc,
    int start_block,
    int block_count,
    int num_avgs)
{
    pulseg_diagnostic local_diag;
    int max_samples, result;
    int num_freq_bins_full;
    float freq_res_full;
    int compute_full_spectrum;

    if (!diag)
    {
        pulseg_diagnostic_init(&local_diag);
        diag = &local_diag;
    }
    else
    {
        pulseg_diagnostic_init(diag);
    }

    if (!waveforms || !spectra)
    {
        diag->code = PULSEG_ERR_NULL_POINTER;
        return diag->code;
    }

    memset(spectra, 0, sizeof(*spectra));

    max_samples = waveforms->num_samples;
    if (max_samples <= 0)
    {
        diag->code = PULSEG_ERR_MECH_RESONANCES_NO_WAVEFORM;
        return diag->code;
    }

    spectra->freq_min_hz = 0.0f;
    spectra->freq_spacing_hz = 0.0f;
    spectra->num_freq_bins = 0;
    spectra->num_instances = num_trs;

    /* Full-TR magnitude spectrum (display-only).  Only computed when the
     * caller requests it via target_spectral_resolution_hz > 0; the
     * safety-check path passes 0.0f and skips this step entirely. */
    compute_full_spectrum = (target_spectral_resolution_hz > 0.0f);
    if (compute_full_spectrum)
    {
        /* First pass: determine output bin count + frequency resolution. */
        result = compute_sequence_spectrum(NULL,
                                           &num_freq_bins_full, &freq_res_full,
                                           waveforms->gx, waveforms->num_samples,
                                           waveforms->raster_us, target_spectral_resolution_hz,
                                           max_frequency_hz);
        if (PULSEG_FAILED(result))
        {
            pulseg_mech_resonances_spectra_free(spectra);
            diag->code = result;
            return result;
        }

        spectra->freq_spacing_hz = freq_res_full;
        spectra->num_freq_bins = num_freq_bins_full;

        spectra->spectrum_full_gx = (float *)PULSEG_ALLOC((size_t)num_freq_bins_full * sizeof(float));
        spectra->spectrum_full_gy = (float *)PULSEG_ALLOC((size_t)num_freq_bins_full * sizeof(float));
        spectra->spectrum_full_gz = (float *)PULSEG_ALLOC((size_t)num_freq_bins_full * sizeof(float));
        if (!spectra->spectrum_full_gx || !spectra->spectrum_full_gy || !spectra->spectrum_full_gz)
        {
            pulseg_mech_resonances_spectra_free(spectra);
            diag->code = PULSEG_ERR_ALLOC_FAILED;
            return diag->code;
        }

        result = compute_sequence_spectrum(
            spectra->spectrum_full_gx, NULL, NULL,
            waveforms->gx, waveforms->num_samples,
            waveforms->raster_us, target_spectral_resolution_hz,
            max_frequency_hz);
        if (PULSEG_FAILED(result))
        {
            pulseg_mech_resonances_spectra_free(spectra);
            diag->code = result;
            return result;
        }
        result = compute_sequence_spectrum(
            spectra->spectrum_full_gy, NULL, NULL,
            waveforms->gy, waveforms->num_samples,
            waveforms->raster_us, target_spectral_resolution_hz,
            max_frequency_hz);
        if (PULSEG_FAILED(result))
        {
            pulseg_mech_resonances_spectra_free(spectra);
            diag->code = result;
            return result;
        }
        result = compute_sequence_spectrum(
            spectra->spectrum_full_gz, NULL, NULL,
            waveforms->gz, waveforms->num_samples,
            waveforms->raster_us, target_spectral_resolution_hz,
            max_frequency_hz);
        if (PULSEG_FAILED(result))
        {
            pulseg_mech_resonances_spectra_free(spectra);
            diag->code = result;
            return result;
        }
    }

    /* ---- structural acoustic analysis (canonical verdict) ---- */
    if (desc && block_count > 0)
    {
        result = sa_check_structural_violations(
            spectra, desc, start_block, block_count,
            num_trs, tr_duration_us,
            num_forbidden_bands, forbidden_bands,
            peak_log10_threshold, peak_norm_scale,
            peak_eps, peak_prominence,
            num_avgs);
        if (PULSEG_FAILED(result))
        {
            pulseg_mech_resonances_spectra_free(spectra);
            diag->code = result;
            return result;
        }
    }

    diag->code = PULSEG_SUCCESS;
    return PULSEG_SUCCESS;
}

/* Select canonical TR window for safety/plotting wrappers.
 * Canonical geometry is always extracted with AMP_MAX_POS.
 * Non-degenerate prep/cooldown: full-pass canonical TR (pass-expanded).
 * Degenerate prep/cooldown: imaging TR canonical window (no pass expansion). */
static void pulseg__select_canonical_tr_window(
    const pulseg_sequence_descriptor *desc,
    int *start_block,
    int *block_count,
    int *amplitude_mode,
    int *num_instances,
    float *tr_duration_us)
{
    const pulseg_tr_descriptor *trd;
    int has_nd_prep, has_nd_cool, n;

    trd = &desc->tr_descriptor;
    has_nd_prep = (trd->num_prep_blocks > 0 && !trd->degenerate_prep);
    has_nd_cool = (trd->num_cooldown_blocks > 0 && !trd->degenerate_cooldown);

    if (has_nd_prep || has_nd_cool)
    {
        *start_block = 0;
        *block_count = desc->pass_len;
        *amplitude_mode = PULSEG_AMP_MAX_POS;
        *num_instances = (desc->num_passes > 1) ? desc->num_passes : 1;

        *tr_duration_us = 0.0f;
        for (n = 0; n < desc->pass_len; ++n)
        {
            const pulseg_block_table_element *bte = &desc->block_table[n];
            const pulseg_block_definition *bdef = &desc->block_definitions[bte->id];
            *tr_duration_us += (bte->duration_us >= 0)
                                   ? (float)bte->duration_us
                                   : (float)bdef->duration_us;
        }
        return;
    }

    *start_block = trd->num_prep_blocks + trd->imaging_tr_start;
    *block_count = trd->tr_size;
    *amplitude_mode = PULSEG_AMP_MAX_POS;
    *num_instances = trd->num_trs;
    *tr_duration_us = trd->tr_duration_us;
}

/* Find unique shot-index patterns across pass-expanded canonical windows.
 * Returns number of unique pass patterns, 0 on allocation/shape failure.
 * Caller must free out arrays when return > 0. */
int pulseg__find_unique_shot_passes(
    const pulseg_sequence_descriptor *desc,
    int **out_unique_pass_indices,
    int **out_pass_group_labels)
{
    int num_passes, pass_size, num_cols;
    int *rows;
    int *unique_defs;
    int *event_table;
    int *result_indices;
    int p, pos, col, bt_pos, block_idx, raw_id;
    int num_unique, g, match, c;
    const pulseg_block_table_element *bte;
    const pulseg_grad_table_element *gte;

    *out_unique_pass_indices = NULL;
    *out_pass_group_labels = NULL;

    num_passes = (desc->num_passes > 1) ? desc->num_passes : 1;
    pass_size = (num_passes > 0) ? (desc->scan_table_len / num_passes) : 0;
    if (num_passes <= 0 || pass_size <= 0 || !desc->scan_table_block_idx)
        return 0;

    num_cols = pass_size * 3;
    rows = (int *)PULSEG_ALLOC((size_t)num_passes * (size_t)num_cols * sizeof(int));
    unique_defs = (int *)PULSEG_ALLOC((size_t)num_passes * sizeof(int));
    event_table = (int *)PULSEG_ALLOC((size_t)num_passes * sizeof(int));
    if (!rows || !unique_defs || !event_table)
    {
        if (rows)
            PULSEG_FREE(rows);
        if (unique_defs)
            PULSEG_FREE(unique_defs);
        if (event_table)
            PULSEG_FREE(event_table);
        return 0;
    }

    for (p = 0; p < num_passes; ++p)
    {
        col = 0;
        for (pos = 0; pos < pass_size; ++pos)
        {
            bt_pos = p * pass_size + pos;
            if (bt_pos < 0 || bt_pos >= desc->scan_table_len)
            {
                rows[p * num_cols + col++] = -1;
                rows[p * num_cols + col++] = -1;
                rows[p * num_cols + col++] = -1;
                continue;
            }
            block_idx = desc->scan_table_block_idx[bt_pos];
            if (block_idx < 0 || block_idx >= desc->num_blocks)
            {
                rows[p * num_cols + col++] = -1;
                rows[p * num_cols + col++] = -1;
                rows[p * num_cols + col++] = -1;
                continue;
            }

            bte = &desc->block_table[block_idx];

            raw_id = bte->gx_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size)
            {
                gte = &desc->grad_table[raw_id];
                rows[p * num_cols + col] = gte->shot_index;
            }
            else
                rows[p * num_cols + col] = -1;
            col++;

            raw_id = bte->gy_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size)
            {
                gte = &desc->grad_table[raw_id];
                rows[p * num_cols + col] = gte->shot_index;
            }
            else
                rows[p * num_cols + col] = -1;
            col++;

            raw_id = bte->gz_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size)
            {
                gte = &desc->grad_table[raw_id];
                rows[p * num_cols + col] = gte->shot_index;
            }
            else
                rows[p * num_cols + col] = -1;
            col++;
        }
    }

    num_unique = 0;
    for (p = 0; p < num_passes; ++p)
    {
        match = -1;
        for (g = 0; g < num_unique; ++g)
        {
            int rep = unique_defs[g];
            int equal = 1;
            for (c = 0; c < num_cols; ++c)
            {
                if (rows[p * num_cols + c] != rows[rep * num_cols + c])
                {
                    equal = 0;
                    break;
                }
            }
            if (equal)
            {
                match = g;
                break;
            }
        }
        if (match >= 0)
        {
            event_table[p] = match;
        }
        else
        {
            unique_defs[num_unique] = p;
            event_table[p] = num_unique;
            num_unique++;
        }
    }

    result_indices = (int *)PULSEG_ALLOC((size_t)num_unique * sizeof(int));
    if (!result_indices)
    {
        PULSEG_FREE(rows);
        PULSEG_FREE(unique_defs);
        PULSEG_FREE(event_table);
        return 0;
    }
    for (g = 0; g < num_unique; ++g)
        result_indices[g] = unique_defs[g];

    *out_unique_pass_indices = result_indices;
    *out_pass_group_labels = event_table;

    PULSEG_FREE(rows);
    PULSEG_FREE(unique_defs);
    return num_unique;
}

/* ================================================================== */
/*  Acoustic spectra (public wrapper)                                 */
/* ================================================================== */

int pulseg_calc_mech_resonances(const pulseg_collection *coll,
                                   pulseg_mech_resonances_spectra *spectra,
                                   pulseg_diagnostic *diag,
                                   int subseq_idx,
                                   int canonical_tr_idx,
                                   const pulseg_opts *opts,
                                   float target_resolution_hz,
                                   float max_freq_hz,
                                   int num_forbidden_bands,
                                   const pulseg_forbidden_band *forbidden_bands)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_descriptor *trd;
    pulseg__uniform_grad_waveforms uw;
    pulseg_diagnostic local_diag;
    int rc, start_block, block_count, amplitude_mode, num_instances;
    int sa_start_block, sa_block_count, num_avgs;
    int *block_order;
    int has_nd_prep, has_nd_cool;
    float tr_duration_us;
    float peak_log10_threshold;
    float peak_norm_scale;
    float peak_eps;
    float peak_prominence;

    memset(&uw, 0, sizeof(uw));
    block_order = NULL;
    if (!diag)
    {
        pulseg_diagnostic_init(&local_diag);
        diag = &local_diag;
    }
    else
    {
        pulseg_diagnostic_init(diag);
    }
    if (!coll || !spectra)
    {
        diag->code = PULSEG_ERR_NULL_POINTER;
        return diag->code;
    }
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
    {
        diag->code = PULSEG_ERR_INVALID_ARGUMENT;
        return diag->code;
    }

    peak_log10_threshold = PULSEG_PEAK_LOG10_THRESHOLD_DEFAULT;
    peak_norm_scale = PULSEG_PEAK_NORM_SCALE_DEFAULT;
    peak_eps = PULSEG_PEAK_EPS_DEFAULT;
    peak_prominence = PULSEG_PEAK_PROMINENCE_DEFAULT;
    if (opts)
    {
        peak_log10_threshold = opts->peak_log10_threshold;
        peak_norm_scale = opts->peak_norm_scale;
        peak_eps = opts->peak_eps;
        peak_prominence = opts->peak_prominence;
    }

    desc = &coll->descriptors[subseq_idx];
    trd = &desc->tr_descriptor;
    /* Select the canonical TR window for the given canonical_tr_idx */
    if (canonical_tr_idx < 0 || canonical_tr_idx >= desc->tr_descriptor.num_trs)
    {
        diag->code = PULSEG_ERR_INVALID_ARGUMENT;
        return diag->code;
    }
    /* Use a new helper to select the correct window for the canonical_tr_idx */
    pulseg__select_canonical_tr_window_idx(
        desc,
        canonical_tr_idx,
        &start_block,
        &block_count,
        &amplitude_mode,
        &num_instances,
        &tr_duration_us);

    sa_start_block = start_block;
    sa_block_count = block_count;
    num_avgs = 1;

    has_nd_prep = (trd->num_prep_blocks > 0 && !trd->degenerate_prep);
    has_nd_cool = (trd->num_cooldown_blocks > 0 && !trd->degenerate_cooldown);
    if (has_nd_prep || has_nd_cool)
    {
        num_avgs = (desc->num_averages > 1) ? desc->num_averages : 1;
        rc = pulseg__build_pass_expanded_block_order(
            desc,
            start_block,
            &block_order,
            &block_count,
            &tr_duration_us);
        if (PULSEG_FAILED(rc))
        {
            diag->code = rc;
            return rc;
        }
        start_block = 0;
    }

    rc = pulseg__get_gradient_waveforms_range(desc, &uw, diag,
                                                 start_block,
                                                 block_count,
                                                 amplitude_mode,
                                                 NULL,
                                                 0,
                                                 block_order);
    if (PULSEG_FAILED(rc))
    {
        if (block_order)
            PULSEG_FREE(block_order);
        return rc;
    }
    rc = calc_mech_resonances_from_uniform(spectra, diag, &uw,
                                           target_resolution_hz, max_freq_hz,
                                           num_instances,
                                           tr_duration_us,
                                           num_forbidden_bands, forbidden_bands,
                                           peak_log10_threshold, peak_norm_scale, peak_eps,
                                           peak_prominence,
                                           desc, sa_start_block, sa_block_count, num_avgs);
    pulseg__uniform_grad_waveforms_free(&uw);
    if (block_order)
        PULSEG_FREE(block_order);
    return rc;
}

/* ================================================================== */
/*  PNS                                                               */
/* ================================================================== */

/* FIX: outputs before inputs */
static int build_pns_kernel(
    float **kernel, int *kernel_len,
    float dt_us, const pulseg_pns_params *params)
{
    int n, i;
    float c_s, dt_s, s_min, tau, denom;
    float *k;

    if (params->chronaxie_us <= 0.0f)
        return PULSEG_ERR_PNS_INVALID_CHRONAXIE;
    if (params->rheobase_hz_per_m_per_s <= 0.0f)
        return PULSEG_ERR_PNS_INVALID_RHEOBASE;
    if (params->alpha <= 0.0f)
        return PULSEG_ERR_PNS_INVALID_PARAMS;

    c_s = params->chronaxie_us * 1e-6f;
    dt_s = dt_us * 1e-6f;
    s_min = params->rheobase_hz_per_m_per_s / params->alpha;

    n = (int)(PNS_KERNEL_DURATION_FACTOR * c_s / dt_s) + 1;
    k = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    if (!k)
        return PULSEG_ERR_ALLOC_FAILED;

    for (i = 0; i < n; ++i)
    {
        tau = (float)i * dt_s;
        denom = (c_s + tau) * (c_s + tau);
        k[i] = (dt_s / s_min) * (c_s / denom);
    }

    *kernel = k;
    *kernel_len = n;
    return PULSEG_SUCCESS;
}

/* FIX: output before inputs, C89-compliant declarations */
static void compute_slew_rate(
    float *slew_out,
    const float *waveform, int num_samples,
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

/* Returns 1 if every sample of waveform[0..num_samples) is exactly zero.
 * Used to skip per-axis convolution work (PNS) for an axis that carries
 * no gradient (e.g. Gx/Gy on a slice-select-only readout). */
static int pulseg__waveform_is_zero(const float *waveform, int num_samples)
{
    int i;

    if (!waveform)
        return 1;
    for (i = 0; i < num_samples; ++i)
    {
        if (waveform[i] != 0.0f)
            return 0;
    }
    return 1;
}

/* FIX: outputs then in-out then scratch then inputs */
static int process_pns_axis_circular(
    float *pns_axis, float *pns_total,
    float *pns_store,
    float *padded_waveform, float *slew_rate, float *pns_conv,
    const float *waveform, int num_samples,
    const float *kernel, int kernel_len,
    float grad_raster_us, float gamma_hz_per_tesla,
    int full_output_len)
{
    int i, padded_len, slew_len, rc;

    (void)full_output_len;

    if (num_samples <= 0 || !waveform)
        return PULSEG_SUCCESS;

    padded_len = num_samples + kernel_len;
    slew_len = padded_len - 1;

    for (i = 0; i < num_samples; ++i)
        padded_waveform[i] = waveform[i];
    for (i = 0; i < kernel_len; ++i)
        padded_waveform[num_samples + i] = waveform[i % num_samples];

    compute_slew_rate(slew_rate, padded_waveform, padded_len, grad_raster_us, gamma_hz_per_tesla);

    rc = pulseg__calc_convolution_fft(pns_conv, slew_rate, slew_len, kernel, kernel_len);
    if (PULSEG_FAILED(rc))
        return rc;

    for (i = 0; i < slew_len; ++i)
    {
        pns_axis[i] = pns_conv[i] * 100.0f;
        pns_total[i] += pns_conv[i] * pns_conv[i];
    }

    if (pns_store)
    {
        for (i = 0; i < slew_len; ++i)
            pns_store[i] = pns_axis[i];
    }
    return PULSEG_SUCCESS;
}

static int calc_pns_from_uniform(
    pulseg_pns_result *result,
    pulseg_diagnostic *diag,
    float gamma_hz_per_tesla,
    const pulseg__uniform_grad_waveforms *waveforms,
    const pulseg_pns_params *params)
{
    pulseg_diagnostic local_diag;
    int max_samples, padded_len, slew_len, full_output_len;
    int kernel_len, i;
    float *kernel;
    float *padded;
    float *slew;
    float *conv;
    float *axis;
    float *pns_x;
    float *pns_y;
    float *pns_z;
    float *pns_tot;
    int rc;

    kernel = NULL;
    padded = NULL;
    slew = NULL;
    conv = NULL;
    axis = NULL;
    pns_x = NULL;
    pns_y = NULL;
    pns_z = NULL;
    pns_tot = NULL;
    rc = PULSEG_SUCCESS;

    if (!diag)
    {
        pulseg_diagnostic_init(&local_diag);
        diag = &local_diag;
    }
    else
    {
        pulseg_diagnostic_init(diag);
    }

    if (!waveforms || !params || !result)
    {
        diag->code = PULSEG_ERR_NULL_POINTER;
        return diag->code;
    }

    /* Vendor dispatch.  Currently only the GE Healthcare exponential
     * (chronaxie / rheobase / alpha) model is implemented in this
     * library.  Other vendors (e.g. Siemens SAFE) require a different
     * convolution kernel and threshold metric; those branches are
     * extension points for downstream developers.
     *
     * vendor == 0 ("unspecified") is treated as GEHC for backward
     * compatibility with callers that don't set the field. */
    if (params->vendor != 0 && params->vendor != PULSEG_VENDOR_GEHC)
    {
        diag->code = PULSEG_ERR_NOT_IMPLEMENTED;
        return diag->code;
    }

    memset(result, 0, sizeof(*result));

    max_samples = waveforms->num_samples;
    if (max_samples <= 1)
    {
        diag->code = PULSEG_ERR_PNS_NO_WAVEFORM;
        return diag->code;
    }

    rc = build_pns_kernel(&kernel, &kernel_len, waveforms->raster_us, params);
    if (PULSEG_FAILED(rc))
    {
        diag->code = rc;
        return rc;
    }

    padded_len = max_samples + kernel_len;
    slew_len = padded_len - 1;
    full_output_len = slew_len;

    padded = (float *)PULSEG_ALLOC((size_t)padded_len * sizeof(float));
    slew = (float *)PULSEG_ALLOC((size_t)slew_len * sizeof(float));
    conv = (float *)PULSEG_ALLOC((size_t)slew_len * sizeof(float));
    axis = (float *)PULSEG_ALLOC((size_t)full_output_len * sizeof(float));
    pns_tot = (float *)PULSEG_ALLOC((size_t)full_output_len * sizeof(float));
    if (!padded || !slew || !conv || !axis || !pns_tot)
    {
        rc = PULSEG_ERR_ALLOC_FAILED;
        goto fail;
    }
    for (i = 0; i < full_output_len; ++i)
        pns_tot[i] = 0.0f;

    pns_x = (float *)PULSEG_ALLOC((size_t)full_output_len * sizeof(float));
    pns_y = (float *)PULSEG_ALLOC((size_t)full_output_len * sizeof(float));
    pns_z = (float *)PULSEG_ALLOC((size_t)full_output_len * sizeof(float));
    if (!pns_x || !pns_y || !pns_z)
    {
        rc = PULSEG_ERR_ALLOC_FAILED;
        goto fail;
    }
    for (i = 0; i < full_output_len; ++i)
    {
        pns_x[i] = 0.0f;
        pns_y[i] = 0.0f;
        pns_z[i] = 0.0f;
    }

    /* X -- skip the FFT convolution entirely for a silent axis; pns_x/
     * pns_tot stay at their zero-initialized value, which is the correct
     * result for a channel with no gradient. */
    if (!pulseg__waveform_is_zero(waveforms->gx, waveforms->num_samples))
    {
        rc = process_pns_axis_circular(axis, pns_tot,
                                       pns_x,
                                       padded, slew, conv,
                                       waveforms->gx, waveforms->num_samples,
                                       kernel, kernel_len,
                                       waveforms->raster_us, gamma_hz_per_tesla,
                                       full_output_len);
        if (PULSEG_FAILED(rc))
            goto fail;
    }

    /* Y */
    if (!pulseg__waveform_is_zero(waveforms->gy, waveforms->num_samples))
    {
        rc = process_pns_axis_circular(axis, pns_tot,
                                       pns_y,
                                       padded, slew, conv,
                                       waveforms->gy, waveforms->num_samples,
                                       kernel, kernel_len,
                                       waveforms->raster_us, gamma_hz_per_tesla,
                                       full_output_len);
        if (PULSEG_FAILED(rc))
            goto fail;
    }

    /* Z */
    if (!pulseg__waveform_is_zero(waveforms->gz, waveforms->num_samples))
    {
        rc = process_pns_axis_circular(axis, pns_tot,
                                       pns_z,
                                       padded, slew, conv,
                                       waveforms->gz, waveforms->num_samples,
                                       kernel, kernel_len,
                                       waveforms->raster_us, gamma_hz_per_tesla,
                                       full_output_len);
        if (PULSEG_FAILED(rc))
            goto fail;
    }

    for (i = 0; i < full_output_len; ++i)
    {
        pns_tot[i] = 100.0f * (float)sqrt((double)pns_tot[i]);
    }

    result->num_samples = full_output_len;
    result->slew_x_hz_per_m_per_s = pns_x;
    pns_x = NULL;
    result->slew_y_hz_per_m_per_s = pns_y;
    pns_y = NULL;
    result->slew_z_hz_per_m_per_s = pns_z;
    pns_z = NULL;

    rc = PULSEG_SUCCESS;
    diag->code = rc;

fail:
    if (kernel)
        PULSEG_FREE(kernel);
    if (padded)
        PULSEG_FREE(padded);
    if (slew)
        PULSEG_FREE(slew);
    if (conv)
        PULSEG_FREE(conv);
    if (axis)
        PULSEG_FREE(axis);
    if (pns_tot)
        PULSEG_FREE(pns_tot);
    if (pns_x)
        PULSEG_FREE(pns_x);
    if (pns_y)
        PULSEG_FREE(pns_y);
    if (pns_z)
        PULSEG_FREE(pns_z);
    return rc;
}

/* ================================================================== */
/*  PNS (public wrapper)                                              */
/* ================================================================== */

int pulseg_calc_pns(const pulseg_collection *coll,
                       pulseg_pns_result *result,
                       pulseg_diagnostic *diag,
                       int subseq_idx,
                       int canonical_tr_idx,
                       const pulseg_opts *opts,
                       const pulseg_pns_params *params)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_descriptor *trd;
    pulseg__uniform_grad_waveforms uw;
    pulseg_diagnostic local_diag;
    int rc, start_block, block_count, amplitude_mode, num_instances;
    int *block_order;
    int has_nd_prep, has_nd_cool;
    float tr_duration_us;

    (void)opts;
    memset(&uw, 0, sizeof(uw));
    block_order = NULL;
    if (!diag)
    {
        pulseg_diagnostic_init(&local_diag);
        diag = &local_diag;
    }
    else
    {
        pulseg_diagnostic_init(diag);
    }
    if (!coll || !result || !params)
    {
        diag->code = PULSEG_ERR_NULL_POINTER;
        return diag->code;
    }
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
    {
        diag->code = PULSEG_ERR_INVALID_ARGUMENT;
        return diag->code;
    }
    desc = &coll->descriptors[subseq_idx];
    trd = &desc->tr_descriptor;
    if (canonical_tr_idx < 0 || canonical_tr_idx >= desc->tr_descriptor.num_trs)
    {
        diag->code = PULSEG_ERR_INVALID_ARGUMENT;
        return diag->code;
    }
    pulseg__select_canonical_tr_window_idx(
        desc,
        canonical_tr_idx,
        &start_block,
        &block_count,
        &amplitude_mode,
        &num_instances,
        &tr_duration_us);
    (void)num_instances;
    (void)tr_duration_us;

    has_nd_prep = (trd->num_prep_blocks > 0 && !trd->degenerate_prep);
    has_nd_cool = (trd->num_cooldown_blocks > 0 && !trd->degenerate_cooldown);
    if (has_nd_prep || has_nd_cool)
    {
        rc = pulseg__build_pass_expanded_block_order(
            desc,
            start_block,
            &block_order,
            &block_count,
            NULL);
        if (PULSEG_FAILED(rc))
        {
            diag->code = rc;
            return rc;
        }
        start_block = 0;
    }

    rc = pulseg__get_gradient_waveforms_range(desc, &uw, diag,
                                                 start_block,
                                                 block_count,
                                                 amplitude_mode,
                                                 NULL,
                                                 0,
                                                 block_order);
    if (PULSEG_FAILED(rc))
    {
        if (block_order)
            PULSEG_FREE(block_order);
        return rc;
    }
    rc = calc_pns_from_uniform(result, diag, opts->gamma_hz_per_t, &uw, params);
    pulseg__uniform_grad_waveforms_free(&uw);
    if (block_order)
        PULSEG_FREE(block_order);
    return rc;
}

/* ================================================================== */
/*  PNS result free                                                   */
/* ================================================================== */

void pulseg_pns_result_free(pulseg_pns_result *r)
{
    if (!r)
        return;
    if (r->slew_x_hz_per_m_per_s)
    {
        PULSEG_FREE(r->slew_x_hz_per_m_per_s);
        r->slew_x_hz_per_m_per_s = NULL;
    }
    if (r->slew_y_hz_per_m_per_s)
    {
        PULSEG_FREE(r->slew_y_hz_per_m_per_s);
        r->slew_y_hz_per_m_per_s = NULL;
    }
    if (r->slew_z_hz_per_m_per_s)
    {
        PULSEG_FREE(r->slew_z_hz_per_m_per_s);
        r->slew_z_hz_per_m_per_s = NULL;
    }
    r->num_samples = 0;
}

/* ================================================================== */
/*  Collection-level safety check                                     */
/* ================================================================== */

int check_max_grad(
    const pulseg_collection *coll,
    pulseg_diagnostic *diag,
    const pulseg_opts *opts)
{
    int s, b, raw_id;
    int worst_subseq, worst_block;
    float gx_amp, gy_amp, gz_amp, gsos, gsos_max, limit_sq, hz_per_mt;
    const pulseg_sequence_descriptor *desc;
    const pulseg_block_table_element *bte;

    if (!coll || !opts)
    {
        if (diag)
        {
            pulseg_diagnostic_init(diag);
            diag->code = PULSEG_ERR_NULL_POINTER;
        }
        return PULSEG_ERR_NULL_POINTER;
    }
    if (diag)
        pulseg_diagnostic_init(diag);

    /* ---- max gradient amplitude (GSOS) check ----
     *
     * `opts->max_grad_hz_per_m` is the per-axis limit already derated by sqrt(3)
     * upstream (in pulserver_init_opts) so that, under an arbitrary rotation,
     * no single physical axis exceeds the scanner's hardware gmax. The vector
     * magnitude (GSOS) of the *unrotated* waveform, however, is the quantity
     * that bounds every rotated axis component; hence the proper GSOS bound is
     * sqrt(3) * per-axis-derated = physical gmax. Compare GSOS^2 against
     * 3 * (per-axis derated)^2.
     */
    gsos_max = 0.0f;
    limit_sq = 3.0f * opts->max_grad_hz_per_m * opts->max_grad_hz_per_m;
    worst_subseq = 0;
    worst_block = 0;

    for (s = 0; s < coll->num_subsequences; ++s)
    {
        desc = &coll->descriptors[s];
        for (b = 0; b < desc->num_blocks; ++b)
        {
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
            if (gsos > gsos_max)
            {
                gsos_max = gsos;
                worst_subseq = s;
                worst_block = b;
            }
        }
    }

    if (limit_sq > 0.0f && gsos_max > limit_sq)
    {
        hz_per_mt = opts->gamma_hz_per_t * 0.001f;
        if (diag)
        {
            float physical_limit_hz_per_m =
                (float)sqrt(3.0) * opts->max_grad_hz_per_m;
            diag->code = PULSEG_ERR_MAX_GRAD_EXCEEDED;
            pulseg__diag_printf(diag,
                                   "amp=%.2fmT/m>%.2fmT/m,s=%d,b=%d",
                                   (double)((float)sqrt((double)gsos_max) / hz_per_mt),
                                   (double)(physical_limit_hz_per_m / hz_per_mt),
                                   worst_subseq, worst_block);
        }
        return PULSEG_ERR_MAX_GRAD_EXCEEDED;
    }

    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Gradient continuity check (cursor dry-run over n repetitions)     */
/* ================================================================== */

int check_grad_continuity(
    pulseg_collection *coll,
    pulseg_diagnostic *diag,
    const pulseg_opts *opts)
{
    pulseg_block_cursor saved_cursor;
    const pulseg_sequence_descriptor *desc;
    const pulseg_block_table_element *bte;
    const pulseg_block_definition *bdef;
    const pulseg_grad_definition *gdef;
    const pulseg_grad_table_element *gte;
    int n, raw_id, rot_id, status, cur_seq;
    int grad_def_ids[3];
    int shot_idx[3];
    float amp[3], first_val[3], last_val[3];
    float first_phys[3], last_phys[3], prev_phys[3];
    float max_allowed, grad_raster_s, step, hz_per_mt;

    if (!coll || !opts)
    {
        if (diag)
        {
            pulseg_diagnostic_init(diag);
            diag->code = PULSEG_ERR_NULL_POINTER;
        }
        return PULSEG_ERR_NULL_POINTER;
    }
    if (diag)
        pulseg_diagnostic_init(diag);

    /* save cursor state */
    saved_cursor = coll->block_cursor;
    coll->block_cursor.sequence_index = 0;
    coll->block_cursor.scan_table_position = 0;
    coll->block_cursor.from_last_reset = 0;

    prev_phys[0] = 0.0f;
    prev_phys[1] = 0.0f;
    prev_phys[2] = 0.0f;
    cur_seq = 0;
    status = PULSEG_CURSOR_BLOCK;

    desc = &coll->descriptors[0];
    grad_raster_s = desc->grad_raster_us * 1e-6f;
    max_allowed = opts->max_slew_hz_per_m_per_s * grad_raster_s;

    while (status != PULSEG_CURSOR_DONE)
    {
        /* detect subsequence change */
        if (coll->block_cursor.sequence_index != cur_seq)
        {
            /* end-of-subsequence: prev must ramp to zero */
            for (n = 0; n < 3; ++n)
            {
                step = prev_phys[n];
                if (step < 0.0f)
                    step = -step;
                if (step > max_allowed)
                {
                    hz_per_mt = opts->gamma_hz_per_t * 0.001f;
                    if (diag)
                    {
                        diag->code = PULSEG_ERR_GRAD_DISCONTINUITY;
                        pulseg__diag_printf(diag,
                                               "step=%.2fmT/m>%.2fmT/m,a=%d,bnd=1",
                                               (double)(step / hz_per_mt), (double)(max_allowed / hz_per_mt), n);
                    }
                    coll->block_cursor = saved_cursor;
                    return PULSEG_ERR_GRAD_DISCONTINUITY;
                }
            }

            cur_seq = coll->block_cursor.sequence_index;
            prev_phys[0] = 0.0f;
            prev_phys[1] = 0.0f;
            prev_phys[2] = 0.0f;

            desc = &coll->descriptors[cur_seq];
            grad_raster_s = desc->grad_raster_us * 1e-6f;
            max_allowed = opts->max_slew_hz_per_m_per_s * grad_raster_s;
        }

        /* read current block */
        {
            int bt_idx = desc->scan_table_block_idx[coll->block_cursor.scan_table_position];
            bte = &desc->block_table[bt_idx];
        }
        bdef = &desc->block_definitions[bte->id];

        /* grad table: amplitude + shot_index */
        grad_def_ids[0] = bdef->gx_id;
        grad_def_ids[1] = bdef->gy_id;
        grad_def_ids[2] = bdef->gz_id;

        raw_id = bte->gx_id;
        if (raw_id >= 0 && raw_id < desc->grad_table_size)
        {
            gte = &desc->grad_table[raw_id];
            amp[0] = gte->amplitude;
            shot_idx[0] = gte->shot_index;
        }
        else
        {
            amp[0] = 0.0f;
            shot_idx[0] = 0;
        }

        raw_id = bte->gy_id;
        if (raw_id >= 0 && raw_id < desc->grad_table_size)
        {
            gte = &desc->grad_table[raw_id];
            amp[1] = gte->amplitude;
            shot_idx[1] = gte->shot_index;
        }
        else
        {
            amp[1] = 0.0f;
            shot_idx[1] = 0;
        }

        raw_id = bte->gz_id;
        if (raw_id >= 0 && raw_id < desc->grad_table_size)
        {
            gte = &desc->grad_table[raw_id];
            amp[2] = gte->amplitude;
            shot_idx[2] = gte->shot_index;
        }
        else
        {
            amp[2] = 0.0f;
            shot_idx[2] = 0;
        }

        /* first_value / last_value from grad definitions, scaled by amplitude */
        for (n = 0; n < 3; ++n)
        {
            if (grad_def_ids[n] >= 0 && grad_def_ids[n] < desc->num_unique_grads)
            {
                gdef = &desc->grad_definitions[grad_def_ids[n]];
                first_val[n] = gdef->first_value[shot_idx[n]] * amp[n];
                last_val[n] = gdef->last_value[shot_idx[n]] * amp[n];
            }
            else
            {
                first_val[n] = 0.0f;
                last_val[n] = 0.0f;
            }
        }

        /* transform logical -> physical */
        rot_id = bte->rotation_id;
        if (rot_id >= 0 && rot_id < desc->num_rotations)
        {
            pulseg__apply_rotation(first_phys, desc->rotation_matrices[rot_id], first_val, 1);
            pulseg__apply_rotation(last_phys, desc->rotation_matrices[rot_id], last_val, 1);
        }
        else
        {
            first_phys[0] = first_val[0];
            first_phys[1] = first_val[1];
            first_phys[2] = first_val[2];
            last_phys[0] = last_val[0];
            last_phys[1] = last_val[1];
            last_phys[2] = last_val[2];
        }

        /* continuity check */
        for (n = 0; n < 3; ++n)
        {
            step = first_phys[n] - prev_phys[n];
            if (step < 0.0f)
                step = -step;
            if (step > max_allowed)
            {
                hz_per_mt = opts->gamma_hz_per_t * 0.001f;
                if (diag)
                {
                    diag->code = PULSEG_ERR_GRAD_DISCONTINUITY;
                    pulseg__diag_printf(diag,
                                           "step=%.2fmT/m>%.2fmT/m,a=%d,p=%d",
                                           (double)(step / hz_per_mt), (double)(max_allowed / hz_per_mt),
                                           n, coll->block_cursor.scan_table_position);
                }
                coll->block_cursor = saved_cursor;
                return PULSEG_ERR_GRAD_DISCONTINUITY;
            }
        }

        prev_phys[0] = last_phys[0];
        prev_phys[1] = last_phys[1];
        prev_phys[2] = last_phys[2];

        /* advance cursor */
        status = pulseg_cursor_next(coll);
    }

    /* final subsequence trailing edge */
    for (n = 0; n < 3; ++n)
    {
        step = prev_phys[n];
        if (step < 0.0f)
            step = -step;
        if (step > max_allowed)
        {
            hz_per_mt = opts->gamma_hz_per_t * 0.001f;
            if (diag)
            {
                diag->code = PULSEG_ERR_GRAD_DISCONTINUITY;
                pulseg__diag_printf(diag,
                                       "step=%.2fmT/m>%.2fmT/m,a=%d,tail=1",
                                       (double)(step / hz_per_mt), (double)(max_allowed / hz_per_mt), n);
            }
            coll->block_cursor = saved_cursor;
            return PULSEG_ERR_GRAD_DISCONTINUITY;
        }
    }

    coll->block_cursor = saved_cursor;
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Max slew rate check (per unique block definition)                 */
/* ================================================================== */

int check_max_slew(
    const pulseg_collection *coll,
    pulseg_diagnostic *diag,
    const pulseg_opts *opts)
{
    /* ---- max slew rate (GSOS) check ----
     *
     * `opts->max_slew_hz_per_m_per_s` is the per-axis limit already derated by
     * sqrt(3) upstream (in pulserver_init_opts), mirroring max_grad.  The GSOS
     * bound (vector magnitude of per-axis slew rates) therefore corresponds to
     * sqrt(3) * derated = physical slew limit.  Compare GSOS_slew^2 against
     * 3 * (derated)^2.
     *
     * Slew per axis at each block = slew_rate_normalised * amplitude, where
     * slew_rate_normalised (1/s) comes from grad_definitions and amplitude from
     * grad_table (the per-block-instance value).  This mirrors check_max_grad
     * which also iterates block_table for per-instance amplitudes.
     */
    int s, b, n, raw_id, def_idx, shot_idx;
    float slew_sq, slew_sq_max, limit_sq, amp;
    float axis_slew[3];
    int raw_ids[3];
    int worst_subseq, worst_block;
    const pulseg_sequence_descriptor *desc;
    const pulseg_block_table_element *bte;
    const pulseg_grad_definition *gdef;

    if (!coll || !opts)
    {
        if (diag)
        {
            pulseg_diagnostic_init(diag);
            diag->code = PULSEG_ERR_NULL_POINTER;
        }
        return PULSEG_ERR_NULL_POINTER;
    }
    if (diag)
        pulseg_diagnostic_init(diag);

    slew_sq_max = 0.0f;
    limit_sq = 3.0f * opts->max_slew_hz_per_m_per_s * opts->max_slew_hz_per_m_per_s;
    worst_subseq = 0;
    worst_block = 0;

    for (s = 0; s < coll->num_subsequences; ++s)
    {
        desc = &coll->descriptors[s];

        for (b = 0; b < desc->num_blocks; ++b)
        {
            bte = &desc->block_table[b];
            raw_ids[0] = bte->gx_id;
            raw_ids[1] = bte->gy_id;
            raw_ids[2] = bte->gz_id;

            for (n = 0; n < 3; ++n)
            {
                axis_slew[n] = 0.0f;
                raw_id = raw_ids[n];
                if (raw_id < 0 || raw_id >= desc->grad_table_size)
                    continue;

                def_idx = desc->grad_table[raw_id].id;
                shot_idx = desc->grad_table[raw_id].shot_index;
                amp = desc->grad_table[raw_id].amplitude;
                if (amp < 0.0f)
                    amp = -amp;

                if (def_idx < 0 || def_idx >= desc->num_unique_grads)
                    continue;

                gdef = &desc->grad_definitions[def_idx];
                if (shot_idx >= 0 && shot_idx < gdef->num_shots)
                    axis_slew[n] = gdef->slew_rate[shot_idx] * amp;
            }

            slew_sq = axis_slew[0] * axis_slew[0] + axis_slew[1] * axis_slew[1] + axis_slew[2] * axis_slew[2];
            if (slew_sq > slew_sq_max)
            {
                slew_sq_max = slew_sq;
                worst_subseq = s;
                worst_block = b;
            }
        }
    }

    if (slew_sq_max > limit_sq)
    {
        if (diag)
        {
            float physical_limit = (float)sqrt(3.0) * opts->max_slew_hz_per_m_per_s;
            diag->code = PULSEG_ERR_MAX_SLEW_EXCEEDED;
            pulseg__diag_printf(diag,
                                   "slew_rss=%.2f>%.2fHz/m/s,s=%d,b=%d",
                                   (double)sqrt((double)slew_sq_max),
                                   (double)physical_limit,
                                   worst_subseq, worst_block);
        }
        return PULSEG_ERR_MAX_SLEW_EXCEEDED;
    }

    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Gradient presence (skip gate for gradient-only safety checks)     */
/* ================================================================== */

/* Returns 1 if any block in the collection has a nonzero gx/gy/gz
 * amplitude, 0 if every gradient channel is silent throughout (e.g. an
 * RF-only or pure-delay sequence). Mirrors the block_table walk in
 * check_max_grad() but stops at the first nonzero sample. */
static int pulseg__collection_has_gradient(const pulseg_collection *coll)
{
    int s, b, raw_id;
    const pulseg_sequence_descriptor *desc;
    const pulseg_block_table_element *bte;

    for (s = 0; s < coll->num_subsequences; ++s)
    {
        desc = &coll->descriptors[s];
        for (b = 0; b < desc->num_blocks; ++b)
        {
            bte = &desc->block_table[b];

            raw_id = bte->gx_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size &&
                desc->grad_table[raw_id].amplitude != 0.0f)
                return 1;

            raw_id = bte->gy_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size &&
                desc->grad_table[raw_id].amplitude != 0.0f)
                return 1;

            raw_id = bte->gz_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size &&
                desc->grad_table[raw_id].amplitude != 0.0f)
                return 1;
        }
    }
    return 0;
}

/* ================================================================== */
/*  Safety check                                                      */
/* ================================================================== */
int pulseg_check_safety(
    pulseg_collection *coll,
    pulseg_diagnostic *diag,
    const pulseg_opts *opts,
    int num_forbidden_bands,
    const pulseg_forbidden_band *forbidden_bands,
    const pulseg_pns_params *pns_params,
    float pns_threshold_percent)
{
    int rc, s, u, i;
    int ci, b;
    int num_unique_trs;
    int *unique_tr_indices;
    int *tr_group_labels;
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_descriptor *trd;
    pulseg__uniform_grad_waveforms uw;
    pulseg_mech_resonances_spectra spectra;
    pulseg_pns_result pns_result;
    int start_block, block_count, amplitude_mode, num_instances;
    int sa_start_block, sa_block_count, num_avgs;
    int *block_order;
    int has_nd_prep, has_nd_cool;
    float tr_duration_us;
    float pns_combined, max_pns;
    float cf_hz, ca_hz_per_m;

    if (!coll || !opts)
    {
        if (diag)
        {
            pulseg_diagnostic_init(diag);
            diag->code = PULSEG_ERR_NULL_POINTER;
        }
        return PULSEG_ERR_NULL_POINTER;
    }
    if (diag)
        pulseg_diagnostic_init(diag);

    /* No gradient event anywhere in the collection (e.g. an RF-only or
     * pure-delay sequence): every check below operates on gx/gy/gz, so
     * there is nothing to validate. Skip continuity/slew/mech-resonance/
     * PNS entirely rather than running them over a silent waveform that
     * may span the full sequence duration (RF/SAR safety is evaluated
     * separately and is unaffected by this skip). */
    if (!pulseg__collection_has_gradient(coll))
        return PULSEG_SUCCESS;

    /* ---- 1. max gradient amplitude ---- */
    rc = check_max_grad(coll, diag, opts);
    if (PULSEG_FAILED(rc))
        return rc;

    /* ---- 2. gradient continuity ---- */
    rc = check_grad_continuity(coll, diag, opts);
    if (PULSEG_FAILED(rc))
        return rc;

    /* ---- 3. max slew rate ---- */
    rc = check_max_slew(coll, diag, opts);
    if (PULSEG_FAILED(rc))
        return rc;

    /* ---- 4. per-subsequence canonical-TR acoustic + PNS ---- */
    for (s = 0; s < coll->num_subsequences; ++s)
    {
        desc = &coll->descriptors[s];
        trd = &desc->tr_descriptor;
        unique_tr_indices = NULL;
        tr_group_labels = NULL;

        pulseg__select_canonical_tr_window(
            desc,
            &start_block,
            &block_count,
            &amplitude_mode,
            &num_instances,
            &tr_duration_us);
        (void)amplitude_mode;
        sa_start_block = start_block;
        sa_block_count = block_count;
        num_avgs = 1;
        block_order = NULL;
        has_nd_prep = (trd->num_prep_blocks > 0 && !trd->degenerate_prep);
        has_nd_cool = (trd->num_cooldown_blocks > 0 && !trd->degenerate_cooldown);
        if (has_nd_prep || has_nd_cool)
        {
            num_avgs = (desc->num_averages > 1) ? desc->num_averages : 1;
            rc = pulseg__build_pass_expanded_block_order(
                desc,
                start_block,
                &block_order,
                &block_count,
                &tr_duration_us);
            if (PULSEG_FAILED(rc))
            {
                if (unique_tr_indices)
                    PULSEG_FREE(unique_tr_indices);
                if (tr_group_labels)
                    PULSEG_FREE(tr_group_labels);
                if (block_order)
                    PULSEG_FREE(block_order);
                return rc;
            }
            start_block = 0;
        }

        /* Evaluate one canonical TR per shot-ID combination. */
        unique_tr_indices = NULL;
        tr_group_labels = NULL;
        if ((trd->num_prep_blocks > 0 && !trd->degenerate_prep) ||
            (trd->num_cooldown_blocks > 0 && !trd->degenerate_cooldown))
        {
            num_unique_trs = pulseg__find_unique_shot_passes(
                desc, &unique_tr_indices, &tr_group_labels);
        }
        else
        {
            num_unique_trs = pulseg__find_unique_shot_trs(
                desc, &unique_tr_indices, &tr_group_labels);
        }
        if (num_unique_trs <= 0)
            num_unique_trs = 1;

        for (u = 0; u < num_unique_trs; ++u)
        {
            memset(&uw, 0, sizeof(uw));
            rc = pulseg__get_gradient_waveforms_range(desc, &uw, diag,
                                                         start_block,
                                                         block_count,
                                                         PULSEG_AMP_MAX_POS,
                                                         tr_group_labels, u, block_order);
            if (PULSEG_FAILED(rc))
            {
                if (unique_tr_indices)
                    PULSEG_FREE(unique_tr_indices);
                if (tr_group_labels)
                    PULSEG_FREE(tr_group_labels);
                if (block_order)
                    PULSEG_FREE(block_order);
                return rc;
            }

            if (num_forbidden_bands > 0)
            {
                memset(&spectra, 0, sizeof(spectra));
                rc = calc_mech_resonances_from_uniform(
                    &spectra, diag, &uw,
                    0.0f, 0.0f,
                    num_instances,
                    tr_duration_us,
                    num_forbidden_bands, forbidden_bands,
                    opts->peak_log10_threshold,
                    opts->peak_norm_scale,
                    opts->peak_eps,
                    opts->peak_prominence,
                    desc, sa_start_block, sa_block_count, num_avgs);

                /* Safety path: fail fast on first violating candidate.
                 * Pattern: for each candidate, scan union of all bands. */
                if (!PULSEG_FAILED(rc) && spectra.num_candidates > 0 &&
                    spectra.candidate_freqs &&
                    spectra.candidate_grad_amps_gx &&
                    spectra.candidate_grad_amps_gy &&
                    spectra.candidate_grad_amps_gz)
                {
                    float *cga[3];
                    int axi;
                    cga[0] = spectra.candidate_grad_amps_gx;
                    cga[1] = spectra.candidate_grad_amps_gy;
                    cga[2] = spectra.candidate_grad_amps_gz;
                    for (ci = 0; ci < spectra.num_candidates; ++ci)
                    {
                        cf_hz = spectra.candidate_freqs[ci];

                        for (b = 0; b < num_forbidden_bands; ++b)
                        {
                            if (cf_hz < forbidden_bands[b].freq_min_hz ||
                                cf_hz > forbidden_bands[b].freq_max_hz)
                                continue;
                            for (axi = 0; axi < 3; ++axi)
                            {
                                ca_hz_per_m = cga[axi][ci];
                                if (ca_hz_per_m > forbidden_bands[b].max_amplitude_hz_per_m)
                                {
                                    rc = PULSEG_ERR_MECH_RESONANCES_VIOLATION;
                                    if (diag)
                                    {
                                        diag->code = rc;
                                        pulseg__diag_printf(
                                            diag,
                                            "f=%.2fHz,a=%.2f>%.2fHz/m,ax=%d,ss=%d,tr=%d",
                                            (double)cf_hz, (double)ca_hz_per_m,
                                            (double)forbidden_bands[b].max_amplitude_hz_per_m,
                                            axi, s, u);
                                    }
                                    break;
                                }
                            }
                            if (PULSEG_FAILED(rc))
                                break;
                        }
                        if (PULSEG_FAILED(rc))
                            break;
                    }
                }

                pulseg_mech_resonances_spectra_free(&spectra);
                if (PULSEG_FAILED(rc))
                {
                    pulseg__uniform_grad_waveforms_free(&uw);
                    if (unique_tr_indices)
                        PULSEG_FREE(unique_tr_indices);
                    if (tr_group_labels)
                        PULSEG_FREE(tr_group_labels);
                    if (block_order)
                        PULSEG_FREE(block_order);
                    return rc;
                }
            }

            if (pns_params)
            {
                memset(&pns_result, 0, sizeof(pns_result));
                rc = calc_pns_from_uniform(
                    &pns_result, diag, opts->gamma_hz_per_t,
                    &uw, pns_params);
                if (!PULSEG_FAILED(rc) && pns_result.num_samples > 0)
                {
                    max_pns = 0.0f;
                    for (i = 0; i < pns_result.num_samples; ++i)
                    {
                        pns_combined = 0.0f;
                        if (pns_result.slew_x_hz_per_m_per_s)
                            pns_combined += pns_result.slew_x_hz_per_m_per_s[i] * pns_result.slew_x_hz_per_m_per_s[i];
                        if (pns_result.slew_y_hz_per_m_per_s)
                            pns_combined += pns_result.slew_y_hz_per_m_per_s[i] * pns_result.slew_y_hz_per_m_per_s[i];
                        if (pns_result.slew_z_hz_per_m_per_s)
                            pns_combined += pns_result.slew_z_hz_per_m_per_s[i] * pns_result.slew_z_hz_per_m_per_s[i];
                        pns_combined = (float)sqrt((double)pns_combined);
                        if (pns_combined > max_pns)
                            max_pns = pns_combined;
                    }
                    if (max_pns > pns_threshold_percent)
                        rc = PULSEG_ERR_PNS_THRESHOLD_EXCEEDED;
                }
                pulseg_pns_result_free(&pns_result);
                if (PULSEG_FAILED(rc))
                {
                    pulseg__uniform_grad_waveforms_free(&uw);
                    if (unique_tr_indices)
                        PULSEG_FREE(unique_tr_indices);
                    if (tr_group_labels)
                        PULSEG_FREE(tr_group_labels);
                    if (block_order)
                        PULSEG_FREE(block_order);
                    return rc;
                }
            }

            pulseg__uniform_grad_waveforms_free(&uw);
        }

        if (unique_tr_indices)
            PULSEG_FREE(unique_tr_indices);
        if (tr_group_labels)
            PULSEG_FREE(tr_group_labels);
        if (block_order)
            PULSEG_FREE(block_order);
    }

    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Vendor-neutral safety entrypoint (load + check + free)            */
/* ================================================================== */

int pulseg_check_safety_from_file(
    pulseg_diagnostic *diag,
    const char *seq_path,
    const pulseg_opts *opts,
    int num_forbidden_bands,
    const pulseg_forbidden_band *forbidden_bands,
    const pulseg_pns_params *pns_params,
    float pns_threshold_percent)
{
    pulseg_collection *coll = NULL;
    int rc;

    if (diag)
        pulseg_diagnostic_init(diag);
    if (!seq_path || !opts)
    {
        if (diag)
            diag->code = PULSEG_ERR_NULL_POINTER;
        return PULSEG_ERR_NULL_POINTER;
    }

    rc = pulseg_read(
        &coll, diag, seq_path, opts,
        /*cache_binary*/ 0,
        /*verify_signature*/ 0,
        /*parse_labels*/ 0,
        /*num_averages*/ 1);
    if (PULSEG_FAILED(rc))
    {
        if (coll)
            pulseg_collection_free(coll);
        return rc;
    }

    rc = pulseg_check_safety(
        coll, diag, opts,
        num_forbidden_bands, forbidden_bands,
        pns_params, pns_threshold_percent);

    pulseg_collection_free(coll);
    return rc;
}
