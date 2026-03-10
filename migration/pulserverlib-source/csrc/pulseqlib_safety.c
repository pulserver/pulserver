/* pulseqlib_safety.c -- safety checks, acoustic analysis, PNS, segment timing
 *
 * Public functions:
 *   pulseqlib_check_safety
 *   pulseqlib_calc_acoustic_spectra   / _free
 *   pulseqlib_calc_pns               / _free
 *
 * Internal:
 *   pulseqlib__calc_segment_timing
 *   check_max_grad / check_grad_continuity / check_max_slew
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "pulseqlib_internal.h"
#include "pulseqlib_methods.h"

#include "external_kiss_fft.h"
#include "external_kiss_fftr.h"

/* ================================================================== */
/*  File-scope constants                                               */
/* ================================================================== */
#define PEAK_LOG10_THRESHOLD    2.25f
#define PEAK_NORM_SCALE         10.0f
#define PEAK_EPS                1e-30f

#define PNS_KERNEL_DURATION_FACTOR 20.0f


/* ================================================================== */
/*  K-space trajectory from uniform gradient waveforms                */
/* ================================================================== */

/*
 * Computes k-space trajectory by cumulative trapezoidal integration of
 * gradient waveforms (already on uniform raster).
 *
 * Output arrays (kx, ky, kz, krss) must be caller-allocated with at
 * least waveforms->gx.num_samples elements.  dt_us is returned for
 * convenience.
 */
static int compute_kspace_trajectory(
    const pulseqlib__uniform_grad_waveforms* waveforms,
    float* kx, float* ky, float* kz, float* krss,
    float* dt_us,
    const int* refocus_samples, int num_refocus)
{
    int i, n, r;
    float dt_s;
    float cum_x, cum_y, cum_z;
    float v;

    if (!waveforms || !waveforms->gx || waveforms->num_samples < 2)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;

    n = waveforms->num_samples;
    *dt_us = waveforms->raster_us;
    dt_s   = (*dt_us) * 1e-6f;

    /* cumulative trapezoidal integration */
    cum_x = 0.0f; cum_y = 0.0f; cum_z = 0.0f;
    kx[0] = 0.0f; ky[0] = 0.0f; kz[0] = 0.0f;

    r = 0;  /* index into refocus_samples */

    for (i = 1; i < n; ++i) {
        cum_x += 0.5f * (waveforms->gx[i - 1] + waveforms->gx[i]) * dt_s;
        cum_y += 0.5f * (waveforms->gy[i - 1] + waveforms->gy[i]) * dt_s;
        cum_z += 0.5f * (waveforms->gz[i - 1] + waveforms->gz[i]) * dt_s;

        /* negate k at refocusing RF isocenter (180 deg pulse) */
        if (refocus_samples && r < num_refocus &&
            i == refocus_samples[r]) {
            cum_x = -cum_x;
            cum_y = -cum_y;
            cum_z = -cum_z;
            r++;
        }

        kx[i] = cum_x;
        ky[i] = cum_y;
        kz[i] = cum_z;
    }

    /* RSS magnitude */
    for (i = 0; i < n; ++i) {
        v = kx[i] * kx[i] + ky[i] * ky[i] + kz[i] * kz[i];
        krss[i] = (v > 0.0f) ? (float)sqrt((double)v) : 0.0f;
    }

    return PULSEQLIB_SUCCESS;
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
    return PULSEQLIB_SUCCESS;
}

/* ================================================================== */
/*  Compute segment timing anchors                                    */
/* ================================================================== */

int pulseqlib__calc_segment_timing(
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
    int rf_raw;
    const pulseqlib_rf_definition* rdef;
    const pulseqlib_adc_definition* adef;
    const pulseqlib_rf_table_element* rte;
    pulseqlib_segment_rf_anchor* rf_arr;
    pulseqlib_segment_adc_anchor* adc_arr;
    int rf_def_id, adc_def_id;
    float adc_dur_us;

    /* k-space trajectory variables */
    pulseqlib__uniform_grad_waveforms min_waveforms;
    float *kx, *ky, *kz, *krss;
    int *kzero_indices;
    int num_kzero, n_samples;
    float dt_us, k_threshold;

    /* refocusing RF detection variables */
    int *refocus_samples;
    int  num_refocus;

    /* ADC-to-kzero mapping variables */
    int a, s, kz_sample;
    float seg_time_offset;
    float kzero_in_adc;
    int pos_in_tr, num_prep;
    int tr_size;
    int adc_raster_start, adc_raster_end, min_raster_idx;
    float min_krss_val;

    int has_kspace;

    if (!diag) { pulseqlib_diagnostic_init(&local_diag); diag = &local_diag; }

    memset(&min_waveforms, 0, sizeof(min_waveforms));
    kx = NULL; ky = NULL; kz = NULL; krss = NULL;
    kzero_indices = NULL;
    refocus_samples = NULL;
    num_refocus = 0;
    num_kzero = 0;
    n_samples = 0;
    dt_us = 0.0f;
    has_kspace = 0;

    if (!desc || desc->num_unique_segments <= 0)
        return PULSEQLIB_SUCCESS;

    num_prep = desc->tr_descriptor.num_prep_blocks;
    tr_size  = desc->tr_descriptor.tr_size;

    /* ---- Step A: build min-amplitude k-space trajectory ---- */
    if (tr_size > 0) {
        result = pulseqlib__get_gradient_waveforms_range(desc, &min_waveforms, diag,
            num_prep, tr_size, PULSEQLIB_AMP_ZERO_VAR, NULL, 0);

        if (!PULSEQLIB_FAILED(result) && min_waveforms.num_samples >= 2) {
            n_samples = min_waveforms.num_samples;

            /* ---- Step A.1: find refocusing RF isocenters in TR ---- */
            /*
             * Walk all blocks in the main TR range and identify
             * refocusing pulses.  If rf_use is tagged in the file,
             * use it directly; otherwise auto-detect from flip
             * angle (|flip| within 10% of 180 deg).
             */
            {
                float rf_t_accum = 0.0f;
                int   rf_cap = 0, rb;

                /* first pass: count */
                for (rb = 0; rb < tr_size; ++rb) {
                    int bi = num_prep + rb;
                    if (bi < 0 || bi >= desc->num_blocks) continue;
                    bte = &desc->block_table[bi];
                    rf_raw = bte->rf_id;
                    if (rf_raw >= 0 && rf_raw < desc->rf_table_size) {
                        rte = &desc->rf_table[rf_raw];
                        rf_def_id = rte->id;
                        if (rf_def_id >= 0 && rf_def_id < desc->num_unique_rfs) {
                            int use = rte->rf_use;
                            if (use == PULSEQLIB_RF_USE_UNKNOWN) {
                                /* auto-detect: actual_flip = base_flip *
                                 * |amplitude| / base_amplitude */
                                rdef = &desc->rf_definitions[rf_def_id];
                                if (rdef->stats.base_amplitude_hz > 0.0f) {
                                    float ratio = (float)fabs((double)rte->amplitude) /
                                                  rdef->stats.base_amplitude_hz;
                                    float actual_flip = rdef->stats.flip_angle_deg * ratio;
                                    if (actual_flip > 162.0f && actual_flip < 198.0f)
                                        use = PULSEQLIB_RF_USE_REFOCUSING;
                                }
                            }
                            if (use == PULSEQLIB_RF_USE_REFOCUSING)
                                rf_cap++;
                        }
                    }
                }

                /* second pass: collect isocenter sample indices */
                if (rf_cap > 0) {
                    refocus_samples = (int*)PULSEQLIB_ALLOC(
                        (size_t)rf_cap * sizeof(int));
                }
                if (refocus_samples) {
                    rf_t_accum = 0.0f;
                    num_refocus = 0;
                    for (rb = 0; rb < tr_size; ++rb) {
                        int bi = num_prep + rb;
                        if (bi < 0 || bi >= desc->num_blocks) continue;
                        bte  = &desc->block_table[bi];
                        bdef = &desc->block_definitions[bte->id];
                        block_dur_us = (bte->duration_us >= 0)
                            ? (float)bte->duration_us
                            : (float)bdef->duration_us;

                        rf_raw = bte->rf_id;
                        if (rf_raw >= 0 && rf_raw < desc->rf_table_size) {
                            rte = &desc->rf_table[rf_raw];
                            rf_def_id = rte->id;
                            if (rf_def_id >= 0 && rf_def_id < desc->num_unique_rfs) {
                                int use = rte->rf_use;
                                if (use == PULSEQLIB_RF_USE_UNKNOWN) {
                                    rdef = &desc->rf_definitions[rf_def_id];
                                    if (rdef->stats.base_amplitude_hz > 0.0f) {
                                        float ratio = (float)fabs((double)rte->amplitude) /
                                                      rdef->stats.base_amplitude_hz;
                                        float actual_flip = rdef->stats.flip_angle_deg * ratio;
                                        if (actual_flip > 162.0f && actual_flip < 198.0f)
                                            use = PULSEQLIB_RF_USE_REFOCUSING;
                                    }
                                }
                                if (use == PULSEQLIB_RF_USE_REFOCUSING) {
                                    float iso_us;
                                    int   iso_sample;
                                    rdef = &desc->rf_definitions[rf_def_id];
                                    iso_us = rf_t_accum + (float)rdef->delay +
                                             (float)rdef->stats.isodelay_us;
                                    iso_sample = (int)(iso_us / min_waveforms.raster_us + 0.5f);
                                    if (iso_sample < 0) iso_sample = 0;
                                    if (iso_sample >= n_samples) iso_sample = n_samples - 1;
                                    refocus_samples[num_refocus++] = iso_sample;
                                }
                            }
                        }
                        rf_t_accum += block_dur_us;
                    }
                }
            }

            /* ---- Step A.2: compute k-space trajectory ---- */
            kx   = (float*)PULSEQLIB_ALLOC((size_t)n_samples * sizeof(float));
            ky   = (float*)PULSEQLIB_ALLOC((size_t)n_samples * sizeof(float));
            kz   = (float*)PULSEQLIB_ALLOC((size_t)n_samples * sizeof(float));
            krss = (float*)PULSEQLIB_ALLOC((size_t)n_samples * sizeof(float));
            if (kx && ky && kz && krss) {
                result = compute_kspace_trajectory(&min_waveforms,
                    kx, ky, kz, krss, &dt_us,
                    refocus_samples, num_refocus);
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

        /* count RF and ADC events (use block_definitions, not
           block_table, because start_block may point to a dummy TR
           instance where ADC/RF events are inactive)                */
        rf_count  = 0;
        adc_count = 0;
        for (blk = 0; blk < seg->num_blocks; ++blk) {
            block_idx = seg->start_block + blk;
            if (block_idx < 0 || block_idx >= desc->num_blocks) continue;
            bte  = &desc->block_table[block_idx];
            bdef = &desc->block_definitions[bte->id];
            if (bdef->rf_id >= 0)  rf_count++;
            if (bdef->adc_id >= 0) adc_count++;
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
                    int use;
                    rdef = &desc->rf_definitions[rf_def_id];
                    rf_arr[rf_count].block_offset = blk;
                    rf_arr[rf_count].start_us     = t_accum + (float)rdef->delay;
                    rf_arr[rf_count].end_us        = t_accum + (float)rdef->delay +
                                                     rdef->stats.duration_us;
                    rf_arr[rf_count].isocenter_us   = t_accum + (float)rdef->delay +
                                                      (float)rdef->stats.isodelay_us;
                    rf_arr[rf_count].base_amplitude_hz = rte->amplitude;

                    /* rf_use: from file tag, or auto-detect from flip angle */
                    use = rte->rf_use;
                    if (use == PULSEQLIB_RF_USE_UNKNOWN &&
                        rdef->stats.base_amplitude_hz > 0.0f) {
                        float ratio = (float)fabs((double)rte->amplitude) /
                                      rdef->stats.base_amplitude_hz;
                        float actual_flip = rdef->stats.flip_angle_deg * ratio;
                        if (actual_flip > 162.0f && actual_flip < 198.0f)
                            use = PULSEQLIB_RF_USE_REFOCUSING;
                        else
                            use = PULSEQLIB_RF_USE_EXCITATION;
                    }
                    rf_arr[rf_count].rf_use = use;
                    rf_count++;
                }
            }

            /* ADC anchor — use block_definition (bdef->adc_id is the
               unique ADC def index, not a raw event ref) because
               start_block may point to a dummy TR without ADC.      */
            adc_def_id = bdef->adc_id;
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

                    /* refine kzero via per-ADC local krss minimum */
                    if (has_kspace && krss && n_samples > 0 &&
                        seg->start_block >= num_prep &&
                        seg->start_block < num_prep + tr_size) {

                        /* convert ADC window to raster indices */
                        adc_raster_start = (int)((seg_time_offset +
                            adc_arr[adc_count].start_us) / dt_us);
                        adc_raster_end   = (int)((seg_time_offset +
                            adc_arr[adc_count].end_us) / dt_us);
                        if (adc_raster_start < 0) adc_raster_start = 0;
                        if (adc_raster_end >= n_samples)
                            adc_raster_end = n_samples - 1;

                        /* find raster index of minimum krss in window */
                        min_raster_idx = adc_raster_start;
                        min_krss_val   = krss[adc_raster_start];
                        for (a = adc_raster_start + 1;
                             a <= adc_raster_end; ++a) {
                            if (krss[a] < min_krss_val) {
                                min_krss_val   = krss[a];
                                min_raster_idx = a;
                            }
                        }

                        /* convert global raster index to ADC sample */
                        kzero_in_adc = (float)min_raster_idx * dt_us -
                            (seg_time_offset + adc_arr[adc_count].start_us);
                        kz_sample = (int)(kzero_in_adc /
                            ((float)adef->dwell_time * 1e-3f));
                        if (kz_sample < 0) kz_sample = 0;
                        if (kz_sample >= adef->num_samples)
                            kz_sample = adef->num_samples - 1;
                        adc_arr[adc_count].kzero_index = kz_sample;
                        adc_arr[adc_count].kzero_us    =
                            (float)min_raster_idx * dt_us - seg_time_offset;
                    }

                    adc_count++;
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
    if (refocus_samples) PULSEQLIB_FREE(refocus_samples);
    pulseqlib__uniform_grad_waveforms_free(&min_waveforms);

    return PULSEQLIB_SUCCESS;

timing_fail:
    if (kx)   PULSEQLIB_FREE(kx);
    if (ky)   PULSEQLIB_FREE(ky);
    if (kz)   PULSEQLIB_FREE(kz);
    if (krss) PULSEQLIB_FREE(krss);
    if (kzero_indices) PULSEQLIB_FREE(kzero_indices);
    if (refocus_samples) PULSEQLIB_FREE(refocus_samples);
    pulseqlib__uniform_grad_waveforms_free(&min_waveforms);
    return PULSEQLIB_ERR_ALLOC_FAILED;
}

/* ================================================================== */
/*  Acoustic spectra free                                             */
/* ================================================================== */

void pulseqlib_acoustic_spectra_free(pulseqlib_acoustic_spectra* s)
{
    if (!s) return;

    if (s->spectrogram_gx)    PULSEQLIB_FREE(s->spectrogram_gx);
    if (s->spectrogram_gy)    PULSEQLIB_FREE(s->spectrogram_gy);
    if (s->spectrogram_gz)    PULSEQLIB_FREE(s->spectrogram_gz);
    if (s->peaks_gx)          PULSEQLIB_FREE(s->peaks_gx);
    if (s->peaks_gy)          PULSEQLIB_FREE(s->peaks_gy);
    if (s->peaks_gz)          PULSEQLIB_FREE(s->peaks_gz);
    if (s->spectrum_full_gx)  PULSEQLIB_FREE(s->spectrum_full_gx);
    if (s->spectrum_full_gy)  PULSEQLIB_FREE(s->spectrum_full_gy);
    if (s->spectrum_full_gz)  PULSEQLIB_FREE(s->spectrum_full_gz);
    if (s->peaks_full_gx)    PULSEQLIB_FREE(s->peaks_full_gx);
    if (s->peaks_full_gy)    PULSEQLIB_FREE(s->peaks_full_gy);
    if (s->peaks_full_gz)    PULSEQLIB_FREE(s->peaks_full_gz);
    if (s->spectrum_seq_gx)   PULSEQLIB_FREE(s->spectrum_seq_gx);
    if (s->spectrum_seq_gy)   PULSEQLIB_FREE(s->spectrum_seq_gy);
    if (s->spectrum_seq_gz)   PULSEQLIB_FREE(s->spectrum_seq_gz);
    if (s->peaks_seq_gx)     PULSEQLIB_FREE(s->peaks_seq_gx);
    if (s->peaks_seq_gy)     PULSEQLIB_FREE(s->peaks_seq_gy);
    if (s->peaks_seq_gz)     PULSEQLIB_FREE(s->peaks_seq_gz);

    memset(s, 0, sizeof(*s));
}

/* ================================================================== */
/*  Acoustic support structure                                        */
/* ================================================================== */

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
    return PULSEQLIB_SUCCESS;

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
    return PULSEQLIB_SUCCESS;
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
    return PULSEQLIB_SUCCESS;
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
    int** out_peaks,
    const float* spectrum, const float* frequencies, int num_freq_bins,
    float max_envelope,
    const pulseqlib_forbidden_band* bands, int num_bands)
{
    int* peaks;
    int i, b;
    float freq;

    (void)max_envelope;

    if (num_bands <= 0 || !bands) {
        if (out_peaks) *out_peaks = NULL;
        return PULSEQLIB_SUCCESS;
    }

    peaks = (int*)PULSEQLIB_ALLOC((size_t)num_freq_bins * sizeof(int));
    if (!peaks) return PULSEQLIB_ERR_ALLOC_FAILED;

    detect_resonances(peaks, spectrum, num_freq_bins);

    /* Check peaks against forbidden bands (informational) */
    for (i = 0; i < num_freq_bins; ++i) {
        if (!peaks[i]) continue;
        if (!frequencies) continue;
        freq = frequencies[i];
        for (b = 0; b < num_bands; ++b) {
            if (freq >= bands[b].freq_min_hz && freq <= bands[b].freq_max_hz)
                break;
        }
    }

    if (out_peaks) *out_peaks = peaks;
    else           PULSEQLIB_FREE(peaks);

    return PULSEQLIB_SUCCESS;
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
                result = check_acoustic_violations(out_peaks ? &win_peaks : NULL,
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
    return PULSEQLIB_SUCCESS;
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
    int* seq_peaks;
    int result;

    result      = PULSEQLIB_SUCCESS;
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
        result = check_acoustic_violations(out_seq_peaks,
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
/*  Acoustic spectra (static helper from uniform waveforms)           */
/* ================================================================== */

static int calc_acoustic_spectra_from_uniform(
    pulseqlib_acoustic_spectra* spectra,
    pulseqlib_diagnostic* diag,
    const pulseqlib__uniform_grad_waveforms* waveforms,
    int target_window_size,
    float target_spectral_resolution_hz,
    float max_frequency_hz,
    int num_trs,
    float tr_duration_us,
    int num_forbidden_bands,
    const pulseqlib_forbidden_band* forbidden_bands)
{
    acoustic_support sup;
    pulseqlib_diagnostic local_diag;
    int max_samples, result, output_size, padded_len;
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
    if (max_samples <= 0) { diag->code = PULSEQLIB_ERR_ACOUSTIC_NO_WAVEFORM; return diag->code; }

    result = acoustic_support_init(&sup, max_samples, target_window_size,
                                   target_spectral_resolution_hz,
                                   waveforms->raster_us, max_frequency_hz);
    if (PULSEQLIB_FAILED(result)) { diag->code = result; return result; }

    spectra->freq_min_hz    = 0.0f;
    spectra->freq_spacing_hz = sup.freq_resolution;
    spectra->num_freq_bins  = sup.output_freq_bins;
    spectra->num_windows    = sup.num_windows;

    output_size = sup.num_windows * sup.output_freq_bins;

    spectra->spectrogram_gx = (float*)PULSEQLIB_ALLOC((size_t)output_size * sizeof(float));
    spectra->spectrogram_gy = (float*)PULSEQLIB_ALLOC((size_t)output_size * sizeof(float));
    spectra->spectrogram_gz = (float*)PULSEQLIB_ALLOC((size_t)output_size * sizeof(float));
    spectra->peaks_gx = (int*)PULSEQLIB_ALLOC((size_t)output_size * sizeof(int));
    spectra->peaks_gy = (int*)PULSEQLIB_ALLOC((size_t)output_size * sizeof(int));
    spectra->peaks_gz = (int*)PULSEQLIB_ALLOC((size_t)output_size * sizeof(int));

    if (!spectra->spectrogram_gx || !spectra->spectrogram_gy || !spectra->spectrogram_gz ||
        !spectra->peaks_gx || !spectra->peaks_gy || !spectra->peaks_gz) {
        pulseqlib_acoustic_spectra_free(spectra);
        acoustic_support_free(&sup);
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED; return diag->code;
    }

    memset(spectra->spectrogram_gx, 0, (size_t)output_size * sizeof(float));
    memset(spectra->spectrogram_gy, 0, (size_t)output_size * sizeof(float));
    memset(spectra->spectrogram_gz, 0, (size_t)output_size * sizeof(float));
    memset(spectra->peaks_gx, 0, (size_t)output_size * sizeof(int));
    memset(spectra->peaks_gy, 0, (size_t)output_size * sizeof(int));
    memset(spectra->peaks_gz, 0, (size_t)output_size * sizeof(int));

    /* padded length */
    padded_len = (max_samples <= sup.nwin) ? max_samples
                 : ((max_samples + sup.nwin - 1) / sup.nwin) * sup.nwin;

    /* sliding window: Gx */
    if (waveforms->num_samples > 0) {
        result = compute_sliding_window_spectra(&sup,
            spectra->spectrogram_gx, NULL, spectra->peaks_gx,
            waveforms->gx, NULL,
            waveforms->num_samples, padded_len, 0,
            forbidden_bands, num_forbidden_bands);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_acoustic_spectra_free(spectra);
            acoustic_support_free(&sup);
            diag->code = result; return result;
        }
    }
    /* sliding window: Gy */
    if (waveforms->num_samples > 0) {
        result = compute_sliding_window_spectra(&sup,
            spectra->spectrogram_gy, NULL, spectra->peaks_gy,
            waveforms->gy, NULL,
            waveforms->num_samples, padded_len, 0,
            forbidden_bands, num_forbidden_bands);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_acoustic_spectra_free(spectra);
            acoustic_support_free(&sup);
            diag->code = result; return result;
        }
    }
    /* sliding window: Gz */
    if (waveforms->num_samples > 0) {
        result = compute_sliding_window_spectra(&sup,
            spectra->spectrogram_gz, NULL, spectra->peaks_gz,
            waveforms->gz, NULL,
            waveforms->num_samples, padded_len, 0,
            forbidden_bands, num_forbidden_bands);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_acoustic_spectra_free(spectra);
            acoustic_support_free(&sup);
            diag->code = result; return result;
        }
    }

    acoustic_support_free(&sup);

    /* ---- sequence spectra ---- */
    if (num_trs > 1 && tr_duration_us > 0.0f) {
        fundamental_freq = 1.0e6f / tr_duration_us;
    } else {
        fundamental_freq = 0.0f;
    }

    /* Gx (first call: determines sizes) */
    if (waveforms->num_samples > 0) {
        result = compute_sequence_spectrum(NULL,
            (fundamental_freq > 0.0f) ? &seq_spec_gx : NULL,
            (fundamental_freq > 0.0f) ? &seq_freqs   : NULL,
            &num_freq_bins_full, &freq_res_full, &num_freq_bins_seq, NULL,
            (fundamental_freq > 0.0f) ? &spectra->peaks_seq_gx : NULL,
            waveforms->gx, waveforms->num_samples,
            waveforms->raster_us, target_spectral_resolution_hz,
            max_frequency_hz, fundamental_freq, num_trs,
            forbidden_bands, num_forbidden_bands);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_acoustic_spectra_free(spectra);
            diag->code = result; return result;
        }
    } else {
        result = compute_sequence_spectrum(NULL, NULL, NULL,
            &num_freq_bins_full, &freq_res_full, NULL, NULL, NULL,
            waveforms->gy ? waveforms->gy : waveforms->gz,
            max_samples, waveforms->raster_us, target_spectral_resolution_hz,
            max_frequency_hz, 0.0f, num_trs,
            forbidden_bands, num_forbidden_bands);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_acoustic_spectra_free(spectra);
            diag->code = result; return result;
        }
    }

    /* full TR spectra allocation */
    spectra->spectrum_full_gx = (float*)PULSEQLIB_ALLOC((size_t)num_freq_bins_full * sizeof(float));
    spectra->spectrum_full_gy = (float*)PULSEQLIB_ALLOC((size_t)num_freq_bins_full * sizeof(float));
    spectra->spectrum_full_gz = (float*)PULSEQLIB_ALLOC((size_t)num_freq_bins_full * sizeof(float));
    if (!spectra->spectrum_full_gx || !spectra->spectrum_full_gy || !spectra->spectrum_full_gz) {
        if (seq_spec_gx) PULSEQLIB_FREE(seq_spec_gx);
        if (seq_freqs)   PULSEQLIB_FREE(seq_freqs);
        pulseqlib_acoustic_spectra_free(spectra);
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED; return diag->code;
    }

    if (fundamental_freq > 0.0f && num_freq_bins_seq > 0) {
        spectra->freq_spacing_seq_hz = fundamental_freq;
        spectra->num_freq_bins_seq = num_freq_bins_seq;
        spectra->spectrum_seq_gx = (float*)PULSEQLIB_ALLOC((size_t)num_freq_bins_seq * sizeof(float));
        spectra->spectrum_seq_gy = (float*)PULSEQLIB_ALLOC((size_t)num_freq_bins_seq * sizeof(float));
        spectra->spectrum_seq_gz = (float*)PULSEQLIB_ALLOC((size_t)num_freq_bins_seq * sizeof(float));
        if (!spectra->spectrum_seq_gx || !spectra->spectrum_seq_gy || !spectra->spectrum_seq_gz) {
            if (seq_spec_gx) PULSEQLIB_FREE(seq_spec_gx);
            if (seq_freqs)   PULSEQLIB_FREE(seq_freqs);
            pulseqlib_acoustic_spectra_free(spectra);
            diag->code = PULSEQLIB_ERR_ALLOC_FAILED; return diag->code;
        }
        if (seq_spec_gx && waveforms->num_samples > 0) {
            memcpy(spectra->spectrum_seq_gx, seq_spec_gx, (size_t)num_freq_bins_seq * sizeof(float));
            PULSEQLIB_FREE(seq_spec_gx); seq_spec_gx = NULL;
        } else {
            memset(spectra->spectrum_seq_gx, 0, (size_t)num_freq_bins_seq * sizeof(float));
        }
    }
    if (seq_spec_gx) { PULSEQLIB_FREE(seq_spec_gx); seq_spec_gx = NULL; }
    if (seq_freqs)   { PULSEQLIB_FREE(seq_freqs);   seq_freqs = NULL; }

    /* full-TR spectrum: Gx */
    if (waveforms->num_samples > 0) {
        result = compute_sequence_spectrum(
            spectra->spectrum_full_gx, NULL, NULL,
            NULL, NULL, NULL, NULL, NULL,
            waveforms->gx, waveforms->num_samples,
            waveforms->raster_us, target_spectral_resolution_hz,
            max_frequency_hz, 0.0f, num_trs,
            forbidden_bands, num_forbidden_bands);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_acoustic_spectra_free(spectra);
            diag->code = result; return result;
        }
    } else {
        memset(spectra->spectrum_full_gx, 0, (size_t)num_freq_bins_full * sizeof(float));
    }

    /* Gy - full TR + seq */
    if (waveforms->num_samples > 0) {
        result = compute_sequence_spectrum(
            spectra->spectrum_full_gy,
            (fundamental_freq > 0.0f) ? &seq_spec_gy : NULL,
            NULL, NULL, NULL, NULL, NULL,
            (fundamental_freq > 0.0f) ? &spectra->peaks_seq_gy : NULL,
            waveforms->gy, waveforms->num_samples,
            waveforms->raster_us, target_spectral_resolution_hz,
            max_frequency_hz, fundamental_freq, num_trs,
            forbidden_bands, num_forbidden_bands);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_acoustic_spectra_free(spectra);
            diag->code = result; return result;
        }
        if (seq_spec_gy && spectra->spectrum_seq_gy) {
            memcpy(spectra->spectrum_seq_gy, seq_spec_gy, (size_t)num_freq_bins_seq * sizeof(float));
            PULSEQLIB_FREE(seq_spec_gy); seq_spec_gy = NULL;
        } else if (seq_spec_gy) { PULSEQLIB_FREE(seq_spec_gy); seq_spec_gy = NULL; }
    } else {
        memset(spectra->spectrum_full_gy, 0, (size_t)num_freq_bins_full * sizeof(float));
        if (spectra->spectrum_seq_gy)
            memset(spectra->spectrum_seq_gy, 0, (size_t)num_freq_bins_seq * sizeof(float));
        if (spectra->peaks_seq_gy)
            memset(spectra->peaks_seq_gy, 0, (size_t)num_freq_bins_seq * sizeof(int));
    }

    /* Gz - full TR + seq */
    if (waveforms->num_samples > 0) {
        result = compute_sequence_spectrum(
            spectra->spectrum_full_gz,
            (fundamental_freq > 0.0f) ? &seq_spec_gz : NULL,
            NULL, NULL, NULL, NULL, NULL,
            (fundamental_freq > 0.0f) ? &spectra->peaks_seq_gz : NULL,
            waveforms->gz, waveforms->num_samples,
            waveforms->raster_us, target_spectral_resolution_hz,
            max_frequency_hz, fundamental_freq, num_trs,
            forbidden_bands, num_forbidden_bands);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_acoustic_spectra_free(spectra);
            diag->code = result; return result;
        }
        if (seq_spec_gz && spectra->spectrum_seq_gz) {
            memcpy(spectra->spectrum_seq_gz, seq_spec_gz, (size_t)num_freq_bins_seq * sizeof(float));
            PULSEQLIB_FREE(seq_spec_gz); seq_spec_gz = NULL;
        } else if (seq_spec_gz) { PULSEQLIB_FREE(seq_spec_gz); seq_spec_gz = NULL; }
    } else {
        memset(spectra->spectrum_full_gz, 0, (size_t)num_freq_bins_full * sizeof(float));
        if (spectra->spectrum_seq_gz)
            memset(spectra->spectrum_seq_gz, 0, (size_t)num_freq_bins_seq * sizeof(float));
        if (spectra->peaks_seq_gz)
            memset(spectra->peaks_seq_gz, 0, (size_t)num_freq_bins_seq * sizeof(int));
    }

    /* full-TR peak detection */
    if (num_freq_bins_full > 0) {
        spectra->peaks_full_gx = (int*)PULSEQLIB_ALLOC((size_t)num_freq_bins_full * sizeof(int));
        spectra->peaks_full_gy = (int*)PULSEQLIB_ALLOC((size_t)num_freq_bins_full * sizeof(int));
        spectra->peaks_full_gz = (int*)PULSEQLIB_ALLOC((size_t)num_freq_bins_full * sizeof(int));
        if (spectra->peaks_full_gx) detect_resonances(spectra->peaks_full_gx, spectra->spectrum_full_gx, num_freq_bins_full);
        if (spectra->peaks_full_gy) detect_resonances(spectra->peaks_full_gy, spectra->spectrum_full_gy, num_freq_bins_full);
        if (spectra->peaks_full_gz) detect_resonances(spectra->peaks_full_gz, spectra->spectrum_full_gz, num_freq_bins_full);
    }

    diag->code = PULSEQLIB_SUCCESS;
    return PULSEQLIB_SUCCESS;
}

/* ================================================================== */
/*  Acoustic spectra (public wrapper)                                 */
/* ================================================================== */

int pulseqlib_calc_acoustic_spectra(
    pulseqlib_acoustic_spectra* spectra,
    pulseqlib_diagnostic* diag,
    const pulseqlib_collection* coll,
    int subseq_idx,
    const pulseqlib_opts* opts,
    int target_window_size,
    float target_resolution_hz,
    float max_freq_hz,
    int num_forbidden_bands,
    const pulseqlib_forbidden_band* forbidden_bands)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_descriptor* trd;
    pulseqlib__uniform_grad_waveforms uw;
    pulseqlib_diagnostic local_diag;
    int rc;

    (void)opts;
    memset(&uw, 0, sizeof(uw));
    if (!diag) { pulseqlib_diagnostic_init(&local_diag); diag = &local_diag; }
    else       { pulseqlib_diagnostic_init(diag); }
    if (!coll || !spectra) { diag->code = PULSEQLIB_ERR_NULL_POINTER; return diag->code; }
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences) {
        diag->code = PULSEQLIB_ERR_INVALID_ARGUMENT; return diag->code;
    }
    desc = &coll->descriptors[subseq_idx];
    trd = &desc->tr_descriptor;
    rc = pulseqlib__get_gradient_waveforms_range(desc, &uw, diag,
        trd->num_prep_blocks + trd->imaging_tr_start,
        trd->tr_size, PULSEQLIB_AMP_MAX_POS, NULL, 0);
    if (PULSEQLIB_FAILED(rc)) return rc;
    rc = calc_acoustic_spectra_from_uniform(spectra, diag, &uw,
        target_window_size, target_resolution_hz, max_freq_hz,
        trd->num_trs, trd->tr_duration_us,
        num_forbidden_bands, forbidden_bands);
    pulseqlib__uniform_grad_waveforms_free(&uw);
    return rc;
}

/* ================================================================== */
/*  PNS                                                               */
/* ================================================================== */

/* FIX: outputs before inputs */
static int build_pns_kernel(
    float** kernel, int* kernel_len,
    float dt_us, const pulseqlib_pns_params* params)
{
    int n, i;
    float c_s, dt_s, s_min, tau, denom;
    float* k;

    if (params->chronaxie_us <= 0.0f) return PULSEQLIB_ERR_PNS_INVALID_CHRONAXIE;
    if (params->rheobase_hz_per_m_per_s     <= 0.0f) return PULSEQLIB_ERR_PNS_INVALID_RHEOBASE;
    if (params->alpha        <= 0.0f) return PULSEQLIB_ERR_PNS_INVALID_PARAMS;

    c_s  = params->chronaxie_us * 1e-6f;
    dt_s = dt_us * 1e-6f;
    s_min = params->rheobase_hz_per_m_per_s / params->alpha;

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
    return PULSEQLIB_SUCCESS;
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

    if (num_samples <= 0 || !waveform) return PULSEQLIB_SUCCESS;

    padded_len = num_samples + kernel_len;
    slew_len   = padded_len - 1;

    for (i = 0; i < num_samples; ++i)
        padded_waveform[i] = waveform[i];
    for (i = 0; i < kernel_len; ++i)
        padded_waveform[num_samples + i] = waveform[i % num_samples];

    compute_slew_rate(slew_rate, padded_waveform, padded_len, grad_raster_us, gamma_hz_per_tesla);

    rc = pulseqlib__calc_convolution_fft(pns_conv, slew_rate, slew_len, kernel, kernel_len);
    if (PULSEQLIB_FAILED(rc)) return rc;

    for (i = 0; i < slew_len; ++i) {
        pns_axis[i]   = pns_conv[i] * 100.0f;
        pns_total[i] += pns_conv[i] * pns_conv[i];
    }

    if (pns_store) {
        for (i = 0; i < slew_len; ++i) pns_store[i] = pns_axis[i];
    }
    return PULSEQLIB_SUCCESS;
}

static int calc_pns_from_uniform(
    pulseqlib_pns_result* result,
    pulseqlib_diagnostic* diag,
    float gamma_hz_per_tesla,
    const pulseqlib__uniform_grad_waveforms* waveforms,
    const pulseqlib_pns_params* params)
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
    rc      = PULSEQLIB_SUCCESS;

    if (!diag) { pulseqlib_diagnostic_init(&local_diag); diag = &local_diag; }
    else       { pulseqlib_diagnostic_init(diag); }

    if (!waveforms || !params || !result) {
        diag->code = PULSEQLIB_ERR_NULL_POINTER; return diag->code;
    }

    memset(result, 0, sizeof(*result));

    max_samples = waveforms->num_samples;
    if (max_samples <= 1) { diag->code = PULSEQLIB_ERR_PNS_NO_WAVEFORM; return diag->code; }

    rc = build_pns_kernel(&kernel, &kernel_len, waveforms->raster_us, params);
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

    pns_x = (float*)PULSEQLIB_ALLOC((size_t)full_output_len * sizeof(float));
    pns_y = (float*)PULSEQLIB_ALLOC((size_t)full_output_len * sizeof(float));
    pns_z = (float*)PULSEQLIB_ALLOC((size_t)full_output_len * sizeof(float));
    if (!pns_x || !pns_y || !pns_z) { rc = PULSEQLIB_ERR_ALLOC_FAILED; goto fail; }
    for (i = 0; i < full_output_len; ++i) { pns_x[i] = 0.0f; pns_y[i] = 0.0f; pns_z[i] = 0.0f; }

    /* X */
    rc = process_pns_axis_circular(axis, pns_tot,
        pns_x,
        padded, slew, conv,
        waveforms->gx, waveforms->num_samples,
        kernel, kernel_len,
        waveforms->raster_us, gamma_hz_per_tesla,
        full_output_len);
    if (PULSEQLIB_FAILED(rc)) goto fail;

    /* Y */
    rc = process_pns_axis_circular(axis, pns_tot,
        pns_y,
        padded, slew, conv,
        waveforms->gy, waveforms->num_samples,
        kernel, kernel_len,
        waveforms->raster_us, gamma_hz_per_tesla,
        full_output_len);
    if (PULSEQLIB_FAILED(rc)) goto fail;

    /* Z */
    rc = process_pns_axis_circular(axis, pns_tot,
        pns_z,
        padded, slew, conv,
        waveforms->gz, waveforms->num_samples,
        kernel, kernel_len,
        waveforms->raster_us, gamma_hz_per_tesla,
        full_output_len);
    if (PULSEQLIB_FAILED(rc)) goto fail;

    for (i = 0; i < full_output_len; ++i) {
        pns_tot[i] = 100.0f * (float)sqrt((double)pns_tot[i]);
    }

    result->num_samples    = full_output_len;
    result->slew_x_hz_per_m_per_s = pns_x; pns_x = NULL;
    result->slew_y_hz_per_m_per_s = pns_y; pns_y = NULL;
    result->slew_z_hz_per_m_per_s = pns_z; pns_z = NULL;

    rc = PULSEQLIB_SUCCESS;
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

/* ================================================================== */
/*  PNS (public wrapper)                                              */
/* ================================================================== */

int pulseqlib_calc_pns(
    pulseqlib_pns_result* result,
    pulseqlib_diagnostic* diag,
    const pulseqlib_collection* coll,
    int subseq_idx,
    const pulseqlib_opts* opts,
    const pulseqlib_pns_params* params)
{
    const pulseqlib_sequence_descriptor* desc;
    pulseqlib__uniform_grad_waveforms uw;
    pulseqlib_diagnostic local_diag;
    int rc;

    (void)opts;
    memset(&uw, 0, sizeof(uw));
    if (!diag) { pulseqlib_diagnostic_init(&local_diag); diag = &local_diag; }
    else       { pulseqlib_diagnostic_init(diag); }
    if (!coll || !result || !params) { diag->code = PULSEQLIB_ERR_NULL_POINTER; return diag->code; }
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences) {
        diag->code = PULSEQLIB_ERR_INVALID_ARGUMENT; return diag->code;
    }
    desc = &coll->descriptors[subseq_idx];
    rc = pulseqlib__get_gradient_waveforms_range(desc, &uw, diag,
        desc->tr_descriptor.num_prep_blocks
            + desc->tr_descriptor.imaging_tr_start,
        desc->tr_descriptor.tr_size, PULSEQLIB_AMP_MAX_POS, NULL, 0);
    if (PULSEQLIB_FAILED(rc)) return rc;
    rc = calc_pns_from_uniform(result, diag, opts->gamma_hz_per_t, &uw, params);
    pulseqlib__uniform_grad_waveforms_free(&uw);
    return rc;
}

/* ================================================================== */
/*  PNS result free                                                   */
/* ================================================================== */

void pulseqlib_pns_result_free(pulseqlib_pns_result* r)
{
    if (!r) return;
    if (r->slew_x_hz_per_m_per_s)     { PULSEQLIB_FREE(r->slew_x_hz_per_m_per_s);     r->slew_x_hz_per_m_per_s = NULL; }
    if (r->slew_y_hz_per_m_per_s)     { PULSEQLIB_FREE(r->slew_y_hz_per_m_per_s);     r->slew_y_hz_per_m_per_s = NULL; }
    if (r->slew_z_hz_per_m_per_s)     { PULSEQLIB_FREE(r->slew_z_hz_per_m_per_s);     r->slew_z_hz_per_m_per_s = NULL; }
    r->num_samples = 0;
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
    limit_sq     = opts->max_grad_hz_per_m * opts->max_grad_hz_per_m;
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
        hz_per_mt = opts->gamma_hz_per_t * 0.001f;
        if (diag) {
            diag->code = PULSEQLIB_ERR_MAX_GRAD_EXCEEDED;
            pulseqlib__diag_printf(diag,
                "max grad exceeded: subseq=%d block=%d amp=%.4f limit=%.4f mT/m",
                worst_subseq, worst_block,
                (double)((float)sqrt((double)gsos_max) / hz_per_mt),
                (double)(opts->max_grad_hz_per_m / hz_per_mt));
        }
        return PULSEQLIB_ERR_MAX_GRAD_EXCEEDED;
    }

    return PULSEQLIB_SUCCESS;
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
    coll->block_cursor.sequence_index = 0;
    coll->block_cursor.scan_table_position = 0;
    coll->block_cursor.from_last_reset = 0;

    prev_phys[0] = 0.0f;
    prev_phys[1] = 0.0f;
    prev_phys[2] = 0.0f;
    cur_seq = 0;
    status  = PULSEQLIB_CURSOR_BLOCK;

    desc = &coll->descriptors[0];
    grad_raster_s = desc->grad_raster_us * 1e-6f;
    max_allowed   = opts->max_slew_hz_per_m_per_s * grad_raster_s;

    while (status != PULSEQLIB_CURSOR_DONE) {
        /* detect subsequence change */
        if (coll->block_cursor.sequence_index != cur_seq) {
            /* end-of-subsequence: prev must ramp to zero */
            for (n = 0; n < 3; ++n) {
                step = prev_phys[n];
                if (step < 0.0f) step = -step;
                if (step > max_allowed) {
                    hz_per_mt = opts->gamma_hz_per_t * 0.001f;
                    if (diag) {
                        diag->code = PULSEQLIB_ERR_GRAD_DISCONTINUITY;
                        pulseqlib__diag_printf(diag,
                            "grad discontinuity at subseq boundary: axis=%d step=%.4f limit=%.4f mT/m",
                            n, (double)(step / hz_per_mt), (double)(max_allowed / hz_per_mt));
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
            grad_raster_s = desc->grad_raster_us * 1e-6f;
            max_allowed   = opts->max_slew_hz_per_m_per_s * grad_raster_s;
        }

        /* read current block */
        {
            int bt_idx = desc->scan_table_block_idx[coll->block_cursor.scan_table_position];
            bte  = &desc->block_table[bt_idx];
        }
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
                hz_per_mt = opts->gamma_hz_per_t * 0.001f;
                if (diag) {
                    diag->code = PULSEQLIB_ERR_GRAD_DISCONTINUITY;
                    pulseqlib__diag_printf(diag,
                        "grad discontinuity: axis=%d scan_pos=%d step=%.4f limit=%.4f mT/m",
                        n, coll->block_cursor.scan_table_position,
                        (double)(step / hz_per_mt), (double)(max_allowed / hz_per_mt));
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
            hz_per_mt = opts->gamma_hz_per_t * 0.001f;
            if (diag) {
                diag->code = PULSEQLIB_ERR_GRAD_DISCONTINUITY;
                pulseqlib__diag_printf(diag,
                    "grad discontinuity at trailing edge: axis=%d step=%.4f limit=%.4f mT/m",
                    n, (double)(step / hz_per_mt), (double)(max_allowed / hz_per_mt));
            }
            coll->block_cursor = saved_cursor;
            return PULSEQLIB_ERR_GRAD_DISCONTINUITY;
        }
    }

    coll->block_cursor = saved_cursor;
    return PULSEQLIB_SUCCESS;
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

    slew_limit = opts->max_slew_hz_per_m_per_s / (float)sqrt(3.0);

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
                            diag->code = PULSEQLIB_ERR_MAX_SLEW_EXCEEDED;
                            pulseqlib__diag_printf(diag,
                                "max slew exceeded: axis=%d def=%d slew=%.4f limit=%.4f Hz/m/s",
                                n, d, (double)slew_phys, (double)slew_limit);
                        }
                        return PULSEQLIB_ERR_MAX_SLEW_EXCEEDED;
                    }
                }
            }
        }
    }

    return PULSEQLIB_SUCCESS;
}

/* ================================================================== */
/*  Safety check                                                      */
/* ================================================================== */
int pulseqlib_check_safety(
    pulseqlib_collection* coll,
    pulseqlib_diagnostic* diag,
    const pulseqlib_opts* opts,
    int num_forbidden_bands,
    const pulseqlib_forbidden_band* forbidden_bands,
    const pulseqlib_pns_params* pns_params,
    float pns_threshold_percent)
{
    int rc, s, u, i;
    int num_unique_trs;
    int* unique_tr_indices;
    int* tr_group_labels;
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_descriptor* trd;
    pulseqlib__uniform_grad_waveforms uw;
    pulseqlib_acoustic_spectra spectra;
    pulseqlib_pns_result pns_result;
    int cd_start, cd_size;
    float pns_combined, max_pns;

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

    /* ---- 4. per-subsequence acoustic + PNS ---- */
    for (s = 0; s < coll->num_subsequences; ++s) {
        desc = &coll->descriptors[s];
        trd  = &desc->tr_descriptor;

        /* --- prep TR (if not degenerate) --- */
        if (trd->num_prep_blocks > 0 && !trd->degenerate_prep) {
            memset(&uw, 0, sizeof(uw));
            rc = pulseqlib__get_gradient_waveforms_range(desc, &uw, diag,
                0, trd->num_prep_blocks + trd->tr_size, PULSEQLIB_AMP_ACTUAL,
                NULL, 0);
            if (PULSEQLIB_FAILED(rc)) return rc;

            if (num_forbidden_bands > 0) {
                memset(&spectra, 0, sizeof(spectra));
                rc = calc_acoustic_spectra_from_uniform(
                    &spectra, diag, &uw,
                    0, 0.0f, 0.0f,
                    1, 0.0f,
                    num_forbidden_bands, forbidden_bands);
                pulseqlib_acoustic_spectra_free(&spectra);
                if (PULSEQLIB_FAILED(rc)) {
                    pulseqlib__uniform_grad_waveforms_free(&uw);
                    return rc;
                }
            }

            if (pns_params) {
                memset(&pns_result, 0, sizeof(pns_result));
                rc = calc_pns_from_uniform(
                    &pns_result, diag, opts->gamma_hz_per_t,
                    &uw, pns_params);
                if (!PULSEQLIB_FAILED(rc) && pns_result.num_samples > 0) {
                    max_pns = 0.0f;
                    for (i = 0; i < pns_result.num_samples; ++i) {
                        pns_combined = 0.0f;
                        if (pns_result.slew_x_hz_per_m_per_s)
                            pns_combined += pns_result.slew_x_hz_per_m_per_s[i] * pns_result.slew_x_hz_per_m_per_s[i];
                        if (pns_result.slew_y_hz_per_m_per_s)
                            pns_combined += pns_result.slew_y_hz_per_m_per_s[i] * pns_result.slew_y_hz_per_m_per_s[i];
                        if (pns_result.slew_z_hz_per_m_per_s)
                            pns_combined += pns_result.slew_z_hz_per_m_per_s[i] * pns_result.slew_z_hz_per_m_per_s[i];
                        pns_combined = (float)sqrt((double)pns_combined);
                        if (pns_combined > max_pns) max_pns = pns_combined;
                    }
                    if (max_pns > pns_threshold_percent) rc = PULSEQLIB_ERR_PNS_THRESHOLD_EXCEEDED;
                }
                pulseqlib_pns_result_free(&pns_result);
                if (PULSEQLIB_FAILED(rc)) {
                    pulseqlib__uniform_grad_waveforms_free(&uw);
                    return rc;
                }
            }

            pulseqlib__uniform_grad_waveforms_free(&uw);
        }

        /* --- main TR (unique shot-index variants) --- */
        unique_tr_indices = NULL;
        tr_group_labels   = NULL;
        num_unique_trs = pulseqlib__find_unique_shot_trs(desc,
            &unique_tr_indices, &tr_group_labels);
        if (num_unique_trs <= 0) num_unique_trs = 1;

        for (u = 0; u < num_unique_trs; ++u) {
            memset(&uw, 0, sizeof(uw));
            rc = pulseqlib__get_gradient_waveforms_range(desc, &uw, diag,
                trd->num_prep_blocks + trd->imaging_tr_start,
                trd->tr_size, PULSEQLIB_AMP_MAX_POS,
                tr_group_labels, u);
            if (PULSEQLIB_FAILED(rc)) {
                if (unique_tr_indices) PULSEQLIB_FREE(unique_tr_indices);
                if (tr_group_labels)   PULSEQLIB_FREE(tr_group_labels);
                return rc;
            }

            if (num_forbidden_bands > 0) {
                memset(&spectra, 0, sizeof(spectra));
                rc = calc_acoustic_spectra_from_uniform(
                    &spectra, diag, &uw,
                    0, 0.0f, 0.0f,
                    trd->num_trs, trd->tr_duration_us,
                    num_forbidden_bands, forbidden_bands);
                pulseqlib_acoustic_spectra_free(&spectra);
                if (PULSEQLIB_FAILED(rc)) {
                    pulseqlib__uniform_grad_waveforms_free(&uw);
                    if (unique_tr_indices) PULSEQLIB_FREE(unique_tr_indices);
                    if (tr_group_labels)   PULSEQLIB_FREE(tr_group_labels);
                    return rc;
                }
            }

            if (pns_params) {
                memset(&pns_result, 0, sizeof(pns_result));
                rc = calc_pns_from_uniform(
                    &pns_result, diag, opts->gamma_hz_per_t,
                    &uw, pns_params);
                if (!PULSEQLIB_FAILED(rc) && pns_result.num_samples > 0) {
                    max_pns = 0.0f;
                    for (i = 0; i < pns_result.num_samples; ++i) {
                        pns_combined = 0.0f;
                        if (pns_result.slew_x_hz_per_m_per_s)
                            pns_combined += pns_result.slew_x_hz_per_m_per_s[i] * pns_result.slew_x_hz_per_m_per_s[i];
                        if (pns_result.slew_y_hz_per_m_per_s)
                            pns_combined += pns_result.slew_y_hz_per_m_per_s[i] * pns_result.slew_y_hz_per_m_per_s[i];
                        if (pns_result.slew_z_hz_per_m_per_s)
                            pns_combined += pns_result.slew_z_hz_per_m_per_s[i] * pns_result.slew_z_hz_per_m_per_s[i];
                        pns_combined = (float)sqrt((double)pns_combined);
                        if (pns_combined > max_pns) max_pns = pns_combined;
                    }
                    if (max_pns > pns_threshold_percent) rc = PULSEQLIB_ERR_PNS_THRESHOLD_EXCEEDED;
                }
                pulseqlib_pns_result_free(&pns_result);
                if (PULSEQLIB_FAILED(rc)) {
                    pulseqlib__uniform_grad_waveforms_free(&uw);
                    if (unique_tr_indices) PULSEQLIB_FREE(unique_tr_indices);
                    if (tr_group_labels)   PULSEQLIB_FREE(tr_group_labels);
                    return rc;
                }
            }

            pulseqlib__uniform_grad_waveforms_free(&uw);
        }

        if (unique_tr_indices) PULSEQLIB_FREE(unique_tr_indices);
        if (tr_group_labels)   PULSEQLIB_FREE(tr_group_labels);

        /* --- cooldown TR (if not degenerate) --- */
        if (trd->num_cooldown_blocks > 0 && !trd->degenerate_cooldown) {
            cd_size  = trd->tr_size + trd->num_cooldown_blocks;
            cd_start = desc->num_blocks - cd_size;

            memset(&uw, 0, sizeof(uw));
            rc = pulseqlib__get_gradient_waveforms_range(desc, &uw, diag,
                cd_start, cd_size, PULSEQLIB_AMP_ACTUAL,
                NULL, 0);
            if (PULSEQLIB_FAILED(rc)) return rc;

            if (num_forbidden_bands > 0) {
                memset(&spectra, 0, sizeof(spectra));
                rc = calc_acoustic_spectra_from_uniform(
                    &spectra, diag, &uw,
                    0, 0.0f, 0.0f,
                    1, 0.0f,
                    num_forbidden_bands, forbidden_bands);
                pulseqlib_acoustic_spectra_free(&spectra);
                if (PULSEQLIB_FAILED(rc)) {
                    pulseqlib__uniform_grad_waveforms_free(&uw);
                    return rc;
                }
            }

            if (pns_params) {
                memset(&pns_result, 0, sizeof(pns_result));
                rc = calc_pns_from_uniform(
                    &pns_result, diag, opts->gamma_hz_per_t,
                    &uw, pns_params);
                if (!PULSEQLIB_FAILED(rc) && pns_result.num_samples > 0) {
                    max_pns = 0.0f;
                    for (i = 0; i < pns_result.num_samples; ++i) {
                        pns_combined = 0.0f;
                        if (pns_result.slew_x_hz_per_m_per_s)
                            pns_combined += pns_result.slew_x_hz_per_m_per_s[i] * pns_result.slew_x_hz_per_m_per_s[i];
                        if (pns_result.slew_y_hz_per_m_per_s)
                            pns_combined += pns_result.slew_y_hz_per_m_per_s[i] * pns_result.slew_y_hz_per_m_per_s[i];
                        if (pns_result.slew_z_hz_per_m_per_s)
                            pns_combined += pns_result.slew_z_hz_per_m_per_s[i] * pns_result.slew_z_hz_per_m_per_s[i];
                        pns_combined = (float)sqrt((double)pns_combined);
                        if (pns_combined > max_pns) max_pns = pns_combined;
                    }
                    if (max_pns > pns_threshold_percent) rc = PULSEQLIB_ERR_PNS_THRESHOLD_EXCEEDED;
                }
                pulseqlib_pns_result_free(&pns_result);
                if (PULSEQLIB_FAILED(rc)) {
                    pulseqlib__uniform_grad_waveforms_free(&uw);
                    return rc;
                }
            }

            pulseqlib__uniform_grad_waveforms_free(&uw);
        }
    }

    return PULSEQLIB_SUCCESS;
}