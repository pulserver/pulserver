/* pulseqlib_core.c -- deduplication, TR detection, segmentation, and loading
 *
 * This file contains the core analysis pipeline:
 *   1. get_unique_blocks   -- deduplicate events and blocks
 *   2. find_tr_in_sequence -- detect repetition time structure
 *   3. find_segments_in_tr -- segment each TR into playable pieces
 *   4. get_collection_descriptors -- chain subsequences
 *   5. pulseqlib_load       -- public entry point
 *
 * All static helpers appear before their callers (no forward declarations).
 * ANSI C89.
 */

#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "pulseqlib_internal.h"
#include "pulseqlib.h"

#if PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_GEHC
#include "external_kiss_fft.h"
#endif

/* ================================================================== */
/*  File-scope constants                                               */
/* ================================================================== */
#define RF_DEF_COLS    4
#define RF_PARAMS_COLS 3
#define GRAD_DEF_COLS  6
#define ADC_DEF_COLS   3
#define ADC_PARAMS_COLS 2
#define BLOCK_DEF_COLS 5

#define PREP_COOLDOWN_THRESHOLD_US   100000   /* 100 ms */
#define SINGLE_TR_MAX_DURATION_US  15000000   /* 15 s  */

#define SEGSTATE_SEEKING_FIRST_ADC 0
#define SEGSTATE_SEEKING_BOUNDARY  1
#define SEGSTATE_OPTIMIZED_MODE    2

/* ================================================================== */
/*  Tiny helpers                                                       */
/* ================================================================== */

static int array_equal(const int* a, const int* b, int len)
{
    int i;
    for (i = 0; i < len; ++i)
        if (a[i] != b[i]) return 0;
    return 1;
}

static size_t next_pow2(size_t x)
{
    size_t v = 1;
    while (v < x) v <<= 1;
    return v;
}

/* ================================================================== */
/*  Hash-based integer-row deduplication                               */
/* ================================================================== */

typedef struct {
    size_t hash;
    int    row_index;
    int    label;
    char   used;
} hash_entry;

static size_t hash_row(const int* row, int num_cols)
{
    size_t h = 2166136261UL;
    int i;
    for (i = 0; i < num_cols; ++i) {
        h ^= (size_t)row[i];
        h *= 16777619UL;
    }
    return h;
}

static int deduplicate_int_rows(
    int* unique_defs, int* event_table, 
    const int* int_rows, int num_rows, int num_cols
) {
    size_t table_size;
    hash_entry* table = NULL;
    int num_unique = 0;
    int r;
    size_t h, idx;

    if (num_rows <= 0) return 0;

    table_size = next_pow2((size_t)(num_rows * 2));
    table = (hash_entry*)ALLOC(table_size * sizeof(hash_entry));
    if (!table) return 0;
    memset(table, 0, table_size * sizeof(hash_entry));

    for (r = 0; r < num_rows; ++r) {
        h = hash_row(&int_rows[r * num_cols], num_cols);
        idx = h & (table_size - 1);

        while (table[idx].used) {
            if (table[idx].hash == h &&
                array_equal(&int_rows[r * num_cols],
                            &int_rows[table[idx].row_index * num_cols],
                            num_cols)) {
                event_table[r] = table[idx].label;
                break;
            }
            idx = (idx + 1) & (table_size - 1);
        }

        if (!table[idx].used) {
            table[idx].hash      = h;
            table[idx].row_index = r;
            table[idx].label     = num_unique;
            table[idx].used      = 1;
            unique_defs[num_unique] = r;
            event_table[r] = num_unique;
            num_unique++;
        }
    }

    FREE(table);
    return num_unique;
}

/* ================================================================== */
/*  RF dedup helpers                                                   */
/* ================================================================== */

static void build_rf_def_row(const pulseqlib__seq_file* seq, int* row, float* params, int rf_idx)
{
    float gamma = seq->opts.gamma;
    float b0    = seq->opts.b0;
    float* rf   = seq->rf_library[rf_idx];
    float ppm_to_hz = 1e-6f * gamma * b0;

    row[0] = (int)rf[1];  /* mag shape id */
    row[1] = (int)rf[2];  /* phase shape id */
    row[2] = (int)rf[3];  /* time shape id */
    row[3] = (int)rf[5];  /* delay */

    params[0] = rf[0];                        /* amplitude */
    params[1] = rf[7] + ppm_to_hz * rf[4];   /* freq offset */
    params[2] = rf[8] + ppm_to_hz * rf[6];   /* phase offset */
}

static int deduplicate_rf_library(const pulseqlib__seq_file* seq, pulseqlib_rf_definition* rf_defs, pulseqlib_rf_table_element* rf_table)
{
    int (*int_rows)[RF_DEF_COLS] = NULL;
    float (*params)[RF_PARAMS_COLS] = NULL;
    int* unique_defs = NULL;
    int* event_table = NULL;
    int num_unique, num_rows, i;

    num_rows = seq->rf_library_size;
    if (num_rows <= 0) return 0;

    int_rows    = ALLOC(num_rows * sizeof(*int_rows));
    params      = ALLOC(num_rows * sizeof(*params));
    unique_defs = (int*)ALLOC(num_rows * sizeof(int));
    event_table = (int*)ALLOC(num_rows * sizeof(int));
    if (!int_rows || !params || !unique_defs || !event_table) {
        if (int_rows)    FREE(int_rows);
        if (params)      FREE(params);
        if (unique_defs) FREE(unique_defs);
        if (event_table) FREE(event_table);
        return 0;
    }

    for (i = 0; i < num_rows; ++i)
        build_rf_def_row(seq, int_rows[i], params[i], i);

    num_unique = deduplicate_int_rows(unique_defs, event_table, (const int*)int_rows, num_rows, RF_DEF_COLS);

    for (i = 0; i < num_unique; ++i) {
        rf_defs[i].id            = unique_defs[i];
        rf_defs[i].mag_shape_id  = int_rows[unique_defs[i]][0];
        rf_defs[i].phase_shape_id = int_rows[unique_defs[i]][1];
        rf_defs[i].time_shape_id = int_rows[unique_defs[i]][2];
        rf_defs[i].delay         = int_rows[unique_defs[i]][3];
    }
    for (i = 0; i < num_rows; ++i) {
        rf_table[i].id           = event_table[i];
        rf_table[i].amplitude    = params[i][0];
        rf_table[i].freq_offset  = params[i][1];
        rf_table[i].phase_offset = params[i][2];
    }

    FREE(int_rows); FREE(params); FREE(unique_defs); FREE(event_table);
    return num_unique;
}

/* ================================================================== */
/*  Grad dedup helpers                                                 */
/* ================================================================== */

static void build_grad_def_row(const pulseqlib__seq_file* seq, int* row, float* param, int grad_idx)
{
    float* grad = seq->grad_library[grad_idx];
    int grad_type = (int)grad[0];
    int wave_id;

    row[0] = grad_type;
    if (grad_type == 0) {
        row[1] = (int)grad[2];  /* rise */
        row[2] = (int)grad[3];  /* flat */
        row[3] = (int)grad[4];  /* fall */
        row[4] = 0;
    } else {
        row[1] = 0;
        row[2] = 0;
        wave_id = (int)grad[4];
        if (wave_id > 0 && seq->is_shapes_library_parsed &&
            wave_id <= seq->shapes_library_size) {
            row[3] = seq->shapes_library[wave_id - 1].num_uncompressed_samples;
        } else {
            row[3] = 0;
        }
        row[4] = (int)grad[5];  /* time shape id */
    }
    row[5] = (int)grad[6];      /* delay */
    *param = grad[1];            /* amplitude */
}

static int deduplicate_grad_library(const pulseqlib__seq_file* seq, pulseqlib_grad_definition* grad_defs, pulseqlib_grad_table_element* grad_table)
{
    int (*int_rows)[GRAD_DEF_COLS] = NULL;
    float* params = NULL;
    int* unique_defs = NULL;
    int* event_table = NULL;
    int num_unique, num_rows, i;

    num_rows = seq->grad_library_size;
    if (num_rows <= 0) return 0;

    int_rows    = ALLOC(num_rows * sizeof(*int_rows));
    params      = (float*)ALLOC(num_rows * sizeof(float));
    unique_defs = (int*)ALLOC(num_rows * sizeof(int));
    event_table = (int*)ALLOC(num_rows * sizeof(int));
    if (!int_rows || !params || !unique_defs || !event_table) {
        if (int_rows)    FREE(int_rows);
        if (params)      FREE(params);
        if (unique_defs) FREE(unique_defs);
        if (event_table) FREE(event_table);
        return 0;
    }

    for (i = 0; i < num_rows; ++i)
        build_grad_def_row(seq, int_rows[i], &params[i], i);

    num_unique = deduplicate_int_rows(unique_defs, event_table, (const int*)int_rows, num_rows, GRAD_DEF_COLS);

    for (i = 0; i < num_unique; ++i) {
        grad_defs[i].id = unique_defs[i];
        grad_defs[i].type                                = int_rows[unique_defs[i]][0];
        grad_defs[i].rise_time_or_unused                 = int_rows[unique_defs[i]][1];
        grad_defs[i].flat_time_or_unused                 = int_rows[unique_defs[i]][2];
        grad_defs[i].fall_time_or_num_uncompressed_samples = int_rows[unique_defs[i]][3];
        grad_defs[i].unused_or_time_shape_id             = int_rows[unique_defs[i]][4];
        grad_defs[i].delay                               = int_rows[unique_defs[i]][5];
    }
    for (i = 0; i < num_rows; ++i) {
        grad_table[i].id        = event_table[i];
        grad_table[i].amplitude = params[i];
    }

    FREE(int_rows); FREE(params); FREE(unique_defs); FREE(event_table);
    return num_unique;
}

/* ================================================================== */
/*  ADC dedup helpers                                                  */
/* ================================================================== */

static void build_adc_def_row(const pulseqlib__seq_file* seq, int* row, float* params, int adc_idx)
{
    float gamma = seq->opts.gamma;
    float b0    = seq->opts.b0;
    float* adc  = seq->adc_library[adc_idx];
    float ppm_to_hz = 1e-6f * gamma * b0;

    row[0] = (int)adc[0];  /* num_samples */
    row[1] = (int)adc[1];  /* dwell_time_ns */
    row[2] = (int)adc[2];  /* delay */
    params[0] = adc[5] + ppm_to_hz * adc[3];  /* freq offset */
    params[1] = adc[6] + ppm_to_hz * adc[4];  /* phase offset */
}

static int deduplicate_adc_library(const pulseqlib__seq_file* seq, pulseqlib_adc_definition* adc_defs, pulseqlib_adc_table_element* adc_table)
{
    int (*int_rows)[ADC_DEF_COLS] = NULL;
    float (*params)[ADC_PARAMS_COLS] = NULL;
    int* unique_defs = NULL;
    int* event_table = NULL;
    int num_unique, num_rows, i;

    num_rows = seq->adc_library_size;
    if (num_rows <= 0) return 0;

    int_rows    = ALLOC(num_rows * sizeof(*int_rows));
    params      = ALLOC(num_rows * sizeof(*params));
    unique_defs = (int*)ALLOC(num_rows * sizeof(int));
    event_table = (int*)ALLOC(num_rows * sizeof(int));
    if (!int_rows || !params || !unique_defs || !event_table) {
        if (int_rows)    FREE(int_rows);
        if (params)      FREE(params);
        if (unique_defs) FREE(unique_defs);
        if (event_table) FREE(event_table);
        return 0;
    }

    for (i = 0; i < num_rows; ++i)
        build_adc_def_row(seq, int_rows[i], params[i], i);

    num_unique = deduplicate_int_rows(unique_defs, event_table, (const int*)int_rows, num_rows, ADC_DEF_COLS);

    for (i = 0; i < num_unique; ++i) {
        adc_defs[i].id          = unique_defs[i];
        adc_defs[i].num_samples = int_rows[unique_defs[i]][0];
        adc_defs[i].dwell_time  = int_rows[unique_defs[i]][1];
        adc_defs[i].delay       = int_rows[unique_defs[i]][2];
    }
    for (i = 0; i < num_rows; ++i) {
        adc_table[i].id           = event_table[i];
        adc_table[i].freq_offset  = params[i][0];
        adc_table[i].phase_offset = params[i][1];
    }

    FREE(int_rows); FREE(params); FREE(unique_defs); FREE(event_table);
    return num_unique;
}

/* ================================================================== */
/*  Gradient shot indices                                              */
/* ================================================================== */

static int compute_grad_shot_indices(
    const pulseqlib__seq_file* seq,
    pulseqlib_grad_definition* grad_defs, pulseqlib_grad_table_element* grad_table,
    int num_unique_grads
) {
    int num_rows = seq->grad_library_size;
    int def_idx, i, j;
    int shape_id, found, shot_count;

    if (num_rows <= 0 || num_unique_grads <= 0) return PULSEQLIB_OK;

    for (def_idx = 0; def_idx < num_unique_grads; ++def_idx) {
        int grad_type = grad_defs[def_idx].type;

        for (j = 0; j < PULSEQLIB_MAX_GRAD_SHOTS; ++j)
            grad_defs[def_idx].shot_shape_ids[j] = 0;

        if (grad_type == 0) {
            grad_defs[def_idx].num_shots = 1;
            for (i = 0; i < num_rows; ++i)
                if (grad_table[i].id == def_idx)
                    grad_table[i].shot_index = 0;
            continue;
        }

        shot_count = 0;
        for (i = 0; i < num_rows; ++i) {
            if (grad_table[i].id != def_idx) continue;
            shape_id = (int)seq->grad_library[i][4];

            found = 0;
            for (j = 0; j < shot_count; ++j) {
                if (grad_defs[def_idx].shot_shape_ids[j] == shape_id) {
                    found = 1;
                    grad_table[i].shot_index = j;
                    break;
                }
            }
            if (!found) {
                if (shot_count >= PULSEQLIB_MAX_GRAD_SHOTS)
                    return PULSEQLIB_ERR_TOO_MANY_GRAD_SHOTS;
                grad_table[i].shot_index = shot_count;
                grad_defs[def_idx].shot_shape_ids[shot_count] = shape_id;
                shot_count++;
            }
        }
        grad_defs[def_idx].num_shots = shot_count > 0 ? shot_count : 1;
    }
    return PULSEQLIB_OK;
}

/* ================================================================== */
/*  Waveform normalisation                                             */
/* ================================================================== */

static float normalize_waveform(float* waveform, int n)
{
    float max_abs;
    int i;

    max_abs = pulseqlib__find_max_abs_real(waveform, n);
    if (max_abs > 1e-9f) {
        for (i = 0; i < n; ++i) waveform[i] /= max_abs;
    }
    return max_abs;
}

/* ================================================================== */
/*  Trapezoid statistics                                                */
/* ================================================================== */

static void compute_trapezoid_stats(
    float* slew, float* energy, float* first_val, float* last_val, 
    float rise_us, float flat_us, float fall_us
) {
    float rise_s = rise_us * 1e-6f;
    float flat_s = flat_us * 1e-6f;
    float fall_s = fall_us * 1e-6f;
    float sr, sf;

    *first_val = 0.0f;
    *last_val  = 0.0f;

    sr = (rise_s > 0.0f) ? (1.0f / rise_s) : 0.0f;
    sf = (fall_s > 0.0f) ? (1.0f / fall_s) : 0.0f;
    *slew = (sr > sf) ? sr : sf;

    *energy = rise_s / 3.0f + flat_s + fall_s / 3.0f;
}

/* ================================================================== */
/*  Gradient statistics                                                */
/* ================================================================== */

static int compute_grad_stats(
    const pulseqlib__seq_file* seq,
    pulseqlib_grad_definition* grad_defs, int num_unique,
    const pulseqlib_grad_table_element* grad_table, int grad_table_size
) {
    int def_idx, i, shot_idx, num_samples, has_time;
    int grad_type, time_id, shape_id;
    float rise_us, flat_us, fall_us, abs_amp;
    float grad_raster_us;
    pulseqlib_shape_arbitrary decomp_wave, decomp_time;
    float* waveform   = NULL;
    float* sq_wave    = NULL;
    float* time_us    = NULL;
    pulseqlib_grad_definition* gd;

    if (!seq || !grad_defs || num_unique <= 0) return PULSEQLIB_OK;

    if (seq->reserved_definitions_library.gradient_raster_time > 0.0f)
        grad_raster_us = seq->reserved_definitions_library.gradient_raster_time;
    else
        grad_raster_us = seq->opts.grad_raster_time;

    decomp_wave.num_samples = 0;
    decomp_wave.num_uncompressed_samples = 0;
    decomp_wave.samples = NULL;
    decomp_time.num_samples = 0;
    decomp_time.num_uncompressed_samples = 0;
    decomp_time.samples = NULL;

    for (def_idx = 0; def_idx < num_unique; ++def_idx) {
        gd = &grad_defs[def_idx];
        grad_type = gd->type;

        for (i = 0; i < PULSEQLIB_MAX_GRAD_SHOTS; ++i) {
            gd->max_amplitude[i] = 0.0f;
            gd->slew_rate[i]     = 0.0f;
            gd->energy[i]        = 0.0f;
            gd->first_value[i]   = 0.0f;
            gd->last_value[i]    = 0.0f;
        }

        /* max amplitude per shot from table */
        if (grad_table && grad_table_size > 0) {
            for (i = 0; i < grad_table_size; ++i) {
                if (grad_table[i].id == def_idx) {
                    shot_idx = grad_table[i].shot_index;
                    if (shot_idx >= 0 && shot_idx < PULSEQLIB_MAX_GRAD_SHOTS) {
                        abs_amp = grad_table[i].amplitude;
                        if (abs_amp < 0.0f) abs_amp = -abs_amp;
                        if (abs_amp > gd->max_amplitude[shot_idx])
                            gd->max_amplitude[shot_idx] = abs_amp;
                    }
                }
            }
        }

        if (grad_type == 0) {
            rise_us = (float)gd->rise_time_or_unused;
            flat_us = (float)gd->flat_time_or_unused;
            fall_us = (float)gd->fall_time_or_num_uncompressed_samples;
            compute_trapezoid_stats(&gd->slew_rate[0], &gd->energy[0], &gd->first_value[0], &gd->last_value[0], rise_us, flat_us, fall_us);
        } else {
            time_id = gd->unused_or_time_shape_id;
            time_us = NULL;
            has_time = 0;
            if (time_id > 0 && time_id <= seq->shapes_library_size) {
                if (!pulseqlib__decompress_shape(&decomp_time,
                        &seq->shapes_library[time_id - 1], grad_raster_us))
                    goto fail;
                time_us = (float*)ALLOC(decomp_time.num_uncompressed_samples * sizeof(float));
                if (!time_us) goto fail;
                for (i = 0; i < decomp_time.num_uncompressed_samples; ++i)
                    time_us[i] = decomp_time.samples[i];
                has_time = 1;
                FREE(decomp_time.samples);
                decomp_time.samples = NULL;
            }

            for (shot_idx = 0; shot_idx < gd->num_shots; ++shot_idx) {
                shape_id = gd->shot_shape_ids[shot_idx];
                if (shape_id <= 0 || shape_id > seq->shapes_library_size) continue;

                if (!pulseqlib__decompress_shape(&decomp_wave,
                        &seq->shapes_library[shape_id - 1], 1.0f))
                    goto fail;
                num_samples = decomp_wave.num_uncompressed_samples;

                waveform = (float*)ALLOC(num_samples * sizeof(float));
                sq_wave  = (float*)ALLOC(num_samples * sizeof(float));
                if (!waveform || !sq_wave) goto fail;

                for (i = 0; i < num_samples; ++i) waveform[i] = decomp_wave.samples[i];
                normalize_waveform(waveform, num_samples);

                for (i = 0; i < num_samples; ++i) sq_wave[i] = waveform[i] * waveform[i];

                gd->first_value[shot_idx] = waveform[0];
                gd->last_value[shot_idx]  = waveform[num_samples - 1];

                if (has_time && time_us) {
                    gd->slew_rate[shot_idx] = pulseqlib__max_slew_real_nonuniform(waveform, time_us, num_samples);
                    gd->energy[shot_idx]    = pulseqlib__trapz_real_nonuniform(sq_wave, time_us, num_samples);
                } else {
                    gd->slew_rate[shot_idx] = pulseqlib__max_slew_real_uniform(waveform, num_samples, grad_raster_us);
                    gd->energy[shot_idx]    = pulseqlib__trapz_real_uniform(sq_wave, num_samples, grad_raster_us);
                }
                gd->slew_rate[shot_idx] *= 1e6f;
                gd->energy[shot_idx]    *= 1e-6f;

                FREE(waveform);  waveform = NULL;
                FREE(sq_wave);   sq_wave  = NULL;
                FREE(decomp_wave.samples); decomp_wave.samples = NULL;
            }

            if (time_us) { FREE(time_us); time_us = NULL; }
        }
    }
    return PULSEQLIB_OK;

fail:
    if (waveform)          FREE(waveform);
    if (sq_wave)           FREE(sq_wave);
    if (time_us)           FREE(time_us);
    if (decomp_wave.samples) FREE(decomp_wave.samples);
    if (decomp_time.samples) FREE(decomp_time.samples);
    return PULSEQLIB_ERR_ALLOC_FAILED;
}

/* ================================================================== */
/*  RF statistics (GEHC only)                                          */
/* ================================================================== */

#if PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_GEHC

static float compute_rf_bandwidth_fft(const float* rf_re, const float* rf_im,
                                      kiss_fft_cfg cfg, int nn, float dw,
                                      float cutoff, float duration,
                                      const float* w,
                                      float* work_re, float* work_im,
                                      kiss_fft_cpx* fft_in, kiss_fft_cpx* fft_out)
{
    int i;
    float w1, w2, bw;
    float fallback_bw;

    fallback_bw = (duration > 0.0f) ? (3.12f / duration) : 0.0f;
    if (!rf_re || !rf_im || !cfg || nn <= 0) return fallback_bw;

    for (i = 0; i < nn; ++i) { work_re[i] = rf_re[i]; work_im[i] = rf_im[i]; }
    pulseqlib__fftshift_complex(work_re, work_im, nn);
    for (i = 0; i < nn; ++i) { fft_in[i].r = work_re[i]; fft_in[i].i = work_im[i]; }

    kiss_fft(cfg, fft_in, fft_out);

    for (i = 0; i < nn; ++i) { work_re[i] = fft_out[i].r; work_im[i] = fft_out[i].i; }
    pulseqlib__fftshift_complex(work_re, work_im, nn);

    w1 = pulseqlib__find_spectrum_flank(w, work_re, work_im, nn, cutoff, 0);
    w2 = pulseqlib__find_spectrum_flank(w, work_re, work_im, nn, cutoff, 1);
    bw = w2 - w1;
    return (bw > 0.0f) ? bw : fallback_bw;
}

static int compute_rf_stats(
    const pulseqlib__seq_file* seq,
    pulseqlib_rf_definition* rf_defs, int num_unique,
    const pulseqlib_rf_table_element* rf_table, int rf_table_size
) {
    int def_idx, i;
    pulseqlib_shape_arbitrary decomp_mag, decomp_phase, decomp_time;
    float* magnitude = NULL;
    float* phase = NULL;
    float* time_us = NULL;
    float* time_us_uniform = NULL;
    float* rf_re = NULL;
    float* rf_im = NULL;
    float* rf_re_uniform = NULL;
    float* rf_im_uniform = NULL;
    float* time_centered = NULL;
    int num_samples, num_uniform, num_real;
    int mag_id, phase_id, time_id;
    int has_phase, has_time;
    int first, last;
    float max_mag, duration, time_center, rf_raster_us;
    pulseqlib_rf_definition* rd;

    const float DTY_THRESHOLD = 0.2236f;
    const float MPW_THRESHOLD = 1e-5f;

    int nn;
    float dw = 10.0f;
    float cutoff = 0.5f;

    kiss_fft_cfg fft_cfg = NULL;
    float* tt = NULL;
    float* w  = NULL;
    float* rfs_re = NULL;
    float* rfs_im = NULL;
    float* work_re = NULL;
    float* work_im = NULL;
    kiss_fft_cpx* fft_in  = NULL;
    kiss_fft_cpx* fft_out = NULL;
    int fft_ready = 0;

    float rf_abs, sum_signed, sum_signed_re, sum_signed_im;
    float sum_abs, sum_sq, time_above_threshold, temp_pw, maxpw;

    if (!seq || !rf_defs || num_unique <= 0) return PULSEQLIB_OK;

    if (seq->reserved_definitions_library.radiofrequency_raster_time > 0.0f)
        rf_raster_us = seq->reserved_definitions_library.radiofrequency_raster_time;
    else
        rf_raster_us = seq->opts.rf_raster_time;

    nn = (int)(1.0f / (dw * rf_raster_us * 1e-6f));
    nn = kiss_fft_next_fast_size(nn);
    if (nn < 2) nn = 2;

    tt      = (float*)ALLOC(nn * sizeof(float));
    w       = (float*)ALLOC(nn * sizeof(float));
    rfs_re  = (float*)ALLOC(nn * sizeof(float));
    rfs_im  = (float*)ALLOC(nn * sizeof(float));
    work_re = (float*)ALLOC(nn * sizeof(float));
    work_im = (float*)ALLOC(nn * sizeof(float));
    fft_in  = (kiss_fft_cpx*)KISS_FFT_MALLOC(nn * sizeof(kiss_fft_cpx));
    fft_out = (kiss_fft_cpx*)KISS_FFT_MALLOC(nn * sizeof(kiss_fft_cpx));
    fft_cfg = kiss_fft_alloc(nn, 0, NULL, NULL);
    if (tt && w && rfs_re && rfs_im && work_re && work_im && fft_in && fft_out && fft_cfg) {
        for (i = 0; i < nn; ++i) {
            tt[i] = (float)(i - nn / 2) * rf_raster_us;
            w[i]  = (float)(i - nn / 2) * dw;
        }
        fft_ready = 1;
    }
    if (!fft_ready) goto fail;

    decomp_mag.num_samples = 0; decomp_mag.num_uncompressed_samples = 0; decomp_mag.samples = NULL;
    decomp_phase.num_samples = 0; decomp_phase.num_uncompressed_samples = 0; decomp_phase.samples = NULL;
    decomp_time.num_samples = 0; decomp_time.num_uncompressed_samples = 0; decomp_time.samples = NULL;

    for (def_idx = 0; def_idx < num_unique; ++def_idx) {
        rd = &rf_defs[def_idx];
        first = -1; last = -1;

        rd->num_samples   = 0;
        rd->flip_angle    = 0.0f;
        rd->max_amplitude = 0.0f;
        rd->area          = 0.0f;
        rd->abswidth      = 0.0f;
        rd->effwidth      = 0.0f;
        rd->dtycyc        = 0.0f;
        rd->maxpw         = 0.0f;
        rd->duration_us   = 0.0f;
        rd->isodelay_us   = 0;
        rd->bandwidth     = 0.0f;

        /* max amplitude from table */
        if (rf_table && rf_table_size > 0) {
            for (i = 0; i < rf_table_size; ++i) {
                if (rf_table[i].id == def_idx) {
                    float amp = (float)fabs(rf_table[i].amplitude);
                    if (amp > rd->max_amplitude) rd->max_amplitude = amp;
                }
            }
        }

        mag_id   = rd->mag_shape_id;
        phase_id = rd->phase_shape_id;
        time_id  = rd->time_shape_id;
        has_phase = 0; has_time = 0;
        magnitude = NULL; phase = NULL; time_us = NULL;
        rf_re = NULL; rf_im = NULL; time_centered = NULL;
        num_samples = 0; duration = 0.0f;

        /* decompress magnitude */
        if (!pulseqlib__decompress_shape(&decomp_mag, &seq->shapes_library[mag_id - 1], 1.0f))
            goto fail;
        num_samples = decomp_mag.num_uncompressed_samples;
        magnitude = (float*)ALLOC(num_samples * sizeof(float));
        if (!magnitude) { FREE(decomp_mag.samples); goto fail; }
        for (i = 0; i < num_samples; ++i) magnitude[i] = decomp_mag.samples[i];
        FREE(decomp_mag.samples); decomp_mag.samples = NULL;
        rd->num_samples = num_samples;

        /* decompress phase (optional) */
        if (phase_id > 0 && phase_id <= seq->shapes_library_size) {
            if (!pulseqlib__decompress_shape(&decomp_phase, &seq->shapes_library[phase_id - 1], 1.0f))
                goto fail;
            phase = (float*)ALLOC(num_samples * sizeof(float));
            if (!phase) { FREE(decomp_phase.samples); goto fail; }
            for (i = 0; i < num_samples; ++i) phase[i] = decomp_phase.samples[i];
            has_phase = 1;
            FREE(decomp_phase.samples); decomp_phase.samples = NULL;
        }

        /* detect real-valued RF */
        if (has_phase && phase) {
            num_real = 0;
            for (i = 0; i < num_samples; ++i) {
                if ((float)fabs(phase[i]) < 1e-6f ||
                    (float)fabs(phase[i] - (float)M_PI) < 1e-6f)
                    ++num_real;
            }
            if (num_real == num_samples) {
                for (i = 0; i < num_samples; ++i)
                    if ((float)fabs(phase[i] - (float)M_PI) < 1e-6f)
                        magnitude[i] *= -1.0f;
                FREE(phase); phase = NULL; has_phase = 0;
            }
        }

        /* decompress time (optional) */
        if (time_id > 0 && time_id <= seq->shapes_library_size) {
            if (!pulseqlib__decompress_shape(&decomp_time, &seq->shapes_library[time_id - 1], rf_raster_us))
                goto fail;
            time_us = (float*)ALLOC(num_samples * sizeof(float));
            if (!time_us) { FREE(decomp_time.samples); goto fail; }
            for (i = 0; i < num_samples; ++i) time_us[i] = decomp_time.samples[i];
            has_time = 1;
            FREE(decomp_time.samples); decomp_time.samples = NULL;
        }
        if (!has_time) {
            time_us = (float*)ALLOC(num_samples * sizeof(float));
            if (!time_us) goto fail;
            for (i = 0; i < num_samples; ++i) time_us[i] = (float)i * rf_raster_us;
            has_time = 1;
        }

        duration = (has_time && num_samples > 0) ? time_us[num_samples - 1] : (num_samples * rf_raster_us);
        rd->duration_us = duration;

        /* find peak indices for isodelay */
        max_mag = pulseqlib__find_max_abs_real(magnitude, num_samples);
        for (i = 0; i < num_samples; ++i) {
            if ((float)fabs(magnitude[i]) >= 0.99999f * max_mag) {
                if (first < 0) first = i;
                last = i;
            }
        }
        if (first < 0) { first = 0; last = 0; }

        time_center = (has_time && time_us)
            ? 0.5f * (time_us[first] + time_us[last])
            : 0.5f * ((float)(first + last)) * rf_raster_us;
        rd->isodelay_us = (int)(duration - time_center);

        /* normalise */
        if (max_mag > 1e-9f)
            for (i = 0; i < num_samples; ++i) magnitude[i] /= max_mag;

        /* build complex RF */
        rf_re = (float*)ALLOC(num_samples * sizeof(float));
        rf_im = (float*)ALLOC(num_samples * sizeof(float));
        if (!rf_re || !rf_im) goto fail;
        if (has_phase && phase) {
            for (i = 0; i < num_samples; ++i) {
                rf_re[i] = magnitude[i] * (float)cos(phase[i]);
                rf_im[i] = magnitude[i] * (float)sin(phase[i]);
            }
        } else {
            for (i = 0; i < num_samples; ++i) { rf_re[i] = magnitude[i]; rf_im[i] = 0.0f; }
        }

        /* uniform grid */
        num_uniform = (int)(duration / rf_raster_us + 0.5f) + 1;
        if (num_uniform < 2) num_uniform = 2;

        time_us_uniform = (float*)ALLOC(num_uniform * sizeof(float));
        rf_re_uniform   = (float*)ALLOC(num_uniform * sizeof(float));
        rf_im_uniform   = (float*)ALLOC(num_uniform * sizeof(float));
        if (!time_us_uniform || !rf_re_uniform || !rf_im_uniform) goto fail;

        for (i = 0; i < num_uniform; ++i)
            time_us_uniform[i] = (float)i * rf_raster_us;

        pulseqlib__interp1_linear_complex(rf_re_uniform, rf_im_uniform,
                                          time_us_uniform, num_uniform,
                                          time_us, rf_re, rf_im, num_samples);

        /* compute stats */
        sum_signed_re = 0.0f; sum_signed_im = 0.0f;
        sum_abs = 0.0f; sum_sq = 0.0f;
        time_above_threshold = 0.0f; maxpw = 0.0f; temp_pw = 0.0f;

        for (i = 0; i < num_uniform; ++i) {
            sum_signed_re += rf_re_uniform[i];
            sum_signed_im += rf_im_uniform[i];
            rf_abs = (float)sqrt(rf_re_uniform[i] * rf_re_uniform[i] +
                                 rf_im_uniform[i] * rf_im_uniform[i]);
            sum_abs += rf_abs;
            sum_sq  += rf_abs * rf_abs;
            if (rf_abs > DTY_THRESHOLD) time_above_threshold += 1.0f;
            if (rf_abs >= MPW_THRESHOLD) { temp_pw += 1.0f; }
            else { if (temp_pw > maxpw) maxpw = temp_pw; temp_pw = 0.0f; }
        }
        if (temp_pw > maxpw) maxpw = temp_pw;

        sum_signed = (float)sqrt(sum_signed_re * sum_signed_re +
                                 sum_signed_im * sum_signed_im) *
                     seq->opts.rf_raster_time * 1e-6f;

        rd->area      = sum_signed;
        rd->abswidth  = sum_abs / num_uniform;
        rd->effwidth  = sum_sq  / num_uniform;
        rd->dtycyc    = time_above_threshold / num_uniform;
        rd->flip_angle = (float)PULSEQLIB__TWO_PI * rd->max_amplitude * sum_signed;
        rd->maxpw     = maxpw / num_uniform;
        if (rd->dtycyc < rd->maxpw) rd->dtycyc = rd->maxpw;

        FREE(time_us_uniform); time_us_uniform = NULL;
        FREE(rf_re_uniform);   rf_re_uniform = NULL;
        FREE(rf_im_uniform);   rf_im_uniform = NULL;

        /* bandwidth via FFT */
        if (fft_ready && time_us) {
            time_centered = (float*)ALLOC(num_samples * sizeof(float));
            if (time_centered) {
                for (i = 0; i < num_samples; ++i)
                    time_centered[i] = time_us[i] - time_center;
                pulseqlib__interp1_linear_complex(rfs_re, rfs_im,
                                                  tt, nn,
                                                  time_centered, rf_re, rf_im, num_samples);
                rd->bandwidth = compute_rf_bandwidth_fft(
                    rfs_re, rfs_im, fft_cfg, nn, dw, cutoff,
                    duration * 1e-6f, w, work_re, work_im, fft_in, fft_out);
                FREE(time_centered); time_centered = NULL;
            }
        }
        if (rf_re)    { FREE(rf_re);    rf_re = NULL; }
        if (rf_im)    { FREE(rf_im);    rf_im = NULL; }
        if (magnitude){ FREE(magnitude); magnitude = NULL; }
        if (phase)    { FREE(phase);     phase = NULL; }
        if (time_us)  { FREE(time_us);   time_us = NULL; }
    }

    if (tt)      FREE(tt);
    if (w)       FREE(w);
    if (rfs_re)  FREE(rfs_re);
    if (rfs_im)  FREE(rfs_im);
    if (work_re) FREE(work_re);
    if (work_im) FREE(work_im);
    if (fft_in)  KISS_FFT_FREE(fft_in);
    if (fft_out) KISS_FFT_FREE(fft_out);
    if (fft_cfg) kiss_fft_free(fft_cfg);
    return PULSEQLIB_OK;

fail:
    if (tt)      FREE(tt);
    if (w)       FREE(w);
    if (rfs_re)  FREE(rfs_re);
    if (rfs_im)  FREE(rfs_im);
    if (work_re) FREE(work_re);
    if (work_im) FREE(work_im);
    if (fft_in)  KISS_FFT_FREE(fft_in);
    if (fft_out) KISS_FFT_FREE(fft_out);
    if (fft_cfg) kiss_fft_free(fft_cfg);
    if (magnitude)      FREE(magnitude);
    if (phase)          FREE(phase);
    if (time_us)        FREE(time_us);
    if (rf_re)          FREE(rf_re);
    if (rf_im)          FREE(rf_im);
    if (time_us_uniform) FREE(time_us_uniform);
    if (rf_re_uniform)  FREE(rf_re_uniform);
    if (rf_im_uniform)  FREE(rf_im_uniform);
    if (time_centered)  FREE(time_centered);
    return PULSEQLIB_ERR_ALLOC_FAILED;
}

#endif /* PULSEQLIB_VENDOR_GEHC */

/* ================================================================== */
/*  Copy auxiliary libraries                                           */
/* ================================================================== */

static int copy_rotation_library(const pulseqlib__seq_file* seq, pulseqlib_sequence_descriptor* desc)
{
    int i, num = seq->rotation_library_size;

    desc->num_rotations = 0;
    desc->rotation_matrices = NULL;
    if (num <= 0 || !seq->rotation_quaternion_library) return PULSEQLIB_OK;

    desc->rotation_matrices = (float(*)[9])ALLOC(num * sizeof(float[9]));
    if (!desc->rotation_matrices) return PULSEQLIB_ERR_ALLOC_FAILED;

    for (i = 0; i < num; ++i)
        pulseqlib__quaternion_to_matrix(desc->rotation_matrices[i], seq->rotation_quaternion_library[i]);
    desc->num_rotations = num;
    return PULSEQLIB_OK;
}

static int copy_trigger_library(const pulseqlib__seq_file* seq, pulseqlib_sequence_descriptor* desc)
{
    int i, num = seq->trigger_library_size;

    desc->num_triggers = 0;
    desc->trigger_events = NULL;
    if (num <= 0 || !seq->trigger_library) return PULSEQLIB_OK;

    desc->trigger_events = (pulseqlib_trigger_event*)ALLOC(num * sizeof(pulseqlib_trigger_event));
    if (!desc->trigger_events) return PULSEQLIB_ERR_ALLOC_FAILED;

    for (i = 0; i < num; ++i) {
        desc->trigger_events[i].type            = 1;
        desc->trigger_events[i].trigger_type    = (int)seq->trigger_library[i][0];
        desc->trigger_events[i].trigger_channel = (int)seq->trigger_library[i][1];
        desc->trigger_events[i].delay           = (long)seq->trigger_library[i][2];
        desc->trigger_events[i].duration        = (long)seq->trigger_library[i][3];
    }
    desc->num_triggers = num;
    return PULSEQLIB_OK;
}

static int copy_shapes_library(const pulseqlib__seq_file* seq, pulseqlib_sequence_descriptor* desc)
{
    int i, j, num = seq->shapes_library_size;
    int ns;

    desc->num_shapes = 0;
    desc->shapes = NULL;
    if (num <= 0 || !seq->shapes_library) return PULSEQLIB_OK;

    desc->shapes = (pulseqlib_shape_arbitrary*)ALLOC(num * sizeof(pulseqlib_shape_arbitrary));
    if (!desc->shapes) return PULSEQLIB_ERR_ALLOC_FAILED;

    for (i = 0; i < num; ++i) {
        desc->shapes[i].num_samples = 0;
        desc->shapes[i].num_uncompressed_samples = 0;
        desc->shapes[i].samples = NULL;
    }
    for (i = 0; i < num; ++i) {
        ns = seq->shapes_library[i].num_samples;
        desc->shapes[i].num_samples = ns;
        desc->shapes[i].num_uncompressed_samples = seq->shapes_library[i].num_uncompressed_samples;
        if (ns > 0 && seq->shapes_library[i].samples) {
            desc->shapes[i].samples = (float*)ALLOC(ns * sizeof(float));
            if (!desc->shapes[i].samples) {
                for (j = 0; j < i; ++j)
                    if (desc->shapes[j].samples) FREE(desc->shapes[j].samples);
                FREE(desc->shapes);
                desc->shapes = NULL;
                return PULSEQLIB_ERR_ALLOC_FAILED;
            }
            memcpy(desc->shapes[i].samples, seq->shapes_library[i].samples,
                   ns * sizeof(float));
        }
    }
    desc->num_shapes = num;
    return PULSEQLIB_OK;
}

/* ================================================================== */
/*  get_unique_blocks                                                  */
/* ================================================================== */

int pulseqlib__get_unique_blocks(pulseqlib_sequence_descriptor* desc, const pulseqlib__seq_file* seq)
{
    int result, num_blocks, num_unique_rf, num_unique_grad, num_unique_adc;
    int n;

    pulseqlib_rf_definition*       tmp_rf_defs   = NULL;
    pulseqlib_rf_table_element*    tmp_rf_tab    = NULL;
    pulseqlib_grad_definition*     tmp_grad_defs = NULL;
    pulseqlib_grad_table_element*  tmp_grad_tab  = NULL;
    pulseqlib_adc_definition*      tmp_adc_defs  = NULL;
    pulseqlib_adc_table_element*   tmp_adc_tab   = NULL;
    pulseqlib_block_definition*    tmp_blk_defs  = NULL;
    pulseqlib_block_table_element* tmp_blk_tab   = NULL;

    int (*int_rows)[BLOCK_DEF_COLS] = NULL;
    int* unique_defs  = NULL;
    int* event_table  = NULL;

    pulseqlib__raw_block raw;
    pulseqlib__raw_extension ext;
    int norot_flag, nopos_flag, once_flag, pmc_flag, nav_flag, once_counter;
    int has_prep, has_cooldown, ctrl;

    if (!seq || !desc) return PULSEQLIB_ERR_INVALID_ARGUMENT;

    num_blocks = seq->num_blocks;
    if (num_blocks <= 0 || !seq->block_library) return PULSEQLIB_ERR_INVALID_ARGUMENT;

    desc->num_prep_blocks    = 0;
    desc->num_cooldown_blocks = 0;
    desc->num_unique_rfs     = 0;
    desc->num_unique_grads   = 0;
    desc->num_unique_adcs    = 0;
    desc->num_unique_blocks  = 0;
    desc->num_blocks         = 0;
    desc->rf_table_size      = 0;
    desc->grad_table_size    = 0;
    desc->adc_table_size     = 0;

    /* rasters */
    desc->rf_raster_time_us = (seq->reserved_definitions_library.radiofrequency_raster_time > 0.0f)
        ? seq->reserved_definitions_library.radiofrequency_raster_time
        : seq->opts.rf_raster_time;
    desc->grad_raster_time_us = (seq->reserved_definitions_library.gradient_raster_time > 0.0f)
        ? seq->reserved_definitions_library.gradient_raster_time
        : seq->opts.grad_raster_time;
    desc->adc_raster_time_us = (seq->reserved_definitions_library.adc_raster_time > 0.0f)
        ? seq->reserved_definitions_library.adc_raster_time
        : seq->opts.adc_raster_time;
    desc->block_duration_raster_us = (seq->reserved_definitions_library.block_duration_raster > 0.0f)
        ? seq->reserved_definitions_library.block_duration_raster
        : seq->opts.block_duration_raster;

    /* ---- allocate temp arrays ---- */
    if (seq->rf_library_size > 0) {
        tmp_rf_defs = (pulseqlib_rf_definition*)ALLOC(seq->rf_library_size * sizeof(pulseqlib_rf_definition));
        tmp_rf_tab  = (pulseqlib_rf_table_element*)ALLOC(seq->rf_library_size * sizeof(pulseqlib_rf_table_element));
        if (!tmp_rf_defs || !tmp_rf_tab) goto fail;
    }
    if (seq->grad_library_size > 0) {
        tmp_grad_defs = (pulseqlib_grad_definition*)ALLOC(seq->grad_library_size * sizeof(pulseqlib_grad_definition));
        tmp_grad_tab  = (pulseqlib_grad_table_element*)ALLOC(seq->grad_library_size * sizeof(pulseqlib_grad_table_element));
        if (!tmp_grad_defs || !tmp_grad_tab) goto fail;
    }
    if (seq->adc_library_size > 0) {
        tmp_adc_defs = (pulseqlib_adc_definition*)ALLOC(seq->adc_library_size * sizeof(pulseqlib_adc_definition));
        tmp_adc_tab  = (pulseqlib_adc_table_element*)ALLOC(seq->adc_library_size * sizeof(pulseqlib_adc_table_element));
        if (!tmp_adc_defs || !tmp_adc_tab) goto fail;
    }
    tmp_blk_defs = (pulseqlib_block_definition*)ALLOC(num_blocks * sizeof(pulseqlib_block_definition));
    tmp_blk_tab  = (pulseqlib_block_table_element*)ALLOC(num_blocks * sizeof(pulseqlib_block_table_element));
    if (!tmp_blk_defs || !tmp_blk_tab) goto fail;

    /* ---- step 1: dedup event libraries ---- */
    if (seq->rf_library_size > 0) {
        num_unique_rf = deduplicate_rf_library(seq, tmp_rf_defs, tmp_rf_tab);
        desc->num_unique_rfs = num_unique_rf;
        desc->rf_table_size  = seq->rf_library_size;
#if PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_GEHC
        result = compute_rf_stats(seq, tmp_rf_defs, num_unique_rf, tmp_rf_tab, seq->rf_library_size);
        if (PULSEQLIB_FAILED(result)) goto fail;
#endif
    }
    if (seq->grad_library_size > 0) {
        num_unique_grad = deduplicate_grad_library(seq, tmp_grad_defs, tmp_grad_tab);
        desc->num_unique_grads = num_unique_grad;
        desc->grad_table_size  = seq->grad_library_size;

        result = compute_grad_shot_indices(seq, tmp_grad_defs, tmp_grad_tab, num_unique_grad);
        if (PULSEQLIB_FAILED(result)) goto fail;

        result = compute_grad_stats(seq, tmp_grad_defs, num_unique_grad, tmp_grad_tab, seq->grad_library_size);
        if (PULSEQLIB_FAILED(result)) goto fail;
    }
    if (seq->adc_library_size > 0) {
        num_unique_adc = deduplicate_adc_library(seq, tmp_adc_defs, tmp_adc_tab);
        desc->num_unique_adcs = num_unique_adc;
        desc->adc_table_size  = seq->adc_library_size;
    }

    /* ---- step 2: block definition matrix ---- */
    int_rows    = ALLOC(num_blocks * sizeof(*int_rows));
    unique_defs = (int*)ALLOC(num_blocks * sizeof(int));
    event_table = (int*)ALLOC(num_blocks * sizeof(int));
    if (!int_rows || !unique_defs || !event_table) goto fail;

    norot_flag = 0; nopos_flag = 0; once_flag = 0; pmc_flag = 1; nav_flag = 0;
    once_counter = 0;

    for (n = 0; n < num_blocks; ++n) {
        if (!pulseqlib__get_raw_block_content_ids(seq, &raw, n, 1)) {
            result = PULSEQLIB_ERR_INVALID_ARGUMENT;
            goto fail;
        }
        int_rows[n][0] = raw.block_duration >= 0 ? raw.block_duration : 0;
        int_rows[n][1] = (raw.rf >= 0 && tmp_rf_tab)   ? tmp_rf_tab[raw.rf].id   : -1;
        int_rows[n][2] = (raw.gx >= 0 && tmp_grad_tab) ? tmp_grad_tab[raw.gx].id : -1;
        int_rows[n][3] = (raw.gy >= 0 && tmp_grad_tab) ? tmp_grad_tab[raw.gy].id : -1;
        int_rows[n][4] = (raw.gz >= 0 && tmp_grad_tab) ? tmp_grad_tab[raw.gz].id : -1;

        tmp_blk_tab[n].rf_id  = raw.rf;
        tmp_blk_tab[n].gx_id  = raw.gx;
        tmp_blk_tab[n].gy_id  = raw.gy;
        tmp_blk_tab[n].gz_id  = raw.gz;
        tmp_blk_tab[n].adc_id = raw.adc;

        tmp_blk_tab[n].duration_us = (raw.rf < 0 && raw.gx < 0 && raw.gy < 0 &&
                                      raw.gz < 0 && raw.adc < 0)
            ? (int)(raw.block_duration * desc->block_duration_raster_us)
            : -1;

        if (raw.ext_count > 0 && seq->is_extensions_library_parsed && seq->extension_lut) {
            pulseqlib__get_raw_extension(seq, &ext, &raw);
            tmp_blk_tab[n].rotation_id = ext.rotation_index;
            tmp_blk_tab[n].trigger_id  = ext.trigger_index;
            norot_flag = (ext.flag.norot >= 0) ? ext.flag.norot : norot_flag;
            nopos_flag = (ext.flag.nopos >= 0) ? ext.flag.nopos : nopos_flag;
            pmc_flag   = (ext.flag.pmc   >= 0) ? ext.flag.pmc   : pmc_flag;
            nav_flag   = (ext.flag.nav   >= 0) ? ext.flag.nav   : nav_flag;
            once_flag  = (ext.flag.once  >= 0) ? ext.flag.once  : once_flag;
            if (once_flag > 0) ++once_counter;
        } else {
            tmp_blk_tab[n].rotation_id = -1;
            tmp_blk_tab[n].trigger_id  = -1;
        }
        tmp_blk_tab[n].norot_flag = norot_flag;
        tmp_blk_tab[n].nopos_flag = nopos_flag;
        tmp_blk_tab[n].pmc_flag   = pmc_flag;
        tmp_blk_tab[n].once_flag  = once_flag;
        tmp_blk_tab[n].nav_flag   = nav_flag;
    }

    /* step 3: dedup blocks */
    desc->num_unique_blocks = deduplicate_int_rows(unique_defs, event_table, (const int*)int_rows, num_blocks, BLOCK_DEF_COLS);
    desc->num_blocks = num_blocks;

    for (n = 0; n < desc->num_unique_blocks; ++n) {
        tmp_blk_defs[n].id          = unique_defs[n];
        tmp_blk_defs[n].duration_us = (int)(int_rows[unique_defs[n]][0] * desc->block_duration_raster_us);
        tmp_blk_defs[n].rf_id       = int_rows[unique_defs[n]][1];
        tmp_blk_defs[n].gx_id       = int_rows[unique_defs[n]][2];
        tmp_blk_defs[n].gy_id       = int_rows[unique_defs[n]][3];
        tmp_blk_defs[n].gz_id       = int_rows[unique_defs[n]][4];
    }
    for (n = 0; n < num_blocks; ++n)
        tmp_blk_tab[n].id = event_table[n];

    FREE(int_rows);    int_rows    = NULL;
    FREE(unique_defs); unique_defs = NULL;
    FREE(event_table); event_table = NULL;

    /* ---- step 4: copy to output (exact sizes) ---- */
#define COPY_ARRAY(dst, src, cnt, type)                                      \
    do {                                                                     \
        if ((cnt) > 0) {                                                     \
            (dst) = (type*)ALLOC((cnt) * sizeof(type));                      \
            if (!(dst)) {                                                    \
                result = PULSEQLIB_ERR_ALLOC_FAILED;                         \
                pulseqlib_sequence_descriptor_free(desc);                    \
                goto fail;                                                   \
            }                                                                \
            memcpy((dst), (src), (cnt) * sizeof(type));                      \
        }                                                                    \
    } while (0)

    COPY_ARRAY(desc->rf_definitions,    tmp_rf_defs,   desc->num_unique_rfs,   pulseqlib_rf_definition);
    COPY_ARRAY(desc->rf_table,          tmp_rf_tab,    desc->rf_table_size,    pulseqlib_rf_table_element);
    COPY_ARRAY(desc->grad_definitions,  tmp_grad_defs, desc->num_unique_grads, pulseqlib_grad_definition);
    COPY_ARRAY(desc->grad_table,        tmp_grad_tab,  desc->grad_table_size,  pulseqlib_grad_table_element);
    COPY_ARRAY(desc->adc_definitions,   tmp_adc_defs,  desc->num_unique_adcs,  pulseqlib_adc_definition);
    COPY_ARRAY(desc->adc_table,         tmp_adc_tab,   desc->adc_table_size,   pulseqlib_adc_table_element);
    COPY_ARRAY(desc->block_definitions, tmp_blk_defs,  desc->num_unique_blocks, pulseqlib_block_definition);
    COPY_ARRAY(desc->block_table,       tmp_blk_tab,   num_blocks,             pulseqlib_block_table_element);

#undef COPY_ARRAY

    /* free temps - done with them */
    if (tmp_rf_defs)   FREE(tmp_rf_defs);   tmp_rf_defs   = NULL;
    if (tmp_rf_tab)    FREE(tmp_rf_tab);    tmp_rf_tab    = NULL;
    if (tmp_grad_defs) FREE(tmp_grad_defs); tmp_grad_defs = NULL;
    if (tmp_grad_tab)  FREE(tmp_grad_tab);  tmp_grad_tab  = NULL;
    if (tmp_adc_defs)  FREE(tmp_adc_defs);  tmp_adc_defs  = NULL;
    if (tmp_adc_tab)   FREE(tmp_adc_tab);   tmp_adc_tab   = NULL;
    if (tmp_blk_defs)  FREE(tmp_blk_defs);  tmp_blk_defs  = NULL;
    if (tmp_blk_tab)   FREE(tmp_blk_tab);   tmp_blk_tab   = NULL;

    /* ---- step 5: auxiliary libraries ---- */
    result = copy_rotation_library(seq, desc);
    if (PULSEQLIB_FAILED(result)) { pulseqlib_sequence_descriptor_free(desc); return result; }
    result = copy_trigger_library(seq, desc);
    if (PULSEQLIB_FAILED(result)) { pulseqlib_sequence_descriptor_free(desc); return result; }
    result = copy_shapes_library(seq, desc);
    if (PULSEQLIB_FAILED(result)) { pulseqlib_sequence_descriptor_free(desc); return result; }

    /* ---- step 6: prep/cooldown ---- */
    has_prep = 0; has_cooldown = 0;
    for (n = 0; n < seq->labelset_library_size; ++n) {
        if ((int)(seq->labelset_library[n][1]) == PULSEQLIB__ONCE) {
            if ((int)(seq->labelset_library[n][0]) == 1) has_prep = 1;
            else if ((int)(seq->labelset_library[n][0]) == 2) has_cooldown = 1;
        }
    }
    if (!has_prep && !has_cooldown) return PULSEQLIB_OK;

    if (has_prep) {
        pulseqlib__get_raw_block_content_ids(seq, &raw, 0, 1);
        pulseqlib__get_raw_extension(seq, &ext, &raw);
        if (ext.flag.once != 1) {
            pulseqlib_sequence_descriptor_free(desc);
            return PULSEQLIB_ERR_INVALID_PREP_POSITION;
        }
        ctrl = 0;
        desc->num_prep_blocks = 1;
        while (ctrl == 0 && desc->num_prep_blocks < num_blocks) {
            pulseqlib__get_raw_block_content_ids(seq, &raw, desc->num_prep_blocks, 1);
            pulseqlib__get_raw_extension(seq, &ext, &raw);
            if (ext.flag.once != 0)
                desc->num_prep_blocks++;
            else
                ctrl = 1;
        }
    }
    if (has_cooldown) {
        ctrl = 0;
        desc->num_cooldown_blocks = 0;
        while (ctrl == 0 && desc->num_cooldown_blocks < num_blocks) {
            pulseqlib__get_raw_block_content_ids(seq, &raw, num_blocks - 1 - desc->num_cooldown_blocks, 1);
            pulseqlib__get_raw_extension(seq, &ext, &raw);
            if (ext.flag.once != 2)
                desc->num_cooldown_blocks++;
            else
                ctrl = 1;
        }
        if (ctrl == 0) {
            pulseqlib_sequence_descriptor_free(desc);
            return PULSEQLIB_ERR_INVALID_COOLDOWN_POSITION;
        }
    }
    if (once_counter != (desc->num_prep_blocks > 0 ? 1 : 0) + (desc->num_cooldown_blocks > 0 ? 1 : 0)) {
        pulseqlib_sequence_descriptor_free(desc);
        return PULSEQLIB_ERR_INVALID_ONCE_FLAGS;
    }
    return PULSEQLIB_OK;

fail:
    if (tmp_rf_defs)   FREE(tmp_rf_defs);
    if (tmp_rf_tab)    FREE(tmp_rf_tab);
    if (tmp_grad_defs) FREE(tmp_grad_defs);
    if (tmp_grad_tab)  FREE(tmp_grad_tab);
    if (tmp_adc_defs)  FREE(tmp_adc_defs);
    if (tmp_adc_tab)   FREE(tmp_adc_tab);
    if (tmp_blk_defs)  FREE(tmp_blk_defs);
    if (tmp_blk_tab)   FREE(tmp_blk_tab);
    if (int_rows)      FREE(int_rows);
    if (unique_defs)   FREE(unique_defs);
    if (event_table)   FREE(event_table);
    return PULSEQLIB_ERR_ALLOC_FAILED;
}

/* ================================================================== */
/*  TR detection helpers                                               */
/* ================================================================== */

static long long sum_durations_us(const int* dur, int start, int count)
{
    long long total = 0;
    int i;
    for (i = 0; i < count; ++i) total += (long long)dur[start + i];
    return total;
}

static int first_repeating_segment(const int* s, int len)
{
    int start, sub_len, l, i, match;
    const int* p;

    if (len <= 1) return len;

    for (start = 0; start < len; ++start) {
        p = s + start;
        sub_len = len - start;
        for (l = 1; l <= sub_len / 2; ++l) {
            match = 1;
            for (i = 0; i < l; ++i) {
                if (p[i] != p[i + l]) { match = 0; break; }
            }
            if (match) return l;
        }
    }
    return len;
}

/* ================================================================== */
/*  find_tr_in_sequence                                                */
/* ================================================================== */

int pulseqlib__find_tr_in_sequence(pulseqlib_sequence_descriptor* desc, pulseqlib_diagnostic* diag)
{
    pulseqlib_tr_descriptor* tr = &desc->tr_descriptor;
    pulseqlib_diagnostic local_diag;
    int i, n;
    int imaging_start, imaging_end, imaging_len;
    int* seq_pat       = NULL;
    int* block_dur     = NULL;
    int prep_dur_us, cooldown_dur_us, active_dur_us;
    int found, l;
    float tr_dur;
    int tr_start;

    if (!diag) { pulseqlib_diagnostic_init(&local_diag); diag = &local_diag; }
    else       pulseqlib_diagnostic_init(diag);

    found = 0; l = 0;

    if (desc->num_blocks <= 0 || !desc->block_table || !desc->block_definitions) {
        diag->code = (desc->num_blocks <= 0)
            ? PULSEQLIB_ERR_TR_NO_BLOCKS : PULSEQLIB_ERR_NULL_POINTER;
        return diag->code;
    }
    if (desc->num_prep_blocks < 0 || desc->num_cooldown_blocks < 0) {
        diag->code = PULSEQLIB_ERR_INVALID_ARGUMENT;
        return diag->code;
    }
    if (desc->num_prep_blocks + desc->num_cooldown_blocks > desc->num_blocks) {
        diag->code = PULSEQLIB_ERR_TR_NO_IMAGING_REGION;
        return diag->code;
    }

    tr->tr_size             = 0;
    tr->num_trs             = 0;
    tr->tr_duration_us      = 0.0f;
    tr->degenerate_prep     = 1;
    tr->num_prep_blocks     = desc->num_prep_blocks;
    tr->num_prep_trs        = 1;
    tr->degenerate_cooldown = 1;
    tr->num_cooldown_blocks = desc->num_cooldown_blocks;
    tr->num_cooldown_trs    = 1;

    imaging_start = desc->num_prep_blocks;
    imaging_end   = desc->num_blocks - desc->num_cooldown_blocks;
    imaging_len   = imaging_end - imaging_start;
    diag->imaging_region_length = imaging_len;

    if (imaging_len <= 0) {
        diag->code = PULSEQLIB_ERR_TR_NO_IMAGING_REGION;
        return diag->code;
    }

    /* unique-block count for diagnostics */
    {
        int max_u = 0;
        for (n = 0; n < desc->num_blocks; ++n)
            if (desc->block_table[n].id > max_u) max_u = desc->block_table[n].id;
        diag->num_unique_blocks = max_u + 1;
    }

    seq_pat   = (int*)ALLOC(desc->num_blocks * sizeof(int));
    block_dur = (int*)ALLOC(desc->num_blocks * sizeof(int));
    if (!seq_pat || !block_dur) {
        if (seq_pat)   FREE(seq_pat);
        if (block_dur) FREE(block_dur);
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        return diag->code;
    }

    for (n = 0; n < desc->num_blocks; ++n) {
        block_dur[n] = desc->block_definitions[desc->block_table[n].id].duration_us;
        seq_pat[n] = (desc->block_table[n].duration_us >= 0)
            ? block_dur[n]
            : -1 * desc->block_table[desc->block_table[n].id].id;
    }

    l = first_repeating_segment(&seq_pat[imaging_start], imaging_len);
    diag->candidate_pattern_length = l;

    found = (l > 0 && l <= imaging_len) ? 1 : 0;

    if (found) {
        for (i = 0; i < imaging_len; ++i) {
            n = imaging_start + i;
            if (seq_pat[n] != seq_pat[imaging_start + (i % l)]) {
                diag->mismatch_position = i;
                diag->block_index = n;
                found = 0;
                break;
            }
        }
    }

    if (!found) {
        active_dur_us = 0;
        for (n = 0; n < desc->num_blocks; ++n)
            if (desc->block_table[n].duration_us < 0)
                active_dur_us += desc->block_definitions[desc->block_table[n].id].duration_us;

        if (active_dur_us <= SINGLE_TR_MAX_DURATION_US) {
            tr->tr_size             = desc->num_blocks;
            tr->num_trs             = 1;
            tr->degenerate_prep     = 1;
            tr->num_prep_blocks     = 0;
            tr->num_prep_trs        = 0;
            tr->degenerate_cooldown = 1;
            tr->num_cooldown_blocks = 0;
            tr->num_cooldown_trs    = 0;
            tr_dur = 0.0f;
            for (i = 0; i < desc->num_blocks; ++i)
                tr_dur += (float)block_dur[i];
            tr->tr_duration_us = tr_dur;
            diag->code = PULSEQLIB_OK;
            FREE(seq_pat); FREE(block_dur);
            return PULSEQLIB_OK;
        }
        diag->code = (diag->mismatch_position >= 0)
            ? PULSEQLIB_ERR_TR_PATTERN_MISMATCH
            : PULSEQLIB_ERR_TR_NO_PERIODIC_PATTERN;
        FREE(seq_pat); FREE(block_dur);
        return diag->code;
    }

    tr->tr_size = l;
    tr->num_trs = imaging_len / l;
    tr_dur = 0.0f;
    tr_start = imaging_start;
    for (i = 0; i < l; ++i) tr_dur += (float)block_dur[tr_start + i];
    tr->tr_duration_us = tr_dur;

    /* prep check */
    if (desc->num_prep_blocks) {
        if (desc->num_prep_blocks % l == 0) {
            for (n = 0; n < (int)(desc->num_prep_blocks / l); ++n) {
                if (!array_equal(&seq_pat[imaging_start], &seq_pat[n * l], l)) {
                    prep_dur_us = (int)sum_durations_us(block_dur, 0, desc->num_prep_blocks);
                    if (prep_dur_us > PREP_COOLDOWN_THRESHOLD_US) {
                        diag->code = PULSEQLIB_ERR_TR_PREP_TOO_LONG;
                        FREE(seq_pat); FREE(block_dur);
                        return diag->code;
                    }
                    tr->degenerate_prep = 0;
                    break;
                }
            }
            if (tr->degenerate_prep == 1) {
                tr->num_prep_blocks = 0;
                tr->num_prep_trs    = desc->num_prep_blocks / l;
            }
        } else {
            prep_dur_us = (int)sum_durations_us(block_dur, 0, desc->num_prep_blocks);
            if (prep_dur_us > PREP_COOLDOWN_THRESHOLD_US) {
                diag->code = PULSEQLIB_ERR_TR_PREP_TOO_LONG;
                FREE(seq_pat); FREE(block_dur);
                return diag->code;
            }
            tr->degenerate_prep = 0;
        }
    }

    /* cooldown check */
    if (desc->num_cooldown_blocks) {
        if (desc->num_cooldown_blocks % l == 0) {
            for (n = 0; n < (int)(desc->num_cooldown_blocks / l); ++n) {
                if (!array_equal(&seq_pat[imaging_start], &seq_pat[imaging_end + n * l], l)) {
                    cooldown_dur_us = (int)sum_durations_us(block_dur, imaging_end, desc->num_cooldown_blocks);
                    if (cooldown_dur_us > PREP_COOLDOWN_THRESHOLD_US) {
                        diag->code = PULSEQLIB_ERR_TR_COOLDOWN_TOO_LONG;
                        FREE(seq_pat); FREE(block_dur);
                        return diag->code;
                    }
                    tr->degenerate_cooldown = 0;
                    break;
                }
            }
            if (tr->degenerate_cooldown == 1) {
                tr->num_cooldown_blocks = 0;
                tr->num_cooldown_trs    = desc->num_cooldown_blocks / l;
            }
        } else {
            cooldown_dur_us = (int)sum_durations_us(block_dur, imaging_end, desc->num_cooldown_blocks);
            if (cooldown_dur_us > PREP_COOLDOWN_THRESHOLD_US) {
                diag->code = PULSEQLIB_ERR_TR_COOLDOWN_TOO_LONG;
                FREE(seq_pat); FREE(block_dur);
                return diag->code;
            }
            tr->degenerate_cooldown = 0;
        }
    }

    diag->code = PULSEQLIB_OK;
    FREE(seq_pat); FREE(block_dur);
    return PULSEQLIB_OK;
}

/* ================================================================== */
/*  Segment state machine                                              */
/* ================================================================== */

static int find_segments_internal(
    const pulseqlib_sequence_descriptor* desc, 
    pulseqlib_tr_segment* segs, int offset, 
    pulseqlib_diagnostic* diag, 
    const pulseqlib_opts* opts,
    int tr_start, int tr_size)
{
    float max_slew, grad_raster_s, max_allowed;
    int grad_ids[3];
    float phys_first, phys_last;
    float grad_last_cur[3], grad_first_next[3];
    const pulseqlib_block_definition* bdef;
    const pulseqlib_grad_definition* gdef;
    int bdef_id, shot_idx;
    int* seg_starts = NULL;
    int* seg_sizes  = NULL;
    int num_seg, seg_start;
    int state, cand_before_rf, saved_cand, has_saved_cand;
    int has_rf, has_adc, is_cand;
    int nb, n, i;

    max_slew = opts->max_slew;
    grad_raster_s = desc->grad_raster_time_us * 1e-6f;
    max_allowed = max_slew * grad_raster_s;
    nb = tr_size;

    seg_starts = (int*)ALLOC(nb * sizeof(int));
    seg_sizes  = (int*)ALLOC(nb * sizeof(int));
    if (!seg_starts || !seg_sizes) {
        if (seg_starts) FREE(seg_starts);
        if (seg_sizes)  FREE(seg_sizes);
        if (diag) diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        return 0;
    }

    /* first block gradient check */
    bdef_id = desc->block_table[tr_start].id;
    bdef = &desc->block_definitions[bdef_id];
    grad_ids[0] = bdef->gx_id; grad_ids[1] = bdef->gy_id; grad_ids[2] = bdef->gz_id;
    for (i = 0; i < 3; ++i) {
        if (grad_ids[i] < 0) continue;
        gdef = &desc->grad_definitions[grad_ids[i]];
        for (shot_idx = 0; shot_idx < gdef->num_shots; ++shot_idx) {
            phys_first = gdef->first_value[shot_idx] * gdef->max_amplitude[shot_idx];
            if ((float)fabs(phys_first) > max_allowed) {
                if (diag) {
                    diag->code = PULSEQLIB_ERR_SEG_NONZERO_START_GRAD;
                    diag->block_index = tr_start;
                    diag->channel = i;
                    diag->gradient_amplitude = phys_first;
                    diag->max_allowed_amplitude = max_allowed;
                }
                FREE(seg_starts); FREE(seg_sizes);
                return 0;
            }
        }
    }

    /* last block gradient check */
    bdef_id = desc->block_table[tr_start + nb - 1].id;
    bdef = &desc->block_definitions[bdef_id];
    grad_ids[0] = bdef->gx_id; grad_ids[1] = bdef->gy_id; grad_ids[2] = bdef->gz_id;
    for (i = 0; i < 3; ++i) {
        if (grad_ids[i] < 0) continue;
        gdef = &desc->grad_definitions[grad_ids[i]];
        for (shot_idx = 0; shot_idx < gdef->num_shots; ++shot_idx) {
            phys_last = gdef->last_value[shot_idx] * gdef->max_amplitude[shot_idx];
            if ((float)fabs(phys_last) > max_allowed) {
                if (diag) {
                    diag->code = PULSEQLIB_ERR_SEG_NONZERO_END_GRAD;
                    diag->block_index = tr_start + nb - 1;
                    diag->channel = i;
                    diag->gradient_amplitude = phys_last;
                    diag->max_allowed_amplitude = max_allowed;
                }
                FREE(seg_starts); FREE(seg_sizes);
                return 0;
            }
        }
    }

    /* state machine */
    num_seg = 0;
    seg_start = tr_start;
    state = SEGSTATE_SEEKING_FIRST_ADC;
    cand_before_rf = -1;
    saved_cand = -1;
    has_saved_cand = 0;

    for (n = tr_start; n < tr_start + nb; ++n) {
        is_cand = 0;
        if (n > tr_start) {
            is_cand = 1;

            bdef_id = desc->block_table[n - 1].id;
            bdef = &desc->block_definitions[bdef_id];
            grad_ids[0] = bdef->gx_id; grad_ids[1] = bdef->gy_id; grad_ids[2] = bdef->gz_id;
            for (i = 0; i < 3; ++i) {
                grad_last_cur[i] = 0.0f;
                if (grad_ids[i] >= 0) {
                    gdef = &desc->grad_definitions[grad_ids[i]];
                    for (shot_idx = 0; shot_idx < gdef->num_shots; ++shot_idx) {
                        phys_last = gdef->last_value[shot_idx] * gdef->max_amplitude[shot_idx];
                        if ((float)fabs(phys_last) > (float)fabs(grad_last_cur[i]))
                            grad_last_cur[i] = phys_last;
                    }
                }
            }

            bdef_id = desc->block_table[n].id;
            bdef = &desc->block_definitions[bdef_id];
            grad_ids[0] = bdef->gx_id; grad_ids[1] = bdef->gy_id; grad_ids[2] = bdef->gz_id;
            for (i = 0; i < 3; ++i) {
                grad_first_next[i] = 0.0f;
                if (grad_ids[i] >= 0) {
                    gdef = &desc->grad_definitions[grad_ids[i]];
                    for (shot_idx = 0; shot_idx < gdef->num_shots; ++shot_idx) {
                        phys_first = gdef->first_value[shot_idx] * gdef->max_amplitude[shot_idx];
                        if ((float)fabs(phys_first) > (float)fabs(grad_first_next[i]))
                            grad_first_next[i] = phys_first;
                    }
                }
            }

            for (i = 0; i < 3; ++i) {
                if ((float)fabs(grad_last_cur[i]) > max_allowed ||
                    (float)fabs(grad_first_next[i]) > max_allowed) {
                    is_cand = 0; break;
                }
            }
        }

        has_rf  = (desc->block_definitions[desc->block_table[n].id].rf_id >= 0);
        has_adc = (desc->block_table[n].adc_id >= 0);

        if (state == SEGSTATE_SEEKING_FIRST_ADC) {
            if (is_cand) saved_cand = n;
            if (has_rf)  { cand_before_rf = saved_cand; saved_cand = -1; }
            if (has_adc) {
                if (cand_before_rf > seg_start) {
                    seg_starts[num_seg] = seg_start;
                    seg_sizes[num_seg]  = cand_before_rf - seg_start;
                    num_seg++;
                    seg_start = cand_before_rf;
                }
                state = SEGSTATE_SEEKING_BOUNDARY;
                has_saved_cand = 0;
                saved_cand = -1;
            }
        } else if (state == SEGSTATE_SEEKING_BOUNDARY) {
            if (is_cand) { saved_cand = n; has_saved_cand = 1; }
            if (has_rf) {
                if (has_saved_cand) {
                    seg_starts[num_seg] = seg_start;
                    seg_sizes[num_seg]  = saved_cand - seg_start;
                    num_seg++;
                    seg_start = saved_cand;
                    has_saved_cand = 0;
                    saved_cand = -1;
                } else {
                    state = SEGSTATE_OPTIMIZED_MODE;
                }
            }
        }
        /* SEGSTATE_OPTIMIZED_MODE: no action */
    }

    seg_starts[num_seg] = seg_start;
    seg_sizes[num_seg]  = tr_start + nb - seg_start;
    num_seg++;

    for (i = 0; i < num_seg; ++i) {
        segs[offset + i].start_block = seg_starts[i];
        segs[offset + i].num_blocks  = seg_sizes[i];
        segs[offset + i].unique_block_indices = NULL;
    }

    FREE(seg_starts); FREE(seg_sizes);
    return num_seg;
}

/* ================================================================== */
/*  Strip pure delays from segments                                    */
/* ================================================================== */

static int strip_pure_delays(
    const pulseqlib_tr_segment* raw_segs, int num_raw,
    pulseqlib_tr_segment* out, int max_out,
    const pulseqlib_block_table_element* bt
) {
    int num_out = 0;
    int s, i, n_blk;
    int leading, trailing, core_start, core_end, core_size;
    const int* idx;

    for (s = 0; s < num_raw; ++s) {
        n_blk = raw_segs[s].num_blocks;
        idx   = raw_segs[s].unique_block_indices;
        if (n_blk == 0 || !idx) continue;

        leading = 0;
        for (i = 0; i < n_blk; ++i) {
            if (bt[raw_segs[s].start_block + i].duration_us >= 0) leading++;
            else break;
        }
        trailing = 0;
        for (i = n_blk - 1; i >= leading; --i) {
            if (bt[raw_segs[s].start_block + i].duration_us >= 0) trailing++;
            else break;
        }
        core_start = leading;
        core_end   = n_blk - trailing;

        for (i = 0; i < leading; ++i) {
            if (num_out >= max_out) return -1;
            out[num_out].start_block = raw_segs[s].start_block + i;
            out[num_out].num_blocks  = 1;
            out[num_out].unique_block_indices = (int*)ALLOC(sizeof(int));
            if (!out[num_out].unique_block_indices) return -1;
            out[num_out].unique_block_indices[0] = idx[i];
            num_out++;
        }
        if (core_end > core_start) {
            core_size = core_end - core_start;
            if (num_out >= max_out) return -1;
            out[num_out].start_block = raw_segs[s].start_block + core_start;
            out[num_out].num_blocks  = core_size;
            out[num_out].unique_block_indices = (int*)ALLOC(core_size * sizeof(int));
            if (!out[num_out].unique_block_indices) return -1;
            for (i = 0; i < core_size; ++i)
                out[num_out].unique_block_indices[i] = idx[core_start + i];
            num_out++;
        }
        for (i = 0; i < trailing; ++i) {
            if (num_out >= max_out) return -1;
            out[num_out].start_block = raw_segs[s].start_block + core_end + i;
            out[num_out].num_blocks  = 1;
            out[num_out].unique_block_indices = (int*)ALLOC(sizeof(int));
            if (!out[num_out].unique_block_indices) return -1;
            out[num_out].unique_block_indices[0] = idx[core_end + i];
            num_out++;
        }
    }
    return num_out;
}

/* ================================================================== */
/*  find_segments_in_tr                                                */
/* ================================================================== */

int pulseqlib__find_segments_in_tr(pulseqlib_sequence_descriptor* desc, pulseqlib_diagnostic* diag, const pulseqlib__seq_file* seq)
{
    const pulseqlib_tr_descriptor* tr = &desc->tr_descriptor;
    const pulseqlib_block_table_element* bte;
    const pulseqlib_block_definition* bdef;
    pulseqlib_tr_segment* raw_segs  = NULL;
    pulseqlib_tr_segment* exp_segs  = NULL;
    pulseqlib_tr_segment* uniq_segs = NULL;
    pulseqlib_diagnostic local_diag;
    int total_blocks;
    int num_raw, n_prep_raw, n_main_raw, n_cool_raw;
    int n_prep, n_main, n_cool;
    int num_total, num_unique;
    int found, tr_start, tr_size;
    int n, b, i, offset;
    int max_expanded, seg_result;
    int pure_delay_idx, is_pure;
    int nb, unique_idx, blk_tab_idx, blk_def_id, shot_idx;
    int ax_grad_ids[3], ax_def_ids[3], ax;
    float* max_energy = NULL;
    float inst_energy, e, amp;
    int num_raw_alloc, num_exp_alloc;

    if (!diag) { pulseqlib_diagnostic_init(&local_diag); diag = &local_diag; }
    else       pulseqlib_diagnostic_init(diag);

    if (!seq || !desc) { diag->code = PULSEQLIB_ERR_NULL_POINTER; return 0; }

    n_prep_raw = 0; n_main_raw = 0; n_cool_raw = 0; num_raw = 0;
    n_prep = 0; n_main = 0; n_cool = 0;
    num_total = 0; num_unique = 0;
    num_raw_alloc = 0; num_exp_alloc = 0;
    total_blocks = tr->tr_size + tr->num_prep_blocks + tr->num_cooldown_blocks;

    raw_segs = (pulseqlib_tr_segment*)ALLOC(total_blocks * sizeof(pulseqlib_tr_segment));
    if (!raw_segs) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; return 0; }

    /* ---- raw segments per section ---- */
    if (tr->degenerate_prep == 0 && tr->num_prep_blocks > 0) {
        tr_start = 0;
        tr_size  = tr->num_prep_blocks + tr->tr_size;
        seg_result = find_segments_internal(desc, raw_segs, num_raw, diag, &seq->opts, tr_start, tr_size);
        if (seg_result == 0 && PULSEQLIB_FAILED(diag->code)) { FREE(raw_segs); return 0; }
        n_prep_raw = seg_result;
        num_raw += n_prep_raw;
    }

    tr_start = tr->num_prep_blocks;
    tr_size  = tr->tr_size;
    seg_result = find_segments_internal(desc, raw_segs, num_raw, diag, &seq->opts, tr_start, tr_size);
    if (seg_result == 0 && PULSEQLIB_FAILED(diag->code)) { FREE(raw_segs); return 0; }
    n_main_raw = seg_result;
    num_raw += n_main_raw;

    if (tr->degenerate_cooldown == 0 && tr->num_cooldown_blocks > 0) {
        tr_start = seq->num_blocks - tr->num_cooldown_blocks - tr->tr_size;
        tr_size  = tr->num_cooldown_blocks + tr->tr_size;
        seg_result = find_segments_internal(desc, raw_segs, num_raw, diag, &seq->opts, tr_start, tr_size);
        if (seg_result == 0 && PULSEQLIB_FAILED(diag->code)) { FREE(raw_segs); return 0; }
        n_cool_raw = seg_result;
        num_raw += n_cool_raw;
    }

    if (num_raw == 0) {
        diag->code = PULSEQLIB_ERR_SEG_NO_SEGMENTS_FOUND;
        FREE(raw_segs);
        return 0;
    }

    /* populate unique_block_indices */
    for (n = 0; n < num_raw; ++n) {
        raw_segs[n].unique_block_indices = (int*)ALLOC(raw_segs[n].num_blocks * sizeof(int));
        if (!raw_segs[n].unique_block_indices) {
            diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
            num_raw_alloc = n;
            goto fail;
        }
        for (i = 0; i < raw_segs[n].num_blocks; ++i)
            raw_segs[n].unique_block_indices[i] = desc->block_table[raw_segs[n].start_block + i].id;
    }
    num_raw_alloc = num_raw;

    /* ---- strip pure delays ---- */
    max_expanded = total_blocks;
    exp_segs = (pulseqlib_tr_segment*)ALLOC(max_expanded * sizeof(pulseqlib_tr_segment));
    if (!exp_segs) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto fail; }

    offset = 0;
    if (n_prep_raw > 0) {
        n_prep = strip_pure_delays(raw_segs, n_prep_raw, exp_segs + offset, max_expanded - offset, desc->block_table);
        if (n_prep < 0) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto fail; }
        offset += n_prep;
    }

    n_main = strip_pure_delays(raw_segs + n_prep_raw, n_main_raw, exp_segs + offset, max_expanded - offset, desc->block_table);
    if (n_main < 0) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto fail; }
    offset += n_main;

    if (n_cool_raw > 0) {
        n_cool = strip_pure_delays(raw_segs + n_prep_raw + n_main_raw, n_cool_raw, exp_segs + offset, max_expanded - offset, desc->block_table);
        if (n_cool < 0) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto fail; }
        offset += n_cool;
    }

    num_total = n_prep + n_main + n_cool;
    num_exp_alloc = num_total;

    /* raw_segs no longer needed */
    for (n = 0; n < num_raw_alloc; ++n) FREE(raw_segs[n].unique_block_indices);
    FREE(raw_segs); raw_segs = NULL;
    num_raw_alloc = 0;

    /* ---- segment tables ---- */
    desc->segment_table.num_prep_segments     = n_prep;
    desc->segment_table.num_main_segments     = n_main;
    desc->segment_table.num_cooldown_segments = n_cool;
    desc->segment_table.prep_segment_table     = (n_prep > 0) ? (int*)ALLOC(n_prep * sizeof(int)) : NULL;
    desc->segment_table.main_segment_table     = (n_main > 0) ? (int*)ALLOC(n_main * sizeof(int)) : NULL;
    desc->segment_table.cooldown_segment_table = (n_cool > 0) ? (int*)ALLOC(n_cool * sizeof(int)) : NULL;

    uniq_segs = (pulseqlib_tr_segment*)ALLOC(num_total * sizeof(pulseqlib_tr_segment));
    if (!uniq_segs) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto fail; }

    num_unique = 0;
    pure_delay_idx = -1;

    for (n = 0; n < num_total; ++n) {
        is_pure = (exp_segs[n].num_blocks == 1 &&
                   desc->block_table[exp_segs[n].start_block].duration_us >= 0);

        if (is_pure) {
            if (pure_delay_idx == -1) {
                uniq_segs[num_unique].num_blocks  = 1;
                uniq_segs[num_unique].start_block = exp_segs[n].start_block;
                uniq_segs[num_unique].unique_block_indices = (int*)ALLOC(sizeof(int));
                if (!uniq_segs[num_unique].unique_block_indices) {
                    diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
                    goto fail;
                }
                uniq_segs[num_unique].unique_block_indices[0] = exp_segs[n].unique_block_indices[0];
                pure_delay_idx = num_unique;
                num_unique++;
            }
            found = pure_delay_idx;
        } else {
            found = -1;
            for (i = 0; i < num_unique; ++i) {
                if (i == pure_delay_idx) continue;
                if (exp_segs[n].num_blocks == uniq_segs[i].num_blocks &&
                    array_equal(exp_segs[n].unique_block_indices, uniq_segs[i].unique_block_indices, exp_segs[n].num_blocks)) {
                    found = i; break;
                }
            }
            if (found == -1) {
                uniq_segs[num_unique].num_blocks  = exp_segs[n].num_blocks;
                uniq_segs[num_unique].start_block = exp_segs[n].start_block;
                uniq_segs[num_unique].unique_block_indices =
                    (int*)ALLOC(exp_segs[n].num_blocks * sizeof(int));
                if (!uniq_segs[num_unique].unique_block_indices) {
                    diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
                    goto fail;
                }
                for (i = 0; i < exp_segs[n].num_blocks; ++i)
                    uniq_segs[num_unique].unique_block_indices[i] = exp_segs[n].unique_block_indices[i];
                found = num_unique;
                num_unique++;
            }
        }

        if (n < n_prep)
            desc->segment_table.prep_segment_table[n] = found;
        else if (n < n_prep + n_main)
            desc->segment_table.main_segment_table[n - n_prep] = found;
        else
            desc->segment_table.cooldown_segment_table[n - n_prep - n_main] = found;
    }

    desc->segment_table.num_unique_segments = num_unique;
    desc->num_unique_segments = num_unique;

    /* transfer ownership from uniq_segs to desc */
    desc->segment_definitions = (pulseqlib_tr_segment*)ALLOC(num_unique * sizeof(pulseqlib_tr_segment));
    if (!desc->segment_definitions) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto fail; }
    for (i = 0; i < num_unique; ++i)
        desc->segment_definitions[i] = uniq_segs[i];
    FREE(uniq_segs); uniq_segs = NULL;
    /* note: unique_block_indices pointers now owned by desc->segment_definitions */

    /* ---- per-block flags ---- */
    for (i = 0; i < num_unique; ++i) {
        nb = desc->segment_definitions[i].num_blocks;
        desc->segment_definitions[i].has_trigger  = (int*)ALLOC(nb * sizeof(int));
        desc->segment_definitions[i].has_rotation = (int*)ALLOC(nb * sizeof(int));
        desc->segment_definitions[i].norot_flag   = (int*)ALLOC(nb * sizeof(int));
        desc->segment_definitions[i].nopos_flag   = (int*)ALLOC(nb * sizeof(int));
        if (!desc->segment_definitions[i].has_trigger ||
            !desc->segment_definitions[i].has_rotation ||
            !desc->segment_definitions[i].norot_flag ||
            !desc->segment_definitions[i].nopos_flag) {
            diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
            goto fail;
        }
        for (n = 0; n < nb; ++n) {
            desc->segment_definitions[i].has_trigger[n]  = 0;
            desc->segment_definitions[i].has_rotation[n] = 0;
            desc->segment_definitions[i].norot_flag[n]   = 0;
            desc->segment_definitions[i].nopos_flag[n]   = 0;
        }
    }

    max_energy = (float*)ALLOC(num_unique * sizeof(float));
    if (!max_energy) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto fail; }
    for (i = 0; i < num_unique; ++i) {
        max_energy[i] = 0.0f;
        desc->segment_definitions[i].max_energy_start_block = 0;
    }

    for (n = 0; n < num_total; ++n) {
        if (n < n_prep)
            unique_idx = desc->segment_table.prep_segment_table[n];
        else if (n < n_prep + n_main)
            unique_idx = desc->segment_table.main_segment_table[n - n_prep];
        else
            unique_idx = desc->segment_table.cooldown_segment_table[n - n_prep - n_main];

        inst_energy = 0.0f;
        for (b = 0; b < exp_segs[n].num_blocks; ++b) {
            blk_tab_idx = exp_segs[n].start_block + b;
            bte = &desc->block_table[blk_tab_idx];
            blk_def_id = bte->id;
            bdef = &desc->block_definitions[blk_def_id];

            if (bte->trigger_id  != -1) desc->segment_definitions[unique_idx].has_trigger[b]  = 1;
            if (bte->rotation_id != -1) desc->segment_definitions[unique_idx].has_rotation[b] = 1;
            if (bte->norot_flag)        desc->segment_definitions[unique_idx].norot_flag[b]   = 1;
            if (bte->nopos_flag)        desc->segment_definitions[unique_idx].nopos_flag[b]   = 1;

            ax_grad_ids[0] = bte->gx_id; ax_grad_ids[1] = bte->gy_id; ax_grad_ids[2] = bte->gz_id;
            ax_def_ids[0]  = bdef->gx_id; ax_def_ids[1] = bdef->gy_id; ax_def_ids[2] = bdef->gz_id;

            for (ax = 0; ax < 3; ++ax) {
                if (ax_grad_ids[ax] >= 0 && ax_grad_ids[ax] < desc->grad_table_size &&
                    ax_def_ids[ax]  >= 0 && ax_def_ids[ax]  < desc->num_unique_grads) {
                    amp = desc->grad_table[ax_grad_ids[ax]].amplitude;
                    shot_idx = desc->grad_table[ax_grad_ids[ax]].shot_index;
                    e = desc->grad_definitions[ax_def_ids[ax]].energy[shot_idx];
                    inst_energy += e * amp * amp;
                }
            }
        }
        if (inst_energy > max_energy[unique_idx]) {
            max_energy[unique_idx] = inst_energy;
            desc->segment_definitions[unique_idx].max_energy_start_block = exp_segs[n].start_block;
        }
    }

    FREE(max_energy); max_energy = NULL;
    for (n = 0; n < num_exp_alloc; ++n) FREE(exp_segs[n].unique_block_indices);
    FREE(exp_segs); exp_segs = NULL;
    num_exp_alloc = 0;

    diag->code = PULSEQLIB_OK;
    return num_unique;

fail:
    if (max_energy) FREE(max_energy);
    if (uniq_segs) {
        for (i = 0; i < num_unique; ++i)
            if (uniq_segs[i].unique_block_indices) FREE(uniq_segs[i].unique_block_indices);
        FREE(uniq_segs);
    }
    if (exp_segs) {
        for (n = 0; n < num_exp_alloc; ++n)
            if (exp_segs[n].unique_block_indices) FREE(exp_segs[n].unique_block_indices);
        FREE(exp_segs);
    }
    if (raw_segs) {
        for (n = 0; n < num_raw_alloc; ++n)
            if (raw_segs[n].unique_block_indices) FREE(raw_segs[n].unique_block_indices);
        FREE(raw_segs);
    }
    return 0;
}

/* ================================================================== */
/*  Descriptor free functions (public)                                 */
/* ================================================================== */

void pulseqlib_sequence_descriptor_free(pulseqlib_sequence_descriptor* d)
{
    int i;
    if (!d) return;

    if (d->block_definitions) { FREE(d->block_definitions); d->block_definitions = NULL; }
    d->num_unique_blocks = 0;
    if (d->block_table) { FREE(d->block_table); d->block_table = NULL; }
    d->num_blocks = 0;

    if (d->rf_definitions) { FREE(d->rf_definitions); d->rf_definitions = NULL; }
    d->num_unique_rfs = 0;
    if (d->rf_table) { FREE(d->rf_table); d->rf_table = NULL; }
    d->rf_table_size = 0;

    if (d->grad_definitions) { FREE(d->grad_definitions); d->grad_definitions = NULL; }
    d->num_unique_grads = 0;
    if (d->grad_table) { FREE(d->grad_table); d->grad_table = NULL; }
    d->grad_table_size = 0;

    if (d->adc_definitions) { FREE(d->adc_definitions); d->adc_definitions = NULL; }
    d->num_unique_adcs = 0;
    if (d->adc_table) { FREE(d->adc_table); d->adc_table = NULL; }
    d->adc_table_size = 0;

    if (d->rotation_matrices) { FREE(d->rotation_matrices); d->rotation_matrices = NULL; }
    d->num_rotations = 0;
    if (d->trigger_events) { FREE(d->trigger_events); d->trigger_events = NULL; }
    d->num_triggers = 0;

    if (d->shapes) {
        for (i = 0; i < d->num_shapes; ++i)
            if (d->shapes[i].samples) FREE(d->shapes[i].samples);
        FREE(d->shapes);
        d->shapes = NULL;
    }
    d->num_shapes = 0;

    d->num_prep_blocks    = 0;
    d->num_cooldown_blocks = 0;

    if (d->segment_definitions) {
        for (i = 0; i < d->num_unique_segments; ++i) {
            if (d->segment_definitions[i].unique_block_indices) FREE(d->segment_definitions[i].unique_block_indices);
            if (d->segment_definitions[i].has_trigger)          FREE(d->segment_definitions[i].has_trigger);
            if (d->segment_definitions[i].has_rotation)         FREE(d->segment_definitions[i].has_rotation);
            if (d->segment_definitions[i].norot_flag)           FREE(d->segment_definitions[i].norot_flag);
            if (d->segment_definitions[i].nopos_flag)           FREE(d->segment_definitions[i].nopos_flag);
        }
        FREE(d->segment_definitions);
        d->segment_definitions = NULL;
    }
    d->num_unique_segments = 0;

    pulseqlib_segment_table_result_free(&d->segment_table);
}

void pulseqlib_sequence_descriptor_collection_free(
    pulseqlib_sequence_descriptor_collection* c)
{
    int i;
    if (!c) return;
    if (c->descriptors) {
        for (i = 0; i < c->num_subsequences; ++i)
            pulseqlib_sequence_descriptor_free(&c->descriptors[i]);
        FREE(c->descriptors);
    }
    if (c->subsequence_info) FREE(c->subsequence_info);
    c->num_subsequences      = 0;
    c->descriptors           = NULL;
    c->subsequence_info      = NULL;
    c->total_unique_segments = 0;
    c->total_unique_adcs     = 0;
    c->total_blocks          = 0;
    c->total_duration_us     = 0.0f;
}

void pulseqlib_segment_table_result_free(pulseqlib_segment_table_result* r)
{
    if (!r) return;
    if (r->prep_segment_table)     FREE(r->prep_segment_table);
    if (r->main_segment_table)     FREE(r->main_segment_table);
    if (r->cooldown_segment_table) FREE(r->cooldown_segment_table);
    r->prep_segment_table     = NULL;
    r->main_segment_table     = NULL;
    r->cooldown_segment_table = NULL;
    r->num_prep_segments      = 0;
    r->num_main_segments      = 0;
    r->num_cooldown_segments  = 0;
    r->num_unique_segments    = 0;
}

/* ================================================================== */
/*  get_collection_descriptors                                         */
/* ================================================================== */

int pulseqlib__get_collection_descriptors(
    pulseqlib_sequence_descriptor_collection* coll,
    pulseqlib_diagnostic* diag,
    const pulseqlib__seq_file_collection* raw)
{
    int i, j, result;
    int adc_off = 0, seg_off = 0, blk_off = 0;
    pulseqlib_diagnostic local_diag;

    if (!diag) { pulseqlib_diagnostic_init(&local_diag); diag = &local_diag; }

    if (!raw || !coll) { diag->code = PULSEQLIB_ERR_NULL_POINTER; return 0; }
    if (raw->num_sequences == 0) { diag->code = PULSEQLIB_ERR_COLLECTION_EMPTY; return 0; }

    coll->descriptors = (pulseqlib_sequence_descriptor*)ALLOC(
        raw->num_sequences * sizeof(pulseqlib_sequence_descriptor));
    coll->subsequence_info = (pulseqlib_subsequence_info*)ALLOC(
        raw->num_sequences * sizeof(pulseqlib_subsequence_info));
    if (!coll->descriptors || !coll->subsequence_info) {
        if (coll->descriptors)     FREE(coll->descriptors);
        if (coll->subsequence_info) FREE(coll->subsequence_info);
        coll->descriptors = NULL;
        coll->subsequence_info = NULL;
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        return 0;
    }

    coll->num_subsequences    = raw->num_sequences;
    coll->total_duration_us   = 0.0f;
    coll->total_unique_segments = 0;
    coll->total_unique_adcs   = 0;
    coll->total_blocks        = 0;

    for (i = 0; i < raw->num_sequences; ++i) {
        pulseqlib_sequence_descriptor desc = PULSEQLIB_SEQUENCE_DESCRIPTOR_INIT;

        coll->subsequence_info[i].sequence_index     = i;
        coll->subsequence_info[i].adc_id_offset      = adc_off;
        coll->subsequence_info[i].segment_id_offset  = seg_off;
        coll->subsequence_info[i].block_index_offset = blk_off;

        result = pulseqlib__get_unique_blocks(&desc, &raw->sequences[i]);
        if (PULSEQLIB_FAILED(result)) { diag->code = result; goto fail; }

        result = pulseqlib__find_tr_in_sequence(&desc, diag);
        if (PULSEQLIB_FAILED(diag->code)) goto fail;

        result = pulseqlib__find_segments_in_tr(&desc, diag, &raw->sequences[i]);
        if (PULSEQLIB_FAILED(diag->code)) goto fail;

        /* apply offsets */
        if (seg_off > 0) {
            for (j = 0; j < desc.segment_table.num_prep_segments; ++j)
                desc.segment_table.prep_segment_table[j] += seg_off;
            for (j = 0; j < desc.segment_table.num_main_segments; ++j)
                desc.segment_table.main_segment_table[j] += seg_off;
            for (j = 0; j < desc.segment_table.num_cooldown_segments; ++j)
                desc.segment_table.cooldown_segment_table[j] += seg_off;
        }
        if (adc_off > 0) {
            for (j = 0; j < desc.adc_table_size; ++j)
                desc.adc_table[j].id += adc_off;
            for (j = 0; j < desc.num_unique_adcs; ++j)
                desc.adc_definitions[j].id += adc_off;
        }

        adc_off += desc.num_unique_adcs;
        seg_off += desc.num_unique_segments;
        blk_off += desc.num_blocks;

        coll->total_duration_us += desc.tr_descriptor.tr_duration_us *
                                   desc.tr_descriptor.num_trs;

        coll->descriptors[i] = desc;
    }

    coll->total_unique_segments = seg_off;
    coll->total_unique_adcs     = adc_off;
    coll->total_blocks          = blk_off;
    diag->code = PULSEQLIB_OK;
    return raw->num_sequences;

fail:
    for (j = 0; j < i; ++j)
        pulseqlib_sequence_descriptor_free(&coll->descriptors[j]);
    FREE(coll->descriptors);
    FREE(coll->subsequence_info);
    coll->descriptors      = NULL;
    coll->subsequence_info = NULL;
    coll->num_subsequences = 0;
    return 0;
}

/* ================================================================== */
/*  pulseqlib_load (public entry point)                                */
/* ================================================================== */

int pulseqlib_load(
    pulseqlib_sequence_descriptor_collection* collection,
    pulseqlib_diagnostic* diag,
    const char* file_path,
    const pulseqlib_opts* opts)
{
    pulseqlib__seq_file_collection raw_coll;
    int rc;

    raw_coll.num_sequences = 0;
    raw_coll.sequences     = NULL;
    raw_coll.base_path     = NULL;

    if (!file_path || !opts || !collection || !diag)
        return PULSEQLIB_ERR_NULL_POINTER;

    pulseqlib_diagnostic_init(diag);

    rc = pulseqlib__read_seq_collection(&raw_coll, file_path, opts);
    if (PULSEQLIB_FAILED(rc)) { diag->code = rc; goto fail; }

    rc = pulseqlib__get_collection_descriptors(collection, diag, &raw_coll);
    if (PULSEQLIB_FAILED(diag->code)) { rc = diag->code; goto fail; }

    pulseqlib__seq_file_collection_free(&raw_coll);
    return PULSEQLIB_OK;

fail:
    pulseqlib__seq_file_collection_free(&raw_coll);
    return rc;
}
