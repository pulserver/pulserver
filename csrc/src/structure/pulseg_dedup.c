/* pulseg_dedup.c -- event deduplication, statistics, and unique-block extraction
 *
 * This file contains:
 *   - Hash-based integer-row deduplication
 *   - RF, gradient, ADC library deduplication
 *   - Gradient shot-index computation
 *   - Gradient and RF statistics (waveform normalisation, slew rate, etc.)
 *   - Auxiliary library copying (rotations, triggers, RF shims, shapes)
 *   - Raster-time divisibility checks
 *   - get_unique_blocks  --  top-level entry for dedup pipeline
 *
 * Split from pulseg_core.c for modularity.
 * ANSI C89.
 */

#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "pulseg_internal.h"
#include "pulseg.h"
#include "external_kiss_fft.h"

/* ================================================================== */
/*  File-scope constants                                              */
/* ================================================================== */
#define RF_DEF_COLS 4
#define RF_PARAMS_COLS 3
#define GRAD_DEF_COLS 6
#define ADC_DEF_COLS 3
#define ADC_PARAMS_COLS 2
#define BLOCK_DEF_COLS 5

/* ================================================================== */
/*  Tiny helpers                                                      */
/* ================================================================== */

static int array_equal(const int *a, const int *b, int len)
{
    int i;
    for (i = 0; i < len; ++i)
        if (a[i] != b[i])
            return 0;
    return 1;
}

/* ================================================================== */
/*  Hash-based integer-row deduplication                              */
/* ================================================================== */

typedef struct
{
    size_t hash;
    int row_index;
    int label;
    char used;
} hash_entry;

static size_t hash_row(const int *row, int num_cols)
{
    size_t h = 2166136261UL;
    int i;
    for (i = 0; i < num_cols; ++i)
    {
        h ^= (size_t)row[i];
        h *= 16777619UL;
    }
    return h;
}

int pulseg__deduplicate_int_rows(
    int *unique_defs, int *event_table,
    const int *int_rows, int num_rows, int num_cols)
{
    size_t table_size;
    hash_entry *table = NULL;
    int num_unique = 0;
    int r;
    size_t h, idx;

    if (num_rows <= 0)
        return 0;

    table_size = pulseg__next_pow2((size_t)(num_rows * 2));
    table = (hash_entry *)PULSEG_ALLOC(table_size * sizeof(hash_entry));
    if (!table)
        return 0;
    memset(table, 0, table_size * sizeof(hash_entry));

    for (r = 0; r < num_rows; ++r)
    {
        h = hash_row(&int_rows[r * num_cols], num_cols);
        idx = h & (table_size - 1);

        while (table[idx].used)
        {
            if (table[idx].hash == h &&
                array_equal(&int_rows[r * num_cols],
                            &int_rows[table[idx].row_index * num_cols],
                            num_cols))
            {
                event_table[r] = table[idx].label;
                break;
            }
            idx = (idx + 1) & (table_size - 1);
        }

        if (!table[idx].used)
        {
            table[idx].hash = h;
            table[idx].row_index = r;
            table[idx].label = num_unique;
            table[idx].used = 1;
            unique_defs[num_unique] = r;
            event_table[r] = num_unique;
            num_unique++;
        }
    }

    PULSEG_FREE(table);
    return num_unique;
}

/* ================================================================== */
/*  RF dedup helpers                                                  */
/* ================================================================== */

static void build_rf_def_row(const pulseg_pulseq_file *seq, int *row, float *params, int rf_idx)
{
    float gamma = seq->opts.gamma_hz_per_t;
    float b0 = seq->opts.b0_t;
    float *rf = seq->rf_library[rf_idx];
    float ppm_to_hz = 1e-6f * gamma * b0;

    row[0] = (int)rf[1]; /* mag shape id */
    row[1] = (int)rf[2]; /* phase shape id */
    row[2] = (int)rf[3]; /* time shape id */
    row[3] = (int)rf[5]; /* delay */

    params[0] = rf[0];                     /* amplitude */
    params[1] = rf[8] + ppm_to_hz * rf[6]; /* freq offset + ppm * freqPPM */
    params[2] = rf[9] + ppm_to_hz * rf[7]; /* phase offset + ppm * phasePPM */
}

static int deduplicate_rf_library(const pulseg_pulseq_file *seq, pulseg_rf_definition *rf_defs, pulseg_rf_table_element *rf_table)
{
    int (*int_rows)[RF_DEF_COLS] = NULL;
    float (*params)[RF_PARAMS_COLS] = NULL;
    int *unique_defs = NULL;
    int *event_table = NULL;
    int num_unique, num_rows, i;

    num_rows = seq->rf_library_size;
    if (num_rows <= 0)
        return 0;

    int_rows = PULSEG_ALLOC(num_rows * sizeof(*int_rows));
    params = PULSEG_ALLOC(num_rows * sizeof(*params));
    unique_defs = (int *)PULSEG_ALLOC(num_rows * sizeof(int));
    event_table = (int *)PULSEG_ALLOC(num_rows * sizeof(int));
    if (!int_rows || !params || !unique_defs || !event_table)
    {
        if (int_rows)
            PULSEG_FREE(int_rows);
        if (params)
            PULSEG_FREE(params);
        if (unique_defs)
            PULSEG_FREE(unique_defs);
        if (event_table)
            PULSEG_FREE(event_table);
        return 0;
    }

    for (i = 0; i < num_rows; ++i)
        build_rf_def_row(seq, int_rows[i], params[i], i);

    num_unique = pulseg__deduplicate_int_rows(unique_defs, event_table, (const int *)int_rows, num_rows, RF_DEF_COLS);

    for (i = 0; i < num_unique; ++i)
    {
        int time_id, nz, j;
        rf_defs[i].id = unique_defs[i];
        rf_defs[i].mag_shape_id = int_rows[unique_defs[i]][0];
        rf_defs[i].phase_shape_id = int_rows[unique_defs[i]][1];
        rf_defs[i].time_shape_id = int_rows[unique_defs[i]][2];
        rf_defs[i].delay = int_rows[unique_defs[i]][3];
        rf_defs[i].num_channels = 1;

        /* detect multichannel RF from tiled time shape */
        time_id = rf_defs[i].time_shape_id;
        if (time_id > 0 && time_id <= seq->shapes_library_size)
        {
            pulseg_shape_arbitrary decomp;
            decomp.num_samples = 0;
            decomp.num_uncompressed_samples = 0;
            decomp.samples = NULL;
            if (pulseg_pulseq_decompress_shape(&decomp,
                                            &seq->shapes_library[time_id - 1], 1.0f))
            {
                nz = 0;
                for (j = 0; j < decomp.num_uncompressed_samples; ++j)
                    if (decomp.samples[j] == 0.0f)
                        ++nz;
                if (nz > 1)
                    rf_defs[i].num_channels = nz;
                PULSEG_FREE(decomp.samples);
            }
        }
    }
    for (i = 0; i < num_rows; ++i)
    {
        rf_table[i].id = event_table[i];
        rf_table[i].amplitude = params[i][0];
        rf_table[i].freq_offset = params[i][1];
        rf_table[i].phase_offset = params[i][2];
        rf_table[i].rf_use = (seq->rf_use_tags)
                                 ? seq->rf_use_tags[i]
                                 : PULSEG_RF_USE_UNKNOWN;
    }

    PULSEG_FREE(int_rows);
    PULSEG_FREE(params);
    PULSEG_FREE(unique_defs);
    PULSEG_FREE(event_table);
    return num_unique;
}

/* ================================================================== */
/*  Grad dedup helpers                                                */
/* ================================================================== */

static void build_grad_def_row(const pulseg_pulseq_file *seq, int *row, float *param, int grad_idx)
{
    float *grad = seq->grad_library[grad_idx];
    int grad_type = (int)grad[0];
    int wave_id;

    row[0] = grad_type;
    if (grad_type == 0)
    {
        row[1] = (int)grad[2]; /* rise */
        row[2] = (int)grad[3]; /* flat */
        row[3] = (int)grad[4]; /* fall */
        row[4] = 0;
        row[5] = (int)grad[5]; /* delay (trap: 6th column = grad[5]) */
    }
    else
    {
        row[1] = 0;
        row[2] = 0;
        wave_id = (int)grad[4];
        if (wave_id > 0 && seq->is_shapes_library_parsed &&
            wave_id <= seq->shapes_library_size)
        {
            row[3] = seq->shapes_library[wave_id - 1].num_uncompressed_samples;
        }
        else
        {
            row[3] = 0;
        }
        row[4] = (int)grad[5]; /* time shape id */
        row[5] = (int)grad[6]; /* delay (arb: 7th column = grad[6]) */
    }
    *param = grad[1]; /* amplitude */
}

static int deduplicate_grad_library(const pulseg_pulseq_file *seq, pulseg_grad_definition *grad_defs, pulseg_grad_table_element *grad_table)
{
    int (*int_rows)[GRAD_DEF_COLS] = NULL;
    float *params = NULL;
    int *unique_defs = NULL;
    int *event_table = NULL;
    int num_unique, num_rows, i;

    num_rows = seq->grad_library_size;
    if (num_rows <= 0)
        return 0;

    int_rows = PULSEG_ALLOC(num_rows * sizeof(*int_rows));
    params = (float *)PULSEG_ALLOC(num_rows * sizeof(float));
    unique_defs = (int *)PULSEG_ALLOC(num_rows * sizeof(int));
    event_table = (int *)PULSEG_ALLOC(num_rows * sizeof(int));
    if (!int_rows || !params || !unique_defs || !event_table)
    {
        if (int_rows)
            PULSEG_FREE(int_rows);
        if (params)
            PULSEG_FREE(params);
        if (unique_defs)
            PULSEG_FREE(unique_defs);
        if (event_table)
            PULSEG_FREE(event_table);
        return 0;
    }

    for (i = 0; i < num_rows; ++i)
        build_grad_def_row(seq, int_rows[i], &params[i], i);

    num_unique = pulseg__deduplicate_int_rows(unique_defs, event_table, (const int *)int_rows, num_rows, GRAD_DEF_COLS);

    for (i = 0; i < num_unique; ++i)
    {
        grad_defs[i].id = unique_defs[i];
        grad_defs[i].type = int_rows[unique_defs[i]][0];
        grad_defs[i].rise_time_or_unused = int_rows[unique_defs[i]][1];
        grad_defs[i].flat_time_or_unused = int_rows[unique_defs[i]][2];
        grad_defs[i].fall_time_or_num_uncompressed_samples = int_rows[unique_defs[i]][3];
        grad_defs[i].unused_or_time_shape_id = int_rows[unique_defs[i]][4];
        grad_defs[i].delay = int_rows[unique_defs[i]][5];
    }
    for (i = 0; i < num_rows; ++i)
    {
        grad_table[i].id = event_table[i];
        grad_table[i].amplitude = params[i];
    }

    PULSEG_FREE(int_rows);
    PULSEG_FREE(params);
    PULSEG_FREE(unique_defs);
    PULSEG_FREE(event_table);
    return num_unique;
}

/* ================================================================== */
/*  ADC dedup helpers                                                 */
/* ================================================================== */

static void build_adc_def_row(const pulseg_pulseq_file *seq, int *row, float *params, int adc_idx)
{
    float gamma = seq->opts.gamma_hz_per_t;
    float b0 = seq->opts.b0_t;
    float *adc = seq->adc_library[adc_idx];
    float ppm_to_hz = 1e-6f * gamma * b0;

    row[0] = (int)adc[0];                    /* num_samples */
    row[1] = (int)adc[1];                    /* dwell_time_ns */
    row[2] = (int)adc[2];                    /* delay */
    params[0] = adc[5] + ppm_to_hz * adc[3]; /* freq offset */
    params[1] = adc[6] + ppm_to_hz * adc[4]; /* phase offset */
}

static int deduplicate_adc_library(const pulseg_pulseq_file *seq, pulseg_adc_definition *adc_defs, pulseg_adc_table_element *adc_table)
{
    int (*int_rows)[ADC_DEF_COLS] = NULL;
    float (*params)[ADC_PARAMS_COLS] = NULL;
    int *unique_defs = NULL;
    int *event_table = NULL;
    int num_unique, num_rows, i;

    num_rows = seq->adc_library_size;
    if (num_rows <= 0)
        return 0;

    int_rows = PULSEG_ALLOC(num_rows * sizeof(*int_rows));
    params = PULSEG_ALLOC(num_rows * sizeof(*params));
    unique_defs = (int *)PULSEG_ALLOC(num_rows * sizeof(int));
    event_table = (int *)PULSEG_ALLOC(num_rows * sizeof(int));
    if (!int_rows || !params || !unique_defs || !event_table)
    {
        if (int_rows)
            PULSEG_FREE(int_rows);
        if (params)
            PULSEG_FREE(params);
        if (unique_defs)
            PULSEG_FREE(unique_defs);
        if (event_table)
            PULSEG_FREE(event_table);
        return 0;
    }

    for (i = 0; i < num_rows; ++i)
        build_adc_def_row(seq, int_rows[i], params[i], i);

    num_unique = pulseg__deduplicate_int_rows(unique_defs, event_table, (const int *)int_rows, num_rows, ADC_DEF_COLS);

    for (i = 0; i < num_unique; ++i)
    {
        adc_defs[i].id = unique_defs[i];
        adc_defs[i].num_samples = int_rows[unique_defs[i]][0];
        adc_defs[i].dwell_time = int_rows[unique_defs[i]][1];
        adc_defs[i].delay = int_rows[unique_defs[i]][2];
    }
    for (i = 0; i < num_rows; ++i)
    {
        adc_table[i].id = event_table[i];
        adc_table[i].freq_offset = params[i][0];
        adc_table[i].phase_offset = params[i][1];
    }

    PULSEG_FREE(int_rows);
    PULSEG_FREE(params);
    PULSEG_FREE(unique_defs);
    PULSEG_FREE(event_table);
    return num_unique;
}

/* ================================================================== */
/*  Gradient shot indices                                             */
/* ================================================================== */

static int compute_grad_shot_indices(
    const pulseg_pulseq_file *seq,
    pulseg_grad_definition *grad_defs, pulseg_grad_table_element *grad_table,
    int num_unique_grads)
{
    int num_rows = seq->grad_library_size;
    int def_idx, i, j;
    int shape_id, found, shot_count;

    if (num_rows <= 0 || num_unique_grads <= 0)
        return PULSEG_SUCCESS;

    for (def_idx = 0; def_idx < num_unique_grads; ++def_idx)
    {
        int grad_type = grad_defs[def_idx].type;

        for (j = 0; j < PULSEG_MAX_GRAD_SHOTS; ++j)
            grad_defs[def_idx].shot_shape_ids[j] = 0;

        if (grad_type == 0)
        {
            grad_defs[def_idx].num_shots = 1;
            for (i = 0; i < num_rows; ++i)
                if (grad_table[i].id == def_idx)
                    grad_table[i].shot_index = 0;
            continue;
        }

        shot_count = 0;
        for (i = 0; i < num_rows; ++i)
        {
            if (grad_table[i].id != def_idx)
                continue;
            shape_id = (int)seq->grad_library[i][4];

            found = 0;
            for (j = 0; j < shot_count; ++j)
            {
                if (grad_defs[def_idx].shot_shape_ids[j] == shape_id)
                {
                    found = 1;
                    grad_table[i].shot_index = j;
                    break;
                }
            }
            if (!found)
            {
                if (shot_count >= PULSEG_MAX_GRAD_SHOTS)
                    return PULSEG_ERR_TOO_MANY_GRAD_SHOTS;
                grad_table[i].shot_index = shot_count;
                grad_defs[def_idx].shot_shape_ids[shot_count] = shape_id;
                shot_count++;
            }
        }
        grad_defs[def_idx].num_shots = shot_count > 0 ? shot_count : 1;
    }
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Waveform normalisation                                            */
/* ================================================================== */

static float normalize_waveform(float *waveform, int n)
{
    float max_abs;
    int i;

    max_abs = pulseg__get_max_abs_real(waveform, n);
    if (max_abs > 1e-9f)
    {
        for (i = 0; i < n; ++i)
            waveform[i] /= max_abs;
    }
    return max_abs;
}

/* ================================================================== */
/*  Trapezoid statistics                                               */
/* ================================================================== */

static void compute_trapezoid_stats(
    float *slew, float *energy, float *first_val, float *last_val,
    float rise_us, float flat_us, float fall_us)
{
    float rise_s = rise_us * 1e-6f;
    float flat_s = flat_us * 1e-6f;
    float fall_s = fall_us * 1e-6f;
    float sr, sf;

    *first_val = 0.0f;
    *last_val = 0.0f;

    sr = (rise_s > 0.0f) ? (1.0f / rise_s) : 0.0f;
    sf = (fall_s > 0.0f) ? (1.0f / fall_s) : 0.0f;
    *slew = (sr > sf) ? sr : sf;

    *energy = rise_s / 3.0f + flat_s + fall_s / 3.0f;
}

/* ================================================================== */
/*  Gradient statistics                                               */
/* ================================================================== */

static int compute_grad_stats(
    const pulseg_pulseq_file *seq,
    pulseg_grad_definition *grad_defs, int num_unique,
    const pulseg_grad_table_element *grad_table, int grad_table_size)
{
    int def_idx, i, shot_idx, num_samples, has_time;
    int grad_type, time_id, shape_id;
    float rise_us, flat_us, fall_us, abs_amp;
    float grad_raster_us;
    pulseg_shape_arbitrary decomp_wave, decomp_time;
    float *waveform = NULL;
    float *sq_wave = NULL;
    float *time_us = NULL;
    pulseg_grad_definition *gd;

    if (!seq || !grad_defs || num_unique <= 0)
        return PULSEG_SUCCESS;

    if (seq->reserved_definitions_library.gradient_raster_time > 0.0f)
        grad_raster_us = seq->reserved_definitions_library.gradient_raster_time;
    else
        grad_raster_us = seq->opts.grad_raster_us;

    decomp_wave.num_samples = 0;
    decomp_wave.num_uncompressed_samples = 0;
    decomp_wave.samples = NULL;
    decomp_time.num_samples = 0;
    decomp_time.num_uncompressed_samples = 0;
    decomp_time.samples = NULL;

    for (def_idx = 0; def_idx < num_unique; ++def_idx)
    {
        gd = &grad_defs[def_idx];
        grad_type = gd->type;

        for (i = 0; i < PULSEG_MAX_GRAD_SHOTS; ++i)
        {
            gd->max_amplitude[i] = 0.0f;
            gd->min_amplitude[i] = 1e30f;
            gd->slew_rate[i] = 0.0f;
            gd->energy[i] = 0.0f;
            gd->first_value[i] = 0.0f;
            gd->last_value[i] = 0.0f;
        }

        /* max amplitude and min amplitude per shot from table. */
        if (grad_table && grad_table_size > 0)
        {
            for (i = 0; i < grad_table_size; ++i)
            {
                if (grad_table[i].id == def_idx)
                {
                    shot_idx = grad_table[i].shot_index;
                    if (shot_idx >= 0 && shot_idx < PULSEG_MAX_GRAD_SHOTS)
                    {
                        abs_amp = grad_table[i].amplitude;
                        if (abs_amp < 0.0f)
                            abs_amp = -abs_amp;
                        if (abs_amp > gd->max_amplitude[shot_idx])
                            gd->max_amplitude[shot_idx] = abs_amp;
                        if (abs_amp < gd->min_amplitude[shot_idx])
                        {
                            gd->min_amplitude[shot_idx] = abs_amp;
                        }
                    }
                }
            }
        }

        /* clamp sentinel: if no table entry touched a shot, min stays 0 */
        for (i = 0; i < PULSEG_MAX_GRAD_SHOTS; ++i)
        {
            if (gd->min_amplitude[i] > 1e29f)
            {
                gd->min_amplitude[i] = 0.0f;
            }
        }

        if (grad_type == 0)
        {
            rise_us = (float)gd->rise_time_or_unused;
            flat_us = (float)gd->flat_time_or_unused;
            fall_us = (float)gd->fall_time_or_num_uncompressed_samples;
            compute_trapezoid_stats(&gd->slew_rate[0], &gd->energy[0], &gd->first_value[0], &gd->last_value[0], rise_us, flat_us, fall_us);
        }
        else
        {
            time_id = gd->unused_or_time_shape_id;
            time_us = NULL;
            has_time = 0;
            if (time_id > 0 && time_id <= seq->shapes_library_size)
            {
                if (!pulseg_pulseq_decompress_shape(&decomp_time,
                                                 &seq->shapes_library[time_id - 1], grad_raster_us))
                    goto fail;
                time_us = (float *)PULSEG_ALLOC(decomp_time.num_uncompressed_samples * sizeof(float));
                if (!time_us)
                    goto fail;
                for (i = 0; i < decomp_time.num_uncompressed_samples; ++i)
                    time_us[i] = decomp_time.samples[i];
                has_time = 1;
                PULSEG_FREE(decomp_time.samples);
                decomp_time.samples = NULL;
            }

            for (shot_idx = 0; shot_idx < gd->num_shots; ++shot_idx)
            {
                shape_id = gd->shot_shape_ids[shot_idx];
                if (shape_id <= 0 || shape_id > seq->shapes_library_size)
                    continue;

                if (!pulseg_pulseq_decompress_shape(&decomp_wave,
                                                 &seq->shapes_library[shape_id - 1], 1.0f))
                    goto fail;
                num_samples = decomp_wave.num_uncompressed_samples;

                waveform = (float *)PULSEG_ALLOC(num_samples * sizeof(float));
                sq_wave = (float *)PULSEG_ALLOC(num_samples * sizeof(float));
                if (!waveform || !sq_wave)
                    goto fail;

                for (i = 0; i < num_samples; ++i)
                    waveform[i] = decomp_wave.samples[i];
                normalize_waveform(waveform, num_samples);

                for (i = 0; i < num_samples; ++i)
                    sq_wave[i] = waveform[i] * waveform[i];

                gd->first_value[shot_idx] = waveform[0];
                gd->last_value[shot_idx] = waveform[num_samples - 1];

                if (has_time && time_us)
                {
                    gd->slew_rate[shot_idx] = pulseg__max_slew_real_nonuniform(waveform, time_us, num_samples);
                    gd->energy[shot_idx] = pulseg__trapz_real_nonuniform(sq_wave, time_us, num_samples);
                }
                else
                {
                    gd->slew_rate[shot_idx] = pulseg__max_slew_real_uniform(waveform, num_samples, grad_raster_us);
                    gd->energy[shot_idx] = pulseg__trapz_real_uniform(sq_wave, num_samples, grad_raster_us);
                }
                gd->slew_rate[shot_idx] *= 1e6f;
                gd->energy[shot_idx] *= 1e-6f;

                PULSEG_FREE(waveform);
                waveform = NULL;
                PULSEG_FREE(sq_wave);
                sq_wave = NULL;
                PULSEG_FREE(decomp_wave.samples);
                decomp_wave.samples = NULL;
            }

            if (time_us)
            {
                PULSEG_FREE(time_us);
                time_us = NULL;
            }
        }
    }
    return PULSEG_SUCCESS;

fail:
    if (waveform)
        PULSEG_FREE(waveform);
    if (sq_wave)
        PULSEG_FREE(sq_wave);
    if (time_us)
        PULSEG_FREE(time_us);
    if (decomp_wave.samples)
        PULSEG_FREE(decomp_wave.samples);
    if (decomp_time.samples)
        PULSEG_FREE(decomp_time.samples);
    return PULSEG_ERR_ALLOC_FAILED;
}

/* ================================================================== */
/*  RF statistics                                                     */
/* ================================================================== */

static float compute_rf_bandwidth_fft(const float *rf_re, const float *rf_im,
                                      kiss_fft_cfg cfg, int nn,
                                      float cutoff, float duration,
                                      const float *w,
                                      float *work_re, float *work_im,
                                      kiss_fft_cpx *fft_in, kiss_fft_cpx *fft_out)
{
    int i;
    float w1, w2, bw;
    float fallback_bw;

    fallback_bw = (duration > 0.0f) ? (3.12f / duration) : 0.0f;
    if (!rf_re || !rf_im || !cfg || nn <= 0)
        return fallback_bw;

    for (i = 0; i < nn; ++i)
    {
        work_re[i] = rf_re[i];
        work_im[i] = rf_im[i];
    }
    pulseg__fftshift_complex(work_re, work_im, nn);
    for (i = 0; i < nn; ++i)
    {
        fft_in[i].r = work_re[i];
        fft_in[i].i = work_im[i];
    }

    kiss_fft(cfg, fft_in, fft_out);

    for (i = 0; i < nn; ++i)
    {
        work_re[i] = fft_out[i].r;
        work_im[i] = fft_out[i].i;
    }
    pulseg__fftshift_complex(work_re, work_im, nn);

    w1 = pulseg__get_spectrum_flank(w, work_re, work_im, nn, cutoff, 0);
    w2 = pulseg__get_spectrum_flank(w, work_re, work_im, nn, cutoff, 1);
    bw = w2 - w1;
    return (bw > 0.0f) ? bw : fallback_bw;
}

static int compute_rf_stats(
    const pulseg_pulseq_file *seq,
    pulseg_rf_definition *rf_defs, int num_unique,
    const pulseg_rf_table_element *rf_table, int rf_table_size)
{
    int def_idx, i;
    pulseg_shape_arbitrary decomp_mag, decomp_phase, decomp_time;
    float *magnitude = NULL;
    float *phase = NULL;
    float *time_us = NULL;
    float *time_us_uniform = NULL;
    float *rf_re = NULL;
    float *rf_im = NULL;
    float *rf_re_uniform = NULL;
    float *rf_im_uniform = NULL;
    float *time_centered = NULL;
    int num_samples, num_uniform, num_real;
    int mag_id, phase_id, time_id;
    int has_phase, has_time;
    int first, last;
    float max_mag, duration, time_center, rf_raster_us;
    pulseg_rf_definition *rd;

    int nn;
    float dw = 10.0f;
    float cutoff = 0.5f;

    kiss_fft_cfg fft_cfg = NULL;
    float *tt = NULL;
    float *w = NULL;
    float *rfs_re = NULL;
    float *rfs_im = NULL;
    float *work_re = NULL;
    float *work_im = NULL;
    kiss_fft_cpx *fft_in = NULL;
    kiss_fft_cpx *fft_out = NULL;
    int fft_ready = 0;

    float rf_abs, sum_signed;
    float sum_sq;
    float *mag_view = NULL;
    float *phase_view = NULL;
    int fail_rc = PULSEG_ERR_ALLOC_FAILED;

    if (!seq || !rf_defs || num_unique <= 0)
        return PULSEG_SUCCESS;

    if (seq->reserved_definitions_library.radiofrequency_raster_time > 0.0f)
        rf_raster_us = seq->reserved_definitions_library.radiofrequency_raster_time;
    else
        rf_raster_us = seq->opts.rf_raster_us;

    nn = (int)(1.0f / (dw * rf_raster_us * 1e-6f));
    nn = kiss_fft_next_fast_size(nn);
    if (nn < 2)
        nn = 2;

    tt = (float *)PULSEG_ALLOC(nn * sizeof(float));
    w = (float *)PULSEG_ALLOC(nn * sizeof(float));
    rfs_re = (float *)PULSEG_ALLOC(nn * sizeof(float));
    rfs_im = (float *)PULSEG_ALLOC(nn * sizeof(float));
    work_re = (float *)PULSEG_ALLOC(nn * sizeof(float));
    work_im = (float *)PULSEG_ALLOC(nn * sizeof(float));
    fft_in = (kiss_fft_cpx *)KISS_FFT_MALLOC(nn * sizeof(kiss_fft_cpx));
    fft_out = (kiss_fft_cpx *)KISS_FFT_MALLOC(nn * sizeof(kiss_fft_cpx));
    fft_cfg = kiss_fft_alloc(nn, 0, NULL, NULL);
    if (tt && w && rfs_re && rfs_im && work_re && work_im && fft_in && fft_out && fft_cfg)
    {
        for (i = 0; i < nn; ++i)
        {
            tt[i] = (float)(i - nn / 2) * rf_raster_us;
            w[i] = (float)(i - nn / 2) * dw;
        }
        fft_ready = 1;
    }
    if (!fft_ready)
    {
        goto fail;
    }

    decomp_mag.num_samples = 0;
    decomp_mag.num_uncompressed_samples = 0;
    decomp_mag.samples = NULL;
    decomp_phase.num_samples = 0;
    decomp_phase.num_uncompressed_samples = 0;
    decomp_phase.samples = NULL;
    decomp_time.num_samples = 0;
    decomp_time.num_uncompressed_samples = 0;
    decomp_time.samples = NULL;

    for (def_idx = 0; def_idx < num_unique; ++def_idx)
    {
        rd = &rf_defs[def_idx];
        first = -1;
        last = -1;

        rd->stats.num_samples = 0;
        rd->stats.flip_angle_rad = 0.0f;
        rd->stats.base_amplitude_hz = 0.0f;
        rd->stats.area = 0.0f;
        rd->stats.vendor_stat[0] = 0.0f;
        rd->stats.vendor_stat[1] = 0.0f;
        rd->stats.vendor_stat[2] = 0.0f;
        rd->stats.vendor_stat[3] = 0.0f;
        rd->stats.duration_us = 0.0f;
        rd->stats.isodelay_us = 0;
        rd->stats.bandwidth_hz = 0.0f;
        rd->stats.num_bands = 1;
        rd->stats.band_bandwidth_hz = 0.0f;
        rd->stats.total_b1sq_power = 0.0f;
        rd->stats.vendor = seq->opts.vendor;
        {
            int bi;
            for (bi = 0; bi < PULSEG_MAX_BANDS; ++bi)
                rd->stats.band_freq_offsets_hz[bi] = 0.0f;
        }

        /* max amplitude from table */
        if (rf_table && rf_table_size > 0)
        {
            for (i = 0; i < rf_table_size; ++i)
            {
                if (rf_table[i].id == def_idx)
                {
                    float amp = (float)fabs(rf_table[i].amplitude);
                    if (amp > rd->stats.base_amplitude_hz)
                        rd->stats.base_amplitude_hz = amp;
                }
            }
        }
        mag_id = rd->mag_shape_id;
        phase_id = rd->phase_shape_id;
        time_id = rd->time_shape_id;
        has_phase = 0;
        has_time = 0;
        magnitude = NULL;
        phase = NULL;
        time_us = NULL;
        rf_re = NULL;
        rf_im = NULL;
        time_centered = NULL;
        num_samples = 0;
        duration = 0.0f;

        /* decompress magnitude */
        if (!pulseg_pulseq_decompress_shape(&decomp_mag, &seq->shapes_library[mag_id - 1], 1.0f))
            goto fail;
        num_samples = decomp_mag.num_uncompressed_samples;
        magnitude = (float *)PULSEG_ALLOC(num_samples * sizeof(float));
        if (!magnitude)
        {
            PULSEG_FREE(decomp_mag.samples);
            goto fail;
        }
        for (i = 0; i < num_samples; ++i)
            magnitude[i] = decomp_mag.samples[i];
        PULSEG_FREE(decomp_mag.samples);
        decomp_mag.samples = NULL;

        /* decompress phase (optional) */
        if (phase_id > 0 && phase_id <= seq->shapes_library_size)
        {
            if (!pulseg_pulseq_decompress_shape(&decomp_phase, &seq->shapes_library[phase_id - 1], 1.0f))
                goto fail;
            phase = (float *)PULSEG_ALLOC(num_samples * sizeof(float));
            if (!phase)
            {
                PULSEG_FREE(decomp_phase.samples);
                goto fail;
            }
            for (i = 0; i < num_samples; ++i)
                phase[i] = decomp_phase.samples[i];
            has_phase = 1;
            PULSEG_FREE(decomp_phase.samples);
            decomp_phase.samples = NULL;
        }

        /* Combine multichannel RF into a single effective waveform for
         * stats by quadrature aggregation. RF shim phases are encoded
         * elsewhere; stats should reflect the effective B1 envelope,
         * not a coherent complex sum across transmit channels. */
        if (rd->num_channels > 1 && num_samples > 0)
        {
            int nch = rd->num_channels;
            int npts = num_samples / nch;
            float *new_mag;
            int ch, s;

            new_mag = (float *)PULSEG_ALLOC(npts * sizeof(float));
            if (!new_mag)
            {
                if (new_mag)
                    PULSEG_FREE(new_mag);
                goto fail;
            }
            for (s = 0; s < npts; ++s)
            {
                float rss = 0.0f;
                for (ch = 0; ch < nch; ++ch)
                {
                    float m = magnitude[ch * npts + s];
                    rss += m * m;
                }
                new_mag[s] = (float)sqrt(rss);
            }
            PULSEG_FREE(magnitude);
            magnitude = new_mag;
            if (phase)
                PULSEG_FREE(phase);
            phase = NULL;
            has_phase = 0;
            num_samples = npts;
        }
        rd->stats.num_samples = num_samples;

        /* detect real-valued RF */
        if (has_phase && phase)
        {
            num_real = 0;
            for (i = 0; i < num_samples; ++i)
            {
                if ((float)fabs(phase[i]) < 1e-6f ||
                    (float)fabs(phase[i] - (float)M_PI) < 1e-6f)
                    ++num_real;
            }
            if (num_real == num_samples)
            {
                for (i = 0; i < num_samples; ++i)
                    if ((float)fabs(phase[i] - (float)M_PI) < 1e-6f)
                        magnitude[i] *= -1.0f;
                PULSEG_FREE(phase);
                phase = NULL;
                has_phase = 0;
            }
        }

        /* decompress time (optional) */
        if (time_id > 0 && time_id <= seq->shapes_library_size)
        {
            if (!pulseg_pulseq_decompress_shape(&decomp_time, &seq->shapes_library[time_id - 1], rf_raster_us))
                goto fail;
            time_us = (float *)PULSEG_ALLOC(num_samples * sizeof(float));
            if (!time_us)
            {
                PULSEG_FREE(decomp_time.samples);
                goto fail;
            }
            for (i = 0; i < num_samples; ++i)
                time_us[i] = decomp_time.samples[i];
            has_time = 1;
            PULSEG_FREE(decomp_time.samples);
            decomp_time.samples = NULL;
        }
        if (!has_time)
        {
            time_us = (float *)PULSEG_ALLOC(num_samples * sizeof(float));
            if (!time_us)
                goto fail;
            /* Pulseq places uniform-raster samples at bin centres:
               t = ((1:N)-0.5)*dwell, i.e. (i+0.5)*raster in 0-based */
            for (i = 0; i < num_samples; ++i)
                time_us[i] = ((float)i + 0.5f) * rf_raster_us;
            has_time = 1;
        }

        duration = (has_time && num_samples > 0) ? time_us[num_samples - 1] : (num_samples * rf_raster_us);
        rd->stats.duration_us = duration;

        /* find peak indices for isodelay */
        max_mag = pulseg__get_max_abs_real(magnitude, num_samples);
        for (i = 0; i < num_samples; ++i)
        {
            if ((float)fabs(magnitude[i]) >= 0.99999f * max_mag)
            {
                if (first < 0)
                    first = i;
                last = i;
            }
        }
        if (first < 0)
        {
            first = 0;
            last = 0;
        }

        time_center = (has_time && time_us)
                          ? 0.5f * (time_us[first] + time_us[last])
                          : 0.5f * ((float)(first + last)) * rf_raster_us;
        rd->stats.isodelay_us = (int)(duration - time_center);

        /* normalise */
        if (max_mag > 1e-9f)
            for (i = 0; i < num_samples; ++i)
                magnitude[i] /= max_mag;

        /* build complex RF */
        rf_re = (float *)PULSEG_ALLOC(num_samples * sizeof(float));
        rf_im = (float *)PULSEG_ALLOC(num_samples * sizeof(float));
        if (!rf_re || !rf_im)
            goto fail;
        if (has_phase && phase)
        {
            for (i = 0; i < num_samples; ++i)
            {
                rf_re[i] = magnitude[i] * (float)cos(phase[i]);
                rf_im[i] = magnitude[i] * (float)sin(phase[i]);
            }
        }
        else
        {
            for (i = 0; i < num_samples; ++i)
            {
                rf_re[i] = magnitude[i];
                rf_im[i] = 0.0f;
            }
        }

        /* uniform grid */
        num_uniform = (int)(duration / rf_raster_us) + 1;
        if (num_uniform < 2)
            num_uniform = 2;

        time_us_uniform = (float *)PULSEG_ALLOC(num_uniform * sizeof(float));
        rf_re_uniform = (float *)PULSEG_ALLOC(num_uniform * sizeof(float));
        rf_im_uniform = (float *)PULSEG_ALLOC(num_uniform * sizeof(float));
        if (!time_us_uniform || !rf_re_uniform || !rf_im_uniform)
            goto fail;

        for (i = 0; i < num_uniform; ++i)
            time_us_uniform[i] = (float)i * rf_raster_us;

        pulseg__interp1_linear_complex(rf_re_uniform, rf_im_uniform,
                                          time_us_uniform, num_uniform,
                                          time_us, rf_re, rf_im, num_samples);

        /* Compute signed complex integral once — used for both area and flip_angle.
         * Trapezoidal rule on the NATIVE (un-interpolated) time grid. */
        {
            double dre = 0.0, dim = 0.0;
            if (has_time && time_us && num_samples >= 2)
            {
                for (i = 0; i < num_samples - 1; ++i)
                {
                    double dt = ((double)time_us[i + 1] - (double)time_us[i]) * 1e-6;
                    dre += 0.5 * dt * ((double)rf_re[i] + (double)rf_re[i + 1]);
                    dim += 0.5 * dt * ((double)rf_im[i] + (double)rf_im[i + 1]);
                }
            }
            else
            {
                /* fall back to uniform-grid integration */
                double dt = (double)rf_raster_us * 1e-6;
                for (i = 0; i < num_samples - 1; ++i)
                {
                    dre += 0.5 * dt * ((double)rf_re[i] + (double)rf_re[i + 1]);
                    dim += 0.5 * dt * ((double)rf_im[i] + (double)rf_im[i + 1]);
                }
            }
            /* area = signed real part = ∫h_norm dt [s] */
            sum_signed = (float)dre;

            /* flip angle = γ|∫B1 dt| [rad]; stored in flip_angle_rad */
            {
                double mag_d = sqrt(dre * dre + dim * dim);
                rd->stats.flip_angle_rad = (float)(2.0 * 3.14159265358979323846 * (double)rd->stats.base_amplitude_hz * mag_d); /* radians */
            }
        }
        /* b1sq power (neutral) still needs the uniform-grid envelope. */
        sum_sq = 0.0f;
        for (i = 0; i < num_uniform; ++i)
        {
            rf_abs = (float)sqrt(rf_re_uniform[i] * rf_re_uniform[i] +
                                 rf_im_uniform[i] * rf_im_uniform[i]);
            sum_sq += rf_abs * rf_abs;
        }

        rd->stats.area = sum_signed;
        /* b1sq power: integral |B1_norm(t)|^2 dt (normalised waveform, units: s) */
        rd->stats.total_b1sq_power = sum_sq * rf_raster_us * 1e-6f;

        /* F10.3: a declared vendor with no callback wired would leave
         * vendor_stat[4] silently at 0 -- for GEHC this corrupts
         * minseqrfamp/maxsar inputs (abswidth/effwidth/dtycyc/maxpw) on a
         * wiring regression. Fail closed instead of degrading silently;
         * PULSEG_VENDOR_UNSPECIFIED legitimately has no callback. */
        if (seq->opts.vendor != PULSEG_VENDOR_UNSPECIFIED && !seq->opts.vendor_rf_stats_fn)
        {
            fail_rc = PULSEG_ERR_INVALID_ARGUMENT;
            goto fail;
        }

        /* Vendor-specific envelope stats (D8-A): computed by the optional
         * callback from a read-only view of the uniform-grid envelope;
         * left at 0 when no callback is wired (PULSEG_VENDOR_UNSPECIFIED
         * or a vendor that doesn't need them). */
        if (seq->opts.vendor_rf_stats_fn)
        {
            mag_view = (float *)PULSEG_ALLOC(num_uniform * sizeof(float));
            phase_view = (float *)PULSEG_ALLOC(num_uniform * sizeof(float));
            if (!mag_view || !phase_view)
                goto fail;
            for (i = 0; i < num_uniform; ++i)
            {
                mag_view[i] = (float)sqrt(rf_re_uniform[i] * rf_re_uniform[i] +
                                          rf_im_uniform[i] * rf_im_uniform[i]);
                phase_view[i] = (float)atan2((double)rf_im_uniform[i], (double)rf_re_uniform[i]);
            }
            {
                pulseg_rf_view view;
                view.mag = mag_view;
                view.phase = phase_view;
                view.n = num_uniform;
                view.dt_us = rf_raster_us;
                view.duration_us = duration;
                view.tr_duration_us = 0.0f; /* not yet known at dedup time */
                seq->opts.vendor_rf_stats_fn(seq->opts.vendor_rf_stats_ctx, &view, rd->stats.vendor_stat);
            }
            PULSEG_FREE(mag_view);
            mag_view = NULL;
            PULSEG_FREE(phase_view);
            phase_view = NULL;
        }

        PULSEG_FREE(time_us_uniform);
        time_us_uniform = NULL;
        PULSEG_FREE(rf_re_uniform);
        rf_re_uniform = NULL;
        PULSEG_FREE(rf_im_uniform);
        rf_im_uniform = NULL;

        /* bandwidth via FFT */
        if (fft_ready && time_us)
        {
            time_centered = (float *)PULSEG_ALLOC(num_samples * sizeof(float));
            if (time_centered)
            {
                for (i = 0; i < num_samples; ++i)
                    time_centered[i] = time_us[i] - time_center;
                pulseg__interp1_linear_complex(rfs_re, rfs_im,
                                                  tt, nn,
                                                  time_centered, rf_re, rf_im, num_samples);
                rd->stats.bandwidth_hz = compute_rf_bandwidth_fft(
                    rfs_re, rfs_im, fft_cfg, nn, cutoff,
                    duration * 1e-6f, w, work_re, work_im, fft_in, fft_out);
                /* After compute_rf_bandwidth_fft, work_re[i]/work_im[i] contain the
                 * fftshifted complex spectrum.  Reuse them for multiband detection. */
                {
                    int num_b, in_band;
                    float peak_max_spec, threshold;
                    float wsum, msum;
                    peak_max_spec = 0.0f;
                    for (i = 0; i < nn; ++i)
                    {
                        work_re[i] = (float)sqrt(work_re[i] * work_re[i] +
                                                 work_im[i] * work_im[i]);
                        if (work_re[i] > peak_max_spec)
                            peak_max_spec = work_re[i];
                    }
                    rd->stats.band_bandwidth_hz = rd->stats.bandwidth_hz;
                    if (peak_max_spec > 1e-9f)
                    {
                        threshold = 0.3f * peak_max_spec;
                        num_b = 0;
                        in_band = 0;
                        wsum = 0.0f;
                        msum = 0.0f;
                        for (i = 0; i <= nn; ++i)
                        {
                            float m = (i < nn) ? work_re[i] : 0.0f;
                            if (m >= threshold)
                            {
                                wsum += w[i] * m;
                                msum += m;
                                in_band = 1;
                            }
                            else if (in_band)
                            {
                                if (num_b < PULSEG_MAX_BANDS)
                                    rd->stats.band_freq_offsets_hz[num_b] =
                                        (msum > 0.0f) ? (wsum / msum) : 0.0f;
                                num_b++;
                                wsum = 0.0f;
                                msum = 0.0f;
                                in_band = 0;
                            }
                        }
                        if (num_b >= 1)
                            rd->stats.num_bands = num_b;
                    }
                }
                PULSEG_FREE(time_centered);
                time_centered = NULL;
            }
        }
        if (rf_re)
        {
            PULSEG_FREE(rf_re);
            rf_re = NULL;
        }
        if (rf_im)
        {
            PULSEG_FREE(rf_im);
            rf_im = NULL;
        }
        if (magnitude)
        {
            PULSEG_FREE(magnitude);
            magnitude = NULL;
        }
        if (phase)
        {
            PULSEG_FREE(phase);
            phase = NULL;
        }
        if (time_us)
        {
            PULSEG_FREE(time_us);
            time_us = NULL;
        }
    }

    if (tt)
        PULSEG_FREE(tt);
    if (w)
        PULSEG_FREE(w);
    if (rfs_re)
        PULSEG_FREE(rfs_re);
    if (rfs_im)
        PULSEG_FREE(rfs_im);
    if (work_re)
        PULSEG_FREE(work_re);
    if (work_im)
        PULSEG_FREE(work_im);
    if (fft_in)
        KISS_FFT_FREE(fft_in);
    if (fft_out)
        KISS_FFT_FREE(fft_out);
    if (fft_cfg)
        kiss_fft_free(fft_cfg);
    return PULSEG_SUCCESS;

fail:
    if (tt)
        PULSEG_FREE(tt);
    if (w)
        PULSEG_FREE(w);
    if (rfs_re)
        PULSEG_FREE(rfs_re);
    if (rfs_im)
        PULSEG_FREE(rfs_im);
    if (work_re)
        PULSEG_FREE(work_re);
    if (work_im)
        PULSEG_FREE(work_im);
    if (fft_in)
        KISS_FFT_FREE(fft_in);
    if (fft_out)
        KISS_FFT_FREE(fft_out);
    if (fft_cfg)
        kiss_fft_free(fft_cfg);
    if (magnitude)
        PULSEG_FREE(magnitude);
    if (phase)
        PULSEG_FREE(phase);
    if (time_us)
        PULSEG_FREE(time_us);
    if (rf_re)
        PULSEG_FREE(rf_re);
    if (rf_im)
        PULSEG_FREE(rf_im);
    if (time_us_uniform)
        PULSEG_FREE(time_us_uniform);
    if (rf_re_uniform)
        PULSEG_FREE(rf_re_uniform);
    if (rf_im_uniform)
        PULSEG_FREE(rf_im_uniform);
    if (time_centered)
        PULSEG_FREE(time_centered);
    if (mag_view)
        PULSEG_FREE(mag_view);
    if (phase_view)
        PULSEG_FREE(phase_view);
    return fail_rc;
}

/* ================================================================== */
/*  Copy auxiliary libraries                                          */
/* ================================================================== */

static int copy_rotation_library(const pulseg_pulseq_file *seq, pulseg_sequence_descriptor *desc)
{
    int i, num = seq->rotation_library_size;

    desc->num_rotations = 0;
    desc->rotation_matrices = NULL;
    if (num <= 0 || !seq->rotation_quaternion_library)
        return PULSEG_SUCCESS;

    desc->rotation_matrices = (float (*)[9])PULSEG_ALLOC(num * sizeof(float[9]));
    if (!desc->rotation_matrices)
        return PULSEG_ERR_ALLOC_FAILED;

    for (i = 0; i < num; ++i)
        pulseg__quaternion_to_matrix(desc->rotation_matrices[i], seq->rotation_quaternion_library[i]);
    desc->num_rotations = num;
    return PULSEG_SUCCESS;
}

static int copy_trigger_library(const pulseg_pulseq_file *seq, pulseg_sequence_descriptor *desc)
{
    int i, num = seq->trigger_library_size;

    desc->num_triggers = 0;
    desc->trigger_events = NULL;
    if (num <= 0 || !seq->trigger_library)
        return PULSEG_SUCCESS;

    desc->trigger_events = (pulseg_trigger_event *)PULSEG_ALLOC(num * sizeof(pulseg_trigger_event));
    if (!desc->trigger_events)
        return PULSEG_ERR_ALLOC_FAILED;

    for (i = 0; i < num; ++i)
    {
        desc->trigger_events[i].type = 1;
        desc->trigger_events[i].trigger_type = (int)seq->trigger_library[i][0];
        desc->trigger_events[i].trigger_channel = (int)seq->trigger_library[i][1];
        desc->trigger_events[i].delay = (long)seq->trigger_library[i][2];
        desc->trigger_events[i].duration = (long)seq->trigger_library[i][3];
    }
    desc->num_triggers = num;
    return PULSEG_SUCCESS;
}

static int copy_rf_shim_library(const pulseg_pulseq_file *seq, pulseg_sequence_descriptor *desc)
{
    int i, j, num = seq->rf_shim_library_size;
    const pulseg__rf_shim_entry *entry;

    desc->num_rf_shims = 0;
    desc->rf_shim_definitions = NULL;
    if (num <= 0 || !seq->rf_shim_library)
        return PULSEG_SUCCESS;

    desc->rf_shim_definitions = (pulseg_rf_shim_definition *)PULSEG_ALLOC(
        num * sizeof(pulseg_rf_shim_definition));
    if (!desc->rf_shim_definitions)
        return PULSEG_ERR_ALLOC_FAILED;

    for (i = 0; i < num; ++i)
    {
        entry = &seq->rf_shim_library[i];
        desc->rf_shim_definitions[i].id = i;
        desc->rf_shim_definitions[i].num_channels = entry->num_channels;
        for (j = 0; j < entry->num_channels && j < PULSEG_MAX_RF_SHIM_CHANNELS; ++j)
        {
            desc->rf_shim_definitions[i].magnitudes[j] = entry->values[2 * j];
            desc->rf_shim_definitions[i].phases[j] = entry->values[2 * j + 1];
        }
        for (j = entry->num_channels; j < PULSEG_MAX_RF_SHIM_CHANNELS; ++j)
        {
            desc->rf_shim_definitions[i].magnitudes[j] = 0.0f;
            desc->rf_shim_definitions[i].phases[j] = 0.0f;
        }
    }
    desc->num_rf_shims = num;
    return PULSEG_SUCCESS;
}

static int copy_shapes_library(const pulseg_pulseq_file *seq, pulseg_sequence_descriptor *desc)
{
    int i, j, num = seq->shapes_library_size;
    int ns;

    desc->num_shapes = 0;
    desc->shapes = NULL;
    if (num <= 0 || !seq->shapes_library)
        return PULSEG_SUCCESS;

    desc->shapes = (pulseg_shape_arbitrary *)PULSEG_ALLOC(num * sizeof(pulseg_shape_arbitrary));
    if (!desc->shapes)
        return PULSEG_ERR_ALLOC_FAILED;

    for (i = 0; i < num; ++i)
    {
        desc->shapes[i].num_samples = 0;
        desc->shapes[i].num_uncompressed_samples = 0;
        desc->shapes[i].samples = NULL;
    }
    for (i = 0; i < num; ++i)
    {
        ns = seq->shapes_library[i].num_samples;
        desc->shapes[i].num_samples = ns;
        desc->shapes[i].num_uncompressed_samples = seq->shapes_library[i].num_uncompressed_samples;
        if (ns > 0 && seq->shapes_library[i].samples)
        {
            desc->shapes[i].samples = (float *)PULSEG_ALLOC(ns * sizeof(float));
            if (!desc->shapes[i].samples)
            {
                for (j = 0; j < i; ++j)
                    if (desc->shapes[j].samples)
                        PULSEG_FREE(desc->shapes[j].samples);
                PULSEG_FREE(desc->shapes);
                desc->shapes = NULL;
                return PULSEG_ERR_ALLOC_FAILED;
            }
            memcpy(desc->shapes[i].samples, seq->shapes_library[i].samples,
                   ns * sizeof(float));
        }
    }
    desc->num_shapes = num;
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Raster-time divisibility check                                    */
/* ================================================================== */

/*
 * Verify that two raster times are integer-multiples of each other.
 * If *either* value is <= 0 the check is skipped (value not set).
 * Returns 1 on success, 0 on failure.
 */
static int rasters_compatible(float a, float b)
{
    float big, small, ratio, rounded;
    if (a <= 0.0f || b <= 0.0f)
        return 1;
    big = (a > b) ? a : b;
    small = (a > b) ? b : a;
    ratio = big / small;
    rounded = (float)((int)(ratio));
    return ((float)fabs(ratio - rounded) < 1e-4f * ratio);
}

/*
 * Check all four raster pairs (sequence-defined vs system opts).
 * Returns PULSEG_SUCCESS or PULSEG_ERR_RASTER_MISMATCH.
 */
static int check_raster_times(const pulseg_pulseq_file *seq)
{
    const pulseg__reserved_definitions *rd = &seq->reserved_definitions_library;
    const pulseg_opts *opts = &seq->opts;

    if (rd->radiofrequency_raster_time > 0.0f &&
        !rasters_compatible(rd->radiofrequency_raster_time, opts->rf_raster_us))
        return PULSEG_ERR_RASTER_MISMATCH;

    if (rd->gradient_raster_time > 0.0f &&
        !rasters_compatible(rd->gradient_raster_time, opts->grad_raster_us))
        return PULSEG_ERR_RASTER_MISMATCH;

    if (rd->adc_raster_time > 0.0f &&
        !rasters_compatible(rd->adc_raster_time, opts->adc_raster_us))
        return PULSEG_ERR_RASTER_MISMATCH;

    if (rd->block_duration_raster > 0.0f &&
        !rasters_compatible(rd->block_duration_raster, opts->block_raster_us))
        return PULSEG_ERR_RASTER_MISMATCH;

    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  get_unique_blocks                                                 */
/* ================================================================== */

int pulseg__get_unique_blocks(pulseg_sequence_descriptor *desc, const pulseg_pulseq_file *seq)
{
    int result, num_blocks, num_unique_rf, num_unique_grad, num_unique_adc;
    int n;

    pulseg_rf_definition *tmp_rf_defs = NULL;
    pulseg_rf_table_element *tmp_rf_tab = NULL;
    pulseg_grad_definition *tmp_grad_defs = NULL;
    pulseg_grad_table_element *tmp_grad_tab = NULL;
    pulseg_adc_definition *tmp_adc_defs = NULL;
    pulseg_adc_table_element *tmp_adc_tab = NULL;
    pulseg_block_definition *tmp_blk_defs = NULL;
    pulseg_block_table_element *tmp_blk_tab = NULL;

    int (*int_rows)[BLOCK_DEF_COLS] = NULL;
    int *unique_defs = NULL;
    int *event_table = NULL;

    pulseg_pulseq_raw_block raw;
    pulseg_pulseq_raw_extension ext;
    int norot_flag, nopos_flag, once_flag, pmc_flag, nav_flag, once_counter;
    int module_id;
    int has_prep, has_cooldown, ctrl;

    if (!seq || !desc)
        return PULSEG_ERR_INVALID_ARGUMENT;

    num_blocks = seq->num_blocks;
    if (num_blocks <= 0 || !seq->block_library)
        return PULSEG_ERR_INVALID_ARGUMENT;

    desc->num_prep_blocks = 0;
    desc->num_cooldown_blocks = 0;
    desc->num_unique_rfs = 0;
    desc->num_unique_grads = 0;
    desc->num_unique_adcs = 0;
    desc->num_unique_blocks = 0;
    desc->num_blocks = 0;
    desc->rf_table_size = 0;
    desc->grad_table_size = 0;
    desc->adc_table_size = 0;

    /* rasters */
    desc->rf_raster_us = (seq->reserved_definitions_library.radiofrequency_raster_time > 0.0f)
                             ? seq->reserved_definitions_library.radiofrequency_raster_time
                             : seq->opts.rf_raster_us;
    desc->grad_raster_us = (seq->reserved_definitions_library.gradient_raster_time > 0.0f)
                               ? seq->reserved_definitions_library.gradient_raster_time
                               : seq->opts.grad_raster_us;
    desc->adc_raster_us = (seq->reserved_definitions_library.adc_raster_time > 0.0f)
                              ? seq->reserved_definitions_library.adc_raster_time
                              : seq->opts.adc_raster_us;
    desc->block_raster_us = (seq->reserved_definitions_library.block_duration_raster > 0.0f)
                                ? seq->reserved_definitions_library.block_duration_raster
                                : seq->opts.block_raster_us;

    /* per-subsequence flags */
    desc->ignore_fov_shift = seq->reserved_definitions_library.ignore_fov_shift;
    desc->enable_pmc = seq->reserved_definitions_library.enable_pmc;
    desc->ignore_averages = seq->reserved_definitions_library.ignore_averages;
    desc->num_gain_cal_readouts = seq->reserved_definitions_library.num_gain_cal_readouts;
    desc->vendor = seq->opts.vendor;
    desc->label_column_map[0] = seq->opts.label_column_map[0];
    desc->label_column_map[1] = seq->opts.label_column_map[1];
    desc->label_column_map[2] = seq->opts.label_column_map[2];
    {
        size_t ext_len = strlen(seq->opts.cache_ext);
        if (ext_len >= sizeof(desc->cache_ext))
            ext_len = sizeof(desc->cache_ext) - 1;
        memcpy(desc->cache_ext, seq->opts.cache_ext, ext_len);
        desc->cache_ext[ext_len] = '\0';
    }

    /* encoding-space definitions */
    memcpy(desc->fov, seq->reserved_definitions_library.fov, sizeof(desc->fov));
    memcpy(desc->matrix, seq->reserved_definitions_library.matrix, sizeof(desc->matrix));
    memcpy(desc->nav_fov, seq->reserved_definitions_library.nav_fov, sizeof(desc->nav_fov));
    memcpy(desc->nav_matrix, seq->reserved_definitions_library.nav_matrix, sizeof(desc->nav_matrix));

    /* deep-copy generic definitions */
    desc->num_definitions = seq->num_definitions;
    desc->definitions = NULL;
    if (seq->num_definitions > 0 && seq->definitions_library)
    {
        int di;
        desc->definitions = (pulseg__definition *)PULSEG_ALLOC(
            (size_t)seq->num_definitions * sizeof(pulseg__definition));
        if (!desc->definitions)
            goto fail;
        for (di = 0; di < seq->num_definitions; ++di)
        {
            strncpy(desc->definitions[di].name,
                    seq->definitions_library[di].name,
                    PULSEG_PULSEQ_DEFINITION_NAME_LENGTH);
            desc->definitions[di].value_size = seq->definitions_library[di].value_size;
            desc->definitions[di].value = NULL;
            if (seq->definitions_library[di].value_size > 0)
            {
                int dj;
                desc->definitions[di].value = (char **)PULSEG_ALLOC(
                    (size_t)seq->definitions_library[di].value_size * sizeof(char *));
                if (!desc->definitions[di].value)
                    goto fail;
                for (dj = 0; dj < seq->definitions_library[di].value_size; ++dj)
                {
                    int slen = (int)strlen(seq->definitions_library[di].value[dj]);
                    desc->definitions[di].value[dj] = (char *)PULSEG_ALLOC((size_t)(slen + 1));
                    if (!desc->definitions[di].value[dj])
                        goto fail;
                    strcpy(desc->definitions[di].value[dj], seq->definitions_library[di].value[dj]);
                }
            }
        }
    }

    /* verify system and sequence raster times are integer multiples */
    {
        int rc = check_raster_times(seq);
        if (PULSEG_FAILED(rc))
            return rc;
    }

    /* ---- allocate temp arrays ---- */
    if (seq->rf_library_size > 0)
    {
        tmp_rf_defs = (pulseg_rf_definition *)PULSEG_ALLOC(seq->rf_library_size * sizeof(pulseg_rf_definition));
        tmp_rf_tab = (pulseg_rf_table_element *)PULSEG_ALLOC(seq->rf_library_size * sizeof(pulseg_rf_table_element));
        if (!tmp_rf_defs || !tmp_rf_tab)
            goto fail;
    }
    if (seq->grad_library_size > 0)
    {
        tmp_grad_defs = (pulseg_grad_definition *)PULSEG_ALLOC(seq->grad_library_size * sizeof(pulseg_grad_definition));
        tmp_grad_tab = (pulseg_grad_table_element *)PULSEG_ALLOC(seq->grad_library_size * sizeof(pulseg_grad_table_element));
        if (!tmp_grad_defs || !tmp_grad_tab)
            goto fail;
    }
    if (seq->adc_library_size > 0)
    {
        tmp_adc_defs = (pulseg_adc_definition *)PULSEG_ALLOC(seq->adc_library_size * sizeof(pulseg_adc_definition));
        tmp_adc_tab = (pulseg_adc_table_element *)PULSEG_ALLOC(seq->adc_library_size * sizeof(pulseg_adc_table_element));
        if (!tmp_adc_defs || !tmp_adc_tab)
            goto fail;
    }
    tmp_blk_defs = (pulseg_block_definition *)PULSEG_ALLOC(num_blocks * sizeof(pulseg_block_definition));
    tmp_blk_tab = (pulseg_block_table_element *)PULSEG_ALLOC(num_blocks * sizeof(pulseg_block_table_element));
    if (!tmp_blk_defs || !tmp_blk_tab)
        goto fail;

    /* ---- step 1: dedup event libraries ---- */
    if (seq->rf_library_size > 0)
    {
        num_unique_rf = deduplicate_rf_library(seq, tmp_rf_defs, tmp_rf_tab);
        desc->num_unique_rfs = num_unique_rf;
        desc->rf_table_size = seq->rf_library_size;
        /* Neutral RF stats (flip angle, amplitudes, area, duration, isodelay,
         * bandwidth, bands, b1sq, num_samples/instances) are always computed;
         * the four vendor-specific envelope stats (vendor_stat[4]) are filled
         * only if the caller wired opts.vendor_rf_stats_fn (D8-A). */
        result = compute_rf_stats(seq, tmp_rf_defs, num_unique_rf, tmp_rf_tab, seq->rf_library_size);
        if (PULSEG_FAILED(result))
            goto fail;
    }
    if (seq->grad_library_size > 0)
    {
        num_unique_grad = deduplicate_grad_library(seq, tmp_grad_defs, tmp_grad_tab);
        desc->grad_table_size = seq->grad_library_size;

        desc->num_unique_grads = num_unique_grad;

        result = compute_grad_shot_indices(seq, tmp_grad_defs, tmp_grad_tab, num_unique_grad);
        if (PULSEG_FAILED(result))
            goto fail;

        result = compute_grad_stats(seq, tmp_grad_defs, num_unique_grad, tmp_grad_tab, seq->grad_library_size);
        if (PULSEG_FAILED(result))
            goto fail;
    }
    if (seq->adc_library_size > 0)
    {
        num_unique_adc = deduplicate_adc_library(seq, tmp_adc_defs, tmp_adc_tab);
        desc->num_unique_adcs = num_unique_adc;
        desc->adc_table_size = seq->adc_library_size;
    }

    /* ---- step 2: block definition matrix ---- */
    int_rows = PULSEG_ALLOC(num_blocks * sizeof(*int_rows));
    unique_defs = (int *)PULSEG_ALLOC(num_blocks * sizeof(int));
    event_table = (int *)PULSEG_ALLOC(num_blocks * sizeof(int));
    if (!int_rows || !unique_defs || !event_table)
        goto fail;

    norot_flag = 0;
    nopos_flag = 0;
    once_flag = 0;
    pmc_flag = 1;
    nav_flag = 0;
    once_counter = 0;
    module_id = 0;

    for (n = 0; n < num_blocks; ++n)
    {
        if (!pulseg_pulseq_get_raw_block_content_ids(seq, &raw, n, 1))
        {
            result = PULSEG_ERR_INVALID_ARGUMENT;
            goto fail;
        }
        int_rows[n][0] = raw.block_duration >= 0 ? raw.block_duration : 0;
        int_rows[n][1] = (raw.rf >= 0 && tmp_rf_tab) ? tmp_rf_tab[raw.rf].id : -1;
        int_rows[n][2] = (raw.gx >= 0 && tmp_grad_tab) ? tmp_grad_tab[raw.gx].id : -1;
        int_rows[n][3] = (raw.gy >= 0 && tmp_grad_tab) ? tmp_grad_tab[raw.gy].id : -1;
        int_rows[n][4] = (raw.gz >= 0 && tmp_grad_tab) ? tmp_grad_tab[raw.gz].id : -1;

        tmp_blk_tab[n].rf_id = raw.rf;
        tmp_blk_tab[n].gx_id = raw.gx;
        tmp_blk_tab[n].gy_id = raw.gy;
        tmp_blk_tab[n].gz_id = raw.gz;
        tmp_blk_tab[n].adc_id = raw.adc;

        tmp_blk_tab[n].duration_us = (raw.rf < 0 && raw.gx < 0 && raw.gy < 0 &&
                                      raw.gz < 0 && raw.adc < 0)
                                         ? (int)(raw.block_duration * desc->block_raster_us)
                                         : -1;

        if (raw.ext_count > 0 && seq->is_extensions_library_parsed && seq->extension_lut)
        {
            pulseg_pulseq_get_raw_extension(seq, &ext, &raw);
            tmp_blk_tab[n].rotation_id = ext.rotation_index;
            tmp_blk_tab[n].digitalout_id = ext.trigger_index;
            tmp_blk_tab[n].rf_shim_id = ext.rf_shim_index;
            norot_flag = (ext.flag.norot >= 0) ? ext.flag.norot : norot_flag;
            nopos_flag = (ext.flag.nopos >= 0) ? ext.flag.nopos : nopos_flag;
            pmc_flag = (ext.flag.pmc >= 0) ? ext.flag.pmc : pmc_flag;
            nav_flag = (ext.flag.nav >= 0) ? ext.flag.nav : nav_flag;
            once_flag = (ext.flag.once >= 0) ? ext.flag.once : once_flag;
            if (ext.flag.once > 0)
                ++once_counter;
            /* MODULE: sticky (pulseq LABEL semantics) -- SET at a block
             * persists until the next SET, exactly like norot/nopos/pmc/nav
             * above. 0 = ungrouped (no MODULE label seen yet). Lives on the
             * per-occurrence block-table entry, never on the deduplicated
             * block definition (int_rows/BLOCK_DEF_COLS above excludes it),
             * so it has zero dedup footprint by construction. */
            module_id = (ext.flag.module_id >= 0) ? ext.flag.module_id : module_id;
        }
        else
        {
            tmp_blk_tab[n].rotation_id = -1;
            tmp_blk_tab[n].digitalout_id = -1;
            tmp_blk_tab[n].rf_shim_id = -1;
        }
        tmp_blk_tab[n].norot_flag = norot_flag;
        tmp_blk_tab[n].nopos_flag = nopos_flag;
        tmp_blk_tab[n].pmc_flag = pmc_flag;
        tmp_blk_tab[n].once_flag = once_flag;
        tmp_blk_tab[n].nav_flag = nav_flag;
        tmp_blk_tab[n].module_id = module_id;
    }

    /* step 3: dedup blocks */
    desc->num_unique_blocks = pulseg__deduplicate_int_rows(unique_defs, event_table, (const int *)int_rows, num_blocks, BLOCK_DEF_COLS);
    desc->num_blocks = num_blocks;

    for (n = 0; n < desc->num_unique_blocks; ++n)
    {
        tmp_blk_defs[n].id = unique_defs[n];
        tmp_blk_defs[n].duration_us = (int)(int_rows[unique_defs[n]][0] * desc->block_raster_us);
        tmp_blk_defs[n].rf_id = int_rows[unique_defs[n]][1];
        tmp_blk_defs[n].gx_id = int_rows[unique_defs[n]][2];
        tmp_blk_defs[n].gy_id = int_rows[unique_defs[n]][3];
        tmp_blk_defs[n].gz_id = int_rows[unique_defs[n]][4];
        tmp_blk_defs[n].adc_id = -1; /* no ADC until proven otherwise */
    }
    for (n = 0; n < num_blocks; ++n)
        tmp_blk_tab[n].id = event_table[n];

    /* step 3b: resolve ADC definition per block definition */
    for (n = 0; n < num_blocks; ++n)
    {
        int blk_def_id, raw_adc, adc_def_id;
        blk_def_id = tmp_blk_tab[n].id;
        raw_adc = tmp_blk_tab[n].adc_id;
        if (raw_adc < 0 || !tmp_adc_tab)
            continue; /* no ADC in this instance */
        adc_def_id = tmp_adc_tab[raw_adc].id;
        if (tmp_blk_defs[blk_def_id].adc_id < 0)
        {
            tmp_blk_defs[blk_def_id].adc_id = adc_def_id; /* first encounter */
        }
        else if (tmp_blk_defs[blk_def_id].adc_id != adc_def_id)
        {
            result = PULSEG_ERR_ADC_DEFINITION_CONFLICT;
            pulseg_sequence_descriptor_free(desc);
            goto fail;
        }
    }

    PULSEG_FREE(int_rows);
    int_rows = NULL;
    PULSEG_FREE(unique_defs);
    unique_defs = NULL;
    PULSEG_FREE(event_table);
    event_table = NULL;

    /* ---- step 4: copy to output (exact sizes) ---- */
#define COPY_ARRAY(dst, src, cnt, type)                            \
    do                                                             \
    {                                                              \
        if ((cnt) > 0)                                             \
        {                                                          \
            (dst) = (type *)PULSEG_ALLOC((cnt) * sizeof(type)); \
            if (!(dst))                                            \
            {                                                      \
                result = PULSEG_ERR_ALLOC_FAILED;               \
                pulseg_sequence_descriptor_free(desc);          \
                goto fail;                                         \
            }                                                      \
            memcpy((dst), (src), (cnt) * sizeof(type));            \
        }                                                          \
    } while (0)

    COPY_ARRAY(desc->rf_definitions, tmp_rf_defs, desc->num_unique_rfs, pulseg_rf_definition);
    COPY_ARRAY(desc->rf_table, tmp_rf_tab, desc->rf_table_size, pulseg_rf_table_element);
    COPY_ARRAY(desc->grad_definitions, tmp_grad_defs, desc->num_unique_grads, pulseg_grad_definition);
    COPY_ARRAY(desc->grad_table, tmp_grad_tab, desc->grad_table_size, pulseg_grad_table_element);
    COPY_ARRAY(desc->adc_definitions, tmp_adc_defs, desc->num_unique_adcs, pulseg_adc_definition);
    COPY_ARRAY(desc->adc_table, tmp_adc_tab, desc->adc_table_size, pulseg_adc_table_element);
    COPY_ARRAY(desc->block_definitions, tmp_blk_defs, desc->num_unique_blocks, pulseg_block_definition);
    COPY_ARRAY(desc->block_table, tmp_blk_tab, num_blocks, pulseg_block_table_element);

#undef COPY_ARRAY

    /* PULSEG_FREE temps - done with them */
    if (tmp_rf_defs)
    {
        PULSEG_FREE(tmp_rf_defs);
        tmp_rf_defs = NULL;
    }
    if (tmp_rf_tab)
    {
        PULSEG_FREE(tmp_rf_tab);
        tmp_rf_tab = NULL;
    }
    if (tmp_grad_defs)
    {
        PULSEG_FREE(tmp_grad_defs);
        tmp_grad_defs = NULL;
    }
    if (tmp_grad_tab)
    {
        PULSEG_FREE(tmp_grad_tab);
        tmp_grad_tab = NULL;
    }
    if (tmp_adc_defs)
    {
        PULSEG_FREE(tmp_adc_defs);
        tmp_adc_defs = NULL;
    }
    if (tmp_adc_tab)
    {
        PULSEG_FREE(tmp_adc_tab);
        tmp_adc_tab = NULL;
    }
    if (tmp_blk_defs)
    {
        PULSEG_FREE(tmp_blk_defs);
        tmp_blk_defs = NULL;
    }
    if (tmp_blk_tab)
    {
        PULSEG_FREE(tmp_blk_tab);
        tmp_blk_tab = NULL;
    }

    /* ---- step 5: auxiliary libraries ---- */
    result = copy_rotation_library(seq, desc);
    if (PULSEG_FAILED(result))
    {
        pulseg_sequence_descriptor_free(desc);
        return result;
    }
    result = copy_trigger_library(seq, desc);
    if (PULSEG_FAILED(result))
    {
        pulseg_sequence_descriptor_free(desc);
        return result;
    }
    result = copy_rf_shim_library(seq, desc);
    if (PULSEG_FAILED(result))
    {
        pulseg_sequence_descriptor_free(desc);
        return result;
    }
    result = copy_shapes_library(seq, desc);
    if (PULSEG_FAILED(result))
    {
        pulseg_sequence_descriptor_free(desc);
        return result;
    }

    /* ---- step 6: prep/cooldown ---- */
    has_prep = 0;
    has_cooldown = 0;
    for (n = 0; n < num_blocks; ++n)
    {
        pulseg_pulseq_get_raw_block_content_ids(seq, &raw, n, 1);
        if (raw.ext_count > 0)
        {
            pulseg_pulseq_get_raw_extension(seq, &ext, &raw);
            if (ext.flag.once == 1)
                has_prep = 1;
            else if (ext.flag.once == 2)
                has_cooldown = 1;
        }
    }
    if (!has_prep && !has_cooldown)
    {
        desc->pass_len = desc->num_blocks;
        return PULSEG_SUCCESS;
    }

    if (has_prep)
    {
        pulseg_pulseq_get_raw_block_content_ids(seq, &raw, 0, 1);
        pulseg_pulseq_get_raw_extension(seq, &ext, &raw);
        if (ext.flag.once != 1)
        {
            pulseg_sequence_descriptor_free(desc);
            return PULSEG_ERR_INVALID_PREP_POSITION;
        }
        ctrl = 0;
        desc->num_prep_blocks = 1;
        while (ctrl == 0 && desc->num_prep_blocks < num_blocks)
        {
            pulseg_pulseq_get_raw_block_content_ids(seq, &raw, desc->num_prep_blocks, 1);
            pulseg_pulseq_get_raw_extension(seq, &ext, &raw);
            if (ext.flag.once != 0)
                desc->num_prep_blocks++;
            else
                ctrl = 1;
        }
    }
    if (has_cooldown)
    {
        ctrl = 0;
        desc->num_cooldown_blocks = 0;
        while (ctrl == 0 && desc->num_cooldown_blocks < num_blocks)
        {
            pulseg_pulseq_get_raw_block_content_ids(seq, &raw, num_blocks - 1 - desc->num_cooldown_blocks, 1);
            pulseg_pulseq_get_raw_extension(seq, &ext, &raw);
            desc->num_cooldown_blocks++;
            if (ext.flag.once == 2)
                ctrl = 1;
        }
        if (ctrl == 0)
        {
            pulseg_sequence_descriptor_free(desc);
            return PULSEG_ERR_INVALID_COOLDOWN_POSITION;
        }
    }
    if (once_counter != (desc->num_prep_blocks > 0 ? 1 : 0) + (desc->num_cooldown_blocks > 0 ? 1 : 0))
    {
        /* Multi-pass detection with per-section verification.
         *
         * A pass boundary is where the once_flag transitions back to
         * the value of the first block (e.g. 2->1 or 0->1).  We split
         * the block table at every such transition, verify per-section
         * structural identity across passes, and set pass_len.
         * No folding — the full block table is preserved so that
         * per-instance RF/ADC freq/phase data is retained.
         * No period-finding here — that is get_tr's responsibility. */
        int first_once, prev_once_val;
        int *pass_starts;
        int num_passes_found, pass_len;
        int num_prep_in_pass, num_cool_in_pass, num_main_in_pass;
        int i, j, p, ok;

        first_once = desc->block_table[0].once_flag;

        /* --- Phase A: Find pass boundaries --- */
        pass_starts = (int *)PULSEG_ALLOC((size_t)(num_blocks + 1) * sizeof(int));
        if (!pass_starts)
        {
            pulseg_sequence_descriptor_free(desc);
            return PULSEG_ERR_ALLOC_FAILED;
        }

        num_passes_found = 1;
        pass_starts[0] = 0;
        prev_once_val = first_once;
        for (i = 1; i < num_blocks; ++i)
        {
            int cur = desc->block_table[i].once_flag;
            if (cur == first_once && prev_once_val != first_once)
                pass_starts[num_passes_found++] = i;
            prev_once_val = cur;
        }
        pass_starts[num_passes_found] = num_blocks; /* sentinel */

        /* --- Phase B: Reject uneven passes --- */
        pass_len = pass_starts[1] - pass_starts[0];
        if (num_passes_found < 2 || num_blocks != num_passes_found * pass_len)
        {
            PULSEG_FREE(pass_starts);
            pulseg_sequence_descriptor_free(desc);
            return PULSEG_ERR_INVALID_ONCE_FLAGS;
        }

        /* Verify every pass has the same length */
        ok = 1;
        for (i = 1; i < num_passes_found && ok; ++i)
        {
            if (pass_starts[i + 1] - pass_starts[i] != pass_len)
                ok = 0;
        }
        if (!ok)
        {
            PULSEG_FREE(pass_starts);
            pulseg_sequence_descriptor_free(desc);
            return PULSEG_ERR_INVALID_ONCE_FLAGS;
        }

        /* --- Phase C: Count section sizes within first pass --- */
        num_prep_in_pass = 0;
        for (i = pass_starts[0]; i < pass_starts[0] + pass_len; ++i)
        {
            if (desc->block_table[i].once_flag != 1)
                break;
            num_prep_in_pass++;
        }

        num_cool_in_pass = 0;
        for (i = pass_starts[0] + pass_len - 1; i >= pass_starts[0]; --i)
        {
            if (desc->block_table[i].once_flag != 2)
                break;
            num_cool_in_pass++;
        }

        num_main_in_pass = pass_len - num_prep_in_pass - num_cool_in_pass;
        if (num_main_in_pass < 0)
        {
            PULSEG_FREE(pass_starts);
            pulseg_sequence_descriptor_free(desc);
            return PULSEG_ERR_INVALID_ONCE_FLAGS;
        }

        /* --- Phase D+E: Compare passes 1..N-1 per section --- */
        for (p = 1; p < num_passes_found && ok; ++p)
        {
            int base_ref = pass_starts[0];
            int base_chk = pass_starts[p];

            /* Prep section */
            for (j = 0; j < num_prep_in_pass && ok; ++j)
            {
                if (desc->block_table[base_chk + j].id !=
                        desc->block_table[base_ref + j].id ||
                    desc->block_table[base_chk + j].once_flag !=
                        desc->block_table[base_ref + j].once_flag)
                {
                    ok = 0;
                }
            }

            /* Main section */
            for (j = 0; j < num_main_in_pass && ok; ++j)
            {
                int off = num_prep_in_pass + j;
                if (desc->block_table[base_chk + off].id !=
                        desc->block_table[base_ref + off].id ||
                    desc->block_table[base_chk + off].once_flag !=
                        desc->block_table[base_ref + off].once_flag)
                {
                    ok = 0;
                }
            }

            /* Cooldown section */
            for (j = 0; j < num_cool_in_pass && ok; ++j)
            {
                int off = num_prep_in_pass + num_main_in_pass + j;
                if (desc->block_table[base_chk + off].id !=
                        desc->block_table[base_ref + off].id ||
                    desc->block_table[base_chk + off].once_flag !=
                        desc->block_table[base_ref + off].once_flag)
                {
                    ok = 0;
                }
            }
        }

        PULSEG_FREE(pass_starts);

        if (!ok)
        {
            pulseg_sequence_descriptor_free(desc);
            return PULSEG_ERR_INVALID_ONCE_FLAGS;
        }

        /* --- Phase F: Set descriptor fields (NO folding) --- */
        desc->num_passes = num_passes_found;
        desc->pass_len = pass_len;
        desc->num_prep_blocks = num_prep_in_pass;
        desc->num_cooldown_blocks = num_cool_in_pass;
        /* num_blocks stays as-is — full unfolded block table preserved */
    }
    else
    {
        /* Single-pass: pass_len equals num_blocks */
        desc->pass_len = desc->num_blocks;
    }

    return PULSEG_SUCCESS;

fail:
    if (tmp_rf_defs)
        PULSEG_FREE(tmp_rf_defs);
    if (tmp_rf_tab)
        PULSEG_FREE(tmp_rf_tab);
    if (tmp_grad_defs)
        PULSEG_FREE(tmp_grad_defs);
    if (tmp_grad_tab)
        PULSEG_FREE(tmp_grad_tab);
    if (tmp_adc_defs)
        PULSEG_FREE(tmp_adc_defs);
    if (tmp_adc_tab)
        PULSEG_FREE(tmp_adc_tab);
    if (tmp_blk_defs)
        PULSEG_FREE(tmp_blk_defs);
    if (tmp_blk_tab)
        PULSEG_FREE(tmp_blk_tab);
    if (int_rows)
        PULSEG_FREE(int_rows);
    if (unique_defs)
        PULSEG_FREE(unique_defs);
    if (event_table)
        PULSEG_FREE(event_table);
    return PULSEG_ERR_ALLOC_FAILED;
}
