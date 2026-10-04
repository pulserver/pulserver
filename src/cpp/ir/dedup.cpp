/**
 * @file dedup.cpp
 * @brief The definitions a scan plays and the table of what each block plays
 *        of them, read from the deduplication pypulseqpp performed.
 *
 * A sequence repeats the same RF pulse, gradient shape and ADC hundreds of
 * times with only amplitudes differing, and pypulseqpp interns them as it is
 * built: each event onto a definition, each block onto a definition of the
 * events it plays. This pass takes those and lays them out as the cache holds
 * them -- a definition library, and a table of per-block instances naming a
 * definition, an amplitude and a shot -- which is the split that lets the
 * pulse generator materialise waveform memory once per definition rather than
 * once per block.
 *
 * The statistics of each definition's waveform are measured here, since they
 * are what a scanner checks and not part of the identity.
 */

#include <string.h>
#include <stdlib.h>
#include <math.h>

/* The passes keep C linkage: the collection is assembled in C, which
 * calls them by the names pulseg_internal.h declares. */
extern "C"
{
#include "pulseg_internal.h"
#include "pulseg.h"
#include "pulseq_file.h"
}

/* ================================================================== */
/*  File-scope constants                                              */
/* ================================================================== */
/* The per-block row the conversion keeps: the duration in block rasters, and
 * the definition of each event the block plays, -1 for one it does not. */
#define BLOCK_DEF_COLS 6
/* What makes a block definition here: the one pypulseqpp published, and the
 * ADC definition it digitises with. */
#define BLOCK_KEY_COLS 2

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

/* A slot holds the row's hash and its label + 1, 0 when empty; the row a
 * label stands for is unique_defs[label]. The table is sized by the labels
 * found rather than by the rows, since a scan of millions of blocks plays a
 * few hundred definitions. */
typedef struct
{
    unsigned int hash;
    int label;
} hash_slot;

static unsigned int hash_row(const int *row, int num_cols)
{
    unsigned int h = 2166136261U;
    int i;
    for (i = 0; i < num_cols; ++i)
    {
        h ^= (unsigned int)row[i];
        h *= 16777619U;
    }
    return h ^ (h >> 15);
}

static void place(hash_slot *table, size_t mask, unsigned int h, int label)
{
    size_t idx = h & mask;
    while (table[idx].label)
        idx = (idx + 1) & mask;
    table[idx].hash = h;
    table[idx].label = label + 1;
}

int pulseg__deduplicate_int_rows(
    int *unique_defs,
    int *event_table,
    const int *int_rows,
    int num_rows,
    int num_cols)
{
    size_t table_size = 1024, mask, idx, k;
    hash_slot *table = NULL;
    int num_unique = 0;
    int r;
    unsigned int h;

    if (num_rows <= 0)
        return 0;

    table = (hash_slot *)PULSEG_ALLOC(table_size * sizeof(hash_slot));
    if (!table)
        return 0;
    memset(table, 0, table_size * sizeof(hash_slot));
    mask = table_size - 1;

    for (r = 0; r < num_rows; ++r)
    {
        const int *row = &int_rows[(size_t)r * num_cols];
        h = hash_row(row, num_cols);
        idx = h & mask;

        while (table[idx].label)
        {
            const int label = table[idx].label - 1;
            if (table[idx].hash == h &&
                array_equal(row, &int_rows[(size_t)unique_defs[label] * num_cols], num_cols))
                break;
            idx = (idx + 1) & mask;
        }
        if (table[idx].label)
        {
            event_table[r] = table[idx].label - 1;
            continue;
        }

        unique_defs[num_unique] = r;
        event_table[r] = num_unique;
        table[idx].hash = h;
        table[idx].label = ++num_unique;

        if ((size_t)num_unique * 2 > table_size)
        {
            hash_slot *grown = (hash_slot *)PULSEG_ALLOC(2 * table_size * sizeof(hash_slot));
            if (!grown)
            {
                PULSEG_FREE(table);
                return 0;
            }
            memset(grown, 0, 2 * table_size * sizeof(hash_slot));
            for (k = 0; k < table_size; ++k)
                if (table[k].label)
                    place(grown, 2 * table_size - 1, table[k].hash, table[k].label - 1);
            PULSEG_FREE(table);
            table = grown;
            table_size *= 2;
            mask = table_size - 1;
        }
    }

    PULSEG_FREE(table);
    return num_unique;
}

/* ================================================================== */
/*  Adopting pypulseqpp's definitions                                 */
/* ================================================================== */

/* Take the definition each row was deduplicated onto into the arrays the
 * conversion keys on: the label of every row, and the first row carrying each
 * label.
 *
 * @p ids are dense and in order of first appearance, which is the numbering
 * pypulseqpp hands out and pulserver.ir preserves when it drops the rows no
 * block plays. An id that is neither one already seen nor the next one would
 * leave a definition with no row to read its waveform from, so it is refused
 * rather than indexed.
 *
 * @return the number of definitions, or a negative error code. */
static int adopt_definitions(const int *ids, int num_rows, int *unique_defs, int *event_table)
{
    int r, num_unique = 0;

    if (num_rows <= 0)
        return 0;
    if (!ids)
        return PULSEG_ERR_NULL_POINTER;
    for (r = 0; r < num_rows; ++r)
    {
        const int id = ids[r];
        if (id < 0 || id > num_unique)
            return PULSEG_ERR_INVALID_ARGUMENT;
        event_table[r] = id;
        if (id == num_unique)
            unique_defs[num_unique++] = r;
    }
    return num_unique;
}

/* ================================================================== */
/*  RF                                                                */
/* ================================================================== */

static int deduplicate_rf_library(
    const pulseq_file *seq,
    pulseg_rf_definition *rf_defs,
    pulseg_rf_table_element *rf_table)
{
    int *unique_defs = NULL;
    int *event_table = NULL;
    int num_unique, num_rows, i;

    num_rows = seq->rf_library_size;
    if (num_rows <= 0)
        return 0;

    unique_defs = (int *)PULSEG_ALLOC(num_rows * sizeof(int));
    event_table = (int *)PULSEG_ALLOC(num_rows * sizeof(int));
    if (!unique_defs || !event_table)
    {
        if (unique_defs)
            PULSEG_FREE(unique_defs);
        if (event_table)
            PULSEG_FREE(event_table);
        return 0;
    }

    num_unique = adopt_definitions(seq->rf_definitions, num_rows, unique_defs, event_table);
    if (num_unique < 0)
    {
        PULSEG_FREE(unique_defs);
        PULSEG_FREE(event_table);
        return num_unique;
    }

    for (i = 0; i < num_unique; ++i)
    {
        const float *rf = seq->rf_library[unique_defs[i]];
        rf_defs[i].id = unique_defs[i];
        rf_defs[i].mag_shape_id = (int)rf[1];
        rf_defs[i].phase_shape_id = (int)rf[2];
        rf_defs[i].time_shape_id = (int)rf[3];
        rf_defs[i].delay = (int)rf[5];
        /* pypulseqpp counts the channels; the time shape is part of the
         * definition, so every row of one holds as many. */
        rf_defs[i].num_channels = seq->rf_channels ? seq->rf_channels[unique_defs[i]] : 1;
    }
    for (i = 0; i < num_rows; ++i)
    {
        const float *rf = seq->rf_library[i];
        rf_table[i].id = event_table[i];
        rf_table[i].amplitude = rf[0];
        rf_table[i].freq_offset = rf[8];  /* ppm resolved on the host (Hz)   */
        rf_table[i].phase_offset = rf[9]; /* ppm resolved on the host (rad)  */
        rf_table[i].rf_use = (seq->rf_use_tags) ? seq->rf_use_tags[i] : PULSEG_RF_USE_UNKNOWN;
    }

    PULSEG_FREE(unique_defs);
    PULSEG_FREE(event_table);
    return num_unique;
}

/* ================================================================== */
/*  Gradients                                                         */
/* ================================================================== */

/* The samples a gradient waveform shape stands for, 0 where it names none. */
static int wave_samples(const pulseq_file *seq, int wave_id)
{
    if (wave_id > 0 && seq->is_shapes_library_parsed && wave_id <= seq->shapes_library_size)
        return seq->shapes_library[wave_id - 1].num_uncompressed_samples;
    return 0;
}

static int deduplicate_grad_library(
    const pulseq_file *seq,
    pulseg_grad_definition *grad_defs,
    pulseg_grad_table_element *grad_table)
{
    int *unique_defs = NULL;
    int *event_table = NULL;
    int num_unique, num_rows, i;

    num_rows = seq->grad_library_size;
    if (num_rows <= 0)
        return 0;

    unique_defs = (int *)PULSEG_ALLOC(num_rows * sizeof(int));
    event_table = (int *)PULSEG_ALLOC(num_rows * sizeof(int));
    if (!unique_defs || !event_table)
    {
        if (unique_defs)
            PULSEG_FREE(unique_defs);
        if (event_table)
            PULSEG_FREE(event_table);
        return 0;
    }

    num_unique = adopt_definitions(seq->grad_definitions, num_rows, unique_defs, event_table);
    if (num_unique < 0)
    {
        PULSEG_FREE(unique_defs);
        PULSEG_FREE(event_table);
        return num_unique;
    }

    for (i = 0; i < num_unique; ++i)
    {
        const float *grad = seq->grad_library[unique_defs[i]];
        const int grad_type = (int)grad[0];

        grad_defs[i].id = unique_defs[i];
        grad_defs[i].type = grad_type;
        if (grad_type == 0)
        {
            grad_defs[i].rise_time_or_unused = (int)grad[2];
            grad_defs[i].flat_time_or_unused = (int)grad[3];
            grad_defs[i].fall_time_or_num_uncompressed_samples = (int)grad[4];
            grad_defs[i].unused_or_time_shape_id = 0;
            grad_defs[i].delay = (int)grad[5];
        }
        else
        {
            grad_defs[i].rise_time_or_unused = 0;
            grad_defs[i].flat_time_or_unused = 0;
            grad_defs[i].fall_time_or_num_uncompressed_samples =
                wave_samples(seq, (int)grad[4]);
            grad_defs[i].unused_or_time_shape_id = (int)grad[5];
            grad_defs[i].delay = (int)grad[6];
        }
    }
    for (i = 0; i < num_rows; ++i)
    {
        grad_table[i].id = event_table[i];
        grad_table[i].amplitude = seq->grad_library[i][1];
    }

    PULSEG_FREE(unique_defs);
    PULSEG_FREE(event_table);
    return num_unique;
}

/* ================================================================== */
/*  ADC                                                               */
/* ================================================================== */

static int deduplicate_adc_library(
    const pulseq_file *seq,
    pulseg_adc_definition *adc_defs,
    pulseg_adc_table_element *adc_table)
{
    int *unique_defs = NULL;
    int *event_table = NULL;
    int num_unique, num_rows, i;

    num_rows = seq->adc_library_size;
    if (num_rows <= 0)
        return 0;

    unique_defs = (int *)PULSEG_ALLOC(num_rows * sizeof(int));
    event_table = (int *)PULSEG_ALLOC(num_rows * sizeof(int));
    if (!unique_defs || !event_table)
    {
        if (unique_defs)
            PULSEG_FREE(unique_defs);
        if (event_table)
            PULSEG_FREE(event_table);
        return 0;
    }

    num_unique = adopt_definitions(seq->adc_definitions, num_rows, unique_defs, event_table);
    if (num_unique < 0)
    {
        PULSEG_FREE(unique_defs);
        PULSEG_FREE(event_table);
        return num_unique;
    }

    for (i = 0; i < num_unique; ++i)
    {
        const float *adc = seq->adc_library[unique_defs[i]];
        adc_defs[i].id = unique_defs[i];
        adc_defs[i].num_samples = (int)adc[0];
        adc_defs[i].dwell_time = (int)adc[1];
        adc_defs[i].delay = (int)adc[2];
    }
    for (i = 0; i < num_rows; ++i)
    {
        const float *adc = seq->adc_library[i];
        adc_table[i].id = event_table[i];
        adc_table[i].freq_offset = adc[5];  /* ppm resolved on the host (Hz)  */
        adc_table[i].phase_offset = adc[6]; /* ppm resolved on the host (rad) */
        adc_table[i].phase_shape_id = (int)adc[7];
    }

    PULSEG_FREE(unique_defs);
    PULSEG_FREE(event_table);
    return num_unique;
}

static int record_grad_shape_ids(
    const pulseq_file *seq,
    const pulseg_grad_definition *grad_defs,
    pulseg_grad_table_element *grad_table,
    int num_unique_grads)
{
    int num_rows = seq->grad_library_size;
    int i;

    if (num_rows <= 0 || num_unique_grads <= 0)
        return PULSEG_SUCCESS;

    for (i = 0; i < num_rows; ++i)
    {
        int def_idx = grad_table[i].id;
        if (def_idx < 0 || def_idx >= num_unique_grads)
            continue;
        /* A trapezoid is described by its corner times and has no shape. */
        grad_table[i].shape_id = (grad_defs[def_idx].type == 0) ? 0 : (int)seq->grad_library[i][4];
    }
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Trapezoid statistics                                               */
/* ================================================================== */

static void compute_trapezoid_stats(
    float *slew,
    float *energy,
    float *first_val,
    float *last_val,
    float rise_us,
    float flat_us,
    float fall_us)
{
    float rise_s = rise_us * 1e-6f;
    float fall_s = fall_us * 1e-6f;
    float sr, sf;

    *first_val = 0.0f;
    *last_val = 0.0f;

    sr = (rise_s > 0.0f) ? (1.0f / rise_s) : 0.0f;
    sf = (fall_s > 0.0f) ? (1.0f / fall_s) : 0.0f;
    *slew = (sr > sf) ? sr : sf;

    *energy = pulseg__trap_energy(rise_us, flat_us, fall_us);
}

/*
 * Keep @p cand if it scores higher than what @p best already holds.
 */
static void grad_keep_best(pulseg_grad_representative *best, const pulseg_grad_representative *cand)
{
    if (cand->score > best->score)
        *best = *cand;
}

/* ================================================================== */
/*  Gradient statistics                                               */
/* ================================================================== */

static int compute_grad_stats(
    const pulseq_file *seq,
    pulseg_sequence_descriptor *desc,
    pulseg_grad_definition *grad_defs,
    int num_unique,
    const pulseg_grad_table_element *grad_table,
    int grad_table_size)
{
    int def_idx, i, row, grad_type, shape_id;
    float rise_us, flat_us, fall_us, abs_amp;
    float slew_energy, amp2;
    pulseg_grad_representative cand;
    int *shape_def_seen = NULL;
    pulseg_grad_definition *gd;

    if (!seq || !grad_defs || num_unique <= 0)
        return PULSEG_SUCCESS;

    /* Which definition last visited each shape: the distinct-shape walk
     * below must skip repeats without scanning the rows already walked,
     * or the walk prices the table quadratically. */
    if (seq->shapes_library_size > 0)
    {
        shape_def_seen = (int *)PULSEG_ALLOC((size_t)seq->shapes_library_size * sizeof(int));
        if (!shape_def_seen)
            return PULSEG_ERR_ALLOC_FAILED;
        for (i = 0; i < seq->shapes_library_size; ++i)
            shape_def_seen[i] = -1;
    }

    /* Per-shape statistics, filled as the shapes are visited below.  Sized by
     * the shape library rather than by anything per definition, which is what
     * makes it uncapped. */
    if (desc && seq->shapes_library_size > 0)
    {
        desc->num_grad_shape_stats = seq->shapes_library_size;
        desc->grad_shape_first =
            (float *)PULSEG_ALLOC((size_t)seq->shapes_library_size * sizeof(float));
        desc->grad_shape_last =
            (float *)PULSEG_ALLOC((size_t)seq->shapes_library_size * sizeof(float));
        desc->grad_shape_slew =
            (float *)PULSEG_ALLOC((size_t)seq->shapes_library_size * sizeof(float));
        desc->grad_shape_energy =
            (float *)PULSEG_ALLOC((size_t)seq->shapes_library_size * sizeof(float));
        if (!desc->grad_shape_first || !desc->grad_shape_last || !desc->grad_shape_slew ||
            !desc->grad_shape_energy)
        {
            if (shape_def_seen)
                PULSEG_FREE(shape_def_seen);
            return PULSEG_ERR_ALLOC_FAILED;
        }
        for (i = 0; i < seq->shapes_library_size; ++i)
        {
            desc->grad_shape_first[i] = 0.0f;
            desc->grad_shape_last[i] = 0.0f;
            desc->grad_shape_slew[i] = 0.0f;
            desc->grad_shape_energy[i] = 0.0f;
        }
    }

    for (def_idx = 0; def_idx < num_unique; ++def_idx)
    {
        gd = &grad_defs[def_idx];
        grad_type = gd->type;

        {
            pulseg_grad_representative empty = PULSEG_GRAD_REPRESENTATIVE_INIT;
            gd->spectral = empty;
        }
        {
            pulseg_grad_aggregate zero = PULSEG_GRAD_AGGREGATE_INIT;
            gd->any = zero;
        }

        /* Amplitude bounds over every instance of this definition. */
        gd->any.min_amplitude = 1e30f;
        if (grad_table && grad_table_size > 0)
        {
            for (i = 0; i < grad_table_size; ++i)
            {
                if (grad_table[i].id != def_idx)
                    continue;
                abs_amp = grad_table[i].amplitude;
                if (abs_amp < 0.0f)
                    abs_amp = -abs_amp;
                if (abs_amp > gd->any.max_amplitude)
                    gd->any.max_amplitude = abs_amp;
                if (abs_amp < gd->any.min_amplitude)
                    gd->any.min_amplitude = abs_amp;
            }
        }
        if (gd->any.min_amplitude > 1e29f)
            gd->any.min_amplitude = 0.0f;

        if (grad_type == 0)
        {
            rise_us = (float)gd->rise_time_or_unused;
            flat_us = (float)gd->flat_time_or_unused;
            fall_us = (float)gd->fall_time_or_num_uncompressed_samples;
            compute_trapezoid_stats(
                &cand.slew_rate,
                &cand.energy,
                &cand.first_value,
                &cand.last_value,
                rise_us,
                flat_us,
                fall_us);

            /* A trapezoid has one shape, so both representatives are it.
             * Its slew energy is closed form: the derivative is 1/rise on the
             * ramp up and 1/fall on the ramp down, so the integral of its
             * square is 1/rise + 1/fall. */
            slew_energy = 0.0f;
            if (rise_us > 0.0f)
                slew_energy += 1.0f / (rise_us * 1e-6f);
            if (fall_us > 0.0f)
                slew_energy += 1.0f / (fall_us * 1e-6f);

            cand.shape_id = 0;
            cand.amplitude = gd->any.max_amplitude;
            amp2 = cand.amplitude * cand.amplitude;
            if (cand.slew_rate > gd->any.max_slew_rate)
                gd->any.max_slew_rate = cand.slew_rate;

            cand.score = amp2 * slew_energy;
            grad_keep_best(&gd->spectral, &cand);
            continue;
        }

        /* An arbitrary gradient: each of this definition's distinct shapes,
         * enumerated from the instance table, since one definition covers
         * every shape of its sample count.  The statistics are pypulseqpp's,
         * measured on the waveform the event plays (seq->grad_statistics),
         * over the amplitude that plays it, so they belong to the normalised
         * shape; the edge values are the ones the library row stores. */
        for (row = 0; row < grad_table_size; ++row)
        {
            float amplitude;
            const PULSEQ_REAL *measured;

            if (!grad_table || grad_table[row].id != def_idx)
                continue;
            shape_id = grad_table[row].shape_id;
            if (shape_id <= 0 || shape_id > seq->shapes_library_size)
                continue;
            if (shape_def_seen && shape_def_seen[shape_id - 1] == def_idx)
                continue;
            if (shape_def_seen)
                shape_def_seen[shape_id - 1] = def_idx;

            cand.first_value = 0.0f;
            cand.last_value = 0.0f;
            cand.slew_rate = 0.0f;
            cand.energy = 0.0f;
            slew_energy = 0.0f;
            amplitude = (float)seq->grad_library[row][1];
            if (amplitude > 1e-9f || amplitude < -1e-9f)
            {
                abs_amp = amplitude < 0.0f ? -amplitude : amplitude;
                cand.first_value = (float)seq->grad_library[row][2] / amplitude;
                cand.last_value = (float)seq->grad_library[row][3] / amplitude;
                if (seq->grad_statistics)
                {
                    measured = seq->grad_statistics[row];
                    cand.slew_rate = (float)measured[0] / abs_amp;
                    cand.energy = (float)(measured[1] / ((double)abs_amp * abs_amp));
                    slew_energy = (float)(measured[2] / ((double)abs_amp * abs_amp));
                }
            }
            if (desc && desc->grad_shape_first && shape_id <= desc->num_grad_shape_stats)
            {
                desc->grad_shape_first[shape_id - 1] = cand.first_value;
                desc->grad_shape_last[shape_id - 1] = cand.last_value;
                desc->grad_shape_slew[shape_id - 1] = cand.slew_rate;
                desc->grad_shape_energy[shape_id - 1] = cand.energy;
            }
            {
                float fv = cand.first_value < 0.0f ? -cand.first_value : cand.first_value;
                float lv = cand.last_value < 0.0f ? -cand.last_value : cand.last_value;
                if (fv > gd->any.max_abs_first)
                    gd->any.max_abs_first = fv;
                if (lv > gd->any.max_abs_last)
                    gd->any.max_abs_last = lv;
            }
            if (cand.slew_rate > gd->any.max_slew_rate)
                gd->any.max_slew_rate = cand.slew_rate;

            cand.shape_id = shape_id;
            cand.amplitude = gd->any.max_amplitude;
            amp2 = cand.amplitude * cand.amplitude;
            cand.score = amp2 * slew_energy;
            grad_keep_best(&gd->spectral, &cand);
        }
    }
    if (shape_def_seen)
        PULSEG_FREE(shape_def_seen);
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  RF statistics                                                     */
/* ================================================================== */

/*
 * The spectral statistics -- bandwidth and multiband split -- are measured by
 * pypulseqpp when the conversion input is built and arrive per RF event in
 * seq->rf_spectra; a definition takes those of the first event that plays it.
 * What is computed here is what the time-domain envelope gives.
 */

static int compute_rf_stats(
    const pulseq_file *seq,
    pulseg_rf_definition *rf_defs,
    int num_unique,
    const pulseg_rf_table_element *rf_table,
    int rf_table_size,
    const pulseg_opts *opts)
{
    int def_idx, i;
    pulseq_shape decomp_mag, decomp_phase, decomp_time;
    float *magnitude = NULL;
    float *phase = NULL;
    float *time_us = NULL;
    float *time_us_uniform = NULL;
    float *rf_re = NULL;
    float *rf_im = NULL;
    float *rf_re_uniform = NULL;
    float *rf_im_uniform = NULL;
    int num_samples, num_uniform, num_real;
    int mag_id, phase_id, time_id;
    int has_phase, has_time, on_raster;
    float max_mag, duration, last_us, time_center, rf_raster_us;
    pulseg_rf_definition *rd;


    float sum_signed;
    float *mag_view = NULL;
    float *phase_view = NULL;
    int fail_rc = PULSEG_ERR_ALLOC_FAILED;

    if (!seq || !rf_defs || num_unique <= 0)
        return PULSEG_SUCCESS;

    if (seq->reserved_definitions_library.radiofrequency_raster_time > 0.0f)
        rf_raster_us = seq->reserved_definitions_library.radiofrequency_raster_time;
    else
        rf_raster_us = opts->rf_raster_us;

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
        rd->stats.vendor = opts->vendor;
        {
            int bi;
            for (bi = 0; bi < PULSEG_MAX_BANDS; ++bi)
                rd->stats.band_freq_offsets_hz[bi] = 0.0f;
        }

        /* The largest amplitude any event of the definition plays, the flip
         * angle pypulseqpp gives that event, and pypulseqpp's integral of the
         * unit-peak envelope, which every event of the definition shares. */
        if (rf_table && rf_table_size > 0)
        {
            for (i = 0; i < rf_table_size; ++i)
            {
                if (rf_table[i].id == def_idx)
                {
                    float amp = (float)fabs(rf_table[i].amplitude);
                    if (seq->rf_flip_deg)
                    {
                        float flip = (float)(seq->rf_flip_deg[i] * M_PI / 180.0);
                        if (flip > rd->stats.flip_angle_rad)
                            rd->stats.flip_angle_rad = flip;
                    }
                    if (amp > rd->stats.base_amplitude_hz)
                        rd->stats.base_amplitude_hz = amp;
                    if (seq->rf_b1sq_integral &&
                        (float)seq->rf_b1sq_integral[i] > rd->stats.total_b1sq_power)
                        rd->stats.total_b1sq_power = (float)seq->rf_b1sq_integral[i];
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
        num_samples = 0;
        duration = 0.0f;

        /* decompress magnitude */
        if (!pulseq_decompress_shape(&decomp_mag, &seq->shapes_library[mag_id - 1], 1.0f))
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
            /* Pulseq stores an RF phase shape as phase/(2*pi), so a pi phase
             * flip is a sample of 0.5.  The reconstruction below and the
             * real-valued test treat phase as radians. */
            if (!pulseq_decompress_shape(
                    &decomp_phase,
                    &seq->shapes_library[phase_id - 1],
                    (float)(2.0 * M_PI)))
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
                if ((float)fabs(phase[i]) < 1e-6f || (float)fabs(phase[i] - (float)M_PI) < 1e-6f)
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
            if (!pulseq_decompress_shape(
                    &decomp_time,
                    &seq->shapes_library[time_id - 1],
                    rf_raster_us))
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
        on_raster = !has_time;
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

        /* The shape duration, as pypulseqpp counts it: the samples on the RF
         * raster, or the last sample time rounded up onto the raster. */
        last_us = (num_samples > 0) ? time_us[num_samples - 1] : 0.0f;
        if (on_raster)
            duration = (float)num_samples * rf_raster_us;
        else
            duration = (float)(ceil((double)last_us / rf_raster_us - 1e-6) * rf_raster_us);
        rd->stats.duration_us = duration;

        /* The isodelay is counted from the centre the design records, which
         * the definition is keyed on. */
        max_mag = pulseg__get_max_abs_real(magnitude, num_samples);
        time_center = seq->rf_library[rd->id][4];
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

        /* uniform grid, from the first raster point to the last sample */
        num_uniform = (int)(last_us / rf_raster_us) + 1;
        if (num_uniform < 2)
            num_uniform = 2;

        time_us_uniform = (float *)PULSEG_ALLOC(num_uniform * sizeof(float));
        rf_re_uniform = (float *)PULSEG_ALLOC(num_uniform * sizeof(float));
        rf_im_uniform = (float *)PULSEG_ALLOC(num_uniform * sizeof(float));
        if (!time_us_uniform || !rf_re_uniform || !rf_im_uniform)
            goto fail;

        for (i = 0; i < num_uniform; ++i)
            time_us_uniform[i] = (float)i * rf_raster_us;

        pulseg__interp1_linear_complex(
            rf_re_uniform,
            rf_im_uniform,
            time_us_uniform,
            num_uniform,
            time_us,
            rf_re,
            rf_im,
            num_samples);

        /* The signed integral of the envelope, for the area.
         * Trapezoidal rule on the NATIVE (un-interpolated) time grid. */
        {
            double dre = 0.0;
            if (has_time && time_us && num_samples >= 2)
            {
                for (i = 0; i < num_samples - 1; ++i)
                {
                    double dt = ((double)time_us[i + 1] - (double)time_us[i]) * 1e-6;
                    dre += 0.5 * dt * ((double)rf_re[i] + (double)rf_re[i + 1]);
                }
            }
            else
            {
                /* fall back to uniform-grid integration */
                double dt = (double)rf_raster_us * 1e-6;
                for (i = 0; i < num_samples - 1; ++i)
                {
                    dre += 0.5 * dt * ((double)rf_re[i] + (double)rf_re[i + 1]);
                }
            }
            /* area = signed real part = ∫h_norm dt [s] */
            sum_signed = (float)dre;
        }
        rd->stats.area = sum_signed;

        /* Vendor-specific envelope stats: computed by the optional
         * callback from a read-only view of the uniform-grid envelope;
         * left at 0 when no callback is wired (PULSEG_VENDOR_UNSPECIFIED
         * or a vendor that doesn't need them). */
        if (opts->vendor_rf_stats_fn)
        {
            mag_view = (float *)PULSEG_ALLOC(num_uniform * sizeof(float));
            phase_view = (float *)PULSEG_ALLOC(num_uniform * sizeof(float));
            if (!mag_view || !phase_view)
                goto fail;
            for (i = 0; i < num_uniform; ++i)
            {
                mag_view[i] = (float)sqrt(
                    rf_re_uniform[i] * rf_re_uniform[i] + rf_im_uniform[i] * rf_im_uniform[i]);
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
                opts->vendor_rf_stats_fn(opts->vendor_rf_stats_ctx, &view, rd->stats.vendor_stat);
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

        /* bandwidth and bands, as measured for the first event of this
         * definition; an unmeasurable spectrum reads as zero, for which the
         * analytic width of a hard pulse of this duration stands in. */
        if (seq->rf_spectra)
        {
            for (i = 0; i < rf_table_size; ++i)
            {
                if (rf_table[i].id == def_idx)
                {
                    const PULSEQ_REAL *spectrum = seq->rf_spectra[i];
                    int b, count = (int)spectrum[1];
                    rd->stats.bandwidth_hz = (float)spectrum[0];
                    rd->stats.band_bandwidth_hz = (float)spectrum[2];
                    if (count >= 1)
                        rd->stats.num_bands = count;
                    if (count > PULSEG_MAX_BANDS)
                        count = PULSEG_MAX_BANDS;
                    for (b = 0; b < count; ++b)
                        rd->stats.band_freq_offsets_hz[b] = (float)spectrum[3 + b];
                    break;
                }
            }
        }
        if (!(rd->stats.bandwidth_hz > 0.0f))
        {
            rd->stats.bandwidth_hz = (duration > 0.0f) ? (3.12f / (duration * 1e-6f)) : 0.0f;
            rd->stats.band_bandwidth_hz = rd->stats.bandwidth_hz;
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

    return PULSEG_SUCCESS;

fail:
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
    if (mag_view)
        PULSEG_FREE(mag_view);
    if (phase_view)
        PULSEG_FREE(phase_view);
    return fail_rc;
}

/* ================================================================== */
/*  Copy auxiliary libraries                                          */
/* ================================================================== */

static int copy_rotation_library(const pulseq_file *seq, pulseg_sequence_descriptor *desc)
{
    int i, num = seq->rotation_library_size;

    desc->num_rotations = 0;
    desc->rotation_matrices = NULL;
    if (num <= 0 || !seq->rotation_quaternion_library)
        return PULSEG_SUCCESS;

    desc->rotation_matrices = (float(*)[9])PULSEG_ALLOC(num * sizeof(float[9]));
    if (!desc->rotation_matrices)
        return PULSEG_ERR_ALLOC_FAILED;

    for (i = 0; i < num; ++i)
        pulseg__quaternion_to_matrix(
            desc->rotation_matrices[i],
            seq->rotation_quaternion_library[i]);
    desc->num_rotations = num;
    return PULSEG_SUCCESS;
}

static int copy_trigger_library(const pulseq_file *seq, pulseg_sequence_descriptor *desc)
{
    int i, num = seq->trigger_library_size;

    desc->num_triggers = 0;
    desc->trigger_events = NULL;
    if (num <= 0 || !seq->trigger_library)
        return PULSEG_SUCCESS;

    desc->trigger_events = (pulseq_trigger_event *)PULSEG_ALLOC(num * sizeof(pulseq_trigger_event));
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

static int copy_rf_shim_library(const pulseq_file *seq, pulseg_sequence_descriptor *desc)
{
    int i, j, num = seq->rf_shim_library_size;
    const pulseq_rf_shim_entry *entry;

    desc->num_rf_shims = 0;
    desc->rf_shim_definitions = NULL;
    if (num <= 0 || !seq->rf_shim_library)
        return PULSEG_SUCCESS;

    desc->rf_shim_definitions =
        (pulseg_rf_shim_definition *)PULSEG_ALLOC(num * sizeof(pulseg_rf_shim_definition));
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

/* Give the descriptor the file's shape library. With @p adopt (the file's
 * own shapes_library) the sample buffers move across and the file's entries
 * are left empty; without it they are copied. */
static int copy_shapes_library(
    const pulseq_file *seq,
    pulseg_sequence_descriptor *desc,
    pulseq_shape *adopt)
{
    int i, j, num = seq->shapes_library_size;
    int ns;

    desc->num_shapes = 0;
    desc->shapes = NULL;
    if (num <= 0 || !seq->shapes_library)
        return PULSEG_SUCCESS;

    desc->shapes = (pulseq_shape *)PULSEG_ALLOC(num * sizeof(pulseq_shape));
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
        if (ns > 0 && adopt && adopt[i].samples)
        {
            desc->shapes[i].samples = adopt[i].samples;
            adopt[i].samples = NULL;
        }
        else if (ns > 0 && seq->shapes_library[i].samples)
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
            memcpy(desc->shapes[i].samples, seq->shapes_library[i].samples, ns * sizeof(float));
        }
    }
    desc->num_shapes = num;
    desc->shapes_borrowed = (adopt && seq->shapes_borrowed) ? 1 : 0;
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
static int check_raster_times(const pulseq_file *seq, const pulseg_opts *opts)
{
    const pulseq_reserved_definitions *rd = &seq->reserved_definitions_library;

    if (rd->radiofrequency_raster_time > 0.0f &&
        !rasters_compatible(rd->radiofrequency_raster_time, opts->rf_raster_us))
        return PULSEG_ERR_RASTER_MISMATCH;

    if (rd->gradient_raster_time > 0.0f &&
        !rasters_compatible(rd->gradient_raster_time, opts->grad_raster_us))
        return PULSEG_ERR_RASTER_MISMATCH;

    if (rd->adc_raster_time > 0.0f && !rasters_compatible(rd->adc_raster_time, opts->adc_raster_us))
        return PULSEG_ERR_RASTER_MISMATCH;

    if (rd->block_duration_raster > 0.0f &&
        !rasters_compatible(rd->block_duration_raster, opts->block_raster_us))
        return PULSEG_ERR_RASTER_MISMATCH;

    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  get_unique_blocks                                                 */
/* ================================================================== */

int pulseg__get_unique_blocks(
    pulseg_sequence_descriptor *desc,
    const pulseq_file *seq,
    const pulseg_opts *opts,
    pulseq_shape *adopt_shapes)
{
    /* `result` is only ever a failure code: it starts as the reason an
     * allocation-failure jump would give, and the sites that know better
     * overwrite it before jumping. Helper return values land in `rc` instead,
     * so a helper that succeeded cannot leave a success code here for a later
     * `goto fail` to return. Reporting a structural conflict as "allocation
     * failed" sends the reader hunting a memory problem that is not there. */
    int result = PULSEG_ERR_ALLOC_FAILED;
    int rc;
    int num_blocks, num_unique_rf, num_unique_grad, num_unique_adc;
    int n;

    pulseg_rf_definition *tmp_rf_defs = NULL;
    pulseg_rf_table_element *tmp_rf_tab = NULL;
    pulseg_grad_definition *tmp_grad_defs = NULL;
    pulseg_grad_table_element *tmp_grad_tab = NULL;
    pulseg_adc_definition *tmp_adc_defs = NULL;
    pulseg_adc_table_element *tmp_adc_tab = NULL;
    pulseg_base_block *tmp_blk_defs = NULL;
    pulseg_block_table_element *tmp_blk_tab = NULL;

    int(*int_rows)[BLOCK_DEF_COLS] = NULL;
    int(*key_rows)[BLOCK_KEY_COLS] = NULL;
    int *geometry_defs = NULL;
    int *geometry_of = NULL;
    int *def_map = NULL;
    int *unique_defs = NULL;
    int *event_table = NULL;

    pulseq_raw_block raw;

    if (!seq || !desc)
        return PULSEG_ERR_INVALID_ARGUMENT;

    num_blocks = seq->num_blocks;
    if (num_blocks <= 0 || !seq->block_library || !seq->block_rotations || !seq->block_shims ||
        !seq->block_flags || !seq->block_trid_set)
        return PULSEG_ERR_INVALID_ARGUMENT;

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
        : opts->rf_raster_us;
    desc->grad_raster_us = (seq->reserved_definitions_library.gradient_raster_time > 0.0f)
        ? seq->reserved_definitions_library.gradient_raster_time
        : opts->grad_raster_us;
    desc->adc_raster_us = (seq->reserved_definitions_library.adc_raster_time > 0.0f)
        ? seq->reserved_definitions_library.adc_raster_time
        : opts->adc_raster_us;
    desc->block_raster_us = (seq->reserved_definitions_library.block_duration_raster > 0.0f)
        ? seq->reserved_definitions_library.block_duration_raster
        : opts->block_raster_us;

    /* per-subsequence flags */
    desc->enable_pmc = seq->reserved_definitions_library.enable_pmc;
    desc->num_gain_cal_readouts = seq->reserved_definitions_library.num_gain_cal_readouts;
    desc->enable_sar_burst_mode = seq->reserved_definitions_library.enable_sar_burst_mode;
    desc->vop_sar_ratio = seq->reserved_definitions_library.vop_sar_ratio;
    desc->vop_global_sar_ratio = seq->reserved_definitions_library.vop_global_sar_ratio;
    desc->vendor = opts->vendor;
    desc->profile = opts->profile;
    desc->label_column_map[0] = opts->label_column_map[0];
    desc->label_column_map[1] = opts->label_column_map[1];
    desc->label_column_map[2] = opts->label_column_map[2];
    {
        size_t ext_len = strlen(opts->cache_ext);
        if (ext_len >= sizeof(desc->cache_ext))
            ext_len = sizeof(desc->cache_ext) - 1;
        memcpy(desc->cache_ext, opts->cache_ext, ext_len);
        desc->cache_ext[ext_len] = '\0';
    }

    /* encoding-space definitions */
    memcpy(desc->fov, seq->reserved_definitions_library.fov, sizeof(desc->fov));
    memcpy(desc->matrix, seq->reserved_definitions_library.matrix, sizeof(desc->matrix));
    memcpy(desc->nav_fov, seq->reserved_definitions_library.nav_fov, sizeof(desc->nav_fov));
    memcpy(
        desc->nav_matrix,
        seq->reserved_definitions_library.nav_matrix,
        sizeof(desc->nav_matrix));

    /* verify system and sequence raster times are integer multiples */
    {
        int rc = check_raster_times(seq, opts);
        if (PULSEG_FAILED(rc))
            return rc;
    }

    /* ---- allocate temp arrays ---- */
    if (seq->rf_library_size > 0)
    {
        tmp_rf_defs = (pulseg_rf_definition *)PULSEG_ALLOC(
            seq->rf_library_size * sizeof(pulseg_rf_definition));
        tmp_rf_tab = (pulseg_rf_table_element *)PULSEG_ALLOC(
            seq->rf_library_size * sizeof(pulseg_rf_table_element));
        if (!tmp_rf_defs || !tmp_rf_tab)
            goto fail;
    }
    if (seq->grad_library_size > 0)
    {
        tmp_grad_defs = (pulseg_grad_definition *)PULSEG_ALLOC(
            seq->grad_library_size * sizeof(pulseg_grad_definition));
        tmp_grad_tab = (pulseg_grad_table_element *)PULSEG_ALLOC(
            seq->grad_library_size * sizeof(pulseg_grad_table_element));
        if (!tmp_grad_defs || !tmp_grad_tab)
            goto fail;
    }
    if (seq->adc_library_size > 0)
    {
        tmp_adc_defs = (pulseg_adc_definition *)PULSEG_ALLOC(
            seq->adc_library_size * sizeof(pulseg_adc_definition));
        tmp_adc_tab = (pulseg_adc_table_element *)PULSEG_ALLOC(
            seq->adc_library_size * sizeof(pulseg_adc_table_element));
        if (!tmp_adc_defs || !tmp_adc_tab)
            goto fail;
    }
    tmp_blk_defs = (pulseg_base_block *)PULSEG_ALLOC(num_blocks * sizeof(pulseg_base_block));
    tmp_blk_tab =
        (pulseg_block_table_element *)PULSEG_ALLOC(num_blocks * sizeof(pulseg_block_table_element));
    if (!tmp_blk_defs || !tmp_blk_tab)
        goto fail;

    desc->structure_only = opts->structure_only ? 1 : 0;

    /* ---- step 1: dedup event libraries ---- */
    if (seq->rf_library_size > 0)
    {
        num_unique_rf = deduplicate_rf_library(seq, tmp_rf_defs, tmp_rf_tab);
        if (num_unique_rf < 0)
        {
            result = num_unique_rf;
            goto fail;
        }
        desc->num_unique_rfs = num_unique_rf;
        desc->rf_table_size = seq->rf_library_size;
        /* Neutral RF stats (flip angle, amplitudes, area, duration, isodelay,
         * bandwidth, bands, b1sq, num_samples/instances) are always computed;
         * the four vendor-specific envelope stats (vendor_stat[4]) are filled
         * only if the caller wired opts.vendor_rf_stats_fn. */
        rc = compute_rf_stats(
            seq,
            tmp_rf_defs,
            num_unique_rf,
            tmp_rf_tab,
            seq->rf_library_size,
            opts);
        if (PULSEG_FAILED(rc))
        {
            result = rc;
            goto fail;
        }
    }
    if (seq->grad_library_size > 0)
    {
        num_unique_grad = deduplicate_grad_library(seq, tmp_grad_defs, tmp_grad_tab);
        if (num_unique_grad < 0)
        {
            result = num_unique_grad;
            goto fail;
        }
        desc->grad_table_size = seq->grad_library_size;

        desc->num_unique_grads = num_unique_grad;

        rc = record_grad_shape_ids(seq, tmp_grad_defs, tmp_grad_tab, num_unique_grad);
        if (PULSEG_FAILED(rc))
        {
            result = rc;
            goto fail;
        }

        rc = compute_grad_stats(
            seq, desc, tmp_grad_defs, num_unique_grad, tmp_grad_tab, seq->grad_library_size);
        if (PULSEG_FAILED(rc))
        {
            result = rc;
            goto fail;
        }
    }
    if (seq->adc_library_size > 0)
    {
        num_unique_adc = deduplicate_adc_library(seq, tmp_adc_defs, tmp_adc_tab);
        if (num_unique_adc < 0)
        {
            result = num_unique_adc;
            goto fail;
        }
        desc->num_unique_adcs = num_unique_adc;
        desc->adc_table_size = seq->adc_library_size;
    }

    /* ---- step 2: block definition matrix ---- */
    int_rows = (int(*)[BLOCK_DEF_COLS])PULSEG_ALLOC(num_blocks * sizeof(*int_rows));
    key_rows = (int(*)[BLOCK_KEY_COLS])PULSEG_ALLOC(num_blocks * sizeof(*key_rows));
    unique_defs = (int *)PULSEG_ALLOC(num_blocks * sizeof(int));
    event_table = (int *)PULSEG_ALLOC(num_blocks * sizeof(int));
    if (!int_rows || !key_rows || !unique_defs || !event_table)
        goto fail;

    for (n = 0; n < num_blocks; ++n)
    {
        if (!pulseq_get_raw_block_content_ids(seq, &raw, n, 1))
        {
            result = PULSEG_ERR_INVALID_ARGUMENT;
            goto fail;
        }
        /* A pure delay's length is a per-instance value -- an interpreter
         * sets how long it waits there at run time -- so it is left out of
         * the key and every pure delay is one definition. */
        int_rows[n][0] =
            (raw.rf < 0 && raw.gx < 0 && raw.gy < 0 && raw.gz < 0 && raw.adc < 0)
            ? 0
            : (raw.block_duration >= 0 ? raw.block_duration : 0);
        int_rows[n][1] = (raw.rf >= 0 && tmp_rf_tab) ? tmp_rf_tab[raw.rf].id : -1;
        int_rows[n][2] = (raw.gx >= 0 && tmp_grad_tab) ? tmp_grad_tab[raw.gx].id : -1;
        int_rows[n][3] = (raw.gy >= 0 && tmp_grad_tab) ? tmp_grad_tab[raw.gy].id : -1;
        int_rows[n][4] = (raw.gz >= 0 && tmp_grad_tab) ? tmp_grad_tab[raw.gz].id : -1;
        int_rows[n][5] = (raw.adc >= 0 && tmp_adc_tab) ? tmp_adc_tab[raw.adc].id : -1;

        tmp_blk_tab[n].rf_id = raw.rf;
        tmp_blk_tab[n].gx_id = raw.gx;
        tmp_blk_tab[n].gy_id = raw.gy;
        tmp_blk_tab[n].gz_id = raw.gz;
        tmp_blk_tab[n].adc_id = raw.adc;

        tmp_blk_tab[n].duration_us =
            (raw.rf < 0 && raw.gx < 0 && raw.gy < 0 && raw.gz < 0 && raw.adc < 0)
            ? (int)(raw.block_duration * desc->block_raster_us)
            : -1;

        tmp_blk_tab[n].rotation_id = seq->block_rotations[n];
        tmp_blk_tab[n].rf_shim_id = seq->block_shims[n];
        tmp_blk_tab[n].digitalout_id = pulseq_block_trigger(seq, &raw);
        tmp_blk_tab[n].norot_flag = seq->block_flags[n][0];
        tmp_blk_tab[n].nopos_flag = seq->block_flags[n][1];
        tmp_blk_tab[n].pmc_flag = seq->block_flags[n][2];
        tmp_blk_tab[n].nav_flag = seq->block_flags[n][3];
        /* TRID is sticky: the group in force, 0 before any. trid_set marks
         * the block that sets it, which is where a repetition starts: an
         * author re-SETs the same id at every one, so the sticky value alone
         * does not say where one ends and the next begins. Both live on the
         * per-occurrence block-table entry, never on the deduplicated block
         * definition, so they have no dedup footprint. */
        tmp_blk_tab[n].trid = seq->block_flags[n][4];
        tmp_blk_tab[n].trid_set = seq->block_trid_set[n];
    }

    /* step 3: the block definitions the cache holds */
    {
        int num_raw_defs, num_geometries, k, g, dense;

        /* pypulseqpp's block definition answers what a position plays, and is
         * deliberately blind to the digitiser: a position digitised two ways
         * still repeats every shot rather than every pair, which is what its
         * repetition is read off.  A pulse generator prepares the readout too,
         * so here a definition is that one AND the ADC definition, and the
         * published one is the geometry the two share. */
        for (n = 0; n < num_blocks; ++n)
        {
            key_rows[n][0] = seq->block_definitions[n];
            key_rows[n][1] = int_rows[n][5];
        }
        num_raw_defs = pulseg__deduplicate_int_rows(
            unique_defs, event_table, (const int *)key_rows, num_blocks, BLOCK_KEY_COLS);
        if (num_raw_defs <= 0)
            goto fail;
        desc->num_blocks = num_blocks;

        /* A non-acquiring instance of an otherwise identical block -- a dummy
         * shot -- keys to a definition of its own, which would then answer "no
         * ADC" to every structural question asked of it.  Fold it into the
         * acquiring definition it stands in for.  The grouping runs over the
         * definitions, not the blocks, so it costs nothing at scan length. */
        geometry_defs = (int *)PULSEG_ALLOC((size_t)num_raw_defs * sizeof(int));
        geometry_of = (int *)PULSEG_ALLOC((size_t)num_raw_defs * sizeof(int));
        def_map = (int *)PULSEG_ALLOC((size_t)num_raw_defs * sizeof(int));
        if (!geometry_defs || !geometry_of || !def_map)
            goto fail;

        /* The geometries are pypulseqpp's definitions, already dense and
         * counted from 0, so grouping by them needs no second pass. */
        num_geometries = 0;
        for (k = 0; k < num_raw_defs; ++k)
        {
            geometry_of[k] = key_rows[unique_defs[k]][0];
            if (geometry_of[k] >= num_geometries)
                num_geometries = geometry_of[k] + 1;
        }

        /* geometry_defs is reused as "an acquiring definition of this
         * geometry", -1 while none is known.  Where the geometry has several,
         * any of them serves: a non-acquiring instance carries adc_id -1 on
         * its own table entry, so the readout on the definition it folds into
         * is never read for it, and the segment it joins is decided later
         * from the readouts its repetition actually plays. */
        for (g = 0; g < num_geometries; ++g)
            geometry_defs[g] = -1;
        for (k = 0; k < num_raw_defs; ++k)
        {
            if (key_rows[unique_defs[k]][1] < 0)
                continue;
            g = geometry_of[k];
            if (geometry_defs[g] == -1)
                geometry_defs[g] = k;
        }

        for (k = 0; k < num_raw_defs; ++k)
        {
            def_map[k] = k;
            if (key_rows[unique_defs[k]][1] >= 0)
                continue;
            g = geometry_defs[geometry_of[k]];
            if (g >= 0)
                def_map[k] = g;
        }

        dense = 0;
        for (k = 0; k < num_raw_defs; ++k)
        {
            const int rep = unique_defs[k];
            if (def_map[k] != k)
                continue;
            tmp_blk_defs[dense].id = rep;
            /* A pure delay's definition carries no duration of its own, an
             * interpreter setting what it waits at run time; it takes the
             * length of the instance that introduced it, and the block table
             * carries what each instance waits. */
            tmp_blk_defs[dense].duration_us =
                (int_rows[rep][1] < 0 && int_rows[rep][2] < 0 && int_rows[rep][3] < 0 &&
                 int_rows[rep][4] < 0 && int_rows[rep][5] < 0)
                ? tmp_blk_tab[rep].duration_us
                : (int)(int_rows[rep][0] * desc->block_raster_us);
            tmp_blk_defs[dense].rf_id = int_rows[rep][1];
            tmp_blk_defs[dense].gx_id = int_rows[rep][2];
            tmp_blk_defs[dense].gy_id = int_rows[rep][3];
            tmp_blk_defs[dense].gz_id = int_rows[rep][4];
            tmp_blk_defs[dense].adc_id = int_rows[rep][5];
            geometry_of[k] = dense; /* reused as raw definition -> dense index */
            ++dense;
        }
        desc->num_unique_blocks = dense;

        for (n = 0; n < num_blocks; ++n)
            tmp_blk_tab[n].id = geometry_of[def_map[event_table[n]]];
    }

    PULSEG_FREE(geometry_defs);
    geometry_defs = NULL;
    PULSEG_FREE(geometry_of);
    geometry_of = NULL;
    PULSEG_FREE(def_map);
    def_map = NULL;
    PULSEG_FREE(key_rows);
    key_rows = NULL;
    PULSEG_FREE(int_rows);
    int_rows = NULL;
    PULSEG_FREE(unique_defs);
    unique_defs = NULL;
    PULSEG_FREE(event_table);
    event_table = NULL;

    /* ---- step 4: copy to output (exact sizes) ---- */
#define COPY_ARRAY(dst, src, cnt, type) \
    do \
    { \
        if ((cnt) > 0) \
        { \
            (dst) = (type *)PULSEG_ALLOC((cnt) * sizeof(type)); \
            if (!(dst)) \
            { \
                result = PULSEG_ERR_ALLOC_FAILED; \
                pulseg_sequence_descriptor_free(desc); \
                goto fail; \
            } \
            memcpy((dst), (src), (cnt) * sizeof(type)); \
        } \
    } while (0)

/* The four per-occurrence tables were allocated at exactly their final length
 * (one entry per library entry, one per block), so the descriptor takes the
 * temp buffer over rather than paying a second allocation and a full copy --
 * on a 2.1M-block scan the block table alone is a 126 MB memcpy, and holding
 * both copies at once is what set the peak. The definition arrays below are
 * over-allocated at library size and genuinely do shrink, so they are copied.
 * Adoption cannot fail, so it runs after every copy that can. */
#define ADOPT_ARRAY(dst, src) \
    do \
    { \
        (dst) = (src); \
        (src) = NULL; \
    } while (0)

    COPY_ARRAY(desc->rf_definitions, tmp_rf_defs, desc->num_unique_rfs, pulseg_rf_definition);
    COPY_ARRAY(
        desc->grad_definitions,
        tmp_grad_defs,
        desc->num_unique_grads,
        pulseg_grad_definition);
    COPY_ARRAY(desc->adc_definitions, tmp_adc_defs, desc->num_unique_adcs, pulseg_adc_definition);
    COPY_ARRAY(desc->base_blocks, tmp_blk_defs, desc->num_unique_blocks, pulseg_base_block);

    ADOPT_ARRAY(desc->rf_table, tmp_rf_tab);
    ADOPT_ARRAY(desc->grad_table, tmp_grad_tab);
    ADOPT_ARRAY(desc->adc_table, tmp_adc_tab);
    ADOPT_ARRAY(desc->block_table, tmp_blk_tab);

#undef ADOPT_ARRAY
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
    result = copy_shapes_library(seq, desc, adopt_shapes);
    if (PULSEG_FAILED(result))
    {
        pulseg_sequence_descriptor_free(desc);
        return result;
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
    if (key_rows)
        PULSEG_FREE(key_rows);
    if (geometry_defs)
        PULSEG_FREE(geometry_defs);
    if (geometry_of)
        PULSEG_FREE(geometry_of);
    if (def_map)
        PULSEG_FREE(def_map);
    if (unique_defs)
        PULSEG_FREE(unique_defs);
    if (event_table)
        PULSEG_FREE(event_table);
    return result;
}
