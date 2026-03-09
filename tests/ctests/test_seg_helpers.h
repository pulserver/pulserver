/*
 * test_seg_helpers.h -- ground-truth file parsers for segmentation tests.
 *
 * Provides parse_meta() for the MATLAB-generated _meta.txt files and
 * parse_tr_waveform() for the binary _tr_waveform.bin files.
 */
#ifndef TEST_SEG_HELPERS_H
#define TEST_SEG_HELPERS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __GNUC__
#define TSEG_MAYBE_UNUSED __attribute__((unused))
#else
#define TSEG_MAYBE_UNUSED
#endif

/* ------------------------------------------------------------------ */
/*  Meta struct — mirrors quantities from example_check.c steps 5-6   */
/* ------------------------------------------------------------------ */

#define MAX_UNIQUE_ADCS 16
#define MAX_SEGMENTS    16

typedef struct seg_meta {
    /* Phase 1: ADC / TR (step 6) */
    int num_unique_adcs;
    int adc_samples[MAX_UNIQUE_ADCS];
    int adc_dwell_ns[MAX_UNIQUE_ADCS];
    int max_b1_subseq;
    int tr_duration_us;
    /* Phase 2: Segment structure (step 5) */
    int num_segments;
    int segment_num_blocks[MAX_SEGMENTS];
    int num_canonical_trs;
} seg_meta;

#define SEG_META_INIT {0, {0}, {0}, 0, 0, 0, {0}, 0}

/* ------------------------------------------------------------------ */
/*  parse_meta                                                        */
/* ------------------------------------------------------------------ */

static TSEG_MAYBE_UNUSED int parse_meta(const char* path, seg_meta* out)
{
    FILE* f;
    char key[64];
    int val, idx;
    char suffix[32];
    seg_meta m = SEG_META_INIT;

    f = fopen(path, "r");
    if (!f) return 0;

    while (fscanf(f, "%63s %d", key, &val) == 2) {
        if (strcmp(key, "num_unique_adcs") == 0) {
            m.num_unique_adcs = val;
        } else if (sscanf(key, "adc_%d_%31s", &idx, suffix) == 2) {
            if (idx >= 0 && idx < MAX_UNIQUE_ADCS) {
                if (strcmp(suffix, "samples") == 0)
                    m.adc_samples[idx] = val;
                else if (strcmp(suffix, "dwell_ns") == 0)
                    m.adc_dwell_ns[idx] = val;
            }
        } else if (strcmp(key, "max_b1_subseq") == 0) {
            m.max_b1_subseq = val;
        } else if (strcmp(key, "tr_duration_us") == 0) {
            m.tr_duration_us = val;
        } else if (strcmp(key, "num_segments") == 0) {
            m.num_segments = val;
        } else if (sscanf(key, "segment_%d_%31s", &idx, suffix) == 2) {
            if (idx >= 0 && idx < MAX_SEGMENTS) {
                if (strcmp(suffix, "num_blocks") == 0)
                    m.segment_num_blocks[idx] = val;
            }
        } else if (strcmp(key, "num_canonical_trs") == 0) {
            m.num_canonical_trs = val;
        }
    }

    fclose(f);
    *out = m;
    return 1;
}

/* ------------------------------------------------------------------ */
/*  TR waveform struct + binary parser                                */
/* ------------------------------------------------------------------ */

typedef struct seg_tr_waveform {
    int    num_samples;
    float* time_us;
    float* gx;
    float* gy;
    float* gz;
} seg_tr_waveform;

#define SEG_TR_WAVEFORM_INIT {0, NULL, NULL, NULL, NULL}

static TSEG_MAYBE_UNUSED void free_tr_waveform(seg_tr_waveform* w)
{
    if (!w) return;
    free(w->time_us); w->time_us = NULL;
    free(w->gx);      w->gx = NULL;
    free(w->gy);      w->gy = NULL;
    free(w->gz);       w->gz = NULL;
    w->num_samples = 0;
}

/**
 * Parse binary TR waveform file.
 * Layout: int32 num_samples, then 4 contiguous float32 arrays:
 *   time_us[N], gx[N], gy[N], gz[N].
 * Returns 1 on success, 0 on failure.
 */
static TSEG_MAYBE_UNUSED int parse_tr_waveform(const char* path, seg_tr_waveform* out)
{
    FILE* f;
    int n;
    size_t ns;
    seg_tr_waveform w = SEG_TR_WAVEFORM_INIT;

    f = fopen(path, "rb");
    if (!f) return 0;

    if (fread(&n, sizeof(int), 1, f) != 1 || n <= 0) { fclose(f); return 0; }
    ns = (size_t)n;

    w.num_samples = n;
    w.time_us = (float*)malloc(ns * sizeof(float));
    w.gx      = (float*)malloc(ns * sizeof(float));
    w.gy      = (float*)malloc(ns * sizeof(float));
    w.gz      = (float*)malloc(ns * sizeof(float));
    if (!w.time_us || !w.gx || !w.gy || !w.gz) {
        free_tr_waveform(&w);
        fclose(f);
        return 0;
    }

    if (fread(w.time_us, sizeof(float), ns, f) != ns ||
        fread(w.gx,      sizeof(float), ns, f) != ns ||
        fread(w.gy,      sizeof(float), ns, f) != ns ||
        fread(w.gz,      sizeof(float), ns, f) != ns) {
        free_tr_waveform(&w);
        fclose(f);
        return 0;
    }

    fclose(f);
    *out = w;
    return 1;
}

/* ------------------------------------------------------------------ */
/*  Phase 3: Block-level ground truth (geninstruction tests)          */
/* ------------------------------------------------------------------ */

#define MAX_BLOCKS 8

/** Per-block timing + per-axis trapezoid corners. */
typedef struct block_meta {
    int num_blocks;

    /* Per-block timing */
    int duration_us[MAX_BLOCKS];
    int start_time_us[MAX_BLOCKS];

    /* RF (block 0) */
    int rf_delay_us;
    int rf_num_samples;
    int rf_is_complex;
    int rf_num_channels;

    /* Trap gradient corners (indexed by [block][axis 0=x,1=y,2=z]).
     * Only filled for blocks/axes that are actual trapezoids. */
    int   has_trap[MAX_BLOCKS][3];
    float trap_amplitude[MAX_BLOCKS][3];
    int   trap_rise_us[MAX_BLOCKS][3];
    int   trap_flat_us[MAX_BLOCKS][3];
    int   trap_fall_us[MAX_BLOCKS][3];
    int   trap_delay_us[MAX_BLOCKS][3];

    /* Arb gradient metadata (indexed by [block][axis]). */
    int has_arb[MAX_BLOCKS][3];
    int arb_num_samples[MAX_BLOCKS][3];
    int arb_delay_us[MAX_BLOCKS][3];

    /* ADC */
    int adc_delay_us;

    /* Segment gap */
    int rf_adc_gap_us;
} block_meta;

#define BLOCK_META_INIT { \
    0, {0}, {0}, \
    0, 0, 0, 0, \
    {{0}}, {{0}}, {{0}}, {{0}}, {{0}}, {{0}}, \
    {{0}}, {{0}}, {{0}}, \
    0, 0 \
}

static TSEG_MAYBE_UNUSED int parse_block_meta(const char* path, block_meta* out)
{
    FILE* f;
    char key[80];
    char val_str[80];
    block_meta m = BLOCK_META_INIT;
    int idx;
    char axis_ch, suffix[40];

    f = fopen(path, "r");
    if (!f) return 0;

    while (fscanf(f, "%79s %79s", key, val_str) == 2) {
        /* Per-block timing */
        if (sscanf(key, "block_%d_duration_us", &idx) == 1 && idx < MAX_BLOCKS) {
            m.duration_us[idx] = atoi(val_str);
            if (idx >= m.num_blocks) m.num_blocks = idx + 1;
        }
        else if (sscanf(key, "block_%d_start_time_us", &idx) == 1 && idx < MAX_BLOCKS) {
            m.start_time_us[idx] = atoi(val_str);
        }
        /* RF (block 0) */
        else if (strcmp(key, "block_0_rf_delay_us") == 0)     m.rf_delay_us     = atoi(val_str);
        else if (strcmp(key, "block_0_rf_num_samples") == 0)   m.rf_num_samples  = atoi(val_str);
        else if (strcmp(key, "block_0_rf_is_complex") == 0)    m.rf_is_complex   = atoi(val_str);
        else if (strcmp(key, "block_0_rf_num_channels") == 0)  m.rf_num_channels = atoi(val_str);
        /* Trap gradients: block_B_gA_suffix */
        else if (sscanf(key, "block_%d_g%c_%39s", &idx, &axis_ch, suffix) == 3
                 && idx < MAX_BLOCKS) {
            int ax = (axis_ch == 'x') ? 0 : (axis_ch == 'y') ? 1 : 2;
            if (strcmp(suffix, "amplitude_hz_m") == 0) {
                m.has_trap[idx][ax] = 1;
                m.trap_amplitude[idx][ax] = (float)atof(val_str);
            }
            else if (strcmp(suffix, "rise_us") == 0)  m.trap_rise_us[idx][ax]  = atoi(val_str);
            else if (strcmp(suffix, "flat_us") == 0)   m.trap_flat_us[idx][ax]  = atoi(val_str);
            else if (strcmp(suffix, "fall_us") == 0)   m.trap_fall_us[idx][ax]  = atoi(val_str);
            else if (strcmp(suffix, "delay_us") == 0)  m.trap_delay_us[idx][ax] = atoi(val_str);
            /* Arb keys */
            else if (strcmp(suffix, "is_arb") == 0)       m.has_arb[idx][ax]         = atoi(val_str);
            else if (strcmp(suffix, "num_samples") == 0)   m.arb_num_samples[idx][ax] = atoi(val_str);
        }
        /* Arb keys without axis in gradient pattern — use block_B_gA_KEY */
        /* ADC */
        else if (strcmp(key, "block_2_adc_delay_us") == 0) m.adc_delay_us = atoi(val_str);
        /* Segment gap */
        else if (strcmp(key, "rf_adc_gap_us") == 0) m.rf_adc_gap_us = atoi(val_str);
    }

    fclose(f);
    *out = m;
    return 1;
}

/* ------------------------------------------------------------------ */
/*  RF magnitude waveform (binary float32)                            */
/* ------------------------------------------------------------------ */

typedef struct rf_mag_waveform {
    int    num_samples;
    float* magnitude;
} rf_mag_waveform;

#define RF_MAG_WAVEFORM_INIT {0, NULL}

static TSEG_MAYBE_UNUSED void free_rf_mag(rf_mag_waveform* w)
{
    if (!w) return;
    free(w->magnitude); w->magnitude = NULL;
    w->num_samples = 0;
}

static TSEG_MAYBE_UNUSED int parse_rf_mag(const char* path, rf_mag_waveform* out)
{
    FILE* f;
    int n;
    size_t ns;
    rf_mag_waveform w = RF_MAG_WAVEFORM_INIT;

    f = fopen(path, "rb");
    if (!f) return 0;

    if (fread(&n, sizeof(int), 1, f) != 1 || n <= 0) { fclose(f); return 0; }
    ns = (size_t)n;

    w.num_samples = n;
    w.magnitude = (float*)malloc(ns * sizeof(float));
    if (!w.magnitude) { fclose(f); return 0; }

    if (fread(w.magnitude, sizeof(float), ns, f) != ns) {
        free_rf_mag(&w);
        fclose(f);
        return 0;
    }

    fclose(f);
    *out = w;
    return 1;
}

/* ------------------------------------------------------------------ */
/*  Arbitrary gradient waveform (binary float32)                      */
/* ------------------------------------------------------------------ */

typedef struct arb_grad_waveform {
    int    num_samples;
    float* amplitude;
    float* time_us;
} arb_grad_waveform;

#define ARB_GRAD_WAVEFORM_INIT {0, NULL, NULL}

static TSEG_MAYBE_UNUSED void free_arb_grad(arb_grad_waveform* w)
{
    if (!w) return;
    free(w->amplitude); w->amplitude = NULL;
    free(w->time_us);   w->time_us = NULL;
    w->num_samples = 0;
}

static TSEG_MAYBE_UNUSED int parse_arb_grad(const char* path, arb_grad_waveform* out)
{
    FILE* f;
    int n;
    size_t ns;
    arb_grad_waveform w = ARB_GRAD_WAVEFORM_INIT;

    f = fopen(path, "rb");
    if (!f) return 0;

    if (fread(&n, sizeof(int), 1, f) != 1 || n <= 0) { fclose(f); return 0; }
    ns = (size_t)n;

    w.num_samples = n;
    w.amplitude = (float*)malloc(ns * sizeof(float));
    w.time_us   = (float*)malloc(ns * sizeof(float));
    if (!w.amplitude || !w.time_us) {
        free_arb_grad(&w);
        fclose(f);
        return 0;
    }

    if (fread(w.amplitude, sizeof(float), ns, f) != ns ||
        fread(w.time_us,   sizeof(float), ns, f) != ns) {
        free_arb_grad(&w);
        fclose(f);
        return 0;
    }

    fclose(f);
    *out = w;
    return 1;
}

#endif /* TEST_SEG_HELPERS_H */
