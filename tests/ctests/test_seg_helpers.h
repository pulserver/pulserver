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

#endif /* TEST_SEG_HELPERS_H */
