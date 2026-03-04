/*
 * test_seg_helpers.h -- ground-truth file parsers for segmentation tests.
 *
 * Provides functions to parse the MATLAB-generated ground-truth files:
 *   _meta.txt, _segments.txt, _scan_table.csv, _tr_*_{gx,gy,gz}.csv,
 *   _tr_*_anchors.txt
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
/*  Meta struct                                                       */
/* ------------------------------------------------------------------ */

typedef struct seg_meta {
    int num_blocks;
    int num_averages;
    int num_adcs;
    int num_prep_blocks;
    int num_cool_blocks;
    int degenerate_prep;
    int degenerate_cool;
    int tr_size;
    int num_segments;
    int num_passes;
    int total_duration_us;
} seg_meta;

#define SEG_META_INIT {0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0}

/* ------------------------------------------------------------------ */
/*  Anchor struct                                                     */
/* ------------------------------------------------------------------ */

#define MAX_ANCHORS 64

typedef struct seg_anchors {
    int   num_rf_isocenters;
    float rf_isocenter_us[MAX_ANCHORS];
    int   num_rf_refocus_isocenters;
    float rf_refocus_isocenter_us[MAX_ANCHORS];
    int   num_adc_kzero;
    float adc_kzero_us[MAX_ANCHORS];
} seg_anchors;

#define SEG_ANCHORS_INIT {0, {0}, 0, {0}, 0, {0}}

/* ------------------------------------------------------------------ */
/*  Waveform data                                                     */
/* ------------------------------------------------------------------ */

#define MAX_WAVEFORM_SAMPLES 16384

typedef struct seg_waveform {
    int   num_samples;
    float time_us[MAX_WAVEFORM_SAMPLES];
    float amplitude[MAX_WAVEFORM_SAMPLES];
} seg_waveform;

/* ------------------------------------------------------------------ */
/*  Scan table data                                                   */
/* ------------------------------------------------------------------ */

#define MAX_SCAN_TABLE 65536

typedef struct seg_scan_table {
    int count;
    int scan_pos[MAX_SCAN_TABLE];
    int block_idx[MAX_SCAN_TABLE];
} seg_scan_table;

/* ------------------------------------------------------------------ */
/*  Segment IDs data                                                  */
/* ------------------------------------------------------------------ */

#define MAX_SEG_IDS 256

typedef struct seg_ids {
    int count;
    int ids[MAX_SEG_IDS];
} seg_ids;

/* ------------------------------------------------------------------ */
/*  parse_meta                                                        */
/* ------------------------------------------------------------------ */

static TSEG_MAYBE_UNUSED int parse_meta(const char* path, seg_meta* out)
{
    FILE* f;
    char key[64];
    int val;
    seg_meta m = SEG_META_INIT;

    f = fopen(path, "r");
    if (!f) return 0;

    while (fscanf(f, "%63s %d", key, &val) == 2) {
        if      (strcmp(key, "num_blocks") == 0)       m.num_blocks = val;
        else if (strcmp(key, "num_averages") == 0)     m.num_averages = val;
        else if (strcmp(key, "num_adcs") == 0)         m.num_adcs = val;
        else if (strcmp(key, "num_prep_blocks") == 0)  m.num_prep_blocks = val;
        else if (strcmp(key, "num_cool_blocks") == 0)  m.num_cool_blocks = val;
        else if (strcmp(key, "degenerate_prep") == 0)  m.degenerate_prep = val;
        else if (strcmp(key, "degenerate_cool") == 0)  m.degenerate_cool = val;
        else if (strcmp(key, "tr_size") == 0)          m.tr_size = val;
        else if (strcmp(key, "num_segments") == 0)     m.num_segments = val;
        else if (strcmp(key, "num_passes") == 0)       m.num_passes = val;
        else if (strcmp(key, "total_duration_us") == 0) m.total_duration_us = val;
    }

    fclose(f);
    *out = m;
    return 1;
}

/* ------------------------------------------------------------------ */
/*  parse_segments — reads the Nth (0-based) line of _segments.txt    */
/* ------------------------------------------------------------------ */

static TSEG_MAYBE_UNUSED int parse_segments(const char* path, int seg_idx,
                                            seg_ids* out)
{
    FILE* f;
    char line[4096];
    int cur_line = 0;
    int id;
    char* tok;

    out->count = 0;
    f = fopen(path, "r");
    if (!f) return 0;

    while (fgets(line, (int)sizeof(line), f)) {
        if (cur_line == seg_idx) {
            tok = strtok(line, " \t\r\n");
            while (tok && out->count < MAX_SEG_IDS) {
                id = atoi(tok);
                out->ids[out->count++] = id;
                tok = strtok(NULL, " \t\r\n");
            }
            fclose(f);
            return 1;
        }
        cur_line++;
    }

    fclose(f);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  parse_scan_table                                                  */
/* ------------------------------------------------------------------ */

static TSEG_MAYBE_UNUSED int parse_scan_table(const char* path,
                                              seg_scan_table* out)
{
    FILE* f;
    char line[256];
    int sp, bi;

    out->count = 0;
    f = fopen(path, "r");
    if (!f) return 0;

    /* skip header */
    if (!fgets(line, (int)sizeof(line), f)) { fclose(f); return 0; }

    while (fgets(line, (int)sizeof(line), f)) {
        if (sscanf(line, "%d,%d", &sp, &bi) == 2) {
            if (out->count < MAX_SCAN_TABLE) {
                out->scan_pos[out->count] = sp;
                out->block_idx[out->count] = bi;
                out->count++;
            }
        }
    }

    fclose(f);
    return 1;
}

/* ------------------------------------------------------------------ */
/*  parse_waveform_csv                                                */
/* ------------------------------------------------------------------ */

static TSEG_MAYBE_UNUSED int parse_waveform_csv(const char* path,
                                                seg_waveform* out)
{
    FILE* f;
    char line[256];
    float t, a;

    out->num_samples = 0;
    f = fopen(path, "r");
    if (!f) return 0;

    /* skip header */
    if (!fgets(line, (int)sizeof(line), f)) { fclose(f); return 0; }

    while (fgets(line, (int)sizeof(line), f)) {
        if (sscanf(line, "%f,%f", &t, &a) == 2) {
            if (out->num_samples < MAX_WAVEFORM_SAMPLES) {
                out->time_us[out->num_samples] = t;
                out->amplitude[out->num_samples] = a;
                out->num_samples++;
            }
        }
    }

    fclose(f);
    return 1;
}

/* ------------------------------------------------------------------ */
/*  parse_anchors                                                     */
/* ------------------------------------------------------------------ */

static TSEG_MAYBE_UNUSED int parse_anchors(const char* path,
                                           seg_anchors* out)
{
    FILE* f;
    char key[64];
    float val;
    seg_anchors a = SEG_ANCHORS_INIT;

    f = fopen(path, "r");
    if (!f) return 0;

    while (fscanf(f, "%63s %f", key, &val) == 2) {
        if (strcmp(key, "rf_isocenter_us") == 0) {
            if (a.num_rf_isocenters < MAX_ANCHORS)
                a.rf_isocenter_us[a.num_rf_isocenters++] = val;
        } else if (strcmp(key, "rf_refocus_isocenter_us") == 0) {
            if (a.num_rf_refocus_isocenters < MAX_ANCHORS)
                a.rf_refocus_isocenter_us[a.num_rf_refocus_isocenters++] = val;
        } else if (strcmp(key, "adc_kzero_us") == 0) {
            if (a.num_adc_kzero < MAX_ANCHORS)
                a.adc_kzero_us[a.num_adc_kzero++] = val;
        }
    }

    fclose(f);
    *out = a;
    return 1;
}

#endif /* TEST_SEG_HELPERS_H */
