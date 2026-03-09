/*
 * test_seg_helpers.h -- ground-truth file parsers for segmentation tests.
 *
 * Provides parse_meta() for the MATLAB-generated _meta.txt files.
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
/*  Meta struct — mirrors quantities from example_check.c step 6      */
/* ------------------------------------------------------------------ */

#define MAX_UNIQUE_ADCS 16

typedef struct seg_meta {
    int num_unique_adcs;
    int adc_samples[MAX_UNIQUE_ADCS];
    int adc_dwell_ns[MAX_UNIQUE_ADCS];
    int max_b1_subseq;
    int tr_duration_us;
} seg_meta;

#define SEG_META_INIT {0, {0}, {0}, 0, 0}

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
        }
    }

    fclose(f);
    *out = m;
    return 1;
}

#endif /* TEST_SEG_HELPERS_H */
