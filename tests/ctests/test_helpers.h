/*
 * test_helpers.h -- shared helpers for the pulseqlib unit test suite.
 *
 * Provides:
 *   - Standard scanner-parameter constants
 *   - load_seq()  : read a .seq file into a pulseqlib_collection*
 *   - make_path() : build full path from TEST_ROOT_DIR + relative
 *   - Each test_*_main() prototype so test_runner.c can wire them
 */
#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

/* Suppress warnings for static helpers / minunit statics that may
 * be unused in stub-only translation units. */
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

#include "minunit.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "pulseqlib_methods.h"
#include "pulseqlib_internal.h"  /* tests need internal error codes */

/* ------------------------------------------------------------------ */
/*  Scanner constants (3T GE-style defaults)                          */
/* ------------------------------------------------------------------ */
#define TEST_GAMMA          42.577478518e6f   /* Hz / T               */
#define TEST_B0             3.0f              /* T                    */
#define TEST_MAX_GRAD       (40.0f * TEST_GAMMA * 1e-3f)  /* Hz/m    */
#define TEST_MAX_SLEW       (150.0f * TEST_GAMMA)         /* Hz/m/s  */
#define TEST_RF_RASTER      1.0f              /* us                  */
#define TEST_GRAD_RASTER    10.0f             /* us                  */
#define TEST_ADC_RASTER     0.1f              /* us                  */
#define TEST_BLOCK_RASTER   10.0f             /* us                  */

/* ------------------------------------------------------------------ */
/*  Path helper                                                       */
/* ------------------------------------------------------------------ */
static char test_path_buf[2048];

static const char* make_path(const char* rel) {
    (void)snprintf(test_path_buf, sizeof(test_path_buf),
                   "%s/%s", TEST_ROOT_DIR, rel);
    return test_path_buf;
}

/* ------------------------------------------------------------------ */
/*  Default opts initializer                                          */
/* ------------------------------------------------------------------ */
static void test_opts_init(pulseqlib_opts* opts) {
    pulseqlib_opts_init(opts,
        TEST_GAMMA, TEST_B0,
        TEST_MAX_GRAD, TEST_MAX_SLEW,
        TEST_RF_RASTER, TEST_GRAD_RASTER,
        TEST_ADC_RASTER, TEST_BLOCK_RASTER);
}

/* ------------------------------------------------------------------ */
/*  Load a .seq file -> collection (returns PULSEQLIB_SUCCESS on success)  */
/* ------------------------------------------------------------------ */
static int load_seq(const char* rel_path,
                    pulseqlib_collection** out,
                    pulseqlib_diagnostic* diag,
                    int parse_labels)
{
    pulseqlib_opts opts;
    test_opts_init(&opts);
    return pulseqlib_read(out, diag, make_path(rel_path),
                          &opts, 0, 0, parse_labels, 1);
}

/* ------------------------------------------------------------------ */
/*  Suite entry points (defined in each test_*.c)                     */
/* ------------------------------------------------------------------ */
int test_error_main(void);
int test_load_main(void);
int test_structure_main(void);
int test_safety_grad_main(void);
int test_safety_acoustic_main(void);
int test_safety_pns_main(void);
int test_consistency_main(void);
int test_waveforms_main(void);
int test_cursor_main(void);
int test_freq_mod_main(void);
int test_labels_main(void);
int test_rf_stats_main(void);
int test_segments_main(void);

#endif /* TEST_HELPERS_H */
