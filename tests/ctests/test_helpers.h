/*
 * test_helpers.h -- shared includes, macros, and helpers for the
 *                   pulseqlib C test suite.
 */
#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

/* ------------------------------------------------------------------ */
/*  Suppress unused-function warnings for static helpers that are     */
/*  not called in every translation unit.                             */
/* ------------------------------------------------------------------ */
#ifdef __GNUC__
#define TEST_MAYBE_UNUSED __attribute__((unused))
#else
#define TEST_MAYBE_UNUSED
#endif

/* ------------------------------------------------------------------ */
/*  Standard + minunit + pulseqlib headers                            */
/* ------------------------------------------------------------------ */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "minunit.h"

#include "pulseqlib_config.h"
#include "pulseqlib_types.h"
#include "pulseqlib_internal.h"
#include "pulseqlib_methods.h"

/* ------------------------------------------------------------------ */
/*  Path helpers -- TEST_ROOT_DIR is set by CMakeLists.txt            */
/* ------------------------------------------------------------------ */
#define TEST_DATA_DIR TEST_ROOT_DIR "/tests/utils/expected/"

/* ------------------------------------------------------------------ */
/*  Physical constants                                                */
/* ------------------------------------------------------------------ */
#define GAMMA_HZ_PER_T 42577478.0f

/* ------------------------------------------------------------------ */
/*  Custom assertion: float near with absolute tolerance              */
/* ------------------------------------------------------------------ */
#define mu_assert_float_near(msg, expected, actual, tol) MU__SAFE_BLOCK(\
    float mu__exp = (expected);\
    float mu__act = (actual);\
    float mu__tol = (tol);\
    minunit_assert++;\
    if ((float)fabs((double)(mu__exp - mu__act)) > mu__tol) {\
        (void)snprintf(minunit_last_message, MINUNIT_MESSAGE_LEN,\
            "%s failed:\n\t%s:%d: %s\n\texpected %.8g  got %.8g  (tol %.8g)",\
            __func__, __FILE__, __LINE__, (msg),\
            (double)mu__exp, (double)mu__act, (double)mu__tol);\
        minunit_status = 1;\
        return;\
    } else {\
        printf(".");\
    }\
)

/* ------------------------------------------------------------------ */
/*  Opts initialisers                                                 */
/* ------------------------------------------------------------------ */

/**
 * Default opts matching MATLAB Pulseq default rasters
 * (grad tests, RF tests, ONCE tests).
 *
 *   rf_raster   = 1.0 us
 *   grad_raster = 10.0 us
 *   adc_raster  = 0.1 us
 *   block_raster= 10.0 us
 *   max_grad    = gamma * 40 mT/m
 *   max_slew    = gamma * 170 T/m/s
 */
static TEST_MAYBE_UNUSED void default_opts_init(pulseqlib_opts* opts)
{
    pulseqlib_opts_init(opts,
        GAMMA_HZ_PER_T,
        3.0f,
        GAMMA_HZ_PER_T * 0.040f,       /* 40 mT/m  -> Hz/m  */
        GAMMA_HZ_PER_T * 170.0f,       /* 170 T/m/s -> Hz/m/s */
        1.0f, 10.0f, 0.1f, 10.0f);
}

/**
 * Opts for the segmentation-generator sequences (GRE, bSSFP, …).
 *
 *   rf_raster   = 2.0 us
 *   grad_raster = 20.0 us
 *   adc_raster  = 2.0 us
 *   block_raster= 20.0 us
 *   max_grad    = gamma * 28 mT/m
 *   max_slew    = gamma * 150 T/m/s
 */
static TEST_MAYBE_UNUSED void gre_opts_init(pulseqlib_opts* opts)
{
    pulseqlib_opts_init(opts,
        GAMMA_HZ_PER_T,
        3.0f,
        GAMMA_HZ_PER_T * 0.028f,       /* 28 mT/m  -> Hz/m  */
        GAMMA_HZ_PER_T * 150.0f,       /* 150 T/m/s -> Hz/m/s */
        2.0f, 20.0f, 2.0f, 20.0f);
}

/* ------------------------------------------------------------------ */
/*  Sequence loader                                                   */
/* ------------------------------------------------------------------ */

/**
 * Load a .seq file from TEST_DATA_DIR.
 *
 * Returns the pulseqlib return code.  On success *coll is set; on
 * failure *coll is NULL.
 */
static TEST_MAYBE_UNUSED int load_seq(
    pulseqlib_collection** coll,
    const char* filename,
    const pulseqlib_opts* opts)
{
    pulseqlib_diagnostic diag = PULSEQLIB_DIAGNOSTIC_INIT;
    char path[512];

    (void)snprintf(path, sizeof(path), "%s%s", TEST_DATA_DIR, filename);
    return pulseqlib_read(coll, &diag, path, opts,
                          0,   /* cache_binary     */
                          0,   /* verify_signature */
                          0,   /* parse_labels     */
                          1);  /* num_averages     */
}

/**
 * Load a .seq file from TEST_DATA_DIR with a custom num_averages.
 *
 * Identical to load_seq but accepts num_averages as a parameter.
 */
static TEST_MAYBE_UNUSED int load_seq_with_averages(
    pulseqlib_collection** coll,
    const char* filename,
    const pulseqlib_opts* opts,
    int num_averages)
{
    pulseqlib_diagnostic diag = PULSEQLIB_DIAGNOSTIC_INIT;
    char path[512];

    (void)snprintf(path, sizeof(path), "%s%s", TEST_DATA_DIR, filename);
    return pulseqlib_read(coll, &diag, path, opts,
                          0,   /* cache_binary     */
                          0,   /* verify_signature */
                          0,   /* parse_labels     */
                          num_averages);
}

/**
 * Load a .seq file from TEST_DATA_DIR with signature verification enabled.
 *
 * Identical to load_seq_with_averages(num_averages=1) but passes
 * verify_signature=1, so an MD5 mismatch returns
 * PULSEQLIB_ERR_SIGNATURE_MISMATCH instead of loading successfully.
 */
static TEST_MAYBE_UNUSED int load_seq_with_signature_check(
    pulseqlib_collection** coll,
    const char* filename,
    const pulseqlib_opts* opts)
{
    pulseqlib_diagnostic diag = PULSEQLIB_DIAGNOSTIC_INIT;
    char path[512];

    (void)snprintf(path, sizeof(path), "%s%s", TEST_DATA_DIR, filename);
    return pulseqlib_read(coll, &diag, path, opts,
                          0,   /* cache_binary     */
                          1,   /* verify_signature — enabled */
                          0,   /* parse_labels     */
                          1);  /* num_averages     */
}

/* ------------------------------------------------------------------ */
/*  Forward declarations for suite entry points                       */
/* ------------------------------------------------------------------ */

int test_safety_grad_main(void);
int test_rf_stats_main(void);
int test_sequences_main(void);
int test_io_main(void);

#endif /* TEST_HELPERS_H */
