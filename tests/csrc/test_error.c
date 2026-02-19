/*
 * test_error.c -- error/diagnostic/opts helpers (fully self-contained).
 *
 * These tests exercise the public API functions that do NOT require
 * a loaded collection:
 *   - pulseqlib_get_error_message / pulseqlib_get_error_hint
 *   - pulseqlib_format_error
 *   - pulseqlib_diagnostic_init
 *   - pulseqlib_opts_init
 *   - INIT macros for public structs
 *   - PULSEQLIB_SUCCEEDED / PULSEQLIB_FAILED macros
 *   - NULL-pointer rejection by main entry points
 */
#include "test_helpers.h"

/* ================================================================== */
/*  Error message / hint coverage                                     */
/* ================================================================== */

MU_TEST(test_error_message_ok)
{
    const char* msg = pulseqlib_get_error_message(PULSEQLIB_OK);
    mu_assert(msg != NULL, "OK message should be non-NULL");
    mu_assert(strlen(msg) > 0, "OK message should be non-empty");
}

MU_TEST(test_error_message_known_codes)
{
    /* Spot-check a handful of error codes */
    static const int codes[] = {
        PULSEQLIB_ERR_NULL_POINTER,
        PULSEQLIB_ERR_FILE_NOT_FOUND,
        PULSEQLIB_ERR_PARSE_FAILED,
        PULSEQLIB_ERR_MAX_GRAD_EXCEEDED,
        PULSEQLIB_ERR_MAX_SLEW_EXCEEDED,
        PULSEQLIB_ERR_CONSISTENCY_RF_PERIODIC,
        PULSEQLIB_ERR_CONSISTENCY_RF_SHIM_PERIODIC,
        PULSEQLIB_ERR_PNS_THRESHOLD_EXCEEDED,
        PULSEQLIB_ERR_ACOUSTIC_VIOLATION,
        PULSEQLIB_ERR_NOT_IMPLEMENTED
    };
    int i;
    for (i = 0; i < (int)(sizeof(codes) / sizeof(codes[0])); i++) {
        const char* msg = pulseqlib_get_error_message(codes[i]);
        mu_assert(msg != NULL, "message should be non-NULL");
        mu_assert(strlen(msg) > 0, "message should be non-empty");
    }
}

MU_TEST(test_error_hint_known_codes)
{
    static const int codes[] = {
        PULSEQLIB_ERR_MAX_GRAD_EXCEEDED,
        PULSEQLIB_ERR_MAX_SLEW_EXCEEDED,
        PULSEQLIB_ERR_CONSISTENCY_RF_PERIODIC,
        PULSEQLIB_ERR_CONSISTENCY_RF_SHIM_PERIODIC,
        PULSEQLIB_ERR_PNS_THRESHOLD_EXCEEDED
    };
    int i;
    for (i = 0; i < (int)(sizeof(codes) / sizeof(codes[0])); i++) {
        const char* hint = pulseqlib_get_error_hint(codes[i]);
        mu_assert(hint != NULL, "hint should be non-NULL");
        mu_assert(strlen(hint) > 0, "hint should be non-empty");
    }
}

MU_TEST(test_error_message_unknown_code)
{
    const char* msg = pulseqlib_get_error_message(-9999);
    mu_assert(msg != NULL, "unknown code should still return a message");
}

/* ================================================================== */
/*  SUCCEEDED / FAILED macros                                         */
/* ================================================================== */

MU_TEST(test_succeeded_failed_macros)
{
    mu_assert(PULSEQLIB_SUCCEEDED(PULSEQLIB_OK),
              "OK should be SUCCEEDED");
    mu_assert(!PULSEQLIB_FAILED(PULSEQLIB_OK),
              "OK should not be FAILED");

    mu_assert(PULSEQLIB_FAILED(PULSEQLIB_ERR_NULL_POINTER),
              "negative code should be FAILED");
    mu_assert(!PULSEQLIB_SUCCEEDED(PULSEQLIB_ERR_NULL_POINTER),
              "negative code should not be SUCCEEDED");

    mu_assert(PULSEQLIB_FAILED(PULSEQLIB_ERR_NOT_IMPLEMENTED),
              "-999 should be FAILED");
}

/* ================================================================== */
/*  Diagnostic init                                                   */
/* ================================================================== */

MU_TEST(test_diagnostic_init)
{
    pulseqlib_diagnostic diag;
    pulseqlib_diagnostic_init(&diag);
    mu_assert(diag.message[0] == '\0',
              "diagnostic message should be empty after init");
}

/* ================================================================== */
/*  format_error                                                      */
/* ================================================================== */

MU_TEST(test_format_error_basic)
{
    char buf[512];
    int n;
    pulseqlib_diagnostic diag;

    pulseqlib_diagnostic_init(&diag);
    n = pulseqlib_format_error(buf, sizeof(buf),
                               PULSEQLIB_ERR_NULL_POINTER, &diag);
    mu_assert(n > 0, "format_error should return > 0 chars");
    mu_assert(strlen(buf) > 0, "formatted string should be non-empty");
}

MU_TEST(test_format_error_null_diag)
{
    char buf[512];
    int n;

    n = pulseqlib_format_error(buf, sizeof(buf),
                               PULSEQLIB_ERR_FILE_NOT_FOUND, NULL);
    mu_assert(n > 0, "format_error with NULL diag should work");
    mu_assert(strlen(buf) > 0, "formatted string should be non-empty");
}

MU_TEST(test_format_error_tiny_buffer)
{
    char buf[8];
    int n;

    n = pulseqlib_format_error(buf, sizeof(buf),
                               PULSEQLIB_ERR_NULL_POINTER, NULL);
    /* Should not crash; output truncated; NUL-terminated */
    mu_assert(buf[sizeof(buf) - 1] == '\0',
              "tiny buffer should still be NUL-terminated");
    (void)n;
}

/* ================================================================== */
/*  opts_init                                                         */
/* ================================================================== */

MU_TEST(test_opts_init_values)
{
    pulseqlib_opts opts;
    test_opts_init(&opts);
    mu_assert((float)fabs(opts.gamma_hz_per_t - TEST_GAMMA) < 1.0f,
              "gamma should match");
    mu_assert((float)fabs(opts.b0_t - TEST_B0) < 0.01f,
              "B0 should match");
    mu_assert(opts.max_grad_hz_per_m > 0.0f,
              "max_grad should be positive");
    mu_assert(opts.max_slew_hz_per_m_per_s > 0.0f,
              "max_slew should be positive");
    mu_assert((float)fabs(opts.rf_raster_us - TEST_RF_RASTER) < 0.01f,
              "rf_raster should match");
    mu_assert((float)fabs(opts.grad_raster_us - TEST_GRAD_RASTER) < 0.01f,
              "grad_raster should match");
}

/* ================================================================== */
/*  INIT macros produce sane defaults                                 */
/* ================================================================== */

MU_TEST(test_block_instance_init)
{
    pulseqlib_block_instance inst = PULSEQLIB_BLOCK_INSTANCE_INIT;
    mu_assert_int_eq(0, inst.duration_us);
    mu_assert((float)fabs(inst.rf_amp_hz) < 1e-12f,
              "rf_amp_hz should be zero");
    mu_assert((float)fabs(inst.gx_amp_hz_per_m) < 1e-12f,
              "gx should be zero");
    mu_assert_int_eq(0, inst.adc_flag);
    mu_assert_int_eq(-1, inst.rf_shim_id);

    /* rotation matrix = identity */
    mu_assert((float)fabs(inst.rotmat[0] - 1.0f) < 1e-6f,
              "rotmat[0,0] should be 1");
    mu_assert((float)fabs(inst.rotmat[4] - 1.0f) < 1e-6f,
              "rotmat[1,1] should be 1");
    mu_assert((float)fabs(inst.rotmat[8] - 1.0f) < 1e-6f,
              "rotmat[2,2] should be 1");
    mu_assert((float)fabs(inst.rotmat[1]) < 1e-6f,
              "rotmat[0,1] should be 0");
}

MU_TEST(test_diagnostic_init_macro)
{
    pulseqlib_diagnostic diag = PULSEQLIB_DIAGNOSTIC_INIT;
    mu_assert(diag.message[0] == '\0',
              "DIAGNOSTIC_INIT should zero message");
}

MU_TEST(test_rf_stats_init)
{
    pulseqlib_rf_stats s = PULSEQLIB_RF_STATS_INIT;
    mu_assert((float)fabs(s.flip_angle_deg) < 1e-12f,
              "flip_angle should be zero");
    mu_assert_int_eq(0, s.num_samples);
}

MU_TEST(test_scan_time_info_init)
{
    pulseqlib_scan_time_info info = PULSEQLIB_SCAN_TIME_INFO_INIT;
    mu_assert((float)fabs(info.total_duration_us) < 1e-12f,
              "total_duration should be zero");
    mu_assert_int_eq(0, info.total_segment_boundaries);
}

/* ================================================================== */
/*  NULL-pointer rejection for key entry points                       */
/* ================================================================== */

MU_TEST(test_read_null_out)
{
    pulseqlib_opts opts;
    int rc;
    test_opts_init(&opts);
    rc = pulseqlib_read(NULL, NULL, "dummy.seq", &opts, 0, 0, 0);
    mu_assert(PULSEQLIB_FAILED(rc),
              "NULL out_coll should return error");
}

MU_TEST(test_collection_free_null)
{
    /* Should not crash */
    pulseqlib_collection_free(NULL);
}

MU_TEST(test_check_consistency_null)
{
    int rc = pulseqlib_check_consistency(NULL, NULL);
    mu_assert(PULSEQLIB_FAILED(rc),
              "NULL coll should return error");
}

MU_TEST(test_getters_null_coll)
{
    /* Getters with NULL coll should return error / -1 / 0 */
    mu_assert(pulseqlib_get_num_subsequences(NULL) <= 0,
              "NULL coll num_subseq");
    mu_assert(pulseqlib_get_num_segments(NULL) <= 0,
              "NULL coll num_segments");
    mu_assert(pulseqlib_get_max_adc_samples(NULL) <= 0,
              "NULL coll max_adc");
    mu_assert(pulseqlib_block_has_rf(NULL, 0, 0) <= 0,
              "NULL coll has_rf");
    mu_assert(pulseqlib_block_has_grad(NULL, 0, 0, 0) <= 0,
              "NULL coll has_grad");
    mu_assert(pulseqlib_get_rf_num_samples(NULL, 0, 0) < 0,
              "NULL coll rf_num_samples");
    mu_assert(pulseqlib_get_rf_num_channels(NULL, 0, 0) < 0,
              "NULL coll rf_num_channels");
}

MU_TEST(test_cursor_null_coll)
{
    int rc;
    pulseqlib_block_instance inst = PULSEQLIB_BLOCK_INSTANCE_INIT;
    rc = pulseqlib_get_block_instance(NULL, &inst);
    mu_assert(PULSEQLIB_FAILED(rc),
              "NULL coll get_block_instance should fail");
}

/* ================================================================== */
/*  Suite                                                             */
/* ================================================================== */

MU_TEST_SUITE(test_error_suite)
{
    /* Error messages */
    MU_RUN_TEST(test_error_message_ok);
    MU_RUN_TEST(test_error_message_known_codes);
    MU_RUN_TEST(test_error_hint_known_codes);
    MU_RUN_TEST(test_error_message_unknown_code);

    /* Macros */
    MU_RUN_TEST(test_succeeded_failed_macros);

    /* Diagnostic */
    MU_RUN_TEST(test_diagnostic_init);

    /* format_error */
    MU_RUN_TEST(test_format_error_basic);
    MU_RUN_TEST(test_format_error_null_diag);
    MU_RUN_TEST(test_format_error_tiny_buffer);

    /* opts */
    MU_RUN_TEST(test_opts_init_values);

    /* INIT macros */
    MU_RUN_TEST(test_block_instance_init);
    MU_RUN_TEST(test_diagnostic_init_macro);
    MU_RUN_TEST(test_rf_stats_init);
    MU_RUN_TEST(test_scan_time_info_init);

    /* NULL-pointer rejection */
    MU_RUN_TEST(test_read_null_out);
    MU_RUN_TEST(test_collection_free_null);
    MU_RUN_TEST(test_check_consistency_null);
    MU_RUN_TEST(test_getters_null_coll);
    MU_RUN_TEST(test_cursor_null_coll);
}

int test_error_main(void)
{
    MU_RUN_SUITE(test_error_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
