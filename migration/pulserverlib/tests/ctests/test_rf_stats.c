/*
 * test_rf_stats.c -- RF statistics and consistency tests.
 *
 * Suite A: RF180 block pulse ground truth (1 test).
 * Suite B: RF periodicity consistency checks (4 tests).
 */
#include "test_helpers.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================================================================== */
/*  Suite A — RF180 block pulse ground truth                          */
/* ================================================================== */

/*
 * Ground truth derived from the standard GE 1 ms hard-pulse
 * RF_PULSE struct, converted to SI / pulseqlib conventions:
 *
 *   abswidth        = 1.0       (normalized)
 *   effwidth        = 1.0       (normalized)
 *   dtycyc          = 1.0       (normalized)
 *   maxpw           = 1.0       (normalized)
 *   maxb1           = 0.1174 G  -> 500 Hz  (gamma * 1e-4 * 0.1174)
 *   nom_fa          = 180 deg   -> pi rad  (2pi * 500 * 0.001)
 *   nom_pw          = 1000 us   -> ~999 us (N-1 raster samples)
 *   isodelay        = 500 us    -> ~499 us (int truncation)
 *   area            = 1.0       -> 0.001 s (duration_s * 1.0)
 *   nom_bw          = 3125 Hz   -> ~3123 Hz (fallback 3.12/duration_s)
 *   num_samples     = 2 (raw decompressed: block pulse has only
 *                       start + end samples in .seq file)
 */

MU_TEST(test_rf180_block_pulse_stats)
{
    pulseqlib_opts opts;
    pulseqlib_collection* coll = NULL;
    pulseqlib_rf_stats stats = PULSEQLIB_RF_STATS_INIT;
    int rc;

    default_opts_init(&opts);
    rc = load_seq(&coll, "00_basic_rfstat.seq", &opts);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq failed");

    rc = pulseqlib_get_rf_stats(coll, &stats, 0, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "get_rf_stats failed");

    mu_assert_float_near("abs_width",  1.0f,    stats.abs_width,        1e-4f);
    mu_assert_float_near("eff_width",  1.0f,    stats.eff_width,        1e-4f);
    mu_assert_float_near("duty_cycle", 1.0f,    stats.duty_cycle,       1e-4f);
    mu_assert_float_near("max_pw",     1.0f,    stats.max_pulse_width,  1e-4f);

    mu_assert_float_near("base_amp_hz", 500.0f, stats.base_amplitude_hz, 1.0f);
    mu_assert_float_near("flip_angle",  (float)M_PI, stats.flip_angle_deg, 0.01f);

    mu_assert_float_near("duration_us", 999.0f, stats.duration_us, 2.0f);
    mu_assert(abs(stats.isodelay_us - 499) <= 2, "isodelay_us");

    mu_assert_float_near("area",        0.001f, stats.area,          1e-5f);
    mu_assert_float_near("bandwidth",   3123.0f, stats.bandwidth_hz, 50.0f);

    mu_assert_int_eq(2, stats.num_samples);

    pulseqlib_collection_free(coll);
}

MU_TEST_SUITE(suite_rf_stats)
{
    MU_RUN_TEST(test_rf180_block_pulse_stats);
}

/* ================================================================== */
/*  Suite B — RF consistency (periodicity) checks                     */
/* ================================================================== */

static pulseqlib_opts s_rf_opts;

static void rf_consistency_setup(void)
{
    default_opts_init(&s_rf_opts);
}

static void run_consistency_check(const char* filename, int expected_code)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic diag = PULSEQLIB_DIAGNOSTIC_INIT;
    int rc;

    rc = load_seq(&coll, filename, &s_rf_opts);
    if (PULSEQLIB_FAILED(rc)) {
        /* Load failed — only acceptable if we expected this error */
        if (expected_code > 0)
            mu_fail("load_seq failed unexpectedly");
        mu_assert_int_eq(expected_code, rc);
        return;
    }

    rc = pulseqlib_check_consistency(coll, &diag);

    if (expected_code > 0) {
        mu_assert(PULSEQLIB_SUCCEEDED(rc), "expected consistency pass");
    } else {
        mu_assert_int_eq(expected_code, rc);
    }

    pulseqlib_collection_free(coll);
}

MU_TEST(test_rf_periodic_ok)
{
    run_consistency_check("01_rfamp_ok_mrfingerprinting.seq",
                          PULSEQLIB_SUCCESS);
}

MU_TEST(test_rf_periodic_fail)
{
    run_consistency_check("02_rfamp_fail_vfa.seq",
                          PULSEQLIB_ERR_TR_PATTERN_MISMATCH);
}

MU_TEST(test_rfshim_periodic_ok)
{
    run_consistency_check("03_rfshim_ok_pnpmrfingerprinting.seq",
                          PULSEQLIB_SUCCESS);
}

MU_TEST(test_rfshim_periodic_fail)
{
    run_consistency_check("04_rfshim_fail_gre.seq",
                          PULSEQLIB_ERR_TR_PATTERN_MISMATCH);
}

MU_TEST_SUITE(suite_rf_consistency)
{
    MU_SUITE_CONFIGURE(rf_consistency_setup, NULL);
    MU_RUN_TEST(test_rf_periodic_ok);
    MU_RUN_TEST(test_rf_periodic_fail);
    MU_RUN_TEST(test_rfshim_periodic_ok);
    MU_RUN_TEST(test_rfshim_periodic_fail);
}

/* ================================================================== */
/*  Entry point                                                       */
/* ================================================================== */

int test_rf_stats_main(void)
{
    minunit_run = 0;
    minunit_fail = 0;
    minunit_assert = 0;
    minunit_status = 0;
    minunit_real_timer = 0;
    minunit_proc_timer = 0;

    MU_RUN_SUITE(suite_rf_stats);
    MU_RUN_SUITE(suite_rf_consistency);
    MU_REPORT();
    return MU_EXIT_CODE;
}
