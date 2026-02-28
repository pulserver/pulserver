/*
 * test_rf_stats.c -- RF statistics tests.
 *
 * Tests:
 *   1. get_rf_stats returns plausible values for seq1 (which has RF).
 *   2. get_rf_array returns an array of RF pulses for the main TR region.
 *
 * Requires: data/seq1.seq
 */
#include "test_helpers.h"

/* ------------------------------------------------------------------ */
/*  Smoke: RF stats on seq1                                           */
/* ------------------------------------------------------------------ */

MU_TEST(test_rf_stats_seq1_smoke)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_rf_stats    stats = PULSEQLIB_RF_STATS_INIT;
    int rc;

    rc = load_seq("data/seq1.seq", &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load seq1");

    /* seq1 has an RF event in block 1 (sinc pulse) */
    rc = pulseqlib_get_rf_stats(coll, &stats, 0, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "get_rf_stats OK");

    mu_assert(stats.flip_angle_deg > 0.0f, "flip > 0");
    mu_assert(stats.duration_us > 0.0f, "duration > 0");
    mu_assert(stats.base_amplitude_hz > 0.0f, "base_amplitude > 0");
    mu_assert(stats.num_samples > 0, "num_samples > 0");

    pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  RF array for main TR region                                       */
/* ------------------------------------------------------------------ */

MU_TEST(test_rf_array_seq1_main)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_rf_stats* pulses = NULL;
    int rc, npulses;

    rc = load_seq("data/seq1.seq", &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load seq1");

    npulses = pulseqlib_get_rf_array(
        coll, &pulses, 0, PULSEQLIB_TR_REGION_MAIN);
    mu_assert(npulses >= 0, "get_rf_array should return >= 0");

    if (npulses > 0) {
        mu_assert(pulses != NULL, "pulse array should be non-NULL");
        mu_assert(pulses[0].flip_angle_deg > 0.0f,
                  "first pulse should have positive flip angle");
    }

    free(pulses);
    pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Suite                                                             */
/* ------------------------------------------------------------------ */

MU_TEST_SUITE(test_rf_stats_suite)
{
    MU_RUN_TEST(test_rf_stats_seq1_smoke);
    MU_RUN_TEST(test_rf_array_seq1_main);
}

int test_rf_stats_main(void)
{
    MU_RUN_SUITE(test_rf_stats_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
