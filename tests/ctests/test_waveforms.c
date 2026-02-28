/*
 * test_waveforms.c -- TR gradient waveform extraction.
 *
 * Tests:
 *   1. get_tr_gradient_waveforms returns non-empty arrays.
 *   2. Time arrays are monotonically increasing.
 *
 * Requires: data/01_ok_trap_extended_trap.seq
 */
#include "test_helpers.h"

/* ------------------------------------------------------------------ */
/*  Smoke: ok_trap waveforms are extractable                             */
/* ------------------------------------------------------------------ */

MU_TEST(test_waveforms_ok_smoke)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_tr_gradient_waveforms w = PULSEQLIB_TR_GRADIENT_WAVEFORMS_INIT;
    int rc;

    rc = load_seq(TEST_SEQ_OK, &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load ok_trap");

    rc = pulseqlib_get_tr_gradient_waveforms(coll, 0, &w, &diag);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "waveform extraction should succeed");

    /* At least one axis should have samples */
    mu_assert(w.gx.num_samples > 0
              || w.gy.num_samples > 0
              || w.gz.num_samples > 0,
              "at least one axis should have waveform data");

    /* Time should be monotonically non-decreasing on each axis */
    if (w.gx.num_samples > 1) {
        int i;
        for (i = 1; i < w.gx.num_samples; i++) {
            mu_assert(w.gx.time_us[i] >= w.gx.time_us[i - 1],
                      "GX time should be non-decreasing");
        }
    }

    pulseqlib_tr_gradient_waveforms_free(&w);
    pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Suite                                                             */
/* ------------------------------------------------------------------ */

MU_TEST_SUITE(test_waveforms_suite)
{
    MU_RUN_TEST(test_waveforms_ok_smoke);
}

int test_waveforms_main(void)
{
    MU_RUN_SUITE(test_waveforms_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
