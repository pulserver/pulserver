/*
 * test_waveforms.c -- TR gradient waveform extraction and
 *                     cross-validation with pulseq toolbox.
 *
 * Tests:
 *   1. get_tr_gradient_waveforms returns non-empty arrays for
 *      sequences with gradients.
 *   2. Time arrays are monotonically increasing.
 *   3. First / last amplitudes match gradient continuity
 *      (should be zero at TR boundaries for safe sequences).
 *   4. Cross-validation: gradient waveforms in .dat files
 *      (produced by Python toolbox get_gradient_waveforms)
 *      match the C library output within tolerance.
 *
 * Requires:
 *   - expected_output/seq1.seq
 *   - expected_output/spgr_waveform.dat   (from make_test_waveforms.py)
 *   - expected_output/bssfp_waveform.dat
 *   - expected_output/megre_waveform.dat
 */
#include "test_helpers.h"

/* ------------------------------------------------------------------ */
/*  Smoke: seq1 waveforms are extractable                             */
/* ------------------------------------------------------------------ */

MU_TEST(test_waveforms_seq1_smoke)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_tr_gradient_waveforms w = PULSEQLIB_TR_GRADIENT_WAVEFORMS_INIT;
    int rc;

    rc = load_seq("expected_output/seq1.seq", &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load seq1");

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
/*  Stub: cross-validate against Python-generated .dat file           */
/* ------------------------------------------------------------------ */

/*
 * TODO: implement once the .dat format is standardized.
 *
 * The reference waveform files are produced by make_test_waveforms.py
 * which calls pypulseq's get_gradient_waveforms() and dumps
 * (time_us, gx, gy, gz) as whitespace-separated columns.
 *
 * Strategy:
 *   1. Load .seq -> get_tr_gradient_waveforms -> interpolate to 10 us raster
 *   2. Read .dat into arrays
 *   3. Compare point-by-point within tolerance (1e-2 Hz/m)
 *
 * MU_TEST(test_waveforms_spgr_crossval)
 * {
 *     // Load seq
 *     // Extract waveforms
 *     // Read expected_output/spgr_waveform.dat
 *     // Per-sample comparison
 * }
 */

/* ------------------------------------------------------------------ */
/*  Suite                                                             */
/* ------------------------------------------------------------------ */

MU_TEST_SUITE(test_waveforms_suite)
{
    MU_RUN_TEST(test_waveforms_seq1_smoke);
    /* MU_RUN_TEST(test_waveforms_spgr_crossval); */
    /* MU_RUN_TEST(test_waveforms_bssfp_crossval); */
    /* MU_RUN_TEST(test_waveforms_megre_crossval); */
}

int test_waveforms_main(void)
{
    MU_RUN_SUITE(test_waveforms_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
