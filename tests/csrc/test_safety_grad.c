/*
 * test_safety_grad.c -- gradient continuity, max-grad, max-slew tests.
 *
 * Test strategy:
 *   1. Well-formed sequences (gre, bssfp, epi, fse) must PASS all
 *      gradient safety checks -- no false positives.
 *   2. Intentionally broken sequences (discontinuity, over-amplitude,
 *      over-slew) must FAIL with the correct error code.
 *
 * Requires:
 *   - expected_output/gre_2d.seq   (gradient-safe)
 *   - expected_output/epi_2d.seq   (gradient-safe, fast slewing)
 *   - expected_output/seq1.seq     (always available)
 *
 * Until the full set of .seq files is generated, the tests use seq1.seq
 * for pass-path smoke and intentionally tight opts for fail-path.
 */
#include "test_helpers.h"

/* ------------------------------------------------------------------ */
/*  Helper: run check_safety with gradient-only params (no acoustic/PNS) */
/* ------------------------------------------------------------------ */

static int check_grads(const char* seq_rel,
                       float max_grad, float max_slew)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_opts opts;
    int rc;

    test_opts_init(&opts);
    rc = pulseqlib_read(&coll, &diag, make_path(seq_rel),
                        &opts, 0, 0, 0, 1);
    if (PULSEQLIB_FAILED(rc)) return rc;

    /* Override limits for purposes of the check */
    opts.max_grad_hz_per_m = max_grad;
    opts.max_slew_hz_per_m_per_s = max_slew;

    rc = pulseqlib_check_safety(coll, &diag, &opts,
                                0, NULL, NULL, 100.0f);
    pulseqlib_collection_free(coll);
    return rc;
}

/* ------------------------------------------------------------------ */
/*  Pass-path: seq1 with generous limits should pass                  */
/* ------------------------------------------------------------------ */

MU_TEST(test_grad_seq1_passes)
{
    int rc = check_grads("expected_output/seq1.seq",
                         TEST_MAX_GRAD, TEST_MAX_SLEW);
    mu_assert(PULSEQLIB_SUCCEEDED(rc),
              "seq1.seq should pass gradient safety with default limits");
}

/* ------------------------------------------------------------------ */
/*  Fail-path: seq1 with very tight max_grad should fail              */
/* ------------------------------------------------------------------ */

MU_TEST(test_grad_max_grad_violation)
{
    /* Use absurdly small max_grad so any gradient exceeds it */
    int rc = check_grads("expected_output/seq1.seq",
                         1.0f,          /* 1 Hz/m -- trivially exceeded */
                         TEST_MAX_SLEW);
    mu_assert(rc == PULSEQLIB_ERR_MAX_GRAD_EXCEEDED,
              "tiny max_grad should trigger MAX_GRAD_EXCEEDED");
}

/* ------------------------------------------------------------------ */
/*  Fail-path: seq1 with very tight max_slew should fail              */
/* ------------------------------------------------------------------ */

MU_TEST(test_grad_max_slew_violation)
{
    /* Use absurdly small slew limit */
    int rc = check_grads("expected_output/seq1.seq",
                         TEST_MAX_GRAD,
                         1.0f);  /* 1 Hz/m/s */
    mu_assert(rc == PULSEQLIB_ERR_MAX_SLEW_EXCEEDED,
              "tiny max_slew should trigger MAX_SLEW_EXCEEDED");
}

/* ------------------------------------------------------------------ */
/*  Stub: known-good EPI with realistic limits (no false positive)    */
/* ------------------------------------------------------------------ */

/*
 * TODO: uncomment once epi_2d.seq is generated.
 *
 * MU_TEST(test_grad_epi_passes)
 * {
 *     int rc = check_grads("expected_output/epi_2d.seq",
 *                          TEST_MAX_GRAD, TEST_MAX_SLEW);
 *     mu_assert(PULSEQLIB_SUCCEEDED(rc),
 *               "well-formed EPI should pass gradient checks");
 * }
 */

/* ------------------------------------------------------------------ */
/*  Suite                                                             */
/* ------------------------------------------------------------------ */

MU_TEST_SUITE(test_safety_grad_suite)
{
    MU_RUN_TEST(test_grad_seq1_passes);
    MU_RUN_TEST(test_grad_max_grad_violation);
    MU_RUN_TEST(test_grad_max_slew_violation);
    /* MU_RUN_TEST(test_grad_epi_passes); */
}

int test_safety_grad_main(void)
{
    MU_RUN_SUITE(test_safety_grad_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
