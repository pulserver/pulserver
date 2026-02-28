/*
 * test_safety_grad.c -- gradient amplitude and slew-rate limit tests.
 *
 * Test strategy:
 *   - seq1.seq has combined gradient amplitudes exceeding default
 *     limits -> check_safety must detect the violation.
 *   - Intentionally broken sequences from the grad-limits test set
 *     must also fail with the correct error codes.
 *
 * Requires: data/seq1.seq, data/01_grad_amplitude_violation.seq, etc.
 */
#include "test_helpers.h"

/* ------------------------------------------------------------------ */
/*  Helper: run check_safety with gradient-only params (no PNS)       */
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
/*  seq1 has combined gradients exceeding default limits               */
/* ------------------------------------------------------------------ */

MU_TEST(test_grad_seq1_exceeds_default)
{
    int rc = check_grads("data/seq1.seq",
                         TEST_MAX_GRAD, TEST_MAX_SLEW);
    mu_assert(rc == PULSEQLIB_ERR_MAX_GRAD_EXCEEDED,
              "seq1 combined gradients should exceed default limits");
}

/* ------------------------------------------------------------------ */
/*  seq1 with very generous limits should pass                        */
/* ------------------------------------------------------------------ */

MU_TEST(test_grad_seq1_generous_passes)
{
    /* 100 mT/m = well above the ~56 mT/m RSS in seq1 */
    float generous_grad = 100.0f * TEST_GAMMA * 1e-3f;
    float generous_slew = 500.0f * TEST_GAMMA;
    int rc = check_grads("data/seq1.seq",
                         generous_grad, generous_slew);
    mu_assert(PULSEQLIB_SUCCEEDED(rc),
              "seq1 should pass with very generous limits");
}

/* ------------------------------------------------------------------ */
/*  Explicit amplitude violation test file                            */
/* ------------------------------------------------------------------ */

MU_TEST(test_grad_amplitude_violation_file)
{
    /* The MATLAB-generated file has very small amplitudes by design;
     * use a tight limit to trigger the violation.                    */
    int rc = check_grads("data/01_grad_amplitude_violation.seq",
                         10.0f, TEST_MAX_SLEW);
    mu_assert(rc == PULSEQLIB_ERR_MAX_GRAD_EXCEEDED,
              "amplitude violation file should fail with tight limit");
}

/* ------------------------------------------------------------------ */
/*  Explicit slew violation test file                                 */
/* ------------------------------------------------------------------ */

MU_TEST(test_grad_slew_violation_file)
{
    /* Use a very tight slew limit so the file triggers the check.   */
    int rc = check_grads("data/02_grad_slew_violation.seq",
                         TEST_MAX_GRAD, 1.0f);
    mu_assert(PULSEQLIB_FAILED(rc),
              "slew violation file should fail with tight slew limit");
}

/* ------------------------------------------------------------------ */
/*  Suite                                                             */
/* ------------------------------------------------------------------ */

MU_TEST_SUITE(test_safety_grad_suite)
{
    MU_RUN_TEST(test_grad_seq1_exceeds_default);
    MU_RUN_TEST(test_grad_seq1_generous_passes);
    MU_RUN_TEST(test_grad_amplitude_violation_file);
    MU_RUN_TEST(test_grad_slew_violation_file);
}

int test_safety_grad_main(void)
{
    MU_RUN_SUITE(test_safety_grad_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
