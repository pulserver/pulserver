/*
 * test_safety_grad.c -- gradient amplitude and slew-rate limit tests.
 *
 * Test strategy:
 *   - MATLAB-generated sequences with known gradient content are tested
 *     against various limit configurations.
 *   - Intentionally broken sequences from the grad-limits generator
 *     must also fail with tight limits.
 *
 * Requires: data/01_ok_trap_extended_trap.seq,
 *           data/01_grad_amplitude_violation.seq,
 *           data/02_grad_slew_violation.seq
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
/*  ok_trap exceeds default slew limits                               */
/* ------------------------------------------------------------------ */

MU_TEST(test_grad_ok_trap_exceeds_default)
{
    int rc = check_grads(TEST_SEQ_OK,
                         TEST_MAX_GRAD, TEST_MAX_SLEW);
    mu_assert(rc == PULSEQLIB_ERR_MAX_SLEW_EXCEEDED,
              "ok_trap should exceed default slew limits");
}

/* ------------------------------------------------------------------ */
/*  ok_trap with very generous limits should pass                     */
/* ------------------------------------------------------------------ */

MU_TEST(test_grad_ok_trap_generous_passes)
{
    float generous_grad = 100.0f * TEST_GAMMA * 1e-3f;
    float generous_slew = 500.0f * TEST_GAMMA;
    int rc = check_grads(TEST_SEQ_OK,
                         generous_grad, generous_slew);
    mu_assert(PULSEQLIB_SUCCEEDED(rc),
              "ok_trap should pass with very generous limits");
}

/* ------------------------------------------------------------------ */
/*  Explicit amplitude violation test file                            */
/* ------------------------------------------------------------------ */

MU_TEST(test_grad_amplitude_violation_file)
{
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
    MU_RUN_TEST(test_grad_ok_trap_exceeds_default);
    MU_RUN_TEST(test_grad_ok_trap_generous_passes);
    MU_RUN_TEST(test_grad_amplitude_violation_file);
    MU_RUN_TEST(test_grad_slew_violation_file);
}

int test_safety_grad_main(void)
{
    MU_RUN_SUITE(test_safety_grad_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
