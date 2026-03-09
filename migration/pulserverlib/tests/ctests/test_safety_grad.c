/*
 * test_safety_grad.c -- gradient safety tests.
 *
 * Suite A: gradient amplitude / slew-rate limit violations (4 tests).
 * Suite B: gradient continuity checks (17 tests).
 */
#include "test_helpers.h"

/* ================================================================== */
/*  Shared data-driven helpers                                        */
/* ================================================================== */

static pulseqlib_opts   s_opts;
static pulseqlib_diagnostic s_diag;

/**
 * Load a sequence, run check_safety with the current s_opts,
 * compare return code to expected_code.
 */
static void run_safety_check(const char* filename, int expected_code)
{
    pulseqlib_collection* coll = NULL;
    int rc;

    rc = load_seq(&coll, filename, &s_opts);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq failed");

    pulseqlib_diagnostic_init(&s_diag);
    rc = pulseqlib_check_safety(coll, &s_diag, &s_opts,
                                0, NULL,   /* no forbidden bands */
                                NULL, 0.0f /* no PNS */);

    if (expected_code > 0) {
        mu_assert(PULSEQLIB_SUCCEEDED(rc), "expected success but got failure");
    } else {
        mu_assert_int_eq(expected_code, rc);
    }

    pulseqlib_collection_free(coll);
}

/* ================================================================== */
/*  Continuity-class error detection                                  */
/* ================================================================== */

/**
 * Returns non-zero if @p rc is a gradient-continuity-class error:
 * either the segmentation check (nonzero start/end gradient at TR
 * boundaries) or the safety check (inter-block discontinuity).
 */
static int is_grad_continuity_error(int rc)
{
    return rc == PULSEQLIB_ERR_SEG_NONZERO_START_GRAD ||
           rc == PULSEQLIB_ERR_SEG_NONZERO_END_GRAD   ||
           rc == PULSEQLIB_ERR_GRAD_DISCONTINUITY;
}

/**
 * Load a sequence and check gradient continuity.
 *
 * The library detects gradient continuity violations at two stages:
 *   1. Segmentation (during pulseqlib_read) — catches gradients whose
 *      first/last sample at TR boundaries exceeds max_slew * grad_raster.
 *   2. Safety check (check_grad_continuity) — catches inter-block
 *      gradient steps that exceed the same threshold.
 *
 * For "should-pass" cases the safety check may still fail for
 * non-continuity reasons (e.g. max-slew-rate on trapezoidal ramps that
 * were generated at pypulseq's max_slew); we only verify that
 * PULSEQLIB_ERR_GRAD_DISCONTINUITY is *not* returned.
 *
 * @param filename    Basename of .seq in TEST_DATA_DIR.
 * @param should_pass 1 = sequence is gradient-continuous (no grad-class
 *                    error expected); 0 = a grad-class error is expected.
 */
static void run_continuity_check(const char* filename, int should_pass)
{
    pulseqlib_collection* coll = NULL;
    int rc;

    rc = load_seq(&coll, filename, &s_opts);

    if (PULSEQLIB_FAILED(rc)) {
        if (is_grad_continuity_error(rc)) {
            /* Segmentation caught a gradient boundary issue */
            mu_assert(!should_pass,
                      "continuous sequence rejected by grad boundary check");
            return;
        }
        /* Non-gradient load failure — unexpected for continuity tests */
        mu_assert(0, "load_seq failed with unexpected error");
        return;
    }

    /* Load succeeded — run full safety check */
    pulseqlib_diagnostic_init(&s_diag);
    rc = pulseqlib_check_safety(coll, &s_diag, &s_opts,
                                0, NULL, NULL, 0.0f);

    if (should_pass) {
        /* Sequence is continuous; safety may fail for non-continuity
         * reasons (e.g. slew rate) but must not be GRAD_DISCONTINUITY. */
        mu_assert(rc != PULSEQLIB_ERR_GRAD_DISCONTINUITY,
                  "expected no discontinuity but got GRAD_DISCONTINUITY");
    } else {
        mu_assert(rc == PULSEQLIB_ERR_GRAD_DISCONTINUITY,
                  "expected GRAD_DISCONTINUITY");
    }

    pulseqlib_collection_free(coll);
}

/* ================================================================== */
/*  Suite A — Gradient limit tests                                    */
/* ================================================================== */

/*
 * For amplitude tests: set max_grad tight, max_slew huge.
 * For slew tests: set max_slew tight, max_grad huge.
 * Non-tested limits are 1e10 so they never fire first.
 */

static void grad_limit_opts(float max_grad, float max_slew)
{
    pulseqlib_opts_init(&s_opts,
        GAMMA_HZ_PER_T, 3.0f,
        max_grad, max_slew,
        1.0f, 10.0f, 0.1f, 10.0f);
}

MU_TEST(test_grad_amplitude_violation)
{
    grad_limit_opts(10.0f, 1e10f);
    run_safety_check("01_grad_amplitude_violation.seq",
                     PULSEQLIB_ERR_MAX_GRAD_EXCEEDED);
}

MU_TEST(test_slew_violation)
{
    grad_limit_opts(1e10f, 100.0f);
    run_safety_check("02_slew_violation.seq",
                     PULSEQLIB_ERR_MAX_SLEW_EXCEEDED);
}

MU_TEST(test_grad_rss_violation)
{
    grad_limit_opts(10.0f, 1e10f);
    run_safety_check("03_grad_rss_violation.seq",
                     PULSEQLIB_ERR_MAX_GRAD_EXCEEDED);
}

MU_TEST(test_slew_rss_violation)
{
    grad_limit_opts(1e10f, 100.0f);
    run_safety_check("04_slew_rss_violation.seq",
                     PULSEQLIB_ERR_MAX_SLEW_EXCEEDED);
}

MU_TEST_SUITE(suite_grad_limits)
{
    MU_RUN_TEST(test_grad_amplitude_violation);
    MU_RUN_TEST(test_slew_violation);
    MU_RUN_TEST(test_grad_rss_violation);
    MU_RUN_TEST(test_slew_rss_violation);
}

/* ================================================================== */
/*  Suite B — Gradient continuity tests (16 of 17 files)              */
/* ================================================================== */

MU_TEST(test_cont_01_ok_trap_extended_trap)
{
    run_continuity_check("01_ok_trap_extended_trap.seq", 1);
}

MU_TEST(test_cont_02_fail_trap_then_startshigh)
{
    run_continuity_check("02_fail_trap_then_startshigh.seq", 0);
}

MU_TEST(test_cont_03_fail_startshigh_first)
{
    run_continuity_check("03_fail_startshigh_first.seq", 0);
}

MU_TEST(test_cont_04_fail_delay_then_allhigh)
{
    run_continuity_check("04_fail_delay_then_allhigh.seq", 0);
}

MU_TEST(test_cont_05_ok_extended_with_delay)
{
    run_continuity_check("05_ok_extended_with_delay.seq", 1);
}

MU_TEST(test_cont_06_fail_delay_then_startshigh)
{
    run_continuity_check("06_fail_delay_then_startshigh.seq", 0);
}

MU_TEST(test_cont_07_fail_nonconnecting)
{
    run_continuity_check("07_fail_nonconnecting.seq", 0);
}

MU_TEST(test_cont_08_ok_rot_identity)
{
    run_continuity_check("08_ok_rot_identity.seq", 1);
}

MU_TEST(test_cont_09_fail_rot_identity)
{
    run_continuity_check("09_fail_rot_identity.seq", 0);
}

MU_TEST(test_cont_10_fail_rot_first_block)
{
    run_continuity_check("10_fail_rot_first_block.seq", 0);
}

MU_TEST(test_cont_11_fail_rot_allhigh)
{
    run_continuity_check("11_fail_rot_allhigh.seq", 0);
}

MU_TEST(test_cont_12_ok_rot_extended_delay)
{
    run_continuity_check("12_ok_rot_extended_delay.seq", 1);
}

MU_TEST(test_cont_13_fail_rot_delay_then_startshigh)
{
    run_continuity_check("13_fail_rot_delay_then_startshigh.seq", 0);
}

MU_TEST(test_cont_14_fail_rot_nonconnecting)
{
    run_continuity_check("14_fail_rot_nonconnecting.seq", 0);
}

MU_TEST(test_cont_15_ok_rot_same_rotation)
{
    run_continuity_check("15_ok_rot_same_rotation.seq", 1);
}

MU_TEST(test_cont_16_fail_rot_diff_rotation_1)
{
    run_continuity_check("16_fail_rot_diff_rotation_1.seq", 0);
}

MU_TEST(test_cont_17_fail_rot_diff_rotation_2)
{
    run_continuity_check("17_fail_rot_diff_rotation_2.seq", 0);
}

static void continuity_setup(void)
{
    default_opts_init(&s_opts);
}

MU_TEST_SUITE(suite_grad_continuity)
{
    MU_SUITE_CONFIGURE(continuity_setup, NULL);
    MU_RUN_TEST(test_cont_01_ok_trap_extended_trap);
    MU_RUN_TEST(test_cont_02_fail_trap_then_startshigh);
    MU_RUN_TEST(test_cont_03_fail_startshigh_first);
    MU_RUN_TEST(test_cont_04_fail_delay_then_allhigh);
    MU_RUN_TEST(test_cont_05_ok_extended_with_delay);
    MU_RUN_TEST(test_cont_06_fail_delay_then_startshigh);
    MU_RUN_TEST(test_cont_07_fail_nonconnecting);
    MU_RUN_TEST(test_cont_08_ok_rot_identity);
    MU_RUN_TEST(test_cont_09_fail_rot_identity);
    MU_RUN_TEST(test_cont_10_fail_rot_first_block);
    MU_RUN_TEST(test_cont_11_fail_rot_allhigh);
    MU_RUN_TEST(test_cont_12_ok_rot_extended_delay);
    MU_RUN_TEST(test_cont_13_fail_rot_delay_then_startshigh);
    MU_RUN_TEST(test_cont_14_fail_rot_nonconnecting);
    MU_RUN_TEST(test_cont_15_ok_rot_same_rotation);
    MU_RUN_TEST(test_cont_16_fail_rot_diff_rotation_1);
    MU_RUN_TEST(test_cont_17_fail_rot_diff_rotation_2);
}

/* ================================================================== */
/*  Entry point                                                       */
/* ================================================================== */

int test_safety_grad_main(void)
{
    minunit_run = 0;
    minunit_fail = 0;
    minunit_assert = 0;
    minunit_status = 0;
    minunit_real_timer = 0;
    minunit_proc_timer = 0;

    MU_RUN_SUITE(suite_grad_limits);
    MU_RUN_SUITE(suite_grad_continuity);
    MU_REPORT();
    return MU_EXIT_CODE;
}
