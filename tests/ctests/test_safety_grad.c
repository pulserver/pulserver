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
/*  Suite C -- Canonical segment-sequence API (gradient safety path)  */
/* ================================================================== */

static void assert_canonical_sequence_matches_expected(
    const pulseqlib_collection* coll,
    int subseq_idx,
    int expect_full_pass)
{
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_descriptor* trd;
    int n_prep, n_main, n_cool;
    int ncanon, expected, p, i, w;
    int* canon_ids = NULL;

    desc = &coll->descriptors[subseq_idx];
    trd = &desc->tr_descriptor;

    n_prep = desc->segment_table.num_prep_segments;
    n_main = desc->segment_table.num_main_segments;
    n_cool = desc->segment_table.num_cooldown_segments;

    if (expect_full_pass) {
        expected = n_prep + n_main * ((desc->num_passes > 1) ? desc->num_passes : 1) + n_cool;
        mu_assert(trd->num_prep_blocks > 0, "expected non-zero prep blocks");
        mu_assert(!trd->degenerate_prep, "expected non-degenerate prep");
    } else {
        expected = n_main;
    }

    ncanon = pulseqlib_get_canonical_segment_sequence(coll, subseq_idx, NULL);
    mu_assert_int_eq(expected, ncanon);

    if (ncanon > 0) {
        canon_ids = (int*)malloc((size_t)ncanon * sizeof(int));
        mu_assert(canon_ids != NULL, "malloc failed for canonical segment ids");
    }

    ncanon = pulseqlib_get_canonical_segment_sequence(coll, subseq_idx, canon_ids);
    mu_assert_int_eq(expected, ncanon);

    if (expect_full_pass) {
        w = 0;
        for (i = 0; i < n_prep; ++i)
            mu_assert_int_eq(desc->segment_table.prep_segment_table[i], canon_ids[w++]);

        for (p = 0; p < ((desc->num_passes > 1) ? desc->num_passes : 1); ++p)
            for (i = 0; i < n_main; ++i)
                mu_assert_int_eq(desc->segment_table.main_segment_table[i], canon_ids[w++]);

        for (i = 0; i < n_cool; ++i)
            mu_assert_int_eq(desc->segment_table.cooldown_segment_table[i], canon_ids[w++]);

        mu_assert_int_eq(expected, w);
    } else {
        for (i = 0; i < n_main; ++i)
            mu_assert_int_eq(desc->segment_table.main_segment_table[i], canon_ids[i]);
    }

    free(canon_ids);
}

MU_TEST(test_canonical_segment_sequence_degenerate_main_only)
{
    pulseqlib_collection* coll = NULL;
    int rc;

    default_opts_init(&s_opts);
    rc = load_seq(&coll, "00_basic_rfstat.seq", &s_opts);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq failed");

    assert_canonical_sequence_matches_expected(coll, 0, 0);

    pulseqlib_collection_free(coll);
}

MU_TEST(test_canonical_segment_sequence_nondegenerate_fullpass)
{
    pulseqlib_collection* coll = NULL;
    int rc;

    default_opts_init(&s_opts);
    rc = load_seq_with_averages(
        &coll, "05_rfprep_ok_canonical_fullpass.seq", &s_opts, 3);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq_with_averages failed");

    assert_canonical_sequence_matches_expected(coll, 0, 1);

    pulseqlib_collection_free(coll);
}

MU_TEST_SUITE(suite_grad_canonical_sequence)
{
    MU_RUN_TEST(test_canonical_segment_sequence_degenerate_main_only);
    MU_RUN_TEST(test_canonical_segment_sequence_nondegenerate_fullpass);
}

/* ================================================================== */
/*  Suite D — Mechanical resonance forbidden-frequency safety tests              */
/* ================================================================== */

#if 0  /* Temporary: acoustic checks are known stub work-in-progress. */

/**
 * Load a sequence, run check_safety with the given forbidden bands,
 * compare return code to expected_code.
 *
 * expected_code > 0 means a passing (success) result is expected.
 * expected_code <= 0 means that specific error code is expected.
 */
static void run_mech_resonances_check(const char* filename, int num_bands,
                                const pulseqlib_forbidden_band* bands,
                                int expected_code)
{
    pulseqlib_collection* coll = NULL;
    int rc;

    rc = load_seq(&coll, filename, &s_opts);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq failed for acoustic test");

    pulseqlib_diagnostic_init(&s_diag);
    rc = pulseqlib_check_safety(coll, &s_diag, &s_opts,
                                num_bands, bands,
                                NULL, 0.0f /* no PNS */);

    if (expected_code > 0) {
        mu_assert(PULSEQLIB_SUCCEEDED(rc), "expected acoustic safety pass");
    } else {
        mu_assert_int_eq(expected_code, rc);
    }

    pulseqlib_collection_free(coll);
}

/*
 * EPI readout produces a peaked acoustic spectrum at 1/ESP and harmonics.
 * For this EPI sequence: readoutTime = 580 us, ESP ~ 640 us,
 * giving a fundamental at ~1560 Hz.  Forbid a band spanning
 * 800–2500 Hz (covers 1/ESP for ESP in 400–1250 us range) at zero
 * amplitude and expect an acoustic violation.
 */
MU_TEST(test_epi_forbidden_readout_peak)
{
    pulseqlib_forbidden_band bands[2];

    /* Band 1: 800–2500 Hz — covers 1/ESP fundamental */
    bands[0].freq_min_hz           = 800.0f;
    bands[0].freq_max_hz           = 2500.0f;
    bands[0].max_amplitude_hz_per_m = 0.0f;

    /* Band 2: 2500–4500 Hz — covers 2nd harmonic */
    bands[1].freq_min_hz           = 2500.0f;
    bands[1].freq_max_hz           = 4500.0f;
    bands[1].max_amplitude_hz_per_m = 0.0f;

    run_mech_resonances_check("epi_2d_1sl_1avg.seq", 2, bands,
                       PULSEQLIB_ERR_MECH_RESONANCES_VIOLATION);
}

static void mech_resonances_setup(void)
{
    gre_opts_init(&s_opts);
}

MU_TEST_SUITE(suite_mech_resonances_safety)
{
    MU_SUITE_CONFIGURE(mech_resonances_setup, NULL);
    MU_RUN_TEST(test_epi_forbidden_readout_peak);
}

#endif

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
    MU_RUN_SUITE(suite_grad_canonical_sequence);
    MU_REPORT();
    return MU_EXIT_CODE;
}
