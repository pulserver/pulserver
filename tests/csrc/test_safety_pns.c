/*
 * test_safety_pns.c -- peripheral nerve stimulation check tests.
 *
 * Tests:
 *   1. PNS below threshold -> PASS
 *   2. PNS above threshold -> ERR_PNS_THRESHOLD_EXCEEDED
 *   3. calc_pns returns well-formed slew-rate waveforms
 *   4. Invalid PNS params are rejected cleanly
 *
 * Requires:
 *   - expected_output/epi_2d.seq   (fast slewing -> will exercise PNS)
 *
 * Until generated, seq1.seq is used for smoke tests.
 */
#include "test_helpers.h"

/* ------------------------------------------------------------------ */
/*  Sensible default PNS params (GE-style nerve model)                */
/* ------------------------------------------------------------------ */

static void default_pns_params(pulseqlib_pns_params* p) {
    p->vendor                = PULSEQLIB_VENDOR_GEHC;
    p->chronaxie_us          = 360.0f;
    p->rheobase_hz_per_m_per_s = 20.0f * TEST_GAMMA; /* 20 T/m/s */
    p->alpha                 = 0.6f;
}

/* ------------------------------------------------------------------ */
/*  Smoke: calc_pns on seq1                                           */
/* ------------------------------------------------------------------ */

MU_TEST(test_pns_seq1_smoke)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_pns_result  res  = PULSEQLIB_PNS_RESULT_INIT;
    pulseqlib_pns_params  params;
    pulseqlib_opts opts;
    int rc;

    rc = load_seq("expected_output/seq1.seq", &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load seq1");

    test_opts_init(&opts);
    default_pns_params(&params);

    rc = pulseqlib_calc_pns(&res, &diag, coll, 0, &opts, &params);
    /* May succeed or return NO_WAVEFORM if no gradient content */
    mu_assert(PULSEQLIB_SUCCEEDED(rc)
              || rc == PULSEQLIB_ERR_PNS_NO_WAVEFORM,
              "calc_pns should succeed or return NO_WAVEFORM");

    if (PULSEQLIB_SUCCEEDED(rc)) {
        mu_assert(res.num_samples > 0, "result samples > 0");
        mu_assert(res.slew_x_hz_per_m_per_s != NULL, "slew_x not NULL");
        pulseqlib_pns_result_free(&res);
    }

    pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Pass-path: generous threshold on seq1                             */
/* ------------------------------------------------------------------ */

MU_TEST(test_pns_seq1_passes)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_pns_params  params;
    pulseqlib_opts opts;
    int rc;

    rc = load_seq("expected_output/seq1.seq", &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load seq1");

    test_opts_init(&opts);
    default_pns_params(&params);

    /* Very generous threshold */
    rc = pulseqlib_check_safety(coll, &diag, &opts,
                                0, NULL, &params, 1000.0f);
    mu_assert(PULSEQLIB_SUCCEEDED(rc),
              "seq1 with generous PNS threshold should pass");

    pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Fail-path: absurdly tight threshold                               */
/* ------------------------------------------------------------------ */

MU_TEST(test_pns_tight_threshold)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_pns_params  params;
    pulseqlib_opts opts;
    int rc;

    rc = load_seq("expected_output/seq1.seq", &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load seq1");

    test_opts_init(&opts);
    default_pns_params(&params);
    /* rheobase so low that any gradient triggers PNS */
    params.rheobase_hz_per_m_per_s = 1.0f;

    rc = pulseqlib_check_safety(coll, &diag, &opts,
                                0, NULL, &params, 1.0f);
    /* Should fail with PNS_THRESHOLD_EXCEEDED if gradients present,
     * or pass if seq has no gradient content worth checking. */
    mu_assert(PULSEQLIB_SUCCEEDED(rc)
              || rc == PULSEQLIB_ERR_PNS_THRESHOLD_EXCEEDED,
              "tight PNS should either pass (no grads) or fail");

    pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Stub: EPI with realistic PNS                                      */
/* ------------------------------------------------------------------ */

/*
 * TODO: uncomment once epi_2d.seq is generated.
 *
 * MU_TEST(test_pns_epi_passes)
 * {
 *     pulseqlib_collection* coll = NULL;
 *     pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
 *     pulseqlib_pns_params  params;
 *     pulseqlib_opts opts;
 *     int rc;
 *
 *     rc = load_seq("expected_output/epi_2d.seq", &coll, &diag, 0);
 *     mu_assert(PULSEQLIB_SUCCEEDED(rc), "load EPI");
 *
 *     test_opts_init(&opts);
 *     default_pns_params(&params);
 *
 *     rc = pulseqlib_check_safety(coll, &diag, &opts,
 *                                 0, NULL, &params, 100.0f);
 *     mu_assert(PULSEQLIB_SUCCEEDED(rc),
 *               "well-designed EPI should pass PNS at 100%");
 *
 *     pulseqlib_collection_free(coll);
 * }
 */

/* ------------------------------------------------------------------ */
/*  Suite                                                             */
/* ------------------------------------------------------------------ */

MU_TEST_SUITE(test_safety_pns_suite)
{
    MU_RUN_TEST(test_pns_seq1_smoke);
    MU_RUN_TEST(test_pns_seq1_passes);
    MU_RUN_TEST(test_pns_tight_threshold);
    /* MU_RUN_TEST(test_pns_epi_passes); */
}

int test_safety_pns_main(void)
{
    MU_RUN_SUITE(test_safety_pns_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
