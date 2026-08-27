/*
 * test_check_plan.c -- the shared preprocessing must not change any answer.
 *
 * A pulseg_check_plan exists to avoid repeating work, so the only thing that
 * can go wrong with it is that a cached window is handed to a question it was
 * not extracted for. Every test here therefore asks the same question twice --
 * once with a private plan and once through a shared one that has already been
 * used for something else -- and requires the two answers to agree exactly.
 */
#include "test_helpers.h"

#include "pulseg_pns_models.h"

static pulseg_opts s_opts;
static pulseg_diagnostic s_diag;

#define CHECK_PLAN_FIXTURE "epi_2d_main.seq"

/* Generous gradient and slew limits: these tests exercise the plan, and a
 * limit this fixture's EPI readout trips would stop pulseg_check_safety
 * before it ever reaches the checks a plan is shared by. */
static void check_plan_opts_init(pulseg_opts *opts)
{
    pulseg_opts_init(
        opts,
        GAMMA_HZ_PER_T,
        3.0f,
        GAMMA_HZ_PER_T * 0.080f, /* 80 mT/m   */
        GAMMA_HZ_PER_T * 400.0f, /* 400 T/m/s */
        2.0f,
        20.0f,
        2.0f,
        20.0f);
}

/* ================================================================== */
/*  PNS model                                                         */
/* ================================================================== */

/* The shipped Irnich model: it publishes a kernel, so the memoized route is
 * the one a plan is most likely to interact with. */
static void plan_pns_model_init(pulseg_pns_model *model, pulseg_pns_irnich *ctx)
{
    pulseg_pns_irnich_init(model, ctx, 360.0f, 10.0f, 0.333f);
}

/* ================================================================== */
/*  Band tables                                                       */
/* ================================================================== */

static void violating_bands(pulseg_forbidden_band *bands, pulseg_forbidden_band_list *list)
{
    bands[0].freq_min_hz = 800.0f;
    bands[0].freq_max_hz = 2500.0f;
    bands[0].max_amplitude_hz_per_m = 0.0f;
    bands[1].freq_min_hz = 2500.0f;
    bands[1].freq_max_hz = 4500.0f;
    bands[1].max_amplitude_hz_per_m = 0.0f;
    list->count = 2;
    list->bands = bands;
}

static void permissive_bands(pulseg_forbidden_band *bands, pulseg_forbidden_band_list *list)
{
    bands[0].freq_min_hz = 800.0f;
    bands[0].freq_max_hz = 2500.0f;
    bands[0].max_amplitude_hz_per_m = 1.0e9f;
    bands[1].freq_min_hz = 2500.0f;
    bands[1].freq_max_hz = 4500.0f;
    bands[1].max_amplitude_hz_per_m = 1.0e9f;
    list->count = 2;
    list->bands = bands;
}

/* ================================================================== */
/*  Suite: a plan changes nothing                                     */
/* ================================================================== */

MU_TEST(test_a_shared_plan_gives_the_same_mech_resonance_verdict_as_a_private_one)
{
    pulseg_collection *coll = NULL;
    pulseg_check_plan *plan = NULL;
    pulseg_forbidden_band bands[2];
    pulseg_forbidden_band_list list;
    int rc_private, rc_shared, rc;

    rc = load_corpus_seq(&coll, CHECK_PLAN_FIXTURE, &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_corpus_seq failed");
    violating_bands(bands, &list);

    pulseg_diagnostic_init(&s_diag);
    rc_private = pulseg_check_mech_resonances(coll, &s_diag, NULL, &s_opts, &list);

    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_check_plan_create(&plan, &s_diag, coll, NULL);
    mu_assert(PULSEG_SUCCEEDED(rc), "plan creation failed");
    rc_shared = pulseg_check_mech_resonances(coll, &s_diag, plan, &s_opts, &list);

    mu_assert_int_eq(PULSEG_ERR_MECH_RESONANCES_VIOLATION, rc_private);
    mu_assert_int_eq(rc_private, rc_shared);

    pulseg_check_plan_destroy(plan);
    pulseg_collection_free(coll);
}

MU_TEST(test_a_shared_plan_gives_the_same_pns_peak_as_a_private_one)
{
    pulseg_collection *coll = NULL;
    pulseg_check_plan *plan = NULL;
    pulseg_pns_irnich ctx;
    pulseg_pns_model model;
    pulseg_pns_result private_result, shared_result;
    int i, rc;

    plan_pns_model_init(&model, &ctx);
    rc = load_corpus_seq(&coll, CHECK_PLAN_FIXTURE, &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_corpus_seq failed");

    memset(&private_result, 0, sizeof(private_result));
    memset(&shared_result, 0, sizeof(shared_result));

    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_calc_pns(coll, &private_result, &s_diag, NULL, 0, 0, &s_opts, &model);
    mu_assert(PULSEG_SUCCEEDED(rc), "PNS on a private plan failed");

    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_check_plan_create(&plan, &s_diag, coll, NULL);
    mu_assert(PULSEG_SUCCEEDED(rc), "plan creation failed");
    rc = pulseg_calc_pns(coll, &shared_result, &s_diag, plan, 0, 0, &s_opts, &model);
    mu_assert(PULSEG_SUCCEEDED(rc), "PNS on a shared plan failed");

    mu_assert(private_result.num_samples > 0, "PNS produced no samples");
    mu_assert_int_eq(private_result.num_samples, shared_result.num_samples);
    for (i = 0; i < private_result.num_samples; ++i)
    {
        mu_assert(
            private_result.slew_x_hz_per_m_per_s[i] == shared_result.slew_x_hz_per_m_per_s[i] &&
                private_result.slew_y_hz_per_m_per_s[i] == shared_result.slew_y_hz_per_m_per_s[i] &&
                private_result.slew_z_hz_per_m_per_s[i] == shared_result.slew_z_hz_per_m_per_s[i],
            "a shared plan changed the PNS waveform");
    }

    pulseg_pns_result_free(&private_result);
    pulseg_pns_result_free(&shared_result);
    pulseg_check_plan_destroy(plan);
    pulseg_collection_free(coll);
}

/* A window extracted for one question must not be reused for another that
 * needs a different one. Running both checks through a single plan, in both
 * orders, is what would surface such a mix-up. */
MU_TEST(test_running_both_checks_through_one_plan_matches_running_them_apart)
{
    pulseg_collection *coll = NULL;
    pulseg_check_plan *plan = NULL;
    pulseg_pns_irnich ctx;
    pulseg_pns_model model;
    pulseg_forbidden_band bands[2];
    pulseg_forbidden_band_list list;
    int apart_mech, apart_pns, together_mech, together_pns, rc;

    plan_pns_model_init(&model, &ctx);
    rc = load_corpus_seq(&coll, CHECK_PLAN_FIXTURE, &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_corpus_seq failed");
    permissive_bands(bands, &list);

    pulseg_diagnostic_init(&s_diag);
    apart_mech = pulseg_check_mech_resonances(coll, &s_diag, NULL, &s_opts, &list);
    pulseg_diagnostic_init(&s_diag);
    apart_pns = pulseg_check_pns(coll, &s_diag, NULL, &s_opts, &model, 100.0f);

    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_check_plan_create(&plan, &s_diag, coll, NULL);
    mu_assert(PULSEG_SUCCEEDED(rc), "plan creation failed");
    together_pns = pulseg_check_pns(coll, &s_diag, plan, &s_opts, &model, 100.0f);
    together_mech = pulseg_check_mech_resonances(coll, &s_diag, plan, &s_opts, &list);

    mu_assert_int_eq(apart_mech, together_mech);
    mu_assert_int_eq(apart_pns, together_pns);

    pulseg_check_plan_destroy(plan);
    pulseg_collection_free(coll);
}

/* A budget too small to hold even one window must still answer, by evicting
 * and re-extracting rather than by failing or by reading a freed window. */
MU_TEST(test_a_budget_too_small_to_cache_anything_still_gives_the_same_verdict)
{
    pulseg_collection *coll = NULL;
    pulseg_check_plan *plan = NULL;
    pulseg_check_plan_config config = PULSEG_CHECK_PLAN_CONFIG_INIT;
    pulseg_forbidden_band bands[2];
    pulseg_forbidden_band_list list;
    int rc_default, rc_starved, rc;

    rc = load_corpus_seq(&coll, CHECK_PLAN_FIXTURE, &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_corpus_seq failed");
    violating_bands(bands, &list);

    pulseg_diagnostic_init(&s_diag);
    rc_default = pulseg_check_mech_resonances(coll, &s_diag, NULL, &s_opts, &list);

    config.cache_budget_kb = 1;
    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_check_plan_create(&plan, &s_diag, coll, &config);
    mu_assert(PULSEG_SUCCEEDED(rc), "plan creation failed");
    rc_starved = pulseg_check_mech_resonances(coll, &s_diag, plan, &s_opts, &list);

    mu_assert_int_eq(rc_default, rc_starved);

    pulseg_check_plan_destroy(plan);
    pulseg_collection_free(coll);
}

/* ================================================================== */
/*  Suite: each check stands alone                                    */
/* ================================================================== */

/* A platform that gates amplitude in hardware wants only the rest, so each
 * check has to answer without the others having run. */
MU_TEST(test_each_check_answers_on_its_own)
{
    pulseg_collection *coll = NULL;
    int rc;

    rc = load_corpus_seq(&coll, CHECK_PLAN_FIXTURE, &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_corpus_seq failed");

    pulseg_diagnostic_init(&s_diag);
    mu_assert(
        PULSEG_SUCCEEDED(pulseg_check_max_grad(coll, &s_diag, &s_opts)),
        "max-grad check failed on a fixture inside the limits");

    pulseg_diagnostic_init(&s_diag);
    mu_assert(
        PULSEG_SUCCEEDED(pulseg_check_max_slew(coll, &s_diag, &s_opts)),
        "max-slew check failed on a fixture inside the limits");

    pulseg_diagnostic_init(&s_diag);
    mu_assert(
        PULSEG_SUCCEEDED(pulseg_check_raster_alignment(coll, &s_diag, &s_opts)),
        "raster alignment check failed on a corpus fixture");

    pulseg_collection_free(coll);
}

/* An amplitude limit below what the fixture plays must be caught by the
 * standalone check exactly as pulseg_check_safety catches it. */
MU_TEST(test_the_standalone_amplitude_check_agrees_with_the_full_gate)
{
    pulseg_collection *coll = NULL;
    pulseg_opts tight;
    int rc_alone, rc_full, rc;

    rc = load_corpus_seq(&coll, CHECK_PLAN_FIXTURE, &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_corpus_seq failed");

    tight = s_opts;
    tight.max_grad_hz_per_m = GAMMA_HZ_PER_T * 0.001f; /* 1 mT/m */

    pulseg_diagnostic_init(&s_diag);
    rc_alone = pulseg_check_max_grad(coll, &s_diag, &tight);

    pulseg_diagnostic_init(&s_diag);
    rc_full = pulseg_check_safety(coll, &s_diag, NULL, &tight, NULL, NULL, 0.0f);

    mu_assert(PULSEG_FAILED(rc_alone), "a 1 mT/m limit should refuse this fixture");
    mu_assert_int_eq(rc_alone, rc_full);

    pulseg_collection_free(coll);
}

/* ================================================================== */
/*  Suites                                                            */
/* ================================================================== */

MU_TEST_SUITE(suite_plan_changes_nothing)
{
    MU_SUITE_CONFIGURE(NULL, NULL);
    check_plan_opts_init(&s_opts);

    MU_RUN_TEST(test_a_shared_plan_gives_the_same_mech_resonance_verdict_as_a_private_one);
    MU_RUN_TEST(test_a_shared_plan_gives_the_same_pns_peak_as_a_private_one);
    MU_RUN_TEST(test_running_both_checks_through_one_plan_matches_running_them_apart);
    MU_RUN_TEST(test_a_budget_too_small_to_cache_anything_still_gives_the_same_verdict);
}

MU_TEST_SUITE(suite_checks_stand_alone)
{
    MU_SUITE_CONFIGURE(NULL, NULL);
    check_plan_opts_init(&s_opts);

    MU_RUN_TEST(test_each_check_answers_on_its_own);
    MU_RUN_TEST(test_the_standalone_amplitude_check_agrees_with_the_full_gate);
}

int test_check_plan_main(void)
{
    minunit_run = 0;
    minunit_fail = 0;
    minunit_assert = 0;
    minunit_status = 0;
    minunit_real_timer = 0;
    minunit_proc_timer = 0;

    MU_RUN_SUITE(suite_plan_changes_nothing);
    MU_RUN_SUITE(suite_checks_stand_alone);
    MU_REPORT();
    return MU_EXIT_CODE;
}
