/*
 * test_module_labels.c -- MODULE (safety-group) label ctests.
 *
 * Suite: pulseg_get_modules() -- identify / verify structural identity /
 * dedup, per PLAN_safety_module_labels.md decision 4.
 */
#include "test_helpers.h"

/* ================================================================== */
/*  Suite — MODULE labels                                              */
/* ================================================================== */

/* Two MODULE-labeled groups within one pass: module 1 = blocks 1-2
 * (100us + 100us = 200us), module 2 = block 3 alone (50us). */
MU_TEST(test_module_two_groups_one_pass)
{
    pulseg_opts opts;
    pulseg_collection* coll = NULL;
    pulseg_module* mods = NULL;
    int rc, nmods;

    default_opts_init(&opts);
    rc = load_seq(&coll, "module_two.seq", &opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

    nmods = pulseg_get_modules(coll, &mods, 0);
    mu_assert(nmods >= 0, "pulseg_get_modules failed");
    mu_assert_int_eq(2, nmods);

    mu_assert_int_eq(1, mods[0].module_id);
    mu_assert_int_eq(1, mods[0].num_instances);
    mu_assert_int_eq(200, mods[0].one_instance_duration_us);
    mu_assert_int_eq(200, mods[0].total_duration_us);

    mu_assert_int_eq(2, mods[1].module_id);
    mu_assert_int_eq(1, mods[1].num_instances);
    mu_assert_int_eq(50, mods[1].one_instance_duration_us);
    mu_assert_int_eq(50, mods[1].total_duration_us);

    PULSEG_FREE(mods);
    pulseg_collection_free(coll);
}

/* Unlabeled sequence: pulseg_get_modules() must return 0 (not an error) --
 * this is the bit-identical-when-unlabeled guarantee's precondition. */
MU_TEST(test_module_none_returns_zero)
{
    pulseg_opts opts;
    pulseg_collection* coll = NULL;
    pulseg_module* mods = NULL;
    int rc, nmods;

    default_opts_init(&opts);
    rc = load_seq(&coll, "00_basic_rfstat.seq", &opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

    nmods = pulseg_get_modules(coll, &mods, 0);
    mu_assert_int_eq(0, nmods);
    mu_assert(mods == NULL, "expected NULL modules array when none found");

    pulseg_collection_free(coll);
}

/* One MODULE spanning the entire (single-block) TR, loaded with
 * num_averages=3: NEX/pass repeats of the SAME module_id across a TR
 * boundary must split into separate occurrences (scan_table_tr_start),
 * not merge into one giant run. Without that fix this would report
 * num_instances=1, one_instance_duration_us=300 instead of
 * num_instances=3, one_instance_duration_us=100. */
MU_TEST(test_module_nex_repeat_splits_on_tr_boundary)
{
    pulseg_opts opts;
    pulseg_collection* coll = NULL;
    pulseg_module* mods = NULL;
    int rc, nmods;

    default_opts_init(&opts);
    rc = load_seq_with_averages(&coll, "module_single_nex.seq", &opts, 3);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq_with_averages failed");

    nmods = pulseg_get_modules(coll, &mods, 0);
    mu_assert(nmods >= 0, "pulseg_get_modules failed");
    mu_assert_int_eq(1, nmods);

    mu_assert_int_eq(1, mods[0].module_id);
    mu_assert_int_eq(100, mods[0].one_instance_duration_us);
    mu_assert_int_eq(3, mods[0].num_instances);
    mu_assert_int_eq(300, mods[0].total_duration_us);

    PULSEG_FREE(mods);
    pulseg_collection_free(coll);
}

/* MODULE 1 appears twice (blocks 1 and 3) with DIFFERENT durations (100us
 * vs 200us) -- must hard-fail with PULSEG_ERR_MODULE_STRUCTURAL_MISMATCH,
 * not silently average or dedup against a stale reference. */
MU_TEST(test_module_structural_mismatch_hard_fails)
{
    pulseg_opts opts;
    pulseg_collection* coll = NULL;
    pulseg_module* mods = NULL;
    int rc, nmods;

    default_opts_init(&opts);
    rc = load_seq(&coll, "module_mismatch.seq", &opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

    nmods = pulseg_get_modules(coll, &mods, 0);
    mu_assert_int_eq(PULSEG_ERR_MODULE_STRUCTURAL_MISMATCH, nmods);
    mu_assert(mods == NULL, "expected NULL modules array on structural mismatch");

    pulseg_collection_free(coll);
}

/* pulseg_get_rf_array()'s module_id per-entry tag must reflect the
 * originating block's sticky MODULE label, not the deduplicated RF
 * definition (which carries no module identity). Reuses the RF fixture
 * with num_averages default (1); the fixture has no MODULE labels, so
 * every rf_stats[i].module_id must come back 0 (ungrouped default). */
MU_TEST(test_rf_array_module_id_defaults_zero)
{
    pulseg_opts opts;
    pulseg_collection* coll = NULL;
    pulseg_rf_stats* stats = NULL;
    int rc, nstats, i;

    default_opts_init(&opts);
    rc = load_seq(&coll, "00_basic_rfstat.seq", &opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

    nstats = pulseg_get_rf_array(coll, &stats, 0);
    mu_assert(nstats > 0, "expected at least one RF event");
    for (i = 0; i < nstats; ++i)
        mu_assert_int_eq(0, stats[i].module_id);

    PULSEG_FREE(stats);
    pulseg_collection_free(coll);
}

MU_TEST_SUITE(suite_module_labels)
{
    MU_RUN_TEST(test_module_two_groups_one_pass);
    MU_RUN_TEST(test_module_none_returns_zero);
    MU_RUN_TEST(test_module_nex_repeat_splits_on_tr_boundary);
    MU_RUN_TEST(test_module_structural_mismatch_hard_fails);
    MU_RUN_TEST(test_rf_array_module_id_defaults_zero);
}

/* ================================================================== */
/*  Entry point                                                       */
/* ================================================================== */

int test_module_labels_main(void)
{
    minunit_run = 0;
    minunit_fail = 0;
    minunit_assert = 0;
    minunit_status = 0;
    minunit_real_timer = 0;
    minunit_proc_timer = 0;

    MU_RUN_SUITE(suite_module_labels);
    MU_REPORT();
    return MU_EXIT_CODE;
}
