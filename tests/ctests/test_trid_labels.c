/*
 * test_trid_labels.c -- TRID (safety-group) label ctests.
 *
 * Suite: pulseg_get_tr_groups() -- identify / verify structural identity /
 * dedup.  TRID is Pulseq's own label for the repeating unit of a sequence,
 * which Pulserver reads as the group a per-contrast SAR check runs over; it
 * replaced this project's MODULE label, which meant the same thing under a
 * name Pulseq does not define.
 */
#include "test_helpers.h"

/* ================================================================== */
/*  Suite — TRID labels                                                */
/* ================================================================== */

/* Two TRID-labeled groups within one pass: group 1 = blocks 1-2
 * (100us + 100us = 200us), group 2 = block 3 alone (50us). */
MU_TEST(test_trid_two_groups_one_pass)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    pulseg_tr_group *groups = NULL;
    int rc, ngroups;

    default_opts_init(&opts);
    rc = load_seq(&coll, "trid_two.seq", &opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

    ngroups = pulseg_get_tr_groups(coll, &groups, 0);
    mu_assert(ngroups >= 0, "pulseg_get_tr_groups failed");
    mu_assert_int_eq(2, ngroups);

    mu_assert_int_eq(1, groups[0].trid);
    mu_assert_int_eq(1, groups[0].num_instances);
    mu_assert_int_eq(200, groups[0].one_instance_duration_us);
    mu_assert_int_eq(200, groups[0].total_duration_us);

    mu_assert_int_eq(2, groups[1].trid);
    mu_assert_int_eq(1, groups[1].num_instances);
    mu_assert_int_eq(50, groups[1].one_instance_duration_us);
    mu_assert_int_eq(50, groups[1].total_duration_us);

    PULSEG_FREE(groups);
    pulseg_collection_free(coll);
}

/* Unlabeled sequence: pulseg_get_tr_groups() must return 0 (not an error) --
 * this is the bit-identical-when-unlabeled guarantee's precondition. */
MU_TEST(test_trid_none_returns_zero)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    pulseg_tr_group *groups = NULL;
    int rc, ngroups;

    default_opts_init(&opts);
    rc = load_seq(&coll, "00_basic_rfstat.seq", &opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

    ngroups = pulseg_get_tr_groups(coll, &groups, 0);
    mu_assert_int_eq(0, ngroups);
    mu_assert(groups == NULL, "expected NULL group array when none found");

    pulseg_collection_free(coll);
}

/* One TRID spanning the entire (single-block) TR, loaded with
 * num_averages=3: NEX/pass repeats of the SAME TRID across a TR
 * boundary must split into separate occurrences (exec_stream_tr_start),
 * not merge into one giant run. Without that fix this would report
 * num_instances=1, one_instance_duration_us=300 instead of
 * num_instances=3, one_instance_duration_us=100. */
MU_TEST(test_trid_nex_repeat_splits_on_tr_boundary)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    pulseg_tr_group *groups = NULL;
    int rc, ngroups;

    default_opts_init(&opts);
    rc = load_seq_with_averages(&coll, "trid_single_nex.seq", &opts, 3);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq_with_averages failed");

    ngroups = pulseg_get_tr_groups(coll, &groups, 0);
    mu_assert(ngroups >= 0, "pulseg_get_tr_groups failed");
    mu_assert_int_eq(1, ngroups);

    mu_assert_int_eq(1, groups[0].trid);
    mu_assert_int_eq(100, groups[0].one_instance_duration_us);
    mu_assert_int_eq(3, groups[0].num_instances);
    mu_assert_int_eq(300, groups[0].total_duration_us);

    PULSEG_FREE(groups);
    pulseg_collection_free(coll);
}

/* TRID 1 appears twice (blocks 1 and 3) with DIFFERENT durations (100us
 * vs 200us) -- must hard-fail with PULSEG_ERR_TRID_STRUCTURAL_MISMATCH,
 * not silently average or dedup against a stale reference. */
MU_TEST(test_trid_structural_mismatch_hard_fails)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    pulseg_tr_group *groups = NULL;
    int rc, ngroups;

    default_opts_init(&opts);
    rc = load_seq(&coll, "trid_mismatch.seq", &opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

    ngroups = pulseg_get_tr_groups(coll, &groups, 0);
    mu_assert_int_eq(PULSEG_ERR_TRID_STRUCTURAL_MISMATCH, ngroups);
    mu_assert(groups == NULL, "expected NULL group array on structural mismatch");

    pulseg_collection_free(coll);
}

/* pulseg_get_rf_array()'s trid per-entry tag must reflect the originating
 * block's sticky TRID, not the deduplicated RF definition (which carries no
 * group identity). Reuses the RF fixture with num_averages default (1); the
 * fixture has no TRID labels, so every rf_stats[i].trid must come back 0
 * (ungrouped default). */
MU_TEST(test_rf_array_trid_defaults_zero)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    pulseg_rf_stats *stats = NULL;
    int rc, nstats, i;

    default_opts_init(&opts);
    rc = load_seq(&coll, "00_basic_rfstat.seq", &opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

    nstats = pulseg_get_rf_array(coll, &stats, 0);
    mu_assert(nstats > 0, "expected at least one RF event");
    for (i = 0; i < nstats; ++i)
        mu_assert_int_eq(0, stats[i].trid);

    PULSEG_FREE(stats);
    pulseg_collection_free(coll);
}

MU_TEST_SUITE(suite_trid_labels)
{
    MU_RUN_TEST(test_trid_two_groups_one_pass);
    MU_RUN_TEST(test_trid_none_returns_zero);
    MU_RUN_TEST(test_trid_nex_repeat_splits_on_tr_boundary);
    MU_RUN_TEST(test_trid_structural_mismatch_hard_fails);
    MU_RUN_TEST(test_rf_array_trid_defaults_zero);
}

/* ================================================================== */
/*  Entry point                                                       */
/* ================================================================== */

int test_trid_labels_main(void)
{
    minunit_run = 0;
    minunit_fail = 0;
    minunit_assert = 0;
    minunit_status = 0;
    minunit_real_timer = 0;
    minunit_proc_timer = 0;

    MU_RUN_SUITE(suite_trid_labels);
    MU_REPORT();
    return MU_EXIT_CODE;
}
