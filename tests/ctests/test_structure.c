/*
 * test_structure.c -- TR detection, prep / cooldown, once-flag tests.
 *
 * Tests:
 *   - ok_trap loads and has structural metadata
 *   - Once-flag valid / invalid detection
 *   - Multipass folding: num_passes, num_prep, num_cooldown
 *
 * Requires: data/01_ok_trap_extended_trap.seq, data/once-flag test files
 */
#include "test_helpers.h"

/* ------------------------------------------------------------------ */
/*  Smoke: ok_trap loads and has structural metadata                     */
/* ------------------------------------------------------------------ */

MU_TEST(test_structure_ok_basic)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_collection_info ci = PULSEQLIB_COLLECTION_INFO_INIT;
    pulseqlib_subseq_info     si = PULSEQLIB_SUBSEQ_INFO_INIT;
    int rc;

    rc = load_seq(TEST_SEQ_OK, &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load should succeed");

    rc = pulseqlib_get_collection_info(coll, &ci);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "collection_info should succeed");
    mu_assert(ci.num_subsequences >= 1, "at least 1 subsequence");

    rc = pulseqlib_get_subseq_info(coll, 0, &si);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "subseq_info should succeed");

    /* TR count and size for first subsequence */
    mu_assert(si.num_trs >= 1,
              "should have at least 1 TR");
    mu_assert(si.tr_size >= 1,
              "TR should contain at least 1 block");
    mu_assert(si.tr_duration_us > 0.0f,
              "TR duration should be positive");

    pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Once-in-middle: valid (bSSFP-like inner loop)                     */
/* ------------------------------------------------------------------ */

MU_TEST(test_structure_valid_once_in_middle)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_subseq_info si   = PULSEQLIB_SUBSEQ_INFO_INIT;
    int rc;

    rc = load_seq("data/11_multipass_valid_prep_cooldown.seq",
                  &coll, &diag, 0);
    mu_assert(rc != PULSEQLIB_ERR_INVALID_ONCE_FLAGS,
              "valid once-in-middle must not be rejected by once-flag check");
    if (PULSEQLIB_SUCCEEDED(rc)) {
        pulseqlib_get_subseq_info(coll, 0, &si);
        mu_assert(si.num_passes == 3,
                  "11: num_passes should be 3");
        mu_assert(si.num_prep_blocks == 1,
                  "11: num_prep_blocks should be 1");
        mu_assert(si.num_cooldown_blocks == 1,
                  "11: num_cooldown_blocks should be 1");
    }
    if (coll) pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Once-in-middle: invalid (non-identical periods)                   */
/* ------------------------------------------------------------------ */

MU_TEST(test_structure_invalid_once_in_middle)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    int rc;

    rc = load_seq("data/10_multi_tr_nonvalid_once_in_the_middle.seq",
                  &coll, &diag, 0);
    mu_assert(rc == PULSEQLIB_ERR_INVALID_ONCE_FLAGS,
              "non-identical once-in-middle should be rejected");
    if (coll) pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Once-at-boundary: existing valid cases still work                 */
/* ------------------------------------------------------------------ */

MU_TEST(test_structure_valid_once_boundary)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    int rc;

    rc = load_seq("data/03_multi_tr_valid_once.seq", &coll, &diag, 0);
    mu_assert(rc != PULSEQLIB_ERR_INVALID_ONCE_FLAGS,
              "once at boundary must not be rejected by once-flag check");
    if (coll) pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Multipass: prep only (no cooldown)                                */
/* ------------------------------------------------------------------ */

MU_TEST(test_structure_multipass_prep_only)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_subseq_info si   = PULSEQLIB_SUBSEQ_INFO_INIT;
    int rc;

    rc = load_seq("data/12_multipass_valid_prep_only.seq",
                  &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc),
              "12: prep-only multipass should load");
    if (PULSEQLIB_SUCCEEDED(rc)) {
        pulseqlib_get_subseq_info(coll, 0, &si);
        mu_assert(si.num_passes == 3,
                  "12: num_passes should be 3");
        mu_assert(si.num_prep_blocks == 1,
                  "12: num_prep_blocks should be 1");
        mu_assert(si.num_cooldown_blocks == 0,
                  "12: num_cooldown_blocks should be 0");
    }
    if (coll) pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Multipass: cooldown only (no prep)                                */
/* ------------------------------------------------------------------ */

MU_TEST(test_structure_multipass_cooldown_only)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_subseq_info si   = PULSEQLIB_SUBSEQ_INFO_INIT;
    int rc;

    rc = load_seq("data/13_multipass_valid_cooldown_only.seq",
                  &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc),
              "13: cooldown-only multipass should load");
    if (PULSEQLIB_SUCCEEDED(rc)) {
        pulseqlib_get_subseq_info(coll, 0, &si);
        mu_assert(si.num_passes == 3,
                  "13: num_passes should be 3");
        mu_assert(si.num_prep_blocks == 0,
                  "13: num_prep_blocks should be 0");
        mu_assert(si.num_cooldown_blocks == 1,
                  "13: num_cooldown_blocks should be 1");
    }
    if (coll) pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Multipass: trailing cooldown separation, 2-pass minimum           */
/* ------------------------------------------------------------------ */

MU_TEST(test_structure_multipass_trailing_cooldown)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_subseq_info si   = PULSEQLIB_SUBSEQ_INFO_INIT;
    int rc;

    rc = load_seq("data/14_multipass_valid_trailing_cooldown.seq",
                  &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc),
              "14: trailing-cooldown multipass should load");
    if (PULSEQLIB_SUCCEEDED(rc)) {
        pulseqlib_get_subseq_info(coll, 0, &si);
        mu_assert(si.num_passes == 2,
                  "14: num_passes should be 2");
        mu_assert(si.num_prep_blocks == 1,
                  "14: num_prep_blocks should be 1");
        mu_assert(si.num_cooldown_blocks == 1,
                  "14: num_cooldown_blocks should be 1");
    }
    if (coll) pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Multipass: multiple TRs per pass                                  */
/* ------------------------------------------------------------------ */

MU_TEST(test_structure_multipass_multi_tr)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_subseq_info si   = PULSEQLIB_SUBSEQ_INFO_INIT;
    int rc;

    rc = load_seq("data/15_multipass_valid_multi_tr.seq",
                  &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc),
              "15: multi-TR multipass should load");
    if (PULSEQLIB_SUCCEEDED(rc)) {
        pulseqlib_get_subseq_info(coll, 0, &si);
        mu_assert(si.num_passes == 3,
                  "15: num_passes should be 3");
        mu_assert(si.num_prep_blocks == 1,
                  "15: num_prep_blocks should be 1");
        mu_assert(si.num_cooldown_blocks == 1,
                  "15: num_cooldown_blocks should be 1");
        mu_assert(si.num_trs >= 2,
                  "15: should have at least 2 TRs per pass");
    }
    if (coll) pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Invalid multipass: different main blocks across passes            */
/* ------------------------------------------------------------------ */

MU_TEST(test_structure_multipass_fail_diff_main)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    int rc;

    rc = load_seq("data/16_multipass_fail_diff_main.seq",
                  &coll, &diag, 0);
    mu_assert(rc == PULSEQLIB_ERR_INVALID_ONCE_FLAGS,
              "16: different main blocks across passes should be rejected");
    if (coll) pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Invalid multipass: different pass lengths                         */
/* ------------------------------------------------------------------ */

MU_TEST(test_structure_multipass_fail_diff_length)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    int rc;

    rc = load_seq("data/17_multipass_fail_diff_length.seq",
                  &coll, &diag, 0);
    mu_assert(rc == PULSEQLIB_ERR_INVALID_ONCE_FLAGS,
              "17: different pass lengths should be rejected");
    if (coll) pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Suite                                                             */
/* ------------------------------------------------------------------ */

MU_TEST_SUITE(test_structure_suite)
{
    MU_RUN_TEST(test_structure_ok_basic);
    MU_RUN_TEST(test_structure_valid_once_in_middle);
    MU_RUN_TEST(test_structure_invalid_once_in_middle);
    MU_RUN_TEST(test_structure_valid_once_boundary);
    MU_RUN_TEST(test_structure_multipass_prep_only);
    MU_RUN_TEST(test_structure_multipass_cooldown_only);
    MU_RUN_TEST(test_structure_multipass_trailing_cooldown);
    MU_RUN_TEST(test_structure_multipass_multi_tr);
    MU_RUN_TEST(test_structure_multipass_fail_diff_main);
    MU_RUN_TEST(test_structure_multipass_fail_diff_length);
}

int test_structure_main(void)
{
    MU_RUN_SUITE(test_structure_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
