/*
 * test_structure.c -- TR detection, prep / cooldown, once-flag tests.
 *
 * Tests:
 *   - seq1 loads and has structural metadata
 *   - Once-flag valid / invalid detection
 *
 * Requires: data/seq1.seq, data/once-flag test files
 */
#include "test_helpers.h"

/* ------------------------------------------------------------------ */
/*  Smoke: seq1 loads and has structural metadata                     */
/* ------------------------------------------------------------------ */

MU_TEST(test_structure_seq1_basic)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_collection_info ci = PULSEQLIB_COLLECTION_INFO_INIT;
    pulseqlib_subseq_info     si = PULSEQLIB_SUBSEQ_INFO_INIT;
    int rc;

    rc = load_seq("data/seq1.seq", &coll, &diag, 0);
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
    int rc;

    rc = load_seq("data/11_multi_tr_valid_once_in_the_middle.seq",
                  &coll, &diag, 0);
    mu_assert(rc != PULSEQLIB_ERR_INVALID_ONCE_FLAGS,
              "valid once-in-middle must not be rejected by once-flag check");
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
/*  Suite                                                             */
/* ------------------------------------------------------------------ */

MU_TEST_SUITE(test_structure_suite)
{
    MU_RUN_TEST(test_structure_seq1_basic);
    MU_RUN_TEST(test_structure_valid_once_in_middle);
    MU_RUN_TEST(test_structure_invalid_once_in_middle);
    MU_RUN_TEST(test_structure_valid_once_boundary);
}

int test_structure_main(void)
{
    MU_RUN_SUITE(test_structure_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
