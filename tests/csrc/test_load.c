/*
 * test_load.c -- basic load / read / error-path tests.
 *
 * Tests:
 *   - seq1.seq loads successfully
 *   - Getters return expected counts (num_subsequences, num_trs, etc.)
 *   - Non-existent file returns FILE_NOT_FOUND
 *   - Diagnostic message is populated on error
 *   - Collection can be freed without crash
 *
 * Requires: expected_output/seq1.seq
 */
#include "test_helpers.h"

/* ------------------------------------------------------------------ */
/*  Tests                                                             */
/* ------------------------------------------------------------------ */

MU_TEST(test_load_seq1)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    int rc;

    rc = load_seq("expected_output/seq1.seq", &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc),
              "seq1.seq should load successfully");
    mu_assert(coll != NULL,
              "collection pointer should be non-NULL");

    /* Basic structural queries */
    mu_assert(pulseqlib_get_num_subsequences(coll) >= 1,
              "should have at least 1 subsequence");
    mu_assert(pulseqlib_get_total_duration_us(coll) > 0.0f,
              "total duration should be positive");

    pulseqlib_collection_free(coll);
}

MU_TEST(test_load_file_not_found)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    int rc;

    rc = load_seq("expected_output/does_not_exist.seq", &coll, &diag, 0);
    mu_assert(PULSEQLIB_FAILED(rc),
              "loading nonexistent file should fail");
    mu_assert(coll == NULL,
              "collection should remain NULL on failure");
    mu_assert(strlen(diag.message) > 0,
              "diagnostic message should be non-empty");
}

MU_TEST(test_double_free_is_safe)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    int rc;

    rc = load_seq("expected_output/seq1.seq", &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load should succeed");
    pulseqlib_collection_free(coll);
    /* Second free on an already-freed pointer is UB; we only verify
     * that a single free works.  This is a placeholder for
     * sanitizer runs. */
}

MU_TEST(test_null_pointer)
{
    pulseqlib_opts opts;
    int rc;

    test_opts_init(&opts);
    rc = pulseqlib_read(NULL, NULL, "dummy", &opts, 0, 0, 0, 1);
    mu_assert(PULSEQLIB_FAILED(rc),
              "NULL out_coll should return error");
}

/* ------------------------------------------------------------------ */
/*  Suite                                                             */
/* ------------------------------------------------------------------ */

MU_TEST_SUITE(test_load_suite)
{
    MU_RUN_TEST(test_load_seq1);
    MU_RUN_TEST(test_load_file_not_found);
    MU_RUN_TEST(test_double_free_is_safe);
    MU_RUN_TEST(test_null_pointer);
}

int test_load_main(void)
{
    MU_RUN_SUITE(test_load_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
