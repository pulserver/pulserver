/*
 * test_labels.c -- ADC label table tests.
 *
 * Tests:
 *   1. parse_labels=0: label getters return empty / error.
 *   2. parse_labels=1: smoke test on seq1.
 *
 * Requires: data/seq1.seq
 */
#include "test_helpers.h"

/* ------------------------------------------------------------------ */
/*  parse_labels=0: label getters should fail or return empty         */
/* ------------------------------------------------------------------ */

MU_TEST(test_labels_disabled)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_label_limits limits;
    int rc;

    rc = load_seq("data/seq1.seq", &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load seq1 without labels");

    rc = pulseqlib_get_label_limits(coll, 0, &limits);
    /* When labels were not parsed, this should fail or return
     * an error code indicating no label data. */
    mu_assert(PULSEQLIB_FAILED(rc)
              || (limits.lin.min == 0 && limits.lin.max == 0),
              "no label data when parse_labels=0");

    pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  parse_labels=1: smoke test on seq1                                */
/* ------------------------------------------------------------------ */

MU_TEST(test_labels_enabled_smoke)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_subseq_info si   = PULSEQLIB_SUBSEQ_INFO_INIT;
    int rc;

    rc = load_seq("data/seq1.seq", &coll, &diag, 1);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load seq1 with labels");

    rc = pulseqlib_get_subseq_info(coll, 0, &si);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "subseq_info");
    mu_assert(si.num_label_columns >= 0, "num_label_columns >= 0");
    mu_assert(si.num_adc_occurrences >= 0, "num_adc_occurrences >= 0");

    pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Suite                                                             */
/* ------------------------------------------------------------------ */

MU_TEST_SUITE(test_labels_suite)
{
    MU_RUN_TEST(test_labels_disabled);
    MU_RUN_TEST(test_labels_enabled_smoke);
}

int test_labels_main(void)
{
    MU_RUN_SUITE(test_labels_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
