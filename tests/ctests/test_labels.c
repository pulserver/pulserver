/*
 * test_labels.c -- ADC label table tests.
 *
 * Tests:
 *   1. parse_labels=0: label getters return NOT_IMPLEMENTED or empty.
 *   2. parse_labels=1: label limits have correct min/max ranges
 *      for a known sequence (e.g. 2D multi-slice GRE with
 *      128 lin, 30 slc -> lin min=0 max=127, slc min=0 max=29).
 *   3. num_adc_occurrences matches expected count (ntrs * adcs_per_tr).
 *   4. get_adc_label returns correct values per occurrence.
 *   5. Label columns for GEHC are 3 (lin, slc, eco).
 *
 * Requires:
 *   - expected_output/gre_2d_multislice.seq (multi-slice with labels)
 *
 * Until generated, seq1.seq is used for smoke test of parse_labels=0.
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

    rc = load_seq("expected_output/seq1.seq", &coll, &diag, 0);
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

    rc = load_seq("expected_output/seq1.seq", &coll, &diag, 1);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load seq1 with labels");

    rc = pulseqlib_get_subseq_info(coll, 0, &si);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "subseq_info");
    mu_assert(si.num_label_columns >= 0, "num_label_columns >= 0");
    mu_assert(si.num_adc_occurrences >= 0, "num_adc_occurrences >= 0");

    pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Stub: multi-slice GRE with known label ranges                     */
/* ------------------------------------------------------------------ */

/*
 * TODO: uncomment once gre_2d_multislice.seq is generated with
 * 128 phase-encode lines and 30 slices.
 *
 * MU_TEST(test_labels_gre_multislice)
 * {
 *     pulseqlib_collection* coll = NULL;
 *     pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
 *     pulseqlib_label_limits limits;
 *     int rc, ncols, nadc;
 *
 *     rc = load_seq("expected_output/gre_2d_multislice.seq",
 *                   &coll, &diag, 1);
 *     mu_assert(PULSEQLIB_SUCCEEDED(rc), "load multi-slice GRE");
 *
 *     ncols = pulseqlib_get_num_label_columns(coll, 0);
 *     mu_assert(ncols == 3, "GEHC should have 3 label columns");
 *
 *     rc = pulseqlib_get_label_limits(coll, 0, &limits);
 *     mu_assert(PULSEQLIB_SUCCEEDED(rc), "get_label_limits OK");
 *
 *     mu_assert_int_eq(0,   limits.lin.min);
 *     mu_assert_int_eq(127, limits.lin.max);
 *     mu_assert_int_eq(0,   limits.slc.min);
 *     mu_assert_int_eq(29,  limits.slc.max);
 *     mu_assert_int_eq(0,   limits.eco.min);
 *     mu_assert_int_eq(0,   limits.eco.max);
 *
 *     nadc = pulseqlib_get_num_adc_occurrences(coll, 0);
 *     mu_assert_int_eq(128 * 30, nadc);
 *
 *     // Spot-check first and last ADC label values
 *     {
 *         int vals[3];
 *         rc = pulseqlib_get_adc_label(coll, 0, 0, vals);
 *         mu_assert(PULSEQLIB_SUCCEEDED(rc), "get first label");
 *         mu_assert_int_eq(0, vals[0]);  // lin
 *         mu_assert_int_eq(0, vals[1]);  // slc
 *
 *         rc = pulseqlib_get_adc_label(coll, 0, nadc - 1, vals);
 *         mu_assert(PULSEQLIB_SUCCEEDED(rc), "get last label");
 *         mu_assert_int_eq(127, vals[0]); // lin
 *         mu_assert_int_eq(29,  vals[1]); // slc
 *     }
 *
 *     pulseqlib_collection_free(coll);
 * }
 */

/* ------------------------------------------------------------------ */
/*  Suite                                                             */
/* ------------------------------------------------------------------ */

MU_TEST_SUITE(test_labels_suite)
{
    MU_RUN_TEST(test_labels_disabled);
    MU_RUN_TEST(test_labels_enabled_smoke);
    /* MU_RUN_TEST(test_labels_gre_multislice); */
}

int test_labels_main(void)
{
    MU_RUN_SUITE(test_labels_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
