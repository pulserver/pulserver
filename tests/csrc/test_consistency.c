/*
 * test_consistency.c -- consistency checks: nonperiodic RF amp patterns,
 *                       RF shim ID patterns, once-flag rejection,
 *                       nonperiodic-sequence length limits.
 *
 * Tests:
 *   1. Consistent sequence passes check_consistency()
 *   2. Nonperiodic RF amplitude pattern -> ERR_CONSISTENCY_RF_PERIODIC
 *   3. Once-flags in middle of sequence -> ERR_INVALID_ONCE_FLAGS
 *   4. Nonperiodic sequence > 15 sec -> rejected at load time
 *
 * Requires:
 *   - expected_output/seq1.seq   (always available)
 *   - expected_output/bad_rf_periodic.seq   (nonperiodic RF amp)
 *   - expected_output/bad_once_flags.seq    (once flags in middle)
 *
 * Until generated, the tests use seq1.seq for the pass-path.
 */
#include "test_helpers.h"

/* ------------------------------------------------------------------ */
/*  Pass-path: seq1 should pass consistency                           */
/* ------------------------------------------------------------------ */

MU_TEST(test_consistency_seq1_passes)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    int rc;

    rc = load_seq("expected_output/seq1.seq", &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load seq1");

    pulseqlib_diagnostic_init(&diag);
    rc = pulseqlib_check_consistency(coll, &diag);
    mu_assert(PULSEQLIB_SUCCEEDED(rc),
              "seq1 should pass consistency check");

    pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Stub: nonperiodic RF amplitude pattern                            */
/* ------------------------------------------------------------------ */

/*
 * TODO: uncomment once bad_rf_periodic.seq is generated.
 * This sequence should have RF events whose amplitude pattern
 * does not repeat with the detected TR period.
 *
 * MU_TEST(test_consistency_nonperiodic_rf)
 * {
 *     pulseqlib_collection* coll = NULL;
 *     pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
 *     int rc;
 *
 *     rc = load_seq("expected_output/bad_rf_periodic.seq",
 *                   &coll, &diag, 0);
 *     // Load may succeed but consistency check should fail
 *     if (PULSEQLIB_FAILED(rc)) {
 *         mu_assert(rc == PULSEQLIB_ERR_CONSISTENCY_RF_PERIODIC,
 *                   "should be RF periodicity error");
 *     } else {
 *         pulseqlib_diagnostic_init(&diag);
 *         rc = pulseqlib_check_consistency(coll, &diag);
 *         mu_assert(rc == PULSEQLIB_ERR_CONSISTENCY_RF_PERIODIC,
 *                   "consistency should catch nonperiodic RF");
 *         pulseqlib_collection_free(coll);
 *     }
 * }
 */

/* ------------------------------------------------------------------ */
/*  Stub: once-flags in middle of sequence                            */
/* ------------------------------------------------------------------ */

/*
 * TODO: uncomment once bad_once_flags.seq is generated.
 * This sequence should have ONCE labels placed not at the start
 * or end of the block list.
 *
 * MU_TEST(test_consistency_bad_once_flags)
 * {
 *     pulseqlib_collection* coll = NULL;
 *     pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
 *     int rc;
 *
 *     rc = load_seq("expected_output/bad_once_flags.seq",
 *                   &coll, &diag, 0);
 *     mu_assert(rc == PULSEQLIB_ERR_INVALID_ONCE_FLAGS,
 *               "once flags in middle should be rejected at load");
 *     if (coll) pulseqlib_collection_free(coll);
 * }
 */

/* ------------------------------------------------------------------ */
/*  Suite                                                             */
/* ------------------------------------------------------------------ */

MU_TEST_SUITE(test_consistency_suite)
{
    MU_RUN_TEST(test_consistency_seq1_passes);
    /* MU_RUN_TEST(test_consistency_nonperiodic_rf); */
    /* MU_RUN_TEST(test_consistency_bad_once_flags); */
}

int test_consistency_main(void)
{
    MU_RUN_SUITE(test_consistency_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
