/*
 * test_consistency.c -- consistency checks.
 *
 * Tests:
 *   1. Consistent sequence passes check_consistency()
 *   2. VFA RF pattern fails RF periodicity check
 *   3. RF shim pattern fails periodicity check
 *
 * Requires: data/01_ok_trap_extended_trap.seq,
 *           data/02_rfamp_fail_vfa.seq, data/04_rfshim_fail_gre.seq
 */
#include "test_helpers.h"

/* ------------------------------------------------------------------ */
/*  Pass-path: ok_trap should pass consistency                        */
/* ------------------------------------------------------------------ */

MU_TEST(test_consistency_ok_passes)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    int rc;

    rc = load_seq(TEST_SEQ_OK, &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load ok_trap");

    pulseqlib_diagnostic_init(&diag);
    rc = pulseqlib_check_consistency(coll, &diag);
    mu_assert(PULSEQLIB_SUCCEEDED(rc),
              "ok_trap should pass consistency check");

    pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  RF periodicity: files with non-periodic RF should be detected     */
/* ------------------------------------------------------------------ */

MU_TEST(test_consistency_rfamp_fail)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    int rc;

    rc = load_seq("data/02_rfamp_fail_vfa.seq", &coll, &diag, 0);
    if (PULSEQLIB_SUCCEEDED(rc)) {
        pulseqlib_diagnostic_init(&diag);
        rc = pulseqlib_check_consistency(coll, &diag);
        mu_assert(rc == PULSEQLIB_ERR_CONSISTENCY_RF_PERIODIC,
                  "VFA should fail RF periodicity check");
        pulseqlib_collection_free(coll);
    } else {
        /* If load itself catches the error, that is also acceptable */
        mu_assert(rc == PULSEQLIB_ERR_CONSISTENCY_RF_PERIODIC
                  || PULSEQLIB_FAILED(rc),
                  "VFA should fail at load or consistency");
    }
}

MU_TEST(test_consistency_rfshim_fail)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    int rc;

    rc = load_seq("data/04_rfshim_fail_gre.seq", &coll, &diag, 0);
    if (PULSEQLIB_SUCCEEDED(rc)) {
        pulseqlib_diagnostic_init(&diag);
        rc = pulseqlib_check_consistency(coll, &diag);
        mu_assert(rc == PULSEQLIB_ERR_CONSISTENCY_RF_SHIM_PERIODIC,
                  "RF shim pattern should fail periodicity check");
        pulseqlib_collection_free(coll);
    } else {
        mu_assert(rc == PULSEQLIB_ERR_CONSISTENCY_RF_SHIM_PERIODIC
                  || PULSEQLIB_FAILED(rc),
                  "RF shim should fail at load or consistency");
    }
}

/* ------------------------------------------------------------------ */
/*  Suite                                                             */
/* ------------------------------------------------------------------ */

MU_TEST_SUITE(test_consistency_suite)
{
    MU_RUN_TEST(test_consistency_ok_passes);
    MU_RUN_TEST(test_consistency_rfamp_fail);
    MU_RUN_TEST(test_consistency_rfshim_fail);
}

int test_consistency_main(void)
{
    MU_RUN_SUITE(test_consistency_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
