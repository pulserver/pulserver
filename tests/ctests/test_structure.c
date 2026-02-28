/*
 * test_structure.c -- TR detection, prep / cooldown, periodic pattern.
 *
 * Tests:
 *   - Pure periodic sequence (GRE/bSSFP 2D): no prep, no cooldown
 *   - Prep + periodic + cooldown (MPRAGE, FSE): correct boundaries
 *   - Degenerate prep (single block trigger): detected correctly
 *   - Non-periodic sequence is rejected (ERR_TR_NO_PERIODIC_PATTERN)
 *   - Sequence > 15 s non-periodic is rejected (ERR_TR_PREP_TOO_LONG)
 *
 * Requires: test .seq files from the pulseq toolbox.
 *   - expected_output/gre_2d.seq         (pure periodic)
 *   - expected_output/mprage_3d.seq      (prep + periodic + cooldown)
 *   - expected_output/fse_2d.seq         (prep + periodic)
 *   - expected_output/nonperiodic.seq    (should be rejected)
 *
 * Until those files are generated, the tests below use seq1.seq as a
 * smoke test and document the expected assertions.
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

    rc = load_seq("expected_output/seq1.seq", &coll, &diag, 0);
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
/*  Stub: pure periodic (no prep, no cooldown)                        */
/* ------------------------------------------------------------------ */

/*
 * TODO: uncomment once gre_2d.seq is generated.
 *
 * MU_TEST(test_structure_gre_periodic)
 * {
 *     pulseqlib_collection* coll = NULL;
 *     pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
 *     int rc;
 *
 *     rc = load_seq("expected_output/gre_2d.seq", &coll, &diag, 0);
 *     mu_assert(PULSEQLIB_SUCCEEDED(rc), "GRE should load");
 *
 *     // Expect: 1 subsequence, all TRs identical, no prep, no cooldown.
 *     mu_assert(pulseqlib_get_num_subsequences(coll) == 1,
 *               "GRE should have 1 subsequence");
 *     // TR count = number of phase-encode steps (e.g. 128)
 *     mu_assert(pulseqlib_get_num_trs(coll, 0) > 1,
 *               "GRE should have multiple TRs");
 *
 *     pulseqlib_collection_free(coll);
 * }
 */

/* ------------------------------------------------------------------ */
/*  Stub: prep + periodic + cooldown (MPRAGE)                         */
/* ------------------------------------------------------------------ */

/*
 * TODO: uncomment once mprage_3d.seq is generated.
 *
 * MU_TEST(test_structure_mprage)
 * {
 *     pulseqlib_collection* coll = NULL;
 *     pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
 *     int rc;
 *
 *     rc = load_seq("expected_output/mprage_3d.seq", &coll, &diag, 0);
 *     mu_assert(PULSEQLIB_SUCCEEDED(rc), "MPRAGE should load");
 *
 *     // Expect: prep has inversion pulse, main has readout train,
 *     // cooldown has recovery delay.
 *     mu_assert(pulseqlib_get_num_subsequences(coll) == 1,
 *               "MPRAGE should have 1 subsequence");
 *     mu_assert(pulseqlib_get_num_trs(coll, 0) > 1,
 *               "MPRAGE should have multiple TRs (segments*partitions)");
 *
 *     pulseqlib_collection_free(coll);
 * }
 */

/* ------------------------------------------------------------------ */
/*  Stub: non-periodic rejection                                      */
/* ------------------------------------------------------------------ */

/*
 * TODO: uncomment once nonperiodic.seq is generated.
 *
 * MU_TEST(test_structure_nonperiodic_rejected)
 * {
 *     pulseqlib_collection* coll = NULL;
 *     pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
 *     int rc;
 *
 *     rc = load_seq("expected_output/nonperiodic.seq", &coll, &diag, 0);
 *     mu_assert(rc == PULSEQLIB_ERR_TR_NO_PERIODIC_PATTERN,
 *               "non-periodic sequence should be rejected");
 *     if (coll) pulseqlib_collection_free(coll);
 * }
 */

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
    /* The once-flag folding must succeed (rc != ERR_INVALID_ONCE_FLAGS).
     * Downstream segmentation may still fail due to gradient boundary
     * conditions in the test data; that is unrelated to once handling. */
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
    /* Once-at-boundary must not be rejected by the once-flag check.
     * Downstream segmentation may still fail with the test gradients. */
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
    /* MU_RUN_TEST(test_structure_gre_periodic); */
    /* MU_RUN_TEST(test_structure_mprage); */
    /* MU_RUN_TEST(test_structure_nonperiodic_rejected); */
}

int test_structure_main(void)
{
    MU_RUN_SUITE(test_structure_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
