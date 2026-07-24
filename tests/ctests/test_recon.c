/*
 * test_recon.c -- Stage 4 (recon-cache-to-C) tests for pulseg_recon.c.
 *
 * Exercises pulseg_recon_cache_read() directly (no C++ wrapper involved),
 * against real fixtures in tests/utils/expected/, and checks the mandatory-
 * section contract (TRAJECTORY/SEQDESC absent -> hard error, never a
 * silent degrade -- see pulseg_recon.h).
 *
 * Expected field values below were cross-checked against
 * tests/utils/expected_recon_dumps/mprage_2d_1sl_1avg.dump (produced by
 * the C++ reader before AND after the C port -- byte-identical).
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "minunit.h"
#include "pulseg_recon.h"

#define TEST_DATA_DIR TEST_ROOT_DIR "/tests/utils/expected/"

/* ================================================================== */
/*  Test: read a real fixture and spot-check known field values       */
/* ================================================================== */

MU_TEST(test_recon_read_mprage)
{
    pulseg_recon_cache cache;
    char diag[256];
    int rc;
    const pulseq_definition *def;

    rc =
        pulseg_recon_cache_read(&cache, TEST_DATA_DIR "mprage_2d_1sl_1avg.pge", diag, sizeof(diag));
    mu_assert_int_eq(PULSEG_SUCCESS, rc);

    /* Section 0 (DEFINITIONS) */
    mu_assert_int_eq(1, cache.num_subsequences);
    def = pulseg_recon_find_definition(&cache, 0, "TR");
    mu_assert(def != NULL, "TR definition should be present");
    mu_assert(def->value_size >= 1, "TR should have at least one value");
    mu_assert(
        pulseg_recon_find_definition(&cache, 0, "NoSuchKey") == NULL,
        "unknown key should return NULL");
    mu_assert(
        pulseg_recon_find_definition(&cache, 5, "TR") == NULL,
        "out-of-range subseq_idx should return NULL");

    /* Section 6 (TRAJECTORY) -- delegated to pulseg_load_trajectory_cache */
    mu_assert_int_eq(0, cache.trajectory.kshots.num_shots);
    mu_assert_int_eq(1, cache.trajectory.num_encoding_spaces);
    mu_assert_int_eq(8, cache.trajectory.num_adc_events);
    mu_assert_int_eq(0, cache.trajectory.num_rotations);

    /* Section 7 (SEQDESC) */
    mu_assert_int_eq(1, cache.seq_params.num_subseqs);
    mu_assert(fabsf(cache.seq_params.min_tr_us - 75200.0f) < 1.0f, "min_tr_us");
    mu_assert(fabsf(cache.seq_params.max_flip_angle_deg - 180.0f) < 1e-3f, "max_flip_angle_deg");
    mu_assert_int_eq(1, cache.num_seq_descs);
    mu_assert_int_eq(36, cache.seq_descs[0].num_events);
    mu_assert_int_eq(0, cache.seq_descs[0].subseq_idx);

    pulseg_recon_cache_free(&cache);
    /* Safe to free twice: internal pointers are nulled after release. */
    pulseg_recon_cache_free(&cache);
}

/* ================================================================== */
/*  Test: TRAJECTORY/SEQDESC are mandatory -- no silent degrade        */
/* ================================================================== */

MU_TEST(test_recon_missing_trajectory_is_hard_error)
{
    pulseg_recon_cache cache;
    char diag[256];
    int rc;

    /* tests/ctests/fixtures/missing_trajectory.pge is a preserved
     * pre-Stage-4 cache artifact with no TRAJECTORY section at all (it used
     * to be 03_grad_rss_violation.pge itself, before that fixture's stale
     * .seq [SIGNATURE] was fixed and it started regenerating as a complete
     * cache like every other tests/utils/expected/ fixture -- see
     * tests/cpptests/dump_recon_cache.cpp for the full story). Kept outside
     * tests/utils/expected/ so it is never swept up by the cpptests
     * AllFixtures test suites' auto-discovery, which expects every .pge
     * there to be a complete, current-format cache. Reading it must fail loudly, not
     * return an empty cache -- that silent-degrade behavior belonged to the
     * pre-Stage-4 C++ reader and was deliberately removed. */
    memset(&cache, 0, sizeof(cache));
    diag[0] = '\0';
    rc = pulseg_recon_cache_read(
        &cache,
        TEST_ROOT_DIR "/tests/ctests/fixtures/missing_trajectory.pge",
        diag,
        sizeof(diag));
    mu_assert(rc != PULSEG_SUCCESS, "missing TRAJECTORY must fail, not degrade");
    mu_assert(diag[0] != '\0', "diag message should be set on failure");

    /* Nonexistent file: distinct error, still a failure. */
    rc = pulseg_recon_cache_read(&cache, TEST_DATA_DIR "does_not_exist.pge", diag, sizeof(diag));
    mu_assert(rc != PULSEG_SUCCESS, "nonexistent file must fail");
}

/* ================================================================== */
/*  Suite setup                                                       */
/* ================================================================== */

MU_TEST_SUITE(recon_suite)
{
    MU_RUN_TEST(test_recon_read_mprage);
    MU_RUN_TEST(test_recon_missing_trajectory_is_hard_error);
}

int test_recon_main(void)
{
    MU_RUN_SUITE(recon_suite);
    MU_REPORT();
    return minunit_fail;
}
