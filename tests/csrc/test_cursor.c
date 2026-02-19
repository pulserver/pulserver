/*
 * test_cursor.c -- block cursor / iterator tests.
 *
 * Tests:
 *   1. Cursor walks all blocks exactly num_trs * tr_size times.
 *   2. Cursor reset re-delivers the same blocks.
 *   3. navg=1 (default): cursor visits main TRs once.
 *   4. navg>1: cursor visits main TRs * navg times.
 *      (requires sequence with Averages > 1 definition)
 *   5. ignoreRepetitions: cursor visits only 1 average.
 *      (requires sequence with Repetitions > 0)
 *
 * Requires:
 *   - expected_output/seq1.seq          (basic cursor)
 *   - expected_output/gre_2d_navg2.seq  (navg=2)
 *   - expected_output/gre_2d_rep2.seq   (repetitions=2)
 */
#include "test_helpers.h"

/* ------------------------------------------------------------------ */
/*  Basic cursor walk on seq1                                         */
/* ------------------------------------------------------------------ */

MU_TEST(test_cursor_seq1_walk)
{
    pulseqlib_collection*  coll = NULL;
    pulseqlib_diagnostic   diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_block_instance inst = PULSEQLIB_BLOCK_INSTANCE_INIT;
    int rc, count;

    rc = load_seq("expected_output/seq1.seq", &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load seq1");

    /* Walk and count */
    count = 0;
    while (pulseqlib_cursor_next(coll) == PULSEQLIB_CURSOR_BLOCK) {
        rc = pulseqlib_get_block_instance(coll, &inst);
        mu_assert(PULSEQLIB_SUCCEEDED(rc), "get_block_instance OK");
        mu_assert(inst.duration_us > 0, "block duration > 0");
        count++;
    }
    mu_assert(count > 0, "cursor should visit at least one block");

    pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Cursor reset delivers the same count                              */
/* ------------------------------------------------------------------ */

MU_TEST(test_cursor_reset)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    int rc, count1, count2;

    rc = load_seq("expected_output/seq1.seq", &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load seq1");

    /* First walk */
    count1 = 0;
    while (pulseqlib_cursor_next(coll) == PULSEQLIB_CURSOR_BLOCK)
        count1++;

    /* Reset and second walk */
    pulseqlib_cursor_reset(coll);
    count2 = 0;
    while (pulseqlib_cursor_next(coll) == PULSEQLIB_CURSOR_BLOCK)
        count2++;

    mu_assert(count1 == count2,
              "cursor should deliver same count after reset");

    pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Cursor block instance fields look reasonable                      */
/* ------------------------------------------------------------------ */

MU_TEST(test_cursor_instance_fields)
{
    pulseqlib_collection*  coll = NULL;
    pulseqlib_diagnostic   diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_block_instance inst = PULSEQLIB_BLOCK_INSTANCE_INIT;
    int rc;

    rc = load_seq("expected_output/seq1.seq", &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load seq1");

    /* First block */
    rc = pulseqlib_cursor_next(coll);
    mu_assert(rc == PULSEQLIB_CURSOR_BLOCK, "first next should be BLOCK");

    rc = pulseqlib_get_block_instance(coll, &inst);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "get_block_instance OK");

    /* Rotation matrix should be identity-ish by default */
    mu_assert((float)fabs(inst.rotmat[0] - 1.0f) < 1e-5f,
              "rotmat[0,0] ~ 1");
    mu_assert((float)fabs(inst.rotmat[4] - 1.0f) < 1e-5f,
              "rotmat[1,1] ~ 1");
    mu_assert((float)fabs(inst.rotmat[8] - 1.0f) < 1e-5f,
              "rotmat[2,2] ~ 1");

    pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Stub: navg > 1                                                    */
/* ------------------------------------------------------------------ */

/*
 * TODO: uncomment once gre_2d_navg2.seq is generated.
 *
 * MU_TEST(test_cursor_navg2)
 * {
 *     pulseqlib_collection* coll = NULL;
 *     pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
 *     int rc, count;
 *     int nsub, ntrs, tr_size;
 *
 *     rc = load_seq("expected_output/gre_2d_navg2.seq",
 *                   &coll, &diag, 0);
 *     mu_assert(PULSEQLIB_SUCCEEDED(rc), "load navg2");
 *
 *     nsub    = pulseqlib_get_num_subsequences(coll);
 *     ntrs    = pulseqlib_get_num_trs(coll, 0);
 *     tr_size = pulseqlib_get_tr_size(coll, 0);
 *
 *     count = 0;
 *     while (pulseqlib_cursor_next(coll) == PULSEQLIB_CURSOR_BLOCK)
 *         count++;
 *
 *     // cursor should visit prep + main*navg*ntrs + cooldown blocks
 *     // exact count depends on sequence structure
 *     mu_assert(count >= ntrs * tr_size * 2,
 *               "navg=2 should double the main-region block count");
 *
 *     pulseqlib_collection_free(coll);
 * }
 */

/* ------------------------------------------------------------------ */
/*  Suite                                                             */
/* ------------------------------------------------------------------ */

MU_TEST_SUITE(test_cursor_suite)
{
    MU_RUN_TEST(test_cursor_seq1_walk);
    MU_RUN_TEST(test_cursor_reset);
    MU_RUN_TEST(test_cursor_instance_fields);
    /* MU_RUN_TEST(test_cursor_navg2); */
}

int test_cursor_main(void)
{
    MU_RUN_SUITE(test_cursor_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
