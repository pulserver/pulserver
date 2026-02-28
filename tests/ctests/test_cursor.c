/*
 * test_cursor.c -- block cursor / iterator tests.
 *
 * Tests:
 *   1. Cursor walks all blocks to completion.
 *   2. Mark / reset: after marking and advancing, reset returns to
 *      the marked position and re-delivers the same blocks.
 *   3. Block instance fields look reasonable (duration, rotation).
 *   4. Cursor info metadata is consistent.
 *
 * Requires: data/seq1.seq
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

    rc = load_seq("data/seq1.seq", &coll, &diag, 0);
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
/*  Mark / reset: re-delivers blocks after the mark                   */
/* ------------------------------------------------------------------ */

MU_TEST(test_cursor_mark_reset)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    int rc, total, after_reset;

    rc = load_seq("data/seq1.seq", &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load seq1");

    /* Count total blocks first */
    total = 0;
    while (pulseqlib_cursor_next(coll) == PULSEQLIB_CURSOR_BLOCK)
        total++;
    mu_assert(total > 0, "should have blocks");

    pulseqlib_collection_free(coll);

    /* Reload and test mark/reset */
    coll = NULL;
    rc = load_seq("data/seq1.seq", &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "reload seq1");

    /* Mark at start, walk 2 blocks, reset, walk to end */
    pulseqlib_cursor_mark(coll);

    rc = pulseqlib_cursor_next(coll);
    mu_assert(rc == PULSEQLIB_CURSOR_BLOCK, "next after mark");
    rc = pulseqlib_cursor_next(coll);
    mu_assert(rc == PULSEQLIB_CURSOR_BLOCK, "second next");

    pulseqlib_cursor_reset(coll);

    after_reset = 0;
    while (pulseqlib_cursor_next(coll) == PULSEQLIB_CURSOR_BLOCK)
        after_reset++;

    mu_assert(after_reset == total,
              "reset to start should deliver all blocks");

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

    rc = load_seq("data/seq1.seq", &coll, &diag, 0);
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

    /* rf_shim_id should be -1 (no shim) for typical sequences */
    mu_assert(inst.rf_shim_id == -1,
              "rf_shim_id should default to -1");

    pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Cursor info metadata is filled                                    */
/* ------------------------------------------------------------------ */

MU_TEST(test_cursor_info)
{
    pulseqlib_collection*  coll = NULL;
    pulseqlib_diagnostic   diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_cursor_info  ci;
    int rc;

    rc = load_seq("data/seq1.seq", &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load seq1");

    rc = pulseqlib_cursor_next(coll);
    mu_assert(rc == PULSEQLIB_CURSOR_BLOCK, "first next");

    rc = pulseqlib_cursor_get_info(coll, &ci);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "cursor_get_info");

    mu_assert(ci.subseq_idx >= 0, "subseq_idx >= 0");
    mu_assert(ci.segment_id >= 0, "segment_id >= 0");

    pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Suite                                                             */
/* ------------------------------------------------------------------ */

MU_TEST_SUITE(test_cursor_suite)
{
    MU_RUN_TEST(test_cursor_seq1_walk);
    MU_RUN_TEST(test_cursor_mark_reset);
    MU_RUN_TEST(test_cursor_instance_fields);
    MU_RUN_TEST(test_cursor_info);
}

int test_cursor_main(void)
{
    MU_RUN_SUITE(test_cursor_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
