/*
 * test_segments.c -- segment-level query tests.
 *
 * Tests:
 *   1. Segment count is positive for loaded sequences.
 *   2. Each segment has positive block count and duration.
 *   3. Block start times are non-decreasing within a segment.
 *   4. Block durations are positive.
 *
 * Requires: data/seq1.seq
 */
#include "test_helpers.h"

/* ------------------------------------------------------------------ */
/*  Smoke: segment queries on seq1                                    */
/* ------------------------------------------------------------------ */

MU_TEST(test_segments_seq1_smoke)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_collection_info ci = PULSEQLIB_COLLECTION_INFO_INIT;
    int rc, s;

    rc = load_seq("data/seq1.seq", &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load seq1");

    rc = pulseqlib_get_collection_info(coll, &ci);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "collection_info");
    mu_assert(ci.num_segments > 0, "should have at least 1 segment");

    for (s = 0; s < ci.num_segments; s++) {
        pulseqlib_segment_info segi = PULSEQLIB_SEGMENT_INFO_INIT;
        rc = pulseqlib_get_segment_info(coll, s, &segi);
        mu_assert(PULSEQLIB_SUCCEEDED(rc), "segment_info");

        mu_assert(segi.num_blocks > 0,
                  "each segment should have at least 1 block");
        mu_assert(segi.duration_us > 0,
                  "segment duration should be positive");

        /* Block times non-decreasing, durations positive */
        if (segi.num_blocks > 0) {
            int b, prev_start;
            pulseqlib_block_info bi0 = PULSEQLIB_BLOCK_INFO_INIT;
            pulseqlib_get_block_info(coll, s, 0, &bi0);
            prev_start = bi0.start_time_us;

            for (b = 0; b < segi.num_blocks; b++) {
                pulseqlib_block_info bi = PULSEQLIB_BLOCK_INFO_INIT;
                pulseqlib_get_block_info(coll, s, b, &bi);
                mu_assert(bi.start_time_us >= prev_start,
                          "block start times should be non-decreasing");
                mu_assert(bi.duration_us > 0,
                          "block duration should be positive");
                prev_start = bi.start_time_us;
            }
        }
    }

    pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Suite                                                             */
/* ------------------------------------------------------------------ */

MU_TEST_SUITE(test_segments_suite)
{
    MU_RUN_TEST(test_segments_seq1_smoke);
}

int test_segments_main(void)
{
    MU_RUN_SUITE(test_segments_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
