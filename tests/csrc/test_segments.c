/*
 * test_segments.c -- segment-level query tests.
 *
 * Tests:
 *   1. get_num_segments returns correct count for known sequences.
 *   2. get_segment_num_blocks matches expected values.
 *   3. get_segment_duration_us is positive for all segments.
 *   4. is_segment_pure_delay correctly identifies delay-only segments.
 *   5. Segment RF / ADC anchors are within block range.
 *   6. get_block_start_time_us is non-decreasing within a segment.
 *   7. get_block_duration_us is positive for all blocks.
 *   8. Block durations within a segment sum to segment duration.
 *
 * Requires:
 *   - expected_output/seq1.seq
 *   - expected_output/gre_2d.seq
 *   - expected_output/epi_2d.seq
 *   - expected_output/mprage_3d.seq
 *
 * Until generated, seq1.seq is used for smoke tests.
 */
#include "test_helpers.h"

/* ------------------------------------------------------------------ */
/*  Smoke: segment queries on seq1                                    */
/* ------------------------------------------------------------------ */

MU_TEST(test_segments_seq1_smoke)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    int rc, nseg, s;

    rc = load_seq("expected_output/seq1.seq", &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load seq1");

    nseg = pulseqlib_get_num_segments(coll);
    mu_assert(nseg > 0, "should have at least 1 segment");

    for (s = 0; s < nseg; s++) {
        int nblocks = pulseqlib_get_segment_num_blocks(coll, s);
        int dur     = pulseqlib_get_segment_duration_us(coll, s);

        mu_assert(nblocks > 0,
                  "each segment should have at least 1 block");
        mu_assert(dur > 0,
                  "segment duration should be positive");

        /* Block times non-decreasing, durations positive */
        if (nblocks > 0) {
            int b, prev_start;
            prev_start = pulseqlib_get_block_start_time_us(coll, s, 0);
            for (b = 0; b < nblocks; b++) {
                int bstart = pulseqlib_get_block_start_time_us(coll, s, b);
                int bdur   = pulseqlib_get_block_duration_us(coll, s, b);
                mu_assert(bstart >= prev_start,
                          "block start times should be non-decreasing");
                mu_assert(bdur > 0,
                          "block duration should be positive");
                prev_start = bstart;
            }
        }
    }

    pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Stub: segment anchors for GRE                                     */
/* ------------------------------------------------------------------ */

/*
 * TODO: uncomment once gre_2d.seq is generated.
 *
 * MU_TEST(test_segments_gre_anchors)
 * {
 *     pulseqlib_collection* coll = NULL;
 *     pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
 *     int rc, nseg, s;
 *
 *     rc = load_seq("expected_output/gre_2d.seq", &coll, &diag, 0);
 *     mu_assert(PULSEQLIB_SUCCEEDED(rc), "load GRE");
 *
 *     nseg = pulseqlib_get_num_segments(coll);
 *     for (s = 0; s < nseg; s++) {
 *         int nblocks = pulseqlib_get_segment_num_blocks(coll, s);
 *         int nrf     = pulseqlib_get_segment_num_rf_anchors(coll, s);
 *         int nadc    = pulseqlib_get_segment_num_adc_anchors(coll, s);
 *         int a;
 *
 *         // RF anchors should be valid block indices
 *         for (a = 0; a < nrf; a++) {
 *             int idx = pulseqlib_get_segment_rf_anchor(coll, s, a);
 *             mu_assert(idx >= 0 && idx < nblocks,
 *                       "RF anchor within block range");
 *         }
 *         // ADC anchors likewise
 *         for (a = 0; a < nadc; a++) {
 *             int idx = pulseqlib_get_segment_adc_anchor(coll, s, a);
 *             mu_assert(idx >= 0 && idx < nblocks,
 *                       "ADC anchor within block range");
 *         }
 *     }
 *
 *     pulseqlib_collection_free(coll);
 * }
 */

/* ------------------------------------------------------------------ */
/*  Suite                                                             */
/* ------------------------------------------------------------------ */

MU_TEST_SUITE(test_segments_suite)
{
    MU_RUN_TEST(test_segments_seq1_smoke);
    /* MU_RUN_TEST(test_segments_gre_anchors); */
}

int test_segments_main(void)
{
    MU_RUN_SUITE(test_segments_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
