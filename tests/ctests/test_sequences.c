/*
 * test_sequences.c -- the C lane's smoke over the collection API.
 *
 * Deep content verification (segment definitions, scan table, canonical TR
 * waveforms, label columns) lives in tests/python/test_pulseg_vs_oracle.py,
 * which holds this same C code against the spec-first Python oracle in one
 * process. What stays here is what only the standalone C89 build can check:
 * the load path and getter contracts compile and behave, the structural
 * invariants that need no external reference, and the collection cursor.
 */
#include "test_helpers.h"

#include <math.h>

/* ================================================================== */
/*  Load + invariants over the zoo corpus                              */
/* ================================================================== */

static const char *kSmokeFixtures[] = {
    "gre_2d.seq",
    "gre_2d_3sl.seq",
    "mprage_3d.seq",
    "fse_2d.seq",
    "bssfp_2d.seq",
    "gre_radial_2d.seq",
    "gre_stack_of_spirals_3d.seq",
    "zte_3d.seq",
};

static void run_smoke_case(const char *seq_file)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    pulseg_collection_info cinfo = PULSEG_COLLECTION_INFO_INIT;
    pulseg_subseq_info sinfo = PULSEG_SUBSEQ_INFO_INIT;
    pulseg_segment_info segi = PULSEG_SEGMENT_INFO_INIT;
    pulseg_cursor_info ci = PULSEG_CURSOR_INFO_INIT;
    pulseg_block_instance inst = PULSEG_BLOCK_INSTANCE_INIT;
    float walked_duration_us = 0.0f;
    int rc, s, num_stream_blocks = 0;

    gre_opts_init(&opts);
    rc = load_corpus_seq(&coll, seq_file, &opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load failed");

    rc = pulseg_get_collection_info(coll, &cinfo);
    mu_assert(PULSEG_SUCCEEDED(rc), "collection info failed");
    mu_assert_int_eq(1, cinfo.num_subsequences);
    mu_assert(cinfo.num_segments > 0, "no segments detected");
    mu_assert(cinfo.total_duration_us > 0.0f, "no duration");

    rc = pulseg_get_subseq_info(coll, &sinfo, 0);
    mu_assert(PULSEG_SUCCEEDED(rc), "subseq info failed");
    mu_assert(sinfo.tr_size >= 1, "tr_size must be at least 1");
    mu_assert(sinfo.num_trs >= 1, "num_trs must be at least 1");
    mu_assert(sinfo.tr_duration_us > 0.0f, "tr duration must be positive");
    mu_assert_int_eq(sinfo.num_trs, sinfo.num_tr_instances);
    mu_assert(sinfo.num_unique_rf > 0, "corpus fixtures all carry RF");

    /* Every segment the collection reports must resolve. */
    for (s = 0; s < cinfo.num_segments; ++s)
    {
        rc = pulseg_get_segment_info(coll, &segi, s);
        mu_assert(PULSEG_SUCCEEDED(rc) && segi.num_blocks > 0, "segment must resolve");
    }

    /* The cursor walks the whole execution stream, in segment-id range,
     * and the per-instance durations sum to the collection total. */
    pulseg_cursor_reset(coll);
    while (pulseg_cursor_next(coll) == PULSEG_CURSOR_BLOCK)
    {
        rc = pulseg_cursor_get_info(coll, &ci);
        mu_assert(PULSEG_SUCCEEDED(rc), "cursor info failed");
        mu_assert(
            ci.segment_id >= 0 && ci.segment_id < cinfo.num_segments,
            "cursor segment id out of range");
        rc = pulseg_get_block_instance(coll, &inst);
        mu_assert(PULSEG_SUCCEEDED(rc), "block instance failed");
        walked_duration_us += (float)inst.duration_us;
        ++num_stream_blocks;
    }
    mu_assert(num_stream_blocks > 0, "cursor visited no blocks");
    mu_assert_float_near(
        "walked vs reported total duration",
        cinfo.total_duration_us,
        walked_duration_us,
        1.0f);

    pulseg_collection_free(coll);
}

MU_TEST(test_smoke_gre_2d)
{
    run_smoke_case(kSmokeFixtures[0]);
}
MU_TEST(test_smoke_gre_2d_3sl)
{
    run_smoke_case(kSmokeFixtures[1]);
}
MU_TEST(test_smoke_mprage_3d)
{
    run_smoke_case(kSmokeFixtures[2]);
}
MU_TEST(test_smoke_fse_2d)
{
    run_smoke_case(kSmokeFixtures[3]);
}
MU_TEST(test_smoke_bssfp_2d)
{
    run_smoke_case(kSmokeFixtures[4]);
}
MU_TEST(test_smoke_gre_radial_2d)
{
    run_smoke_case(kSmokeFixtures[5]);
}
MU_TEST(test_smoke_gre_stack_of_spirals_3d)
{
    run_smoke_case(kSmokeFixtures[6]);
}
MU_TEST(test_smoke_zte_3d)
{
    run_smoke_case(kSmokeFixtures[7]);
}

MU_TEST_SUITE(suite_sequences_smoke)
{
    MU_RUN_TEST(test_smoke_gre_2d);
    MU_RUN_TEST(test_smoke_gre_2d_3sl);
    MU_RUN_TEST(test_smoke_mprage_3d);
    MU_RUN_TEST(test_smoke_fse_2d);
    MU_RUN_TEST(test_smoke_bssfp_2d);
    MU_RUN_TEST(test_smoke_gre_radial_2d);
    MU_RUN_TEST(test_smoke_gre_stack_of_spirals_3d);
    MU_RUN_TEST(test_smoke_zte_3d);
}

/* ================================================================== */
/*  NextSequence collection: two subsequences, one cursor              */
/* ================================================================== */

MU_TEST(test_collection_epi_chain)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    pulseg_collection_info cinfo = PULSEG_COLLECTION_INFO_INIT;
    pulseg_subseq_info lead_info = PULSEG_SUBSEQ_INFO_INIT;
    pulseg_subseq_info main_info = PULSEG_SUBSEQ_INFO_INIT;
    pulseg_cursor_info ci = PULSEG_CURSOR_INFO_INIT;
    pulseg_block_instance inst = PULSEG_BLOCK_INSTANCE_INIT;
    float walked_duration_us = 0.0f;
    int rc;
    int prev_subseq = -1;
    int saw_transition = 0;
    int num_blocks = 0;

    gre_opts_init(&opts);
    rc = load_corpus_seq(&coll, "epi_2d.seq", &opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "chain load failed");

    rc = pulseg_get_collection_info(coll, &cinfo);
    mu_assert(PULSEG_SUCCEEDED(rc), "collection info failed");
    mu_assert_int_eq(2, cinfo.num_subsequences);

    rc = pulseg_get_subseq_info(coll, &lead_info, 0);
    mu_assert(PULSEG_SUCCEEDED(rc), "lead subseq info failed");
    rc = pulseg_get_subseq_info(coll, &main_info, 1);
    mu_assert(PULSEG_SUCCEEDED(rc), "main subseq info failed");

    mu_assert_int_eq(0, lead_info.segment_offset);
    mu_assert(
        main_info.segment_offset > lead_info.segment_offset,
        "the chained subsequence's segments follow the lead's");

    pulseg_cursor_reset(coll);
    while (pulseg_cursor_next(coll) == PULSEG_CURSOR_BLOCK)
    {
        rc = pulseg_cursor_get_info(coll, &ci);
        mu_assert(PULSEG_SUCCEEDED(rc), "cursor info failed");
        mu_assert(ci.subseq_idx >= 0 && ci.subseq_idx < 2, "subseq index out of range");
        if (prev_subseq == 0 && ci.subseq_idx == 1)
            saw_transition = 1;
        mu_assert(
            prev_subseq <= ci.subseq_idx,
            "cursor subsequence index should be monotonic across the chain");
        prev_subseq = ci.subseq_idx;
        rc = pulseg_get_block_instance(coll, &inst);
        mu_assert(PULSEG_SUCCEEDED(rc), "block instance failed");
        walked_duration_us += (float)inst.duration_us;
        ++num_blocks;
    }
    mu_assert(num_blocks > 0, "chain cursor did not visit any blocks");
    mu_assert(saw_transition, "cursor never crossed into the chained subsequence");
    mu_assert_float_near(
        "walked vs reported chain duration",
        cinfo.total_duration_us,
        walked_duration_us,
        1.0f);

    pulseg_collection_free(coll);
}

MU_TEST_SUITE(suite_sequences_collection)
{
    MU_RUN_TEST(test_collection_epi_chain);
}

/* ================================================================== */
/*  Interior-delay classification                                      */
/*                                                                     */
/*  Both fixtures: 3 TRs of [RF, delay, ADC] + [TR-tail delay].       */
/*  TR1 == TR2 always; TR3's interior delay matches TR1/TR2 in the     */
/*  static fixture and differs in the dynamic one.  Confirms: (1) the  */
/*  delay-flex dedup collapses all 3 TR instances into ONE unique      */
/*  [RF, delay, ADC] segment regardless, and (2) is_variable_delay is  */
/*  0 for the never-varying delay and 1 only when the duration          */
/*  actually differs across instances.                                 */
/* ================================================================== */

static int find_delay_block(const pulseg_collection *coll, int seg_idx, int num_blocks)
{
    pulseg_block_info bi = PULSEG_BLOCK_INFO_INIT;
    int blk, rc;

    for (blk = 0; blk < num_blocks; ++blk)
    {
        rc = pulseg_get_block_info(coll, &bi, seg_idx, blk);
        if (!PULSEG_SUCCEEDED(rc))
            continue;
        if (!bi.has_rf && !bi.has_adc && !bi.has_grad[0] && !bi.has_grad[1] && !bi.has_grad[2])
            return blk;
    }
    return -1;
}

static void run_delay_classification_case(const char *seq_file, int expect_variable)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    pulseg_segment_info segi = PULSEG_SEGMENT_INFO_INIT;
    pulseg_block_info bi = PULSEG_BLOCK_INFO_INIT;
    int rc, target_seg, delay_blk;

    default_opts_init(&opts);
    rc = load_seq(&coll, seq_file, &opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

    /* Exactly 2 unique segments total: the shared [RF, delay, ADC] segment
     * and the whole-segment TR-tail delay.  If TR-period detection or the
     * delay-flex dedup regressed (stopped collapsing the 3 TR instances),
     * this would fail.
     *
     * pulseg_get_segment_info() always returns PULSEG_SUCCESS (it only
     * NULL-checks coll/info); an out-of-range seg_idx is signaled by the
     * sub-getters populating sentinel fields (num_blocks == -1), not by
     * the return code -- so validity is checked via num_blocks > 0. */
    rc = pulseg_get_segment_info(coll, &segi, 0);
    mu_assert(PULSEG_SUCCEEDED(rc) && segi.num_blocks > 0, "segment 0 should exist");
    rc = pulseg_get_segment_info(coll, &segi, 1);
    mu_assert(PULSEG_SUCCEEDED(rc) && segi.num_blocks > 0, "segment 1 should exist");
    rc = pulseg_get_segment_info(coll, &segi, 2);
    mu_assert(
        PULSEG_SUCCEEDED(rc) && segi.num_blocks <= 0,
        "expected exactly 2 unique segments (3 TRs should dedup to 1)");

    /* Find the 3-block [RF, delay, ADC] segment among the 2 unique ones. */
    target_seg = -1;
    rc = pulseg_get_segment_info(coll, &segi, 0);
    mu_assert(PULSEG_SUCCEEDED(rc) && segi.num_blocks > 0, "segment 0 should exist");
    if (segi.num_blocks == 3)
    {
        target_seg = 0;
    }
    else
    {
        rc = pulseg_get_segment_info(coll, &segi, 1);
        mu_assert(PULSEG_SUCCEEDED(rc) && segi.num_blocks > 0, "segment 1 should exist");
        mu_assert_int_eq(3, segi.num_blocks);
        target_seg = 1;
    }

    delay_blk = find_delay_block(coll, target_seg, segi.num_blocks);
    mu_assert(delay_blk >= 0, "interior delay block not found");

    rc = pulseg_get_block_info(coll, &bi, target_seg, delay_blk);
    mu_assert(PULSEG_SUCCEEDED(rc), "get_block_info on delay block failed");
    mu_assert_int_eq(expect_variable, bi.is_variable_delay);

    pulseg_collection_free(coll);
}

MU_TEST(test_interior_delay_static)
{
    run_delay_classification_case("99_interior_delay_static.seq", 0);
}

MU_TEST(test_interior_delay_dynamic)
{
    run_delay_classification_case("99_interior_delay_dynamic.seq", 1);
}

MU_TEST_SUITE(suite_sequences_delay_classification)
{
    MU_RUN_TEST(test_interior_delay_static);
    MU_RUN_TEST(test_interior_delay_dynamic);
}

/* ================================================================== */
/*  Sequence-description echo flag                                    */
/*                                                                     */
/*  ZERO_VAR intentionally creates the virtual canonical TR used by   */
/*  prescan. It must not imply that a real scan-table instance reaches */
/*  k=0. This regression removes the actual zero-amplitude phase-encode */
/*  line from the GRE fixture while leaving variable_grad_flags intact.*/
int test_sequences_main(void)
{
    minunit_run = 0;
    minunit_fail = 0;
    minunit_assert = 0;
    minunit_status = 0;
    minunit_real_timer = 0;
    minunit_proc_timer = 0;

    MU_RUN_SUITE(suite_sequences_smoke);
    MU_RUN_SUITE(suite_sequences_collection);
    MU_RUN_SUITE(suite_sequences_delay_classification);
    MU_REPORT();
    return MU_EXIT_CODE;
}
