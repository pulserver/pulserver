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
    mu_assert_int_eq(sinfo.num_averages * sinfo.num_trs, sinfo.num_tr_instances);
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

/*
 * Copy a corpus .seq, adding one line to its [DEFINITIONS].
 *
 * The design side no longer writes IgnoreAverages -- whether a sequence
 * carries its own repetitions is stated by whether it was tiled -- so a file
 * that exercises the interpreter's clamp has to be made here rather than
 * found among the fixtures.  Writes to @p out_path; returns 1 on success.
 */
static int copy_with_definition(
    const char *filename,
    const char *line,
    char *out_path,
    size_t out_size)
{
    char in_path[512];
    char buffer[1024];
    FILE *src;
    FILE *dst;
    int inserted = 0;

    (void)snprintf(in_path, sizeof(in_path), "%s%s", TEST_CORPUS_DIR, filename);
    (void)snprintf(out_path, out_size, "%signore_averages_probe.seq", TEST_TMP_DIR);

    src = fopen(in_path, "r");
    if (!src)
        return 0;
    dst = fopen(out_path, "w");
    if (!dst)
    {
        fclose(src);
        return 0;
    }

    while (fgets(buffer, (int)sizeof(buffer), src))
    {
        fputs(buffer, dst);
        if (!inserted && strncmp(buffer, "[DEFINITIONS]", 13) == 0)
        {
            fputs(line, dst);
            inserted = 1;
        }
    }

    fclose(src);
    fclose(dst);
    return inserted;
}

/* A file carrying IgnoreAverages materialized its repeats at design time;
 * a console-side average count must NOT multiply it a second time. */
MU_TEST(test_ignore_averages_is_honored)
{
    pulseg_opts opts;
    pulseg_diagnostic diag = PULSEG_DIAGNOSTIC_INIT;
    pulseg_collection *coll = NULL;
    pulseg_subseq_info sinfo = PULSEG_SUBSEQ_INFO_INIT;
    char path[512];
    int rc;

    mu_assert(
        copy_with_definition("gre_2d.seq", "IgnoreAverages 1 \n", path, sizeof(path)),
        "could not stage a file carrying IgnoreAverages");

    gre_opts_init(&opts);
    rc = pulseg_read(&coll, &diag, path, &opts, 0, 0, 0, 3);
    mu_assert(PULSEG_SUCCEEDED(rc), "load failed");

    rc = pulseg_get_subseq_info(coll, &sinfo, 0);
    mu_assert(PULSEG_SUCCEEDED(rc), "subseq info failed");
    mu_assert_int_eq(1, sinfo.num_averages);
    mu_assert_int_eq(sinfo.num_trs, sinfo.num_tr_instances);

    pulseg_collection_free(coll);
    (void)remove(path);
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
    MU_RUN_TEST(test_ignore_averages_is_honored);
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
/* ================================================================== */

MU_TEST(test_seqdesc_echo_uses_actual_instances_not_zero_var)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    pulseg_sequence_descriptor *desc;
    pulseg_sequence_description before = {0, 0.0f, 0, NULL};
    pulseg_sequence_description after = {0, 0.0f, 0, NULL};
    int rc;
    int i;
    int before_adc = -1;
    int after_adc = -1;
    int changed = 0;

    gre_opts_init(&opts);
    rc = load_corpus_seq(&coll, "gre_2d.seq", &opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load failed");

    rc = pulseg_get_sequence_description(&before, coll, 0);
    mu_assert(PULSEG_SUCCEEDED(rc), "initial sequence description failed");
    for (i = 0; i < before.num_rows; ++i)
    {
        if (before.rows[i].type == PULSEG_SEQ_EVENT_ADC)
        {
            before_adc = i;
            break;
        }
    }
    mu_assert(before_adc >= 0, "canonical ADC row not found");
    mu_assert_int_eq(1, (int)before.rows[before_adc].params[2]);

    desc = &coll->descriptors[0];
    for (i = 0; i < desc->num_blocks; ++i)
    {
        pulseg_block_table_element *bte = &desc->block_table[i];
        int tr_pos = i % desc->tr_descriptor.tr_size;
        if (desc->variable_grad_flags && desc->variable_grad_flags[tr_pos * 3 + 1] &&
            bte->gy_id >= 0 && bte->gy_id < desc->grad_table_size &&
            fabsf(desc->grad_table[bte->gy_id].amplitude) < 0.1f)
        {
            desc->grad_table[bte->gy_id].amplitude = 4638.22f;
            changed++;
        }
    }
    mu_assert(changed > 0, "fixture has no zero-amplitude variable phase encode");

    rc = pulseg_get_sequence_description(&after, coll, 0);
    mu_assert(PULSEG_SUCCEEDED(rc), "mutated sequence description failed");
    for (i = 0; i < after.num_rows; ++i)
    {
        if (after.rows[i].type == PULSEG_SEQ_EVENT_ADC)
        {
            after_adc = i;
            break;
        }
    }
    mu_assert_int_eq(before_adc, after_adc);

    /* The existing canonical prescan metadata is unchanged. Only the new
     * real-instance echo analyzer observes that no acquired PE is zero. */
    mu_assert_int_eq((int)before.rows[before_adc].params[0], (int)after.rows[after_adc].params[0]);
    mu_assert_float_near(
        "ZERO_VAR canonical ADC anchor",
        before.rows[before_adc].timestamp_us,
        after.rows[after_adc].timestamp_us,
        1.0e-3f);
    mu_assert_int_eq(0, (int)after.rows[after_adc].params[2]);

    pulseg_sequence_description_free(&before);
    pulseg_sequence_description_free(&after);
    pulseg_collection_free(coll);
}

static void assert_seqdesc_echo_pattern(const char *seq_file, int expect_all_adc_positions)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    pulseg_sequence_description description = {0, 0.0f, 0, NULL};
    int rc;
    int i;
    int num_adcs = 0;
    int num_echoes = 0;

    gre_opts_init(&opts);
    rc = load_corpus_seq(&coll, seq_file, &opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load failed");
    rc = pulseg_get_sequence_description(&description, coll, 0);
    mu_assert(PULSEG_SUCCEEDED(rc), "sequence description failed");

    for (i = 0; i < description.num_rows; ++i)
    {
        if (description.rows[i].type == PULSEG_SEQ_EVENT_ADC)
        {
            num_adcs++;
            if ((int)description.rows[i].params[2])
                num_echoes++;
        }
    }

    mu_assert(num_adcs > 1, "fixture must have multiple ADC positions");
    if (expect_all_adc_positions)
        mu_assert_int_eq(num_adcs, num_echoes);
    else
        mu_assert_int_eq(1, num_echoes);

    pulseg_sequence_description_free(&description);
    pulseg_collection_free(coll);
}

MU_TEST(test_seqdesc_cartesian_mprage_has_one_echo_position)
{
    assert_seqdesc_echo_pattern("mprage_3d.seq", 0);
}

MU_TEST(test_seqdesc_cartesian_fse_has_one_echo_position)
{
    assert_seqdesc_echo_pattern("fse_2d.seq", 0);
}

MU_TEST(test_seqdesc_multiecho_marks_every_readout_as_echo)
{
    assert_seqdesc_echo_pattern("gre_multiecho_2d.seq", 1);
}

MU_TEST_SUITE(suite_sequences_seqdesc)
{
    MU_RUN_TEST(test_seqdesc_echo_uses_actual_instances_not_zero_var);
    MU_RUN_TEST(test_seqdesc_cartesian_mprage_has_one_echo_position);
    MU_RUN_TEST(test_seqdesc_cartesian_fse_has_one_echo_position);
    MU_RUN_TEST(test_seqdesc_multiecho_marks_every_readout_as_echo);
}

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
    MU_RUN_SUITE(suite_sequences_seqdesc);
    MU_REPORT();
    return MU_EXIT_CODE;
}
