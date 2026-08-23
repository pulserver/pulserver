/*
 * test_block_instance.c -- per-instance resolved view (PulSeg
 *                          SegmentInstance, spec 3.3).
 *
 * pulseg_get_block_instance_at() is a VIEW: it must report exactly what the
 * underlying block/RF/gradient/ADC/rotation tables hold, and must agree with
 * the cursor-driven pulseg_get_block_instance() at every position.  Both
 * properties are checked on a cartesian and a rotated fixture.
 */
#include "test_helpers.h"

/* ================================================================== */
/*  Helpers                                                            */
/* ================================================================== */

/* Compare one resolved instance against the descriptor tables it is a view
 * over.  Any divergence means the view has grown its own storage/logic. */
static void assert_matches_tables(
    const pulseg_sequence_descriptor *desc,
    const pulseg_block_instance *inst,
    int pos)
{
    const pulseg_block_table_element *bte;
    const pulseg_base_block *bdef;
    int idx, i;

    idx = pulseg__exec_block_idx(desc, pos);
    mu_assert(idx >= 0 && idx < desc->num_blocks, "exec-stream index out of range");

    bte = &desc->block_table[idx];
    bdef = &desc->base_blocks[bte->id];

    mu_assert_int_eq(
        (bte->duration_us >= 0) ? bte->duration_us : bdef->duration_us,
        inst->duration_us);

    if (bte->rf_id >= 0 && bte->rf_id < desc->rf_table_size)
    {
        mu_assert_float_near(
            "rf_amp_hz",
            desc->rf_table[bte->rf_id].amplitude,
            inst->rf_amp_hz,
            0.0f);
        mu_assert_float_near(
            "rf_freq_hz",
            desc->rf_table[bte->rf_id].freq_offset,
            inst->rf_freq_hz,
            0.0f);
        mu_assert_float_near(
            "rf_phase_rad",
            desc->rf_table[bte->rf_id].phase_offset,
            inst->rf_phase_rad,
            0.0f);
    }
    else
    {
        mu_assert_float_near("rf_amp_hz", 0.0f, inst->rf_amp_hz, 0.0f);
    }

    if (bte->gx_id >= 0 && bte->gx_id < desc->grad_table_size)
    {
        mu_assert_float_near(
            "gx_amp_hz_per_m",
            desc->grad_table[bte->gx_id].amplitude,
            inst->gx_amp_hz_per_m,
            0.0f);
        mu_assert_int_eq(desc->grad_table[bte->gx_id].shape_id, inst->gx_shape_id);
    }
    if (bte->gy_id >= 0 && bte->gy_id < desc->grad_table_size)
    {
        mu_assert_float_near(
            "gy_amp_hz_per_m",
            desc->grad_table[bte->gy_id].amplitude,
            inst->gy_amp_hz_per_m,
            0.0f);
        mu_assert_int_eq(desc->grad_table[bte->gy_id].shape_id, inst->gy_shape_id);
    }
    if (bte->gz_id >= 0 && bte->gz_id < desc->grad_table_size)
    {
        mu_assert_float_near(
            "gz_amp_hz_per_m",
            desc->grad_table[bte->gz_id].amplitude,
            inst->gz_amp_hz_per_m,
            0.0f);
        mu_assert_int_eq(desc->grad_table[bte->gz_id].shape_id, inst->gz_shape_id);
    }

    if (bte->rotation_id >= 0 && bte->rotation_id < desc->num_rotations)
    {
        for (i = 0; i < 9; ++i)
            mu_assert_float_near(
                "rotmat",
                desc->rotation_matrices[bte->rotation_id][i],
                inst->rotmat[i],
                0.0f);
    }
    else
    {
        /* No rotation event: identity, never a stale previous matrix. */
        for (i = 0; i < 9; ++i)
            mu_assert_float_near(
                "rotmat identity",
                (i % 4 == 0) ? 1.0f : 0.0f,
                inst->rotmat[i],
                0.0f);
    }

    if (bte->adc_id >= 0 && bte->adc_id < desc->adc_table_size)
    {
        mu_assert_int_eq(1, inst->adc_flag);
        mu_assert_float_near(
            "adc_freq_hz",
            desc->adc_table[bte->adc_id].freq_offset,
            inst->adc_freq_hz,
            0.0f);
        mu_assert_float_near(
            "adc_phase_rad",
            desc->adc_table[bte->adc_id].phase_offset,
            inst->adc_phase_rad,
            0.0f);
    }
    else
    {
        mu_assert_int_eq(0, inst->adc_flag);
    }

    mu_assert_int_eq(bte->rf_shim_id, inst->rf_shim_id);
    mu_assert_int_eq(bte->trid, inst->trid);
    mu_assert_int_eq(bte->norot_flag, inst->norot_flag);
    mu_assert_int_eq(bte->nopos_flag, inst->nopos_flag);
}

/* Walk every execution-stream position of subsequence 0 of @p filename:
 * the random-access view must equal the tables AND the cursor view. */
static void check_fixture(const char *filename)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    const pulseg_sequence_descriptor *desc;
    int rc, pos;

    default_opts_init(&opts);
    rc = load_corpus_seq(&coll, filename, &opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

    desc = &coll->descriptors[0];
    mu_assert(desc->exec_stream_len > 0, "fixture has an empty execution stream");

    /* The cursor rests one position BEFORE the first block, so each
     * iteration advances first and then reads. */
    pulseg_cursor_reset(coll);
    for (pos = 0; pos < desc->exec_stream_len; ++pos)
    {
        pulseg_block_instance direct = PULSEG_BLOCK_INSTANCE_INIT;
        pulseg_block_instance viacur = PULSEG_BLOCK_INSTANCE_INIT;

        mu_assert(
            pulseg_cursor_next(coll) == PULSEG_CURSOR_BLOCK,
            "cursor ended before the execution stream did");

        rc = pulseg_get_block_instance_at(coll, &direct, 0, pos);
        mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_block_instance_at failed");
        assert_matches_tables(desc, &direct, pos);

        rc = pulseg_get_block_instance(coll, &viacur);
        mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_block_instance failed");
        mu_assert(
            memcmp(&direct, &viacur, sizeof(direct)) == 0,
            "random-access view differs from the cursor view");
    }

    pulseg_collection_free(coll);
}

/* ================================================================== */
/*  Suite — per-instance resolved view                                 */
/* ================================================================== */

/* Cartesian fixture: no rotation events, so every instance must report
 * identity rotation. */
MU_TEST(test_block_instance_view_cartesian)
{
    check_fixture("gre_2d_3sl.seq");
}

/* Rotated fixture: exercises the rotation_matrices lookup path. */
MU_TEST(test_block_instance_view_rotated)
{
    check_fixture("gre_stack_of_stars_3d.seq");
}

/* Out-of-range indices are rejected rather than reading past a table. */
MU_TEST(test_block_instance_at_rejects_bad_indices)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    pulseg_block_instance inst = PULSEG_BLOCK_INSTANCE_INIT;
    int rc;

    default_opts_init(&opts);
    rc = load_corpus_seq(&coll, "gre_2d_3sl.seq", &opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

    mu_assert(
        PULSEG_FAILED(pulseg_get_block_instance_at(coll, &inst, -1, 0)),
        "negative subsequence index accepted");
    mu_assert(
        PULSEG_FAILED(pulseg_get_block_instance_at(coll, &inst, coll->num_subsequences, 0)),
        "out-of-range subsequence index accepted");
    mu_assert(
        PULSEG_FAILED(pulseg_get_block_instance_at(coll, &inst, 0, -1)),
        "negative stream position accepted");
    mu_assert(
        PULSEG_FAILED(
            pulseg_get_block_instance_at(coll, &inst, 0, coll->descriptors[0].exec_stream_len)),
        "out-of-range stream position accepted");
    mu_assert(
        PULSEG_FAILED(pulseg_get_block_instance_at(NULL, &inst, 0, 0)),
        "NULL collection accepted");

    pulseg_collection_free(coll);
}

/* ================================================================== */
/*  Moving an excitation with the carrier alone                        */
/* ================================================================== */

/*
 * rf_grad_constant is what lets prospective motion correction translate a
 * *selective* excitation: the pulse sees one gradient vector across its
 * window, so moving what it excites by dr is a carrier offset of G.dr.  A
 * slice-selective sequence must offer that on every RF block, with the level
 * on the slice axis and nowhere else.
 */
MU_TEST(test_a_slice_selective_excitation_can_be_moved_by_its_carrier)
{
    pulseg_collection *coll = NULL;
    pulseg_opts opts;
    pulseg_collection_info ci = PULSEG_COLLECTION_INFO_INIT;
    int seg, blk, rf_blocks;

    gre_opts_init(&opts);
    mu_assert(PULSEG_SUCCEEDED(load_corpus_seq(&coll, "gre_2d.seq", &opts)), "load failed");
    pulseg_get_collection_info(coll, &ci);

    rf_blocks = 0;
    for (seg = 0; seg < ci.num_segments; ++seg)
    {
        pulseg_segment_info si = PULSEG_SEGMENT_INFO_INIT;
        pulseg_get_segment_info(coll, &si, seg);
        for (blk = 0; blk < si.num_blocks; ++blk)
        {
            pulseg_block_info bi = PULSEG_BLOCK_INFO_INIT;
            pulseg_get_block_info(coll, &bi, seg, blk);
            if (!bi.has_rf)
                continue;
            rf_blocks++;
            mu_assert(bi.rf_grad_constant, "a slice-selective pulse is not carrier-movable");
            mu_assert(
                bi.rf_grad_level[2] > 0.5f || bi.rf_grad_level[2] < -0.5f,
                "no slice-select level reported");
        }
    }
    mu_assert(rf_blocks > 0, "fixture has no RF");

    pulseg_collection_free(coll);
}

/*
 * A hard pulse excites everything, so there is nothing to move it relative
 * to.  That is a zero offset, not a refusal -- the distinction is what a
 * motion-correction gate reads to decide whether it can run at all.
 */
MU_TEST(test_a_nonselective_excitation_is_movable_at_zero_offset)
{
    pulseg_collection *coll = NULL;
    pulseg_opts opts;
    pulseg_collection_info ci = PULSEG_COLLECTION_INFO_INIT;
    int seg, blk, ax, hard_pulses;

    gre_opts_init(&opts);
    mu_assert(PULSEG_SUCCEEDED(load_corpus_seq(&coll, "mprage_3d.seq", &opts)), "load failed");
    pulseg_get_collection_info(coll, &ci);

    hard_pulses = 0;
    for (seg = 0; seg < ci.num_segments; ++seg)
    {
        pulseg_segment_info si = PULSEG_SEGMENT_INFO_INIT;
        pulseg_get_segment_info(coll, &si, seg);
        for (blk = 0; blk < si.num_blocks; ++blk)
        {
            pulseg_block_info bi = PULSEG_BLOCK_INFO_INIT;
            int quiet = 1;
            pulseg_get_block_info(coll, &bi, seg, blk);
            if (!bi.has_rf || !bi.rf_grad_constant)
                continue;
            for (ax = 0; ax < 3; ++ax)
                if (bi.rf_grad_level[ax] > 1.0e-6f || bi.rf_grad_level[ax] < -1.0e-6f)
                    quiet = 0;
            if (quiet)
                hard_pulses++;
        }
    }
    mu_assert(hard_pulses > 0, "no nonselective pulse reported a zero-level offset");

    pulseg_collection_free(coll);
}

MU_TEST_SUITE(suite_block_instance)
{
    MU_RUN_TEST(test_block_instance_view_cartesian);
    MU_RUN_TEST(test_block_instance_view_rotated);
    MU_RUN_TEST(test_block_instance_at_rejects_bad_indices);
    MU_RUN_TEST(test_a_slice_selective_excitation_can_be_moved_by_its_carrier);
    MU_RUN_TEST(test_a_nonselective_excitation_is_movable_at_zero_offset);
}

/* ================================================================== */
/*  Entry point                                                        */
/* ================================================================== */

int test_block_instance_main(void)
{
    minunit_run = 0;
    minunit_fail = 0;
    minunit_assert = 0;
    minunit_status = 0;
    minunit_real_timer = 0;
    minunit_proc_timer = 0;

    MU_RUN_SUITE(suite_block_instance);
    MU_REPORT();
    return MU_EXIT_CODE;
}
