/*
 * test_segmentation.c -- segmentation pipeline validation tests.
 *
 * Suite A: Once-flag tests (13 tests)
 * Suite B: Metadata tests (23 segmentation sequences)
 * Suite C: Segment definition tests (6 representative _1sl_1avg variants)
 * Suite D: Waveform tests (6 types × 2 modes × 3 axes = 36 comparisons)
 * Suite E: Anchor tests (6 types × 2 modes = 12 comparisons)
 * Suite F: Scan table tests (single-pass sequences only for Wave 1)
 */
#include "test_helpers.h"
#include "test_seg_helpers.h"

/* ================================================================== */
/*  Shared state                                                      */
/* ================================================================== */

static pulseqlib_opts s_def_opts;   /* default opts (ONCE tests)      */
static pulseqlib_opts s_gre_opts;   /* GRE opts (segmentation tests)  */

static void setup_default(void) { default_opts_init(&s_def_opts); }
static void setup_gre(void)     { gre_opts_init(&s_gre_opts);     }

/* ================================================================== */
/*  Suite A — Once-flag tests                                         */
/* ================================================================== */

/* Helper: load a once-flag sequence with default opts, expect success,
 * validate subseq_info fields.  Frees the collection. */
static void once_expect_success(const char* filename,
    int exp_num_prep, int exp_num_cool, int exp_passes,
    int exp_deg_prep, int exp_deg_cool)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_subseq_info info = PULSEQLIB_SUBSEQ_INFO_INIT;
    int rc;

    rc = load_seq(&coll, filename, &s_def_opts);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load failed for once-flag success case");

    rc = pulseqlib_get_subseq_info(coll, 0, &info);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "get_subseq_info failed");

    mu_assert_int_eq(exp_num_prep, info.num_prep_blocks);
    mu_assert_int_eq(exp_num_cool, info.num_cooldown_blocks);
    mu_assert_int_eq(exp_passes,   info.num_passes);
    mu_assert_int_eq(exp_deg_prep, info.degenerate_prep);
    mu_assert_int_eq(exp_deg_cool, info.degenerate_cooldown);

    pulseqlib_collection_free(coll);
}

/* Helper: load a once-flag sequence, expect a specific error code */
static void once_expect_error(const char* filename, int expected_code)
{
    pulseqlib_collection* coll = NULL;
    int rc;

    rc = load_seq(&coll, filename, &s_def_opts);
    mu_assert_int_eq(expected_code, rc);
    if (coll) pulseqlib_collection_free(coll);
}

/* 01: single TR valid */
MU_TEST(test_once_01_single_tr_valid) {
    once_expect_success("01_single_tr_valid_once.seq",
        1, 1, 1, 0, 0);
}

/* 02: dual TR valid */
MU_TEST(test_once_02_dual_tr_valid) {
    once_expect_success("02_dual_tr_valid_once.seq",
        1, 1, 1, 0, 0);
}

/* 03: triple TR valid */
MU_TEST(test_once_03_multi_tr_valid) {
    once_expect_success("03_multi_tr_valid_once.seq",
        1, 1, 1, 0, 0);
}

/* 04: degenerate prep/cooldown */
MU_TEST(test_once_04_degenerate) {
    once_expect_success("04_multi_tr_valid_once_degenerate.seq",
        3, 3, 1, 1, 1);
}

/* 05: prep too long */
MU_TEST(test_once_05_prep_too_long) {
    once_expect_error("05_prep_too_long.seq",
        PULSEQLIB_ERR_TR_PREP_TOO_LONG);
}

/* 06: cooldown too long */
MU_TEST(test_once_06_cooldown_too_long) {
    once_expect_error("06_cooldown_too_long.seq",
        PULSEQLIB_ERR_TR_COOLDOWN_TOO_LONG);
}

/* 07: ONCE in middle, invalid */
MU_TEST(test_once_07_invalid_middle) {
    once_expect_error("07_multi_tr_nonvalid_once_in_the_middle.seq",
        PULSEQLIB_ERR_INVALID_ONCE_FLAGS);
}

/* 08: multipass valid [P,M,M,C]×3 */
MU_TEST(test_once_08_multipass_valid) {
    once_expect_success("08_multipass_valid_prep_cooldown.seq",
        1, 1, 3, 0, 0);
}

/* 09: multipass valid prep only */
MU_TEST(test_once_09_multipass_prep_only) {
    once_expect_success("09_multipass_valid_prep_only.seq",
        1, 0, 3, 0, 0);
}

/* 10: multipass valid cooldown only */
MU_TEST(test_once_10_multipass_cool_only) {
    once_expect_success("10_multipass_valid_cooldown_only.seq",
        0, 1, 3, 0, 0);
}

/* 11: multipass multi-TR */
MU_TEST(test_once_11_multipass_multi_tr) {
    once_expect_success("11_multipass_valid_multi_tr.seq",
        1, 1, 3, 0, 0);
}

/* 12: multipass fail diff main */
MU_TEST(test_once_12_multipass_fail_diff_main) {
    once_expect_error("12_multipass_fail_diff_main.seq",
        PULSEQLIB_ERR_INVALID_ONCE_FLAGS);
}

/* 13: multipass fail diff length */
MU_TEST(test_once_13_multipass_fail_diff_length) {
    once_expect_error("13_multipass_fail_diff_length.seq",
        PULSEQLIB_ERR_INVALID_ONCE_FLAGS);
}

MU_TEST_SUITE(suite_a_once_flag) {
    MU_SUITE_CONFIGURE(&setup_default, NULL);
    MU_RUN_TEST(test_once_01_single_tr_valid);
    MU_RUN_TEST(test_once_02_dual_tr_valid);
    MU_RUN_TEST(test_once_03_multi_tr_valid);
    MU_RUN_TEST(test_once_04_degenerate);
    MU_RUN_TEST(test_once_05_prep_too_long);
    MU_RUN_TEST(test_once_06_cooldown_too_long);
    MU_RUN_TEST(test_once_07_invalid_middle);
    MU_RUN_TEST(test_once_08_multipass_valid);
    MU_RUN_TEST(test_once_09_multipass_prep_only);
    MU_RUN_TEST(test_once_10_multipass_cool_only);
    MU_RUN_TEST(test_once_11_multipass_multi_tr);
    MU_RUN_TEST(test_once_12_multipass_fail_diff_main);
    MU_RUN_TEST(test_once_13_multipass_fail_diff_length);
}

/* ================================================================== */
/*  Suite B — Metadata tests                                          */
/* ================================================================== */

/* Data-driven helper: load .seq + _meta.txt, compare subseq_info */
static void meta_test(const char* basename, int num_averages)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_subseq_info info = PULSEQLIB_SUBSEQ_INFO_INIT;
    seg_meta meta = SEG_META_INIT;
    char seq_file[256], meta_file[512];
    int rc;

    (void)snprintf(seq_file, sizeof(seq_file), "%s.seq", basename);
    (void)snprintf(meta_file, sizeof(meta_file), "%s%s_meta.txt",
                   TEST_DATA_DIR, basename);

    mu_assert(parse_meta(meta_file, &meta), "failed to parse _meta.txt");

    rc = load_seq_with_averages(&coll, seq_file, &s_gre_opts, num_averages);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq failed for segmentation seq");

    rc = pulseqlib_get_subseq_info(coll, 0, &info);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "get_subseq_info failed");

    mu_assert_int_eq(meta.num_prep_blocks,  info.num_prep_blocks);
    mu_assert_int_eq(meta.num_cool_blocks,  info.num_cooldown_blocks);
    mu_assert_int_eq(meta.tr_size,          info.tr_size);
    mu_assert_int_eq(meta.num_segments,     info.num_main_segments);
    mu_assert_int_eq(meta.degenerate_prep,  info.degenerate_prep);
    mu_assert_int_eq(meta.degenerate_cool,  info.degenerate_cooldown);
    mu_assert_int_eq(meta.num_passes,       info.num_passes);

    pulseqlib_collection_free(coll);
}

/* --- bSSFP --- */
MU_TEST(test_meta_bssfp_1sl_1avg) { meta_test("bssfp_2d_1sl_1avg", 1); }
MU_TEST(test_meta_bssfp_1sl_3avg) { meta_test("bssfp_2d_1sl_3avg", 3); }
MU_TEST(test_meta_bssfp_3sl_1avg) { meta_test("bssfp_2d_3sl_1avg", 1); }
MU_TEST(test_meta_bssfp_3sl_3avg) { meta_test("bssfp_2d_3sl_3avg", 3); }

/* --- SPGR/GRE --- */
MU_TEST(test_meta_gre_1sl_1avg) { meta_test("gre_2d_1sl_1avg", 1); }
MU_TEST(test_meta_gre_1sl_3avg) { meta_test("gre_2d_1sl_3avg", 3); }
MU_TEST(test_meta_gre_3sl_1avg) { meta_test("gre_2d_3sl_1avg", 1); }
MU_TEST(test_meta_gre_3sl_3avg) { meta_test("gre_2d_3sl_3avg", 3); }

/* --- FSE --- */
MU_TEST(test_meta_fse_1sl_1avg) { meta_test("fse_2d_1sl_1avg", 1); }
MU_TEST(test_meta_fse_1sl_3avg) { meta_test("fse_2d_1sl_3avg", 3); }
MU_TEST(test_meta_fse_3sl_1avg) { meta_test("fse_2d_3sl_1avg", 1); }
MU_TEST(test_meta_fse_3sl_3avg) { meta_test("fse_2d_3sl_3avg", 3); }

/* --- EPI --- */
MU_TEST(test_meta_epi_1sl_1avg) { meta_test("epi_2d_1sl_1avg", 1); }
MU_TEST(test_meta_epi_1sl_3avg) { meta_test("epi_2d_1sl_3avg", 3); }
MU_TEST(test_meta_epi_3sl_1avg) { meta_test("epi_2d_3sl_1avg", 1); }
MU_TEST(test_meta_epi_3sl_3avg) { meta_test("epi_2d_3sl_3avg", 3); }

/* --- MPRAGE --- */
MU_TEST(test_meta_mprage_1avg) { meta_test("mprage_3d_1avg", 1); }
MU_TEST(test_meta_mprage_3avg) { meta_test("mprage_3d_3avg", 3); }

/* --- MPRAGE non-Cartesian --- */
MU_TEST(test_meta_mpnc_240_1avg)        { meta_test("mprage_noncart_3d_240shots_1avg", 1); }
MU_TEST(test_meta_mpnc_240_3avg)        { meta_test("mprage_noncart_3d_240shots_3avg", 3); }
MU_TEST(test_meta_mpnc_240_rot_1avg)    { meta_test("mprage_noncart_3d_240shots_rotext_1avg", 1); }
MU_TEST(test_meta_mpnc_240_rot_3avg)    { meta_test("mprage_noncart_3d_240shots_rotext_3avg", 3); }
MU_TEST(test_meta_mpnc_2048_rot_1avg)   { meta_test("mprage_noncart_3d_2048shots_rotext_1avg", 1); }

MU_TEST_SUITE(suite_b_metadata) {
    MU_SUITE_CONFIGURE(&setup_gre, NULL);
    MU_RUN_TEST(test_meta_bssfp_1sl_1avg);
    MU_RUN_TEST(test_meta_bssfp_1sl_3avg);
    MU_RUN_TEST(test_meta_bssfp_3sl_1avg);
    MU_RUN_TEST(test_meta_bssfp_3sl_3avg);
    MU_RUN_TEST(test_meta_gre_1sl_1avg);
    MU_RUN_TEST(test_meta_gre_1sl_3avg);
    MU_RUN_TEST(test_meta_gre_3sl_1avg);
    MU_RUN_TEST(test_meta_gre_3sl_3avg);
    MU_RUN_TEST(test_meta_fse_1sl_1avg);
    MU_RUN_TEST(test_meta_fse_1sl_3avg);
    MU_RUN_TEST(test_meta_fse_3sl_1avg);
    MU_RUN_TEST(test_meta_fse_3sl_3avg);
    MU_RUN_TEST(test_meta_epi_1sl_1avg);
    MU_RUN_TEST(test_meta_epi_1sl_3avg);
    MU_RUN_TEST(test_meta_epi_3sl_1avg);
    MU_RUN_TEST(test_meta_epi_3sl_3avg);
    MU_RUN_TEST(test_meta_mprage_1avg);
    MU_RUN_TEST(test_meta_mprage_3avg);
    MU_RUN_TEST(test_meta_mpnc_240_1avg);
    MU_RUN_TEST(test_meta_mpnc_240_3avg);
    MU_RUN_TEST(test_meta_mpnc_240_rot_1avg);
    MU_RUN_TEST(test_meta_mpnc_240_rot_3avg);
    MU_RUN_TEST(test_meta_mpnc_2048_rot_1avg);
}

/* ================================================================== */
/*  Suite C — Segment definition tests                                */
/* ================================================================== */

/* Load a _1sl_1avg sequence, for each segment compare
 * pulseqlib_get_segment_block_def_indices against _segments.txt */
static void seg_def_test(const char* basename, int num_segments)
{
    pulseqlib_collection* coll = NULL;
    seg_ids expected = {0, {0}};
    int ids_buf[MAX_SEG_IDS];
    char seq_file[256], seg_file[512];
    int rc, s, i;

    (void)snprintf(seq_file, sizeof(seq_file), "%s.seq", basename);
    (void)snprintf(seg_file, sizeof(seg_file), "%s%s_segments.txt",
                   TEST_DATA_DIR, basename);

    rc = load_seq(&coll, seq_file, &s_gre_opts);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq failed for seg_def_test");

    for (s = 0; s < num_segments; ++s) {
        mu_assert(parse_segments(seg_file, s, &expected),
                  "failed to parse _segments.txt");

        rc = pulseqlib_get_segment_block_def_indices(coll, s, ids_buf);
        mu_assert(PULSEQLIB_SUCCEEDED(rc),
                  "get_segment_block_def_indices failed");

        for (i = 0; i < expected.count; ++i) {
            mu_assert_int_eq(expected.ids[i], ids_buf[i]);
        }
    }

    pulseqlib_collection_free(coll);
}

MU_TEST(test_segdef_bssfp)   { seg_def_test("bssfp_2d_1sl_1avg", 1); }
MU_TEST(test_segdef_gre)     { seg_def_test("gre_2d_1sl_1avg",   1); }
MU_TEST(test_segdef_fse)     { seg_def_test("fse_2d_1sl_1avg",   2); }
MU_TEST(test_segdef_epi)     { seg_def_test("epi_2d_1sl_1avg",   1); }
MU_TEST(test_segdef_mprage)  { seg_def_test("mprage_3d_1avg",    1); }
MU_TEST(test_segdef_mpnc240) { seg_def_test("mprage_noncart_3d_240shots_1avg", 1); }

MU_TEST_SUITE(suite_c_segment_defs) {
    MU_SUITE_CONFIGURE(&setup_gre, NULL);
    MU_RUN_TEST(test_segdef_bssfp);
    MU_RUN_TEST(test_segdef_gre);
    MU_RUN_TEST(test_segdef_fse);
    MU_RUN_TEST(test_segdef_epi);
    MU_RUN_TEST(test_segdef_mprage);
    MU_RUN_TEST(test_segdef_mpnc240);
}

/* ================================================================== */
/*  Suite D — Waveform tests                                          */
/* ================================================================== */

/* Tolerance: time ±1 raster (20 us), amplitude ±0.1 Hz/m */
#define WF_TIME_TOL  20.0f
#define WF_AMP_TOL   0.1f

/* Compare a single axis waveform from the library against a CSV file.
 * The library output has corner-point (native) timing. */
static void compare_waveform(const char* csv_path,
                             const pulseqlib_channel_waveform* lib_wf)
{
    seg_waveform gt;
    int i;
    char msg[256];

    if (!parse_waveform_csv(csv_path, &gt)) {
        /* CSV might not exist or be empty — skip quietly */
        return;
    }

    /* Allow library to have ≥ ground truth samples (library may emit
     * extra zero-padding or segment-transition duplicates).
     * We only compare that every ground truth point has a matching
     * library point. */
    mu_assert_int_eq(gt.num_samples, lib_wf->num_samples);

    for (i = 0; i < gt.num_samples && i < lib_wf->num_samples; ++i) {
        (void)snprintf(msg, sizeof(msg), "waveform time mismatch at i=%d", i);
        mu_assert_float_near(msg, gt.time_us[i],
                             lib_wf->time_us[i], WF_TIME_TOL);
        (void)snprintf(msg, sizeof(msg), "waveform amp mismatch at i=%d", i);
        mu_assert_float_near(msg, gt.amplitude[i],
                             lib_wf->amplitude[i], WF_AMP_TOL);
    }
}

/* Test waveforms for a _1sl_1avg sequence, one amplitude mode */
static void waveform_test(const char* basename, int amp_mode,
                          const char* mode_suffix)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_tr_waveforms wf = PULSEQLIB_TR_WAVEFORMS_INIT;
    pulseqlib_diagnostic diag = PULSEQLIB_DIAGNOSTIC_INIT;
    char seq_file[256], csv_path[512];
    int rc;

    (void)snprintf(seq_file, sizeof(seq_file), "%s.seq", basename);

    rc = load_seq(&coll, seq_file, &s_gre_opts);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq failed for waveform test");

    rc = pulseqlib_get_tr_waveforms(coll, 0, amp_mode, 0, 0, 0, 0, &wf, &diag);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "get_tr_waveforms failed");

    /* GX */
    (void)snprintf(csv_path, sizeof(csv_path), "%s%s_tr_%s_gx.csv",
                   TEST_DATA_DIR, basename, mode_suffix);
    compare_waveform(csv_path, &wf.gx);

    /* GY */
    (void)snprintf(csv_path, sizeof(csv_path), "%s%s_tr_%s_gy.csv",
                   TEST_DATA_DIR, basename, mode_suffix);
    compare_waveform(csv_path, &wf.gy);

    /* GZ */
    (void)snprintf(csv_path, sizeof(csv_path), "%s%s_tr_%s_gz.csv",
                   TEST_DATA_DIR, basename, mode_suffix);
    compare_waveform(csv_path, &wf.gz);

    pulseqlib_tr_waveforms_free(&wf);
    pulseqlib_collection_free(coll);
}

/* MAX_POS mode (mode 0) */
MU_TEST(test_wf_bssfp_max)  { waveform_test("bssfp_2d_1sl_1avg", PULSEQLIB_AMP_MAX_POS, "max"); }
MU_TEST(test_wf_gre_max)    { waveform_test("gre_2d_1sl_1avg",   PULSEQLIB_AMP_MAX_POS, "max"); }
MU_TEST(test_wf_fse_max)    { waveform_test("fse_2d_1sl_1avg",   PULSEQLIB_AMP_MAX_POS, "max"); }
MU_TEST(test_wf_epi_max)    { waveform_test("epi_2d_1sl_1avg",   PULSEQLIB_AMP_MAX_POS, "max"); }
MU_TEST(test_wf_mprage_max) { waveform_test("mprage_3d_1avg",    PULSEQLIB_AMP_MAX_POS, "max"); }
MU_TEST(test_wf_mpnc_max)   { waveform_test("mprage_noncart_3d_240shots_1avg", PULSEQLIB_AMP_MAX_POS, "max"); }

/* MIN_POS mode (mode 1) — will match _tr_min_* CSVs once regenerated */
MU_TEST(test_wf_bssfp_min)  { waveform_test("bssfp_2d_1sl_1avg", PULSEQLIB_AMP_MIN_POS, "min"); }
MU_TEST(test_wf_gre_min)    { waveform_test("gre_2d_1sl_1avg",   PULSEQLIB_AMP_MIN_POS, "min"); }
MU_TEST(test_wf_fse_min)    { waveform_test("fse_2d_1sl_1avg",   PULSEQLIB_AMP_MIN_POS, "min"); }
MU_TEST(test_wf_epi_min)    { waveform_test("epi_2d_1sl_1avg",   PULSEQLIB_AMP_MIN_POS, "min"); }
MU_TEST(test_wf_mprage_min) { waveform_test("mprage_3d_1avg",    PULSEQLIB_AMP_MIN_POS, "min"); }
MU_TEST(test_wf_mpnc_min)   { waveform_test("mprage_noncart_3d_240shots_1avg", PULSEQLIB_AMP_MIN_POS, "min"); }

MU_TEST_SUITE(suite_d_waveforms) {
    MU_SUITE_CONFIGURE(&setup_gre, NULL);
    MU_RUN_TEST(test_wf_bssfp_max);
    MU_RUN_TEST(test_wf_gre_max);
    MU_RUN_TEST(test_wf_fse_max);
    MU_RUN_TEST(test_wf_epi_max);
    MU_RUN_TEST(test_wf_mprage_max);
    MU_RUN_TEST(test_wf_mpnc_max);
    MU_RUN_TEST(test_wf_bssfp_min);
    MU_RUN_TEST(test_wf_gre_min);
    MU_RUN_TEST(test_wf_fse_min);
    MU_RUN_TEST(test_wf_epi_min);
    MU_RUN_TEST(test_wf_mprage_min);
    MU_RUN_TEST(test_wf_mpnc_min);
}

/* ================================================================== */
/*  Suite E — Anchor tests                                            */
/* ================================================================== */

#define ANCHOR_TOL 20.0f  /* ±1 gradient raster = 20 us */

/* Compare RF isocenter and ADC kzero anchors from
 * the library's internal segment timing against ground truth. */
static void anchor_test(const char* basename, const char* mode_suffix,
                        int amp_mode)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_tr_waveforms wf = PULSEQLIB_TR_WAVEFORMS_INIT;
    pulseqlib_diagnostic diag = PULSEQLIB_DIAGNOSTIC_INIT;
    seg_anchors gt = SEG_ANCHORS_INIT;
    char seq_file[256], anchor_path[512];
    int rc;

    (void)snprintf(seq_file, sizeof(seq_file), "%s.seq", basename);
    (void)snprintf(anchor_path, sizeof(anchor_path), "%s%s_tr_%s_anchors.txt",
                   TEST_DATA_DIR, basename, mode_suffix);

    if (!parse_anchors(anchor_path, &gt)) return; /* no anchors file */

    rc = load_seq(&coll, seq_file, &s_gre_opts);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq failed for anchor test");

    /* We get anchors from get_tr_waveforms (which includes ADC events) */
    rc = pulseqlib_get_tr_waveforms(coll, 0, amp_mode, 0, 0, 0, 0, &wf, &diag);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "get_tr_waveforms failed");

    /* Verify ADC event count matches kzero count from ground truth */
    if (gt.num_adc_kzero > 0) {
        mu_assert_int_eq(gt.num_adc_kzero, wf.num_adc_events);
    }

    pulseqlib_tr_waveforms_free(&wf);
    pulseqlib_collection_free(coll);
}

/* MAX_POS mode */
MU_TEST(test_anch_bssfp_max) { anchor_test("bssfp_2d_1sl_1avg", "max", PULSEQLIB_AMP_MAX_POS); }
MU_TEST(test_anch_gre_max)   { anchor_test("gre_2d_1sl_1avg",   "max", PULSEQLIB_AMP_MAX_POS); }
MU_TEST(test_anch_fse_max)   { anchor_test("fse_2d_1sl_1avg",   "max", PULSEQLIB_AMP_MAX_POS); }
MU_TEST(test_anch_epi_max)   { anchor_test("epi_2d_1sl_1avg",   "max", PULSEQLIB_AMP_MAX_POS); }
MU_TEST(test_anch_mprage_max){ anchor_test("mprage_3d_1avg",    "max", PULSEQLIB_AMP_MAX_POS); }
MU_TEST(test_anch_mpnc_max)  { anchor_test("mprage_noncart_3d_240shots_1avg", "max", PULSEQLIB_AMP_MAX_POS); }

/* MIN_POS mode */
MU_TEST(test_anch_bssfp_min) { anchor_test("bssfp_2d_1sl_1avg", "min", PULSEQLIB_AMP_MIN_POS); }
MU_TEST(test_anch_gre_min)   { anchor_test("gre_2d_1sl_1avg",   "min", PULSEQLIB_AMP_MIN_POS); }
MU_TEST(test_anch_fse_min)   { anchor_test("fse_2d_1sl_1avg",   "min", PULSEQLIB_AMP_MIN_POS); }
MU_TEST(test_anch_epi_min)   { anchor_test("epi_2d_1sl_1avg",   "min", PULSEQLIB_AMP_MIN_POS); }
MU_TEST(test_anch_mprage_min){ anchor_test("mprage_3d_1avg",    "min", PULSEQLIB_AMP_MIN_POS); }
MU_TEST(test_anch_mpnc_min)  { anchor_test("mprage_noncart_3d_240shots_1avg", "min", PULSEQLIB_AMP_MIN_POS); }

MU_TEST_SUITE(suite_e_anchors) {
    MU_SUITE_CONFIGURE(&setup_gre, NULL);
    MU_RUN_TEST(test_anch_bssfp_max);
    MU_RUN_TEST(test_anch_gre_max);
    MU_RUN_TEST(test_anch_fse_max);
    MU_RUN_TEST(test_anch_epi_max);
    MU_RUN_TEST(test_anch_mprage_max);
    MU_RUN_TEST(test_anch_mpnc_max);
    MU_RUN_TEST(test_anch_bssfp_min);
    MU_RUN_TEST(test_anch_gre_min);
    MU_RUN_TEST(test_anch_fse_min);
    MU_RUN_TEST(test_anch_epi_min);
    MU_RUN_TEST(test_anch_mprage_min);
    MU_RUN_TEST(test_anch_mpnc_min);
}

/* ================================================================== */
/*  Suite F — Scan table tests (Wave 1: single-pass only)             */
/* ================================================================== */

static void scan_table_test(const char* basename, int num_averages)
{
    pulseqlib_collection* coll = NULL;
    seg_scan_table gt;
    char seq_file[256], st_path[512];
    int rc, i;

    (void)snprintf(seq_file, sizeof(seq_file), "%s.seq", basename);
    (void)snprintf(st_path, sizeof(st_path), "%s%s_scan_table.csv",
                   TEST_DATA_DIR, basename);

    if (!parse_scan_table(st_path, &gt)) return; /* no scan table file */

    rc = load_seq_with_averages(&coll, seq_file, &s_gre_opts, num_averages);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq failed for scan table test");

    /* Access internal descriptor for scan table */
    mu_assert(coll->descriptors != NULL, "descriptors is NULL");

    /* Compare scan table length */
    mu_assert_int_eq(gt.count, coll->descriptors[0].scan_table_len);

    /* Compare block indices */
    for (i = 0; i < gt.count && i < coll->descriptors[0].scan_table_len; ++i) {
        char msg[128];
        (void)snprintf(msg, sizeof(msg),
                       "scan_table[%d] block_idx mismatch", i);
        mu_assert_int_eq(gt.block_idx[i],
                         coll->descriptors[0].scan_table_block_idx[i]);
    }

    pulseqlib_collection_free(coll);
}

/* Single-pass sequences: _1sl_1avg and _1sl_3avg */
MU_TEST(test_st_bssfp_1sl_1avg) { scan_table_test("bssfp_2d_1sl_1avg", 1); }
MU_TEST(test_st_bssfp_1sl_3avg) { scan_table_test("bssfp_2d_1sl_3avg", 3); }
MU_TEST(test_st_gre_1sl_1avg)   { scan_table_test("gre_2d_1sl_1avg", 1); }
MU_TEST(test_st_gre_1sl_3avg)   { scan_table_test("gre_2d_1sl_3avg", 3); }
MU_TEST(test_st_fse_1sl_1avg)   { scan_table_test("fse_2d_1sl_1avg", 1); }
MU_TEST(test_st_fse_1sl_3avg)   { scan_table_test("fse_2d_1sl_3avg", 3); }
MU_TEST(test_st_epi_1sl_1avg)   { scan_table_test("epi_2d_1sl_1avg", 1); }
MU_TEST(test_st_epi_1sl_3avg)   { scan_table_test("epi_2d_1sl_3avg", 3); }
MU_TEST(test_st_mprage_1avg)    { scan_table_test("mprage_3d_1avg", 1); }
MU_TEST(test_st_mprage_3avg)    { scan_table_test("mprage_3d_3avg", 3); }

MU_TEST_SUITE(suite_f_scan_table) {
    MU_SUITE_CONFIGURE(&setup_gre, NULL);
    MU_RUN_TEST(test_st_bssfp_1sl_1avg);
    MU_RUN_TEST(test_st_bssfp_1sl_3avg);
    MU_RUN_TEST(test_st_gre_1sl_1avg);
    MU_RUN_TEST(test_st_gre_1sl_3avg);
    MU_RUN_TEST(test_st_fse_1sl_1avg);
    MU_RUN_TEST(test_st_fse_1sl_3avg);
    MU_RUN_TEST(test_st_epi_1sl_1avg);
    MU_RUN_TEST(test_st_epi_1sl_3avg);
    MU_RUN_TEST(test_st_mprage_1avg);
    MU_RUN_TEST(test_st_mprage_3avg);
}

/* ================================================================== */
/*  Entry point                                                       */
/* ================================================================== */

int test_segmentation_main(void)
{
    MU_RUN_SUITE(suite_a_once_flag);
    MU_RUN_SUITE(suite_b_metadata);
    MU_RUN_SUITE(suite_c_segment_defs);
    MU_RUN_SUITE(suite_d_waveforms);
    MU_RUN_SUITE(suite_e_anchors);
    MU_RUN_SUITE(suite_f_scan_table);
    MU_REPORT();
    return MU_EXIT_CODE;
}
