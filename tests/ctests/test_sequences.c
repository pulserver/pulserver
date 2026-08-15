/*
 * test_segmentation.c -- segmentation tests (phases 1-5).
 *
 * Phase 1: Validates example_check.c step 6 quantities:
 *   1. Unique ADC definitions (count, num_samples, dwell_ns)
 *   2. max_b1_subseq index
 *   3. Nominal TR duration
 *
 * Phase 2: Validates example_check.c step 5 quantities + TR waveforms:
 *   4. Segment structure (count, blocks per segment)
 *   5. Worst-case TR gradient waveforms vs MATLAB ground truth
 *
 * Phase 4: Frequency-modulation base definitions:
 *   Builds freq-mod collection with known shift vectors, compares
 *   1D output against MATLAB-serialized 3-channel waveforms.
 *   Tests orthogonal shifts (X/Y/Z/combined) x 4 FOV rotations.
 *
 * Phase 5: Scan table — block instance validation:
 *   Walks the full scan table via cursor, comparing each block
 *   instance against MATLAB ground truth (amplitudes, offsets,
 *   flags, rotation matrices).
 */
#include "test_helpers.h"
#include "test_seg_helpers.h"

#include <math.h>

static int tr_waveform_matches_ref(
    const seg_tr_waveform *ref_wf,
    const pulseg_tr_gradient_waveforms *lib_wf)
{
    const float wave_rel_tol = 1e-3f;
    const float wave_time_abs_tol = 0.5f;
    int i, n;

    n = ref_wf->num_samples < lib_wf->gx.num_samples ? ref_wf->num_samples : lib_wf->gx.num_samples;

    if (abs(ref_wf->num_samples - lib_wf->gx.num_samples) > 1)
        return 0;

    for (i = 0; i < n; ++i)
    {
        float ref_t = ref_wf->time_us[i];
        float lib_t = lib_wf->gx.time_us[i];
        float dt = ref_t - lib_t;
        float ref_gx = ref_wf->gx[i];
        float ref_gy = ref_wf->gy[i];
        float ref_gz = ref_wf->gz[i];
        float lib_gx = lib_wf->gx.amplitude_hz_per_m[i];
        float lib_gy = lib_wf->gy.amplitude_hz_per_m[i];
        float lib_gz = lib_wf->gz.amplitude_hz_per_m[i];
        float tol_gx, tol_gy, tol_gz;

        if (dt < 0)
            dt = -dt;
        if (dt > wave_time_abs_tol)
            return 0;

        tol_gx = (ref_gx < 0 ? -ref_gx : ref_gx) * wave_rel_tol;
        if (tol_gx < 1.0f)
            tol_gx = 1.0f;
        tol_gy = (ref_gy < 0 ? -ref_gy : ref_gy) * wave_rel_tol;
        if (tol_gy < 1.0f)
            tol_gy = 1.0f;
        tol_gz = (ref_gz < 0 ? -ref_gz : ref_gz) * wave_rel_tol;
        if (tol_gz < 1.0f)
            tol_gz = 1.0f;

        if (fabsf(ref_gx - lib_gx) > tol_gx || fabsf(ref_gy - lib_gy) > tol_gy ||
            fabsf(ref_gz - lib_gz) > tol_gz)
            return 0;
    }

    return 1;
}

typedef struct
{
    const char *name;
    const char *seq_file;
    const char *base;
    int num_averages;
} seq_case;

static const seq_case kGreCases[] = {
    {"gre_2d_1sl_1avg", "gre_2d_1sl_1avg.seq", "gre_2d_1sl_1avg", 1},
    {"gre_2d_1sl_3avg", "gre_2d_1sl_3avg.seq", "gre_2d_1sl_3avg", 3},
    {"gre_2d_3sl_1avg", "gre_2d_3sl_1avg.seq", "gre_2d_3sl_1avg", 1},
    {"gre_2d_3sl_3avg", "gre_2d_3sl_3avg.seq", "gre_2d_3sl_3avg", 3},
};

static const seq_case kMprageCases[] = {
    {"mprage_2d_1sl_1avg", "mprage_2d_1sl_1avg.seq", "mprage_2d_1sl_1avg", 1},
    {"mprage_2d_1sl_3avg", "mprage_2d_1sl_3avg.seq", "mprage_2d_1sl_3avg", 3},
    {"mprage_2d_3sl_1avg", "mprage_2d_3sl_1avg.seq", "mprage_2d_3sl_1avg", 1},
    {"mprage_2d_3sl_3avg", "mprage_2d_3sl_3avg.seq", "mprage_2d_3sl_3avg", 3},
};

static const seq_case kBssfpCases[] = {
    {"bssfp_2d_1sl_1avg", "bssfp_2d_1sl_1avg.seq", "bssfp_2d_1sl_1avg", 1},
    {"bssfp_2d_1sl_3avg", "bssfp_2d_1sl_3avg.seq", "bssfp_2d_1sl_3avg", 3},
    {"bssfp_2d_3sl_1avg", "bssfp_2d_3sl_1avg.seq", "bssfp_2d_3sl_1avg", 1},
    {"bssfp_2d_3sl_3avg", "bssfp_2d_3sl_3avg.seq", "bssfp_2d_3sl_3avg", 3},
};

static const seq_case kFseCases[] = {
    {"fse_2d_1sl_1avg", "fse_2d_1sl_1avg.seq", "fse_2d_1sl_1avg", 1},
    {"fse_2d_1sl_3avg", "fse_2d_1sl_3avg.seq", "fse_2d_1sl_3avg", 3},
    {"fse_2d_3sl_1avg", "fse_2d_3sl_1avg.seq", "fse_2d_3sl_1avg", 1},
    {"fse_2d_3sl_3avg", "fse_2d_3sl_3avg.seq", "fse_2d_3sl_3avg", 3},
};

static const seq_case kMprageNoncartCases[] = {
    {"mprage_noncart_3d_1sl_1avg_userotext0",
     "mprage_noncart_3d_1sl_1avg_userotext0.seq",
     "mprage_noncart_3d_1sl_1avg_userotext0",
     1},
    {"mprage_noncart_3d_1sl_3avg_userotext0",
     "mprage_noncart_3d_1sl_3avg_userotext0.seq",
     "mprage_noncart_3d_1sl_3avg_userotext0",
     3},
    {"mprage_noncart_3d_3sl_1avg_userotext0",
     "mprage_noncart_3d_3sl_1avg_userotext0.seq",
     "mprage_noncart_3d_3sl_1avg_userotext0",
     1},
    {"mprage_noncart_3d_3sl_3avg_userotext0",
     "mprage_noncart_3d_3sl_3avg_userotext0.seq",
     "mprage_noncart_3d_3sl_3avg_userotext0",
     3},
    {"mprage_noncart_3d_1sl_1avg_userotext1",
     "mprage_noncart_3d_1sl_1avg_userotext1.seq",
     "mprage_noncart_3d_1sl_1avg_userotext1",
     1},
    {"mprage_noncart_3d_1sl_3avg_userotext1",
     "mprage_noncart_3d_1sl_3avg_userotext1.seq",
     "mprage_noncart_3d_1sl_3avg_userotext1",
     3},
    {"mprage_noncart_3d_3sl_1avg_userotext1",
     "mprage_noncart_3d_3sl_1avg_userotext1.seq",
     "mprage_noncart_3d_3sl_1avg_userotext1",
     1},
    {"mprage_noncart_3d_3sl_3avg_userotext1",
     "mprage_noncart_3d_3sl_3avg_userotext1.seq",
     "mprage_noncart_3d_3sl_3avg_userotext1",
     3},
};

static const seq_case kQalasNoncartCases[] = {
    {"qalas_noncart_3d_1sl_1avg_userotext1",
     "qalas_noncart_3d_1sl_1avg_userotext1.seq",
     "qalas_noncart_3d_1sl_1avg_userotext1",
     1},
    {"qalas_noncart_3d_1sl_3avg_userotext1",
     "qalas_noncart_3d_1sl_3avg_userotext1.seq",
     "qalas_noncart_3d_1sl_3avg_userotext1",
     3},
    {"qalas_noncart_3d_3sl_1avg_userotext1",
     "qalas_noncart_3d_3sl_1avg_userotext1.seq",
     "qalas_noncart_3d_3sl_1avg_userotext1",
     1},
    {"qalas_noncart_3d_3sl_3avg_userotext1",
     "qalas_noncart_3d_3sl_3avg_userotext1.seq",
     "qalas_noncart_3d_3sl_3avg_userotext1",
     3},
};

static const seq_case kMprageNavCases[] = {
    {"mprage_nav_2d_1sl_1avg", "mprage_nav_2d_1sl_1avg.seq", "mprage_nav_2d_1sl_1avg", 1},
    {"mprage_nav_2d_1sl_3avg", "mprage_nav_2d_1sl_3avg.seq", "mprage_nav_2d_1sl_3avg", 3},
    {"mprage_nav_2d_3sl_1avg", "mprage_nav_2d_3sl_1avg.seq", "mprage_nav_2d_3sl_1avg", 1},
    {"mprage_nav_2d_3sl_3avg", "mprage_nav_2d_3sl_3avg.seq", "mprage_nav_2d_3sl_3avg", 3},
};

static const seq_case kGreEpiCollectionCases[] = {
    {"gre_epi_collection_2d_1sl_1avg",
     "gre_epi_collection_2d_1sl_1avg.seq",
     "gre_epi_collection_2d_1sl_1avg",
     1},
    {"gre_epi_collection_2d_1sl_3avg",
     "gre_epi_collection_2d_1sl_3avg.seq",
     "gre_epi_collection_2d_1sl_3avg",
     3},
};

static void build_case_path(char *dst, size_t dst_sz, const seq_case *tc, const char *suffix)
{
    (void)snprintf(dst, dst_sz, TEST_DATA_DIR "%s%s", tc->base, suffix);
}

/* ------------------------------------------------------------------ */
/*  Phase 1: example_check step 6 (ADC, max_b1, TR)                  */
/* ------------------------------------------------------------------ */

static void run_check_case(const seq_case *tc)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    pulseg_collection_info cinfo = PULSEG_COLLECTION_INFO_INIT;
    pulseg_subseq_info sinfo = PULSEG_SUBSEQ_INFO_INIT;
    seg_meta meta = SEG_META_INIT;
    char meta_path[512];
    int rc, a, ok;
    int expected_max_adc_samples = 0;

    /* Load sequence */
    gre_opts_init(&opts);
    rc = load_seq_with_averages(&coll, tc->seq_file, &opts, tc->num_averages);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed for GRE test case");

    /* Collection must have exactly one subsequence */
    rc = pulseg_get_collection_info(coll, &cinfo);
    mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_collection_info failed");
    mu_assert_int_eq(1, cinfo.num_subsequences);

    /* Get subsequence info */
    rc = pulseg_get_subseq_info(coll, &sinfo, 0);
    mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_subseq_info failed");

    /* Parse MATLAB ground truth */
    build_case_path(meta_path, sizeof(meta_path), tc, "_meta.txt");
    ok = parse_meta(meta_path, &meta);
    mu_assert(ok, "failed to parse case _meta.txt");

    /* 1. Unique ADC definitions */
    mu_assert_int_eq(meta.num_unique_adcs, sinfo.num_unique_adcs);
    for (a = 0; a < sinfo.num_unique_adcs; ++a)
    {
        pulseg_adc_def ad = PULSEG_ADC_DEF_INIT;
        rc = pulseg_get_adc_def(coll, &ad, a);
        mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_adc_def failed");
        if (meta.adc_samples[a] > expected_max_adc_samples)
            expected_max_adc_samples = meta.adc_samples[a];
        mu_assert_int_eq(meta.adc_samples[a], ad.num_samples);
        mu_assert_int_eq(meta.adc_dwell_ns[a], ad.dwell_ns);
    }

    /* Explicitly verify collection-level max ADC samples. */
    mu_assert_int_eq(expected_max_adc_samples, cinfo.max_adc_samples);

    /* 2. max_b1_subseq — trivially 0 for single-subsequence collection */
    mu_assert_int_eq(0, meta.max_b1_subseq);

    /* 3. Nominal TR */
    fprintf(
        stderr,
        "[check][%s] tr=%.3f meta_tr=%d num_trs=%d tr_size=%d num_tr_instances=%d\n",
        tc->name,
        sinfo.tr_duration_us,
        meta.tr_duration_us,
        sinfo.num_trs,
        sinfo.tr_size,
        sinfo.num_tr_instances);
    mu_assert_float_near("TR duration", (float)meta.tr_duration_us, sinfo.tr_duration_us, 1.0f);

    /* 4. Num segments — verify segment count matches truth */
    fprintf(
        stderr,
        "[check][%s] num_segments: meta=%d  lib=%d\n",
        tc->name,
        meta.num_segments,
        cinfo.num_segments);
    mu_assert_int_eq(meta.num_segments, cinfo.num_segments);

    pulseg_collection_free(coll);
}

MU_TEST(test_check_gre_2d_1sl_1avg)
{
    run_check_case(&kGreCases[0]);
}
MU_TEST(test_check_gre_2d_1sl_3avg)
{
    run_check_case(&kGreCases[1]);
}
MU_TEST(test_check_gre_2d_3sl_1avg)
{
    run_check_case(&kGreCases[2]);
}
MU_TEST(test_check_gre_2d_3sl_3avg)
{
    run_check_case(&kGreCases[3]);
}

MU_TEST(test_check_mprage_2d_1sl_1avg)
{
    run_check_case(&kMprageCases[0]);
}
MU_TEST(test_check_mprage_2d_1sl_3avg)
{
    run_check_case(&kMprageCases[1]);
}
MU_TEST(test_check_mprage_2d_3sl_1avg)
{
    run_check_case(&kMprageCases[2]);
}
MU_TEST(test_check_mprage_2d_3sl_3avg)
{
    run_check_case(&kMprageCases[3]);
}
MU_TEST(test_check_mprage_nc_1sl_1avg)
{
    run_check_case(&kMprageNoncartCases[0]);
}
MU_TEST(test_check_mprage_nc_1sl_3avg_userotext0)
{
    run_check_case(&kMprageNoncartCases[1]);
}
MU_TEST(test_check_mprage_nc_3sl_1avg_userotext0)
{
    run_check_case(&kMprageNoncartCases[2]);
}
MU_TEST(test_check_mprage_nc_3sl_3avg_userotext0)
{
    run_check_case(&kMprageNoncartCases[3]);
}
MU_TEST(test_check_mprage_nc_1sl_1avg_userotext1)
{
    run_check_case(&kMprageNoncartCases[4]);
}
MU_TEST(test_check_mprage_nc_1sl_3avg_userotext1)
{
    run_check_case(&kMprageNoncartCases[5]);
}
MU_TEST(test_check_mprage_nc_3sl_1avg_userotext1)
{
    run_check_case(&kMprageNoncartCases[6]);
}
MU_TEST(test_check_mprage_nc_3sl_3avg_userotext1)
{
    run_check_case(&kMprageNoncartCases[7]);
}

MU_TEST(test_check_bssfp_2d_1sl_1avg)
{
    run_check_case(&kBssfpCases[0]);
}
MU_TEST(test_check_bssfp_2d_1sl_3avg)
{
    run_check_case(&kBssfpCases[1]);
}
MU_TEST(test_check_bssfp_2d_3sl_1avg)
{
    run_check_case(&kBssfpCases[2]);
}
MU_TEST(test_check_bssfp_2d_3sl_3avg)
{
    run_check_case(&kBssfpCases[3]);
}

MU_TEST(test_check_fse_2d_1sl_1avg)
{
    run_check_case(&kFseCases[0]);
}
MU_TEST(test_check_fse_2d_1sl_3avg)
{
    run_check_case(&kFseCases[1]);
}
MU_TEST(test_check_fse_2d_3sl_1avg)
{
    run_check_case(&kFseCases[2]);
}
MU_TEST(test_check_fse_2d_3sl_3avg)
{
    run_check_case(&kFseCases[3]);
}

MU_TEST(test_check_qalas_nc_3d_1sl_1avg_userotext1)
{
    run_check_case(&kQalasNoncartCases[0]);
}
MU_TEST(test_check_qalas_nc_3d_1sl_3avg_userotext1)
{
    run_check_case(&kQalasNoncartCases[1]);
}
MU_TEST(test_check_qalas_nc_3d_3sl_1avg_userotext1)
{
    run_check_case(&kQalasNoncartCases[2]);
}
MU_TEST(test_check_qalas_nc_3d_3sl_3avg_userotext1)
{
    run_check_case(&kQalasNoncartCases[3]);
}

MU_TEST(test_check_mprage_nav_2d_1sl_1avg)
{
    run_check_case(&kMprageNavCases[0]);
}
MU_TEST(test_check_mprage_nav_2d_1sl_3avg)
{
    run_check_case(&kMprageNavCases[1]);
}
MU_TEST(test_check_mprage_nav_2d_3sl_1avg)
{
    run_check_case(&kMprageNavCases[2]);
}
MU_TEST(test_check_mprage_nav_2d_3sl_3avg)
{
    run_check_case(&kMprageNavCases[3]);
}

MU_TEST_SUITE(suite_sequences_check)
{
    MU_RUN_TEST(test_check_gre_2d_1sl_1avg);
    MU_RUN_TEST(test_check_gre_2d_1sl_3avg);
    MU_RUN_TEST(test_check_gre_2d_3sl_1avg);
    MU_RUN_TEST(test_check_gre_2d_3sl_3avg);
    MU_RUN_TEST(test_check_mprage_2d_1sl_1avg);
    MU_RUN_TEST(test_check_mprage_2d_1sl_3avg);
    MU_RUN_TEST(test_check_mprage_2d_3sl_1avg);
    MU_RUN_TEST(test_check_mprage_2d_3sl_3avg);
    MU_RUN_TEST(test_check_mprage_nc_1sl_1avg);
    MU_RUN_TEST(test_check_mprage_nc_1sl_3avg_userotext0);
    MU_RUN_TEST(test_check_mprage_nc_3sl_1avg_userotext0);
    MU_RUN_TEST(test_check_mprage_nc_3sl_3avg_userotext0);
    MU_RUN_TEST(test_check_mprage_nc_1sl_1avg_userotext1);
    MU_RUN_TEST(test_check_mprage_nc_1sl_3avg_userotext1);
    MU_RUN_TEST(test_check_mprage_nc_3sl_1avg_userotext1);
    MU_RUN_TEST(test_check_mprage_nc_3sl_3avg_userotext1);
    MU_RUN_TEST(test_check_bssfp_2d_1sl_1avg);
    MU_RUN_TEST(test_check_bssfp_2d_1sl_3avg);
    MU_RUN_TEST(test_check_bssfp_2d_3sl_1avg);
    MU_RUN_TEST(test_check_bssfp_2d_3sl_3avg);
    MU_RUN_TEST(test_check_fse_2d_1sl_1avg);
    MU_RUN_TEST(test_check_fse_2d_1sl_3avg);
    MU_RUN_TEST(test_check_fse_2d_3sl_1avg);
    MU_RUN_TEST(test_check_fse_2d_3sl_3avg);
    MU_RUN_TEST(test_check_qalas_nc_3d_1sl_1avg_userotext1);
    MU_RUN_TEST(test_check_qalas_nc_3d_1sl_3avg_userotext1);
    MU_RUN_TEST(test_check_qalas_nc_3d_3sl_1avg_userotext1);
    MU_RUN_TEST(test_check_qalas_nc_3d_3sl_3avg_userotext1);
    MU_RUN_TEST(test_check_mprage_nav_2d_1sl_1avg);
    MU_RUN_TEST(test_check_mprage_nav_2d_1sl_3avg);
    MU_RUN_TEST(test_check_mprage_nav_2d_3sl_1avg);
    MU_RUN_TEST(test_check_mprage_nav_2d_3sl_3avg);
}

/* ------------------------------------------------------------------ */
/*  Phase 2: example_check step 5 (segments) + TR waveforms           */
/* ------------------------------------------------------------------ */

/* Relative tolerance for waveform amplitude comparison. */
#define WAVE_REL_TOL 1e-3f
#define WAVE_TIME_ABS_TOL 0.5f /* us — half a raster step */

static void run_sequences_uieval_case(const seq_case *tc)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    pulseg_collection_info cinfo = PULSEG_COLLECTION_INFO_INIT;
    pulseg_subseq_info sinfo = PULSEG_SUBSEQ_INFO_INIT;
    seg_meta meta = SEG_META_INIT;
    seg_tr_waveform_set ref_wfs = SEG_TR_WAVEFORM_SET_INIT;
    /* removed unused lib_wf variable */
    pulseg_diagnostic diag;
    char meta_path[512];
    char tr_path[512];
    int rc, s, i, ok;

    /* Load sequence */
    gre_opts_init(&opts);
    rc = load_seq_with_averages(&coll, tc->seq_file, &opts, tc->num_averages);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed for GRE test case");

    /* Collection info */
    rc = pulseg_get_collection_info(coll, &cinfo);
    mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_collection_info failed");

    /* Subseq info */
    rc = pulseg_get_subseq_info(coll, &sinfo, 0);
    mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_subseq_info failed");

    /* Parse MATLAB ground truth */
    build_case_path(meta_path, sizeof(meta_path), tc, "_meta.txt");
    ok = parse_meta(meta_path, &meta);
    mu_assert(ok, "failed to parse case _meta.txt");

    /* 4. Segment structure */
    fprintf(
        stderr,
        "[uieval][%s] num_segments: meta=%d  lib=%d\n",
        tc->name,
        meta.num_segments,
        cinfo.num_segments);
    mu_assert_int_eq(meta.num_segments, cinfo.num_segments);
    for (s = 0; s < cinfo.num_segments && s < MAX_SEGMENTS; ++s)
    {
        pulseg_segment_info segi = PULSEG_SEGMENT_INFO_INIT;
        rc = pulseg_get_segment_info(coll, &segi, s);
        mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_segment_info failed");
        fprintf(
            stderr,
            "[uieval][%s] segment %d num_blocks: meta=%d  lib=%d\n",
            tc->name,
            s,
            meta.segment_num_blocks[s],
            segi.num_blocks);
        mu_assert_int_eq(meta.segment_num_blocks[s], segi.num_blocks);
    }

    /* 5. Worst-case TR gradient waveforms (multi-subsequence aware) */
    int num_subseq = cinfo.num_subsequences;
    mu_assert(num_subseq > 0, "invalid num_subsequences from library");

    for (int subseq_idx = 0; subseq_idx < num_subseq; ++subseq_idx)
    {
        rc = pulseg_get_subseq_info(coll, &sinfo, subseq_idx);
        mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_subseq_info failed");
        mu_assert(sinfo.num_tr_instances > 0, "invalid num_tr_instances from library");

        /* Build ground truth path for this subsequence if needed (here assumed single ground truth for all, adjust if needed) */
        build_case_path(tr_path, sizeof(tr_path), tc, "_tr_waveform.truth");
        ok = parse_tr_waveform_set(tr_path, &ref_wfs);
        mu_assert(ok, "failed to parse case _tr_waveform.truth");
        mu_assert_int_eq(meta.num_canonical_trs, ref_wfs.num_trs);

        for (int tr_idx = 0; tr_idx < ref_wfs.num_trs; ++tr_idx)
        {
            pulseg_tr_gradient_waveforms tr_wf = PULSEG_TR_GRADIENT_WAVEFORMS_INIT;
            pulseg_diagnostic_init(&diag);
            rc = pulseg_get_tr_gradient_waveforms(coll, &tr_wf, &diag, subseq_idx, tr_idx);
            if (!PULSEG_SUCCEEDED(rc))
            {
                fprintf(
                    stderr,
                    "[DIAG] pulseg_get_tr_gradient_waveforms failed: case=%s subseq=%d tr_idx=%d "
                    "rc=%d\n",
                    tc->name,
                    subseq_idx,
                    tr_idx,
                    rc);
            }
            mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_tr_gradient_waveforms failed");

            int matched = 0;
            for (i = 0; i < ref_wfs.num_trs; ++i)
            {
                if (tr_waveform_matches_ref(&ref_wfs.waveforms[i], &tr_wf))
                {
                    matched = 1;
                    break;
                }
            }
            if (!matched)
            {
                fprintf(
                    stderr,
                    "[FAIL] Canonical TR %d waveform does not match any reference\n",
                    tr_idx);
            }
            mu_assert(matched, "Canonical TR waveform does not match any reference");
            pulseg_tr_gradient_waveforms_free(&tr_wf);
        }

        /* Optionally, check that backend does not return extra canonical TRs. */
        pulseg_tr_gradient_waveforms tr_wf_extra = PULSEG_TR_GRADIENT_WAVEFORMS_INIT;
        pulseg_diagnostic_init(&diag);
        rc = pulseg_get_tr_gradient_waveforms(
            coll,
            &tr_wf_extra,
            &diag,
            subseq_idx,
            ref_wfs.num_trs);
        mu_assert(!PULSEG_SUCCEEDED(rc), "Backend returned more canonical TRs than expected");
        pulseg_tr_gradient_waveforms_free(&tr_wf_extra);

        free_tr_waveform_set(&ref_wfs);
    }
    pulseg_collection_free(coll);
}

MU_TEST(test_sequences_uieval_gre_2d_1sl_1avg)
{
    run_sequences_uieval_case(&kGreCases[0]);
}
MU_TEST(test_sequences_uieval_gre_2d_1sl_3avg)
{
    run_sequences_uieval_case(&kGreCases[1]);
}
MU_TEST(test_sequences_uieval_gre_2d_3sl_1avg)
{
    run_sequences_uieval_case(&kGreCases[2]);
}
MU_TEST(test_sequences_uieval_gre_2d_3sl_3avg)
{
    run_sequences_uieval_case(&kGreCases[3]);
}

MU_TEST(test_sequences_uieval_mprage_2d_1sl_1avg)
{
    run_sequences_uieval_case(&kMprageCases[0]);
}
MU_TEST(test_sequences_uieval_mprage_2d_1sl_3avg)
{
    run_sequences_uieval_case(&kMprageCases[1]);
}
MU_TEST(test_sequences_uieval_mprage_2d_3sl_1avg)
{
    run_sequences_uieval_case(&kMprageCases[2]);
}
MU_TEST(test_sequences_uieval_mprage_2d_3sl_3avg)
{
    run_sequences_uieval_case(&kMprageCases[3]);
}
MU_TEST(test_sequences_uieval_mprage_nc_1sl_1avg)
{
    run_sequences_uieval_case(&kMprageNoncartCases[0]);
}
MU_TEST(test_sequences_uieval_mprage_nc_1sl_3avg_userotext0)
{
    run_sequences_uieval_case(&kMprageNoncartCases[1]);
}
MU_TEST(test_sequences_uieval_mprage_nc_3sl_1avg_userotext0)
{
    run_sequences_uieval_case(&kMprageNoncartCases[2]);
}
MU_TEST(test_sequences_uieval_mprage_nc_3sl_3avg_userotext0)
{
    run_sequences_uieval_case(&kMprageNoncartCases[3]);
}
MU_TEST(test_sequences_uieval_mprage_nc_1sl_1avg_userotext1)
{
    run_sequences_uieval_case(&kMprageNoncartCases[4]);
}
MU_TEST(test_sequences_uieval_mprage_nc_1sl_3avg_userotext1)
{
    run_sequences_uieval_case(&kMprageNoncartCases[5]);
}
MU_TEST(test_sequences_uieval_mprage_nc_3sl_1avg_userotext1)
{
    run_sequences_uieval_case(&kMprageNoncartCases[6]);
}
MU_TEST(test_sequences_uieval_mprage_nc_3sl_3avg_userotext1)
{
    run_sequences_uieval_case(&kMprageNoncartCases[7]);
}

MU_TEST(test_sequences_uieval_bssfp_2d_1sl_1avg)
{
    run_sequences_uieval_case(&kBssfpCases[0]);
}
MU_TEST(test_sequences_uieval_bssfp_2d_1sl_3avg)
{
    run_sequences_uieval_case(&kBssfpCases[1]);
}
MU_TEST(test_sequences_uieval_bssfp_2d_3sl_1avg)
{
    run_sequences_uieval_case(&kBssfpCases[2]);
}
MU_TEST(test_sequences_uieval_bssfp_2d_3sl_3avg)
{
    run_sequences_uieval_case(&kBssfpCases[3]);
}

MU_TEST(test_sequences_uieval_fse_2d_1sl_1avg)
{
    run_sequences_uieval_case(&kFseCases[0]);
}
MU_TEST(test_sequences_uieval_fse_2d_1sl_3avg)
{
    run_sequences_uieval_case(&kFseCases[1]);
}
MU_TEST(test_sequences_uieval_fse_2d_3sl_1avg)
{
    run_sequences_uieval_case(&kFseCases[2]);
}
MU_TEST(test_sequences_uieval_fse_2d_3sl_3avg)
{
    run_sequences_uieval_case(&kFseCases[3]);
}

MU_TEST(test_sequences_uieval_qalas_nc_3d_1sl_1avg_userotext1)
{
    run_sequences_uieval_case(&kQalasNoncartCases[0]);
}
MU_TEST(test_sequences_uieval_qalas_nc_3d_1sl_3avg_userotext1)
{
    run_sequences_uieval_case(&kQalasNoncartCases[1]);
}
MU_TEST(test_sequences_uieval_qalas_nc_3d_3sl_1avg_userotext1)
{
    run_sequences_uieval_case(&kQalasNoncartCases[2]);
}
MU_TEST(test_sequences_uieval_qalas_nc_3d_3sl_3avg_userotext1)
{
    run_sequences_uieval_case(&kQalasNoncartCases[3]);
}

MU_TEST(test_sequences_uieval_mprage_nav_2d_1sl_1avg)
{
    run_sequences_uieval_case(&kMprageNavCases[0]);
}
MU_TEST(test_sequences_uieval_mprage_nav_2d_1sl_3avg)
{
    run_sequences_uieval_case(&kMprageNavCases[1]);
}
MU_TEST(test_sequences_uieval_mprage_nav_2d_3sl_1avg)
{
    run_sequences_uieval_case(&kMprageNavCases[2]);
}
MU_TEST(test_sequences_uieval_mprage_nav_2d_3sl_3avg)
{
    run_sequences_uieval_case(&kMprageNavCases[3]);
}

MU_TEST_SUITE(suite_sequences_uieval)
{
    MU_RUN_TEST(test_sequences_uieval_gre_2d_1sl_1avg);
    MU_RUN_TEST(test_sequences_uieval_gre_2d_1sl_3avg);
    MU_RUN_TEST(test_sequences_uieval_gre_2d_3sl_1avg);
    MU_RUN_TEST(test_sequences_uieval_gre_2d_3sl_3avg);
    MU_RUN_TEST(test_sequences_uieval_mprage_2d_1sl_1avg);
    MU_RUN_TEST(test_sequences_uieval_mprage_2d_1sl_3avg);
    MU_RUN_TEST(test_sequences_uieval_mprage_2d_3sl_1avg);
    MU_RUN_TEST(test_sequences_uieval_mprage_2d_3sl_3avg);
    MU_RUN_TEST(test_sequences_uieval_mprage_nc_1sl_1avg);
    MU_RUN_TEST(test_sequences_uieval_mprage_nc_1sl_3avg_userotext0);
    MU_RUN_TEST(test_sequences_uieval_mprage_nc_3sl_1avg_userotext0);
    MU_RUN_TEST(test_sequences_uieval_mprage_nc_3sl_3avg_userotext0);
    MU_RUN_TEST(test_sequences_uieval_mprage_nc_1sl_1avg_userotext1);
    MU_RUN_TEST(test_sequences_uieval_mprage_nc_1sl_3avg_userotext1);
    MU_RUN_TEST(test_sequences_uieval_mprage_nc_3sl_1avg_userotext1);
    MU_RUN_TEST(test_sequences_uieval_mprage_nc_3sl_3avg_userotext1);
    MU_RUN_TEST(test_sequences_uieval_bssfp_2d_1sl_1avg);
    MU_RUN_TEST(test_sequences_uieval_bssfp_2d_1sl_3avg);
    MU_RUN_TEST(test_sequences_uieval_bssfp_2d_3sl_1avg);
    MU_RUN_TEST(test_sequences_uieval_bssfp_2d_3sl_3avg);
    MU_RUN_TEST(test_sequences_uieval_fse_2d_1sl_1avg);
    MU_RUN_TEST(test_sequences_uieval_fse_2d_1sl_3avg);
    MU_RUN_TEST(test_sequences_uieval_fse_2d_3sl_1avg);
    MU_RUN_TEST(test_sequences_uieval_fse_2d_3sl_3avg);
    MU_RUN_TEST(test_sequences_uieval_qalas_nc_3d_1sl_1avg_userotext1);
    MU_RUN_TEST(test_sequences_uieval_qalas_nc_3d_1sl_3avg_userotext1);
    MU_RUN_TEST(test_sequences_uieval_qalas_nc_3d_3sl_1avg_userotext1);
    MU_RUN_TEST(test_sequences_uieval_qalas_nc_3d_3sl_3avg_userotext1);
    MU_RUN_TEST(test_sequences_uieval_mprage_nav_2d_1sl_1avg);
    MU_RUN_TEST(test_sequences_uieval_mprage_nav_2d_1sl_3avg);
    MU_RUN_TEST(test_sequences_uieval_mprage_nav_2d_3sl_1avg);
    MU_RUN_TEST(test_sequences_uieval_mprage_nav_2d_3sl_3avg);
}

/* ------------------------------------------------------------------ */
/*  Phase 3: geninstruction pipeline validation                        */
/*                                                                     */
/*  Mirrors example_geninstructions.c block-walk against the binary   */
/*  segment definition produced by export_segment_def in MATLAB.      */
/*  For each segment/block we check:                                   */
/*    - flags  (has_rf, has_grad[3], has_adc, has_rotation,           */
/*              has_digital_out)                                       */
/*    - RF     (delay, amp, num_samples, waveform shape, time array)   */
/*    - Grads  (delay, amp, num_samples, waveform shape, time array)   */
/*    - ADC    (delay, adc_def_id)                                     */
/*    - Digitalout (delay, duration)                                   */
/*    - Anchors (rf_isocenter_us, adc_kzero_us)                        */
/*  Per segment:                                                       */
/*    - Segment-level gaps (rf_adc_gap_us, adc_adc_gap_us) vs truth    */
/*    - Segment properties (pure_delay, trigger params)                */
/*    - Event walk gap verification (mirrors walk_segment_events)      */
/* ------------------------------------------------------------------ */

#define GENI_AMP_REL_TOL 1e-3f  /* relative tolerance for normalised amps */
#define GENI_DELAY_ABS_TOL 1.0f /* us — half a raster step                */

/* Relative amplitude comparison with absolute floor of 1.0 */
#define GENI_AMP_NEAR(a, b) \
    (fabsf((a) - (b)) <= (((fabsf(a) > 1.0f ? fabsf(a) : 1.0f)) * GENI_AMP_REL_TOL))

static void run_sequences_geninstructions_case(const seq_case *tc)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    pulseg_collection_info cinfo = PULSEG_COLLECTION_INFO_INIT;
    static seg_def_file ref; /* static: too large (~8 MB) for stack */
    char seg_path[512];
    int rc, ok;
    int s, b, ax;

    /* Load sequence */
    gre_opts_init(&opts);
    rc = load_seq_with_averages(&coll, tc->seq_file, &opts, tc->num_averages);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed for GRE test case");

    rc = pulseg_get_collection_info(coll, &cinfo);
    mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_collection_info failed");

    /* Load MATLAB ground truth */
    build_case_path(seg_path, sizeof(seg_path), tc, "_segment_def.truth");
    ok = parse_seg_def(seg_path, &ref);
    mu_assert(ok, "failed to parse case _segment_def.truth");

    /* Number of segments must match */
    fprintf(
        stderr,
        "[geninstr][%s] num_segments: ref=%d  lib=%d\n",
        tc->name,
        ref.num_segments,
        cinfo.num_segments);
    mu_assert_int_eq(ref.num_segments, cinfo.num_segments);

    for (s = 0; s < ref.num_segments; ++s)
    {
        pulseg_segment_info segi = PULSEG_SEGMENT_INFO_INIT;
        rc = pulseg_get_segment_info(coll, &segi, s);
        mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_segment_info failed");
        fprintf(
            stderr,
            "[geninstr][%s] segment %d num_blocks: ref=%d  lib=%d\n",
            tc->name,
            s,
            ref.num_blocks[s],
            segi.num_blocks);
        mu_assert_int_eq(ref.num_blocks[s], segi.num_blocks);

        for (b = 0; b < ref.num_blocks[s]; ++b)
        {
            const seg_block_def *ref_blk = &ref.blocks[s][b];
            pulseg_block_info bi = PULSEG_BLOCK_INFO_INIT;
            rc = pulseg_get_block_info(coll, &bi, s, b);
            mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_block_info failed");

            /* --- Flags -------------------------------------------- */
            mu_assert_int_eq(ref_blk->has_rf, bi.has_rf);
            for (ax = 0; ax < 3; ++ax)
                mu_assert_int_eq(ref_blk->has_grad[ax], bi.has_grad[ax]);
            mu_assert_int_eq(ref_blk->has_adc, bi.has_adc);
            mu_assert_int_eq(ref_blk->has_rotation, bi.has_rotation);
            mu_assert_int_eq(ref_blk->has_digital_out, bi.has_digitalout);

            /* --- RF ----------------------------------------------- */
            if (ref_blk->has_rf)
            {
                int num_channels = 0, num_samples = 0;
                float **mag;
                int i;

                mu_assert(
                    fabsf(ref_blk->rf_delay - (float)bi.rf_delay_us * 1e-6f) <=
                        GENI_DELAY_ABS_TOL * 1e-6f,
                    "RF delay mismatch");
                mu_assert_int_eq(ref_blk->rf_n, bi.rf_num_samples);

                mag = pulseg_get_rf_magnitude(coll, &num_channels, &num_samples, s, b);
                mu_assert(mag != NULL, "pulseg_get_rf_magnitude returned NULL");
                mu_assert_int_eq(ref_blk->rf_n, num_samples);

                /* Both library and MATLAB store normalised shapes
                   (peak ≈ 1.0).  Compare directly. */
                for (i = 0; i < num_samples; ++i)
                {
                    mu_assert(
                        GENI_AMP_NEAR(ref_blk->rf_rho[i], mag[0][i]),
                        "RF magnitude shape mismatch");
                }

                /* Amplitude checks via new getters */
                {
                    float init_amp = pulseg_get_rf_initial_amplitude_hz(coll, s, b);
                    float max_amp = pulseg_get_rf_max_amplitude_hz(coll, s, b);
                    mu_assert(
                        GENI_AMP_NEAR(ref_blk->rf_amp, init_amp),
                        "RF initial amplitude mismatch");
                    mu_assert(
                        GENI_AMP_NEAR(fabsf(ref_blk->rf_amp), max_amp),
                        "RF max amplitude mismatch");
                }

                {
                    int ch;
                    for (ch = 0; ch < num_channels; ++ch)
                        free(mag[ch]);
                    free(mag);
                }

                /* RF timing array check */
                {
                    float *lib_rf_time = pulseg_get_rf_time_us(coll, s, b);
                    if (lib_rf_time != NULL)
                    {
                        /* Non-uniform raster: compare against truth rf_time_s */
                        for (i = 0; i < ref_blk->rf_n; ++i)
                        {
                            float ref_us = ref_blk->rf_time_s[i] * 1e6f;
                            float lib_us = lib_rf_time[i];
                            if (fabsf(ref_us - lib_us) > 0.5f)
                                fprintf(
                                    stderr,
                                    "[geninstr][%s] seg%d blk%d rf_time@%d: ref=%.6f us  lib=%.6f "
                                    "us\n",
                                    tc->name,
                                    s,
                                    b,
                                    i,
                                    ref_us,
                                    lib_us);
                            mu_assert(
                                fabsf(ref_us - lib_us) <= 0.5f,
                                "RF time array mismatch (>0.5 us)");
                        }
                        free(lib_rf_time);
                    }
                    else
                    {
                        /* Uniform raster: verify truth times match raster expectation */
                        for (i = 0; i < ref_blk->rf_n; ++i)
                        {
                            float ref_us = ref_blk->rf_time_s[i] * 1e6f;
                            float expected_us = (float)i * ref_blk->rf_raster_us;
                            if (fabsf(ref_us - expected_us) > 0.5f)
                                fprintf(
                                    stderr,
                                    "[geninstr][%s] seg%d blk%d rf_time@%d: ref=%.6f us  "
                                    "expected=%.6f us (uniform)\n",
                                    tc->name,
                                    s,
                                    b,
                                    i,
                                    ref_us,
                                    expected_us);
                            mu_assert(
                                fabsf(ref_us - expected_us) <= 0.5f,
                                "RF time array does not match uniform raster");
                        }
                    }
                }
            }

            /* --- Gradients ---------------------------------------- */
            for (ax = 0; ax < 3; ++ax)
            {
                if (ref_blk->has_grad[ax])
                {
                    int num_shots = 0, num_samples = 0;
                    float **amps;
                    int i;
                    int init_shot;

                    mu_assert(
                        fabsf(ref_blk->grad_delay[ax] - (float)bi.grad_delay_us[ax] * 1e-6f) <=
                            GENI_DELAY_ABS_TOL * 1e-6f,
                        "grad delay mismatch");
                    mu_assert_int_eq(ref_blk->grad_n[ax], bi.grad_num_samples[ax]);

                    amps = pulseg_get_grad_amplitude(coll, &num_shots, &num_samples, s, b, ax);
                    mu_assert(amps != NULL, "pulseg_get_grad_amplitude returned NULL");
                    mu_assert_int_eq(ref_blk->grad_n[ax], num_samples);

                    /* The shape the max-energy segment instance plays --
                       a pulseq shape id now, not an ordinal into a
                       per-definition table, so it is bounded by the shape
                       library rather than by a shot count.  0 means the
                       gradient is a trapezoid. */
                    init_shot = pulseg_get_grad_initial_shape_id(coll, s, b, ax);
                    mu_assert(init_shot >= 0, "initial shape id out of range");
                    /* One waveform per definition now -- the representative
                     * one -- so it is amps[0].  init_shot is a pulseq shape
                     * id and would index far past the end. */
                    mu_assert_int_eq(1, num_shots);
                    for (i = 0; i < num_samples; ++i)
                    {
                        if (!GENI_AMP_NEAR(ref_blk->grad_wave[ax][i], amps[0][i]))
                            fprintf(
                                stderr,
                                "[geninstr][%s] seg%d blk%d ax%d wave@%d: ref=%.4g  lib=%.4g "
                                "(shape=%d)\n",
                                tc->name,
                                s,
                                b,
                                ax,
                                i,
                                ref_blk->grad_wave[ax][i],
                                amps[0][i],
                                init_shot);
                        mu_assert(
                            GENI_AMP_NEAR(ref_blk->grad_wave[ax][i], amps[0][i]),
                            "grad waveform shape mismatch");
                    }

                    /* Gradient timing array check */
                    {
                        float *lib_time_us = pulseg_get_grad_time_us(coll, s, b, ax);
                        mu_assert(lib_time_us != NULL, "pulseg_get_grad_time_us returned NULL");
                        for (i = 0; i < ref_blk->grad_n[ax]; ++i)
                        {
                            float ref_us = ref_blk->grad_time_s[ax][i] * 1e6f;
                            float lib_us = lib_time_us[i];
                            if (fabsf(ref_us - lib_us) > 0.5f)
                                fprintf(
                                    stderr,
                                    "[geninstr][%s] seg%d blk%d ax%d time@%d: ref=%.6f us  "
                                    "lib=%.6f us\n",
                                    tc->name,
                                    s,
                                    b,
                                    ax,
                                    i,
                                    ref_us,
                                    lib_us);
                            mu_assert(
                                fabsf(ref_us - lib_us) <= 0.5f,
                                "grad time array mismatch (>0.5 us)");
                        }
                        free(lib_time_us);
                    }

                    /* Amplitude checks via new getters */
                    {
                        float init_amp = pulseg_get_grad_initial_amplitude_hz_per_m(coll, s, b, ax);
                        float max_amp = pulseg_get_grad_max_amplitude_hz_per_m(coll, s, b, ax);
                        if (!GENI_AMP_NEAR(ref_blk->grad_amp[ax], init_amp))
                            fprintf(
                                stderr,
                                "[geninstr][%s] seg%d blk%d ax%d init_amp: ref=%.6g  lib=%.6g\n",
                                tc->name,
                                s,
                                b,
                                ax,
                                ref_blk->grad_amp[ax],
                                init_amp);
                        mu_assert(
                            GENI_AMP_NEAR(ref_blk->grad_amp[ax], init_amp),
                            "grad initial amplitude mismatch");
                        if (!GENI_AMP_NEAR(fabsf(ref_blk->grad_amp[ax]), max_amp))
                            fprintf(
                                stderr,
                                "[geninstr][%s] seg%d blk%d ax%d max_amp: ref=%.6g  lib=%.6g\n",
                                tc->name,
                                s,
                                b,
                                ax,
                                fabsf(ref_blk->grad_amp[ax]),
                                max_amp);
                        mu_assert(
                            GENI_AMP_NEAR(fabsf(ref_blk->grad_amp[ax]), max_amp),
                            "grad max amplitude mismatch");
                    }

                    {
                        int sh;
                        for (sh = 0; sh < num_shots; ++sh)
                            free(amps[sh]);
                        free(amps);
                    }
                }
            }

            /* --- ADC ---------------------------------------------- */
            if (ref_blk->has_adc)
            {
                mu_assert(
                    fabsf(ref_blk->adc_delay - (float)bi.adc_delay_us * 1e-6f) <=
                        GENI_DELAY_ABS_TOL * 1e-6f,
                    "ADC delay mismatch");
                mu_assert_int_eq(ref_blk->adc_def_id, bi.adc_def_id);
            }

            /* --- Digital output ------------------------------------ */
            if (ref_blk->has_digital_out)
            {
                mu_assert(
                    fabsf(ref_blk->digital_out_delay - (float)bi.digitalout_delay_us * 1e-6f) <=
                        GENI_DELAY_ABS_TOL * 1e-6f,
                    "digital-out delay mismatch");
                mu_assert(
                    fabsf(
                        ref_blk->digital_out_duration - (float)bi.digitalout_duration_us * 1e-6f) <=
                        GENI_DELAY_ABS_TOL * 1e-6f,
                    "digital-out duration mismatch");
            }

            /* --- Pure-delay segment-def duration canonicalization -- */
            /* Only single-block pure-delay segments have their duration
             * canonicalized to one block raster at geninstruction time.
             * Pure-delay blocks embedded in multi-block standard segments
             * carry their actual fixed duration set by the sequence. */
            if (!ref_blk->has_rf && !ref_blk->has_grad[0] && !ref_blk->has_grad[1] &&
                !ref_blk->has_grad[2] && !ref_blk->has_adc && ref.num_blocks[s] == 1)
            {
                int expected_delay_us = (int)(opts.block_raster_us + 0.5f);
                mu_assert_int_eq(expected_delay_us, bi.duration_us);
            }

            /* --- Anchors ------------------------------------------ */
            if (ref_blk->has_rf)
            {
                float lib_iso = pulseg_get_rf_isocenter_us(coll, s, b);
                mu_assert(
                    fabsf(ref_blk->rf_isocenter_us - lib_iso) <= 1.0f,
                    "RF isocenter_us mismatch");
            }
            if (ref_blk->has_adc)
            {
                float lib_kz = pulseg_get_adc_kzero_us(coll, s, b);
                if (fabsf(ref_blk->adc_kzero_us - lib_kz) > 1.0f)
                    fprintf(
                        stderr,
                        "[geninstr][%s] seg%d blk%d adc_kzero_us: ref=%.4g  lib=%.4g\n",
                        tc->name,
                        s,
                        b,
                        ref_blk->adc_kzero_us,
                        lib_kz);
                mu_assert(fabsf(ref_blk->adc_kzero_us - lib_kz) <= 1.0f, "ADC kzero_us mismatch");
            }
        }

        /* --- Segment-level gaps ------------------------------- */
        {
            int ref_rf_adc = (int)roundf(ref.rf_adc_gap_us[s]);
            int ref_adc_adc = (int)roundf(ref.adc_adc_gap_us[s]);

            if (ref_rf_adc >= 0)
            {
                mu_assert_int_eq(ref_rf_adc, segi.rf_adc_gap_us);
            }
            else
            {
                mu_assert_int_eq(ref_rf_adc, segi.rf_adc_gap_us);
            }
            if (ref_adc_adc >= 0)
            {
                mu_assert_int_eq(ref_adc_adc, segi.adc_adc_gap_us);
            }
            else
            {
                mu_assert_int_eq(ref_adc_adc, segi.adc_adc_gap_us);
            }
        }

        /* --- Segment property structural checks ------------------- */
        /*
         * Mirrors example_geninstructions.c / example_check.c:
         *   - pure_delay: segment with 1 block, no RF, no grads
         *   - is_nav: exposed by segment_info for scanloop PMC logic
         *   - has_trigger: exposed for vendor trigger arming
         *   - trigger_delay_us / trigger_duration_us: valid when has_trigger
         */
        {
            int seg_has_active_block = 0;
            int seg_block_count = ref.num_blocks[s];
            int sb;

            for (sb = 0; sb < seg_block_count; ++sb)
            {
                const seg_block_def *sblk = &ref.blocks[s][sb];
                if (sblk->has_rf || sblk->has_grad[0] || sblk->has_grad[1] || sblk->has_grad[2])
                {
                    seg_has_active_block = 1;
                    break;
                }
            }

            /* pure_delay: 1 block, no RF, no grads (may have ADC trigger) */
            if (seg_block_count == 1 && !seg_has_active_block)
            {
                mu_assert(segi.pure_delay, "segment with 1 empty block should be pure_delay");
            }
            else
            {
                mu_assert(!segi.pure_delay, "segment with active blocks should not be pure_delay");
            }

            /* has_trigger: when set, delay and duration must be valid */
            if (segi.has_trigger)
            {
                mu_assert(
                    segi.trigger_delay_us >= 0,
                    "trigger_delay_us should be >= 0 when has_trigger");
                mu_assert(
                    segi.trigger_duration_us >= 0,
                    "trigger_duration_us should be >= 0 when has_trigger");
            }
            else
            {
                mu_assert_int_eq(-1, segi.trigger_delay_us);
                mu_assert_int_eq(-1, segi.trigger_duration_us);
            }
        }

        /* --- Event walk gap verification (mirrors walk_segment_events
         *     in example_geninstructions.c) ------------------------ */
        {
            typedef struct
            {
                int kind;
                int start_us;
                int end_us;
            } gap_evt;
            gap_evt events[SEG_DEF_MAX_BLOCKS * 2];
            int n_evt = 0, t_walk = 0, wb, je;
            int walk_rf_adc = -1, walk_adc_adc = -1;

            for (wb = 0; wb < ref.num_blocks[s]; ++wb)
            {
                const seg_block_def *wb_blk = &ref.blocks[s][wb];
                pulseg_block_info wbi = PULSEG_BLOCK_INFO_INIT;
                rc = pulseg_get_block_info(coll, &wbi, s, wb);
                mu_assert(PULSEG_SUCCEEDED(rc), "get_block_info for event walk");

                if (wb_blk->has_rf)
                {
                    /* Use the library's own RF duration (accounts for custom
                     * non-uniform time shapes, e.g. hard pulses in noncart
                     * sequences where the last time-shape sample can equal the
                     * full block duration rather than N*raster - raster/2). */
                    int rf_dur = wbi.rf_duration_us;
                    if (rf_dur < 0)
                    {
                        /* fallback: uniform-raster bin-centre convention */
                        int rf_raster_int = (int)(opts.rf_raster_us + 0.5f);
                        rf_dur = wbi.rf_num_samples * rf_raster_int - rf_raster_int / 2;
                    }
                    events[n_evt].kind = 0; /* RF */
                    events[n_evt].start_us = t_walk + wbi.rf_delay_us;
                    events[n_evt].end_us = t_walk + wbi.rf_delay_us + rf_dur;
                    n_evt++;
                }
                if (wb_blk->has_adc && wbi.adc_def_id >= 0)
                {
                    pulseg_adc_def ad = PULSEG_ADC_DEF_INIT;
                    int adc_dur;
                    rc = pulseg_get_adc_def(coll, &ad, wbi.adc_def_id);
                    mu_assert(PULSEG_SUCCEEDED(rc), "get_adc_def for walk");
                    adc_dur = (int)(ad.num_samples * ad.dwell_ns * 1e-3f);
                    events[n_evt].kind = 1; /* ADC */
                    events[n_evt].start_us = t_walk + wbi.adc_delay_us;
                    events[n_evt].end_us = t_walk + wbi.adc_delay_us + adc_dur;
                    n_evt++;
                }
                t_walk += wbi.duration_us;
            }

            /* insertion-sort by start_us (stable) */
            for (je = 1; je < n_evt; ++je)
            {
                gap_evt tmp = events[je];
                int k = je - 1;
                while (k >= 0 && events[k].start_us > tmp.start_us)
                {
                    events[k + 1] = events[k];
                    --k;
                }
                events[k + 1] = tmp;
            }

            /* walk events — compute minimum RF->ADC and ADC->ADC gaps */
            for (je = 0; je < n_evt; ++je)
            {
                if (events[je].kind == 0)
                { /* RF */
                    int k2;
                    for (k2 = je + 1; k2 < n_evt; ++k2)
                    {
                        if (events[k2].kind == 1)
                        { /* next ADC */
                            int gap = events[k2].start_us - events[je].end_us;
                            if (walk_rf_adc < 0 || gap < walk_rf_adc)
                                walk_rf_adc = gap;
                            break;
                        }
                        if (events[k2].kind == 0)
                            break; /* next RF first */
                    }
                }
                else
                { /* ADC */
                    if (je > 0 && events[je - 1].kind == 1)
                    { /* prev ADC */
                        int gap = events[je].start_us - events[je - 1].end_us;
                        if (walk_adc_adc < 0 || gap < walk_adc_adc)
                            walk_adc_adc = gap;
                    }
                }
            }

            /* Computed gaps must agree with segment_info */
            mu_assert_int_eq(walk_rf_adc, segi.rf_adc_gap_us);
            mu_assert_int_eq(walk_adc_adc, segi.adc_adc_gap_us);
        }
    }

    pulseg_collection_free(coll);
}

MU_TEST(test_sequences_geninstructions_gre_2d_1sl_1avg)
{
    run_sequences_geninstructions_case(&kGreCases[0]);
}
MU_TEST(test_sequences_geninstructions_gre_2d_1sl_3avg)
{
    run_sequences_geninstructions_case(&kGreCases[1]);
}
MU_TEST(test_sequences_geninstructions_gre_2d_3sl_1avg)
{
    run_sequences_geninstructions_case(&kGreCases[2]);
}
MU_TEST(test_sequences_geninstructions_gre_2d_3sl_3avg)
{
    run_sequences_geninstructions_case(&kGreCases[3]);
}

MU_TEST(test_sequences_geninstructions_mprage_2d_1sl_1avg)
{
    run_sequences_geninstructions_case(&kMprageCases[0]);
}
MU_TEST(test_sequences_geninstructions_mprage_2d_1sl_3avg)
{
    run_sequences_geninstructions_case(&kMprageCases[1]);
}
MU_TEST(test_sequences_geninstructions_mprage_2d_3sl_1avg)
{
    run_sequences_geninstructions_case(&kMprageCases[2]);
}
MU_TEST(test_sequences_geninstructions_mprage_2d_3sl_3avg)
{
    run_sequences_geninstructions_case(&kMprageCases[3]);
}
MU_TEST(test_sequences_geninstructions_mprage_nc_1sl_1avg)
{
    run_sequences_geninstructions_case(&kMprageNoncartCases[0]);
}
MU_TEST(test_sequences_geninstructions_mprage_nc_1sl_3avg_userotext0)
{
    run_sequences_geninstructions_case(&kMprageNoncartCases[1]);
}
MU_TEST(test_sequences_geninstructions_mprage_nc_3sl_1avg_userotext0)
{
    run_sequences_geninstructions_case(&kMprageNoncartCases[2]);
}
MU_TEST(test_sequences_geninstructions_mprage_nc_3sl_3avg_userotext0)
{
    run_sequences_geninstructions_case(&kMprageNoncartCases[3]);
}
MU_TEST(test_sequences_geninstructions_mprage_nc_1sl_1avg_userotext1)
{
    run_sequences_geninstructions_case(&kMprageNoncartCases[4]);
}
MU_TEST(test_sequences_geninstructions_mprage_nc_1sl_3avg_userotext1)
{
    run_sequences_geninstructions_case(&kMprageNoncartCases[5]);
}
MU_TEST(test_sequences_geninstructions_mprage_nc_3sl_1avg_userotext1)
{
    run_sequences_geninstructions_case(&kMprageNoncartCases[6]);
}
MU_TEST(test_sequences_geninstructions_mprage_nc_3sl_3avg_userotext1)
{
    run_sequences_geninstructions_case(&kMprageNoncartCases[7]);
}

MU_TEST(test_sequences_geninstructions_bssfp_2d_1sl_1avg)
{
    run_sequences_geninstructions_case(&kBssfpCases[0]);
}
MU_TEST(test_sequences_geninstructions_bssfp_2d_1sl_3avg)
{
    run_sequences_geninstructions_case(&kBssfpCases[1]);
}
MU_TEST(test_sequences_geninstructions_bssfp_2d_3sl_1avg)
{
    run_sequences_geninstructions_case(&kBssfpCases[2]);
}
MU_TEST(test_sequences_geninstructions_bssfp_2d_3sl_3avg)
{
    run_sequences_geninstructions_case(&kBssfpCases[3]);
}

MU_TEST(test_sequences_geninstructions_fse_2d_1sl_1avg)
{
    run_sequences_geninstructions_case(&kFseCases[0]);
}
MU_TEST(test_sequences_geninstructions_fse_2d_1sl_3avg)
{
    run_sequences_geninstructions_case(&kFseCases[1]);
}
MU_TEST(test_sequences_geninstructions_fse_2d_3sl_1avg)
{
    run_sequences_geninstructions_case(&kFseCases[2]);
}
MU_TEST(test_sequences_geninstructions_fse_2d_3sl_3avg)
{
    run_sequences_geninstructions_case(&kFseCases[3]);
}

MU_TEST(test_sequences_geninstructions_qalas_nc_3d_1sl_1avg_userotext1)
{
    run_sequences_geninstructions_case(&kQalasNoncartCases[0]);
}
MU_TEST(test_sequences_geninstructions_qalas_nc_3d_1sl_3avg_userotext1)
{
    run_sequences_geninstructions_case(&kQalasNoncartCases[1]);
}
MU_TEST(test_sequences_geninstructions_qalas_nc_3d_3sl_1avg_userotext1)
{
    run_sequences_geninstructions_case(&kQalasNoncartCases[2]);
}
MU_TEST(test_sequences_geninstructions_qalas_nc_3d_3sl_3avg_userotext1)
{
    run_sequences_geninstructions_case(&kQalasNoncartCases[3]);
}

MU_TEST(test_sequences_geninstructions_mprage_nav_2d_1sl_1avg)
{
    run_sequences_geninstructions_case(&kMprageNavCases[0]);
}
MU_TEST(test_sequences_geninstructions_mprage_nav_2d_1sl_3avg)
{
    run_sequences_geninstructions_case(&kMprageNavCases[1]);
}
MU_TEST(test_sequences_geninstructions_mprage_nav_2d_3sl_1avg)
{
    run_sequences_geninstructions_case(&kMprageNavCases[2]);
}
MU_TEST(test_sequences_geninstructions_mprage_nav_2d_3sl_3avg)
{
    run_sequences_geninstructions_case(&kMprageNavCases[3]);
}


MU_TEST_SUITE(suite_sequences_geninstructions)
{
    MU_RUN_TEST(test_sequences_geninstructions_gre_2d_1sl_1avg);
    MU_RUN_TEST(test_sequences_geninstructions_gre_2d_1sl_3avg);
    MU_RUN_TEST(test_sequences_geninstructions_gre_2d_3sl_1avg);
    MU_RUN_TEST(test_sequences_geninstructions_gre_2d_3sl_3avg);
    MU_RUN_TEST(test_sequences_geninstructions_mprage_2d_1sl_1avg);
    MU_RUN_TEST(test_sequences_geninstructions_mprage_2d_1sl_3avg);
    MU_RUN_TEST(test_sequences_geninstructions_mprage_2d_3sl_1avg);
    MU_RUN_TEST(test_sequences_geninstructions_mprage_2d_3sl_3avg);
    MU_RUN_TEST(test_sequences_geninstructions_mprage_nc_1sl_1avg);
    MU_RUN_TEST(test_sequences_geninstructions_mprage_nc_1sl_3avg_userotext0);
    MU_RUN_TEST(test_sequences_geninstructions_mprage_nc_3sl_1avg_userotext0);
    MU_RUN_TEST(test_sequences_geninstructions_mprage_nc_3sl_3avg_userotext0);
    MU_RUN_TEST(test_sequences_geninstructions_mprage_nc_1sl_1avg_userotext1);
    MU_RUN_TEST(test_sequences_geninstructions_mprage_nc_1sl_3avg_userotext1);
    MU_RUN_TEST(test_sequences_geninstructions_mprage_nc_3sl_1avg_userotext1);
    MU_RUN_TEST(test_sequences_geninstructions_mprage_nc_3sl_3avg_userotext1);
    MU_RUN_TEST(test_sequences_geninstructions_bssfp_2d_1sl_1avg);
    MU_RUN_TEST(test_sequences_geninstructions_bssfp_2d_1sl_3avg);
    MU_RUN_TEST(test_sequences_geninstructions_bssfp_2d_3sl_1avg);
    MU_RUN_TEST(test_sequences_geninstructions_bssfp_2d_3sl_3avg);
    MU_RUN_TEST(test_sequences_geninstructions_fse_2d_1sl_1avg);
    MU_RUN_TEST(test_sequences_geninstructions_fse_2d_1sl_3avg);
    MU_RUN_TEST(test_sequences_geninstructions_fse_2d_3sl_1avg);
    MU_RUN_TEST(test_sequences_geninstructions_fse_2d_3sl_3avg);
    MU_RUN_TEST(test_sequences_geninstructions_qalas_nc_3d_1sl_1avg_userotext1);
    MU_RUN_TEST(test_sequences_geninstructions_qalas_nc_3d_1sl_3avg_userotext1);
    MU_RUN_TEST(test_sequences_geninstructions_qalas_nc_3d_3sl_1avg_userotext1);
    MU_RUN_TEST(test_sequences_geninstructions_qalas_nc_3d_3sl_3avg_userotext1);
    MU_RUN_TEST(test_sequences_geninstructions_mprage_nav_2d_1sl_1avg);
    MU_RUN_TEST(test_sequences_geninstructions_mprage_nav_2d_1sl_3avg);
    MU_RUN_TEST(test_sequences_geninstructions_mprage_nav_2d_3sl_1avg);
    MU_RUN_TEST(test_sequences_geninstructions_mprage_nav_2d_3sl_3avg);
}

/* ------------------------------------------------------------------ */
/*  Phase 5: Scan table — block instance validation                   */
/* ------------------------------------------------------------------ */

static void run_exec_stream_case(const seq_case *tc)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    pulseg_collection_info cinfo = PULSEG_COLLECTION_INFO_INIT;
    exec_stream_file ref = SCAN_TABLE_FILE_INIT;
    seg_meta meta = SEG_META_INIT;
    char scan_path[512];
    char meta_path[512];
    int rc, ok, pos;
    int is_mprage;
    int pure_seg_id = -1;
    int pure_seg_duration_us = -1;
    int pure_inst_durations[8];
    int pure_inst_unique = 0;
    int saw_noncanonical_instance = 0;

    /* Segment patterns per subsequence: built from segment_order + segment_num_blocks */
    int pattern_seg_id[16][MAX_SCAN_TABLE_ENTRIES];
    int pattern_len[16];
    int pattern_pos[16];
    int subseq_seg_start[16];
    int subseq_seg_end[16];
    int total_readout_count = 0;
    int seg_trigger_seen[MAX_SEGMENTS];

    memset(pure_inst_durations, 0, sizeof(pure_inst_durations));
    memset(seg_trigger_seen, 0, sizeof(seg_trigger_seen));
    memset(pattern_len, 0, sizeof(pattern_len));
    memset(pattern_pos, 0, sizeof(pattern_pos));
    memset(subseq_seg_start, 0, sizeof(subseq_seg_start));
    memset(subseq_seg_end, 0, sizeof(subseq_seg_end));
    is_mprage =
        (strncmp(tc->name, "mprage_", 7) == 0 && strncmp(tc->name, "mprage_nav_", 11) != 0) ? 1 : 0;

    gre_opts_init(&opts);
    rc = load_seq_with_averages(&coll, tc->seq_file, &opts, tc->num_averages);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

    rc = pulseg_get_collection_info(coll, &cinfo);
    mu_assert(PULSEG_SUCCEEDED(rc), "get_collection_info in scan table test");

    build_case_path(scan_path, sizeof(scan_path), tc, "_scan_table.truth");
    ok = parse_exec_stream(scan_path, &ref);
    mu_assert(ok, "failed to parse scan_table.truth");
    mu_assert(ref.num_entries > 0, "exec_stream has no entries");

    /* Parse meta to get segment_order for validation */
    build_case_path(meta_path, sizeof(meta_path), tc, "_meta.txt");
    ok = parse_meta(meta_path, &meta);
    mu_assert(ok, "failed to parse case _meta.txt");
    mu_assert(meta.num_segment_order_entries > 0, "meta has no segment_order entries");
    mu_assert_int_eq(meta.num_segments, cinfo.num_segments);

    mu_assert(
        cinfo.num_subsequences > 0 && cinfo.num_subsequences <= 16,
        "unsupported number of subsequences for scan-table pattern validation");

    {
        int s;
        pulseg_subseq_info si = PULSEG_SUBSEQ_INFO_INIT;
        for (s = 0; s < cinfo.num_subsequences; ++s)
        {
            rc = pulseg_get_subseq_info(coll, &si, s);
            mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_subseq_info failed");
            subseq_seg_start[s] = si.segment_offset;
        }
        for (s = 0; s < cinfo.num_subsequences; ++s)
        {
            if (s + 1 < cinfo.num_subsequences)
                subseq_seg_end[s] = subseq_seg_start[s + 1];
            else
                subseq_seg_end[s] = meta.num_segments;
            mu_assert(
                subseq_seg_end[s] > subseq_seg_start[s],
                "invalid subsequence segment-id range");
        }
    }

    /* Build per-block pattern from segment_order + segment_num_blocks.
     * One TR = walking segment_order, each entry lasting segment_num_blocks[seg] blocks.
     * For collections, each subsequence uses only its own segment-id range. */
    {
        int s;
        for (s = 0; s < cinfo.num_subsequences; ++s)
        {
            int si;
            for (si = 0; si < meta.num_segment_order_entries; ++si)
            {
                int seg = meta.segment_order[si];
                int nblk;
                int bi;
                if (seg < subseq_seg_start[s] || seg >= subseq_seg_end[s])
                    continue;
                nblk = meta.segment_num_blocks[seg];
                mu_assert(
                    seg >= 0 && seg < meta.num_segments,
                    "segment_order references invalid segment");
                for (bi = 0; bi < nblk; ++bi)
                {
                    mu_assert(
                        pattern_len[s] < MAX_SCAN_TABLE_ENTRIES,
                        "segment pattern exceeds max entries");
                    pattern_seg_id[s][pattern_len[s]] = seg;
                    pattern_len[s]++;
                }
            }
            mu_assert(pattern_len[s] > 0, "subsequence segment pattern is empty");
        }
    }

    /* Walk the scan table via cursor and compare each block instance */
    pulseg_cursor_reset(coll);
    pos = 0;

    while (pulseg_cursor_next(coll) == PULSEG_CURSOR_BLOCK)
    {
        pulseg_block_instance inst = PULSEG_BLOCK_INSTANCE_INIT;
        pulseg_cursor_info ci = PULSEG_CURSOR_INFO_INIT;
        const exec_stream_entry *e;
        float tol;
        int i;

        mu_assert(pos < ref.num_entries, "more blocks than exec_stream entries");

        rc = pulseg_get_block_instance(coll, &inst);
        mu_assert(PULSEG_SUCCEEDED(rc), "get_block_instance failed");

        rc = pulseg_cursor_get_info(coll, &ci);
        mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_cursor_get_info failed");

        /* ---- Segment validation: pattern-based ---------------------- */
        /* Cross-check nav/trigger at segment starts */
        if (ci.segment_start)
        {
            pulseg_segment_info si = PULSEG_SEGMENT_INFO_INIT;
            rc = pulseg_get_segment_info(coll, &si, ci.segment_id);
            mu_assert(PULSEG_SUCCEEDED(rc), "get_segment_info for cursor cross-check");
            mu_assert_int_eq(si.is_nav, ci.is_nav);
            mu_assert_int_eq(si.has_trigger, ci.has_trigger);
        }

        if (is_mprage)
        {
            pulseg_segment_info segi = PULSEG_SEGMENT_INFO_INIT;
            rc = pulseg_get_segment_info(coll, &segi, ci.segment_id);
            mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_segment_info failed");

            if (segi.pure_delay)
            {
                int found = 0;
                if (pure_seg_id < 0)
                {
                    pure_seg_id = ci.segment_id;
                    pure_seg_duration_us = segi.duration_us;
                }
                else
                {
                    mu_assert_int_eq(pure_seg_id, ci.segment_id);
                    mu_assert_int_eq(pure_seg_duration_us, segi.duration_us);
                }

                /* Segment definition stays canonical (one block raster). */
                mu_assert_int_eq((int)(opts.block_raster_us + 0.5f), segi.duration_us);

                if (inst.duration_us != segi.duration_us)
                    saw_noncanonical_instance = 1;

                for (i = 0; i < pure_inst_unique; ++i)
                {
                    if (pure_inst_durations[i] == inst.duration_us)
                    {
                        found = 1;
                        break;
                    }
                }
                if (!found &&
                    pure_inst_unique <
                        (int)(sizeof(pure_inst_durations) / sizeof(pure_inst_durations[0])))
                {
                    pure_inst_durations[pure_inst_unique++] = inst.duration_us;
                }
            }
        }

        e = &ref.entries[pos];

        /* Validate segment_id and within_segment_idx against the expected
         * pattern built from segment_order + segment_num_blocks.
         * The pattern tiles across TRs; pattern_pos wraps at pattern_len. */
        {
            int sidx = ci.subseq_idx;
            int pp = pattern_pos[sidx] % pattern_len[sidx];
            int expected_seg = pattern_seg_id[sidx][pp];
            char seg_msg[256];
            char wseg_msg[256];

            snprintf(
                seg_msg,
                sizeof(seg_msg),
                "[%s] segment_id mismatch at pos %d: cursor=%d, fixture=%d, expected=%d",
                tc->name,
                pos,
                ci.segment_id,
                e->segment_id,
                expected_seg);
            mu_assert(ci.segment_id == expected_seg, seg_msg);
            mu_assert(e->segment_id == expected_seg, seg_msg);

            snprintf(
                wseg_msg,
                sizeof(wseg_msg),
                "[%s] within_segment_idx out of range at pos %d: fixture=%d, segment=%d, nblk=%d",
                tc->name,
                pos,
                e->within_segment_idx,
                expected_seg,
                meta.segment_num_blocks[expected_seg]);
            mu_assert(
                e->within_segment_idx >= 0 &&
                    e->within_segment_idx < meta.segment_num_blocks[expected_seg],
                wseg_msg);

            pattern_pos[sidx]++;
        }

        /* RF amplitude (relative tolerance or absolute for zero) */
        tol = (float)fabs(e->rf_amp_hz) * 1e-4f;
        if (tol < 1e-6f)
            tol = 1e-6f;
        if ((float)fabs(inst.rf_amp_hz - e->rf_amp_hz) > tol)
            fprintf(
                stderr,
                "[scantable][%s] rf_amp @pos%d: ref=%.6g  lib=%.6g\n",
                tc->name,
                pos,
                e->rf_amp_hz,
                inst.rf_amp_hz);
        mu_assert((float)fabs(inst.rf_amp_hz - e->rf_amp_hz) <= tol, "rf_amp_hz mismatch");

        /* RF phase */
        tol = (float)fabs(e->rf_phase_rad) * 1e-4f;
        if (tol < 1e-8f)
            tol = 1e-8f;
        if ((float)fabs(inst.rf_phase_rad - e->rf_phase_rad) > tol)
            fprintf(
                stderr,
                "[scantable][%s] rf_phase @pos%d: ref=%.9g  lib=%.9g\n",
                tc->name,
                pos,
                e->rf_phase_rad,
                inst.rf_phase_rad);
        mu_assert((float)fabs(inst.rf_phase_rad - e->rf_phase_rad) <= tol, "rf_phase_rad mismatch");

        /* RF freq */
        tol = (float)fabs(e->rf_freq_hz) * 1e-4f;
        if (tol < 1e-6f)
            tol = 1e-6f;
        mu_assert((float)fabs(inst.rf_freq_hz - e->rf_freq_hz) <= tol, "rf_freq_hz mismatch");

        /* GX amplitude */
        tol = (float)fabs(e->gx_amp_hz_per_m) * 1e-4f;
        if (tol < 1e-6f)
            tol = 1e-6f;
        if ((float)fabs(inst.gx_amp_hz_per_m - e->gx_amp_hz_per_m) > tol)
            fprintf(
                stderr,
                "[scantable][%s] gx_amp @pos%d: ref=%.6g  lib=%.6g\n",
                tc->name,
                pos,
                e->gx_amp_hz_per_m,
                inst.gx_amp_hz_per_m);
        mu_assert((float)fabs(inst.gx_amp_hz_per_m - e->gx_amp_hz_per_m) <= tol, "gx_amp mismatch");

        /* GY amplitude */
        tol = (float)fabs(e->gy_amp_hz_per_m) * 1e-4f;
        if (tol < 1e-6f)
            tol = 1e-6f;
        if ((float)fabs(inst.gy_amp_hz_per_m - e->gy_amp_hz_per_m) > tol)
            fprintf(
                stderr,
                "[scantable][%s] gy_amp @pos%d: ref=%.6g  lib=%.6g\n",
                tc->name,
                pos,
                e->gy_amp_hz_per_m,
                inst.gy_amp_hz_per_m);
        mu_assert((float)fabs(inst.gy_amp_hz_per_m - e->gy_amp_hz_per_m) <= tol, "gy_amp mismatch");

        /* GZ amplitude */
        tol = (float)fabs(e->gz_amp_hz_per_m) * 1e-4f;
        if (tol < 1e-6f)
            tol = 1e-6f;
        if ((float)fabs(inst.gz_amp_hz_per_m - e->gz_amp_hz_per_m) > tol)
            fprintf(
                stderr,
                "[scantable][%s] gz_amp @pos%d: ref=%.6g  lib=%.6g\n",
                tc->name,
                pos,
                e->gz_amp_hz_per_m,
                inst.gz_amp_hz_per_m);
        mu_assert((float)fabs(inst.gz_amp_hz_per_m - e->gz_amp_hz_per_m) <= tol, "gz_amp mismatch");

        /* ADC flag */
        mu_assert_int_eq(e->adc_flag, inst.adc_flag);

        /* ADC phase */
        tol = (float)fabs(e->adc_phase_rad) * 1e-4f;
        if (tol < 1e-8f)
            tol = 1e-8f;
        mu_assert(
            (float)fabs(inst.adc_phase_rad - e->adc_phase_rad) <= tol,
            "adc_phase_rad mismatch");

        /* ADC freq */
        tol = (float)fabs(e->adc_freq_hz) * 1e-4f;
        if (tol < 1e-6f)
            tol = 1e-6f;
        mu_assert((float)fabs(inst.adc_freq_hz - e->adc_freq_hz) <= tol, "adc_freq_hz mismatch");

        /* Digital output flag */
        mu_assert_int_eq(e->digitalout_flag, inst.digitalout_flag);

        /* Block duration — must match library for all blocks;
         * for pure-delay segments the library uses the per-instance
         * block table entry, not the canonical segment definition. */
        if (e->block_dur_us > 0 && inst.duration_us != e->block_dur_us)
            fprintf(
                stderr,
                "[scantable][%s] block_dur @pos%d: ref=%d  lib=%d\n",
                tc->name,
                pos,
                e->block_dur_us,
                inst.duration_us);
        if (e->block_dur_us > 0)
            mu_assert_int_eq(e->block_dur_us, inst.duration_us);

        /* Rotation matrix */
        for (i = 0; i < 9; ++i)
        {
            mu_assert((float)fabs(inst.rotmat[i] - e->rotmat[i]) < 1e-5f, "rotmat mismatch");
        }

        /* ---- Trigger flag (per-block truth vs segment-level cursor) */
        if (e->trigger_flag)
        {
            mu_assert(ci.has_trigger, "cursor has_trigger should be set when truth trigger_flag=1");
            if (ci.segment_id >= 0 && ci.segment_id < MAX_SEGMENTS)
                seg_trigger_seen[ci.segment_id] = 1;
        }

        /* Count ADC readouts */
        if (e->adc_flag)
            total_readout_count++;

        ++pos;
    }

    mu_assert_int_eq(ref.num_entries, pos);

    /* ---- Total readouts cross-check (example_check.c step 6) ----- */
    mu_assert_int_eq(total_readout_count, cinfo.total_readouts);

    /* ---- Segment trigger consistency ----------------------------- */
    /*
     * For each segment where we saw truth trigger_flag=1 in at least
     * one block, the library's segment_info.has_trigger must be set.
     */
    {
        int seg;
        for (seg = 0; seg < cinfo.num_segments && seg < MAX_SEGMENTS; ++seg)
        {
            pulseg_segment_info si = PULSEG_SEGMENT_INFO_INIT;
            rc = pulseg_get_segment_info(coll, &si, seg);
            if (PULSEG_SUCCEEDED(rc) && seg_trigger_seen[seg])
            {
                mu_assert(
                    si.has_trigger,
                    "segment_info.has_trigger should be set "
                    "when scan table has trigger_flag");
            }
        }
    }

    if (is_mprage)
    {
        fprintf(
            stderr,
            "[scantable][%s] pure_seg_id=%d  pure_inst_unique=%d  saw_noncanonical=%d\n",
            tc->name,
            pure_seg_id,
            pure_inst_unique,
            saw_noncanonical_instance);
        mu_assert(pure_seg_id >= 0, "expected a pure-delay segment in MPRAGE scan loop");
        mu_assert(
            pure_inst_unique >= 2,
            "expected multiple pure-delay instance durations (e.g. TI and TR delays)");
        mu_assert(
            saw_noncanonical_instance,
            "expected scan-loop pure-delay instance duration to differ from canonical segment-def "
            "duration");
    }

    pulseg_collection_free(coll);
}

static void run_collection_case(const seq_case *tc)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    pulseg_collection_info cinfo = PULSEG_COLLECTION_INFO_INIT;
    pulseg_subseq_info gre_info = PULSEG_SUBSEQ_INFO_INIT;
    pulseg_subseq_info epi_info = PULSEG_SUBSEQ_INFO_INIT;
    seg_meta meta = SEG_META_INIT;
    exec_stream_file ref = SCAN_TABLE_FILE_INIT;
    pulseg_cursor_info ci = PULSEG_CURSOR_INFO_INIT;
    char scan_path[512];
    char meta_path[512];
    int rc, ok, i;
    int prev_subseq = -1;
    int saw_transition = 0;
    int num_blocks = 0;
    float expected_total_duration_us;

    gre_opts_init(&opts);
    rc = load_seq_with_averages(&coll, tc->seq_file, &opts, tc->num_averages);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed for collection case");

    rc = pulseg_get_collection_info(coll, &cinfo);
    mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_collection_info failed for collection case");
    mu_assert_int_eq(2, cinfo.num_subsequences);

    rc = pulseg_get_subseq_info(coll, &gre_info, 0);
    mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_subseq_info failed for GRE subsequence");
    rc = pulseg_get_subseq_info(coll, &epi_info, 1);
    mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_subseq_info failed for EPI subsequence");

    build_case_path(meta_path, sizeof(meta_path), tc, "_meta.txt");
    ok = parse_meta(meta_path, &meta);
    mu_assert(ok, "failed to parse collection _meta.txt");

    mu_assert_int_eq(meta.num_segments, cinfo.num_segments);
    mu_assert_int_eq(meta.num_unique_adcs, gre_info.num_unique_adcs + epi_info.num_unique_adcs);
    mu_assert_int_eq(0, gre_info.segment_offset);
    mu_assert(
        epi_info.segment_offset > gre_info.segment_offset,
        "EPI subsequence should have a positive segment offset");

    pulseg_cursor_reset(coll);
    while (pulseg_cursor_next(coll) == PULSEG_CURSOR_BLOCK)
    {
        rc = pulseg_cursor_get_info(coll, &ci);
        mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_cursor_get_info failed for collection case");
        mu_assert(ci.subseq_idx >= 0 && ci.subseq_idx < 2, "cursor subsequence index out of range");
        if (prev_subseq == 0 && ci.subseq_idx == 1)
            saw_transition = 1;
        mu_assert(
            prev_subseq <= ci.subseq_idx,
            "cursor subsequence index should be monotonic across collection");
        prev_subseq = ci.subseq_idx;
        ++num_blocks;
    }
    mu_assert(num_blocks > 0, "collection cursor did not visit any blocks");
    mu_assert(saw_transition, "collection cursor never transitioned from GRE to EPI");

    /* Duration: sum of block_dur_us from scan table fixture vs
     * cinfo.total_duration_us (now computed via scan-table walk). */
    build_case_path(scan_path, sizeof(scan_path), tc, "_scan_table.truth");
    ok = parse_exec_stream(scan_path, &ref);
    mu_assert(ok, "failed to parse scan_table.truth for duration check");
    expected_total_duration_us = 0.0f;
    for (i = 0; i < ref.num_entries; ++i)
        expected_total_duration_us += (float)ref.entries[i].block_dur_us;
    mu_assert_float_near(
        "collection duration",
        expected_total_duration_us,
        cinfo.total_duration_us,
        1.0f);

    pulseg_collection_free(coll);
}

MU_TEST(test_collection_gre_epi_1avg)
{
    run_collection_case(&kGreEpiCollectionCases[0]);
}
MU_TEST(test_collection_gre_epi_3avg)
{
    run_collection_case(&kGreEpiCollectionCases[1]);
}

MU_TEST_SUITE(suite_sequences_collection)
{
    MU_RUN_TEST(test_collection_gre_epi_1avg);
    MU_RUN_TEST(test_collection_gre_epi_3avg);
}

MU_TEST(test_exec_stream_gre_2d_1sl_1avg)
{
    run_exec_stream_case(&kGreCases[0]);
}
MU_TEST(test_exec_stream_gre_2d_1sl_3avg)
{
    run_exec_stream_case(&kGreCases[1]);
}
MU_TEST(test_exec_stream_gre_2d_3sl_1avg)
{
    run_exec_stream_case(&kGreCases[2]);
}
MU_TEST(test_exec_stream_gre_2d_3sl_3avg)
{
    run_exec_stream_case(&kGreCases[3]);
}
MU_TEST(test_exec_stream_gre_epi_collection_1avg)
{
    run_exec_stream_case(&kGreEpiCollectionCases[0]);
}
MU_TEST(test_exec_stream_gre_epi_collection_3avg)
{
    run_exec_stream_case(&kGreEpiCollectionCases[1]);
}

MU_TEST(test_exec_stream_mprage_2d_1sl_1avg)
{
    run_exec_stream_case(&kMprageCases[0]);
}
MU_TEST(test_exec_stream_mprage_2d_1sl_3avg)
{
    run_exec_stream_case(&kMprageCases[1]);
}
MU_TEST(test_exec_stream_mprage_2d_3sl_1avg)
{
    run_exec_stream_case(&kMprageCases[2]);
}
MU_TEST(test_exec_stream_mprage_2d_3sl_3avg)
{
    run_exec_stream_case(&kMprageCases[3]);
}
MU_TEST(test_exec_stream_mprage_nc_1sl_1avg)
{
    run_exec_stream_case(&kMprageNoncartCases[0]);
}
MU_TEST(test_exec_stream_mprage_nc_1sl_3avg_userotext0)
{
    run_exec_stream_case(&kMprageNoncartCases[1]);
}
MU_TEST(test_exec_stream_mprage_nc_3sl_1avg_userotext0)
{
    run_exec_stream_case(&kMprageNoncartCases[2]);
}
MU_TEST(test_exec_stream_mprage_nc_3sl_3avg_userotext0)
{
    run_exec_stream_case(&kMprageNoncartCases[3]);
}
MU_TEST(test_exec_stream_mprage_nc_1sl_1avg_userotext1)
{
    run_exec_stream_case(&kMprageNoncartCases[4]);
}
MU_TEST(test_exec_stream_mprage_nc_1sl_3avg_userotext1)
{
    run_exec_stream_case(&kMprageNoncartCases[5]);
}
MU_TEST(test_exec_stream_mprage_nc_3sl_1avg_userotext1)
{
    run_exec_stream_case(&kMprageNoncartCases[6]);
}
MU_TEST(test_exec_stream_mprage_nc_3sl_3avg_userotext1)
{
    run_exec_stream_case(&kMprageNoncartCases[7]);
}

MU_TEST(test_exec_stream_bssfp_2d_1sl_1avg)
{
    run_exec_stream_case(&kBssfpCases[0]);
}
MU_TEST(test_exec_stream_bssfp_2d_1sl_3avg)
{
    run_exec_stream_case(&kBssfpCases[1]);
}
MU_TEST(test_exec_stream_bssfp_2d_3sl_1avg)
{
    run_exec_stream_case(&kBssfpCases[2]);
}
MU_TEST(test_exec_stream_bssfp_2d_3sl_3avg)
{
    run_exec_stream_case(&kBssfpCases[3]);
}

MU_TEST(test_exec_stream_fse_2d_1sl_1avg)
{
    run_exec_stream_case(&kFseCases[0]);
}
MU_TEST(test_exec_stream_fse_2d_1sl_3avg)
{
    run_exec_stream_case(&kFseCases[1]);
}
MU_TEST(test_exec_stream_fse_2d_3sl_1avg)
{
    run_exec_stream_case(&kFseCases[2]);
}
MU_TEST(test_exec_stream_fse_2d_3sl_3avg)
{
    run_exec_stream_case(&kFseCases[3]);
}

MU_TEST(test_exec_stream_qalas_nc_3d_1sl_1avg_userotext1)
{
    run_exec_stream_case(&kQalasNoncartCases[0]);
}
MU_TEST(test_exec_stream_qalas_nc_3d_1sl_3avg_userotext1)
{
    run_exec_stream_case(&kQalasNoncartCases[1]);
}
MU_TEST(test_exec_stream_qalas_nc_3d_3sl_1avg_userotext1)
{
    run_exec_stream_case(&kQalasNoncartCases[2]);
}
MU_TEST(test_exec_stream_qalas_nc_3d_3sl_3avg_userotext1)
{
    run_exec_stream_case(&kQalasNoncartCases[3]);
}

MU_TEST(test_exec_stream_mprage_nav_2d_1sl_1avg)
{
    run_exec_stream_case(&kMprageNavCases[0]);
}
MU_TEST(test_exec_stream_mprage_nav_2d_1sl_3avg)
{
    run_exec_stream_case(&kMprageNavCases[1]);
}
MU_TEST(test_exec_stream_mprage_nav_2d_3sl_1avg)
{
    run_exec_stream_case(&kMprageNavCases[2]);
}
MU_TEST(test_exec_stream_mprage_nav_2d_3sl_3avg)
{
    run_exec_stream_case(&kMprageNavCases[3]);
}

/* ================================================================== */
/*  Suite — interior pure-delay static/dynamic classification         */
/*                                                                     */
/*  Fixtures: 99_interior_delay_static.seq / _dynamic.seq, each 3 TRs  */
/*  of [RF, delay, ADC] + [TR-tail delay].  TR1 == TR2 always (so TR-  */
/*  period detection finds tr_size=4 blocks from the first repeat     */
/*  alone); TR3's interior delay (position 1 in the 3-block segment)  */
/*  matches TR1/TR2 in the static fixture and differs in the dynamic   */
/*  one.  Confirms: (1) segs_delay_flex_equal still collapses all 3   */
/*  TR instances into ONE unique [RF,delay,ADC] segment regardless,   */
/*  and (2) is_variable_delay is 0 for the never-varying delay and 1  */
/*  only when the duration actually differs across instances.        */
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
    rc = load_seq(&coll, "gre_2d_1sl_1avg.seq", &opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

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
        if (desc->variable_grad_flags &&
            desc->variable_grad_flags[tr_pos * 3 + 1] &&
            bte->gy_id >= 0 &&
            bte->gy_id < desc->grad_table_size &&
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
    mu_assert_int_eq(
        (int)before.rows[before_adc].params[0],
        (int)after.rows[after_adc].params[0]);
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

static void assert_seqdesc_echo_pattern(
    const char *seq_file,
    int expect_all_adc_positions)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    pulseg_sequence_description description = {0, 0.0f, 0, NULL};
    int rc;
    int i;
    int num_adcs = 0;
    int num_echoes = 0;

    gre_opts_init(&opts);
    rc = load_seq(&coll, seq_file, &opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");
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
    assert_seqdesc_echo_pattern("mprage_2d_1sl_1avg.seq", 0);
}

MU_TEST(test_seqdesc_cartesian_fse_has_one_echo_position)
{
    assert_seqdesc_echo_pattern("fse_2d_1sl_1avg.seq", 0);
}

MU_TEST(test_seqdesc_radial_mprage_marks_every_readout_as_echo)
{
    assert_seqdesc_echo_pattern(
        "mprage_noncart_3d_1sl_1avg_userotext1.seq",
        1);
}

MU_TEST_SUITE(suite_sequences_seqdesc)
{
    MU_RUN_TEST(test_seqdesc_echo_uses_actual_instances_not_zero_var);
    MU_RUN_TEST(test_seqdesc_cartesian_mprage_has_one_echo_position);
    MU_RUN_TEST(test_seqdesc_cartesian_fse_has_one_echo_position);
    MU_RUN_TEST(test_seqdesc_radial_mprage_marks_every_readout_as_echo);
}

MU_TEST_SUITE(suite_sequences_scanloop)
{
    MU_RUN_TEST(test_exec_stream_gre_2d_1sl_1avg);
    MU_RUN_TEST(test_exec_stream_gre_2d_1sl_3avg);
    MU_RUN_TEST(test_exec_stream_gre_2d_3sl_1avg);
    MU_RUN_TEST(test_exec_stream_gre_2d_3sl_3avg);
    MU_RUN_TEST(test_exec_stream_gre_epi_collection_1avg);
    MU_RUN_TEST(test_exec_stream_gre_epi_collection_3avg);
    MU_RUN_TEST(test_exec_stream_mprage_2d_1sl_1avg);
    MU_RUN_TEST(test_exec_stream_mprage_2d_1sl_3avg);
    MU_RUN_TEST(test_exec_stream_mprage_2d_3sl_1avg);
    MU_RUN_TEST(test_exec_stream_mprage_2d_3sl_3avg);
    MU_RUN_TEST(test_exec_stream_mprage_nc_1sl_1avg);
    MU_RUN_TEST(test_exec_stream_mprage_nc_1sl_3avg_userotext0);
    MU_RUN_TEST(test_exec_stream_mprage_nc_3sl_1avg_userotext0);
    MU_RUN_TEST(test_exec_stream_mprage_nc_3sl_3avg_userotext0);
    MU_RUN_TEST(test_exec_stream_mprage_nc_1sl_1avg_userotext1);
    MU_RUN_TEST(test_exec_stream_mprage_nc_1sl_3avg_userotext1);
    MU_RUN_TEST(test_exec_stream_mprage_nc_3sl_1avg_userotext1);
    MU_RUN_TEST(test_exec_stream_mprage_nc_3sl_3avg_userotext1);
    MU_RUN_TEST(test_exec_stream_bssfp_2d_1sl_1avg);
    MU_RUN_TEST(test_exec_stream_bssfp_2d_1sl_3avg);
    MU_RUN_TEST(test_exec_stream_bssfp_2d_3sl_1avg);
    MU_RUN_TEST(test_exec_stream_bssfp_2d_3sl_3avg);
    MU_RUN_TEST(test_exec_stream_fse_2d_1sl_1avg);
    MU_RUN_TEST(test_exec_stream_fse_2d_1sl_3avg);
    MU_RUN_TEST(test_exec_stream_fse_2d_3sl_1avg);
    MU_RUN_TEST(test_exec_stream_fse_2d_3sl_3avg);
    MU_RUN_TEST(test_exec_stream_qalas_nc_3d_1sl_1avg_userotext1);
    MU_RUN_TEST(test_exec_stream_qalas_nc_3d_1sl_3avg_userotext1);
    MU_RUN_TEST(test_exec_stream_qalas_nc_3d_3sl_1avg_userotext1);
    MU_RUN_TEST(test_exec_stream_qalas_nc_3d_3sl_3avg_userotext1);
    MU_RUN_TEST(test_exec_stream_mprage_nav_2d_1sl_1avg);
    MU_RUN_TEST(test_exec_stream_mprage_nav_2d_1sl_3avg);
    MU_RUN_TEST(test_exec_stream_mprage_nav_2d_3sl_1avg);
    MU_RUN_TEST(test_exec_stream_mprage_nav_2d_3sl_3avg);
}

int test_sequences_main(void)
{
    minunit_run = 0;
    minunit_fail = 0;
    minunit_assert = 0;
    minunit_status = 0;
    minunit_real_timer = 0;
    minunit_proc_timer = 0;

    MU_RUN_SUITE(suite_sequences_check);
    MU_RUN_SUITE(suite_sequences_uieval);
    MU_RUN_SUITE(suite_sequences_geninstructions);
    MU_RUN_SUITE(suite_sequences_collection);
    MU_RUN_SUITE(suite_sequences_scanloop);
    MU_RUN_SUITE(suite_sequences_delay_classification);
    MU_RUN_SUITE(suite_sequences_seqdesc);
    MU_REPORT();
    return MU_EXIT_CODE;
}
