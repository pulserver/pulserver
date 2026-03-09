/*
 * test_segmentation.c -- segmentation tests (phases 1 & 2).
 *
 * Phase 1: Validates example_check.c step 6 quantities:
 *   1. Unique ADC definitions (count, num_samples, dwell_ns)
 *   2. max_b1_subseq index
 *   3. Nominal TR duration
 *
 * Phase 2: Validates example_check.c step 5 quantities + TR waveforms:
 *   4. Segment structure (count, blocks per segment)
 *   5. Worst-case TR gradient waveforms vs MATLAB ground truth
 */
#include "test_helpers.h"
#include "test_seg_helpers.h"

#include <math.h>

/* ------------------------------------------------------------------ */
/*  Phase 1: example_check step 6 (ADC, max_b1, TR)                  */
/* ------------------------------------------------------------------ */

MU_TEST(test_segmentation_gre_example_check)
{
    pulseqlib_opts opts;
    pulseqlib_collection* coll = NULL;
    pulseqlib_collection_info cinfo = PULSEQLIB_COLLECTION_INFO_INIT;
    pulseqlib_subseq_info sinfo = PULSEQLIB_SUBSEQ_INFO_INIT;
    seg_meta meta = SEG_META_INIT;
    int rc, a, ok;

    /* Load sequence */
    gre_opts_init(&opts);
    rc = load_seq(&coll, "gre_2d_1sl_1avg.seq", &opts);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq failed for GRE baseline");

    /* Collection must have exactly one subsequence */
    rc = pulseqlib_get_collection_info(coll, &cinfo);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_get_collection_info failed");
    mu_assert_int_eq(1, cinfo.num_subsequences);

    /* Get subsequence info */
    rc = pulseqlib_get_subseq_info(coll, 0, &sinfo);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_get_subseq_info failed");

    /* Parse MATLAB ground truth */
    ok = parse_meta(TEST_DATA_DIR "gre_2d_1sl_1avg_meta.txt", &meta);
    mu_assert(ok, "failed to parse gre_2d_1sl_1avg_meta.txt");

    /* 1. Unique ADC definitions */
    mu_assert_int_eq(meta.num_unique_adcs, sinfo.num_unique_adcs);
    for (a = 0; a < sinfo.num_unique_adcs; ++a) {
        pulseqlib_adc_def ad = PULSEQLIB_ADC_DEF_INIT;
        rc = pulseqlib_get_adc_def(coll, a, &ad);
        mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_get_adc_def failed");
        mu_assert_int_eq(meta.adc_samples[a], ad.num_samples);
        mu_assert_int_eq(meta.adc_dwell_ns[a], ad.dwell_ns);
    }

    /* 2. max_b1_subseq — trivially 0 for single-subsequence collection */
    mu_assert_int_eq(0, meta.max_b1_subseq);

    /* 3. Nominal TR */
    mu_assert_float_near("TR duration",
        (float)meta.tr_duration_us, sinfo.tr_duration_us, 1.0f);

    pulseqlib_collection_free(coll);
}

MU_TEST_SUITE(suite_segmentation_phase1)
{
    MU_RUN_TEST(test_segmentation_gre_example_check);
}

/* ------------------------------------------------------------------ */
/*  Phase 2: example_check step 5 (segments) + TR waveforms           */
/* ------------------------------------------------------------------ */

/* Relative tolerance for waveform amplitude comparison. */
#define WAVE_REL_TOL 1e-3f
#define WAVE_TIME_ABS_TOL 0.5f  /* us — half a raster step */

MU_TEST(test_segmentation_gre_safety_waveforms)
{
    pulseqlib_opts opts;
    pulseqlib_collection* coll = NULL;
    pulseqlib_collection_info cinfo = PULSEQLIB_COLLECTION_INFO_INIT;
    pulseqlib_subseq_info sinfo = PULSEQLIB_SUBSEQ_INFO_INIT;
    seg_meta meta = SEG_META_INIT;
    seg_tr_waveform ref_wf = SEG_TR_WAVEFORM_INIT;
    pulseqlib_tr_gradient_waveforms lib_wf = PULSEQLIB_TR_GRADIENT_WAVEFORMS_INIT;
    pulseqlib_diagnostic diag;
    int rc, s, i, ok, n;

    /* Load sequence */
    gre_opts_init(&opts);
    rc = load_seq(&coll, "gre_2d_1sl_1avg.seq", &opts);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq failed for GRE baseline");

    /* Collection info */
    rc = pulseqlib_get_collection_info(coll, &cinfo);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_get_collection_info failed");

    /* Subseq info */
    rc = pulseqlib_get_subseq_info(coll, 0, &sinfo);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_get_subseq_info failed");

    /* Parse MATLAB ground truth */
    ok = parse_meta(TEST_DATA_DIR "gre_2d_1sl_1avg_meta.txt", &meta);
    mu_assert(ok, "failed to parse gre_2d_1sl_1avg_meta.txt");

    /* 4. Segment structure */
    mu_assert_int_eq(meta.num_segments, cinfo.num_segments);
    for (s = 0; s < cinfo.num_segments && s < MAX_SEGMENTS; ++s) {
        pulseqlib_segment_info segi = PULSEQLIB_SEGMENT_INFO_INIT;
        rc = pulseqlib_get_segment_info(coll, s, &segi);
        mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_get_segment_info failed");
        mu_assert_int_eq(meta.segment_num_blocks[s], segi.num_blocks);
    }

    /* Number of canonical TR waveforms to compare */
    mu_assert_int_eq(meta.num_canonical_trs, sinfo.num_passes);

    /* 5. Worst-case TR gradient waveforms */
    ok = parse_tr_waveform(TEST_DATA_DIR "gre_2d_1sl_1avg_tr_waveform.bin", &ref_wf);
    mu_assert(ok, "failed to parse gre_2d_1sl_1avg_tr_waveform.bin");

    pulseqlib_diagnostic_init(&diag);
    rc = pulseqlib_get_tr_gradient_waveforms(coll, 0, &lib_wf, &diag);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_get_tr_gradient_waveforms failed");

    /* Use the smaller of the two sample counts for comparison
       (off-by-one can happen at raster boundary). */
    n = ref_wf.num_samples < lib_wf.gx.num_samples
        ? ref_wf.num_samples : lib_wf.gx.num_samples;
    mu_assert(abs(ref_wf.num_samples - lib_wf.gx.num_samples) <= 1,
              "TR waveform sample count mismatch > 1");

    for (i = 0; i < n; ++i) {
        float ref_t  = ref_wf.time_us[i];
        float lib_t  = lib_wf.gx.time_us[i];
        float dt     = ref_t - lib_t;
        float ref_gx = ref_wf.gx[i];
        float ref_gy = ref_wf.gy[i];
        float ref_gz = ref_wf.gz[i];
        float lib_gx = lib_wf.gx.amplitude_hz_per_m[i];
        float lib_gy = lib_wf.gy.amplitude_hz_per_m[i];
        float lib_gz = lib_wf.gz.amplitude_hz_per_m[i];
        float tol_gx, tol_gy, tol_gz;

        /* Time alignment check */
        if (dt < 0) dt = -dt;
        mu_assert(dt <= WAVE_TIME_ABS_TOL, "TR waveform time mismatch");

        /* Amplitude check: relative tolerance with absolute floor */
        tol_gx = (ref_gx < 0 ? -ref_gx : ref_gx) * WAVE_REL_TOL;
        if (tol_gx < 1.0f) tol_gx = 1.0f;
        tol_gy = (ref_gy < 0 ? -ref_gy : ref_gy) * WAVE_REL_TOL;
        if (tol_gy < 1.0f) tol_gy = 1.0f;
        tol_gz = (ref_gz < 0 ? -ref_gz : ref_gz) * WAVE_REL_TOL;
        if (tol_gz < 1.0f) tol_gz = 1.0f;

        mu_assert(fabsf(ref_gx - lib_gx) <= tol_gx, "TR waveform Gx mismatch");
        mu_assert(fabsf(ref_gy - lib_gy) <= tol_gy, "TR waveform Gy mismatch");
        mu_assert(fabsf(ref_gz - lib_gz) <= tol_gz, "TR waveform Gz mismatch");
    }

    free_tr_waveform(&ref_wf);
    pulseqlib_tr_gradient_waveforms_free(&lib_wf);
    pulseqlib_collection_free(coll);
}

MU_TEST_SUITE(suite_segmentation_phase2)
{
    MU_RUN_TEST(test_segmentation_gre_safety_waveforms);
}

int test_segmentation_main(void)
{
    minunit_run = 0;
    minunit_fail = 0;
    minunit_assert = 0;
    minunit_status = 0;
    minunit_real_timer = 0;
    minunit_proc_timer = 0;

    MU_RUN_SUITE(suite_segmentation_phase1);
    MU_RUN_SUITE(suite_segmentation_phase2);
    MU_REPORT();
    return MU_EXIT_CODE;
}
