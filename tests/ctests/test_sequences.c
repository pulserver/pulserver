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

typedef struct {
    const char* name;
    const char* seq_file;
    const char* base;
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

static const seq_case kMprageNoncartCases[] = {
    {"mprage_noncart_3d_1sl_1avg_userotext0", "mprage_noncart_3d_1sl_1avg_userotext0.seq", "mprage_noncart_3d_1sl_1avg_userotext0", 1},
};

static const seq_case kMprageNavCases[] = {
    {"mprage_nav_2d_1sl_1avg", "mprage_nav_2d_1sl_1avg.seq", "mprage_nav_2d_1sl_1avg", 1},
    {"mprage_nav_2d_1sl_3avg", "mprage_nav_2d_1sl_3avg.seq", "mprage_nav_2d_1sl_3avg", 3},
    {"mprage_nav_2d_3sl_1avg", "mprage_nav_2d_3sl_1avg.seq", "mprage_nav_2d_3sl_1avg", 1},
    {"mprage_nav_2d_3sl_3avg", "mprage_nav_2d_3sl_3avg.seq", "mprage_nav_2d_3sl_3avg", 3},
};

static void build_case_path(char* dst, size_t dst_sz, const seq_case* tc, const char* suffix)
{
    (void)snprintf(dst, dst_sz, TEST_DATA_DIR "%s%s", tc->base, suffix);
}

/* ------------------------------------------------------------------ */
/*  Phase 1: example_check step 6 (ADC, max_b1, TR)                  */
/* ------------------------------------------------------------------ */

static void run_check_case(const seq_case* tc)
{
    pulseqlib_opts opts;
    pulseqlib_collection* coll = NULL;
    pulseqlib_collection_info cinfo = PULSEQLIB_COLLECTION_INFO_INIT;
    pulseqlib_subseq_info sinfo = PULSEQLIB_SUBSEQ_INFO_INIT;
    seg_meta meta = SEG_META_INIT;
    char meta_path[512];
    int rc, a, ok;
    int expected_max_adc_samples = 0;

    /* Load sequence */
    gre_opts_init(&opts);
    rc = load_seq_with_averages(&coll, tc->seq_file, &opts, tc->num_averages);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq failed for GRE test case");

    /* Collection must have exactly one subsequence */
    rc = pulseqlib_get_collection_info(coll, &cinfo);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_get_collection_info failed");
    mu_assert_int_eq(1, cinfo.num_subsequences);

    /* Get subsequence info */
    rc = pulseqlib_get_subseq_info(coll, 0, &sinfo);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_get_subseq_info failed");

    /* Parse MATLAB ground truth */
    build_case_path(meta_path, sizeof(meta_path), tc, "_meta.txt");
    ok = parse_meta(meta_path, &meta);
    mu_assert(ok, "failed to parse case _meta.txt");

    /* 1. Unique ADC definitions */
    mu_assert_int_eq(meta.num_unique_adcs, sinfo.num_unique_adcs);
    for (a = 0; a < sinfo.num_unique_adcs; ++a) {
        pulseqlib_adc_def ad = PULSEQLIB_ADC_DEF_INIT;
        rc = pulseqlib_get_adc_def(coll, a, &ad);
        mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_get_adc_def failed");
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
    mu_assert_float_near("TR duration",
        (float)meta.tr_duration_us, sinfo.tr_duration_us, 1.0f);

    /* 4. Num segments — verify segment count matches truth */
    fprintf(stderr, "[check][%s] num_segments: meta=%d  lib=%d\n",
            tc->name, meta.num_segments, cinfo.num_segments);
    mu_assert_int_eq(meta.num_segments, cinfo.num_segments);

    pulseqlib_collection_free(coll);
}

MU_TEST(test_check_gre_2d_1sl_1avg) { run_check_case(&kGreCases[0]); }
MU_TEST(test_check_gre_2d_1sl_3avg) { run_check_case(&kGreCases[1]); }
MU_TEST(test_check_gre_2d_3sl_1avg) { run_check_case(&kGreCases[2]); }
MU_TEST(test_check_gre_2d_3sl_3avg) { run_check_case(&kGreCases[3]); }

MU_TEST(test_check_mprage_2d_1sl_1avg) { run_check_case(&kMprageCases[0]); }
MU_TEST(test_check_mprage_2d_1sl_3avg) { run_check_case(&kMprageCases[1]); }
MU_TEST(test_check_mprage_2d_3sl_1avg) { run_check_case(&kMprageCases[2]); }
MU_TEST(test_check_mprage_2d_3sl_3avg) { run_check_case(&kMprageCases[3]); }
MU_TEST(test_check_mprage_nc_1sl_1avg) { run_check_case(&kMprageNoncartCases[0]); }

MU_TEST(test_check_mprage_nav_2d_1sl_1avg) { run_check_case(&kMprageNavCases[0]); }
MU_TEST(test_check_mprage_nav_2d_1sl_3avg) { run_check_case(&kMprageNavCases[1]); }
MU_TEST(test_check_mprage_nav_2d_3sl_1avg) { run_check_case(&kMprageNavCases[2]); }
MU_TEST(test_check_mprage_nav_2d_3sl_3avg) { run_check_case(&kMprageNavCases[3]); }

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
#define WAVE_TIME_ABS_TOL 0.5f  /* us — half a raster step */

static void run_sequences_uieval_case(const seq_case* tc)
{
    pulseqlib_opts opts;
    pulseqlib_collection* coll = NULL;
    pulseqlib_collection_info cinfo = PULSEQLIB_COLLECTION_INFO_INIT;
    pulseqlib_subseq_info sinfo = PULSEQLIB_SUBSEQ_INFO_INIT;
    seg_meta meta = SEG_META_INIT;
    seg_tr_waveform_set ref_wfs = SEG_TR_WAVEFORM_SET_INIT;
    pulseqlib_tr_gradient_waveforms lib_wf = PULSEQLIB_TR_GRADIENT_WAVEFORMS_INIT;
    pulseqlib_diagnostic diag;
    char meta_path[512];
    char tr_path[512];
    int rc, s, i, ok, n;

    /* Load sequence */
    gre_opts_init(&opts);
    rc = load_seq_with_averages(&coll, tc->seq_file, &opts, tc->num_averages);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq failed for GRE test case");

    /* Collection info */
    rc = pulseqlib_get_collection_info(coll, &cinfo);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_get_collection_info failed");

    /* Subseq info */
    rc = pulseqlib_get_subseq_info(coll, 0, &sinfo);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_get_subseq_info failed");

    /* Parse MATLAB ground truth */
    build_case_path(meta_path, sizeof(meta_path), tc, "_meta.txt");
    ok = parse_meta(meta_path, &meta);
    mu_assert(ok, "failed to parse case _meta.txt");


    /* 4. Segment structure */
    fprintf(stderr, "[uieval][%s] num_segments: meta=%d  lib=%d\n",
            tc->name, meta.num_segments, cinfo.num_segments);
    mu_assert_int_eq(meta.num_segments, cinfo.num_segments);
    for (s = 0; s < cinfo.num_segments && s < MAX_SEGMENTS; ++s) {
        pulseqlib_segment_info segi = PULSEQLIB_SEGMENT_INFO_INIT;
        rc = pulseqlib_get_segment_info(coll, s, &segi);
        mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_get_segment_info failed");
        fprintf(stderr, "[uieval][%s] segment %d num_blocks: meta=%d  lib=%d\n",
                tc->name, s, meta.segment_num_blocks[s], segi.num_blocks);
        mu_assert_int_eq(meta.segment_num_blocks[s], segi.num_blocks);
    }

    /* Number of canonical TR waveforms to compare */
    mu_assert(sinfo.num_passes > 0, "invalid num_passes from library");

    /* 5. Worst-case TR gradient waveforms */
    build_case_path(tr_path, sizeof(tr_path), tc, "_tr_waveform.bin");
    ok = parse_tr_waveform_set(tr_path, &ref_wfs);
    mu_assert(ok, "failed to parse case _tr_waveform.bin");
    mu_assert_int_eq(meta.num_canonical_trs, ref_wfs.num_trs);

    /* Compare first canonical TR against library output.
       (Public API currently returns only the first canonical TR.) */
    pulseqlib_diagnostic_init(&diag);
    rc = pulseqlib_get_tr_gradient_waveforms(coll, 0, &lib_wf, &diag);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_get_tr_gradient_waveforms failed");

    {
        const seg_tr_waveform* ref_wf = &ref_wfs.waveforms[0];

        /* Use the smaller of the two sample counts for comparison
           (off-by-one can happen at raster boundary). */
        n = ref_wf->num_samples < lib_wf.gx.num_samples
            ? ref_wf->num_samples : lib_wf.gx.num_samples;
        mu_assert(abs(ref_wf->num_samples - lib_wf.gx.num_samples) <= 1,
                  "TR waveform sample count mismatch > 1");

        for (i = 0; i < n; ++i) {
            float ref_t  = ref_wf->time_us[i];
            float lib_t  = lib_wf.gx.time_us[i];
            float dt     = ref_t - lib_t;
            float ref_gx = ref_wf->gx[i];
            float ref_gy = ref_wf->gy[i];
            float ref_gz = ref_wf->gz[i];
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

            if (fabsf(ref_gx - lib_gx) > tol_gx)
                fprintf(stderr, "[uieval][%s] TR Gx mismatch @%d: ref=%.4g  lib=%.4g\n",
                        tc->name, i, ref_gx, lib_gx);
            mu_assert(fabsf(ref_gx - lib_gx) <= tol_gx, "TR waveform Gx mismatch");
            mu_assert(fabsf(ref_gy - lib_gy) <= tol_gy, "TR waveform Gy mismatch");
            mu_assert(fabsf(ref_gz - lib_gz) <= tol_gz, "TR waveform Gz mismatch");
        }
    }

    free_tr_waveform_set(&ref_wfs);
    pulseqlib_tr_gradient_waveforms_free(&lib_wf);
    pulseqlib_collection_free(coll);
}

MU_TEST(test_sequences_uieval_gre_2d_1sl_1avg) { run_sequences_uieval_case(&kGreCases[0]); }
MU_TEST(test_sequences_uieval_gre_2d_1sl_3avg) { run_sequences_uieval_case(&kGreCases[1]); }
MU_TEST(test_sequences_uieval_gre_2d_3sl_1avg) { run_sequences_uieval_case(&kGreCases[2]); }
MU_TEST(test_sequences_uieval_gre_2d_3sl_3avg) { run_sequences_uieval_case(&kGreCases[3]); }

MU_TEST(test_sequences_uieval_mprage_2d_1sl_1avg) { run_sequences_uieval_case(&kMprageCases[0]); }
MU_TEST(test_sequences_uieval_mprage_2d_1sl_3avg) { run_sequences_uieval_case(&kMprageCases[1]); }
MU_TEST(test_sequences_uieval_mprage_2d_3sl_1avg) { run_sequences_uieval_case(&kMprageCases[2]); }
MU_TEST(test_sequences_uieval_mprage_2d_3sl_3avg) { run_sequences_uieval_case(&kMprageCases[3]); }
MU_TEST(test_sequences_uieval_mprage_nc_1sl_1avg) { run_sequences_uieval_case(&kMprageNoncartCases[0]); }

MU_TEST(test_sequences_uieval_mprage_nav_2d_1sl_1avg) { run_sequences_uieval_case(&kMprageNavCases[0]); }
MU_TEST(test_sequences_uieval_mprage_nav_2d_1sl_3avg) { run_sequences_uieval_case(&kMprageNavCases[1]); }
MU_TEST(test_sequences_uieval_mprage_nav_2d_3sl_1avg) { run_sequences_uieval_case(&kMprageNavCases[2]); }
MU_TEST(test_sequences_uieval_mprage_nav_2d_3sl_3avg) { run_sequences_uieval_case(&kMprageNavCases[3]); }

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
/*              has_digital_out, has_freq_mod)                         */
/*    - RF     (delay, amp, num_samples, waveform shape)               */
/*    - Grads  (delay, amp, num_samples, waveform shape per axis)      */
/*    - ADC    (delay)                                                 */
/*    - Digitalout (delay, duration)                                   */
/*    - Freq-mod  (num_samples)                                        */
/* ------------------------------------------------------------------ */

#define GENI_AMP_REL_TOL  1e-3f   /* relative tolerance for normalised amps */
#define GENI_DELAY_ABS_TOL 1.0f   /* us — half a raster step                */

/* Relative amplitude comparison with absolute floor of 1.0 */
#define GENI_AMP_NEAR(a, b) \
    (fabsf((a) - (b)) <= (((fabsf(a) > 1.0f ? fabsf(a) : 1.0f)) * GENI_AMP_REL_TOL))

static void run_sequences_geninstructions_case(const seq_case* tc)
{
    pulseqlib_opts opts;
    pulseqlib_collection* coll = NULL;
    pulseqlib_collection_info cinfo = PULSEQLIB_COLLECTION_INFO_INIT;
    static seg_def_file ref;   /* static: too large (~8 MB) for stack */
    char seg_path[512];
    int rc, ok;
    int s, b, ax;

    /* Load sequence */
    gre_opts_init(&opts);
    rc = load_seq_with_averages(&coll, tc->seq_file, &opts, tc->num_averages);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq failed for GRE test case");

    rc = pulseqlib_get_collection_info(coll, &cinfo);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_get_collection_info failed");

    /* Load MATLAB ground truth */
    build_case_path(seg_path, sizeof(seg_path), tc, "_segment_def.bin");
    ok = parse_seg_def(seg_path, &ref);
    mu_assert(ok, "failed to parse case _segment_def.bin");

    /* Number of segments must match */
    fprintf(stderr, "[geninstr][%s] num_segments: ref=%d  lib=%d\n",
            tc->name, ref.num_segments, cinfo.num_segments);
    mu_assert_int_eq(ref.num_segments, cinfo.num_segments);

    for (s = 0; s < ref.num_segments; ++s) {
        pulseqlib_segment_info segi = PULSEQLIB_SEGMENT_INFO_INIT;
        rc = pulseqlib_get_segment_info(coll, s, &segi);
        mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_get_segment_info failed");
        fprintf(stderr, "[geninstr][%s] segment %d num_blocks: ref=%d  lib=%d\n",
                tc->name, s, ref.num_blocks[s], segi.num_blocks);
        mu_assert_int_eq(ref.num_blocks[s], segi.num_blocks);

        for (b = 0; b < ref.num_blocks[s]; ++b) {
            const seg_block_def* ref_blk = &ref.blocks[s][b];
            pulseqlib_block_info bi = PULSEQLIB_BLOCK_INFO_INIT;
            int need_ns = 0;
            int need;
            rc = pulseqlib_get_block_info(coll, s, b, &bi);
            mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_get_block_info failed");
            need = pulseqlib_block_needs_freq_mod(coll, s, b, &need_ns);

            /* --- Flags -------------------------------------------- */
            mu_assert_int_eq(ref_blk->has_rf,          bi.has_rf);
            for (ax = 0; ax < 3; ++ax)
                mu_assert_int_eq(ref_blk->has_grad[ax], bi.has_grad[ax]);
            mu_assert_int_eq(ref_blk->has_adc,         bi.has_adc);
            mu_assert_int_eq(ref_blk->has_rotation,    bi.has_rotation);
            mu_assert_int_eq(ref_blk->has_digital_out, bi.has_digitalout);
            mu_assert_int_eq(ref_blk->has_freq_mod, need);

            /* --- RF ----------------------------------------------- */
            if (ref_blk->has_rf) {
                int num_channels = 0, num_samples = 0;
                float** mag;
                int i;

                mu_assert(fabsf(ref_blk->rf_delay - (float)bi.rf_delay_us * 1e-6f)
                          <= GENI_DELAY_ABS_TOL * 1e-6f,
                          "RF delay mismatch");
                mu_assert_int_eq(ref_blk->rf_n, bi.rf_num_samples);

                mag = pulseqlib_get_rf_magnitude(coll, s, b, &num_channels, &num_samples);
                mu_assert(mag != NULL, "pulseqlib_get_rf_magnitude returned NULL");
                mu_assert_int_eq(ref_blk->rf_n, num_samples);

                /* Both library and MATLAB store normalised shapes
                   (peak ≈ 1.0).  Compare directly. */
                for (i = 0; i < num_samples; ++i) {
                    mu_assert(GENI_AMP_NEAR(ref_blk->rf_rho[i], mag[0][i]),
                              "RF magnitude shape mismatch");
                }

                /* Amplitude checks via new getters */
                {
                    float init_amp = pulseqlib_get_rf_initial_amplitude_hz(coll, s, b);
                    float max_amp  = pulseqlib_get_rf_max_amplitude_hz(coll, s, b);
                    mu_assert(GENI_AMP_NEAR(ref_blk->rf_amp, init_amp),
                              "RF initial amplitude mismatch");
                    mu_assert(GENI_AMP_NEAR(fabsf(ref_blk->rf_amp), max_amp),
                              "RF max amplitude mismatch");
                }

                { int ch; for (ch = 0; ch < num_channels; ++ch) free(mag[ch]); free(mag); }
            }

            /* --- Gradients ---------------------------------------- */
            for (ax = 0; ax < 3; ++ax) {
                if (ref_blk->has_grad[ax]) {
                    int num_shots = 0, num_samples = 0;
                    float** amps;
                    int i;

                    mu_assert(fabsf(ref_blk->grad_delay[ax]
                                    - (float)bi.grad_delay_us[ax] * 1e-6f)
                              <= GENI_DELAY_ABS_TOL * 1e-6f,
                              "grad delay mismatch");
                    mu_assert_int_eq(ref_blk->grad_n[ax], bi.grad_num_samples[ax]);

                    amps = pulseqlib_get_grad_amplitude(coll, s, b, ax,
                                                        &num_shots, &num_samples);
                    mu_assert(amps != NULL, "pulseqlib_get_grad_amplitude returned NULL");
                    mu_assert_int_eq(ref_blk->grad_n[ax], num_samples);

                    /* Both library and MATLAB store normalised shapes
                       (peak ≈ 1.0).  Compare directly. */
                    for (i = 0; i < num_samples; ++i) {
                        if (!GENI_AMP_NEAR(ref_blk->grad_wave[ax][i], amps[0][i]))
                            fprintf(stderr, "[geninstr][%s] seg%d blk%d ax%d wave@%d: ref=%.4g  lib=%.4g\n",
                                    tc->name, s, b, ax, i,
                                    ref_blk->grad_wave[ax][i], amps[0][i]);
                        mu_assert(GENI_AMP_NEAR(ref_blk->grad_wave[ax][i], amps[0][i]),
                                  "grad waveform shape mismatch");
                    }

                    /* Amplitude checks via new getters */
                    {
                        float init_amp = pulseqlib_get_grad_initial_amplitude_hz_per_m(
                                             coll, s, b, ax);
                        float max_amp  = pulseqlib_get_grad_max_amplitude_hz_per_m(
                                             coll, s, b, ax);
                        if (!GENI_AMP_NEAR(ref_blk->grad_amp[ax], init_amp))
                            fprintf(stderr, "[geninstr][%s] seg%d blk%d ax%d init_amp: ref=%.6g  lib=%.6g\n",
                                    tc->name, s, b, ax, ref_blk->grad_amp[ax], init_amp);
                        mu_assert(GENI_AMP_NEAR(ref_blk->grad_amp[ax], init_amp),
                                  "grad initial amplitude mismatch");
                        if (!GENI_AMP_NEAR(fabsf(ref_blk->grad_amp[ax]), max_amp))
                            fprintf(stderr, "[geninstr][%s] seg%d blk%d ax%d max_amp: ref=%.6g  lib=%.6g\n",
                                    tc->name, s, b, ax, fabsf(ref_blk->grad_amp[ax]), max_amp);
                        mu_assert(GENI_AMP_NEAR(fabsf(ref_blk->grad_amp[ax]), max_amp),
                                  "grad max amplitude mismatch");
                    }

                    { int sh; for (sh = 0; sh < num_shots; ++sh) free(amps[sh]); free(amps); }
                }
            }

            /* --- ADC ---------------------------------------------- */
            if (ref_blk->has_adc) {
                mu_assert(fabsf(ref_blk->adc_delay - (float)bi.adc_delay_us * 1e-6f)
                          <= GENI_DELAY_ABS_TOL * 1e-6f,
                          "ADC delay mismatch");
            }

            /* --- Digital output ------------------------------------ */
            if (ref_blk->has_digital_out) {
                mu_assert(fabsf(ref_blk->digital_out_delay
                                - (float)bi.digitalout_delay_us * 1e-6f)
                          <= GENI_DELAY_ABS_TOL * 1e-6f,
                          "digital-out delay mismatch");
                mu_assert(fabsf(ref_blk->digital_out_duration
                                - (float)bi.digitalout_duration_us * 1e-6f)
                          <= GENI_DELAY_ABS_TOL * 1e-6f,
                          "digital-out duration mismatch");
            }

            /* --- Freq-mod ----------------------------------------- */
            if (ref_blk->has_freq_mod) {
                int raster_us = 2; /* vendor raster, matches MATLAB sys.rfRasterTime/adcRasterTime */
                int lib_num_samples = bi.duration_us / raster_us;
                mu_assert_int_eq(ref_blk->freq_mod_num_samples, lib_num_samples);
            }

            /* --- Pure-delay segment-def duration canonicalization -- */
            if (!ref_blk->has_rf &&
                !ref_blk->has_grad[0] && !ref_blk->has_grad[1] && !ref_blk->has_grad[2] &&
                !ref_blk->has_adc) {
                int expected_delay_us = (int)(opts.block_raster_us + 0.5f);
                mu_assert_int_eq(expected_delay_us, bi.duration_us);
            }

            /* --- Freq-mod (overlap API) --------------------------- */
            {
                mu_assert_int_eq(ref_blk->has_freq_mod, need);
                if (ref_blk->has_freq_mod) {
                    mu_assert_int_eq(ref_blk->freq_mod_num_samples, need_ns);
                }
            }

            /* --- Anchors ------------------------------------------ */
            if (ref_blk->has_rf) {
                float lib_iso = pulseqlib_get_rf_isocenter_us(coll, s, b);
                mu_assert(fabsf(ref_blk->rf_isocenter_us - lib_iso) <= 1.0f,
                          "RF isocenter_us mismatch");
            }
            if (ref_blk->has_adc) {
                float lib_kz = pulseqlib_get_adc_kzero_us(coll, s, b);
                mu_assert(fabsf(ref_blk->adc_kzero_us - lib_kz) <= 1.0f,
                          "ADC kzero_us mismatch");
            }
        }

        /* --- Segment-level gaps ------------------------------- */
        {
            int ref_rf_adc = (int)roundf(ref.rf_adc_gap_us[s]);
            int ref_adc_adc = (int)roundf(ref.adc_adc_gap_us[s]);

            if (ref_rf_adc >= 0) {
                mu_assert_int_eq(ref_rf_adc, segi.rf_adc_gap_us);
            } else {
                mu_assert_int_eq(ref_rf_adc, segi.rf_adc_gap_us);
            }
            if (ref_adc_adc >= 0) {
                mu_assert_int_eq(ref_adc_adc, segi.adc_adc_gap_us);
            } else {
                mu_assert_int_eq(ref_adc_adc, segi.adc_adc_gap_us);
            }
        }
    }

    pulseqlib_collection_free(coll);
}

MU_TEST(test_sequences_geninstructions_gre_2d_1sl_1avg) { run_sequences_geninstructions_case(&kGreCases[0]); }
MU_TEST(test_sequences_geninstructions_gre_2d_1sl_3avg) { run_sequences_geninstructions_case(&kGreCases[1]); }
MU_TEST(test_sequences_geninstructions_gre_2d_3sl_1avg) { run_sequences_geninstructions_case(&kGreCases[2]); }
MU_TEST(test_sequences_geninstructions_gre_2d_3sl_3avg) { run_sequences_geninstructions_case(&kGreCases[3]); }

MU_TEST(test_sequences_geninstructions_mprage_2d_1sl_1avg) { run_sequences_geninstructions_case(&kMprageCases[0]); }
MU_TEST(test_sequences_geninstructions_mprage_2d_1sl_3avg) { run_sequences_geninstructions_case(&kMprageCases[1]); }
MU_TEST(test_sequences_geninstructions_mprage_2d_3sl_1avg) { run_sequences_geninstructions_case(&kMprageCases[2]); }
MU_TEST(test_sequences_geninstructions_mprage_2d_3sl_3avg) { run_sequences_geninstructions_case(&kMprageCases[3]); }
MU_TEST(test_sequences_geninstructions_mprage_nc_1sl_1avg) { run_sequences_geninstructions_case(&kMprageNoncartCases[0]); }

MU_TEST(test_sequences_geninstructions_mprage_nav_2d_1sl_1avg) { run_sequences_geninstructions_case(&kMprageNavCases[0]); }
MU_TEST(test_sequences_geninstructions_mprage_nav_2d_1sl_3avg) { run_sequences_geninstructions_case(&kMprageNavCases[1]); }
MU_TEST(test_sequences_geninstructions_mprage_nav_2d_3sl_1avg) { run_sequences_geninstructions_case(&kMprageNavCases[2]); }
MU_TEST(test_sequences_geninstructions_mprage_nav_2d_3sl_3avg) { run_sequences_geninstructions_case(&kMprageNavCases[3]); }

/* ------------------------------------------------------------------ */
/*  Phase 4: Frequency-modulation definition waveforms                */
/* ------------------------------------------------------------------ */

static void check_fmod_shift(const pulseqlib_collection* coll,
    const fmod_def_file* ref,
    const scan_table_file* scan,
    const float* shift,
    const float* fov_rotation,
    const char* label)
{
    pulseqlib_freq_mod_collection* fmc = NULL;
    int rc;
    int pos;
    int seen_defs[MAX_FMOD_DEFS] = {0};
    int used_plan_count = 0;
    pulseqlib_freq_mod_library* lib;
    int* used_plan;

    rc = pulseqlib_build_freq_mod_collection(&fmc, coll, shift, fov_rotation);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), label);

    lib = fmc->libs[0];
    mu_assert(lib != NULL, "freq_mod lib missing for subsequence 0");
    mu_assert_int_eq(scan->num_entries, lib->scan_table_len);
    used_plan = (int*)calloc((size_t)lib->num_plan_instances, sizeof(int));
    mu_assert(used_plan != NULL, "alloc failed for used_plan flags");

    for (pos = 0; pos < scan->num_entries; ++pos) {
        const scan_table_entry* se = &scan->entries[pos];
        const float* waveform = NULL;
        int ns = 0, s;
        float phase_rad = 0.0f;
        int has;

        has = pulseqlib_freq_mod_collection_get(
            fmc, 0, pos, &waveform, &ns, &phase_rad);

        if (se->freq_mod_id <= 0) {
            mu_assert(!has, "unexpected freq_mod at scan position");
            continue;
        }

        {
            const fmod_def* fd;
            float expected_phase;
            float phase_tol;
            float max_val = 0.0f;
            float tol;
            int def_idx = se->freq_mod_id - 1;
            int plan_idx = lib->scan_to_plan[pos];

            mu_assert(def_idx >= 0 && def_idx < ref->num_defs,
                      "invalid expected freq_mod_id in scan table");
            fd = &ref->defs[def_idx];

            mu_assert(has, "missing freq_mod at expected scan position");

            mu_assert(plan_idx >= 0 && plan_idx < lib->num_plan_instances,
                      "invalid scan_to_plan mapping");
            if (!used_plan[plan_idx]) {
                used_plan[plan_idx] = 1;
                used_plan_count++;
            }

            mu_assert_int_eq(fd->num_samples, ns);

            for (s = 0; s < ns; ++s) {
                float expected = fd->waveform_gx[s] * shift[0]
                               + fd->waveform_gy[s] * shift[1]
                               + fd->waveform_gz[s] * shift[2];
                if ((float)fabs(expected) > max_val)
                    max_val = (float)fabs(expected);
            }
            tol = max_val * 1e-4f;
            if (tol < 1e-6f) tol = 1e-6f;

            for (s = 0; s < ns; ++s) {
                float expected = fd->waveform_gx[s] * shift[0]
                               + fd->waveform_gy[s] * shift[1]
                               + fd->waveform_gz[s] * shift[2];
                mu_assert((float)fabs(waveform[s] - expected) <= tol,
                          "freq_mod waveform sample mismatch");
            }

            expected_phase = fd->ref_integral[0] * shift[0]
                           + fd->ref_integral[1] * shift[1]
                           + fd->ref_integral[2] * shift[2];
            phase_tol = (float)fabs(expected_phase) * 1e-4f;
            if (phase_tol < 1e-8f) phase_tol = 1e-8f;
            mu_assert((float)fabs(phase_rad - expected_phase) <= phase_tol,
                      "freq_mod phase mismatch");

            seen_defs[def_idx] = 1;
        }
    }

    {
        int d;
        for (d = 0; d < ref->num_defs; ++d) {
            mu_assert(seen_defs[d], "expected freq_mod definition not referenced");
        }

        /* Each ground-truth def maps to one or more plan instances
         * (e.g. nav orientations share a def but have different
         * gradient amplitudes, creating separate plan instances). */
        mu_assert(used_plan_count >= ref->num_defs,
                  "fewer plan instances than expected freq_mod defs");
    }

    free(used_plan);

    pulseqlib_freq_mod_collection_free(fmc);
}

static void run_freq_mod_definitions_case(const seq_case* tc)
{
    pulseqlib_opts opts;
    pulseqlib_collection* coll = NULL;
    fmod_def_file ref = FMOD_DEF_FILE_INIT;
    scan_table_file scan = SCAN_TABLE_FILE_INIT;
    char fmod_path[512];
    char scan_path[512];
    int rc, ok, t;

    /* Three orthogonal shifts + one combined shift */
    float shifts[4][3] = {
        {1.0e-3f, 0.0f,    0.0f   },  /* X only */
        {0.0f,    2.0e-3f, 0.0f   },  /* Y only */
        {0.0f,    0.0f,    3.0e-3f},  /* Z only */
        {1.0e-3f, 2.0e-3f, 3.0e-3f}   /* combined */
    };

    /* Three representative FOV rotations (ax, cor, sag) +
     * identity.  For blocks WITHOUT norot flag the rotation
     * has no effect — we verify invariance. */
    float rotations[4][9] = {
        {1,0,0, 0,1,0, 0,0,1},   /* identity (axial) */
        {1,0,0, 0,0,1, 0,-1,0},  /* coronal:  y->z, z->-y */
        {0,0,-1, 0,1,0, 1,0,0},  /* sagittal: x->-z, z->x */
        {0.6f,0.8f,0, -0.8f,0.6f,0, 0,0,1}  /* oblique 53° */
    };

    gre_opts_init(&opts);
    rc = load_seq_with_averages(&coll, tc->seq_file, &opts, tc->num_averages);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq failed");

    build_case_path(fmod_path, sizeof(fmod_path), tc, "_freqmod_def.bin");
    ok = parse_fmod_defs(fmod_path, &ref);
    mu_assert(ok, "failed to parse freqmod_def.bin");
    mu_assert(ref.num_defs >= 2, "expected at least 2 freq_mod defs");

    /* Parse scan table to get total position span for robust freq-mod search. */
    build_case_path(scan_path, sizeof(scan_path), tc, "_scan_table.bin");
    ok = parse_scan_table(scan_path, &scan);
    mu_assert(ok, "failed to parse scan_table.bin for freqmod positions");
    mu_assert(scan.num_entries > 0, "scan_table has no entries for freqmod search");

    /* For each shift, test all rotations — results must be identical
     * because this sequence has no rotation events / norot blocks. */
    for (t = 0; t < 4; ++t) {
        int r;
        for (r = 0; r < 4; ++r) {
            check_fmod_shift(coll, &ref,
                             &scan,
                             shifts[t],
                             rotations[r],
                             "build_freq_mod_collection failed");
        }
    }

    pulseqlib_collection_free(coll);
}

MU_TEST(test_freq_mod_definitions_gre_2d_1sl_1avg) { run_freq_mod_definitions_case(&kGreCases[0]); }
MU_TEST(test_freq_mod_definitions_gre_2d_1sl_3avg) { run_freq_mod_definitions_case(&kGreCases[1]); }
MU_TEST(test_freq_mod_definitions_gre_2d_3sl_1avg) { run_freq_mod_definitions_case(&kGreCases[2]); }
MU_TEST(test_freq_mod_definitions_gre_2d_3sl_3avg) { run_freq_mod_definitions_case(&kGreCases[3]); }

MU_TEST(test_freq_mod_definitions_mprage_2d_1sl_1avg) { run_freq_mod_definitions_case(&kMprageCases[0]); }
MU_TEST(test_freq_mod_definitions_mprage_2d_1sl_3avg) { run_freq_mod_definitions_case(&kMprageCases[1]); }
MU_TEST(test_freq_mod_definitions_mprage_2d_3sl_1avg) { run_freq_mod_definitions_case(&kMprageCases[2]); }
MU_TEST(test_freq_mod_definitions_mprage_2d_3sl_3avg) { run_freq_mod_definitions_case(&kMprageCases[3]); }
MU_TEST(test_freq_mod_definitions_mprage_nc_1sl_1avg) { run_freq_mod_definitions_case(&kMprageNoncartCases[0]); }

MU_TEST(test_freq_mod_definitions_mprage_nav_2d_1sl_1avg) { run_freq_mod_definitions_case(&kMprageNavCases[0]); }
MU_TEST(test_freq_mod_definitions_mprage_nav_2d_1sl_3avg) { run_freq_mod_definitions_case(&kMprageNavCases[1]); }
MU_TEST(test_freq_mod_definitions_mprage_nav_2d_3sl_1avg) { run_freq_mod_definitions_case(&kMprageNavCases[2]); }
MU_TEST(test_freq_mod_definitions_mprage_nav_2d_3sl_3avg) { run_freq_mod_definitions_case(&kMprageNavCases[3]); }

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
    MU_RUN_TEST(test_sequences_geninstructions_mprage_nav_2d_1sl_1avg);
    MU_RUN_TEST(test_sequences_geninstructions_mprage_nav_2d_1sl_3avg);
    MU_RUN_TEST(test_sequences_geninstructions_mprage_nav_2d_3sl_1avg);
    MU_RUN_TEST(test_sequences_geninstructions_mprage_nav_2d_3sl_3avg);
    MU_RUN_TEST(test_freq_mod_definitions_gre_2d_1sl_1avg);
    MU_RUN_TEST(test_freq_mod_definitions_gre_2d_1sl_3avg);
    MU_RUN_TEST(test_freq_mod_definitions_gre_2d_3sl_1avg);
    MU_RUN_TEST(test_freq_mod_definitions_gre_2d_3sl_3avg);
    MU_RUN_TEST(test_freq_mod_definitions_mprage_2d_1sl_1avg);
    MU_RUN_TEST(test_freq_mod_definitions_mprage_2d_1sl_3avg);
    MU_RUN_TEST(test_freq_mod_definitions_mprage_2d_3sl_1avg);
    MU_RUN_TEST(test_freq_mod_definitions_mprage_2d_3sl_3avg);
    MU_RUN_TEST(test_freq_mod_definitions_mprage_nc_1sl_1avg);
    MU_RUN_TEST(test_freq_mod_definitions_mprage_nav_2d_1sl_1avg);
    MU_RUN_TEST(test_freq_mod_definitions_mprage_nav_2d_1sl_3avg);
    MU_RUN_TEST(test_freq_mod_definitions_mprage_nav_2d_3sl_1avg);
    MU_RUN_TEST(test_freq_mod_definitions_mprage_nav_2d_3sl_3avg);
}


/* ------------------------------------------------------------------ */
/*  Phase 5: Scan table — block instance validation                   */
/* ------------------------------------------------------------------ */

static void run_scan_table_case(const seq_case* tc)
{
    pulseqlib_opts opts;
    pulseqlib_collection* coll = NULL;
    scan_table_file ref = SCAN_TABLE_FILE_INIT;
    char scan_path[512];
    int rc, ok, pos;
    int is_mprage;
    int pure_seg_id = -1;
    int pure_seg_duration_us = -1;
    int pure_inst_durations[8];
    int pure_inst_unique = 0;
    int saw_noncanonical_instance = 0;

    memset(pure_inst_durations, 0, sizeof(pure_inst_durations));
    is_mprage = (strncmp(tc->name, "mprage_", 7) == 0 &&
                 strncmp(tc->name, "mprage_nav_", 11) != 0) ? 1 : 0;

    gre_opts_init(&opts);
    rc = load_seq_with_averages(&coll, tc->seq_file, &opts, tc->num_averages);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq failed");

    build_case_path(scan_path, sizeof(scan_path), tc, "_scan_table.bin");
    ok = parse_scan_table(scan_path, &ref);
    mu_assert(ok, "failed to parse scan_table.bin");
    mu_assert(ref.num_entries > 0, "scan_table has no entries");

    /* Walk the scan table via cursor and compare each block instance */
    pulseqlib_cursor_reset(coll);
    pos = 0;

    while (pulseqlib_cursor_next(coll) == PULSEQLIB_CURSOR_BLOCK) {
        pulseqlib_block_instance inst = PULSEQLIB_BLOCK_INSTANCE_INIT;
        pulseqlib_cursor_info ci = PULSEQLIB_CURSOR_INFO_INIT;
        const scan_table_entry* e;
        float tol;
        int i;

        mu_assert(pos < ref.num_entries, "more blocks than scan_table entries");

        rc = pulseqlib_get_block_instance(coll, &inst);
        mu_assert(PULSEQLIB_SUCCEEDED(rc), "get_block_instance failed");

        rc = pulseqlib_cursor_get_info(coll, &ci);
        mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_cursor_get_info failed");

        if (is_mprage) {
            pulseqlib_segment_info segi = PULSEQLIB_SEGMENT_INFO_INIT;
            rc = pulseqlib_get_segment_info(coll, ci.segment_id, &segi);
            mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_get_segment_info failed");

            if (segi.pure_delay) {
                int found = 0;
                if (pure_seg_id < 0) {
                    pure_seg_id = ci.segment_id;
                    pure_seg_duration_us = segi.duration_us;
                } else {
                    mu_assert_int_eq(pure_seg_id, ci.segment_id);
                    mu_assert_int_eq(pure_seg_duration_us, segi.duration_us);
                }

                /* Segment definition stays canonical (one block raster). */
                mu_assert_int_eq((int)(opts.block_raster_us + 0.5f), segi.duration_us);

                if (inst.duration_us != segi.duration_us)
                    saw_noncanonical_instance = 1;

                for (i = 0; i < pure_inst_unique; ++i) {
                    if (pure_inst_durations[i] == inst.duration_us) {
                        found = 1;
                        break;
                    }
                }
                if (!found && pure_inst_unique < (int)(sizeof(pure_inst_durations)/sizeof(pure_inst_durations[0]))) {
                    pure_inst_durations[pure_inst_unique++] = inst.duration_us;
                }
            }
        }

        e = &ref.entries[pos];

        /* RF amplitude (relative tolerance or absolute for zero) */
        tol = (float)fabs(e->rf_amp_hz) * 1e-4f;
        if (tol < 1e-6f) tol = 1e-6f;
        mu_assert((float)fabs(inst.rf_amp_hz - e->rf_amp_hz) <= tol,
                  "rf_amp_hz mismatch");

        /* RF phase */
        tol = (float)fabs(e->rf_phase_rad) * 1e-4f;
        if (tol < 1e-8f) tol = 1e-8f;
        mu_assert((float)fabs(inst.rf_phase_rad - e->rf_phase_rad) <= tol,
                  "rf_phase_rad mismatch");

        /* RF freq */
        tol = (float)fabs(e->rf_freq_hz) * 1e-4f;
        if (tol < 1e-6f) tol = 1e-6f;
        mu_assert((float)fabs(inst.rf_freq_hz - e->rf_freq_hz) <= tol,
                  "rf_freq_hz mismatch");

        /* GX amplitude */
        tol = (float)fabs(e->gx_amp_hz_per_m) * 1e-4f;
        if (tol < 1e-6f) tol = 1e-6f;
            if ((float)fabs(inst.gx_amp_hz_per_m - e->gx_amp_hz_per_m) > tol)
                fprintf(stderr, "[scantable][%s] gx_amp @pos%d: ref=%.6g  lib=%.6g\n",
                    tc->name, pos, e->gx_amp_hz_per_m, inst.gx_amp_hz_per_m);
            mu_assert((float)fabs(inst.gx_amp_hz_per_m - e->gx_amp_hz_per_m) <= tol,
                  "gx_amp mismatch");

        /* GY amplitude */
        tol = (float)fabs(e->gy_amp_hz_per_m) * 1e-4f;
        if (tol < 1e-6f) tol = 1e-6f;
        mu_assert((float)fabs(inst.gy_amp_hz_per_m - e->gy_amp_hz_per_m) <= tol,
                  "gy_amp mismatch");

        /* GZ amplitude */
        tol = (float)fabs(e->gz_amp_hz_per_m) * 1e-4f;
        if (tol < 1e-6f) tol = 1e-6f;
        mu_assert((float)fabs(inst.gz_amp_hz_per_m - e->gz_amp_hz_per_m) <= tol,
                  "gz_amp mismatch");

        /* ADC flag */
        mu_assert_int_eq(e->adc_flag, inst.adc_flag);

        /* ADC phase */
        tol = (float)fabs(e->adc_phase_rad) * 1e-4f;
        if (tol < 1e-8f) tol = 1e-8f;
        mu_assert((float)fabs(inst.adc_phase_rad - e->adc_phase_rad) <= tol,
                  "adc_phase_rad mismatch");

        /* ADC freq */
        tol = (float)fabs(e->adc_freq_hz) * 1e-4f;
        if (tol < 1e-6f) tol = 1e-6f;
        mu_assert((float)fabs(inst.adc_freq_hz - e->adc_freq_hz) <= tol,
                  "adc_freq_hz mismatch");

        /* Digital output flag */
        mu_assert_int_eq(e->digitalout_flag, inst.digitalout_flag);

        /* Rotation matrix */
        for (i = 0; i < 9; ++i) {
            mu_assert((float)fabs(inst.rotmat[i] - e->rotmat[i]) < 1e-5f,
                      "rotmat mismatch");
        }

        ++pos;
    }

    mu_assert_int_eq(ref.num_entries, pos);

    if (is_mprage) {
        fprintf(stderr, "[scantable][%s] pure_seg_id=%d  pure_inst_unique=%d  saw_noncanonical=%d\n",
                tc->name, pure_seg_id, pure_inst_unique, saw_noncanonical_instance);
        mu_assert(pure_seg_id >= 0, "expected a pure-delay segment in MPRAGE scan loop");
        mu_assert(pure_inst_unique >= 2,
                  "expected multiple pure-delay instance durations (e.g. TI and TR delays)");
        mu_assert(saw_noncanonical_instance,
                  "expected scan-loop pure-delay instance duration to differ from canonical segment-def duration");
    }

    pulseqlib_collection_free(coll);
}

MU_TEST(test_scan_table_gre_2d_1sl_1avg) { run_scan_table_case(&kGreCases[0]); }
MU_TEST(test_scan_table_gre_2d_1sl_3avg) { run_scan_table_case(&kGreCases[1]); }
MU_TEST(test_scan_table_gre_2d_3sl_1avg) { run_scan_table_case(&kGreCases[2]); }
MU_TEST(test_scan_table_gre_2d_3sl_3avg) { run_scan_table_case(&kGreCases[3]); }

MU_TEST(test_scan_table_mprage_2d_1sl_1avg) { run_scan_table_case(&kMprageCases[0]); }
MU_TEST(test_scan_table_mprage_2d_1sl_3avg) { run_scan_table_case(&kMprageCases[1]); }
MU_TEST(test_scan_table_mprage_2d_3sl_1avg) { run_scan_table_case(&kMprageCases[2]); }
MU_TEST(test_scan_table_mprage_2d_3sl_3avg) { run_scan_table_case(&kMprageCases[3]); }
MU_TEST(test_scan_table_mprage_nc_1sl_1avg) { run_scan_table_case(&kMprageNoncartCases[0]); }

MU_TEST(test_scan_table_mprage_nav_2d_1sl_1avg) { run_scan_table_case(&kMprageNavCases[0]); }
MU_TEST(test_scan_table_mprage_nav_2d_1sl_3avg) { run_scan_table_case(&kMprageNavCases[1]); }
MU_TEST(test_scan_table_mprage_nav_2d_3sl_1avg) { run_scan_table_case(&kMprageNavCases[2]); }
MU_TEST(test_scan_table_mprage_nav_2d_3sl_3avg) { run_scan_table_case(&kMprageNavCases[3]); }

MU_TEST_SUITE(suite_sequences_scanloop)
{
    MU_RUN_TEST(test_scan_table_gre_2d_1sl_1avg);
    MU_RUN_TEST(test_scan_table_gre_2d_1sl_3avg);
    MU_RUN_TEST(test_scan_table_gre_2d_3sl_1avg);
    MU_RUN_TEST(test_scan_table_gre_2d_3sl_3avg);
    MU_RUN_TEST(test_scan_table_mprage_2d_1sl_1avg);
    MU_RUN_TEST(test_scan_table_mprage_2d_1sl_3avg);
    MU_RUN_TEST(test_scan_table_mprage_2d_3sl_1avg);
    MU_RUN_TEST(test_scan_table_mprage_2d_3sl_3avg);
    MU_RUN_TEST(test_scan_table_mprage_nc_1sl_1avg);
    MU_RUN_TEST(test_scan_table_mprage_nav_2d_1sl_1avg);
    MU_RUN_TEST(test_scan_table_mprage_nav_2d_1sl_3avg);
    MU_RUN_TEST(test_scan_table_mprage_nav_2d_3sl_1avg);
    MU_RUN_TEST(test_scan_table_mprage_nav_2d_3sl_3avg);
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
    MU_RUN_SUITE(suite_sequences_scanloop);
    MU_REPORT();
    return MU_EXIT_CODE;
}
