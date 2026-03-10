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

MU_TEST(test_segmentation_gre_geninstructions)
{
    pulseqlib_opts opts;
    pulseqlib_collection* coll = NULL;
    pulseqlib_collection_info cinfo = PULSEQLIB_COLLECTION_INFO_INIT;
    static seg_def_file ref;   /* static: too large (~8 MB) for stack */
    int rc, ok;
    int s, b, ax;

    /* Load sequence */
    gre_opts_init(&opts);
    rc = load_seq(&coll, "gre_2d_1sl_1avg.seq", &opts);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq failed for GRE baseline");

    rc = pulseqlib_get_collection_info(coll, &cinfo);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_get_collection_info failed");

    /* Load MATLAB ground truth */
    ok = parse_seg_def(TEST_DATA_DIR "gre_2d_1sl_1avg_segment_def.bin", &ref);
    mu_assert(ok, "failed to parse gre_2d_1sl_1avg_segment_def.bin");

    /* Number of segments must match */
    mu_assert_int_eq(ref.num_segments, cinfo.num_segments);

    for (s = 0; s < ref.num_segments; ++s) {
        pulseqlib_segment_info segi = PULSEQLIB_SEGMENT_INFO_INIT;
        rc = pulseqlib_get_segment_info(coll, s, &segi);
        mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_get_segment_info failed");
        mu_assert_int_eq(ref.num_blocks[s], segi.num_blocks);

        for (b = 0; b < ref.num_blocks[s]; ++b) {
            const seg_block_def* ref_blk = &ref.blocks[s][b];
            pulseqlib_block_info bi = PULSEQLIB_BLOCK_INFO_INIT;
            rc = pulseqlib_get_block_info(coll, s, b, &bi);
            mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_get_block_info failed");

            /* --- Flags -------------------------------------------- */
            mu_assert_int_eq(ref_blk->has_rf,          bi.has_rf);
            for (ax = 0; ax < 3; ++ax)
                mu_assert_int_eq(ref_blk->has_grad[ax], bi.has_grad[ax]);
            if (ref_blk->has_adc != bi.has_adc)
                fprintf(stderr, "has_adc mismatch seg=%d blk=%d ref=%d lib=%d\n",
                        s, b, ref_blk->has_adc, bi.has_adc);
            mu_assert_int_eq(ref_blk->has_adc,         bi.has_adc);
            mu_assert_int_eq(ref_blk->has_rotation,    bi.has_rotation);
            if (ref_blk->has_digital_out != bi.has_digitalout)
                fprintf(stderr, "DBG dout mismatch: seg=%d blk=%d ref=%d lib=%d\n",
                        s, b, ref_blk->has_digital_out, bi.has_digitalout);
            mu_assert_int_eq(ref_blk->has_digital_out, bi.has_digitalout);
            if (ref_blk->has_freq_mod != bi.has_freq_mod)
                fprintf(stderr, "DBG freq_mod mismatch: seg=%d blk=%d ref=%d lib=%d\n",
                        s, b, ref_blk->has_freq_mod, bi.has_freq_mod);
            mu_assert_int_eq(ref_blk->has_freq_mod,    bi.has_freq_mod);

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

                /* For GEHC builds the library returns physical Hz
                   (shape × base_amplitude_hz); MATLAB stores normalised
                   wave (peak=1) + amplitude. Reconstruct full-scale
                   reference for comparison. */
                for (i = 0; i < num_samples; ++i) {
                    float ref_val = ref_blk->rf_rho[i] * ref_blk->rf_amp;
                    mu_assert(GENI_AMP_NEAR(ref_val, mag[0][i]),
                              "RF magnitude shape mismatch");
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

                    /* Library returns shape × per-instance amplitude from
                       the max-energy segment instance (signed).  MATLAB
                       stores normalised wave (peak=1) + signed amplitude.
                       Reconstruct full-scale reference for comparison. */
                    for (i = 0; i < num_samples; ++i) {
                        float ref_val = ref_blk->grad_wave[ax][i] * ref_blk->grad_amp[ax];
                        if (!GENI_AMP_NEAR(ref_val, amps[0][i])) {
                            fprintf(stderr, "GRAD MISMATCH seg=%d blk=%d ax=%d i=%d ref=%e lib=%e ref_wave=%e ref_amp=%e\n",
                                    s, b, ax, i, ref_val, amps[0][i],
                                    ref_blk->grad_wave[ax][i], ref_blk->grad_amp[ax]);
                        }
                        mu_assert(GENI_AMP_NEAR(ref_val, amps[0][i]),
                                  "grad waveform amplitude mismatch");
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
        }
    }

    pulseqlib_collection_free(coll);
}

MU_TEST_SUITE(suite_segmentation_phase3)
{
    MU_RUN_TEST(test_segmentation_gre_geninstructions);
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
    MU_RUN_SUITE(suite_segmentation_phase3);
    MU_REPORT();
    return MU_EXIT_CODE;
}
