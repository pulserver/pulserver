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
            mu_assert_int_eq(ref_blk->has_adc,         bi.has_adc);
            mu_assert_int_eq(ref_blk->has_rotation,    bi.has_rotation);
            mu_assert_int_eq(ref_blk->has_digital_out, bi.has_digitalout);
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
                        mu_assert(GENI_AMP_NEAR(ref_blk->grad_wave[ax][i], amps[0][i]),
                                  "grad waveform shape mismatch");
                    }

                    /* Amplitude checks via new getters */
                    {
                        float init_amp = pulseqlib_get_grad_initial_amplitude_hz_per_m(
                                             coll, s, b, ax);
                        float max_amp  = pulseqlib_get_grad_max_amplitude_hz_per_m(
                                             coll, s, b, ax);
                        mu_assert(GENI_AMP_NEAR(ref_blk->grad_amp[ax], init_amp),
                                  "grad initial amplitude mismatch");
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

            /* --- Freq-mod (overlap API) --------------------------- */
            {
                int need_ns = 0;
                int need = pulseqlib_block_needs_freq_mod(coll, s, b, &need_ns);
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

MU_TEST_SUITE(suite_segmentation_phase3)
{
    MU_RUN_TEST(test_segmentation_gre_geninstructions);
}

/* ------------------------------------------------------------------ */
/*  Phase 4: Frequency-modulation definition waveforms                */
/* ------------------------------------------------------------------ */

/* Helper: verify freq-mod waveforms + phase for a given shift/rotation. */
static void check_fmod_shift(
    const pulseqlib_collection* coll,
    const fmod_def_file* ref,
    const float* shift,
    const float* fov_rotation,
    const int* test_positions,
    const char* label)
{
    pulseqlib_freq_mod_collection* fmc = NULL;
    int rc, d;

    rc = pulseqlib_build_freq_mod_collection(&fmc, coll, shift, fov_rotation);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), label);

    for (d = 0; d < ref->num_defs; ++d) {
        const fmod_def* fd = &ref->defs[d];
        const float* waveform = NULL;
        int ns = 0, s;
        float phase_rad = 0.0f;
        int has;

        has = pulseqlib_freq_mod_collection_get(
            fmc, 0, test_positions[d], &waveform, &ns, &phase_rad);
        mu_assert(has, "expected freq_mod at test position");
        mu_assert_int_eq(fd->num_samples, ns);

        /* Compare 1D waveform: expected = gx*s[0] + gy*s[1] + gz*s[2] */
        {
            float max_val = 0.0f;
            float tol;

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
        }

        /* Compare phase: expected = ref_integral . shift */
        {
            float expected_phase = fd->ref_integral[0] * shift[0]
                                 + fd->ref_integral[1] * shift[1]
                                 + fd->ref_integral[2] * shift[2];
            float phase_tol = (float)fabs(expected_phase) * 1e-4f;
            if (phase_tol < 1e-8f) phase_tol = 1e-8f;
            mu_assert((float)fabs(phase_rad - expected_phase) <= phase_tol,
                      "freq_mod phase mismatch");
        }
    }

    pulseqlib_freq_mod_collection_free(fmc);
}

MU_TEST(test_freq_mod_definitions)
{
    pulseqlib_opts opts;
    pulseqlib_collection* coll = NULL;
    fmod_def_file ref = FMOD_DEF_FILE_INIT;
    int rc, ok, t;

    /* RF block: scan pos 0.  ADC block: scan pos 22. */
    int test_positions[2] = {0, 22};

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
    rc = load_seq(&coll, "gre_2d_1sl_1avg.seq", &opts);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq failed");

    ok = parse_fmod_defs(TEST_DATA_DIR "gre_2d_1sl_1avg_freqmod_def.bin", &ref);
    mu_assert(ok, "failed to parse freqmod_def.bin");
    mu_assert_int_eq(2, ref.num_defs);

    /* For each shift, test all rotations — results must be identical
     * because this sequence has no rotation events / norot blocks. */
    for (t = 0; t < 4; ++t) {
        int r;
        for (r = 0; r < 4; ++r) {
            check_fmod_shift(coll, &ref, shifts[t],
                             rotations[r], test_positions,
                             "build_freq_mod_collection failed");
        }
    }

    pulseqlib_collection_free(coll);
}

MU_TEST_SUITE(suite_segmentation_phase4)
{
    MU_RUN_TEST(test_freq_mod_definitions);
}

/* ------------------------------------------------------------------ */
/*  Phase 5: Scan table — block instance validation                   */
/* ------------------------------------------------------------------ */

MU_TEST(test_scan_table)
{
    pulseqlib_opts opts;
    pulseqlib_collection* coll = NULL;
    scan_table_file ref = SCAN_TABLE_FILE_INIT;
    int rc, ok, pos;

    gre_opts_init(&opts);
    rc = load_seq(&coll, "gre_2d_1sl_1avg.seq", &opts);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq failed");

    ok = parse_scan_table(TEST_DATA_DIR "gre_2d_1sl_1avg_scan_table.bin", &ref);
    mu_assert(ok, "failed to parse scan_table.bin");
    mu_assert(ref.num_entries > 0, "scan_table has no entries");

    /* Walk the scan table via cursor and compare each block instance */
    pulseqlib_cursor_reset(coll);
    pos = 0;

    while (pulseqlib_cursor_next(coll) == PULSEQLIB_CURSOR_BLOCK) {
        pulseqlib_block_instance inst = PULSEQLIB_BLOCK_INSTANCE_INIT;
        const scan_table_entry* e;
        float tol;
        int i;

        mu_assert(pos < ref.num_entries, "more blocks than scan_table entries");

        rc = pulseqlib_get_block_instance(coll, &inst);
        mu_assert(PULSEQLIB_SUCCEEDED(rc), "get_block_instance failed");

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

    pulseqlib_collection_free(coll);
}

MU_TEST_SUITE(suite_segmentation_phase5)
{
    MU_RUN_TEST(test_scan_table);
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
    MU_RUN_SUITE(suite_segmentation_phase4);
    MU_RUN_SUITE(suite_segmentation_phase5);
    MU_REPORT();
    return MU_EXIT_CODE;
}
