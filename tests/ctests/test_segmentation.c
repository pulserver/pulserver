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
/*  Phase 3: block-level geninstruction tests                         */
/* ------------------------------------------------------------------ */

/* Tolerances */
#define BLK_AMP_REL_TOL   1e-3f   /* 0.1% relative for waveforms */
#define BLK_AMP_ABS_FLOOR 1.0f    /* Hz/m absolute floor          */
#define BLK_TIME_TOL      1.0f    /* us for time values            */
#define BLK_RF_TOL        1e-5f   /* absolute for normalized RF    */

static void free_2d(float** arr, int n)
{
    int i;
    if (!arr) return;
    for (i = 0; i < n; ++i)
        free(arr[i]);
    free(arr);
}

MU_TEST(test_segmentation_gre_block_instructions)
{
    pulseqlib_opts opts;
    pulseqlib_collection* coll = NULL;
    pulseqlib_segment_info segi = PULSEQLIB_SEGMENT_INFO_INIT;
    block_meta bm = BLOCK_META_INIT;
    rf_mag_waveform rf_ref = RF_MAG_WAVEFORM_INIT;
    arb_grad_waveform arb_b2x = ARB_GRAD_WAVEFORM_INIT;
    arb_grad_waveform arb_b3x = ARB_GRAD_WAVEFORM_INIT;
    int rc, b, ok, axis;

    /* Load sequence */
    gre_opts_init(&opts);
    rc = load_seq(&coll, "gre_2d_1sl_1avg.seq", &opts);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq failed for GRE baseline");

    /* Parse ground truth files */
    ok = parse_block_meta(TEST_DATA_DIR "gre_2d_1sl_1avg_block_meta.txt", &bm);
    mu_assert(ok, "failed to parse gre_2d_1sl_1avg_block_meta.txt");

    ok = parse_rf_mag(TEST_DATA_DIR "gre_2d_1sl_1avg_rf_mag.bin", &rf_ref);
    mu_assert(ok, "failed to parse gre_2d_1sl_1avg_rf_mag.bin");

    ok = parse_arb_grad(TEST_DATA_DIR "gre_2d_1sl_1avg_arb_grad_b2_x.bin", &arb_b2x);
    mu_assert(ok, "failed to parse arb_grad_b2_x.bin");

    ok = parse_arb_grad(TEST_DATA_DIR "gre_2d_1sl_1avg_arb_grad_b3_x.bin", &arb_b3x);
    mu_assert(ok, "failed to parse arb_grad_b3_x.bin");

    /* --- Segment gap ------------------------------------------------- */
    rc = pulseqlib_get_segment_info(coll, 0, &segi);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_get_segment_info failed");
    mu_assert_int_eq(bm.rf_adc_gap_us, segi.rf_adc_gap_us);

    /* --- Per-block metadata ------------------------------------------ */
    for (b = 0; b < bm.num_blocks; ++b) {
        pulseqlib_block_info bi = PULSEQLIB_BLOCK_INFO_INIT;
        rc = pulseqlib_get_block_info(coll, 0, b, &bi);
        mu_assert(PULSEQLIB_SUCCEEDED(rc), "pulseqlib_get_block_info failed");

        /* Timing */
        mu_assert_int_eq(bm.duration_us[b], bi.duration_us);
        mu_assert_int_eq(bm.start_time_us[b], bi.start_time_us);

        /* Boolean flags (SPGR: no digitalout, no rotation) */
        mu_assert_int_eq(0, bi.has_digitalout);
        mu_assert_int_eq(0, bi.has_rotation);
        mu_assert_int_eq(0, bi.norot_flag);
        mu_assert_int_eq(0, bi.nopos_flag);

        /* freq_mod: blocks 0,2 have (rf||adc)+grad; blocks 1,3 do not */
        mu_assert_int_eq((b == 0 || b == 2) ? 1 : 0, bi.has_freq_mod);
    }

    /* --- Block 0: RF ------------------------------------------------- */
    {
        pulseqlib_block_info bi = PULSEQLIB_BLOCK_INFO_INIT;
        int nch = 0, ns = 0, i;
        float** mag;

        pulseqlib_get_block_info(coll, 0, 0, &bi);

        mu_assert_int_eq(1, bi.has_rf);
        mu_assert_int_eq(bm.rf_delay_us, bi.rf_delay_us);
        mu_assert_int_eq(bm.rf_num_samples, bi.rf_num_samples);
        mu_assert_int_eq(bm.rf_is_complex, bi.rf_is_complex);
        mu_assert_int_eq(bm.rf_num_channels, bi.rf_num_channels);

        /* RF magnitude waveform */
        mag = pulseqlib_get_rf_magnitude(coll, 0, 0, &nch, &ns);
        mu_assert(mag != NULL, "pulseqlib_get_rf_magnitude returned NULL");
        mu_assert_int_eq(bm.rf_num_channels, nch);
        mu_assert_int_eq(rf_ref.num_samples, ns);

        for (i = 0; i < ns; ++i) {
            mu_assert_float_near("RF magnitude sample",
                rf_ref.magnitude[i], mag[0][i], BLK_RF_TOL);
        }
        free_2d(mag, nch);
    }

    /* --- Block 0: Gz trap (slice-select) ----------------------------- */
    {
        pulseqlib_block_info bi = PULSEQLIB_BLOCK_INFO_INIT;
        int nshots = 0, ns = 0;
        float** amps;
        float* times;

        pulseqlib_get_block_info(coll, 0, 0, &bi);
        mu_assert_int_eq(1, bi.has_grad[2]);
        mu_assert_int_eq(1, bi.grad_is_trapezoid[2]);

        amps = pulseqlib_get_grad_amplitude(coll, 0, 0, 2, &nshots, &ns);
        mu_assert(amps != NULL, "grad amp block 0 Gz NULL");
        times = pulseqlib_get_grad_time_us(coll, 0, 0, 2);
        mu_assert(times != NULL, "grad time block 0 Gz NULL");

        /* Trap corners: 4 points (has flat) or 3 (no flat) */
        {
            int rise = bm.trap_rise_us[0][2];
            int flat = bm.trap_flat_us[0][2];
            int fall = bm.trap_fall_us[0][2];
            float amp = bm.trap_amplitude[0][2];
            int expected_ns = (flat > 0) ? 4 : 3;

            mu_assert_int_eq(expected_ns, ns);
            /* Time corners */
            mu_assert_float_near("Gz b0 t0", 0.0f, times[0], BLK_TIME_TOL);
            mu_assert_float_near("Gz b0 t1", (float)rise, times[1], BLK_TIME_TOL);
            if (flat > 0) {
                mu_assert_float_near("Gz b0 t2", (float)(rise + flat), times[2], BLK_TIME_TOL);
                mu_assert_float_near("Gz b0 t3", (float)(rise + flat + fall), times[3], BLK_TIME_TOL);
            } else {
                mu_assert_float_near("Gz b0 t2", (float)(rise + fall), times[2], BLK_TIME_TOL);
            }
            /* Amplitude corners */
            mu_assert_float_near("Gz b0 a0", 0.0f, amps[0][0], BLK_AMP_ABS_FLOOR);
            mu_assert_float_near("Gz b0 a1", amp, amps[0][1], fabsf(amp) * BLK_AMP_REL_TOL + BLK_AMP_ABS_FLOOR);
            if (flat > 0) {
                mu_assert_float_near("Gz b0 a2", amp, amps[0][2], fabsf(amp) * BLK_AMP_REL_TOL + BLK_AMP_ABS_FLOOR);
                mu_assert_float_near("Gz b0 a3", 0.0f, amps[0][3], BLK_AMP_ABS_FLOOR);
            } else {
                mu_assert_float_near("Gz b0 a2", 0.0f, amps[0][2], BLK_AMP_ABS_FLOOR);
            }
        }
        free(times);
        free_2d(amps, nshots);
    }

    /* --- Block 1: Gx_pre, Gy_pre, Gz_reph (all traps) --------------- */
    for (axis = 0; axis < 3; ++axis) {
        pulseqlib_block_info bi = PULSEQLIB_BLOCK_INFO_INIT;
        int nshots = 0, ns = 0;
        float** amps;
        float* times;
        int rise, flat, fall;
        float amp;
        int expected_ns;

        if (!bm.has_trap[1][axis]) continue;

        pulseqlib_get_block_info(coll, 0, 1, &bi);
        mu_assert_int_eq(1, bi.has_grad[axis]);
        mu_assert_int_eq(1, bi.grad_is_trapezoid[axis]);

        amps = pulseqlib_get_grad_amplitude(coll, 0, 1, axis, &nshots, &ns);
        mu_assert(amps != NULL, "grad amp block 1 NULL");
        times = pulseqlib_get_grad_time_us(coll, 0, 1, axis);
        mu_assert(times != NULL, "grad time block 1 NULL");

        rise = bm.trap_rise_us[1][axis];
        flat = bm.trap_flat_us[1][axis];
        fall = bm.trap_fall_us[1][axis];
        amp  = bm.trap_amplitude[1][axis];
        expected_ns = (flat > 0) ? 4 : 3;
        mu_assert_int_eq(expected_ns, ns);

        /* Time corners */
        mu_assert_float_near("b1 t0", 0.0f, times[0], BLK_TIME_TOL);
        mu_assert_float_near("b1 t1", (float)rise, times[1], BLK_TIME_TOL);
        if (flat > 0) {
            mu_assert_float_near("b1 t2", (float)(rise + flat), times[2], BLK_TIME_TOL);
            mu_assert_float_near("b1 t3", (float)(rise + flat + fall), times[3], BLK_TIME_TOL);
        } else {
            mu_assert_float_near("b1 t2", (float)(rise + fall), times[2], BLK_TIME_TOL);
        }

        /* Amplitude corners */
        mu_assert_float_near("b1 a0", 0.0f, amps[0][0], BLK_AMP_ABS_FLOOR);
        mu_assert_float_near("b1 a1", amp, amps[0][1], fabsf(amp) * BLK_AMP_REL_TOL + BLK_AMP_ABS_FLOOR);
        if (flat > 0) {
            mu_assert_float_near("b1 a2", amp, amps[0][2], fabsf(amp) * BLK_AMP_REL_TOL + BLK_AMP_ABS_FLOOR);
            mu_assert_float_near("b1 a3", 0.0f, amps[0][3], BLK_AMP_ABS_FLOOR);
        } else {
            mu_assert_float_near("b1 a2", 0.0f, amps[0][2], BLK_AMP_ABS_FLOOR);
        }

        free(times);
        free_2d(amps, nshots);
    }

    /* --- Block 2: Gx arbitrary (split readout trap) + ADC ------------ */
    {
        pulseqlib_block_info bi = PULSEQLIB_BLOCK_INFO_INIT;
        int nshots = 0, ns = 0, i, n;
        float** amps;
        float* times;

        pulseqlib_get_block_info(coll, 0, 2, &bi);

        /* ADC metadata */
        mu_assert_int_eq(1, bi.has_adc);
        mu_assert_int_eq(bm.adc_delay_us, bi.adc_delay_us);
        mu_assert_int_eq(0, bi.adc_def_id);  /* single ADC type */

        /* Gx arbitrary */
        mu_assert_int_eq(1, bi.has_grad[0]);
        mu_assert_int_eq(0, bi.grad_is_trapezoid[0]);

        amps = pulseqlib_get_grad_amplitude(coll, 0, 2, 0, &nshots, &ns);
        mu_assert(amps != NULL, "grad amp block 2 Gx NULL");
        times = pulseqlib_get_grad_time_us(coll, 0, 2, 0);
        mu_assert(times != NULL, "grad time block 2 Gx NULL");

        n = (arb_b2x.num_samples < ns) ? arb_b2x.num_samples : ns;
        mu_assert(abs(arb_b2x.num_samples - ns) <= 1, "arb b2 Gx sample count mismatch > 1");

        for (i = 0; i < n; ++i) {
            float ref_a = arb_b2x.amplitude[i];
            float tol = fabsf(ref_a) * BLK_AMP_REL_TOL;
            if (tol < BLK_AMP_ABS_FLOOR) tol = BLK_AMP_ABS_FLOOR;
            mu_assert_float_near("b2 Gx amp", ref_a, amps[0][i], tol);
            mu_assert_float_near("b2 Gx time", arb_b2x.time_us[i], times[i], BLK_TIME_TOL);
        }

        free(times);
        free_2d(amps, nshots);
    }

    /* --- Block 3: Gx_spoil arbitrary + Gy_rew trap + Gz_spoil trap --- */
    {
        pulseqlib_block_info bi = PULSEQLIB_BLOCK_INFO_INIT;
        pulseqlib_get_block_info(coll, 0, 3, &bi);

        /* Gx_spoil (arbitrary) */
        {
            int nshots = 0, ns = 0, i, n;
            float** amps;
            float* times;

            mu_assert_int_eq(1, bi.has_grad[0]);
            mu_assert_int_eq(0, bi.grad_is_trapezoid[0]);

            amps = pulseqlib_get_grad_amplitude(coll, 0, 3, 0, &nshots, &ns);
            mu_assert(amps != NULL, "grad amp block 3 Gx NULL");
            times = pulseqlib_get_grad_time_us(coll, 0, 3, 0);
            mu_assert(times != NULL, "grad time block 3 Gx NULL");

            n = (arb_b3x.num_samples < ns) ? arb_b3x.num_samples : ns;
            mu_assert(abs(arb_b3x.num_samples - ns) <= 1,
                      "arb b3 Gx sample count mismatch > 1");

            for (i = 0; i < n; ++i) {
                float ref_a = arb_b3x.amplitude[i];
                float tol = fabsf(ref_a) * BLK_AMP_REL_TOL;
                if (tol < BLK_AMP_ABS_FLOOR) tol = BLK_AMP_ABS_FLOOR;
                mu_assert_float_near("b3 Gx amp", ref_a, amps[0][i], tol);
                mu_assert_float_near("b3 Gx time", arb_b3x.time_us[i], times[i], BLK_TIME_TOL);
            }

            free(times);
            free_2d(amps, nshots);
        }

        /* Gy_rew (trap) */
        if (bm.has_trap[3][1]) {
            int nshots = 0, ns = 0;
            float** amps;
            float* times;
            int rise = bm.trap_rise_us[3][1];
            int flat = bm.trap_flat_us[3][1];
            int fall = bm.trap_fall_us[3][1];
            float amp = bm.trap_amplitude[3][1];
            int expected_ns = (flat > 0) ? 4 : 3;

            mu_assert_int_eq(1, bi.has_grad[1]);
            mu_assert_int_eq(1, bi.grad_is_trapezoid[1]);

            amps = pulseqlib_get_grad_amplitude(coll, 0, 3, 1, &nshots, &ns);
            mu_assert(amps != NULL, "grad amp block 3 Gy NULL");
            times = pulseqlib_get_grad_time_us(coll, 0, 3, 1);
            mu_assert(times != NULL, "grad time block 3 Gy NULL");

            mu_assert_int_eq(expected_ns, ns);
            mu_assert_float_near("b3 Gy t0", 0.0f, times[0], BLK_TIME_TOL);
            mu_assert_float_near("b3 Gy t1", (float)rise, times[1], BLK_TIME_TOL);
            if (flat > 0) {
                mu_assert_float_near("b3 Gy t2", (float)(rise + flat), times[2], BLK_TIME_TOL);
                mu_assert_float_near("b3 Gy t3", (float)(rise + flat + fall), times[3], BLK_TIME_TOL);
            } else {
                mu_assert_float_near("b3 Gy t2", (float)(rise + fall), times[2], BLK_TIME_TOL);
            }
            mu_assert_float_near("b3 Gy a0", 0.0f, amps[0][0], BLK_AMP_ABS_FLOOR);
            mu_assert_float_near("b3 Gy a1", amp, amps[0][1], fabsf(amp) * BLK_AMP_REL_TOL + BLK_AMP_ABS_FLOOR);
            if (flat > 0) {
                mu_assert_float_near("b3 Gy a2", amp, amps[0][2], fabsf(amp) * BLK_AMP_REL_TOL + BLK_AMP_ABS_FLOOR);
                mu_assert_float_near("b3 Gy a3", 0.0f, amps[0][3], BLK_AMP_ABS_FLOOR);
            } else {
                mu_assert_float_near("b3 Gy a2", 0.0f, amps[0][2], BLK_AMP_ABS_FLOOR);
            }

            free(times);
            free_2d(amps, nshots);
        }

        /* Gz_spoil (trap) */
        if (bm.has_trap[3][2]) {
            int nshots = 0, ns = 0;
            float** amps;
            float* times;
            int rise = bm.trap_rise_us[3][2];
            int flat = bm.trap_flat_us[3][2];
            int fall = bm.trap_fall_us[3][2];
            float amp = bm.trap_amplitude[3][2];
            int expected_ns = (flat > 0) ? 4 : 3;

            mu_assert_int_eq(1, bi.has_grad[2]);
            mu_assert_int_eq(1, bi.grad_is_trapezoid[2]);

            amps = pulseqlib_get_grad_amplitude(coll, 0, 3, 2, &nshots, &ns);
            mu_assert(amps != NULL, "grad amp block 3 Gz NULL");
            times = pulseqlib_get_grad_time_us(coll, 0, 3, 2);
            mu_assert(times != NULL, "grad time block 3 Gz NULL");

            mu_assert_int_eq(expected_ns, ns);
            mu_assert_float_near("b3 Gz t0", 0.0f, times[0], BLK_TIME_TOL);
            mu_assert_float_near("b3 Gz t1", (float)rise, times[1], BLK_TIME_TOL);
            if (flat > 0) {
                mu_assert_float_near("b3 Gz t2", (float)(rise + flat), times[2], BLK_TIME_TOL);
                mu_assert_float_near("b3 Gz t3", (float)(rise + flat + fall), times[3], BLK_TIME_TOL);
            } else {
                mu_assert_float_near("b3 Gz t2", (float)(rise + fall), times[2], BLK_TIME_TOL);
            }
            mu_assert_float_near("b3 Gz a0", 0.0f, amps[0][0], BLK_AMP_ABS_FLOOR);
            mu_assert_float_near("b3 Gz a1", amp, amps[0][1], fabsf(amp) * BLK_AMP_REL_TOL + BLK_AMP_ABS_FLOOR);
            if (flat > 0) {
                mu_assert_float_near("b3 Gz a2", amp, amps[0][2], fabsf(amp) * BLK_AMP_REL_TOL + BLK_AMP_ABS_FLOOR);
                mu_assert_float_near("b3 Gz a3", 0.0f, amps[0][3], BLK_AMP_ABS_FLOOR);
            } else {
                mu_assert_float_near("b3 Gz a2", 0.0f, amps[0][2], BLK_AMP_ABS_FLOOR);
            }

            free(times);
            free_2d(amps, nshots);
        }
    }

    /* Cleanup */
    free_arb_grad(&arb_b3x);
    free_arb_grad(&arb_b2x);
    free_rf_mag(&rf_ref);
    pulseqlib_collection_free(coll);
}

MU_TEST_SUITE(suite_segmentation_phase3)
{
    MU_RUN_TEST(test_segmentation_gre_block_instructions);
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
