/*
 * test_rf_stats.c -- RF statistics and consistency tests.
 *
 * Suite A: RF180 block pulse ground truth (1 test).
 * Suite B: RF periodicity consistency checks (4 tests).
 */
#include "test_helpers.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================================================================== */
/*  Test-local vendor RF-stats callback (exercises pulseg_opts.
 *  vendor_rf_stats_fn / pulseg_rf_view -- the D8-A hook). Reproduces the
 *  GE-style abswidth/effwidth/dtycyc/maxpw envelope stats verbatim from
 *  the on-scanner implementation (src_gelib/pulserver_ge_rf_stats.c) so
 *  the public ctest ground truth below stays meaningful without the
 *  public library computing anything vendor-specific itself.          */
/* ================================================================== */

static int test_ge_rf_stats_cb(void *ctx, const pulseg_rf_view *rf, float out_stat[4])
{
    const float DTY_THRESHOLD = 0.2236f;
    float sum_abs, sum_sq, time_above_threshold, temp_pw, maxpw, rf_abs;
    int i;

    (void)ctx;
    sum_abs = 0.0f;
    sum_sq = 0.0f;
    time_above_threshold = 0.0f;
    maxpw = 0.0f;
    temp_pw = 0.0f;
    for (i = 0; i < rf->n; ++i)
    {
        rf_abs = rf->mag[i];
        sum_abs += rf_abs;
        sum_sq += rf_abs * rf_abs;
        if (rf_abs > DTY_THRESHOLD)
        {
            time_above_threshold += 1.0f;
            temp_pw += 1.0f;
        }
        else
        {
            if (temp_pw > maxpw)
                maxpw = temp_pw;
            temp_pw = 0.0f;
        }
    }
    if (temp_pw > maxpw)
        maxpw = temp_pw;
    if (time_above_threshold < maxpw)
        time_above_threshold = maxpw;

    out_stat[0] = sum_abs / (float)rf->n;
    out_stat[1] = sum_sq / (float)rf->n;
    out_stat[2] = time_above_threshold / (float)rf->n;
    out_stat[3] = maxpw / (float)rf->n;
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Suite A — RF180 block pulse ground truth                          */
/* ================================================================== */

/*
 * Ground truth derived from the standard GE 1 ms hard-pulse
 * RF_PULSE struct, converted to SI / pulseg conventions:
 *
 *   abswidth        = 1.0       (normalized)
 *   effwidth        = 1.0       (normalized)
 *   dtycyc          = 1.0       (normalized)
 *   maxpw           = 1.0       (normalized)
 *   maxb1           = 0.1174 G  -> 500 Hz  (gamma * 1e-4 * 0.1174)
 *   nom_fa          = 180 deg   -> pi rad  (2pi * 500 * 0.001)
 *   nom_pw          = 1000 us   -> ~999 us (N-1 raster samples)
 *   isodelay        = 500 us    -> ~499 us (int truncation)
 *   area            = 1.0       -> 0.001 s (duration_s * 1.0)
 *   num_samples     = 2 (raw decompressed: block pulse has only
 *                       start + end samples in .seq file)
 *
 * The bandwidth is measured, and is **not** the GE struct's nom_bw of 3125 Hz.
 *
 * This assertion used to read 3123, which is `3.12 / duration` -- the analytic
 * stand-in for a *sinc*, and near GE's nominal figure by coincidence rather
 * than by agreement.  It was the fallback, and it fired because the resampling
 * ahead of the transform clamped instead of zero-padding
 * (`pulseg__interp1_linear` holds the first and last sample outside the source
 * range, where `mr.calcRfBandwidth` uses `interp1(..., 'linear', 0)`).  A hard
 * pulse is magnitude 1 at both ends, so clamping extended it to DC across the
 * whole +-50 ms window, the spectrum collapsed onto a spike at zero, both
 * flanks landed there, and the width came back non-positive.
 *
 * With the transform moved to pulseq_rf.c and the zero-padding restored, the
 * answer is the physical one: a rect of duration T has a -6 dB full width of
 * 1.2067 / T, which is 1207 Hz here, reported as 1200 on the 10 Hz grid.
 * Verified independently against a Hamming-windowed sinc of TBW 8, which comes
 * back at 8000 Hz.
 */

MU_TEST(test_rf180_block_pulse_stats)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    pulseg_rf_stats stats = PULSEG_RF_STATS_INIT;
    int rc;

    default_opts_init(&opts);
    opts.vendor_rf_stats_fn = test_ge_rf_stats_cb;
    opts.vendor_rf_stats_ctx = NULL;
    rc = load_seq(&coll, "00_basic_rfstat.seq", &opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

    rc = pulseg_get_rf_stats(coll, &stats, 0, 0);
    mu_assert(PULSEG_SUCCEEDED(rc), "get_rf_stats failed");

    mu_assert_float_near("vendor_stat[0] (abs_width)", 1.0f, stats.vendor_stat[0], 1e-4f);
    mu_assert_float_near("vendor_stat[1] (eff_width)", 1.0f, stats.vendor_stat[1], 1e-4f);
    mu_assert_float_near("vendor_stat[2] (duty_cycle)", 1.0f, stats.vendor_stat[2], 1e-4f);
    mu_assert_float_near("vendor_stat[3] (max_pw)", 1.0f, stats.vendor_stat[3], 1e-4f);

    mu_assert_float_near("base_amp_hz", 500.0f, stats.base_amplitude_hz, 1.0f);
    mu_assert_float_near("flip_angle", (float)M_PI, stats.flip_angle_rad, 0.01f);

    mu_assert_float_near("duration_us", 999.0f, stats.duration_us, 2.0f);
    mu_assert(abs(stats.isodelay_us - 499) <= 2, "isodelay_us");

    mu_assert_float_near("area", 0.001f, stats.area, 1e-5f);
    /* 1.2067 / 1 ms, on a 10 Hz grid.  See the header comment. */
    mu_assert_float_near("bandwidth", 1200.0f, stats.bandwidth_hz, 20.0f);

    mu_assert_int_eq(2, stats.num_samples);

    pulseg_collection_free(coll);
}

MU_TEST(test_rf_array_basic_canonical_tr)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    pulseg_rf_stats *pulses = NULL;
    int rc, npulses;

    default_opts_init(&opts);
    rc = load_seq(&coll, "00_basic_rfstat.seq", &opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

    npulses = pulseg_get_rf_array(coll, &pulses, 0);
    mu_assert_int_eq(1, npulses);
    mu_assert_int_eq(1, pulses[0].num_instances);
    mu_assert_float_near("canonical base_amp_hz", 500.0f, pulses[0].base_amplitude_hz, 1.0f);

    free(pulses);
    pulseg_collection_free(coll);
}

MU_TEST(test_rf_array_nondegenerate_fullpass_expanded)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    pulseg_rf_stats *pulses = NULL;
    int rc, npulses, i;

    default_opts_init(&opts);
    rc = load_seq_with_averages(&coll, "05_rfprep_ok_canonical_fullpass.seq", &opts, 3);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq_with_averages failed");

    npulses = pulseg_get_rf_array(coll, &pulses, 0);
    mu_assert_int_eq(8, npulses);
    mu_assert_float_near("prep act_amp_hz", 125.0f, pulses[0].act_amplitude_hz, 1.0f);
    mu_assert_float_near("cooldown act_amp_hz", 500.0f, pulses[npulses - 1].act_amplitude_hz, 1.0f);
    for (i = 0; i < npulses; ++i)
        mu_assert_int_eq(1, pulses[i].num_instances);

    free(pulses);
    pulseg_collection_free(coll);
}

MU_TEST_SUITE(suite_rf_stats)
{
    MU_RUN_TEST(test_rf180_block_pulse_stats);
    MU_RUN_TEST(test_rf_array_basic_canonical_tr);
    MU_RUN_TEST(test_rf_array_nondegenerate_fullpass_expanded);
}

/* ================================================================== */
/*  Suite B — RF consistency (periodicity) checks                     */
/* ================================================================== */

static pulseg_opts s_rf_opts;

static void rf_consistency_setup(void)
{
    default_opts_init(&s_rf_opts);
}

static void run_consistency_check(const char *filename, int expected_code)
{
    pulseg_collection *coll = NULL;
    pulseg_diagnostic diag = PULSEG_DIAGNOSTIC_INIT;
    int rc;

    rc = load_seq(&coll, filename, &s_rf_opts);
    if (PULSEG_FAILED(rc))
    {
        /* Load failed — only acceptable if we expected this error */
        if (expected_code > 0)
            mu_fail("load_seq failed unexpectedly");
        mu_assert_int_eq(expected_code, rc);
        return;
    }

    rc = pulseg_check_consistency(coll, &diag);

    if (expected_code > 0)
    {
        mu_assert(PULSEG_SUCCEEDED(rc), "expected consistency pass");
    }
    else
    {
        mu_assert_int_eq(expected_code, rc);
    }

    pulseg_collection_free(coll);
}

/* Build a synthetic descriptor in-memory to trigger a structural
 * TR pattern mismatch (-103) without involving RF periodicity logic.
 * Pattern is A,B,A,B,A,C with long active duration (>15 s) so
 * single-TR fallback is not allowed. */
static int run_structural_tr_mismatch_probe(void)
{
    pulseg_sequence_descriptor desc = PULSEG_SEQUENCE_DESCRIPTOR_INIT;
    pulseg_diagnostic diag = PULSEG_DIAGNOSTIC_INIT;
    pulseg_base_block defs[3];
    pulseg_block_table_element table[6];

    defs[0].id = 0;
    defs[0].duration_us = 3000000;
    defs[0].rf_id = 0;
    defs[0].gx_id = 0;
    defs[0].gy_id = -1;
    defs[0].gz_id = -1;
    defs[0].adc_id = -1;

    defs[1].id = 1;
    defs[1].duration_us = 3000000;
    defs[1].rf_id = 0;
    defs[1].gx_id = -1;
    defs[1].gy_id = -1;
    defs[1].gz_id = -1;
    defs[1].adc_id = -1;

    defs[2].id = 2;
    defs[2].duration_us = 3000000;
    defs[2].rf_id = -1;
    defs[2].gx_id = 0;
    defs[2].gy_id = -1;
    defs[2].gz_id = -1;
    defs[2].adc_id = -1;

    memset(table, 0, sizeof(table));

    table[0].id = 0;
    table[0].duration_us = -1;
    table[1].id = 1;
    table[1].duration_us = -1;
    table[2].id = 0;
    table[2].duration_us = -1;
    table[3].id = 1;
    table[3].duration_us = -1;
    table[4].id = 0;
    table[4].duration_us = -1;
    table[5].id = 2;
    table[5].duration_us = -1;

    desc.num_unique_blocks = 3;
    desc.base_blocks = defs;
    desc.num_blocks = 6;
    desc.pass_len = 6;
    desc.block_table = table;
    desc.num_prep_blocks = 0;
    desc.num_cooldown_blocks = 0;

    return pulseg__get_tr_in_sequence(&desc, &diag);
}

MU_TEST(test_rf_periodic_ok)
{
    run_consistency_check("01_rfamp_ok_mrfingerprinting.seq", PULSEG_SUCCESS);
}

/* 02_rfamp_fail_vfa.seq: true VFA -- the same (rf90, gx) TR shape replayed
 * with a different RF amplitude on every repeat. Default opts
 * (allow_variable_rf_amplitude=1) now ACCEPT this: pulseg_check_consistency
 * always re-checks permissively (matches PULSEG_OPTS_INIT default -- see
 * its doc comment), and check_rf_amplitude_periodicity only raises
 * desc->rf_amplitude_variable, it does not error, when allow_variable is
 * set. The strict rejection is still reachable with the flag off, see
 * test_rf_periodic_fail_flag_off below. */
MU_TEST(test_rf_periodic_fail)
{
    run_consistency_check("02_rfamp_fail_vfa.seq", PULSEG_SUCCESS);
}

/* Same fixture, flag off: PULSEG_ERR_CONSISTENCY_RF_PERIODIC still fires.
 * Load must happen with allow_variable_rf_amplitude=0 (pulseg_convert_collection
 * time), not just pulseg_check_consistency (which is always permissive). */
MU_TEST(test_rf_periodic_fail_flag_off)
{
    pulseg_collection *coll = NULL;
    int rc;

    s_rf_opts.allow_variable_rf_amplitude = 0;
    rc = load_seq(&coll, "02_rfamp_fail_vfa.seq", &s_rf_opts);
    mu_assert_int_eq(PULSEG_ERR_CONSISTENCY_RF_PERIODIC, rc);
}

MU_TEST(test_rfshim_periodic_ok)
{
    run_consistency_check("03_rfshim_ok_pnpmrfingerprinting.seq", PULSEG_SUCCESS);
}

MU_TEST(test_rfshim_periodic_fail)
{
    run_consistency_check("04_rfshim_fail_gre.seq", PULSEG_ERR_CONSISTENCY_RF_SHIM_PERIODIC);
}

MU_TEST(test_error_code_partition_structural_vs_rf)
{
    int rc;

    rc = run_structural_tr_mismatch_probe();
    mu_assert_int_eq(PULSEG_ERR_TR_PATTERN_MISMATCH, rc);

    run_consistency_check("02_rfamp_fail_vfa.seq", PULSEG_SUCCESS);

    run_consistency_check("04_rfshim_fail_gre.seq", PULSEG_ERR_CONSISTENCY_RF_SHIM_PERIODIC);
}

MU_TEST_SUITE(suite_rf_consistency)
{
    MU_SUITE_CONFIGURE(rf_consistency_setup, NULL);
    MU_RUN_TEST(test_rf_periodic_ok);
    MU_RUN_TEST(test_rf_periodic_fail);
    MU_RUN_TEST(test_rf_periodic_fail_flag_off);
    MU_RUN_TEST(test_rfshim_periodic_ok);
    MU_RUN_TEST(test_rfshim_periodic_fail);
    MU_RUN_TEST(test_error_code_partition_structural_vs_rf);
}

/* ================================================================== */
/*  Suite C — Canonical full-pass RF periodicity                      */
/* ================================================================== */

/* test case 06: two-pass sequence where pass-2 uses a different RF
 * amplitude than pass-1. Default opts accept it (rf_amplitude_variable is
 * raised, no error) -- same relaxation as suite_rf_consistency's VFA case,
 * cross-pass variant (check_cross_pass_rf_consistency). */
MU_TEST(test_rf_multipass_variable_structure)
{
    run_consistency_check(
        "06_rfprep_fail_multipass_variable.seq",
        PULSEG_SUCCESS);
}

MU_TEST(test_rf_multipass_variable_flag_off)
{
    pulseg_collection *coll = NULL;
    int rc;

    s_rf_opts.allow_variable_rf_amplitude = 0;
    rc = load_seq(&coll, "06_rfprep_fail_multipass_variable.seq", &s_rf_opts);
    mu_assert_int_eq(PULSEG_ERR_CONSISTENCY_RF_PERIODIC, rc);
}

/* pulseg_get_rf_array over the accepted cross-pass-variable case: the
 * worst-B1rms pass (pass 2, 0.45*pi > pass 1's 0.30*pi -- uniform within
 * each pass, so B1rms ranking reduces to plain amplitude here) supplies the
 * REAL act_amplitude_hz; peak_amplitude_hz is the positional max, which
 * coincides with it since amplitude is uniform within each pass. */
MU_TEST(test_rf_multipass_variable_array_uses_worst_pass)
{
    pulseg_collection *coll = NULL;
    pulseg_rf_stats *pulses = NULL;
    int rc, npulses, i;

    rc = load_seq(&coll, "06_rfprep_fail_multipass_variable.seq", &s_rf_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

    npulses = pulseg_get_rf_array(coll, &pulses, 0);
    mu_assert(npulses > 0, "expected RF pulses");

    for (i = 0; i < npulses; ++i)
    {
        mu_assert_float_near("act_amplitude_hz (pass-2 winner)", 225.0f, pulses[i].act_amplitude_hz, 1.0f);
        mu_assert_float_near("peak_amplitude_hz (uniform-per-pass envelope)", 225.0f, pulses[i].peak_amplitude_hz, 1.0f);
    }

    free(pulses);
    pulseg_collection_free(coll);
}

MU_TEST_SUITE(suite_rf_canonical_periodicity)
{
    MU_SUITE_CONFIGURE(rf_consistency_setup, NULL);
    MU_RUN_TEST(test_rf_multipass_variable_structure);
    MU_RUN_TEST(test_rf_multipass_variable_flag_off);
    MU_RUN_TEST(test_rf_multipass_variable_array_uses_worst_pass);
}

/* ================================================================== */
/*  Suite D — 8-channel CP quadrature target                          */
/* ================================================================== */

/* 8-channel CP shim with per-channel weight 1/sqrt(8) should yield
 * the same RF stats as the single-channel 1 ms 180-degree baseline
 * when multichannel RF is reduced to an effective waveform via
 * quadrature aggregation before compute_rf_stats(). */
MU_TEST(test_cp_8ch_matches_1ch_180deg)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    pulseg_rf_stats stats8 = PULSEG_RF_STATS_INIT;
    int rc;

    default_opts_init(&opts);
    rc = load_seq(&coll, "07_rfstat_cp_8ch_180.seq", &opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed for 8ch CP case");

    rc = pulseg_get_rf_stats(coll, &stats8, 0, 0);
    mu_assert(PULSEG_SUCCEEDED(rc), "get_rf_stats failed for 8ch CP");

    /* Under quadrature (RSS) aggregation across 8 channels each at
     * 500/sqrt(8) Hz, the combined base amplitude must match the
     * single-channel 1 ms 180-degree reference (500 Hz). */
    mu_assert_float_near("8ch CP base_amp_hz", 500.0f, stats8.base_amplitude_hz, 5.0f);
    mu_assert_float_near("8ch CP flip_angle", (float)M_PI, stats8.flip_angle_rad, 0.01f);
    mu_assert_float_near("8ch CP duration_us", 999.0f, stats8.duration_us, 2.0f);

    pulseg_collection_free(coll);
}

MU_TEST_SUITE(suite_rf_cp_8ch)
{
    MU_RUN_TEST(test_cp_8ch_matches_1ch_180deg);
}

/* ================================================================== */
/*  Suite E — worst-B1rms real instance vs positional-max envelope    */
/* ================================================================== */

/*
 * Synthetic in-memory descriptor (bypasses .seq parsing, same technique as
 * run_structural_tr_mismatch_probe above), built to make the worst-B1rms
 * instance diverge from the positional-max envelope at one position --
 * something no checked-in .seq fixture exercises (they're all either
 * single-pulse-per-TR VFA or uniform-within-instance trains, where the two
 * coincide by construction).
 *
 * Two 2-pulse TR instances, positions 0 and 1:
 *   instance A: amp = (10, 1)   B1rms^2 numerator = 10^2*1.0 + 1^2*0.5 = 100.5
 *   instance B: amp = (1, 10)   B1rms^2 numerator = 1^2*1.0 + 10^2*0.5 =  51.0
 * (position 0's pulse shape has total_b1sq_power=1.0, position 1's = 0.5)
 *
 * Instance A wins on B1rms despite instance B holding the larger amplitude
 * at position 1. Expected pulseg_get_rf_array output:
 *   act_amplitude_hz  (worst-B1rms real instance A) = {10, 1}
 *   peak_amplitude_hz (positional max across A, B)   = {10, 10}
 * -- act_amplitude_hz[1] != peak_amplitude_hz[1] is the point of the test.
 */
static int build_divergent_variable_rf_collection(pulseg_collection *coll, pulseg_sequence_descriptor *desc)
{
    static pulseg_rf_definition rf_defs[2];
    static pulseg_rf_table_element rf_table[4];
    static pulseg_block_table_element blocks[4];
    int i;

    for (i = 0; i < 2; ++i)
    {
        rf_defs[i].id = i;
        rf_defs[i].mag_shape_id = -1;
        rf_defs[i].phase_shape_id = -1;
        rf_defs[i].time_shape_id = -1;
        rf_defs[i].delay = 0;
        rf_defs[i].num_channels = 1;
        {
            pulseg_rf_stats s = PULSEG_RF_STATS_INIT;
            s.total_b1sq_power = (i == 0) ? 1.0f : 0.5f;
            rf_defs[i].stats = s;
        }
    }

    /* instance A: positions 0,1 -> rf_table[0,1] */
    rf_table[0].id = 0;
    rf_table[0].amplitude = 10.0f;
    rf_table[0].freq_offset = 0.0f;
    rf_table[0].phase_offset = 0.0f;
    rf_table[0].rf_use = 0;
    rf_table[1].id = 1;
    rf_table[1].amplitude = 1.0f;
    rf_table[1].freq_offset = 0.0f;
    rf_table[1].phase_offset = 0.0f;
    rf_table[1].rf_use = 0;
    /* instance B: positions 0,1 -> rf_table[2,3] */
    rf_table[2].id = 0;
    rf_table[2].amplitude = 1.0f;
    rf_table[2].freq_offset = 0.0f;
    rf_table[2].phase_offset = 0.0f;
    rf_table[2].rf_use = 0;
    rf_table[3].id = 1;
    rf_table[3].amplitude = 10.0f;
    rf_table[3].freq_offset = 0.0f;
    rf_table[3].phase_offset = 0.0f;
    rf_table[3].rf_use = 0;

    memset(blocks, 0, sizeof(blocks));
    for (i = 0; i < 4; ++i)
    {
        blocks[i].id = i;
        blocks[i].duration_us = -1;
        blocks[i].rf_id = i;
        blocks[i].gx_id = -1;
        blocks[i].gy_id = -1;
        blocks[i].gz_id = -1;
        blocks[i].adc_id = -1;
        blocks[i].digitalout_id = -1;
        blocks[i].rotation_id = -1;
        blocks[i].rf_shim_id = -1;
        blocks[i].module_id = 0;
    }

    *desc = (pulseg_sequence_descriptor)PULSEG_SEQUENCE_DESCRIPTOR_INIT;
    desc->num_unique_rfs = 2;
    desc->rf_definitions = rf_defs;
    desc->rf_table_size = 4;
    desc->rf_table = rf_table;
    desc->num_blocks = 4;
    desc->block_table = blocks;
    desc->rf_amplitude_variable = 1;
    desc->num_averages = 1;
    desc->tr_descriptor.tr_size = 2;
    desc->tr_descriptor.num_trs = 2;
    desc->tr_descriptor.num_prep_blocks = 0;
    desc->tr_descriptor.num_cooldown_blocks = 0;
    desc->tr_descriptor.degenerate_prep = 1;
    desc->tr_descriptor.degenerate_cooldown = 1;
    desc->tr_descriptor.imaging_tr_start = 0;
    desc->tr_descriptor.num_prep_trs = 0;
    desc->tr_descriptor.num_cooldown_trs = 0;

    coll->num_subsequences = 1;
    coll->descriptors = desc;
    return 1;
}

MU_TEST(test_rf_array_worst_b1rms_diverges_from_peak_envelope)
{
    pulseg_collection coll;
    pulseg_sequence_descriptor desc;
    pulseg_rf_stats *pulses = NULL;
    int npulses;

    memset(&coll, 0, sizeof(coll));
    build_divergent_variable_rf_collection(&coll, &desc);

    npulses = pulseg_get_rf_array(&coll, &pulses, 0);
    mu_assert_int_eq(2, npulses);

    /* act_amplitude_hz: real amplitudes of the winning (worst-B1rms)
     * instance A -- never a synthetic per-position amalgam. */
    mu_assert_float_near("act_amplitude_hz[0] (instance A)", 10.0f, pulses[0].act_amplitude_hz, 1e-4f);
    mu_assert_float_near("act_amplitude_hz[1] (instance A)", 1.0f, pulses[1].act_amplitude_hz, 1e-4f);

    /* peak_amplitude_hz: positional max across A and B -- dominates both
     * instances at every position, including position 1 where B (not the
     * B1rms winner) holds the larger amplitude. */
    mu_assert_float_near("peak_amplitude_hz[0] (max(A,B))", 10.0f, pulses[0].peak_amplitude_hz, 1e-4f);
    mu_assert_float_near("peak_amplitude_hz[1] (max(A,B))", 10.0f, pulses[1].peak_amplitude_hz, 1e-4f);

    free(pulses);
}

/* Same descriptor with rf_amplitude_variable=0 (periodic-path contract):
 * act_amplitude_hz and peak_amplitude_hz must both collapse to the
 * canonical (instance 0 / A) value, bit-identical to pre-existing
 * behavior. */
MU_TEST(test_rf_array_periodic_path_unaffected)
{
    pulseg_collection coll;
    pulseg_sequence_descriptor desc;
    pulseg_rf_stats *pulses = NULL;
    int npulses;

    memset(&coll, 0, sizeof(coll));
    build_divergent_variable_rf_collection(&coll, &desc);
    desc.rf_amplitude_variable = 0;

    npulses = pulseg_get_rf_array(&coll, &pulses, 0);
    mu_assert_int_eq(2, npulses);

    mu_assert_float_near("act_amplitude_hz[0] (canonical)", 10.0f, pulses[0].act_amplitude_hz, 1e-4f);
    mu_assert_float_near("act_amplitude_hz[1] (canonical)", 1.0f, pulses[1].act_amplitude_hz, 1e-4f);
    mu_assert_float_near("peak_amplitude_hz[0] == act", pulses[0].act_amplitude_hz, pulses[0].peak_amplitude_hz, 1e-6f);
    mu_assert_float_near("peak_amplitude_hz[1] == act", pulses[1].act_amplitude_hz, pulses[1].peak_amplitude_hz, 1e-6f);

    free(pulses);
}

MU_TEST_SUITE(suite_rf_variable_amplitude_arrays)
{
    MU_RUN_TEST(test_rf_array_worst_b1rms_diverges_from_peak_envelope);
    MU_RUN_TEST(test_rf_array_periodic_path_unaffected);
}

/* ================================================================== */
/*  Entry point                                                       */
/* ================================================================== */

int test_rf_stats_main(void)
{
    minunit_run = 0;
    minunit_fail = 0;
    minunit_assert = 0;
    minunit_status = 0;
    minunit_real_timer = 0;
    minunit_proc_timer = 0;

    MU_RUN_SUITE(suite_rf_stats);
    MU_RUN_SUITE(suite_rf_consistency);
    MU_RUN_SUITE(suite_rf_canonical_periodicity);
    MU_RUN_SUITE(suite_rf_cp_8ch);
    MU_RUN_SUITE(suite_rf_variable_amplitude_arrays);
    MU_REPORT();
    return MU_EXIT_CODE;
}
