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
/*  Suite A — RF180 block pulse ground truth                          */
/* ================================================================== */

/*
 * Ground truth derived from the standard GE 1 ms hard-pulse
 * RF_PULSE struct, converted to SI / pulseqlib conventions:
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
 *   nom_bw          = 3125 Hz   -> ~3123 Hz (fallback 3.12/duration_s)
 *   num_samples     = 2 (raw decompressed: block pulse has only
 *                       start + end samples in .seq file)
 */

MU_TEST(test_rf180_block_pulse_stats)
{
    pulseqlib_opts opts;
    pulseqlib_collection* coll = NULL;
    pulseqlib_rf_stats stats = PULSEQLIB_RF_STATS_INIT;
    int rc;

    default_opts_init(&opts);
    rc = load_seq(&coll, "00_basic_rfstat.seq", &opts);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq failed");

    rc = pulseqlib_get_rf_stats(coll, &stats, 0, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "get_rf_stats failed");

    mu_assert_float_near("abs_width",  1.0f,    stats.abs_width,        1e-4f);
    mu_assert_float_near("eff_width",  1.0f,    stats.eff_width,        1e-4f);
    mu_assert_float_near("duty_cycle", 1.0f,    stats.duty_cycle,       1e-4f);
    mu_assert_float_near("max_pw",     1.0f,    stats.max_pulse_width,  1e-4f);

    mu_assert_float_near("base_amp_hz", 500.0f, stats.base_amplitude_hz, 1.0f);
    mu_assert_float_near("flip_angle",  (float)M_PI, stats.flip_angle_rad, 0.01f);

    mu_assert_float_near("duration_us", 999.0f, stats.duration_us, 2.0f);
    mu_assert(abs(stats.isodelay_us - 499) <= 2, "isodelay_us");

    mu_assert_float_near("area",        0.001f, stats.area,          1e-5f);
    mu_assert_float_near("bandwidth",   3123.0f, stats.bandwidth_hz, 50.0f);

    mu_assert_int_eq(2, stats.num_samples);

    pulseqlib_collection_free(coll);
}

MU_TEST(test_rf_array_basic_canonical_tr)
{
    pulseqlib_opts opts;
    pulseqlib_collection* coll = NULL;
    pulseqlib_rf_stats* pulses = NULL;
    int rc, npulses;

    default_opts_init(&opts);
    rc = load_seq(&coll, "00_basic_rfstat.seq", &opts);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq failed");

    npulses = pulseqlib_get_rf_array(coll, &pulses, 0);
    mu_assert_int_eq(1, npulses);
    mu_assert_int_eq(1, pulses[0].num_instances);
    mu_assert_float_near("canonical base_amp_hz",
        500.0f, pulses[0].base_amplitude_hz, 1.0f);

    free(pulses);
    pulseqlib_collection_free(coll);
}

MU_TEST(test_rf_array_nondegenerate_fullpass_expanded)
{
    pulseqlib_opts opts;
    pulseqlib_collection* coll = NULL;
    pulseqlib_rf_stats* pulses = NULL;
    int rc, npulses, i;

    default_opts_init(&opts);
    rc = load_seq_with_averages(
        &coll, "05_rfprep_ok_canonical_fullpass.seq", &opts, 3);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq_with_averages failed");

    npulses = pulseqlib_get_rf_array(coll, &pulses, 0);
    mu_assert_int_eq(8, npulses);
    mu_assert_float_near("prep act_amp_hz",
        125.0f, pulses[0].act_amplitude_hz, 1.0f);
    mu_assert_float_near("cooldown act_amp_hz",
        500.0f, pulses[npulses - 1].act_amplitude_hz, 1.0f);
    for (i = 0; i < npulses; ++i)
        mu_assert_int_eq(1, pulses[i].num_instances);

    free(pulses);
    pulseqlib_collection_free(coll);
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

static pulseqlib_opts s_rf_opts;

static void rf_consistency_setup(void)
{
    default_opts_init(&s_rf_opts);
}

static void run_consistency_check(const char* filename, int expected_code)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic diag = PULSEQLIB_DIAGNOSTIC_INIT;
    int rc;

    rc = load_seq(&coll, filename, &s_rf_opts);
    if (PULSEQLIB_FAILED(rc)) {
        /* Load failed — only acceptable if we expected this error */
        if (expected_code > 0)
            mu_fail("load_seq failed unexpectedly");
        mu_assert_int_eq(expected_code, rc);
        return;
    }

    rc = pulseqlib_check_consistency(coll, &diag);

    if (expected_code > 0) {
        mu_assert(PULSEQLIB_SUCCEEDED(rc), "expected consistency pass");
    } else {
        mu_assert_int_eq(expected_code, rc);
    }

    pulseqlib_collection_free(coll);
}

/* Build a synthetic descriptor in-memory to trigger a structural
 * TR pattern mismatch (-103) without involving RF periodicity logic.
 * Pattern is A,B,A,B,A,C with long active duration (>15 s) so
 * single-TR fallback is not allowed. */
static int run_structural_tr_mismatch_probe(void)
{
    pulseqlib_sequence_descriptor desc = PULSEQLIB_SEQUENCE_DESCRIPTOR_INIT;
    pulseqlib_diagnostic diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_block_definition defs[3];
    pulseqlib_block_table_element table[6];

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

    table[0].id = 0; table[0].duration_us = -1;
    table[1].id = 1; table[1].duration_us = -1;
    table[2].id = 0; table[2].duration_us = -1;
    table[3].id = 1; table[3].duration_us = -1;
    table[4].id = 0; table[4].duration_us = -1;
    table[5].id = 2; table[5].duration_us = -1;

    desc.num_unique_blocks = 3;
    desc.block_definitions = defs;
    desc.num_blocks = 6;
    desc.pass_len = 6;
    desc.block_table = table;
    desc.num_prep_blocks = 0;
    desc.num_cooldown_blocks = 0;

    return pulseqlib__get_tr_in_sequence(&desc, &diag);
}

MU_TEST(test_rf_periodic_ok)
{
    run_consistency_check("01_rfamp_ok_mrfingerprinting.seq",
                          PULSEQLIB_SUCCESS);
}

MU_TEST(test_rf_periodic_fail)
{
    run_consistency_check("02_rfamp_fail_vfa.seq",
                          PULSEQLIB_ERR_CONSISTENCY_RF_PERIODIC);
}

MU_TEST(test_rfshim_periodic_ok)
{
    run_consistency_check("03_rfshim_ok_pnpmrfingerprinting.seq",
                          PULSEQLIB_SUCCESS);
}

MU_TEST(test_rfshim_periodic_fail)
{
    run_consistency_check("04_rfshim_fail_gre.seq",
                          PULSEQLIB_ERR_CONSISTENCY_RF_SHIM_PERIODIC);
}

MU_TEST(test_error_code_partition_structural_vs_rf)
{
    int rc;

    rc = run_structural_tr_mismatch_probe();
    mu_assert_int_eq(PULSEQLIB_ERR_TR_PATTERN_MISMATCH, rc);

    run_consistency_check("02_rfamp_fail_vfa.seq",
                          PULSEQLIB_ERR_CONSISTENCY_RF_PERIODIC);

    run_consistency_check("04_rfshim_fail_gre.seq",
                          PULSEQLIB_ERR_CONSISTENCY_RF_SHIM_PERIODIC);
}

MU_TEST_SUITE(suite_rf_consistency)
{
    MU_SUITE_CONFIGURE(rf_consistency_setup, NULL);
    MU_RUN_TEST(test_rf_periodic_ok);
    MU_RUN_TEST(test_rf_periodic_fail);
    MU_RUN_TEST(test_rfshim_periodic_ok);
    MU_RUN_TEST(test_rfshim_periodic_fail);
    MU_RUN_TEST(test_error_code_partition_structural_vs_rf);
}

/* ================================================================== */
/*  Suite C — Canonical full-pass RF periodicity                      */
/* ================================================================== */

/* test case 06: two-pass sequence where pass-2 uses a different RF
 * amplitude than pass-1.  The canonical full-pass RF consistency check
 * fires when pulseqlib_check_consistency() is called. */
MU_TEST(test_rf_multipass_variable_structure)
{
    run_consistency_check("06_rfprep_fail_multipass_variable.seq",
                          PULSEQLIB_ERR_CONSISTENCY_RF_PERIODIC);
}

MU_TEST_SUITE(suite_rf_canonical_periodicity)
{
    MU_SUITE_CONFIGURE(rf_consistency_setup, NULL);
    MU_RUN_TEST(test_rf_multipass_variable_structure);
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
    pulseqlib_opts opts;
    pulseqlib_collection* coll = NULL;
    pulseqlib_rf_stats stats8 = PULSEQLIB_RF_STATS_INIT;
    int rc;

    default_opts_init(&opts);
    rc = load_seq(&coll, "07_rfstat_cp_8ch_180.seq", &opts);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load_seq failed for 8ch CP case");

    rc = pulseqlib_get_rf_stats(coll, &stats8, 0, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "get_rf_stats failed for 8ch CP");

    /* Under quadrature (RSS) aggregation across 8 channels each at
     * 500/sqrt(8) Hz, the combined base amplitude must match the
     * single-channel 1 ms 180-degree reference (500 Hz). */
    mu_assert_float_near("8ch CP base_amp_hz",
        500.0f, stats8.base_amplitude_hz, 5.0f);
    mu_assert_float_near("8ch CP flip_angle",
        (float)M_PI, stats8.flip_angle_rad, 0.01f);
    mu_assert_float_near("8ch CP duration_us",
        999.0f, stats8.duration_us, 2.0f);

    pulseqlib_collection_free(coll);
}

MU_TEST_SUITE(suite_rf_cp_8ch)
{
    MU_RUN_TEST(test_cp_8ch_matches_1ch_180deg);
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
    MU_REPORT();
    return MU_EXIT_CODE;
}
