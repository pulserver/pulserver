/*
 * test_ptx_getters.c -- pTx VOP-SAR support getters.
 *
 * Suite: pulseg_get_rf_event_array() index-alignment with
 * pulseg_get_rf_array(), and the definition-keyed RF waveform getters.
 */
#include "test_helpers.h"

/* ================================================================== */
/*  Suite — pTx getters                                                */
/* ================================================================== */

MU_TEST(test_event_array_alignment)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    pulseg_rf_stats *stats = NULL;
    pulseg_rf_event *events = NULL;
    pulseg_subseq_info info = PULSEG_SUBSEQ_INFO_INIT;
    int rc, nstats, nevents, i;

    default_opts_init(&opts);
    rc = load_seq(&coll, "00_basic_rfstat.seq", &opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

    rc = pulseg_get_subseq_info(coll, &info, 0);
    mu_assert(PULSEG_SUCCEEDED(rc), "get_subseq_info failed");

    nstats = pulseg_get_rf_array(coll, &stats, 0);
    nevents = pulseg_get_rf_event_array(coll, &events, 0);

    mu_assert(nstats >= 0, "pulseg_get_rf_array failed");
    mu_assert_int_eq(nstats, nevents);

    for (i = 0; i < nstats; ++i)
    {
        mu_assert_float_near(
            "events[i].amplitude_hz == stats[i].act_amplitude_hz",
            stats[i].act_amplitude_hz,
            events[i].amplitude_hz,
            1e-6f);
        mu_assert(
            events[i].rf_def_id >= 0 && events[i].rf_def_id < info.num_unique_rf,
            "events[i].rf_def_id out of range");
    }

    free(stats);
    PULSEG_FREE(events);
    pulseg_collection_free(coll);
}

MU_TEST(test_def_waveform_getters)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    pulseg_rf_stats *stats = NULL;
    pulseg_rf_event *events = NULL;
    float **mag = NULL;
    int rc, nstats, nevents, nch, npts, s;

    default_opts_init(&opts);
    rc = load_seq(&coll, "00_basic_rfstat.seq", &opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

    nstats = pulseg_get_rf_array(coll, &stats, 0);
    nevents = pulseg_get_rf_event_array(coll, &events, 0);
    mu_assert(nstats > 0, "expected at least one RF event");
    mu_assert_int_eq(nstats, nevents);

    mag = pulseg_get_rf_def_magnitude(coll, &nch, &npts, 0, events[0].rf_def_id);
    mu_assert(mag != NULL, "pulseg_get_rf_def_magnitude returned NULL");
    mu_assert_int_eq(1, nch); /* fixture is single-channel */
    mu_assert_int_eq(stats[0].num_samples, npts);

    for (s = 0; s < npts; ++s)
    {
        mu_assert(mag[0][s] >= -1.0f && mag[0][s] <= 1.0f, "magnitude sample out of [-1,1]");
    }

    PULSEG_FREE(mag[0]);
    PULSEG_FREE(mag);
    free(stats);
    PULSEG_FREE(events);
    pulseg_collection_free(coll);
}

MU_TEST_SUITE(suite_ptx_getters)
{
    MU_RUN_TEST(test_event_array_alignment);
    MU_RUN_TEST(test_def_waveform_getters);
}

/* ================================================================== */
/*  Entry point                                                       */
/* ================================================================== */

int test_ptx_getters_main(void)
{
    minunit_run = 0;
    minunit_fail = 0;
    minunit_assert = 0;
    minunit_status = 0;
    minunit_real_timer = 0;
    minunit_proc_timer = 0;

    MU_RUN_SUITE(suite_ptx_getters);
    MU_REPORT();
    return MU_EXIT_CODE;
}
