/*
 * test_rf_stats.c -- RF statistics tests.
 *
 * Tests:
 *   1. get_num_unique_rf returns correct count.
 *   2. get_rf_stats returns plausible values (flip angle, bandwidth,
 *      max_amplitude, duration, etc.) for known pulse shapes.
 *   3. get_rf_base_amplitude_hz matches expected per-event amplitude.
 *   4. get_tr_rf_ids returns correct mapping (blocks without RF get -1).
 *   5. Multi-channel RF: get_rf_num_channels returns correct count.
 *   6. Multi-channel RF: CP-mode combination yields correct flip angle
 *      (N channels with unit magnitude -> flip_angle * N factor).
 *
 * Requires:
 *   - expected_output/gre_2d.seq      (single-channel RF, known flip)
 *   - expected_output/epi_2d.seq      (single-channel, different flip)
 *   - expected_output/ptx_2ch.seq     (2-channel RF, known flip)
 *
 * Until generated, only stubs are provided.
 */
#include "test_helpers.h"

/* ------------------------------------------------------------------ */
/*  Stub: single-channel RF stats for GRE                             */
/* ------------------------------------------------------------------ */

/*
 * TODO: uncomment once gre_2d.seq is generated.
 *
 * MU_TEST(test_rf_stats_gre_single_channel)
 * {
 *     pulseqlib_collection* coll = NULL;
 *     pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
 *     pulseqlib_rf_stats    stats = PULSEQLIB_RF_STATS_INIT;
 *     int rc, nrf;
 *
 *     rc = load_seq("expected_output/gre_2d.seq", &coll, &diag, 0);
 *     mu_assert(PULSEQLIB_SUCCEEDED(rc), "load GRE");
 *
 *     nrf = pulseqlib_get_num_unique_rf(coll, 0);
 *     mu_assert(nrf >= 1, "GRE should have at least 1 unique RF");
 *
 *     rc = pulseqlib_get_rf_stats(coll, &stats, 0, 0);
 *     mu_assert(PULSEQLIB_SUCCEEDED(rc), "get_rf_stats OK");
 *
 *     // Known GRE flip angle (e.g. 15 deg)
 *     mu_assert(fabs(stats.flip_angle_deg - 15.0f) < 1.0f,
 *               "flip angle should be ~15 deg");
 *     mu_assert(stats.duration_us > 0.0f, "duration > 0");
 *     mu_assert(stats.max_amplitude_hz > 0.0f, "max amp > 0");
 *     mu_assert(stats.bandwidth_hz > 0.0f, "bandwidth > 0");
 *     mu_assert(stats.num_samples > 0, "num_samples > 0");
 *
 *     pulseqlib_collection_free(coll);
 * }
 */

/* ------------------------------------------------------------------ */
/*  Stub: multi-channel RF                                            */
/* ------------------------------------------------------------------ */

/*
 * TODO: uncomment once ptx_2ch.seq is generated.
 *
 * MU_TEST(test_rf_stats_multichannel)
 * {
 *     pulseqlib_collection* coll = NULL;
 *     pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
 *     pulseqlib_rf_stats    stats = PULSEQLIB_RF_STATS_INIT;
 *     int rc, nch;
 *     float** mag;
 *     int num_channels, num_samples;
 *
 *     rc = load_seq("expected_output/ptx_2ch.seq", &coll, &diag, 0);
 *     mu_assert(PULSEQLIB_SUCCEEDED(rc), "load pTx");
 *
 *     // Check channel count from first RF block
 *     nch = pulseqlib_get_rf_num_channels(coll, 0, 0);
 *     mu_assert_int_eq(2, nch);
 *
 *     // Magnitude should return 2 rows
 *     mag = pulseqlib_get_rf_magnitude(coll, 0, 0,
 *                                      &num_channels, &num_samples);
 *     mu_assert(mag != NULL, "magnitude should be non-NULL");
 *     mu_assert_int_eq(2, num_channels);
 *     mu_assert(num_samples > 0, "per-channel samples > 0");
 *
 *     { int ch; for (ch = 0; ch < num_channels; ch++) PULSEQLIB_FREE(mag[ch]); }
 *     PULSEQLIB_FREE(mag);
 *
 *     // RF stats: CP-mode combined flip angle
 *     rc = pulseqlib_get_rf_stats(coll, &stats, 0, 0);
 *     mu_assert(PULSEQLIB_SUCCEEDED(rc), "RF stats OK");
 *     mu_assert(stats.flip_angle_deg > 0.0f, "flip > 0");
 *
 *     pulseqlib_collection_free(coll);
 * }
 */

/* ------------------------------------------------------------------ */
/*  Stub: TR RF IDs mapping                                           */
/* ------------------------------------------------------------------ */

/*
 * TODO: uncomment once gre_2d.seq is generated.
 *
 * MU_TEST(test_tr_rf_ids)
 * {
 *     pulseqlib_collection* coll = NULL;
 *     pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
 *     int rc, tr_size;
 *     int* ids;
 *
 *     rc = load_seq("expected_output/gre_2d.seq", &coll, &diag, 0);
 *     mu_assert(PULSEQLIB_SUCCEEDED(rc), "load GRE");
 *
 *     tr_size = pulseqlib_get_tr_size(coll, 0);
 *     ids = (int*)malloc(tr_size * sizeof(int));
 *     mu_assert(ids != NULL, "alloc");
 *
 *     rc = pulseqlib_get_tr_rf_ids(coll, ids, 0);
 *     mu_assert(rc == tr_size, "get_tr_rf_ids returns tr_size");
 *
 *     // Exactly one block should have RF (rf_id >= 0); rest are -1
 *     {
 *         int i, rf_count = 0;
 *         for (i = 0; i < tr_size; i++) {
 *             if (ids[i] >= 0) rf_count++;
 *         }
 *         mu_assert_int_eq(1, rf_count);
 *     }
 *
 *     free(ids);
 *     pulseqlib_collection_free(coll);
 * }
 */

/* ------------------------------------------------------------------ */
/*  Suite                                                             */
/* ------------------------------------------------------------------ */

MU_TEST_SUITE(test_rf_stats_suite)
{
    /* MU_RUN_TEST(test_rf_stats_gre_single_channel); */
    /* MU_RUN_TEST(test_rf_stats_multichannel); */
    /* MU_RUN_TEST(test_tr_rf_ids); */
}

int test_rf_stats_main(void)
{
    MU_RUN_SUITE(test_rf_stats_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
