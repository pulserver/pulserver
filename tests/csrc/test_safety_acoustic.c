/*
 * test_safety_acoustic.c -- acoustic spectrum and forbidden-band checks.
 *
 * Tests:
 *   1. EPI echo-spacing -> acoustic peak at 1 / (2 * ESP).
 *   2. Forbidden-band violation is detected when the band straddles
 *      the peak frequency.
 *   3. No false positive when the forbidden band avoids the peak.
 *   4. Spectrogram dimensions are reasonable (num_windows > 0,
 *      num_freq_bins > 0).
 *
 * Requires:
 *   - expected_output/epi_2d.seq   (with known ESP)
 *
 * Until generated, seq1.seq is used for smoke tests.
 */
#include "test_helpers.h"

/* ------------------------------------------------------------------ */
/*  Smoke: calc_acoustic_spectra on seq1 (should not crash)           */
/* ------------------------------------------------------------------ */

MU_TEST(test_acoustic_seq1_smoke)
{
    pulseqlib_collection*    coll = NULL;
    pulseqlib_diagnostic     diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_acoustic_spectra spec = PULSEQLIB_ACOUSTIC_SPECTRA_INIT;
    pulseqlib_opts opts;
    int rc;

    rc = load_seq("expected_output/seq1.seq", &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load seq1.seq");

    test_opts_init(&opts);
    rc = pulseqlib_calc_acoustic_spectra(
            &spec, &diag, coll, 0, &opts,
            0, 0.0f, 0.0f,   /* auto window / resolution / max_freq */
            0, NULL);
    /* Even if the sequence has no meaningful gradient content,
     * the function should succeed (or return NO_WAVEFORM). */
    mu_assert(PULSEQLIB_SUCCEEDED(rc)
              || rc == PULSEQLIB_ERR_ACOUSTIC_NO_WAVEFORM,
              "acoustic spectra should succeed or return NO_WAVEFORM");

    if (PULSEQLIB_SUCCEEDED(rc)) {
        mu_assert(spec.num_freq_bins > 0, "freq bins > 0");
        pulseqlib_acoustic_spectra_free(&spec);
    }

    pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Stub: EPI peak at 1/(2*ESP)                                       */
/* ------------------------------------------------------------------ */

/*
 * TODO: uncomment once epi_2d.seq is generated.
 * The ESP (echo spacing) is known from the sequence design.
 * For a standard 2D EPI with ESP = 0.5 ms, the dominant acoustic
 * frequency is 1/(2*0.5e-3) = 1000 Hz.
 *
 * MU_TEST(test_acoustic_epi_peak)
 * {
 *     pulseqlib_collection*    coll = NULL;
 *     pulseqlib_diagnostic     diag = PULSEQLIB_DIAGNOSTIC_INIT;
 *     pulseqlib_acoustic_spectra spec = PULSEQLIB_ACOUSTIC_SPECTRA_INIT;
 *     pulseqlib_opts opts;
 *     int rc, k;
 *     float peak_freq, expected_freq;
 *     float peak_val;
 *
 *     rc = load_seq("expected_output/epi_2d.seq", &coll, &diag, 0);
 *     mu_assert(PULSEQLIB_SUCCEEDED(rc), "load EPI");
 *     test_opts_init(&opts);
 *
 *     rc = pulseqlib_calc_acoustic_spectra(
 *             &spec, &diag, coll, 0, &opts,
 *             0, 0.0f, 0.0f, 0, NULL);
 *     mu_assert(PULSEQLIB_SUCCEEDED(rc), "acoustic spectra OK");
 *
 *     // Find peak in the full-TR GX spectrum
 *     peak_val = 0.0f;
 *     peak_freq = 0.0f;
 *     for (k = 0; k < spec.num_freq_bins; k++) {
 *         float f = spec.freq_min_hz + k * spec.freq_spacing_hz;
 *         if (spec.spectrum_full_gx[k] > peak_val) {
 *             peak_val = spec.spectrum_full_gx[k];
 *             peak_freq = f;
 *         }
 *     }
 *     expected_freq = 1000.0f;  // 1 / (2 * ESP)
 *     mu_assert(fabs(peak_freq - expected_freq) < 2 * spec.freq_spacing_hz,
 *               "GX peak should be near 1/(2*ESP)");
 *
 *     pulseqlib_acoustic_spectra_free(&spec);
 *     pulseqlib_collection_free(coll);
 * }
 */

/* ------------------------------------------------------------------ */
/*  Stub: forbidden-band violation on EPI                             */
/* ------------------------------------------------------------------ */

/*
 * MU_TEST(test_acoustic_epi_forbidden_violation)
 * {
 *     pulseqlib_collection* coll = NULL;
 *     pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
 *     pulseqlib_opts opts;
 *     pulseqlib_forbidden_band band;
 *     int rc;
 *
 *     rc = load_seq("expected_output/epi_2d.seq", &coll, &diag, 0);
 *     mu_assert(PULSEQLIB_SUCCEEDED(rc), "load EPI");
 *     test_opts_init(&opts);
 *
 *     // Place a forbidden band right on the expected peak
 *     band.freq_min_hz = 950.0f;
 *     band.freq_max_hz = 1050.0f;
 *     band.max_amplitude_hz_per_m = 1.0f;  // very tight
 *
 *     rc = pulseqlib_check_safety(coll, &diag, &opts,
 *                                 1, &band, NULL, 100.0f);
 *     mu_assert(rc == PULSEQLIB_ERR_ACOUSTIC_VIOLATION,
 *               "forbidden band at peak should trigger violation");
 *
 *     pulseqlib_collection_free(coll);
 * }
 */

/* ------------------------------------------------------------------ */
/*  Stub: forbidden-band passes when band avoids peak                 */
/* ------------------------------------------------------------------ */

/*
 * MU_TEST(test_acoustic_epi_forbidden_passes)
 * {
 *     pulseqlib_collection* coll = NULL;
 *     pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
 *     pulseqlib_opts opts;
 *     pulseqlib_forbidden_band band;
 *     int rc;
 *
 *     rc = load_seq("expected_output/epi_2d.seq", &coll, &diag, 0);
 *     mu_assert(PULSEQLIB_SUCCEEDED(rc), "load EPI");
 *     test_opts_init(&opts);
 *
 *     // Place a forbidden band far away from the peak
 *     band.freq_min_hz = 50.0f;
 *     band.freq_max_hz = 100.0f;
 *     band.max_amplitude_hz_per_m = 1.0f;
 *
 *     rc = pulseqlib_check_safety(coll, &diag, &opts,
 *                                 1, &band, NULL, 100.0f);
 *     mu_assert(PULSEQLIB_SUCCEEDED(rc),
 *               "forbidden band away from peak should not trigger");
 *
 *     pulseqlib_collection_free(coll);
 * }
 */

/* ------------------------------------------------------------------ */
/*  Suite                                                             */
/* ------------------------------------------------------------------ */

MU_TEST_SUITE(test_safety_acoustic_suite)
{
    MU_RUN_TEST(test_acoustic_seq1_smoke);
    /* MU_RUN_TEST(test_acoustic_epi_peak); */
    /* MU_RUN_TEST(test_acoustic_epi_forbidden_violation); */
    /* MU_RUN_TEST(test_acoustic_epi_forbidden_passes); */
}

int test_safety_acoustic_main(void)
{
    MU_RUN_SUITE(test_safety_acoustic_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
