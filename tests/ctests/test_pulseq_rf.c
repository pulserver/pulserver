/*
 * test_pulseq_rf.c -- the RF spectrum and the bandwidth read off it.
 *
 * The oracle here is analysis rather than another implementation: the Fourier
 * transform of a rectangle and of a windowed sinc are both known in closed
 * form, so the assertions are against theory and not against whatever the code
 * happened to produce first.
 *
 * This exists because the estimator it replaced was silently not working.
 * pulseg's version resampled with a routine that *clamps* outside the source
 * range rather than zero-padding as `mr.calcRfBandwidth` does, so any pulse
 * whose envelope did not start and end at zero -- a hard pulse above all -- was
 * extended to DC across the whole window, its spectrum collapsed onto a spike
 * at zero frequency, and the measured width came back non-positive.  The
 * analytic fallback then fired and looked like an answer.  A test against
 * theory is what makes that visible; a test against the previous output is what
 * hid it.
 */

#include "test_helpers.h"

#include "pulseq.h"
#include "pulseq_rf.h"

#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define RF_RASTER_US 1.0
#define RF_SAMPLES 1000 /* 1 ms on a 1 us raster */

/* Fill a complex pulse and its time base; returns the centre in us. */
static double build_rect(PULSEQ_REAL *re, PULSEQ_REAL *im, PULSEQ_REAL *t, int n)
{
    int i;
    for (i = 0; i < n; ++i)
    {
        t[i] = (PULSEQ_REAL)((double)i * RF_RASTER_US);
        re[i] = (PULSEQ_REAL)1.0;
        im[i] = (PULSEQ_REAL)0.0;
    }
    return 0.5 * (double)(n - 1) * RF_RASTER_US;
}

/* A Hamming-windowed sinc with @p lobes zero crossings each side, so its
 * time-bandwidth product is 2 * lobes. */
static double build_sinc(PULSEQ_REAL *re, PULSEQ_REAL *im, PULSEQ_REAL *t, int n, double lobes,
                         double freq_offset_hz)
{
    int i;
    for (i = 0; i < n; ++i)
    {
        double x = ((double)i - 0.5 * (n - 1)) / (0.5 * (n - 1)) * lobes;
        double s = (fabs(x) < 1.0e-12) ? 1.0 : sin(M_PI * x) / (M_PI * x);
        double win = 0.54 + 0.46 * cos(M_PI * x / lobes);
        double phase = 2.0 * M_PI * freq_offset_hz * ((double)i * RF_RASTER_US * 1.0e-6);
        t[i] = (PULSEQ_REAL)((double)i * RF_RASTER_US);
        re[i] = (PULSEQ_REAL)(s * win * cos(phase));
        im[i] = (PULSEQ_REAL)(s * win * sin(phase));
    }
    return 0.5 * (double)(n - 1) * RF_RASTER_US;
}

/*
 * A rectangle transforms to a sinc, whose -6 dB full width is 1.2067 / T.
 *
 * The case the old estimator got wrong, and the reason it is first: a hard
 * pulse is the one whose envelope is furthest from zero at its edges.
 */
MU_TEST(test_a_rect_pulse_has_the_bandwidth_of_a_sinc)
{
    pulseq_rf_spectrum *plan = NULL;
    PULSEQ_REAL re[RF_SAMPLES], im[RF_SAMPLES], t[RF_SAMPLES];
    PULSEQ_REAL bw, fc;
    double centre, expected;
    int rc;

    rc = pulseq_rf_spectrum_create(&plan, (PULSEQ_REAL)RF_RASTER_US,
                                   (PULSEQ_REAL)PULSEQ_RF_DEFAULT_RESOLUTION_HZ);
    mu_assert(PULSEQ_SUCCEEDED(rc), "spectrum_create failed");

    centre = build_rect(re, im, t, RF_SAMPLES);
    rc = pulseq_rf_spectrum_run(plan, re, im, t, RF_SAMPLES, (PULSEQ_REAL)centre);
    mu_assert(PULSEQ_SUCCEEDED(rc), "spectrum_run failed");

    bw = pulseq_rf_bandwidth(plan, (PULSEQ_REAL)PULSEQ_RF_DEFAULT_CUTOFF, &fc);

    /* T is the sample span, (n-1) rasters, not n. */
    expected = 1.2067 / ((double)(RF_SAMPLES - 1) * RF_RASTER_US * 1.0e-6);
    /* One grid step of slack: a flank can only land on a grid point. */
    mu_assert(fabs((double)bw - expected) <= 2.0 * PULSEQ_RF_DEFAULT_RESOLUTION_HZ,
              "rect bandwidth is not 1.2067 / T");
    mu_assert(fabs((double)fc) <= PULSEQ_RF_DEFAULT_RESOLUTION_HZ,
              "a real symmetric pulse is not centred on zero");

    pulseq_rf_spectrum_free(plan);
}

/*
 * A windowed sinc's bandwidth is its time-bandwidth product over its duration.
 *
 * The relation a slice-selective pulse is designed around, and the one
 * `SliceThickness` then divides by the gradient.
 */
MU_TEST(test_a_sinc_pulse_reports_its_time_bandwidth_product)
{
    pulseq_rf_spectrum *plan = NULL;
    PULSEQ_REAL re[RF_SAMPLES], im[RF_SAMPLES], t[RF_SAMPLES];
    PULSEQ_REAL bw;
    double centre, duration_s, expected;
    double lobes[3];
    int i, rc;

    lobes[0] = 2.0;
    lobes[1] = 4.0;
    lobes[2] = 6.0;

    rc = pulseq_rf_spectrum_create(&plan, (PULSEQ_REAL)RF_RASTER_US,
                                   (PULSEQ_REAL)PULSEQ_RF_DEFAULT_RESOLUTION_HZ);
    mu_assert(PULSEQ_SUCCEEDED(rc), "spectrum_create failed");

    duration_s = (double)(RF_SAMPLES - 1) * RF_RASTER_US * 1.0e-6;
    for (i = 0; i < 3; ++i)
    {
        centre = build_sinc(re, im, t, RF_SAMPLES, lobes[i], 0.0);
        rc = pulseq_rf_spectrum_run(plan, re, im, t, RF_SAMPLES, (PULSEQ_REAL)centre);
        mu_assert(PULSEQ_SUCCEEDED(rc), "spectrum_run failed");

        bw = pulseq_rf_bandwidth(plan, (PULSEQ_REAL)PULSEQ_RF_DEFAULT_CUTOFF, NULL);
        expected = 2.0 * lobes[i] / duration_s;
        /* 3% covers the windowing, which broadens the passband slightly. */
        mu_assert(fabs((double)bw - expected) <= 0.03 * expected,
                  "sinc bandwidth is not TBW / duration");
    }

    pulseq_rf_spectrum_free(plan);
}

/*
 * A frequency offset moves the band without widening it.
 *
 * Worth pinning because the two are read off the same pair of flanks: a sign
 * error in the transform's shift would move both the same way and change the
 * width, and a symmetric pulse would never show it.
 */
MU_TEST(test_a_frequency_offset_moves_the_band_and_not_its_width)
{
    pulseq_rf_spectrum *plan = NULL;
    PULSEQ_REAL re[RF_SAMPLES], im[RF_SAMPLES], t[RF_SAMPLES];
    PULSEQ_REAL bw_at_zero, bw_shifted, fc;
    double centre;
    int rc;

    rc = pulseq_rf_spectrum_create(&plan, (PULSEQ_REAL)RF_RASTER_US,
                                   (PULSEQ_REAL)PULSEQ_RF_DEFAULT_RESOLUTION_HZ);
    mu_assert(PULSEQ_SUCCEEDED(rc), "spectrum_create failed");

    centre = build_sinc(re, im, t, RF_SAMPLES, 4.0, 0.0);
    pulseq_rf_spectrum_run(plan, re, im, t, RF_SAMPLES, (PULSEQ_REAL)centre);
    bw_at_zero = pulseq_rf_bandwidth(plan, (PULSEQ_REAL)0.5, NULL);

    centre = build_sinc(re, im, t, RF_SAMPLES, 4.0, 1500.0);
    pulseq_rf_spectrum_run(plan, re, im, t, RF_SAMPLES, (PULSEQ_REAL)centre);
    bw_shifted = pulseq_rf_bandwidth(plan, (PULSEQ_REAL)0.5, &fc);

    mu_assert(fabs((double)bw_shifted - (double)bw_at_zero) <=
                  2.0 * PULSEQ_RF_DEFAULT_RESOLUTION_HZ,
              "a frequency offset changed the bandwidth");
    mu_assert(fabs((double)fc - 1500.0) <= 2.0 * PULSEQ_RF_DEFAULT_RESOLUTION_HZ,
              "the centre frequency does not follow the offset");

    pulseq_rf_spectrum_free(plan);
}

/*
 * Samples outside the window contribute nothing.
 *
 * This is the zero-padding the old path did not do.  A pulse placed entirely
 * outside the grid has no spectrum at all, and the answer is zero -- not a
 * band smeared out of a clamped endpoint.
 */
MU_TEST(test_the_pulse_is_zero_padded_not_held)
{
    pulseq_rf_spectrum *plan = NULL;
    PULSEQ_REAL re[RF_SAMPLES], im[RF_SAMPLES], t[RF_SAMPLES];
    PULSEQ_REAL bw;
    int rc, n;
    double window_us;

    rc = pulseq_rf_spectrum_create(&plan, (PULSEQ_REAL)RF_RASTER_US,
                                   (PULSEQ_REAL)PULSEQ_RF_DEFAULT_RESOLUTION_HZ);
    mu_assert(PULSEQ_SUCCEEDED(rc), "spectrum_create failed");

    n = pulseq_rf_spectrum_size(plan);
    window_us = 0.5 * (double)n * RF_RASTER_US;

    /* A rect, but centred far beyond the end of the grid. */
    build_rect(re, im, t, RF_SAMPLES);
    rc = pulseq_rf_spectrum_run(plan, re, im, t, RF_SAMPLES,
                                (PULSEQ_REAL)(-10.0 * window_us));
    mu_assert(PULSEQ_SUCCEEDED(rc), "spectrum_run failed");

    bw = pulseq_rf_bandwidth(plan, (PULSEQ_REAL)0.5, NULL);
    mu_assert((double)bw == 0.0, "a pulse outside the window still produced a band");

    pulseq_rf_spectrum_free(plan);
}

/* Bad arguments are refused rather than guessed at. */
MU_TEST(test_rf_bad_arguments_are_refused)
{
    pulseq_rf_spectrum *plan = NULL;
    PULSEQ_REAL re[4], im[4], t[4];
    int i;

    mu_assert(PULSEQ_FAILED(pulseq_rf_spectrum_create(NULL, (PULSEQ_REAL)1.0, (PULSEQ_REAL)10.0)),
              "a NULL out pointer was accepted");
    mu_assert(PULSEQ_FAILED(pulseq_rf_spectrum_create(&plan, (PULSEQ_REAL)0.0, (PULSEQ_REAL)10.0)),
              "a zero raster was accepted");

    mu_assert(PULSEQ_SUCCEEDED(pulseq_rf_spectrum_create(&plan, (PULSEQ_REAL)1.0,
                                                         (PULSEQ_REAL)10.0)),
              "spectrum_create failed");
    for (i = 0; i < 4; ++i)
    {
        re[i] = (PULSEQ_REAL)1.0;
        im[i] = (PULSEQ_REAL)0.0;
        t[i] = (PULSEQ_REAL)i;
    }
    mu_assert(PULSEQ_FAILED(pulseq_rf_spectrum_run(plan, NULL, im, t, 4, (PULSEQ_REAL)0.0)),
              "a NULL waveform was accepted");
    mu_assert(PULSEQ_FAILED(pulseq_rf_spectrum_run(plan, re, im, t, 0, (PULSEQ_REAL)0.0)),
              "an empty pulse was accepted");

    /* And the accessors tolerate a NULL plan rather than dereferencing it. */
    mu_assert_int_eq(0, pulseq_rf_spectrum_size(NULL));
    mu_assert(pulseq_rf_spectrum_freq(NULL) == NULL, "freq(NULL) is not NULL");
    mu_assert((double)pulseq_rf_bandwidth(NULL, (PULSEQ_REAL)0.5, NULL) == 0.0,
              "bandwidth(NULL) is not zero");

    pulseq_rf_spectrum_free(plan);
    pulseq_rf_spectrum_free(NULL);
}

MU_TEST_SUITE(pulseq_rf_suite)
{
    MU_RUN_TEST(test_a_rect_pulse_has_the_bandwidth_of_a_sinc);
    MU_RUN_TEST(test_a_sinc_pulse_reports_its_time_bandwidth_product);
    MU_RUN_TEST(test_a_frequency_offset_moves_the_band_and_not_its_width);
    MU_RUN_TEST(test_the_pulse_is_zero_padded_not_held);
    MU_RUN_TEST(test_rf_bad_arguments_are_refused);
}

int test_pulseq_rf_main(void)
{
    MU_RUN_SUITE(pulseq_rf_suite);
    MU_REPORT();
    return minunit_fail;
}
