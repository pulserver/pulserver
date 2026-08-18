/**
 * @file pulseq_rf.c
 * @brief The spectrum of an RF pulse.  See pulseq_rf.h.
 *
 * Small enough to say what it does in one place: resample the complex pulse
 * onto a uniform grid centred on its centre, transform it, and report the
 * outermost frequencies where the magnitude clears a fraction of its peak.
 *
 * The two fftshifts around the transform are not decoration.  The input grid
 * is centred on the pulse -- index n/2 is t = 0 -- and the transform wants the
 * origin at index 0; the output comes back with zero frequency at index 0 and
 * the frequency axis here is ascending through zero.  Shifting both ends is
 * what makes those two statements true at once, and it is what
 * `mr.calcRfBandwidth` does with its `fftshift(fft(fftshift(...)))`.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "pulseq.h"
#include "pulseq_rf.h"

#include "external_kiss_fft.h"

struct pulseq_rf_spectrum
{
    int n;
    double raster_us;
    double resolution_hz;

    kiss_fft_cfg cfg;
    kiss_fft_cpx *in;
    kiss_fft_cpx *out;

    PULSEQ_REAL *t;    /* resample grid, us, centred on zero */
    PULSEQ_REAL *freq; /* Hz, ascending through zero         */
    PULSEQ_REAL *re;
    PULSEQ_REAL *im;
};

/*
 * Linear interpolation onto a uniform target grid, zero outside the source.
 *
 * Both grids are ascending, so this walks them together rather than searching
 * per point: the target has a hundred thousand points and the source a few
 * thousand, and a binary search per point would be the whole cost of the
 * transform.
 */
static void interp_linear(
    PULSEQ_REAL *out,
    const PULSEQ_REAL *x,
    int nx,
    const PULSEQ_REAL *xp,
    const PULSEQ_REAL *fp,
    int np)
{
    int i, j = 0;

    for (i = 0; i < nx; ++i)
    {
        double xi = (double)x[i];

        if (np <= 0 || xi < (double)xp[0] || xi > (double)xp[np - 1])
        {
            out[i] = (PULSEQ_REAL)0.0;
            continue;
        }
        while (j + 1 < np - 1 && (double)xp[j + 1] < xi)
            ++j;
        {
            double x0 = (double)xp[j];
            double x1 = (double)xp[j + 1];
            double span = x1 - x0;
            double f;
            if (!(span > 0.0))
                f = (double)fp[j];
            else
                f = (double)fp[j] + ((double)fp[j + 1] - (double)fp[j]) * (xi - x0) / span;
            out[i] = (PULSEQ_REAL)f;
        }
    }
}

static void fftshift(PULSEQ_REAL *re, PULSEQ_REAL *im, int n)
{
    int i, half = n / 2, shift = (n + 1) / 2;

    for (i = 0; i < half; ++i)
    {
        PULSEQ_REAL tr = re[i];
        PULSEQ_REAL ti = im[i];
        re[i] = re[i + shift];
        im[i] = im[i + shift];
        re[i + shift] = tr;
        im[i + shift] = ti;
    }
}

static double magnitude_at(const PULSEQ_REAL *re, const PULSEQ_REAL *im, int i)
{
    return sqrt((double)re[i] * (double)re[i] + (double)im[i] * (double)im[i]);
}

/*
 * The frequency, from either end, at which the magnitude crosses @p level.
 *
 * Interpolated between the last point below the level and the first above,
 * rather than reported as whichever grid point happened to be first.  The grid
 * is the requested resolution -- 10 Hz by default -- and taking the grid point
 * always lands *inside* the true crossing at both ends, so the width comes out
 * systematically narrow: measured 0.45% low on a 5 mm slice, which is 22 um of
 * slice thickness invented by the choice of grid.  `mr.aux.findFlank`
 * interpolates for the same reason, and this is its arithmetic.
 */
static double flank(
    const PULSEQ_REAL *freq,
    const PULSEQ_REAL *re,
    const PULSEQ_REAL *im,
    int n,
    double level,
    int from_the_end)
{
    int i, step, first;

    step = from_the_end ? -1 : 1;
    first = from_the_end ? (n - 1) : 0;

    for (i = first; i >= 0 && i < n; i += step)
    {
        double here = magnitude_at(re, im, i) - level;
        double before, x0, x1;

        if (!(here > 0.0))
            continue;

        /* Nothing to interpolate against at the very end of the grid: the
         * band runs off it, and the edge is the honest answer. */
        if (i == first)
            return (double)freq[i];

        before = magnitude_at(re, im, i - step) - level;
        x0 = (double)freq[i - step];
        x1 = (double)freq[i];
        if (here == before)
            return x1;
        return (here * x0 - before * x1) / (here - before);
    }
    return 0.0;
}

int pulseq_rf_spectrum_create(
    pulseq_rf_spectrum **out,
    PULSEQ_REAL raster_us,
    PULSEQ_REAL resolution_hz)
{
    pulseq_rf_spectrum *plan;
    double raster, resolution;
    int i, n;

    if (!out)
        return PULSEQ_ERR_NULL_POINTER;
    *out = NULL;

    raster = (double)raster_us;
    resolution = (resolution_hz > (PULSEQ_REAL)0.0) ? (double)resolution_hz
                                                    : PULSEQ_RF_DEFAULT_RESOLUTION_HZ;
    if (!(raster > 0.0))
        return PULSEQ_ERR_INVALID_ARGUMENT;

    /* Points needed to resolve `resolution` over a grid of this raster: the
     * transform's bin spacing is 1 / (n * dt), so n = 1 / (resolution * dt). */
    n = (int)(1.0 / (resolution * raster * 1e-6));
    n = kiss_fft_next_fast_size(n);
    if (n < 2)
        n = 2;

    plan = (pulseq_rf_spectrum *)PULSEQ_ALLOC(sizeof(pulseq_rf_spectrum));
    if (!plan)
        return PULSEQ_ERR_ALLOC_FAILED;
    memset(plan, 0, sizeof(*plan));

    plan->n = n;
    plan->raster_us = raster;
    plan->resolution_hz = resolution;

    plan->t = (PULSEQ_REAL *)PULSEQ_ALLOC((size_t)n * sizeof(PULSEQ_REAL));
    plan->freq = (PULSEQ_REAL *)PULSEQ_ALLOC((size_t)n * sizeof(PULSEQ_REAL));
    plan->re = (PULSEQ_REAL *)PULSEQ_ALLOC((size_t)n * sizeof(PULSEQ_REAL));
    plan->im = (PULSEQ_REAL *)PULSEQ_ALLOC((size_t)n * sizeof(PULSEQ_REAL));
    plan->in = (kiss_fft_cpx *)PULSEQ_ALLOC((size_t)n * sizeof(kiss_fft_cpx));
    plan->out = (kiss_fft_cpx *)PULSEQ_ALLOC((size_t)n * sizeof(kiss_fft_cpx));
    plan->cfg = kiss_fft_alloc(n, 0, NULL, NULL);

    if (!plan->t || !plan->freq || !plan->re || !plan->im || !plan->in || !plan->out || !plan->cfg)
    {
        pulseq_rf_spectrum_free(plan);
        return PULSEQ_ERR_ALLOC_FAILED;
    }

    for (i = 0; i < n; ++i)
    {
        plan->t[i] = (PULSEQ_REAL)((double)(i - n / 2) * raster);
        plan->freq[i] = (PULSEQ_REAL)((double)(i - n / 2) * resolution);
        plan->re[i] = (PULSEQ_REAL)0.0;
        plan->im[i] = (PULSEQ_REAL)0.0;
    }

    *out = plan;
    return PULSEQ_SUCCESS;
}

void pulseq_rf_spectrum_free(pulseq_rf_spectrum *plan)
{
    if (!plan)
        return;
    PULSEQ_FREE(plan->t);
    PULSEQ_FREE(plan->freq);
    PULSEQ_FREE(plan->re);
    PULSEQ_FREE(plan->im);
    PULSEQ_FREE(plan->in);
    PULSEQ_FREE(plan->out);
    if (plan->cfg)
        kiss_fft_free(plan->cfg);
    PULSEQ_FREE(plan);
}

int pulseq_rf_spectrum_run(
    pulseq_rf_spectrum *plan,
    const PULSEQ_REAL *re,
    const PULSEQ_REAL *im,
    const PULSEQ_REAL *t_us,
    int count,
    PULSEQ_REAL center_us)
{
    PULSEQ_REAL *centered = NULL;
    int i, n;

    if (!plan || !re || !im || !t_us)
        return PULSEQ_ERR_NULL_POINTER;
    if (count <= 0)
        return PULSEQ_ERR_INVALID_ARGUMENT;

    n = plan->n;

    centered = (PULSEQ_REAL *)PULSEQ_ALLOC((size_t)count * sizeof(PULSEQ_REAL));
    if (!centered)
        return PULSEQ_ERR_ALLOC_FAILED;
    for (i = 0; i < count; ++i)
        centered[i] = (PULSEQ_REAL)((double)t_us[i] - (double)center_us);

    interp_linear(plan->re, plan->t, n, centered, re, count);
    interp_linear(plan->im, plan->t, n, centered, im, count);
    PULSEQ_FREE(centered);

    fftshift(plan->re, plan->im, n);
    for (i = 0; i < n; ++i)
    {
        plan->in[i].r = (kiss_fft_scalar)plan->re[i];
        plan->in[i].i = (kiss_fft_scalar)plan->im[i];
    }

    kiss_fft(plan->cfg, plan->in, plan->out);

    for (i = 0; i < n; ++i)
    {
        plan->re[i] = (PULSEQ_REAL)plan->out[i].r;
        plan->im[i] = (PULSEQ_REAL)plan->out[i].i;
    }
    fftshift(plan->re, plan->im, n);

    return PULSEQ_SUCCESS;
}

int pulseq_rf_spectrum_size(const pulseq_rf_spectrum *plan)
{
    return plan ? plan->n : 0;
}

const PULSEQ_REAL *pulseq_rf_spectrum_freq(const pulseq_rf_spectrum *plan)
{
    return plan ? plan->freq : NULL;
}

const PULSEQ_REAL *pulseq_rf_spectrum_re(const pulseq_rf_spectrum *plan)
{
    return plan ? plan->re : NULL;
}

const PULSEQ_REAL *pulseq_rf_spectrum_im(const pulseq_rf_spectrum *plan)
{
    return plan ? plan->im : NULL;
}

PULSEQ_REAL pulseq_rf_bandwidth(
    const pulseq_rf_spectrum *plan,
    PULSEQ_REAL cutoff,
    PULSEQ_REAL *out_center_hz)
{
    double level = 0.0;
    double peak = 0.0;
    double lo, hi;
    int i;

    if (out_center_hz)
        *out_center_hz = (PULSEQ_REAL)0.0;
    if (!plan || plan->n <= 0)
        return (PULSEQ_REAL)0.0;

    for (i = 0; i < plan->n; ++i)
    {
        double m = sqrt(
            (double)plan->re[i] * (double)plan->re[i] + (double)plan->im[i] * (double)plan->im[i]);
        if (m > peak)
            peak = m;
    }
    if (!(peak > 1.0e-12))
        return (PULSEQ_REAL)0.0;

    level = ((cutoff > (PULSEQ_REAL)0.0) ? (double)cutoff : PULSEQ_RF_DEFAULT_CUTOFF) * peak;

    lo = flank(plan->freq, plan->re, plan->im, plan->n, level, 0);
    hi = flank(plan->freq, plan->re, plan->im, plan->n, level, 1);

    if (out_center_hz)
        *out_center_hz = (PULSEQ_REAL)(0.5 * (lo + hi));
    return (PULSEQ_REAL)((hi > lo) ? (hi - lo) : 0.0);
}
