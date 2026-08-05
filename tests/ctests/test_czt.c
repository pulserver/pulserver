/*
 * test_czt.c -- the chirp-z transform against direct evaluation.
 *
 * One question, asked at several sizes: evaluating a real-coefficient
 * polynomial at points spaced evenly in angle on the unit circle must give
 * the same answer whether it is done term by term or by Bluestein.
 *
 * The direct sum is the reference, and it is computed here in double with
 * the terms taken in the same order the transform's own coefficients are
 * stored -- so the two differ only by the summation structure, which is the
 * whole point of the comparison.
 *
 * Tolerances are relative to the l1 norm of the coefficients rather than to
 * the result, because that is what the error of either method scales with:
 * both are ~eps * (a few) * ||a||_1, and a result that sits in cancellation
 * has a large relative error under BOTH methods. The transform is checked
 * against a bound it should beat comfortably (1e-12 * ||a||_1); the point is
 * to catch a wrong chirp or a wrapped index, which miss by whole percent,
 * not to certify the last bit.
 */

#include "test_helpers.h"

#include <math.h>
#include <stdlib.h>

#include "pulseg_internal.h"

/* P_j = sum_{k<n} a[k] e^{i k (theta0 + j dtheta)}, summed term by term. */
static void czt_reference(
    double *out_re,
    double *out_im,
    const float *a,
    int n,
    double theta0,
    double dtheta,
    int m)
{
    int j, k;
    for (j = 0; j < m; ++j)
    {
        double sr = 0.0, si = 0.0;
        double th = theta0 + (double)j * dtheta;
        for (k = 0; k < n; ++k)
        {
            double ang = th * (double)k;
            sr += (double)a[k] * cos(ang);
            si += (double)a[k] * sin(ang);
        }
        out_re[j] = sr;
        out_im[j] = si;
    }
}

/* A waveform with the shape the evaluator actually meets: smooth, sign
 * changing, and with a ramp at each end. */
static void fill_waveform(float *a, int n)
{
    int k;
    if (n == 1)
    {
        a[0] = 0.75f;
        return;
    }
    for (k = 0; k < n; ++k)
    {
        double x = (double)k / (double)(n - 1);
        double env = sin(M_PI * x);
        a[k] = (float)(env * sin(2.0 * M_PI * 9.0 * x * x));
    }
}

static int check_case(int n, int m, double theta0, double dtheta, const char *what)
{
    float *a;
    double *cr, *ci, *rr, *ri;
    double l1 = 0.0, worst = 0.0;
    int k, rc, ok;

    a = (float *)malloc((size_t)n * sizeof(float));
    cr = (double *)malloc((size_t)m * sizeof(double));
    ci = (double *)malloc((size_t)m * sizeof(double));
    rr = (double *)malloc((size_t)m * sizeof(double));
    ri = (double *)malloc((size_t)m * sizeof(double));
    if (!a || !cr || !ci || !rr || !ri)
    {
        printf("  czt %s: allocation failed\n", what);
        return 0;
    }

    fill_waveform(a, n);
    for (k = 0; k < n; ++k)
        l1 += fabs((double)a[k]);

    rc = pulseg__czt_unit(cr, ci, a, n, theta0, dtheta, m);
    if (PULSEG_FAILED(rc))
    {
        printf("  czt %s: returned %d\n", what, rc);
        free(a);
        free(cr);
        free(ci);
        free(rr);
        free(ri);
        return 0;
    }

    czt_reference(rr, ri, a, n, theta0, dtheta, m);

    for (k = 0; k < m; ++k)
    {
        double dr = cr[k] - rr[k];
        double di = ci[k] - ri[k];
        double d = sqrt(dr * dr + di * di);
        if (d > worst)
            worst = d;
    }

    ok = (worst <= 1.0e-12 * l1);
    if (!ok)
        printf(
            "  czt %s: n=%d m=%d worst=%.3e l1=%.3e (%.3e relative)\n",
            what,
            n,
            m,
            worst,
            l1,
            worst / l1);

    free(a);
    free(cr);
    free(ci);
    free(rr);
    free(ri);
    return ok;
}

MU_TEST(test_czt_matches_direct_sum)
{
    /* dtheta is 2*pi*dt/T_TR in the caller, hence always tiny; theta0 is the
     * frequency the band starts at, hence not. */
    mu_assert(check_case(64, 96, 0.37, -1.0e-4, "short"), "czt: short");
    mu_assert(check_case(512, 100, 1.9, -3.0e-5, "m<n"), "czt: m below n");
    mu_assert(check_case(100, 512, -0.8, 7.0e-5, "m>n"), "czt: m above n");
    /* The size the evaluator actually meets: a long definition against a
     * whole band of candidates. */
    mu_assert(check_case(3857, 5440, 0.0754, -1.4e-6, "long"), "czt: long waveform");
}

MU_TEST(test_czt_single_point)
{
    /* m == 1 degenerates to a plain evaluation, and the wrapped negative
     * lags are then the only thing filling the kernel. */
    mu_assert(check_case(300, 1, 0.5, -1.0e-4, "m=1"), "czt: single output");
    /* n == 1 is a constant polynomial: every output is a[0]. */
    mu_assert(check_case(1, 64, 0.5, -1.0e-4, "n=1"), "czt: single coefficient");
}

MU_TEST_SUITE(suite_czt)
{
    MU_RUN_TEST(test_czt_matches_direct_sum);
    MU_RUN_TEST(test_czt_single_point);
}

int test_czt_main(void)
{
    MU_RUN_SUITE(suite_czt);
    MU_REPORT();
    return minunit_status;
}
