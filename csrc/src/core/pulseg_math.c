/**
 * @file pulseg_math.c
 * @brief Numeric helpers shared across the library.
 *
 * Integration, slew, interpolation, rotation and FFT convolution. Nothing
 * here knows about sequences; keep it that way, since the vendor PNS plug-ins
 * link against these directly.
 */

#include <math.h>
#include <string.h>

#include "pulseg_internal.h"

/* ================================================================== */
/*  Array statistics                                                  */
/* ================================================================== */

float pulseg__get_max_abs_real(const float *samples, int n)
{
    int i;
    float max_abs = 0.0f;
    float abs_val;

    if (!samples || n <= 0)
        return 0.0f;

    for (i = 0; i < n; ++i)
    {
        abs_val = (float)fabs(samples[i]);
        if (abs_val > max_abs)
            max_abs = abs_val;
    }
    return max_abs;
}

/* ================================================================== */
/*  Trapezoidal integration                                           */
/* ================================================================== */

float pulseg__trapz_real_uniform(const float *s, int n, float dt)
{
    int i;
    float sum = 0.0f;

    if (!s || n < 2 || dt <= 0.0f)
        return 0.0f;

    for (i = 1; i < n; ++i)
        sum += 0.5f * (s[i - 1] + s[i]) * dt;

    return sum;
}

float pulseg__trapz_real_nonuniform(const float *s, const float *t, int n)
{
    int i;
    float sum = 0.0f;
    float dt;

    if (!s || !t || n < 2)
        return 0.0f;

    for (i = 1; i < n; ++i)
    {
        dt = t[i] - t[i - 1];
        if (dt > 0.0f)
            sum += 0.5f * (s[i - 1] + s[i]) * dt;
    }
    return sum;
}

/* ================================================================== */
/*  Slew rate                                                         */
/* ================================================================== */

float pulseg__max_slew_real_uniform(const float *s, int n, float dt)
{
    int i;
    float max_slew = 0.0f;
    float slew;

    if (!s || n < 2 || dt <= 0.0f)
        return 0.0f;

    for (i = 1; i < n; ++i)
    {
        slew = (float)fabs((s[i] - s[i - 1]) / dt);
        if (slew > max_slew)
            max_slew = slew;
    }
    return max_slew;
}

float pulseg__max_slew_real_nonuniform(const float *s, const float *t, int n)
{
    int i;
    float max_slew = 0.0f;
    float dt, slew;

    if (!s || !t || n < 2)
        return 0.0f;

    for (i = 1; i < n; ++i)
    {
        dt = t[i] - t[i - 1];
        if (dt > 0.0f)
        {
            slew = (float)fabs((s[i] - s[i - 1]) / dt);
            if (slew > max_slew)
                max_slew = slew;
        }
    }
    return max_slew;
}

/* ================================================================== */
/*  Quaternion to rotation matrix                                     */
/* ================================================================== */

void pulseg__quaternion_to_matrix(float *matrix, const float *quat)
{
    float w = quat[0];
    float x = quat[1];
    float y = quat[2];
    float z = quat[3];
    float norm, inv_norm;
    float xx, yy, zz, xy, xz, yz, wx, wy, wz;

    norm = (float)sqrt(w * w + x * x + y * y + z * z);
    if (norm > 1e-9f)
    {
        inv_norm = 1.0f / norm;
        w *= inv_norm;
        x *= inv_norm;
        y *= inv_norm;
        z *= inv_norm;
    }
    else
    {
        /* degenerate quaternion -- return identity */
        matrix[0] = 1.0f;
        matrix[1] = 0.0f;
        matrix[2] = 0.0f;
        matrix[3] = 0.0f;
        matrix[4] = 1.0f;
        matrix[5] = 0.0f;
        matrix[6] = 0.0f;
        matrix[7] = 0.0f;
        matrix[8] = 1.0f;
        return;
    }

    xx = x * x;
    yy = y * y;
    zz = z * z;
    xy = x * y;
    xz = x * z;
    yz = y * z;
    wx = w * x;
    wy = w * y;
    wz = w * z;

    matrix[0] = 1.0f - 2.0f * (yy + zz);
    matrix[1] = 2.0f * (xy - wz);
    matrix[2] = 2.0f * (xz + wy);
    matrix[3] = 2.0f * (xy + wz);
    matrix[4] = 1.0f - 2.0f * (xx + zz);
    matrix[5] = 2.0f * (yz - wx);
    matrix[6] = 2.0f * (xz - wy);
    matrix[7] = 2.0f * (yz + wx);
    matrix[8] = 1.0f - 2.0f * (xx + yy);
}

/* ================================================================== */
/*  Identity check for a 3x3 rotation matrix                          */
/* ================================================================== */

int pulseg__is_identity3(const float *matrix)
{
    static const float I[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    int i;
    for (i = 0; i < 9; ++i)
        if ((float)fabs(matrix[i] - I[i]) > 1e-7f)
            return 0;
    return 1;
}

/* ================================================================== */
/*  3x3 rotation: out = R * v  (transpose=0)  or  out = R^T * v (1)  */
/* ================================================================== */

void pulseg__apply_rotation(float *out, const float *R, const float *v, int transpose)
{
    if (transpose)
    {
        out[0] = R[0] * v[0] + R[3] * v[1] + R[6] * v[2];
        out[1] = R[1] * v[0] + R[4] * v[1] + R[7] * v[2];
        out[2] = R[2] * v[0] + R[5] * v[1] + R[8] * v[2];
    }
    else
    {
        out[0] = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
        out[1] = R[3] * v[0] + R[4] * v[1] + R[5] * v[2];
        out[2] = R[6] * v[0] + R[7] * v[1] + R[8] * v[2];
    }
}

/* ================================================================== */
/*  1-D linear interpolation                                          */
/* ================================================================== */

void pulseg__interp1_linear(
    float *out,
    const float *x,
    int nx,
    const float *xp,
    const float *fp,
    int nxp)
{
    int i, j;
    float t;

    if (!x || !xp || !fp || !out || nx <= 0 || nxp <= 0)
        return;

    if (nxp == 1)
    {
        for (i = 0; i < nx; ++i)
            out[i] = fp[0];
        return;
    }

    j = 0;
    for (i = 0; i < nx; ++i)
    {
        if (x[i] <= xp[0])
        {
            out[i] = fp[0];
            continue;
        }
        if (x[i] >= xp[nxp - 1])
        {
            out[i] = fp[nxp - 1];
            continue;
        }
        while (j < nxp - 2 && xp[j + 1] < x[i])
            ++j;

        t = (x[i] - xp[j]) / (xp[j + 1] - xp[j]);
        out[i] = fp[j] + t * (fp[j + 1] - fp[j]);
    }
}

void pulseg__interp1_linear_complex(
    float *out_re,
    float *out_im,
    const float *x,
    int nx,
    const float *xp,
    const float *fp_re,
    const float *fp_im,
    int nxp)
{
    pulseg__interp1_linear(out_re, x, nx, xp, fp_re, nxp);
    pulseg__interp1_linear(out_im, x, nx, xp, fp_im, nxp);
}

/* ================================================================== */
/*  FFT helpers                                                       */
/* ================================================================== */

void pulseg__fftshift_complex(float *re, float *im, int n)
{
    int i, half, shift;
    float tmp_re, tmp_im;

    if (!re || !im || n <= 1)
        return;

    half = n / 2;
    shift = (n + 1) / 2;

    for (i = 0; i < half; ++i)
    {
        tmp_re = re[i];
        tmp_im = im[i];
        re[i] = re[i + shift];
        im[i] = im[i + shift];
        re[i + shift] = tmp_re;
        im[i + shift] = tmp_im;
    }
}

float pulseg__get_spectrum_flank(
    const float *x,
    const float *re,
    const float *im,
    int n,
    float cutoff,
    int reverse)
{
    int i;
    float max_mag, mag, threshold;

    if (!x || !re || !im || n <= 0)
        return 0.0f;

    max_mag = 0.0f;
    for (i = 0; i < n; ++i)
    {
        mag = (float)sqrt(re[i] * re[i] + im[i] * im[i]);
        if (mag > max_mag)
            max_mag = mag;
    }

    if (max_mag < 1e-12f)
        return 0.0f;

    threshold = cutoff * max_mag;

    if (reverse)
    {
        for (i = n - 1; i >= 0; --i)
        {
            mag = (float)sqrt(re[i] * re[i] + im[i] * im[i]);
            if (mag > threshold)
                return x[i];
        }
    }
    else
    {
        for (i = 0; i < n; ++i)
        {
            mag = (float)sqrt(re[i] * re[i] + im[i] * im[i]);
            if (mag > threshold)
                return x[i];
        }
    }
    return 0.0f;
}

/* ================================================================== */
/*  Next power of two                                                 */
/* ================================================================== */

size_t pulseg__next_pow2(size_t x)
{
    size_t v = 1;
    while (v < x)
        v <<= 1;
    return v;
}

/* ================================================================== */
/*  FFT convolution (real signals)                                    */
/* ================================================================== */
#include "external_kiss_fft.h"
#include "external_kiss_fftr.h"

struct pulseg__conv_fft_plan
{
    int signal_len;
    int nfft;
    int nfreq;
    kiss_fftr_cfg fwd;
    kiss_fftr_cfg inv;
    kiss_fft_cpx *kern_fft; /* [nfreq] transformed kernel, reused */
    float *pad_sig;         /* [nfft]  scratch */
    kiss_fft_cpx *sig_fft;  /* [nfreq] scratch */
    float *conv;            /* [nfft]  scratch */
};

int pulseg__conv_fft_plan_create(
    pulseg__conv_fft_plan **out_plan,
    int signal_len,
    const float *kernel,
    int kernel_len)
{
    pulseg__conv_fft_plan *p;
    int i, result;

    if (!out_plan)
        return PULSEG_ERR_NULL_POINTER;
    *out_plan = NULL;
    if (!kernel || signal_len <= 0 || kernel_len <= 0)
        return PULSEG_ERR_NULL_POINTER;

    p = (pulseg__conv_fft_plan *)PULSEG_ALLOC(sizeof(*p));
    if (!p)
        return PULSEG_ERR_ALLOC_FAILED;
    /* Set every pointer before the first goto: free() walks all of them. */
    p->fwd = NULL;
    p->inv = NULL;
    p->kern_fft = NULL;
    p->pad_sig = NULL;
    p->sig_fft = NULL;
    p->conv = NULL;

    p->signal_len = signal_len;
    p->nfft = (int)pulseg__next_pow2((size_t)(signal_len + kernel_len - 1));
    p->nfreq = p->nfft / 2 + 1;

    p->pad_sig = (float *)PULSEG_ALLOC((size_t)p->nfft * sizeof(float));
    p->conv = (float *)PULSEG_ALLOC((size_t)p->nfft * sizeof(float));
    p->sig_fft = (kiss_fft_cpx *)PULSEG_ALLOC((size_t)p->nfreq * sizeof(kiss_fft_cpx));
    p->kern_fft = (kiss_fft_cpx *)PULSEG_ALLOC((size_t)p->nfreq * sizeof(kiss_fft_cpx));
    if (!p->pad_sig || !p->conv || !p->sig_fft || !p->kern_fft)
    {
        result = PULSEG_ERR_ALLOC_FAILED;
        goto fail;
    }

    p->fwd = kiss_fftr_alloc(p->nfft, 0, NULL, NULL);
    p->inv = kiss_fftr_alloc(p->nfft, 1, NULL, NULL);
    if (!p->fwd || !p->inv)
    {
        result = PULSEG_ERR_PNS_FFT_FAILED;
        goto fail;
    }

    /* Transform the kernel once, through the signal scratch buffer --
     * apply() overwrites it on every call anyway. */
    for (i = 0; i < kernel_len; ++i)
        p->pad_sig[i] = kernel[i];
    for (i = kernel_len; i < p->nfft; ++i)
        p->pad_sig[i] = 0.0f;
    kiss_fftr(p->fwd, p->pad_sig, p->kern_fft);

    *out_plan = p;
    return PULSEG_SUCCESS;

fail:
    pulseg__conv_fft_plan_free(p);
    return result;
}

int pulseg__conv_fft_plan_apply(
    pulseg__conv_fft_plan *plan,
    float *output,
    const float *signal)
{
    int i, nfft, nfreq, signal_len;
    kiss_fft_cpx *sig_fft;
    const kiss_fft_cpx *kern_fft;
    float re, im, scale;

    if (!plan || !output || !signal)
        return PULSEG_ERR_NULL_POINTER;

    nfft = plan->nfft;
    nfreq = plan->nfreq;
    signal_len = plan->signal_len;
    sig_fft = plan->sig_fft;
    kern_fft = plan->kern_fft;

    for (i = 0; i < signal_len; ++i)
        plan->pad_sig[i] = signal[i];
    for (i = signal_len; i < nfft; ++i)
        plan->pad_sig[i] = 0.0f;

    kiss_fftr(plan->fwd, plan->pad_sig, sig_fft);

    for (i = 0; i < nfreq; ++i)
    {
        re = sig_fft[i].r * kern_fft[i].r - sig_fft[i].i * kern_fft[i].i;
        im = sig_fft[i].r * kern_fft[i].i + sig_fft[i].i * kern_fft[i].r;
        sig_fft[i].r = re;
        sig_fft[i].i = im;
    }

    kiss_fftri(plan->inv, sig_fft, plan->conv);
    scale = 1.0f / (float)nfft;
    for (i = 0; i < signal_len; ++i)
        output[i] = plan->conv[i] * scale;

    return PULSEG_SUCCESS;
}

void pulseg__conv_fft_plan_free(pulseg__conv_fft_plan *plan)
{
    if (!plan)
        return;
    if (plan->pad_sig)
        PULSEG_FREE(plan->pad_sig);
    if (plan->conv)
        PULSEG_FREE(plan->conv);
    if (plan->sig_fft)
        PULSEG_FREE(plan->sig_fft);
    if (plan->kern_fft)
        PULSEG_FREE(plan->kern_fft);
    if (plan->fwd)
        kiss_fftr_free(plan->fwd);
    if (plan->inv)
        kiss_fftr_free(plan->inv);
    PULSEG_FREE(plan);
}

int pulseg__calc_convolution_fft(
    float *output,
    const float *signal,
    int signal_len,
    const float *kernel,
    int kernel_len)
{
    pulseg__conv_fft_plan *plan;
    int result;

    plan = NULL;
    result = pulseg__conv_fft_plan_create(&plan, signal_len, kernel, kernel_len);
    if (PULSEG_FAILED(result))
        return result;

    result = pulseg__conv_fft_plan_apply(plan, output, signal);
    pulseg__conv_fft_plan_free(plan);
    return result;
}

/* ================================================================== */
/*  Chirp-z transform (double precision)                              */
/* ================================================================== */
/* The FFT underneath is the vendored kissfft compiled a second time in
 * double (src/vendor/external_kiss_fft_double.c); see the note there for
 * why the float copy beside it will not do. */

/* Chirp phases e^{i*h*k^2} for k = 0..count-1.
 *
 * By the recurrence h*(k+1)^2 = h*k^2 + h*(2k+1), whose increment is itself
 * an arithmetic progression -- so the angle is accumulated rather than
 * formed as h*k*k, which for large k would be a big number whose sin/cos
 * loses exactly the low bits this transform exists to keep. The running
 * angle is folded back into [-pi, pi] every step, which bounds it no matter
 * how many samples the waveform has. */
static void pulseg__chirp_phases(double *cr, double *ci, double h, int count)
{
    double angle = 0.0;
    double delta = h;
    int k;

    for (k = 0; k < count; ++k)
    {
        cr[k] = cos(angle);
        ci[k] = sin(angle);
        angle += delta;
        delta += 2.0 * h;
        if (angle > M_PI || angle < -M_PI)
            angle -= 2.0 * M_PI * floor((angle + M_PI) / (2.0 * M_PI));
    }
}

/* A reusable chirp-z setup.
 *
 * Everything except the samples and the starting angle depends only on
 * (n, m, dtheta): the chirp phases, the transformed convolution kernel, the
 * FFT plans and the scratch. The caller that motivated this evaluates the
 * same definition at seventeen offsets from the same comb, which share a
 * dtheta and differ only in theta0 -- so held this way, each extra offset
 * costs two transforms instead of three plus a full setup. */
struct pulseg__czt_plan
{
    int n;
    int m;
    int size;
    double dtheta;
    void *fwd;
    void *inv;
    double *chr; /* [max(n,m)] cos(h k^2) */
    double *chi; /* [max(n,m)] sin(h k^2) */
    double *kernel_f; /* [2*size] FFT of the chirp kernel */
    double *work;     /* [2*size] */
    double *spectrum; /* [2*size] */
};

void pulseg__czt_plan_free(pulseg__czt_plan *plan)
{
    if (!plan)
        return;
    if (plan->chr)
        PULSEG_FREE(plan->chr);
    if (plan->chi)
        PULSEG_FREE(plan->chi);
    if (plan->kernel_f)
        PULSEG_FREE(plan->kernel_f);
    if (plan->work)
        PULSEG_FREE(plan->work);
    if (plan->spectrum)
        PULSEG_FREE(plan->spectrum);
    pulseg__fft_double_free(plan->fwd);
    pulseg__fft_double_free(plan->inv);
    PULSEG_FREE(plan);
}

int pulseg__czt_plan_create(pulseg__czt_plan **out_plan, int n, int m, double dtheta)
{
    pulseg__czt_plan *plan;
    double h = 0.5 * dtheta;
    int longest, i;

    if (!out_plan || n < 1 || m < 1)
        return PULSEG_ERR_INVALID_ARGUMENT;
    *out_plan = NULL;

    plan = (pulseg__czt_plan *)PULSEG_ALLOC(sizeof(pulseg__czt_plan));
    if (!plan)
        return PULSEG_ERR_ALLOC_FAILED;
    memset(plan, 0, sizeof(*plan));

    plan->n = n;
    plan->m = m;
    plan->dtheta = dtheta;
    /* Not rounded up to a power of two: kissfft is mixed-radix, and the
     * cheapest size it likes that still holds the linear convolution is
     * often far below the next power of two (9375 against 16384 for a long
     * definition against a whole band). */
    plan->size = pulseg__fft_double_size(n + m - 1);
    longest = (n > m) ? n : m;

    plan->chr = (double *)PULSEG_ALLOC((size_t)longest * sizeof(double));
    plan->chi = (double *)PULSEG_ALLOC((size_t)longest * sizeof(double));
    plan->kernel_f = (double *)PULSEG_ALLOC((size_t)plan->size * 2 * sizeof(double));
    plan->work = (double *)PULSEG_ALLOC((size_t)plan->size * 2 * sizeof(double));
    plan->spectrum = (double *)PULSEG_ALLOC((size_t)plan->size * 2 * sizeof(double));
    plan->fwd = pulseg__fft_double_alloc(plan->size, 0);
    plan->inv = pulseg__fft_double_alloc(plan->size, 1);
    if (!plan->chr || !plan->chi || !plan->kernel_f || !plan->work || !plan->spectrum ||
        !plan->fwd || !plan->inv)
    {
        pulseg__czt_plan_free(plan);
        return PULSEG_ERR_ALLOC_FAILED;
    }

    pulseg__chirp_phases(plan->chr, plan->chi, h, longest);

    /* kernel[t] = e^{-i h t^2}, with the negative lags wrapped to the top of
     * the buffer so the cyclic convolution reproduces the linear one. */
    for (i = 0; i < plan->size * 2; ++i)
        plan->work[i] = 0.0;
    for (i = 0; i < m; ++i)
    {
        plan->work[2 * i] = plan->chr[i];
        plan->work[2 * i + 1] = -plan->chi[i];
    }
    for (i = 1; i < n; ++i)
    {
        plan->work[2 * (plan->size - i)] = plan->chr[i];
        plan->work[2 * (plan->size - i) + 1] = -plan->chi[i];
    }
    pulseg__fft_double_run(plan->fwd, plan->work, plan->kernel_f);

    *out_plan = plan;
    return PULSEG_SUCCESS;
}

int pulseg__czt_plan_apply(
    pulseg__czt_plan *plan,
    double *out_re,
    double *out_im,
    const float *a,
    double theta0)
{
    int i, size, n, m;
    double scale;

    if (!plan || !out_re || !out_im || !a)
        return PULSEG_ERR_NULL_POINTER;

    size = plan->size;
    n = plan->n;
    m = plan->m;

    for (i = 0; i < size * 2; ++i)
        plan->work[i] = 0.0;

    /* f[k] = a[k] * e^{i k theta0} * e^{i h k^2} */
    {
        double pr = 1.0, pi = 0.0; /* e^{i k theta0} */
        double sr = cos(theta0), si = sin(theta0);
        for (i = 0; i < n; ++i)
        {
            double ar = (double)a[i] * pr;
            double ai = (double)a[i] * pi;
            double npr;
            plan->work[2 * i] = ar * plan->chr[i] - ai * plan->chi[i];
            plan->work[2 * i + 1] = ar * plan->chi[i] + ai * plan->chr[i];
            npr = pr * sr - pi * si;
            pi = pr * si + pi * sr;
            pr = npr;
        }
    }

    pulseg__fft_double_run(plan->fwd, plan->work, plan->spectrum);
    for (i = 0; i < size; ++i)
    {
        double xr = plan->spectrum[2 * i] * plan->kernel_f[2 * i] -
                    plan->spectrum[2 * i + 1] * plan->kernel_f[2 * i + 1];
        double xi = plan->spectrum[2 * i] * plan->kernel_f[2 * i + 1] +
                    plan->spectrum[2 * i + 1] * plan->kernel_f[2 * i];
        plan->spectrum[2 * i] = xr;
        plan->spectrum[2 * i + 1] = xi;
    }
    pulseg__fft_double_run(plan->inv, plan->spectrum, plan->work);

    /* kissfft's inverse is unscaled, so 1/size rides along with the outgoing
     * chirp. */
    scale = 1.0 / (double)size;
    for (i = 0; i < m; ++i)
    {
        double cr = plan->chr[i] * scale;
        double ci = plan->chi[i] * scale;
        out_re[i] = plan->work[2 * i] * cr - plan->work[2 * i + 1] * ci;
        out_im[i] = plan->work[2 * i] * ci + plan->work[2 * i + 1] * cr;
    }
    return PULSEG_SUCCESS;
}

int pulseg__czt_unit(
    double *out_re,
    double *out_im,
    const float *a,
    int n,
    double theta0,
    double dtheta,
    int m)
{
    /* P_j = sum_{k<n} a[k] e^{i k (theta0 + j dtheta)}, j < m.
     *
     * Bluestein: k*j = (k^2 + j^2 - (j-k)^2)/2 turns the evaluation into one
     * convolution, so all m outputs cost O((n+m) log(n+m)) instead of the
     * O(n*m) of evaluating them one at a time. Every point lies on the unit
     * circle here (the frequencies are real and the samples are a real
     * waveform), which is why the geometry is given as two angles rather
     * than as complex ratios.
     *
     * One-shot form; a caller with several theta0 over the same geometry
     * should hold a plan instead. */
    pulseg__czt_plan *plan = NULL;
    int result;

    result = pulseg__czt_plan_create(&plan, n, m, dtheta);
    if (PULSEG_FAILED(result))
        return result;
    result = pulseg__czt_plan_apply(plan, out_re, out_im, a, theta0);
    pulseg__czt_plan_free(plan);
    return result;
}
