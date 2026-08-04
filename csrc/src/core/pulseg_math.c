/**
 * @file pulseg_math.c
 * @brief Numeric helpers shared across the library.
 *
 * Integration, slew, interpolation, rotation and FFT convolution. Nothing
 * here knows about sequences; keep it that way, since the vendor PNS plug-ins
 * link against these directly.
 */

#include <math.h>

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