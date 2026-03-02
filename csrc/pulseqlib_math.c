/* pulseqlib_math.c -- math, interpolation, and FFT helpers */

#include <math.h>

#include "pulseqlib_internal.h"

/* ------------------------------------------------------------------ */
/*  Array statistics                                                  */
/* ------------------------------------------------------------------ */

float pulseqlib__get_max_abs_real(const float* samples, int n)
{
    int i;
    float max_abs = 0.0f;
    float abs_val;

    if (!samples || n <= 0)
        return 0.0f;

    for (i = 0; i < n; ++i) {
        abs_val = (float)fabs(samples[i]);
        if (abs_val > max_abs)
            max_abs = abs_val;
    }
    return max_abs;
}

int pulseqlib__get_max_abs_index_real(const float* samples, int n)
{
    int i;
    int max_idx = 0;
    float max_abs = 0.0f;
    float abs_val;

    if (!samples || n <= 0)
        return 0;

    for (i = 0; i < n; ++i) {
        abs_val = (float)fabs(samples[i]);
        if (abs_val > max_abs) {
            max_abs = abs_val;
            max_idx = i;
        }
    }
    return max_idx;
}

/* ------------------------------------------------------------------ */
/*  Complex conversion                                                */
/* ------------------------------------------------------------------ */

void pulseqlib__mag_phase_to_real_imag(
    float* re, float* im,
    const float* mag, const float* phase, int n)
{
    int i;

    if (!mag || !phase || !re || !im || n <= 0)
        return;

    for (i = 0; i < n; ++i) {
        re[i] = mag[i] * (float)cos(phase[i]);
        im[i] = mag[i] * (float)sin(phase[i]);
    }
}

/* ------------------------------------------------------------------ */
/*  Trapezoidal integration                                           */
/* ------------------------------------------------------------------ */

float pulseqlib__trapz_real_uniform(const float* s, int n, float dt)
{
    int i;
    float sum = 0.0f;

    if (!s || n < 2 || dt <= 0.0f)
        return 0.0f;

    for (i = 1; i < n; ++i)
        sum += 0.5f * (s[i - 1] + s[i]) * dt;

    return sum;
}

float pulseqlib__trapz_real_nonuniform(const float* s, const float* t, int n)
{
    int i;
    float sum = 0.0f;
    float dt;

    if (!s || !t || n < 2)
        return 0.0f;

    for (i = 1; i < n; ++i) {
        dt = t[i] - t[i - 1];
        if (dt > 0.0f)
            sum += 0.5f * (s[i - 1] + s[i]) * dt;
    }
    return sum;
}

float pulseqlib__trapz_complex_mag_uniform(
    const float* re, const float* im, int n, float dt)
{
    int i;
    float sum = 0.0f;
    float mag_prev, mag_curr;

    if (!re || !im || n < 2 || dt <= 0.0f)
        return 0.0f;

    mag_prev = (float)sqrt(re[0] * re[0] + im[0] * im[0]);

    for (i = 1; i < n; ++i) {
        mag_curr = (float)sqrt(re[i] * re[i] + im[i] * im[i]);
        sum += 0.5f * (mag_prev + mag_curr) * dt;
        mag_prev = mag_curr;
    }
    return sum;
}

float pulseqlib__trapz_complex_mag_nonuniform(
    const float* re, const float* im, const float* t, int n)
{
    int i;
    float sum = 0.0f;
    float dt;
    float mag_prev, mag_curr;

    if (!re || !im || !t || n < 2)
        return 0.0f;

    mag_prev = (float)sqrt(re[0] * re[0] + im[0] * im[0]);

    for (i = 1; i < n; ++i) {
        dt = t[i] - t[i - 1];
        if (dt > 0.0f) {
            mag_curr = (float)sqrt(re[i] * re[i] + im[i] * im[i]);
            sum += 0.5f * (mag_prev + mag_curr) * dt;
            mag_prev = mag_curr;
        }
    }
    return sum;
}

/* ------------------------------------------------------------------ */
/*  Slew rate                                                         */
/* ------------------------------------------------------------------ */

float pulseqlib__max_slew_real_uniform(const float* s, int n, float dt)
{
    int i;
    float max_slew = 0.0f;
    float slew;

    if (!s || n < 2 || dt <= 0.0f)
        return 0.0f;

    for (i = 1; i < n; ++i) {
        slew = (float)fabs((s[i] - s[i - 1]) / dt);
        if (slew > max_slew)
            max_slew = slew;
    }
    return max_slew;
}

float pulseqlib__max_slew_real_nonuniform(const float* s, const float* t, int n)
{
    int i;
    float max_slew = 0.0f;
    float dt, slew;

    if (!s || !t || n < 2)
        return 0.0f;

    for (i = 1; i < n; ++i) {
        dt = t[i] - t[i - 1];
        if (dt > 0.0f) {
            slew = (float)fabs((s[i] - s[i - 1]) / dt);
            if (slew > max_slew)
                max_slew = slew;
        }
    }
    return max_slew;
}

/* ------------------------------------------------------------------ */
/*  Quaternion to rotation matrix                                     */
/* ------------------------------------------------------------------ */

void pulseqlib__quaternion_to_matrix(float* matrix, const float* quat)
{
    float w = quat[0];
    float x = quat[1];
    float y = quat[2];
    float z = quat[3];
    float norm, inv_norm;
    float xx, yy, zz, xy, xz, yz, wx, wy, wz;

    norm = (float)sqrt(w*w + x*x + y*y + z*z);
    if (norm > 1e-9f) {
        inv_norm = 1.0f / norm;
        w *= inv_norm;
        x *= inv_norm;
        y *= inv_norm;
        z *= inv_norm;
    } else {
        /* degenerate quaternion -- return identity */
        matrix[0] = 1.0f; matrix[1] = 0.0f; matrix[2] = 0.0f;
        matrix[3] = 0.0f; matrix[4] = 1.0f; matrix[5] = 0.0f;
        matrix[6] = 0.0f; matrix[7] = 0.0f; matrix[8] = 1.0f;
        return;
    }

    xx = x * x;  yy = y * y;  zz = z * z;
    xy = x * y;  xz = x * z;  yz = y * z;
    wx = w * x;  wy = w * y;  wz = w * z;

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

/* ------------------------------------------------------------------ */
/*  3x3 rotation: out = R * v  (transpose=0)  or  out = R^T * v (1)  */
/* ------------------------------------------------------------------ */

void pulseqlib__apply_rotation(float* out, const float* R, const float* v,
                               int transpose)
{
    if (transpose) {
        out[0] = R[0] * v[0] + R[3] * v[1] + R[6] * v[2];
        out[1] = R[1] * v[0] + R[4] * v[1] + R[7] * v[2];
        out[2] = R[2] * v[0] + R[5] * v[1] + R[8] * v[2];
    } else {
        out[0] = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
        out[1] = R[3] * v[0] + R[4] * v[1] + R[5] * v[2];
        out[2] = R[6] * v[0] + R[7] * v[1] + R[8] * v[2];
    }
}

/* ------------------------------------------------------------------ */
/*  1-D linear interpolation                                          */
/* ------------------------------------------------------------------ */

void pulseqlib__interp1_linear(
    float* out,
    const float* x, int nx,
    const float* xp, const float* fp, int nxp)
{
    int i, j;
    float t;

    if (!x || !xp || !fp || !out || nx <= 0 || nxp <= 0)
        return;

    if (nxp == 1) {
        for (i = 0; i < nx; ++i)
            out[i] = fp[0];
        return;
    }

    j = 0;
    for (i = 0; i < nx; ++i) {
        if (x[i] <= xp[0]) {
            out[i] = fp[0];
            continue;
        }
        if (x[i] >= xp[nxp - 1]) {
            out[i] = fp[nxp - 1];
            continue;
        }
        while (j < nxp - 2 && xp[j + 1] < x[i])
            ++j;

        t = (x[i] - xp[j]) / (xp[j + 1] - xp[j]);
        out[i] = fp[j] + t * (fp[j + 1] - fp[j]);
    }
}

void pulseqlib__interp1_linear_complex(
    float* out_re, float* out_im,
    const float* x, int nx,
    const float* xp,
    const float* fp_re, const float* fp_im, int nxp)
{
    pulseqlib__interp1_linear(out_re, x, nx, xp, fp_re, nxp);
    pulseqlib__interp1_linear(out_im, x, nx, xp, fp_im, nxp);
}

/* ------------------------------------------------------------------ */
/*  FFT helpers                                                       */
/* ------------------------------------------------------------------ */

void pulseqlib__fftshift_complex(float* re, float* im, int n)
{
    int i, half, shift;
    float tmp_re, tmp_im;

    if (!re || !im || n <= 1)
        return;

    half = n / 2;
    shift = (n + 1) / 2;

    for (i = 0; i < half; ++i) {
        tmp_re = re[i];
        tmp_im = im[i];
        re[i] = re[i + shift];
        im[i] = im[i + shift];
        re[i + shift] = tmp_re;
        im[i + shift] = tmp_im;
    }
}

float pulseqlib__get_spectrum_flank(
    const float* x, const float* re, const float* im,
    int n, float cutoff, int reverse)
{
    int i;
    float max_mag, mag, threshold;

    if (!x || !re || !im || n <= 0)
        return 0.0f;

    max_mag = 0.0f;
    for (i = 0; i < n; ++i) {
        mag = (float)sqrt(re[i] * re[i] + im[i] * im[i]);
        if (mag > max_mag)
            max_mag = mag;
    }

    if (max_mag < 1e-12f)
        return 0.0f;

    threshold = cutoff * max_mag;

    if (reverse) {
        for (i = n - 1; i >= 0; --i) {
            mag = (float)sqrt(re[i] * re[i] + im[i] * im[i]);
            if (mag > threshold)
                return x[i];
        }
    } else {
        for (i = 0; i < n; ++i) {
            mag = (float)sqrt(re[i] * re[i] + im[i] * im[i]);
            if (mag > threshold)
                return x[i];
        }
    }
    return 0.0f;
}

/* ------------------------------------------------------------------ */
/*  Next power of two                                                 */
/* ------------------------------------------------------------------ */

size_t pulseqlib__next_pow2(size_t x)
{
    size_t v = 1;
    while (v < x) v <<= 1;
    return v;
}

/* ------------------------------------------------------------------ */
/*  FFT convolution (real signals)                                    */
/* ------------------------------------------------------------------ */
#include "external_kiss_fft.h"
#include "external_kiss_fftr.h"

int pulseqlib__calc_convolution_fft(
    float* output,
    const float* signal, int signal_len,
    const float* kernel, int kernel_len)
{
    int nfft, nfreq, i;
    kiss_fftr_cfg fwd;
    kiss_fftr_cfg inv;
    float* pad_sig;
    float* pad_kern;
    kiss_fft_cpx* sig_fft;
    kiss_fft_cpx* kern_fft;
    float* conv;
    float re, im, scale;
    int result;

    result = PULSEQLIB_SUCCESS;
    fwd = NULL;
    inv = NULL;
    pad_sig  = NULL;
    pad_kern = NULL;
    sig_fft  = NULL;
    kern_fft = NULL;
    conv     = NULL;

    nfft  = (int)pulseqlib__next_pow2((size_t)(signal_len + kernel_len - 1));
    nfreq = nfft / 2 + 1;

    pad_sig  = (float*)PULSEQLIB_ALLOC((size_t)nfft * sizeof(float));
    pad_kern = (float*)PULSEQLIB_ALLOC((size_t)nfft * sizeof(float));
    sig_fft  = (kiss_fft_cpx*)PULSEQLIB_ALLOC((size_t)nfreq * sizeof(kiss_fft_cpx));
    kern_fft = (kiss_fft_cpx*)PULSEQLIB_ALLOC((size_t)nfreq * sizeof(kiss_fft_cpx));
    conv     = (float*)PULSEQLIB_ALLOC((size_t)nfft * sizeof(float));
    if (!pad_sig || !pad_kern || !sig_fft || !kern_fft || !conv) {
        result = PULSEQLIB_ERR_ALLOC_FAILED; goto fail;
    }

    for (i = 0; i < signal_len; ++i) pad_sig[i] = signal[i];
    for (i = signal_len; i < nfft; ++i) pad_sig[i] = 0.0f;

    for (i = 0; i < kernel_len; ++i) pad_kern[i] = kernel[i];
    for (i = kernel_len; i < nfft; ++i) pad_kern[i] = 0.0f;

    fwd = kiss_fftr_alloc(nfft, 0, NULL, NULL);
    inv = kiss_fftr_alloc(nfft, 1, NULL, NULL);
    if (!fwd || !inv) { result = PULSEQLIB_ERR_PNS_FFT_FAILED; goto fail; }

    kiss_fftr(fwd, pad_sig, sig_fft);
    kiss_fftr(fwd, pad_kern, kern_fft);

    for (i = 0; i < nfreq; ++i) {
        re = sig_fft[i].r * kern_fft[i].r - sig_fft[i].i * kern_fft[i].i;
        im = sig_fft[i].r * kern_fft[i].i + sig_fft[i].i * kern_fft[i].r;
        sig_fft[i].r = re;
        sig_fft[i].i = im;
    }

    kiss_fftri(inv, sig_fft, conv);
    scale = 1.0f / (float)nfft;
    for (i = 0; i < signal_len; ++i) output[i] = conv[i] * scale;

fail:
    if (pad_sig)  PULSEQLIB_FREE(pad_sig);
    if (pad_kern) PULSEQLIB_FREE(pad_kern);
    if (sig_fft)  PULSEQLIB_FREE(sig_fft);
    if (kern_fft) PULSEQLIB_FREE(kern_fft);
    if (conv)     PULSEQLIB_FREE(conv);
    if (fwd)      kiss_fftr_free(fwd);
    if (inv)      kiss_fftr_free(inv);
    return result;
}