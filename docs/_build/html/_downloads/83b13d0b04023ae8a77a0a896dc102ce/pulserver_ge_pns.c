/*
 * pulserver_ge_pns.c -- GE Irnich rheobase-chronaxie PNS model
 * implementation (c/(c+tau)^2).
 *
 * See pulserver_ge_pns.h. Math lifted verbatim from the pre-Stage-2
 * pulseg_safety.c build_pns_kernel() + process_pns_axis_circular()'s
 * convolution/scale step; only the split across required_padding()/
 * evaluate() and the vendor-neutral padding (now done by the public
 * safety core) are new.
 */

#include "pulserver_ge_pns.h"

#include "pulseg_internal.h"

/* Same constant as the pre-Stage-2 PNS_KERNEL_DURATION_FACTOR: how many
 * chronaxie time-constants of kernel history to keep before truncating
 * the 1/tau^2-decaying tail (Irnich rheobase-chronaxie form, not
 * exponential). */
#define PULSERVER_GE_PNS_KERNEL_DURATION_FACTOR 20.0f

typedef struct pulserver_ge_pns_ctx
{
    float chronaxie_us;
    float rheobase_hz_per_m_per_s;
    float alpha;
} pulserver_ge_pns_ctx;

/* Single static instance: predownload evaluates PNS for one PSD session
 * at a time (matches the pre-Stage-2 single static g_pns global in
 * pulserver.e); not reentrant/thread-safe by design. */
static pulserver_ge_pns_ctx s_ctx;

/* Builds the Irnich rheobase-chronaxie kernel k[i] = (dt/s_min) *
 * c / (c+tau)^2, i=0..n-1, verbatim from the pre-Stage-2
 * build_pns_kernel. */
static int pulserver_ge__build_kernel(
    const pulserver_ge_pns_ctx *ctx,
    float dt_us,
    float **kernel, int *kernel_len)
{
    int n, i;
    float c_s, dt_s, s_min, tau, denom;
    float *k;

    if (ctx->chronaxie_us <= 0.0f)
        return PULSEG_ERR_PNS_INVALID_CHRONAXIE;
    if (ctx->rheobase_hz_per_m_per_s <= 0.0f)
        return PULSEG_ERR_PNS_INVALID_RHEOBASE;
    if (ctx->alpha <= 0.0f)
        return PULSEG_ERR_PNS_INVALID_PARAMS;

    c_s = ctx->chronaxie_us * 1e-6f;
    dt_s = dt_us * 1e-6f;
    s_min = ctx->rheobase_hz_per_m_per_s / ctx->alpha;

    n = (int)(PULSERVER_GE_PNS_KERNEL_DURATION_FACTOR * c_s / dt_s) + 1;
    k = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    if (!k)
        return PULSEG_ERR_ALLOC_FAILED;

    for (i = 0; i < n; ++i)
    {
        tau = (float)i * dt_s;
        denom = (c_s + tau) * (c_s + tau);
        k[i] = (dt_s / s_min) * (c_s / denom);
    }

    *kernel = k;
    *kernel_len = n;
    return PULSEG_SUCCESS;
}

static int pulserver_ge__required_padding(void *ctx_v, float dt_us)
{
    pulserver_ge_pns_ctx *ctx = (pulserver_ge_pns_ctx *)ctx_v;
    float *kernel;
    int kernel_len, rc;

    kernel = NULL;
    rc = pulserver_ge__build_kernel(ctx, dt_us, &kernel, &kernel_len);
    if (PULSEG_FAILED(rc))
        return rc; /* negative -- caller treats <0 as invalid */
    PULSEG_FREE(kernel);
    return kernel_len;
}

/* The Irnich response is a plain convolution of dG/dt with the kernel above,
 * scaled to percent -- exactly what evaluate() computes -- so it is safe to
 * publish it and let the safety core exploit linearity. */
static int pulserver_ge__kernel(void *ctx_v,
                                 float dt_us,
                                 float **out_kernel, int *out_len, float *out_scale)
{
    int rc;

    if (!out_kernel || !out_len || !out_scale)
        return PULSEG_ERR_NULL_POINTER;

    rc = pulserver_ge__build_kernel((pulserver_ge_pns_ctx *)ctx_v, dt_us, out_kernel, out_len);
    if (PULSEG_FAILED(rc))
        return rc;

    *out_scale = 100.0f;
    return PULSEG_SUCCESS;
}

static int pulserver_ge__evaluate(void *ctx_v,
                                   const float *dgdt_x, const float *dgdt_y, const float *dgdt_z,
                                   int n, float dt_us,
                                   float *out_x, float *out_y, float *out_z)
{
    pulserver_ge_pns_ctx *ctx = (pulserver_ge_pns_ctx *)ctx_v;
    pulseg__conv_fft_plan *plan;
    float *kernel;
    int kernel_len, i, rc;

    kernel = NULL;
    plan = NULL;
    rc = pulserver_ge__build_kernel(ctx, dt_us, &kernel, &kernel_len);
    if (PULSEG_FAILED(rc))
        return rc;

    /* One plan for all three axes: same length, same kernel. The kernel
     * transform and the kiss_fft twiddle tables are built once here
     * instead of once per axis. */
    rc = pulseg__conv_fft_plan_create(&plan, n, kernel, kernel_len);
    if (PULSEG_FAILED(rc))
        goto fail;

    rc = pulseg__conv_fft_plan_apply(plan, out_x, dgdt_x);
    if (PULSEG_FAILED(rc))
        goto fail;
    rc = pulseg__conv_fft_plan_apply(plan, out_y, dgdt_y);
    if (PULSEG_FAILED(rc))
        goto fail;
    rc = pulseg__conv_fft_plan_apply(plan, out_z, dgdt_z);
    if (PULSEG_FAILED(rc))
        goto fail;

    for (i = 0; i < n; ++i)
    {
        out_x[i] *= 100.0f;
        out_y[i] *= 100.0f;
        out_z[i] *= 100.0f;
    }

    pulseg__conv_fft_plan_free(plan);
    PULSEG_FREE(kernel);
    return PULSEG_SUCCESS;

fail:
    pulseg__conv_fft_plan_free(plan);
    PULSEG_FREE(kernel);
    return rc;
}

void pulserver_ge_pns_model_init(pulseg_pns_model *model,
                                  float chronaxie_us,
                                  float rheobase_hz_per_m_per_s,
                                  float alpha)
{
    if (!model)
        return;
    s_ctx.chronaxie_us = chronaxie_us;
    s_ctx.rheobase_hz_per_m_per_s = rheobase_hz_per_m_per_s;
    s_ctx.alpha = alpha;
    model->ctx = &s_ctx;
    model->required_padding = pulserver_ge__required_padding;
    model->evaluate = pulserver_ge__evaluate;
    model->kernel = pulserver_ge__kernel;
}
