/**
 * @file pulseg_pns_irnich.c
 * @brief The Irnich / den Boer rheobase-chronaxie PNS model.
 *
 * The response is the causal convolution of dG/dt with
 *
 *     k[i] = (dt / s_min) * c / (c + i*dt)^2,   s_min = rheobase / alpha
 *
 * reported as a percentage of the stimulation threshold.  The `1/tau^2` tail
 * never reaches zero, so the kernel is truncated at
 * PULSEG_PNS_IRNICH_KERNEL_TAU -- and the same length is what the model asks
 * the core to pre-pad with, so the filter enters the window already warm.
 *
 * Because this is exactly a convolution, the model publishes its impulse
 * response (pulseg_pns_model::kernel) and the core is free to assemble the
 * response per distinct gradient shape instead.
 */

#include <stdlib.h>

#include "pulseg_config.h"
#include "pulseg_errors.h"
#include "pulseg_pns_models.h"
#include "pulseg_types.h"

/** Chronaxie constants of kernel kept before the 1/tau^2 tail is truncated. */
#define PULSEG_PNS_IRNICH_KERNEL_TAU 20.0f

/** The percentage the evaluated response is reported in. */
#define PULSEG_PNS_IRNICH_SCALE 100.0f

/**
 * @brief Number of kernel samples at a given raster.
 * @return Sample count (>= 1), or 0 if the parameters are degenerate.
 */
static int irnich_kernel_length(const pulseg_pns_irnich *ctx, float dt_us)
{
    float c_s;
    float dt_s;

    if (!ctx || ctx->chronaxie_us <= 0.0f || dt_us <= 0.0f)
        return 0;

    c_s = ctx->chronaxie_us * 1e-6f;
    dt_s = dt_us * 1e-6f;
    return (int)(PULSEG_PNS_IRNICH_KERNEL_TAU * c_s / dt_s) + 1;
}

/**
 * @brief Build the impulse response.
 * @return Newly allocated kernel of @p len samples, or NULL on failure.
 */
static float *irnich_build_kernel(const pulseg_pns_irnich *ctx, float dt_us, int len)
{
    float *kernel;
    float c_s;
    float dt_s;
    float s_min;
    int i;

    if (len <= 0 || ctx->rheobase_t_per_m_per_s <= 0.0f || ctx->alpha <= 0.0f)
        return NULL;

    kernel = (float *)PULSEG_ALLOC((size_t)len * sizeof(float));
    if (!kernel)
        return NULL;

    c_s = ctx->chronaxie_us * 1e-6f;
    dt_s = dt_us * 1e-6f;
    s_min = ctx->rheobase_t_per_m_per_s / ctx->alpha;

    for (i = 0; i < len; i++)
    {
        float tau = (float)i * dt_s;
        float denom = (c_s + tau) * (c_s + tau);
        kernel[i] = (dt_s / s_min) * (c_s / denom);
    }
    return kernel;
}

/** @brief out[i] = scale * sum_{k=0..min(i,len-1)} kernel[k] * sig[i-k]. */
static void irnich_convolve_causal(
    float *out,
    const float *sig,
    int n,
    const float *kernel,
    int kernel_len)
{
    int i;

    for (i = 0; i < n; i++)
    {
        float acc = 0.0f;
        int kmax = (i < kernel_len - 1) ? i : (kernel_len - 1);
        int k;

        for (k = 0; k <= kmax; k++)
            acc += kernel[k] * sig[i - k];
        out[i] = acc * PULSEG_PNS_IRNICH_SCALE;
    }
}

static int irnich_required_padding(void *ctx, float dt_us)
{
    return irnich_kernel_length((const pulseg_pns_irnich *)ctx, dt_us);
}

static int irnich_evaluate(
    void *ctx,
    const float *dgdt_x,
    const float *dgdt_y,
    const float *dgdt_z,
    int n,
    float dt_us,
    float *out_x,
    float *out_y,
    float *out_z)
{
    const pulseg_pns_irnich *self = (const pulseg_pns_irnich *)ctx;
    float *kernel;
    int len;

    len = irnich_kernel_length(self, dt_us);
    kernel = irnich_build_kernel(self, dt_us, len);
    if (!kernel)
        return PULSEG_ERR_ALLOC_FAILED;

    irnich_convolve_causal(out_x, dgdt_x, n, kernel, len);
    irnich_convolve_causal(out_y, dgdt_y, n, kernel, len);
    irnich_convolve_causal(out_z, dgdt_z, n, kernel, len);

    PULSEG_FREE(kernel);
    return PULSEG_SUCCESS;
}

static int irnich_kernel(void *ctx, float dt_us, float **out_kernel, int *out_len, float *out_scale)
{
    const pulseg_pns_irnich *self = (const pulseg_pns_irnich *)ctx;
    float *kernel;
    int len;

    if (!out_kernel || !out_len || !out_scale)
        return PULSEG_ERR_INVALID_ARGUMENT;

    len = irnich_kernel_length(self, dt_us);
    kernel = irnich_build_kernel(self, dt_us, len);
    if (!kernel)
        return PULSEG_ERR_ALLOC_FAILED;

    *out_kernel = kernel;
    *out_len = len;
    *out_scale = PULSEG_PNS_IRNICH_SCALE;
    return PULSEG_SUCCESS;
}

void pulseg_pns_irnich_init(
    pulseg_pns_model *model,
    pulseg_pns_irnich *ctx,
    float chronaxie_us,
    float rheobase_t_per_m_per_s,
    float alpha)
{
    if (!model || !ctx)
        return;

    ctx->chronaxie_us = chronaxie_us;
    ctx->rheobase_t_per_m_per_s = rheobase_t_per_m_per_s;
    ctx->alpha = alpha;

    model->ctx = ctx;
    model->required_padding = irnich_required_padding;
    model->evaluate = irnich_evaluate;
    model->kernel = irnich_kernel;
}
