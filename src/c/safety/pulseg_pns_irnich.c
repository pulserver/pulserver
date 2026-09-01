/**
 * @file pulseg_pns_irnich.c
 * @brief The Irnich / den Boer rheobase-chronaxie PNS model.
 *
 * The response is the causal convolution of dG/dt with the nerve kernel
 * c/(c+tau)^2 scaled by s_min = rheobase / alpha, each tap being that kernel
 * integrated across its own sample interval,
 *
 *     k[i] = c*dt / (s_min * (c + i*dt) * (c + (i+1)*dt))
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
 *
 * Each tap is the kernel *integrated over its sample interval*, not sampled at
 * the interval's left edge, because the signal it convolves is itself a bin
 * integral: dG/dt on a raster is G differenced on that raster. In closed form,
 *
 *     int_{tau_i}^{tau_i + dt} c/(c+tau)^2 dtau = c*dt / ((c+tau_i)(c+tau_i+dt))
 *
 * written as the product rather than as the difference of two reciprocals,
 * which cancels catastrophically once tau >> c. This is what makes the
 * response a property of the waveform rather than of the raster it is
 * evaluated on; the invariant is held by
 * test_the_response_does_not_move_when_the_raster_is_halved.
 *
 * @return Newly allocated kernel of @p len samples, or NULL on failure.
 */
static float *irnich_build_kernel(const pulseg_pns_irnich *ctx, float dt_us, int len)
{
    float *kernel;
    double c_s;
    double dt_s;
    double s_min;
    double tau0;
    int i;

    if (len <= 0 || ctx->rheobase_t_per_m_per_s <= 0.0f || ctx->alpha <= 0.0f)
        return NULL;

    kernel = (float *)PULSEG_ALLOC((size_t)len * sizeof(float));
    if (!kernel)
        return NULL;

    c_s = (double)ctx->chronaxie_us * 1e-6;
    dt_s = (double)dt_us * 1e-6;
    s_min = (double)ctx->rheobase_t_per_m_per_s / (double)ctx->alpha;

    for (i = 0; i < len; i++)
    {
        tau0 = (double)i * dt_s;
        kernel[i] = (float)(c_s * dt_s / (s_min * (c_s + tau0) * (c_s + tau0 + dt_s)));
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
