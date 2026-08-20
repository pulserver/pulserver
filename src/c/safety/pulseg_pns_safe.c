/**
 * @file pulseg_pns_safe.c
 * @brief The SAFE three-branch PNS model.
 *
 * Hebrank & Gebhardt's stimulation model (ISMRM 2000, No. 2007), in the
 * implementation Witzel and Szczepankiewicz published.  Per axis, with
 * s = dG/dt in T/m/s:
 *
 *     stim = ( a1*|LP(s, tau1)| + a2*LP(|s|, tau2) + a3*|LP(s, tau3)| )
 *            / stim_limit * g_scale * 100
 *
 * The first and third branches rectify *after* their lowpass and the second
 * *before* it, so the model is not a convolution: it publishes no kernel and
 * the safety core always evaluates it over the whole waveform.
 *
 * Only the coefficients are scanner-specific, and none are distributed with
 * this library.
 */

#include <math.h>
#include <stdlib.h>

#include "pulseg_config.h"
#include "pulseg_errors.h"
#include "pulseg_pns_models.h"
#include "pulseg_types.h"

/**
 * Time constants of history the slowest branch is given before the window.
 * The reference implementation pads by this much on both sides; the core
 * only appends, and only needs the branches to have settled.
 */
#define PULSEG_PNS_SAFE_PADDING_TAU 4.0f

/** @brief The longest time constant across all three axes, in ms. */
static float safe_longest_tau_ms(const pulseg_pns_safe *ctx)
{
    float taus[9];
    float longest;
    int i;

    taus[0] = ctx->x.tau1_ms;
    taus[1] = ctx->x.tau2_ms;
    taus[2] = ctx->x.tau3_ms;
    taus[3] = ctx->y.tau1_ms;
    taus[4] = ctx->y.tau2_ms;
    taus[5] = ctx->y.tau3_ms;
    taus[6] = ctx->z.tau1_ms;
    taus[7] = ctx->z.tau2_ms;
    taus[8] = ctx->z.tau3_ms;

    longest = taus[0];
    for (i = 1; i < 9; i++)
        if (taus[i] > longest)
            longest = taus[i];
    return longest;
}

/**
 * @brief One-pole RC lowpass, @p tau and @p dt in the same unit.
 *
 * The leading alpha applies to the first sample too, which is the 230206
 * correction the reference implementation carries.
 */
static void safe_lowpass(float *out, const float *signal, int n, float tau, float dt)
{
    float alpha = dt / (tau + dt);
    float state = 0.0f;
    int i;

    for (i = 0; i < n; i++)
    {
        state = alpha * signal[i] + (1.0f - alpha) * state;
        out[i] = state;
    }
}

/**
 * @brief Evaluate one axis into @p out.
 * @return PULSEG_SUCCESS, or a negative code if the scratch buffers fail.
 */
static int safe_evaluate_axis(
    float *out,
    const float *dgdt,
    int n,
    const pulseg_pns_safe_axis *axis,
    float dt_ms)
{
    float *rectified;
    float *branch;
    float scale;
    int i;

    rectified = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    branch = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    if (!rectified || !branch)
    {
        PULSEG_FREE(rectified);
        PULSEG_FREE(branch);
        return PULSEG_ERR_ALLOC_FAILED;
    }

    for (i = 0; i < n; i++)
        rectified[i] = (float)fabs((double)dgdt[i]);

    safe_lowpass(branch, dgdt, n, axis->tau1_ms, dt_ms);
    for (i = 0; i < n; i++)
        out[i] = axis->a1 * (float)fabs((double)branch[i]);

    safe_lowpass(branch, rectified, n, axis->tau2_ms, dt_ms);
    for (i = 0; i < n; i++)
        out[i] += axis->a2 * branch[i];

    safe_lowpass(branch, dgdt, n, axis->tau3_ms, dt_ms);
    for (i = 0; i < n; i++)
        out[i] += axis->a3 * (float)fabs((double)branch[i]);

    scale = (axis->stim_limit > 0.0f) ? (axis->g_scale * 100.0f / axis->stim_limit) : 0.0f;
    for (i = 0; i < n; i++)
        out[i] *= scale;

    PULSEG_FREE(rectified);
    PULSEG_FREE(branch);
    return PULSEG_SUCCESS;
}

static int safe_required_padding(void *ctx, float dt_us)
{
    const pulseg_pns_safe *self = (const pulseg_pns_safe *)ctx;
    float dt_ms;

    if (!self || dt_us <= 0.0f)
        return 0;

    dt_ms = dt_us * 1e-3f;
    return (int)(PULSEG_PNS_SAFE_PADDING_TAU * safe_longest_tau_ms(self) / dt_ms) + 1;
}

static int safe_evaluate(
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
    const pulseg_pns_safe *self = (const pulseg_pns_safe *)ctx;
    float dt_ms = dt_us * 1e-3f;
    int rc;

    rc = safe_evaluate_axis(out_x, dgdt_x, n, &self->x, dt_ms);
    if (rc != PULSEG_SUCCESS)
        return rc;
    rc = safe_evaluate_axis(out_y, dgdt_y, n, &self->y, dt_ms);
    if (rc != PULSEG_SUCCESS)
        return rc;
    return safe_evaluate_axis(out_z, dgdt_z, n, &self->z, dt_ms);
}

void pulseg_pns_safe_init(pulseg_pns_model *model, pulseg_pns_safe *ctx)
{
    if (!model || !ctx)
        return;

    model->ctx = ctx;
    model->required_padding = safe_required_padding;
    model->evaluate = safe_evaluate;
    model->kernel = NULL;
}
