/**
 * @file pulseg_pns_models.h
 * @brief The two published PNS models, as pulseg_pns_model implementations.
 *
 * @ref pulseg_pns_model is an interface, so that a vendor with a proprietary
 * stimulation-threshold form can inject it without this library knowing
 * anything about it.  The two forms that *are* published are supplied here:
 *
 * - **Irnich / den Boer**, the rheobase-chronaxie `c/(c+tau)^2` kernel.  It
 *   is linear and time-invariant, so it publishes its impulse response and
 *   the safety core may take its memoized per-shape route (see
 *   pulseg_pns_memo.c), which on a long canonical TR is one to two orders of
 *   magnitude cheaper than convolving the whole window.
 * - **SAFE**, Hebrank & Gebhardt's three-branch filter (ISMRM 2000, No.
 *   2007), in the implementation Witzel and Szczepankiewicz published.  Its
 *   branches rectify on both sides of their lowpass, so it is not a
 *   convolution and deliberately publishes no kernel.
 *
 * Only the per-scanner coefficients are proprietary, and none are
 * distributed here: both models are initialised from numbers the caller
 * supplies.
 *
 * Both `*_init` functions leave the model borrowing the context struct, which
 * must outlive the pulseg_calc_pns / pulseg_check_safety call it is passed
 * to.
 */

#ifndef PULSEG_PNS_MODELS_H
#define PULSEG_PNS_MODELS_H

#include "pulseg_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Irnich rheobase-chronaxie model state.
     *
     * Fields are the caller's to fill, or to pass to pulseg_pns_irnich_init.
     */
    typedef struct pulseg_pns_irnich
    {
        float chronaxie_us;           /**< chronaxie time constant (us) */
        float rheobase_t_per_m_per_s; /**< rheobase slew rate (T/m/s) */
        float alpha;                  /**< threshold scaling (dimensionless) */
    } pulseg_pns_irnich;

    /**
     * @brief Point a pulseg_pns_model at an Irnich context.
     *
     * @param model  Receives the model. Publishes a kernel: the safety core may
     *               evaluate it by per-shape assembly rather than convolution.
     * @param ctx    Model state, filled from the parameters below. Must outlive
     *               every call made through @p model.
     * @param chronaxie_us  Chronaxie time constant (us), > 0.
     * @param rheobase_t_per_m_per_s  Rheobase slew rate (T/m/s), > 0 --
     *               the units dG/dt reaches a model in.
     * @param alpha  Threshold scaling; the effective minimum slew is
     *               rheobase / alpha.
     */
    void pulseg_pns_irnich_init(
        pulseg_pns_model *model,
        pulseg_pns_irnich *ctx,
        float chronaxie_us,
        float rheobase_t_per_m_per_s,
        float alpha);

    /** @brief SAFE coefficients for one gradient axis, in the units .asc states. */
    typedef struct pulseg_pns_safe_axis
    {
        float a1;         /**< weight of the |LP(s, tau1)| branch */
        float a2;         /**< weight of the LP(|s|, tau2) branch */
        float a3;         /**< weight of the |LP(s, tau3)| branch */
        float tau1_ms;    /**< branch 1 time constant (ms) */
        float tau2_ms;    /**< branch 2 time constant (ms) */
        float tau3_ms;    /**< branch 3 time constant (ms) */
        float stim_limit; /**< stimulation limit (T/m/s) */
        float g_scale;    /**< axis scaling (dimensionless) */
    } pulseg_pns_safe_axis;

    /** @brief SAFE model state: one coefficient set per axis. */
    typedef struct pulseg_pns_safe
    {
        pulseg_pns_safe_axis x;
        pulseg_pns_safe_axis y;
        pulseg_pns_safe_axis z;
    } pulseg_pns_safe;

    /**
     * @brief Point a pulseg_pns_model at a SAFE context.
     *
     * @param model  Receives the model. Publishes no kernel -- SAFE is not a
     *               convolution -- so the core always takes its exact path.
     * @param ctx    Model state, already filled. Must outlive every call made
     *               through @p model.
     */
    void pulseg_pns_safe_init(pulseg_pns_model *model, pulseg_pns_safe *ctx);

    /* ================================================================== */
    /*  Convolution plan, for a caller's own linear model                 */
    /* ================================================================== */

    /**
     * @brief A reusable FFT-convolution plan.
     *
     * A model that is a convolution has to evaluate three gradient axes
     * against one kernel.  A plan transforms that kernel once and allocates
     * the transform's twiddle tables once, rather than repeating both per
     * axis.
     *
     * This is published because a vendor's stimulation model is caller-side
     * code: it would otherwise have to bring its own FFT to do what the
     * models here already do.
     */
    typedef struct pulseg_conv_fft_plan pulseg_conv_fft_plan;

    /**
     * @brief Build a plan for convolving @p signal_len samples with @p kernel.
     *
     * @param[out] out_plan    Receives the plan; free with
     *                         pulseg_conv_fft_plan_free().
     * @param[in]  signal_len  Length of every signal the plan will be applied
     *                         to.
     * @param[in]  kernel      Impulse response, borrowed only for this call.
     * @param[in]  kernel_len  Length of @p kernel.
     * @return PULSEG_SUCCESS or a negative error code.
     */
    int pulseg_conv_fft_plan_create(
        pulseg_conv_fft_plan **out_plan,
        int signal_len,
        const float *kernel,
        int kernel_len);

    /**
     * @brief Convolve one signal through a plan.
     *
     * @param[in]  plan    Plan from pulseg_conv_fft_plan_create().
     * @param[out] output  Receives @c signal_len samples.
     * @param[in]  signal  @c signal_len input samples.
     * @return PULSEG_SUCCESS or a negative error code.
     */
    int pulseg_conv_fft_plan_apply(pulseg_conv_fft_plan *plan, float *output, const float *signal);

    /** @brief Free a plan. Safe on NULL. */
    void pulseg_conv_fft_plan_free(pulseg_conv_fft_plan *plan);

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_PNS_MODELS_H */
