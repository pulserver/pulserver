/**
 * @file pulseg_safety.h
 * @brief Vendor-neutral gradient safety: amplitude, slew, continuity,
 *        acoustic resonance and PNS.
 *
 * pulseg_check_safety() is the single gate a vendor integration must pass a
 * collection through before playing it. The PNS model is injected by the
 * caller (pulseg_pns_model), so no vendor-proprietary nerve-stimulation
 * formula lives in this library.
 *
 * RF safety is deliberately absent: SAR limits are vendor-proprietary.
 * Consumers read the per-pulse summary via pulseg_get_rf_stats() and apply
 * their own scanner limits.
 */

#ifndef PULSEG_SAFETY_H
#define PULSEG_SAFETY_H

#include "pulseg_config.h"
#include "pulseg_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /* ================================================================== */
    /*  Safety checks (detect violation and return immediately)           */
    /* ================================================================== */

    /**
     * @brief Run all safety checks (gradient limits, acoustic, PNS).
     *
     * Detects the first violation and returns immediately with a
     * descriptive diagnostic message.  Does NOT track worst-case.
     *
     * Internally, TR gradient waveforms are extracted once (without
     * segment labels) and interpolated to a uniform raster.  The
     * resulting uniform waveforms are shared between acoustic and PNS
     * checks to avoid redundant computation.
     *
     * @param[in]  coll                   Collection (non-const: cursor dry-run).
     * @param[out] diag                   Diagnostic on violation.
     * @param[in]  opts                   Scanner limits.
     * @param[in]  num_forbidden_bands    Number of acoustic bands.
     * @param[in]  forbidden_bands        Array of forbidden bands.
     * @param[in]  pns_model              PNS evaluator (NULL to skip PNS).
     * @param[in]  pns_threshold_percent  PNS threshold (100 = 100 %).
     * @return PULSEG_SUCCESS if safe, negative error code on violation.
     */
    int pulseg_check_safety(
        pulseg_collection *coll,
        pulseg_diagnostic *diag,
        const pulseg_opts *opts,
        int num_forbidden_bands,
        const pulseg_forbidden_band *forbidden_bands,
        const pulseg_pns_model *pns_model,
        float pns_threshold_percent);

    /* ================================================================== */
    /*  Acoustic spectra (for wrapper-side plotting)                      */
    /* ================================================================== */

    /**
     * @brief Compute mechanical resonances spectral data for wrapper-side plotting.
     *
     * Independently extracts TR gradient waveforms (without segment
     * labels), interpolates them to uniform raster, and computes
     * spectrograms, full-TR spectra, and sequence-level harmonics.
     * Peak candidate masks are included for forbidden-band detection
     * in the wrapper.
     *
     * @param[out] spectra                  Receives spectral data (caller frees
     *                                       via pulseg_mech_resonances_spectra_free).
     * @param[out] diag                     Diagnostic on failure.
     * @param[in]  coll                     Loaded collection.
     * @param[in]  subseq_idx              Subsequence index.
     * @param[in]  opts                     Scanner limits.
     * @param[in]  target_resolution_hz     Spectral resolution (0 = auto).
     * @param[in]  max_freq_hz             Max frequency to report (0 = auto).
     * @param[in]  num_forbidden_bands      Number of forbidden bands.
     * @param[in]  forbidden_bands          Array of forbidden bands.
     * @return PULSEG_SUCCESS on success, negative error code on failure.
     */

    /**
     * @brief Compute mechanical resonances spectral data for a specific canonical TR of a subsequence.
     *
     * @param[out] spectra                  Receives spectral data (caller frees via pulseg_mech_resonances_spectra_free).
     * @param[out] diag                     Diagnostic on failure.
     * @param[in]  coll                     Loaded collection.
     * @param[in]  subseq_idx               Subsequence index.
     * @param[in]  canonical_tr_idx         Canonical TR index (0-based, within subsequence).
     * @param[in]  opts                     Scanner limits.
     * @param[in]  target_resolution_hz     Spectral resolution (0 = auto).
     * @param[in]  max_freq_hz              Max frequency to report (0 = auto).
     * @param[in]  num_forbidden_bands      Number of forbidden bands.
     * @param[in]  forbidden_bands          Array of forbidden bands.
     * @param[in]  compress_trains          Nonzero to evaluate equally-spaced
     *                                       occurrence trains in compressed
     *                                       form, exactly as pulseg_check_safety
     *                                       does — pass 1 for plots that must
     *                                       show the lines the headless gate
     *                                       actually decides on.  Pass 0 for the
     *                                       uncompressed reference evaluation of
     *                                       the same math (a debugging aid, and
     *                                       the only mode in which a component
     *                                       term maps to a single materialised
     *                                       occurrence rather than a train).
     * @return PULSEG_SUCCESS on success, negative error code on failure.
     */
    int pulseg_calc_mech_resonances(
        const pulseg_collection *coll,
        pulseg_mech_resonances_spectra *spectra,
        pulseg_diagnostic *diag,
        int subseq_idx,
        int canonical_tr_idx,
        const pulseg_opts *opts,
        float target_resolution_hz,
        float max_freq_hz,
        int num_forbidden_bands,
        const pulseg_forbidden_band *forbidden_bands,
        int compress_trains);

    /** @brief Free arrays inside a pulseg_mech_resonances_spectra. */
    void pulseg_mech_resonances_spectra_free(pulseg_mech_resonances_spectra *s);

    /* ================================================================== */
    /*  PNS slew-rate computation (for wrapper-side plotting)             */
    /* ================================================================== */

    /**
     * @brief Compute convolved slew-rate waveforms for PNS plotting.
     *
     * Independently extracts TR gradient waveforms (without segment
     * labels), interpolates them to uniform raster, and convolves with
     * the PNS model kernel.  Returns per-axis slew rates; the wrapper
     * can trivially compute combined PNS = sqrt(x^2 + y^2 + z^2) and
     * threshold percentage.
     *
     * @param[out] result       Receives slew-rate waveforms (caller frees
     *                           via pulseg_pns_result_free).
     * @param[out] diag         Diagnostic on failure.
     * @param[in]  coll         Loaded collection.
     * @param[in]  subseq_idx   Subsequence index.
     * @param[in]  opts         Scanner limits.
     * @param[in]  params       PNS model parameters.
     * @return PULSEG_SUCCESS on success, negative error code on failure.
     */

    /**
     * @brief Compute PNS slew-rate waveforms for a specific canonical TR of a subsequence.
     *
     * @param[out] result       Receives slew-rate waveforms (caller frees via pulseg_pns_result_free).
     * @param[out] diag         Diagnostic on failure.
     * @param[in]  coll         Loaded collection.
     * @param[in]  subseq_idx   Subsequence index.
     * @param[in]  canonical_tr_idx Canonical TR index (0-based, within subsequence).
     * @param[in]  opts         Scanner limits.
     * @param[in]  model        PNS evaluator.
     * @return PULSEG_SUCCESS on success, negative error code on failure.
     */
    int pulseg_calc_pns(
        const pulseg_collection *coll,
        pulseg_pns_result *result,
        pulseg_diagnostic *diag,
        int subseq_idx,
        int canonical_tr_idx,
        const pulseg_opts *opts,
        const pulseg_pns_model *model);

    /** @brief Free arrays inside a pulseg_pns_result. */
    void pulseg_pns_result_free(pulseg_pns_result *r);

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_SAFETY_H */
