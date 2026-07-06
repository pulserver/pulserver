/**
 * @file pulseg_safety.h
 * @brief RF/gradient safety checks, mechanical-resonance spectra, and PNS.
 *
 * Split out of the former pulseg_methods.h (Stage 1 layout normalization).
 * All functions use the pulseg_ prefix and are declared extern "C" when
 * compiled with a C++ compiler.
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
     * @param[in]  pns_params             PNS model parameters (NULL to skip PNS).
     * @param[in]  pns_threshold_percent  PNS threshold (100 = 100 %).
     * @return PULSEG_SUCCESS if safe, negative error code on violation.
     */
    int pulseg_check_safety(
        pulseg_collection *coll,
        pulseg_diagnostic *diag,
        const pulseg_opts *opts,
        int num_forbidden_bands,
        const pulseg_forbidden_band *forbidden_bands,
        const pulseg_pns_params *pns_params,
        float pns_threshold_percent);

    /**
     * @brief Vendor-neutral one-shot safety check from a `.seq` file path.
     *
     * Convenience facade for third-party vendors that do not want to manage
     * a `pulseg_collection` lifetime: internally calls
     * `pulseg_read()` (no cache, no signature verification, no label
     * parsing), then `pulseg_check_safety()`, then
     * `pulseg_collection_free()`.  Performs the same suite of checks as
     * `pulseg_check_safety()` (max gradient amplitude, gradient
     * continuity, max slew rate, structural mechanical-resonance forbidden
     * bands, and the vendor PNS model selected by `pns_params->vendor`).
     *
     * @param[out] diag                   Diagnostic on violation / load error.
     * @param[in]  seq_path               Path to a (possibly chained) `.seq` file.
     * @param[in]  opts                   Scanner limits and rasters.
     * @param[in]  num_forbidden_bands    Number of acoustic forbidden bands.
     * @param[in]  forbidden_bands        Array of forbidden bands.
     * @param[in]  pns_params             PNS model parameters
     *                                    (NULL to skip PNS).
     * @param[in]  pns_threshold_percent  PNS threshold (100 = 100 %).
     * @return PULSEG_SUCCESS if safe, negative error code on load failure
     *         or safety violation.
     */
    int pulseg_check_safety_from_file(
        pulseg_diagnostic *diag,
        const char *seq_path,
        const pulseg_opts *opts,
        int num_forbidden_bands,
        const pulseg_forbidden_band *forbidden_bands,
        const pulseg_pns_params *pns_params,
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
     * @return PULSEG_SUCCESS on success, negative error code on failure.
     */
    int pulseg_calc_mech_resonances(const pulseg_collection *coll,
                                       pulseg_mech_resonances_spectra *spectra,
                                       pulseg_diagnostic *diag,
                                       int subseq_idx,
                                       int canonical_tr_idx,
                                       const pulseg_opts *opts,
                                       float target_resolution_hz,
                                       float max_freq_hz,
                                       int num_forbidden_bands,
                                       const pulseg_forbidden_band *forbidden_bands);

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
     * @param[in]  params       PNS model parameters.
     * @return PULSEG_SUCCESS on success, negative error code on failure.
     */
    int pulseg_calc_pns(const pulseg_collection *coll,
                           pulseg_pns_result *result,
                           pulseg_diagnostic *diag,
                           int subseq_idx,
                           int canonical_tr_idx,
                           const pulseg_opts *opts,
                           const pulseg_pns_params *params);

    /** @brief Free arrays inside a pulseg_pns_result. */
    void pulseg_pns_result_free(pulseg_pns_result *r);

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_SAFETY_H */
