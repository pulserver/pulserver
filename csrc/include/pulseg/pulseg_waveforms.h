/**
 * @file pulseg_waveforms.h
 * @brief Per-TR waveform extraction: flatten one TR of a subsequence onto a
 *        native or uniform raster for safety analysis and wrapper-side plotting.
 *
 * These getters read a loaded pulseg_collection; they do not touch the cache.
 * The k-space trajectory (a recon-side data product persisted in the cache) is
 * a separate concern -- see pulseg_trajectory.h.
 */

#ifndef PULSEG_WAVEFORMS_H
#define PULSEG_WAVEFORMS_H

#include "pulseg_config.h"
#include "pulseg_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /* ================================================================== */
    /*  TR gradient waveforms (uniform raster)                            */
    /* ================================================================== */

    /**
     * @brief Extract uniform-raster canonical-TR gradient waveforms for a subsequence.
     *
     * Returns canonical TR gradient waveforms (gx, gy, gz) for the requested
     * canonical TR index within the specified subsequence. The canonical TRs
     * are defined as unique shot-index patterns across the imaging region (for
     * degenerate prep/cooldown) or unique pass patterns (for non-degenerate
     * prep/cooldown).
     *
     * The output arrays are allocated by the library and must be freed by the
     * caller via pulseg_tr_gradient_waveforms_free().
     *
     * @param[in]  coll             Loaded collection.
     * @param[in]  subseq_idx       Subsequence index (0-based).
     * @param[in]  canonical_tr_idx Canonical TR index (0-based, within subsequence).
     * @param[out] waveforms        Output waveforms (caller frees).
     * @param[out] diag             Diagnostic on failure.
     * @return PULSEG_SUCCESS on success, negative error code on failure.
     */
    int pulseg_get_tr_gradient_waveforms(
        const pulseg_collection *coll,
        pulseg_tr_gradient_waveforms *waveforms,
        pulseg_diagnostic *diag,
        int subseq_idx,
        int canonical_tr_idx);

    /** @brief Free waveform arrays inside a pulseg_tr_gradient_waveforms. */
    void pulseg_tr_gradient_waveforms_free(pulseg_tr_gradient_waveforms *w);

    /* ================================================================== */
    /*  Native-timing TR waveforms (for plotting)                        */
    /* ================================================================== */

    /**
     * @brief Extract native-timing TR waveforms for all channels.
     *
     * Returns gradient (gx, gy, gz), RF (magnitude, phase), and ADC
     * event descriptors for the requested TR view.  Gradient waveforms
     * use native timing (trap corner-points, arb raster samples) and are
     * NOT interpolated to a uniform raster.  RF uses the RF raster.
     *
     * Amplitude modes:
     *   - PULSEG_AMP_MAX_POS  (0) — position-max across all TRs
     *   - PULSEG_AMP_ZERO_VAR (1) — zero variable-amplitude gradients, keep constant ones
     *   - PULSEG_AMP_ACTUAL   (2) — signed amplitude for given TR index
     *
     * For modes 0 and 1, @p tr_index is ignored (canonical main TR is used).
     * For mode 2, @p tr_index selects the TR instance (0-based).
     *
     * For degenerate (or absent) prep/cooldown the output covers a single
     * canonical TR (which already includes dummy TRs).  For non-degenerate
     * prep/cooldown the output covers the full pass.
     *
     * Block descriptors in the output carry segment assignment (or -1 for
     * prep / cooldown blocks).  The caller is responsible for freeing via
     * pulseg_tr_waveforms_free().
     *
     * @param[in]  coll               Loaded collection.
     * @param[in]  subseq_idx         Subsequence index (0 for single-seq).
     * @param[in]  amplitude_mode     PULSEG_AMP_MAX_POS / _ZERO_VAR / _ACTUAL.
     * @param[in]  tr_index           TR instance (only for _ACTUAL mode).
     * @param[in]  collapse_delays    Non-zero to shrink pure-delay blocks.
     * @param[in]  num_averages       Override average count (0 = use descriptor default).
     * @param[out] out                Output waveforms (caller frees).
     * @param[out] diag               Diagnostic on error.
     * @return PULSEG_SUCCESS on success, negative error code on failure.
     */
    int pulseg_get_tr_waveforms(
        const pulseg_collection *coll,
        pulseg_tr_waveforms *out,
        pulseg_diagnostic *diag,
        int subseq_idx,
        int amplitude_mode,
        int tr_index,
        int collapse_delays,
        int num_averages);

    /** @brief Free all arrays inside a pulseg_tr_waveforms. */
    void pulseg_tr_waveforms_free(pulseg_tr_waveforms *w);

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_WAVEFORMS_H */
