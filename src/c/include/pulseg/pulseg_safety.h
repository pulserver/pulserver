/**
 * @file pulseg_safety.h
 * @brief Vendor-neutral gradient safety: amplitude, slew, continuity,
 *        acoustic resonance and PNS.
 *
 * pulseg_check_safety() runs every check in one call, and is what a vendor
 * integration that gates on all of them uses. Each check is also callable on
 * its own, because a platform may enforce some of them in hardware and want
 * only the rest: nothing here assumes the others ran.
 *
 * The PNS model is injected by the caller (pulseg_pns_model), so no
 * vendor-proprietary nerve-stimulation formula lives in this library.
 *
 * The PNS and mechanical-resonance checks derive their answers from the same
 * kind of preprocessing -- gradient waveforms extracted over a window of the
 * canonical TR. A pulseg_check_plan holds that work so a caller asking
 * several questions of one sequence does not pay for it repeatedly; passing
 * NULL instead is always allowed and simply keeps the work private to the
 * call.
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
    /*  Shared preprocessing                                              */
    /* ================================================================== */

    /**
     * @brief Create a plan the gradient checks can reuse work through.
     *
     * A plan belongs to one collection and holds two things: the sequence's
     * repetitions grouped by the set of gradient shapes they play, and the
     * uniform-raster gradient waveforms extracted over each window a check
     * asks about.  Both are built on first request, never up front.
     *
     * Reuse is what a plan is for, and it pays wherever the same window is
     * asked for twice -- a check followed by its plotting counterpart, or a
     * check re-run against a different band table or PNS threshold.  The
     * windows the PNS and mechanical-resonance checks evaluate are not the
     * same windows, so running those two back to back does not itself reuse
     * anything.
     *
     * The plan must not outlive @p coll.
     *
     * @param[out] out     Receives the plan; free with
     *                     pulseg_check_plan_destroy().
     * @param[out] diag    Diagnostic on failure; may be NULL.
     * @param[in]  coll    Collection the plan is built against.
     * @param[in]  config  What the plan may keep; NULL selects the defaults.
     * @return PULSEG_SUCCESS or a negative error code.
     */
    int pulseg_check_plan_create(
        pulseg_check_plan **out,
        pulseg_diagnostic *diag,
        const pulseg_collection *coll,
        const pulseg_check_plan_config *config);

    /** @brief Free a plan and everything it holds.  Safe on NULL. */
    void pulseg_check_plan_destroy(pulseg_check_plan *plan);

    /* ================================================================== */
    /*  Safety checks (detect violation and return immediately)           */
    /* ================================================================== */

    /**
     * @brief Run every gradient safety check.
     *
     * Raster alignment, gradient amplitude, continuity across block
     * boundaries, slew rate, mechanical resonance and PNS, in that order,
     * stopping at the first violation with a diagnostic naming it.  Does NOT
     * track worst-case.
     *
     * A sequence with no gradient event anywhere is checked for raster
     * alignment and then accepted: every other check reads gx/gy/gz.
     *
     * @param[in]     coll   Collection (non-const: cursor dry-run).
     * @param[out]    diag   Diagnostic on violation.
     * @param[in,out] plan   Shared preprocessing; NULL keeps it private to
     *                       this call.
     * @param[in]     opts   Scanner limits.
     * @param[in]     bands  Forbidden acoustic bands; NULL or an empty list
     *                       skips the mechanical-resonance check.
     * @param[in]     pns_model              PNS evaluator; NULL skips PNS.
     * @param[in]     pns_threshold_percent  PNS threshold (100 = 100 %).
     * @return PULSEG_SUCCESS if safe, negative error code on violation.
     */
    int pulseg_check_safety(
        pulseg_collection *coll,
        pulseg_diagnostic *diag,
        pulseg_check_plan *plan,
        const pulseg_opts *opts,
        const pulseg_forbidden_band_list *bands,
        const pulseg_pns_model *pns_model,
        float pns_threshold_percent);

    /**
     * @brief Check that no gradient exceeds the amplitude limit.
     *
     * The quantity compared is the vector magnitude of the unrotated
     * waveform, which is what bounds every axis component the amplifiers see
     * under an arbitrary rotation.
     *
     * @param[in]  coll  Collection to check.
     * @param[out] diag  Diagnostic naming the subsequence and block on
     *                   violation.
     * @param[in]  opts  Scanner limits; max_grad_hz_per_m is the one that
     *                   decides.
     * @return PULSEG_SUCCESS or PULSEG_ERR_GRAD_AMPLITUDE_EXCEEDED.
     */
    int pulseg_check_max_grad(
        const pulseg_collection *coll,
        pulseg_diagnostic *diag,
        const pulseg_opts *opts);

    /**
     * @brief Check that no gradient exceeds the slew-rate limit.
     *
     * Judged per block instance, from the amplitude that instance plays and
     * the normalised slew of the shape it names, and compared as a vector
     * magnitude for the same reason pulseg_check_max_grad() does.
     *
     * This bounds the slew a single event demands.  The step *between* two
     * adjacent events is what pulseg_check_grad_continuity() judges.
     *
     * @param[in]  coll  Collection to check.
     * @param[out] diag  Diagnostic naming the subsequence and block on
     *                   violation.
     * @param[in]  opts  Scanner limits; max_slew_hz_per_m_per_s decides.
     * @return PULSEG_SUCCESS or PULSEG_ERR_SLEW_RATE_EXCEEDED.
     */
    int pulseg_check_max_slew(
        const pulseg_collection *coll,
        pulseg_diagnostic *diag,
        const pulseg_opts *opts);

    /**
     * @brief Check the PNS response against a threshold.
     *
     * Evaluates the injected model over the canonical TR, once per distinct
     * set of gradient shapes the repetitions play, and compares the peak
     * combined response against @p threshold_percent.
     *
     * @param[in]     coll   Collection to check.
     * @param[out]    diag   Diagnostic on violation.
     * @param[in,out] plan   Shared preprocessing; NULL keeps it private.
     * @param[in]     opts   Scanner limits.
     * @param[in]     model  PNS evaluator; NULL accepts without checking.
     * @param[in]     threshold_percent  Threshold (100 = 100 %).
     * @return PULSEG_SUCCESS or PULSEG_ERR_PNS_THRESHOLD_EXCEEDED.
     */
    int pulseg_check_pns(
        const pulseg_collection *coll,
        pulseg_diagnostic *diag,
        pulseg_check_plan *plan,
        const pulseg_opts *opts,
        const pulseg_pns_model *model,
        float threshold_percent);

    /**
     * @brief Check that no gradient harmonic falls in a forbidden band.
     *
     * Evaluates the canonical TR's harmonic lines, bounded over every
     * instance of that TR, and refuses any line inside a band -- widened by a
     * guard of half the narrowest band's width -- whose per-axis amplitude
     * exceeds what that band allows.
     *
     * @param[in]     coll   Collection to check.
     * @param[out]    diag   Diagnostic naming frequency, amplitude and axis.
     * @param[in,out] plan   Shared preprocessing; NULL keeps it private.
     * @param[in]     opts   Scanner limits and peak-detection parameters.
     * @param[in]     bands  Forbidden bands; NULL or empty accepts without
     *                       checking.
     * @return PULSEG_SUCCESS or PULSEG_ERR_MECH_RESONANCES_VIOLATION.
     */
    int pulseg_check_mech_resonances(
        const pulseg_collection *coll,
        pulseg_diagnostic *diag,
        pulseg_check_plan *plan,
        const pulseg_opts *opts,
        const pulseg_forbidden_band_list *bands);

    /**
     * @brief Check that gradients are continuous across every block boundary.
     *
     * Also run as part of pulseg_check_safety(); exposed separately so a
     * design tool can ask this one question offline, without a PNS model or
     * an acoustic band table.  Building a sequence does not check it -- the
     * cost would fall on every add_block -- so this is what a writer or an
     * authoring session calls instead.
     *
     * The endpoints compared are those of the shape each block instance
     * actually plays, scaled by that instance's own amplitude, and both are
     * transformed by the block's rotation first: the comparison is per axis
     * in the physical frame, which is the frame the amplifiers slew in.  A
     * step larger than max_slew * grad_raster between two adjacent raster
     * points is a violation, as is a subsequence ending without ramping to
     * zero.
     *
     * @param[in]  coll  Collection (non-const: cursor dry-run).
     * @param[out] diag  Diagnostic naming the axis and block on violation.
     * @param[in]  opts  Scanner limits; max_slew_hz_per_m_per_s is the one
     *                   that decides.
     * @return PULSEG_SUCCESS if continuous, PULSEG_ERR_GRAD_DISCONTINUITY if
     *         not.
     */
    int pulseg_check_grad_continuity(
        pulseg_collection *coll,
        pulseg_diagnostic *diag,
        const pulseg_opts *opts);

    /**
     * @brief Check that every event time lands on the raster it is played on.
     *
     * Also run as part of pulseg_check_safety(); exposed separately so a
     * design tool can ask it offline against design limits rather than a
     * scanner's.  @p opts supplies the four rasters the times are judged
     * against -- online those are the scanner's own.
     *
     * The rasters a file declares are checked against @p opts elsewhere, when
     * the collection is built.  That comparison accepts either direction, so a
     * sequence laid out on a raster finer than the scanner's passes it while
     * still holding times the hardware cannot address; this is the check that
     * catches those.  RF and ADC start times go against the RF raster, ADC
     * dwell against the ADC raster, gradient delays and trapezoid ramps
     * against the gradient raster, block durations against the block duration
     * raster.  An arbitrary event's own time shape is checked sample by
     * sample.
     *
     * Also reports an event that ends after its block does
     * (PULSEG_ERR_BLOCK_DURATION_OVERRUN).  A block longer than its events is
     * legal and silent.
     *
     * RF dead time and ringdown, and ADC dead time, are deliberately not
     * checked: they bound what a transmit chain can do rather than what the
     * sequencer can address, and are the design side's to enforce.
     *
     * Every time field lives in a deduplicated definition, so the cost is the
     * number of distinct events in the scan, not the number of blocks.
     *
     * @param[in]  coll  Collection to check.
     * @param[out] diag  Diagnostic naming the event and value on violation.
     * @param[in]  opts  The rasters to judge against.
     * @return PULSEG_SUCCESS, PULSEG_ERR_RASTER_ALIGNMENT or
     *         PULSEG_ERR_BLOCK_DURATION_OVERRUN.
     */
    int pulseg_check_raster_alignment(
        const pulseg_collection *coll,
        pulseg_diagnostic *diag,
        const pulseg_opts *opts);

    /* ================================================================== */
    /*  Spectra and waveforms (what a check decides on, for plotting)     */
    /* ================================================================== */

    /**
     * @brief Compute the mechanical-resonance spectra of one canonical TR.
     *
     * The data behind pulseg_check_mech_resonances(): spectrograms, full-TR
     * spectra, sequence-level harmonics and the candidate lines, with the
     * display products a plot needs and the verdict does not.
     *
     * Pass the same @p plan the check used and the extraction is not repeated.
     *
     * @param[in]     coll     Loaded collection.
     * @param[out]    spectra  Receives the spectral data; free with
     *                         pulseg_mech_resonances_spectra_free().
     * @param[out]    diag     Diagnostic on failure.
     * @param[in,out] plan     Shared preprocessing; NULL keeps it private.
     * @param[in]     opts     Scanner limits and peak-detection parameters.
     * @param[in]     request  What to analyse, and how finely.
     * @return PULSEG_SUCCESS on success, negative error code on failure.
     */
    int pulseg_calc_mech_resonances(
        const pulseg_collection *coll,
        pulseg_mech_resonances_spectra *spectra,
        pulseg_diagnostic *diag,
        pulseg_check_plan *plan,
        const pulseg_opts *opts,
        const pulseg_mech_resonances_request *request);

    /** @brief Free arrays inside a pulseg_mech_resonances_spectra. */
    void pulseg_mech_resonances_spectra_free(pulseg_mech_resonances_spectra *s);

    /**
     * @brief Compute the PNS slew-rate waveforms of one canonical TR.
     *
     * The data behind pulseg_check_pns(): per-axis slew rates, from which the
     * combined response is sqrt(x^2 + y^2 + z^2) and the threshold percentage
     * follows.
     *
     * A @p canonical_tr_idx of 0 asks for the worst case, which is the worst
     * over every set of shapes the repetitions play.  Any other index asks
     * for that one window, on its own shapes.
     *
     * Pass the same @p plan the check used and the extraction is not repeated.
     *
     * @param[in]     coll    Loaded collection.
     * @param[out]    result  Receives the slew-rate waveforms; free with
     *                        pulseg_pns_result_free().
     * @param[out]    diag    Diagnostic on failure.
     * @param[in,out] plan    Shared preprocessing; NULL keeps it private.
     * @param[in]     subseq_idx        Subsequence index.
     * @param[in]     canonical_tr_idx  Canonical TR index within it.
     * @param[in]     opts    Scanner limits.
     * @param[in]     model   PNS evaluator.
     * @return PULSEG_SUCCESS on success, negative error code on failure.
     */
    int pulseg_calc_pns(
        const pulseg_collection *coll,
        pulseg_pns_result *result,
        pulseg_diagnostic *diag,
        pulseg_check_plan *plan,
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
