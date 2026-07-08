/**
 * @file pulseg_trajectory.h
 * @brief TR gradient/waveform extraction and k-space trajectory computation.
 *
 * Split out of the former pulseg_methods.h (Stage 1 layout normalization).
 * All functions use the pulseg_ prefix and are declared extern "C" when
 * compiled with a C++ compiler.
 */

#ifndef PULSEG_TRAJECTORY_H
#define PULSEG_TRAJECTORY_H

#include "pulseg_config.h"
#include "pulseg_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Extract uniform-raster canonical TR gradient waveforms for a given subsequence.
     *
     * Returns canonical TR gradient waveforms (gx, gy, gz) for the requested
     * canonical TR index within the specified subsequence. The canonical TRs are defined as unique shot-index
     * patterns across the imaging region (for degenerate prep/cooldown) or unique
     * pass patterns (for non-degenerate prep/cooldown).
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
    int pulseg_get_tr_gradient_waveforms(const pulseg_collection *coll,
                                            pulseg_tr_gradient_waveforms *waveforms,
                                            pulseg_diagnostic *diag,
                                            int subseq_idx,
                                            int canonical_tr_idx);

    /* ================================================================== */
    /*  TR gradient waveforms (for plotting)                              */
    /* ================================================================== */

    /**
     * @brief Extract per-axis gradient waveforms for a single canonical TR.
     *
     * Returns waveforms in their native (non-interpolated) timing:
     * each axis carries its own time base as (time, amplitude,
     * segment_label) tuples.  This is suitable for wrapper-side
     * gradient-shape plotting with segment colour-coding.
     *
     * Safety and acoustic/PNS functions do NOT call this function;
     * they use an internal variant that skips segment-label
     * computation and then interpolate to uniform raster.
     *
     * For multishot sequences (e.g. 3-D non-Cartesian) the collection
     * may contain multiple unique canonical TRs — one per unique
     * shot-ID combination (pass pattern).  The number of valid indices
     * equals the number of unique pass patterns returned by
     * pulseg__find_unique_shot_passes (non-degenerate prep/cooldown)
     * or pulseg__find_unique_shot_trs (degenerate).
     *
     * @param[in]  coll             Loaded collection.
     * @param[in]  canonical_tr_idx Zero-based canonical TR index.
     *                              Returns PULSEG_ERR_INVALID_ARGUMENT
     *                              when the index is out of range.
     * @param[out] waveforms        Receives the waveform data (caller frees
     *                              via pulseg_tr_gradient_waveforms_free).
     * @param[out] diag             Diagnostic on failure.
     * @return PULSEG_SUCCESS on success, negative error code on failure.
     */

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
    int pulseg_get_tr_waveforms(const pulseg_collection *coll,
                                   pulseg_tr_waveforms *out,
                                   pulseg_diagnostic *diag,
                                   int subseq_idx,
                                   int amplitude_mode,
                                   int tr_index,
                                   int collapse_delays,
                                   int num_averages);

    /** @brief Free all arrays inside a pulseg_tr_waveforms. */
    void pulseg_tr_waveforms_free(pulseg_tr_waveforms *w);

    /* ================================================================== */
    /*  K-space trajectory                                                */
    /* ================================================================== */

    /**
     * @brief Compute the k-space trajectory for a subsequence.
     *
     * Builds a library of unique per-axis k-space shots (ADC-sampled,
     * k-zero centred) and a per-ADC-event table with shot IDs, gradient
     * amplitudes, rotation IDs, and resolved labels.
     *
     * Requires the segment timing anchors (k-zero) that
     * pulseg__calc_segment_timing populates at parse; no safety pass is
     * required.
     *
     * @param[in]  coll        Loaded collection.
     * @param[out] out         Trajectory output (caller-allocated struct).
     * @param[out] diag        Diagnostic (optional, may be NULL).
     * @param[in]  subseq_idx  Subsequence index.
     * @return PULSEG_SUCCESS or negative error code.
     */
    int pulseg_compute_trajectory(const pulseg_collection *coll,
                                     pulseg_trajectory *out,
                                     pulseg_diagnostic *diag,
                                     int subseq_idx);

    /**
     * @brief Free all memory owned by a pulseg_trajectory.
     */
    void pulseg_trajectory_free(pulseg_trajectory *traj);

    /**
     * @brief Merge one trajectory into another (append src into dst).
     *
     * Appends kshots, encoding spaces, rotation matrices, and table entries
     * from @p src into @p dst.  Kshot IDs, encoding_space_ref, and
     * rotation_id values in the appended table entries are offset so they
     * index correctly into the combined kshot library, encoding-space
     * array, and rotation-matrix library (Stage 1.5c).
     *
     * @param[in,out] dst  Destination trajectory (accumulator).
     * @param[in]     src  Source trajectory to merge (unmodified).
     * @return PULSEG_SUCCESS or negative error code.
     */
    int pulseg_merge_trajectory(pulseg_trajectory *dst,
                                   const pulseg_trajectory *src);

    /**
     * @brief Append the trajectory as the TRAJECTORY section to the binary cache.
     *
     * Opens the existing cache file (written by pulseg_read with
     * cache_binary=1), appends the trajectory section, and patches the
     * header.  Must be called AFTER pulseg_compute_trajectory().
     *
     * @param[in] traj      Computed trajectory.
     * @param[in] seq_path  Path to the .seq file (cache extension per D10).
     * @param[in] cache_ext Cache file extension incl. dot, or NULL for the
     *                      public default (PULSEG_CACHE_EXT_DEFAULT).
     * @return PULSEG_SUCCESS or negative error code.
     */
    int pulseg_write_trajectory_cache(const pulseg_trajectory *traj,
                                         const char *seq_path,
                                         const char *cache_ext);

    /**
     * @brief Compute + merge per-subsequence trajectories and append the
     *        TRAJECTORY section to the binary cache.
     *
     * Convenience wrapper used by the unified cache dump: loops every
     * subsequence in @p coll, computes its trajectory, merges into an
     * accumulator and appends the result. No-op (returns success) when the
     * collection has no subsequences. The kzero anchors it relies on are
     * populated at parse (calc_segment_timing); no safety pass is required.
     *
     * @param[in] coll      Loaded collection.
     * @param[in] seq_path  Path to the .seq file (cache extension per D10: default .pseg, GE .pge).
     * @return PULSEG_SUCCESS or negative error code.
     */
    int pulseg_write_trajectory_cache_from_collection(
        const pulseg_collection *coll,
        const char *seq_path);

    /**
     * @brief Load trajectory from the TRAJECTORY cache section.
     *
     * @param[out] out       Trajectory output (caller-allocated struct).
     * @param[in]  seq_path  Path to the .seq file.
     * @return PULSEG_SUCCESS or negative error code.
     */
    int pulseg_load_trajectory_cache(pulseg_trajectory *out,
                                        const char *seq_path);

    /**
     * @brief Load trajectory from the TRAJECTORY cache section, given the
     *        cache file path directly (no .seq -> cache-path derivation).
     *
     * For standalone recon consumers (e.g. pulseg_recon) that only have
     * the .pseg/.pge cache file on disk, not the original .seq path.
     *
     * @param[out] out         Trajectory output (caller-allocated struct).
     * @param[in]  cache_path  Path to the .pseg/.pge cache file.
     * @return PULSEG_SUCCESS or negative error code.
     */
    int pulseg_load_trajectory_cache_from_cache_path(pulseg_trajectory *out,
                                                         const char *cache_path);

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_TRAJECTORY_H */
