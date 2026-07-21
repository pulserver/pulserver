/**
 * @file pulseg_freqmod.h
 * @brief Per-block frequency-modulation waveforms for FOV shift and motion
 *        correction.
 *
 * A spatial shift becomes a phase ramp along the gradient moment, played as a
 * hardware frequency-modulation waveform. The shift-independent base is built
 * once (and cached); the shift-dependent plans are recomputed whenever the
 * shift changes -- per subsequence, or per TR under prospective motion
 * correction (PMC).
 */

#ifndef PULSEG_FREQMOD_H
#define PULSEG_FREQMOD_H

#include <stdio.h>
#include "pulseg_config.h"
#include "pulseg_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * @brief Build frequency modulation data for all subsequences.
     *
     * Constructs deduped amplitude-scaled 3-channel gradient modulators
     * and computes shift-resolved 1D plan waveforms for every subsequence
     * in the collection.
     *
     * For PMC-enabled subsequences the 3-channel data is retained so that
     * pulseg_update_freq_mod_collection() can recompute waveforms with
     * a new shift at each TR boundary.  For non-PMC subsequences the
     * 3-channel data is discarded after the initial plan computation to
     * save memory.
     *
     * @param[out] out_fmc       Receives an allocated collection (caller frees).
     * @param[in]  coll          Loaded sequence collection.
     * @param[in]  shift_m       Spatial shift (dx, dy, dz) in metres.
     * @param[in]  fov_rotation  FOV rotation matrix (3x3, row-major,
     *                           logical-to-physical).  Used to correct
     *                           frequency modulation for blocks flagged
     *                           with @c norot.  Pass NULL for identity.
     * @return PULSEG_SUCCESS on success.
     */
    int pulseg_build_freq_mod_collection(
        pulseg_freq_mod_collection **out_fmc,
        const pulseg_collection *coll,
        const float *shift_m,
        const float *fov_rotation);

    /**
     * @brief Look up the freq-mod waveform for a scan-table position.
     *
     * @param[in]  fmc             Freq-mod collection.
     * @param[in]  subseq_idx     0-based subsequence index.
     * @param[in]  exec_stream_pos Position in the subsequence scan table.
     * @param[out] out_hw_waveform Pointer to short DAC waveform (do NOT free).
     * @param[out] out_num_samples Waveform length.
     * @param[out] out_phase_rad  Phase compensation (rad) computed from the
     *                            full 3-channel definition (no axis masking).
     * @return 1 if the block has a freq-mod event, 0 if not.
     */
    int pulseg_freq_mod_collection_get(
        const pulseg_freq_mod_collection *fmc,
        const short **out_hw_waveform,
        int *out_num_samples,
        float *out_phase_rad,
        int subseq_idx,
        int exec_stream_pos);

    /** @brief Free a frequency modulation collection and all owned memory. */
    void pulseg_freq_mod_collection_free(pulseg_freq_mod_collection *fmc);

    /* ================================================================== */
    /*  Collection-level freq-mod wrappers                                */
    /* ================================================================== */

    /**
     * @brief Recompute freq-mod waveforms for one subsequence.
     *
     * Forwards to coll->freq_mod.  Only valid for PMC-enabled subsequences
     * where 3-channel data was retained.
     */
    int pulseg_update_freq_mod(pulseg_collection *coll, int subseq_idx, const float *shift_m);

    /**
     * @brief Fetch freq-mod waveform for a scan-table position.
     *
     * @return 1 if freq-mod data is available; 0 otherwise.
     */
    int pulseg_get_freq_mod(
        const pulseg_collection *coll,
        const short **out_hw_waveform,
        int *out_num_samples,
        float *out_phase_rad,
        int subseq_idx,
        int exec_stream_pos);

    /**
     * @brief Targeted per-TR freq-mod update (PMC mode).
     *
     * Recomputes plan waveforms and phase ONLY for plan instances that
     * are referenced by scan-table positions in [tr_scan_start,
     * tr_scan_start + tr_scan_count).  Cost: O(plans_in_TR * max_samples).
     *
     * Requires retained 3-channel data (PMC-enabled subsequence).
     */
    int pulseg_update_freq_mod_for_tr(
        pulseg_collection *coll,
        int subseq_idx,
        int current_scan_pos,
        const float *shift_m);

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_FREQMOD_H */
