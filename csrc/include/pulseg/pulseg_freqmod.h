/**
 * @file pulseg_freqmod.h
 * @brief Frequency-modulation collection build/update/cache.
 *
 * Split out of the former pulseg_methods.h (Stage 1 layout normalization).
 * All functions use the pulseg_ prefix and are declared extern "C" when
 * compiled with a C++ compiler.
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
     * @brief Recompute freq-mod waveforms for one subsequence.
     *
     * Only valid for PMC-enabled subsequences (3-channel data is still
     * resident).  Returns an error if the 3-channel data was freed.
     *
     * @param[in,out] fmc         Freq-mod collection.
     * @param[in]     subseq_idx  0-based subsequence index.
     * @param[in]     shift_m     New spatial shift (dx, dy, dz) in metres.
     * @return PULSEG_SUCCESS on success.
     */
    int pulseg_update_freq_mod_collection(
        pulseg_freq_mod_collection *fmc,
        int subseq_idx,
        const float *shift_m);

    /**
     * @brief Look up the freq-mod waveform for a scan-table position.
     *
     * @param[in]  fmc             Freq-mod collection.
     * @param[in]  subseq_idx     0-based subsequence index.
     * @param[in]  scan_table_pos Position in the subsequence scan table.
     * @param[out] out_hw_waveform Pointer to short DAC waveform (do NOT free).
     * @param[out] out_num_samples Waveform length.
     * @param[out] out_phase_rad  Phase compensation (rad) computed from the
     *                            full 3-channel definition (no axis masking).
     * @return 1 if the block has a freq-mod event, 0 if not.
     */
    int pulseg_freq_mod_collection_get(const pulseg_freq_mod_collection *fmc,
                                          const short **out_hw_waveform,
                                          int *out_num_samples,
                                          float *out_phase_rad,
                                          int subseq_idx,
                                          int scan_table_pos);

    /**
     * @brief Write all per-subsequence freq-mod data to a single cache file.
     *
     * @param[in]  fmc   Built collection (3-channel data must be resident).
     * @param[in]  path  Output file path (e.g. "seq.fmod.pseg").
     * @return PULSEG_SUCCESS on success.
     */
    int pulseg_freq_mod_collection_write_cache(
        const pulseg_freq_mod_collection *fmc,
        const char *path);

    /**
     * @brief Write freq-mod collection data to an already-open FILE.
     *
     * Used by the unified cache writer to embed freq-mod as a section.
     */
    int pulseg_freq_mod_collection_write_cache_f(
        const pulseg_freq_mod_collection *fmc,
        FILE *f);

    /**
     * @brief Read freq-mod collection from cache and compute plans.
     *
     * @param[out] out_fmc     Receives an allocated collection (caller frees).
     * @param[in]  path        Cache file path.
     * @param[in]  coll        Loaded sequence collection (provides PMC flags
     *                          and subsequence count).
     * @param[in]  shift_m     Spatial shift for plan computation.
     * @return PULSEG_SUCCESS on success.
     */
    int pulseg_freq_mod_collection_read_cache(
        pulseg_freq_mod_collection **out_fmc,
        const char *path,
        const pulseg_collection *coll,
        const float *shift_m);

    /**
     * @brief Read freq-mod collection data from an already-open FILE.
     *
     * Used by the unified cache loader to read freq-mod from a section.
     */
    int pulseg_freq_mod_collection_read_cache_f(
        pulseg_freq_mod_collection **out_fmc,
        FILE *f,
        const pulseg_collection *coll,
        const float *shift_m);

    /** @brief Free a frequency modulation collection and all owned memory. */
    void pulseg_freq_mod_collection_free(pulseg_freq_mod_collection *fmc);

    /* ================================================================== */
    /*  Collection-level freq-mod wrappers                                */
    /* ================================================================== */

    /**
     * @brief Build freq-mod library and store in coll->freq_mod.
     *
     * Replaces any previously stored freq-mod collection.  For non-PMC
     * sequences the 3-channel data is freed after construction.
     */
    int pulseg_build_freq_mod(
        pulseg_collection *coll,
        const float *shift_m,
        const float *fov_rotation);

    /**
     * @brief Recompute freq-mod waveforms for one subsequence.
     *
     * Forwards to coll->freq_mod.  Only valid for PMC-enabled subsequences
     * where 3-channel data was retained.
     */
    int pulseg_update_freq_mod(
        pulseg_collection *coll,
        int subseq_idx,
        const float *shift_m);

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
        int scan_table_pos);

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

    /**
     * @brief Build the shift-independent freq-mod base into @p coll and append
     *        the FREQMOD section to the collection cache.
     *
     * Convenience wrapper used by the unified cache dump. Builds the base with
     * a zero shift and identity rotation (only the base, which is
     * shift-independent, is cached), then appends the FREQMOD section.
     *
     * @param[in,out] coll      Loaded collection (coll->freq_mod is populated).
     * @param[in]     seq_path  Path to the .seq file (cache extension per D10: default .pseg, GE .pge).
     * @return PULSEG_SUCCESS or negative error code.
     */
    int pulseg_write_freq_mod_cache_from_collection(
        pulseg_collection *coll,
        const char *seq_path);

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_FREQMOD_H */
