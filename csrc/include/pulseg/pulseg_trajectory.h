/**
 * @file pulseg_trajectory.h
 * @brief The k-space trajectory: a library of unique per-axis ADC-sampled
 *        shots plus a per-ADC table of shot ids, amplitudes, rotations and
 *        labels, persisted as the TRAJECTORY cache section so the recon side
 *        can read it back without the original .seq file.
 *
 * Per-TR waveform extraction (gradient/RF/ADC getters for safety and plotting)
 * is a separate concern -- see pulseg_waveforms.h.
 */

#ifndef PULSEG_TRAJECTORY_H
#define PULSEG_TRAJECTORY_H

#include "pulseg_config.h"
#include "pulseg_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /** @brief Free all memory owned by a pulseg_trajectory. */
    void pulseg_trajectory_free(pulseg_trajectory *traj);

    /**
     * @brief Load trajectory from the TRAJECTORY cache section, given the
     *        cache file path directly (no .seq -> cache-path derivation).
     *
     * For standalone consumers that only have
     * the .pseg/.pge cache file on disk, not the original .seq path.
     *
     * @param[out] out         Trajectory output (caller-allocated struct).
     * @param[in]  cache_path  Path to the .pseg/.pge cache file.
     * @return PULSEG_SUCCESS or negative error code.
     */
    int pulseg_load_trajectory_cache(pulseg_trajectory *out, const char *cache_path);

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_TRAJECTORY_H */
