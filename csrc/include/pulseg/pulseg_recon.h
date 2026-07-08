/**
 * @file pulseg_recon.h
 * @brief Standalone ANSI-C reader for the recon-relevant sections of a
 *        pulseg binary cache (.pseg/.pge).
 *
 * Stage 4 of the vendor-neutral refactor: C89 port of the former
 * cxx/recon/trajectory_cache_reader.cpp byte-parsing logic. Reads directly
 * from the cache file -- no loaded pulseg_collection required -- so it is
 * usable by any recon consumer (C, or the thin cxx/recon C++ wrapper) that
 * only has the cache file on disk:
 *   - Section 0 (DEFINITIONS): per-subsequence generic [DEFINITIONS] kv
 *     (FOV/Matrix/NavFOV/NavMatrix/TR/TE/TI/FlipAngle/... as pulseq
 *     strings). Absent if the cache predates this section (tolerated).
 *   - Section 6 (TRAJECTORY): kshot library, encoding spaces, table, and a
 *     folded-in rotation-matrix library. Delegates to the existing public
 *     reader pulseg_load_trajectory_cache() (pulseg_trajectory.h) -- do not
 *     duplicate that byte-parsing here.
 *   - Section 7 (SEQDESC): per-subsequence event rows + a per-subsequence
 *     RF-definition library (bandwidth/bands/b1sq + still-compressed
 *     mag/phase/time shapes), plus the deduped RF-shape-tuple derivation
 *     (rf_def_id, rf_shim_id, ss_grad_amp_hz_per_m) with slice-thickness/
 *     slice-selectivity classification.
 *
 * TRAJECTORY and SEQDESC are both MANDATORY as of cache format v2.0.0
 * (master plan D11: all cache sections are written unconditionally by
 * pulseg__write_cache()) -- a missing or malformed section is a hard read
 * error via the return code, never a silent degrade. Do not add tolerance
 * for an absent SEQDESC; a cache lacking it is a fixture/writer bug.
 *
 * Section 5 (FREQMOD / off-isocenter shift) is intentionally NOT parsed:
 * frequency modulation is applied PSD-side, so data arriving at recon are
 * already centered -- this is a permanent design choice, not a workaround
 * for section-write ordering. See tests/utils/SCHEMA.md.
 *
 * All integer and float fields in the cache are 4 bytes; endianness is
 * resolved via the file's marker word (matches pulseg_trajectory.c).
 */

#ifndef PULSEG_RECON_H
#define PULSEG_RECON_H

#include "pulseg_config.h"
#include "pulseg_types.h"
#include "pulseg_io.h"
#include "pulseg_trajectory.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /* ================================================================== */
    /*  Section 0 -- per-subsequence [DEFINITIONS]                        */
    /* ================================================================== */

    /** @brief One subsequence's generic [DEFINITIONS] kv list. */
    typedef struct pulseg_recon_definitions
    {
        int num_definitions;
        pulseg__definition *definitions; /* NULL if num_definitions == 0 */
    } pulseg_recon_definitions;

    /* ================================================================== */
    /*  Section 7 -- SEQDESC                                              */
    /* ================================================================== */

    /** @brief One (still-compressed) RF waveform shape, copied verbatim.
     * Consumers that need samples decompress it themselves; this reader
     * never does (matches the SEQDESC writer contract). */
    typedef struct pulseg_recon_rf_shape
    {
        int num_uncompressed;
        int num_samples;
        float *samples; /* NULL if num_samples == 0 */
    } pulseg_recon_rf_shape;

    /** @brief Per-subsequence RF-definition library entry. rf_def_id is the
     * array index into this subsequence's library, matching
     * pulseg_seq_event::params[0] (cast to int) for RF rows. */
    typedef struct pulseg_recon_rf_def
    {
        int rf_def_id;
        float bandwidth_hz;
        int num_bands;
        float band_freq_offsets_hz[PULSEG_MAX_BANDS];
        float band_bandwidth_hz;
        float total_b1sq_power;
        pulseg_recon_rf_shape mag;
        int has_phase;
        pulseg_recon_rf_shape phase;
        int has_time;
        pulseg_recon_rf_shape time;
    } pulseg_recon_rf_def;

    /** @brief Unique (rf_def_id, rf_shim_id, ss_grad_amp) triplet, deduped
     * over a subsequence's event rows. slice_thickness_mm/slice_selective
     * are derived from the matching pulseg_recon_rf_def's bandwidth_hz. */
    typedef struct pulseg_recon_rf_shape_tuple
    {
        int tuple_id;
        int rf_def_id;
        int rf_shim_id;             /* -1 if no shim applied */
        float ss_grad_amp_hz_per_m; /* slice-selection gradient amplitude (Hz/m) */
        float slice_thickness_mm;   /* 0 if ss_grad_amp == 0 or bandwidth unknown */
        int slice_selective;        /* 1 if slice_thickness_mm < 10 mm, else 0 */
    } pulseg_recon_rf_shape_tuple;

    /** @brief Per-subsequence sequence description: event rows (reuses the
     * existing public pulseg_seq_event row type), RF-definition library,
     * and the derived RF-shape-tuple dedup. */
    typedef struct pulseg_recon_seq_desc
    {
        int subseq_idx;
        float tr_duration_us;
        int num_events;
        pulseg_seq_event *events;
        int num_rf_defs;
        pulseg_recon_rf_def *rf_defs;
        int num_rf_shape_tuples;
        pulseg_recon_rf_shape_tuple *rf_shape_tuples;
    } pulseg_recon_seq_desc;

    /* ================================================================== */
    /*  Top-level recon cache                                             */
    /* ================================================================== */

    /** @brief Everything a recon consumer needs from a .pseg/.pge file. */
    typedef struct pulseg_recon_cache
    {
        /* Section 0 (optional -- absent on caches written before this
         * section existed; num_subsequences == 0 in that case). */
        int num_subsequences;
        pulseg_recon_definitions *definitions_by_subseq; /* len == num_subsequences */

        /* Section 6 (mandatory). */
        pulseg_trajectory trajectory;

        /* Section 7 (mandatory). */
        pulseg_sequence_parameters seq_params;
        int num_seq_descs;
        pulseg_recon_seq_desc *seq_descs; /* len == num_seq_descs */
    } pulseg_recon_cache;

    /**
     * @brief Read the recon-relevant sections from a pulseg binary cache.
     *
     * @param[out] out         Populated on success; caller-owned, zeroed by
     *                          this call before reading (no need to
     *                          pre-zero, but any prior contents are leaked
     *                          -- free with pulseg_recon_cache_free() first
     *                          if reusing a struct).
     * @param[in]  cache_path  Path to the .pseg/.pge cache file.
     * @param[out] diag        Optional caller buffer for a human-readable
     *                          diagnostic message (may be NULL). Always
     *                          NUL-terminated when non-NULL and
     *                          diag_size > 0.
     * @param[in]  diag_size   Size in bytes of @p diag.
     * @return PULSEG_SUCCESS on success, negative PULSEG_ERR_* on failure
     *         (missing file, bad header, or a missing/truncated mandatory
     *         section).
     */
    int pulseg_recon_cache_read(pulseg_recon_cache *out,
                                 const char *cache_path,
                                 char *diag,
                                 int diag_size);

    /** @brief Free all memory owned by a pulseg_recon_cache. Safe to call
     * on a zero-initialised or partially-populated struct. */
    void pulseg_recon_cache_free(pulseg_recon_cache *cache);

    /**
     * @brief Look up a single-subsequence [DEFINITIONS] entry by key.
     *
     * Linear scan -- per-subsequence definition counts are small enough
     * that a hash map would be pure overhead (no std::map equivalent in a
     * C89 module).
     *
     * @return Pointer to the matching entry (owned by @p cache), or NULL
     *         if subseq_idx is out of range or no definition named @p name
     *         exists for it.
     */
    const pulseg__definition *pulseg_recon_find_definition(
        const pulseg_recon_cache *cache,
        int subseq_idx,
        const char *name);

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_RECON_H */
