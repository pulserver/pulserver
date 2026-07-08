/**
 * @file pulseg_collection.h
 * @brief Collection object: load/free, diagnostics, getters, and block cursor.
 *
 * Split out of the former pulseg_methods.h (Stage 1 layout normalization).
 * All functions use the pulseg_ prefix and are declared extern "C" when
 * compiled with a C++ compiler.
 */

#ifndef PULSEG_COLLECTION_H
#define PULSEG_COLLECTION_H

#include "pulseg_config.h"
#include "pulseg_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /* ================================================================== */
    /*  Read / load                                                       */
    /* ================================================================== */

    /**
     * @brief Read a (possibly chained) Pulseq sequence from disk.
     *
     * On success the library heap-allocates the collection and writes it
     * to @p *out_coll.  The caller owns the collection and must free it
     * with pulseg_collection_free().
     *
     * @param[out] out_coll         Receives the allocated collection.
     * @param[out] diag             Diagnostic info on failure.
     * @param[in]  file_path        Path to the first .seq file.
     * @param[in]  opts             Scanner limits / rasters.
     * @param[in]  cache_binary     1 = read/write binary cache alongside .seq (extension per D10).
     * @param[in]  verify_signature 1 = verify MD5 signature for every .seq
     *                              file in the chain.
     * @param[in]  parse_labels     1 = build ADC label table via dry-run.
     * @param[in]  num_averages     Number of scan averages (>= 1).
     * @return PULSEG_SUCCESS on success, negative error code on failure.
     */
    int pulseg_read(
        pulseg_collection **out_coll,
        pulseg_diagnostic *diag,
        const char *file_path,
        const pulseg_opts *opts,
        int cache_binary,
        int verify_signature,
        int parse_labels,
        int num_averages);

    /**
     * @brief Read one or more Pulseq subsequences from in-memory buffers.
     *
     * Wrapper-friendly counterpart of pulseg_read(): the caller supplies
     * pre-read file contents (e.g.\ from a Python bytes object) and the
     * library parses them without touching the filesystem.  Caching and
     * signature verification are skipped.
     *
     * @param[out] out_coll      Receives the allocated collection.
     * @param[out] diag          Diagnostic info on failure.
     * @param[in]  buffers       Array of NUL-terminated .seq contents.
     * @param[in]  buffer_sizes  Byte length of each buffer (excl. NUL).
     * @param[in]  num_buffers   Number of buffers (>= 1).
     * @param[in]  opts          Scanner limits / rasters.
     * @param[in]  num_averages  Number of scan averages (>= 1).
     * @return PULSEG_SUCCESS on success, negative error code on failure.
     */
    int pulseg_read_from_buffers(
        pulseg_collection **out_coll,
        pulseg_diagnostic *diag,
        const char *const *buffers,
        const int *buffer_sizes,
        int num_buffers,
        const pulseg_opts *opts,
        int parse_labels,
        int num_averages);

    /* ================================================================== */
    /*  Diagnostic helpers                                                */
    /* ================================================================== */

    /** @brief Zero-initialize a diagnostic struct. */
    void pulseg_diagnostic_init(pulseg_diagnostic *diag);

    /** @brief Return a human-readable message for an error code. */
    const char *pulseg_get_error_message(int code);

    /** @brief Return a fix-suggestion hint for an error code. */
    const char *pulseg_get_error_hint(int code);

    /**
     * @brief Format error code + diagnostic into a single string.
     *
     * Writes at most @p buf_size bytes (always NUL-terminated).
     *
     * @param[out] buf       Output buffer (>= 512 bytes recommended).
     * @param[in]  buf_size  Size of @p buf.
     * @param[in]  code      Error code.
     * @param[in]  diag      Optional diagnostic (NULL to omit context).
     * @return Characters written (excluding NUL), 0 on error.
     */
    int pulseg_format_error(
        char *buf, int buf_size,
        int code,
        const pulseg_diagnostic *diag);

    /* ================================================================== */
    /*  Consistency check                                                 */
    /* ================================================================== */

    /**
     * @brief Re-run internal consistency checks on a loaded collection.
     *
     * Already called by pulseg_read / pulseg_read_from_buffers.
     * Exposed for unit-test or post-hoc validation workflows.
     *
     * @param[in]  coll  Loaded collection.
     * @param[out] diag  Diagnostic (may be NULL).
     * @return PULSEG_SUCCESS on success, negative error code on failure.
     */
    int pulseg_check_consistency(
        const pulseg_collection *coll,
        pulseg_diagnostic *diag);
    void pulseg_collection_free(pulseg_collection *coll);

    /**
     * @brief Heap-allocate and zero-initialize an empty collection, ready to
     * be populated by pulseg_convert_collection() (Stage 3). This is the same
     * allocation pulseg_read() / pulseg_read_from_buffers() perform
     * internally; exposed so external producers of pulseg_pulseq_file (e.g.
     * the ExternalSequence adapter, cxx/pulseq_adapter) that call
     * pulseg_convert_collection() directly don't need to know
     * pulseg_collection's (intentionally opaque) internal layout.
     * @return A freshly allocated collection, or NULL on allocation failure.
     *         Free with pulseg_collection_free().
     */
    pulseg_collection *pulseg_collection_alloc(void);

    /* ================================================================== */
    /*  Subsequence getters                                               */
    /* ================================================================== */

    /**
     * @brief Fill a pulseg_collection_info with collection-level summary.
     *
     * Replaces pulseg_get_num_subsequences, pulseg_get_num_segments,
     * pulseg_get_max_adc_samples, pulseg_get_total_readouts,
     * pulseg_get_total_duration_us.
     */
    int pulseg_get_collection_info(const pulseg_collection *coll,
                                      pulseg_collection_info *info);

    /**
     * @brief Fill a pulseg_subseq_info for one subsequence.
     *
     * Replaces ~18 individual per-subsequence getters (TR structure,
     * prep/cooldown counts, degenerate flags, segment counts, label info).
     */
    int pulseg_get_subseq_info(const pulseg_collection *coll,
                                  pulseg_subseq_info *info,
                                  int subseq_idx);

    /**
     * @brief Fill a pulseg_segment_info for one segment.
     *
     * Replaces ~11 individual per-segment getters (duration, blocks,
     * trigger, NAV, timing gaps).
     */
    int pulseg_get_segment_info(const pulseg_collection *coll,
                                   pulseg_segment_info *info,
                                   int seg_idx);

    /**
     * @brief Return 1 if any block in the segment has X/Y/Z gradient, else 0.
     */
    int pulseg_segment_has_grad(const pulseg_collection *coll,
                                   int seg_idx);

    /**
     * @brief Fill a pulseg_block_info for one block within a segment.
     *
     * Replaces all block-level has_xxx / get_xxx accessor pairs.
     * Waveform data is NOT included; use the dedicated waveform getters
     * keyed by metadata from this struct.
     */
    int pulseg_get_block_info(const pulseg_collection *coll,
                                 pulseg_block_info *info,
                                 int seg_idx,
                                 int blk_idx);

    /**
     * @brief Fill a pulseg_adc_def for a unique ADC definition.
     *
     * @p adc_idx is a global index across all subsequences (same as
     * block_info.adc_def_id).
     */
    int pulseg_get_adc_def(const pulseg_collection *coll,
                              pulseg_adc_def *def,
                              int adc_idx);

    /**
     * @brief Fill a pulseg_rf_shim_def for one RF shim definition.
     *
     * @p shim_idx is the rf_shim_id from pulseg_block_instance.  It is
     * LOCAL to the given @p subseq_idx (same convention as rf_id, gx_id, etc.);
     * each subsequence stores its own shim table starting at index 0.
     * Returns PULSEG_ERR_INDEX if either index is out of range.
     */
    int pulseg_get_rf_shim_def(const pulseg_collection *coll,
                                  pulseg_rf_shim_def *def,
                                  int subseq_idx,
                                  int shim_idx);

    /**
     * @brief Return the number of RF shim definitions in a subsequence.
     */
    int pulseg_get_num_rf_shims(const pulseg_collection *coll,
                                   int subseq_idx);

    /**
     * @brief Check if a block needs frequency modulation.
     *
     * Returns 1 if the block requires freq-mod: the block must have RF or ADC,
     * must NOT have the nopos flag set, and at least one gradient axis must have
     * nonzero amplitude within the RF or ADC temporal window (overlap check).
     *
     * For trapezoid gradients the flat region is tested; for arbitrary gradients
     * the decompressed waveform samples within the window are checked.
     *
     * If @p num_samples is non-NULL and the function returns 1, the number of
     * freq-mod samples (block_duration / raster) is written.  The raster used
     * is rf_raster_us when triggered by an RF overlap, or adc_raster_us when
     * triggered by an ADC overlap.
     */
    int pulseg_block_needs_freq_mod(const pulseg_collection *coll,
                                       int *num_samples,
                                       int seg_idx,
                                       int blk_idx);

    /**
     * @brief Return the RF isocenter time (us) relative to segment start.
     *
     * Looks up the segment timing RF anchor matching @p blk_idx.
     * Returns -1.0f if the block has no RF anchor.
     */
    float pulseg_get_rf_isocenter_us(
        const pulseg_collection *coll,
        int seg_idx, int blk_idx);

    /**
     * @brief Return the ADC k-zero time (us) relative to segment start.
     *
     * Looks up the segment timing ADC anchor matching @p blk_idx.
     * Returns -1.0f if the block has no ADC anchor.
     */
    float pulseg_get_adc_kzero_us(
        const pulseg_collection *coll,
        int seg_idx, int blk_idx);

    /**
     * @brief Compute scan-time info from a fully loaded collection.
     *
     * Uses the accurate formula that accounts for prep/cooldown block
     * durations, degenerate TR folding, and the number-of-averages
     * multiplier (controlled by @c IgnoreAverages per subsequence).
     *
     * @c num_reps is the number of repetitions the consumer intends to
     * play (>= 1).  For subsequences whose @c IgnoreAverages flag is
     * set, the multiplier is clamped to 1.
     *
     * Both @c total_duration_us and @c total_segment_boundaries are
     * populated.
     *
     * @param[in]  coll      Loaded collection.
     * @param[in]  num_reps  Number of repetitions (>= 1).
     * @param[out] info      Receives scan time summary.
     * @return PULSEG_SUCCESS on success, negative error code on failure.
     */
    int pulseg_get_scan_time(const pulseg_collection *coll,
                                pulseg_scan_time_info *info,
                                int num_reps);

    /* ================================================================== */
    /*  Segment table getters (copy to caller buffer)                     */
    /* ================================================================== */

    /**
     * @brief Copy prep segment IDs into caller-supplied buffer.
     * @param[out] out_ids   Buffer of at least num_prep_segments ints.
     * @return Number of IDs written, or negative error code.
     */
    int pulseg_get_prep_segment_table(const pulseg_collection *coll,
                                         int *out_ids, int subseq_idx);

    /**
     * @brief Copy main segment IDs into caller-supplied buffer.
     * @param[out] out_ids   Buffer of at least num_main_segments ints.
     * @return Number of IDs written, or negative error code.
     */
    int pulseg_get_main_segment_table(const pulseg_collection *coll,
                                         int *out_ids, int subseq_idx);

    /**
     * @brief Copy cooldown segment IDs into caller-supplied buffer.
     * @param[out] out_ids   Buffer of at least num_cooldown_segments ints.
     * @return Number of IDs written, or negative error code.
     */
    int pulseg_get_cooldown_segment_table(const pulseg_collection *coll,
                                             int *out_ids, int subseq_idx);

    /**
     * @brief Get canonical segment-ID sequence for vendor gradient-heating checks.
     *
     * Canonical-sequence rules:
     *   - Non-degenerate prep/cooldown: prep + (main repeated num_passes) + cooldown.
     *   - Degenerate prep/cooldown: main only (no pass expansion).
     *
     * If @p out_ids is NULL, the function returns the required count only.
     * Otherwise, @p out_ids must point to a buffer of at least that many ints.
     *
     * @param[in]  coll        Loaded collection.
     * @param[out] out_ids     Output buffer, or NULL for count query.
     * @param[in]  subseq_idx  Subsequence index.
     * @return Number of IDs (>= 0), or negative error code.
     */
    int pulseg_get_canonical_segment_sequence(const pulseg_collection *coll,
                                                 int *out_ids, int subseq_idx);

    /* ================================================================== */
    /*  RF getters                                                        */
    /* ================================================================== */

    /**
     * @brief Get RF statistics for a unique RF definition.
     * @return PULSEG_SUCCESS on success.
     */
    int pulseg_get_rf_stats(const pulseg_collection *coll,
                               pulseg_rf_stats *stats,
                               int subseq_idx, int rf_idx);

    /**
     * @brief Get per-block RF definition IDs for one TR.
     *
     * @p out_rf_ids must point to a pre-allocated array of tr_size ints.
     * Blocks without RF get -1.
     * @return tr_size on success, negative error code on failure.
     */
    int pulseg_get_tr_rf_ids(const pulseg_collection *coll,
                                int *out_rf_ids, int subseq_idx);

    /**
     * @brief Build an ordered array of RF stats for the canonical TR.
     *
     * Walks the canonical RF playback unit for the specified subsequence and,
     * for each block that carries an RF event, hard-copies the base rf_stats,
     * then patches event-specific amplitude-dependent fields from the actual
     * amplitude at that block position, and sets num_instances to the
     * repetition count for that canonical unit.
     *
     * Canonical-unit rules:
     *   - Standard / degenerate prep-cooldown subsequences use one imaging TR.
     *   - Non-degenerate prep/cooldown subsequences use one full pass including
     *     average expansion.
     *
     * The library allocates @p *out_pulses via PULSEG_ALLOC(); the caller
     * must release it with PULSEG_FREE() when done.  On return @p *out_pulses is NULL if the canonical
     * unit contains no RF events.
     *
     * @param[in]  coll          Loaded collection.
     * @param[out] out_pulses    Set to a malloc'd array; caller must free().
     * @param[in]  subseq_idx    Subsequence index.
     * @return Number of RF entries (>= 0), or negative error code.
     */
    int pulseg_get_rf_array(const pulseg_collection *coll,
                               pulseg_rf_stats **out_pulses,
                               int subseq_idx);

    /**
     * @brief Return decompressed RF magnitude waveform (multi-channel).
     *
     * Returns an array of num_channels pointers, each pointing to
     * num_samples floats.  The waveform is normalised (peak \u2248 1.0).
     * Use pulseg_get_rf_initial_amplitude_hz() and
     * pulseg_get_rf_max_amplitude_hz() for the physical scale.
     * Caller must free each result[ch] with PULSEG_FREE, then
     * free the result pointer itself with PULSEG_FREE.
     */
    float **pulseg_get_rf_magnitude(const pulseg_collection *coll,
                                       int *num_channels,
                                       int *num_samples,
                                       int seg_idx,
                                       int blk_idx);

    /**
     * @brief Return decompressed RF phase waveform (rad, multi-channel).
     *
     * Returns an array of num_channels pointers, each pointing to
     * num_samples floats.  Caller must free each result[ch] with
     * PULSEG_FREE, then the result pointer with PULSEG_FREE.
     */
    float **pulseg_get_rf_phase(const pulseg_collection *coll,
                                   int *num_channels,
                                   int *num_samples,
                                   int seg_idx,
                                   int blk_idx);

    /**
     * @brief Return RF time-point array (us, per-channel).
     *
     * For multi-channel RF the tiled time shape is truncated to the
     * first channel (all channels share the same time base).
     * Caller must free the returned array with PULSEG_FREE.
     */
    float *pulseg_get_rf_time_us(const pulseg_collection *coll,
                                    int seg_idx, int blk_idx);

    /** @brief Return initial RF amplitude (Hz) from the max-energy segment instance. */
    float pulseg_get_rf_initial_amplitude_hz(
        const pulseg_collection *coll,
        int seg_idx, int blk_idx);

    /** @brief Return peak RF amplitude (Hz) from the definition (unsigned max). */
    float pulseg_get_rf_max_amplitude_hz(
        const pulseg_collection *coll,
        int seg_idx, int blk_idx);

    /* ================================================================== */
    /*  Gradient getters (waveform data only)                             */
    /* ================================================================== */

    /**
     * @brief Return decompressed gradient amplitude waveforms (normalised).
     *
     * Waveforms are normalised (peak \u2248 1.0).  Use
     * pulseg_get_grad_initial_amplitude_hz_per_m() and
     * pulseg_get_grad_max_amplitude_hz_per_m() for the physical scale.
     * For multi-shot gradients, returns one waveform per shot.
     * All shots share the same number of samples.
     * Caller must free the returned array with PULSEG_FREE.
     */
    float **pulseg_get_grad_amplitude(const pulseg_collection *coll,
                                         int *num_shots,
                                         int *num_samples,
                                         int seg_idx,
                                         int blk_idx,
                                         int axis);

    /** @brief Return initial amplitude of a gradient event (Hz/m). */
    float pulseg_get_grad_initial_amplitude_hz_per_m(
        const pulseg_collection *coll,
        int seg_idx, int blk_idx, int axis);

    /** @brief Return initial shot ID for a gradient event. */
    int pulseg_get_grad_initial_shot_id(const pulseg_collection *coll,
                                           int seg_idx, int blk_idx, int axis);

    /** @brief Return peak gradient amplitude (Hz/m, unsigned) from the definition. */
    float pulseg_get_grad_max_amplitude_hz_per_m(
        const pulseg_collection *coll,
        int seg_idx, int blk_idx, int axis);

    /**
     * @brief Return gradient time-point array (us).
     *
     * The number of time points matches the amplitude waveform returned
     * by pulseg_get_grad_amplitude (or 3/4 for trapezoids).
     * Caller must free the returned array with PULSEG_FREE.
     */
    float *pulseg_get_grad_time_us(const pulseg_collection *coll,
                                      int seg_idx, int blk_idx, int axis);

    /* ================================================================== */
    /*  Label getters                                                     */
    /* ================================================================== */

    /** @brief Return label limits (min/max per label type) for a subsequence. */
    int pulseg_get_label_limits(const pulseg_collection *coll,
                                   pulseg_label_limits *limits,
                                   int subseq_idx);

    /* ================================================================== */
    /*  Definition getters                                                */
    /* ================================================================== */

    /** @brief Opaque read-only view of a single [DEFINITIONS] entry. */
    typedef struct pulseg_definition_entry
    {
        const char *name;          /**< key name                        */
        int num_values;            /**< number of space-separated values */
        const char *const *values; /**< value tokens (string array)     */
    } pulseg_definition_entry;

    /**
     * @brief Return all generic [DEFINITIONS] key-value pairs for a subsequence.
     *
     * On success, *out points to an internal array of num_entries entries.
     * The pointers are valid until pulseg_collection_free().
     *
     * @return PULSEG_SUCCESS on success, negative error code on failure.
     */
    int pulseg_get_definitions(const pulseg_collection *coll,
                                  const pulseg_definition_entry **out,
                                  int *num_entries,
                                  int subseq_idx);

    /**
     * @brief Get label values for a specific ADC occurrence.
     *
     * @p out_values must point to a pre-allocated array of at least
     * subseq_info.num_label_columns ints.
     *
     * @return PULSEG_SUCCESS on success, negative error code on failure.
     */
    int pulseg_get_adc_label(const pulseg_collection *coll,
                                int *out_values,
                                int subseq_idx,
                                int occurrence_idx);

    /* ================================================================== */
    /*  Block cursor / iterator                                           */
    /* ================================================================== */

    /**
     * @brief Advance the block cursor to the next block.
     * @return PULSEG_CURSOR_BLOCK or PULSEG_CURSOR_DONE.
     */
    int pulseg_cursor_next(pulseg_collection *coll);

    /**
     * @brief Reset the cursor to the last marked position.
     *
     * Rewinds the cursor by the number of blocks advanced since the last
     * pulseg_cursor_mark() call (or since the start of the current
     * subsequence if no mark was set).  Typically used for PMC rescan.
     */
    void pulseg_cursor_rewind(pulseg_collection *coll);

    /**
     * @brief Bookmark the current cursor position.
     *
     * Sets the rewind anchor so that a subsequent pulseg_cursor_rewind()
     * returns to this position.  Call at each TR boundary to enable
     * single-TR rescans.
     */
    void pulseg_cursor_mark(pulseg_collection *coll);

    /**
     * @brief Rewind the cursor to the absolute start of the collection.
     *
     * Unlike pulseg_cursor_rewind() (which is a relative rewind-to-mark),
     * this resets sequence_index as well, so a collection whose cursor has
     * already reached PULSEG_CURSOR_DONE can be traversed again from the
     * top.  Use before replaying a loaded collection from a fresh RSP entry
     * point.
     */
    void pulseg_cursor_reset(pulseg_collection *coll);

    /**
     * @brief Get the resolved block instance at the current cursor position.
     * @return PULSEG_SUCCESS on success, error code if cursor is done.
     */
    int pulseg_get_block_instance(const pulseg_collection *coll,
                                     pulseg_block_instance *inst);

    /**
     * @brief Get position and context metadata at the current cursor block.
     *
     * Returns segment boundaries, TR boundaries, trigger/NAV status, and
     * the scan-table position needed for freq-mod library lookup.
     *
     * @param[in]  coll  Loaded collection.
     * @param[out] info  Filled with cursor metadata.
     * @return PULSEG_SUCCESS on success.
     */
    int pulseg_cursor_get_info(const pulseg_collection *coll,
                                  pulseg_cursor_info *info);

    /* ================================================================== */
    /*  Frequency modulation collection                                   */
    /* ================================================================== */

    /**
     * @brief Count RF+ADC events across the entire sequence.
     */
    int pulseg_get_freq_mod_count(const pulseg_collection *coll);

    /**
     * @brief Count RF+ADC events in a specific TR region.
     *
     * @param tr_type   PULSEG_TR_REGION_PREP / _MAIN / _COOLDOWN.
     * @param tr_index  0-based TR instance (ignored for PREP/COOLDOWN).
     */
    int pulseg_get_freq_mod_count_tr(const pulseg_collection *coll,
                                        int tr_type, int tr_index);

    /* ================================================================== */
    /*  Unique-block and segment-block getters                            */
    /* ================================================================== */

    /**
     * @brief Return the number of unique block definitions for a subsequence.
     *
     * @param[in]  coll        Loaded collection.
     * @param[in]  subseq_idx  0-based subsequence index.
     * @return Number of unique blocks (>= 0), or negative error code.
     */
    int pulseg_get_num_unique_blocks(const pulseg_collection *coll,
                                        int subseq_idx);

    /**
     * @brief Return the 1-based .seq block ID for the n-th unique block.
     *
     * This is the key into pypulseq's block_events / Pulseq MATLAB toolbox
     * block_events table.  The ID corresponds to the FIRST occurrence of
     * that unique block pattern in the original .seq file.
     *
     * @param[in]  coll        Loaded collection.
     * @param[in]  subseq_idx  0-based subsequence index.
     * @param[in]  blk_def_idx 0-based index into the unique block list.
     * @return 1-based block ID (> 0), or negative error code.
     */
    int pulseg_get_unique_block_id(const pulseg_collection *coll,
                                      int subseq_idx, int blk_def_idx);

    /**
     * @brief Copy unique-block-definition indices for a segment.
     *
     * For segment @p seg_idx, writes @c segment_info.num_blocks indices
     * to @p out_ids.  Each value is a 0-based index into the unique block
     * list (suitable for passing to pulseg_get_unique_block_id).
     *
     * @param[in]  coll      Loaded collection.
     * @param[out] out_ids   Caller buffer (at least num_blocks ints).
     * @param[in]  seg_idx   Global segment index.
     * @return Number of IDs written (>= 0), or negative error code.
     */
    int pulseg_get_segment_block_def_indices(const pulseg_collection *coll,
                                                int *out_ids, int seg_idx);

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_COLLECTION_H */
