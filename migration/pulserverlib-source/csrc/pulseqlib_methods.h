/**
 * @file pulseqlib_methods.h
 * @brief Public API for the Pulseq interpreter library.
 *
 * Include this header in application code.  It pulls in
 * pulseqlib_config.h and pulseqlib_types.h.
 *
 * Naming conventions:
 *   - read / write   for file I/O
 *   - calc            for computation
 *   - get_             for in-memory retrieval
 *   - parse            for filling structs from pre-read data (internal)
 *
 * All functions use the pulseqlib_ prefix and are declared
 * extern "C" when compiled with a C++ compiler.
 */

#ifndef PULSEQLIB_METHODS_H
#define PULSEQLIB_METHODS_H

#include "pulseqlib_config.h"
#include "pulseqlib_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================== */
/*  Read / load                                                       */
/* ================================================================== */

/**
 * @brief Read a (possibly chained) Pulseq sequence from disk.
 *
 * On success the library heap-allocates the collection and writes it
 * to @p *out_coll.  The caller owns the collection and must free it
 * with pulseqlib_collection_free().
 *
 * @param[out] out_coll         Receives the allocated collection.
 * @param[out] diag             Diagnostic info on failure.
 * @param[in]  file_path        Path to the first .seq file.
 * @param[in]  opts             Scanner limits / rasters.
 * @param[in]  cache_binary     1 = read/write .bin cache alongside .seq.
 * @param[in]  verify_signature 1 = verify MD5 signature for every .seq
 *                              file in the chain.
 * @param[in]  parse_labels     1 = build ADC label table via dry-run.
 * @param[in]  num_averages     Number of scan averages (>= 1).
 * @return PULSEQLIB_SUCCESS on success, negative error code on failure.
 */
int pulseqlib_read(
    pulseqlib_collection** out_coll,
    pulseqlib_diagnostic*  diag,
    const char*            file_path,
    const pulseqlib_opts*  opts,
    int                    cache_binary,
    int                    verify_signature,
    int                    parse_labels,
    int                    num_averages);

/**
 * @brief Read one or more Pulseq subsequences from in-memory buffers.
 *
 * Wrapper-friendly counterpart of pulseqlib_read(): the caller supplies
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
 * @return PULSEQLIB_SUCCESS on success, negative error code on failure.
 */
int pulseqlib_read_from_buffers(
    pulseqlib_collection** out_coll,
    pulseqlib_diagnostic*  diag,
    const char* const*     buffers,
    const int*             buffer_sizes,
    int                    num_buffers,
    const pulseqlib_opts*  opts,
    int                    parse_labels,
    int                    num_averages);

/* ================================================================== */
/*  Options initializer                                               */
/* ================================================================== */

/**
 * @brief Fill a pulseqlib_opts struct with scanner parameters.
 *
 * The @c vendor field is set to @c PULSEQLIB_VENDOR (compile-time
 * default).  Override it after calling this function if needed.
 */
void pulseqlib_opts_init(
    pulseqlib_opts* opts,
    float gamma_hz_per_t,
    float b0_t,
    float max_grad_hz_per_m,
    float max_slew_hz_per_m_per_s,
    float rf_raster_us,
    float grad_raster_us,
    float adc_raster_us,
    float block_raster_us);

/* ================================================================== */
/*  Diagnostic helpers                                                */
/* ================================================================== */

/** @brief Zero-initialize a diagnostic struct. */
void pulseqlib_diagnostic_init(pulseqlib_diagnostic* diag);

/** @brief Return a human-readable message for an error code. */
const char* pulseqlib_get_error_message(int code);

/** @brief Return a fix-suggestion hint for an error code. */
const char* pulseqlib_get_error_hint(int code);

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
int pulseqlib_format_error(
    char* buf, int buf_size,
    int code,
    const pulseqlib_diagnostic* diag);

/* ================================================================== */
/*  Consistency check                                                 */
/* ================================================================== */

/**
 * @brief Re-run internal consistency checks on a loaded collection.
 *
 * Already called by pulseqlib_read / pulseqlib_read_from_buffers.
 * Exposed for unit-test or post-hoc validation workflows.
 *
 * @param[in]  coll  Loaded collection.
 * @param[out] diag  Diagnostic (may be NULL).
 * @return PULSEQLIB_SUCCESS on success, negative error code on failure.
 */
int pulseqlib_check_consistency(
    const pulseqlib_collection* coll,
    pulseqlib_diagnostic*       diag);

/* ================================================================== */
/*  Scan-time peek (fast estimate from definitions only)              */
/* ================================================================== */

/**
 * @brief Peek at scan time without full sequence loading.
 *
 * Reads only the [DEFINITIONS] sections from a (possibly chained)
 * .seq file to obtain @c TotalDuration.  The result is an
 * approximation: dead time between segments is not accounted for
 * and @c total_segment_boundaries is left at 0.
 *
 * @c num_reps controls the number of repetitions the consumer
 * intends to play (>= 1).  For subsequences whose
 * @c IgnoreAverages definition is set, the multiplier is clamped
 * to 1.
 *
 * Typical use: UI preview of scan time before the sequence is
 * fully loaded.
 *
 * @param[out] info       Receives scan time summary.
 * @param[in]  file_path  Path to the first .seq file.
 * @param[in]  opts       Scanner limits (used for chain traversal).
 * @param[in]  num_reps   Number of repetitions (>= 1).
 * @return PULSEQLIB_SUCCESS on success, negative error code on failure.
 */
int pulseqlib_peek_scan_time(
    pulseqlib_scan_time_info* info,
    const char*               file_path,
    const pulseqlib_opts*     opts,
    int                       num_reps);

/* ================================================================== */
/*  Collection lifetime                                               */
/* ================================================================== */

/** @brief Free a collection and all owned memory. */
void pulseqlib_collection_free(pulseqlib_collection* coll);

/* ================================================================== */
/*  Binary cache (serialization / deserialization)                    */
/* ================================================================== */

/**
 * @brief Save a loaded collection to a binary cache file.
 *
 * @param[in]  coll          Collection to save.
 * @param[in]  path          Output file path (e.g. "my_sequence.bin").
 * @param[in]  source_size   Size (in bytes) of the original .seq buffer
 *                            used to load this collection.  Written into
 *                            the cache header for integrity validation on
 *                            reload.
 * @return PULSEQLIB_SUCCESS on success, negative on failure.
 */
int pulseqlib_save_cache(
    const pulseqlib_collection* coll,
    const char*                 path,
    int                         source_size);

/**
 * @brief Load a collection from a binary cache file.
 *
 * The caller must allocate a pulseqlib_collection (e.g. via calloc)
 * before calling this function.  On success the collection is populated
 * from the cache.  On failure the collection is unchanged.
 *
 * @param[out] coll          Pre-allocated collection to populate.
 * @param[in]  path          Cache file path.
 * @param[in]  source_size   Expected size (bytes) of the original .seq
 *                            buffer.  Must match the value stored in the
 *                            cache header (guards against stale caches).
 * @return PULSEQLIB_SUCCESS on success, negative on failure.
 */
int pulseqlib_load_cache(
    pulseqlib_collection* coll,
    const char*           path,
    int                   source_size);

/* ================================================================== */
/*  TR gradient waveforms (for plotting)                              */
/* ================================================================== */

/**
 * @brief Extract per-axis gradient waveforms for a single TR.
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
 * @param[in]  coll        Loaded collection.
 * @param[in]  subseq_idx  Subsequence index.
 * @param[out] waveforms   Receives the waveform data (caller frees
 *                          via pulseqlib_tr_gradient_waveforms_free).
 * @param[out] diag        Diagnostic on failure.
 * @return PULSEQLIB_SUCCESS on success, negative error code on failure.
 */
int pulseqlib_get_tr_gradient_waveforms(
    const pulseqlib_collection*    coll,
    int                            subseq_idx,
    pulseqlib_tr_gradient_waveforms* waveforms,
    pulseqlib_diagnostic*          diag);

/** @brief Free waveform arrays inside a pulseqlib_tr_gradient_waveforms. */
void pulseqlib_tr_gradient_waveforms_free(pulseqlib_tr_gradient_waveforms* w);

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
 *   - PULSEQLIB_AMP_MAX_POS  (0) — position-max across all TRs
 *   - PULSEQLIB_AMP_ZERO_VAR (1) — zero variable-amplitude gradients, keep constant ones
 *   - PULSEQLIB_AMP_ACTUAL   (2) — signed amplitude for given TR index
 *
 * For modes 0 and 1, @p tr_index is ignored (canonical main TR is used).
 * For mode 2, @p tr_index selects the TR instance (0-based).
 *
 * @p include_prep / @p include_cooldown add prep / cooldown blocks
 * before / after the main TR.  Prep is only valid for the first TR,
 * cooldown only for the last.
 *
 * Block descriptors in the output carry segment assignment (or -1 for
 * prep / cooldown blocks).  The caller is responsible for freeing via
 * pulseqlib_tr_waveforms_free().
 *
 * @param[in]  coll               Loaded collection.
 * @param[in]  subseq_idx         Subsequence index (0 for single-seq).
 * @param[in]  amplitude_mode     PULSEQLIB_AMP_MAX_POS / _ZERO_VAR / _ACTUAL.
 * @param[in]  tr_index           TR instance (only for _ACTUAL mode).
 * @param[in]  include_prep       Non-zero to prepend prep blocks.
 * @param[in]  include_cooldown   Non-zero to append cooldown blocks.
 * @param[out] out                Output waveforms (caller frees).
 * @param[out] diag               Diagnostic on error.
 * @return PULSEQLIB_SUCCESS on success, negative error code on failure.
 */
int pulseqlib_get_tr_waveforms(
    const pulseqlib_collection*     coll,
    int                             subseq_idx,
    int                             amplitude_mode,
    int                             tr_index,
    int                             include_prep,
    int                             include_cooldown,
    int                             collapse_delays,
    pulseqlib_tr_waveforms*         out,
    pulseqlib_diagnostic*           diag);

/** @brief Free all arrays inside a pulseqlib_tr_waveforms. */
void pulseqlib_tr_waveforms_free(pulseqlib_tr_waveforms* w);

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
 * @return PULSEQLIB_SUCCESS if safe, negative error code on violation.
 */
int pulseqlib_check_safety(
    pulseqlib_collection*          coll,
    pulseqlib_diagnostic*          diag,
    const pulseqlib_opts*          opts,
    int                            num_forbidden_bands,
    const pulseqlib_forbidden_band* forbidden_bands,
    const pulseqlib_pns_params*    pns_params,
    float                          pns_threshold_percent);

/* ================================================================== */
/*  Acoustic spectra (for wrapper-side plotting)                      */
/* ================================================================== */

/**
 * @brief Compute acoustic spectral data for wrapper-side plotting.
 *
 * Independently extracts TR gradient waveforms (without segment
 * labels), interpolates them to uniform raster, and computes
 * spectrograms, full-TR spectra, and sequence-level harmonics.
 * Peak candidate masks are included for forbidden-band detection
 * in the wrapper.
 *
 * @param[out] spectra                  Receives spectral data (caller frees
 *                                       via pulseqlib_acoustic_spectra_free).
 * @param[out] diag                     Diagnostic on failure.
 * @param[in]  coll                     Loaded collection.
 * @param[in]  subseq_idx              Subsequence index.
 * @param[in]  opts                     Scanner limits.
 * @param[in]  target_window_size       Sliding window length (0 = auto).
 * @param[in]  target_resolution_hz     Spectral resolution (0 = auto).
 * @param[in]  max_freq_hz             Max frequency to report (0 = auto).
 * @param[in]  num_forbidden_bands      Number of forbidden bands.
 * @param[in]  forbidden_bands          Array of forbidden bands.
 * @return PULSEQLIB_SUCCESS on success, negative error code on failure.
 */
int pulseqlib_calc_acoustic_spectra(
    pulseqlib_acoustic_spectra*    spectra,
    pulseqlib_diagnostic*          diag,
    const pulseqlib_collection*    coll,
    int                            subseq_idx,
    const pulseqlib_opts*          opts,
    int                            target_window_size,
    float                          target_resolution_hz,
    float                          max_freq_hz,
    int                            num_forbidden_bands,
    const pulseqlib_forbidden_band* forbidden_bands);

/** @brief Free arrays inside a pulseqlib_acoustic_spectra. */
void pulseqlib_acoustic_spectra_free(pulseqlib_acoustic_spectra* s);

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
 *                           via pulseqlib_pns_result_free).
 * @param[out] diag         Diagnostic on failure.
 * @param[in]  coll         Loaded collection.
 * @param[in]  subseq_idx   Subsequence index.
 * @param[in]  opts         Scanner limits.
 * @param[in]  params       PNS model parameters.
 * @return PULSEQLIB_SUCCESS on success, negative error code on failure.
 */
int pulseqlib_calc_pns(
    pulseqlib_pns_result*       result,
    pulseqlib_diagnostic*       diag,
    const pulseqlib_collection* coll,
    int                         subseq_idx,
    const pulseqlib_opts*       opts,
    const pulseqlib_pns_params* params);

/** @brief Free arrays inside a pulseqlib_pns_result. */
void pulseqlib_pns_result_free(pulseqlib_pns_result* r);

/* ================================================================== */
/*  Subsequence getters                                               */
/* ================================================================== */

/**
 * @brief Fill a pulseqlib_collection_info with collection-level summary.
 *
 * Replaces pulseqlib_get_num_subsequences, pulseqlib_get_num_segments,
 * pulseqlib_get_max_adc_samples, pulseqlib_get_total_readouts,
 * pulseqlib_get_total_duration_us.
 */
int pulseqlib_get_collection_info(const pulseqlib_collection* coll,
                                  pulseqlib_collection_info*  info);

/**
 * @brief Fill a pulseqlib_subseq_info for one subsequence.
 *
 * Replaces ~18 individual per-subsequence getters (TR structure,
 * prep/cooldown counts, degenerate flags, segment counts, label info).
 */
int pulseqlib_get_subseq_info(const pulseqlib_collection* coll,
                              int                         subseq_idx,
                              pulseqlib_subseq_info*      info);

/**
 * @brief Fill a pulseqlib_segment_info for one segment.
 *
 * Replaces ~11 individual per-segment getters (duration, blocks,
 * trigger, NAV, timing gaps).
 */
int pulseqlib_get_segment_info(const pulseqlib_collection* coll,
                               int                         seg_idx,
                               pulseqlib_segment_info*     info);

/**
 * @brief Fill a pulseqlib_block_info for one block within a segment.
 *
 * Replaces all block-level has_xxx / get_xxx accessor pairs.
 * Waveform data is NOT included; use the dedicated waveform getters
 * keyed by metadata from this struct.
 */
int pulseqlib_get_block_info(const pulseqlib_collection* coll,
                             int                         seg_idx,
                             int                         blk_idx,
                             pulseqlib_block_info*       info);

/**
 * @brief Fill a pulseqlib_adc_def for a unique ADC definition.
 *
 * @p adc_idx is a global index across all subsequences (same as
 * block_info.adc_def_id).
 */
int pulseqlib_get_adc_def(const pulseqlib_collection* coll,
                          int                         adc_idx,
                          pulseqlib_adc_def*          def);

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
int pulseqlib_block_needs_freq_mod(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx,
    int* num_samples);

/**
 * @brief Return the RF isocenter time (us) relative to segment start.
 *
 * Looks up the segment timing RF anchor matching @p blk_idx.
 * Returns -1.0f if the block has no RF anchor.
 */
float pulseqlib_get_rf_isocenter_us(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx);

/**
 * @brief Return the ADC k-zero time (us) relative to segment start.
 *
 * Looks up the segment timing ADC anchor matching @p blk_idx.
 * Returns -1.0f if the block has no ADC anchor.
 */
float pulseqlib_get_adc_kzero_us(
    const pulseqlib_collection* coll,
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
 * @return PULSEQLIB_SUCCESS on success, negative error code on failure.
 */
int pulseqlib_get_scan_time(const pulseqlib_collection* coll,
                           int                        num_reps,
                           pulseqlib_scan_time_info*  info);

/* ================================================================== */
/*  Segment table getters (copy to caller buffer)                     */
/* ================================================================== */

/**
 * @brief Copy prep segment IDs into caller-supplied buffer.
 * @param[out] out_ids   Buffer of at least num_prep_segments ints.
 * @return Number of IDs written, or negative error code.
 */
int pulseqlib_get_prep_segment_table(const pulseqlib_collection* coll,
                                     int subseq_idx, int* out_ids);

/**
 * @brief Copy main segment IDs into caller-supplied buffer.
 * @param[out] out_ids   Buffer of at least num_main_segments ints.
 * @return Number of IDs written, or negative error code.
 */
int pulseqlib_get_main_segment_table(const pulseqlib_collection* coll,
                                     int subseq_idx, int* out_ids);

/**
 * @brief Copy cooldown segment IDs into caller-supplied buffer.
 * @param[out] out_ids   Buffer of at least num_cooldown_segments ints.
 * @return Number of IDs written, or negative error code.
 */
int pulseqlib_get_cooldown_segment_table(const pulseqlib_collection* coll,
                                         int subseq_idx, int* out_ids);

/* ================================================================== */
/*  RF getters                                                        */
/* ================================================================== */

/**
 * @brief Get RF statistics for a unique RF definition.
 * @return PULSEQLIB_SUCCESS on success.
 */
int pulseqlib_get_rf_stats(const pulseqlib_collection* coll,
                           pulseqlib_rf_stats* stats,
                           int subseq_idx, int rf_idx);

/**
 * @brief Get per-block RF definition IDs for one TR.
 *
 * @p out_rf_ids must point to a pre-allocated array of tr_size ints.
 * Blocks without RF get -1.
 * @return tr_size on success, negative error code on failure.
 */
int pulseqlib_get_tr_rf_ids(const pulseqlib_collection* coll,
                            int* out_rf_ids, int subseq_idx);

/**
 * @brief Build an ordered array of RF stats for a TR region.
 *
 * Walks the block table for the specified region and, for each block
 * that carries an RF event, hard-copies the base rf_stats, then patches
 * act_amplitude_hz from the actual amplitude at that block position,
 * and sets num_instances to the repetition count for that region.
 *
 * The library allocates @p *out_pulses via malloc(); the caller must
 * free() it when done.  On return @p *out_pulses is NULL if the region
 * contains no RF events.
 *
 * @param[in]  coll          Loaded collection.
 * @param[out] out_pulses    Set to a malloc'd array; caller must free().
 * @param[in]  subseq_idx    Subsequence index.
 * @param[in]  region        PULSEQLIB_TR_REGION_PREP/_MAIN/_COOLDOWN.
 * @return Number of RF entries (>= 0), or negative error code.
 */
int pulseqlib_get_rf_array(const pulseqlib_collection* coll,
                           pulseqlib_rf_stats** out_pulses,
                           int subseq_idx,
                           int region);

/**
 * @brief Return decompressed RF magnitude waveform (multi-channel).
 *
 * Returns an array of num_channels pointers, each pointing to
 * num_samples floats.  The waveform is normalised (peak \u2248 1.0).
 * Use pulseqlib_get_rf_initial_amplitude_hz() and
 * pulseqlib_get_rf_max_amplitude_hz() for the physical scale.
 * Caller must free each result[ch] with PULSEQLIB_FREE, then
 * free the result pointer itself with PULSEQLIB_FREE.
 */
float** pulseqlib_get_rf_magnitude(const pulseqlib_collection* coll,
                                   int seg_idx, int blk_idx,
                                   int* num_channels,
                                   int* num_samples);

/**
 * @brief Return decompressed RF phase waveform (rad, multi-channel).
 *
 * Returns an array of num_channels pointers, each pointing to
 * num_samples floats.  Caller must free each result[ch] with
 * PULSEQLIB_FREE, then the result pointer with PULSEQLIB_FREE.
 */
float** pulseqlib_get_rf_phase(const pulseqlib_collection* coll,
                               int seg_idx, int blk_idx,
                               int* num_channels,
                               int* num_samples);

/**
 * @brief Return RF time-point array (us, per-channel).
 *
 * For multi-channel RF the tiled time shape is truncated to the
 * first channel (all channels share the same time base).
 * Caller must free the returned array with PULSEQLIB_FREE.
 */
float* pulseqlib_get_rf_time_us(const pulseqlib_collection* coll,
                                int seg_idx, int blk_idx);

/** @brief Return initial RF amplitude (Hz) from the max-energy segment instance. */
float pulseqlib_get_rf_initial_amplitude_hz(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx);

/** @brief Return peak RF amplitude (Hz) from the definition (unsigned max). */
float pulseqlib_get_rf_max_amplitude_hz(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx);

/* ================================================================== */
/*  Gradient getters (waveform data only)                             */
/* ================================================================== */

/**
 * @brief Return decompressed gradient amplitude waveforms (normalised).
 *
 * Waveforms are normalised (peak \u2248 1.0).  Use
 * pulseqlib_get_grad_initial_amplitude_hz_per_m() and
 * pulseqlib_get_grad_max_amplitude_hz_per_m() for the physical scale.
 * For multi-shot gradients, returns one waveform per shot.
 * All shots share the same number of samples.
 * Caller must free the returned array with PULSEQLIB_FREE.
 */
float** pulseqlib_get_grad_amplitude(const pulseqlib_collection* coll,
                                     int seg_idx, int blk_idx, int axis,
                                     int* num_shots,
                                     int* num_samples);

/** @brief Return initial amplitude of a gradient event (Hz/m). */
float pulseqlib_get_grad_initial_amplitude_hz_per_m(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx, int axis);

/** @brief Return initial shot ID for a gradient event. */
int pulseqlib_get_grad_initial_shot_id(const pulseqlib_collection* coll,
                                       int seg_idx, int blk_idx, int axis);

/** @brief Return peak gradient amplitude (Hz/m, unsigned) from the definition. */
float pulseqlib_get_grad_max_amplitude_hz_per_m(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx, int axis);

/**
 * @brief Return gradient time-point array (us).
 *
 * The number of time points matches the amplitude waveform returned
 * by pulseqlib_get_grad_amplitude (or 3/4 for trapezoids).
 * Caller must free the returned array with PULSEQLIB_FREE.
 */
float* pulseqlib_get_grad_time_us(const pulseqlib_collection* coll,
                                  int seg_idx, int blk_idx, int axis);

/* ================================================================== */
/*  Label getters                                                     */
/* ================================================================== */

/** @brief Return label limits (min/max per label type) for a subsequence. */
int pulseqlib_get_label_limits(const pulseqlib_collection* coll,
                               int subseq_idx,
                               pulseqlib_label_limits* limits);

/**
 * @brief Get label values for a specific ADC occurrence.
 *
 * @p out_values must point to a pre-allocated array of at least
 * subseq_info.num_label_columns ints.
 *
 * @return PULSEQLIB_SUCCESS on success, negative error code on failure.
 */
int pulseqlib_get_adc_label(const pulseqlib_collection* coll,
                            int subseq_idx,
                            int occurrence_idx,
                            int* out_values);

/* ================================================================== */
/*  Block cursor / iterator                                           */
/* ================================================================== */

/**
 * @brief Advance the block cursor to the next block.
 * @return PULSEQLIB_CURSOR_BLOCK or PULSEQLIB_CURSOR_DONE.
 */
int pulseqlib_cursor_next(pulseqlib_collection* coll);

/**
 * @brief Reset the cursor to the last marked position.
 *
 * Rewinds the cursor by the number of blocks advanced since the last
 * pulseqlib_cursor_mark() call (or since the start of the current
 * subsequence if no mark was set).  Typically used for PMC rescan.
 */
void pulseqlib_cursor_reset(pulseqlib_collection* coll);

/**
 * @brief Bookmark the current cursor position.
 *
 * Sets the rewind anchor so that a subsequent pulseqlib_cursor_reset()
 * returns to this position.  Call at each TR boundary to enable
 * single-TR rescans.
 */
void pulseqlib_cursor_mark(pulseqlib_collection* coll);

/**
 * @brief Get the resolved block instance at the current cursor position.
 * @return PULSEQLIB_SUCCESS on success, error code if cursor is done.
 */
int pulseqlib_get_block_instance(const pulseqlib_collection* coll,
                                 pulseqlib_block_instance*    inst);

/**
 * @brief Get position and context metadata at the current cursor block.
 *
 * Returns segment boundaries, TR boundaries, trigger/NAV status, and
 * the scan-table position needed for freq-mod library lookup.
 *
 * @param[in]  coll  Loaded collection.
 * @param[out] info  Filled with cursor metadata.
 * @return PULSEQLIB_SUCCESS on success.
 */
int pulseqlib_cursor_get_info(const pulseqlib_collection* coll,
                              pulseqlib_cursor_info*       info);

/* ================================================================== */
/*  Frequency modulation collection                                   */
/* ================================================================== */

/**
 * @brief Count RF+ADC events across the entire sequence.
 */
int pulseqlib_get_freq_mod_count(const pulseqlib_collection* coll);

/**
 * @brief Count RF+ADC events in a specific TR region.
 *
 * @param tr_type   PULSEQLIB_TR_REGION_PREP / _MAIN / _COOLDOWN.
 * @param tr_index  0-based TR instance (ignored for PREP/COOLDOWN).
 */
int pulseqlib_get_freq_mod_count_tr(const pulseqlib_collection* coll,
                                    int tr_type, int tr_index);

/**
 * @brief Build frequency modulation data for all subsequences.
 *
 * Constructs deduped amplitude-scaled 3-channel gradient modulators
 * and computes shift-resolved 1D plan waveforms for every subsequence
 * in the collection.
 *
 * For PMC-enabled subsequences the 3-channel data is retained so that
 * pulseqlib_update_freq_mod_collection() can recompute waveforms with
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
 * @return PULSEQLIB_SUCCESS on success.
 */
int pulseqlib_build_freq_mod_collection(
    pulseqlib_freq_mod_collection** out_fmc,
    const pulseqlib_collection* coll,
    const float* shift_m,
    const float* fov_rotation);

/**
 * @brief Recompute freq-mod waveforms for one subsequence.
 *
 * Only valid for PMC-enabled subsequences (3-channel data is still
 * resident).  Returns an error if the 3-channel data was freed.
 *
 * @param[in,out] fmc         Freq-mod collection.
 * @param[in]     subseq_idx  0-based subsequence index.
 * @param[in]     shift_m     New spatial shift (dx, dy, dz) in metres.
 * @return PULSEQLIB_SUCCESS on success.
 */
int pulseqlib_update_freq_mod_collection(
    pulseqlib_freq_mod_collection* fmc,
    int subseq_idx,
    const float* shift_m);

/**
 * @brief Look up the freq-mod waveform for a scan-table position.
 *
 * @param[in]  fmc             Freq-mod collection.
 * @param[in]  subseq_idx     0-based subsequence index.
 * @param[in]  scan_table_pos Position in the subsequence scan table.
 * @param[out] out_waveform   Pointer into library (do NOT free).
 * @param[out] out_num_samples Waveform length.
 * @param[out] out_phase_rad  Phase compensation (rad).
 * @return 1 if the block has a freq-mod event, 0 if not.
 */
int pulseqlib_freq_mod_collection_get(
    const pulseqlib_freq_mod_collection* fmc,
    int subseq_idx,
    int scan_table_pos,
    const float** out_waveform,
    int* out_num_samples,
    float* out_phase_rad);

/**
 * @brief Write all per-subsequence freq-mod data to a single cache file.
 *
 * @param[in]  fmc   Built collection (3-channel data must be resident
 *                    for at least one subsequence).
 * @param[in]  path  Output file path (e.g. "seq.fmod.bin").
 * @return PULSEQLIB_SUCCESS on success.
 */
int pulseqlib_freq_mod_collection_write_cache(
    const pulseqlib_freq_mod_collection* fmc,
    const char* path);

/**
 * @brief Read freq-mod collection from cache and compute plans.
 *
 * @param[out] out_fmc     Receives an allocated collection (caller frees).
 * @param[in]  path        Cache file path.
 * @param[in]  coll        Loaded sequence collection (provides PMC flags
 *                          and subsequence count).
 * @param[in]  shift_m     Spatial shift for plan computation.
 * @return PULSEQLIB_SUCCESS on success.
 */
int pulseqlib_freq_mod_collection_read_cache(
    pulseqlib_freq_mod_collection** out_fmc,
    const char* path,
    const pulseqlib_collection* coll,
    const float* shift_m);

/** @brief Free a frequency modulation collection and all owned memory. */
void pulseqlib_freq_mod_collection_free(pulseqlib_freq_mod_collection* fmc);

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
int pulseqlib_get_num_unique_blocks(const pulseqlib_collection* coll,
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
int pulseqlib_get_unique_block_id(const pulseqlib_collection* coll,
                                  int subseq_idx, int blk_def_idx);

/**
 * @brief Copy unique-block-definition indices for a segment.
 *
 * For segment @p seg_idx, writes @c segment_info.num_blocks indices
 * to @p out_ids.  Each value is a 0-based index into the unique block
 * list (suitable for passing to pulseqlib_get_unique_block_id).
 *
 * @param[in]  coll      Loaded collection.
 * @param[in]  seg_idx   Global segment index.
 * @param[out] out_ids   Caller buffer (at least num_blocks ints).
 * @return Number of IDs written (>= 0), or negative error code.
 */
int pulseqlib_get_segment_block_def_indices(const pulseqlib_collection* coll,
                                            int seg_idx, int* out_ids);

#ifdef __cplusplus
}
#endif

#endif /* PULSEQLIB_METHODS_H */
