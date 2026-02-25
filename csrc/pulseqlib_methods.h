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
 * @return PULSEQLIB_OK on success, negative error code on failure.
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
 * @return PULSEQLIB_OK on success, negative error code on failure.
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
 * @return PULSEQLIB_OK on success, negative error code on failure.
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
 * @return PULSEQLIB_OK on success, negative error code on failure.
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
 * @return PULSEQLIB_OK on success, negative error code on failure.
 */
int pulseqlib_get_tr_gradient_waveforms(
    const pulseqlib_collection*    coll,
    int                            subseq_idx,
    pulseqlib_tr_gradient_waveforms* waveforms,
    pulseqlib_diagnostic*          diag);

/** @brief Free waveform arrays inside a pulseqlib_tr_gradient_waveforms. */
void pulseqlib_tr_gradient_waveforms_free(pulseqlib_tr_gradient_waveforms* w);

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
 * @return PULSEQLIB_OK if safe, negative error code on violation.
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
 * @return PULSEQLIB_OK on success, negative error code on failure.
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
 * @return PULSEQLIB_OK on success, negative error code on failure.
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

/** @brief Return number of subsequences in the collection. */
int pulseqlib_get_num_subsequences(const pulseqlib_collection* coll);

/** @brief Return TR duration for a subsequence (us). */
float pulseqlib_get_tr_duration_us(const pulseqlib_collection* coll,
                                   int subseq_idx);

/** @brief Return number of TRs in a subsequence. */
int pulseqlib_get_num_trs(const pulseqlib_collection* coll,
                          int subseq_idx);

/** @brief Return number of blocks per TR in a subsequence. */
int pulseqlib_get_tr_size(const pulseqlib_collection* coll,
                          int subseq_idx);

/** @brief Return number of preparation blocks before the first TR. */
int pulseqlib_get_num_prep_blocks(const pulseqlib_collection* coll,
                                  int subseq_idx);

/** @brief Return number of cooldown blocks after the last TR. */
int pulseqlib_get_num_cooldown_blocks(const pulseqlib_collection* coll,
                                      int subseq_idx);

/** @brief Return 1 if preparation blocks are degenerate (same as first TR). */
int pulseqlib_get_degenerate_prep(const pulseqlib_collection* coll,
                                  int subseq_idx);

/** @brief Return 1 if cooldown blocks are degenerate (same as last TR). */
int pulseqlib_get_degenerate_cooldown(const pulseqlib_collection* coll,
                                      int subseq_idx);

/** @brief Return number of preparation TRs. */
int pulseqlib_get_num_prep_trs(const pulseqlib_collection* coll,
                               int subseq_idx);

/** @brief Return number of cooldown TRs. */
int pulseqlib_get_num_cooldown_trs(const pulseqlib_collection* coll,
                                   int subseq_idx);

/** @brief Return number of unique ADC events in a subsequence. */
int pulseqlib_get_num_unique_adcs(const pulseqlib_collection* coll,
                                  int subseq_idx);

/** @brief Return 1 if PMC (prospective motion correction) is enabled. */
int pulseqlib_is_pmc_enabled(const pulseqlib_collection* coll,
                             int subseq_idx);

/** @brief Return global segment offset for subsequence @p subseq_idx. */
int pulseqlib_get_subseq_segment_offset(const pulseqlib_collection* coll,
                                        int subseq_idx);

/**
 * @brief Return total number of ADC readout events across all
 *        subsequences (accounting for TR repetitions, prep, cooldown).
 */
int pulseqlib_get_total_readouts(const pulseqlib_collection* coll);

/** @brief Return total sequence duration (us). */
float pulseqlib_get_total_duration_us(const pulseqlib_collection* coll);

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
 * @return PULSEQLIB_OK on success, negative error code on failure.
 */
int pulseqlib_get_scan_time(const pulseqlib_collection* coll,
                           int                        num_reps,
                           pulseqlib_scan_time_info*  info);

/* ================================================================== */
/*  Segment getters                                                   */
/* ================================================================== */

/** @brief Return total number of unique segments across all subsequences. */
int pulseqlib_get_num_segments(const pulseqlib_collection* coll);

/** @brief Return duration of segment @p seg_idx (us). */
int pulseqlib_get_segment_duration_us(const pulseqlib_collection* coll,
                                      int seg_idx);

/** @brief Return 1 if segment contains only delays (no events). */
int pulseqlib_is_segment_pure_delay(const pulseqlib_collection* coll,
                                    int seg_idx);

/** @brief Return number of unique blocks in a segment. */
int pulseqlib_get_segment_num_blocks(const pulseqlib_collection* coll,
                                     int seg_idx);

/** @brief Return start block index (in the original sequence) for a segment. */
int pulseqlib_get_segment_start_block(const pulseqlib_collection* coll,
                                      int seg_idx);

/* ================================================================== */
/*  Segment table getters                                             */
/* ================================================================== */

/** @brief Return number of segments in the prep region. */
int pulseqlib_get_num_prep_segments(const pulseqlib_collection* coll,
                                    int subseq_idx);

/** @brief Return number of segments in the main TR region. */
int pulseqlib_get_num_main_segments(const pulseqlib_collection* coll,
                                    int subseq_idx);

/** @brief Return number of segments in the cooldown region. */
int pulseqlib_get_num_cooldown_segments(const pulseqlib_collection* coll,
                                        int subseq_idx);

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
/*  Block getters (within segments)                                   */
/* ================================================================== */

/** @brief Return start time of block within segment (us). */
int pulseqlib_get_block_start_time_us(const pulseqlib_collection* coll,
                                      int seg_idx, int blk_idx);

/** @brief Return duration of a block (us). */
int pulseqlib_get_block_duration_us(const pulseqlib_collection* coll,
                                    int seg_idx, int blk_idx);

/* ================================================================== */
/*  RF getters                                                        */
/* ================================================================== */

/** @brief Return number of unique RF events in a subsequence. */
int pulseqlib_get_num_unique_rf(const pulseqlib_collection* coll,
                                int subseq_idx);

/**
 * @brief Get RF statistics for a unique RF definition.
 * @return PULSEQLIB_OK on success.
 */
int pulseqlib_get_rf_stats(const pulseqlib_collection* coll,
                           pulseqlib_rf_stats* stats,
                           int subseq_idx, int rf_idx);

/** @brief Get base (peak) RF amplitude in Hz. */
float pulseqlib_get_rf_base_amplitude_hz(const pulseqlib_collection* coll,
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
 * Region semantics:
 *   PULSEQLIB_TR_REGION_PREP     — prep blocks + first main TR (1 instance)
 *   PULSEQLIB_TR_REGION_MAIN     — one main TR (num_trs adjusted for
 *                                   non-degenerate prep/cooldown)
 *   PULSEQLIB_TR_REGION_COOLDOWN — last main TR + cooldown blocks (1 instance)
 *
 * @param[in]  coll          Loaded collection.
 * @param[out] out_pulses    Set to a malloc'd array; caller must free().
 * @param[in]  subseq_idx    Subsequence index.
 * @param[in]  region        PULSEQLIB_TR_REGION_PREP/_MAIN/_COOLDOWN.
 * @return Number of RF entries (≥ 0), or negative error code.
 */
int pulseqlib_get_rf_array(const pulseqlib_collection* coll,
                           pulseqlib_rf_stats** out_pulses,
                           int subseq_idx,
                           int region);

/** @brief Return 1 if block has an RF event. */
int pulseqlib_block_has_rf(const pulseqlib_collection* coll,
                           int seg_idx, int blk_idx);

/** @brief Return 1 if block's RF has uniform time raster. */
int pulseqlib_block_rf_has_uniform_raster(const pulseqlib_collection* coll,
                                          int seg_idx, int blk_idx);

/** @brief Return 1 if block's RF has a nonzero phase shape. */
int pulseqlib_block_rf_is_complex(const pulseqlib_collection* coll,
                                  int seg_idx, int blk_idx);

/** @brief Return RF per-channel sample count. */
int pulseqlib_get_rf_num_samples(const pulseqlib_collection* coll,
                                 int seg_idx, int blk_idx);

/** @brief Return number of RF channels (1 for standard, >1 for pTx). */
int pulseqlib_get_rf_num_channels(const pulseqlib_collection* coll,
                                  int seg_idx, int blk_idx);

/** @brief Return RF delay within block (us). */
int pulseqlib_get_rf_delay_us(const pulseqlib_collection* coll,
                              int seg_idx, int blk_idx);

/**
 * @brief Return decompressed RF magnitude waveform (multi-channel).
 *
 * Returns an array of num_channels pointers, each pointing to
 * num_samples floats.  For single-channel RF num_channels == 1.
 * On GEHC targets magnitudes are pre-scaled by base_amplitude_hz.
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
 * The number of time points equals num_samples from
 * pulseqlib_get_rf_magnitude.
 * Caller must free the returned array with PULSEQLIB_FREE.
 */
float* pulseqlib_get_rf_time_us(const pulseqlib_collection* coll,
                                int seg_idx, int blk_idx);

/* ================================================================== */
/*  Gradient getters                                                  */
/* ================================================================== */

/** @brief Return 1 if block has a gradient on the given axis. */
int pulseqlib_block_has_grad(const pulseqlib_collection* coll,
                             int seg_idx, int blk_idx, int axis);

/** @brief Return 1 if the gradient is a trapezoid (not arbitrary). */
int pulseqlib_block_grad_is_trapezoid(const pulseqlib_collection* coll,
                                      int seg_idx, int blk_idx, int axis);

/** @brief Return gradient waveform sample count. */
int pulseqlib_get_grad_num_samples(const pulseqlib_collection* coll,
                                   int seg_idx, int blk_idx, int axis);

/** @brief Return number of gradient shots. */
int pulseqlib_get_grad_num_shots(const pulseqlib_collection* coll,
                                 int seg_idx, int blk_idx, int axis);

/** @brief Return gradient delay within block (us). */
int pulseqlib_get_grad_delay_us(const pulseqlib_collection* coll,
                                int seg_idx, int blk_idx, int axis);

/**
 * @brief Return decompressed gradient amplitude waveforms (Hz/m).
 *
 * For multi-shot gradients, returns one waveform per shot.
 * Caller must free the returned array with PULSEQLIB_FREE.
 */
float** pulseqlib_get_grad_amplitude(const pulseqlib_collection* coll,
                                     int seg_idx, int blk_idx, int axis,
                                     int* num_shots,
                                     int** num_samples_per_shot);

/** @brief Return initial amplitude of a gradient event (Hz/m). */
float pulseqlib_get_grad_initial_amplitude_hz_per_m(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx, int axis);

/** @brief Return initial shot ID for a gradient event. */
int pulseqlib_get_grad_initial_shot_id(const pulseqlib_collection* coll,
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
/*  ADC getters                                                       */
/* ================================================================== */

/** @brief Return max ADC sample count across all ADC events. */
int pulseqlib_get_max_adc_samples(const pulseqlib_collection* coll);

/** @brief Return dwell time for an ADC event (us). */
int pulseqlib_get_adc_dwell_us(const pulseqlib_collection* coll,
                               int adc_idx);

/** @brief Return sample count for an ADC event. */
int pulseqlib_get_adc_num_samples(const pulseqlib_collection* coll,
                                  int adc_idx);

/** @brief Return 1 if block has an ADC event. */
int pulseqlib_block_has_adc(const pulseqlib_collection* coll,
                            int seg_idx, int blk_idx);

/** @brief Return ADC delay within block (us). */
int pulseqlib_get_adc_delay_us(const pulseqlib_collection* coll,
                               int seg_idx, int blk_idx);

/** @brief Return ADC library index for a block. */
int pulseqlib_get_adc_library_index(const pulseqlib_collection* coll,
                                    int seg_idx, int blk_idx);

/* ================================================================== */
/*  Digital output getters (block-level, OUTPUT-type triggers)        */
/* ================================================================== */

/** @brief Return 1 if block has a digital output event. */
int pulseqlib_block_has_digitalout(const pulseqlib_collection* coll,
                                   int seg_idx, int blk_idx);

/** @brief Return digital output delay within block (us). */
int pulseqlib_get_digitalout_delay_us(const pulseqlib_collection* coll,
                                      int seg_idx, int blk_idx);

/** @brief Return digital output duration within block (us). */
int pulseqlib_get_digitalout_duration_us(const pulseqlib_collection* coll,
                                         int seg_idx, int blk_idx);

/* ================================================================== */
/*  Physio trigger getters (segment-level, INPUT-type triggers)       */
/* ================================================================== */

/** @brief Return 1 if segment has a physio trigger. */
int pulseqlib_segment_has_trigger(const pulseqlib_collection* coll,
                                  int seg_idx);

/** @brief Return physio trigger delay (us) for a segment. */
int pulseqlib_get_segment_trigger_delay_us(const pulseqlib_collection* coll,
                                           int seg_idx);

/** @brief Return physio trigger duration (us) for a segment. */
int pulseqlib_get_segment_trigger_duration_us(const pulseqlib_collection* coll,
                                              int seg_idx);

/** @brief Return 1 if segment is a navigator (NAV) segment. */
int pulseqlib_segment_is_nav(const pulseqlib_collection* coll,
                             int seg_idx);

/** @brief Return 1 if block has a frequency modulation event. */
int pulseqlib_block_has_freq_mod(const pulseqlib_collection* coll,
                                 int seg_idx, int blk_idx);

/** @brief Return 1 if block has a rotation event. */
int pulseqlib_block_has_rotation(const pulseqlib_collection* coll,
                                 int seg_idx, int blk_idx);

/** @brief Return 1 if block has a no-rotation flag. */
int pulseqlib_block_has_norot(const pulseqlib_collection* coll,
                              int seg_idx, int blk_idx);

/** @brief Return 1 if block has a no-position flag. */
int pulseqlib_block_has_nopos(const pulseqlib_collection* coll,
                              int seg_idx, int blk_idx);

/* ================================================================== */
/*  Segment timing getters                                            */
/* ================================================================== */

/** @brief Return number of k-space zero-crossings in a segment. */
int pulseqlib_get_segment_num_kzero_crossings(
    const pulseqlib_collection* coll, int seg_idx);

/**
 * @brief Return RF-to-ADC gap within a segment (us).
 *
 * Finds the last RF anchor and the first following ADC anchor in the
 * segment and returns (adc_start - rf_end) in us.  Returns -1 if the
 * segment has no RF+ADC pair in that order.
 */
int pulseqlib_get_segment_rf_adc_gap_us(
    const pulseqlib_collection* coll, int seg_idx);

/**
 * @brief Return minimum ADC-to-ADC gap within a segment (us).
 *
 * For consecutive ADC anchors in the segment, returns the smallest
 * (next_adc_start - prev_adc_end).  Returns -1 if the segment has
 * fewer than 2 ADC events.
 */
int pulseqlib_get_segment_adc_adc_gap_us(
    const pulseqlib_collection* coll, int seg_idx);

/* ================================================================== */
/*  Label getters                                                     */
/* ================================================================== */

/** @brief Return label limits (min/max per label type) for a subsequence. */
int pulseqlib_get_label_limits(const pulseqlib_collection* coll,
                               int subseq_idx,
                               pulseqlib_label_limits* limits);

/** @brief Return number of ADC occurrences in the label table. */
int pulseqlib_get_num_adc_occurrences(const pulseqlib_collection* coll,
                                      int subseq_idx);

/** @brief Return number of label columns (vendor-dependent). */
int pulseqlib_get_num_label_columns(const pulseqlib_collection* coll,
                                    int subseq_idx);

/**
 * @brief Get label values for a specific ADC occurrence.
 *
 * @p out_values must point to a pre-allocated array of at least
 * num_label_columns ints.  For GEHC, the 3 columns are
 * [lin, slc, eco] in that order.
 *
 * @return PULSEQLIB_OK on success, negative error code on failure.
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

/** @brief Reset the cursor to the start of the sequence. */
void pulseqlib_cursor_reset(pulseqlib_collection* coll);

/**
 * @brief Get the resolved block instance at the current cursor position.
 * @return PULSEQLIB_OK on success, error code if cursor is done.
 */
int pulseqlib_get_block_instance(const pulseqlib_collection* coll,
                                 pulseqlib_block_instance*    inst);

/* ================================================================== */
/*  Frequency modulation library                                      */
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
 * @brief Build a frequency modulation library for one subsequence.
 *
 * Constructs deduped amplitude-scaled 3-channel gradient modulators
 * and computes shift-resolved 1D plan waveforms.
 *
 * For PMC-enabled subsequences the 3-channel data is retained so that
 * pulseqlib_update_freq_mod_library() can recompute waveforms with a
 * new shift at each TR boundary.  For non-PMC subsequences the
 * 3-channel data is discarded after the initial plan computation to
 * save memory.
 *
 * @param[out] lib         Receives an allocated library (caller frees).
 * @param[in]  coll        Loaded collection.
 * @param[in]  subseq_idx  0-based subsequence index.
 * @param[in]  shift_m     Spatial shift (dx, dy, dz) in metres.
 * @return PULSEQLIB_OK on success.
 */
int pulseqlib_build_freq_mod_library(
    pulseqlib_freq_mod_library** lib,
    const pulseqlib_collection* coll,
    int subseq_idx,
    const float* shift_m);

/**
 * @brief Recompute library waveforms with a new spatial shift.
 *
 * Only valid for PMC-enabled libraries (3-channel data is still
 * resident).  Returns an error if the 3-channel data was freed.
 *
 * @param[in,out] lib       Built library.
 * @param[in]     shift_m   New spatial shift (dx, dy, dz) in metres.
 * @return PULSEQLIB_OK on success.
 */
int pulseqlib_update_freq_mod_library(
    pulseqlib_freq_mod_library* lib,
    const float* shift_m);

/**
 * @brief Look up the freq-mod waveform for a scan-table position.
 *
 * @param[in]  lib              Built library.
 * @param[in]  scan_table_pos   Position in the subsequence scan table.
 * @param[out] out_waveform     Pointer into library (do NOT free).
 * @param[out] out_num_samples  Waveform length.
 * @param[out] out_phase_rad    Phase compensation (rad).
 * @return 1 if the block has a freq-mod event, 0 if not.
 */
int pulseqlib_freq_mod_library_get(
    const pulseqlib_freq_mod_library* lib,
    int scan_table_pos,
    const float** out_waveform,
    int* out_num_samples,
    float* out_phase_rad);

/**
 * @brief Write the shift-independent library data to a binary cache.
 *
 * @param[in]  lib   Built library (3-channel data must be resident).
 * @param[in]  path  Output file path (e.g. "seq.fmod.0.bin").
 * @return PULSEQLIB_OK on success.
 */
int pulseqlib_freq_mod_library_write_cache(
    const pulseqlib_freq_mod_library* lib,
    const char* path);

/**
 * @brief Read library from cache and compute plan for given shift.
 *
 * @param[out] lib         Receives an allocated library (caller frees).
 * @param[in]  path        Cache file path.
 * @param[in]  shift_m     Spatial shift for plan computation.
 * @param[in]  pmc_enabled If 0, 3-channel data is freed after plan.
 * @return PULSEQLIB_OK on success.
 */
int pulseqlib_freq_mod_library_read_cache(
    pulseqlib_freq_mod_library** lib,
    const char* path,
    const float* shift_m,
    int pmc_enabled);

/** @brief Free a frequency modulation library. */
void pulseqlib_freq_mod_library_free(pulseqlib_freq_mod_library* lib);

#ifdef __cplusplus
}
#endif

#endif /* PULSEQLIB_METHODS_H */
