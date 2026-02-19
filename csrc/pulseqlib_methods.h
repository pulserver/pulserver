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
 * @return PULSEQLIB_OK on success, negative error code on failure.
 */
int pulseqlib_read(
    pulseqlib_collection** out_coll,
    pulseqlib_diagnostic*  diag,
    const char*            file_path,
    const pulseqlib_opts*  opts,
    int                    cache_binary,
    int                    verify_signature,
    int                    parse_labels);

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
 * @return PULSEQLIB_OK on success, negative error code on failure.
 */
int pulseqlib_read_from_buffers(
    pulseqlib_collection** out_coll,
    pulseqlib_diagnostic*  diag,
    const char* const*     buffers,
    const int*             buffer_sizes,
    int                    num_buffers,
    const pulseqlib_opts*  opts,
    int                    parse_labels);

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
/*  Scan-time query                                                   */
/* ================================================================== */

/**
 * @brief Fast scan-time estimate without full sequence loading.
 *
 * Reads only the definitions sections from a (possibly chained) .seq
 * file to compute total duration and segment count.
 *
 * @param[out] info       Receives scan time summary.
 * @param[in]  file_path  Path to the first .seq file.
 * @param[in]  opts       Scanner limits (used for chain traversal).
 * @return PULSEQLIB_OK on success, negative error code on failure.
 */
int pulseqlib_get_scan_time(
    pulseqlib_scan_time_info* info,
    const char*               file_path,
    const pulseqlib_opts*     opts);

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

/** @brief Return number of unique ADC events in a subsequence. */
int pulseqlib_get_num_unique_adcs(const pulseqlib_collection* coll,
                                  int subseq_idx);

/** @brief Return total sequence duration (us). */
float pulseqlib_get_total_duration_us(const pulseqlib_collection* coll);

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
 * On GEHC targets magnitudes are pre-scaled by max_amplitude_hz.
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
                                int seg_idx, int blk_idx,
                                int* num_samples);

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
 * Caller must free the returned array with PULSEQLIB_FREE.
 */
float* pulseqlib_get_grad_time_us(const pulseqlib_collection* coll,
                                  int seg_idx, int blk_idx, int axis,
                                  int* num_samples);

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
/*  Flow control getters                                              */
/* ================================================================== */

/** @brief Return 1 if block has a trigger event. */
int pulseqlib_block_has_trigger(const pulseqlib_collection* coll,
                                int seg_idx, int blk_idx);

/** @brief Return trigger delay within block (us). */
int pulseqlib_get_trigger_delay_us(const pulseqlib_collection* coll,
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

/** @brief Return number of RF anchors in a segment. */
int pulseqlib_get_segment_num_rf_anchors(const pulseqlib_collection* coll,
                                         int seg_idx);

/** @brief Get an RF anchor for a segment. */
int pulseqlib_get_segment_rf_anchor(const pulseqlib_collection* coll,
                                    int seg_idx, int rf_idx,
                                    pulseqlib_segment_rf_anchor* out);

/** @brief Return number of ADC anchors in a segment. */
int pulseqlib_get_segment_num_adc_anchors(const pulseqlib_collection* coll,
                                          int seg_idx);

/** @brief Get an ADC anchor for a segment. */
int pulseqlib_get_segment_adc_anchor(const pulseqlib_collection* coll,
                                     int seg_idx, int adc_idx,
                                     pulseqlib_segment_adc_anchor* out);

/** @brief Return number of k-space zero-crossings in a segment. */
int pulseqlib_get_segment_num_kzero_crossings(
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
/*  Frequency modulation plan                                         */
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
 * @brief Build a precomputed frequency modulation plan.
 *
 * @param[out] plan      Receives an allocated opaque plan (caller frees).
 * @param[in]  coll      Loaded collection.
 * @param[in]  shift_m   Spatial shift (dx, dy, dz) in metres.
 * @param[in]  tr_type   PULSEQLIB_TR_REGION_ALL / _PREP / _MAIN / _COOLDOWN.
 * @param[in]  tr_index  0-based TR instance (ignored when ALL/PREP/COOLDOWN).
 * @return PULSEQLIB_OK on success.
 */
int pulseqlib_build_freq_mod_plan(
    pulseqlib_freq_mod_plan** plan,
    const pulseqlib_collection* coll,
    const float* shift_m,
    int tr_type, int tr_index);

/**
 * @brief Recompute waveforms in-place with a new spatial shift.
 *
 * No allocation/free — reuses existing plan memory.  Use for
 * prospective motion correction (call once per TR).
 */
int pulseqlib_update_freq_mod_plan(pulseqlib_freq_mod_plan* plan,
                                   const float* shift_m);

/** @brief Free a frequency modulation plan. */
void pulseqlib_freq_mod_plan_free(pulseqlib_freq_mod_plan* plan);

/**
 * @brief Get the freq-mod waveform for a specific block.
 *
 * @param[in]  plan            Built plan.
 * @param[in]  block_idx       Absolute block index.
 * @param[out] out_waveform    Pointer into plan memory (do NOT free).
 * @param[out] out_num_samples Number of waveform samples.
 * @param[out] out_phase_rad   Phase compensation (rad).
 * @return 1 if the block has a freq-mod event, 0 if not.
 */
int pulseqlib_get_freq_mod_waveform(
    const pulseqlib_freq_mod_plan* plan,
    int block_idx,
    const float** out_waveform,
    int* out_num_samples,
    float* out_phase_rad);

#ifdef __cplusplus
}
#endif

#endif /* PULSEQLIB_METHODS_H */
