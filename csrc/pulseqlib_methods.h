/* pulseqlib.h -- public API for the Pulseq interpreter library
 *
 * Include this header in application code.  It pulls in:
 *   pulseqlib_config.h  (vendor / allocator macros)
 *   pulseqlib_types.h   (public types, error codes)
 *
 * All functions use the pulseqlib_ prefix and are declared
 * extern "C" when compiled with a C++ compiler.
 */

#ifndef PULSEQLIB_H
#define PULSEQLIB_H

#include "pulseqlib_config.h"
#include "pulseqlib_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  High-level loading                                                */
/* ------------------------------------------------------------------ */

/**
 * Load a (possibly chained) Pulseq sequence from disk, returning a
 * fully populated descriptor collection ready for accessor queries.
 *
 * @param collection       Output: populated descriptor collection.
 * @param diag             Output: diagnostic (error details on failure).
 * @param file_path        Path to the first .seq file.
 * @param opts             System limits / rasters.
 * @param cache_binary     1 = write/read .bin alongside .seq.
 * @param verify_signature 1 = verify MD5 signature for every .seq file
 *                         in the chain before loading.  Use on scanner;
 *                         skip (0) during development or quick queries.
 * @return PULSEQLIB_OK on success, negative error code on failure.
 */
int pulseqlib_load(
    pulseqlib_sequence_descriptor_collection* collection,
    pulseqlib_diagnostic* diag,
    const char* file_path,
    const pulseqlib_opts* opts,
    int cache_binary,
    int verify_signature);

/* ------------------------------------------------------------------ */
/*  Opts                                                              */
/* ------------------------------------------------------------------ */
void pulseqlib_opts_init(
    pulseqlib_opts* opts,
    float gamma, float b0,
    float max_grad, float max_slew,
    float rf_raster_time, float grad_raster_time,
    float adc_raster_time, float block_duration_raster);

/* ------------------------------------------------------------------ */
/*  Diagnostic helpers                                                */
/* ------------------------------------------------------------------ */
void pulseqlib_diagnostic_init(pulseqlib_diagnostic* diag);
const char* pulseqlib_get_error_message(int code);
const char* pulseqlib_get_error_hint(int code);

/* ------------------------------------------------------------------ */
/*  Lightweight scan-time query                                       */
/* ------------------------------------------------------------------ */

/**
 * Read only the definitions sections from a (possibly chained) .seq file
 * to extract per-subsequence duration and ignore_averages flag.
 * Much faster than pulseqlib_load -- no block/event/shape parsing.
 *
 * @param info       Output: populated scan time info (caller must free).
 * @param file_path  Path to the first .seq file.
 * @param opts       System limits (only used for chain traversal).
 * @return PULSEQLIB_OK on success, negative error code on failure.
 */
int pulseqlib_query_scan_time(
    pulseqlib_scan_time_info* info,
    const char* file_path,
    const pulseqlib_opts* opts);

void pulseqlib_scan_time_info_free(pulseqlib_scan_time_info* info);

/* ------------------------------------------------------------------ */
/*  Descriptor lifecycle                                              */
/* ------------------------------------------------------------------ */
void pulseqlib_sequence_descriptor_free(pulseqlib_sequence_descriptor* desc);
void pulseqlib_sequence_descriptor_collection_free(
    pulseqlib_sequence_descriptor_collection* coll);
void pulseqlib_segment_table_result_free(pulseqlib_segment_table_result* result);

/* ------------------------------------------------------------------ */
/*  Gradient waveforms                                                */
/* ------------------------------------------------------------------ */

int pulseqlib_get_tr_gradient_waveforms(
    const pulseqlib_sequence_descriptor* desc,
    pulseqlib_tr_gradient_waveforms* waveforms,
    pulseqlib_diagnostic* diag
);
void pulseqlib_tr_gradient_waveforms_free(pulseqlib_tr_gradient_waveforms* w);

/* ------------------------------------------------------------------ */
/*  Safety checks                                                     */
/* ------------------------------------------------------------------ */
int pulseqlib_check_safety(
    pulseqlib_sequence_descriptor_collection* coll,   /* non-const: cursor dry-run */
    pulseqlib_diagnostic* diag,
    const pulseqlib_opts* opts,
    int num_forbidden_bands,
    const int* band_freq_min_hz,
    const int* band_freq_max_hz,
    const float* band_amp_max_tesla_m,
    int pns_decay_time_us,
    float pns_stim_threshold_tesla_m_s);

/* ------------------------------------------------------------------ */
/*  Acoustic analysis                                                 */
/* ------------------------------------------------------------------ */
int pulseqlib_get_tr_acoustic_spectra(
    pulseqlib_tr_acoustic_spectra* spectra,
    pulseqlib_diagnostic* diag,
    const pulseqlib_tr_gradient_waveforms* waveforms,
    float grad_raster_time_us,
    int target_window_size,
    float target_spectral_resolution_hz,
    float max_frequency_hz,
    int combined,
    int num_trs,
    float tr_duration_us,
    int num_forbidden_bands,
    const pulseqlib_forbidden_band* forbidden_bands,
    int store_results);
void pulseqlib_tr_acoustic_spectra_free(pulseqlib_tr_acoustic_spectra* s);

/* ------------------------------------------------------------------ */
/*  PNS analysis                                                      */
/* ------------------------------------------------------------------ */
int pulseqlib_compute_pns(
    pulseqlib_pns_result* result,
    pulseqlib_diagnostic* diag,
    float gamma_hz_per_tesla,
    float pns_threshold,
    const pulseqlib_tr_gradient_waveforms* waveforms,
    float grad_raster_time_us,
    const pulseqlib_pns_params* params,
    int store_waveforms);
void pulseqlib_pns_result_free(pulseqlib_pns_result* r);

/* ------------------------------------------------------------------ */
/*  Collection-level accessors                                        */
/* ------------------------------------------------------------------ */
/* Subsequence queries */
int pulseqlib_get_num_subsequences(
    const pulseqlib_sequence_descriptor_collection* coll);
float pulseqlib_get_tr_duration(
    const pulseqlib_sequence_descriptor_collection* coll,
    int subseq_idx);

/* Segment queries */
int pulseqlib_get_num_segments(
    const pulseqlib_sequence_descriptor_collection* coll);
int pulseqlib_get_segment_duration(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx);
int pulseqlib_is_segment_pure_delay(
    const pulseqlib_sequence_descriptor_collection* coll, int seg_idx);
int pulseqlib_get_segment_num_blocks(
    const pulseqlib_sequence_descriptor_collection* coll, int seg_idx);

/* Block queries within segments */
int pulseqlib_get_block_start_time(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx);
int pulseqlib_get_block_duration(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx);

/* RF queries */
int pulseqlib_get_num_unique_rf(
    const pulseqlib_sequence_descriptor_collection* coll,
    int subseq_idx);
#if PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_GEHC
int pulseqlib_get_rf_stats(
    const pulseqlib_sequence_descriptor_collection* coll,
    pulseqlib_rf_stats* stats,
    int subseq_idx, int rf_idx);
float pulseqlib_get_rf_base_amplitude(
    const pulseqlib_sequence_descriptor_collection* coll,
    int subseq_idx, int rf_idx);
#endif
int pulseqlib_get_tr_rf_ids(
    const pulseqlib_sequence_descriptor_collection* coll,
    int* out_rf_ids,
    int subseq_idx);
int pulseqlib_block_has_rf(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx);
int pulseqlib_block_rf_has_uniform_raster(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx);
int pulseqlib_block_rf_is_complex(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx);
int pulseqlib_get_rf_num_samples(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx);
int pulseqlib_get_rf_delay(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx);
float* pulseqlib_get_rf_magnitude(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx, int* num_samples);
float* pulseqlib_get_rf_phase(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx, int* num_samples);
float* pulseqlib_get_rf_time(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx, int* num_samples);

/* Gradient queries */
int pulseqlib_block_has_grad(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx, int axis);
int pulseqlib_block_grad_is_trapezoid(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx, int axis);
int pulseqlib_get_grad_num_samples(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx, int axis);
int pulseqlib_get_grad_num_shots(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx, int axis);
int pulseqlib_get_grad_delay(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx, int axis);
float** pulseqlib_get_grad_amplitude(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx, int axis,
    int* num_shots, int** num_samples_per_shot);
float pulseqlib_get_grad_initial_amplitude(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx, int axis);
int pulseqlib_get_grad_initial_shot_id(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx, int axis);
float* pulseqlib_get_grad_time(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx, int axis, int* num_samples);
    
/* ADC queries */
int pulseqlib_get_max_adc_samples(
    const pulseqlib_sequence_descriptor_collection* coll);
int pulseqlib_get_adc_dwell(
    const pulseqlib_sequence_descriptor_collection* coll, int adc_idx);
int pulseqlib_get_adc_num_samples(
    const pulseqlib_sequence_descriptor_collection* coll, int adc_idx);
int pulseqlib_block_has_adc(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx);
int pulseqlib_get_adc_delay(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx);
int pulseqlib_get_adc_library_index(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx);

/* Flow control queries */
int pulseqlib_block_has_trigger(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx);
int pulseqlib_get_trigger_delay(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx);
int pulseqlib_block_has_rotation(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx);
int pulseqlib_block_has_norot(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx);
int pulseqlib_block_has_nopos(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx);

/* Segment timing queries */
int pulseqlib_get_segment_num_rf_anchors(
    const pulseqlib_sequence_descriptor_collection* coll, int seg_idx);
int pulseqlib_get_segment_rf_anchor(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int rf_idx,
    pulseqlib_segment_rf_anchor* out);
int pulseqlib_get_segment_num_adc_anchors(
    const pulseqlib_sequence_descriptor_collection* coll, int seg_idx);
int pulseqlib_get_segment_adc_anchor(
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int adc_idx,
    pulseqlib_segment_adc_anchor* out);
int pulseqlib_get_segment_num_kzero_crossings(
    const pulseqlib_sequence_descriptor_collection* coll, int seg_idx);

/* ------------------------------------------------------------------ */
/*  Block cursor / iterator                                           */
/* ------------------------------------------------------------------ */
int pulseqlib_cursor_next(
    pulseqlib_sequence_descriptor_collection* coll);
void pulseqlib_cursor_reset(
    pulseqlib_sequence_descriptor_collection* coll);
int pulseqlib_get_block_instance(
    const pulseqlib_sequence_descriptor_collection* coll,
    pulseqlib_block_instance* inst);

/* ------------------------------------------------------------------ */
/*  Frequency modulation plan                                         */
/* ------------------------------------------------------------------ */

/* Total number of RF+ADC events across the entire sequence */
int pulseqlib_count_freq_mod_events(
    const pulseqlib_sequence_descriptor_collection* coll);

/* Number of RF+ADC events in a specific TR region.
 * tr_type: PULSEQLIB_TR_REGION_PREP / _MAIN / _COOLDOWN
 * tr_index: 0-based TR instance index (ignored for PREP/COOLDOWN) */
int pulseqlib_count_freq_mod_events_tr(
    const pulseqlib_sequence_descriptor_collection* coll,
    int tr_type, int tr_index);

/* Build a frequency modulation plan for a given spatial shift.
 * shift[3]: (dx, dy, dz) in metres.
 * tr_type: PULSEQLIB_TR_REGION_ALL / _PREP / _MAIN / _COOLDOWN
 * tr_index: 0-based TR instance (ignored when tr_type == ALL or PREP/COOLDOWN).
 * Returns PULSEQLIB_OK on success. */
int pulseqlib_build_freq_mod_plan(
    pulseqlib_freq_mod_plan* plan,
    const pulseqlib_sequence_descriptor_collection* coll,
    const float* shift,
    int tr_type, int tr_index);

/* Recompute waveforms and phase offsets in-place with a new shift.
 * The plan must have been previously built with pulseqlib_build_freq_mod_plan.
 * Reuses all existing allocations -- no malloc/free.
 * Use this for prospective motion correction (call once per TR). */
int pulseqlib_update_freq_mod_plan(
    pulseqlib_freq_mod_plan* plan,
    const float* shift);

void pulseqlib_freq_mod_plan_free(pulseqlib_freq_mod_plan* plan);

#ifdef __cplusplus
}
#endif

#endif /* PULSEQLIB_H */