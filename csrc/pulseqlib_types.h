/**
 * @file pulseqlib_types.h
 * @brief Public type definitions, error codes, and initializer macros.
 *
 * All types intended for consumption by calling code are defined here.
 * Internal / opaque types live in pulseqlib_internal.h.
 *
 * Naming conventions:
 *   - Physical quantities carry unit suffixes: _us, _hz, _hz_per_m, etc.
 *   - All identifiers are snake_case.
 *   - INIT macros are C89 and C++ compatible (positional, no designated init).
 */

#ifndef PULSEQLIB_TYPES_H
#define PULSEQLIB_TYPES_H

#include "pulseqlib_config.h"

/* ================================================================== */
/*  Gradient axes                                                     */
/* ================================================================== */
#define PULSEQLIB_GRAD_AXIS_X 0
#define PULSEQLIB_GRAD_AXIS_Y 1
#define PULSEQLIB_GRAD_AXIS_Z 2

/* ================================================================== */
/*  Error codes                                                       */
/* ================================================================== */

/** @defgroup errcodes Error codes
 *  Every public function returns a plain int:
 *    positive  = success (PULSEQLIB_OK)
 *    negative  = failure
 *
 *  On failure the caller should read the diagnostic message string
 *  (filled by every function that accepts a pulseqlib_diagnostic*)
 *  and pass it to the vendor error-reporting routine.  Specific
 *  negative values are library-internal and must NOT be matched by
 *  consumers.
 *  @{ */

#define PULSEQLIB_SUCCESS  1

#define PULSEQLIB_SUCCEEDED(code) ((code) > 0)
#define PULSEQLIB_FAILED(code)    ((code) < 0)

/** @} */

/* ================================================================== */
/*  Cursor states                                                     */
/* ================================================================== */
#define PULSEQLIB_CURSOR_BLOCK  0
#define PULSEQLIB_CURSOR_DONE   1

/* ================================================================== */
/*  Max-size constants                                                */
/* ================================================================== */
#define PULSEQLIB_MAX_GRAD_SHOTS       16
#define PULSEQLIB_DIAG_MSG_LEN        256

/* ================================================================== */
/*  Diagnostic                                                        */
/* ================================================================== */

/**
 * @brief Diagnostic info returned by library functions on failure.
 *
 * On error, @c code is set to a negative PULSEQLIB_ERR_* value and
 * @c message contains a human-readable description (may include
 * offending block index, axis, amplitude, etc.).
 */
typedef struct pulseqlib_diagnostic {
    int  code;
    char message[PULSEQLIB_DIAG_MSG_LEN];
} pulseqlib_diagnostic;

#define PULSEQLIB_DIAGNOSTIC_INIT {PULSEQLIB_SUCCESS, {'\0'}}

/* ================================================================== */
/*  System options                                                    */
/* ================================================================== */

/**
 * @brief Scanner hardware limits and raster times.
 *
 * All raster times are in microseconds.  Gradient / slew limits use
 * internal Pulseq units (Hz/m and Hz/m/s respectively).
 */
typedef struct pulseqlib_opts {
    int   vendor;                    /**< PULSEQLIB_VENDOR_* constant       */
    float gamma_hz_per_t;            /**< gyromagnetic ratio  (Hz / T)      */
    float b0_t;                      /**< static field strength (T)         */
    float max_grad_hz_per_m;         /**< gradient amplitude limit (Hz / m) */
    float max_slew_hz_per_m_per_s;   /**< slew rate limit (Hz / m / s)      */
    float rf_raster_us;              /**< RF sample raster (us)             */
    float grad_raster_us;            /**< gradient sample raster (us)       */
    float adc_raster_us;             /**< ADC dwell raster (us)             */
    float block_raster_us;           /**< block duration raster (us)        */
} pulseqlib_opts;

#define PULSEQLIB_OPTS_INIT {0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}

/* ================================================================== */
/*  RF statistics                                                     */
/* ================================================================== */

/**
 * @brief Per-RF-definition statistics (always available).
 */
typedef struct pulseqlib_rf_stats {
    float flip_angle_deg;       /**< nominal flip angle (degrees)           */
    float act_amplitude_hz;     /**< actual |gamma*B1| amplitude (Hz)       */
    float area;                 /**< integral of |B1(t)| dt  (a.u.)        */
    float abs_width;            /**< fraction of duration with |B1|>0      */
    float eff_width;            /**< equivalent rectangular pulse fraction */
    float duty_cycle;           /**< fraction of TR occupied by RF         */
    float max_pulse_width;      /**< longest contiguous |B1|>0 segment (s) */
    float duration_us;          /**< total RF event duration (us)          */
    int   isodelay_us;          /**< isodelay from center to echo (us)     */
    float bandwidth_hz;         /**< estimated bandwidth (Hz, via FFT)     */
    float base_amplitude_hz;    /**< base (nominal) peak |gamma*B1| (Hz)   */
    int   num_samples;          /**< waveform sample count                 */
    int   num_instances;        /**< repetition count for this RF pulse    */
} pulseqlib_rf_stats;

#define PULSEQLIB_RF_STATS_INIT { \
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0, 0.0f, 0.0f, 0, 0 \
}

/* ================================================================== */
/*  TR region selectors (for freq-mod plan)                           */
/* ================================================================== */
#define PULSEQLIB_TR_REGION_PREP       0
#define PULSEQLIB_TR_REGION_MAIN       1
#define PULSEQLIB_TR_REGION_COOLDOWN   2

/* ================================================================== */
/*  Frequency modulation collection                                   */
/* ================================================================== */

/**
 * @brief Opaque handle to frequency modulation data for all subsequences.
 *
 * Wraps per-subsequence libraries into a single object.  The entire
 * collection is built, cached, and freed as a unit.
 *
 * Created by pulseqlib_build_freq_mod_collection() or
 * pulseqlib_freq_mod_collection_read_cache(), queried via
 * pulseqlib_freq_mod_collection_get(), freed by
 * pulseqlib_freq_mod_collection_free().
 */
typedef struct pulseqlib_freq_mod_collection pulseqlib_freq_mod_collection;

/* ================================================================== */
/*  Opaque collection handle                                          */
/* ================================================================== */

/**
 * @brief Opaque handle to a loaded Pulseq sequence collection.
 *
 * Created by pulseqlib_read() or pulseqlib_read_from_buffers().
 * All getter functions take a const pointer to this type.
 * Freed by pulseqlib_collection_free().
 */
typedef struct pulseqlib_collection pulseqlib_collection;

/* ================================================================== */
/*  Per-axis gradient waveform (for plotting)                         */
/* ================================================================== */

/**
 * @brief Single-axis gradient waveform with per-sample segment label.
 *
 * Each element i represents a time-point with amplitude and the
 * segment it belongs to.  The time array is NOT interpolated to a
 * uniform raster -- it follows the native event timing.
 */
typedef struct pulseqlib_grad_axis_waveform {
    int    num_samples;           /**< number of time-points          */
    float* time_us;               /**< time of each sample (us)       */
    float* amplitude_hz_per_m;    /**< gradient amplitude (Hz / m)    */
    int*   seg_label;             /**< segment index for each sample  */
} pulseqlib_grad_axis_waveform;

#define PULSEQLIB_GRAD_AXIS_WAVEFORM_INIT {0, NULL, NULL, NULL}

/**
 * @brief Per-TR gradient waveforms for all three axes.
 *
 * Used for gradient-shape plotting in the wrapper.  Each axis carries
 * its own time base (not interpolated to a common raster).
 */
typedef struct pulseqlib_tr_gradient_waveforms {
    pulseqlib_grad_axis_waveform gx;
    pulseqlib_grad_axis_waveform gy;
    pulseqlib_grad_axis_waveform gz;
} pulseqlib_tr_gradient_waveforms;

#define PULSEQLIB_TR_GRADIENT_WAVEFORMS_INIT { \
    PULSEQLIB_GRAD_AXIS_WAVEFORM_INIT, \
    PULSEQLIB_GRAD_AXIS_WAVEFORM_INIT, \
    PULSEQLIB_GRAD_AXIS_WAVEFORM_INIT  \
}

/* ================================================================== */
/*  Native-timing TR waveforms (for plotting)                        */
/* ================================================================== */

/** @brief Amplitude modes for pulseqlib_get_tr_waveforms. */
#define PULSEQLIB_AMP_MAX_POS  0  /**< Position-max (safety worst case) */
#define PULSEQLIB_AMP_ZERO_VAR 1  /**< Zero variable-amplitude gradients, keep constant ones */
#define PULSEQLIB_AMP_ACTUAL   2  /**< Actual signed amplitude for given TR */

/**
 * @brief Single-channel waveform with native (non-uniform) timing.
 *
 * Both arrays have @c num_samples elements.  Units depend on the
 * channel:  Hz/m for gradients, Hz for RF magnitude, radians for
 * RF phase.
 */
typedef struct pulseqlib_channel_waveform {
    int    num_samples;
    float* time_us;       /**< [num_samples] */
    float* amplitude;     /**< [num_samples] */
} pulseqlib_channel_waveform;

#define PULSEQLIB_CHANNEL_WAVEFORM_INIT {0, NULL, NULL}

/**
 * @brief ADC event descriptor within a TR.
 */
typedef struct pulseqlib_adc_event {
    float onset_us;           /**< start time within TR (us)         */
    float duration_us;        /**< num_samples * dwell_time (us)     */
    int   num_samples;        /**< number of ADC samples             */
    float freq_offset_hz;     /**< per-instance freq offset (Hz)     */
    float phase_offset_rad;   /**< per-instance phase offset (rad)   */
} pulseqlib_adc_event;

#define PULSEQLIB_ADC_EVENT_INIT {0.0f, 0.0f, 0, 0.0f, 0.0f}

/**
 * @brief Per-block metadata within a TR.
 */
typedef struct pulseqlib_tr_block_descriptor {
    float start_us;           /**< block start time within TR (us)   */
    float duration_us;        /**< block duration (us)               */
    int   segment_idx;        /**< segment index, or -1 (prep/cooldown) */
} pulseqlib_tr_block_descriptor;

#define PULSEQLIB_TR_BLOCK_DESCRIPTOR_INIT {0.0f, 0.0f, -1}

/**
 * @brief Complete native-timing TR waveforms for plotting.
 *
 * Each channel carries its own time base.  Gradient channels
 * preserve native timing (trap corner-points, arb raster samples).
 * RF channels use the RF raster.  ADC events are descriptors only.
 *
 * Block descriptors provide timing and segment assignment for
 * drawing block/segment boundaries.
 */
typedef struct pulseqlib_tr_waveforms {
    /* Gradient channels (Hz/m) */
    pulseqlib_channel_waveform gx;
    pulseqlib_channel_waveform gy;
    pulseqlib_channel_waveform gz;

    /* RF channels (channel-0 only for pTx) */
    pulseqlib_channel_waveform rf_mag;    /**< amplitude in Hz           */
    pulseqlib_channel_waveform rf_phase;  /**< amplitude in rad          */

    /* ADC events */
    int                  num_adc_events;
    pulseqlib_adc_event* adc_events;

    /* Block-level metadata */
    int                          num_blocks;
    pulseqlib_tr_block_descriptor* blocks;

    /* Total duration */
    float total_duration_us;
} pulseqlib_tr_waveforms;

#define PULSEQLIB_TR_WAVEFORMS_INIT { \
    PULSEQLIB_CHANNEL_WAVEFORM_INIT, \
    PULSEQLIB_CHANNEL_WAVEFORM_INIT, \
    PULSEQLIB_CHANNEL_WAVEFORM_INIT, \
    PULSEQLIB_CHANNEL_WAVEFORM_INIT, \
    PULSEQLIB_CHANNEL_WAVEFORM_INIT, \
    0, NULL,  0, NULL,  0.0f \
}

/* ================================================================== */
/*  Acoustic spectra (for plotting)                                   */
/* ================================================================== */

/**
 * @brief Acoustic spectral data for wrapper-side plotting.
 *
 * Frequency axes are specified by (min, spacing, num_bins) so
 * the caller can reconstruct: freq[k] = freq_min_hz + k * freq_spacing_hz.
 *
 * Spectrograms are flat row-major arrays [num_windows * num_freq_bins].
 * Peak masks are binary (0 / 1) with the same layout.
 */
typedef struct pulseqlib_acoustic_spectra {
    /* -- sliding window -------------------------------------------- */
    float freq_min_hz;          /**< lowest frequency bin (Hz)         */
    float freq_spacing_hz;      /**< bin width (Hz)                    */
    int   num_freq_bins;        /**< frequency bins per window         */
    int   num_windows;          /**< number of sliding windows         */
    float* spectrogram_gx;      /**< [num_windows * num_freq_bins]     */
    float* spectrogram_gy;
    float* spectrogram_gz;
    int*   peaks_gx;            /**< binary peak mask (same shape)     */
    int*   peaks_gy;
    int*   peaks_gz;

    /* -- full TR spectrum ------------------------------------------ */
    float* spectrum_full_gx;    /**< [num_freq_bins]                   */
    float* spectrum_full_gy;
    float* spectrum_full_gz;
    int*   peaks_full_gx;       /**< binary peak mask [num_freq_bins]  */
    int*   peaks_full_gy;
    int*   peaks_full_gz;

    /* -- sequence-level harmonics ---------------------------------- */
    float  freq_spacing_seq_hz; /**< harmonic spacing (Hz)             */
    int    num_freq_bins_seq;   /**< number of harmonic bins           */
    float* spectrum_seq_gx;     /**< [num_freq_bins_seq]               */
    float* spectrum_seq_gy;
    float* spectrum_seq_gz;
    int*   peaks_seq_gx;        /**< binary peak mask                  */
    int*   peaks_seq_gy;
    int*   peaks_seq_gz;
} pulseqlib_acoustic_spectra;

#define PULSEQLIB_ACOUSTIC_SPECTRA_INIT { \
    0.0f, 0.0f, 0, 0,  NULL, NULL, NULL,  NULL, NULL, NULL, \
    NULL, NULL, NULL,  NULL, NULL, NULL, \
    0.0f, 0,  NULL, NULL, NULL,  NULL, NULL, NULL \
}

/* ================================================================== */
/*  Forbidden frequency band (for acoustic check)                     */
/* ================================================================== */

/**
 * @brief A forbidden acoustic frequency band.
 *
 * @c max_amplitude_hz_per_m is the maximum allowed gradient spectral
 * amplitude (in Hz / m) within the band [freq_min_hz, freq_max_hz].
 */
typedef struct pulseqlib_forbidden_band {
    float freq_min_hz;              /**< lower band edge (Hz)          */
    float freq_max_hz;              /**< upper band edge (Hz)          */
    float max_amplitude_hz_per_m;   /**< max spectral amplitude (Hz/m) */
} pulseqlib_forbidden_band;

#define PULSEQLIB_FORBIDDEN_BAND_INIT {0.0f, 0.0f, 0.0f}

/* ================================================================== */
/*  PNS parameters (vendor-independent)                               */
/* ================================================================== */

/**
 * @brief PNS model parameters.
 *
 * Set @c vendor to the appropriate PULSEQLIB_VENDOR_* constant.
 * Currently only PULSEQLIB_VENDOR_GEHC is implemented (exponential
 * model with chronaxie / rheobase / alpha).
 */
typedef struct pulseqlib_pns_params {
    int   vendor;                   /**< PULSEQLIB_VENDOR_* constant   */
    float chronaxie_us;             /**< nerve time constant (us)      */
    float rheobase_hz_per_m_per_s;  /**< threshold slew rate (Hz/m/s)  */
    float alpha;                    /**< model exponent (dimensionless) */
} pulseqlib_pns_params;

#define PULSEQLIB_PNS_PARAMS_INIT {0, 0.0f, 0.0f, 1.0f}

/* ================================================================== */
/*  PNS result (for plotting)                                         */
/* ================================================================== */

/**
 * @brief Convolved slew-rate waveforms per axis.
 *
 * The wrapper can compute combined PNS = sqrt(x^2+y^2+z^2) and
 * percentage = slew / rheobase.  This avoids duplicating model
 * logic across languages.
 */
typedef struct pulseqlib_pns_result {
    int    num_samples;
    float* slew_x_hz_per_m_per_s;   /**< convolved dG/dt on X (Hz/m/s) */
    float* slew_y_hz_per_m_per_s;   /**< convolved dG/dt on Y (Hz/m/s) */
    float* slew_z_hz_per_m_per_s;   /**< convolved dG/dt on Z (Hz/m/s) */
} pulseqlib_pns_result;

#define PULSEQLIB_PNS_RESULT_INIT {0, NULL, NULL, NULL}

/* ================================================================== */
/*  Label limits                                                      */
/* ================================================================== */

typedef struct pulseqlib_label_limit {
    int min;
    int max;
} pulseqlib_label_limit;

typedef struct pulseqlib_label_limits {
    pulseqlib_label_limit slc;
    pulseqlib_label_limit phs;
    pulseqlib_label_limit rep;
    pulseqlib_label_limit avg;
    pulseqlib_label_limit seg;
    pulseqlib_label_limit set;
    pulseqlib_label_limit eco;
    pulseqlib_label_limit par;
    pulseqlib_label_limit lin;
    pulseqlib_label_limit acq;
} pulseqlib_label_limits;

/* ================================================================== */
/*  Block instance (cursor output)                                    */
/* ================================================================== */

/**
 * @brief Resolved block data for the current cursor position.
 *
 * Returned by pulseqlib_get_block_instance().  Amplitudes are in
 * Pulseq native units (Hz for RF, Hz/m for gradients).
 */
typedef struct pulseqlib_block_instance {
    int   duration_us;          /**< block duration (us)                */

    /* RF */
    float rf_amp_hz;            /**< RF amplitude (Hz, = gamma*B1)     */
    float rf_freq_hz;           /**< RF frequency offset (Hz)          */
    float rf_phase_rad;         /**< RF phase offset (rad)             */
    int   rf_shim_id;           /**< RF shim definition index (-1=none)*/

    /* Gradients */
    float gx_amp_hz_per_m;      /**< GX amplitude (Hz / m)             */
    float gy_amp_hz_per_m;      /**< GY amplitude (Hz / m)             */
    float gz_amp_hz_per_m;      /**< GZ amplitude (Hz / m)             */
    int   gx_shot_idx;          /**< GX multi-shot index               */
    int   gy_shot_idx;          /**< GY multi-shot index               */
    int   gz_shot_idx;          /**< GZ multi-shot index               */

    /* Rotation */
    float rotmat[9];            /**< 3x3 rotation matrix (row-major)   */
    int   norot_flag;           /**< 1 = skip rotation for this block  */
    int   nopos_flag;           /**< 1 = skip repositioning            */

    /* Digital output */
    int   digitalout_flag;      /**< 1 = digital output event present  */

    /* ADC */
    int   adc_flag;             /**< 1 = ADC acquisition active        */
    float adc_freq_hz;          /**< ADC frequency offset (Hz)         */
    float adc_phase_rad;        /**< ADC phase offset (rad)            */
} pulseqlib_block_instance;

#define PULSEQLIB_BLOCK_INSTANCE_INIT { \
    0, \
    0.0f, 0.0f, 0.0f, -1, \
    0.0f, 0.0f, 0.0f, \
    0, 0, 0, \
    {1,0,0, 0,1,0, 0,0,1}, 0, 0, \
    0, \
    0, 0.0f, 0.0f, \
}

/* ================================================================== */
/*  Cursor info                                                       */
/* ================================================================== */

/**
 * @brief Position and context metadata for the current cursor block.
 *
 * Returned by pulseqlib_cursor_get_info() after a successful
 * pulseqlib_cursor_next() call.
 */
typedef struct pulseqlib_cursor_info {
    int subseq_idx;       /**< current subsequence index                     */
    int scan_pos;         /**< scan-table position (for freq-mod lookup)     */
    int segment_id;       /**< current segment ID (global)                   */
    int segment_start;    /**< 1 if first block of current segment           */
    int segment_end;      /**< 1 if last block of current segment            */
    int is_nav;           /**< 1 if current segment is a NAV segment         */
    int has_trigger;      /**< 1 if current segment has a trigger/digitalout */
    int tr_start;         /**< 1 if first block of a main-region TR          */
    int pmc;              /**< 1 if current subsequence has PMC enabled      */
} pulseqlib_cursor_info;

#define PULSEQLIB_CURSOR_INFO_INIT {0, 0, -1, 0, 0, 0, 0, 0, 0}

/* ================================================================== */
/*  Scan-time query result                                            */
/* ================================================================== */

/**
 * @brief Scan-time summary.
 *
 * When returned by pulseqlib_peek_scan_time(), only
 * @c total_duration_us is populated (approximated from the
 * [DEFINITIONS] section, multiplied by @c num_reps with
 * per-subsequence @c IgnoreAverages clamping) and
 * @c total_segment_boundaries is left at 0.
 *
 * When computed from a fully-loaded collection via
 * pulseqlib_get_scan_time(), both fields are accurate and
 * account for prep/cooldown block durations, degenerate
 * TR folding, and the consumer-supplied @c num_reps.
 */
typedef struct pulseqlib_scan_time_info {
    float total_duration_us;        /**< total sequence duration (us)  */
    int   total_segment_boundaries; /**< total segment boundary count  */
} pulseqlib_scan_time_info;

#define PULSEQLIB_SCAN_TIME_INFO_INIT {0.0f, 0}

/* ================================================================== */
/*  Collection info (replaces individual collection-level getters)    */
/* ================================================================== */

/**
 * @brief Summary information about a loaded collection.
 *
 * Returned by pulseqlib_get_collection_info().
 */
typedef struct pulseqlib_collection_info {
    int   num_subsequences;     /**< number of subsequences              */
    int   num_segments;         /**< total unique segments               */
    int   max_adc_samples;      /**< max sample count across all ADCs    */
    int   total_readouts;       /**< total ADC readout events            */
    float total_duration_us;    /**< total sequence duration (us)        */
} pulseqlib_collection_info;

#define PULSEQLIB_COLLECTION_INFO_INIT {0, 0, 0, 0, 0.0f}

/* ================================================================== */
/*  Subsequence info (replaces per-subsequence getters)               */
/* ================================================================== */

/**
 * @brief Metadata for a single subsequence.
 *
 * Returned by pulseqlib_get_subseq_info().
 */
typedef struct pulseqlib_subseq_info {
    float tr_duration_us;       /**< TR duration (us)                    */
    int   num_trs;              /**< number of TRs                       */
    int   tr_size;              /**< blocks per TR                       */
    int   num_prep_blocks;      /**< preparation blocks before first TR  */
    int   num_cooldown_blocks;  /**< cooldown blocks after last TR       */
    int   num_prep_trs;         /**< preparation TRs                     */
    int   num_cooldown_trs;     /**< cooldown TRs                        */
    int   degenerate_prep;      /**< 1 if prep == first TR               */
    int   degenerate_cooldown;  /**< 1 if cooldown == last TR            */
    int   num_unique_adcs;      /**< unique ADC definitions              */
    int   num_unique_rf;        /**< unique RF definitions               */
    int   pmc_enabled;          /**< 1 if PMC (prospective motion corr)  */
    int   segment_offset;       /**< global segment index offset         */
    int   num_prep_segments;    /**< segments in prep region             */
    int   num_main_segments;    /**< segments in main TR region          */
    int   num_cooldown_segments;/**< segments in cooldown region         */
    int   num_adc_occurrences;  /**< ADC entries in label table          */
    int   num_label_columns;    /**< label columns (vendor-dependent)    */
    int   num_passes;           /**< number of inner-loop passes (>=1)   */
} pulseqlib_subseq_info;

#define PULSEQLIB_SUBSEQ_INFO_INIT { \
    0.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 \
}

/* ================================================================== */
/*  Segment info (replaces per-segment getters)                       */
/* ================================================================== */

/**
 * @brief Metadata for a single segment.
 *
 * Returned by pulseqlib_get_segment_info().
 */
typedef struct pulseqlib_segment_info {
    int   duration_us;          /**< total segment duration (us)         */
    int   num_blocks;           /**< unique blocks in the segment        */
    int   start_block;          /**< start block index in the sequence   */
    int   pure_delay;           /**< 1 if segment is a bare delay        */
    int   has_trigger;          /**< 1 if physio trigger attached        */
    int   trigger_delay_us;     /**< trigger delay (us), -1 if none      */
    int   trigger_duration_us;  /**< trigger duration (us), -1 if none   */
    int   is_nav;               /**< 1 if navigator segment              */
    int   num_kzero_crossings;  /**< k-space zero-crossings              */
    int   rf_adc_gap_us;        /**< RF->ADC gap (us), -1 if no pair     */
    int   adc_adc_gap_us;       /**< min ADC->ADC gap (us), -1 if < 2    */
} pulseqlib_segment_info;

#define PULSEQLIB_SEGMENT_INFO_INIT { \
    0, 0, 0, 0, 0, -1, -1, 0, 0, -1, -1 \
}

/* ================================================================== */
/*  Block info (replaces per-block has/get accessor pairs)            */
/* ================================================================== */

/**
 * @brief Metadata for a single block within a segment.
 *
 * Returned by pulseqlib_get_block_info().
 * Waveform data is NOT included — use the dedicated waveform getters
 * (e.g.\ pulseqlib_get_grad_amplitude) keyed by the metadata here.
 */
typedef struct pulseqlib_block_info {
    int   duration_us;            /**< block duration (us)               */
    int   start_time_us;          /**< start time within segment (us)    */

    /* Gradient (per axis: [0]=X, [1]=Y, [2]=Z) */
    int   has_grad[3];            /**< 1 if gradient present             */
    int   grad_is_trapezoid[3];   /**< 1 if trapezoid (not arbitrary)    */
    int   grad_delay_us[3];       /**< gradient delay (us), -1 if absent */
    int   grad_num_shots[3];      /**< shot count, -1 if absent          */
    int   grad_num_samples[3];    /**< sample count, -1 if absent        */

    /* RF */
    int   has_rf;                 /**< 1 if RF event present             */
    int   rf_delay_us;            /**< RF delay (us), -1 if absent       */
    int   rf_num_channels;        /**< Tx channel count, -1 if absent    */
    int   rf_num_samples;         /**< samples per channel, -1 if absent */
    int   rf_is_complex;          /**< 1 if phase shape exists           */
    int   rf_uniform_raster;      /**< 1 if time shape present           */

    /* ADC */
    int   has_adc;                /**< 1 if ADC acquisition active       */
    int   adc_delay_us;           /**< ADC delay (us), -1 if absent      */
    int   adc_def_id;             /**< global ADC library index, -1      */

    /* Digital output */
    int   has_digitalout;         /**< 1 if digital output present       */
    int   digitalout_delay_us;    /**< delay (us), -1 if absent          */
    int   digitalout_duration_us; /**< duration (us), -1 if absent       */

    /* Flags */
    int   has_rotation;           /**< 1 if rotation event present       */
    int   norot_flag;             /**< 1 if no-rotation override         */
    int   nopos_flag;             /**< 1 if no-position override         */
    int   has_freq_mod;           /**< 1 if frequency modulation present */
} pulseqlib_block_info;

#define PULSEQLIB_BLOCK_INFO_INIT { \
    0, 0, \
    {0,0,0}, {0,0,0}, {-1,-1,-1}, {-1,-1,-1}, {-1,-1,-1}, \
    0, -1, -1, -1, 0, 0, \
    0, -1, -1, \
    0, -1, -1, \
    0, 0, 0, 0 \
}

/* ================================================================== */
/*  ADC definition (replaces per-ADC getters)                         */
/* ================================================================== */

/**
 * @brief Information about a unique ADC definition.
 *
 * Returned by pulseqlib_get_adc_def().
 */
typedef struct pulseqlib_adc_def {
    int   dwell_ns;             /**< dwell time (ns)                     */
    int   num_samples;          /**< sample count                        */
} pulseqlib_adc_def;

#define PULSEQLIB_ADC_DEF_INIT {0, 0}

#endif /* PULSEQLIB_TYPES_H */
