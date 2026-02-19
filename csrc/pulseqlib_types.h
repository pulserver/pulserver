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
 *  Positive = success, negative = error.
 *  @{ */

#define PULSEQLIB_OK                          1

/* Generic errors (-1 to -9) */
#define PULSEQLIB_ERR_NULL_POINTER           -1
#define PULSEQLIB_ERR_INVALID_ARGUMENT       -2
#define PULSEQLIB_ERR_ALLOC_FAILED           -3

/* Parsing / file errors (-10 to -19) */
#define PULSEQLIB_ERR_FILE_NOT_FOUND        -10
#define PULSEQLIB_ERR_FILE_READ_FAILED      -11
#define PULSEQLIB_ERR_UNSUPPORTED_VERSION   -12
#define PULSEQLIB_ERR_PARSE_FAILED          -13

/* Unique-block errors (-50 to -59) */
#define PULSEQLIB_ERR_INVALID_PREP_POSITION      -50
#define PULSEQLIB_ERR_INVALID_COOLDOWN_POSITION  -51
#define PULSEQLIB_ERR_INVALID_ONCE_FLAGS         -52
#define PULSEQLIB_ERR_RASTER_MISMATCH            -53
#define PULSEQLIB_ERR_SIGNATURE_MISMATCH         -54
#define PULSEQLIB_ERR_SIGNATURE_MISSING          -55

/* TR detection errors (-100 to -199) */
#define PULSEQLIB_ERR_TR_NO_BLOCKS          -100
#define PULSEQLIB_ERR_TR_NO_IMAGING_REGION  -101
#define PULSEQLIB_ERR_TR_NO_PERIODIC_PATTERN -102
#define PULSEQLIB_ERR_TR_PATTERN_MISMATCH   -103
#define PULSEQLIB_ERR_TR_PREP_TOO_LONG      -104
#define PULSEQLIB_ERR_TR_COOLDOWN_TOO_LONG  -105

/* Segmentation errors (-200 to -299) */
#define PULSEQLIB_ERR_SEG_NONZERO_START_GRAD -200
#define PULSEQLIB_ERR_SEG_NONZERO_END_GRAD   -201
#define PULSEQLIB_ERR_SEG_NO_SEGMENTS_FOUND  -202
#define PULSEQLIB_ERR_TOO_MANY_GRAD_SHOTS    -203

/* Acoustic errors (-400 to -449) */
#define PULSEQLIB_ERR_ACOUSTIC_INVALID_WINDOW    -400
#define PULSEQLIB_ERR_ACOUSTIC_INVALID_RESOLUTION -401
#define PULSEQLIB_ERR_ACOUSTIC_NO_WAVEFORM       -402
#define PULSEQLIB_ERR_ACOUSTIC_FFT_FAILED        -403
#define PULSEQLIB_ERR_ACOUSTIC_VIOLATION         -404

/* PNS errors (-450 to -499) */
#define PULSEQLIB_ERR_PNS_INVALID_PARAMS         -450
#define PULSEQLIB_ERR_PNS_INVALID_CHRONAXIE      -451
#define PULSEQLIB_ERR_PNS_INVALID_RHEOBASE       -452
#define PULSEQLIB_ERR_PNS_NO_WAVEFORM            -453
#define PULSEQLIB_ERR_PNS_FFT_FAILED             -454
#define PULSEQLIB_ERR_PNS_THRESHOLD_EXCEEDED     -455

/* Collection errors (-500 to -559) */
#define PULSEQLIB_ERR_COLLECTION_EMPTY           -500
#define PULSEQLIB_ERR_COLLECTION_CHAIN_BROKEN    -501
#define PULSEQLIB_ERR_COLLECTION_MAX_DEPTH       -503
#define PULSEQLIB_ERR_MAX_GRAD_EXCEEDED          -550
#define PULSEQLIB_ERR_GRAD_DISCONTINUITY         -551
#define PULSEQLIB_ERR_MAX_SLEW_EXCEEDED          -552

/* Consistency errors (-560 to -569) */
#define PULSEQLIB_ERR_CONSISTENCY_SEG_MISMATCH   -560
#define PULSEQLIB_ERR_CONSISTENCY_RF_PERIODIC    -561
#define PULSEQLIB_ERR_CONSISTENCY_RF_SHIM_PERIODIC -562

#define PULSEQLIB_ERR_NOT_IMPLEMENTED      -999

/** @} */

/* Code checking macros */
#define PULSEQLIB_SUCCEEDED(code) ((code) > 0)
#define PULSEQLIB_FAILED(code)    ((code) < 0)

/* ================================================================== */
/*  Cursor states                                                     */
/* ================================================================== */
#define PULSEQLIB_CURSOR_BLOCK  0
#define PULSEQLIB_CURSOR_DONE   1

/* ================================================================== */
/*  Max-size constants                                                */
/* ================================================================== */
#define PULSEQLIB_MAX_GRAD_SHOTS       16
#define PULSEQLIB_MAX_RF_SHIM_CHANNELS 64
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

#define PULSEQLIB_DIAGNOSTIC_INIT {PULSEQLIB_OK, {'\0'}}

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
    float flip_angle_deg;   /**< nominal flip angle (degrees)           */
    float area;             /**< integral of |B1(t)| dt  (a.u.)        */
    float abs_width;        /**< fraction of duration with |B1|>0      */
    float eff_width;        /**< equivalent rectangular pulse fraction */
    float duty_cycle;       /**< fraction of TR occupied by RF         */
    float max_pulse_width;  /**< longest contiguous |B1|>0 segment (s) */
    float duration_us;      /**< total RF event duration (us)          */
    int   isodelay_us;      /**< isodelay from center to echo (us)     */
    float bandwidth_hz;     /**< estimated bandwidth (Hz, via FFT)     */
    float max_amplitude_hz; /**< peak |gamma*B1| (Hz)                  */
    int   num_samples;      /**< waveform sample count                 */
} pulseqlib_rf_stats;

#define PULSEQLIB_RF_STATS_INIT { \
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0, 0.0f, 0.0f, 0 \
}

/* ================================================================== */
/*  TR region selectors (for freq-mod plan)                           */
/* ================================================================== */
#define PULSEQLIB_TR_REGION_ALL      (-1)
#define PULSEQLIB_TR_REGION_PREP       0
#define PULSEQLIB_TR_REGION_MAIN       1
#define PULSEQLIB_TR_REGION_COOLDOWN   2

/* ================================================================== */
/*  Frequency modulation plan (opaque)                                */
/* ================================================================== */

/**
 * @brief Opaque handle to a precomputed frequency modulation plan.
 *
 * Created by pulseqlib_build_freq_mod_plan(), queried via
 * pulseqlib_get_freq_mod_waveform(), freed by
 * pulseqlib_freq_mod_plan_free().
 */
typedef struct pulseqlib_freq_mod_plan pulseqlib_freq_mod_plan;

/* ================================================================== */
/*  Segment timing anchors                                            */
/* ================================================================== */

/** @brief RF event anchor relative to segment start. */
typedef struct pulseqlib_segment_rf_anchor {
    int   block_offset;     /**< block index within segment           */
    float start_us;         /**< RF start relative to segment (us)    */
    float end_us;           /**< RF end relative to segment (us)      */
    float isocenter_us;     /**< RF isodelay point rel. to seg. (us)  */
    float base_amplitude_hz;/**< amplitude of first TR appearance (Hz)*/
} pulseqlib_segment_rf_anchor;

#define PULSEQLIB_SEGMENT_RF_ANCHOR_INIT {0, 0.0f, 0.0f, 0.0f, 0.0f}

/** @brief ADC event anchor relative to segment start. */
typedef struct pulseqlib_segment_adc_anchor {
    int   block_offset;     /**< block index within segment           */
    float start_us;         /**< ADC start relative to segment (us)   */
    float end_us;           /**< ADC end relative to segment (us)     */
    int   kzero_index;      /**< sample index of k=0 in readout      */
    float kzero_us;         /**< time of k=0 relative to segment (us) */
} pulseqlib_segment_adc_anchor;

#define PULSEQLIB_SEGMENT_ADC_ANCHOR_INIT {0, 0.0f, 0.0f, 0, 0.0f}

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

    /* Trigger */
    int   trigon_flag;          /**< 1 = trigger-on event present      */

    /* ADC */
    int   adc_flag;             /**< 1 = ADC acquisition active        */
    float adc_freq_hz;          /**< ADC frequency offset (Hz)         */
    float adc_phase_rad;        /**< ADC phase offset (rad)            */

    /* RF shimming */
    int   rf_shim_id;           /**< RF shim definition index (-1=none)*/
} pulseqlib_block_instance;

#define PULSEQLIB_BLOCK_INSTANCE_INIT { \
    0, \
    0.0f, 0.0f, 0.0f, \
    0.0f, 0.0f, 0.0f, \
    0, 0, 0, \
    {1,0,0, 0,1,0, 0,0,1}, 0, 0, \
    0, \
    0, 0.0f, 0.0f, \
    -1 \
}

/* ================================================================== */
/*  Scan-time query result                                            */
/* ================================================================== */

/**
 * @brief Quick scan-time summary (no full load required).
 *
 * Returned by pulseqlib_get_scan_time().  The total number of
 * segment boundaries equals the sum across subsequences of
 * (segments_per_TR * num_TRs).
 */
typedef struct pulseqlib_scan_time_info {
    float total_duration_us;        /**< total sequence duration (us)  */
    int   total_segment_boundaries; /**< total segment boundary count  */
} pulseqlib_scan_time_info;

#define PULSEQLIB_SCAN_TIME_INFO_INIT {0.0f, 0}

#endif /* PULSEQLIB_TYPES_H */
