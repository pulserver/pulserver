/* pulseqlib_types.h -- public type definitions, constants, and init macros
 *
 * All types intended for consumption by the caller are defined here.
 * Internal / opaque types live in pulseqlib_internal.h.
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

/* Success */
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

/* Selective excitation errors (-300 to -399) -- REMOVED:
 * Explicit frequency modulation eliminates the need for these checks.
 * Error codes reserved but no longer used.
 */

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

/* Collection errors (-500 to -509) */
#define PULSEQLIB_ERR_COLLECTION_EMPTY           -500
#define PULSEQLIB_ERR_COLLECTION_CHAIN_BROKEN    -501
#define PULSEQLIB_ERR_COLLECTION_MAX_DEPTH       -503
#define PULSEQLIB_ERR_MAX_GRAD_EXCEEDED          -550
#define PULSEQLIB_ERR_GRAD_DISCONTINUITY         -551
#define PULSEQLIB_ERR_MAX_SLEW_EXCEEDED          -552

/* Consistancy errors */
#define PULSEQLIB_ERR_CONSISTENCY_SEG_MISMATCH   -560
#define PULSEQLIB_ERR_CONSISTENCY_RF_PERIODIC    -561

#define PULSEQLIB_ERR_NOT_IMPLEMENTED      -999

/* Code checking */
#define PULSEQLIB_SUCCEEDED(code) ((code) > 0)
#define PULSEQLIB_FAILED(code)    ((code) < 0)

/* ================================================================== */
/*  Cursor info                                                       */
/* ================================================================== */
#define PULSEQLIB_CURSOR_BLOCK  0
#define PULSEQLIB_CURSOR_DONE   1

/* ================================================================== */
/*  Max-size constants (public)                                       */
/* ================================================================== */
#define PULSEQLIB_MAX_GRAD_SHOTS 16
/* Legacy alias */
#ifndef MAX_GRAD_SHOTS
#define MAX_GRAD_SHOTS PULSEQLIB_MAX_GRAD_SHOTS
#endif

#define PULSEQLIB_MAX_RF_SHIM_CHANNELS 64

/* ================================================================== */
/*  Diagnostic                                                        */
/* ================================================================== */
typedef struct pulseqlib_diagnostic {
    int code;
    int block_index;
    int channel;
    int num_unique_blocks;
    int imaging_region_length;
    int candidate_pattern_length;
    int mismatch_position;
    float gradient_amplitude;
    float max_allowed_amplitude;
} pulseqlib_diagnostic;

#define PULSEQLIB_DIAGNOSTIC_INIT { \
    PULSEQLIB_OK, -1, -1, 0, 0, 0, -1, 0.0f, 0.0f \
}

/* ================================================================== */
/*  System options                                                    */
/* ================================================================== */
typedef struct pulseqlib_opts {
    float gamma;
    float b0;
    float max_grad;
    float max_slew;
    float rf_raster_time;
    float grad_raster_time;
    float adc_raster_time;
    float block_duration_raster;
} pulseqlib_opts;

#define PULSEQLIB_OPTS_INIT {0}

/* ================================================================== */
/*  RF stats (public, vendor-independent view of RF definition stats) */
/* ================================================================== */
typedef struct pulseqlib_rf_stats {
    float flip_angle;
    float area;
    float abswidth;
    float effwidth;
    float dtycyc;
    float maxpw;
    float duration_us;
    int   isodelay_us;
    float bandwidth;
    float max_amplitude;
    int   num_samples;
} pulseqlib_rf_stats;

#define PULSEQLIB_RF_STATS_INIT {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0, 0.0f, 0.0f, 0}

/* ================================================================== */
/*  RF definitions and table                                          */
/* ================================================================== */
/* (Internal -- see pulseqlib_internal.h)                             */

/* TR region selectors for freq_mod plan building */
#define PULSEQLIB_TR_REGION_ALL      (-1)
#define PULSEQLIB_TR_REGION_PREP       0
#define PULSEQLIB_TR_REGION_MAIN       1
#define PULSEQLIB_TR_REGION_COOLDOWN   2

/* Computed frequency modulation plan for a set of RF/ADC instances */
typedef struct pulseqlib_freq_mod_plan {
    int num_instances;      /* number of freq-mod events in plan */
    int max_samples;        /* longest waveform (zero-padded length) */
    int num_blocks;         /* total blocks in descriptor (block_table size) */
    float raster_us;        /* common raster (us) */
    float** waveforms;      /* [num_instances] pointers, each -> max_samples Hz */
    int* num_samples;       /* [num_instances] actual length per row */
    float* phase_offset;    /* [num_instances] phase compensation in rad */
    int* block_to_instance; /* [num_blocks] absolute block idx -> instance, -1 */
    float* _waveform_data;  /* backing store (flat), freed by plan_free */
    const void* _desc;      /* opaque pointer to descriptor (for update) */
} pulseqlib_freq_mod_plan;

#define PULSEQLIB_FREQ_MOD_PLAN_INIT {0, 0, 0, 0.0f, NULL, NULL, NULL, NULL, NULL, NULL}

/* ================================================================== */
/*  Segment timing anchors                                            */
/* ================================================================== */

/* RF event anchor relative to segment start (us) */
typedef struct pulseqlib_segment_rf_anchor {
    int   block_offset;         /* block index within segment */
    float start_us;             /* RF delay relative to segment start */
    float end_us;               /* RF end relative to segment start */
    float isocenter_us;         /* RF isodelay point relative to segment start */
    float base_amplitude;       /* amplitude of first TR appearance */
} pulseqlib_segment_rf_anchor;

#define PULSEQLIB_SEGMENT_RF_ANCHOR_INIT {0, 0.0f, 0.0f, 0.0f, 0.0f}

/* ADC event anchor relative to segment start (us) */
typedef struct pulseqlib_segment_adc_anchor {
    int   block_offset;         /* block index within segment */
    float start_us;             /* ADC delay relative to segment start */
    float end_us;               /* ADC end relative to segment start */
    int   kzero_index;          /* sample index of k=0 in readout */
    float kzero_us;             /* time of k=0 relative to segment start */
} pulseqlib_segment_adc_anchor;

#define PULSEQLIB_SEGMENT_ADC_ANCHOR_INIT {0, 0.0f, 0.0f, 0, 0.0f}

/* ================================================================== */
/*  Opaque collection handle                                          */
/* ================================================================== */
typedef struct pulseqlib_collection pulseqlib_collection;

/* ================================================================== */
/*  TR gradient waveforms                                             */
/* ================================================================== */
typedef struct pulseqlib_tr_gradient_waveforms {
    int num_samples;
    float* time;
    float* waveform_gx;
    float* waveform_gy;
    float* waveform_gz;
} pulseqlib_tr_gradient_waveforms;

#define PULSEQLIB_TR_GRADIENT_WAVEFORMS_INIT {0, NULL, NULL, NULL, NULL}

/* ================================================================== */
/*  Acoustic violations                                               */
/* ================================================================== */
typedef struct pulseqlib_acoustic_violation {
    int detected;
    int band_index;
    float peak_frequency_hz;
    float max_amplitude;
    float allowed_amplitude;
} pulseqlib_acoustic_violation;

#define PULSEQLIB_ACOUSTIC_VIOLATION_INIT {0, -1, 0.0f, 0.0f, 0.0f}

typedef struct pulseqlib_acoustic_check_result {
    pulseqlib_acoustic_violation gx;
    pulseqlib_acoustic_violation gy;
    pulseqlib_acoustic_violation gz;
} pulseqlib_acoustic_check_result;

#define PULSEQLIB_ACOUSTIC_CHECK_RESULT_INIT { \
    PULSEQLIB_ACOUSTIC_VIOLATION_INIT, \
    PULSEQLIB_ACOUSTIC_VIOLATION_INIT, \
    PULSEQLIB_ACOUSTIC_VIOLATION_INIT \
}

/* ================================================================== */
/*  TR acoustic spectra                                               */
/* ================================================================== */
typedef struct pulseqlib_tr_acoustic_spectra {
    int num_windows;
    int num_freq_bins;
    int combined;
    float freq_resolution;
    float* frequencies;
    float* spectra_gx;
    float* spectra_gy;
    float* spectra_gz;
    float* max_envelope_gx;
    float* max_envelope_gy;
    float* max_envelope_gz;
    int* peaks_gx;
    int* peaks_gy;
    int* peaks_gz;

    int num_freq_bins_full;
    float freq_resolution_full;
    float* frequencies_full;
    float* spectra_gx_full;
    float* spectra_gy_full;
    float* spectra_gz_full;
    float max_envelope_gx_full;
    float max_envelope_gy_full;
    float max_envelope_gz_full;

    int num_trs;
    float tr_duration_us;
    float fundamental_freq;
    int num_freq_bins_seq;
    float* frequencies_seq;
    float* spectra_gx_seq;
    float* spectra_gy_seq;
    float* spectra_gz_seq;
    int* peaks_gx_seq;
    int* peaks_gy_seq;
    int* peaks_gz_seq;

    pulseqlib_acoustic_check_result sliding_window_check;
    pulseqlib_acoustic_check_result sequence_check;
} pulseqlib_tr_acoustic_spectra;

#define PULSEQLIB_TR_ACOUSTIC_SPECTRA_INIT { \
    0, 0, 0, 0.0f, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, \
    0, 0.0f, NULL, NULL, NULL, NULL, 0.0f, 0.0f, 0.0f, \
    0, 0.0f, 0.0f, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, \
    PULSEQLIB_ACOUSTIC_CHECK_RESULT_INIT, PULSEQLIB_ACOUSTIC_CHECK_RESULT_INIT \
}

/* ================================================================== */
/*  Forbidden band                                                    */
/* ================================================================== */
typedef struct pulseqlib_forbidden_band {
    float freq_min_hz;
    float freq_max_hz;
    float max_amplitude;
} pulseqlib_forbidden_band;

#define PULSEQLIB_FORBIDDEN_BAND_INIT {0.0f, 0.0f, 0.0f}

/* ================================================================== */
/*  PNS parameters and result (vendor-specific)                       */
/* ================================================================== */
#if PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_SIEMENS
typedef struct pulseqlib_safe_params {
    float a1;
    float tau1;
    float a2;
    float tau2;
    float a3;
    float tau3;
    float g_scale;
    float stim_thresh;
    float stim_limit;
} pulseqlib_safe_params;

#define PULSEQLIB_SAFE_PARAMS_INIT {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}
#endif

typedef struct pulseqlib_pns_params {
#if PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_GEHC
    float chronaxie_us;
    float rheobase;
    float alpha;
#elif PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_SIEMENS
    pulseqlib_safe_params x;
    pulseqlib_safe_params y;
    pulseqlib_safe_params z;
#endif
} pulseqlib_pns_params;

#if PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_GEHC
#define PULSEQLIB_PNS_PARAMS_INIT {0.0f, 0.0f, 1.0f}
#elif PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_SIEMENS
#define PULSEQLIB_PNS_PARAMS_INIT {PULSEQLIB_SAFE_PARAMS_INIT, PULSEQLIB_SAFE_PARAMS_INIT, PULSEQLIB_SAFE_PARAMS_INIT}
#endif

typedef struct pulseqlib_pns_result {
    int num_samples;
    float* pns_x;
    float* pns_y;
    float* pns_z;
    float* pns_total;
    float max_pns;
    int max_pns_index;
    float max_pns_time_us;
} pulseqlib_pns_result;

#define PULSEQLIB_PNS_RESULT_INIT {0, NULL, NULL, NULL, NULL, 0.0f, 0, 0.0f}

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

typedef struct pulseqlib_block_instance {
    /* Block duration (resolved: pure delay uses instance, normal uses definition) */
    int duration_us;

    /* RF */
    float rf_amp;
    float rf_freq;
    float rf_phase;

    /* Gradients — amplitudes for this TR instance */
    float gx_amp;
    float gy_amp;
    float gz_amp;

    /* Gradient shot indices for this TR instance */
    int gx_shot_idx;
    int gy_shot_idx;
    int gz_shot_idx;

    /* Rotation */
    float rotmat[9];
    int norot_flag;
    int nopos_flag;

    /* Trigger */
    int trigon_flag;

    /* ADC */
    int adc_flag;        /* 0 = no ADC, 1 = has ADC */
    float adc_freq;
    float adc_phase;
} pulseqlib_block_instance;

#define PULSEQLIB_BLOCK_INSTANCE_INIT { \
    0, \
    0.0f, 0.0f, 0.0f, \
    0.0f, 0.0f, 0.0f, \
    0, 0, 0, \
    {1,0,0, 0,1,0, 0,0,1}, 0, 0, \
    0, \
    0, 0.0f, 0.0f \
}

/* ================================================================== */
/*  Lightweight scan-time query result                                */
/* ================================================================== */
typedef struct pulseqlib_scan_time_info {
    int num_subsequences;
    float* durations_us;       /* per-subsequence duration in us  */
    int*   ignore_averages;    /* per-subsequence flag (0 or 1)   */
} pulseqlib_scan_time_info;

#define PULSEQLIB_SCAN_TIME_INFO_INIT {0, NULL, NULL}

#endif /* PULSEQLIB_TYPES_H */
