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

/* Selective excitation errors (-300 to -399) */
#define PULSEQLIB_ERR_SELEXC_GRAD_SCALING    -300
#define PULSEQLIB_ERR_SELEXC_ROTATION        -301

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
typedef struct pulseqlib_rf_definition {
    int id;
    int mag_shape_id;
    int phase_shape_id;
    int time_shape_id;
    int delay;
#if PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_GEHC
    pulseqlib_rf_stats stats;
#endif
} pulseqlib_rf_definition;

#if PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_GEHC
#define PULSEQLIB_RF_DEFINITION_INIT {0, 0, 0, 0, 0, PULSEQLIB_RF_STATS_INIT}
#else
#define PULSEQLIB_RF_DEFINITION_INIT {0, 0, 0, 0, 0}
#endif

typedef struct pulseqlib_rf_table_element {
    int id;
    float amplitude;
    float freq_offset;
    float phase_offset;
} pulseqlib_rf_table_element;

#define PULSEQLIB_RF_TABLE_ELEMENT_INIT {0, 0.0f, 0.0f, 0.0f}

/* ================================================================== */
/*  Gradient definitions and table                                    */
/* ================================================================== */
typedef struct pulseqlib_grad_definition {
    int id;
    int type;
    int rise_time_or_unused;
    int flat_time_or_unused;
    int fall_time_or_num_uncompressed_samples;
    int unused_or_time_shape_id;
    int delay;
    int num_shots;
    int shot_shape_ids[PULSEQLIB_MAX_GRAD_SHOTS];
    float max_amplitude[PULSEQLIB_MAX_GRAD_SHOTS];
    float slew_rate[PULSEQLIB_MAX_GRAD_SHOTS];
    float energy[PULSEQLIB_MAX_GRAD_SHOTS];
    float first_value[PULSEQLIB_MAX_GRAD_SHOTS];
    float last_value[PULSEQLIB_MAX_GRAD_SHOTS];
} pulseqlib_grad_definition;

#define PULSEQLIB_GRAD_DEFINITION_INIT {0, 0, 0, 0, 0, 0, 0, 1, {0}, {0.0f}, {0.0f}, {0.0f}, {0.0f}}

typedef struct pulseqlib_grad_table_element {
    int id;
    int shot_index;
    float amplitude;
} pulseqlib_grad_table_element;

#define PULSEQLIB_GRAD_TABLE_ELEMENT_INIT {0, 0, 0.0f}

/* ================================================================== */
/*  ADC definitions and table                                         */
/* ================================================================== */
typedef struct pulseqlib_adc_definition {
    int id;
    int num_samples;
    int dwell_time;
    int delay;
} pulseqlib_adc_definition;

#define PULSEQLIB_ADC_DEFINITION_INIT {0, 0, 0, 0}

typedef struct pulseqlib_adc_table_element {
    int id;
    float freq_offset;
    float phase_offset;
} pulseqlib_adc_table_element;

#define PULSEQLIB_ADC_TABLE_ELEMENT_INIT {0, 0.0f, 0.0f}

/* ================================================================== */
/*  Block definitions and table                                       */
/* ================================================================== */
typedef struct pulseqlib_block_definition {
    int id;
    int duration_us;
    int rf_id;
    int gx_id;
    int gy_id;
    int gz_id;
} pulseqlib_block_definition;

#define PULSEQLIB_BLOCK_DEFINITION_INIT {0, 0, 0, 0, 0, 0}

typedef struct pulseqlib_block_table_element {
    int id;
    int duration_us;
    int rf_id;
    int gx_id;
    int gy_id;
    int gz_id;
    int adc_id;
    int trigger_id;
    int rotation_id;
    int once_flag;
    int norot_flag;
    int nopos_flag;
    int pmc_flag;
    int nav_flag;
} pulseqlib_block_table_element;

#define PULSEQLIB_BLOCK_TABLE_ELEMENT_INIT {0, 0, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0}

/* ================================================================== */
/*  Trigger event (public - used in descriptor)                       */
/* ================================================================== */
typedef struct pulseqlib_trigger_event {
    short type;
    long duration;
    long delay;
    int trigger_type;
    int trigger_channel;
} pulseqlib_trigger_event;

#define PULSEQLIB_TRIGGER_EVENT_INIT {0, 0L, 0L, 0, 0}

/* ================================================================== */
/*  Shape (public - used in descriptor for decompressed waveforms)    */
/* ================================================================== */
typedef struct pulseqlib_shape_arbitrary {
    int num_uncompressed_samples;
    int num_samples;
    float *samples;
} pulseqlib_shape_arbitrary;

#define PULSEQLIB_SHAPE_ARBITRARY_INIT {0, 0, NULL}

/* ================================================================== */
/*  TR descriptor                                                     */
/* ================================================================== */
typedef struct pulseqlib_tr_descriptor {
    int num_prep_blocks;
    int num_cooldown_blocks;
    int tr_size;
    int num_trs;
    int num_prep_trs;
    int degenerate_prep;
    int num_cooldown_trs;
    int degenerate_cooldown;
    float tr_duration_us;
} pulseqlib_tr_descriptor;

#define PULSEQLIB_TR_DESCRIPTOR_INIT {0, 0, 0, 0, 0, 0, 0, 0, 0.0f}

/* ================================================================== */
/*  TR segment                                                        */
/* ================================================================== */
typedef struct pulseqlib_tr_segment {
    int start_block;
    int num_blocks;
    int* unique_block_indices;
    int* has_trigger;
    int* has_rotation;
    int* norot_flag;
    int* nopos_flag;
    int max_energy_start_block;
} pulseqlib_tr_segment;

#define PULSEQLIB_TR_SEGMENT_INIT {0, 0, NULL, NULL, NULL, NULL, NULL, 0}

/* ================================================================== */
/*  Segment table result                                              */
/* ================================================================== */
typedef struct pulseqlib_segment_table_result {
    int num_unique_segments;
    int num_prep_segments;
    int* prep_segment_table;
    int num_main_segments;
    int* main_segment_table;
    int num_cooldown_segments;
    int* cooldown_segment_table;
} pulseqlib_segment_table_result;

#define PULSEQLIB_SEGMENT_TABLE_RESULT_INIT {0, 0, NULL, 0, NULL, 0, NULL}

/* ================================================================== */
/*  Sequence descriptor                                               */
/* ================================================================== */
typedef struct pulseqlib_sequence_descriptor {
    int num_prep_blocks;
    int num_cooldown_blocks;
    float rf_raster_time_us;
    float grad_raster_time_us;
    float adc_raster_time_us;
    float block_duration_raster_us;

    int num_unique_blocks;
    pulseqlib_block_definition* block_definitions;
    int num_blocks;
    pulseqlib_block_table_element* block_table;

    int num_unique_rfs;
    pulseqlib_rf_definition* rf_definitions;
    int rf_table_size;
    pulseqlib_rf_table_element* rf_table;

    int num_unique_grads;
    pulseqlib_grad_definition* grad_definitions;
    int grad_table_size;
    pulseqlib_grad_table_element* grad_table;

    int num_unique_adcs;
    pulseqlib_adc_definition* adc_definitions;
    int adc_table_size;
    pulseqlib_adc_table_element* adc_table;

    int num_rotations;
    float (*rotation_matrices)[9];

    int num_triggers;
    pulseqlib_trigger_event* trigger_events;

    int num_shapes;
    pulseqlib_shape_arbitrary* shapes;

    pulseqlib_tr_descriptor tr_descriptor;

    int num_unique_segments;
    pulseqlib_tr_segment* segment_definitions;
    pulseqlib_segment_table_result segment_table;
} pulseqlib_sequence_descriptor;

#define PULSEQLIB_SEQUENCE_DESCRIPTOR_INIT { \
    0, 0, 0.0f, 0.0f, 0.0f, 0.0f, \
    0, NULL, 0, NULL, \
    0, NULL, 0, NULL, \
    0, NULL, 0, NULL, \
    0, NULL, 0, NULL, \
    0, NULL, 0, NULL, 0, NULL, \
    PULSEQLIB_TR_DESCRIPTOR_INIT, \
    0, NULL, PULSEQLIB_SEGMENT_TABLE_RESULT_INIT \
}

/* ================================================================== */
/*  Subsequence info                                                  */
/* ================================================================== */
typedef struct pulseqlib_subsequence_info {
    int sequence_index;
    int adc_id_offset;
    int segment_id_offset;
    int block_index_offset;
} pulseqlib_subsequence_info;

#define PULSEQLIB_SUBSEQUENCE_INFO_INIT {0, 0, 0, 0}

/* ================================================================== */
/*  Block cursor                                                      */
/* ================================================================== */

typedef struct pulseqlib_block_cursor {
    int current_repetition;
    int sequence_index;
    int within_sequence_block_index;
    int from_last_reset;
} pulseqlib_block_cursor;

#define PULSEQLIB_BLOCK_CURSOR_INIT {0, 0, 0, 0}

/* ================================================================== */
/*  Sequence descriptor collection                                    */
/* ================================================================== */
typedef struct pulseqlib_sequence_descriptor_collection {
    int num_subsequences;
    int num_repetitions;
    pulseqlib_block_cursor block_cursor;
    pulseqlib_sequence_descriptor* descriptors;
    pulseqlib_subsequence_info* subsequence_info;
    int total_unique_segments;
    int total_unique_adcs;
    int total_blocks;
    float total_duration_us;
} pulseqlib_sequence_descriptor_collection;

#define PULSEQLIB_SEQUENCE_DESCRIPTOR_COLLECTION_INIT {0, 1, PULSEQLIB_BLOCK_CURSOR_INIT, NULL, NULL, 0, 0, 0, 0.0f}

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

#endif /* PULSEQLIB_TYPES_H */