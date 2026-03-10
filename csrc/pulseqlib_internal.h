/* pulseqlib_internal.h -- internal types and shared helpers
 *
 * This header is included by implementation (.c) files only.
 * It is NOT part of the public API.
 */

#ifndef PULSEQLIB_INTERNAL_H
#define PULSEQLIB_INTERNAL_H

#include <math.h>
#include <stdio.h>

#include "pulseqlib_config.h"
#include "pulseqlib_types.h"

/* ================================================================== */
/*  Internal error codes                                              */
/*  NOT part of the public API.  Consumers must use                   */
/*  PULSEQLIB_FAILED() / PULSEQLIB_SUCCEEDED() and the diagnostic    */
/*  message string rather than matching on specific values.           */
/* ================================================================== */

/* Generic errors (-1 to -9) */
#define PULSEQLIB_ERR_NULL_POINTER           -1
#define PULSEQLIB_ERR_INVALID_ARGUMENT       -2
#define PULSEQLIB_ERR_ALLOC_FAILED           -3

/* Parsing / file errors (-10 to -19) */
#define PULSEQLIB_ERR_FILE_NOT_FOUND        -10
#define PULSEQLIB_ERR_FILE_READ_FAILED      -11
#define PULSEQLIB_ERR_UNSUPPORTED_VERSION   -12

/* Unique-block errors (-50 to -59) */
#define PULSEQLIB_ERR_INVALID_PREP_POSITION      -50
#define PULSEQLIB_ERR_INVALID_COOLDOWN_POSITION  -51
#define PULSEQLIB_ERR_INVALID_ONCE_FLAGS         -52
#define PULSEQLIB_ERR_RASTER_MISMATCH            -53
#define PULSEQLIB_ERR_SIGNATURE_MISMATCH         -54
#define PULSEQLIB_ERR_SIGNATURE_MISSING          -55
#define PULSEQLIB_ERR_ADC_DEFINITION_CONFLICT    -56
#define PULSEQLIB_ERR_INDEX                      -57

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
#define PULSEQLIB_ERR_SEG_MULTIPLE_PHYSIO_TRIGGERS -204
#define PULSEQLIB_ERR_SEG_MULTIPLE_NAV_SEGMENTS    -205

/* Acoustic errors (-400 to -449) */
#define PULSEQLIB_ERR_ACOUSTIC_NO_WAVEFORM       -402
#define PULSEQLIB_ERR_ACOUSTIC_VIOLATION         -404

/* PNS errors (-450 to -499) */
#define PULSEQLIB_ERR_PNS_INVALID_PARAMS         -450
#define PULSEQLIB_ERR_PNS_INVALID_CHRONAXIE      -451
#define PULSEQLIB_ERR_PNS_INVALID_RHEOBASE       -452
#define PULSEQLIB_ERR_PNS_NO_WAVEFORM            -453
#define PULSEQLIB_ERR_PNS_FFT_FAILED             -454
#define PULSEQLIB_ERR_PNS_THRESHOLD_EXCEEDED     -455

/* Collection / safety errors (-500 to -559) */
#define PULSEQLIB_ERR_COLLECTION_EMPTY           -500
#define PULSEQLIB_ERR_COLLECTION_CHAIN_BROKEN    -501
#define PULSEQLIB_ERR_MAX_GRAD_EXCEEDED          -550
#define PULSEQLIB_ERR_GRAD_DISCONTINUITY         -551
#define PULSEQLIB_ERR_MAX_SLEW_EXCEEDED          -552

/* Consistency errors (-560 to -569) */
#define PULSEQLIB_ERR_CONSISTENCY_SEG_MISMATCH   -560
#define PULSEQLIB_ERR_CONSISTENCY_RF_PERIODIC    -561
#define PULSEQLIB_ERR_CONSISTENCY_RF_SHIM_PERIODIC -562

/* Sentinel */
#define PULSEQLIB_ERR_NOT_IMPLEMENTED      -999


/* ================================================================== */
/*  Constants moved from public header (implementation details)       */
/* ================================================================== */
#define PULSEQLIB_RF_USE_UNKNOWN     0
#define PULSEQLIB_RF_USE_EXCITATION  1
#define PULSEQLIB_RF_USE_REFOCUSING  2
#define PULSEQLIB_RF_USE_INVERSION   3
#define PULSEQLIB_RF_USE_SATURATION  4

#define PULSEQLIB_MAX_RF_SHIM_CHANNELS 64

#define PULSEQLIB_TR_REGION_ALL      (-1)

/* ================================================================== */
/*  Segment timing anchors (internal)                                 */
/* ================================================================== */
typedef struct pulseqlib_segment_rf_anchor {
    int   block_offset;        /* block index within segment         */
    float start_us;            /* RF start time within segment (us)  */
    float end_us;              /* RF end time within segment (us)    */
    float isocenter_us;        /* isodelay time within segment (us)  */
    float base_amplitude_hz;   /* base RF amplitude (Hz)             */
    int   rf_use;              /* PULSEQLIB_RF_USE_*                 */
} pulseqlib_segment_rf_anchor;

#define PULSEQLIB_SEGMENT_RF_ANCHOR_INIT {0, 0.0f, 0.0f, 0.0f, 0.0f, 0}

typedef struct pulseqlib_segment_adc_anchor {
    int   block_offset;        /* block index within segment         */
    float start_us;            /* ADC start time within segment (us) */
    float end_us;              /* ADC end time within segment (us)   */
    int   kzero_index;         /* k=0 sample index within readout    */
    float kzero_us;            /* k=0 time within segment (us)       */
} pulseqlib_segment_adc_anchor;

#define PULSEQLIB_SEGMENT_ADC_ANCHOR_INIT {0, 0.0f, 0.0f, 0, 0.0f}

/* ================================================================== */
/*  Shape (used in descriptor for decompressed waveforms)             */
/* ================================================================== */
typedef struct pulseqlib_shape_arbitrary {
    int num_uncompressed_samples;
    int num_samples;
    float *samples;
} pulseqlib_shape_arbitrary;

#define PULSEQLIB_SHAPE_ARBITRARY_INIT {0, 0, NULL}

/* ================================================================== */
/*  Trigger event (used in descriptor)                                */
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
/*  RF definitions and table                                          */
/* ================================================================== */
typedef struct pulseqlib_rf_definition {
    int id;
    int mag_shape_id;
    int phase_shape_id;
    int time_shape_id;
    int delay;
    int num_channels;     /* 1 for standard, >1 for dynamic pTx */
    pulseqlib_rf_stats stats; /* always present (runtime vendor check) */
} pulseqlib_rf_definition;

#define PULSEQLIB_RF_DEFINITION_INIT {0, 0, 0, 0, 0, 1, PULSEQLIB_RF_STATS_INIT}

typedef struct pulseqlib_rf_table_element {
    int id;
    float amplitude;
    float freq_offset;
    float phase_offset;
    int rf_use;              /* PULSEQLIB_RF_USE_* (0 = unknown) */
} pulseqlib_rf_table_element;

#define PULSEQLIB_RF_TABLE_ELEMENT_INIT {0, 0.0f, 0.0f, 0.0f, 0}

/* ================================================================== */
/*  RF shim definitions (parallel transmit channel weights)           */
/* ================================================================== */
typedef struct pulseqlib_rf_shim_definition {
    int id;
    int num_channels;
    float magnitudes[PULSEQLIB_MAX_RF_SHIM_CHANNELS];
    float phases[PULSEQLIB_MAX_RF_SHIM_CHANNELS];
} pulseqlib_rf_shim_definition;

#define PULSEQLIB_RF_SHIM_DEFINITION_INIT {0, 0, {0}, {0}}

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
    float min_amplitude[PULSEQLIB_MAX_GRAD_SHOTS];
    float slew_rate[PULSEQLIB_MAX_GRAD_SHOTS];
    float energy[PULSEQLIB_MAX_GRAD_SHOTS];
    float first_value[PULSEQLIB_MAX_GRAD_SHOTS];
    float last_value[PULSEQLIB_MAX_GRAD_SHOTS];
} pulseqlib_grad_definition;

#define PULSEQLIB_GRAD_DEFINITION_INIT {0, 0, 0, 0, 0, 0, 0, 1, {0}, {0.0f}, {0.0f}, {0.0f}, {0.0f}, {0.0f}, {0.0f}}

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
/*  Frequency modulation definitions                                  */
/* ================================================================== */
typedef struct pulseqlib_freq_mod_definition {
    int id;
    int num_samples;          /* samples per axis (uniform raster) */
    float raster_us;          /* sample spacing in us */
    float duration_us;        /* active region duration */
    float* waveform_gx;       /* [num_samples] peak-normalized gradient, x */
    float* waveform_gy;       /* [num_samples] peak-normalized gradient, y */
    float* waveform_gz;       /* [num_samples] peak-normalized gradient, z */
    float ref_integral[3];    /* integral from start to reference point
                               * (gx, gy, gz) in [rad/Hz], pre-multiplied
                               * by 2*pi so that phase = ref_integral * freq */
    float ref_time_us;        /* reference time relative to active region
                               * start (isodelay for RF, kzero for ADC) */
} pulseqlib_freq_mod_definition;

#define PULSEQLIB_FREQ_MOD_DEFINITION_INIT \
    {0, 0, 0.0f, 0.0f, NULL, NULL, NULL, {0.0f, 0.0f, 0.0f}, 0.0f}

/* ================================================================== */
/*  Frequency modulation library (internal per-subsequence struct)     */
/* ================================================================== */

/*
 * Per-subsequence library of precomputed frequency modulators.
 *
 * Contains amplitude-scaled 3-channel gradient waveforms (entries) and
 * shift-resolved 1D plan waveforms.  Built internally by
 * pulseqlib_build_freq_mod_collection(); queried via
 * pulseqlib_freq_mod_collection_get() using subsequence index and
 * scan-table position.
 *
 * For PMC-enabled subsequences the 3-channel entries are kept so that
 * update() can recompute plan waveforms with a new shift.  For
 * non-PMC subsequences they are freed after the initial plan build.
 */
typedef struct pulseqlib_freq_mod_library {
    /* --- Deduped 3-channel entries (shift-independent) --- */
    int  num_entries;           /* unique (base_shape, eff_amp) combos     */
    int  max_samples;           /* longest entry waveform (zero-padded)    */
    float raster_us;            /* common time raster (us)                 */
    int* entry_num_samples;     /* [num_entries]                           */

    /* Planar layout: 3ch[e * max_samples * 3 + ch * max_samples + s].
     * NULL after construction for non-PMC subsequences.               */
    float* entry_waveform_3ch;  /* [num_entries * max_samples * 3] or NULL */
    float* entry_ref_3ch;       /* [num_entries * 3]               or NULL */

    /* Deep-copy rotation matrices from descriptor. */
    int   num_rotations;
    float (*rotations)[9];      /* [num_rotations][9]                      */

    /* --- Plan instances (deduped on entry_idx x rotation_idx) --- */
    int  num_plan_instances;
    int* pi_entry_idx;          /* [num_plan_instances]                    */
    int* pi_rotation_idx;       /* [num_plan_instances]                    */

    /* Precomputed 1D waveforms (shift-dependent). */
    float* plan_waveform_data;  /* flat [num_plan_instances * max_samples] */
    float** plan_waveforms;     /* [num_plan_instances] row pointers       */
    int* plan_num_samples;      /* [num_plan_instances] actual length      */
    float* plan_phase;          /* [num_plan_instances] phase comp (rad)   */

    /* O(1) accessor by scan-table position. */
    int  scan_table_len;
    int* scan_to_plan;          /* [scan_table_len] -> plan instance, -1   */
} pulseqlib_freq_mod_library;

/* ================================================================== */
/*  Frequency modulation collection (opaque from public API)          */
/* ================================================================== */

/*
 * Wraps per-subsequence freq-mod libraries into a single object.
 * The public opaque type pulseqlib_freq_mod_collection points here.
 */
struct pulseqlib_freq_mod_collection {
    int num_subsequences;
    pulseqlib_freq_mod_library** libs;   /* [num_subsequences] (owned) */
};

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
    int adc_id;    /* unique ADC definition index, -1 = no ADC */
} pulseqlib_block_definition;

#define PULSEQLIB_BLOCK_DEFINITION_INIT {0, 0, 0, 0, 0, 0, -1}

typedef struct pulseqlib_block_table_element {
    int id;
    int duration_us;
    int rf_id;
    int gx_id;
    int gy_id;
    int gz_id;
    int adc_id;
    int digitalout_id;
    int rotation_id;
    int once_flag;
    int norot_flag;
    int nopos_flag;
    int pmc_flag;
    int nav_flag;
    int freq_mod_id;    /* boolean: >= 0 if block needs freq-mod, -1 otherwise */
    int rf_shim_id;     /* index into rf_shim_definitions, or -1 */
} pulseqlib_block_table_element;

#define PULSEQLIB_BLOCK_TABLE_ELEMENT_INIT {0, 0, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1}
/* NOTE: digitalout_id occupies the former trigger_id position */

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
    int imaging_tr_start;
    float tr_duration_us;
} pulseqlib_tr_descriptor;

#define PULSEQLIB_TR_DESCRIPTOR_INIT {0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0f}

/* Per-segment timing summary */
typedef struct pulseqlib_segment_timing {
    int num_rf_anchors;
    pulseqlib_segment_rf_anchor* rf_anchors;
    int num_adc_anchors;
    pulseqlib_segment_adc_anchor* adc_anchors;
    int num_kzero_crossings;
    int* kzero_crossing_indices;
} pulseqlib_segment_timing;

#define PULSEQLIB_SEGMENT_TIMING_INIT {0, NULL, 0, NULL, 0, NULL}

/* ================================================================== */
/*  TR segment                                                        */
/* ================================================================== */
typedef struct pulseqlib_tr_segment {
    int start_block;
    int num_blocks;
    int* unique_block_indices;
    int* has_digitalout;
    int* has_rotation;
    int* norot_flag;
    int* nopos_flag;
    int* has_freq_mod;
    int max_energy_start_block;
    int trigger_id;             /* segment-level physio trigger (INPUT type),
                                   index into trigger_events[], or -1          */
    int is_nav;                 /* 1 if all blocks in segment are NAV          */
    pulseqlib_segment_timing timing;
} pulseqlib_tr_segment;

#define PULSEQLIB_TR_SEGMENT_INIT {0, 0, NULL, NULL, NULL, NULL, NULL, NULL, 0, -1, 0, PULSEQLIB_SEGMENT_TIMING_INIT}

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
    float rf_raster_us;
    float grad_raster_us;
    float adc_raster_us;
    float block_raster_us;
    int ignore_fov_shift;
    int enable_pmc;
    int ignore_averages;
    int num_passes;
    int pass_len;           /**< blocks per pass (= num_blocks when single-pass) */
    int vendor;             /**< PULSEQLIB_VENDOR_* runtime constant */

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

    int num_freq_mod_defs;
    pulseqlib_freq_mod_definition* freq_mod_definitions;

    int num_rf_shims;
    pulseqlib_rf_shim_definition* rf_shim_definitions;

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

    /* Scan table (expanded playback order).
     * Each row has 3 columns: block_table_idx, tr_id, seg_id.
     * Stored as 3 parallel arrays of length scan_table_len. */
    int  scan_table_len;
    int* scan_table_block_idx;  /* [scan_table_len] index into block_table */
    int* scan_table_tr_id;      /* [scan_table_len] TR region id           */
    int* scan_table_seg_id;     /* [scan_table_len] segment id             */
    int* scan_table_tr_start;   /* [scan_table_len] 1 at first block of each main-region TR */

    /* Per-position variable-gradient flags  [tr_size * 3].
     * Layout: flags[pos * 3 + axis] where axis 0=gx, 1=gy, 2=gz.
     * Value 1 means the gradient amplitude varies across TR instances
     * at that (position, axis); 0 means constant (or absent).  Used by
     * ZERO_VAR amplitude mode to zero out only the variable axes. */
    int* variable_grad_flags;

    /* label table (populated by dry-run if parse_labels is set) */
    int label_num_columns;
    int label_num_entries;
    int* label_table;
    pulseqlib_label_limits label_limits;
} pulseqlib_sequence_descriptor;

#define PULSEQLIB_SEQUENCE_DESCRIPTOR_INIT { \
    0, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0, 0, 0, 1, 0, 0, \
    0, NULL, 0, NULL, \
    0, NULL, 0, NULL, \
    0, NULL, 0, NULL, \
    0, NULL, 0, NULL, \
    0, NULL, 0, NULL, \
    0, NULL, 0, NULL, 0, NULL, \
    PULSEQLIB_TR_DESCRIPTOR_INIT, \
    0, NULL, PULSEQLIB_SEGMENT_TABLE_RESULT_INIT, \
    0, NULL, NULL, NULL, NULL, \
    NULL, \
    0, 0, NULL, {{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0}} \
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
    int sequence_index;
    int scan_table_position;   /* -1 = before first block */
    int from_last_reset;
} pulseqlib_block_cursor;

#define PULSEQLIB_BLOCK_CURSOR_INIT {0, -1, 0}

/* ================================================================== */
/*  Sequence descriptor collection                                    */
/* ================================================================== */
struct pulseqlib_collection {
    int num_subsequences;
    int num_repetitions;
    pulseqlib_block_cursor block_cursor;
    pulseqlib_sequence_descriptor* descriptors;
    pulseqlib_subsequence_info* subsequence_info;
    int total_unique_segments;
    int total_unique_adcs;
    int total_blocks;
    float total_duration_us;
};

#define PULSEQLIB_COLLECTION_INIT {0, 1, PULSEQLIB_BLOCK_CURSOR_INIT, NULL, NULL, 0, 0, 0, 0.0f}

/* ================================================================== */
/*  Uniform-raster gradient waveforms (internal, post-interpolation)  */
/* ================================================================== */

/*
 * After extracting per-axis raw gradient tuples, the safety / acoustic
 * / PNS code interpolates them onto a common uniform raster.  This
 * struct holds the result.
 */
typedef struct pulseqlib__uniform_grad_waveforms {
    int    num_samples;   /* same for all 3 axes */
    float  raster_us;     /* uniform sample spacing */
    float* gx;            /* [num_samples] amplitude (Hz / m) */
    float* gy;            /* [num_samples] amplitude (Hz / m) */
    float* gz;            /* [num_samples] amplitude (Hz / m) */
} pulseqlib__uniform_grad_waveforms;

#define PULSEQLIB__UNIFORM_GRAD_WAVEFORMS_INIT {0, 0.0f, NULL, NULL, NULL}

/* ================================================================== */
/*  Internal constants                                                */
/* ================================================================== */
#define PULSEQLIB__TWO_PI 6.283185307179586476925286766558
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PULSEQLIB__DEFINITION_NAME_LENGTH 32
#define PULSEQLIB__EXT_NAME_LENGTH        32
#define PULSEQLIB__LABEL_NAME_LENGTH      32
#define PULSEQLIB__SEQUENCE_NAME_LENGTH   256
#define PULSEQLIB__SEQUENCE_FILENAME_LENGTH 256
#define PULSEQLIB__SOFT_DELAY_HINT_LENGTH 32
#define PULSEQLIB__MAX_EXTENSIONS_PER_BLOCK 64
#define PULSEQLIB__MAX_LINE_LENGTH        256
#define PULSEQLIB__MAX_SCALE_SIZE         16
#define PULSEQLIB__MAX_RF_SHIM_CHANNELS   64

/* Gradient types */
#define PULSEQLIB__GRAD_TRAP 1
#define PULSEQLIB__GRAD_ARB  2

/* Extension type IDs */
#define PULSEQLIB__EXT_LIST      0
#define PULSEQLIB__EXT_TRIGGER   1
#define PULSEQLIB__EXT_ROTATION  2
#define PULSEQLIB__EXT_LABELSET  3
#define PULSEQLIB__EXT_LABELINC  4
#define PULSEQLIB__EXT_RF_SHIM   5
#define PULSEQLIB__EXT_DELAY     6
#define PULSEQLIB__EXT_UNKNOWN   7

/* Trigger types */
#define PULSEQLIB__TRIGGER_TYPE_OUTPUT 1
#define PULSEQLIB__TRIGGER_TYPE_INPUT  2

#define PULSEQLIB__TRIGGER_CHANNEL_INPUT_PHYSIO_1  1
#define PULSEQLIB__TRIGGER_CHANNEL_INPUT_PHYSIO_2  2
#define PULSEQLIB__TRIGGER_CHANNEL_OUTPUT_OSC_0    1
#define PULSEQLIB__TRIGGER_CHANNEL_OUTPUT_OSC_1    2
#define PULSEQLIB__TRIGGER_CHANNEL_OUTPUT_EXT_1    3

/* Time hints */
#define PULSEQLIB__HINT_TE      1
#define PULSEQLIB__HINT_TR      2
#define PULSEQLIB__HINT_TI      3
#define PULSEQLIB__HINT_ESP     4
#define PULSEQLIB__HINT_RECTIME 5
#define PULSEQLIB__HINT_T2PREP  6
#define PULSEQLIB__HINT_TE2     7

/* Labels and flags */
#define PULSEQLIB__SLC   1
#define PULSEQLIB__SEG   2
#define PULSEQLIB__REP   3
#define PULSEQLIB__AVG   4
#define PULSEQLIB__SET   5
#define PULSEQLIB__ECO   6
#define PULSEQLIB__PHS   7
#define PULSEQLIB__LIN   8
#define PULSEQLIB__PAR   9
#define PULSEQLIB__ACQ  10
#define PULSEQLIB__NAV  11
#define PULSEQLIB__REV  12
#define PULSEQLIB__SMS  13
#define PULSEQLIB__REF  14
#define PULSEQLIB__IMA  15
#define PULSEQLIB__NOISE 16
#define PULSEQLIB__PMC  17
#define PULSEQLIB__NOROT 18
#define PULSEQLIB__NOPOS 19
#define PULSEQLIB__NOSCL 20
#define PULSEQLIB__ONCE 21
#define PULSEQLIB__TRID 22

/* ================================================================== */
/*  Internal shape types                                              */
/* ================================================================== */
typedef struct pulseqlib__shape_trap {
    long rise_time;
    long flat_time;
    long fall_time;
} pulseqlib__shape_trap;

/* ================================================================== */
/*  Internal event types                                              */
/* ================================================================== */
typedef struct pulseqlib__rf_event {
    short type;
    float amplitude;
    pulseqlib_shape_arbitrary mag_shape;
    pulseqlib_shape_arbitrary phase_shape;
    pulseqlib_shape_arbitrary time_shape;
    float center;
    float freq_ppm;
    float phase_ppm;
    float freq_offset;
    float phase_offset;
    int delay;
    char use;
} pulseqlib__rf_event;

typedef struct pulseqlib__grad_event {
    short type;
    float amplitude;
    int delay;
    pulseqlib__shape_trap trap;
    pulseqlib_shape_arbitrary wave_shape;
    pulseqlib_shape_arbitrary time_shape;
    float first;
    float last;
} pulseqlib__grad_event;

typedef struct pulseqlib__adc_event {
    short type;
    int num_samples;
    int dwell_time;
    int delay;
    float freq_ppm;
    float phase_ppm;
    float freq_offset;
    float phase_offset;
    pulseqlib_shape_arbitrary phase_modulation_shape;
} pulseqlib__adc_event;

typedef struct pulseqlib__rotation_event {
    short type;
    union {
        float rot_quaternion[4];
        float rot_matrix[9];
    } data;
} pulseqlib__rotation_event;

typedef struct pulseqlib__label_event {
    int slc;
    int seg;
    int rep;
    int avg;
    int set;
    int eco;
    int phs;
    int lin;
    int par;
    int acq;
} pulseqlib__label_event;

typedef struct pulseqlib__flag_event {
    int trid;
    int nav;
    int rev;
    int sms;
    int ref;
    int ima;
    int noise;
    int pmc;
    int norot;
    int nopos;
    int noscl;
    int once;
} pulseqlib__flag_event;

typedef struct pulseqlib__soft_delay_event {
    short type;
    int num_id;
    int offset;
    int factor;
    int hint_id;
} pulseqlib__soft_delay_event;

typedef struct pulseqlib__rf_shimming_event {
    short type;
    int num_channels;
    float* amplitudes;
    float* phases;
} pulseqlib__rf_shimming_event;

/* ================================================================== */
/*  Internal block types                                              */
/* ================================================================== */
typedef struct pulseqlib__raw_block {
    int block_duration;
    int rf;
    int gx;
    int gy;
    int gz;
    int adc;
    int ext_count;
    int ext[PULSEQLIB__MAX_EXTENSIONS_PER_BLOCK][2];
} pulseqlib__raw_block;

typedef struct pulseqlib__raw_extension {
    pulseqlib__label_event labelset;
    pulseqlib__label_event labelinc;
    pulseqlib__flag_event flag;
    int rotation_index;
    int rf_shim_index;
    int trigger_index;
    int soft_delay_index;
} pulseqlib__raw_extension;

typedef struct pulseqlib__extension_block {
    pulseqlib__label_event labelset;
    pulseqlib__label_event labelinc;
    pulseqlib__flag_event flag;
    pulseqlib__rotation_event rotation;
    pulseqlib__rf_shimming_event rf_shimming;
    pulseqlib_trigger_event trigger;
    pulseqlib__soft_delay_event soft_delay;
} pulseqlib__extension_block;

typedef struct pulseqlib__seq_block {
    int duration;
    pulseqlib__rf_event rf;
    pulseqlib__grad_event gx;
    pulseqlib__grad_event gy;
    pulseqlib__grad_event gz;
    pulseqlib__adc_event adc;
    pulseqlib_trigger_event trigger;
    pulseqlib__rotation_event rotation;
    pulseqlib__flag_event flag;
    pulseqlib__label_event labelset;
    pulseqlib__label_event labelinc;
    pulseqlib__soft_delay_event delay;
    pulseqlib__rf_shimming_event rf_shimming;
} pulseqlib__seq_block;

/* ================================================================== */
/*  SeqFile structs (opaque from public API)                          */
/* ================================================================== */
typedef struct pulseqlib__section_offsets {
    long scan_cursor;
    long version;
    long definitions;
    long blocks;
    long rf;
    long grad;
    long trap;
    long adc;
    long extensions;
    long triggers;
    long rotations;
    long labelset;
    long labelinc;
    long delays;
    long rfshim;
    long shapes;
    long signature;
} pulseqlib__section_offsets;

typedef struct pulseqlib__definition {
    char name[PULSEQLIB__DEFINITION_NAME_LENGTH];
    int value_size;
    char** value;
} pulseqlib__definition;

typedef struct pulseqlib__reserved_definitions {
    float gradient_raster_time;
    float radiofrequency_raster_time;
    float adc_raster_time;
    float block_duration_raster;
    char name[PULSEQLIB__SEQUENCE_NAME_LENGTH];
    float fov[3];
    float total_duration;
    char next_sequence[PULSEQLIB__SEQUENCE_FILENAME_LENGTH];
    int ignore_fov_shift;
    int enable_pmc;
    int ignore_averages;
} pulseqlib__reserved_definitions;

typedef struct pulseqlib__global_label_table {
    int slc;
    int seg;
    int rep;
    int avg;
    int set;
    int echo;
    int phs;
    int lin;
    int par;
    int acq;
} pulseqlib__global_label_table;

typedef struct pulseqlib__rf_shim_entry {
    int num_channels;
    float values[2 * PULSEQLIB__MAX_RF_SHIM_CHANNELS];
} pulseqlib__rf_shim_entry;

typedef struct pulseqlib__seq_file {
    pulseqlib_opts opts;
    char* file_path;
    pulseqlib__section_offsets offsets;
    int is_version_parsed;
    int version_combined;
    int version_major;
    int version_minor;
    int version_revision;
    int is_definitions_library_parsed;
    int num_definitions;
    pulseqlib__definition* definitions_library;
    pulseqlib__reserved_definitions reserved_definitions_library;
    int is_block_library_parsed;
    int num_blocks;
    float (*block_library)[7];
    int* block_ids;
    int is_rf_library_parsed;
    int rf_library_size;
    float (*rf_library)[10];
    int* rf_use_tags;            /* per-RF-event use tag (parsed from .seq) */
    int is_grad_library_parsed;
    int grad_library_size;
    float (*grad_library)[7];
    int is_adc_library_parsed;
    int adc_library_size;
    float (*adc_library)[8];
    int is_extensions_library_parsed;
    int extensions_library_size;
    float (*extensions_library)[3];
    int trigger_library_size;
    float (*trigger_library)[4];
    int rotation_library_size;
    float (*rotation_quaternion_library)[4];
    float (*rotation_matrix_library)[9];
    int is_label_defined[22];
    int labelset_library_size;
    float (*labelset_library)[2];
    int labelinc_library_size;
    float (*labelinc_library)[2];
    pulseqlib_label_limit label_limits;
    int is_delay_defined[8];
    int soft_delay_library_size;
    float (*soft_delay_library)[4];
    int rf_shim_library_size;
    pulseqlib__rf_shim_entry* rf_shim_library;
    int extension_map[8];
    int extension_lut_size;
    int* extension_lut;
    int is_shapes_library_parsed;
    int shapes_library_size;
    pulseqlib_shape_arbitrary* shapes_library;
} pulseqlib__seq_file;

typedef struct pulseqlib__seq_file_collection {
    int num_sequences;
    pulseqlib__seq_file* sequences;
    char* base_path;
} pulseqlib__seq_file_collection;

/* ================================================================== */
/*  Internal table entry for label/hint lookup                        */
/* ================================================================== */
typedef struct pulseqlib__table_entry {
    const char *name;
    int value;
} pulseqlib__table_entry;

/* ================================================================== */
/*  Internal scale helper for library reading                         */
/* ================================================================== */
typedef struct pulseqlib__scale {
    int size;
    float* values;
} pulseqlib__scale;

/* ================================================================== */
/*  Cross-file internal helper declarations (pulseqlib__ prefix)      */
/* ================================================================== */

/* --- pulseqlib_error.c --- */
int pulseqlib__label2enum(const char *label);
int pulseqlib__hint2enum(const char *hint);
void pulseqlib__diag_printf(pulseqlib_diagnostic* diag, const char* fmt, ...);

/* --- pulseqlib_math.c --- */
float pulseqlib__trapz_real_uniform(const float* s, int n, float dt);
float pulseqlib__trapz_real_nonuniform(const float* s, const float* t, int n);
float pulseqlib__trapz_complex_mag_uniform(const float* re, const float* im, int n, float dt);
float pulseqlib__trapz_complex_mag_nonuniform(const float* re, const float* im, const float* t, int n);
float pulseqlib__max_slew_real_uniform(const float* s, int n, float dt);
float pulseqlib__max_slew_real_nonuniform(const float* s, const float* t, int n);
float pulseqlib__get_max_abs_real(const float* samples, int n);
int   pulseqlib__get_max_abs_index_real(const float* samples, int n);
void  pulseqlib__mag_phase_to_real_imag(float* re, float* im, const float* mag, const float* phase, int n);
void  pulseqlib__quaternion_to_matrix(float* matrix, const float* quat);
void  pulseqlib__apply_rotation(float* out, const float* R, const float* v, int transpose);
void  pulseqlib__interp1_linear(float* out, const float* x, int nx, const float* xp, const float* fp, int nxp);
void  pulseqlib__interp1_linear_complex(float* out_re, float* out_im, const float* x, int nx, const float* xp, const float* fp_re, const float* fp_im, int nxp);
void  pulseqlib__fftshift_complex(float* re, float* im, int n);
float pulseqlib__get_spectrum_flank(const float* x, const float* re, const float* im, int n, float cutoff, int reverse);
size_t pulseqlib__next_pow2(size_t x);
int   pulseqlib__calc_convolution_fft(float* output, const float* signal, int signal_len, const float* kernel, int kernel_len);

/* --- pulseqlib_parse.c --- */
void  pulseqlib__seq_file_init(pulseqlib__seq_file* seq, const pulseqlib_opts* opts);
void  pulseqlib__seq_file_free(pulseqlib__seq_file* seq);
void  pulseqlib__seq_file_collection_free(pulseqlib__seq_file_collection* coll);
void  pulseqlib__seq_block_init(pulseqlib__seq_block* block);
void  pulseqlib__seq_block_free(pulseqlib__seq_block* block);
int   pulseqlib__read_seq(pulseqlib__seq_file* seq, const char* file_path);
int   pulseqlib__read_seq_from_buffer(pulseqlib__seq_file* seq, FILE* f);
int   pulseqlib__read_seq_collection(pulseqlib__seq_file_collection* coll, const char* first_file_path, const pulseqlib_opts* opts);
int   pulseqlib__get_raw_block_content_ids(const pulseqlib__seq_file* seq, pulseqlib__raw_block* block, int block_index, int parse_extensions);
void  pulseqlib__get_raw_extension(const pulseqlib__seq_file* seq, pulseqlib__raw_extension* re, const pulseqlib__raw_block* raw);
void  pulseqlib__get_block(const pulseqlib__seq_file* seq, pulseqlib__seq_block* block, int block_index);
float pulseqlib__get_grad_library_max_amplitude(const pulseqlib__seq_file* seq);
int   pulseqlib__decompress_shape(pulseqlib_shape_arbitrary* result, const pulseqlib_shape_arbitrary* encoded, float scale);
int   pulseqlib__verify_signature(const char* file_path);

/* --- pulseqlib_core.c --- */
int   pulseqlib__deduplicate_int_rows(int* unique_defs, int* event_table, const int* int_rows, int num_rows, int num_cols);
int   pulseqlib__get_unique_blocks(pulseqlib_sequence_descriptor* desc, const pulseqlib__seq_file* seq);

/* --- pulseqlib_structure.c --- */
int   pulseqlib__get_tr_in_sequence(pulseqlib_sequence_descriptor* desc, pulseqlib_diagnostic* diag);
int   pulseqlib__build_scan_table(pulseqlib_sequence_descriptor* desc, int num_averages, pulseqlib_diagnostic* diag);
int   pulseqlib__get_scan_table_segments(pulseqlib_sequence_descriptor* desc, pulseqlib_diagnostic* diag, const pulseqlib_opts* opts);
int   pulseqlib__build_freq_mod_flags(pulseqlib_sequence_descriptor* desc);
void  pulseqlib__compute_scan_table_tr_start(pulseqlib_sequence_descriptor* desc);
int   pulseqlib__build_label_table(pulseqlib_sequence_descriptor* desc, const pulseqlib__seq_file* seq);

/* --- pulseqlib_core.c (continued) --- */
int   pulseqlib__get_collection_descriptors(pulseqlib_collection* desc_coll, pulseqlib_diagnostic* diag, const pulseqlib__seq_file_collection* coll, int parse_labels, int num_averages);
void  pulseqlib_sequence_descriptor_free(pulseqlib_sequence_descriptor* desc);
void  pulseqlib_segment_table_result_free(pulseqlib_segment_table_result* result);

/* --- pulseqlib_safety.c --- */
int   pulseqlib__calc_segment_timing(pulseqlib_sequence_descriptor* desc, pulseqlib_diagnostic* diag);

/* --- pulseqlib_waveforms.c --- */

/* Compute per-position variable-gradient flags for ZERO_VAR mode.
 * Allocates desc->variable_grad_flags (tr_size * 3 ints).
 * Must be called after pulseqlib__get_tr_in_sequence. */
int   pulseqlib__compute_variable_grad_flags(pulseqlib_sequence_descriptor* desc);

/* Free uniform waveforms. */
void  pulseqlib__uniform_grad_waveforms_free(
          pulseqlib__uniform_grad_waveforms* w);

/* Extract gradient waveforms for an arbitrary block range,
 * interpolated to uniform raster (half gradient raster). */
int   pulseqlib__get_gradient_waveforms_range(
          const pulseqlib_sequence_descriptor* desc,
          pulseqlib__uniform_grad_waveforms* out,
          pulseqlib_diagnostic* diag,
          int block_start, int block_count,
          int amplitude_mode,
          const int* tr_group_labels, int target_group);

/* Find unique shot-index TR variants (multi-shot).
 * Returns count of unique groups; caller frees both output arrays. */
int   pulseqlib__find_unique_shot_trs(
          const pulseqlib_sequence_descriptor* desc,
          int** out_unique_tr_indices,
          int** out_tr_group_labels);

/* --- pulseqlib_cache.c --- */
int   pulseqlib__try_read_cache(pulseqlib_collection* coll, const char* seq_path);
int   pulseqlib__write_cache(const pulseqlib_collection* seq_coll, const char* seq_path);

/* --- Helper to locate segment/block in collection --- */
int pulseqlib__resolve_segment(
    const pulseqlib_sequence_descriptor** out_desc,
    int* out_local_seg,
    const pulseqlib_collection* coll,
    int seg_idx);

int pulseqlib__resolve_block(
    const pulseqlib_sequence_descriptor** out_desc,
    const pulseqlib_tr_segment** out_seg,
    int* out_local_blk,
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx);

#endif /* PULSEQLIB_INTERNAL_H */
