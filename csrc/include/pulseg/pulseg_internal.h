/* pulseg_internal.h -- internal types and shared helpers
 *
 * This header is included by implementation (.c) files only.
 * It is NOT part of the public API.
 */

#ifndef PULSEG_INTERNAL_H
#define PULSEG_INTERNAL_H

#include <math.h>
#include <stdio.h>

#include "pulseg_config.h"
#include "pulseg_types.h"

/* ================================================================== */
/*  Internal error codes                                              */
/*  NOT part of the public API.  Consumers must use                   */
/*  PULSEG_FAILED() / PULSEG_SUCCEEDED() and the diagnostic    */
/*  message string rather than matching on specific values.           */
/* ================================================================== */

/* Generic errors (-1 to -9) */
#define PULSEG_ERR_NULL_POINTER           -1
#define PULSEG_ERR_INVALID_ARGUMENT       -2
#define PULSEG_ERR_ALLOC_FAILED           -3

/* Parsing / file errors (-10 to -19) */
#define PULSEG_ERR_FILE_NOT_FOUND        -10
#define PULSEG_ERR_FILE_READ_FAILED      -11
#define PULSEG_ERR_UNSUPPORTED_VERSION   -12

/* Unique-block errors (-50 to -59) */
#define PULSEG_ERR_INVALID_PREP_POSITION      -50
#define PULSEG_ERR_INVALID_COOLDOWN_POSITION  -51
#define PULSEG_ERR_INVALID_ONCE_FLAGS         -52
#define PULSEG_ERR_RASTER_MISMATCH            -53
#define PULSEG_ERR_SIGNATURE_MISMATCH         -54
#define PULSEG_ERR_SIGNATURE_MISSING          -55
#define PULSEG_ERR_ADC_DEFINITION_CONFLICT    -56
#define PULSEG_ERR_INDEX                      -57
#define PULSEG_ERR_MISSING_KSPACE_ANCHOR      -58

/* TR detection errors (-100 to -199) */
#define PULSEG_ERR_TR_NO_BLOCKS          -100
#define PULSEG_ERR_TR_NO_IMAGING_REGION  -101
#define PULSEG_ERR_TR_NO_PERIODIC_PATTERN -102
#define PULSEG_ERR_TR_PATTERN_MISMATCH   -103
#define PULSEG_ERR_TR_PREP_TOO_LONG      -104
#define PULSEG_ERR_TR_COOLDOWN_TOO_LONG  -105

/* Segmentation errors (-200 to -299) */
#define PULSEG_ERR_SEG_NONZERO_START_GRAD -200
#define PULSEG_ERR_SEG_NONZERO_END_GRAD   -201
#define PULSEG_ERR_SEG_NO_SEGMENTS_FOUND  -202
#define PULSEG_ERR_TOO_MANY_GRAD_SHOTS    -203
#define PULSEG_ERR_SEG_MULTIPLE_PHYSIO_TRIGGERS -204
#define PULSEG_ERR_SEG_MULTIPLE_NAV_SEGMENTS    -205

/* Mechanical resonance errors (-400 to -449) */
#define PULSEG_ERR_MECH_RESONANCES_NO_WAVEFORM       -402
#define PULSEG_ERR_MECH_RESONANCES_VIOLATION         -404

/* PNS errors (-450 to -499) */
#define PULSEG_ERR_PNS_INVALID_PARAMS         -450
#define PULSEG_ERR_PNS_INVALID_CHRONAXIE      -451
#define PULSEG_ERR_PNS_INVALID_RHEOBASE       -452
#define PULSEG_ERR_PNS_NO_WAVEFORM            -453
#define PULSEG_ERR_PNS_FFT_FAILED             -454
#define PULSEG_ERR_PNS_THRESHOLD_EXCEEDED     -455

/* Collection / safety errors (-500 to -559) */
#define PULSEG_ERR_COLLECTION_EMPTY           -500
#define PULSEG_ERR_COLLECTION_CHAIN_BROKEN    -501
#define PULSEG_ERR_MAX_GRAD_EXCEEDED          -550
#define PULSEG_ERR_GRAD_DISCONTINUITY         -551
#define PULSEG_ERR_MAX_SLEW_EXCEEDED          -552

/* Consistency errors (-560 to -569) */
#define PULSEG_ERR_CONSISTENCY_SEG_MISMATCH   -560
#define PULSEG_ERR_CONSISTENCY_RF_PERIODIC    -561
#define PULSEG_ERR_CONSISTENCY_RF_SHIM_PERIODIC -562

/* Sentinel */
#define PULSEG_ERR_NOT_IMPLEMENTED      -999


/* ================================================================== */
/*  Constants moved from public header (implementation details)       */
/* ================================================================== */
#define PULSEG_RF_USE_UNKNOWN     0
#define PULSEG_RF_USE_EXCITATION  1
#define PULSEG_RF_USE_REFOCUSING  2
#define PULSEG_RF_USE_INVERSION   3
#define PULSEG_RF_USE_SATURATION  4

#define PULSEG_MAX_RF_SHIM_CHANNELS 64

#define PULSEG_TR_REGION_ALL      (-1)

/* Forward declaration for use in sequence_descriptor */
typedef struct pulseg__definition pulseg__definition;

/* ================================================================== */
/*  Segment timing anchors (internal)                                 */
/* ================================================================== */
typedef struct pulseg_segment_rf_anchor {
    int   block_offset;        /* block index within segment         */
    float start_us;            /* RF start time within segment (us)  */
    float end_us;              /* RF end time within segment (us)    */
    float isocenter_us;        /* isodelay time within segment (us)  */
    float base_amplitude_hz;   /* base RF amplitude (Hz)             */
    int   rf_use;              /* PULSEG_RF_USE_*                 */
} pulseg_segment_rf_anchor;

#define PULSEG_SEGMENT_RF_ANCHOR_INIT {0, 0.0f, 0.0f, 0.0f, 0.0f, 0}

typedef struct pulseg_segment_adc_anchor {
    int   block_offset;        /* block index within segment         */
    float start_us;            /* ADC start time within segment (us) */
    float end_us;              /* ADC end time within segment (us)   */
    int   kzero_index;         /* k=0 sample index within readout    */
    float kzero_us;            /* k=0 time within segment (us)       */
} pulseg_segment_adc_anchor;

#define PULSEG_SEGMENT_ADC_ANCHOR_INIT {0, 0.0f, 0.0f, 0, 0.0f}

/* ================================================================== */
/*  Shape (used in descriptor for decompressed waveforms)             */
/* ================================================================== */
typedef struct pulseg_shape_arbitrary {
    int num_uncompressed_samples;
    int num_samples;
    float *samples;
} pulseg_shape_arbitrary;

#define PULSEG_SHAPE_ARBITRARY_INIT {0, 0, NULL}

/* ================================================================== */
/*  Trigger event (used in descriptor)                                */
/* ================================================================== */
typedef struct pulseg_trigger_event {
    short type;
    long duration;
    long delay;
    int trigger_type;
    int trigger_channel;
} pulseg_trigger_event;

#define PULSEG_TRIGGER_EVENT_INIT {0, 0L, 0L, 0, 0}

/* ================================================================== */
/*  RF definitions and table                                          */
/* ================================================================== */
typedef struct pulseg_rf_definition {
    int id;
    int mag_shape_id;
    int phase_shape_id;
    int time_shape_id;
    int delay;
    int num_channels;     /* 1 for standard, >1 for dynamic pTx */
    pulseg_rf_stats stats; /* always present (runtime vendor check) */
} pulseg_rf_definition;

#define PULSEG_RF_DEFINITION_INIT {0, 0, 0, 0, 0, 1, PULSEG_RF_STATS_INIT}

typedef struct pulseg_rf_table_element {
    int id;
    float amplitude;
    float freq_offset;
    float phase_offset;
    int rf_use;              /* PULSEG_RF_USE_* (0 = unknown) */
} pulseg_rf_table_element;

#define PULSEG_RF_TABLE_ELEMENT_INIT {0, 0.0f, 0.0f, 0.0f, 0}

/* ================================================================== */
/*  RF shim definitions (parallel transmit channel weights)           */
/* ================================================================== */
typedef struct pulseg_rf_shim_definition {
    int id;
    int num_channels;
    float magnitudes[PULSEG_MAX_RF_SHIM_CHANNELS];
    float phases[PULSEG_MAX_RF_SHIM_CHANNELS];
} pulseg_rf_shim_definition;

#define PULSEG_RF_SHIM_DEFINITION_INIT {0, 0, {0}, {0}}

/* ================================================================== */
/*  Gradient definitions and table                                    */
/* ================================================================== */
typedef struct pulseg_grad_definition {
    int id;
    int type;
    int rise_time_or_unused;
    int flat_time_or_unused;
    int fall_time_or_num_uncompressed_samples;
    int unused_or_time_shape_id;
    int delay;
    int num_shots;
    int shot_shape_ids[PULSEG_MAX_GRAD_SHOTS];
    float max_amplitude[PULSEG_MAX_GRAD_SHOTS];
    float min_amplitude[PULSEG_MAX_GRAD_SHOTS];
    float slew_rate[PULSEG_MAX_GRAD_SHOTS];
    float energy[PULSEG_MAX_GRAD_SHOTS];
    float first_value[PULSEG_MAX_GRAD_SHOTS];
    float last_value[PULSEG_MAX_GRAD_SHOTS];
} pulseg_grad_definition;

#define PULSEG_GRAD_DEFINITION_INIT {0, 0, 0, 0, 0, 0, 0, 1, {0}, {0.0f}, {0.0f}, {0.0f}, {0.0f}, {0.0f}, {0.0f}}

typedef struct pulseg_grad_table_element {
    int id;
    int shot_index;
    float amplitude;
} pulseg_grad_table_element;

#define PULSEG_GRAD_TABLE_ELEMENT_INIT {0, 0, 0.0f}

/* ================================================================== */
/*  ADC definitions and table                                         */
/* ================================================================== */
typedef struct pulseg_adc_definition {
    int id;
    int num_samples;
    int dwell_time;
    int delay;
} pulseg_adc_definition;

#define PULSEG_ADC_DEFINITION_INIT {0, 0, 0, 0}

typedef struct pulseg_adc_table_element {
    int id;
    float freq_offset;
    float phase_offset;
} pulseg_adc_table_element;

#define PULSEG_ADC_TABLE_ELEMENT_INIT {0, 0.0f, 0.0f}

/* ================================================================== */
/*  Frequency modulation definitions                                  */
/* ================================================================== */
typedef struct pulseg_freq_mod_definition {
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
} pulseg_freq_mod_definition;

#define PULSEG_FREQ_MOD_DEFINITION_INIT \
    {0, 0, 0.0f, 0.0f, NULL, NULL, NULL, {0.0f, 0.0f, 0.0f}, 0.0f}

/* ================================================================== */
/*  Frequency modulation library (internal per-subsequence struct)     */
/* ================================================================== */

/*
 * Per-subsequence library of precomputed frequency modulators.
 *
 * Contains amplitude-scaled 3-channel gradient waveforms (entries) and
 * shift-resolved 1D plan waveforms.  Built internally by
 * pulseg_build_freq_mod_collection(); queried via
 * pulseg_freq_mod_collection_get() using subsequence index and
 * scan-table position.
 *
 * For PMC-enabled subsequences the 3-channel entries are kept so that
 * update() can recompute plan waveforms with a new shift.  For
 * non-PMC subsequences they are freed after the initial plan build.
 */
typedef struct pulseg_freq_mod_library {
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
    float* plan_phase;          /* [num_plan_instances] phase comp (rad)
                                 * from all 3 channels                     */

    /* Hardware-format waveforms: short DAC units with WEOS_BIT on last.
     * Conversion: sample = (int16)(Hz / (4 * TARDIS_FREQ_RES))          */
    short* plan_hw_data;        /* flat [num_plan_instances * max_samples] */
    short** plan_hw_waveforms;  /* [num_plan_instances] row pointers       */

    /* O(1) accessor by scan-table position. */
    int  scan_table_len;
    int* scan_to_plan;          /* [scan_table_len] -> plan instance, -1   */

    /* Optional cache fields retained for backward cache compatibility. */
    float* scan_inactive_area_3ch; /* [scan_table_len * 3]  or NULL */
    float* scan_phase_extra;       /* [scan_table_len]      or NULL */
} pulseg_freq_mod_library;

/* ================================================================== */
/*  Frequency modulation collection (opaque from public API)          */
/* ================================================================== */

/*
 * Wraps per-subsequence freq-mod libraries into a single object.
 * The public opaque type pulseg_freq_mod_collection points here.
 */
struct pulseg_freq_mod_collection {
    int num_subsequences;
    pulseg_freq_mod_library** libs;   /* [num_subsequences] (owned) */
};

/* Free a freq-mod collection (used by pulseg_collection_free). */
void pulseg_freq_mod_collection_free(struct pulseg_freq_mod_collection* fmc);

/* ================================================================== */
/*  Block definitions and table                                       */
/* ================================================================== */
typedef struct pulseg_block_definition {
    int id;
    int duration_us;
    int rf_id;
    int gx_id;
    int gy_id;
    int gz_id;
    int adc_id;    /* unique ADC definition index, -1 = no ADC */
} pulseg_block_definition;

#define PULSEG_BLOCK_DEFINITION_INIT {0, 0, 0, 0, 0, 0, -1}

typedef struct pulseg_block_table_element {
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
} pulseg_block_table_element;

#define PULSEG_BLOCK_TABLE_ELEMENT_INIT {0, 0, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1}
/* NOTE: digitalout_id occupies the former trigger_id position */

/* ================================================================== */
/*  TR descriptor                                                     */
/* ================================================================== */
typedef struct pulseg_tr_descriptor {
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
} pulseg_tr_descriptor;

#define PULSEG_TR_DESCRIPTOR_INIT {0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0f}

/* Per-segment timing summary */
typedef struct pulseg_segment_timing {
    int num_rf_anchors;
    pulseg_segment_rf_anchor* rf_anchors;
    int num_adc_anchors;
    pulseg_segment_adc_anchor* adc_anchors;
    int num_kzero_crossings;
    int* kzero_crossing_indices;
} pulseg_segment_timing;

#define PULSEG_SEGMENT_TIMING_INIT {0, NULL, 0, NULL, 0, NULL}

/* ================================================================== */
/*  TR segment                                                        */
/* ================================================================== */
typedef struct pulseg_tr_segment {
    int start_block;
    int num_blocks;
    int* unique_block_indices;
    int* has_digitalout;
    int* has_rotation;
    int* norot_flag;
    int* nopos_flag;
    int* has_freq_mod;
    int* has_adc;          /* OR-reduced: 1 if at least one segment instance has an ADC
                              event at this block position, 0 otherwise          */
    int max_energy_start_block;
    int trigger_id;             /* segment-level physio trigger (INPUT type),
                                   index into trigger_events[], or -1          */
    int is_nav;                 /* 1 if all blocks in segment are NAV          */
    pulseg_segment_timing timing;
} pulseg_tr_segment;

#define PULSEG_TR_SEGMENT_INIT {0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, -1, 0, PULSEG_SEGMENT_TIMING_INIT}

/* ================================================================== */
/*  Segment table result                                              */
/* ================================================================== */
typedef struct pulseg_segment_table_result {
    int num_unique_segments;
    int num_prep_segments;
    int* prep_segment_table;
    int num_main_segments;
    int* main_segment_table;
    int num_cooldown_segments;
    int* cooldown_segment_table;
} pulseg_segment_table_result;

#define PULSEG_SEGMENT_TABLE_RESULT_INIT {0, 0, NULL, 0, NULL, 0, NULL}

/* ================================================================== */
/*  Sequence descriptor                                               */
/* ================================================================== */
typedef struct pulseg_sequence_descriptor {
    int num_prep_blocks;
    int num_cooldown_blocks;
    float rf_raster_us;
    float grad_raster_us;
    float adc_raster_us;
    float block_raster_us;
    int ignore_fov_shift;
    int enable_pmc;
    int ignore_averages;
    int num_gain_cal_readouts;  /**< calibration readouts for APS2 receive gain (pislquant) */
    int num_passes;
    int pass_len;           /**< blocks per pass (= num_blocks when single-pass) */
    int num_averages;       /**< number of averages (1 if ignore_averages)       */
    int vendor;             /**< PULSEG_VENDOR_* runtime constant */
    int label_column_map[3]; /**< copy of pulseg_opts.label_column_map at dedup time (D3) */

    float fov[3];           /**< field of view (mm), from [DEFINITIONS] FOV */
    float matrix[3];        /**< matrix size, from [DEFINITIONS] Matrix    */
    float nav_fov[3];       /**< navigator FOV (mm), from [DEFINITIONS] NavFOV */
    float nav_matrix[3];    /**< navigator matrix, from [DEFINITIONS] NavMatrix */

    int num_unique_blocks;
    pulseg_block_definition* block_definitions;
    int num_blocks;
    pulseg_block_table_element* block_table;

    int num_unique_rfs;
    pulseg_rf_definition* rf_definitions;
    int rf_table_size;
    pulseg_rf_table_element* rf_table;

    int num_unique_grads;
    pulseg_grad_definition* grad_definitions;
    int grad_table_size;
    pulseg_grad_table_element* grad_table;

    int num_unique_adcs;
    pulseg_adc_definition* adc_definitions;
    int adc_table_size;
    pulseg_adc_table_element* adc_table;

    int num_freq_mod_defs;
    pulseg_freq_mod_definition* freq_mod_definitions;

    int num_rf_shims;
    pulseg_rf_shim_definition* rf_shim_definitions;

    int num_rotations;
    float (*rotation_matrices)[9];

    int num_triggers;
    pulseg_trigger_event* trigger_events;

    int num_shapes;
    pulseg_shape_arbitrary* shapes;

    pulseg_tr_descriptor tr_descriptor;

    int num_unique_segments;
    pulseg_tr_segment* segment_definitions;
    pulseg_segment_table_result segment_table;

    /* Scan table (expanded playback order).
     * Each row has 3 columns: block_table_idx, tr_id, seg_id.
     * Stored as 3 parallel arrays of length scan_table_len. */
    int  scan_table_len;
    int* scan_table_block_idx;  /* [scan_table_len] index into block_table */
    int* scan_table_tr_id;      /* [scan_table_len] TR region id           */
    int* scan_table_seg_id;     /* [scan_table_len] segment id             */
    int* scan_table_avg_id;     /* [scan_table_len] average (rep) index 0..num_averages-1 */
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
    pulseg_label_limits label_limits;
    /* Per-ADC OFF flag (parallel to label_table rows; entries == label_num_entries).
     * Pulseq v1.5.1 LABELSET column "OFF". 1 = acquisition should be discarded
     * downstream (LiveSDK), 0 = keep. NULL when no LABELSET OFF is present. */
    int* off_table;

    /* generic [DEFINITIONS] key-value pairs (all keys, not just reserved) */
    int num_definitions;
    pulseg__definition* definitions;

    /* Full-TR canonical (PULSEG_AMP_ZERO_VAR) k-space trajectory,
     * retained by pulseg__calc_segment_timing (pulseg_structure.c)
     * over the whole main TR (num_prep..num_prep+tr_size) with excitation-
     * reset + refocus-negation already applied. TRAJECTORY (section 6)
     * base shots are SLICED from this array (Stage 1.5c) instead of being
     * re-integrated + re-centered per block. NULL/0 if has_canonical_kspace
     * is 0 (e.g. zero-length TR). */
    int has_canonical_kspace;
    int canonical_kspace_num_samples;
    float canonical_kspace_dt_us; /* raster period of kx/ky/kz below (us) */
    float* canonical_kx;
    float* canonical_ky;
    float* canonical_kz;

    /* Copy of pulseg_opts.cache_ext at dedup time (D10). Not part of the
     * cache payload itself (it only names the cache FILE, not its
     * contents) -- deliberately placed after all cache-serialized fields
     * so it needs no swap4_array count bump. */
    char cache_ext[PULSEG_CACHE_EXT_MAX];
} pulseg_sequence_descriptor;

#define PULSEG_SEQUENCE_DESCRIPTOR_INIT { \
    0, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0, 0, 0, 0, 1, 0, 1, 0, \
    {0,1,2}, \
    {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, \
    0, NULL, 0, NULL, \
    0, NULL, 0, NULL, \
    0, NULL, 0, NULL, \
    0, NULL, 0, NULL, \
    0, NULL, 0, NULL, \
    0, NULL, 0, NULL, 0, NULL, \
    PULSEG_TR_DESCRIPTOR_INIT, \
    0, NULL, PULSEG_SEGMENT_TABLE_RESULT_INIT, \
    0, NULL, NULL, NULL, NULL, NULL, \
    NULL, \
    0, 0, NULL, {{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0}}, NULL, \
    0, NULL, \
    0, 0, 0.0f, NULL, NULL, NULL, \
    PULSEG_CACHE_EXT_DEFAULT \
}

/* ================================================================== */
/*  Subsequence info                                                  */
/* ================================================================== */
typedef struct pulseg_subsequence_info {
    int sequence_index;
    int adc_id_offset;
    int segment_id_offset;
    int block_index_offset;
} pulseg_subsequence_info;

#define PULSEG_SUBSEQUENCE_INFO_INIT {0, 0, 0, 0}

/* ================================================================== */
/*  Block cursor                                                      */
/* ================================================================== */

typedef struct pulseg_block_cursor {
    int sequence_index;
    int scan_table_position;   /* -1 = before first block */
    int from_last_reset;
} pulseg_block_cursor;

#define PULSEG_BLOCK_CURSOR_INIT {0, -1, 0}

/* ================================================================== */
/*  Sequence descriptor collection                                    */
/* ================================================================== */
struct pulseg_collection {
    int num_subsequences;
    int num_repetitions;
    pulseg_block_cursor block_cursor;
    pulseg_sequence_descriptor* descriptors;
    pulseg_subsequence_info* subsequence_info;
    int total_unique_segments;
    int total_unique_adcs;
    int total_blocks;
    float total_duration_us;
    struct pulseg_freq_mod_collection* freq_mod;  /* owned, may be NULL */
};

#define PULSEG_COLLECTION_INIT {0, 1, PULSEG_BLOCK_CURSOR_INIT, NULL, NULL, 0, 0, 0, 0.0f, NULL}

/* ================================================================== */
/*  Uniform-raster gradient waveforms (internal, post-interpolation)  */
/* ================================================================== */

/*
 * After extracting per-axis raw gradient tuples, the safety / acoustic
 * / PNS code interpolates them onto a common uniform raster.  This
 * struct holds the result.
 */
typedef struct pulseg__uniform_grad_waveforms {
    int    num_samples;   /* same for all 3 axes */
    float  raster_us;     /* uniform sample spacing */
    float* gx;            /* [num_samples] amplitude (Hz / m) */
    float* gy;            /* [num_samples] amplitude (Hz / m) */
    float* gz;            /* [num_samples] amplitude (Hz / m) */
} pulseg__uniform_grad_waveforms;

#define PULSEG__UNIFORM_GRAD_WAVEFORMS_INIT {0, 0.0f, NULL, NULL, NULL}

/* ================================================================== */
/*  Internal constants                                                */
/* ================================================================== */
#define PULSEG__TWO_PI 6.283185307179586476925286766558
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PULSEG__DEFINITION_NAME_LENGTH 32
#define PULSEG__EXT_NAME_LENGTH        32
#define PULSEG__LABEL_NAME_LENGTH      32
#define PULSEG__SEQUENCE_NAME_LENGTH   256
#define PULSEG__SEQUENCE_FILENAME_LENGTH 256
#define PULSEG__SOFT_DELAY_HINT_LENGTH 32
#define PULSEG__MAX_EXTENSIONS_PER_BLOCK 64
#define PULSEG__MAX_LINE_LENGTH        256
#define PULSEG__MAX_SCALE_SIZE         16
#define PULSEG__MAX_RF_SHIM_CHANNELS   64

/* Gradient types */
#define PULSEG__GRAD_TRAP 1
#define PULSEG__GRAD_ARB  2

/* Extension type IDs */
#define PULSEG__EXT_LIST      0
#define PULSEG__EXT_TRIGGER   1
#define PULSEG__EXT_ROTATION  2
#define PULSEG__EXT_LABELSET  3
#define PULSEG__EXT_LABELINC  4
#define PULSEG__EXT_RF_SHIM   5
#define PULSEG__EXT_DELAY     6
#define PULSEG__EXT_UNKNOWN   7

/* Trigger types */
#define PULSEG__TRIGGER_TYPE_OUTPUT 1
#define PULSEG__TRIGGER_TYPE_INPUT  2

#define PULSEG__TRIGGER_CHANNEL_INPUT_PHYSIO_1  1
#define PULSEG__TRIGGER_CHANNEL_INPUT_PHYSIO_2  2
#define PULSEG__TRIGGER_CHANNEL_OUTPUT_OSC_0    1
#define PULSEG__TRIGGER_CHANNEL_OUTPUT_OSC_1    2
#define PULSEG__TRIGGER_CHANNEL_OUTPUT_EXT_1    3

/* Time hints */
#define PULSEG__HINT_TE      1
#define PULSEG__HINT_TR      2
#define PULSEG__HINT_TI      3
#define PULSEG__HINT_ESP     4
#define PULSEG__HINT_RECTIME 5
#define PULSEG__HINT_T2PREP  6
#define PULSEG__HINT_TE2     7

/* Labels and flags */
#define PULSEG__SLC   1
#define PULSEG__SEG   2
#define PULSEG__REP   3
#define PULSEG__AVG   4
#define PULSEG__SET   5
#define PULSEG__ECO   6
#define PULSEG__PHS   7
#define PULSEG__LIN   8
#define PULSEG__PAR   9
#define PULSEG__ACQ  10
#define PULSEG__NAV  11
#define PULSEG__REV  12
#define PULSEG__SMS  13
#define PULSEG__REF  14
#define PULSEG__IMA  15
#define PULSEG__NOISE 16
#define PULSEG__PMC  17
#define PULSEG__NOROT 18
#define PULSEG__NOPOS 19
#define PULSEG__NOSCL 20
#define PULSEG__ONCE 21
#define PULSEG__TRID 22
#define PULSEG__OFF  23

/* ================================================================== */
/*  Internal shape types                                              */
/* ================================================================== */
typedef struct pulseg__shape_trap {
    long rise_time;
    long flat_time;
    long fall_time;
} pulseg__shape_trap;

/* ================================================================== */
/*  Internal event types                                              */
/* ================================================================== */
typedef struct pulseg__rf_event {
    short type;
    float amplitude;
    pulseg_shape_arbitrary mag_shape;
    pulseg_shape_arbitrary phase_shape;
    pulseg_shape_arbitrary time_shape;
    float center;
    float freq_ppm;
    float phase_ppm;
    float freq_offset;
    float phase_offset;
    int delay;
    char use;
} pulseg__rf_event;

typedef struct pulseg__grad_event {
    short type;
    float amplitude;
    int delay;
    pulseg__shape_trap trap;
    pulseg_shape_arbitrary wave_shape;
    pulseg_shape_arbitrary time_shape;
    float first;
    float last;
} pulseg__grad_event;

typedef struct pulseg__adc_event {
    short type;
    int num_samples;
    int dwell_time;
    int delay;
    float freq_ppm;
    float phase_ppm;
    float freq_offset;
    float phase_offset;
    pulseg_shape_arbitrary phase_modulation_shape;
} pulseg__adc_event;

typedef struct pulseg__rotation_event {
    short type;
    union {
        float rot_quaternion[4];
        float rot_matrix[9];
    } data;
} pulseg__rotation_event;

typedef struct pulseg__label_event {
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
} pulseg__label_event;

typedef struct pulseg__flag_event {
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
} pulseg__flag_event;

typedef struct pulseg__soft_delay_event {
    short type;
    int num_id;
    int offset;
    int factor;
    int hint_id;
} pulseg__soft_delay_event;

typedef struct pulseg__rf_shimming_event {
    short type;
    int num_channels;
    float* amplitudes;
    float* phases;
} pulseg__rf_shimming_event;

/* ================================================================== */
/*  Internal block types                                              */
/* ================================================================== */
typedef struct pulseg__raw_block {
    int block_duration;
    int rf;
    int gx;
    int gy;
    int gz;
    int adc;
    int ext_count;
    int ext[PULSEG__MAX_EXTENSIONS_PER_BLOCK][2];
} pulseg__raw_block;

typedef struct pulseg__raw_extension {
    pulseg__label_event labelset;
    pulseg__label_event labelinc;
    pulseg__flag_event flag;
    int rotation_index;
    int rf_shim_index;
    int trigger_index;
    int soft_delay_index;
} pulseg__raw_extension;

typedef struct pulseg__extension_block {
    pulseg__label_event labelset;
    pulseg__label_event labelinc;
    pulseg__flag_event flag;
    pulseg__rotation_event rotation;
    pulseg__rf_shimming_event rf_shimming;
    pulseg_trigger_event trigger;
    pulseg__soft_delay_event soft_delay;
} pulseg__extension_block;

typedef struct pulseg__seq_block {
    int duration;
    pulseg__rf_event rf;
    pulseg__grad_event gx;
    pulseg__grad_event gy;
    pulseg__grad_event gz;
    pulseg__adc_event adc;
    pulseg_trigger_event trigger;
    pulseg__rotation_event rotation;
    pulseg__flag_event flag;
    pulseg__label_event labelset;
    pulseg__label_event labelinc;
    pulseg__soft_delay_event delay;
    pulseg__rf_shimming_event rf_shimming;
} pulseg__seq_block;

/* ================================================================== */
/*  SeqFile structs (opaque from public API)                          */
/* ================================================================== */
typedef struct pulseg__section_offsets {
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
} pulseg__section_offsets;

struct pulseg__definition {
    char name[PULSEG__DEFINITION_NAME_LENGTH];
    int value_size;
    char** value;
};

typedef struct pulseg__reserved_definitions {
    float gradient_raster_time;
    float radiofrequency_raster_time;
    float adc_raster_time;
    float block_duration_raster;
    char name[PULSEG__SEQUENCE_NAME_LENGTH];
    float fov[3];
    float matrix[3];
    float nav_fov[3];
    float nav_matrix[3];
    float total_duration;
    char next_sequence[PULSEG__SEQUENCE_FILENAME_LENGTH];
    int ignore_fov_shift;
    int enable_pmc;
    int ignore_averages;
    int num_gain_cal_readouts;  /**< calibration readouts for APS2 receive gain (pislquant) */
} pulseg__reserved_definitions;

typedef struct pulseg__global_label_table {
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
} pulseg__global_label_table;

typedef struct pulseg__rf_shim_entry {
    int num_channels;
    float values[2 * PULSEG__MAX_RF_SHIM_CHANNELS];
} pulseg__rf_shim_entry;

typedef struct pulseg__seq_file {
    pulseg_opts opts;
    char* file_path;
    pulseg__section_offsets offsets;
    int is_version_parsed;
    int version_combined;
    int version_major;
    int version_minor;
    int version_revision;
    int is_definitions_library_parsed;
    int num_definitions;
    pulseg__definition* definitions_library;
    pulseg__reserved_definitions reserved_definitions_library;
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
    pulseg_label_limit label_limits;
    int is_delay_defined[8];
    int soft_delay_library_size;
    float (*soft_delay_library)[4];
    int rf_shim_library_size;
    pulseg__rf_shim_entry* rf_shim_library;
    int extension_map[8];
    int extension_lut_size;
    int* extension_lut;
    int is_shapes_library_parsed;
    int shapes_library_size;
    pulseg_shape_arbitrary* shapes_library;
} pulseg__seq_file;

typedef struct pulseg__seq_file_collection {
    int num_sequences;
    pulseg__seq_file* sequences;
    char* base_path;
} pulseg__seq_file_collection;

/* ================================================================== */
/*  Internal table entry for label/hint lookup                        */
/* ================================================================== */
typedef struct pulseg__table_entry {
    const char *name;
    int value;
} pulseg__table_entry;

/* ================================================================== */
/*  Internal scale helper for library reading                         */
/* ================================================================== */
typedef struct pulseg__scale {
    int size;
    float* values;
} pulseg__scale;

/* ================================================================== */
/*  Cross-file internal helper declarations (pulseg__ prefix)      */
/* ================================================================== */

/* --- pulseg_error.c --- */
int pulseg__label2enum(const char *label);
int pulseg__hint2enum(const char *hint);
void pulseg__diag_printf(pulseg_diagnostic* diag, const char* fmt, ...);

/* --- pulseg_math.c --- */
float pulseg__trapz_real_uniform(const float* s, int n, float dt);
float pulseg__trapz_real_nonuniform(const float* s, const float* t, int n);
float pulseg__trapz_complex_mag_uniform(const float* re, const float* im, int n, float dt);
float pulseg__trapz_complex_mag_nonuniform(const float* re, const float* im, const float* t, int n);
float pulseg__max_slew_real_uniform(const float* s, int n, float dt);
float pulseg__max_slew_real_nonuniform(const float* s, const float* t, int n);
float pulseg__get_max_abs_real(const float* samples, int n);
int   pulseg__get_max_abs_index_real(const float* samples, int n);
void  pulseg__mag_phase_to_real_imag(float* re, float* im, const float* mag, const float* phase, int n);
void  pulseg__quaternion_to_matrix(float* matrix, const float* quat);
int   pulseg__is_identity3(const float* matrix);
void  pulseg__apply_rotation(float* out, const float* R, const float* v, int transpose);
void  pulseg__interp1_linear(float* out, const float* x, int nx, const float* xp, const float* fp, int nxp);
void  pulseg__interp1_linear_complex(float* out_re, float* out_im, const float* x, int nx, const float* xp, const float* fp_re, const float* fp_im, int nxp);
void  pulseg__fftshift_complex(float* re, float* im, int n);
float pulseg__get_spectrum_flank(const float* x, const float* re, const float* im, int n, float cutoff, int reverse);
size_t pulseg__next_pow2(size_t x);
int   pulseg__calc_convolution_fft(float* output, const float* signal, int signal_len, const float* kernel, int kernel_len);

/* --- pulseg_parse.c --- */
void  pulseg__seq_file_init(pulseg__seq_file* seq, const pulseg_opts* opts);
void  pulseg__seq_file_free(pulseg__seq_file* seq);
void  pulseg__seq_file_collection_free(pulseg__seq_file_collection* coll);
void  pulseg__seq_block_init(pulseg__seq_block* block);
void  pulseg__seq_block_free(pulseg__seq_block* block);
int   pulseg__read_seq(pulseg__seq_file* seq, const char* file_path);
int   pulseg__read_seq_from_buffer(pulseg__seq_file* seq, FILE* f);
int   pulseg__read_seq_collection(pulseg__seq_file_collection* coll, const char* first_file_path, const pulseg_opts* opts);
int   pulseg__get_raw_block_content_ids(const pulseg__seq_file* seq, pulseg__raw_block* block, int block_index, int parse_extensions);
void  pulseg__get_raw_extension(const pulseg__seq_file* seq, pulseg__raw_extension* re, const pulseg__raw_block* raw);
void  pulseg__get_block(const pulseg__seq_file* seq, pulseg__seq_block* block, int block_index);
float pulseg__get_grad_library_max_amplitude(const pulseg__seq_file* seq);
int   pulseg__decompress_shape(pulseg_shape_arbitrary* result, const pulseg_shape_arbitrary* encoded, float scale);
int   pulseg__verify_signature(const char* file_path);

/* --- pulseg_core.c --- */
int   pulseg__deduplicate_int_rows(int* unique_defs, int* event_table, const int* int_rows, int num_rows, int num_cols);
int   pulseg__get_unique_blocks(pulseg_sequence_descriptor* desc, const pulseg__seq_file* seq);

/* --- pulseg_structure.c --- */
int   pulseg__get_tr_in_sequence(pulseg_sequence_descriptor* desc, pulseg_diagnostic* diag);
int   pulseg__build_scan_table(pulseg_sequence_descriptor* desc, int num_averages, pulseg_diagnostic* diag);
int   pulseg__get_scan_table_segments(pulseg_sequence_descriptor* desc, pulseg_diagnostic* diag, const pulseg_opts* opts);
int   pulseg__build_freq_mod_flags(pulseg_sequence_descriptor* desc);
void  pulseg__compute_scan_table_tr_start(pulseg_sequence_descriptor* desc);
int   pulseg__build_label_table(pulseg_sequence_descriptor* desc, const pulseg__seq_file* seq);
int   pulseg__calc_segment_timing(pulseg_sequence_descriptor* desc, pulseg_diagnostic* diag);

/* --- pulseg_core.c (continued) --- */
int   pulseg__get_collection_descriptors(pulseg_collection* desc_coll, pulseg_diagnostic* diag, const pulseg__seq_file_collection* coll, int parse_labels, int num_averages);
void  pulseg_sequence_descriptor_free(pulseg_sequence_descriptor* desc);
void  pulseg_segment_table_result_free(pulseg_segment_table_result* result);

/* --- pulseg_waveforms.c --- */

/* Compute per-position variable-gradient flags for ZERO_VAR mode.
 * Allocates desc->variable_grad_flags (tr_size * 3 ints).
 * Must be called after pulseg__get_tr_in_sequence. */
int   pulseg__compute_variable_grad_flags(pulseg_sequence_descriptor* desc);

/* Free uniform waveforms. */
void  pulseg__uniform_grad_waveforms_free(
          pulseg__uniform_grad_waveforms* w);

/* Extract gradient waveforms for an arbitrary block range,
 * interpolated to uniform raster (half gradient raster). */
int   pulseg__get_gradient_waveforms_range(
          const pulseg_sequence_descriptor* desc,
          pulseg__uniform_grad_waveforms* out,
          pulseg_diagnostic* diag,
          int block_start, int block_count,
          int amplitude_mode,
          const int* tr_group_labels, int target_group,
          const int* block_order);

/* Find unique shot-index TR variants (multi-shot, degenerate prep/cooldown).
 * Returns count of unique groups; caller frees both output arrays. */
int   pulseg__find_unique_shot_trs(
          const pulseg_sequence_descriptor* desc,
          int** out_unique_tr_indices,
          int** out_tr_group_labels);

/* Find unique shot-index pass patterns (non-degenerate prep/cooldown, e.g. MPRAGE).
 * Returns count of unique pass patterns; caller frees both output arrays. */
int   pulseg__find_unique_shot_passes(
          const pulseg_sequence_descriptor* desc,
          int** out_unique_pass_indices,
          int** out_pass_group_labels);

/* --- pulseg_cache.c --- */
int   pulseg__try_read_cache(pulseg_collection* coll, const char* seq_path, const char* cache_ext);
int   pulseg__write_cache(pulseg_collection* seq_coll, const char* seq_path, const pulseg_opts* opts);

/* --- Helper to locate segment/block in collection --- */
int pulseg__resolve_segment(
    const pulseg_sequence_descriptor** out_desc,
    int* out_local_seg,
    const pulseg_collection* coll,
    int seg_idx);

int pulseg__resolve_block(
    const pulseg_sequence_descriptor** out_desc,
    const pulseg_tr_segment** out_seg,
    int* out_local_blk,
    const pulseg_collection* coll,
    int seg_idx, int blk_idx);

#endif /* PULSEG_INTERNAL_H */
