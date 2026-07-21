/**
 * @file pulseg_internal.h
 * @brief Private types and cross-translation-unit helpers of the pulseg core.
 *
 * NOT part of the public API: it lives under @c csrc/src/, outside the public
 * @c csrc/include/ tree, and is reachable only by a target that opts into the
 * private include directory. Everything declared here may change without
 * notice.
 *
 * Naming rule enforced across the library:
 *   - @c pulseg_  ... public, declared in @c csrc/include/pulseg/
 *   - @c pulseg__ ... private but shared across .c files, declared here
 *   - unprefixed  ... file-static, declared and defined in one .c file
 */

#ifndef PULSEG_INTERNAL_H
#define PULSEG_INTERNAL_H

#include <math.h>
#include <stdio.h>

#include "pulseg_config.h"
#include "pulseg_types.h"
#include "pulseg_io.h"

/* Error codes are public (pulseg_errors.h, included via pulseg_types.h). */

/* ================================================================== */
/*  Constants moved from public header (implementation details)       */
/* ================================================================== */
/* PULSEG_RF_USE_* now defined in pulseg_types.h (public; pulseg_seq_event's
 * doc comment already referenced these as public API). */

#define PULSEG_MAX_RF_SHIM_CHANNELS 64

#define PULSEG_TR_REGION_ALL (-1)

/* pulseq_definition and pulseq_trigger_event (used below by pointer in
 * pulseg_sequence_descriptor) are defined in pulseg_io.h, included above. */

/* ================================================================== */
/*  Segment timing anchors (internal)                                 */
/* ================================================================== */
typedef struct pulseg_segment_rf_anchor
{
    int block_offset;        /* block index within segment         */
    float start_us;          /* RF start time within segment (us)  */
    float end_us;            /* RF end time within segment (us)    */
    float isocenter_us;      /* isodelay time within segment (us)  */
    float base_amplitude_hz; /* base RF amplitude (Hz)             */
    int rf_use;              /* PULSEG_RF_USE_*                 */
} pulseg_segment_rf_anchor;

typedef struct pulseg_segment_adc_anchor
{
    int block_offset; /* block index within segment         */
    float start_us;   /* ADC start time within segment (us) */
    float end_us;     /* ADC end time within segment (us)   */
    int kzero_index;  /* k=0 sample index within readout    */
    float kzero_us;   /* k=0 time within segment (us)       */
} pulseg_segment_adc_anchor;

/* pulseq_shape is defined in pulseg_types.h (public, leaf type).
 * pulseq_trigger_event is defined in pulseg_io.h (embedded by value in
 * pulseq_block); PULSEQ_TRIGGER_EVENT_INIT lives there too. */

/* ================================================================== */
/*  RF definitions and table                                          */
/* ================================================================== */
typedef struct pulseg_rf_definition
{
    int id;
    int mag_shape_id;
    int phase_shape_id;
    int time_shape_id;
    int delay;
    int num_channels;      /* 1 for standard, >1 for dynamic pTx */
    pulseg_rf_stats stats; /* always present (runtime vendor check) */
} pulseg_rf_definition;

typedef struct pulseg_rf_table_element
{
    int id;
    float amplitude;
    float freq_offset;
    float phase_offset;
    int rf_use; /* PULSEG_RF_USE_* (0 = unknown) */
} pulseg_rf_table_element;

/* ================================================================== */
/*  RF shim definitions (parallel transmit channel weights)           */
/* ================================================================== */
typedef struct pulseg_rf_shim_definition
{
    int id;
    int num_channels;
    float magnitudes[PULSEG_MAX_RF_SHIM_CHANNELS];
    float phases[PULSEG_MAX_RF_SHIM_CHANNELS];
} pulseg_rf_shim_definition;

/* ================================================================== */
/*  Gradient definitions and table                                    */
/* ================================================================== */
typedef struct pulseg_grad_definition
{
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

typedef struct pulseg_grad_table_element
{
    int id;
    int shot_index;
    float amplitude;
} pulseg_grad_table_element;

/* ================================================================== */
/*  ADC definitions and table                                         */
/* ================================================================== */
typedef struct pulseg_adc_definition
{
    int id;
    int num_samples;
    int dwell_time;
    int delay;
} pulseg_adc_definition;

typedef struct pulseg_adc_table_element
{
    int id;
    float freq_offset;
    float phase_offset;
} pulseg_adc_table_element;

/* ================================================================== */
/*  Frequency modulation definitions                                  */
/* ================================================================== */
typedef struct pulseg_freq_mod_definition
{
    int id;
    int num_samples;       /* samples per axis (uniform raster) */
    float raster_us;       /* sample spacing in us */
    float duration_us;     /* active region duration */
    float *waveform_gx;    /* [num_samples] peak-normalized gradient, x */
    float *waveform_gy;    /* [num_samples] peak-normalized gradient, y */
    float *waveform_gz;    /* [num_samples] peak-normalized gradient, z */
    float ref_integral[3]; /* integral from start to reference point
                               * (gx, gy, gz) in [rad/Hz], pre-multiplied
                               * by 2*pi so that phase = ref_integral * freq */
    float ref_time_us;     /* reference time relative to active region
                               * start (isodelay for RF, kzero for ADC) */
} pulseg_freq_mod_definition;

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
typedef struct pulseg_freq_mod_library
{
    /* --- Deduped 3-channel entries (shift-independent) --- */
    int num_entries;        /* unique (base_shape, eff_amp) combos     */
    int max_samples;        /* longest entry waveform (zero-padded)    */
    float raster_us;        /* common time raster (us)                 */
    int *entry_num_samples; /* [num_entries]                           */

    /* Planar layout: 3ch[e * max_samples * 3 + ch * max_samples + s].
     * NULL after construction for non-PMC subsequences.               */
    float *entry_waveform_3ch; /* [num_entries * max_samples * 3] or NULL */
    float *entry_ref_3ch;      /* [num_entries * 3]               or NULL */

    /* Deep-copy rotation matrices from descriptor. */
    int num_rotations;
    float (*rotations)[9]; /* [num_rotations][9]                      */

    /* --- Plan instances (deduped on entry_idx x rotation_idx) --- */
    int num_plan_instances;
    int *pi_entry_idx;    /* [num_plan_instances]                    */
    int *pi_rotation_idx; /* [num_plan_instances]                    */

    /* Precomputed 1D waveforms (shift-dependent). */
    float *plan_waveform_data; /* flat [num_plan_instances * max_samples] */
    float **plan_waveforms;    /* [num_plan_instances] row pointers       */
    int *plan_num_samples;     /* [num_plan_instances] actual length      */
    float *plan_phase;         /* [num_plan_instances] phase comp (rad)
                                 * from all 3 channels                     */

    /* Hardware-format waveforms: short DAC units with WEOS_BIT on last.
     * Conversion: sample = (int16)(Hz / (4 * TARDIS_FREQ_RES))          */
    short *plan_hw_data;       /* flat [num_plan_instances * max_samples] */
    short **plan_hw_waveforms; /* [num_plan_instances] row pointers       */

    /* O(1) accessor by scan-table position. */
    int exec_stream_len;
    int *scan_to_plan; /* [exec_stream_len] -> plan instance, -1   */

    /* Optional cache fields retained for backward cache compatibility. */
    float *scan_inactive_area_3ch; /* [exec_stream_len * 3]  or NULL */
    float *scan_phase_extra;       /* [exec_stream_len]      or NULL */
} pulseg_freq_mod_library;

/* ================================================================== */
/*  Frequency modulation collection (opaque from public API)          */
/* ================================================================== */

/*
 * Wraps per-subsequence freq-mod libraries into a single object.
 * The public opaque type pulseg_freq_mod_collection points here.
 */
struct pulseg_freq_mod_collection
{
    int num_subsequences;
    pulseg_freq_mod_library **libs; /* [num_subsequences] (owned) */
};

/* Free a freq-mod collection (used by pulseg_collection_free). */
void pulseg_freq_mod_collection_free(struct pulseg_freq_mod_collection *fmc);

/* ================================================================== */
/*  Base blocks and block table                                      */
/* ================================================================== */
typedef struct pulseg_base_block
{
    int id;
    int duration_us;
    int rf_id;
    int gx_id;
    int gy_id;
    int gz_id;
    int adc_id; /* unique ADC definition index, -1 = no ADC */
} pulseg_base_block;

/* clang-format off */
#define PULSEG_BASE_BLOCK_INIT {0, 0, 0, 0, 0, 0, -1}
/* clang-format on */

typedef struct pulseg_block_table_element
{
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
    int freq_mod_id; /* boolean: >= 0 if block needs freq-mod, -1 otherwise */
    int rf_shim_id;  /* index into rf_shim_definitions, or -1 */
    int module_id;   /* sticky MODULE label id, 0 = ungrouped (default) */
} pulseg_block_table_element;

/* NOTE: digitalout_id occupies the former trigger_id position */

/* ================================================================== */
/*  TR descriptor                                                     */
/* ================================================================== */
typedef struct pulseg_tr_descriptor
{
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

/* clang-format off */
#define PULSEG_TR_DESCRIPTOR_INIT {0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0f}
/* clang-format on */

/* Per-segment timing summary */
typedef struct pulseg_segment_timing
{
    int num_rf_anchors;
    pulseg_segment_rf_anchor *rf_anchors;
    int num_adc_anchors;
    pulseg_segment_adc_anchor *adc_anchors;
} pulseg_segment_timing;

/* clang-format off */
#define PULSEG_SEGMENT_TIMING_INIT {0, NULL, 0, NULL}
/* clang-format on */

/* ================================================================== */
/*  Per-(segment, block-position) initial event state                 */
/* ================================================================== */
/* Resolved ONCE at parse time from the representative (max-energy) scan
 * instance of the segment, so that consumers which materialise segment
 * memory -- the pulsegen pass and the cross-subsequence dedup key -- never
 * need the O(scan-length) per-instance tables (exec_stream, block_table,
 * rf_table, grad_table).  Runtime per-instance values (amplitudes actually
 * played, wave swaps, rotations) still come from the scan cursor.
 *
 * Fallback values match the pre-computation behaviour when no representative
 * instance existed: rf_amplitude_hz = the RF definition's base amplitude,
 * grad_amplitude_hz_per_m = 1.0, grad_shot_index = 0, ids = -1. */
typedef struct pulseg_block_initial_state
{
    int base_block_id;  /* base_blocks index at the representative instance, -1 */
    int digitalout_id;  /* trigger_events index at this position, -1 = none     */
    int grad_def_id[3]; /* grad_definitions index per axis, -1 = no gradient    */
    int grad_shot_index[3];
    float rf_amplitude_hz;
    float grad_amplitude_hz_per_m[3];
} pulseg_block_initial_state;

/* clang-format off */
#define PULSEG_BLOCK_INITIAL_STATE_INIT \
    {-1, -1, {-1, -1, -1}, {0, 0, 0}, 0.0f, {1.0f, 1.0f, 1.0f}}
/* clang-format on */

/* Number of 4-byte words in pulseg_block_initial_state (cache serialization). */
#define PULSEG_BLOCK_INITIAL_STATE_WORDS 12

/* ================================================================== */
/*  Virtual segment                                                   */
/* ================================================================== */
typedef struct pulseg_virtual_segment
{
    int start_block;
    int num_blocks;
    int *unique_block_indices;
    int *has_digitalout;
    int *has_rotation;
    int *norot_flag;
    int *nopos_flag;
    int *has_freq_mod;
    int *has_adc;          /* OR-reduced: 1 if at least one segment instance has an ADC
                              event at this block position, 0 otherwise          */
    int *is_dynamic_delay; /* OR-reduced: 1 if this block position is an adjustable
                              pure delay (see is_adjustable_delay_at) AND its duration
                              actually differs across at least two scan-table instances
                              of this segment.  0 for a "static" delay -- same duration
                              in every instance -- which needs no runtime setperiod wait
                              and is represented purely by block position/offset.       */
    pulseg_block_initial_state *initial_states; /* [num_blocks], see above */
    int max_energy_start_block;
    int trigger_id; /* segment-level physio trigger (INPUT type),
                                   index into trigger_events[], or -1          */
    int is_nav;     /* 1 if all blocks in segment are NAV          */
    pulseg_segment_timing timing;
} pulseg_virtual_segment;

/* clang-format off */
#define PULSEG_VIRTUAL_SEGMENT_INIT \
    { \
    0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, -1, 0, \
    PULSEG_SEGMENT_TIMING_INIT \
    }
/* clang-format on */

/* ================================================================== */
/*  Segment table result                                              */
/* ================================================================== */
typedef struct pulseg_segment_table_result
{
    int num_unique_segments;
    int num_prep_segments;
    int *prep_segment_table;
    int num_main_segments;
    int *main_segment_table;
    int num_cooldown_segments;
    int *cooldown_segment_table;
} pulseg_segment_table_result;

/* clang-format off */
#define PULSEG_SEGMENT_TABLE_RESULT_INIT {0, 0, NULL, 0, NULL, 0, NULL}
/* clang-format on */

/* ================================================================== */
/*  Sequence descriptor                                               */
/* ================================================================== */
typedef struct pulseg_sequence_descriptor
{
    int num_prep_blocks;
    int num_cooldown_blocks;
    float rf_raster_us;
    float grad_raster_us;
    float adc_raster_us;
    float block_raster_us;
    int ignore_fov_shift;
    int enable_pmc;
    int ignore_averages;
    int num_gain_cal_readouts; /**< calibration readouts for APS2 receive gain (pislquant) */
    int num_passes;
    int pass_len;            /**< blocks per pass (= num_blocks when single-pass) */
    int num_averages;        /**< number of averages (1 if ignore_averages)       */
    int vendor;              /**< PULSEG_VENDOR_* runtime constant */
    int label_column_map[3]; /**< copy of pulseg_opts.label_column_map at dedup time */

    float fov[3];        /**< field of view (mm), from [DEFINITIONS] FOV */
    float matrix[3];     /**< matrix size, from [DEFINITIONS] Matrix    */
    float nav_fov[3];    /**< navigator FOV (mm), from [DEFINITIONS] NavFOV */
    float nav_matrix[3]; /**< navigator matrix, from [DEFINITIONS] NavMatrix */

    int num_unique_blocks;
    pulseg_base_block *base_blocks;
    int num_blocks;
    pulseg_block_table_element *block_table;

    int num_unique_rfs;
    pulseg_rf_definition *rf_definitions;
    int rf_table_size;
    pulseg_rf_table_element *rf_table;

    int num_unique_grads;
    pulseg_grad_definition *grad_definitions;
    int grad_table_size;
    pulseg_grad_table_element *grad_table;

    int num_unique_adcs;
    pulseg_adc_definition *adc_definitions;
    int adc_table_size;
    pulseg_adc_table_element *adc_table;

    int num_freq_mod_defs;
    pulseg_freq_mod_definition *freq_mod_definitions;

    int num_rf_shims;
    pulseg_rf_shim_definition *rf_shim_definitions;

    int num_rotations;
    float (*rotation_matrices)[9];

    int num_triggers;
    pulseq_trigger_event *trigger_events;

    int num_shapes;
    pulseq_shape *shapes;

    pulseg_tr_descriptor tr_descriptor;

    int num_unique_segments;
    pulseg_virtual_segment *segment_definitions;
    pulseg_segment_table_result segment_table;

    /* Scan table (expanded playback order).
     * Each row has 3 columns: block_table_idx, tr_id, seg_id.
     * Stored as 3 parallel arrays of length exec_stream_len. */
    int exec_stream_len;
    int *exec_stream_block_idx; /* [exec_stream_len] index into block_table */
    int *exec_stream_tr_id;     /* [exec_stream_len] TR region id           */
    int *exec_stream_seg_id;    /* [exec_stream_len] segment id             */
    int *exec_stream_avg_id;    /* [exec_stream_len] average (rep) index 0..num_averages-1 */
    int *exec_stream_tr_start;  /* [exec_stream_len] 1 at first block of each main-region TR */

    /* Per-position variable-gradient flags  [tr_size * 3].
     * Layout: flags[pos * 3 + axis] where axis 0=gx, 1=gy, 2=gz.
     * Value 1 means the gradient amplitude varies across TR instances
     * at that (position, axis); 0 means constant (or absent).  Used by
     * ZERO_VAR amplitude mode to zero out only the variable axes. */
    int *variable_grad_flags;

    /* label table (populated by dry-run if parse_labels is set) */
    int label_num_columns;
    int label_num_entries;
    int *label_table;
    pulseg_label_limits label_limits;
    /* Per-ADC OFF flag (parallel to label_table rows; entries == label_num_entries).
     * Pulseq v1.5.1 LABELSET column "OFF". 1 = acquisition should be discarded
     * downstream (LiveSDK), 0 = keep. NULL when no LABELSET OFF is present. */
    int *off_table;

    /* generic [DEFINITIONS] key-value pairs (all keys, not just reserved) */
    int num_definitions;
    pulseq_definition *definitions;

    /* Full-TR canonical (PULSEG_AMP_ZERO_VAR) k-space trajectory,
     * retained by pulseg__calc_segment_timing (pulseg_structure.c)
     * over the whole main TR (num_prep..num_prep+tr_size) with excitation-
     * reset + refocus-negation already applied. TRAJECTORY (section 6)
     * base shots are SLICED from this array instead of being
     * re-integrated + re-centered per block. NULL/0 if has_canonical_kspace
     * is 0 (e.g. zero-length TR). */
    int has_canonical_kspace;
    int canonical_kspace_num_samples;
    float canonical_kspace_dt_us; /* raster period of kx/ky/kz below (us) */
    float *canonical_kx;
    float *canonical_ky;
    float *canonical_kz;

    /* Copy of pulseg_opts.cache_ext at dedup time. Not part of the
     * cache payload itself (it only names the cache FILE, not its
     * contents) -- deliberately placed after all cache-serialized fields
     * so it needs no swap4_array count bump. */
    char cache_ext[PULSEG_CACHE_EXT_MAX];
} pulseg_sequence_descriptor;

/* clang-format off */
#define PULSEG_SEQUENCE_DESCRIPTOR_INIT \
    { \
    0, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0, 0, 0, 0, 1, 0, 1, 0, {0, 1, 2}, {0, 0, 0}, {0, 0, \
    0}, {0, 0, 0}, {0, 0, 0}, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, \
    NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, \
    PULSEG_TR_DESCRIPTOR_INIT, 0, NULL, PULSEG_SEGMENT_TABLE_RESULT_INIT, 0, NULL, NULL, \
    NULL, NULL, NULL, NULL, 0, 0, NULL, {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, \
    {0, 0}, {0, 0}, {0, 0}, {0, 0}}, NULL, 0, NULL, 0, 0, 0.0f, NULL, NULL, NULL, \
    PULSEG_CACHE_EXT_DEFAULT \
    }
/* clang-format on */

/* ================================================================== */
/*  Subsequence info                                                  */
/* ================================================================== */
typedef struct pulseg_subsequence_info
{
    int sequence_index;
    int adc_id_offset;
    int segment_id_offset;
    int block_index_offset;
} pulseg_subsequence_info;

/* ================================================================== */
/*  Block cursor                                                      */
/* ================================================================== */

typedef struct pulseg_block_cursor
{
    int sequence_index;
    int exec_stream_position; /* -1 = before first block */
    int from_last_reset;
} pulseg_block_cursor;

/* clang-format off */
#define PULSEG_BLOCK_CURSOR_INIT {0, -1, 0}
/* clang-format on */

/* ================================================================== */
/*  Sequence descriptor collection                                    */
/* ================================================================== */
struct pulseg_collection
{
    int num_subsequences;
    int num_repetitions;
    pulseg_block_cursor block_cursor;
    pulseg_sequence_descriptor *descriptors;
    pulseg_subsequence_info *subsequence_info;
    int total_unique_segments;
    int total_unique_adcs;
    int total_blocks;
    int total_readouts; /* ADC-bearing exec_stream positions, frozen at
                           assembly (the tables it derives from are not
                           loaded on the pulsegen cache path) */
    float total_duration_us;
    struct pulseg_freq_mod_collection *freq_mod; /* owned, may be NULL */

    /* Cross-subsequence segment deduplication remap (DERIVED state; rebuilt
     * by pulseg__build_segment_remap() after assembly and after cache load,
     * never serialized).  Without dedup these are the identity map.
     *
     *   seg_local_to_global[ subsequence_info[s].segment_id_offset + local ]
     *       = global (deduplicated) segment id, in [0, total_unique_segments)
     *   seg_repr_subseq[g] / seg_repr_local[g]
     *       = the (subseq, local) whose descriptor materialises global seg g.
     *
     * total_unique_segments is the DEDUPLICATED count once the remap is built;
     * segment_id_offset keeps its pre-dedup running-sum value (the flat index
     * base into seg_local_to_global).                                        */
    int *seg_local_to_global; /* [seg_l2g_len] local->global               */
    int seg_l2g_len;          /* == pre-dedup sum of num_unique_segments    */
    int *seg_repr_subseq;     /* [total_unique_segments] global->repr subseq */
    int *seg_repr_local;      /* [total_unique_segments] global->repr local  */
};

/* ================================================================== */
/*  Uniform-raster gradient waveforms (internal, post-interpolation)  */
/* ================================================================== */

/*
 * After extracting per-axis raw gradient tuples, the safety / acoustic
 * / PNS code interpolates them onto a common uniform raster.  This
 * struct holds the result.
 */
typedef struct pulseg__uniform_grad_waveforms
{
    int num_samples; /* same for all 3 axes */
    float raster_us; /* uniform sample spacing */
    float *gx;       /* [num_samples] amplitude (Hz / m) */
    float *gy;       /* [num_samples] amplitude (Hz / m) */
    float *gz;       /* [num_samples] amplitude (Hz / m) */
} pulseg__uniform_grad_waveforms;

/* clang-format off */
#define PULSEG__UNIFORM_GRAD_WAVEFORMS_INIT {0, 0.0f, NULL, NULL, NULL}
/* clang-format on */

/* ================================================================== */
/*  Internal constants                                                */
/* ================================================================== */
#define PULSEG__TWO_PI 6.283185307179586476925286766558
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================================================================== */
/*  Cross-file internal helper declarations (pulseg__ prefix)      */
/* ================================================================== */

/* --- pulseg_error.c --- */
void pulseg__diag_printf(pulseg_diagnostic *diag, const char *fmt, ...);

/* --- pulseg_math.c --- */
float pulseg__trapz_real_uniform(const float *s, int n, float dt);
float pulseg__trapz_real_nonuniform(const float *s, const float *t, int n);
float pulseg__max_slew_real_uniform(const float *s, int n, float dt);
float pulseg__max_slew_real_nonuniform(const float *s, const float *t, int n);
float pulseg__get_max_abs_real(const float *samples, int n);
void pulseg__quaternion_to_matrix(float *matrix, const float *quat);
int pulseg__is_identity3(const float *matrix);
void pulseg__apply_rotation(float *out, const float *R, const float *v, int transpose);
void pulseg__interp1_linear(
    float *out,
    const float *x,
    int nx,
    const float *xp,
    const float *fp,
    int nxp);
void pulseg__interp1_linear_complex(
    float *out_re,
    float *out_im,
    const float *x,
    int nx,
    const float *xp,
    const float *fp_re,
    const float *fp_im,
    int nxp);
void pulseg__fftshift_complex(float *re, float *im, int n);
float pulseg__get_spectrum_flank(
    const float *x,
    const float *re,
    const float *im,
    int n,
    float cutoff,
    int reverse);
size_t pulseg__next_pow2(size_t x);
int pulseg__calc_convolution_fft(
    float *output,
    const float *signal,
    int signal_len,
    const float *kernel,
    int kernel_len);

/* pulseg_parse.c's public entry points (pulseq_read family,
 * accessors, pulseq_decompress_shape, etc.) are declared in the
 * public header pulseg_io.h, included above. */

/* --- pulseg_core.c --- */
int pulseg__deduplicate_int_rows(
    int *unique_defs,
    int *event_table,
    const int *int_rows,
    int num_rows,
    int num_cols);
int pulseg__get_unique_blocks(
    pulseg_sequence_descriptor *desc,
    const pulseq_file *seq,
    const pulseg_opts *opts);

/* --- pulseg_structure.c --- */
int pulseg__get_tr_in_sequence(pulseg_sequence_descriptor *desc, pulseg_diagnostic *diag);
int pulseg__build_exec_stream(
    pulseg_sequence_descriptor *desc,
    pulseg_diagnostic *diag,
    int num_averages);
int pulseg__get_exec_stream_segments(
    pulseg_sequence_descriptor *desc,
    pulseg_diagnostic *diag,
    const pulseg_opts *opts);
int pulseg__build_freq_mod_flags(pulseg_sequence_descriptor *desc);
void pulseg__compute_exec_stream_tr_start(pulseg_sequence_descriptor *desc);
int pulseg__build_label_table(pulseg_sequence_descriptor *desc, const pulseq_file *seq);
int pulseg__calc_segment_timing(pulseg_sequence_descriptor *desc, pulseg_diagnostic *diag);

/* --- pulseg_core.c (continued) --- */
/* pulseg__get_collection_descriptors was promoted+renamed to the public
 * pulseg_convert_collection() (pulseg_convert.h). */
void pulseg_sequence_descriptor_free(pulseg_sequence_descriptor *desc);

/* --- pulseg_waveforms.c --- */

/* Compute per-position variable-gradient flags for ZERO_VAR mode.
 * Allocates desc->variable_grad_flags (tr_size * 3 ints).
 * Must be called after pulseg__get_tr_in_sequence. */
int pulseg__compute_variable_grad_flags(pulseg_sequence_descriptor *desc);

/* Free uniform waveforms. */
void pulseg__uniform_grad_waveforms_free(pulseg__uniform_grad_waveforms *w);

/* Extract gradient waveforms for an arbitrary block range,
 * interpolated to uniform raster (half gradient raster). */
int pulseg__get_gradient_waveforms_range(
    const pulseg_sequence_descriptor *desc,
    pulseg__uniform_grad_waveforms *out,
    pulseg_diagnostic *diag,
    int block_start,
    int block_count,
    int amplitude_mode,
    const int *tr_group_labels,
    int target_group,
    const int *block_order);

/* Find unique shot-index TR variants (multi-shot, degenerate prep/cooldown).
 * Returns count of unique groups; caller frees both output arrays. */
int pulseg__find_unique_shot_trs(
    const pulseg_sequence_descriptor *desc,
    int **out_unique_tr_indices,
    int **out_tr_group_labels);

/* Find unique shot-index pass patterns (non-degenerate prep/cooldown, e.g. MPRAGE).
 * Returns count of unique pass patterns; caller frees both output arrays. */
int pulseg__find_unique_shot_passes(
    const pulseg_sequence_descriptor *desc,
    int **out_unique_pass_indices,
    int **out_pass_group_labels);

/* --- pulseg_cache.c --- */
int pulseg__try_read_cache(pulseg_collection *coll, const char *seq_path, const char *cache_ext);
int pulseg__write_cache(pulseg_collection *seq_coll, const char *seq_path, const pulseg_opts *opts);

/* --- Helper to locate segment/block in collection --- */

/* Build (or rebuild) the cross-subsequence segment deduplication remap:
 * populates coll->seg_local_to_global / seg_repr_subseq / seg_repr_local and
 * collapses coll->total_unique_segments to the deduplicated count.  Idempotent;
 * must be called after all descriptors are assembled (post-convert and
 * post-cache-load).  Returns PULSEG_SUCCESS or an error code (on error the
 * collection is left with the identity map so it stays usable).            */
int pulseg__build_segment_remap(pulseg_collection *coll);

/* Free the derived remap arrays (identity-safe; sets them to NULL). */

/* --- pulseg_cache.c / pulseg_cache_seqdesc.c: .pge section writers --- *
 * Plumbing of pulseg_save_cache(); each appends one section to the cache
 * file already on disk and patches the section index. */
int pulseg__save_freq_mod_section(const pulseg_collection *coll, const char *seq_path);
int pulseg__save_seqdesc_cache_section(const pulseg_collection *coll, const char *seq_path);
int pulseg__save_trajectory_cache_section(const pulseg_collection *coll, const char *seq_path);
int pulseg__save_freq_mod_cache_section(pulseg_collection *coll, const char *seq_path);

/* --- pulseg_freqmod.c: FREQMOD payload codec on an already-open cache --- *
 * do_swap is non-zero when the cache file's endianness differs from the
 * reader's; every 4-byte word of the payload is swapped after reading. */
int pulseg__freq_mod_collection_write_f(const pulseg_freq_mod_collection *fmc, FILE *f);
int pulseg__freq_mod_collection_read_f(
    pulseg_freq_mod_collection **out_fmc,
    FILE *f,
    const pulseg_collection *coll,
    const float *shift_m,
    int do_swap);

#endif /* PULSEG_INTERNAL_H */
