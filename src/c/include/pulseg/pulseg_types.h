/**
 * @file pulseg_types.h
 * @brief Public type definitions, error codes, and initializer macros.
 *
 * All types intended for consumption by calling code are defined here.
 * Internal / opaque types live in pulseg_internal.h.
 *
 * Naming conventions:
 *   - Physical quantities carry unit suffixes: _us, _hz, _hz_per_m, etc.
 *   - All identifiers are snake_case.
 *   - INIT macros are C89 and C++ compatible (positional, no designated init).
 */

#ifndef PULSEG_TYPES_H
#define PULSEG_TYPES_H

#include "pulseq_types.h"

#include "pulseg_config.h"
#include "pulseg_errors.h"

/* ================================================================== */
/*  Gradient axes                                                     */
/* ================================================================== */
#define PULSEG_GRAD_AXIS_X 0
#define PULSEG_GRAD_AXIS_Y 1
#define PULSEG_GRAD_AXIS_Z 2

/* ================================================================== */
/*  RF use codes: the use tag an RF event carries.                    */
/*  Aliases of the raw Pulseq RF library's trailing e/r/i/s/p/o use   */
/*  tag, which the pulseq module owns.  All seven are aliased because */
/*  the parser produces all seven and seqdesc copies the field        */
/*  through verbatim -- a consumer switching on these must be able to */
/*  name the two that nothing here treats specially.                  */
/* ================================================================== */
#define PULSEG_RF_USE_UNKNOWN PULSEQ_RF_USE_UNKNOWN
#define PULSEG_RF_USE_EXCITATION PULSEQ_RF_USE_EXCITATION
#define PULSEG_RF_USE_REFOCUSING PULSEQ_RF_USE_REFOCUSING
#define PULSEG_RF_USE_INVERSION PULSEQ_RF_USE_INVERSION
#define PULSEG_RF_USE_SATURATION PULSEQ_RF_USE_SATURATION
#define PULSEG_RF_USE_PREPARATION PULSEQ_RF_USE_PREPARATION
#define PULSEG_RF_USE_OTHER PULSEQ_RF_USE_OTHER

/* ================================================================== */
/*  Error codes                                                       */
/* ================================================================== */

/** @defgroup errcodes Error codes
 *  Every public function returns a plain int:
 *    positive  = success (PULSEG_OK)
 *    negative  = failure
 *
 *  On failure the caller should read the diagnostic message string
 *  (filled by every function that accepts a pulseg_diagnostic*)
 *  and pass it to the vendor error-reporting routine.  Specific
 *  negative values are library-internal and must NOT be matched by
 *  consumers.
 *  @{ */

#define PULSEG_SUCCESS 1

#define PULSEG_SUCCEEDED(code) ((code) > 0)
#define PULSEG_FAILED(code) ((code) < 0)

/** @} */

/* ================================================================== */
/*  Cursor states                                                     */
/* ================================================================== */
#define PULSEG_CURSOR_BLOCK 0
#define PULSEG_CURSOR_DONE 1

/* ================================================================== */
/*  Max-size constants                                                */
/* ================================================================== */
#define PULSEG_DIAG_MSG_LEN 256

/* ================================================================== */
/*  Diagnostic                                                        */
/* ================================================================== */

/**
 * @brief Diagnostic info returned by library functions on failure.
 *
 * On error, @c code is set to a negative PULSEG_ERR_* value and
 * @c message contains a human-readable description (may include
 * offending block index, axis, amplitude, etc.).
 */
typedef struct pulseg_diagnostic
{
    int code;
    char message[PULSEG_DIAG_MSG_LEN];
} pulseg_diagnostic;

/* clang-format off */
#define PULSEG_DIAGNOSTIC_INIT {PULSEG_SUCCESS, { '\0' }}
/* clang-format on */

/* ================================================================== */
/*  Shape (RLE-decompressible waveform)                               */
/*                                                                    */
/*  Owned by the pulseq module (pulseq_types.h) -- the IR keeps Pulseq */
/*  RLE shapes as its waveform store, so the raw shape library entry   */
/*  and the descriptor's shape storage are deliberately the same type. */
/*  pulseg_shape_arbitrary remains available as a compatibility alias. */
/* ================================================================== */

typedef pulseq_shape pulseg_shape_arbitrary;

/* ================================================================== */
/*  RF envelope view (for the vendor RF-stats callback)               */
/* ================================================================== */

/**
 * @brief Read-only view of a uniform-raster RF envelope, handed to an
 * optional vendor callback (@c pulseg_opts.vendor_rf_stats_fn) so it can
 * compute vendor-specific envelope statistics without owning any of the
 * dedup-time buffers.
 */
typedef struct pulseg_rf_view
{
    const float *mag;     /**< |B1(t)| envelope, normalised, length n */
    const float *phase;   /**< phase (rad), length n                  */
    int n;                /**< sample count                           */
    float dt_us;          /**< uniform raster period (us)             */
    float duration_us;    /**< RF event duration (us)                 */
    float tr_duration_us; /**< enclosing TR duration (us); 0 if unknown at
                                 dedup time                            */
} pulseg_rf_view;

/* ================================================================== */
/*  System options                                                    */
/* ================================================================== */

/**
 * @brief The scanner the conversion reads a sequence for.
 *
 * All raster times are in microseconds.
 */
typedef struct pulseg_opts
{
    int vendor;                    /**< PULSEG_VENDOR_* constant       */
    float gamma_hz_per_t;          /**< gyromagnetic ratio  (Hz / T)      */
    float b0_t;                    /**< static field strength (T)         */
    float rf_raster_us;            /**< RF sample raster (us)             */
    float grad_raster_us;          /**< gradient sample raster (us)       */
    float adc_raster_us;           /**< ADC dwell raster (us)             */
    float block_raster_us;         /**< block duration raster (us)        */

    /** Optional vendor RF envelope-stats callback. NULL -> the four
     *  pulseg_rf_stats.vendor_stat[] slots are left at 0. */
    int (*vendor_rf_stats_fn)(void *ctx, const pulseg_rf_view *rf, float out_stat[4]);
    void *vendor_rf_stats_ctx;

    /**
     * @brief Which Pulseq label fills output column 0/1/2 of the 3-column
     * ADC label table. Values are Pulseq label *state-array* indices:
     * 0=SLC, 1=PHS, 2=REP, 3=AVG, 4=SEG, 5=SET, 6=ECO, 7=PAR, 8=LIN, 9=ACQ.
     * Example (GE convention): {8, 0, 6} = [LIN, SLC, ECO]. Public default
     * is the identity {0, 1, 2} = [SLC, PHS, REP]; vendor layers override
     * this before parsing (see pulserver_ge_config.h in the private
     * pulserver-interpreter for the GE values).
     */
    int label_column_map[3];

    /** Binary cache file extension, including the dot. Default
     *  ".pseg"; GE overrides to ".pge" (see pulserver_ge_config.h). Only
     *  the main pulseg_read()/pulseg_save_cache() path honors this;
     *  standalone cache utilities (pulseg_load_cache, pulseg_clear_cache,
     *  etc.) that run before any collection exists always use the public
     *  default. */
    char cache_ext[PULSEG_CACHE_EXT_MAX];

    /** Optional opaque vendor cache section. Writer emits a section
     *  only when set; GE leaves this unused. ctx/buf ownership: the
     *  callback allocates *out_buf via PULSEG_ALLOC; the cache writer
     *  frees it after use. */
    int (*vendor_section_write_fn)(void *ctx, unsigned char **out_buf, int *out_len);
    void *vendor_section_ctx;

    /**
     * @brief Accept RF amplitude that varies across canonical TR instances.
     *
     * Default 1. When set, a subsequence whose positional RF amplitude
     * pattern differs between TR instances (or passes) is accepted instead
     * of returning @c PULSEG_ERR_CONSISTENCY_RF_PERIODIC; the descriptor's
     * @c rf_amplitude_variable flag is raised and the RF safety model is
     * built from the *positional-max envelope* rather than one canonical
     * instance -- see pulseg_get_rf_array().
     *
     * RF *shim* pattern variation stays rejected regardless of this flag:
     * VOP SAR with changing shim vectors is not order-monotone in any
     * per-position scalar, so no envelope dominates it.
     *
     * Set to 0 to restore the strict periodicity gate.
     */
    int allow_variable_rf_amplitude;
    /**
     * @brief Convert only what the scan's structure needs.
     *
     * A file written for structure alone carries its gradient shapes without
     * samples; with this set the conversion computes no gradient statistics
     * from them, and the collection answers structural questions (the TR,
     * its instances, the segments) while refusing any waveform or safety
     * request. Default 0.
     */
    int structure_only;
    /**
     * @brief Let a collection read from memory keep its gradient shape
     * samples in the caller's buffers instead of copying them.
     *
     * Only pulseg_read_from_buffers() honours it, and only when the file's
     * sample cells are the host's floats; the caller then keeps every
     * buffer alive for as long as the collection lives. Default 0.
     */
    int borrow_buffer_shapes;
} pulseg_opts;

/* clang-format off */
#define PULSEG_OPTS_INIT \
    { \
    0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, NULL, NULL, {0, 1, 2}, \
    PULSEG_CACHE_EXT_DEFAULT, NULL, NULL, 1, 0, 0 \
    }
/* clang-format on */

/* ================================================================== */
/*  RF statistics                                                     */
/* ================================================================== */

/** Maximum simultaneous frequency bands detectable in a multiband RF pulse. */
#define PULSEG_MAX_BANDS 8

/**
 * @brief Per-RF-definition statistics (always available).
 */
typedef struct pulseg_rf_stats
{
    float flip_angle_rad;    /**< nominal flip angle (radians)           */
    float act_amplitude_hz;  /**< actual |gamma*B1| amplitude (Hz). When the
                              *  descriptor's rf_amplitude_variable flag is
                              *  set, this is the REAL amplitude of the
                              *  worst-B1rms TR instance at this position
                              *  (not a synthetic per-position envelope) --
                              *  feeds time-averaged SAR / amplifier-duty
                              *  consumers (GE minseqrfamp/maxsar). See
                              *  peak_amplitude_hz for the peak-dominant
                              *  counterpart. */
    float peak_amplitude_hz; /**< positional-max |gamma*B1| amplitude (Hz)
                               *  across every TR instance at this position
                               *  -- for peak-only consumers (GE peakB1())
                               *  that need per-position dominance across
                               *  ALL instances, which act_amplitude_hz no
                               *  longer guarantees once it tracks a single
                               *  real worst-B1rms instance. Equal to
                               *  act_amplitude_hz for periodic sequences. */
    float area;              /**< integral of |B1(t)| dt  (a.u.)        */
    /** Vendor-specific envelope statistics, filled by the optional
     *  pulseg_opts.vendor_rf_stats_fn callback; all 0 when unset. Meaning
     *  is vendor-defined -- e.g. GE's abswidth/effwidth/dtycyc/maxpw live
     *  in src_gelib/pulserver_ge_rf_stats.h as PULSERVER_GE_RF_* accessors. */
    float vendor_stat[4];
    float duration_us;       /**< total RF event duration (us)          */
    int isodelay_us;         /**< isodelay from center to echo (us)     */
    float bandwidth_hz;      /**< bandwidth at half the spectral peak (Hz) */
    float base_amplitude_hz; /**< base (nominal) peak |gamma*B1| (Hz)   */
    int num_samples;         /**< waveform sample count                 */
    int num_instances;       /**< repetition count for this RF pulse    */
    /* --- multiband / power fields (appended; do not reorder above) --- */
    int num_bands; /**< number of simultaneous frequency bands (>=1) */
    float band_freq_offsets_hz
        [PULSEG_MAX_BANDS];  /**< per-band center offsets relative to carrier (Hz) */
    float band_bandwidth_hz; /**< widest band's bandwidth (Hz) */
    float total_b1sq_power;  /**< integral |B1(t)|^2 dt normalised (a.u.) */
    /* --- vendor tag (appended; identifies the meaning of the
     *     vendor-specific interpretation of the fields above; for new
     *     vendor variants, a sibling struct may be added later and
     *     selected via this field) ---                                 */
    int vendor; /**< PULSEG_VENDOR_* constant (0 = unspecified -> GEHC for back-compat) */
    /* --- safety-group label (appended; do not reorder above) --- */
    int trid; /**< sticky pulseq TRID of the originating block, 0 = ungrouped */
} pulseg_rf_stats;

/* clang-format off */
#define PULSEG_RF_STATS_INIT \
    { \
    0.0f, 0.0f, 0.0f, 0.0f, {0.0f}, 0.0f, 0, 0.0f, 0.0f, 0, 0, 1, {0.0f}, 0.0f, 0.0f, 0, 0 \
    }
/* clang-format on */

/**
 * @brief One entry per distinct TRID-labeled group in a subsequence, as
 * identified/verified/deduplicated by pulseg_get_tr_groups() from the
 * materialized scan table. See pulseg_get_tr_groups() for the identify ->
 * verify-structural-identity -> dedup algorithm.
 *
 * TRID is Pulseq's own label for "the repeating unit of the sequence"
 * (mr.getSupportedLabels: "an integer ID of the TR (sequence segment) used by
 * the GE interpreter (and some others) to optimize the execution on the
 * scanner"), which is exactly the grouping a per-contrast SAR check wants --
 * see the NeuroMix scheme, where each contrast is evaluated against the 10 s
 * limit and the whole run against the 6 min one.
 */
typedef struct pulseg_tr_group
{
    int trid;                     /**< sticky pulseq TRID (>=1)          */
    int one_instance_duration_us; /**< duration of the validated reference occurrence */
    int total_duration_us;        /**< num_instances * one_instance_duration_us */
    int num_instances;            /**< count of structurally-identical occurrences */
} pulseg_tr_group;

/* ================================================================== */
/*  TR region selectors (for freq-mod plan)                           */
/* ================================================================== */
#define PULSEG_TR_REGION_PREP 0
#define PULSEG_TR_REGION_MAIN 1
#define PULSEG_TR_REGION_COOLDOWN 2

/* ================================================================== */
/*  Frequency modulation collection                                   */
/* ================================================================== */

/* ================================================================== */
/*  Opaque collection handle                                          */
/* ================================================================== */

/**
 * @brief Opaque handle to a loaded Pulseq sequence collection.
 *
 * Created by pulseg_read() or pulseg_read_from_buffers().
 * All getter functions take a const pointer to this type.
 * Freed by pulseg_collection_free().
 */
typedef struct pulseg_collection pulseg_collection;

/**
 * @brief A caller-owned character buffer the library writes into.
 *
 * @c capacity counts bytes including the terminating NUL, so the library
 * writes at most @c capacity-1 characters.  A zero @c capacity (@c data may
 * then be NULL) means the caller does not want the text.
 */
typedef struct pulseg_text_buffer
{
    int capacity; /**< bytes available in @c data, NUL included */
    char *data;   /**< [capacity] destination, or NULL          */
} pulseg_text_buffer;

/* clang-format off */
#define PULSEG_TEXT_BUFFER_INIT {0, NULL}
/* clang-format on */

/* ================================================================== */
/*  Label limits                                                      */
/*  pulseq_label_limit (the per-label min/max pair) is owned by the    */
/*  pulseq module; pulseg groups one per Pulseq label below.           */
/* ================================================================== */

typedef pulseq_label_limit pulseg_label_limit;

/** @brief Observed [min, max] of every Pulseq counter label in a subsequence. */
typedef struct pulseg_label_limits
{
    pulseq_label_limit slc;
    pulseq_label_limit phs;
    pulseq_label_limit rep;
    pulseq_label_limit avg;
    pulseq_label_limit seg;
    pulseq_label_limit set;
    pulseq_label_limit eco;
    pulseq_label_limit par;
    pulseq_label_limit lin;
    pulseq_label_limit acq;
} pulseg_label_limits;

/* ================================================================== */
/*  Block instance (cursor output)                                    */
/* ================================================================== */

/**
 * @brief Resolved block data for the current cursor position.
 *
 * Returned by pulseg_get_block_instance().  Amplitudes are in
 * Pulseq native units (Hz for RF, Hz/m for gradients).
 */
typedef struct pulseg_block_instance
{
    int duration_us; /**< block duration (us)                */

    /* RF */
    float rf_amp_hz;    /**< RF amplitude (Hz, = gamma*B1)     */
    float rf_freq_hz;   /**< RF frequency offset (Hz)          */
    float rf_phase_rad; /**< RF phase offset (rad)             */
    int rf_shim_id;     /**< RF shim definition index (-1=none)*/

    /* Gradients */
    float gx_amp_hz_per_m; /**< GX amplitude (Hz / m)             */
    float gy_amp_hz_per_m; /**< GY amplitude (Hz / m)             */
    float gz_amp_hz_per_m; /**< GZ amplitude (Hz / m)             */
    int gx_shape_id;       /**< GX pulseq shape id, 0 = trapezoid */
    int gy_shape_id;       /**< GY pulseq shape id, 0 = trapezoid */
    int gz_shape_id;       /**< GZ pulseq shape id, 0 = trapezoid */
    int gx_variable;       /**< 1 if GX amplitude varies across TRs */
    int gy_variable;       /**< 1 if GY amplitude varies across TRs */
    int gz_variable;       /**< 1 if GZ amplitude varies across TRs */

    /* Rotation */
    float rotmat[9]; /**< 3x3 rotation matrix (row-major)   */
    int norot_flag;  /**< 1 = skip rotation for this block  */
    int nopos_flag;  /**< 1 = skip repositioning            */

    /* Digital output */
    int digitalout_flag;    /**< 1 = digital output event present  */
    int digitalout_channel; /**< trigger channel, -1 if absent     */

    /* ADC */
    int adc_flag;        /**< 1 = ADC acquisition active        */
    float adc_freq_hz;   /**< ADC frequency offset (Hz)         */
    float adc_phase_rad; /**< ADC phase offset (rad)            */

    /* Safety group (sticky pulseq TRID, 0 = ungrouped) */
    int trid;
} pulseg_block_instance;

/* clang-format off */
#define PULSEG_BLOCK_INSTANCE_INIT \
    { \
    0, 0.0f, 0.0f, 0.0f, -1, 0.0f, 0.0f, 0.0f, 0, 0, 0, 0, 0, 0, {1, 0, 0, 0, 1, 0, 0, \
    0, 1}, 0, 0, 0, -1, 0, 0.0f, 0.0f, 0 \
    }
/* clang-format on */

/* ================================================================== */
/*  Cursor info                                                       */
/* ================================================================== */

/**
 * @brief Position and context metadata for the current cursor block.
 *
 * Returned by pulseg_cursor_get_info() after a successful
 * pulseg_cursor_next() call.
 */
typedef struct pulseg_cursor_info
{
    int subseq_idx;    /**< current subsequence index                     */
    int scan_pos;      /**< exec-stream position; indexes a chunk plan's
                            position_wave[] to find this block's waveform */
    int segment_id;    /**< current segment ID (global)                   */
    int segment_start; /**< 1 if first block of current segment           */
    int segment_end;   /**< 1 if last block of current segment            */
    int is_nav;        /**< 1 if current segment is a NAV segment         */
    int has_trigger;   /**< 1 if current segment has a trigger/digitalout */
    int tr_start;      /**< 1 if first block of a main-region TR          */
    int pmc;           /**< 1 if current subsequence has PMC enabled      */
} pulseg_cursor_info;

/* clang-format off */
#define PULSEG_CURSOR_INFO_INIT {0, 0, -1, 0, 0, 0, 0, 0, 0}
/* clang-format on */

/* ================================================================== */
/*  Scan-time query result                                            */
/* ================================================================== */

/**
 * @brief Scan-time summary.
 *
 * When returned by pulseg_peek_scan_time(), only
 * @c total_duration_us is populated (summed from the [DEFINITIONS]
 * sections of the chain) and @c total_segment_boundaries is left at 0.
 *
 * When computed from a fully-loaded collection via
 * pulseg_get_scan_time(), both fields are accurate: every block duration
 * and every segment boundary in the scan table.
 */
typedef struct pulseg_scan_time_info
{
    float total_duration_us;      /**< total sequence duration (us)  */
    int total_segment_boundaries; /**< total segment boundary count  */
} pulseg_scan_time_info;

/* clang-format off */
#define PULSEG_SCAN_TIME_INFO_INIT {0.0f, 0}
/* clang-format on */

/* ================================================================== */
/*  Collection-level sequence flags                                   */
/* ================================================================== */

/**
 * @brief Flags a collection declares about itself.
 *
 * These are properties of the whole scan, so they are read from the head
 * .seq of a NextSequence chain; the rest of the chain does not carry them.
 */
typedef struct pulseg_sequence_flags
{
    int enable_sar_burst_mode; /**< 1 if the scan asks for SAR burst limits */
} pulseg_sequence_flags;

/* clang-format off */
#define PULSEG_SEQUENCE_FLAGS_INIT {0}
/* clang-format on */

/* ================================================================== */
/*  Collection info (replaces individual collection-level getters)    */
/* ================================================================== */

/**
 * @brief Summary information about a loaded collection.
 *
 * Returned by pulseg_get_collection_info().
 */
typedef struct pulseg_collection_info
{
    int num_subsequences;    /**< number of subsequences              */
    int num_segments;        /**< total unique segments               */
    int max_adc_samples;     /**< max sample count across all ADCs    */
    int total_readouts;      /**< total ADC readout events            */
    float total_duration_us; /**< total sequence duration (us)        */
} pulseg_collection_info;

/* clang-format off */
#define PULSEG_COLLECTION_INFO_INIT {0, 0, 0, 0, 0.0f}
/* clang-format on */

/* ================================================================== */
/*  Subsequence info (replaces per-subsequence getters)               */
/* ================================================================== */

/**
 * @brief Metadata for a single subsequence.
 *
 * Returned by pulseg_get_subseq_info().
 */
typedef struct pulseg_subseq_info
{
    float tr_duration_us;      /**< TR duration (us)                    */
    int num_trs;               /**< number of TRs                       */
    int tr_size;               /**< blocks per TR                       */
    int num_unique_adcs;       /**< unique ADC definitions              */
    int num_unique_rf;         /**< unique RF definitions               */
    int pmc_enabled;           /**< 1 if PMC (prospective motion corr)  */
    int segment_offset;        /**< global segment index offset         */
    int num_adc_occurrences;   /**< ADC entries in label table          */
    int num_label_columns;     /**< label columns (vendor-dependent)    */
    int num_gain_cal_readouts; /**< calibration readouts for APS2 gain cal (pislquant) */
    /** TR instances the subsequence plays; always >= 1. */
    int num_tr_instances;
    /** 1 where the RF amplitude at some position differs between TR
     *  instances, which is what makes pulseg_get_rf_array() report a
     *  positional-max envelope rather than one canonical instance. */
    int rf_amplitude_variable;
} pulseg_subseq_info;

/* clang-format off */
#define PULSEG_SUBSEQ_INFO_INIT \
    { \
    0.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 \
    }
/* clang-format on */

/* ================================================================== */
/*  Segment info (replaces per-segment getters)                       */
/* ================================================================== */

/**
 * @brief Metadata for a single segment.
 *
 * Returned by pulseg_get_segment_info().
 */
typedef struct pulseg_segment_info
{
    int duration_us;         /**< total segment duration (us)         */
    int num_blocks;          /**< unique blocks in the segment        */
    int start_block;         /**< start block index in the sequence   */
    int pure_delay;          /**< 1 if segment is a bare delay        */
    int has_trigger;         /**< 1 if physio trigger attached        */
    int trigger_type;        /**< trigger type (1=output/TTL, 2=input/ECG), 0 if none */
    int trigger_delay_us;    /**< trigger delay (us), -1 if none      */
    int trigger_duration_us; /**< trigger duration (us), -1 if none   */
    int is_nav;              /**< 1 if navigator segment              */
    int rf_adc_gap_us;       /**< RF->ADC gap (us), -1 if no pair     */
    int adc_adc_gap_us;      /**< min ADC->ADC gap (us), -1 if < 2    */
} pulseg_segment_info;

/* clang-format off */
#define PULSEG_SEGMENT_INFO_INIT {0, 0, 0, 0, 0, 0, -1, -1, 0, -1, -1}
/* clang-format on */

/** Trigger type constants (public, matching internal definitions). */
#define PULSEG_TRIGGER_TYPE_OUTPUT 1 /**< TTL / digital output */
#define PULSEG_TRIGGER_TYPE_INPUT 2  /**< ECG / cardiac gating */

/* ================================================================== */
/*  Segment layout (subsequence-local segment, resolved to a scan     */
/*  instance -- replaces ad hoc consumer-side exec_stream walking)    */
/* ================================================================== */

/**
 * @brief Position/layout of one subsequence-local unique segment,
 * resolved to a concrete .seq block range.
 *
 * Returned by pulseg_get_subseq_segment_layout(). @c global_index is the
 * deduplicated cross-subsequence segment id (same space as every seg_idx
 * elsewhere in this API, e.g. pulseg_get_segment_info()); pair with
 * pulseg_get_subseq_segment_block_indices() for the resolved per-position
 * .seq block indices.
 */
typedef struct pulseg_segment_layout
{
    int global_index;             /**< deduplicated global segment id      */
    int num_blocks;               /**< blocks in the segment               */
    int start_block;              /**< segment definition's own start block
                                        (used when no scan-table instance
                                        is available)                      */
    int max_energy_start_block;   /**< start block of the max-energy scan
                                        instance, -1 if none                */
    int from_max_energy_instance; /**< 1 if pulseg_get_subseq_segment_-
                                        block_indices() resolved via the
                                        max-energy scan instance, 0 if it
                                        fell back to @c start_block         */
} pulseg_segment_layout;

/* clang-format off */
#define PULSEG_SEGMENT_LAYOUT_INIT {-1, 0, 0, -1, 0}
/* clang-format on */

/* ================================================================== */
/*  Block info (replaces per-block has/get accessor pairs)            */
/* ================================================================== */

/**
 * @brief Metadata for a single block within a segment.
 *
 * Returned by pulseg_get_block_info().
 * Waveform data is NOT included — use the dedicated waveform getters
 * (e.g.\ pulseg_get_grad_amplitude) keyed by the metadata here.
 */
typedef struct pulseg_block_info
{
    int duration_us;   /**< block duration (us)               */
    int start_time_us; /**< start time within segment (us)    */

    /* Gradient (per axis: [0]=X, [1]=Y, [2]=Z) */
    int has_grad[3];          /**< 1 if gradient present             */
    int grad_is_trapezoid[3]; /**< 1 if trapezoid (not arbitrary)    */
    int grad_delay_us[3];     /**< gradient delay (us), -1 if absent */
    int grad_num_shots[3];    /**< shot count, -1 if absent          */
    int grad_num_samples[3];  /**< sample count, -1 if absent        */
    int grad_def_id[3];       /**< gradient definition index, -1 if absent: the
                                   id a mechanical-resonance refusal names */

    /* RF */
    int has_rf;            /**< 1 if RF event present             */
    int rf_delay_us;       /**< RF delay (us), -1 if absent       */
    int rf_num_channels;   /**< Tx channel count, -1 if absent    */
    int rf_num_samples;    /**< samples per channel, -1 if absent */
    int rf_duration_us;    /**< RF duration (us) from last time-shape sample; -1 if absent */
    int rf_is_complex;     /**< 1 if phase shape exists           */
    int rf_uniform_raster; /**< 1 if time shape present           */

    /* ADC */
    int has_adc;      /**< 1 if ADC acquisition active       */
    int adc_delay_us; /**< ADC delay (us), -1 if absent      */
    int adc_def_id;   /**< global ADC library index, -1      */

    /* Digital output */
    int has_digitalout;         /**< 1 if digital output present       */
    int digitalout_delay_us;    /**< delay (us), -1 if absent          */
    int digitalout_duration_us; /**< duration (us), -1 if absent       */
    int digitalout_channel;     /**< trigger channel, -1 if absent     */

    /* Flags */
    int has_rotation;       /**< 1 if rotation event present       */
    int norot_flag;         /**< 1 if no-rotation override         */
    int nopos_flag;         /**< 1 if no-position override         */
    int is_variable_delay;  /**< 1 if a pure-delay block (no RF/grad/ADC): its
                            *   duration is runtime-adjustable via setperiod, so
                            *   two segments differing only in such a block's
                            *   duration share one segment definition. */
    int rf_grad_constant;   /**< 1 if RF is present and every accompanying gradient is
                            *   flat across the RF's active window -- the excitation
                            *   can then be moved at run time by a carrier offset
                            *   alone.  A nonselective pulse qualifies with a zero
                            *   level.  0 when the block carries a rotation extension,
                            *   and 0 for a block with no RF. */
    float rf_grad_level[3]; /**< normalised gradient level over that window, per axis;
                             *   multiply by the instance amplitude for the physical
                             *   gradient.  Meaningless when rf_grad_constant is 0. */
} pulseg_block_info;

/* clang-format off */
#define PULSEG_BLOCK_INFO_INIT \
    { \
    0, 0, {0, 0, 0}, {0, 0, 0}, {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}, 0, \
    -1, -1, -1, -1, 0, 0, 0, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, {0.0f, 0.0f, 0.0f} \
    }
/* clang-format on */

/* ================================================================== */
/*  ADC definition (replaces per-ADC getters)                         */
/* ================================================================== */

/**
 * @brief Information about a unique ADC definition.
 *
 * Returned by pulseg_get_adc_def().
 */
typedef struct pulseg_adc_def
{
    int dwell_ns;    /**< dwell time (ns)                     */
    int num_samples; /**< sample count                        */
} pulseg_adc_def;

/* clang-format off */
#define PULSEG_ADC_DEF_INIT {0, 0}
/* clang-format on */

/* ================================================================== */
/*  RF shim definition (parallel-transmit channel weights)            */
/* ================================================================== */

#define PULSEG_MAX_RF_SHIM_CHANNELS 64

/**
 * @brief Per-channel amplitude and phase weights for parallel transmit.
 *
 * Returned by pulseg_get_rf_shim_def().  The rf_shim_id field in
 * pulseg_block_instance is LOCAL to its subsequence (same convention as
 * rf_id, gx_id, etc.) — index 0 is the first shim of that subsequence.
 */
typedef struct pulseg_rf_shim_def
{
    int num_channels;                              /**< Tx channel count       */
    float magnitudes[PULSEG_MAX_RF_SHIM_CHANNELS]; /**< per-ch magnitude [0,1] */
    float phases[PULSEG_MAX_RF_SHIM_CHANNELS];     /**< per-ch phase (rad)     */
} pulseg_rf_shim_def;

/* clang-format off */
#define PULSEG_RF_SHIM_DEF_INIT {0, {0}, { 0 }}
/* clang-format on */

/* ================================================================== */
/*  RF event (per-occurrence identity, for pTx SAR accumulation)       */
/* ================================================================== */

/**
 * @brief Identity of one RF occurrence in the canonical TR, index-aligned
 * with pulseg_get_rf_array().
 *
 * Returned by pulseg_get_rf_event_array().
 */
typedef struct pulseg_rf_event
{
    int rf_def_id;      /**< local RF definition index within subsequence */
    float amplitude_hz; /**< per-event |amplitude| from rf_table (Hz)     */
    int rf_shim_id;     /**< local shim index, -1 if none                 */
    int num_channels;   /**< channels in the RF definition waveform (>=1) */
} pulseg_rf_event;

#endif /* PULSEG_TYPES_H */
