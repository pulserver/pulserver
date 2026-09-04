/**
 * @file pulseg_internal.h
 * @brief Private types and cross-translation-unit helpers of the pulseg core.
 *
 * NOT part of the public API: it lives under @c src/c/, outside the public
 * @c src/c/include/ tree, and is reachable only by a target that opts into the
 * private include directory. Everything declared here may change without
 * notice.
 *
 * Naming rule enforced across the library:
 *   - @c pulseg_  ... public, declared in @c src/c/include/pulseg/
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
#include "pulseg/pulseg_pns_models.h"

/* Error codes are public (pulseg_errors.h, included via pulseg_types.h). */

/* ================================================================== */
/*  Constants moved from public header (implementation details)       */
/* ================================================================== */
#define PULSEG_MAX_RF_SHIM_CHANNELS 64

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

/*
 * One representative instance of a gradient definition.
 *
 * A definition is played many times, at different amplitudes and -- for an
 * arbitrary gradient -- with different waveforms.  Enumerating them all in the
 * definition is what imposed the old 16-shot ceiling; the per-instance shape
 * id lives in the grad table (and so in the exec stream) instead, and what the
 * definition keeps is the instance safety has to look at when it cannot look
 * at the instance that actually plays.
 *
 * `spectral` maximises the second moment of
 *
 *     S(f) = |Gx(f)|^2 + |Gy(f)|^2 + |Gz(f)|^2
 *
 * -- rotation-invariant, because a rotation sends G to RG and R preserves
 * norms, which is what lets it rank instances without the rotation
 * confounding the comparison.  Parseval turns it into a time-domain integral
 * of (dg/dt)^2 that needs no FFT: PNS, mechanical resonance, SPL.
 *
 * Selection is per definition, i.e. per axis, and S is additive over axes, so
 * maximising each axis independently bounds the maximum of the sum.  That is
 * conservative, which is the direction a safety gate has to err in.
 *
 * Two honest limits.  The second moment is broadband and mechanical resonance
 * is not, so a shape with the largest total slew energy is not necessarily the
 * worst inside a narrow resonance band, and the band-resolved check still runs
 * on the materialised waveform.  And a representative pairs the steepest shape
 * with the largest amplitude, which need not be the same instance -- so where
 * the instance that plays is known, ask it directly rather than asking the
 * definition.  Gradient heating does exactly that, via `grad_shape_energy`.
 */
typedef struct pulseg_grad_representative
{
    int shape_id;      /* pulseq shape id; 0 for a trapezoid            */
    float amplitude;   /* |amplitude| of the instance that won, Hz/m    */
    float slew_rate;   /* of the normalised waveform, 1/s               */
    float energy;      /* integral of w^2 over the event, s             */
    float first_value; /* normalised waveform at the event's start      */
    float last_value;  /* ... and at its end                            */
    float score;       /* the moment this instance maximised            */
} pulseg_grad_representative;

/* clang-format off */
#define PULSEG_GRAD_REPRESENTATIVE_INIT {0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f}
/* clang-format on */

/*
 * Bounds over every instance of a definition, for the questions a
 * representative structurally cannot answer.
 *
 * A representative is the *worst* instance by some functional.  Some checks
 * are not of that shape: "is the gradient at zero across this junction" has to
 * hold for all of them, and a peak-slew ceiling has to be an upper bound, not
 * the slew of whichever instance happened to carry the most energy.  Kept as
 * one struct so the cache writes it by sizeof and no repeated field count can
 * drift out of step with it.
 */
typedef struct pulseg_grad_aggregate
{
    float max_amplitude; /* largest |amplitude| any instance reaches  */
    float min_amplitude; /* smallest                                  */
    float max_abs_first; /* largest |w[0]| over the shapes used        */
    float max_abs_last;  /* largest |w[n-1]|                          */
    float max_slew_rate; /* largest normalised slew, 1/s              */
} pulseg_grad_aggregate;

/* clang-format off */
#define PULSEG_GRAD_AGGREGATE_INIT {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
/* clang-format on */

typedef struct pulseg_grad_definition
{
    int id;
    int type;
    int rise_time_or_unused;
    int flat_time_or_unused;
    int fall_time_or_num_uncompressed_samples;
    int unused_or_time_shape_id;
    int delay;
    pulseg_grad_aggregate any;
    pulseg_grad_representative spectral;
} pulseg_grad_definition;

typedef struct pulseg_grad_table_element
{
    int id;
    int shape_id; /* pulseq shape id of THIS instance; 0 for a trapezoid */
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
/*  Frequency modulation collection (opaque from public API)          */
/* ================================================================== */

/* ================================================================== */
/*  Base blocks and block table                                      */
/* ================================================================== */

/* Compile-time claim that `type` is exactly `nwords` 4-byte members with no
 * padding, so an array of it can be serialized in one call and read back the
 * same way (see write_instances/read_instances). C89, hence a negative array
 * bound rather than a static assertion: a member of another width, or one
 * added without updating the count, fails the build instead of silently
 * changing the cache layout. */
#define PULSEG_ASSERT_PACKED(type, nwords) \
    typedef char pulseg__packed_check_##type[(sizeof(type) == (nwords)*4) ? 1 : -1]

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
    int norot_flag;
    int nopos_flag;
    int pmc_flag;
    int nav_flag;
    int rf_shim_id; /* index into rf_shim_definitions, or -1 */
    int trid;       /* sticky pulseq TRID, 0 = ungrouped (default) */
} pulseg_block_table_element;

/* NOTE: digitalout_id occupies the former trigger_id position */

/* ================================================================== */
/*  TR descriptor                                                     */
/* ================================================================== */
typedef struct pulseg_tr_descriptor
{
    int tr_size;
    int num_trs;
    float tr_duration_us;
} pulseg_tr_descriptor;

/* clang-format off */
#define PULSEG_TR_DESCRIPTOR_INIT {0, 0, 0.0f}
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
 * grad_amplitude_hz_per_m = 1.0, grad_shape_id = 0, ids = -1. */
typedef struct pulseg_block_initial_state
{
    int base_block_id;  /* base_blocks index at the representative instance, -1 */
    int digitalout_id;  /* trigger_events index at this position, -1 = none     */
    int grad_def_id[3]; /* grad_definitions index per axis, -1 = no gradient    */
    int grad_shape_id[3];
    float rf_amplitude_hz;
    float grad_amplitude_hz_per_m[3];

    /* Can this excitation be moved by a frequency offset alone?
     *
     * Set when the position has RF and every accompanying gradient is FLAT
     * across the RF's active window -- so the whole pulse sees one gradient
     * vector G, and translating what it excites by dr is exactly a carrier
     * offset of G.dr.  That is the one case where prospective motion
     * correction can move a *selective* excitation at run time, which the
     * interpreter otherwise cannot do: the RF waveform is fixed at design
     * time.  A pulse with no gradient at all qualifies with G = 0: it excites
     * everything, so the offset is zero and there is nothing to get wrong.
     *
     * AND-reduced over every instance of the position, and 0 when the block
     * carries a rotation extension -- the rotated gradient is still flat, but
     * it is no longer the vector recorded here, and a wrong G moves the slab
     * to the wrong place rather than failing.
     *
     * rf_grad_level is the NORMALISED level over that window, per axis; the
     * physical gradient is grad_amplitude_hz_per_m (per instance) times it. */
    int rf_grad_constant;
    float rf_grad_level[3];
} pulseg_block_initial_state;

/* clang-format off */
#define PULSEG_BLOCK_INITIAL_STATE_INIT \
    {-1, -1, {-1, -1, -1}, {0, 0, 0}, 0.0f, {1.0f, 1.0f, 1.0f}, 0, {0.0f, 0.0f, 0.0f}}
/* clang-format on */

/* Number of 4-byte words in pulseg_block_initial_state (cache serialization). */
#define PULSEG_BLOCK_INITIAL_STATE_WORDS 16

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
    0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, -1, 0, \
    PULSEG_SEGMENT_TIMING_INIT \
    }
/* clang-format on */

/* ================================================================== */
/*  Segment table result                                              */
/* ================================================================== */
typedef struct pulseg_segment_table_result
{
    int num_unique_segments;
    int num_main_segments;
    int *main_segment_table;
} pulseg_segment_table_result;

/* clang-format off */
#define PULSEG_SEGMENT_TABLE_RESULT_INIT {0, 0, NULL}
/* clang-format on */

/* ================================================================== */
/*  Compact execution stream                                          */
/* ================================================================== */
/* The stream pulseg__build_exec_stream emits walks the block table in
 * table order, so the block_table index advances by exactly 1.  Storing
 * (emit_start, block_start, length) per run reproduces
 * exec_stream_block_idx at a handful of ints instead of one per block.
 *
 * exec_stream_seg_id is piecewise-constant over segment instances and is
 * run-length encoded separately below. */
typedef struct pulseg_exec_run
{
    int emit_start;  /* first execution-stream position in this run     */
    int block_start; /* block_table index at emit_start                 */
    int length;      /* number of stream positions covered              */
} pulseg_exec_run;

/* ================================================================== */
/*  Sequence descriptor                                               */
/* ================================================================== */
typedef struct pulseg_sequence_descriptor
{
    float rf_raster_us;
    float grad_raster_us;
    float adc_raster_us;
    float block_raster_us;
    int enable_pmc;
    int num_gain_cal_readouts; /**< calibration readouts for APS2 receive gain (pislquant) */
    int vendor;                /**< PULSEG_VENDOR_* runtime constant */
    int label_column_map[3];   /**< copy of pulseg_opts.label_column_map at dedup time */

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

    /** 1 when positional RF amplitude differs across canonical TR instances
     *  and pulseg_opts.allow_variable_rf_amplitude accepted it. The RF safety
     *  model is then built from the positional-max envelope
     *  (pulseg__rf_position_max) instead of one canonical instance, so it
     *  dominates every instance in peak and in time-averaged terms alike.
     *  Serialized in the cache COMMON section. */
    int rf_amplitude_variable;

    int num_unique_grads;
    pulseg_grad_definition *grad_definitions;
    int grad_table_size;
    pulseg_grad_table_element *grad_table;

    /* Per-SHAPE statistics of the normalised waveform, indexed by pulseq
     * shape id - 1; `num_grad_shape_stats` is the shape library's size.
     * `grad_shape_slew` is the steepest normalised slew the shape reaches,
     * in 1/s, and `grad_shape_energy` the integral of its square over the
     * event, in s -- both over the shape's own time shape where it has one.
     *
     * These live here rather than on the definition because they are
     * properties of the shape alone, and because the questions they answer
     * are per instance rather than worst case: "is the gradient at zero
     * across this junction", "how steep is what plays here" and "how much
     * energy does this instance carry" all have to be asked of the instance
     * that actually plays there.  Grad definitions are deduplicated without
     * the magnitude shape id, so one definition covers every shape of equal
     * sample count, and its representative pairs the steepest shape with the
     * largest amplitude -- which need not be the same instance.
     *
     * A trapezoid has no shape id and no entry; its endpoints are zero by
     * construction, and its slew and energy both follow from its ramp times
     * (pulseg__trap_energy). */
    int num_grad_shape_stats;
    float *grad_shape_first;
    float *grad_shape_last;
    float *grad_shape_slew;
    float *grad_shape_energy;

    int num_unique_adcs;
    pulseg_adc_definition *adc_definitions;
    int adc_table_size;
    pulseg_adc_table_element *adc_table;

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

    /* Scan table (expanded playback order), stored as parallel arrays of
     * length exec_stream_len. */
    int exec_stream_len;
    int *exec_stream_block_idx; /* [exec_stream_len] index into block_table */
    int *exec_stream_seg_id;    /* [exec_stream_len] segment id             */
    int *exec_stream_tr_start;  /* [exec_stream_len] 1 at the first block of each TR */

    /* Compact form of the arrays above. Built by
     * pulseg__build_exec_stream alongside them; pulseg__verify_exec_runs
     * checks the two agree position by position. */
    int num_exec_runs;
    pulseg_exec_run *exec_runs;

    /* Run-length encoding of exec_stream_seg_id over ONE period. Run r
     * covers reduced positions [seg_run_start[r], seg_run_start[r+1]) with
     * value seg_run_id[r]; seg_run_start[num_seg_runs] == seg_period.
     *
     * seg_period is chosen by MEASUREMENT, never assumed: candidate periods
     * are tested with seg_id[n] == seg_id[n % P] over the whole stream and
     * the smallest that verifies wins, falling back to seg_period ==
     * exec_stream_len (plain RLE, no periodicity claimed) when none does.
     * A sequence whose segmentation genuinely does not repeat therefore
     * still encodes correctly, just without the extra factor. */
    int num_seg_runs;
    int seg_period;
    int *seg_run_start;
    int *seg_run_id;

    /* Sequential-access hints. scancore walks the stream in order, so a
     * remembered run index makes the accessors O(1) amortized (check the
     * cached run, then its successor) and keeps the binary search purely as
     * a random-access fallback. Pure caches: they never change the value
     * returned, only the cost of finding it. */
    int exec_run_hint;
    int seg_run_hint;

    /* Cached TR anchor for pulseg__exec_tr_start: the stream position of
     * the first TR block (-1 when no TR was detected). */
    int tr_start_first;

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

    /* Copy of pulseg_opts.cache_ext at dedup time. Not part of the
     * cache payload itself (it only names the cache FILE, not its
     * contents) -- deliberately placed after all cache-serialized fields
     * so it needs no swap4_array count bump. */
    char cache_ext[PULSEG_CACHE_EXT_MAX];
    /* Converted for structure alone: gradient shapes may hold no samples
     * and no gradient statistics were computed, so waveform and safety
     * requests are refused. Not part of the cache payload. */
    int structure_only;
    /* The shape samples belong to a buffer the caller keeps alive; freeing
     * the descriptor leaves them alone. Not part of the cache payload. */
    int shapes_borrowed;
    /* Where a running check keeps the logical rotation table and block ids
     * while the prescription frame is installed (pulseg_prescription.c). */
    float (*rotation_matrices_logical)[9];
    int num_rotations_logical;
    int *rotation_id_logical;
    int prescription_depth;
} pulseg_sequence_descriptor;

/* Identity of the waveform one block drives one axis with, up to amplitude.
 *
 * def_id fixes the timing skeleton (delay, rise/flat/fall or sample count,
 * time shape), shape_id fixes the samples, and dur_us fixes the span a held
 * type runs to and therefore the slice length. Two blocks agreeing on all
 * three drive that axis with the same waveform scaled by their amplitudes.
 *
 * axis and rotation_id participate only for consumers working in the
 * physical frame, where a rotation mixes the logical axes and one shape at
 * two orientations is no longer a scalar multiple of itself. The Irnich
 * stimulation path works in the logical frame -- its verdict is a
 * root-sum-square under one kernel, which an orthogonal rotation leaves
 * alone -- and leaves both at -1.
 */
typedef struct pulseg__wave_key
{
    int def_id;
    int shape_id;
    int dur_us;
    int axis;        /* -1 in the logical frame */
    int rotation_id; /* -1 in the logical frame */
} pulseg__wave_key;

/* Fill @p out for the gradient @p bte drives @p axis with. Returns 0 when the
 * block is silent on that axis, in which case @p out is left cleared. */
int pulseg__wave_key_axis(
    const pulseg_sequence_descriptor *desc,
    const pulseg_block_table_element *bte,
    int axis,
    int physical_frame,
    pulseg__wave_key *out);

int pulseg__wave_key_equal(const pulseg__wave_key *a, const pulseg__wave_key *b);

/* The same identity folded into one int, for callers holding an int-keyed
 * cache. Definitions deduplicate on timing and sample count, so a
 * materialised multishot readout plays many shapes under one definition id --
 * folding the shape in is what keeps every arm from being handed the first
 * arm's answer. */
int pulseg__wave_key_flat(const pulseg_sequence_descriptor *desc, int def_index, int shape_id);

/* clang-format off */
#define PULSEG_SEQUENCE_DESCRIPTOR_INIT \
    { \
    0.0f, 0.0f, 0.0f, 0.0f, 0, 0, 0, {0, 1, 2}, {0, 0, 0}, {0, 0, \
    0}, {0, 0, 0}, {0, 0, 0}, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, /* rf_amplitude_variable */ \
    0, NULL, 0, NULL, /* grad shape stats */ 0, NULL, NULL, NULL, NULL, \
    0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, \
    PULSEG_TR_DESCRIPTOR_INIT, 0, NULL, PULSEG_SEGMENT_TABLE_RESULT_INIT, 0, NULL, NULL, \
    NULL, /* exec_runs */ 0, NULL, /* seg runs */ 0, 0, NULL, NULL, \
    /* hints */ 0, 0, /* tr_start anchor */ -1, NULL, 0, 0, NULL, \
    {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, \
    {0, 0}, {0, 0}, {0, 0}, {0, 0}}, NULL, \
    PULSEG_CACHE_EXT_DEFAULT, 0, 0, \
    /* prescription frame */ NULL, 0, NULL, 0 \
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

/* Native gradient corner points, before any resampling.  Each axis keeps
 * its own time base: a trapezoid contributes its ramp and plateau corners,
 * an arbitrary waveform its raster samples. */
typedef struct pulseg__grad_corner_arrays
{
    int num_gx, num_gy, num_gz;
    float *time_gx, *time_gy, *time_gz; /* us, per-axis time base */
    float *wf_gx, *wf_gy, *wf_gz;       /* Hz / m */
} pulseg__grad_corner_arrays;

/* clang-format off */
#define PULSEG__GRAD_CORNER_ARRAYS_INIT {0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL}
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

/* --- pulseg_cache.c: binary-cache IO primitives.
 * All cache words are 4 bytes; swap on an endianness mismatch. --- */
void pulseg__swap4(void *p);
void pulseg__swap4_array(void *p, int count);
int pulseg__write4(FILE *f, const void *p, int count);
int pulseg__read4(FILE *f, void *p, int count);
/* Derive a cache path from a .seq path by swapping the extension (ext
 * includes the dot; NULL/empty -> PULSEG_CACHE_EXT_DEFAULT). Caller frees. */
char *pulseg__make_cache_path(const char *seq_path, const char *ext);

/* --- Shared block-definition predicates (pulseg_structure.c) --- */
int pulseg__block_def_is_pure_delay(const pulseg_base_block *b);

/* Worst-case gradient magnitude at the start / end of the instance held in
 * grad table row @p raw_id, in Hz/m.
 *
 * Exact on the shape -- the row names the shape the instance actually plays --
 * and worst case on the amplitude, since the definition's largest amplitude
 * bounds every instance that shares the shape.  Used by the zero-gradient
 * junction tests, where the question is whether anything can be live across a
 * boundary, so erring high is the safe direction.  A trapezoid has no shape
 * and returns 0: it starts and ends at zero by construction. */
float pulseg__grad_boundary_first(const pulseg_sequence_descriptor *desc, int raw_id);
float pulseg__grad_boundary_last(const pulseg_sequence_descriptor *desc, int raw_id);

/* Signed endpoint values of the NORMALISED waveform of pulseq shape @p
 * shape_id, or 0 when there is no such shape (a trapezoid, notably, which
 * starts and ends at zero by construction).  Multiply by an instance's own
 * amplitude for the value that instance actually plays. */
float pulseg__grad_shape_first(const pulseg_sequence_descriptor *desc, int shape_id);
float pulseg__grad_shape_last(const pulseg_sequence_descriptor *desc, int shape_id);

/* Steepest slew of the NORMALISED waveform of pulseq shape @p shape_id, in
 * 1/s, or 0 when there is no such shape.  Multiply by an instance's own
 * amplitude for the slew that instance actually plays. */
float pulseg__grad_shape_slew(const pulseg_sequence_descriptor *desc, int shape_id);

/**
 * @brief Integral of w^2 over one gradient instance, in s.
 *
 * @param desc      Descriptor holding the per-shape statistics.
 * @param gd        Definition the instance belongs to (supplies the ramp
 *                  times when @p shape_id is 0).
 * @param shape_id  Pulseq shape id of THIS instance; 0 for a trapezoid.
 * @return The normalised waveform's energy; 0 if it cannot be determined.
 */
float pulseg__grad_instance_energy(
    const pulseg_sequence_descriptor *desc,
    const pulseg_grad_definition *gd,
    int shape_id);

/**
 * @brief Integral of w^2 over a normalised trapezoid, in s.
 *
 * @param rise_us  Ramp-up duration (us).
 * @param flat_us  Plateau duration (us).
 * @param fall_us  Ramp-down duration (us).
 * @return rise/3 + flat + fall/3, converted to seconds.
 */
float pulseg__trap_energy(float rise_us, float flat_us, float fall_us);
int pulseg__block_defs_structurally_equal(
    const pulseg_sequence_descriptor *desc,
    int id_a,
    int id_b);

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

/* Double-precision complex FFT: the vendored kissfft compiled a second time
 * (src/vendor/external_kiss_fft_double.c), because the copy the display path
 * uses is float and these sums are carried in double on purpose. Buffers are
 * interleaved (re, im) doubles; `cfg` is opaque. Sizes need not be powers of
 * two -- ask pulseg__fft_double_size for the cheapest one that fits. */
int pulseg__fft_double_size(int at_least);
void *pulseg__fft_double_alloc(int nfft, int inverse);
void pulseg__fft_double_run(void *cfg, const double *in_interleaved, double *out_interleaved);
void pulseg__fft_double_free(void *cfg);

/* Chirp-z: P_j = sum_{k<n} a[k] * e^{i k (theta0 + j*dtheta)} for j < m.
 *
 * Evaluates a real-coefficient polynomial at `m` points spaced evenly in
 * angle on the unit circle, in O((n+m) log(n+m)) rather than O(n*m). Used
 * where one waveform definition has to be transformed at many frequencies
 * that happen to form an arithmetic progression. */
int pulseg__czt_unit(
    double *out_re,
    double *out_im,
    const float *a,
    int n,
    double theta0,
    double dtheta,
    int m);

/* The same transform with the geometry held, for a caller evaluating several
 * `theta0` over one (n, m, dtheta) -- which is every caller that walks a
 * frequency comb, since the offsets within it share a spacing. Holding the
 * plan keeps the chirp, the transformed kernel and the FFT setup, leaving
 * two transforms per apply. */
typedef struct pulseg__czt_plan pulseg__czt_plan;

int pulseg__czt_plan_create(pulseg__czt_plan **out_plan, int n, int m, double dtheta);
int pulseg__czt_plan_apply(
    pulseg__czt_plan *plan,
    double *out_re,
    double *out_im,
    const float *a,
    double theta0);
void pulseg__czt_plan_free(pulseg__czt_plan *plan);
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

/* pulseg_conv_fft_plan is public (pulseg_pns_models.h): a vendor-supplied
 * PNS model is a caller-side implementation and needs the same convolution
 * the published models use. pulseg__calc_convolution_fft() is
 * create/apply/free around it and is numerically identical. */

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
/* @p adopt_shapes, when it is @p seq's own shapes_library, lets the
 * descriptor take the sample buffers instead of copying them; the file's
 * entries are left empty and must not be read afterwards. NULL copies. */
int pulseg__get_unique_blocks(
    pulseg_sequence_descriptor *desc,
    const pulseq_file *seq,
    const pulseg_opts *opts,
    pulseq_shape *adopt_shapes);

/* --- pulseg_structure.c --- */
int pulseg__get_tr_in_sequence(
    pulseg_sequence_descriptor *desc,
    const pulseq_file *seq,
    pulseg_diagnostic *diag);
int pulseg__build_exec_stream(pulseg_sequence_descriptor *desc, pulseg_diagnostic *diag);
int pulseg__get_exec_stream_segments(
    pulseg_sequence_descriptor *desc,
    pulseg_diagnostic *diag,
    const pulseg_opts *opts);
void pulseg__compute_exec_stream_tr_start(pulseg_sequence_descriptor *desc);

/* Compact execution stream: O(log num_exec_runs) equivalents of the
 * exec_stream_* arrays (see pulseg_exec_run). */
int pulseg__exec_block_idx(const pulseg_sequence_descriptor *desc, int n);
int pulseg__exec_seg_id(const pulseg_sequence_descriptor *desc, int n);
int pulseg__exec_tr_start(const pulseg_sequence_descriptor *desc, int n);
int pulseg__build_seg_runs(pulseg_sequence_descriptor *desc);
long pulseg__verify_exec_runs(const pulseg_sequence_descriptor *desc);
void pulseg__free_exec_stream_scratch(pulseg_sequence_descriptor *desc);
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

/* Label the TR instances by the gradient *definitions* they play, so that a
 * caller can evaluate one canonical window per group. Instances in one group
 * differ only in amplitude, which the per-position maximum bounds; instances
 * in different groups play different shapes, which it does not.
 *
 * @p out_labels is [num_trs] and @p out_first is [*out_num_groups], the first
 * instance of each group; both are NULL (and *out_num_groups is 1) when every
 * instance falls in one group, which is the plain whole-scan envelope. The
 * caller frees both. Returns PULSEG_ERR_INVALID_ARGUMENT when the sequence
 * has more groups than @p max_groups. */
#define PULSEG__MAX_SHAPE_GROUPS 64

int pulseg__group_tr_instances_by_shape(
    const pulseg_sequence_descriptor *desc,
    int **out_labels,
    int **out_first,
    int *out_num_groups,
    int max_groups);

/* Free uniform waveforms. */
void pulseg__uniform_grad_waveforms_free(pulseg__uniform_grad_waveforms *w);

/* Per-block gradient rendering primitives. Exposed (rather than static)
 * so the PNS template builder in pulseg_pns_memo.c renders a definition's
 * unit-amplitude waveform through the very same code the full-window
 * extraction uses, and the two cannot drift apart. */
int pulseg__count_grad_samples_for_block(
    const pulseg_sequence_descriptor *desc,
    const pulseg_grad_definition *gdef,
    float block_duration_us);
int pulseg__fill_grad_waveform_for_block(
    const pulseg_sequence_descriptor *desc,
    float *time,
    float *waveform,
    int start_idx,
    const pulseg_grad_definition *gdef,
    const pulseg_grad_table_element *gte,
    float t0,
    const float *pos_max_amp,
    float block_duration_us);
/* Resample a (time, amplitude) point list onto a uniform grid in place;
 * reallocates *time / *waveform and updates *num_samples. */
int pulseg__interpolate_to_uniform(
    float **time,
    float **waveform,
    int *num_samples,
    float target_raster_us);

/* --- pulseg_prescription.c --- */
/* Install opts->prescription_rotation on every descriptor of coll for the
 * duration of a check (no-op without one, or for the identity; nested calls
 * count) and hand the logical tables back. Every public check entry wraps
 * its body in this pair. */
int pulseg__prescription_enter(pulseg_collection *coll, const pulseg_opts *opts);
void pulseg__prescription_leave(pulseg_collection *coll, const pulseg_opts *opts);

/* --- pulseg_pns_exact.c --- */

/* The shape of pulseg_opts.parallel_for_fn, for the loops that take it. */
typedef void (*pulseg__parallel_for_fn)(
    void *ctx,
    int count,
    void (*body)(void *arg, int begin, int end),
    void *arg);

/* The hook a check runs under: the caller's, else the library's own when it
 * was built with PULSEG_HAVE_PTHREADS, else NULL (a sequential loop). */
pulseg__parallel_for_fn pulseg__parallel_for_default(void);
pulseg__parallel_for_fn pulseg__opts_par_fn(const pulseg_opts *opts);
void *pulseg__opts_par_ctx(const pulseg_opts *opts);

/* The stimulation check of a scan past the shape-group cap: every block's
 * exact response placed on the scan's timeline, rotated into the physical
 * frame, root-sum-squared, at its peak. out_peak is in percent of the
 * model's threshold; out_peak_block the block the peak falls in. A model
 * with a kernel runs by FFT under the parallel hook; one without runs its
 * own evaluator over rendered chunks, sequentially. */
int pulseg__pns_exact_scan_peak(
    pulseg_check_plan *plan,
    pulseg_diagnostic *diag,
    const pulseg_sequence_descriptor *desc,
    int subseq_idx,
    const pulseg_pns_model *model,
    pulseg__parallel_for_fn par_fn,
    void *par_ctx,
    float gamma,
    double *out_peak,
    int *out_peak_block);

/* The same, judged: PULSEG_ERR_PNS_THRESHOLD_EXCEEDED with the block named
 * in the diagnostic when the peak is over threshold_percent. */
int pulseg__pns_exact_scan_check(
    pulseg_check_plan *plan,
    pulseg_diagnostic *diag,
    const pulseg_sequence_descriptor *desc,
    int subseq_idx,
    const pulseg_pns_model *model,
    pulseg__parallel_for_fn par_fn,
    void *par_ctx,
    float gamma,
    float threshold_percent,
    double *out_peak,
    int *out_peak_block);

/* Exact response peak of blocks [start, start + count), rendered and run
 * through the model's own evaluator, judged from judge_from_us into the
 * range up to judge_until_us (0 = to the end). Test-facing, and the route a
 * model without a kernel takes. */
int pulseg__pns_exact_range_peak(
    pulseg_check_plan *plan,
    pulseg_diagnostic *diag,
    const pulseg_sequence_descriptor *desc,
    int subseq_idx,
    int start,
    int count,
    double judge_from_us,
    double judge_until_us,
    const pulseg_pns_model *model,
    float gamma,
    double *out_peak,
    double *out_peak_time_us);

/* A uniform frequency grid: count points from f0_hz, df_hz apart. */
typedef struct
{
    double f0_hz;
    double df_hz;
    int count;
} pulseg__mech_scan_grid;

/* Probe flag: every event's transform evaluated directly, never from its
 * FFT record. */
#define PULSEG__MECH_SCAN_DIRECT 1

/* The sustained amplitude per axis at every point of the grids, over the
 * whole scan, each grid's window window_us[g] wide (NULL or 0 = the axis's
 * longest event; a reading below one period of the window is zero), no grid
 * guard; out_span_max_us carries one span per grid:
 * the quantity the past-the-cap mechanical-resonance check judges, at the
 * frequencies given. Outputs run grid by grid. Test-facing. */
int pulseg__mech_scan_window_probe(
    const pulseg_sequence_descriptor *desc,
    const pulseg__mech_scan_grid *grids,
    int num_grids,
    const double *window_us,
    int flags,
    pulseg__parallel_for_fn par_fn,
    void *par_ctx,
    float *out_amp_gx,
    float *out_amp_gy,
    float *out_amp_gz,
    double *out_span_max_us);

/* The scan window's interpolation kernel at x bins from a sample, and the
 * bound on what truncating it to its half-width costs per unit of a
 * waveform's gradient L1. Test-facing. */
double pulseg__mech_scan_kernel(double x);
double pulseg__mech_scan_kernel_e(void);

/* --- pulseg_pns_memo.c --- */

/* Assemble a linear PNS model's response from one convolution per distinct
 * gradient shape instead of one transform over the whole canonical window.
 * Writes n samples per axis and sets *applied to 1 on success. Sets
 * *applied to 0 (returning PULSEG_SUCCESS) when the decomposition does not
 * apply -- an off-grid block boundary, too little repetition, a window too
 * short to be worth it -- and the caller must then run the exact path. */
int pulseg__calc_pns_memoized(
    float *out_x,
    float *out_y,
    float *out_z,
    int n,
    int *applied,
    const pulseg_sequence_descriptor *desc,
    int block_start,
    int block_count,
    const int *block_order,
    const pulseg__uniform_grad_waveforms *uw,
    float gamma_hz_per_tesla,
    const float *kernel,
    int kernel_len,
    float out_scale,
    int pad,
    double *out_macs);

/* Extract gradient waveforms for an arbitrary block range,
 * interpolated to uniform raster (half gradient raster). */
int pulseg__collect_grad_corners_range(
    const pulseg_sequence_descriptor *desc,
    pulseg__grad_corner_arrays *out,
    pulseg_diagnostic *diag,
    int block_start,
    int block_count,
    int amplitude_mode,
    const int *tr_group_labels,
    int target_group,
    const int *block_order);

void pulseg__grad_corner_arrays_free(pulseg__grad_corner_arrays *c);

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

/* --- pulseg_check_plan.c ---
 * The shared preprocessing behind pulseg_check_plan. Both accessors take a
 * non-NULL plan; the public checks create a private one when the caller
 * passes none, so no path below ever sees NULL. */

/* Repetitions grouped by the set of gradient shapes they play. The returned
 * arrays are the plan's and stay valid for its lifetime. */
int pulseg__plan_shape_groups(
    pulseg_check_plan *plan,
    const int **out_labels,
    const int **out_group_first,
    int *out_num_groups,
    pulseg_diagnostic *diag,
    const pulseg_sequence_descriptor *desc,
    int subseq_idx);

/* Uniform-raster gradient waveforms over one window, extracted on first
 * request and retained. The result is borrowed and must not be freed; it
 * stays valid until the next pulseg__plan_waveforms() call on the same plan,
 * which may evict it. */
int pulseg__plan_waveforms(
    pulseg_check_plan *plan,
    const pulseg__uniform_grad_waveforms **out,
    pulseg_diagnostic *diag,
    const pulseg_sequence_descriptor *desc,
    int subseq_idx,
    int block_start,
    int block_count,
    int amplitude_mode,
    const int *labels,
    int target_group);

/* --- pulseg_safety.c ---
 * Optional internal observer used by the documentation benchmark.  Keeping
 * the clock outside libpulseg means the production build remains free of OS
 * timing APIs while the benchmark can measure the real headless
 * pulseg_check_safety path rather than composing plotting-oriented calls. */
enum pulseg__safety_profile_stage
{
    PULSEG__SAFETY_PROFILE_GRAD_PRESENCE = 0,
    PULSEG__SAFETY_PROFILE_MAX_GRAD,
    PULSEG__SAFETY_PROFILE_CONTINUITY,
    PULSEG__SAFETY_PROFILE_MAX_SLEW,
    PULSEG__SAFETY_PROFILE_WAVEFORM_EXTRACT,
    PULSEG__SAFETY_PROFILE_MECH_RESONANCE,
    PULSEG__SAFETY_PROFILE_PNS,
    PULSEG__SAFETY_PROFILE_PNS_BASIS_BUILD,
    PULSEG__SAFETY_PROFILE_PNS_SCORE,
    PULSEG__SAFETY_PROFILE_STAGE_COUNT
};

typedef void (
    *pulseg__safety_profile_fn)(void *ctx, enum pulseg__safety_profile_stage stage, int entering);

/* Arithmetic a stage issued, in multiply-accumulates, reported when it leaves.
 * Counted by the code doing the work rather than derived from the shapes it
 * ran over, so a throughput figure divides one measured clock by one measured
 * operation count. A stage that cannot count itself reports nothing, which is
 * distinguishable from reporting zero. */
typedef void (
    *pulseg__safety_work_fn)(void *ctx, enum pulseg__safety_profile_stage stage, double macs);

int pulseg__check_safety_profiled(
    pulseg_collection *coll,
    pulseg_diagnostic *diag,
    pulseg_check_plan *plan,
    const pulseg_opts *opts,
    const pulseg_forbidden_band_list *bands,
    const pulseg_pns_model *pns_model,
    float pns_threshold_percent,
    pulseg__safety_profile_fn profile_fn,
    pulseg__safety_work_fn work_fn,
    void *profile_ctx);

/* --- pulseg_cache.c --- */
int pulseg__try_read_cache(pulseg_collection *coll, const char *seq_path, const char *cache_ext);
int pulseg__write_cache(pulseg_collection *seq_coll, const char *seq_path, const pulseg_opts *opts);
int pulseg__load_geninstructions_cache_ext(
    pulseg_collection **out_coll,
    const char *seq_path,
    const char *cache_ext);
int pulseg__load_scanloop_cache_ext(
    pulseg_collection **out_coll,
    const char *seq_path,
    const char *cache_ext);

/* --- Helper to locate segment/block in collection --- */

/* Build (or rebuild) the cross-subsequence segment deduplication remap:
 * populates coll->seg_local_to_global / seg_repr_subseq / seg_repr_local and
 * collapses coll->total_unique_segments to the deduplicated count.  Idempotent;
 * must be called after all descriptors are assembled (post-convert and
 * post-cache-load).  Returns PULSEG_SUCCESS or an error code (on error the
 * collection is left with the identity map so it stays usable).            */
int pulseg__build_segment_remap(pulseg_collection *coll);

#endif /* PULSEG_INTERNAL_H */
