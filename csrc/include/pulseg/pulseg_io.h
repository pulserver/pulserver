/**
 * @file pulseg_io.h
 * @brief Pulseq .seq reader and scan-time-peek entry points.
 *
 * Split out of the former pulseg_methods.h (Stage 1 layout normalization).
 * All functions use the pulseg_ prefix and are declared extern "C" when
 * compiled with a C++ compiler.
 */

#ifndef PULSEG_IO_H
#define PULSEG_IO_H

#include <stdio.h>

#include "pulseg_config.h"
#include "pulseg_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /* ================================================================== */
    /*  Length / size constants for the raw pulseq model                  */
    /*  (Stage 3: promoted from pulseg_internal.h)                        */
    /* ================================================================== */
#define PULSEG_PULSEQ_DEFINITION_NAME_LENGTH 32
#define PULSEG_PULSEQ_EXT_NAME_LENGTH        32
#define PULSEG_PULSEQ_LABEL_NAME_LENGTH      32
#define PULSEG_PULSEQ_SEQUENCE_NAME_LENGTH   256
#define PULSEG_PULSEQ_SEQUENCE_FILENAME_LENGTH 256
#define PULSEG_PULSEQ_SOFT_DELAY_HINT_LENGTH 32
#define PULSEG_PULSEQ_MAX_EXTENSIONS_PER_BLOCK 64
#define PULSEG_PULSEQ_MAX_LINE_LENGTH        256
#define PULSEG_PULSEQ_MAX_SCALE_SIZE         16
#define PULSEG_PULSEQ_MAX_RF_SHIM_CHANNELS   64

    /* ================================================================== */
    /*  Extension type codes for pulseg_pulseq_raw_block's extension chain */
    /*  and any raw extensions_library row written by an external raw-    */
    /*  model producer (e.g. the ExternalSequence adapter, cxx/pulseq_adapter). */
    /*  These values are load-bearing: pulseg_pulseq_get_raw_block_content_ids() */
    /*  and pulseg_pulseq_get_raw_extension() switch on them internally.  */
    /* ================================================================== */
#define PULSEG_PULSEQ_EXT_LIST      0
#define PULSEG_PULSEQ_EXT_TRIGGER   1
#define PULSEG_PULSEQ_EXT_ROTATION  2
#define PULSEG_PULSEQ_EXT_LABELSET  3
#define PULSEG_PULSEQ_EXT_LABELINC  4
#define PULSEG_PULSEQ_EXT_RF_SHIM   5
#define PULSEG_PULSEQ_EXT_DELAY     6
#define PULSEG_PULSEQ_EXT_UNKNOWN   7

    /* ================================================================== */
    /*  Raw event types (one raw pulseq library entry, pre-dedup)          */
    /* ================================================================== */
    typedef struct pulseg__shape_trap
    {
        long rise_time;
        long flat_time;
        long fall_time;
    } pulseg__shape_trap;

    typedef struct pulseg__rf_event
    {
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

    typedef struct pulseg__grad_event
    {
        short type;
        float amplitude;
        int delay;
        pulseg__shape_trap trap;
        pulseg_shape_arbitrary wave_shape;
        pulseg_shape_arbitrary time_shape;
        float first;
        float last;
    } pulseg__grad_event;

    typedef struct pulseg__adc_event
    {
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

    typedef struct pulseg__rotation_event
    {
        short type;
        union
        {
            float rot_quaternion[4];
            float rot_matrix[9];
        } data;
    } pulseg__rotation_event;

    typedef struct pulseg__label_event
    {
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

    typedef struct pulseg__flag_event
    {
        int trid;
        int module_id;  /* MODULE: sticky int id, not boolean -- same non-flag
                          * treatment as trid above; both live here only
                          * because LABELMAP_COUNTER's LABELINC-increment
                          * semantics don't apply to either. */
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

    typedef struct pulseg__soft_delay_event
    {
        short type;
        int num_id;
        int offset;
        int factor;
        int hint_id;
    } pulseg__soft_delay_event;

    typedef struct pulseg__rf_shimming_event
    {
        short type;
        int num_channels;
        float *amplitudes;
        float *phases;
    } pulseg__rf_shimming_event;

    /**
     * @brief Digital trigger / physio event (raw pulseq TRIGGERS library
     * entry, and per-block resolved trigger).
     */
    typedef struct pulseg_trigger_event
    {
        short type;
        long duration;
        long delay;
        int trigger_type;
        int trigger_channel;
    } pulseg_trigger_event;

#define PULSEG_TRIGGER_EVENT_INIT {0, 0L, 0L, 0, 0}

    /* ================================================================== */
    /*  Raw block (unresolved content ids, straight off the BLOCKS table) */
    /* ================================================================== */
    typedef struct pulseg_pulseq_raw_block
    {
        int block_duration;
        int rf;
        int gx;
        int gy;
        int gz;
        int adc;
        int ext_count;
        int ext[PULSEG_PULSEQ_MAX_EXTENSIONS_PER_BLOCK][2];
    } pulseg_pulseq_raw_block;

    /**
     * @brief Resolved extension chain for one block (labels/flags/rotation/
     * shim/trigger/soft-delay indices; rotation/shim/trigger/soft-delay are
     * indices into the owning pulseg_pulseq_file's libraries, -1 if absent).
     */
    typedef struct pulseg_pulseq_raw_extension
    {
        pulseg__label_event labelset;
        pulseg__label_event labelinc;
        pulseg__flag_event flag;
        int rotation_index;
        int rf_shim_index;
        int trigger_index;
        int soft_delay_index;
    } pulseg_pulseq_raw_extension;

    /**
     * @brief Fully-resolved single block: every referenced event inlined.
     * Populated by pulseg_pulseq_get_block().
     */
    typedef struct pulseg_pulseq_block
    {
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
    } pulseg_pulseq_block;

    /* ================================================================== */
    /*  Raw pulseq file (opaque-by-convention; fields are the completed,   */
    /*  fully-parsed raw model -- no half-parsed / file-handle state is    */
    /*  ever exposed once pulseg_pulseq_file_read*() returns success)      */
    /* ================================================================== */

    /** @brief Parse-cursor byte offsets for each [SECTION] (internal bookkeeping). */
    typedef struct pulseg__section_offsets
    {
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

    /** @brief One generic [DEFINITIONS] key/value entry. */
    typedef struct pulseg__definition
    {
        char name[PULSEG_PULSEQ_DEFINITION_NAME_LENGTH];
        int value_size;
        char **value;
    } pulseg__definition;

    /** @brief Reserved (well-known) [DEFINITIONS] keys, parsed eagerly. */
    typedef struct pulseg__reserved_definitions
    {
        float gradient_raster_time;
        float radiofrequency_raster_time;
        float adc_raster_time;
        float block_duration_raster;
        char name[PULSEG_PULSEQ_SEQUENCE_NAME_LENGTH];
        float fov[3];
        float matrix[3];
        float nav_fov[3];
        float nav_matrix[3];
        float total_duration;
        char next_sequence[PULSEG_PULSEQ_SEQUENCE_FILENAME_LENGTH];
        int ignore_fov_shift;
        int enable_pmc;
        int ignore_averages;
        int num_gain_cal_readouts; /**< calibration readouts for APS2 receive gain (pislquant) */
    } pulseg__reserved_definitions;

    /** @brief One RF_SHIMS library entry (parallel-transmit channel weights). */
    typedef struct pulseg__rf_shim_entry
    {
        int num_channels;
        float values[2 * PULSEG_PULSEQ_MAX_RF_SHIM_CHANNELS];
    } pulseg__rf_shim_entry;

    /**
     * @brief A fully-parsed Pulseq .seq file: every library section decoded
     * into flat arrays, ready for conversion (pulseg_convert_collection()).
     *
     * Created by pulseg_pulseq_file_read() / pulseg_pulseq_file_read_from_buffer().
     * Freed by pulseg_pulseq_file_free().
     */
    typedef struct pulseg_pulseq_file
    {
        pulseg_opts opts;
        char *file_path;
        pulseg__section_offsets offsets;
        int is_version_parsed;
        int version_combined;
        int version_major;
        int version_minor;
        int version_revision;
        int is_definitions_library_parsed;
        int num_definitions;
        pulseg__definition *definitions_library;
        pulseg__reserved_definitions reserved_definitions_library;
        int is_block_library_parsed;
        int num_blocks;
        float (*block_library)[7];
        int *block_ids;
        int is_rf_library_parsed;
        int rf_library_size;
        float (*rf_library)[10];
        int *rf_use_tags; /* per-RF-event use tag (parsed from .seq) */
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
        pulseg__rf_shim_entry *rf_shim_library;
        int extension_map[8];
        int extension_lut_size;
        int *extension_lut;
        int is_shapes_library_parsed;
        int shapes_library_size;
        pulseg_shape_arbitrary *shapes_library;
    } pulseg_pulseq_file;

    /**
     * @brief A chain of pulseq files (Pulseq "next sequence" chaining).
     *
     * Created by pulseg_pulseq_file_set_read(). Freed by
     * pulseg_pulseq_file_set_free().
     */
    typedef struct pulseg_pulseq_file_set
    {
        int num_sequences;
        pulseg_pulseq_file *sequences;
        char *base_path;
    } pulseg_pulseq_file_set;

    /* ================================================================== */
    /*  Raw pulseq reader / accessors (Stage 3: promoted from             */
    /*  csrc/src/io/pulseg_parse.c's former pulseg__ internal API)        */
    /* ================================================================== */

    /** @brief Zero-initialize a pulseg_pulseq_file before reading into it. */
    void pulseg_pulseq_file_init(pulseg_pulseq_file *seq, const pulseg_opts *opts);

    /** @brief Free all heap allocations inside a pulseg_pulseq_file. */
    void pulseg_pulseq_file_free(pulseg_pulseq_file *seq);

    /** @brief Free every file in a pulseg_pulseq_file_set (and the array itself). */
    void pulseg_pulseq_file_set_free(pulseg_pulseq_file_set *coll);

    /** @brief Zero-initialize a pulseg_pulseq_block before pulseg_pulseq_get_block(). */
    void pulseg_pulseq_block_init(pulseg_pulseq_block *block);

    /** @brief Free all heap allocations inside a pulseg_pulseq_block. */
    void pulseg_pulseq_block_free(pulseg_pulseq_block *block);

    /**
     * @brief Parse a .seq file from disk into a raw pulseq model.
     * @param seq        Caller-allocated, must have been pulseg_pulseq_file_init'd.
     * @param file_path  Path to the .seq file.
     * @return PULSEG_SUCCESS on success, negative error code on failure.
     */
    int pulseg_pulseq_file_read(pulseg_pulseq_file *seq, const char *file_path);

    /**
     * @brief Parse a .seq file already open on a FILE* (e.g. a tmpfile()
     * populated from an in-memory buffer) into a raw pulseq model.
     */
    int pulseg_pulseq_file_read_from_buffer(pulseg_pulseq_file *seq, FILE *f);

    /**
     * @brief Parse a (possibly chained via "next sequence") .seq file into
     * a pulseg_pulseq_file_set.
     */
    int pulseg_pulseq_file_set_read(
        pulseg_pulseq_file_set *coll,
        const char *first_file_path,
        const pulseg_opts *opts);

    /**
     * @brief Resolve a block's raw content ids (rf/gx/gy/gz/adc/extension
     * chain head) from the BLOCKS table, without inlining event data.
     */
    int pulseg_pulseq_get_raw_block_content_ids(
        const pulseg_pulseq_file *seq,
        pulseg_pulseq_raw_block *block,
        int block_index,
        int parse_extensions);

    /** @brief Walk a block's extension chain into a resolved index set. */
    void pulseg_pulseq_get_raw_extension(
        const pulseg_pulseq_file *seq,
        pulseg_pulseq_raw_extension *re,
        const pulseg_pulseq_raw_block *raw);

    /** @brief Resolve a block index into a fully-inlined pulseg_pulseq_block. */
    void pulseg_pulseq_get_block(
        const pulseg_pulseq_file *seq,
        pulseg_pulseq_block *block,
        int block_index);

    /** @brief Max |amplitude| across the whole GRAD library (trap + arb). */
    float pulseg_pulseq_get_grad_library_max_amplitude(const pulseg_pulseq_file *seq);

    /**
     * @brief Decompress a Pulseq run-length-encoded SHAPES library entry.
     * @param[out] result   Receives the decompressed samples (caller frees).
     * @param[in]  encoded  Raw (possibly RLE-compressed) shape.
     * @param[in]  scale    Multiplier applied to every decompressed sample.
     * @return 1 on success, 0 on failure.
     */
    int pulseg_pulseq_decompress_shape(
        pulseg_shape_arbitrary *result,
        const pulseg_shape_arbitrary *encoded,
        float scale);

    /** @brief Verify a .seq file's embedded [SIGNATURE] MD5 hash. */
    int pulseg_pulseq_verify_signature(const char *file_path);

    /**
     * @brief Map a Pulseq label/flag name (e.g. "SLC", "PMC", "TRID") to the
     * numeric label id used in pulseg_pulseq_file.labelset_library /
     * .labelinc_library raw rows (column 1) and decoded by
     * pulseg_pulseq_get_raw_extension(). Needed by any external producer of
     * a pulseg_pulseq_file (e.g. the ExternalSequence adapter,
     * cxx/pulseq_adapter) that cannot re-derive this id itself.
     * @return The numeric id, or -1 if @p name is not a recognized label/flag.
     */
    int pulseg_pulseq_label_id_for_name(const char *name);

    /**
     * @brief Map a soft-delay hint name (e.g. "TE", "TR", "TI") to the
     * numeric hint id used in pulseg_pulseq_file.soft_delay_library raw rows
     * (column 3, hint_id). Needed by any external producer of a
     * pulseg_pulseq_file for the same reason as
     * pulseg_pulseq_label_id_for_name().
     * @return The numeric id, or -1 if @p name is not a recognized hint.
     */
    int pulseg_pulseq_hint_id_for_name(const char *name);

    /* ================================================================== */
    /*  Options initializer                                               */
    /* ================================================================== */

    /**
     * @brief Fill a pulseg_opts struct with scanner parameters.
     *
     * The @c vendor field is set to @c PULSEG_VENDOR (compile-time
     * default).  Override it after calling this function if needed.
     */
    void pulseg_opts_init_full(
        pulseg_opts *opts,
        float gamma_hz_per_t,
        float b0_t,
        float max_grad_hz_per_m,
        float max_slew_hz_per_m_per_s,
        float rf_raster_us,
        float grad_raster_us,
        float adc_raster_us,
        float block_raster_us,
        float peak_log10_threshold,
        float peak_norm_scale,
        float peak_eps);

    /**
     * @brief Legacy initializer using default peak-detection parameters.
     */
    void pulseg_opts_init(
        pulseg_opts *opts,
        float gamma_hz_per_t,
        float b0_t,
        float max_grad_hz_per_m,
        float max_slew_hz_per_m_per_s,
        float rf_raster_us,
        float grad_raster_us,
        float adc_raster_us,
        float block_raster_us);

    /* ================================================================== */
    /*  Scan-time peek (fast estimate from definitions only)              */
    /* ================================================================== */

    /**
     * @brief Peek at scan time without full sequence loading.
     *
     * Reads only the [DEFINITIONS] sections from a (possibly chained)
     * .seq file to obtain @c TotalDuration.  The result is an
     * approximation: dead time between segments is not accounted for
     * and @c total_segment_boundaries is left at 0.
     *
     * @c num_reps controls the number of repetitions the consumer
     * intends to play (>= 1).  For subsequences whose
     * @c IgnoreAverages definition is set, the multiplier is clamped to 1.
     *
     * Both @c total_duration_us and @c total_segment_boundaries are populated.
     *
     * @param[out] info       Receives scan time summary.
     * @param[in]  file_path  Path to the first .seq file (may be chained).
     * @param[in]  opts       Library options.
     * @param[in]  num_reps   Number of repetitions (>= 1).
     * @return PULSEG_SUCCESS on success, negative error code on failure.
     */
    int pulseg_peek_scan_time(
        pulseg_scan_time_info *info,
        const char *file_path,
        const pulseg_opts *opts,
        int num_reps);

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_IO_H */
