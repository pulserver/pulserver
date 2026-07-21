/**
 * @file pulseq_types.h
 * @brief Type definitions for the raw Pulseq (.seq) sequence-file model.
 *
 * These types mirror the on-disk Pulseq file format one-to-one: every
 * library section decodes into a flat array, and a block is a set of
 * indices into those arrays.  Nothing here knows about any higher-level
 * intermediate representation.
 *
 * Naming: all identifiers are snake_case with a pulseq_ prefix.  INIT
 * macros are C89 and C++ compatible (positional, no designated init).
 */

#ifndef PULSEQ_TYPES_H
#define PULSEQ_TYPES_H

#include "pulseq_config.h"

/* ================================================================== */
/*  Length / size constants                                           */
/* ================================================================== */
#define PULSEQ_DEFINITION_NAME_LENGTH 32
#define PULSEQ_EXT_NAME_LENGTH 32
#define PULSEQ_LABEL_NAME_LENGTH 32
#define PULSEQ_SEQUENCE_NAME_LENGTH 256
#define PULSEQ_SEQUENCE_FILENAME_LENGTH 256
#define PULSEQ_SOFT_DELAY_HINT_LENGTH 32
#define PULSEQ_MAX_EXTENSIONS_PER_BLOCK 64
#define PULSEQ_MAX_LINE_LENGTH 256
#define PULSEQ_MAX_SCALE_SIZE 16
#define PULSEQ_MAX_RF_SHIM_CHANNELS 64

/* ================================================================== */
/*  RF use codes (the trailing e/r/i/s use tag on an RF library row)   */
/* ================================================================== */
#define PULSEQ_RF_USE_UNKNOWN 0
#define PULSEQ_RF_USE_EXCITATION 1
#define PULSEQ_RF_USE_REFOCUSING 2
#define PULSEQ_RF_USE_INVERSION 3
#define PULSEQ_RF_USE_SATURATION 4

/* ================================================================== */
/*  Extension type codes for a block's extension chain, and for any    */
/*  extensions_library row written by an external raw-model producer   */
/*  (e.g. the ExternalSequence adapter, cxx/pulseq_adapter).           */
/*  These values are load-bearing wire values: pulseq_get_raw_block_-  */
/*  content_ids() and pulseq_get_raw_extension() switch on them.       */
/* ================================================================== */
#define PULSEQ_EXT_LIST 0
#define PULSEQ_EXT_TRIGGER 1
#define PULSEQ_EXT_ROTATION 2
#define PULSEQ_EXT_LABELSET 3
#define PULSEQ_EXT_LABELINC 4
#define PULSEQ_EXT_RF_SHIM 5
#define PULSEQ_EXT_DELAY 6
#define PULSEQ_EXT_UNKNOWN 7

/* ================================================================== */
/*  Label / flag ids                                                  */
/*                                                                    */
/*  Numeric ids for the names appearing in a LABELSET / LABELINC row  */
/*  (column 1).  Load-bearing wire values: pulseq_get_raw_extension() */
/*  decodes against them, and any external producer of a pulseq_file  */
/*  must use pulseq_label_id_for_name() to obtain them.               */
/* ================================================================== */
#define PULSEQ_LABEL_SLC 1
#define PULSEQ_LABEL_SEG 2
#define PULSEQ_LABEL_REP 3
#define PULSEQ_LABEL_AVG 4
#define PULSEQ_LABEL_SET 5
#define PULSEQ_LABEL_ECO 6
#define PULSEQ_LABEL_PHS 7
#define PULSEQ_LABEL_LIN 8
#define PULSEQ_LABEL_PAR 9
#define PULSEQ_LABEL_ACQ 10
#define PULSEQ_LABEL_NAV 11
#define PULSEQ_LABEL_REV 12
#define PULSEQ_LABEL_SMS 13
#define PULSEQ_LABEL_REF 14
#define PULSEQ_LABEL_IMA 15
#define PULSEQ_LABEL_NOISE 16
#define PULSEQ_LABEL_PMC 17
#define PULSEQ_LABEL_NOROT 18
#define PULSEQ_LABEL_NOPOS 19
#define PULSEQ_LABEL_NOSCL 20
#define PULSEQ_LABEL_ONCE 21
#define PULSEQ_LABEL_TRID 22
#define PULSEQ_LABEL_OFF 23
#define PULSEQ_LABEL_MODULE 24

/* ================================================================== */
/*  Soft-delay hint ids (soft_delay_library column 3)                 */
/* ================================================================== */
#define PULSEQ_HINT_TE 1
#define PULSEQ_HINT_TR 2
#define PULSEQ_HINT_TI 3
#define PULSEQ_HINT_ESP 4
#define PULSEQ_HINT_RECTIME 5
#define PULSEQ_HINT_T2PREP 6
#define PULSEQ_HINT_TE2 7

/* ================================================================== */
/*  Gradient types (grad library row discriminator)                   */
/* ================================================================== */
#define PULSEQ_GRAD_TRAP 1
#define PULSEQ_GRAD_ARB 2

/* ================================================================== */
/*  Trigger types and channels                                        */
/* ================================================================== */
#define PULSEQ_TRIGGER_TYPE_OUTPUT 1 /**< TTL / digital output */
#define PULSEQ_TRIGGER_TYPE_INPUT 2  /**< ECG / cardiac gating */

#define PULSEQ_TRIGGER_CHANNEL_INPUT_PHYSIO_1 1
#define PULSEQ_TRIGGER_CHANNEL_INPUT_PHYSIO_2 2
#define PULSEQ_TRIGGER_CHANNEL_OUTPUT_OSC_0 1
#define PULSEQ_TRIGGER_CHANNEL_OUTPUT_OSC_1 2
#define PULSEQ_TRIGGER_CHANNEL_OUTPUT_EXT_1 3

/* ================================================================== */
/*  Shape (RLE-decompressible waveform)                               */
/* ================================================================== */

/**
 * @brief One SHAPES library entry, possibly run-length encoded.
 *
 * Decompress with pulseq_decompress_shape().  When num_samples ==
 * num_uncompressed_samples the payload is already raw.
 */
typedef struct pulseq_shape
{
    int num_uncompressed_samples;
    int num_samples;
    float *samples;
} pulseq_shape;

/* clang-format off */
#define PULSEQ_SHAPE_INIT {0, 0, NULL}
/* clang-format on */

/* ================================================================== */
/*  Raw event types (one raw pulseq library entry, pre-dedup)          */
/* ================================================================== */

/** @brief The ten Pulseq counter labels carried by one block. */
typedef struct pulseq_label_event
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
} pulseq_label_event;

/** @brief Boolean flags (plus the two sticky ids) carried by one block. */
typedef struct pulseq_flag_event
{
    int trid;
    int module_id; /* MODULE: sticky int id, not boolean -- same non-flag
                    * treatment as trid above; both live here only because
                    * LABELMAP_COUNTER's LABELINC-increment semantics don't
                    * apply to either. */
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
} pulseq_flag_event;

/**
 * @brief Digital trigger / physio event (raw TRIGGERS library entry, and
 * per-block resolved trigger).
 */
typedef struct pulseq_trigger_event
{
    short type;
    long duration;
    long delay;
    int trigger_type;
    int trigger_channel;
} pulseq_trigger_event;

/* ================================================================== */
/*  Label limits                                                      */
/* ================================================================== */

typedef struct pulseq_label_limit
{
    int min;
    int max;
} pulseq_label_limit;

/* ================================================================== */
/*  Blocks                                                            */
/* ================================================================== */

/** @brief Raw block: unresolved content ids straight off the BLOCKS table. */
typedef struct pulseq_raw_block
{
    int block_duration;
    int rf;
    int gx;
    int gy;
    int gz;
    int adc;
    int ext_count;
    int ext[PULSEQ_MAX_EXTENSIONS_PER_BLOCK][2];
} pulseq_raw_block;

/**
 * @brief Resolved extension chain for one block (labels/flags/rotation/
 * shim/trigger/soft-delay indices; rotation/shim/trigger/soft-delay are
 * indices into the owning pulseq_file's libraries, -1 if absent).
 */
typedef struct pulseq_raw_extension
{
    pulseq_label_event labelset;
    pulseq_label_event labelinc;
    pulseq_flag_event flag;
    int rotation_index;
    int rf_shim_index;
    int trigger_index;
    int soft_delay_index;
} pulseq_raw_extension;

/* ================================================================== */
/*  Sequence file                                                     */
/* ================================================================== */

/** @brief Parse-cursor byte offsets for each [SECTION] (internal bookkeeping). */
typedef struct pulseq_section_offsets
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
} pulseq_section_offsets;

/** @brief One generic [DEFINITIONS] key/value entry. */
typedef struct pulseq_definition
{
    char name[PULSEQ_DEFINITION_NAME_LENGTH];
    int value_size;
    char **value;
} pulseq_definition;

/** @brief Reserved (well-known) [DEFINITIONS] keys, parsed eagerly. */
typedef struct pulseq_reserved_definitions
{
    float gradient_raster_time;
    float radiofrequency_raster_time;
    float adc_raster_time;
    float block_duration_raster;
    char name[PULSEQ_SEQUENCE_NAME_LENGTH];
    float fov[3];
    float matrix[3];
    float nav_fov[3];
    float nav_matrix[3];
    float total_duration;
    char next_sequence[PULSEQ_SEQUENCE_FILENAME_LENGTH];
    int ignore_fov_shift;
    int enable_pmc;
    int ignore_averages;
    int num_gain_cal_readouts; /**< calibration readouts for receive gain */
} pulseq_reserved_definitions;

/** @brief One RF_SHIMS library entry (parallel-transmit channel weights). */
typedef struct pulseq_rf_shim_entry
{
    int num_channels;
    float values[2 * PULSEQ_MAX_RF_SHIM_CHANNELS];
} pulseq_rf_shim_entry;

/**
 * @brief Design-time raster times, in microseconds.
 *
 * The grid the sequence was designed on, supplied by the caller.  It is
 * consulted ONLY for rasters the .seq file itself fails to declare in
 * [DEFINITIONS]; a well-formed file declares its own, and those always win.
 *
 * Note the distinction, which this module deliberately does not blur:
 *   - The rasters in [DEFINITIONS] (see pulseq_reserved_definitions) describe
 *     the grid the file's own waveforms and event times are written on.  They
 *     are a property of the FILE, and are what this module parses against.
 *   - A scanner's hardware rasters are a property of the SYSTEM.  Deciding
 *     whether a file is playable on a given system, and resampling its
 *     waveforms onto the hardware grid, is the job of the layer above this
 *     one -- pulseq neither stores nor consults system rasters.
 *
 * Pass NULL to pulseq_file_init() to leave these zero.
 */
typedef struct pulseq_raster
{
    float rf_us;    /**< RF sample raster (us)       */
    float grad_us;  /**< gradient sample raster (us) */
    float block_us; /**< block duration raster (us)  */
} pulseq_raster;

/**
 * @brief A fully-parsed Pulseq .seq file: every library section decoded
 * into flat arrays.
 *
 * Created by pulseq_read() / pulseq_read_from_buffer().  Freed by
 * pulseq_file_free().  Fields are the completed, fully-parsed raw model --
 * no half-parsed / file-handle state is ever exposed once a read returns
 * success.
 */
typedef struct pulseq_file
{
    /** Design-time rasters, used for the ones [DEFINITIONS] omits; see
     *  pulseq_raster.  The file's own declared rasters live in
     *  reserved_definitions_library and take precedence wherever both are
     *  available. */
    pulseq_raster design_raster;
    char *file_path;
    pulseq_section_offsets offsets;
    int is_version_parsed;
    int version_combined;
    int version_major;
    int version_minor;
    int version_revision;
    int is_definitions_library_parsed;
    int num_definitions;
    pulseq_definition *definitions_library;
    pulseq_reserved_definitions reserved_definitions_library;
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
    pulseq_label_limit label_limits;
    int is_delay_defined[8];
    int soft_delay_library_size;
    float (*soft_delay_library)[4];
    int rf_shim_library_size;
    pulseq_rf_shim_entry *rf_shim_library;
    int extension_map[8];
    int extension_lut_size;
    int *extension_lut;
    int is_shapes_library_parsed;
    int shapes_library_size;
    pulseq_shape *shapes_library;
} pulseq_file;

/**
 * @brief A chain of pulseq files (Pulseq "next sequence" chaining).
 *
 * Created by pulseq_file_set_read().  Freed by pulseq_file_set_free().
 */
typedef struct pulseq_file_set
{
    int num_sequences;
    pulseq_file *sequences;
    char *base_path;
} pulseq_file_set;

#endif /* PULSEQ_TYPES_H */
