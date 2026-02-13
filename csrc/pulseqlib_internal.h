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
    int n_chan;
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
    int n_channels;
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

/* --- pulseqlib_math.c --- */
float pulseqlib__find_max_abs_real(const float* samples, int n);
int   pulseqlib__find_max_abs_index_real(const float* samples, int n);
void  pulseqlib__mag_phase_to_real_imag(float* re, float* im, const float* mag, const float* phase, int n);
void  pulseqlib__quaternion_to_matrix(float* matrix, const float* quat);
void  pulseqlib__apply_rotation(float* out, const float* R, const float* v, int transpose);
void  pulseqlib__interp1_linear(float* out, const float* x, int nx, const float* xp, const float* fp, int nxp);
void  pulseqlib__interp1_linear_complex(float* out_re, float* out_im, const float* x, int nx, const float* xp, const float* fp_re, const float* fp_im, int nxp);
void  pulseqlib__fftshift_complex(float* re, float* im, int n);
float pulseqlib__find_spectrum_flank(const float* x, const float* re, const float* im, int n, float cutoff, int reverse);
size_t pulseqlib__next_pow2(size_t x);
#if PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_GEHC
int   pulseqlib__convolve_fft(float* output, const float* signal, int signal_len, const float* kernel, int kernel_len);
#endif

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

/* --- pulseqlib_core.c --- */
int   pulseqlib__get_unique_blocks(pulseqlib_sequence_descriptor* desc, const pulseqlib__seq_file* seq);
int   pulseqlib__find_tr_in_sequence(pulseqlib_sequence_descriptor* desc, pulseqlib_diagnostic* diag);
int   pulseqlib__find_segments_in_tr(pulseqlib_sequence_descriptor* desc, pulseqlib_diagnostic* diag, const pulseqlib__seq_file* seq);
int   pulseqlib__get_collection_descriptors(pulseqlib_sequence_descriptor_collection* desc_coll, pulseqlib_diagnostic* diag, const pulseqlib__seq_file_collection* coll);

/* --- Helper to locate segment/block in collection --- */
int pulseqlib__resolve_segment(
    const pulseqlib_sequence_descriptor** out_desc,
    int* out_local_seg,
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx);

int pulseqlib__resolve_block(
    const pulseqlib_sequence_descriptor** out_desc,
    const pulseqlib_tr_segment** out_seg,
    int* out_local_blk,
    const pulseqlib_sequence_descriptor_collection* coll,
    int seg_idx, int blk_idx);

#endif /* PULSEQLIB_INTERNAL_H */