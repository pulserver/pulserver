/* pulseqlib_parse.c -- .seq file parsing, library init/read, block retrieval */

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "external_md5.h"

#include "pulseqlib_internal.h"

/* ================================================================== */
/*  Internal macro                                                    */
/* ================================================================== */
#define INIT_LIBRARY(seq, field_ptr, size_field, flag_field) \
    do { \
        (seq)->field_ptr = NULL; \
        (seq)->size_field = 0; \
        (seq)->flag_field = 0; \
    } while (0)


/* ================================================================== */
/*  Path helpers (static)                                             */
/* ================================================================== */

static char* extract_base_path(const char* file_path)
{
    char* base;
    const char* last_slash;
    size_t len;

    last_slash = strrchr(file_path, '/');
    if (!last_slash) last_slash = strrchr(file_path, '\\');

    if (last_slash) {
        len = (size_t)(last_slash - file_path + 1);
        base = (char*)PULSEQLIB_ALLOC(len + 1);
        if (base) { strncpy(base, file_path, len); base[len] = '\0'; }
    } else {
        base = (char*)PULSEQLIB_ALLOC(3);
        if (base) strcpy(base, "./");
    }
    return base;
}

static char* build_full_path(const char* base_path, const char* filename)
{
    size_t len;
    char* full;

    len = strlen(base_path) + strlen(filename) + 1;
    full = (char*)PULSEQLIB_ALLOC(len);
    if (full) { strcpy(full, base_path); strcat(full, filename); }
    return full;
}

/* ================================================================== */
/*  seq_file defaults / init / reset / free                           */
/* ================================================================== */

static void seq_file_set_defaults(pulseqlib__seq_file* seq)
{
    int i;
    if (!seq) return;

    seq->offsets.scan_cursor = 0;
    seq->offsets.version     = -1;
    seq->offsets.definitions = -1;
    seq->offsets.blocks      = -1;
    seq->offsets.rf          = -1;
    seq->offsets.grad        = -1;
    seq->offsets.trap        = -1;
    seq->offsets.adc         = -1;
    seq->offsets.extensions  = -1;
    seq->offsets.triggers    = -1;
    seq->offsets.rfshim      = -1;
    seq->offsets.labelset    = -1;
    seq->offsets.labelinc    = -1;
    seq->offsets.delays      = -1;
    seq->offsets.rotations   = -1;
    seq->offsets.shapes      = -1;
    seq->offsets.signature   = -1;

    seq->is_version_parsed = 0;
    seq->version_combined  = 0;
    seq->version_major     = 0;
    seq->version_minor     = 0;
    seq->version_revision  = 0;

    INIT_LIBRARY(seq, definitions_library, num_definitions, is_definitions_library_parsed);
    memset(&seq->reserved_definitions_library, 0, sizeof(seq->reserved_definitions_library));

    INIT_LIBRARY(seq, block_library, num_blocks, is_block_library_parsed);
    seq->block_ids = NULL;
    INIT_LIBRARY(seq, rf_library, rf_library_size, is_rf_library_parsed);
    seq->rf_use_tags = NULL;
    INIT_LIBRARY(seq, grad_library, grad_library_size, is_grad_library_parsed);
    INIT_LIBRARY(seq, adc_library, adc_library_size, is_adc_library_parsed);
    INIT_LIBRARY(seq, extensions_library, extensions_library_size, is_extensions_library_parsed);
    INIT_LIBRARY(seq, trigger_library, trigger_library_size, is_extensions_library_parsed);
    INIT_LIBRARY(seq, rotation_quaternion_library, rotation_library_size, is_extensions_library_parsed);
    INIT_LIBRARY(seq, rotation_matrix_library, rotation_library_size, is_extensions_library_parsed);
    INIT_LIBRARY(seq, labelset_library, labelset_library_size, is_extensions_library_parsed);
    INIT_LIBRARY(seq, labelinc_library, labelinc_library_size, is_extensions_library_parsed);
    for (i = 0; i < 22; i++) seq->is_label_defined[i] = 0;
    memset(&seq->label_limits, 0, sizeof(seq->label_limits));
    for (i = 0; i < 8; i++) {
        seq->is_delay_defined[i] = 0;
        seq->extension_map[i] = -1;
    }
    INIT_LIBRARY(seq, soft_delay_library, soft_delay_library_size, is_extensions_library_parsed);
    INIT_LIBRARY(seq, rf_shim_library, rf_shim_library_size, is_extensions_library_parsed);
    seq->extension_lut_size = 0;
    seq->extension_lut = NULL;
    INIT_LIBRARY(seq, shapes_library, shapes_library_size, is_shapes_library_parsed);
}

void pulseqlib__seq_file_init(pulseqlib__seq_file* seq, const pulseqlib_opts* opts)
{
    if (!seq) return;
    seq->file_path = NULL;
    seq->opts = *opts;
    seq_file_set_defaults(seq);
}

static void seq_file_reset(pulseqlib__seq_file* seq)
{
    int i, j;
    if (!seq) return;

    if (seq->is_definitions_library_parsed && seq->definitions_library) {
        for (i = 0; i < seq->num_definitions; i++) {
            for (j = 0; j < seq->definitions_library[i].value_size; j++)
                PULSEQLIB_FREE(seq->definitions_library[i].value[j]);
            PULSEQLIB_FREE(seq->definitions_library[i].value);
        }
        PULSEQLIB_FREE(seq->definitions_library);
    }
    if (seq->is_block_library_parsed) {
        PULSEQLIB_FREE(seq->block_library);
        PULSEQLIB_FREE(seq->block_ids);
        seq->block_ids = NULL;
    }
    if (seq->is_rf_library_parsed) {
        PULSEQLIB_FREE(seq->rf_library);
        if (seq->rf_use_tags) PULSEQLIB_FREE(seq->rf_use_tags);
        seq->rf_use_tags = NULL;
    }
    if (seq->is_grad_library_parsed) PULSEQLIB_FREE(seq->grad_library);
    if (seq->is_adc_library_parsed)  PULSEQLIB_FREE(seq->adc_library);
    if (seq->is_extensions_library_parsed) {
        PULSEQLIB_FREE(seq->extensions_library);
        PULSEQLIB_FREE(seq->trigger_library);
        PULSEQLIB_FREE(seq->rotation_quaternion_library);
        PULSEQLIB_FREE(seq->rotation_matrix_library);
        PULSEQLIB_FREE(seq->labelset_library);
        PULSEQLIB_FREE(seq->labelinc_library);
        PULSEQLIB_FREE(seq->soft_delay_library);
        PULSEQLIB_FREE(seq->rf_shim_library);
    }
    if (seq->is_shapes_library_parsed && seq->shapes_library) {
        for (i = 0; i < seq->shapes_library_size; i++) {
            PULSEQLIB_FREE(seq->shapes_library[i].samples);
            seq->shapes_library[i].samples = NULL;
            seq->shapes_library[i].num_uncompressed_samples = 0;
            seq->shapes_library[i].num_samples = 0;
        }
        PULSEQLIB_FREE(seq->shapes_library);
    }
    PULSEQLIB_FREE(seq->extension_lut);
    seq->extension_lut = NULL;

    seq_file_set_defaults(seq);
}

void pulseqlib__seq_file_free(pulseqlib__seq_file* seq)
{
    if (!seq) return;
    seq_file_reset(seq);
    if (seq->file_path) { PULSEQLIB_FREE(seq->file_path); seq->file_path = NULL; }
    memset(&seq->opts, 0, sizeof(seq->opts));
    PULSEQLIB_FREE(seq);
}

void pulseqlib__seq_file_collection_free(pulseqlib__seq_file_collection* coll)
{
    int i;
    if (!coll) return;
    if (coll->sequences) {
        for (i = 0; i < coll->num_sequences; ++i) {
            seq_file_reset(&coll->sequences[i]);
            if (coll->sequences[i].file_path) {
                PULSEQLIB_FREE(coll->sequences[i].file_path);
                coll->sequences[i].file_path = NULL;
            }
        }
        PULSEQLIB_FREE(coll->sequences);
    }
    if (coll->base_path) PULSEQLIB_FREE(coll->base_path);
    coll->num_sequences = 0;
    coll->sequences = NULL;
    coll->base_path = NULL;
}

/* ================================================================== */
/*  Library init helpers (static)                                     */
/* ================================================================== */

static int init_standard_library(FILE* f, const long* offsets, int num_sections,
                                 void** target, int* target_count, int n)
{
    char line[PULSEQLIB__MAX_LINE_LENGTH];
    int max_idx = -1;
    int sec, idx, i;
    char* p;
    float* arr;

    if (!f) return 1;
    for (sec = 0; sec < num_sections; sec++) {
        if (offsets[sec] < 0) continue;
        if (fseek(f, offsets[sec], SEEK_SET) != 0) return 1;
        if (!fgets(line, sizeof(line), f)) return 1;
        while (fgets(line, sizeof(line), f)) {
            p = line;
            while (*p == ' ' || *p == '\t') p++;
            if (*p == '[' || *p == 'e') break;
            if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue;
            if (sscanf(p, "%d", &idx) == 1 && idx > max_idx) max_idx = idx;
        }
    }
    if (max_idx <= 0) { *target = NULL; *target_count = 0; return 0; }
    arr = (float*)PULSEQLIB_ALLOC(sizeof(float) * n * max_idx);
    if (!arr) return 1;
    for (i = 0; i < max_idx * n; i++) arr[i] = 0.0f;
    *target = (void*)arr;
    *target_count = max_idx;
    return 0;
}

static int init_definitions_library(FILE* f, long offset,
                                    pulseqlib__definition** target, int* target_count)
{
    char line[PULSEQLIB__MAX_LINE_LENGTH];
    int count = 0;
    char* p;
    char* name_tok;
    pulseqlib__definition* defs;

    if (!f || offset < 0 || !target || !target_count) return 1;
    if (fseek(f, offset, SEEK_SET) != 0) return 1;
    if (!fgets(line, sizeof(line), f)) return 1;
    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '[' || *p == 'e') break;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue;
        name_tok = strtok(p, " \t\r\n");
        if (name_tok) count++;
    }
    if (count == 0) { *target = NULL; *target_count = 0; return 1; }
    defs = (pulseqlib__definition*)PULSEQLIB_ALLOC(sizeof(pulseqlib__definition) * count);
    if (!defs) return 1;
    *target = defs;
    *target_count = count;
    return 0;
}

static int init_shapes_library(FILE* f, long offset,
                               pulseqlib_shape_arbitrary** target, int* target_count)
{
    char line[PULSEQLIB__MAX_LINE_LENGTH];
    int max_idx = -1;
    int n, i, j, idx = 0;
    char* p;
    pulseqlib_shape_arbitrary* shapes;

    if (!f || !offset || !target || !target_count) return 1;
    if (fseek(f, offset, SEEK_SET) != 0) return 1;
    if (!fgets(line, sizeof(line), f)) return 1;

    /* Pass 1: find max shape_id */
    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '[' || *p == 'e') break;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue;
        if (strncmp(p, "shape_id", 8) == 0) {
            if (sscanf(p + 8, "%d", &idx) == 1 && idx > max_idx) max_idx = idx;
        }
    }
    if (max_idx <= 0) { *target = NULL; *target_count = 0; return 0; }

    shapes = (pulseqlib_shape_arbitrary*)PULSEQLIB_ALLOC(sizeof(pulseqlib_shape_arbitrary) * max_idx);
    if (!shapes) return 1;
    for (i = 0; i < max_idx; i++) {
        shapes[i].num_samples = 0;
        shapes[i].num_uncompressed_samples = 0;
        shapes[i].samples = NULL;
    }

    /* Pass 2: count samples per shape */
    if (fseek(f, offset, SEEK_SET) != 0) { PULSEQLIB_FREE(shapes); return 1; }
    if (!fgets(line, sizeof(line), f)) return 1;
    idx = 0;
    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '[' || *p == 'e') break;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue;
        if (strncmp(p, "shape_id", 8) == 0) {
            if (sscanf(p + 8, "%d", &idx) == 1) {
                shapes[idx - 1].num_samples = 0;
                shapes[idx - 1].num_uncompressed_samples = 0;
                shapes[idx - 1].samples = NULL;
            }
        } else if (strncmp(p, "num_samples", 11) == 0) {
            if (sscanf(p + 11, "%d", &n) == 1)
                shapes[idx - 1].num_uncompressed_samples = n;
        } else {
            shapes[idx - 1].num_samples++;
        }
    }

    /* Allocate sample arrays */
    for (i = 0; i < max_idx; i++) {
        n = shapes[i].num_samples;
        if (n > 0) {
            shapes[i].samples = (float*)PULSEQLIB_ALLOC(sizeof(float) * n);
            if (!shapes[i].samples) {
                for (j = 0; j < i; j++) { if (shapes[j].samples) PULSEQLIB_FREE(shapes[j].samples); }
                PULSEQLIB_FREE(shapes);
                return 1;
            }
        }
    }

    *target = shapes;
    *target_count = max_idx;
    return 0;
}

static int init_rf_shim_library(FILE* f, long offset,
                                pulseqlib__rf_shim_entry** target, int* target_count)
{
    char line[PULSEQLIB__MAX_LINE_LENGTH];
    int max_idx = -1;
    char* p;
    int idx, i;
    pulseqlib__rf_shim_entry* arr;

    if (!f || !target || !target_count) return 1;
    if (fseek(f, offset, SEEK_SET) != 0) return 1;
    if (!fgets(line, sizeof(line), f)) return 1;
    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '[' || *p == 'e') break;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue;
        if (sscanf(p, "%d", &idx) == 1 && idx > max_idx) max_idx = idx;
    }
    if (max_idx <= 0) { *target = NULL; *target_count = 0; return 1; }
    arr = (pulseqlib__rf_shim_entry*)PULSEQLIB_ALLOC(sizeof(pulseqlib__rf_shim_entry) * max_idx);
    if (!arr) return 1;
    for (i = 0; i < max_idx; i++) arr[i].num_channels = 0;
    *target = arr;
    *target_count = max_idx;
    return 0;
}

/* ================================================================== */
/*  Library read helpers (static)                                     */
/* ================================================================== */

static int read_standard_library(FILE* f, long offset, void* target, int target_count,
                                 int n, pulseqlib__scale scale, int flag)
{
    char line[PULSEQLIB__MAX_LINE_LENGTH];
    int idx, parsed, consumed, col, offset_col;
    float vals[PULSEQLIB__MAX_SCALE_SIZE];
    char* scan_ptr;
    char* p;
    float v;
    float* arr = (float*)target;

    if (!f) return 1;
    if (scale.size > PULSEQLIB__MAX_SCALE_SIZE) return 1;
    if (fseek(f, offset, SEEK_SET) != 0) return 1;
    if (!fgets(line, sizeof(line), f)) return 1;

    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '[' || *p == 'e') break;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue;
        if (sscanf(p, "%d", &idx) != 1) continue;
        if (idx <= 0 || idx > target_count) continue;

        while (*p && *p != ' ' && *p != '\t') p++;
        while (*p == ' ' || *p == '\t') p++;

        parsed = 0;
        scan_ptr = p;
        for (col = 0; col < scale.size; col++) {
            consumed = 0;
            if (sscanf(scan_ptr, "%f%n", &v, &consumed) != 1) break;
            vals[col] = v;
            scan_ptr += consumed;
            while (*scan_ptr == ' ' || *scan_ptr == '\t') scan_ptr++;
            parsed++;
        }
        if (parsed != scale.size) continue;

        offset_col = (flag >= 0) ? 1 : 0;
        for (col = 0; col < scale.size; col++)
            arr[(idx - 1) * n + col + offset_col] = vals[col] * scale.values[col];
        if (flag >= 0)
            arr[(idx - 1) * n + 0] = (float)flag;
    }
    return 0;
}

static void get_section_offsets(pulseqlib__seq_file* seq, FILE* f)
{
    char line[PULSEQLIB__MAX_LINE_LENGTH];
    char* p;
    long pos;
    char ext_name[PULSEQLIB__EXT_NAME_LENGTH];
    int ext_id, ext_enum;

    if (fseek(f, 0L, SEEK_SET) != 0) return;

    while (fgets(line, sizeof(line), f)) {
        pos = ftell(f);
        if (pos < 0) break;
        p = line;
        while (*p == ' ' || *p == '\t') p++;

        if (*p == '[') {
            if      (strncmp(p, "[VERSION]",    9) == 0)  seq->offsets.version    = pos - (long)strlen(line);
            else if (strncmp(p, "[DEFINITIONS]",13) == 0) seq->offsets.definitions= pos - (long)strlen(line);
            else if (strncmp(p, "[BLOCKS]",     8) == 0)  seq->offsets.blocks     = pos - (long)strlen(line);
            else if (strncmp(p, "[RF]",         4) == 0)  seq->offsets.rf         = pos - (long)strlen(line);
            else if (strncmp(p, "[GRADIENTS]", 11) == 0)  seq->offsets.grad       = pos - (long)strlen(line);
            else if (strncmp(p, "[TRAP]",       6) == 0)  seq->offsets.trap       = pos - (long)strlen(line);
            else if (strncmp(p, "[ADC]",        5) == 0)  seq->offsets.adc        = pos - (long)strlen(line);
            else if (strncmp(p, "[EXTENSIONS]",12) == 0)  seq->offsets.extensions = pos - (long)strlen(line);
            else if (strncmp(p, "[SHAPES]",     8) == 0)  seq->offsets.shapes     = pos - (long)strlen(line);
            else if (strncmp(p, "[SIGNATURE]", 11) == 0)  seq->offsets.signature  = pos - (long)strlen(line);
        }
        else if (strncmp(p, "extension", 9) == 0 && (*(p + 9) == ' ' || *(p + 9) == '\t')) {
            ext_id = -1;
            ext_enum = PULSEQLIB__EXT_UNKNOWN;
            if (sscanf(p, "extension %31s %d", ext_name, &ext_id) == 2) {
                if      (strcmp(ext_name, "TRIGGERS")  == 0) ext_enum = PULSEQLIB__EXT_TRIGGER;
                else if (strcmp(ext_name, "ROTATIONS") == 0) ext_enum = PULSEQLIB__EXT_ROTATION;
                else if (strcmp(ext_name, "LABELSET")  == 0) ext_enum = PULSEQLIB__EXT_LABELSET;
                else if (strcmp(ext_name, "LABELINC")  == 0) ext_enum = PULSEQLIB__EXT_LABELINC;
                else if (strcmp(ext_name, "RF_SHIMS")  == 0) ext_enum = PULSEQLIB__EXT_RF_SHIM;
                else if (strcmp(ext_name, "DELAYS")    == 0) ext_enum = PULSEQLIB__EXT_DELAY;
                switch (ext_enum) {
                    case PULSEQLIB__EXT_TRIGGER:  seq->offsets.triggers  = pos - (long)strlen(line); break;
                    case PULSEQLIB__EXT_ROTATION: seq->offsets.rotations = pos - (long)strlen(line); break;
                    case PULSEQLIB__EXT_LABELSET: seq->offsets.labelset  = pos - (long)strlen(line); break;
                    case PULSEQLIB__EXT_LABELINC: seq->offsets.labelinc  = pos - (long)strlen(line); break;
                    case PULSEQLIB__EXT_RF_SHIM:  seq->offsets.rfshim    = pos - (long)strlen(line); break;
                    case PULSEQLIB__EXT_DELAY:    seq->offsets.delays    = pos - (long)strlen(line); break;
                    default: break;
                }
                if (ext_enum >= 0 && ext_enum < PULSEQLIB__EXT_UNKNOWN)
                    seq->extension_map[ext_enum] = ext_id;
            }
        }
    }
    seq->offsets.scan_cursor = ftell(f);
}

static void read_version(pulseqlib__seq_file* seq, FILE* f)
{
    char line[PULSEQLIB__MAX_LINE_LENGTH];
    int major = 0, minor = 0, revision = 0;
    char key[32];
    int value;
    char* p;

    if (seq->is_version_parsed) return;
    if (seq->offsets.version < 0) { seq->is_version_parsed = 1; return; }
    if (fseek(f, seq->offsets.version, SEEK_SET) != 0) return;
    if (!fgets(line, sizeof(line), f)) return;

    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue;
        if (*p == '[') break;
        if (sscanf(p, "%31s %d", key, &value) == 2) {
            if      (strcmp(key, "major")    == 0) major    = value;
            else if (strcmp(key, "minor")    == 0) minor    = value;
            else if (strcmp(key, "revision") == 0) revision = value;
        }
    }
    seq->version_major    = major;
    seq->version_minor    = minor;
    seq->version_revision = revision;
    seq->version_combined = major * 1000000 + minor * 1000 + revision;
    seq->is_version_parsed = 1;
}

static void read_definitions_library(pulseqlib__seq_file* seq, FILE* f)
{
    int ret;
    char line[PULSEQLIB__MAX_LINE_LENGTH];
    int def_index = 0;
    char* p;
    char* name_tok;
    char* token;
    char** new_array;
    int i;
    pulseqlib__definition def;

    if (seq->is_definitions_library_parsed) return;
    if (seq->offsets.definitions < 0) { seq->is_definitions_library_parsed = 1; return; }

    ret = init_definitions_library(f, seq->offsets.definitions,
                                   &seq->definitions_library, &seq->num_definitions);
    if (ret != 0) return;

    if (fseek(f, seq->offsets.definitions, SEEK_SET) != 0) return;
    if (!fgets(line, sizeof(line), f)) return;

    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue;
        if (*p == '[') break;

        def.value_size = 0;
        def.value = NULL;

        name_tok = strtok(p, " \t\r\n");
        if (!name_tok) continue;
        strncpy(def.name, name_tok, PULSEQLIB__DEFINITION_NAME_LENGTH - 1);
        def.name[PULSEQLIB__DEFINITION_NAME_LENGTH - 1] = '\0';

        while ((token = strtok(NULL, " \t\r\n")) != NULL) {
            new_array = (char**)PULSEQLIB_ALLOC(sizeof(char*) * (def.value_size + 1));
            for (i = 0; i < def.value_size; i++) new_array[i] = def.value[i];
            new_array[def.value_size] = (char*)PULSEQLIB_ALLOC(strlen(token) + 1);
            strcpy(new_array[def.value_size], token);
            if (def.value) PULSEQLIB_FREE(def.value);
            def.value = new_array;
            def.value_size++;
        }
        seq->definitions_library[def_index++] = def;
    }
    seq->is_definitions_library_parsed = 1;
}

static void read_definitions(pulseqlib__seq_file* seq)
{
    int i;
    char* key;
    char* value;
    float temp[3];

    for (i = 0; i < seq->num_definitions; i++) {
        key   = seq->definitions_library[i].name;
        value = seq->definitions_library[i].value[0];

        if (strcmp(key, "GradientRasterTime") == 0) {
            seq->reserved_definitions_library.gradient_raster_time = (float)(atof(value) * 1e6);
        } else if (strcmp(key, "RadiofrequencyRasterTime") == 0) {
            seq->reserved_definitions_library.radiofrequency_raster_time = (float)(atof(value) * 1e6);
        } else if (strcmp(key, "AdcRasterTime") == 0) {
            seq->reserved_definitions_library.adc_raster_time = (float)(atof(value) * 1e6);
        } else if (strcmp(key, "BlockDurationRaster") == 0) {
            seq->reserved_definitions_library.block_duration_raster = (float)(atof(value) * 1e6);
        } else if (strcmp(key, "Name") == 0) {
            strncpy(seq->reserved_definitions_library.name, value,
                    sizeof(seq->reserved_definitions_library.name) - 1);
            seq->reserved_definitions_library.name[sizeof(seq->reserved_definitions_library.name) - 1] = '\0';
        } else if (strcmp(key, "FOV") == 0) {
            if (sscanf(value, "%f %f %f", &temp[0], &temp[1], &temp[2]) == 3) {
                seq->reserved_definitions_library.fov[0] = temp[0] * 100.0f;
                seq->reserved_definitions_library.fov[1] = temp[1] * 100.0f;
                seq->reserved_definitions_library.fov[2] = temp[2] * 100.0f;
            }
        } else if (strcmp(key, "TotalDuration") == 0) {
            seq->reserved_definitions_library.total_duration = (float)atof(value);
        } else if (strcmp(key, "NextSequence") == 0) {
            strncpy(seq->reserved_definitions_library.next_sequence, value,
                    sizeof(seq->reserved_definitions_library.next_sequence) - 1);
            seq->reserved_definitions_library.next_sequence[sizeof(seq->reserved_definitions_library.next_sequence) - 1] = '\0';
        } else if (strcmp(key, "IgnoreFovShift") == 0) {
            seq->reserved_definitions_library.ignore_fov_shift = atoi(value);
        } else if (strcmp(key, "EnablePmc") == 0) {
            seq->reserved_definitions_library.enable_pmc = atoi(value);
        } else if (strcmp(key, "IgnoreAverages") == 0) {
            seq->reserved_definitions_library.ignore_averages = atoi(value);
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Section readers                                                   */
/* ------------------------------------------------------------------ */

static void read_block_library(pulseqlib__seq_file* seq, FILE* f)
{
    int ret;
    float block_vals[7] = {1,1,1,1,1,1,1};
    pulseqlib__scale s;
    s.size = 7; s.values = block_vals;

    if (seq->is_block_library_parsed) return;
    if (seq->offsets.blocks < 0) { seq->is_block_library_parsed = 1; return; }
    ret = init_standard_library(f, &seq->offsets.blocks, 1,
                                (void**)&seq->block_library, &seq->num_blocks, s.size);
    if (ret != 0) return;
    ret = read_standard_library(f, seq->offsets.blocks, seq->block_library,
                                seq->num_blocks, s.size, s, -1);
    if (ret != 0) return;
    seq->is_block_library_parsed = 1;
}

static void read_rf_library(pulseqlib__seq_file* seq, FILE* f)
{
    int ret;
    float rf_vals[10] = {1,1,1,1,1,1,1,1,1,1};
    pulseqlib__scale s;
    s.size = 10; s.values = rf_vals;

    if (seq->is_rf_library_parsed) return;
    if (seq->offsets.rf < 0) { seq->is_rf_library_parsed = 1; return; }
    ret = init_standard_library(f, &seq->offsets.rf, 1,
                                (void**)&seq->rf_library, &seq->rf_library_size, s.size);
    if (ret != 0) return;
    ret = read_standard_library(f, seq->offsets.rf, seq->rf_library,
                                seq->rf_library_size, s.size, s, -1);
    if (ret != 0) return;
    seq->is_rf_library_parsed = 1;

    /* parse trailing RF use tags (e/r/s/i) from the [RF] section */
    if (seq->rf_library_size > 0 && seq->offsets.rf >= 0) {
        char use_line[PULSEQLIB__MAX_LINE_LENGTH];
        seq->rf_use_tags = (int*)PULSEQLIB_ALLOC(
            (size_t)seq->rf_library_size * sizeof(int));
        if (seq->rf_use_tags) {
            int ui;
            for (ui = 0; ui < seq->rf_library_size; ++ui)
                seq->rf_use_tags[ui] = PULSEQLIB_RF_USE_UNKNOWN;

            if (fseek(f, seq->offsets.rf, SEEK_SET) == 0 &&
                fgets(use_line, sizeof(use_line), f)) {
                while (fgets(use_line, sizeof(use_line), f)) {
                    char* up = use_line;
                    int use_idx;
                    float dummy;
                    int col, consumed;
                    char* scan_p;
                    while (*up == ' ' || *up == '\t') up++;
                    if (*up == '[' || *up == '\0' || *up == '\n' ||
                        *up == '\r' || *up == '#')
                        break;
                    if (sscanf(up, "%d", &use_idx) != 1) continue;
                    if (use_idx < 1 || use_idx > seq->rf_library_size) continue;
                    /* skip index */
                    while (*up && *up != ' ' && *up != '\t') up++;
                    while (*up == ' ' || *up == '\t') up++;
                    /* skip 10 float columns */
                    scan_p = up;
                    for (col = 0; col < 10; ++col) {
                        consumed = 0;
                        if (sscanf(scan_p, "%f%n", &dummy, &consumed) != 1)
                            break;
                        scan_p += consumed;
                        while (*scan_p == ' ' || *scan_p == '\t') scan_p++;
                    }
                    /* trailing character is the rf_use tag */
                    if (col == 10) {
                        char tag = *scan_p;
                        int use_val = PULSEQLIB_RF_USE_UNKNOWN;
                        if (tag == 'e') use_val = PULSEQLIB_RF_USE_EXCITATION;
                        else if (tag == 'r') use_val = PULSEQLIB_RF_USE_REFOCUSING;
                        else if (tag == 'i') use_val = PULSEQLIB_RF_USE_INVERSION;
                        else if (tag == 's') use_val = PULSEQLIB_RF_USE_SATURATION;
                        seq->rf_use_tags[use_idx - 1] = use_val;
                    }
                }
            }
        }
    }
}

static void read_grad_library(pulseqlib__seq_file* seq, FILE* f)
{
    int ret;
    long offsets[2];
    int num_sections = 0;
    static const float grad_vals[6] = {1,1,1,1,1,1};
    static const float trap_vals[5] = {1,1,1,1,1};
    pulseqlib__scale grad_s, trap_s;

    grad_s.size = 6; grad_s.values = (float*)grad_vals;
    trap_s.size = 5; trap_s.values = (float*)trap_vals;

    if (seq->is_grad_library_parsed) return;
    offsets[0] = seq->offsets.grad;
    offsets[1] = seq->offsets.trap;
    if (offsets[0] >= 0) num_sections++;
    if (offsets[1] >= 0) num_sections++;
    if (num_sections == 0) { seq->is_grad_library_parsed = 1; return; }

    ret = init_standard_library(f, offsets, 2,
                                (void**)&seq->grad_library, &seq->grad_library_size,
                                grad_s.size + 1);
    if (ret != 0) return;
    if (offsets[0] >= 0) {
        ret = read_standard_library(f, offsets[0], seq->grad_library,
                                    seq->grad_library_size, grad_s.size + 1, grad_s, 1);
        if (ret != 0) return;
    }
    if (offsets[1] >= 0) {
        ret = read_standard_library(f, offsets[1], seq->grad_library,
                                    seq->grad_library_size, grad_s.size + 1, trap_s, 0);
        if (ret != 0) return;
    }
    seq->is_grad_library_parsed = 1;
}

static void read_adc_library(pulseqlib__seq_file* seq, FILE* f)
{
    int ret;
    static const float adc_vals[8] = {1,1,1,1,1,1,1,1};
    pulseqlib__scale s;
    s.size = 8; s.values = (float*)adc_vals;

    if (seq->is_adc_library_parsed) return;
    if (seq->offsets.adc < 0) { seq->is_adc_library_parsed = 1; return; }
    ret = init_standard_library(f, &seq->offsets.adc, 1,
                                (void**)&seq->adc_library, &seq->adc_library_size, s.size);
    if (ret != 0) return;
    ret = read_standard_library(f, seq->offsets.adc, seq->adc_library,
                                seq->adc_library_size, s.size, s, -1);
    if (ret != 0) return;
    seq->is_adc_library_parsed = 1;
}

static void read_shapes_library(pulseqlib__seq_file* seq, FILE* f)
{
    int ret;
    char line[PULSEQLIB__MAX_LINE_LENGTH];
    int shape_index = 0, sample_index = 0;
    char* p;
    float val;

    if (seq->is_shapes_library_parsed) return;
    if (seq->offsets.shapes < 0) { seq->is_shapes_library_parsed = 1; return; }

    ret = init_shapes_library(f, seq->offsets.shapes,
                              &seq->shapes_library, &seq->shapes_library_size);
    if (ret != 0) return;

    if (fseek(f, seq->offsets.shapes, SEEK_SET) != 0) return;
    if (!fgets(line, sizeof(line), f)) return;

    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue;
        if (*p == '[') break;
        if (strncmp(p, "shape_id", 8) == 0) {
            if (sscanf(p + 8, "%d", &shape_index) == 1) sample_index = 0;
            continue;
        }
        if (strncmp(p, "num_samples", 11) == 0) continue;
        if (shape_index > 0 && shape_index <= seq->shapes_library_size) {
            if (sscanf(p, "%f", &val) == 1) {
                if (sample_index < seq->shapes_library[shape_index - 1].num_samples)
                    seq->shapes_library[shape_index - 1].samples[sample_index++] = val;
            }
        }
    }
    seq->is_shapes_library_parsed = 1;
}

static int read_label_library(FILE* f, long offset, void* target, int target_count,
                              int n, int* is_label_defined)
{
    char line[PULSEQLIB__MAX_LINE_LENGTH];
    char* p;
    int idx, label_code;
    float val;
    char label[PULSEQLIB__LABEL_NAME_LENGTH];
    float* arr = (float*)target;

    if (!f || offset < 0) return 1;
    if (fseek(f, offset, SEEK_SET) != 0) return 1;
    if (!fgets(line, sizeof(line), f)) return 1;

    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '[' || *p == 'e') break;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue;
        if (sscanf(p, "%d %f %31s", &idx, &val, label) != 3) continue;
        if (idx <= 0 || idx > target_count) continue;
        label_code = pulseqlib__label2enum(label);
        if (label_code > 0) is_label_defined[label_code - 1] = 1;
        arr[(idx - 1) * n + 0] = val;
        arr[(idx - 1) * n + 1] = (float)label_code;
    }
    return 0;
}

static int read_delay_library(FILE* f, long offset, void* target, int target_count,
                              int n, int* is_delay_defined)
{
    char line[PULSEQLIB__MAX_LINE_LENGTH];
    char* p;
    int idx, hint_code;
    float num_id, offset_val, scale_val;
    char hint[PULSEQLIB__SOFT_DELAY_HINT_LENGTH];
    float* arr = (float*)target;

    if (!f || offset < 0) return 1;
    if (fseek(f, offset, SEEK_SET) != 0) return 1;
    if (!fgets(line, sizeof(line), f)) return 1;

    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '[' || *p == 'e') break;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue;
        if (sscanf(p, "%d %f %f %f %31s", &idx, &num_id, &offset_val, &scale_val, hint) != 5) continue;
        if (idx <= 0 || idx > target_count) continue;
        hint_code = pulseqlib__hint2enum(hint);
        if (hint_code > 0) is_delay_defined[hint_code - 1] = 1;
        arr[(idx - 1) * n + 0] = num_id;
        arr[(idx - 1) * n + 1] = offset_val;
        arr[(idx - 1) * n + 2] = scale_val;
        arr[(idx - 1) * n + 3] = (float)hint_code;
    }
    return 0;
}

static int read_rf_shim_library(FILE* f, long offset,
                                pulseqlib__rf_shim_entry* target, int target_count)
{
    char line[PULSEQLIB__MAX_LINE_LENGTH];
    char* p;
    int idx, n_ch, i, consumed;
    float val;

    if (!f || !target) return 1;
    if (fseek(f, offset, SEEK_SET) != 0) return 1;
    if (!fgets(line, sizeof(line), f)) return 1;

    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '[' || *p == 'e') break;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue;
        if (sscanf(p, "%d %d", &idx, &n_ch) != 2) continue;
        if (idx <= 0 || idx > target_count || n_ch <= 0) continue;

        while (*p && *p != ' ') p++;
        while (*p == ' ') p++;
        while (*p && *p != ' ') p++;
        while (*p == ' ') p++;

        if (n_ch > PULSEQLIB__MAX_RF_SHIM_CHANNELS) return 1;
        target[idx - 1].num_channels = n_ch;
        for (i = 0; i < 2 * n_ch; i++) {
            consumed = 0;
            if (sscanf(p, "%f%n", &val, &consumed) != 1) break;
            target[idx - 1].values[i] = val;
            p += consumed;
            while (*p == ' ' || *p == '\t') p++;
        }
    }
    return 0;
}

static void read_extensions_library(pulseqlib__seq_file* seq, FILE* f)
{
    int ret, n, k;
    float qn;
    static const float ext_vals[3]  = {1,1,1};
    static const float trig_vals[4] = {1,1,1,1};
    static const float rot_vals[4]  = {1,1,1,1};
    pulseqlib__scale ext_s, trig_s, rot_s;

    ext_s.size  = 3; ext_s.values  = (float*)ext_vals;
    trig_s.size = 4; trig_s.values = (float*)trig_vals;
    rot_s.size  = 4; rot_s.values  = (float*)rot_vals;

    if (seq->is_extensions_library_parsed) return;
    if (seq->offsets.extensions < 0) { seq->is_extensions_library_parsed = 1; return; }

    /* Init & read extensions list */
    ret = init_standard_library(f, &seq->offsets.extensions, 1,
                                (void**)&seq->extensions_library, &seq->extensions_library_size, 3);
    if (ret != 0) return;

    if (seq->offsets.triggers >= 0) {
        ret = init_standard_library(f, &seq->offsets.triggers, 1,
                                    (void**)&seq->trigger_library, &seq->trigger_library_size, 4);
        if (ret != 0) return;
    }
    if (seq->offsets.rotations >= 0) {
        ret = init_standard_library(f, &seq->offsets.rotations, 1,
                                    (void**)&seq->rotation_quaternion_library, &seq->rotation_library_size, 4);
        if (ret != 0) return;
    }
    if (seq->offsets.labelset >= 0) {
        ret = init_standard_library(f, &seq->offsets.labelset, 1,
                                    (void**)&seq->labelset_library, &seq->labelset_library_size, 2);
        if (ret != 0) return;
    }
    if (seq->offsets.labelinc >= 0) {
        ret = init_standard_library(f, &seq->offsets.labelinc, 1,
                                    (void**)&seq->labelinc_library, &seq->labelinc_library_size, 2);
        if (ret != 0) return;
    }
    if (seq->offsets.delays >= 0) {
        ret = init_standard_library(f, &seq->offsets.delays, 1,
                                    (void**)&seq->soft_delay_library, &seq->soft_delay_library_size, 4);
        if (ret != 0) return;
    }
    if (seq->offsets.rfshim >= 0) {
        ret = init_rf_shim_library(f, seq->offsets.rfshim,
                                   &seq->rf_shim_library, &seq->rf_shim_library_size);
        if (ret != 0) return;
    }

    /* Read data */
    ret = read_standard_library(f, seq->offsets.extensions, seq->extensions_library,
                                seq->extensions_library_size, 3, ext_s, -1);
    if (ret != 0) return;

    if (seq->offsets.triggers >= 0) {
        ret = read_standard_library(f, seq->offsets.triggers, seq->trigger_library,
                                    seq->trigger_library_size, 4, trig_s, -1);
        if (ret != 0) return;
    }

    if (seq->offsets.rotations >= 0) {
        ret = read_standard_library(f, seq->offsets.rotations, seq->rotation_quaternion_library,
                                    seq->rotation_library_size, 4, rot_s, -1);
        if (ret != 0) return;
        /* Normalize quaternions */
        for (k = 1; k < seq->rotation_library_size; k++) {
            qn = (float)sqrt(
                (double)seq->rotation_quaternion_library[k][0] * seq->rotation_quaternion_library[k][0] +
                (double)seq->rotation_quaternion_library[k][1] * seq->rotation_quaternion_library[k][1] +
                (double)seq->rotation_quaternion_library[k][2] * seq->rotation_quaternion_library[k][2] +
                (double)seq->rotation_quaternion_library[k][3] * seq->rotation_quaternion_library[k][3]);
            seq->rotation_quaternion_library[k][0] /= qn;
            seq->rotation_quaternion_library[k][1] /= qn;
            seq->rotation_quaternion_library[k][2] /= qn;
            seq->rotation_quaternion_library[k][3] /= qn;
        }
    }

    if (seq->offsets.labelset >= 0) {
        ret = read_label_library(f, seq->offsets.labelset, seq->labelset_library,
                                 seq->labelset_library_size, 2, seq->is_label_defined);
        if (ret != 0) return;
    }
    if (seq->offsets.labelinc >= 0) {
        ret = read_label_library(f, seq->offsets.labelinc, seq->labelinc_library,
                                 seq->labelinc_library_size, 2, seq->is_label_defined);
        if (ret != 0) return;
    }
    if (seq->offsets.delays >= 0) {
        ret = read_delay_library(f, seq->offsets.delays, seq->soft_delay_library,
                                 seq->soft_delay_library_size, 4, seq->is_delay_defined);
        if (ret != 0) return;
    }
    if (seq->offsets.rfshim >= 0) {
        ret = read_rf_shim_library(f, seq->offsets.rfshim,
                                   seq->rf_shim_library, seq->rf_shim_library_size);
        if (ret != 0) return;
    }

    /* Build extension LUT */
    for (n = 0; n < 8; n++) {
        if (seq->extension_lut_size < seq->extension_map[n])
            seq->extension_lut_size = seq->extension_map[n];
    }
    if (seq->extension_lut_size > 0) {
        seq->extension_lut = (int*)PULSEQLIB_ALLOC(sizeof(int) * (seq->extension_lut_size + 1));
        for (n = 0; n < 8; n++) {
            if (seq->extension_map[n] > 0)
                seq->extension_lut[seq->extension_map[n]] = n;
        }
    }

    seq->is_extensions_library_parsed = 1;
}

/* ================================================================== */
/*  Shape decompression (cross-file)                                  */
/* ================================================================== */

int pulseqlib__decompress_shape(pulseqlib_shape_arbitrary* result,
                                const pulseqlib_shape_arbitrary* encoded, float scale)
{
    int i, rep;
    const float* packed;
    int num_packed, num_samples;
    int count_pack = 1;
    int count_unpack = 1;
    float* unpacked;

    if (!encoded || !result) return 0;

    packed      = encoded->samples;
    num_packed  = encoded->num_samples;
    num_samples = encoded->num_uncompressed_samples;

    /* Uncompressed — copy */
    if (encoded->num_samples == encoded->num_uncompressed_samples) {
        result->num_samples = encoded->num_samples;
        result->num_uncompressed_samples = encoded->num_uncompressed_samples;
        result->samples = (float*)PULSEQLIB_ALLOC(sizeof(float) * encoded->num_samples);
        if (!result->samples) return 0;
        for (i = 0; i < encoded->num_samples; ++i)
            result->samples[i] = encoded->samples[i] * scale;
        return 1;
    }

    unpacked = (float*)PULSEQLIB_ALLOC(sizeof(float) * num_samples);
    if (!unpacked) return 0;

    while (count_pack < num_packed) {
        if (packed[count_pack - 1] != packed[count_pack]) {
            unpacked[count_unpack - 1] = packed[count_pack - 1];
            count_pack++;
            count_unpack++;
        } else {
            rep = (int)(packed[count_pack + 1]) + 2;
            if (fabs(packed[count_pack + 1] + 2 - (float)rep) > 1e-6f) {
                PULSEQLIB_FREE(unpacked);
                return 0;
            }
            for (i = count_unpack - 1; i <= count_unpack + rep - 2; i++)
                unpacked[i] = packed[count_pack - 1];
            count_pack  += 3;
            count_unpack += rep;
        }
    }
    if (count_pack == num_packed)
        unpacked[count_unpack - 1] = packed[count_pack - 1];

    /* Cumulative sum */
    for (i = 1; i < num_samples; i++)
        unpacked[i] += unpacked[i - 1];

    /* Scale */
    for (i = 0; i < num_samples; i++)
        unpacked[i] *= scale;

    result->num_samples = num_samples;
    result->num_uncompressed_samples = num_samples;
    result->samples = unpacked;
    return 1;
}

/* ================================================================== */
/*  Opts init (public)                                                */
/* ================================================================== */

void pulseqlib_opts_init(pulseqlib_opts* opts,
                         float gamma, float b0,
                         float max_grad, float max_slew,
                         float rf_raster_time, float grad_raster_time,
                         float adc_raster_time, float block_duration_raster)
{
    if (!opts) return;
    opts->vendor = PULSEQLIB_VENDOR;
    opts->gamma_hz_per_t = gamma;
    opts->b0_t = b0;
    opts->max_grad_hz_per_m = max_grad;
    opts->max_slew_hz_per_m_per_s = max_slew;
    opts->rf_raster_us = rf_raster_time;
    opts->grad_raster_us = grad_raster_time;
    opts->adc_raster_us = adc_raster_time;
    opts->block_raster_us = block_duration_raster;
}

/* ================================================================== */
/*  seq_block init / free (cross-file)                                */
/* ================================================================== */

void pulseqlib__seq_block_init(pulseqlib__seq_block* block)
{
    if (!block) return;
    memset(block, 0, sizeof(*block));
    block->rf.type  = 0;
    block->gx.type  = 0;
    block->gy.type  = 0;
    block->gz.type  = 0;
    block->adc.type = 0;
    block->trigger.type  = 0;
    block->rotation.type = 0;
    block->delay.type    = 0;
    block->rf_shimming.type = 0;

    /* Flags default to -1 (undefined) */
    block->flag.trid  = -1;
    block->flag.nav   = -1;
    block->flag.rev   = -1;
    block->flag.sms   = -1;
    block->flag.ref   = -1;
    block->flag.ima   = -1;
    block->flag.noise = -1;
    block->flag.pmc   = -1;
    block->flag.norot = -1;
    block->flag.nopos = -1;
    block->flag.noscl = -1;
    block->flag.once  = -1;

    /* Labels default to 0 */
    memset(&block->labelset, 0, sizeof(block->labelset));
    memset(&block->labelinc, 0, sizeof(block->labelinc));
}

void pulseqlib__seq_block_free(pulseqlib__seq_block* block)
{
    if (!block) return;
    if (block->rf.type > 0) {
        if (block->rf.mag_shape.samples)   { PULSEQLIB_FREE(block->rf.mag_shape.samples);   block->rf.mag_shape.samples   = NULL; }
        if (block->rf.phase_shape.samples) { PULSEQLIB_FREE(block->rf.phase_shape.samples); block->rf.phase_shape.samples = NULL; }
        if (block->rf.time_shape.samples)  { PULSEQLIB_FREE(block->rf.time_shape.samples);  block->rf.time_shape.samples  = NULL; }
    }
    if (block->gx.type > 1) {
        if (block->gx.wave_shape.samples) { PULSEQLIB_FREE(block->gx.wave_shape.samples); block->gx.wave_shape.samples = NULL; }
        if (block->gx.time_shape.samples) { PULSEQLIB_FREE(block->gx.time_shape.samples); block->gx.time_shape.samples = NULL; }
    }
    if (block->gy.type > 1) {
        if (block->gy.wave_shape.samples) { PULSEQLIB_FREE(block->gy.wave_shape.samples); block->gy.wave_shape.samples = NULL; }
        if (block->gy.time_shape.samples) { PULSEQLIB_FREE(block->gy.time_shape.samples); block->gy.time_shape.samples = NULL; }
    }
    if (block->gz.type > 1) {
        if (block->gz.wave_shape.samples) { PULSEQLIB_FREE(block->gz.wave_shape.samples); block->gz.wave_shape.samples = NULL; }
        if (block->gz.time_shape.samples) { PULSEQLIB_FREE(block->gz.time_shape.samples); block->gz.time_shape.samples = NULL; }
    }
    if (block->adc.type > 0) {
        if (block->adc.phase_modulation_shape.samples) { PULSEQLIB_FREE(block->adc.phase_modulation_shape.samples); block->adc.phase_modulation_shape.samples = NULL; }
    }
    if (block->rf_shimming.type > 0) {
        if (block->rf_shimming.amplitudes) { PULSEQLIB_FREE(block->rf_shimming.amplitudes); block->rf_shimming.amplitudes = NULL; }
        if (block->rf_shimming.phases)     { PULSEQLIB_FREE(block->rf_shimming.phases);     block->rf_shimming.phases     = NULL; }
    }
}

/* ================================================================== */
/*  Read seq from buffer / file (cross-file)                          */
/* ================================================================== */

int pulseqlib__read_seq_from_buffer(pulseqlib__seq_file* seq, FILE* f)
{
    if (!seq || !f) return PULSEQLIB_ERR_NULL_POINTER;
    seq_file_reset(seq);
    if (seq->file_path) { PULSEQLIB_FREE(seq->file_path); seq->file_path = NULL; }

    get_section_offsets(seq, f);
    read_version(seq, f);
    if (seq->version_combined < 1005000) return PULSEQLIB_ERR_UNSUPPORTED_VERSION;
    read_definitions_library(seq, f);
    read_definitions(seq);
    read_block_library(seq, f);
    read_rf_library(seq, f);
    read_grad_library(seq, f);
    read_adc_library(seq, f);
    read_shapes_library(seq, f);
    read_extensions_library(seq, f);
    return PULSEQLIB_SUCCESS;
}

int pulseqlib__read_seq(pulseqlib__seq_file* seq, const char* file_path)
{
    FILE* f;
    int code;
    if (!seq || !file_path) return PULSEQLIB_ERR_INVALID_ARGUMENT;
    f = fopen(file_path, "r");
    if (!f) return PULSEQLIB_ERR_FILE_NOT_FOUND;
    code = pulseqlib__read_seq_from_buffer(seq, f);
    fclose(f);
    return code;
}

/* ================================================================== */
/*  Sequence collection (cross-file)                                  */
/* ================================================================== */

static int count_sequences_in_chain(const char* first_path, const pulseqlib_opts* opts)
{
    int count = 0;
    int max_count = 1000;
    char* current_path;
    char* base_path;
    pulseqlib__seq_file temp;
    int result;

    base_path = extract_base_path(first_path);
    if (!base_path) return -1;
    current_path = (char*)PULSEQLIB_ALLOC(strlen(first_path) + 1);
    if (!current_path) { PULSEQLIB_FREE(base_path); return -1; }
    strcpy(current_path, first_path);

    while (current_path && current_path[0] != '\0' && count < max_count) {
        pulseqlib__seq_file_init(&temp, opts);
        result = pulseqlib__read_seq(&temp, current_path);
        if (PULSEQLIB_FAILED(result)) {
            seq_file_reset(&temp);
            PULSEQLIB_FREE(current_path);
            PULSEQLIB_FREE(base_path);
            return -1;
        }
        count++;
        PULSEQLIB_FREE(current_path);
        current_path = NULL;
        if (temp.reserved_definitions_library.next_sequence[0] != '\0') {
            current_path = build_full_path(base_path,
                                           temp.reserved_definitions_library.next_sequence);
            if (!current_path) { seq_file_reset(&temp); PULSEQLIB_FREE(base_path); return -1; }
        }
        seq_file_reset(&temp);
    }
    PULSEQLIB_FREE(base_path);
    if (count >= max_count) return -1;
    return count;
}

int pulseqlib__read_seq_collection(pulseqlib__seq_file_collection* coll,
                                   const char* first_file_path,
                                   const pulseqlib_opts* opts)
{
    int num_seq, i, j, result;
    char* current_path;

    if (!coll || !first_file_path || !opts) return PULSEQLIB_ERR_NULL_POINTER;

    num_seq = count_sequences_in_chain(first_file_path, opts);
    if (num_seq <= 0) {
        coll->num_sequences = 0;
        coll->sequences = NULL;
        coll->base_path = NULL;
        return (num_seq == 0) ? PULSEQLIB_ERR_COLLECTION_EMPTY : PULSEQLIB_ERR_COLLECTION_CHAIN_BROKEN;
    }

    coll->base_path = extract_base_path(first_file_path);
    if (!coll->base_path) return PULSEQLIB_ERR_ALLOC_FAILED;

    coll->sequences = (pulseqlib__seq_file*)PULSEQLIB_ALLOC(num_seq * sizeof(pulseqlib__seq_file));
    if (!coll->sequences) { PULSEQLIB_FREE(coll->base_path); coll->base_path = NULL; return PULSEQLIB_ERR_ALLOC_FAILED; }

    current_path = (char*)PULSEQLIB_ALLOC(strlen(first_file_path) + 1);
    if (!current_path) {
        PULSEQLIB_FREE(coll->sequences); PULSEQLIB_FREE(coll->base_path);
        coll->sequences = NULL; coll->base_path = NULL;
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }
    strcpy(current_path, first_file_path);

    for (i = 0; i < num_seq; ++i) {
        pulseqlib__seq_file_init(&coll->sequences[i], opts);
        result = pulseqlib__read_seq(&coll->sequences[i], current_path);
        if (PULSEQLIB_FAILED(result)) {
            for (j = 0; j < i; ++j) seq_file_reset(&coll->sequences[j]);
            PULSEQLIB_FREE(coll->sequences); PULSEQLIB_FREE(current_path); PULSEQLIB_FREE(coll->base_path);
            coll->sequences = NULL; coll->base_path = NULL;
            return result;
        }
        /* store file path for later use (e.g. signature verification) */
        coll->sequences[i].file_path = (char*)PULSEQLIB_ALLOC(strlen(current_path) + 1);
        if (coll->sequences[i].file_path)
            strcpy(coll->sequences[i].file_path, current_path);
        if (i < num_seq - 1) {
            PULSEQLIB_FREE(current_path);
            current_path = build_full_path(coll->base_path,
                                           coll->sequences[i].reserved_definitions_library.next_sequence);
            if (!current_path) {
                for (j = 0; j <= i; ++j) seq_file_reset(&coll->sequences[j]);
                PULSEQLIB_FREE(coll->sequences); PULSEQLIB_FREE(coll->base_path);
                coll->sequences = NULL; coll->base_path = NULL;
                return PULSEQLIB_ERR_ALLOC_FAILED;
            }
        }
    }
    PULSEQLIB_FREE(current_path);
    coll->num_sequences = num_seq;
    return PULSEQLIB_SUCCESS;
}

/* ================================================================== */
/*  Raw block / extension parsing (public)                            */
/* ================================================================== */

int pulseqlib__get_raw_block_content_ids(const pulseqlib__seq_file* seq, pulseqlib__raw_block* block, int block_index, int parse_extensions)
{
    int next_ext_id, ext_count;
    float* ev;
    float* ext_data;

    if (!seq || !block || block_index < 0 || block_index >= seq->num_blocks) return 0;

    block->block_duration = 0;
    block->rf  = -1;
    block->gx  = -1;
    block->gy  = -1;
    block->gz  = -1;
    block->adc = -1;
    block->ext_count = 0;

    if (!seq->block_library || !seq->block_library[block_index]) return 0;

    ev = seq->block_library[block_index];
    block->block_duration = (int)ev[0];
    block->rf  = (int)ev[1] - 1;
    block->gx  = (int)ev[2] - 1;
    block->gy  = (int)ev[3] - 1;
    block->gz  = (int)ev[4] - 1;
    block->adc = (int)ev[5] - 1;

    if (!parse_extensions) { block->ext_count = 0; return 1; }
    if (!seq->is_extensions_library_parsed || !seq->extensions_library || seq->extensions_library_size <= 0)
        return 1;

    next_ext_id = (int)ev[6];
    ext_count = 0;
    while (next_ext_id > 0 && next_ext_id <= seq->extensions_library_size &&
           ext_count < PULSEQLIB__MAX_EXTENSIONS_PER_BLOCK) {
        ext_data = seq->extensions_library[next_ext_id - 1];
        block->ext[ext_count][0] = (int)ext_data[0];
        block->ext[ext_count][1] = (int)ext_data[1] - 1;
        next_ext_id = (int)ext_data[2];
        ext_count++;
    }
    block->ext_count = ext_count;
    return 1;
}

static void raw_extension_init(pulseqlib__raw_extension* re)
{
    if (!re) return;
    memset(&re->labelset, 0, sizeof(re->labelset));
    memset(&re->labelinc, 0, sizeof(re->labelinc));
    re->flag.trid  = -1; re->flag.nav   = -1; re->flag.rev   = -1;
    re->flag.sms   = -1; re->flag.ref   = -1; re->flag.ima   = -1;
    re->flag.noise = -1; re->flag.pmc   = -1; re->flag.norot = -1;
    re->flag.nopos = -1; re->flag.noscl = -1; re->flag.once  = -1;
    re->rotation_index   = -1;
    re->rf_shim_index    = -1;
    re->trigger_index    = -1;
    re->soft_delay_index = -1;
}

static void extension_block_init(pulseqlib__extension_block* eb)
{
    if (!eb) return;
    memset(&eb->labelset, 0, sizeof(eb->labelset));
    memset(&eb->labelinc, 0, sizeof(eb->labelinc));
    eb->flag.trid  = -1; eb->flag.nav   = -1; eb->flag.rev   = -1;
    eb->flag.sms   = -1; eb->flag.ref   = -1; eb->flag.ima   = -1;
    eb->flag.noise = -1; eb->flag.pmc   = -1; eb->flag.norot = -1;
    eb->flag.nopos = -1; eb->flag.noscl = -1; eb->flag.once  = -1;
    eb->rotation.type = 0;
    memset(&eb->rotation.data, 0, sizeof(eb->rotation.data));
    eb->rf_shimming.type = 0;
    eb->rf_shimming.num_channels = 0;
    eb->rf_shimming.amplitudes = NULL;
    eb->rf_shimming.phases = NULL;
    eb->trigger.type = 0;
    eb->trigger.duration = 0;
    eb->trigger.delay = 0;
    eb->trigger.trigger_type = 0;
    eb->trigger.trigger_channel = 0;
    eb->soft_delay.type = 0;
    eb->soft_delay.num_id = 0;
    eb->soft_delay.hint_id = 0;
    eb->soft_delay.offset = 0;
    eb->soft_delay.factor = 0;
}

static void extension_block_free(pulseqlib__extension_block* eb)
{
    if (!eb) return;
    if (eb->rf_shimming.amplitudes) { PULSEQLIB_FREE(eb->rf_shimming.amplitudes); eb->rf_shimming.amplitudes = NULL; }
    if (eb->rf_shimming.phases)     { PULSEQLIB_FREE(eb->rf_shimming.phases);     eb->rf_shimming.phases     = NULL; }
    eb->rf_shimming.type  = 0;
    eb->rf_shimming.num_channels = 0;
}

void pulseqlib__get_raw_extension(const pulseqlib__seq_file* seq, pulseqlib__raw_extension* re, const pulseqlib__raw_block* raw)
{
    int i, type_idx, ref_idx, ext_type, label_value, label_id;

    raw_extension_init(re);
    if (!seq || !re || !raw) return;
    if (!seq->is_extensions_library_parsed || !seq->extension_lut) return;

    for (i = 0; i < raw->ext_count; ++i) {
        type_idx = raw->ext[i][0];
        ref_idx  = raw->ext[i][1];
        if (type_idx < 0 || type_idx > seq->extension_lut_size) continue;
        ext_type = seq->extension_lut[type_idx];
        if (ref_idx < 0) continue;

        switch (ext_type) {
        case PULSEQLIB__EXT_LABELSET:
            if (seq->labelset_library && ref_idx < seq->labelset_library_size) {
                label_value = (int)seq->labelset_library[ref_idx][0];
                label_id    = (int)seq->labelset_library[ref_idx][1];
                switch (label_id) {
                    case PULSEQLIB__SLC: re->labelset.slc = label_value; break;
                    case PULSEQLIB__SEG: re->labelset.seg = label_value; break;
                    case PULSEQLIB__REP: re->labelset.rep = label_value; break;
                    case PULSEQLIB__AVG: re->labelset.avg = label_value; break;
                    case PULSEQLIB__SET: re->labelset.set = label_value; break;
                    case PULSEQLIB__ECO: re->labelset.eco = label_value; break;
                    case PULSEQLIB__PHS: re->labelset.phs = label_value; break;
                    case PULSEQLIB__LIN: re->labelset.lin = label_value; break;
                    case PULSEQLIB__PAR: re->labelset.par = label_value; break;
                    case PULSEQLIB__ACQ: re->labelset.acq = label_value; break;
                    case PULSEQLIB__NAV:   re->flag.nav   = label_value; break;
                    case PULSEQLIB__REV:   re->flag.rev   = label_value; break;
                    case PULSEQLIB__SMS:   re->flag.sms   = label_value; break;
                    case PULSEQLIB__REF:   re->flag.ref   = label_value; break;
                    case PULSEQLIB__IMA:   re->flag.ima   = label_value; break;
                    case PULSEQLIB__NOISE: re->flag.noise = label_value; break;
                    case PULSEQLIB__PMC:   re->flag.pmc   = label_value; break;
                    case PULSEQLIB__NOROT: re->flag.norot = label_value; break;
                    case PULSEQLIB__NOPOS: re->flag.nopos = label_value; break;
                    case PULSEQLIB__NOSCL: re->flag.noscl = label_value; break;
                    case PULSEQLIB__ONCE:  re->flag.once  = label_value; break;
                    case PULSEQLIB__TRID:  re->flag.trid  = label_value; break;
                    default: break;
                }
            }
            break;
        case PULSEQLIB__EXT_LABELINC:
            if (seq->labelinc_library && ref_idx < seq->labelinc_library_size) {
                label_value = (int)seq->labelinc_library[ref_idx][0];
                label_id    = (int)seq->labelinc_library[ref_idx][1];
                switch (label_id) {
                    case PULSEQLIB__SLC: re->labelinc.slc = label_value; break;
                    case PULSEQLIB__SEG: re->labelinc.seg = label_value; break;
                    case PULSEQLIB__REP: re->labelinc.rep = label_value; break;
                    case PULSEQLIB__AVG: re->labelinc.avg = label_value; break;
                    case PULSEQLIB__SET: re->labelinc.set = label_value; break;
                    case PULSEQLIB__ECO: re->labelinc.eco = label_value; break;
                    case PULSEQLIB__PHS: re->labelinc.phs = label_value; break;
                    case PULSEQLIB__LIN: re->labelinc.lin = label_value; break;
                    case PULSEQLIB__PAR: re->labelinc.par = label_value; break;
                    case PULSEQLIB__ACQ: re->labelinc.acq = label_value; break;
                    default: break;
                }
            }
            break;
        case PULSEQLIB__EXT_ROTATION: re->rotation_index   = ref_idx; break;
        case PULSEQLIB__EXT_RF_SHIM:  re->rf_shim_index    = ref_idx; break;
        case PULSEQLIB__EXT_TRIGGER:  re->trigger_index    = ref_idx; break;
        case PULSEQLIB__EXT_DELAY:    re->soft_delay_index = ref_idx; break;
        default: break;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Extension sub-parsers (static)                                    */
/* ------------------------------------------------------------------ */

static int parse_rotation_from_raw(const pulseqlib__seq_file* seq,
                                   pulseqlib__extension_block* eb,
                                   const pulseqlib__raw_extension* re)
{
    int i;
    int ref = re->rotation_index;
    if (ref < 0) return 1;
    if (seq->rotation_quaternion_library && ref < seq->rotation_library_size) {
        eb->rotation.type = 1;
        for (i = 0; i < 4; ++i)
            eb->rotation.data.rot_quaternion[i] = seq->rotation_quaternion_library[ref][i];
    }
    return 1;
}

static int parse_rf_shim_from_raw(const pulseqlib__seq_file* seq,
                                  pulseqlib__extension_block* eb,
                                  const pulseqlib__raw_extension* re)
{
    int ref = re->rf_shim_index;
    int i, n;
    const pulseqlib__rf_shim_entry* entry;
    float* amps;
    float* phs;

    if (ref < 0) return 1;
    if (!seq->rf_shim_library || ref >= seq->rf_shim_library_size) return 1;
    entry = &seq->rf_shim_library[ref];
    n = entry->num_channels;
    if (n <= 0 || n > PULSEQLIB__MAX_RF_SHIM_CHANNELS) return 1;

    amps = (float*)PULSEQLIB_ALLOC(sizeof(float) * n);
    phs  = (float*)PULSEQLIB_ALLOC(sizeof(float) * n);
    if (!amps || !phs) { if (amps) PULSEQLIB_FREE(amps); if (phs) PULSEQLIB_FREE(phs); return 0; }
    for (i = 0; i < n; ++i) { amps[i] = entry->values[2*i]; phs[i] = entry->values[2*i+1]; }
    eb->rf_shimming.type = 1;
    eb->rf_shimming.num_channels = n;
    eb->rf_shimming.amplitudes = amps;
    eb->rf_shimming.phases = phs;
    return 1;
}

static int parse_trigger_from_raw(const pulseqlib__seq_file* seq,
                                  pulseqlib__extension_block* eb,
                                  const pulseqlib__raw_extension* re)
{
    int ref = re->trigger_index;
    if (ref < 0) return 1;
    if (!seq->trigger_library || ref >= seq->trigger_library_size) return 1;
    eb->trigger.type            = 1;
    eb->trigger.trigger_type    = (int)seq->trigger_library[ref][0];
    eb->trigger.trigger_channel = (int)seq->trigger_library[ref][1];
    eb->trigger.delay           = (long)seq->trigger_library[ref][2];
    eb->trigger.duration        = (long)seq->trigger_library[ref][3];
    return 1;
}

static int parse_soft_delay_from_raw(const pulseqlib__seq_file* seq,
                                     pulseqlib__extension_block* eb,
                                     const pulseqlib__raw_extension* re)
{
    int ref = re->soft_delay_index;
    if (ref < 0) return 1;
    if (!seq->soft_delay_library || ref >= seq->soft_delay_library_size) return 1;
    eb->soft_delay.type    = 1;
    eb->soft_delay.num_id  = (int)seq->soft_delay_library[ref][0];
    eb->soft_delay.offset  = (int)seq->soft_delay_library[ref][1];
    eb->soft_delay.factor  = (int)seq->soft_delay_library[ref][2];
    eb->soft_delay.hint_id = (int)seq->soft_delay_library[ref][3];
    return 1;
}

static int parse_extension(const pulseqlib__seq_file* seq,
                           pulseqlib__extension_block* eb,
                           const pulseqlib__raw_extension* re)
{
    if (!eb) return 0;
    extension_block_init(eb);
    if (!seq || !re) return 1;
    eb->labelset = re->labelset;
    eb->labelinc = re->labelinc;
    eb->flag     = re->flag;
    if (!parse_rotation_from_raw(seq, eb, re))   { extension_block_free(eb); return 0; }
    if (!parse_rf_shim_from_raw(seq, eb, re))    { extension_block_free(eb); return 0; }
    if (!parse_trigger_from_raw(seq, eb, re))    { extension_block_free(eb); return 0; }
    if (!parse_soft_delay_from_raw(seq, eb, re)) { extension_block_free(eb); return 0; }
    return 1;
}

static void apply_extension(const pulseqlib__extension_block* eb,
                            pulseqlib__seq_block* block)
{
    int i, n;
    float* amps;
    float* phs;

    if (!eb || !block) return;
    block->labelset = eb->labelset;
    block->labelinc = eb->labelinc;
    block->flag     = eb->flag;
    block->rotation = eb->rotation;

    if (eb->rf_shimming.type && eb->rf_shimming.num_channels > 0) {
        n = eb->rf_shimming.num_channels;
        amps = (float*)PULSEQLIB_ALLOC(sizeof(float) * n);
        phs  = (float*)PULSEQLIB_ALLOC(sizeof(float) * n);
        if (amps && phs) {
            for (i = 0; i < n; ++i) { amps[i] = eb->rf_shimming.amplitudes[i]; phs[i] = eb->rf_shimming.phases[i]; }
            block->rf_shimming.type = 1;
            block->rf_shimming.num_channels = n;
            block->rf_shimming.amplitudes = amps;
            block->rf_shimming.phases = phs;
        } else {
            if (amps) PULSEQLIB_FREE(amps);
            if (phs)  PULSEQLIB_FREE(phs);
        }
    }
    block->trigger = eb->trigger;
    block->delay   = eb->soft_delay;
}

/* ================================================================== */
/*  Parse gradient helper (static, avoids repeating 3x)               */
/* ================================================================== */

static int parse_grad_event(const pulseqlib__seq_file* seq,
                            pulseqlib__grad_event* g, int raw_idx,
                            float grad_raster_us)
{
    float* fa;
    int idx;
    pulseqlib_shape_arbitrary shape;

    if (raw_idx < 0 || !seq->grad_library || raw_idx >= seq->grad_library_size) return 1;
    fa = seq->grad_library[raw_idx];
    g->amplitude = fa[1];

    if ((int)fa[0] == 0) {
        /* Trapezoid */
        g->type = 1;
        g->trap.rise_time = (long)fa[2];
        g->trap.flat_time = (long)fa[3];
        g->trap.fall_time = (long)fa[4];
        g->delay = (int)fa[5];
        g->first = 0;
        g->last  = 0;
    } else if ((int)fa[0] == 1) {
        /* Arbitrary */
        g->type  = 2;
        g->first = fa[2];
        g->last  = fa[3];
        idx = (int)fa[4];
        if (idx > 0 && seq->is_shapes_library_parsed && idx <= seq->shapes_library_size) {
            if (!pulseqlib__decompress_shape(&shape, &seq->shapes_library[idx - 1], 1.0f))
                return 0;
            g->wave_shape = shape;
        }
        idx = (int)fa[5];
        if (idx > 0 && seq->is_shapes_library_parsed && idx <= seq->shapes_library_size) {
            if (!pulseqlib__decompress_shape(&shape, &seq->shapes_library[idx - 1], grad_raster_us)) return 0;
            g->time_shape = shape;
        } else {
            g->time_shape.num_samples = 0;
            g->time_shape.num_uncompressed_samples = 0;
            g->time_shape.samples = NULL;
        }
        g->delay = (int)fa[6];
    }
    return 1;
}

/* ================================================================== */
/*  Parse block without extensions (static)                           */
/* ================================================================== */

static int parse_block_without_ext(const pulseqlib__seq_file* seq,
                                   pulseqlib__seq_block* block,
                                   const pulseqlib__raw_block* raw,
                                   const pulseqlib__raw_extension* re)
{
    float* fa;
    int idx, i, num_real;
    int* is_real;
    float block_raster_us, rf_raster_us, grad_raster_us;
    float* trig;
    float* dl;
    pulseqlib_shape_arbitrary shape;

    if (!seq || !raw || !block) return 0;

    block_raster_us = (seq->reserved_definitions_library.block_duration_raster > 0.0f)
        ? seq->reserved_definitions_library.block_duration_raster
        : seq->opts.block_raster_us;
    rf_raster_us = (seq->reserved_definitions_library.radiofrequency_raster_time > 0.0f)
        ? seq->reserved_definitions_library.radiofrequency_raster_time
        : seq->opts.rf_raster_us;
    grad_raster_us = (seq->reserved_definitions_library.gradient_raster_time > 0.0f)
        ? seq->reserved_definitions_library.gradient_raster_time
        : seq->opts.grad_raster_us;

    block->duration = raw->block_duration * (int)block_raster_us;

    /* RF */
    if (raw->rf >= 0 && seq->rf_library && raw->rf < seq->rf_library_size) {
        num_real = 0;
        fa = seq->rf_library[raw->rf];
        block->rf.type = 1;
        block->rf.amplitude = fa[0];

        /* Mag shape */
        idx = (int)fa[1];
        if (idx > 0 && seq->is_shapes_library_parsed && idx <= seq->shapes_library_size) {
            if (!pulseqlib__decompress_shape(&shape, &seq->shapes_library[idx - 1], 1.0f))
                goto fail;
            block->rf.mag_shape = shape;
        } else {
            block->rf.mag_shape.num_samples = 0;
            block->rf.mag_shape.num_uncompressed_samples = 0;
            block->rf.mag_shape.samples = NULL;
        }

        /* Phase shape */
        idx = (int)fa[2];
        if (idx > 0 && seq->is_shapes_library_parsed && idx <= seq->shapes_library_size) {
            if (!pulseqlib__decompress_shape(&shape, &seq->shapes_library[idx - 1], 1.0f))
                goto fail;
            block->rf.phase_shape = shape;
            for (i = 0; i < block->rf.phase_shape.num_samples; i++)
                block->rf.phase_shape.samples[i] *= (float)PULSEQLIB__TWO_PI;
        } else {
            block->rf.phase_shape.num_samples = 0;
            block->rf.phase_shape.num_uncompressed_samples = 0;
            block->rf.phase_shape.samples = NULL;
        }

        /* Detect and fold real-valued RF pulses */
        if (block->rf.mag_shape.num_samples > 0 && block->rf.phase_shape.num_samples > 0) {
            is_real = (int*)PULSEQLIB_ALLOC(block->rf.mag_shape.num_samples * sizeof(int));
            if (!is_real) goto fail;
            for (i = 0; i < block->rf.mag_shape.num_samples; i++)
                is_real[i] = (float)fabs(block->rf.phase_shape.samples[i]) < 1e-6f ||
                             (float)fabs(block->rf.phase_shape.samples[i] - M_PI) < 1e-6f;
            for (i = 0; i < block->rf.mag_shape.num_samples; i++)
                if (is_real[i]) num_real++;
            if (num_real == block->rf.mag_shape.num_samples) {
                for (i = 0; i < block->rf.mag_shape.num_samples; i++)
                    if ((float)fabs(block->rf.phase_shape.samples[i] - M_PI) < 1e-6f)
                        block->rf.mag_shape.samples[i] *= -1;
                PULSEQLIB_FREE(block->rf.phase_shape.samples);
                block->rf.phase_shape.num_samples = 0;
                block->rf.phase_shape.num_uncompressed_samples = 0;
                block->rf.phase_shape.samples = NULL;
            }
            PULSEQLIB_FREE(is_real);
        }

        /* Time shape */
        idx = (int)fa[3];
        if (idx > 0 && seq->is_shapes_library_parsed && idx <= seq->shapes_library_size) {
            if (!pulseqlib__decompress_shape(&shape, &seq->shapes_library[idx - 1], rf_raster_us))
                goto fail;
            block->rf.time_shape = shape;
        } else {
            block->rf.time_shape.num_samples = 0;
            block->rf.time_shape.num_uncompressed_samples = 0;
            block->rf.time_shape.samples = NULL;
        }

        block->rf.center       = fa[4];
        block->rf.delay        = (int)fa[5];
        block->rf.freq_ppm     = fa[6];
        block->rf.phase_ppm    = fa[7];
        block->rf.freq_offset  = fa[8];
        block->rf.phase_offset = fa[9];
    }

    /* Gradients */
    if (!parse_grad_event(seq, &block->gx, raw->gx, grad_raster_us)) goto fail;
    if (!parse_grad_event(seq, &block->gy, raw->gy, grad_raster_us)) goto fail;
    if (!parse_grad_event(seq, &block->gz, raw->gz, grad_raster_us)) goto fail;

    /* ADC */
    if (raw->adc >= 0 && seq->adc_library && raw->adc < seq->adc_library_size) {
        fa = seq->adc_library[raw->adc];
        block->adc.type         = 1;
        block->adc.num_samples  = (int)fa[0];
        block->adc.dwell_time   = (int)fa[1];
        block->adc.delay        = (int)fa[2];
        block->adc.freq_ppm     = fa[3];
        block->adc.phase_ppm    = fa[4];
        block->adc.freq_offset  = fa[5];
        block->adc.phase_offset = fa[6];
        idx = (int)fa[7];
        if (idx > 0 && seq->is_shapes_library_parsed && idx <= seq->shapes_library_size) {
            if (!pulseqlib__decompress_shape(&shape, &seq->shapes_library[idx - 1], fa[1] * 1e-3f))
                goto fail;
            block->adc.phase_modulation_shape = shape;
        } else {
            block->adc.phase_modulation_shape.num_samples = 0;
            block->adc.phase_modulation_shape.num_uncompressed_samples = 0;
            block->adc.phase_modulation_shape.samples = NULL;
        }
    }

    /* Trigger / delay from raw extension */
    if (re) {
        if (re->trigger_index >= 0 && seq->trigger_library) {
            trig = seq->trigger_library[re->trigger_index];
            block->trigger.type            = 1;
            block->trigger.trigger_type    = (int)trig[0];
            block->trigger.trigger_channel = (int)trig[1];
            block->trigger.delay           = (long)trig[2];
            block->trigger.duration        = (long)trig[3];
        }
        if (re->soft_delay_index >= 0 && seq->soft_delay_library) {
            dl = seq->soft_delay_library[re->soft_delay_index];
            block->delay.type    = 1;
            block->delay.num_id  = (int)dl[0];
            block->delay.offset  = (int)dl[1];
            block->delay.factor  = (int)dl[2];
            block->delay.hint_id = (int)dl[3];
        }
    }
    return 1;

fail:
    pulseqlib__seq_block_free(block);
    pulseqlib__seq_block_init(block);
    return 0;
}

/* ================================================================== */
/*  get_block (cross-file)                                            */
/* ================================================================== */

void pulseqlib__get_block(const pulseqlib__seq_file* seq,
                          pulseqlib__seq_block* block, int block_index)
{
    pulseqlib__raw_block raw;
    pulseqlib__raw_extension re;
    pulseqlib__extension_block eb;
    int has_ext;

    if (!seq || !block || block_index < 0 || block_index >= seq->num_blocks) return;

    if (!pulseqlib__get_raw_block_content_ids(seq, &raw, block_index, 1)) return;
    has_ext = (raw.ext_count > 0) && seq->is_extensions_library_parsed && seq->extension_lut;

    if (has_ext) pulseqlib__get_raw_extension(seq, &re, &raw);
    else         raw_extension_init(&re);

    if (!parse_block_without_ext(seq, block, &raw, has_ext ? &re : NULL)) return;

    if (has_ext) {
        if (parse_extension(seq, &eb, &re)) {
            apply_extension(&eb, block);
            extension_block_free(&eb);
        }
    }
}

/* ================================================================== */
/*  Grad library max amplitude (cross-file)                           */
/* ================================================================== */

float pulseqlib__get_grad_library_max_amplitude(const pulseqlib__seq_file* seq)
{
    float max_amp = 0.0f;
    int i;
    float amp;

    if (!seq || !seq->is_grad_library_parsed || !seq->grad_library || seq->grad_library_size <= 0)
        return 0.0f;

    for (i = 0; i < seq->grad_library_size; ++i) {
        amp = seq->grad_library[i][1];
        if (amp > max_amp) max_amp = amp;
    }
    return max_amp;
}

/* ================================================================== */
/*  MD5 signature verification                                        */
/* ================================================================== */

int pulseqlib__verify_signature(const char* file_path)
{
    FILE* f;
    long sig_offset, hash_start;
    char line[PULSEQLIB__MAX_LINE_LENGTH];
    char stored_hash[33];
    unsigned char digest[16];
    struct MD5Context ctx;
    unsigned char buf[4096];
    size_t to_read, n;
    long remaining;
    char computed[33];
    int i;
    char* p;

    if (!file_path) return PULSEQLIB_ERR_INVALID_ARGUMENT;
    f = fopen(file_path, "rb");
    if (!f) return PULSEQLIB_ERR_FILE_NOT_FOUND;

    /* locate [SIGNATURE] by scanning from start */
    sig_offset = -1;
    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (strncmp(p, "[SIGNATURE]", 11) == 0) {
            sig_offset = ftell(f) - (long)strlen(line);
            break;
        }
    }
    if (sig_offset < 0) { fclose(f); return PULSEQLIB_ERR_SIGNATURE_MISSING; }

    /* read stored hash from section body */
    stored_hash[0] = '\0';
    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0') continue;
        if (strncmp(p, "Type", 4) == 0) continue;
        if (strncmp(p, "Hash", 4) == 0) {
            p += 4;
            while (*p == ' ' || *p == '\t') p++;
            strncpy(stored_hash, p, 32);
            stored_hash[32] = '\0';
            /* strip trailing whitespace */
            for (i = 31; i >= 0 && (stored_hash[i] == ' ' || stored_hash[i] == '\n' ||
                                     stored_hash[i] == '\r'); --i)
                stored_hash[i] = '\0';
            break;
        }
    }
    if (stored_hash[0] == '\0') { fclose(f); return PULSEQLIB_ERR_SIGNATURE_MISSING; }

    /*
     * Hash bytes [0 .. sig_offset - 2] inclusive.
     * The newline preceding [SIGNATURE] belongs to the signature
     * and must be stripped, so we hash up to sig_offset - 1 bytes
     * (i.e. the \n at position sig_offset-1 is excluded).
     */
    hash_start = sig_offset - 1;
    if (hash_start <= 0) { fclose(f); return PULSEQLIB_ERR_SIGNATURE_MISSING; }

    MD5Init(&ctx);
    if (fseek(f, 0L, SEEK_SET) != 0) { fclose(f); return PULSEQLIB_ERR_FILE_READ_FAILED; }
    remaining = hash_start;
    while (remaining > 0) {
        to_read = (remaining > (long)sizeof(buf)) ? sizeof(buf) : (size_t)remaining;
        n = fread(buf, 1, to_read, f);
        if (n == 0) { fclose(f); return PULSEQLIB_ERR_FILE_READ_FAILED; }
        MD5Update(&ctx, buf, (unsigned)n);
        remaining -= (long)n;
    }
    MD5Final(digest, &ctx);
    fclose(f);

    /* format computed hash as hex string */
    for (i = 0; i < 16; ++i)
        sprintf(computed + i * 2, "%02x", digest[i]);
    computed[32] = '\0';

    /* compare */
    if (strcmp(computed, stored_hash) != 0)
        return PULSEQLIB_ERR_SIGNATURE_MISMATCH;

    return PULSEQLIB_SUCCESS;
}

/* ================================================================== */
/*  Lightweight scan-time query                                       */
/* ================================================================== */

/*
 * Open a single .seq file and parse ONLY version + definitions.
 * Fills a temporary seq_file just enough to read reserved definitions.
 */
static int read_definitions_only(pulseqlib__seq_file* seq, const char* path)
{
    FILE* f;
    if (!seq || !path) return PULSEQLIB_ERR_INVALID_ARGUMENT;
    f = fopen(path, "r");
    if (!f) return PULSEQLIB_ERR_FILE_NOT_FOUND;

    get_section_offsets(seq, f);
    read_version(seq, f);
    if (seq->version_combined < 1005000) { fclose(f); return PULSEQLIB_ERR_UNSUPPORTED_VERSION; }
    read_definitions_library(seq, f);
    read_definitions(seq);
    fclose(f);
    return PULSEQLIB_SUCCESS;
}

int pulseqlib_peek_scan_time(
    pulseqlib_scan_time_info* info,
    const char* file_path,
    const pulseqlib_opts* opts,
    int num_reps)
{
    int count = 0;
    int max_depth = 1000;
    char* current_path;
    char* base_path;
    pulseqlib__seq_file temp;
    int result;
    int navg;

    if (!info || !file_path || !opts) return PULSEQLIB_ERR_NULL_POINTER;
    if (num_reps < 1) return PULSEQLIB_ERR_INVALID_ARGUMENT;
    info->total_duration_us = 0.0f;
    info->total_segment_boundaries = 0;

    base_path = extract_base_path(file_path);
    if (!base_path) return PULSEQLIB_ERR_ALLOC_FAILED;

    current_path = (char*)PULSEQLIB_ALLOC(strlen(file_path) + 1);
    if (!current_path) { PULSEQLIB_FREE(base_path); return PULSEQLIB_ERR_ALLOC_FAILED; }
    strcpy(current_path, file_path);

    while (current_path && current_path[0] != '\0' && count < max_depth) {
        pulseqlib__seq_file_init(&temp, opts);
        result = read_definitions_only(&temp, current_path);
        if (PULSEQLIB_FAILED(result)) {
            seq_file_reset(&temp);
            PULSEQLIB_FREE(current_path); PULSEQLIB_FREE(base_path);
            return result;
        }

        navg = temp.reserved_definitions_library.ignore_averages ? 1 : num_reps;
        info->total_duration_us +=
            temp.reserved_definitions_library.total_duration * 1e6f * (float)navg;
        count++;

        PULSEQLIB_FREE(current_path);
        current_path = NULL;
        if (temp.reserved_definitions_library.next_sequence[0] != '\0') {
            current_path = build_full_path(base_path,
                                           temp.reserved_definitions_library.next_sequence);
            if (!current_path) { seq_file_reset(&temp); PULSEQLIB_FREE(base_path); return PULSEQLIB_ERR_ALLOC_FAILED; }
        }
        seq_file_reset(&temp);
    }

    PULSEQLIB_FREE(current_path);
    PULSEQLIB_FREE(base_path);
    return (count > 0) ? PULSEQLIB_SUCCESS : PULSEQLIB_ERR_COLLECTION_EMPTY;
}
