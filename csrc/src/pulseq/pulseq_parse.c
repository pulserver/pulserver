/**
 * @file pulseq_parse.c
 * @brief The Pulseq .seq reader: section scan, library decode, shape codec.
 *
 * Produces the raw model only -- library rows exactly as written, blocks as
 * unresolved content ids. Interpretation (dedup, segmentation, units) is the
 * caller's job, which is what keeps this module free of any pulseg header.
 */

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "external_md5.h"

#include "pulseq_internal.h"

/* ================================================================== */
/*  Internal macro                                                    */
/* ================================================================== */
#define INIT_LIBRARY(seq, field_ptr, size_field, flag_field) \
    do \
    { \
        (seq)->field_ptr = NULL; \
        (seq)->size_field = 0; \
        (seq)->flag_field = 0; \
    } while (0)

/* ================================================================== */
/*  Path helpers (static)                                             */
/* ================================================================== */

char *pulseq_path_dirname(const char *file_path)
{
    char *base;
    const char *last_slash;
    size_t len;

    last_slash = strrchr(file_path, '/');
    if (!last_slash)
        last_slash = strrchr(file_path, '\\');

    if (last_slash)
    {
        len = (size_t)(last_slash - file_path + 1);
        base = (char *)PULSEQ_ALLOC(len + 1);
        if (base)
        {
            strncpy(base, file_path, len);
            base[len] = '\0';
        }
    }
    else
    {
        base = (char *)PULSEQ_ALLOC(3);
        if (base)
            strcpy(base, "./");
    }
    return base;
}

char *pulseq_path_join(const char *base_path, const char *filename)
{
    size_t len;
    char *full;

    len = strlen(base_path) + strlen(filename) + 1;
    full = (char *)PULSEQ_ALLOC(len);
    if (full)
    {
        strcpy(full, base_path);
        strcat(full, filename);
    }
    return full;
}

/* ================================================================== */
/*  seq_file defaults / init / reset / free                           */
/* ================================================================== */

static void seq_file_set_defaults(pulseq_file *seq)
{
    int i;
    if (!seq)
        return;

    seq->offsets.scan_cursor = 0;
    seq->offsets.version = -1;
    seq->offsets.definitions = -1;
    seq->offsets.blocks = -1;
    seq->offsets.rf = -1;
    seq->offsets.grad = -1;
    seq->offsets.trap = -1;
    seq->offsets.adc = -1;
    seq->offsets.extensions = -1;
    seq->offsets.triggers = -1;
    seq->offsets.rfshim = -1;
    seq->offsets.labelset = -1;
    seq->offsets.labelinc = -1;
    seq->offsets.delays = -1;
    seq->offsets.rotations = -1;
    seq->offsets.shapes = -1;
    seq->offsets.signature = -1;

    seq->is_version_parsed = 0;
    seq->version_combined = 0;
    seq->version_major = 0;
    seq->version_minor = 0;
    seq->version_revision = 0;

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
    INIT_LIBRARY(
        seq,
        rotation_quaternion_library,
        rotation_library_size,
        is_extensions_library_parsed);
    INIT_LIBRARY(seq, rotation_matrix_library, rotation_library_size, is_extensions_library_parsed);
    INIT_LIBRARY(seq, labelset_library, labelset_library_size, is_extensions_library_parsed);
    INIT_LIBRARY(seq, labelinc_library, labelinc_library_size, is_extensions_library_parsed);
    for (i = 0; i < 22; i++)
        seq->is_label_defined[i] = 0;
    memset(&seq->label_limits, 0, sizeof(seq->label_limits));
    for (i = 0; i < 8; i++)
    {
        seq->is_delay_defined[i] = 0;
        seq->extension_map[i] = -1;
    }
    INIT_LIBRARY(seq, soft_delay_library, soft_delay_library_size, is_extensions_library_parsed);
    INIT_LIBRARY(seq, rf_shim_library, rf_shim_library_size, is_extensions_library_parsed);
    seq->extension_lut_size = 0;
    seq->extension_lut = NULL;
    INIT_LIBRARY(seq, shapes_library, shapes_library_size, is_shapes_library_parsed);
}

void pulseq_file_init(pulseq_file *seq, const pulseq_raster *raster)
{
    if (!seq)
        return;
    seq->file_path = NULL;
    if (raster)
    {
        seq->design_raster = *raster;
    }
    else
    {
        seq->design_raster.rf_us = 0.0f;
        seq->design_raster.grad_us = 0.0f;
        seq->design_raster.block_us = 0.0f;
    }
    seq_file_set_defaults(seq);
}

static void seq_file_reset(pulseq_file *seq)
{
    int i, j;
    if (!seq)
        return;

    if (seq->is_definitions_library_parsed && seq->definitions_library)
    {
        for (i = 0; i < seq->num_definitions; i++)
        {
            for (j = 0; j < seq->definitions_library[i].value_size; j++)
                PULSEQ_FREE(seq->definitions_library[i].value[j]);
            PULSEQ_FREE(seq->definitions_library[i].value);
        }
        PULSEQ_FREE(seq->definitions_library);
    }
    if (seq->is_block_library_parsed)
    {
        PULSEQ_FREE(seq->block_library);
        PULSEQ_FREE(seq->block_ids);
        seq->block_ids = NULL;
    }
    if (seq->is_rf_library_parsed)
    {
        PULSEQ_FREE(seq->rf_library);
        if (seq->rf_use_tags)
            PULSEQ_FREE(seq->rf_use_tags);
        seq->rf_use_tags = NULL;
    }
    if (seq->is_grad_library_parsed)
        PULSEQ_FREE(seq->grad_library);
    if (seq->is_adc_library_parsed)
        PULSEQ_FREE(seq->adc_library);
    if (seq->is_extensions_library_parsed)
    {
        PULSEQ_FREE(seq->extensions_library);
        PULSEQ_FREE(seq->trigger_library);
        PULSEQ_FREE(seq->rotation_quaternion_library);
        PULSEQ_FREE(seq->rotation_matrix_library);
        PULSEQ_FREE(seq->labelset_library);
        PULSEQ_FREE(seq->labelinc_library);
        PULSEQ_FREE(seq->soft_delay_library);
        PULSEQ_FREE(seq->rf_shim_library);
    }
    if (seq->is_shapes_library_parsed && seq->shapes_library)
    {
        for (i = 0; i < seq->shapes_library_size; i++)
        {
            PULSEQ_FREE(seq->shapes_library[i].samples);
            seq->shapes_library[i].samples = NULL;
            seq->shapes_library[i].num_uncompressed_samples = 0;
            seq->shapes_library[i].num_samples = 0;
        }
        PULSEQ_FREE(seq->shapes_library);
    }
    PULSEQ_FREE(seq->extension_lut);
    seq->extension_lut = NULL;

    seq_file_set_defaults(seq);
}

void pulseq_file_free(pulseq_file *seq)
{
    if (!seq)
        return;
    seq_file_reset(seq);
    if (seq->file_path)
    {
        PULSEQ_FREE(seq->file_path);
        seq->file_path = NULL;
    }
    memset(&seq->design_raster, 0, sizeof(seq->design_raster));
}

void pulseq_file_set_free(pulseq_file_set *coll)
{
    int i;
    if (!coll)
        return;
    if (coll->sequences)
    {
        for (i = 0; i < coll->num_sequences; ++i)
        {
            seq_file_reset(&coll->sequences[i]);
            if (coll->sequences[i].file_path)
            {
                PULSEQ_FREE(coll->sequences[i].file_path);
                coll->sequences[i].file_path = NULL;
            }
        }
        PULSEQ_FREE(coll->sequences);
    }
    if (coll->base_path)
        PULSEQ_FREE(coll->base_path);
    coll->num_sequences = 0;
    coll->sequences = NULL;
    coll->base_path = NULL;
}

/* ================================================================== */
/*  Library init helpers (static)                                     */
/* ================================================================== */

static int init_standard_library(
    void **target,
    int *target_count,
    FILE *f,
    const long *offsets,
    int num_sections,
    int n)
{
    char line[PULSEQ_MAX_LINE_LENGTH];
    int max_idx = -1;
    int sec, idx, i;
    char *p;
    float *arr;

    if (!f)
        return 1;
    for (sec = 0; sec < num_sections; sec++)
    {
        if (offsets[sec] < 0)
            continue;
        if (fseek(f, offsets[sec], SEEK_SET) != 0)
            return 1;
        if (!fgets(line, sizeof(line), f))
            return 1;
        while (fgets(line, sizeof(line), f))
        {
            p = line;
            while (*p == ' ' || *p == '\t')
                p++;
            if (*p == '[' || *p == 'e')
                break;
            if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#')
                continue;
            if (sscanf(p, "%d", &idx) == 1 && idx > max_idx)
                max_idx = idx;
        }
    }
    if (max_idx <= 0)
    {
        *target = NULL;
        *target_count = 0;
        return 0;
    }
    arr = (float *)PULSEQ_ALLOC(sizeof(float) * n * max_idx);
    if (!arr)
        return 1;
    for (i = 0; i < max_idx * n; i++)
        arr[i] = 0.0f;
    *target = (void *)arr;
    *target_count = max_idx;
    return 0;
}

static int init_definitions_library(
    pulseq_definition **target,
    int *target_count,
    FILE *f,
    long offset)
{
    char line[PULSEQ_MAX_LINE_LENGTH];
    int count = 0;
    char *p;
    char *name_tok;
    pulseq_definition *defs;

    if (!f || offset < 0 || !target || !target_count)
        return 1;
    if (fseek(f, offset, SEEK_SET) != 0)
        return 1;
    if (!fgets(line, sizeof(line), f))
        return 1;
    while (fgets(line, sizeof(line), f))
    {
        p = line;
        while (isspace((unsigned char)*p))
            p++;
        if (*p == '[' || *p == 'e')
            break;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#')
            continue;
        name_tok = strtok(p, " \t\r\n");
        if (name_tok)
            count++;
    }
    if (count == 0)
    {
        *target = NULL;
        *target_count = 0;
        return 1;
    }
    defs = (pulseq_definition *)PULSEQ_ALLOC(sizeof(pulseq_definition) * count);
    if (!defs)
        return 1;
    *target = defs;
    *target_count = count;
    return 0;
}

static int init_shapes_library(pulseq_shape **target, int *target_count, FILE *f, long offset)
{
    char line[PULSEQ_MAX_LINE_LENGTH];
    int max_idx = -1;
    int n, i, j, idx = 0;
    char *p;
    pulseq_shape *shapes;

    if (!f || !offset || !target || !target_count)
        return 1;
    if (fseek(f, offset, SEEK_SET) != 0)
        return 1;
    if (!fgets(line, sizeof(line), f))
        return 1;

    /* Pass 1: find max shape_id */
    while (fgets(line, sizeof(line), f))
    {
        p = line;
        while (*p == ' ' || *p == '\t')
            p++;
        if (*p == '[' || *p == 'e')
            break;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#')
            continue;
        if (strncmp(p, "shape_id", 8) == 0)
        {
            if (sscanf(p + 8, "%d", &idx) == 1 && idx > max_idx)
                max_idx = idx;
        }
    }
    if (max_idx <= 0)
    {
        *target = NULL;
        *target_count = 0;
        return 0;
    }

    shapes = (pulseq_shape *)PULSEQ_ALLOC(sizeof(pulseq_shape) * max_idx);
    if (!shapes)
        return 1;
    for (i = 0; i < max_idx; i++)
    {
        shapes[i].num_samples = 0;
        shapes[i].num_uncompressed_samples = 0;
        shapes[i].samples = NULL;
    }

    /* Pass 2: count samples per shape */
    if (fseek(f, offset, SEEK_SET) != 0)
    {
        PULSEQ_FREE(shapes);
        return 1;
    }
    if (!fgets(line, sizeof(line), f))
        return 1;
    idx = 0;
    while (fgets(line, sizeof(line), f))
    {
        p = line;
        while (*p == ' ' || *p == '\t')
            p++;
        if (*p == '[' || *p == 'e')
            break;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#')
            continue;
        if (strncmp(p, "shape_id", 8) == 0)
        {
            if (sscanf(p + 8, "%d", &idx) == 1)
            {
                shapes[idx - 1].num_samples = 0;
                shapes[idx - 1].num_uncompressed_samples = 0;
                shapes[idx - 1].samples = NULL;
            }
        }
        else if (strncmp(p, "num_samples", 11) == 0)
        {
            if (sscanf(p + 11, "%d", &n) == 1)
                shapes[idx - 1].num_uncompressed_samples = n;
        }
        else
        {
            shapes[idx - 1].num_samples++;
        }
    }

    /* Allocate sample arrays */
    for (i = 0; i < max_idx; i++)
    {
        n = shapes[i].num_samples;
        if (n > 0)
        {
            shapes[i].samples = (float *)PULSEQ_ALLOC(sizeof(float) * n);
            if (!shapes[i].samples)
            {
                for (j = 0; j < i; j++)
                {
                    if (shapes[j].samples)
                        PULSEQ_FREE(shapes[j].samples);
                }
                PULSEQ_FREE(shapes);
                return 1;
            }
        }
    }

    *target = shapes;
    *target_count = max_idx;
    return 0;
}

static int init_rf_shim_library(
    pulseq_rf_shim_entry **target,
    int *target_count,
    FILE *f,
    long offset)
{
    char line[PULSEQ_MAX_LINE_LENGTH];
    int max_idx = -1;
    char *p;
    int idx, i;
    pulseq_rf_shim_entry *arr;

    if (!f || !target || !target_count)
        return 1;
    if (fseek(f, offset, SEEK_SET) != 0)
        return 1;
    if (!fgets(line, sizeof(line), f))
        return 1;
    while (fgets(line, sizeof(line), f))
    {
        p = line;
        while (*p == ' ' || *p == '\t')
            p++;
        if (*p == '[' || *p == 'e')
            break;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#')
            continue;
        if (sscanf(p, "%d", &idx) == 1 && idx > max_idx)
            max_idx = idx;
    }
    if (max_idx <= 0)
    {
        *target = NULL;
        *target_count = 0;
        return 1;
    }
    arr = (pulseq_rf_shim_entry *)PULSEQ_ALLOC(sizeof(pulseq_rf_shim_entry) * max_idx);
    if (!arr)
        return 1;
    for (i = 0; i < max_idx; i++)
        arr[i].num_channels = 0;
    *target = arr;
    *target_count = max_idx;
    return 0;
}

/* ================================================================== */
/*  Library read helpers (static)                                     */
/* ================================================================== */

static int read_standard_library(
    void *target,
    int target_count,
    FILE *f,
    long offset,
    int n,
    pulseq_scale scale,
    int flag)
{
    char line[PULSEQ_MAX_LINE_LENGTH];
    int idx, parsed, consumed, col, offset_col;
    float vals[PULSEQ_MAX_SCALE_SIZE];
    char *scan_ptr;
    char *p;
    float v;
    float *arr = (float *)target;

    if (!f)
        return 1;
    if (scale.size > PULSEQ_MAX_SCALE_SIZE)
        return 1;
    if (fseek(f, offset, SEEK_SET) != 0)
        return 1;
    if (!fgets(line, sizeof(line), f))
        return 1;

    while (fgets(line, sizeof(line), f))
    {
        p = line;
        while (*p == ' ' || *p == '\t')
            p++;
        if (*p == '[' || *p == 'e')
            break;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#')
            continue;
        if (sscanf(p, "%d", &idx) != 1)
            continue;
        if (idx <= 0 || idx > target_count)
            continue;

        while (*p && *p != ' ' && *p != '\t')
            p++;
        while (*p == ' ' || *p == '\t')
            p++;

        parsed = 0;
        scan_ptr = p;
        for (col = 0; col < scale.size; col++)
        {
            consumed = 0;
            if (sscanf(scan_ptr, "%f%n", &v, &consumed) != 1)
                break;
            vals[col] = v;
            scan_ptr += consumed;
            while (*scan_ptr == ' ' || *scan_ptr == '\t')
                scan_ptr++;
            parsed++;
        }
        if (parsed != scale.size)
            continue;

        offset_col = (flag >= 0) ? 1 : 0;
        for (col = 0; col < scale.size; col++)
            arr[(idx - 1) * n + col + offset_col] = vals[col] * scale.values[col];
        if (flag >= 0)
            arr[(idx - 1) * n + 0] = (float)flag;
    }
    return 0;
}

static void get_section_offsets(pulseq_file *seq, FILE *f)
{
    char line[PULSEQ_MAX_LINE_LENGTH];
    char *p;
    long pos;
    char ext_name[PULSEQ_EXT_NAME_LENGTH];
    int ext_id, ext_enum;

    if (fseek(f, 0L, SEEK_SET) != 0)
        return;

    while (fgets(line, sizeof(line), f))
    {
        pos = ftell(f);
        if (pos < 0)
            break;
        p = line;
        while (*p == ' ' || *p == '\t')
            p++;

        if (*p == '[')
        {
            if (strncmp(p, "[VERSION]", 9) == 0)
                seq->offsets.version = pos - (long)strlen(line);
            else if (strncmp(p, "[DEFINITIONS]", 13) == 0)
                seq->offsets.definitions = pos - (long)strlen(line);
            else if (strncmp(p, "[BLOCKS]", 8) == 0)
                seq->offsets.blocks = pos - (long)strlen(line);
            else if (strncmp(p, "[RF]", 4) == 0)
                seq->offsets.rf = pos - (long)strlen(line);
            else if (strncmp(p, "[GRADIENTS]", 11) == 0)
                seq->offsets.grad = pos - (long)strlen(line);
            else if (strncmp(p, "[TRAP]", 6) == 0)
                seq->offsets.trap = pos - (long)strlen(line);
            else if (strncmp(p, "[ADC]", 5) == 0)
                seq->offsets.adc = pos - (long)strlen(line);
            else if (strncmp(p, "[EXTENSIONS]", 12) == 0)
                seq->offsets.extensions = pos - (long)strlen(line);
            else if (strncmp(p, "[SHAPES]", 8) == 0)
                seq->offsets.shapes = pos - (long)strlen(line);
            else if (strncmp(p, "[SIGNATURE]", 11) == 0)
                seq->offsets.signature = pos - (long)strlen(line);
        }
        else if (strncmp(p, "extension", 9) == 0 && (*(p + 9) == ' ' || *(p + 9) == '\t'))
        {
            ext_id = -1;
            ext_enum = PULSEQ_EXT_UNKNOWN;
            if (sscanf(p, "extension %31s %d", ext_name, &ext_id) == 2)
            {
                if (strcmp(ext_name, "TRIGGERS") == 0)
                    ext_enum = PULSEQ_EXT_TRIGGER;
                else if (strcmp(ext_name, "ROTATIONS") == 0)
                    ext_enum = PULSEQ_EXT_ROTATION;
                else if (strcmp(ext_name, "LABELSET") == 0)
                    ext_enum = PULSEQ_EXT_LABELSET;
                else if (strcmp(ext_name, "LABELINC") == 0)
                    ext_enum = PULSEQ_EXT_LABELINC;
                else if (strcmp(ext_name, "RF_SHIMS") == 0)
                    ext_enum = PULSEQ_EXT_RF_SHIM;
                else if (strcmp(ext_name, "DELAYS") == 0)
                    ext_enum = PULSEQ_EXT_DELAY;
                switch (ext_enum)
                {
                case PULSEQ_EXT_TRIGGER:
                    seq->offsets.triggers = pos - (long)strlen(line);
                    break;
                case PULSEQ_EXT_ROTATION:
                    seq->offsets.rotations = pos - (long)strlen(line);
                    break;
                case PULSEQ_EXT_LABELSET:
                    seq->offsets.labelset = pos - (long)strlen(line);
                    break;
                case PULSEQ_EXT_LABELINC:
                    seq->offsets.labelinc = pos - (long)strlen(line);
                    break;
                case PULSEQ_EXT_RF_SHIM:
                    seq->offsets.rfshim = pos - (long)strlen(line);
                    break;
                case PULSEQ_EXT_DELAY:
                    seq->offsets.delays = pos - (long)strlen(line);
                    break;
                default:
                    break;
                }
                if (ext_enum >= 0 && ext_enum < PULSEQ_EXT_UNKNOWN)
                    seq->extension_map[ext_enum] = ext_id;
            }
        }
    }
    seq->offsets.scan_cursor = ftell(f);
}

static void read_version(pulseq_file *seq, FILE *f)
{
    char line[PULSEQ_MAX_LINE_LENGTH];
    int major = 0, minor = 0, revision = 0;
    char key[32];
    int value;
    char *p;

    if (seq->is_version_parsed)
        return;
    if (seq->offsets.version < 0)
    {
        seq->is_version_parsed = 1;
        return;
    }
    if (fseek(f, seq->offsets.version, SEEK_SET) != 0)
        return;
    if (!fgets(line, sizeof(line), f))
        return;

    while (fgets(line, sizeof(line), f))
    {
        p = line;
        while (*p == ' ' || *p == '\t')
            p++;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#')
            continue;
        if (*p == '[')
            break;
        if (sscanf(p, "%31s %d", key, &value) == 2)
        {
            if (strcmp(key, "major") == 0)
                major = value;
            else if (strcmp(key, "minor") == 0)
                minor = value;
            else if (strcmp(key, "revision") == 0)
                revision = value;
        }
    }
    seq->version_major = major;
    seq->version_minor = minor;
    seq->version_revision = revision;
    seq->version_combined = major * 1000000 + minor * 1000 + revision;
    seq->is_version_parsed = 1;
}

static void read_definitions_library(pulseq_file *seq, FILE *f)
{
    int ret;
    char line[PULSEQ_MAX_LINE_LENGTH];
    int def_index = 0;
    char *p;
    char *name_tok;
    char *token;
    char **new_array;
    int i;
    pulseq_definition def;

    if (seq->is_definitions_library_parsed)
        return;
    if (seq->offsets.definitions < 0)
    {
        seq->is_definitions_library_parsed = 1;
        return;
    }

    ret = init_definitions_library(
        &seq->definitions_library,
        &seq->num_definitions,
        f,
        seq->offsets.definitions);
    if (ret != 0)
        return;

    if (fseek(f, seq->offsets.definitions, SEEK_SET) != 0)
        return;
    if (!fgets(line, sizeof(line), f))
        return;

    while (fgets(line, sizeof(line), f))
    {
        p = line;
        while (isspace((unsigned char)*p))
            p++;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#')
            continue;
        if (*p == '[')
            break;

        def.value_size = 0;
        def.value = NULL;

        name_tok = strtok(p, " \t\r\n");
        if (!name_tok)
            continue;
        strncpy(def.name, name_tok, PULSEQ_DEFINITION_NAME_LENGTH - 1);
        def.name[PULSEQ_DEFINITION_NAME_LENGTH - 1] = '\0';

        while ((token = strtok(NULL, " \t\r\n")) != NULL)
        {
            new_array = (char **)PULSEQ_ALLOC(sizeof(char *) * (def.value_size + 1));
            for (i = 0; i < def.value_size; i++)
                new_array[i] = def.value[i];
            new_array[def.value_size] = (char *)PULSEQ_ALLOC(strlen(token) + 1);
            strcpy(new_array[def.value_size], token);
            if (def.value)
                PULSEQ_FREE(def.value);
            def.value = new_array;
            def.value_size++;
        }
        seq->definitions_library[def_index++] = def;
    }
    seq->is_definitions_library_parsed = 1;
}

static void read_definitions(pulseq_file *seq)
{
    int i;
    int nvals;
    char *key;
    char *value;
    float temp[3];

    for (i = 0; i < seq->num_definitions; i++)
    {
        key = seq->definitions_library[i].name;
        nvals = seq->definitions_library[i].value_size;
        value = (nvals > 0 && seq->definitions_library[i].value != NULL)
            ? seq->definitions_library[i].value[0]
            : NULL;

        /* Some definition lines may intentionally carry no value
         * (for example optional extensions fields). */
        if (value == NULL)
        {
            continue;
        }

        if (strcmp(key, "GradientRasterTime") == 0)
        {
            seq->reserved_definitions_library.gradient_raster_time = (float)(atof(value) * 1e6);
        }
        else if (strcmp(key, "RadiofrequencyRasterTime") == 0)
        {
            seq->reserved_definitions_library.radiofrequency_raster_time =
                (float)(atof(value) * 1e6);
        }
        else if (strcmp(key, "AdcRasterTime") == 0)
        {
            seq->reserved_definitions_library.adc_raster_time = (float)(atof(value) * 1e6);
        }
        else if (strcmp(key, "BlockDurationRaster") == 0)
        {
            seq->reserved_definitions_library.block_duration_raster = (float)(atof(value) * 1e6);
        }
        else if (strcmp(key, "Name") == 0)
        {
            strncpy(
                seq->reserved_definitions_library.name,
                value,
                sizeof(seq->reserved_definitions_library.name) - 1);
            seq->reserved_definitions_library
                .name[sizeof(seq->reserved_definitions_library.name) - 1] = '\0';
        }
        else if (strcmp(key, "FOV") == 0)
        {
            if (nvals >= 3)
            {
                temp[0] = (float)atof(seq->definitions_library[i].value[0]);
                temp[1] = (float)atof(seq->definitions_library[i].value[1]);
                temp[2] = (float)atof(seq->definitions_library[i].value[2]);
                seq->reserved_definitions_library.fov[0] = temp[0] * 100.0f;
                seq->reserved_definitions_library.fov[1] = temp[1] * 100.0f;
                seq->reserved_definitions_library.fov[2] = temp[2] * 100.0f;
            }
        }
        else if (strcmp(key, "Matrix") == 0)
        {
            if (nvals >= 3)
            {
                seq->reserved_definitions_library.matrix[0] =
                    (float)atof(seq->definitions_library[i].value[0]);
                seq->reserved_definitions_library.matrix[1] =
                    (float)atof(seq->definitions_library[i].value[1]);
                seq->reserved_definitions_library.matrix[2] =
                    (float)atof(seq->definitions_library[i].value[2]);
            }
        }
        else if (strcmp(key, "NavFOV") == 0)
        {
            if (nvals >= 3)
            {
                seq->reserved_definitions_library.nav_fov[0] =
                    (float)atof(seq->definitions_library[i].value[0]) * 100.0f;
                seq->reserved_definitions_library.nav_fov[1] =
                    (float)atof(seq->definitions_library[i].value[1]) * 100.0f;
                seq->reserved_definitions_library.nav_fov[2] =
                    (float)atof(seq->definitions_library[i].value[2]) * 100.0f;
            }
        }
        else if (strcmp(key, "NavMatrix") == 0)
        {
            if (nvals >= 3)
            {
                seq->reserved_definitions_library.nav_matrix[0] =
                    (float)atof(seq->definitions_library[i].value[0]);
                seq->reserved_definitions_library.nav_matrix[1] =
                    (float)atof(seq->definitions_library[i].value[1]);
                seq->reserved_definitions_library.nav_matrix[2] =
                    (float)atof(seq->definitions_library[i].value[2]);
            }
        }
        else if (strcmp(key, "TotalDuration") == 0)
        {
            seq->reserved_definitions_library.total_duration = (float)atof(value);
        }
        else if (strcmp(key, "NextSequence") == 0)
        {
            strncpy(
                seq->reserved_definitions_library.next_sequence,
                value,
                sizeof(seq->reserved_definitions_library.next_sequence) - 1);
            seq->reserved_definitions_library
                .next_sequence[sizeof(seq->reserved_definitions_library.next_sequence) - 1] = '\0';
        }
        else if (strcmp(key, "IgnoreFovShift") == 0)
        {
            seq->reserved_definitions_library.ignore_fov_shift = atoi(value);
        }
        else if (strcmp(key, "EnablePmc") == 0)
        {
            seq->reserved_definitions_library.enable_pmc = atoi(value);
        }
        else if (strcmp(key, "IgnoreAverages") == 0)
        {
            seq->reserved_definitions_library.ignore_averages = atoi(value);
        }
        else if (strcmp(key, "NumGainCalibrationReadouts") == 0)
        {
            seq->reserved_definitions_library.num_gain_cal_readouts = atoi(value);
        }
    }
}

/* ================================================================== */
/*  Section readers                                                   */
/* ================================================================== */

static void read_block_library(pulseq_file *seq, FILE *f)
{
    int ret;
    float block_vals[7] = {1, 1, 1, 1, 1, 1, 1};
    pulseq_scale s;
    s.size = 7;
    s.values = block_vals;

    if (seq->is_block_library_parsed)
        return;
    if (seq->offsets.blocks < 0)
    {
        seq->is_block_library_parsed = 1;
        return;
    }
    ret = init_standard_library(
        (void **)&seq->block_library,
        &seq->num_blocks,
        f,
        &seq->offsets.blocks,
        1,
        s.size);
    if (ret != 0)
        return;
    ret = read_standard_library(
        seq->block_library,
        seq->num_blocks,
        f,
        seq->offsets.blocks,
        s.size,
        s,
        -1);
    if (ret != 0)
        return;
    seq->is_block_library_parsed = 1;
}

static void read_rf_library(pulseq_file *seq, FILE *f)
{
    int ret;
    float rf_vals[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    pulseq_scale s;
    s.size = 10;
    s.values = rf_vals;

    if (seq->is_rf_library_parsed)
        return;
    if (seq->offsets.rf < 0)
    {
        seq->is_rf_library_parsed = 1;
        return;
    }
    ret = init_standard_library(
        (void **)&seq->rf_library,
        &seq->rf_library_size,
        f,
        &seq->offsets.rf,
        1,
        s.size);
    if (ret != 0)
        return;
    ret = read_standard_library(
        seq->rf_library,
        seq->rf_library_size,
        f,
        seq->offsets.rf,
        s.size,
        s,
        -1);
    if (ret != 0)
        return;
    seq->is_rf_library_parsed = 1;

    /* parse trailing RF use tags (e/r/s/i) from the [RF] section */
    if (seq->rf_library_size > 0 && seq->offsets.rf >= 0)
    {
        char use_line[PULSEQ_MAX_LINE_LENGTH];
        seq->rf_use_tags = (int *)PULSEQ_ALLOC((size_t)seq->rf_library_size * sizeof(int));
        if (seq->rf_use_tags)
        {
            int ui;
            for (ui = 0; ui < seq->rf_library_size; ++ui)
                seq->rf_use_tags[ui] = PULSEQ_RF_USE_UNKNOWN;

            if (fseek(f, seq->offsets.rf, SEEK_SET) == 0 && fgets(use_line, sizeof(use_line), f))
            {
                while (fgets(use_line, sizeof(use_line), f))
                {
                    char *up = use_line;
                    int use_idx;
                    float dummy;
                    int col, consumed;
                    char *scan_p;
                    while (*up == ' ' || *up == '\t')
                        up++;
                    if (*up == '[' || *up == '\0' || *up == '\n' || *up == '\r' || *up == '#')
                        break;
                    if (sscanf(up, "%d", &use_idx) != 1)
                        continue;
                    if (use_idx < 1 || use_idx > seq->rf_library_size)
                        continue;
                    /* skip index */
                    while (*up && *up != ' ' && *up != '\t')
                        up++;
                    while (*up == ' ' || *up == '\t')
                        up++;
                    /* skip 10 float columns */
                    scan_p = up;
                    for (col = 0; col < 10; ++col)
                    {
                        consumed = 0;
                        if (sscanf(scan_p, "%f%n", &dummy, &consumed) != 1)
                            break;
                        scan_p += consumed;
                        while (*scan_p == ' ' || *scan_p == '\t')
                            scan_p++;
                    }
                    /* trailing character is the rf_use tag */
                    if (col == 10)
                    {
                        char tag = *scan_p;
                        int use_val = PULSEQ_RF_USE_UNKNOWN;
                        if (tag == 'e')
                            use_val = PULSEQ_RF_USE_EXCITATION;
                        else if (tag == 'r')
                            use_val = PULSEQ_RF_USE_REFOCUSING;
                        else if (tag == 'i')
                            use_val = PULSEQ_RF_USE_INVERSION;
                        else if (tag == 's')
                            use_val = PULSEQ_RF_USE_SATURATION;
                        seq->rf_use_tags[use_idx - 1] = use_val;
                    }
                }
            }
        }
    }
}

static void read_grad_library(pulseq_file *seq, FILE *f)
{
    int ret;
    long offsets[2];
    int num_sections = 0;
    static const float grad_vals[6] = {1, 1, 1, 1, 1, 1};
    static const float trap_vals[5] = {1, 1, 1, 1, 1};
    pulseq_scale grad_s, trap_s;

    grad_s.size = 6;
    grad_s.values = (float *)grad_vals;
    trap_s.size = 5;
    trap_s.values = (float *)trap_vals;

    if (seq->is_grad_library_parsed)
        return;
    offsets[0] = seq->offsets.grad;
    offsets[1] = seq->offsets.trap;
    if (offsets[0] >= 0)
        num_sections++;
    if (offsets[1] >= 0)
        num_sections++;
    if (num_sections == 0)
    {
        seq->is_grad_library_parsed = 1;
        return;
    }

    ret = init_standard_library(
        (void **)&seq->grad_library,
        &seq->grad_library_size,
        f,
        offsets,
        2,
        grad_s.size + 1);
    if (ret != 0)
        return;
    if (offsets[0] >= 0)
    {
        ret = read_standard_library(
            seq->grad_library,
            seq->grad_library_size,
            f,
            offsets[0],
            grad_s.size + 1,
            grad_s,
            1);
        if (ret != 0)
            return;
    }
    if (offsets[1] >= 0)
    {
        ret = read_standard_library(
            seq->grad_library,
            seq->grad_library_size,
            f,
            offsets[1],
            grad_s.size + 1,
            trap_s,
            0);
        if (ret != 0)
            return;
    }
    seq->is_grad_library_parsed = 1;
}

static void read_adc_library(pulseq_file *seq, FILE *f)
{
    int ret;
    static const float adc_vals[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    pulseq_scale s;
    s.size = 8;
    s.values = (float *)adc_vals;

    if (seq->is_adc_library_parsed)
        return;
    if (seq->offsets.adc < 0)
    {
        seq->is_adc_library_parsed = 1;
        return;
    }
    ret = init_standard_library(
        (void **)&seq->adc_library,
        &seq->adc_library_size,
        f,
        &seq->offsets.adc,
        1,
        s.size);
    if (ret != 0)
        return;
    ret = read_standard_library(
        seq->adc_library,
        seq->adc_library_size,
        f,
        seq->offsets.adc,
        s.size,
        s,
        -1);
    if (ret != 0)
        return;
    seq->is_adc_library_parsed = 1;
}

static void read_shapes_library(pulseq_file *seq, FILE *f)
{
    int ret;
    char line[PULSEQ_MAX_LINE_LENGTH];
    int shape_index = 0, sample_index = 0;
    char *p;
    float val;

    if (seq->is_shapes_library_parsed)
        return;
    if (seq->offsets.shapes < 0)
    {
        seq->is_shapes_library_parsed = 1;
        return;
    }

    ret = init_shapes_library(
        &seq->shapes_library,
        &seq->shapes_library_size,
        f,
        seq->offsets.shapes);
    if (ret != 0)
        return;

    if (fseek(f, seq->offsets.shapes, SEEK_SET) != 0)
        return;
    if (!fgets(line, sizeof(line), f))
        return;

    while (fgets(line, sizeof(line), f))
    {
        p = line;
        while (*p == ' ' || *p == '\t')
            p++;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#')
            continue;
        if (*p == '[')
            break;
        if (strncmp(p, "shape_id", 8) == 0)
        {
            if (sscanf(p + 8, "%d", &shape_index) == 1)
                sample_index = 0;
            continue;
        }
        if (strncmp(p, "num_samples", 11) == 0)
            continue;
        if (shape_index > 0 && shape_index <= seq->shapes_library_size)
        {
            if (sscanf(p, "%f", &val) == 1)
            {
                if (sample_index < seq->shapes_library[shape_index - 1].num_samples)
                    seq->shapes_library[shape_index - 1].samples[sample_index++] = val;
            }
        }
    }
    seq->is_shapes_library_parsed = 1;
}

static int read_label_library(
    void *target,
    int target_count,
    int *is_label_defined,
    FILE *f,
    long offset,
    int n)
{
    char line[PULSEQ_MAX_LINE_LENGTH];
    char *p;
    int idx, label_code;
    float val;
    char label[PULSEQ_LABEL_NAME_LENGTH];
    float *arr = (float *)target;

    if (!f || offset < 0)
        return 1;
    if (fseek(f, offset, SEEK_SET) != 0)
        return 1;
    if (!fgets(line, sizeof(line), f))
        return 1;

    while (fgets(line, sizeof(line), f))
    {
        p = line;
        while (*p == ' ' || *p == '\t')
            p++;
        if (*p == '[' || *p == 'e')
            break;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#')
            continue;
        if (sscanf(p, "%d %f %31s", &idx, &val, label) != 3)
            continue;
        if (idx <= 0 || idx > target_count)
            continue;
        label_code = pulseq_label_id_for_name(label);
        if (label_code > 0)
            is_label_defined[label_code - 1] = 1;
        arr[(idx - 1) * n + 0] = val;
        arr[(idx - 1) * n + 1] = (float)label_code;
    }
    return 0;
}

static int read_delay_library(
    void *target,
    int target_count,
    int *is_delay_defined,
    FILE *f,
    long offset,
    int n)
{
    char line[PULSEQ_MAX_LINE_LENGTH];
    char *p;
    int idx, hint_code;
    float num_id, offset_val, scale_val;
    char hint[PULSEQ_SOFT_DELAY_HINT_LENGTH];
    float *arr = (float *)target;

    if (!f || offset < 0)
        return 1;
    if (fseek(f, offset, SEEK_SET) != 0)
        return 1;
    if (!fgets(line, sizeof(line), f))
        return 1;

    while (fgets(line, sizeof(line), f))
    {
        p = line;
        while (*p == ' ' || *p == '\t')
            p++;
        if (*p == '[' || *p == 'e')
            break;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#')
            continue;
        if (sscanf(p, "%d %f %f %f %31s", &idx, &num_id, &offset_val, &scale_val, hint) != 5)
            continue;
        if (idx <= 0 || idx > target_count)
            continue;
        hint_code = pulseq_hint_id_for_name(hint);
        if (hint_code > 0)
            is_delay_defined[hint_code - 1] = 1;
        arr[(idx - 1) * n + 0] = num_id;
        arr[(idx - 1) * n + 1] = offset_val;
        arr[(idx - 1) * n + 2] = scale_val;
        arr[(idx - 1) * n + 3] = (float)hint_code;
    }
    return 0;
}

static int read_rf_shim_library(
    pulseq_rf_shim_entry *target,
    int target_count,
    FILE *f,
    long offset)
{
    char line[PULSEQ_MAX_LINE_LENGTH];
    char *p;
    int idx, n_ch, i, consumed;
    float val;

    if (!f || !target)
        return 1;
    if (fseek(f, offset, SEEK_SET) != 0)
        return 1;
    if (!fgets(line, sizeof(line), f))
        return 1;

    while (fgets(line, sizeof(line), f))
    {
        p = line;
        while (*p == ' ' || *p == '\t')
            p++;
        if (*p == '[' || *p == 'e')
            break;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#')
            continue;
        if (sscanf(p, "%d %d", &idx, &n_ch) != 2)
            continue;
        if (idx <= 0 || idx > target_count || n_ch <= 0)
            continue;

        while (*p && *p != ' ')
            p++;
        while (*p == ' ')
            p++;
        while (*p && *p != ' ')
            p++;
        while (*p == ' ')
            p++;

        if (n_ch > PULSEQ_MAX_RF_SHIM_CHANNELS)
            return 1;
        target[idx - 1].num_channels = n_ch;
        for (i = 0; i < 2 * n_ch; i++)
        {
            consumed = 0;
            if (sscanf(p, "%f%n", &val, &consumed) != 1)
                break;
            target[idx - 1].values[i] = val;
            p += consumed;
            while (*p == ' ' || *p == '\t')
                p++;
        }
    }
    return 0;
}

static void read_extensions_library(pulseq_file *seq, FILE *f)
{
    int ret, n, k;
    float qn;
    static const float ext_vals[3] = {1, 1, 1};
    static const float trig_vals[4] = {1, 1, 1, 1};
    static const float rot_vals[4] = {1, 1, 1, 1};
    pulseq_scale ext_s, trig_s, rot_s;

    ext_s.size = 3;
    ext_s.values = (float *)ext_vals;
    trig_s.size = 4;
    trig_s.values = (float *)trig_vals;
    rot_s.size = 4;
    rot_s.values = (float *)rot_vals;

    if (seq->is_extensions_library_parsed)
        return;
    if (seq->offsets.extensions < 0)
    {
        seq->is_extensions_library_parsed = 1;
        return;
    }

    /* Init & read extensions list */
    ret = init_standard_library(
        (void **)&seq->extensions_library,
        &seq->extensions_library_size,
        f,
        &seq->offsets.extensions,
        1,
        3);
    if (ret != 0)
        return;

    if (seq->offsets.triggers >= 0)
    {
        ret = init_standard_library(
            (void **)&seq->trigger_library,
            &seq->trigger_library_size,
            f,
            &seq->offsets.triggers,
            1,
            4);
        if (ret != 0)
            return;
    }
    if (seq->offsets.rotations >= 0)
    {
        ret = init_standard_library(
            (void **)&seq->rotation_quaternion_library,
            &seq->rotation_library_size,
            f,
            &seq->offsets.rotations,
            1,
            4);
        if (ret != 0)
            return;
    }
    if (seq->offsets.labelset >= 0)
    {
        ret = init_standard_library(
            (void **)&seq->labelset_library,
            &seq->labelset_library_size,
            f,
            &seq->offsets.labelset,
            1,
            2);
        if (ret != 0)
            return;
    }
    if (seq->offsets.labelinc >= 0)
    {
        ret = init_standard_library(
            (void **)&seq->labelinc_library,
            &seq->labelinc_library_size,
            f,
            &seq->offsets.labelinc,
            1,
            2);
        if (ret != 0)
            return;
    }
    if (seq->offsets.delays >= 0)
    {
        ret = init_standard_library(
            (void **)&seq->soft_delay_library,
            &seq->soft_delay_library_size,
            f,
            &seq->offsets.delays,
            1,
            4);
        if (ret != 0)
            return;
    }
    if (seq->offsets.rfshim >= 0)
    {
        ret = init_rf_shim_library(
            &seq->rf_shim_library,
            &seq->rf_shim_library_size,
            f,
            seq->offsets.rfshim);
        if (ret != 0)
            return;
    }

    /* Read data */
    ret = read_standard_library(
        seq->extensions_library,
        seq->extensions_library_size,
        f,
        seq->offsets.extensions,
        3,
        ext_s,
        -1);
    if (ret != 0)
        return;

    if (seq->offsets.triggers >= 0)
    {
        ret = read_standard_library(
            seq->trigger_library,
            seq->trigger_library_size,
            f,
            seq->offsets.triggers,
            4,
            trig_s,
            -1);
        if (ret != 0)
            return;
    }

    if (seq->offsets.rotations >= 0)
    {
        ret = read_standard_library(
            seq->rotation_quaternion_library,
            seq->rotation_library_size,
            f,
            seq->offsets.rotations,
            4,
            rot_s,
            -1);
        if (ret != 0)
            return;
        /* Normalize quaternions */
        for (k = 1; k < seq->rotation_library_size; k++)
        {
            qn = (float)sqrt(
                (double)seq->rotation_quaternion_library[k][0] *
                    seq->rotation_quaternion_library[k][0] +
                (double)seq->rotation_quaternion_library[k][1] *
                    seq->rotation_quaternion_library[k][1] +
                (double)seq->rotation_quaternion_library[k][2] *
                    seq->rotation_quaternion_library[k][2] +
                (double)seq->rotation_quaternion_library[k][3] *
                    seq->rotation_quaternion_library[k][3]);
            seq->rotation_quaternion_library[k][0] /= qn;
            seq->rotation_quaternion_library[k][1] /= qn;
            seq->rotation_quaternion_library[k][2] /= qn;
            seq->rotation_quaternion_library[k][3] /= qn;
        }
    }

    if (seq->offsets.labelset >= 0)
    {
        ret = read_label_library(
            seq->labelset_library,
            seq->labelset_library_size,
            seq->is_label_defined,
            f,
            seq->offsets.labelset,
            2);
        if (ret != 0)
            return;
    }
    if (seq->offsets.labelinc >= 0)
    {
        ret = read_label_library(
            seq->labelinc_library,
            seq->labelinc_library_size,
            seq->is_label_defined,
            f,
            seq->offsets.labelinc,
            2);
        if (ret != 0)
            return;
    }
    if (seq->offsets.delays >= 0)
    {
        ret = read_delay_library(
            seq->soft_delay_library,
            seq->soft_delay_library_size,
            seq->is_delay_defined,
            f,
            seq->offsets.delays,
            4);
        if (ret != 0)
            return;
    }
    if (seq->offsets.rfshim >= 0)
    {
        ret = read_rf_shim_library(
            seq->rf_shim_library,
            seq->rf_shim_library_size,
            f,
            seq->offsets.rfshim);
        if (ret != 0)
            return;
    }

    /* Build extension LUT */
    for (n = 0; n < 8; n++)
    {
        if (seq->extension_lut_size < seq->extension_map[n])
            seq->extension_lut_size = seq->extension_map[n];
    }
    if (seq->extension_lut_size > 0)
    {
        seq->extension_lut = (int *)PULSEQ_ALLOC(sizeof(int) * (seq->extension_lut_size + 1));
        for (n = 0; n < 8; n++)
        {
            if (seq->extension_map[n] > 0)
                seq->extension_lut[seq->extension_map[n]] = n;
        }
    }

    seq->is_extensions_library_parsed = 1;
}

/* ================================================================== */
/*  Shape decompression (cross-file)                                  */
/* ================================================================== */

int pulseq_decompress_shape(pulseq_shape *result, const pulseq_shape *encoded, float scale)
{
    int i, rep;
    const float *packed;
    int num_packed, num_samples;
    int count_pack = 1;
    int count_unpack = 1;
    float *unpacked;

    if (!encoded || !result)
        return 0;

    packed = encoded->samples;
    num_packed = encoded->num_samples;
    num_samples = encoded->num_uncompressed_samples;

    /* Uncompressed — copy */
    if (encoded->num_samples == encoded->num_uncompressed_samples)
    {
        result->num_samples = encoded->num_samples;
        result->num_uncompressed_samples = encoded->num_uncompressed_samples;
        result->samples = (float *)PULSEQ_ALLOC(sizeof(float) * encoded->num_samples);
        if (!result->samples)
            return 0;
        for (i = 0; i < encoded->num_samples; ++i)
            result->samples[i] = encoded->samples[i] * scale;
        return 1;
    }

    unpacked = (float *)PULSEQ_ALLOC(sizeof(float) * num_samples);
    if (!unpacked)
        return 0;

    while (count_pack < num_packed)
    {
        if (packed[count_pack - 1] != packed[count_pack])
        {
            unpacked[count_unpack - 1] = packed[count_pack - 1];
            count_pack++;
            count_unpack++;
        }
        else
        {
            rep = (int)(packed[count_pack + 1]) + 2;
            if (fabs(packed[count_pack + 1] + 2 - (float)rep) > 1e-6f)
            {
                PULSEQ_FREE(unpacked);
                return 0;
            }
            for (i = count_unpack - 1; i <= count_unpack + rep - 2; i++)
                unpacked[i] = packed[count_pack - 1];
            count_pack += 3;
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
/*  Read seq from buffer / file (cross-file)                          */
/* ================================================================== */

int pulseq_read_from_buffer(pulseq_file *seq, FILE *f)
{
    if (!seq || !f)
        return PULSEQ_ERR_NULL_POINTER;
    seq_file_reset(seq);
    if (seq->file_path)
    {
        PULSEQ_FREE(seq->file_path);
        seq->file_path = NULL;
    }

    get_section_offsets(seq, f);
    read_version(seq, f);
    if (seq->version_combined < 1005000)
        return PULSEQ_ERR_UNSUPPORTED_VERSION;
    read_definitions_library(seq, f);
    read_definitions(seq);
    read_block_library(seq, f);
    read_rf_library(seq, f);
    read_grad_library(seq, f);
    read_adc_library(seq, f);
    read_shapes_library(seq, f);
    read_extensions_library(seq, f);
    return PULSEQ_SUCCESS;
}

int pulseq_read(pulseq_file *seq, const char *file_path)
{
    FILE *f;
    int code;
    if (!seq || !file_path)
        return PULSEQ_ERR_INVALID_ARGUMENT;
    f = fopen(file_path, "r");
    if (!f)
        return PULSEQ_ERR_FILE_NOT_FOUND;
    code = pulseq_read_from_buffer(seq, f);
    fclose(f);
    return code;
}

/* ================================================================== */
/*  Sequence collection (cross-file)                                  */
/* ================================================================== */

static int count_sequences_in_chain(const char *first_path, const pulseq_raster *raster)
{
    int count = 0;
    int max_count = 1000;
    char *current_path;
    char *base_path;
    pulseq_file temp;
    int result;

    base_path = pulseq_path_dirname(first_path);
    if (!base_path)
        return -1;
    current_path = (char *)PULSEQ_ALLOC(strlen(first_path) + 1);
    if (!current_path)
    {
        PULSEQ_FREE(base_path);
        return -1;
    }
    strcpy(current_path, first_path);

    while (current_path && current_path[0] != '\0' && count < max_count)
    {
        pulseq_file_init(&temp, raster);
        result = pulseq_read(&temp, current_path);
        if (PULSEQ_FAILED(result))
        {
            seq_file_reset(&temp);
            PULSEQ_FREE(current_path);
            PULSEQ_FREE(base_path);
            return -1;
        }
        count++;
        PULSEQ_FREE(current_path);
        current_path = NULL;
        if (temp.reserved_definitions_library.next_sequence[0] != '\0')
        {
            current_path =
                pulseq_path_join(base_path, temp.reserved_definitions_library.next_sequence);
            if (!current_path)
            {
                seq_file_reset(&temp);
                PULSEQ_FREE(base_path);
                return -1;
            }
        }
        seq_file_reset(&temp);
    }
    PULSEQ_FREE(base_path);
    if (count >= max_count)
        return -1;
    return count;
}

int pulseq_file_set_read(
    pulseq_file_set *set,
    const char *first_file_path,
    const pulseq_raster *raster)
{
    int num_seq, i, j, result;
    char *current_path;

    if (!set || !first_file_path || !raster)
        return PULSEQ_ERR_NULL_POINTER;

    num_seq = count_sequences_in_chain(first_file_path, raster);
    if (num_seq <= 0)
    {
        set->num_sequences = 0;
        set->sequences = NULL;
        set->base_path = NULL;
        return (num_seq == 0) ? PULSEQ_ERR_EMPTY : PULSEQ_ERR_CHAIN_BROKEN;
    }

    set->base_path = pulseq_path_dirname(first_file_path);
    if (!set->base_path)
        return PULSEQ_ERR_ALLOC_FAILED;

    set->sequences = (pulseq_file *)PULSEQ_ALLOC(num_seq * sizeof(pulseq_file));
    if (!set->sequences)
    {
        PULSEQ_FREE(set->base_path);
        set->base_path = NULL;
        return PULSEQ_ERR_ALLOC_FAILED;
    }

    current_path = (char *)PULSEQ_ALLOC(strlen(first_file_path) + 1);
    if (!current_path)
    {
        PULSEQ_FREE(set->sequences);
        PULSEQ_FREE(set->base_path);
        set->sequences = NULL;
        set->base_path = NULL;
        return PULSEQ_ERR_ALLOC_FAILED;
    }
    strcpy(current_path, first_file_path);

    for (i = 0; i < num_seq; ++i)
    {
        pulseq_file_init(&set->sequences[i], raster);
        result = pulseq_read(&set->sequences[i], current_path);
        if (PULSEQ_FAILED(result))
        {
            for (j = 0; j < i; ++j)
                seq_file_reset(&set->sequences[j]);
            PULSEQ_FREE(set->sequences);
            PULSEQ_FREE(current_path);
            PULSEQ_FREE(set->base_path);
            set->sequences = NULL;
            set->base_path = NULL;
            return result;
        }
        /* store file path for later use (e.g. signature verification) */
        set->sequences[i].file_path = (char *)PULSEQ_ALLOC(strlen(current_path) + 1);
        if (set->sequences[i].file_path)
            strcpy(set->sequences[i].file_path, current_path);
        if (i < num_seq - 1)
        {
            PULSEQ_FREE(current_path);
            current_path = pulseq_path_join(
                set->base_path,
                set->sequences[i].reserved_definitions_library.next_sequence);
            if (!current_path)
            {
                for (j = 0; j <= i; ++j)
                    seq_file_reset(&set->sequences[j]);
                PULSEQ_FREE(set->sequences);
                PULSEQ_FREE(set->base_path);
                set->sequences = NULL;
                set->base_path = NULL;
                return PULSEQ_ERR_ALLOC_FAILED;
            }
        }
    }
    PULSEQ_FREE(current_path);
    set->num_sequences = num_seq;
    return PULSEQ_SUCCESS;
}

/* ================================================================== */
/*  Raw block / extension parsing (public)                            */
/* ================================================================== */

int pulseq_get_raw_block_content_ids(
    const pulseq_file *seq,
    pulseq_raw_block *block,
    int block_index,
    int parse_extensions)
{
    int next_ext_id, ext_count;
    float *ev;
    float *ext_data;

    if (!seq || !block || block_index < 0 || block_index >= seq->num_blocks)
        return 0;

    block->block_duration = 0;
    block->rf = -1;
    block->gx = -1;
    block->gy = -1;
    block->gz = -1;
    block->adc = -1;
    block->ext_count = 0;

    if (!seq->block_library)
        return 0;

    ev = seq->block_library[block_index];
    block->block_duration = (int)ev[0];
    block->rf = (int)ev[1] - 1;
    block->gx = (int)ev[2] - 1;
    block->gy = (int)ev[3] - 1;
    block->gz = (int)ev[4] - 1;
    block->adc = (int)ev[5] - 1;

    if (!parse_extensions)
    {
        block->ext_count = 0;
        return 1;
    }
    if (!seq->is_extensions_library_parsed || !seq->extensions_library ||
        seq->extensions_library_size <= 0)
        return 1;

    next_ext_id = (int)ev[6];
    ext_count = 0;
    while (next_ext_id > 0 && next_ext_id <= seq->extensions_library_size &&
           ext_count < PULSEQ_MAX_EXTENSIONS_PER_BLOCK)
    {
        ext_data = seq->extensions_library[next_ext_id - 1];
        block->ext[ext_count][0] = (int)ext_data[0];
        block->ext[ext_count][1] = (int)ext_data[1] - 1;
        next_ext_id = (int)ext_data[2];
        ext_count++;
    }
    block->ext_count = ext_count;
    return 1;
}

static void raw_extension_init(pulseq_raw_extension *re)
{
    if (!re)
        return;
    memset(&re->labelset, 0, sizeof(re->labelset));
    memset(&re->labelinc, 0, sizeof(re->labelinc));
    re->flag.trid = -1;
    re->flag.module_id = -1;
    re->flag.nav = -1;
    re->flag.rev = -1;
    re->flag.sms = -1;
    re->flag.ref = -1;
    re->flag.ima = -1;
    re->flag.noise = -1;
    re->flag.pmc = -1;
    re->flag.norot = -1;
    re->flag.nopos = -1;
    re->flag.noscl = -1;
    re->flag.once = -1;
    re->rotation_index = -1;
    re->rf_shim_index = -1;
    re->trigger_index = -1;
    re->soft_delay_index = -1;
}

void pulseq_get_raw_extension(
    const pulseq_file *seq,
    pulseq_raw_extension *re,
    const pulseq_raw_block *raw)
{
    int i, type_idx, ref_idx, ext_type, label_value, label_id;

    raw_extension_init(re);
    if (!seq || !re || !raw)
        return;
    if (!seq->is_extensions_library_parsed || !seq->extension_lut)
        return;

    for (i = 0; i < raw->ext_count; ++i)
    {
        type_idx = raw->ext[i][0];
        ref_idx = raw->ext[i][1];
        if (type_idx < 0 || type_idx > seq->extension_lut_size)
            continue;
        ext_type = seq->extension_lut[type_idx];
        if (ref_idx < 0)
            continue;

        switch (ext_type)
        {
        case PULSEQ_EXT_LABELSET:
            if (seq->labelset_library && ref_idx < seq->labelset_library_size)
            {
                label_value = (int)seq->labelset_library[ref_idx][0];
                label_id = (int)seq->labelset_library[ref_idx][1];
                switch (label_id)
                {
                case PULSEQ_LABEL_SLC:
                    re->labelset.slc = label_value;
                    break;
                case PULSEQ_LABEL_SEG:
                    re->labelset.seg = label_value;
                    break;
                case PULSEQ_LABEL_REP:
                    re->labelset.rep = label_value;
                    break;
                case PULSEQ_LABEL_AVG:
                    re->labelset.avg = label_value;
                    break;
                case PULSEQ_LABEL_SET:
                    re->labelset.set = label_value;
                    break;
                case PULSEQ_LABEL_ECO:
                    re->labelset.eco = label_value;
                    break;
                case PULSEQ_LABEL_PHS:
                    re->labelset.phs = label_value;
                    break;
                case PULSEQ_LABEL_LIN:
                    re->labelset.lin = label_value;
                    break;
                case PULSEQ_LABEL_PAR:
                    re->labelset.par = label_value;
                    break;
                case PULSEQ_LABEL_ACQ:
                    re->labelset.acq = label_value;
                    break;
                case PULSEQ_LABEL_NAV:
                    re->flag.nav = label_value;
                    break;
                case PULSEQ_LABEL_REV:
                    re->flag.rev = label_value;
                    break;
                case PULSEQ_LABEL_SMS:
                    re->flag.sms = label_value;
                    break;
                case PULSEQ_LABEL_REF:
                    re->flag.ref = label_value;
                    break;
                case PULSEQ_LABEL_IMA:
                    re->flag.ima = label_value;
                    break;
                case PULSEQ_LABEL_NOISE:
                    re->flag.noise = label_value;
                    break;
                case PULSEQ_LABEL_PMC:
                    re->flag.pmc = label_value;
                    break;
                case PULSEQ_LABEL_NOROT:
                    re->flag.norot = label_value;
                    break;
                case PULSEQ_LABEL_NOPOS:
                    re->flag.nopos = label_value;
                    break;
                case PULSEQ_LABEL_NOSCL:
                    re->flag.noscl = label_value;
                    break;
                case PULSEQ_LABEL_ONCE:
                    re->flag.once = label_value;
                    break;
                case PULSEQ_LABEL_TRID:
                    re->flag.trid = label_value;
                    break;
                case PULSEQ_LABEL_MODULE:
                    re->flag.module_id = label_value;
                    break;
                default:
                    break;
                }
            }
            break;
        case PULSEQ_EXT_LABELINC:
            if (seq->labelinc_library && ref_idx < seq->labelinc_library_size)
            {
                label_value = (int)seq->labelinc_library[ref_idx][0];
                label_id = (int)seq->labelinc_library[ref_idx][1];
                switch (label_id)
                {
                case PULSEQ_LABEL_SLC:
                    re->labelinc.slc = label_value;
                    break;
                case PULSEQ_LABEL_SEG:
                    re->labelinc.seg = label_value;
                    break;
                case PULSEQ_LABEL_REP:
                    re->labelinc.rep = label_value;
                    break;
                case PULSEQ_LABEL_AVG:
                    re->labelinc.avg = label_value;
                    break;
                case PULSEQ_LABEL_SET:
                    re->labelinc.set = label_value;
                    break;
                case PULSEQ_LABEL_ECO:
                    re->labelinc.eco = label_value;
                    break;
                case PULSEQ_LABEL_PHS:
                    re->labelinc.phs = label_value;
                    break;
                case PULSEQ_LABEL_LIN:
                    re->labelinc.lin = label_value;
                    break;
                case PULSEQ_LABEL_PAR:
                    re->labelinc.par = label_value;
                    break;
                case PULSEQ_LABEL_ACQ:
                    re->labelinc.acq = label_value;
                    break;
                default:
                    break;
                }
            }
            break;
        case PULSEQ_EXT_ROTATION:
            re->rotation_index = ref_idx;
            break;
        case PULSEQ_EXT_RF_SHIM:
            re->rf_shim_index = ref_idx;
            break;
        case PULSEQ_EXT_TRIGGER:
            re->trigger_index = ref_idx;
            break;
        case PULSEQ_EXT_DELAY:
            re->soft_delay_index = ref_idx;
            break;
        default:
            break;
        }
    }
}

/* ================================================================== */
/*  MD5 signature verification                                        */
/* ================================================================== */

int pulseq_verify_signature(const char *file_path)
{
    FILE *f;
    long sig_offset, hash_start;
    char line[PULSEQ_MAX_LINE_LENGTH];
    char stored_hash[33];
    unsigned char digest[16];
    struct MD5Context ctx;
    unsigned char buf[4096];
    size_t to_read, n;
    long remaining;
    char computed[33];
    int i;
    char *p;

    if (!file_path)
        return PULSEQ_ERR_INVALID_ARGUMENT;
    f = fopen(file_path, "rb");
    if (!f)
        return PULSEQ_ERR_FILE_NOT_FOUND;

    /* locate [SIGNATURE] by scanning from start */
    sig_offset = -1;
    while (fgets(line, sizeof(line), f))
    {
        p = line;
        while (*p == ' ' || *p == '\t')
            p++;
        if (strncmp(p, "[SIGNATURE]", 11) == 0)
        {
            sig_offset = ftell(f) - (long)strlen(line);
            break;
        }
    }
    if (sig_offset < 0)
    {
        fclose(f);
        return PULSEQ_ERR_SIGNATURE_MISSING;
    }

    /* read stored hash from section body */
    stored_hash[0] = '\0';
    while (fgets(line, sizeof(line), f))
    {
        p = line;
        while (*p == ' ' || *p == '\t')
            p++;
        if (*p == '#' || *p == '\n' || *p == '\r' || *p == '\0')
            continue;
        if (strncmp(p, "Type", 4) == 0)
            continue;
        if (strncmp(p, "Hash", 4) == 0)
        {
            p += 4;
            while (*p == ' ' || *p == '\t')
                p++;
            strncpy(stored_hash, p, 32);
            stored_hash[32] = '\0';
            /* strip trailing whitespace */
            for (i = 31; i >= 0 &&
                 (stored_hash[i] == ' ' || stored_hash[i] == '\n' || stored_hash[i] == '\r');
                 --i)
                stored_hash[i] = '\0';
            break;
        }
    }
    if (stored_hash[0] == '\0')
    {
        fclose(f);
        return PULSEQ_ERR_SIGNATURE_MISSING;
    }

    /*
     * Hash bytes [0 .. sig_offset - 2] inclusive.
     * The newline preceding [SIGNATURE] belongs to the signature
     * and must be stripped, so we hash up to sig_offset - 1 bytes
     * (i.e. the \n at position sig_offset-1 is excluded).
     */
    hash_start = sig_offset - 1;
    if (hash_start <= 0)
    {
        fclose(f);
        return PULSEQ_ERR_SIGNATURE_MISSING;
    }

    MD5Init(&ctx);
    if (fseek(f, 0L, SEEK_SET) != 0)
    {
        fclose(f);
        return PULSEQ_ERR_FILE_READ_FAILED;
    }
    remaining = hash_start;
    while (remaining > 0)
    {
        to_read = (remaining > (long)sizeof(buf)) ? sizeof(buf) : (size_t)remaining;
        n = fread(buf, 1, to_read, f);
        if (n == 0)
        {
            fclose(f);
            return PULSEQ_ERR_FILE_READ_FAILED;
        }
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
        return PULSEQ_ERR_SIGNATURE_MISMATCH;

    return PULSEQ_SUCCESS;
}

/* ================================================================== */
/*  Lightweight scan-time query                                       */
/* ================================================================== */

/*
 * Open a single .seq file and parse ONLY version + definitions.
 * Fills a temporary seq_file just enough to read reserved definitions.
 */
int pulseq_read_definitions_only(pulseq_file *seq, const char *path)
{
    FILE *f;
    if (!seq || !path)
        return PULSEQ_ERR_INVALID_ARGUMENT;
    f = fopen(path, "r");
    if (!f)
        return PULSEQ_ERR_FILE_NOT_FOUND;

    get_section_offsets(seq, f);
    read_version(seq, f);
    if (seq->version_combined < 1005000)
    {
        fclose(f);
        return PULSEQ_ERR_UNSUPPORTED_VERSION;
    }
    read_definitions_library(seq, f);
    read_definitions(seq);
    fclose(f);
    return PULSEQ_SUCCESS;
}
