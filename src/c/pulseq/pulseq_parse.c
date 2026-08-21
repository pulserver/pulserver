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
/*  Numeric field scanning (static)                                   */
/* ================================================================== */

/*
 * The library sections are the whole cost of reading a large .seq: a
 * two-million-block scan is two million [BLOCKS] rows of seven fields each,
 * plus one [EXTENSIONS] row per labelled block.  Every one of those fields
 * used to go through sscanf, which re-parses its format string and consults
 * the locale on each call; that alone was six sevenths of the parse.
 *
 * These two scanners read a field and advance the cursor past it, and are
 * the only thing the hot loops call.
 */

/** Read a decimal integer. Sets *ok to 0 (leaving the cursor put) if the
 *  cursor is not on one. */
static long scan_long(char **cursor, int *ok)
{
    char *p = *cursor;
    long value = 0;
    int negative = 0;
    int digits = 0;

    while (*p == ' ' || *p == '\t')
        p++;
    if (*p == '-')
    {
        negative = 1;
        p++;
    }
    else if (*p == '+')
        p++;
    while (*p >= '0' && *p <= '9')
    {
        value = value * 10 + (*p - '0');
        p++;
        digits++;
    }
    if (digits == 0)
    {
        *ok = 0;
        return 0;
    }
    *cursor = p;
    *ok = 1;
    return negative ? -value : value;
}

/*
 * Read one numeric field.
 *
 * Whole numbers -- which is every field of [BLOCKS], of [EXTENSIONS] and of the
 * label libraries, so all of the hot ones -- are accumulated as an integer and
 * converted. That conversion rounds to nearest, which for a value the decimal
 * names exactly is the same value sscanf produced, so the fast path is
 * bit-identical by construction rather than by measurement.
 *
 * Everything else -- a fraction, an exponent, a special form, or a run of
 * digits long enough to overflow the accumulator -- goes to sscanf, the same
 * call the reader has always made. Keeping the fallback on sscanf rather than
 * on strtod is deliberate: in the float build strtod would round the decimal to
 * a double and then to a float, and double rounding is not always the same
 * answer. Shape samples are the only fields that reach it in practice, and they
 * are exactly the ones the double build exists to keep: a .seq writes them to
 * nine significant figures, which is more than a float can hold.
 */
static PULSEQ_REAL scan_float(char **cursor, int *ok)
{
    char *p = *cursor;
    char *start;
    long value = 0;
    int negative = 0;
    int digits = 0;
    int consumed = 0;
    PULSEQ_REAL parsed;

    while (*p == ' ' || *p == '\t')
        p++;
    start = p;
    if (*p == '-')
    {
        negative = 1;
        p++;
    }
    else if (*p == '+')
        p++;
    while (*p >= '0' && *p <= '9')
    {
        value = value * 10 + (*p - '0');
        p++;
        digits++;
    }
    if (digits > 0 && digits <= 18 && *p != '.' && *p != 'e' && *p != 'E')
    {
        *cursor = p;
        *ok = 1;
        return negative ? -(PULSEQ_REAL)value : (PULSEQ_REAL)value;
    }

    if (sscanf(start, PULSEQ_REAL_FMT "%n", &parsed, &consumed) != 1)
    {
        *ok = 0;
        return (PULSEQ_REAL)0.0;
    }
    *cursor = start + consumed;
    *ok = 1;
    return parsed;
}

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
    for (i = 0; i < PULSEQ_LABEL_ID_MAX; i++)
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

/* Not static: the binary reader resets the same way, both on entry and when
 * a partially built file has to be thrown away. */
void pulseq__file_reset(pulseq_file *seq)
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
    pulseq__file_reset(seq);
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
            pulseq__file_reset(&coll->sequences[i]);
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
/*  Whole-line reading                                                */
/* ================================================================== */

/**
 * @brief One whole logical line, however long.
 *
 * @c fgets stops at the end of its buffer and hands the remainder back on the
 * next call.  For a section whose rows are bounded that never happens; for one
 * whose rows are not, a long row silently becomes two, and the tail is read as
 * a row of its own.  Two sections here are unbounded by construction:
 * @c [DEFINITIONS] carries one value per slice (@c SlicePositions) and
 * @c [RF_SHIMS] two per transmit channel.  Nothing raises: the file parses,
 * the values past the cut are gone, and the tail arrives as a definition
 * named after whichever digits the split landed between.
 *
 * The common path allocates nothing -- a row that fits is returned in @p buf,
 * which is every row in every other section.  Only a row that did not fit
 * reaches the heap, and the block is reused across calls, so a section of long
 * rows allocates a handful of times rather than once per row.
 *
 * @param buf       caller's stack buffer, used whenever the row fits.
 * @param heap      reused overflow block; the caller frees it once, after the
 *                  section.  Must be @c NULL and @p heap_cap 0 on first use.
 * @param oom       set to 1 if the overflow block could not be grown -- so a
 *                  caller can tell "the section ended" from "the row was lost".
 *
 * @return the line, NUL-terminated, or @c NULL at end of file or on failure.
 */
static char *read_full_line(
    FILE *f,
    char *buf,
    size_t buf_size,
    char **heap,
    size_t *heap_cap,
    int *oom)
{
    size_t len, used, need;
    char *grown;

    *oom = 0;
    if (!fgets(buf, (int)buf_size, f))
        return NULL;

    len = strlen(buf);
    /* Short of the buffer's end means fgets stopped for a reason of its own --
     * a newline, or the end of the file -- so the row is whole either way. */
    if (len + 1 < buf_size || (len > 0 && buf[len - 1] == '\n'))
        return buf;

    used = 0;
    for (;;)
    {
        need = used + len + 1;
        if (need > *heap_cap)
        {
            size_t cap = (*heap_cap > 0) ? *heap_cap : buf_size;
            while (cap < need)
                cap *= 2;
            grown = (char *)PULSEQ_ALLOC(cap);
            if (!grown)
            {
                *oom = 1;
                return NULL;
            }
            if (*heap)
            {
                memcpy(grown, *heap, used);
                PULSEQ_FREE(*heap);
            }
            *heap = grown;
            *heap_cap = cap;
        }
        memcpy(*heap + used, buf, len + 1);
        used += len;

        if (used > 0 && (*heap)[used - 1] == '\n')
            break;
        if (!fgets(buf, (int)buf_size, f))
            break;
        len = strlen(buf);
    }
    return *heap;
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
    int sec, idx, i, ok;
    char *p;
    PULSEQ_REAL *arr;

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
            idx = (int)scan_long(&p, &ok);
            if (ok && idx > max_idx)
                max_idx = idx;
        }
    }
    if (max_idx <= 0)
    {
        *target = NULL;
        *target_count = 0;
        return 0;
    }
    arr = (PULSEQ_REAL *)PULSEQ_ALLOC(sizeof(PULSEQ_REAL) * n * max_idx);
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
    char *heap = NULL;
    size_t heap_cap = 0;
    int oom = 0;
    char *row;
    int count = 0;
    char *p;
    char *name_tok;
    pulseq_definition *defs;

    if (!f || offset < 0 || !target || !target_count)
        return 1;
    if (fseek(f, offset, SEEK_SET) != 0)
        return 1;
    if (!read_full_line(f, line, sizeof(line), &heap, &heap_cap, &oom))
    {
        if (heap)
            PULSEQ_FREE(heap);
        return 1;
    }
    while ((row = read_full_line(f, line, sizeof(line), &heap, &heap_cap, &oom)) != NULL)
    {
        p = row;
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
    if (heap)
        PULSEQ_FREE(heap);
    if (oom)
        return 1;
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
    int n, i, j, idx = 0, ok;
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
            p += 8;
            idx = (int)scan_long(&p, &ok);
            if (ok && idx > max_idx)
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
            p += 8;
            idx = (int)scan_long(&p, &ok);
            if (ok)
            {
                shapes[idx - 1].num_samples = 0;
                shapes[idx - 1].num_uncompressed_samples = 0;
                shapes[idx - 1].samples = NULL;
            }
        }
        else if (strncmp(p, "num_samples", 11) == 0)
        {
            p += 11;
            n = (int)scan_long(&p, &ok);
            if (ok)
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
            shapes[i].samples = (PULSEQ_REAL *)PULSEQ_ALLOC(sizeof(PULSEQ_REAL) * n);
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
    char *heap = NULL;
    size_t heap_cap = 0;
    int oom = 0;
    char *row;
    int max_idx = -1;
    char *p;
    int idx, i;
    pulseq_rf_shim_entry *arr;

    if (!f || !target || !target_count)
        return 1;
    if (fseek(f, offset, SEEK_SET) != 0)
        return 1;
    if (!read_full_line(f, line, sizeof(line), &heap, &heap_cap, &oom))
    {
        if (heap)
            PULSEQ_FREE(heap);
        return 1;
    }
    while ((row = read_full_line(f, line, sizeof(line), &heap, &heap_cap, &oom)) != NULL)
    {
        p = row;
        while (*p == ' ' || *p == '\t')
            p++;
        if (*p == '[' || *p == 'e')
            break;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#')
            continue;
        if (sscanf(p, "%d", &idx) == 1 && idx > max_idx)
            max_idx = idx;
    }
    if (heap)
        PULSEQ_FREE(heap);
    if (oom)
        return 1;
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
    int idx, parsed, col, offset_col, ok;
    PULSEQ_REAL vals[PULSEQ_MAX_SCALE_SIZE];
    char *scan_ptr;
    char *p;
    PULSEQ_REAL v;
    PULSEQ_REAL *arr = (PULSEQ_REAL *)target;

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
        idx = (int)scan_long(&p, &ok);
        if (!ok)
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
            v = scan_float(&scan_ptr, &ok);
            if (!ok)
                break;
            vals[col] = v;
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
            arr[(idx - 1) * n + 0] = (PULSEQ_REAL)flag;
    }
    return 0;
}

/*
 * Scan the file for its section headers.
 *
 * `definitions_only` stops the scan once [DEFINITIONS] has been read and the
 * next section begins: a caller that wants nothing but the definitions -- the
 * chain walk, and pulseg's option setup -- has no reason to walk eighty
 * megabytes of block rows to find headers it will not use.
 */
static void get_section_offsets(pulseq_file *seq, FILE *f, int definitions_only)
{
    char line[PULSEQ_MAX_LINE_LENGTH];
    char *p;
    long pos;
    char ext_name[PULSEQ_EXT_NAME_LENGTH];
    int ext_id, ext_enum;

    if (fseek(f, 0L, SEEK_SET) != 0)
        return;

    /* The line start is tracked rather than derived from ftell(), which is a
     * library call per line and was measurable on its own at this size. */
    pos = 0;
    while (fgets(line, sizeof(line), f))
    {
        long line_start = pos;
        pos += (long)strlen(line);
        p = line;
        while (*p == ' ' || *p == '\t')
            p++;

        if (*p == '[')
        {
            if (definitions_only && seq->offsets.definitions >= 0)
                break;
            if (strncmp(p, "[VERSION]", 9) == 0)
                seq->offsets.version = line_start;
            else if (strncmp(p, "[DEFINITIONS]", 13) == 0)
                seq->offsets.definitions = line_start;
            else if (strncmp(p, "[BLOCKS]", 8) == 0)
                seq->offsets.blocks = line_start;
            else if (strncmp(p, "[RF]", 4) == 0)
                seq->offsets.rf = line_start;
            else if (strncmp(p, "[GRADIENTS]", 11) == 0)
                seq->offsets.grad = line_start;
            else if (strncmp(p, "[TRAP]", 6) == 0)
                seq->offsets.trap = line_start;
            else if (strncmp(p, "[ADC]", 5) == 0)
                seq->offsets.adc = line_start;
            else if (strncmp(p, "[EXTENSIONS]", 12) == 0)
                seq->offsets.extensions = line_start;
            else if (strncmp(p, "[SHAPES]", 8) == 0)
                seq->offsets.shapes = line_start;
            else if (strncmp(p, "[SIGNATURE]", 11) == 0)
                seq->offsets.signature = line_start;
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
                    seq->offsets.triggers = line_start;
                    break;
                case PULSEQ_EXT_ROTATION:
                    seq->offsets.rotations = line_start;
                    break;
                case PULSEQ_EXT_LABELSET:
                    seq->offsets.labelset = line_start;
                    break;
                case PULSEQ_EXT_LABELINC:
                    seq->offsets.labelinc = line_start;
                    break;
                case PULSEQ_EXT_RF_SHIM:
                    seq->offsets.rfshim = line_start;
                    break;
                case PULSEQ_EXT_DELAY:
                    seq->offsets.delays = line_start;
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
    char *heap = NULL;
    size_t heap_cap = 0;
    int oom = 0;
    char *row;
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
    if (!read_full_line(f, line, sizeof(line), &heap, &heap_cap, &oom))
    {
        if (heap)
            PULSEQ_FREE(heap);
        return;
    }

    while ((row = read_full_line(f, line, sizeof(line), &heap, &heap_cap, &oom)) != NULL)
    {
        p = row;
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
    if (heap)
        PULSEQ_FREE(heap);
    seq->is_definitions_library_parsed = 1;
}

/* Not static: the binary reader (pulseq_binary.c) builds the same generic
 * definitions library and then needs the same reserved-key extraction. */
void pulseq__read_definitions(pulseq_file *seq)
{
    int i;
    int nvals;
    char *key;
    char *value;
    PULSEQ_REAL temp[3];

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
            seq->reserved_definitions_library.gradient_raster_time =
                (PULSEQ_REAL)(atof(value) * 1e6);
        }
        else if (strcmp(key, "RadiofrequencyRasterTime") == 0)
        {
            seq->reserved_definitions_library.radiofrequency_raster_time =
                (PULSEQ_REAL)(atof(value) * 1e6);
        }
        else if (strcmp(key, "AdcRasterTime") == 0)
        {
            seq->reserved_definitions_library.adc_raster_time = (PULSEQ_REAL)(atof(value) * 1e6);
        }
        else if (strcmp(key, "BlockDurationRaster") == 0)
        {
            seq->reserved_definitions_library.block_duration_raster =
                (PULSEQ_REAL)(atof(value) * 1e6);
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
                temp[0] = (PULSEQ_REAL)atof(seq->definitions_library[i].value[0]);
                temp[1] = (PULSEQ_REAL)atof(seq->definitions_library[i].value[1]);
                temp[2] = (PULSEQ_REAL)atof(seq->definitions_library[i].value[2]);
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
                    (PULSEQ_REAL)atof(seq->definitions_library[i].value[0]);
                seq->reserved_definitions_library.matrix[1] =
                    (PULSEQ_REAL)atof(seq->definitions_library[i].value[1]);
                seq->reserved_definitions_library.matrix[2] =
                    (PULSEQ_REAL)atof(seq->definitions_library[i].value[2]);
            }
        }
        else if (strcmp(key, "NavFOV") == 0)
        {
            if (nvals >= 3)
            {
                seq->reserved_definitions_library.nav_fov[0] =
                    (PULSEQ_REAL)atof(seq->definitions_library[i].value[0]) * 100.0f;
                seq->reserved_definitions_library.nav_fov[1] =
                    (PULSEQ_REAL)atof(seq->definitions_library[i].value[1]) * 100.0f;
                seq->reserved_definitions_library.nav_fov[2] =
                    (PULSEQ_REAL)atof(seq->definitions_library[i].value[2]) * 100.0f;
            }
        }
        else if (strcmp(key, "NavMatrix") == 0)
        {
            if (nvals >= 3)
            {
                seq->reserved_definitions_library.nav_matrix[0] =
                    (PULSEQ_REAL)atof(seq->definitions_library[i].value[0]);
                seq->reserved_definitions_library.nav_matrix[1] =
                    (PULSEQ_REAL)atof(seq->definitions_library[i].value[1]);
                seq->reserved_definitions_library.nav_matrix[2] =
                    (PULSEQ_REAL)atof(seq->definitions_library[i].value[2]);
            }
        }
        else if (strcmp(key, "TotalDuration") == 0)
        {
            seq->reserved_definitions_library.total_duration = (PULSEQ_REAL)atof(value);
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
    PULSEQ_REAL block_vals[7] = {1, 1, 1, 1, 1, 1, 1};
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
    PULSEQ_REAL rf_vals[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
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
                    PULSEQ_REAL dummy;
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
                    /* skip 10 numeric columns */
                    scan_p = up;
                    for (col = 0; col < 10; ++col)
                    {
                        consumed = 0;
                        if (sscanf(scan_p, PULSEQ_REAL_FMT "%n", &dummy, &consumed) != 1)
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
                        else if (tag == 'p')
                            use_val = PULSEQ_RF_USE_PREPARATION;
                        else if (tag == 'o')
                            use_val = PULSEQ_RF_USE_OTHER;
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
    static const PULSEQ_REAL grad_vals[6] = {1, 1, 1, 1, 1, 1};
    static const PULSEQ_REAL trap_vals[5] = {1, 1, 1, 1, 1};
    pulseq_scale grad_s, trap_s;

    grad_s.size = 6;
    grad_s.values = (PULSEQ_REAL *)grad_vals;
    trap_s.size = 5;
    trap_s.values = (PULSEQ_REAL *)trap_vals;

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
    static const PULSEQ_REAL adc_vals[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    pulseq_scale s;
    s.size = 8;
    s.values = (PULSEQ_REAL *)adc_vals;

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
    int ret, ok;
    char line[PULSEQ_MAX_LINE_LENGTH];
    int shape_index = 0, sample_index = 0;
    char *p;
    PULSEQ_REAL val;

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
            p += 8;
            shape_index = (int)scan_long(&p, &ok);
            if (ok)
                sample_index = 0;
            continue;
        }
        if (strncmp(p, "num_samples", 11) == 0)
            continue;
        if (shape_index > 0 && shape_index <= seq->shapes_library_size)
        {
            val = scan_float(&p, &ok);
            if (ok)
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
    PULSEQ_REAL val;
    char label[PULSEQ_LABEL_NAME_LENGTH];
    PULSEQ_REAL *arr = (PULSEQ_REAL *)target;

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
        if (sscanf(p, "%d " PULSEQ_REAL_FMT " %31s", &idx, &val, label) != 3)
            continue;
        if (idx <= 0 || idx > target_count)
            continue;
        /* Registering rather than looking up is what keeps a label name the
         * file defines for itself: it gets an id past the built-in list, which
         * nothing downstream switches on, and the name can be read back off
         * that id when the sequence is written out again. */
        label_code = pulseq_label_register_name(label);
        if (label_code > 0 && label_code <= PULSEQ_LABEL_ID_MAX)
            is_label_defined[label_code - 1] = 1;
        arr[(idx - 1) * n + 0] = val;
        arr[(idx - 1) * n + 1] = (PULSEQ_REAL)label_code;
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
    PULSEQ_REAL num_id, offset_val, scale_val;
    char hint[PULSEQ_SOFT_DELAY_HINT_LENGTH];
    PULSEQ_REAL *arr = (PULSEQ_REAL *)target;

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
        if (sscanf(
                p,
                "%d " PULSEQ_REAL_FMT " " PULSEQ_REAL_FMT " " PULSEQ_REAL_FMT " %31s",
                &idx,
                &num_id,
                &offset_val,
                &scale_val,
                hint) != 5)
            continue;
        if (idx <= 0 || idx > target_count)
            continue;
        hint_code = pulseq_hint_id_for_name(hint);
        if (hint_code > 0)
            is_delay_defined[hint_code - 1] = 1;
        arr[(idx - 1) * n + 0] = num_id;
        arr[(idx - 1) * n + 1] = offset_val;
        arr[(idx - 1) * n + 2] = scale_val;
        arr[(idx - 1) * n + 3] = (PULSEQ_REAL)hint_code;
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
    char *heap = NULL;
    size_t heap_cap = 0;
    int oom = 0;
    char *row;
    int status = 0;
    char *p;
    int idx, n_ch, i, consumed;
    PULSEQ_REAL val;

    if (!f || !target)
        return 1;
    if (fseek(f, offset, SEEK_SET) != 0)
        return 1;
    if (!read_full_line(f, line, sizeof(line), &heap, &heap_cap, &oom))
    {
        if (heap)
            PULSEQ_FREE(heap);
        return 1;
    }

    while ((row = read_full_line(f, line, sizeof(line), &heap, &heap_cap, &oom)) != NULL)
    {
        p = row;
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

        /* Past the compile-time channel cap there is nowhere to put the
         * values.  Returning 2 rather than 1 says so: the caller distinguishes
         * "this section held nothing" -- ordinary, most files have no shims --
         * from "this section held something that cannot be read", which the
         * binary reader already reports as a read failure. */
        if (n_ch > PULSEQ_MAX_RF_SHIM_CHANNELS)
        {
            status = 2;
            break;
        }
        target[idx - 1].num_channels = n_ch;
        for (i = 0; i < 2 * n_ch; i++)
        {
            consumed = 0;
            if (sscanf(p, PULSEQ_REAL_FMT "%n", &val, &consumed) != 1)
                break;
            target[idx - 1].values[i] = val;
            p += consumed;
            while (*p == ' ' || *p == '\t')
                p++;
        }
        /* A row that ran out of numbers before its declared channel count is
         * a truncated row, not a shorter shim -- the values that are missing
         * would otherwise be read as whatever the array last held. */
        if (i < 2 * n_ch)
        {
            status = 2;
            break;
        }
    }
    if (heap)
        PULSEQ_FREE(heap);
    if (oom)
        status = 2;
    return status;
}

/**
 * @return @c PULSEQ_SUCCESS, or an error for a section that is present but
 *         cannot be read.  An *absent* section is not an error -- most files
 *         carry no shims and no triggers -- which is why the early returns
 *         below are successes and only the shim reader's fatal code is not.
 */
static int read_extensions_library(pulseq_file *seq, FILE *f)
{
    int ret, n;
    static const PULSEQ_REAL ext_vals[3] = {1, 1, 1};
    static const PULSEQ_REAL trig_vals[4] = {1, 1, 1, 1};
    static const PULSEQ_REAL rot_vals[4] = {1, 1, 1, 1};
    pulseq_scale ext_s, trig_s, rot_s;

    ext_s.size = 3;
    ext_s.values = (PULSEQ_REAL *)ext_vals;
    trig_s.size = 4;
    trig_s.values = (PULSEQ_REAL *)trig_vals;
    rot_s.size = 4;
    rot_s.values = (PULSEQ_REAL *)rot_vals;

    if (seq->is_extensions_library_parsed)
        return PULSEQ_SUCCESS;
    if (seq->offsets.extensions < 0)
    {
        seq->is_extensions_library_parsed = 1;
        return PULSEQ_SUCCESS;
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
        return PULSEQ_SUCCESS;

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
            return PULSEQ_SUCCESS;
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
            return PULSEQ_SUCCESS;
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
            return PULSEQ_SUCCESS;
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
            return PULSEQ_SUCCESS;
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
            return PULSEQ_SUCCESS;
    }
    if (seq->offsets.rfshim >= 0)
    {
        ret = init_rf_shim_library(
            &seq->rf_shim_library,
            &seq->rf_shim_library_size,
            f,
            seq->offsets.rfshim);
        if (ret != 0)
            return PULSEQ_SUCCESS;
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
        return PULSEQ_SUCCESS;

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
            return PULSEQ_SUCCESS;
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
            return PULSEQ_SUCCESS;
        /* The quaternion is kept exactly as the file wrote it.
         *
         * It used to be normalized here, which was both lossy and inconsistent:
         * lossy because a sequence read and written again no longer carried the
         * numbers it arrived with, and inconsistent because the loop started at
         * row 1, so the first rotation in every file was left alone while the
         * rest were rewritten.
         *
         * Nothing needed it.  The one consumer of this library,
         * copy_rotation_library() in pulseg_dedup.c, hands each row to
         * pulseg__quaternion_to_matrix(), which normalizes internally before
         * building the matrix -- so the rotation the scanner plays is unchanged
         * either way. */
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
            return PULSEQ_SUCCESS;
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
            return PULSEQ_SUCCESS;
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
            return PULSEQ_SUCCESS;
    }
    if (seq->offsets.rfshim >= 0)
    {
        ret = read_rf_shim_library(
            seq->rf_shim_library,
            seq->rf_shim_library_size,
            f,
            seq->offsets.rfshim);
        if (ret == 2)
            return PULSEQ_ERR_FILE_READ_FAILED;
        if (ret != 0)
            return PULSEQ_SUCCESS;
    }

    /* Build extension LUT */
    for (n = 0; n < 8; n++)
    {
        if (seq->extension_lut_size < seq->extension_map[n])
            seq->extension_lut_size = seq->extension_map[n];
    }
    if (seq->extension_lut_size > 0)
    {
        seq->extension_lut =
            (int *)PULSEQ_ALLOC(sizeof(int) * (size_t)(seq->extension_lut_size + 1));
        if (!seq->extension_lut)
            return PULSEQ_ERR_ALLOC_FAILED;
        /* A type number the file never declared still indexes this table, so
         * every entry has to mean something before any of them are set. */
        for (n = 0; n <= seq->extension_lut_size; ++n)
            seq->extension_lut[n] = PULSEQ_EXT_UNKNOWN;
        for (n = 0; n < 8; n++)
        {
            if (seq->extension_map[n] > 0)
                seq->extension_lut[seq->extension_map[n]] = n;
        }
    }

    seq->is_extensions_library_parsed = 1;
    return PULSEQ_SUCCESS;
}

/* ================================================================== */
/*  Shape decompression (cross-file)                                  */
/* ================================================================== */

int pulseq_decompress_shape(pulseq_shape *result, const pulseq_shape *encoded, PULSEQ_REAL scale)
{
    int i, rep;
    const PULSEQ_REAL *packed;
    int num_packed, num_samples;
    int count_pack = 1;
    int count_unpack = 1;
    PULSEQ_REAL *unpacked;

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
        result->samples = (PULSEQ_REAL *)PULSEQ_ALLOC(sizeof(PULSEQ_REAL) * encoded->num_samples);
        if (!result->samples)
            return 0;
        for (i = 0; i < encoded->num_samples; ++i)
            result->samples[i] = encoded->samples[i] * scale;
        return 1;
    }

    unpacked = (PULSEQ_REAL *)PULSEQ_ALLOC(sizeof(PULSEQ_REAL) * num_samples);
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
            if (fabs(packed[count_pack + 1] + 2 - (PULSEQ_REAL)rep) > 1e-6f)
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

/* ================================================================== */
/*  Reference validation                                              */
/* ================================================================== */

/*
 * An id in a Pulseq file is a 1-based row number in another library, and 0
 * means "no event".  Nothing in the format stops a file naming a row that is
 * not there, and every consumer downstream indexes with what it is given --
 * so a file that does is caught here, once, rather than read out of bounds
 * later by whichever consumer gets to it first.
 */
static int id_in_range(PULSEQ_REAL id, int library_size)
{
    int n = (int)id;
    return n >= 0 && n <= library_size;
}

/*
 * Which library an extension row's reference names depends on its type, so
 * the target size does too.  An extension this reader does not know carries a
 * reference it cannot resolve and therefore cannot judge -- and never
 * dereferences it either, so it is left alone.
 */
static int ext_ref_in_range(const pulseq_file *seq, int ext_type, int ref)
{
    switch (ext_type)
    {
    case PULSEQ_EXT_LABELSET:
        return ref <= seq->labelset_library_size;
    case PULSEQ_EXT_LABELINC:
        return ref <= seq->labelinc_library_size;
    case PULSEQ_EXT_ROTATION:
        return ref <= seq->rotation_library_size;
    case PULSEQ_EXT_RF_SHIM:
        return ref <= seq->rf_shim_library_size;
    case PULSEQ_EXT_TRIGGER:
        return ref <= seq->trigger_library_size;
    case PULSEQ_EXT_DELAY:
        return ref <= seq->soft_delay_library_size;
    default:
        return 1;
    }
}

static int check_extension_references(const pulseq_file *seq)
{
    int i, type, ref, next;
    PULSEQ_REAL *row;

    if (!seq->is_extensions_library_parsed || !seq->extensions_library)
        return PULSEQ_SUCCESS;

    for (i = 0; i < seq->extensions_library_size; ++i)
    {
        row = seq->extensions_library[i];
        type = (int)row[0];
        ref = (int)row[1];
        next = (int)row[2];

        if (ref < 0 || next < 0 || next > seq->extensions_library_size)
            return PULSEQ_ERR_DANGLING_ID;
        if (seq->extension_lut && type >= 0 && type <= seq->extension_lut_size &&
            !ext_ref_in_range(seq, seq->extension_lut[type], ref))
            return PULSEQ_ERR_DANGLING_ID;
    }
    return PULSEQ_SUCCESS;
}

static int check_references(const pulseq_file *seq)
{
    int i;
    PULSEQ_REAL *row;

    if (!seq->is_block_library_parsed || !seq->block_library)
        return PULSEQ_SUCCESS;

    for (i = 0; i < seq->num_blocks; ++i)
    {
        row = seq->block_library[i];
        if (!id_in_range(row[1], seq->rf_library_size) ||
            !id_in_range(row[2], seq->grad_library_size) ||
            !id_in_range(row[3], seq->grad_library_size) ||
            !id_in_range(row[4], seq->grad_library_size) ||
            !id_in_range(row[5], seq->adc_library_size) ||
            !id_in_range(row[6], seq->extensions_library_size))
            return PULSEQ_ERR_DANGLING_ID;
    }

    for (i = 0; i < seq->rf_library_size; ++i)
    {
        row = seq->rf_library[i];
        if (!id_in_range(row[1], seq->shapes_library_size) ||
            !id_in_range(row[2], seq->shapes_library_size) ||
            !id_in_range(row[3], seq->shapes_library_size))
            return PULSEQ_ERR_DANGLING_ID;
    }

    /* An arbitrary gradient names its waveform in column 4 and its optional
     * time shape in column 5; a trapezoid carries times in those columns and
     * names no shape at all. */
    for (i = 0; i < seq->grad_library_size; ++i)
    {
        row = seq->grad_library[i];
        if ((int)row[0] == 0)
            continue;
        if (!id_in_range(row[4], seq->shapes_library_size) ||
            !id_in_range(row[5], seq->shapes_library_size))
            return PULSEQ_ERR_DANGLING_ID;
    }

    for (i = 0; i < seq->adc_library_size; ++i)
        if (!id_in_range(seq->adc_library[i][7], seq->shapes_library_size))
            return PULSEQ_ERR_DANGLING_ID;

    return check_extension_references(seq);
}

int pulseq__check_references(const pulseq_file *seq)
{
    if (!seq)
        return PULSEQ_ERR_NULL_POINTER;
    return check_references(seq);
}

int pulseq_read_from_buffer(pulseq_file *seq, FILE *f)
{
    unsigned char magic[8];
    size_t got;
    int code;

    if (!seq || !f)
        return PULSEQ_ERR_NULL_POINTER;
    pulseq__file_reset(seq);
    if (seq->file_path)
    {
        PULSEQ_FREE(seq->file_path);
        seq->file_path = NULL;
    }

    /* Same content-based dispatch as pulseq_read(), for the in-memory case:
     * a caller holding a buffer knows even less about its format than one
     * holding a path. */
    got = fread(magic, 1, 8, f);
    if (fseek(f, 0L, SEEK_SET) != 0)
        return PULSEQ_ERR_FILE_READ_FAILED;
    if (got == 8 && magic[0] == 0x01 && magic[7] == 0x02 && memcmp(magic + 1, "pulseq", 6) == 0)
        return pulseq_read_binary_from_buffer(seq, f);

    get_section_offsets(seq, f, /*definitions_only=*/0);
    read_version(seq, f);
    if (seq->version_combined < 1005000)
        return PULSEQ_ERR_UNSUPPORTED_VERSION;
    read_definitions_library(seq, f);
    pulseq__read_definitions(seq);
    read_block_library(seq, f);
    read_rf_library(seq, f);
    read_grad_library(seq, f);
    read_adc_library(seq, f);
    read_shapes_library(seq, f);
    code = read_extensions_library(seq, f);
    if (PULSEQ_FAILED(code))
        return code;
    return check_references(seq);
}

int pulseq_read(pulseq_file *seq, const char *file_path)
{
    FILE *f;
    int code;
    if (!seq || !file_path)
        return PULSEQ_ERR_INVALID_ARGUMENT;
    /* Format is decided by what the file starts with, not by its extension:
     * a .seq that happens to hold binary still reads, and a caller that has
     * only a path never has to ask. */
    if (pulseq_file_is_binary(file_path))
        return pulseq_read_binary(seq, file_path);
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
        /* Only NextSequence decides the chain, and it is a definition. Reading
         * the libraries here would parse every file twice -- once to learn how
         * many there are, once for real -- which on a two-million-block scan is
         * a second or two spent building tables that are thrown away. */
        pulseq_file_init(&temp, raster);
        result = pulseq_read_definitions_only(&temp, current_path);
        if (PULSEQ_FAILED(result))
        {
            pulseq__file_reset(&temp);
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
                pulseq__file_reset(&temp);
                PULSEQ_FREE(base_path);
                return -1;
            }
        }
        pulseq__file_reset(&temp);
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
                pulseq__file_reset(&set->sequences[j]);
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
                    pulseq__file_reset(&set->sequences[j]);
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
    PULSEQ_REAL *ev;
    PULSEQ_REAL *ext_data;

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
    if (pulseq_file_is_binary(path))
        return pulseq_read_binary_definitions_only(seq, path);
    f = fopen(path, "r");
    if (!f)
        return PULSEQ_ERR_FILE_NOT_FOUND;

    get_section_offsets(seq, f, /*definitions_only=*/1);
    read_version(seq, f);
    if (seq->version_combined < 1005000)
    {
        fclose(f);
        return PULSEQ_ERR_UNSUPPORTED_VERSION;
    }
    read_definitions_library(seq, f);
    pulseq__read_definitions(seq);
    fclose(f);
    return PULSEQ_SUCCESS;
}
