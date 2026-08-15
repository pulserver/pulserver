/**
 * @file pulseq_binary.c
 * @brief Reader for the native Pulseq *binary* sequence file format.
 *
 * The binary format is upstream Pulseq's own, not something invented here.
 * Reference implementation:
 *   refcode/pulseq/matlab/+mr/@Sequence/{writeBinary,readBinary}.m
 *   refcode/pulseq/matlab/+mr/@Sequence/Sequence.m::getBinaryCodes
 *
 * Layout: an 8-byte header (0x01 "pulseq" 0x02), three int64 version fields,
 * then a flat run of sections.  Each section opens with an int64 code
 * 0xFFFFFFFF_0000000N and -- this is the part that matters for speed -- an
 * explicit int64 entry count.  Because the count is in the file, neither of
 * the text reader's two preliminary scans is needed: no header hunt to find
 * where a section starts, and no max-index pass to size its array.
 *
 * The output is the same pulseq_file the text reader produces, in the same
 * units, so everything downstream is unaware of which path was taken.  The
 * one thing that is NOT identical is precision, and it favours this path: the
 * binary format carries float64 amplitudes and picosecond int64 times, where
 * the text format carries whatever survived being printed to nine significant
 * digits.  A shape sample can therefore land one ULP away from the same
 * sample read out of a .seq -- measured at 44 samples of a 2.1M-block MPRAGE,
 * all exactly 1 ULP, with the binary value the more faithful of the two.
 *
 * Portability: the scanner target is 32-bit big-endian PowerPC compiled as
 * C89, where `long long` does not exist and `long` is 32 bits.  So no 64-bit
 * integer type is used anywhere below; int64 fields are decoded byte-by-byte
 * into a (hi, lo) pair of 32-bit halves and converted to double when a value
 * is actually needed.  A double holds every picosecond time this format can
 * express exactly (2^53 ps = 2.5 hours), which is why that is safe.
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pulseq_internal.h"

/* ================================================================== */
/*  Section codes                                                     */
/* ================================================================== */
/* Low word of the int64 section code; the high word is always 0xFFFFFFFF. */
#define PULSEQ_BIN_SECTION_HI 0xFFFFFFFFUL

#define PULSEQ_BIN_DEFINITIONS 1
#define PULSEQ_BIN_BLOCKS 2
#define PULSEQ_BIN_RF 3
#define PULSEQ_BIN_GRADIENTS 4
#define PULSEQ_BIN_TRAPEZOIDS 5
#define PULSEQ_BIN_ADC 6
#define PULSEQ_BIN_DELAYS 7 /* legacy, read and discarded */
#define PULSEQ_BIN_SHAPES 8
#define PULSEQ_BIN_EXTENSIONS 9
#define PULSEQ_BIN_TRIGGERS 10
#define PULSEQ_BIN_LABELSET 11
#define PULSEQ_BIN_LABELINC 12
#define PULSEQ_BIN_SOFTDELAYS 13
#define PULSEQ_BIN_RFSHIMS 14
#define PULSEQ_BIN_ROTATIONS 15
#define PULSEQ_BIN_LABELNAMES 16

/* Largest file label id the LABELNAMES remap covers; ids past it read
 * unmapped, exactly as a file without the section reads. */
#define PULSEQ_BIN_MAX_LABEL_REMAP 256

/* Blocks are decoded in batches rather than one fread per record; at two
 * million blocks the per-call overhead is otherwise the dominant cost. */
#define PULSEQ_BIN_BLOCK_STRIDE 32
#define PULSEQ_BIN_BLOCK_BATCH 4096
/* id + type + ref + next_id, all int32. */
#define PULSEQ_BIN_EXT_STRIDE 16
#define PULSEQ_BIN_EXT_BATCH 4096

/* A count read out of a corrupt file must not become an allocation request.
 * No real sequence approaches this, and the text reader is bounded the same
 * way by the size of the file it is scanning. */
#define PULSEQ_BIN_MAX_COUNT 200000000L

/* ================================================================== */
/*  Byte-order-explicit decoding                                      */
/* ================================================================== */

/** @brief An int64 field, kept as two 32-bit halves (no `long long` on CX). */
typedef struct
{
    unsigned long hi;
    unsigned long lo;
} bin_i64;

typedef struct
{
    FILE *f;
    int file_is_big;  /**< byte order of the file itself */
    int reverse_real; /**< file and host disagree, so IEEE real bytes need flipping */

    /* LABELNAMES remap: the file's label ids resolved through the names it
     * declares, so a name outside the builtin table reads back as itself in
     * any process. Zero entries mean "no remap" (section absent, or the id
     * was not declared) and the id passes through unchanged. */
    int label_remap[PULSEQ_BIN_MAX_LABEL_REMAP];
} bin_reader;

static int host_is_big_endian(void)
{
    /* clang-format off */
    union { unsigned long v; unsigned char b[4]; } probe;
    /* clang-format on */
    probe.v = 1UL;
    return probe.b[0] == 0;
}

/* Integers are assembled from bytes, so these are correct on either host
 * without knowing anything about it. */
static unsigned long dec_u32(const unsigned char *p, int big)
{
    if (big)
        return ((unsigned long)p[3]) | ((unsigned long)p[2] << 8) | ((unsigned long)p[1] << 16) |
            ((unsigned long)p[0] << 24);
    return ((unsigned long)p[0]) | ((unsigned long)p[1] << 8) | ((unsigned long)p[2] << 16) |
        ((unsigned long)p[3] << 24);
}

static int dec_i32(const unsigned char *p, int big)
{
    unsigned long u = dec_u32(p, big) & 0xFFFFFFFFUL;
    /* Two's complement, without assuming int is 32 bits or that the
     * implementation-defined unsigned->signed conversion does the right
     * thing at the top of the range. */
    if (u >= 0x80000000UL)
        return -(int)(0xFFFFFFFFUL - u + 1UL);
    return (int)u;
}

static void dec_i64(const unsigned char *p, int big, bin_i64 *out)
{
    if (big)
    {
        out->hi = dec_u32(p, 1) & 0xFFFFFFFFUL;
        out->lo = dec_u32(p + 4, 1) & 0xFFFFFFFFUL;
    }
    else
    {
        out->lo = dec_u32(p, 0) & 0xFFFFFFFFUL;
        out->hi = dec_u32(p + 4, 0) & 0xFFFFFFFFUL;
    }
}

/** @brief int64 -> double.  Exact for |v| < 2^53, which covers every
 *  picosecond time and every count this format can carry. */
static double i64_to_double(bin_i64 v)
{
    if ((v.hi & 0x80000000UL) != 0)
    {
        /* Negative: negate the 64-bit two's complement value. */
        unsigned long lo = (~v.lo) & 0xFFFFFFFFUL;
        unsigned long hi = (~v.hi) & 0xFFFFFFFFUL;
        lo = (lo + 1UL) & 0xFFFFFFFFUL;
        if (lo == 0UL)
            hi = (hi + 1UL) & 0xFFFFFFFFUL;
        return -((double)hi * 4294967296.0 + (double)lo);
    }
    return (double)v.hi * 4294967296.0 + (double)v.lo;
}

/* Reals are memcpy'd, so they do depend on host order; `reverse` says the two
 * disagree.  IEEE-754 layout is assumed, as it is everywhere else here. */
static double dec_f64(const unsigned char *p, int reverse)
{
    unsigned char tmp[8];
    double out;
    int i;

    if (reverse)
    {
        for (i = 0; i < 8; ++i)
            tmp[i] = p[7 - i];
        memcpy(&out, tmp, 8);
    }
    else
    {
        memcpy(&out, p, 8);
    }
    return out;
}

/* `float` here is the *wire* type, not the storage type: this field is four
 * bytes in the file whichever precision PULSEQ_REAL happens to be, so it is
 * decoded at its own width and widened on return. */
static PULSEQ_REAL dec_f32(const unsigned char *p, int reverse)
{
    unsigned char tmp[4];
    float out;
    int i;

    if (reverse)
    {
        for (i = 0; i < 4; ++i)
            tmp[i] = p[3 - i];
        memcpy(&out, tmp, 4);
    }
    else
    {
        memcpy(&out, p, 4);
    }
    return out;
}

/* ------------------------------------------------------------------ */
/*  Stream helpers.  All return PULSEQ_SUCCESS or a negative code.     */
/* ------------------------------------------------------------------ */

static int rd_bytes(bin_reader *r, unsigned char *dst, size_t n)
{
    if (n == 0)
        return PULSEQ_SUCCESS;
    if (fread(dst, 1, n, r->f) != n)
        return PULSEQ_ERR_FILE_READ_FAILED;
    return PULSEQ_SUCCESS;
}

static int rd_i32(bin_reader *r, int *out)
{
    unsigned char b[4];
    int rc = rd_bytes(r, b, 4);
    if (PULSEQ_FAILED(rc))
        return rc;
    *out = dec_i32(b, r->file_is_big);
    return PULSEQ_SUCCESS;
}

static int rd_i64(bin_reader *r, bin_i64 *out)
{
    unsigned char b[8];
    int rc = rd_bytes(r, b, 8);
    if (PULSEQ_FAILED(rc))
        return rc;
    dec_i64(b, r->file_is_big, out);
    return PULSEQ_SUCCESS;
}

/** @brief An int64 that is meant to be a count / small integer. */
static int rd_count(bin_reader *r, long *out)
{
    bin_i64 raw;
    double v;
    int rc = rd_i64(r, &raw);
    if (PULSEQ_FAILED(rc))
        return rc;
    v = i64_to_double(raw);
    if (v < 0.0 || v > (double)PULSEQ_BIN_MAX_COUNT)
        return PULSEQ_ERR_FILE_READ_FAILED;
    *out = (long)v;
    return PULSEQ_SUCCESS;
}

static int rd_f64(bin_reader *r, double *out)
{
    unsigned char b[8];
    int rc = rd_bytes(r, b, 8);
    if (PULSEQ_FAILED(rc))
        return rc;
    *out = dec_f64(b, r->reverse_real);
    return PULSEQ_SUCCESS;
}

/** @brief Read an int64 picosecond time as microseconds. */
static int rd_ps_as_us(bin_reader *r, PULSEQ_REAL *out)
{
    bin_i64 raw;
    int rc = rd_i64(r, &raw);
    if (PULSEQ_FAILED(rc))
        return rc;
    *out = (PULSEQ_REAL)(i64_to_double(raw) * 1e-6);
    return PULSEQ_SUCCESS;
}

/** @brief Read an int64 picosecond time as nanoseconds (ADC dwell). */
static int rd_ps_as_ns(bin_reader *r, PULSEQ_REAL *out)
{
    bin_i64 raw;
    int rc = rd_i64(r, &raw);
    if (PULSEQ_FAILED(rc))
        return rc;
    *out = (PULSEQ_REAL)(i64_to_double(raw) * 1e-3);
    return PULSEQ_SUCCESS;
}

/* ================================================================== */
/*  Row-indexed library staging                                       */
/* ================================================================== */
/*
 * Library rows are addressed by id-1, and the library's *size* is the largest
 * id present -- exactly what the text reader's init_standard_library computes.
 * The binary format gives an entry count rather than a maximum id, and the two
 * differ whenever ids are sparse, so rows are scattered into an array that can
 * grow.  Gradients need that anyway: [GRADIENTS] and [TRAP] are separate
 * sections sharing one library.
 *
 * Capacity is tracked separately from size; see lib_reserve for why.
 */
static int lib_reserve(PULSEQ_REAL **rows, int *cap, int *size, int cols, int needed)
{
    PULSEQ_REAL *grown;
    int new_cap, i;

    /* A caller starting from cap 0 on a library that already has rows is a
     * second section writing into the same array -- [GRADIENTS] then [TRAP].
     * Adopting the existing size as the capacity keeps that safe: worst case
     * it reallocates once more than it had to, where assuming cap 0 would
     * copy nothing and silently drop the first section's rows. */
    if (*cap == 0 && *rows != NULL)
        *cap = *size;

    if (needed > *size)
        *size = needed;
    if (needed <= *cap)
        return PULSEQ_SUCCESS;

    /* Geometric, and never "exactly what this id needs". The two per-scan
     * sections have one row per id, so growing to fit each id in turn copies
     * the whole array every row -- quadratic, and on a two-million-block scan
     * it does not finish. Each section pre-reserves its entry count, so for
     * dense ids this is one allocation and the growth path never runs. */
    new_cap = (*cap > 0) ? *cap * 2 : 64;
    if (new_cap < needed)
        new_cap = needed;
    if (new_cap <= 0 || (size_t)new_cap > (size_t)-1 / ((size_t)cols * sizeof(PULSEQ_REAL)))
        return PULSEQ_ERR_ALLOC_FAILED;

    grown = (PULSEQ_REAL *)PULSEQ_ALLOC((size_t)new_cap * (size_t)cols * sizeof(PULSEQ_REAL));
    if (!grown)
        return PULSEQ_ERR_ALLOC_FAILED;
    for (i = 0; i < new_cap * cols; ++i)
        grown[i] = 0.0f;
    if (*rows)
    {
        memcpy(grown, *rows, (size_t)(*cap) * (size_t)cols * sizeof(PULSEQ_REAL));
        PULSEQ_FREE(*rows);
    }
    *rows = grown;
    *cap = new_cap;
    return PULSEQ_SUCCESS;
}

/* ================================================================== */
/*  Definitions                                                       */
/* ================================================================== */

/*
 * The generic definitions library holds *strings*, because that is what the
 * text reader produces and what pulseq__read_definitions() and pulseg's deep
 * copy both consume -- and because those strings are stored verbatim in the
 * .pge cache.  Numeric values are therefore formatted back out here.
 *
 * "%.9g" is deliberate, and it is the writer's precision rather than the
 * double's: it is exactly what fastseq's text writer emits for a numeric
 * definition, so the same sequence yields the same definition strings (and
 * so the same cache bytes) whichever format it was stored in.  Nothing is
 * lost by it -- every consumer of these strings runs them through atof() into
 * a PULSEQ_REAL, which holds at least the seven significant digits a float
 * does and, in the double build, more than the nine printed here.
 */
#define PULSEQ_BIN_DEFINITION_FORMAT "%.9g"

static char *dup_token(const char *s, int len)
{
    char *out = (char *)PULSEQ_ALLOC((size_t)len + 1);
    if (!out)
        return NULL;
    memcpy(out, s, (size_t)len);
    out[len] = '\0';
    return out;
}

/** @brief Append one already-owned string to a definition's value list. */
static int def_push(pulseq_definition *def, char *owned)
{
    char **grown;
    int i;

    if (!owned)
        return PULSEQ_ERR_ALLOC_FAILED;
    grown = (char **)PULSEQ_ALLOC(sizeof(char *) * (size_t)(def->value_size + 1));
    if (!grown)
    {
        PULSEQ_FREE(owned);
        return PULSEQ_ERR_ALLOC_FAILED;
    }
    for (i = 0; i < def->value_size; ++i)
        grown[i] = def->value[i];
    grown[def->value_size] = owned;
    if (def->value)
        PULSEQ_FREE(def->value);
    def->value = grown;
    def->value_size++;
    return PULSEQ_SUCCESS;
}

static int read_binary_definitions(bin_reader *r, pulseq_file *seq)
{
    long count, i;
    int rc, j;
    pulseq_definition *defs;

    rc = rd_count(r, &count);
    if (PULSEQ_FAILED(rc))
        return rc;
    if (count == 0)
    {
        seq->is_definitions_library_parsed = 1;
        return PULSEQ_SUCCESS;
    }

    defs = (pulseq_definition *)PULSEQ_ALLOC((size_t)count * sizeof(pulseq_definition));
    if (!defs)
        return PULSEQ_ERR_ALLOC_FAILED;
    for (i = 0; i < count; ++i)
    {
        defs[i].name[0] = '\0';
        defs[i].value_size = 0;
        defs[i].value = NULL;
    }
    /* Published before the loop so a mid-way failure still frees what was
     * built: seq_file_reset walks num_definitions entries. */
    seq->definitions_library = defs;
    seq->num_definitions = (int)count;
    seq->is_definitions_library_parsed = 1;

    for (i = 0; i < count; ++i)
    {
        unsigned char b;
        int name_len = 0;
        int n_values;
        char type;
        char scratch[64];

        /* key: null-terminated */
        for (;;)
        {
            rc = rd_bytes(r, &b, 1);
            if (PULSEQ_FAILED(rc))
                return rc;
            if (b == 0)
                break;
            if (name_len < PULSEQ_DEFINITION_NAME_LENGTH - 1)
                defs[i].name[name_len++] = (char)b;
            else if (name_len > 4096)
                return PULSEQ_ERR_FILE_READ_FAILED;
            else
                name_len++;
        }
        if (name_len > PULSEQ_DEFINITION_NAME_LENGTH - 1)
            name_len = PULSEQ_DEFINITION_NAME_LENGTH - 1;
        defs[i].name[name_len] = '\0';

        rc = rd_bytes(r, &b, 1);
        if (PULSEQ_FAILED(rc))
            return rc;
        n_values = (int)b;
        rc = rd_bytes(r, &b, 1);
        if (PULSEQ_FAILED(rc))
            return rc;
        type = (char)b;

        if (type == 'f')
        {
            for (j = 0; j < n_values; ++j)
            {
                double v;
                rc = rd_f64(r, &v);
                if (PULSEQ_FAILED(rc))
                    return rc;
                sprintf(scratch, PULSEQ_BIN_DEFINITION_FORMAT, v);
                rc = def_push(&defs[i], dup_token(scratch, (int)strlen(scratch)));
                if (PULSEQ_FAILED(rc))
                    return rc;
            }
        }
        else if (type == 'i')
        {
            for (j = 0; j < n_values; ++j)
            {
                int v;
                rc = rd_i32(r, &v);
                if (PULSEQ_FAILED(rc))
                    return rc;
                sprintf(scratch, "%d", v);
                rc = def_push(&defs[i], dup_token(scratch, (int)strlen(scratch)));
                if (PULSEQ_FAILED(rc))
                    return rc;
            }
        }
        else if (type == 'c')
        {
            /* One blob on disk, but the text reader splits a definition line
             * on whitespace -- so "ROTATIONS LABELS" must arrive as two
             * values here too, or a consumer counting them sees a different
             * definition depending on which format it was handed. */
            char *blob;
            int start, pos;

            blob = (char *)PULSEQ_ALLOC((size_t)n_values + 1);
            if (!blob)
                return PULSEQ_ERR_ALLOC_FAILED;
            rc = rd_bytes(r, (unsigned char *)blob, (size_t)n_values);
            if (PULSEQ_FAILED(rc))
            {
                PULSEQ_FREE(blob);
                return rc;
            }
            blob[n_values] = '\0';
            /* writeBinary stopped null-terminating char values, but files
             * from the older convention still carry the terminator. */
            pos = n_values;
            while (pos > 0 && blob[pos - 1] == '\0')
                pos--;

            start = 0;
            for (j = 0; j <= pos; ++j)
            {
                int at_end = (j == pos);
                char c = at_end ? ' ' : blob[j];
                if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\0')
                {
                    if (j > start)
                    {
                        rc = def_push(&defs[i], dup_token(blob + start, j - start));
                        if (PULSEQ_FAILED(rc))
                        {
                            PULSEQ_FREE(blob);
                            return rc;
                        }
                    }
                    start = j + 1;
                }
            }
            PULSEQ_FREE(blob);
        }
        else
        {
            return PULSEQ_ERR_FILE_READ_FAILED;
        }
    }
    return PULSEQ_SUCCESS;
}

/* ================================================================== */
/*  Blocks                                                            */
/* ================================================================== */

static int read_binary_blocks(bin_reader *r, pulseq_file *seq)
{
    unsigned char batch[PULSEQ_BIN_BLOCK_STRIDE * PULSEQ_BIN_BLOCK_BATCH];
    long count, done;
    int rc, big = r->file_is_big;
    PULSEQ_REAL(*rows)[7];

    rc = rd_count(r, &count);
    if (PULSEQ_FAILED(rc))
        return rc;

    seq->is_block_library_parsed = 1;
    if (count == 0)
        return PULSEQ_SUCCESS;

    rows = (PULSEQ_REAL(*)[7])PULSEQ_ALLOC((size_t)count * 7 * sizeof(PULSEQ_REAL));
    if (!rows)
        return PULSEQ_ERR_ALLOC_FAILED;
    seq->block_library = rows;
    seq->num_blocks = (int)count;

    done = 0;
    while (done < count)
    {
        long n = count - done;
        long k;
        if (n > PULSEQ_BIN_BLOCK_BATCH)
            n = PULSEQ_BIN_BLOCK_BATCH;
        rc = rd_bytes(r, batch, (size_t)n * PULSEQ_BIN_BLOCK_STRIDE);
        if (PULSEQ_FAILED(rc))
            return rc;
        for (k = 0; k < n; ++k)
        {
            const unsigned char *p = batch + k * PULSEQ_BIN_BLOCK_STRIDE;
            PULSEQ_REAL *row = rows[done + k];
            bin_i64 dur;
            dec_i64(p, big, &dur);
            row[0] = (PULSEQ_REAL)i64_to_double(dur);
            row[1] = (PULSEQ_REAL)dec_i32(p + 8, big);
            row[2] = (PULSEQ_REAL)dec_i32(p + 12, big);
            row[3] = (PULSEQ_REAL)dec_i32(p + 16, big);
            row[4] = (PULSEQ_REAL)dec_i32(p + 20, big);
            row[5] = (PULSEQ_REAL)dec_i32(p + 24, big);
            row[6] = (PULSEQ_REAL)dec_i32(p + 28, big);
        }
        done += n;
    }
    return PULSEQ_SUCCESS;
}

/* ================================================================== */
/*  Event libraries                                                   */
/* ================================================================== */

/** @brief Grow the RF use-tag array in step with the RF library.
 *
 * pulseq_file has no size field for it -- the text reader allocates it once,
 * knowing the library size up front -- so the running size is a local here.
 * The whole [RF] section is read in one call, so nothing else can observe a
 * half-grown array. */
static int tags_reserve(int **tags, int *have, int needed)
{
    int *grown;
    int i;

    if (needed <= *have)
        return PULSEQ_SUCCESS;
    grown = (int *)PULSEQ_ALLOC((size_t)needed * sizeof(int));
    if (!grown)
        return PULSEQ_ERR_ALLOC_FAILED;
    for (i = 0; i < needed; ++i)
        grown[i] = PULSEQ_RF_USE_UNKNOWN;
    if (*tags)
    {
        memcpy(grown, *tags, (size_t)(*have) * sizeof(int));
        PULSEQ_FREE(*tags);
    }
    *tags = grown;
    *have = needed;
    return PULSEQ_SUCCESS;
}

static int read_binary_rf(bin_reader *r, pulseq_file *seq)
{
    long count, i;
    int rc;
    int cap = 0;
    int tags_size = 0;

    rc = rd_count(r, &count);
    if (PULSEQ_FAILED(rc))
        return rc;
    seq->is_rf_library_parsed = 1;

    /* Dense ids are the norm, so one allocation covers the section. */
    rc = lib_reserve((PULSEQ_REAL **)&seq->rf_library, &cap, &seq->rf_library_size, 10, (int)count);
    if (PULSEQ_FAILED(rc))
        return rc;

    for (i = 0; i < count; ++i)
    {
        int id, mag_id, phase_id, time_id;
        double amp, fppm, pppm, freq, phase;
        PULSEQ_REAL center_us, delay_us;
        unsigned char use;
        PULSEQ_REAL *row;

        rc = rd_i32(r, &id);
        if (PULSEQ_FAILED(rc))
            return rc;
        if (id <= 0)
            return PULSEQ_ERR_FILE_READ_FAILED;
        rc = rd_f64(r, &amp);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_i32(r, &mag_id);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_i32(r, &phase_id);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_i32(r, &time_id);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_ps_as_us(r, &center_us);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_ps_as_us(r, &delay_us);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_f64(r, &fppm);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_f64(r, &pppm);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_f64(r, &freq);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_f64(r, &phase);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_bytes(r, &use, 1);
        if (PULSEQ_FAILED(rc))
            return rc;

        rc = lib_reserve((PULSEQ_REAL **)&seq->rf_library, &cap, &seq->rf_library_size, 10, id);
        if (PULSEQ_FAILED(rc))
            return rc;
        row = &((PULSEQ_REAL *)seq->rf_library)[(id - 1) * 10];
        row[0] = (PULSEQ_REAL)amp;
        row[1] = (PULSEQ_REAL)mag_id;
        row[2] = (PULSEQ_REAL)phase_id;
        row[3] = (PULSEQ_REAL)time_id;
        row[4] = center_us;
        row[5] = delay_us;
        row[6] = (PULSEQ_REAL)fppm;
        row[7] = (PULSEQ_REAL)pppm;
        row[8] = (PULSEQ_REAL)freq;
        row[9] = (PULSEQ_REAL)phase;

        rc = tags_reserve(&seq->rf_use_tags, &tags_size, seq->rf_library_size);
        if (PULSEQ_FAILED(rc))
            return rc;
        seq->rf_use_tags[id - 1] = (use == 'e') ? PULSEQ_RF_USE_EXCITATION
            : (use == 'r')                      ? PULSEQ_RF_USE_REFOCUSING
            : (use == 'i')                      ? PULSEQ_RF_USE_INVERSION
            : (use == 's')                      ? PULSEQ_RF_USE_SATURATION
            : (use == 'p')                      ? PULSEQ_RF_USE_PREPARATION
            : (use == 'o')                      ? PULSEQ_RF_USE_OTHER
                                                : PULSEQ_RF_USE_UNKNOWN;
    }
    /* A sparse [RF] section can leave the library longer than the last id
     * that was written; the tag array must still cover every row. */
    if (seq->rf_library_size > 0)
        return tags_reserve(&seq->rf_use_tags, &tags_size, seq->rf_library_size);
    return PULSEQ_SUCCESS;
}

static int read_binary_arb_grad(bin_reader *r, pulseq_file *seq)
{
    long count, i;
    int rc;
    int cap = 0;

    rc = rd_count(r, &count);
    if (PULSEQ_FAILED(rc))
        return rc;
    seq->is_grad_library_parsed = 1;

    /* Dense ids are the norm, so one allocation covers the section. */
    rc = lib_reserve((PULSEQ_REAL **)&seq->grad_library, &cap, &seq->grad_library_size, 7, (int)count);
    if (PULSEQ_FAILED(rc))
        return rc;

    for (i = 0; i < count; ++i)
    {
        int id, amp_shape, time_shape;
        double amp, first, last;
        PULSEQ_REAL delay_us;
        PULSEQ_REAL *row;

        rc = rd_i32(r, &id);
        if (PULSEQ_FAILED(rc))
            return rc;
        if (id <= 0)
            return PULSEQ_ERR_FILE_READ_FAILED;
        rc = rd_f64(r, &amp);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_f64(r, &first);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_f64(r, &last);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_i32(r, &amp_shape);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_i32(r, &time_shape);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_ps_as_us(r, &delay_us);
        if (PULSEQ_FAILED(rc))
            return rc;

        rc = lib_reserve((PULSEQ_REAL **)&seq->grad_library, &cap, &seq->grad_library_size, 7, id);
        if (PULSEQ_FAILED(rc))
            return rc;
        row = &((PULSEQ_REAL *)seq->grad_library)[(id - 1) * 7];
        row[0] = 1.0f; /* text reader's discriminator: 1 = arbitrary, 0 = trapezoid */
        row[1] = (PULSEQ_REAL)amp;
        row[2] = (PULSEQ_REAL)first;
        row[3] = (PULSEQ_REAL)last;
        row[4] = (PULSEQ_REAL)amp_shape;
        row[5] = (PULSEQ_REAL)time_shape;
        row[6] = delay_us;
    }
    return PULSEQ_SUCCESS;
}

static int read_binary_trap_grad(bin_reader *r, pulseq_file *seq)
{
    long count, i;
    int rc;
    int cap = 0;

    rc = rd_count(r, &count);
    if (PULSEQ_FAILED(rc))
        return rc;
    seq->is_grad_library_parsed = 1;

    /* Dense ids are the norm, so one allocation covers the section. */
    rc = lib_reserve((PULSEQ_REAL **)&seq->grad_library, &cap, &seq->grad_library_size, 7, (int)count);
    if (PULSEQ_FAILED(rc))
        return rc;

    for (i = 0; i < count; ++i)
    {
        int id;
        double amp;
        PULSEQ_REAL rise, flat, fall, delay;
        PULSEQ_REAL *row;

        rc = rd_i32(r, &id);
        if (PULSEQ_FAILED(rc))
            return rc;
        if (id <= 0)
            return PULSEQ_ERR_FILE_READ_FAILED;
        rc = rd_f64(r, &amp);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_ps_as_us(r, &rise);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_ps_as_us(r, &flat);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_ps_as_us(r, &fall);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_ps_as_us(r, &delay);
        if (PULSEQ_FAILED(rc))
            return rc;

        rc = lib_reserve((PULSEQ_REAL **)&seq->grad_library, &cap, &seq->grad_library_size, 7, id);
        if (PULSEQ_FAILED(rc))
            return rc;
        row = &((PULSEQ_REAL *)seq->grad_library)[(id - 1) * 7];
        row[0] = 0.0f; /* text reader's flag: 0 = trapezoid */
        row[1] = (PULSEQ_REAL)amp;
        row[2] = rise;
        row[3] = flat;
        row[4] = fall;
        row[5] = delay;
        row[6] = 0.0f;
    }
    return PULSEQ_SUCCESS;
}

static int read_binary_adc(bin_reader *r, pulseq_file *seq)
{
    long count, i;
    int rc;
    int cap = 0;

    rc = rd_count(r, &count);
    if (PULSEQ_FAILED(rc))
        return rc;
    seq->is_adc_library_parsed = 1;

    /* Dense ids are the norm, so one allocation covers the section. */
    rc = lib_reserve((PULSEQ_REAL **)&seq->adc_library, &cap, &seq->adc_library_size, 8, (int)count);
    if (PULSEQ_FAILED(rc))
        return rc;

    for (i = 0; i < count; ++i)
    {
        int id, phase_id;
        bin_i64 num;
        double fppm, pppm, freq, phase;
        PULSEQ_REAL dwell_ns, delay_us;
        PULSEQ_REAL *row;

        rc = rd_i32(r, &id);
        if (PULSEQ_FAILED(rc))
            return rc;
        if (id <= 0)
            return PULSEQ_ERR_FILE_READ_FAILED;
        rc = rd_i64(r, &num);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_ps_as_ns(r, &dwell_ns);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_ps_as_us(r, &delay_us);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_f64(r, &fppm);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_f64(r, &pppm);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_f64(r, &freq);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_f64(r, &phase);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_i32(r, &phase_id);
        if (PULSEQ_FAILED(rc))
            return rc;

        rc = lib_reserve((PULSEQ_REAL **)&seq->adc_library, &cap, &seq->adc_library_size, 8, id);
        if (PULSEQ_FAILED(rc))
            return rc;
        row = &((PULSEQ_REAL *)seq->adc_library)[(id - 1) * 8];
        row[0] = (PULSEQ_REAL)i64_to_double(num);
        row[1] = dwell_ns;
        row[2] = delay_us;
        row[3] = (PULSEQ_REAL)fppm;
        row[4] = (PULSEQ_REAL)pppm;
        row[5] = (PULSEQ_REAL)freq;
        row[6] = (PULSEQ_REAL)phase;
        row[7] = (PULSEQ_REAL)phase_id;
    }
    return PULSEQ_SUCCESS;
}

/* Same capacity-vs-size split as lib_reserve, for the two libraries whose
 * rows are structs rather than numeric tuples. */
static int shapes_reserve(pulseq_file *seq, int *cap, int needed)
{
    pulseq_shape *grown;
    int new_cap, k;

    if (*cap == 0 && seq->shapes_library != NULL)
        *cap = seq->shapes_library_size;
    if (needed > seq->shapes_library_size)
        seq->shapes_library_size = needed;
    if (needed <= *cap)
        return PULSEQ_SUCCESS;

    new_cap = (*cap > 0) ? *cap * 2 : 64;
    if (new_cap < needed)
        new_cap = needed;
    if (new_cap <= 0 || (size_t)new_cap > (size_t)-1 / sizeof(pulseq_shape))
        return PULSEQ_ERR_ALLOC_FAILED;

    grown = (pulseq_shape *)PULSEQ_ALLOC((size_t)new_cap * sizeof(pulseq_shape));
    if (!grown)
        return PULSEQ_ERR_ALLOC_FAILED;
    for (k = 0; k < new_cap; ++k)
    {
        grown[k].num_samples = 0;
        grown[k].num_uncompressed_samples = 0;
        grown[k].samples = NULL;
    }
    if (seq->shapes_library)
    {
        memcpy(grown, seq->shapes_library, (size_t)(*cap) * sizeof(pulseq_shape));
        PULSEQ_FREE(seq->shapes_library);
    }
    seq->shapes_library = grown;
    *cap = new_cap;
    return PULSEQ_SUCCESS;
}

static int shims_reserve(pulseq_file *seq, int *cap, int needed)
{
    pulseq_rf_shim_entry *grown;
    int new_cap;

    if (*cap == 0 && seq->rf_shim_library != NULL)
        *cap = seq->rf_shim_library_size;
    if (needed > seq->rf_shim_library_size)
        seq->rf_shim_library_size = needed;
    if (needed <= *cap)
        return PULSEQ_SUCCESS;

    new_cap = (*cap > 0) ? *cap * 2 : 64;
    if (new_cap < needed)
        new_cap = needed;
    if (new_cap <= 0 || (size_t)new_cap > (size_t)-1 / sizeof(pulseq_rf_shim_entry))
        return PULSEQ_ERR_ALLOC_FAILED;

    grown = (pulseq_rf_shim_entry *)PULSEQ_ALLOC((size_t)new_cap * sizeof(pulseq_rf_shim_entry));
    if (!grown)
        return PULSEQ_ERR_ALLOC_FAILED;
    memset(grown, 0, (size_t)new_cap * sizeof(pulseq_rf_shim_entry));
    if (seq->rf_shim_library)
    {
        memcpy(grown, seq->rf_shim_library, (size_t)(*cap) * sizeof(pulseq_rf_shim_entry));
        PULSEQ_FREE(seq->rf_shim_library);
    }
    seq->rf_shim_library = grown;
    *cap = new_cap;
    return PULSEQ_SUCCESS;
}

static int read_binary_shapes(bin_reader *r, pulseq_file *seq)
{
    long count, i;
    int rc;
    int cap = 0;

    rc = rd_count(r, &count);
    if (PULSEQ_FAILED(rc))
        return rc;
    seq->is_shapes_library_parsed = 1;

    for (i = 0; i < count; ++i)
    {
        int id;
        long n_uncompressed, n_compressed, j;
        pulseq_shape *slot;
        PULSEQ_REAL *samples = NULL;

        rc = rd_i32(r, &id);
        if (PULSEQ_FAILED(rc))
            return rc;
        if (id <= 0)
            return PULSEQ_ERR_FILE_READ_FAILED;
        rc = rd_count(r, &n_uncompressed);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_count(r, &n_compressed);
        if (PULSEQ_FAILED(rc))
            return rc;

        rc = shapes_reserve(seq, &cap, id);
        if (PULSEQ_FAILED(rc))
            return rc;

        if (n_compressed > 0)
        {
            unsigned char buf[4];
            samples = (PULSEQ_REAL *)PULSEQ_ALLOC((size_t)n_compressed * sizeof(PULSEQ_REAL));
            if (!samples)
                return PULSEQ_ERR_ALLOC_FAILED;
            for (j = 0; j < n_compressed; ++j)
            {
                rc = rd_bytes(r, buf, 4);
                if (PULSEQ_FAILED(rc))
                {
                    PULSEQ_FREE(samples);
                    return rc;
                }
                samples[j] = dec_f32(buf, r->reverse_real);
            }
        }
        slot = &seq->shapes_library[id - 1];
        if (slot->samples)
            PULSEQ_FREE(slot->samples);
        slot->num_uncompressed_samples = (int)n_uncompressed;
        slot->num_samples = (int)n_compressed;
        slot->samples = samples;
    }
    return PULSEQ_SUCCESS;
}

static int read_binary_extension_list(bin_reader *r, pulseq_file *seq)
{
    unsigned char batch[PULSEQ_BIN_EXT_STRIDE * PULSEQ_BIN_EXT_BATCH];
    long count, done;
    int rc, big = r->file_is_big;
    int cap = 0;

    rc = rd_count(r, &count);
    if (PULSEQ_FAILED(rc))
        return rc;

    /* Dense ids are the norm, so one allocation covers the section. */
    rc = lib_reserve(
        (PULSEQ_REAL **)&seq->extensions_library,
        &cap,
        &seq->extensions_library_size,
        3,
        (int)count);
    if (PULSEQ_FAILED(rc))
        return rc;

    /* The other section whose length is the scan rather than the number of
     * distinct events -- a 3D scan has one chain per phase-encode pair -- so
     * it is read in batches for the same reason [BLOCKS] is. */
    done = 0;
    while (done < count)
    {
        long n = count - done;
        long k;
        if (n > PULSEQ_BIN_EXT_BATCH)
            n = PULSEQ_BIN_EXT_BATCH;
        rc = rd_bytes(r, batch, (size_t)n * PULSEQ_BIN_EXT_STRIDE);
        if (PULSEQ_FAILED(rc))
            return rc;
        for (k = 0; k < n; ++k)
        {
            const unsigned char *p = batch + k * PULSEQ_BIN_EXT_STRIDE;
            int id = dec_i32(p, big);
            PULSEQ_REAL *row;

            if (id <= 0)
                return PULSEQ_ERR_FILE_READ_FAILED;
            rc = lib_reserve(
                (PULSEQ_REAL **)&seq->extensions_library,
                &cap,
                &seq->extensions_library_size,
                3,
                id);
            if (PULSEQ_FAILED(rc))
                return rc;
            row = &((PULSEQ_REAL *)seq->extensions_library)[(id - 1) * 3];
            row[0] = (PULSEQ_REAL)dec_i32(p + 4, big);
            row[1] = (PULSEQ_REAL)dec_i32(p + 8, big);
            row[2] = (PULSEQ_REAL)dec_i32(p + 12, big);
        }
        done += n;
    }
    return PULSEQ_SUCCESS;
}

static int read_binary_triggers(bin_reader *r, pulseq_file *seq)
{
    long count, i;
    int rc, ext_id;
    int cap = 0;

    rc = rd_i32(r, &ext_id);
    if (PULSEQ_FAILED(rc))
        return rc;
    seq->extension_map[PULSEQ_EXT_TRIGGER] = ext_id;
    rc = rd_count(r, &count);
    if (PULSEQ_FAILED(rc))
        return rc;

    /* Dense ids are the norm, so one allocation covers the section. */
    rc = lib_reserve(
        (PULSEQ_REAL **)&seq->trigger_library,
        &cap,
        &seq->trigger_library_size,
        4,
        (int)count);
    if (PULSEQ_FAILED(rc))
        return rc;

    for (i = 0; i < count; ++i)
    {
        int id, type, channel;
        PULSEQ_REAL delay_us, duration_us, *row;

        rc = rd_i32(r, &id);
        if (PULSEQ_FAILED(rc))
            return rc;
        if (id <= 0)
            return PULSEQ_ERR_FILE_READ_FAILED;
        rc = rd_i32(r, &type);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_i32(r, &channel);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_ps_as_us(r, &delay_us);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_ps_as_us(r, &duration_us);
        if (PULSEQ_FAILED(rc))
            return rc;

        rc = lib_reserve((PULSEQ_REAL **)&seq->trigger_library, &cap, &seq->trigger_library_size, 4, id);
        if (PULSEQ_FAILED(rc))
            return rc;
        row = &((PULSEQ_REAL *)seq->trigger_library)[(id - 1) * 4];
        row[0] = (PULSEQ_REAL)type;
        row[1] = (PULSEQ_REAL)channel;
        row[2] = delay_us;
        row[3] = duration_us;
    }
    return PULSEQ_SUCCESS;
}

/** @brief LABELNAMES: the file's own label-id space, defined by name.
 *
 * Each entry pairs a file label id with its name; registering the name
 * gives this process's id for it, and the difference becomes the remap the
 * label rows are read through. Ids past the remap's bound, and files
 * without the section, pass through unchanged. */
static int read_binary_label_names(bin_reader *r, pulseq_file *seq)
{
    long count, i;
    int rc;

    (void)seq;
    rc = rd_count(r, &count);
    if (PULSEQ_FAILED(rc))
        return rc;

    for (i = 0; i < count; ++i)
    {
        char name[PULSEQ_LABEL_NAME_LENGTH];
        int file_id, registered;
        size_t at;
        int ch;

        rc = rd_i32(r, &file_id);
        if (PULSEQ_FAILED(rc))
            return rc;

        at = 0;
        for (;;)
        {
            ch = fgetc(r->f);
            if (ch == EOF)
                return PULSEQ_ERR_FILE_READ_FAILED;
            if (ch == '\0')
                break;
            if (at + 1 < sizeof(name))
                name[at++] = (char)ch;
        }
        name[at] = '\0';

        registered = pulseq_label_register_name(name);
        if (registered > 0 && file_id > 0 && file_id <= PULSEQ_BIN_MAX_LABEL_REMAP)
            r->label_remap[file_id - 1] = registered;
    }
    return PULSEQ_SUCCESS;
}

static int read_binary_labels(bin_reader *r, pulseq_file *seq, int is_inc)
{
    long count, i;
    int rc, ext_id;
    int cap = 0;
    PULSEQ_REAL **rows = is_inc ? (PULSEQ_REAL **)&seq->labelinc_library : (PULSEQ_REAL **)&seq->labelset_library;
    int *size = is_inc ? &seq->labelinc_library_size : &seq->labelset_library_size;

    rc = rd_i32(r, &ext_id);
    if (PULSEQ_FAILED(rc))
        return rc;
    seq->extension_map[is_inc ? PULSEQ_EXT_LABELINC : PULSEQ_EXT_LABELSET] = ext_id;
    rc = rd_count(r, &count);
    if (PULSEQ_FAILED(rc))
        return rc;

    /* Dense ids are the norm, so one allocation covers the section. */
    rc = lib_reserve(rows, &cap, size, 2, (int)count);
    if (PULSEQ_FAILED(rc))
        return rc;

    for (i = 0; i < count; ++i)
    {
        int id, value, label_index;
        PULSEQ_REAL *row;

        rc = rd_i32(r, &id);
        if (PULSEQ_FAILED(rc))
            return rc;
        if (id <= 0)
            return PULSEQ_ERR_FILE_READ_FAILED;
        rc = rd_i32(r, &value);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_i32(r, &label_index);
        if (PULSEQ_FAILED(rc))
            return rc;
        if (label_index > 0 && label_index <= PULSEQ_BIN_MAX_LABEL_REMAP &&
            r->label_remap[label_index - 1] > 0)
            label_index = r->label_remap[label_index - 1];

        rc = lib_reserve(rows, &cap, size, 2, id);
        if (PULSEQ_FAILED(rc))
            return rc;
        row = &(*rows)[(id - 1) * 2];
        row[0] = (PULSEQ_REAL)value;
        row[1] = (PULSEQ_REAL)label_index;
        if (label_index > 0 && label_index <= PULSEQ_LABEL_ID_MAX)
            seq->is_label_defined[label_index - 1] = 1;
    }
    return PULSEQ_SUCCESS;
}

static int read_binary_soft_delays(bin_reader *r, pulseq_file *seq)
{
    long count, i;
    int rc, ext_id;
    int cap = 0;

    rc = rd_i32(r, &ext_id);
    if (PULSEQ_FAILED(rc))
        return rc;
    seq->extension_map[PULSEQ_EXT_DELAY] = ext_id;
    rc = rd_count(r, &count);
    if (PULSEQ_FAILED(rc))
        return rc;

    /* Dense ids are the norm, so one allocation covers the section. */
    rc = lib_reserve(
        (PULSEQ_REAL **)&seq->soft_delay_library,
        &cap,
        &seq->soft_delay_library_size,
        4,
        (int)count);
    if (PULSEQ_FAILED(rc))
        return rc;

    for (i = 0; i < count; ++i)
    {
        int id, num, hint_len, hint_code;
        double factor;
        PULSEQ_REAL offset_us, *row;
        char hint[PULSEQ_SOFT_DELAY_HINT_LENGTH];

        rc = rd_i32(r, &id);
        if (PULSEQ_FAILED(rc))
            return rc;
        if (id <= 0)
            return PULSEQ_ERR_FILE_READ_FAILED;
        rc = rd_i32(r, &num);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_ps_as_us(r, &offset_us);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_f64(r, &factor);
        if (PULSEQ_FAILED(rc))
            return rc;
        rc = rd_i32(r, &hint_len);
        if (PULSEQ_FAILED(rc))
            return rc;
        if (hint_len < 0 || hint_len > 4096)
            return PULSEQ_ERR_FILE_READ_FAILED;
        {
            int keep = hint_len;
            int skip = 0;
            if (keep > PULSEQ_SOFT_DELAY_HINT_LENGTH - 1)
            {
                skip = keep - (PULSEQ_SOFT_DELAY_HINT_LENGTH - 1);
                keep = PULSEQ_SOFT_DELAY_HINT_LENGTH - 1;
            }
            rc = rd_bytes(r, (unsigned char *)hint, (size_t)keep);
            if (PULSEQ_FAILED(rc))
                return rc;
            hint[keep] = '\0';
            while (skip-- > 0)
            {
                unsigned char junk;
                rc = rd_bytes(r, &junk, 1);
                if (PULSEQ_FAILED(rc))
                    return rc;
            }
        }

        hint_code = pulseq_hint_id_for_name(hint);
        if (hint_code > 0 && hint_code <= 8)
            seq->is_delay_defined[hint_code - 1] = 1;

        rc = lib_reserve(
            (PULSEQ_REAL **)&seq->soft_delay_library,
            &cap,
            &seq->soft_delay_library_size,
            4,
            id);
        if (PULSEQ_FAILED(rc))
            return rc;
        row = &((PULSEQ_REAL *)seq->soft_delay_library)[(id - 1) * 4];
        row[0] = (PULSEQ_REAL)num;
        row[1] = offset_us;
        row[2] = (PULSEQ_REAL)factor;
        row[3] = (PULSEQ_REAL)hint_code;
    }
    return PULSEQ_SUCCESS;
}

static int read_binary_rf_shims(bin_reader *r, pulseq_file *seq)
{
    long count, i;
    int rc, ext_id;
    int cap = 0;

    rc = rd_i32(r, &ext_id);
    if (PULSEQ_FAILED(rc))
        return rc;
    seq->extension_map[PULSEQ_EXT_RF_SHIM] = ext_id;
    rc = rd_count(r, &count);
    if (PULSEQ_FAILED(rc))
        return rc;

    for (i = 0; i < count; ++i)
    {
        int id, num_chan, j;

        rc = rd_i32(r, &id);
        if (PULSEQ_FAILED(rc))
            return rc;
        if (id <= 0)
            return PULSEQ_ERR_FILE_READ_FAILED;
        rc = rd_i32(r, &num_chan);
        if (PULSEQ_FAILED(rc))
            return rc;
        if (num_chan <= 0 || num_chan > PULSEQ_MAX_RF_SHIM_CHANNELS)
            return PULSEQ_ERR_FILE_READ_FAILED;

        rc = shims_reserve(seq, &cap, id);
        if (PULSEQ_FAILED(rc))
            return rc;

        seq->rf_shim_library[id - 1].num_channels = num_chan;
        for (j = 0; j < 2 * num_chan; ++j)
        {
            double v;
            rc = rd_f64(r, &v);
            if (PULSEQ_FAILED(rc))
                return rc;
            seq->rf_shim_library[id - 1].values[j] = (PULSEQ_REAL)v;
        }
    }
    return PULSEQ_SUCCESS;
}

static int read_binary_rotations(bin_reader *r, pulseq_file *seq)
{
    long count, i;
    int rc, ext_id;
    int cap = 0;

    rc = rd_i32(r, &ext_id);
    if (PULSEQ_FAILED(rc))
        return rc;
    seq->extension_map[PULSEQ_EXT_ROTATION] = ext_id;
    rc = rd_count(r, &count);
    if (PULSEQ_FAILED(rc))
        return rc;

    /* Dense ids are the norm, so one allocation covers the section. */
    rc = lib_reserve(
        (PULSEQ_REAL **)&seq->rotation_quaternion_library,
        &cap,
        &seq->rotation_library_size,
        4,
        (int)count);
    if (PULSEQ_FAILED(rc))
        return rc;

    for (i = 0; i < count; ++i)
    {
        int id, j;
        double q[4];
        PULSEQ_REAL *row;

        rc = rd_i32(r, &id);
        if (PULSEQ_FAILED(rc))
            return rc;
        if (id <= 0)
            return PULSEQ_ERR_FILE_READ_FAILED;
        for (j = 0; j < 4; ++j)
        {
            rc = rd_f64(r, &q[j]);
            if (PULSEQ_FAILED(rc))
                return rc;
        }
        rc = lib_reserve(
            (PULSEQ_REAL **)&seq->rotation_quaternion_library,
            &cap,
            &seq->rotation_library_size,
            4,
            id);
        if (PULSEQ_FAILED(rc))
            return rc;
        row = &((PULSEQ_REAL *)seq->rotation_quaternion_library)[(id - 1) * 4];
        for (j = 0; j < 4; ++j)
            row[j] = (PULSEQ_REAL)q[j];
    }

    /* Quaternions are kept as written, matching the text reader -- which no
     * longer normalizes either.  See the note in read_extensions_library()
     * (pulseq_parse.c) for why that eager normalization went away. */
    return PULSEQ_SUCCESS;
}

static int skip_binary_legacy_delays(bin_reader *r)
{
    long count, i;
    int rc = rd_count(r, &count);
    if (PULSEQ_FAILED(rc))
        return rc;
    for (i = 0; i < count; ++i)
    {
        unsigned char junk[12];
        rc = rd_bytes(r, junk, 12);
        if (PULSEQ_FAILED(rc))
            return rc;
    }
    return PULSEQ_SUCCESS;
}

/* ================================================================== */
/*  Header                                                            */
/* ================================================================== */

static const unsigned char pulseq_bin_magic[8] = {0x01, 'p', 'u', 'l', 's', 'e', 'q', 0x02};

int pulseq_file_is_binary(const char *file_path)
{
    unsigned char header[8];
    FILE *f;
    size_t got;

    if (!file_path)
        return 0;
    f = fopen(file_path, "rb");
    if (!f)
        return 0;
    got = fread(header, 1, 8, f);
    fclose(f);
    if (got != 8)
        return 0;
    return memcmp(header, pulseq_bin_magic, 8) == 0 ? 1 : 0;
}

/*
 * Read the magic and the three version int64s, and decide byte order while
 * doing it.  The magic itself is byte-order-free (it is a byte string), so
 * the version fields are the probe: a real major version is a small positive
 * number, and reading a little-endian 1 as big-endian gives 2^56.
 */
static int read_binary_header(bin_reader *r, pulseq_file *seq)
{
    unsigned char header[8 + 24];
    bin_i64 v[3];
    int order, chosen = -1;

    if (PULSEQ_FAILED(rd_bytes(r, header, 8)))
        return PULSEQ_ERR_FILE_READ_FAILED;
    if (memcmp(header, pulseq_bin_magic, 8) != 0)
        return PULSEQ_ERR_INVALID_ARGUMENT;
    if (PULSEQ_FAILED(rd_bytes(r, header + 8, 24)))
        return PULSEQ_ERR_FILE_READ_FAILED;

    for (order = 0; order <= 1 && chosen < 0; ++order)
    {
        double major, minor, revision;
        dec_i64(header + 8, order, &v[0]);
        dec_i64(header + 16, order, &v[1]);
        dec_i64(header + 24, order, &v[2]);
        major = i64_to_double(v[0]);
        minor = i64_to_double(v[1]);
        revision = i64_to_double(v[2]);
        if (major >= 1.0 && major < 1000.0 && minor >= 0.0 && minor < 1000.0 && revision >= 0.0 &&
            revision < 1000.0)
            chosen = order;
    }
    if (chosen < 0)
        return PULSEQ_ERR_FILE_READ_FAILED;

    r->file_is_big = chosen;
    r->reverse_real = (chosen != host_is_big_endian());

    dec_i64(header + 8, chosen, &v[0]);
    dec_i64(header + 16, chosen, &v[1]);
    dec_i64(header + 24, chosen, &v[2]);
    seq->version_major = (int)i64_to_double(v[0]);
    seq->version_minor = (int)i64_to_double(v[1]);
    seq->version_revision = (int)i64_to_double(v[2]);
    seq->version_combined =
        seq->version_major * 1000000 + seq->version_minor * 1000 + seq->version_revision;
    seq->is_version_parsed = 1;
    return PULSEQ_SUCCESS;
}

/** @brief Read the next section code.  Returns 0 at clean end of file. */
static int next_section(bin_reader *r, int *code, int *at_eof)
{
    unsigned char b[8];
    bin_i64 raw;
    size_t got;

    *at_eof = 0;
    got = fread(b, 1, 8, r->f);
    if (got == 0)
    {
        *at_eof = 1;
        return PULSEQ_SUCCESS;
    }
    if (got != 8)
        return PULSEQ_ERR_FILE_READ_FAILED;
    dec_i64(b, r->file_is_big, &raw);
    if (raw.hi != PULSEQ_BIN_SECTION_HI || raw.lo == 0UL ||
        raw.lo > (unsigned long)PULSEQ_BIN_LABELNAMES)
        return PULSEQ_ERR_FILE_READ_FAILED;
    *code = (int)raw.lo;
    return PULSEQ_SUCCESS;
}

/* ================================================================== */
/*  Entry points                                                      */
/* ================================================================== */

static int read_binary_stream(pulseq_file *seq, FILE *f, int definitions_only)
{
    bin_reader r;
    int rc, code, at_eof, n;

    r.f = f;
    r.file_is_big = 0;
    r.reverse_real = 0;
    memset(r.label_remap, 0, sizeof(r.label_remap));

    pulseq__file_reset(seq);

    rc = read_binary_header(&r, seq);
    if (PULSEQ_FAILED(rc))
        return rc;
    if (seq->version_combined < 1005000)
        return PULSEQ_ERR_UNSUPPORTED_VERSION;

    for (;;)
    {
        rc = next_section(&r, &code, &at_eof);
        if (PULSEQ_FAILED(rc))
            return rc;
        if (at_eof)
            break;

        if (code == PULSEQ_BIN_DEFINITIONS)
        {
            rc = read_binary_definitions(&r, seq);
            if (PULSEQ_FAILED(rc))
                return rc;
            /* Everything a definitions-only caller wants -- TotalDuration,
             * the rasters, NextSequence -- is now in hand, and the section
             * is written first, so there is nothing left to walk. */
            if (definitions_only)
                break;
            continue;
        }
        if (definitions_only)
        {
            /* A file whose definitions come later than expected: stop rather
             * than decode megabytes the caller did not ask for.  Section
             * order is fixed by writeBinary, so this is a corrupt-file path. */
            break;
        }

        switch (code)
        {
        case PULSEQ_BIN_BLOCKS:
            rc = read_binary_blocks(&r, seq);
            break;
        case PULSEQ_BIN_RF:
            rc = read_binary_rf(&r, seq);
            break;
        case PULSEQ_BIN_GRADIENTS:
            rc = read_binary_arb_grad(&r, seq);
            break;
        case PULSEQ_BIN_TRAPEZOIDS:
            rc = read_binary_trap_grad(&r, seq);
            break;
        case PULSEQ_BIN_ADC:
            rc = read_binary_adc(&r, seq);
            break;
        case PULSEQ_BIN_DELAYS:
            rc = skip_binary_legacy_delays(&r);
            break;
        case PULSEQ_BIN_SHAPES:
            rc = read_binary_shapes(&r, seq);
            break;
        case PULSEQ_BIN_EXTENSIONS:
            rc = read_binary_extension_list(&r, seq);
            break;
        case PULSEQ_BIN_TRIGGERS:
            rc = read_binary_triggers(&r, seq);
            break;
        case PULSEQ_BIN_LABELNAMES:
            rc = read_binary_label_names(&r, seq);
            break;
        case PULSEQ_BIN_LABELSET:
            rc = read_binary_labels(&r, seq, 0);
            break;
        case PULSEQ_BIN_LABELINC:
            rc = read_binary_labels(&r, seq, 1);
            break;
        case PULSEQ_BIN_SOFTDELAYS:
            rc = read_binary_soft_delays(&r, seq);
            break;
        case PULSEQ_BIN_RFSHIMS:
            rc = read_binary_rf_shims(&r, seq);
            break;
        case PULSEQ_BIN_ROTATIONS:
            rc = read_binary_rotations(&r, seq);
            break;
        default:
            rc = PULSEQ_ERR_FILE_READ_FAILED;
            break;
        }
        if (PULSEQ_FAILED(rc))
            return rc;
    }

    pulseq__read_definitions(seq);

    if (!definitions_only)
    {
        /* The extension libraries are all present or all absent together as
         * far as the free path is concerned, so the flag is set once here
         * rather than by each section reader. */
        seq->is_extensions_library_parsed = 1;

        for (n = 0; n < 8; ++n)
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
            for (n = 0; n <= seq->extension_lut_size; ++n)
                seq->extension_lut[n] = PULSEQ_EXT_UNKNOWN;
            for (n = 0; n < 8; ++n)
            {
                if (seq->extension_map[n] > 0)
                    seq->extension_lut[seq->extension_map[n]] = n;
            }
        }
    }
    return PULSEQ_SUCCESS;
}

int pulseq_read_binary_from_buffer(pulseq_file *seq, FILE *f)
{
    int rc;

    if (!seq || !f)
        return PULSEQ_ERR_NULL_POINTER;
    rc = read_binary_stream(seq, f, 0);
    if (PULSEQ_FAILED(rc))
        pulseq__file_reset(seq);
    return rc;
}

int pulseq_read_binary(pulseq_file *seq, const char *file_path)
{
    FILE *f;
    int rc;

    if (!seq || !file_path)
        return PULSEQ_ERR_INVALID_ARGUMENT;
    f = fopen(file_path, "rb");
    if (!f)
        return PULSEQ_ERR_FILE_NOT_FOUND;
    rc = read_binary_stream(seq, f, 0);
    fclose(f);
    if (PULSEQ_FAILED(rc))
        pulseq__file_reset(seq);
    return rc;
}

int pulseq_read_binary_definitions_only(pulseq_file *seq, const char *file_path)
{
    FILE *f;
    int rc;

    if (!seq || !file_path)
        return PULSEQ_ERR_INVALID_ARGUMENT;
    f = fopen(file_path, "rb");
    if (!f)
        return PULSEQ_ERR_FILE_NOT_FOUND;
    rc = read_binary_stream(seq, f, 1);
    fclose(f);
    if (PULSEQ_FAILED(rc))
        pulseq__file_reset(seq);
    return rc;
}
