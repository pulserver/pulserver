/**
 * @file pulseg_recon.c
 * @brief Standalone reader for the recon-relevant cache sections.
 *
 * Reads DEFINITIONS, TRAJECTORY and SEQDESC straight from the cache file with
 * no loaded collection, for consumers that only have the .pseg/.pge on disk.
 * TRAJECTORY is not re-parsed here: it delegates to
 * pulseg_load_trajectory_cache() so the byte layout has exactly one reader.
 *
 * See pulseg_recon.h for the section-by-section contract.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pulseg_recon.h"

#define PGR_CACHE_ENDIAN_MARKER 0x01020304
#define PGR_SECTION_DEFINITIONS 0
#define PGR_SECTION_TRAJECTORY 6
#define PGR_SECTION_SEQDESC 7

/* ================================================================== */
/*  Byte-swap / typed I/O helpers (mirrors pulseg_trajectory.c /       */
/*  pulseg_cache_seqdesc.c -- duplicated locally per module convention */
/*  since these helpers are file-static everywhere else too).          */
/* ================================================================== */

static int pgr_read4(FILE *f, void *p, int count)
{
    return (int)fread(p, 4, (size_t)count, f) == count;
}

static void pgr_swap4(void *p)
{
    unsigned char *b = (unsigned char *)p;
    unsigned char t;
    t = b[0];
    b[0] = b[3];
    b[3] = t;
    t = b[1];
    b[1] = b[2];
    b[2] = t;
}

static void pgr_swap4_array(void *p, int count)
{
    int i;
    for (i = 0; i < count; ++i)
        pgr_swap4((unsigned char *)p + (size_t)i * 4);
}

static void pgr_set_diag(char *diag, int diag_size, const char *msg)
{
    size_t n;
    if (!diag || diag_size <= 0)
        return;
    n = strlen(msg);
    if (n >= (size_t)diag_size)
        n = (size_t)diag_size - 1;
    memcpy(diag, msg, n);
    diag[n] = '\0';
}

/* ================================================================== */
/*  Section table-of-contents                                          */
/* ================================================================== */

typedef struct pgr_section
{
    int id;
    int offset;
    int size;
} pgr_section;

/* Reads the file header (marker/version/vendor/stored_size/num_sections)
 * and the section table, locating sections 0/6/7 in a single pass.
 * Leaves the file position undefined on return (callers always seek
 * explicitly before reading section payloads). */
static int pgr_read_header_and_toc(
    FILE *f,
    int *do_swap_out,
    pgr_section *definitions_section,
    pgr_section *traj_section,
    pgr_section *seqdesc_section,
    char *diag,
    int diag_size)
{
    int marker;
    int version_major, version_minor, version_revision, vendor, stored_size;
    int num_sections, i, do_swap;

    definitions_section->id = -1;
    traj_section->id = -1;
    seqdesc_section->id = -1;

    if (!pgr_read4(f, &marker, 1))
    {
        pgr_set_diag(diag, diag_size, "unexpected EOF reading cache header marker");
        return 0;
    }
    do_swap = 0;
    if (marker != PGR_CACHE_ENDIAN_MARKER)
    {
        pgr_swap4(&marker);
        if (marker != PGR_CACHE_ENDIAN_MARKER)
        {
            pgr_set_diag(diag, diag_size, "invalid cache file: bad endian marker");
            return 0;
        }
        do_swap = 1;
    }

    if (!pgr_read4(f, &version_major, 1) || !pgr_read4(f, &version_minor, 1) ||
        !pgr_read4(f, &version_revision, 1) || !pgr_read4(f, &vendor, 1) ||
        !pgr_read4(f, &stored_size, 1) || !pgr_read4(f, &num_sections, 1))
    {
        pgr_set_diag(diag, diag_size, "unexpected EOF reading cache header");
        return 0;
    }
    if (do_swap)
        pgr_swap4(&num_sections);
    /* Only num_sections gates the TOC loop below; the other header fields
     * are not needed by this reader (matches trajectory_cache_reader.cpp,
     * which read and discarded them too). */

    if (num_sections <= 0 || num_sections > 64)
    {
        pgr_set_diag(diag, diag_size, "invalid cache: bad num_sections");
        return 0;
    }

    for (i = 0; i < num_sections; ++i)
    {
        int id, off, sz;
        if (!pgr_read4(f, &id, 1) || !pgr_read4(f, &off, 1) || !pgr_read4(f, &sz, 1))
        {
            pgr_set_diag(diag, diag_size, "unexpected EOF reading section table");
            return 0;
        }
        if (do_swap)
        {
            pgr_swap4(&id);
            pgr_swap4(&off);
            pgr_swap4(&sz);
        }
        if (id == PGR_SECTION_DEFINITIONS)
        {
            definitions_section->id = id;
            definitions_section->offset = off;
            definitions_section->size = sz;
        }
        if (id == PGR_SECTION_TRAJECTORY)
        {
            traj_section->id = id;
            traj_section->offset = off;
            traj_section->size = sz;
        }
        if (id == PGR_SECTION_SEQDESC)
        {
            seqdesc_section->id = id;
            seqdesc_section->offset = off;
            seqdesc_section->size = sz;
        }
    }

    *do_swap_out = do_swap;
    return 1;
}

/* ================================================================== */
/*  Section 0 (DEFINITIONS) -- per-subsequence generic [DEFINITIONS] kv */
/* ================================================================== */

/* Mirrors read_definitions_cache() in pulseg_cache.c, but standalone: the
 * subsequence count is read from the file itself rather than taken from a
 * pre-loaded pulseg_collection. */
static int pgr_read_one_definitions_block(
    pulseg_recon_definitions *out,
    FILE *f,
    int do_swap,
    char *diag,
    int diag_size)
{
    int i;

    out->num_definitions = 0;
    out->definitions = NULL;

    if (!pgr_read4(f, &out->num_definitions, 1))
    {
        pgr_set_diag(diag, diag_size, "EOF reading DEFINITIONS num_definitions");
        return 0;
    }
    if (do_swap)
        pgr_swap4(&out->num_definitions);
    if (out->num_definitions <= 0)
    {
        out->num_definitions = 0;
        return 1;
    }

    out->definitions =
        (pulseq_definition *)PULSEG_ALLOC((size_t)out->num_definitions * sizeof(pulseq_definition));
    if (!out->definitions)
    {
        pgr_set_diag(diag, diag_size, "allocation failed for DEFINITIONS entries");
        return 0;
    }
    memset(out->definitions, 0, (size_t)out->num_definitions * sizeof(pulseq_definition));

    for (i = 0; i < out->num_definitions; ++i)
    {
        int name_len, j;
        pulseq_definition *def = &out->definitions[i];

        if (!pgr_read4(f, &name_len, 1))
        {
            pgr_set_diag(diag, diag_size, "EOF reading DEFINITIONS name length");
            return 0;
        }
        if (do_swap)
            pgr_swap4(&name_len);
        if (name_len > 0 && name_len < PULSEQ_DEFINITION_NAME_LENGTH)
        {
            if (fread(def->name, 1, (size_t)name_len, f) != (size_t)name_len)
            {
                pgr_set_diag(diag, diag_size, "EOF reading DEFINITIONS name");
                return 0;
            }
            def->name[name_len] = '\0';
        }
        else if (name_len > 0)
        {
            /* Name too long for the fixed buffer -- skip it rather than
             * overflow (should not happen for well-formed caches). */
            if (fseek(f, name_len, SEEK_CUR) != 0)
            {
                pgr_set_diag(diag, diag_size, "seek failed skipping oversized DEFINITIONS name");
                return 0;
            }
        }

        if (!pgr_read4(f, &def->value_size, 1))
        {
            pgr_set_diag(diag, diag_size, "EOF reading DEFINITIONS value_size");
            return 0;
        }
        if (do_swap)
            pgr_swap4(&def->value_size);
        if (def->value_size > 0)
        {
            def->value = (char **)PULSEG_ALLOC((size_t)def->value_size * sizeof(char *));
            if (!def->value)
            {
                pgr_set_diag(diag, diag_size, "allocation failed for DEFINITIONS values");
                return 0;
            }
            memset(def->value, 0, (size_t)def->value_size * sizeof(char *));
            for (j = 0; j < def->value_size; ++j)
            {
                int vlen;
                if (!pgr_read4(f, &vlen, 1))
                {
                    pgr_set_diag(diag, diag_size, "EOF reading DEFINITIONS value length");
                    return 0;
                }
                if (do_swap)
                    pgr_swap4(&vlen);
                if (vlen > 0)
                {
                    def->value[j] = (char *)PULSEG_ALLOC((size_t)(vlen + 1));
                    if (!def->value[j])
                    {
                        pgr_set_diag(
                            diag,
                            diag_size,
                            "allocation failed for DEFINITIONS value string");
                        return 0;
                    }
                    if (fread(def->value[j], 1, (size_t)vlen, f) != (size_t)vlen)
                    {
                        pgr_set_diag(diag, diag_size, "EOF reading DEFINITIONS value string");
                        return 0;
                    }
                    def->value[j][vlen] = '\0';
                }
            }
        }
    }

    return 1;
}

static int pgr_read_definitions_section(
    pulseg_recon_cache *out,
    FILE *f,
    int do_swap,
    long section_offset,
    char *diag,
    int diag_size)
{
    int num_subseq, s;

    if (fseek(f, section_offset, SEEK_SET) != 0)
    {
        pgr_set_diag(diag, diag_size, "cannot seek to DEFINITIONS section");
        return 0;
    }
    if (!pgr_read4(f, &num_subseq, 1))
    {
        pgr_set_diag(diag, diag_size, "EOF reading DEFINITIONS num_subsequences");
        return 0;
    }
    if (do_swap)
        pgr_swap4(&num_subseq);
    if (num_subseq <= 0)
    {
        out->num_subsequences = 0;
        out->definitions_by_subseq = NULL;
        return 1;
    }

    out->definitions_by_subseq = (pulseg_recon_definitions *)PULSEG_ALLOC(
        (size_t)num_subseq * sizeof(pulseg_recon_definitions));
    if (!out->definitions_by_subseq)
    {
        pgr_set_diag(diag, diag_size, "allocation failed for per-subsequence DEFINITIONS");
        return 0;
    }
    memset(out->definitions_by_subseq, 0, (size_t)num_subseq * sizeof(pulseg_recon_definitions));
    out->num_subsequences = num_subseq;

    for (s = 0; s < num_subseq; ++s)
    {
        if (!pgr_read_one_definitions_block(
                &out->definitions_by_subseq[s],
                f,
                do_swap,
                diag,
                diag_size))
            return 0;
    }

    return 1;
}

/* ================================================================== */
/*  Section 7 (SEQDESC)                                                */
/* ================================================================== */

static int pgr_read_rf_shape(
    pulseg_recon_rf_shape *shape,
    FILE *f,
    int do_swap,
    char *diag,
    int diag_size)
{
    shape->num_uncompressed = 0;
    shape->num_samples = 0;
    shape->samples = NULL;

    if (!pgr_read4(f, &shape->num_uncompressed, 1) || !pgr_read4(f, &shape->num_samples, 1))
    {
        pgr_set_diag(diag, diag_size, "EOF reading RF shape header");
        return 0;
    }
    if (do_swap)
    {
        pgr_swap4(&shape->num_uncompressed);
        pgr_swap4(&shape->num_samples);
    }
    if (shape->num_samples > 0)
    {
        shape->samples = (float *)PULSEG_ALLOC((size_t)shape->num_samples * sizeof(float));
        if (!shape->samples)
        {
            pgr_set_diag(diag, diag_size, "allocation failed for RF shape samples");
            return 0;
        }
        if (!pgr_read4(f, shape->samples, shape->num_samples))
        {
            pgr_set_diag(diag, diag_size, "EOF reading RF shape samples");
            return 0;
        }
        if (do_swap)
            pgr_swap4_array(shape->samples, shape->num_samples);
    }
    return 1;
}

static int pgr_read_rf_def_library(
    pulseg_recon_seq_desc *sd,
    FILE *f,
    int do_swap,
    char *diag,
    int diag_size)
{
    int num_rf_defs, rd;

    if (!pgr_read4(f, &num_rf_defs, 1))
    {
        pgr_set_diag(diag, diag_size, "EOF reading SEQDESC num_rf_defs");
        return 0;
    }
    if (do_swap)
        pgr_swap4(&num_rf_defs);
    sd->num_rf_defs = 0;
    sd->rf_defs = NULL;
    if (num_rf_defs <= 0)
        return 1;

    sd->rf_defs =
        (pulseg_recon_rf_def *)PULSEG_ALLOC((size_t)num_rf_defs * sizeof(pulseg_recon_rf_def));
    if (!sd->rf_defs)
    {
        pgr_set_diag(diag, diag_size, "allocation failed for RF-def library");
        return 0;
    }
    memset(sd->rf_defs, 0, (size_t)num_rf_defs * sizeof(pulseg_recon_rf_def));
    sd->num_rf_defs = num_rf_defs;

    for (rd = 0; rd < num_rf_defs; ++rd)
    {
        pulseg_recon_rf_def *def = &sd->rf_defs[rd];
        int has_phase, has_time;

        if (!pgr_read4(f, &def->rf_def_id, 1) || !pgr_read4(f, &def->bandwidth_hz, 1) ||
            !pgr_read4(f, &def->num_bands, 1))
        {
            pgr_set_diag(diag, diag_size, "EOF reading RF-def header");
            return 0;
        }
        if (!pgr_read4(f, def->band_freq_offsets_hz, PULSEG_MAX_BANDS))
        {
            pgr_set_diag(diag, diag_size, "EOF reading RF-def band_freq_offsets_hz");
            return 0;
        }
        if (!pgr_read4(f, &def->band_bandwidth_hz, 1) || !pgr_read4(f, &def->total_b1sq_power, 1))
        {
            pgr_set_diag(diag, diag_size, "EOF reading RF-def band/b1sq fields");
            return 0;
        }
        if (do_swap)
        {
            pgr_swap4(&def->rf_def_id);
            pgr_swap4(&def->bandwidth_hz);
            pgr_swap4(&def->num_bands);
            pgr_swap4_array(def->band_freq_offsets_hz, PULSEG_MAX_BANDS);
            pgr_swap4(&def->band_bandwidth_hz);
            pgr_swap4(&def->total_b1sq_power);
        }

        if (!pgr_read_rf_shape(&def->mag, f, do_swap, diag, diag_size))
            return 0;

        if (!pgr_read4(f, &has_phase, 1))
        {
            pgr_set_diag(diag, diag_size, "EOF reading RF-def has_phase");
            return 0;
        }
        if (do_swap)
            pgr_swap4(&has_phase);
        def->has_phase = has_phase != 0;
        if (def->has_phase && !pgr_read_rf_shape(&def->phase, f, do_swap, diag, diag_size))
            return 0;

        if (!pgr_read4(f, &has_time, 1))
        {
            pgr_set_diag(diag, diag_size, "EOF reading RF-def has_time");
            return 0;
        }
        if (do_swap)
            pgr_swap4(&has_time);
        def->has_time = has_time != 0;
        if (def->has_time && !pgr_read_rf_shape(&def->time, f, do_swap, diag, diag_size))
            return 0;
    }

    return 1;
}

/* Build the deduped (rf_def_id, rf_shim_id, ss_grad_amp_hz_per_m)
 * RF-shape-tuple library from a subsequence's event rows. Linear scan --
 * no std::map equivalent for a C89 module, and event counts per
 * subsequence are small. */
static int pgr_build_rf_shape_tuples(pulseg_recon_seq_desc *sd, char *diag, int diag_size)
{
    int max_tuples, i, n_tuples;
    pulseg_recon_rf_shape_tuple *tuples;

    sd->num_rf_shape_tuples = 0;
    sd->rf_shape_tuples = NULL;
    if (sd->num_events <= 0)
        return 1;

    tuples = (pulseg_recon_rf_shape_tuple *)PULSEG_ALLOC(
        (size_t)sd->num_events * sizeof(pulseg_recon_rf_shape_tuple));
    if (!tuples)
    {
        pgr_set_diag(diag, diag_size, "allocation failed for RF-shape tuples");
        return 0;
    }
    max_tuples = sd->num_events;
    n_tuples = 0;

    for (i = 0; i < sd->num_events; ++i)
    {
        const pulseg_seq_event *ev = &sd->events[i];
        int rf_def_id, rf_shim_id, found, k;
        float ss_grad_amp, bw;

        if (ev->type != PULSEG_SEQ_EVENT_RF)
            continue;

        rf_def_id = (int)ev->params[0];
        rf_shim_id = (int)ev->params[5];
        ss_grad_amp = ev->params[6];

        found = 0;
        for (k = 0; k < n_tuples; ++k)
        {
            if (tuples[k].rf_def_id == rf_def_id && tuples[k].rf_shim_id == rf_shim_id &&
                tuples[k].ss_grad_amp_hz_per_m == ss_grad_amp)
            {
                found = 1;
                break;
            }
        }
        if (found)
            continue;

        bw = 0.0f;
        if (rf_def_id >= 0 && rf_def_id < sd->num_rf_defs)
            bw = sd->rf_defs[rf_def_id].bandwidth_hz;

        /* n_tuples < max_tuples always holds: at most one new tuple is
         * appended per event iteration. */
        tuples[n_tuples].tuple_id = n_tuples;
        tuples[n_tuples].rf_def_id = rf_def_id;
        tuples[n_tuples].rf_shim_id = rf_shim_id;
        tuples[n_tuples].ss_grad_amp_hz_per_m = ss_grad_amp;
        if (ss_grad_amp > 0.0f && bw > 0.0f)
        {
            tuples[n_tuples].slice_thickness_mm = (bw / ss_grad_amp) * 1000.0f;
            tuples[n_tuples].slice_selective =
                (tuples[n_tuples].slice_thickness_mm < 10.0f) ? 1 : 0;
        }
        else
        {
            tuples[n_tuples].slice_thickness_mm = 0.0f;
            tuples[n_tuples].slice_selective = 0;
        }
        ++n_tuples;
    }

    (void)max_tuples;
    sd->rf_shape_tuples = tuples;
    sd->num_rf_shape_tuples = n_tuples;
    return 1;
}

static int pgr_read_one_seqdesc_subseq(
    pulseg_recon_seq_desc *sd,
    FILE *f,
    int do_swap,
    char *diag,
    int diag_size)
{
    int num_rows, r;

    memset(sd, 0, sizeof(*sd));

    if (!pgr_read4(f, &sd->subseq_idx, 1) || !pgr_read4(f, &sd->tr_duration_us, 1))
    {
        pgr_set_diag(diag, diag_size, "EOF reading SEQDESC subseq header");
        return 0;
    }
    if (do_swap)
    {
        pgr_swap4(&sd->subseq_idx);
        pgr_swap4(&sd->tr_duration_us);
    }

    if (!pgr_read4(f, &num_rows, 1))
    {
        pgr_set_diag(diag, diag_size, "EOF reading SEQDESC num_rows");
        return 0;
    }
    if (do_swap)
        pgr_swap4(&num_rows);
    sd->num_events = 0;
    sd->events = NULL;
    if (num_rows > 0)
    {
        sd->events = (pulseg_seq_event *)PULSEG_ALLOC((size_t)num_rows * sizeof(pulseg_seq_event));
        if (!sd->events)
        {
            pgr_set_diag(diag, diag_size, "allocation failed for SEQDESC event rows");
            return 0;
        }
        sd->num_events = num_rows;
        for (r = 0; r < num_rows; ++r)
        {
            pulseg_seq_event *ev = &sd->events[r];
            if (!pgr_read4(f, &ev->type, 1) || !pgr_read4(f, &ev->timestamp_us, 1) ||
                !pgr_read4(f, ev->params, PULSEG_SEQ_EVENT_PARAMS))
            {
                pgr_set_diag(diag, diag_size, "EOF reading SEQDESC event row");
                return 0;
            }
            if (do_swap)
            {
                pgr_swap4(&ev->type);
                pgr_swap4(&ev->timestamp_us);
                pgr_swap4_array(ev->params, PULSEG_SEQ_EVENT_PARAMS);
            }
        }
    }

    if (!pgr_read_rf_def_library(sd, f, do_swap, diag, diag_size))
        return 0;

    return pgr_build_rf_shape_tuples(sd, diag, diag_size);
}

static int pgr_read_seqdesc_section(
    pulseg_recon_cache *out,
    FILE *f,
    int do_swap,
    long section_offset,
    char *diag,
    int diag_size)
{
    pulseg_sequence_parameters *sp = &out->seq_params;
    int ss;

    memset(sp, 0, sizeof(*sp));

    if (fseek(f, section_offset, SEEK_SET) != 0)
    {
        pgr_set_diag(diag, diag_size, "cannot seek to SEQDESC section");
        return 0;
    }

    if (!pgr_read4(f, &sp->min_te_us, 1) || !pgr_read4(f, &sp->min_tr_us, 1) ||
        !pgr_read4(f, &sp->max_tr_us, 1) || !pgr_read4(f, &sp->max_flip_angle_deg, 1) ||
        !pgr_read4(f, &sp->total_scan_time_us, 1) || !pgr_read4(f, &sp->num_subseqs, 1) ||
        !pgr_read4(f, sp->reserved, 3))
    {
        pgr_set_diag(diag, diag_size, "EOF reading SEQDESC header");
        return 0;
    }
    if (do_swap)
    {
        pgr_swap4(&sp->min_te_us);
        pgr_swap4(&sp->min_tr_us);
        pgr_swap4(&sp->max_tr_us);
        pgr_swap4(&sp->max_flip_angle_deg);
        pgr_swap4(&sp->total_scan_time_us);
        pgr_swap4(&sp->num_subseqs);
        pgr_swap4_array(sp->reserved, 3);
    }

    out->num_seq_descs = 0;
    out->seq_descs = NULL;
    if (sp->num_subseqs <= 0)
        return 1;

    out->seq_descs = (pulseg_recon_seq_desc *)PULSEG_ALLOC(
        (size_t)sp->num_subseqs * sizeof(pulseg_recon_seq_desc));
    if (!out->seq_descs)
    {
        pgr_set_diag(diag, diag_size, "allocation failed for SEQDESC subsequences");
        return 0;
    }
    memset(out->seq_descs, 0, (size_t)sp->num_subseqs * sizeof(pulseg_recon_seq_desc));
    out->num_seq_descs = sp->num_subseqs;

    for (ss = 0; ss < sp->num_subseqs; ++ss)
    {
        if (!pgr_read_one_seqdesc_subseq(&out->seq_descs[ss], f, do_swap, diag, diag_size))
            return 0;
    }

    return 1;
}

/* ================================================================== */
/*  Public API                                                        */
/* ================================================================== */

int pulseg_recon_cache_read(
    pulseg_recon_cache *out,
    const char *cache_path,
    char *diag,
    int diag_size)
{
    FILE *f;
    int do_swap;
    pgr_section definitions_section, traj_section, seqdesc_section;
    int traj_status;

    if (diag && diag_size > 0)
        diag[0] = '\0';

    if (!out || !cache_path)
    {
        pgr_set_diag(diag, diag_size, "null pointer argument");
        return PULSEG_ERR_NULL_POINTER;
    }
    memset(out, 0, sizeof(*out));

    f = fopen(cache_path, "rb");
    if (!f)
    {
        pgr_set_diag(diag, diag_size, "cannot open cache file");
        return PULSEG_ERR_FILE_NOT_FOUND;
    }

    if (!pgr_read_header_and_toc(
            f,
            &do_swap,
            &definitions_section,
            &traj_section,
            &seqdesc_section,
            diag,
            diag_size))
    {
        fclose(f);
        return PULSEG_ERR_FILE_READ_FAILED;
    }

    /* TRAJECTORY (section 6) and SEQDESC (section 7) are both mandatory as
     * of cache v2.0.0 -- a missing section is a hard error, never a
     * silent degrade. DEFINITIONS (section 0) predates some legacy caches
     * and is tolerated absent. */
    if (traj_section.id < 0)
    {
        pgr_set_diag(diag, diag_size, "TRAJECTORY section (id 6) missing from cache");
        fclose(f);
        return PULSEG_ERR_FILE_READ_FAILED;
    }
    if (seqdesc_section.id < 0)
    {
        pgr_set_diag(diag, diag_size, "SEQDESC section (id 7) missing from cache");
        fclose(f);
        return PULSEG_ERR_FILE_READ_FAILED;
    }

    if (definitions_section.id >= 0)
    {
        if (!pgr_read_definitions_section(
                out,
                f,
                do_swap,
                definitions_section.offset,
                diag,
                diag_size))
        {
            fclose(f);
            pulseg_recon_cache_free(out);
            return PULSEG_ERR_FILE_READ_FAILED;
        }
    }

    if (!pgr_read_seqdesc_section(out, f, do_swap, seqdesc_section.offset, diag, diag_size))
    {
        fclose(f);
        pulseg_recon_cache_free(out);
        return PULSEG_ERR_FILE_READ_FAILED;
    }

    /* Delegate TRAJECTORY parsing to the existing public reader -- do not
     * duplicate that byte-parsing here. It reopens the file itself. */
    fclose(f);
    traj_status = pulseg_load_trajectory_cache(&out->trajectory, cache_path);
    if (traj_status != PULSEG_SUCCESS)
    {
        pgr_set_diag(diag, diag_size, "pulseg_load_trajectory_cache failed on TRAJECTORY section");
        pulseg_recon_cache_free(out);
        return traj_status;
    }

    return PULSEG_SUCCESS;
}

void pulseg_recon_cache_free(pulseg_recon_cache *cache)
{
    int s;

    if (!cache)
        return;

    if (cache->definitions_by_subseq)
    {
        for (s = 0; s < cache->num_subsequences; ++s)
        {
            pulseg_recon_definitions *d = &cache->definitions_by_subseq[s];
            int i;
            for (i = 0; i < d->num_definitions; ++i)
            {
                pulseq_definition *def = &d->definitions[i];
                int j;
                if (def->value)
                {
                    for (j = 0; j < def->value_size; ++j)
                        PULSEG_FREE(def->value[j]);
                    PULSEG_FREE(def->value);
                }
            }
            PULSEG_FREE(d->definitions);
        }
        PULSEG_FREE(cache->definitions_by_subseq);
    }
    cache->definitions_by_subseq = NULL;
    cache->num_subsequences = 0;

    pulseg_trajectory_free(&cache->trajectory);
    memset(&cache->trajectory, 0, sizeof(cache->trajectory));

    if (cache->seq_descs)
    {
        int ss;
        for (ss = 0; ss < cache->num_seq_descs; ++ss)
        {
            pulseg_recon_seq_desc *sd = &cache->seq_descs[ss];
            int rd;
            PULSEG_FREE(sd->events);
            for (rd = 0; rd < sd->num_rf_defs; ++rd)
            {
                pulseg_recon_rf_def *def = &sd->rf_defs[rd];
                PULSEG_FREE(def->mag.samples);
                PULSEG_FREE(def->phase.samples);
                PULSEG_FREE(def->time.samples);
            }
            PULSEG_FREE(sd->rf_defs);
            PULSEG_FREE(sd->rf_shape_tuples);
        }
        PULSEG_FREE(cache->seq_descs);
    }
    cache->seq_descs = NULL;
    cache->num_seq_descs = 0;
}

const pulseq_definition *pulseg_recon_find_definition(
    const pulseg_recon_cache *cache,
    int subseq_idx,
    const char *name)
{
    const pulseg_recon_definitions *d;
    int i;

    if (!cache || !name)
        return NULL;
    if (subseq_idx < 0 || subseq_idx >= cache->num_subsequences)
        return NULL;
    if (!cache->definitions_by_subseq)
        return NULL;

    d = &cache->definitions_by_subseq[subseq_idx];
    for (i = 0; i < d->num_definitions; ++i)
    {
        if (strcmp(d->definitions[i].name, name) == 0)
            return &d->definitions[i];
    }
    return NULL;
}
