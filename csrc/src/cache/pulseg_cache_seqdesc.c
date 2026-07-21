/* pulseg_cache_seqdesc.c -- SEQDESC (section 7) cache writer for sequence description
 *
 * Implements:
 *   pulseg__save_seqdesc_cache_section()
 *
 * SEQDESC (section 7) serialization format (all values are 4 bytes, little-endian
 * by default — same convention as all other cache sections):
 *
 * [sequence parameters]
 *   min_te_us, min_tr_us, max_tr_us, max_flip_angle_deg, total_scan_time_us  (float x5)
 *   num_subseqs                                                               (int x1)
 *   reserved[3]                                                               (int x3)
 *
 * [per-subsequence block, for ss = 0 .. num_subseqs-1]
 *   subseq_idx          (int)
 *   tr_duration_us      (float)
 *   num_rows            (int)     — number of blocks in the pass
 *   [rows, for r = 0 .. num_rows-1]
 *     type              (int)     — PULSEG_SEQ_EVENT_{OTHER,RF,ADC}
 *     timestamp_us      (float)   — pass-relative anchor time (us)
 *     params[7]         (float x7) — type-specific payload (zero-padded)
 *       RF:    params[0]=rf_def_id, [1]=rf_use, [2]=act_amplitude_hz,
 *              [3]=phase_offset_rad, [4]=freq_offset_hz, [5]=rf_shim_id,
 *              [6]=ss_grad_amp_hz_per_m
 *       ADC:   params[0]=adc_role, [1]=phase_offset_rad, [2..6]=0
 *       OTHER: params[0..6]=0
 *
 * [per-subsequence RF-def library, appended after the row table]
 *   num_rf_defs                                                               (int)
 *   [per rf_def, rf_def_id = array index = rows[].params[0] for RF rows]
 *     rf_def_id           (int)
 *     bandwidth_hz        (float)
 *     num_bands           (int)
 *     band_freq_offsets_hz[PULSEG_MAX_BANDS]                               (float x8)
 *     band_bandwidth_hz   (float)
 *     total_b1sq          (float)
 *     mag  shape: num_uncompressed (int), num_samples (int), samples[]        (float xN) -- COMPRESSED
 *     has_phase (int); if 1: phase shape (same triplet as mag)                -- COMPRESSED
 *     has_time  (int); if 1: time  shape (same triplet as mag)                -- COMPRESSED
 *   Shapes are copied verbatim (still compressed) from the descriptor's
 *   shapes[] table via rf_definitions[].{mag,phase,time}_shape_id — the
 *   recon reader decompresses, SEQDESC never does.
 */

#include <string.h>
#include <stdio.h>
#include <errno.h>

#include "pulseg_internal.h"
#include "pulseg.h"

/* ================================================================== */
/*  I/O helpers (duplicated from pulseg_trajectory.c pattern)       */
/* ================================================================== */

static int sd_write4(FILE *f, const void *p, int count)
{
    return (int)fwrite(p, 4, (size_t)count, f) == count;
}

static int sd_read4(FILE *f, void *p, int count)
{
    return (int)fread(p, 4, (size_t)count, f) == count;
}

static void sd_swap4(void *p)
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

static void sd_swap4_array(void *p, int count)
{
    int i;
    for (i = 0; i < count; ++i)
        sd_swap4((unsigned char *)p + (size_t)i * 4);
}

/* ================================================================== */
/*  Path helper: .seq -> cache_ext (same helper as in trajectory.c)      */
/* ================================================================== */
static char *sd_make_cache_path(const char *seq_path, const char *ext)
{
    size_t len = strlen(seq_path);
    size_t ext_len;
    char *p;
    if (len < 4 || strcmp(seq_path + len - 4, ".seq") != 0)
        return NULL;
    if (!ext || !ext[0])
        ext = PULSEG_CACHE_EXT_DEFAULT;
    ext_len = strlen(ext);
    p = (char *)PULSEG_ALLOC(len - 4 + ext_len + 1);
    if (!p)
        return NULL;
    memcpy(p, seq_path, len - 4);
    memcpy(p + len - 4, ext, ext_len + 1);
    return p;
}

/* ================================================================== */
/*  Write one (still-compressed) RF shape triplet, copied verbatim     */
/*  from desc->shapes[shape_id - 1]. shape_id == 0 means absent.       */
/* ================================================================== */
static int sd_write_rf_shape(FILE *f, const pulseg_sequence_descriptor *desc, int shape_id)
{
    const pulseq_shape *shape;
    int n;

    if (shape_id <= 0 || shape_id > desc->num_shapes)
    {
        int zero = 0;
        return sd_write4(f, &zero, 1) && sd_write4(f, &zero, 1);
    }

    shape = &desc->shapes[shape_id - 1];
    if (!sd_write4(f, &shape->num_uncompressed_samples, 1))
        return 0;
    if (!sd_write4(f, &shape->num_samples, 1))
        return 0;
    n = shape->num_samples;
    if (n > 0 && shape->samples)
        if (!sd_write4(f, shape->samples, n))
            return 0;

    return 1;
}

/* ================================================================== */
/*  Write the per-subsequence RF-def library (rf_def_id == array       */
/*  index, matching rows[].params[0] for RF rows).                     */
/* ================================================================== */
static int sd_write_rf_def_library(FILE *f, const pulseg_sequence_descriptor *desc)
{
    int i;

    if (!sd_write4(f, &desc->num_unique_rfs, 1))
        return 0;
    for (i = 0; i < desc->num_unique_rfs; ++i)
    {
        const pulseg_rf_definition *rdef = &desc->rf_definitions[i];
        const pulseg_rf_stats *stats = &rdef->stats;
        int has_phase, has_time;

        if (!sd_write4(f, &i, 1))
            return 0;
        if (!sd_write4(f, &stats->bandwidth_hz, 1))
            return 0;
        if (!sd_write4(f, &stats->num_bands, 1))
            return 0;
        if (!sd_write4(f, stats->band_freq_offsets_hz, PULSEG_MAX_BANDS))
            return 0;
        if (!sd_write4(f, &stats->band_bandwidth_hz, 1))
            return 0;
        if (!sd_write4(f, &stats->total_b1sq_power, 1))
            return 0;

        if (!sd_write_rf_shape(f, desc, rdef->mag_shape_id))
            return 0;

        has_phase = rdef->phase_shape_id > 0 && rdef->phase_shape_id <= desc->num_shapes;
        if (!sd_write4(f, &has_phase, 1))
            return 0;
        if (has_phase && !sd_write_rf_shape(f, desc, rdef->phase_shape_id))
            return 0;

        has_time = rdef->time_shape_id > 0 && rdef->time_shape_id <= desc->num_shapes;
        if (!sd_write4(f, &has_time, 1))
            return 0;
        if (has_time && !sd_write_rf_shape(f, desc, rdef->time_shape_id))
            return 0;
    }

    return 1;
}

/* ================================================================== */
/*  Write one subsequence's sequence description block                 */
/* ================================================================== */
static int sd_write_subseq(
    FILE *f,
    const pulseg_sequence_description *sd,
    const pulseg_sequence_descriptor *desc)
{
    int i;

    if (!sd_write4(f, &sd->subseq_idx, 1))
        return 0;
    if (!sd_write4(f, &sd->tr_duration_us, 1))
        return 0;

    /* Compact row table: one row per pass block */
    if (!sd_write4(f, &sd->num_rows, 1))
        return 0;
    for (i = 0; i < sd->num_rows; ++i)
    {
        const pulseg_seq_event *row = &sd->rows[i];
        if (!sd_write4(f, &row->type, 1))
            return 0;
        if (!sd_write4(f, &row->timestamp_us, 1))
            return 0;
        if (!sd_write4(f, row->params, PULSEG_SEQ_EVENT_PARAMS))
            return 0;
    }

    return sd_write_rf_def_library(f, desc);
}

/* ================================================================== */
/*  pulseg__save_seqdesc_cache_section                        */
/* ================================================================== */

#define SD_CACHE_ENDIAN_MARKER 0x01020304
#define SD_CACHE_SECTION_SEQDESC 7

int pulseg__save_seqdesc_cache_section(const pulseg_collection *coll, const char *seq_path)
{
    char *cache_path = NULL;
    FILE *f = NULL;
    int marker, num_sections;
    int version_major, version_minor, version_revision, vendor, stored_size;
    int do_swap;
    long entries_pos, data_start, data_end, hdr_ns_pos;
    int i, found_idx;
    int entries_buf[18 * 3]; /* up to 18 sections x 3 ints */
    pulseg_sequence_description sd;
    pulseg_sequence_parameters sp;
    int ret = PULSEG_SUCCESS;

    if (!coll || !seq_path)
        return PULSEG_ERR_NULL_POINTER;

    cache_path = sd_make_cache_path(
        seq_path,
        coll->num_subsequences > 0 ? coll->descriptors[0].cache_ext : NULL);
    if (!cache_path)
        return PULSEG_ERR_ALLOC_FAILED;

    f = fopen(cache_path, "r+b");
    if (!f)
    {
        ret = PULSEG_ERR_FILE_READ_FAILED;
        goto done;
    }

    /* Read file header */
    if (!sd_read4(f, &marker, 1))
    {
        ret = PULSEG_ERR_FILE_READ_FAILED;
        goto done;
    }
    do_swap = 0;
    if (marker != SD_CACHE_ENDIAN_MARKER)
    {
        sd_swap4(&marker);
        if (marker != SD_CACHE_ENDIAN_MARKER)
        {
            ret = PULSEG_ERR_FILE_READ_FAILED;
            goto done;
        }
        do_swap = 1;
    }
    if (!sd_read4(f, &version_major, 1))
    {
        ret = PULSEG_ERR_FILE_READ_FAILED;
        goto done;
    }
    if (!sd_read4(f, &version_minor, 1))
    {
        ret = PULSEG_ERR_FILE_READ_FAILED;
        goto done;
    }
    if (!sd_read4(f, &version_revision, 1))
    {
        ret = PULSEG_ERR_FILE_READ_FAILED;
        goto done;
    }
    if (!sd_read4(f, &vendor, 1))
    {
        ret = PULSEG_ERR_FILE_READ_FAILED;
        goto done;
    }
    if (!sd_read4(f, &stored_size, 1))
    {
        ret = PULSEG_ERR_FILE_READ_FAILED;
        goto done;
    }
    hdr_ns_pos = ftell(f);
    if (hdr_ns_pos < 0)
    {
        ret = PULSEG_ERR_FILE_READ_FAILED;
        goto done;
    }
    if (!sd_read4(f, &num_sections, 1))
    {
        ret = PULSEG_ERR_FILE_READ_FAILED;
        goto done;
    }
    if (do_swap)
    {
        sd_swap4(&version_major);
        sd_swap4(&version_minor);
        sd_swap4(&version_revision);
        sd_swap4(&vendor);
        sd_swap4(&stored_size);
        sd_swap4(&num_sections);
    }
    if (num_sections <= 0 || num_sections > 17)
    {
        ret = PULSEG_ERR_FILE_READ_FAILED;
        goto done;
    }

    entries_pos = ftell(f);
    if (entries_pos < 0)
    {
        ret = PULSEG_ERR_FILE_READ_FAILED;
        goto done;
    }

    /* Read existing section entries */
    for (i = 0; i < num_sections; ++i)
    {
        if (!sd_read4(f, &entries_buf[i * 3], 3))
        {
            ret = PULSEG_ERR_FILE_READ_FAILED;
            goto done;
        }
        if (do_swap)
            sd_swap4_array(&entries_buf[i * 3], 3);
    }

    /* Find or allocate slot for the seqdesc section */
    found_idx = -1;
    for (i = 0; i < num_sections; ++i)
    {
        if (entries_buf[i * 3] == SD_CACHE_SECTION_SEQDESC)
        {
            found_idx = i;
            break;
        }
    }
    if (found_idx < 0)
    {
        found_idx = num_sections;
        entries_buf[found_idx * 3] = SD_CACHE_SECTION_SEQDESC;
        num_sections++;
    }

    /* Seek to end and start writing */
    if (fseek(f, 0, SEEK_END) != 0)
    {
        ret = PULSEG_ERR_FILE_READ_FAILED;
        goto done;
    }
    data_start = ftell(f);
    if (data_start < 0)
    {
        ret = PULSEG_ERR_FILE_READ_FAILED;
        goto done;
    }

    /* Sequence parameters */
    ret = pulseg_get_sequence_parameters(&sp, coll);
    if (ret != PULSEG_SUCCESS)
        goto done;

    if (!sd_write4(f, &sp.min_te_us, 1) || !sd_write4(f, &sp.min_tr_us, 1) ||
        !sd_write4(f, &sp.max_tr_us, 1) || !sd_write4(f, &sp.max_flip_angle_deg, 1) ||
        !sd_write4(f, &sp.total_scan_time_us, 1) || !sd_write4(f, &sp.num_subseqs, 1) ||
        !sd_write4(f, sp.reserved, 3))
    {
        ret = PULSEG_ERR_FILE_READ_FAILED;
        goto done;
    }

    /* Per-subsequence blocks */
    for (i = 0; i < coll->num_subsequences; ++i)
    {
        memset(&sd, 0, sizeof(sd));
        ret = pulseg_get_sequence_description(&sd, coll, i);
        if (ret != PULSEG_SUCCESS)
        {
            pulseg_sequence_description_free(&sd);
            goto done;
        }
        if (!sd_write_subseq(f, &sd, &coll->descriptors[i]))
        {
            pulseg_sequence_description_free(&sd);
            ret = PULSEG_ERR_FILE_READ_FAILED;
            goto done;
        }
        pulseg_sequence_description_free(&sd);
    }

    data_end = ftell(f);
    if (data_end < 0)
    {
        ret = PULSEG_ERR_FILE_READ_FAILED;
        goto done;
    }

    entries_buf[found_idx * 3 + 1] = (int)data_start;
    entries_buf[found_idx * 3 + 2] = (int)(data_end - data_start);

    /* Patch num_sections in header */
    if (fseek(f, hdr_ns_pos, SEEK_SET) != 0)
    {
        ret = PULSEG_ERR_FILE_READ_FAILED;
        goto done;
    }
    if (!sd_write4(f, &num_sections, 1))
    {
        ret = PULSEG_ERR_FILE_READ_FAILED;
        goto done;
    }

    /* Rewrite all section entries */
    if (fseek(f, entries_pos, SEEK_SET) != 0)
    {
        ret = PULSEG_ERR_FILE_READ_FAILED;
        goto done;
    }
    for (i = 0; i < num_sections; ++i)
    {
        if (!sd_write4(f, &entries_buf[i * 3], 3))
        {
            ret = PULSEG_ERR_FILE_READ_FAILED;
            goto done;
        }
    }

done:
    if (f)
        fclose(f);
    if (cache_path)
        PULSEG_FREE(cache_path);
    return ret;
}
