/**
 * @file pulseg_cache.c
 * @brief The binary sequence cache: section index, writer, and loaders.
 *
 * One file beside the .seq holds every derived structure, split into
 * independently-loadable sections so a consumer pays only for what it reads:
 * the pulse-generation pass takes COMMON+SHAPES, the scan loop additionally
 * takes INSTANCES, ROTATIONS and SCANLOOP, and recon reads DEFINITIONS,
 * TRAJECTORY and SEQDESC through pulseg_recon.c.
 *
 * All integer and float fields are 4 bytes. Endianness is recorded in the
 * header, and a reader on the opposite endianness byte-swaps on the way in --
 * the GE target processor is big-endian while the host that writes the cache
 * is not.
 */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>

#include "pulseg_internal.h"
#include "pulseg.h"

/* ================================================================== */
/*  Binary cache: serialization / deserialization                     */
/* ================================================================== */

#define PULSEG_CACHE_ENDIAN_MARKER 0x01020304
#define PULSEG_CACHE_VERSION_MAJOR 2
#define PULSEG_CACHE_VERSION_MINOR 0
/* Bumped when COMMON gained label_column_map[3]
 * (after vendor), pulseg_rf_stats.vendor_stat[4] replaces the four named
 * GE fields (same layout, no bump needed for that alone), and a VENDOR
 * section id was added -- the label_column_map addition changes COMMON's
 * on-disk layout, so old caches must be rejected, not silently misread. */
/* Bumped again for the interior static/dynamic pure-delay split: each
 * pulseg_virtual_segment gained an is_dynamic_delay[num_blocks] array (written
 * right after has_adc), changing COMMON's per-segment on-disk layout. */
/* Bumped again for the safety-group MODULE label: pulseg_block_table_element
 * gained module_id (written/read right after rf_shim_id), changing the
 * block-table's per-entry on-disk layout. */
/* Bumped again for the per-instance/definition split: the O(scan-length)
 * event tables (block/rf/grad/adc tables and the label table) moved out of
 * COMMON into the new INSTANCES section, COMMON's collection scalars gained
 * total_readouts, and each segment definition gained its frozen per-position
 * initial-state records. */
/* Bumped again for the label-table row-count fix: the table is now one row
 * per ADC along the execution stream. The on-disk framing is unchanged
 * (count followed by that many rows), but a stale cache carries a
 * label_num_entries inflated by num_trs (or, on multi-pass sequences,
 * truncated to a single pass), so it must be regenerated rather than
 * loaded and trusted. */
/* Bumped again for the compact execution stream: SCANLOOP now stores
 * exec_runs[] + the seg-id RLE (one period) + the tr_start anchor instead
 * of four int-per-block arrays, so the on-disk SCANLOOP layout changed
 * outright. */
#define PULSEG_CACHE_VERSION_REVISION 6

/* Per-consumer sections. Each carries its own distinct payload.
 * COMMON establishes the collection + descriptor framing; DEFINITIONS,
 * ROTATIONS, SHAPES, SCANLOOP and INSTANCES augment the descriptors already
 * allocated by COMMON, so COMMON must always be read first. */
#define PULSEG_CACHE_SECTION_DEFINITIONS 0
#define PULSEG_CACHE_SECTION_COMMON 1
#define PULSEG_CACHE_SECTION_ROTATIONS 2
#define PULSEG_CACHE_SECTION_SHAPES 3
#define PULSEG_CACHE_SECTION_SCANLOOP 4
#define PULSEG_CACHE_SECTION_FREQMOD 5
#define PULSEG_CACHE_SECTION_TRAJECTORY 6
#define PULSEG_CACHE_SECTION_SEQDESC 7
/* Opaque vendor extension section: length-prefixed blob, written
 * only when pulseg_opts.vendor_section_write_fn is set (GE leaves it
 * unused). Format: [int byte_len][byte_len raw bytes]. */
#define PULSEG_CACHE_SECTION_VENDOR 8
/* Per-instance event tables: block_table, rf/grad/adc tables and the label
 * table -- everything whose size scales with the scan length rather than
 * with the number of unique definitions. Loaded by the scan path only; the
 * pulsegen path resolves per-position state from the segment definitions'
 * frozen initial-state records instead. */
#define PULSEG_CACHE_SECTION_INSTANCES 9

/* Total number of defined section IDs (0..9 above). write_cache() reserves
 * a section-entry table sized for this many slots up front -- even though
 * it only writes the 6 base sections itself -- so that the later append
 * passes (pulseg_write_trajectory_cache, pulseg__save_freq_mod_section,
 * the optional SEQDESC writer, the optional VENDOR writer) can insert
 * their own entries into the same table without growing past its reserved
 * size and overflowing into the COMMON payload that immediately follows
 * it on disk. */
#define PULSEG_CACHE_MAX_SECTIONS 10

typedef struct pulseg_cache_section_entry
{
    int section_id;
    int offset;
    int size;
} pulseg_cache_section_entry;

/* ------ Byte-swap helpers ------ */

void pulseg__swap4(void *p)
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

void pulseg__swap4_array(void *p, int count)
{
    int i;
    for (i = 0; i < count; ++i)
        pulseg__swap4((unsigned char *)p + (size_t)i * 4);
}

/* ------ I/O helpers ------ */

int pulseg__write4(FILE *f, const void *p, int count)
{
    return (int)fwrite(p, 4, (size_t)count, f) == count;
}

int pulseg__read4(FILE *f, void *p, int count)
{
    return (int)fread(p, 4, (size_t)count, f) == count;
}

/* ------ Path helper ------ */

/* ext: cache file extension including the dot (e.g. ".pseg"); NULL or
 * empty falls back to PULSEG_CACHE_EXT_DEFAULT. */
char *pulseg__make_cache_path(const char *seq_path, const char *ext)
{
    size_t len, ext_len;
    char *out;
    const char *dot;

    if (!ext || !ext[0])
        ext = PULSEG_CACHE_EXT_DEFAULT;
    ext_len = strlen(ext);

    len = strlen(seq_path);
    out = (char *)PULSEG_ALLOC(len + ext_len + 1);
    if (!out)
        return NULL;

    strcpy(out, seq_path);
    dot = strrchr(out, '.');
    if (dot && dot > strrchr(out, '/') && dot > strrchr(out, '\\'))
    {
        /* replace extension */
        strcpy((char *)(out + (dot - out)), ext);
    }
    else
    {
        strcat(out, ext);
    }
    return out;
}

/* ------ File size helper (C89) ------ */

static long get_file_size(const char *path)
{
    FILE *f;
    long sz;
    f = fopen(path, "rb");
    if (!f)
        return -1;
    fseek(f, 0, SEEK_END);
    sz = ftell(f);
    fclose(f);
    return sz;
}

/* ------ Serialize the COMMON region of a descriptor ------ */
/* The definition-scale state only: scalars, base blocks, the RF/gradient/ADC/
 * shim/trigger definition libraries, the TR descriptor and the segment
 * definitions (including their frozen per-position initial-state records).
 * Everything whose size scales with the scan length lives in INSTANCES; raw
 * shape samples, rotation matrices, exec_stream and variable_grad_flags live
 * in SHAPES/ROTATIONS/SCANLOOP. */

static int write_common(FILE *f, const pulseg_sequence_descriptor *d)
{
    int i;
    int ival;

    /* scalars */
    if (!pulseg__write4(f, &d->num_prep_blocks, 1))
        return 0;
    if (!pulseg__write4(f, &d->num_cooldown_blocks, 1))
        return 0;
    if (!pulseg__write4(f, &d->rf_raster_us, 1))
        return 0;
    if (!pulseg__write4(f, &d->grad_raster_us, 1))
        return 0;
    if (!pulseg__write4(f, &d->adc_raster_us, 1))
        return 0;
    if (!pulseg__write4(f, &d->block_raster_us, 1))
        return 0;
    if (!pulseg__write4(f, &d->ignore_fov_shift, 1))
        return 0;
    if (!pulseg__write4(f, &d->enable_pmc, 1))
        return 0;
    if (!pulseg__write4(f, &d->ignore_averages, 1))
        return 0;
    if (!pulseg__write4(f, &d->num_gain_cal_readouts, 1))
        return 0;
    if (!pulseg__write4(f, &d->num_passes, 1))
        return 0;
    if (!pulseg__write4(f, &d->vendor, 1))
        return 0;
    if (!pulseg__write4(f, d->label_column_map, 3))
        return 0;
    if (!pulseg__write4(f, d->fov, 3))
        return 0;
    if (!pulseg__write4(f, d->matrix, 3))
        return 0;
    if (!pulseg__write4(f, d->nav_fov, 3))
        return 0;
    if (!pulseg__write4(f, d->nav_matrix, 3))
        return 0;

    /* block definitions */
    if (!pulseg__write4(f, &d->num_unique_blocks, 1))
        return 0;
    for (i = 0; i < d->num_unique_blocks; ++i)
    {
        if (!pulseg__write4(f, &d->base_blocks[i].id, 1))
            return 0;
        if (!pulseg__write4(f, &d->base_blocks[i].duration_us, 1))
            return 0;
        if (!pulseg__write4(f, &d->base_blocks[i].rf_id, 1))
            return 0;
        if (!pulseg__write4(f, &d->base_blocks[i].gx_id, 1))
            return 0;
        if (!pulseg__write4(f, &d->base_blocks[i].gy_id, 1))
            return 0;
        if (!pulseg__write4(f, &d->base_blocks[i].gz_id, 1))
            return 0;
        if (!pulseg__write4(f, &d->base_blocks[i].adc_id, 1))
            return 0;
    }

    /* RF definitions */
    if (!pulseg__write4(f, &d->num_unique_rfs, 1))
        return 0;
    for (i = 0; i < d->num_unique_rfs; ++i)
    {
        if (!pulseg__write4(f, &d->rf_definitions[i].id, 1))
            return 0;
        if (!pulseg__write4(f, &d->rf_definitions[i].mag_shape_id, 1))
            return 0;
        if (!pulseg__write4(f, &d->rf_definitions[i].phase_shape_id, 1))
            return 0;
        if (!pulseg__write4(f, &d->rf_definitions[i].time_shape_id, 1))
            return 0;
        if (!pulseg__write4(f, &d->rf_definitions[i].delay, 1))
            return 0;
        if (!pulseg__write4(f, &d->rf_definitions[i].num_channels, 1))
            return 0;
        if (!pulseg__write4(f, &d->rf_definitions[i].stats.flip_angle_rad, 1))
            return 0;
        if (!pulseg__write4(f, &d->rf_definitions[i].stats.area, 1))
            return 0;
        if (!pulseg__write4(f, &d->rf_definitions[i].stats.vendor_stat[0], 1))
            return 0;
        if (!pulseg__write4(f, &d->rf_definitions[i].stats.vendor_stat[1], 1))
            return 0;
        if (!pulseg__write4(f, &d->rf_definitions[i].stats.vendor_stat[2], 1))
            return 0;
        if (!pulseg__write4(f, &d->rf_definitions[i].stats.vendor_stat[3], 1))
            return 0;
        if (!pulseg__write4(f, &d->rf_definitions[i].stats.duration_us, 1))
            return 0;
        if (!pulseg__write4(f, &d->rf_definitions[i].stats.isodelay_us, 1))
            return 0;
        if (!pulseg__write4(f, &d->rf_definitions[i].stats.bandwidth_hz, 1))
            return 0;
        if (!pulseg__write4(f, &d->rf_definitions[i].stats.base_amplitude_hz, 1))
            return 0;
        if (!pulseg__write4(f, &d->rf_definitions[i].stats.num_samples, 1))
            return 0;
        /* v20: multiband/power fields */
        if (!pulseg__write4(f, &d->rf_definitions[i].stats.num_bands, 1))
            return 0;
        if (!pulseg__write4(f, d->rf_definitions[i].stats.band_freq_offsets_hz, PULSEG_MAX_BANDS))
            return 0;
        if (!pulseg__write4(f, &d->rf_definitions[i].stats.band_bandwidth_hz, 1))
            return 0;
        if (!pulseg__write4(f, &d->rf_definitions[i].stats.total_b1sq_power, 1))
            return 0;
        /* v1.3: vendor tag */
        if (!pulseg__write4(f, &d->rf_definitions[i].stats.vendor, 1))
            return 0;
    }

    /* gradient definitions */
    if (!pulseg__write4(f, &d->num_unique_grads, 1))
        return 0;
    for (i = 0; i < d->num_unique_grads; ++i)
    {
        const pulseg_grad_definition *gd = &d->grad_definitions[i];
        if (!pulseg__write4(f, &gd->id, 1))
            return 0;
        if (!pulseg__write4(f, &gd->type, 1))
            return 0;
        if (!pulseg__write4(f, &gd->rise_time_or_unused, 1))
            return 0;
        if (!pulseg__write4(f, &gd->flat_time_or_unused, 1))
            return 0;
        if (!pulseg__write4(f, &gd->fall_time_or_num_uncompressed_samples, 1))
            return 0;
        if (!pulseg__write4(f, &gd->unused_or_time_shape_id, 1))
            return 0;
        if (!pulseg__write4(f, &gd->delay, 1))
            return 0;
        if (!pulseg__write4(f, &gd->num_shots, 1))
            return 0;
        if (!pulseg__write4(f, gd->shot_shape_ids, PULSEG_MAX_GRAD_SHOTS))
            return 0;
        if (!pulseg__write4(f, gd->max_amplitude, PULSEG_MAX_GRAD_SHOTS))
            return 0;
        if (!pulseg__write4(f, gd->min_amplitude, PULSEG_MAX_GRAD_SHOTS))
            return 0;
        if (!pulseg__write4(f, gd->slew_rate, PULSEG_MAX_GRAD_SHOTS))
            return 0;
        if (!pulseg__write4(f, gd->energy, PULSEG_MAX_GRAD_SHOTS))
            return 0;
        if (!pulseg__write4(f, gd->first_value, PULSEG_MAX_GRAD_SHOTS))
            return 0;
        if (!pulseg__write4(f, gd->last_value, PULSEG_MAX_GRAD_SHOTS))
            return 0;
    }

    /* ADC definitions */
    if (!pulseg__write4(f, &d->num_unique_adcs, 1))
        return 0;
    for (i = 0; i < d->num_unique_adcs; ++i)
    {
        if (!pulseg__write4(f, &d->adc_definitions[i].id, 1))
            return 0;
        if (!pulseg__write4(f, &d->adc_definitions[i].num_samples, 1))
            return 0;
        if (!pulseg__write4(f, &d->adc_definitions[i].dwell_time, 1))
            return 0;
        if (!pulseg__write4(f, &d->adc_definitions[i].delay, 1))
            return 0;
    }

    /* freq_mod definitions (no longer stored; write count = 0) */
    {
        int zero = 0;
        if (!pulseg__write4(f, &zero, 1))
            return 0;
    }

    /* rf_shim definitions */
    if (!pulseg__write4(f, &d->num_rf_shims, 1))
        return 0;
    for (i = 0; i < d->num_rf_shims; ++i)
    {
        const pulseg_rf_shim_definition *rs = &d->rf_shim_definitions[i];
        if (!pulseg__write4(f, &rs->id, 1))
            return 0;
        if (!pulseg__write4(f, &rs->num_channels, 1))
            return 0;
        if (rs->num_channels > 0)
        {
            if (!pulseg__write4(f, rs->magnitudes, rs->num_channels))
                return 0;
            if (!pulseg__write4(f, rs->phases, rs->num_channels))
                return 0;
        }
    }

    /* rotations: emitted in the ROTATIONS section (write_rotations) */

    /* triggers — serialize long/short as int for portability */
    if (!pulseg__write4(f, &d->num_triggers, 1))
        return 0;
    for (i = 0; i < d->num_triggers; ++i)
    {
        ival = (int)d->trigger_events[i].type;
        if (!pulseg__write4(f, &ival, 1))
            return 0;
        ival = (int)d->trigger_events[i].duration;
        if (!pulseg__write4(f, &ival, 1))
            return 0;
        ival = (int)d->trigger_events[i].delay;
        if (!pulseg__write4(f, &ival, 1))
            return 0;
        if (!pulseg__write4(f, &d->trigger_events[i].trigger_type, 1))
            return 0;
        if (!pulseg__write4(f, &d->trigger_events[i].trigger_channel, 1))
            return 0;
    }

    /* shapes: emitted in the SHAPES section (write_shapes) */

    /* TR descriptor (10 fields: 9 int + 1 float) */
    if (!pulseg__write4(f, &d->tr_descriptor.num_prep_blocks, 1))
        return 0;
    if (!pulseg__write4(f, &d->tr_descriptor.num_cooldown_blocks, 1))
        return 0;
    if (!pulseg__write4(f, &d->tr_descriptor.tr_size, 1))
        return 0;
    if (!pulseg__write4(f, &d->tr_descriptor.num_trs, 1))
        return 0;
    if (!pulseg__write4(f, &d->tr_descriptor.num_prep_trs, 1))
        return 0;
    if (!pulseg__write4(f, &d->tr_descriptor.degenerate_prep, 1))
        return 0;
    if (!pulseg__write4(f, &d->tr_descriptor.num_cooldown_trs, 1))
        return 0;
    if (!pulseg__write4(f, &d->tr_descriptor.degenerate_cooldown, 1))
        return 0;
    if (!pulseg__write4(f, &d->tr_descriptor.imaging_tr_start, 1))
        return 0;
    if (!pulseg__write4(f, &d->tr_descriptor.tr_duration_us, 1))
        return 0;

    /* segment definitions */
    if (!pulseg__write4(f, &d->num_unique_segments, 1))
        return 0;
    for (i = 0; i < d->num_unique_segments; ++i)
    {
        const pulseg_virtual_segment *seg = &d->segment_definitions[i];
        if (!pulseg__write4(f, &seg->start_block, 1))
            return 0;
        if (!pulseg__write4(f, &seg->num_blocks, 1))
            return 0;
        if (!pulseg__write4(f, &seg->max_energy_start_block, 1))
            return 0;
        if (seg->num_blocks > 0)
        {
            if (!pulseg__write4(f, seg->unique_block_indices, seg->num_blocks))
                return 0;
            if (!pulseg__write4(f, seg->has_digitalout, seg->num_blocks))
                return 0;
            if (!pulseg__write4(f, seg->has_rotation, seg->num_blocks))
                return 0;
            if (!pulseg__write4(f, seg->norot_flag, seg->num_blocks))
                return 0;
            if (!pulseg__write4(f, seg->nopos_flag, seg->num_blocks))
                return 0;
            if (!pulseg__write4(f, seg->has_freq_mod, seg->num_blocks))
                return 0;
            if (!pulseg__write4(f, seg->has_adc, seg->num_blocks))
                return 0;
            if (!pulseg__write4(f, seg->is_dynamic_delay, seg->num_blocks))
                return 0;
            /* Frozen per-position initial event state (structure step 11e):
             * what the representative scan instance resolves to, so the
             * pulsegen path never needs the INSTANCES section. All fields are
             * 4-byte, so the records serialize as a packed word array. */
            if (!pulseg__write4(
                    f,
                    seg->initial_states,
                    seg->num_blocks * PULSEG_BLOCK_INITIAL_STATE_WORDS))
                return 0;
        }
        if (!pulseg__write4(f, &seg->trigger_id, 1))
            return 0;
        if (!pulseg__write4(f, &seg->is_nav, 1))
            return 0;

        /* Segment timing anchors (k-space refs: RF isocenter, ADC kzero, plus
         * the gap edges). calc_segment_timing builds these during parse, so they
         * are live here. They MUST be serialized: the geninstructions/scanloop
         * cache load paths do not rebuild them (no trajectory available), and
         * freq-mod requires the exact isocenter/kzero. All fields are 4-byte, so
         * the structs serialize as packed word arrays. */
        if (!pulseg__write4(f, &seg->timing.num_rf_anchors, 1))
            return 0;
        if (seg->timing.num_rf_anchors > 0)
        {
            if (!pulseg__write4(
                    f,
                    seg->timing.rf_anchors,
                    seg->timing.num_rf_anchors * (int)(sizeof(pulseg_segment_rf_anchor) / 4)))
                return 0;
        }
        if (!pulseg__write4(f, &seg->timing.num_adc_anchors, 1))
            return 0;
        if (seg->timing.num_adc_anchors > 0)
        {
            if (!pulseg__write4(
                    f,
                    seg->timing.adc_anchors,
                    seg->timing.num_adc_anchors * (int)(sizeof(pulseg_segment_adc_anchor) / 4)))
                return 0;
        }
    }

    /* segment table */
    if (!pulseg__write4(f, &d->segment_table.num_unique_segments, 1))
        return 0;
    if (!pulseg__write4(f, &d->segment_table.num_prep_segments, 1))
        return 0;
    if (d->segment_table.num_prep_segments > 0)
        if (!pulseg__write4(
                f,
                d->segment_table.prep_segment_table,
                d->segment_table.num_prep_segments))
            return 0;
    if (!pulseg__write4(f, &d->segment_table.num_main_segments, 1))
        return 0;
    if (d->segment_table.num_main_segments > 0)
        if (!pulseg__write4(
                f,
                d->segment_table.main_segment_table,
                d->segment_table.num_main_segments))
            return 0;
    if (!pulseg__write4(f, &d->segment_table.num_cooldown_segments, 1))
        return 0;
    if (d->segment_table.num_cooldown_segments > 0)
        if (!pulseg__write4(
                f,
                d->segment_table.cooldown_segment_table,
                d->segment_table.num_cooldown_segments))
            return 0;

    /* generic [DEFINITIONS] kv: emitted in the DEFINITIONS section
     * (write_definitions). COMMON keeps only the structured fov/matrix scalars
     * above for PSD-internal use. */

    /* exec_stream + variable_grad_flags: emitted in the SCANLOOP section
     * (write_scanloop) */

    return 1;
}

/* ------ Serialize the DEFINITIONS region of a descriptor ------ */
/* Per-subsequence [DEFINITIONS]: the generic length-prefixed name->values kv,
 * copied verbatim from the source .seq file's [DEFINITIONS] block (recon's
 * authoritative ISMRMRD-header override + per-ES seq params). FOV/Matrix/
 * NavFOV/NavMatrix are already present in this kv as the original pulseq
 * string entries (parsing them into d->fov/matrix/nav_fov/nav_matrix for
 * PSD-internal use does not remove them here) — no separate geometry block
 * needed; COMMON's structured floats stay PSD-internal only. */

static int write_definitions(FILE *f, const pulseg_sequence_descriptor *d)
{
    int i;

    fwrite(&d->num_definitions, sizeof(int), 1, f);
    for (i = 0; i < d->num_definitions; ++i)
    {
        int name_len = (int)strlen(d->definitions[i].name);
        fwrite(&name_len, sizeof(int), 1, f);
        fwrite(d->definitions[i].name, 1, (size_t)name_len, f);
        fwrite(&d->definitions[i].value_size, sizeof(int), 1, f);
        {
            int j;
            for (j = 0; j < d->definitions[i].value_size; ++j)
            {
                int vlen = (int)strlen(d->definitions[i].value[j]);
                fwrite(&vlen, sizeof(int), 1, f);
                fwrite(d->definitions[i].value[j], 1, (size_t)vlen, f);
            }
        }
    }

    return 1;
}

/* ------ Serialize the ROTATIONS region of a descriptor ------ */

static int write_rotations(FILE *f, const pulseg_sequence_descriptor *d)
{
    int i;

    if (!pulseg__write4(f, &d->num_rotations, 1))
        return 0;
    for (i = 0; i < d->num_rotations; ++i)
        if (!pulseg__write4(f, d->rotation_matrices[i], 9))
            return 0;

    return 1;
}

/* ------ Serialize the SHAPES region of a descriptor ------ */

static int write_shapes(FILE *f, const pulseg_sequence_descriptor *d)
{
    int i, n;

    if (!pulseg__write4(f, &d->num_shapes, 1))
        return 0;
    for (i = 0; i < d->num_shapes; ++i)
    {
        if (!pulseg__write4(f, &d->shapes[i].num_uncompressed_samples, 1))
            return 0;
        if (!pulseg__write4(f, &d->shapes[i].num_samples, 1))
            return 0;
        n = d->shapes[i].num_samples;
        if (n > 0 && d->shapes[i].samples)
            if (!pulseg__write4(f, d->shapes[i].samples, n))
                return 0;
    }

    return 1;
}

/* ------ Serialize the INSTANCES region of a descriptor ------ */
/* The per-instance event tables -- everything sized by the scan length
 * rather than by the number of unique definitions. Only the scan path
 * loads this; pulsegen resolves per-position state from the segment
 * definitions' initial-state records in COMMON. */

static int write_instances(FILE *f, const pulseg_sequence_descriptor *d)
{
    int i;

    /* block table */
    if (!pulseg__write4(f, &d->num_blocks, 1))
        return 0;
    for (i = 0; i < d->num_blocks; ++i)
    {
        if (!pulseg__write4(f, &d->block_table[i].id, 1))
            return 0;
        if (!pulseg__write4(f, &d->block_table[i].duration_us, 1))
            return 0;
        if (!pulseg__write4(f, &d->block_table[i].rf_id, 1))
            return 0;
        if (!pulseg__write4(f, &d->block_table[i].gx_id, 1))
            return 0;
        if (!pulseg__write4(f, &d->block_table[i].gy_id, 1))
            return 0;
        if (!pulseg__write4(f, &d->block_table[i].gz_id, 1))
            return 0;
        if (!pulseg__write4(f, &d->block_table[i].adc_id, 1))
            return 0;
        if (!pulseg__write4(f, &d->block_table[i].digitalout_id, 1))
            return 0;
        if (!pulseg__write4(f, &d->block_table[i].rotation_id, 1))
            return 0;
        if (!pulseg__write4(f, &d->block_table[i].once_flag, 1))
            return 0;
        if (!pulseg__write4(f, &d->block_table[i].norot_flag, 1))
            return 0;
        if (!pulseg__write4(f, &d->block_table[i].nopos_flag, 1))
            return 0;
        if (!pulseg__write4(f, &d->block_table[i].pmc_flag, 1))
            return 0;
        if (!pulseg__write4(f, &d->block_table[i].nav_flag, 1))
            return 0;
        if (!pulseg__write4(f, &d->block_table[i].freq_mod_id, 1))
            return 0;
        if (!pulseg__write4(f, &d->block_table[i].rf_shim_id, 1))
            return 0;
        if (!pulseg__write4(f, &d->block_table[i].module_id, 1))
            return 0;
    }

    /* RF table */
    if (!pulseg__write4(f, &d->rf_table_size, 1))
        return 0;
    for (i = 0; i < d->rf_table_size; ++i)
    {
        if (!pulseg__write4(f, &d->rf_table[i].id, 1))
            return 0;
        if (!pulseg__write4(f, &d->rf_table[i].amplitude, 1))
            return 0;
        if (!pulseg__write4(f, &d->rf_table[i].freq_offset, 1))
            return 0;
        if (!pulseg__write4(f, &d->rf_table[i].phase_offset, 1))
            return 0;
        if (!pulseg__write4(f, &d->rf_table[i].rf_use, 1))
            return 0;
    }

    /* gradient table */
    if (!pulseg__write4(f, &d->grad_table_size, 1))
        return 0;
    for (i = 0; i < d->grad_table_size; ++i)
    {
        if (!pulseg__write4(f, &d->grad_table[i].id, 1))
            return 0;
        if (!pulseg__write4(f, &d->grad_table[i].shot_index, 1))
            return 0;
        if (!pulseg__write4(f, &d->grad_table[i].amplitude, 1))
            return 0;
    }

    /* ADC table */
    if (!pulseg__write4(f, &d->adc_table_size, 1))
        return 0;
    for (i = 0; i < d->adc_table_size; ++i)
    {
        if (!pulseg__write4(f, &d->adc_table[i].id, 1))
            return 0;
        if (!pulseg__write4(f, &d->adc_table[i].freq_offset, 1))
            return 0;
        if (!pulseg__write4(f, &d->adc_table[i].phase_offset, 1))
            return 0;
    }

    /* label table */
    fwrite(&d->label_num_columns, sizeof(int), 1, f);
    fwrite(&d->label_num_entries, sizeof(int), 1, f);
    if (d->label_num_entries > 0 && d->label_table)
    {
        fwrite(
            d->label_table,
            sizeof(int),
            (size_t)d->label_num_entries * (size_t)d->label_num_columns,
            f);
    }
    fwrite(&d->label_limits, sizeof(pulseg_label_limits), 1, f);

    return 1;
}

/* ------ Serialize the SCANLOOP region of a descriptor ------ */

static int write_scanloop(FILE *f, const pulseg_sequence_descriptor *d)
{
    /* Execution stream, compact form. The four per-position arrays
     * (block_idx / tr_id / seg_id / avg_id) are build-time scratch and are
     * never serialized: exec_runs reproduces the first three, the seg RLE
     * the fourth, and tr_start is derived. See pulseg_exec_run. */
    if (!pulseg__write4(f, &d->exec_stream_len, 1))
        return 0;
    if (d->exec_stream_len > 0)
    {
        int i;
        if (!pulseg__write4(f, &d->num_exec_runs, 1))
            return 0;
        for (i = 0; i < d->num_exec_runs; ++i)
        {
            if (!pulseg__write4(f, &d->exec_runs[i].emit_start, 1))
                return 0;
            if (!pulseg__write4(f, &d->exec_runs[i].block_start, 1))
                return 0;
            if (!pulseg__write4(f, &d->exec_runs[i].length, 1))
                return 0;
            if (!pulseg__write4(f, &d->exec_runs[i].tr_id, 1))
                return 0;
            if (!pulseg__write4(f, &d->exec_runs[i].avg_id, 1))
                return 0;
        }
        if (!pulseg__write4(f, &d->num_seg_runs, 1))
            return 0;
        if (!pulseg__write4(f, &d->seg_period, 1))
            return 0;
        if (d->num_seg_runs > 0)
        {
            /* seg_run_start carries num_seg_runs+1 entries (the trailing
             * bound), seg_run_id carries num_seg_runs. */
            if (!pulseg__write4(f, d->seg_run_start, d->num_seg_runs + 1))
                return 0;
            if (!pulseg__write4(f, d->seg_run_id, d->num_seg_runs))
                return 0;
        }
        if (!pulseg__write4(f, &d->tr_start_main_id, 1))
            return 0;
        if (!pulseg__write4(f, &d->tr_start_first, 1))
            return 0;
    }

    /* variable grad flags */
    {
        int vgf_len = (d->variable_grad_flags && d->tr_descriptor.tr_size > 0)
            ? d->tr_descriptor.tr_size * 3
            : 0;
        if (!pulseg__write4(f, &vgf_len, 1))
            return 0;
        if (vgf_len > 0)
        {
            if (!pulseg__write4(f, d->variable_grad_flags, vgf_len))
                return 0;
        }
    }

    return 1;
}

/* ------ Deserialize the COMMON region of a descriptor ------ */

static int read_common(FILE *f, pulseg_sequence_descriptor *d, int do_swap)
{
    int i, n;
    int ival;

    memset(d, 0, sizeof(*d));

    /* scalars */
    if (!pulseg__read4(f, &d->num_prep_blocks, 1))
        return 0;
    if (!pulseg__read4(f, &d->num_cooldown_blocks, 1))
        return 0;
    if (!pulseg__read4(f, &d->rf_raster_us, 1))
        return 0;
    if (!pulseg__read4(f, &d->grad_raster_us, 1))
        return 0;
    if (!pulseg__read4(f, &d->adc_raster_us, 1))
        return 0;
    if (!pulseg__read4(f, &d->block_raster_us, 1))
        return 0;
    if (!pulseg__read4(f, &d->ignore_fov_shift, 1))
        return 0;
    if (!pulseg__read4(f, &d->enable_pmc, 1))
        return 0;
    if (!pulseg__read4(f, &d->ignore_averages, 1))
        return 0;
    if (!pulseg__read4(f, &d->num_gain_cal_readouts, 1))
        return 0;
    if (!pulseg__read4(f, &d->num_passes, 1))
        return 0;
    if (!pulseg__read4(f, &d->vendor, 1))
        return 0;
    if (!pulseg__read4(f, d->label_column_map, 3))
        return 0;
    if (do_swap)
        pulseg__swap4_array(&d->num_prep_blocks, 15);
    if (!pulseg__read4(f, d->fov, 3))
        return 0;
    if (!pulseg__read4(f, d->matrix, 3))
        return 0;
    if (!pulseg__read4(f, d->nav_fov, 3))
        return 0;
    if (!pulseg__read4(f, d->nav_matrix, 3))
        return 0;
    if (do_swap)
        pulseg__swap4_array((int *)d->fov, 12);

    /* block definitions */
    if (!pulseg__read4(f, &d->num_unique_blocks, 1))
        return 0;
    if (do_swap)
        pulseg__swap4(&d->num_unique_blocks);
    d->base_blocks =
        (pulseg_base_block *)PULSEG_ALLOC((size_t)d->num_unique_blocks * sizeof(pulseg_base_block));
    if (!d->base_blocks)
        return 0;
    for (i = 0; i < d->num_unique_blocks; ++i)
    {
        if (!pulseg__read4(f, &d->base_blocks[i].id, 1))
            return 0;
        if (!pulseg__read4(f, &d->base_blocks[i].duration_us, 1))
            return 0;
        if (!pulseg__read4(f, &d->base_blocks[i].rf_id, 1))
            return 0;
        if (!pulseg__read4(f, &d->base_blocks[i].gx_id, 1))
            return 0;
        if (!pulseg__read4(f, &d->base_blocks[i].gy_id, 1))
            return 0;
        if (!pulseg__read4(f, &d->base_blocks[i].gz_id, 1))
            return 0;
        if (!pulseg__read4(f, &d->base_blocks[i].adc_id, 1))
            return 0;
        if (do_swap)
            pulseg__swap4_array(&d->base_blocks[i].id, 7);
    }

    /* RF definitions */
    if (!pulseg__read4(f, &d->num_unique_rfs, 1))
        return 0;
    if (do_swap)
        pulseg__swap4(&d->num_unique_rfs);
    d->rf_definitions = (pulseg_rf_definition *)PULSEG_ALLOC(
        (size_t)d->num_unique_rfs * sizeof(pulseg_rf_definition));
    if (!d->rf_definitions)
        return 0;
    for (i = 0; i < d->num_unique_rfs; ++i)
    {
        if (!pulseg__read4(f, &d->rf_definitions[i].id, 1))
            return 0;
        if (!pulseg__read4(f, &d->rf_definitions[i].mag_shape_id, 1))
            return 0;
        if (!pulseg__read4(f, &d->rf_definitions[i].phase_shape_id, 1))
            return 0;
        if (!pulseg__read4(f, &d->rf_definitions[i].time_shape_id, 1))
            return 0;
        if (!pulseg__read4(f, &d->rf_definitions[i].delay, 1))
            return 0;
        if (!pulseg__read4(f, &d->rf_definitions[i].num_channels, 1))
            return 0;
        if (do_swap)
            pulseg__swap4_array(&d->rf_definitions[i].id, 6);
        if (!pulseg__read4(f, &d->rf_definitions[i].stats.flip_angle_rad, 1))
            return 0;
        if (!pulseg__read4(f, &d->rf_definitions[i].stats.area, 1))
            return 0;
        if (!pulseg__read4(f, &d->rf_definitions[i].stats.vendor_stat[0], 1))
            return 0;
        if (!pulseg__read4(f, &d->rf_definitions[i].stats.vendor_stat[1], 1))
            return 0;
        if (!pulseg__read4(f, &d->rf_definitions[i].stats.vendor_stat[2], 1))
            return 0;
        if (!pulseg__read4(f, &d->rf_definitions[i].stats.vendor_stat[3], 1))
            return 0;
        if (!pulseg__read4(f, &d->rf_definitions[i].stats.duration_us, 1))
            return 0;
        if (!pulseg__read4(f, &d->rf_definitions[i].stats.isodelay_us, 1))
            return 0;
        if (!pulseg__read4(f, &d->rf_definitions[i].stats.bandwidth_hz, 1))
            return 0;
        if (!pulseg__read4(f, &d->rf_definitions[i].stats.base_amplitude_hz, 1))
            return 0;
        if (!pulseg__read4(f, &d->rf_definitions[i].stats.num_samples, 1))
            return 0;
        if (do_swap)
            pulseg__swap4_array(&d->rf_definitions[i].stats.flip_angle_rad, 11);
        /* v20: multiband/power fields */
        if (!pulseg__read4(f, &d->rf_definitions[i].stats.num_bands, 1))
            return 0;
        if (!pulseg__read4(f, d->rf_definitions[i].stats.band_freq_offsets_hz, PULSEG_MAX_BANDS))
            return 0;
        if (!pulseg__read4(f, &d->rf_definitions[i].stats.band_bandwidth_hz, 1))
            return 0;
        if (!pulseg__read4(f, &d->rf_definitions[i].stats.total_b1sq_power, 1))
            return 0;
        if (do_swap)
            pulseg__swap4_array(&d->rf_definitions[i].stats.num_bands, 1 + PULSEG_MAX_BANDS + 2);
        /* v1.3: vendor tag */
        if (!pulseg__read4(f, &d->rf_definitions[i].stats.vendor, 1))
            return 0;
        if (do_swap)
            pulseg__swap4(&d->rf_definitions[i].stats.vendor);
    }

    /* gradient definitions */
    if (!pulseg__read4(f, &d->num_unique_grads, 1))
        return 0;
    if (do_swap)
        pulseg__swap4(&d->num_unique_grads);
    d->grad_definitions = (pulseg_grad_definition *)PULSEG_ALLOC(
        (size_t)d->num_unique_grads * sizeof(pulseg_grad_definition));
    if (!d->grad_definitions)
        return 0;
    for (i = 0; i < d->num_unique_grads; ++i)
    {
        pulseg_grad_definition *gd = &d->grad_definitions[i];
        if (!pulseg__read4(f, &gd->id, 1))
            return 0;
        if (!pulseg__read4(f, &gd->type, 1))
            return 0;
        if (!pulseg__read4(f, &gd->rise_time_or_unused, 1))
            return 0;
        if (!pulseg__read4(f, &gd->flat_time_or_unused, 1))
            return 0;
        if (!pulseg__read4(f, &gd->fall_time_or_num_uncompressed_samples, 1))
            return 0;
        if (!pulseg__read4(f, &gd->unused_or_time_shape_id, 1))
            return 0;
        if (!pulseg__read4(f, &gd->delay, 1))
            return 0;
        if (!pulseg__read4(f, &gd->num_shots, 1))
            return 0;
        if (do_swap)
            pulseg__swap4_array(&gd->id, 8);
        if (!pulseg__read4(f, gd->shot_shape_ids, PULSEG_MAX_GRAD_SHOTS))
            return 0;
        if (!pulseg__read4(f, gd->max_amplitude, PULSEG_MAX_GRAD_SHOTS))
            return 0;
        if (!pulseg__read4(f, gd->min_amplitude, PULSEG_MAX_GRAD_SHOTS))
            return 0;
        if (!pulseg__read4(f, gd->slew_rate, PULSEG_MAX_GRAD_SHOTS))
            return 0;
        if (!pulseg__read4(f, gd->energy, PULSEG_MAX_GRAD_SHOTS))
            return 0;
        if (!pulseg__read4(f, gd->first_value, PULSEG_MAX_GRAD_SHOTS))
            return 0;
        if (!pulseg__read4(f, gd->last_value, PULSEG_MAX_GRAD_SHOTS))
            return 0;
        if (do_swap)
            /* 7 contiguous MAX_GRAD_SHOTS arrays end the struct
             * (shot_shape_ids .. last_value); swapping 8 ran one array
             * past the allocation for the final element. */
            pulseg__swap4_array(gd->shot_shape_ids, 7 * PULSEG_MAX_GRAD_SHOTS);
    }

    /* ADC definitions */
    if (!pulseg__read4(f, &d->num_unique_adcs, 1))
        return 0;
    if (do_swap)
        pulseg__swap4(&d->num_unique_adcs);
    d->adc_definitions = (pulseg_adc_definition *)PULSEG_ALLOC(
        (size_t)d->num_unique_adcs * sizeof(pulseg_adc_definition));
    if (!d->adc_definitions)
        return 0;
    for (i = 0; i < d->num_unique_adcs; ++i)
    {
        if (!pulseg__read4(f, &d->adc_definitions[i].id, 4))
            return 0;
        if (do_swap)
            pulseg__swap4_array(&d->adc_definitions[i].id, 4);
    }

    /* freq_mod definitions (legacy: read and skip if count > 0) */
    if (!pulseg__read4(f, &d->num_freq_mod_defs, 1))
        return 0;
    if (do_swap)
        pulseg__swap4(&d->num_freq_mod_defs);
    d->num_freq_mod_defs = 0;
    d->freq_mod_definitions = NULL;

    /* rf_shim definitions */
    if (!pulseg__read4(f, &d->num_rf_shims, 1))
        return 0;
    if (do_swap)
        pulseg__swap4(&d->num_rf_shims);
    if (d->num_rf_shims > 0)
    {
        d->rf_shim_definitions = (pulseg_rf_shim_definition *)PULSEG_ALLOC(
            (size_t)d->num_rf_shims * sizeof(pulseg_rf_shim_definition));
        if (!d->rf_shim_definitions)
            return 0;
        for (i = 0; i < d->num_rf_shims; ++i)
        {
            pulseg_rf_shim_definition *rs = &d->rf_shim_definitions[i];
            memset(rs, 0, sizeof(*rs));
            if (!pulseg__read4(f, &rs->id, 1))
                return 0;
            if (!pulseg__read4(f, &rs->num_channels, 1))
                return 0;
            if (do_swap)
            {
                pulseg__swap4(&rs->id);
                pulseg__swap4(&rs->num_channels);
            }
            n = rs->num_channels;
            if (n > 0 && n <= PULSEG_MAX_RF_SHIM_CHANNELS)
            {
                if (!pulseg__read4(f, rs->magnitudes, n))
                    return 0;
                if (!pulseg__read4(f, rs->phases, n))
                    return 0;
                if (do_swap)
                {
                    pulseg__swap4_array(rs->magnitudes, n);
                    pulseg__swap4_array(rs->phases, n);
                }
            }
        }
    }

    /* rotations: read from the ROTATIONS section (read_rotations) */

    /* triggers */
    if (!pulseg__read4(f, &d->num_triggers, 1))
        return 0;
    if (do_swap)
        pulseg__swap4(&d->num_triggers);
    if (d->num_triggers > 0)
    {
        d->trigger_events = (pulseq_trigger_event *)PULSEG_ALLOC(
            (size_t)d->num_triggers * sizeof(pulseq_trigger_event));
        if (!d->trigger_events)
            return 0;
        for (i = 0; i < d->num_triggers; ++i)
        {
            if (!pulseg__read4(f, &ival, 1))
                return 0;
            if (do_swap)
                pulseg__swap4(&ival);
            d->trigger_events[i].type = (short)ival;
            if (!pulseg__read4(f, &ival, 1))
                return 0;
            if (do_swap)
                pulseg__swap4(&ival);
            d->trigger_events[i].duration = (long)ival;
            if (!pulseg__read4(f, &ival, 1))
                return 0;
            if (do_swap)
                pulseg__swap4(&ival);
            d->trigger_events[i].delay = (long)ival;
            if (!pulseg__read4(f, &d->trigger_events[i].trigger_type, 1))
                return 0;
            if (!pulseg__read4(f, &d->trigger_events[i].trigger_channel, 1))
                return 0;
            if (do_swap)
                pulseg__swap4_array(&d->trigger_events[i].trigger_type, 2);
        }
    }

    /* shapes: read from the SHAPES section (read_shapes) */

    /* TR descriptor */
    if (!pulseg__read4(f, &d->tr_descriptor.num_prep_blocks, 1))
        return 0;
    if (!pulseg__read4(f, &d->tr_descriptor.num_cooldown_blocks, 1))
        return 0;
    if (!pulseg__read4(f, &d->tr_descriptor.tr_size, 1))
        return 0;
    if (!pulseg__read4(f, &d->tr_descriptor.num_trs, 1))
        return 0;
    if (!pulseg__read4(f, &d->tr_descriptor.num_prep_trs, 1))
        return 0;
    if (!pulseg__read4(f, &d->tr_descriptor.degenerate_prep, 1))
        return 0;
    if (!pulseg__read4(f, &d->tr_descriptor.num_cooldown_trs, 1))
        return 0;
    if (!pulseg__read4(f, &d->tr_descriptor.degenerate_cooldown, 1))
        return 0;
    if (!pulseg__read4(f, &d->tr_descriptor.imaging_tr_start, 1))
        return 0;
    if (!pulseg__read4(f, &d->tr_descriptor.tr_duration_us, 1))
        return 0;
    if (do_swap)
        pulseg__swap4_array(&d->tr_descriptor.num_prep_blocks, 10);

    /* segment definitions */
    if (!pulseg__read4(f, &d->num_unique_segments, 1))
        return 0;
    if (do_swap)
        pulseg__swap4(&d->num_unique_segments);
    if (d->num_unique_segments > 0)
    {
        d->segment_definitions = (pulseg_virtual_segment *)PULSEG_ALLOC(
            (size_t)d->num_unique_segments * sizeof(pulseg_virtual_segment));
        if (!d->segment_definitions)
            return 0;
        for (i = 0; i < d->num_unique_segments; ++i)
        {
            pulseg_virtual_segment *seg = &d->segment_definitions[i];
            seg->unique_block_indices = NULL;
            seg->has_digitalout = NULL;
            seg->has_rotation = NULL;
            seg->norot_flag = NULL;
            seg->nopos_flag = NULL;
            seg->has_freq_mod = NULL;
            seg->has_adc = NULL;
            seg->is_dynamic_delay = NULL;
            seg->trigger_id = -1;
            seg->is_nav = 0;
            seg->timing.num_rf_anchors = 0;
            seg->timing.rf_anchors = NULL;
            seg->timing.num_adc_anchors = 0;
            seg->timing.adc_anchors = NULL;

            if (!pulseg__read4(f, &seg->start_block, 1))
                return 0;
            if (!pulseg__read4(f, &seg->num_blocks, 1))
                return 0;
            if (!pulseg__read4(f, &seg->max_energy_start_block, 1))
                return 0;
            if (do_swap)
                pulseg__swap4_array(&seg->start_block, 3);

            n = seg->num_blocks;
            if (n > 0)
            {
                seg->unique_block_indices = (int *)PULSEG_ALLOC((size_t)n * sizeof(int));
                seg->has_digitalout = (int *)PULSEG_ALLOC((size_t)n * sizeof(int));
                seg->has_rotation = (int *)PULSEG_ALLOC((size_t)n * sizeof(int));
                seg->norot_flag = (int *)PULSEG_ALLOC((size_t)n * sizeof(int));
                seg->nopos_flag = (int *)PULSEG_ALLOC((size_t)n * sizeof(int));
                seg->has_freq_mod = (int *)PULSEG_ALLOC((size_t)n * sizeof(int));
                seg->has_adc = (int *)PULSEG_ALLOC((size_t)n * sizeof(int));
                seg->is_dynamic_delay = (int *)PULSEG_ALLOC((size_t)n * sizeof(int));
                seg->initial_states = (pulseg_block_initial_state *)PULSEG_ALLOC(
                    (size_t)n * sizeof(pulseg_block_initial_state));
                if (!seg->unique_block_indices || !seg->has_digitalout || !seg->has_rotation ||
                    !seg->norot_flag || !seg->nopos_flag || !seg->has_freq_mod || !seg->has_adc ||
                    !seg->is_dynamic_delay || !seg->initial_states)
                    return 0;
                if (!pulseg__read4(f, seg->unique_block_indices, n))
                    return 0;
                if (!pulseg__read4(f, seg->has_digitalout, n))
                    return 0;
                if (!pulseg__read4(f, seg->has_rotation, n))
                    return 0;
                if (!pulseg__read4(f, seg->norot_flag, n))
                    return 0;
                if (!pulseg__read4(f, seg->nopos_flag, n))
                    return 0;
                if (!pulseg__read4(f, seg->has_freq_mod, n))
                    return 0;
                if (!pulseg__read4(f, seg->has_adc, n))
                    return 0;
                if (!pulseg__read4(f, seg->is_dynamic_delay, n))
                    return 0;
                if (!pulseg__read4(f, seg->initial_states, n * PULSEG_BLOCK_INITIAL_STATE_WORDS))
                    return 0;
                if (do_swap)
                {
                    pulseg__swap4_array(seg->unique_block_indices, n);
                    pulseg__swap4_array(seg->has_digitalout, n);
                    pulseg__swap4_array(seg->has_rotation, n);
                    pulseg__swap4_array(seg->norot_flag, n);
                    pulseg__swap4_array(seg->nopos_flag, n);
                    pulseg__swap4_array(seg->has_freq_mod, n);
                    pulseg__swap4_array(seg->has_adc, n);
                    pulseg__swap4_array(seg->is_dynamic_delay, n);
                    pulseg__swap4_array(seg->initial_states, n * PULSEG_BLOCK_INITIAL_STATE_WORDS);
                }
            }
            if (!pulseg__read4(f, &seg->trigger_id, 1))
                return 0;
            if (do_swap)
                pulseg__swap4(&seg->trigger_id);
            if (!pulseg__read4(f, &seg->is_nav, 1))
                return 0;
            if (do_swap)
                pulseg__swap4(&seg->is_nav);

            /* Segment timing anchors (k-space refs), serialized by
             * write_common. (num_*_anchors / *_anchors were zeroed above.) */
            if (!pulseg__read4(f, &seg->timing.num_rf_anchors, 1))
                return 0;
            if (do_swap)
                pulseg__swap4(&seg->timing.num_rf_anchors);
            if (seg->timing.num_rf_anchors > 0)
            {
                int nw = seg->timing.num_rf_anchors * (int)(sizeof(pulseg_segment_rf_anchor) / 4);
                seg->timing.rf_anchors = (pulseg_segment_rf_anchor *)PULSEG_ALLOC(
                    (size_t)seg->timing.num_rf_anchors * sizeof(pulseg_segment_rf_anchor));
                if (!seg->timing.rf_anchors)
                    return 0;
                if (!pulseg__read4(f, seg->timing.rf_anchors, nw))
                    return 0;
                if (do_swap)
                    pulseg__swap4_array(seg->timing.rf_anchors, nw);
            }
            if (!pulseg__read4(f, &seg->timing.num_adc_anchors, 1))
                return 0;
            if (do_swap)
                pulseg__swap4(&seg->timing.num_adc_anchors);
            if (seg->timing.num_adc_anchors > 0)
            {
                int nw = seg->timing.num_adc_anchors * (int)(sizeof(pulseg_segment_adc_anchor) / 4);
                seg->timing.adc_anchors = (pulseg_segment_adc_anchor *)PULSEG_ALLOC(
                    (size_t)seg->timing.num_adc_anchors * sizeof(pulseg_segment_adc_anchor));
                if (!seg->timing.adc_anchors)
                    return 0;
                if (!pulseg__read4(f, seg->timing.adc_anchors, nw))
                    return 0;
                if (do_swap)
                    pulseg__swap4_array(seg->timing.adc_anchors, nw);
            }
        }
    }

    /* segment table */
    if (!pulseg__read4(f, &d->segment_table.num_unique_segments, 1))
        return 0;
    if (!pulseg__read4(f, &d->segment_table.num_prep_segments, 1))
        return 0;
    if (do_swap)
        pulseg__swap4_array(&d->segment_table.num_unique_segments, 2);
    if (d->segment_table.num_prep_segments > 0)
    {
        d->segment_table.prep_segment_table =
            (int *)PULSEG_ALLOC((size_t)d->segment_table.num_prep_segments * sizeof(int));
        if (!d->segment_table.prep_segment_table)
            return 0;
        if (!pulseg__read4(
                f,
                d->segment_table.prep_segment_table,
                d->segment_table.num_prep_segments))
            return 0;
        if (do_swap)
            pulseg__swap4_array(
                d->segment_table.prep_segment_table,
                d->segment_table.num_prep_segments);
    }
    if (!pulseg__read4(f, &d->segment_table.num_main_segments, 1))
        return 0;
    if (do_swap)
        pulseg__swap4(&d->segment_table.num_main_segments);
    if (d->segment_table.num_main_segments > 0)
    {
        d->segment_table.main_segment_table =
            (int *)PULSEG_ALLOC((size_t)d->segment_table.num_main_segments * sizeof(int));
        if (!d->segment_table.main_segment_table)
            return 0;
        if (!pulseg__read4(
                f,
                d->segment_table.main_segment_table,
                d->segment_table.num_main_segments))
            return 0;
        if (do_swap)
            pulseg__swap4_array(
                d->segment_table.main_segment_table,
                d->segment_table.num_main_segments);
    }
    if (!pulseg__read4(f, &d->segment_table.num_cooldown_segments, 1))
        return 0;
    if (do_swap)
        pulseg__swap4(&d->segment_table.num_cooldown_segments);
    if (d->segment_table.num_cooldown_segments > 0)
    {
        d->segment_table.cooldown_segment_table =
            (int *)PULSEG_ALLOC((size_t)d->segment_table.num_cooldown_segments * sizeof(int));
        if (!d->segment_table.cooldown_segment_table)
            return 0;
        if (!pulseg__read4(
                f,
                d->segment_table.cooldown_segment_table,
                d->segment_table.num_cooldown_segments))
            return 0;
        if (do_swap)
            pulseg__swap4_array(
                d->segment_table.cooldown_segment_table,
                d->segment_table.num_cooldown_segments);
    }

    /* generic [DEFINITIONS] kv: read from the DEFINITIONS section
     * (read_definitions). Initialised empty here; COMMON no longer carries them. */
    d->num_definitions = 0;
    d->definitions = NULL;

    /* exec_stream + variable_grad_flags: read from the SCANLOOP section
     * (read_scanloop) */

    return 1;
}

/* ------ Deserialize the INSTANCES region into an existing descriptor ------ */
/* Mirrors write_instances. COMMON must have been read first (it allocates
 * and zeroes the descriptor); the scan path is the only PSD consumer. */

static int read_instances(FILE *f, pulseg_sequence_descriptor *d, int do_swap)
{
    int i;

    /* block table */
    if (!pulseg__read4(f, &d->num_blocks, 1))
        return 0;
    if (do_swap)
        pulseg__swap4(&d->num_blocks);
    d->block_table = (pulseg_block_table_element *)PULSEG_ALLOC(
        (size_t)d->num_blocks * sizeof(pulseg_block_table_element));
    if (!d->block_table)
        return 0;
    for (i = 0; i < d->num_blocks; ++i)
    {
        if (!pulseg__read4(f, &d->block_table[i].id, 17))
            return 0;
        if (do_swap)
            pulseg__swap4_array(&d->block_table[i].id, 17);
    }

    /* RF table */
    if (!pulseg__read4(f, &d->rf_table_size, 1))
        return 0;
    if (do_swap)
        pulseg__swap4(&d->rf_table_size);
    d->rf_table = (pulseg_rf_table_element *)PULSEG_ALLOC(
        (size_t)d->rf_table_size * sizeof(pulseg_rf_table_element));
    if (!d->rf_table)
        return 0;
    for (i = 0; i < d->rf_table_size; ++i)
    {
        if (!pulseg__read4(f, &d->rf_table[i].id, 4))
            return 0;
        if (do_swap)
            pulseg__swap4_array(&d->rf_table[i].id, 4);
        if (!pulseg__read4(f, &d->rf_table[i].rf_use, 1))
            return 0;
        if (do_swap)
            pulseg__swap4(&d->rf_table[i].rf_use);
    }

    /* gradient table */
    if (!pulseg__read4(f, &d->grad_table_size, 1))
        return 0;
    if (do_swap)
        pulseg__swap4(&d->grad_table_size);
    d->grad_table = (pulseg_grad_table_element *)PULSEG_ALLOC(
        (size_t)d->grad_table_size * sizeof(pulseg_grad_table_element));
    if (!d->grad_table)
        return 0;
    for (i = 0; i < d->grad_table_size; ++i)
    {
        if (!pulseg__read4(f, &d->grad_table[i].id, 3))
            return 0;
        if (do_swap)
            pulseg__swap4_array(&d->grad_table[i].id, 3);
    }

    /* ADC table */
    if (!pulseg__read4(f, &d->adc_table_size, 1))
        return 0;
    if (do_swap)
        pulseg__swap4(&d->adc_table_size);
    d->adc_table = (pulseg_adc_table_element *)PULSEG_ALLOC(
        (size_t)d->adc_table_size * sizeof(pulseg_adc_table_element));
    if (!d->adc_table)
        return 0;
    for (i = 0; i < d->adc_table_size; ++i)
    {
        if (!pulseg__read4(f, &d->adc_table[i].id, 3))
            return 0;
        if (do_swap)
            pulseg__swap4_array(&d->adc_table[i].id, 3);
    }

    /* label table.
     * These reads MUST honour do_swap like everything above: unswapped
     * counts on a big-endian reader (IPG) misalign the rest of the stream
     * and silently corrupt the heap while still returning success. */
    if (fread(&d->label_num_columns, sizeof(int), 1, f) != 1)
        return 0;
    if (fread(&d->label_num_entries, sizeof(int), 1, f) != 1)
        return 0;
    if (do_swap)
    {
        pulseg__swap4(&d->label_num_columns);
        pulseg__swap4(&d->label_num_entries);
    }
    if (d->label_num_entries > 0)
    {
        d->label_table = (int *)PULSEG_ALLOC(
            (size_t)d->label_num_entries * (size_t)d->label_num_columns * sizeof(int));
        if (!d->label_table)
            return 0;
        if (fread(
                d->label_table,
                sizeof(int),
                (size_t)d->label_num_entries * (size_t)d->label_num_columns,
                f) != (size_t)d->label_num_entries * (size_t)d->label_num_columns)
            return 0;
        if (do_swap)
            pulseg__swap4_array(d->label_table, d->label_num_entries * d->label_num_columns);
    }
    else
    {
        d->label_table = NULL;
    }
    if (fread(&d->label_limits, sizeof(pulseg_label_limits), 1, f) != 1)
        return 0;
    if (do_swap)
        pulseg__swap4_array(&d->label_limits, (int)(sizeof(pulseg_label_limits) / 4));

    return 1;
}

/* ------ Deserialize the DEFINITIONS region into an existing descriptor ------ */
/* Mirrors write_definitions: generic kv into d->definitions, verbatim. The
 * recon C++ reader sources FOV/Matrix/NavFOV/NavMatrix directly from these
 * kv entries (no separate geometry block — see write_definitions). */

static int read_definitions_cache(FILE *f, pulseg_sequence_descriptor *d, int do_swap)
{
    int i;

    d->num_definitions = 0;
    d->definitions = NULL;
    if (fread(&d->num_definitions, sizeof(int), 1, f) != 1)
        return 0;
    if (do_swap)
        pulseg__swap4(&d->num_definitions);
    if (d->num_definitions > 0)
    {
        d->definitions = (pulseq_definition *)PULSEG_ALLOC(
            (size_t)d->num_definitions * sizeof(pulseq_definition));
        if (!d->definitions)
            return 0;
        for (i = 0; i < d->num_definitions; ++i)
        {
            int name_len;
            d->definitions[i].value = NULL;
            d->definitions[i].value_size = 0;
            memset(d->definitions[i].name, 0, PULSEQ_DEFINITION_NAME_LENGTH);
            if (fread(&name_len, sizeof(int), 1, f) != 1)
                return 0;
            if (do_swap)
                pulseg__swap4(&name_len);
            if (name_len > 0 && name_len < PULSEQ_DEFINITION_NAME_LENGTH)
            {
                if (fread(d->definitions[i].name, 1, (size_t)name_len, f) != (size_t)name_len)
                    return 0;
                d->definitions[i].name[name_len] = '\0';
            }
            if (fread(&d->definitions[i].value_size, sizeof(int), 1, f) != 1)
                return 0;
            if (do_swap)
                pulseg__swap4(&d->definitions[i].value_size);
            if (d->definitions[i].value_size > 0)
            {
                int j;
                d->definitions[i].value =
                    (char **)PULSEG_ALLOC((size_t)d->definitions[i].value_size * sizeof(char *));
                if (!d->definitions[i].value)
                    return 0;
                for (j = 0; j < d->definitions[i].value_size; ++j)
                {
                    int vlen;
                    d->definitions[i].value[j] = NULL;
                    if (fread(&vlen, sizeof(int), 1, f) != 1)
                        return 0;
                    if (do_swap)
                        pulseg__swap4(&vlen);
                    if (vlen > 0)
                    {
                        d->definitions[i].value[j] = (char *)PULSEG_ALLOC((size_t)(vlen + 1));
                        if (!d->definitions[i].value[j])
                            return 0;
                        if (fread(d->definitions[i].value[j], 1, (size_t)vlen, f) != (size_t)vlen)
                            return 0;
                        d->definitions[i].value[j][vlen] = '\0';
                    }
                }
            }
        }
    }

    return 1;
}

/* ------ Deserialize the ROTATIONS region into an existing descriptor ------ */

static int read_rotations(FILE *f, pulseg_sequence_descriptor *d, int do_swap)
{
    int i;

    if (!pulseg__read4(f, &d->num_rotations, 1))
        return 0;
    if (do_swap)
        pulseg__swap4(&d->num_rotations);
    if (d->num_rotations > 0)
    {
        d->rotation_matrices =
            (float(*)[9])PULSEG_ALLOC((size_t)d->num_rotations * 9 * sizeof(float));
        if (!d->rotation_matrices)
            return 0;
        for (i = 0; i < d->num_rotations; ++i)
        {
            if (!pulseg__read4(f, d->rotation_matrices[i], 9))
                return 0;
            if (do_swap)
                pulseg__swap4_array(d->rotation_matrices[i], 9);
        }
    }

    return 1;
}

/* ------ Deserialize the SHAPES region into an existing descriptor ------ */

static int read_shapes(FILE *f, pulseg_sequence_descriptor *d, int do_swap)
{
    int i, n;

    if (!pulseg__read4(f, &d->num_shapes, 1))
        return 0;
    if (do_swap)
        pulseg__swap4(&d->num_shapes);
    if (d->num_shapes > 0)
    {
        d->shapes = (pulseq_shape *)PULSEG_ALLOC((size_t)d->num_shapes * sizeof(pulseq_shape));
        if (!d->shapes)
            return 0;
        for (i = 0; i < d->num_shapes; ++i)
        {
            d->shapes[i].samples = NULL;
            if (!pulseg__read4(f, &d->shapes[i].num_uncompressed_samples, 1))
                return 0;
            if (!pulseg__read4(f, &d->shapes[i].num_samples, 1))
                return 0;
            if (do_swap)
                pulseg__swap4_array(&d->shapes[i].num_uncompressed_samples, 2);
            n = d->shapes[i].num_samples;
            if (n > 0)
            {
                d->shapes[i].samples = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
                if (!d->shapes[i].samples)
                    return 0;
                if (!pulseg__read4(f, d->shapes[i].samples, n))
                    return 0;
                if (do_swap)
                    pulseg__swap4_array(d->shapes[i].samples, n);
            }
        }
    }

    return 1;
}

/* ------ Deserialize the SCANLOOP region into an existing descriptor ------ */

static int read_scanloop(FILE *f, pulseg_sequence_descriptor *d, int do_swap)
{
    /* scan table */
    if (fread(&d->exec_stream_len, sizeof(int), 1, f) != 1)
        return 0;
    if (do_swap)
        pulseg__swap4(&d->exec_stream_len);
    if (d->exec_stream_len > 0)
    {
        int i;

        /* Compact form only -- the per-position scratch arrays stay NULL on
         * a cache load, which is the whole point of the split. */
        if (!pulseg__read4(f, &d->num_exec_runs, 1))
            return 0;
        if (do_swap)
            pulseg__swap4(&d->num_exec_runs);
        if (d->num_exec_runs < 0)
            return 0;
        if (d->num_exec_runs > 0)
        {
            d->exec_runs = (pulseg_exec_run *)PULSEG_ALLOC(
                (size_t)d->num_exec_runs * sizeof(pulseg_exec_run));
            if (!d->exec_runs)
                return 0;
            for (i = 0; i < d->num_exec_runs; ++i)
            {
                if (!pulseg__read4(f, &d->exec_runs[i].emit_start, 1))
                    return 0;
                if (!pulseg__read4(f, &d->exec_runs[i].block_start, 1))
                    return 0;
                if (!pulseg__read4(f, &d->exec_runs[i].length, 1))
                    return 0;
                if (!pulseg__read4(f, &d->exec_runs[i].tr_id, 1))
                    return 0;
                if (!pulseg__read4(f, &d->exec_runs[i].avg_id, 1))
                    return 0;
                if (do_swap)
                    pulseg__swap4_array(&d->exec_runs[i].emit_start, 5);
            }
        }

        if (!pulseg__read4(f, &d->num_seg_runs, 1))
            return 0;
        if (!pulseg__read4(f, &d->seg_period, 1))
            return 0;
        if (do_swap)
        {
            pulseg__swap4(&d->num_seg_runs);
            pulseg__swap4(&d->seg_period);
        }
        if (d->num_seg_runs < 0)
            return 0;
        if (d->num_seg_runs > 0)
        {
            d->seg_run_start = (int *)PULSEG_ALLOC((size_t)(d->num_seg_runs + 1) * sizeof(int));
            d->seg_run_id = (int *)PULSEG_ALLOC((size_t)d->num_seg_runs * sizeof(int));
            if (!d->seg_run_start || !d->seg_run_id)
                return 0;
            if (!pulseg__read4(f, d->seg_run_start, d->num_seg_runs + 1))
                return 0;
            if (!pulseg__read4(f, d->seg_run_id, d->num_seg_runs))
                return 0;
            if (do_swap)
            {
                pulseg__swap4_array(d->seg_run_start, d->num_seg_runs + 1);
                pulseg__swap4_array(d->seg_run_id, d->num_seg_runs);
            }
        }

        if (!pulseg__read4(f, &d->tr_start_main_id, 1))
            return 0;
        if (!pulseg__read4(f, &d->tr_start_first, 1))
            return 0;
        if (do_swap)
        {
            pulseg__swap4(&d->tr_start_main_id);
            pulseg__swap4(&d->tr_start_first);
        }
        d->exec_run_hint = 0;
        d->seg_run_hint = 0;
    }

    /* variable grad flags */
    {
        int vgf_len;
        if (fread(&vgf_len, sizeof(int), 1, f) != 1)
            return 0;
        if (do_swap)
            pulseg__swap4(&vgf_len);
        if (vgf_len > 0)
        {
            d->variable_grad_flags = (int *)PULSEG_ALLOC((size_t)vgf_len * sizeof(int));
            if (!d->variable_grad_flags)
                return 0;
            if (fread(d->variable_grad_flags, sizeof(int), (size_t)vgf_len, f) != (size_t)vgf_len)
                return 0;
            if (do_swap)
                pulseg__swap4_array(d->variable_grad_flags, vgf_len);
        }
        else
        {
            d->variable_grad_flags = NULL;
        }
    }

    return 1;
}

/* ------ Write collection payload (header handled by caller) ------ */

static int write_common_payload(FILE *f, const pulseg_collection *coll)
{
    int i;

    /* collection scalars */
    if (!pulseg__write4(f, &coll->num_subsequences, 1))
    {
        return 0;
    }
    if (!pulseg__write4(f, &coll->num_repetitions, 1))
    {
        return 0;
    }
    if (!pulseg__write4(f, &coll->total_unique_segments, 1))
    {
        return 0;
    }
    if (!pulseg__write4(f, &coll->total_unique_adcs, 1))
    {
        return 0;
    }
    if (!pulseg__write4(f, &coll->total_blocks, 1))
    {
        return 0;
    }
    if (!pulseg__write4(f, &coll->total_readouts, 1))
    {
        return 0;
    }
    if (!pulseg__write4(f, &coll->total_duration_us, 1))
    {
        return 0;
    }

    /* subsequence info */
    for (i = 0; i < coll->num_subsequences; ++i)
    {
        if (!pulseg__write4(f, &coll->subsequence_info[i].sequence_index, 1))
        {
            return 0;
        }
        if (!pulseg__write4(f, &coll->subsequence_info[i].adc_id_offset, 1))
        {
            return 0;
        }
        if (!pulseg__write4(f, &coll->subsequence_info[i].segment_id_offset, 1))
        {
            return 0;
        }
        if (!pulseg__write4(f, &coll->subsequence_info[i].block_index_offset, 1))
        {
            return 0;
        }
    }

    /* per-subsequence COMMON descriptors */
    for (i = 0; i < coll->num_subsequences; ++i)
    {
        if (!write_common(f, &coll->descriptors[i]))
        {
            return 0;
        }
    }

    return 1;
}

/* ------ Augment-section payloads (ROTATIONS / SHAPES / SCANLOOP) ------ */
/* These carry no collection scalars; they write a num_subsequences token
 * (validated on read) then one per-descriptor region. They rely on COMMON
 * having been written/read first. */

typedef int (*desc_writer_fn)(FILE *, const pulseg_sequence_descriptor *);

static int write_augment_payload(FILE *f, const pulseg_collection *coll, desc_writer_fn wfn)
{
    int i;

    if (!pulseg__write4(f, &coll->num_subsequences, 1))
        return 0;
    for (i = 0; i < coll->num_subsequences; ++i)
        if (!wfn(f, &coll->descriptors[i]))
            return 0;

    return 1;
}

static int write_definitions_payload(FILE *f, const pulseg_collection *coll)
{
    return write_augment_payload(f, coll, write_definitions);
}

static int write_rotations_payload(FILE *f, const pulseg_collection *coll)
{
    return write_augment_payload(f, coll, write_rotations);
}

static int write_shapes_payload(FILE *f, const pulseg_collection *coll)
{
    return write_augment_payload(f, coll, write_shapes);
}

static int write_scanloop_payload(FILE *f, const pulseg_collection *coll)
{
    return write_augment_payload(f, coll, write_scanloop);
}

static int write_instances_payload(FILE *f, const pulseg_collection *coll)
{
    return write_augment_payload(f, coll, write_instances);
}

/* ------ Write full collection to sectioned cache ------ */

typedef int (*payload_writer_fn)(FILE *, const pulseg_collection *);

static int write_cache(const char *cache_path, const pulseg_collection *coll, int seq_file_size)
{
    FILE *f;
    int marker, vendor;
    int version_major, version_minor, version_revision;
    int num_sections, i;
    long entries_pos, end_pos;
    pulseg_cache_section_entry entries[6];
    static const int section_ids[6] = {
        PULSEG_CACHE_SECTION_COMMON,
        PULSEG_CACHE_SECTION_INSTANCES,
        PULSEG_CACHE_SECTION_ROTATIONS,
        PULSEG_CACHE_SECTION_SHAPES,
        PULSEG_CACHE_SECTION_SCANLOOP,
        PULSEG_CACHE_SECTION_DEFINITIONS};
    static const payload_writer_fn writers[6] = {
        write_common_payload,
        write_instances_payload,
        write_rotations_payload,
        write_shapes_payload,
        write_scanloop_payload,
        write_definitions_payload};

    f = fopen(cache_path, "wb");
    if (!f)
        return 0;

    marker = PULSEG_CACHE_ENDIAN_MARKER;
    vendor = PULSEG_VENDOR;
    version_major = PULSEG_CACHE_VERSION_MAJOR;
    version_minor = PULSEG_CACHE_VERSION_MINOR;
    version_revision = PULSEG_CACHE_VERSION_REVISION;
    num_sections = 6;

    if (!pulseg__write4(f, &marker, 1))
    {
        fclose(f);
        return 0;
    }
    if (!pulseg__write4(f, &version_major, 1))
    {
        fclose(f);
        return 0;
    }
    if (!pulseg__write4(f, &version_minor, 1))
    {
        fclose(f);
        return 0;
    }
    if (!pulseg__write4(f, &version_revision, 1))
    {
        fclose(f);
        return 0;
    }
    if (!pulseg__write4(f, &vendor, 1))
    {
        fclose(f);
        return 0;
    }
    if (!pulseg__write4(f, &seq_file_size, 1))
    {
        fclose(f);
        return 0;
    }
    if (!pulseg__write4(f, &num_sections, 1))
    {
        fclose(f);
        return 0;
    }

    entries_pos = ftell(f);
    if (entries_pos < 0)
    {
        fclose(f);
        return 0;
    }

    /* Reserve slots for the maximum possible section count, not just the
     * num_sections (6) written by this function -- the trajectory/freqmod/
     * seqdesc append passes insert their own entries into this same table
     * afterward and must not overflow into the COMMON payload that follows
     * it (see PULSEG_CACHE_MAX_SECTIONS). */
    for (i = 0; i < PULSEG_CACHE_MAX_SECTIONS * 3; ++i)
    {
        int zero = 0;
        if (!pulseg__write4(f, &zero, 1))
        {
            fclose(f);
            return 0;
        }
    }

    /* Each section carries its own distinct payload. */
    for (i = 0; i < num_sections; ++i)
    {
        long start, stop;
        start = ftell(f);
        if (start < 0)
        {
            fclose(f);
            return 0;
        }
        if (!writers[i](f, coll))
        {
            fclose(f);
            return 0;
        }
        stop = ftell(f);
        if (stop < 0)
        {
            fclose(f);
            return 0;
        }
        entries[i].section_id = section_ids[i];
        entries[i].offset = (int)start;
        entries[i].size = (int)(stop - start);
    }

    end_pos = ftell(f);
    if (end_pos < 0)
    {
        fclose(f);
        return 0;
    }
    if (fseek(f, entries_pos, SEEK_SET) != 0)
    {
        fclose(f);
        return 0;
    }

    for (i = 0; i < num_sections; ++i)
    {
        if (!pulseg__write4(f, &entries[i].section_id, 1))
        {
            fclose(f);
            return 0;
        }
        if (!pulseg__write4(f, &entries[i].offset, 1))
        {
            fclose(f);
            return 0;
        }
        if (!pulseg__write4(f, &entries[i].size, 1))
        {
            fclose(f);
            return 0;
        }
    }

    if (fseek(f, end_pos, SEEK_SET) != 0)
    {
        fclose(f);
        return 0;
    }

    fclose(f);
    return 1;
}

/* ------ Read collection payload from sectioned cache ------ */

static int read_common_payload(FILE *f, pulseg_collection *coll, int do_swap)
{
    int i;

    /* collection scalars */
    if (!pulseg__read4(f, &coll->num_subsequences, 1))
    {
        return 0;
    }
    if (!pulseg__read4(f, &coll->num_repetitions, 1))
    {
        return 0;
    }
    if (!pulseg__read4(f, &coll->total_unique_segments, 1))
    {
        return 0;
    }
    if (!pulseg__read4(f, &coll->total_unique_adcs, 1))
    {
        return 0;
    }
    if (!pulseg__read4(f, &coll->total_blocks, 1))
    {
        return 0;
    }
    if (!pulseg__read4(f, &coll->total_readouts, 1))
    {
        return 0;
    }
    if (!pulseg__read4(f, &coll->total_duration_us, 1))
    {
        return 0;
    }
    if (do_swap)
    {
        pulseg__swap4(&coll->num_subsequences);
        pulseg__swap4(&coll->num_repetitions);
        pulseg__swap4(&coll->total_unique_segments);
        pulseg__swap4(&coll->total_unique_adcs);
        pulseg__swap4(&coll->total_blocks);
        pulseg__swap4(&coll->total_readouts);
        pulseg__swap4(&coll->total_duration_us);
    }

    /* allocate arrays */
    coll->descriptors = (pulseg_sequence_descriptor *)PULSEG_ALLOC(
        (size_t)coll->num_subsequences * sizeof(pulseg_sequence_descriptor));
    coll->subsequence_info = (pulseg_subsequence_info *)PULSEG_ALLOC(
        (size_t)coll->num_subsequences * sizeof(pulseg_subsequence_info));
    if (!coll->descriptors || !coll->subsequence_info)
    {
        if (coll->descriptors)
            PULSEG_FREE(coll->descriptors);
        if (coll->subsequence_info)
            PULSEG_FREE(coll->subsequence_info);
        coll->descriptors = NULL;
        coll->subsequence_info = NULL;
        return 0;
    }

    /* subsequence info */
    for (i = 0; i < coll->num_subsequences; ++i)
    {
        if (!pulseg__read4(f, &coll->subsequence_info[i].sequence_index, 4))
        {
            return 0;
        }
        if (do_swap)
            pulseg__swap4_array(&coll->subsequence_info[i].sequence_index, 4);
    }

    /* per-subsequence COMMON descriptors */
    for (i = 0; i < coll->num_subsequences; ++i)
    {
        if (!read_common(f, &coll->descriptors[i], do_swap))
        {
            /* clean up already-read descriptors */
            int j;
            for (j = 0; j < i; ++j)
                pulseg_sequence_descriptor_free(&coll->descriptors[j]);
            PULSEG_FREE(coll->descriptors);
            PULSEG_FREE(coll->subsequence_info);
            coll->descriptors = NULL;
            coll->subsequence_info = NULL;
            coll->num_subsequences = 0;
            return 0;
        }
    }

    /* init cursor */
    memset(&coll->block_cursor, 0, sizeof(coll->block_cursor));
    coll->block_cursor.exec_stream_position = -1;

    return 1;
}

/* ------ Augment-section readers (ROTATIONS / SHAPES / SCANLOOP) ------ */
/* Read a num_subsequences token (validated against COMMON) then one
 * per-descriptor region into the already-allocated descriptors. COMMON must
 * have been read into coll first. */

typedef int (*desc_reader_fn)(FILE *, pulseg_sequence_descriptor *, int);

static int read_augment_payload(FILE *f, pulseg_collection *coll, int do_swap, desc_reader_fn rfn)
{
    int i, ns;

    if (!coll->descriptors)
        return 0; /* COMMON must be read first */
    if (!pulseg__read4(f, &ns, 1))
        return 0;
    if (do_swap)
        pulseg__swap4(&ns);
    if (ns != coll->num_subsequences)
        return 0;
    for (i = 0; i < coll->num_subsequences; ++i)
        if (!rfn(f, &coll->descriptors[i], do_swap))
            return 0;

    return 1;
}

static int read_definitions_payload(FILE *f, pulseg_collection *coll, int do_swap)
{
    return read_augment_payload(f, coll, do_swap, read_definitions_cache);
}

static int read_rotations_payload(FILE *f, pulseg_collection *coll, int do_swap)
{
    return read_augment_payload(f, coll, do_swap, read_rotations);
}

static int read_shapes_payload(FILE *f, pulseg_collection *coll, int do_swap)
{
    return read_augment_payload(f, coll, do_swap, read_shapes);
}

static int read_scanloop_payload(FILE *f, pulseg_collection *coll, int do_swap)
{
    return read_augment_payload(f, coll, do_swap, read_scanloop);
}

static int read_instances_payload(FILE *f, pulseg_collection *coll, int do_swap)
{
    return read_augment_payload(f, coll, do_swap, read_instances);
}

/* ------ Read a set of sections from a sectioned cache ------ */
/* readers[k] is invoked for the section section_ids[k], in the order given,
 * after seeking to that section's payload. COMMON must be listed first when
 * any augment section is requested. */

typedef int (*payload_reader_fn)(FILE *, pulseg_collection *, int);

static int read_sections(
    const char *cache_path,
    pulseg_collection *coll,
    int expected_seq_file_size,
    int enforce_source_size,
    const int *section_ids,
    const payload_reader_fn *readers,
    int n_req)
{
    FILE *f;
    int marker, vendor, stored_size, num_sections;
    int version_major, version_minor, version_revision;
    int do_swap, i, k;
    pulseg_cache_section_entry entries[16];

    f = fopen(cache_path, "rb");
    if (!f)
        return 0;

    if (!pulseg__read4(f, &marker, 1))
    {
        fclose(f);
        return 0;
    }

    do_swap = 0;
    if (marker != PULSEG_CACHE_ENDIAN_MARKER)
    {
        pulseg__swap4(&marker);
        if (marker != PULSEG_CACHE_ENDIAN_MARKER)
        {
            fclose(f);
            return 0;
        }
        do_swap = 1;
    }

    if (!pulseg__read4(f, &version_major, 1))
    {
        fclose(f);
        return 0;
    }
    if (!pulseg__read4(f, &version_minor, 1))
    {
        fclose(f);
        return 0;
    }
    if (!pulseg__read4(f, &version_revision, 1))
    {
        fclose(f);
        return 0;
    }
    if (do_swap)
    {
        pulseg__swap4(&version_major);
        pulseg__swap4(&version_minor);
        pulseg__swap4(&version_revision);
    }
    if (version_major != PULSEG_CACHE_VERSION_MAJOR ||
        version_minor != PULSEG_CACHE_VERSION_MINOR ||
        version_revision != PULSEG_CACHE_VERSION_REVISION)
    {
        fclose(f);
        return 0;
    }

    if (!pulseg__read4(f, &vendor, 1))
    {
        fclose(f);
        return 0;
    }
    if (do_swap)
        pulseg__swap4(&vendor);
    if (vendor != PULSEG_VENDOR)
    {
        fclose(f);
        return 0;
    }

    if (!pulseg__read4(f, &stored_size, 1))
    {
        fclose(f);
        return 0;
    }
    if (do_swap)
        pulseg__swap4(&stored_size);
    if (enforce_source_size && stored_size != expected_seq_file_size)
    {
        fclose(f);
        return 0;
    }

    if (!pulseg__read4(f, &num_sections, 1))
    {
        fclose(f);
        return 0;
    }
    if (do_swap)
        pulseg__swap4(&num_sections);
    if (num_sections <= 0 || num_sections > 16)
    {
        fclose(f);
        return 0;
    }

    for (i = 0; i < num_sections; ++i)
    {
        if (!pulseg__read4(f, &entries[i].section_id, 1))
        {
            fclose(f);
            return 0;
        }
        if (!pulseg__read4(f, &entries[i].offset, 1))
        {
            fclose(f);
            return 0;
        }
        if (!pulseg__read4(f, &entries[i].size, 1))
        {
            fclose(f);
            return 0;
        }
        if (do_swap)
        {
            pulseg__swap4(&entries[i].section_id);
            pulseg__swap4(&entries[i].offset);
            pulseg__swap4(&entries[i].size);
        }
    }

    for (k = 0; k < n_req; ++k)
    {
        int found = 0;
        pulseg_cache_section_entry section;
        memset(&section, 0, sizeof(section));
        for (i = 0; i < num_sections; ++i)
        {
            if (entries[i].section_id == section_ids[k])
            {
                section = entries[i];
                found = 1;
            }
        }
        if (!found || section.offset <= 0 || section.size <= 0)
        {
            fclose(f);
            return 0;
        }
        if (fseek(f, (long)section.offset, SEEK_SET) != 0)
        {
            fclose(f);
            return 0;
        }
        if (!readers[k](f, coll, do_swap))
        {
            fclose(f);
            return 0;
        }
    }

    fclose(f);
    return 1;
}

/* ------ Read the full descriptor (all PSD-internal sections) ------ */

static int read_full_cache(
    const char *cache_path,
    pulseg_collection *coll,
    int expected_seq_file_size,
    int enforce_source_size)
{
    static const int ids[6] = {
        PULSEG_CACHE_SECTION_COMMON,
        PULSEG_CACHE_SECTION_INSTANCES,
        PULSEG_CACHE_SECTION_ROTATIONS,
        PULSEG_CACHE_SECTION_SHAPES,
        PULSEG_CACHE_SECTION_SCANLOOP,
        PULSEG_CACHE_SECTION_DEFINITIONS};
    static const payload_reader_fn readers[6] = {
        read_common_payload,
        read_instances_payload,
        read_rotations_payload,
        read_shapes_payload,
        read_scanloop_payload,
        read_definitions_payload};
    int ok = read_sections(
        cache_path,
        coll,
        expected_seq_file_size,
        enforce_source_size,
        ids,
        readers,
        6);
    /* Every PSD-internal section is present here, so the cross-subsequence
     * segment dedup remap can be rebuilt deterministically (identical to the
     * convert-time map). */
    if (ok)
        pulseg__build_segment_remap(coll);
    return ok;
}

/* ================================================================== */
/*  Public wrappers (called from pulseg_core.c)                    */
/* ================================================================== */

int pulseg__write_cache(pulseg_collection *coll, const char *seq_path, const pulseg_opts *opts)
{
    char *cache_path;
    long sz;
    int ok;

    if (!coll || !seq_path || !opts)
        return 0;

    cache_path = pulseg__make_cache_path(seq_path, opts->cache_ext);
    if (!cache_path)
        return 0;

    sz = get_file_size(seq_path);
    if (sz < 0)
    {
        PULSEG_FREE(cache_path);
        return 0;
    }

    /* Write the four base sections (COMMON/ROTATIONS/SHAPES/SCANLOOP). */
    ok = write_cache(cache_path, coll, (int)sz);
    PULSEG_FREE(cache_path);
    if (!ok)
        return 0;

    /* Append the remaining static sections. The whole cache is a pure
     * function of the loaded collection (shift/rotation independent), so
     * it is produced here in one shot at load time. These are best-effort:
     * each consumer treats its section as optional, so a failure does not
     * invalidate the base cache and not every sequence has every section.
     * Each callee reads its own cache_ext from coll->descriptors[0]
     * (set at dedup time from this same opts->cache_ext). */
    (void)pulseg__save_trajectory_cache_section(coll, seq_path);
    (void)pulseg__save_freq_mod_cache_section(coll, seq_path);
    (void)pulseg__save_seqdesc_cache_section(coll, seq_path);
    (void)pulseg_write_vendor_cache_section(coll, seq_path, opts);

    return ok;
}

int pulseg__try_read_cache(pulseg_collection *coll, const char *seq_path, const char *cache_ext)
{
    char *cache_path;
    long sz;
    int ok;

    if (!coll || !seq_path)
        return 0;

    cache_path = pulseg__make_cache_path(seq_path, cache_ext);
    if (!cache_path)
        return 0;

    sz = get_file_size(seq_path);
    if (sz < 0)
    {
        PULSEG_FREE(cache_path);
        return 0;
    }

    ok = read_full_cache(cache_path, coll, (int)sz, 1);
    PULSEG_FREE(cache_path);
    return ok;
}

/* ================================================================== */
/*  Public API: explicit-path cache save / load                       */
/* ================================================================== */

int pulseg_save_cache(const pulseg_collection *coll, const char *path, int source_size)
{
    if (!coll || !path)
        return PULSEG_ERR_NULL_POINTER;
    if (source_size <= 0)
        return PULSEG_ERR_INVALID_ARGUMENT;
    return write_cache(path, coll, source_size) ? PULSEG_SUCCESS : PULSEG_ERR_FILE_READ_FAILED;
}

int pulseg_load_cache(pulseg_collection *coll, const char *path, int source_size)
{
    if (!coll || !path)
        return PULSEG_ERR_NULL_POINTER;
    if (source_size <= 0)
        return PULSEG_ERR_INVALID_ARGUMENT;
    return read_full_cache(path, coll, source_size, 1) ? PULSEG_SUCCESS
                                                       : PULSEG_ERR_FILE_READ_FAILED;
}

static int load_cache_from_seq_path(
    pulseg_collection **out_coll,
    const char *seq_path,
    const int *section_ids,
    const payload_reader_fn *readers,
    int n_req,
    int enforce_source_size)
{
    pulseg_collection *coll;
    char *cache_path;
    long source_size;
    int ok;

    if (!out_coll || !seq_path)
        return PULSEG_ERR_NULL_POINTER;

    *out_coll = NULL;

    coll = (pulseg_collection *)PULSEG_ALLOC(sizeof(pulseg_collection));
    if (!coll)
        return PULSEG_ERR_ALLOC_FAILED;
    memset(coll, 0, sizeof(*coll));

    cache_path = pulseg__make_cache_path(seq_path, NULL);
    if (!cache_path)
    {
        PULSEG_FREE(coll);
        return PULSEG_ERR_ALLOC_FAILED;
    }

    source_size = get_file_size(seq_path);
    if (source_size < 0 && enforce_source_size)
    {
        PULSEG_FREE(cache_path);
        PULSEG_FREE(coll);
        return PULSEG_ERR_FILE_NOT_FOUND;
    }

    ok = read_sections(
        cache_path,
        coll,
        source_size < 0 ? 0 : (int)source_size,
        enforce_source_size,
        section_ids,
        readers,
        n_req);
    PULSEG_FREE(cache_path);
    if (!ok)
    {
        pulseg_collection_free(coll);
        return PULSEG_ERR_FILE_READ_FAILED;
    }

    *out_coll = coll;
    return PULSEG_SUCCESS;
}

/* Pulsegen: COMMON + SHAPES. Neither INSTANCES nor SCANLOOP is loaded — this
 * pass builds one hardware image per segment DEFINITION, and every per-position
 * quantity it needs (the representative instance's block definition, RF
 * amplitude, gradient definition/shot/amplitude, digital-output event) is
 * frozen into the segment definitions' initial-state records at parse time
 * (pulseg_structure.c step 11e). SHAPES is required both for the waveforms
 * themselves and for pulseg__build_segment_remap()'s content comparison. */
int pulseg_load_geninstructions_cache(pulseg_collection **out_coll, const char *seq_path)
{
    static const int ids[2] = {PULSEG_CACHE_SECTION_COMMON, PULSEG_CACHE_SECTION_SHAPES};
    static const payload_reader_fn readers[2] = {read_common_payload, read_shapes_payload};
    int rc = load_cache_from_seq_path(out_coll, seq_path, ids, readers, 2, 0);
    /* Rebuild the cross-subsequence segment dedup remap so the global segment
     * ids the pulsegen loop builds match those the scan cursor emits. */
    if (rc == PULSEG_SUCCESS && out_coll && *out_coll)
        pulseg__build_segment_remap(*out_coll);
    return rc;
}

/* Scan: COMMON + INSTANCES + ROTATIONS + SHAPES + SCANLOOP.  INSTANCES and
 * SCANLOOP carry the per-instance event tables the cursor replays; SHAPES is
 * loaded (even though scan never plays raw waveforms) so
 * pulseg__build_segment_remap() recomputes the SAME cross-subsequence segment
 * deduplication the pulsegen path computed — otherwise the cursor's global
 * segment ids would not match the built buffers. */
int pulseg_load_scanloop_cache(pulseg_collection **out_coll, const char *seq_path)
{
    static const int ids[5] = {
        PULSEG_CACHE_SECTION_COMMON,
        PULSEG_CACHE_SECTION_INSTANCES,
        PULSEG_CACHE_SECTION_ROTATIONS,
        PULSEG_CACHE_SECTION_SHAPES,
        PULSEG_CACHE_SECTION_SCANLOOP};
    static const payload_reader_fn readers[5] = {
        read_common_payload,
        read_instances_payload,
        read_rotations_payload,
        read_shapes_payload,
        read_scanloop_payload};
    int rc = load_cache_from_seq_path(out_coll, seq_path, ids, readers, 5, 0);
    if (rc == PULSEG_SUCCESS && out_coll && *out_coll)
        pulseg__build_segment_remap(*out_coll);
    return rc;
}

int pulseg_clear_cache(const char *seq_path)
{
    char *cache_path;
    int rc;

    if (!seq_path)
        return PULSEG_ERR_NULL_POINTER;

    cache_path = pulseg__make_cache_path(seq_path, NULL);
    if (!cache_path)
        return PULSEG_ERR_ALLOC_FAILED;

    rc = remove(cache_path);
    PULSEG_FREE(cache_path);

    if (rc == 0 || errno == ENOENT)
        return PULSEG_SUCCESS;
    return PULSEG_ERR_FILE_READ_FAILED;
}

/* ================================================================== */
/*  Freq-mod unified cache (FREQMOD section of the binary cache)                  */
/* ================================================================== */

int pulseg__save_freq_mod_section(const pulseg_collection *coll, const char *seq_path)
{
    char *cache_path;
    FILE *f;
    int marker, num_sections;
    int version_major, version_minor, version_revision, vendor, stored_size;
    int do_swap;
    long entries_pos, data_start, data_end, hdr_ns_pos;
    int i, found_idx;
    pulseg_cache_section_entry entries[16];

    if (!coll || !seq_path)
        return PULSEG_ERR_NULL_POINTER;
    if (!coll->freq_mod)
        return PULSEG_ERR_NULL_POINTER;

    cache_path = pulseg__make_cache_path(
        seq_path,
        coll->num_subsequences > 0 ? coll->descriptors[0].cache_ext : NULL);
    if (!cache_path)
        return PULSEG_ERR_ALLOC_FAILED;

    f = fopen(cache_path, "r+b");
    if (!f)
    {
        PULSEG_FREE(cache_path);
        return PULSEG_ERR_FILE_READ_FAILED;
    }

    /* Read header */
    if (!pulseg__read4(f, &marker, 1))
        goto fm_write_fail;
    do_swap = 0;
    if (marker != PULSEG_CACHE_ENDIAN_MARKER)
    {
        pulseg__swap4(&marker);
        if (marker != PULSEG_CACHE_ENDIAN_MARKER)
            goto fm_write_fail;
        do_swap = 1;
    }
    if (!pulseg__read4(f, &version_major, 1))
        goto fm_write_fail;
    if (!pulseg__read4(f, &version_minor, 1))
        goto fm_write_fail;
    if (!pulseg__read4(f, &version_revision, 1))
        goto fm_write_fail;
    if (!pulseg__read4(f, &vendor, 1))
        goto fm_write_fail;
    if (!pulseg__read4(f, &stored_size, 1))
        goto fm_write_fail;
    hdr_ns_pos = ftell(f); /* position of num_sections in file */
    if (!pulseg__read4(f, &num_sections, 1))
        goto fm_write_fail;
    if (do_swap)
    {
        pulseg__swap4(&version_major);
        pulseg__swap4(&version_minor);
        pulseg__swap4(&version_revision);
        pulseg__swap4(&vendor);
        pulseg__swap4(&stored_size);
        pulseg__swap4(&num_sections);
    }
    if (num_sections <= 0 || num_sections > 15)
        goto fm_write_fail;

    entries_pos = ftell(f);
    if (entries_pos < 0)
        goto fm_write_fail;

    for (i = 0; i < num_sections; ++i)
    {
        if (!pulseg__read4(f, &entries[i].section_id, 1))
            goto fm_write_fail;
        if (!pulseg__read4(f, &entries[i].offset, 1))
            goto fm_write_fail;
        if (!pulseg__read4(f, &entries[i].size, 1))
            goto fm_write_fail;
        if (do_swap)
        {
            pulseg__swap4(&entries[i].section_id);
            pulseg__swap4(&entries[i].offset);
            pulseg__swap4(&entries[i].size);
        }
    }

    /* Check if the freq-mod section already exists */
    found_idx = -1;
    for (i = 0; i < num_sections; ++i)
    {
        if (entries[i].section_id == PULSEG_CACHE_SECTION_FREQMOD)
        {
            found_idx = i;
            break;
        }
    }

    if (found_idx < 0)
    {
        found_idx = num_sections;
        entries[found_idx].section_id = PULSEG_CACHE_SECTION_FREQMOD;
        num_sections++;
    }

    /* Seek to end, write freq-mod data */
    fseek(f, 0, SEEK_END);
    data_start = ftell(f);
    if (data_start < 0)
        goto fm_write_fail;

    if (pulseg__freq_mod_collection_write_f(coll->freq_mod, f) != PULSEG_SUCCESS)
        goto fm_write_fail;

    data_end = ftell(f);
    if (data_end < 0)
        goto fm_write_fail;

    entries[found_idx].offset = (int)data_start;
    entries[found_idx].size = (int)(data_end - data_start);

    /* Patch num_sections */
    if (fseek(f, hdr_ns_pos, SEEK_SET) != 0)
        goto fm_write_fail;
    if (!pulseg__write4(f, &num_sections, 1))
        goto fm_write_fail;

    /* Rewrite all section entries (at the same position, but extend if needed) */
    if (fseek(f, entries_pos, SEEK_SET) != 0)
        goto fm_write_fail;
    for (i = 0; i < num_sections; ++i)
    {
        if (!pulseg__write4(f, &entries[i].section_id, 1))
            goto fm_write_fail;
        if (!pulseg__write4(f, &entries[i].offset, 1))
            goto fm_write_fail;
        if (!pulseg__write4(f, &entries[i].size, 1))
            goto fm_write_fail;
    }

    fclose(f);
    PULSEG_FREE(cache_path);
    return PULSEG_SUCCESS;

fm_write_fail:
    fclose(f);
    PULSEG_FREE(cache_path);
    return PULSEG_ERR_FILE_READ_FAILED;
}

/* ================================================================== */
/*  Vendor extension section (opaque, optional)                  */
/* ================================================================== */

int pulseg_write_vendor_cache_section(
    const pulseg_collection *coll,
    const char *seq_path,
    const pulseg_opts *opts)
{
    char *cache_path;
    FILE *f;
    int marker, num_sections;
    int version_major, version_minor, version_revision, vendor, stored_size;
    int do_swap;
    long entries_pos, data_start, data_end, hdr_ns_pos;
    int i, found_idx;
    pulseg_cache_section_entry entries[16];
    unsigned char *blob;
    int blob_len;
    int rc;

    if (!coll || !seq_path || !opts)
        return PULSEG_ERR_NULL_POINTER;
    if (!opts->vendor_section_write_fn)
        return PULSEG_SUCCESS; /* nothing to do -- not an error */

    blob = NULL;
    blob_len = 0;
    rc = opts->vendor_section_write_fn(opts->vendor_section_ctx, &blob, &blob_len);
    if (PULSEG_FAILED(rc))
        return rc;
    if (!blob || blob_len <= 0)
    {
        if (blob)
            PULSEG_FREE(blob);
        return PULSEG_SUCCESS; /* callback opted out this time */
    }

    cache_path = pulseg__make_cache_path(
        seq_path,
        coll->num_subsequences > 0 ? coll->descriptors[0].cache_ext : NULL);
    if (!cache_path)
    {
        PULSEG_FREE(blob);
        return PULSEG_ERR_ALLOC_FAILED;
    }

    f = fopen(cache_path, "r+b");
    if (!f)
    {
        PULSEG_FREE(cache_path);
        PULSEG_FREE(blob);
        return PULSEG_ERR_FILE_READ_FAILED;
    }

    if (!pulseg__read4(f, &marker, 1))
        goto vs_write_fail;
    do_swap = 0;
    if (marker != PULSEG_CACHE_ENDIAN_MARKER)
    {
        pulseg__swap4(&marker);
        if (marker != PULSEG_CACHE_ENDIAN_MARKER)
            goto vs_write_fail;
        do_swap = 1;
    }
    if (!pulseg__read4(f, &version_major, 1))
        goto vs_write_fail;
    if (!pulseg__read4(f, &version_minor, 1))
        goto vs_write_fail;
    if (!pulseg__read4(f, &version_revision, 1))
        goto vs_write_fail;
    if (!pulseg__read4(f, &vendor, 1))
        goto vs_write_fail;
    if (!pulseg__read4(f, &stored_size, 1))
        goto vs_write_fail;
    hdr_ns_pos = ftell(f);
    if (!pulseg__read4(f, &num_sections, 1))
        goto vs_write_fail;
    if (do_swap)
    {
        pulseg__swap4(&version_major);
        pulseg__swap4(&version_minor);
        pulseg__swap4(&version_revision);
        pulseg__swap4(&vendor);
        pulseg__swap4(&stored_size);
        pulseg__swap4(&num_sections);
    }
    if (num_sections <= 0 || num_sections > 15)
        goto vs_write_fail;

    entries_pos = ftell(f);
    if (entries_pos < 0)
        goto vs_write_fail;

    for (i = 0; i < num_sections; ++i)
    {
        if (!pulseg__read4(f, &entries[i].section_id, 1))
            goto vs_write_fail;
        if (!pulseg__read4(f, &entries[i].offset, 1))
            goto vs_write_fail;
        if (!pulseg__read4(f, &entries[i].size, 1))
            goto vs_write_fail;
        if (do_swap)
        {
            pulseg__swap4(&entries[i].section_id);
            pulseg__swap4(&entries[i].offset);
            pulseg__swap4(&entries[i].size);
        }
    }

    found_idx = -1;
    for (i = 0; i < num_sections; ++i)
    {
        if (entries[i].section_id == PULSEG_CACHE_SECTION_VENDOR)
        {
            found_idx = i;
            break;
        }
    }
    if (found_idx < 0)
    {
        found_idx = num_sections;
        entries[found_idx].section_id = PULSEG_CACHE_SECTION_VENDOR;
        num_sections++;
    }

    fseek(f, 0, SEEK_END);
    data_start = ftell(f);
    if (data_start < 0)
        goto vs_write_fail;

    if (!pulseg__write4(f, &blob_len, 1))
        goto vs_write_fail;
    if (blob_len > 0 && (int)fwrite(blob, 1, (size_t)blob_len, f) != blob_len)
        goto vs_write_fail;

    data_end = ftell(f);
    if (data_end < 0)
        goto vs_write_fail;

    entries[found_idx].offset = (int)data_start;
    entries[found_idx].size = (int)(data_end - data_start);

    if (fseek(f, hdr_ns_pos, SEEK_SET) != 0)
        goto vs_write_fail;
    if (!pulseg__write4(f, &num_sections, 1))
        goto vs_write_fail;

    if (fseek(f, entries_pos, SEEK_SET) != 0)
        goto vs_write_fail;
    for (i = 0; i < num_sections; ++i)
    {
        if (!pulseg__write4(f, &entries[i].section_id, 1))
            goto vs_write_fail;
        if (!pulseg__write4(f, &entries[i].offset, 1))
            goto vs_write_fail;
        if (!pulseg__write4(f, &entries[i].size, 1))
            goto vs_write_fail;
    }

    fclose(f);
    PULSEG_FREE(cache_path);
    PULSEG_FREE(blob);
    return PULSEG_SUCCESS;

vs_write_fail:
    fclose(f);
    PULSEG_FREE(cache_path);
    PULSEG_FREE(blob);
    return PULSEG_ERR_FILE_READ_FAILED;
}

int pulseg_read_vendor_cache_section(
    const char *seq_path,
    const char *cache_ext,
    unsigned char **out_buf,
    int *out_len)
{
    char *cache_path;
    FILE *f;
    int marker, num_sections;
    int version_major, version_minor, version_revision, vendor, stored_size;
    int do_swap, i, found;
    pulseg_cache_section_entry section;
    int blob_len;
    unsigned char *blob;

    if (!seq_path || !out_buf || !out_len)
        return PULSEG_ERR_NULL_POINTER;
    *out_buf = NULL;
    *out_len = 0;

    cache_path = pulseg__make_cache_path(seq_path, cache_ext);
    if (!cache_path)
        return PULSEG_ERR_ALLOC_FAILED;

    f = fopen(cache_path, "rb");
    if (!f)
    {
        PULSEG_FREE(cache_path);
        return PULSEG_ERR_FILE_READ_FAILED;
    }

    if (!pulseg__read4(f, &marker, 1))
        goto vs_read_fail;
    do_swap = 0;
    if (marker != PULSEG_CACHE_ENDIAN_MARKER)
    {
        pulseg__swap4(&marker);
        if (marker != PULSEG_CACHE_ENDIAN_MARKER)
            goto vs_read_fail;
        do_swap = 1;
    }
    if (!pulseg__read4(f, &version_major, 1))
        goto vs_read_fail;
    if (!pulseg__read4(f, &version_minor, 1))
        goto vs_read_fail;
    if (!pulseg__read4(f, &version_revision, 1))
        goto vs_read_fail;
    if (!pulseg__read4(f, &vendor, 1))
        goto vs_read_fail;
    if (!pulseg__read4(f, &stored_size, 1))
        goto vs_read_fail;
    if (!pulseg__read4(f, &num_sections, 1))
        goto vs_read_fail;
    if (do_swap)
        pulseg__swap4(&num_sections);
    if (num_sections <= 0 || num_sections > 15)
        goto vs_read_fail;

    found = 0;
    section.section_id = -1;
    section.offset = 0;
    section.size = 0;
    for (i = 0; i < num_sections; ++i)
    {
        pulseg_cache_section_entry e;
        if (!pulseg__read4(f, &e.section_id, 1))
            goto vs_read_fail;
        if (!pulseg__read4(f, &e.offset, 1))
            goto vs_read_fail;
        if (!pulseg__read4(f, &e.size, 1))
            goto vs_read_fail;
        if (do_swap)
        {
            pulseg__swap4(&e.section_id);
            pulseg__swap4(&e.offset);
            pulseg__swap4(&e.size);
        }
        if (e.section_id == PULSEG_CACHE_SECTION_VENDOR)
        {
            section = e;
            found = 1;
        }
    }

    if (!found)
    {
        fclose(f);
        PULSEG_FREE(cache_path);
        return PULSEG_SUCCESS; /* no VENDOR section -- not an error */
    }

    if (fseek(f, section.offset, SEEK_SET) != 0)
        goto vs_read_fail;
    if (!pulseg__read4(f, &blob_len, 1))
        goto vs_read_fail;
    if (do_swap)
        pulseg__swap4(&blob_len);
    if (blob_len < 0 || blob_len > section.size)
        goto vs_read_fail;

    blob = NULL;
    if (blob_len > 0)
    {
        blob = (unsigned char *)PULSEG_ALLOC((size_t)blob_len);
        if (!blob)
            goto vs_read_fail;
        if ((int)fread(blob, 1, (size_t)blob_len, f) != blob_len)
        {
            PULSEG_FREE(blob);
            goto vs_read_fail;
        }
    }

    fclose(f);
    PULSEG_FREE(cache_path);
    *out_buf = blob;
    *out_len = blob_len;
    return PULSEG_SUCCESS;

vs_read_fail:
    fclose(f);
    PULSEG_FREE(cache_path);
    return PULSEG_ERR_FILE_READ_FAILED;
}

int pulseg_load_freq_mod_cache(pulseg_collection *coll, const char *seq_path)
{
    char *cache_path;
    FILE *f;
    int marker, num_sections;
    int version_major, version_minor, version_revision, vendor, stored_size;
    int do_swap, i, found;
    float zero_shift[3] = {0.0f, 0.0f, 0.0f};
    pulseg_cache_section_entry section;

    if (!coll || !seq_path)
        return PULSEG_ERR_NULL_POINTER;

    if (coll->freq_mod)
    {
        pulseg_freq_mod_collection_free(coll->freq_mod);
        coll->freq_mod = NULL;
    }

    cache_path = pulseg__make_cache_path(
        seq_path,
        coll->num_subsequences > 0 ? coll->descriptors[0].cache_ext : NULL);
    if (!cache_path)
        return PULSEG_ERR_ALLOC_FAILED;

    f = fopen(cache_path, "rb");
    PULSEG_FREE(cache_path);
    if (!f)
        return PULSEG_ERR_FILE_READ_FAILED;

    /* Read header */
    if (!pulseg__read4(f, &marker, 1))
    {
        fclose(f);
        return PULSEG_ERR_FILE_READ_FAILED;
    }
    do_swap = 0;
    if (marker != PULSEG_CACHE_ENDIAN_MARKER)
    {
        pulseg__swap4(&marker);
        if (marker != PULSEG_CACHE_ENDIAN_MARKER)
        {
            fclose(f);
            return PULSEG_ERR_FILE_READ_FAILED;
        }
        do_swap = 1;
    }
    if (!pulseg__read4(f, &version_major, 1))
    {
        fclose(f);
        return PULSEG_ERR_FILE_READ_FAILED;
    }
    if (!pulseg__read4(f, &version_minor, 1))
    {
        fclose(f);
        return PULSEG_ERR_FILE_READ_FAILED;
    }
    if (!pulseg__read4(f, &version_revision, 1))
    {
        fclose(f);
        return PULSEG_ERR_FILE_READ_FAILED;
    }
    if (!pulseg__read4(f, &vendor, 1))
    {
        fclose(f);
        return PULSEG_ERR_FILE_READ_FAILED;
    }
    if (!pulseg__read4(f, &stored_size, 1))
    {
        fclose(f);
        return PULSEG_ERR_FILE_READ_FAILED;
    }
    if (!pulseg__read4(f, &num_sections, 1))
    {
        fclose(f);
        return PULSEG_ERR_FILE_READ_FAILED;
    }
    if (do_swap)
    {
        pulseg__swap4(&version_major);
        pulseg__swap4(&version_minor);
        pulseg__swap4(&version_revision);
        pulseg__swap4(&vendor);
        pulseg__swap4(&stored_size);
        pulseg__swap4(&num_sections);
    }
    if (num_sections <= 0 || num_sections > 16)
    {
        fclose(f);
        return PULSEG_ERR_FILE_READ_FAILED;
    }

    /* Find the freq-mod section */
    found = 0;
    memset(&section, 0, sizeof(section));
    for (i = 0; i < num_sections; ++i)
    {
        pulseg_cache_section_entry entry;
        if (!pulseg__read4(f, &entry.section_id, 1))
        {
            fclose(f);
            return PULSEG_ERR_FILE_READ_FAILED;
        }
        if (!pulseg__read4(f, &entry.offset, 1))
        {
            fclose(f);
            return PULSEG_ERR_FILE_READ_FAILED;
        }
        if (!pulseg__read4(f, &entry.size, 1))
        {
            fclose(f);
            return PULSEG_ERR_FILE_READ_FAILED;
        }
        if (do_swap)
        {
            pulseg__swap4(&entry.section_id);
            pulseg__swap4(&entry.offset);
            pulseg__swap4(&entry.size);
        }
        if (entry.section_id == PULSEG_CACHE_SECTION_FREQMOD)
        {
            section = entry;
            found = 1;
        }
    }

    if (!found || section.offset <= 0 || section.size <= 0)
    {
        fclose(f);
        return PULSEG_ERR_FILE_READ_FAILED;
    }

    if (fseek(f, (long)section.offset, SEEK_SET) != 0)
    {
        fclose(f);
        return PULSEG_ERR_FILE_READ_FAILED;
    }

    {
        int rc = pulseg__freq_mod_collection_read_f(&coll->freq_mod, f, coll, zero_shift, do_swap);
        fclose(f);
        return rc;
    }
}
