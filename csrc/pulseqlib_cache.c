/* pulseqlib_cache.c -- binary cache for descriptor collections */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "pulseqlib_internal.h"
#include "pulseqlib_methods.h"

/* ================================================================== */
/*  Binary cache: serialization / deserialization                     */
/* ================================================================== */

#define PULSEQLIB_CACHE_ENDIAN_MARKER  0x01020304
#define PULSEQLIB_CACHE_VERSION        15

/* ------ Byte-swap helpers ------ */

static void swap4(void* p)
{
    unsigned char* b = (unsigned char*)p;
    unsigned char t;
    t = b[0]; b[0] = b[3]; b[3] = t;
    t = b[1]; b[1] = b[2]; b[2] = t;
}

static void swap4_array(void* p, int count)
{
    int i;
    for (i = 0; i < count; ++i)
        swap4((unsigned char*)p + (size_t)i * 4);
}

/* ------ I/O helpers ------ */

static int write4(FILE* f, const void* p, int count)
{
    return (int)fwrite(p, 4, (size_t)count, f) == count;
}

static int read4(FILE* f, void* p, int count)
{
    return (int)fread(p, 4, (size_t)count, f) == count;
}

/* ------ Path helper ------ */

static char* make_cache_path(const char* seq_path)
{
    size_t len;
    char* out;
    const char* dot;

    len = strlen(seq_path);
    out = (char*)PULSEQLIB_ALLOC(len + 5); /* worst case: no dot, append ".bin\0" */
    if (!out) return NULL;

    strcpy(out, seq_path);
    dot = strrchr(out, '.');
    if (dot && dot > strrchr(out, '/') && dot > strrchr(out, '\\')) {
        /* replace extension */
        strcpy((char*)(out + (dot - out)), ".bin");
    } else {
        strcat(out, ".bin");
    }
    return out;
}

/* ------ File size helper (C89) ------ */

static long get_file_size(const char* path)
{
    FILE* f;
    long sz;
    f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    sz = ftell(f);
    fclose(f);
    return sz;
}

/* ------ Get seq file sizes for all files in chain ------ */

static int get_seq_file_sizes(const char* first_file_path,
                              const pulseqlib_opts* opts,
                              int* out_sizes, int max_files)
{
    long sz;

    (void)opts;
    (void)max_files;

    sz = get_file_size(first_file_path);
    if (sz < 0) return 0;
    out_sizes[0] = (int)sz;

    /* For single-file or when we don't have the chain yet,
     * return 1.  The full chain sizes are stored at write time. */
    return 1;
}

/* ------ Serialize a single sequence descriptor ------ */

static int write_descriptor(FILE* f, const pulseqlib_sequence_descriptor* d)
{
    int i, n;
    int ival;

    /* scalars */
    if (!write4(f, &d->num_prep_blocks, 1)) return 0;
    if (!write4(f, &d->num_cooldown_blocks, 1)) return 0;
    if (!write4(f, &d->rf_raster_us, 1)) return 0;
    if (!write4(f, &d->grad_raster_us, 1)) return 0;
    if (!write4(f, &d->adc_raster_us, 1)) return 0;
    if (!write4(f, &d->block_raster_us, 1)) return 0;
    if (!write4(f, &d->ignore_fov_shift, 1)) return 0;
    if (!write4(f, &d->enable_pmc, 1)) return 0;
    if (!write4(f, &d->ignore_averages, 1)) return 0;
    if (!write4(f, &d->num_passes, 1)) return 0;
    if (!write4(f, &d->vendor, 1)) return 0;

    /* block definitions */
    if (!write4(f, &d->num_unique_blocks, 1)) return 0;
    for (i = 0; i < d->num_unique_blocks; ++i) {
        if (!write4(f, &d->block_definitions[i].id, 1)) return 0;
        if (!write4(f, &d->block_definitions[i].duration_us, 1)) return 0;
        if (!write4(f, &d->block_definitions[i].rf_id, 1)) return 0;
        if (!write4(f, &d->block_definitions[i].gx_id, 1)) return 0;
        if (!write4(f, &d->block_definitions[i].gy_id, 1)) return 0;
        if (!write4(f, &d->block_definitions[i].gz_id, 1)) return 0;
        if (!write4(f, &d->block_definitions[i].adc_id, 1)) return 0;
    }

    /* block table */
    if (!write4(f, &d->num_blocks, 1)) return 0;
    for (i = 0; i < d->num_blocks; ++i) {
        if (!write4(f, &d->block_table[i].id, 1)) return 0;
        if (!write4(f, &d->block_table[i].duration_us, 1)) return 0;
        if (!write4(f, &d->block_table[i].rf_id, 1)) return 0;
        if (!write4(f, &d->block_table[i].gx_id, 1)) return 0;
        if (!write4(f, &d->block_table[i].gy_id, 1)) return 0;
        if (!write4(f, &d->block_table[i].gz_id, 1)) return 0;
        if (!write4(f, &d->block_table[i].adc_id, 1)) return 0;
        if (!write4(f, &d->block_table[i].digitalout_id, 1)) return 0;
        if (!write4(f, &d->block_table[i].rotation_id, 1)) return 0;
        if (!write4(f, &d->block_table[i].once_flag, 1)) return 0;
        if (!write4(f, &d->block_table[i].norot_flag, 1)) return 0;
        if (!write4(f, &d->block_table[i].nopos_flag, 1)) return 0;
        if (!write4(f, &d->block_table[i].pmc_flag, 1)) return 0;
        if (!write4(f, &d->block_table[i].nav_flag, 1)) return 0;
        if (!write4(f, &d->block_table[i].freq_mod_id, 1)) return 0;
        if (!write4(f, &d->block_table[i].rf_shim_id, 1)) return 0;
    }

    /* RF definitions */
    if (!write4(f, &d->num_unique_rfs, 1)) return 0;
    for (i = 0; i < d->num_unique_rfs; ++i) {
        if (!write4(f, &d->rf_definitions[i].id, 1)) return 0;
        if (!write4(f, &d->rf_definitions[i].mag_shape_id, 1)) return 0;
        if (!write4(f, &d->rf_definitions[i].phase_shape_id, 1)) return 0;
        if (!write4(f, &d->rf_definitions[i].time_shape_id, 1)) return 0;
        if (!write4(f, &d->rf_definitions[i].delay, 1)) return 0;
        if (!write4(f, &d->rf_definitions[i].num_channels, 1)) return 0;
        if (!write4(f, &d->rf_definitions[i].stats.flip_angle_deg, 1)) return 0;
        if (!write4(f, &d->rf_definitions[i].stats.area, 1)) return 0;
        if (!write4(f, &d->rf_definitions[i].stats.abs_width, 1)) return 0;
        if (!write4(f, &d->rf_definitions[i].stats.eff_width, 1)) return 0;
        if (!write4(f, &d->rf_definitions[i].stats.duty_cycle, 1)) return 0;
        if (!write4(f, &d->rf_definitions[i].stats.max_pulse_width, 1)) return 0;
        if (!write4(f, &d->rf_definitions[i].stats.duration_us, 1)) return 0;
        if (!write4(f, &d->rf_definitions[i].stats.isodelay_us, 1)) return 0;
        if (!write4(f, &d->rf_definitions[i].stats.bandwidth_hz, 1)) return 0;
        if (!write4(f, &d->rf_definitions[i].stats.base_amplitude_hz, 1)) return 0;
        if (!write4(f, &d->rf_definitions[i].stats.num_samples, 1)) return 0;
    }

    /* RF table */
    if (!write4(f, &d->rf_table_size, 1)) return 0;
    for (i = 0; i < d->rf_table_size; ++i) {
        if (!write4(f, &d->rf_table[i].id, 1)) return 0;
        if (!write4(f, &d->rf_table[i].amplitude, 1)) return 0;
        if (!write4(f, &d->rf_table[i].freq_offset, 1)) return 0;
        if (!write4(f, &d->rf_table[i].phase_offset, 1)) return 0;
        if (!write4(f, &d->rf_table[i].rf_use, 1)) return 0;
    }

    /* gradient definitions */
    if (!write4(f, &d->num_unique_grads, 1)) return 0;
    for (i = 0; i < d->num_unique_grads; ++i) {
        const pulseqlib_grad_definition* gd = &d->grad_definitions[i];
        if (!write4(f, &gd->id, 1)) return 0;
        if (!write4(f, &gd->type, 1)) return 0;
        if (!write4(f, &gd->rise_time_or_unused, 1)) return 0;
        if (!write4(f, &gd->flat_time_or_unused, 1)) return 0;
        if (!write4(f, &gd->fall_time_or_num_uncompressed_samples, 1)) return 0;
        if (!write4(f, &gd->unused_or_time_shape_id, 1)) return 0;
        if (!write4(f, &gd->delay, 1)) return 0;
        if (!write4(f, &gd->num_shots, 1)) return 0;
        if (!write4(f, gd->shot_shape_ids, PULSEQLIB_MAX_GRAD_SHOTS)) return 0;
        if (!write4(f, gd->max_amplitude, PULSEQLIB_MAX_GRAD_SHOTS)) return 0;
        if (!write4(f, gd->min_amplitude, PULSEQLIB_MAX_GRAD_SHOTS)) return 0;
        if (!write4(f, gd->slew_rate, PULSEQLIB_MAX_GRAD_SHOTS)) return 0;
        if (!write4(f, gd->energy, PULSEQLIB_MAX_GRAD_SHOTS)) return 0;
        if (!write4(f, gd->first_value, PULSEQLIB_MAX_GRAD_SHOTS)) return 0;
        if (!write4(f, gd->last_value, PULSEQLIB_MAX_GRAD_SHOTS)) return 0;
    }

    /* gradient table */
    if (!write4(f, &d->grad_table_size, 1)) return 0;
    for (i = 0; i < d->grad_table_size; ++i) {
        if (!write4(f, &d->grad_table[i].id, 1)) return 0;
        if (!write4(f, &d->grad_table[i].shot_index, 1)) return 0;
        if (!write4(f, &d->grad_table[i].amplitude, 1)) return 0;
    }

    /* ADC definitions */
    if (!write4(f, &d->num_unique_adcs, 1)) return 0;
    for (i = 0; i < d->num_unique_adcs; ++i) {
        if (!write4(f, &d->adc_definitions[i].id, 1)) return 0;
        if (!write4(f, &d->adc_definitions[i].num_samples, 1)) return 0;
        if (!write4(f, &d->adc_definitions[i].dwell_time, 1)) return 0;
        if (!write4(f, &d->adc_definitions[i].delay, 1)) return 0;
    }

    /* ADC table */
    if (!write4(f, &d->adc_table_size, 1)) return 0;
    for (i = 0; i < d->adc_table_size; ++i) {
        if (!write4(f, &d->adc_table[i].id, 1)) return 0;
        if (!write4(f, &d->adc_table[i].freq_offset, 1)) return 0;
        if (!write4(f, &d->adc_table[i].phase_offset, 1)) return 0;
    }

    /* freq_mod definitions (no longer stored; write count = 0) */
    {
        int zero = 0;
        if (!write4(f, &zero, 1)) return 0;
    }

    /* rf_shim definitions */
    if (!write4(f, &d->num_rf_shims, 1)) return 0;
    for (i = 0; i < d->num_rf_shims; ++i) {
        const pulseqlib_rf_shim_definition* rs = &d->rf_shim_definitions[i];
        if (!write4(f, &rs->id, 1)) return 0;
        if (!write4(f, &rs->num_channels, 1)) return 0;
        if (rs->num_channels > 0) {
            if (!write4(f, rs->magnitudes, rs->num_channels)) return 0;
            if (!write4(f, rs->phases, rs->num_channels)) return 0;
        }
    }

    /* rotations */
    if (!write4(f, &d->num_rotations, 1)) return 0;
    for (i = 0; i < d->num_rotations; ++i)
        if (!write4(f, d->rotation_matrices[i], 9)) return 0;

    /* triggers — serialize long/short as int for portability */
    if (!write4(f, &d->num_triggers, 1)) return 0;
    for (i = 0; i < d->num_triggers; ++i) {
        ival = (int)d->trigger_events[i].type;
        if (!write4(f, &ival, 1)) return 0;
        ival = (int)d->trigger_events[i].duration;
        if (!write4(f, &ival, 1)) return 0;
        ival = (int)d->trigger_events[i].delay;
        if (!write4(f, &ival, 1)) return 0;
        if (!write4(f, &d->trigger_events[i].trigger_type, 1)) return 0;
        if (!write4(f, &d->trigger_events[i].trigger_channel, 1)) return 0;
    }

    /* shapes */
    if (!write4(f, &d->num_shapes, 1)) return 0;
    for (i = 0; i < d->num_shapes; ++i) {
        if (!write4(f, &d->shapes[i].num_uncompressed_samples, 1)) return 0;
        if (!write4(f, &d->shapes[i].num_samples, 1)) return 0;
        n = d->shapes[i].num_samples;
        if (n > 0 && d->shapes[i].samples)
            if (!write4(f, d->shapes[i].samples, n)) return 0;
    }

    /* TR descriptor (10 fields: 9 int + 1 float) */
    if (!write4(f, &d->tr_descriptor.num_prep_blocks, 1)) return 0;
    if (!write4(f, &d->tr_descriptor.num_cooldown_blocks, 1)) return 0;
    if (!write4(f, &d->tr_descriptor.tr_size, 1)) return 0;
    if (!write4(f, &d->tr_descriptor.num_trs, 1)) return 0;
    if (!write4(f, &d->tr_descriptor.num_prep_trs, 1)) return 0;
    if (!write4(f, &d->tr_descriptor.degenerate_prep, 1)) return 0;
    if (!write4(f, &d->tr_descriptor.num_cooldown_trs, 1)) return 0;
    if (!write4(f, &d->tr_descriptor.degenerate_cooldown, 1)) return 0;
    if (!write4(f, &d->tr_descriptor.imaging_tr_start, 1)) return 0;
    if (!write4(f, &d->tr_descriptor.tr_duration_us, 1)) return 0;

    /* segment definitions */
    if (!write4(f, &d->num_unique_segments, 1)) return 0;
    for (i = 0; i < d->num_unique_segments; ++i) {
        const pulseqlib_tr_segment* seg = &d->segment_definitions[i];
        if (!write4(f, &seg->start_block, 1)) return 0;
        if (!write4(f, &seg->num_blocks, 1)) return 0;
        if (!write4(f, &seg->max_energy_start_block, 1)) return 0;
        if (seg->num_blocks > 0) {
            if (!write4(f, seg->unique_block_indices, seg->num_blocks)) return 0;
            if (!write4(f, seg->has_digitalout, seg->num_blocks)) return 0;
            if (!write4(f, seg->has_rotation, seg->num_blocks)) return 0;
            if (!write4(f, seg->norot_flag, seg->num_blocks)) return 0;
            if (!write4(f, seg->nopos_flag, seg->num_blocks)) return 0;
        }
        if (!write4(f, &seg->trigger_id, 1)) return 0;
        if (!write4(f, &seg->is_nav, 1)) return 0;
    }

    /* segment table */
    if (!write4(f, &d->segment_table.num_unique_segments, 1)) return 0;
    if (!write4(f, &d->segment_table.num_prep_segments, 1)) return 0;
    if (d->segment_table.num_prep_segments > 0)
        if (!write4(f, d->segment_table.prep_segment_table, d->segment_table.num_prep_segments)) return 0;
    if (!write4(f, &d->segment_table.num_main_segments, 1)) return 0;
    if (d->segment_table.num_main_segments > 0)
        if (!write4(f, d->segment_table.main_segment_table, d->segment_table.num_main_segments)) return 0;
    if (!write4(f, &d->segment_table.num_cooldown_segments, 1)) return 0;
    if (d->segment_table.num_cooldown_segments > 0)
        if (!write4(f, d->segment_table.cooldown_segment_table, d->segment_table.num_cooldown_segments)) return 0;

    /* label table */
    fwrite(&d->label_num_columns, sizeof(int), 1, f);
    fwrite(&d->label_num_entries, sizeof(int), 1, f);
    if (d->label_num_entries > 0 && d->label_table) {
        fwrite(d->label_table, sizeof(int),
               (size_t)d->label_num_entries * (size_t)d->label_num_columns, f);
    }
    fwrite(&d->label_limits, sizeof(pulseqlib_label_limits), 1, f);

    /* scan table */
    if (!write4(f, &d->scan_table_len, 1)) return 0;
    if (d->scan_table_len > 0) {
        if (!write4(f, d->scan_table_block_idx, d->scan_table_len)) return 0;
        if (!write4(f, d->scan_table_tr_id,    d->scan_table_len)) return 0;
        if (!write4(f, d->scan_table_seg_id,   d->scan_table_len)) return 0;
    }

    return 1;
}

/* ------ Deserialize a single sequence descriptor ------ */

static int read_descriptor(FILE* f, pulseqlib_sequence_descriptor* d, int do_swap)
{
    int i, n;
    int ival;

    memset(d, 0, sizeof(*d));

    /* scalars */
    if (!read4(f, &d->num_prep_blocks, 1)) return 0;
    if (!read4(f, &d->num_cooldown_blocks, 1)) return 0;
    if (!read4(f, &d->rf_raster_us, 1)) return 0;
    if (!read4(f, &d->grad_raster_us, 1)) return 0;
    if (!read4(f, &d->adc_raster_us, 1)) return 0;
    if (!read4(f, &d->block_raster_us, 1)) return 0;
    if (!read4(f, &d->ignore_fov_shift, 1)) return 0;
    if (!read4(f, &d->enable_pmc, 1)) return 0;
    if (!read4(f, &d->ignore_averages, 1)) return 0;
    if (!read4(f, &d->num_passes, 1)) return 0;
    if (!read4(f, &d->vendor, 1)) return 0;
    if (do_swap) swap4_array(&d->num_prep_blocks, 11);

    /* block definitions */
    if (!read4(f, &d->num_unique_blocks, 1)) return 0;
    if (do_swap) swap4(&d->num_unique_blocks);
    d->block_definitions = (pulseqlib_block_definition*)PULSEQLIB_ALLOC(
        (size_t)d->num_unique_blocks * sizeof(pulseqlib_block_definition));
    if (!d->block_definitions) return 0;
    for (i = 0; i < d->num_unique_blocks; ++i) {
        if (!read4(f, &d->block_definitions[i].id, 1)) return 0;
        if (!read4(f, &d->block_definitions[i].duration_us, 1)) return 0;
        if (!read4(f, &d->block_definitions[i].rf_id, 1)) return 0;
        if (!read4(f, &d->block_definitions[i].gx_id, 1)) return 0;
        if (!read4(f, &d->block_definitions[i].gy_id, 1)) return 0;
        if (!read4(f, &d->block_definitions[i].gz_id, 1)) return 0;
        if (!read4(f, &d->block_definitions[i].adc_id, 1)) return 0;
        if (do_swap) swap4_array(&d->block_definitions[i].id, 7);
    }

    /* block table */
    if (!read4(f, &d->num_blocks, 1)) return 0;
    if (do_swap) swap4(&d->num_blocks);
    d->block_table = (pulseqlib_block_table_element*)PULSEQLIB_ALLOC(
        (size_t)d->num_blocks * sizeof(pulseqlib_block_table_element));
    if (!d->block_table) return 0;
    for (i = 0; i < d->num_blocks; ++i) {
        if (!read4(f, &d->block_table[i].id, 16)) return 0;
        if (do_swap) swap4_array(&d->block_table[i].id, 16);
    }

    /* RF definitions */
    if (!read4(f, &d->num_unique_rfs, 1)) return 0;
    if (do_swap) swap4(&d->num_unique_rfs);
    d->rf_definitions = (pulseqlib_rf_definition*)PULSEQLIB_ALLOC(
        (size_t)d->num_unique_rfs * sizeof(pulseqlib_rf_definition));
    if (!d->rf_definitions) return 0;
    for (i = 0; i < d->num_unique_rfs; ++i) {
        if (!read4(f, &d->rf_definitions[i].id, 1)) return 0;
        if (!read4(f, &d->rf_definitions[i].mag_shape_id, 1)) return 0;
        if (!read4(f, &d->rf_definitions[i].phase_shape_id, 1)) return 0;
        if (!read4(f, &d->rf_definitions[i].time_shape_id, 1)) return 0;
        if (!read4(f, &d->rf_definitions[i].delay, 1)) return 0;
        if (!read4(f, &d->rf_definitions[i].num_channels, 1)) return 0;
        if (do_swap) swap4_array(&d->rf_definitions[i].id, 6);
        if (!read4(f, &d->rf_definitions[i].stats.flip_angle_deg, 1)) return 0;
        if (!read4(f, &d->rf_definitions[i].stats.area, 1)) return 0;
        if (!read4(f, &d->rf_definitions[i].stats.abs_width, 1)) return 0;
        if (!read4(f, &d->rf_definitions[i].stats.eff_width, 1)) return 0;
        if (!read4(f, &d->rf_definitions[i].stats.duty_cycle, 1)) return 0;
        if (!read4(f, &d->rf_definitions[i].stats.max_pulse_width, 1)) return 0;
        if (!read4(f, &d->rf_definitions[i].stats.duration_us, 1)) return 0;
        if (!read4(f, &d->rf_definitions[i].stats.isodelay_us, 1)) return 0;
        if (!read4(f, &d->rf_definitions[i].stats.bandwidth_hz, 1)) return 0;
        if (!read4(f, &d->rf_definitions[i].stats.base_amplitude_hz, 1)) return 0;
        if (!read4(f, &d->rf_definitions[i].stats.num_samples, 1)) return 0;
        if (do_swap) swap4_array(&d->rf_definitions[i].stats.flip_angle_deg, 11);
    }

    /* RF table */
    if (!read4(f, &d->rf_table_size, 1)) return 0;
    if (do_swap) swap4(&d->rf_table_size);
    d->rf_table = (pulseqlib_rf_table_element*)PULSEQLIB_ALLOC(
        (size_t)d->rf_table_size * sizeof(pulseqlib_rf_table_element));
    if (!d->rf_table) return 0;
    for (i = 0; i < d->rf_table_size; ++i) {
        if (!read4(f, &d->rf_table[i].id, 4)) return 0;
        if (do_swap) swap4_array(&d->rf_table[i].id, 4);
        if (!read4(f, &d->rf_table[i].rf_use, 1)) return 0;
        if (do_swap) swap4(&d->rf_table[i].rf_use);
    }

    /* gradient definitions */
    if (!read4(f, &d->num_unique_grads, 1)) return 0;
    if (do_swap) swap4(&d->num_unique_grads);
    d->grad_definitions = (pulseqlib_grad_definition*)PULSEQLIB_ALLOC(
        (size_t)d->num_unique_grads * sizeof(pulseqlib_grad_definition));
    if (!d->grad_definitions) return 0;
    for (i = 0; i < d->num_unique_grads; ++i) {
        pulseqlib_grad_definition* gd = &d->grad_definitions[i];
        if (!read4(f, &gd->id, 1)) return 0;
        if (!read4(f, &gd->type, 1)) return 0;
        if (!read4(f, &gd->rise_time_or_unused, 1)) return 0;
        if (!read4(f, &gd->flat_time_or_unused, 1)) return 0;
        if (!read4(f, &gd->fall_time_or_num_uncompressed_samples, 1)) return 0;
        if (!read4(f, &gd->unused_or_time_shape_id, 1)) return 0;
        if (!read4(f, &gd->delay, 1)) return 0;
        if (!read4(f, &gd->num_shots, 1)) return 0;
        if (do_swap) swap4_array(&gd->id, 8);
        if (!read4(f, gd->shot_shape_ids, PULSEQLIB_MAX_GRAD_SHOTS)) return 0;
        if (!read4(f, gd->max_amplitude, PULSEQLIB_MAX_GRAD_SHOTS)) return 0;
        if (!read4(f, gd->min_amplitude, PULSEQLIB_MAX_GRAD_SHOTS)) return 0;
        if (!read4(f, gd->slew_rate, PULSEQLIB_MAX_GRAD_SHOTS)) return 0;
        if (!read4(f, gd->energy, PULSEQLIB_MAX_GRAD_SHOTS)) return 0;
        if (!read4(f, gd->first_value, PULSEQLIB_MAX_GRAD_SHOTS)) return 0;
        if (!read4(f, gd->last_value, PULSEQLIB_MAX_GRAD_SHOTS)) return 0;
        if (do_swap) swap4_array(gd->shot_shape_ids, 8 * PULSEQLIB_MAX_GRAD_SHOTS);
    }

    /* gradient table */
    if (!read4(f, &d->grad_table_size, 1)) return 0;
    if (do_swap) swap4(&d->grad_table_size);
    d->grad_table = (pulseqlib_grad_table_element*)PULSEQLIB_ALLOC(
        (size_t)d->grad_table_size * sizeof(pulseqlib_grad_table_element));
    if (!d->grad_table) return 0;
    for (i = 0; i < d->grad_table_size; ++i) {
        if (!read4(f, &d->grad_table[i].id, 3)) return 0;
        if (do_swap) swap4_array(&d->grad_table[i].id, 3);
    }

    /* ADC definitions */
    if (!read4(f, &d->num_unique_adcs, 1)) return 0;
    if (do_swap) swap4(&d->num_unique_adcs);
    d->adc_definitions = (pulseqlib_adc_definition*)PULSEQLIB_ALLOC(
        (size_t)d->num_unique_adcs * sizeof(pulseqlib_adc_definition));
    if (!d->adc_definitions) return 0;
    for (i = 0; i < d->num_unique_adcs; ++i) {
        if (!read4(f, &d->adc_definitions[i].id, 4)) return 0;
        if (do_swap) swap4_array(&d->adc_definitions[i].id, 4);
    }

    /* ADC table */
    if (!read4(f, &d->adc_table_size, 1)) return 0;
    if (do_swap) swap4(&d->adc_table_size);
    d->adc_table = (pulseqlib_adc_table_element*)PULSEQLIB_ALLOC(
        (size_t)d->adc_table_size * sizeof(pulseqlib_adc_table_element));
    if (!d->adc_table) return 0;
    for (i = 0; i < d->adc_table_size; ++i) {
        if (!read4(f, &d->adc_table[i].id, 3)) return 0;
        if (do_swap) swap4_array(&d->adc_table[i].id, 3);
    }

    /* freq_mod definitions (legacy: read and skip if count > 0) */
    if (!read4(f, &d->num_freq_mod_defs, 1)) return 0;
    if (do_swap) swap4(&d->num_freq_mod_defs);
    d->num_freq_mod_defs = 0;
    d->freq_mod_definitions = NULL;

    /* rf_shim definitions */
    if (!read4(f, &d->num_rf_shims, 1)) return 0;
    if (do_swap) swap4(&d->num_rf_shims);
    if (d->num_rf_shims > 0) {
        d->rf_shim_definitions = (pulseqlib_rf_shim_definition*)PULSEQLIB_ALLOC(
            (size_t)d->num_rf_shims * sizeof(pulseqlib_rf_shim_definition));
        if (!d->rf_shim_definitions) return 0;
        for (i = 0; i < d->num_rf_shims; ++i) {
            pulseqlib_rf_shim_definition* rs = &d->rf_shim_definitions[i];
            memset(rs, 0, sizeof(*rs));
            if (!read4(f, &rs->id, 1)) return 0;
            if (!read4(f, &rs->num_channels, 1)) return 0;
            if (do_swap) { swap4(&rs->id); swap4(&rs->num_channels); }
            n = rs->num_channels;
            if (n > 0 && n <= PULSEQLIB_MAX_RF_SHIM_CHANNELS) {
                if (!read4(f, rs->magnitudes, n)) return 0;
                if (!read4(f, rs->phases, n)) return 0;
                if (do_swap) { swap4_array(rs->magnitudes, n); swap4_array(rs->phases, n); }
            }
        }
    }

    /* rotations */
    if (!read4(f, &d->num_rotations, 1)) return 0;
    if (do_swap) swap4(&d->num_rotations);
    if (d->num_rotations > 0) {
        d->rotation_matrices = (float(*)[9])PULSEQLIB_ALLOC(
            (size_t)d->num_rotations * 9 * sizeof(float));
        if (!d->rotation_matrices) return 0;
        for (i = 0; i < d->num_rotations; ++i) {
            if (!read4(f, d->rotation_matrices[i], 9)) return 0;
            if (do_swap) swap4_array(d->rotation_matrices[i], 9);
        }
    }

    /* triggers */
    if (!read4(f, &d->num_triggers, 1)) return 0;
    if (do_swap) swap4(&d->num_triggers);
    if (d->num_triggers > 0) {
        d->trigger_events = (pulseqlib_trigger_event*)PULSEQLIB_ALLOC(
            (size_t)d->num_triggers * sizeof(pulseqlib_trigger_event));
        if (!d->trigger_events) return 0;
        for (i = 0; i < d->num_triggers; ++i) {
            if (!read4(f, &ival, 1)) return 0;
            if (do_swap) swap4(&ival);
            d->trigger_events[i].type = (short)ival;
            if (!read4(f, &ival, 1)) return 0;
            if (do_swap) swap4(&ival);
            d->trigger_events[i].duration = (long)ival;
            if (!read4(f, &ival, 1)) return 0;
            if (do_swap) swap4(&ival);
            d->trigger_events[i].delay = (long)ival;
            if (!read4(f, &d->trigger_events[i].trigger_type, 1)) return 0;
            if (!read4(f, &d->trigger_events[i].trigger_channel, 1)) return 0;
            if (do_swap) swap4_array(&d->trigger_events[i].trigger_type, 2);
        }
    }

    /* shapes */
    if (!read4(f, &d->num_shapes, 1)) return 0;
    if (do_swap) swap4(&d->num_shapes);
    if (d->num_shapes > 0) {
        d->shapes = (pulseqlib_shape_arbitrary*)PULSEQLIB_ALLOC(
            (size_t)d->num_shapes * sizeof(pulseqlib_shape_arbitrary));
        if (!d->shapes) return 0;
        for (i = 0; i < d->num_shapes; ++i) {
            d->shapes[i].samples = NULL;
            if (!read4(f, &d->shapes[i].num_uncompressed_samples, 1)) return 0;
            if (!read4(f, &d->shapes[i].num_samples, 1)) return 0;
            if (do_swap) swap4_array(&d->shapes[i].num_uncompressed_samples, 2);
            n = d->shapes[i].num_samples;
            if (n > 0) {
                d->shapes[i].samples = (float*)PULSEQLIB_ALLOC((size_t)n * sizeof(float));
                if (!d->shapes[i].samples) return 0;
                if (!read4(f, d->shapes[i].samples, n)) return 0;
                if (do_swap) swap4_array(d->shapes[i].samples, n);
            }
        }
    }

    /* TR descriptor */
    if (!read4(f, &d->tr_descriptor.num_prep_blocks, 1)) return 0;
    if (!read4(f, &d->tr_descriptor.num_cooldown_blocks, 1)) return 0;
    if (!read4(f, &d->tr_descriptor.tr_size, 1)) return 0;
    if (!read4(f, &d->tr_descriptor.num_trs, 1)) return 0;
    if (!read4(f, &d->tr_descriptor.num_prep_trs, 1)) return 0;
    if (!read4(f, &d->tr_descriptor.degenerate_prep, 1)) return 0;
    if (!read4(f, &d->tr_descriptor.num_cooldown_trs, 1)) return 0;
    if (!read4(f, &d->tr_descriptor.degenerate_cooldown, 1)) return 0;
    if (!read4(f, &d->tr_descriptor.imaging_tr_start, 1)) return 0;
    if (!read4(f, &d->tr_descriptor.tr_duration_us, 1)) return 0;
    if (do_swap) swap4_array(&d->tr_descriptor.num_prep_blocks, 10);

    /* segment definitions */
    if (!read4(f, &d->num_unique_segments, 1)) return 0;
    if (do_swap) swap4(&d->num_unique_segments);
    if (d->num_unique_segments > 0) {
        d->segment_definitions = (pulseqlib_tr_segment*)PULSEQLIB_ALLOC(
            (size_t)d->num_unique_segments * sizeof(pulseqlib_tr_segment));
        if (!d->segment_definitions) return 0;
        for (i = 0; i < d->num_unique_segments; ++i) {
            pulseqlib_tr_segment* seg = &d->segment_definitions[i];
            seg->unique_block_indices = NULL;
            seg->has_digitalout = NULL;
            seg->has_rotation = NULL;
            seg->norot_flag = NULL;
            seg->nopos_flag = NULL;
            seg->trigger_id = -1;
            seg->is_nav = 0;

            if (!read4(f, &seg->start_block, 1)) return 0;
            if (!read4(f, &seg->num_blocks, 1)) return 0;
            if (!read4(f, &seg->max_energy_start_block, 1)) return 0;
            if (do_swap) swap4_array(&seg->start_block, 3);

            n = seg->num_blocks;
            if (n > 0) {
                seg->unique_block_indices = (int*)PULSEQLIB_ALLOC((size_t)n * sizeof(int));
                seg->has_digitalout  = (int*)PULSEQLIB_ALLOC((size_t)n * sizeof(int));
                seg->has_rotation = (int*)PULSEQLIB_ALLOC((size_t)n * sizeof(int));
                seg->norot_flag   = (int*)PULSEQLIB_ALLOC((size_t)n * sizeof(int));
                seg->nopos_flag   = (int*)PULSEQLIB_ALLOC((size_t)n * sizeof(int));
                if (!seg->unique_block_indices || !seg->has_digitalout ||
                    !seg->has_rotation || !seg->norot_flag || !seg->nopos_flag)
                    return 0;
                if (!read4(f, seg->unique_block_indices, n)) return 0;
                if (!read4(f, seg->has_digitalout, n)) return 0;
                if (!read4(f, seg->has_rotation, n)) return 0;
                if (!read4(f, seg->norot_flag, n)) return 0;
                if (!read4(f, seg->nopos_flag, n)) return 0;
                if (do_swap) {
                    swap4_array(seg->unique_block_indices, n);
                    swap4_array(seg->has_digitalout, n);
                    swap4_array(seg->has_rotation, n);
                    swap4_array(seg->norot_flag, n);
                    swap4_array(seg->nopos_flag, n);
                }
            }
            if (!read4(f, &seg->trigger_id, 1)) return 0;
            if (do_swap) swap4(&seg->trigger_id);
            if (!read4(f, &seg->is_nav, 1)) return 0;
            if (do_swap) swap4(&seg->is_nav);
        }
    }

    /* segment table */
    if (!read4(f, &d->segment_table.num_unique_segments, 1)) return 0;
    if (!read4(f, &d->segment_table.num_prep_segments, 1)) return 0;
    if (do_swap) swap4_array(&d->segment_table.num_unique_segments, 2);
    if (d->segment_table.num_prep_segments > 0) {
        d->segment_table.prep_segment_table = (int*)PULSEQLIB_ALLOC(
            (size_t)d->segment_table.num_prep_segments * sizeof(int));
        if (!d->segment_table.prep_segment_table) return 0;
        if (!read4(f, d->segment_table.prep_segment_table, d->segment_table.num_prep_segments)) return 0;
        if (do_swap) swap4_array(d->segment_table.prep_segment_table, d->segment_table.num_prep_segments);
    }
    if (!read4(f, &d->segment_table.num_main_segments, 1)) return 0;
    if (do_swap) swap4(&d->segment_table.num_main_segments);
    if (d->segment_table.num_main_segments > 0) {
        d->segment_table.main_segment_table = (int*)PULSEQLIB_ALLOC(
            (size_t)d->segment_table.num_main_segments * sizeof(int));
        if (!d->segment_table.main_segment_table) return 0;
        if (!read4(f, d->segment_table.main_segment_table, d->segment_table.num_main_segments)) return 0;
        if (do_swap) swap4_array(d->segment_table.main_segment_table, d->segment_table.num_main_segments);
    }
    if (!read4(f, &d->segment_table.num_cooldown_segments, 1)) return 0;
    if (do_swap) swap4(&d->segment_table.num_cooldown_segments);
    if (d->segment_table.num_cooldown_segments > 0) {
        d->segment_table.cooldown_segment_table = (int*)PULSEQLIB_ALLOC(
            (size_t)d->segment_table.num_cooldown_segments * sizeof(int));
        if (!d->segment_table.cooldown_segment_table) return 0;
        if (!read4(f, d->segment_table.cooldown_segment_table, d->segment_table.num_cooldown_segments)) return 0;
        if (do_swap) swap4_array(d->segment_table.cooldown_segment_table, d->segment_table.num_cooldown_segments);
    }

    /* label table */
    if (fread(&d->label_num_columns, sizeof(int), 1, f) != 1) return 0;
    if (fread(&d->label_num_entries, sizeof(int), 1, f) != 1) return 0;
    if (d->label_num_entries > 0) {
        d->label_table = (int*)PULSEQLIB_ALLOC(
            (size_t)d->label_num_entries * (size_t)d->label_num_columns * sizeof(int));
        if (!d->label_table) return 0;
        if (fread(d->label_table, sizeof(int),
                  (size_t)d->label_num_entries * (size_t)d->label_num_columns, f)
            != (size_t)d->label_num_entries * (size_t)d->label_num_columns)
            return 0;
    } else {
        d->label_table = NULL;
    }
    if (fread(&d->label_limits, sizeof(pulseqlib_label_limits), 1, f) != 1) return 0;

    /* scan table */
    if (fread(&d->scan_table_len, sizeof(int), 1, f) != 1) return 0;
    if (do_swap) swap4(&d->scan_table_len);
    if (d->scan_table_len > 0) {
        d->scan_table_block_idx = (int*)PULSEQLIB_ALLOC((size_t)d->scan_table_len * sizeof(int));
        d->scan_table_tr_id     = (int*)PULSEQLIB_ALLOC((size_t)d->scan_table_len * sizeof(int));
        d->scan_table_seg_id    = (int*)PULSEQLIB_ALLOC((size_t)d->scan_table_len * sizeof(int));
        if (!d->scan_table_block_idx || !d->scan_table_tr_id || !d->scan_table_seg_id) return 0;
        if (fread(d->scan_table_block_idx, sizeof(int), (size_t)d->scan_table_len, f) != (size_t)d->scan_table_len) return 0;
        if (fread(d->scan_table_tr_id,     sizeof(int), (size_t)d->scan_table_len, f) != (size_t)d->scan_table_len) return 0;
        if (fread(d->scan_table_seg_id,    sizeof(int), (size_t)d->scan_table_len, f) != (size_t)d->scan_table_len) return 0;
        if (do_swap) {
            swap4_array(d->scan_table_block_idx, d->scan_table_len);
            swap4_array(d->scan_table_tr_id,     d->scan_table_len);
            swap4_array(d->scan_table_seg_id,    d->scan_table_len);
        }
    } else {
        d->scan_table_block_idx = NULL;
        d->scan_table_tr_id     = NULL;
        d->scan_table_seg_id    = NULL;
    }

    return 1;
}

/* ------ Write full collection to cache ------ */

static int write_cache(const char* cache_path,
                       const pulseqlib_collection* coll,
                       int seq_file_size)
{
    FILE* f;
    int marker, version, vendor, i;

    f = fopen(cache_path, "wb");
    if (!f) return 0;

    marker  = PULSEQLIB_CACHE_ENDIAN_MARKER;
    version = PULSEQLIB_CACHE_VERSION;
    vendor  = PULSEQLIB_VENDOR;

    if (!write4(f, &marker, 1))  { fclose(f); return 0; }
    if (!write4(f, &version, 1)) { fclose(f); return 0; }
    if (!write4(f, &vendor, 1))  { fclose(f); return 0; }
    if (!write4(f, &seq_file_size, 1)) { fclose(f); return 0; }

    /* collection scalars */
    if (!write4(f, &coll->num_subsequences, 1))      { fclose(f); return 0; }
    if (!write4(f, &coll->num_repetitions, 1))        { fclose(f); return 0; }
    if (!write4(f, &coll->total_unique_segments, 1))  { fclose(f); return 0; }
    if (!write4(f, &coll->total_unique_adcs, 1))      { fclose(f); return 0; }
    if (!write4(f, &coll->total_blocks, 1))           { fclose(f); return 0; }
    if (!write4(f, &coll->total_duration_us, 1))      { fclose(f); return 0; }

    /* subsequence info */
    for (i = 0; i < coll->num_subsequences; ++i) {
        if (!write4(f, &coll->subsequence_info[i].sequence_index, 1))     { fclose(f); return 0; }
        if (!write4(f, &coll->subsequence_info[i].adc_id_offset, 1))     { fclose(f); return 0; }
        if (!write4(f, &coll->subsequence_info[i].segment_id_offset, 1)) { fclose(f); return 0; }
        if (!write4(f, &coll->subsequence_info[i].block_index_offset, 1)){ fclose(f); return 0; }
    }

    /* per-subsequence descriptors */
    for (i = 0; i < coll->num_subsequences; ++i) {
        if (!write_descriptor(f, &coll->descriptors[i])) { fclose(f); return 0; }
    }

    fclose(f);
    return 1;
}

/* ------ Read full collection from cache ------ */

static int read_cache(const char* cache_path,
                      pulseqlib_collection* coll,
                      int expected_seq_file_size)
{
    FILE* f;
    int marker, version, vendor, stored_size;
    int do_swap, i;

    f = fopen(cache_path, "rb");
    if (!f) return 0;

    if (!read4(f, &marker, 1)) { fclose(f); return 0; }

    do_swap = 0;
    if (marker != PULSEQLIB_CACHE_ENDIAN_MARKER) {
        swap4(&marker);
        if (marker != PULSEQLIB_CACHE_ENDIAN_MARKER) { fclose(f); return 0; }
        do_swap = 1;
    }

    if (!read4(f, &version, 1)) { fclose(f); return 0; }
    if (do_swap) swap4(&version);
    if (version != PULSEQLIB_CACHE_VERSION) { fclose(f); return 0; }

    if (!read4(f, &vendor, 1)) { fclose(f); return 0; }
    if (do_swap) swap4(&vendor);
    if (vendor != PULSEQLIB_VENDOR) { fclose(f); return 0; }

    if (!read4(f, &stored_size, 1)) { fclose(f); return 0; }
    if (do_swap) swap4(&stored_size);
    if (stored_size != expected_seq_file_size) { fclose(f); return 0; }

    /* collection scalars */
    if (!read4(f, &coll->num_subsequences, 1))      { fclose(f); return 0; }
    if (!read4(f, &coll->num_repetitions, 1))        { fclose(f); return 0; }
    if (!read4(f, &coll->total_unique_segments, 1))  { fclose(f); return 0; }
    if (!read4(f, &coll->total_unique_adcs, 1))      { fclose(f); return 0; }
    if (!read4(f, &coll->total_blocks, 1))           { fclose(f); return 0; }
    if (!read4(f, &coll->total_duration_us, 1))      { fclose(f); return 0; }
    if (do_swap) {
        swap4(&coll->num_subsequences);
        swap4(&coll->num_repetitions);
        swap4(&coll->total_unique_segments);
        swap4(&coll->total_unique_adcs);
        swap4(&coll->total_blocks);
        swap4(&coll->total_duration_us);
    }

    /* allocate arrays */
    coll->descriptors = (pulseqlib_sequence_descriptor*)PULSEQLIB_ALLOC(
        (size_t)coll->num_subsequences * sizeof(pulseqlib_sequence_descriptor));
    coll->subsequence_info = (pulseqlib_subsequence_info*)PULSEQLIB_ALLOC(
        (size_t)coll->num_subsequences * sizeof(pulseqlib_subsequence_info));
    if (!coll->descriptors || !coll->subsequence_info) {
        if (coll->descriptors)     PULSEQLIB_FREE(coll->descriptors);
        if (coll->subsequence_info) PULSEQLIB_FREE(coll->subsequence_info);
        coll->descriptors = NULL;
        coll->subsequence_info = NULL;
        fclose(f);
        return 0;
    }

    /* subsequence info */
    for (i = 0; i < coll->num_subsequences; ++i) {
        if (!read4(f, &coll->subsequence_info[i].sequence_index, 4)) { fclose(f); return 0; }
        if (do_swap) swap4_array(&coll->subsequence_info[i].sequence_index, 4);
    }

    /* per-subsequence descriptors */
    for (i = 0; i < coll->num_subsequences; ++i) {
        if (!read_descriptor(f, &coll->descriptors[i], do_swap)) {
            /* clean up already-read descriptors */
            int j;
            for (j = 0; j < i; ++j)
                pulseqlib_sequence_descriptor_free(&coll->descriptors[j]);
            PULSEQLIB_FREE(coll->descriptors);
            PULSEQLIB_FREE(coll->subsequence_info);
            coll->descriptors = NULL;
            coll->subsequence_info = NULL;
            coll->num_subsequences = 0;
            fclose(f);
            return 0;
        }
    }

    /* init cursor */
    memset(&coll->block_cursor, 0, sizeof(coll->block_cursor));
    coll->block_cursor.scan_table_position = -1;

    fclose(f);
    return 1;
}

/* ================================================================== */
/*  Public wrappers (called from pulseqlib_core.c)                    */
/* ================================================================== */

int pulseqlib__write_cache(const pulseqlib_collection* coll,
                           const char* seq_path)
{
    char* cache_path;
    long sz;
    int ok;

    if (!coll || !seq_path) return 0;

    /* suppress unused-function warning for reserved helper */
    (void)get_seq_file_sizes;

    cache_path = make_cache_path(seq_path);
    if (!cache_path) return 0;

    sz = get_file_size(seq_path);
    if (sz < 0) { PULSEQLIB_FREE(cache_path); return 0; }

    ok = write_cache(cache_path, coll, (int)sz);
    PULSEQLIB_FREE(cache_path);
    return ok;
}

int pulseqlib__try_read_cache(pulseqlib_collection* coll,
                              const char* seq_path)
{
    char* cache_path;
    long sz;
    int ok;

    if (!coll || !seq_path) return 0;

    cache_path = make_cache_path(seq_path);
    if (!cache_path) return 0;

    sz = get_file_size(seq_path);
    if (sz < 0) { PULSEQLIB_FREE(cache_path); return 0; }

    ok = read_cache(cache_path, coll, (int)sz);
    PULSEQLIB_FREE(cache_path);
    return ok;
}

/* ================================================================== */
/*  Public API: explicit-path cache save / load                       */
/* ================================================================== */

int pulseqlib_save_cache(const pulseqlib_collection* coll,
                         const char* path,
                         int source_size)
{
    if (!coll || !path) return PULSEQLIB_ERR_NULL_POINTER;
    if (source_size <= 0) return PULSEQLIB_ERR_INVALID_ARGUMENT;
    return write_cache(path, coll, source_size)
         ? PULSEQLIB_SUCCESS : PULSEQLIB_ERR_FILE_READ_FAILED;
}

int pulseqlib_load_cache(pulseqlib_collection* coll,
                         const char* path,
                         int source_size)
{
    if (!coll || !path) return PULSEQLIB_ERR_NULL_POINTER;
    if (source_size <= 0) return PULSEQLIB_ERR_INVALID_ARGUMENT;
    return read_cache(path, coll, source_size)
         ? PULSEQLIB_SUCCESS : PULSEQLIB_ERR_FILE_READ_FAILED;
}