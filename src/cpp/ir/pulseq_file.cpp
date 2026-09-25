/**
 * @file pulseq_file.cpp
 * @brief The pulseq_file the conversion is fed, and the accessors over it.
 *
 * A file is filled from the libraries a sequence was read into (see
 * from_libraries.cpp); what is here is its lifecycle and the two accessors
 * the passes resolve a block through -- the content ids a block names, and
 * the trigger its extension chain carries.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* The conversion calls these by the names pulseq.h declares. */
extern "C"
{
#include "pulseq_internal.h"
#include "pulseq_file.h"
}

/* Every library starts absent: no rows, no count, and not read. */
#define INIT_LIBRARY(seq, field_ptr, size_field, flag_field) \
    do \
    { \
        (seq)->field_ptr = NULL; \
        (seq)->size_field = 0; \
        (seq)->flag_field = 0; \
    } while (0)

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
    seq->block_rotations = NULL;
    seq->block_shims = NULL;
    seq->block_flags = NULL;
    seq->block_trid_set = NULL;
    seq->num_adc_labels = 0;
    seq->adc_labels = NULL;
    INIT_LIBRARY(seq, rf_library, rf_library_size, is_rf_library_parsed);
    seq->rf_use_tags = NULL;
    seq->rf_spectra = NULL;
    seq->rf_flip_deg = NULL;
    seq->rf_channels = NULL;
    seq->rf_b1sq_integral = NULL;
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
        PULSEQ_FREE(seq->block_rotations);
        PULSEQ_FREE(seq->block_shims);
        PULSEQ_FREE(seq->block_flags);
        PULSEQ_FREE(seq->block_trid_set);
        PULSEQ_FREE(seq->adc_labels);
    }
    if (seq->is_rf_library_parsed)
    {
        PULSEQ_FREE(seq->rf_library);
        if (seq->rf_use_tags)
            PULSEQ_FREE(seq->rf_use_tags);
        seq->rf_use_tags = NULL;
        if (seq->rf_spectra)
            PULSEQ_FREE(seq->rf_spectra);
        seq->rf_spectra = NULL;
        if (seq->rf_flip_deg)
            PULSEQ_FREE(seq->rf_flip_deg);
        seq->rf_flip_deg = NULL;
        if (seq->rf_channels)
            PULSEQ_FREE(seq->rf_channels);
        seq->rf_channels = NULL;
        if (seq->rf_b1sq_integral)
            PULSEQ_FREE(seq->rf_b1sq_integral);
        seq->rf_b1sq_integral = NULL;
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
            if (!seq->shapes_borrowed)
                PULSEQ_FREE(seq->shapes_library[i].samples);
            seq->shapes_library[i].samples = NULL;
            seq->shapes_library[i].num_uncompressed_samples = 0;
            seq->shapes_library[i].num_samples = 0;
        }
        PULSEQ_FREE(seq->shapes_library);
    }
    seq->shapes_borrowed = 0;
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

int pulseq_block_trigger(const pulseq_file *seq, const pulseq_raw_block *raw)
{
    int i, type_idx, found = -1;

    if (!seq || !raw || !seq->is_extensions_library_parsed || !seq->extension_lut)
        return -1;
    for (i = 0; i < raw->ext_count; ++i)
    {
        type_idx = raw->ext[i][0];
        if (type_idx < 0 || type_idx > seq->extension_lut_size || raw->ext[i][1] < 0)
            continue;
        if (seq->extension_lut[type_idx] == PULSEQ_EXT_TRIGGER)
            found = raw->ext[i][1];
    }
    return found;
}
