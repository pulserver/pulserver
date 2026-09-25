/**
 * @file pulseq_file.cpp
 * @brief The pulseq_file the conversion is fed, and the accessors over it.
 *
 * A file is filled from the libraries a sequence was read into (see
 * from_libraries.cpp); what is here is its lifecycle and the two accessors
 * the passes resolve a block through -- the content ids a block names, and
 * the extension chain it carries.
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
