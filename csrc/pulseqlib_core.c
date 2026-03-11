/* pulseqlib_core.c -- descriptor lifecycle, consistency checks, collection
 *                     assembly, and public load entry point
 *
 * This file contains:
 *   - Descriptor/collection free functions
 *   - Consistency checks (RF amplitude periodicity, segment walk)
 *   - get_collection_descriptors  --  chain subsequences
 *   - pulseqlib_read              --  public entry point
 *
 * Deduplication / unique-block extraction lives in pulseqlib_dedup.c.
 * TR detection / segmentation / freq_mod lives in pulseqlib_structure.c.
 * ANSI C89.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "pulseqlib_internal.h"
#include "pulseqlib_methods.h"

/* ================================================================== */
/*  Descriptor free functions (public)                                */
/* ================================================================== */

void pulseqlib_sequence_descriptor_free(pulseqlib_sequence_descriptor* d)
{
    int i;
    if (!d) return;

    if (d->block_definitions) { PULSEQLIB_FREE(d->block_definitions); d->block_definitions = NULL; }
    d->num_unique_blocks = 0;
    if (d->block_table) { PULSEQLIB_FREE(d->block_table); d->block_table = NULL; }
    d->num_blocks = 0;

    if (d->rf_definitions) { PULSEQLIB_FREE(d->rf_definitions); d->rf_definitions = NULL; }
    d->num_unique_rfs = 0;
    if (d->rf_table) { PULSEQLIB_FREE(d->rf_table); d->rf_table = NULL; }
    d->rf_table_size = 0;

    if (d->grad_definitions) { PULSEQLIB_FREE(d->grad_definitions); d->grad_definitions = NULL; }
    d->num_unique_grads = 0;
    if (d->grad_table) { PULSEQLIB_FREE(d->grad_table); d->grad_table = NULL; }
    d->grad_table_size = 0;

    if (d->adc_definitions) { PULSEQLIB_FREE(d->adc_definitions); d->adc_definitions = NULL; }
    d->num_unique_adcs = 0;
    if (d->adc_table) { PULSEQLIB_FREE(d->adc_table); d->adc_table = NULL; }
    d->adc_table_size = 0;

    if (d->freq_mod_definitions) {
        PULSEQLIB_FREE(d->freq_mod_definitions);
        d->freq_mod_definitions = NULL;
    }
    d->num_freq_mod_defs = 0;

    if (d->rf_shim_definitions) { PULSEQLIB_FREE(d->rf_shim_definitions); d->rf_shim_definitions = NULL; }
    d->num_rf_shims = 0;

    if (d->rotation_matrices) { PULSEQLIB_FREE(d->rotation_matrices); d->rotation_matrices = NULL; }
    d->num_rotations = 0;
    if (d->trigger_events) { PULSEQLIB_FREE(d->trigger_events); d->trigger_events = NULL; }
    d->num_triggers = 0;

    if (d->shapes) {
        for (i = 0; i < d->num_shapes; ++i)
            if (d->shapes[i].samples) PULSEQLIB_FREE(d->shapes[i].samples);
        PULSEQLIB_FREE(d->shapes);
        d->shapes = NULL;
    }
    d->num_shapes = 0;

    d->num_prep_blocks    = 0;
    d->num_cooldown_blocks = 0;
    d->num_passes          = 1;

    if (d->segment_definitions) {
        for (i = 0; i < d->num_unique_segments; ++i) {
            if (d->segment_definitions[i].unique_block_indices) PULSEQLIB_FREE(d->segment_definitions[i].unique_block_indices);
            if (d->segment_definitions[i].has_digitalout)       PULSEQLIB_FREE(d->segment_definitions[i].has_digitalout);
            if (d->segment_definitions[i].has_rotation)         PULSEQLIB_FREE(d->segment_definitions[i].has_rotation);
            if (d->segment_definitions[i].norot_flag)           PULSEQLIB_FREE(d->segment_definitions[i].norot_flag);
            if (d->segment_definitions[i].nopos_flag)           PULSEQLIB_FREE(d->segment_definitions[i].nopos_flag);
            if (d->segment_definitions[i].has_freq_mod)          PULSEQLIB_FREE(d->segment_definitions[i].has_freq_mod);
            if (d->segment_definitions[i].timing.rf_anchors)    PULSEQLIB_FREE(d->segment_definitions[i].timing.rf_anchors);
            if (d->segment_definitions[i].timing.adc_anchors)   PULSEQLIB_FREE(d->segment_definitions[i].timing.adc_anchors);
            if (d->segment_definitions[i].timing.kzero_crossing_indices) PULSEQLIB_FREE(d->segment_definitions[i].timing.kzero_crossing_indices);
        }
        PULSEQLIB_FREE(d->segment_definitions);
        d->segment_definitions = NULL;
    }
    d->num_unique_segments = 0;

    pulseqlib_segment_table_result_free(&d->segment_table);

    /* Scan table arrays */
    if (d->scan_table_block_idx) { PULSEQLIB_FREE(d->scan_table_block_idx); d->scan_table_block_idx = NULL; }
    if (d->scan_table_tr_id)     { PULSEQLIB_FREE(d->scan_table_tr_id);     d->scan_table_tr_id     = NULL; }
    if (d->scan_table_seg_id)    { PULSEQLIB_FREE(d->scan_table_seg_id);    d->scan_table_seg_id    = NULL; }
    if (d->scan_table_tr_start)  { PULSEQLIB_FREE(d->scan_table_tr_start);  d->scan_table_tr_start  = NULL; }
    d->scan_table_len = 0;

    if (d->variable_grad_flags) { PULSEQLIB_FREE(d->variable_grad_flags); d->variable_grad_flags = NULL; }

    if (d->label_table) { PULSEQLIB_FREE(d->label_table); d->label_table = NULL; }
    d->label_num_columns = 0;
    d->label_num_entries = 0;
}

void pulseqlib_collection_free(
    pulseqlib_collection* c)
{
    int i;
    if (!c) return;
    if (c->descriptors) {
        for (i = 0; i < c->num_subsequences; ++i)
            pulseqlib_sequence_descriptor_free(&c->descriptors[i]);
        PULSEQLIB_FREE(c->descriptors);
    }
    if (c->subsequence_info) PULSEQLIB_FREE(c->subsequence_info);
    /* Free the struct itself (allocated by pulseqlib_read) */
    PULSEQLIB_FREE(c);
}

void pulseqlib_segment_table_result_free(pulseqlib_segment_table_result* r)
{
    if (!r) return;
    if (r->prep_segment_table)     PULSEQLIB_FREE(r->prep_segment_table);
    if (r->main_segment_table)     PULSEQLIB_FREE(r->main_segment_table);
    if (r->cooldown_segment_table) PULSEQLIB_FREE(r->cooldown_segment_table);
    r->prep_segment_table     = NULL;
    r->main_segment_table     = NULL;
    r->cooldown_segment_table = NULL;
    r->num_prep_segments      = 0;
    r->num_main_segments      = 0;
    r->num_cooldown_segments  = 0;
    r->num_unique_segments    = 0;
}

/* ================================================================== */
/*  Consistency check helpers                                         */
/* ================================================================== */

/*
 * get_block_rf_amplitude --
 *   Return the RF amplitude for block at absolute index 'block_idx',
 *   or 0 if the block has no RF.
 */
static float get_block_rf_amplitude(
    const pulseqlib_sequence_descriptor* desc,
    int block_idx)
{
    const pulseqlib_block_table_element* bte;

    bte = &desc->block_table[block_idx];
    if (bte->rf_id >= 0 && bte->rf_id < desc->rf_table_size)
        return desc->rf_table[bte->rf_id].amplitude;
    return 0.0f;
}

/*
 * get_block_rf_shim_id --
 *   Return the RF shim definition index for block at absolute index
 *   'block_idx', or -1 if the block has no RF shim.
 */
static int get_block_rf_shim_id(
    const pulseqlib_sequence_descriptor* desc,
    int block_idx)
{
    return desc->block_table[block_idx].rf_shim_id;
}

/*
 * check_rf_amplitude_periodicity --
 *   Verify that the RF amplitude pattern within a TR is identical
 *   across the "pure main" TR instances (excluding those adjacent
 *   to non-degenerate prep/cooldown).
 *
 *   ref_tr:   index of the reference TR (0-based within main TRs)
 *   first_tr: first TR index to check (inclusive)
 *   last_tr:  last TR index to check (inclusive)
 *
 *   Compares each TR in [first_tr, last_tr] against ref_tr.
 */
static int check_rf_amplitude_periodicity(
    const pulseqlib_sequence_descriptor* desc,
    int ref_tr,
    int first_tr,
    int last_tr,
    pulseqlib_diagnostic* diag)
{
    const pulseqlib_tr_descriptor* trd;
    int tr_size, prep_blocks;
    int ref_start, tr_idx, chk_start;
    int j;
    float ref_amp, chk_amp;

    trd = &desc->tr_descriptor;
    tr_size    = trd->tr_size;
    prep_blocks = trd->num_prep_blocks;

    ref_start = prep_blocks + ref_tr * tr_size;

    for (tr_idx = first_tr; tr_idx <= last_tr; ++tr_idx) {
        if (tr_idx == ref_tr) continue;
        chk_start = prep_blocks + tr_idx * tr_size;
        for (j = 0; j < tr_size; ++j) {
            ref_amp = get_block_rf_amplitude(desc, ref_start + j);
            chk_amp = get_block_rf_amplitude(desc, chk_start + j);
            if (ref_amp != chk_amp) {
                if (diag) {
                    pulseqlib__diag_printf(diag,
                        "RF periodicity: TR %d block %d has amplitude %.6g, "
                        "expected %.6g (from reference TR %d)\n",
                        tr_idx, j, (double)chk_amp,
                        (double)ref_amp, ref_tr);
                }
                return PULSEQLIB_ERR_CONSISTENCY_RF_PERIODIC;
            }
        }
    }

    return PULSEQLIB_SUCCESS;
}

/*
 * check_rf_shim_periodicity --
 *   Same logic as check_rf_amplitude_periodicity but compares
 *   rf_shim_id values instead of RF amplitudes.
 */
static int check_rf_shim_periodicity(
    const pulseqlib_sequence_descriptor* desc,
    int ref_tr,
    int first_tr,
    int last_tr,
    pulseqlib_diagnostic* diag)
{
    const pulseqlib_tr_descriptor* trd;
    int tr_size, prep_blocks;
    int ref_start, tr_idx, chk_start;
    int j;
    int ref_shim, chk_shim;

    trd = &desc->tr_descriptor;
    tr_size    = trd->tr_size;
    prep_blocks = trd->num_prep_blocks;

    ref_start = prep_blocks + ref_tr * tr_size;

    for (tr_idx = first_tr; tr_idx <= last_tr; ++tr_idx) {
        if (tr_idx == ref_tr) continue;
        chk_start = prep_blocks + tr_idx * tr_size;
        for (j = 0; j < tr_size; ++j) {
            ref_shim = get_block_rf_shim_id(desc, ref_start + j);
            chk_shim = get_block_rf_shim_id(desc, chk_start + j);
            if (ref_shim != chk_shim) {
                if (diag) {
                    pulseqlib__diag_printf(diag,
                        "RF shim periodicity: TR %d block %d has shim_id %d, "
                        "expected %d (from reference TR %d)\n",
                        tr_idx, j, chk_shim, ref_shim, ref_tr);
                }
                return PULSEQLIB_ERR_CONSISTENCY_RF_SHIM_PERIODIC;
            }
        }
    }

    return PULSEQLIB_SUCCESS;
}

/*
 * check_cross_pass_rf_consistency --
 *   For multi-pass sequences, verify that the RF amplitude and shim ID
 *   patterns are identical across passes.  Pass 0 is the reference;
 *   passes 1..N-1 are compared position-by-position.
 */
static int check_cross_pass_rf_consistency(
    const pulseqlib_sequence_descriptor* desc,
    pulseqlib_diagnostic* diag)
{
    int num_passes, pass_size, p, j;
    int ref_bt, chk_bt;
    float ref_amp, chk_amp;
    int ref_shim, chk_shim;

    num_passes = (desc->num_passes > 1) ? desc->num_passes : 1;
    if (num_passes <= 1) return PULSEQLIB_SUCCESS;

    pass_size = desc->scan_table_len / num_passes;
    if (pass_size <= 0) return PULSEQLIB_SUCCESS;

    for (p = 1; p < num_passes; ++p) {
        for (j = 0; j < pass_size; ++j) {
            ref_bt = desc->scan_table_block_idx[j];
            chk_bt = desc->scan_table_block_idx[p * pass_size + j];

            /* RF amplitude */
            ref_amp = get_block_rf_amplitude(desc, ref_bt);
            chk_amp = get_block_rf_amplitude(desc, chk_bt);
            if (ref_amp != chk_amp) {
                if (diag) {
                    pulseqlib__diag_printf(diag,
                        "Cross-pass RF amplitude mismatch: pass %d "
                        "pos %d has %.6g, pass 0 has %.6g\n",
                        p, j, (double)chk_amp, (double)ref_amp);
                }
                return PULSEQLIB_ERR_CONSISTENCY_RF_PERIODIC;
            }

            /* RF shim ID */
            ref_shim = get_block_rf_shim_id(desc, ref_bt);
            chk_shim = get_block_rf_shim_id(desc, chk_bt);
            if (ref_shim != chk_shim) {
                if (diag) {
                    pulseqlib__diag_printf(diag,
                        "Cross-pass RF shim mismatch: pass %d "
                        "pos %d has shim_id %d, pass 0 has %d\n",
                        p, j, chk_shim, ref_shim);
                }
                return PULSEQLIB_ERR_CONSISTENCY_RF_SHIM_PERIODIC;
            }
        }
    }

    return PULSEQLIB_SUCCESS;
}

/*
 * check_scan_table_segments --
 *   Walk the scan table and verify that each entry's block definition ID
 *   matches the segment definition indicated by scan_table_seg_id.
 *
 *   For each contiguous group of entries sharing the same seg_id,
 *   position within the group gives the position within the segment.
 */
static int check_scan_table_segments(
    const pulseqlib_sequence_descriptor* desc,
    pulseqlib_diagnostic* diag)
{
    int n, seg_id, prev_seg_id, pos_in_seg;
    int bt_idx, bdef_id, expected_id;
    const pulseqlib_tr_segment* seg;

    prev_seg_id = -2;  /* impossible value to force reset */
    pos_in_seg  = 0;

    for (n = 0; n < desc->scan_table_len; ++n) {
        seg_id = desc->scan_table_seg_id[n];
        if (seg_id < 0) {
            prev_seg_id = seg_id;
            pos_in_seg  = 0;
            continue;
        }
        /* Reset position when the segment type changes. */
        if (seg_id != prev_seg_id) {
            pos_in_seg  = 0;
            prev_seg_id = seg_id;
        }

        if (seg_id >= desc->segment_table.num_unique_segments) {
            if (diag) {
                pulseqlib__diag_printf(diag,
                    "Consistency: scan_table_seg_id[%d] = %d out of range "
                    "(num_unique = %d)\n",
                    n, seg_id, desc->segment_table.num_unique_segments);
            }
            return PULSEQLIB_ERR_CONSISTENCY_SEG_MISMATCH;
        }

        seg = &desc->segment_definitions[seg_id];

        /* When the same segment repeats across consecutive TRs (same
         * seg_id throughout), pos_in_seg naturally reaches num_blocks.
         * Wrap it so the next repetition is verified from UBI[0]. */
        if (pos_in_seg >= seg->num_blocks) {
            pos_in_seg = 0;
        }

        bt_idx      = desc->scan_table_block_idx[n];
        bdef_id     = desc->block_table[bt_idx].id;
        expected_id = seg->unique_block_indices[pos_in_seg];

        if (bdef_id != expected_id) {
            if (diag) {
                pulseqlib__diag_printf(diag,
                    "Consistency: scan pos %d (block_table[%d]) has def ID %d, "
                    "expected %d (segment %d, position %d)\n",
                    n, bt_idx, bdef_id, expected_id, seg_id, pos_in_seg);
            }
            return PULSEQLIB_ERR_CONSISTENCY_SEG_MISMATCH;
        }

        ++pos_in_seg;
    }
    return PULSEQLIB_SUCCESS;
}

static int check_consistency(
    const pulseqlib_collection* coll,
    pulseqlib_diagnostic* diag)
{
    int subseq_idx, rc;
    const pulseqlib_sequence_descriptor* desc;
    const pulseqlib_tr_descriptor* trd;
    int ref_tr, first_check, last_check;

    if (!coll) return PULSEQLIB_ERR_NULL_POINTER;

    for (subseq_idx = 0; subseq_idx < coll->num_subsequences; ++subseq_idx) {
        desc = &coll->descriptors[subseq_idx];
        trd  = &desc->tr_descriptor;

        /* (a) Scan-table segment consistency: walk the scan table and
         *     verify that each entry's block definition ID matches what
         *     its seg_id expects. */
        if (desc->scan_table_len > 0 && desc->scan_table_seg_id) {
            rc = check_scan_table_segments(desc, diag);
            if (PULSEQLIB_FAILED(rc)) {
                if (diag) {
                    pulseqlib__diag_printf(diag,
                        "Segment consistency check failed "
                        "in subsequence %d\n", subseq_idx);
                }
                return rc;
            }
        }

        /* (b) RF amplitude periodicity across pure main TRs.
         *
         *   prep degen   cooldown degen   ref   check range
         *   ----------   --------------   ---   -----------
         *   yes          yes              0     1 .. num_trs-1
         *   no           yes              1     2 .. num_trs-1
         *   yes          no               0     1 .. num_trs-2
         *   no           no               1     2 .. num_trs-2
         */
        if (trd->num_trs > 1) {
            ref_tr      = trd->degenerate_prep ? 0 : 1;
            first_check = ref_tr + 1;
            last_check  = trd->degenerate_cooldown
                        ? trd->num_trs - 1
                        : trd->num_trs - 2;

            if (first_check <= last_check) {
                rc = check_rf_amplitude_periodicity(desc,
                    ref_tr, first_check, last_check, diag);
                if (PULSEQLIB_FAILED(rc)) {
                    if (diag) {
                        pulseqlib__diag_printf(diag,
                            "Consistency check failed: RF amplitude "
                            "not periodic in subsequence %d\n",
                            subseq_idx);
                    }
                    return rc;
                }

                /* (c) RF shim ID periodicity (same TR range as amplitude) */
                rc = check_rf_shim_periodicity(desc,
                    ref_tr, first_check, last_check, diag);
                if (PULSEQLIB_FAILED(rc)) {
                    if (diag) {
                        pulseqlib__diag_printf(diag,
                            "Consistency check failed: RF shim ID "
                            "not periodic in subsequence %d\n",
                            subseq_idx);
                    }
                    return rc;
                }
            }
        }

        /* (d) Cross-pass RF amplitude + shim ID consistency */
        if (desc->num_passes > 1) {
            rc = check_cross_pass_rf_consistency(desc, diag);
            if (PULSEQLIB_FAILED(rc)) {
                if (diag) {
                    pulseqlib__diag_printf(diag,
                        "Consistency check failed: cross-pass RF "
                        "mismatch in subsequence %d\n",
                        subseq_idx);
                }
                return rc;
            }
        }
    }

    return PULSEQLIB_SUCCESS;
}

/* Public wrapper around check_consistency */
int pulseqlib_check_consistency(
    const pulseqlib_collection* coll,
    pulseqlib_diagnostic* diag)
{
    pulseqlib_diagnostic local_diag;
    if (!diag) {
        pulseqlib_diagnostic_init(&local_diag);
        diag = &local_diag;
    }
    return check_consistency(coll, diag);
}

/* ================================================================== */
/*  Error formatting convenience function                             */
/* ================================================================== */

int pulseqlib_format_error(
    char* buf, int buf_size,
    int code,
    const pulseqlib_diagnostic* diag)
{
    const char* msg;
    const char* hint;
    int written;

    if (!buf || buf_size <= 0) return 0;
    buf[0] = '\0';

    msg  = pulseqlib_get_error_message(code);
    hint = pulseqlib_get_error_hint(code);

    /* Build the string with sprintf; caller must provide >= 512 bytes.
     * We guard against overrun by checking buf_size, but the assembled
     * string is never longer than ~380 chars (msg + hint + diag). */
    if (diag && diag->message[0] != '\0') {
        if (buf_size < 512) { buf[0] = '\0'; return 0; }
        written = sprintf(buf, "%s (%s)", msg, diag->message);
    } else if (hint && hint[0] != '\0') {
        if (buf_size < 256) { buf[0] = '\0'; return 0; }
        written = sprintf(buf, "%s (%s)", msg, hint);
    } else {
        if (buf_size < 128) { buf[0] = '\0'; return 0; }
        written = sprintf(buf, "%s", msg);
    }
    if (written < 0) written = 0;
    return written;
}

/* ================================================================== */
/*  get_collection_descriptors                                        */
/* ================================================================== */

int pulseqlib__get_collection_descriptors(
    pulseqlib_collection* coll,
    pulseqlib_diagnostic* diag,
    const pulseqlib__seq_file_collection* raw,
    int parse_labels,
    int num_averages)
{
    int i, j, result;
    int adc_off = 0, seg_off = 0, blk_off = 0;
    pulseqlib_diagnostic local_diag;

    if (!diag) { pulseqlib_diagnostic_init(&local_diag); diag = &local_diag; }

    if (!raw || !coll) { diag->code = PULSEQLIB_ERR_NULL_POINTER; return 0; }
    if (raw->num_sequences == 0) { diag->code = PULSEQLIB_ERR_COLLECTION_EMPTY; return 0; }

    coll->descriptors = (pulseqlib_sequence_descriptor*)PULSEQLIB_ALLOC(
        raw->num_sequences * sizeof(pulseqlib_sequence_descriptor));
    coll->subsequence_info = (pulseqlib_subsequence_info*)PULSEQLIB_ALLOC(
        raw->num_sequences * sizeof(pulseqlib_subsequence_info));
    if (!coll->descriptors || !coll->subsequence_info) {
        if (coll->descriptors)     PULSEQLIB_FREE(coll->descriptors);
        if (coll->subsequence_info) PULSEQLIB_FREE(coll->subsequence_info);
        coll->descriptors = NULL;
        coll->subsequence_info = NULL;
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        return 0;
    }

    coll->num_subsequences    = raw->num_sequences;
    coll->total_duration_us   = 0.0f;
    coll->total_unique_segments = 0;
    coll->total_unique_adcs   = 0;
    coll->total_blocks        = 0;

    for (i = 0; i < raw->num_sequences; ++i) {
        pulseqlib_sequence_descriptor desc = PULSEQLIB_SEQUENCE_DESCRIPTOR_INIT;

        coll->subsequence_info[i].sequence_index     = i;
        coll->subsequence_info[i].adc_id_offset      = adc_off;
        coll->subsequence_info[i].segment_id_offset  = seg_off;
        coll->subsequence_info[i].block_index_offset = blk_off;

        result = pulseqlib__get_unique_blocks(&desc, &raw->sequences[i]);
        if (PULSEQLIB_FAILED(result)) { diag->code = result; goto fail; }

        result = pulseqlib__get_tr_in_sequence(&desc, diag);
        if (PULSEQLIB_FAILED(diag->code)) goto fail;

        result = pulseqlib__compute_variable_grad_flags(&desc);
        if (PULSEQLIB_FAILED(result)) { diag->code = result; goto fail; }

        result = pulseqlib__build_scan_table(&desc, num_averages, diag);
        if (PULSEQLIB_FAILED(diag->code)) goto fail;

        /* Scan-table-only segmentation (prep / main / cooldown) */
        result = pulseqlib__get_scan_table_segments(&desc, diag, &raw->sequences[i].opts);
        if (PULSEQLIB_FAILED(diag->code)) goto fail;

        result = pulseqlib__calc_segment_timing(&desc, diag);
        if (PULSEQLIB_FAILED(result)) { diag->code = result; goto fail; }

        pulseqlib__compute_scan_table_tr_start(&desc);

        result = pulseqlib__build_freq_mod_flags(&desc);
        if (PULSEQLIB_FAILED(result)) { diag->code = result; goto fail; }

        if (parse_labels) {
            result = pulseqlib__build_label_table(&desc, &raw->sequences[i]);
            if (PULSEQLIB_FAILED(result)) { diag->code = result; goto fail; }
        }

        /* apply offsets */
        if (seg_off > 0) {
            for (j = 0; j < desc.segment_table.num_prep_segments; ++j)
                desc.segment_table.prep_segment_table[j] += seg_off;
            for (j = 0; j < desc.segment_table.num_main_segments; ++j)
                desc.segment_table.main_segment_table[j] += seg_off;
            for (j = 0; j < desc.segment_table.num_cooldown_segments; ++j)
                desc.segment_table.cooldown_segment_table[j] += seg_off;
        }
        if (adc_off > 0) {
            for (j = 0; j < desc.adc_table_size; ++j)
                desc.adc_table[j].id += adc_off;
            for (j = 0; j < desc.num_unique_adcs; ++j)
                desc.adc_definitions[j].id += adc_off;
        }

        adc_off += desc.num_unique_adcs;
        seg_off += desc.num_unique_segments;
        blk_off += desc.num_blocks;

        coll->total_duration_us += desc.tr_descriptor.tr_duration_us *
                                   desc.tr_descriptor.num_trs;

        coll->descriptors[i] = desc;
    }

    coll->total_unique_segments = seg_off;
    coll->total_unique_adcs     = adc_off;
    coll->total_blocks          = blk_off;
    diag->code = PULSEQLIB_SUCCESS;
    return raw->num_sequences;

fail:
    for (j = 0; j < i; ++j)
        pulseqlib_sequence_descriptor_free(&coll->descriptors[j]);
    PULSEQLIB_FREE(coll->descriptors);
    PULSEQLIB_FREE(coll->subsequence_info);
    coll->descriptors      = NULL;
    coll->subsequence_info = NULL;
    coll->num_subsequences = 0;
    return 0;
}

/* ================================================================== */
/*  pulseqlib_read (public entry point)                               */
/* ================================================================== */

int pulseqlib_read(
    pulseqlib_collection** out_coll,
    pulseqlib_diagnostic* diag,
    const char* file_path,
    const pulseqlib_opts* opts,
    int cache_binary,
    int verify_signature,
    int parse_labels,
    int num_averages)
{
    pulseqlib__seq_file_collection raw_coll;
    pulseqlib_collection* collection;
    int rc, i;

    raw_coll.num_sequences = 0;
    raw_coll.sequences     = NULL;
    raw_coll.base_path     = NULL;

    if (!file_path || !opts || !out_coll || !diag)
        return PULSEQLIB_ERR_NULL_POINTER;

    *out_coll = NULL;
    pulseqlib_diagnostic_init(diag);

    /* Heap-allocate the opaque collection */
    collection = (pulseqlib_collection*)PULSEQLIB_ALLOC(sizeof(pulseqlib_collection));
    if (!collection) return PULSEQLIB_ERR_ALLOC_FAILED;
    memset(collection, 0, sizeof(*collection));
    collection->block_cursor.scan_table_position = -1;
    collection->num_repetitions = 1;

    /* Try cache */
    if (cache_binary && pulseqlib__try_read_cache(collection, file_path)) {
        /* Segment timing and TR-start flags are derived, not cached */
        for (i = 0; i < collection->num_subsequences; ++i) {
            pulseqlib__calc_segment_timing(&collection->descriptors[i], NULL);
            pulseqlib__compute_scan_table_tr_start(&collection->descriptors[i]);
        }
        *out_coll = collection;
        return PULSEQLIB_SUCCESS;
    }

    /* Full parse */
    rc = pulseqlib__read_seq_collection(&raw_coll, file_path, opts);
    if (PULSEQLIB_FAILED(rc)) { diag->code = rc; goto fail; }

    /* Optional MD5 signature verification (all files in chain) */
    if (verify_signature) {
        for (i = 0; i < raw_coll.num_sequences; ++i) {
            const char* fpath = raw_coll.sequences[i].file_path;
            if (!fpath) continue;
            rc = pulseqlib__verify_signature(fpath);
            if (PULSEQLIB_FAILED(rc)) {
                diag->code = rc;
                pulseqlib__diag_printf(diag, " subsequence=%d", i);
                goto fail;
            }
        }
    }

    rc = pulseqlib__get_collection_descriptors(collection, diag, &raw_coll, parse_labels, num_averages);
    if (PULSEQLIB_FAILED(diag->code)) { rc = diag->code; goto fail; }

    rc = check_consistency(collection, diag);
    if (PULSEQLIB_FAILED(rc)) { diag->code = rc; goto fail; }

    pulseqlib__seq_file_collection_free(&raw_coll);

    /* Write cache (best-effort) */
    if (cache_binary) pulseqlib__write_cache(collection, file_path);

    *out_coll = collection;
    return PULSEQLIB_SUCCESS;

fail:
    pulseqlib__seq_file_collection_free(&raw_coll);
    PULSEQLIB_FREE(collection);
    return rc;
}

/* ================================================================== */
/*  pulseqlib_read_from_buffers (public entry point)                  */
/* ================================================================== */

int pulseqlib_read_from_buffers(
    pulseqlib_collection** out_coll,
    pulseqlib_diagnostic* diag,
    const char* const* buffers,
    const int* buffer_sizes,
    int num_buffers,
    const pulseqlib_opts* opts,
    int parse_labels,
    int num_averages)
{
    pulseqlib__seq_file_collection raw_coll;
    pulseqlib_collection* collection;
    int rc, i;

    raw_coll.num_sequences = 0;
    raw_coll.sequences     = NULL;
    raw_coll.base_path     = NULL;

    if (!out_coll || !diag || !buffers || !buffer_sizes || !opts)
        return PULSEQLIB_ERR_NULL_POINTER;
    if (num_buffers < 1) return PULSEQLIB_ERR_INVALID_ARGUMENT;

    *out_coll = NULL;
    pulseqlib_diagnostic_init(diag);

    /* Build raw collection from in-memory buffers */
    raw_coll.sequences = (pulseqlib__seq_file*)PULSEQLIB_ALLOC(
        num_buffers * sizeof(pulseqlib__seq_file));
    if (!raw_coll.sequences) return PULSEQLIB_ERR_ALLOC_FAILED;
    raw_coll.num_sequences = 0;
    raw_coll.base_path     = NULL;

    for (i = 0; i < num_buffers; ++i) {
        FILE* tmp;
        if (!buffers[i] || buffer_sizes[i] < 0) {
            rc = PULSEQLIB_ERR_INVALID_ARGUMENT;
            diag->code = rc;
            goto fail_raw;
        }

        tmp = tmpfile();
        if (!tmp) { rc = PULSEQLIB_ERR_FILE_NOT_FOUND; diag->code = rc; goto fail_raw; }

        if (buffer_sizes[i] > 0) {
            if ((int)fwrite(buffers[i], 1, (size_t)buffer_sizes[i], tmp)
                != buffer_sizes[i]) {
                fclose(tmp);
                rc = PULSEQLIB_ERR_FILE_NOT_FOUND;
                diag->code = rc;
                goto fail_raw;
            }
        }
        rewind(tmp);

        pulseqlib__seq_file_init(&raw_coll.sequences[i], opts);
        rc = pulseqlib__read_seq_from_buffer(&raw_coll.sequences[i], tmp);
        fclose(tmp);
        if (PULSEQLIB_FAILED(rc)) { diag->code = rc; goto fail_raw; }
        raw_coll.num_sequences = i + 1;
    }

    /* Heap-allocate the opaque collection */
    collection = (pulseqlib_collection*)PULSEQLIB_ALLOC(sizeof(pulseqlib_collection));
    if (!collection) { rc = PULSEQLIB_ERR_ALLOC_FAILED; goto fail_raw; }
    memset(collection, 0, sizeof(*collection));
    collection->block_cursor.scan_table_position = -1;
    collection->num_repetitions = 1;

    rc = pulseqlib__get_collection_descriptors(collection, diag, &raw_coll, parse_labels, num_averages);
    if (PULSEQLIB_FAILED(diag->code)) { rc = diag->code; goto fail_coll; }

    rc = check_consistency(collection, diag);
    if (PULSEQLIB_FAILED(rc)) { diag->code = rc; goto fail_coll; }

    pulseqlib__seq_file_collection_free(&raw_coll);

    *out_coll = collection;
    return PULSEQLIB_SUCCESS;

fail_coll:
    PULSEQLIB_FREE(collection);
fail_raw:
    pulseqlib__seq_file_collection_free(&raw_coll);
    return rc;
}