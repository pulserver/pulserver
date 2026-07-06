/* pulseg_core.c -- descriptor lifecycle, consistency checks, collection
 *                     assembly, and public load entry point
 *
 * This file contains:
 *   - Descriptor/collection free functions
 *   - Consistency checks (RF amplitude periodicity, segment walk)
 *   - get_collection_descriptors  --  chain subsequences
 *   - pulseg_read              --  public entry point
 *
 * Deduplication / unique-block extraction lives in pulseg_dedup.c.
 * TR detection / segmentation / freq_mod lives in pulseg_structure.c.
 * ANSI C89.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "pulseg_internal.h"
#include "pulseg.h"

/* ================================================================== */
/*  Descriptor free functions (public)                                */
/* ================================================================== */

void pulseg_sequence_descriptor_free(pulseg_sequence_descriptor* d)
{
    int i;
    if (!d) return;

    if (d->block_definitions) { PULSEG_FREE(d->block_definitions); d->block_definitions = NULL; }
    d->num_unique_blocks = 0;
    if (d->block_table) { PULSEG_FREE(d->block_table); d->block_table = NULL; }
    d->num_blocks = 0;

    if (d->rf_definitions) { PULSEG_FREE(d->rf_definitions); d->rf_definitions = NULL; }
    d->num_unique_rfs = 0;
    if (d->rf_table) { PULSEG_FREE(d->rf_table); d->rf_table = NULL; }
    d->rf_table_size = 0;

    if (d->grad_definitions) { PULSEG_FREE(d->grad_definitions); d->grad_definitions = NULL; }
    d->num_unique_grads = 0;
    if (d->grad_table) { PULSEG_FREE(d->grad_table); d->grad_table = NULL; }
    d->grad_table_size = 0;

    if (d->adc_definitions) { PULSEG_FREE(d->adc_definitions); d->adc_definitions = NULL; }
    d->num_unique_adcs = 0;
    if (d->adc_table) { PULSEG_FREE(d->adc_table); d->adc_table = NULL; }
    d->adc_table_size = 0;

    if (d->freq_mod_definitions) {
        PULSEG_FREE(d->freq_mod_definitions);
        d->freq_mod_definitions = NULL;
    }
    d->num_freq_mod_defs = 0;

    if (d->rf_shim_definitions) { PULSEG_FREE(d->rf_shim_definitions); d->rf_shim_definitions = NULL; }
    d->num_rf_shims = 0;

    if (d->rotation_matrices) { PULSEG_FREE(d->rotation_matrices); d->rotation_matrices = NULL; }
    d->num_rotations = 0;
    if (d->trigger_events) { PULSEG_FREE(d->trigger_events); d->trigger_events = NULL; }
    d->num_triggers = 0;

    if (d->shapes) {
        for (i = 0; i < d->num_shapes; ++i)
            if (d->shapes[i].samples) PULSEG_FREE(d->shapes[i].samples);
        PULSEG_FREE(d->shapes);
        d->shapes = NULL;
    }
    d->num_shapes = 0;

    d->num_prep_blocks    = 0;
    d->num_cooldown_blocks = 0;
    d->num_passes          = 1;

    if (d->segment_definitions) {
        for (i = 0; i < d->num_unique_segments; ++i) {
            if (d->segment_definitions[i].unique_block_indices) PULSEG_FREE(d->segment_definitions[i].unique_block_indices);
            if (d->segment_definitions[i].has_digitalout)       PULSEG_FREE(d->segment_definitions[i].has_digitalout);
            if (d->segment_definitions[i].has_rotation)         PULSEG_FREE(d->segment_definitions[i].has_rotation);
            if (d->segment_definitions[i].norot_flag)           PULSEG_FREE(d->segment_definitions[i].norot_flag);
            if (d->segment_definitions[i].nopos_flag)           PULSEG_FREE(d->segment_definitions[i].nopos_flag);
            if (d->segment_definitions[i].has_freq_mod)          PULSEG_FREE(d->segment_definitions[i].has_freq_mod);
            if (d->segment_definitions[i].has_adc)               PULSEG_FREE(d->segment_definitions[i].has_adc);
            if (d->segment_definitions[i].timing.rf_anchors)    PULSEG_FREE(d->segment_definitions[i].timing.rf_anchors);
            if (d->segment_definitions[i].timing.adc_anchors)   PULSEG_FREE(d->segment_definitions[i].timing.adc_anchors);
            if (d->segment_definitions[i].timing.kzero_crossing_indices) PULSEG_FREE(d->segment_definitions[i].timing.kzero_crossing_indices);
        }
        PULSEG_FREE(d->segment_definitions);
        d->segment_definitions = NULL;
    }
    d->num_unique_segments = 0;

    pulseg_segment_table_result_free(&d->segment_table);

    /* Scan table arrays */
    if (d->scan_table_block_idx) { PULSEG_FREE(d->scan_table_block_idx); d->scan_table_block_idx = NULL; }
    if (d->scan_table_tr_id)     { PULSEG_FREE(d->scan_table_tr_id);     d->scan_table_tr_id     = NULL; }
    if (d->scan_table_seg_id)    { PULSEG_FREE(d->scan_table_seg_id);    d->scan_table_seg_id    = NULL; }
    if (d->scan_table_avg_id)    { PULSEG_FREE(d->scan_table_avg_id);    d->scan_table_avg_id    = NULL; }
    if (d->scan_table_tr_start)  { PULSEG_FREE(d->scan_table_tr_start);  d->scan_table_tr_start  = NULL; }
    d->scan_table_len = 0;

    if (d->variable_grad_flags) { PULSEG_FREE(d->variable_grad_flags); d->variable_grad_flags = NULL; }

    if (d->label_table) { PULSEG_FREE(d->label_table); d->label_table = NULL; }
    if (d->off_table)   { PULSEG_FREE(d->off_table);   d->off_table   = NULL; }
    d->label_num_columns = 0;
    d->label_num_entries = 0;

    /* generic definitions */
    if (d->definitions) {
        for (i = 0; i < d->num_definitions; ++i) {
            int j;
            for (j = 0; j < d->definitions[i].value_size; ++j)
                if (d->definitions[i].value[j]) PULSEG_FREE(d->definitions[i].value[j]);
            if (d->definitions[i].value) PULSEG_FREE(d->definitions[i].value);
        }
        PULSEG_FREE(d->definitions);
        d->definitions = NULL;
    }
    d->num_definitions = 0;

    if (d->canonical_kx) { PULSEG_FREE(d->canonical_kx); d->canonical_kx = NULL; }
    if (d->canonical_ky) { PULSEG_FREE(d->canonical_ky); d->canonical_ky = NULL; }
    if (d->canonical_kz) { PULSEG_FREE(d->canonical_kz); d->canonical_kz = NULL; }
    d->has_canonical_kspace = 0;
    d->canonical_kspace_num_samples = 0;
    d->canonical_kspace_dt_us = 0.0f;
}

void pulseg_collection_free(
    pulseg_collection* c)
{
    int i;
    if (!c) return;
    if (c->freq_mod) {
        pulseg_freq_mod_collection_free(c->freq_mod);
        c->freq_mod = NULL;
    }
    if (c->descriptors) {
        for (i = 0; i < c->num_subsequences; ++i)
            pulseg_sequence_descriptor_free(&c->descriptors[i]);
        PULSEG_FREE(c->descriptors);
    }
    if (c->subsequence_info) PULSEG_FREE(c->subsequence_info);
    /* Free the struct itself (allocated by pulseg_read) */
    PULSEG_FREE(c);
}

pulseg_collection* pulseg_collection_alloc(void)
{
    pulseg_collection* c = (pulseg_collection*)PULSEG_ALLOC(sizeof(pulseg_collection));
    if (!c) return NULL;
    memset(c, 0, sizeof(*c));
    c->block_cursor.scan_table_position = -1;
    c->num_repetitions = 1;
    return c;
}

void pulseg_segment_table_result_free(pulseg_segment_table_result* r)
{
    if (!r) return;
    if (r->prep_segment_table)     PULSEG_FREE(r->prep_segment_table);
    if (r->main_segment_table)     PULSEG_FREE(r->main_segment_table);
    if (r->cooldown_segment_table) PULSEG_FREE(r->cooldown_segment_table);
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
    const pulseg_sequence_descriptor* desc,
    int block_idx)
{
    const pulseg_block_table_element* bte;

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
    const pulseg_sequence_descriptor* desc,
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
    const pulseg_sequence_descriptor* desc,
    int ref_tr,
    int first_tr,
    int last_tr,
    pulseg_diagnostic* diag)
{
    const pulseg_tr_descriptor* trd;
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
                    pulseg__diag_printf(diag,
                        "RF periodicity: TR %d block %d has amplitude %.6g, "
                        "expected %.6g (from reference TR %d)\n",
                        tr_idx, j, (double)chk_amp,
                        (double)ref_amp, ref_tr);
                }
                return PULSEG_ERR_CONSISTENCY_RF_PERIODIC;
            }
        }
    }

    return PULSEG_SUCCESS;
}

/*
 * check_rf_shim_periodicity --
 *   Same logic as check_rf_amplitude_periodicity but compares
 *   rf_shim_id values instead of RF amplitudes.
 */
static int check_rf_shim_periodicity(
    const pulseg_sequence_descriptor* desc,
    int ref_tr,
    int first_tr,
    int last_tr,
    pulseg_diagnostic* diag)
{
    const pulseg_tr_descriptor* trd;
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
                    pulseg__diag_printf(diag,
                        "RF shim periodicity: TR %d block %d has shim_id %d, "
                        "expected %d (from reference TR %d)\n",
                        tr_idx, j, chk_shim, ref_shim, ref_tr);
                }
                return PULSEG_ERR_CONSISTENCY_RF_SHIM_PERIODIC;
            }
        }
    }

    return PULSEG_SUCCESS;
}

/*
 * check_cross_pass_rf_consistency --
 *   Verify that scan-table-expanded pass waveforms have identical RF
 *   amplitude and shim-ID patterns. For non-degenerate prep/cooldown
 *   subsequences, one expanded pass is the canonical RF unit.
 */
static int check_cross_pass_rf_consistency(
    const pulseg_sequence_descriptor* desc,
    pulseg_diagnostic* diag)
{
    int num_passes, pass_size, p, j;
    int ref_bt, chk_bt;
    float ref_amp, chk_amp;
    int ref_shim, chk_shim;

    num_passes = (desc->num_passes > 1) ? desc->num_passes : 1;
    if (num_passes <= 1) return PULSEG_SUCCESS;

    pass_size = desc->scan_table_len / num_passes;
    if (pass_size <= 0) return PULSEG_SUCCESS;

    for (p = 1; p < num_passes; ++p) {
        for (j = 0; j < pass_size; ++j) {
            ref_bt = desc->scan_table_block_idx[j];
            chk_bt = desc->scan_table_block_idx[p * pass_size + j];

            /* RF amplitude */
            ref_amp = get_block_rf_amplitude(desc, ref_bt);
            chk_amp = get_block_rf_amplitude(desc, chk_bt);
            if (ref_amp != chk_amp) {
                if (diag) {
                    pulseg__diag_printf(diag,
                        "Cross-pass RF amplitude mismatch: pass %d "
                        "pos %d has %.6g, pass 0 has %.6g\n",
                        p, j, (double)chk_amp, (double)ref_amp);
                }
                return PULSEG_ERR_CONSISTENCY_RF_PERIODIC;
            }

            /* RF shim ID */
            ref_shim = get_block_rf_shim_id(desc, ref_bt);
            chk_shim = get_block_rf_shim_id(desc, chk_bt);
            if (ref_shim != chk_shim) {
                if (diag) {
                    pulseg__diag_printf(diag,
                        "Cross-pass RF shim mismatch: pass %d "
                        "pos %d has shim_id %d, pass 0 has %d\n",
                        p, j, chk_shim, ref_shim);
                }
                return PULSEG_ERR_CONSISTENCY_RF_SHIM_PERIODIC;
            }
        }
    }

    return PULSEG_SUCCESS;
}

static int block_defs_structurally_equal_core(
    const pulseg_sequence_descriptor* desc,
    int id_a,
    int id_b)
{
    const pulseg_block_definition* a;
    const pulseg_block_definition* b;

    if (!desc) return 0;
    if (id_a < 0 || id_a >= desc->num_unique_blocks) return 0;
    if (id_b < 0 || id_b >= desc->num_unique_blocks) return 0;

    a = &desc->block_definitions[id_a];
    b = &desc->block_definitions[id_b];

    if (a->duration_us != b->duration_us) return 0;
    if ((a->rf_id  >= 0) != (b->rf_id  >= 0)) return 0;
    if ((a->gx_id  >= 0) != (b->gx_id  >= 0)) return 0;
    if ((a->gy_id  >= 0) != (b->gy_id  >= 0)) return 0;
    if ((a->gz_id  >= 0) != (b->gz_id  >= 0)) return 0;
    if ((a->adc_id >= 0) != (b->adc_id >= 0)) return 0;
    return 1;
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
    const pulseg_sequence_descriptor* desc,
    pulseg_diagnostic* diag)
{
    int n, seg_id, prev_seg_id, pos_in_seg;
    int bt_idx, bdef_id, expected_id;
    const pulseg_tr_segment* seg;
    const pulseg_block_definition* bdef_actual;
    const pulseg_block_definition* bdef_expected;
    int both_pure_delay;
    int structural_match;

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
                pulseg__diag_printf(diag,
                    "Consistency: scan_table_seg_id[%d] = %d out of range "
                    "(num_unique = %d)\n",
                    n, seg_id, desc->segment_table.num_unique_segments);
            }
            return PULSEG_ERR_CONSISTENCY_SEG_MISMATCH;
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

        both_pure_delay = 0;
        structural_match = 0;
        if (bdef_id >= 0 && bdef_id < desc->num_unique_blocks &&
            expected_id >= 0 && expected_id < desc->num_unique_blocks) {
            bdef_actual   = &desc->block_definitions[bdef_id];
            bdef_expected = &desc->block_definitions[expected_id];
            both_pure_delay =
                (bdef_actual->rf_id  == -1 && bdef_actual->gx_id  == -1 &&
                 bdef_actual->gy_id  == -1 && bdef_actual->gz_id  == -1 &&
                 bdef_actual->adc_id == -1 &&
                 bdef_expected->rf_id  == -1 && bdef_expected->gx_id  == -1 &&
                 bdef_expected->gy_id  == -1 && bdef_expected->gz_id  == -1 &&
                 bdef_expected->adc_id == -1) ? 1 : 0;
            if (!both_pure_delay &&
                desc->tr_descriptor.num_prep_blocks == 0 &&
                desc->tr_descriptor.num_cooldown_blocks == 0) {
                structural_match = block_defs_structurally_equal_core(
                    desc, bdef_id, expected_id);
            }
        }

        if (bdef_id != expected_id && !both_pure_delay && !structural_match) {
            if (diag) {
                pulseg__diag_printf(diag,
                    "Consistency: scan pos %d (block_table[%d]) has def ID %d, "
                    "expected %d (segment %d, position %d)\n",
                    n, bt_idx, bdef_id, expected_id, seg_id, pos_in_seg);
            }
            return PULSEG_ERR_CONSISTENCY_SEG_MISMATCH;
        }

        ++pos_in_seg;
    }
    return PULSEG_SUCCESS;
}

static int check_consistency(
    const pulseg_collection* coll,
    pulseg_diagnostic* diag)
{
    int subseq_idx, rc;
    const pulseg_sequence_descriptor* desc;
    const pulseg_tr_descriptor* trd;
    int ref_tr, first_check, last_check;
    int has_nd_prep, has_nd_cool;

    if (!coll) return PULSEG_ERR_NULL_POINTER;

    for (subseq_idx = 0; subseq_idx < coll->num_subsequences; ++subseq_idx) {
        desc = &coll->descriptors[subseq_idx];
        trd  = &desc->tr_descriptor;
        has_nd_prep = (trd->num_prep_blocks > 0 && !trd->degenerate_prep);
        has_nd_cool = (trd->num_cooldown_blocks > 0 && !trd->degenerate_cooldown);

        /* (a) Scan-table segment consistency: walk the scan table and
         *     verify that each entry's block definition ID matches what
         *     its seg_id expects. */
        if (desc->scan_table_len > 0 && desc->scan_table_seg_id) {
            rc = check_scan_table_segments(desc, diag);
            if (PULSEG_FAILED(rc)) {
                if (diag) {
                    pulseg__diag_printf(diag,
                        "Segment consistency check failed "
                        "in subsequence %d\n", subseq_idx);
                }
                return rc;
            }
        }

        /* (b) RF periodicity on canonical units.
         *
         * Standard / degenerate subsequences use imaging TRs as the
         * canonical RF unit. Non-degenerate prep/cooldown subsequences
         * use one full expanded pass (including average expansion). */
        if (has_nd_prep || has_nd_cool) {
            rc = check_cross_pass_rf_consistency(desc, diag);
            if (PULSEG_FAILED(rc)) {
                if (diag) {
                    pulseg__diag_printf(diag,
                        "Consistency check failed: canonical RF "
                        "mismatch in subsequence %d\n",
                        subseq_idx);
                }
                return rc;
            }
        } else if (trd->num_trs > 1) {
            ref_tr      = 0;
            first_check = 1;
            last_check  = trd->num_trs - 1;

            rc = check_rf_amplitude_periodicity(desc,
                ref_tr, first_check, last_check, diag);
            if (PULSEG_FAILED(rc)) {
                if (diag) {
                    pulseg__diag_printf(diag,
                        "Consistency check failed: canonical RF amplitude "
                        "not periodic in subsequence %d\n",
                        subseq_idx);
                }
                return rc;
            }

            rc = check_rf_shim_periodicity(desc,
                ref_tr, first_check, last_check, diag);
            if (PULSEG_FAILED(rc)) {
                if (diag) {
                    pulseg__diag_printf(diag,
                        "Consistency check failed: canonical RF shim ID "
                        "not periodic in subsequence %d\n",
                        subseq_idx);
                }
                return rc;
            }
        }
    }

    return PULSEG_SUCCESS;
}

/* Public wrapper around check_consistency */
int pulseg_check_consistency(
    const pulseg_collection* coll,
    pulseg_diagnostic* diag)
{
    pulseg_diagnostic local_diag;
    if (!diag) {
        pulseg_diagnostic_init(&local_diag);
        diag = &local_diag;
    }
    return check_consistency(coll, diag);
}

/* ================================================================== */
/*  Error formatting convenience function                             */
/* ================================================================== */

int pulseg_format_error(
    char* buf, int buf_size,
    int code,
    const pulseg_diagnostic* diag)
{
    const char* msg;
    const char* hint;
    int written;

    if (!buf || buf_size <= 0) return 0;
    buf[0] = '\0';

    msg  = pulseg_get_error_message(code);
    hint = pulseg_get_error_hint(code);

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
/*  pulseg_convert_collection (public convert entry point, Stage 3)    */
/* ================================================================== */

int pulseg_convert_collection(
    pulseg_collection* coll,
    pulseg_diagnostic* diag,
    const pulseg_pulseq_file* files,
    int n,
    int parse_labels,
    int num_averages)
{
    int i, j, result, rc;
    int adc_off = 0, seg_off = 0, blk_off = 0;
    pulseg_diagnostic local_diag;

    if (!diag) { pulseg_diagnostic_init(&local_diag); diag = &local_diag; }

    if (!files || !coll) { diag->code = PULSEG_ERR_NULL_POINTER; return 0; }
    if (n == 0) { diag->code = PULSEG_ERR_COLLECTION_EMPTY; return 0; }

    coll->descriptors = (pulseg_sequence_descriptor*)PULSEG_ALLOC(
        n * sizeof(pulseg_sequence_descriptor));
    coll->subsequence_info = (pulseg_subsequence_info*)PULSEG_ALLOC(
        n * sizeof(pulseg_subsequence_info));
    if (!coll->descriptors || !coll->subsequence_info) {
        if (coll->descriptors)     PULSEG_FREE(coll->descriptors);
        if (coll->subsequence_info) PULSEG_FREE(coll->subsequence_info);
        coll->descriptors = NULL;
        coll->subsequence_info = NULL;
        diag->code = PULSEG_ERR_ALLOC_FAILED;
        return 0;
    }

    coll->num_subsequences    = n;
    coll->total_duration_us   = 0.0f;
    coll->total_unique_segments = 0;
    coll->total_unique_adcs   = 0;
    coll->total_blocks        = 0;

    for (i = 0; i < n; ++i) {
        pulseg_sequence_descriptor desc = PULSEG_SEQUENCE_DESCRIPTOR_INIT;

        coll->subsequence_info[i].sequence_index     = i;
        coll->subsequence_info[i].adc_id_offset      = adc_off;
        coll->subsequence_info[i].segment_id_offset  = seg_off;
        coll->subsequence_info[i].block_index_offset = blk_off;

        result = pulseg__get_unique_blocks(&desc, &files[i]);
        if (PULSEG_FAILED(result)) { diag->code = result; goto fail; }

        result = pulseg__get_tr_in_sequence(&desc, diag);
        if (PULSEG_FAILED(diag->code)) goto fail;

        result = pulseg__compute_variable_grad_flags(&desc);
        if (PULSEG_FAILED(result)) { diag->code = result; goto fail; }

        result = pulseg__build_scan_table(&desc, num_averages, diag);
        if (PULSEG_FAILED(diag->code)) goto fail;

        /* Non-degenerate pass TR duration:
         * For any sequence with non-degenerate prep or cooldown (once==1 / once==2
         * blocks that are structurally distinct from the imaging TR), the canonical
         * TR equals one full per-slice pass (prep + all averages of imaging +
         * cooldown), computed as total scan-table duration divided by num_passes.
         * This applies to both single-pass (e.g. bSSFP 1sl) and multi-pass
         * (e.g. bSSFP 3sl) sequences. */
           if ((!desc.tr_descriptor.degenerate_prep ||
               !desc.tr_descriptor.degenerate_cooldown) &&
              (desc.tr_descriptor.num_prep_blocks > 0 ||
               desc.tr_descriptor.num_cooldown_blocks > 0)) {
            float total_dur = 0.0f;
            int n;
            for (n = 0; n < desc.scan_table_len; ++n) {
                int bt_idx = desc.scan_table_block_idx[n];
                const pulseg_block_table_element* bte =
                    &desc.block_table[bt_idx];
                const pulseg_block_definition* bdef =
                    &desc.block_definitions[bte->id];
                total_dur += (bte->duration_us >= 0)
                    ? (float)bte->duration_us
                    : (float)bdef->duration_us;
            }
            desc.tr_descriptor.tr_duration_us = total_dur / (float)desc.num_passes;
        }

        /* Scan-table-only segmentation (prep / main / cooldown) */
        result = pulseg__get_scan_table_segments(&desc, diag, &files[i].opts);
        if (PULSEG_FAILED(diag->code)) goto fail;

        /* get_scan_table_segments may adjust TR topology (e.g. sparse
         * multipass patterns can update tr_descriptor.tr_size). Refresh
         * variable-gradient flags so ZERO_VAR indexing matches final TR size. */
        result = pulseg__compute_variable_grad_flags(&desc);
        if (PULSEG_FAILED(result)) { diag->code = result; goto fail; }

        result = pulseg__calc_segment_timing(&desc, diag);
        if (PULSEG_FAILED(result)) { diag->code = result; goto fail; }

        pulseg__compute_scan_table_tr_start(&desc);

        result = pulseg__build_freq_mod_flags(&desc);
        if (PULSEG_FAILED(result)) { diag->code = result; goto fail; }

        if (parse_labels) {
            result = pulseg__build_label_table(&desc, &files[i]);
            if (PULSEG_FAILED(result)) { diag->code = result; goto fail; }
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

        /* Accumulate actual scan-table duration (not the peek-style
         * tr_duration × num_trs approximation). */
        {
            float subseq_dur = 0.0f;
            int n;
            for (n = 0; n < desc.scan_table_len; ++n) {
                int bt_idx = desc.scan_table_block_idx[n];
                const pulseg_block_table_element* bte =
                    &desc.block_table[bt_idx];
                const pulseg_block_definition* bdef =
                    &desc.block_definitions[bte->id];
                subseq_dur += (bte->duration_us >= 0)
                    ? (float)bte->duration_us
                    : (float)bdef->duration_us;
            }
            coll->total_duration_us += subseq_dur;
        }

        coll->descriptors[i] = desc;
    }

    coll->total_unique_segments = seg_off;
    coll->total_unique_adcs     = adc_off;
    coll->total_blocks          = blk_off;

    /* Cross-subsequence consistency (folded in from the former separate
     * pulseg_read()/pulseg_read_from_buffers() call, Stage 3 Step 2). */
    rc = check_consistency(coll, diag);
    if (PULSEG_FAILED(rc)) { diag->code = rc; i = n; goto fail; }

    diag->code = PULSEG_SUCCESS;
    return n;

fail:
    for (j = 0; j < i; ++j)
        pulseg_sequence_descriptor_free(&coll->descriptors[j]);
    PULSEG_FREE(coll->descriptors);
    PULSEG_FREE(coll->subsequence_info);
    coll->descriptors      = NULL;
    coll->subsequence_info = NULL;
    coll->num_subsequences = 0;
    return 0;
}

/* ================================================================== */
/*  pulseg_read (public entry point)                               */
/* ================================================================== */

int pulseg_read(
    pulseg_collection** out_coll,
    pulseg_diagnostic* diag,
    const char* file_path,
    const pulseg_opts* opts,
    int cache_binary,
    int verify_signature,
    int parse_labels,
    int num_averages)
{
    pulseg_pulseq_file_set raw_coll;
    pulseg_collection* collection;
    int rc, i;

    raw_coll.num_sequences = 0;
    raw_coll.sequences     = NULL;
    raw_coll.base_path     = NULL;

    if (!file_path || !opts || !out_coll || !diag)
        return PULSEG_ERR_NULL_POINTER;

    *out_coll = NULL;
    pulseg_diagnostic_init(diag);

    /* Heap-allocate the opaque collection */
    collection = pulseg_collection_alloc();
    if (!collection) return PULSEG_ERR_ALLOC_FAILED;

    /* Try cache */
    if (cache_binary && pulseg__try_read_cache(collection, file_path, opts->cache_ext)) {
        /* Segment timing and TR-start flags are derived, not cached */
        for (i = 0; i < collection->num_subsequences; ++i) {
            pulseg__calc_segment_timing(&collection->descriptors[i], NULL);
            pulseg__compute_scan_table_tr_start(&collection->descriptors[i]);
        }
        *out_coll = collection;
        return PULSEG_SUCCESS;
    }

    /* Full parse */
    rc = pulseg_pulseq_file_set_read(&raw_coll, file_path, opts);
    if (PULSEG_FAILED(rc)) { diag->code = rc; goto fail; }

    /* Optional MD5 signature verification (all files in chain) */
    if (verify_signature) {
        for (i = 0; i < raw_coll.num_sequences; ++i) {
            const char* fpath = raw_coll.sequences[i].file_path;
            if (!fpath) continue;
            rc = pulseg_pulseq_verify_signature(fpath);
            if (PULSEG_FAILED(rc)) {
                diag->code = rc;
                pulseg__diag_printf(diag, " subsequence=%d", i);
                goto fail;
            }
        }
    }

    rc = pulseg_convert_collection(collection, diag, raw_coll.sequences,
                                    raw_coll.num_sequences, parse_labels, num_averages);
    if (PULSEG_FAILED(diag->code)) { rc = diag->code; goto fail; }

    pulseg_pulseq_file_set_free(&raw_coll);

    /* Write cache (best-effort) */
    if (cache_binary) pulseg__write_cache(collection, file_path, opts);

    *out_coll = collection;
    return PULSEG_SUCCESS;

fail:
    pulseg_pulseq_file_set_free(&raw_coll);
    PULSEG_FREE(collection);
    return rc;
}

/* ================================================================== */
/*  pulseg_read_from_buffers (public entry point)                  */
/* ================================================================== */

int pulseg_read_from_buffers(
    pulseg_collection** out_coll,
    pulseg_diagnostic* diag,
    const char* const* buffers,
    const int* buffer_sizes,
    int num_buffers,
    const pulseg_opts* opts,
    int parse_labels,
    int num_averages)
{
    pulseg_pulseq_file_set raw_coll;
    pulseg_collection* collection;
    int rc, i;

    raw_coll.num_sequences = 0;
    raw_coll.sequences     = NULL;
    raw_coll.base_path     = NULL;

    if (!out_coll || !diag || !buffers || !buffer_sizes || !opts)
        return PULSEG_ERR_NULL_POINTER;
    if (num_buffers < 1) return PULSEG_ERR_INVALID_ARGUMENT;

    *out_coll = NULL;
    pulseg_diagnostic_init(diag);

    /* Build raw collection from in-memory buffers */
    raw_coll.sequences = (pulseg_pulseq_file*)PULSEG_ALLOC(
        num_buffers * sizeof(pulseg_pulseq_file));
    if (!raw_coll.sequences) return PULSEG_ERR_ALLOC_FAILED;
    raw_coll.num_sequences = 0;
    raw_coll.base_path     = NULL;

    for (i = 0; i < num_buffers; ++i) {
        FILE* tmp;
        if (!buffers[i] || buffer_sizes[i] < 0) {
            rc = PULSEG_ERR_INVALID_ARGUMENT;
            diag->code = rc;
            goto fail_raw;
        }

        tmp = tmpfile();
        if (!tmp) { rc = PULSEG_ERR_FILE_NOT_FOUND; diag->code = rc; goto fail_raw; }

        if (buffer_sizes[i] > 0) {
            if ((int)fwrite(buffers[i], 1, (size_t)buffer_sizes[i], tmp)
                != buffer_sizes[i]) {
                fclose(tmp);
                rc = PULSEG_ERR_FILE_NOT_FOUND;
                diag->code = rc;
                goto fail_raw;
            }
        }
        rewind(tmp);

        pulseg_pulseq_file_init(&raw_coll.sequences[i], opts);
        rc = pulseg_pulseq_file_read_from_buffer(&raw_coll.sequences[i], tmp);
        fclose(tmp);
        if (PULSEG_FAILED(rc)) { diag->code = rc; goto fail_raw; }
        raw_coll.num_sequences = i + 1;
    }

    /* Heap-allocate the opaque collection */
    collection = pulseg_collection_alloc();
    if (!collection) { rc = PULSEG_ERR_ALLOC_FAILED; goto fail_raw; }

    rc = pulseg_convert_collection(collection, diag, raw_coll.sequences,
                                    raw_coll.num_sequences, parse_labels, num_averages);
    if (PULSEG_FAILED(diag->code)) { rc = diag->code; goto fail_coll; }

    pulseg_pulseq_file_set_free(&raw_coll);

    *out_coll = collection;
    return PULSEG_SUCCESS;

fail_coll:
    PULSEG_FREE(collection);
fail_raw:
    pulseg_pulseq_file_set_free(&raw_coll);
    return rc;
}