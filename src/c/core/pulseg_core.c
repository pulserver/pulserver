/**
 * @file pulseg_core.c
 * @brief Collection lifecycle, subsequence chaining, and the load entry point.
 *
 * pulseg_read() and pulseg_convert_collection() live here: they drive the
 * per-subsequence pipeline (dedup in pulseg_dedup.c, TR detection and
 * segmentation in pulseg_structure.c), chain the results into one collection,
 * and run the cross-subsequence consistency checks that must hold before any
 * consumer sees it.
 *
 * The matching free functions are here too -- every allocation made anywhere
 * in the pipeline is released by pulseg_sequence_descriptor_free() or
 * pulseg_collection_free().
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "pulseg_internal.h"
#include "pulseg.h"

static void segment_table_result_free(pulseg_segment_table_result *result);
static void free_segment_remap(pulseg_collection *coll);

/* ================================================================== */
/*  Descriptor free functions (public)                                */
/* ================================================================== */

void pulseg_sequence_descriptor_free(pulseg_sequence_descriptor *d)
{
    int i;
    if (!d)
        return;

    if (d->base_blocks)
    {
        PULSEG_FREE(d->base_blocks);
        d->base_blocks = NULL;
    }
    d->num_unique_blocks = 0;
    if (d->block_table)
    {
        PULSEG_FREE(d->block_table);
        d->block_table = NULL;
    }
    d->num_blocks = 0;

    if (d->rf_definitions)
    {
        PULSEG_FREE(d->rf_definitions);
        d->rf_definitions = NULL;
    }
    d->num_unique_rfs = 0;
    if (d->rf_table)
    {
        PULSEG_FREE(d->rf_table);
        d->rf_table = NULL;
    }
    d->rf_table_size = 0;

    if (d->grad_definitions)
    {
        PULSEG_FREE(d->grad_definitions);
        d->grad_definitions = NULL;
    }
    d->num_unique_grads = 0;
    if (d->grad_table)
    {
        PULSEG_FREE(d->grad_table);
        d->grad_table = NULL;
    }
    d->grad_table_size = 0;

    if (d->adc_definitions)
    {
        PULSEG_FREE(d->adc_definitions);
        d->adc_definitions = NULL;
    }
    d->num_unique_adcs = 0;
    if (d->adc_table)
    {
        PULSEG_FREE(d->adc_table);
        d->adc_table = NULL;
    }
    d->adc_table_size = 0;

    if (d->rf_shim_definitions)
    {
        PULSEG_FREE(d->rf_shim_definitions);
        d->rf_shim_definitions = NULL;
    }
    d->num_rf_shims = 0;

    if (d->rotation_matrices)
    {
        PULSEG_FREE(d->rotation_matrices);
        d->rotation_matrices = NULL;
    }
    d->num_rotations = 0;
    if (d->trigger_events)
    {
        PULSEG_FREE(d->trigger_events);
        d->trigger_events = NULL;
    }
    d->num_triggers = 0;

    if (d->grad_shape_first)
    {
        PULSEG_FREE(d->grad_shape_first);
        d->grad_shape_first = NULL;
    }
    if (d->grad_shape_last)
    {
        PULSEG_FREE(d->grad_shape_last);
        d->grad_shape_last = NULL;
    }
    d->num_grad_shape_stats = 0;

    if (d->shapes)
    {
        for (i = 0; i < d->num_shapes; ++i)
            if (d->shapes[i].samples)
                PULSEG_FREE(d->shapes[i].samples);
        PULSEG_FREE(d->shapes);
        d->shapes = NULL;
    }
    d->num_shapes = 0;

    if (d->segment_definitions)
    {
        for (i = 0; i < d->num_unique_segments; ++i)
        {
            if (d->segment_definitions[i].unique_block_indices)
                PULSEG_FREE(d->segment_definitions[i].unique_block_indices);
            if (d->segment_definitions[i].has_digitalout)
                PULSEG_FREE(d->segment_definitions[i].has_digitalout);
            if (d->segment_definitions[i].has_rotation)
                PULSEG_FREE(d->segment_definitions[i].has_rotation);
            if (d->segment_definitions[i].norot_flag)
                PULSEG_FREE(d->segment_definitions[i].norot_flag);
            if (d->segment_definitions[i].nopos_flag)
                PULSEG_FREE(d->segment_definitions[i].nopos_flag);
            if (d->segment_definitions[i].has_adc)
                PULSEG_FREE(d->segment_definitions[i].has_adc);
            if (d->segment_definitions[i].is_dynamic_delay)
                PULSEG_FREE(d->segment_definitions[i].is_dynamic_delay);
            if (d->segment_definitions[i].initial_states)
                PULSEG_FREE(d->segment_definitions[i].initial_states);
            if (d->segment_definitions[i].timing.rf_anchors)
                PULSEG_FREE(d->segment_definitions[i].timing.rf_anchors);
            if (d->segment_definitions[i].timing.adc_anchors)
                PULSEG_FREE(d->segment_definitions[i].timing.adc_anchors);
        }
        PULSEG_FREE(d->segment_definitions);
        d->segment_definitions = NULL;
    }
    d->num_unique_segments = 0;

    segment_table_result_free(&d->segment_table);

    /* Scan table arrays */
    if (d->exec_stream_block_idx)
    {
        PULSEG_FREE(d->exec_stream_block_idx);
        d->exec_stream_block_idx = NULL;
    }
    if (d->exec_stream_seg_id)
    {
        PULSEG_FREE(d->exec_stream_seg_id);
        d->exec_stream_seg_id = NULL;
    }
    if (d->exec_stream_avg_id)
    {
        PULSEG_FREE(d->exec_stream_avg_id);
        d->exec_stream_avg_id = NULL;
    }
    if (d->exec_stream_tr_start)
    {
        PULSEG_FREE(d->exec_stream_tr_start);
        d->exec_stream_tr_start = NULL;
    }
    if (d->exec_runs)
    {
        PULSEG_FREE(d->exec_runs);
        d->exec_runs = NULL;
    }
    d->num_exec_runs = 0;
    if (d->seg_run_start)
    {
        PULSEG_FREE(d->seg_run_start);
        d->seg_run_start = NULL;
    }
    if (d->seg_run_id)
    {
        PULSEG_FREE(d->seg_run_id);
        d->seg_run_id = NULL;
    }
    d->num_seg_runs = 0;
    d->tr_start_first = -1;
    d->exec_stream_len = 0;

    if (d->variable_grad_flags)
    {
        PULSEG_FREE(d->variable_grad_flags);
        d->variable_grad_flags = NULL;
    }

    if (d->label_table)
    {
        PULSEG_FREE(d->label_table);
        d->label_table = NULL;
    }
    if (d->off_table)
    {
        PULSEG_FREE(d->off_table);
        d->off_table = NULL;
    }
    d->label_num_columns = 0;
    d->label_num_entries = 0;
}

void pulseg_collection_free(pulseg_collection *c)
{
    int i;
    if (!c)
        return;
    if (c->descriptors)
    {
        for (i = 0; i < c->num_subsequences; ++i)
            pulseg_sequence_descriptor_free(&c->descriptors[i]);
        PULSEG_FREE(c->descriptors);
    }
    if (c->subsequence_info)
        PULSEG_FREE(c->subsequence_info);
    free_segment_remap(c);
    /* Free the struct itself (allocated by pulseg_read) */
    PULSEG_FREE(c);
}

pulseg_collection *pulseg_collection_alloc(void)
{
    pulseg_collection *c = (pulseg_collection *)PULSEG_ALLOC(sizeof(pulseg_collection));
    if (!c)
        return NULL;
    memset(c, 0, sizeof(*c));
    c->block_cursor.exec_stream_position = -1;
    c->num_repetitions = 1;
    return c;
}

static void segment_table_result_free(pulseg_segment_table_result *r)
{
    if (!r)
        return;
    if (r->main_segment_table)
        PULSEG_FREE(r->main_segment_table);
    r->main_segment_table = NULL;
    r->num_main_segments = 0;
    r->num_unique_segments = 0;
}

/* ================================================================== */
/*  Consistency check helpers                                         */
/* ================================================================== */

/*
 * get_block_rf_amplitude --
 *   Return the RF amplitude for block at absolute index 'block_idx',
 *   or 0 if the block has no RF.
 */
static float get_block_rf_amplitude(const pulseg_sequence_descriptor *desc, int block_idx)
{
    const pulseg_block_table_element *bte;

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
static int get_block_rf_shim_id(const pulseg_sequence_descriptor *desc, int block_idx)
{
    return desc->block_table[block_idx].rf_shim_id;
}

/*
 * check_rf_amplitude_periodicity --
 *   Verify that the RF amplitude pattern within a TR is identical
 *   across TR instances.
 *
 *   ref_tr:   index of the reference TR
 *   first_tr: first TR index to check (inclusive)
 *   last_tr:  last TR index to check (inclusive)
 *
 *   Compares each TR in [first_tr, last_tr] against ref_tr.
 */
static int check_rf_amplitude_periodicity(
    pulseg_sequence_descriptor *desc,
    int allow_variable,
    int ref_tr,
    int first_tr,
    int last_tr,
    pulseg_diagnostic *diag)
{
    const pulseg_tr_descriptor *trd;
    int tr_size;
    int ref_start, tr_idx, chk_start;
    int j;
    float ref_amp, chk_amp;

    trd = &desc->tr_descriptor;
    tr_size = trd->tr_size;
    ref_start = ref_tr * tr_size;

    for (tr_idx = first_tr; tr_idx <= last_tr; ++tr_idx)
    {
        if (tr_idx == ref_tr)
            continue;
        chk_start = tr_idx * tr_size;
        for (j = 0; j < tr_size; ++j)
        {
            ref_amp = get_block_rf_amplitude(desc, ref_start + j);
            chk_amp = get_block_rf_amplitude(desc, chk_start + j);
            if (ref_amp != chk_amp)
            {
                if (!allow_variable)
                {
                    if (diag)
                    {
                        pulseg__diag_printf(
                            diag,
                            "RF periodicity: TR %d block %d has amplitude %.6g, "
                            "expected %.6g (from reference TR %d)\n",
                            tr_idx,
                            j,
                            (double)chk_amp,
                            (double)ref_amp,
                            ref_tr);
                    }
                    return PULSEG_ERR_CONSISTENCY_RF_PERIODIC;
                }
                /* Accepted: the RF safety model switches to the worst-B1rms
                 * real instance for time-averaged limits, with a
                 * positional-max envelope kept alongside for peak-only
                 * limits (pulseg_get_rf_array). Record the first mismatch
                 * for debuggability, then keep scanning -- shim periodicity
                 * below still applies. */
                if (!desc->rf_amplitude_variable && diag)
                {
                    pulseg__diag_printf(
                        diag,
                        "RF amplitude varies (TR %d block %d: %.6g vs %.6g at "
                        "TR %d); using positional-max safety envelope\n",
                        tr_idx,
                        j,
                        (double)chk_amp,
                        (double)ref_amp,
                        ref_tr);
                }
                desc->rf_amplitude_variable = 1;
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
    const pulseg_sequence_descriptor *desc,
    int ref_tr,
    int first_tr,
    int last_tr,
    pulseg_diagnostic *diag)
{
    const pulseg_tr_descriptor *trd;
    int tr_size;
    int ref_start, tr_idx, chk_start;
    int j;
    int ref_shim, chk_shim;

    trd = &desc->tr_descriptor;
    tr_size = trd->tr_size;
    ref_start = ref_tr * tr_size;

    for (tr_idx = first_tr; tr_idx <= last_tr; ++tr_idx)
    {
        if (tr_idx == ref_tr)
            continue;
        chk_start = tr_idx * tr_size;
        for (j = 0; j < tr_size; ++j)
        {
            ref_shim = get_block_rf_shim_id(desc, ref_start + j);
            chk_shim = get_block_rf_shim_id(desc, chk_start + j);
            if (ref_shim != chk_shim)
            {
                if (diag)
                {
                    pulseg__diag_printf(
                        diag,
                        "RF shim periodicity: TR %d block %d has shim_id %d, "
                        "expected %d (from reference TR %d)\n",
                        tr_idx,
                        j,
                        chk_shim,
                        ref_shim,
                        ref_tr);
                }
                return PULSEG_ERR_CONSISTENCY_RF_SHIM_PERIODIC;
            }
        }
    }

    return PULSEG_SUCCESS;
}

/*
 * check_exec_stream_segments --
 *   Walk the scan table and verify that each entry's block definition ID
 *   matches the segment definition indicated by exec_stream_seg_id.
 *
 *   For each contiguous group of entries sharing the same seg_id,
 *   position within the group gives the position within the segment.
 */
static int check_exec_stream_segments(
    const pulseg_sequence_descriptor *desc,
    pulseg_diagnostic *diag)
{
    int n, seg_id, prev_seg_id, pos_in_seg;
    int bt_idx, bdef_id, expected_id;
    const pulseg_virtual_segment *seg;
    const pulseg_base_block *bdef_actual;
    const pulseg_base_block *bdef_expected;
    int both_pure_delay;
    int structural_match;

    prev_seg_id = -2; /* impossible value to force reset */
    pos_in_seg = 0;

    for (n = 0; n < desc->exec_stream_len; ++n)
    {
        seg_id = pulseg__exec_seg_id(desc, n);
        if (seg_id < 0)
        {
            prev_seg_id = seg_id;
            pos_in_seg = 0;
            continue;
        }
        /* Reset position when the segment type changes. */
        if (seg_id != prev_seg_id)
        {
            pos_in_seg = 0;
            prev_seg_id = seg_id;
        }

        if (seg_id >= desc->segment_table.num_unique_segments)
        {
            if (diag)
            {
                pulseg__diag_printf(
                    diag,
                    "Consistency: exec_stream_seg_id[%d] = %d out of range "
                    "(num_unique = %d)\n",
                    n,
                    seg_id,
                    desc->segment_table.num_unique_segments);
            }
            return PULSEG_ERR_CONSISTENCY_SEG_MISMATCH;
        }

        seg = &desc->segment_definitions[seg_id];

        /* When the same segment repeats across consecutive TRs (same
         * seg_id throughout), pos_in_seg naturally reaches num_blocks.
         * Wrap it so the next repetition is verified from UBI[0]. */
        if (pos_in_seg >= seg->num_blocks)
        {
            pos_in_seg = 0;
        }

        bt_idx = pulseg__exec_block_idx(desc, n);
        bdef_id = desc->block_table[bt_idx].id;
        expected_id = seg->unique_block_indices[pos_in_seg];

        both_pure_delay = 0;
        structural_match = 0;
        if (bdef_id >= 0 && bdef_id < desc->num_unique_blocks && expected_id >= 0 &&
            expected_id < desc->num_unique_blocks)
        {
            bdef_actual = &desc->base_blocks[bdef_id];
            bdef_expected = &desc->base_blocks[expected_id];
            both_pure_delay = (bdef_actual->rf_id == -1 && bdef_actual->gx_id == -1 &&
                               bdef_actual->gy_id == -1 && bdef_actual->gz_id == -1 &&
                               bdef_actual->adc_id == -1 && bdef_expected->rf_id == -1 &&
                               bdef_expected->gx_id == -1 && bdef_expected->gy_id == -1 &&
                               bdef_expected->gz_id == -1 && bdef_expected->adc_id == -1)
                ? 1
                : 0;
            if (!both_pure_delay)
            {
                structural_match =
                    pulseg__block_defs_structurally_equal(desc, bdef_id, expected_id);
            }
        }

        if (bdef_id != expected_id && !both_pure_delay && !structural_match)
        {
            if (diag)
            {
                pulseg__diag_printf(
                    diag,
                    "Consistency: scan pos %d (block_table[%d]) has def ID %d, "
                    "expected %d (segment %d, position %d)\n",
                    n,
                    bt_idx,
                    bdef_id,
                    expected_id,
                    seg_id,
                    pos_in_seg);
            }
            return PULSEG_ERR_CONSISTENCY_SEG_MISMATCH;
        }

        ++pos_in_seg;
    }
    return PULSEG_SUCCESS;
}

static int check_consistency(
    pulseg_collection *coll,
    int allow_variable_rf,
    pulseg_diagnostic *diag)
{
    int subseq_idx, rc;
    pulseg_sequence_descriptor *desc;
    const pulseg_tr_descriptor *trd;
    int ref_tr, first_check, last_check;

    if (!coll)
        return PULSEG_ERR_NULL_POINTER;

    for (subseq_idx = 0; subseq_idx < coll->num_subsequences; ++subseq_idx)
    {
        desc = &coll->descriptors[subseq_idx];
        trd = &desc->tr_descriptor;

        /* (a) Scan-table segment consistency: walk the scan table and
         *     verify that each entry's block definition ID matches what
         *     its seg_id expects. */
        if (desc->exec_stream_len > 0 && desc->seg_run_id)
        {
            rc = check_exec_stream_segments(desc, diag);
            if (PULSEG_FAILED(rc))
            {
                if (diag)
                {
                    pulseg__diag_printf(
                        diag,
                        "Segment consistency check failed "
                        "in subsequence %d\n",
                        subseq_idx);
                }
                return rc;
            }
        }

        /* (b) RF periodicity across TR instances. */
        if (trd->num_trs > 1)
        {
            ref_tr = 0;
            first_check = 1;
            last_check = trd->num_trs - 1;

            rc = check_rf_amplitude_periodicity(
                desc,
                allow_variable_rf,
                ref_tr,
                first_check,
                last_check,
                diag);
            if (PULSEG_FAILED(rc))
            {
                if (diag)
                {
                    pulseg__diag_printf(
                        diag,
                        "Consistency check failed: canonical RF amplitude "
                        "not periodic in subsequence %d\n",
                        subseq_idx);
                }
                return rc;
            }

            rc = check_rf_shim_periodicity(desc, ref_tr, first_check, last_check, diag);
            if (PULSEG_FAILED(rc))
            {
                if (diag)
                {
                    pulseg__diag_printf(
                        diag,
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

/* Public wrapper around check_consistency.
 *
 * Uses the permissive gate (variable RF amplitude accepted), matching the
 * PULSEG_OPTS_INIT default: a caller re-running the check on an
 * already-converted collection should not reject what conversion accepted. To
 * assert strict periodicity, run conversion with
 * pulseg_opts.allow_variable_rf_amplitude = 0, which fails before this point.
 * The cast drops const: on a genuinely variable sequence this raises the
 * descriptor's rf_amplitude_variable flag (idempotent -- it is already set
 * from conversion). */
int pulseg_check_consistency(const pulseg_collection *coll, pulseg_diagnostic *diag)
{
    pulseg_diagnostic local_diag;
    if (!diag)
    {
        pulseg_diagnostic_init(&local_diag);
        diag = &local_diag;
    }
    return check_consistency((pulseg_collection *)coll, /*allow_variable_rf=*/1, diag);
}

/* ================================================================== */
/*  Error formatting convenience function                             */
/* ================================================================== */

int pulseg_format_error(char *buf, int buf_size, int code, const pulseg_diagnostic *diag)
{
    const char *msg;
    const char *hint;
    int written;

    if (!buf || buf_size <= 0)
        return 0;
    buf[0] = '\0';

    msg = pulseg_get_error_message(code);
    hint = pulseg_get_error_hint(code);

    /* Build the string with sprintf; caller must provide >= 512 bytes.
     * We guard against overrun by checking buf_size, but the assembled
     * string is never longer than ~380 chars (msg + hint + diag). */
    if (diag && diag->message[0] != '\0')
    {
        if (buf_size < 512)
        {
            buf[0] = '\0';
            return 0;
        }
        written = sprintf(buf, "%s (%s)", msg, diag->message);
    }
    else if (hint && hint[0] != '\0')
    {
        if (buf_size < 256)
        {
            buf[0] = '\0';
            return 0;
        }
        written = sprintf(buf, "%s (%s)", msg, hint);
    }
    else
    {
        if (buf_size < 128)
        {
            buf[0] = '\0';
            return 0;
        }
        written = sprintf(buf, "%s", msg);
    }
    if (written < 0)
        written = 0;
    return written;
}

/* ================================================================== */
/*  Cross-subsequence segment deduplication (footprint minimisation)   */
/*                                                                     */
/*  Two segments in DIFFERENT subsequences materialise byte-identical  */
/*  EPIC instruction memory whenever their normalised waveform shapes, */
/*  timing and event topology match — per-instance amplitude / phase / */
/*  rotation are applied at scan time from the REAL subsequence's block */
/*  instance (pulseg_get_block_instance), never baked into the shared   */
/*  segment buffer.  So it is safe to give such segments one global id  */
/*  and build the instruction memory once.  The per-subsequence         */
/*  descriptors are left untouched, so all safety / trajectory analysis */
/*  (which runs on descriptors, never on the global segment space) is   */
/*  unaffected.                                                          */
/* ================================================================== */

/* Compare two decompressed shapes referenced by 1-based local shape ids
 * (<= 0 means "no shape").  Returns 1 if both absent or both decompress to
 * equal sample vectors (within a small tolerance), 0 otherwise. */
static int seg_shape_equal(
    const pulseg_sequence_descriptor *da,
    int sid_a,
    const pulseg_sequence_descriptor *db,
    int sid_b)
{
    pulseq_shape xa, xb;
    int i, ok, ha, hb;

    ha = (sid_a > 0 && sid_a <= da->num_shapes);
    hb = (sid_b > 0 && sid_b <= db->num_shapes);
    if (ha != hb)
        return 0;
    if (!ha)
        return 1; /* both absent */

    xa.samples = NULL;
    xa.num_samples = 0;
    xa.num_uncompressed_samples = 0;
    xb.samples = NULL;
    xb.num_samples = 0;
    xb.num_uncompressed_samples = 0;

    if (!pulseq_decompress_shape(&xa, &da->shapes[sid_a - 1], 1.0f))
    {
        if (xa.samples)
            PULSEG_FREE(xa.samples);
        return 0;
    }
    if (!pulseq_decompress_shape(&xb, &db->shapes[sid_b - 1], 1.0f))
    {
        PULSEG_FREE(xa.samples);
        if (xb.samples)
            PULSEG_FREE(xb.samples);
        return 0;
    }

    ok = (xa.num_uncompressed_samples == xb.num_uncompressed_samples);
    for (i = 0; ok && i < xa.num_uncompressed_samples; ++i)
    {
        float d = xa.samples[i] - xb.samples[i];
        if (d < 0.0f)
            d = -d;
        if (d > 1e-6f)
            ok = 0;
    }
    PULSEG_FREE(xa.samples);
    PULSEG_FREE(xb.samples);
    return ok;
}

/* Compare two gradient definitions (by label) for buffer-equal content. */
static int seg_grad_def_equal(
    const pulseg_sequence_descriptor *da,
    int gid_a,
    const pulseg_sequence_descriptor *db,
    int gid_b)
{
    const pulseg_grad_definition *ga;
    const pulseg_grad_definition *gb;

    if (gid_a < 0 || gid_a >= da->num_unique_grads)
        return 0;
    if (gid_b < 0 || gid_b >= db->num_unique_grads)
        return 0;
    ga = &da->grad_definitions[gid_a];
    gb = &db->grad_definitions[gid_b];

    if (ga->type != gb->type)
        return 0;
    if (ga->delay != gb->delay)
        return 0;

    if (ga->type == 0)
    {
        /* trapezoid: corner-point timing only */
        return (
            ga->rise_time_or_unused == gb->rise_time_or_unused &&
            ga->flat_time_or_unused == gb->flat_time_or_unused &&
            ga->fall_time_or_num_uncompressed_samples == gb->fall_time_or_num_uncompressed_samples);
    }

    /* Arbitrary: sample count and time shape.
     *
     * Deliberately NOT the set of amplitude shapes.  What makes two segments
     * the same segment is their structure -- the timings: delays, durations,
     * sample counts, the time shape.  Which waveform an instance plays at a
     * position is per instance, carried by the exec stream and swapped in at
     * playout, so two definitions that differ only in that are the same
     * definition as far as the materialised segment is concerned. */
    if (ga->fall_time_or_num_uncompressed_samples != gb->fall_time_or_num_uncompressed_samples)
        return 0;
    if (!seg_shape_equal(da, ga->unused_or_time_shape_id, db, gb->unused_or_time_shape_id))
        return 0;
    return 1;
}

/* Compare two RF definitions (by label) for buffer-equal content. */
static int seg_rf_def_equal(
    const pulseg_sequence_descriptor *da,
    int rid_a,
    const pulseg_sequence_descriptor *db,
    int rid_b)
{
    const pulseg_rf_definition *ra;
    const pulseg_rf_definition *rb;

    if (rid_a < 0 || rid_a >= da->num_unique_rfs)
        return 0;
    if (rid_b < 0 || rid_b >= db->num_unique_rfs)
        return 0;
    ra = &da->rf_definitions[rid_a];
    rb = &db->rf_definitions[rid_b];

    if (ra->num_channels != rb->num_channels)
        return 0;
    if (ra->delay != rb->delay)
        return 0;
    return (
        seg_shape_equal(da, ra->mag_shape_id, db, rb->mag_shape_id) &&
        seg_shape_equal(da, ra->phase_shape_id, db, rb->phase_shape_id) &&
        seg_shape_equal(da, ra->time_shape_id, db, rb->time_shape_id));
}

/* Compare two ADC definitions (by label). */
static int seg_adc_def_equal(
    const pulseg_sequence_descriptor *da,
    int aid_a,
    const pulseg_sequence_descriptor *db,
    int aid_b)
{
    const pulseg_adc_definition *aa;
    const pulseg_adc_definition *ab;

    if (aid_a < 0 || aid_a >= da->num_unique_adcs)
        return 0;
    if (aid_b < 0 || aid_b >= db->num_unique_adcs)
        return 0;
    aa = &da->adc_definitions[aid_a];
    ab = &db->adc_definitions[aid_b];
    return (
        aa->num_samples == ab->num_samples && aa->dwell_time == ab->dwell_time &&
        aa->delay == ab->delay);
}

/* Resolve the block DEFINITION that materialises block b of a segment.
 * Reads the frozen max-energy-representative record (pulseg_structure.c step
 * 11e) so the compared content matches the actually-built buffer without the
 * per-instance tables; falls back to the segment's first-instance
 * unique_block_indices when no representative was recorded. */
static const pulseg_base_block *seg_block_def(
    const pulseg_sequence_descriptor *desc,
    const pulseg_virtual_segment *seg,
    int b)
{
    if (seg->initial_states)
    {
        int id = seg->initial_states[b].base_block_id;
        if (id >= 0 && id < desc->num_unique_blocks)
            return &desc->base_blocks[id];
    }
    if (seg->unique_block_indices)
    {
        int id = seg->unique_block_indices[b];
        if (id >= 0 && id < desc->num_unique_blocks)
            return &desc->base_blocks[id];
    }
    return NULL;
}

/* Segment-level physio-trigger TYPE (or 0 if none), for topology compare. */
static int seg_trigger_type(
    const pulseg_sequence_descriptor *desc,
    const pulseg_virtual_segment *seg)
{
    if (seg->trigger_id >= 0 && seg->trigger_id < desc->num_triggers)
        return desc->trigger_events[seg->trigger_id].trigger_type;
    return 0;
}

/* Deep content equality of two segments (possibly in different subsequences).
 * Returns 1 iff their EPIC-materialised instruction memory is identical. */
static int segments_content_equal(
    const pulseg_collection *coll,
    int subseq_a,
    int local_a,
    int subseq_b,
    int local_b)
{
    const pulseg_sequence_descriptor *da = &coll->descriptors[subseq_a];
    const pulseg_sequence_descriptor *db = &coll->descriptors[subseq_b];
    const pulseg_virtual_segment *sa;
    const pulseg_virtual_segment *sb;
    int b;

    if (local_a < 0 || local_a >= da->num_unique_segments)
        return 0;
    if (local_b < 0 || local_b >= db->num_unique_segments)
        return 0;
    sa = &da->segment_definitions[local_a];
    sb = &db->segment_definitions[local_b];

    if (sa->num_blocks != sb->num_blocks)
        return 0;
    if (sa->is_nav != sb->is_nav)
        return 0;
    if (seg_trigger_type(da, sa) != seg_trigger_type(db, sb))
        return 0;

    for (b = 0; b < sa->num_blocks; ++b)
    {
        const pulseg_base_block *ba = seg_block_def(da, sa, b);
        const pulseg_base_block *bb = seg_block_def(db, sb, b);
        if (!ba || !bb)
            return 0;

        /* Block duration must match, EXCEPT a DYNAMIC adjustable pure-delay
         * block: no RF/grad/ADC AND no digital-output trigger or rotation,
         * AND its duration actually varies across its own subsequence's
         * instances, so it is applied per-instance at scan time (setperiod)
         * and two segments differing only there can share one definition.
         * This must match pulseg_get_block_info()'s is_variable_delay
         * exactly -- otherwise a merged block that EPIC does NOT setperiod
         * would play a fixed (wrong) duration.  A STATIC adjustable delay
         * (is_dynamic_delay == 0) is never setperiod'd by EPIC, so its baked
         * duration must still match exactly, same as any other fixed block --
         * skipping the check for it here would merge two definitions that
         * silently disagree on duration.  The digitalout/rotation presence
         * checks below are still enforced (they must be equal), so a
         * trigger/rotation delay block is never collapsed onto a fixed
         * wait. */
        {
            int a_adj = pulseg__block_def_is_pure_delay(ba) && !sa->has_digitalout[b] &&
                !sa->has_rotation[b] && (sa->is_dynamic_delay ? sa->is_dynamic_delay[b] : 1);
            int b_adj = pulseg__block_def_is_pure_delay(bb) && !sb->has_digitalout[b] &&
                !sb->has_rotation[b] && (sb->is_dynamic_delay ? sb->is_dynamic_delay[b] : 1);
            if (!(a_adj && b_adj))
            {
                if (ba->duration_us != bb->duration_us)
                    return 0;
            }
        }

        /* RF */
        if ((ba->rf_id >= 0) != (bb->rf_id >= 0))
            return 0;
        if (ba->rf_id >= 0 && !seg_rf_def_equal(da, ba->rf_id, db, bb->rf_id))
            return 0;

        /* Gradients per axis */
        if ((ba->gx_id >= 0) != (bb->gx_id >= 0))
            return 0;
        if (ba->gx_id >= 0 && !seg_grad_def_equal(da, ba->gx_id, db, bb->gx_id))
            return 0;
        if ((ba->gy_id >= 0) != (bb->gy_id >= 0))
            return 0;
        if (ba->gy_id >= 0 && !seg_grad_def_equal(da, ba->gy_id, db, bb->gy_id))
            return 0;
        if ((ba->gz_id >= 0) != (bb->gz_id >= 0))
            return 0;
        if (ba->gz_id >= 0 && !seg_grad_def_equal(da, ba->gz_id, db, bb->gz_id))
            return 0;

        /* ADC */
        if ((ba->adc_id >= 0) != (bb->adc_id >= 0))
            return 0;
        if (ba->adc_id >= 0 && !seg_adc_def_equal(da, ba->adc_id, db, bb->adc_id))
            return 0;

        /* Per-block topology flags (OR-reduced across instances). These drive
         * OMEGA / ISI / TTL / rotation instruction allocation, so they must
         * match for the shared buffer to be valid.                          */
        if (sa->has_rotation[b] != sb->has_rotation[b])
            return 0;
        if (sa->has_digitalout[b] != sb->has_digitalout[b])
            return 0;
        if (sa->has_adc[b] != sb->has_adc[b])
            return 0;
        if (sa->norot_flag[b] != sb->norot_flag[b])
            return 0;
        if (sa->nopos_flag[b] != sb->nopos_flag[b])
            return 0;
    }
    return 1;
}

static void free_segment_remap(pulseg_collection *coll)
{
    if (!coll)
        return;
    if (coll->seg_local_to_global)
    {
        PULSEG_FREE(coll->seg_local_to_global);
        coll->seg_local_to_global = NULL;
    }
    if (coll->seg_repr_subseq)
    {
        PULSEG_FREE(coll->seg_repr_subseq);
        coll->seg_repr_subseq = NULL;
    }
    if (coll->seg_repr_local)
    {
        PULSEG_FREE(coll->seg_repr_local);
        coll->seg_repr_local = NULL;
    }
    coll->seg_l2g_len = 0;
}

int pulseg__build_segment_remap(pulseg_collection *coll)
{
    int i, local, g, old_total, new_total, off;
    int *l2g = NULL;
    int *repr_s = NULL;
    int *repr_l = NULL;

    if (!coll)
        return PULSEG_ERR_NULL_POINTER;

    free_segment_remap(coll);

    /* Pre-dedup running-sum offsets (also the flat index base for l2g). */
    old_total = 0;
    for (i = 0; i < coll->num_subsequences; ++i)
    {
        coll->subsequence_info[i].segment_id_offset = old_total;
        old_total += coll->descriptors[i].num_unique_segments;
    }
    if (old_total <= 0)
    {
        coll->total_unique_segments = 0;
        return PULSEG_SUCCESS;
    }

    l2g = (int *)PULSEG_ALLOC((size_t)old_total * sizeof(int));
    repr_s = (int *)PULSEG_ALLOC((size_t)old_total * sizeof(int));
    repr_l = (int *)PULSEG_ALLOC((size_t)old_total * sizeof(int));
    if (!l2g || !repr_s || !repr_l)
    {
        if (l2g)
            PULSEG_FREE(l2g);
        if (repr_s)
            PULSEG_FREE(repr_s);
        if (repr_l)
            PULSEG_FREE(repr_l);
        /* Degrade to identity: resolve_segment / cursor fall back on the
         * offset walk when the remap arrays are NULL.                       */
        coll->total_unique_segments = old_total;
        return PULSEG_ERR_ALLOC_FAILED;
    }

    new_total = 0;
    off = 0;
    for (i = 0; i < coll->num_subsequences; ++i)
    {
        int nseg = coll->descriptors[i].num_unique_segments;
        for (local = 0; local < nseg; ++local)
        {
            int found = -1;
            for (g = 0; g < new_total; ++g)
            {
                if (segments_content_equal(coll, i, local, repr_s[g], repr_l[g]))
                {
                    found = g;
                    break;
                }
            }
            if (found < 0)
            {
                repr_s[new_total] = i;
                repr_l[new_total] = local;
                found = new_total;
                new_total++;
            }
            l2g[off + local] = found;
        }
        off += nseg;
    }

    coll->seg_local_to_global = l2g;
    coll->seg_l2g_len = old_total;
    coll->seg_repr_subseq = repr_s;
    coll->seg_repr_local = repr_l;
    coll->total_unique_segments = new_total;
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  pulseg_convert_collection (public convert entry point)    */
/* ================================================================== */

int pulseg_convert_collection(
    pulseg_collection *coll,
    pulseg_diagnostic *diag,
    const pulseq_file *files,
    int n,
    const pulseg_opts *opts,
    int parse_labels,
    int num_averages)
{
    int i, j, result, rc;
    int adc_off = 0, seg_off = 0, blk_off = 0;
    pulseg_diagnostic local_diag;

    if (!diag)
    {
        pulseg_diagnostic_init(&local_diag);
        diag = &local_diag;
    }

    if (!files || !coll || !opts)
    {
        diag->code = PULSEG_ERR_NULL_POINTER;
        return 0;
    }
    if (n == 0)
    {
        diag->code = PULSEG_ERR_COLLECTION_EMPTY;
        return 0;
    }

    coll->descriptors =
        (pulseg_sequence_descriptor *)PULSEG_ALLOC(n * sizeof(pulseg_sequence_descriptor));
    coll->subsequence_info =
        (pulseg_subsequence_info *)PULSEG_ALLOC(n * sizeof(pulseg_subsequence_info));
    if (!coll->descriptors || !coll->subsequence_info)
    {
        if (coll->descriptors)
            PULSEG_FREE(coll->descriptors);
        if (coll->subsequence_info)
            PULSEG_FREE(coll->subsequence_info);
        coll->descriptors = NULL;
        coll->subsequence_info = NULL;
        diag->code = PULSEG_ERR_ALLOC_FAILED;
        return 0;
    }

    coll->num_subsequences = n;
    coll->total_duration_us = 0.0f;
    coll->total_unique_segments = 0;
    coll->total_unique_adcs = 0;
    coll->total_blocks = 0;
    coll->total_readouts = 0;

    for (i = 0; i < n; ++i)
    {
        pulseg_sequence_descriptor desc = PULSEG_SEQUENCE_DESCRIPTOR_INIT;

        coll->subsequence_info[i].sequence_index = i;
        coll->subsequence_info[i].adc_id_offset = adc_off;
        coll->subsequence_info[i].segment_id_offset = seg_off;
        coll->subsequence_info[i].block_index_offset = blk_off;

        result = pulseg__get_unique_blocks(&desc, &files[i], opts);
        if (PULSEG_FAILED(result))
        {
            diag->code = result;
            goto fail;
        }

        result = pulseg__get_tr_in_sequence(&desc, diag);
        if (PULSEG_FAILED(diag->code))
            goto fail;

        result = pulseg__compute_variable_grad_flags(&desc);
        if (PULSEG_FAILED(result))
        {
            diag->code = result;
            goto fail;
        }

        result = pulseg__build_exec_stream(&desc, diag, num_averages);
        if (PULSEG_FAILED(diag->code))
            goto fail;

        /* Scan-table-only segmentation */
        result = pulseg__get_exec_stream_segments(&desc, diag, opts);
        if (PULSEG_FAILED(diag->code))
            goto fail;

        /* get_exec_stream_segments may adjust TR topology (e.g. sparse
         * multipass patterns can update tr_descriptor.tr_size). Refresh
         * variable-gradient flags so ZERO_VAR indexing matches final TR size. */
        result = pulseg__compute_variable_grad_flags(&desc);
        if (PULSEG_FAILED(result))
        {
            diag->code = result;
            goto fail;
        }

        result = pulseg__calc_segment_timing(&desc, diag);
        if (PULSEG_FAILED(result))
        {
            diag->code = result;
            goto fail;
        }

        pulseg__compute_exec_stream_tr_start(&desc);

        if (parse_labels)
        {
            result = pulseg__build_label_table(&desc, &files[i]);
            if (PULSEG_FAILED(result))
            {
                diag->code = result;
                goto fail;
            }
        }

        /* Everything that needed the position-indexed scratch arrays has
         * run; from here on the compact execution stream is the only
         * representation, exactly as it is after a cache load. */
        pulseg__free_exec_stream_scratch(&desc);

        /* apply offsets */
        if (seg_off > 0)
        {
            for (j = 0; j < desc.segment_table.num_main_segments; ++j)
                desc.segment_table.main_segment_table[j] += seg_off;
        }
        if (adc_off > 0)
        {
            for (j = 0; j < desc.adc_table_size; ++j)
                desc.adc_table[j].id += adc_off;
            for (j = 0; j < desc.num_unique_adcs; ++j)
                desc.adc_definitions[j].id += adc_off;
        }

        adc_off += desc.num_unique_adcs;
        seg_off += desc.num_unique_segments;
        blk_off += desc.num_blocks;

        /* Accumulate actual scan-table duration (not the peek-style
         * tr_duration × num_trs approximation) and the readout count.  Both
         * are frozen here because the per-instance tables they are derived
         * from are not loaded on every cache path (pulsegen loads neither). */
        {
            float subseq_dur = 0.0f;
            int n;
            for (n = 0; n < desc.exec_stream_len; ++n)
            {
                int bt_idx = pulseg__exec_block_idx(&desc, n);
                const pulseg_block_table_element *bte = &desc.block_table[bt_idx];
                const pulseg_base_block *bdef = &desc.base_blocks[bte->id];
                subseq_dur +=
                    (bte->duration_us >= 0) ? (float)bte->duration_us : (float)bdef->duration_us;
                if (bte->adc_id >= 0)
                    coll->total_readouts++;
            }
            coll->total_duration_us += subseq_dur;
        }

        coll->descriptors[i] = desc;
    }

    coll->total_unique_segments = seg_off;
    coll->total_unique_adcs = adc_off;
    coll->total_blocks = blk_off;

    /* Cross-subsequence consistency (folded in from the former separate
     * pulseg_read()/pulseg_read_from_buffers() call). */
    rc = check_consistency(coll, opts ? opts->allow_variable_rf_amplitude : 1, diag);
    if (PULSEG_FAILED(rc))
    {
        diag->code = rc;
        i = n;
        goto fail;
    }

    /* Collapse duplicate segments across subsequences into one global
     * instruction-memory entry (safety-preserving; see notes above). A
     * failure here degrades gracefully to the identity map. */
    pulseg__build_segment_remap(coll);

    diag->code = PULSEG_SUCCESS;
    return n;

fail:
    for (j = 0; j < i; ++j)
        pulseg_sequence_descriptor_free(&coll->descriptors[j]);
    PULSEG_FREE(coll->descriptors);
    PULSEG_FREE(coll->subsequence_info);
    coll->descriptors = NULL;
    coll->subsequence_info = NULL;
    coll->num_subsequences = 0;
    return 0;
}

/* ================================================================== */
/*  pulseg_read (public entry point)                               */
/* ================================================================== */

int pulseg_read(
    pulseg_collection **out_coll,
    pulseg_diagnostic *diag,
    const char *file_path,
    const pulseg_opts *opts,
    int cache_binary,
    int verify_signature,
    int parse_labels,
    int num_averages)
{
    pulseq_file_set raw_coll;
    pulseq_raster raster;
    pulseg_collection *collection;
    int rc, i;

    raw_coll.num_sequences = 0;
    raw_coll.sequences = NULL;
    raw_coll.base_path = NULL;

    if (!file_path || !opts || !out_coll || !diag)
        return PULSEG_ERR_NULL_POINTER;

    *out_coll = NULL;
    pulseg_diagnostic_init(diag);

    /* Heap-allocate the opaque collection */
    collection = pulseg_collection_alloc();
    if (!collection)
        return PULSEG_ERR_ALLOC_FAILED;

    /* Try cache */
    if (cache_binary && pulseg__try_read_cache(collection, file_path, opts->cache_ext))
    {
        /* Segment timing is derived, not cached. TR-start flags no longer
         * need recomputing here: pulseg__exec_tr_start derives them from
         * the (tr_start_main_id, tr_start_first) anchor, which the SCANLOOP
         * section carries -- and the per-position arrays it used to walk do
         * not exist after a cache load. */
        for (i = 0; i < collection->num_subsequences; ++i)
            pulseg__calc_segment_timing(&collection->descriptors[i], NULL);
        *out_coll = collection;
        return PULSEG_SUCCESS;
    }

    /* Full parse */
    pulseg_opts_get_design_raster(&raster, opts);
    rc = pulseq_file_set_read(&raw_coll, file_path, &raster);
    if (PULSEG_FAILED(rc))
    {
        diag->code = rc;
        goto fail;
    }

    /* Optional MD5 signature verification (all files in chain) */
    if (verify_signature)
    {
        for (i = 0; i < raw_coll.num_sequences; ++i)
        {
            const char *fpath = raw_coll.sequences[i].file_path;
            if (!fpath)
                continue;
            rc = pulseq_verify_signature(fpath);
            if (PULSEG_FAILED(rc))
            {
                diag->code = rc;
                pulseg__diag_printf(diag, " subsequence=%d", i);
                goto fail;
            }
        }
    }

    rc = pulseg_convert_collection(
        collection,
        diag,
        raw_coll.sequences,
        raw_coll.num_sequences,
        opts,
        parse_labels,
        num_averages);
    if (PULSEG_FAILED(diag->code))
    {
        rc = diag->code;
        goto fail;
    }

    pulseq_file_set_free(&raw_coll);

    /* Write cache (best-effort) */
    if (cache_binary)
        pulseg__write_cache(collection, file_path, opts);

    *out_coll = collection;
    return PULSEG_SUCCESS;

fail:
    pulseq_file_set_free(&raw_coll);
    PULSEG_FREE(collection);
    return rc;
}

/* ================================================================== */
/*  pulseg_read_from_buffers (public entry point)                  */
/* ================================================================== */

int pulseg_read_from_buffers(
    pulseg_collection **out_coll,
    pulseg_diagnostic *diag,
    const char *const *buffers,
    const int *buffer_sizes,
    int num_buffers,
    const pulseg_opts *opts,
    int parse_labels,
    int num_averages)
{
    pulseq_file_set raw_coll;
    pulseq_raster raster;
    pulseg_collection *collection;
    int rc, i;

    raw_coll.num_sequences = 0;
    raw_coll.sequences = NULL;
    raw_coll.base_path = NULL;

    if (!out_coll || !diag || !buffers || !buffer_sizes || !opts)
        return PULSEG_ERR_NULL_POINTER;
    if (num_buffers < 1)
        return PULSEG_ERR_INVALID_ARGUMENT;

    *out_coll = NULL;
    pulseg_diagnostic_init(diag);
    pulseg_opts_get_design_raster(&raster, opts);

    /* Build raw collection from in-memory buffers */
    raw_coll.sequences = (pulseq_file *)PULSEG_ALLOC(num_buffers * sizeof(pulseq_file));
    if (!raw_coll.sequences)
        return PULSEG_ERR_ALLOC_FAILED;
    raw_coll.num_sequences = 0;
    raw_coll.base_path = NULL;

    for (i = 0; i < num_buffers; ++i)
    {
        FILE *tmp;
        if (!buffers[i] || buffer_sizes[i] < 0)
        {
            rc = PULSEG_ERR_INVALID_ARGUMENT;
            diag->code = rc;
            goto fail_raw;
        }

        tmp = tmpfile();
        if (!tmp)
        {
            rc = PULSEG_ERR_FILE_NOT_FOUND;
            diag->code = rc;
            goto fail_raw;
        }

        if (buffer_sizes[i] > 0)
        {
            if ((int)fwrite(buffers[i], 1, (size_t)buffer_sizes[i], tmp) != buffer_sizes[i])
            {
                fclose(tmp);
                rc = PULSEG_ERR_FILE_NOT_FOUND;
                diag->code = rc;
                goto fail_raw;
            }
        }
        rewind(tmp);

        pulseq_file_init(&raw_coll.sequences[i], &raster);
        rc = pulseq_read_from_buffer(&raw_coll.sequences[i], tmp);
        fclose(tmp);
        if (PULSEG_FAILED(rc))
        {
            diag->code = rc;
            goto fail_raw;
        }
        raw_coll.num_sequences = i + 1;
    }

    /* Heap-allocate the opaque collection */
    collection = pulseg_collection_alloc();
    if (!collection)
    {
        rc = PULSEG_ERR_ALLOC_FAILED;
        goto fail_raw;
    }

    rc = pulseg_convert_collection(
        collection,
        diag,
        raw_coll.sequences,
        raw_coll.num_sequences,
        opts,
        parse_labels,
        num_averages);
    if (PULSEG_FAILED(diag->code))
    {
        rc = diag->code;
        goto fail_coll;
    }

    pulseq_file_set_free(&raw_coll);

    *out_coll = collection;
    return PULSEG_SUCCESS;

fail_coll:
    PULSEG_FREE(collection);
fail_raw:
    pulseq_file_set_free(&raw_coll);
    return rc;
}