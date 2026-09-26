/**
 * @file collection.cpp
 * @brief Converting parsed pulseq files into a pulseg collection.
 *
 * Drives the per-subsequence pipeline -- deduplication, TR detection,
 * segmentation, label tables and timing -- chains the results into one
 * collection and runs the cross-subsequence consistency checks that must hold
 * before any consumer sees it.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* Conversion keeps C linkage: it is the entry point pulseg_convert.h
 * declares. */
extern "C"
{
#include "pulseg_internal.h"
#include "pulseg.h"
#include "pulseg_convert.h"
}

static int convert_collection(
    pulseg_collection *coll,
    pulseg_diagnostic *diag,
    const pulseq_file *files,
    int n,
    const pulseg_opts *opts,
    int parse_labels,
    int adopt_shapes);

int pulseg_convert_collection(
    pulseg_collection *coll,
    pulseg_diagnostic *diag,
    const pulseq_file *files,
    int n,
    const pulseg_opts *opts,
    int parse_labels)
{
    return convert_collection(coll, diag, files, n, opts, parse_labels, 0);
}

/* @p adopt_shapes moves the files' sample buffers into the descriptors
 * instead of copying them, for a caller that frees @p files right after. */
static int convert_collection(
    pulseg_collection *coll,
    pulseg_diagnostic *diag,
    const pulseq_file *files,
    int n,
    const pulseg_opts *opts,
    int parse_labels,
    int adopt_shapes)
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

        result = pulseg__get_unique_blocks(
            &desc,
            &files[i],
            opts,
            adopt_shapes ? files[i].shapes_library : NULL);
        if (PULSEG_FAILED(result))
        {
            diag->code = result;
            goto fail;
        }

        result = pulseg__get_tr_in_sequence(&desc, &files[i], diag);
        if (PULSEG_FAILED(diag->code))
            goto fail;

        result = pulseg__compute_variable_grad_flags(&desc);
        if (PULSEG_FAILED(result))
        {
            diag->code = result;
            goto fail;
        }

        result = pulseg__build_exec_stream(&desc, diag);
        if (PULSEG_FAILED(diag->code))
            goto fail;

        /* Scan-table-only segmentation */
        result = pulseg__get_exec_stream_segments(&desc, &files[i], diag, opts);
        if (PULSEG_FAILED(diag->code))
            goto fail;

        /* get_exec_stream_segments may adjust TR topology (e.g. sparse
         * multipass patterns can update tr_descriptor.tr_size). Refresh the
         * variable-gradient flags so they index the final TR size. */
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

        result = pulseg__build_waves(&desc);
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
         * from are not loaded on every cache path (the pulse-generation load
         * reads neither). */
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

    /* Cross-subsequence consistency. */
    rc = pulseg__check_consistency(
        coll, opts ? opts->allow_variable_rf_amplitude : 1, diag);
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
