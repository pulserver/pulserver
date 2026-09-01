/**
 * @file pulseg_getters.c
 * @brief Read-only accessors over a loaded collection.
 *
 * Every getter here is a projection of the internal descriptor tables onto
 * the public value types (pulseg_segment_info, pulseg_block_info, the
 * waveform accessors, ...), so consumers never see pulseg_internal.h.
 *
 * All of them take (seg_idx, blk_idx) in COLLECTION coordinates -- segment
 * indices are global across chained subsequences -- and go through
 * resolve_segment() / resolve_block(), which map those onto the owning
 * descriptor plus subsequence-local indices. A getter that bypasses those
 * two helpers is a bug.
 */

#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "pulseg_internal.h"
#include "pulseg.h"

/* ================================================================== */
/*  Resolve helpers                                                   */
/* ================================================================== */

static int resolve_segment(
    const pulseg_collection *coll,
    const pulseg_sequence_descriptor **out_desc,
    int *out_local_seg,
    int seg_idx)
{
    int i, num_segs, global_idx;

    if (!coll || seg_idx < 0 || seg_idx >= coll->total_unique_segments)
        return 0;

    /* Deduplicated global segment space: resolve through the representative
     * (subseq, local) map built by pulseg__build_segment_remap(). */
    if (coll->seg_repr_subseq && coll->seg_repr_local)
    {
        int s = coll->seg_repr_subseq[seg_idx];
        int l = coll->seg_repr_local[seg_idx];
        if (s < 0 || s >= coll->num_subsequences)
            return 0;
        if (out_desc)
            *out_desc = &coll->descriptors[s];
        if (out_local_seg)
            *out_local_seg = l;
        return 1;
    }

    /* Fallback (remap not built): contiguous per-subsequence offset walk. */
    global_idx = 0;
    for (i = 0; i < coll->num_subsequences; ++i)
    {
        num_segs = coll->descriptors[i].num_unique_segments;
        if (seg_idx < global_idx + num_segs)
        {
            if (out_desc)
                *out_desc = &coll->descriptors[i];
            if (out_local_seg)
                *out_local_seg = seg_idx - global_idx;
            return 1;
        }
        global_idx += num_segs;
    }
    return 0;
}

static int resolve_block(
    const pulseg_collection *coll,
    const pulseg_sequence_descriptor **out_desc,
    const pulseg_virtual_segment **out_seg,
    int *out_local_blk,
    int seg_idx,
    int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg;
    const pulseg_virtual_segment *seg;

    desc = NULL;
    if (!resolve_segment(coll, &desc, &local_seg, seg_idx))
        return 0;

    seg = &desc->segment_definitions[local_seg];
    if (blk_idx < 0 || blk_idx >= seg->num_blocks)
        return 0;

    if (out_desc)
        *out_desc = desc;
    if (out_seg)
        *out_seg = seg;
    if (out_local_blk)
        *out_local_blk = blk_idx;
    return 1;
}

/* The per-(segment, block-position) record frozen at parse time from the
 * segment's representative (max-energy) scan instance -- see
 * pulseg_structure.c step 11e.  NULL when the descriptor predates the record
 * (no representative resolvable); callers then use the same fallbacks the
 * pre-computation resolution used when no instance was found. */
static const pulseg_block_initial_state *resolve_initial_state(
    const pulseg_sequence_descriptor *desc,
    const pulseg_virtual_segment *seg,
    int local_blk)
{
    if (!desc || !seg || !seg->initial_states)
        return NULL;
    if (local_blk < 0 || local_blk >= seg->num_blocks)
        return NULL;

    return &seg->initial_states[local_blk];
}

static int resolve_grad_def_via_max_energy_instance(
    const pulseg_sequence_descriptor *desc,
    const pulseg_virtual_segment *seg,
    int local_blk,
    int axis)
{
    const pulseg_block_initial_state *st;

    if (axis < PULSEG_GRAD_AXIS_X || axis > PULSEG_GRAD_AXIS_Z)
        return -1;

    st = resolve_initial_state(desc, seg, local_blk);
    if (!st)
        return -1;

    return st->grad_def_id[axis];
}

/* ================================================================== */
/*  Subsequence accessors (internal helpers for batch getters)         */
/* ================================================================== */

static int get_num_subsequences(const pulseg_collection *coll)
{
    if (!coll)
        return 0;
    return coll->num_subsequences;
}

static float get_tr_duration_us(const pulseg_collection *coll, int subseq_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_descriptor *tr;

    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0.0f;

    desc = &coll->descriptors[subseq_idx];
    tr = &desc->tr_descriptor;
    return tr->tr_duration_us;
}

static int get_num_trs(const pulseg_collection *coll, int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].tr_descriptor.num_trs;
}

static int get_tr_size(const pulseg_collection *coll, int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].tr_descriptor.tr_size;
}

static int get_num_unique_adcs(const pulseg_collection *coll, int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].num_unique_adcs;
}

static int is_pmc_enabled(const pulseg_collection *coll, int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].enable_pmc;
}

static int get_subseq_segment_offset(const pulseg_collection *coll, int subseq_idx)
{
    int i, offset = 0;
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    for (i = 0; i < subseq_idx; ++i)
        offset += coll->descriptors[i].num_unique_segments;
    return offset;
}

static float get_total_duration_us(const pulseg_collection *coll)
{
    if (!coll)
        return 0.0f;
    return coll->total_duration_us;
}

int pulseg_get_scan_time(const pulseg_collection *coll, pulseg_scan_time_info *info)
{
    int i, n, bt_idx;
    const pulseg_sequence_descriptor *desc;
    const pulseg_block_table_element *bte;
    const pulseg_base_block *bdef;
    int prev_seg, cur_seg;

    if (!coll || !info)
        return PULSEG_ERR_NULL_POINTER;
    if (coll->num_subsequences <= 0)
        return PULSEG_ERR_COLLECTION_EMPTY;

    info->total_duration_us = 0.0f;
    info->total_segment_boundaries = 0;

    for (i = 0; i < coll->num_subsequences; ++i)
    {
        desc = &coll->descriptors[i];
        prev_seg = -1;

        for (n = 0; n < desc->exec_stream_len; ++n)
        {
            bt_idx = pulseg__exec_block_idx(desc, n);
            bte = &desc->block_table[bt_idx];
            bdef = &desc->base_blocks[bte->id];

            /* Duration: pure delay uses instance value, normal uses definition */
            info->total_duration_us +=
                (bte->duration_us >= 0) ? (float)bte->duration_us : (float)bdef->duration_us;

            /* Count segment boundaries (transitions) */
            cur_seg = pulseg__exec_seg_id(desc, n);
            if (cur_seg >= 0 && cur_seg != prev_seg)
                info->total_segment_boundaries += 1;
            prev_seg = cur_seg;
        }
    }

    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  RF accessors                                                      */
/* ================================================================== */

static int get_num_unique_rf(const pulseg_collection *coll, int subseq_idx)
{
    if (!coll || subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].num_unique_rfs;
}

int pulseg_get_rf_stats(
    const pulseg_collection *coll,
    pulseg_rf_stats *stats,
    int subseq_idx,
    int rf_idx)
{
    const pulseg_sequence_descriptor *desc;

    if (!coll || !stats)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;
    desc = &coll->descriptors[subseq_idx];
    if (rf_idx < 0 || rf_idx >= desc->num_unique_rfs)
        return PULSEG_ERR_INVALID_ARGUMENT;

    *stats = desc->rf_definitions[rf_idx].stats;
    return PULSEG_SUCCESS;
}

/*
 * pulseg_get_tr_rf_ids --
 *   Return an array of RF definition IDs for each block position
 *   within the first main TR.  Blocks without RF get -1.
 *
 *   out_rf_ids must point to a pre-allocated array of tr_size ints.
 *   Returns tr_size on success, negative error code on failure.
 */
int pulseg_get_tr_rf_ids(const pulseg_collection *coll, int *out_rf_ids, int subseq_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_descriptor *trd;
    const pulseg_block_table_element *bte;
    int i, block_idx, tr_size;

    if (!coll || !out_rf_ids)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[subseq_idx];
    trd = &desc->tr_descriptor;
    tr_size = trd->tr_size;

    for (i = 0; i < tr_size; ++i)
    {
        block_idx = i;
        bte = &desc->block_table[block_idx];
        if (bte->rf_id >= 0 && bte->rf_id < desc->rf_table_size)
            out_rf_ids[i] = desc->rf_table[bte->rf_id].id;
        else
            out_rf_ids[i] = -1;
    }

    return tr_size;
}

/* ================================================================== */
/*  Variable-RF-amplitude safety arrays                                */
/* ================================================================== */

/*
 * pulseg__rf_pulse_at --
 *   RF occurrence at absolute walk position @p st (a block_table index, or
 *   an exec_stream slot when @p use_exec_stream). Returns 0 (no RF / out of
 *   range) or 1, filling *out_amp (|amplitude|, Hz) and *out_rf_def_id
 *   (index into desc->rf_definitions) when non-NULL.
 */
static int pulseg__rf_pulse_at(
    const pulseg_sequence_descriptor *desc,
    int use_exec_stream,
    int st,
    float *out_amp,
    int *out_rf_def_id)
{
    int blk_idx, rf_def_id;
    const pulseg_block_table_element *bte;

    if (use_exec_stream)
    {
        if (st < 0 || st >= desc->exec_stream_len)
            return 0;
        blk_idx = pulseg__exec_block_idx(desc, st);
    }
    else
    {
        blk_idx = st;
    }
    if (blk_idx < 0 || blk_idx >= desc->num_blocks)
        return 0;

    bte = &desc->block_table[blk_idx];
    if (bte->rf_id < 0 || bte->rf_id >= desc->rf_table_size)
        return 0;

    rf_def_id = desc->rf_table[bte->rf_id].id;
    if (rf_def_id < 0 || rf_def_id >= desc->num_unique_rfs)
        return 0;

    if (out_amp)
    {
        float a = desc->rf_table[bte->rf_id].amplitude;
        *out_amp = (a < 0.0f) ? -a : a;
    }
    if (out_rf_def_id)
        *out_rf_def_id = rf_def_id;

    return 1;
}

/*
 * pulseg__rf_position_max --
 *   Worst-case |amplitude| at canonical position @p pos across every TR
 *   instance (degenerate path) or pass (non-degenerate path). Feeds
 *   pulseg_rf_stats.peak_amplitude_hz: peak-only consumers (GE peakB1())
 *   need per-position dominance across every instance --
 *   Â_pos >= |A_{u,pos}| for every instance u by construction.
 *
 *   The walk mirrors pulseg_get_rf_array's own instance layout so the
 *   envelope and the canonical-instance value coincide exactly for a
 *   periodic sequence (Â_pos == |A_{0,pos}|), which is the regression
 *   contract.
 */
static float pulseg__rf_position_max(
    const pulseg_sequence_descriptor *desc,
    int use_exec_stream,
    int base_start,
    int unit_size,
    int num_units,
    int pos)
{
    float best = 0.0f;
    int u;

    for (u = 0; u < num_units; ++u)
    {
        float amp;
        if (pulseg__rf_pulse_at(
                desc,
                use_exec_stream,
                base_start + u * unit_size + pos,
                &amp,
                NULL) &&
            amp > best)
            best = amp;
    }
    return best;
}

/*
 * pulseg__rf_worst_b1rms_instance --
 *   Among @p num_units TR instances (or passes) of @p count positions each,
 *   find the instance with the largest time-averaged B1^2 energy,
 *   Sum_j A_{u,j}^2 * total_b1sq_power_j. TR duration is constant across
 *   instances here (only RF amplitude may vary -- pulseg_core.c's
 *   periodicity gate), so it is a common divisor and dropped from the
 *   ranking; the sum alone orders instances by B1rms.
 *
 *   total_b1sq_power is integral |h_norm(t)|^2 dt of the unit-peak-
 *   normalised pulse shape (pulseg_dedup.c), so A^2 * total_b1sq_power
 *   recovers integral |B1(t)|^2 dt for this occurrence directly -- no
 *   separate duration factor needed.
 *
 *   Returns the winning instance index in [0, num_units); 0 when num_units
 *   <= 1 or no RF pulses are found anywhere (degenerates to the canonical
 *   instance).
 *
 *   Bounds are small in practice (num_units is the periodicity-checked TR
 *   count, not multiplied by NEX; count is one TR's block span) --
 *   O(num_units * count) float multiply-adds, no memoization needed.
 */
static int pulseg__rf_worst_b1rms_instance(
    const pulseg_sequence_descriptor *desc,
    int use_exec_stream,
    int base_start,
    int unit_size,
    int num_units,
    int count)
{
    int u, pos, best_u;
    double best_score;

    best_u = 0;
    best_score = -1.0;

    for (u = 0; u < num_units; ++u)
    {
        double score = 0.0;

        for (pos = 0; pos < count; ++pos)
        {
            float amp;
            int rf_def_id;

            if (!pulseg__rf_pulse_at(
                    desc,
                    use_exec_stream,
                    base_start + u * unit_size + pos,
                    &amp,
                    &rf_def_id))
                continue;

            score += (double)amp * (double)amp *
                (double)desc->rf_definitions[rf_def_id].stats.total_b1sq_power;
        }

        if (score > best_score)
        {
            best_score = score;
            best_u = u;
        }
    }

    return best_u;
}

/* ================================================================== */
/*  pulseg_get_rf_array --                                         */
/*    Build an ordered array of RF stats for a TR region.             */
/*    Each entry gets the base rf_stats patched with the actual       */
/*    amplitude from the rf_table and the repetition count for        */
/*    that region.  The library allocates; caller must free().        */
/* ================================================================== */
int pulseg_get_rf_array(const pulseg_collection *coll, pulseg_rf_stats **out_pulses, int subseq_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_descriptor *trd;
    const pulseg_block_table_element *bte;
    const pulseg_rf_definition *rfdef;
    int start, count, num_instances;
    int use_exec_stream;
    int i, n, num_rf;
    /* Variable-RF-amplitude instance walk (only used when
     * rf_amplitude_variable): the u-th instance's position p lives at
     * env_base + u*env_stride + p. env_base == start (u=0 IS the canonical
     * instance already extracted below), so the walk and the canonical
     * fill stay position-aligned. */
    int env_base, env_stride, env_units;
    int winner_u;

    if (!coll || !out_pulses)
        return PULSEG_ERR_NULL_POINTER;
    *out_pulses = NULL;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[subseq_idx];
    trd = &desc->tr_descriptor;

    use_exec_stream = 0;
    {
        start = 0;
        count = trd->tr_size;
        num_instances = trd->num_trs;
        if (num_instances < 0)
            num_instances = 0;

        env_base = start;
        env_stride = trd->tr_size;
        env_units = trd->num_trs;
    }
    if (env_units < 1)
        env_units = 1;

    /* Clamp to available block range */
    if (use_exec_stream)
    {
        if (start + count > desc->exec_stream_len)
            count = desc->exec_stream_len - start;
    }
    else if (start + count > desc->num_blocks)
    {
        count = desc->num_blocks - start;
    }
    if (count < 0)
        count = 0;

    /* Instance carrying the worst (largest) time-averaged B1^2 energy over
     * this TR/pass, real amplitudes only -- see
     * pulseg__rf_worst_b1rms_instance(). u=0 (the canonical instance
     * already selected by start/count above) when the sequence is
     * periodic, which keeps this a no-op for the regression contract. */
    winner_u = 0;
    if (desc->rf_amplitude_variable && env_units > 1)
        winner_u = pulseg__rf_worst_b1rms_instance(
            desc,
            use_exec_stream,
            env_base,
            env_stride,
            env_units,
            count);

    /* Pass 1: count RF-bearing blocks */
    num_rf = 0;
    for (i = 0; i < count; ++i)
    {
        int blk_idx = use_exec_stream ? pulseg__exec_block_idx(desc, start + i) : (start + i);
        bte = &desc->block_table[blk_idx];
        if (bte->rf_id >= 0 && bte->rf_id < desc->rf_table_size)
        {
            int id = desc->rf_table[bte->rf_id].id;
            if (id >= 0 && id < desc->num_unique_rfs)
                num_rf++;
        }
    }

    if (num_rf == 0)
        return 0;

    /* Allocate output array (caller frees with PULSEG_FREE) */
    *out_pulses = (pulseg_rf_stats *)PULSEG_ALLOC((size_t)num_rf * sizeof(pulseg_rf_stats));
    if (!*out_pulses)
        return PULSEG_ERR_ALLOC_FAILED;

    /* Pass 2: fill entries */
    n = 0;
    for (i = 0; i < count; ++i)
    {
        int blk_idx = use_exec_stream ? pulseg__exec_block_idx(desc, start + i) : (start + i);
        int rf_def_id;
        float act_amp;

        bte = &desc->block_table[blk_idx];
        if (bte->rf_id < 0 || bte->rf_id >= desc->rf_table_size)
            continue;

        rf_def_id = desc->rf_table[bte->rf_id].id;
        if (rf_def_id < 0 || rf_def_id >= desc->num_unique_rfs)
            continue;

        rfdef = &desc->rf_definitions[rf_def_id];

        /* Hard-copy base stats */
        (*out_pulses)[n] = rfdef->stats;

        /* Patch event-specific amplitude-dependent stats from rf_table.
         * See pulseg_rf_stats.act_amplitude_hz / peak_amplitude_hz for the
         * worst-B1rms-instance vs positional-max split; both collapse to
         * the canonical instance's value when !rf_amplitude_variable
         * (periodic sequence), which is the regression contract. */
        if (desc->rf_amplitude_variable && env_units > 1)
        {
            float winner_amp = 0.0f;
            (void)pulseg__rf_pulse_at(
                desc,
                use_exec_stream,
                env_base + winner_u * env_stride + i,
                &winner_amp,
                NULL);
            act_amp = winner_amp;
            (*out_pulses)[n].peak_amplitude_hz =
                pulseg__rf_position_max(desc, use_exec_stream, env_base, env_stride, env_units, i);
        }
        else
        {
            act_amp = desc->rf_table[bte->rf_id].amplitude;
            act_amp = (act_amp >= 0.0f) ? act_amp : -act_amp;
            (*out_pulses)[n].peak_amplitude_hz = act_amp;
        }
        (*out_pulses)[n].act_amplitude_hz = act_amp;

        /* base_amplitude_hz retains definition-level nominal amplitude
         * from the hard-copy above (rfdef->stats.base_amplitude_hz). */
        (*out_pulses)[n].flip_angle_rad = rfdef->stats.flip_angle_rad;

        /* Set repetition count */
        (*out_pulses)[n].num_instances = num_instances;

        /* The safety group (TRID) rides on the per-occurrence block-table
         * entry, never on the deduplicated rfdef->stats hard-copied above. */
        (*out_pulses)[n].trid = bte->trid;

        n++;
    }

    return n;
}

/* ================================================================== */
/*  pulseg_get_rf_event_array --                                      */
/*    Build an ordered array of RF event identities for a TR region.  */
/*    Walk logic is index-aligned with pulseg_get_rf_array() above    */
/*    (same use_exec_stream/start/count selection, same clamps, same   */
/*    skip conditions) -- keep the two walks in sync.                 */
/* ================================================================== */
int pulseg_get_rf_event_array(
    const pulseg_collection *coll,
    pulseg_rf_event **out_events,
    int subseq_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_descriptor *trd;
    const pulseg_block_table_element *bte;
    const pulseg_rf_definition *rfdef;
    int start, count, num_instances;
    int use_exec_stream;
    int i, n, num_rf;

    if (!coll || !out_events)
        return PULSEG_ERR_NULL_POINTER;
    *out_events = NULL;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[subseq_idx];
    trd = &desc->tr_descriptor;

    use_exec_stream = 0;
    {
        start = 0;
        count = trd->tr_size;
        num_instances = trd->num_trs;
        if (num_instances < 0)
            num_instances = 0;
    }

    (void)num_instances; /* not part of the event identity */

    /* Clamp to available block range */
    if (use_exec_stream)
    {
        if (start + count > desc->exec_stream_len)
            count = desc->exec_stream_len - start;
    }
    else if (start + count > desc->num_blocks)
    {
        count = desc->num_blocks - start;
    }
    if (count < 0)
        count = 0;

    /* Pass 1: count RF-bearing blocks */
    num_rf = 0;
    for (i = 0; i < count; ++i)
    {
        int blk_idx = use_exec_stream ? pulseg__exec_block_idx(desc, start + i) : (start + i);
        bte = &desc->block_table[blk_idx];
        if (bte->rf_id >= 0 && bte->rf_id < desc->rf_table_size)
        {
            int id = desc->rf_table[bte->rf_id].id;
            if (id >= 0 && id < desc->num_unique_rfs)
                num_rf++;
        }
    }

    if (num_rf == 0)
        return 0;

    /* Allocate output array (caller frees with PULSEG_FREE) */
    *out_events = (pulseg_rf_event *)PULSEG_ALLOC((size_t)num_rf * sizeof(pulseg_rf_event));
    if (!*out_events)
        return PULSEG_ERR_ALLOC_FAILED;

    /* Pass 2: fill entries */
    n = 0;
    for (i = 0; i < count; ++i)
    {
        int blk_idx = use_exec_stream ? pulseg__exec_block_idx(desc, start + i) : (start + i);
        int rf_def_id;
        float act_amp;

        bte = &desc->block_table[blk_idx];
        if (bte->rf_id < 0 || bte->rf_id >= desc->rf_table_size)
            continue;

        rf_def_id = desc->rf_table[bte->rf_id].id;
        if (rf_def_id < 0 || rf_def_id >= desc->num_unique_rfs)
            continue;

        rfdef = &desc->rf_definitions[rf_def_id];

        act_amp = desc->rf_table[bte->rf_id].amplitude;

        (*out_events)[n].rf_def_id = rf_def_id;
        (*out_events)[n].amplitude_hz = (act_amp >= 0.0f) ? act_amp : -act_amp;
        (*out_events)[n].rf_shim_id = bte->rf_shim_id;
        (*out_events)[n].num_channels = (rfdef->num_channels > 1) ? rfdef->num_channels : 1;

        n++;
    }

    return n;
}

/* ================================================================== */
/*  TRID (safety-group) accessors                                     */
/* ================================================================== */

int pulseg_get_tr_groups(
    const pulseg_collection *coll,
    pulseg_tr_group **out_groups,
    int subseq_idx)
{
    const pulseg_sequence_descriptor *desc;
    int *run_trid = NULL;
    int *run_start = NULL;
    int *run_len = NULL;
    int num_runs = 0;
    int *distinct_ids = NULL;
    int num_distinct = 0;
    pulseg_tr_group *groups = NULL;
    int i, m;
    int ret = 0;

    if (!coll || !out_groups)
        return PULSEG_ERR_NULL_POINTER;
    *out_groups = NULL;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[subseq_idx];
    if (desc->exec_stream_len <= 0 || !desc->exec_runs || !desc->block_table)
        return 0;

    /* ---- Pass 1: split the materialized scan table into maximal runs of
     * consecutive positions sharing the same non-zero TRID. Worst
     * case one run per position, so this bound is always safe. ---- */
    run_trid = (int *)PULSEG_ALLOC((size_t)desc->exec_stream_len * sizeof(int));
    run_start = (int *)PULSEG_ALLOC((size_t)desc->exec_stream_len * sizeof(int));
    run_len = (int *)PULSEG_ALLOC((size_t)desc->exec_stream_len * sizeof(int));
    if (!run_trid || !run_start || !run_len)
    {
        ret = PULSEG_ERR_ALLOC_FAILED;
        goto cleanup;
    }

    {
        int prev_mid = 0;
        for (i = 0; i < desc->exec_stream_len; ++i)
        {
            int blk = pulseg__exec_block_idx(desc, i);
            int mid = desc->block_table[blk].trid;
            /* A run also breaks at every main-region TR boundary
             * (exec_stream_tr_start[i]==1), not just on a TRID
             * transition -- otherwise adjacent NEX/pass repeats of the
             * same main-region group (identical TRID across the repeat
             * boundary, since TRID is sticky and the author never re-SETs
             * it) would silently merge into one giant run instead of being
             * counted as separate occurrences. */
            int is_tr_start = pulseg__exec_tr_start(desc, i);
            if (mid != 0 && (mid != prev_mid || is_tr_start))
            {
                /* New run starts here. */
                run_trid[num_runs] = mid;
                run_start[num_runs] = i;
                run_len[num_runs] = 0;
                num_runs++;
            }
            if (mid != 0)
                run_len[num_runs - 1]++;
            prev_mid = mid;
        }
    }

    if (num_runs == 0)
    {
        ret = 0;
        goto cleanup;
    }

    /* ---- Pass 2: distinct TRIDs, first-seen order (num_runs is a
     * safe upper bound). ---- */
    distinct_ids = (int *)PULSEG_ALLOC((size_t)num_runs * sizeof(int));
    if (!distinct_ids)
    {
        ret = PULSEG_ERR_ALLOC_FAILED;
        goto cleanup;
    }
    for (i = 0; i < num_runs; ++i)
    {
        int found = 0;
        for (m = 0; m < num_distinct; ++m)
        {
            if (distinct_ids[m] == run_trid[i])
            {
                found = 1;
                break;
            }
        }
        if (!found)
            distinct_ids[num_distinct++] = run_trid[i];
    }

    groups = (pulseg_tr_group *)PULSEG_ALLOC((size_t)num_distinct * sizeof(pulseg_tr_group));
    if (!groups)
    {
        ret = PULSEG_ERR_ALLOC_FAILED;
        goto cleanup;
    }

    /* ---- Per distinct TRID: verify every occurrence after the first
     * is structurally identical to the first, then dedup. ---- */
    for (m = 0; m < num_distinct; ++m)
    {
        int mid = distinct_ids[m];
        int ref_run = -1;
        int ref_len = 0;
        int one_instance_duration_us = 0;
        int n_inst = 0;
        int r;

        for (r = 0; r < num_runs; ++r)
        {
            if (run_trid[r] != mid)
                continue;

            if (ref_run < 0)
            {
                /* First occurrence: this is the reference. */
                ref_run = r;
                ref_len = run_len[r];
                for (i = 0; i < ref_len; ++i)
                {
                    int blk = pulseg__exec_block_idx(desc, run_start[r] + i);
                    one_instance_duration_us +=
                        desc->base_blocks[desc->block_table[blk].id].duration_us;
                }
            }
            else
            {
                /* Subsequent occurrence: must match the reference exactly
                 * (length, and per-position content signature + order). */
                if (run_len[r] != ref_len)
                {
                    ret = PULSEG_ERR_TRID_STRUCTURAL_MISMATCH;
                    goto cleanup;
                }
                for (i = 0; i < ref_len; ++i)
                {
                    /* The block-definition id is the whole structural
                     * signature -- duration, RF, gradients and ADC -- and
                     * carries none of the amplitude, phase or rotation that
                     * may legitimately vary per repeat or slice. */
                    int ref_blk = pulseg__exec_block_idx(desc, run_start[ref_run] + i);
                    int cur_blk = pulseg__exec_block_idx(desc, run_start[r] + i);

                    if (desc->block_table[ref_blk].id != desc->block_table[cur_blk].id)
                    {
                        ret = PULSEG_ERR_TRID_STRUCTURAL_MISMATCH;
                        goto cleanup;
                    }
                }
            }
            n_inst++;
        }

        groups[m].trid = mid;
        groups[m].one_instance_duration_us = one_instance_duration_us;
        groups[m].num_instances = n_inst;
        {
            /* n_inst * one_instance_duration_us can overflow a 32-bit int
             * for large num_trs even when one_instance_duration_us alone
             * passes the caller's ceiling check -- same defensive clamp
             * pattern as PulserverImplementationPredownload.e's nettime_us
             * computation. */
            double total_d = (double)n_inst * (double)one_instance_duration_us;
            if (total_d > 2.0e9)
                total_d = 2.0e9;
            groups[m].total_duration_us = (int)total_d;
        }
    }

    *out_groups = groups;
    groups = NULL; /* ownership transferred to caller */
    ret = num_distinct;

cleanup:
    if (run_trid)
        PULSEG_FREE(run_trid);
    if (run_start)
        PULSEG_FREE(run_start);
    if (run_len)
        PULSEG_FREE(run_len);
    if (distinct_ids)
        PULSEG_FREE(distinct_ids);
    if (groups)
        PULSEG_FREE(groups);
    return ret;
}

/* ================================================================== */
/*  ADC collection accessors                                          */
/* ================================================================== */

static int get_total_readouts(const pulseg_collection *coll)
{
    if (!coll)
        return 0;
    return coll->total_readouts;
}

static int get_max_adc_samples(const pulseg_collection *coll)
{
    int i, j, max_samples;

    if (!coll)
        return 0;

    max_samples = 0;
    for (i = 0; i < coll->num_subsequences; ++i)
    {
        for (j = 0; j < coll->descriptors[i].num_unique_adcs; ++j)
        {
            if (coll->descriptors[i].adc_definitions[j].num_samples > max_samples)
                max_samples = coll->descriptors[i].adc_definitions[j].num_samples;
        }
    }
    return max_samples;
}

static int get_adc_dwell_ns(const pulseg_collection *coll, int adc_idx)
{
    int i, global_idx, num_adcs, local;

    if (!coll || adc_idx < 0 || adc_idx >= coll->total_unique_adcs)
        return 0;

    global_idx = 0;
    for (i = 0; i < coll->num_subsequences; ++i)
    {
        num_adcs = coll->descriptors[i].num_unique_adcs;
        if (adc_idx < global_idx + num_adcs)
        {
            local = adc_idx - global_idx;
            return coll->descriptors[i].adc_definitions[local].dwell_time;
        }
        global_idx += num_adcs;
    }
    return 0;
}

static int get_adc_num_samples(const pulseg_collection *coll, int adc_idx)
{
    int i, global_idx, num_adcs, local;

    if (!coll || adc_idx < 0 || adc_idx >= coll->total_unique_adcs)
        return 0;

    global_idx = 0;
    for (i = 0; i < coll->num_subsequences; ++i)
    {
        num_adcs = coll->descriptors[i].num_unique_adcs;
        if (adc_idx < global_idx + num_adcs)
        {
            local = adc_idx - global_idx;
            return coll->descriptors[i].adc_definitions[local].num_samples;
        }
        global_idx += num_adcs;
    }
    return 0;
}

/* ================================================================== */
/*  Segment accessors (internal helpers for batch getters)             */
/* ================================================================== */

static int get_num_segments(const pulseg_collection *coll)
{
    if (!coll)
        return 0;
    return coll->total_unique_segments;
}

static int get_segment_duration_us(const pulseg_collection *coll, int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg, k, total;
    const pulseg_virtual_segment *seg;
    const pulseg_base_block *bdef;

    if (!resolve_segment(coll, &desc, &local_seg, seg_idx))
        return -1;

    seg = &desc->segment_definitions[local_seg];

    /* Segment-definition pure delays are canonicalized to one block raster;
     * scan-loop instances still expose per-instance delay from block table. */
    if (seg->num_blocks == 1)
    {
        bdef = &desc->base_blocks[seg->unique_block_indices[0]];
        if (bdef->rf_id == -1 && bdef->gx_id == -1 && bdef->gy_id == -1 && bdef->gz_id == -1 &&
            bdef->adc_id == -1)
        {
            if (desc->block_raster_us > 0.0f)
                return (int)(desc->block_raster_us);
            return 0;
        }
    }

    total = 0;
    for (k = 0; k < seg->num_blocks; ++k)
        total += desc->base_blocks[seg->unique_block_indices[k]].duration_us;

    return total;
}

static int is_segment_pure_delay(const pulseg_collection *coll, int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg;
    const pulseg_virtual_segment *seg;
    const pulseg_base_block *bdef;

    if (!resolve_segment(coll, &desc, &local_seg, seg_idx))
        return -1;

    seg = &desc->segment_definitions[local_seg];
    if (seg->num_blocks == 1)
    {
        bdef = &desc->base_blocks[seg->unique_block_indices[0]];
        if (bdef->rf_id == -1 && bdef->gx_id == -1 && bdef->gy_id == -1 && bdef->gz_id == -1 &&
            bdef->adc_id == -1 && !seg->has_digitalout[0])
            return 1;
    }
    return 0;
}

static int get_segment_num_blocks(const pulseg_collection *coll, int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg;

    if (!resolve_segment(coll, &desc, &local_seg, seg_idx))
        return -1;

    return desc->segment_definitions[local_seg].num_blocks;
}

static int get_segment_start_block(const pulseg_collection *coll, int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg;

    if (!resolve_segment(coll, &desc, &local_seg, seg_idx))
        return -1;

    return desc->segment_definitions[local_seg].start_block;
}

/* ================================================================== */
/*  Segment table queries                                             */
/* ================================================================== */

int pulseg_get_canonical_segment_sequence(
    const pulseg_collection *coll,
    int *out_ids,
    int subseq_idx)
{
    const pulseg_sequence_descriptor *desc;
    int n_main, w, i;

    if (!coll)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[subseq_idx];
    n_main = desc->segment_table.num_main_segments;

    if (!out_ids)
        return n_main;

    w = 0;
    for (i = 0; i < n_main; ++i)
        out_ids[w++] = desc->segment_table.main_segment_table[i];

    return w;
}

/* ================================================================== */
/*  Segment timing queries                                            */
/* ================================================================== */

static int get_segment_rf_adc_gap_us(const pulseg_collection *coll, int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg;
    const pulseg_segment_timing *tm;
    int r, a;

    if (!resolve_segment(coll, &desc, &local_seg, seg_idx))
        return -1;

    tm = &desc->segment_definitions[local_seg].timing;

    /* For each RF anchor (in order), find the first ADC anchor whose
     * start is after the RF end.  Return the smallest such gap. */
    {
        int best = -1;
        for (r = 0; r < tm->num_rf_anchors; ++r)
        {
            int rf_end = tm->rf_anchors[r].end_us;
            for (a = 0; a < tm->num_adc_anchors; ++a)
            {
                int adc_start = tm->adc_anchors[a].start_us;
                if (adc_start >= rf_end)
                {
                    int gap = adc_start - rf_end;
                    if (best < 0 || gap < best)
                        best = gap;
                    break; /* first matching ADC for this RF */
                }
            }
        }
        return best;
    }
}

static int get_segment_adc_adc_gap_us(const pulseg_collection *coll, int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg;
    const pulseg_segment_timing *tm;
    int a, r, best;

    if (!resolve_segment(coll, &desc, &local_seg, seg_idx))
        return -1;

    tm = &desc->segment_definitions[local_seg].timing;
    if (tm->num_adc_anchors < 2)
        return -1;

    best = -1;
    for (a = 1; a < tm->num_adc_anchors; ++a)
    {
        int adc_prev_end = (int)tm->adc_anchors[a - 1].end_us;
        int adc_curr_start = (int)tm->adc_anchors[a].start_us;

        /* skip this ADC pair if any RF event starts between them */
        int interleaved = 0;
        for (r = 0; r < tm->num_rf_anchors; ++r)
        {
            int rf_start = (int)tm->rf_anchors[r].start_us;
            if (rf_start > adc_prev_end && rf_start < adc_curr_start)
            {
                interleaved = 1;
                break;
            }
        }
        if (interleaved)
            continue;

        {
            int gap = adc_curr_start - adc_prev_end;
            if (best < 0 || gap < best)
                best = gap;
        }
    }
    return best;
}

/* ================================================================== */
/*  Per-block RF isocenter from the segment timing anchors            */
/* ================================================================== */

float pulseg_get_rf_isocenter_us(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    const pulseg_base_block *bdef;
    const pulseg_rf_definition *rdef;
    int local_blk, k;
    float start_us = 0.0f;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1.0f;

    for (k = 0; k < local_blk; ++k)
        start_us += (float)desc->base_blocks[seg->unique_block_indices[k]].duration_us;

    bdef = &desc->base_blocks[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id < 0 || bdef->rf_id >= desc->num_unique_rfs)
        return -1.0f;

    rdef = &desc->rf_definitions[bdef->rf_id];
    return start_us + (float)rdef->delay + (float)rdef->stats.isodelay_us;
}

/* ================================================================== */
/*  Block-level queries (internal helpers for batch getter)            */
/* ================================================================== */

static int get_block_start_time_us(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk, k, start_time;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    start_time = 0;
    for (k = 0; k < local_blk; ++k)
        start_time += desc->base_blocks[seg->unique_block_indices[k]].duration_us;

    return start_time;
}

static int get_block_duration_us(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk;
    const pulseg_base_block *bdef;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    if (seg->num_blocks == 1 && local_blk == 0)
    {
        bdef = &desc->base_blocks[seg->unique_block_indices[0]];
        if (bdef->rf_id == -1 && bdef->gx_id == -1 && bdef->gy_id == -1 && bdef->gz_id == -1 &&
            bdef->adc_id == -1)
        {
            if (desc->block_raster_us > 0.0f)
                return (int)(desc->block_raster_us);
            return 0;
        }
    }

    return desc->base_blocks[seg->unique_block_indices[local_blk]].duration_us;
}

/* ================================================================== */
/*  RF queries (internal helpers + public waveform getters)            */
/* ================================================================== */

static int block_has_rf(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk;
    const pulseg_base_block *bdef;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    bdef = &desc->base_blocks[seg->unique_block_indices[local_blk]];
    return (bdef->rf_id != -1) ? 1 : 0;
}

static int block_rf_has_uniform_raster(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk;
    const pulseg_base_block *bdef;
    const pulseg_rf_definition *rdef;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    bdef = &desc->base_blocks[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1)
        return -1;

    rdef = &desc->rf_definitions[bdef->rf_id];
    return (rdef->time_shape_id == 0) ? 1 : 0;
}

static float *alloc_uniform_time_us(int num_samples, float raster_us)
{
    float *t;
    int i;

    if (num_samples <= 0 || raster_us <= 0.0f)
        return NULL;

    t = (float *)PULSEG_ALLOC((size_t)num_samples * sizeof(float));
    if (!t)
        return NULL;

    for (i = 0; i < num_samples; ++i)
        t[i] = ((float)i + 0.5f) * raster_us;

    return t;
}

static int block_rf_is_complex(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk;
    const pulseg_base_block *bdef;
    const pulseg_rf_definition *rdef;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    bdef = &desc->base_blocks[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1)
        return -1;

    rdef = &desc->rf_definitions[bdef->rf_id];
    return (rdef->phase_shape_id != 0) ? 1 : 0;
}

static int get_rf_num_samples(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk, shape_idx, total, nch;
    const pulseg_base_block *bdef;
    const pulseg_rf_definition *rdef;
    const pulseq_shape *shape;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    bdef = &desc->base_blocks[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1)
        return -1;

    rdef = &desc->rf_definitions[bdef->rf_id];
    total = -1;

    /* try mag, then phase, then time shape */
    if (rdef->mag_shape_id > 0)
    {
        shape_idx = rdef->mag_shape_id - 1;
        if (shape_idx >= 0 && shape_idx < desc->num_shapes)
        {
            shape = &desc->shapes[shape_idx];
            if (shape->num_uncompressed_samples > 0)
                total = shape->num_uncompressed_samples;
        }
    }
    if (total < 0 && rdef->phase_shape_id > 0)
    {
        shape_idx = rdef->phase_shape_id - 1;
        if (shape_idx >= 0 && shape_idx < desc->num_shapes)
        {
            shape = &desc->shapes[shape_idx];
            if (shape->num_uncompressed_samples > 0)
                total = shape->num_uncompressed_samples;
        }
    }
    if (total < 0 && rdef->time_shape_id > 0)
    {
        shape_idx = rdef->time_shape_id - 1;
        if (shape_idx >= 0 && shape_idx < desc->num_shapes)
        {
            shape = &desc->shapes[shape_idx];
            if (shape->num_uncompressed_samples > 0)
                total = shape->num_uncompressed_samples;
        }
    }
    if (total < 0)
        return -1;

    nch = (rdef->num_channels > 1) ? rdef->num_channels : 1;
    return total / nch;
}

static int get_rf_delay_us(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk;
    const pulseg_base_block *bdef;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    bdef = &desc->base_blocks[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1)
        return -1;

    return desc->rf_definitions[bdef->rf_id].delay;
}

static int get_rf_duration_us(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk;
    const pulseg_base_block *bdef;
    const pulseg_rf_definition *rdef;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    bdef = &desc->base_blocks[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1)
        return -1;

    rdef = &desc->rf_definitions[bdef->rf_id];
    /* stats.duration_us is the last time-shape sample time (accounts for
     * custom non-uniform time shapes). Round to nearest integer µs. */
    return (int)(rdef->stats.duration_us);
}

static int get_rf_num_channels(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk;
    const pulseg_base_block *bdef;
    const pulseg_rf_definition *rdef;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    bdef = &desc->base_blocks[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1)
        return -1;

    rdef = &desc->rf_definitions[bdef->rf_id];
    return (rdef->num_channels > 1) ? rdef->num_channels : 1;
}

#define PULSEG__RF_SHAPE_MAG 0
#define PULSEG__RF_SHAPE_PHASE 1

/* Shared decompress+split body for magnitude/phase RF waveform getters,
 * keyed off an already-resolved rf definition. Returns a malloc'd
 * (PULSEG_ALLOC) array of *num_channels row pointers, each *num_samples
 * floats, or NULL (shape absent / decompress failure / alloc failure). */
static float **get_rf_def_shape(
    const pulseg_sequence_descriptor *desc,
    int *num_channels,
    int *num_samples,
    const pulseg_rf_definition *rdef,
    int which_shape)
{
    int shape_id, shape_idx, nch, npts, ch;
    pulseq_shape decompressed;
    float *flat;
    float **result;

    *num_channels = 0;
    *num_samples = 0;

    shape_id = (which_shape == PULSEG__RF_SHAPE_MAG) ? rdef->mag_shape_id : rdef->phase_shape_id;
    if (shape_id <= 0)
        return NULL;

    shape_idx = shape_id - 1;
    if (shape_idx < 0 || shape_idx >= desc->num_shapes)
        return NULL;

    decompressed.num_samples = 0;
    decompressed.num_uncompressed_samples = 0;
    decompressed.samples = NULL;

    if (!pulseq_decompress_shape(&decompressed, &desc->shapes[shape_idx], 1.0f))
        return NULL;

    flat = decompressed.samples;
    nch = (rdef->num_channels > 1) ? rdef->num_channels : 1;
    npts = decompressed.num_samples / nch;

    /* allocate channel-pointer array */
    result = (float **)PULSEG_ALLOC((size_t)nch * sizeof(float *));
    if (!result)
    {
        PULSEG_FREE(flat);
        return NULL;
    }

    /* split tiled flat array into per-channel rows */
    for (ch = 0; ch < nch; ++ch)
    {
        result[ch] = (float *)PULSEG_ALLOC((size_t)npts * sizeof(float));
        if (!result[ch])
        {
            int k;
            for (k = 0; k < ch; ++k)
                PULSEG_FREE(result[k]);
            PULSEG_FREE(result);
            PULSEG_FREE(flat);
            return NULL;
        }
        memcpy(result[ch], flat + ch * npts, (size_t)npts * sizeof(float));
    }

    PULSEG_FREE(flat);
    *num_channels = nch;
    *num_samples = npts;
    return result;
}

float **pulseg_get_rf_magnitude(
    const pulseg_collection *coll,
    int *num_channels,
    int *num_samples,
    int seg_idx,
    int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk;
    const pulseg_base_block *bdef;
    const pulseg_rf_definition *rdef;

    if (!num_channels || !num_samples)
        return NULL;
    *num_channels = 0;
    *num_samples = 0;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return NULL;

    bdef = &desc->base_blocks[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1)
        return NULL;

    rdef = &desc->rf_definitions[bdef->rf_id];
    return get_rf_def_shape(desc, num_channels, num_samples, rdef, PULSEG__RF_SHAPE_MAG);
}

float **pulseg_get_rf_phase(
    const pulseg_collection *coll,
    int *num_channels,
    int *num_samples,
    int seg_idx,
    int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk;
    const pulseg_base_block *bdef;
    const pulseg_rf_definition *rdef;

    if (!num_channels || !num_samples)
        return NULL;
    *num_channels = 0;
    *num_samples = 0;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return NULL;

    bdef = &desc->base_blocks[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1)
        return NULL;

    rdef = &desc->rf_definitions[bdef->rf_id];
    return get_rf_def_shape(desc, num_channels, num_samples, rdef, PULSEG__RF_SHAPE_PHASE);
}

float *pulseg_get_rf_time_us(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk, shape_idx, nch, npts;
    const pulseg_base_block *bdef;
    const pulseg_rf_definition *rdef;
    pulseq_shape decompressed;
    float *result;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return NULL;

    bdef = &desc->base_blocks[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1)
        return NULL;

    rdef = &desc->rf_definitions[bdef->rf_id];
    if (rdef->time_shape_id <= 0)
    {
        npts = get_rf_num_samples(coll, seg_idx, blk_idx);
        return alloc_uniform_time_us(npts, desc->rf_raster_us);
    }

    shape_idx = rdef->time_shape_id - 1;
    if (shape_idx < 0 || shape_idx >= desc->num_shapes)
        return NULL;

    decompressed.num_samples = 0;
    decompressed.num_uncompressed_samples = 0;
    decompressed.samples = NULL;

    if (!pulseq_decompress_shape(&decompressed, &desc->shapes[shape_idx], desc->rf_raster_us))
        return NULL;

    nch = (rdef->num_channels > 1) ? rdef->num_channels : 1;
    npts = decompressed.num_samples / nch;

    if (nch > 1)
    {
        /* return only first channel's time (all channels share time base) */
        result = (float *)PULSEG_ALLOC((size_t)npts * sizeof(float));
        if (!result)
        {
            PULSEG_FREE(decompressed.samples);
            return NULL;
        }
        memcpy(result, decompressed.samples, (size_t)npts * sizeof(float));
        PULSEG_FREE(decompressed.samples);
    }
    else
    {
        result = decompressed.samples;
    }

    return result;
}

/* ================================================================== */
/*  Definition-keyed RF waveform getters (pTx SAR: keyed by subseq_idx +   */
/*  rf_def_id, not seg/blk -- one lookup per unique RF definition,        */
/*  independent of where it is played in the sequence).                   */
/* ================================================================== */

float **pulseg_get_rf_def_magnitude(
    const pulseg_collection *coll,
    int *num_channels,
    int *num_samples,
    int subseq_idx,
    int rf_def_id)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_rf_definition *rdef;

    if (!num_channels || !num_samples)
        return NULL;
    *num_channels = 0;
    *num_samples = 0;

    if (!coll)
        return NULL;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return NULL;

    desc = &coll->descriptors[subseq_idx];
    if (rf_def_id < 0 || rf_def_id >= desc->num_unique_rfs)
        return NULL;

    rdef = &desc->rf_definitions[rf_def_id];
    return get_rf_def_shape(desc, num_channels, num_samples, rdef, PULSEG__RF_SHAPE_MAG);
}

float **pulseg_get_rf_def_phase(
    const pulseg_collection *coll,
    int *num_channels,
    int *num_samples,
    int subseq_idx,
    int rf_def_id)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_rf_definition *rdef;

    if (!num_channels || !num_samples)
        return NULL;
    *num_channels = 0;
    *num_samples = 0;

    if (!coll)
        return NULL;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return NULL;

    desc = &coll->descriptors[subseq_idx];
    if (rf_def_id < 0 || rf_def_id >= desc->num_unique_rfs)
        return NULL;

    rdef = &desc->rf_definitions[rf_def_id];
    return get_rf_def_shape(desc, num_channels, num_samples, rdef, PULSEG__RF_SHAPE_PHASE);
}

float *pulseg_get_rf_def_time(
    const pulseg_collection *coll,
    int *num_samples,
    int subseq_idx,
    int rf_def_id)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_rf_definition *rdef;
    int shape_idx, nch, npts;
    pulseq_shape decompressed;
    float *result;

    if (!num_samples)
        return NULL;
    *num_samples = 0;

    if (!coll)
        return NULL;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return NULL;

    desc = &coll->descriptors[subseq_idx];
    if (rf_def_id < 0 || rf_def_id >= desc->num_unique_rfs)
        return NULL;

    rdef = &desc->rf_definitions[rf_def_id];
    if (rdef->time_shape_id <= 0)
        return NULL; /* no time shape -- caller falls back to uniform raster */

    shape_idx = rdef->time_shape_id - 1;
    if (shape_idx < 0 || shape_idx >= desc->num_shapes)
        return NULL;

    decompressed.num_samples = 0;
    decompressed.num_uncompressed_samples = 0;
    decompressed.samples = NULL;

    if (!pulseq_decompress_shape(&decompressed, &desc->shapes[shape_idx], desc->rf_raster_us))
        return NULL;

    nch = (rdef->num_channels > 1) ? rdef->num_channels : 1;
    npts = decompressed.num_samples / nch;

    if (nch > 1)
    {
        /* return only first channel's time (all channels share time base) */
        result = (float *)PULSEG_ALLOC((size_t)npts * sizeof(float));
        if (!result)
        {
            PULSEG_FREE(decompressed.samples);
            return NULL;
        }
        memcpy(result, decompressed.samples, (size_t)npts * sizeof(float));
        PULSEG_FREE(decompressed.samples);
    }
    else
    {
        result = decompressed.samples;
    }

    *num_samples = npts;
    return result;
}

float pulseg_get_rf_initial_amplitude_hz(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk;
    const pulseg_block_initial_state *st;
    const pulseg_base_block *bdef;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return 0.0f;

    bdef = &desc->base_blocks[seg->unique_block_indices[local_blk]];
    if (bdef->rf_id == -1)
        return 0.0f;

    st = resolve_initial_state(desc, seg, local_blk);
    if (!st)
        return desc->rf_definitions[bdef->rf_id].stats.base_amplitude_hz;

    return st->rf_amplitude_hz;
}

float pulseg_get_rf_max_amplitude_hz(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    return (float)fabs(pulseg_get_rf_initial_amplitude_hz(coll, seg_idx, blk_idx));
}

/* ================================================================== */
/*  Gradient queries (internal helpers + public waveform getters)      */
/* ================================================================== */

static int block_has_grad(const pulseg_collection *coll, int seg_idx, int blk_idx, int axis)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk, grad_id;

    if (axis < PULSEG_GRAD_AXIS_X || axis > PULSEG_GRAD_AXIS_Z)
        return -1;
    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    grad_id = resolve_grad_def_via_max_energy_instance(desc, seg, local_blk, axis);
    return (grad_id != -1) ? 1 : 0;
}

static int block_grad_is_trapezoid(
    const pulseg_collection *coll,
    int seg_idx,
    int blk_idx,
    int axis)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk, grad_id;
    const pulseg_grad_definition *gdef;

    if (axis < PULSEG_GRAD_AXIS_X || axis > PULSEG_GRAD_AXIS_Z)
        return -1;
    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    grad_id = resolve_grad_def_via_max_energy_instance(desc, seg, local_blk, axis);
    if (grad_id == -1)
        return -1;

    gdef = &desc->grad_definitions[grad_id];
    if (gdef->type == 0)
        return 1;
    if (gdef->unused_or_time_shape_id > 0)
        return 1;
    return 0;
}

static int get_grad_num_samples(const pulseg_collection *coll, int seg_idx, int blk_idx, int axis)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk, grad_id, shape_idx;
    const pulseg_grad_definition *gdef;
    const pulseq_shape *shape;

    if (axis < PULSEG_GRAD_AXIS_X || axis > PULSEG_GRAD_AXIS_Z)
        return -1;
    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    grad_id = resolve_grad_def_via_max_energy_instance(desc, seg, local_blk, axis);
    if (grad_id == -1)
        return -1;

    gdef = &desc->grad_definitions[grad_id];

    if (gdef->type == 0)
    {
        return (gdef->flat_time_or_unused > 0) ? 4 : 3;
    }

    /* arbitrary: every shape a definition plays has the same sample count,
     * that being part of what makes them one definition, so any of them
     * answers this. */
    if (gdef->spectral.shape_id > 0)
    {
        shape_idx = gdef->spectral.shape_id - 1;
        if (shape_idx >= 0 && shape_idx < desc->num_shapes)
        {
            shape = &desc->shapes[shape_idx];
            if (shape->num_uncompressed_samples > 0)
                return shape->num_uncompressed_samples;
        }
    }
    return -1;
}

static int get_grad_num_shots(const pulseg_collection *coll, int seg_idx, int blk_idx, int axis)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk, grad_id;

    if (axis < PULSEG_GRAD_AXIS_X || axis > PULSEG_GRAD_AXIS_Z)
        return -1;
    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    grad_id = resolve_grad_def_via_max_energy_instance(desc, seg, local_blk, axis);
    if (grad_id == -1)
        return -1;

    (void)grad_id;
    return 1; /* one waveform per definition; instances swap it */
}

static int get_grad_delay_us(const pulseg_collection *coll, int seg_idx, int blk_idx, int axis)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk, grad_id;

    if (axis < PULSEG_GRAD_AXIS_X || axis > PULSEG_GRAD_AXIS_Z)
        return -1;
    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    grad_id = resolve_grad_def_via_max_energy_instance(desc, seg, local_blk, axis);
    if (grad_id == -1)
        return -1;

    return desc->grad_definitions[grad_id].delay;
}

float **pulseg_get_grad_amplitude(
    const pulseg_collection *coll,
    int *num_shots,
    int *num_samples,
    int seg_idx,
    int blk_idx,
    int axis)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk, grad_id, shot, k, shape_idx;
    const pulseg_grad_definition *gdef;
    float **waveforms;
    float *trap_waveform;
    int samples_per_shot;
    int flat_time;
    pulseq_shape decompressed;

    if (!num_shots || !num_samples)
    {
        if (num_shots)
            *num_shots = 0;
        if (num_samples)
            *num_samples = 0;
        return NULL;
    }
    *num_shots = 0;
    *num_samples = 0;

    if (axis < PULSEG_GRAD_AXIS_X || axis > PULSEG_GRAD_AXIS_Z)
        return NULL;
    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return NULL;

    /* Resolve through the frozen representative (max-energy) instance record
     * so that the grad_definition (and thus the set of shot waveforms)
     * matches the actual physical block at that instance.  This is necessary
     * when different segment instances have gradients with different
     * time_shape_ids, which places them in separate grad_definitions despite
     * occupying the same segment position. */
    grad_id = resolve_grad_def_via_max_energy_instance(desc, seg, local_blk, axis);
    if (grad_id < 0)
        return NULL;
    gdef = &desc->grad_definitions[grad_id];

    waveforms = (float **)PULSEG_ALLOC(sizeof(float *));
    if (!waveforms)
        return NULL;

    *num_shots = 1;

    if (gdef->type == 0)
    {
        flat_time = gdef->flat_time_or_unused;
        samples_per_shot = (flat_time > 0) ? 4 : 3;
        *num_samples = samples_per_shot;

        for (shot = 0; shot < 1; ++shot)
        {
            trap_waveform = (float *)PULSEG_ALLOC(samples_per_shot * sizeof(float));
            if (!trap_waveform)
            {
                for (k = 0; k < shot; ++k)
                    PULSEG_FREE(waveforms[k]);
                PULSEG_FREE(waveforms);
                *num_shots = 0;
                *num_samples = 0;
                return NULL;
            }

            trap_waveform[0] = 0.0f;
            trap_waveform[1] = 1.0f;
            if (flat_time > 0)
            {
                trap_waveform[2] = 1.0f;
                trap_waveform[3] = 0.0f;
            }
            else
            {
                trap_waveform[2] = 0.0f;
            }

            waveforms[shot] = trap_waveform;
        }
    }
    else
    {
        for (shot = 0; shot < 1; ++shot)
        {
            /* The shape the representative (max-energy) instance plays -- the
             * same one the initial state names, and the one pulsegen binds.
             * NOT a safety representative: those answer "what is the worst
             * case", and this answers "what does this block actually play". */
            int init_shape = pulseg_get_grad_initial_shape_id(coll, seg_idx, blk_idx, axis);
            if (init_shape <= 0)
            {
                waveforms[shot] = NULL;
                continue;
            }

            shape_idx = init_shape - 1;
            if (shape_idx < 0 || shape_idx >= desc->num_shapes)
            {
                waveforms[shot] = NULL;
                continue;
            }

            decompressed.num_samples = 0;
            decompressed.num_uncompressed_samples = 0;
            decompressed.samples = NULL;

            if (!pulseq_decompress_shape(&decompressed, &desc->shapes[shape_idx], 1.0f))
            {
                waveforms[shot] = NULL;
                continue;
            }

            waveforms[shot] = decompressed.samples;
            if (*num_samples == 0)
                *num_samples = decompressed.num_samples;
        }
    }

    return waveforms;
}

float pulseg_get_grad_initial_amplitude_hz_per_m(
    const pulseg_collection *coll,
    int seg_idx,
    int blk_idx,
    int axis)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk;
    const pulseg_block_initial_state *st;

    if (axis < PULSEG_GRAD_AXIS_X || axis > PULSEG_GRAD_AXIS_Z)
        return 1.0f;
    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return 1.0f;

    st = resolve_initial_state(desc, seg, local_blk);
    if (!st)
        return 1.0f;

    return st->grad_amplitude_hz_per_m[axis];
}

int pulseg_get_grad_initial_shape_id(
    const pulseg_collection *coll,
    int seg_idx,
    int blk_idx,
    int axis)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk;
    const pulseg_block_initial_state *st;

    if (axis < PULSEG_GRAD_AXIS_X || axis > PULSEG_GRAD_AXIS_Z)
        return 0;
    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return 0;

    st = resolve_initial_state(desc, seg, local_blk);
    if (!st)
        return 0;

    return st->grad_shape_id[axis];
}

float pulseg_get_grad_max_amplitude_hz_per_m(
    const pulseg_collection *coll,
    int seg_idx,
    int blk_idx,
    int axis)
{
    float init_amp;

    if (axis < PULSEG_GRAD_AXIS_X || axis > PULSEG_GRAD_AXIS_Z)
        return 0.0f;
    init_amp = pulseg_get_grad_initial_amplitude_hz_per_m(coll, seg_idx, blk_idx, axis);
    return (init_amp >= 0.0f) ? init_amp : -init_amp;
}

float *pulseg_get_grad_time_us(const pulseg_collection *coll, int seg_idx, int blk_idx, int axis)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk, grad_id, shape_idx;
    const pulseg_grad_definition *gdef;
    float *time_waveform;
    float accum;
    int rise_time, flat_time, fall_time, ns;
    pulseq_shape decompressed;

    if (axis < PULSEG_GRAD_AXIS_X || axis > PULSEG_GRAD_AXIS_Z)
        return NULL;
    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return NULL;

    /* Resolve through the frozen representative-instance record (see comment
     * in pulseg_get_grad_amplitude for rationale). */
    grad_id = resolve_grad_def_via_max_energy_instance(desc, seg, local_blk, axis);
    if (grad_id < 0)
        return NULL;
    gdef = &desc->grad_definitions[grad_id];

    if (gdef->type == 0)
    {
        rise_time = gdef->rise_time_or_unused;
        flat_time = gdef->flat_time_or_unused;
        fall_time = gdef->fall_time_or_num_uncompressed_samples;

        ns = (flat_time > 0) ? 4 : 3;

        time_waveform = (float *)PULSEG_ALLOC((size_t)ns * sizeof(float));
        if (!time_waveform)
            return NULL;

        accum = 0.0f;
        time_waveform[0] = accum;
        accum += (float)rise_time;
        time_waveform[1] = accum;
        if (flat_time > 0)
        {
            accum += (float)flat_time;
            time_waveform[2] = accum;
            accum += (float)fall_time;
            time_waveform[3] = accum;
        }
        else
        {
            accum += (float)fall_time;
            time_waveform[2] = accum;
        }

        return time_waveform;
    }

    /* arbitrary: decompress time shape */
    if (gdef->unused_or_time_shape_id <= 0)
    {
        ns = get_grad_num_samples(coll, seg_idx, blk_idx, axis);
        return alloc_uniform_time_us(ns, desc->grad_raster_us);
    }

    shape_idx = gdef->unused_or_time_shape_id - 1;
    if (shape_idx < 0 || shape_idx >= desc->num_shapes)
        return NULL;

    decompressed.num_samples = 0;
    decompressed.num_uncompressed_samples = 0;
    decompressed.samples = NULL;

    if (!pulseq_decompress_shape(&decompressed, &desc->shapes[shape_idx], desc->grad_raster_us))
        return NULL;

    return decompressed.samples;
}

/* ================================================================== */
/*  ADC block queries (internal helpers)                               */
/* ================================================================== */

static int block_has_adc(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    /* Use OR-reduced flag: true if at least one segment instance has an ADC
     * event at this block position (not just the canonical block def). */
    if (seg->has_adc && seg->has_adc[local_blk])
        return 1;

    /* Fallback to canonical block definition (pre-11d sequences / cache) */
    {
        const pulseg_base_block *bdef = &desc->base_blocks[seg->unique_block_indices[local_blk]];
        return (bdef->adc_id != -1) ? 1 : 0;
    }
}

/* True iff this block position's duration actually differs across at least
 * two scan-table instances of the segment (see pulseg_structure.c step 11d).
 * A NULL is_dynamic_delay array (pre-11d sequences / older cache) falls back
 * to the pre-existing conservative behavior of always treating an adjustable
 * pure delay as dynamic. */
static int block_is_dynamic_delay(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return 0;

    if (seg->is_dynamic_delay)
        return seg->is_dynamic_delay[local_blk] ? 1 : 0;
    return 1;
}

static int get_adc_delay_us(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk, adc_id;
    const pulseg_base_block *bdef;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    bdef = &desc->base_blocks[seg->unique_block_indices[local_blk]];
    adc_id = bdef->adc_id;
    if (adc_id < 0 || adc_id >= desc->num_unique_adcs)
        return -1;

    return desc->adc_definitions[adc_id].delay;
}

static int get_adc_library_index(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk, adc_id, global_adc_idx, i;
    const pulseg_base_block *bdef;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    bdef = &desc->base_blocks[seg->unique_block_indices[local_blk]];
    adc_id = bdef->adc_id;
    if (adc_id < 0 || adc_id >= desc->num_unique_adcs)
        return -1;

    /* compute global index: sum ADC counts from prior subsequences */
    global_adc_idx = 0;
    for (i = 0; i < coll->num_subsequences; ++i)
    {
        if (&coll->descriptors[i] == desc)
            break;
        global_adc_idx += coll->descriptors[i].num_unique_adcs;
    }
    return global_adc_idx + adc_id;
}

/* ================================================================== */
/*  Digital output / trigger / flag queries (internal helpers)         */
/* ================================================================== */

static int block_has_digitalout(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    return seg->has_digitalout[local_blk];
}

static int get_digitalout_delay_us(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk, digitalout_id;
    const pulseg_block_initial_state *st;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    if (!seg->has_digitalout[local_blk])
        return -1;

    st = resolve_initial_state(desc, seg, local_blk);
    if (!st)
        return -1;
    digitalout_id = st->digitalout_id;
    if (digitalout_id == -1 || digitalout_id >= desc->num_triggers)
        return -1;

    return (int)desc->trigger_events[digitalout_id].delay;
}

static int get_digitalout_duration_us(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk, digitalout_id;
    const pulseg_block_initial_state *st;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    if (!seg->has_digitalout[local_blk])
        return -1;

    st = resolve_initial_state(desc, seg, local_blk);
    if (!st)
        return -1;
    digitalout_id = st->digitalout_id;
    if (digitalout_id == -1 || digitalout_id >= desc->num_triggers)
        return -1;

    return (int)desc->trigger_events[digitalout_id].duration;
}

static int get_digitalout_channel(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk, digitalout_id;
    const pulseg_block_initial_state *st;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    if (!seg->has_digitalout[local_blk])
        return -1;

    st = resolve_initial_state(desc, seg, local_blk);
    if (!st)
        return -1;
    digitalout_id = st->digitalout_id;
    if (digitalout_id == -1 || digitalout_id >= desc->num_triggers)
        return -1;

    return desc->trigger_events[digitalout_id].trigger_channel;
}

/* ---- Segment-level physio trigger queries ------------------------ */

static int segment_has_trigger(const pulseg_collection *coll, int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg;

    if (!resolve_segment(coll, &desc, &local_seg, seg_idx))
        return 0;

    return (desc->segment_definitions[local_seg].trigger_id >= 0) ? 1 : 0;
}

static int get_segment_trigger_delay_us(const pulseg_collection *coll, int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg, tid;

    if (!resolve_segment(coll, &desc, &local_seg, seg_idx))
        return -1;

    tid = desc->segment_definitions[local_seg].trigger_id;
    if (tid < 0 || tid >= desc->num_triggers)
        return -1;

    return (int)desc->trigger_events[tid].delay;
}

static int get_segment_trigger_duration_us(const pulseg_collection *coll, int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg, tid;

    if (!resolve_segment(coll, &desc, &local_seg, seg_idx))
        return -1;

    tid = desc->segment_definitions[local_seg].trigger_id;
    if (tid < 0 || tid >= desc->num_triggers)
        return -1;

    return (int)desc->trigger_events[tid].duration;
}

static int get_segment_trigger_type(const pulseg_collection *coll, int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg, tid;

    if (!resolve_segment(coll, &desc, &local_seg, seg_idx))
        return 0;

    tid = desc->segment_definitions[local_seg].trigger_id;
    if (tid < 0 || tid >= desc->num_triggers)
        return 0;

    return desc->trigger_events[tid].trigger_type;
}

/* ---- Navigator flag query ---------------------------------------- */

static int segment_is_nav(const pulseg_collection *coll, int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    int local_seg;

    if (!resolve_segment(coll, &desc, &local_seg, seg_idx))
        return 0;

    return desc->segment_definitions[local_seg].is_nav;
}

static int block_has_rotation(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    return seg->has_rotation[local_blk];
}

static int block_has_norot(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    return seg->norot_flag[local_blk];
}

static int block_has_nopos(const pulseg_collection *coll, int seg_idx, int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return -1;

    return seg->nopos_flag[local_blk];
}

/* The frozen per-position record (pulseg_block_initial_state), or NULL. */
static const pulseg_block_initial_state *block_initial_state(
    const pulseg_collection *coll,
    int seg_idx,
    int blk_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_blk;

    if (!resolve_block(coll, &desc, &seg, &local_blk, seg_idx, blk_idx))
        return NULL;
    if (!seg->initial_states)
        return NULL;

    return &seg->initial_states[local_blk];
}

int pulseg_cursor_next(pulseg_collection *coll)
{
    pulseg_block_cursor *cursor;
    const pulseg_sequence_descriptor *desc;
    int next_pos;

    cursor = &coll->block_cursor;

    if (cursor->sequence_index >= coll->num_subsequences)
        return PULSEG_CURSOR_DONE;

    desc = &coll->descriptors[cursor->sequence_index];
    next_pos = cursor->exec_stream_position + 1;

    /* Past end of scan table: advance to next subsequence */
    if (next_pos >= desc->exec_stream_len)
    {
        cursor->sequence_index += 1;
        cursor->exec_stream_position = 0;
        cursor->from_last_reset = 0;
        if (cursor->sequence_index >= coll->num_subsequences)
            return PULSEG_CURSOR_DONE;
        return PULSEG_CURSOR_BLOCK;
    }

    /* Normal advance */
    cursor->exec_stream_position = next_pos;
    cursor->from_last_reset += 1;
    return PULSEG_CURSOR_BLOCK;
}

int pulseg_cursor_advance(pulseg_collection *coll, pulseg_cursor_info *info)
{
    int status = pulseg_cursor_next(coll);
    if (status != PULSEG_CURSOR_BLOCK)
        return status; /* PULSEG_CURSOR_DONE */

    status = pulseg_cursor_get_info(coll, info);
    if (PULSEG_FAILED(status))
        return status; /* propagate info-retrieval error */

    return PULSEG_CURSOR_BLOCK;
}

void pulseg_cursor_rewind(pulseg_collection *coll)
{
    pulseg_block_cursor *cursor;

    cursor = &coll->block_cursor;

    /* Go back by the number of blocks advanced since the last mark */
    cursor->exec_stream_position -= cursor->from_last_reset;
    cursor->from_last_reset = 0;
}

void pulseg_cursor_reset(pulseg_collection *coll)
{
    if (!coll)
        return;

    /* Full rewind to the absolute start of the collection (pristine load
     * state {0, -1, 0}).  Unlike pulseg_cursor_rewind(), this also clears
     * sequence_index, so a collection whose cursor has already run to its
     * terminal PULSEG_CURSOR_DONE state (sequence_index == num_subsequences)
     * can be replayed from the top.  Required when one loaded collection is
     * traversed by more than one RSP entry point (e.g. aps2 then scan, which
     * reuse s_sc_coll because the SIM framework does not call psdcleanup
     * between entry points). */
    coll->block_cursor.sequence_index = 0;
    coll->block_cursor.exec_stream_position = -1; /* -1 = before first block */
    coll->block_cursor.from_last_reset = 0;
}

void pulseg_cursor_mark(pulseg_collection *coll)
{
    if (!coll)
        return;
    coll->block_cursor.from_last_reset = 0;
}

int pulseg_cursor_get_info(const pulseg_collection *coll, pulseg_cursor_info *info)
{
    const pulseg_block_cursor *cursor;
    const pulseg_sequence_descriptor *desc;
    int pos, seg_id;

    if (!coll || !info)
        return PULSEG_ERR_NULL_POINTER;

    cursor = &coll->block_cursor;
    if (cursor->sequence_index < 0 || cursor->sequence_index >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[cursor->sequence_index];
    pos = cursor->exec_stream_position;
    if (pos < 0 || pos >= desc->exec_stream_len)
        return PULSEG_ERR_INVALID_ARGUMENT;

    seg_id = pulseg__exec_seg_id(desc, pos);

    info->subseq_idx = cursor->sequence_index;
    info->scan_pos = pos;
    /* Map the subsequence-local segment id to its deduplicated global id.
     * segment_id_offset is the flat index base into seg_local_to_global. */
    {
        int flat = coll->subsequence_info[cursor->sequence_index].segment_id_offset + seg_id;
        if (coll->seg_local_to_global && seg_id >= 0 && flat >= 0 && flat < coll->seg_l2g_len)
            info->segment_id = coll->seg_local_to_global[flat];
        else
            info->segment_id =
                seg_id + coll->subsequence_info[cursor->sequence_index].segment_id_offset;
    }
    /* segment_start fires at the first block of every segment instance,
     * including consecutive instances of the same segment (e.g. the ny
     * phase-encoding readout lines in MPRAGE within one TR).
     * A new instance begins when the seg_id changes, or when enough
     * blocks have elapsed to complete one full instance of the segment
     * (detected by counting back to the start of the same-seg_id run). */
    {
        int new_inst;
        if (pos == 0 || pulseg__exec_seg_id(desc, pos) != pulseg__exec_seg_id(desc, pos - 1))
        {
            new_inst = 1;
        }
        else if (seg_id >= 0 && seg_id < desc->num_unique_segments)
        {
            int nb = desc->segment_definitions[seg_id].num_blocks;
            int run_start = pos;
            if (nb <= 0)
                nb = 1;
            while (run_start > 0 && pulseg__exec_seg_id(desc, run_start - 1) == seg_id)
                run_start--;
            new_inst = (((pos - run_start) % nb) == 0) ? 1 : 0;
        }
        else
        {
            new_inst = 0;
        }
        info->segment_start = new_inst;
    }
    {
        int last_inst;
        if (pos == desc->exec_stream_len - 1 ||
            pulseg__exec_seg_id(desc, pos) != pulseg__exec_seg_id(desc, pos + 1))
        {
            last_inst = 1;
        }
        else if (seg_id >= 0 && seg_id < desc->num_unique_segments)
        {
            int nb = desc->segment_definitions[seg_id].num_blocks;
            int run_start = pos;
            if (nb <= 0)
                nb = 1;
            while (run_start > 0 && pulseg__exec_seg_id(desc, run_start - 1) == seg_id)
                run_start--;
            last_inst = ((((pos - run_start) + 1) % nb) == 0) ? 1 : 0;
        }
        else
        {
            last_inst = 0;
        }
        info->segment_end = last_inst;
    }
    info->tr_start = pulseg__exec_tr_start(desc, pos);
    info->pmc = desc->enable_pmc;

    /* Segment properties via local segment index */
    {
        int local_seg = seg_id; /* seg_id in scan table is local (before offset) */
        if (local_seg >= 0 && local_seg < desc->num_unique_segments)
        {
            info->is_nav = desc->segment_definitions[local_seg].is_nav;
            info->has_trigger = (desc->segment_definitions[local_seg].trigger_id >= 0) ? 1 : 0;
        }
        else
        {
            info->is_nav = 0;
            info->has_trigger = 0;
        }
    }

    return PULSEG_SUCCESS;
}

/* Shared resolver behind pulseg_get_block_instance (cursor) and
 * pulseg_get_block_instance_at (random access).  Fills the per-instance
 * resolved view (PulSeg SegmentInstance, spec 3.3) for one execution-stream
 * position: it reads the existing tables, it does not store anything. */
static int resolve_block_instance(
    const pulseg_sequence_descriptor *desc,
    pulseg_block_instance *inst,
    int exec_stream_position)
{
    const pulseg_block_table_element *bte;
    const pulseg_base_block *bdef;
    int idx, i;

    if (exec_stream_position < 0 || exec_stream_position >= desc->exec_stream_len)
        return PULSEG_ERR_INVALID_ARGUMENT;

    idx = pulseg__exec_block_idx(desc, exec_stream_position);
    if (idx < 0 || idx >= desc->num_blocks)
        return PULSEG_ERR_INVALID_ARGUMENT;

    bte = &desc->block_table[idx];
    bdef = &desc->base_blocks[bte->id];

    /* Duration: pure delay uses instance value, normal block uses definition */
    inst->duration_us = (bte->duration_us >= 0) ? bte->duration_us : bdef->duration_us;

    /* RF */
    if (bte->rf_id >= 0 && bte->rf_id < desc->rf_table_size)
    {
        inst->rf_amp_hz = desc->rf_table[bte->rf_id].amplitude;
        inst->rf_freq_hz = desc->rf_table[bte->rf_id].freq_offset;
        inst->rf_phase_rad = desc->rf_table[bte->rf_id].phase_offset;
    }
    else
    {
        inst->rf_amp_hz = 0.0f;
        inst->rf_freq_hz = 0.0f;
        inst->rf_phase_rad = 0.0f;
    }

    /* Gradients */
    if (bte->gx_id >= 0 && bte->gx_id < desc->grad_table_size)
    {
        inst->gx_amp_hz_per_m = desc->grad_table[bte->gx_id].amplitude;
        inst->gx_shape_id = desc->grad_table[bte->gx_id].shape_id;
    }
    else
    {
        inst->gx_amp_hz_per_m = 0.0f;
        inst->gx_shape_id = 0;
    }
    if (bte->gy_id >= 0 && bte->gy_id < desc->grad_table_size)
    {
        inst->gy_amp_hz_per_m = desc->grad_table[bte->gy_id].amplitude;
        inst->gy_shape_id = desc->grad_table[bte->gy_id].shape_id;
    }
    else
    {
        inst->gy_amp_hz_per_m = 0.0f;
        inst->gy_shape_id = 0;
    }
    if (bte->gz_id >= 0 && bte->gz_id < desc->grad_table_size)
    {
        inst->gz_amp_hz_per_m = desc->grad_table[bte->gz_id].amplitude;
        inst->gz_shape_id = desc->grad_table[bte->gz_id].shape_id;
    }
    else
    {
        inst->gz_amp_hz_per_m = 0.0f;
        inst->gz_shape_id = 0;
    }

    /* Rotation */
    if (bte->rotation_id >= 0 && bte->rotation_id < desc->num_rotations)
    {
        for (i = 0; i < 9; ++i)
            inst->rotmat[i] = desc->rotation_matrices[bte->rotation_id][i];
    }
    else
    {
        inst->rotmat[0] = 1.0f;
        inst->rotmat[1] = 0.0f;
        inst->rotmat[2] = 0.0f;
        inst->rotmat[3] = 0.0f;
        inst->rotmat[4] = 1.0f;
        inst->rotmat[5] = 0.0f;
        inst->rotmat[6] = 0.0f;
        inst->rotmat[7] = 0.0f;
        inst->rotmat[8] = 1.0f;
    }

    /* Flags */
    inst->norot_flag = bte->norot_flag;
    inst->nopos_flag = bte->nopos_flag;

    /* Digital output */
    inst->digitalout_flag = (bte->digitalout_id >= 0) ? 1 : 0;
    if (inst->digitalout_flag && bte->digitalout_id < desc->num_triggers)
    {
        inst->digitalout_channel = desc->trigger_events[bte->digitalout_id].trigger_channel;
    }
    else
    {
        inst->digitalout_channel = -1;
    }

    /* ADC */
    if (bte->adc_id >= 0 && bte->adc_id < desc->adc_table_size)
    {
        inst->adc_flag = 1;
        inst->adc_freq_hz = desc->adc_table[bte->adc_id].freq_offset;
        inst->adc_phase_rad = desc->adc_table[bte->adc_id].phase_offset;
    }
    else
    {
        inst->adc_flag = 0;
        inst->adc_freq_hz = 0.0f;
        inst->adc_phase_rad = 0.0f;
    }

    /* RF shimming */
    inst->rf_shim_id = bte->rf_shim_id;

    /* Safety group (TRID) */
    inst->trid = bte->trid;

    /* Variable-amplitude flags (TR-level) */
    {
        int vg_tr_size = desc->tr_descriptor.tr_size;
        int vg_pos;
        if (vg_tr_size > 0 && desc->variable_grad_flags)
        {
            vg_pos = exec_stream_position % vg_tr_size;
            inst->gx_variable = desc->variable_grad_flags[vg_pos * 3 + 0];
            inst->gy_variable = desc->variable_grad_flags[vg_pos * 3 + 1];
            inst->gz_variable = desc->variable_grad_flags[vg_pos * 3 + 2];
        }
        else
        {
            inst->gx_variable = 0;
            inst->gy_variable = 0;
            inst->gz_variable = 0;
        }
    }

    return PULSEG_SUCCESS;
}

int pulseg_get_block_instance(const pulseg_collection *coll, pulseg_block_instance *inst)
{
    const pulseg_block_cursor *cursor;

    if (!coll || !inst)
        return PULSEG_ERR_NULL_POINTER;

    cursor = &coll->block_cursor;
    if (cursor->sequence_index >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;

    return resolve_block_instance(
        &coll->descriptors[cursor->sequence_index],
        inst,
        cursor->exec_stream_position);
}

int pulseg_get_block_instance_at(
    const pulseg_collection *coll,
    pulseg_block_instance *inst,
    int subseq_idx,
    int exec_stream_position)
{
    if (!coll || !inst)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;

    return resolve_block_instance(&coll->descriptors[subseq_idx], inst, exec_stream_position);
}

/* ================================================================== */
/*  Label getters                                                     */
/* ================================================================== */

int pulseg_get_label_limits(
    const pulseg_collection *coll,
    pulseg_label_limits *limits,
    int subseq_idx)
{
    if (!coll || !limits)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;
    *limits = coll->descriptors[subseq_idx].label_limits;
    return PULSEG_SUCCESS;
}

static int get_num_adc_occurrences(const pulseg_collection *coll, int subseq_idx)
{
    if (!coll)
        return 0;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].label_num_entries;
}

static int get_num_label_columns(const pulseg_collection *coll, int subseq_idx)
{
    if (!coll)
        return 0;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return 0;
    return coll->descriptors[subseq_idx].label_num_columns;
}

int pulseg_get_adc_label(
    const pulseg_collection *coll,
    int *out_values,
    int subseq_idx,
    int occurrence_idx)
{
    const pulseg_sequence_descriptor *desc;
    int ncols, row_start, c;

    if (!coll || !out_values)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[subseq_idx];
    if (!desc->label_table || desc->label_num_entries == 0)
        return PULSEG_ERR_INVALID_ARGUMENT;
    if (occurrence_idx < 0 || occurrence_idx >= desc->label_num_entries)
        return PULSEG_ERR_INVALID_ARGUMENT;

    ncols = desc->label_num_columns;
    row_start = occurrence_idx * ncols;
    for (c = 0; c < ncols; ++c)
        out_values[c] = desc->label_table[row_start + c];

    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Batch getters (public API)                                        */
/* ================================================================== */

int pulseg_get_collection_info(const pulseg_collection *coll, pulseg_collection_info *info)
{
    if (!coll || !info)
        return PULSEG_ERR_NULL_POINTER;

    info->num_subsequences = get_num_subsequences(coll);
    info->num_segments = get_num_segments(coll);
    info->max_adc_samples = get_max_adc_samples(coll);
    info->total_readouts = get_total_readouts(coll);
    info->total_duration_us = get_total_duration_us(coll);

    return PULSEG_SUCCESS;
}

int pulseg_get_subseq_info(const pulseg_collection *coll, pulseg_subseq_info *info, int subseq_idx)
{
    if (!coll || !info)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;

    info->tr_duration_us = get_tr_duration_us(coll, subseq_idx);
    info->num_trs = get_num_trs(coll, subseq_idx);
    info->tr_size = get_tr_size(coll, subseq_idx);
    info->num_unique_adcs = get_num_unique_adcs(coll, subseq_idx);
    info->num_unique_rf = get_num_unique_rf(coll, subseq_idx);
    info->pmc_enabled = is_pmc_enabled(coll, subseq_idx);
    info->segment_offset = get_subseq_segment_offset(coll, subseq_idx);
    info->num_adc_occurrences = get_num_adc_occurrences(coll, subseq_idx);
    info->num_label_columns = get_num_label_columns(coll, subseq_idx);
    info->num_gain_cal_readouts = coll->descriptors[subseq_idx].num_gain_cal_readouts;

    /* The TR-instance contract of pulseg_get_tr_waveforms: every played TR
     * is a unit. */
    {
        const pulseg_tr_descriptor *trd = &coll->descriptors[subseq_idx].tr_descriptor;
        info->num_tr_instances = trd->num_trs;
        if (info->num_tr_instances < 1)
            info->num_tr_instances = 1;
    }

    /* One canonical window per shape group: repetitions that play the same
     * set of gradient waveforms share a worst-case envelope, and ones that
     * play different waveforms cannot be covered by a single window.  This
     * is the count pulseg_get_tr_gradient_waveforms indexes. */
    {
        int *labels = NULL;
        int *group_first = NULL;
        int num_groups = 0;
        int rc;

        rc = pulseg__group_tr_instances_by_shape(
            &coll->descriptors[subseq_idx],
            &labels,
            &group_first,
            &num_groups,
            PULSEG__MAX_SHAPE_GROUPS);
        info->num_canonical_trs = (PULSEG_SUCCEEDED(rc) && num_groups > 0) ? num_groups : 1;
        if (labels)
            PULSEG_FREE(labels);
        if (group_first)
            PULSEG_FREE(group_first);
    }

    return PULSEG_SUCCESS;
}

int pulseg_get_segment_info(const pulseg_collection *coll, pulseg_segment_info *info, int seg_idx)
{
    if (!coll || !info)
        return PULSEG_ERR_NULL_POINTER;

    info->duration_us = get_segment_duration_us(coll, seg_idx);
    info->num_blocks = get_segment_num_blocks(coll, seg_idx);
    info->start_block = get_segment_start_block(coll, seg_idx);
    info->pure_delay = is_segment_pure_delay(coll, seg_idx);
    info->has_trigger = segment_has_trigger(coll, seg_idx);
    info->trigger_type = get_segment_trigger_type(coll, seg_idx);
    info->trigger_delay_us = get_segment_trigger_delay_us(coll, seg_idx);
    info->trigger_duration_us = get_segment_trigger_duration_us(coll, seg_idx);
    info->is_nav = segment_is_nav(coll, seg_idx);
    info->rf_adc_gap_us = get_segment_rf_adc_gap_us(coll, seg_idx);
    info->adc_adc_gap_us = get_segment_adc_adc_gap_us(coll, seg_idx);

    return PULSEG_SUCCESS;
}

int pulseg_segment_has_grad(const pulseg_collection *coll, int seg_idx)
{
    pulseg_segment_info si = PULSEG_SEGMENT_INFO_INIT;
    pulseg_block_info bi = PULSEG_BLOCK_INFO_INIT;
    int blk;

    if (!coll)
        return 0;
    if (!PULSEG_SUCCEEDED(pulseg_get_segment_info(coll, &si, seg_idx)))
        return 0;

    for (blk = 0; blk < si.num_blocks; ++blk)
    {
        if (!PULSEG_SUCCEEDED(pulseg_get_block_info(coll, &bi, seg_idx, blk)))
            continue;

        if (bi.has_grad[0] || bi.has_grad[1] || bi.has_grad[2])
            return 1;
    }

    return 0;
}

int pulseg_get_block_info(
    const pulseg_collection *coll,
    pulseg_block_info *info,
    int seg_idx,
    int blk_idx)
{
    int axis;

    if (!coll || !info)
        return PULSEG_ERR_NULL_POINTER;

    info->duration_us = get_block_duration_us(coll, seg_idx, blk_idx);
    info->start_time_us = get_block_start_time_us(coll, seg_idx, blk_idx);

    /* Gradient (per axis) */
    for (axis = 0; axis < 3; ++axis)
    {
        info->has_grad[axis] = block_has_grad(coll, seg_idx, blk_idx, axis);
        info->grad_is_trapezoid[axis] =
            info->has_grad[axis] ? block_grad_is_trapezoid(coll, seg_idx, blk_idx, axis) : 0;
        info->grad_delay_us[axis] =
            info->has_grad[axis] ? get_grad_delay_us(coll, seg_idx, blk_idx, axis) : -1;
        info->grad_num_shots[axis] =
            info->has_grad[axis] ? get_grad_num_shots(coll, seg_idx, blk_idx, axis) : -1;
        info->grad_num_samples[axis] =
            info->has_grad[axis] ? get_grad_num_samples(coll, seg_idx, blk_idx, axis) : -1;
    }

    /* RF */
    info->has_rf = block_has_rf(coll, seg_idx, blk_idx);
    info->rf_delay_us = info->has_rf ? get_rf_delay_us(coll, seg_idx, blk_idx) : -1;
    info->rf_num_channels = info->has_rf ? get_rf_num_channels(coll, seg_idx, blk_idx) : -1;
    info->rf_num_samples = info->has_rf ? get_rf_num_samples(coll, seg_idx, blk_idx) : -1;
    info->rf_duration_us = info->has_rf ? get_rf_duration_us(coll, seg_idx, blk_idx) : -1;
    info->rf_is_complex = info->has_rf ? block_rf_is_complex(coll, seg_idx, blk_idx) : 0;
    info->rf_uniform_raster =
        info->has_rf ? block_rf_has_uniform_raster(coll, seg_idx, blk_idx) : 0;

    /* ADC */
    info->has_adc = block_has_adc(coll, seg_idx, blk_idx);
    info->adc_delay_us = info->has_adc ? get_adc_delay_us(coll, seg_idx, blk_idx) : -1;
    info->adc_def_id = info->has_adc ? get_adc_library_index(coll, seg_idx, blk_idx) : -1;

    /* Digital output */
    info->has_digitalout = block_has_digitalout(coll, seg_idx, blk_idx);
    info->digitalout_delay_us =
        info->has_digitalout ? get_digitalout_delay_us(coll, seg_idx, blk_idx) : -1;
    info->digitalout_duration_us =
        info->has_digitalout ? get_digitalout_duration_us(coll, seg_idx, blk_idx) : -1;
    info->digitalout_channel =
        info->has_digitalout ? get_digitalout_channel(coll, seg_idx, blk_idx) : -1;

    /* Flags */
    info->has_rotation = block_has_rotation(coll, seg_idx, blk_idx);
    info->norot_flag = block_has_norot(coll, seg_idx, blk_idx);
    info->nopos_flag = block_has_nopos(coll, seg_idx, blk_idx);

    /* Pure-delay block: no RF/gradient/ADC waveform AND no digital-output
     * trigger or rotation -- literally only a duration -- so it CAN be played
     * as a runtime setperiod wait.  Excluding digitalout/rotation keeps this in
     * lock-step with the segment-dedup delay-flex criterion (a trigger/rotation
     * block must NOT be collapsed onto a fixed-duration wait).
     *
     * is_variable_delay is only set when the duration ALSO actually differs
     * across scan-table instances (block_is_dynamic_delay).  A pure
     * delay whose duration is constant in every instance is "static": it
     * needs no setperiod wait and no interior SSP packet, and is represented
     * purely by its block position -- exactly like any other fixed block. */
    info->is_variable_delay =
        (!info->has_grad[0] && !info->has_grad[1] && !info->has_grad[2] && !info->has_rf &&
         !info->has_adc && !info->has_digitalout && !info->has_rotation &&
         block_is_dynamic_delay(coll, seg_idx, blk_idx))
        ? 1
        : 0;

    /* Frozen at structure-build time, where the gradient shapes still exist:
     * scan loads no SHAPES section and could not re-derive this. */
    {
        const pulseg_block_initial_state *st = block_initial_state(coll, seg_idx, blk_idx);
        if (st)
        {
            info->rf_grad_constant = st->rf_grad_constant;
            info->rf_grad_level[0] = st->rf_grad_level[0];
            info->rf_grad_level[1] = st->rf_grad_level[1];
            info->rf_grad_level[2] = st->rf_grad_level[2];
        }
    }

    return PULSEG_SUCCESS;
}

int pulseg_get_adc_def(const pulseg_collection *coll, pulseg_adc_def *def, int adc_idx)
{
    if (!coll || !def)
        return PULSEG_ERR_NULL_POINTER;

    def->dwell_ns = get_adc_dwell_ns(coll, adc_idx);
    def->num_samples = get_adc_num_samples(coll, adc_idx);

    return PULSEG_SUCCESS;
}

int pulseg_get_rf_shim_def(
    const pulseg_collection *coll,
    pulseg_rf_shim_def *def,
    int subseq_idx,
    int shim_idx)
{
    const pulseg_sequence_descriptor *desc;
    int i;
    const pulseg_rf_shim_definition *src;

    if (!coll || !def)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INDEX;

    desc = &coll->descriptors[subseq_idx];

    /* shim_idx is LOCAL to its subsequence (same convention as rf_id, gx_id, etc.).
     * Each subseq stores its own rf_shim_definitions[] starting at index 0. */
    if (shim_idx < 0 || shim_idx >= desc->num_rf_shims)
        return PULSEG_ERR_INDEX;

    src = &desc->rf_shim_definitions[shim_idx];
    def->num_channels = src->num_channels;
    for (i = 0; i < src->num_channels && i < PULSEG_MAX_RF_SHIM_CHANNELS; i++)
    {
        def->magnitudes[i] = src->magnitudes[i];
        def->phases[i] = src->phases[i];
    }
    return PULSEG_SUCCESS;
}

int pulseg_get_num_rf_shims(const pulseg_collection *coll, int subseq_idx)
{
    if (!coll)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INDEX;
    return coll->descriptors[subseq_idx].num_rf_shims;
}

/* ================================================================== */
/*  Unique-block and segment-block getters                            */
/* ================================================================== */

int pulseg_get_num_unique_blocks(const pulseg_collection *coll, int subseq_idx)
{
    if (!coll)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INDEX;
    return coll->descriptors[subseq_idx].num_unique_blocks;
}

int pulseg_get_num_unique_segments(const pulseg_collection *coll, int subseq_idx)
{
    if (!coll)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INDEX;
    return coll->descriptors[subseq_idx].num_unique_segments;
}

int pulseg_get_unique_block_id(const pulseg_collection *coll, int subseq_idx, int blk_def_idx)
{
    const pulseg_sequence_descriptor *desc;
    if (!coll)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INDEX;
    desc = &coll->descriptors[subseq_idx];
    if (blk_def_idx < 0 || blk_def_idx >= desc->num_unique_blocks)
        return PULSEG_ERR_INDEX;
    /* base_blocks[].id is the 1-based .seq block index */
    return desc->base_blocks[blk_def_idx].id;
}

int pulseg_get_segment_block_def_indices(const pulseg_collection *coll, int *out_ids, int seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int local_seg, i;

    if (!coll || !out_ids)
        return PULSEG_ERR_NULL_POINTER;
    if (!resolve_segment(coll, &desc, &local_seg, seg_idx))
        return PULSEG_ERR_INDEX;

    seg = &desc->segment_definitions[local_seg];
    for (i = 0; i < seg->num_blocks; ++i)
        out_ids[i] = seg->unique_block_indices[i];
    return seg->num_blocks;
}

/* ================================================================== */
/*  Subsequence-local segment layout                                  */
/* ================================================================== */

/* Resolve subsequence-local segment local_seg_idx to its deduplicated
 * global segment id (same space as resolve_segment()'s seg_idx). Falls
 * back to the pre-dedup contiguous offset (identity map) when the
 * cross-subsequence remap has not been built. */
static int resolve_subseq_local_segment_global_id(
    const pulseg_collection *coll,
    int subseq_idx,
    int local_seg_idx)
{
    int offset;

    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return -1;
    if (local_seg_idx < 0 || local_seg_idx >= coll->descriptors[subseq_idx].num_unique_segments)
        return -1;

    offset = coll->subsequence_info[subseq_idx].segment_id_offset;
    if (coll->seg_local_to_global)
        return coll->seg_local_to_global[offset + local_seg_idx];
    return offset + local_seg_idx;
}

int pulseg_get_subseq_segment_layout(
    const pulseg_collection *coll,
    pulseg_segment_layout *info,
    int subseq_idx,
    int local_seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int global_id;

    if (!coll || !info)
        return PULSEG_ERR_NULL_POINTER;

    global_id = resolve_subseq_local_segment_global_id(coll, subseq_idx, local_seg_idx);
    if (global_id < 0)
        return PULSEG_ERR_INDEX;

    desc = &coll->descriptors[subseq_idx];
    seg = &desc->segment_definitions[local_seg_idx];

    info->global_index = global_id;
    info->num_blocks = seg->num_blocks;
    info->start_block = seg->start_block;
    info->max_energy_start_block = seg->max_energy_start_block;
    info->from_max_energy_instance =
        (seg->max_energy_start_block >= 0 && desc->exec_runs != NULL &&
         seg->max_energy_start_block + seg->num_blocks <= desc->exec_stream_len)
        ? 1
        : 0;

    return PULSEG_SUCCESS;
}

int pulseg_get_subseq_segment_block_indices(
    const pulseg_collection *coll,
    int *out_indices,
    int subseq_idx,
    int local_seg_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_virtual_segment *seg;
    int global_id, from_exec, i, blk;

    if (!coll || !out_indices)
        return PULSEG_ERR_NULL_POINTER;

    global_id = resolve_subseq_local_segment_global_id(coll, subseq_idx, local_seg_idx);
    if (global_id < 0)
        return PULSEG_ERR_INDEX;

    desc = &coll->descriptors[subseq_idx];
    seg = &desc->segment_definitions[local_seg_idx];

    from_exec =
        (seg->max_energy_start_block >= 0 && desc->exec_runs != NULL &&
         seg->max_energy_start_block + seg->num_blocks <= desc->exec_stream_len);

    for (i = 0; i < seg->num_blocks; ++i)
    {
        blk = from_exec ? pulseg__exec_block_idx(desc, seg->max_energy_start_block + i)
                        : (seg->start_block + i);
        if (blk < 0 || blk >= desc->num_blocks)
            return 0;
        out_indices[i] = blk;
    }
    return seg->num_blocks;
}

/* ================================================================== */
/*  Waveform identity, shared by the gradient-side safety checks       */
/* ================================================================== */

int pulseg__wave_key_axis(
    const pulseg_sequence_descriptor *desc,
    const pulseg_block_table_element *bte,
    int axis,
    int physical_frame,
    pulseg__wave_key *out)
{
    const pulseg_grad_table_element *tab;
    const pulseg_base_block *bdef;
    int raw;

    if (!out)
        return 0;
    out->def_id = -1;
    out->shape_id = 0;
    out->dur_us = 0;
    out->axis = -1;
    out->rotation_id = -1;
    if (!desc || !bte)
        return 0;

    raw = (axis == 0) ? bte->gx_id : (axis == 1) ? bte->gy_id : bte->gz_id;
    if (raw < 0 || raw >= desc->grad_table_size)
        return 0;
    tab = &desc->grad_table[raw];
    if (tab->id < 0 || tab->id >= desc->num_unique_grads)
        return 0;

    bdef = &desc->base_blocks[bte->id];
    out->def_id = tab->id;
    out->shape_id = tab->shape_id;
    out->dur_us = (bte->duration_us >= 0) ? bte->duration_us : bdef->duration_us;
    if (physical_frame)
    {
        out->axis = axis;
        out->rotation_id = bte->norot_flag ? -1 : bte->rotation_id;
    }
    return 1;
}

int pulseg__wave_key_equal(const pulseg__wave_key *a, const pulseg__wave_key *b)
{
    if (!a || !b)
        return 0;
    return a->def_id == b->def_id && a->shape_id == b->shape_id && a->dur_us == b->dur_us &&
        a->axis == b->axis && a->rotation_id == b->rotation_id;
}

int pulseg__wave_key_flat(const pulseg_sequence_descriptor *desc, int def_index, int shape_id)
{
    if (shape_id > 0)
        return desc->num_unique_grads + shape_id;
    return def_index;
}
