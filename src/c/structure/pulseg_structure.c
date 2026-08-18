/**
 * @file pulseg_structure.c
 * @brief Sequence structure: TR detection, segmentation, execution stream,
 *        and timing anchors.
 *
 * Turns a deduplicated block list into a playable structure: finds the
 * repeating TR pattern and its prep/cooldown regions, cuts the TR into
 * virtual segments at zero-gradient boundaries, expands the execution stream
 * over passes and averages, and derives the per-segment RF isocenter / ADC
 * k-zero anchors that freq-mod, SSP placement and the trajectory all key off.
 */

#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "pulseg_internal.h"
#include "pulseg.h"

/* ================================================================== */
/*  File-scope constants                                              */
/* ================================================================== */
/* Ceiling on what may be accepted as a single TR.  Measured over the whole
 * span, delays included, not just the active blocks: every downstream
 * consumer that resamples a TR (pulseg__get_gradient_waveforms_range, and
 * through it the safety analysis) allocates span/raster samples per axis, so
 * it is the span -- not the fraction of it that plays something -- that
 * bounds the cost.  A "single TR" longer than this is a sequence whose real
 * period was not recognised, and silently accepting it costs gigabytes. */
#define SINGLE_TR_MAX_DURATION_US 15000000 /* 15 s  */

#define SEGSTATE_SEEKING_FIRST_ADC 0
#define SEGSTATE_SEEKING_BOUNDARY 1
#define SEGSTATE_OPTIMIZED_MODE 2

/* Fixed threshold (Hz/m) for treating a gradient boundary value as zero.
 * Segment boundaries are structural: they require gradients to be zero at
 * the split point.  This must NOT depend on runtime system parameters
 * (max_slew) so that segment detection is hardware-independent.  The value
 * is generous enough to absorb floating-point noise from shape
 * decompression while remaining far below the smallest real non-zero
 * gradient boundary in practice (~4 600 Hz/m).                          */
#define SEG_ZERO_GRAD_THRESHOLD_HZ_PER_M 100.0f

/* ================================================================== */
/*  Tiny helpers                                                      */
/* ================================================================== */

/* A block definition is a pure delay when it carries no RF, gradient or ADC
 * event -- only a duration (matches the parser's block_table duration_us >= 0
 * marker).  Its duration is runtime-adjustable via setperiod. */
int pulseg__block_def_is_pure_delay(const pulseg_base_block *b)
{
    return (b->rf_id < 0 && b->gx_id < 0 && b->gy_id < 0 && b->gz_id < 0 && b->adc_id < 0);
}

/* True if the block at scan-table position scan_pos is an ADJUSTABLE pure
 * delay: no RF/grad/ADC (definition) AND no digital-output trigger or
 * non-identity rotation (table).  Matches pulseg_get_block_info()'s
 * is_variable_delay so the segment-dedup relaxation and the EPIC setperiod
 * wait stay in lock-step (a trigger/rotation delay must NOT be collapsed onto
 * a fixed wait). */
static int is_adjustable_delay_at(
    const pulseg_sequence_descriptor *desc,
    const int *scan_block_idx,
    int scan_pos)
{
    int bt_idx;
    const pulseg_block_table_element *bt;

    if (scan_pos < 0 || scan_pos >= desc->exec_stream_len)
        return 0;
    bt_idx = scan_block_idx[scan_pos];
    if (bt_idx < 0 || bt_idx >= desc->num_blocks)
        return 0;
    bt = &desc->block_table[bt_idx];
    if (!pulseg__block_def_is_pure_delay(&desc->base_blocks[bt->id]))
        return 0;
    if (bt->digitalout_id != -1)
        return 0;
    if (bt->rotation_id != -1 && bt->rotation_id < desc->num_rotations &&
        !pulseg__is_identity3(desc->rotation_matrices[bt->rotation_id]))
        return 0;
    return 1;
}

/* Compare two expanded segments for dedup equality with pure-delay flex: a
 * position whose two blocks differ is still "equal" iff BOTH are adjustable
 * pure delays (played as runtime setperiod waits), so segments differing only
 * in such a block's duration share one definition.  start_block here is a
 * scan-table position (step 7 runs before it is remapped to a block index). */
static int segs_delay_flex_equal(
    const pulseg_sequence_descriptor *desc,
    const int *scan_block_idx,
    const pulseg_virtual_segment *sa,
    const pulseg_virtual_segment *sb)
{
    int k;
    if (sa->num_blocks != sb->num_blocks)
        return 0;
    for (k = 0; k < sa->num_blocks; ++k)
    {
        if (sa->unique_block_indices[k] == sb->unique_block_indices[k])
            continue;
        if (is_adjustable_delay_at(desc, scan_block_idx, sa->start_block + k) &&
            is_adjustable_delay_at(desc, scan_block_idx, sb->start_block + k))
            continue;
        return 0;
    }
    return 1;
}

/* After strip_pure_delays_scan, pure delays are represented as one-block
 * segments whose block-table entry carries duration_us >= 0. */
static int is_single_pure_delay_segment_scan(
    const pulseg_virtual_segment *seg,
    const pulseg_block_table_element *bt,
    const int *scan_block_idx)
{
    int bt_idx;
    if (!seg || !bt || !scan_block_idx)
        return 0;
    if (seg->num_blocks != 1 || seg->start_block < 0)
        return 0;

    bt_idx = scan_block_idx[seg->start_block];
    return (bt_idx >= 0 && bt[bt_idx].duration_us >= 0) ? 1 : 0;
}

int pulseg__block_defs_structurally_equal(
    const pulseg_sequence_descriptor *desc,
    int id_a,
    int id_b)
{
    const pulseg_base_block *a;
    const pulseg_base_block *b;

    if (!desc)
        return 0;
    if (id_a < 0 || id_a >= desc->num_unique_blocks)
        return 0;
    if (id_b < 0 || id_b >= desc->num_unique_blocks)
        return 0;

    a = &desc->base_blocks[id_a];
    b = &desc->base_blocks[id_b];

    if (a->duration_us != b->duration_us)
        return 0;
    if ((a->rf_id >= 0) != (b->rf_id >= 0))
        return 0;
    if ((a->gx_id >= 0) != (b->gx_id >= 0))
        return 0;
    if ((a->gy_id >= 0) != (b->gy_id >= 0))
        return 0;
    if ((a->gz_id >= 0) != (b->gz_id >= 0))
        return 0;
    if ((a->adc_id >= 0) != (b->adc_id >= 0))
        return 0;
    return 1;
}

static int segments_structurally_equal(
    const pulseg_sequence_descriptor *desc,
    const pulseg_virtual_segment *sa,
    const pulseg_virtual_segment *sb)
{
    int i;

    if (!desc || !sa || !sb)
        return 0;
    if (sa->num_blocks != sb->num_blocks)
        return 0;

    for (i = 0; i < sa->num_blocks; ++i)
    {
        if (!pulseg__block_defs_structurally_equal(
                desc,
                sa->unique_block_indices[i],
                sb->unique_block_indices[i]))
            return 0;
    }
    return 1;
}

/* ================================================================== */
/*  TR detection helpers                                              */
/* ================================================================== */

static int first_repeating_segment(const int *s, int len)
{
    int l, i, match;

    if (len <= 1)
        return len;

    /* Find the shortest period starting at offset 0.
     * The caller already strips the prep region, so the repeating
     * pattern always starts at the beginning of the array.
     * Searching from non-zero offsets is harmful: it picks up short
     * sub-patterns (e.g. [rephaser,nav,rephaser,nav] inside an EPI
     * readout train) that don't span the whole array.               */
    for (l = 1; l <= len / 2; ++l)
    {
        match = 1;
        for (i = 0; i < l; ++i)
        {
            if (s[i] != s[i + l])
            {
                match = 0;
                break;
            }
        }
        if (match)
            return l;
    }
    return len;
}

static int first_repeating_segment_structural(
    const pulseg_sequence_descriptor *desc,
    int start,
    int len)
{
    int l, i, match;
    int a_idx, b_idx;
    int a_id, b_id;

    if (!desc || len <= 1)
        return len;

    for (l = 1; l <= len / 2; ++l)
    {
        if (len % l != 0)
            continue;
        match = 1;
        for (i = 0; i < len; ++i)
        {
            a_idx = start + i;
            b_idx = start + (i % l);
            a_id = desc->block_table[a_idx].id;
            b_id = desc->block_table[b_idx].id;
            if (!pulseg__block_defs_structurally_equal(desc, a_id, b_id))
            {
                match = 0;
                break;
            }
        }
        if (match)
            return l;
    }

    return len;
}

/* ================================================================== */
/*  build_exec_stream                                                  */
/* ================================================================== */

/*
 * Build the scan table: the block table played once per average, in table
 * order, as a flat array.
 *
 * Each entry stores the block_table index.
 * seg_id column is initialised to -1 (filled later by segment detection).
 */

static int build_exec_runs(pulseg_sequence_descriptor *desc);

int pulseg__build_exec_stream(
    pulseg_sequence_descriptor *desc,
    pulseg_diagnostic *diag,
    int num_averages)
{
    pulseg_diagnostic local_diag;
    int avg, blk, count, idx;

    if (!diag)
    {
        pulseg_diagnostic_init(&local_diag);
        diag = &local_diag;
    }
    else
        pulseg_diagnostic_init(diag);

    if (!desc)
    {
        diag->code = PULSEG_ERR_NULL_POINTER;
        return diag->code;
    }
    if (num_averages < 1)
        num_averages = 1;
    if (desc->ignore_averages)
        num_averages = 1;
    desc->num_averages = num_averages;

    count = num_averages * desc->num_blocks;

    desc->exec_stream_len = count;
    desc->exec_stream_block_idx = (int *)PULSEG_ALLOC((size_t)count * sizeof(int));
    desc->exec_stream_seg_id = (int *)PULSEG_ALLOC((size_t)count * sizeof(int));
    desc->exec_stream_avg_id = (int *)PULSEG_ALLOC((size_t)count * sizeof(int));
    if (!desc->exec_stream_block_idx || !desc->exec_stream_seg_id || !desc->exec_stream_avg_id)
    {
        if (desc->exec_stream_block_idx)
        {
            PULSEG_FREE(desc->exec_stream_block_idx);
            desc->exec_stream_block_idx = NULL;
        }
        if (desc->exec_stream_seg_id)
        {
            PULSEG_FREE(desc->exec_stream_seg_id);
            desc->exec_stream_seg_id = NULL;
        }
        if (desc->exec_stream_avg_id)
        {
            PULSEG_FREE(desc->exec_stream_avg_id);
            desc->exec_stream_avg_id = NULL;
        }
        desc->exec_stream_len = 0;
        diag->code = PULSEG_ERR_ALLOC_FAILED;
        return diag->code;
    }

    idx = 0;
    for (avg = 0; avg < num_averages; ++avg)
    {
        for (blk = 0; blk < desc->num_blocks; ++blk)
        {
            desc->exec_stream_block_idx[idx] = blk;
            desc->exec_stream_seg_id[idx] = -1;
            desc->exec_stream_avg_id[idx] = avg;
            ++idx;
        }
    }

    /* Compact mirror of the arrays just filled. */
    {
        int rc = build_exec_runs(desc);
        if (PULSEG_FAILED(rc))
        {
            diag->code = rc;
            return rc;
        }
    }

    diag->code = PULSEG_SUCCESS;
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Compact execution-stream accessors                                 */
/* ================================================================== */

/* Locate the run covering stream position @p n.
 *
 * scancore is real-time and walks the stream strictly in order, so the
 * cached hint is checked first: the same run, then its successor. Both are
 * O(1), which is what the hot path actually sees. Binary search is only the
 * random-access fallback (safety sweeps, trajectory build). The hint is a
 * pure cost optimisation -- it can be stale or wrong without affecting the
 * value returned. */
static const pulseg_exec_run *find_exec_run(const pulseg_sequence_descriptor *desc, int n)
{
    int lo, hi, h;
    const pulseg_exec_run *runs;

    if (!desc || !desc->exec_runs || desc->num_exec_runs <= 0)
        return NULL;
    if (n < 0 || n >= desc->exec_stream_len)
        return NULL;

    runs = desc->exec_runs;
    h = desc->exec_run_hint;
    if (h >= 0 && h < desc->num_exec_runs)
    {
        if (n >= runs[h].emit_start && n < runs[h].emit_start + runs[h].length)
            return &runs[h];
        if (h + 1 < desc->num_exec_runs && n >= runs[h + 1].emit_start &&
            n < runs[h + 1].emit_start + runs[h + 1].length)
        {
            ((pulseg_sequence_descriptor *)desc)->exec_run_hint = h + 1;
            return &runs[h + 1];
        }
    }

    lo = 0;
    hi = desc->num_exec_runs - 1;
    while (lo < hi)
    {
        int mid = lo + (hi - lo + 1) / 2;
        if (runs[mid].emit_start <= n)
            lo = mid;
        else
            hi = mid - 1;
    }
    ((pulseg_sequence_descriptor *)desc)->exec_run_hint = lo;
    return &runs[lo];
}

int pulseg__exec_block_idx(const pulseg_sequence_descriptor *desc, int n)
{
    const pulseg_exec_run *r = find_exec_run(desc, n);
    return r ? r->block_start + (n - r->emit_start) : -1;
}

int pulseg__exec_avg_id(const pulseg_sequence_descriptor *desc, int n)
{
    const pulseg_exec_run *r = find_exec_run(desc, n);
    return r ? r->avg_id : 0;
}

int pulseg__exec_seg_id(const pulseg_sequence_descriptor *desc, int n)
{
    const int *start;
    int lo, hi, h, m;

    if (!desc || !desc->seg_run_start || desc->num_seg_runs <= 0)
        return -1;
    if (n < 0 || n >= desc->exec_stream_len)
        return -1;

    /* Reduce into the stored period. When seg_period == exec_stream_len
     * (no periodicity verified) this is a no-op and the encoding is plain
     * RLE, so the division never runs. */
    m = (desc->seg_period > 0 && desc->seg_period < desc->exec_stream_len) ? (n % desc->seg_period)
                                                                           : n;

    start = desc->seg_run_start;
    h = desc->seg_run_hint;
    if (h >= 0 && h < desc->num_seg_runs && m >= start[h] && m < start[h + 1])
        return desc->seg_run_id[h];
    if (h + 1 < desc->num_seg_runs && m >= start[h + 1] && m < start[h + 2])
    {
        ((pulseg_sequence_descriptor *)desc)->seg_run_hint = h + 1;
        return desc->seg_run_id[h + 1];
    }

    lo = 0;
    hi = desc->num_seg_runs - 1;
    while (lo < hi)
    {
        int mid = lo + (hi - lo + 1) / 2;
        if (start[mid] <= m)
            lo = mid;
        else
            hi = mid - 1;
    }
    ((pulseg_sequence_descriptor *)desc)->seg_run_hint = lo;
    return desc->seg_run_id[lo];
}

/* 1 at the first block of each TR.  Mirrors
 * pulseg__compute_exec_stream_tr_start: positions are counted from the
 * anchor and every tr_size-th is a TR boundary. */
int pulseg__exec_tr_start(const pulseg_sequence_descriptor *desc, int n)
{
    int tr_size;

    if (!desc || n < 0 || n >= desc->exec_stream_len)
        return 0;
    if (desc->tr_start_first < 0 || n < desc->tr_start_first)
        return 0;

    tr_size = desc->tr_descriptor.tr_size;
    if (tr_size <= 0)
        return 0;
    return (((n - desc->tr_start_first) % tr_size) == 0) ? 1 : 0;
}

/* Build exec_runs[] from the same loop structure build_exec_stream uses,
 * plus the seg_id run-length encoding.  Returns PULSEG_SUCCESS or an
 * allocation error; the caller keeps the expanded arrays alive so
 * pulseg__verify_exec_runs can cross-check the two. */
static int build_exec_runs(pulseg_sequence_descriptor *desc)
{
    int n, cap, count;
    pulseg_exec_run *runs;

    if (!desc || desc->exec_stream_len <= 0)
        return PULSEG_SUCCESS;

    /* A new run starts whenever the block index is not contiguous with the
     * previous position, or avg_id changes. */
    cap = 8;
    runs = (pulseg_exec_run *)PULSEG_ALLOC((size_t)cap * sizeof(pulseg_exec_run));
    if (!runs)
        return PULSEG_ERR_ALLOC_FAILED;
    count = 0;

    for (n = 0; n < desc->exec_stream_len; ++n)
    {
        int b = desc->exec_stream_block_idx[n];
        int a = desc->exec_stream_avg_id[n];
        int extend = 0;

        if (count > 0)
        {
            pulseg_exec_run *last = &runs[count - 1];
            extend = (last->avg_id == a && last->block_start + last->length == b);
        }
        if (extend)
        {
            ++runs[count - 1].length;
            continue;
        }
        if (count == cap)
        {
            pulseg_exec_run *grown;
            int newcap = cap * 2;
            grown = (pulseg_exec_run *)PULSEG_ALLOC((size_t)newcap * sizeof(pulseg_exec_run));
            if (!grown)
            {
                PULSEG_FREE(runs);
                return PULSEG_ERR_ALLOC_FAILED;
            }
            memcpy(grown, runs, (size_t)cap * sizeof(pulseg_exec_run));
            PULSEG_FREE(runs);
            runs = grown;
            cap = newcap;
        }
        runs[count].emit_start = n;
        runs[count].block_start = b;
        runs[count].length = 1;
        runs[count].avg_id = a;
        ++count;
    }

    if (desc->exec_runs)
        PULSEG_FREE(desc->exec_runs);
    desc->exec_runs = runs;
    desc->exec_run_hint = 0;
    desc->num_exec_runs = count;
    return PULSEG_SUCCESS;
}

/* Run-length encode exec_stream_seg_id. Separate from build_exec_runs
 * because seg_id is only assigned later, by
 * pulseg__get_exec_stream_segments. */
/* Is seg_id[n] == seg_id[n % period] for every position? Verified against
 * the real data -- nothing about TR topology is taken on faith. */
static int seg_period_holds(const pulseg_sequence_descriptor *desc, int period)
{
    int n, len = desc->exec_stream_len;
    const int *v = desc->exec_stream_seg_id;

    if (period <= 0 || period >= len)
        return 0;
    for (n = period; n < len; ++n)
        if (v[n] != v[n - period])
            return 0;
    return 1;
}

int pulseg__build_seg_runs(pulseg_sequence_descriptor *desc)
{
    int n, count, w, period, cand[2], ci;
    int *starts, *ids;

    if (!desc || desc->exec_stream_len <= 0 || !desc->exec_stream_seg_id)
        return PULSEG_SUCCESS;

    /* Pick the smallest candidate period that actually verifies. The
     * candidates are the two structural repeats the converter already
     * knows about (one imaging TR, one average); each is CHECKED, and if
     * neither holds the encoding stays flat. */
    cand[0] = desc->tr_descriptor.tr_size;
    cand[1] = desc->num_blocks;
    if (cand[1] < cand[0])
    {
        int t = cand[0];
        cand[0] = cand[1];
        cand[1] = t;
    }
    period = desc->exec_stream_len;
    for (ci = 0; ci < 2; ++ci)
    {
        if (cand[ci] > 0 && cand[ci] < period && (desc->exec_stream_len % cand[ci]) == 0 &&
            seg_period_holds(desc, cand[ci]))
        {
            period = cand[ci];
            break;
        }
    }
    desc->seg_period = period;

    count = 0;
    for (n = 0; n < period; ++n)
        if (n == 0 || desc->exec_stream_seg_id[n] != desc->exec_stream_seg_id[n - 1])
            ++count;

    /* One extra start entry so run r spans [start[r], start[r+1]). */
    starts = (int *)PULSEG_ALLOC((size_t)(count + 1) * sizeof(int));
    ids = (int *)PULSEG_ALLOC((size_t)count * sizeof(int));
    if (!starts || !ids)
    {
        if (starts)
            PULSEG_FREE(starts);
        if (ids)
            PULSEG_FREE(ids);
        return PULSEG_ERR_ALLOC_FAILED;
    }

    w = 0;
    for (n = 0; n < period; ++n)
    {
        if (n == 0 || desc->exec_stream_seg_id[n] != desc->exec_stream_seg_id[n - 1])
        {
            starts[w] = n;
            ids[w] = desc->exec_stream_seg_id[n];
            ++w;
        }
    }
    starts[count] = period;
    desc->seg_run_hint = 0;

    if (desc->seg_run_start)
        PULSEG_FREE(desc->seg_run_start);
    if (desc->seg_run_id)
        PULSEG_FREE(desc->seg_run_id);
    desc->seg_run_start = starts;
    desc->seg_run_id = ids;
    desc->num_seg_runs = count;
    return PULSEG_SUCCESS;
}

/* Release the per-position scratch arrays once the compact form is built.
 * They exist only so the builders (and pulseg__verify_exec_runs) can work
 * position-by-position during conversion; every consumer goes through the
 * accessors, and a cache load never materialises them at all. */
void pulseg__free_exec_stream_scratch(pulseg_sequence_descriptor *desc)
{
    if (!desc)
        return;
    if (desc->exec_stream_block_idx)
    {
        PULSEG_FREE(desc->exec_stream_block_idx);
        desc->exec_stream_block_idx = NULL;
    }
    if (desc->exec_stream_seg_id)
    {
        PULSEG_FREE(desc->exec_stream_seg_id);
        desc->exec_stream_seg_id = NULL;
    }
    if (desc->exec_stream_avg_id)
    {
        PULSEG_FREE(desc->exec_stream_avg_id);
        desc->exec_stream_avg_id = NULL;
    }
    if (desc->exec_stream_tr_start)
    {
        PULSEG_FREE(desc->exec_stream_tr_start);
        desc->exec_stream_tr_start = NULL;
    }
}

/* Cross-check the compact form against the expanded arrays at every
 * position. Returns 0 on full agreement, or the number of mismatches. */
long pulseg__verify_exec_runs(const pulseg_sequence_descriptor *desc)
{
    long bad = 0;
    int n;

    if (!desc)
        return 0;
    for (n = 0; n < desc->exec_stream_len; ++n)
    {
        if (desc->exec_stream_block_idx &&
            pulseg__exec_block_idx(desc, n) != desc->exec_stream_block_idx[n])
            ++bad;
        if (desc->exec_stream_avg_id && pulseg__exec_avg_id(desc, n) != desc->exec_stream_avg_id[n])
            ++bad;
        if (desc->exec_stream_seg_id && pulseg__exec_seg_id(desc, n) != desc->exec_stream_seg_id[n])
            ++bad;
        if (desc->exec_stream_tr_start &&
            pulseg__exec_tr_start(desc, n) != desc->exec_stream_tr_start[n])
            ++bad;
    }
    return bad;
}

/* ================================================================== */
/*  Compute exec_stream_tr_start flags                                 */
/* ================================================================== */

/**
 * @brief Set exec_stream_tr_start[i] = 1 at the first block of each TR.
 *
 * Anchors at stream position 0 and marks every tr_size-th position.
 * Safe to call multiple times (re-allocates and re-computes).
 */
void pulseg__compute_exec_stream_tr_start(pulseg_sequence_descriptor *desc)
{
    int i, tr_size;

    if (!desc || desc->exec_stream_len == 0)
        return;

    /* (Re)allocate */
    if (desc->exec_stream_tr_start)
        PULSEG_FREE(desc->exec_stream_tr_start);
    desc->exec_stream_tr_start = (int *)PULSEG_ALLOC((size_t)desc->exec_stream_len * sizeof(int));
    if (!desc->exec_stream_tr_start)
        return;
    memset(desc->exec_stream_tr_start, 0, (size_t)desc->exec_stream_len * sizeof(int));

    desc->tr_start_first = -1;

    tr_size = desc->tr_descriptor.tr_size;
    if (tr_size <= 0)
        return;

    desc->tr_start_first = 0;
    for (i = 0; i < desc->exec_stream_len; ++i)
        if (i % tr_size == 0)
            desc->exec_stream_tr_start[i] = 1;
}

/* ================================================================== */
/*  find_tr_in_sequence                                               */
/* ================================================================== */

int pulseg__get_tr_in_sequence(pulseg_sequence_descriptor *desc, pulseg_diagnostic *diag)
{
    pulseg_tr_descriptor *tr = &desc->tr_descriptor;
    pulseg_diagnostic local_diag;
    int i, n;
    int imaging_start, imaging_end, imaging_len;
    int *seq_pat = NULL;
    int *base_pat = NULL;
    int *block_dur = NULL;
    double span_dur_us; /* one pass, delays included; can exceed an int */
    int found, l;
    int mismatch_pos;
    float tr_dur;
    int tr_start;

    if (!diag)
    {
        pulseg_diagnostic_init(&local_diag);
        diag = &local_diag;
    }
    else
        pulseg_diagnostic_init(diag);

    found = 0;
    l = 0;
    mismatch_pos = -1;

    if (desc->num_blocks <= 0 || !desc->block_table || !desc->base_blocks)
    {
        diag->code = (desc->num_blocks <= 0) ? PULSEG_ERR_TR_NO_BLOCKS : PULSEG_ERR_NULL_POINTER;
        return diag->code;
    }
    tr->tr_size = 0;
    tr->num_trs = 0;
    tr->tr_duration_us = 0.0f;

    imaging_start = 0;
    imaging_end = desc->num_blocks;
    imaging_len = imaging_end - imaging_start;
    pulseg__diag_printf(diag, "imaging region length=%d", imaging_len);

    if (imaging_len <= 0)
    {
        diag->code = PULSEG_ERR_TR_NO_IMAGING_REGION;
        return diag->code;
    }

    /* unique-block count for diagnostics */
    {
        int max_u = 0;
        for (n = 0; n < desc->num_blocks; ++n)
            if (desc->block_table[n].id > max_u)
                max_u = desc->block_table[n].id;
        pulseg__diag_printf(diag, " unique blocks=%d", max_u + 1);
    }

    seq_pat = (int *)PULSEG_ALLOC(desc->num_blocks * sizeof(int));
    block_dur = (int *)PULSEG_ALLOC(desc->num_blocks * sizeof(int));
    if (!seq_pat || !block_dur)
    {
        if (seq_pat)
            PULSEG_FREE(seq_pat);
        if (block_dur)
            PULSEG_FREE(block_dur);
        diag->code = PULSEG_ERR_ALLOC_FAILED;
        return diag->code;
    }

    for (n = 0; n < desc->num_blocks; ++n)
    {
        block_dur[n] = desc->base_blocks[desc->block_table[n].id].duration_us;
        seq_pat[n] =
            (desc->block_table[n].duration_us >= 0) ? block_dur[n] : -1 * desc->block_table[n].id;
    }

    /* Save a copy of seq_pat before RF augmentation (used for VFA check) */
    base_pat = (int *)PULSEG_ALLOC(desc->num_blocks * sizeof(int));
    if (!base_pat)
    {
        PULSEG_FREE(seq_pat);
        PULSEG_FREE(block_dur);
        diag->code = PULSEG_ERR_ALLOC_FAILED;
        return diag->code;
    }
    for (n = 0; n < desc->num_blocks; ++n)
        base_pat[n] = seq_pat[n];

    /* Canonical TR identification is timing/block-structure based.
     * Per-instance RF amplitude or shim patterns are validated by
     * dedicated safety consistency checks, not by TR-period finding. */

    l = first_repeating_segment(&seq_pat[imaging_start], imaging_len);
    if (l == imaging_len)
    {
        int l_struct = first_repeating_segment_structural(desc, imaging_start, imaging_len);
        if (l_struct > 0 && l_struct < l)
            l = l_struct;
    }
    pulseg__diag_printf(diag, " candidate TR=%d", l);

    /* l == imaging_len means neither search found a period: the region is its
     * own shortest repeat.  That is not a discovery, so it must not short
     * circuit past the fallback chain below -- the single-TR branch there is
     * where such a region is length-checked before being accepted. */
    found = (l > 0 && l < imaging_len) ? 1 : 0;

    if (found)
    {
        for (i = 0; i < imaging_len; ++i)
        {
            n = imaging_start + i;
            if (seq_pat[n] != seq_pat[imaging_start + (i % l)])
            {
                mismatch_pos = i;
                pulseg__diag_printf(diag, " mismatch at offset=%d", i);
                pulseg__diag_printf(diag, " block=%d", n);
                found = 0;
                break;
            }
        }
    }

    if (!found)
    {
        /* ---------------------------------------------------------
         * VFA rejection: if the structural (base) pattern has a
         * valid shorter period but the RF-augmented pattern does
         * not, the sequence contains non-periodic RF over a
         * repeating structure (e.g. VFA SPGR).  Such sequences
         * should be designed as separate subsequences, so we
         * reject them instead of falling through to single-TR.
         * --------------------------------------------------------- */
        int base_l, base_ok;
        base_l = first_repeating_segment(&base_pat[imaging_start], imaging_len);
        base_ok = 0;
        if (base_l > 0 && base_l < imaging_len)
        {
            base_ok = 1;
            for (i = 0; i < imaging_len && base_ok; ++i)
            {
                if (base_pat[imaging_start + i] != base_pat[imaging_start + (i % base_l)])
                    base_ok = 0;
            }
        }

        if (!base_ok)
        {
            int structural_l;

            /* Fallback for degenerate/no-prep-no-cool flows where
             * block IDs differ across interleaves but the ordered
             * block structure is periodic. */
            structural_l = first_repeating_segment_structural(desc, imaging_start, imaging_len);
            if (structural_l > 0 && structural_l < imaging_len)
            {
                l = structural_l;
                found = 1;
            }

            if (found)
            {
                /* Structural period recovered; continue with normal
                 * TR descriptor population and degeneracy checks. */
            }
            else
            {
                /* Base pattern also non-periodic -> genuine single-TR:
                 * the whole table IS the single TR. */
                span_dur_us = 0.0;
                for (n = 0; n < desc->num_blocks; ++n)
                    span_dur_us +=
                        (double)((desc->block_table[n].duration_us >= 0)
                                     ? desc->block_table[n].duration_us
                                     : desc->base_blocks[desc->block_table[n].id].duration_us);

                if (span_dur_us <= (double)SINGLE_TR_MAX_DURATION_US)
                {
                    l = imaging_len; /* single-TR = the entire table */
                    found = 1;
                }
            }
        }

        if (!found)
        {
            /* VFA case (base_ok=1) or genuinely too-long non-periodic:
             * reject with a pattern error. */
            diag->code = (mismatch_pos >= 0) ? PULSEG_ERR_TR_PATTERN_MISMATCH
                                             : PULSEG_ERR_TR_NO_PERIODIC_PATTERN;
            PULSEG_FREE(seq_pat);
            PULSEG_FREE(block_dur);
            PULSEG_FREE(base_pat);
            return diag->code;
        }
    }

    tr->tr_size = l;
    tr_dur = 0.0f;
    tr_start = imaging_start;
    for (i = 0; i < l; ++i)
        tr_dur += (float)block_dur[tr_start + i];
    tr->tr_duration_us = tr_dur;

    tr->num_trs = imaging_len / l;

    diag->code = PULSEG_SUCCESS;
    PULSEG_FREE(seq_pat);
    PULSEG_FREE(block_dur);
    PULSEG_FREE(base_pat);
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  NAV-aware split / merge                                           */
/* ================================================================== */

/**
 * @brief Split segments at NAV / non-NAV boundaries, merge adjacent NAV.
 *
 * After strip_pure_delays, every expanded segment covers a contiguous
 * run of blocks.  This function:
 *   1. Splits any segment whose blocks have mixed nav_flag values into
 *      contiguous runs of identical NAV state.
 *   2. Merges adjacent segments that are both NAV and contiguous in
 *      block order.
 *
 * Ownership of in[].unique_block_indices is transferred: they are freed
 * inside this function (set to NULL on the input side).
 *
 * @param[in]  in       Input expanded segments for one section
 * @param[in]  num_in   Count of input segments
 * @param[out] out      Pre-allocated output array
 * @param[in]  max_out  Capacity of output array
 * @param[in]  bt       Block table (for nav_flag lookup)
 * @param[in]  scan_bi  If non-NULL, resolve through exec_stream_block_idx
 * @return Number of output segments, or -1 on allocation failure.
 */
static int nav_split_merge(
    pulseg_virtual_segment *in,
    int num_in,
    pulseg_virtual_segment *out,
    int max_out,
    const pulseg_block_table_element *bt,
    const int *scan_bi)
{
    pulseg_virtual_segment *split_buf = NULL;
    int *new_ubi;
    int split_max, num_split, num_out;
    int n, b, k;
    int sb, nb, run_start, run_len, cur_nav, blk_nav;
    int bt_idx, this_nav, prev_nav;

    if (num_in == 0)
        return 0;

    /* worst case: every block becomes its own segment */
    split_max = 0;
    for (n = 0; n < num_in; ++n)
        split_max += in[n].num_blocks;
    if (split_max == 0)
        return 0;

    split_buf =
        (pulseg_virtual_segment *)PULSEG_ALLOC((size_t)split_max * sizeof(pulseg_virtual_segment));
    if (!split_buf)
        return -1;

    /* ---- Pass 1: split at NAV transitions ---- */
    num_split = 0;
    for (n = 0; n < num_in; ++n)
    {
        sb = in[n].start_block;
        nb = in[n].num_blocks;
        if (nb <= 0)
        {
            PULSEG_FREE(in[n].unique_block_indices);
            in[n].unique_block_indices = NULL;
            continue;
        }

        bt_idx = scan_bi ? scan_bi[sb] : sb;
        cur_nav = bt[bt_idx].nav_flag ? 1 : 0;
        run_start = 0;

        for (b = 1; b <= nb; ++b)
        {
            if (b < nb)
            {
                bt_idx = scan_bi ? scan_bi[sb + b] : (sb + b);
                blk_nav = bt[bt_idx].nav_flag ? 1 : 0;
            }
            else
            {
                blk_nav = -1; /* sentinel — force flush of last run */
            }

            if (blk_nav != cur_nav)
            {
                run_len = b - run_start;
                split_buf[num_split].start_block = sb + run_start;
                split_buf[num_split].num_blocks = run_len;
                split_buf[num_split].unique_block_indices =
                    (int *)PULSEG_ALLOC((size_t)run_len * sizeof(int));
                if (!split_buf[num_split].unique_block_indices)
                {
                    for (k = 0; k < num_split; ++k)
                        PULSEG_FREE(split_buf[k].unique_block_indices);
                    PULSEG_FREE(split_buf);
                    return -1;
                }
                for (k = 0; k < run_len; ++k)
                    split_buf[num_split].unique_block_indices[k] =
                        in[n].unique_block_indices[run_start + k];
                num_split++;
                run_start = b;
                cur_nav = blk_nav;
            }
        }

        PULSEG_FREE(in[n].unique_block_indices);
        in[n].unique_block_indices = NULL;
    }

    /* ---- Pass 2: merge adjacent NAV segments ---- */
    num_out = 0;
    for (n = 0; n < num_split; ++n)
    {
        bt_idx = scan_bi ? scan_bi[split_buf[n].start_block] : split_buf[n].start_block;
        this_nav = bt[bt_idx].nav_flag ? 1 : 0;

        if (this_nav && num_out > 0)
        {
            pulseg_virtual_segment *prev = &out[num_out - 1];
            bt_idx = scan_bi ? scan_bi[prev->start_block] : prev->start_block;
            prev_nav = bt[bt_idx].nav_flag ? 1 : 0;

            if (prev_nav && prev->start_block + prev->num_blocks == split_buf[n].start_block)
            {
                /* merge into previous segment */
                int old_nb = prev->num_blocks;
                int add_nb = split_buf[n].num_blocks;
                int new_nb = old_nb + add_nb;
                new_ubi = (int *)PULSEG_ALLOC((size_t)new_nb * sizeof(int));
                if (!new_ubi)
                {
                    for (k = n; k < num_split; ++k)
                        PULSEG_FREE(split_buf[k].unique_block_indices);
                    PULSEG_FREE(split_buf);
                    return -1;
                }
                for (k = 0; k < old_nb; ++k)
                    new_ubi[k] = prev->unique_block_indices[k];
                for (k = 0; k < add_nb; ++k)
                    new_ubi[old_nb + k] = split_buf[n].unique_block_indices[k];
                PULSEG_FREE(prev->unique_block_indices);
                PULSEG_FREE(split_buf[n].unique_block_indices);
                split_buf[n].unique_block_indices = NULL;
                prev->unique_block_indices = new_ubi;
                prev->num_blocks = new_nb;
                continue;
            }
        }

        if (num_out >= max_out)
        {
            for (k = n; k < num_split; ++k)
                PULSEG_FREE(split_buf[k].unique_block_indices);
            PULSEG_FREE(split_buf);
            return -1;
        }
        out[num_out] = split_buf[n];
        split_buf[n].unique_block_indices = NULL; /* ownership transferred */
        num_out++;
    }

    PULSEG_FREE(split_buf);
    return num_out;
}

/* ================================================================== */
/*  Label table dry-run                                               */
/* ================================================================== */

/*
 * apply_block_labels --
 *   Scan a block's extension chain and apply LABELSET / LABELINC
 *   operations to the running label state.
 *
 *   state[0..9]  = { slc, phs, rep, avg, seg, set, eco, par, lin, acq }
 *   state[10]    = OFF (Pulseq v1.5.1 LABELSET; 1 = discard downstream)
 */
static void apply_block_labels(int *state, const pulseq_file *seq, const pulseq_raw_block *raw)
{
    int i, type_idx, ref_idx, ext_type, label_value, label_id;

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

        if (ext_type == PULSEQ_EXT_LABELSET)
        {
            if (!seq->labelset_library || ref_idx >= seq->labelset_library_size)
                continue;
            label_value = (int)seq->labelset_library[ref_idx][0];
            label_id = (int)seq->labelset_library[ref_idx][1];
            switch (label_id)
            {
            case PULSEQ_LABEL_SLC:
                state[0] = label_value;
                break;
            case PULSEQ_LABEL_PHS:
                state[1] = label_value;
                break;
            case PULSEQ_LABEL_REP:
                state[2] = label_value;
                break;
            case PULSEQ_LABEL_AVG:
                state[3] = label_value;
                break;
            case PULSEQ_LABEL_SEG:
                state[4] = label_value;
                break;
            case PULSEQ_LABEL_SET:
                state[5] = label_value;
                break;
            case PULSEQ_LABEL_ECO:
                state[6] = label_value;
                break;
            case PULSEQ_LABEL_PAR:
                state[7] = label_value;
                break;
            case PULSEQ_LABEL_LIN:
                state[8] = label_value;
                break;
            case PULSEQ_LABEL_ACQ:
                state[9] = label_value;
                break;
            case PULSEQ_LABEL_OFF:
                state[10] = label_value ? 1 : 0;
                break;
            default:
                break;
            }
        }
        else if (ext_type == PULSEQ_EXT_LABELINC)
        {
            if (!seq->labelinc_library || ref_idx >= seq->labelinc_library_size)
                continue;
            label_value = (int)seq->labelinc_library[ref_idx][0];
            label_id = (int)seq->labelinc_library[ref_idx][1];
            switch (label_id)
            {
            case PULSEQ_LABEL_SLC:
                state[0] += label_value;
                break;
            case PULSEQ_LABEL_PHS:
                state[1] += label_value;
                break;
            case PULSEQ_LABEL_REP:
                state[2] += label_value;
                break;
            case PULSEQ_LABEL_AVG:
                state[3] += label_value;
                break;
            case PULSEQ_LABEL_SEG:
                state[4] += label_value;
                break;
            case PULSEQ_LABEL_SET:
                state[5] += label_value;
                break;
            case PULSEQ_LABEL_ECO:
                state[6] += label_value;
                break;
            case PULSEQ_LABEL_PAR:
                state[7] += label_value;
                break;
            case PULSEQ_LABEL_LIN:
                state[8] += label_value;
                break;
            case PULSEQ_LABEL_ACQ:
                state[9] += label_value;
                break;
            /* OFF is boolean; LABELINC on OFF is undefined and ignored. */
            default:
                break;
            }
        }
    }
}

/*
 * record_adc_label --
 *   Record the current label state into one row of the label table
 *   and update label_limits min/max tracking.
 *
 *   The 3 output columns are driven by label_column_map: each entry
 *   is a state-array index (0=SLC,1=PHS,2=REP,3=AVG,4=SEG,5=SET,6=ECO,
 *   7=PAR,8=LIN,9=ACQ). GE's convention is {8,0,6} = [LIN, SLC, ECO].
 *   Named limits (lin/slc/eco/...) always track every label by its fixed
 *   state index, independent of which 3 are chosen as table columns.
 */
static void record_adc_label(
    int *table_row,
    int num_columns,
    pulseg_label_limits *limits,
    const int *state,
    const int *label_column_map,
    int is_first)
{
    int col;

    (void)num_columns; /* always 3 */

    for (col = 0; col < 3; ++col)
        table_row[col] = state[label_column_map[col]];

    if (is_first)
    {
        limits->lin.min = state[8];
        limits->lin.max = state[8];
        limits->slc.min = state[0];
        limits->slc.max = state[0];
        limits->eco.min = state[6];
        limits->eco.max = state[6];
        /* Also init the other label limits from state */
        limits->phs.min = state[1];
        limits->phs.max = state[1];
        limits->rep.min = state[2];
        limits->rep.max = state[2];
        limits->avg.min = state[3];
        limits->avg.max = state[3];
        limits->seg.min = state[4];
        limits->seg.max = state[4];
        limits->set.min = state[5];
        limits->set.max = state[5];
        limits->par.min = state[7];
        limits->par.max = state[7];
        limits->acq.min = state[9];
        limits->acq.max = state[9];
    }
    else
    {
        if (state[8] < limits->lin.min)
            limits->lin.min = state[8];
        if (state[8] > limits->lin.max)
            limits->lin.max = state[8];
        if (state[0] < limits->slc.min)
            limits->slc.min = state[0];
        if (state[0] > limits->slc.max)
            limits->slc.max = state[0];
        if (state[6] < limits->eco.min)
            limits->eco.min = state[6];
        if (state[6] > limits->eco.max)
            limits->eco.max = state[6];
        if (state[1] < limits->phs.min)
            limits->phs.min = state[1];
        if (state[1] > limits->phs.max)
            limits->phs.max = state[1];
        if (state[2] < limits->rep.min)
            limits->rep.min = state[2];
        if (state[2] > limits->rep.max)
            limits->rep.max = state[2];
        if (state[3] < limits->avg.min)
            limits->avg.min = state[3];
        if (state[3] > limits->avg.max)
            limits->avg.max = state[3];
        if (state[4] < limits->seg.min)
            limits->seg.min = state[4];
        if (state[4] > limits->seg.max)
            limits->seg.max = state[4];
        if (state[5] < limits->set.min)
            limits->set.min = state[5];
        if (state[5] > limits->set.max)
            limits->set.max = state[5];
        if (state[7] < limits->par.min)
            limits->par.min = state[7];
        if (state[7] > limits->par.max)
            limits->par.max = state[7];
        if (state[9] < limits->acq.min)
            limits->acq.min = state[9];
        if (state[9] > limits->acq.max)
            limits->acq.max = state[9];
    }
}

int pulseg__build_label_table(pulseg_sequence_descriptor *desc, const pulseq_file *seq)
{
    int num_columns, total_adcs;
    int n, b, entry_idx;
    int state[11];
    int *table;
    int *off_table = NULL;
    pulseq_raw_block raw;

    if (!desc || !seq)
        return PULSEG_ERR_NULL_POINTER;

    /* The label table is always 3 columns wide; WHICH labels occupy them is
     * the vendor's choice, injected via desc->label_column_map. */
    num_columns = 3;

    /* One row per ADC occurrence, indexed by the ordinal of that ADC along
     * the execution stream -- the same index space pulseg__compute_trajectory
     * uses for traj_table_entry[], gated on the identical per-instance
     * block_table[b].adc_id >= 0 test (dummy ADC placeholders carry -1).
     *
     * The stream is the only correct domain to walk here: block_table is
     * already fully expanded over every TR, and the stream additionally
     * covers all passes and average replays.  Counting per region and
     * scaling by num_trs (as this did before) double-counted the main
     * region by num_trs, while a pass_len-bounded walk missed every pass
     * after the first. */
    total_adcs = 0;
    for (n = 0; n < desc->exec_stream_len; ++n)
    {
        b = pulseg__exec_block_idx(desc, n);
        if (b >= 0 && b < desc->num_blocks && desc->block_table[b].adc_id >= 0)
            ++total_adcs;
    }

    if (total_adcs == 0)
    {
        desc->label_num_columns = num_columns;
        desc->label_num_entries = 0;
        desc->label_table = NULL;
        desc->off_table = NULL;
        memset(&desc->label_limits, 0, sizeof(desc->label_limits));
        return PULSEG_SUCCESS;
    }

    /* Allocate table */
    table = (int *)PULSEG_ALLOC((size_t)total_adcs * (size_t)num_columns * sizeof(int));
    if (!table)
        return PULSEG_ERR_ALLOC_FAILED;
    memset(table, 0, (size_t)total_adcs * (size_t)num_columns * sizeof(int));

    /* Parallel OFF flag table (one int per ADC). */
    off_table = (int *)PULSEG_ALLOC((size_t)total_adcs * sizeof(int));
    if (!off_table)
    {
        PULSEG_FREE(table);
        return PULSEG_ERR_ALLOC_FAILED;
    }
    memset(off_table, 0, (size_t)total_adcs * sizeof(int));

    /* Initialize running label state to zero */
    memset(state, 0, sizeof(state));
    memset(&desc->label_limits, 0, sizeof(desc->label_limits));
    entry_idx = 0;

    /* Walk the execution stream in playback order.  block_table index ==
     * raw block index (pulseg__deduplicate_blocks fills block_table[n] from
     * raw block n), so the stream position maps straight onto the raw block
     * whose LABELSET/LABELINC directives advance the running state. */
    for (n = 0; n < desc->exec_stream_len; ++n)
    {
        b = pulseg__exec_block_idx(desc, n);
        if (b < 0 || b >= desc->num_blocks)
            continue;

        pulseq_get_raw_block_content_ids(seq, &raw, b, 1);
        apply_block_labels(state, seq, &raw);
        if (desc->block_table[b].adc_id >= 0)
        {
            record_adc_label(
                &table[entry_idx * num_columns],
                num_columns,
                &desc->label_limits,
                state,
                desc->label_column_map,
                entry_idx == 0);
            off_table[entry_idx] = state[10];
            ++entry_idx;
        }
    }

    desc->label_num_columns = num_columns;
    desc->label_num_entries = entry_idx;
    desc->label_table = table;
    desc->off_table = off_table;

    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Scan-table-aware segment state machine                            */
/* ================================================================== */

/* See pulseg_internal.h. */
static float grad_boundary_value(const pulseg_sequence_descriptor *desc, int raw_id, int want_last)
{
    const pulseg_grad_table_element *gte;
    const pulseg_grad_definition *gdef;
    float normalised;

    if (!desc || raw_id < 0 || raw_id >= desc->grad_table_size)
        return 0.0f;
    gte = &desc->grad_table[raw_id];
    if (gte->shape_id <= 0 || gte->shape_id > desc->num_grad_shape_stats)
        return 0.0f; /* trapezoid, or a shape with no recorded endpoints */
    if (!desc->grad_shape_first || !desc->grad_shape_last)
        return 0.0f;

    normalised = want_last ? pulseg__grad_shape_last(desc, gte->shape_id)
                           : pulseg__grad_shape_first(desc, gte->shape_id);
    gdef = &desc->grad_definitions[gte->id];
    return normalised * gdef->any.max_amplitude;
}

float pulseg__grad_shape_first(const pulseg_sequence_descriptor *desc, int shape_id)
{
    if (!desc || !desc->grad_shape_first || shape_id <= 0 || shape_id > desc->num_grad_shape_stats)
        return 0.0f;
    return desc->grad_shape_first[shape_id - 1];
}

float pulseg__grad_shape_last(const pulseg_sequence_descriptor *desc, int shape_id)
{
    if (!desc || !desc->grad_shape_last || shape_id <= 0 || shape_id > desc->num_grad_shape_stats)
        return 0.0f;
    return desc->grad_shape_last[shape_id - 1];
}

float pulseg__grad_boundary_first(const pulseg_sequence_descriptor *desc, int raw_id)
{
    return grad_boundary_value(desc, raw_id, 0);
}

float pulseg__grad_boundary_last(const pulseg_sequence_descriptor *desc, int raw_id)
{
    return grad_boundary_value(desc, raw_id, 1);
}

/* Normalised waveform values are in [-1, 1], so a plain absolute tolerance is
 * a relative one.  Loose enough to absorb the float32 round trip through the
 * shape codec, tight enough that a real ramp -- which crosses the whole range
 * over a few tens of microseconds -- never reads as flat. */
#define PULSEG_GRAD_FLAT_TOL 1e-5f

/*
 * Is the gradient that grad-table row @p raw_id plays FLAT across the window
 * [@p t0_us, @p t1_us], both measured from the start of the block playing it?
 *
 * Flat means the normalised waveform does not change over the window, so the
 * physical gradient during it is one vector rather than a function of time.
 * That is the entire premise of moving a selective excitation with a carrier
 * offset: with a single G, translating the excited slab by dr is exactly a
 * frequency shift of G.dr, and with a time-varying G it is not a frequency
 * shift at all.
 *
 * A window that runs off either end of the event is deliberately NOT flat.
 * Outside its support the gradient is zero, which is constant but is not the
 * gradient the window is asking about; accepting it would silently sweep the
 * ramps into the answer.
 *
 * Returns 1 and writes the signed normalised level to *level_out, else 0.
 */
static int grad_flat_over_window(
    const pulseg_sequence_descriptor *desc,
    int raw_id,
    float t0_us,
    float t1_us,
    float *level_out)
{
    const pulseg_grad_table_element *gte;
    const pulseg_grad_definition *gdef;
    pulseq_shape wave, tshape;
    float delay_us, rise_us, flat_us;
    float grad_raster_us, level;
    int num_samples, has_time_shape, i, i0, i1, flat;

    if (!desc || !level_out || raw_id < 0 || raw_id >= desc->grad_table_size)
        return 0;
    gte = &desc->grad_table[raw_id];
    if (gte->id < 0 || gte->id >= desc->num_unique_grads)
        return 0;
    gdef = &desc->grad_definitions[gte->id];
    delay_us = (float)gdef->delay;

    if (gdef->type == 0)
    {
        /* Trapezoid: the only flat stretch is the plateau, and it is flat by
         * construction -- no shape to inspect, just the timing. */
        rise_us = (float)gdef->rise_time_or_unused;
        flat_us = (float)gdef->flat_time_or_unused;
        if (flat_us <= 0.0f)
            return 0;
        if (t0_us < delay_us + rise_us || t1_us > delay_us + rise_us + flat_us)
            return 0;
        *level_out = 1.0f;
        return 1;
    }

    /* Arbitrary: walk the samples the window spans.  The times must match
     * pulseg__render_grad_block exactly, or the window lands on the wrong
     * samples -- hence the same time-shape / half-raster split. */
    num_samples = gdef->fall_time_or_num_uncompressed_samples;
    if (num_samples <= 0 || gte->shape_id <= 0 || gte->shape_id > desc->num_shapes)
        return 0;

    wave.samples = NULL;
    tshape.samples = NULL;
    if (!pulseq_decompress_shape(&wave, &desc->shapes[gte->shape_id - 1], 1.0f))
        return 0;
    if (wave.num_uncompressed_samples < num_samples)
    {
        PULSEG_FREE(wave.samples);
        return 0;
    }

    grad_raster_us = desc->grad_raster_us;
    has_time_shape = 0;
    if (gdef->unused_or_time_shape_id > 0 && gdef->unused_or_time_shape_id <= desc->num_shapes)
    {
        if (pulseq_decompress_shape(
                &tshape,
                &desc->shapes[gdef->unused_or_time_shape_id - 1],
                grad_raster_us) &&
            tshape.num_uncompressed_samples >= num_samples)
            has_time_shape = 1;
    }

    /* Bracket the window: the last sample at or before t0, the first at or
     * after t1.  Interpolation runs between samples, so both brackets have to
     * be inside the event and every sample between them has to agree. */
    i0 = -1;
    i1 = -1;
    for (i = 0; i < num_samples; ++i)
    {
        float t = has_time_shape ? delay_us + (float)tshape.samples[i]
                                 : delay_us + 0.5f * grad_raster_us + (float)i * grad_raster_us;
        if (t <= t0_us)
            i0 = i;
        if (t >= t1_us && i1 < 0)
            i1 = i;
    }

    flat = 0;
    if (i0 >= 0 && i1 >= i0)
    {
        level = (float)wave.samples[i0];
        flat = 1;
        for (i = i0 + 1; i <= i1; ++i)
        {
            float d = (float)wave.samples[i] - level;
            if (d < -PULSEG_GRAD_FLAT_TOL || d > PULSEG_GRAD_FLAT_TOL)
            {
                flat = 0;
                break;
            }
        }
        if (flat)
            *level_out = level;
    }

    PULSEG_FREE(wave.samples);
    if (tshape.samples)
        PULSEG_FREE(tshape.samples);
    return flat;
}

/*
 * Can the excitation in block-table row @p bt_idx be moved by a carrier
 * offset alone?  See pulseg_block_initial_state::rf_grad_constant.
 *
 * Writes the per-axis normalised level on success.  Answers for ONE instance;
 * the caller AND-reduces over the instances of the position, because a
 * position that is flat in some instances and not in others has no single
 * gradient to offset against.
 */
static int rf_grad_constant_at(
    const pulseg_sequence_descriptor *desc,
    int bt_idx,
    float level_out[3])
{
    const pulseg_block_table_element *bte;
    const pulseg_rf_definition *rdef;
    int ax_raw[3];
    float t0_us, t1_us;
    int ax, rf_def_id, have_grad;

    level_out[0] = 0.0f;
    level_out[1] = 0.0f;
    level_out[2] = 0.0f;

    if (!desc || bt_idx < 0 || bt_idx >= desc->num_blocks)
        return 0;
    bte = &desc->block_table[bt_idx];

    /* A rotation extension rotates the trajectory inside the logical frame,
     * so the gradient this block actually plays is R times the one recorded
     * here.  Still flat -- R of a constant vector is constant -- but no longer
     * this vector, and an excitation moved with the wrong G lands in the wrong
     * place silently.  Refuse rather than guess. */
    if (bte->rotation_id != -1 && bte->rotation_id < desc->num_rotations &&
        !pulseg__is_identity3(desc->rotation_matrices[bte->rotation_id]))
        return 0;

    if (bte->rf_id < 0 || bte->rf_id >= desc->rf_table_size)
        return 0;
    rf_def_id = desc->rf_table[bte->rf_id].id;
    if (rf_def_id < 0 || rf_def_id >= desc->num_unique_rfs)
        return 0;
    rdef = &desc->rf_definitions[rf_def_id];

    /* The RF's active window, from the start of the block.  Nominal duration,
     * not the support of |B1| > 0: a pulse whose tails are zero still has to
     * see the same gradient there, because the window is what the offset is
     * claimed to be valid over. */
    t0_us = (float)rdef->delay;
    t1_us = (float)rdef->delay + rdef->stats.duration_us;
    if (t1_us <= t0_us)
        return 0;

    ax_raw[0] = bte->gx_id;
    ax_raw[1] = bte->gy_id;
    ax_raw[2] = bte->gz_id;

    have_grad = 0;
    for (ax = 0; ax < 3; ++ax)
    {
        if (ax_raw[ax] < 0 || ax_raw[ax] >= desc->grad_table_size)
            continue; /* no gradient on this axis: flat at zero, nothing to add */
        if (!grad_flat_over_window(desc, ax_raw[ax], t0_us, t1_us, &level_out[ax]))
            return 0;
        have_grad = 1;
    }

    /* No accompanying gradient at all is a nonselective pulse: nothing to
     * offset, and the caller has no correction to apply. */
    return have_grad;
}

/*
 * Does the matrix the interpreter has to program differ between two blocks?
 *
 * The interpreter's rotation is the *prescription* -- the FOV rotation, times
 * whatever prospective motion correction contributes.  Two per-block flags
 * change it:
 *
 *   - norot_flag    the block opts out of the prescribed FOV rotation
 *   - pmc_flag      the block opts out of motion correction
 *
 * `rotation_id` is deliberately NOT here.  A ROTATIONS extension rotates the
 * trajectory *within* the logical frame, so it is a property of the waveform
 * and not of the prescription.  It stays an event all the way through: the
 * `.seq` carries it, this collection carries it, and the `.pge` cache carries
 * it.  Each consumer materialises what it needs when it loads that cache --
 * the PSD the rotated gradient waveform, the recon the rotated trajectory and
 * phase modulation -- so a spoke that points elsewhere costs a wave pointer,
 * not a segment.  Splitting on it would put every radial spoke in a segment of
 * its own for nothing.
 *
 * PMC is here even though it looks static in the file, because its matrix
 * arrives from the tracking system during the scan: a segment mixing PMC-on
 * and PMC-off blocks has no single matrix it could be programmed with.
 */
static int rotation_state_differs(
    const pulseg_block_table_element *a,
    const pulseg_block_table_element *b)
{
    return a->norot_flag != b->norot_flag || a->pmc_flag != b->pmc_flag;
}

/*
 * Like find_segments_internal but resolves blocks through scan table
 * indirection.  scan_block_idx[pat_start + i] gives the block_table
 * index for position i within the pattern.
 *
 * start_block in returned segs is a SCAN TABLE position.
 */
static int find_segments_on_exec_stream(
    const pulseg_sequence_descriptor *desc,
    pulseg_virtual_segment *segs,
    int offset,
    pulseg_diagnostic *diag,
    const pulseg_opts *opts,
    const int *scan_block_idx,
    int pat_start,
    int pat_size)
{
    float max_allowed;
    int grad_ids[3];
    float phys_first, phys_last;
    float grad_last_cur[3], grad_first_next[3];
    int bt_idx, prev_bt;
    int *seg_starts = NULL;
    int *seg_sizes = NULL;
    int num_seg, seg_start;
    int state, cand_before_rf, saved_cand, has_saved_cand;
    int has_rf, has_adc, is_cand;
    int nb, n, i;

    (void)opts;
    max_allowed = SEG_ZERO_GRAD_THRESHOLD_HZ_PER_M;
    nb = pat_size;

    seg_starts = (int *)PULSEG_ALLOC(nb * sizeof(int));
    seg_sizes = (int *)PULSEG_ALLOC(nb * sizeof(int));
    if (!seg_starts || !seg_sizes)
    {
        if (seg_starts)
            PULSEG_FREE(seg_starts);
        if (seg_sizes)
            PULSEG_FREE(seg_sizes);
        if (diag)
            diag->code = PULSEG_ERR_ALLOC_FAILED;
        return 0;
    }

    /* first block gradient check */
    bt_idx = scan_block_idx[pat_start];
    grad_ids[0] = desc->block_table[bt_idx].gx_id;
    grad_ids[1] = desc->block_table[bt_idx].gy_id;
    grad_ids[2] = desc->block_table[bt_idx].gz_id;
    for (i = 0; i < 3; ++i)
    {
        if (grad_ids[i] < 0)
            continue;
        phys_first = pulseg__grad_boundary_first(desc, grad_ids[i]);
        if ((float)fabs(phys_first) > max_allowed)
        {
            if (diag)
            {
                diag->code = PULSEG_ERR_SEG_NONZERO_START_GRAD;
                pulseg__diag_printf(diag, " scan_pos=%d block=%d", pat_start, bt_idx);
                pulseg__diag_printf(diag, " channel=%d", i);
            }
            PULSEG_FREE(seg_starts);
            PULSEG_FREE(seg_sizes);
            return 0;
        }
    }

    /* last block gradient check */
    bt_idx = scan_block_idx[pat_start + nb - 1];
    grad_ids[0] = desc->block_table[bt_idx].gx_id;
    grad_ids[1] = desc->block_table[bt_idx].gy_id;
    grad_ids[2] = desc->block_table[bt_idx].gz_id;
    for (i = 0; i < 3; ++i)
    {
        if (grad_ids[i] < 0)
            continue;
        phys_last = pulseg__grad_boundary_last(desc, grad_ids[i]);
        if ((float)fabs(phys_last) > max_allowed)
        {
            if (diag)
            {
                diag->code = PULSEG_ERR_SEG_NONZERO_END_GRAD;
                pulseg__diag_printf(diag, " scan_pos=%d block=%d", pat_start + nb - 1, bt_idx);
                pulseg__diag_printf(diag, " channel=%d", i);
            }
            PULSEG_FREE(seg_starts);
            PULSEG_FREE(seg_sizes);
            return 0;
        }
    }

    /* state machine */
    num_seg = 0;
    seg_start = pat_start;
    state = SEGSTATE_SEEKING_FIRST_ADC;
    cand_before_rf = -1;
    saved_cand = -1;
    has_saved_cand = 0;

    for (n = pat_start; n < pat_start + nb; ++n)
    {
        bt_idx = scan_block_idx[n];
        is_cand = 0;
        if (n > pat_start)
        {
            prev_bt = scan_block_idx[n - 1];
            is_cand = 1;

            grad_ids[0] = desc->block_table[prev_bt].gx_id;
            grad_ids[1] = desc->block_table[prev_bt].gy_id;
            grad_ids[2] = desc->block_table[prev_bt].gz_id;
            for (i = 0; i < 3; ++i)
            {
                grad_last_cur[i] = 0.0f;
                if (grad_ids[i] >= 0)
                    grad_last_cur[i] = pulseg__grad_boundary_last(desc, grad_ids[i]);
            }

            grad_ids[0] = desc->block_table[bt_idx].gx_id;
            grad_ids[1] = desc->block_table[bt_idx].gy_id;
            grad_ids[2] = desc->block_table[bt_idx].gz_id;
            for (i = 0; i < 3; ++i)
            {
                grad_first_next[i] = 0.0f;
                if (grad_ids[i] >= 0)
                    grad_first_next[i] = pulseg__grad_boundary_first(desc, grad_ids[i]);
            }

            for (i = 0; i < 3; ++i)
            {
                if ((float)fabs(grad_last_cur[i]) > max_allowed ||
                    (float)fabs(grad_first_next[i]) > max_allowed)
                {
                    is_cand = 0;
                    break;
                }
            }
        }

        /* A rotation change forces a boundary.
         *
         * The interpreter programs one rotation matrix per segment, at its
         * start, so the segment is the granularity at which rotation can
         * change at all.  Splitting here is what makes that true, and is why
         * no mid-segment rotation update is needed: everything that used to
         * be committed block by block now falls on a segment start.
         *
         * The join still has to be at zero gradient -- a rotation remaps the
         * axes and cannot take effect under a live waveform.  A change that
         * lands anywhere else is a design error, and is reported rather than
         * absorbed by letting the block play in its predecessor's frame,
         * which is silently the wrong image.
         */
        if (n > pat_start &&
            rotation_state_differs(
                &desc->block_table[scan_block_idx[n - 1]],
                &desc->block_table[bt_idx]))
        {
            if (!is_cand)
            {
                if (diag)
                {
                    diag->code = PULSEG_ERR_SEG_ROTATION_MID_GRADIENT;
                    pulseg__diag_printf(diag, " scan_pos=%d block=%d", n, bt_idx);
                    pulseg__diag_printf(diag, " prev_block=%d", scan_block_idx[n - 1]);
                }
                PULSEG_FREE(seg_starts);
                PULSEG_FREE(seg_sizes);
                return 0;
            }
            if (n > seg_start)
            {
                seg_starts[num_seg] = seg_start;
                seg_sizes[num_seg] = n - seg_start;
                num_seg++;
                seg_start = n;
            }
            /* The candidates found so far belong to the segment just closed. */
            state = SEGSTATE_SEEKING_FIRST_ADC;
            cand_before_rf = -1;
            saved_cand = -1;
            has_saved_cand = 0;
        }

        has_rf = (desc->base_blocks[desc->block_table[bt_idx].id].rf_id >= 0);
        has_adc = (desc->block_table[bt_idx].adc_id >= 0);

        if (state == SEGSTATE_SEEKING_FIRST_ADC)
        {
            if (is_cand)
                saved_cand = n;
            if (has_rf)
            {
                cand_before_rf = saved_cand;
                saved_cand = -1;
            }
            if (has_adc)
            {
                if (cand_before_rf > seg_start)
                {
                    seg_starts[num_seg] = seg_start;
                    seg_sizes[num_seg] = cand_before_rf - seg_start;
                    num_seg++;
                    seg_start = cand_before_rf;
                }
                state = SEGSTATE_SEEKING_BOUNDARY;
                has_saved_cand = 0;
                saved_cand = -1;
            }
        }
        else if (state == SEGSTATE_SEEKING_BOUNDARY)
        {
            if (is_cand)
            {
                saved_cand = n;
                has_saved_cand = 1;
            }
            if (has_rf)
            {
                if (has_saved_cand)
                {
                    seg_starts[num_seg] = seg_start;
                    seg_sizes[num_seg] = saved_cand - seg_start;
                    num_seg++;
                    seg_start = saved_cand;
                    has_saved_cand = 0;
                    saved_cand = -1;
                }
                else
                {
                    state = SEGSTATE_OPTIMIZED_MODE;
                }
            }
        }
        /* SEGSTATE_OPTIMIZED_MODE: no action */
    }

    seg_starts[num_seg] = seg_start;
    seg_sizes[num_seg] = pat_start + nb - seg_start;
    num_seg++;

    for (i = 0; i < num_seg; ++i)
    {
        segs[offset + i].start_block = seg_starts[i];
        segs[offset + i].num_blocks = seg_sizes[i];
        segs[offset + i].unique_block_indices = NULL;
    }

    PULSEG_FREE(seg_starts);
    PULSEG_FREE(seg_sizes);
    return num_seg;
}

/* ================================================================== */
/*  Strip pure delays (scan table variant)                            */
/* ================================================================== */

static int strip_pure_delays_scan(
    const pulseg_virtual_segment *raw_segs,
    int num_raw,
    pulseg_virtual_segment *out,
    int max_out,
    const pulseg_block_table_element *bt,
    const int *scan_block_idx)
{
    int num_out = 0;
    int s, i, n_blk, bt_idx;
    int leading, trailing, core_start, core_end, core_size;
    const int *idx;

    for (s = 0; s < num_raw; ++s)
    {
        n_blk = raw_segs[s].num_blocks;
        idx = raw_segs[s].unique_block_indices;
        if (n_blk == 0 || !idx)
            continue;

        leading = 0;
        for (i = 0; i < n_blk; ++i)
        {
            bt_idx = scan_block_idx[raw_segs[s].start_block + i];
            if (bt[bt_idx].duration_us >= 0)
                leading++;
            else
                break;
        }
        trailing = 0;
        for (i = n_blk - 1; i >= leading; --i)
        {
            bt_idx = scan_block_idx[raw_segs[s].start_block + i];
            if (bt[bt_idx].duration_us >= 0)
                trailing++;
            else
                break;
        }
        core_start = leading;
        core_end = n_blk - trailing;

        for (i = 0; i < leading; ++i)
        {
            if (num_out >= max_out)
                return -1;
            out[num_out].start_block = raw_segs[s].start_block + i;
            out[num_out].num_blocks = 1;
            out[num_out].unique_block_indices = (int *)PULSEG_ALLOC(sizeof(int));
            if (!out[num_out].unique_block_indices)
                return -1;
            out[num_out].unique_block_indices[0] = idx[i];
            num_out++;
        }
        if (core_end > core_start)
        {
            core_size = core_end - core_start;
            if (num_out >= max_out)
                return -1;
            out[num_out].start_block = raw_segs[s].start_block + core_start;
            out[num_out].num_blocks = core_size;
            out[num_out].unique_block_indices = (int *)PULSEG_ALLOC(core_size * sizeof(int));
            if (!out[num_out].unique_block_indices)
                return -1;
            for (i = 0; i < core_size; ++i)
                out[num_out].unique_block_indices[i] = idx[core_start + i];
            num_out++;
        }
        for (i = 0; i < trailing; ++i)
        {
            if (num_out >= max_out)
                return -1;
            out[num_out].start_block = raw_segs[s].start_block + core_end + i;
            out[num_out].num_blocks = 1;
            out[num_out].unique_block_indices = (int *)PULSEG_ALLOC(sizeof(int));
            if (!out[num_out].unique_block_indices)
                return -1;
            out[num_out].unique_block_indices[0] = idx[core_end + i];
            num_out++;
        }
    }
    return num_out;
}

/* ================================================================== */
/*  Scan-table boundary pre-check for segmentation retry              */
/* ================================================================== */

/*
 * Check whether the first scan-table position's start-gradients and
 * the last scan-table position's end-gradients are within max_allowed,
 * using each block's *actual* shot (resolved through scan_block_idx).
 *
 * Returns 1 if OK, 0 if any axis violates.
 * Arguments are scan-table positions.
 */
static int scan_boundary_gradients_ok(
    const pulseg_sequence_descriptor *desc,
    const int *scan_block_idx,
    int first_scan_pos,
    int last_scan_pos,
    float max_allowed)
{
    int ch, gid, bt;
    float pv;

    bt = scan_block_idx[first_scan_pos];
    for (ch = 0; ch < 3; ++ch)
    {
        gid = (ch == 0) ? desc->block_table[bt].gx_id
            : (ch == 1) ? desc->block_table[bt].gy_id
                        : desc->block_table[bt].gz_id;
        if (gid < 0)
            continue;
        pv = pulseg__grad_boundary_first(desc, gid);
        if ((float)fabs(pv) > max_allowed)
            return 0;
    }

    bt = scan_block_idx[last_scan_pos];
    for (ch = 0; ch < 3; ++ch)
    {
        gid = (ch == 0) ? desc->block_table[bt].gx_id
            : (ch == 1) ? desc->block_table[bt].gy_id
                        : desc->block_table[bt].gz_id;
        if (gid < 0)
            continue;
        pv = pulseg__grad_boundary_last(desc, gid);
        if ((float)fabs(pv) > max_allowed)
            return 0;
    }

    return 1;
}

/* ================================================================== */
/*  Scan-table-based segment detection                                */
/* ================================================================== */

int pulseg__get_exec_stream_segments(
    pulseg_sequence_descriptor *desc,
    pulseg_diagnostic *diag,
    const pulseg_opts *opts)
{
    pulseg_diagnostic local_diag;
    int *scan_pat = NULL;
    int *pattern_seg_id = NULL;
    float *max_energy = NULL;
    int **delay_baseline = NULL;
    pulseg_virtual_segment *raw_segs = NULL;
    pulseg_virtual_segment *exp_segs = NULL;
    pulseg_virtual_segment *uniq_segs = NULL;
    int scan_len, pass_size;
    int num_raw, num_total, num_unique;
    int n_prep_raw, n_main_raw, n_cool_raw;
    int n_prep, n_main, n_cool;
    int n, b, i, found, offset;
    int num_raw_alloc, num_exp_alloc;
    int seg_result, max_expanded;
    int nb, unique_idx, blk_tab_idx, blk_def_id;
    int ax_grad_ids[3], ax_def_ids[3], ax;
    float inst_energy, e, amp;
    const pulseg_block_table_element *bte;
    const pulseg_base_block *bdef;
    int mult, max_mult;
    int tr_size, region_start, region_size;
    int main_region_start, main_region_size;
    float max_allowed;

    if (!diag)
    {
        pulseg_diagnostic_init(&local_diag);
        diag = &local_diag;
    }
    else
        pulseg_diagnostic_init(diag);

    if (!desc || !opts)
    {
        diag->code = PULSEG_ERR_NULL_POINTER;
        return 0;
    }
    if (desc->exec_stream_len <= 0 || !desc->exec_stream_block_idx)
    {
        diag->code = PULSEG_ERR_INVALID_ARGUMENT;
        return 0;
    }

    scan_len = desc->exec_stream_len;
    pass_size = scan_len;

    tr_size = desc->tr_descriptor.tr_size;
    max_allowed = SEG_ZERO_GRAD_THRESHOLD_HZ_PER_M;

    /* max_mult: maximum number of TRs a segmentation retry can absorb.
     * The entire stream is the upper bound. */
    max_mult = (tr_size > 0) ? (pass_size / tr_size) : 1;
    if (max_mult < 1)
        max_mult = 1;

    num_raw = 0;
    num_total = 0;
    num_unique = 0;
    n_prep_raw = 0;
    n_main_raw = 0;
    n_cool_raw = 0;
    n_prep = 0;
    n_main = 0;
    n_cool = 0;
    num_raw_alloc = 0;
    num_exp_alloc = 0;
    main_region_start = 0;
    main_region_size = 0;

    /* ---- 1. Map the scan table to a block-def-ID pattern ---- */
    scan_pat = (int *)PULSEG_ALLOC((size_t)pass_size * sizeof(int));
    if (!scan_pat)
    {
        diag->code = PULSEG_ERR_ALLOC_FAILED;
        return 0;
    }
    for (n = 0; n < pass_size; ++n)
        scan_pat[n] = desc->block_table[pulseg__exec_block_idx(desc, n)].id;

    raw_segs =
        (pulseg_virtual_segment *)PULSEG_ALLOC((size_t)pass_size * sizeof(pulseg_virtual_segment));
    if (!raw_segs)
    {
        PULSEG_FREE(scan_pat);
        diag->code = PULSEG_ERR_ALLOC_FAILED;
        return 0;
    }

    /*
     * Segmentation over the whole stream, in TR multiples.
     * Each retry starts with a fast boundary gradient pre-check.
     */

    /* ---- 2. TR-window segmentation ---- */
    {
        region_start = 0;
        seg_result = 0;
        for (mult = 1; mult <= max_mult; ++mult)
        {
            region_size = mult * tr_size;
            if (region_start + region_size > pass_size)
                break;
            if (!scan_boundary_gradients_ok(
                    desc,
                    desc->exec_stream_block_idx,
                    region_start,
                    region_start + region_size - 1,
                    max_allowed))
                continue;
            pulseg_diagnostic_init(diag);
            seg_result = find_segments_on_exec_stream(
                desc,
                raw_segs,
                num_raw,
                diag,
                opts,
                desc->exec_stream_block_idx,
                region_start,
                region_size);
            if (seg_result > 0)
                break;
            if (diag->code != PULSEG_ERR_SEG_NONZERO_START_GRAD &&
                diag->code != PULSEG_ERR_SEG_NONZERO_END_GRAD)
                break;
        }
        if (seg_result == 0 && PULSEG_FAILED(diag->code))
        {
            PULSEG_FREE(scan_pat);
            PULSEG_FREE(raw_segs);
            return 0;
        }
        n_main_raw = seg_result;
        num_raw += n_main_raw;

        /* Record main region geometry for tiling in step 11 */
        main_region_start = region_start;
        main_region_size = region_size;
    }

    /* ---- 2d. If all sections produced nothing, run a single
     *          find_segments over [0, pass_size) WITHOUT boundary
     *          pre-check so the real error code propagates.  This also
     *          serves as the ultimate fallback when cooldown has 0
     *          blocks and thus its branch was skipped entirely. ---- */
    if (num_raw == 0)
    {
        pulseg_diagnostic_init(diag);
        seg_result = find_segments_on_exec_stream(
            desc,
            raw_segs,
            0,
            diag,
            opts,
            desc->exec_stream_block_idx,
            0,
            pass_size);
        if (seg_result > 0)
        {
            n_prep_raw = 0;
            n_main_raw = seg_result;
            n_cool_raw = 0;
            num_raw = seg_result;
        }
        else
        {
            /* Propagate the actual error from find_segments */
            if (!PULSEG_FAILED(diag->code))
                diag->code = PULSEG_ERR_SEG_NO_SEGMENTS_FOUND;
            PULSEG_FREE(scan_pat);
            PULSEG_FREE(raw_segs);
            return 0;
        }
    }

    /* ---- 3. Populate unique_block_indices ---- */
    for (n = 0; n < num_raw; ++n)
    {
        raw_segs[n].unique_block_indices =
            (int *)PULSEG_ALLOC(raw_segs[n].num_blocks * sizeof(int));
        if (!raw_segs[n].unique_block_indices)
        {
            diag->code = PULSEG_ERR_ALLOC_FAILED;
            num_raw_alloc = n;
            goto scan_seg_fail;
        }
        for (i = 0; i < raw_segs[n].num_blocks; ++i)
            raw_segs[n].unique_block_indices[i] = scan_pat[raw_segs[n].start_block + i];
    }
    num_raw_alloc = num_raw;

    /* ---- 4. Strip pure delays (per section) ---- */
    max_expanded = pass_size;
    exp_segs = (pulseg_virtual_segment *)PULSEG_ALLOC(
        (size_t)max_expanded * sizeof(pulseg_virtual_segment));
    if (!exp_segs)
    {
        diag->code = PULSEG_ERR_ALLOC_FAILED;
        goto scan_seg_fail;
    }

    offset = 0;
    if (n_prep_raw > 0)
    {
        n_prep = strip_pure_delays_scan(
            raw_segs,
            n_prep_raw,
            exp_segs + offset,
            max_expanded - offset,
            desc->block_table,
            desc->exec_stream_block_idx);
        if (n_prep < 0)
        {
            diag->code = PULSEG_ERR_ALLOC_FAILED;
            goto scan_seg_fail;
        }
        offset += n_prep;
    }

    if (n_main_raw > 0)
    {
        n_main = strip_pure_delays_scan(
            raw_segs + n_prep_raw,
            n_main_raw,
            exp_segs + offset,
            max_expanded - offset,
            desc->block_table,
            desc->exec_stream_block_idx);
        if (n_main < 0)
        {
            diag->code = PULSEG_ERR_ALLOC_FAILED;
            goto scan_seg_fail;
        }
        offset += n_main;
    }

    if (n_cool_raw > 0)
    {
        n_cool = strip_pure_delays_scan(
            raw_segs + n_prep_raw + n_main_raw,
            n_cool_raw,
            exp_segs + offset,
            max_expanded - offset,
            desc->block_table,
            desc->exec_stream_block_idx);
        if (n_cool < 0)
        {
            diag->code = PULSEG_ERR_ALLOC_FAILED;
            goto scan_seg_fail;
        }
        offset += n_cool;
    }

    num_total = n_prep + n_main + n_cool;
    num_exp_alloc = num_total;

    /* Free raw segments */
    for (n = 0; n < num_raw_alloc; ++n)
        PULSEG_FREE(raw_segs[n].unique_block_indices);
    PULSEG_FREE(raw_segs);
    raw_segs = NULL;
    num_raw_alloc = 0;

    if (num_total == 0)
    {
        diag->code = PULSEG_ERR_SEG_NO_SEGMENTS_FOUND;
        goto scan_seg_fail;
    }

    /* ---- 5. NAV-aware split and merge (per section, PMC only) ---- */
    if (desc->enable_pmc)
    {
        pulseg_virtual_segment *nav_segs;
        int nav_total = 0, r;

        nav_segs = (pulseg_virtual_segment *)PULSEG_ALLOC(
            (size_t)max_expanded * sizeof(pulseg_virtual_segment));
        if (!nav_segs)
        {
            diag->code = PULSEG_ERR_ALLOC_FAILED;
            goto scan_seg_fail;
        }

        if (n_prep > 0)
        {
            r = nav_split_merge(
                exp_segs,
                n_prep,
                nav_segs + nav_total,
                max_expanded - nav_total,
                desc->block_table,
                desc->exec_stream_block_idx);
            if (r < 0)
            {
                for (n = 0; n < nav_total; ++n)
                    if (nav_segs[n].unique_block_indices)
                        PULSEG_FREE(nav_segs[n].unique_block_indices);
                PULSEG_FREE(nav_segs);
                diag->code = PULSEG_ERR_ALLOC_FAILED;
                goto scan_seg_fail;
            }
            n_prep = r;
            nav_total += r;
        }

        r = nav_split_merge(
            exp_segs + (num_total - n_cool - n_main),
            n_main,
            nav_segs + nav_total,
            max_expanded - nav_total,
            desc->block_table,
            desc->exec_stream_block_idx);
        if (r < 0)
        {
            for (n = 0; n < nav_total; ++n)
                if (nav_segs[n].unique_block_indices)
                    PULSEG_FREE(nav_segs[n].unique_block_indices);
            PULSEG_FREE(nav_segs);
            diag->code = PULSEG_ERR_ALLOC_FAILED;
            goto scan_seg_fail;
        }
        n_main = r;
        nav_total += r;

        if (n_cool > 0)
        {
            r = nav_split_merge(
                exp_segs + (num_total - n_cool),
                n_cool,
                nav_segs + nav_total,
                max_expanded - nav_total,
                desc->block_table,
                desc->exec_stream_block_idx);
            if (r < 0)
            {
                for (n = 0; n < nav_total; ++n)
                    if (nav_segs[n].unique_block_indices)
                        PULSEG_FREE(nav_segs[n].unique_block_indices);
                PULSEG_FREE(nav_segs);
                diag->code = PULSEG_ERR_ALLOC_FAILED;
                goto scan_seg_fail;
            }
            n_cool = r;
            nav_total += r;
        }

        /* Replace exp_segs with nav_segs */
        PULSEG_FREE(exp_segs);
        exp_segs = nav_segs;
        num_total = nav_total;
        num_exp_alloc = nav_total;
    }

    /* ---- 6. Segment table ---- */
    desc->segment_table.num_main_segments = num_total;
    desc->segment_table.main_segment_table =
        (num_total > 0) ? (int *)PULSEG_ALLOC(num_total * sizeof(int)) : NULL;

    uniq_segs =
        (pulseg_virtual_segment *)PULSEG_ALLOC((size_t)num_total * sizeof(pulseg_virtual_segment));
    if (!uniq_segs)
    {
        diag->code = PULSEG_ERR_ALLOC_FAILED;
        goto scan_seg_fail;
    }

    /* ---- 7. Deduplicate segments across all sections ---- */
    num_unique = 0;

    for (n = 0; n < num_total; ++n)
    {
        int n_is_pure_delay = is_single_pure_delay_segment_scan(
            &exp_segs[n],
            desc->block_table,
            desc->exec_stream_block_idx);

        /* Find existing match by UBI, except one-block pure delays:
         * those are definition-equivalent regardless of delay duration. */
        found = -1;
        for (i = 0; i < num_unique; ++i)
        {
            int i_is_pure_delay = is_single_pure_delay_segment_scan(
                &uniq_segs[i],
                desc->block_table,
                desc->exec_stream_block_idx);

            if (n_is_pure_delay && i_is_pure_delay)
            {
                found = i;
                break;
            }

            if (!n_is_pure_delay && !i_is_pure_delay &&
                exp_segs[n].num_blocks == uniq_segs[i].num_blocks)
            {
                if (segs_delay_flex_equal(
                        desc,
                        desc->exec_stream_block_idx,
                        &exp_segs[n],
                        &uniq_segs[i]))
                {
                    found = i;
                    break;
                }

                if (segments_structurally_equal(desc, &exp_segs[n], &uniq_segs[i]))
                {
                    found = i;
                    break;
                }
            }
        }
        if (found == -1)
        {
            uniq_segs[num_unique].num_blocks = exp_segs[n].num_blocks;
            uniq_segs[num_unique].start_block = exp_segs[n].start_block;
            uniq_segs[num_unique].unique_block_indices =
                (int *)PULSEG_ALLOC(exp_segs[n].num_blocks * sizeof(int));
            if (!uniq_segs[num_unique].unique_block_indices)
            {
                diag->code = PULSEG_ERR_ALLOC_FAILED;
                goto scan_seg_fail;
            }
            for (i = 0; i < exp_segs[n].num_blocks; ++i)
                uniq_segs[num_unique].unique_block_indices[i] = exp_segs[n].unique_block_indices[i];
            found = num_unique;
            num_unique++;
        }

        desc->segment_table.main_segment_table[n] = found;
    }

    desc->segment_table.num_unique_segments = num_unique;
    desc->num_unique_segments = num_unique;

    /* ---- 8. Transfer segment definitions ---- */
    desc->segment_definitions =
        (pulseg_virtual_segment *)PULSEG_ALLOC((size_t)num_unique * sizeof(pulseg_virtual_segment));
    if (!desc->segment_definitions)
    {
        diag->code = PULSEG_ERR_ALLOC_FAILED;
        goto scan_seg_fail;
    }
    for (i = 0; i < num_unique; ++i)
    {
        desc->segment_definitions[i] = uniq_segs[i];
        /* Convert start_block from scan table pos to block_table index */
        desc->segment_definitions[i].start_block =
            pulseg__exec_block_idx(desc, uniq_segs[i].start_block);
    }
    PULSEG_FREE(uniq_segs);
    uniq_segs = NULL;

    /* ---- 9. Per-block flags ---- */
    for (i = 0; i < num_unique; ++i)
    {
        nb = desc->segment_definitions[i].num_blocks;
        desc->segment_definitions[i].has_digitalout = (int *)PULSEG_ALLOC(nb * sizeof(int));
        desc->segment_definitions[i].has_rotation = (int *)PULSEG_ALLOC(nb * sizeof(int));
        desc->segment_definitions[i].norot_flag = (int *)PULSEG_ALLOC(nb * sizeof(int));
        desc->segment_definitions[i].nopos_flag = (int *)PULSEG_ALLOC(nb * sizeof(int));
        desc->segment_definitions[i].has_adc = (int *)PULSEG_ALLOC(nb * sizeof(int));
        desc->segment_definitions[i].is_dynamic_delay = (int *)PULSEG_ALLOC(nb * sizeof(int));
        desc->segment_definitions[i].initial_states = (pulseg_block_initial_state *)PULSEG_ALLOC(
            (size_t)nb * sizeof(pulseg_block_initial_state));
        if (!desc->segment_definitions[i].has_digitalout ||
            !desc->segment_definitions[i].has_rotation ||
            !desc->segment_definitions[i].norot_flag || !desc->segment_definitions[i].nopos_flag ||
            !desc->segment_definitions[i].has_adc ||
            !desc->segment_definitions[i].is_dynamic_delay ||
            !desc->segment_definitions[i].initial_states)
        {
            diag->code = PULSEG_ERR_ALLOC_FAILED;
            goto scan_seg_fail;
        }
        for (n = 0; n < nb; ++n)
        {
            desc->segment_definitions[i].has_digitalout[n] = 0;
            desc->segment_definitions[i].has_rotation[n] = 0;
            desc->segment_definitions[i].norot_flag[n] = 0;
            desc->segment_definitions[i].nopos_flag[n] = 0;
            desc->segment_definitions[i].has_adc[n] = 0;
            desc->segment_definitions[i].is_dynamic_delay[n] = 0;
            {
                pulseg_block_initial_state init = PULSEG_BLOCK_INITIAL_STATE_INIT;
                desc->segment_definitions[i].initial_states[n] = init;
            }
        }
        desc->segment_definitions[i].trigger_id = -1;
        /* is_nav is only assigned in the enable_pmc tagging block below; give
         * it a definite value here so non-PMC sequences don't carry
         * uninitialised heap data (which breaks cross-subsequence segment
         * dedup and any other consumer that reads it). */
        desc->segment_definitions[i].is_nav = 0;
    }

    /* ---- 10. Walk expanded segments, populate flags + max energy ---- */
    max_energy = (float *)PULSEG_ALLOC((size_t)num_unique * sizeof(float));
    if (!max_energy)
    {
        diag->code = PULSEG_ERR_ALLOC_FAILED;
        goto scan_seg_fail;
    }
    for (i = 0; i < num_unique; ++i)
    {
        max_energy[i] = -1.0f;
        desc->segment_definitions[i].max_energy_start_block = -1;
    }

    for (n = 0; n < num_total; ++n)
    {
        unique_idx = desc->segment_table.main_segment_table[n];

        inst_energy = 0.0f;

        for (b = 0; b < exp_segs[n].num_blocks; ++b)
        {
            blk_tab_idx = pulseg__exec_block_idx(desc, exp_segs[n].start_block + b);
            bte = &desc->block_table[blk_tab_idx];
            blk_def_id = bte->id;
            bdef = &desc->base_blocks[blk_def_id];

            /* Classify trigger: OUTPUT → block-level digitalout,
             *                    INPUT  → segment-level trigger */
            if (bte->digitalout_id != -1 && bte->digitalout_id < desc->num_triggers)
            {
                const pulseq_trigger_event *te = &desc->trigger_events[bte->digitalout_id];
                if (te->trigger_type == PULSEQ_TRIGGER_TYPE_OUTPUT)
                {
                    desc->segment_definitions[unique_idx].has_digitalout[b] = 1;
                }
                else if (te->trigger_type == PULSEQ_TRIGGER_TYPE_INPUT)
                {
                    int prev = desc->segment_definitions[unique_idx].trigger_id;
                    if (prev >= 0 && prev != bte->digitalout_id)
                    {
                        diag->code = PULSEG_ERR_SEG_MULTIPLE_PHYSIO_TRIGGERS;
                        goto scan_seg_fail;
                    }
                    desc->segment_definitions[unique_idx].trigger_id = bte->digitalout_id;
                }
            }
            if (bte->rotation_id != -1 && bte->rotation_id < desc->num_rotations &&
                !pulseg__is_identity3(desc->rotation_matrices[bte->rotation_id]))
                desc->segment_definitions[unique_idx].has_rotation[b] = 1;
            if (bte->norot_flag)
                desc->segment_definitions[unique_idx].norot_flag[b] = 1;
            if (bte->nopos_flag)
                desc->segment_definitions[unique_idx].nopos_flag[b] = 1;

            ax_grad_ids[0] = bte->gx_id;
            ax_grad_ids[1] = bte->gy_id;
            ax_grad_ids[2] = bte->gz_id;
            ax_def_ids[0] = bdef->gx_id;
            ax_def_ids[1] = bdef->gy_id;
            ax_def_ids[2] = bdef->gz_id;

            for (ax = 0; ax < 3; ++ax)
            {
                if (ax_grad_ids[ax] >= 0 && ax_grad_ids[ax] < desc->grad_table_size &&
                    ax_def_ids[ax] >= 0 && ax_def_ids[ax] < desc->num_unique_grads)
                {
                    amp = desc->grad_table[ax_grad_ids[ax]].amplitude;
                    /* The definition's own worst instance by energy; see
                     * pulseg_grad_representative. */
                    e = desc->grad_definitions[ax_def_ids[ax]].heat.energy;
                    inst_energy += e * amp * amp;
                }
            }
        }

        if (inst_energy > max_energy[unique_idx])
        {
            max_energy[unique_idx] = inst_energy;
            /* Store scan-table start index of the representative instance.
             * In average-expanded passes, consecutive local block positions do
             * not map to consecutive block_table indices, so getters must
             * resolve through exec_stream_block_idx[start + local_blk]. */
            desc->segment_definitions[unique_idx].max_energy_start_block = exp_segs[n].start_block;
        }
    }

    /* max_energy freed after step 11b rescan */

    /* ---- tag segments as NAV; verify at most 1 unique NAV ---- */
    if (desc->enable_pmc)
    {
        int nav_count = 0;
        for (i = 0; i < num_unique; ++i)
        {
            /* start_block already resolved to block_table index in step 8 */
            int bt0 = desc->segment_definitions[i].start_block;
            desc->segment_definitions[i].is_nav = (desc->block_table[bt0].nav_flag) ? 1 : 0;
            if (desc->segment_definitions[i].is_nav)
                nav_count++;
        }
        if (nav_count > 1)
        {
            diag->code = PULSEG_ERR_SEG_MULTIPLE_NAV_SEGMENTS;
            goto scan_seg_fail;
        }
    }

    /* ---- 11. Build pattern_seg_id and fill exec_stream_seg_id ---- */
    pattern_seg_id = (int *)PULSEG_ALLOC((size_t)pass_size * sizeof(int));
    if (!pattern_seg_id)
    {
        diag->code = PULSEG_ERR_ALLOC_FAILED;
        goto scan_seg_fail;
    }
    for (n = 0; n < pass_size; ++n)
        pattern_seg_id[n] = -1;

    /* Walk the expanded segments and assign each position */
    for (n = 0; n < num_total; ++n)
    {
        unique_idx = desc->segment_table.main_segment_table[n];

        for (b = 0; b < exp_segs[n].num_blocks; ++b)
        {
            i = exp_segs[n].start_block + b;
            if (i >= 0 && i < pass_size)
                pattern_seg_id[i] = unique_idx;
        }
    }

    /* ---- 11b. Tile main-section seg_id across all main TRs ---------
     *
     * Steps 2a-2c only segment a single instance of each section.
     * The main section's pattern (covering main_region_size blocks
     * starting at main_region_start) repeats with the same period
     * for every subsequent main TR.  Fill remaining -1 gaps by
     * tiling; positions already assigned by prep/cooldown are kept. */
    if (main_region_size > 0)
    {
        int src_start = main_region_start;
        int src_len = main_region_size;

        for (i = src_start + src_len; i < pass_size; ++i)
        {
            if (pattern_seg_id[i] == -1)
            {
                int src = src_start + ((i - src_start) % src_len);
                if (pattern_seg_id[src] >= 0)
                    pattern_seg_id[i] = pattern_seg_id[src];
            }
        }
    }

    /* Tile first-pass pattern across full scan table (all passes) */
    for (n = 0; n < scan_len; ++n)
        desc->exec_stream_seg_id[n] = pattern_seg_id[n % pass_size];

    PULSEG_FREE(pattern_seg_id);
    pattern_seg_id = NULL;

    /* seg_id is final here -- run-length encode it for pulseg__exec_seg_id. */
    {
        int rc = pulseg__build_seg_runs(desc);
        if (PULSEG_FAILED(rc))
        {
            diag->code = rc;
            goto scan_seg_fail;
        }
    }

    /* ---- 11c. Rescan full scan table for max-energy per segment ----
     *
     * Step 10 only saw one instance per segment (the first one).
     * Now that exec_stream_seg_id covers all tiled repetitions we
     * can walk the entire scan table, group consecutive blocks into
     * segment instances, compute their energy, and update
     * max_energy_start_block when a higher-energy repetition is found. */
    for (i = 0; i < num_unique; ++i)
        max_energy[i] = -1.0f;

    for (n = 0; n < scan_len; /* advance inside */)
    {
        int seg_id = pulseg__exec_seg_id(desc, n);
        if (seg_id < 0 || seg_id >= num_unique)
        {
            ++n;
            continue;
        }

        nb = desc->segment_definitions[seg_id].num_blocks;
        if (n + nb > scan_len)
        {
            ++n;
            continue;
        }

        /* Verify all nb positions belong to the same segment */
        {
            int ok = 1;
            for (b = 1; b < nb; ++b)
            {
                if (pulseg__exec_seg_id(desc, n + b) != seg_id)
                {
                    ok = 0;
                    break;
                }
            }
            if (!ok)
            {
                ++n;
                continue;
            }
        }

        /* Compute energy for this instance */
        inst_energy = 0.0f;
        for (b = 0; b < nb; ++b)
        {
            blk_tab_idx = pulseg__exec_block_idx(desc, n + b);
            bte = &desc->block_table[blk_tab_idx];
            bdef = &desc->base_blocks[bte->id];

            ax_grad_ids[0] = bte->gx_id;
            ax_grad_ids[1] = bte->gy_id;
            ax_grad_ids[2] = bte->gz_id;
            ax_def_ids[0] = bdef->gx_id;
            ax_def_ids[1] = bdef->gy_id;
            ax_def_ids[2] = bdef->gz_id;

            for (ax = 0; ax < 3; ++ax)
            {
                if (ax_grad_ids[ax] >= 0 && ax_grad_ids[ax] < desc->grad_table_size &&
                    ax_def_ids[ax] >= 0 && ax_def_ids[ax] < desc->num_unique_grads)
                {
                    amp = desc->grad_table[ax_grad_ids[ax]].amplitude;
                    /* The definition's own worst instance by energy; see
                     * pulseg_grad_representative. */
                    e = desc->grad_definitions[ax_def_ids[ax]].heat.energy;
                    inst_energy += e * amp * amp;
                }
            }
        }

        if (inst_energy > max_energy[seg_id])
        {
            max_energy[seg_id] = inst_energy;
            /* Same semantics as step 10: scan-table start index. */
            desc->segment_definitions[seg_id].max_energy_start_block = n;
        }

        n += nb;
    }

    PULSEG_FREE(max_energy);
    max_energy = NULL;

    /* ---- 11d. OR-reduce per-block flags across ALL segment instances ----
     *
     * Step 10 only populated flags from the first expanded instance of
     * each segment.  Flags like has_digitalout, has_rotation, norot,
     * and nopos must reflect ANY instance, not just the first one
     * that happened to be expanded.
     *
     * Repetitions (num_averages > 1) share the same block_table
     * entries.  For multi-pass sequences the pass dimension is the
     * outer loop (passes are interleaved), but all passes have the
     * same structure by definition.  Walking just the first pass
     * (0 .. pass_size-1) is therefore sufficient.                     */
    for (i = 0; i < num_unique; ++i)
    {
        nb = desc->segment_definitions[i].num_blocks;
        for (n = 0; n < nb; ++n)
        {
            desc->segment_definitions[i].has_digitalout[n] = 0;
            desc->segment_definitions[i].has_rotation[n] = 0;
            desc->segment_definitions[i].norot_flag[n] = 0;
            desc->segment_definitions[i].nopos_flag[n] = 0;
            desc->segment_definitions[i].has_adc[n] = 0;
            desc->segment_definitions[i].is_dynamic_delay[n] = 0;
            /* -1 = no instance seen yet; the walk below AND-reduces onto it. */
            desc->segment_definitions[i].initial_states[n].rf_grad_constant = -1;
        }
        desc->segment_definitions[i].trigger_id = -1;
    }

    /* Per-(segment, block-position) baseline duration for adjustable pure
     * delays, used below to detect cross-instance variance.  -2 = not yet
     * observed; any other value is the first instance's duration_us. */
    delay_baseline = (int **)PULSEG_ALLOC((size_t)num_unique * sizeof(int *));
    if (!delay_baseline)
    {
        diag->code = PULSEG_ERR_ALLOC_FAILED;
        goto scan_seg_fail;
    }
    for (i = 0; i < num_unique; ++i)
        delay_baseline[i] = NULL;
    for (i = 0; i < num_unique; ++i)
    {
        nb = desc->segment_definitions[i].num_blocks;
        delay_baseline[i] = (int *)PULSEG_ALLOC((size_t)nb * sizeof(int));
        if (!delay_baseline[i])
        {
            diag->code = PULSEG_ERR_ALLOC_FAILED;
            goto scan_seg_fail;
        }
        for (n = 0; n < nb; ++n)
            delay_baseline[i][n] = -2;
    }

    for (n = 0; n < pass_size; /* advance inside */)
    {
        int seg_id = pulseg__exec_seg_id(desc, n);
        if (seg_id < 0 || seg_id >= num_unique)
        {
            ++n;
            continue;
        }

        nb = desc->segment_definitions[seg_id].num_blocks;
        if (n + nb > pass_size)
        {
            ++n;
            continue;
        }

        /* Verify contiguous segment instance */
        {
            int ok = 1;
            for (b = 1; b < nb; ++b)
            {
                if (pulseg__exec_seg_id(desc, n + b) != seg_id)
                {
                    ok = 0;
                    break;
                }
            }
            if (!ok)
            {
                ++n;
                continue;
            }
        }

        for (b = 0; b < nb; ++b)
        {
            bte = &desc->block_table[pulseg__exec_block_idx(desc, n + b)];
            bdef = &desc->base_blocks[bte->id];

            /* digitalout / trigger */
            if (bte->digitalout_id != -1 && bte->digitalout_id < desc->num_triggers)
            {
                const pulseq_trigger_event *te = &desc->trigger_events[bte->digitalout_id];
                if (te->trigger_type == PULSEQ_TRIGGER_TYPE_OUTPUT)
                {
                    desc->segment_definitions[seg_id].has_digitalout[b] = 1;
                }
                else if (te->trigger_type == PULSEQ_TRIGGER_TYPE_INPUT)
                {
                    int prev = desc->segment_definitions[seg_id].trigger_id;
                    if (prev < 0)
                        desc->segment_definitions[seg_id].trigger_id = bte->digitalout_id;
                }
            }
            if (bte->rotation_id != -1 && bte->rotation_id < desc->num_rotations &&
                !pulseg__is_identity3(desc->rotation_matrices[bte->rotation_id]))
                desc->segment_definitions[seg_id].has_rotation[b] = 1;
            if (bte->norot_flag)
                desc->segment_definitions[seg_id].norot_flag[b] = 1;
            if (bte->nopos_flag)
                desc->segment_definitions[seg_id].nopos_flag[b] = 1;

            /* has_adc: OR-reduce — true if any instance at this position has ADC */
            if (bte->adc_id >= 0)
                desc->segment_definitions[seg_id].has_adc[b] = 1;

            /* is_dynamic_delay: OR-reduce — true iff this position is an
             * adjustable pure delay AND its duration differs from the first
             * instance observed at this position.  A position whose duration
             * never varies stays "static": no runtime setperiod wait, no
             * interior SSP packet -- represented purely by block position. */
            if (is_adjustable_delay_at(desc, desc->exec_stream_block_idx, n + b))
            {
                if (delay_baseline[seg_id][b] == -2)
                    delay_baseline[seg_id][b] = bte->duration_us;
                else if (delay_baseline[seg_id][b] != bte->duration_us)
                    desc->segment_definitions[seg_id].is_dynamic_delay[b] = 1;
            }

            /* rf_grad_constant: AND-reduce, and the level has to agree too.
             * A position whose instances disagree -- flat in one, ramping in
             * another, or flat at different levels -- has no single gradient
             * for the interpreter to offset against, so it gets nothing. */
            {
                pulseg_block_initial_state *st =
                    &desc->segment_definitions[seg_id].initial_states[b];
                float lvl[3];
                int ok = rf_grad_constant_at(desc, pulseg__exec_block_idx(desc, n + b), lvl);

                if (st->rf_grad_constant == -1)
                {
                    st->rf_grad_constant = ok;
                    st->rf_grad_level[0] = lvl[0];
                    st->rf_grad_level[1] = lvl[1];
                    st->rf_grad_level[2] = lvl[2];
                }
                else if (!ok)
                {
                    st->rf_grad_constant = 0;
                }
                else if (st->rf_grad_constant)
                {
                    int ax;
                    for (ax = 0; ax < 3; ++ax)
                    {
                        float d = lvl[ax] - st->rf_grad_level[ax];
                        if (d < -PULSEG_GRAD_FLAT_TOL || d > PULSEG_GRAD_FLAT_TOL)
                        {
                            st->rf_grad_constant = 0;
                            break;
                        }
                    }
                }
            }
        }

        n += nb;
    }

    /* Positions no instance of the walk reached keep the "unseen" sentinel;
     * the field is public from here on, so settle it to 0. */
    for (i = 0; i < num_unique; ++i)
    {
        nb = desc->segment_definitions[i].num_blocks;
        for (n = 0; n < nb; ++n)
            if (desc->segment_definitions[i].initial_states[n].rf_grad_constant < 0)
                desc->segment_definitions[i].initial_states[n].rf_grad_constant = 0;
    }

    for (i = 0; i < num_unique; ++i)
        if (delay_baseline[i])
            PULSEG_FREE(delay_baseline[i]);
    PULSEG_FREE(delay_baseline);
    delay_baseline = NULL;

    /* ---- 11e. Freeze per-(segment, block-position) initial event state ----
     *
     * Resolve, once, the representative (max-energy) scan instance of every
     * segment position into a flat record on the segment definition.  After
     * this, every consumer that materialises segment memory reads the record
     * instead of walking exec_stream/block_table/rf_table/grad_table, so the
     * pulsegen cache load can drop those O(scan-length) sections entirely.
     * Must run AFTER 11c (max_energy_start_block final). */
    for (i = 0; i < num_unique; ++i)
    {
        pulseg_virtual_segment *sd = &desc->segment_definitions[i];

        nb = sd->num_blocks;
        for (b = 0; b < nb; ++b)
        {
            pulseg_block_initial_state *st = &sd->initial_states[b];
            const pulseg_block_table_element *rep = NULL;
            const pulseg_base_block *pos_def = NULL;
            int scan_idx, bt_idx;

            if (sd->max_energy_start_block >= 0)
            {
                scan_idx = sd->max_energy_start_block + b;
                if (scan_idx >= 0 && scan_idx < desc->exec_stream_len)
                {
                    bt_idx = pulseg__exec_block_idx(desc, scan_idx);
                    if (bt_idx >= 0 && bt_idx < desc->num_blocks)
                        rep = &desc->block_table[bt_idx];
                }
            }
            if (rep)
                st->base_block_id = rep->id;

            /* digital-output event: the segment's own first block_table row
             * (NOT the max-energy instance) -- trigger events are structural. */
            bt_idx = sd->start_block + b;
            if (bt_idx >= 0 && bt_idx < desc->num_blocks)
                st->digitalout_id = desc->block_table[bt_idx].digitalout_id;

            /* RF: representative instance amplitude, else the definition's
             * base amplitude, else 0 for a position with no RF at all. */
            if (sd->unique_block_indices)
            {
                int pos_id = sd->unique_block_indices[b];
                if (pos_id >= 0 && pos_id < desc->num_unique_blocks)
                    pos_def = &desc->base_blocks[pos_id];
            }
            if (pos_def && pos_def->rf_id >= 0 && pos_def->rf_id < desc->num_unique_rfs)
            {
                st->rf_amplitude_hz = desc->rf_definitions[pos_def->rf_id].stats.base_amplitude_hz;
                if (rep && rep->rf_id >= 0 && rep->rf_id < desc->rf_table_size)
                    st->rf_amplitude_hz = desc->rf_table[rep->rf_id].amplitude;
            }

            /* Gradients: definition, shot and amplitude of the representative
             * instance -- the instance decides which grad_definition applies
             * (instances at one position may differ in time_shape_id). */
            if (rep)
            {
                ax_grad_ids[0] = rep->gx_id;
                ax_grad_ids[1] = rep->gy_id;
                ax_grad_ids[2] = rep->gz_id;
                for (ax = 0; ax < 3; ++ax)
                {
                    int raw = ax_grad_ids[ax];
                    int def_id;
                    if (raw < 0 || raw >= desc->grad_table_size)
                        continue;
                    def_id = desc->grad_table[raw].id;
                    if (def_id < 0 || def_id >= desc->num_unique_grads)
                        continue;
                    st->grad_def_id[ax] = def_id;
                    st->grad_shape_id[ax] = desc->grad_table[raw].shape_id;
                    st->grad_amplitude_hz_per_m[ax] = desc->grad_table[raw].amplitude;
                }
            }
        }
    }

    /* ---- Cleanup ---- */
    for (n = 0; n < num_exp_alloc; ++n)
        PULSEG_FREE(exp_segs[n].unique_block_indices);
    PULSEG_FREE(exp_segs);
    exp_segs = NULL;
    PULSEG_FREE(scan_pat);
    scan_pat = NULL;

    diag->code = PULSEG_SUCCESS;
    return num_unique;

scan_seg_fail:
    if (pattern_seg_id)
        PULSEG_FREE(pattern_seg_id);
    if (max_energy)
        PULSEG_FREE(max_energy);
    if (delay_baseline)
    {
        for (i = 0; i < num_unique; ++i)
            if (delay_baseline[i])
                PULSEG_FREE(delay_baseline[i]);
        PULSEG_FREE(delay_baseline);
    }
    if (uniq_segs)
    {
        for (i = 0; i < num_unique; ++i)
            if (uniq_segs[i].unique_block_indices)
                PULSEG_FREE(uniq_segs[i].unique_block_indices);
        PULSEG_FREE(uniq_segs);
    }
    if (exp_segs)
    {
        for (n = 0; n < num_exp_alloc; ++n)
            if (exp_segs[n].unique_block_indices)
                PULSEG_FREE(exp_segs[n].unique_block_indices);
        PULSEG_FREE(exp_segs);
    }
    if (raw_segs)
    {
        for (n = 0; n < num_raw_alloc; ++n)
            if (raw_segs[n].unique_block_indices)
                PULSEG_FREE(raw_segs[n].unique_block_indices);
        PULSEG_FREE(raw_segs);
    }
    if (scan_pat)
        PULSEG_FREE(scan_pat);
    return 0;
}

/* ================================================================== */
/*  Compute segment timing anchors                                    */
/* ================================================================== */

int pulseg__calc_segment_timing(pulseg_sequence_descriptor *desc, pulseg_diagnostic *diag)
{
    pulseg_diagnostic local_diag;
    int seg_idx, blk, block_idx;
    const pulseg_virtual_segment *seg;
    const pulseg_block_table_element *bte;
    const pulseg_base_block *bdef;
    int rf_count, adc_count;
    float t_accum, block_dur_us;
    int rf_raw;
    const pulseg_rf_definition *rdef;
    const pulseg_adc_definition *adef;
    const pulseg_rf_table_element *rte;
    pulseg_segment_rf_anchor *rf_arr;
    pulseg_segment_adc_anchor *adc_arr;
    int rf_def_id, adc_def_id;
    float adc_dur_us;

    if (!diag)
    {
        pulseg_diagnostic_init(&local_diag);
        diag = &local_diag;
    }

    if (!desc || desc->num_unique_segments <= 0)
        return PULSEG_SUCCESS;

    /* ---- Step B: for each segment, collect RF and ADC anchors ---- */
    for (seg_idx = 0; seg_idx < desc->num_unique_segments; ++seg_idx)
    {
        seg = &desc->segment_definitions[seg_idx];

        /* count RF and ADC events (use base_blocks, not
           block_table, because start_block may point to a dummy TR
           instance where ADC/RF events are inactive)                */
        rf_count = 0;
        adc_count = 0;
        for (blk = 0; blk < seg->num_blocks; ++blk)
        {
            int scan_idx = seg->max_energy_start_block + blk;
            if (scan_idx >= 0 && scan_idx < desc->exec_stream_len)
                block_idx = pulseg__exec_block_idx(desc, scan_idx);
            else
                block_idx = seg->start_block + blk;

            if (block_idx < 0 || block_idx >= desc->num_blocks)
                continue;
            bte = &desc->block_table[block_idx];
            bdef = &desc->base_blocks[bte->id];
            if (bdef->rf_id >= 0)
                rf_count++;
            if (bdef->adc_id >= 0)
                adc_count++;
        }

        /* allocate anchor arrays */
        rf_arr = NULL;
        adc_arr = NULL;
        if (rf_count > 0)
        {
            rf_arr = (pulseg_segment_rf_anchor *)PULSEG_ALLOC(
                (size_t)rf_count * sizeof(pulseg_segment_rf_anchor));
            if (!rf_arr)
                goto timing_fail;
        }
        if (adc_count > 0)
        {
            adc_arr = (pulseg_segment_adc_anchor *)PULSEG_ALLOC(
                (size_t)adc_count * sizeof(pulseg_segment_adc_anchor));
            if (!adc_arr)
            {
                if (rf_arr)
                    PULSEG_FREE(rf_arr);
                goto timing_fail;
            }
        }

        /* fill anchors */
        rf_count = 0;
        adc_count = 0;
        t_accum = 0.0f;

        for (blk = 0; blk < seg->num_blocks; ++blk)
        {
            int scan_idx = seg->max_energy_start_block + blk;
            if (scan_idx >= 0 && scan_idx < desc->exec_stream_len)
                block_idx = pulseg__exec_block_idx(desc, scan_idx);
            else
                block_idx = seg->start_block + blk;

            if (block_idx < 0 || block_idx >= desc->num_blocks)
                continue;
            bte = &desc->block_table[block_idx];
            bdef = &desc->base_blocks[bte->id];
            block_dur_us =
                (bte->duration_us >= 0) ? (float)bte->duration_us : (float)bdef->duration_us;

            /* RF anchor */
            rf_raw = bte->rf_id;
            if (rf_raw >= 0 && rf_raw < desc->rf_table_size)
            {
                rte = &desc->rf_table[rf_raw];
                rf_def_id = rte->id;
                if (rf_def_id >= 0 && rf_def_id < desc->num_unique_rfs)
                {
                    int use;
                    rdef = &desc->rf_definitions[rf_def_id];
                    rf_arr[rf_count].block_offset = blk;
                    rf_arr[rf_count].start_us = t_accum + (float)rdef->delay;
                    rf_arr[rf_count].end_us =
                        t_accum + (float)rdef->delay + rdef->stats.duration_us;
                    rf_arr[rf_count].isocenter_us =
                        t_accum + (float)rdef->delay + (float)rdef->stats.isodelay_us;
                    rf_arr[rf_count].base_amplitude_hz = rte->amplitude;

                    /* rf_use: from file tag, or auto-detect from flip angle */
                    use = rte->rf_use;
                    if (use == PULSEG_RF_USE_UNKNOWN && rdef->stats.base_amplitude_hz > 0.0f)
                    {
                        float ratio =
                            (float)fabs((double)rte->amplitude) / rdef->stats.base_amplitude_hz;
                        float actual_flip =
                            rdef->stats.flip_angle_rad * ratio * (180.0f / (float)M_PI);
                        if (actual_flip > 162.0f && actual_flip < 198.0f)
                            use = PULSEG_RF_USE_REFOCUSING;
                        else
                            use = PULSEG_RF_USE_EXCITATION;
                    }
                    rf_arr[rf_count].rf_use = use;
                    rf_count++;
                }
            }

            /* ADC anchor */
            adc_def_id = bdef->adc_id;
            if (adc_def_id >= 0 && adc_def_id < desc->num_unique_adcs)
            {
                adef = &desc->adc_definitions[adc_def_id];
                adc_dur_us = (float)adef->num_samples * (float)adef->dwell_time * 1e-3f;

                adc_arr[adc_count].block_offset = blk;
                adc_arr[adc_count].start_us = t_accum + (float)adef->delay;
                adc_arr[adc_count].end_us = t_accum + (float)adef->delay + adc_dur_us;

                adc_count++;
            }

            t_accum += block_dur_us;
        }

        /* store timing */
        ((pulseg_virtual_segment *)seg)->timing.num_rf_anchors = rf_count;
        ((pulseg_virtual_segment *)seg)->timing.rf_anchors = rf_arr;
        ((pulseg_virtual_segment *)seg)->timing.num_adc_anchors = adc_count;
        ((pulseg_virtual_segment *)seg)->timing.adc_anchors = adc_arr;
    }

    return PULSEG_SUCCESS;

timing_fail:
    return PULSEG_ERR_ALLOC_FAILED;
}
