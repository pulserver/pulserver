/**
 * @file pulseg_descriptor.c
 * @brief Readers over what a sequence descriptor stores: block-definition
 *        structure, the compact execution stream, and gradient shape
 *        statistics.
 *
 * These answer from a descriptor as it stands, however it was filled, so a
 * scanner build links them without the conversion passes that fill it.
 */

#include <math.h>
#include <stddef.h>

#include "pulseg_internal.h"
#include "pulseg.h"

/* ================================================================== */
/*  Block definition structure                                        */
/* ================================================================== */

/* A block definition is a pure delay when it carries no RF, gradient or ADC
 * event -- only a duration (matches the parser's block_table duration_us >= 0
 * marker).  Its duration can be set per instance at run time. */
int pulseg__block_def_is_pure_delay(const pulseg_base_block *b)
{
    return (b->rf_id < 0 && b->gx_id < 0 && b->gy_id < 0 && b->gz_id < 0 && b->adc_id < 0);
}

int pulseg__rf_event_use(const pulseg_rf_definition *rdef, const pulseg_rf_table_element *rte)
{
    double flip_deg;

    if (rte->rf_use != PULSEG_RF_USE_UNKNOWN || rdef->stats.base_amplitude_hz <= 0.0f)
        return rte->rf_use;
    flip_deg = (double)rdef->stats.flip_angle_rad * fabs((double)rte->amplitude) /
               (double)rdef->stats.base_amplitude_hz * (180.0 / M_PI);
    return (flip_deg > 162.0 && flip_deg < 198.0) ? PULSEG_RF_USE_REFOCUSING
                                                  : PULSEG_RF_USE_EXCITATION;
}

/* Two block definitions play the same RF and gradient definitions for the
 * same duration, and differ at most in the ADC definition.  A prepared segment
 * position is built from one definition's pulses, so only such definitions
 * may share it; which readout it digitises with is refined per repetition. */
int pulseg__block_defs_play_same_pulses(
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

    return a->duration_us == b->duration_us && a->rf_id == b->rf_id && a->gx_id == b->gx_id &&
           a->gy_id == b->gy_id && a->gz_id == b->gz_id && (a->adc_id >= 0) == (b->adc_id >= 0);
}

/* ================================================================== */
/*  Compact execution-stream accessors                                 */
/* ================================================================== */

/* Locate the run covering stream position @p n.
 *
 * The scan loop is real-time and walks the stream strictly in order, so the
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

/* ================================================================== */
/*  Gradient shape statistics                                          */
/* ================================================================== */

int pulseg__played_duration_us(const pulseg_sequence_descriptor *desc, int block)
{
    const pulseg_block_table_element *bte = &desc->block_table[block];

    if (bte->duration_us >= 0)
        return bte->duration_us;
    if (bte->id >= 0 && bte->id < desc->num_unique_blocks)
        return desc->base_blocks[bte->id].duration_us;
    return 0;
}

float pulseg__trap_energy(float rise_us, float flat_us, float fall_us)
{
    return (rise_us / 3.0f + flat_us + fall_us / 3.0f) * 1e-6f;
}

float pulseg__grad_instance_energy(
    const pulseg_sequence_descriptor *desc,
    const pulseg_grad_definition *gd,
    int shape_id)
{
    if (shape_id > 0)
    {
        if (!desc || !desc->grad_shape_energy || shape_id > desc->num_grad_shape_stats)
            return 0.0f;
        return desc->grad_shape_energy[shape_id - 1];
    }
    if (!gd || gd->type != 0)
        return 0.0f;
    return pulseg__trap_energy(
        (float)gd->rise_time_or_unused,
        (float)gd->flat_time_or_unused,
        (float)gd->fall_time_or_num_uncompressed_samples);
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

float pulseg__grad_shape_slew(const pulseg_sequence_descriptor *desc, int shape_id)
{
    if (!desc || !desc->grad_shape_slew || shape_id <= 0 || shape_id > desc->num_grad_shape_stats)
        return 0.0f;
    return desc->grad_shape_slew[shape_id - 1];
}
