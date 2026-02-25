/* pulseqlib_structure.c -- TR detection, segmentation, frequency modulation
 *
 * This file contains:
 *   - TR detection (find_tr_in_sequence)
 *   - Segment state machine (find_segments_in_tr)
 *   - Frequency modulation library building (build_freq_mod_library)
 *
 * Split from pulseqlib_core.c for modularity.
 * ANSI C89.
 */

#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "pulseqlib_internal.h"
#include "pulseqlib_methods.h"

/* ================================================================== */
/*  File-scope constants                                              */
/* ================================================================== */
#define PREP_COOLDOWN_THRESHOLD_US   100000   /* 100 ms */
#define SINGLE_TR_MAX_DURATION_US  15000000   /* 15 s  */

#define SEGSTATE_SEEKING_FIRST_ADC 0
#define SEGSTATE_SEEKING_BOUNDARY  1
#define SEGSTATE_OPTIMIZED_MODE    2

/* ================================================================== */
/*  Tiny helpers                                                      */
/* ================================================================== */

static int array_equal(const int* a, const int* b, int len)
{
    int i;
    for (i = 0; i < len; ++i)
        if (a[i] != b[i]) return 0;
    return 1;
}

/* ================================================================== */
/*  TR detection helpers                                              */
/* ================================================================== */

static double sum_durations_us(const int* dur, int start, int count)
{
    double total = 0.0;
    int i;
    for (i = 0; i < count; ++i) total += (double)dur[start + i];
    return total;
}

static int first_repeating_segment(const int* s, int len)
{
    int start, sub_len, l, i, match;
    const int* p;

    if (len <= 1) return len;

    for (start = 0; start < len; ++start) {
        p = s + start;
        sub_len = len - start;
        for (l = 1; l <= sub_len / 2; ++l) {
            match = 1;
            for (i = 0; i < l; ++i) {
                if (p[i] != p[i + l]) { match = 0; break; }
            }
            if (match) return l;
        }
    }
    return len;
}

/* ================================================================== */
/*  build_scan_table                                                  */
/* ================================================================== */

/*
 * Build the scan table: an expanded playback order that resolves
 * ONCE semantics and multiple averages into a flat array.
 *
 * For avg == 0:             play ONCE=1 (prep) and ONCE=0 (main)
 * For 0 < avg < navg-1:    play ONCE=0 (main) only
 * For avg == navg-1:        play ONCE=0 (main) and ONCE=2 (cooldown)
 *
 * When navg == 1, all three flags are played (entire block table).
 *
 * Each entry stores the block_table index.
 * tr_id column is filled based on non-degenerate prep/cooldown:
 *   - No non-degenerate prep or cooldown:   main = 0
 *   - Prep only:                            prep = 0, main = 1
 *   - Cooldown only:                        main = 0, cooldown = 1
 *   - Both:                                 prep = 0, main = 1, cooldown = 2
 *
 * seg_id column is initialised to -1 (filled later by segment detection).
 */

int pulseqlib__build_scan_table(
    pulseqlib_sequence_descriptor* desc,
    int num_averages,
    pulseqlib_diagnostic* diag)
{
    pulseqlib_diagnostic local_diag;
    int avg, blk, once, count, idx;
    int has_nd_prep, has_nd_cool;
    int prep_tr_id, main_tr_id, cool_tr_id;
    int play_prep, play_main, play_cool;

    if (!diag) { pulseqlib_diagnostic_init(&local_diag); diag = &local_diag; }
    else       pulseqlib_diagnostic_init(diag);

    if (!desc) { diag->code = PULSEQLIB_ERR_NULL_POINTER; return diag->code; }
    if (num_averages < 1) num_averages = 1;

    has_nd_prep = (desc->tr_descriptor.num_prep_blocks > 0 &&
                   !desc->tr_descriptor.degenerate_prep);
    has_nd_cool = (desc->tr_descriptor.num_cooldown_blocks > 0 &&
                   !desc->tr_descriptor.degenerate_cooldown);

    /* Assign tr_id values */
    if (!has_nd_prep && !has_nd_cool) {
        prep_tr_id = -1; main_tr_id = 0; cool_tr_id = -1;
    } else if (has_nd_prep && !has_nd_cool) {
        prep_tr_id = 0;  main_tr_id = 1; cool_tr_id = -1;
    } else if (!has_nd_prep && has_nd_cool) {
        prep_tr_id = -1; main_tr_id = 0; cool_tr_id = 1;
    } else {
        prep_tr_id = 0;  main_tr_id = 1; cool_tr_id = 2;
    }

    /* Pass 1: count entries */
    count = 0;
    for (avg = 0; avg < num_averages; ++avg) {
        play_prep = (avg == 0) ? 1 : 0;
        play_main = 1;
        play_cool = (avg == num_averages - 1) ? 1 : 0;

        for (blk = 0; blk < desc->num_blocks; ++blk) {
            once = desc->block_table[blk].once_flag;
            if (once == 1 && play_prep)      ++count;
            else if (once == 0 && play_main) ++count;
            else if (once == 2 && play_cool) ++count;
        }
    }

    /* Allocate */
    desc->scan_table_len       = count;
    desc->scan_table_block_idx = (int*)PULSEQLIB_ALLOC((size_t)count * sizeof(int));
    desc->scan_table_tr_id     = (int*)PULSEQLIB_ALLOC((size_t)count * sizeof(int));
    desc->scan_table_seg_id    = (int*)PULSEQLIB_ALLOC((size_t)count * sizeof(int));
    if (!desc->scan_table_block_idx ||
        !desc->scan_table_tr_id ||
        !desc->scan_table_seg_id) {
        if (desc->scan_table_block_idx) { PULSEQLIB_FREE(desc->scan_table_block_idx); desc->scan_table_block_idx = NULL; }
        if (desc->scan_table_tr_id)     { PULSEQLIB_FREE(desc->scan_table_tr_id);     desc->scan_table_tr_id = NULL; }
        if (desc->scan_table_seg_id)    { PULSEQLIB_FREE(desc->scan_table_seg_id);    desc->scan_table_seg_id = NULL; }
        desc->scan_table_len = 0;
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        return diag->code;
    }

    /* Pass 2: fill */
    idx = 0;
    for (avg = 0; avg < num_averages; ++avg) {
        play_prep = (avg == 0) ? 1 : 0;
        play_main = 1;
        play_cool = (avg == num_averages - 1) ? 1 : 0;

        for (blk = 0; blk < desc->num_blocks; ++blk) {
            once = desc->block_table[blk].once_flag;
            if (once == 1 && play_prep) {
                desc->scan_table_block_idx[idx] = blk;
                desc->scan_table_tr_id[idx]     = prep_tr_id;
                desc->scan_table_seg_id[idx]    = -1;
                ++idx;
            } else if (once == 0 && play_main) {
                desc->scan_table_block_idx[idx] = blk;
                desc->scan_table_tr_id[idx]     = main_tr_id;
                desc->scan_table_seg_id[idx]    = -1;
                ++idx;
            } else if (once == 2 && play_cool) {
                desc->scan_table_block_idx[idx] = blk;
                desc->scan_table_tr_id[idx]     = cool_tr_id;
                desc->scan_table_seg_id[idx]    = -1;
                ++idx;
            }
        }
    }

    diag->code = PULSEQLIB_OK;
    return PULSEQLIB_OK;
}

/* ================================================================== */
/*  find_tr_in_sequence                                               */
/* ================================================================== */

int pulseqlib__get_tr_in_sequence(pulseqlib_sequence_descriptor* desc, pulseqlib_diagnostic* diag)
{
    pulseqlib_tr_descriptor* tr = &desc->tr_descriptor;
    pulseqlib_diagnostic local_diag;
    int i, n;
    int imaging_start, imaging_end, imaging_len;
    int* seq_pat       = NULL;
    int* block_dur     = NULL;
    int prep_dur_us, cooldown_dur_us, active_dur_us;
    int found, l;
    int mismatch_pos;
    float tr_dur;
    int tr_start;

    if (!diag) { pulseqlib_diagnostic_init(&local_diag); diag = &local_diag; }
    else       pulseqlib_diagnostic_init(diag);

    found = 0; l = 0; mismatch_pos = -1;

    if (desc->num_blocks <= 0 || !desc->block_table || !desc->block_definitions) {
        diag->code = (desc->num_blocks <= 0)
            ? PULSEQLIB_ERR_TR_NO_BLOCKS : PULSEQLIB_ERR_NULL_POINTER;
        return diag->code;
    }
    if (desc->num_prep_blocks < 0 || desc->num_cooldown_blocks < 0) {
        diag->code = PULSEQLIB_ERR_INVALID_ARGUMENT;
        return diag->code;
    }
    if (desc->num_prep_blocks + desc->num_cooldown_blocks > desc->num_blocks) {
        diag->code = PULSEQLIB_ERR_TR_NO_IMAGING_REGION;
        return diag->code;
    }

    tr->tr_size             = 0;
    tr->num_trs             = 0;
    tr->tr_duration_us      = 0.0f;
    tr->degenerate_prep     = 1;
    tr->num_prep_blocks     = desc->num_prep_blocks;
    tr->num_prep_trs        = 1;
    tr->degenerate_cooldown = 1;
    tr->num_cooldown_blocks = desc->num_cooldown_blocks;
    tr->num_cooldown_trs    = 1;

    imaging_start = desc->num_prep_blocks;
    imaging_end   = desc->num_blocks - desc->num_cooldown_blocks;
    imaging_len   = imaging_end - imaging_start;
    pulseqlib__diag_printf(diag, "imaging region length=%d", imaging_len);

    if (imaging_len <= 0) {
        diag->code = PULSEQLIB_ERR_TR_NO_IMAGING_REGION;
        return diag->code;
    }

    /* unique-block count for diagnostics */
    {
        int max_u = 0;
        for (n = 0; n < desc->num_blocks; ++n)
            if (desc->block_table[n].id > max_u) max_u = desc->block_table[n].id;
        pulseqlib__diag_printf(diag, " unique blocks=%d", max_u + 1);
    }

    seq_pat   = (int*)PULSEQLIB_ALLOC(desc->num_blocks * sizeof(int));
    block_dur = (int*)PULSEQLIB_ALLOC(desc->num_blocks * sizeof(int));
    if (!seq_pat || !block_dur) {
        if (seq_pat)   PULSEQLIB_FREE(seq_pat);
        if (block_dur) PULSEQLIB_FREE(block_dur);
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        return diag->code;
    }

    for (n = 0; n < desc->num_blocks; ++n) {
        block_dur[n] = desc->block_definitions[desc->block_table[n].id].duration_us;
        seq_pat[n] = (desc->block_table[n].duration_us >= 0)
            ? block_dur[n]
            : -1 * desc->block_table[desc->block_table[n].id].id;
    }

    l = first_repeating_segment(&seq_pat[imaging_start], imaging_len);
    pulseqlib__diag_printf(diag, " candidate TR=%d", l);

    found = (l > 0 && l <= imaging_len) ? 1 : 0;

    if (found) {
        for (i = 0; i < imaging_len; ++i) {
            n = imaging_start + i;
            if (seq_pat[n] != seq_pat[imaging_start + (i % l)]) {
                mismatch_pos = i;
                pulseqlib__diag_printf(diag, " mismatch at offset=%d", i);
                pulseqlib__diag_printf(diag, " block=%d", n);
                found = 0;
                break;
            }
        }
    }

    if (!found) {
        active_dur_us = 0;
        for (n = 0; n < desc->num_blocks; ++n)
            if (desc->block_table[n].duration_us < 0)
                active_dur_us += desc->block_definitions[desc->block_table[n].id].duration_us;

        if (active_dur_us <= SINGLE_TR_MAX_DURATION_US) {
            tr->tr_size             = desc->num_blocks;
            tr->num_trs             = 1;
            tr->degenerate_prep     = 1;
            tr->num_prep_blocks     = 0;
            tr->num_prep_trs        = 0;
            tr->degenerate_cooldown = 1;
            tr->num_cooldown_blocks = 0;
            tr->num_cooldown_trs    = 0;
            tr_dur = 0.0f;
            for (i = 0; i < desc->num_blocks; ++i)
                tr_dur += (float)block_dur[i];
            tr->tr_duration_us = tr_dur;
            diag->code = PULSEQLIB_OK;
            PULSEQLIB_FREE(seq_pat); PULSEQLIB_FREE(block_dur);
            return PULSEQLIB_OK;
        }
        diag->code = (mismatch_pos >= 0)
            ? PULSEQLIB_ERR_TR_PATTERN_MISMATCH
            : PULSEQLIB_ERR_TR_NO_PERIODIC_PATTERN;
        PULSEQLIB_FREE(seq_pat); PULSEQLIB_FREE(block_dur);
        return diag->code;
    }

    tr->tr_size = l;
    tr->num_trs = imaging_len / l;
    tr_dur = 0.0f;
    tr_start = imaging_start;
    for (i = 0; i < l; ++i) tr_dur += (float)block_dur[tr_start + i];
    tr->tr_duration_us = tr_dur;

    /* prep check */
    if (desc->num_prep_blocks) {
        if (desc->num_prep_blocks % l == 0) {
            for (n = 0; n < (int)(desc->num_prep_blocks / l); ++n) {
                if (!array_equal(&seq_pat[imaging_start], &seq_pat[n * l], l)) {
                    prep_dur_us = (int)sum_durations_us(block_dur, 0, desc->num_prep_blocks);
                    if (prep_dur_us > PREP_COOLDOWN_THRESHOLD_US) {
                        diag->code = PULSEQLIB_ERR_TR_PREP_TOO_LONG;
                        PULSEQLIB_FREE(seq_pat); PULSEQLIB_FREE(block_dur);
                        return diag->code;
                    }
                    tr->degenerate_prep = 0;
                    break;
                }
            }
            if (tr->degenerate_prep == 1) {
                tr->num_prep_blocks = 0;
                tr->num_prep_trs    = desc->num_prep_blocks / l;
            }
        } else {
            prep_dur_us = (int)sum_durations_us(block_dur, 0, desc->num_prep_blocks);
            if (prep_dur_us > PREP_COOLDOWN_THRESHOLD_US) {
                diag->code = PULSEQLIB_ERR_TR_PREP_TOO_LONG;
                PULSEQLIB_FREE(seq_pat); PULSEQLIB_FREE(block_dur);
                return diag->code;
            }
            tr->degenerate_prep = 0;
        }
    }

    /* cooldown check */
    if (desc->num_cooldown_blocks) {
        if (desc->num_cooldown_blocks % l == 0) {
            for (n = 0; n < (int)(desc->num_cooldown_blocks / l); ++n) {
                if (!array_equal(&seq_pat[imaging_start], &seq_pat[imaging_end + n * l], l)) {
                    cooldown_dur_us = (int)sum_durations_us(block_dur, imaging_end, desc->num_cooldown_blocks);
                    if (cooldown_dur_us > PREP_COOLDOWN_THRESHOLD_US) {
                        diag->code = PULSEQLIB_ERR_TR_COOLDOWN_TOO_LONG;
                        PULSEQLIB_FREE(seq_pat); PULSEQLIB_FREE(block_dur);
                        return diag->code;
                    }
                    tr->degenerate_cooldown = 0;
                    break;
                }
            }
            if (tr->degenerate_cooldown == 1) {
                tr->num_cooldown_blocks = 0;
                tr->num_cooldown_trs    = desc->num_cooldown_blocks / l;
            }
        } else {
            cooldown_dur_us = (int)sum_durations_us(block_dur, imaging_end, desc->num_cooldown_blocks);
            if (cooldown_dur_us > PREP_COOLDOWN_THRESHOLD_US) {
                diag->code = PULSEQLIB_ERR_TR_COOLDOWN_TOO_LONG;
                PULSEQLIB_FREE(seq_pat); PULSEQLIB_FREE(block_dur);
                return diag->code;
            }
            tr->degenerate_cooldown = 0;
        }
    }

    diag->code = PULSEQLIB_OK;
    PULSEQLIB_FREE(seq_pat); PULSEQLIB_FREE(block_dur);
    return PULSEQLIB_OK;
}

/* ================================================================== */
/*  Segment state machine                                             */
/* ================================================================== */

static int find_segments_internal(
    const pulseqlib_sequence_descriptor* desc, 
    pulseqlib_tr_segment* segs, int offset, 
    pulseqlib_diagnostic* diag, 
    const pulseqlib_opts* opts,
    int tr_start, int tr_size)
{
    float max_slew, grad_raster_s, max_allowed;
    int grad_ids[3];
    float phys_first, phys_last;
    float grad_last_cur[3], grad_first_next[3];
    const pulseqlib_block_definition* bdef;
    const pulseqlib_grad_definition* gdef;
    int bdef_id, shot_idx;
    int* seg_starts = NULL;
    int* seg_sizes  = NULL;
    int num_seg, seg_start;
    int state, cand_before_rf, saved_cand, has_saved_cand;
    int has_rf, has_adc, is_cand;
    int nb, n, i;

    max_slew = opts->max_slew_hz_per_m_per_s;
    grad_raster_s = desc->grad_raster_us * 1e-6f;
    max_allowed = max_slew * grad_raster_s;
    nb = tr_size;

    seg_starts = (int*)PULSEQLIB_ALLOC(nb * sizeof(int));
    seg_sizes  = (int*)PULSEQLIB_ALLOC(nb * sizeof(int));
    if (!seg_starts || !seg_sizes) {
        if (seg_starts) PULSEQLIB_FREE(seg_starts);
        if (seg_sizes)  PULSEQLIB_FREE(seg_sizes);
        if (diag) diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        return 0;
    }

    /* first block gradient check */
    bdef_id = desc->block_table[tr_start].id;
    bdef = &desc->block_definitions[bdef_id];
    grad_ids[0] = bdef->gx_id; grad_ids[1] = bdef->gy_id; grad_ids[2] = bdef->gz_id;
    for (i = 0; i < 3; ++i) {
        if (grad_ids[i] < 0) continue;
        gdef = &desc->grad_definitions[grad_ids[i]];
        for (shot_idx = 0; shot_idx < gdef->num_shots; ++shot_idx) {
            phys_first = gdef->first_value[shot_idx] * gdef->max_amplitude[shot_idx];
            if ((float)fabs(phys_first) > max_allowed) {
                if (diag) {
                    diag->code = PULSEQLIB_ERR_SEG_NONZERO_START_GRAD;
                    pulseqlib__diag_printf(diag, " block=%d", tr_start);
                    pulseqlib__diag_printf(diag, " channel=%d", i);
                    pulseqlib__diag_printf(diag, " gradient_amplitude=%g", (double)phys_first);
                    pulseqlib__diag_printf(diag, " max_allowed_amplitude=%g", (double)max_allowed);
                }
                PULSEQLIB_FREE(seg_starts); PULSEQLIB_FREE(seg_sizes);
                return 0;
            }
        }
    }

    /* last block gradient check */
    bdef_id = desc->block_table[tr_start + nb - 1].id;
    bdef = &desc->block_definitions[bdef_id];
    grad_ids[0] = bdef->gx_id; grad_ids[1] = bdef->gy_id; grad_ids[2] = bdef->gz_id;
    for (i = 0; i < 3; ++i) {
        if (grad_ids[i] < 0) continue;
        gdef = &desc->grad_definitions[grad_ids[i]];
        for (shot_idx = 0; shot_idx < gdef->num_shots; ++shot_idx) {
            phys_last = gdef->last_value[shot_idx] * gdef->max_amplitude[shot_idx];
            if ((float)fabs(phys_last) > max_allowed) {
                if (diag) {
                    diag->code = PULSEQLIB_ERR_SEG_NONZERO_END_GRAD;
                    pulseqlib__diag_printf(diag, " block=%d", tr_start + nb - 1);
                    pulseqlib__diag_printf(diag, " channel=%d", i);
                    pulseqlib__diag_printf(diag, " gradient_amplitude=%g", (double)phys_last);
                    pulseqlib__diag_printf(diag, " max_allowed_amplitude=%g", (double)max_allowed);
                }
                PULSEQLIB_FREE(seg_starts); PULSEQLIB_FREE(seg_sizes);
                return 0;
            }
        }
    }

    /* state machine */
    num_seg = 0;
    seg_start = tr_start;
    state = SEGSTATE_SEEKING_FIRST_ADC;
    cand_before_rf = -1;
    saved_cand = -1;
    has_saved_cand = 0;

    for (n = tr_start; n < tr_start + nb; ++n) {
        is_cand = 0;
        if (n > tr_start) {
            is_cand = 1;

            bdef_id = desc->block_table[n - 1].id;
            bdef = &desc->block_definitions[bdef_id];
            grad_ids[0] = bdef->gx_id; grad_ids[1] = bdef->gy_id; grad_ids[2] = bdef->gz_id;
            for (i = 0; i < 3; ++i) {
                grad_last_cur[i] = 0.0f;
                if (grad_ids[i] >= 0) {
                    gdef = &desc->grad_definitions[grad_ids[i]];
                    for (shot_idx = 0; shot_idx < gdef->num_shots; ++shot_idx) {
                        phys_last = gdef->last_value[shot_idx] * gdef->max_amplitude[shot_idx];
                        if ((float)fabs(phys_last) > (float)fabs(grad_last_cur[i]))
                            grad_last_cur[i] = phys_last;
                    }
                }
            }

            bdef_id = desc->block_table[n].id;
            bdef = &desc->block_definitions[bdef_id];
            grad_ids[0] = bdef->gx_id; grad_ids[1] = bdef->gy_id; grad_ids[2] = bdef->gz_id;
            for (i = 0; i < 3; ++i) {
                grad_first_next[i] = 0.0f;
                if (grad_ids[i] >= 0) {
                    gdef = &desc->grad_definitions[grad_ids[i]];
                    for (shot_idx = 0; shot_idx < gdef->num_shots; ++shot_idx) {
                        phys_first = gdef->first_value[shot_idx] * gdef->max_amplitude[shot_idx];
                        if ((float)fabs(phys_first) > (float)fabs(grad_first_next[i]))
                            grad_first_next[i] = phys_first;
                    }
                }
            }

            for (i = 0; i < 3; ++i) {
                if ((float)fabs(grad_last_cur[i]) > max_allowed ||
                    (float)fabs(grad_first_next[i]) > max_allowed) {
                    is_cand = 0; break;
                }
            }
        }

        has_rf  = (desc->block_definitions[desc->block_table[n].id].rf_id >= 0);
        has_adc = (desc->block_table[n].adc_id >= 0);

        if (state == SEGSTATE_SEEKING_FIRST_ADC) {
            if (is_cand) saved_cand = n;
            if (has_rf)  { cand_before_rf = saved_cand; saved_cand = -1; }
            if (has_adc) {
                if (cand_before_rf > seg_start) {
                    seg_starts[num_seg] = seg_start;
                    seg_sizes[num_seg]  = cand_before_rf - seg_start;
                    num_seg++;
                    seg_start = cand_before_rf;
                }
                state = SEGSTATE_SEEKING_BOUNDARY;
                has_saved_cand = 0;
                saved_cand = -1;
            }
        } else if (state == SEGSTATE_SEEKING_BOUNDARY) {
            if (is_cand) { saved_cand = n; has_saved_cand = 1; }
            if (has_rf) {
                if (has_saved_cand) {
                    seg_starts[num_seg] = seg_start;
                    seg_sizes[num_seg]  = saved_cand - seg_start;
                    num_seg++;
                    seg_start = saved_cand;
                    has_saved_cand = 0;
                    saved_cand = -1;
                } else {
                    state = SEGSTATE_OPTIMIZED_MODE;
                }
            }
        }
        /* SEGSTATE_OPTIMIZED_MODE: no action */
    }

    seg_starts[num_seg] = seg_start;
    seg_sizes[num_seg]  = tr_start + nb - seg_start;
    num_seg++;

    for (i = 0; i < num_seg; ++i) {
        segs[offset + i].start_block = seg_starts[i];
        segs[offset + i].num_blocks  = seg_sizes[i];
        segs[offset + i].unique_block_indices = NULL;
    }

    PULSEQLIB_FREE(seg_starts); PULSEQLIB_FREE(seg_sizes);
    return num_seg;
}

/* ================================================================== */
/*  Strip pure delays from segments                                   */
/* ================================================================== */

static int strip_pure_delays(
    const pulseqlib_tr_segment* raw_segs, int num_raw,
    pulseqlib_tr_segment* out, int max_out,
    const pulseqlib_block_table_element* bt
) {
    int num_out = 0;
    int s, i, n_blk;
    int leading, trailing, core_start, core_end, core_size;
    const int* idx;

    for (s = 0; s < num_raw; ++s) {
        n_blk = raw_segs[s].num_blocks;
        idx   = raw_segs[s].unique_block_indices;
        if (n_blk == 0 || !idx) continue;

        leading = 0;
        for (i = 0; i < n_blk; ++i) {
            if (bt[raw_segs[s].start_block + i].duration_us >= 0) leading++;
            else break;
        }
        trailing = 0;
        for (i = n_blk - 1; i >= leading; --i) {
            if (bt[raw_segs[s].start_block + i].duration_us >= 0) trailing++;
            else break;
        }
        core_start = leading;
        core_end   = n_blk - trailing;

        for (i = 0; i < leading; ++i) {
            if (num_out >= max_out) return -1;
            out[num_out].start_block = raw_segs[s].start_block + i;
            out[num_out].num_blocks  = 1;
            out[num_out].unique_block_indices = (int*)PULSEQLIB_ALLOC(sizeof(int));
            if (!out[num_out].unique_block_indices) return -1;
            out[num_out].unique_block_indices[0] = idx[i];
            num_out++;
        }
        if (core_end > core_start) {
            core_size = core_end - core_start;
            if (num_out >= max_out) return -1;
            out[num_out].start_block = raw_segs[s].start_block + core_start;
            out[num_out].num_blocks  = core_size;
            out[num_out].unique_block_indices = (int*)PULSEQLIB_ALLOC(core_size * sizeof(int));
            if (!out[num_out].unique_block_indices) return -1;
            for (i = 0; i < core_size; ++i)
                out[num_out].unique_block_indices[i] = idx[core_start + i];
            num_out++;
        }
        for (i = 0; i < trailing; ++i) {
            if (num_out >= max_out) return -1;
            out[num_out].start_block = raw_segs[s].start_block + core_end + i;
            out[num_out].num_blocks  = 1;
            out[num_out].unique_block_indices = (int*)PULSEQLIB_ALLOC(sizeof(int));
            if (!out[num_out].unique_block_indices) return -1;
            out[num_out].unique_block_indices[0] = idx[core_end + i];
            num_out++;
        }
    }
    return num_out;
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
 * @param[in]  scan_bi  If non-NULL, resolve through scan_table_block_idx
 * @return Number of output segments, or -1 on allocation failure.
 */
static int nav_split_merge(
    pulseqlib_tr_segment* in,  int num_in,
    pulseqlib_tr_segment* out, int max_out,
    const pulseqlib_block_table_element* bt,
    const int* scan_bi)
{
    pulseqlib_tr_segment* split_buf = NULL;
    int* new_ubi;
    int  split_max, num_split, num_out;
    int  n, b, k;
    int  sb, nb, run_start, run_len, cur_nav, blk_nav;
    int  bt_idx, this_nav, prev_nav;

    if (num_in == 0) return 0;

    /* worst case: every block becomes its own segment */
    split_max = 0;
    for (n = 0; n < num_in; ++n) split_max += in[n].num_blocks;
    if (split_max == 0) return 0;

    split_buf = (pulseqlib_tr_segment*)PULSEQLIB_ALLOC(
        (size_t)split_max * sizeof(pulseqlib_tr_segment));
    if (!split_buf) return -1;

    /* ---- Pass 1: split at NAV transitions ---- */
    num_split = 0;
    for (n = 0; n < num_in; ++n) {
        sb = in[n].start_block;
        nb = in[n].num_blocks;
        if (nb <= 0) {
            PULSEQLIB_FREE(in[n].unique_block_indices);
            in[n].unique_block_indices = NULL;
            continue;
        }

        bt_idx  = scan_bi ? scan_bi[sb] : sb;
        cur_nav = bt[bt_idx].nav_flag ? 1 : 0;
        run_start = 0;

        for (b = 1; b <= nb; ++b) {
            if (b < nb) {
                bt_idx  = scan_bi ? scan_bi[sb + b] : (sb + b);
                blk_nav = bt[bt_idx].nav_flag ? 1 : 0;
            } else {
                blk_nav = -1;   /* sentinel — force flush of last run */
            }

            if (blk_nav != cur_nav) {
                run_len = b - run_start;
                split_buf[num_split].start_block = sb + run_start;
                split_buf[num_split].num_blocks  = run_len;
                split_buf[num_split].unique_block_indices =
                    (int*)PULSEQLIB_ALLOC((size_t)run_len * sizeof(int));
                if (!split_buf[num_split].unique_block_indices) {
                    for (k = 0; k < num_split; ++k)
                        PULSEQLIB_FREE(split_buf[k].unique_block_indices);
                    PULSEQLIB_FREE(split_buf);
                    return -1;
                }
                for (k = 0; k < run_len; ++k)
                    split_buf[num_split].unique_block_indices[k] =
                        in[n].unique_block_indices[run_start + k];
                num_split++;
                run_start = b;
                cur_nav   = blk_nav;
            }
        }

        PULSEQLIB_FREE(in[n].unique_block_indices);
        in[n].unique_block_indices = NULL;
    }

    /* ---- Pass 2: merge adjacent NAV segments ---- */
    num_out = 0;
    for (n = 0; n < num_split; ++n) {
        bt_idx   = scan_bi ? scan_bi[split_buf[n].start_block]
                           : split_buf[n].start_block;
        this_nav = bt[bt_idx].nav_flag ? 1 : 0;

        if (this_nav && num_out > 0) {
            pulseqlib_tr_segment* prev = &out[num_out - 1];
            bt_idx   = scan_bi ? scan_bi[prev->start_block]
                               : prev->start_block;
            prev_nav = bt[bt_idx].nav_flag ? 1 : 0;

            if (prev_nav &&
                prev->start_block + prev->num_blocks ==
                    split_buf[n].start_block) {
                /* merge into previous segment */
                int old_nb = prev->num_blocks;
                int add_nb = split_buf[n].num_blocks;
                int new_nb = old_nb + add_nb;
                new_ubi = (int*)PULSEQLIB_ALLOC((size_t)new_nb * sizeof(int));
                if (!new_ubi) {
                    for (k = n; k < num_split; ++k)
                        PULSEQLIB_FREE(split_buf[k].unique_block_indices);
                    PULSEQLIB_FREE(split_buf);
                    return -1;
                }
                for (k = 0; k < old_nb; ++k)
                    new_ubi[k] = prev->unique_block_indices[k];
                for (k = 0; k < add_nb; ++k)
                    new_ubi[old_nb + k] =
                        split_buf[n].unique_block_indices[k];
                PULSEQLIB_FREE(prev->unique_block_indices);
                PULSEQLIB_FREE(split_buf[n].unique_block_indices);
                split_buf[n].unique_block_indices = NULL;
                prev->unique_block_indices = new_ubi;
                prev->num_blocks = new_nb;
                continue;
            }
        }

        if (num_out >= max_out) {
            for (k = n; k < num_split; ++k)
                PULSEQLIB_FREE(split_buf[k].unique_block_indices);
            PULSEQLIB_FREE(split_buf);
            return -1;
        }
        out[num_out] = split_buf[n];
        split_buf[n].unique_block_indices = NULL;   /* ownership transferred */
        num_out++;
    }

    PULSEQLIB_FREE(split_buf);
    return num_out;
}

/* ================================================================== */
/*  find_segments_in_tr                                               */
/* ================================================================== */

int pulseqlib__get_segments_in_tr(pulseqlib_sequence_descriptor* desc, pulseqlib_diagnostic* diag, const pulseqlib__seq_file* seq)
{
    const pulseqlib_tr_descriptor* tr = &desc->tr_descriptor;
    const pulseqlib_block_table_element* bte;
    const pulseqlib_block_definition* bdef;
    pulseqlib_tr_segment* raw_segs  = NULL;
    pulseqlib_tr_segment* exp_segs  = NULL;
    pulseqlib_tr_segment* uniq_segs = NULL;
    pulseqlib_diagnostic local_diag;
    int total_blocks;
    int num_raw, n_prep_raw, n_main_raw, n_cool_raw;
    int n_prep, n_main, n_cool;
    int num_total, num_unique;
    int found, tr_start, tr_size;
    int n, b, i, offset;
    int max_expanded, seg_result;
    int pure_delay_idx, is_pure;
    int nb, unique_idx, blk_tab_idx, blk_def_id, shot_idx;
    int ax_grad_ids[3], ax_def_ids[3], ax;
    float* max_energy = NULL;
    float inst_energy, e, amp;
    int num_raw_alloc, num_exp_alloc;

    if (!diag) { pulseqlib_diagnostic_init(&local_diag); diag = &local_diag; }
    else       pulseqlib_diagnostic_init(diag);

    if (!seq || !desc) { diag->code = PULSEQLIB_ERR_NULL_POINTER; return 0; }

    n_prep_raw = 0; n_main_raw = 0; n_cool_raw = 0; num_raw = 0;
    n_prep = 0; n_main = 0; n_cool = 0;
    num_total = 0; num_unique = 0;
    num_raw_alloc = 0; num_exp_alloc = 0;
    total_blocks = tr->tr_size + tr->num_prep_blocks + tr->num_cooldown_blocks;

    raw_segs = (pulseqlib_tr_segment*)PULSEQLIB_ALLOC(total_blocks * sizeof(pulseqlib_tr_segment));
    if (!raw_segs) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; return 0; }

    /* ---- raw segments per section ---- */
    if (tr->degenerate_prep == 0 && tr->num_prep_blocks > 0) {
        tr_start = 0;
        tr_size  = tr->num_prep_blocks + tr->tr_size;
        seg_result = find_segments_internal(desc, raw_segs, num_raw, diag, &seq->opts, tr_start, tr_size);
        if (seg_result == 0 && PULSEQLIB_FAILED(diag->code)) { PULSEQLIB_FREE(raw_segs); return 0; }
        n_prep_raw = seg_result;
        num_raw += n_prep_raw;
    }

    tr_start = tr->num_prep_blocks;
    tr_size  = tr->tr_size;
    seg_result = find_segments_internal(desc, raw_segs, num_raw, diag, &seq->opts, tr_start, tr_size);
    if (seg_result == 0 && PULSEQLIB_FAILED(diag->code)) { PULSEQLIB_FREE(raw_segs); return 0; }
    n_main_raw = seg_result;
    num_raw += n_main_raw;

    if (tr->degenerate_cooldown == 0 && tr->num_cooldown_blocks > 0) {
        tr_start = seq->num_blocks - tr->num_cooldown_blocks - tr->tr_size;
        tr_size  = tr->num_cooldown_blocks + tr->tr_size;
        seg_result = find_segments_internal(desc, raw_segs, num_raw, diag, &seq->opts, tr_start, tr_size);
        if (seg_result == 0 && PULSEQLIB_FAILED(diag->code)) { PULSEQLIB_FREE(raw_segs); return 0; }
        n_cool_raw = seg_result;
        num_raw += n_cool_raw;
    }

    if (num_raw == 0) {
        diag->code = PULSEQLIB_ERR_SEG_NO_SEGMENTS_FOUND;
        PULSEQLIB_FREE(raw_segs);
        return 0;
    }

    /* populate unique_block_indices */
    for (n = 0; n < num_raw; ++n) {
        raw_segs[n].unique_block_indices = (int*)PULSEQLIB_ALLOC(raw_segs[n].num_blocks * sizeof(int));
        if (!raw_segs[n].unique_block_indices) {
            diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
            num_raw_alloc = n;
            goto fail;
        }
        for (i = 0; i < raw_segs[n].num_blocks; ++i)
            raw_segs[n].unique_block_indices[i] = desc->block_table[raw_segs[n].start_block + i].id;
    }
    num_raw_alloc = num_raw;

    /* ---- strip pure delays ---- */
    max_expanded = total_blocks;
    exp_segs = (pulseqlib_tr_segment*)PULSEQLIB_ALLOC(max_expanded * sizeof(pulseqlib_tr_segment));
    if (!exp_segs) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto fail; }

    offset = 0;
    if (n_prep_raw > 0) {
        n_prep = strip_pure_delays(raw_segs, n_prep_raw, exp_segs + offset, max_expanded - offset, desc->block_table);
        if (n_prep < 0) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto fail; }
        offset += n_prep;
    }

    n_main = strip_pure_delays(raw_segs + n_prep_raw, n_main_raw, exp_segs + offset, max_expanded - offset, desc->block_table);
    if (n_main < 0) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto fail; }
    offset += n_main;

    if (n_cool_raw > 0) {
        n_cool = strip_pure_delays(raw_segs + n_prep_raw + n_main_raw, n_cool_raw, exp_segs + offset, max_expanded - offset, desc->block_table);
        if (n_cool < 0) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto fail; }
        offset += n_cool;
    }

    num_total = n_prep + n_main + n_cool;
    num_exp_alloc = num_total;

    /* raw_segs no longer needed */
    for (n = 0; n < num_raw_alloc; ++n) PULSEQLIB_FREE(raw_segs[n].unique_block_indices);
    PULSEQLIB_FREE(raw_segs); raw_segs = NULL;
    num_raw_alloc = 0;

    /* ---- NAV-aware split and merge (per section, only when PMC enabled) ---- */
    if (desc->enable_pmc) {
        pulseqlib_tr_segment* nav_segs;
        int nav_total = 0, r;

        nav_segs = (pulseqlib_tr_segment*)PULSEQLIB_ALLOC(
            (size_t)max_expanded * sizeof(pulseqlib_tr_segment));
        if (!nav_segs) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto fail; }

        if (n_prep > 0) {
            r = nav_split_merge(exp_segs, n_prep,
                    nav_segs + nav_total, max_expanded - nav_total,
                    desc->block_table, NULL);
            if (r < 0) {
                for (n = 0; n < nav_total; ++n)
                    if (nav_segs[n].unique_block_indices)
                        PULSEQLIB_FREE(nav_segs[n].unique_block_indices);
                PULSEQLIB_FREE(nav_segs);
                diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto fail;
            }
            n_prep = r; nav_total += r;
        }

        r = nav_split_merge(exp_segs + (num_total - n_cool - n_main), n_main,
                nav_segs + nav_total, max_expanded - nav_total,
                desc->block_table, NULL);
        if (r < 0) {
            for (n = 0; n < nav_total; ++n)
                if (nav_segs[n].unique_block_indices)
                    PULSEQLIB_FREE(nav_segs[n].unique_block_indices);
            PULSEQLIB_FREE(nav_segs);
            diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto fail;
        }
        n_main = r; nav_total += r;

        if (n_cool > 0) {
            r = nav_split_merge(exp_segs + (num_total - n_cool), n_cool,
                    nav_segs + nav_total, max_expanded - nav_total,
                    desc->block_table, NULL);
            if (r < 0) {
                for (n = 0; n < nav_total; ++n)
                    if (nav_segs[n].unique_block_indices)
                        PULSEQLIB_FREE(nav_segs[n].unique_block_indices);
                PULSEQLIB_FREE(nav_segs);
                diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto fail;
            }
            n_cool = r; nav_total += r;
        }

        /* replace exp_segs with nav_segs */
        PULSEQLIB_FREE(exp_segs);
        exp_segs = nav_segs;
        num_total = nav_total;
        num_exp_alloc = nav_total;
    }

    /* ---- segment tables ---- */
    desc->segment_table.num_prep_segments     = n_prep;
    desc->segment_table.num_main_segments     = n_main;
    desc->segment_table.num_cooldown_segments = n_cool;
    desc->segment_table.prep_segment_table     = (n_prep > 0) ? (int*)PULSEQLIB_ALLOC(n_prep * sizeof(int)) : NULL;
    desc->segment_table.main_segment_table     = (n_main > 0) ? (int*)PULSEQLIB_ALLOC(n_main * sizeof(int)) : NULL;
    desc->segment_table.cooldown_segment_table = (n_cool > 0) ? (int*)PULSEQLIB_ALLOC(n_cool * sizeof(int)) : NULL;

    uniq_segs = (pulseqlib_tr_segment*)PULSEQLIB_ALLOC(num_total * sizeof(pulseqlib_tr_segment));
    if (!uniq_segs) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto fail; }

    num_unique = 0;
    pure_delay_idx = -1;

    for (n = 0; n < num_total; ++n) {
        is_pure = (exp_segs[n].num_blocks == 1 &&
                   desc->block_table[exp_segs[n].start_block].duration_us >= 0);

        if (is_pure) {
            if (pure_delay_idx == -1) {
                uniq_segs[num_unique].num_blocks  = 1;
                uniq_segs[num_unique].start_block = exp_segs[n].start_block;
                uniq_segs[num_unique].unique_block_indices = (int*)PULSEQLIB_ALLOC(sizeof(int));
                if (!uniq_segs[num_unique].unique_block_indices) {
                    diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
                    goto fail;
                }
                uniq_segs[num_unique].unique_block_indices[0] = exp_segs[n].unique_block_indices[0];
                pure_delay_idx = num_unique;
                num_unique++;
            }
            found = pure_delay_idx;
        } else {
            found = -1;
            for (i = 0; i < num_unique; ++i) {
                if (i == pure_delay_idx) continue;
                if (exp_segs[n].num_blocks == uniq_segs[i].num_blocks &&
                    array_equal(exp_segs[n].unique_block_indices, uniq_segs[i].unique_block_indices, exp_segs[n].num_blocks)) {
                    found = i; break;
                }
            }
            if (found == -1) {
                uniq_segs[num_unique].num_blocks  = exp_segs[n].num_blocks;
                uniq_segs[num_unique].start_block = exp_segs[n].start_block;
                uniq_segs[num_unique].unique_block_indices =
                    (int*)PULSEQLIB_ALLOC(exp_segs[n].num_blocks * sizeof(int));
                if (!uniq_segs[num_unique].unique_block_indices) {
                    diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
                    goto fail;
                }
                for (i = 0; i < exp_segs[n].num_blocks; ++i)
                    uniq_segs[num_unique].unique_block_indices[i] = exp_segs[n].unique_block_indices[i];
                found = num_unique;
                num_unique++;
            }
        }

        if (n < n_prep)
            desc->segment_table.prep_segment_table[n] = found;
        else if (n < n_prep + n_main)
            desc->segment_table.main_segment_table[n - n_prep] = found;
        else
            desc->segment_table.cooldown_segment_table[n - n_prep - n_main] = found;
    }

    desc->segment_table.num_unique_segments = num_unique;
    desc->num_unique_segments = num_unique;

    /* transfer ownership from uniq_segs to desc */
    desc->segment_definitions = (pulseqlib_tr_segment*)PULSEQLIB_ALLOC(num_unique * sizeof(pulseqlib_tr_segment));
    if (!desc->segment_definitions) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto fail; }
    for (i = 0; i < num_unique; ++i)
        desc->segment_definitions[i] = uniq_segs[i];
    PULSEQLIB_FREE(uniq_segs); uniq_segs = NULL;
    /* note: unique_block_indices pointers now owned by desc->segment_definitions */

    /* ---- per-block flags ---- */
    for (i = 0; i < num_unique; ++i) {
        nb = desc->segment_definitions[i].num_blocks;
        desc->segment_definitions[i].has_digitalout = (int*)PULSEQLIB_ALLOC(nb * sizeof(int));
        desc->segment_definitions[i].has_rotation = (int*)PULSEQLIB_ALLOC(nb * sizeof(int));
        desc->segment_definitions[i].norot_flag   = (int*)PULSEQLIB_ALLOC(nb * sizeof(int));
        desc->segment_definitions[i].nopos_flag   = (int*)PULSEQLIB_ALLOC(nb * sizeof(int));
        if (!desc->segment_definitions[i].has_digitalout ||
            !desc->segment_definitions[i].has_rotation ||
            !desc->segment_definitions[i].norot_flag ||
            !desc->segment_definitions[i].nopos_flag) {
            diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
            goto fail;
        }
        for (n = 0; n < nb; ++n) {
            desc->segment_definitions[i].has_digitalout[n] = 0;
            desc->segment_definitions[i].has_rotation[n] = 0;
            desc->segment_definitions[i].norot_flag[n]   = 0;
            desc->segment_definitions[i].nopos_flag[n]   = 0;
        }
        desc->segment_definitions[i].trigger_id = -1;
    }

    max_energy = (float*)PULSEQLIB_ALLOC(num_unique * sizeof(float));
    if (!max_energy) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto fail; }
    for (i = 0; i < num_unique; ++i) {
        max_energy[i] = 0.0f;
        desc->segment_definitions[i].max_energy_start_block = 0;
    }

    for (n = 0; n < num_total; ++n) {
        if (n < n_prep)
            unique_idx = desc->segment_table.prep_segment_table[n];
        else if (n < n_prep + n_main)
            unique_idx = desc->segment_table.main_segment_table[n - n_prep];
        else
            unique_idx = desc->segment_table.cooldown_segment_table[n - n_prep - n_main];

        inst_energy = 0.0f;
        for (b = 0; b < exp_segs[n].num_blocks; ++b) {
            blk_tab_idx = exp_segs[n].start_block + b;
            bte = &desc->block_table[blk_tab_idx];
            blk_def_id = bte->id;
            bdef = &desc->block_definitions[blk_def_id];

            /* Classify trigger: OUTPUT → block-level digitalout,
             *                    INPUT  → segment-level trigger */
            if (bte->digitalout_id != -1 && bte->digitalout_id < desc->num_triggers) {
                const pulseqlib_trigger_event* te = &desc->trigger_events[bte->digitalout_id];
                if (te->trigger_type == PULSEQLIB__TRIGGER_TYPE_OUTPUT) {
                    desc->segment_definitions[unique_idx].has_digitalout[b] = 1;
                } else if (te->trigger_type == PULSEQLIB__TRIGGER_TYPE_INPUT) {
                    int prev = desc->segment_definitions[unique_idx].trigger_id;
                    if (prev >= 0 && prev != bte->digitalout_id) {
                        diag->code = PULSEQLIB_ERR_SEG_MULTIPLE_PHYSIO_TRIGGERS;
                        goto fail;
                    }
                    desc->segment_definitions[unique_idx].trigger_id = bte->digitalout_id;
                }
            }
            if (bte->rotation_id != -1) desc->segment_definitions[unique_idx].has_rotation[b] = 1;
            if (bte->norot_flag)        desc->segment_definitions[unique_idx].norot_flag[b]   = 1;
            if (bte->nopos_flag)        desc->segment_definitions[unique_idx].nopos_flag[b]   = 1;

            ax_grad_ids[0] = bte->gx_id; ax_grad_ids[1] = bte->gy_id; ax_grad_ids[2] = bte->gz_id;
            ax_def_ids[0]  = bdef->gx_id; ax_def_ids[1] = bdef->gy_id; ax_def_ids[2] = bdef->gz_id;

            for (ax = 0; ax < 3; ++ax) {
                if (ax_grad_ids[ax] >= 0 && ax_grad_ids[ax] < desc->grad_table_size &&
                    ax_def_ids[ax]  >= 0 && ax_def_ids[ax]  < desc->num_unique_grads) {
                    amp = desc->grad_table[ax_grad_ids[ax]].amplitude;
                    shot_idx = desc->grad_table[ax_grad_ids[ax]].shot_index;
                    e = desc->grad_definitions[ax_def_ids[ax]].energy[shot_idx];
                    inst_energy += e * amp * amp;
                }
            }
        }
        if (inst_energy > max_energy[unique_idx]) {
            max_energy[unique_idx] = inst_energy;
            desc->segment_definitions[unique_idx].max_energy_start_block = exp_segs[n].start_block;
        }
    }

    PULSEQLIB_FREE(max_energy); max_energy = NULL;

    /* ---- tag segments as NAV; verify at most 1 unique NAV ---- */
    if (desc->enable_pmc) {
        int nav_count = 0;
        for (i = 0; i < num_unique; ++i) {
            int bt0 = desc->segment_definitions[i].start_block;
            desc->segment_definitions[i].is_nav =
                (desc->block_table[bt0].nav_flag) ? 1 : 0;
            if (desc->segment_definitions[i].is_nav) nav_count++;
        }
        if (nav_count > 1) {
            diag->code = PULSEQLIB_ERR_SEG_MULTIPLE_NAV_SEGMENTS;
            goto fail;
        }
    }

    for (n = 0; n < num_exp_alloc; ++n) PULSEQLIB_FREE(exp_segs[n].unique_block_indices);
    PULSEQLIB_FREE(exp_segs); exp_segs = NULL;
    num_exp_alloc = 0;

    diag->code = PULSEQLIB_OK;
    return num_unique;

fail:
    if (max_energy) PULSEQLIB_FREE(max_energy);
    if (uniq_segs) {
        for (i = 0; i < num_unique; ++i)
            if (uniq_segs[i].unique_block_indices) PULSEQLIB_FREE(uniq_segs[i].unique_block_indices);
        PULSEQLIB_FREE(uniq_segs);
    }
    if (exp_segs) {
        for (n = 0; n < num_exp_alloc; ++n)
            if (exp_segs[n].unique_block_indices) PULSEQLIB_FREE(exp_segs[n].unique_block_indices);
        PULSEQLIB_FREE(exp_segs);
    }
    if (raw_segs) {
        for (n = 0; n < num_raw_alloc; ++n)
            if (raw_segs[n].unique_block_indices) PULSEQLIB_FREE(raw_segs[n].unique_block_indices);
        PULSEQLIB_FREE(raw_segs);
    }
    return 0;
}

/* ================================================================== */
/*  Fill scan_table_seg_id from blockTable-based segmentation         */
/* ================================================================== */

/*
 * After get_segments_in_tr succeeds, this function maps its segment
 * tables onto the scan table's seg_id column.
 *
 * For each section (prep, main, cooldown) we build a per-block-position
 * seg_id map from the corresponding segment table, then look up each
 * scan table entry by its block_table index and tr_id.
 *
 * Main blocks always use the main segment map (period tr_size).
 * Prep/cooldown blocks use maps derived from the prep/cooldown segment
 * tables, which cover [0, prep+tr_size) and [cool_start, num_blocks)
 * respectively.
 */
int pulseqlib__fill_scan_seg_id_from_blocktable(
    pulseqlib_sequence_descriptor* desc)
{
    const pulseqlib_tr_descriptor*        tr = &desc->tr_descriptor;
    const pulseqlib_segment_table_result* st = &desc->segment_table;
    int has_nd_prep, has_nd_cool;
    int prep_tr_id, main_tr_id, cool_tr_id;
    int main_len, prep_len, cool_len, cool_start;
    int* main_map = NULL;
    int* prep_map = NULL;
    int* cool_map = NULL;
    int n, i, uid, blk_count, bt_pos, pos;

    if (!desc || !desc->scan_table_seg_id) return PULSEQLIB_ERR_NULL_POINTER;

    has_nd_prep = (tr->num_prep_blocks > 0 && !tr->degenerate_prep);
    has_nd_cool = (tr->num_cooldown_blocks > 0 && !tr->degenerate_cooldown);

    /* Determine tr_id values (same logic as build_scan_table) */
    if (!has_nd_prep && !has_nd_cool) {
        prep_tr_id = -1; main_tr_id = 0; cool_tr_id = -1;
    } else if (has_nd_prep && !has_nd_cool) {
        prep_tr_id = 0;  main_tr_id = 1; cool_tr_id = -1;
    } else if (!has_nd_prep && has_nd_cool) {
        prep_tr_id = -1; main_tr_id = 0; cool_tr_id = 1;
    } else {
        prep_tr_id = 0;  main_tr_id = 1; cool_tr_id = 2;
    }

    /* ---- Build main map [tr_size] ---- */
    main_len = tr->tr_size;
    main_map = (int*)PULSEQLIB_ALLOC((size_t)main_len * sizeof(int));
    if (!main_map) return PULSEQLIB_ERR_ALLOC_FAILED;
    for (i = 0; i < main_len; ++i) main_map[i] = -1;

    blk_count = 0;
    for (n = 0; n < st->num_main_segments; ++n) {
        uid = st->main_segment_table[n];
        for (i = 0; i < desc->segment_definitions[uid].num_blocks; ++i) {
            if (blk_count < main_len) main_map[blk_count] = uid;
            ++blk_count;
        }
    }

    /* ---- Build prep map [num_prep_blocks + tr_size] ---- */
    if (has_nd_prep && st->num_prep_segments > 0) {
        prep_len = tr->num_prep_blocks + tr->tr_size;
        prep_map = (int*)PULSEQLIB_ALLOC((size_t)prep_len * sizeof(int));
        if (!prep_map) { PULSEQLIB_FREE(main_map); return PULSEQLIB_ERR_ALLOC_FAILED; }
        for (i = 0; i < prep_len; ++i) prep_map[i] = -1;

        blk_count = 0;
        for (n = 0; n < st->num_prep_segments; ++n) {
            uid = st->prep_segment_table[n];
            for (i = 0; i < desc->segment_definitions[uid].num_blocks; ++i) {
                if (blk_count < prep_len) prep_map[blk_count] = uid;
                ++blk_count;
            }
        }
    }

    /* ---- Build cool map [tr_size + num_cooldown_blocks] ---- */
    if (has_nd_cool && st->num_cooldown_segments > 0) {
        cool_start = desc->num_blocks - tr->num_cooldown_blocks - tr->tr_size;
        cool_len   = tr->tr_size + tr->num_cooldown_blocks;
        cool_map   = (int*)PULSEQLIB_ALLOC((size_t)cool_len * sizeof(int));
        if (!cool_map) {
            PULSEQLIB_FREE(main_map);
            if (prep_map) PULSEQLIB_FREE(prep_map);
            return PULSEQLIB_ERR_ALLOC_FAILED;
        }
        for (i = 0; i < cool_len; ++i) cool_map[i] = -1;

        blk_count = 0;
        for (n = 0; n < st->num_cooldown_segments; ++n) {
            uid = st->cooldown_segment_table[n];
            for (i = 0; i < desc->segment_definitions[uid].num_blocks; ++i) {
                if (blk_count < cool_len) cool_map[blk_count] = uid;
                ++blk_count;
            }
        }
    } else {
        cool_start = 0;
        cool_len   = 0;
    }

    /* ---- Walk scan table and assign seg_id ---- */
    for (n = 0; n < desc->scan_table_len; ++n) {
        bt_pos = desc->scan_table_block_idx[n];

        if (desc->scan_table_tr_id[n] == prep_tr_id && prep_map) {
            /* Prep block: bt_pos in [0, num_prep_blocks) */
            desc->scan_table_seg_id[n] = (bt_pos >= 0 && bt_pos < prep_len)
                ? prep_map[bt_pos] : -1;
        } else if (desc->scan_table_tr_id[n] == cool_tr_id && cool_map) {
            /* Cooldown block: bt_pos in [num_blocks-cool_blocks, num_blocks) */
            pos = bt_pos - cool_start;
            desc->scan_table_seg_id[n] = (pos >= 0 && pos < cool_len)
                ? cool_map[pos] : -1;
        } else {
            /* Main block: bt_pos in [num_prep_blocks, ...), modular in tr_size */
            pos = (bt_pos - tr->num_prep_blocks) % tr->tr_size;
            if (pos < 0) pos += tr->tr_size;
            desc->scan_table_seg_id[n] = (pos >= 0 && pos < main_len)
                ? main_map[pos] : -1;
        }
    }

    PULSEQLIB_FREE(main_map);
    if (prep_map) PULSEQLIB_FREE(prep_map);
    if (cool_map) PULSEQLIB_FREE(cool_map);
    return PULSEQLIB_OK;
}

/* ================================================================== */
/*  Frequency modulation flags                                        */
/* ================================================================== */

/*
 * Set freq_mod_id in each block_table entry:
 *   >= 0  if the block has (RF or ADC) and at least one gradient axis
 *   -1    otherwise
 *
 * This is a lightweight pass; the actual entry/plan computation is
 * done lazily in pulseqlib_build_freq_mod_library() (freqmod.c).
 *
 * Must be called after unique blocks are resolved.
 */
int pulseqlib__build_freq_mod_flags(pulseqlib_sequence_descriptor* desc)
{
    int n;

    if (!desc) return PULSEQLIB_OK;

    desc->num_freq_mod_defs    = 0;
    desc->freq_mod_definitions = NULL;

    for (n = 0; n < desc->num_blocks; ++n) {
        const pulseqlib_block_table_element* bte = &desc->block_table[n];
        const pulseqlib_block_definition* bdef   = &desc->block_definitions[bte->id];
        int has_rf   = (bdef->rf_id >= 0);
        int has_adc  = (bte->adc_id >= 0);
        int has_grad = (bdef->gx_id >= 0 || bdef->gy_id >= 0 || bdef->gz_id >= 0);

        desc->block_table[n].freq_mod_id = ((has_rf || has_adc) && has_grad) ? 0 : -1;
    }

    return PULSEQLIB_OK;
}

/* ================================================================== */
/*  Label table dry-run                                               */
/* ================================================================== */

/*
 * apply_block_labels --
 *   Scan a block's extension chain and apply LABELSET / LABELINC
 *   operations to the running label state.
 *
 *   state[0..9] = { slc, phs, rep, avg, seg, set, eco, par, lin, acq }
 */
static void apply_block_labels(
    int* state,
    const pulseqlib__seq_file* seq,
    const pulseqlib__raw_block* raw)
{
    int i, type_idx, ref_idx, ext_type, label_value, label_id;

    if (!seq->is_extensions_library_parsed || !seq->extension_lut) return;

    for (i = 0; i < raw->ext_count; ++i) {
        type_idx = raw->ext[i][0];
        ref_idx  = raw->ext[i][1];
        if (type_idx < 0 || type_idx > seq->extension_lut_size) continue;
        ext_type = seq->extension_lut[type_idx];
        if (ref_idx < 0) continue;

        if (ext_type == PULSEQLIB__EXT_LABELSET) {
            if (!seq->labelset_library || ref_idx >= seq->labelset_library_size)
                continue;
            label_value = (int)seq->labelset_library[ref_idx][0];
            label_id    = (int)seq->labelset_library[ref_idx][1];
            switch (label_id) {
                case PULSEQLIB__SLC: state[0] = label_value; break;
                case PULSEQLIB__PHS: state[1] = label_value; break;
                case PULSEQLIB__REP: state[2] = label_value; break;
                case PULSEQLIB__AVG: state[3] = label_value; break;
                case PULSEQLIB__SEG: state[4] = label_value; break;
                case PULSEQLIB__SET: state[5] = label_value; break;
                case PULSEQLIB__ECO: state[6] = label_value; break;
                case PULSEQLIB__PAR: state[7] = label_value; break;
                case PULSEQLIB__LIN: state[8] = label_value; break;
                case PULSEQLIB__ACQ: state[9] = label_value; break;
                default: break;
            }
        } else if (ext_type == PULSEQLIB__EXT_LABELINC) {
            if (!seq->labelinc_library || ref_idx >= seq->labelinc_library_size)
                continue;
            label_value = (int)seq->labelinc_library[ref_idx][0];
            label_id    = (int)seq->labelinc_library[ref_idx][1];
            switch (label_id) {
                case PULSEQLIB__SLC: state[0] += label_value; break;
                case PULSEQLIB__PHS: state[1] += label_value; break;
                case PULSEQLIB__REP: state[2] += label_value; break;
                case PULSEQLIB__AVG: state[3] += label_value; break;
                case PULSEQLIB__SEG: state[4] += label_value; break;
                case PULSEQLIB__SET: state[5] += label_value; break;
                case PULSEQLIB__ECO: state[6] += label_value; break;
                case PULSEQLIB__PAR: state[7] += label_value; break;
                case PULSEQLIB__LIN: state[8] += label_value; break;
                case PULSEQLIB__ACQ: state[9] += label_value; break;
                default: break;
            }
        }
    }
}

/*
 * record_adc_label --
 *   Record the current label state into one row of the label table
 *   and update label_limits min/max tracking.
 *
 *   For GEHC: 3 columns = [lin, slc, eco].
 *   label state indices: lin=8, slc=0, eco=6.
 */
static void record_adc_label(
    int* table_row,
    int num_columns,
    const int* state,
    pulseqlib_label_limits* limits,
    int is_first)
{
    /* GEHC column mapping: col0=lin, col1=slc, col2=eco */
    int lin_val = state[8];
    int slc_val = state[0];
    int eco_val = state[6];

    (void)num_columns; /* always 3 for GEHC */

    table_row[0] = lin_val;
    table_row[1] = slc_val;
    table_row[2] = eco_val;

    if (is_first) {
        limits->lin.min = lin_val; limits->lin.max = lin_val;
        limits->slc.min = slc_val; limits->slc.max = slc_val;
        limits->eco.min = eco_val; limits->eco.max = eco_val;
        /* Also init the other label limits from state */
        limits->phs.min = state[1]; limits->phs.max = state[1];
        limits->rep.min = state[2]; limits->rep.max = state[2];
        limits->avg.min = state[3]; limits->avg.max = state[3];
        limits->seg.min = state[4]; limits->seg.max = state[4];
        limits->set.min = state[5]; limits->set.max = state[5];
        limits->par.min = state[7]; limits->par.max = state[7];
        limits->acq.min = state[9]; limits->acq.max = state[9];
    } else {
        if (lin_val < limits->lin.min) limits->lin.min = lin_val;
        if (lin_val > limits->lin.max) limits->lin.max = lin_val;
        if (slc_val < limits->slc.min) limits->slc.min = slc_val;
        if (slc_val > limits->slc.max) limits->slc.max = slc_val;
        if (eco_val < limits->eco.min) limits->eco.min = eco_val;
        if (eco_val > limits->eco.max) limits->eco.max = eco_val;
        if (state[1] < limits->phs.min) limits->phs.min = state[1];
        if (state[1] > limits->phs.max) limits->phs.max = state[1];
        if (state[2] < limits->rep.min) limits->rep.min = state[2];
        if (state[2] > limits->rep.max) limits->rep.max = state[2];
        if (state[3] < limits->avg.min) limits->avg.min = state[3];
        if (state[3] > limits->avg.max) limits->avg.max = state[3];
        if (state[4] < limits->seg.min) limits->seg.min = state[4];
        if (state[4] > limits->seg.max) limits->seg.max = state[4];
        if (state[5] < limits->set.min) limits->set.min = state[5];
        if (state[5] > limits->set.max) limits->set.max = state[5];
        if (state[7] < limits->par.min) limits->par.min = state[7];
        if (state[7] > limits->par.max) limits->par.max = state[7];
        if (state[9] < limits->acq.min) limits->acq.min = state[9];
        if (state[9] > limits->acq.max) limits->acq.max = state[9];
    }
}

int pulseqlib__build_label_table(
    pulseqlib_sequence_descriptor* desc,
    const pulseqlib__seq_file* seq)
{
    int num_columns, total_adcs, adcs_per_tr;
    int imaging_start, cooldown_start, num_trs;
    int b, rep, entry_idx;
    int state[10];
    int* table;
    pulseqlib__raw_block raw;

    if (!desc || !seq) return PULSEQLIB_ERR_NULL_POINTER;

#if PULSEQLIB_VENDOR != 2
    (void)num_columns; (void)total_adcs; (void)adcs_per_tr;
    (void)imaging_start; (void)cooldown_start; (void)num_trs;
    (void)b; (void)rep; (void)entry_idx;
    (void)state; (void)table; (void)raw;
    return PULSEQLIB_ERR_NOT_IMPLEMENTED;
#else
    num_columns = 3; /* GEHC: [lin, slc, eco] */

    imaging_start  = desc->num_prep_blocks;
    cooldown_start = desc->num_blocks - desc->num_cooldown_blocks;
    num_trs        = desc->tr_descriptor.num_trs;

    /* Count total ADC occurrences */
    total_adcs  = 0;
    adcs_per_tr = 0;

    for (b = 0; b < imaging_start; ++b) {
        if (desc->block_table[b].adc_id >= 0) ++total_adcs;
    }
    for (b = imaging_start; b < cooldown_start; ++b) {
        if (desc->block_table[b].adc_id >= 0) ++adcs_per_tr;
    }
    total_adcs += adcs_per_tr * num_trs;
    for (b = cooldown_start; b < desc->num_blocks; ++b) {
        if (desc->block_table[b].adc_id >= 0) ++total_adcs;
    }

    if (total_adcs == 0) {
        desc->label_num_columns = num_columns;
        desc->label_num_entries = 0;
        desc->label_table       = NULL;
        memset(&desc->label_limits, 0, sizeof(desc->label_limits));
        return PULSEQLIB_OK;
    }

    /* Allocate table */
    table = (int*)PULSEQLIB_ALLOC((size_t)total_adcs * (size_t)num_columns * sizeof(int));
    if (!table) return PULSEQLIB_ERR_ALLOC_FAILED;
    memset(table, 0, (size_t)total_adcs * (size_t)num_columns * sizeof(int));

    /* Initialize running label state to zero */
    memset(state, 0, sizeof(state));
    memset(&desc->label_limits, 0, sizeof(desc->label_limits));
    entry_idx = 0;

    /* 1. Prep blocks */
    for (b = 0; b < imaging_start; ++b) {
        pulseqlib__get_raw_block_content_ids(seq, &raw, b, 1);
        apply_block_labels(state, seq, &raw);
        if (desc->block_table[b].adc_id >= 0) {
            record_adc_label(&table[entry_idx * num_columns],
                             num_columns, state, &desc->label_limits,
                             entry_idx == 0);
            ++entry_idx;
        }
    }

    /* 2. Main blocks x num_trs */
    for (rep = 0; rep < num_trs; ++rep) {
        for (b = imaging_start; b < cooldown_start; ++b) {
            pulseqlib__get_raw_block_content_ids(seq, &raw, b, 1);
            apply_block_labels(state, seq, &raw);
            if (desc->block_table[b].adc_id >= 0) {
                record_adc_label(&table[entry_idx * num_columns],
                                 num_columns, state, &desc->label_limits,
                                 entry_idx == 0);
                ++entry_idx;
            }
        }
    }

    /* 3. Cooldown blocks */
    for (b = cooldown_start; b < desc->num_blocks; ++b) {
        pulseqlib__get_raw_block_content_ids(seq, &raw, b, 1);
        apply_block_labels(state, seq, &raw);
        if (desc->block_table[b].adc_id >= 0) {
            record_adc_label(&table[entry_idx * num_columns],
                             num_columns, state, &desc->label_limits,
                             entry_idx == 0);
            ++entry_idx;
        }
    }

    desc->label_num_columns = num_columns;
    desc->label_num_entries = entry_idx;
    desc->label_table       = table;

    return PULSEQLIB_OK;
#endif
}

/* ================================================================== */
/*  Scan-table-aware segment state machine                            */
/* ================================================================== */

/*
 * Like find_segments_internal but resolves blocks through scan table
 * indirection.  scan_block_idx[pat_start + i] gives the block_table
 * index for position i within the pattern.
 *
 * start_block in returned segs is a SCAN TABLE position.
 */
static int find_segments_on_scan_table(
    const pulseqlib_sequence_descriptor* desc,
    pulseqlib_tr_segment* segs, int offset,
    pulseqlib_diagnostic* diag,
    const pulseqlib_opts* opts,
    const int* scan_block_idx,
    int pat_start, int pat_size)
{
    float max_slew, grad_raster_s, max_allowed;
    int grad_ids[3];
    float phys_first, phys_last;
    float grad_last_cur[3], grad_first_next[3];
    const pulseqlib_block_definition* bdef;
    const pulseqlib_grad_definition* gdef;
    int bdef_id, shot_idx, bt_idx, prev_bt;
    int* seg_starts = NULL;
    int* seg_sizes  = NULL;
    int num_seg, seg_start;
    int state, cand_before_rf, saved_cand, has_saved_cand;
    int has_rf, has_adc, is_cand;
    int nb, n, i;

    max_slew = opts->max_slew_hz_per_m_per_s;
    grad_raster_s = desc->grad_raster_us * 1e-6f;
    max_allowed = max_slew * grad_raster_s;
    nb = pat_size;

    seg_starts = (int*)PULSEQLIB_ALLOC(nb * sizeof(int));
    seg_sizes  = (int*)PULSEQLIB_ALLOC(nb * sizeof(int));
    if (!seg_starts || !seg_sizes) {
        if (seg_starts) PULSEQLIB_FREE(seg_starts);
        if (seg_sizes)  PULSEQLIB_FREE(seg_sizes);
        if (diag) diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        return 0;
    }

    /* first block gradient check */
    bt_idx = scan_block_idx[pat_start];
    bdef_id = desc->block_table[bt_idx].id;
    bdef = &desc->block_definitions[bdef_id];
    grad_ids[0] = bdef->gx_id; grad_ids[1] = bdef->gy_id; grad_ids[2] = bdef->gz_id;
    for (i = 0; i < 3; ++i) {
        if (grad_ids[i] < 0) continue;
        gdef = &desc->grad_definitions[grad_ids[i]];
        for (shot_idx = 0; shot_idx < gdef->num_shots; ++shot_idx) {
            phys_first = gdef->first_value[shot_idx] * gdef->max_amplitude[shot_idx];
            if ((float)fabs(phys_first) > max_allowed) {
                if (diag) {
                    diag->code = PULSEQLIB_ERR_SEG_NONZERO_START_GRAD;
                    pulseqlib__diag_printf(diag, " scan_pos=%d block=%d", pat_start, bt_idx);
                    pulseqlib__diag_printf(diag, " channel=%d", i);
                }
                PULSEQLIB_FREE(seg_starts); PULSEQLIB_FREE(seg_sizes);
                return 0;
            }
        }
    }

    /* last block gradient check */
    bt_idx = scan_block_idx[pat_start + nb - 1];
    bdef_id = desc->block_table[bt_idx].id;
    bdef = &desc->block_definitions[bdef_id];
    grad_ids[0] = bdef->gx_id; grad_ids[1] = bdef->gy_id; grad_ids[2] = bdef->gz_id;
    for (i = 0; i < 3; ++i) {
        if (grad_ids[i] < 0) continue;
        gdef = &desc->grad_definitions[grad_ids[i]];
        for (shot_idx = 0; shot_idx < gdef->num_shots; ++shot_idx) {
            phys_last = gdef->last_value[shot_idx] * gdef->max_amplitude[shot_idx];
            if ((float)fabs(phys_last) > max_allowed) {
                if (diag) {
                    diag->code = PULSEQLIB_ERR_SEG_NONZERO_END_GRAD;
                    pulseqlib__diag_printf(diag, " scan_pos=%d block=%d", pat_start + nb - 1, bt_idx);
                    pulseqlib__diag_printf(diag, " channel=%d", i);
                }
                PULSEQLIB_FREE(seg_starts); PULSEQLIB_FREE(seg_sizes);
                return 0;
            }
        }
    }

    /* state machine */
    num_seg = 0;
    seg_start = pat_start;
    state = SEGSTATE_SEEKING_FIRST_ADC;
    cand_before_rf = -1;
    saved_cand = -1;
    has_saved_cand = 0;

    for (n = pat_start; n < pat_start + nb; ++n) {
        bt_idx = scan_block_idx[n];
        is_cand = 0;
        if (n > pat_start) {
            prev_bt = scan_block_idx[n - 1];
            is_cand = 1;

            bdef_id = desc->block_table[prev_bt].id;
            bdef = &desc->block_definitions[bdef_id];
            grad_ids[0] = bdef->gx_id; grad_ids[1] = bdef->gy_id; grad_ids[2] = bdef->gz_id;
            for (i = 0; i < 3; ++i) {
                grad_last_cur[i] = 0.0f;
                if (grad_ids[i] >= 0) {
                    gdef = &desc->grad_definitions[grad_ids[i]];
                    for (shot_idx = 0; shot_idx < gdef->num_shots; ++shot_idx) {
                        phys_last = gdef->last_value[shot_idx] * gdef->max_amplitude[shot_idx];
                        if ((float)fabs(phys_last) > (float)fabs(grad_last_cur[i]))
                            grad_last_cur[i] = phys_last;
                    }
                }
            }

            bdef_id = desc->block_table[bt_idx].id;
            bdef = &desc->block_definitions[bdef_id];
            grad_ids[0] = bdef->gx_id; grad_ids[1] = bdef->gy_id; grad_ids[2] = bdef->gz_id;
            for (i = 0; i < 3; ++i) {
                grad_first_next[i] = 0.0f;
                if (grad_ids[i] >= 0) {
                    gdef = &desc->grad_definitions[grad_ids[i]];
                    for (shot_idx = 0; shot_idx < gdef->num_shots; ++shot_idx) {
                        phys_first = gdef->first_value[shot_idx] * gdef->max_amplitude[shot_idx];
                        if ((float)fabs(phys_first) > (float)fabs(grad_first_next[i]))
                            grad_first_next[i] = phys_first;
                    }
                }
            }

            for (i = 0; i < 3; ++i) {
                if ((float)fabs(grad_last_cur[i]) > max_allowed ||
                    (float)fabs(grad_first_next[i]) > max_allowed) {
                    is_cand = 0; break;
                }
            }
        }

        has_rf  = (desc->block_definitions[desc->block_table[bt_idx].id].rf_id >= 0);
        has_adc = (desc->block_table[bt_idx].adc_id >= 0);

        if (state == SEGSTATE_SEEKING_FIRST_ADC) {
            if (is_cand) saved_cand = n;
            if (has_rf)  { cand_before_rf = saved_cand; saved_cand = -1; }
            if (has_adc) {
                if (cand_before_rf > seg_start) {
                    seg_starts[num_seg] = seg_start;
                    seg_sizes[num_seg]  = cand_before_rf - seg_start;
                    num_seg++;
                    seg_start = cand_before_rf;
                }
                state = SEGSTATE_SEEKING_BOUNDARY;
                has_saved_cand = 0;
                saved_cand = -1;
            }
        } else if (state == SEGSTATE_SEEKING_BOUNDARY) {
            if (is_cand) { saved_cand = n; has_saved_cand = 1; }
            if (has_rf) {
                if (has_saved_cand) {
                    seg_starts[num_seg] = seg_start;
                    seg_sizes[num_seg]  = saved_cand - seg_start;
                    num_seg++;
                    seg_start = saved_cand;
                    has_saved_cand = 0;
                    saved_cand = -1;
                } else {
                    state = SEGSTATE_OPTIMIZED_MODE;
                }
            }
        }
        /* SEGSTATE_OPTIMIZED_MODE: no action */
    }

    seg_starts[num_seg] = seg_start;
    seg_sizes[num_seg]  = pat_start + nb - seg_start;
    num_seg++;

    for (i = 0; i < num_seg; ++i) {
        segs[offset + i].start_block = seg_starts[i];
        segs[offset + i].num_blocks  = seg_sizes[i];
        segs[offset + i].unique_block_indices = NULL;
    }

    PULSEQLIB_FREE(seg_starts); PULSEQLIB_FREE(seg_sizes);
    return num_seg;
}

/* ================================================================== */
/*  Strip pure delays (scan table variant)                            */
/* ================================================================== */

static int strip_pure_delays_scan(
    const pulseqlib_tr_segment* raw_segs, int num_raw,
    pulseqlib_tr_segment* out, int max_out,
    const pulseqlib_block_table_element* bt,
    const int* scan_block_idx)
{
    int num_out = 0;
    int s, i, n_blk, bt_idx;
    int leading, trailing, core_start, core_end, core_size;
    const int* idx;

    for (s = 0; s < num_raw; ++s) {
        n_blk = raw_segs[s].num_blocks;
        idx   = raw_segs[s].unique_block_indices;
        if (n_blk == 0 || !idx) continue;

        leading = 0;
        for (i = 0; i < n_blk; ++i) {
            bt_idx = scan_block_idx[raw_segs[s].start_block + i];
            if (bt[bt_idx].duration_us >= 0) leading++;
            else break;
        }
        trailing = 0;
        for (i = n_blk - 1; i >= leading; --i) {
            bt_idx = scan_block_idx[raw_segs[s].start_block + i];
            if (bt[bt_idx].duration_us >= 0) trailing++;
            else break;
        }
        core_start = leading;
        core_end   = n_blk - trailing;

        for (i = 0; i < leading; ++i) {
            if (num_out >= max_out) return -1;
            out[num_out].start_block = raw_segs[s].start_block + i;
            out[num_out].num_blocks  = 1;
            out[num_out].unique_block_indices = (int*)PULSEQLIB_ALLOC(sizeof(int));
            if (!out[num_out].unique_block_indices) return -1;
            out[num_out].unique_block_indices[0] = idx[i];
            num_out++;
        }
        if (core_end > core_start) {
            core_size = core_end - core_start;
            if (num_out >= max_out) return -1;
            out[num_out].start_block = raw_segs[s].start_block + core_start;
            out[num_out].num_blocks  = core_size;
            out[num_out].unique_block_indices = (int*)PULSEQLIB_ALLOC(core_size * sizeof(int));
            if (!out[num_out].unique_block_indices) return -1;
            for (i = 0; i < core_size; ++i)
                out[num_out].unique_block_indices[i] = idx[core_start + i];
            num_out++;
        }
        for (i = 0; i < trailing; ++i) {
            if (num_out >= max_out) return -1;
            out[num_out].start_block = raw_segs[s].start_block + core_end + i;
            out[num_out].num_blocks  = 1;
            out[num_out].unique_block_indices = (int*)PULSEQLIB_ALLOC(sizeof(int));
            if (!out[num_out].unique_block_indices) return -1;
            out[num_out].unique_block_indices[0] = idx[core_end + i];
            num_out++;
        }
    }
    return num_out;
}

/* ================================================================== */
/*  Scan-table-based segment detection                                */
/* ================================================================== */

int pulseqlib__get_scan_table_segments(
    pulseqlib_sequence_descriptor* desc,
    pulseqlib_diagnostic* diag,
    const pulseqlib_opts* opts)
{
    pulseqlib_diagnostic local_diag;
    int* scan_pat         = NULL;
    int* pattern_seg_id   = NULL;
    float* max_energy     = NULL;
    pulseqlib_tr_segment* raw_segs  = NULL;
    pulseqlib_tr_segment* exp_segs  = NULL;
    pulseqlib_tr_segment* uniq_segs = NULL;
    int scan_len, scan_tr_size;
    int num_raw, num_total, num_unique;
    int n, b, i, found;
    int num_raw_alloc, num_exp_alloc;
    int pure_delay_idx, is_pure;
    int seg_result, max_expanded;
    int nb, unique_idx, blk_tab_idx, blk_def_id, shot_idx;
    int ax_grad_ids[3], ax_def_ids[3], ax;
    float inst_energy, e, amp;
    const pulseqlib_block_table_element* bte;
    const pulseqlib_block_definition* bdef;

    if (!diag) { pulseqlib_diagnostic_init(&local_diag); diag = &local_diag; }
    else       pulseqlib_diagnostic_init(diag);

    if (!desc || !opts) {
        diag->code = PULSEQLIB_ERR_NULL_POINTER;
        return 0;
    }
    if (desc->scan_table_len <= 0 || !desc->scan_table_block_idx) {
        diag->code = PULSEQLIB_ERR_INVALID_ARGUMENT;
        return 0;
    }

    scan_len = desc->scan_table_len;
    num_raw = 0; num_total = 0; num_unique = 0;
    num_raw_alloc = 0; num_exp_alloc = 0;

    /* ---- 1. Map scan table to block-def-ID pattern ---- */
    scan_pat = (int*)PULSEQLIB_ALLOC((size_t)scan_len * sizeof(int));
    if (!scan_pat) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; return 0; }
    for (n = 0; n < scan_len; ++n)
        scan_pat[n] = desc->block_table[desc->scan_table_block_idx[n]].id;

    /* ---- 2. Find repeating pattern ---- */
    scan_tr_size = first_repeating_segment(scan_pat, scan_len);

    /* Verify tiling */
    for (n = 0; n < scan_len; ++n) {
        if (scan_pat[n] != scan_pat[n % scan_tr_size]) {
            /* Pattern does not tile: treat entire scan table as one period */
            scan_tr_size = scan_len;
            break;
        }
    }

    /* ---- 3. Find segments on the scan TR pattern ---- */
    raw_segs = (pulseqlib_tr_segment*)PULSEQLIB_ALLOC(
        (size_t)scan_tr_size * sizeof(pulseqlib_tr_segment));
    if (!raw_segs) {
        PULSEQLIB_FREE(scan_pat);
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        return 0;
    }

    seg_result = find_segments_on_scan_table(
        desc, raw_segs, 0, diag, opts,
        desc->scan_table_block_idx, 0, scan_tr_size);
    if (seg_result == 0 && PULSEQLIB_FAILED(diag->code)) {
        PULSEQLIB_FREE(scan_pat);
        PULSEQLIB_FREE(raw_segs);
        return 0;
    }
    num_raw = seg_result;

    /* ---- 4. Populate unique_block_indices ---- */
    for (n = 0; n < num_raw; ++n) {
        raw_segs[n].unique_block_indices =
            (int*)PULSEQLIB_ALLOC(raw_segs[n].num_blocks * sizeof(int));
        if (!raw_segs[n].unique_block_indices) {
            diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
            num_raw_alloc = n;
            goto scan_seg_fail;
        }
        for (i = 0; i < raw_segs[n].num_blocks; ++i)
            raw_segs[n].unique_block_indices[i] =
                scan_pat[raw_segs[n].start_block + i];
    }
    num_raw_alloc = num_raw;

    /* ---- 5. Strip pure delays ---- */
    max_expanded = scan_tr_size;
    exp_segs = (pulseqlib_tr_segment*)PULSEQLIB_ALLOC(
        (size_t)max_expanded * sizeof(pulseqlib_tr_segment));
    if (!exp_segs) {
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        goto scan_seg_fail;
    }

    num_total = strip_pure_delays_scan(
        raw_segs, num_raw, exp_segs, max_expanded,
        desc->block_table, desc->scan_table_block_idx);
    if (num_total < 0) {
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        goto scan_seg_fail;
    }
    num_exp_alloc = num_total;

    /* Free raw segments */
    for (n = 0; n < num_raw_alloc; ++n)
        PULSEQLIB_FREE(raw_segs[n].unique_block_indices);
    PULSEQLIB_FREE(raw_segs); raw_segs = NULL;
    num_raw_alloc = 0;

    if (num_total == 0) {
        diag->code = PULSEQLIB_ERR_SEG_NO_SEGMENTS_FOUND;
        goto scan_seg_fail;
    }

    /* ---- 5b. NAV-aware split and merge (only when PMC enabled) ---- */
    if (desc->enable_pmc) {
        pulseqlib_tr_segment* nav_segs;
        int nav_total;

        nav_segs = (pulseqlib_tr_segment*)PULSEQLIB_ALLOC(
            (size_t)max_expanded * sizeof(pulseqlib_tr_segment));
        if (!nav_segs) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto scan_seg_fail; }

        nav_total = nav_split_merge(exp_segs, num_total,
                nav_segs, max_expanded,
                desc->block_table, desc->scan_table_block_idx);
        if (nav_total < 0) {
            PULSEQLIB_FREE(nav_segs);
            diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto scan_seg_fail;
        }

        PULSEQLIB_FREE(exp_segs);
        exp_segs = nav_segs;
        num_total = nav_total;
        num_exp_alloc = nav_total;
    }

    /* ---- 6. Deduplicate segments ---- */
    uniq_segs = (pulseqlib_tr_segment*)PULSEQLIB_ALLOC(
        (size_t)num_total * sizeof(pulseqlib_tr_segment));
    if (!uniq_segs) {
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        goto scan_seg_fail;
    }

    /* Build main segment table (single flat table, no prep/cooldown split) */
    desc->segment_table.num_prep_segments     = 0;
    desc->segment_table.prep_segment_table    = NULL;
    desc->segment_table.num_cooldown_segments = 0;
    desc->segment_table.cooldown_segment_table = NULL;
    desc->segment_table.num_main_segments     = num_total;
    desc->segment_table.main_segment_table    =
        (int*)PULSEQLIB_ALLOC((size_t)num_total * sizeof(int));
    if (!desc->segment_table.main_segment_table) {
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        goto scan_seg_fail;
    }

    num_unique = 0;
    pure_delay_idx = -1;

    for (n = 0; n < num_total; ++n) {
        is_pure = (exp_segs[n].num_blocks == 1 &&
                   desc->block_table[
                       desc->scan_table_block_idx[exp_segs[n].start_block]
                   ].duration_us >= 0);

        if (is_pure) {
            if (pure_delay_idx == -1) {
                uniq_segs[num_unique].num_blocks  = 1;
                uniq_segs[num_unique].start_block = exp_segs[n].start_block;
                uniq_segs[num_unique].unique_block_indices =
                    (int*)PULSEQLIB_ALLOC(sizeof(int));
                if (!uniq_segs[num_unique].unique_block_indices) {
                    diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
                    goto scan_seg_fail;
                }
                uniq_segs[num_unique].unique_block_indices[0] =
                    exp_segs[n].unique_block_indices[0];
                pure_delay_idx = num_unique;
                num_unique++;
            }
            found = pure_delay_idx;
        } else {
            found = -1;
            for (i = 0; i < num_unique; ++i) {
                if (i == pure_delay_idx) continue;
                if (exp_segs[n].num_blocks == uniq_segs[i].num_blocks &&
                    array_equal(exp_segs[n].unique_block_indices,
                                uniq_segs[i].unique_block_indices,
                                exp_segs[n].num_blocks)) {
                    found = i; break;
                }
            }
            if (found == -1) {
                uniq_segs[num_unique].num_blocks  = exp_segs[n].num_blocks;
                uniq_segs[num_unique].start_block = exp_segs[n].start_block;
                uniq_segs[num_unique].unique_block_indices =
                    (int*)PULSEQLIB_ALLOC(exp_segs[n].num_blocks * sizeof(int));
                if (!uniq_segs[num_unique].unique_block_indices) {
                    diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
                    goto scan_seg_fail;
                }
                for (i = 0; i < exp_segs[n].num_blocks; ++i)
                    uniq_segs[num_unique].unique_block_indices[i] =
                        exp_segs[n].unique_block_indices[i];
                found = num_unique;
                num_unique++;
            }
        }

        desc->segment_table.main_segment_table[n] = found;
    }

    desc->segment_table.num_unique_segments = num_unique;
    desc->num_unique_segments = num_unique;

    /* ---- 7. Transfer segment definitions ---- */
    desc->segment_definitions = (pulseqlib_tr_segment*)PULSEQLIB_ALLOC(
        (size_t)num_unique * sizeof(pulseqlib_tr_segment));
    if (!desc->segment_definitions) {
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        goto scan_seg_fail;
    }
    for (i = 0; i < num_unique; ++i) {
        desc->segment_definitions[i] = uniq_segs[i];
        /* Convert start_block from scan table pos to block_table index */
        desc->segment_definitions[i].start_block =
            desc->scan_table_block_idx[uniq_segs[i].start_block];
    }
    PULSEQLIB_FREE(uniq_segs); uniq_segs = NULL;

    /* ---- 8. Per-block flags ---- */
    for (i = 0; i < num_unique; ++i) {
        nb = desc->segment_definitions[i].num_blocks;
        desc->segment_definitions[i].has_digitalout = (int*)PULSEQLIB_ALLOC(nb * sizeof(int));
        desc->segment_definitions[i].has_rotation = (int*)PULSEQLIB_ALLOC(nb * sizeof(int));
        desc->segment_definitions[i].norot_flag   = (int*)PULSEQLIB_ALLOC(nb * sizeof(int));
        desc->segment_definitions[i].nopos_flag   = (int*)PULSEQLIB_ALLOC(nb * sizeof(int));
        if (!desc->segment_definitions[i].has_digitalout ||
            !desc->segment_definitions[i].has_rotation ||
            !desc->segment_definitions[i].norot_flag ||
            !desc->segment_definitions[i].nopos_flag) {
            diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
            goto scan_seg_fail;
        }
        for (n = 0; n < nb; ++n) {
            desc->segment_definitions[i].has_digitalout[n] = 0;
            desc->segment_definitions[i].has_rotation[n] = 0;
            desc->segment_definitions[i].norot_flag[n]   = 0;
            desc->segment_definitions[i].nopos_flag[n]   = 0;
        }
        desc->segment_definitions[i].trigger_id = -1;
    }

    /* ---- 9. Walk expanded segments, populate flags + max energy ---- */
    max_energy = (float*)PULSEQLIB_ALLOC((size_t)num_unique * sizeof(float));
    if (!max_energy) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto scan_seg_fail; }
    for (i = 0; i < num_unique; ++i) {
        max_energy[i] = 0.0f;
        desc->segment_definitions[i].max_energy_start_block = 0;
    }

    for (n = 0; n < num_total; ++n) {
        unique_idx = desc->segment_table.main_segment_table[n];
        inst_energy = 0.0f;

        for (b = 0; b < exp_segs[n].num_blocks; ++b) {
            blk_tab_idx = desc->scan_table_block_idx[exp_segs[n].start_block + b];
            bte = &desc->block_table[blk_tab_idx];
            blk_def_id = bte->id;
            bdef = &desc->block_definitions[blk_def_id];

            /* Classify trigger: OUTPUT → block-level digitalout,
             *                    INPUT  → segment-level trigger */
            if (bte->digitalout_id != -1 && bte->digitalout_id < desc->num_triggers) {
                const pulseqlib_trigger_event* te = &desc->trigger_events[bte->digitalout_id];
                if (te->trigger_type == PULSEQLIB__TRIGGER_TYPE_OUTPUT) {
                    desc->segment_definitions[unique_idx].has_digitalout[b] = 1;
                } else if (te->trigger_type == PULSEQLIB__TRIGGER_TYPE_INPUT) {
                    int prev = desc->segment_definitions[unique_idx].trigger_id;
                    if (prev >= 0 && prev != bte->digitalout_id) {
                        diag->code = PULSEQLIB_ERR_SEG_MULTIPLE_PHYSIO_TRIGGERS;
                        goto scan_seg_fail;
                    }
                    desc->segment_definitions[unique_idx].trigger_id = bte->digitalout_id;
                }
            }
            if (bte->rotation_id != -1)
                desc->segment_definitions[unique_idx].has_rotation[b] = 1;
            if (bte->norot_flag)
                desc->segment_definitions[unique_idx].norot_flag[b]   = 1;
            if (bte->nopos_flag)
                desc->segment_definitions[unique_idx].nopos_flag[b]   = 1;

            ax_grad_ids[0] = bte->gx_id;
            ax_grad_ids[1] = bte->gy_id;
            ax_grad_ids[2] = bte->gz_id;
            ax_def_ids[0]  = bdef->gx_id;
            ax_def_ids[1]  = bdef->gy_id;
            ax_def_ids[2]  = bdef->gz_id;

            for (ax = 0; ax < 3; ++ax) {
                if (ax_grad_ids[ax] >= 0 &&
                    ax_grad_ids[ax] < desc->grad_table_size &&
                    ax_def_ids[ax]  >= 0 &&
                    ax_def_ids[ax]  < desc->num_unique_grads) {
                    amp = desc->grad_table[ax_grad_ids[ax]].amplitude;
                    shot_idx = desc->grad_table[ax_grad_ids[ax]].shot_index;
                    e = desc->grad_definitions[ax_def_ids[ax]].energy[shot_idx];
                    inst_energy += e * amp * amp;
                }
            }
        }

        if (inst_energy > max_energy[unique_idx]) {
            max_energy[unique_idx] = inst_energy;
            desc->segment_definitions[unique_idx].max_energy_start_block =
                desc->scan_table_block_idx[exp_segs[n].start_block];
        }
    }

    PULSEQLIB_FREE(max_energy); max_energy = NULL;

    /* ---- tag segments as NAV; verify at most 1 unique NAV ---- */
    if (desc->enable_pmc) {
        int nav_count = 0;
        for (i = 0; i < num_unique; ++i) {
            /* start_block already resolved to block_table index in step 7 */
            int bt0 = desc->segment_definitions[i].start_block;
            desc->segment_definitions[i].is_nav =
                (desc->block_table[bt0].nav_flag) ? 1 : 0;
            if (desc->segment_definitions[i].is_nav) nav_count++;
        }
        if (nav_count > 1) {
            diag->code = PULSEQLIB_ERR_SEG_MULTIPLE_NAV_SEGMENTS;
            goto scan_seg_fail;
        }
    }

    /* ---- 10. Build pattern_seg_id and fill scan_table_seg_id ---- */
    pattern_seg_id = (int*)PULSEQLIB_ALLOC((size_t)scan_tr_size * sizeof(int));
    if (!pattern_seg_id) {
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        goto scan_seg_fail;
    }
    /* Init to -1 */
    for (n = 0; n < scan_tr_size; ++n) pattern_seg_id[n] = -1;

    /* Walk expanded segments and assign each position its unique seg id */
    for (n = 0; n < num_total; ++n) {
        unique_idx = desc->segment_table.main_segment_table[n];
        for (b = 0; b < exp_segs[n].num_blocks; ++b) {
            i = exp_segs[n].start_block + b;
            if (i >= 0 && i < scan_tr_size)
                pattern_seg_id[i] = unique_idx;
        }
    }

    /* Tile pattern across full scan table */
    for (n = 0; n < scan_len; ++n)
        desc->scan_table_seg_id[n] = pattern_seg_id[n % scan_tr_size];

    PULSEQLIB_FREE(pattern_seg_id); pattern_seg_id = NULL;

    /* ---- Cleanup ---- */
    for (n = 0; n < num_exp_alloc; ++n)
        PULSEQLIB_FREE(exp_segs[n].unique_block_indices);
    PULSEQLIB_FREE(exp_segs); exp_segs = NULL;
    PULSEQLIB_FREE(scan_pat); scan_pat = NULL;

    diag->code = PULSEQLIB_OK;
    return num_unique;

scan_seg_fail:
    if (pattern_seg_id) PULSEQLIB_FREE(pattern_seg_id);
    if (max_energy) PULSEQLIB_FREE(max_energy);
    if (uniq_segs) {
        for (i = 0; i < num_unique; ++i)
            if (uniq_segs[i].unique_block_indices)
                PULSEQLIB_FREE(uniq_segs[i].unique_block_indices);
        PULSEQLIB_FREE(uniq_segs);
    }
    if (exp_segs) {
        for (n = 0; n < num_exp_alloc; ++n)
            if (exp_segs[n].unique_block_indices)
                PULSEQLIB_FREE(exp_segs[n].unique_block_indices);
        PULSEQLIB_FREE(exp_segs);
    }
    if (raw_segs) {
        for (n = 0; n < num_raw_alloc; ++n)
            if (raw_segs[n].unique_block_indices)
                PULSEQLIB_FREE(raw_segs[n].unique_block_indices);
        PULSEQLIB_FREE(raw_segs);
    }
    if (scan_pat) PULSEQLIB_FREE(scan_pat);
    return 0;
}
