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
        desc->segment_definitions[i].has_trigger  = (int*)PULSEQLIB_ALLOC(nb * sizeof(int));
        desc->segment_definitions[i].has_rotation = (int*)PULSEQLIB_ALLOC(nb * sizeof(int));
        desc->segment_definitions[i].norot_flag   = (int*)PULSEQLIB_ALLOC(nb * sizeof(int));
        desc->segment_definitions[i].nopos_flag   = (int*)PULSEQLIB_ALLOC(nb * sizeof(int));
        if (!desc->segment_definitions[i].has_trigger ||
            !desc->segment_definitions[i].has_rotation ||
            !desc->segment_definitions[i].norot_flag ||
            !desc->segment_definitions[i].nopos_flag) {
            diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
            goto fail;
        }
        for (n = 0; n < nb; ++n) {
            desc->segment_definitions[i].has_trigger[n]  = 0;
            desc->segment_definitions[i].has_rotation[n] = 0;
            desc->segment_definitions[i].norot_flag[n]   = 0;
            desc->segment_definitions[i].nopos_flag[n]   = 0;
        }
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

            if (bte->trigger_id  != -1) desc->segment_definitions[unique_idx].has_trigger[b]  = 1;
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
/*  Frequency modulation library                                      */
/* ================================================================== */

#define FREQ_MOD_DEF_COLS 5    /* (rf_def_id, adc_def_id, gx_def_id, gy_def_id, gz_def_id) */

/*
 * Build a freq-mod waveform for a single unique (rf, adc, gx, gy, gz) tuple.
 *
 * The waveform covers the active event region [active_start_us, active_end_us]
 * within the block, interpolated to a uniform raster at grad_raster_us.
 * Each axis is peak-normalized (as stored in shapes / grad definitions).
 * Zero-padding is applied where the gradient doesn't cover the active region.
 *
 * ref_time_us is relative to active_start_us (e.g. isodelay for RF, kzero for ADC).
 */
static int build_freq_mod_for_block(
    const pulseqlib_sequence_descriptor* desc,
    int bdef_idx,
    float active_start_us, float active_end_us,
    float ref_time_us,
    pulseqlib_freq_mod_definition* fmod)
{
    float grad_raster_us = desc->grad_raster_us;
    const pulseqlib_block_definition* bdef = &desc->block_definitions[bdef_idx];
    int grad_ids[3], axis;
    const pulseqlib_grad_definition* gdef;
    int shape_id, time_shape_id, num_samples, has_time_shape;
    pulseqlib_shape_arbitrary decomp_wave, decomp_time;
    float* raw_time = NULL;
    float* raw_wave = NULL;
    int raw_n, idx, i, j;
    float delay_us, rise_us, flat_us, fall_us;
    float active_dur_us;
    float* uniform_t = NULL;

    active_dur_us = active_end_us - active_start_us;
    if (active_dur_us <= 0.0f || grad_raster_us <= 0.0f)
        return PULSEQLIB_ERR_INVALID_ARGUMENT;

    grad_ids[0] = bdef->gx_id;
    grad_ids[1] = bdef->gy_id;
    grad_ids[2] = bdef->gz_id;

    fmod->raster_us   = grad_raster_us;
    fmod->duration_us = active_dur_us;
    fmod->num_samples = (int)(active_dur_us / grad_raster_us) + 1;
    if (fmod->num_samples < 2) fmod->num_samples = 2;

    fmod->waveform_gx = (float*)PULSEQLIB_ALLOC((size_t)fmod->num_samples * sizeof(float));
    fmod->waveform_gy = (float*)PULSEQLIB_ALLOC((size_t)fmod->num_samples * sizeof(float));
    fmod->waveform_gz = (float*)PULSEQLIB_ALLOC((size_t)fmod->num_samples * sizeof(float));
    if (!fmod->waveform_gx || !fmod->waveform_gy || !fmod->waveform_gz)
        goto fmod_fail;

    /* Oversized temp arrays for building non-uniform waveform */
    raw_time = (float*)PULSEQLIB_ALLOC((size_t)(fmod->num_samples + 512) * sizeof(float));
    raw_wave = (float*)PULSEQLIB_ALLOC((size_t)(fmod->num_samples + 512) * sizeof(float));
    uniform_t = (float*)PULSEQLIB_ALLOC((size_t)fmod->num_samples * sizeof(float));
    if (!raw_time || !raw_wave || !uniform_t)
        goto fmod_fail;

    /* Build uniform time axis for the active region [0, active_dur_us] */
    for (i = 0; i < fmod->num_samples; ++i)
        uniform_t[i] = (float)i * grad_raster_us;

    for (axis = 0; axis < 3; ++axis) {
        float* out_wave = (axis == 0) ? fmod->waveform_gx :
                          (axis == 1) ? fmod->waveform_gy : fmod->waveform_gz;

        if (grad_ids[axis] < 0 || grad_ids[axis] >= desc->num_unique_grads) {
            /* No gradient on this axis: zero waveform */
            for (i = 0; i < fmod->num_samples; ++i) out_wave[i] = 0.0f;
            fmod->ref_integral[axis] = 0.0f;
            continue;
        }

        gdef = &desc->grad_definitions[grad_ids[axis]];
        decomp_wave.samples = NULL;
        decomp_time.samples = NULL;
        idx = 0;
        delay_us = (float)gdef->delay;

        if (gdef->type == 0) {
            /* Trapezoid: build piecewise-linear in block coordinates,
             * then shift to active-region coordinates */
            rise_us = (float)gdef->rise_time_or_unused;
            flat_us = (float)gdef->flat_time_or_unused;
            fall_us = (float)gdef->fall_time_or_num_uncompressed_samples;

            if (flat_us > 0.0f) {
                raw_time[idx] = delay_us - active_start_us;
                raw_wave[idx] = 0.0f; idx++;
                raw_time[idx] = delay_us + rise_us - active_start_us;
                raw_wave[idx] = 1.0f; idx++;
                raw_time[idx] = delay_us + rise_us + flat_us - active_start_us;
                raw_wave[idx] = 1.0f; idx++;
                raw_time[idx] = delay_us + rise_us + flat_us + fall_us - active_start_us;
                raw_wave[idx] = 0.0f; idx++;
            } else {
                raw_time[idx] = delay_us - active_start_us;
                raw_wave[idx] = 0.0f; idx++;
                raw_time[idx] = delay_us + rise_us - active_start_us;
                raw_wave[idx] = 1.0f; idx++;
                raw_time[idx] = delay_us + rise_us + fall_us - active_start_us;
                raw_wave[idx] = 0.0f; idx++;
            }
        } else {
            /* Arbitrary waveform: decompress shape 0 (peak-normalized) */
            num_samples   = gdef->fall_time_or_num_uncompressed_samples;
            time_shape_id = gdef->unused_or_time_shape_id;
            shape_id      = gdef->shot_shape_ids[0];

            if (shape_id <= 0 || shape_id > desc->num_shapes) {
                for (i = 0; i < fmod->num_samples; ++i) out_wave[i] = 0.0f;
                fmod->ref_integral[axis] = 0.0f;
                continue;
            }

            if (!pulseqlib__decompress_shape(&decomp_wave,
                    &desc->shapes[shape_id - 1], 1.0f)) {
                for (i = 0; i < fmod->num_samples; ++i) out_wave[i] = 0.0f;
                fmod->ref_integral[axis] = 0.0f;
                continue;
            }

            has_time_shape = 0;
            if (time_shape_id > 0 && time_shape_id <= desc->num_shapes) {
                if (pulseqlib__decompress_shape(&decomp_time,
                        &desc->shapes[time_shape_id - 1], grad_raster_us))
                    has_time_shape = 1;
            }

            if (has_time_shape) {
                for (i = 0; i < num_samples && i < decomp_wave.num_uncompressed_samples; ++i) {
                    raw_time[idx] = delay_us + decomp_time.samples[i] - active_start_us;
                    raw_wave[idx] = decomp_wave.samples[i];
                    idx++;
                }
            } else {
                for (i = 0; i < num_samples && i < decomp_wave.num_uncompressed_samples; ++i) {
                    raw_time[idx] = delay_us + 0.5f * grad_raster_us +
                                    (float)i * grad_raster_us - active_start_us;
                    raw_wave[idx] = decomp_wave.samples[i];
                    idx++;
                }
            }

            if (decomp_wave.samples) PULSEQLIB_FREE(decomp_wave.samples);
            if (decomp_time.samples) PULSEQLIB_FREE(decomp_time.samples);
        }

        raw_n = idx;

        /* Ensure coverage at t=0 and t=active_dur_us (zero-pad) */
        if (raw_n > 0 && raw_time[0] > 0.0f) {
            for (j = raw_n; j > 0; --j) {
                raw_time[j] = raw_time[j - 1];
                raw_wave[j] = raw_wave[j - 1];
            }
            raw_time[0] = 0.0f;
            raw_wave[0] = 0.0f;
            raw_n++;
        } else if (raw_n == 0) {
            raw_time[0] = 0.0f;
            raw_wave[0] = 0.0f;
            raw_n = 1;
        }
        if (raw_time[raw_n - 1] < active_dur_us) {
            raw_time[raw_n] = active_dur_us;
            raw_wave[raw_n] = 0.0f;
            raw_n++;
        }

        /* Interpolate to uniform raster */
        pulseqlib__interp1_linear(out_wave, uniform_t, fmod->num_samples,
                                  raw_time, raw_wave, raw_n);

        /* Compute partial integral from start to reference point */
        {
            int ref_sample = (int)(ref_time_us / grad_raster_us);
            if (ref_sample < 0) ref_sample = 0;
            if (ref_sample >= fmod->num_samples) ref_sample = fmod->num_samples - 1;
            fmod->ref_integral[axis] = (float)(PULSEQLIB__TWO_PI * 1e-6) *
                pulseqlib__trapz_real_uniform(
                out_wave, ref_sample + 1, grad_raster_us);
        }
    }

    fmod->ref_time_us = ref_time_us;

    PULSEQLIB_FREE(raw_time);
    PULSEQLIB_FREE(raw_wave);
    PULSEQLIB_FREE(uniform_t);
    return PULSEQLIB_OK;

fmod_fail:
    if (fmod->waveform_gx) { PULSEQLIB_FREE(fmod->waveform_gx); fmod->waveform_gx = NULL; }
    if (fmod->waveform_gy) { PULSEQLIB_FREE(fmod->waveform_gy); fmod->waveform_gy = NULL; }
    if (fmod->waveform_gz) { PULSEQLIB_FREE(fmod->waveform_gz); fmod->waveform_gz = NULL; }
    if (raw_time)  PULSEQLIB_FREE(raw_time);
    if (raw_wave)  PULSEQLIB_FREE(raw_wave);
    if (uniform_t) PULSEQLIB_FREE(uniform_t);
    return PULSEQLIB_ERR_ALLOC_FAILED;
}

/*
 * Build the frequency modulation library for a single sequence descriptor.
 *
 * Three passes:
 *   1) Count blocks with RF or ADC
 *   2) Deduplicate (rf_def_id, adc_def_id, gx_def_id, gy_def_id, gz_def_id)
 *      and build a freq_mod_definition for each unique tuple
 *   3) Assign freq_mod_id in block_table for every block
 *
 * Must be called after segment timing is computed since reference points
 * (isodelay, kzero) may be needed.
 */
int pulseqlib__build_freq_mod_library(pulseqlib_sequence_descriptor* desc)
{
    int n, count, num_unique, result;
    int* block_indices = NULL;
    int (*int_rows)[FREQ_MOD_DEF_COLS] = NULL;
    int* unique_defs = NULL;
    int* event_table = NULL;
    pulseqlib_freq_mod_definition* fm_defs = NULL;

    const pulseqlib_block_table_element* bte;
    const pulseqlib_block_definition* bdef;
    int has_rf, has_adc, adc_def_id;
    int u, row_idx, blk_idx, bdef_id;

    /* Active region & reference point */
    float active_start_us, active_end_us, ref_time_us;
    const pulseqlib_rf_definition* rdef;
    const pulseqlib_adc_definition* adef;
    float adc_dur_us;

    if (!desc || desc->num_blocks <= 0) {
        if (desc) {
            desc->num_freq_mod_defs = 0;
            desc->freq_mod_definitions = NULL;
        }
        return PULSEQLIB_OK;
    }

    /* ---- Pass 1: count blocks with RF or ADC ---- */
    count = 0;
    for (n = 0; n < desc->num_blocks; ++n) {
        bte  = &desc->block_table[n];
        bdef = &desc->block_definitions[bte->id];
        if (bdef->rf_id >= 0 || bte->adc_id >= 0)
            count++;
    }

    if (count == 0) {
        desc->num_freq_mod_defs = 0;
        desc->freq_mod_definitions = NULL;
        for (n = 0; n < desc->num_blocks; ++n)
            desc->block_table[n].freq_mod_id = -1;
        return PULSEQLIB_OK;
    }

    /* Allocate working arrays */
    block_indices = (int*)PULSEQLIB_ALLOC((size_t)count * sizeof(int));
    int_rows      = PULSEQLIB_ALLOC((size_t)count * sizeof(*int_rows));
    unique_defs   = (int*)PULSEQLIB_ALLOC((size_t)count * sizeof(int));
    event_table   = (int*)PULSEQLIB_ALLOC((size_t)count * sizeof(int));
    if (!block_indices || !int_rows || !unique_defs || !event_table)
        goto fm_lib_fail;

    /* ---- Pass 2: build dedup rows ---- */
    count = 0;
    for (n = 0; n < desc->num_blocks; ++n) {
        bte  = &desc->block_table[n];
        bdef = &desc->block_definitions[bte->id];
        has_rf  = (bdef->rf_id >= 0);
        has_adc = (bte->adc_id >= 0);
        if (!has_rf && !has_adc) continue;

        block_indices[count] = n;
        int_rows[count][0] = bdef->rf_id;

        /* Resolve adc definition id */
        if (has_adc && bte->adc_id < desc->adc_table_size)
            int_rows[count][1] = desc->adc_table[bte->adc_id].id;
        else
            int_rows[count][1] = -1;

        int_rows[count][2] = bdef->gx_id;
        int_rows[count][3] = bdef->gy_id;
        int_rows[count][4] = bdef->gz_id;
        count++;
    }

    /* Deduplicate */
    num_unique = pulseqlib__deduplicate_int_rows(
        unique_defs, event_table,
        (const int*)int_rows, count, FREQ_MOD_DEF_COLS);

    /* ---- Build freq_mod definitions for each unique tuple ---- */
    fm_defs = (pulseqlib_freq_mod_definition*)PULSEQLIB_ALLOC(
        (size_t)num_unique * sizeof(pulseqlib_freq_mod_definition));
    if (!fm_defs) goto fm_lib_fail;

    for (u = 0; u < num_unique; ++u) {
        memset(&fm_defs[u], 0, sizeof(fm_defs[u]));
        fm_defs[u].id = u;

        row_idx = unique_defs[u];   /* first occurrence in the row array */
        blk_idx = block_indices[row_idx];
        bte     = &desc->block_table[blk_idx];
        bdef_id = bte->id;
        bdef    = &desc->block_definitions[bdef_id];

        has_rf  = (bdef->rf_id >= 0);
        has_adc = (bte->adc_id >= 0 && bte->adc_id < desc->adc_table_size);

        /* ---- Determine active region and reference point ---- */
        if (has_rf && bdef->rf_id < desc->num_unique_rfs) {
            rdef = &desc->rf_definitions[bdef->rf_id];
            active_start_us = (float)rdef->delay;
#if PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_GEHC
            active_end_us = active_start_us + rdef->stats.duration_us;
            ref_time_us   = (float)rdef->stats.isodelay_us;  /* relative to RF start */
#else
            /* Stub for non-GEHC: use full block as active region */
            active_end_us = (float)bdef->duration_us;
            ref_time_us   = 0.0f;
#endif
        } else if (has_adc) {
            adc_def_id = desc->adc_table[bte->adc_id].id;
            if (adc_def_id >= 0 && adc_def_id < desc->num_unique_adcs) {
                adef = &desc->adc_definitions[adc_def_id];
                adc_dur_us = (float)adef->num_samples *
                             (float)adef->dwell_time * 1e-3f;
                active_start_us = (float)adef->delay;
                active_end_us   = active_start_us + adc_dur_us;
                ref_time_us     = adc_dur_us * 0.5f;  /* default k0 = center */
            } else {
                /* invalid ADC def — skip */
                continue;
            }
        } else {
            continue;  /* shouldn't reach here */
        }

        /* Clamp active region */
        if (active_start_us < 0.0f) active_start_us = 0.0f;
        if (active_end_us > (float)bdef->duration_us)
            active_end_us = (float)bdef->duration_us;
        if (ref_time_us < 0.0f) ref_time_us = 0.0f;
        if (ref_time_us > (active_end_us - active_start_us))
            ref_time_us = active_end_us - active_start_us;

        result = build_freq_mod_for_block(desc, bdef_id,
            active_start_us, active_end_us, ref_time_us, &fm_defs[u]);
        if (PULSEQLIB_FAILED(result)) {
            /* Non-fatal: leave as zeroed init */
        }
    }

    /* ---- Pass 3: assign freq_mod_id in block table ---- */
    for (n = 0; n < desc->num_blocks; ++n)
        desc->block_table[n].freq_mod_id = -1;

    for (n = 0; n < count; ++n)
        desc->block_table[block_indices[n]].freq_mod_id = event_table[n];

    /* Store in descriptor */
    desc->num_freq_mod_defs    = num_unique;
    desc->freq_mod_definitions = fm_defs;

    /* Cleanup working arrays */
    PULSEQLIB_FREE(block_indices);
    PULSEQLIB_FREE(int_rows);
    PULSEQLIB_FREE(unique_defs);
    PULSEQLIB_FREE(event_table);
    return PULSEQLIB_OK;

fm_lib_fail:
    if (block_indices) PULSEQLIB_FREE(block_indices);
    if (int_rows)      PULSEQLIB_FREE(int_rows);
    if (unique_defs)   PULSEQLIB_FREE(unique_defs);
    if (event_table)   PULSEQLIB_FREE(event_table);
    if (fm_defs) {
        for (n = 0; n < num_unique; ++n) {
            if (fm_defs[n].waveform_gx) PULSEQLIB_FREE(fm_defs[n].waveform_gx);
            if (fm_defs[n].waveform_gy) PULSEQLIB_FREE(fm_defs[n].waveform_gy);
            if (fm_defs[n].waveform_gz) PULSEQLIB_FREE(fm_defs[n].waveform_gz);
        }
        PULSEQLIB_FREE(fm_defs);
    }
    return PULSEQLIB_ERR_ALLOC_FAILED;
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
