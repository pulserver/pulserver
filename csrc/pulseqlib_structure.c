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
    int l, i, match;

    if (len <= 1) return len;

    /* Find the shortest period starting at offset 0.
     * The caller already strips the prep region, so the repeating
     * pattern always starts at the beginning of the array.
     * Searching from non-zero offsets is harmful: it picks up short
     * sub-patterns (e.g. [rephaser,nav,rephaser,nav] inside an EPI
     * readout train) that don't span the whole array.               */
    for (l = 1; l <= len / 2; ++l) {
        match = 1;
        for (i = 0; i < l; ++i) {
            if (s[i] != s[i + l]) { match = 0; break; }
        }
        if (match) return l;
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
    int pass, avg, blk, once, count, idx;
    int has_nd_prep, has_nd_cool;
    int prep_tr_id, main_tr_id, cool_tr_id;
    int play_prep, play_main, play_cool;
    int num_passes;

    if (!diag) { pulseqlib_diagnostic_init(&local_diag); diag = &local_diag; }
    else       pulseqlib_diagnostic_init(diag);

    if (!desc) { diag->code = PULSEQLIB_ERR_NULL_POINTER; return diag->code; }
    if (num_averages < 1) num_averages = 1;
    num_passes = (desc->num_passes > 1) ? desc->num_passes : 1;

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

    /* Pass 1: count entries.
     * Passes are the outer loop: each pass plays an average-expanded
     * copy of the per-pass block table (prep on first avg, cooldown on
     * last avg, main on every avg). */
    count = 0;
    for (pass = 0; pass < num_passes; ++pass) {
        int base = pass * desc->pass_len;
        for (avg = 0; avg < num_averages; ++avg) {
            play_prep = (avg == 0) ? 1 : 0;
            play_main = 1;
            play_cool = (avg == num_averages - 1) ? 1 : 0;

            for (blk = 0; blk < desc->pass_len; ++blk) {
                once = desc->block_table[base + blk].once_flag;
                if (once == 1 && play_prep)      ++count;
                else if (once == 0 && play_main) ++count;
                else if (once == 2 && play_cool) ++count;
            }
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
    for (pass = 0; pass < num_passes; ++pass) {
        int base = pass * desc->pass_len;
        for (avg = 0; avg < num_averages; ++avg) {
            play_prep = (avg == 0) ? 1 : 0;
            play_main = 1;
            play_cool = (avg == num_averages - 1) ? 1 : 0;

            for (blk = 0; blk < desc->pass_len; ++blk) {
                once = desc->block_table[base + blk].once_flag;
                if (once == 1 && play_prep) {
                    desc->scan_table_block_idx[idx] = base + blk;
                    desc->scan_table_tr_id[idx]     = prep_tr_id;
                    desc->scan_table_seg_id[idx]    = -1;
                    ++idx;
                } else if (once == 0 && play_main) {
                    desc->scan_table_block_idx[idx] = base + blk;
                    desc->scan_table_tr_id[idx]     = main_tr_id;
                    desc->scan_table_seg_id[idx]    = -1;
                    ++idx;
                } else if (once == 2 && play_cool) {
                    desc->scan_table_block_idx[idx] = base + blk;
                    desc->scan_table_tr_id[idx]     = cool_tr_id;
                    desc->scan_table_seg_id[idx]    = -1;
                    ++idx;
                }
            }
        }
    }

    diag->code = PULSEQLIB_SUCCESS;
    return PULSEQLIB_SUCCESS;
}

/* ================================================================== */
/*  Compute scan_table_tr_start flags                                 */
/* ================================================================== */

/**
 * @brief Set scan_table_tr_start[i] = 1 at the first block of each
 *        main-region TR.
 *
 * Derives main_tr_id from the TR descriptor flags, then walks the
 * scan table marking every tr_size-th main-region block.
 * Safe to call multiple times (re-allocates and re-computes).
 */
void pulseqlib__compute_scan_table_tr_start(
    pulseqlib_sequence_descriptor* desc)
{
    int has_nd_prep, has_nd_cool, main_tr_id;
    int first_main, i, tr_size;

    if (!desc || desc->scan_table_len == 0) return;

    /* (Re)allocate */
    if (desc->scan_table_tr_start)
        PULSEQLIB_FREE(desc->scan_table_tr_start);
    desc->scan_table_tr_start = (int*)PULSEQLIB_ALLOC(
        (size_t)desc->scan_table_len * sizeof(int));
    if (!desc->scan_table_tr_start) return;
    memset(desc->scan_table_tr_start, 0,
           (size_t)desc->scan_table_len * sizeof(int));

    /* Derive main_tr_id from TR descriptor */
    has_nd_prep = (desc->tr_descriptor.num_prep_blocks > 0 &&
                   !desc->tr_descriptor.degenerate_prep);
    has_nd_cool = (desc->tr_descriptor.num_cooldown_blocks > 0 &&
                   !desc->tr_descriptor.degenerate_cooldown);
    if (!has_nd_prep && !has_nd_cool)      main_tr_id = 0;
    else if (has_nd_prep && !has_nd_cool)  main_tr_id = 1;
    else if (!has_nd_prep && has_nd_cool)  main_tr_id = 0;
    else                                   main_tr_id = 1;

    tr_size = desc->tr_descriptor.tr_size;
    if (tr_size <= 0) return;

    first_main = -1;
    for (i = 0; i < desc->scan_table_len; ++i) {
        if (desc->scan_table_tr_id[i] == main_tr_id) {
            if (first_main < 0) {
                first_main = i;
                desc->scan_table_tr_start[i] = 1;
            } else if ((i - first_main) % tr_size == 0) {
                desc->scan_table_tr_start[i] = 1;
            }
        }
    }
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
    int* base_pat      = NULL;
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
    if (desc->num_prep_blocks + desc->num_cooldown_blocks > desc->pass_len) {
        diag->code = PULSEQLIB_ERR_TR_NO_IMAGING_REGION;
        return diag->code;
    }

    tr->tr_size             = 0;
    tr->num_trs             = 0;
    tr->tr_duration_us      = 0.0f;
    tr->degenerate_prep     = (desc->num_prep_blocks == 0) ? 1 : 0;
    tr->degenerate_cooldown = (desc->num_cooldown_blocks == 0) ? 1 : 0;
    tr->num_prep_blocks     = desc->num_prep_blocks;
    tr->num_prep_trs        = 1;
    tr->num_cooldown_blocks = desc->num_cooldown_blocks;
    tr->num_cooldown_trs    = 1;
    tr->imaging_tr_start    = 0;

    /* Always search main region only for TR pattern.
     * After finding the period, we compare prep/cooldown to the
     * pattern to determine degeneracy.  Once-flags are preserved
     * on the block table for scan-table construction. */
    imaging_start = desc->num_prep_blocks;
    imaging_end   = desc->pass_len - desc->num_cooldown_blocks;
    imaging_len   = imaging_end - imaging_start;
    pulseqlib__diag_printf(diag, "imaging region length=%d (prep=%d cool=%d)",
                           imaging_len, desc->num_prep_blocks, desc->num_cooldown_blocks);

    if (imaging_len <= 0) {
        diag->code = PULSEQLIB_ERR_TR_NO_IMAGING_REGION;
        return diag->code;
    }

    /* unique-block count for diagnostics (first pass only) */
    {
        int max_u = 0;
        for (n = 0; n < desc->pass_len; ++n)
            if (desc->block_table[n].id > max_u) max_u = desc->block_table[n].id;
        pulseqlib__diag_printf(diag, " unique blocks=%d", max_u + 1);
    }

    seq_pat   = (int*)PULSEQLIB_ALLOC(desc->pass_len * sizeof(int));
    block_dur = (int*)PULSEQLIB_ALLOC(desc->pass_len * sizeof(int));
    if (!seq_pat || !block_dur) {
        if (seq_pat)   PULSEQLIB_FREE(seq_pat);
        if (block_dur) PULSEQLIB_FREE(block_dur);
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        return diag->code;
    }

    for (n = 0; n < desc->pass_len; ++n) {
        block_dur[n] = desc->block_definitions[desc->block_table[n].id].duration_us;
        seq_pat[n] = (desc->block_table[n].duration_us >= 0)
            ? block_dur[n]
            : -1 * desc->block_table[desc->block_table[n].id].id;
    }

    /* Save a copy of seq_pat before RF augmentation (used for VFA check) */
    base_pat = (int*)PULSEQLIB_ALLOC(desc->pass_len * sizeof(int));
    if (!base_pat) {
        PULSEQLIB_FREE(seq_pat); PULSEQLIB_FREE(block_dur);
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        return diag->code;
    }
    for (n = 0; n < desc->pass_len; ++n)
        base_pat[n] = seq_pat[n];

    /* ----------------------------------------------------------------
     * RF-aware pattern augmentation.
     *
     * Blocks that share the same structural definition (same seq_pat
     * value) but differ in RF amplitude or RF shim ID must get distinct
     * pattern values so that first_repeating_segment discovers the true
     * period.  We quantise the RF amplitude to an integer, form a tuple
     * (base_pat, quant_amp, shim_id) and use the existing hash-based
     * dedup to assign a unique label to each distinct combination.
     * A new synthetic pattern value (above the current maximum) is then
     * assigned to every RF-carrying block.
     *
     * This only affects the temporary seq_pat array — the original
     * block table and definitions are untouched.
     * ---------------------------------------------------------------- */
    {
        int max_pat, rf_id, shim_id, q_amp;
        int* rf_rows;       /* [pass_len * 3]  (base, amp, shim) */
        int* rf_labels;     /* [pass_len]       dedup label      */
        int* rf_unique;     /* [pass_len]       unique row idx   */
        int  num_rf_combos;

        /* Find current max absolute value in seq_pat */
        max_pat = 0;
        for (n = 0; n < desc->pass_len; ++n) {
            int v = seq_pat[n] < 0 ? -seq_pat[n] : seq_pat[n];
            if (v > max_pat) max_pat = v;
        }

        rf_rows   = (int*)PULSEQLIB_ALLOC(desc->pass_len * 3 * sizeof(int));
        rf_labels = (int*)PULSEQLIB_ALLOC(desc->pass_len * sizeof(int));
        rf_unique = (int*)PULSEQLIB_ALLOC(desc->pass_len * sizeof(int));
        if (!rf_rows || !rf_labels || !rf_unique) {
            if (rf_rows)   PULSEQLIB_FREE(rf_rows);
            if (rf_labels) PULSEQLIB_FREE(rf_labels);
            if (rf_unique) PULSEQLIB_FREE(rf_unique);
            PULSEQLIB_FREE(seq_pat); PULSEQLIB_FREE(block_dur);
            PULSEQLIB_FREE(base_pat);
            diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
            return diag->code;
        }

        /* Build 3-column rows: (base_pat, quantised_rf_amp, rf_shim_id) */
        for (n = 0; n < desc->pass_len; ++n) {
            rf_id   = desc->block_table[n].rf_id;
            shim_id = desc->block_table[n].rf_shim_id;

            /* Quantise amplitude: multiply by 1e6 and round to int.
             * This makes amplitudes differing by < 1e-6 identical.
             * Blocks without RF get (base, 0, -1) which dedup will
             * still group correctly. */
            q_amp = (rf_id >= 0 && rf_id < desc->rf_table_size)
                  ? (int)(desc->rf_table[rf_id].amplitude * 1e6f + 0.5f)
                  : 0;

            rf_rows[n * 3 + 0] = seq_pat[n];
            rf_rows[n * 3 + 1] = q_amp;
            rf_rows[n * 3 + 2] = shim_id;
        }

        num_rf_combos = pulseqlib__deduplicate_int_rows(
            rf_unique, rf_labels, rf_rows, desc->pass_len, 3);

        /* Remap: each label becomes max_pat + 1 + label */
        if (num_rf_combos > 0) {
            for (n = 0; n < desc->pass_len; ++n)
                seq_pat[n] = max_pat + 1 + rf_labels[n];
        }

        PULSEQLIB_FREE(rf_rows);
        PULSEQLIB_FREE(rf_labels);
        PULSEQLIB_FREE(rf_unique);
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
        /* ---------------------------------------------------------
         * VFA rejection: if the structural (base) pattern has a
         * valid shorter period but the RF-augmented pattern does
         * not, the sequence contains non-periodic RF over a
         * repeating structure (e.g. VFA SPGR).  Such sequences
         * should be designed as separate subsequences, so we
         * reject them instead of falling through to single-TR.
         * --------------------------------------------------------- */
        int base_l, base_ok;
        base_l = first_repeating_segment(
                     &base_pat[imaging_start], imaging_len);
        base_ok = 0;
        if (base_l > 0 && base_l < imaging_len) {
            base_ok = 1;
            for (i = 0; i < imaging_len && base_ok; ++i) {
                if (base_pat[imaging_start + i] !=
                    base_pat[imaging_start + (i % base_l)])
                    base_ok = 0;
            }
        }

        if (!base_ok) {
            /* Base pattern also non-periodic → genuine single-TR.
             * The main region IS the single TR.  Then compare
             * prep/cooldown to the TR pattern for degeneracy. */
            active_dur_us = 0;
            for (n = 0; n < desc->pass_len; ++n)
                if (desc->block_table[n].duration_us < 0)
                    active_dur_us += desc->block_definitions[
                        desc->block_table[n].id].duration_us;

            if (active_dur_us <= SINGLE_TR_MAX_DURATION_US) {
                l = imaging_len;  /* single-TR = entire main region */
                found = 1;        /* fall through to post-hoc degenerate check */
            }
        }

        if (!found) {
            /* VFA case (base_ok=1) or genuinely too-long non-periodic:
             * reject with a pattern error. */
            diag->code = (mismatch_pos >= 0)
                ? PULSEQLIB_ERR_TR_PATTERN_MISMATCH
                : PULSEQLIB_ERR_TR_NO_PERIODIC_PATTERN;
            PULSEQLIB_FREE(seq_pat); PULSEQLIB_FREE(block_dur);
            PULSEQLIB_FREE(base_pat);
            return diag->code;
        }
    }

    tr->tr_size = l;
    tr_dur = 0.0f;
    tr_start = imaging_start;
    for (i = 0; i < l; ++i) tr_dur += (float)block_dur[tr_start + i];
    tr->tr_duration_us = tr_dur;

    /* num_trs counts main-region TRs */
    tr->num_trs = imaging_len / l;

    /* ----------------------------------------------------------------
     * Post-hoc degenerate prep/cooldown detection.
     *
     * Now that we know the TR period l found in the main region,
     * compare each once-flagged region to the TR pattern.  If the
     * prep/cooldown seq_pat matches the corresponding portion of
     * the repeating pattern AND its length is a multiple of l, the
     * region is "degenerate" — structurally identical to main.
     * ---------------------------------------------------------------- */
    /* Prep: compare [0..np-1] against TR pattern starting at imaging_start */
    if (desc->num_prep_blocks > 0 && desc->num_prep_blocks % l == 0) {
        int match = 1;
        for (i = 0; i < desc->num_prep_blocks && match; ++i) {
            if (seq_pat[i] != seq_pat[imaging_start + (i % l)])
                match = 0;
        }
        if (match) {
            tr->degenerate_prep     = 1;
            tr->num_prep_blocks     = 0;
            tr->num_prep_trs        = desc->num_prep_blocks / l;
            tr->imaging_tr_start    = desc->num_prep_blocks;
        }
    }
    /* Cooldown: compare [pass_len-nc..pass_len-1] against TR pattern */
    if (desc->num_cooldown_blocks > 0 && desc->num_cooldown_blocks % l == 0) {
        int nc = desc->num_cooldown_blocks;
        int cool_start = desc->pass_len - nc;
        int match = 1;
        for (i = 0; i < nc && match; ++i) {
            if (seq_pat[cool_start + i] != seq_pat[imaging_start + (i % l)])
                match = 0;
        }
        if (match) {
            tr->degenerate_cooldown  = 1;
            tr->num_cooldown_blocks  = 0;
            tr->num_cooldown_trs     = nc / l;
        }
    }

    /* Non-degenerate prep: verify duration is within threshold */
    if (!tr->degenerate_prep && desc->num_prep_blocks > 0) {
        prep_dur_us = (int)sum_durations_us(block_dur, 0, desc->num_prep_blocks);
        if (prep_dur_us > PREP_COOLDOWN_THRESHOLD_US) {
            diag->code = PULSEQLIB_ERR_TR_PREP_TOO_LONG;
            PULSEQLIB_FREE(seq_pat); PULSEQLIB_FREE(block_dur);
            PULSEQLIB_FREE(base_pat);
            return diag->code;
        }
    }
    /* Non-degenerate cooldown: verify duration is within threshold */
    if (!tr->degenerate_cooldown && desc->num_cooldown_blocks > 0) {
        cooldown_dur_us = (int)sum_durations_us(block_dur,
            desc->pass_len - desc->num_cooldown_blocks,
            desc->num_cooldown_blocks);
        if (cooldown_dur_us > PREP_COOLDOWN_THRESHOLD_US) {
            diag->code = PULSEQLIB_ERR_TR_COOLDOWN_TOO_LONG;
            PULSEQLIB_FREE(seq_pat); PULSEQLIB_FREE(block_dur);
            PULSEQLIB_FREE(base_pat);
            return diag->code;
        }
    }

    diag->code = PULSEQLIB_SUCCESS;
    PULSEQLIB_FREE(seq_pat); PULSEQLIB_FREE(block_dur);
    PULSEQLIB_FREE(base_pat);
    return PULSEQLIB_SUCCESS;
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

    if (!desc) return PULSEQLIB_SUCCESS;

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

    return PULSEQLIB_SUCCESS;
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

    if (desc->vendor != PULSEQLIB_VENDOR_GEHC) {
        return PULSEQLIB_ERR_NOT_IMPLEMENTED;
    }

    num_columns = 3; /* GEHC: [lin, slc, eco] */

    imaging_start  = desc->num_prep_blocks;
    cooldown_start = desc->pass_len - desc->num_cooldown_blocks;
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
    for (b = cooldown_start; b < desc->pass_len; ++b) {
        if (desc->block_table[b].adc_id >= 0) ++total_adcs;
    }

    if (total_adcs == 0) {
        desc->label_num_columns = num_columns;
        desc->label_num_entries = 0;
        desc->label_table       = NULL;
        memset(&desc->label_limits, 0, sizeof(desc->label_limits));
        return PULSEQLIB_SUCCESS;
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
    for (b = cooldown_start; b < desc->pass_len; ++b) {
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

    return PULSEQLIB_SUCCESS;
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
    const pulseqlib_grad_definition* gdef;
    int shot_idx, bt_idx, prev_bt;
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
    grad_ids[0] = desc->block_table[bt_idx].gx_id;
    grad_ids[1] = desc->block_table[bt_idx].gy_id;
    grad_ids[2] = desc->block_table[bt_idx].gz_id;
    for (i = 0; i < 3; ++i) {
        if (grad_ids[i] < 0) continue;
        gdef = &desc->grad_definitions[desc->grad_table[grad_ids[i]].id];
        shot_idx = desc->grad_table[grad_ids[i]].shot_index;
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

    /* last block gradient check */
    bt_idx = scan_block_idx[pat_start + nb - 1];
    grad_ids[0] = desc->block_table[bt_idx].gx_id;
    grad_ids[1] = desc->block_table[bt_idx].gy_id;
    grad_ids[2] = desc->block_table[bt_idx].gz_id;
    for (i = 0; i < 3; ++i) {
        if (grad_ids[i] < 0) continue;
        gdef = &desc->grad_definitions[desc->grad_table[grad_ids[i]].id];
        shot_idx = desc->grad_table[grad_ids[i]].shot_index;
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

            grad_ids[0] = desc->block_table[prev_bt].gx_id;
            grad_ids[1] = desc->block_table[prev_bt].gy_id;
            grad_ids[2] = desc->block_table[prev_bt].gz_id;
            for (i = 0; i < 3; ++i) {
                grad_last_cur[i] = 0.0f;
                if (grad_ids[i] >= 0) {
                    gdef = &desc->grad_definitions[desc->grad_table[grad_ids[i]].id];
                    shot_idx = desc->grad_table[grad_ids[i]].shot_index;
                    grad_last_cur[i] = gdef->last_value[shot_idx] * gdef->max_amplitude[shot_idx];
                }
            }

            grad_ids[0] = desc->block_table[bt_idx].gx_id;
            grad_ids[1] = desc->block_table[bt_idx].gy_id;
            grad_ids[2] = desc->block_table[bt_idx].gz_id;
            for (i = 0; i < 3; ++i) {
                grad_first_next[i] = 0.0f;
                if (grad_ids[i] >= 0) {
                    gdef = &desc->grad_definitions[desc->grad_table[grad_ids[i]].id];
                    shot_idx = desc->grad_table[grad_ids[i]].shot_index;
                    grad_first_next[i] = gdef->first_value[shot_idx] * gdef->max_amplitude[shot_idx];
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
    const pulseqlib_sequence_descriptor* desc,
    const int* scan_block_idx,
    int first_scan_pos, int last_scan_pos,
    float max_allowed)
{
    int ch, gid, si, bt;
    float pv;
    const pulseqlib_grad_definition* gdef;

    bt = scan_block_idx[first_scan_pos];
    for (ch = 0; ch < 3; ++ch) {
        gid = (ch == 0) ? desc->block_table[bt].gx_id
            : (ch == 1) ? desc->block_table[bt].gy_id
            :             desc->block_table[bt].gz_id;
        if (gid < 0) continue;
        si   = desc->grad_table[gid].shot_index;
        gdef = &desc->grad_definitions[desc->grad_table[gid].id];
        pv   = gdef->first_value[si] * gdef->max_amplitude[si];
        if ((float)fabs(pv) > max_allowed) return 0;
    }

    bt = scan_block_idx[last_scan_pos];
    for (ch = 0; ch < 3; ++ch) {
        gid = (ch == 0) ? desc->block_table[bt].gx_id
            : (ch == 1) ? desc->block_table[bt].gy_id
            :             desc->block_table[bt].gz_id;
        if (gid < 0) continue;
        si   = desc->grad_table[gid].shot_index;
        gdef = &desc->grad_definitions[desc->grad_table[gid].id];
        pv   = gdef->last_value[si] * gdef->max_amplitude[si];
        if ((float)fabs(pv) > max_allowed) return 0;
    }

    return 1;
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
    int scan_len, pass_size, num_passes;
    int num_raw, num_total, num_unique;
    int n_prep_raw, n_main_raw, n_cool_raw;
    int n_prep, n_main, n_cool;
    int n, b, i, found, offset;
    int num_raw_alloc, num_exp_alloc;
    int seg_result, max_expanded;
    int nb, unique_idx, blk_tab_idx, blk_def_id, shot_idx;
    int ax_grad_ids[3], ax_def_ids[3], ax;
    float inst_energy, e, amp;
    const pulseqlib_block_table_element* bte;
    const pulseqlib_block_definition* bdef;
    int mult, max_mult, all_covered;
    int num_prep_blk, num_cool_blk, tr_size, region_start, region_size;
    int prep_absorbed_trs, cool_absorbed_trs;
    int main_region_start, main_region_size;
    float max_allowed, new_tr_dur;

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

    scan_len   = desc->scan_table_len;
    num_passes = (desc->num_passes > 1) ? desc->num_passes : 1;
    pass_size  = scan_len / num_passes;

    num_prep_blk = desc->tr_descriptor.num_prep_blocks;
    num_cool_blk = desc->tr_descriptor.num_cooldown_blocks;
    tr_size      = desc->tr_descriptor.tr_size;
    max_allowed  = opts->max_slew_hz_per_m_per_s
                 * desc->grad_raster_us * 1e-6f;

    /* max_mult: maximum number of TRs we can absorb into a section retry.
     * The entire first pass is the upper bound. */
    max_mult = (tr_size > 0) ? (pass_size / tr_size) : 1;
    if (max_mult < 1) max_mult = 1;

    num_raw = 0; num_total = 0; num_unique = 0;
    n_prep_raw = 0; n_main_raw = 0; n_cool_raw = 0;
    n_prep = 0; n_main = 0; n_cool = 0;
    num_raw_alloc = 0; num_exp_alloc = 0;
    all_covered = 0;
    prep_absorbed_trs = 0;
    cool_absorbed_trs = 0;
    main_region_start = num_prep_blk;
    main_region_size  = 0;

    /* ---- 1. Map first pass of scan table to block-def-ID pattern ---- */
    scan_pat = (int*)PULSEQLIB_ALLOC((size_t)pass_size * sizeof(int));
    if (!scan_pat) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; return 0; }
    for (n = 0; n < pass_size; ++n)
        scan_pat[n] = desc->block_table[desc->scan_table_block_idx[n]].id;

    raw_segs = (pulseqlib_tr_segment*)PULSEQLIB_ALLOC(
        (size_t)pass_size * sizeof(pulseqlib_tr_segment));
    if (!raw_segs) {
        PULSEQLIB_FREE(scan_pat);
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        return 0;
    }

    /*
     * Three-section segmentation on the first pass.
     *
     *   Prep:     [0,           num_prep + k*tr_size)   k=1,2,…
     *   Main:     [num_prep,    num_prep + k*tr_size)   k=1,2,…
     *   Cooldown: [pass_size - num_cool - k*tr_size, pass_size)  k=1,2,…
     *             fallback: [0, pass_size)
     *
     * Each retry starts with a fast boundary gradient pre-check.
     */

    /* ---- 2a. Prep section ---- */
    if (!desc->tr_descriptor.degenerate_prep && num_prep_blk > 0) {
        seg_result = 0;
        for (mult = 1; mult <= max_mult; ++mult) {
            region_size = num_prep_blk + mult * tr_size;
            if (region_size > pass_size) break;
            if (!scan_boundary_gradients_ok(desc,
                    desc->scan_table_block_idx,
                    0, region_size - 1, max_allowed))
                continue;
            pulseqlib_diagnostic_init(diag);
            seg_result = find_segments_on_scan_table(
                desc, raw_segs, num_raw, diag, opts,
                desc->scan_table_block_idx, 0, region_size);
            if (seg_result > 0) break;
            if (diag->code != PULSEQLIB_ERR_SEG_NONZERO_START_GRAD &&
                diag->code != PULSEQLIB_ERR_SEG_NONZERO_END_GRAD)
                break;
        }
        if (seg_result == 0 && PULSEQLIB_FAILED(diag->code)) {
            PULSEQLIB_FREE(scan_pat); PULSEQLIB_FREE(raw_segs);
            return 0;
        }
        n_prep_raw = seg_result;
        num_raw += n_prep_raw;
        if (region_size >= pass_size) all_covered = 1;
        /* Record how many main TRs the prep section absorbed */
        prep_absorbed_trs = mult;
    }

    /* ---- 2b. Main section ---- */
    if (!all_covered) {
        /* Start main after the TRs already absorbed by prep */
        region_start = num_prep_blk + prep_absorbed_trs * tr_size;
        seg_result = 0;
        for (mult = 1; mult <= max_mult; ++mult) {
            region_size = mult * tr_size;
            if (region_start + region_size > pass_size) break;
            if (!scan_boundary_gradients_ok(desc,
                    desc->scan_table_block_idx,
                    region_start, region_start + region_size - 1,
                    max_allowed))
                continue;
            pulseqlib_diagnostic_init(diag);
            seg_result = find_segments_on_scan_table(
                desc, raw_segs, num_raw, diag, opts,
                desc->scan_table_block_idx,
                region_start, region_size);
            if (seg_result > 0) break;
            if (diag->code != PULSEQLIB_ERR_SEG_NONZERO_START_GRAD &&
                diag->code != PULSEQLIB_ERR_SEG_NONZERO_END_GRAD)
                break;
        }
        if (seg_result == 0 && PULSEQLIB_FAILED(diag->code)) {
            PULSEQLIB_FREE(scan_pat); PULSEQLIB_FREE(raw_segs);
            return 0;
        }
        n_main_raw = seg_result;
        num_raw += n_main_raw;

        /* Record main region geometry for tiling in step 11 */
        main_region_start = region_start;
        main_region_size  = region_size;

        /* Update TR descriptor when main needed more than one original TR */
        if (mult > 1 && seg_result > 0) {
            new_tr_dur = 0.0f;
            for (n = region_start; n < region_start + region_size; ++n) {
                blk_def_id = desc->block_table[
                    desc->scan_table_block_idx[n]].id;
                new_tr_dur += (float)desc->block_definitions[blk_def_id]
                                  .duration_us;
            }
            desc->tr_descriptor.tr_size = mult * tr_size;
            desc->tr_descriptor.num_trs =
                (pass_size - num_prep_blk - num_cool_blk)
                / (mult * tr_size);
            desc->tr_descriptor.tr_duration_us = new_tr_dur;
        }

        if (region_start + region_size >= pass_size) all_covered = 1;
    }

    /* ---- 2c. Cooldown section ---- */
    if (!all_covered &&
        !desc->tr_descriptor.degenerate_cooldown && num_cool_blk > 0) {
        /* Use (updated) tr_size: if main absorbed multiple original TRs,
         * the effective TR size has been enlarged above. */
        seg_result = 0;
        for (mult = 1; mult <= max_mult; ++mult) {
            region_size  = num_cool_blk + mult * desc->tr_descriptor.tr_size;
            region_start = pass_size - region_size;
            if (region_start < 0) break;
            /* Cooldown must not overlap the main segmentation region */
            if (region_start < main_region_start + main_region_size)
                break;
            if (!scan_boundary_gradients_ok(desc,
                    desc->scan_table_block_idx,
                    region_start, pass_size - 1, max_allowed))
                continue;
            pulseqlib_diagnostic_init(diag);
            seg_result = find_segments_on_scan_table(
                desc, raw_segs, num_raw, diag, opts,
                desc->scan_table_block_idx,
                region_start, region_size);
            if (seg_result > 0) break;
            if (diag->code != PULSEQLIB_ERR_SEG_NONZERO_START_GRAD &&
                diag->code != PULSEQLIB_ERR_SEG_NONZERO_END_GRAD)
                break;
        }
        if (seg_result == 0 && PULSEQLIB_FAILED(diag->code)) {
            PULSEQLIB_FREE(scan_pat); PULSEQLIB_FREE(raw_segs);
            return 0;
        }
        if (!all_covered) {
            n_cool_raw = seg_result;
            num_raw += n_cool_raw;
            cool_absorbed_trs = mult;
        }
    }

    /* ---- 2d. If all sections produced nothing, run a single
     *          find_segments over [0, pass_size) WITHOUT boundary
     *          pre-check so the real error code propagates.  This also
     *          serves as the ultimate fallback when cooldown has 0
     *          blocks and thus its branch was skipped entirely. ---- */
    if (num_raw == 0) {
        pulseqlib_diagnostic_init(diag);
        seg_result = find_segments_on_scan_table(
            desc, raw_segs, 0, diag, opts,
            desc->scan_table_block_idx,
            0, pass_size);
        if (seg_result > 0) {
            n_prep_raw = 0;
            n_main_raw = seg_result;
            n_cool_raw = 0;
            num_raw    = seg_result;
            all_covered = 1;
        } else {
            /* Propagate the actual error from find_segments */
            if (!PULSEQLIB_FAILED(diag->code))
                diag->code = PULSEQLIB_ERR_SEG_NO_SEGMENTS_FOUND;
            PULSEQLIB_FREE(scan_pat); PULSEQLIB_FREE(raw_segs);
            return 0;
        }
    }

    /* ---- 3. Populate unique_block_indices ---- */
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

    /* ---- 4. Strip pure delays (per section) ---- */
    max_expanded = pass_size;
    exp_segs = (pulseqlib_tr_segment*)PULSEQLIB_ALLOC(
        (size_t)max_expanded * sizeof(pulseqlib_tr_segment));
    if (!exp_segs) {
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        goto scan_seg_fail;
    }

    offset = 0;
    if (n_prep_raw > 0) {
        n_prep = strip_pure_delays_scan(
            raw_segs, n_prep_raw, exp_segs + offset,
            max_expanded - offset,
            desc->block_table, desc->scan_table_block_idx);
        if (n_prep < 0) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto scan_seg_fail; }
        offset += n_prep;
    }

    if (n_main_raw > 0) {
        n_main = strip_pure_delays_scan(
            raw_segs + n_prep_raw, n_main_raw,
            exp_segs + offset, max_expanded - offset,
            desc->block_table, desc->scan_table_block_idx);
        if (n_main < 0) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto scan_seg_fail; }
        offset += n_main;
    }

    if (n_cool_raw > 0) {
        n_cool = strip_pure_delays_scan(
            raw_segs + n_prep_raw + n_main_raw, n_cool_raw,
            exp_segs + offset, max_expanded - offset,
            desc->block_table, desc->scan_table_block_idx);
        if (n_cool < 0) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto scan_seg_fail; }
        offset += n_cool;
    }

    num_total = n_prep + n_main + n_cool;
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

    /* ---- 5. NAV-aware split and merge (per section, PMC only) ---- */
    if (desc->enable_pmc) {
        pulseqlib_tr_segment* nav_segs;
        int nav_total = 0, r;

        nav_segs = (pulseqlib_tr_segment*)PULSEQLIB_ALLOC(
            (size_t)max_expanded * sizeof(pulseqlib_tr_segment));
        if (!nav_segs) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto scan_seg_fail; }

        if (n_prep > 0) {
            r = nav_split_merge(exp_segs, n_prep,
                    nav_segs + nav_total, max_expanded - nav_total,
                    desc->block_table, desc->scan_table_block_idx);
            if (r < 0) {
                for (n = 0; n < nav_total; ++n)
                    if (nav_segs[n].unique_block_indices)
                        PULSEQLIB_FREE(nav_segs[n].unique_block_indices);
                PULSEQLIB_FREE(nav_segs);
                diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto scan_seg_fail;
            }
            n_prep = r; nav_total += r;
        }

        r = nav_split_merge(
                exp_segs + (num_total - n_cool - n_main), n_main,
                nav_segs + nav_total, max_expanded - nav_total,
                desc->block_table, desc->scan_table_block_idx);
        if (r < 0) {
            for (n = 0; n < nav_total; ++n)
                if (nav_segs[n].unique_block_indices)
                    PULSEQLIB_FREE(nav_segs[n].unique_block_indices);
            PULSEQLIB_FREE(nav_segs);
            diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto scan_seg_fail;
        }
        n_main = r; nav_total += r;

        if (n_cool > 0) {
            r = nav_split_merge(
                    exp_segs + (num_total - n_cool), n_cool,
                    nav_segs + nav_total, max_expanded - nav_total,
                    desc->block_table, desc->scan_table_block_idx);
            if (r < 0) {
                for (n = 0; n < nav_total; ++n)
                    if (nav_segs[n].unique_block_indices)
                        PULSEQLIB_FREE(nav_segs[n].unique_block_indices);
                PULSEQLIB_FREE(nav_segs);
                diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto scan_seg_fail;
            }
            n_cool = r; nav_total += r;
        }

        /* Replace exp_segs with nav_segs */
        PULSEQLIB_FREE(exp_segs);
        exp_segs = nav_segs;
        num_total = nav_total;
        num_exp_alloc = nav_total;
    }

    /* ---- 6. Segment tables (prep / main / cooldown split) ---- */
    desc->segment_table.num_prep_segments     = n_prep;
    desc->segment_table.num_main_segments     = n_main;
    desc->segment_table.num_cooldown_segments = n_cool;
    desc->segment_table.prep_segment_table     =
        (n_prep > 0) ? (int*)PULSEQLIB_ALLOC(n_prep * sizeof(int)) : NULL;
    desc->segment_table.main_segment_table     =
        (n_main > 0) ? (int*)PULSEQLIB_ALLOC(n_main * sizeof(int)) : NULL;
    desc->segment_table.cooldown_segment_table =
        (n_cool > 0) ? (int*)PULSEQLIB_ALLOC(n_cool * sizeof(int)) : NULL;

    uniq_segs = (pulseqlib_tr_segment*)PULSEQLIB_ALLOC(
        (size_t)num_total * sizeof(pulseqlib_tr_segment));
    if (!uniq_segs) {
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        goto scan_seg_fail;
    }

    /* ---- 7. Deduplicate segments across all sections ---- */
    num_unique = 0;

    for (n = 0; n < num_total; ++n) {
        /* Find existing match by UBI (works for both pure-delay and
         * active segments — pure delays have num_blocks==1). */
        found = -1;
        for (i = 0; i < num_unique; ++i) {
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

        /* Assign to the appropriate section table */
        if (n < n_prep)
            desc->segment_table.prep_segment_table[n] = found;
        else if (n < n_prep + n_main)
            desc->segment_table.main_segment_table[n - n_prep] = found;
        else
            desc->segment_table.cooldown_segment_table[n - n_prep - n_main] = found;
    }

    desc->segment_table.num_unique_segments = num_unique;
    desc->num_unique_segments = num_unique;

    /* ---- 8. Transfer segment definitions ---- */
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

    /* ---- 9. Per-block flags ---- */
    for (i = 0; i < num_unique; ++i) {
        nb = desc->segment_definitions[i].num_blocks;
        desc->segment_definitions[i].has_digitalout = (int*)PULSEQLIB_ALLOC(nb * sizeof(int));
        desc->segment_definitions[i].has_rotation = (int*)PULSEQLIB_ALLOC(nb * sizeof(int));
        desc->segment_definitions[i].norot_flag   = (int*)PULSEQLIB_ALLOC(nb * sizeof(int));
        desc->segment_definitions[i].nopos_flag   = (int*)PULSEQLIB_ALLOC(nb * sizeof(int));
        desc->segment_definitions[i].has_freq_mod = (int*)PULSEQLIB_ALLOC(nb * sizeof(int));
        if (!desc->segment_definitions[i].has_digitalout ||
            !desc->segment_definitions[i].has_rotation ||
            !desc->segment_definitions[i].norot_flag ||
            !desc->segment_definitions[i].nopos_flag ||
            !desc->segment_definitions[i].has_freq_mod) {
            diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
            goto scan_seg_fail;
        }
        for (n = 0; n < nb; ++n) {
            desc->segment_definitions[i].has_digitalout[n] = 0;
            desc->segment_definitions[i].has_rotation[n] = 0;
            desc->segment_definitions[i].norot_flag[n]   = 0;
            desc->segment_definitions[i].nopos_flag[n]   = 0;
            desc->segment_definitions[i].has_freq_mod[n] = 0;
        }
        desc->segment_definitions[i].trigger_id = -1;
    }

    /* ---- 10. Walk expanded segments, populate flags + max energy ---- */
    max_energy = (float*)PULSEQLIB_ALLOC((size_t)num_unique * sizeof(float));
    if (!max_energy) { diag->code = PULSEQLIB_ERR_ALLOC_FAILED; goto scan_seg_fail; }
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

    /* max_energy freed after step 11b rescan */

    /* ---- tag segments as NAV; verify at most 1 unique NAV ---- */
    if (desc->enable_pmc) {
        int nav_count = 0;
        for (i = 0; i < num_unique; ++i) {
            /* start_block already resolved to block_table index in step 8 */
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

    /* ---- 11. Build pattern_seg_id and fill scan_table_seg_id ---- */
    pattern_seg_id = (int*)PULSEQLIB_ALLOC((size_t)pass_size * sizeof(int));
    if (!pattern_seg_id) {
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        goto scan_seg_fail;
    }
    for (n = 0; n < pass_size; ++n) pattern_seg_id[n] = -1;

    /* Walk all three expanded section arrays and assign each position */
    for (n = 0; n < num_total; ++n) {
        if (n < n_prep)
            unique_idx = desc->segment_table.prep_segment_table[n];
        else if (n < n_prep + n_main)
            unique_idx = desc->segment_table.main_segment_table[n - n_prep];
        else
            unique_idx = desc->segment_table.cooldown_segment_table[n - n_prep - n_main];

        for (b = 0; b < exp_segs[n].num_blocks; ++b) {
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
    if (main_region_size > 0) {
        int src_start = main_region_start;
        int src_len   = main_region_size;

        for (i = src_start + src_len; i < pass_size; ++i) {
            if (pattern_seg_id[i] == -1) {
                int src = src_start + ((i - src_start) % src_len);
                if (pattern_seg_id[src] >= 0)
                    pattern_seg_id[i] = pattern_seg_id[src];
            }
        }
    }

    /* Tile first-pass pattern across full scan table (all passes) */
    for (n = 0; n < scan_len; ++n)
        desc->scan_table_seg_id[n] = pattern_seg_id[n % pass_size];

    PULSEQLIB_FREE(pattern_seg_id); pattern_seg_id = NULL;

    /* ---- 11c. Rescan full scan table for max-energy per segment ----
     *
     * Step 10 only saw one instance per segment (the first one).
     * Now that scan_table_seg_id covers all tiled repetitions we
     * can walk the entire scan table, group consecutive blocks into
     * segment instances, compute their energy, and update
     * max_energy_start_block when a higher-energy repetition is found. */
    for (i = 0; i < num_unique; ++i) max_energy[i] = 0.0f;

    for (n = 0; n < scan_len; /* advance inside */) {
        int seg_id = desc->scan_table_seg_id[n];
        if (seg_id < 0 || seg_id >= num_unique) { ++n; continue; }

        nb = desc->segment_definitions[seg_id].num_blocks;
        if (n + nb > scan_len) { ++n; continue; }

        /* Verify all nb positions belong to the same segment */
        {
            int ok = 1;
            for (b = 1; b < nb; ++b) {
                if (desc->scan_table_seg_id[n + b] != seg_id) { ok = 0; break; }
            }
            if (!ok) { ++n; continue; }
        }

        /* Compute energy for this instance */
        inst_energy = 0.0f;
        for (b = 0; b < nb; ++b) {
            blk_tab_idx = desc->scan_table_block_idx[n + b];
            bte  = &desc->block_table[blk_tab_idx];
            bdef = &desc->block_definitions[bte->id];

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
                    amp      = desc->grad_table[ax_grad_ids[ax]].amplitude;
                    shot_idx = desc->grad_table[ax_grad_ids[ax]].shot_index;
                    e = desc->grad_definitions[ax_def_ids[ax]].energy[shot_idx];
                    inst_energy += e * amp * amp;
                }
            }
        }

        if (inst_energy > max_energy[seg_id]) {
            max_energy[seg_id] = inst_energy;
            desc->segment_definitions[seg_id].max_energy_start_block =
                desc->scan_table_block_idx[n];
        }

        n += nb;
    }

    PULSEQLIB_FREE(max_energy); max_energy = NULL;

    /* ---- 11d. OR-reduce per-block flags across ALL segment instances ----
     *
     * Step 10 only populated flags from the first expanded instance of
     * each segment.  Flags like has_digitalout, has_rotation, norot,
     * nopos, and has_freq_mod must reflect ANY instance (e.g. ADC
     * only appears in main TRs, not dummies, so freq_mod would be
     * missed if only the dummy instance was scanned).
     *
     * Repetitions (num_averages > 1) share the same block_table
     * entries.  For multi-pass sequences the pass dimension is the
     * outer loop (passes are interleaved), but all passes have the
     * same structure by definition.  Walking just the first pass
     * (0 .. pass_size-1) is therefore sufficient.                     */
    for (i = 0; i < num_unique; ++i) {
        nb = desc->segment_definitions[i].num_blocks;
        for (n = 0; n < nb; ++n) {
            desc->segment_definitions[i].has_digitalout[n] = 0;
            desc->segment_definitions[i].has_rotation[n]   = 0;
            desc->segment_definitions[i].norot_flag[n]     = 0;
            desc->segment_definitions[i].nopos_flag[n]     = 0;
            desc->segment_definitions[i].has_freq_mod[n]   = 0;
        }
        desc->segment_definitions[i].trigger_id = -1;
    }

    for (n = 0; n < pass_size; /* advance inside */) {
        int seg_id = desc->scan_table_seg_id[n];
        if (seg_id < 0 || seg_id >= num_unique) { ++n; continue; }

        nb = desc->segment_definitions[seg_id].num_blocks;
        if (n + nb > pass_size) { ++n; continue; }

        /* Verify contiguous segment instance */
        {
            int ok = 1;
            for (b = 1; b < nb; ++b) {
                if (desc->scan_table_seg_id[n + b] != seg_id) { ok = 0; break; }
            }
            if (!ok) { ++n; continue; }
        }

        for (b = 0; b < nb; ++b) {
            bte  = &desc->block_table[n + b];
            bdef = &desc->block_definitions[bte->id];

            /* digitalout / trigger */
            if (bte->digitalout_id != -1 && bte->digitalout_id < desc->num_triggers) {
                const pulseqlib_trigger_event* te = &desc->trigger_events[bte->digitalout_id];
                if (te->trigger_type == PULSEQLIB__TRIGGER_TYPE_OUTPUT) {
                    desc->segment_definitions[seg_id].has_digitalout[b] = 1;
                } else if (te->trigger_type == PULSEQLIB__TRIGGER_TYPE_INPUT) {
                    int prev = desc->segment_definitions[seg_id].trigger_id;
                    if (prev < 0)
                        desc->segment_definitions[seg_id].trigger_id = bte->digitalout_id;
                }
            }
            if (bte->rotation_id != -1)
                desc->segment_definitions[seg_id].has_rotation[b] = 1;
            if (bte->norot_flag)
                desc->segment_definitions[seg_id].norot_flag[b]   = 1;
            if (bte->nopos_flag)
                desc->segment_definitions[seg_id].nopos_flag[b]   = 1;

            /* freq_mod: computed inline because build_freq_mod_flags
             * has not run yet at this point in the pipeline.         */
            {
                int has_rf_b   = (bdef->rf_id >= 0);
                int has_adc_b  = (bte->adc_id >= 0);
                int has_grad_b = (bdef->gx_id >= 0 || bdef->gy_id >= 0 || bdef->gz_id >= 0);
                if ((has_rf_b || has_adc_b) && has_grad_b)
                    desc->segment_definitions[seg_id].has_freq_mod[b] = 1;
            }
        }

        n += nb;
    }

    /* ---- Cleanup ---- */
    for (n = 0; n < num_exp_alloc; ++n)
        PULSEQLIB_FREE(exp_segs[n].unique_block_indices);
    PULSEQLIB_FREE(exp_segs); exp_segs = NULL;
    PULSEQLIB_FREE(scan_pat); scan_pat = NULL;

    diag->code = PULSEQLIB_SUCCESS;
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
