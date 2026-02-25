/**
 * @file example_scanloop.c
 * @brief Flat scan-table scan loop with PMC support.
 *
 * Demonstrates the recommended scan-loop architecture:
 *
 *   1. Load the cached sequence collection.
 *   2. Build per-subsequence frequency-modulation libraries.
 *      Non-PMC: computed once, 3-channel data freed immediately.
 *      PMC:     3-channel data retained for TR-boundary updates.
 *   3. For each subsequence, walk the flat scan table
 *      (prep + main x num_trs + cooldown segments).
 *      For each segment:
 *        - Iterate blocks: fetch block_instance + freq-mod,
 *          program each block via vendor_set_block().
 *        - Set FOV rotation, arm trigger if flagged, play segment.
 *   4. PMC-enabled subsequences: at main-TR boundaries, update the
 *      freq-mod library with the new position; after NAV segments,
 *      evaluate motion and optionally rescan the TR.
 *   5. Label readout for reconstruction metadata.
 *
 * Compile:
 *   cc -I../../csrc example_scanloop.c ../../csrc/pulseqlib_*.c -lm -o scanloop
 *
 * Run:
 *   ./scanloop path/to/sequence.seq
 */

#include "pulseqlib_methods.h"
#include "example_vendorlib.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(rc, diag)                                 \
    do {                                                \
        if (PULSEQLIB_FAILED(rc)) {                     \
            vendor_report_error(rc, (diag));             \
            goto fail;                                  \
        }                                               \
    } while (0)

/* ================================================================== */
/*  Global mutable state                                              */
/* ================================================================== */

/** Patient-table / prescription shift in metres (in the physical frame). */
static float g_fovshift[3] = {0.05f, 0.0f, 0.0f};

/** FOV rotation matrix (3x3 row-major, logical -> physical). */
static float g_fovrotation[9] = {1,0,0, 0,1,0, 0,0,1};

/** Per-subsequence freq-mod libraries. */
static pulseqlib_freq_mod_library** g_freqlibs = NULL;
static int                          g_nlibs    = 0;

/* ================================================================== */
/*  Freq-mod library helpers                                          */
/* ================================================================== */

/**
 * @brief Build freq-mod libraries for every subsequence.
 *
 * Called once after loading the collection.  Each library is built
 * with the current g_fovshift.  Non-PMC libraries immediately
 * discard 3-channel data; PMC libraries keep it for in-place
 * update at TR boundaries.
 */
static int build_freqmod_libraries(pulseqlib_collection* coll)
{
    int nsub = pulseqlib_get_num_subsequences(coll);
    int s, rc;

    g_freqlibs = (pulseqlib_freq_mod_library**)calloc(
        (size_t)nsub, sizeof(*g_freqlibs));
    if (!g_freqlibs) return PULSEQLIB_ERR_ALLOC_FAILED;
    g_nlibs = nsub;

    for (s = 0; s < nsub; ++s) {
        rc = pulseqlib_build_freq_mod_library(
            &g_freqlibs[s], coll, s, g_fovshift);
        if (PULSEQLIB_FAILED(rc)) return rc;
    }
    return PULSEQLIB_OK;
}

/**
 * @brief Update the freq-mod library for a PMC-enabled subsequence.
 *
 * Recomputes 1D plan waveforms from the retained 3-channel entries
 * using the current g_fovshift.  No allocation -- O(entries x samples).
 */
static int update_freqmod_library(int subseq_idx)
{
    if (!g_freqlibs || subseq_idx >= g_nlibs || !g_freqlibs[subseq_idx])
        return PULSEQLIB_ERR_INVALID_ARGUMENT;
    return pulseqlib_update_freq_mod_library(
        g_freqlibs[subseq_idx], g_fovshift);
}

/**
 * @brief Look up precomputed freq-mod for a scan-table position.
 *
 * @param s           Subsequence index.
 * @param scan_pos    Scan-table position within the subsequence.
 * @return 1 if the block has a freq-mod event, 0 otherwise.
 */
static int get_freq_modulation(int s, int scan_pos,
                               const float** waveform,
                               int* nsamples,
                               float* phase_rad)
{
    *waveform = NULL;
    *nsamples = 0;
    *phase_rad = 0.0f;
    if (!g_freqlibs || s >= g_nlibs || !g_freqlibs[s]) return 0;
    return pulseqlib_freq_mod_library_get(
        g_freqlibs[s], scan_pos, waveform, nsamples, phase_rad);
}

/**
 * @brief Free all freq-mod libraries.
 */
static void free_freqmod_libraries(void)
{
    int s;
    if (!g_freqlibs) return;
    for (s = 0; s < g_nlibs; ++s)
        if (g_freqlibs[s]) pulseqlib_freq_mod_library_free(g_freqlibs[s]);
    free(g_freqlibs);
    g_freqlibs = NULL;
    g_nlibs    = 0;
}

/* ================================================================== */
/*  Vendor stubs                                                      */
/* ================================================================== */

/**
 * @brief Program one block on the hardware sequencer.
 */
static void vendor_set_block(const pulseqlib_block_instance* inst,
                             const float* fmod_waveform,
                             int fmod_nsamples,
                             float fmod_phase_rad)
{
    (void)inst;
    (void)fmod_waveform;
    (void)fmod_nsamples;
    (void)fmod_phase_rad;
}

/** @brief Set FOV rotation matrix for the next segment play. */
static void vendor_set_rotation(const float* rot)
{
    (void)rot;
}

/** @brief Arm the physio trigger gate for the next segment play. */
static void vendor_set_trigger(void)
{
    /* vendor_hw_arm_physio_trigger(); */
}

/** @brief Issue hardware play for the prepared segment. */
static void vendor_play_segment(int seg_idx)
{
    (void)seg_idx;
}

/**
 * @brief Evaluate PMC (navigator) feedback.
 *
 * In a real driver:
 *   1. Receives motion estimate from the reconstruction pipeline.
 *   2. If accepted: updates g_fovshift and g_fovrotation.
 *   3. Returns 0 (accepted, proceed) or 1 (rescan this TR).
 */
static int vendor_get_pmc_feedback(void)
{
    g_fovshift[2] += 0.001f;
    return 0;   /* 0 = accepted, 1 = rescan */
}

/* ================================================================== */
/*  Scan-table builder                                                */
/* ================================================================== */

/**
 * @brief Build a flat scan table for subsequence @p s.
 *
 * Flattens [prep, main x num_trs, cooldown] into a single ordered
 * array of segment IDs, matching the internal scan-table expansion.
 */
static int build_scan_table(const pulseqlib_collection* coll,
                            int s, int* out_ids)
{
    int prep[64], main_seg[64], cool[64];
    int n_p, n_m, n_c, n_tr, pos, i, tr;

    n_p  = pulseqlib_get_num_prep_segments(coll, s);
    n_m  = pulseqlib_get_num_main_segments(coll, s);
    n_c  = pulseqlib_get_num_cooldown_segments(coll, s);
    n_tr = pulseqlib_get_num_trs(coll, s);

    pulseqlib_get_prep_segment_table(coll, s, prep);
    pulseqlib_get_main_segment_table(coll, s, main_seg);
    pulseqlib_get_cooldown_segment_table(coll, s, cool);

    pos = 0;
    for (i = 0; i < n_p; ++i)
        out_ids[pos++] = prep[i];
    for (tr = 0; tr < n_tr; ++tr)
        for (i = 0; i < n_m; ++i)
            out_ids[pos++] = main_seg[i];
    for (i = 0; i < n_c; ++i)
        out_ids[pos++] = cool[i];

    return pos;
}

/* ================================================================== */
/*  Main                                                              */
/* ================================================================== */

int main(int argc, char** argv)
{
    const char*           seq_path;
    pulseqlib_opts        opts = PULSEQLIB_OPTS_INIT;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_collection* coll = NULL;
    int rc, nsub, s;
    int n = 0;   /* global block counter (across all subsequences) */

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <sequence.seq>\n", argv[0]);
        return 1;
    }
    seq_path = argv[1];

    vendor_opts_init(&opts, 42577478.0f, 3.0f, 50.0f, 200.0f);

    /* ============================================================== */
    /*  1. Load (with cache + labels)                                 */
    /* ============================================================== */
    rc = pulseqlib_read(&coll, &diag, seq_path, &opts, 1, 1, 1, 1);
    CHECK(rc, &diag);

    nsub = pulseqlib_get_num_subsequences(coll);
    printf("Loaded: %d subsequences, %.2f s\n",
           nsub, pulseqlib_get_total_duration_us(coll) / 1e6);

    /* ============================================================== */
    /*  2. Build per-subsequence freq-mod libraries                   */
    /* ============================================================== */
    rc = build_freqmod_libraries(coll);
    CHECK(rc, &diag);

    /* ============================================================== */
    /*  3. Scan loop                                                  */
    /* ============================================================== */
    {
        int trigger_on = 0;
        int is_nav     = 0;
        int reset      = 0;

        pulseqlib_cursor_reset(coll);

        for (s = 0; s < nsub; ++s) {
            int scan_table[4096];
            int scan_table_size;
            int n_prep, n_main, num_trs, pmc;
            int tr_start_i, tr_start_scan;
            int i;
            int scan_pos = 0;   /* per-subsequence scan-table position */

            n_prep  = pulseqlib_get_num_prep_segments(coll, s);
            n_main  = pulseqlib_get_num_main_segments(coll, s);
            num_trs = pulseqlib_get_num_trs(coll, s);
            pmc     = pulseqlib_is_pmc_enabled(coll, s);

            scan_table_size = build_scan_table(coll, s, scan_table);

            printf("\nSubseq %d: %d entries, %d TRs%s\n",
                   s, scan_table_size, num_trs,
                   pmc ? " [PMC]" : "");

            tr_start_i    = n_prep;
            tr_start_scan = scan_pos;

            i = 0;
            while (i < scan_table_size) {
                int seg_id      = scan_table[i];
                int seg_nblocks = pulseqlib_get_segment_num_blocks(
                                      coll, seg_id);
                int j;

                /* -------------------------------------------------- */
                /*  PMC sync at main-TR start                         */
                /* -------------------------------------------------- */
                {
                    int in_main = (i >= n_prep)
                                  && (i < n_prep + num_trs * n_main);
                    int at_tr_start = in_main
                                  && (((i - n_prep) % n_main) == 0);

                    if (pmc && at_tr_start) {
                        rc = update_freqmod_library(s);
                        if (PULSEQLIB_FAILED(rc)) goto fail;

                        if (reset) {
                            int k;
                            int rewind_blocks = scan_pos - tr_start_scan;
                            pulseqlib_cursor_reset(coll);
                            for (k = 0; k < n - rewind_blocks; ++k)
                                pulseqlib_cursor_next(coll);
                            i        = tr_start_i;
                            scan_pos = tr_start_scan;
                            n       -= rewind_blocks;
                            reset    = 0;
                            continue;
                        }

                        tr_start_i    = i;
                        tr_start_scan = scan_pos;
                    }
                }

                /* -------------------------------------------------- */
                /*  Iterate blocks in segment                         */
                /* -------------------------------------------------- */
                trigger_on = 0;
                is_nav     = pulseqlib_segment_is_nav(coll, seg_id);

                if (pulseqlib_segment_has_trigger(coll, seg_id))
                    trigger_on = 1;

                for (j = 0; j < seg_nblocks; ++j) {
                    pulseqlib_block_instance inst =
                        PULSEQLIB_BLOCK_INSTANCE_INIT;
                    const float* fmod_waveform = NULL;
                    int   fmod_nsamples = 0;
                    float fmod_phase    = 0.0f;

                    get_freq_modulation(s, scan_pos,
                                        &fmod_waveform,
                                        &fmod_nsamples, &fmod_phase);

                    if (pulseqlib_cursor_next(coll)
                            != PULSEQLIB_CURSOR_BLOCK)
                        goto fail;
                    if (PULSEQLIB_FAILED(
                            pulseqlib_get_block_instance(coll, &inst)))
                        goto fail;

                    vendor_set_block(&inst, fmod_waveform,
                                     fmod_nsamples, fmod_phase);
                    ++scan_pos;
                    ++n;
                }

                /* -------------------------------------------------- */
                /*  Play segment                                      */
                /* -------------------------------------------------- */
                vendor_set_rotation(g_fovrotation);
                if (trigger_on)
                    vendor_set_trigger();
                vendor_play_segment(seg_id);
                trigger_on = 0;

                /* -------------------------------------------------- */
                /*  PMC feedback after NAV                            */
                /* -------------------------------------------------- */
                if (pmc && is_nav) {
                    reset = vendor_get_pmc_feedback();
                }

                ++i;
            }
        }

        printf("\nScan loop complete: %d blocks\n", n);
    }

    /* ============================================================== */
    /*  4. Label readout                                              */
    /* ============================================================== */
    for (s = 0; s < nsub; ++s) {
        int num_adc  = pulseqlib_get_num_adc_occurrences(coll, s);
        int num_cols = pulseqlib_get_num_label_columns(coll, s);
        int occ, c;
        int label_vals[16];

        if (num_adc == 0 || num_cols == 0) continue;
        if (num_cols > 16) num_cols = 16;

        printf("\nSubseq %d: first 10 ADC labels "
               "(%d total, %d cols):\n", s, num_adc, num_cols);

        for (occ = 0; occ < num_adc && occ < 10; ++occ) {
            rc = pulseqlib_get_adc_label(coll, s, occ, label_vals);
            if (PULSEQLIB_FAILED(rc)) break;

            printf("  ADC %3d:", occ);
            for (c = 0; c < num_cols; ++c)
                printf(" %4d", label_vals[c]);
            printf("\n");
        }
        if (num_adc > 10)
            printf("  ... (%d more)\n", num_adc - 10);
    }

    /* ============================================================== */
    /*  Cleanup                                                       */
    /* ============================================================== */
    free_freqmod_libraries();
    pulseqlib_collection_free(coll);
    return 0;

fail:
    free_freqmod_libraries();
    if (coll) pulseqlib_collection_free(coll);
    return 1;
}
