/**
 * @file example_scanloop.c
 * @brief Segment-based real-time scan loop with PMC support.
 *
 * This example demonstrates the recommended scan-loop architecture
 * for a vendor driver.  The loop is structured around segments rather
 * than a flat cursor walk, giving the driver explicit control over:
 *
 *   - Subsequence ordering
 *   - Prep / main-TR / cooldown regions
 *   - Per-segment physio trigger wait + pure-delay handling
 *   - Per-block frequency modulation and FOV rotation
 *   - Prospective motion correction (PMC) with NAV rescan/rewind
 *
 * Workflow:
 *   1. Load the cached sequence collection.
 *   2. Outer loop over subsequences.
 *   3. For each subsequence:
 *        a. Query segment tables (prep, main, cooldown).
 *        b. Build freq-mod plans (per TR-region).
 *        c. Play prep segments (once).
 *        d. TR loop: play main segments per TR.
 *             - Pure-delay segments  → vendor_play_delay()
 *             - NAV segments (PMC)   → vendor_play_segment() then
 *                                      evaluate motion; rescan TR if
 *                                      threshold exceeded.
 *             - Normal segments      → iterate blocks via cursor,
 *                                      apply block_instance + freq-mod
 *                                      + FOV rotation, then
 *                                      vendor_play_segment().
 *        e. Play cooldown segments (once).
 *   4. Label readout for reconstruction metadata.
 *
 * Compile:
 *   cc -I../../csrc example_scanloop.c ../../csrc/pulseqlib_*.c -lm -o scanloop
 *
 * Run:
 *   ./scanloop path/to/sequence.seq
 */

#include "example_vendorlib.h"   /* must come first */
#include "pulseqlib_methods.h"

#include <stdio.h>
#include <string.h>
#include <math.h>

/* ================================================================== */
/*  Error check macro                                                 */
/* ================================================================== */

#define CHECK(rc, diag)                                 \
    do {                                                \
        if (PULSEQLIB_FAILED(rc)) {                     \
            vendor_report_error(rc, (diag));             \
            goto fail;                                  \
        }                                               \
    } while (0)

/* ================================================================== */
/*  Vendor stubs                                                      */
/* ================================================================== */
/*
 * In a real vendor integration each of these would talk to the
 * hardware sequencer.  Here they are no-ops for illustration.
 */

/** @brief Play a pure-delay segment (no waveforms). */
static void vendor_play_delay(int duration_us)
{
    printf("    [DELAY] %d us\n", duration_us);
    /* vendor_hw_idle(duration_us); */
}

/**
 * @brief Wait for a physio trigger (cardiac / respiratory gate).
 *
 * In a real driver, blocks until the trigger fires or a timeout
 * expires, then returns 1 (trigger received) or 0 (timeout).
 */
static int vendor_wait_trigger(int delay_us, int duration_us)
{
    printf("    [TRIGGER] wait delay=%d us duration=%d us\n",
           delay_us, duration_us);
    /* return vendor_hw_wait_physio(delay_us, duration_us); */
    return 1;
}

/**
 * @brief Evaluate NAV data and check motion threshold.
 *
 * @param[out] new_shift_m  Updated spatial shift (dx,dy,dz) if
 *                          motion is within threshold.
 * @return 1 if motion is within the acceptance window (proceed),
 *         0 if motion exceeds the threshold (rescan this TR).
 */
static int vendor_evaluate_nav(float new_shift_m[3])
{
    /*
     * In a real driver:
     *   1. Read navigator k-space / image data from the last ADC.
     *   2. Estimate translation (dx, dy, dz) and rotation.
     *   3. Compare against acceptance window.
     *   4. If accepted, write new_shift_m for freq-mod update.
     *
     * For this example, always accept with a small Z shift.
     */
    new_shift_m[0] = 0.05f;
    new_shift_m[1] = 0.0f;
    new_shift_m[2] = 0.001f;   /* 1 mm drift in Z */
    return 1;                  /* accepted */
}

/**
 * @brief Apply FOV rotation to gradient amplitudes.
 *
 * Multiplies the logical gradient amplitudes (gx, gy, gz) by the
 * block's 3x3 rotation matrix to produce physical-axis amplitudes.
 */
static void vendor_apply_rotation(pulseqlib_block_instance* inst)
{
    float lx = inst->gx_amp_hz_per_m;
    float ly = inst->gy_amp_hz_per_m;
    float lz = inst->gz_amp_hz_per_m;
    const float* R = inst->rotmat;

    inst->gx_amp_hz_per_m = R[0]*lx + R[1]*ly + R[2]*lz;
    inst->gy_amp_hz_per_m = R[3]*lx + R[4]*ly + R[5]*lz;
    inst->gz_amp_hz_per_m = R[6]*lx + R[7]*ly + R[8]*lz;
}

/**
 * @brief Apply frequency-modulation corrections to a block instance.
 *
 * For RF blocks, adds the grad-induced frequency offset to rf_freq_hz
 * and the phase compensation to rf_phase_rad.  For ADC blocks, adds
 * the centre-sample frequency offset to adc_freq_hz and the phase
 * compensation to adc_phase_rad.
 */
static void vendor_apply_freq_mod(
    pulseqlib_block_instance* inst,
    const pulseqlib_freq_mod_plan* plan,
    int block_idx)
{
    const float* fmod_waveform = NULL;
    int           fmod_nsamples = 0;
    float         fmod_phase_rad = 0.0f;

    if (!plan) return;

    if (pulseqlib_get_freq_mod_waveform(plan, block_idx, &fmod_waveform, &fmod_nsamples, &fmod_phase_rad))
    {
        vendor_set_freq_mod_waveform()
        if (inst->rf_amp_hz != 0.0f) {
            /*
             * For RF: the waveform contains instantaneous frequency
             * offsets at each sample point — the vendor would merge
             * this into the RF frequency modulation channel.
             * The scalar phase offset compensates the reference time.
             */
            inst->rf_phase_rad += fmod_phase_rad;
        }

        if (inst->adc_flag) {
            /*
             * For ADC: use the centre-sample frequency as the ADC
             * demodulation offset; apply phase compensation.
             */
            inst->adc_phase_rad += fmod_phase_rad;
        }
    }
}

/* ================================================================== */
/*  Print helper                                                      */
/* ================================================================== */

static void print_block(const pulseqlib_block_instance* b, int idx)
{
    printf("      [%04d] dur=%5d us", idx, b->duration_us);

    if (b->rf_amp_hz != 0.0f)
        printf("  RF(%.1f Hz, %.1f Hz, %.3f rad)",
               b->rf_amp_hz, b->rf_freq_hz, b->rf_phase_rad);

    if (b->gx_amp_hz_per_m != 0.0f)
        printf("  GX(%.0f)", b->gx_amp_hz_per_m);
    if (b->gy_amp_hz_per_m != 0.0f)
        printf("  GY(%.0f)", b->gy_amp_hz_per_m);
    if (b->gz_amp_hz_per_m != 0.0f)
        printf("  GZ(%.0f)", b->gz_amp_hz_per_m);

    if (b->adc_flag)
        printf("  ADC");
    if (b->digitalout_flag)
        printf("  DIGOUT");
    if (!b->norot_flag)
        printf("  ROT");

    printf("\n");
}

/* ================================================================== */
/*  Segment player                                                    */
/* ================================================================== */

/**
 * @brief Play one segment by iterating its blocks through the cursor.
 *
 * Advances the cursor through @p num_blocks blocks (the exact count
 * for a segment), resolving each block_instance, applying freq-mod
 * and FOV rotation, and printing the first few for demo purposes.
 *
 * @param coll          Collection (cursor state is advanced).
 * @param fmod_plan     Freq-mod plan (may be NULL).
 * @param[in,out] block_counter  Running block index (for freq-mod lookup).
 * @param[in,out] adc_counter    Running ADC count.
 * @param num_blocks    Number of blocks in this segment.
 * @param verbose_limit Print blocks while block_counter < this.
 * @return 0 on success, -1 on cursor error.
 */
static int play_segment_blocks(
    pulseqlib_collection* coll,
    const pulseqlib_freq_mod_plan* fmod_plan,
    int* block_counter,
    int* adc_counter,
    int  num_blocks,
    int  verbose_limit)
{
    int b;
    for (b = 0; b < num_blocks; ++b) {
        pulseqlib_block_instance inst = PULSEQLIB_BLOCK_INSTANCE_INIT;

        if (pulseqlib_cursor_next(coll) != PULSEQLIB_CURSOR_BLOCK)
            return -1;

        if (PULSEQLIB_FAILED(pulseqlib_get_block_instance(coll, &inst)))
            return -1;

        /* Frequency modulation */
        vendor_apply_freq_mod(&inst, fmod_plan, *block_counter);

        /* FOV rotation */
        if (!inst.norot_flag)
            vendor_apply_rotation(&inst);

        /* Demo print */
        if (*block_counter < verbose_limit)
            print_block(&inst, *block_counter);
        else if (*block_counter == verbose_limit)
            printf("      ... (remaining blocks omitted)\n");

        if (inst.adc_flag)
            (*adc_counter)++;

        (*block_counter)++;
    }
    return 0;
}

/* ================================================================== */
/*  Region player (prep / cooldown)                                   */
/* ================================================================== */

/**
 * @brief Play a list of segments (prep or cooldown region).
 *
 * Handles pure-delay segments, physio triggers, and normal
 * block-by-block iteration for each segment in the list.
 */
static int play_region(
    pulseqlib_collection* coll,
    const pulseqlib_freq_mod_plan* fmod_plan,
    const int* seg_ids,
    int        num_segments,
    int*       block_counter,
    int*       adc_counter,
    int        verbose_limit,
    const char* region_name)
{
    int s;
    for (s = 0; s < num_segments; ++s) {
        int seg_id     = seg_ids[s];
        int seg_nblocks = pulseqlib_get_segment_num_blocks(coll, seg_id);

        /* Pure-delay segment: skip cursor, just idle */
        if (pulseqlib_is_segment_pure_delay(coll, seg_id)) {
            int dur = pulseqlib_get_segment_duration_us(coll, seg_id);
            printf("  %s seg %d: pure delay\n", region_name, seg_id);
            vendor_play_delay(dur);
            continue;
        }

        /* Physio trigger at segment boundary */
        if (pulseqlib_segment_has_trigger(coll, seg_id)) {
            int delay = pulseqlib_get_segment_trigger_delay_us(coll, seg_id);
            int dur   = pulseqlib_get_segment_trigger_duration_us(coll, seg_id);
            printf("  %s seg %d: physio trigger\n", region_name, seg_id);
            vendor_wait_trigger(delay, dur);
        }

        /* Normal segment: iterate blocks */
        printf("  %s seg %d: %d blocks\n", region_name, seg_id, seg_nblocks);
        if (play_segment_blocks(coll, fmod_plan, block_counter,
                                adc_counter, seg_nblocks,
                                verbose_limit) < 0)
        {
            fprintf(stderr, "Cursor error in %s seg %d\n",
                    region_name, seg_id);
            return -1;
        }
    }
    return 0;
}

/* ================================================================== */
/*  Main                                                              */
/* ================================================================== */

int main(int argc, char** argv)
{
    const char*              seq_path;
    pulseqlib_opts           opts = PULSEQLIB_OPTS_INIT;
    pulseqlib_diagnostic     diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_collection*    coll = NULL;
    int rc;

    /* Per-subsequence freq-mod plans (freed at end) */
    pulseqlib_freq_mod_plan* fmod_prep = NULL;
    pulseqlib_freq_mod_plan* fmod_main = NULL;
    pulseqlib_freq_mod_plan* fmod_cool = NULL;

    /* Segment table scratch buffers (sized to MAX plausible) */
    int seg_ids_prep[64];
    int seg_ids_main[64];
    int seg_ids_cool[64];

    int block_counter = 0;
    int adc_counter   = 0;
    int nsub, s;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <sequence.seq>\n", argv[0]);
        return 1;
    }
    seq_path = argv[1];

    /* --- Scanner parameters (Siemens 3 T example) --- */
    vendor_opts_init(&opts, 42577478.0f, 3.0f, 50.0f, 200.0f);

    /* ============================================================== */
    /*  1. Load (with cache + labels)                                 */
    /* ============================================================== */
    rc = pulseqlib_read(&coll, &diag, seq_path, &opts, 1, 1, 1, 1);
    CHECK(rc, &diag);

    nsub = pulseqlib_get_num_subsequences(coll);
    printf("Loaded: %d subsequences, total duration %.2f s\n",
           nsub, pulseqlib_get_total_duration_us(coll) / 1e6);

    /* ============================================================== */
    /*  2. Label table overview                                       */
    /* ============================================================== */
    for (s = 0; s < nsub; ++s) {
        int num_adc  = pulseqlib_get_num_adc_occurrences(coll, s);
        int num_cols = pulseqlib_get_num_label_columns(coll, s);
        printf("  Subseq %d: %d TRs, %d ADC occurrences, %d label cols",
               s, pulseqlib_get_num_trs(coll, s), num_adc, num_cols);
        if (pulseqlib_is_pmc_enabled(coll, s))
            printf("  [PMC]");
        printf("\n");
    }

    /* ============================================================== */
    /*  3. Segment-based scan loop                                    */
    /* ============================================================== */
    /*
     * The loop walks subsequences → regions → segments → blocks.
     * The cursor is used inside play_segment_blocks() to iterate
     * individual blocks; segment-level decisions (pure delay,
     * trigger wait, NAV evaluation) happen at a higher level.
     *
     * Frequency-modulation plans are built per TR-region so that
     * the block_idx passed to get_freq_mod_waveform matches the
     * plan's internal indexing.  For the first TR (which includes
     * prep blocks) and for cooldown, we build separate plans to
     * get the correct waveform-to-block mapping.
     */

    /* Example: 5 cm shift in X, on-isocenter in Y/Z */
    {
        float shift_m[3] = {0.05f, 0.0f, 0.0f};

        printf("\n--- Scan loop ---\n");
        pulseqlib_cursor_reset(coll);

        for (s = 0; s < nsub; ++s) {
            int n_prep = pulseqlib_get_num_prep_segments(coll, s);
            int n_main = pulseqlib_get_num_main_segments(coll, s);
            int n_cool = pulseqlib_get_num_cooldown_segments(coll, s);
            int num_trs = pulseqlib_get_num_trs(coll, s);
            int pmc     = pulseqlib_is_pmc_enabled(coll, s);
            int tr;

            printf("\n=== Subsequence %d: %d TRs, "
                   "segments prep=%d main=%d cool=%d%s ===\n",
                   s, num_trs, n_prep, n_main, n_cool,
                   pmc ? " [PMC]" : "");

            /* --- Fetch segment tables --- */
            pulseqlib_get_prep_segment_table(coll, s, seg_ids_prep);
            pulseqlib_get_main_segment_table(coll, s, seg_ids_main);
            pulseqlib_get_cooldown_segment_table(coll, s, seg_ids_cool);

            /* -------------------------------------------------------- */
            /*  Build freq-mod plans per region                         */
            /* -------------------------------------------------------- */
            /*
             * PREP plan:   covers prep blocks + first main TR.
             * MAIN plan:   covers one steady-state main TR (reused
             *              for TRs 1 .. num_trs-1, updated for PMC).
             * COOLDOWN plan: covers last main TR + cooldown blocks.
             *
             * Block indices within each plan start at 0 and are
             * independent of the global block counter.
             */
            if (fmod_prep) { pulseqlib_freq_mod_plan_free(fmod_prep); fmod_prep = NULL; }
            if (fmod_main) { pulseqlib_freq_mod_plan_free(fmod_main); fmod_main = NULL; }
            if (fmod_cool) { pulseqlib_freq_mod_plan_free(fmod_cool); fmod_cool = NULL; }

            rc = pulseqlib_build_freq_mod_plan(
                &fmod_prep, coll, shift_m,
                PULSEQLIB_TR_REGION_PREP, 0);
            CHECK(rc, &diag);

            rc = pulseqlib_build_freq_mod_plan(
                &fmod_main, coll, shift_m,
                PULSEQLIB_TR_REGION_MAIN, 0);
            CHECK(rc, &diag);

            rc = pulseqlib_build_freq_mod_plan(
                &fmod_cool, coll, shift_m,
                PULSEQLIB_TR_REGION_COOLDOWN, 0);
            CHECK(rc, &diag);

            /* -------------------------------------------------------- */
            /*  3a. Play prep segments (once)                           */
            /* -------------------------------------------------------- */
            if (n_prep > 0) {
                printf("\n  --- Prep region ---\n");
                if (play_region(coll, fmod_prep, seg_ids_prep, n_prep,
                                &block_counter, &adc_counter, 20,
                                "PREP") < 0)
                    goto fail;
            }

            /* -------------------------------------------------------- */
            /*  3b. Main TR loop                                        */
            /* -------------------------------------------------------- */
            for (tr = 0; tr < num_trs; ++tr) {
                int seg_s;
                int tr_rescan = 0;

                printf("\n  --- TR %d/%d ---\n", tr + 1, num_trs);

            rescan_tr:

                for (seg_s = 0; seg_s < n_main; ++seg_s) {
                    int seg_id     = seg_ids_main[seg_s];
                    int seg_nblocks = pulseqlib_get_segment_num_blocks(
                                         coll, seg_id);

                    /* -- Pure-delay segment -- */
                    if (pulseqlib_is_segment_pure_delay(coll, seg_id)) {
                        int dur = pulseqlib_get_segment_duration_us(
                                      coll, seg_id);
                        printf("  MAIN seg %d: pure delay\n", seg_id);
                        vendor_play_delay(dur);
                        continue;
                    }

                    /* -- Physio trigger at segment boundary -- */
                    if (pulseqlib_segment_has_trigger(coll, seg_id)) {
                        int delay = pulseqlib_get_segment_trigger_delay_us(
                                        coll, seg_id);
                        int dur   = pulseqlib_get_segment_trigger_duration_us(
                                        coll, seg_id);
                        printf("  MAIN seg %d: physio trigger\n", seg_id);
                        vendor_wait_trigger(delay, dur);
                    }

                    /* -- NAV segment (PMC) -- */
                    if (pmc && pulseqlib_segment_is_nav(coll, seg_id)) {
                        float nav_shift[3];

                        printf("  MAIN seg %d: NAV (%d blocks)\n",
                               seg_id, seg_nblocks);

                        /* Play the NAV segment blocks normally */
                        if (play_segment_blocks(coll, fmod_main,
                                                &block_counter,
                                                &adc_counter,
                                                seg_nblocks, 20) < 0)
                            goto fail;

                        /* Evaluate navigator result */
                        if (!vendor_evaluate_nav(nav_shift)) {
                            /*
                             * Motion exceeds threshold — rescan this TR.
                             * Reset the cursor to the start of the TR and
                             * update the freq-mod plan with the last
                             * accepted position.
                             *
                             * In a real driver, rewind hardware buffers
                             * and re-arm acquisition for this TR.
                             */
                            tr_rescan++;
                            if (tr_rescan > 5) {
                                printf("    PMC: max rescans reached, "
                                       "accepting with drift\n");
                            } else {
                                printf("    PMC: motion rejected, "
                                       "rescan #%d\n", tr_rescan);
                                /*
                                 * NOTE: in a real driver you would
                                 * pulseqlib_cursor_reset() and advance
                                 * to the start of this TR, or use a
                                 * vendor-specific rewind mechanism.
                                 * For this demo we simply re-enter
                                 * the segment loop.
                                 */
                                goto rescan_tr;
                            }
                        } else {
                            printf("    PMC: motion accepted "
                                   "(dz=%.1f mm)\n",
                                   nav_shift[2] * 1000.0f);

                            /* Update freq-mod plan with new position */
                            pulseqlib_update_freq_mod_plan(
                                fmod_main, nav_shift);
                        }

                        continue;  /* NAV segment already played */
                    }

                    /* -- Normal segment: iterate blocks -- */
                    printf("  MAIN seg %d: %d blocks\n",
                           seg_id, seg_nblocks);
                    if (play_segment_blocks(coll, fmod_main,
                                            &block_counter,
                                            &adc_counter,
                                            seg_nblocks, 20) < 0)
                    {
                        fprintf(stderr,
                                "Cursor error in TR %d seg %d\n",
                                tr, seg_id);
                        goto fail;
                    }
                }
            }

            /* -------------------------------------------------------- */
            /*  3c. Play cooldown segments (once)                       */
            /* -------------------------------------------------------- */
            if (n_cool > 0) {
                printf("\n  --- Cooldown region ---\n");
                if (play_region(coll, fmod_cool, seg_ids_cool, n_cool,
                                &block_counter, &adc_counter, 20,
                                "COOL") < 0)
                    goto fail;
            }
        }

        printf("\nScan loop complete: %d blocks, %d ADC windows\n",
               block_counter, adc_counter);
    }

    /* ============================================================== */
    /*  4. Label readout — reconstruction metadata                    */
    /* ============================================================== */
    /*
     * After the scan loop, read back labels for each ADC occurrence.
     * This tells the reconstruction pipeline where each readout goes
     * in k-space (lin, slc, eco, ...).
     */
    for (s = 0; s < nsub; ++s) {
        int num_adc  = pulseqlib_get_num_adc_occurrences(coll, s);
        int num_cols = pulseqlib_get_num_label_columns(coll, s);
        int occ, c;
        int label_vals[16];

        if (num_adc == 0 || num_cols == 0) continue;
        if (num_cols > 16) num_cols = 16;

        printf("\nSubseq %d: first 10 ADC labels "
               "(of %d total, %d columns):\n", s, num_adc, num_cols);

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
    if (fmod_prep) pulseqlib_freq_mod_plan_free(fmod_prep);
    if (fmod_main) pulseqlib_freq_mod_plan_free(fmod_main);
    if (fmod_cool) pulseqlib_freq_mod_plan_free(fmod_cool);
    pulseqlib_collection_free(coll);
    return 0;

fail:
    if (fmod_prep) pulseqlib_freq_mod_plan_free(fmod_prep);
    if (fmod_main) pulseqlib_freq_mod_plan_free(fmod_main);
    if (fmod_cool) pulseqlib_freq_mod_plan_free(fmod_cool);
    if (coll)      pulseqlib_collection_free(coll);
    return 1;
}
