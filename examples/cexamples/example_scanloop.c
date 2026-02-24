/**
 * @file example_scanloop.c
 * @brief Real-time scan loop using the block cursor and frequency modulation.
 *
 * Workflow:
 *   1. Load the cached sequence collection.
 *   2. Build a frequency-modulation plan (for off-isocenter imaging).
 *   3. Run the block cursor in a tight loop — this is the inner scan
 *      loop that a vendor driver would execute at TR cadence.
 *   4. For each block, retrieve the resolved block_instance
 *      (runtime amplitudes, phases, rotation), apply freq-mod,
 *      and access the label table for reconstruction metadata.
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

#define CHECK(rc, diag)                                 \
    do {                                                \
        if (PULSEQLIB_FAILED(rc)) {                     \
            vendor_report_error(rc, (diag));             \
            goto fail;                                  \
        }                                               \
    } while (0)

/* ================================================================== */
/*  Print helpers                                                     */
/* ================================================================== */

static void print_block_instance(const pulseqlib_block_instance* b,
                                 int block_counter)
{
    printf("  [%04d] dur=%5d us", block_counter, b->duration_us);

    /* RF */
    if (b->rf_amp_hz != 0.0f)
        printf("  RF(amp=%.1f Hz, freq=%.1f Hz, phase=%.3f rad)",
               b->rf_amp_hz, b->rf_freq_hz, b->rf_phase_rad);

    /* Gradients */
    if (b->gx_amp_hz_per_m != 0.0f)
        printf("  GX(amp=%.0f, shot=%d)", b->gx_amp_hz_per_m, b->gx_shot_idx);
    if (b->gy_amp_hz_per_m != 0.0f)
        printf("  GY(amp=%.0f, shot=%d)", b->gy_amp_hz_per_m, b->gy_shot_idx);
    if (b->gz_amp_hz_per_m != 0.0f)
        printf("  GZ(amp=%.0f, shot=%d)", b->gz_amp_hz_per_m, b->gz_shot_idx);

    /* ADC */
    if (b->adc_flag)
        printf("  ADC(freq=%.1f Hz, phase=%.3f rad)",
               b->adc_freq_hz, b->adc_phase_rad);

    /* Rotation */
    if (!b->norot_flag) {
        /* Check if rotation differs from identity */
        if (b->rotmat[0] != 1.0f || b->rotmat[4] != 1.0f || b->rotmat[8] != 1.0f)
            printf("  ROT");
    }

    if (b->trigon_flag)
        printf("  TRIG");

    printf("\n");
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
    pulseqlib_freq_mod_plan* fmod_plan = NULL;
    int rc;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <sequence.seq>\n", argv[0]);
        return 1;
    }
    seq_path = argv[1];

    vendor_opts_init(&opts);

    /* ============================================================== */
    /*  1. Load (with cache + labels)                                 */
    /* ============================================================== */
    rc = pulseqlib_read(&coll, &diag, seq_path, &opts, 1, 1, 1, 1);
    CHECK(rc, &diag);

    printf("Loaded: %d subsequences, total duration %.2f s\n",
           pulseqlib_get_num_subsequences(coll),
           pulseqlib_get_total_duration_us(coll) / 1e6);

    /* ============================================================== */
    /*  2. Label table overview                                       */
    /* ============================================================== */
    {
        int nsub = pulseqlib_get_num_subsequences(coll);
        int s;
        for (s = 0; s < nsub; ++s) {
            int num_adc  = pulseqlib_get_num_adc_occurrences(coll, s);
            int num_cols = pulseqlib_get_num_label_columns(coll, s);
            printf("Subseq %d: %d ADC occurrences, %d label columns\n",
                   s, num_adc, num_cols);
        }
    }

    /* ============================================================== */
    /*  3. Build frequency-modulation plan                            */
    /* ============================================================== */
    /*
     * The freq-mod plan precomputes gradient-induced frequency offsets
     * for off-isocenter imaging.  Set shift_m to the patient-table
     * offset or PRS shift from the prescription.
     *
     * For on-isocenter imaging, skip this step entirely -- the plan
     * pointer can be NULL and get_freq_mod_waveform returns 0.
     */
    {
        /* Example: 5 cm shift in X, on-isocenter in Y/Z */
        float shift_m[3] = {0.05f, 0.0f, 0.0f};

        int n_events = pulseqlib_get_freq_mod_count(coll);
        printf("Freq-mod events across full sequence: %d\n", n_events);

        /* Build plan for the entire sequence (all TRs) */
        rc = pulseqlib_build_freq_mod_plan(
            &fmod_plan, coll, shift_m,
            PULSEQLIB_TR_REGION_ALL, 0);
        CHECK(rc, &diag);
    }

    /* ============================================================== */
    /*  4. Main scan loop — cursor iteration                          */
    /* ============================================================== */
    /*
     * The cursor walks through the flattened scan table, which
     * expands the segment/TR structure into an ordered block stream.
     * Each call to cursor_next() advances to the next block; the
     * returned block_instance contains all runtime-resolved values
     * (amplitudes, phases, shot indices, rotation matrix).
     *
     * In a real vendor driver this loop is the hardware sequencer's
     * inner loop — you'd convert block_instance fields into register
     * writes or DMA descriptors.
     */
    {
        pulseqlib_block_instance inst = PULSEQLIB_BLOCK_INSTANCE_INIT;
        int block_counter = 0;
        int adc_counter   = 0;
        int total_adc_us  = 0;

        printf("\n--- Scan loop ---\n");

        while (pulseqlib_cursor_next(coll) == PULSEQLIB_CURSOR_BLOCK) {
            rc = pulseqlib_get_block_instance(coll, &inst);
            if (PULSEQLIB_FAILED(rc)) {
                fprintf(stderr, "get_block_instance failed at block %d\n",
                        block_counter);
                break;
            }

            /* ---- Print first 20 blocks for demo ---- */
            if (block_counter < 20)
                print_block_instance(&inst, block_counter);
            else if (block_counter == 20)
                printf("  ... (remaining blocks omitted)\n");

            /* ---- Apply frequency modulation ---- */
            if (fmod_plan) {
                const float* fmod_waveform = NULL;
                int           fmod_nsamples = 0;
                float         fmod_phase_rad = 0.0f;

                int has_fmod = pulseqlib_get_freq_mod_waveform(
                    fmod_plan, block_counter,
                    &fmod_waveform, &fmod_nsamples, &fmod_phase_rad);

                if (has_fmod) {
                    /*
                     * In a real driver:
                     *
                     *  - For RF blocks: add fmod_waveform to the base
                     *    RF frequency offset (inst.rf_freq_hz already
                     *    contains the sequence-defined offset).
                     *    Add fmod_phase_rad to inst.rf_phase_rad.
                     *
                     *  - For ADC blocks: set the ADC demod frequency
                     *    to inst.adc_freq_hz + fmod_waveform[center].
                     *    Add fmod_phase_rad to inst.adc_phase_rad.
                     *
                     * The waveform pointer points into plan memory —
                     * do NOT free it.
                     */
                    (void)fmod_waveform;   /* suppress unused warning */
                    (void)fmod_phase_rad;
                }
            }

            /* ---- Rotation / repositioning ---- */
            if (!inst.norot_flag) {
                /*
                 * Apply inst.rotmat[9] to gradient amplitudes for
                 * oblique slice orientation.  The matrix is already
                 * in logical-to-physical (PRS) convention.
                 *
                 * float phys_gx = rotmat[0]*gx + rotmat[1]*gy + rotmat[2]*gz;
                 * float phys_gy = rotmat[3]*gx + rotmat[4]*gy + rotmat[5]*gz;
                 * float phys_gz = rotmat[6]*gx + rotmat[7]*gy + rotmat[8]*gz;
                 */
            }

            /* ---- Track ADC occurrences ---- */
            if (inst.adc_flag) {
                adc_counter++;
                total_adc_us += inst.duration_us;
            }

            block_counter++;
        }

        printf("Scan loop complete: %d blocks, %d ADC windows, "
               "%.2f s total ADC time\n",
               block_counter, adc_counter, total_adc_us / 1e6);
    }

    /* ============================================================== */
    /*  5. Prospective motion correction — update plan in-place       */
    /* ============================================================== */
    /*
     * For navigated / prospective motion correction, call
     * update_freq_mod_plan() before each TR (or each nav cycle)
     * with the updated patient position, then cursor_reset +
     * re-iterate (or advance within the already-running loop).
     *
     * Example: the navigator measured a 2 mm shift in Z.
     */
    {
        float new_shift_m[3] = {0.05f, 0.0f, 0.002f};

        rc = pulseqlib_update_freq_mod_plan(fmod_plan, new_shift_m);
        if (PULSEQLIB_SUCCEEDED(rc))
            printf("\nFreq-mod plan updated for motion correction.\n");

        /*
         * If you need to re-run the cursor from the top:
         *
         *   pulseqlib_cursor_reset(coll);
         *   while (pulseqlib_cursor_next(coll) == PULSEQLIB_CURSOR_BLOCK) { ... }
         */
    }

    /* ============================================================== */
    /*  6. Label readout — reconstruction metadata                    */
    /* ============================================================== */
    /*
     * After the scan loop, read back labels for each ADC occurrence.
     * This tells the reconstruction pipeline where each readout goes
     * in k-space (lin, slc, eco, ...).
     */
    {
        int nsub = pulseqlib_get_num_subsequences(coll);
        int s;
        for (s = 0; s < nsub; ++s) {
            int num_adc  = pulseqlib_get_num_adc_occurrences(coll, s);
            int num_cols = pulseqlib_get_num_label_columns(coll, s);
            int occ, c;
            int label_vals[16]; /* large enough for any vendor */

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
    }

    /* ============================================================== */
    /*  Cleanup                                                       */
    /* ============================================================== */
    pulseqlib_freq_mod_plan_free(fmod_plan);
    pulseqlib_collection_free(coll);
    return 0;

fail:
    if (fmod_plan) pulseqlib_freq_mod_plan_free(fmod_plan);
    if (coll)      pulseqlib_collection_free(coll);
    return 1;
}
