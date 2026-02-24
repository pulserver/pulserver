/**
 * @file example_geninstructions.c
 * @brief Generate per-segment base instructions from a cached collection.
 *
 * Workflow:
 *   1. Load the cached sequence collection (fast .bin reload).
 *   2. Query the segment table (prep / main / cooldown).
 *   3. For each unique segment, build a "base instruction" by reading
 *      block definitions (RF, gradients, ADC, trigger, rotation).
 *   4. Print the instruction table — in a real driver this would be
 *      written into hardware instruction memory.
 *
 * The idea is that segment definitions are *static* — they describe
 * the waveform shapes.  The dynamic part (amplitudes, frequencies,
 * phases, shot indices) is handled at scan time by the cursor loop
 * (see example_scanloop.c).
 *
 * Compile:
 *   cc -I../../csrc example_geninstructions.c ../../csrc/pulseqlib_*.c -lm -o geninstructions
 *
 * Run:
 *   ./geninstructions path/to/sequence.seq
 */

#include "example_vendorlib.h"   /* must come first */
#include "pulseqlib_methods.h"

#include <stdio.h>
#include <string.h>

#define CHECK(rc, diag)                                 \
    do {                                                \
        if (PULSEQLIB_FAILED(rc)) {                     \
            vendor_report_error(rc, (diag));             \
            goto fail;                                  \
        }                                               \
    } while (0)

/* ================================================================== */
/*  Per-block instruction (vendor-specific)                           */
/* ================================================================== */

/**
 * @brief Minimal example of a vendor instruction for one block.
 *
 * In a real driver these fields map to hardware registers or DMA
 * descriptors — the exact layout is vendor-specific.  This struct
 * captures the *base* (initial) values; the cursor loop in
 * example_scanloop.c patches amplitudes / phases / shots at runtime.
 */
typedef struct vendor_block_instruction {
    /* timing */
    int   duration_us;

    /* RF — base definition */
    int   has_rf;
    int   rf_num_samples;
    int   rf_num_channels;
    int   rf_delay_us;
    int   rf_is_complex;        /* has nonzero phase shape? */
    float rf_max_amplitude_hz;  /* peak |gamma*B1| */
    /* rf waveform pointer(s) would go here */

    /* Gradient — per axis */
    struct {
        int   has_grad;
        int   is_trapezoid;
        int   num_shots;
        int   num_samples;
        int   delay_us;
        float initial_amplitude;
        /* gradient waveform pointer would go here */
    } grad[3]; /* [X, Y, Z] */

    /* ADC */
    int   has_adc;
    int   adc_num_samples;
    int   adc_dwell_us;
    int   adc_delay_us;

    /* Flow control */
    int   has_trigger;
    int   trigger_delay_us;
    int   has_rotation;
    int   has_norot;
    int   has_nopos;
} vendor_block_instruction;

/* ================================================================== */
/*  Build base instruction for one block in a segment                 */
/* ================================================================== */

static void build_block_instruction(
    vendor_block_instruction* instr,
    const pulseqlib_collection* coll,
    int seg_idx,
    int blk_idx)
{
    int axis;

    memset(instr, 0, sizeof(*instr));

    /* Timing */
    instr->duration_us = pulseqlib_get_block_duration_us(coll, seg_idx, blk_idx);

    /* RF */
    instr->has_rf = pulseqlib_block_has_rf(coll, seg_idx, blk_idx);
    if (instr->has_rf) {
        instr->rf_num_samples  = pulseqlib_get_rf_num_samples(coll, seg_idx, blk_idx);
        instr->rf_num_channels = pulseqlib_get_rf_num_channels(coll, seg_idx, blk_idx);
        instr->rf_delay_us     = pulseqlib_get_rf_delay_us(coll, seg_idx, blk_idx);
        instr->rf_is_complex   = pulseqlib_block_rf_is_complex(coll, seg_idx, blk_idx);

        /*
         * In a real driver you would also call:
         *
         *   float** mag = pulseqlib_get_rf_magnitude(coll, seg_idx, blk_idx,
         *                                            &nch, &ns);
         *   float** phs = pulseqlib_get_rf_phase(coll, seg_idx, blk_idx,
         *                                        &nch, &ns);
         *   float*  t   = pulseqlib_get_rf_time_us(coll, seg_idx, blk_idx, &ns);
         *
         * and copy / DMA-map the waveform data into hardware memory.
         * Remember to PULSEQLIB_FREE each channel array and the array
         * of pointers afterwards.
         */
    }

    /* Gradients (X=0, Y=1, Z=2) */
    for (axis = 0; axis < 3; ++axis) {
        instr->grad[axis].has_grad = pulseqlib_block_has_grad(
            coll, seg_idx, blk_idx, axis);

        if (!instr->grad[axis].has_grad) continue;

        instr->grad[axis].is_trapezoid = pulseqlib_block_grad_is_trapezoid(
            coll, seg_idx, blk_idx, axis);
        instr->grad[axis].num_shots = pulseqlib_get_grad_num_shots(
            coll, seg_idx, blk_idx, axis);
        instr->grad[axis].num_samples = pulseqlib_get_grad_num_samples(
            coll, seg_idx, blk_idx, axis);
        instr->grad[axis].delay_us = pulseqlib_get_grad_delay_us(
            coll, seg_idx, blk_idx, axis);
        instr->grad[axis].initial_amplitude =
            pulseqlib_get_grad_initial_amplitude_hz_per_m(
                coll, seg_idx, blk_idx, axis);

        /*
         * In a real driver you would also call:
         *
         *   int* ns_per_shot;
         *   float** amps = pulseqlib_get_grad_amplitude(
         *       coll, seg_idx, blk_idx, axis, &num_shots, &ns_per_shot);
         *   float* t = pulseqlib_get_grad_time_us(
         *       coll, seg_idx, blk_idx, axis, &ns);
         *
         * and upload each shot waveform into hardware memory.
         * For multi-shot gradients the cursor loop selects the
         * active shot index at runtime.
         */
    }

    /* ADC */
    instr->has_adc = pulseqlib_block_has_adc(coll, seg_idx, blk_idx);
    if (instr->has_adc) {
        int adc_lib_idx = pulseqlib_get_adc_library_index(coll, seg_idx, blk_idx);
        instr->adc_delay_us    = pulseqlib_get_adc_delay_us(coll, seg_idx, blk_idx);
        instr->adc_num_samples = pulseqlib_get_adc_num_samples(coll, adc_lib_idx);
        instr->adc_dwell_us    = pulseqlib_get_adc_dwell_us(coll, adc_lib_idx);
    }

    /* Flow control */
    instr->has_trigger   = pulseqlib_block_has_trigger(coll, seg_idx, blk_idx);
    if (instr->has_trigger)
        instr->trigger_delay_us = pulseqlib_get_trigger_delay_us(coll, seg_idx, blk_idx);
    instr->has_rotation  = pulseqlib_block_has_rotation(coll, seg_idx, blk_idx);
    instr->has_norot     = pulseqlib_block_has_norot(coll, seg_idx, blk_idx);
    instr->has_nopos     = pulseqlib_block_has_nopos(coll, seg_idx, blk_idx);
}

/* ================================================================== */
/*  Print helpers                                                     */
/* ================================================================== */

static void print_block_instruction(const vendor_block_instruction* instr,
                                    int seg_idx, int blk_idx)
{
    const char* axis_name[] = {"GX", "GY", "GZ"};
    int a;

    printf("  Block [seg=%d, blk=%d]  dur=%d us", seg_idx, blk_idx,
           instr->duration_us);

    if (instr->has_rf)
        printf("  RF(%d samp, %d ch, delay=%d us)",
               instr->rf_num_samples, instr->rf_num_channels,
               instr->rf_delay_us);

    for (a = 0; a < 3; ++a) {
        if (instr->grad[a].has_grad)
            printf("  %s(%s, %d shots, %d samp)",
                   axis_name[a],
                   instr->grad[a].is_trapezoid ? "trap" : "arb",
                   instr->grad[a].num_shots,
                   instr->grad[a].num_samples);
    }

    if (instr->has_adc)
        printf("  ADC(%d samp, dwell=%d us)",
               instr->adc_num_samples, instr->adc_dwell_us);
    if (instr->has_trigger)
        printf("  TRIG(delay=%d us)", instr->trigger_delay_us);

    printf("\n");
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
    int rc, s, nseg, nblk, seg_idx, blk_idx;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <sequence.seq>\n", argv[0]);
        return 1;
    }
    seq_path = argv[1];

    vendor_opts_init(&opts);

    /* -- Load (with cache) ---------------------------------------- */
    rc = pulseqlib_read(&coll, &diag, seq_path, &opts, 1, 1, 1, 1);
    CHECK(rc, &diag);

    /* -- Walk segments -------------------------------------------- */
    nseg = pulseqlib_get_num_segments(coll);
    printf("Total unique segments: %d\n\n", nseg);

    for (seg_idx = 0; seg_idx < nseg; ++seg_idx) {
        int  seg_dur_us  = pulseqlib_get_segment_duration_us(coll, seg_idx);
        int  pure_delay  = pulseqlib_is_segment_pure_delay(coll, seg_idx);
        int  start_block = pulseqlib_get_segment_start_block(coll, seg_idx);

        nblk = pulseqlib_get_segment_num_blocks(coll, seg_idx);

        printf("Segment %d: %d blocks, duration=%d us, start_block=%d%s\n",
               seg_idx, nblk, seg_dur_us, start_block,
               pure_delay ? " (pure delay)" : "");

        for (blk_idx = 0; blk_idx < nblk; ++blk_idx) {
            vendor_block_instruction instr;
            build_block_instruction(&instr, coll, seg_idx, blk_idx);
            print_block_instruction(&instr, seg_idx, blk_idx);
        }
        printf("\n");
    }

    /* -- Print segment tables for each subsequence ---------------- */
    {
        int nsub = pulseqlib_get_num_subsequences(coll);
        for (s = 0; s < nsub; ++s) {
            int n_prep = pulseqlib_get_num_prep_segments(coll, s);
            int n_main = pulseqlib_get_num_main_segments(coll, s);
            int n_cool = pulseqlib_get_num_cooldown_segments(coll, s);
            int i;

            /* Stack-allocate small tables (real code would use ALLOC) */
            int prep_ids[64], main_ids[256], cool_ids[64];

            printf("Subseq %d segment tables:\n", s);

            if (n_prep > 0 && n_prep <= 64) {
                pulseqlib_get_prep_segment_table(coll, s, prep_ids);
                printf("  Prep:     ");
                for (i = 0; i < n_prep; ++i) printf("%d ", prep_ids[i]);
                printf("\n");
            }

            if (n_main > 0 && n_main <= 256) {
                pulseqlib_get_main_segment_table(coll, s, main_ids);
                printf("  Main:     ");
                for (i = 0; i < n_main; ++i) printf("%d ", main_ids[i]);
                printf("\n");
            }

            if (n_cool > 0 && n_cool <= 64) {
                pulseqlib_get_cooldown_segment_table(coll, s, cool_ids);
                printf("  Cooldown: ");
                for (i = 0; i < n_cool; ++i) printf("%d ", cool_ids[i]);
                printf("\n");
            }
            printf("\n");
        }
    }

    /* -- RF statistics -------------------------------------------- */
    {
        int nsub = pulseqlib_get_num_subsequences(coll);
        for (s = 0; s < nsub; ++s) {
            int nrf = pulseqlib_get_num_unique_rf(coll, s);
            int r;
            printf("Subseq %d: %d unique RF event(s)\n", s, nrf);
            for (r = 0; r < nrf; ++r) {
                pulseqlib_rf_stats stats = PULSEQLIB_RF_STATS_INIT;
                rc = pulseqlib_get_rf_stats(coll, &stats, s, r);
                if (PULSEQLIB_SUCCEEDED(rc)) {
                    printf("  RF %d: flip=%.1f deg, bw=%.0f Hz, "
                           "dur=%.0f us, max_amp=%.0f Hz\n",
                           r, stats.flip_angle_deg, stats.bandwidth_hz,
                           stats.duration_us, stats.max_amplitude_hz);
                }
            }
        }
    }

    /* -- Label limits (if labels were parsed) --------------------- */
    {
        pulseqlib_label_limits limits;
        rc = pulseqlib_get_label_limits(coll, 0, &limits);
        if (PULSEQLIB_SUCCEEDED(rc)) {
            printf("\nLabel limits (subseq 0):\n");
            printf("  LIN: [%d, %d]\n", limits.lin.min, limits.lin.max);
            printf("  SLC: [%d, %d]\n", limits.slc.min, limits.slc.max);
            printf("  ECO: [%d, %d]\n", limits.eco.min, limits.eco.max);
            printf("  REP: [%d, %d]\n", limits.rep.min, limits.rep.max);
            printf("  AVG: [%d, %d]\n", limits.avg.min, limits.avg.max);
            printf("  SEG: [%d, %d]\n", limits.seg.min, limits.seg.max);
            printf("  SET: [%d, %d]\n", limits.set.min, limits.set.max);
            printf("  PAR: [%d, %d]\n", limits.par.min, limits.par.max);
        }
    }

    pulseqlib_collection_free(coll);
    return 0;

fail:
    if (coll) pulseqlib_collection_free(coll);
    return 1;
}
