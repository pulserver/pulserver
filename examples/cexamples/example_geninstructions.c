/**
 * @file example_geninstructions.c
 * @brief Generate hardware instructions from a cached collection.
 *
 * Workflow (per segment, per block):
 *   FOREACH segment:
 *     query RF-ADC gap for vendor timing optimization
 *     t = 0
 *     FOREACH block in segment:
 *       get gradient waveforms  (X, Y, Z)
 *       get RF waveforms        (magnitude + optional phase + optional time)
 *       get ADC definition ID   (maps to echo filter from check phase)
 *       get trigger info        (delay + duration)
 *       get rotation flags      (has_rotation + norot)
 *       check freq-mod presence (spans whole block, uniform raster)
 *       vendorCreateInstruction(...)
 *       t += block_duration_us
 *
 * Board waveform layout:
 *   Gradient:  [num_shots x num_samples] amplitude + optional time array
 *   RF:        [num_channels x num_samples] magnitude + optional phase
 *              + optional time array
 *   ADC:       delay + unique definition ID (links to echo filter)
 *   Trigger:   delay + duration
 *   Rotation:  presence flag + norot flag
 *   Freq-mod:  presence flag only (uniform raster, no delay, no time array)
 *
 * Compile:
 *   cc -I../../csrc example_geninstructions.c \
 *      ../../csrc/pulseqlib_*.c -lm -o geninstructions
 */

#include "example_vendorlib.h"   /* must come first */
#include "pulseqlib_methods.h"

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
/*  Vendor-side stubs (placeholders)                                  */
/* ================================================================== */

/**
 * @brief Create a gradient instruction for one axis.
 *
 * @param[in] t_us         absolute time in segment
 * @param[in] delay_us     delay from block start
 * @param[in] num_shots    leftmost dimension
 * @param[in] ns_per_shot  per-shot sample counts
 * @param[in] amps         [num_shots][ns_per_shot[s]] amplitude values
 * @param[in] time_us      optional time array (NULL for uniform raster)
 */
static void vendorCreateGradInstruction(
    int axis,
    int t_us, int delay_us,
    float** amps, int* ns_per_shot,
    float* time_us, int num_time)
{
    (void)axis; (void)amps; (void)ns_per_shot;
    (void)time_us; (void)num_time; (void)delay_us; (void)t_us;
}

/**
 * @brief Create an RF instruction.
 *
 * @param[in] t_us         absolute time in segment
 * @param[in] delay_us     delay from block start
 * @param[in] num_channels Tx channel count
 * @param[in] num_samples  samples per channel
 * @param[in] mag          [num_channels][num_samples] magnitude
 * @param[in] phase        [num_channels][num_samples] phase (NULL if real)
 * @param[in] time_us      optional time array (NULL for uniform raster)
 */
static void vendorCreateRFInstruction(
    int t_us, int delay_us,
    float** mag, float** phase, int num_channels, int num_samples,
    float* time_us)
{
    (void)mag; (void)phase; (void)num_channels; (void)num_samples;
    (void)time_us; (void)delay_us; (void)t_us;
}

/**
 * @brief Create an ADC instruction.
 *
 * @param[in] t_us         absolute time in segment
 * @param[in] delay_us     delay from block start
 * @param[in] adc_def_id   unique ADC definition index (maps to echo filter)
 */
static void vendorCreateADCInstruction(int t_us, int delay_us, int adc_def_id)
{
    (void)adc_def_id; (void)delay_us; (void)t_us;
}

/**
 * @brief Create a frequency modulation instruction.
 *
 * Freq-mod is an independent channel (separate from RF and ADC).
 * It is created when the block has a freq_mod_id AND a simultaneous
 * gradient active during the RF/ADC window.  The waveform spans the
 * entire block at uniform raster (rf_raster or adc_raster, which are
 * equal on some vendors, e.g. 2 us on GE).
 *
 * @param[in] num_samples  block_duration_us / raster_us
 */
static void vendorCreateFreqModInstruction(int num_samples)
{
    (void)num_samples;
}

/**
 * @brief Create a trigger instruction.
 *
 * @param[in] t_us        absolute time in segment
 * @param[in] delay_us    delay from block start
 * @param[in] duration_us trigger duration
 */
static void vendorCreateTriggerInstruction(
    int t_us, int delay_us, int duration_us)
{
    (void)delay_us; (void)duration_us; (void)t_us;
}

/**
 * @brief Set rotation flag for this block.
 *
 * @param[in] has_rotation 1 if block carries ANY rotation ID
 * @param[in] norot_flag   1 if block has the no-rotation override
 */
static void vendorSetRotation(int has_rotation, int norot_flag)
{
    (void)has_rotation; (void)norot_flag;
}

/**
 * @brief Notify vendor driver of intra-segment RF-to-ADC gap.
 *
 * When RF and the following ADC within the same segment are closer
 * than a certain threshold, vendor-specific timing adjustments are
 * required.
 *
 * @param[in] gap_us  RF end -> next ADC start (us), or -1 if none.
 */
static void vendorSetSegmentRFADCGap(int gap_us)
{
    (void)gap_us;
}

/**
 * @brief Notify vendor driver of intra-segment ADC-to-ADC gap.
 *
 * When consecutive ADC events within a segment are closer than a
 * threshold, vendor-specific data acquisition placement must be
 * adjusted.
 *
 * @param[in] gap_us  prev ADC end -> next ADC start (us), or -1 if < 2 ADCs.
 */
static void vendorSetSegmentADCADCGap(int gap_us)
{
    (void)gap_us;
}

/* ================================================================== */
/*  Free helpers (waveform arrays returned by pulseqlib)              */
/* ================================================================== */

static void free_2d(float** arr, int n)
{
    int i;
    if (!arr) return;
    for (i = 0; i < n; ++i)
        free(arr[i]);
    free(arr);
}

/* ================================================================== */
/*  Generate instructions for one block                               */
/* ================================================================== */

static void generate_block_instructions(
    const pulseqlib_collection* coll,
    int seg_idx, int blk_idx, int t_us)
{
    int axis;
    int has_rf, has_adc;

    /* -- Gradients (X=0, Y=1, Z=2) ------------------------------- */
    for (axis = 0; axis < 3; ++axis) {
        int   delay_us;
        int   num_shots;
        int*  ns_per_shot;
        float** amps;
        float* time_arr;

        if (!pulseqlib_block_has_grad(coll, seg_idx, blk_idx, axis))
            continue;

        num_shots   = 0;
        ns_per_shot = NULL;
        delay_us    = pulseqlib_get_grad_delay_us(
                          coll, seg_idx, blk_idx, axis);
        amps        = pulseqlib_get_grad_amplitude(
                          coll, seg_idx, blk_idx, axis,
                          &num_shots, &ns_per_shot);
        if (!amps)
            continue;

        /* Optional time array (for traps / extended traps) */
        time_arr = pulseqlib_get_grad_time_us(
                       coll, seg_idx, blk_idx, axis);

        vendorCreateGradInstruction(
            axis, t_us, delay_us,
            amps, ns_per_shot,
            time_arr, num_shots);

        free(time_arr);
        free(ns_per_shot);
        free_2d(amps, num_shots);
    }

    /* -- RF -------------------------------------------------------- */
    has_rf = pulseqlib_block_has_rf(coll, seg_idx, blk_idx);
    if (has_rf) {
        int   delay_us;
        int   num_channels;
        int   num_samples;
        float** mag;
        float** phase;
        float* time_arr;

        num_channels = 0;
        num_samples  = 0;
        delay_us = pulseqlib_get_rf_delay_us(coll, seg_idx, blk_idx);
        mag      = pulseqlib_get_rf_magnitude(
                       coll, seg_idx, blk_idx, &num_channels, &num_samples);
        if (!mag)
            goto skip_rf;

        /* Phase: NULL if RF is real-valued */
        phase = NULL;
        if (pulseqlib_block_rf_is_complex(coll, seg_idx, blk_idx)) {
            int pch = 0, pns = 0;
            phase = pulseqlib_get_rf_phase(
                        coll, seg_idx, blk_idx, &pch, &pns);
            /* phase can still be NULL on alloc failure */
        }

        /* Optional time array (same length as mag samples) */
        time_arr = NULL;
        if (!pulseqlib_block_rf_has_uniform_raster(coll, seg_idx, blk_idx)) {
            time_arr = pulseqlib_get_rf_time_us(
                           coll, seg_idx, blk_idx);
        }

        vendorCreateRFInstruction(
            t_us, delay_us,
            mag, phase, num_channels, num_samples,
            time_arr);

        free(time_arr);
        if (phase) free_2d(phase, num_channels);
        free_2d(mag, num_channels);
    }
skip_rf:

    /* -- ADC ------------------------------------------------------- */
    has_adc = pulseqlib_block_has_adc(coll, seg_idx, blk_idx);
    if (has_adc) {
        int adc_def_id = pulseqlib_get_adc_library_index(
                             coll, seg_idx, blk_idx);
        int delay_us   = pulseqlib_get_adc_delay_us(
                             coll, seg_idx, blk_idx);

        vendorCreateADCInstruction(t_us, delay_us, adc_def_id);
    }

    /* -- Trigger --------------------------------------------------- */
    if (pulseqlib_block_has_trigger(coll, seg_idx, blk_idx)) {
        int delay_us    = pulseqlib_get_trigger_delay_us(
                              coll, seg_idx, blk_idx);
        int duration_us = pulseqlib_get_trigger_duration_us(
                              coll, seg_idx, blk_idx);

        vendorCreateTriggerInstruction(t_us, delay_us, duration_us);
    }

    /* -- Rotation flags -------------------------------------------- */
    {
        int has_rot = pulseqlib_block_has_rotation(coll, seg_idx, blk_idx);
        int norot   = pulseqlib_block_has_norot(coll, seg_idx, blk_idx);
        vendorSetRotation(has_rot, norot);
    }

    /* -- Freq-mod (independent channel) ---------------------------- */
    if (pulseqlib_block_has_freq_mod(coll, seg_idx, blk_idx)
        && (has_rf || has_adc)) {
        int grad_active = 0;
        for (axis = 0; axis < 3; ++axis) {
            if (pulseqlib_block_has_grad(coll, seg_idx, blk_idx, axis)) {
                grad_active = 1;
                break;
            }
        }
        if (grad_active) {
            int dur = pulseqlib_get_block_duration_us(coll, seg_idx, blk_idx);
            int raster_us = 2;  /* vendor-specific raster (e.g. 2 us on GE) */
            int num_samples = dur / raster_us;
            vendorCreateFreqModInstruction(num_samples, t_us);
        }
    }
}

/* ================================================================== */
/*  Main                                                              */
/* ================================================================== */

int main(int argc, char** argv)
{
    const char*           seq_path;
    pulseqlib_opts        opts  = PULSEQLIB_OPTS_INIT;
    pulseqlib_diagnostic  diag  = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_collection* coll  = NULL;
    int rc, nseg, seg_idx;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <sequence.seq>\n", argv[0]);
        return 1;
    }
    seq_path = argv[1];

    vendor_opts_init(&opts, 42577478.0f, 3.0f, 50.0f, 200.0f);

    /* -- Load (with cache, no labels needed) ---------------------- */
    rc = pulseqlib_read(&coll, &diag, seq_path, &opts,
                        1,   /* cache_binary     */
                        1,   /* verify_signature */
                        0,   /* parse_labels     */
                        1);  /* num_averages     */
    CHECK(rc, &diag);

    /* -- Walk segments and generate instructions ------------------ */
    nseg = pulseqlib_get_num_segments(coll);

    for (seg_idx = 0; seg_idx < nseg; ++seg_idx) {
        int nblk = pulseqlib_get_segment_num_blocks(coll, seg_idx);
        int blk_idx;
        int t_us = 0;

        /* Segment-level gaps for vendor timing tweaks */
        {
            int rf_adc_gap  = pulseqlib_get_segment_rf_adc_gap_us(coll, seg_idx);
            int adc_adc_gap = pulseqlib_get_segment_adc_adc_gap_us(coll, seg_idx);
            vendorSetSegmentRFADCGap(rf_adc_gap);
            vendorSetSegmentADCADCGap(adc_adc_gap);
        }

        for (blk_idx = 0; blk_idx < nblk; ++blk_idx) {
            generate_block_instructions(coll, seg_idx, blk_idx, t_us);
            t_us += pulseqlib_get_block_duration_us(coll, seg_idx, blk_idx);
        }
    }

    pulseqlib_collection_free(coll);
    return 0;

fail:
    if (coll) pulseqlib_collection_free(coll);
    return 1;
}
