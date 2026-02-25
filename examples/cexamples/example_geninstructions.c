/**
 * @file example_geninstructions.c
 * @brief Generate hardware instructions from a cached collection.
 *
 * Workflow (per segment, per block):
 *   FOREACH segment:
 *     walk RF/ADC events for per-event gap computation
 *     t = 0
 *     FOREACH block in segment:
 *       get gradient waveforms  (X, Y, Z)
 *       get RF waveforms        (magnitude + optional phase + optional time)
 *       get ADC definition ID   (maps to echo filter from check phase)
 *       get digitalout info      (delay + duration)
 *       get trigger info        (segment-level physio trigger)
 *       get rotation flags      (has_rotation + norot)
 *       check freq-mod presence (spans whole block, uniform raster)
 *       vendor_create_instruction(...)
 *       t += block_duration_us
 *
 * Board waveform layout:
 *   Gradient:  [num_shots x num_samples] amplitude + optional time array
 *   RF:        [num_channels x num_samples] magnitude + optional phase
 *              + optional time array
 *   ADC:       delay + unique definition ID (links to echo filter)
 *   Digitalout: delay + duration (block-level digital output)
 *   Trigger:   segment-level physio trigger (delay + duration)
 *   Rotation:  presence flag + norot flag
 *   Freq-mod:  presence flag only (uniform raster, no delay, no time array)
 *
 * Compile:
 *   cc -I../../csrc example_geninstructions.c \
 *      ../../csrc/pulseqlib_*.c -lm -o geninstructions
 */

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
/*  Vendor-side stubs (placeholders)                                  */
/* ================================================================== */

/**
 * @brief Create a gradient instruction for one axis.
 *
 * @param[in] t_us         absolute time in segment
 * @param[in] delay_us     delay from block start
 * @param[in] num_shots    number of interleaved shots
 * @param[in] num_samples  samples per shot (same for all shots)
 * @param[in] amps         [num_shots][num_samples] amplitude values
 * @param[in] time_us      optional time array (NULL for uniform raster)
 */
static void vendor_create_grad_instruction(
    int axis,
    int t_us, int delay_us,
    int num_shots, int num_samples,
    float** amps, float* time_us)
{
    (void)axis; (void)delay_us; (void)t_us; (void)num_shots; (void)num_samples; (void)amps; (void)time_us; 
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
static void vendor_create_rf_instruction(
    int t_us, int delay_us,
    int num_channels, int num_samples,
    float** mag, float** phase, float* time_us)
{
    (void)delay_us; (void)t_us; (void)num_channels; (void)num_samples; (void)mag; (void)phase; (void)time_us; 
    
}

/**
 * @brief Create an ADC instruction.
 *
 * @param[in] t_us         absolute time in segment
 * @param[in] delay_us     delay from block start
 * @param[in] adc_def_id   unique ADC definition index (maps to echo filter)
 */
static void vendor_create_adc_instruction(int t_us, int delay_us, int adc_def_id)
{
    (void)t_us; (void)delay_us; (void)adc_def_id;
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
static void vendor_create_freq_mod_instruction(int num_samples)
{
    (void)num_samples;
}

/**
 * @brief Create a digital output instruction.
 *
 * @param[in] t_us        absolute time in segment
 * @param[in] delay_us    delay from block start
 * @param[in] duration_us digital output duration
 */
static void vendor_create_digitalout_instruction(int t_us, int delay_us, int duration_us)
{
    (void)t_us; (void)delay_us; (void)duration_us; 
}

/**
 * @brief Set rotation flag for this block.
 *
 * @param[in] has_rotation 1 if block carries ANY rotation ID
 * @param[in] norot_flag   1 if block has the no-rotation override
 */
static void vendor_set_rotation(int has_rotation, int norot_flag)
{
    (void)has_rotation; (void)norot_flag;
}

/**
 * @brief Adjust RF parameters based on the RF-to-next-ADC gap.
 *
 * Called for each RF event during the segment event walk.
 * When the RF->ADC gap is small, vendor-specific timing adjustments
 * may be needed.
 *
 * @param[in] blk_idx  block index within the segment
 * @param[in] gap_us   RF end -> next ADC start (us), or -1 if no following
 *                      ADC before the next RF or end of segment.
 */
static void vendor_adjust_rf_for_gap(int blk_idx, int gap_us)
{
    (void)blk_idx; (void)gap_us;
}

/**
 * @brief Adjust ADC parameters based on the gap from the preceding event.
 *
 * Called for each ADC event during the segment event walk.
 * When the gap from the preceding RF or ADC is small, vendor-specific
 * data acquisition placement may need adjustment.
 *
 * @param[in] blk_idx  block index within the segment
 * @param[in] gap_us   preceding event end -> ADC start (us),
 *                      or -1 if this is the first event in the segment.
 * @param[in] from_rf  1 if the preceding event was RF, 0 if it was ADC.
 */
static void vendor_adjust_adc_for_gap(int blk_idx, int gap_us, int from_rf)
{
    (void)blk_idx; (void)gap_us; (void)from_rf;
}

/* ================================================================== */
/*  Segment event walk — per-event gap computation                    */
/* ================================================================== */

/*
 * Chronological walk through RF and ADC events in a segment.
 *
 * Starting from the beginning of the segment:
 *
 *  1. Find the first RF or ADC event.
 *     a) RF first: look ahead for the next ADC before the next RF or
 *        end-of-segment.  If found, compute the RF->ADC gap and call
 *        vendor_adjust_rf_for_gap() to decide RF params.
 *        If no ADC before the next RF or end, no adjustment needed.
 *     b) ADC first: no preceding RF to worry about.
 *
 *  2. Move to the next RF or ADC event.
 *     a) RF: same as 1a.
 *     b) ADC:
 *        - If the preceding event was RF, compute the preceding-RF->ADC
 *          gap and call vendor_adjust_adc_for_gap(..., from_rf=1) to decide
 *          ADC params.
 *        - If the preceding event was ADC, compute the ADC->ADC gap and
 *          call vendor_adjust_adc_for_gap(..., from_rf=0).
 *
 *  3. Repeat step 2 until end of segment.
 */

#define EVT_RF  0
#define EVT_ADC 1

typedef struct {
    int kind;          /* EVT_RF or EVT_ADC */
    int blk_idx;       /* block index within segment */
    int start_us;      /* absolute start time in segment (us) */
    int end_us;        /* absolute end time in segment (us) */
} seg_event;

#define MAX_SEG_EVENTS 256

static void walk_segment_events(
    const pulseqlib_collection* coll,
    int seg_idx)
{
    int nblk = pulseqlib_get_segment_num_blocks(coll, seg_idx);
    seg_event events[MAX_SEG_EVENTS];
    int num_events = 0;
    int t_us = 0;
    int i, j;

    /* --- Pass 1: collect RF and ADC events with absolute timing --- */
    for (i = 0; i < nblk; ++i) {
        int dur = pulseqlib_get_block_duration_us(coll, seg_idx, i);

        if (pulseqlib_block_has_rf(coll, seg_idx, i)
            && num_events < MAX_SEG_EVENTS) {
            int delay  = pulseqlib_get_rf_delay_us(coll, seg_idx, i);
            int ns     = pulseqlib_get_rf_num_samples(coll, seg_idx, i);
            int rf_dur = (int)(ns * VENDOR_RF_RASTER_US);

            events[num_events].kind     = EVT_RF;
            events[num_events].blk_idx  = i;
            events[num_events].start_us = t_us + delay;
            events[num_events].end_us   = t_us + delay + rf_dur;
            num_events++;
        }

        if (pulseqlib_block_has_adc(coll, seg_idx, i)
            && num_events < MAX_SEG_EVENTS) {
            int adc_id  = pulseqlib_get_adc_library_index(coll, seg_idx, i);
            int delay   = pulseqlib_get_adc_delay_us(coll, seg_idx, i);
            int ns      = pulseqlib_get_adc_num_samples(coll, adc_id);
            int dwell   = pulseqlib_get_adc_dwell_us(coll, adc_id);
            int adc_dur = (int)(ns * dwell * 1e-3f);  /* dwell is ns */

            events[num_events].kind     = EVT_ADC;
            events[num_events].blk_idx  = i;
            events[num_events].start_us = t_us + delay;
            events[num_events].end_us   = t_us + delay + adc_dur;
            num_events++;
        }

        t_us += dur;
    }

    /* --- Sort by start time (insertion sort, stable) -------------- */
    for (i = 1; i < num_events; ++i) {
        seg_event tmp = events[i];
        j = i - 1;
        while (j >= 0 && events[j].start_us > tmp.start_us) {
            events[j + 1] = events[j];
            --j;
        }
        events[j + 1] = tmp;
    }

    /* --- Pass 2: walk events, compute per-event gaps -------------- */
    for (i = 0; i < num_events; ++i) {
        if (events[i].kind == EVT_RF) {
            /* Look ahead for the next ADC before the next RF */
            int rf_adc_gap = -1;
            for (j = i + 1; j < num_events; ++j) {
                if (events[j].kind == EVT_ADC) {
                    rf_adc_gap = events[j].start_us - events[i].end_us;
                    break;
                }
                if (events[j].kind == EVT_RF)
                    break;  /* next RF before any ADC — no RF->ADC pair */
            }
            vendor_adjust_rf_for_gap(events[i].blk_idx, rf_adc_gap);
        }
        else { /* EVT_ADC */
            if (i == 0) {
                /* First event is ADC — no preceding RF or ADC */
                vendor_adjust_adc_for_gap(events[i].blk_idx, -1, 0);
            }
            else {
                int gap     = events[i].start_us - events[i - 1].end_us;
                int from_rf = (events[i - 1].kind == EVT_RF) ? 1 : 0;
                vendor_adjust_adc_for_gap(events[i].blk_idx, gap, from_rf);
            }
        }
    }
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
        int   num_samples;
        float** amps;
        float* time_arr;

        if (!pulseqlib_block_has_grad(coll, seg_idx, blk_idx, axis))
            continue;

        num_shots = 0;
        num_samples = 0;
        delay_us = pulseqlib_get_grad_delay_us(coll, seg_idx, blk_idx, axis);
        amps = pulseqlib_get_grad_amplitude(coll, seg_idx, blk_idx, axis, &num_shots, &num_samples);
        if (!amps)
            continue;

        /* Optional time array (for traps / extended traps) */
        time_arr = pulseqlib_get_grad_time_us(coll, seg_idx, blk_idx, axis);

        vendor_create_grad_instruction(
            axis, t_us, delay_us,
            num_shots, num_samples,
            amps, time_arr);

        free(time_arr);
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
        num_samples = 0;
        delay_us = pulseqlib_get_rf_delay_us(coll, seg_idx, blk_idx);
        mag = pulseqlib_get_rf_magnitude(coll, seg_idx, blk_idx, &num_channels, &num_samples);
        if (!mag)
            goto skip_rf;

        /* Phase: NULL if RF is real-valued */
        phase = NULL;
        if (pulseqlib_block_rf_is_complex(coll, seg_idx, blk_idx)) {
            int pch = 0, pns = 0;
            phase = pulseqlib_get_rf_phase(coll, seg_idx, blk_idx, &pch, &pns);
            /* phase can still be NULL on alloc failure */
        }

        /* Optional time array (same length as mag samples) */
        time_arr = NULL;
        if (!pulseqlib_block_rf_has_uniform_raster(coll, seg_idx, blk_idx)) {
            time_arr = pulseqlib_get_rf_time_us(coll, seg_idx, blk_idx);
        }

        vendor_create_rf_instruction(
            t_us, delay_us,
            num_channels, num_samples,
            mag, phase, time_arr);

        if (time_arr)free(time_arr);
        if (phase) free_2d(phase, num_channels);
        free_2d(mag, num_channels);
    }
skip_rf:

    /* -- ADC ------------------------------------------------------- */
    has_adc = pulseqlib_block_has_adc(coll, seg_idx, blk_idx);
    if (has_adc) {
        int adc_def_id = pulseqlib_get_adc_library_index(coll, seg_idx, blk_idx);
        int delay_us = pulseqlib_get_adc_delay_us(coll, seg_idx, blk_idx);

        vendor_create_adc_instruction(t_us, delay_us, adc_def_id);
    }

    /* -- Digitalout ------------------------------------------------ */
    if (pulseqlib_block_has_digitalout(coll, seg_idx, blk_idx)) {
        int delay_us = pulseqlib_get_digitalout_delay_us(coll, seg_idx, blk_idx);
        int duration_us = pulseqlib_get_digitalout_duration_us(coll, seg_idx, blk_idx);

        vendor_create_digitalout_instruction(t_us, delay_us, duration_us);
    }

    /* -- Rotation flags -------------------------------------------- */
    {
        int has_rot = pulseqlib_block_has_rotation(coll, seg_idx, blk_idx);
        int norot   = pulseqlib_block_has_norot(coll, seg_idx, blk_idx);
        vendor_set_rotation(has_rot, norot);
    }

    /* -- Freq-mod (independent channel) ---------------------------- */
    if (pulseqlib_block_has_freq_mod(coll, seg_idx, blk_idx) && (has_rf || has_adc)) {
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
            vendor_create_freq_mod_instruction(num_samples);
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

        /* Per-event gap walk for vendor timing tweaks */
        walk_segment_events(coll, seg_idx);

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
