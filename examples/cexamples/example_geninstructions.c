/**
 * @file example_geninstructions.c
 * @brief Generate hardware instructions from a cached collection.
 *
 * Workflow overview:
 *
 *   FOREACH segment:
 *     walk RF/ADC events for per-event gap computation
 *     t = 0
 *     FOREACH block in segment:
 *       -- Create waveforms (normalised, peak ~ 1.0) --
 *       get gradient waveforms  (X, Y, Z)  → vendor_create_grad_waveform(t_us, delay)
 *       get RF waveforms        (mag+phase) → vendor_create_rf_waveform(t_us, delay)
 *       -- Set initial amplitude state (from max-energy instance) --
 *       get grad initial + max amplitude  → vendor_set_grad_amplitude
 *       get grad initial shot id          → vendor_set_grad_state
 *       get RF initial + max amplitude    → vendor_set_rf_amplitude
 *       -- Ancillary channels --
 *       get ADC definition ID   (maps to echo filter from check phase)
 *       get digitalout info     (delay + duration)
 *       get trigger info        (segment-level physio trigger)
 *       get rotation flags      (has_rotation + norot)
 *       check freq-mod (overlap + nopos via pulseqlib_block_needs_freq_mod)
 *       t += block_duration_us
 *
 * Amplitude convention:
 *   Waveforms returned by the library are normalised (peak ≈ 1.0).
 *   The vendor creates hardware waveforms once at full DAC scale
 *   (e.g. 32 767).  The amplitude state is set separately:
 *
 *       hw_amplitude = DAC_MAX × initial_amp / max_amp
 *
 * Multi-shot convention:
 *   For multi-shot trajectories, pulseqlib_get_grad_amplitude() returns a
 *   2D waveform [num_shots][num_samples]. The vendor stores one instruction
 *   with a state dimension (state = shot index), then initialises the active
 *   state with pulseqlib_get_grad_initial_shot_id() from the max-energy
 *   segment instance.
 *
 *   where initial_amp is the signed physical amplitude from the
 *   max-energy segment instance and max_amp is the unsigned peak
 *   from the gradient/RF definition.  During the scan loop
 *   (see example_scanloop.c) the amplitude is updated per-TR
 *   via block_instance.gx_amp_hz_per_m / rf_amp_hz.
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
 * @brief Create a gradient waveform on the hardware sequencer.
 *
 * The waveform is normalised (peak ≈ 1.0).  The vendor scales it to
 * full DAC range (e.g. 32 767) internally.  Amplitude state is set
 * separately by vendor_set_grad_amplitude().
 *
 * @param[in] axis         0=X, 1=Y, 2=Z
 * @param[in] t_us         absolute time of block start in segment (us)
 * @param[in] delay_us     delay from block start
 * @param[in] num_shots    number of interleaved shots
 * @param[in] num_samples  samples per shot
 * @param[in] wave         [num_shots][num_samples] normalised waveform
 * @param[in] time_us      optional time array (NULL for uniform raster)
 */
static void vendor_create_grad_waveform(
    int axis, int t_us, int delay_us,
    int num_shots, int num_samples,
    float** wave, float* time_us)
{
    (void)axis; (void)t_us; (void)delay_us; (void)num_shots; (void)num_samples;
    (void)wave; (void)time_us;
}

/**
 * @brief Set the gradient amplitude state for one axis.
 *
 * The vendor computes the hardware amplitude register as:
 *     hw_amplitude = DAC_MAX × initial_amp / max_amp
 *
 * @param[in] axis          0=X, 1=Y, 2=Z
 * @param[in] initial_amp   signed amplitude (Hz/m) from max-energy instance
 * @param[in] max_amp       unsigned peak amplitude (Hz/m) from definition
 */
static void vendor_set_grad_amplitude(
    int axis, float initial_amp, float max_amp)
{
    (void)axis; (void)initial_amp; (void)max_amp;
}

/**
 * @brief Set the active gradient waveform state (shot index).
 *
 * For multi-shot gradients, the waveform instruction is stateful and this
 * selects which shot waveform is active at segment initialisation.
 *
 * @param[in] axis      0=X, 1=Y, 2=Z
 * @param[in] shot_id   shot/state index from max-energy segment instance
 */
static void vendor_set_grad_state(int axis, int shot_id)
{
    (void)axis; (void)shot_id;
}

/**
 * @brief Create an RF waveform on the hardware sequencer.
 *
 * The magnitude waveform is normalised (peak ≈ 1.0).
 *
 * @param[in] t_us          absolute time of block start in segment (us)
 * @param[in] delay_us      delay from block start
 * @param[in] num_channels  Tx channel count
 * @param[in] num_samples   samples per channel
 * @param[in] mag           [num_channels][num_samples] normalised magnitude
 * @param[in] phase         [num_channels][num_samples] phase (NULL if real)
 * @param[in] time_us       optional time array (NULL for uniform raster)
 */
static void vendor_create_rf_waveform(
    int t_us, int delay_us,
    int num_channels, int num_samples,
    float** mag, float** phase, float* time_us)
{
    (void)t_us; (void)delay_us; (void)num_channels; (void)num_samples;
    (void)mag; (void)phase; (void)time_us;
}

/**
 * @brief Set the RF amplitude state.
 *
 * The vendor computes the hardware amplitude register as:
 *     hw_amplitude = DAC_MAX × initial_amp / max_amp
 *
 * @param[in] initial_amp   signed amplitude (Hz) from max-energy instance
 * @param[in] max_amp       unsigned peak amplitude (Hz) from definition
 */
static void vendor_set_rf_amplitude(float initial_amp, float max_amp)
{
    (void)initial_amp; (void)max_amp;
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
    pulseqlib_segment_info segi = PULSEQLIB_SEGMENT_INFO_INIT;
    seg_event events[MAX_SEG_EVENTS];
    int num_events = 0;
    int t_us = 0;
    int i, j;

    pulseqlib_get_segment_info(coll, seg_idx, &segi);

    /* --- Pass 1: collect RF and ADC events with absolute timing --- */
    for (i = 0; i < segi.num_blocks; ++i) {
        pulseqlib_block_info bi = PULSEQLIB_BLOCK_INFO_INIT;
        pulseqlib_get_block_info(coll, seg_idx, i, &bi);

        if (bi.has_rf && num_events < MAX_SEG_EVENTS) {
            int rf_dur = (int)(bi.rf_num_samples * VENDOR_RF_RASTER_US);

            events[num_events].kind     = EVT_RF;
            events[num_events].blk_idx  = i;
            events[num_events].start_us = t_us + bi.rf_delay_us;
            events[num_events].end_us   = t_us + bi.rf_delay_us + rf_dur;
            num_events++;
        }

        if (bi.has_adc && num_events < MAX_SEG_EVENTS) {
            pulseqlib_adc_def ad = PULSEQLIB_ADC_DEF_INIT;
            int adc_dur;

            pulseqlib_get_adc_def(coll, bi.adc_def_id, &ad);
            adc_dur = (int)(ad.num_samples * ad.dwell_ns * 1e-3f);

            events[num_events].kind     = EVT_ADC;
            events[num_events].blk_idx  = i;
            events[num_events].start_us = t_us + bi.adc_delay_us;
            events[num_events].end_us   = t_us + bi.adc_delay_us + adc_dur;
            num_events++;
        }

        t_us += bi.duration_us;
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
    pulseqlib_block_info bi = PULSEQLIB_BLOCK_INFO_INIT;

    pulseqlib_get_block_info(coll, seg_idx, blk_idx, &bi);

    /* -- Gradients (X=0, Y=1, Z=2) ------------------------------- */
    for (axis = 0; axis < 3; ++axis) {
        int     num_shots;
        int     num_samples;
        int     shot_id;
        float** wave;
        float*  time_arr;
        float   initial_amp, max_amp;

        if (!bi.has_grad[axis])
            continue;

        /* 1) Create normalised waveform (peak ≈ 1.0). */
        num_shots = 0;
        num_samples = 0;
        wave = pulseqlib_get_grad_amplitude(coll, seg_idx, blk_idx, axis,
                                            &num_shots, &num_samples);
        if (!wave)
            continue;

        time_arr = pulseqlib_get_grad_time_us(coll, seg_idx, blk_idx, axis);

        vendor_create_grad_waveform(
            axis, t_us, bi.grad_delay_us[axis],
            num_shots, num_samples,
            wave, time_arr);

        free(time_arr);
        free_2d(wave, num_shots);

        /* 2) Set initial amplitude state.
         *    initial_amp = signed Hz/m from max-energy instance.
         *    max_amp     = unsigned peak Hz/m from definition.
         *    Vendor computes: hw_amplitude = DAC_MAX × initial / max */
        initial_amp = pulseqlib_get_grad_initial_amplitude_hz_per_m(
            coll, seg_idx, blk_idx, axis);
        max_amp = pulseqlib_get_grad_max_amplitude_hz_per_m(
            coll, seg_idx, blk_idx, axis);
        shot_id = pulseqlib_get_grad_initial_shot_id(
            coll, seg_idx, blk_idx, axis);

        vendor_set_grad_amplitude(axis, initial_amp, max_amp);
        vendor_set_grad_state(axis, shot_id);
    }

    /* -- RF -------------------------------------------------------- */
    if (bi.has_rf) {
        int     num_channels;
        int     num_samples;
        float** mag;
        float** phase;
        float*  time_arr;
        float   initial_amp, max_amp;

        /* 1) Create normalised RF waveform (peak ≈ 1.0). */
        num_channels = 0;
        num_samples = 0;
        mag = pulseqlib_get_rf_magnitude(coll, seg_idx, blk_idx,
                                         &num_channels, &num_samples);
        if (!mag)
            goto skip_rf;

        phase = NULL;
        if (bi.rf_is_complex) {
            int pch = 0, pns = 0;
            phase = pulseqlib_get_rf_phase(coll, seg_idx, blk_idx, &pch, &pns);
        }

        time_arr = NULL;
        if (!bi.rf_uniform_raster)
            time_arr = pulseqlib_get_rf_time_us(coll, seg_idx, blk_idx);

        vendor_create_rf_waveform(
            t_us, bi.rf_delay_us,
            num_channels, num_samples,
            mag, phase, time_arr);

        if (time_arr) free(time_arr);
        if (phase) free_2d(phase, num_channels);
        free_2d(mag, num_channels);

        /* 2) Set initial RF amplitude state. */
        initial_amp = pulseqlib_get_rf_initial_amplitude_hz(
            coll, seg_idx, blk_idx);
        max_amp = pulseqlib_get_rf_max_amplitude_hz(
            coll, seg_idx, blk_idx);

        vendor_set_rf_amplitude(initial_amp, max_amp);
    }
skip_rf:

    /* -- ADC ------------------------------------------------------- */
    if (bi.has_adc) {
        vendor_create_adc_instruction(t_us, bi.adc_delay_us, bi.adc_def_id);
    }

    /* -- Digitalout ------------------------------------------------ */
    if (bi.has_digitalout) {
        vendor_create_digitalout_instruction(
            t_us, bi.digitalout_delay_us, bi.digitalout_duration_us);
    }

    /* -- Rotation flags -------------------------------------------- */
    vendor_set_rotation(bi.has_rotation, bi.norot_flag);

    /* -- Freq-mod (independent channel) ---------------------------- */
    {
        int freq_mod_samples = 0;
        if (pulseqlib_block_needs_freq_mod(coll, seg_idx, blk_idx,
                                           &freq_mod_samples)) {
            vendor_create_freq_mod_instruction(freq_mod_samples);
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
    pulseqlib_collection_info ci = PULSEQLIB_COLLECTION_INFO_INIT;
    int rc, seg_idx;

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

    rc = pulseqlib_get_collection_info(coll, &ci);
    CHECK(rc, &diag);

    /* -- Walk segments and generate instructions ------------------ */
    for (seg_idx = 0; seg_idx < ci.num_segments; ++seg_idx) {
        pulseqlib_segment_info segi = PULSEQLIB_SEGMENT_INFO_INIT;
        int blk_idx;
        int t_us = 0;

        rc = pulseqlib_get_segment_info(coll, seg_idx, &segi);
        CHECK(rc, &diag);

        /* Per-event gap walk for vendor timing tweaks */
        walk_segment_events(coll, seg_idx);

        for (blk_idx = 0; blk_idx < segi.num_blocks; ++blk_idx) {
            pulseqlib_block_info bi = PULSEQLIB_BLOCK_INFO_INIT;
            generate_block_instructions(coll, seg_idx, blk_idx, t_us);
            pulseqlib_get_block_info(coll, seg_idx, blk_idx, &bi);
            t_us += bi.duration_us;
        }
    }

    pulseqlib_collection_free(coll);
    return 0;

fail:
    if (coll) pulseqlib_collection_free(coll);
    return 1;
}
