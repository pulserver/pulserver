/**
 * @file vendor.h
 * @brief The scanner side of an integration, stubbed out.
 *
 * Everything a real interpreter supplies and this library does not: the
 * hardware's limits, the console it draws parameters on, the waveform memory
 * it uploads into, the sequencer it hands blocks to. Each stub here does
 * nothing but print what a real one would have done, so an example can be run
 * on a workstation and read as a description of the integration.
 *
 * A vendor integration replaces this file. Nothing below is part of the
 * public API, and no name here is prefixed `pulseg_` or `pulseq_`: these
 * are the caller's side of the boundary, not the library's.
 */

#ifndef EXAMPLE_VENDOR_H
#define EXAMPLE_VENDOR_H

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "pulseg/pulseg.h"

/* These live in a header shared by every example, and no single example uses
 * all of them. */
#ifdef __GNUC__
#define VENDOR_MAYBE_UNUSED __attribute__((unused))
#else
#define VENDOR_MAYBE_UNUSED
#endif

/* --- Reporting --- */

/** What a real integration writes to the scanner's log. */
static VENDOR_MAYBE_UNUSED void vendor_log(const char *format, ...)
{
    va_list arguments;
    va_start(arguments, format);
    (void)vprintf(format, arguments);
    va_end(arguments);
    (void)printf("\n");
}

/** What a real integration shows the operator before refusing to scan. */
static VENDOR_MAYBE_UNUSED void vendor_refuse(int code, const pulseg_diagnostic *diagnostic)
{
    char message[512];
    pulseg_format_error(message, (int)sizeof message, code, diagnostic);
    (void)fprintf(stderr, "refused: %s\n", message);
}

/* --- Setup --- */

/**
 * The hardware's own limits, wherever a platform keeps them.
 *
 * The gradient and slew ceilings are per axis and already derated by
 * sqrt(3), so that no physical axis exceeds the amplifier under an arbitrary
 * rotation. The rasters are the scanner's, not the ones a .seq declares.
 */
static VENDOR_MAYBE_UNUSED void vendor_system_limits(pulseg_opts *opts)
{
    const float gamma_hz_per_t = 42.576e6f;

    pulseg_opts_init(
        opts,
        gamma_hz_per_t,
        3.0f,                             /* B0 (T)                     */
        gamma_hz_per_t * 0.080f / 1.732f, /* 80 mT/m, derated           */
        gamma_hz_per_t * 250.0f / 1.732f, /* 250 T/m/s, derated         */
        2.0f,                             /* RF raster (us)             */
        4.0f,                             /* gradient raster (us)       */
        2.0f,                             /* ADC raster (us)            */
        4.0f);                            /* block duration raster (us) */
}

/** The acoustic bands this magnet resonates in, from the vendor's table. */
static VENDOR_MAYBE_UNUSED void vendor_forbidden_bands(pulseg_forbidden_band_list *bands)
{
    static pulseg_forbidden_band table[2];

    table[0].freq_min_hz = 550.0f;
    table[0].freq_max_hz = 620.0f;
    table[0].max_amplitude_hz_per_m = 0.0f;
    table[1].freq_min_hz = 1180.0f;
    table[1].freq_max_hz = 1260.0f;
    table[1].max_amplitude_hz_per_m = 0.0f;

    bands->count = 2;
    bands->bands = table;
}

/** One control on the console, drawn from a parameter the plugin declared. */
static VENDOR_MAYBE_UNUSED void vendor_ui_declare(const char *wire_name, int type)
{
    static const char *type_names[] = {"float", "int", "bool", "list", "text", "config"};
    const char *type_name = (type >= 0 && type <= 5) ? type_names[type] : "?";

    vendor_log("    ui: %-24s (%s)", wire_name, type_name);
}

/** What the console shows where the scan duration goes. */
static VENDOR_MAYBE_UNUSED void vendor_ui_scan_time(double seconds)
{
    vendor_log("    ui: scan time %d:%02d", (int)seconds / 60, (int)seconds % 60);
}

/* --- Waveform generation --- */

/** How many samples the gradient waveform memory holds. */
static VENDOR_MAYBE_UNUSED long vendor_waveform_memory_samples(void)
{
    return 512L * 1024L;
}

/** Upload one materialised waveform, and hand back the hardware's id for it. */
static VENDOR_MAYBE_UNUSED int vendor_load_waveform(int axis, const float *amplitudes, int num_points)
{
    static int next_id;

    (void)amplitudes;
    vendor_log("    upload: axis %d, %d points -> hardware id %d", axis, num_points, next_id);
    return next_id++;
}

/* --- Playout --- */

/** Point the sequencer at a prepared segment. */
static VENDOR_MAYBE_UNUSED void vendor_begin_segment(int segment_id)
{
    vendor_log("    segment %d", segment_id);
}

/**
 * Play one block: set the amplitudes and the rotation, fire the RF, open the
 * receiver if the block acquires.
 */
static VENDOR_MAYBE_UNUSED void vendor_play_block(const pulseg_block_instance *block)
{
    vendor_log(
        "      %6d us  rf %8.1f Hz  g (%9.1f %9.1f %9.1f) Hz/m%s",
        block->duration_us,
        (double)block->rf_amp_hz,
        (double)block->gx_amp_hz_per_m,
        (double)block->gy_amp_hz_per_m,
        (double)block->gz_amp_hz_per_m,
        block->adc_flag ? "  [acquire]" : "");
}

#endif /* EXAMPLE_VENDOR_H */
