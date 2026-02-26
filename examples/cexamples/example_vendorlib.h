/**
 * @file example_vendorlib.h
 * @brief Example vendor-specific configuration header.
 *
 * In a real vendor integration this file is included *before* any
 * pulseqlib header.  It:
 *
 *   1. Overrides PULSEQLIB_ALLOC / PULSEQLIB_FREE with the vendor
 *      toolchain's allocators (so pulseqlib memory is tracked by the
 *      vendor heap).
 *
 *   2. Sets PULSEQLIB_VENDOR to the correct target.
 *
 *   3. Provides a thin vendor_error() facade that maps
 *      pulseqlib_diagnostic to the vendor's error-reporting API.
 *
 * Usage (in every .c file that calls pulseqlib):
 *
 *     #include "example_vendorlib.h"   // includes pulseqlib headers
 */

#ifndef EXAMPLE_VENDORLIB_H
#define EXAMPLE_VENDORLIB_H

/* Suppress unused-function warnings for static helpers that may
 * not be called in every translation unit. */
#ifdef __GNUC__
#define VENDOR_MAYBE_UNUSED __attribute__((unused))
#else
#define VENDOR_MAYBE_UNUSED
#endif

/* ================================================================== */
/*  1. Vendor selector                                                */
/* ================================================================== */

/*
 * Compile-time vendor override.
 * Must be defined *before* pulseqlib_config.h is included.
 * Values: PULSEQLIB_VENDOR_SIEMENS  (1)
 *         PULSEQLIB_VENDOR_GEHC     (2)
 *         PULSEQLIB_VENDOR_PHILIPS  (3)
 *         ...
 *
 * Can also be set via the build system:
 *   -DPULSEQLIB_VENDOR=2
 */
#ifndef PULSEQLIB_VENDOR
#define PULSEQLIB_VENDOR 1   /* SIEMENS */
#endif

/* ================================================================== */
/*  2. Allocator overrides                                            */
/* ================================================================== */

/*
 * Replace with vendor-specific heap functions.  Every pointer that
 * pulseqlib allocates internally will go through these macros, so
 * vendor heap accounting and leak detection work transparently.
 *
 * Example for a fictional vendor API:
 *
 *   #include "vendor_heap.h"
 *   #define PULSEQLIB_ALLOC(sz)  vendor_heap_alloc(sz)
 *   #define PULSEQLIB_FREE(ptr)  vendor_heap_free(ptr)
 *
 * For this example we just use the C standard library (the default).
 */
#include <stdlib.h>

#define PULSEQLIB_ALLOC(sz)  malloc(sz)
#define PULSEQLIB_FREE(ptr)  free(ptr)

/* ================================================================== */
/*  3. Scanner hardware constants                                     */
/* ================================================================== */

/*
 * Define system-specific constants here so that all examples / the
 * real driver share one source of truth.
 *
 * In a real vendor integration these would come from a system
 * configuration database loaded at startup.  Using compile-time
 * constants here for illustration.
 */


/** RF raster time (us). */
#define VENDOR_RF_RASTER_US         1.0f

/** Gradient raster time (us). */
#define VENDOR_GRAD_RASTER_US       10.0f

/** ADC dwell raster time (us). */
#define VENDOR_ADC_RASTER_US        0.1f

/** Block duration raster time (us). */
#define VENDOR_BLOCK_RASTER_US      10.0f

/* ================================================================== */
/*  Pulseqlib headers (after allocator + vendor overrides)            */
/* ================================================================== */

#include "pulseqlib_config.h"
#include "pulseqlib_types.h"
#include "pulseqlib_methods.h"

/* ================================================================== */
/*  4. Vendor error reporting facade                                  */
/* ================================================================== */

#include <stdio.h>

/**
 * @brief Report a pulseqlib error through the vendor error channel.
 *
 * In a real integration, replace the body with the vendor-specific
 * error API, e.g.:
 *
 *     vendor_report_error(
 *         USE_ERMES,                        // display flag
 *         formatted_msg,                    // user-facing message
 *         VENDOR_ERR_PSD_PULSEQ_FAILURE,    // vendor error code
 *         0);                               // extra args
 *
 * The pulseqlib code is an opaque negative int — the vendor maps
 * every failure to a single vendor error code.
 *
 * @param code  pulseqlib return code (negative on failure).
 * @param diag  Diagnostic struct (may be NULL).
 */
static VENDOR_MAYBE_UNUSED void vendor_report_error(
    int code,
    const pulseqlib_diagnostic* diag
) {
    char buf[512];
    pulseqlib_format_error(buf, sizeof(buf), code, diag);

    /*
     * --- Replace with vendor error API ---
     *
     * vendor_report_error(USE_ERMES, buf,
     *                     VENDOR_ERR_PSD_PULSEQ_FAILURE, 0);
     */
    fprintf(stderr, "[pulseqlib] %s\n", buf);
}

/* ================================================================== */
/*  5. Convenience: fill an opts struct from the #defines above       */
/* ================================================================== */

/**
 * @brief Initialise a pulseqlib_opts from the vendor constants.
 *
 * Call this once at startup; pass the result to every pulseqlib_read()
 * and pulseqlib_check_safety() call.
 */
static VENDOR_MAYBE_UNUSED void vendor_opts_init(
    pulseqlib_opts* opts, 
    float gamma_hz_per_t, 
    float b0_t, 
    float max_grad_millitesla_per_m, 
    float max_slew_tesla_per_m_per_s
) {
    pulseqlib_opts_init(opts,
        gamma_hz_per_t,
        b0_t,
        gamma_hz_per_t * 1e-3f * max_grad_millitesla_per_m,
        gamma_hz_per_t * max_slew_tesla_per_m_per_s,
        VENDOR_RF_RASTER_US,
        VENDOR_GRAD_RASTER_US,
        VENDOR_ADC_RASTER_US,
        VENDOR_BLOCK_RASTER_US);
}

/**
 * @brief Initialise PNS parameters from the vendor constants.
 */
static VENDOR_MAYBE_UNUSED void vendor_pns_params_init(
    pulseqlib_pns_params* p,
    pulseqlib_opts* opts,
    float chronaxie_us, 
    float rheobase_t_m_s, 
    float alpha
) {
    p->vendor                  = PULSEQLIB_VENDOR;
    p->chronaxie_us            = chronaxie_us;
    p->rheobase_hz_per_m_per_s = rheobase_t_m_s * opts->gamma_hz_per_t;
    p->alpha                   = alpha;
}

#endif /* EXAMPLE_VENDORLIB_H */
