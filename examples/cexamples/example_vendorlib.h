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
 *     #include "example_vendorlib.h"   // <-- first
 *     #include "pulseqlib_methods.h"   // <-- then the library
 */

#ifndef EXAMPLE_VENDORLIB_H
#define EXAMPLE_VENDORLIB_H

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
#define PULSEQLIB_VENDOR 2   /* GEHC */
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

/** Gyromagnetic ratio (Hz / T). */
#define VENDOR_GAMMA_HZ_PER_T      42577478.0f

/** Static field strength (T). */
#define VENDOR_B0_T                 3.0f

/** Maximum gradient amplitude (Hz / m). */
#define VENDOR_MAX_GRAD_HZ_PER_M   (50.0e-3f * VENDOR_GAMMA_HZ_PER_T)

/** Maximum slew rate (Hz / m / s). */
#define VENDOR_MAX_SLEW_HZ_PER_M_S (200.0f * VENDOR_GAMMA_HZ_PER_T)

/** RF raster time (us). */
#define VENDOR_RF_RASTER_US         1.0f

/** Gradient raster time (us). */
#define VENDOR_GRAD_RASTER_US       4.0f

/** ADC dwell raster time (us). */
#define VENDOR_ADC_RASTER_US        0.1f

/** Block duration raster time (us). */
#define VENDOR_BLOCK_RASTER_US      10.0f

/* ================================================================== */
/*  4. PNS model parameters                                          */
/* ================================================================== */

/** Chronaxie (us) – IEC 60601-2-33:2022. */
#define VENDOR_PNS_CHRONAXIE_US     360.0f

/** Rheobase (T/m/s) – scanner-specific calibration. */
#define VENDOR_PNS_RHEOBASE_T_M_S   20.0f

/** Effective coil length (m). */
#define VENDOR_PNS_ALPHA            0.333f

/** PNS threshold (%). */
#define VENDOR_PNS_THRESHOLD_PCT    100.0f

/* ================================================================== */
/*  5. Acoustic forbidden bands                                       */
/* ================================================================== */

/**
 * Number of forbidden acoustic bands (scanner-specific).
 * Set to 0 in this example — fill in from the system config at runtime.
 */
#define VENDOR_NUM_FORBIDDEN_BANDS  0

/* ================================================================== */
/*  6. Vendor error reporting facade                                  */
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
 * The vendor error code can be derived from the pulseqlib code with
 * a simple mapping table.
 *
 * @param code  pulseqlib error code (negative PULSEQLIB_ERR_*).
 * @param diag  Diagnostic struct (may be NULL).
 */
static void vendor_report_error(int code,
                                const pulseqlib_diagnostic* diag)
{
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
/*  7. Convenience: fill an opts struct from the #defines above       */
/* ================================================================== */

#include "pulseqlib_methods.h"

/**
 * @brief Initialise a pulseqlib_opts from the vendor constants.
 *
 * Call this once at startup; pass the result to every pulseqlib_read()
 * and pulseqlib_check_safety() call.
 */
static inline void vendor_opts_init(pulseqlib_opts* opts)
{
    pulseqlib_opts_init(opts,
        VENDOR_GAMMA_HZ_PER_T,
        VENDOR_B0_T,
        VENDOR_MAX_GRAD_HZ_PER_M,
        VENDOR_MAX_SLEW_HZ_PER_M_S,
        VENDOR_RF_RASTER_US,
        VENDOR_GRAD_RASTER_US,
        VENDOR_ADC_RASTER_US,
        VENDOR_BLOCK_RASTER_US);
}

/**
 * @brief Initialise PNS parameters from the vendor constants.
 */
static inline void vendor_pns_params_init(pulseqlib_pns_params* p)
{
    p->vendor                  = PULSEQLIB_VENDOR;
    p->chronaxie_us            = VENDOR_PNS_CHRONAXIE_US;
    p->rheobase_hz_per_m_per_s = VENDOR_PNS_RHEOBASE_T_M_S
                                 * VENDOR_GAMMA_HZ_PER_T;
    p->alpha                   = VENDOR_PNS_ALPHA;
}

#endif /* EXAMPLE_VENDORLIB_H */
