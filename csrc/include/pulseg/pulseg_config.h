/* pulseg_config.h -- platform configuration
 *
 * This header MUST be included (directly or transitively) before any
 * other pulseg header or source file.  It provides:
 *
 *   PULSEG_VENDOR_*  -- vendor ID constants (used at runtime)
 *   PULSEG_VENDOR    -- compile-time default vendor
 *   PULSEG_ALLOC     -- heap allocator  (default: malloc)
 *   PULSEG_FREE      -- heap deallocator (default: free)
 */

#ifndef PULSEG_CONFIG_H
#define PULSEG_CONFIG_H

/* Suppress -Wfloat-equal for intentional exact float comparisons
 * (shape decompression RLE, zero-detection, etc.).
 * Required when building with GE EPIC's -Werror -Wfloat-equal. */
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif

#include <stdlib.h>

/* ------------------------------------------------------------------ */
/*  Vendor identifiers (runtime constants)                            */
/* ------------------------------------------------------------------ */
#define PULSEG_VENDOR_UNSPECIFIED   0
#define PULSEG_VENDOR_SIEMENS        1
#define PULSEG_VENDOR_GEHC           2
#define PULSEG_VENDOR_PHILIPS        3
#define PULSEG_VENDOR_UNITED_IMAGING 4
#define PULSEG_VENDOR_BRUKER         5

/* Compile-time default (overrideable via -DPULSEG_VENDOR=N). Public
 * builds are vendor-neutral; vendor layers (e.g. the private
 * pulserver-interpreter's src_gelib/pulserver_ge_config.h) define
 * PULSEG_VENDOR=PULSEG_VENDOR_GEHC before including any pulseg header. */
#ifndef PULSEG_VENDOR
#define PULSEG_VENDOR PULSEG_VENDOR_UNSPECIFIED
#endif

/* ------------------------------------------------------------------ */
/*  Acoustic peak-detection defaults                                  */
/* ------------------------------------------------------------------ */

#ifndef PULSEG_PEAK_LOG10_THRESHOLD_DEFAULT
#define PULSEG_PEAK_LOG10_THRESHOLD_DEFAULT 2.25f
#endif

#ifndef PULSEG_PEAK_NORM_SCALE_DEFAULT
#define PULSEG_PEAK_NORM_SCALE_DEFAULT 10.0f
#endif

#ifndef PULSEG_PEAK_EPS_DEFAULT
#define PULSEG_PEAK_EPS_DEFAULT 1e-30f
#endif

#ifndef PULSEG_PEAK_PROMINENCE_DEFAULT
#define PULSEG_PEAK_PROMINENCE_DEFAULT 0.0f
#endif

/* ------------------------------------------------------------------ */
/*  Structural mechanical resonance analysis defaults                  */
/* ------------------------------------------------------------------ */

/** Minimum waveform samples before sub-period detection is applied. */
#ifndef PULSEG_MIN_ARBITRARY_SAMPLES
#define PULSEG_MIN_ARBITRARY_SAMPLES 10
#endif

/* ------------------------------------------------------------------ */
/*  Allocator overrides                                               */
/* ------------------------------------------------------------------ */

/*
 * Override PULSEG_ALLOC / PULSEG_FREE *before* including this
 * header to use vendor-specific allocators, e.g.:
 *
 *   #define PULSEG_ALLOC(sz)  MyVendorAlloc(sz)
 *   #define PULSEG_FREE(ptr)  MyVendorFree(ptr)
 *   #include "pulseg_config.h"
 */
#ifndef PULSEG_ALLOC
#define PULSEG_ALLOC(sz) malloc(sz)
#endif

#ifndef PULSEG_FREE
#define PULSEG_FREE(ptr) free(ptr)
#endif

/* ------------------------------------------------------------------ */
/*  Hardware frequency-conversion overrides (D9)                      */
/* ------------------------------------------------------------------ */

/*
 * Override PULSEG_HW_FREQ_CONVERSION / PULSEG_HW_WAVEFORM_END before
 * including this header for vendor-specific DAC parameters, e.g.:
 *
 *   #define PULSEG_HW_FREQ_CONVERSION  VendorFreqRes
 *   #define PULSEG_HW_WAVEFORM_END     VendorWeosBit
 *   #include "pulseg_config.h"
 */
#ifndef PULSEG_HW_FREQ_CONVERSION
#define PULSEG_HW_FREQ_CONVERSION 0.25f
#endif

#ifndef PULSEG_HW_WAVEFORM_END
#define PULSEG_HW_WAVEFORM_END 0
#endif

/* ------------------------------------------------------------------ */
/*  Binary cache defaults (D10)                                       */
/* ------------------------------------------------------------------ */

#ifndef PULSEG_CACHE_EXT_MAX
#define PULSEG_CACHE_EXT_MAX 16
#endif

#ifndef PULSEG_CACHE_EXT_DEFAULT
#define PULSEG_CACHE_EXT_DEFAULT ".pseg"
#endif

#endif /* PULSEG_CONFIG_H */
