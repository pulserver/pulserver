/* pulseqlib_config.h -- platform configuration
 *
 * This header MUST be included (directly or transitively) before any
 * other pulseqlib header or source file.  It provides:
 *
 *   PULSEQLIB_VENDOR_*  -- vendor ID constants (used at runtime)
 *   PULSEQLIB_VENDOR    -- compile-time default vendor
 *   PULSEQLIB_ALLOC     -- heap allocator  (default: malloc)
 *   PULSEQLIB_FREE      -- heap deallocator (default: free)
 */

#ifndef PULSEQLIB_CONFIG_H
#define PULSEQLIB_CONFIG_H

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
#define PULSEQLIB_VENDOR_SIEMENS        1
#define PULSEQLIB_VENDOR_GEHC           2
#define PULSEQLIB_VENDOR_PHILIPS        3
#define PULSEQLIB_VENDOR_UNITED_IMAGING 4
#define PULSEQLIB_VENDOR_BRUKER         5

/* Compile-time default (overrideable via -DPULSEQLIB_VENDOR=N) */
#ifndef PULSEQLIB_VENDOR
#define PULSEQLIB_VENDOR PULSEQLIB_VENDOR_GEHC
#endif

/* ------------------------------------------------------------------ */
/*  Acoustic peak-detection defaults                                  */
/* ------------------------------------------------------------------ */

#ifndef PULSEQLIB_PEAK_LOG10_THRESHOLD_DEFAULT
#define PULSEQLIB_PEAK_LOG10_THRESHOLD_DEFAULT 2.25f
#endif

#ifndef PULSEQLIB_PEAK_NORM_SCALE_DEFAULT
#define PULSEQLIB_PEAK_NORM_SCALE_DEFAULT 10.0f
#endif

#ifndef PULSEQLIB_PEAK_EPS_DEFAULT
#define PULSEQLIB_PEAK_EPS_DEFAULT 1e-30f
#endif

#ifndef PULSEQLIB_PEAK_PROMINENCE_DEFAULT
#define PULSEQLIB_PEAK_PROMINENCE_DEFAULT 0.0f
#endif

/* ------------------------------------------------------------------ */
/*  Structural mechanical resonance analysis defaults                  */
/* ------------------------------------------------------------------ */

/** Minimum waveform samples before sub-period detection is applied. */
#ifndef PULSEQLIB_MIN_ARBITRARY_SAMPLES
#define PULSEQLIB_MIN_ARBITRARY_SAMPLES 10
#endif

/* ------------------------------------------------------------------ */
/*  Allocator overrides                                               */
/* ------------------------------------------------------------------ */

/*
 * Override PULSEQLIB_ALLOC / PULSEQLIB_FREE *before* including this
 * header to use vendor-specific allocators, e.g.:
 *
 *   #define PULSEQLIB_ALLOC(sz)  MyVendorAlloc(sz)
 *   #define PULSEQLIB_FREE(ptr)  MyVendorFree(ptr)
 *   #include "pulseqlib_config.h"
 */
#ifndef PULSEQLIB_ALLOC
#define PULSEQLIB_ALLOC(sz) malloc(sz)
#endif

#ifndef PULSEQLIB_FREE
#define PULSEQLIB_FREE(ptr) free(ptr)
#endif

/* ------------------------------------------------------------------ */
/*  Hardware frequency-conversion overrides                           */
/* ------------------------------------------------------------------ */

/*
 * Override PULSEQLIB_FREQ_CONVERSION / PULSEQLIB_WAVEFORM_END before
 * including this header for vendor-specific DAC parameters, e.g.:
 *
 *   #define PULSEQLIB_FREQ_CONVERSION  VendorFreqRes
 *   #define PULSEQLIB_WAVEFORM_END     VendorWeosBit
 *   #include "pulseqlib_config.h"
 */
#ifndef PULSEQLIB_FREQ_CONVERSION
#define PULSEQLIB_FREQ_CONVERSION 0.25f
#endif

#ifndef PULSEQLIB_WAVEFORM_END
#define PULSEQLIB_WAVEFORM_END 0
#endif

#endif /* PULSEQLIB_CONFIG_H */
