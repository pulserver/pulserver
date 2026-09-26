/* pulseg_config.h -- platform configuration
 *
 * This header MUST be included (directly or transitively) before any
 * other pulseg header or source file.  It provides:
 *
 *   PULSEG_VENDOR_*  -- vendor ID constants (used at runtime)
 *   PULSEG_VENDOR    -- compile-time default vendor
 *   PULSEG_ALLOC     -- heap allocator  (default: malloc)
 *   PULSEG_FREE      -- heap deallocator (default: free)
 *   PULSEG_WAVE_*    -- the type and scale of a waveform-memory sample
 */

#ifndef PULSEG_CONFIG_H
#define PULSEG_CONFIG_H

/* Suppress -Wfloat-equal for intentional exact float comparisons
 * (shape decompression RLE, zero-detection, etc.).
 * Required by toolchains that build with -Werror -Wfloat-equal. */
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif

#include <stdlib.h>

/* ================================================================== */
/*  Vendor identifiers (runtime constants)                            */
/* ================================================================== */
#define PULSEG_VENDOR_UNSPECIFIED 0
#define PULSEG_VENDOR_SIEMENS 1
#define PULSEG_VENDOR_GEHC 2
#define PULSEG_VENDOR_PHILIPS 3
#define PULSEG_VENDOR_UNITED_IMAGING 4
#define PULSEG_VENDOR_BRUKER 5

/* Compile-time default (overrideable via -DPULSEG_VENDOR=N). Public
 * builds are vendor-neutral; a vendor layer defines PULSEG_VENDOR before
 * including any pulseg header. */
#ifndef PULSEG_VENDOR
#define PULSEG_VENDOR PULSEG_VENDOR_UNSPECIFIED
#endif

/* ================================================================== */
/*  Allocator overrides                                               */
/* ================================================================== */

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

/* ================================================================== */
/*  Waveform-memory samples                                           */
/* ================================================================== */

/*
 * A playout's waveform memory holds samples of type PULSEG_WAVE_SAMPLE.
 * A value normalised to unit peak is scaled by PULSEG_WAVE_FULL_SCALE, and
 * a phase, in radians, by PULSEG_WAVE_FULL_SCALE / PULSEG_WAVE_PHASE_FULL_SCALE;
 * PULSEG_WAVE_QUANTIZE(x) turns the scaled value, a double within
 * [-PULSEG_WAVE_FULL_SCALE, PULSEG_WAVE_FULL_SCALE], into a sample.  The
 * physical scale of a gradient or an RF magnitude is not in its samples: it
 * is the amplitude each block plays them at.  Override before including
 * this header, e.g. for signed 16-bit samples:
 *
 *   #define PULSEG_WAVE_SAMPLE          short
 *   #define PULSEG_WAVE_FULL_SCALE      32767.0
 *   #define PULSEG_WAVE_QUANTIZE(x)     ((short)((x) < 0.0 ? (x) - 0.5 : (x) + 0.5))
 *   #include "pulseg_config.h"
 */
#ifndef PULSEG_WAVE_SAMPLE
#define PULSEG_WAVE_SAMPLE float
#endif

#ifndef PULSEG_WAVE_FULL_SCALE
#define PULSEG_WAVE_FULL_SCALE 1.0
#endif

/* The phase, in radians, a sample of PULSEG_WAVE_FULL_SCALE plays: with pi,
 * the signed samples span the 2 pi a phase is wrapped into. */
#ifndef PULSEG_WAVE_PHASE_FULL_SCALE
#define PULSEG_WAVE_PHASE_FULL_SCALE 3.14159265358979323846
#endif

#ifndef PULSEG_WAVE_QUANTIZE
#define PULSEG_WAVE_QUANTIZE(x) ((PULSEG_WAVE_SAMPLE)(x))
#endif

/* ================================================================== */
/*  Binary cache defaults                                       */
/* ================================================================== */

#ifndef PULSEG_CACHE_EXT_MAX
#define PULSEG_CACHE_EXT_MAX 16
#endif

#ifndef PULSEG_CACHE_EXT_DEFAULT
#define PULSEG_CACHE_EXT_DEFAULT ".pseg"
#endif

#endif /* PULSEG_CONFIG_H */
