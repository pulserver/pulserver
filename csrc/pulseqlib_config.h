/* pulseqlib_config.h -- vendor and platform configuration
 *
 * This header MUST be included (directly or transitively) before any other
 * pulseqlib header or source file.  It provides:
 *
 *   PULSEQLIB_VENDOR   -- active vendor ID
 *   PULSEQLIB_ALLOC    -- heap allocator  (default: malloc)
 *   PULSEQLIB_FREE     -- heap deallocator (default: free)
 *
 * Users may pre-define these macros before including this header
 * to override the defaults.
 */

#ifndef PULSEQLIB_CONFIG_H
#define PULSEQLIB_CONFIG_H

#include <stdlib.h>

/* ------------------------------------------------------------------ */
/*  Vendor identifiers                                                 */
/* ------------------------------------------------------------------ */
#define PULSEQLIB_VENDOR_SIEMENS        1
#define PULSEQLIB_VENDOR_GEHC           2
#define PULSEQLIB_VENDOR_PHILIPS        3
#define PULSEQLIB_VENDOR_UNITED_IMAGING 4
#define PULSEQLIB_VENDOR_BRUKER         5

#ifndef PULSEQLIB_VENDOR
#define PULSEQLIB_VENDOR PULSEQLIB_VENDOR_GEHC
#endif

/* ------------------------------------------------------------------ */
/*  RF detection mode                                                  */
/* ------------------------------------------------------------------ */
#if PULSEQLIB_VENDOR == PULSEQLIB_VENDOR_GEHC
#define PULSEQLIB_DETECT_REAL_RF 1
#else
#define PULSEQLIB_DETECT_REAL_RF 0
#endif

/* ------------------------------------------------------------------ */
/*  Allocator overrides                                                */
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

#endif /* PULSEQLIB_CONFIG_H */