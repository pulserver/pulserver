/* Heap accounting for bench_pipeline. See bench_alloc.c. */
#ifndef BENCH_ALLOC_H
#define BENCH_ALLOC_H

#include <stddef.h>

/** Bytes currently held by the C library (allocator rounding included). */
size_t bench_heap_live(void);

/** High-water mark of bench_heap_live() since the last reset. */
size_t bench_heap_peak(void);

/** Set the high-water mark back to the current live figure. */
void bench_heap_reset_peak(void);

#endif /* BENCH_ALLOC_H */
