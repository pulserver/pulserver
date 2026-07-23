/*
 * bench_alloc.c -- exact heap accounting for bench_pipeline.
 *
 * Documentation-only tooling. Linked with -Wl,--wrap=malloc (and friends) so
 * every allocation the C library makes passes through here, whatever route it
 * took: PULSEG_ALLOC, PULSEQ_ALLOC, or the handful of direct realloc/free
 * calls in the trajectory builder. That matters because the point of the
 * measurement is the memory a scanner has to find, and undercounting one path
 * would flatter exactly the stage that allocates most.
 *
 * malloc_usable_size() is used rather than the requested size, so the numbers
 * include allocator rounding -- again, what the process actually holds rather
 * than what it asked for.
 *
 * Not thread-safe; the benchmark is single-threaded.
 */

#include <malloc.h>
#include <stddef.h>

#include "bench_alloc.h"

void *__real_malloc(size_t size);
void *__real_calloc(size_t nmemb, size_t size);
void *__real_realloc(void *ptr, size_t size);
void __real_free(void *ptr);

static size_t g_live;
static size_t g_peak;

static void account(long delta)
{
    if (delta < 0 && (size_t)(-delta) > g_live)
        g_live = 0;
    else
        g_live = (size_t)((long)g_live + delta);
    if (g_live > g_peak)
        g_peak = g_live;
}

void *__wrap_malloc(size_t size)
{
    void *p = __real_malloc(size);
    if (p)
        account((long)malloc_usable_size(p));
    return p;
}

void *__wrap_calloc(size_t nmemb, size_t size)
{
    void *p = __real_calloc(nmemb, size);
    if (p)
        account((long)malloc_usable_size(p));
    return p;
}

void *__wrap_realloc(void *ptr, size_t size)
{
    size_t before = ptr ? malloc_usable_size(ptr) : 0;
    void *p = __real_realloc(ptr, size);
    if (p)
        account((long)malloc_usable_size(p) - (long)before);
    return p;
}

void __wrap_free(void *ptr)
{
    if (ptr)
        account(-(long)malloc_usable_size(ptr));
    __real_free(ptr);
}

size_t bench_heap_live(void)
{
    return g_live;
}

size_t bench_heap_peak(void)
{
    return g_peak;
}

void bench_heap_reset_peak(void)
{
    g_peak = g_live;
}
