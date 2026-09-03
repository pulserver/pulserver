/*
 * The library's own parallel hook, for a build that has POSIX threads.
 *
 * pulseg_opts.parallel_for_fn deals a loop of independent items to the
 * cores the host has. A caller that installs its own hook keeps it; one
 * that leaves the field NULL gets this one when the library was built
 * with PULSEG_HAVE_PTHREADS, and a sequential loop otherwise. Nothing here
 * is compiled into a build without the define, which keeps the scanner
 * target free of any dependency.
 */
#ifdef PULSEG_HAVE_PTHREADS
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <pthread.h>
#include <unistd.h>
#endif

#include "pulseg_internal.h"

#ifdef PULSEG_HAVE_PTHREADS

#define PULSEG_PARALLEL_WORKERS 8

typedef struct
{
    void (*body)(void *arg, int begin, int end);
    void *arg;
    int count;
    int chunk;
    int next;
    pthread_mutex_t lock;
} pulseg_parallel_job;

static void *pulseg_parallel_worker(void *p)
{
    pulseg_parallel_job *job = (pulseg_parallel_job *)p;
    for (;;)
    {
        int begin, end;
        pthread_mutex_lock(&job->lock);
        begin = job->next;
        job->next += job->chunk;
        pthread_mutex_unlock(&job->lock);
        if (begin >= job->count)
            return NULL;
        end = begin + job->chunk;
        if (end > job->count)
            end = job->count;
        job->body(job->arg, begin, end);
    }
}

/* Ranges of about count / (workers * 8) items: a body that sets up per
 * range (an FFT plan, a scratch buffer) amortises it, and the tail still
 * balances across the pool. The calling thread works alongside. */
static void pulseg_parallel_for_posix(
    void *ctx,
    int count,
    void (*body)(void *arg, int begin, int end),
    void *arg)
{
    pthread_t threads[PULSEG_PARALLEL_WORKERS];
    pulseg_parallel_job job;
    long cores;
    int workers, w, started;
    (void)ctx;
    if (count <= 0)
        return;
    cores = sysconf(_SC_NPROCESSORS_ONLN);
    workers = (cores < 1) ? 1 : (int)cores;
    if (workers > PULSEG_PARALLEL_WORKERS)
        workers = PULSEG_PARALLEL_WORKERS;
    if (workers == 1 || count < 4)
    {
        body(arg, 0, count);
        return;
    }
    job.body = body;
    job.arg = arg;
    job.count = count;
    job.chunk = count / (workers * 4);
    if (job.chunk < 1)
        job.chunk = 1;
    job.next = 0;
    if (pthread_mutex_init(&job.lock, NULL) != 0)
    {
        body(arg, 0, count);
        return;
    }
    started = 0;
    for (w = 1; w < workers; ++w)
    {
        if (pthread_create(&threads[started], NULL, pulseg_parallel_worker, &job) == 0)
            ++started;
    }
    pulseg_parallel_worker(&job);
    for (w = 0; w < started; ++w)
        pthread_join(threads[w], NULL);
    pthread_mutex_destroy(&job.lock);
}

pulseg__parallel_for_fn pulseg__parallel_for_default(void)
{
    return pulseg_parallel_for_posix;
}

#else

pulseg__parallel_for_fn pulseg__parallel_for_default(void)
{
    return NULL;
}

#endif

pulseg__parallel_for_fn pulseg__opts_par_fn(const pulseg_opts *opts)
{
    if (opts && opts->parallel_for_fn)
        return opts->parallel_for_fn;
    return pulseg__parallel_for_default();
}

void *pulseg__opts_par_ctx(const pulseg_opts *opts)
{
    if (opts && opts->parallel_for_fn)
        return opts->parallel_ctx;
    return NULL;
}
