/*
 * bench_pipeline.c -- runtime and heap footprint of the on-scanner C pipeline.
 *
 * Documentation-only tooling: it produces the tables in
 * docs/explanations/benchmarks.md and is not part of the shipped library.
 * Everything it calls is public pulseg/pulseq API, so it doubles as a worked
 * example of driving the pipeline one stage at a time -- pulseg_read() fuses
 * the first three.
 *
 * The stages, in the order the scanner runs them:
 *
 *   HOST, once per prescription (predownload)
 *     parse      pulseq_file_set_read()            .seq text -> raw model
 *     convert    pulseg_convert_collection()       raw model -> pulseg IR
 *     write      pulseg_save_cache()               IR        -> cache file
 *
 *   TARGET, once per scan
 *     load_gen   pulseg_load_geninstructions_cache()  COMMON + SHAPES
 *     load_scan  pulseg_load_scanloop_cache()         COMMON + INSTANCES +
 *                                                     ROTATIONS + SHAPES +
 *                                                     SCANLOOP
 *     load_full  pulseg_load_cache()                  every section, for
 *                                                     reference
 *
 * Each stage reports wall time (minimum of N repeats) and the heap it leaves
 * live, measured exactly by interposing the allocator (see bench_alloc.c).
 * Live heap after a load is the figure that matters on the target: it is what
 * has to fit, and it is why the per-consumer section split exists.
 *
 * Usage:  bench_pipeline <sequence.seq> [repeats]
 * Output: one JSON object on stdout.
 *
 * Build:  bash docs/_bench/build_bench.sh
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>

#include "pulseg.h"
/* Private, for the base-block dedup count: how many unique block definitions
 * the raw block list collapses to is the whole point of the conversion stage,
 * and no public getter exposes it. The C test suite includes this header for
 * the same reason. */
#include "pulseg_internal.h"

#include "bench_alloc.h"

#define GAMMA_HZ_PER_T 42577478.0f
#define MAX_GRAD_MT_PER_M 50.0f
#define MAX_SLEW_T_PER_M_PER_S 200.0f

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static long peak_rss_kb(void)
{
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss;
}

static long file_size(const char *path)
{
    FILE *f = fopen(path, "rb");
    long n;
    if (!f)
        return -1;
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fclose(f);
    return n;
}

/* Replace the .seq extension with the cache extension pulseg_save_cache uses. */
static void cache_path_of(char *out, size_t cap, const char *seq_path, const char *ext)
{
    const char *dot = strrchr(seq_path, '.');
    size_t stem = dot ? (size_t)(dot - seq_path) : strlen(seq_path);
    if (stem >= cap)
        stem = cap - 1;
    memcpy(out, seq_path, stem);
    out[stem] = '\0';
    strncat(out, ext, cap - strlen(out) - 1);
}

static void opts_init(pulseg_opts *opts)
{
    pulseg_opts_init(
        opts,
        GAMMA_HZ_PER_T,
        3.0f,
        MAX_GRAD_MT_PER_M * 1e-3f * GAMMA_HZ_PER_T,
        MAX_SLEW_T_PER_M_PER_S * GAMMA_HZ_PER_T,
        /*rf_raster_us=*/2.0f,
        /*grad_raster_us=*/10.0f,
        /*adc_raster_us=*/2.0f,
        /*block_raster_us=*/10.0f);
}

/* Per-stage measurements, all minima over the repeat count. */
typedef struct
{
    double parse_ms, convert_ms, write_ms;
    double load_full_ms, load_gen_ms, load_scan_ms;
    size_t heap_parse, heap_convert;
    size_t heap_full, heap_gen, heap_scan;
    size_t heap_peak_convert;
    int raw_blocks, unique_blocks;
} stages;

/* One parse + convert. Records the heap held by the raw model alone, then by
 * the collection alone after the raw model is released -- the two are never
 * live at the same moment on the scanner either. */
static pulseg_collection *parse_and_convert(const char *seq_path, stages *st)
{
    pulseg_opts opts = PULSEG_OPTS_INIT;
    pulseq_raster raster;
    pulseq_file_set raw;
    pulseg_collection *coll;
    pulseg_diagnostic diag;
    size_t base;
    double t0, dt;
    int rc, i;

    raw.num_sequences = 0;
    raw.sequences = NULL;
    raw.base_path = NULL;

    opts_init(&opts);
    pulseg_opts_get_design_raster(&raster, &opts);

    base = bench_heap_live();
    bench_heap_reset_peak();

    t0 = now_ms();
    rc = pulseq_file_set_read(&raw, seq_path, &raster);
    dt = now_ms() - t0;
    if (PULSEG_FAILED(rc))
    {
        fprintf(stderr, "pulseq_file_set_read failed: %d\n", rc);
        pulseq_file_set_free(&raw);
        return NULL;
    }
    if (dt < st->parse_ms)
        st->parse_ms = dt;
    if (bench_heap_live() - base > st->heap_parse)
        st->heap_parse = bench_heap_live() - base;

    st->raw_blocks = 0;
    for (i = 0; i < raw.num_sequences; ++i)
        st->raw_blocks += raw.sequences[i].num_blocks;

    coll = (pulseg_collection *)calloc(1, sizeof(pulseg_collection));
    if (!coll)
    {
        pulseq_file_set_free(&raw);
        return NULL;
    }

    pulseg_diagnostic_init(&diag);
    t0 = now_ms();
    rc = pulseg_convert_collection(
        coll, &diag, raw.sequences, raw.num_sequences, &opts, /*parse_labels=*/1, /*num_averages=*/1);
    dt = now_ms() - t0;
    if (rc <= 0)
    {
        fprintf(stderr, "pulseg_convert_collection failed: %d (%s)\n", diag.code, diag.message);
        pulseq_file_set_free(&raw);
        free(coll);
        return NULL;
    }
    if (dt < st->convert_ms)
        st->convert_ms = dt;
    if (bench_heap_peak() - base > st->heap_peak_convert)
        st->heap_peak_convert = bench_heap_peak() - base;

    pulseq_file_set_free(&raw);
    if (bench_heap_live() - base > st->heap_convert)
        st->heap_convert = bench_heap_live() - base;

    st->unique_blocks = coll->descriptors[0].num_unique_blocks;
    return coll;
}

/* Time one cache load and record the heap the loaded collection holds. */
typedef int (*loader_fn)(pulseg_collection **out, const char *path);

/* pulseg_load_cache() re-checks the source .seq size recorded in the cache
 * header; the two per-consumer loaders take the .seq path and do not. Carried
 * in a file-scope variable so all three share one loader_fn signature. */
static int g_expected_source_size;

static int load_full_adapter(pulseg_collection **out, const char *path)
{
    pulseg_collection *coll = (pulseg_collection *)calloc(1, sizeof(pulseg_collection));
    int rc;
    if (!coll)
        return PULSEG_ERR_ALLOC_FAILED;
    rc = pulseg_load_cache(coll, path, g_expected_source_size);
    if (PULSEG_FAILED(rc))
    {
        free(coll);
        return rc;
    }
    *out = coll;
    return PULSEG_SUCCESS;
}

static int time_load(
    loader_fn load,
    const char *path,
    int repeats,
    double *best_ms,
    size_t *heap_bytes,
    const char *label)
{
    int i;
    for (i = 0; i < repeats; ++i)
    {
        pulseg_collection *coll = NULL;
        size_t base = bench_heap_live();
        double t0 = now_ms(), dt;
        int rc = load(&coll, path);
        dt = now_ms() - t0;
        if (PULSEG_FAILED(rc))
        {
            fprintf(stderr, "%s failed: %d\n", label, rc);
            return rc;
        }
        if (dt < *best_ms)
            *best_ms = dt;
        if (bench_heap_live() - base > *heap_bytes)
            *heap_bytes = bench_heap_live() - base;
        pulseg_collection_free(coll);
    }
    return PULSEG_SUCCESS;
}

int main(int argc, char **argv)
{
    const char *seq_path;
    char cache_path[1024];
    int repeats, i, rc;
    pulseg_collection *coll = NULL;
    pulseg_collection_info info = PULSEG_COLLECTION_INFO_INIT;
    pulseg_subseq_info sub = PULSEG_SUBSEQ_INFO_INIT;
    long seq_bytes, cache_bytes;
    stages st;

    if (argc < 2)
    {
        fprintf(stderr, "usage: %s <sequence.seq> [repeats]\n", argv[0]);
        return 2;
    }
    seq_path = argv[1];
    repeats = (argc > 2) ? atoi(argv[2]) : 3;
    if (repeats < 1)
        repeats = 1;

    memset(&st, 0, sizeof(st));
    st.parse_ms = st.convert_ms = st.write_ms = 1e30;
    st.load_full_ms = st.load_gen_ms = st.load_scan_ms = 1e30;

    cache_path_of(cache_path, sizeof(cache_path), seq_path, PULSEG_CACHE_EXT_DEFAULT);
    seq_bytes = file_size(seq_path);

    for (i = 0; i < repeats; ++i)
    {
        if (coll)
        {
            pulseg_collection_free(coll);
            coll = NULL;
        }
        coll = parse_and_convert(seq_path, &st);
        if (!coll)
            return 1;
    }
    pulseg_get_collection_info(coll, &info);
    pulseg_get_subseq_info(coll, &sub, 0);

    for (i = 0; i < repeats; ++i)
    {
        double t0 = now_ms(), dt;
        rc = pulseg_save_cache(coll, cache_path, (int)seq_bytes);
        dt = now_ms() - t0;
        if (PULSEG_FAILED(rc))
        {
            fprintf(stderr, "pulseg_save_cache failed: %d\n", rc);
            pulseg_collection_free(coll);
            return 1;
        }
        if (dt < st.write_ms)
            st.write_ms = dt;
    }
    cache_bytes = file_size(cache_path);
    pulseg_collection_free(coll);

    g_expected_source_size = (int)seq_bytes;
    if (PULSEG_FAILED(time_load(
            load_full_adapter, cache_path, repeats, &st.load_full_ms, &st.heap_full, "pulseg_load_cache")))
        return 1;
    if (PULSEG_FAILED(time_load(
            pulseg_load_geninstructions_cache,
            seq_path,
            repeats,
            &st.load_gen_ms,
            &st.heap_gen,
            "pulseg_load_geninstructions_cache")))
        return 1;
    if (PULSEG_FAILED(time_load(
            pulseg_load_scanloop_cache,
            seq_path,
            repeats,
            &st.load_scan_ms,
            &st.heap_scan,
            "pulseg_load_scanloop_cache")))
        return 1;

    printf(
        "{\"seq\": \"%s\", \"seq_bytes\": %ld, \"cache_bytes\": %ld, "
        "\"raw_blocks\": %d, \"base_blocks\": %d, \"segments\": %d, "
        "\"subsequences\": %d, \"readouts\": %d, \"tr_size\": %d, \"num_trs\": %d, "
        "\"parse_ms\": %.3f, \"convert_ms\": %.3f, \"cache_write_ms\": %.3f, "
        "\"load_full_ms\": %.3f, \"load_pulsegen_ms\": %.3f, \"load_scanloop_ms\": %.3f, "
        "\"heap_parse_kb\": %.1f, \"heap_convert_kb\": %.1f, \"heap_convert_peak_kb\": %.1f, "
        "\"heap_full_kb\": %.1f, \"heap_pulsegen_kb\": %.1f, \"heap_scanloop_kb\": %.1f, "
        "\"rss_peak_kb\": %ld}\n",
        seq_path,
        seq_bytes,
        cache_bytes,
        st.raw_blocks,
        st.unique_blocks,
        info.num_segments,
        info.num_subsequences,
        info.total_readouts,
        sub.tr_size,
        sub.num_trs,
        st.parse_ms,
        st.convert_ms,
        st.write_ms,
        st.load_full_ms,
        st.load_gen_ms,
        st.load_scan_ms,
        st.heap_parse / 1024.0,
        st.heap_convert / 1024.0,
        st.heap_peak_convert / 1024.0,
        st.heap_full / 1024.0,
        st.heap_gen / 1024.0,
        st.heap_scan / 1024.0,
        peak_rss_kb());

    return 0;
}
