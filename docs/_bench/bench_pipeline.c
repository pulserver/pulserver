/*
 * bench_pipeline.c -- runtime and heap footprint of the on-scanner C pipeline.
 *
 * Documentation-only tooling: it produces the tables in
 * docs/explanations/benchmarks.md and is not part of the shipped library.
 * The pipeline stages use public pulseg/pulseq API.  Safety uses one private
 * observer around pulseg_check_safety() so that the benchmark sees the exact
 * headless path (including shared setup) without putting a platform clock in
 * the scanner library.
 *
 * The stages, in the order the scanner runs them:
 *
 *   HOST, once per prescription (predownload)
 *     parse      pulseq_file_set_read()            .seq text -> raw model
 *     convert    pulseg_convert_collection()       raw model -> pulseg IR
 *     write      pulseg_save_cache()               IR        -> cache file
 *     safety     pulseg_check_safety()              headless gradient gate
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
 * Usage:  bench_pipeline <sequence.seq> [repeats] [cache-extension]
 * Output: one JSON object on stdout.
 *
 * Build:  bash docs/_bench/build_bench.sh
 */

#include <float.h>
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

#ifdef BENCH_ALLOC_DISABLED
#define HEAP_ACCOUNTING_JSON "false"
#else
#define HEAP_ACCOUNTING_JSON "true"
#endif

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

static int opts_init(pulseg_opts *opts, const char *seq_path)
{
    pulseq_file defs;
    const pulseq_reserved_definitions *rd;
    int rc;

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

    /* The benchmark is routinely used on sequences with 1/10 us and 2/4 us
     * design rasters.  Read only the tiny DEFINITIONS section up front so the
     * real parse is configured from the file rather than from benchmark
     * defaults.  This setup read is deliberately outside all timed stages. */
    pulseq_file_init(&defs, NULL);
    rc = pulseq_read_definitions_only(&defs, seq_path);
    if (PULSEG_FAILED(rc))
    {
        pulseq_file_free(&defs);
        return rc;
    }
    rd = &defs.reserved_definitions_library;
    if (rd->radiofrequency_raster_time > 0.0f)
        opts->rf_raster_us = rd->radiofrequency_raster_time * 1e6f;
    if (rd->gradient_raster_time > 0.0f)
        opts->grad_raster_us = rd->gradient_raster_time * 1e6f;
    if (rd->adc_raster_time > 0.0f)
        opts->adc_raster_us = rd->adc_raster_time * 1e6f;
    if (rd->block_duration_raster > 0.0f)
        opts->block_raster_us = rd->block_duration_raster * 1e6f;
    pulseq_file_free(&defs);
    return PULSEG_SUCCESS;
}

/* Per-stage measurements, all minima over the repeat count. */
typedef struct
{
    double parse_ms, convert_ms, write_ms;
    double load_full_ms, load_gen_ms, load_scan_ms;
    double safety_total_ms;
    double safety_stage_ms[PULSEG__SAFETY_PROFILE_STAGE_COUNT];
    size_t heap_parse, heap_convert;
    size_t heap_full, heap_gen, heap_scan;
    size_t heap_peak_convert;
    int raw_blocks, unique_blocks;
} stages;

typedef struct
{
    double entered_ms[PULSEG__SAFETY_PROFILE_STAGE_COUNT];
    double elapsed_ms[PULSEG__SAFETY_PROFILE_STAGE_COUNT];
} safety_run;

static void safety_profile_event(
    void *ctx,
    enum pulseg__safety_profile_stage stage,
    int entering)
{
    safety_run *run = (safety_run *)ctx;
    double t = now_ms();
    if (entering)
        run->entered_ms[stage] = t;
    else
        run->elapsed_ms[stage] += t - run->entered_ms[stage];
}

/* Representative GE Irnich/rheobase model.  This is the same callback shape
 * and arithmetic used by the interpreter; only the scanner-provided numeric
 * threshold is replaced by a public benchmark constant.  The threshold does
 * not affect runtime, while chronaxie (and therefore kernel length) does. */
typedef struct
{
    float chronaxie_us;
    float rheobase_hz_per_m_per_s;
    float alpha;
} benchmark_pns_ctx;

static int benchmark_pns_build_kernel(
    const benchmark_pns_ctx *ctx,
    float dt_us,
    float **out_kernel,
    int *out_len)
{
    float c_s, dt_s, s_min, *kernel;
    int i, n;
    if (!ctx || !out_kernel || !out_len || ctx->chronaxie_us <= 0.0f ||
        ctx->rheobase_hz_per_m_per_s <= 0.0f || ctx->alpha <= 0.0f || dt_us <= 0.0f)
        return PULSEG_ERR_PNS_INVALID_PARAMS;
    c_s = ctx->chronaxie_us * 1e-6f;
    dt_s = dt_us * 1e-6f;
    s_min = ctx->rheobase_hz_per_m_per_s / ctx->alpha;
    n = (int)(20.0f * c_s / dt_s) + 1;
    kernel = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    if (!kernel)
        return PULSEG_ERR_ALLOC_FAILED;
    for (i = 0; i < n; ++i)
    {
        float tau = (float)i * dt_s;
        float denom = (c_s + tau) * (c_s + tau);
        kernel[i] = (dt_s / s_min) * (c_s / denom);
    }
    *out_kernel = kernel;
    *out_len = n;
    return PULSEG_SUCCESS;
}

static int benchmark_pns_required_padding(void *ctx, float dt_us)
{
    float *kernel = NULL;
    int n = 0;
    int rc = benchmark_pns_build_kernel((const benchmark_pns_ctx *)ctx, dt_us, &kernel, &n);
    if (PULSEG_FAILED(rc))
        return rc;
    PULSEG_FREE(kernel);
    return n;
}

static int benchmark_pns_evaluate(
    void *ctx,
    const float *dgdt_x,
    const float *dgdt_y,
    const float *dgdt_z,
    int n,
    float dt_us,
    float *out_x,
    float *out_y,
    float *out_z)
{
    float *kernel = NULL;
    int kernel_len = 0, i, rc;
    rc = benchmark_pns_build_kernel(
        (const benchmark_pns_ctx *)ctx, dt_us, &kernel, &kernel_len);
    if (PULSEG_FAILED(rc))
        return rc;
    rc = pulseg__calc_convolution_fft(out_x, dgdt_x, n, kernel, kernel_len);
    if (!PULSEG_FAILED(rc))
        rc = pulseg__calc_convolution_fft(out_y, dgdt_y, n, kernel, kernel_len);
    if (!PULSEG_FAILED(rc))
        rc = pulseg__calc_convolution_fft(out_z, dgdt_z, n, kernel, kernel_len);
    if (!PULSEG_FAILED(rc))
        for (i = 0; i < n; ++i)
        {
            out_x[i] *= 100.0f;
            out_y[i] *= 100.0f;
            out_z[i] *= 100.0f;
        }
    PULSEG_FREE(kernel);
    return rc;
}

/* One parse + convert. Records the heap held by the raw model alone, then by
 * the collection alone after the raw model is released -- the two are never
 * live at the same moment on the scanner either. */
static pulseg_collection *parse_and_convert(
    const char *seq_path,
    const pulseg_opts *opts,
    stages *st)
{
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

    pulseg_opts_get_design_raster(&raster, opts);

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
        coll, &diag, raw.sequences, raw.num_sequences, opts, /*parse_labels=*/1, /*num_averages=*/1);
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
static const char *g_cache_ext;

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

static int load_gen_adapter(pulseg_collection **out, const char *path)
{
    return pulseg__load_geninstructions_cache_ext(out, path, g_cache_ext);
}

static int load_scan_adapter(pulseg_collection **out, const char *path)
{
    return pulseg__load_scanloop_cache_ext(out, path, g_cache_ext);
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
    const char *cache_ext;
    char cache_path[1024];
    int repeats, i, j, rc;
    pulseg_collection *coll = NULL;
    pulseg_opts opts = PULSEG_OPTS_INIT;
    pulseg_diagnostic diag = PULSEG_DIAGNOSTIC_INIT;
    pulseg_collection_info info = PULSEG_COLLECTION_INFO_INIT;
    pulseg_subseq_info sub = PULSEG_SUBSEQ_INFO_INIT;
    pulseg_forbidden_band bands[] = {
        {540.0f, 640.0f, FLT_MAX},
        {960.0f, 1080.0f, FLT_MAX},
        {1180.0f, 1260.0f, FLT_MAX}};
    benchmark_pns_ctx pns_ctx = {360.0f, 4.25e8f, 0.333f};
    pulseg_pns_model pns_model = {
        &pns_ctx, benchmark_pns_required_padding, benchmark_pns_evaluate};
    long seq_bytes, cache_bytes;
    stages st;

    if (argc < 2)
    {
        fprintf(stderr, "usage: %s <sequence.seq> [repeats] [cache-extension]\n", argv[0]);
        return 2;
    }
    seq_path = argv[1];
    repeats = (argc > 2) ? atoi(argv[2]) : 3;
    cache_ext = (argc > 3) ? argv[3] : PULSEG_CACHE_EXT_DEFAULT;
    if (repeats < 1)
        repeats = 1;
    if (cache_ext[0] != '.')
    {
        fprintf(stderr, "cache extension must start with '.': %s\n", cache_ext);
        return 2;
    }

    rc = opts_init(&opts, seq_path);
    if (PULSEG_FAILED(rc))
    {
        fprintf(stderr, "could not read sequence rasters: %d\n", rc);
        return 1;
    }

    memset(&st, 0, sizeof(st));
    st.parse_ms = st.convert_ms = st.write_ms = 1e30;
    st.load_full_ms = st.load_gen_ms = st.load_scan_ms = 1e30;
    st.safety_total_ms = 1e30;
    for (j = 0; j < PULSEG__SAFETY_PROFILE_STAGE_COUNT; ++j)
        st.safety_stage_ms[j] = 1e30;

    cache_path_of(cache_path, sizeof(cache_path), seq_path, cache_ext);
    seq_bytes = file_size(seq_path);

    for (i = 0; i < repeats; ++i)
    {
        if (coll)
        {
            pulseg_collection_free(coll);
            coll = NULL;
        }
        coll = parse_and_convert(seq_path, &opts, &st);
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

    /* Headless safety, exactly as predownload calls it.  The bands are public
     * synthetic stand-ins and deliberately have an infinite pass threshold:
     * all structural candidate work runs, but this runtime benchmark cannot
     * fail merely because no site-specific ESP table was supplied. */
    for (i = 0; i < repeats; ++i)
    {
        safety_run run;
        double t0, dt;
        memset(&run, 0, sizeof(run));
        t0 = now_ms();
        rc = pulseg__check_safety_profiled(
            coll,
            &diag,
            &opts,
            (int)(sizeof(bands) / sizeof(bands[0])),
            bands,
            &pns_model,
            FLT_MAX,
            safety_profile_event,
            &run);
        dt = now_ms() - t0;
        if (PULSEG_FAILED(rc))
        {
            fprintf(stderr, "pulseg_check_safety failed: %d (%s)\n", rc, diag.message);
            pulseg_collection_free(coll);
            return 1;
        }
        if (dt < st.safety_total_ms)
            st.safety_total_ms = dt;
        for (j = 0; j < PULSEG__SAFETY_PROFILE_STAGE_COUNT; ++j)
            if (run.elapsed_ms[j] < st.safety_stage_ms[j])
                st.safety_stage_ms[j] = run.elapsed_ms[j];
    }
    pulseg_collection_free(coll);

    g_expected_source_size = (int)seq_bytes;
    g_cache_ext = cache_ext;
    if (PULSEG_FAILED(time_load(
            load_full_adapter, cache_path, repeats, &st.load_full_ms, &st.heap_full, "pulseg_load_cache")))
        return 1;
    if (PULSEG_FAILED(time_load(
            load_gen_adapter,
            seq_path,
            repeats,
            &st.load_gen_ms,
            &st.heap_gen,
            "pulseg_load_geninstructions_cache")))
        return 1;
    if (PULSEG_FAILED(time_load(
            load_scan_adapter,
            seq_path,
            repeats,
            &st.load_scan_ms,
            &st.heap_scan,
            "pulseg_load_scanloop_cache")))
        return 1;

    printf(
        "{\"seq\": \"%s\", \"cache\": \"%s\", \"heap_accounting\": %s, "
        "\"seq_bytes\": %ld, \"cache_bytes\": %ld, "
        "\"raw_blocks\": %d, \"base_blocks\": %d, \"segments\": %d, "
        "\"subsequences\": %d, \"readouts\": %d, \"tr_size\": %d, \"num_trs\": %d, "
        "\"parse_ms\": %.3f, \"convert_ms\": %.3f, \"cache_write_ms\": %.3f, "
        "\"safety_total_ms\": %.3f, \"safety_grad_presence_ms\": %.3f, "
        "\"safety_max_grad_ms\": %.3f, \"safety_continuity_ms\": %.3f, "
        "\"safety_max_slew_ms\": %.3f, \"safety_waveform_extract_ms\": %.3f, "
        "\"safety_mech_resonance_ms\": %.3f, \"safety_pns_ms\": %.3f, "
        "\"load_full_ms\": %.3f, \"load_pulsegen_ms\": %.3f, \"load_scanloop_ms\": %.3f, "
        "\"heap_parse_kb\": %.1f, \"heap_convert_kb\": %.1f, \"heap_convert_peak_kb\": %.1f, "
        "\"heap_full_kb\": %.1f, \"heap_pulsegen_kb\": %.1f, \"heap_scanloop_kb\": %.1f, "
        "\"rss_peak_kb\": %ld}\n",
        seq_path,
        cache_path,
        HEAP_ACCOUNTING_JSON,
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
        st.safety_total_ms,
        st.safety_stage_ms[PULSEG__SAFETY_PROFILE_GRAD_PRESENCE],
        st.safety_stage_ms[PULSEG__SAFETY_PROFILE_MAX_GRAD],
        st.safety_stage_ms[PULSEG__SAFETY_PROFILE_CONTINUITY],
        st.safety_stage_ms[PULSEG__SAFETY_PROFILE_MAX_SLEW],
        st.safety_stage_ms[PULSEG__SAFETY_PROFILE_WAVEFORM_EXTRACT],
        st.safety_stage_ms[PULSEG__SAFETY_PROFILE_MECH_RESONANCE],
        st.safety_stage_ms[PULSEG__SAFETY_PROFILE_PNS],
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
