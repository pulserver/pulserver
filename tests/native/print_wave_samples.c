/*
 * Print waveform-memory samples as a scanner build converts them, with
 * whatever PULSEG_WAVE_* macros it is compiled with:
 *
 *   values V...                            pulseg_wave_samples()
 *   cycles V... / radians V...             pulseg_phase_samples()
 *   loads CACHE SOURCE_SIZE MAX_SAMPLES RASTER_US
 *                                          every load both stages of a
 *                                          playout make
 *
 * one line per conversion or load, the samples as numbers.  Compiled by
 * tests/test_ir.py with the scanner's word size and vendor.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pulseg.h"
#include "pulseg_cache.h"

static void print_samples(const PULSEG_WAVE_SAMPLE *samples, long count)
{
    long i;

    for (i = 0; i < count; ++i)
        printf(" %.9g", (double)samples[i]);
    printf("\n");
}

static int convert_values(
    const char *mode,
    const float *values,
    int count,
    PULSEG_WAVE_SAMPLE *samples)
{
    if (strcmp(mode, "values") == 0)
        pulseg_wave_samples(values, count, samples);
    else if (strcmp(mode, "cycles") == 0)
        pulseg_phase_samples(values, count, 2.0 * 3.14159265358979323846, samples);
    else if (strcmp(mode, "radians") == 0)
        pulseg_phase_samples(values, count, 1.0, samples);
    else
        return 2;
    return 0;
}

static int convert(const char *mode, int count, char **numbers)
{
    float *values = (float *)malloc((size_t)(count + 1) * sizeof(float));
    PULSEG_WAVE_SAMPLE *samples =
        (PULSEG_WAVE_SAMPLE *)malloc((size_t)(count + 1) * sizeof(PULSEG_WAVE_SAMPLE));
    int i, rc = (values && samples) ? 0 : 1;

    for (i = 0; rc == 0 && i < count; ++i)
        values[i] = (float)atof(numbers[i]);
    if (rc == 0)
        rc = convert_values(mode, values, count, samples);
    if (rc == 0)
    {
        printf("%s", mode);
        print_samples(samples, count);
    }
    free(values);
    free(samples);
    return rc;
}

static int print_load(void *ctx, const pulseg_wave_load *load)
{
    (void)ctx;
    printf("load %d %d %d %ld %ld", load->subsequence, load->wave, load->axis, load->offset,
           load->count);
    print_samples(load->samples, load->count);
    return PULSEG_SUCCESS;
}

static int loads(const char *cache, int source_size, long max_samples, float raster_us)
{
    pulseg_collection *coll = pulseg_collection_alloc();
    pulseg_wave_budget budget = PULSEG_WAVE_BUDGET_INIT;
    pulseg_wave_plan plan = PULSEG_WAVE_PLAN_INIT;
    pulseg_playout_backend backend;
    int rc;

    if (!coll)
        return 1;
    rc = pulseg_load_cache(coll, cache, source_size);
    memset(&backend, 0, sizeof(backend));
    backend.load_wave = print_load;
    budget.max_samples = max_samples;
    budget.raster_us = raster_us;
    if (PULSEG_SUCCEEDED(rc))
        rc = pulseg_get_wave_plan(coll, &budget, &plan, NULL);
    if (PULSEG_SUCCEEDED(rc))
        rc = pulseg_playout_prepare(coll, &plan, &backend);
    if (PULSEG_SUCCEEDED(rc))
        rc = pulseg_playout_scan(coll, &plan, &backend, NULL, NULL);
    pulseg_free_wave_plan(&plan);
    pulseg_collection_free(coll);
    return PULSEG_SUCCEEDED(rc) ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc == 6 && strcmp(argv[1], "loads") == 0)
        return loads(argv[2], atoi(argv[3]), atol(argv[4]), (float)atof(argv[5]));
    if (argc >= 2)
        return convert(argv[1], argc - 2, argv + 2);
    fprintf(stderr, "usage: %s values|cycles|radians V... | loads CACHE SIZE SAMPLES RASTER\n",
            argv[0]);
    return 2;
}
