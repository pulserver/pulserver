/*
 * Print a cache's summary as a scanner build of the library loads it: one
 * "key value" line per quantity, in the order pulserver.ir.summary returns
 * them; and, given a waveform memory and a raster, the layout of its rotated
 * waves, as pulserver.ir.plan_waves returns it.  Compiled by tests/test_ir.py
 * with the scanner's word size and vendor.
 */

#include <stdio.h>
#include <stdlib.h>

#include "pulseg.h"
#include "pulseg_cache.h"

static void print_subsequence(const pulseg_collection *coll, int i)
{
    pulseg_subseq_info s = PULSEG_SUBSEQ_INFO_INIT;
    pulseg_tr_group *groups = NULL;
    int n, num_groups, num_waves;

    pulseg_get_subseq_info(coll, &s, i);
    printf("subsequence %d num_trs %d tr_size %d num_unique_adcs %d num_unique_rf %d "
           "vop_sar_ratio %g vop_global_sar_ratio %g\n",
           i, s.num_trs, s.tr_size, s.num_unique_adcs, s.num_unique_rf,
           (double)s.vop_sar_ratio, (double)s.vop_global_sar_ratio);
    num_groups = pulseg_get_tr_groups(coll, &groups, i);
    for (n = 0; n < num_groups; ++n)
        printf("group %d %d trid %d num_instances %d one_instance_duration_us %d\n",
               i, n, groups[n].trid, groups[n].num_instances,
               groups[n].one_instance_duration_us);
    if (groups)
        free(groups);
    num_waves = pulseg_get_num_waves(coll, i);
    for (n = 0; n < num_waves; ++n)
    {
        float peak[3] = {0.0f, 0.0f, 0.0f};
        int axis, points = 0;
        for (axis = 0; axis < 3; ++axis)
            pulseg_materialize_wave(coll, i, n, axis, NULL, NULL, 0, &points, &peak[axis]);
        printf("wave %d %d points %d peak %.4f %.4f %.4f\n", i, n, points,
               (double)peak[0], (double)peak[1], (double)peak[2]);
    }
}

static void print_region(const char *what, int i, int j, const pulseg_wave_region *region)
{
    printf("%s %d %d offset %ld %ld %ld samples %ld start_us %.3f\n", what, i, j,
           region->offset[0], region->offset[1], region->offset[2], region->samples,
           (double)region->start_us);
}

static void print_wave_plan(const pulseg_collection *coll, long max_samples, float raster_us)
{
    pulseg_wave_budget budget = PULSEG_WAVE_BUDGET_INIT;
    pulseg_wave_plan plan = PULSEG_WAVE_PLAN_INIT;
    int i, j, rc;

    budget.max_samples = max_samples;
    budget.raster_us = raster_us;
    rc = pulseg_plan_waves(coll, &budget, &plan, NULL);
    printf("wave_plan rc %d mode %d samples %ld %ld %ld\n", rc, plan.mode, plan.samples[0],
           plan.samples[1], plan.samples[2]);
    for (i = 0; i < plan.num_subsequences; ++i)
        for (j = 0; j < plan.num_waves[i]; ++j)
            print_region("wave", i, j, &plan.waves[i][j]);
    for (i = 0; i < plan.num_segments; ++i)
        for (j = 0; j < 2 * plan.num_positions[i]; ++j)
            print_region("slot", i, j, &plan.slots[i][j]);
    pulseg_free_wave_plan(&plan);
}

int main(int argc, char **argv)
{
    pulseg_collection *coll;
    pulseg_collection_info info = PULSEG_COLLECTION_INFO_INIT;
    int rc;
    int i;

    if (argc != 3 && argc != 5)
    {
        fprintf(stderr, "usage: %s CACHE SOURCE_SIZE [MAX_SAMPLES RASTER_US]\n", argv[0]);
        return 2;
    }

    coll = pulseg_collection_alloc();
    if (!coll)
        return 1;
    rc = pulseg_load_cache(coll, argv[1], atoi(argv[2]));
    if (PULSEG_FAILED(rc) || PULSEG_FAILED(pulseg_get_collection_info(coll, &info)))
    {
        fprintf(stderr, "cannot load %s: %d\n", argv[1], rc);
        pulseg_collection_free(coll);
        return 1;
    }

    printf("num_subsequences %d\n", info.num_subsequences);
    printf("num_segments %d\n", info.num_segments);
    printf("max_adc_samples %d\n", info.max_adc_samples);
    printf("total_readouts %d\n", info.total_readouts);
    for (i = 0; i < info.num_subsequences; ++i)
        print_subsequence(coll, i);
    for (i = 0; i < info.num_segments; ++i)
    {
        pulseg_segment_info g = PULSEG_SEGMENT_INFO_INIT;
        pulseg_get_segment_info(coll, &g, i);
        printf("segment %d duration_us %d num_blocks %d start_block %d is_nav %d\n",
               i, g.duration_us, g.num_blocks, g.start_block, g.is_nav);
    }

    if (argc == 5)
        print_wave_plan(coll, atol(argv[3]), (float)atof(argv[4]));

    pulseg_collection_free(coll);
    return 0;
}
