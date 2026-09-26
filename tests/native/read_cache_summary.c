/*
 * Print a cache's summary as a scanner build of the library loads it: one
 * "key value" line per quantity, in the order pulserver.ir.summary returns
 * them. Compiled by tests/test_ir.py with the scanner's word size and vendor.
 */

#include <stdio.h>
#include <stdlib.h>

#include "pulseg.h"
#include "pulseg_cache.h"

int main(int argc, char **argv)
{
    pulseg_collection *coll;
    pulseg_collection_info info = PULSEG_COLLECTION_INFO_INIT;
    int rc;
    int i;

    if (argc != 3)
    {
        fprintf(stderr, "usage: %s CACHE SOURCE_SIZE\n", argv[0]);
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
    for (i = 0; i < info.num_segments; ++i)
    {
        pulseg_segment_info g = PULSEG_SEGMENT_INFO_INIT;
        pulseg_get_segment_info(coll, &g, i);
        printf("segment %d duration_us %d num_blocks %d start_block %d is_nav %d\n",
               i, g.duration_us, g.num_blocks, g.start_block, g.is_nav);
    }

    pulseg_collection_free(coll);
    return 0;
}
