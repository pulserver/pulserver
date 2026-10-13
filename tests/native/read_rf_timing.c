/*
 * Print how a scanner build of the library reads the timing of each RF a
 * cache's segments play: one "rf <segment> <block> <uniform> <samples>
 * <duration in us>" line per RF-bearing block, <uniform> being the block
 * info's rf_uniform_raster.  Compiled by tests/test_ir_rf.py with the
 * scanner's word size and vendor.
 */

#include <stdio.h>
#include <stdlib.h>

#include "pulseg.h"
#include "pulseg_cache.h"

int main(int argc, char **argv)
{
    pulseg_collection *coll = NULL;
    pulseg_collection_info ci = PULSEG_COLLECTION_INFO_INIT;
    int seg, blk;

    if (argc != 3)
    {
        fprintf(stderr, "usage: %s CACHE SOURCE_SIZE\n", argv[0]);
        return 2;
    }
    coll = pulseg_collection_alloc();
    if (!coll)
        return 1;
    if (PULSEG_FAILED(pulseg_load_cache(coll, argv[1], atoi(argv[2]))))
    {
        pulseg_collection_free(coll);
        return 3;
    }
    pulseg_get_collection_info(coll, &ci);
    for (seg = 0; seg < ci.num_segments; ++seg)
    {
        pulseg_segment_info si = PULSEG_SEGMENT_INFO_INIT;
        pulseg_get_segment_info(coll, &si, seg);
        for (blk = 0; blk < si.num_blocks; ++blk)
        {
            pulseg_block_info bi = PULSEG_BLOCK_INFO_INIT;
            pulseg_get_block_info(coll, &bi, seg, blk);
            if (bi.has_rf)
                printf("rf %d %d %d %d %d\n", seg, blk, bi.rf_uniform_raster,
                       bi.rf_num_samples, bi.rf_duration_us);
        }
    }
    pulseg_collection_free(coll);
    return 0;
}
