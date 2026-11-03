/*
 * Print the output trigger each block of a cache's scan plays, in play order:
 * one "trigger DELAY_US DURATION_US" line per block that fires one, timed as
 * a scanner build of the library places it, from the block's definition.
 * Compiled by tests/test_ir.py with the scanner's word size and vendor.
 */

#include <stdio.h>
#include <stdlib.h>

#include "pulseg.h"
#include "pulseg_cache.h"

int main(int argc, char **argv)
{
    pulseg_collection *coll;
    pulseg_cursor_info cur = PULSEG_CURSOR_INFO_INIT;
    int rc, blk = 0;

    if (argc != 3)
    {
        fprintf(stderr, "usage: %s CACHE SOURCE_SIZE\n", argv[0]);
        return 2;
    }
    coll = pulseg_collection_alloc();
    if (!coll)
        return 1;
    rc = pulseg_load_cache(coll, argv[1], atoi(argv[2]));
    if (PULSEG_FAILED(rc))
    {
        fprintf(stderr, "cannot load %s: %d\n", argv[1], rc);
        pulseg_collection_free(coll);
        return 1;
    }
    pulseg_cursor_reset(coll);
    while (pulseg_cursor_advance(coll, &cur) == PULSEG_CURSOR_BLOCK)
    {
        pulseg_block_instance inst = PULSEG_BLOCK_INSTANCE_INIT;
        pulseg_block_info bi = PULSEG_BLOCK_INFO_INIT;
        if (cur.segment_start)
            blk = 0;
        if (!PULSEG_FAILED(pulseg_get_block_instance(coll, &inst)) && inst.digitalout_flag &&
            !PULSEG_FAILED(pulseg_get_block_info(coll, &bi, cur.segment_id, blk)))
            printf("trigger %d %d\n", bi.digitalout_delay_us, bi.digitalout_duration_us);
        ++blk;
    }
    pulseg_collection_free(coll);
    return 0;
}
