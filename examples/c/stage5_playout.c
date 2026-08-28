/*
 * stage5_playout.c -- walk the execution stream and play it.
 *
 * The scan loop. Everything structural was decided at preparation and
 * everything about waveforms at generation; what is left is to walk the
 * stream in the order it plays and hand the sequencer one block instance at a
 * time.
 *
 * A block *definition* is shared by every occurrence of it. A block
 * *instance* is one occurrence: the amplitudes it plays at, its rotation, its
 * RF shim, whether it acquires. That distinction is what keeps a million-block
 * scan in a few megabytes.
 *
 *   stage5_playout <scan.seq> [max_blocks]
 */

#include "vendor.h"

int main(int argc, char **argv)
{
    pulseg_collection *scan = NULL;
    pulseg_cursor_info where = PULSEG_CURSOR_INFO_INIT;
    pulseg_block_instance block = PULSEG_BLOCK_INSTANCE_INIT;
    long played = 0, limit = 20;
    int code, current_segment = -1;

    if (argc < 2 || argc > 3)
    {
        (void)fprintf(stderr, "usage: %s <scan.seq> [max_blocks]\n", argv[0]);
        return 2;
    }
    if (argc == 3)
        limit = atol(argv[2]);

    /* 1. Load the stream */
    code = pulseg_load_scanloop_cache(&scan, argv[1]);
    if (PULSEG_FAILED(code))
    {
        vendor_refuse(code, NULL);
        return 1;
    }

    /* 2. Walk it
     * The cursor rests before the first block, so it is advanced and then
     * read, never the other way round. pulseg_cursor_advance moves and
     * reports where it landed; PULSEG_CURSOR_DONE is the end of the scan. */
    while (played < limit)
    {
        code = pulseg_cursor_advance(scan, &where);
        if (code == PULSEG_CURSOR_DONE)
            break;
        if (PULSEG_FAILED(code))
        {
            vendor_refuse(code, NULL);
            pulseg_collection_free(scan);
            return 1;
        }

        /* A segment is the unit the hardware is pointed at, so the boundary
         * is where a real interpreter arms the next prepared segment. */
        if (where.segment_id != current_segment)
        {
            current_segment = where.segment_id;
            vendor_begin_segment(current_segment);
        }

        code = pulseg_get_block_instance(scan, &block);
        if (PULSEG_FAILED(code))
        {
            vendor_refuse(code, NULL);
            pulseg_collection_free(scan);
            return 1;
        }

        vendor_play_block(&block);
        ++played;
    }

    vendor_log("played %ld block(s)", played);

    /* pulseg_cursor_mark and pulseg_cursor_reset are the pair a prescan
     * uses: run a stretch of the stream, then return to where it began and
     * run the real acquisition from the same point. */

    pulseg_collection_free(scan);
    return 0;
}
