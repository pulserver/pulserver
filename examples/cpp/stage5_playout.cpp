/*
 * stage5_playout.cpp -- walk the execution stream and play it.
 *
 * C++ counterpart of examples/c/stage5_playout.c.
 *
 *   stage5_playout <scan.seq> [max_blocks]
 */

#include "vendor.hpp"

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3)
    {
        std::fprintf(stderr, "usage: %s <scan.seq> [max_blocks]\n", argv[0]);
        return 2;
    }
    const long limit = (argc == 3) ? std::atol(argv[2]) : 20;

    try
    {
        /* 1. Load the stream. */
        pulseg::Collection scan =
            pulseg::Collection::from_scanloop_cache(argv[1], vendor::system_limits());

        /* 2. Walk it. The cursor rests before the first block, so it is
         * advanced and then read. cursor_advance reports where it landed;
         * PULSEG_CURSOR_DONE is the end of the scan. */
        pulseg_cursor_info where = PULSEG_CURSOR_INFO_INIT;
        long played = 0;
        int current_segment = -1;

        while (played < limit && scan.cursor_advance(where) != PULSEG_CURSOR_DONE)
        {
            /* A segment is the unit the hardware is pointed at. */
            if (where.segment_id != current_segment)
            {
                current_segment = where.segment_id;
                vendor::begin_segment(current_segment);
            }

            vendor::play_block(scan.get_block_instance());
            ++played;
        }

        std::printf("played %ld block(s)\n", played);

        /* cursor_mark and cursor_reset are the pair a prescan uses: run a
         * stretch of the stream, then acquire from the same point. */
    }
    catch (const pulseg::Error& error)
    {
        std::fprintf(stderr, "refused: %s\n", error.what());
        return 1;
    }
    return 0;
}
