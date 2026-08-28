/*
 * stage3_preparation.cpp -- Preparation: read, structure, gate, cache.
 *
 * C++ counterpart of examples/c/stage3_preparation.c.
 *
 *   stage3_preparation <scan.seq>
 */

#include "vendor.hpp"

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::fprintf(stderr, "usage: %s <scan.seq>\n", argv[0]);
        return 2;
    }

    try
    {
        /* 1. Read and structure. Follows the NextSequence chain,
         * deduplicates the unique blocks, detects the TR and the segments
         * from block content, and expands the execution stream. */
        pulseg::Collection scan(argv[1], vendor::system_limits());

        const pulseg_collection_info info = scan.collection_info();
        const pulseg::ScanTimeInfo timing = scan.get_scan_time();
        std::printf(
            "read: %d subsequence(s), %d segments, %.1f s\n",
            info.num_subsequences,
            info.num_segments,
            timing.total_duration_us / 1e6);

        /* 2. Gate it. One plan across every question asked below; omitting
         * it gives identical verdicts and costs a little more.
         *
         * No PNS model is passed (nullptr skips that check): the model is
         * caller-injected because its coefficients are vendor property. */
        pulseg::CheckPlan plan(scan.handle());
        scan.check_safety(vendor::forbidden_bands(), nullptr, 80.0f, &plan);
        vendor::log("checks: passed");

        /* Each check is also callable on its own, for a platform that gates
         * some of them in hardware and wants only the rest. */
        scan.check_max_slew();
        vendor::log("slew alone: within limits");

        /* 3. Cache it. The path comes from the sequence path and
         * opts.cache_ext, so it pairs with the section loaders. */
        scan.save_cache(argv[1]);
        vendor::log("cache: written beside the sequence");
    }
    catch (const pulseg::Error& error)
    {
        std::fprintf(stderr, "refused: %s\n", error.what());
        return 1;
    }
    return 0;
}
