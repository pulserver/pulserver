// SPDX-License-Identifier: MIT
//
// Stage 4 (recon-cache-to-C) Step 0 golden-dump tool.
//
// Usage: dump_recon_cache <fixtures_dir> <output_dir>
//
// Loads every *.pge fixture in <fixtures_dir> (skipping MATLAB truth
// companions) through mrdserver::read_sequence_cache() and writes a
// deterministic text dump of every parsed field to
// <output_dir>/<stem>.dump. Run once against the pre-port reader to
// capture golden dumps (committed under tests/utils/expected_recon_dumps/),
// then again after the C port lands to diff for byte-exact equivalence.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "dump_recon_cache.h"
#include "trajectory_cache_reader.h"

namespace fs = std::filesystem;

namespace
{

    /* A handful of tests/utils/expected/*.pge files are not real trajectory
     * caches at all -- e.g. 03_grad_rss_violation.pge / 04_slew_rss_violation.pge
     * are stale pre-refactor artifacts kept only because their companion .seq
     * intentionally violates gradient/slew limits and cache regeneration for
     * them fails signature validation (see pge-fixture-regeneration notes).
     * They are consumed via their .seq by tests/ctests/test_safety_grad.c, not
     * via this .pge. Detect "no TRAJECTORY section" and skip rather than
     * baking a meaningless empty dump into the golden set. */
    bool has_trajectory_section(const fs::path &p)
    {
        std::ifstream f(p, std::ios::binary);
        if (!f)
            return false;
        int32_t marker = 0;
        if (!f.read(reinterpret_cast<char *>(&marker), 4))
            return false;
        bool do_swap = (marker != 0x01020304);
        auto swap4 = [](int32_t &v)
        {
            uint8_t *b = reinterpret_cast<uint8_t *>(&v);
            std::swap(b[0], b[3]);
            std::swap(b[1], b[2]);
        };
        auto read_i32 = [&](int32_t &v) -> bool
        {
            if (!f.read(reinterpret_cast<char *>(&v), 4))
                return false;
            if (do_swap)
                swap4(v);
            return true;
        };
        int32_t skip;
        if (!read_i32(skip) || !read_i32(skip) || !read_i32(skip) || !read_i32(skip) || !read_i32(skip))
            return false;
        int32_t num_sections = 0;
        if (!read_i32(num_sections) || num_sections <= 0 || num_sections > 64)
            return false;
        for (int i = 0; i < num_sections; ++i)
        {
            int32_t id, off, sz;
            if (!read_i32(id) || !read_i32(off) || !read_i32(sz))
                return false;
            if (id == 6)
                return true;
        }
        return false;
    }

    std::vector<fs::path> discover_fixtures(const fs::path &dir)
    {
        std::vector<fs::path> out;
        static const char *const truth_suffixes[] = {
            "_trajectory", "_tr_waveform", "_scan_table", "_segment_def",
            "_freqmod_def", "_freqmod_plan", "_label_state", "_seq_desc",
        };
        for (const auto &entry : fs::directory_iterator(dir))
        {
            if (!entry.is_regular_file())
                continue;
            const auto &p = entry.path();
            if (p.extension() != ".pge" && p.extension() != ".pseg")
                continue;
            const std::string stem = p.stem().string();
            bool skip = false;
            for (const char *suf : truth_suffixes)
            {
                const size_t sl = std::string(suf).size();
                if (stem.size() >= sl && stem.compare(stem.size() - sl, std::string::npos, suf) == 0)
                {
                    skip = true;
                    break;
                }
            }
            if (!skip && !has_trajectory_section(p))
            {
                std::cout << "skipping " << p
                          << " (no TRAJECTORY section -- not a recon cache)\n";
                skip = true;
            }
            if (!skip)
                out.push_back(p);
        }
        std::sort(out.begin(), out.end());
        return out;
    }

} // namespace

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "usage: dump_recon_cache <fixtures_dir> <output_dir>\n";
        return 2;
    }

    const fs::path fixtures_dir(argv[1]);
    const fs::path output_dir(argv[2]);

    if (!fs::is_directory(fixtures_dir))
    {
        std::cerr << "not a directory: " << fixtures_dir << "\n";
        return 2;
    }
    fs::create_directories(output_dir);

    const auto fixtures = discover_fixtures(fixtures_dir);
    if (fixtures.empty())
    {
        std::cerr << "no .pge/.pseg fixtures found under " << fixtures_dir << "\n";
        return 2;
    }

    int failures = 0;
    for (const auto &fixture : fixtures)
    {
        const std::string stem = fixture.stem().string();
        const fs::path out_path = output_dir / (stem + ".dump");

        mrdserver::SequenceCache cache;
        try
        {
            cache = mrdserver::read_sequence_cache(fixture.string());
        }
        catch (const std::exception &e)
        {
            std::cerr << "FAILED to read " << fixture << ": " << e.what() << "\n";
            ++failures;
            continue;
        }

        std::ofstream out(out_path);
        if (!out)
        {
            std::cerr << "FAILED to open " << out_path << " for writing\n";
            ++failures;
            continue;
        }
        dump_recon_cache::dump_sequence_cache(out, cache);
        std::cout << "wrote " << out_path << "\n";
    }

    if (failures)
    {
        std::cerr << failures << " fixture(s) failed to dump\n";
        return 1;
    }
    return 0;
}
