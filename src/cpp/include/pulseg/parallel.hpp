#pragma once
/**
 * @file parallel.hpp
 * @brief The host's implementation of pulseg_opts.parallel_for_fn.
 *
 * The C safety engine takes an optional loop runner so a host with cores to
 * spare can spread its independent per-element work; this is that runner,
 * on std::thread. Ranges are dealt in small chunks from an atomic counter,
 * so uneven elements balance, and a short loop runs inline.
 */
#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

namespace pulseg
{
    inline void parallel_for_threads(
        void* /*ctx*/,
        int count,
        void (*body)(void* arg, int begin, int end),
        void* arg)
    {
        unsigned workers = std::thread::hardware_concurrency();
        workers = std::min<unsigned>(std::max<unsigned>(workers, 1), 8);
        // Deals about eight ranges per worker: a body that sets up per range
        // (an FFT plan, a scratch buffer) amortises it over count / 64 items
        // instead of eight, and the tail still balances across the pool.
        const int kChunk = std::max(8, count / (static_cast<int>(workers) * 8));
        if (workers == 1 || count < 16)
        {
            if (count > 0)
                body(arg, 0, count);
            return;
        }
        std::atomic<int> next{0};
        const auto run = [&]()
        {
            for (;;)
            {
                const int begin = next.fetch_add(kChunk);
                if (begin >= count)
                    return;
                body(arg, begin, std::min(begin + kChunk, count));
            }
        };
        std::vector<std::thread> pool;
        for (unsigned w = 1; w < workers; ++w)
            pool.emplace_back(run);
        run();
        for (auto& t : pool)
            t.join();
    }
} // namespace pulseg
