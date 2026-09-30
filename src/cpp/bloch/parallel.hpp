/**
 * @file parallel.hpp
 * @brief Work over a range of items split between threads.
 */

#ifndef PULSERVER_BLOCH_PARALLEL_HPP
#define PULSERVER_BLOCH_PARALLEL_HPP

#include <algorithm>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

namespace bloch
{

    /** How many workers parallel() runs for @p count items. */
    inline size_t workers_for(size_t count, size_t threads, size_t least)
    {
        return std::max<size_t>(1, std::min(threads, (count + least - 1) / least));
    }

    /**
     * Run body(worker, first, last) over [0, count) on up to @p threads
     * threads, each given at least @p least items. The body must not throw.
     */
    inline void parallel(
        size_t count, size_t threads, size_t least, const std::function<void(size_t, size_t, size_t)>& body)
    {
        if (count == 0)
            return;
        const size_t workers = workers_for(count, threads, least);
        if (workers == 1)
        {
            body(0, 0, count);
            return;
        }
        const size_t chunk = (count + workers - 1) / workers;
        std::vector<std::thread> pool;
        pool.reserve(workers - 1);
        for (size_t worker = 1; worker < workers && worker * chunk < count; ++worker)
            pool.emplace_back(body, worker, worker * chunk, std::min(count, (worker + 1) * chunk));
        body(0, 0, std::min(count, chunk));
        for (std::thread& thread : pool)
            thread.join();
    }

} // namespace bloch

#endif /* PULSERVER_BLOCH_PARALLEL_HPP */
