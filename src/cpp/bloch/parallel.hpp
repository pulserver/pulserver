/**
 * @file parallel.hpp
 * @brief Work over a range of items split between threads that wait from
 *        one call to the next.
 */

#ifndef PULSERVER_BLOCH_PARALLEL_HPP
#define PULSERVER_BLOCH_PARALLEL_HPP

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace bloch
{

    /** How many workers parallel() runs for @p count items. */
    inline size_t workers_for(size_t count, size_t threads, size_t least)
    {
        return std::max<size_t>(1, std::min(threads, (count + least - 1) / least));
    }

    namespace detail
    {

        using Body = std::function<void(size_t, size_t, size_t)>;

        inline long this_process()
        {
#if defined(_WIN32)
            return static_cast<long>(_getpid());
#else
            return static_cast<long>(getpid());
#endif
        }

        /** Whether this thread runs a part of a pool's call, where a call of
         *  its own would wait on the pool it runs in. */
        inline thread_local bool in_pool = false;

        /** Checks of a condition before its thread sleeps on it: calls come
         *  in quick succession, and a wake-up costs more than a wait of a few
         *  tens of microseconds. */
        constexpr int kSpins = 1 << 11;

        inline void relax()
        {
#if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
#elif defined(__aarch64__)
            __asm__ __volatile__("yield");
#else
            std::this_thread::yield();
#endif
        }

        /** Whether @p ready() holds within kSpins checks. */
        template <typename Ready>
        bool spin(Ready ready)
        {
            for (int n = 0; n < kSpins; ++n)
            {
                if (ready())
                    return true;
                relax();
            }
            return ready();
        }

        /**
         * Threads that run the parts of one parallel() call at a time and wait
         * between calls, so that a call costs their waking rather than their
         * creation. A child of fork() has none of its parent's threads, and
         * makes a pool of its own.
         */
        class Pool
        {
        public:
            static Pool& get()
            {
                static std::atomic<Pool*> pool{nullptr};
                const long process = this_process();
                Pool* held = pool.load(std::memory_order_acquire);
                if (held != nullptr && held->process_ == process)
                    return *held;
                /* A parent's pool is left as it is: its threads, and whatever
                 * held its locks, are not in this process. */
                Pool* made = new Pool(process);
                if (pool.compare_exchange_strong(held, made, std::memory_order_acq_rel))
                    return *made;
                delete made;
                return *held;
            }

            /** Run body(worker, first, last) over parts of @p chunk items of
             *  [0, count), part 0 on the calling thread; false, running
             *  nothing, where another call holds the pool or the caller runs
             *  in one. */
            bool run(size_t count, size_t chunk, const Body& body)
            {
                if (in_pool)
                    return false;
                std::unique_lock<std::mutex> held(held_, std::try_to_lock);
                if (!held.owns_lock())
                    return false;
                const size_t workers = (count + chunk - 1) / chunk;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    while (threads_.size() + 1 < workers)
                    {
                        const size_t index = threads_.size() + 1;
                        const size_t round = round_.load(std::memory_order_relaxed);
                        threads_.emplace_back([this, index, round] { serve(index, round); });
                        threads_.back().detach();
                    }
                    body_ = &body;
                    workers_ = workers;
                    count_ = count;
                    chunk_ = chunk;
                    /* Every thread answers every call, those without a part
                     * too, so that none reads a call's fields as the next
                     * call's are set. */
                    pending_.store(threads_.size(), std::memory_order_relaxed);
                    round_.fetch_add(1, std::memory_order_release);
                }
                start_.notify_all();
                in_pool = true;
                body(0, 0, std::min(count, chunk));
                in_pool = false;
                const auto done = [this] { return pending_.load(std::memory_order_acquire) == 0; };
                if (!spin(done))
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    finish_.wait(lock, done);
                }
                return true;
            }

        private:
            explicit Pool(long process) : process_(process)
            {
            }

            /** Thread @p index's loop: the part of each call from the one
             *  after @p round on that it is given. */
            void serve(size_t index, size_t round)
            {
                in_pool = true;
                for (;;)
                {
                    const auto called = [this, &round] { return round_.load(std::memory_order_acquire) != round; };
                    if (!spin(called))
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        start_.wait(lock, called);
                    }
                    round = round_.load(std::memory_order_acquire);
                    if (index < workers_)
                    {
                        const size_t first = index * chunk_;
                        (*body_)(index, first, std::min(count_, first + chunk_));
                    }
                    if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1)
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        finish_.notify_one();
                    }
                }
            }

            const long process_;
            /** Held by the call the pool runs. */
            std::mutex held_;
            std::mutex mutex_;
            std::condition_variable start_;
            std::condition_variable finish_;
            std::vector<std::thread> threads_;
            /** The call running, set before round_ is advanced and read by
             *  each thread before it lowers pending_. */
            const Body* body_ = nullptr;
            size_t workers_ = 0;
            size_t count_ = 0;
            size_t chunk_ = 0;
            std::atomic<size_t> round_{0};
            std::atomic<size_t> pending_{0};
        };

    } // namespace detail

    /**
     * Run body(worker, first, last) over [0, count) on up to @p threads
     * threads, each given at least @p least items. The body must not throw.
     * A call made while another runs, or from within a body, runs on threads
     * of its own.
     */
    inline void parallel(size_t count, size_t threads, size_t least, const detail::Body& body)
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
        if (detail::Pool::get().run(count, chunk, body))
            return;
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
