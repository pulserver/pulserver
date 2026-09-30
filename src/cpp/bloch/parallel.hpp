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
#include <memory>
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
         *  tens of microseconds. A thread that waits gives way to others at
         *  every kYieldEvery checks, and sleeps at once after a call of more
         *  parts than the processor has cores. */
        constexpr int kSpins = 1 << 11;
        constexpr int kYieldEvery = 1 << 7;

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

        /** Whether @p ready() holds within @p spins checks. */
        template <typename Ready>
        bool spin(Ready ready, int spins)
        {
            for (int n = 1; n <= spins; ++n)
            {
                if (ready())
                    return true;
                if (n % kYieldEvery == 0)
                    std::this_thread::yield();
                else
                    relax();
            }
            return ready();
        }

        /**
         * Threads that run the parts of one parallel() call at a time and wait
         * between calls, so that a call costs their waking rather than their
         * creation. Each thread is given its part in a place of its own, and
         * only the threads a call gives parts to are woken. A child of fork()
         * has none of its parent's threads, and makes a pool of its own.
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
                const size_t parts = (count + chunk - 1) / chunk;
                while (threads_.size() + 1 < parts)
                {
                    /* A thread that fails to start leaves no place behind for
                     * a call to wait on. */
                    threads_.reserve(threads_.size() + 1);
                    auto made = std::make_unique<Thread>();
                    Thread* thread = made.get();
                    const size_t index = threads_.size() + 1;
                    std::thread([this, thread, index] { serve(*thread, index); }).detach();
                    threads_.push_back(std::move(made));
                }
                const int spins = cores_ == 0 || parts <= cores_ ? kSpins : 0;
                pending_.store(parts - 1, std::memory_order_relaxed);
                for (size_t index = 1; index < parts; ++index)
                {
                    Thread& thread = *threads_[index - 1];
                    thread.body = &body;
                    thread.first = index * chunk;
                    thread.last = std::min(count, thread.first + chunk);
                    thread.spins = spins;
                    {
                        std::lock_guard<std::mutex> lock(thread.mutex);
                        thread.calls.fetch_add(1, std::memory_order_release);
                    }
                    thread.wake.notify_one();
                }
                in_pool = true;
                body(0, 0, std::min(count, chunk));
                in_pool = false;
                const auto done = [this] { return pending_.load(std::memory_order_acquire) == 0; };
                if (!spin(done, spins))
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    finish_.wait(lock, done);
                }
                return true;
            }

        private:
            /** A pool thread, and the part it is given, with how long it
             *  waits for its next once done: set before calls is advanced,
             *  and read by the thread before it lowers pending_. */
            struct Thread
            {
                std::mutex mutex;
                std::condition_variable wake;
                std::atomic<size_t> calls{0};
                const Body* body = nullptr;
                size_t first = 0;
                size_t last = 0;
                int spins = 0;
            };

            explicit Pool(long process) : process_(process), cores_(std::thread::hardware_concurrency())
            {
            }

            /** Pool thread @p index's loop: each part it is given. */
            void serve(Thread& thread, size_t index)
            {
                in_pool = true;
                size_t served = 0;
                int spins = 0;
                for (;;)
                {
                    const auto given = [&thread, &served] {
                        return thread.calls.load(std::memory_order_acquire) != served;
                    };
                    if (!spin(given, spins))
                    {
                        std::unique_lock<std::mutex> lock(thread.mutex);
                        thread.wake.wait(lock, given);
                    }
                    ++served;
                    spins = thread.spins;
                    (*thread.body)(index, thread.first, thread.last);
                    if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1)
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        finish_.notify_one();
                    }
                }
            }

            const long process_;
            /** Cores of the processor; none where they are not known. */
            const size_t cores_;
            /** Held by the call the pool runs. */
            std::mutex held_;
            std::mutex mutex_;
            std::condition_variable finish_;
            std::vector<std::unique_ptr<Thread>> threads_;
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
