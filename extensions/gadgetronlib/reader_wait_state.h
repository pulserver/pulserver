#pragma once

#include <condition_variable>
#include <chrono>
#include <exception>
#include <mutex>

class ReaderWaitState {
public:
    ReaderWaitState();

    void reset();
    void mark_done(std::exception_ptr ex = nullptr);
    bool wait_for_done(unsigned int timeout_ms);
    std::exception_ptr consume_exception();

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool done_;
    std::exception_ptr exception_;
};
