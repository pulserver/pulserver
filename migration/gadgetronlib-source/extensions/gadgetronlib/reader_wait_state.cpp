#include "reader_wait_state.h"

ReaderWaitState::ReaderWaitState()
    : done_(false)
    , exception_(nullptr)
{
}

void ReaderWaitState::reset()
{
    std::lock_guard<std::mutex> lock(mutex_);
    done_ = false;
    exception_ = nullptr;
}

void ReaderWaitState::mark_done(std::exception_ptr ex)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!exception_) {
            exception_ = ex;
        }
        done_ = true;
    }
    cv_.notify_all();
}

bool ReaderWaitState::wait_for_done(unsigned int timeout_ms)
{
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this]() { return done_; });
}

std::exception_ptr ReaderWaitState::consume_exception()
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::exception_ptr out = exception_;
    exception_ = nullptr;
    return out;
}
