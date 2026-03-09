#include <cassert>
#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>

#include "reader_wait_state.h"

int main()
{
    ReaderWaitState state;

    // Verify timeout behavior when no completion signal arrives.
    bool completed = state.wait_for_done(10);
    assert(!completed);

    state.reset();
    std::thread done_thread([&state]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        state.mark_done();
    });
    completed = state.wait_for_done(200);
    done_thread.join();
    assert(completed);
    assert(state.consume_exception() == nullptr);

    state.reset();
    std::thread exception_thread([&state]() {
        try {
            throw std::runtime_error("reader failure");
        } catch (...) {
            state.mark_done(std::current_exception());
        }
    });
    completed = state.wait_for_done(200);
    exception_thread.join();
    assert(completed);

    std::exception_ptr ex = state.consume_exception();
    assert(ex != nullptr);

    bool saw_expected_exception = false;
    try {
        std::rethrow_exception(ex);
    } catch (const std::runtime_error& err) {
        saw_expected_exception = (std::string(err.what()) == "reader failure");
    }
    assert(saw_expected_exception);
    assert(state.consume_exception() == nullptr);

    return 0;
}
