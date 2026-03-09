/**
 * @file error.hpp
 * @brief C++ exception types for pulseqlib error codes.
 */

#ifndef PULSEQLIB_ERROR_HPP
#define PULSEQLIB_ERROR_HPP

#include <stdexcept>
#include <string>

#include "pulseqlib_methods.h"
#include "pulseqlib_types.h"

namespace pulseqlib {

/**
 * Base exception for all pulseqlib errors.
 * Carries the original error code and an optional diagnostic message.
 */
class Error : public std::runtime_error {
public:
    explicit Error(int code)
        : std::runtime_error(build_message(code, nullptr))
        , code_(code) {}

    Error(int code, const pulseqlib_diagnostic& diag)
        : std::runtime_error(build_message(code, &diag))
        , code_(code) {}

    int code() const noexcept { return code_; }

private:
    int code_;

    static std::string build_message(int code, const pulseqlib_diagnostic* diag) {
        char buf[512];
        int n = pulseqlib_format_error(buf, sizeof(buf), code, diag);
        if (n > 0) return std::string(buf, n);
        const char* msg = pulseqlib_get_error_message(code);
        return msg ? std::string(msg) : "Unknown pulseqlib error";
    }
};

/**
 * Throw pulseqlib::Error if code indicates failure.
 */
inline void check(int code) {
    if (PULSEQLIB_FAILED(code))
        throw Error(code);
}

/**
 * Throw pulseqlib::Error with diagnostic if code indicates failure.
 */
inline void check(int code, const pulseqlib_diagnostic& diag) {
    if (PULSEQLIB_FAILED(code))
        throw Error(code, diag);
}

} // namespace pulseqlib

#endif // PULSEQLIB_ERROR_HPP
