/**
 * @file error.hpp
 * @brief C++ exception types for pulseg error codes.
 */

#ifndef PULSEG_ERROR_HPP
#define PULSEG_ERROR_HPP

#include <stdexcept>
#include <string>

#include "pulseg.h"
#include "pulseg_types.h"

namespace pulseg
{

/**
 * Base exception for all pulseg errors.
 * Carries the original error code and an optional diagnostic message.
 */
class Error : public std::runtime_error
{
  public:
    explicit Error(int code) : std::runtime_error(build_message(code, nullptr)), code_(code)
    {
    }

    Error(int code, const pulseg_diagnostic &diag)
        : std::runtime_error(build_message(code, &diag)), code_(code)
    {
    }

    int code() const noexcept
    {
        return code_;
    }

  private:
    int code_;

    static std::string build_message(int code, const pulseg_diagnostic *diag)
    {
        char buf[512];
        int n = pulseg_format_error(buf, sizeof(buf), code, diag);
        if (n > 0)
            return std::string(buf, n);
        const char *msg = pulseg_get_error_message(code);
        return msg ? std::string(msg) : "Unknown pulseg error";
    }
};

/**
 * Throw pulseg::Error if code indicates failure.
 */
inline void check(int code)
{
    if (PULSEG_FAILED(code))
        throw Error(code);
}

/**
 * Throw pulseg::Error with diagnostic if code indicates failure.
 */
inline void check(int code, const pulseg_diagnostic &diag)
{
    if (PULSEG_FAILED(code))
        throw Error(code, diag);
}

} // namespace pulseg

#endif // PULSEG_ERROR_HPP
