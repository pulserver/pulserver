/**
 * @file error.hpp
 * @brief C++ exception type for pulseq error codes.
 */

#ifndef PULSEQ_CXX_ERROR_HPP
#define PULSEQ_CXX_ERROR_HPP

#include <stdexcept>
#include <string>

#include "pulseq_config.h"

namespace pulseq
{

    /**
 * @brief Human-readable message for a PULSEQ_ERR_* code.
 *
 * pulseq (src/c/include/pulseq) has no message table of its own -- unlike
 * pulseg, it never surfaces error strings to a UI -- so this wrapper owns
 * one, matching the codes declared in pulseq_config.h.
 */
    inline const char* error_message(int code)
    {
        switch (code)
        {
        case PULSEQ_SUCCESS:
            return "success";
        case PULSEQ_ERR_NULL_POINTER:
            return "null pointer argument";
        case PULSEQ_ERR_INVALID_ARGUMENT:
            return "invalid argument";
        case PULSEQ_ERR_ALLOC_FAILED:
            return "allocation failed";
        case PULSEQ_ERR_FILE_NOT_FOUND:
            return "file not found";
        case PULSEQ_ERR_FILE_READ_FAILED:
            return "file read failed";
        case PULSEQ_ERR_UNSUPPORTED_VERSION:
            return "unsupported .seq version";
        case PULSEQ_ERR_SIGNATURE_MISMATCH:
            return "signature mismatch";
        case PULSEQ_ERR_SIGNATURE_MISSING:
            return "signature missing";
        case PULSEQ_ERR_EMPTY:
            return "empty file set";
        case PULSEQ_ERR_CHAIN_BROKEN:
            return "\"next sequence\" chain broken";
        default:
            return "unknown pulseq error";
        }
    }

    /** Base exception for all pulseq errors. Carries the original error code. */
    class Error : public std::runtime_error
    {
    public:
        explicit Error(int code)
            : std::runtime_error(
                  std::string(error_message(code)) + " (code " + std::to_string(code) + ")"),
              code_(code)
        {
        }

        int code() const noexcept
        {
            return code_;
        }

    private:
        int code_;
    };

    /** Throw pulseq::Error if code indicates failure. */
    inline void check(int code)
    {
        if (PULSEQ_FAILED(code))
            throw Error(code);
    }

} // namespace pulseq

#endif // PULSEQ_CXX_ERROR_HPP
