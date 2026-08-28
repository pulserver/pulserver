/**
 * @file file.hpp
 * @brief Peeking a .seq, and building a collection from parsed files.
 *
 * The entry points that take a path or parsed files rather than a collection,
 * so they are free functions rather than members. See pulseg_io.h and
 * pulseg_convert.h.
 */

#ifndef PULSEG_FILE_HPP
#define PULSEG_FILE_HPP

#include <string>
#include <vector>

#include "pulseg.h"

#include "error.hpp"
#include "types.hpp"

namespace pulseg
{
    /**
     * Scan time from the [DEFINITIONS] section alone.
     *
     * No parse and no structure. The duration is an approximation: dead time
     * between segments is not accounted for, and total_segment_boundaries is
     * left at 0.
     */
    inline ScanTimeInfo peek_scan_time(const std::string& seq_path, const Opts& opts)
    {
        pulseg_scan_time_info info = PULSEG_SCAN_TIME_INFO_INIT;
        const pulseg_opts copts = opts.to_c();
        check(pulseg_peek_scan_time(&info, seq_path.c_str(), &copts));
        return ScanTimeInfo::from_c(info);
    }

    /**
     * The collection-level flags a scan declares about itself.
     *
     * Reads the [DEFINITIONS] section of the head of the chain only: these
     * describe the whole scan, so the rest of the chain is not consulted.
     */
    inline pulseg_sequence_flags peek_sequence_flags(const std::string& seq_path, const Opts& opts)
    {
        pulseg_sequence_flags flags = PULSEG_SEQUENCE_FLAGS_INIT;
        const pulseg_opts copts = opts.to_c();
        check(pulseg_peek_sequence_flags(&flags, seq_path.c_str(), &copts));
        return flags;
    }

    /** Delete the cache file beside a sequence. Absent is not an error. */
    inline void clear_cache(const std::string& seq_path)
    {
        check(pulseg_clear_cache(seq_path.c_str()));
    }
} // namespace pulseg

#endif // PULSEG_FILE_HPP
