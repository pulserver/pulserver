/**
 * @file read.hpp
 * @brief Reading a Pulseq file into a pulseq::Sequence.
 */

#ifndef PULSEQ_CXX_READ_HPP
#define PULSEQ_CXX_READ_HPP

#include <string>

namespace pulseq
{

    class Sequence;

    /**
     * Read a Pulseq sequence file.
     *
     * Text or binary -- the format is detected from the file's leading bytes,
     * not its name, so `.seq`, `.bin` and anything else all work.  Values come
     * back in SI units and at full double precision.
     *
     * @throws std::runtime_error if the file cannot be read or parsed.
     */
    Sequence read_file(const std::string& path);

} // namespace pulseq

#endif /* PULSEQ_CXX_READ_HPP */
