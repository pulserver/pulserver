/**
 * @file write.hpp
 * @brief Serializing a pulseq::Sequence, as text and as Pulseq's binary format.
 *
 * Both writers take the sequence by non-const reference because both record
 * `TotalDuration` in `[DEFINITIONS]` before they start, which is what PyPulseq
 * does and what makes the two formats agree on what they contain.
 */

#ifndef PULSEQ_CXX_WRITE_HPP
#define PULSEQ_CXX_WRITE_HPP

#include <string>

namespace pulseq
{

    class Sequence;

    /**
     * The Pulseq revision @p seq actually needs.
     *
     * Rotations and RF shims are 1.5.1; a label outside Pulseq's built-in set
     * is 1.5.2.  A sequence using none of them stays at whatever revision it
     * declares, so the ordinary case writes a file any interpreter reads.
     */
    int required_revision(const Sequence& seq);

    /**
     * Serialize as a Pulseq `.seq` text file.
     *
     * @param create_signature  Append the `[SIGNATURE]` section: an MD5 of
     *                          everything above it, including the newline that
     *                          precedes the header.
     * @throws std::runtime_error if a block duration is not a whole number of
     *         block-duration rasters.
     */
    std::string write_text(Sequence& seq, bool create_signature);

    /**
     * Serialize in Pulseq's native binary format (upstream's `writeBinary.m`).
     *
     * The same sequence, in a form the C reader parses in about a tenth of the
     * time, and at better precision: amplitudes go out as float64 and times as
     * integer picoseconds rather than as printed decimals.  There is no
     * signature section -- the format has no equivalent of `[SIGNATURE]`.
     */
    /** The binary file. With @p structure_only the gradient shapes of blocks
     *  that carry no RF are written without their samples: enough for a
     *  reader to recover the scan's structure, not to play or check it. */
    std::string write_binary(Sequence& seq, bool structure_only = false);

    /** The size the binary file will have, so a caller can fill its own
     *  buffer with write_binary_into and skip a copy. */
    size_t write_binary_size(Sequence& seq, bool structure_only = false);
    /** Write the binary file into @p buffer of exactly write_binary_size
     *  bytes. Throws if the two disagree. */
    void write_binary_into(Sequence& seq, bool structure_only, char* buffer, size_t size);

} // namespace pulseq

#endif /* PULSEQ_CXX_WRITE_HPP */
