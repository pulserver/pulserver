/**
 * @file sequence_file.hpp
 * @brief Lazy, section-level access to a Pulseq file, text or binary.
 *
 * A consumer that needs one section should not pay for the whole file. The
 * dominant case is the recon side reading [DEFINITIONS] alone -- the
 * NextSequence chain link, the TRSize gate, FOV/Matrix geometry -- before
 * deciding whether to parse anything else. `SequenceFile` gives that
 * consumer:
 *
 *   - format detection (binary magic vs text) on open;
 *   - a section index built on first demand: byte offsets of `[NAME]`
 *     headers for text, a section-head walk for binary (every section's
 *     payload length is computable from its head: fixed row strides, or an
 *     entry walk for the variable ones);
 *   - `definitions()`, parsed from that one section without touching the
 *     rest of the file;
 *   - `sequence()`, the ordinary full parse, cached, for consumers that end
 *     up needing everything after all.
 *
 * NextSequence chains stay lazy at this level too: `next_sequence()` hands
 * back the link as written; the caller opens the next `SequenceFile` when
 * (and if) it walks the chain.
 */

#ifndef PULSEQ_CXX_SEQUENCE_FILE_HPP
#define PULSEQ_CXX_SEQUENCE_FILE_HPP

#include "pulseq/sequence.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace pulseq
{

    class SequenceFile
    {
    public:
        /** Opens and reads the raw bytes; indexes nothing yet.
         *
         * Parameters: path of a `.seq` file in either format.
         * Raises: std::runtime_error when the file cannot be read or is
         * neither a Pulseq text nor binary file.
         */
        explicit SequenceFile(const std::string& path);

        const std::string& path() const
        {
            return path_;
        }
        bool is_binary() const
        {
            return binary_;
        }

        /** Section names present in the file, in file order.
         *
         * Text names ("VERSION", "DEFINITIONS", "BLOCKS", ...); binary
         * section codes are reported under the same names.
         */
        const std::vector<std::string>& section_names();

        bool has_section(const std::string& name);

        /** The [DEFINITIONS] content alone, parsed on first call.
         *
         * Binary files stop at the definitions section's end; text files
         * parse only the indexed section span. Empty map when the file has
         * no definitions.
         */
        const std::map<std::string, Definition>& definitions();

        /** The NextSequence chain link, or "" when the file is the last. */
        std::string next_sequence();

        /** The whole file through the ordinary reader, parsed once. */
        Sequence& sequence();

    private:
        struct Span
        {
            std::string name;
            size_t begin; /**< payload start (after the header/code) */
            size_t end;   /**< one past the payload's last byte */
        };

        void build_index();
        void index_text();
        void index_binary();
        void parse_definitions();
        void parse_definitions_text(const Span& span);
        void parse_definitions_binary(const Span& span);

        uint64_t get_u64(size_t offset) const;
        uint32_t get_u32(size_t offset) const;

        std::string path_;
        std::string bytes_;
        bool binary_ = false;
        bool swapped_ = false; /**< binary written in the other byte order */

        bool indexed_ = false;
        std::vector<Span> spans_;
        std::vector<std::string> names_;

        bool definitions_parsed_ = false;
        std::map<std::string, Definition> definitions_;

        std::unique_ptr<Sequence> sequence_;
    };

} // namespace pulseq

#endif /* PULSEQ_CXX_SEQUENCE_FILE_HPP */
