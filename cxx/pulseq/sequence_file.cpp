/**
 * @file sequence_file.cpp
 * @brief Lazy section index and definitions-only parsing for Pulseq files.
 *
 * Binary section payload lengths are computed from each section's head, per
 * the layouts write_binary.cpp emits: a fixed row stride for the fixed-width
 * sections, an entry walk for the variable ones (DEFINITIONS, SHAPES,
 * SOFTDELAYS, RFSHIMS). An unknown section code fails the index loudly --
 * a guessed length would corrupt every later offset.
 */

#include "pulseq/sequence_file.hpp"

#include "pulseq/read.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace pulseq
{

    namespace
    {

        const char FILE_HEADER[8] = {1, 'p', 'u', 'l', 's', 'e', 'q', 2};

        /** Binary section code (low word) -> the text format's section name. */
        const char* binary_section_name(uint32_t code)
        {
            switch (code)
            {
                case 1: return "DEFINITIONS";
                case 2: return "BLOCKS";
                case 3: return "RF";
                case 4: return "GRADIENTS";
                case 5: return "TRAP";
                case 6: return "ADC";
                case 8: return "SHAPES";
                case 9: return "EXTENSIONS";
                case 10: return "TRIGGERS";
                case 11: return "LABELSET";
                case 12: return "LABELINC";
                case 13: return "SOFTDELAYS";
                case 14: return "RFSHIMS";
                case 15: return "ROTATIONS";
                case 16: return "LABELNAMES";
                default: return nullptr;
            }
        }

        /** Fixed per-row byte stride of a binary section, 0 when variable. */
        size_t binary_row_stride(uint32_t code)
        {
            switch (code)
            {
                case 2: return 32;  /* BLOCKS: i64 duration + 6 x i32     */
                case 3: return 73;  /* RF: id + f64 + 3 i32 + 2 i64 + 4 f64 + use */
                case 4: return 44;  /* GRADIENTS: id + 3 f64 + 2 i32 + i64 */
                case 5: return 44;  /* TRAP: id + f64 + 4 i64             */
                case 6: return 64;  /* ADC: id + 3 i64 + 4 f64 + i32      */
                case 9: return 16;  /* EXTENSIONS: 4 x i32                 */
                case 10: return 28; /* TRIGGERS: 3 i32 + 2 i64             */
                case 11: return 12; /* LABELSET: 3 i32                     */
                case 12: return 12; /* LABELINC: 3 i32                     */
                case 15: return 36; /* ROTATIONS: i32 + quaternion (4 f64) */
                default: return 0;
            }
        }

        /** Extension-spec sections open with the registered type id. */
        bool binary_has_type_id(uint32_t code) { return code >= 10 && code <= 15; }

        bool parse_full_double(const std::string& token, double* value)
        {
            char* end = nullptr;
            *value = std::strtod(token.c_str(), &end);
            return end != nullptr && *end == '\0' && end != token.c_str();
        }

        bool parse_full_long(const std::string& token)
        {
            char* end = nullptr;
            (void)std::strtol(token.c_str(), &end, 10);
            return end != nullptr && *end == '\0' && end != token.c_str();
        }

    }  // namespace

    SequenceFile::SequenceFile(const std::string& path) : path_(path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
            throw std::runtime_error("could not open Pulseq file '" + path + "'");
        std::ostringstream buffer;
        buffer << in.rdbuf();
        bytes_ = buffer.str();

        binary_ = bytes_.size() >= 8 && std::memcmp(bytes_.data(), FILE_HEADER, 8) == 0;
        if (binary_)
        {
            if (bytes_.size() < 8 + 3 * 8)
                throw std::runtime_error("binary Pulseq file '" + path + "' is truncated");
            /* The version fields reveal the byte order: a plausible major
             * version is a small positive number in exactly one of the two
             * readings. */
            uint64_t major = 0;
            for (int i = 0; i < 8; ++i)
                major |= static_cast<uint64_t>(static_cast<unsigned char>(bytes_[8 + i]))
                         << (8 * i);
            swapped_ = !(major > 0 && major < 1000);
        }
        else if (bytes_.find('[') == std::string::npos)
        {
            throw std::runtime_error("'" + path + "' is neither a Pulseq text nor binary file");
        }
    }

    uint64_t SequenceFile::get_u64(size_t offset) const
    {
        uint64_t value = 0;
        for (int i = 0; i < 8; ++i)
        {
            const int shift = swapped_ ? 8 * (7 - i) : 8 * i;
            value |= static_cast<uint64_t>(static_cast<unsigned char>(bytes_[offset + i]))
                     << shift;
        }
        return value;
    }

    uint32_t SequenceFile::get_u32(size_t offset) const
    {
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i)
        {
            const int shift = swapped_ ? 8 * (3 - i) : 8 * i;
            value |= static_cast<uint32_t>(static_cast<unsigned char>(bytes_[offset + i]))
                     << shift;
        }
        return value;
    }

    void SequenceFile::build_index()
    {
        if (indexed_)
            return;
        if (binary_)
            index_binary();
        else
            index_text();
        indexed_ = true;
        names_.reserve(spans_.size());
        for (const Span& span : spans_)
            names_.push_back(span.name);
    }

    void SequenceFile::index_text()
    {
        size_t line_start = 0;
        while (line_start < bytes_.size())
        {
            size_t line_end = bytes_.find('\n', line_start);
            if (line_end == std::string::npos)
                line_end = bytes_.size();

            if (bytes_[line_start] == '[')
            {
                size_t close = bytes_.find(']', line_start);
                if (close != std::string::npos && close < line_end)
                {
                    if (!spans_.empty())
                        spans_.back().end = line_start;
                    Span span;
                    span.name = bytes_.substr(line_start + 1, close - line_start - 1);
                    span.begin = line_end < bytes_.size() ? line_end + 1 : line_end;
                    span.end = bytes_.size();
                    spans_.push_back(span);
                }
            }
            line_start = line_end + 1;
        }
    }

    void SequenceFile::index_binary()
    {
        size_t cursor = 8 + 3 * 8; /* magic + version triplet */

        while (cursor + 8 <= bytes_.size())
        {
            const uint64_t raw_code = get_u64(cursor);
            if ((raw_code >> 32) != 0xFFFFFFFFULL)
                throw std::runtime_error(
                    "binary Pulseq file '" + path_ + "': expected a section code at byte " +
                    std::to_string(cursor));
            const uint32_t code = static_cast<uint32_t>(raw_code & 0xFFFFFFFFULL);
            const char* name = binary_section_name(code);
            if (name == nullptr)
                throw std::runtime_error(
                    "binary Pulseq file '" + path_ + "': unknown section code " +
                    std::to_string(code) + "; cannot index past it");
            cursor += 8;

            const size_t begin = cursor;

            if (code == 1) /* DEFINITIONS: entry walk */
            {
                const uint64_t count = get_u64(cursor);
                cursor += 8;
                for (uint64_t i = 0; i < count; ++i)
                {
                    const size_t nul = bytes_.find('\0', cursor);
                    if (nul == std::string::npos)
                        throw std::runtime_error("truncated definitions section");
                    cursor = nul + 1;
                    const unsigned values = static_cast<unsigned char>(bytes_[cursor]);
                    const char tag = bytes_[cursor + 1];
                    cursor += 2;
                    cursor += (tag == 'c') ? values : (tag == 'i') ? values * 4u : values * 8u;
                }
            }
            else if (code == 8) /* SHAPES: entry walk */
            {
                const uint64_t count = get_u64(cursor);
                cursor += 8;
                for (uint64_t i = 0; i < count; ++i)
                {
                    const uint64_t samples = get_u64(cursor + 4 + 8);
                    cursor += 4 + 8 + 8 + samples * 4;
                }
            }
            else if (code == 16) /* LABELNAMES: entry walk (id + NUL name) */
            {
                const uint64_t count = get_u64(cursor);
                cursor += 8;
                for (uint64_t i = 0; i < count; ++i)
                {
                    const size_t nul = bytes_.find('\0', cursor + 4);
                    if (nul == std::string::npos)
                        throw std::runtime_error("truncated label-names section");
                    cursor = nul + 1;
                }
            }
            else if (code == 13) /* SOFTDELAYS: entry walk */
            {
                cursor += 4; /* extension type id */
                const uint64_t count = get_u64(cursor);
                cursor += 8;
                for (uint64_t i = 0; i < count; ++i)
                {
                    const uint32_t hint_length = get_u32(cursor + 4 + 4 + 8 + 8);
                    cursor += 4 + 4 + 8 + 8 + 4 + hint_length;
                }
            }
            else if (code == 14) /* RFSHIMS: entry walk */
            {
                cursor += 4;
                const uint64_t count = get_u64(cursor);
                cursor += 8;
                for (uint64_t i = 0; i < count; ++i)
                {
                    const uint32_t channels = get_u32(cursor + 4);
                    cursor += 4 + 4 + static_cast<size_t>(channels) * 2 * 8;
                }
            }
            else
            {
                if (binary_has_type_id(code))
                    cursor += 4;
                const uint64_t count = get_u64(cursor);
                cursor += 8;
                cursor += count * binary_row_stride(code);
            }

            if (cursor > bytes_.size())
                throw std::runtime_error(
                    "binary Pulseq file '" + path_ + "': section " + name +
                    " runs past the end of the file");

            Span span;
            span.name = name;
            span.begin = begin;
            span.end = cursor;
            spans_.push_back(span);
        }
    }

    const std::vector<std::string>& SequenceFile::section_names()
    {
        build_index();
        return names_;
    }

    bool SequenceFile::has_section(const std::string& name)
    {
        build_index();
        for (const Span& span : spans_)
        {
            if (span.name == name)
                return true;
        }
        return false;
    }

    const std::map<std::string, Definition>& SequenceFile::definitions()
    {
        parse_definitions();
        return definitions_;
    }

    void SequenceFile::parse_definitions()
    {
        if (definitions_parsed_)
            return;
        build_index();
        for (const Span& span : spans_)
        {
            if (span.name != "DEFINITIONS")
                continue;
            if (binary_)
                parse_definitions_binary(span);
            else
                parse_definitions_text(span);
            break;
        }
        definitions_parsed_ = true;
    }

    void SequenceFile::parse_definitions_text(const Span& span)
    {
        std::istringstream lines(bytes_.substr(span.begin, span.end - span.begin));
        std::string line;
        while (std::getline(lines, line))
        {
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            if (line.empty() || line[0] == '#')
                continue;

            std::istringstream fields(line);
            std::string key;
            fields >> key;
            std::vector<std::string> tokens;
            std::string token;
            while (fields >> token)
                tokens.push_back(token);

            bool numeric = !tokens.empty();
            bool integral = numeric;
            std::vector<double> numbers;
            numbers.reserve(tokens.size());
            for (const std::string& value : tokens)
            {
                double parsed = 0.0;
                if (!parse_full_double(value, &parsed))
                {
                    numeric = false;
                    break;
                }
                integral = integral && parse_full_long(value);
                numbers.push_back(parsed);
            }

            if (numeric)
            {
                definitions_[key] = integral ? Definition::integers(std::move(numbers))
                                             : Definition(std::move(numbers));
            }
            else
            {
                std::string joined;
                for (size_t i = 0; i < tokens.size(); ++i)
                {
                    if (i)
                        joined.push_back(' ');
                    joined.append(tokens[i]);
                }
                definitions_[key] = Definition(std::move(joined));
            }
        }
    }

    void SequenceFile::parse_definitions_binary(const Span& span)
    {
        size_t cursor = span.begin;
        const uint64_t count = get_u64(cursor);
        cursor += 8;
        for (uint64_t i = 0; i < count; ++i)
        {
            const size_t nul = bytes_.find('\0', cursor);
            const std::string key = bytes_.substr(cursor, nul - cursor);
            cursor = nul + 1;
            const unsigned values = static_cast<unsigned char>(bytes_[cursor]);
            const char tag = bytes_[cursor + 1];
            cursor += 2;

            if (tag == 'c')
            {
                definitions_[key] = Definition(bytes_.substr(cursor, values));
                cursor += values;
            }
            else if (tag == 'i')
            {
                std::vector<double> numbers;
                numbers.reserve(values);
                for (unsigned v = 0; v < values; ++v)
                {
                    numbers.push_back(static_cast<int32_t>(get_u32(cursor)));
                    cursor += 4;
                }
                definitions_[key] = Definition::integers(std::move(numbers));
            }
            else
            {
                std::vector<double> numbers;
                numbers.reserve(values);
                for (unsigned v = 0; v < values; ++v)
                {
                    const uint64_t bits = get_u64(cursor);
                    double value;
                    std::memcpy(&value, &bits, 8);
                    numbers.push_back(value);
                    cursor += 8;
                }
                definitions_[key] = Definition(std::move(numbers));
            }
        }
    }

    std::string SequenceFile::next_sequence()
    {
        const auto& defs = definitions();
        const auto it = defs.find("NextSequence");
        if (it == defs.end() || it->second.kind() != Definition::Kind::Text)
            return "";
        return it->second.text();
    }

    Sequence& SequenceFile::sequence()
    {
        if (!sequence_)
            sequence_ = std::make_unique<Sequence>(read_file(path_));
        return *sequence_;
    }

}  // namespace pulseq
