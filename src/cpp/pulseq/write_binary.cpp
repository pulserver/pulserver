/**
 * @file write_binary.cpp
 * @brief Writer for Pulseq's native binary sequence format.
 *
 * The format is upstream Pulseq's own (`writeBinary.m` / `readBinary.m`).
 * What this emits is diffed byte for byte against the Python writer's output
 * for the same sequence.
 *
 * Two details are load-bearing and easy to get wrong:
 *
 * **Rounding.**  Times go out as integer picoseconds via Python's `round()`,
 * which rounds halves to even -- not the away-from-zero that `llround` does.
 * `std::nearbyint` under the default rounding mode is the one that agrees.
 *
 * **Section codes are unsigned.**  Their high word is 0xFFFFFFFF, so as a
 * signed int64 they are negative; they have to be written as the bit pattern
 * MATLAB's `bitshift` produces and `readBinary` switches on.
 */

#include "pulseq/raw64.hpp"

#include "pulseq/sequence.hpp"
#include "pulseq/write.hpp"
#include "pulseq/shape.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

namespace pulseq
{
    namespace
    {
        /* The writer runs over a sink twice: once counting, to learn the
         * file's size, then filling a buffer of exactly that size. One body
         * serves both, so the count cannot drift from the bytes. */
        struct CountingSink
        {
            size_t total = 0;
            void append(const char*, size_t n)
            {
                total += n;
            }
            void append(const std::string& text)
            {
                total += text.size();
            }
            void push_back(char)
            {
                ++total;
            }
            size_t size() const
            {
                return total;
            }
            void reserve(size_t)
            {
            }
            /** Room for @p n bytes; nothing to write into while counting. */
            char* grow(size_t n)
            {
                total += n;
                return nullptr;
            }
        };

        struct MemorySink
        {
            char* base;
            char* cursor;
            char* end;
            MemorySink(char* buffer, size_t capacity)
                : base(buffer), cursor(buffer), end(buffer + capacity)
            {
            }
            char* grow(size_t n)
            {
                if (static_cast<size_t>(end - cursor) < n)
                    throw std::logic_error("binary writer overran its counted size");
                char* at = cursor;
                cursor += n;
                return at;
            }
            void append(const char* bytes, size_t n)
            {
                std::memcpy(grow(n), bytes, n);
            }
            void append(const std::string& text)
            {
                append(text.data(), text.size());
            }
            void push_back(char c)
            {
                *grow(1) = c;
            }
            size_t size() const
            {
                return static_cast<size_t>(cursor - base);
            }
            void reserve(size_t)
            {
            }
        };

        namespace raw = ::pulseq::raw64;
        /** Magic bytes.  Byte-order free -- the version fields reveal the order. */
        const char FILE_HEADER[8] = {1, 'p', 'u', 'l', 's', 'e', 'q', 2};

        constexpr uint64_t SECTION_PREFIX = 0xFFFFFFFFULL << 32;

        enum Section : uint64_t
        {
            SEC_DEFINITIONS = SECTION_PREFIX | 1,
            SEC_BLOCKS = SECTION_PREFIX | 2,
            SEC_RF = SECTION_PREFIX | 3,
            SEC_GRADIENTS = SECTION_PREFIX | 4,
            SEC_TRAPEZOIDS = SECTION_PREFIX | 5,
            SEC_ADC = SECTION_PREFIX | 6,
            SEC_SHAPES = SECTION_PREFIX | 8,
            SEC_EXTENSIONS = SECTION_PREFIX | 9,
            SEC_TRIGGERS = SECTION_PREFIX | 10,
            SEC_LABELSET = SECTION_PREFIX | 11,
            SEC_LABELINC = SECTION_PREFIX | 12,
            SEC_SOFTDELAYS = SECTION_PREFIX | 13,
            SEC_RFSHIMS = SECTION_PREFIX | 14,
            SEC_ROTATIONS = SECTION_PREFIX | 15,
            SEC_LABELNAMES = SECTION_PREFIX | 16,
        };

        /** Seconds to integer picoseconds, rounding halves to even as Python does. */
        inline int64_t picoseconds(double seconds)
        {
            return static_cast<int64_t>(std::nearbyint(seconds * 1e12));
        }

        /* Everything is written little-endian, byte by byte, so the output does
         * not depend on the host.  The C reader accepts either order. */

        template <class Sink> void put_u64(Sink& out, uint64_t value)
        {
            char bytes[8];
            for (int i = 0; i < 8; ++i)
                bytes[i] = static_cast<char>((value >> (8 * i)) & 0xFF);
            out.append(bytes, 8);
        }

        template <class Sink> void put_i64(Sink& out, int64_t value)
        {
            put_u64(out, static_cast<uint64_t>(value));
        }

        template <class Sink> void put_i32(Sink& out, int32_t value)
        {
            const uint32_t u = static_cast<uint32_t>(value);
            char bytes[4];
            for (int i = 0; i < 4; ++i)
                bytes[i] = static_cast<char>((u >> (8 * i)) & 0xFF);
            out.append(bytes, 4);
        }

        template <class Sink> void put_f64(Sink& out, double value)
        {
            uint64_t bits;
            std::memcpy(&bits, &value, 8);
            put_u64(out, bits);
        }

        /* A structure-only shape record: two equal values open a run whose
         * length, -1.5, no decoder accepts; then the first sample, the last
         * sample and the peak magnitude. */
        constexpr int STUB_LENGTH = 6;
        constexpr double STUB_RUN = -1.5;

        template <class Sink> void put_f32(Sink& out, double value)
        {
            const float narrowed = static_cast<float>(value);
            uint32_t bits;
            std::memcpy(&bits, &narrowed, 4);
            char bytes[4];
            for (int i = 0; i < 4; ++i)
                bytes[i] = static_cast<char>((bits >> (8 * i)) & 0xFF);
            out.append(bytes, 4);
        }

        /** `count` values narrowed to f32 in one append: the same bytes
         *  put_f32 emits one value at a time, written straight into the
         *  buffer's tail. */
        bool host_is_little_endian()
        {
            const uint32_t probe = 1;
            unsigned char first;
            std::memcpy(&first, &probe, 1);
            return first == 1;
        }

        template <class Sink> void put_f32_run(Sink& out, const double* values, int count)
        {
            char* dst = out.grow(static_cast<size_t>(count) * 4);
            if (!dst)
                return;
            if (host_is_little_endian())
            {
                // The file's byte order is the host's: narrow straight into
                // place, a loop the compiler turns into packed conversions.
                float* as_float = reinterpret_cast<float*>(dst);
                for (int i = 0; i < count; ++i)
                    as_float[i] = static_cast<float>(values[i]);
                return;
            }
            for (int i = 0; i < count; ++i)
            {
                const float narrowed = static_cast<float>(values[i]);
                uint32_t bits;
                std::memcpy(&bits, &narrowed, 4);
                for (int b = 0; b < 4; ++b)
                    dst[b] = static_cast<char>((bits >> (8 * b)) & 0xFF);
                dst += 4;
            }
        }

        template <class Sink> void put_section(Sink& out, uint64_t code)
        {
            put_u64(out, code);
        }

        template <class Sink> void write_definitions(Sink& out, const Sequence& seq)
        {
            put_i64(out, static_cast<int64_t>(seq.definitions().size()));
            for (const auto& entry : seq.definitions())
            {
                out.append(entry.first);
                out.push_back('\0');

                const Definition& def = entry.second;
                if (def.kind() == Definition::Kind::Text)
                {
                    const std::string& text = def.text();
                    if (text.size() > 255)
                    {
                        throw std::runtime_error(
                            "definition '" + entry.first + "' is " + std::to_string(text.size()) +
                            " bytes; the format stores the value count in one byte");
                    }
                    out.push_back(static_cast<char>(text.size()));
                    out.push_back('c');
                    out.append(text);
                    continue;
                }

                const std::vector<double>& numbers = def.numbers();
                if (numbers.size() > 255)
                {
                    throw std::runtime_error(
                        "definition '" + entry.first + "' has " + std::to_string(numbers.size()) +
                        " values; the format allows 255");
                }

                out.push_back(static_cast<char>(numbers.size()));
                // A definition with no values at all -- `RequiredExtensions`
                // with nothing after it -- is tagged as integers.  That falls
                // out of the reference writer testing `all(isinstance(v, int)
                // for v in values)`, which is vacuously true when there are no
                // values; it is not a considered choice, and with no values
                // to tag it makes no difference to anything except the byte.
                if (def.kind() == Definition::Kind::Int || numbers.empty())
                {
                    // An integer stays an integer: writing 64 as 64.0 would come
                    // back as a float and print differently downstream.
                    out.push_back('i');
                    for (double value : numbers)
                        put_i32(out, static_cast<int32_t>(value));
                }
                else
                {
                    out.push_back('f');
                    for (double value : numbers)
                        put_f64(out, value);
                }
            }
        }
    } // namespace

    namespace
    {
        /* Prewrite: a sequence built block by block registers its waveforms as
         * it was given them, because compressing a candidate that
         * deduplication is about to drop is work spent on a shape that will
         * not be in the file.  This is where the survivors are encoded, and
         * where the definitions the file carries are settled. Idempotent, so
         * the counting and filling passes see one sequence. */
        void prepare_for_writing(Sequence& seq)
        {
            seq.compress_shapes();
            seq.set_definition("TotalDuration", Definition(seq.duration()));
            seq.publish_rasters();
        }

        /* One body for both passes. Every read goes through `reading`, a const
         * view, so the writer does not look like a mutation to the sequence:
         * taking `Table&` from a non-const Sequence is what gives up its
         * deduplicated claim. */
        template <class Sink> void write_binary_to(Sequence& seq, bool structure_only, Sink& out)
        {
            const int n_blocks = seq.num_blocks();
            const double block_raster = seq.block_duration_raster();
            const Sequence& reading = seq;

            out.append(FILE_HEADER, 8);
            put_i64(out, reading.version_major());
            put_i64(out, reading.version_minor());
            put_i64(out, required_revision(seq));

            if (!reading.definitions().empty())
            {
                put_section(out, SEC_DEFINITIONS);
                write_definitions(out, seq);
            }

            /* -- blocks: int64 duration then six int32, 32 bytes each -------- */

            put_section(out, SEC_BLOCKS);
            put_i64(out, n_blocks);
            {
                // Every block is exactly 32 bytes -- an int64 duration and six
                // int32 ids -- so this section's size is known before it is
                // written.  It is also most of a large file, which is why it is
                // laid into one allocation through a cursor rather than appended a
                // field at a time.
                const int32_t* events = reading.block_events();
                const double* durations = reading.block_durations();
                constexpr size_t STRIDE = 8 + 4 * BLOCK_WIDTH;

                unsigned char* cursor = reinterpret_cast<unsigned char*>(
                    out.grow(static_cast<size_t>(n_blocks) * STRIDE));

                for (int i = 0; i < n_blocks; ++i)
                {
                    const double exact = durations[i] / block_raster;
                    const double rounded = std::rint(exact);
                    if (std::fabs(rounded - exact) >= 1e-6)
                    {
                        throw std::runtime_error(
                            "block " + std::to_string(i + 1) +
                            " duration is not a multiple of the block duration raster");
                    }

                    if (!cursor)
                        continue;
                    const uint64_t ticks = static_cast<uint64_t>(static_cast<int64_t>(rounded));
                    for (int byte = 0; byte < 8; ++byte)
                        *cursor++ = static_cast<unsigned char>((ticks >> (8 * byte)) & 0xFF);

                    const int32_t* row = events + static_cast<size_t>(i) * BLOCK_WIDTH;
                    for (int column = 0; column < BLOCK_WIDTH; ++column)
                    {
                        const uint32_t value = static_cast<uint32_t>(row[column]);
                        for (int byte = 0; byte < 4; ++byte)
                            *cursor++ = static_cast<unsigned char>((value >> (8 * byte)) & 0xFF);
                    }
                }
            }

            /* -- RF ---------------------------------------------------------- */

            if (!reading.rf_library().empty())
            {
                put_section(out, SEC_RF);
                put_i64(out, reading.rf_library().size());
                for (int id = 1; id <= reading.rf_library().size(); ++id)
                {
                    const double* d = reading.rf_library().row(id);
                    put_i32(out, id);
                    put_f64(out, d[0]);
                    put_i32(out, static_cast<int32_t>(d[1]));
                    put_i32(out, static_cast<int32_t>(d[2]));
                    put_i32(out, static_cast<int32_t>(d[3]));
                    put_i64(out, picoseconds(d[4]));
                    put_i64(out, picoseconds(d[5]));
                    for (int i = 6; i < 10; ++i)
                        put_f64(out, d[i]);
                    const char use = reading.rf_uses()[static_cast<size_t>(id) - 1];
                    out.push_back(use ? use : 'u');
                }
            }

            /* -- gradients --------------------------------------------------- */

            std::vector<int> arbitrary, trapezoids;
            for (int id = 1; id <= reading.num_gradients(); ++id)
            {
                if (reading.grad_kind(id) == GradKind::Arbitrary)
                    arbitrary.push_back(id);
                else
                    trapezoids.push_back(id);
            }

            if (!arbitrary.empty())
            {
                put_section(out, SEC_GRADIENTS);
                put_i64(out, static_cast<int64_t>(arbitrary.size()));
                for (int id : arbitrary)
                {
                    const double* d = reading.arb_library().row(reading.grad_row(id));
                    put_i32(out, id);
                    put_f64(out, d[0]);
                    put_f64(out, d[1]);
                    put_f64(out, d[2]);
                    put_i32(out, static_cast<int32_t>(d[3]));
                    put_i32(out, static_cast<int32_t>(d[4]));
                    put_i64(out, picoseconds(d[5]));
                }
            }

            if (!trapezoids.empty())
            {
                put_section(out, SEC_TRAPEZOIDS);
                put_i64(out, static_cast<int64_t>(trapezoids.size()));
                for (int id : trapezoids)
                {
                    const double* d = reading.trap_library().row(reading.grad_row(id));
                    put_i32(out, id);
                    put_f64(out, d[0]);
                    for (int i = 1; i < TRAP_WIDTH; ++i)
                        put_i64(out, picoseconds(d[i]));
                }
            }

            /* -- ADC --------------------------------------------------------- */

            if (!reading.adc_library().empty())
            {
                put_section(out, SEC_ADC);
                put_i64(out, reading.adc_library().size());
                for (int id = 1; id <= reading.adc_library().size(); ++id)
                {
                    const double* d = reading.adc_library().row(id);
                    put_i32(out, id);
                    put_i64(out, static_cast<int64_t>(std::nearbyint(d[0])));
                    put_i64(out, picoseconds(d[1]));
                    put_i64(out, picoseconds(d[2]));
                    for (int i = 3; i < 7; ++i)
                        put_f64(out, d[i]);
                    put_i32(out, static_cast<int32_t>(d[7]));
                }
            }

            /* -- shapes ------------------------------------------------------ */

            if (!reading.shape_library().empty())
            {
                const int n_shapes = reading.shape_library().size();

                // Which shapes a structure-only file still carries in full: every
                // RF and ADC shape, every gradient time shape, and the shapes of
                // gradients played alongside an RF pulse (segmentation asks
                // whether that gradient is flat).
                std::vector<uint8_t> with_samples(
                    static_cast<size_t>(n_shapes) + 1,
                    structure_only ? 0 : 1);
                if (structure_only)
                {
                    const auto mark = [&](int shape_id)
                    {
                        if (shape_id > 0 && shape_id <= n_shapes)
                            with_samples[static_cast<size_t>(shape_id)] = 1;
                    };
                    for (int id = 1; id <= reading.rf_library().size(); ++id)
                    {
                        const double* d = reading.rf_library().row(id);
                        mark(static_cast<int>(d[1]));
                        mark(static_cast<int>(d[2]));
                        mark(static_cast<int>(d[3]));
                    }
                    for (int id = 1; id <= reading.adc_library().size(); ++id)
                        mark(static_cast<int>(reading.adc_library().row(id)[7]));
                    // A time shape is a few corner times, and the conversion reads
                    // it for every gradient that has one.
                    for (int id = 1; id <= reading.num_gradients(); ++id)
                        if (reading.grad_kind(id) == GradKind::Arbitrary)
                            mark(static_cast<int>(
                                reading.arb_library().row(reading.grad_row(id))[4]));
                    const int32_t* events = reading.block_events();
                    for (int i = 0; i < n_blocks; ++i)
                    {
                        const int32_t* row = events + static_cast<size_t>(i) * BLOCK_WIDTH;
                        if (row[0] <= 0)
                            continue;
                        for (int axis = 1; axis <= 3; ++axis)
                        {
                            const int gid = row[axis];
                            if (gid > 0 && reading.grad_kind(gid) == GradKind::Arbitrary)
                            {
                                const double* d = reading.arb_library().row(reading.grad_row(gid));
                                mark(static_cast<int>(d[3]));
                                mark(static_cast<int>(d[4]));
                            }
                        }
                    }
                }

                // Sized once for the whole section: the sample payload is the
                // bulk of a file, and growing into it would copy it over again.
                size_t section_bytes = 16;
                for (int id = 1; id <= n_shapes; ++id)
                    section_bytes += 20 +
                        (with_samples[static_cast<size_t>(id)]
                             ? static_cast<size_t>(reading.shape_library().num_compressed(id)) * 4
                             : STUB_LENGTH * 4);
                out.reserve(out.size() + section_bytes);

                put_section(out, SEC_SHAPES);
                put_i64(out, n_shapes);
                for (int id = 1; id <= n_shapes; ++id)
                {
                    const int count = reading.shape_library().num_compressed(id);
                    const int n_uncompressed = reading.shape_library().num_uncompressed(id);
                    const double* samples = reading.shape_library().samples(id);
                    put_i32(out, id);
                    put_i64(out, n_uncompressed);
                    if (with_samples[static_cast<size_t>(id)] || count <= STUB_LENGTH)
                    {
                        put_i64(out, count);
                        // The samples are the compressed (run-length) form exactly
                        // as the text format carries them; only the container
                        // changes.
                        put_f32_run(out, samples, count);
                        continue;
                    }
                    // A shape carried for structure alone: its first sample, last
                    // sample and peak magnitude behind a run-length marker no
                    // decoder accepts, so a reader that wants the waveform fails
                    // rather than reads nonsense.
                    double first_sample, last_sample, peak;
                    reading.shape_library().edge_stats(id, &first_sample, &last_sample, &peak);
                    const double stub[STUB_LENGTH] =
                        {0.0, 0.0, STUB_RUN, first_sample, last_sample, peak};
                    put_i64(out, STUB_LENGTH);
                    put_f32_run(out, stub, STUB_LENGTH);
                }
            }

            /* -- extension chains -------------------------------------------- */

            if (!reading.extensions_library().empty())
            {
                // The other section that grows with the scan: a fixed sixteen bytes
                // per chain, so it goes in the same way the blocks did.
                const int count = reading.extensions_library().size();
                put_section(out, SEC_EXTENSIONS);
                put_i64(out, count);

                unsigned char* cursor =
                    reinterpret_cast<unsigned char*>(out.grow(static_cast<size_t>(count) * 16));

                for (int id = 1; id <= count && cursor; ++id)
                {
                    const int32_t* row = reading.extensions_library().row(id);
                    const int32_t fields[4] = {id, row[0], row[1], row[2]};
                    for (int field = 0; field < 4; ++field)
                    {
                        const uint32_t value = static_cast<uint32_t>(fields[field]);
                        for (int byte = 0; byte < 4; ++byte)
                            *cursor++ = static_cast<unsigned char>((value >> (8 * byte)) & 0xFF);
                    }
                }
            }

            /* -- extension specifications ------------------------------------ */

            if (!reading.trigger_library().empty())
            {
                put_section(out, SEC_TRIGGERS);
                put_i32(out, seq.extension_type_id("TRIGGERS"));
                put_i64(out, reading.trigger_library().size());
                for (int id = 1; id <= reading.trigger_library().size(); ++id)
                {
                    const double* d = reading.trigger_library().row(id);
                    put_i32(out, id);
                    put_i32(out, static_cast<int32_t>(d[0]));
                    put_i32(out, static_cast<int32_t>(d[1]));
                    put_i64(out, picoseconds(d[2]));
                    put_i64(out, picoseconds(d[3]));
                }
            }

            {
                const std::pair<uint64_t, const IntTable*> label_sections[2] = {
                    {SEC_LABELSET, &reading.label_set_library()},
                    {SEC_LABELINC, &reading.label_inc_library()},
                };
                const char* names[2] = {"LABELSET", "LABELINC"};

                /* The file's label-id space is defined by the file itself: a
             * LABELNAMES section maps every id the label rows use onto its
             * name, so a reader in any process resolves them -- names
             * outside the builtin table included.  Ids are the writer's
             * registry ids, which the section makes self-describing. */
                std::map<int, std::string> used;
                for (int s = 0; s < 2; ++s)
                {
                    for (int id = 1; id <= label_sections[s].second->size(); ++id)
                    {
                        const int32_t* row = label_sections[s].second->row(id);
                        const std::string& label = reading.label_name(row[1]);
                        const int file_id = raw::pulseq_label_register_name(label.c_str());
                        if (file_id <= 0)
                            throw std::runtime_error(
                                "cannot encode label '" + label + "' in the binary format");
                        used.emplace(file_id, label);
                    }
                }
                if (!used.empty())
                {
                    put_section(out, SEC_LABELNAMES);
                    put_i64(out, static_cast<int64_t>(used.size()));
                    for (const auto& entry : used)
                    {
                        put_i32(out, entry.first);
                        out.append(entry.second);
                        out.push_back('\0');
                    }
                }

                for (int s = 0; s < 2; ++s)
                {
                    if (label_sections[s].second->empty())
                        continue;
                    put_section(out, label_sections[s].first);
                    put_i32(out, seq.extension_type_id(names[s]));
                    put_i64(out, label_sections[s].second->size());
                    for (int id = 1; id <= label_sections[s].second->size(); ++id)
                    {
                        const int32_t* row = label_sections[s].second->row(id);
                        const std::string& label = reading.label_name(row[1]);
                        put_i32(out, id);
                        put_i32(out, row[0]);
                        put_i32(out, raw::pulseq_label_register_name(label.c_str()));
                    }
                }
            }

            if (!reading.soft_delay_library().empty())
            {
                put_section(out, SEC_SOFTDELAYS);
                put_i32(out, seq.extension_type_id("DELAYS"));
                put_i64(out, static_cast<int64_t>(reading.soft_delay_library().size()));
                int id = 1;
                for (const SoftDelay& row : reading.soft_delay_library())
                {
                    put_i32(out, id++);
                    put_i32(out, row.num);
                    put_i64(out, picoseconds(row.offset));
                    put_f64(out, row.factor);
                    put_i32(out, static_cast<int32_t>(row.hint.size()));
                    out.append(row.hint);
                }
            }

            if (!reading.rf_shim_library().empty())
            {
                put_section(out, SEC_RFSHIMS);
                put_i32(out, seq.extension_type_id("RF_SHIMS"));
                put_i64(out, reading.rf_shim_library().size());
                for (int id = 1; id <= reading.rf_shim_library().size(); ++id)
                {
                    const int length = reading.rf_shim_library().length(id);
                    const double* values = reading.rf_shim_library().row(id);
                    put_i32(out, id);
                    put_i32(out, length / 2);
                    for (int i = 0; i < length; ++i)
                        put_f64(out, values[i]);
                }
            }

            if (!reading.rotation_library().empty())
            {
                put_section(out, SEC_ROTATIONS);
                put_i32(out, seq.extension_type_id("ROTATIONS"));
                put_i64(out, reading.rotation_library().size());
                for (int id = 1; id <= reading.rotation_library().size(); ++id)
                {
                    const double* d = reading.rotation_library().row(id);
                    put_i32(out, id);
                    for (int i = 0; i < ROTATION_WIDTH; ++i)
                        put_f64(out, d[i]);
                }
            }
        }
    } // namespace

    size_t write_binary_size(Sequence& seq, bool structure_only)
    {
        prepare_for_writing(seq);
        CountingSink counting;
        write_binary_to(seq, structure_only, counting);
        return counting.size();
    }

    void write_binary_into(Sequence& seq, bool structure_only, char* buffer, size_t size)
    {
        prepare_for_writing(seq);
        MemorySink sink(buffer, size);
        write_binary_to(seq, structure_only, sink);
        if (sink.size() != size)
            throw std::logic_error("binary writer did not fill its counted size");
    }

    std::string write_binary(Sequence& seq, bool structure_only)
    {
        const size_t size = write_binary_size(seq, structure_only);
        std::string out(size, '\0');
        write_binary_into(seq, structure_only, &out[0], size);
        return out;
    }

} // namespace pulseq
