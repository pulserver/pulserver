/**
 * @file read.cpp
 * @brief Building a pulseq::Sequence from a `.seq` or binary Pulseq file.
 *
 * The parsing is not done here.  src/c/pulseq already reads both formats,
 * and this file compiles a second, double-precision instance of it (see
 * src/cpp/include/pulseq/raw64.hpp) and converts what it produces.  Maintaining a
 * second parser to get a second precision would have been the alternative.
 *
 * Two things change on the way across:
 *
 * **Units.**  The raw model holds what the file holds -- microseconds,
 * nanoseconds, raster ticks -- because that is the cheapest thing for a parser
 * to do and the scanner wants the raw numbers anyway.  A `Sequence` holds SI,
 * as PyPulseq does, because that is what everything above it expects and what
 * the writers convert back out of.  The conversions below are the inverse of
 * the ones in write_text.cpp, one for one.
 *
 * **Label and hint names.**  The raw model stores a number, and the number
 * means whatever the C reader's table says it means -- which is not what
 * PyPulseq's table says, and neither is what a file inventing its own label
 * name means.  So names, not numbers, are what cross: each is looked up in the
 * C table and re-registered in the sequence's own.
 */

#include "pulseq/raw64.hpp"

#include "pulseq/read.hpp"
#include "pulseq/sequence.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace pulseq
{

    namespace
    {
        namespace raw = ::pulseq::raw64;

        constexpr double US = 1e-6;
        constexpr double NS = 1e-9;

        /** The raw grad table's first column: 1 for arbitrary, 0 for trapezoid. */
        constexpr double RAW_GRAD_ARBITRARY = 1.0;

        /**
         * Is @p text a number at all?
         *
         * Note what is deliberately *not* asked: whether it is a whole one.
         * A definition printed as "64" could have been an integer or a float,
         * and the file does not say -- so reading cannot know, and guessing
         * would put an `i` tag in the binary file where PyPulseq puts an `f`.
         * PyPulseq reads every numeric definition as float64, so this does
         * too; the Int kind is for a sequence built in Python out of Python
         * ints, where the distinction is real and is still known.
         */
        bool is_number(const char* text)
        {
            if (text == nullptr || *text == '\0')
                return false;

            char* end = nullptr;
            std::strtod(text, &end);
            return end != text && *end == '\0';
        }

        void copy_definitions(const raw::pulseq_file& file, Sequence& seq)
        {
            for (int i = 0; i < file.num_definitions; ++i)
            {
                const raw::pulseq_definition& def = file.definitions_library[i];
                const int count = def.value_size;
                if (count <= 0)
                {
                    // A key with no values is a real thing in the wild --
                    // `RequiredExtensions` with nothing after it -- and it has
                    // to be written back the same way, key and separator and
                    // no value.
                    seq.set_definition(def.name, Definition(std::vector<double>{}));
                    continue;
                }

                bool all_number = true;
                for (int v = 0; v < count; ++v)
                {
                    if (!is_number(def.value[v]))
                    {
                        all_number = false;
                        break;
                    }
                }

                if (!all_number)
                {
                    std::string joined;
                    for (int v = 0; v < count; ++v)
                    {
                        if (v)
                            joined.push_back(' ');
                        joined.append(def.value[v]);
                    }
                    seq.set_definition(def.name, Definition(std::move(joined)));
                    continue;
                }

                std::vector<double> numbers;
                numbers.reserve(static_cast<size_t>(count));
                for (int v = 0; v < count; ++v)
                    numbers.push_back(std::strtod(def.value[v], nullptr));

                seq.set_definition(def.name, Definition(std::move(numbers)));
            }
        }

        /** Re-register a raw label id under its name, in the sequence's table. */
        int32_t label_across(int raw_id, Sequence& seq)
        {
            const char* name = raw::pulseq_label_name_for_id(raw_id);
            if (name == nullptr)
                return 0;
            return static_cast<int32_t>(seq.label_id(name));
        }

        void copy_extension_type_ids(const raw::pulseq_file& file, Sequence& seq)
        {
            const std::pair<int, const char*> sections[] = {
                {PULSEQ_EXT_TRIGGER, "TRIGGERS"},
                {PULSEQ_EXT_ROTATION, "ROTATIONS"},
                {PULSEQ_EXT_LABELSET, "LABELSET"},
                {PULSEQ_EXT_LABELINC, "LABELINC"},
                {PULSEQ_EXT_RF_SHIM, "RF_SHIMS"},
                {PULSEQ_EXT_DELAY, "DELAYS"},
            };
            for (const auto& section : sections)
            {
                const int id = file.extension_map[section.first];
                if (id > 0)
                    seq.set_extension_type_id(section.second, id);
            }
        }
    } // namespace

    Sequence read_file(const std::string& path)
    {
        raw::pulseq_file file;
        raw::pulseq_file_init(&file, nullptr);

        const int rc = raw::pulseq_read(&file, path.c_str());
        if (PULSEQ_FAILED(rc))
        {
            raw::pulseq_file_free(&file);
            throw std::runtime_error(
                "could not read Pulseq file '" + path + "' (error " + std::to_string(rc) + ")");
        }

        Sequence seq;
        try
        {
            seq.set_version(file.version_major, file.version_minor, file.version_revision);

            const auto& reserved = file.reserved_definitions_library;
            seq.set_rasters(
                reserved.radiofrequency_raster_time * US,
                reserved.gradient_raster_time * US,
                reserved.adc_raster_time * US,
                reserved.block_duration_raster * US);

            copy_definitions(file, seq);

            // Prefer the definition over the reserved copy above.  The parser
            // keeps rasters in microseconds, so recovering seconds means
            // multiplying by 1e6 and dividing by it again -- and that is not
            // the identity in floating point.  A block duration is the raster
            // times a tick count, so one bit of error there lands in every
            // duration in the scan, and from there in TotalDuration, which the
            // binary format writes out in full.  The definition string is what
            // the file actually said.
            {
                const std::pair<const char*, int> raster_definitions[] = {
                    {"RadiofrequencyRasterTime", 0},
                    {"GradientRasterTime", 1},
                    {"AdcRasterTime", 2},
                    {"BlockDurationRaster", 3},
                };
                double rasters[4] = {
                    seq.rf_raster_time(),
                    seq.grad_raster_time(),
                    seq.adc_raster_time(),
                    seq.block_duration_raster()};
                for (const auto& entry : raster_definitions)
                {
                    const Definition* def = seq.definition(entry.first);
                    if (def != nullptr && !def->numbers().empty())
                        rasters[entry.second] = def->numbers()[0];
                }
                seq.set_rasters(rasters[0], rasters[1], rasters[2], rasters[3]);
            }
            copy_extension_type_ids(file, seq);

            /* -- RF ---------------------------------------------------- */

            for (int i = 0; i < file.rf_library_size; ++i)
            {
                const auto& src = file.rf_library[i];
                double row[RF_WIDTH] = {
                    src[0],
                    src[1],
                    src[2],
                    src[3],
                    src[4] * US,
                    src[5] * US,
                    src[6],
                    src[7],
                    src[8],
                    src[9],
                };
                const int tag = file.rf_use_tags ? file.rf_use_tags[i] : 0;
                // Indexed by PULSEQ_RF_USE_*; the initials Pulseq writes.
                static const char kUseChars[] = {'u', 'e', 'r', 'i', 's', 'p', 'o'};
                const char use =
                    (tag >= 0 && tag < static_cast<int>(sizeof(kUseChars))) ? kUseChars[tag] : 'u';
                seq.register_rf(row, use);
            }

            /* -- gradients --------------------------------------------- */
            //
            // One table in the raw model, discriminated by its first column,
            // and one numbering: walking it in order is what reproduces the
            // ids the file used.

            for (int i = 0; i < file.grad_library_size; ++i)
            {
                const auto& src = file.grad_library[i];
                if (src[0] == RAW_GRAD_ARBITRARY)
                {
                    const double row[ARB_WIDTH] =
                        {src[1], src[2], src[3], src[4], src[5], src[6] * US};
                    seq.register_arbitrary(row);
                }
                else
                {
                    const double row[TRAP_WIDTH] =
                        {src[1], src[2] * US, src[3] * US, src[4] * US, src[5] * US};
                    seq.register_trap(row);
                }
            }

            /* -- ADC --------------------------------------------------- */

            for (int i = 0; i < file.adc_library_size; ++i)
            {
                const auto& src = file.adc_library[i];
                const double row[ADC_WIDTH] = {
                    src[0],
                    src[1] * NS,
                    src[2] * US,
                    src[3],
                    src[4],
                    src[5],
                    src[6],
                    src[7],
                };
                seq.register_adc(row);
            }

            /* -- shapes ------------------------------------------------ */

            for (int i = 0; i < file.shapes_library_size; ++i)
            {
                const raw::pulseq_shape& shape = file.shapes_library[i];
                seq.register_shape(
                    shape.num_uncompressed_samples,
                    shape.samples,
                    shape.num_samples);
            }

            /* -- extension specifications ------------------------------ */

            for (int i = 0; i < file.trigger_library_size; ++i)
            {
                const auto& src = file.trigger_library[i];
                const double row[TRIGGER_WIDTH] = {src[0], src[1], src[2] * US, src[3] * US};
                seq.register_trigger(row);
            }

            for (int i = 0; i < file.rotation_library_size; ++i)
            {
                const auto& src = file.rotation_quaternion_library[i];
                const double row[ROTATION_WIDTH] = {src[0], src[1], src[2], src[3]};
                seq.register_rotation(row);
            }

            for (int i = 0; i < file.labelset_library_size; ++i)
            {
                const auto& src = file.labelset_library[i];
                seq.register_label_set(
                    static_cast<int32_t>(src[0]),
                    label_across(static_cast<int>(src[1]), seq));
            }

            for (int i = 0; i < file.labelinc_library_size; ++i)
            {
                const auto& src = file.labelinc_library[i];
                seq.register_label_inc(
                    static_cast<int32_t>(src[0]),
                    label_across(static_cast<int>(src[1]), seq));
            }

            for (int i = 0; i < file.rf_shim_library_size; ++i)
            {
                const raw::pulseq_rf_shim_entry& entry = file.rf_shim_library[i];
                seq.register_rf_shim(entry.values, 2 * entry.num_channels);
            }

            for (int i = 0; i < file.soft_delay_library_size; ++i)
            {
                const auto& src = file.soft_delay_library[i];
                SoftDelay row;
                row.num = static_cast<int32_t>(src[0]);
                row.offset = src[1] * US;
                row.factor = src[2];
                const char* hint = raw::pulseq_hint_name_for_id(static_cast<int>(src[3]));
                row.hint = hint ? hint : "";
                seq.register_soft_delay(row);
            }

            /* -- extension chains -------------------------------------- */
            //
            // Appended rather than chained through chain_extension(): the file
            // already assigned these ids, and a chain's id is referred to both
            // by blocks and by other chains, so it has to be kept as written.

            seq.extensions_library().resize(file.extensions_library_size);
            for (int i = 0; i < file.extensions_library_size; ++i)
            {
                const auto& src = file.extensions_library[i];
                int32_t* dst = seq.extensions_library().row(i + 1);
                dst[0] = static_cast<int32_t>(src[0]);
                dst[1] = static_cast<int32_t>(src[1]);
                dst[2] = static_cast<int32_t>(src[2]);
            }

            /* -- blocks ------------------------------------------------ */

            const double block_raster = seq.block_duration_raster();
            std::vector<int32_t> events(static_cast<size_t>(file.num_blocks) * BLOCK_WIDTH);
            std::vector<double> durations(static_cast<size_t>(file.num_blocks));
            for (int i = 0; i < file.num_blocks; ++i)
            {
                const auto& src = file.block_library[i];
                durations[static_cast<size_t>(i)] = src[0] * block_raster;
                int32_t* dst = events.data() + static_cast<size_t>(i) * BLOCK_WIDTH;
                for (int column = 0; column < BLOCK_WIDTH; ++column)
                    dst[column] = static_cast<int32_t>(src[column + 1]);
            }
            seq.set_blocks(events.data(), durations.data(), file.num_blocks);
        }
        catch (...)
        {
            raw::pulseq_file_free(&file);
            throw;
        }

        raw::pulseq_file_free(&file);
        return seq;
    }

} // namespace pulseq
