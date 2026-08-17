/**
 * @file write_text.cpp
 * @brief The Pulseq `.seq` text writer.
 *
 * A port of fastseq's writer, which is itself PyPulseq's with the two 1.5.1
 * extension sections added.  Every format string below is the one Python
 * used, character for character, and that is not stylistic: the test that
 * proves this file correct diffs its output against what fastseq wrote for
 * the same sequence, so a `%g` that should have been a `%12g` shows up as a
 * failed byte comparison rather than as a subtly different file.
 *
 * (C's printf and Python's `%` agree on every conversion used here -- checked
 * across denormals, -0.0, and the exponent boundaries -- so matching the
 * format string is enough to match the bytes.)
 *
 * Two sections are as long as the scan rather than as long as the design:
 * `[BLOCKS]`, which has a row per block, and `[EXTENSIONS]`, which has one per
 * distinct chain and therefore one per labelled readout before deduplication.
 * Those two go through `render_rows` below, which lays digits
 * straight into a preallocated buffer, in parallel; everything else is per
 * distinct event and goes through snprintf.
 */

#include "pulseq/sequence.hpp"
#include "pulseq/write.hpp"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

extern "C"
{
#include "external_md5.h"
}

namespace pulseq
{

    namespace
    {
        /** snprintf into the output string. */
        void appendf(std::string& out, const char* format, ...)
#if defined(__GNUC__)
            __attribute__((format(printf, 2, 3)))
#endif
            ;

        void appendf(std::string& out, const char* format, ...)
        {
            char stack[512];
            va_list args;
            va_start(args, format);
            const int needed = std::vsnprintf(stack, sizeof(stack), format, args);
            va_end(args);

            if (needed < 0)
                throw std::runtime_error("write(): formatting failed");
            if (static_cast<size_t>(needed) < sizeof(stack))
            {
                out.append(stack, static_cast<size_t>(needed));
                return;
            }

            std::vector<char> heap(static_cast<size_t>(needed) + 1);
            va_start(args, format);
            std::vsnprintf(heap.data(), heap.size(), format, args);
            va_end(args);
            out.append(heap.data(), static_cast<size_t>(needed));
        }

        /** Decimal width of a non-negative count, for the block-number column. */
        int decimal_width(long value)
        {
            int width = 1;
            while (value >= 10)
            {
                value /= 10;
                ++width;
            }
            return width;
        }

        /* ============================================================== */
        /*  The two sections whose size is the scan                       */
        /* ============================================================== */
        //
        // `[BLOCKS]` has a row per block and `[EXTENSIONS]` one per distinct
        // chain, so on a large 3D protocol these two are most of the file and
        // rendering them is most of the cost of writing one.  Appending a
        // character at a time -- a bounds check and a possible reallocation per
        // digit -- is what makes that expensive, so instead each row's maximum
        // length is known up front, a buffer is allocated for the worst case,
        // and the digits go in through a raw cursor.
        //
        // Chunks are rendered in parallel and concatenated in order, so the
        // output does not depend on how many threads the machine offered.

        /** Write @p value right-aligned in @p width; returns the new cursor. */
        inline char* put_int_field(char* out, long value, int width)
        {
            char digits[24];
            int len = 0;
            unsigned long magnitude;

            if (value < 0)
                magnitude = static_cast<unsigned long>(-(value + 1)) + 1UL;
            else
                magnitude = static_cast<unsigned long>(value);

            do
            {
                digits[len++] = static_cast<char>('0' + (magnitude % 10));
                magnitude /= 10;
            } while (magnitude != 0);

            if (value < 0)
                digits[len++] = '-';

            for (int pad = len; pad < width; ++pad)
                *out++ = ' ';
            while (len > 0)
                *out++ = digits[--len];
            return out;
        }

        unsigned worker_count(size_t rows)
        {
            // Below this the threads cost more than the work they save.
            if (rows < (1u << 16))
                return 1;
            unsigned available = std::thread::hardware_concurrency();
            if (available == 0)
                available = 1;
            return std::min(available, 8u);
        }

        /**
         * Render @p rows rows of integer fields, in parallel, onto @p out.
         *
         * @p emit writes one row and returns the cursor it left off at; it must
         * never write more than @p row_bound bytes.
         *
         * The pieces land in @p out with a single copy each.  Going through an
         * intermediate string first would cost a second pass over what, on a
         * large scan, is eighty megabytes.
         */
        template <typename Emit>
        void render_rows(std::string& out, size_t rows, size_t row_bound, const Emit& emit)
        {
            if (rows == 0)
                return;

            const unsigned workers = worker_count(rows);
            const size_t chunk = (rows + workers - 1) / workers;

            std::vector<std::string> pieces(workers);
            auto run = [&](unsigned worker) {
                const size_t first = std::min(rows, static_cast<size_t>(worker) * chunk);
                const size_t last = std::min(rows, first + chunk);
                if (first >= last)
                    return;
                std::string& piece = pieces[worker];
                piece.resize((last - first) * row_bound);
                char* cursor = &piece[0];
                for (size_t row = first; row < last; ++row)
                    cursor = emit(cursor, row);
                piece.resize(static_cast<size_t>(cursor - piece.data()));
            };

            if (workers == 1)
            {
                run(0);
            }
            else
            {
                std::vector<std::thread> threads;
                threads.reserve(workers - 1);
                for (unsigned worker = 1; worker < workers; ++worker)
                    threads.emplace_back(run, worker);
                run(0);
                for (std::thread& thread : threads)
                    thread.join();
            }

            size_t total = 0;
            for (const std::string& piece : pieces)
                total += piece.size();

            // In worker order, so the file does not depend on how many threads
            // the machine offered.
            const size_t base = out.size();
            out.resize(base + total);
            char* destination = &out[base];
            for (const std::string& piece : pieces)
            {
                std::memcpy(destination, piece.data(), piece.size());
                destination += piece.size();
            }
        }

        /** Block durations as raster ticks, refusing any that is not a multiple. */
        std::vector<long> duration_ticks(const Sequence& seq)
        {
            const int n = seq.num_blocks();
            const double raster = seq.block_duration_raster();
            std::vector<long> ticks(static_cast<size_t>(n));
            const double* durations = seq.block_durations();

            for (int i = 0; i < n; ++i)
            {
                const double exact = durations[i] / raster;
                const double rounded = std::rint(exact);
                if (std::fabs(rounded - exact) >= 1e-6)
                {
                    throw std::runtime_error(
                        "block " + std::to_string(i + 1) +
                        " duration is not a multiple of the block duration raster (" +
                        std::to_string(exact * raster) + " s)");
                }
                ticks[static_cast<size_t>(i)] = static_cast<long>(rounded);
            }
            return ticks;
        }

        void write_definition_value(std::string& out, const Definition& def)
        {
            switch (def.kind())
            {
            case Definition::Kind::Text:
                out.append(def.text());
                out.push_back(' ');
                break;
            case Definition::Kind::Int:
            case Definition::Kind::Real:
                for (double value : def.numbers())
                    appendf(out, "%0.9g ", value);
                break;
            }
        }
    }  // namespace

    /* ================================================================== */
    /*  Revision                                                          */
    /* ================================================================== */

    int required_revision(const Sequence& seq)
    {
        // Rotations and RF shims are 1.5.1; a label Pulseq does not define is
        // 1.5.2.  A sequence using none of them stays plain 1.5.0, so the
        // common case produces a file any interpreter reads.
        int revision = seq.version_revision();

        for (const IntTable* library : {&seq.label_set_library(), &seq.label_inc_library()})
        {
            for (int id = 1; id <= library->size(); ++id)
            {
                if (seq.is_custom_label(library->row(id)[1]))
                    return std::max(revision, 2);
            }
        }

        if (!seq.rotation_library().empty() || !seq.rf_shim_library().empty())
            revision = std::max(revision, 1);
        return revision;
    }

    /* ================================================================== */
    /*  The writer                                                        */
    /* ================================================================== */

    std::string write_text(Sequence& seq, bool create_signature)
    {
        // Prewrite: a sequence built block by block registers its waveforms as
        // it was given them, because compressing a candidate that
        // deduplication is about to drop is work spent on a shape that will
        // not be in the file.  This is where the survivors are encoded.
        seq.compress_shapes();

        const int n_blocks = seq.num_blocks();
        const std::vector<long> ticks = duration_ticks(seq);

        seq.set_definition("TotalDuration", Definition(seq.duration()));
        seq.publish_rasters();

        /* Every read below goes through a const reference, so the writer does
         * not look like a mutation to the sequence.  Taking `Table&` from a
         * non-const Sequence is what gives up its deduplicated claim, and a
         * writer that only reads must not cost a caller that pass. */
        const Sequence& reading = seq;


        std::string out;
        // Blocks dominate a large file; ~35 characters each is close enough
        // that the buffer grows a couple of times rather than a couple of
        // dozen.
        out.reserve(static_cast<size_t>(n_blocks) * 36 + 4096);

        out.append("# Pulseq sequence file\n# Created by PyPulseq\n\n");

        out.append("[VERSION]\n");
        appendf(out, "major %d\n", reading.version_major());
        appendf(out, "minor %d\n", reading.version_minor());
        appendf(out, "revision %d\n", required_revision(seq));
        out.push_back('\n');

        if (!reading.definitions().empty())
        {
            out.append("[DEFINITIONS]\n");
            for (const auto& entry : reading.definitions())
            {
                out.append(entry.first);
                out.push_back(' ');
                write_definition_value(out, entry.second);
                out.push_back('\n');
            }
            out.push_back('\n');
        }

        /* -- blocks ---------------------------------------------------- */

        out.append("# Format of blocks:\n");
        out.append("# NUM DUR RF  GX  GY  GZ  ADC  EXT\n");
        out.append("[BLOCKS]\n");
        if (n_blocks > 0)
        {
            const int number_width = decimal_width(n_blocks);
            const int widths[8] = {number_width, 3, 3, 3, 3, 3, 2, 2};
            const int32_t* events = reading.block_events();
            const long* tick = ticks.data();

            // The worst a row can be: every field at its widest, a separator
            // between each, and the newline.
            size_t row_bound = 1;  // newline
            for (int column = 0; column < 8; ++column)
                row_bound += 1 + std::max(widths[column], 12);

            render_rows(
                out, static_cast<size_t>(n_blocks), row_bound,
                [&](char* cursor, size_t i) {
                    const int32_t* row = events + i * BLOCK_WIDTH;
                    cursor = put_int_field(cursor, static_cast<long>(i) + 1, widths[0]);
                    *cursor++ = ' ';
                    cursor = put_int_field(cursor, tick[i], widths[1]);
                    for (int column = 0; column < BLOCK_WIDTH; ++column)
                    {
                        *cursor++ = ' ';
                        cursor = put_int_field(cursor, row[column], widths[column + 2]);
                    }
                    *cursor++ = '\n';
                    return cursor;
                });
        }
        out.push_back('\n');

        /* -- RF -------------------------------------------------------- */

        if (!reading.rf_library().empty())
        {
            out.append("# Format of RF events:\n");
            out.append("# id ampl. mag_id phase_id time_shape_id center delay freqPPm phasePPM freq phase use\n");
            out.append("# ..   Hz      ..       ..            ..     us    us     ppm  rad/MHz   Hz   rad  ..\n");
            out.append("# Field \"use\" is the initial of: excitation refocusing inversion saturation preparation other undefined\n");
            out.append("[RF]\n");

            const double raster = reading.rf_raster_time();
            for (int id = 1; id <= reading.rf_library().size(); ++id)
            {
                const double* d = reading.rf_library().row(id);
                const double center = d[4] * 1e6;
                const double delay = std::rint(d[5] / raster) * raster * 1e6;
                const char use = reading.rf_uses()[static_cast<size_t>(id) - 1];
                appendf(out, "%.0f %12g %.0f %.0f %.0f %g %g %g %g %g %g %c\n",
                        static_cast<double>(id), d[0], d[1], d[2], d[3], center, delay,
                        d[6], d[7], d[8], d[9], use ? use : 'u');
            }
            out.push_back('\n');
        }

        /* -- gradients ------------------------------------------------- */
        //
        // The two kinds share one numbering, so they are collected by walking
        // that numbering once rather than by iterating the tables they are
        // stored in -- which is what keeps the ids in the file the ids the
        // sequence handed out.

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
            out.append("# Format of arbitrary gradients:\n");
            out.append("#   time_shape_id of 0 means default timing (stepping with grad_raster starting at 1/2 of grad_raster)\n");
            out.append("# id amplitude first last amp_shape_id time_shape_id delay\n");
            out.append("# ..      Hz/m  Hz/m Hz/m        ..         ..          us\n");
            out.append("[GRADIENTS]\n");
            for (int id : arbitrary)
            {
                const double* d = reading.arb_library().row(reading.grad_row(id));
                appendf(out, "%.0f %12g %12g %12g %.0f %.0f %.0f\n",
                        static_cast<double>(id), d[0], d[1], d[2], d[3], d[4],
                        std::rint(d[5] * 1e6));
            }
            out.push_back('\n');
        }

        if (!trapezoids.empty())
        {
            out.append("# Format of trapezoid gradients:\n");
            out.append("# id amplitude rise flat fall delay\n");
            out.append("# ..      Hz/m   us   us   us    us\n");
            out.append("[TRAP]\n");
            for (int id : trapezoids)
            {
                const double* d = reading.trap_library().row(reading.grad_row(id));
                appendf(out, "%2.0f %12g %3.0f %4.0f %3.0f %3.0f\n",
                        static_cast<double>(id), d[0], 1e6 * d[1], 1e6 * d[2], 1e6 * d[3],
                        1e6 * d[4]);
            }
            out.push_back('\n');
        }

        /* -- ADC ------------------------------------------------------- */

        if (!reading.adc_library().empty())
        {
            out.append("# Format of ADC events:\n");
            out.append("# id num dwell delay freqPPM phasePPM freq phase phase_id\n");
            out.append("# ..  ..    ns    us     ppm  rad/MHz   Hz   rad       ..\n");
            out.append("[ADC]\n");
            for (int id = 1; id <= reading.adc_library().size(); ++id)
            {
                const double* d = reading.adc_library().row(id);
                appendf(out, "%.0f %.0f %.0f %.0f %g %g %g %g %.0f\n",
                        static_cast<double>(id), d[0], 1e9 * d[1], 1e6 * d[2], d[3], d[4],
                        d[5], d[6], d[7]);
            }
            out.push_back('\n');
        }

        /* -- extension chains ------------------------------------------ */

        if (!reading.extensions_library().empty())
        {
            out.append("# Format of extension lists:\n");
            out.append("# id type ref next_id\n");
            out.append("# next_id of 0 terminates the list\n");
            out.append("# Extension list is followed by extension specifications\n");
            out.append("[EXTENSIONS]\n");
            const int count = reading.extensions_library().size();
            const IntTable& chains = reading.extensions_library();
            render_rows(
                out, static_cast<size_t>(count), 4 * 13 + 1,
                [&](char* cursor, size_t i) {
                    const int32_t* row = chains.row(static_cast<int>(i) + 1);
                    cursor = put_int_field(cursor, static_cast<long>(i) + 1, 1);
                    for (int column = 0; column < EXTENSION_WIDTH; ++column)
                    {
                        *cursor++ = ' ';
                        cursor = put_int_field(cursor, row[column], 1);
                    }
                    *cursor++ = '\n';
                    return cursor;
                });
            out.push_back('\n');
        }

        /* -- extension specifications ---------------------------------- */

        if (!reading.trigger_library().empty())
        {
            out.append("# Extension specification for digital output and input triggers:\n");
            out.append("# id type channel delay (us) duration (us)\n");
            appendf(out, "extension TRIGGERS %d\n", seq.extension_type_id("TRIGGERS"));
            for (int id = 1; id <= reading.trigger_library().size(); ++id)
            {
                const double* d = reading.trigger_library().row(id);
                appendf(out, "%.0f %.0f %.0f %.0f %.0f\n", static_cast<double>(id), d[0], d[1],
                        1e6 * d[2], 1e6 * d[3]);
            }
            out.push_back('\n');
        }

        {
            const std::pair<const char*, const IntTable*> label_sections[2] = {
                {"LABELSET", &reading.label_set_library()},
                {"LABELINC", &reading.label_inc_library()},
            };
            for (const auto& section : label_sections)
            {
                if (section.second->empty())
                    continue;
                out.append("# Extension specification for setting labels:\n");
                out.append("# id set labelstring\n");
                appendf(out, "extension %s %d\n", section.first,
                        seq.extension_type_id(section.first));
                for (int id = 1; id <= section.second->size(); ++id)
                {
                    const int32_t* row = section.second->row(id);
                    // The name is what the file carries; the number was only
                    // ever how this sequence indexed it.
                    const std::string& name = reading.label_name(row[1]);
                    appendf(out, "%.0f %.0f %s\n", static_cast<double>(id),
                            static_cast<double>(row[0]),
                            name.empty() ? "UNKNOWN" : name.c_str());
                }
                out.push_back('\n');
            }
        }

        if (!reading.rf_shim_library().empty())
        {
            out.append("# Extension specification for RF shimming:\n");
            out.append("# id num_chan factor magn_c1 phase_c1 magn_c2 phase_c2 ...\n");
            appendf(out, "extension RF_SHIMS %d\n", seq.extension_type_id("RF_SHIMS"));
            for (int id = 1; id <= reading.rf_shim_library().size(); ++id)
            {
                const int length = reading.rf_shim_library().length(id);
                const double* values = reading.rf_shim_library().row(id);
                appendf(out, "%d %d", id, length / 2);
                for (int i = 0; i < length; ++i)
                    appendf(out, " %g", values[i]);
                out.push_back('\n');
            }
            out.push_back('\n');
        }

        if (!reading.rotation_library().empty())
        {
            out.append("# Extension specification for rotation events:\n");
            out.append("# id RotQuat0 RotQuatX RotQuatY RotQuatZ\n");
            appendf(out, "extension ROTATIONS %d\n", seq.extension_type_id("ROTATIONS"));
            for (int id = 1; id <= reading.rotation_library().size(); ++id)
            {
                const double* d = reading.rotation_library().row(id);
                appendf(out, "%.0f %12g %12g %12g %12g\n", static_cast<double>(id), d[0], d[1],
                        d[2], d[3]);
            }
            out.push_back('\n');
        }

        if (!reading.soft_delay_library().empty())
        {
            out.append("# Extension specification for soft delays:\n");
            out.append("# id num offset factor hint\n");
            out.append("# ..  ..     us     ..   ..\n");
            appendf(out, "extension DELAYS %d\n", seq.extension_type_id("DELAYS"));
            int id = 1;
            for (const SoftDelay& row : reading.soft_delay_library())
            {
                appendf(out, "%.0f %.0f %.0f %.0f %s\n", static_cast<double>(id),
                        static_cast<double>(row.num), std::rint(row.offset * 1e6), row.factor,
                        row.hint.c_str());
                ++id;
            }
            out.push_back('\n');
        }

        /* -- shapes ---------------------------------------------------- */

        if (!reading.shape_library().empty())
        {
            out.append("# Sequence Shapes\n");
            out.append("[SHAPES]\n\n");
            for (int id = 1; id <= reading.shape_library().size(); ++id)
            {
                appendf(out, "shape_id %.0f\n", static_cast<double>(id));
                appendf(out, "num_samples %.0f\n",
                        static_cast<double>(reading.shape_library().num_uncompressed(id)));
                const int count = reading.shape_library().num_compressed(id);
                const double* samples = reading.shape_library().samples(id);
                for (int i = 0; i < count; ++i)
                    appendf(out, "%.9g\n", samples[i]);
                out.push_back('\n');
            }
        }

        /* -- signature -------------------------------------------------- */

        if (create_signature)
        {
            struct MD5Context context;
            unsigned char digest[16];
            MD5Init(&context);
            MD5Update(&context, reinterpret_cast<unsigned char*>(const_cast<char*>(out.data())),
                      static_cast<unsigned>(out.size()));
            MD5Final(digest, &context);

            char hex[33];
            for (int i = 0; i < 16; ++i)
                std::snprintf(hex + 2 * i, 3, "%02x", digest[i]);

            out.append("\n[SIGNATURE]\n");
            out.append("# This is the hash of the Pulseq file, calculated right before the [SIGNATURE] section was added\n");
            out.append("# It can be reproduced/verified with md5sum if the file trimmed to the position right above [SIGNATURE]\n");
            out.append("# The new line character preceding [SIGNATURE] BELONGS to the signature (and needs to be stripped away for recalculating/verification)\n");
            out.append("Type md5\n");
            appendf(out, "Hash %s\n", hex);
        }

        return out;
    }

}  // namespace pulseq
