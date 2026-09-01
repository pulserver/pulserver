/**
 * @file shape.cpp
 * @brief Pulseq's shape codec, and the pass that applies it to a library.
 *
 * A port of `pypulseq.compress_shape`, which is itself the MATLAB toolbox's
 * `compressShape`.  The arithmetic is reproduced rather than improved on: the
 * quantisation step, the correction term and the "keep the original if it is
 * no longer" rule all decide what a `.seq` file contains, so a shape written
 * here and the same shape written there have to come out the same.
 */

#include <atomic>
#include <thread>
#include "pulseq/shape.hpp"

#include "pulseq/sequence.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace pulseq
{
    namespace
    {
        /** The encoding when it is shorter than the shape, else empty. */
        std::vector<double> encode_if_shorter(const double* samples, int count)
        {
            // Very short shapes are left alone: four numbers cannot be encoded
            // in fewer than four, and Pulseq does not try.
            if (count <= 4)
                return std::vector<double>();

            // Single-precision floats carry about 7.25 decimal digits, so that is
            // the grid the derivative is quantised onto.
            constexpr double QUANT = 1e-7;
            const size_t n = static_cast<size_t>(count);

            // datq: the derivative, quantised.  qcor: the correction that keeps
            // the *running sum* on the original rather than letting the
            // quantisation error accumulate over the waveform.
            std::vector<double> datd(n);
            double previous_scaled = samples[0] / QUANT;
            double running = std::rint(previous_scaled);
            datd[0] = running;
            double previous_error = std::rint(previous_scaled - running);

            for (size_t i = 1; i < n; ++i)
            {
                const double scaled = samples[i] / QUANT;
                const double step = std::rint(scaled - previous_scaled);
                running += step;
                const double error = std::rint(scaled - running);
                datd[i] = step + (error - previous_error);
                previous_scaled = scaled;
                previous_error = error;
            }

            // Run-length encode: a run longer than one is written as the value
            // twice and then the count less two, which is what tells a reader that
            // a repeat follows.
            std::vector<double> out;
            out.reserve(n);
            size_t i = 0;
            while (i < n)
            {
                size_t run = 1;
                // An encoding as long as the shape is already no shorter,
                // and it only ever grows: the rest cannot change the answer.
                if (out.size() >= n)
                    return std::vector<double>();
                while (i + run < n && datd[i + run] == datd[i])
                    ++run;
                const double value = datd[i] * QUANT;
                out.push_back(value);
                if (run > 1)
                {
                    out.push_back(value);
                    out.push_back(static_cast<double>(run) - 2.0);
                }
                i += run;
            }

            // Compressing is only worth it if it shrank the shape.
            if (n > out.size())
                return out;
            return std::vector<double>();
        }
    } // namespace

    std::vector<double> compress_shape(const double* samples, int count)
    {
        std::vector<double> encoded = encode_if_shorter(samples, count);
        if (!encoded.empty())
            return encoded;
        return std::vector<double>(samples, samples + count);
    }

    std::vector<double> decompress_shape(const double* samples, int count, int num_uncompressed)
    {
        // Equal counts mean compress_shape kept the original, so there is no
        // derivative to sum and nothing to expand.
        if (count == num_uncompressed)
            return std::vector<double>(samples, samples + count);

        if (num_uncompressed <= 0 || count <= 0)
            return std::vector<double>();

        std::vector<double> out(static_cast<size_t>(num_uncompressed), 0.0);

        // A run is the value twice followed by the count less two, so a pair of
        // equal neighbours is what says to read a third number.
        int packed = 1;
        int unpacked = 1;
        while (packed < count)
        {
            if (samples[packed - 1] != samples[packed])
            {
                out[static_cast<size_t>(unpacked) - 1] = samples[packed - 1];
                ++packed;
                ++unpacked;
                continue;
            }

            if (packed + 1 >= count)
                throw std::runtime_error("decompress_shape: run length past the end of the shape");

            const double encoded = samples[packed + 1];
            const int repeats = static_cast<int>(encoded) + 2;
            if (std::fabs(encoded + 2.0 - static_cast<double>(repeats)) > 1e-6)
                throw std::runtime_error("decompress_shape: run length is not a whole number");
            if (repeats < 2 || unpacked - 1 + repeats > num_uncompressed)
                throw std::runtime_error("decompress_shape: run overruns the sample count");

            for (int i = unpacked - 1; i <= unpacked + repeats - 2; ++i)
                out[static_cast<size_t>(i)] = samples[packed - 1];
            packed += 3;
            unpacked += repeats;
        }
        if (packed == count)
        {
            if (unpacked - 1 >= num_uncompressed)
                throw std::runtime_error("decompress_shape: trailing sample overruns the count");
            out[static_cast<size_t>(unpacked) - 1] = samples[packed - 1];
        }

        // The encoding is of the derivative.
        for (int i = 1; i < num_uncompressed; ++i)
            out[static_cast<size_t>(i)] += out[static_cast<size_t>(i) - 1];

        return out;
    }

    /* ================================================================== */
    /*  The library                                                       */
    /* ================================================================== */

    int ShapeLibrary::append_raw(const double* samples, int count)
    {
        double peak = 0.0;
        for (int i = 0; i < count; ++i)
            peak = std::max(peak, std::fabs(samples[i]));
        num_uncompressed_.push_back(count);
        is_compressed_.push_back(0);
        first_.push_back(count > 0 ? samples[0] : 0.0);
        last_.push_back(count > 0 ? samples[count - 1] : 0.0);
        peak_.push_back(peak);
        return data_.append(samples, count);
    }

    int ShapeLibrary::append(int num_uncompressed, const double* samples, int count)
    {
        num_uncompressed_.push_back(num_uncompressed);
        is_compressed_.push_back(1);
        first_.push_back(std::numeric_limits<double>::quiet_NaN());
        last_.push_back(std::numeric_limits<double>::quiet_NaN());
        peak_.push_back(std::numeric_limits<double>::quiet_NaN());
        return data_.append(samples, count);
    }

    void ShapeLibrary::edge_stats(int id, double* first, double* last, double* peak) const
    {
        const size_t i = static_cast<size_t>(id) - 1;
        if (std::isnan(peak_[i]))
        {
            const int n = num_uncompressed_[i];
            const int count = data_.length(id);
            std::vector<double> whole;
            const double* w = data_.row(id);
            if (count != n)
            {
                whole = decompress_shape(w, count, n);
                w = whole.data();
            }
            double p = 0.0;
            for (int k = 0; k < n; ++k)
                p = std::max(p, std::fabs(w[k]));
            first_[i] = n > 0 ? w[0] : 0.0;
            last_[i] = n > 0 ? w[n - 1] : 0.0;
            peak_[i] = p;
        }
        *first = first_[i];
        *last = last_[i];
        *peak = peak_[i];
    }

    std::vector<int32_t> ShapeLibrary::keep_first_appearances(const std::vector<int32_t>& first)
    {
        const int total = size();
        std::vector<uint8_t> keep(static_cast<size_t>(total), 0);
        for (int id = 1; id <= total; ++id)
            keep[static_cast<size_t>(id) - 1] = first[static_cast<size_t>(id)] == id ? 1 : 0;

        std::vector<int32_t> new_id(static_cast<size_t>(total) + 1, 0);
        data_.compact(keep.data(), new_id.data());
        int kept = 0;
        for (int id = 1; id <= total; ++id)
        {
            if (!keep[static_cast<size_t>(id) - 1])
                continue;
            num_uncompressed_[static_cast<size_t>(kept)] =
                num_uncompressed_[static_cast<size_t>(id) - 1];
            is_compressed_[static_cast<size_t>(kept)] = is_compressed_[static_cast<size_t>(id) - 1];
            first_[static_cast<size_t>(kept)] = first_[static_cast<size_t>(id) - 1];
            last_[static_cast<size_t>(kept)] = last_[static_cast<size_t>(id) - 1];
            peak_[static_cast<size_t>(kept)] = peak_[static_cast<size_t>(id) - 1];
            ++kept;
        }
        num_uncompressed_.resize(static_cast<size_t>(kept));
        is_compressed_.resize(static_cast<size_t>(kept));
        first_.resize(static_cast<size_t>(kept));
        last_.resize(static_cast<size_t>(kept));
        peak_.resize(static_cast<size_t>(kept));
        for (int id = 1; id <= total; ++id)
            new_id[static_cast<size_t>(id)] =
                new_id[static_cast<size_t>(first[static_cast<size_t>(id)])];
        return new_id;
    }

    bool ShapeLibrary::compress()
    {
        bool any = false;
        for (uint8_t flag : is_compressed_)
        {
            if (!flag)
            {
                any = true;
                break;
            }
        }
        if (!any)
            return false;

        // Only a row that shrinks changes the table: the encodes are
        // independent per shape and run on every core, each handing back its
        // encoding or nothing. A library where nothing shrank keeps its
        // storage as it stands; otherwise the table is rebuilt in id order,
        // sized once, so the result is the same whatever the thread count.
        const int total = size();
        std::vector<std::vector<double>> packed(static_cast<size_t>(total));
        {
            unsigned workers = std::thread::hardware_concurrency();
            if (workers < 1)
                workers = 1;
            if (workers > 8)
                workers = 8;
            std::atomic<int> next{1};
            const auto encode = [&]()
            {
                for (;;)
                {
                    const int id = next.fetch_add(1);
                    if (id > total)
                        return;
                    if (is_compressed_[static_cast<size_t>(id) - 1])
                        continue;
                    packed[static_cast<size_t>(id) - 1] =
                        encode_if_shorter(data_.row(id), data_.length(id));
                }
            };
            if (workers == 1 || total < 16)
            {
                encode();
            }
            else
            {
                std::vector<std::thread> pool;
                for (unsigned w = 1; w < workers; ++w)
                    pool.emplace_back(encode);
                encode();
                for (auto& t : pool)
                    t.join();
            }
        }
        size_t rebuilt_values = 0;
        bool any_shrank = false;
        for (int id = 1; id <= total; ++id)
        {
            const std::vector<double>& row = packed[static_cast<size_t>(id) - 1];
            if (row.empty())
            {
                rebuilt_values += static_cast<size_t>(data_.length(id));
                continue;
            }
            rebuilt_values += row.size();
            any_shrank = true;
        }
        if (!any_shrank)
        {
            std::fill(is_compressed_.begin(), is_compressed_.end(), 1);
            return false;
        }

        RaggedTable rebuilt;
        rebuilt.reserve(total, rebuilt_values);
        for (int id = 1; id <= total; ++id)
        {
            const std::vector<double>& row = packed[static_cast<size_t>(id) - 1];
            if (row.empty())
                rebuilt.append(data_.row(id), data_.length(id));
            else
                rebuilt.append(row.data(), static_cast<int>(row.size()));
            is_compressed_[static_cast<size_t>(id) - 1] = 1;
        }
        data_ = std::move(rebuilt);
        return true;
    }

} // namespace pulseq
