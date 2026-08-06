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

#include "pulseq/shape.hpp"

#include "pulseq/sequence.hpp"

#include <cmath>
#include <vector>

namespace pulseq
{

    std::vector<double> compress_shape(const double* samples, int count)
    {
        // Very short shapes are left alone: four numbers cannot be encoded in
        // fewer than four, and Pulseq does not try.
        if (count <= 4)
            return std::vector<double>(samples, samples + count);

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
        return std::vector<double>(samples, samples + count);
    }

    /* ================================================================== */
    /*  The library                                                       */
    /* ================================================================== */

    int ShapeLibrary::append_raw(const double* samples, int count)
    {
        num_uncompressed_.push_back(count);
        is_compressed_.push_back(0);
        return data_.append(samples, count);
    }

    int ShapeLibrary::append(int num_uncompressed, const double* samples, int count)
    {
        num_uncompressed_.push_back(num_uncompressed);
        is_compressed_.push_back(1);
        return data_.append(samples, count);
    }

    void ShapeLibrary::compress()
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
            return;

        // Rebuilt rather than edited in place: a compressed row is shorter
        // than the raw one it replaces, so every row after it moves.
        RaggedTable rebuilt;
        for (int id = 1; id <= size(); ++id)
        {
            const int count = data_.length(id);
            if (is_compressed_[static_cast<size_t>(id) - 1])
            {
                rebuilt.append(data_.row(id), count);
                continue;
            }
            const std::vector<double> packed = compress_shape(data_.row(id), count);
            rebuilt.append(packed.data(), static_cast<int>(packed.size()));
            is_compressed_[static_cast<size_t>(id) - 1] = 1;
        }
        data_ = std::move(rebuilt);
    }

}  // namespace pulseq
