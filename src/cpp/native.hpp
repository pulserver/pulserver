/**
 * @file native.hpp
 * @brief What the extension's translation units share: failures raised to
 *        Python, arrays handed to it, and arrays the C library allocated.
 */

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulseg.h"

namespace native
{

namespace py = pybind11;

inline void require(int code, const char *what)
{
    if (PULSEG_FAILED(code))
        throw std::invalid_argument(std::string(what) + " failed: error " + std::to_string(code));
}

/* Raise a failure the library explained in @p diag. */
[[noreturn]] inline void raise_diagnosed(int code, const pulseg_diagnostic &diag)
{
    throw std::invalid_argument(std::string(pulseg_get_error_message(code)) + ": " + diag.message);
}

template <typename T>
py::array_t<T> as_array(const std::vector<T> &values, std::vector<py::ssize_t> shape)
{
    py::array_t<T> out(shape);
    if (!values.empty())
        std::memcpy(out.mutable_data(), values.data(), values.size() * sizeof(T));
    return out;
}

/* A waveform the library allocated: freed with it, however the caller leaves. */
struct Waveform
{
    float *samples = nullptr;
    ~Waveform() { PULSEG_FREE(samples); }
};

/* A waveform per channel the library allocated: each channel, then the array
 * of them, freed with it. */
struct Channels
{
    float **samples = nullptr;
    int count = 0;
    ~Channels()
    {
        if (!samples)
            return;
        for (int c = 0; c < count; ++c)
            PULSEG_FREE(samples[c]);
        PULSEG_FREE(samples);
    }
};

/* Append the corners of a gradient event of @p n samples @p w at times @p t
 * from its start, played @p delay_us into its block and at @p amplitude.  A
 * waveform on the gradient raster is sampled at the middle of each raster
 * interval; over the half intervals at its ends it holds its end values,
 * which keeps the area its samples give. */
inline void append_event(
    const float *t,
    const float *w,
    int n,
    float delay_us,
    float amplitude,
    std::vector<float> &times,
    std::vector<float> &values)
{
    const bool centred = n > 1 && t[0] > 0.0f && std::fabs(t[0] - 0.5f * (t[1] - t[0])) < 1e-3f;
    if (centred)
    {
        times.push_back(delay_us);
        values.push_back(amplitude * w[0]);
    }
    for (int i = 0; i < n; ++i)
    {
        times.push_back(delay_us + t[i]);
        values.push_back(amplitude * w[i]);
    }
    if (centred)
    {
        times.push_back(delay_us + t[n - 1] + 0.5f * (t[1] - t[0]));
        values.push_back(amplitude * w[n - 1]);
    }
}

} // namespace native
