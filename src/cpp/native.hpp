/**
 * @file native.hpp
 * @brief What the extension's translation units share: failures raised to
 *        Python, arrays handed to it, and arrays the C library allocated.
 */

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
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

/* The RF centre the cache records at a segment position, from the block's
 * start, in us; NaN without RF. */
inline float recorded_rf_centre_us(
    const pulseg_collection *coll, int seg, int blk, const pulseg_block_info &b)
{
    const float isocentre = pulseg_get_rf_isocenter_us(coll, seg, blk);
    if (isocentre < 0.0f)
        return std::numeric_limits<float>::quiet_NaN();
    return isocentre - static_cast<float>(b.start_time_us);
}

/* The RF pulse a segment position plays, at unit amplitude: its samples,
 * timed from the block's start, and the magnitude shape and the phase shape
 * of its definition, the phase in radians.  The channels of a pTx pulse
 * follow one another, each over the one time base. */
struct RfShape
{
    std::vector<float> time_us, magnitude, phase_rad;

    RfShape(const pulseg_collection *coll, int segment, int position, const pulseg_block_info &b)
    {
        Channels mag, phase;
        int samples = 0;
        int phase_samples = 0;
        mag.samples = pulseg_get_rf_magnitude(coll, &mag.count, &samples, segment, position);
        phase.samples =
            pulseg_get_rf_phase(coll, &phase.count, &phase_samples, segment, position);
        Waveform time;
        time.samples = pulseg_get_rf_time_us(coll, segment, position);
        if (!mag.samples || samples <= 0 || !time.samples)
            throw std::runtime_error("cannot read the RF pulse a block plays");
        const bool phased = phase.samples && phase.count == mag.count && phase_samples == samples;
        const float delay = static_cast<float>(std::max(b.rf_delay_us, 0));
        for (int c = 0; c < mag.count; ++c)
            for (int i = 0; i < samples; ++i)
            {
                time_us.push_back(delay + time.samples[i]);
                magnitude.push_back(mag.samples[c][i]);
                phase_rad.push_back(
                    static_cast<float>(2.0 * M_PI) * (phased ? phase.samples[c][i] : 0.0f));
            }
    }

    /* Append its samples at @p amplitude. */
    void play(
        float amplitude,
        std::vector<float> &times,
        std::vector<std::complex<float>> &values) const
    {
        times.insert(times.end(), time_us.begin(), time_us.end());
        for (size_t i = 0; i < magnitude.size(); ++i)
            values.push_back(std::polar(amplitude * magnitude[i], phase_rad[i]));
    }
};

/* Append the RF pulse a segment position plays at @p amplitude. */
inline void played_rf(
    const pulseg_collection *coll,
    int segment,
    int position,
    float amplitude,
    const pulseg_block_info &b,
    std::vector<float> &times,
    std::vector<std::complex<float>> &values)
{
    RfShape(coll, segment, position, b).play(amplitude, times, values);
}

/* Append the phase modulation of the ADC the block at the cursor plays, one
 * phase per sample in radians; nothing when it carries none. */
inline void played_modulation(const pulseg_collection *coll, std::vector<float> &phases)
{
    Waveform phase;
    const int samples = pulseg_get_cursor_adc_phase_modulation(coll, &phase.samples);
    if (samples < 0)
        throw std::runtime_error("cannot read the phase modulation a readout plays");
    phases.insert(phases.end(), phase.samples, phase.samples + samples);
}

/* The waves of a cache, each materialised once: every axis of wave w
 * of subsequence s, normalised, timed from its block's start. */
class Waves
{
  public:
    explicit Waves(const pulseg_collection *coll) : coll_(coll) {}

    /* Append one axis of wave @p wave played at @p amplitude.  The position
     * the block plays at reserves @p reserved points for it. */
    void append(
        int subsequence,
        int wave,
        int axis,
        float amplitude,
        int reserved,
        std::vector<float> &times,
        std::vector<float> &values)
    {
        const Axes &axes = get(subsequence, wave);
        const auto &t = axes[static_cast<size_t>(axis)].first;
        const auto &a = axes[static_cast<size_t>(axis)].second;
        if (static_cast<int>(t.size()) > reserved)
            throw std::runtime_error(
                "a block plays a wave of " + std::to_string(t.size()) +
                " points at a segment position reserving " + std::to_string(reserved));
        for (size_t i = 0; i < t.size(); ++i)
        {
            times.push_back(t[i]);
            values.push_back(amplitude * a[i]);
        }
    }

  private:
    using Axis = std::pair<std::vector<float>, std::vector<float>>;
    using Axes = std::array<Axis, 3>;

    const Axes &get(int subsequence, int wave)
    {
        const auto key = std::make_pair(subsequence, wave);
        auto found = cache_.find(key);
        if (found != cache_.end())
            return found->second;
        Axes axes;
        for (int axis = 0; axis < 3; ++axis)
        {
            int points = 0;
            require(
                pulseg_materialize_wave(
                    coll_, subsequence, wave, axis, nullptr, nullptr, 0, &points, nullptr),
                "wave");
            Axis &out = axes[static_cast<size_t>(axis)];
            out.first.resize(static_cast<size_t>(points));
            out.second.resize(static_cast<size_t>(points));
            require(
                pulseg_materialize_wave(
                    coll_, subsequence, wave, axis, out.first.data(), out.second.data(), points,
                    &points, nullptr),
                "wave");
        }
        return cache_.emplace(key, std::move(axes)).first->second;
    }

    const pulseg_collection *coll_;
    std::map<std::pair<int, int>, Axes> cache_;
};

} // namespace native
