/**
 * @file events.cpp
 * @brief One block's events as the Bloch equation takes them.
 */

#include "bloch/events.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace bloch
{

    namespace
    {

        constexpr double kPi = 3.14159265358979323846;

        /** A nanosecond: what two corner times have to differ by to be two. */
        constexpr double kEps = 1e-9;

        /** The piecewise-linear waveform through the corners at @p when, zero
         *  outside them; a time within kEps of a corner is that corner. */
        double sampled(const std::vector<double>& times, const std::vector<double>& values, double when)
        {
            const size_t count = times.size();
            if (count == 0 || when < times.front() - kEps || when > times.back() + kEps)
                return 0.0;
            const size_t after =
                static_cast<size_t>(std::lower_bound(times.begin(), times.end(), when - kEps) - times.begin());
            if (after >= count)
                return values.back();
            const double span = after > 0 ? times[after] - times[after - 1] : 0.0;
            if (times[after] <= when + kEps || !(span > 0.0))
                return values[after];
            return values[after - 1] + (when - times[after - 1]) / span * (values[after] - values[after - 1]);
        }

        /** The waveform through the corners just before and just after
         *  @p when: a gradient is zero outside its corners, so it steps there
         *  from or to a value other than zero. */
        void limits(const std::vector<double>& times, const std::vector<double>& values, double when, double& before, double& after)
        {
            before = after = sampled(times, values, when);
            if (times.empty())
                return;
            if (std::fabs(when - times.front()) <= kEps)
                before = 0.0;
            if (std::fabs(when - times.back()) <= kEps)
                after = 0.0;
        }

        /** The sum @p row weights the given axes by, just before and just
         *  after @p when. */
        void mix(const GradientCorners& given, const double row[3], double when, double& before, double& after)
        {
            for (int from = 0; from < 3; ++from)
            {
                if (given.times[from].empty() || row[from] == 0.0)
                    continue;
                double left = 0.0;
                double right = 0.0;
                limits(given.times[from], given.values[from], when, left, right);
                before += row[from] * left;
                after += row[from] * right;
            }
        }

        void check_corners(const GradientCorners& given)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                const std::vector<double>& times = given.times[axis];
                const std::vector<double>& values = given.values[axis];
                bool valid = times.size() == values.size();
                for (size_t i = 0; valid && i < times.size(); ++i)
                    valid = std::isfinite(times[i]) && std::isfinite(values[i]) && !(i > 0 && times[i] < times[i - 1]);
                if (!valid)
                    throw std::invalid_argument("a gradient's corners must be finite, at times in increasing order");
            }
        }

        /** Every corner time of the given axes, once. */
        std::vector<double> union_of(const GradientCorners& given)
        {
            std::vector<double> merged;
            for (int axis = 0; axis < 3; ++axis)
                merged.insert(merged.end(), given.times[axis].begin(), given.times[axis].end());
            std::sort(merged.begin(), merged.end());
            merged.erase(
                std::unique(merged.begin(), merged.end(), [](double a, double b) { return std::fabs(a - b) <= kEps; }),
                merged.end());
            return merged;
        }

        /** Whether the first @p per times are the middles of equal intervals
         *  from the pulse's start, and so steps as they stand. */
        bool at_middles(const std::vector<double>& t, size_t per)
        {
            if (per < 2)
                return false;
            const double interval = t[1] - t[0];
            for (size_t i = 1; i < per; ++i)
                if (std::fabs(t[i] - t[i - 1] - interval) > 1e-9 * interval)
                    return false;
            return std::fabs(t[0] - 0.5 * interval) <= 1e-9 * interval;
        }

        /** The carrier Pulseq modulates a pulse with: its phase offset,
         *  advancing at its frequency offset from the pulse's start. */
        struct Carrier
        {
            double phase = 0.0;
            double frequency = 0.0;

            /** The pulse's field: the conjugate of Pulseq's waveform. */
            std::complex<double> field(std::complex<double> sample, double time) const
            {
                return std::conj(sample * std::polar(1.0, phase + 2.0 * kPi * frequency * time));
            }
        };

        void held_steps(
            const std::vector<double>& t,
            const std::vector<std::complex<double>>& waveform,
            size_t per,
            const Carrier& carrier,
            double delay,
            PulseSteps& out)
        {
            const double interval = t[1] - t[0];
            out.start = delay + t[0] - 0.5 * interval;
            out.step = interval;
            out.steps = per;
            out.rf.resize(out.channels * per);
            for (size_t c = 0; c < out.channels; ++c)
                for (size_t i = 0; i < per; ++i)
                    out.rf[c * per + i] = carrier.field(waveform[c * per + i], t[i]);
        }

        /** A channel's @p waveform at @p when, joined linearly between its
         *  samples; @p j is the sample at or before the previous time asked. */
        std::complex<double> joined(
            const std::vector<double>& t, const std::complex<double>* waveform, size_t per, size_t& j, double when)
        {
            while (j + 1 < per && t[j + 1] < when)
                ++j;
            if (j + 1 >= per || !(t[j + 1] > t[j]))
                return waveform[j];
            return waveform[j] + (when - t[j]) / (t[j + 1] - t[j]) * (waveform[j + 1] - waveform[j]);
        }

        /** Steps of the raster, or of the pulse's shortest interval where that
         *  is shorter, over which the joined samples are held. */
        void joined_steps(
            const std::vector<double>& t,
            const std::vector<std::complex<double>>& waveform,
            size_t per,
            const Carrier& carrier,
            double delay,
            double raster,
            PulseSteps& out)
        {
            double shortest = raster;
            for (size_t i = 1; i < per; ++i)
                shortest = t[i] > t[i - 1] ? std::min(shortest, t[i] - t[i - 1]) : shortest;
            const double span = t[per - 1] - t[0];
            const size_t steps = span > 0.0 ? static_cast<size_t>(std::max(1.0, std::ceil(span / shortest - 1e-9))) : 1;
            out.step = span > 0.0 ? span / static_cast<double>(steps) : raster;
            out.start = delay + (span > 0.0 ? t[0] : t[0] - 0.5 * raster);
            out.steps = steps;
            out.rf.resize(out.channels * steps);
            for (size_t c = 0; c < out.channels; ++c)
            {
                size_t j = 0;
                for (size_t s = 0; s < steps; ++s)
                {
                    const double when = span > 0.0 ? t[0] + (static_cast<double>(s) + 0.5) * out.step : t[0];
                    out.rf[c * steps + s] = carrier.field(joined(t, &waveform[c * per], per, j, when), when);
                }
            }
        }

        /** The channels of a dynamic pTx pulse, which holds them one after
         *  another over one time base: the number of samples at the first
         *  sample time, where the times are that many identical copies, and
         *  one otherwise. */
        size_t rf_channels(const std::vector<double>& times)
        {
            if (times.empty())
                return 1;
            const size_t count = static_cast<size_t>(std::count(times.begin(), times.end(), times[0]));
            if (count < 2 || times.size() % count != 0)
                return 1;
            const size_t per_channel = times.size() / count;
            for (size_t i = per_channel; i < times.size(); ++i)
                if (times[i] != times[i - per_channel])
                    return 1;
            return count;
        }

        void check_pulse_times(const std::vector<double>& times, size_t per)
        {
            for (size_t i = 0; i < times.size(); ++i)
                if (!std::isfinite(times[i]) || (i % per != 0 && times[i] < times[i - 1]))
                    throw std::invalid_argument("an RF pulse's sample times must be finite and in increasing order");
        }

    } // namespace

    void rotate_gradients(const double matrix[3][3], const GradientCorners& given, GradientCorners& out)
    {
        check_corners(given);
        const std::vector<double> merged = union_of(given);
        for (int into = 0; into < 3; ++into)
        {
            out.times[into].clear();
            out.values[into].clear();
            const double* row = matrix[into];
            bool anything = false;
            for (int from = 0; from < 3; ++from)
                anything = anything || (!given.times[from].empty() && row[from] != 0.0);
            for (size_t i = 0; anything && i < merged.size(); ++i)
            {
                double before = 0.0;
                double after = 0.0;
                mix(given, row, merged[i], before, after);
                if (before != after)
                {
                    out.times[into].push_back(merged[i]);
                    out.values[into].push_back(before);
                }
                out.times[into].push_back(merged[i]);
                out.values[into].push_back(after);
            }
        }
    }

    void pulse_steps(
        const std::vector<double>& times,
        const std::vector<std::complex<double>>& waveform,
        double delay,
        double phase,
        double frequency,
        double raster,
        PulseSteps& out)
    {
        if (times.size() != waveform.size())
            throw std::invalid_argument("an RF pulse needs one sample time per sample");
        if (!(raster > 0.0) || !std::isfinite(raster))
            throw std::invalid_argument("the RF raster must be positive");
        out.steps = 0;
        out.channels = rf_channels(times);
        out.rf.clear();
        const size_t per = times.size() / out.channels;
        if (per == 0)
            return;
        check_pulse_times(times, per);
        const Carrier carrier{phase, frequency};
        if (at_middles(times, per))
            held_steps(times, waveform, per, carrier, delay, out);
        else
            joined_steps(times, waveform, per, carrier, delay, raster, out);
    }

    void adc_window(
        size_t samples,
        double dwell,
        double delay,
        double phase,
        double frequency,
        const std::vector<double>& modulation,
        AdcWindow& out)
    {
        out.times.resize(samples);
        out.receiver.resize(samples);
        for (size_t k = 0; k < samples; ++k)
        {
            const double within = dwell * (static_cast<double>(k) + 0.5);
            out.times[k] = delay + within;
            out.receiver[k] = phase + (k < modulation.size() ? modulation[k] : 0.0) + 2.0 * kPi * frequency * within;
        }
    }

} // namespace bloch
