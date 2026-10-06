/**
 * @file timeline.cpp
 * @brief The gradients a playout's blocks play, and the moments the Fourier
 *        engine's timeline reads off them.  See timeline.hpp.
 */

#include "fourier/timeline.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>
#include <utility>

#include "fourier/parallel.hpp"

namespace fourier
{

    namespace
    {

        /** Queries a worker takes at the least. */
        constexpr size_t kLeast = size_t(1) << 14;

        size_t threads()
        {
            return std::max<unsigned>(1, std::thread::hardware_concurrency());
        }

        /** Turn a moment in the physical frame to the logical axes. */
        void turned(const std::array<double, 9>& r, const double physical[3], double logical[3])
        {
            for (int j = 0; j < 3; ++j)
                logical[j] = physical[0] * r[j] + physical[1] * r[3 + j] + physical[2] * r[6 + j];
        }

    } // namespace

    GradientTable::GradientTable(
        std::vector<double> time_us,
        std::vector<double> value,
        std::vector<int64_t> span,
        std::vector<double> starts_us,
        std::vector<uint8_t> physical,
        const std::array<double, 9>& rotation)
        : time_(std::move(time_us)), value_(std::move(value)), starts_(std::move(starts_us)),
          span_(std::move(span)), physical_(std::move(physical)), rotation_(rotation)
    {
        const size_t n = blocks();
        start_moment_.assign(3 * (n + 1), 0.0);
        const double end = std::numeric_limits<double>::infinity();
        for (size_t b = 0; b < n; ++b)
        {
            double area[3];
            for (int axis = 0; axis < 3; ++axis)
                area[axis] = within(b, axis, end);
            double logical[3] = {area[0], area[1], area[2]};
            if (physical_[b])
                turned(rotation_, area, logical);
            for (int axis = 0; axis < 3; ++axis)
                start_moment_[3 * (b + 1) + axis] = start_moment_[3 * b + axis] + logical[axis];
        }
    }

    double GradientTable::within(size_t block, int axis, double since_us) const
    {
        const int64_t first = span_[6 * block + 2 * axis];
        const int64_t last = span_[6 * block + 2 * axis + 1];
        double area = 0.0;
        for (int64_t i = first; i + 1 < last; ++i)
        {
            const double t0 = time_[i];
            const double t1 = time_[i + 1];
            if (since_us <= t0)
                break;
            const double width = t1 - t0;
            if (width > 0.0)
            {
                const double elapsed = std::min(since_us, t1) - t0;
                const double slope = (value_[i + 1] - value_[i]) / width;
                area += elapsed * (value_[i] + 0.5 * slope * elapsed);
            }
            if (since_us <= t1)
                break;
        }
        return 1e-6 * area;
    }

    void GradientTable::moment(const int64_t* block, const double* since_us, size_t n, double* out) const
    {
        parallel(
            n, threads(), kLeast,
            [&](size_t, size_t begin, size_t end)
            {
                for (size_t q = begin; q < end; ++q)
                    moment_at(static_cast<size_t>(block[q]), since_us[q], out + 3 * q);
            });
    }

    void GradientTable::moment_at(size_t block, double since_us, double out[3]) const
    {
        double area[3];
        for (int axis = 0; axis < 3; ++axis)
            area[axis] = within(block, axis, since_us);
        double logical[3] = {area[0], area[1], area[2]};
        if (physical_[block])
            turned(rotation_, area, logical);
        for (int axis = 0; axis < 3; ++axis)
            out[axis] = start_moment_[3 * block + axis] + logical[axis];
    }

    void GradientTable::value(const int64_t* block, const double* since_us, size_t n, double* out) const
    {
        parallel(
            n, threads(), kLeast,
            [&](size_t, size_t begin, size_t end)
            {
                for (size_t q = begin; q < end; ++q)
                {
                    const auto b = static_cast<size_t>(block[q]);
                    const double at = since_us[q];
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        const int64_t first = span_[6 * b + 2 * axis];
                        const int64_t last = span_[6 * b + 2 * axis + 1];
                        double v = 0.0;
                        if (last - first == 1 && at == time_[first])
                            v = value_[first];
                        else if (last - first > 1 && at >= time_[first] && at <= time_[last - 1])
                        {
                            int64_t i = first;
                            while (i + 2 < last && at > time_[i + 1])
                                ++i;
                            const double width = time_[i + 1] - time_[i];
                            v = width > 0.0
                                    ? value_[i] + (at - time_[i]) / width * (value_[i + 1] - value_[i])
                                    : value_[i + 1];
                        }
                        out[3 * q + axis] = v;
                    }
                }
            });
    }

    double GradientTable::gradient_at(size_t block, int axis, double since_us) const
    {
        const int64_t first = span_[6 * block + 2 * axis];
        const int64_t last = span_[6 * block + 2 * axis + 1];
        if (last - first < 2 || since_us <= time_[first] || since_us >= time_[last - 1])
            return 0.0;
        const double* begin = time_.data() + first;
        const auto i = static_cast<int64_t>(std::upper_bound(begin, time_.data() + last, since_us) - begin) - 1 + first;
        const double width = time_[i + 1] - time_[i];
        return width > 0.0 ? value_[i] + (since_us - time_[i]) / width * (value_[i + 1] - value_[i]) : value_[i + 1];
    }

    namespace
    {

        /** Append to @p edges, relative to @p start, each of the ascending @p times (read through @p order where given) strictly inside the block of @p duration. */
        void add_times_within(
            const double* times, size_t count, const size_t* order, double start, double duration, std::vector<double>& edges)
        {
            for (size_t i = 0; i < count; ++i)
            {
                const double t = times[order ? order[i] : i];
                if (t >= start + duration)
                    break;
                if (t > start)
                    edges.push_back(t - start);
            }
        }

    }  // namespace

    double GradientTable::stretch(size_t block, double from_us, double width_us, const double* origin, double moment[3]) const
    {
        // Three Gauss-Legendre points on [0, 1] integrate the quartic |k|² of a
        // stretch on which every gradient is linear.
        static const double kNodes[3] = {0.5 - 0.5 * std::sqrt(0.6), 0.5, 0.5 + 0.5 * std::sqrt(0.6)};
        static const double kWeights[3] = {5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0};
        double g1[3], g3[3];
        for (int axis = 0; axis < 3; ++axis)
        {
            g1[axis] = gradient_at(block, axis, from_us + 0.25 * width_us);
            g3[axis] = gradient_at(block, axis, from_us + 0.75 * width_us);
        }
        if (physical_[block])
        {
            double t1[3], t3[3];
            turned(rotation_, g1, t1);
            turned(rotation_, g3, t3);
            std::copy(t1, t1 + 3, g1);
            std::copy(t3, t3 + 3, g3);
        }
        double slope[3], g0[3];
        for (int axis = 0; axis < 3; ++axis)
        {
            slope[axis] = (g3[axis] - g1[axis]) / (0.5 * width_us);
            g0[axis] = g1[axis] - 0.25 * width_us * slope[axis];
        }
        auto moment_after = [&](double tau, int axis)
        { return moment[axis] + 1e-6 * tau * (g0[axis] + 0.5 * slope[axis] * tau); };
        double integral = 0.0;
        if (!std::isnan(origin[0]))
            for (int q = 0; q < 3; ++q)
            {
                double squared = 0.0;
                for (int axis = 0; axis < 3; ++axis)
                {
                    const double k = moment_after(kNodes[q] * width_us, axis) - origin[axis];
                    squared += k * k;
                }
                integral += 1e-6 * width_us * kWeights[q] * squared;
            }
        for (int axis = 0; axis < 3; ++axis)
            moment[axis] = moment_after(width_us, axis);
        return integral;
    }

    void GradientTable::stretches(
        size_t block,
        const double* pulse_us,
        size_t pulses,
        const double* echo_us,
        const size_t* order,
        size_t echoes,
        std::vector<double>& edges) const
    {
        const double start = starts_[block];
        const double duration = starts_[block + 1] - start;
        edges.assign({0.0, duration});
        for (int axis = 0; axis < 3; ++axis)
            for (int64_t i = span_[6 * block + 2 * axis]; i < span_[6 * block + 2 * axis + 1]; ++i)
                if (time_[i] > 0.0 && time_[i] < duration)
                    edges.push_back(time_[i]);
        add_times_within(pulse_us, pulses, nullptr, start, duration, edges);
        add_times_within(echo_us, echoes, order, start, duration, edges);
        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    }

    void GradientTable::b_values(
        const double* pulse_us,
        const uint8_t* excites,
        const double* origins,
        size_t pulses,
        const double* echo_us,
        size_t n,
        double* out) const
    {
        const double kTwoPiSquared = 4.0 * std::acos(-1.0) * std::acos(-1.0);
        std::vector<size_t> order(n);
        for (size_t i = 0; i < n; ++i)
            order[i] = i;
        std::sort(order.begin(), order.end(), [echo_us](size_t a, size_t b) { return echo_us[a] < echo_us[b]; });
        const double* origin = origins;  // before any pulse
        double b = 0.0;
        size_t next_pulse = 0, next_echo = 0;
        std::vector<double> edges;
        for (size_t block = 0; block < blocks() && next_echo < n; ++block)
        {
            const double start = starts_[block];
            stretches(block, pulse_us + next_pulse, pulses - next_pulse, echo_us, order.data() + next_echo, n - next_echo, edges);
            double moment[3];
            for (int axis = 0; axis < 3; ++axis)
                moment[axis] = start_moment_[3 * block + axis];
            for (size_t s = 0; s < edges.size(); ++s)
            {
                const double at = start + edges[s];
                while (next_pulse < pulses && pulse_us[next_pulse] <= at)
                {
                    origin = origins + 3 * (next_pulse + 1);
                    if (excites[next_pulse])
                        b = 0.0;
                    ++next_pulse;
                }
                while (next_echo < n && echo_us[order[next_echo]] <= at)
                    out[order[next_echo++]] = std::isnan(origin[0]) ? 0.0 : kTwoPiSquared * b;
                if (s + 1 == edges.size())
                    break;
                b += stretch(block, edges[s], edges[s + 1] - edges[s], origin, moment);
            }
        }
        for (; next_echo < n; ++next_echo)
            out[order[next_echo]] = 0.0;
    }

    namespace
    {

        /** The last pulse at or before @p at in @p file, -1 where none. */
        int64_t last_pulse(const Pulses& pulses, double at, int64_t file)
        {
            const double* end = pulses.time_us + pulses.count;
            const int64_t last =
                static_cast<int64_t>(std::upper_bound(pulses.time_us, end, at) - pulses.time_us) - 1;
            return (last >= 0 && pulses.file[last] == file) ? last : -1;
        }

    } // namespace

    void read(
        const GradientTable& table,
        const Pulses& pulses,
        const Readout* readouts,
        size_t count,
        int64_t coarse,
        int64_t pathways,
        Reading* out)
    {
        parallel(
            count, threads(), 64,
            [&](size_t, size_t begin, size_t end)
            {
                std::vector<int64_t> index;
                std::vector<double> k;
                for (size_t r = begin; r < end; ++r)
                {
                    const Readout& readout = readouts[r];
                    const auto block = static_cast<size_t>(readout.block);
                    const double start = table.start_of(block);
                    const int64_t samples = std::max<int64_t>(readout.samples, 1);
                    const int64_t step = std::max<int64_t>(1, (samples + coarse - 1) / coarse);
                    // k at a sample, from the origin the last pulse leaves.
                    auto at = [&](int64_t i, double kk[3]) -> double
                    {
                        const double since = readout.delay_us + readout.dwell_us * (static_cast<double>(i) + 0.5);
                        double moment[3];
                        table.moment_at(block, since, moment);
                        const int64_t last = last_pulse(pulses, start + since, readout.file);
                        const double* origin = pulses.origins + 3 * (last + 1);
                        for (int a = 0; a < 3; ++a)
                        {
                            const double value = moment[a] - origin[a];
                            kk[a] = std::isnan(value) ? 0.0 : value;
                        }
                        return since;
                    };
                    index.clear();
                    for (int64_t i = 0; i < samples; i += step)
                        index.push_back(i);
                    index.push_back(samples - 1);
                    k.resize(3 * index.size());
                    for (size_t j = 0; j < index.size(); ++j)
                        at(index[j], &k[3 * j]);

                    Reading& reading = out[r];
                    const int64_t first = last_pulse(
                        pulses, start + readout.delay_us + 0.5 * readout.dwell_us, readout.file);
                    const bool wound = first >= 0 && !pulses.refocusing[first];
                    double norm_w = 0.0;
                    for (int a = 0; a < 3; ++a)
                    {
                        reading.winding[a] = wound ? pulses.interval[3 * first + a] : 0.0;
                        norm_w += reading.winding[a] * reading.winding[a];
                    }
                    norm_w = std::sqrt(norm_w);
                    auto distance = [&](const double* kk, int64_t n)
                    {
                        double sum = 0.0;
                        for (int a = 0; a < 3; ++a)
                        {
                            const double d = kk[a] - static_cast<double>(n) * reading.winding[a];
                            sum += d * d;
                        }
                        return std::sqrt(sum);
                    };
                    // A later pathway is read only where it passes nearer the
                    // centre than the free induction does by half the winding.
                    double free = std::numeric_limits<double>::infinity();
                    for (size_t j = 0; j < index.size(); ++j)
                        free = std::min(free, distance(&k[3 * j], 0));
                    int64_t pathway = 0;
                    double best = free;
                    for (int64_t n = 1; n < pathways; ++n)
                    {
                        double nearest = std::numeric_limits<double>::infinity();
                        for (size_t j = 0; j < index.size(); ++j)
                            nearest = std::min(nearest, distance(&k[3 * j], n));
                        if (nearest < free - 0.5 * norm_w && nearest < best)
                        {
                            best = nearest;
                            pathway = n;
                        }
                    }
                    reading.pathway = pathway;
                    for (int a = 0; a < 3; ++a)
                        reading.reach[a] = 0.0;
                    auto shifted = [&](double* kk)
                    {
                        double sum = 0.0;
                        for (int a = 0; a < 3; ++a)
                        {
                            kk[a] -= static_cast<double>(pathway) * reading.winding[a];
                            reading.reach[a] = std::max(reading.reach[a], std::abs(kk[a]));
                            sum += kk[a] * kk[a];
                        }
                        return sum;
                    };
                    int64_t centre = 0;
                    double least = std::numeric_limits<double>::infinity();
                    for (size_t j = 0; j < index.size(); ++j)
                    {
                        const double d = shifted(&k[3 * j]);
                        if (d < least)
                        {
                            least = d;
                            centre = index[j];
                        }
                    }
                    least = std::numeric_limits<double>::infinity();
                    for (int64_t i = std::max<int64_t>(0, centre - step);
                         i <= std::min<int64_t>(samples - 1, centre + step); ++i)
                    {
                        double kk[3];
                        const double since = at(i, kk);
                        const double d = shifted(kk);
                        if (d < least)
                        {
                            least = d;
                            reading.echo = i;
                            reading.echo_us = start + since;
                        }
                    }
                }
            });
    }

    void origins(
        const double* moment,
        const double* time_us,
        const int64_t* use,
        const int64_t* file,
        size_t n,
        int64_t excitation,
        int64_t refocusing,
        double* origins,
        double* precession)
    {
        const double none = std::numeric_limits<double>::quiet_NaN();
        double origin[3] = {none, none, none};
        double since = none;
        for (int axis = 0; axis < 3; ++axis)
            origins[axis] = none;
        precession[0] = none;
        for (size_t at = 0; at < n; ++at)
        {
            if (at && file[at] != file[at - 1])
            {
                origin[0] = origin[1] = origin[2] = none;
                since = none;
            }
            if (use[at] == excitation)
            {
                for (int axis = 0; axis < 3; ++axis)
                    origin[axis] = moment[3 * at + axis];
                since = time_us[at];
            }
            else if (use[at] == refocusing)
            {
                for (int axis = 0; axis < 3; ++axis)
                    origin[axis] = 2.0 * moment[3 * at + axis] - origin[axis];
                since = 2.0 * time_us[at] - since;
            }
            for (int axis = 0; axis < 3; ++axis)
                origins[3 * (at + 1) + axis] = origin[axis];
            precession[at + 1] = since;
        }
    }

} // namespace fourier
