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

#include "bloch/parallel.hpp"

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
        bloch::parallel(
            n, threads(), kLeast,
            [&](size_t, size_t begin, size_t end)
            {
                for (size_t q = begin; q < end; ++q)
                {
                    const auto b = static_cast<size_t>(block[q]);
                    double area[3];
                    for (int axis = 0; axis < 3; ++axis)
                        area[axis] = within(b, axis, since_us[q]);
                    double logical[3] = {area[0], area[1], area[2]};
                    if (physical_[b])
                        turned(rotation_, area, logical);
                    for (int axis = 0; axis < 3; ++axis)
                        out[3 * q + axis] = start_moment_[3 * b + axis] + logical[axis];
                }
            });
    }

    void GradientTable::value(const int64_t* block, const double* since_us, size_t n, double* out) const
    {
        bloch::parallel(
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
