#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace
{

class Pcg32
{
  public:
    explicit Pcg32(std::uint64_t seed) : state_(0), increment_((seed << 1u) | 1u)
    {
        next();
        state_ += 0x853c49e6748fea9bULL;
        next();
    }

    std::uint32_t next()
    {
        const std::uint64_t old = state_;
        state_ = old * 6364136223846793005ULL + increment_;
        const auto xorshifted = static_cast<std::uint32_t>(((old >> 18u) ^ old) >> 27u);
        const auto rotation = static_cast<std::uint32_t>(old >> 59u);
        return (xorshifted >> rotation) | (xorshifted << ((-rotation) & 31u));
    }

    double uniform()
    {
        return std::ldexp(static_cast<double>(next()), -32);
    }

    std::uint32_t bounded(std::uint32_t bound)
    {
        if (bound == 0)
        {
            throw std::invalid_argument("random bound must be positive");
        }
        const std::uint32_t threshold = static_cast<std::uint32_t>(-bound) % bound;
        while (true)
        {
            const auto value = next();
            if (value >= threshold)
            {
                return value % bound;
            }
        }
    }

  private:
    std::uint64_t state_;
    std::uint64_t increment_;
};

struct Geometry
{
    std::size_t ny;
    std::size_t nx;
    std::size_t cy;
    std::size_t cx;
    std::vector<double> radius;
};

std::size_t checked_product(std::size_t first, std::size_t second)
{
    if (first && second > std::numeric_limits<std::size_t>::max() / first)
    {
        throw std::overflow_error("shape product overflows size_t");
    }
    return first * second;
}

Geometry make_geometry(
    const std::array<py::ssize_t, 2> &shape,
    const std::array<py::ssize_t, 2> &calib)
{
    if (shape[0] <= 0 || shape[1] <= 0)
    {
        throw std::invalid_argument("shape entries must be positive");
    }
    if (calib[0] < 0 || calib[1] < 0 || calib[0] > shape[0] || calib[1] > shape[1])
    {
        throw std::invalid_argument("calibration entries must lie within shape");
    }
    Geometry geometry{
        static_cast<std::size_t>(shape[0]),
        static_cast<std::size_t>(shape[1]),
        static_cast<std::size_t>(calib[0]),
        static_cast<std::size_t>(calib[1]),
        {}};
    const auto total = checked_product(geometry.ny, geometry.nx);
    if (total > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::overflow_error("shape is too large for the Poisson active-index table");
    }
    geometry.radius.resize(total);

    double max_x = 0.0;
    double max_y = 0.0;
    std::vector<double> x_distance(geometry.nx);
    std::vector<double> y_distance(geometry.ny);
    for (std::size_t x = 0; x < geometry.nx; ++x)
    {
        x_distance[x] = std::max(
            std::abs(static_cast<double>(x) - static_cast<double>(geometry.nx) / 2.0) -
                static_cast<double>(geometry.cx) / 2.0,
            0.0);
        max_x = std::max(max_x, x_distance[x]);
    }
    for (std::size_t y = 0; y < geometry.ny; ++y)
    {
        y_distance[y] = std::max(
            std::abs(static_cast<double>(y) - static_cast<double>(geometry.ny) / 2.0) -
                static_cast<double>(geometry.cy) / 2.0,
            0.0);
        max_y = std::max(max_y, y_distance[y]);
    }
    for (std::size_t y = 0; y < geometry.ny; ++y)
    {
        const double yn = max_y > 0.0 ? y_distance[y] / max_y : 0.0;
        for (std::size_t x = 0; x < geometry.nx; ++x)
        {
            const double xn = max_x > 0.0 ? x_distance[x] / max_x : 0.0;
            geometry.radius[y * geometry.nx + x] = std::hypot(xn, yn);
        }
    }
    return geometry;
}

void add_calibration(std::vector<std::uint8_t> &mask, const Geometry &geometry)
{
    const auto y0 = static_cast<std::size_t>(
        static_cast<double>(geometry.ny) / 2.0 - static_cast<double>(geometry.cy) / 2.0);
    const auto x0 = static_cast<std::size_t>(
        static_cast<double>(geometry.nx) / 2.0 - static_cast<double>(geometry.cx) / 2.0);
    for (std::size_t y = y0; y < y0 + geometry.cy; ++y)
    {
        for (std::size_t x = x0; x < x0 + geometry.cx; ++x)
        {
            mask[y * geometry.nx + x] = 1;
        }
    }
}

std::vector<std::uint8_t> bridson(
    const Geometry &geometry,
    double slope,
    std::size_t max_attempts,
    std::uint64_t seed,
    bool crop_corner)
{
    const auto total = checked_product(geometry.ny, geometry.nx);
    const auto maximum = static_cast<double>(std::max(geometry.nx, geometry.ny));
    std::vector<double> radius_x(total);
    std::vector<double> radius_y(total);
    for (std::size_t i = 0; i < total; ++i)
    {
        radius_x[i] = std::max((1.0 + geometry.radius[i] * slope) * geometry.nx / maximum, 1.0);
        radius_y[i] = std::max((1.0 + geometry.radius[i] * slope) * geometry.ny / maximum, 1.0);
    }

    std::vector<std::uint8_t> mask(total, 0);
    add_calibration(mask, geometry);
    std::vector<std::size_t> active_x(total);
    std::vector<std::size_t> active_y(total);
    Pcg32 rng(seed);
    active_x[0] = rng.bounded(static_cast<std::uint32_t>(geometry.nx));
    active_y[0] = rng.bounded(static_cast<std::uint32_t>(geometry.ny));
    std::size_t active_count = 1;
    constexpr double tau = 6.283185307179586476925286766559;

    while (active_count > 0 && active_count < total)
    {
        const auto active =
            static_cast<std::size_t>(rng.bounded(static_cast<std::uint32_t>(active_count)));
        const auto px = active_x[active];
        const auto py = active_y[active];
        const auto origin = py * geometry.nx + px;
        const double rx = radius_x[origin];
        const double ry = radius_y[origin];
        bool accepted = false;
        double qx = 0.0;
        double qy = 0.0;

        for (std::size_t attempt = 0; attempt < max_attempts && !accepted; ++attempt)
        {
            const double scale = std::sqrt(rng.uniform() * 3.0 + 1.0);
            const double angle = tau * rng.uniform();
            qx = static_cast<double>(px) + scale * rx * std::cos(angle);
            qy = static_cast<double>(py) + scale * ry * std::sin(angle);
            if (qx < 0.0 || qx >= geometry.nx || qy < 0.0 || qy >= geometry.ny)
            {
                continue;
            }
            const auto start_x = static_cast<std::size_t>(std::max(std::floor(qx - rx), 0.0));
            const auto start_y = static_cast<std::size_t>(std::max(std::floor(qy - ry), 0.0));
            const auto end_x = std::min(
                static_cast<std::size_t>(std::max(std::floor(qx + rx + 1.0), 0.0)),
                geometry.nx);
            const auto end_y = std::min(
                static_cast<std::size_t>(std::max(std::floor(qy + ry + 1.0), 0.0)),
                geometry.ny);
            accepted = true;
            for (std::size_t y = start_y; y < end_y && accepted; ++y)
            {
                for (std::size_t x = start_x; x < end_x; ++x)
                {
                    const auto index = y * geometry.nx + x;
                    if (!mask[index])
                    {
                        continue;
                    }
                    const double dx = (qx - static_cast<double>(x)) / radius_x[index];
                    const double dy = (qy - static_cast<double>(y)) / radius_y[index];
                    if (dx * dx + dy * dy < 1.0)
                    {
                        accepted = false;
                        break;
                    }
                }
            }
        }
        if (accepted)
        {
            const auto x = static_cast<std::size_t>(qx);
            const auto y = static_cast<std::size_t>(qy);
            active_x[active_count] = x;
            active_y[active_count] = y;
            mask[y * geometry.nx + x] = 1;
            ++active_count;
        }
        else
        {
            active_x[active] = active_x[active_count - 1];
            active_y[active] = active_y[active_count - 1];
            --active_count;
        }
    }

    if (crop_corner)
    {
        for (std::size_t i = 0; i < total; ++i)
        {
            if (geometry.radius[i] >= 1.0)
            {
                mask[i] = 0;
            }
        }
        // Calibration support is an invariant, including unusually large ACS regions.
        add_calibration(mask, geometry);
    }
    return mask;
}

std::vector<std::uint8_t> poisson(
    const Geometry &geometry,
    double acceleration,
    std::size_t max_attempts,
    double tolerance,
    std::uint64_t seed,
    bool crop_corner)
{
    double low = 0.0;
    double high = static_cast<double>(std::max(geometry.nx, geometry.ny));
    double actual = std::numeric_limits<double>::infinity();
    std::vector<std::uint8_t> best;
    const auto total = checked_product(geometry.ny, geometry.nx);
    for (int iteration = 0; iteration < 64; ++iteration)
    {
        const double slope = (low + high) / 2.0;
        auto mask = bridson(geometry, slope, max_attempts, seed, crop_corner);
        const auto count = std::count(mask.begin(), mask.end(), static_cast<std::uint8_t>(1));
        actual =
            static_cast<double>(total) / static_cast<double>(std::max<std::ptrdiff_t>(count, 1));
        best = std::move(mask);
        if (std::abs(actual - acceleration) < tolerance)
        {
            return best;
        }
        if (actual < acceleration)
        {
            low = slope;
        }
        else
        {
            high = slope;
        }
    }
    std::ostringstream message;
    message << "cannot generate Poisson mask for acceleration " << acceleration
            << " within tolerance " << tolerance << "; last acceleration was " << actual;
    throw std::invalid_argument(message.str());
}

} // namespace

PYBIND11_MODULE(_sampling_wrapper, module)
{
    module.doc() = "Deterministic C++ sampling-pattern kernels";
    module.def(
        "poisson_disc_mask",
        [](const std::array<py::ssize_t, 2> &shape,
           double acceleration,
           const std::array<py::ssize_t, 2> &calib,
           std::uint64_t seed,
           std::size_t max_attempts,
           double tolerance,
           bool crop_corner)
        {
            if (!std::isfinite(acceleration) || acceleration <= 1.0)
            {
                throw std::invalid_argument("acceleration must be finite and greater than 1");
            }
            if (!std::isfinite(tolerance) || tolerance <= 0.0 || max_attempts == 0)
            {
                throw std::invalid_argument("tolerance and max_attempts must be positive");
            }
            const auto geometry = make_geometry(shape, calib);
            std::vector<std::uint8_t> mask;
            {
                py::gil_scoped_release release;
                mask = poisson(geometry, acceleration, max_attempts, tolerance, seed, crop_corner);
            }
            py::array_t<bool> result({shape[0], shape[1]});
            auto *output = result.mutable_data();
            std::transform(
                mask.begin(),
                mask.end(),
                output,
                [](std::uint8_t value) { return value != 0; });
            return result;
        },
        py::arg("shape"),
        py::arg("acceleration"),
        py::arg("calib") = std::array<py::ssize_t, 2>{0, 0},
        py::arg("seed") = 0,
        py::arg("max_attempts") = 30,
        py::arg("tolerance") = 0.1,
        py::arg("crop_corner") = true);
}
