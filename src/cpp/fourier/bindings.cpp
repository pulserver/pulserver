/**
 * @file bindings.cpp
 * @brief Python bindings of the Fourier engine's compiled parts.
 */

#include "fourier/bindings.hpp"

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#include "fourier/parallel.hpp"
#include "fourier/timeline.hpp"

namespace py = pybind11;

namespace
{

    using Complex = std::complex<double>;
    using Doubles = py::array_t<double, py::array::c_style | py::array::forcecast>;
    using SingleComplexes = py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast>;

    /** Points one worker interpolates at least. */
    constexpr size_t kPointsPerWorker = 4096;

    /** An axis of a grid at a point: its two samples and their weights. */
    struct Straddle
    {
        size_t low;
        size_t high;
        double weight[2];
    };

    Straddle straddle(double position, double step, double centre, size_t size)
    {
        const double index = std::min(std::max(position / step + centre, 0.0), static_cast<double>(size) - 1.0);
        const size_t low = std::min(static_cast<size_t>(std::floor(index)), size > 1 ? size - 2 : 0);
        const double above = index - static_cast<double>(low);
        return {low, std::min(low + 1, size - 1), {1.0 - above, above}};
    }

    /** Maps, (channels, z, y, x), and where their samples lie: sample i
     *  along an axis (i - centre) * step m from the isocentre, @c centre and
     *  @c step along x, y and z. */
    struct MapGrid
    {
        const std::complex<float>* values;
        size_t channels;
        size_t size[3];
        const double* centre;
        const double* step;
    };

    MapGrid map_grid(const SingleComplexes& values, const Doubles& centre, const Doubles& step)
    {
        if (values.ndim() != 4)
            throw std::invalid_argument("the maps must be (channels, z, y, x)");
        if (centre.ndim() != 1 || centre.shape(0) != 3 || step.ndim() != 1 || step.shape(0) != 3)
            throw std::invalid_argument("the grid's centre and step must hold three values, along x, y and z");
        MapGrid grid{values.data(),
                     static_cast<size_t>(values.shape(0)),
                     {static_cast<size_t>(values.shape(1)),
                      static_cast<size_t>(values.shape(2)),
                      static_cast<size_t>(values.shape(3))},
                     centre.data(),
                     step.data()};
        if (grid.size[0] == 0 || grid.size[1] == 0 || grid.size[2] == 0)
            throw std::invalid_argument("the maps must hold a sample along each axis");
        return grid;
    }

    /** @p grid interpolated trilinearly at @p point, along x, y and z in m,
     *  into @p re and @p im, a value per channel; a point beyond the grid
     *  takes its edge value. Each weight is rounded to single precision and
     *  the eight corners are summed in single precision, z slowest and x
     *  fastest, as ``_coils._trilinear`` sums them. */
    void interpolate(const MapGrid& grid, const double* point, float* re, float* im)
    {
        Straddle axes[3];
        for (size_t k = 0; k < 3; ++k)
            axes[k] = straddle(point[2 - k], grid.step[2 - k], grid.centre[2 - k], grid.size[k]);
        std::fill(re, re + grid.channels, 0.0f);
        std::fill(im, im + grid.channels, 0.0f);
        const size_t per_channel = grid.size[0] * grid.size[1] * grid.size[2];
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 2; ++x)
                {
                    const float weight =
                        static_cast<float>(axes[0].weight[z] * axes[1].weight[y] * axes[2].weight[x]);
                    const size_t row = (z ? axes[0].high : axes[0].low) * grid.size[1] + (y ? axes[1].high : axes[1].low);
                    const std::complex<float>* sample = grid.values + row * grid.size[2] + (x ? axes[2].high : axes[2].low);
                    for (size_t c = 0; c < grid.channels; ++c, sample += per_channel)
                    {
                        const float real = weight * sample->real();
                        const float imaginary = weight * sample->imag();
                        re[c] += real;
                        im[c] += imaginary;
                    }
                }
    }

    /** @p values, (channels, z, y, x), interpolated trilinearly at @p points,
     *  (n, 3) along x, y and z in m, into @p out, (n, channels), on every
     *  core, as interpolate() interpolates them. */
    void trilinear(
        const SingleComplexes& values,
        const Doubles& centre,
        const Doubles& step,
        const Doubles& points,
        py::array_t<Complex, py::array::c_style> out)
    {
        const MapGrid grid = map_grid(values, centre, step);
        if (points.ndim() != 2 || points.shape(1) != 3)
            throw std::invalid_argument("the points must be (n, 3)");
        const size_t count = static_cast<size_t>(points.shape(0));
        if (out.ndim() != 2 || static_cast<size_t>(out.shape(0)) != count ||
            static_cast<size_t>(out.shape(1)) != grid.channels)
            throw std::invalid_argument("out must be (points, channels)");
        const double* at = points.data();
        Complex* into = out.mutable_data();
        py::gil_scoped_release unlocked;
        fourier::parallel(
            count, std::max(1u, std::thread::hardware_concurrency()), kPointsPerWorker,
            [&](size_t, size_t first, size_t last) {
                std::vector<float> re(grid.channels), im(grid.channels);
                for (size_t n = first; n < last; ++n)
                {
                    interpolate(grid, at + 3 * n, re.data(), im.data());
                    for (size_t c = 0; c < grid.channels; ++c)
                        into[n * grid.channels + c] = Complex(re[c], im[c]);
                }
            });
    }


    template <typename T>
    std::vector<T> copied(const py::array_t<T, py::array::c_style | py::array::forcecast>& values)
    {
        return std::vector<T>(values.data(), values.data() + values.size());
    }

    using Blocks = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
    using Times = py::array_t<double, py::array::c_style | py::array::forcecast>;

    /** Run @p query over (block, since_us) pairs into an (n, 3) array. */
    template <typename Query>
    py::array_t<double> queried(const Blocks& block, const Times& since_us, size_t blocks, Query query)
    {
        if (block.size() != since_us.size())
            throw std::invalid_argument("block and since_us differ in length");
        const auto n = static_cast<size_t>(block.size());
        const int64_t* b = block.data();
        for (size_t q = 0; q < n; ++q)
            if (b[q] < 0 || static_cast<size_t>(b[q]) >= blocks)
                throw std::out_of_range("a block outside the playout");
        py::array_t<double> out({static_cast<py::ssize_t>(n), py::ssize_t(3)});
        double* data = out.mutable_data();
        {
            py::gil_scoped_release released;
            query(b, since_us.data(), n, data);
        }
        return out;
    }

} // namespace

void bind_fourier(py::module_& module)
{
    py::class_<fourier::GradientTable>(module, "GradientTable", R"doc(
Each block's gradient along each axis, linear between its corners and zero
outside them, and the moment from the start of the scan along the logical
axes; a block that plays in the physical frame is turned back by ``rotation``.
)doc")
        .def(
            py::init(
                [](const py::array_t<double, py::array::c_style | py::array::forcecast>& time_us,
                   const py::array_t<double, py::array::c_style | py::array::forcecast>& value,
                   const py::array_t<int64_t, py::array::c_style | py::array::forcecast>& span,
                   const py::array_t<double, py::array::c_style | py::array::forcecast>& starts_us,
                   const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& physical,
                   const py::array_t<double, py::array::c_style | py::array::forcecast>& rotation)
                {
                    if (span.size() != 6 * physical.size() || starts_us.size() != physical.size() + 1)
                        throw std::invalid_argument("span, starts_us and physical disagree on the blocks");
                    if (rotation.size() != 9)
                        throw std::invalid_argument("rotation is not 3 x 3");
                    std::array<double, 9> turn{};
                    std::copy(rotation.data(), rotation.data() + 9, turn.begin());
                    return fourier::GradientTable(
                        copied(time_us), copied(value), copied(span), copied(starts_us),
                        copied(physical), turn);
                }),
            py::arg("time_us"), py::arg("value"), py::arg("span"), py::arg("starts_us"),
            py::arg("physical"), py::arg("rotation"))
        .def(
            "start_moments",
            [](const fourier::GradientTable& table)
            {
                const auto& moments = table.start_moments();
                py::array_t<double> out(
                    {static_cast<py::ssize_t>(table.blocks() + 1), py::ssize_t(3)});
                std::copy(moments.begin(), moments.end(), out.mutable_data());
                return out;
            },
            "``(blocks + 1, 3)`` the moment at each block's start and at the scan's end, in 1/m.")
        .def(
            "moment",
            [](const fourier::GradientTable& table, const Blocks& block, const Times& since_us)
            {
                return queried(
                    block, since_us, table.blocks(),
                    [&table](const int64_t* b, const double* t, size_t n, double* out)
                    { table.moment(b, t, n, out); });
            },
            py::arg("block"), py::arg("since_us"),
            "``(n, 3)`` the moment from the start of the scan to times from block starts, in 1/m.")
        .def(
            "value",
            [](const fourier::GradientTable& table, const Blocks& block, const Times& since_us)
            {
                return queried(
                    block, since_us, table.blocks(),
                    [&table](const int64_t* b, const double* t, size_t n, double* out)
                    { table.value(b, t, n, out); });
            },
            py::arg("block"), py::arg("since_us"),
            "``(n, 3)`` the gradient at times from block starts, in Hz/m, in the frame each block plays in.")
        .def(
            "b_values",
            [](const fourier::GradientTable& table,
               const Times& pulse_us,
               const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& excites,
               const py::array_t<double, py::array::c_style | py::array::forcecast>& origins,
               const Times& echo_us)
            {
                if (excites.size() != pulse_us.size() || origins.size() != 3 * (pulse_us.size() + 1))
                    throw std::invalid_argument("pulse_us, excites and origins disagree on the pulses");
                const auto n = static_cast<size_t>(echo_us.size());
                py::array_t<double> out(static_cast<py::ssize_t>(n));
                {
                    py::gil_scoped_release released;
                    table.b_values(
                        pulse_us.data(), excites.data(), origins.data(),
                        static_cast<size_t>(pulse_us.size()), echo_us.data(), n, out.mutable_data());
                }
                return out;
            },
            py::arg("pulse_us"), py::arg("excites"), py::arg("origins"), py::arg("echo_us"),
            "``(n,)`` the b-value at each echo, in s/m², of k measured from the origins each pulse leaves, "
            "``(pulses + 1, 3)``, since the last excitation; pulses in play order.");

    module.def(
        "read",
        [](const fourier::GradientTable& table,
           const Times& pulse_time_us,
           const Blocks& pulse_file,
           const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& refocusing,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& interval,
           const py::array_t<double, py::array::c_style | py::array::forcecast>& origins,
           const Blocks& block,
           const Blocks& file,
           const Blocks& samples,
           const Times& dwell_us,
           const Times& delay_us,
           int64_t coarse,
           int64_t pathways)
        {
            const auto pulses = static_cast<size_t>(pulse_time_us.size());
            if (pulse_file.size() != pulse_time_us.size() || refocusing.size() != pulse_time_us.size() ||
                interval.size() != 3 * pulse_time_us.size() ||
                origins.size() != 3 * (pulse_time_us.size() + 1))
                throw std::invalid_argument("the pulses' arrays disagree");
            const auto n = static_cast<size_t>(block.size());
            if (file.size() != block.size() || samples.size() != block.size() ||
                dwell_us.size() != block.size() || delay_us.size() != block.size())
                throw std::invalid_argument("the readouts' arrays disagree");
            std::vector<fourier::Readout> readouts(n);
            for (size_t r = 0; r < n; ++r)
            {
                if (block.data()[r] < 0 || static_cast<size_t>(block.data()[r]) >= table.blocks())
                    throw std::out_of_range("a readout outside the playout");
                readouts[r] = {block.data()[r], file.data()[r], samples.data()[r],
                               dwell_us.data()[r], delay_us.data()[r]};
            }
            const fourier::Pulses held{pulse_time_us.data(), pulse_file.data(), refocusing.data(),
                                       interval.data(), origins.data(), pulses};
            std::vector<fourier::Reading> readings(n);
            {
                py::gil_scoped_release released;
                fourier::read(table, held, readouts.data(), n, coarse, pathways, readings.data());
            }
            py::array_t<int64_t> echo(static_cast<py::ssize_t>(n));
            py::array_t<double> echo_us(static_cast<py::ssize_t>(n));
            py::array_t<double> reach({static_cast<py::ssize_t>(n), py::ssize_t(3)});
            py::array_t<int64_t> pathway(static_cast<py::ssize_t>(n));
            py::array_t<double> winding({static_cast<py::ssize_t>(n), py::ssize_t(3)});
            for (size_t r = 0; r < n; ++r)
            {
                echo.mutable_data()[r] = readings[r].echo;
                echo_us.mutable_data()[r] = readings[r].echo_us;
                pathway.mutable_data()[r] = readings[r].pathway;
                for (int a = 0; a < 3; ++a)
                {
                    reach.mutable_data()[3 * r + a] = readings[r].reach[a];
                    winding.mutable_data()[3 * r + a] = readings[r].winding[a];
                }
            }
            return py::make_tuple(echo, echo_us, reach, pathway, winding);
        },
        py::arg("table"), py::arg("pulse_time_us"), py::arg("pulse_file"), py::arg("refocusing"),
        py::arg("interval"), py::arg("origins"), py::arg("block"), py::arg("file"),
        py::arg("samples"), py::arg("dwell_us"), py::arg("delay_us"), py::arg("coarse"),
        py::arg("pathways"),
        R"doc(Return what is read off each readout: its echo, the sample nearest the centre
of k-space, and that sample's time in µs from the start of the scan; the widest
|k| along each axis ``(n, 3)``; the pathway it reads; and the moment the
interval it lies in winds ``(n, 3)``. Each readout is read off ``coarse`` of its
samples, evenly spaced, and its last; a pathway after the free induction only
where it passes nearer the centre by half the winding; the echo among the
samples within a step of the nearest of those.)doc");

    module.def(
        "origins",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& moment,
           const Times& time_us,
           const Blocks& use,
           const Blocks& file,
           int64_t excitation,
           int64_t refocusing)
        {
            const auto n = static_cast<size_t>(time_us.size());
            if (moment.size() != 3 * time_us.size() || use.size() != time_us.size() ||
                file.size() != time_us.size())
                throw std::invalid_argument("moment, time_us, use and file disagree on the pulses");
            py::array_t<double> origins({static_cast<py::ssize_t>(n + 1), py::ssize_t(3)});
            py::array_t<double> precession(static_cast<py::ssize_t>(n + 1));
            fourier::origins(
                moment.data(), time_us.data(), use.data(), file.data(), n, excitation, refocusing,
                origins.mutable_data(), precession.mutable_data());
            return py::make_tuple(origins, precession);
        },
        py::arg("moment"), py::arg("time_us"), py::arg("use"), py::arg("file"),
        py::arg("excitation"), py::arg("refocusing"),
        R"doc(Return the moment k is measured from and the time precession is measured from
after each pulse, ``(pulses + 1, 3)`` and ``(pulses + 1,)``: an excitation sets
both, a refocusing pulse mirrors both about its own, a new file of a chain
forgets them, as NaN.)doc");

    module.def(
        "trilinear",
        &trilinear,
        py::arg("values"),
        py::arg("centre"),
        py::arg("step"),
        py::arg("points"),
        py::arg("out"),
        "Maps (channels, z, y, x) interpolated trilinearly at (n, 3) points into out, (n, channels).");
}
