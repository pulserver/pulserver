/**
 * @file bindings.cpp
 * @brief Python bindings of the Fourier engine's compiled parts.
 */

#include "fourier/bindings.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "fourier/timeline.hpp"

namespace py = pybind11;

namespace
{

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
            "``(n, 3)`` the gradient at times from block starts, in Hz/m, in the frame each block plays in.");

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
}
