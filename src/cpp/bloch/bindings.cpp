/**
 * @file bindings.cpp
 * @brief The isochromat engine, bound as `pulserver._ext.bloch`.
 */

#include "bloch/bindings.hpp"

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "bloch/bloch.hpp"
#include "bloch/events.hpp"
#include "bloch/repetitions.hpp"

namespace py = pybind11;

namespace
{
    using Complex = std::complex<double>;
    using Doubles = py::array_t<double, py::array::c_style | py::array::forcecast>;
    using Complexes = py::array_t<Complex, py::array::c_style | py::array::forcecast>;

    std::vector<double> column(const Doubles& values, size_t count, const char* name)
    {
        if (values.ndim() != 1 || static_cast<size_t>(values.shape(0)) != count)
            throw std::invalid_argument(std::string(name) + " must hold one value per isochromat");
        return std::vector<double>(values.data(), values.data() + count);
    }

    /** Sensitivities as (isochromats, channels), the given array itself where
     *  it is C-contiguous complex128; an empty array for none. */
    Complexes sensitivities(const py::object& given, size_t count, size_t& channels, const char* name)
    {
        channels = 0;
        if (given.is_none())
            return Complexes();
        Complexes values = py::cast<Complexes>(given);
        if (values.ndim() != 2 || static_cast<size_t>(values.shape(0)) != count || values.shape(1) < 1)
            throw std::invalid_argument(
                std::string(name) + " must be (isochromats, channels) with at least one channel");
        channels = static_cast<size_t>(values.shape(1));
        return values;
    }

    bloch::Isochromats* make_isochromats(
        const Doubles& positions,
        const Doubles& proton_density,
        const Doubles& t1,
        const Doubles& t2,
        const Doubles& off_resonance,
        const py::object& transmit,
        const py::object& receive,
        size_t threads)
    {
        if (positions.ndim() != 2 || positions.shape(1) != 3)
            throw std::invalid_argument("positions must be (isochromats, 3)");
        const size_t count = static_cast<size_t>(positions.shape(0));
        bloch::IsochromatProperties properties;
        properties.x.resize(count);
        properties.y.resize(count);
        properties.z.resize(count);
        const double* xyz = positions.data();
        for (size_t i = 0; i < count; ++i)
        {
            properties.x[i] = xyz[3 * i];
            properties.y[i] = xyz[3 * i + 1];
            properties.z[i] = xyz[3 * i + 2];
        }
        properties.proton_density = column(proton_density, count, "proton_density");
        properties.t1 = column(t1, count, "t1");
        properties.t2 = column(t2, count, "t2");
        properties.off_resonance = column(off_resonance, count, "off_resonance");
        const Complexes transmitted = sensitivities(transmit, count, properties.transmit_channels, "transmit");
        properties.transmit.assign(transmitted.data(), transmitted.data() + transmitted.size());
        // Read in place while the isochromats are constructed, so that a
        // memory-mapped array is never copied whole into this process.
        const Complexes received = sensitivities(receive, count, properties.coils, "receive");
        properties.receive = properties.coils != 0 ? received.data() : nullptr;
        py::gil_scoped_release unlocked;
        return new bloch::Isochromats(std::move(properties), threads);
    }

    py::array_t<double> magnetization(bloch::Isochromats& self)
    {
        py::array_t<double> out({static_cast<py::ssize_t>(self.size()), static_cast<py::ssize_t>(3)});
        double* into = out.mutable_data();
        {
            py::gil_scoped_release unlocked;
            self.magnetization(into);
        }
        return out;
    }

    void set_magnetization(bloch::Isochromats& self, const Doubles& values)
    {
        if (values.ndim() != 2 || static_cast<size_t>(values.shape(0)) != self.size() || values.shape(1) != 3)
            throw std::invalid_argument("the magnetisation must be (isochromats, 3)");
        self.set_magnetization(values.data());
    }

    /** Point @p block at the corners each entry of @p gradients holds, kept
     *  alive in @p held. */
    void take_gradients(const py::sequence& gradients, std::vector<Doubles>& held, bloch::BlockEvents& block)
    {
        if (py::len(gradients) != 3)
            throw std::invalid_argument("gradients must hold the three axes");
        held.reserve(3);
        for (size_t axis = 0; axis < 3; ++axis)
        {
            const py::object given = gradients[axis];
            if (given.is_none())
                continue;
            held.push_back(py::cast<Doubles>(given));
            const Doubles& corners = held.back();
            if (corners.ndim() != 2 || corners.shape(0) != 2)
                throw std::invalid_argument("a gradient must be a (2, n) array of time over amplitude");
            const size_t count = static_cast<size_t>(corners.shape(1));
            block.gradient_times[axis] = corners.data();
            block.gradient_values[axis] = corners.data() + count;
            block.gradient_corners[axis] = count;
        }
    }

    /** Point @p block at the corners @p rotation turns its gradients into,
     *  held in @p out. */
    void turn_gradients(
        const Doubles& rotation, bloch::GradientCorners& given, bloch::GradientCorners& out, bloch::BlockEvents& block)
    {
        if (rotation.ndim() != 2 || rotation.shape(0) != 3 || rotation.shape(1) != 3)
            throw std::invalid_argument("a rotation must be a (3, 3) matrix");
        double matrix[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                matrix[i][j] = rotation.data()[3 * i + j];
        for (int axis = 0; axis < 3; ++axis)
        {
            const size_t count = block.gradient_corners[axis];
            given.times[axis].assign(block.gradient_times[axis], block.gradient_times[axis] + count);
            given.values[axis].assign(block.gradient_values[axis], block.gradient_values[axis] + count);
        }
        bloch::rotate_gradients(matrix, given, out);
        for (int axis = 0; axis < 3; ++axis)
        {
            block.gradient_times[axis] = out.times[axis].data();
            block.gradient_values[axis] = out.values[axis].data();
            block.gradient_corners[axis] = out.times[axis].size();
        }
    }

    py::array_t<Complex> play(
        bloch::Isochromats& self,
        double duration,
        const py::sequence& gradients,
        const py::object& rotation,
        double rf_start,
        double rf_step,
        const py::object& rf,
        const py::object& adc)
    {
        bloch::BlockEvents block;
        block.duration = duration;
        std::vector<Doubles> held;
        take_gradients(gradients, held, block);
        bloch::GradientCorners given;
        bloch::GradientCorners turned;
        if (!rotation.is_none())
            turn_gradients(py::cast<Doubles>(rotation), given, turned, block);
        Complexes samples;
        if (!rf.is_none())
        {
            samples = py::cast<Complexes>(rf);
            if (samples.ndim() != 2)
                throw std::invalid_argument("an RF pulse must be (channels, steps)");
            block.rf_start = rf_start;
            block.rf_step = rf_step;
            block.rf_channels = static_cast<size_t>(samples.shape(0));
            block.rf_steps = static_cast<size_t>(samples.shape(1));
            block.rf = samples.data();
        }
        Doubles times;
        if (!adc.is_none())
        {
            times = py::cast<Doubles>(adc);
            if (times.ndim() != 1)
                throw std::invalid_argument("ADC sample times must be one-dimensional");
            block.adc_times = times.data();
            block.adc_samples = static_cast<size_t>(times.shape(0));
        }
        py::array_t<Complex> signal(
            {static_cast<py::ssize_t>(self.coils()), static_cast<py::ssize_t>(block.adc_samples)});
        Complex* out = signal.mutable_data();
        {
            py::gil_scoped_release unlocked;
            self.play(block, out);
        }
        return signal;
    }

    /** The RF pulse of a repeated block, given[3] to given[5] as
     *  owned_block() takes them, where it plays one. */
    void own_pulse(const py::tuple& given, bloch::OwnedBlock& owned)
    {
        if (given[5].is_none())
            return;
        const Complexes samples = py::cast<Complexes>(given[5]);
        if (samples.ndim() != 2)
            throw std::invalid_argument("an RF pulse must be (channels, steps)");
        owned.rf_start = py::cast<double>(given[3]);
        owned.rf_step = py::cast<double>(given[4]);
        owned.rf_channels = static_cast<size_t>(samples.shape(0));
        owned.rf_steps = static_cast<size_t>(samples.shape(1));
        owned.rf.assign(samples.data(), samples.data() + samples.size());
    }

    /** The ADC window of a repeated block, given[6] and given[7] as
     *  owned_block() takes them, where it reads one. */
    void own_window(const py::tuple& given, bloch::OwnedBlock& owned)
    {
        if (given[6].is_none())
            return;
        const Doubles times = py::cast<Doubles>(given[6]);
        if (times.ndim() != 1)
            throw std::invalid_argument("ADC sample times must be one-dimensional");
        owned.adc_times.assign(times.data(), times.data() + times.size());
        owned.receiver.assign(owned.adc_times.size(), 0.0);
        if (given[7].is_none())
            return;
        const Doubles receiver = py::cast<Doubles>(given[7]);
        if (receiver.ndim() != 1 || receiver.size() != times.size())
            throw std::invalid_argument("an ADC's receiver phases must be one per sample");
        owned.receiver.assign(receiver.data(), receiver.data() + receiver.size());
    }

    /** One block's events, as play() takes them, copied into a block that
     *  owns them. */
    bloch::OwnedBlock owned_block(const py::tuple& given)
    {
        if (py::len(given) != 8)
            throw std::invalid_argument(
                "a repeated block is (duration, gradients, rotation, rf_start, rf_step, rf, adc, receiver)");
        bloch::BlockEvents block;
        block.duration = py::cast<double>(given[0]);
        std::vector<Doubles> held;
        take_gradients(py::cast<py::sequence>(given[1]), held, block);
        bloch::GradientCorners unturned;
        bloch::GradientCorners turned;
        if (!given[2].is_none())
            turn_gradients(py::cast<Doubles>(given[2]), unturned, turned, block);
        bloch::OwnedBlock owned;
        owned.duration = block.duration;
        for (int axis = 0; axis < 3; ++axis)
        {
            const size_t count = block.gradient_corners[axis];
            if (count == 0)
                continue;
            owned.gradient_times[axis].assign(block.gradient_times[axis], block.gradient_times[axis] + count);
            owned.gradient_values[axis].assign(block.gradient_values[axis], block.gradient_values[axis] + count);
        }
        own_pulse(given, owned);
        own_window(given, owned);
        return owned;
    }

    std::vector<double> values_of(const Doubles& given)
    {
        return std::vector<double>(given.data(), given.data() + given.size());
    }

    bloch::Repetitions* make_repetitions(
        bloch::Isochromats& isochromats,
        const py::sequence& blocks,
        const Doubles& phases,
        const Doubles& adc_phases,
        const Doubles& areas,
        double tolerance)
    {
        std::vector<bloch::OwnedBlock> owned;
        owned.reserve(py::len(blocks));
        std::transform(blocks.begin(), blocks.end(), std::back_inserter(owned), [](const py::handle& block) {
            return owned_block(py::reinterpret_borrow<py::tuple>(block));
        });
        std::vector<double> turns = values_of(phases);
        std::vector<double> adc_turns = values_of(adc_phases);
        std::vector<double> encoded = values_of(areas);
        py::gil_scoped_release unlocked;
        return new bloch::Repetitions(
            isochromats, std::move(owned), std::move(turns), std::move(adc_turns), std::move(encoded), tolerance);
    }

    py::array_t<Complex> play_repetitions(bloch::Repetitions& self, size_t count)
    {
        py::array_t<Complex> signal({static_cast<py::ssize_t>(count),
                                     static_cast<py::ssize_t>(self.coils()),
                                     static_cast<py::ssize_t>(self.samples())});
        Complex* out = signal.mutable_data();
        {
            py::gil_scoped_release unlocked;
            self.play(count, out);
        }
        return signal;
    }

    /** The fixed points' samples of @p window summed over the columns along
     *  the axes @p along, (*lattice sizes, coils, samples). */
    py::array_t<Complex> column_sums_of(
        const bloch::Repetitions& self, size_t window, const py::sequence& along, const Doubles& encoding)
    {
        if (encoding.ndim() != 1 || encoding.size() != 3)
            throw std::invalid_argument("the encoding must be three areas");
        if (window >= self.windows())
            throw std::invalid_argument("no ADC window " + std::to_string(window));
        std::vector<int> axes;
        std::vector<py::ssize_t> shape;
        for (const py::handle& axis : along)
        {
            axes.push_back(axis.cast<int>());
            if (axes.back() < 0 || axes.back() > 2)
                throw std::invalid_argument("an axis is 0, 1 or 2");
            shape.push_back(static_cast<py::ssize_t>(self.lattice(axes.back()).size()));
        }
        shape.push_back(static_cast<py::ssize_t>(self.coils()));
        shape.push_back(static_cast<py::ssize_t>(self.window_samples(window)));
        py::array_t<Complex> out(shape);
        Complex* at = out.mutable_data();
        {
            py::gil_scoped_release unlocked;
            self.column_sums(window, axes, encoding.data(), at);
        }
        return out;
    }

    py::array_t<double> lattice_of(const bloch::Repetitions& self, int axis)
    {
        if (axis < 0 || axis > 2)
            throw std::invalid_argument("an axis is 0, 1 or 2");
        const std::vector<double>& values = self.lattice(axis);
        return py::array_t<double>(static_cast<py::ssize_t>(values.size()), values.data());
    }

    py::tuple pulse_steps(
        const Doubles& times, const Complexes& waveform, double delay, double phase, double frequency, double raster)
    {
        if (times.ndim() != 1 || waveform.ndim() != 1)
            throw std::invalid_argument("an RF pulse's times and samples must be one-dimensional");
        const std::vector<double> t(times.data(), times.data() + times.size());
        const std::vector<Complex> w(waveform.data(), waveform.data() + waveform.size());
        bloch::PulseSteps steps;
        bloch::pulse_steps(t, w, delay, phase, frequency, raster, steps);
        py::array_t<Complex> samples(
            {static_cast<py::ssize_t>(steps.channels), static_cast<py::ssize_t>(steps.steps)});
        std::copy(steps.rf.begin(), steps.rf.end(), samples.mutable_data());
        return py::make_tuple(steps.start, steps.step, samples);
    }

    py::tuple adc_window(size_t samples, double dwell, double delay, double phase, double frequency, const Doubles& modulation)
    {
        if (modulation.ndim() != 1)
            throw std::invalid_argument("an ADC's phase modulation must be one-dimensional");
        bloch::AdcWindow window;
        bloch::adc_window(
            samples,
            dwell,
            delay,
            phase,
            frequency,
            std::vector<double>(modulation.data(), modulation.data() + modulation.size()),
            window);
        return py::make_tuple(
            py::array_t<double>(static_cast<py::ssize_t>(window.times.size()), window.times.data()),
            py::array_t<double>(static_cast<py::ssize_t>(window.receiver.size()), window.receiver.data()));
    }

} // namespace

void bind_bloch(py::module_& module)
{
    module.doc() = "Isochromats the Bloch equation carries from one block to the next";
    py::class_<bloch::Isochromats>(module, "Isochromats")
        .def(py::init(&make_isochromats),
             py::arg("positions"),
             py::arg("proton_density"),
             py::arg("t1"),
             py::arg("t2"),
             py::arg("off_resonance"),
             py::arg("transmit") = py::none(),
             py::arg("receive") = py::none(),
             py::arg("threads") = 0)
        .def_property_readonly("size", &bloch::Isochromats::size)
        .def_property_readonly("coils", &bloch::Isochromats::coils)
        .def_property_readonly("transmit_channels", &bloch::Isochromats::transmit_channels)
        .def_property_readonly("elapsed", &bloch::Isochromats::elapsed)
        .def("reset", &bloch::Isochromats::reset)
        .def("magnetization", &magnetization)
        .def("set_magnetization", &set_magnetization)
        .def("play",
             &play,
             py::arg("duration"),
             py::arg("gradients"),
             py::arg("rotation") = py::none(),
             py::arg("rf_start") = 0.0,
             py::arg("rf_step") = 0.0,
             py::arg("rf") = py::none(),
             py::arg("adc") = py::none());

    py::class_<bloch::Repetitions>(module, "Repetitions")
        .def(py::init(&make_repetitions),
             py::arg("isochromats"),
             py::arg("blocks"),
             py::arg("phases"),
             py::arg("adc_phases"),
             py::arg("areas"),
             py::arg("tolerance") = 0.0,
             py::keep_alive<1, 2>())
        .def_property_readonly("count", &bloch::Repetitions::count)
        .def_property_readonly("played", &bloch::Repetitions::played)
        .def_property_readonly("windows", &bloch::Repetitions::windows)
        .def_property_readonly("samples", &bloch::Repetitions::samples)
        .def_property_readonly("coils", &bloch::Repetitions::coils)
        .def_property_readonly("divided", &bloch::Repetitions::divided)
        .def_property_readonly("carried", &bloch::Repetitions::carried)
        .def_property_readonly("reach", &bloch::Repetitions::reach)
        .def("split", &bloch::Repetitions::split, py::call_guard<py::gil_scoped_release>())
        .def("column_sums", &column_sums_of, py::arg("window"), py::arg("axes"), py::arg("encoding"))
        .def("lattice", &lattice_of, py::arg("axis"))
        .def("play", &play_repetitions, py::arg("count"));

    module.def(
        "pulse_steps",
        &pulse_steps,
        py::arg("times"),
        py::arg("waveform"),
        py::arg("delay"),
        py::arg("phase"),
        py::arg("frequency"),
        py::arg("raster"),
        "The field an RF pulse plays, as (start, step, (channels, steps) b1 in Hz).");

    module.def(
        "adc_window",
        &adc_window,
        py::arg("samples"),
        py::arg("dwell"),
        py::arg("delay"),
        py::arg("phase"),
        py::arg("frequency"),
        py::arg("modulation"),
        "An ADC's sample times from the block's start and the phase each is demodulated by.");
}
