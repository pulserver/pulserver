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
#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "bloch/bloch.hpp"
#include "bloch/events.hpp"
#include "bloch/finufft.hpp"
#include "bloch/parallel.hpp"
#include "bloch/repetitions.hpp"

namespace py = pybind11;

namespace
{
    using Complex = std::complex<double>;
    using Doubles = py::array_t<double, py::array::c_style | py::array::forcecast>;
    using Complexes = py::array_t<Complex, py::array::c_style | py::array::forcecast>;
    using SingleComplexes = py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast>;

    /** Points a worker interpolates maps at, at least. */
    constexpr size_t kPointsPerWorker = 4096;

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
        py::gil_scoped_release unlocked;
        self.set_magnetization(values.data());
    }

    py::array_t<double> positions(const bloch::Isochromats& self)
    {
        py::array_t<double> out({static_cast<py::ssize_t>(self.size()), static_cast<py::ssize_t>(3)});
        double* into = out.mutable_data();
        {
            py::gil_scoped_release unlocked;
            self.positions(into);
        }
        return out;
    }

    void set_positions(bloch::Isochromats& self, const Doubles& values)
    {
        if (values.ndim() != 2 || static_cast<size_t>(values.shape(0)) != self.size() || values.shape(1) != 3)
            throw std::invalid_argument("the positions must be (isochromats, 3)");
        const double* data = values.data();
        for (size_t i = 0; i < 3 * self.size(); ++i)
            if (!std::isfinite(data[i]))
                throw std::invalid_argument("the positions must be finite");
        py::gil_scoped_release unlocked;
        self.set_positions(data);
    }

    void precess(bloch::Isochromats& self, const Doubles& radians)
    {
        if (radians.ndim() != 1 || static_cast<size_t>(radians.shape(0)) != self.size())
            throw std::invalid_argument("precess takes one angle per isochromat");
        py::gil_scoped_release unlocked;
        self.precess(radians.data());
    }

    /** A read-only view of an array the engine owns, valid during a call. */
    template <typename T>
    py::array view(const T* data, std::vector<py::ssize_t> shape)
    {
        py::array made(py::dtype::of<T>(), std::move(shape), {}, data, py::none());
        py::detail::array_proxy(made.ptr())->flags &= ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
        return made;
    }

    /** The window as the device reads it: the engine's arrays, viewed. */
    py::dict lattice_window(const bloch::LatticeWindowRead& read)
    {
        const auto n = static_cast<py::ssize_t>(read.isochromats);
        const auto coils = static_cast<py::ssize_t>(read.coils);
        const auto segments = static_cast<py::ssize_t>(read.segments);
        const auto samples = static_cast<py::ssize_t>(read.samples);
        py::dict window;
        window["engine"] = read.engine;
        window["layout"] = read.layout;
        window["axes"] = read.axes;
        window["modes"] = view(read.modes, {read.dimensions});
        window["order"] = view(read.order, {n});
        window["starts"] = view(read.starts, {static_cast<py::ssize_t>(read.points) + 1});
        window["off_resonance"] = view(read.off_resonance, {n});
        window["decay_of"] = view(read.decay_of, {n});
        window["coils"] = read.coils;
        window["receive_re"] = read.receive_re != nullptr ? py::object(view(read.receive_re, {coils, n})) : py::none();
        window["receive_im"] = read.receive_im != nullptr ? py::object(view(read.receive_im, {coils, n})) : py::none();
        window["mx"] = view(read.mx, {n});
        window["my"] = view(read.my, {n});
        window["frequency"] = read.frequency;
        window["nodes"] = view(read.nodes, {segments});
        window["decay"] = view(read.decay, {static_cast<py::ssize_t>(read.decays), segments});
        window["basis"] = view(read.basis, {samples, segments});
        py::list x;
        for (int i = 0; i < read.dimensions; ++i)
            x.append(view(read.x[i], {samples}));
        window["x"] = x;
        window["tolerance"] = read.tolerance;
        window["single"] = read.single;
        window["out"] = py::array_t<Complex>({coils, samples}, read.out, py::none());
        return window;
    }

    /** The window read sample by sample as the device reads it: the
     *  engine's arrays, viewed. */
    py::dict sample_window(const bloch::SampleWindowRead& read)
    {
        const auto n = static_cast<py::ssize_t>(read.isochromats);
        const auto coils = static_cast<py::ssize_t>(read.coils);
        const auto samples = static_cast<py::ssize_t>(read.samples);
        py::dict window;
        window["engine"] = read.engine;
        window["layout"] = read.layout;
        window["x"] = view(read.x, {n});
        window["y"] = view(read.y, {n});
        window["z"] = view(read.z, {n});
        window["off_resonance"] = view(read.off_resonance, {n});
        window["decay_of"] = view(read.decay_of, {n});
        window["rates"] = view(read.rates, {static_cast<py::ssize_t>(read.decays)});
        window["coils"] = read.coils;
        window["receive_re"] = read.receive_re != nullptr ? py::object(view(read.receive_re, {coils, n})) : py::none();
        window["receive_im"] = read.receive_im != nullptr ? py::object(view(read.receive_im, {coils, n})) : py::none();
        window["mx"] = view(read.mx, {n});
        window["my"] = view(read.my, {n});
        window["k"] = view(read.k, {samples, 3});
        window["time"] = view(read.time, {samples});
        window["frequency"] = read.frequency;
        window["tolerance"] = read.tolerance;
        window["single"] = read.single;
        window["out"] = py::array_t<Complex>({coils, samples}, read.out, py::none());
        return window;
    }

    /** The engine's device calling @p device's @c lattice, @c samples and
     *  @c finish, each where it has one; none for None. The last reference
     *  to @p device is dropped holding the GIL, from whichever thread drops
     *  it. */
    bloch::WindowDevice window_device(const py::object& device)
    {
        if (device.is_none())
            return {};
        std::shared_ptr<py::object> held(new py::object(device), [](py::object* object) {
            py::gil_scoped_acquire acquired;
            delete object;
        });
        bloch::WindowDevice made;
        if (py::hasattr(device, "lattice"))
            made.lattice = [held](const bloch::LatticeWindowRead& read) {
                py::gil_scoped_acquire acquired;
                return py::cast<bool>(held->attr("lattice")(lattice_window(read)));
            };
        if (py::hasattr(device, "samples"))
            made.samples = [held](const bloch::SampleWindowRead& read) {
                py::gil_scoped_acquire acquired;
                return py::cast<bool>(held->attr("samples")(sample_window(read)));
            };
        if (py::hasattr(device, "finish"))
            made.finish = [held]() {
                py::gil_scoped_acquire acquired;
                held->attr("finish")();
            };
        return made;
    }

    /** A view of a run's array of reals, float in single precision and
     *  double otherwise; read-only unless @p writable. */
    py::array reals(const void* data, bool single, std::vector<py::ssize_t> shape, bool writable = false)
    {
        py::array made(
            single ? py::dtype::of<float>() : py::dtype::of<double>(), std::move(shape), {}, data, py::none());
        if (!writable)
            py::detail::array_proxy(made.ptr())->flags &= ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
        return made;
    }

    /** A run's packs, (packs, width, lanes). */
    py::array packs(const void* data, bool single, size_t slots, size_t lanes, size_t width, bool writable)
    {
        const size_t count = lanes == 0 ? 0 : (slots + lanes - 1) / lanes;
        return reals(
            data,
            single,
            {static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(width), static_cast<py::ssize_t>(lanes)},
            writable);
    }

    /** The run's carried slots as the device takes them: the run's arrays,
     *  viewed. */
    py::dict run_set(const bloch::RunSet& set)
    {
        const auto n = static_cast<py::ssize_t>(set.slots);
        const auto windows = static_cast<py::ssize_t>(set.windows);
        py::dict made;
        made["run"] = set.run;
        made["single"] = set.single;
        made["slots"] = set.slots;
        made["pack"] = packs(set.pack, set.single, set.slots, set.lanes, set.width, false);
        made["u_at"] = set.u_at;
        made["u_width"] = set.u_width;
        made["limit_at"] = set.limit_at;
        made["offsets"] = set.offsets;
        made["limits"] = set.limits;
        made["coils"] = set.coils;
        made["taps"] = set.taps;
        made["classes"] = set.classes;
        made["cells"] = view(set.cells, {windows});
        made["region"] = view(set.region, {windows + 1});
        made["turned"] = view(set.turned, {windows});
        made["start"] = view(set.start, {n, windows});
        made["weight"] = reals(set.weight, set.single, {n, windows, static_cast<py::ssize_t>(set.taps)});
        made["factor"] = reals(set.factor, set.single, {n, windows, static_cast<py::ssize_t>(set.coils), 2});
        made["decay"] = view(set.decay, {n});
        py::list index, coordinate;
        for (int axis = 0; axis < 3; ++axis)
        {
            index.append(set.index[axis] != nullptr ? py::object(view(set.index[axis], {n})) : py::none());
            coordinate.append(
                set.coordinate[axis] != nullptr ? py::object(reals(set.coordinate[axis], set.single, {n})) : py::none());
        }
        made["index"] = index;
        made["lattice"] = py::make_tuple(set.lattice[0], set.lattice[1], set.lattice[2]);
        made["coordinate"] = coordinate;
        made["origin"] = set.origin != nullptr ? py::object(view(set.origin, {n, windows})) : py::none();
        made["place"] = set.place != nullptr ? py::object(view(set.place, {n, 3})) : py::none();
        return made;
    }

    /** A tile of a run as the device carries it: the run's arrays, viewed,
     *  and its grid, written. */
    py::dict run_tile(const bloch::RunTile& tile)
    {
        constexpr auto T = static_cast<py::ssize_t>(bloch::RunTile::kRepetitions);
        py::dict made;
        made["run"] = tile.run;
        made["count"] = tile.count;
        made["turn_cos"] = reals(tile.turn_cos, tile.single, {T});
        made["turn_sin"] = reals(tile.turn_sin, tile.single, {T});
        made["netted"] = tile.netted;
        made["drop"] = tile.drop;
        const auto sets = static_cast<py::ssize_t>(tile.sets);
        made["encoding"] = view(tile.encoding, {sets, 3});
        py::list tables;
        for (size_t at = 0; at < tile.sets * 3; ++at)
        {
            const auto values = static_cast<py::ssize_t>(tile.lattice[at % 3]);
            tables.append(
                tile.encoding[at] == 1
                    ? py::object(py::make_tuple(
                          reals(tile.table_re[at], tile.single, {values, T}),
                          reals(tile.table_im[at], tile.single, {values, T})))
                    : py::none());
        }
        made["tables"] = tables;
        made["angle"] = view(tile.angle, {sets, 3, T});
        const auto windows = static_cast<py::ssize_t>(tile.windows);
        made["delta"] = view(tile.delta, {windows, 3, T});
        made["polynomials"] = reals(
            tile.polynomials,
            tile.single,
            {windows, static_cast<py::ssize_t>(tile.powers), static_cast<py::ssize_t>(tile.taps)});
        made["grid"] = reals(tile.grid, tile.single, {static_cast<py::ssize_t>(tile.grid_size)}, true);
        return made;
    }

    /** The engine's run device calling @p device's @c begin_run, @c carry,
     *  @c write_state and @c end_run, where it has the first three; none
     *  otherwise, or for None. */
    bloch::RunDevice run_device(const py::object& device)
    {
        if (device.is_none() || !py::hasattr(device, "begin_run") || !py::hasattr(device, "carry") ||
            !py::hasattr(device, "write_state"))
            return {};
        std::shared_ptr<py::object> held(new py::object(device), [](py::object* object) {
            py::gil_scoped_acquire acquired;
            delete object;
        });
        bloch::RunDevice made;
        made.begin = [held](const bloch::RunSet& set) {
            py::gil_scoped_acquire acquired;
            return py::cast<bool>(held->attr("begin_run")(run_set(set)));
        };
        made.tile = [held](const bloch::RunTile& tile) {
            py::gil_scoped_acquire acquired;
            return py::cast<size_t>(held->attr("carry")(run_tile(tile)));
        };
        made.state = [held](const bloch::RunState& state) {
            py::gil_scoped_acquire acquired;
            py::dict made_state;
            made_state["run"] = state.run;
            made_state["slots"] = state.slots;
            made_state["pack"] = packs(state.pack, state.single, state.slots, state.lanes, state.width, true);
            held->attr("write_state")(made_state);
        };
        if (py::hasattr(device, "end_run"))
            /* Called from the run's destructor, which must not throw. */
            made.release = [held](size_t run) {
                py::gil_scoped_acquire acquired;
                try
                {
                    held->attr("end_run")(run);
                }
                catch (py::error_already_set& error)
                {
                    error.discard_as_unraisable("freeing a run of repetitions on a device");
                }
            };
        return made;
    }

    void use_device(bloch::Isochromats& self, const py::object& device)
    {
        bloch::WindowDevice windows = window_device(device);
        bloch::RunDevice runs = run_device(device);
        py::gil_scoped_release unlocked;
        self.use_device(std::move(windows));
        self.use_run_device(std::move(runs));
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
        const py::object& adc,
        double tolerance)
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
            self.play(block, out, tolerance);
        }
        return signal;
    }

    /** FINUFFT's entry points, as addresses by name, and its options'
     *  layout: the struct's size and the byte offsets of its fields, by
     *  name. */
    bool use_finufft(const py::dict& entries, const py::dict& layout)
    {
        const auto entry = [&](const char* name) {
            return reinterpret_cast<void*>(py::cast<uintptr_t>(entries[name]));
        };
        const auto plans = [&](const std::string& prefix) {
            bloch::FinufftPlans made;
            made.makeplan = entry((prefix + "_makeplan").c_str());
            made.setpts = entry((prefix + "_setpts").c_str());
            made.execute = entry((prefix + "_execute").c_str());
            made.destroy = entry((prefix + "_destroy").c_str());
            return made;
        };
        const auto field = [&](const char* name) { return py::cast<size_t>(layout[name]); };
        bloch::FinufftOptions options;
        options.size = field("size");
        options.threads_at = field("nthreads");
        options.fftw_at = field("fftw");
        options.upsampling_at = field("upsampfac");
        options.warnings_at = field("showwarn");
        return bloch::use_finufft(plans("finufft"), plans("finufftf"), entry("finufft_default_opts"), options);
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
        const Doubles& readouts,
        const Doubles& nets,
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
        std::vector<double> read = values_of(readouts);
        std::vector<double> left = values_of(nets);
        py::gil_scoped_release unlocked;
        return new bloch::Repetitions(
            isochromats,
            std::move(owned),
            std::move(turns),
            std::move(adc_turns),
            std::move(encoded),
            std::move(read),
            std::move(left),
            tolerance);
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
        bloch::parallel(
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
        // Every call that takes the engine's lock lets the GIL go first: a
        // device reading a window takes the GIL while the engine holds it.
        .def_property_readonly(
            "elapsed", py::cpp_function(&bloch::Isochromats::elapsed, py::call_guard<py::gil_scoped_release>()))
        .def_property_readonly(
            "lattice_windows",
            py::cpp_function(&bloch::Isochromats::lattice_windows, py::call_guard<py::gil_scoped_release>()))
        .def_property_readonly(
            "device_windows",
            py::cpp_function(&bloch::Isochromats::device_windows, py::call_guard<py::gil_scoped_release>()))
        .def_property_readonly(
            "ungrouped_pulses",
            py::cpp_function(&bloch::Isochromats::ungrouped_pulses, py::call_guard<py::gil_scoped_release>()))
        .def("reset", &bloch::Isochromats::reset, py::call_guard<py::gil_scoped_release>())
        .def("magnetization", &magnetization)
        .def("set_magnetization", &set_magnetization)
        .def("positions", &positions)
        .def("set_positions", &set_positions)
        .def("precess", &precess, py::arg("radians"))
        .def("use_device", &use_device, py::arg("device"))
        .def("play",
             &play,
             py::arg("duration"),
             py::arg("gradients"),
             py::arg("rotation") = py::none(),
             py::arg("rf_start") = 0.0,
             py::arg("rf_step") = 0.0,
             py::arg("rf") = py::none(),
             py::arg("adc") = py::none(),
             py::arg("tolerance") = 0.0);

    py::class_<bloch::Repetitions>(module, "Repetitions")
        .def(py::init(&make_repetitions),
             py::arg("isochromats"),
             py::arg("blocks"),
             py::arg("phases"),
             py::arg("adc_phases"),
             py::arg("areas"),
             py::arg("readouts"),
             py::arg("nets"),
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
        "trilinear",
        &trilinear,
        py::arg("values"),
        py::arg("centre"),
        py::arg("step"),
        py::arg("points"),
        py::arg("out"),
        "Maps (channels, z, y, x) interpolated trilinearly at (n, 3) points into out, (n, channels).");

    module.def(
        "use_finufft",
        &use_finufft,
        py::arg("entries"),
        py::arg("layout"),
        "Take FINUFFT's entry points, addresses by name (finufft_ and finufftf_ makeplan, setpts, execute and "
        "destroy, and finufft_default_opts), and its options' size and the byte offsets of nthreads, fftw, "
        "upsampfac and showwarn; whether they were taken.");

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
