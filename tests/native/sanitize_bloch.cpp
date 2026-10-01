/**
 * @file sanitize_bloch.cpp
 * @brief Drive the isochromat engine through each of its paths with the
 *        sanitisers watching.
 *
 * A read past the end of an array is not a wrong answer but whatever
 * happened to be there, which the numerical tests catch only when it changes
 * a number. This plays blocks that reach every path of the engine -- a pulse
 * stepped for each group of isochromats, one reused turned by its phase, one
 * computed on a grid of fields and drives, a window read sample by sample, one
 * read by the non-uniform FFT and one read on the isochromats' lattice, by the
 * engine and by a device, and again once they have moved, gradients turned
 * with steps in them, and windows off the lattice read by a device sample by
 * sample -- and
 * repetitions of them, exact and to a tolerance, on one transmit channel and
 * on two, by the engine and by a run device that carries them in plain loops,
 * with AddressSanitizer and UndefinedBehaviorSanitizer on. It asserts that the
 * windows meant for the lattice are read there and that the devices read what
 * the engine reads; the sanitisers assert the rest.
 *
 * The lattice is transformed by a stand-in for FINUFFT's plans that sums each
 * point directly, handed to the engine as the finufft wheel's entry points
 * are, so that no library outside the engine runs under the sanitisers.
 */

#include "bloch/bloch.hpp"
#include "bloch/events.hpp"
#include "bloch/finufft.hpp"
#include "bloch/repetitions.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <vector>

namespace
{

    using Complex = std::complex<double>;

    constexpr double kPi = 3.14159265358979323846;

    /** Somewhere for the results to go, so nothing is optimised away. */
    double sink = 0.0;

    void touch(const std::vector<Complex>& values)
    {
        sink = std::accumulate(
            values.begin(), values.end(), sink, [](double sum, const Complex& value) { return sum + std::abs(value); });
    }

    /** A type-2 plan summed point by point: f(x) = sum_m f_m exp(-i m . x),
     *  each mode from -modes / 2, the first dimension fastest. */
    struct DirectPlan
    {
        int dimensions = 1;
        int64_t modes[3] = {1, 1, 1};
        int vectors = 1;
        std::vector<double> points[3];
    };

    template <typename Real>
    int direct_makeplan(int, int dimensions, int64_t* modes, int, int vectors, Real, void** plan, void*)
    {
        DirectPlan* made = new DirectPlan();
        made->dimensions = dimensions;
        std::copy(modes, modes + dimensions, made->modes);
        made->vectors = vectors;
        *plan = made;
        return 0;
    }

    template <typename Real>
    int direct_setpts(void* plan, int64_t count, Real* x, Real* y, Real* z, int64_t, Real*, Real*, Real*)
    {
        DirectPlan& made = *static_cast<DirectPlan*>(plan);
        const Real* given[3] = {x, y, z};
        for (int d = 0; d < made.dimensions; ++d)
            made.points[d].assign(given[d], given[d] + count);
        return 0;
    }

    template <typename Real>
    int direct_execute(void* plan, std::complex<Real>* sums, std::complex<Real>* modes)
    {
        const DirectPlan& made = *static_cast<const DirectPlan*>(plan);
        const size_t count = made.points[0].size();
        const int64_t size = made.modes[0] * made.modes[1] * made.modes[2];
        for (int v = 0; v < made.vectors; ++v)
            for (size_t j = 0; j < count; ++j)
            {
                std::complex<double> sum = 0.0;
                for (int64_t m = 0; m < size; ++m)
                {
                    double phase = 0.0;
                    int64_t rest = m;
                    for (int d = 0; d < made.dimensions; ++d)
                    {
                        const int64_t index = rest % made.modes[d] - made.modes[d] / 2;
                        rest /= made.modes[d];
                        phase += static_cast<double>(index) * made.points[d][j];
                    }
                    sum += std::complex<double>(modes[v * size + m]) * std::polar(1.0, -phase);
                }
                sums[static_cast<size_t>(v) * count + j] = std::complex<Real>(sum);
            }
        return 0;
    }

    int direct_destroy(void* plan)
    {
        delete static_cast<DirectPlan*>(plan);
        return 0;
    }

    /** An options struct of the stand-in's own layout: FINUFFT's size and
     *  offsets, as the finufft wheel reports them. */
    constexpr size_t kOptions = 96;

    void direct_default_opts(void* options)
    {
        std::memset(options, 0, kOptions);
    }

    template <typename Real>
    bloch::FinufftPlans direct_plans()
    {
        bloch::FinufftPlans plans;
        plans.makeplan = reinterpret_cast<void*>(&direct_makeplan<Real>);
        plans.setpts = reinterpret_cast<void*>(&direct_setpts<Real>);
        plans.execute = reinterpret_cast<void*>(&direct_execute<Real>);
        plans.destroy = reinterpret_cast<void*>(&direct_destroy);
        return plans;
    }

    void use_direct_plans()
    {
        bloch::FinufftOptions options;
        options.size = kOptions;
        options.warnings_at = 16;
        options.threads_at = 20;
        options.fftw_at = 24;
        options.upsampling_at = 40;
        bloch::use_finufft(
            direct_plans<double>(), direct_plans<float>(), reinterpret_cast<void*>(&direct_default_opts), options);
    }

    /** Isochromats on a lattice, each at an off-resonance of its own. */
    bloch::IsochromatProperties lattice(size_t channels, size_t coils, std::vector<Complex>& receive)
    {
        bloch::IsochromatProperties p;
        for (int i = 0; i < 12; ++i)
            for (int j = 0; j < 10; ++j)
                for (int k = 0; k < 4; ++k)
                {
                    p.x.push_back(3e-3 * (i - 6));
                    p.y.push_back(3e-3 * (j - 5));
                    p.z.push_back(2e-3 * (k - 2));
                }
        const size_t n = p.x.size();
        for (size_t i = 0; i < n; ++i)
        {
            p.proton_density.push_back(0.5 + 0.5 * std::sin(0.37 * i));
            p.t1.push_back(i % 3 == 0 ? 0.8 : 1.2);
            p.t2.push_back(i % 2 == 0 ? 0.05 : 0.09);
            p.off_resonance.push_back(40.0 * std::sin(1.3 * i));
        }
        p.transmit_channels = channels;
        for (size_t i = 0; i < n * channels; ++i)
            p.transmit.push_back(std::polar(0.7 + 0.2 * std::cos(0.11 * i), 0.3 * i));
        receive.clear();
        for (size_t i = 0; i < n * coils; ++i)
            receive.push_back(std::polar(1.0, 0.07 * i));
        p.coils = coils;
        p.receive = receive.data();
        return p;
    }

    /** A pulse of @p steps samples of a sinc, one row per channel, each
     *  channel its own weight. */
    std::vector<Complex> sinc(size_t steps, size_t channels, double peak)
    {
        std::vector<Complex> rf(steps * channels);
        for (size_t c = 0; c < channels; ++c)
            for (size_t s = 0; s < steps; ++s)
            {
                const double x = 4.0 * (static_cast<double>(s) + 0.5) / steps - 2.0;
                const double shape = std::fabs(x) < 1e-12 ? 1.0 : std::sin(kPi * x) / (kPi * x);
                rf[c * steps + s] = std::polar(peak * shape, 0.4 * c);
            }
        return rf;
    }

    struct Axis
    {
        std::vector<double> times, values;
    };

    void set(bloch::BlockEvents& block, int axis, const std::vector<double>& times, const std::vector<double>& values)
    {
        block.gradient_times[axis] = times.data();
        block.gradient_values[axis] = values.data();
        block.gradient_corners[axis] = times.size();
    }

    void set(bloch::BlockEvents& block, int axis, const Axis& g)
    {
        set(block, axis, g.times, g.values);
    }

    /** A window on a spiral in the xy plane, which the block points into. */
    struct Spiral
    {
        Axis x, y;
        std::vector<double> times;
        bloch::BlockEvents block;

        Spiral()
        {
            for (int k = 0; k <= 80; ++k)
            {
                const double t = 5e-3 * k / 80.0;
                const Complex g = std::polar(3e4 * t / 5e-3, 6.0 * kPi * t / 5e-3);
                x.times.push_back(t);
                x.values.push_back(g.real());
                y.times.push_back(t);
                y.values.push_back(g.imag());
            }
            for (int k = 0; k < 400; ++k)
                times.push_back(0.05e-3 + 12e-6 * k);
            block.duration = 5e-3;
            set(block, 0, x);
            set(block, 1, y);
            block.adc_times = times.data();
            block.adc_samples = times.size();
        }
        Spiral(const Spiral&) = delete;
        Spiral& operator=(const Spiral&) = delete;
    };

    /** One of each kind of block, played in turn. */
    void blocks(bloch::Isochromats& spins, size_t channels)
    {
        const size_t rows = channels ? channels : 1;
        std::vector<Complex> signal;
        auto play = [&](const bloch::BlockEvents& block) {
            signal.assign(spins.coils() * block.adc_samples, 0.0);
            spins.play(block, signal.data());
            touch(signal);
        };

        // A pulse under a slice-selection gradient held throughout it: the grid.
        const std::vector<Complex> selective = sinc(200, rows, 800.0);
        const Axis held_z{{0.0, 0.1e-3, 2.1e-3, 2.2e-3}, {0.0, 2e4, 2e4, 0.0}};
        bloch::BlockEvents excite;
        excite.duration = 2.3e-3;
        set(excite, 2, held_z);
        excite.rf_start = 0.1e-3;
        excite.rf_step = 1e-5;
        excite.rf_steps = 200;
        excite.rf_channels = rows;
        excite.rf = selective.data();
        play(excite);

        // The same pulse turned by a phase: its maps, reused.
        std::vector<Complex> turned(selective.size());
        std::transform(selective.begin(), selective.end(), turned.begin(), [](const Complex& value) {
            return value * std::polar(1.0, 1.1);
        });
        excite.rf = turned.data();
        play(excite);

        // A pulse under gradients that change during it, on two axes: stepped.
        const Axis ramp_x{{0.0, 0.5e-3, 1.0e-3}, {0.0, 1e4, -5e3}};
        const Axis ramp_y{{0.0, 1.0e-3}, {3e3, -3e3}};
        bloch::BlockEvents stepped;
        stepped.duration = 1.0e-3;
        set(stepped, 0, ramp_x);
        set(stepped, 1, ramp_y);
        const std::vector<Complex> hard = sinc(50, rows, 300.0);
        stepped.rf_start = 0.25e-3;
        stepped.rf_step = 1e-5;
        stepped.rf_steps = 50;
        stepped.rf_channels = rows;
        stepped.rf = hard.data();
        play(stepped);

        // Gradients turned by a rotation, one of them stepping mid-block.
        bloch::GradientCorners given;
        given.times[0] = {0.2e-3, 0.8e-3};
        given.values[0] = {1e4, 1e4};
        given.times[1] = {0.0, 0.1e-3, 0.9e-3, 1.0e-3};
        given.values[1] = {0.0, 5e3, 5e3, 0.0};
        const double matrix[3][3] = {{0.5, -0.8, 0.3}, {0.8, 0.5, 0.0}, {-0.3, 0.1, 0.9}};
        bloch::GradientCorners out;
        bloch::rotate_gradients(matrix, given, out);
        bloch::BlockEvents rotated;
        rotated.duration = 1.0e-3;
        for (int axis = 0; axis < 3; ++axis)
            set(rotated, axis, out.times[axis], out.values[axis]);
        play(rotated);

        // A window under a held gradient, read by the non-uniform FFT.
        const Axis read_x{{0.0, 0.2e-3, 5.4e-3, 5.6e-3}, {0.0, 4e3, 4e3, 0.0}};
        bloch::AdcWindow window;
        bloch::adc_window(200, 20e-6, 0.3e-3, 0.2, 150.0, std::vector<double>(200, 0.1), window);
        bloch::BlockEvents readout;
        readout.duration = 5.6e-3;
        set(readout, 0, read_x);
        readout.adc_times = window.times.data();
        readout.adc_samples = window.times.size();
        play(readout);

        // A window on a trapezoid's ramps as well, too short for a lattice to
        // pay: read sample by sample.
        std::vector<double> times;
        for (int k = 0; k < 13; ++k)
            times.push_back(0.02e-3 + 0.42e-3 * k);
        bloch::BlockEvents ramped;
        ramped.duration = 5.6e-3;
        set(ramped, 0, read_x);
        ramped.adc_times = times.data();
        ramped.adc_samples = times.size();
        play(ramped);

        // A window on a spiral, read on the lattice in single precision and
        // in double.
        const Spiral spiral;
        for (double tolerance : {1e-4, 1e-7})
        {
            signal.assign(spins.coils() * spiral.block.adc_samples, 0.0);
            spins.play(spiral.block, signal.data(), tolerance);
            touch(signal);
        }

        // A time-shaped pulse joined between its samples, and a dynamic pTx
        // pulse holding its channels one after another over one time base.
        const std::vector<double> shape_times{0.0, 0.1e-3, 0.35e-3, 0.4e-3};
        const std::vector<Complex> shape{0.0, 400.0, 400.0, 0.0};
        std::vector<double> ptx_times;
        std::vector<Complex> ptx_samples;
        for (size_t c = 0; c < rows; ++c)
            for (size_t i = 0; i < shape_times.size(); ++i)
            {
                ptx_times.push_back(shape_times[i]);
                ptx_samples.push_back(shape[i] * std::polar(1.0, 0.5 * c));
            }
        for (const auto& [t, w] : {std::make_pair(shape_times, shape), std::make_pair(ptx_times, ptx_samples)})
        {
            bloch::PulseSteps steps;
            bloch::pulse_steps(t, w, 0.1e-3, 0.3, 200.0, 1e-6, steps);
            // Transmit sensitivities take one channel each; without them the
            // channels are summed.
            if (channels && steps.channels != channels)
                continue;
            bloch::BlockEvents shaped;
            shaped.duration = 0.6e-3;
            shaped.rf_start = steps.start;
            shaped.rf_step = steps.step;
            shaped.rf_steps = steps.steps;
            shaped.rf_channels = steps.channels;
            shaped.rf = steps.rf.data();
            play(shaped);
        }
    }

    /** A repetition: a pulse, and a window under a held gradient after a
     *  phase encoding and before its rewinder; @p read reads one or not. */
    std::vector<bloch::OwnedBlock> repetition(size_t rows, bool read)
    {
        bloch::OwnedBlock pulse;
        pulse.duration = 0.6e-3;
        pulse.rf_start = 0.05e-3;
        pulse.rf_step = 1e-5;
        pulse.rf_steps = 50;
        pulse.rf_channels = rows;
        pulse.rf = sinc(50, rows, 200.0);

        bloch::OwnedBlock readout;
        readout.duration = 3.2e-3;
        readout.gradient_times[0] = {0.4e-3, 0.5e-3, 2.8e-3, 2.9e-3};
        readout.gradient_values[0] = {0.0, 5e3, 5e3, 0.0};
        if (read)
        {
            for (int k = 0; k < 64; ++k)
                readout.adc_times.push_back(0.55e-3 + 35e-6 * k);
            readout.receiver.assign(64, 0.0);
        }
        return {pulse, readout};
    }

    /** Phases alternating by pi, stepped unevenly unless played to a
     *  tolerance, and phase encodings along y and z where a window is read. */
    void schedule(size_t count, double tolerance, bool read, std::vector<double>& phases, std::vector<double>& areas)
    {
        const double uneven = tolerance == 0.0 ? 0.01 : 0.0;
        for (size_t n = 0; n < count; ++n)
        {
            phases.push_back(kPi * static_cast<double>(n % 2) + uneven * static_cast<double>(n * n));
            if (!read)
                continue;
            areas.push_back(0.0);
            areas.push_back(20.0 * (static_cast<double>(n % 8) - 4.0));
            areas.push_back(25.0 * (static_cast<double>(n / 8) - 1.0));
        }
    }

    /** The readout turned about z by a step per repetition, and the phase
     *  encoding along y left unrewound. */
    void turn(size_t count, std::vector<double>& readouts, std::vector<double>& nets)
    {
        for (size_t n = 0; n < count; ++n)
        {
            const double angle = 0.1 * static_cast<double>(n);
            readouts.insert(readouts.end(), {5e3 * (std::cos(angle) - 1.0), 5e3 * std::sin(angle), 0.0});
            nets.insert(nets.end(), {0.0, 20.0 * (static_cast<double>(n % 8) - 4.0), 0.0});
        }
    }

    /** The fixed points' samples summed over the columns along y, and along
     *  z and y. */
    void sum_columns(const bloch::Repetitions& scan, size_t coils)
    {
        const double constant[3] = {0.0, 0.0, 0.0};
        for (const std::vector<int>& axes : {std::vector<int>{1}, std::vector<int>{2, 1}})
        {
            const size_t size = std::accumulate(
                axes.begin(), axes.end(), coils * scan.window_samples(0), [&scan](size_t product, int axis) {
                    return product * scan.lattice(axis).size();
                });
            std::vector<Complex> sums(size);
            scan.column_sums(0, axes, constant, sums.data());
            touch(sums);
        }
    }

    void repetitions(bloch::Isochromats& spins, size_t channels, double tolerance, bool read, bool turned)
    {
        const size_t count = 24;
        std::vector<double> phases, areas, readouts, nets;
        schedule(count, tolerance, read, phases, areas);
        if (turned)
            turn(count, readouts, nets);
        bloch::Repetitions scan(
            spins, repetition(channels ? channels : 1, read), phases, phases, areas, readouts, nets, tolerance);
        if (tolerance > 0.0 && read && scan.split())
            sum_columns(scan, spins.coils());
        // In parts of three and of seven, the last what is left.
        for (size_t played = 0; played < count;)
        {
            const size_t take = std::min(played < 3 ? size_t{3} : size_t{7}, count - played);
            std::vector<Complex> signal(take * scan.coils() * scan.samples());
            scan.play(take, signal.data());
            touch(signal);
            played += take;
        }
    }

    /** A device that reads every array the engine hands it, over the extents
     *  the window states, and writes every value it is to write. */
    bool read_whole(const bloch::LatticeWindowRead& w)
    {
        double sum = 0.0;
        size_t points = 1;
        for (int d = 0; d < w.dimensions; ++d)
            points *= static_cast<size_t>(w.modes[d]);
        sum += static_cast<double>(points == w.points);
        for (size_t q = 0; q <= w.points; ++q)
            sum += w.starts[q];
        for (size_t k = 0; k < w.isochromats; ++k)
            sum += w.order[k] + w.off_resonance[k] + w.decay[w.decay_of[k] * w.segments] + w.mx[k] + w.my[k];
        if (w.receive_re != nullptr)
            for (size_t k = 0; k < w.coils * w.isochromats; ++k)
                sum += w.receive_re[k] + w.receive_im[k];
        for (size_t l = 0; l < w.segments; ++l)
            sum += w.nodes[l];
        for (size_t k = 0; k < w.samples * w.segments; ++k)
            sum += w.basis[k];
        for (int d = 0; d < w.dimensions; ++d)
            for (size_t k = 0; k < w.samples; ++k)
                sum += w.x[d][k];
        sum += w.frequency + w.tolerance;
        for (size_t k = 0; k < w.coils * w.samples; ++k)
            w.out[k] = sum;
        return true;
    }

    /** Excite isochromats many to a lattice point, read by @p coils coils or,
     *  for none, by one of unit sensitivity, and read the spiral's window in
     *  double precision and in single, by the engine and by a device; the
     *  windows read on the lattice. */
    size_t many_coils(size_t coils)
    {
        bloch::IsochromatProperties p;
        for (int i = 0; i < 12; ++i)
            for (int j = 0; j < 10; ++j)
                for (int k = 0; k < 32; ++k)
                {
                    p.x.push_back(3e-3 * (i - 6));
                    p.y.push_back(3e-3 * (j - 5));
                    p.z.push_back(0.25e-3 * (k - 16));
                }
        const size_t n = p.x.size();
        for (size_t i = 0; i < n; ++i)
        {
            p.t2.push_back(i % 2 == 0 ? 0.05 : 0.09);
            p.off_resonance.push_back(40.0 * std::sin(1.3 * i));
        }
        std::vector<Complex> receive;
        for (size_t i = 0; i < n * coils; ++i)
            receive.push_back(std::polar(1.0 + 0.1 * std::sin(0.3 * i), 0.07 * i));
        p.coils = coils;
        p.receive = coils ? receive.data() : nullptr;
        bloch::Isochromats spins(p, 2);

        const std::vector<Complex> hard = sinc(50, 1, 300.0);
        bloch::BlockEvents excite;
        excite.duration = 0.6e-3;
        excite.rf_start = 0.05e-3;
        excite.rf_step = 1e-5;
        excite.rf_steps = 50;
        excite.rf_channels = 1;
        excite.rf = hard.data();
        spins.play(excite, nullptr);

        const Spiral spiral;
        std::vector<Complex> signal;
        for (bool device : {false, true})
        {
            if (device)
                spins.use_device({read_whole, nullptr, nullptr});
            for (double tolerance : {1e-7, 1e-4})
            {
                signal.assign(spins.coils() * spiral.block.adc_samples, 0.0);
                spins.play(spiral.block, signal.data(), tolerance);
                touch(signal);
            }
        }
        // Moved by a whole number of lattice steps, mirrored, and turned by
        // a phase each: the lattice is found again.
        std::vector<double> positions(3 * n);
        spins.positions(positions.data());
        for (size_t i = 0; i < n; ++i)
        {
            positions[3 * i] += 2e-3;
            positions[3 * i + 1] = -positions[3 * i + 1];
        }
        spins.set_positions(positions.data());
        std::vector<double> radians(n);
        for (size_t i = 0; i < n; ++i)
            radians[i] = 0.01 * static_cast<double>(i);
        spins.precess(radians.data());
        signal.assign(spins.coils() * spiral.block.adc_samples, 0.0);
        spins.play(spiral.block, signal.data(), 1e-4);
        touch(signal);
        return spins.lattice_windows() - spins.device_windows() == 2 && spins.device_windows() == 3 ? 2 : 0;
    }

    /** A device that sums every isochromat's term at every sample, from
     *  every array the engine hands it, as the engine would. */
    bool sum_samples(const bloch::SampleWindowRead& w)
    {
        for (size_t c = 0; c < std::max<size_t>(w.coils, 1); ++c)
            for (size_t s = 0; s < w.samples; ++s)
            {
                Complex sum = 0.0;
                for (size_t i = 0; i < w.isochromats; ++i)
                {
                    const double cycles = w.x[i] * w.k[3 * s] + w.y[i] * w.k[3 * s + 1] +
                        w.z[i] * w.k[3 * s + 2] + w.off_resonance[i] * w.time[s];
                    const Complex term = Complex(w.mx[i], w.my[i]) *
                        std::polar(std::exp(-w.rates[w.decay_of[i]] * w.time[s]), -2.0 * kPi * cycles);
                    const Complex receive = w.receive_re == nullptr
                        ? Complex(1.0)
                        : Complex(w.receive_re[c * w.isochromats + i], w.receive_im[c * w.isochromats + i]);
                    sum += receive * term;
                }
                w.out[c * w.samples + s] = sum;
            }
        return true;
    }

    /** Read a spiral's window and one under a held gradient on isochromats
     *  turned off their lattice, by the engine and by a device summing them
     *  sample by sample; whether the device read both, as the engine does. */
    bool off_lattice(size_t coils)
    {
        std::vector<Complex> receive;
        bloch::IsochromatProperties p = lattice(0, coils, receive);
        for (size_t i = 0; i < p.x.size(); ++i)
        {
            const double x = p.x[i];
            p.x[i] = std::cos(0.3) * x - std::sin(0.3) * p.y[i];
            p.y[i] = std::sin(0.3) * x + std::cos(0.3) * p.y[i];
        }
        bloch::Isochromats engine(p, 2);
        bloch::Isochromats device(p, 2);
        device.use_device({nullptr, sum_samples, nullptr});

        const std::vector<Complex> hard = sinc(50, 1, 300.0);
        bloch::BlockEvents excite;
        excite.duration = 0.6e-3;
        excite.rf_start = 0.05e-3;
        excite.rf_step = 1e-5;
        excite.rf_steps = 50;
        excite.rf_channels = 1;
        excite.rf = hard.data();
        const Spiral spiral;
        const Axis held{{0.0, 1e-3}, {2e4, 2e4}};
        std::vector<double> times;
        for (int k = 0; k < 64; ++k)
            times.push_back(0.1e-3 + 10e-6 * k);
        bloch::BlockEvents line;
        line.duration = 1e-3;
        set(line, 0, held);
        line.adc_times = times.data();
        line.adc_samples = times.size();

        double most = 0.0;
        const bloch::BlockEvents* const played[] = {&excite, &spiral.block, &line};
        for (const bloch::BlockEvents* block : played)
        {
            std::vector<Complex> read(engine.coils() * block->adc_samples), summed(read.size());
            engine.play(*block, read.data());
            device.play(*block, summed.data());
            touch(summed);
            for (size_t k = 0; k < read.size(); ++k)
                most = std::max(most, std::abs(read[k] - summed[k]));
        }
        std::vector<double> left(3 * engine.size()), settled(3 * device.size());
        engine.magnetization(left.data());
        device.magnetization(settled.data());
        for (size_t k = 0; k < left.size(); ++k)
            most = std::max(most, std::abs(left[k] - settled[k]));
        return device.device_windows() == 2 && device.lattice_windows() == 0 && most < 1e-9;
    }


    /** A run device that carries a run's slots in plain loops, from every
     *  array the engine hands it, as RunSet and RunTile state the carry. */
    class CarryDevice
    {
    public:
        size_t begun = 0;
        size_t tiles = 0;
        size_t released = 0;

        bloch::RunDevice device()
        {
            bloch::RunDevice made;
            made.begin = [this](const bloch::RunSet& set) { return begin(set); };
            made.tile = [this](const bloch::RunTile& tile) { return carry(tile); };
            made.state = [this](const bloch::RunState& state) { write(state); };
            made.release = [this](size_t run) { released += runs_.erase(run); };
            return made;
        }

    private:
        struct Held
        {
            bloch::RunSet set;
            std::vector<double> values;
            std::vector<size_t> cells, region;
            std::vector<uint32_t> start, decay, index[3];
            std::vector<double> weight, factor, coordinate[3];
            std::vector<unsigned char> turned;
            std::vector<double> origin, place;
        };
        std::map<size_t, Held> runs_;

        static std::vector<double> doubles(const void* data, size_t count, bool single)
        {
            if (single)
            {
                const float* from = static_cast<const float*>(data);
                return std::vector<double>(from, from + count);
            }
            const double* from = static_cast<const double*>(data);
            return std::vector<double>(from, from + count);
        }

        bool begin(const bloch::RunSet& s)
        {
            Held held;
            held.set = s;
            const size_t packs = (s.slots + s.lanes - 1) / s.lanes;
            held.values = doubles(s.pack, packs * s.width * s.lanes, s.single);
            held.cells.assign(s.cells, s.cells + s.windows);
            held.region.assign(s.region, s.region + s.windows + 1);
            held.start.assign(s.start, s.start + s.slots * s.windows);
            held.decay.assign(s.decay, s.decay + s.slots);
            held.weight = doubles(s.weight, s.slots * s.windows * s.taps, s.single);
            held.factor = doubles(s.factor, s.slots * s.windows * s.coils * 2, s.single);
            for (int axis = 0; axis < 3; ++axis)
            {
                if (s.index[axis] != nullptr)
                    held.index[axis].assign(s.index[axis], s.index[axis] + s.slots);
                if (s.coordinate[axis] != nullptr)
                    held.coordinate[axis] = doubles(s.coordinate[axis], s.slots, s.single);
            }
            held.turned.assign(s.turned, s.turned + s.windows);
            if (s.origin != nullptr)
            {
                held.origin.assign(s.origin, s.origin + s.slots * s.windows);
                held.place.assign(s.place, s.place + s.slots * 3);
            }
            runs_[s.run] = std::move(held);
            ++begun;
            return true;
        }

        static double& value(Held& h, size_t slot, size_t at)
        {
            const size_t lanes = h.set.lanes;
            return h.values[(slot / lanes * h.set.width + at) * lanes + slot % lanes];
        }

        /** Set @p set's phase-encoding phase of slot @p n at repetition @p r. */
        static Complex phase(const Held& h, const bloch::RunTile& t, size_t set, size_t n, size_t r)
        {
            constexpr size_t T = bloch::RunTile::kRepetitions;
            Complex factor(1.0, 0.0);
            for (size_t axis = 0; axis < 3; ++axis)
            {
                const size_t at = set * 3 + axis;
                if (t.encoding[at] == 1)
                {
                    const size_t row = static_cast<size_t>(h.index[axis][n]) * T + r;
                    const double re = t.single ? static_cast<const float*>(t.table_re[at])[row]
                                               : static_cast<const double*>(t.table_re[at])[row];
                    const double im = t.single ? static_cast<const float*>(t.table_im[at])[row]
                                               : static_cast<const double*>(t.table_im[at])[row];
                    factor *= Complex(re, im);
                }
                else if (t.encoding[at] == 2)
                    factor *= std::polar(1.0, t.angle[at * T + r] * h.coordinate[axis][n]);
            }
            return factor;
        }

        size_t carry(const bloch::RunTile& t)
        {
            constexpr size_t T = bloch::RunTile::kRepetitions;
            Held& h = runs_.at(t.run);
            const bloch::RunSet& s = h.set;
            std::vector<double> grid(t.grid_size, 0.0);
            const std::vector<double> cosines = doubles(t.turn_cos, T, t.single);
            const std::vector<double> sines = doubles(t.turn_sin, T, t.single);
            size_t dropped = 0;
            for (size_t n = 0; n < s.slots; ++n)
            {
                double m[3] = {value(h, n, 0), value(h, n, 1), value(h, n, 2)};
                for (size_t r = 0; r < t.count; ++r)
                {
                    read(h, t, n, r, m, grid);
                    advance(h, t, n, r, Complex(cosines[r], sines[r]), m);
                }
                const double size = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
                if (t.drop && s.limits && size > 0.0 && !(size > value(h, n, s.limit_at)))
                {
                    m[0] = m[1] = m[2] = 0.0;
                    ++dropped;
                }
                for (size_t k = 0; k < 3; ++k)
                    value(h, n, k) = m[k];
            }
            store(t, grid);
            ++tiles;
            return dropped;
        }

        /** Each window's coefficient of slot @p n's magnetisation @p m at
         *  repetition @p r, phase-encoded, onto @p grid. */
        static void read(
            Held& h, const bloch::RunTile& t, size_t n, size_t r, const double m[3], std::vector<double>& grid)
        {
            const bloch::RunSet& s = h.set;
            for (size_t w = 0; w < s.windows; ++w)
            {
                const size_t u = s.u_at + w * s.u_width;
                Complex q(
                    value(h, n, u) * m[0] + value(h, n, u + 1) * m[1] + value(h, n, u + 2) * m[2],
                    value(h, n, u + 3) * m[0] + value(h, n, u + 4) * m[1] + value(h, n, u + 5) * m[2]);
                if (s.offsets)
                    q += Complex(value(h, n, u + 6), value(h, n, u + 7));
                if (h.turned[w])
                    spread_turned(h, t, n, w, r, q * phase(h, t, w, n, r), grid);
                else
                    spread(h, n, w, r, q * phase(h, t, w, n, r), grid);
            }
        }

        /** Slot @p n's magnetisation @p m carried through repetition @p r,
         *  turned by @p turn and, where netted, by its net area's phase. */
        static void advance(Held& h, const bloch::RunTile& t, size_t n, size_t r, Complex turn, double m[3])
        {
            const bloch::RunSet& s = h.set;
            double next[3];
            for (size_t k = 0; k < 3; ++k)
                next[k] =
                    value(h, n, 3 + 3 * k) * m[0] + value(h, n, 4 + 3 * k) * m[1] + value(h, n, 5 + 3 * k) * m[2];
            if (s.offsets)
            {
                if (t.netted)
                    turn *= phase(h, t, s.windows, n, r);
                const Complex transverse = turn * Complex(next[0] + value(h, n, 12), next[1] + value(h, n, 13));
                next[0] = transverse.real();
                next[1] = transverse.imag();
                next[2] += value(h, n, 14);
            }
            std::copy(next, next + 3, m);
        }

        /** @p grid into the tile's grids, in the tile's precision. */
        static void store(const bloch::RunTile& t, const std::vector<double>& grid)
        {
            for (size_t k = 0; k < t.grid_size; ++k)
                if (t.single)
                    static_cast<float*>(t.grid)[k] = static_cast<float>(grid[k]);
                else
                    static_cast<double*>(t.grid)[k] = grid[k];
        }

        /** Slot @p n's coefficient @p q of window @p w at repetition @p r
         *  times each coil's factor, onto the window's grid points. */
        static void spread(const Held& h, size_t n, size_t w, size_t r, Complex q, std::vector<double>& grid)
        {
            constexpr size_t T = bloch::RunTile::kRepetitions;
            const bloch::RunSet& s = h.set;
            const size_t cells = h.cells[w];
            for (size_t c = 0; c < s.coils; ++c)
            {
                const size_t f = ((n * s.windows + w) * s.coils + c) * 2;
                const Complex term = Complex(h.factor[f], h.factor[f + 1]) * q;
                if (cells == 0)
                {
                    grid[h.region[w] + c * 2 * T + r] += term.real();
                    grid[h.region[w] + c * 2 * T + T + r] += term.imag();
                    continue;
                }
                for (size_t j = 0; j < s.taps; ++j)
                {
                    const double weight = h.weight[(n * s.windows + w) * s.taps + j];
                    const size_t g = h.start[n * s.windows + w] + j;
                    const size_t at = h.region[w] + ((h.decay[n] * s.coils + c) * (cells + s.taps) + g) * 2 * T;
                    grid[at + r] += weight * term.real();
                    grid[at + T + r] += weight * term.imag();
                }
            }
        }

        /** As spread(), for a turned window: onto the grid points repetition
         *  @p r reaches, [T2 class][repetition][grid point][coil][Re, Im]. */
        static void spread_turned(
            const Held& h, const bloch::RunTile& t, size_t n, size_t w, size_t r, Complex q, std::vector<double>& grid)
        {
            constexpr size_t T = bloch::RunTile::kRepetitions;
            const bloch::RunSet& s = h.set;
            const size_t cells = h.cells[w];
            const size_t taps = s.taps;
            double u = h.origin[n * s.windows + w];
            for (size_t axis = 0; axis < 3; ++axis)
                u += t.delta[(w * 3 + axis) * T + r] * h.place[n * 3 + axis];
            const double y = static_cast<double>(cells) * (u - std::nearbyint(u)) - 0.5 * static_cast<double>(taps) + 0.5;
            double first = std::nearbyint(y);
            const double offset = y - first;
            if (first < 0.0)
                first += static_cast<double>(cells);
            const std::vector<double> polynomials = doubles(t.polynomials, s.windows * t.powers * taps, t.single);
            for (size_t j = 0; j < taps; ++j)
            {
                double weight = 0.0;
                for (size_t power = t.powers; power-- > 0;)
                    weight = weight * offset + polynomials[(w * t.powers + power) * taps + j];
                const size_t g = static_cast<size_t>(first) + j;
                for (size_t c = 0; c < s.coils; ++c)
                {
                    const size_t f = ((n * s.windows + w) * s.coils + c) * 2;
                    const Complex term = weight * Complex(h.factor[f], h.factor[f + 1]) * q;
                    const size_t at = h.region[w] + (((h.decay[n] * T + r) * (cells + taps) + g) * s.coils + c) * 2;
                    grid[at] += term.real();
                    grid[at + 1] += term.imag();
                }
            }
        }

        void write(const bloch::RunState& state)
        {
            Held& h = runs_.at(state.run);
            for (size_t n = 0; n < state.slots; ++n)
                for (size_t k = 0; k < 3; ++k)
                {
                    const size_t at = (n / state.lanes * state.width + k) * state.lanes + n % state.lanes;
                    if (state.single)
                        static_cast<float*>(state.pack)[at] = static_cast<float>(value(h, n, k));
                    else
                        static_cast<double*>(state.pack)[at] = value(h, n, k);
                }
        }
    };

    /** Whether @p theirs differs from @p mine by at most @p within of the
     *  largest of @p mine. */
    bool read_alike(const std::vector<Complex>& mine, const std::vector<Complex>& theirs, double within)
    {
        double scale = 0.0, most = 0.0;
        for (size_t k = 0; k < mine.size(); ++k)
        {
            scale = std::max(scale, std::abs(mine[k]));
            most = std::max(most, std::abs(mine[k] - theirs[k]));
        }
        return most <= within * scale;
    }

    /** Whether @p a and @p b hold magnetisations at most @p within apart. */
    bool left_alike(bloch::Isochromats& a, bloch::Isochromats& b, double within)
    {
        std::vector<double> left(3 * a.size()), settled(3 * b.size());
        a.magnetization(left.data());
        b.magnetization(settled.data());
        double moved = 0.0;
        for (size_t k = 0; k < left.size(); ++k)
            moved = std::max(moved, std::abs(left[k] - settled[k]));
        return moved <= within;
    }

    /** Net areas along z of @p count repetitions, and readouts turned about
     *  x, where asked for. */
    void netted_and_turned(
        size_t count, bool netted, bool turned, std::vector<double>& nets, std::vector<double>& readouts)
    {
        for (size_t n = 0; n < count && netted; ++n)
            nets.insert(nets.end(), {0.0, 0.0, 15.0 * static_cast<double>(n % 3)});
        for (size_t n = 0; n < count && turned; ++n)
            readouts.insert(
                readouts.end(),
                {0.0, 1e3 * std::sin(0.1 * static_cast<double>(n)), 5e2 * std::cos(0.3 * static_cast<double>(n))});
    }

    /** Play repetitions by the engine and by a CarryDevice to @p tolerance,
     *  split, with net areas where @p netted, and read along readouts turned
     *  where @p turned; whether the device carried them and read and left
     *  what the engine read and left. */
    bool carried_as_engine(const bloch::IsochromatProperties& p, double tolerance, bool netted, bool turned)
    {
        bloch::Isochromats engine(p, 2);
        bloch::Isochromats device(p, 2);
        CarryDevice carry;
        device.use_run_device(carry.device());
        const size_t count = 40;
        std::vector<double> phases, areas, readouts, nets;
        schedule(count, tolerance, true, phases, areas);
        netted_and_turned(count, netted, turned, nets, readouts);
        bool split = true;
        std::vector<Complex> mine, theirs;
        {
            bloch::Repetitions a(engine, repetition(1, true), phases, phases, areas, readouts, nets, tolerance);
            bloch::Repetitions b(device, repetition(1, true), phases, phases, areas, readouts, nets, tolerance);
            if (tolerance > 0.0 && !netted)
                split = a.split() == b.split();
            const size_t per = a.coils() * a.samples();
            mine.resize(count * per);
            theirs.resize(count * per);
            a.play(count / 2, mine.data());
            b.play(count / 2, theirs.data());
            a.play(count - count / 2, mine.data() + count / 2 * per);
            b.play(count - count / 2, theirs.data() + count / 2 * per);
        }
        const double within = tolerance > 0.0 ? 1e-5 : 1e-12;
        return split && carry.begun == 1 && carry.tiles == 4 && carry.released == 1 &&
            read_alike(mine, theirs, within) && left_alike(engine, device, within);
    }

    /** Whether a CarryDevice carries repetitions as the engine does, exact
     *  and to a tolerance, with and without net areas and turned readouts. */
    bool carried(size_t coils)
    {
        std::vector<Complex> receive;
        const bloch::IsochromatProperties p = lattice(0, coils, receive);
        bool agree = true;
        for (double tolerance : {0.0, 1e-4})
            for (bool netted : {false, true})
                for (bool turned : {false, true})
                    agree = carried_as_engine(p, tolerance, netted, turned) && agree;
        return agree;
    }

} // namespace

int main()
{
    use_direct_plans();
    for (size_t channels : {size_t{0}, size_t{2}})
    {
        std::vector<Complex> receive;
        bloch::Isochromats spins(lattice(channels, channels ? 3 : 1, receive), 2);
        blocks(spins, channels);
        for (double tolerance : {0.0, 1e-4})
        {
            for (bool read : {true, false})
                repetitions(spins, channels, tolerance, read, false);
            repetitions(spins, channels, tolerance, true, true);
        }
        std::vector<double> magnetization(3 * spins.size());
        spins.magnetization(magnetization.data());
        sink = std::accumulate(magnetization.begin(), magnetization.end(), sink);
        sink += static_cast<double>(spins.lattice_windows());
    }
    // More coils than a tile sums at once, and not a multiple of those
    // summed together.
    for (size_t coils : {size_t{0}, size_t{37}})
        if (many_coils(coils) != 2)
        {
            std::fprintf(stderr, "the windows of %zu coils were not read on the lattice by both\n", coils);
            return 1;
        }
    for (size_t coils : {size_t{0}, size_t{3}})
        if (!off_lattice(coils))
        {
            std::fprintf(stderr, "the windows of %zu coils off the lattice were not read by the device\n", coils);
            return 1;
        }
    for (size_t coils : {size_t{0}, size_t{3}})
        if (!carried(coils))
        {
            std::fprintf(stderr, "the repetitions of %zu coils were not carried as the engine carries them\n", coils);
            return 1;
        }
    std::printf("%g\n", sink);
    return 0;
}
