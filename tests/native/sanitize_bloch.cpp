/**
 * @file sanitize_bloch.cpp
 * @brief Drive the isochromat engine through each of its paths with the
 *        sanitisers watching.
 *
 * A read past the end of an array is not a wrong answer but whatever
 * happened to be there, which the numerical tests catch only when it changes
 * a number. This plays blocks that reach every path of the engine -- a pulse
 * stepped for each group of isochromats, one reused turned by its phase, one
 * computed on a grid of fields and drives, a window read sample by sample and
 * one read by the non-uniform FFT, gradients turned with steps in them -- and
 * repetitions of them, exact and to a tolerance, on one transmit channel and
 * on two, with AddressSanitizer and UndefinedBehaviorSanitizer on. It asserts
 * nothing. The sanitisers do.
 */

#include "bloch/bloch.hpp"
#include "bloch/events.hpp"
#include "bloch/repetitions.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
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

        // A window on a trapezoid's ramps as well: read sample by sample.
        std::vector<double> times;
        for (int k = 0; k < 64; ++k)
            times.push_back(0.02e-3 + 0.085e-3 * k);
        bloch::BlockEvents ramped;
        ramped.duration = 5.6e-3;
        set(ramped, 0, read_x);
        ramped.adc_times = times.data();
        ramped.adc_samples = times.size();
        play(ramped);

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

} // namespace

int main()
{
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
    }
    std::printf("%g\n", sink);
    return 0;
}
