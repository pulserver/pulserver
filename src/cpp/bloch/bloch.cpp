/**
 * @file bloch.cpp
 * @brief Isochromats whose magnetisation the Bloch equation carries from one
 *        block to the next.
 */

#include "bloch/bloch.hpp"
#include "bloch/cycles.hpp"
#include "bloch/finufft.hpp"
#include "bloch/nufft.hpp"
#include "bloch/parallel.hpp"
#include "bloch/simd.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

namespace bloch
{

    /** The gradient area from a block's start, per axis, in 1/m. */
    class GradientAreas
    {
    public:
        explicit GradientAreas(const BlockEvents& block)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                times_[axis] = block.gradient_times[axis];
                values_[axis] = block.gradient_values[axis];
                count_[axis] =
                    (times_[axis] != nullptr && values_[axis] != nullptr) ? block.gradient_corners[axis] : 0;
                std::vector<double>& cumulative = cumulative_[axis];
                cumulative.assign(count_[axis], 0.0);
                for (size_t i = 1; i < count_[axis]; ++i)
                    cumulative[i] = cumulative[i - 1] +
                        0.5 * (values_[axis][i] + values_[axis][i - 1]) * (times_[axis][i] - times_[axis][i - 1]);
            }
        }

        void at(double time, double area[3]) const
        {
            for (int axis = 0; axis < 3; ++axis)
                area[axis] = along(axis, time);
        }

    private:
        double along(int axis, double time) const
        {
            const size_t count = count_[axis];
            if (count == 0 || time <= times_[axis][0])
                return 0.0;
            const double* times = times_[axis];
            if (time >= times[count - 1])
                return cumulative_[axis][count - 1];
            const size_t after = static_cast<size_t>(std::upper_bound(times, times + count, time) - times);
            const size_t before = after - 1;
            const double span = times[after] - times[before];
            const double elapsed = time - times[before];
            const double start = values_[axis][before];
            const double slope = span > 0.0 ? (values_[axis][after] - start) / span : 0.0;
            return cumulative_[axis][before] + start * elapsed + 0.5 * slope * elapsed * elapsed;
        }

        const double* times_[3] = {nullptr, nullptr, nullptr};
        const double* values_[3] = {nullptr, nullptr, nullptr};
        size_t count_[3] = {0, 0, 0};
        std::vector<double> cumulative_[3];
    };

    namespace
    {

        constexpr double kTwoPi = 6.283185307179586476925286766559;

        /** Positions along the gradient of an RF pulse closer than this, in m,
         *  share one computation of the pulse. */
        constexpr double kQuantum = 1e-12;

        /** Tolerance, in s, on event times against the block's bounds. */
        constexpr double kTimeTolerance = 1e-12;

        /** Isochromats a worker takes at the least. */
        constexpr size_t kChunk = 4096;

        /** Isochromats an ADC window is read over together. */
        constexpr size_t kTile = 256;

        /** Groupings kept for reuse, by the field a pulse plays under. */
        constexpr size_t kGroupings = 4;

        /** Pulses whose maps are kept for reuse, and the bytes they may take. */
        constexpr size_t kHeldPulses = 64;
        constexpr size_t kHeldBytes = size_t(1) << 30;

        /** Largest difference between a pulse and an earlier one turned by a
         *  phase, relative to the earlier one's peak, at which the earlier
         *  one's maps serve. */
        constexpr double kPhaseTolerance = 1e-12;

        /** Points of a pulse's grid per 1/T, T the pulse's duration: the field
         *  an isochromat sees during the pulse is tabulated this finely. */
        constexpr double kGridPoints = 64.0;

        /** Largest variation of a pulse's gradient over its steps, relative
         *  to the gradient, at which the gradient is taken to be held. */
        constexpr double kHeldGradient = 1e-9;

        /** Pulse tables kept for reuse, and the bytes they may take. */
        constexpr size_t kPulseTables = 64;
        constexpr size_t kTableBytes = size_t(1) << 30;

        /** T2s beyond which an ADC window is read sample by sample, and the
         *  bytes its transform's grids may take. */
        constexpr size_t kWindowDecays = 32;
        constexpr size_t kWindowBytes = size_t(1) << 28;

        /** Transforms of ADC windows kept for reuse. */
        constexpr size_t kTransforms = 8;

        /** On what of an isochromat's position the field during a pulse
         *  depends: nothing, its position along one direction, or any. */
        constexpr int kNoGradient = 0;
        constexpr int kOneDirection = 1;
        constexpr int kAnyDirection = 2;

        uint64_t bits_of(double value)
        {
            uint64_t bits = 0;
            std::memcpy(&bits, &value, sizeof bits);
            return bits;
        }

        bool same_bits(const double a[3], const double b[3])
        {
            return std::memcmp(a, b, 3 * sizeof(double)) == 0;
        }

        double rate(double time)
        {
            return std::isinf(time) ? 0.0 : 1.0 / time;
        }

        void fill(std::vector<double>& values, size_t count, double value, const char* name)
        {
            if (values.empty())
                values.assign(count, value);
            else if (values.size() != count)
                throw std::invalid_argument(
                    std::string(name) + " holds " + std::to_string(values.size()) + " values for " +
                    std::to_string(count) + " isochromats");
        }

        void require_finite(const std::vector<double>& values, const char* name)
        {
            if (!std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); }))
                throw std::invalid_argument(std::string(name) + " must be finite");
        }

        void require_positive(const std::vector<double>& values)
        {
            if (!std::all_of(values.begin(), values.end(), [](double value) { return value > 0.0; }))
                throw std::invalid_argument("relaxation times must be positive");
        }

        /** Check @p values holds @p per finite sensitivities per isochromat,
         *  none where @p per is zero. */
        void require_finite(const std::complex<double>* values, size_t size)
        {
            for (size_t k = 0; k < size; ++k)
                if (!std::isfinite(values[k].real()) || !std::isfinite(values[k].imag()))
                    throw std::invalid_argument("sensitivities must be finite");
        }

        void require_sensitivities(
            const std::vector<std::complex<double>>& values, size_t count, size_t per, const char* name)
        {
            if (values.size() != count * per)
                throw std::invalid_argument(
                    std::string(name) + " must hold one row of sensitivities per isochromat");
            require_finite(values.data(), values.size());
        }

        /**
         * The rotation of the magnetisation about (wx, wy, wz) by its length,
         * in rad, clockwise seen from its tip: R = cI + (1 - c) n n^T - s [n]x.
         */
        void rotation(double wx, double wy, double wz, double r[3][3])
        {
            const double angle = std::sqrt(wx * wx + wy * wy + wz * wz);
            if (!(angle > 1e-300))
            {
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        r[i][j] = i == j ? 1.0 : 0.0;
                return;
            }
            const double nx = wx / angle;
            const double ny = wy / angle;
            const double nz = wz / angle;
            const double c = std::cos(angle);
            const double s = std::sin(angle);
            const double half = std::sin(0.5 * angle);
            const double k = 2.0 * half * half;
            r[0][0] = c + k * nx * nx;
            r[0][1] = k * nx * ny + s * nz;
            r[0][2] = k * nx * nz - s * ny;
            r[1][0] = k * ny * nx - s * nz;
            r[1][1] = c + k * ny * ny;
            r[1][2] = k * ny * nz + s * nx;
            r[2][0] = k * nz * nx + s * ny;
            r[2][1] = k * nz * ny - s * nx;
            r[2][2] = c + k * nz * nz;
        }

        /**
         * Compose one step, a rotation between two half steps of relaxation,
         * after the affine map M -> A M + c M0 accumulated so far.
         */
        void compose(const double turn[3][3], const double decay[3], double recovery, double a[3][3], double c[3])
        {
            double step[3][3];
            double shift[3];
            for (int row = 0; row < 3; ++row)
            {
                for (int col = 0; col < 3; ++col)
                    step[row][col] = decay[row] * turn[row][col] * decay[col];
                shift[row] = decay[row] * turn[row][2] * recovery + (row == 2 ? recovery : 0.0);
            }
            double next[3][3];
            double moved[3];
            for (int row = 0; row < 3; ++row)
            {
                for (int col = 0; col < 3; ++col)
                    next[row][col] =
                        step[row][0] * a[0][col] + step[row][1] * a[1][col] + step[row][2] * a[2][col];
                moved[row] = step[row][0] * c[0] + step[row][1] * c[1] + step[row][2] * c[2] + shift[row];
            }
            std::memcpy(a, next, sizeof next);
            std::memcpy(c, moved, sizeof moved);
        }

    } // namespace

    /** The gradient area over each step of a pulse, and on what of an
     *  isochromat's position the field it sees depends. */
    struct PulseGradient
    {
        int mode = kNoGradient;
        /** For kOneDirection the direction; for kAnyDirection, 1 on each
         *  axis a gradient plays on. */
        double direction[3] = {0.0, 0.0, 0.0};
        /** Per step, the gradient area in 1/m. */
        std::vector<double> delta;
        /** Per step, the mean gradient along the direction, in Hz/m. */
        std::vector<double> along;
    };

    namespace
    {

        /** Whether every step's area lies along the gradient's direction,
         *  filling the mean gradient along it. */
        bool along_one_direction(PulseGradient& gradient, size_t steps, double dt)
        {
            gradient.along.resize(steps);
            const double* direction = gradient.direction;
            for (size_t i = 0; i < steps; ++i)
            {
                const double* d = &gradient.delta[3 * i];
                const double projection = d[0] * direction[0] + d[1] * direction[1] + d[2] * direction[2];
                double off = 0.0;
                double size = 0.0;
                for (int axis = 0; axis < 3; ++axis)
                {
                    const double rest = d[axis] - projection * direction[axis];
                    off += rest * rest;
                    size += d[axis] * d[axis];
                }
                if (off > 1e-18 * size)
                    return false;
                gradient.along[i] = projection / dt;
            }
            return true;
        }

        /** Fill each step's gradient area, and return the step of the largest. */
        size_t step_areas(const BlockEvents& block, const GradientAreas& areas, PulseGradient& gradient, double& largest)
        {
            gradient.delta.resize(3 * block.rf_steps);
            double previous[3];
            areas.at(block.rf_start, previous);
            largest = 0.0;
            size_t loudest = 0;
            for (size_t i = 0; i < block.rf_steps; ++i)
            {
                double next[3];
                areas.at(block.rf_start + static_cast<double>(i + 1) * block.rf_step, next);
                double norm = 0.0;
                for (int axis = 0; axis < 3; ++axis)
                {
                    const double d = next[axis] - previous[axis];
                    gradient.delta[3 * i + axis] = d;
                    norm += d * d;
                    previous[axis] = next[axis];
                }
                if (norm > largest)
                {
                    largest = norm;
                    loudest = i;
                }
            }
            return loudest;
        }

        PulseGradient pulse_gradient(const BlockEvents& block, const GradientAreas& areas)
        {
            PulseGradient gradient;
            double largest = 0.0;
            const size_t loudest = step_areas(block, areas, gradient, largest);
            if (!(largest > 0.0))
                return gradient;

            const double norm = std::sqrt(largest);
            for (int axis = 0; axis < 3; ++axis)
                gradient.direction[axis] = gradient.delta[3 * loudest + axis] / norm;
            gradient.mode = kOneDirection;
            if (along_one_direction(gradient, block.rf_steps, block.rf_step))
                return gradient;

            gradient.mode = kAnyDirection;
            for (int axis = 0; axis < 3; ++axis)
            {
                gradient.direction[axis] = 0.0;
                for (size_t i = 0; i < block.rf_steps; ++i)
                    if (gradient.delta[3 * i + axis] != 0.0)
                        gradient.direction[axis] = 1.0;
            }
            return gradient;
        }

        /** Whether @p rf is @p held times one phase factor, to within
         *  kPhaseTolerance of the held peak; the factor in @p turn. */
        bool one_phase_apart(
            const std::vector<std::complex<double>>& held, const std::complex<double>* rf, std::complex<double>& turn)
        {
            size_t peak = 0;
            double largest = 0.0;
            for (size_t k = 0; k < held.size(); ++k)
                if (std::norm(held[k]) > largest)
                {
                    largest = std::norm(held[k]);
                    peak = k;
                }
            turn = 1.0;
            if (largest > 0.0)
            {
                turn = rf[peak] / held[peak];
                const double size = std::abs(turn);
                if (!(std::abs(size - 1.0) <= kPhaseTolerance))
                    return false;
                turn /= size;
            }
            const double limit = kPhaseTolerance * std::sqrt(largest);
            for (size_t k = 0; k < held.size(); ++k)
                if (!(std::abs(rf[k] - turn * held[k]) <= limit))
                    return false;
            return true;
        }

        /** The field along z isochromat @p r sees over step @p i, in Hz. */
        double step_field(const IsochromatProperties& p, size_t r, const PulseGradient& gradient, size_t i, double dt)
        {
            const double* d = gradient.direction;
            if (gradient.mode == kOneDirection)
                return p.off_resonance[r] + (p.x[r] * d[0] + p.y[r] * d[1] + p.z[r] * d[2]) * gradient.along[i];
            if (gradient.mode == kAnyDirection)
            {
                const double* delta = &gradient.delta[3 * i];
                return p.off_resonance[r] + (p.x[r] * delta[0] + p.y[r] * delta[1] + p.z[r] * delta[2]) / dt;
            }
            return p.off_resonance[r];
        }

        /** Whether the pulse plays under no gradient or one that holds its
         *  amplitude along its direction over every step, so that each
         *  isochromat sees one field throughout; that amplitude, in Hz/m, in
         *  @p along. */
        bool held_gradient(const PulseGradient& gradient, double& along)
        {
            along = 0.0;
            if (gradient.mode == kNoGradient)
                return true;
            if (gradient.mode != kOneDirection || gradient.along.empty())
                return false;
            const double first = gradient.along[0];
            along = first;
            return std::all_of(gradient.along.begin(), gradient.along.end(), [first](double value) {
                return std::abs(value - first) <= kHeldGradient * std::abs(first);
            });
        }

        /** Turn the affine map (@p a, @p c) about z by @p angle on either
         *  side: a -> R a R, c -> R c, R counterclockwise. */
        void turn_both_sides(double a[3][3], double c[3], double angle)
        {
            const double cosine = std::cos(angle);
            const double sine = std::sin(angle);
            for (int row = 0; row < 3; ++row)
            {
                const double first = a[row][0];
                a[row][0] = cosine * first + sine * a[row][1];
                a[row][1] = cosine * a[row][1] - sine * first;
            }
            for (int col = 0; col < 3; ++col)
            {
                const double first = a[0][col];
                a[0][col] = cosine * first - sine * a[1][col];
                a[1][col] = sine * first + cosine * a[1][col];
            }
            const double first = c[0];
            c[0] = cosine * first - sine * c[1];
            c[1] = sine * first + cosine * c[1];
        }

        /**
         * Write the affine map @p rf times @p drive, held for @p dt a step,
         * applies to an isochromat that sees the field @p nu, in Hz,
         * throughout, and relaxes with @p t1 and @p t2, to @p map: A then c,
         * in the frame the isochromat's own precession over half the pulse,
         * P, turns on either side, A = P A~ P and c = P c~. What is left
         * varies with @p nu as slowly as the pulse's response does, which the
         * precession would not.
         */
        void grid_map(
            double nu,
            double t1,
            double t2,
            const std::vector<std::complex<double>>& rf,
            double drive,
            double dt,
            double* map)
        {
            const double e1 = std::exp(-0.5 * dt * rate(t1));
            const double e2 = std::exp(-0.5 * dt * rate(t2));
            const double decay[3] = {e2, e2, e1};
            const double angle = kTwoPi * dt * drive;
            double a[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
            double c[3] = {0.0, 0.0, 0.0};
            for (const std::complex<double>& b1 : rf)
            {
                double turn[3][3];
                rotation(angle * b1.real(), angle * b1.imag(), kTwoPi * dt * nu, turn);
                compose(turn, decay, 1.0 - e1, a, c);
            }
            turn_both_sides(a, c, 0.5 * kTwoPi * nu * dt * static_cast<double>(rf.size()));
            std::memcpy(map, a, 9 * sizeof(double));
            std::memcpy(map + 9, c, 3 * sizeof(double));
        }

        /** The pulse's transverse field summed over its channels, per step, in Hz. */
        std::vector<std::complex<double>> channel_sum(const BlockEvents& block)
        {
            std::vector<std::complex<double>> summed(block.rf_steps, 0.0);
            for (size_t c = 0; c < block.rf_channels; ++c)
                for (size_t i = 0; i < block.rf_steps; ++i)
                    summed[i] += block.rf[c * block.rf_steps + i];
            return summed;
        }

        /**
         * Whether every channel of the pulse plays one waveform times a weight
         * of its own, to within kPhaseTolerance of the pulse's peak: the
         * waveform, one at the peak, in @p shape, and each channel's weight,
         * in Hz, in @p weights.
         */
        bool one_waveform(
            const BlockEvents& block,
            std::vector<std::complex<double>>& shape,
            std::vector<std::complex<double>>& weights)
        {
            const size_t steps = block.rf_steps;
            size_t peak = 0;
            double largest = 0.0;
            for (size_t k = 0; k < block.rf_channels * steps; ++k)
                if (std::norm(block.rf[k]) > largest)
                {
                    largest = std::norm(block.rf[k]);
                    peak = k;
                }
            if (!(largest > 0.0))
                return false;
            const size_t at = peak % steps;
            const std::complex<double>* reference = block.rf + (peak - at);
            shape.resize(steps);
            for (size_t i = 0; i < steps; ++i)
                shape[i] = reference[i] / reference[at];
            weights.resize(block.rf_channels);
            const double limit = kPhaseTolerance * std::sqrt(largest);
            for (size_t c = 0; c < block.rf_channels; ++c)
            {
                const std::complex<double>* channel = block.rf + c * steps;
                weights[c] = channel[at];
                for (size_t i = 0; i < steps; ++i)
                    if (!(std::abs(channel[i] - weights[c] * shape[i]) <= limit))
                        return false;
            }
            return true;
        }

        /** Rows below and above an isochromat's drive its map is interpolated
         *  from: along the drive, where the map turns as fast as it does
         *  across the field, a quintic reaches the accuracy a cubic reaches
         *  across the field, where half the pulse's precession is taken out. */
        constexpr long long kRowsBelow = 2;
        constexpr long long kRowsAbove = 3;
        constexpr size_t kRows = static_cast<size_t>(kRowsBelow + kRowsAbove + 1);

        /**
         * Per class, the first and last rows and columns of a table of
         * @p spacing, in Hz, and @p drive_spacing that the interpolation
         * around each of its isochromats' @p field and the magnitude of its
         * @p drive reads; row 0 alone without a drive.
         */
        void grid_bounds(
            const std::vector<double>& field,
            const std::vector<std::complex<double>>& drive,
            const std::vector<uint32_t>& class_of,
            double spacing,
            double drive_spacing,
            std::vector<std::array<long long, 4>>& bounds)
        {
            for (std::array<long long, 4>& b : bounds)
                b = {std::numeric_limits<long long>::max(), std::numeric_limits<long long>::min(),
                     std::numeric_limits<long long>::max(), std::numeric_limits<long long>::min()};
            for (size_t i = 0; i < field.size(); ++i)
            {
                std::array<long long, 4>& b = bounds[class_of[i]];
                long long row = 0;
                long long before = 0;
                long long after = 0;
                if (!drive.empty())
                {
                    row = static_cast<long long>(std::floor(std::abs(drive[i]) / drive_spacing));
                    before = kRowsBelow;
                    after = kRowsAbove;
                }
                const long long column = static_cast<long long>(std::floor(field[i] / spacing));
                b[0] = std::min(b[0], row - before);
                b[1] = std::max(b[1], row + after);
                b[2] = std::min(b[2], column - 1);
                b[3] = std::max(b[3], column + 2);
            }
        }

        /** Rows first_row to last_row and columns first to last of a table;
         *  none where last_row < first_row. */
        struct Points
        {
            long long first_row;
            long long last_row;
            long long first;
            long long last;

            long long rows() const
            {
                return last_row < first_row ? 0 : last_row - first_row + 1;
            }
            long long columns() const
            {
                return last - first + 1;
            }
            long long count() const
            {
                return rows() * columns();
            }
            bool holds(long long row, long long column) const
            {
                return row >= first_row && row <= last_row && column >= first && column <= last;
            }
            /** Where a point's 12 values start in a table of these points. */
            size_t at(long long row, long long column) const
            {
                return 12 * static_cast<size_t>((row - first_row) * columns() + (column - first));
            }
        };

        /** The points of a table holding @p rows rows from @p first_row and
         *  @p columns columns from @p first. */
        Points table_points(long long first_row, long long rows, long long first, long long columns)
        {
            return {first_row, first_row + rows - 1, first, first + columns - 1};
        }

        /** The smallest rectangle of points covering @p held and @p wanted. */
        Points covering(const Points& held, const Points& wanted)
        {
            if (held.rows() == 0)
                return wanted;
            return {
                std::min(held.first_row, wanted.first_row), std::max(held.last_row, wanted.last_row),
                std::min(held.first, wanted.first), std::max(held.last, wanted.last)};
        }

        /** The points of @p made outside @p held, row by row. */
        std::vector<std::pair<long long, long long>> points_beyond(const Points& made, const Points& held)
        {
            std::vector<std::pair<long long, long long>> beyond;
            for (long long row = made.first_row; row <= made.last_row; ++row)
                for (long long column = made.first; column <= made.last; ++column)
                    if (!held.holds(row, column))
                        beyond.emplace_back(row, column);
            return beyond;
        }

        /** Copy the maps of @p from, laid out as @p held, to @p to, laid out
         *  as @p made. */
        void copy_points(
            const std::vector<double>& from, const Points& held, std::vector<double>& to, const Points& made)
        {
            for (long long row = held.first_row; row <= held.last_row; ++row)
                std::copy_n(
                    from.begin() + static_cast<std::ptrdiff_t>(held.at(row, held.first)),
                    12 * held.columns(),
                    to.begin() + static_cast<std::ptrdiff_t>(made.at(row, held.first)));
        }

        /** The weights of the Lagrange cubic through points -1, 0, 1 and 2,
         *  at @p u in [0, 1). */
        void cubic_weights(double u, double w[4])
        {
            w[0] = -u * (u - 1.0) * (u - 2.0) / 6.0;
            w[1] = (u + 1.0) * (u - 1.0) * (u - 2.0) / 2.0;
            w[2] = -(u + 1.0) * u * (u - 2.0) / 2.0;
            w[3] = (u + 1.0) * u * (u - 1.0) / 6.0;
        }

        /** The cubic of weights @p w through the four consecutive maps from
         *  @p around. */
        void cubic(const double* around, const double w[4], double* map)
        {
            for (int e = 0; e < 12; ++e)
                map[e] = w[0] * around[e] + w[1] * around[12 + e] + w[2] * around[24 + e] + w[3] * around[36 + e];
        }

        /** Write the affine map @p map turned about z by @p before before it
         *  and by @p after after it to @p out: A -> R(after) A R(before) and
         *  c -> R(after) c, as Isochromats::apply turns it. */
        void turned(const double* map, std::complex<double> before, std::complex<double> after, double* out)
        {
            const double br = before.real();
            const double bi = before.imag();
            const double ar = after.real();
            const double ai = after.imag();
            double a[3][3];
            for (int row = 0; row < 3; ++row)
            {
                const double* m = map + 3 * row;
                a[row][0] = br * m[0] + bi * m[1];
                a[row][1] = br * m[1] - bi * m[0];
                a[row][2] = m[2];
            }
            for (int col = 0; col < 3; ++col)
            {
                out[col] = ar * a[0][col] - ai * a[1][col];
                out[3 + col] = ai * a[0][col] + ar * a[1][col];
                out[6 + col] = a[2][col];
            }
            out[9] = ar * map[9] - ai * map[10];
            out[10] = ai * map[9] + ar * map[10];
            out[11] = map[11];
        }

        /** The weights of the Lagrange polynomial through the kRows points
         *  from -kRowsBelow on, at @p v in [0, 1). */
        void row_weights(double v, double w[kRows])
        {
            for (size_t j = 0; j < kRows; ++j)
            {
                const double at = static_cast<double>(static_cast<long long>(j) - kRowsBelow);
                double weight = 1.0;
                for (size_t m = 0; m < kRows; ++m)
                    if (m != j)
                    {
                        const double other = static_cast<double>(static_cast<long long>(m) - kRowsBelow);
                        weight *= (v - other) / (at - other);
                    }
                w[j] = weight;
            }
        }

        /** Write the affine map a pulse applies to isochromat @p r, A then c,
         *  to @p map. */
        void pulse_map(
            const IsochromatProperties& p,
            size_t r,
            const BlockEvents& block,
            const PulseGradient& gradient,
            const std::vector<std::complex<double>>& summed,
            double* map)
        {
            const size_t steps = block.rf_steps;
            const double dt = block.rf_step;
            const double e1 = std::exp(-0.5 * dt * rate(p.t1[r]));
            const double e2 = std::exp(-0.5 * dt * rate(p.t2[r]));
            const double decay[3] = {e2, e2, e1};
            const std::complex<double>* transmit =
                p.transmit_channels ? &p.transmit[r * p.transmit_channels] : nullptr;

            double a[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
            double c[3] = {0.0, 0.0, 0.0};
            for (size_t i = 0; i < steps; ++i)
            {
                std::complex<double> b1 = transmit ? 0.0 : summed[i];
                for (size_t k = 0; transmit && k < block.rf_channels; ++k)
                    b1 += transmit[k] * block.rf[k * steps + i];
                const double bz = step_field(p, r, gradient, i, dt);
                double turn[3][3];
                rotation(kTwoPi * dt * b1.real(), kTwoPi * dt * b1.imag(), kTwoPi * dt * bz, turn);
                compose(turn, decay, 1.0 - e1, a, c);
            }
            std::memcpy(map, a, 9 * sizeof(double));
            std::memcpy(map + 9, c, 3 * sizeof(double));
        }

        /** The gradient area and time from one ADC sample to the next. */
        struct SampleSteps
        {
            std::vector<double> area;
            std::vector<double> time;
            /** One increment throughout, which makes an isochromat's turn per
             *  sample one complex factor. */
            bool uniform = true;
            /** The time from one sample to the next where it is one
             *  throughout, which makes an isochromat's decay per sample one
             *  factor; 0 otherwise. */
            double dwell = 0.0;
        };

        SampleSteps sample_steps(const double* times, size_t samples, const GradientAreas& areas)
        {
            SampleSteps steps;
            steps.area.assign(3 * samples, 0.0);
            steps.time.assign(samples, 0.0);
            bool dwell = samples > 1;
            double previous[3];
            areas.at(times[0], previous);
            for (size_t k = 1; k < samples; ++k)
            {
                double next[3];
                areas.at(times[k], next);
                steps.time[k] = times[k] - times[k - 1];
                for (int axis = 0; axis < 3; ++axis)
                {
                    steps.area[3 * k + axis] = next[axis] - previous[axis];
                    previous[axis] = next[axis];
                    if (k > 1 &&
                        std::fabs(steps.area[3 * k + axis] - steps.area[3 + axis]) >
                            1e-12 * (1.0 + std::fabs(next[axis])))
                        steps.uniform = false;
                }
                if (k > 1 && std::fabs(steps.time[k] - steps.time[1]) > 1e-12 * steps.time[1] + 1e-18)
                    steps.uniform = dwell = false;
            }
            steps.dwell = dwell ? steps.time[1] : 0.0;
            return steps;
        }

        /** Receive sensitivities, coil-major: entry c * count + i. */
        struct Receive
        {
            const std::vector<double>& re;
            const std::vector<double>& im;
            size_t coils;
            size_t count;
        };

        /** A tile's receive sensitivities: @c coils rows of @c count, from the
         *  tile's first isochromat. */
        struct Sensitivities
        {
            const double* re;
            const double* im;
            size_t coils;
            size_t count;
        };

        /** Add to out[c * stride + 2 k], real then imaginary, for each of
         *  @p samples samples k, the sum over @p size isochromats of coil c's
         *  sensitivity times the sample's Mx + i My, at zr[k * pitch] and
         *  zi[k * pitch]. */
        using CoilSum = void (*)(
            const Sensitivities& g,
            size_t size,
            const double* zr,
            const double* zi,
            size_t pitch,
            size_t samples,
            double* out,
            size_t stride);

        /** Coils summed together, sharing each read of the magnetisation. */
        constexpr size_t kCoilBlock = 4;

        /** Samples summed together by the vectorised coil sums, sharing each
         *  read of a sensitivity. */
        constexpr size_t kSampleBlock = 4;

        /** Isochromats one partial sum of the portable coil sum holds. */
        constexpr size_t kLanes = 2;

        /** Add coil @p c's products over isochromats [@p from, @p size) to
         *  (@p sr, @p si). */
        void coil_products(
            const Sensitivities& g,
            size_t c,
            size_t from,
            size_t size,
            const double* zr,
            const double* zi,
            double& sr,
            double& si)
        {
            const double* gr = g.re + c * g.count;
            const double* gi = g.im + c * g.count;
            for (size_t j = from; j < size; ++j)
            {
                sr += gr[j] * zr[j] - gi[j] * zi[j];
                si += gr[j] * zi[j] + gi[j] * zr[j];
            }
        }

        /** Coils [@p c, @p c + Block) of the coil sum, kLanes isochromats at
         *  a time in partial sums of their own. */
        template <size_t Block>
        void coil_block(
            const Sensitivities& g,
            size_t c,
            size_t size,
            const double* zr,
            const double* zi,
            double* out,
            size_t stride)
        {
            const size_t whole = size / kLanes * kLanes;
            double ar[Block][kLanes] = {};
            double ai[Block][kLanes] = {};
            for (size_t j = 0; j < whole; j += kLanes)
                for (size_t q = 0; q < Block; ++q)
                {
                    const double* gr = g.re + (c + q) * g.count + j;
                    const double* gi = g.im + (c + q) * g.count + j;
                    for (size_t l = 0; l < kLanes; ++l)
                    {
                        ar[q][l] += gr[l] * zr[j + l];
                        ar[q][l] -= gi[l] * zi[j + l];
                        ai[q][l] += gr[l] * zi[j + l];
                        ai[q][l] += gi[l] * zr[j + l];
                    }
                }
            for (size_t q = 0; q < Block; ++q)
            {
                double sr = std::accumulate(ar[q], ar[q] + kLanes, 0.0);
                double si = std::accumulate(ai[q], ai[q] + kLanes, 0.0);
                coil_products(g, c + q, whole, size, zr, zi, sr, si);
                out[(c + q) * stride] += sr;
                out[(c + q) * stride + 1] += si;
            }
        }

        void coil_sum(
            const Sensitivities& g,
            size_t size,
            const double* zr,
            const double* zi,
            size_t pitch,
            size_t samples,
            double* out,
            size_t stride)
        {
            for (size_t k = 0; k < samples; ++k)
            {
                size_t c = 0;
                for (; c + kCoilBlock <= g.coils; c += kCoilBlock)
                    coil_block<kCoilBlock>(g, c, size, zr + k * pitch, zi + k * pitch, out + 2 * k, stride);
                for (; c < g.coils; ++c)
                    coil_block<1>(g, c, size, zr + k * pitch, zi + k * pitch, out + 2 * k, stride);
            }
        }

#ifdef BLOCH_X86_64
        BLOCH_AVX2 double lane_sum(__m256d v)
        {
            const __m128d half = _mm_add_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd(v, 1));
            return _mm_cvtsd_f64(_mm_add_sd(half, _mm_unpackhi_pd(half, half)));
        }

        /** Coils [@p c, @p c + Block) of the coil sums of @p Samples samples,
         *  with AVX2 and FMA, four isochromats to a vector. */
        template <size_t Block, size_t Samples>
        BLOCH_AVX2 void coil_block_avx2(
            const Sensitivities& g,
            size_t c,
            size_t size,
            const double* zr,
            const double* zi,
            size_t pitch,
            double* out,
            size_t stride)
        {
            const size_t whole = size / 4 * 4;
            __m256d ar[Block][Samples];
            __m256d ai[Block][Samples];
            for (size_t q = 0; q < Block; ++q)
                for (size_t k = 0; k < Samples; ++k)
                    ar[q][k] = ai[q][k] = _mm256_setzero_pd();
            for (size_t j = 0; j < whole; j += 4)
                for (size_t q = 0; q < Block; ++q)
                {
                    const __m256d gr = _mm256_loadu_pd(g.re + (c + q) * g.count + j);
                    const __m256d gi = _mm256_loadu_pd(g.im + (c + q) * g.count + j);
                    for (size_t k = 0; k < Samples; ++k)
                    {
                        const __m256d xr = _mm256_loadu_pd(zr + k * pitch + j);
                        const __m256d xi = _mm256_loadu_pd(zi + k * pitch + j);
                        ar[q][k] = _mm256_fnmadd_pd(gi, xi, _mm256_fmadd_pd(gr, xr, ar[q][k]));
                        ai[q][k] = _mm256_fmadd_pd(gi, xr, _mm256_fmadd_pd(gr, xi, ai[q][k]));
                    }
                }
            for (size_t q = 0; q < Block; ++q)
                for (size_t k = 0; k < Samples; ++k)
                {
                    double sr = lane_sum(ar[q][k]);
                    double si = lane_sum(ai[q][k]);
                    coil_products(g, c + q, whole, size, zr + k * pitch, zi + k * pitch, sr, si);
                    out[(c + q) * stride + 2 * k] += sr;
                    out[(c + q) * stride + 2 * k + 1] += si;
                }
        }

        /** The coil sums of @p Samples samples, in blocks of @p Block coils. */
        template <size_t Block, size_t Samples>
        BLOCH_AVX2 void coil_sums_avx2(
            const Sensitivities& g,
            size_t size,
            const double* zr,
            const double* zi,
            size_t pitch,
            double* out,
            size_t stride)
        {
            size_t c = 0;
            for (; c + Block <= g.coils; c += Block)
                coil_block_avx2<Block, Samples>(g, c, size, zr, zi, pitch, out, stride);
            for (; c < g.coils; ++c)
                coil_block_avx2<1, Samples>(g, c, size, zr, zi, pitch, out, stride);
        }

        /** The coil sum with AVX2 and FMA, kSampleBlock samples at a time
         *  where there are as many, so that a tile's sensitivities are read
         *  once for them all. */
        BLOCH_AVX2 void coil_sum_avx2(
            const Sensitivities& g,
            size_t size,
            const double* zr,
            const double* zi,
            size_t pitch,
            size_t samples,
            double* out,
            size_t stride)
        {
            size_t k = 0;
            for (; k + kSampleBlock <= samples; k += kSampleBlock)
                coil_sums_avx2<1, kSampleBlock>(g, size, zr + k * pitch, zi + k * pitch, pitch, out + 2 * k, stride);
            for (; k < samples; ++k)
                coil_sums_avx2<kCoilBlock, 1>(g, size, zr + k * pitch, zi + k * pitch, pitch, out + 2 * k, stride);
        }

        /** Coils [@p c, @p c + Block) of the coil sums of @p Samples samples,
         *  with AVX-512, eight isochromats to a vector. */
        template <size_t Block, size_t Samples>
        BLOCH_AVX512 void coil_block_avx512(
            const Sensitivities& g,
            size_t c,
            size_t size,
            const double* zr,
            const double* zi,
            size_t pitch,
            double* out,
            size_t stride)
        {
            const size_t whole = size / 8 * 8;
            __m512d ar[Block][Samples];
            __m512d ai[Block][Samples];
            for (size_t q = 0; q < Block; ++q)
                for (size_t k = 0; k < Samples; ++k)
                    ar[q][k] = ai[q][k] = _mm512_setzero_pd();
            for (size_t j = 0; j < whole; j += 8)
            {
                __m512d xr[Samples];
                __m512d xi[Samples];
                for (size_t k = 0; k < Samples; ++k)
                {
                    xr[k] = _mm512_loadu_pd(zr + k * pitch + j);
                    xi[k] = _mm512_loadu_pd(zi + k * pitch + j);
                }
                for (size_t q = 0; q < Block; ++q)
                {
                    const __m512d gr = _mm512_loadu_pd(g.re + (c + q) * g.count + j);
                    const __m512d gi = _mm512_loadu_pd(g.im + (c + q) * g.count + j);
                    for (size_t k = 0; k < Samples; ++k)
                    {
                        ar[q][k] = _mm512_fnmadd_pd(gi, xi[k], _mm512_fmadd_pd(gr, xr[k], ar[q][k]));
                        ai[q][k] = _mm512_fmadd_pd(gi, xr[k], _mm512_fmadd_pd(gr, xi[k], ai[q][k]));
                    }
                }
            }
            for (size_t q = 0; q < Block; ++q)
                for (size_t k = 0; k < Samples; ++k)
                {
                    double sr = _mm512_reduce_add_pd(ar[q][k]);
                    double si = _mm512_reduce_add_pd(ai[q][k]);
                    coil_products(g, c + q, whole, size, zr + k * pitch, zi + k * pitch, sr, si);
                    out[(c + q) * stride + 2 * k] += sr;
                    out[(c + q) * stride + 2 * k + 1] += si;
                }
        }

        /** The coil sums of @p Samples samples, two coils at a time. */
        template <size_t Samples>
        BLOCH_AVX512 void coil_sums_avx512(
            const Sensitivities& g,
            size_t size,
            const double* zr,
            const double* zi,
            size_t pitch,
            double* out,
            size_t stride)
        {
            size_t c = 0;
            for (; c + 2 <= g.coils; c += 2)
                coil_block_avx512<2, Samples>(g, c, size, zr, zi, pitch, out, stride);
            for (; c < g.coils; ++c)
                coil_block_avx512<1, Samples>(g, c, size, zr, zi, pitch, out, stride);
        }

        /** The coil sum with AVX-512, kSampleBlock samples at a time where
         *  there are as many. */
        BLOCH_AVX512 void coil_sum_avx512(
            const Sensitivities& g,
            size_t size,
            const double* zr,
            const double* zi,
            size_t pitch,
            size_t samples,
            double* out,
            size_t stride)
        {
            size_t k = 0;
            for (; k + kSampleBlock <= samples; k += kSampleBlock)
                coil_sums_avx512<kSampleBlock>(g, size, zr + k * pitch, zi + k * pitch, pitch, out + 2 * k, stride);
            for (; k < samples; ++k)
                coil_sums_avx512<1>(g, size, zr + k * pitch, zi + k * pitch, pitch, out + 2 * k, stride);
        }

#endif

        /** The coil sum this processor runs fastest. */
        CoilSum fastest_coil_sum()
        {
#ifdef BLOCH_X86_64
            static const CoilSum chosen = avx512f() ? coil_sum_avx512 : avx2_and_fma() ? coil_sum_avx2 : coil_sum;
            return chosen;
#else
            return coil_sum;
#endif
        }

        /** Add Mx + i My summed over @p size isochromats to @p out, in
         *  partial sums of four isochromats each. */
        void magnetisation_sum(size_t size, const double* zr, const double* zi, double* out)
        {
            constexpr size_t kSums = 4;
            const size_t whole = size / kSums * kSums;
            double ar[kSums] = {};
            double ai[kSums] = {};
            for (size_t j = 0; j < whole; j += kSums)
                for (size_t l = 0; l < kSums; ++l)
                {
                    ar[l] += zr[j + l];
                    ai[l] += zi[j + l];
                }
            out[0] += std::accumulate(zr + whole, zr + size, std::accumulate(ar, ar + kSums, 0.0));
            out[1] += std::accumulate(zi + whole, zi + size, std::accumulate(ai, ai + kSums, 0.0));
        }

        /** A tile's positions, in m, and off-resonances, in Hz, from its first
         *  isochromat. */
        struct Precession
        {
            const double* x;
            const double* y;
            const double* z;
            const double* f;
        };

        /** Each of @p size isochromats' magnetisation (@p zr, @p zi) turned
         *  over an increment of @p area, in 1/m, and @p time, in s, by
         *  -2 pi (r . area + f time), and times its @p decay, into (@p tr,
         *  @p ti). */
        using AdvanceTile = void (*)(
            const Precession& p,
            const double* area,
            double time,
            const double* decay,
            size_t size,
            const double* zr,
            const double* zi,
            double* tr,
            double* ti);

        BLOCH_INLINE void advance_tile(
            const Precession& p,
            const double* area,
            double time,
            const double* decay,
            size_t size,
            const double* zr,
            const double* zi,
            double* tr,
            double* ti)
        {
            const double a0 = area[0];
            const double a1 = area[1];
            const double a2 = area[2];
            BLOCH_INDEPENDENT
            for (size_t j = 0; j < size; ++j)
            {
                double c = 0.0, s = 0.0;
                cis(-(p.x[j] * a0 + p.y[j] * a1 + p.z[j] * a2 + p.f[j] * time), c, s);
                const double wr = decay[j] * c;
                const double wi = decay[j] * s;
                tr[j] = zr[j] * wr - zi[j] * wi;
                ti[j] = zr[j] * wi + zi[j] * wr;
            }
        }

        void advance_tile_plain(
            const Precession& p,
            const double* area,
            double time,
            const double* decay,
            size_t size,
            const double* zr,
            const double* zi,
            double* tr,
            double* ti)
        {
            advance_tile(p, area, time, decay, size, zr, zi, tr, ti);
        }

#ifdef BLOCH_X86_64
        BLOCH_AVX2 void advance_tile_avx2(
            const Precession& p,
            const double* area,
            double time,
            const double* decay,
            size_t size,
            const double* zr,
            const double* zi,
            double* tr,
            double* ti)
        {
            advance_tile(p, area, time, decay, size, zr, zi, tr, ti);
        }

        BLOCH_AVX512 void advance_tile_avx512(
            const Precession& p,
            const double* area,
            double time,
            const double* decay,
            size_t size,
            const double* zr,
            const double* zi,
            double* tr,
            double* ti)
        {
            advance_tile(p, area, time, decay, size, zr, zi, tr, ti);
        }
#endif

        /** The tile's advance this processor computes fastest. */
        AdvanceTile fastest_advance_tile()
        {
#ifdef BLOCH_X86_64
            static const AdvanceTile chosen =
                avx512f() ? advance_tile_avx512 : avx2_and_fma() ? advance_tile_avx2 : advance_tile_plain;
            return chosen;
#else
            return advance_tile_plain;
#endif
        }

        /** Samples a window reader holds at once, so that their coil sums are
         *  formed together. */
        constexpr size_t kBatch = kSampleBlock;
        static_assert(kBatch > 1, "a sample is turned from the one before it into a row of its own");

        /** One worker's reading of an ADC window, a tile of isochromats at a time. */
        class WindowReader
        {
        public:
            WindowReader(const IsochromatProperties& p, const SampleSteps& steps, size_t samples, const Receive& receive)
                : p_(p), steps_(steps), samples_(samples), coils_(receive.coils), count_(receive.count),
                  re_(receive.re.empty() ? nullptr : receive.re.data()),
                  im_(receive.im.empty() ? nullptr : receive.im.data()), coil_sum_(fastest_coil_sum()),
                  advance_tile_(fastest_advance_tile())
            {
            }

            void tile(size_t first, size_t size, double* mx, double* my, double* sum) const
            {
                /* The magnetisation at a batch of consecutive samples,
                 * [sample][isochromat]; each isochromat's turn to the next
                 * sample where that is one factor; and its decay per sample
                 * where that is. */
                double zr[kBatch][kTile], zi[kBatch][kTile], wr[kTile], wi[kTile], decay[kTile];
                std::copy(mx + first, mx + first + size, zr[0]);
                std::copy(my + first, my + first + size, zi[0]);
                const bool uniform = steps_.uniform && samples_ > 1;
                if (uniform)
                    for (size_t j = 0; j < size; ++j)
                        turn(first + j, &steps_.area[3], steps_.time[1], wr[j], wi[j]);
                else if (steps_.dwell > 0.0)
                    for (size_t j = 0; j < size; ++j)
                        decay[j] = std::exp(-steps_.dwell * rate(p_.t2[first + j]));
                /* The row holding the latest sample. */
                size_t at = 0;
                for (size_t k0 = 0; k0 < samples_; k0 += kBatch)
                {
                    const size_t batch = std::min(kBatch, samples_ - k0);
                    for (size_t b = k0 == 0 ? 1 : 0; b < batch; ++b)
                    {
                        if (uniform)
                            rotate(zr[at], zi[at], wr, wi, zr[b], zi[b], size);
                        else
                            advance(first, size, k0 + b, decay, zr[at], zi[at], zr[b], zi[b]);
                        at = b;
                    }
                    accumulate(zr, zi, first, size, k0, batch, sum);
                }
                std::copy(zr[at], zr[at] + size, mx + first);
                std::copy(zi[at], zi[at] + size, my + first);
            }

        private:
            /** (@p tr, @p ti) = (@p zr, @p zi) times (@p wr, @p wi), a row of
             *  the batch from another. */
            static void rotate(
                const double* zr, const double* zi, const double* wr, const double* wi, double* tr, double* ti, size_t size)
            {
                BLOCH_INDEPENDENT
                for (size_t j = 0; j < size; ++j)
                {
                    tr[j] = zr[j] * wr[j] - zi[j] * wi[j];
                    ti[j] = zr[j] * wi[j] + zi[j] * wr[j];
                }
            }

            void turn(size_t i, const double* area, double time, double& re, double& im) const
            {
                const double phase = -kTwoPi *
                    (p_.x[i] * area[0] + p_.y[i] * area[1] + p_.z[i] * area[2] + p_.off_resonance[i] * time);
                const double e2 = std::exp(-time * rate(p_.t2[i]));
                re = e2 * std::cos(phase);
                im = e2 * std::sin(phase);
            }

            /** The tile's magnetisation turned and decayed from sample @p k - 1
             *  to sample @p k, with @p decay where the dwell is one. */
            void advance(
                size_t first,
                size_t size,
                size_t k,
                double* decay,
                const double* zr,
                const double* zi,
                double* tr,
                double* ti) const
            {
                const double time = steps_.time[k];
                if (!(steps_.dwell > 0.0))
                    for (size_t j = 0; j < size; ++j)
                        decay[j] = std::exp(-time * rate(p_.t2[first + j]));
                const Precession precession{
                    p_.x.data() + first, p_.y.data() + first, p_.z.data() + first, p_.off_resonance.data() + first};
                advance_tile_(precession, &steps_.area[3 * k], time, decay, size, zr, zi, tr, ti);
            }

            void accumulate(
                const double (*zr)[kTile],
                const double (*zi)[kTile],
                size_t first,
                size_t size,
                size_t k0,
                size_t batch,
                double* sum) const
            {
                if (re_ == nullptr)
                    for (size_t b = 0; b < batch; ++b)
                        magnetisation_sum(size, zr[b], zi[b], sum + 2 * (k0 + b));
                else
                    coil_sum_(
                        {re_ + first, im_ + first, coils_, count_},
                        size,
                        zr[0],
                        zi[0],
                        kTile,
                        batch,
                        sum + 2 * k0,
                        2 * samples_);
            }

            const IsochromatProperties& p_;
            const SampleSteps& steps_;
            size_t samples_;
            size_t coils_;
            size_t count_;
            const double* re_;
            const double* im_;
            CoilSum coil_sum_;
            AdvanceTile advance_tile_;
        };

        void check_gradients(const BlockEvents& block)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                const size_t count = block.gradient_corners[axis];
                const double* times = block.gradient_times[axis];
                const double* values = block.gradient_values[axis];
                if (count > 0 && (times == nullptr || values == nullptr))
                    throw std::invalid_argument("a gradient's corners are missing");
                for (size_t i = 0; i < count; ++i)
                    if (!std::isfinite(times[i]) || !std::isfinite(values[i]) || (i > 0 && times[i] < times[i - 1]))
                        throw std::invalid_argument(
                            "a gradient's corners must be finite, at times in increasing order");
            }
        }

        double pulse_end(const BlockEvents& block)
        {
            return block.rf_start + static_cast<double>(block.rf_steps) * block.rf_step;
        }

        bool within(double time, double duration)
        {
            return std::isfinite(time) && time >= -kTimeTolerance && time <= duration + kTimeTolerance;
        }

        void check_channels(const BlockEvents& block, size_t transmit_channels)
        {
            if (transmit_channels != 0 && block.rf_channels != transmit_channels)
                throw std::invalid_argument(
                    "an RF pulse of " + std::to_string(block.rf_channels) + " channels, and transmit sensitivities of " +
                    std::to_string(transmit_channels));
        }

        void check_pulse(const BlockEvents& block, size_t transmit_channels)
        {
            if (block.rf_steps == 0)
                return;
            if (block.rf == nullptr || block.rf_channels == 0)
                throw std::invalid_argument("an RF pulse needs at least one channel of samples");
            if (!(block.rf_step > 0.0) || !std::isfinite(block.rf_step))
                throw std::invalid_argument("an RF pulse's step must be positive");
            if (!within(block.rf_start, block.duration) || !within(pulse_end(block), block.duration))
                throw std::invalid_argument("an RF pulse must play within its block");
            check_channels(block, transmit_channels);
            const double* values = reinterpret_cast<const double*>(block.rf);
            if (!std::all_of(values, values + 2 * block.rf_steps * block.rf_channels, [](double v) { return std::isfinite(v); }))
                throw std::invalid_argument("an RF pulse's samples must be finite");
        }

        /** Check the ADC samples, and return how many come before the pulse. */
        size_t check_samples(const BlockEvents& block)
        {
            const bool pulse = block.rf_steps > 0;
            const double start = block.rf_start;
            const double end = pulse_end(block);
            size_t before = 0;
            for (size_t k = 0; k < block.adc_samples; ++k)
            {
                const double t = block.adc_times[k];
                if (!within(t, block.duration) || (k > 0 && t < block.adc_times[k - 1]))
                    throw std::invalid_argument("ADC samples must be in increasing order within their block");
                if (pulse && t > start + kTimeTolerance && t < end - kTimeTolerance)
                    throw std::invalid_argument("an ADC sample falls inside the RF pulse");
                before = (pulse && t <= start + kTimeTolerance) ? k + 1 : before;
            }
            return before;
        }

        /** Each isochromat's position on which the field during a pulse
         *  depends, in units of the quantum. */
        std::vector<std::array<double, 3>> quantised(const IsochromatProperties& p, int mode, const double direction[3])
        {
            const size_t count = p.x.size();
            std::vector<std::array<double, 3>> positions(count, {0.0, 0.0, 0.0});
            for (size_t i = 0; i < count && mode == kOneDirection; ++i)
                positions[i][0] = std::floor(
                    (p.x[i] * direction[0] + p.y[i] * direction[1] + p.z[i] * direction[2]) / kQuantum + 0.5);
            const std::vector<double>* axes[3] = {&p.x, &p.y, &p.z};
            for (int axis = 0; axis < 3 && mode == kAnyDirection; ++axis)
                for (size_t i = 0; i < count && direction[axis] != 0.0; ++i)
                    positions[i][axis] = std::floor((*axes[axis])[i] / kQuantum + 0.5);
            return positions;
        }

        /** Most points of an axis's lattice, and of the lattice a window is
         *  summed onto; and the bytes the sums may take at once. */
        constexpr size_t kLatticeAxisPoints = size_t(1) << 16;
        constexpr size_t kLatticePoints = size_t(1) << 22;
        constexpr size_t kLatticeBytes = size_t(1) << 28;

        /** Largest distance of a position from its lattice point, relative to
         *  the lattice's spacing, at which it is taken to lie on the point. */
        constexpr double kLatticeDeviation = 1e-6;

        /** Most Chebyshev points across a window read on a lattice. */
        constexpr size_t kMostSegments = 64;

        /** What a window read on a lattice holds to at a tolerance of zero,
         *  relative to the sum of the magnitudes of its terms. */
        constexpr double kExactError = 1e-11;

        /** Lattice transforms and segmentations kept for reuse. */
        constexpr size_t kLatticeTransforms = 32;
        constexpr size_t kSegmentations = 64;

        /** @p count Chebyshev points of the first kind across [0, @p span],
         *  in @p nodes, and their barycentric weights. */
        void chebyshev_points(size_t count, double span, std::vector<double>& nodes, std::vector<double>& weights)
        {
            nodes.resize(count);
            weights.resize(count);
            for (size_t l = 0; l < count; ++l)
            {
                const double angle =
                    0.5 * kTwoPi * (2.0 * static_cast<double>(l) + 1.0) / (2.0 * static_cast<double>(count));
                nodes[l] = 0.5 * span * (1.0 + std::cos(angle));
                weights[l] = (l % 2 == 0 ? 1.0 : -1.0) * std::sin(angle);
            }
        }

        /** The Lagrange basis of @p nodes at @p t, into @p out. */
        void lagrange(const std::vector<double>& nodes, const std::vector<double>& weights, double t, double* out)
        {
            const size_t count = nodes.size();
            const auto hit = std::find(nodes.begin(), nodes.end(), t);
            if (hit != nodes.end())
            {
                std::fill(out, out + count, 0.0);
                out[hit - nodes.begin()] = 1.0;
                return;
            }
            double sum = 0.0;
            for (size_t l = 0; l < count; ++l)
            {
                out[l] = weights[l] / (t - nodes[l]);
                sum += out[l];
            }
            for (size_t l = 0; l < count; ++l)
                out[l] /= sum;
        }

        /** The rate on the boundary of [-a, a] x [-b, b] at @p u of its
         *  perimeter, from 0 to 4, a side per unit. */
        std::complex<double> boundary_rate(double u, double a, double b)
        {
            const int side = static_cast<int>(u);
            const double along = 2.0 * (u - side) - 1.0;
            switch (side)
            {
            case 0:
                return {a * along, -b};
            case 1:
                return {a, b * along};
            case 2:
                return {-a * along, b};
            default:
                return {-a, -b * along};
            }
        }

        /**
         * The largest error of exp(-z t) interpolated between @p count
         * Chebyshev points across [0, @p span], relative to its magnitude,
         * over the times across the window and the rates z on the boundary of
         * [-a, a] x [-b, b], in 1/s and rad/s: the error is analytic in z, so
         * it is largest there.
         */
        double segmentation_error(size_t count, double span, double a, double b)
        {
            std::vector<double> nodes, weights;
            chebyshev_points(count, span, nodes, weights);
            const size_t times = 16 * count + 32;
            std::vector<double> basis(times * count);
            for (size_t i = 0; i < times; ++i)
                lagrange(nodes, weights, span * static_cast<double>(i) / static_cast<double>(times - 1), &basis[i * count]);
            constexpr size_t kPerSide = 64;
            std::vector<std::complex<double>> at(count);
            double worst = 0.0;
            for (size_t e = 0; e < 4 * kPerSide; ++e)
            {
                const std::complex<double> z = boundary_rate(static_cast<double>(e) / kPerSide, a, b);
                for (size_t l = 0; l < count; ++l)
                    at[l] = std::exp(-z * nodes[l]);
                for (size_t i = 0; i < times; ++i)
                {
                    const double t = span * static_cast<double>(i) / static_cast<double>(times - 1);
                    const std::complex<double> value = std::inner_product(
                        at.begin(), at.end(), basis.begin() + static_cast<std::ptrdiff_t>(i * count), std::complex<double>(0.0));
                    worst = std::max(worst, std::abs(value - std::exp(-z * t)) * std::exp(z.real() * t));
                }
            }
            return worst;
        }

        /** Isochromats a lattice sum holds the factors of at once, unless one
         *  lattice point has more. */
        constexpr size_t kSumIsochromats = 768;

        /** Sums a lattice sum holds per tile of lattice points before writing
         *  them out, at least a group's. */
        constexpr size_t kSumValues = size_t(1) << 15;

        /** Lattice points summed together before their sums are laid out in
         *  the tile's rows. */
        constexpr size_t kSumGroup = 8;

        /** Most coils summed together, sharing each read of an isochromat's
         *  factors, and the coils summed together L Chebyshev points at a
         *  time: as many as keep their sums in the registers L lanes come
         *  with. */
        constexpr size_t kSumCoils = 4;
        template <size_t L>
        constexpr size_t kCoilsTogether = L >= 8 ? kSumCoils : kSumCoils / 2;

        /** Coils a tile sums at once, sharing its factors. */
        constexpr size_t kTileCoils = 16;

        /** What summing the isochromats of a range of lattice points reads
         *  and writes. Positions count the isochromats in lattice order. */
        struct LatticeSum
        {
            /** The isochromat at each position, and each point's first
             *  position. */
            const uint32_t* order;
            const uint32_t* starts;
            const double* mx;
            const double* my;
            /** Each position's off-resonance, in Hz, and T2. */
            const double* off_resonance;
            const uint32_t* decay_of;
            /** The chunk's receive sensitivities from its first coil: each
             *  position's coils together, @c stride apart, where @c sorted
             *  holds them; otherwise coil-major, @c count per coil. None for
             *  one coil of unit sensitivity. */
            const std::complex<float>* sorted;
            size_t stride;
            const double* re;
            const double* im;
            size_t count;
            size_t coils;
            double frequency;
            /** Each T2's decay at each Chebyshev point, [T2][padded], and the
             *  points, both zero past the last point. */
            const double* decay;
            const double* nodes;
            size_t segments;
            size_t padded;
            size_t points;
            /** Isochromats in a tile, at least those of the fullest point,
             *  lattice points in a tile, and coils a tile sums at once. */
            size_t tile_isochromats;
            size_t tile_points;
            size_t tile_coils;
            /** [coil][Chebyshev point][lattice point], complex<float> where
             *  single, complex<double> otherwise. */
            void* sums;
            bool single;
        };

        /** A worker's tile of lattice points: its isochromats' factors at the
         *  Chebyshev points [isochromat][padded], zero past the last point,
         *  their magnetisation, and their sensitivities to the coils it sums
         *  at once [isochromat][coil]; a group's sums [point][coil][padded];
         *  and the tile's sums of those coils [coil][Chebyshev point][point],
         *  in the precision they are written in. Real and imaginary parts lie
         *  apart. */
        struct SumTile
        {
            std::vector<double> fr, fi, hr, hi, gr, gi, mr, mi;
            std::vector<std::complex<float>> single_rows;
            std::vector<std::complex<double>> double_rows;

            explicit SumTile(const LatticeSum& s)
                : fr(s.tile_isochromats * s.padded),
                  fi(s.tile_isochromats * s.padded),
                  hr(s.tile_isochromats * s.tile_coils),
                  hi(s.tile_isochromats * s.tile_coils),
                  gr(kSumGroup * kSumCoils * s.padded),
                  gi(kSumGroup * kSumCoils * s.padded),
                  mr(s.tile_isochromats),
                  mi(s.tile_isochromats),
                  single_rows(s.single ? s.tile_coils * s.segments * s.tile_points : 0),
                  double_rows(s.single ? 0 : s.tile_coils * s.segments * s.tile_points)
            {
            }
        };

        /** Sum the isochromats of lattice points [@p first, @p last). */
        using SumPoints = void (*)(const LatticeSum& s, size_t first, size_t last, SumTile& tile);

        /** The factors of the @p n isochromats from position @p base: each
         *  one's transverse magnetisation times its decay and turn at each
         *  Chebyshev point, relative to the window's middle rate and
         *  frequency. */
        BLOCH_INLINE void point_factors(const LatticeSum& s, size_t base, size_t n, SumTile& t)
        {
            const size_t padded = s.padded;
            for (size_t i = 0; i < n; ++i)
            {
                const size_t j = s.order[base + i];
                const double shift = s.frequency - s.off_resonance[base + i];
                const double* d = s.decay + s.decay_of[base + i] * padded;
                double* cycles = &t.fr[i * padded];
                double* decay = &t.fi[i * padded];
                BLOCH_INDEPENDENT
                for (size_t l = 0; l < padded; ++l)
                {
                    cycles[l] = shift * s.nodes[l];
                    decay[l] = d[l];
                }
                t.mr[i] = s.mx[j];
                t.mi[i] = s.my[j];
            }
            double* fr = t.fr.data();
            double* fi = t.fi.data();
            const size_t values = n * padded;
            BLOCH_INDEPENDENT
            for (size_t u = 0; u < values; ++u)
            {
                double c = 0.0, sine = 0.0;
                cis(fr[u], c, sine);
                const double d = fi[u];
                fr[u] = d * c;
                fi[u] = d * sine;
            }
            /* Each factor carries its isochromat's magnetisation. */
            for (size_t i = 0; i < n; ++i)
            {
                const double mr = t.mr[i], mi = t.mi[i];
                double* re = &fr[i * padded];
                double* im = &fi[i * padded];
                BLOCH_INDEPENDENT
                for (size_t l = 0; l < padded; ++l)
                {
                    const double a = re[l], b = im[l];
                    re[l] = mr * a - mi * b;
                    im[l] = mr * b + mi * a;
                }
            }
        }

        /** The sensitivities of the @p n isochromats from position @p base to
         *  the @p k coils from @p first, [isochromat][coil]: read in lattice
         *  order where the sum holds them so, gathered from their coils' rows
         *  otherwise. */
        BLOCH_INLINE void coil_factors(const LatticeSum& s, size_t base, size_t n, size_t first, size_t k, SumTile& t)
        {
            const size_t coils = s.tile_coils;
            if (s.sorted != nullptr)
            {
                for (size_t i = 0; i < n; ++i)
                {
                    const std::complex<float>* g = s.sorted + (base + i) * s.stride + first;
                    double* hr = &t.hr[i * coils];
                    double* hi = &t.hi[i * coils];
                    for (size_t c = 0; c < k; ++c)
                    {
                        hr[c] = g[c].real();
                        hi[c] = g[c].imag();
                    }
                }
                return;
            }
            for (size_t c = 0; c < k; ++c)
            {
                const double* re = s.re != nullptr ? s.re + (first + c) * s.count : nullptr;
                const double* im = s.im != nullptr ? s.im + (first + c) * s.count : nullptr;
                for (size_t i = 0; i < n; ++i)
                {
                    const size_t j = s.order[base + i];
                    t.hr[i * coils + c] = re != nullptr ? re[j] : 1.0;
                    t.hi[i * coils + c] = im != nullptr ? im[j] : 0.0;
                }
            }
        }

        /** Sum the tile's isochromats [@p begin, @p end), those of the
         *  group's point @p at, into the group's sums of K coils from
         *  @p first at every Chebyshev point, L points at a time. The
         *  products of the real and the imaginary parts accumulate apart, so
         *  that no sum waits on the one before it. */
        template <size_t L, size_t K>
        BLOCH_INLINE void sum_coils(
            const LatticeSum& s, size_t begin, size_t end, size_t at, size_t first, SumTile& t)
        {
            using Vector = typename Lanes<double, L>::Vector;
            const size_t padded = s.padded;
            const size_t coils = s.tile_coils;
            for (size_t v = 0; v < padded; v += L)
            {
                Vector rr[K], ii[K], ri[K], ir[K];
                for (size_t k = 0; k < K; ++k)
                    rr[k] = ii[k] = ri[k] = ir[k] = Vector{};
                for (size_t i = begin; i < end; ++i)
                {
                    const Vector fr = *reinterpret_cast<const Vector*>(&t.fr[i * padded + v]);
                    const Vector fi = *reinterpret_cast<const Vector*>(&t.fi[i * padded + v]);
                    const double* hr = &t.hr[i * coils + first];
                    const double* hi = &t.hi[i * coils + first];
                    for (size_t k = 0; k < K; ++k)
                    {
                        rr[k] += hr[k] * fr;
                        ii[k] += hi[k] * fi;
                        ri[k] += hr[k] * fi;
                        ir[k] += hi[k] * fr;
                    }
                }
                for (size_t k = 0; k < K; ++k)
                {
                    const size_t row = (at * kSumCoils + k) * padded + v;
                    *reinterpret_cast<Vector*>(&t.gr[row]) = rr[k] - ii[k];
                    *reinterpret_cast<Vector*>(&t.gi[row]) = ri[k] + ir[k];
                }
            }
        }

        /** Lay the group's sums of @p k coils from @p first, its @p m points
         *  from the tile's point @p at, into the tile's rows. */
        template <typename Real>
        BLOCH_INLINE void stage_rows(
            const LatticeSum& s, size_t at, size_t m, size_t first, size_t k, const SumTile& t, std::complex<Real>* rows)
        {
            for (size_t c = 0; c < k; ++c)
                for (size_t l = 0; l < s.segments; ++l)
                {
                    std::complex<Real>* row = rows + ((first + c) * s.segments + l) * s.tile_points + at;
                    for (size_t p = 0; p < m; ++p)
                    {
                        const size_t from = (p * kSumCoils + c) * s.padded + l;
                        row[p] = std::complex<Real>(static_cast<Real>(t.gr[from]), static_cast<Real>(t.gi[from]));
                    }
                }
        }

        /** Sum the group of lattice points [@p g0, @p g1) of the tile from
         *  point @p q0, whose first isochromat is at position @p base, for
         *  the @p k coils the tile sums at once. */
        template <size_t L>
        BLOCH_INLINE void sum_group(
            const LatticeSum& s, size_t q0, size_t g0, size_t g1, size_t base, size_t k, SumTile& t)
        {
            constexpr size_t K = kCoilsTogether<L>;
            size_t c = 0;
            for (; c + K <= k; c += K)
            {
                for (size_t q = g0; q < g1; ++q)
                    sum_coils<L, K>(s, s.starts[q] - base, s.starts[q + 1] - base, q - g0, c, t);
                if (s.single)
                    stage_rows(s, g0 - q0, g1 - g0, c, K, t, t.single_rows.data());
                else
                    stage_rows(s, g0 - q0, g1 - g0, c, K, t, t.double_rows.data());
            }
            for (; c < k; ++c)
            {
                for (size_t q = g0; q < g1; ++q)
                    sum_coils<L, 1>(s, s.starts[q] - base, s.starts[q + 1] - base, q - g0, c, t);
                if (s.single)
                    stage_rows(s, g0 - q0, g1 - g0, c, 1, t, t.single_rows.data());
                else
                    stage_rows(s, g0 - q0, g1 - g0, c, 1, t, t.double_rows.data());
            }
        }

        /** Write the tile's rows of the @p k coils from @p coil, lattice
         *  points [@p first, @p last), into those coils' rows of the sums. */
        template <typename Real>
        BLOCH_INLINE void write_rows(
            const LatticeSum& s, size_t first, size_t last, size_t coil, size_t k, const std::complex<Real>* rows)
        {
            std::complex<Real>* sums = static_cast<std::complex<Real>*>(s.sums) + coil * s.segments * s.points;
            const size_t n = last - first;
            for (size_t row = 0; row < k * s.segments; ++row)
                std::copy(rows + row * s.tile_points, rows + row * s.tile_points + n, sums + row * s.points + first);
        }

        template <size_t L>
        BLOCH_INLINE void sum_points(const LatticeSum& s, size_t first, size_t last, SumTile& t)
        {
            for (size_t q0 = first; q0 < last;)
            {
                size_t q1 = q0 + 1;
                while (q1 < last && q1 - q0 < s.tile_points && s.starts[q1 + 1] - s.starts[q0] <= s.tile_isochromats)
                    ++q1;
                const size_t base = s.starts[q0];
                const size_t n = s.starts[q1] - base;
                point_factors(s, base, n, t);
                for (size_t coil = 0; coil < s.coils; coil += s.tile_coils)
                {
                    const size_t k = std::min(s.tile_coils, s.coils - coil);
                    coil_factors(s, base, n, coil, k, t);
                    for (size_t g0 = q0; g0 < q1; g0 += kSumGroup)
                        sum_group<L>(s, q0, g0, std::min(q1, g0 + kSumGroup), base, k, t);
                    if (s.single)
                        write_rows(s, q0, q1, coil, k, t.single_rows.data());
                    else
                        write_rows(s, q0, q1, coil, k, t.double_rows.data());
                }
                q0 = q1;
            }
        }

        template <size_t L>
        void sum_points_plain(const LatticeSum& s, size_t first, size_t last, SumTile& t)
        {
            sum_points<L>(s, first, last, t);
        }

#if defined(BLOCH_X86_64) && defined(__GNUC__)
        BLOCH_AVX2 void sum_points_avx2(const LatticeSum& s, size_t first, size_t last, SumTile& t)
        {
            sum_points<4>(s, first, last, t);
        }

        BLOCH_AVX512 void sum_points_avx512(const LatticeSum& s, size_t first, size_t last, SumTile& t)
        {
            sum_points<8>(s, first, last, t);
        }
#endif

        /** The lattice sum this processor computes fastest, and the Chebyshev
         *  points it takes at once. */
        struct LatticeKernel
        {
            SumPoints run;
            size_t lanes;
        };

        LatticeKernel fastest_sum_points()
        {
#if defined(BLOCH_X86_64) && defined(__GNUC__)
            static const bool widest = avx512f();
            static const bool wide = avx2_and_fma();
            if (widest)
                return {sum_points_avx512, 8};
            if (wide)
                return {sum_points_avx2, 4};
#endif
#if defined(__GNUC__)
            return {sum_points_plain<2>, 2};
#else
            return {sum_points_plain<1>, 1};
#endif
        }

        /** The most coils whose sums of @p segments vectors of @p points
         *  values of @p bytes each fit within kLatticeBytes at once, at least
         *  one. */
        size_t coils_at_once(size_t coils, size_t segments, size_t points, size_t bytes)
        {
            const size_t per_coil = segments * points * bytes;
            return std::max<size_t>(1, std::min(coils, kLatticeBytes / std::max<size_t>(per_coil, 1)));
        }

        /**
         * Whether summing onto a lattice of @p modes points along @p axes axes
         * and transforming it costs less than reading the window sample by
         * sample. Costs are counted in one coil's share of turning and
         * summing one isochromat at one sample, the direct reader's; the
         * weights are the stages' times on four threads with AVX-512.
         */
        bool lattice_pays(
            size_t isochromats,
            size_t samples,
            size_t coils,
            size_t segments,
            const int64_t modes[3],
            int axes,
            double tolerance)
        {
            const double width = std::min(16.0, std::ceil(-std::log10(tolerance)) + 2.0);
            const double upsampling = tolerance >= 1e-9 ? 1.25 : 2.0;
            const size_t bytes =
                LatticeTransform::single_for(tolerance) ? sizeof(std::complex<float>) : sizeof(std::complex<double>);
            double grid = 1.0;
            size_t points = 1;
            for (int i = 0; i < axes; ++i)
            {
                grid *= std::max(upsampling * static_cast<double>(modes[i]), 2.0 * width);
                points *= static_cast<size_t>(modes[i]);
            }
            const double n = static_cast<double>(isochromats);
            const double s = static_cast<double>(samples);
            const double c = static_cast<double>(coils);
            const double l = static_cast<double>(segments);
            const double chunks = std::ceil(c / static_cast<double>(coils_at_once(coils, segments, points, bytes)));
            const double direct = n * s * (21.0 + c);
            /* Each chunk's factors at the Chebyshev points, each coil's sums
             * and the isochromats' state after the window; each vector's FFT
             * and interpolation. */
            const double summed = n * l * (32.0 * chunks + 5.0 * c) + 160.0 * n +
                c * l * (3.0 * grid * std::log2(grid) + 2.0 * s * std::pow(width, axes));
            return 1.5 * summed < direct;
        }

    } // namespace

    Isochromats::Isochromats(IsochromatProperties properties, size_t threads)
        : properties_(std::move(properties))
    {
        check();
        threads_ = threads != 0 ? threads : std::max(1u, std::thread::hardware_concurrency());
        lay_out_receive();
        classify();
        reset();
    }

    Isochromats::~Isochromats() = default;

    void Isochromats::check()
    {
        IsochromatProperties& p = properties_;
        count_ = p.x.size();
        if (p.y.size() != count_ || p.z.size() != count_)
            throw std::invalid_argument("x, y and z must hold one position per isochromat");
        if (count_ > std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument("too many isochromats");
        fill(p.proton_density, count_, 1.0, "proton_density");
        fill(p.t1, count_, std::numeric_limits<double>::infinity(), "t1");
        fill(p.t2, count_, std::numeric_limits<double>::infinity(), "t2");
        fill(p.off_resonance, count_, 0.0, "off_resonance");
        for (const auto& named : {std::make_pair(&p.x, "x"), std::make_pair(&p.y, "y"), std::make_pair(&p.z, "z"),
                                  std::make_pair(&p.proton_density, "proton_density"),
                                  std::make_pair(&p.off_resonance, "off_resonance")})
            require_finite(*named.first, named.second);
        require_positive(p.t1);
        require_positive(p.t2);
        require_sensitivities(p.transmit, count_, p.transmit_channels, "transmit");
        if (p.coils != 0 && p.receive == nullptr)
            throw std::invalid_argument("receive must hold one row of sensitivities per isochromat");
        if (p.receive != nullptr)
            require_finite(p.receive, count_ * p.coils);
        coils_ = p.coils == 0 ? 1 : p.coils;
    }

    void Isochromats::lay_out_receive()
    {
        IsochromatProperties& p = properties_;
        receive_re_.resize(count_ * p.coils);
        receive_im_.resize(count_ * p.coils);
        const std::complex<double>* given = p.receive;
        const size_t coils = p.coils;
        /* A block of isochromats at a time, so that the rows it reads stay
         * in cache while each coil's column is written. */
        constexpr size_t kBlock = 128;
        parallel(count_, threads_, kChunk, [&](size_t, size_t first, size_t last) {
            for (size_t block = first; block < last; block += kBlock)
            {
                const size_t end = std::min(last, block + kBlock);
                for (size_t c = 0; c < coils; ++c)
                    for (size_t i = block; i < end; ++i)
                    {
                        receive_re_[c * count_ + i] = given[i * coils + c].real();
                        receive_im_[c * count_ + i] = given[i * coils + c].imag();
                    }
            }
        });
        p.receive = nullptr;
    }

    void Isochromats::classify()
    {
        /* Isochromats equal in everything but position see one field during
         * a pulse wherever their positions along its gradient agree. */
        const IsochromatProperties& p = properties_;
        const size_t width = 3 + 2 * p.transmit_channels;
        std::vector<uint64_t> keys(count_ * width);
        for (size_t i = 0; i < count_; ++i)
        {
            uint64_t* key = &keys[i * width];
            key[0] = bits_of(p.off_resonance[i]);
            key[1] = bits_of(p.t1[i]);
            key[2] = bits_of(p.t2[i]);
            for (size_t c = 0; c < p.transmit_channels; ++c)
            {
                key[3 + 2 * c] = bits_of(p.transmit[i * p.transmit_channels + c].real());
                key[4 + 2 * c] = bits_of(p.transmit[i * p.transmit_channels + c].imag());
            }
        }
        const auto row = [&](uint32_t i) { return keys.cbegin() + static_cast<std::ptrdiff_t>(i * width); };
        std::vector<uint32_t> order(count_);
        std::iota(order.begin(), order.end(), 0u);
        std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return std::lexicographical_compare(row(a), row(a) + width, row(b), row(b) + width);
        });
        class_of_.assign(count_, 0);
        uint32_t current = 0;
        for (size_t n = 1; n < count_; ++n)
        {
            if (!std::equal(row(order[n]), row(order[n]) + width, row(order[n - 1])))
                ++current;
            class_of_[order[n]] = current;
        }

        std::map<std::pair<uint64_t, uint64_t>, uint32_t> relaxations;
        std::map<uint64_t, uint32_t> decays;
        relaxation_of_.assign(count_, 0);
        relaxations_.clear();
        decay_of_.assign(count_, 0);
        decays_.clear();
        for (size_t i = 0; i < count_; ++i)
        {
            const auto made = relaxations.emplace(
                std::make_pair(bits_of(p.t1[i]), bits_of(p.t2[i])), static_cast<uint32_t>(relaxations_.size()));
            if (made.second)
                relaxations_.push_back({p.t1[i], p.t2[i]});
            relaxation_of_[i] = made.first->second;
            const auto decay = decays.emplace(bits_of(p.t2[i]), static_cast<uint32_t>(decays_.size()));
            if (decay.second)
                decays_.push_back(rate(p.t2[i]));
            decay_of_[i] = decay.first->second;
        }
    }

    void Isochromats::reset()
    {
        const std::lock_guard<std::mutex> held(mutex_);
        mx_.assign(count_, 0.0);
        my_.assign(count_, 0.0);
        mz_ = properties_.proton_density;
        std::fill(pending_area_, pending_area_ + 3, 0.0);
        pending_time_ = 0.0;
        elapsed_ = 0.0;
    }

    void Isochromats::magnetization(double* into)
    {
        const std::lock_guard<std::mutex> held(mutex_);
        flush();
        for (size_t i = 0; i < count_; ++i)
        {
            into[3 * i] = mx_[i];
            into[3 * i + 1] = my_[i];
            into[3 * i + 2] = mz_[i];
        }
    }

    void Isochromats::set_magnetization(const double* from)
    {
        const std::lock_guard<std::mutex> held(mutex_);
        for (size_t i = 0; i < count_; ++i)
        {
            mx_[i] = from[3 * i];
            my_[i] = from[3 * i + 1];
            mz_[i] = from[3 * i + 2];
        }
        std::fill(pending_area_, pending_area_ + 3, 0.0);
        pending_time_ = 0.0;
    }

    void Isochromats::set_positions(const double* positions)
    {
        const std::lock_guard<std::mutex> held(mutex_);
        flush();
        IsochromatProperties& p = properties_;
        for (size_t i = 0; i < count_; ++i)
        {
            p.x[i] = positions[3 * i];
            p.y[i] = positions[3 * i + 1];
            p.z[i] = positions[3 * i + 2];
        }
        groupings_.clear();
        held_.clear();
        held_bytes_ = 0;
        for (std::unique_ptr<Lattice>& lattice : lattices_)
            lattice.reset();
        lattice_orders_.clear();
        ++layout_;
    }

    void Isochromats::positions(double* into) const
    {
        const std::lock_guard<std::mutex> held(mutex_);
        const IsochromatProperties& p = properties_;
        for (size_t i = 0; i < count_; ++i)
        {
            into[3 * i] = p.x[i];
            into[3 * i + 1] = p.y[i];
            into[3 * i + 2] = p.z[i];
        }
    }

    void Isochromats::precess(const double* radians)
    {
        const std::lock_guard<std::mutex> held(mutex_);
        flush();
        parallel(count_, threads_, kChunk, [&](size_t, size_t first, size_t last) {
            for (size_t i = first; i < last; ++i)
            {
                const double c = std::cos(radians[i]);
                const double s = -std::sin(radians[i]);
                const double x = mx_[i];
                const double y = my_[i];
                mx_[i] = c * x - s * y;
                my_[i] = s * x + c * y;
            }
        });
    }

    void Isochromats::advance(const GradientAreas& areas, double& now, double to)
    {
        double from_area[3];
        double to_area[3];
        areas.at(now, from_area);
        areas.at(to, to_area);
        for (int axis = 0; axis < 3; ++axis)
            pending_area_[axis] += to_area[axis] - from_area[axis];
        pending_time_ += to - now;
        now = to;
    }

    void Isochromats::flush()
    {
        const double time = pending_time_;
        const double ax = pending_area_[0];
        const double ay = pending_area_[1];
        const double az = pending_area_[2];
        if (time == 0.0 && ax == 0.0 && ay == 0.0 && az == 0.0)
            return;
        const IsochromatProperties& p = properties_;
        parallel(count_, threads_, kChunk, [&](size_t, size_t first, size_t last) {
            for (size_t i = first; i < last; ++i)
            {
                const double phase = -kTwoPi * (p.x[i] * ax + p.y[i] * ay + p.z[i] * az + p.off_resonance[i] * time);
                const double e2 = std::exp(-time * rate(p.t2[i]));
                const double e1 = std::exp(-time * rate(p.t1[i]));
                const double c = e2 * std::cos(phase);
                const double s = e2 * std::sin(phase);
                const double x = mx_[i];
                const double y = my_[i];
                mx_[i] = c * x - s * y;
                my_[i] = s * x + c * y;
                mz_[i] = e1 * mz_[i] + (1.0 - e1) * p.proton_density[i];
            }
        });
        std::fill(pending_area_, pending_area_ + 3, 0.0);
        pending_time_ = 0.0;
    }

    const Isochromats::Grouping& Isochromats::grouping(int mode, const double direction[3])
    {
        const auto known = std::find_if(groupings_.begin(), groupings_.end(), [&](const Grouping& held) {
            return held.mode == mode && same_bits(held.direction, direction);
        });
        if (known != groupings_.end())
            return *known;

        const std::vector<std::array<double, 3>> positions = quantised(properties_, mode, direction);
        const auto less = [&](uint32_t i, uint32_t j) {
            return class_of_[i] != class_of_[j] ? class_of_[i] < class_of_[j] : positions[i] < positions[j];
        };
        std::vector<uint32_t> order(count_);
        std::iota(order.begin(), order.end(), 0u);
        std::stable_sort(order.begin(), order.end(), less);

        Grouping made;
        made.mode = mode;
        std::copy(direction, direction + 3, made.direction);
        made.group_of.assign(count_, 0);
        for (size_t n = 0; n < count_; ++n)
        {
            if (n == 0 || less(order[n - 1], order[n]))
                made.representative.push_back(order[n]);
            made.group_of[order[n]] = static_cast<uint32_t>(made.representative.size() - 1);
        }

        if (groupings_.size() >= kGroupings)
            groupings_.erase(groupings_.begin());
        groupings_.push_back(std::move(made));
        return groupings_.back();
    }

    std::list<Isochromats::HeldPulse>::iterator Isochromats::find_held(
        const BlockEvents& block, const PulseGradient& gradient, std::complex<double>& turn)
    {
        /* A pulse e^{i phi} times a held one, under the same gradient, turns
         * each step's field, and so the held maps, about z by phi. */
        const size_t samples = block.rf_channels * block.rf_steps;
        const auto found = std::find_if(held_.rbegin(), held_.rend(), [&](const HeldPulse& held) {
            return held.mode == gradient.mode && same_bits(held.direction, gradient.direction) &&
                held.step == block.rf_step && held.channels == block.rf_channels && held.rf.size() == samples &&
                held.delta == gradient.delta && one_phase_apart(held.rf, block.rf, turn);
        });
        return found == held_.rend() ? held_.end() : std::prev(found.base());
    }

    Isochromats::HeldPulse Isochromats::held_pulse(
        const BlockEvents& block, const PulseGradient& gradient, const Grouping& groups) const
    {
        HeldPulse made;
        made.mode = gradient.mode;
        std::copy(gradient.direction, gradient.direction + 3, made.direction);
        made.step = block.rf_step;
        made.channels = block.rf_channels;
        made.delta = gradient.delta;
        made.rf.assign(block.rf, block.rf + block.rf_channels * block.rf_steps);
        made.maps.resize(12 * groups.representative.size());
        return made;
    }

    void Isochromats::keep_held(HeldPulse made)
    {
        while (!held_.empty() && (held_.size() >= kHeldPulses || held_bytes_ + made.bytes() > kHeldBytes))
        {
            held_bytes_ -= held_.front().bytes();
            held_.pop_front();
        }
        held_bytes_ += made.bytes();
        held_.push_back(std::move(made));
    }

    void Isochromats::hold(const BlockEvents& block, const PulseGradient& gradient, const Grouping& groups)
    {
        const std::vector<std::complex<double>> summed =
            properties_.transmit_channels == 0 ? channel_sum(block) : std::vector<std::complex<double>>();

        /* Each group's pulse as one affine map, M -> A M + c * proton density. */
        HeldPulse made = held_pulse(block, gradient, groups);
        parallel(groups.representative.size(), threads_, 8, [&](size_t, size_t first, size_t last) {
            for (size_t g = first; g < last; ++g)
                pulse_map(properties_, groups.representative[g], block, gradient, summed, &made.maps[12 * g]);
        });
        keep_held(std::move(made));
    }

    void Isochromats::apply(size_t i, const double* map, std::complex<double> before, std::complex<double> after)
    {
        const double x = before.real() * mx_[i] - before.imag() * my_[i];
        const double y = before.imag() * mx_[i] + before.real() * my_[i];
        const double z = mz_[i];
        const double density = properties_.proton_density[i];
        const double u = map[0] * x + map[1] * y + map[2] * z + map[9] * density;
        const double v = map[3] * x + map[4] * y + map[5] * z + map[10] * density;
        mz_[i] = map[6] * x + map[7] * y + map[8] * z + map[11] * density;
        mx_[i] = after.real() * u - after.imag() * v;
        my_[i] = after.imag() * u + after.real() * v;
    }

    void Isochromats::apply(const std::vector<double>& maps, const Grouping& groups, std::complex<double> turn)
    {
        parallel(count_, threads_, kChunk, [&](size_t, size_t first, size_t last) {
            for (size_t i = first; i < last; ++i)
                apply(i, &maps[12 * groups.group_of[i]], std::conj(turn), turn);
        });
    }

    void Isochromats::excite(const BlockEvents& block, const GradientAreas& areas)
    {
        const PulseGradient gradient = pulse_gradient(block, areas);
        const Grouping& groups = grouping(gradient.mode, gradient.direction);
        std::complex<double> turn = 1.0;
        const auto held = find_held(block, gradient, turn);
        if (held == held_.end())
        {
            double along = 0.0;
            if (!(held_gradient(gradient, along) && hold_on_grid(block, gradient, along, groups)))
                hold(block, gradient, groups);
            turn = 1.0;
        }
        else
            held_.splice(held_.end(), held_, held);
        apply(held_.back().maps, groups, turn);
    }

    std::list<Isochromats::PulseTable>::iterator Isochromats::find_table(
        const std::array<double, 2>& relaxation,
        double step,
        double spacing,
        double drive_spacing,
        const std::vector<std::complex<double>>& rf,
        std::complex<double>& turn)
    {
        return std::find_if(tables_.begin(), tables_.end(), [&](const PulseTable& table) {
            return table.step == step && table.spacing == spacing && table.drive_spacing == drive_spacing &&
                table.rf.size() == rf.size() && bits_of(table.t1) == bits_of(relaxation[0]) &&
                bits_of(table.t2) == bits_of(relaxation[1]) && one_phase_apart(table.rf, rf.data(), turn);
        });
    }

    void Isochromats::extend(PulseTable& table, const std::array<long long, 4>& bounds)
    {
        const Points held = table_points(table.first_row, table.rows, table.first, table.columns);
        const Points made = covering(held, {bounds[0], bounds[1], bounds[2], bounds[3]});
        const std::vector<std::pair<long long, long long>> todo = points_beyond(made, held);
        if (todo.empty())
            return;
        std::vector<double> maps(12 * static_cast<size_t>(made.count()));
        copy_points(table.maps, held, maps, made);
        const bool driven = table.drive_spacing > 0.0;
        parallel(todo.size(), threads_, 8, [&](size_t, size_t begin, size_t end) {
            for (size_t k = begin; k < end; ++k)
            {
                const long long row = todo[k].first;
                const long long column = todo[k].second;
                grid_map(
                    static_cast<double>(column) * table.spacing,
                    table.t1,
                    table.t2,
                    table.rf,
                    driven ? static_cast<double>(row) * table.drive_spacing : 1.0,
                    table.step,
                    &maps[made.at(row, column)]);
            }
        });
        table_bytes_ += (maps.size() - table.maps.size()) * sizeof(double);
        table.maps.swap(maps);
        table.first_row = made.first_row;
        table.first = made.first;
        table.rows = made.rows();
        table.columns = made.columns();
    }

    void Isochromats::table_maps(
        const std::vector<double>& field,
        const std::vector<std::complex<double>>& drive,
        const std::vector<uint32_t>& class_of,
        const std::vector<std::list<PulseTable>::iterator>& table,
        const std::vector<std::complex<double>>& turn,
        double duration,
        std::vector<double>& maps)
    {
        /* Each map, the cubic through the four points around its field, and
         * the quintic through six such around its drive's magnitude, turned
         * by its own precession over half the pulse on either side, by its
         * drive's phase and by the pulse's phase against its table's. */
        parallel(field.size(), threads_, kChunk, [&](size_t, size_t first, size_t last) {
            for (size_t g = first; g < last; ++g)
            {
                const uint32_t c = class_of[g];
                const PulseTable& held = *table[c];
                const double at = field[g] / held.spacing;
                const double point = std::floor(at);
                const size_t column = static_cast<size_t>(static_cast<long long>(point) - 1 - held.first);
                double across[4];
                cubic_weights(at - point, across);
                double map[12];
                std::complex<double> phase = turn[c];
                if (drive.empty())
                    cubic(&held.maps[12 * column], across, map);
                else
                {
                    const double magnitude = std::abs(drive[g]);
                    const double on = magnitude / held.drive_spacing;
                    const double row = std::floor(on);
                    double down[kRows];
                    row_weights(on - row, down);
                    const size_t top =
                        static_cast<size_t>(static_cast<long long>(row) - kRowsBelow - held.first_row);
                    std::fill(map, map + 12, 0.0);
                    for (size_t r = 0; r < kRows; ++r)
                    {
                        double along[12];
                        cubic(&held.maps[12 * ((top + r) * static_cast<size_t>(held.columns) + column)], across, along);
                        for (int e = 0; e < 12; ++e)
                            map[e] += down[r] * along[e];
                    }
                    if (magnitude > 0.0)
                        phase *= drive[g] / magnitude;
                }
                const std::complex<double> precession = std::polar(1.0, -0.5 * kTwoPi * field[g] * duration);
                turned(map, precession * std::conj(phase), precession * phase, &maps[12 * g]);
            }
        });
    }

    bool Isochromats::hold_on_grid(
        const BlockEvents& block, const PulseGradient& gradient, double along, const Grouping& groups)
    {
        const double duration = block.rf_step * static_cast<double>(block.rf_steps);
        const double spacing = 1.0 / (kGridPoints * duration);
        const std::vector<uint32_t>& chosen = groups.representative;
        const size_t count = chosen.size();
        const IsochromatProperties& p = properties_;

        /* With transmit sensitivities, the waveform the channels share, and
         * each group's drive of it: its transmit field is the drive times the
         * waveform, whose response the drive's magnitude scales and whose
         * phase turns it about z. The drive's grid turns the field over the
         * pulse as finely as the field's grid turns the precession. */
        std::vector<std::complex<double>> waveform;
        std::vector<std::complex<double>> drive;
        double drive_spacing = 0.0;
        if (p.transmit_channels == 0)
            waveform = channel_sum(block);
        else
        {
            std::vector<std::complex<double>> weights;
            if (!one_waveform(block, waveform, weights))
                return false;
            const size_t channels = p.transmit_channels;
            drive.resize(count);
            for (size_t g = 0; g < count; ++g)
            {
                std::complex<double> sum = 0.0;
                for (size_t c = 0; c < channels; ++c)
                    sum += p.transmit[chosen[g] * channels + c] * weights[c];
                drive[g] = sum;
            }
            const double integral = std::accumulate(
                waveform.begin(), waveform.end(), 0.0, [](double sum, const std::complex<double>& value) {
                    return sum + std::abs(value);
                });
            drive_spacing = 1.0 / (kGridPoints * block.rf_step * integral);
        }

        const double* direction = gradient.direction;
        std::vector<double> field(count);
        std::vector<uint32_t> class_of(count);
        for (size_t g = 0; g < count; ++g)
        {
            const size_t i = chosen[g];
            field[g] = p.off_resonance[i] + along * (p.x[i] * direction[0] + p.y[i] * direction[1] + p.z[i] * direction[2]);
            class_of[g] = relaxation_of_[i];
        }
        const size_t classes = relaxations_.size();
        std::vector<std::array<long long, 4>> bounds(classes);
        grid_bounds(field, drive, class_of, spacing, drive_spacing, bounds);

        std::vector<std::list<PulseTable>::iterator> table(classes);
        std::vector<std::complex<double>> turn(classes, 1.0);
        size_t missing = 0;
        for (size_t c = 0; c < classes; ++c)
        {
            table[c] = find_table(relaxations_[c], block.rf_step, spacing, drive_spacing, waveform, turn[c]);
            const Points held = table[c] == tables_.end()
                ? table_points(0, 0, 0, 0)
                : table_points(table[c]->first_row, table[c]->rows, table[c]->first, table[c]->columns);
            const Points wanted{bounds[c][0], bounds[c][1], bounds[c][2], bounds[c][3]};
            missing += static_cast<size_t>(covering(held, wanted).count() - held.count());
        }
        if (missing >= count)
            return false;

        for (size_t c = 0; c < classes; ++c)
        {
            if (table[c] == tables_.end())
            {
                PulseTable made;
                made.step = block.rf_step;
                made.t1 = relaxations_[c][0];
                made.t2 = relaxations_[c][1];
                made.spacing = spacing;
                made.drive_spacing = drive_spacing;
                made.rf = waveform;
                table_bytes_ += made.bytes();
                table[c] = tables_.insert(tables_.end(), std::move(made));
                turn[c] = 1.0;
            }
            extend(*table[c], bounds[c]);
        }
        HeldPulse made = held_pulse(block, gradient, groups);
        table_maps(field, drive, class_of, table, turn, duration, made.maps);
        keep(table);
        keep_held(std::move(made));
        return true;
    }

    void Isochromats::keep(const std::vector<std::list<PulseTable>::iterator>& played)
    {
        for (const auto& it : played)
            tables_.splice(tables_.end(), tables_, it);
        while (tables_.size() > played.size() && (tables_.size() > kPulseTables || table_bytes_ > kTableBytes))
        {
            table_bytes_ -= tables_.front().bytes();
            tables_.pop_front();
        }
    }

    void Isochromats::acquire(
        const BlockEvents& block,
        const GradientAreas& areas,
        size_t first,
        size_t last,
        std::complex<double>* signal,
        double tolerance)
    {
        const IsochromatProperties& p = properties_;
        const size_t samples = last - first;
        const double* times = block.adc_times + first;
        const SampleSteps steps = sample_steps(times, samples, areas);
        const double span = times[samples - 1] - times[0];
        const size_t width = tolerance > 0.0 ? Nufft::width_for(tolerance) : Nufft::width_of();
        if (steps.uniform && samples > 1 &&
            read_transformed(&steps.area[3], steps.time[1], samples, span, width, signal + first, block.adc_samples))
            return;
        if (!steps.uniform &&
            read_on_lattice(
                steps.area,
                steps.time,
                samples,
                tolerance > 0.0 ? tolerance : kExactError,
                signal + first,
                block.adc_samples))
            return;
        const Receive receive{receive_re_, receive_im_, coils_, count_};

        const size_t workers = workers_for(count_, threads_, kChunk);
        std::vector<std::vector<double>> sums(workers, std::vector<double>(2 * coils_ * samples, 0.0));
        parallel(count_, threads_, kChunk, [&](size_t worker, size_t begin, size_t end) {
            WindowReader reader(p, steps, samples, receive);
            for (size_t tile = begin; tile < end; tile += kTile)
                reader.tile(tile, std::min(kTile, end - tile), mx_.data(), my_.data(), sums[worker].data());
            for (size_t i = begin; i < end; ++i)
            {
                const double e1 = std::exp(-span * rate(p.t1[i]));
                mz_[i] = e1 * mz_[i] + (1.0 - e1) * p.proton_density[i];
            }
        });

        for (size_t coil = 0; coil < coils_; ++coil)
            for (size_t k = 0; k < samples; ++k)
            {
                const size_t at = 2 * (coil * samples + k);
                signal[coil * block.adc_samples + first + k] = std::accumulate(
                    sums.begin(),
                    sums.end(),
                    std::complex<double>(0.0),
                    [at](const std::complex<double>& total, const std::vector<double>& sum) {
                        return total + std::complex<double>(sum[at], sum[at + 1]);
                    });
            }
    }

    const Nufft& Isochromats::window_transform(size_t samples, size_t width)
    {
        const auto made =
            std::find_if(transforms_.begin(), transforms_.end(), [samples, width](const std::unique_ptr<Nufft>& transform) {
                return transform->modes() == samples && transform->width() == width;
            });
        if (made != transforms_.end())
        {
            transforms_.splice(transforms_.end(), transforms_, made);
            return *transforms_.back();
        }
        if (transforms_.size() >= kTransforms)
            transforms_.pop_front();
        transforms_.push_back(std::unique_ptr<Nufft>(new Nufft(samples, width)));
        return *transforms_.back();
    }

    bool Isochromats::read_transformed(
        const double area[3],
        double step,
        size_t samples,
        double span,
        size_t width,
        std::complex<double>* signal,
        size_t stride)
    {
        if (!transform_pays(samples, width))
            return false;
        const Nufft& transform = window_transform(samples, width);
        const std::vector<std::complex<double>> grid = spread_window(transform, area, step, span);
        finish_window(transform, step, grid, signal, stride);
        return true;
    }

    bool Isochromats::transform_pays(size_t samples, size_t width) const
    {
        /* Under one increment per sample, an isochromat's transverse
         * magnetisation is a geometric series whose ratio's phase is its
         * turn per sample: the isochromats of one T2 sum at each sample to a
         * sum of exponentials at integer frequencies, times their shared
         * decay. Reading sample by sample turns and sums every isochromat at
         * every sample; the transform spreads each once, onto a grid per T2
         * and coil. */
        const size_t decays = decays_.size();
        const size_t grid = Nufft::grid_of(samples, width);
        const size_t workers = workers_for(count_, threads_, kChunk);
        const size_t series = decays * coils_;
        const double direct = static_cast<double>(count_) * static_cast<double>(samples) * static_cast<double>(coils_ + 1);
        const double spread = static_cast<double>(count_) * static_cast<double>(width * (coils_ + 2)) +
            static_cast<double>(series * grid) * (std::log2(static_cast<double>(grid)) + static_cast<double>(workers));
        return decays <= kWindowDecays && 2.0 * spread < direct &&
            workers * series * grid * sizeof(std::complex<double>) <= kWindowBytes;
    }

    std::vector<std::complex<double>> Isochromats::spread_window(
        const Nufft& transform, const double area[3], double step, double span)
    {
        const size_t grid = transform.grid();
        const size_t width = transform.width();
        const size_t series = decays_.size() * coils_;
        const size_t workers = workers_for(count_, threads_, kChunk);
        const IsochromatProperties& p = properties_;
        const bool sensitivities = !receive_re_.empty();
        const double last = static_cast<double>(transform.modes() - 1);
        std::vector<double> kept(decays_.size());
        for (size_t d = 0; d < decays_.size(); ++d)
            kept[d] = std::exp(-span * decays_[d]);
        std::vector<double> recovered(relaxations_.size());
        for (size_t c = 0; c < relaxations_.size(); ++c)
            recovered[c] = std::exp(-span * rate(relaxations_[c][0]));

        /* Each T2's grid holds the coils of a grid point together, so that a
         * term's spread writes consecutive values. */
        std::vector<std::vector<std::complex<double>>> grids(
            workers, std::vector<std::complex<double>>(series * grid));
        parallel(count_, threads_, kChunk, [&](size_t worker, size_t begin, size_t end) {
            std::vector<double> weights(width);
            std::vector<std::complex<double>> values(coils_);
            std::complex<double>* own = grids[worker].data();
            for (size_t i = begin; i < end; ++i)
            {
                const double u = p.x[i] * area[0] + p.y[i] * area[1] + p.z[i] * area[2] + p.off_resonance[i] * step;
                std::complex<double> shift;
                const size_t start = transform.spread(u, weights.data(), shift);
                const std::complex<double> m(mx_[i], my_[i]);
                const std::complex<double> term = m * shift;
                for (size_t c = 0; c < coils_; ++c)
                    values[c] = sensitivities
                        ? term * std::complex<double>(receive_re_[c * count_ + i], receive_im_[c * count_ + i])
                        : term;
                const double* value = reinterpret_cast<const double*>(values.data());
                double* points = reinterpret_cast<double*>(own + decay_of_[i] * grid * coils_);
                size_t at = start;
                for (size_t j = 0; j < width; ++j)
                {
                    double* point = points + 2 * at * coils_;
                    for (size_t k = 0; k < 2 * coils_; ++k)
                        point[k] += weights[j] * value[k];
                    at = at + 1 == grid ? 0 : at + 1;
                }
                /* The window leaves the isochromat as it stands at its last
                 * sample. */
                const double fraction = u - std::nearbyint(u);
                const std::complex<double> turned =
                    m * std::polar(kept[decay_of_[i]], -kTwoPi * fraction * last);
                mx_[i] = turned.real();
                my_[i] = turned.imag();
                const double e1 = recovered[relaxation_of_[i]];
                mz_[i] = e1 * mz_[i] + (1.0 - e1) * p.proton_density[i];
            }
        });
        for (size_t worker = 1; worker < workers; ++worker)
            for (size_t k = 0; k < grids[0].size(); ++k)
                grids[0][k] += grids[worker][k];
        return std::move(grids[0]);
    }

    void Isochromats::finish_window(
        const Nufft& transform,
        double step,
        const std::vector<std::complex<double>>& spread,
        std::complex<double>* signal,
        size_t stride)
    {
        const size_t grid = transform.grid();
        const size_t samples = transform.modes();
        const size_t decays = decays_.size();
        std::vector<double> decay(decays * samples);
        for (size_t d = 0; d < decays; ++d)
            for (size_t k = 0; k < samples; ++k)
                decay[d * samples + k] = std::exp(-decays_[d] * step * static_cast<double>(k));
        parallel(coils_, threads_, 1, [&](size_t, size_t first, size_t last) {
            std::vector<std::complex<double>> column(grid);
            std::vector<std::complex<double>> modes(samples);
            for (size_t c = first; c < last; ++c)
            {
                std::complex<double>* out = signal + c * stride;
                std::fill(out, out + samples, std::complex<double>(0.0));
                for (size_t d = 0; d < decays; ++d)
                {
                    for (size_t k = 0; k < grid; ++k)
                        column[k] = spread[(d * grid + k) * coils_ + c];
                    transform.finish(column.data(), modes.data());
                    for (size_t k = 0; k < samples; ++k)
                        out[k] += decay[d * samples + k] * modes[k];
                }
            }
        });
    }

    void Isochromats::play(const BlockEvents& block, std::complex<double>* signal, double tolerance)
    {
        if (!(block.duration >= 0.0) || !std::isfinite(block.duration))
            throw std::invalid_argument("a block's duration must be finite and not negative");
        check_gradients(block);
        check_pulse(block, properties_.transmit_channels);
        const size_t before = check_samples(block);
        const size_t samples = block.adc_samples;
        if (samples > 0 && signal == nullptr)
            throw std::invalid_argument("no room for the signal");

        const std::lock_guard<std::mutex> held(mutex_);
        const GradientAreas areas(block);
        double now = 0.0;
        if (before > 0)
        {
            advance(areas, now, block.adc_times[0]);
            flush();
            acquire(block, areas, 0, before, signal, tolerance);
            now = block.adc_times[before - 1];
        }
        if (block.rf_steps > 0)
        {
            advance(areas, now, block.rf_start);
            flush();
            excite(block, areas);
            now = block.rf_start + static_cast<double>(block.rf_steps) * block.rf_step;
        }
        if (before < samples)
        {
            advance(areas, now, block.adc_times[before]);
            flush();
            acquire(block, areas, before, samples, signal, tolerance);
            now = block.adc_times[samples - 1];
        }
        advance(areas, now, block.duration);
        elapsed_ += block.duration;
    }

    void Isochromats::play_quietly(const BlockEvents& block, std::complex<double>* first_sample)
    {
        if (!(block.duration >= 0.0) || !std::isfinite(block.duration))
            throw std::invalid_argument("a block's duration must be finite and not negative");
        check_gradients(block);
        check_pulse(block, properties_.transmit_channels);
        const size_t before = check_samples(block);
        const size_t samples = block.adc_samples;
        if (before > 0 && before < samples)
            throw std::invalid_argument("a repeated block's ADC samples must lie on one side of its RF pulse");
        const GradientAreas areas(block);
        double now = 0.0;
        const auto capture = [&]() {
            advance(areas, now, block.adc_times[0]);
            flush();
            if (first_sample != nullptr)
                for (size_t i = 0; i < count_; ++i)
                    first_sample[i] = std::complex<double>(mx_[i], my_[i]);
        };
        if (before > 0)
            capture();
        if (block.rf_steps > 0)
        {
            advance(areas, now, block.rf_start);
            flush();
            excite(block, areas);
            now = pulse_end(block);
        }
        if (samples > 0 && before == 0)
            capture();
        advance(areas, now, block.duration);
        elapsed_ += block.duration;
    }

    bool Isochromats::window_steps(const BlockEvents& block, double area[3], double& step) const
    {
        std::fill(area, area + 3, 0.0);
        step = 0.0;
        if (block.adc_samples < 2)
            return true;
        const GradientAreas areas(block);
        const SampleSteps steps = sample_steps(block.adc_times, block.adc_samples, areas);
        std::copy(&steps.area[3], &steps.area[6], area);
        step = steps.time[1];
        return steps.uniform;
    }

    const Isochromats::Lattice& Isochromats::lattice(int axis)
    {
        std::unique_ptr<Lattice>& held = lattices_[static_cast<size_t>(axis)];
        if (held)
            return *held;
        held.reset(new Lattice());
        Lattice& made = *held;
        const std::vector<double>& values = axis == 0 ? properties_.x : axis == 1 ? properties_.y : properties_.z;
        std::vector<double> sorted(values);
        std::sort(sorted.begin(), sorted.end());
        const double range = sorted.back() - sorted.front();
        made.origin = sorted.front();
        if (!(range > 0.0))
        {
            made.spacing = 1.0;
            made.points = 1;
            made.index.assign(values.size(), 0);
            return made;
        }
        /* The least gap between distinct positions, which tells positions
         * rounded apart from one point from its neighbours. */
        double gap = range;
        for (size_t i = 1; i < sorted.size(); ++i)
            if (sorted[i] - sorted[i - 1] > kLatticeDeviation * range)
                gap = std::min(gap, sorted[i] - sorted[i - 1]);
        const double steps = std::nearbyint(range / gap);
        if (!(steps < static_cast<double>(kLatticeAxisPoints)))
            return made;
        const double spacing = range / steps;
        std::vector<uint32_t> index(values.size());
        double deviation = 0.0;
        for (size_t i = 0; i < values.size(); ++i)
        {
            const double n = std::nearbyint((values[i] - made.origin) / spacing);
            const double off = std::abs(values[i] - (made.origin + n * spacing));
            if (off > kLatticeDeviation * spacing)
                return made;
            deviation = std::max(deviation, off);
            index[i] = static_cast<uint32_t>(n);
        }
        made.spacing = spacing;
        made.points = static_cast<size_t>(steps) + 1;
        made.deviation = deviation;
        made.index = std::move(index);
        return made;
    }

    Isochromats::LatticeOrder& Isochromats::lattice_order(unsigned axes)
    {
        for (const std::unique_ptr<LatticeOrder>& made : lattice_orders_)
            if (made->axes == axes)
                return *made;
        std::unique_ptr<LatticeOrder> made(new LatticeOrder());
        made->axes = axes;
        std::vector<size_t> key(count_, 0);
        size_t points = 1;
        for (int axis = 0; axis < 3; ++axis)
        {
            if ((axes & (1u << axis)) == 0)
                continue;
            const Lattice& along = lattice(axis);
            for (size_t i = 0; i < count_; ++i)
                key[i] += points * along.index[i];
            points *= along.points;
        }
        made->starts.assign(points + 1, 0);
        for (size_t i = 0; i < count_; ++i)
            ++made->starts[key[i] + 1];
        std::partial_sum(made->starts.begin(), made->starts.end(), made->starts.begin());
        std::vector<uint32_t> next(made->starts.begin(), made->starts.end() - 1);
        made->order.resize(count_);
        for (size_t i = 0; i < count_; ++i)
            made->order[next[key[i]]++] = static_cast<uint32_t>(i);
        for (size_t q = 0; q < points; ++q)
            made->fullest = std::max<size_t>(made->fullest, made->starts[q + 1] - made->starts[q]);
        made->off_resonance.resize(count_);
        made->decay_of.resize(count_);
        for (size_t k = 0; k < count_; ++k)
        {
            made->off_resonance[k] = properties_.off_resonance[made->order[k]];
            made->decay_of[k] = decay_of_[made->order[k]];
        }
        lattice_orders_.push_back(std::move(made));
        return *lattice_orders_.back();
    }

    size_t Isochromats::segments_for(double span, double error)
    {
        for (const Segmentation& made : segmentations_)
            if (made.span == span && made.error == error)
                return made.count;
        const auto decays = std::minmax_element(decays_.begin(), decays_.end());
        const auto fields = std::minmax_element(properties_.off_resonance.begin(), properties_.off_resonance.end());
        const double a = 0.5 * (*decays.second - *decays.first);
        const double b = 0.5 * kTwoPi * (*fields.second - *fields.first);
        /* exp(-i b s) over s in [-1, 1] takes more Chebyshev points than b. */
        size_t count = 0;
        for (size_t n = std::max<size_t>(1, static_cast<size_t>(0.5 * b * span)); n <= kMostSegments && count == 0; ++n)
            if (segmentation_error(n, span, a, b) <= error)
                count = n;
        if (segmentations_.size() >= kSegmentations)
            segmentations_.erase(segmentations_.begin());
        segmentations_.push_back({span, error, count});
        return count;
    }

    bool Isochromats::lattice_window(
        const std::vector<double>& area,
        const std::vector<double>& time,
        size_t samples,
        double error,
        LatticeWindow& window)
    {
        window.k.assign(3 * samples, 0.0);
        window.time.assign(samples, 0.0);
        for (size_t s = 1; s < samples; ++s)
        {
            window.time[s] = window.time[s - 1] + time[s];
            for (size_t axis = 0; axis < 3; ++axis)
                window.k[3 * s + axis] = window.k[3 * (s - 1) + axis] + area[3 * s + axis];
        }
        window.span = window.time[samples - 1];
        const std::vector<double>* positions[3] = {&properties_.x, &properties_.y, &properties_.z};
        double reference[3];
        double most[3];
        for (int axis = 0; axis < 3; ++axis)
        {
            const auto range = std::minmax_element(positions[axis]->begin(), positions[axis]->end());
            reference[axis] = 0.5 * (*range.first + *range.second);
            most[axis] = 0.0;
            for (size_t s = 0; s < samples; ++s)
                most[axis] = std::max(most[axis], std::abs(window.k[3 * s + static_cast<size_t>(axis)]));
            /* An axis whose k stays within this of zero turns every term by
             * its reference's phase alone. */
            if (kTwoPi * 0.5 * (*range.second - *range.first) * most[axis] <= 0.125 * error)
                continue;
            const Lattice& along = lattice(axis);
            if (along.spacing == 0.0 || kTwoPi * along.deviation * most[axis] > 0.125 * error)
                return false;
            window.along[window.axes] = axis;
            window.modes[window.axes++] = static_cast<int64_t>(along.points);
            window.mask |= 1u << axis;
            window.points *= along.points;
            reference[axis] = along.origin + static_cast<double>(along.points / 2) * along.spacing;
        }
        if (window.axes == 0 || window.points > kLatticePoints)
            return false;
        window.centre.assign(samples, 0.0);
        for (size_t s = 0; s < samples; ++s)
            for (size_t axis = 0; axis < 3; ++axis)
                window.centre[s] += reference[axis] * window.k[3 * s + axis];
        const auto decays = std::minmax_element(decays_.begin(), decays_.end());
        const auto fields = std::minmax_element(properties_.off_resonance.begin(), properties_.off_resonance.end());
        window.rate = 0.5 * (*decays.first + *decays.second);
        window.frequency = 0.5 * (*fields.first + *fields.second);
        return true;
    }

    template <typename Real>
    void Isochromats::sum_onto_lattice(
        const LatticeWindow& window,
        const Segments& segments,
        size_t first_coil,
        size_t coils,
        std::complex<Real>* sums)
    {
        LatticeOrder& sorted = lattice_order(window.mask);
        const LatticeKernel kernel = fastest_sum_points();
        const size_t count = segments.count;
        const size_t padded = (count + kernel.lanes - 1) / kernel.lanes * kernel.lanes;
        const size_t points = window.points;
        /* The Chebyshev points and each T2's decay at them, relative to the
         * window's middle rate, zero past the last. */
        std::vector<double> nodes(padded, 0.0);
        std::copy(segments.nodes.begin(), segments.nodes.end(), nodes.begin());
        std::vector<double> decay(decays_.size() * padded, 0.0);
        for (size_t d = 0; d < decays_.size(); ++d)
            for (size_t l = 0; l < count; ++l)
                decay[d * padded + l] = std::exp(-(decays_[d] - window.rate) * nodes[l]);
        const bool sensitivities = !receive_re_.empty();
        /* Sums in single precision read the sensitivities from a copy in
         * lattice order, in single precision, half the memory of the coils'
         * rows; sums in double precision gather them from those rows. */
        const bool single = std::is_same<Real, float>::value;
        if (single && sensitivities && sorted.receive.empty())
        {
            sorted.receive.resize(count_ * coils_);
            parallel(count_, threads_, kChunk, [&](size_t, size_t begin, size_t end) {
                for (size_t k = begin; k < end; ++k)
                {
                    const size_t j = sorted.order[k];
                    std::complex<float>* row = &sorted.receive[k * coils_];
                    for (size_t c = 0; c < coils_; ++c)
                        row[c] = std::complex<float>(
                            static_cast<float>(receive_re_[c * count_ + j]), static_cast<float>(receive_im_[c * count_ + j]));
                }
            });
        }
        const bool gathered = sensitivities && !single;
        const LatticeSum sum{
            sorted.order.data(),
            sorted.starts.data(),
            mx_.data(),
            my_.data(),
            sorted.off_resonance.data(),
            sorted.decay_of.data(),
            sensitivities && single ? sorted.receive.data() + first_coil : nullptr,
            coils_,
            gathered ? receive_re_.data() + first_coil * count_ : nullptr,
            gathered ? receive_im_.data() + first_coil * count_ : nullptr,
            count_,
            coils,
            window.frequency,
            decay.data(),
            nodes.data(),
            count,
            padded,
            points,
            std::max(kSumIsochromats, sorted.fullest),
            std::max(kSumGroup, kSumValues / (std::min(coils, kTileCoils) * count) / kSumGroup * kSumGroup),
            std::min(coils, kTileCoils),
            sums,
            single};
        const auto starts_end = sorted.starts.begin() + static_cast<std::ptrdiff_t>(points);
        parallel(count_, threads_, kChunk, [&](size_t, size_t begin, size_t end) {
            /* A worker sums the points whose first isochromat it was given. */
            const size_t first =
                static_cast<size_t>(std::lower_bound(sorted.starts.begin(), starts_end, begin) - sorted.starts.begin());
            const size_t last = end == count_
                ? points
                : static_cast<size_t>(std::lower_bound(sorted.starts.begin(), starts_end, end) - sorted.starts.begin());
            SumTile tile(sum);
            kernel.run(sum, first, last, tile);
        });
    }

    LatticeTransform& Isochromats::lattice_transform(
        int axes, const int64_t modes[3], int vectors, double tolerance, size_t worker)
    {
        const auto made = std::find_if(
            lattice_transforms_.begin(), lattice_transforms_.end(), [&](const LatticeTransformHeld& held) {
                const LatticeTransform& plan = *held.plan;
                return held.worker == worker && plan.dimensions() == axes &&
                    std::equal(modes, modes + axes, plan.modes()) && plan.vectors() == vectors &&
                    plan.tolerance() == tolerance;
            });
        if (made != lattice_transforms_.end())
        {
            lattice_transforms_.splice(lattice_transforms_.end(), lattice_transforms_, made);
            return *lattice_transforms_.back().plan;
        }
        if (lattice_transforms_.size() >= kLatticeTransforms)
            lattice_transforms_.pop_front();
        /* One thread: FINUFFT's own threads do not survive fork(). */
        lattice_transforms_.push_back(
            {worker, std::unique_ptr<LatticeTransform>(new LatticeTransform(axes, modes, vectors, tolerance, 1))});
        return *lattice_transforms_.back().plan;
    }

    template <typename Real>
    void Isochromats::transform_vectors(
        const LatticeWindow& window,
        const std::vector<std::vector<double>>& x,
        double tolerance,
        size_t vectors,
        std::complex<Real>* modes,
        std::complex<Real>* sums)
    {
        const size_t samples = window.time.size();
        const size_t workers = std::min(threads_, vectors);
        std::vector<LatticeTransform*> plans(workers);
        for (size_t w = 0; w < workers; ++w)
        {
            const size_t share = (w + 1) * vectors / workers - w * vectors / workers;
            plans[w] = &lattice_transform(window.axes, window.modes, static_cast<int>(share), tolerance, w);
            plans[w]->points(
                static_cast<int64_t>(samples),
                x[0].data(),
                window.axes > 1 ? x[1].data() : nullptr,
                window.axes > 2 ? x[2].data() : nullptr);
        }
        parallel(workers, workers, 1, [&](size_t, size_t begin, size_t end) {
            for (size_t w = begin; w < end; ++w)
            {
                const size_t first = w * vectors / workers;
                plans[w]->execute(modes + first * window.points, sums + first * samples);
            }
        });
    }

    void Isochromats::settle_after(const LatticeWindow& window)
    {
        const IsochromatProperties& p = properties_;
        const size_t last = window.time.size() - 1;
        const double* k = &window.k[3 * last];
        std::vector<double> kept(decays_.size());
        for (size_t d = 0; d < decays_.size(); ++d)
            kept[d] = std::exp(-window.span * decays_[d]);
        std::vector<double> recovered(relaxations_.size());
        for (size_t c = 0; c < relaxations_.size(); ++c)
            recovered[c] = std::exp(-window.span * rate(relaxations_[c][0]));
        parallel(count_, threads_, kChunk, [&](size_t, size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
            {
                double c = 0.0, s = 0.0;
                cis(-(p.x[i] * k[0] + p.y[i] * k[1] + p.z[i] * k[2] + p.off_resonance[i] * window.span), c, s);
                const double e2 = kept[decay_of_[i]];
                const double r = e2 * (mx_[i] * c - my_[i] * s);
                my_[i] = e2 * (mx_[i] * s + my_[i] * c);
                mx_[i] = r;
                const double e1 = recovered[relaxation_of_[i]];
                mz_[i] = e1 * mz_[i] + (1.0 - e1) * p.proton_density[i];
            }
        });
    }

    template <typename Real>
    void Isochromats::transform_lattice(
        const LatticeWindow& window,
        const Segments& segments,
        const std::vector<std::vector<double>>& x,
        double tolerance,
        std::vector<std::complex<double>>& out)
    {
        std::vector<std::complex<Real>>* buffers[2] = {nullptr, nullptr};
        if constexpr (std::is_same<Real, float>::value)
            buffers[0] = &single_sums_, buffers[1] = &single_values_;
        else
            buffers[0] = &lattice_sums_, buffers[1] = &lattice_values_;
        std::vector<std::complex<Real>>& sums = *buffers[0];
        std::vector<std::complex<Real>>& values = *buffers[1];
        const size_t samples = window.time.size();
        const size_t count = segments.count;
        const size_t at_once = coils_at_once(coils_, count, window.points, sizeof(std::complex<Real>));
        for (size_t first = 0; first < coils_; first += at_once)
        {
            const size_t coils = std::min(at_once, coils_ - first);
            /* Every value is written before it is read. */
            if (sums.size() < coils * count * window.points)
                sums.resize(coils * count * window.points);
            if (values.size() < coils * count * samples)
                values.resize(coils * count * samples);
            sum_onto_lattice(window, segments, first, coils, sums.data());
            transform_vectors(window, x, tolerance, coils * count, sums.data(), values.data());
            for (size_t c = 0; c < coils; ++c)
                for (size_t l = 0; l < count; ++l)
                    for (size_t s = 0; s < samples; ++s)
                        out[(first + c) * samples + s] += segments.basis[s * count + l] *
                            std::complex<double>(values[(c * count + l) * samples + s]);
        }
    }

    void Isochromats::use_lattice_device(LatticeDevice device)
    {
        const std::lock_guard<std::mutex> held(mutex_);
        lattice_device_ = std::move(device);
    }

    bool Isochromats::read_on_device(
        const LatticeWindow& window,
        const Segments& segments,
        const std::vector<std::vector<double>>& x,
        double tolerance,
        std::vector<std::complex<double>>& out)
    {
        const LatticeOrder& sorted = lattice_order(window.mask);
        const size_t count = segments.count;
        std::vector<double> decay(decays_.size() * count);
        for (size_t d = 0; d < decays_.size(); ++d)
            for (size_t l = 0; l < count; ++l)
                decay[d * count + l] = std::exp(-(decays_[d] - window.rate) * segments.nodes[l]);
        const bool sensitivities = !receive_re_.empty();
        LatticeWindowRead read;
        read.layout = layout_;
        read.axes = window.mask;
        read.dimensions = window.axes;
        read.modes = window.modes;
        read.points = window.points;
        read.isochromats = count_;
        read.order = sorted.order.data();
        read.starts = sorted.starts.data();
        read.off_resonance = sorted.off_resonance.data();
        read.decay_of = sorted.decay_of.data();
        read.coils = coils_;
        read.receive_re = sensitivities ? receive_re_.data() : nullptr;
        read.receive_im = sensitivities ? receive_im_.data() : nullptr;
        read.mx = mx_.data();
        read.my = my_.data();
        read.frequency = window.frequency;
        read.segments = count;
        read.nodes = segments.nodes.data();
        read.decays = decays_.size();
        read.decay = decay.data();
        read.samples = window.time.size();
        read.basis = segments.basis.data();
        for (size_t i = 0; i < x.size(); ++i)
            read.x[i] = x[i].data();
        read.tolerance = tolerance;
        read.single = LatticeTransform::single_for(tolerance);
        read.out = out.data();
        return lattice_device_.read(read);
    }

    bool Isochromats::read_on_lattice(
        const std::vector<double>& area,
        const std::vector<double>& time,
        size_t samples,
        double error,
        std::complex<double>* signal,
        size_t stride)
    {
        LatticeWindow window;
        if (samples < 2 || !finufft_ready() || !lattice_window(area, time, samples, error, window))
            return false;
        Segments segments;
        segments.count = segments_for(window.span, 0.25 * error);
        if (segments.count == 0)
            return false;
        const size_t count = segments.count;
        std::vector<double> weights;
        chebyshev_points(count, window.span, segments.nodes, weights);
        segments.basis.resize(samples * count);
        double lebesgue = 0.0;
        for (size_t s = 0; s < samples; ++s)
        {
            double* at = &segments.basis[s * count];
            lagrange(segments.nodes, weights, window.time[s], at);
            lebesgue = std::max(lebesgue, std::accumulate(at, at + count, 0.0, [](double sum, double b) {
                return sum + std::abs(b);
            }));
        }
        /* The sums at the Chebyshev points grow from the window's middle rate
         * by at most the spread of 1/T2 over the window. */
        const auto decays = std::minmax_element(decays_.begin(), decays_.end());
        const double tolerance =
            0.25 * error / (lebesgue * std::exp((*decays.second - *decays.first) * window.span));

        std::vector<std::vector<double>> x(static_cast<size_t>(window.axes), std::vector<double>(samples));
        for (int i = 0; i < window.axes; ++i)
        {
            const double spacing = lattice(window.along[i]).spacing;
            for (size_t s = 0; s < samples; ++s)
            {
                const double v = spacing * window.k[3 * s + static_cast<size_t>(window.along[i])];
                x[static_cast<size_t>(i)][s] = kTwoPi * (v - std::nearbyint(v));
            }
        }
        std::vector<std::complex<double>> out(coils_ * samples, 0.0);
        if (lattice_device_.read && read_on_device(window, segments, x, tolerance, out))
        {
            /* The device reads the window while the isochromats move on. */
            settle_after(window);
            if (lattice_device_.finish)
                lattice_device_.finish();
            ++device_windows_;
        }
        else if (!lattice_pays(count_, samples, coils_, count, window.modes, window.axes, tolerance))
            return false;
        else
        {
            if (LatticeTransform::single_for(tolerance))
                transform_lattice<float>(window, segments, x, tolerance, out);
            else
                transform_lattice<double>(window, segments, x, tolerance, out);
            settle_after(window);
        }
        for (size_t s = 0; s < samples; ++s)
        {
            double c = 0.0, sine = 0.0;
            cis(-(window.frequency * window.time[s] + window.centre[s]), c, sine);
            const std::complex<double> factor = std::exp(-window.rate * window.time[s]) * std::complex<double>(c, sine);
            for (size_t coil = 0; coil < coils_; ++coil)
                signal[coil * stride + s] = out[coil * samples + s] * factor;
        }
        ++lattice_windows_;
        return true;
    }

} // namespace bloch
