/**
 * @file nufft.cpp
 * @brief A type-1 non-uniform FFT in one dimension.  See nufft.hpp.
 */

#include "bloch/nufft.hpp"

#include <algorithm>
#include <cmath>

#include "third_party/pocketfft_hdronly.h"

namespace bloch
{

    namespace
    {

        constexpr double kTwoPi = 6.283185307179586476925286766559;

        /** Grid points a term is spread onto at the widest: the width the
         *  reference gives a grid of twice the modes for a tolerance of
         *  1e-12, whose kernel's rate is 2.30 times its width. */
        constexpr size_t kWidth = 13;
        constexpr double kRatePerPoint = 2.30;

        /** Points of the kernel's table per grid spacing. */
        constexpr size_t kTablePoints = 512;

        /** Points per grid spacing of the rule the kernel's transform is
         *  integrated by. */
        constexpr size_t kQuadraturePoints = 64;

        /** The exponential-of-semicircle kernel @p width grid spacings wide
         *  at @p z grid spacings from its centre. */
        double kernel(double z, size_t width)
        {
            const double x = 2.0 * z / static_cast<double>(width);
            if (!(std::abs(x) < 1.0))
                return 0.0;
            return std::exp(kRatePerPoint * static_cast<double>(width) * (std::sqrt(1.0 - x * x) - 1.0));
        }

    } // namespace

    struct Nufft::Plan
    {
        explicit Plan(size_t length) : fft(length)
        {
        }
        pocketfft::detail::pocketfft_c<double> fft;
    };

    size_t Nufft::grid_of(size_t modes, size_t width)
    {
        return 2 * std::max(modes, width);
    }

    size_t Nufft::width_of()
    {
        return kWidth;
    }

    size_t Nufft::width_for(double tolerance)
    {
        /* The reference's rule for a grid of twice the modes: a width of w
         * holds to about 10^(1 - w). */
        if (!(tolerance > 1e-12))
            return kWidth;
        const double points = std::ceil(-std::log10(tolerance / 10.0));
        return std::min(kWidth, static_cast<size_t>(std::max(2.0, points)));
    }

    Nufft::Nufft(size_t modes, size_t width)
        : modes_(modes), grid_(grid_of(modes, width)), width_(width), centre_(modes / 2), plan_(new Plan(grid_))
    {
        /* Table point m + 1 holds the kernel at -width / 2 + m / kTablePoints,
         * m from -1 on, so that the cubic around any point of the support
         * reads within the table. */
        const double half = 0.5 * static_cast<double>(width_);
        table_.resize(width_ * kTablePoints + 4);
        for (size_t t = 0; t < table_.size(); ++t)
            table_[t] = kernel(-half + (static_cast<double>(t) - 1.0) / static_cast<double>(kTablePoints), width_);

        /* The kernel is even and falls to exp(-2.30 width) at the edges of
         * its support, where the trapezoidal rule's error vanishes with it. */
        const size_t intervals = width_ * kQuadraturePoints;
        const double h = 1.0 / static_cast<double>(kQuadraturePoints);
        std::vector<double> z(intervals + 1);
        std::vector<double> value(intervals + 1);
        for (size_t i = 0; i <= intervals; ++i)
        {
            z[i] = -half + static_cast<double>(i) * h;
            value[i] = kernel(z[i], width_);
        }
        transform_.resize(modes_);
        for (size_t k = 0; k < modes_; ++k)
        {
            const double frequency =
                kTwoPi * (static_cast<double>(k) - static_cast<double>(centre_)) / static_cast<double>(grid_);
            double sum = 0.0;
            for (size_t i = 0; i <= intervals; ++i)
                sum += value[i] * std::cos(frequency * z[i]);
            transform_[k] = sum * h;
        }
    }

    Nufft::~Nufft() = default;

    size_t Nufft::spread(double u, double* weights, std::complex<double>& shift) const
    {
        /* Only u's fractional part matters to exp(-2 pi i k u) at integer k,
         * which puts the term within half a period of the grid's origin. */
        const double fraction = u - std::nearbyint(u);
        const double half = 0.5 * static_cast<double>(width_);
        const double x = static_cast<double>(grid_) * fraction;
        const double first = std::ceil(x - half);
        const double position = (x - first + half) * static_cast<double>(kTablePoints);
        const double below = std::floor(position);
        const double s = position - below;
        /* The kernel at the term's grid points lies at whole multiples of
         * the table's spacing apart, so one cubic's weights serve them all. */
        const double lagrange[4] = {
            -s * (s - 1.0) * (s - 2.0) / 6.0,
            (s + 1.0) * (s - 1.0) * (s - 2.0) / 2.0,
            -(s + 1.0) * s * (s - 2.0) / 2.0,
            (s + 1.0) * s * (s - 1.0) / 6.0};
        const size_t base = static_cast<size_t>(below);
        for (size_t j = 0; j < width_; ++j)
        {
            const double* around = &table_[base - j * kTablePoints];
            weights[j] = lagrange[0] * around[0] + lagrange[1] * around[1] + lagrange[2] * around[2] +
                lagrange[3] * around[3];
        }
        shift = std::polar(1.0, -kTwoPi * fraction * static_cast<double>(centre_));
        const long long period = static_cast<long long>(grid_);
        const long long start = static_cast<long long>(first) % period;
        return static_cast<size_t>(start < 0 ? start + period : start);
    }

    void Nufft::finish(std::complex<double>* grid, std::complex<double>* out) const
    {
        /* std::complex<double> and pocketfft's cmplx<double> are both two
         * doubles, real part first. */
        plan_->fft.exec(reinterpret_cast<pocketfft::detail::cmplx<double>*>(grid), 1.0, true);
        for (size_t k = 0; k < modes_; ++k)
            out[k] = grid[(k + grid_ - centre_) % grid_] / transform_[k];
    }

} // namespace bloch
