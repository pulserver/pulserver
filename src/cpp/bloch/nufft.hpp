/**
 * @file nufft.hpp
 * @brief Sums of complex exponentials at consecutive integer frequencies: a
 *        type-1 non-uniform FFT in one dimension.
 */

#ifndef PULSERVER_BLOCH_NUFFT_HPP
#define PULSERVER_BLOCH_NUFFT_HPP

#include <complex>
#include <cstddef>
#include <memory>
#include <vector>

namespace bloch
{

    /**
     * F(k) = sum_n c_n exp(-2 pi i k u_n), k = 0 ... modes - 1, for any real
     * u_n, to within about 1e-13 of sum_n |c_n| at the widest kernel, and to
     * within a tolerance at the kernel width_for() gives it.
     *
     * Each term is spread onto a periodic grid of twice the modes by the
     * exponential-of-semicircle kernel (Barnett, Magland and af Klinteberg,
     * SIAM J Sci Comput 41:C479, 2019), tabulated and read by cubic
     * interpolation; the grid is transformed and the kernel's transform
     * divided out. The caller spreads into grids of its own, so that series
     * whose terms share their u_n share one evaluation of the kernel per term.
     */
    class Nufft
    {
    public:
        explicit Nufft(size_t modes, size_t width = width_of());
        ~Nufft();
        Nufft(const Nufft&) = delete;
        Nufft& operator=(const Nufft&) = delete;

        /** Points of a grid of a transform of @p modes whose terms are
         *  spread onto @p width points. */
        static size_t grid_of(size_t modes, size_t width = width_of());
        /** Grid points one term is spread onto by the widest kernel. */
        static size_t width_of();
        /** Grid points a term is spread onto for the transform to hold to
         *  within @p tolerance of sum_n |c_n|; the widest kernel's below
         *  1e-12. */
        static size_t width_for(double tolerance);

        size_t modes() const
        {
            return modes_;
        }
        size_t grid() const
        {
            return grid_;
        }
        size_t width() const
        {
            return width_;
        }

        /**
         * The kernel's weights of the term at @p u, width() of them, onto the
         * consecutive grid points from the one returned on, modulo grid(), in
         * @p weights; and in @p shift the factor its coefficient is spread
         * times.
         */
        size_t spread(double u, double* weights, std::complex<double>& shift) const;

        /** Transform @p grid, grid() values spread onto, in place, and write
         *  F(0) ... F(modes() - 1) to @p out. */
        void finish(std::complex<double>* grid, std::complex<double>* out) const;

    private:
        struct Plan;
        size_t modes_;
        size_t grid_;
        size_t width_;
        /** Modes below zero the grid's transform holds F's from. */
        size_t centre_;
        /** The kernel at each 1/kTablePoints of its support, and a point
         *  more on either side for the cubic. */
        std::vector<double> table_;
        /** The kernel's transform at each mode. */
        std::vector<double> transform_;
        std::unique_ptr<Plan> plan_;
    };

} // namespace bloch

#endif /* PULSERVER_BLOCH_NUFFT_HPP */
