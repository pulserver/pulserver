/**
 * @file cycles.hpp
 * @brief Phases counted in cycles: the nearest whole number of cycles, and a
 *        phase's cosine and sine, free of branches and calls so that loops of
 *        them vectorise.
 */

#ifndef PULSERVER_BLOCH_CYCLES_HPP
#define PULSERVER_BLOCH_CYCLES_HPP

namespace bloch
{

    /** The whole number nearest @p v, ties to even, for |v| below 2^51
     *  under the default rounding mode and without reassociation. */
    inline double nearest(double v)
    {
        constexpr double kShift = 6755399441055744.0; /* 1.5 * 2^52 */
        return (v + kShift) - kShift;
    }

    /** cos(2 pi @p c) in @p re and sin(2 pi @p c) in @p im, to within about
     *  2e-16, for |c| below 2^51. */
    inline void cis(double c, double& re, double& im)
    {
        constexpr double kTwoPi = 6.283185307179586476925286766559;
        /* The Taylor series of sin(x) / x and cos(x) in x^2, the highest
         * power first, to the first term below the last bit at |x| = pi / 4. */
        constexpr double kSine[] = {
            -1.0 / 1307674368000.0,
            1.0 / 6227020800.0,
            -1.0 / 39916800.0,
            1.0 / 362880.0,
            -1.0 / 5040.0,
            1.0 / 120.0,
            -1.0 / 6.0,
            1.0};
        constexpr double kCosine[] = {
            1.0 / 20922789888000.0,
            -1.0 / 87178291200.0,
            1.0 / 479001600.0,
            -1.0 / 3628800.0,
            1.0 / 40320.0,
            -1.0 / 720.0,
            1.0 / 24.0,
            -0.5,
            1.0};
        /* The phase lies within an eighth of a cycle of q quarters. */
        const double r = c - nearest(c);
        const double q = nearest(4.0 * r);
        const double x = kTwoPi * (r - 0.25 * q);
        const double x2 = x * x;
        double s = 0.0;
        for (const double k : kSine)
            s = s * x2 + k;
        s *= x;
        double o = 0.0;
        for (const double k : kCosine)
            o = o * x2 + k;
        /* Turned by q quarters. */
        const bool odd = q == 1.0 || q == -1.0;
        const double a = odd ? s : o;
        const double b = odd ? o : s;
        re = q > 0.5 || q < -1.5 ? -a : a;
        im = q < -0.5 || q > 1.5 ? -b : b;
    }

} // namespace bloch

#endif /* PULSERVER_BLOCH_CYCLES_HPP */
