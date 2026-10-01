/**
 * @file tables.hpp
 * @brief A pulse's affine map read off its table at a field and a drive: the
 *        interpolation the isochromats and the repetitions share.
 */

#ifndef PULSERVER_BLOCH_TABLES_HPP
#define PULSERVER_BLOCH_TABLES_HPP

#include <complex>
#include <cstddef>

namespace bloch::table
{

    /** Rows below and above an isochromat's drive its map is interpolated
     *  from: along the drive, where the map turns as fast as it does across
     *  the field, a quintic reaches the accuracy a cubic reaches across the
     *  field, where half the pulse's precession is taken out. */
    constexpr long long kRowsBelow = 2;
    constexpr long long kRowsAbove = 3;
    constexpr size_t kRows = static_cast<size_t>(kRowsBelow + kRowsAbove + 1);

    /** The weights of the Lagrange cubic through points -1, 0, 1 and 2, at
     *  @p u in [0, 1). */
    template <typename Real>
    inline void cubic_weights(Real u, Real w[4])
    {
        w[0] = -u * (u - Real(1)) * (u - Real(2)) / Real(6);
        w[1] = (u + Real(1)) * (u - Real(1)) * (u - Real(2)) / Real(2);
        w[2] = -(u + Real(1)) * u * (u - Real(2)) / Real(2);
        w[3] = (u + Real(1)) * u * (u - Real(1)) / Real(6);
    }

    /** The cubic of weights @p w through the four consecutive maps from
     *  @p around, 12 values each. */
    template <typename Real>
    inline void cubic(const Real* around, const Real w[4], Real* map)
    {
        for (int e = 0; e < 12; ++e)
            map[e] = w[0] * around[e] + w[1] * around[12 + e] + w[2] * around[24 + e] + w[3] * around[36 + e];
    }

    /** The weights of the Lagrange polynomial through the kRows points from
     *  -kRowsBelow on, at @p v in [0, 1). */
    template <typename Real>
    inline void row_weights(Real v, Real w[kRows])
    {
        for (size_t j = 0; j < kRows; ++j)
        {
            const Real at = static_cast<Real>(static_cast<long long>(j) - kRowsBelow);
            Real weight = 1;
            for (size_t m = 0; m < kRows; ++m)
                if (m != j)
                {
                    const Real other = static_cast<Real>(static_cast<long long>(m) - kRowsBelow);
                    weight *= (v - other) / (at - other);
                }
            w[j] = weight;
        }
    }

    /** Write the affine map @p map, A row-major then c, turned about z by
     *  @p before before it and by @p after after it to @p out:
     *  A -> R(after) A R(before) and c -> R(after) c. */
    template <typename Real>
    inline void turned(const Real* map, std::complex<Real> before, std::complex<Real> after, Real* out)
    {
        const Real br = before.real();
        const Real bi = before.imag();
        const Real ar = after.real();
        const Real ai = after.imag();
        Real a[3][3];
        for (int row = 0; row < 3; ++row)
        {
            const Real* m = map + 3 * row;
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

} // namespace bloch::table

#endif /* PULSERVER_BLOCH_TABLES_HPP */
