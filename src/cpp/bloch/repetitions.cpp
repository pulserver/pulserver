/**
 * @file repetitions.cpp
 * @brief Repetitions of a sequence of blocks, played on isochromats from the
 *        affine map one repetition applies to each isochromat.  See
 *        repetitions.hpp.
 */

#include "bloch/repetitions.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "bloch/parallel.hpp"
#include "bloch/simd.hpp"

namespace bloch
{

    namespace
    {

        constexpr double kTwoPi = 6.283185307179586476925286766559;

        /** Isochromats a worker takes at the least. */
        constexpr size_t kLeast = 4096;

        /** Largest difference, in rad, between two repetitions' turns from
         *  the one before at which they are taken as one step: a sequence
         *  file keeps a phase to about 1e-5. */
        constexpr double kStepTolerance = 1e-4;

        /** Determinant of I - A below which an isochromat has no fixed point
         *  a split can rest on. */
        constexpr double kSingular = 1e-12;

        /** Repetitions an isochromat is carried through before the next: the
         *  length of the vectors a grid point accumulates. */
        constexpr size_t kTile = 16;

        /** Isochromats a worker carries through a tile at a time. */
        constexpr size_t kChunk = 128;

        /** Distinct coordinates along an axis beyond which a phase-encoding
         *  phase is computed per isochromat rather than tabulated. */
        constexpr size_t kLatticeValues = size_t(1) << 16;

        /** Tolerance from which the magnetisation is carried in single
         *  precision: a contraction's rounding stays below it. */
        constexpr double kSingleFrom = 1e-4;

        /** Share of the carried isochromats dropped at which they are
         *  compacted. */
        constexpr double kCompactAt = 0.25;

        /** Each isochromat's coordinate as an index into the distinct
         *  coordinates, unless there are more than kLatticeValues of them or
         *  fewer than four isochromats to one on average. */
        bool tabulate(
            const std::vector<double>& coordinates,
            size_t threads,
            std::vector<double>& values,
            std::vector<uint32_t>& index)
        {
            const size_t count = coordinates.size();
            std::vector<std::unordered_set<double>> seen(workers_for(count, threads, kLeast));
            std::atomic<bool> many{false};
            parallel(count, threads, kLeast, [&](size_t worker, size_t begin, size_t end) {
                std::unordered_set<double>& own = seen[worker];
                for (size_t i = begin; i < end && !many; ++i)
                {
                    own.insert(coordinates[i]);
                    if (own.size() > kLatticeValues)
                        many = true;
                }
            });
            values.clear();
            if (!many)
                for (const std::unordered_set<double>& own : seen)
                    values.insert(values.end(), own.begin(), own.end());
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
            if (many || values.size() > kLatticeValues || values.size() * 4 > count + 4)
            {
                values.clear();
                return false;
            }
            index.resize(count);
            parallel(count, threads, kLeast, [&](size_t, size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i)
                    index[i] = static_cast<uint32_t>(
                        std::lower_bound(values.begin(), values.end(), coordinates[i]) - values.begin());
            });
            return true;
        }

        /** The positions of @p key in the stable order of their keys, each
         *  below @p keys, in @p order; returns where each key's run starts
         *  in it, keys + 1 offsets. */
        std::vector<size_t> runs(const std::vector<uint32_t>& key, size_t keys, std::vector<uint32_t>& order)
        {
            std::vector<size_t> first(keys + 1, 0);
            for (const uint32_t k : key)
                ++first[k + 1];
            std::partial_sum(first.begin(), first.end(), first.begin());
            std::vector<size_t> at(first.begin(), first.end() - 1);
            order.resize(key.size());
            for (size_t n = 0; n < key.size(); ++n)
                order[at[key[n]]++] = static_cast<uint32_t>(n);
            return first;
        }

        std::vector<std::complex<double>> phasors(const std::vector<double>& phases)
        {
            std::vector<std::complex<double>> out(phases.size());
            std::transform(phases.begin(), phases.end(), out.begin(), [](double phase) { return std::polar(1.0, phase); });
            return out;
        }

        /** One array of per-isochromat data, @p width bytes per slot. */
        struct Column
        {
            unsigned char* data;
            size_t width;
        };

        /** How a slot's phase-encoding phase along an axis is found. */
        enum class Encoding
        {
            none,
            tabulated,
            computed
        };

        /**
         * The isochromats carried, one slot each, in arrays over the slots:
         * those a pass over many isochromats reads (magnetisation, map,
         * readout coefficients, phase-encoding indices) one array per
         * component, and those one isochromat's spreading reads (weights,
         * factors) together per slot.
         */
        template <typename Real>
        struct Slots
        {
            using Value = Real;
            size_t size = 0;
            size_t stride = 0;
            size_t windows = 0;
            size_t coils = 0;
            size_t taps = 0;
            /** Whether the maps' and the windows' constant terms apply: not to
             *  a transient about a fixed point. */
            bool offsets = true;
            std::vector<uint32_t> id;
            /** [component][slot]: m 3, a 9 row-major, b 3. */
            std::vector<Real> m, a, b;
            /** [window][component][slot]: u Re x, y, z then Im x, y, z; v Re,
             *  Im. */
            std::vector<Real> u, v;
            /** [window][slot]: the first grid point spread onto. */
            std::vector<uint32_t> start;
            /** [window][slot][tap]. */
            std::vector<Real> weight;
            /** [window][slot][coil][Re, Im]: receive sensitivity times the
             *  kernel's shift. */
            std::vector<Real> factor;
            /** Per axis encoded along, [slot]: the lattice index, or the
             *  coordinate in m where the axis is not tabulated. */
            std::vector<uint32_t> index[3];
            std::vector<Real> coordinate[3];
            /** [slot]: the T2 class. */
            std::vector<uint32_t> decay;
            /** [slot]: the squared magnitude below which a transient is
             *  dropped. */
            std::vector<Real> limit;

            /** Room for a slot per isochromat of @p order, the windows, coils
             *  and taps set, encoded along each axis as @p along says, and
             *  with a limit each where @p limits. */
            void allocate(const std::vector<uint32_t>& order, const Encoding along[3], bool limits)
            {
                size = stride = order.size();
                id = order;
                m.resize(3 * size);
                a.resize(9 * size);
                b.resize(offsets ? 3 * size : 0);
                u.resize(windows * 6 * size);
                v.resize(offsets ? windows * 2 * size : 0);
                start.resize(windows * size);
                weight.resize(windows * size * taps);
                factor.resize(windows * size * 2 * coils);
                for (int axis = 0; axis < 3; ++axis)
                {
                    index[axis].assign(along[axis] == Encoding::tabulated ? size : 0, 0);
                    coordinate[axis].assign(along[axis] == Encoding::computed ? size : 0, Real(0));
                }
                decay.resize(size);
                limit.resize(limits ? size : 0);
            }

            std::vector<Column> columns()
            {
                std::vector<Column> out;
                const auto add = [&](auto& vector, size_t arrays, size_t per_slot) {
                    using Item = typename std::decay_t<decltype(vector)>::value_type;
                    for (size_t k = 0; k < arrays && !vector.empty(); ++k)
                        out.push_back(
                            {reinterpret_cast<unsigned char*>(vector.data() + k * stride * per_slot),
                             per_slot * sizeof(Item)});
                };
                add(id, 1, 1);
                add(m, 3, 1);
                add(a, 9, 1);
                add(b, 3, 1);
                add(u, windows * 6, 1);
                add(v, windows * 2, 1);
                add(start, windows, 1);
                add(weight, windows, taps);
                add(factor, windows, 2 * coils);
                for (int axis = 0; axis < 3; ++axis)
                {
                    add(index[axis], 1, 1);
                    add(coordinate[axis], 1, 1);
                }
                add(decay, 1, 1);
                add(limit, 1, 1);
                return out;
            }
        };

        /** Tables held at the most: phase-encoding areas take few values over
         *  a scan, and past this many the tables are made anew. */
        constexpr size_t kTableValues = size_t(1) << 22;

        /** No table. */
        constexpr size_t kNone = ~size_t(0);

        /** The phase-encoding phase exp(-2 pi i area x) per tabulated
         *  coordinate x of an axis, real parts then imaginary, per area
         *  played along it. */
        template <typename Real>
        struct Tables
        {
            std::unordered_map<double, size_t> at[3];
            std::vector<Real> values;

            /** The offset of the table of @p area along @p axis, made where
             *  it is not held; offsets stay valid until clear(). */
            size_t find(int axis, double area, const std::vector<double>& coordinates)
            {
                const auto held = at[axis].find(area);
                if (held != at[axis].end())
                    return held->second;
                const size_t offset = values.size();
                values.resize(offset + 2 * coordinates.size());
                for (size_t k = 0; k < coordinates.size(); ++k)
                {
                    const double phase = -kTwoPi * area * coordinates[k];
                    values[offset + k] = static_cast<Real>(std::cos(phase));
                    values[offset + coordinates.size() + k] = static_cast<Real>(std::sin(phase));
                }
                at[axis].emplace(area, offset);
                return offset;
            }

            void clear()
            {
                for (auto& axis : at)
                    axis.clear();
                values.clear();
            }
        };

        /** What playing the slots through one tile of repetitions reads: the
         *  carry, and the reading of the grids it spreads onto. */
        template <typename Real>
        struct Tile
        {
            using Value = Real;
            size_t count = 0;
            size_t windows = 0;
            size_t coils = 0;
            size_t taps = 0;
            size_t classes = 0;
            /** Per repetition, the turn from its frame to the next one's. */
            Real turn_cos[kTile] = {};
            Real turn_sin[kTile] = {};
            /** Per repetition, window and axis, [(r * windows + w) * 3 +
             *  axis]: the phase-encoding phase per tabulated coordinate, or
             *  null; and -2 pi times the area where it is computed per slot. */
            std::vector<const Real*> table_re, table_im;
            std::vector<double> angle;
            std::vector<unsigned char> computed;
            /** Per window, the grid points (none for a window of one sample)
             *  and the start of its part of a worker's grid. */
            std::vector<size_t> cells, region;
            /** Whether transients at or below their limit are dropped. */
            bool drop = false;
            /** Per window: its samples, the first of them among a
             *  repetition's, its transform, and each T2's decay from its first
             *  sample to each. */
            std::vector<size_t> samples, offset;
            std::vector<const Nufft*> transform;
            std::vector<std::vector<double>> decay;
            /** Samples of a repetition per coil. */
            size_t length = 0;
            /** The workers' grids, region.back() values each, and which of
             *  them carried slots through the tile. */
            Real* grids = nullptr;
            std::vector<unsigned char> ran;
        };

        /** The coefficient at a window's first sample: u . m, plus v with
         *  the constant terms. */
        template <typename Real, bool Offsets>
        BLOCH_INLINE void coefficients(
            size_t n,
            size_t st,
            const Real* __restrict u,
            const Real* __restrict v,
            const Real* __restrict m0,
            const Real* __restrict m1,
            const Real* __restrict m2,
            Real* __restrict pr,
            Real* __restrict pi)
        {
            BLOCH_INDEPENDENT
            for (size_t k = 0; k < n; ++k)
            {
                Real re = u[k] * m0[k] + u[st + k] * m1[k] + u[2 * st + k] * m2[k];
                Real im = u[3 * st + k] * m0[k] + u[4 * st + k] * m1[k] + u[5 * st + k] * m2[k];
                if (Offsets)
                {
                    re += v[k];
                    im += v[st + k];
                }
                pr[k] = re;
                pi[k] = im;
            }
        }

        /** The coefficients turned by a tabulated phase-encoding phase. */
        template <typename Real>
        BLOCH_INLINE void encode(
            size_t n,
            const Real* __restrict tr,
            const Real* __restrict ti,
            const uint32_t* __restrict index,
            Real* __restrict pr,
            Real* __restrict pi)
        {
            BLOCH_INDEPENDENT
            for (size_t k = 0; k < n; ++k)
            {
                const Real er = tr[index[k]];
                const Real ei = ti[index[k]];
                const Real re = pr[k] * er - pi[k] * ei;
                pi[k] = pr[k] * ei + pi[k] * er;
                pr[k] = re;
            }
        }

        /** The coefficients turned by a phase-encoding phase computed per
         *  slot. */
        template <typename Real>
        BLOCH_INLINE void encode_computed(
            size_t n, double angle, const Real* __restrict coordinate, Real* __restrict pr, Real* __restrict pi)
        {
            for (size_t k = 0; k < n; ++k)
            {
                const double phase = angle * static_cast<double>(coordinate[k]);
                const Real er = static_cast<Real>(std::cos(phase));
                const Real ei = static_cast<Real>(std::sin(phase));
                const Real re = pr[k] * er - pi[k] * ei;
                pi[k] = pr[k] * ei + pi[k] * er;
                pr[k] = re;
            }
        }

        /** The magnetisation one repetition on: A m, plus b and turned
         *  with the constant terms. */
        template <typename Real, bool Offsets>
        BLOCH_INLINE void propagate(
            size_t n,
            size_t st,
            const Real* __restrict a,
            const Real* __restrict b,
            Real c,
            Real sn,
            Real* __restrict m0,
            Real* __restrict m1,
            Real* __restrict m2)
        {
            BLOCH_INDEPENDENT
            for (size_t k = 0; k < n; ++k)
            {
                const Real x = m0[k], y = m1[k], z = m2[k];
                Real nx = a[k] * x + a[st + k] * y + a[2 * st + k] * z;
                Real ny = a[3 * st + k] * x + a[4 * st + k] * y + a[5 * st + k] * z;
                Real nz = a[6 * st + k] * x + a[7 * st + k] * y + a[8 * st + k] * z;
                if (Offsets)
                {
                    const Real bx = nx + b[k];
                    const Real by = ny + b[st + k];
                    nz += b[2 * st + k];
                    nx = c * bx - sn * by;
                    ny = sn * bx + c * by;
                }
                m0[k] = nx;
                m1[k] = ny;
                m2[k] = nz;
            }
        }

#if defined(__GNUC__)
        /* A tile's repetitions of one grid point as vectors of 256 bits,
         * written out: a compiler left to itself vectorises spread_slot's
         * loop over the taps instead. */
        typedef float Floats __attribute__((vector_size(32), aligned(4), may_alias));
        typedef double Doubles __attribute__((vector_size(32), aligned(8), may_alias));

        template <typename Real>
        struct Lanes;
        template <>
        struct Lanes<float>
        {
            using Vector = Floats;
        };
        template <>
        struct Lanes<double>
        {
            using Vector = Doubles;
        };

        /** One slot's coefficients of a tile, times a coil's factor, added
         *  onto the grid points from @p row on, each by its weight. */
        template <typename Real>
        BLOCH_INLINE void spread_slot(
            const Real* er, const Real* ei, Real fr, Real fi, const Real* weight, size_t taps, Real* row)
        {
            using Vector = typename Lanes<Real>::Vector;
            constexpr size_t T = kTile;
            constexpr size_t L = sizeof(Vector) / sizeof(Real);
            Vector vr[T / L], vi[T / L];
            for (size_t h = 0; h < T / L; ++h)
            {
                const Vector a = *reinterpret_cast<const Vector*>(er + L * h);
                const Vector b = *reinterpret_cast<const Vector*>(ei + L * h);
                vr[h] = fr * a - fi * b;
                vi[h] = fr * b + fi * a;
            }
            if (taps == 0)
            {
                for (size_t h = 0; h < T / L; ++h)
                {
                    *reinterpret_cast<Vector*>(row + L * h) += vr[h];
                    *reinterpret_cast<Vector*>(row + T + L * h) += vi[h];
                }
                return;
            }
            for (size_t j = 0; j < taps; ++j)
            {
                const Real wj = weight[j];
                Real* point = row + j * 2 * T;
                for (size_t h = 0; h < T / L; ++h)
                {
                    *reinterpret_cast<Vector*>(point + L * h) += wj * vr[h];
                    *reinterpret_cast<Vector*>(point + T + L * h) += wj * vi[h];
                }
            }
        }
#else
        /** One slot's coefficients of a tile, times a coil's factor, added
         *  onto the grid points from @p row on, each by its weight. */
        template <typename Real>
        BLOCH_INLINE void spread_slot(
            const Real* __restrict er,
            const Real* __restrict ei,
            Real fr,
            Real fi,
            const Real* __restrict weight,
            size_t taps,
            Real* __restrict row)
        {
            constexpr size_t T = kTile;
            Real vr[T], vi[T];
            for (size_t r = 0; r < T; ++r)
            {
                vr[r] = fr * er[r] - fi * ei[r];
                vi[r] = fr * ei[r] + fi * er[r];
            }
            if (taps == 0)
            {
                for (size_t r = 0; r < T; ++r)
                {
                    row[r] += vr[r];
                    row[T + r] += vi[r];
                }
                return;
            }
            for (size_t j = 0; j < taps; ++j)
            {
                const Real wj = weight[j];
                Real* __restrict point = row + j * 2 * T;
                for (size_t r = 0; r < T; ++r)
                {
                    point[r] += wj * vr[r];
                    point[T + r] += wj * vi[r];
                }
            }
        }
#endif

        /** The coefficients of repetition @p r's window @p w turned by its
         *  phase-encoding phase along each axis. */
        template <typename Real>
        BLOCH_INLINE void encode_window(
            const Tile<Real>& tile, const Slots<Real>& s, size_t r, size_t w, size_t lo, size_t n, Real* pr, Real* pi)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                const size_t at = (r * tile.windows + w) * 3 + static_cast<size_t>(axis);
                if (tile.table_re[at] != nullptr)
                    encode(n, tile.table_re[at], tile.table_im[at], &s.index[axis][lo], pr, pi);
                else if (tile.computed[at])
                    encode_computed(n, tile.angle[at], &s.coordinate[axis][lo], pr, pi);
            }
        }

        /** Slots @p lo to @p lo + @p n through the tile: each repetition's
         *  coefficient at each window's first sample, [window][repetition]
         *  [slot] in @p qr and @p qi and zero past the tile's repetitions, and
         *  the magnetisation at the start of the repetition after the tile. */
        template <typename Real, bool Offsets>
        BLOCH_INLINE void chunk_coefficients(
            const Tile<Real>& tile, Slots<Real>& s, size_t lo, size_t n, Real* qr, Real* qi)
        {
            constexpr size_t T = kTile;
            const size_t st = s.stride;
            Real* m0 = &s.m[lo];
            Real* m1 = &s.m[st + lo];
            Real* m2 = &s.m[2 * st + lo];
            for (size_t r = 0; r < tile.count; ++r)
            {
                for (size_t w = 0; w < tile.windows; ++w)
                {
                    Real* pr = qr + (w * T + r) * kChunk;
                    Real* pi = qi + (w * T + r) * kChunk;
                    coefficients<Real, Offsets>(
                        n, st, &s.u[w * 6 * st + lo], Offsets ? &s.v[w * 2 * st + lo] : nullptr, m0, m1, m2, pr, pi);
                    encode_window(tile, s, r, w, lo, n, pr, pi);
                }
                propagate<Real, Offsets>(
                    n, st, &s.a[lo], Offsets ? &s.b[lo] : nullptr, tile.turn_cos[r], tile.turn_sin[r], m0, m1, m2);
            }
            for (size_t w = 0; w < tile.windows; ++w)
                for (size_t r = tile.count; r < T; ++r)
                {
                    std::fill(qr + (w * T + r) * kChunk, qr + (w * T + r) * kChunk + n, Real(0));
                    std::fill(qi + (w * T + r) * kChunk, qi + (w * T + r) * kChunk + n, Real(0));
                }
        }

        /** The coefficients of slots @p lo to @p lo + @p n, times each coil's
         *  factor, spread onto @p grid: per window, onto the grid points of
         *  the slot's T2 and coil, or summed without a kernel for a window of
         *  one sample. */
        template <typename Real>
        BLOCH_INLINE void spread_chunk(
            const Tile<Real>& tile, const Slots<Real>& s, size_t lo, size_t n, const Real* qr, const Real* qi, Real* grid)
        {
            constexpr size_t T = kTile;
            const size_t C = tile.coils;
            const size_t taps = tile.taps;
            const size_t st = s.stride;
            for (size_t k = 0; k < n; ++k)
            {
                const size_t slot = lo + k;
                for (size_t w = 0; w < tile.windows; ++w)
                {
                    Real er[T], ei[T];
                    for (size_t r = 0; r < T; ++r)
                    {
                        er[r] = qr[(w * T + r) * kChunk + k];
                        ei[r] = qi[(w * T + r) * kChunk + k];
                    }
                    const size_t cells = tile.cells[w];
                    const size_t first = cells == 0 ? 0 : (s.decay[slot] * C * (cells + taps) + s.start[w * st + slot]) * 2 * T;
                    const size_t per_coil = cells == 0 ? 2 * T : (cells + taps) * 2 * T;
                    const Real* factor = s.factor.data() + (w * st + slot) * 2 * C;
                    const Real* weight = s.weight.data() + (w * st + slot) * taps;
                    Real* row = grid + tile.region[w] + first;
                    for (size_t coil = 0; coil < C; ++coil)
                        spread_slot(
                            er, ei, factor[2 * coil], factor[2 * coil + 1], weight, cells == 0 ? 0 : taps, row + coil * per_coil);
                }
            }
        }

        /** Carry slots @p begin to @p end through the tile: each repetition's
         *  coefficient at each window's first sample spread onto @p grid, and
         *  the magnetisation at the start of the repetition after the tile. */
        template <typename Real, bool Offsets>
        BLOCH_INLINE void carry(const Tile<Real>& tile, Slots<Real>& s, size_t begin, size_t end, Real* grid, Real* scratch)
        {
            /* [window][repetition][slot of the chunk] */
            Real* qr = scratch;
            Real* qi = scratch + tile.windows * kTile * kChunk;
            for (size_t lo = begin; lo < end; lo += kChunk)
            {
                const size_t n = std::min(kChunk, end - lo);
                chunk_coefficients<Real, Offsets>(tile, s, lo, n, qr, qi);
                spread_chunk(tile, s, lo, n, qr, qi, grid);
            }
        }

        template <typename Real>
        using CarryRange = void (*)(const Tile<Real>&, Slots<Real>&, size_t, size_t, Real*, Real*);

        template <typename Real, bool Offsets>
        void carry_plain(const Tile<Real>& tile, Slots<Real>& s, size_t begin, size_t end, Real* grid, Real* scratch)
        {
            carry<Real, Offsets>(tile, s, begin, end, grid, scratch);
        }

#ifdef BLOCH_X86_64
        template <typename Real, bool Offsets>
        BLOCH_AVX2 void carry_avx2(
            const Tile<Real>& tile, Slots<Real>& s, size_t begin, size_t end, Real* grid, Real* scratch)
        {
            carry<Real, Offsets>(tile, s, begin, end, grid, scratch);
        }
#endif

        /** The carry this processor runs fastest. */
        template <typename Real>
        CarryRange<Real> fastest_carry(bool offsets)
        {
#ifdef BLOCH_X86_64
            static const bool wide = avx2_and_fma();
            if (wide)
                return offsets ? carry_avx2<Real, true> : carry_avx2<Real, false>;
#endif
            return offsets ? carry_plain<Real, true> : carry_plain<Real, false>;
        }

        /** Zero the transients of slots @p begin to @p end at or below their
         *  limit; return how many were not zero. */
        template <typename Real>
        size_t drop(Slots<Real>& s, size_t begin, size_t end)
        {
            Real* m0 = &s.m[0];
            Real* m1 = &s.m[s.stride];
            Real* m2 = &s.m[2 * s.stride];
            size_t dropped = 0;
            for (size_t i = begin; i < end; ++i)
            {
                const Real size = m0[i] * m0[i] + m1[i] * m1[i] + m2[i] * m2[i];
                if (size > Real(0) && !(size > s.limit[i]))
                {
                    m0[i] = m1[i] = m2[i] = Real(0);
                    ++dropped;
                }
            }
            return dropped;
        }

        /** Carry every slot through the tile, each worker onto its own grid,
         *  and drop the transients at or below their limit where the tile
         *  drops them; return how many were dropped. */
        template <typename Real>
        size_t carry_tile(Tile<Real>& tile, Slots<Real>& slots, CarryRange<Real> range, Real* scratch, size_t threads)
        {
            const size_t grid_size = tile.region.back();
            const size_t scratch_size = tile.windows * kTile * kChunk * 2;
            tile.ran.assign(workers_for(slots.size, threads, kLeast), 0);
            std::atomic<size_t> dropped{0};
            parallel(slots.size, threads, kLeast, [&](size_t worker, size_t begin, size_t end) {
                Real* grid = tile.grids + worker * grid_size;
                std::fill(grid, grid + grid_size, Real(0));
                range(tile, slots, begin, end, grid, scratch + worker * scratch_size);
                if (tile.drop)
                    dropped += drop(slots, begin, end);
                tile.ran[worker] = 1;
            });
            return dropped.load();
        }

        /** Move the slots of @p begin to @p end whose transient is above
         *  their limit to the front of the range, in their order, in every
         *  column; return how many. */
        template <typename Real>
        size_t compact_range(Slots<Real>& s, const std::vector<Column>& columns, size_t begin, size_t end)
        {
            const Real* m0 = &s.m[0];
            const Real* m1 = &s.m[s.stride];
            const Real* m2 = &s.m[2 * s.stride];
            std::vector<unsigned char> keep(end - begin);
            for (size_t i = begin; i < end; ++i)
                keep[i - begin] = m0[i] * m0[i] + m1[i] * m1[i] + m2[i] * m2[i] > s.limit[i];
            size_t to = begin;
            for (const Column& column : columns)
            {
                to = begin;
                for (size_t i = begin; i < end; ++i)
                    if (keep[i - begin])
                    {
                        if (to != i)
                            std::memcpy(column.data + to * column.width, column.data + i * column.width, column.width);
                        ++to;
                    }
            }
            return to - begin;
        }

        /** Move the @p kept slots of each part, from @p first on, to follow
         *  the parts before it, in every column. */
        void join_parts(
            const std::vector<Column>& columns,
            const std::vector<size_t>& first,
            const std::vector<size_t>& kept,
            size_t threads)
        {
            parallel(columns.size(), threads, 1, [&](size_t, size_t begin, size_t end) {
                for (size_t c = begin; c < end; ++c)
                {
                    size_t to = 0;
                    for (size_t part = 0; part < first.size(); ++part)
                    {
                        if (to != first[part])
                            std::memmove(
                                columns[c].data + to * columns[c].width,
                                columns[c].data + first[part] * columns[c].width,
                                kept[part] * columns[c].width);
                        to += kept[part];
                    }
                }
            });
        }

        /** Keep only the slots whose transient is above its limit, in their
         *  order. */
        template <typename Real>
        void compact(Slots<Real>& s, size_t threads)
        {
            const std::vector<Column> columns = s.columns();
            const size_t workers = workers_for(s.size, threads, kLeast);
            std::vector<size_t> first(workers, 0), kept(workers, 0);
            parallel(s.size, threads, kLeast, [&](size_t worker, size_t begin, size_t end) {
                first[worker] = begin;
                kept[worker] = compact_range(s, columns, begin, end);
            });
            join_parts(columns, first, kept, threads);
            s.size = std::accumulate(kept.begin(), kept.end(), size_t(0));
        }

        /** Window @p w's sample, a window of one, per repetition of the tile
         *  and coil: the workers' sums. */
        template <typename Real>
        void read_single(const Tile<Real>& tile, size_t w, std::complex<double>* out)
        {
            constexpr size_t T = kTile;
            const size_t grid_size = tile.region.back();
            for (size_t r = 0; r < tile.count; ++r)
                for (size_t k = 0; k < tile.coils; ++k)
                {
                    std::complex<double> sum(0.0, 0.0);
                    for (size_t worker = 0; worker < tile.ran.size(); ++worker)
                        if (tile.ran[worker])
                        {
                            const Real* row = tile.grids + worker * grid_size + tile.region[w] + k * 2 * T;
                            sum += std::complex<double>(row[r], row[T + r]);
                        }
                    out[(r * tile.coils + k) * tile.length + tile.offset[w]] = sum;
                }
        }

        /** The workers' grids of window @p w, T2 class @p d and coil @p k,
         *  summed and folded onto the window's grid points, per repetition of
         *  the tile, into @p columns; false where no worker spread onto
         *  them. */
        template <typename Real>
        bool fold_grids(const Tile<Real>& tile, size_t w, size_t d, size_t k, std::vector<std::complex<double>>& columns)
        {
            constexpr size_t T = kTile;
            const size_t cells = tile.cells[w];
            const size_t grid_size = tile.region.back();
            std::fill(columns.begin(), columns.end(), std::complex<double>(0.0, 0.0));
            bool any = false;
            for (size_t worker = 0; worker < tile.ran.size(); ++worker)
            {
                if (!tile.ran[worker])
                    continue;
                any = true;
                const Real* row =
                    tile.grids + worker * grid_size + tile.region[w] + (d * tile.coils + k) * (cells + tile.taps) * 2 * T;
                for (size_t g = 0; g < cells + tile.taps; ++g)
                {
                    const size_t at = g < cells ? g : g - cells;
                    const Real* point = row + g * 2 * T;
                    for (size_t r = 0; r < tile.count; ++r)
                        columns[r * cells + at] += std::complex<double>(point[r], point[T + r]);
                }
            }
            return any;
        }

        /** Each repetition's folded grid of window @p w transformed as the
         *  window reads it, times the decay of T2 class @p d, into @p into,
         *  [repetition][sample]. */
        template <typename Real>
        void finish_grids(
            const Tile<Real>& tile, size_t w, size_t d, std::vector<std::complex<double>>& columns, std::complex<double>* into)
        {
            const size_t samples = tile.samples[w];
            const double* decay = &tile.decay[w][d * samples];
            for (size_t r = 0; r < tile.count; ++r)
            {
                std::complex<double>* sample = into + r * samples;
                tile.transform[w]->finish(&columns[r * tile.cells[w]], sample);
                for (size_t j = 0; j < samples; ++j)
                    sample[j] *= decay[j];
            }
        }

        /** Window @p w's samples of repetition @p r and coil @p k: the sum
         *  over the T2 classes of @p partial, [class][coil][repetition]
         *  [sample]. */
        template <typename Real>
        void sum_classes(
            const Tile<Real>& tile,
            size_t w,
            size_t r,
            size_t k,
            const std::vector<std::complex<double>>& partial,
            std::complex<double>* out)
        {
            const size_t samples = tile.samples[w];
            std::complex<double>* into = out + (r * tile.coils + k) * tile.length + tile.offset[w];
            for (size_t j = 0; j < samples; ++j)
            {
                std::complex<double> sum(0.0, 0.0);
                for (size_t d = 0; d < tile.classes; ++d)
                    sum += partial[((d * tile.coils + k) * kTile + r) * samples + j];
                into[j] = sum;
            }
        }

        /** Window @p w's samples per repetition of the tile and coil: per T2
         *  class and coil, the workers' grids summed, folded and transformed,
         *  times the class's decay; then summed over the classes. */
        template <typename Real>
        void read_transformed(
            const Tile<Real>& tile,
            size_t w,
            size_t threads,
            std::vector<std::complex<double>>& partial,
            std::complex<double>* out)
        {
            const size_t samples = tile.samples[w];
            const size_t coils = tile.coils;
            partial.assign(tile.classes * coils * kTile * samples, std::complex<double>(0.0, 0.0));
            parallel(tile.classes * coils, threads, 1, [&](size_t, size_t from, size_t to) {
                std::vector<std::complex<double>> columns(kTile * tile.cells[w]);
                for (size_t item = from; item < to; ++item)
                    if (fold_grids(tile, w, item / coils, item % coils, columns))
                        finish_grids(tile, w, item / coils, columns, &partial[item * kTile * samples]);
            });
            parallel(tile.count * coils, threads, 1, [&](size_t, size_t from, size_t to) {
                for (size_t item = from; item < to; ++item)
                    sum_classes(tile, w, item / coils, item % coils, partial, out);
            });
        }

        /** What summing the fixed points' samples over columns of
         *  isochromats reads and writes. */
        struct ColumnSums
        {
            size_t count = 0;
            size_t coils = 0;
            size_t samples = 0;
            size_t classes = 0;
            const size_t* first = nullptr;
            const uint32_t* members = nullptr;
            const std::complex<double>* u = nullptr;
            const std::complex<double>* v = nullptr;
            const double* fixed = nullptr;
            /** Null without receive sensitivities. */
            const double* receive_re = nullptr;
            const double* receive_im = nullptr;
            const double* x = nullptr;
            const double* y = nullptr;
            const double* z = nullptr;
            const double* off_resonance = nullptr;
            const uint32_t* decay_of = nullptr;
            /** Per T2, its decay from the first sample to each. */
            std::vector<double> decay;
            /** The window's transform; null for a window of one sample. */
            const Nufft* transform = nullptr;
            double area[3] = {0.0, 0.0, 0.0};
            double step = 0.0;
            double encoding[3] = {0.0, 0.0, 0.0};
            std::complex<double>* out = nullptr;
        };

        /** Isochromat @p i's sample per coil at the window's first sample in
         *  its fixed point, times its phase-encoding phase. */
        BLOCH_INLINE void steady_weights(const ColumnSums& c, size_t i, std::complex<double>* weighed)
        {
            const std::complex<double>* u = &c.u[3 * i];
            const std::complex<double> steady =
                (u[0] * c.fixed[3 * i] + u[1] * c.fixed[3 * i + 1] + u[2] * c.fixed[3 * i + 2] + c.v[i]) *
                std::polar(1.0, -kTwoPi * (c.encoding[0] * c.x[i] + c.encoding[1] * c.y[i] + c.encoding[2] * c.z[i]));
            for (size_t k = 0; k < c.coils; ++k)
                weighed[k] = c.receive_re != nullptr
                    ? steady * std::complex<double>(c.receive_re[k * c.count + i], c.receive_im[k * c.count + i])
                    : steady;
        }

        /** Isochromat @p i's samples per coil spread onto its column's grids
         *  of its T2 class; return the class. */
        BLOCH_INLINE size_t spread_member(
            const ColumnSums& c, size_t i, const std::complex<double>* weighed, double* weights, std::complex<double>* grids)
        {
            const Nufft& transform = *c.transform;
            const size_t span = transform.grid() + transform.width();
            std::complex<double> shift;
            const size_t start = transform.spread(
                c.area[0] * c.x[i] + c.area[1] * c.y[i] + c.area[2] * c.z[i] + c.off_resonance[i] * c.step, weights, shift);
            const size_t d = c.decay_of[i];
            for (size_t k = 0; k < c.coils; ++k)
            {
                const std::complex<double> term = weighed[k] * shift;
                std::complex<double>* row = &grids[(d * c.coils + k) * span + start];
                for (size_t j = 0; j < transform.width(); ++j)
                    row[j] += weights[j] * term;
            }
            return d;
        }

        /** A column's grids of each T2 class spread onto, folded and
         *  transformed as the window reads them, times the class's decay,
         *  summed into @p out. */
        BLOCH_INLINE void finish_column(
            const ColumnSums& c,
            const std::complex<double>* grids,
            const std::vector<unsigned char>& used,
            std::complex<double>* folded,
            std::complex<double>* modes,
            std::complex<double>* out)
        {
            const Nufft& transform = *c.transform;
            const size_t cells = transform.grid();
            const size_t taps = transform.width();
            std::fill(out, out + c.coils * c.samples, std::complex<double>(0.0, 0.0));
            for (size_t d = 0; d < c.classes; ++d)
            {
                if (!used[d])
                    continue;
                for (size_t k = 0; k < c.coils; ++k)
                {
                    const std::complex<double>* row = &grids[(d * c.coils + k) * (cells + taps)];
                    std::copy(row, row + cells, folded);
                    for (size_t g = 0; g < taps; ++g)
                        folded[g] += row[cells + g];
                    transform.finish(folded, modes);
                    for (size_t j = 0; j < c.samples; ++j)
                        out[k * c.samples + j] += c.decay[d * c.samples + j] * modes[j];
                }
            }
        }

        /** Columns @p begin to @p end: each member's samples per coil at the
         *  first sample, spread onto its column's grid of its T2 and
         *  transformed as the window reads it, then each T2's decay. */
        BLOCH_INLINE void sum_columns(const ColumnSums& c, size_t begin, size_t end)
        {
            const Nufft* transform = c.transform;
            const size_t cells = transform != nullptr ? transform->grid() : 0;
            const size_t span = transform != nullptr ? cells + transform->width() : 0;
            std::vector<std::complex<double>> grids(transform != nullptr ? c.classes * c.coils * span : c.coils);
            std::vector<unsigned char> used(c.classes);
            std::vector<double> weights(Nufft::width_of());
            std::vector<std::complex<double>> folded(cells), modes(c.samples), weighed(c.coils);
            for (size_t column = begin; column < end; ++column)
            {
                std::fill(grids.begin(), grids.end(), std::complex<double>(0.0, 0.0));
                std::fill(used.begin(), used.end(), 0);
                for (size_t at = c.first[column]; at < c.first[column + 1]; ++at)
                {
                    steady_weights(c, c.members[at], weighed.data());
                    if (transform != nullptr)
                        used[spread_member(c, c.members[at], weighed.data(), weights.data(), grids.data())] = 1;
                    else
                        std::transform(weighed.begin(), weighed.end(), grids.begin(), grids.begin(), std::plus<>());
                }
                std::complex<double>* out = c.out + column * c.coils * c.samples;
                if (transform != nullptr)
                    finish_column(c, grids.data(), used, folded.data(), modes.data(), out);
                else
                    std::copy(grids.begin(), grids.end(), out);
            }
        }

        using ColumnRange = void (*)(const ColumnSums&, size_t, size_t);

        void sum_columns_plain(const ColumnSums& c, size_t begin, size_t end)
        {
            sum_columns(c, begin, end);
        }

#ifdef BLOCH_X86_64
        BLOCH_AVX2 void sum_columns_avx2(const ColumnSums& c, size_t begin, size_t end)
        {
            sum_columns(c, begin, end);
        }
#endif

        /** The column sum this processor runs fastest. */
        ColumnRange fastest_columns()
        {
#ifdef BLOCH_X86_64
            if (avx2_and_fma())
                return sum_columns_avx2;
#endif
            return sum_columns_plain;
        }

        /** The map M -> R (A M + b), R = Rz(-step) given by its cosine @p c
         *  and sine @p sn, as @p turned, and its fixed point (I - R A)^-1 R b
         *  as @p fixed; false where I - R A is singular. */
        bool fixed_point(const double* a, const double* b, double c, double sn, double* turned, double* fixed)
        {
            double* t = turned;
            for (int col = 0; col < 3; ++col)
            {
                t[col] = c * a[col] - sn * a[3 + col];
                t[3 + col] = sn * a[col] + c * a[3 + col];
                t[6 + col] = a[6 + col];
            }
            const double rb[3] = {c * b[0] - sn * b[1], sn * b[0] + c * b[1], b[2]};
            const double m[9] = {1.0 - t[0], -t[1], -t[2], -t[3], 1.0 - t[4], -t[5], -t[6], -t[7], 1.0 - t[8]};
            const double co[9] = {
                m[4] * m[8] - m[5] * m[7], m[2] * m[7] - m[1] * m[8], m[1] * m[5] - m[2] * m[4],
                m[5] * m[6] - m[3] * m[8], m[0] * m[8] - m[2] * m[6], m[2] * m[3] - m[0] * m[5],
                m[3] * m[7] - m[4] * m[6], m[1] * m[6] - m[0] * m[7], m[0] * m[4] - m[1] * m[3]};
            const double det = m[0] * co[0] + m[1] * co[3] + m[2] * co[6];
            if (!(std::fabs(det) > kSingular) || !std::isfinite(det))
                return false;
            for (int row = 0; row < 3; ++row)
                fixed[row] = (co[3 * row] * rb[0] + co[3 * row + 1] * rb[1] + co[3 * row + 2] * rb[2]) / det;
            return true;
        }

    } // namespace

    struct Repetitions::Set
    {
        bool single = false;
        Slots<float> single_slots;
        Slots<double> double_slots;
        /** Per worker, its grid; and its chunk's coefficients. */
        std::vector<float> single_grids, single_scratch;
        std::vector<double> double_grids, double_scratch;
        Tables<float> single_tables;
        Tables<double> double_tables;
        /** Slots dropped since the last compaction. */
        size_t dropped = 0;

        template <typename Real>
        Slots<Real>& slots();
        template <typename Real>
        Tables<Real>& tables();
        template <typename Real>
        std::vector<Real>& grids();
        template <typename Real>
        std::vector<Real>& scratch();
    };

    template <>
    Slots<float>& Repetitions::Set::slots<float>()
    {
        return single_slots;
    }
    template <>
    Slots<double>& Repetitions::Set::slots<double>()
    {
        return double_slots;
    }
    template <>
    Tables<float>& Repetitions::Set::tables<float>()
    {
        return single_tables;
    }
    template <>
    Tables<double>& Repetitions::Set::tables<double>()
    {
        return double_tables;
    }
    template <>
    std::vector<float>& Repetitions::Set::grids<float>()
    {
        return single_grids;
    }
    template <>
    std::vector<double>& Repetitions::Set::grids<double>()
    {
        return double_grids;
    }
    template <>
    std::vector<float>& Repetitions::Set::scratch<float>()
    {
        return single_scratch;
    }
    template <>
    std::vector<double>& Repetitions::Set::scratch<double>()
    {
        return double_scratch;
    }

    BlockEvents OwnedBlock::events() const
    {
        BlockEvents block;
        block.duration = duration;
        for (int axis = 0; axis < 3; ++axis)
        {
            block.gradient_times[axis] = gradient_times[axis].data();
            block.gradient_values[axis] = gradient_values[axis].data();
            block.gradient_corners[axis] = gradient_times[axis].size();
        }
        block.rf_start = rf_start;
        block.rf_step = rf_step;
        block.rf_steps = rf_steps;
        block.rf_channels = rf_channels;
        block.rf = rf.empty() ? nullptr : rf.data();
        block.adc_times = adc_times.data();
        block.adc_samples = adc_times.size();
        return block;
    }

    Repetitions::Repetitions(
        Isochromats& isochromats,
        std::vector<OwnedBlock> blocks,
        std::vector<double> phases,
        std::vector<double> adc_phases,
        std::vector<double> areas,
        double tolerance)
        : isochromats_(isochromats),
          blocks_(std::move(blocks)),
          phases_(std::move(phases)),
          adc_phases_(std::move(adc_phases)),
          areas_(std::move(areas)),
          tolerance_(tolerance)
    {
        if (!(tolerance >= 0.0) || !std::isfinite(tolerance))
            throw std::invalid_argument("the tolerance must be finite and not negative");
        read_windows();
        if (adc_phases_.size() != phases_.size())
            throw std::invalid_argument("the repetitions need one ADC phase per RF phase");
        if (areas_.size() != phases_.size() * windows_.size() * 3)
            throw std::invalid_argument(
                "the repetitions need three phase-encoding areas per ADC window, " +
                std::to_string(windows_.size()) + " per repetition");
        const size_t width = Nufft::width_for(tolerance_);
        for (const Window& window : windows_)
        {
            transforms_.push_back(window.samples > 1 ? std::make_unique<Nufft>(window.samples, width) : nullptr);
            receivers_.push_back(phasors(window.receiver));
        }
        if (phases_.empty())
            return;

        Isochromats& s = isochromats_;
        const std::lock_guard<std::mutex> held(s.mutex_);
        s.flush();
        const std::vector<double> x0 = s.mx_;
        const std::vector<double> y0 = s.my_;
        const std::vector<double> z0 = s.mz_;
        const double elapsed = s.elapsed_;
        play_maps();
        s.mx_ = x0;
        s.my_ = y0;
        s.mz_ = z0;
        std::fill(s.pending_area_, s.pending_area_ + 3, 0.0);
        s.pending_time_ = 0.0;
        s.elapsed_ = elapsed;
        tabulate_encoded();

        /* The magnetisation in the frame of the first repetition's pulses. */
        const double c = std::cos(turn(0));
        const double sn = std::sin(turn(0));
        m_.resize(3 * x0.size());
        for (size_t i = 0; i < x0.size(); ++i)
        {
            m_[3 * i] = c * x0[i] + sn * y0[i];
            m_[3 * i + 1] = -sn * x0[i] + c * y0[i];
            m_[3 * i + 2] = z0[i];
        }
    }

    Repetitions::~Repetitions() = default;

    void Repetitions::read_windows()
    {
        for (size_t b = 0; b < blocks_.size(); ++b)
        {
            const OwnedBlock& block = blocks_[b];
            duration_ += block.duration;
            if (block.adc_times.empty())
                continue;
            if (block.receiver.size() != block.adc_times.size())
                throw std::invalid_argument("a repeated block's ADC needs one receiver phase per sample");
            Window window;
            window.block = b;
            window.samples = block.adc_times.size();
            window.offset = samples_;
            window.receiver = block.receiver;
            if (!isochromats_.window_steps(block.events(), window.area, window.step))
                throw std::invalid_argument(
                    "a repeated block's ADC window must be read under a gradient held throughout it, at equal steps");
            samples_ += window.samples;
            windows_.push_back(std::move(window));
        }
    }

    void Repetitions::play_maps()
    {
        const size_t total = isochromats_.count_;
        a_.assign(9 * total, 0.0);
        b_.assign(3 * total, 0.0);
        for (Window& window : windows_)
        {
            window.u.assign(3 * total, 0.0);
            window.v.assign(total, 0.0);
        }
        std::vector<std::vector<std::complex<double>>> first(windows_.size(), std::vector<std::complex<double>>(total));
        for (int column = -1; column < 3; ++column)
        {
            play_from(column, first);
            store_map(column, first);
        }
    }

    void Repetitions::play_from(int column, std::vector<std::vector<std::complex<double>>>& first)
    {
        Isochromats& s = isochromats_;
        std::fill(s.mx_.begin(), s.mx_.end(), column == 0 ? 1.0 : 0.0);
        std::fill(s.my_.begin(), s.my_.end(), column == 1 ? 1.0 : 0.0);
        std::fill(s.mz_.begin(), s.mz_.end(), column == 2 ? 1.0 : 0.0);
        std::fill(s.pending_area_, s.pending_area_ + 3, 0.0);
        s.pending_time_ = 0.0;
        size_t w = 0;
        for (size_t b = 0; b < blocks_.size(); ++b)
        {
            const bool read = w < windows_.size() && windows_[w].block == b;
            s.play_quietly(blocks_[b].events(), read ? first[w].data() : nullptr);
            w += read ? 1 : 0;
        }
        s.flush();
    }

    void Repetitions::store_map(int column, const std::vector<std::vector<std::complex<double>>>& first)
    {
        const Isochromats& s = isochromats_;
        parallel(s.count_, s.threads_, kLeast, [&](size_t, size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
            {
                const double last[3] = {s.mx_[i], s.my_[i], s.mz_[i]};
                for (int row = 0; row < 3; ++row)
                {
                    if (column < 0)
                        b_[3 * i + row] = last[row];
                    else
                        a_[9 * i + 3 * row + column] = last[row] - b_[3 * i + row];
                }
                for (size_t w = 0; w < windows_.size(); ++w)
                {
                    if (column < 0)
                        windows_[w].v[i] = first[w][i];
                    else
                        windows_[w].u[3 * i + column] = first[w][i] - windows_[w].v[i];
                }
            }
        });
    }

    void Repetitions::tabulate_encoded()
    {
        const Isochromats& s = isochromats_;
        const std::vector<double>* coordinates[3] = {&s.properties_.x, &s.properties_.y, &s.properties_.z};
        for (int axis = 0; axis < 3; ++axis)
        {
            for (size_t k = static_cast<size_t>(axis); k < areas_.size() && !encoded_[axis]; k += 3)
                encoded_[axis] = areas_[k] != 0.0;
            if (encoded_[axis])
                lattice_[axis].tabulated =
                    tabulate(*coordinates[axis], s.threads_, lattice_[axis].values, lattice_[axis].index);
        }
    }

    double Repetitions::window_phase(const Window& window, size_t i) const
    {
        const IsochromatProperties& p = isochromats_.properties_;
        return p.x[i] * window.area[0] + p.y[i] * window.area[1] + p.z[i] * window.area[2] +
            p.off_resonance[i] * window.step;
    }

    bool Repetitions::transient_above(size_t i) const
    {
        const double* d = &m_[3 * i];
        const double limit = tolerance_ * isochromats_.properties_.proton_density[i];
        return d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > limit * limit;
    }

    std::vector<uint32_t> Repetitions::carried_order() const
    {
        const Isochromats& s = isochromats_;
        std::vector<uint32_t> chosen;
        chosen.reserve(s.count_);
        for (size_t i = 0; i < s.count_; ++i)
            if (!divided_ || transient_above(i))
                chosen.push_back(static_cast<uint32_t>(i));
        const Nufft* transform = windows_.empty() ? nullptr : transforms_[0].get();
        const size_t cells = transform != nullptr ? transform->grid() : 1;
        std::vector<uint32_t> key(chosen.size());
        parallel(chosen.size(), s.threads_, kLeast, [&](size_t, size_t begin, size_t end) {
            std::vector<double> weights(Nufft::width_of());
            std::complex<double> shift;
            for (size_t n = begin; n < end; ++n)
            {
                const size_t i = chosen[n];
                const size_t start =
                    transform != nullptr ? transform->spread(window_phase(windows_[0], i), weights.data(), shift) : 0;
                key[n] = static_cast<uint32_t>(s.decay_of_[i] * cells + start);
            }
        });
        std::vector<uint32_t> order;
        runs(key, s.decays_.size() * cells, order);
        std::transform(order.begin(), order.end(), order.begin(), [&](uint32_t n) { return chosen[n]; });
        return order;
    }

    template <typename Real>
    void Repetitions::gather()
    {
        const Isochromats& s = isochromats_;
        const std::vector<uint32_t> order = carried_order();
        const bool transformed = std::any_of(
            transforms_.begin(), transforms_.end(), [](const std::unique_ptr<Nufft>& transform) { return transform != nullptr; });
        Slots<Real>& slots = set_->template slots<Real>();
        slots.windows = windows_.size();
        slots.coils = s.coils_;
        slots.taps = transformed ? Nufft::width_for(tolerance_) : 0;
        slots.offsets = !divided_;
        Encoding along[3];
        for (int axis = 0; axis < 3; ++axis)
            along[axis] = !encoded_[axis] ? Encoding::none
                : lattice_[axis].tabulated ? Encoding::tabulated
                                           : Encoding::computed;
        slots.allocate(order, along, divided_ && tolerance_ > 0.0);
        parallel(order.size(), s.threads_, kLeast, [&](size_t, size_t begin, size_t end) {
            std::vector<double> weights(Nufft::width_of());
            for (size_t n = begin; n < end; ++n)
                fill_slot(slots, n, order[n], weights.data());
        });
        release_maps();
    }

    template <typename SlotsOf>
    void Repetitions::fill_slot(SlotsOf& slots, size_t n, size_t i, double* weights) const
    {
        using Real = typename SlotsOf::Value;
        const Isochromats& s = isochromats_;
        const IsochromatProperties& p = s.properties_;
        const size_t size = slots.stride;
        for (int k = 0; k < 3; ++k)
            slots.m[k * size + n] = static_cast<Real>(m_[3 * i + k]);
        for (int k = 0; k < 9; ++k)
            slots.a[k * size + n] = static_cast<Real>(a_[9 * i + k]);
        for (int k = 0; k < 3 && slots.offsets; ++k)
            slots.b[k * size + n] = static_cast<Real>(b_[3 * i + k]);
        for (size_t w = 0; w < windows_.size(); ++w)
            fill_window(slots, w, n, i, weights);
        const std::vector<double>* coordinates[3] = {&p.x, &p.y, &p.z};
        for (int axis = 0; axis < 3; ++axis)
        {
            if (!slots.index[axis].empty())
                slots.index[axis][n] = lattice_[axis].index[i];
            else if (!slots.coordinate[axis].empty())
                slots.coordinate[axis][n] = static_cast<Real>((*coordinates[axis])[i]);
        }
        slots.decay[n] = s.decay_of_[i];
        const double limit = tolerance_ * p.proton_density[i];
        if (!slots.limit.empty())
            slots.limit[n] = static_cast<Real>(limit * limit);
    }

    template <typename SlotsOf>
    void Repetitions::fill_window(SlotsOf& slots, size_t w, size_t n, size_t i, double* weights) const
    {
        using Real = typename SlotsOf::Value;
        const Isochromats& s = isochromats_;
        const Window& window = windows_[w];
        const size_t size = slots.stride;
        for (int k = 0; k < 3; ++k)
        {
            slots.u[(w * 6 + k) * size + n] = static_cast<Real>(window.u[3 * i + k].real());
            slots.u[(w * 6 + 3 + k) * size + n] = static_cast<Real>(window.u[3 * i + k].imag());
        }
        if (slots.offsets)
        {
            slots.v[(w * 2) * size + n] = static_cast<Real>(window.v[i].real());
            slots.v[(w * 2 + 1) * size + n] = static_cast<Real>(window.v[i].imag());
        }
        std::complex<double> shift(1.0, 0.0);
        if (transforms_[w] != nullptr)
        {
            slots.start[w * size + n] =
                static_cast<uint32_t>(transforms_[w]->spread(window_phase(window, i), weights, shift));
            for (size_t j = 0; j < slots.taps; ++j)
                slots.weight[(w * size + n) * slots.taps + j] = static_cast<Real>(weights[j]);
        }
        const bool sensitivities = !s.receive_re_.empty();
        for (size_t k = 0; k < slots.coils; ++k)
        {
            const std::complex<double> receive = sensitivities
                ? std::complex<double>(s.receive_re_[k * s.count_ + i], s.receive_im_[k * s.count_ + i])
                : std::complex<double>(1.0, 0.0);
            const std::complex<double> factor = receive * shift;
            slots.factor[((w * size + n) * slots.coils + k) * 2] = static_cast<Real>(factor.real());
            slots.factor[((w * size + n) * slots.coils + k) * 2 + 1] = static_cast<Real>(factor.imag());
        }
    }

    void Repetitions::release_maps()
    {
        std::vector<double>().swap(a_);
        std::vector<double>().swap(b_);
        std::vector<double>().swap(m_);
        for (Window& window : windows_)
        {
            std::vector<std::complex<double>>().swap(window.u);
            std::vector<std::complex<double>>().swap(window.v);
        }
        for (Lattice& along : lattice_)
            std::vector<uint32_t>().swap(along.index);
    }

    std::vector<double> Repetitions::decays(const Window& window) const
    {
        const std::vector<double>& rates = isochromats_.decays_;
        std::vector<double> out(rates.size() * window.samples);
        for (size_t d = 0; d < rates.size(); ++d)
            for (size_t j = 0; j < window.samples; ++j)
                out[d * window.samples + j] = std::exp(-rates[d] * window.step * static_cast<double>(j));
        return out;
    }

    void Repetitions::play(size_t count, std::complex<double>* signal)
    {
        if (count > phases_.size() - next_)
            throw std::invalid_argument(
                "only " + std::to_string(phases_.size() - next_) + " repetitions remain to be played");
        if (count == 0)
            return;
        Isochromats& s = isochromats_;
        const std::lock_guard<std::mutex> held(s.mutex_);
        const bool single = tolerance_ >= kSingleFrom;
        if (!set_)
        {
            set_ = std::make_unique<Set>();
            set_->single = single;
            if (single)
                gather<float>();
            else
                gather<double>();
        }
        if (set_->single)
        {
            play_tiles<float>(count, signal);
            next_ += count;
            settle<float>(next_);
        }
        else
        {
            play_tiles<double>(count, signal);
            next_ += count;
            settle<double>(next_);
        }
        s.elapsed_ += static_cast<double>(count) * duration_;
    }

    template <typename Real>
    void Repetitions::play_tiles(size_t count, std::complex<double>* signal)
    {
        Slots<Real>& slots = set_->template slots<Real>();
        const size_t length = isochromats_.coils_ * samples_;
        if (slots.size == 0)
        {
            std::fill(signal, signal + count * length, std::complex<double>(0.0, 0.0));
            return;
        }
        const size_t threads = std::max<size_t>(1, isochromats_.threads_);
        Tile<Real> tile;
        plan_tile(tile, slots.taps);
        std::vector<Real>& grids = set_->template grids<Real>();
        std::vector<Real>& scratch = set_->template scratch<Real>();
        grids.resize(threads * tile.region.back());
        scratch.resize(threads * tile.windows * kTile * kChunk * 2);
        tile.grids = grids.data();
        const CarryRange<Real> range = fastest_carry<Real>(slots.offsets);
        std::vector<std::complex<double>> partial;
        for (size_t done = 0; done < count; done += kTile)
        {
            const size_t first = next_ + done;
            tile.count = std::min(kTile, count - done);
            prepare_tile(tile, first);
            const size_t dropped = carry_tile(tile, slots, range, scratch.data(), threads);
            std::complex<double>* out = signal + done * length;
            for (size_t w = 0; w < tile.windows; ++w)
            {
                if (tile.cells[w] == 0)
                    read_single(tile, w, out);
                else
                    read_transformed(tile, w, threads, partial, out);
            }
            demodulate(first, tile.count, out);
            set_->dropped += dropped;
            if (tile.drop && static_cast<double>(set_->dropped) > kCompactAt * static_cast<double>(slots.size))
            {
                compact(slots, threads);
                set_->dropped = 0;
            }
        }
    }

    template <typename TileOf>
    void Repetitions::plan_tile(TileOf& tile, size_t taps) const
    {
        const Isochromats& s = isochromats_;
        const size_t W = windows_.size();
        tile.windows = W;
        tile.coils = s.coils_;
        tile.taps = taps;
        tile.classes = s.decays_.size();
        tile.drop = divided_ && tolerance_ > 0.0;
        tile.length = samples_;
        tile.table_re.assign(kTile * W * 3, nullptr);
        tile.table_im.assign(kTile * W * 3, nullptr);
        tile.angle.assign(kTile * W * 3, 0.0);
        tile.computed.assign(kTile * W * 3, 0);
        tile.region.assign(1, 0);
        for (size_t w = 0; w < W; ++w)
        {
            const Nufft* transform = transforms_[w].get();
            const size_t cells = transform != nullptr ? transform->grid() : 0;
            const size_t points = cells > 0 ? tile.classes * tile.coils * (cells + taps) : tile.coils;
            tile.cells.push_back(cells);
            tile.samples.push_back(windows_[w].samples);
            tile.offset.push_back(windows_[w].offset);
            tile.transform.push_back(transform);
            tile.decay.push_back(decays(windows_[w]));
            tile.region.push_back(tile.region.back() + points * 2 * kTile);
        }
    }

    template <typename TileOf>
    void Repetitions::prepare_tile(TileOf& tile, size_t first)
    {
        using Real = typename TileOf::Value;
        for (size_t r = 0; r < kTile; ++r)
        {
            tile.turn_cos[r] = Real(1);
            tile.turn_sin[r] = Real(0);
            if (r >= tile.count || divided_)
                continue;
            const size_t n = first + r;
            const double step = turn(n) - turn(n + 1 < phases_.size() ? n + 1 : n);
            tile.turn_cos[r] = static_cast<Real>(std::cos(step));
            tile.turn_sin[r] = static_cast<Real>(std::sin(step));
        }
        encode_tile(tile, first);
    }

    template <typename TileOf>
    void Repetitions::encode_tile(TileOf& tile, size_t first)
    {
        using Real = typename TileOf::Value;
        Tables<Real>& tables = set_->template tables<Real>();
        if (tables.values.size() > kTableValues)
            tables.clear();
        const size_t per_repetition = windows_.size() * 3;
        std::vector<size_t> offsets(kTile * per_repetition, kNone);
        for (size_t at = 0; at < offsets.size(); ++at)
        {
            const int axis = static_cast<int>(at % 3);
            const double area = at / per_repetition < tile.count ? areas_[first * per_repetition + at] : 0.0;
            const bool tabulated = lattice_[axis].tabulated;
            tile.computed[at] = area != 0.0 && !tabulated;
            tile.angle[at] = -kTwoPi * area;
            if (area != 0.0 && tabulated)
                offsets[at] = tables.find(axis, area, lattice_[axis].values);
        }
        /* The tables stay where they are until the next tile's. */
        for (size_t at = 0; at < offsets.size(); ++at)
        {
            const bool held = offsets[at] != kNone;
            tile.table_re[at] = held ? &tables.values[offsets[at]] : nullptr;
            tile.table_im[at] = held ? &tables.values[offsets[at] + lattice_[at % 3].values.size()] : nullptr;
        }
    }

    void Repetitions::demodulate(size_t first, size_t count, std::complex<double>* out) const
    {
        const size_t C = isochromats_.coils_;
        for (size_t w = 0; w < windows_.size(); ++w)
        {
            const Window& window = windows_[w];
            for (size_t r = 0; r < count; ++r)
            {
                const std::complex<double> pulse = std::polar(1.0, turn(first + r) + adc_phases_[first + r]);
                for (size_t k = 0; k < C; ++k)
                {
                    std::complex<double>* into = out + (r * C + k) * samples_ + window.offset;
                    for (size_t j = 0; j < window.samples; ++j)
                        into[j] *= pulse * receivers_[w][j];
                }
            }
        }
    }

    bool Repetitions::split()
    {
        if (divided_)
            return true;
        if (set_)
            throw std::logic_error("the magnetisation is split before the first repetition is played");
        if (next_ >= phases_.size())
            return false;
        Isochromats& s = isochromats_;
        const std::lock_guard<std::mutex> held(s.mutex_);
        double step = 0.0;
        if (!mean_step(step))
            return false;
        const size_t total = s.count_;
        const double c = std::cos(-step);
        const double sn = std::sin(-step);
        std::vector<double> turned(9 * total);
        std::vector<double> fixed(3 * total);
        std::atomic<bool> solvable{true};
        parallel(total, s.threads_, kLeast, [&](size_t, size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i)
                if (!fixed_point(&a_[9 * i], &b_[3 * i], c, sn, &turned[9 * i], &fixed[3 * i]))
                    solvable = false;
        });
        if (!solvable)
            return false;
        a_ = std::move(turned);
        fixed_ = std::move(fixed);
        std::transform(m_.begin(), m_.end(), fixed_.begin(), m_.begin(), std::minus<>());
        step_ = step;
        split_at_ = next_;
        divided_ = true;
        return true;
    }

    bool Repetitions::mean_step(double& step) const
    {
        /* The mean step, from the first: phase registers of limited
         * precision step by one increment only to within their rounding. */
        step = 0.0;
        if (next_ + 1 >= phases_.size())
            return true;
        const double first = std::remainder(turn(next_ + 1) - turn(next_), kTwoPi);
        double offset = 0.0;
        for (size_t n = next_; n + 1 < phases_.size(); ++n)
        {
            const double off = std::remainder(turn(n + 1) - turn(n) - first, kTwoPi);
            if (std::fabs(off) > kStepTolerance)
                return false;
            offset += off;
        }
        step = first + offset / static_cast<double>(phases_.size() - next_ - 1);
        return true;
    }

    double Repetitions::reach() const
    {
        const IsochromatProperties& p = isochromats_.properties_;
        double largest = 0.0;
        for (const std::vector<double>* axis : {&p.x, &p.y, &p.z})
            largest = std::accumulate(axis->begin(), axis->end(), largest, [](double most, double value) {
                return std::max(most, std::fabs(value));
            });
        return largest;
    }

    size_t Repetitions::carried() const
    {
        if (set_)
            return (set_->single ? set_->single_slots.size : set_->double_slots.size) - set_->dropped;
        if (!divided_)
            return isochromats_.count_;
        size_t above = 0;
        for (size_t i = 0; i < isochromats_.count_; ++i)
            above += transient_above(i) ? 1 : 0;
        return above;
    }

    std::vector<size_t> Repetitions::columns_along(const std::vector<int>& axes, std::vector<uint32_t>& members) const
    {
        const size_t total = isochromats_.count_;
        size_t columns = 1;
        std::vector<uint32_t> column(total, 0);
        for (const int axis : axes)
        {
            if (axis < 0 || axis > 2 || !lattice_[axis].tabulated)
                throw std::invalid_argument("the columns run along tabulated axes");
            const size_t size = lattice_[axis].values.size();
            const std::vector<uint32_t>& index = lattice_[axis].index;
            for (size_t i = 0; i < total; ++i)
                column[i] = static_cast<uint32_t>(column[i] * size + index[i]);
            columns *= size;
        }
        return runs(column, columns, members);
    }

    void Repetitions::column_sums(
        size_t window, const std::vector<int>& axes, const double encoding[3], std::complex<double>* out) const
    {
        if (!divided_)
            throw std::invalid_argument("the magnetisation has not been split");
        if (set_)
            throw std::logic_error("the column sums are read before the first repetition is played");
        if (window >= windows_.size())
            throw std::invalid_argument("no ADC window " + std::to_string(window));
        std::vector<uint32_t> members;
        const std::vector<size_t> first = columns_along(axes, members);
        const Isochromats& s = isochromats_;
        const IsochromatProperties& p = s.properties_;
        const Window& w = windows_[window];
        const bool sensitivities = !s.receive_re_.empty();
        /* The widest kernel: the grid, and so the transforms, cost the same. */
        const std::unique_ptr<Nufft> transform = w.samples > 1 ? std::make_unique<Nufft>(w.samples) : nullptr;
        ColumnSums work;
        work.count = s.count_;
        work.coils = s.coils_;
        work.samples = w.samples;
        work.classes = s.decays_.size();
        work.decay = decays(w);
        work.transform = transform.get();
        work.first = first.data();
        work.members = members.data();
        work.u = w.u.data();
        work.v = w.v.data();
        work.fixed = fixed_.data();
        work.receive_re = sensitivities ? s.receive_re_.data() : nullptr;
        work.receive_im = sensitivities ? s.receive_im_.data() : nullptr;
        work.x = p.x.data();
        work.y = p.y.data();
        work.z = p.z.data();
        work.off_resonance = p.off_resonance.data();
        work.decay_of = s.decay_of_.data();
        std::copy(w.area, w.area + 3, work.area);
        work.step = w.step;
        std::copy(encoding, encoding + 3, work.encoding);
        work.out = out;
        const ColumnRange range = fastest_columns();
        /* Parts of about as many isochromats each, whole columns apiece. */
        const size_t columns = first.size() - 1;
        const size_t parts = std::max<size_t>(1, s.threads_);
        std::vector<size_t> bounds(parts + 1, columns);
        for (size_t part = 0; part < parts; ++part)
            bounds[part] = static_cast<size_t>(
                std::lower_bound(first.begin(), first.end() - 1, s.count_ * part / parts) - first.begin());
        parallel(parts, parts, 1, [&](size_t, size_t begin, size_t end) {
            for (size_t part = begin; part < end; ++part)
                range(work, bounds[part], bounds[part + 1]);
        });
    }

    template <typename Real>
    void Repetitions::settle(size_t n)
    {
        /* After the last repetition the magnetisation stays in its frame,
         * unless the map carried it one step further. */
        const double angle = divided_ ? turn(split_at_) + static_cast<double>(n - split_at_) * step_
                                      : turn(n < phases_.size() ? n : phases_.size() - 1);
        const double c = std::cos(angle);
        const double sn = std::sin(angle);
        Isochromats& s = isochromats_;
        const Slots<Real>& slots = set_->template slots<Real>();
        if (divided_)
            parallel(s.count_, s.threads_, kLeast, [&](size_t, size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i)
                {
                    const double x = fixed_[3 * i], y = fixed_[3 * i + 1];
                    s.mx_[i] = c * x - sn * y;
                    s.my_[i] = sn * x + c * y;
                    s.mz_[i] = fixed_[3 * i + 2];
                }
            });
        parallel(slots.size, s.threads_, kLeast, [&](size_t, size_t begin, size_t end) {
            for (size_t slot = begin; slot < end; ++slot)
            {
                const size_t i = slots.id[slot];
                double x = slots.m[slot], y = slots.m[slots.stride + slot], z = slots.m[2 * slots.stride + slot];
                if (divided_)
                {
                    x += fixed_[3 * i];
                    y += fixed_[3 * i + 1];
                    z += fixed_[3 * i + 2];
                }
                s.mx_[i] = c * x - sn * y;
                s.my_[i] = sn * x + c * y;
                s.mz_[i] = z;
            }
        });
        std::fill(s.pending_area_, s.pending_area_ + 3, 0.0);
        s.pending_time_ = 0.0;
    }

} // namespace bloch
