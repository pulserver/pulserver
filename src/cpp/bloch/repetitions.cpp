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

        /** Distinct coordinates along an axis beyond which a phase-encoding
         *  phase is computed per isochromat rather than tabulated. */
        constexpr size_t kLatticeValues = size_t(1) << 16;

        /** Tolerance from which the magnetisation is carried in single
         *  precision: a contraction's rounding stays below it. */
        constexpr double kSingleFrom = 1e-4;

        /** Share of the carried isochromats dropped at which they are
         *  compacted. */
        constexpr double kCompactAt = 0.25;

        /** Cells along each axis of the Z-order curve turned windows' slots
         *  are ordered along. */
        constexpr uint32_t kCurveCells = 1024;

        /** The ten low bits of @p v, two zeros after each: one axis's part of
         *  a Z-order code. */
        uint64_t interleave(uint32_t v)
        {
            uint64_t x = v & (kCurveCells - 1);
            x = (x | (x << 16)) & 0x030000FFull;
            x = (x | (x << 8)) & 0x0300F00Full;
            x = (x | (x << 4)) & 0x030C30C3ull;
            x = (x | (x << 2)) & 0x09249249ull;
            return x;
        }

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

        /** How a slot's phase-encoding phase along an axis is found. */
        enum class Encoding
        {
            none,
            tabulated,
            computed
        };

        /** Bytes the widest vectors the carry uses hold, and the alignment of
         *  the arrays it reads and writes with them. */
        constexpr size_t kVectorBytes = 64;

        /** An allocator of arrays aligned to kVectorBytes. */
        template <typename T>
        struct Aligned
        {
            using value_type = T;
            Aligned() = default;
            template <typename U>
            Aligned(const Aligned<U>&) noexcept
            {
            }
            T* allocate(size_t n)
            {
                return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(kVectorBytes)));
            }
            void deallocate(T* p, size_t) noexcept
            {
                ::operator delete(p, std::align_val_t(kVectorBytes));
            }
            template <typename U>
            bool operator==(const Aligned<U>&) const noexcept
            {
                return true;
            }
            template <typename U>
            bool operator!=(const Aligned<U>&) const noexcept
            {
                return false;
            }
        };

        template <typename T>
        using AlignedVector = std::vector<T, Aligned<T>>;

        /** Where one value of every slot lies: @p width bytes per slot, in
         *  packs of @p lanes slots @p stride bytes apart, or, with no lanes,
         *  one after another. */
        struct Column
        {
            unsigned char* data;
            size_t width;
            size_t lanes;
            size_t stride;
        };

        /**
         * The isochromats carried, one slot each. What the carry reads for
         * several slots at once (magnetisation, map, readout coefficients,
         * limit) is held in packs of as many slots as a vector has lanes, the
         * slots' values of one kind together; what spreading reads for one
         * slot (grid point, weights, factors, phase-encoding index) is held
         * per slot.
         */
        template <typename Real>
        struct Slots
        {
            using Value = Real;
            size_t size = 0;
            size_t lanes = 1;
            size_t windows = 0;
            size_t coils = 0;
            size_t taps = 0;
            /** Whether the maps' and the windows' constant terms apply: not to
             *  a transient about a fixed point. */
            bool offsets = true;
            bool limits = false;
            /** Whether a window is turned, and its slots hold where they lie. */
            bool turned = false;
            /** Values per slot of a pack: m 3, A 9 row-major, b 3 with the
             *  offsets; per window from u_at on, u_width values: u Re x, y, z
             *  then Im x, y, z, and v Re, Im with the offsets; then the
             *  squared magnitude at or below which a transient is dropped. */
            size_t width = 0;
            size_t u_at = 0;
            size_t u_width = 0;
            size_t limit_at = 0;
            /** [pack][value][lane]. */
            AlignedVector<Real> pack;
            std::vector<uint32_t> id;
            /** [slot][window]: the first grid point spread onto. */
            std::vector<uint32_t> start;
            /** [slot][window][tap]. */
            std::vector<Real> weight;
            /** [slot][window][coil][Re, Im]: receive sensitivity times the
             *  kernel's shift. */
            std::vector<Real> factor;
            /** Per axis encoded along, [slot]: the lattice index, or the
             *  coordinate in m where the axis is not tabulated. */
            std::vector<uint32_t> index[3];
            std::vector<Real> coordinate[3];
            /** [slot]: the T2 class. */
            std::vector<uint32_t> decay;
            /** With a turned window, [slot][window]: the phase, in cycles, the
             *  slot turns by from one of the blocks' samples to the next; and
             *  [slot][axis]: its coordinates, in m. */
            std::vector<double> origin;
            std::vector<double> place;

            size_t packs() const
            {
                return (size + lanes - 1) / lanes;
            }

            Real& value(size_t slot, size_t at)
            {
                return pack[(slot / lanes * width + at) * lanes + slot % lanes];
            }

            Real value(size_t slot, size_t at) const
            {
                return pack[(slot / lanes * width + at) * lanes + slot % lanes];
            }

            /** Room, zeroed, for a slot per isochromat of @p order in packs
             *  of @p in_pack slots, the windows, coils, taps and offsets set,
             *  encoded along each axis as @p along says, and with a limit
             *  each where @p with_limits. */
            void allocate(const std::vector<uint32_t>& order, size_t in_pack, const Encoding along[3], bool with_limits)
            {
                size = order.size();
                lanes = in_pack;
                limits = with_limits;
                u_at = offsets ? 15 : 12;
                u_width = offsets ? 8 : 6;
                limit_at = u_at + windows * u_width;
                width = limit_at + (limits ? 1 : 0);
                pack.assign(packs() * width * lanes, Real(0));
                id = order;
                start.assign(size * windows, 0);
                weight.assign(size * windows * taps, Real(0));
                factor.assign(size * windows * coils * 2, Real(0));
                for (int axis = 0; axis < 3; ++axis)
                {
                    index[axis].assign(along[axis] == Encoding::tabulated ? size : 0, 0);
                    coordinate[axis].assign(along[axis] == Encoding::computed ? size : 0, Real(0));
                }
                decay.assign(size, 0);
                origin.assign(turned ? size * windows : 0, 0.0);
                place.assign(turned ? size * 3 : 0, 0.0);
            }

            std::vector<Column> columns()
            {
                std::vector<Column> out;
                for (size_t at = 0; at < width; ++at)
                    out.push_back(
                        {reinterpret_cast<unsigned char*>(pack.data() + at * lanes),
                         sizeof(Real),
                         lanes,
                         width * lanes * sizeof(Real)});
                const auto add = [&](auto& vector, size_t per_slot) {
                    using Item = typename std::decay_t<decltype(vector)>::value_type;
                    if (!vector.empty())
                        out.push_back({reinterpret_cast<unsigned char*>(vector.data()), per_slot * sizeof(Item), 0, 0});
                };
                add(id, 1);
                add(start, windows);
                add(weight, windows * taps);
                add(factor, windows * coils * 2);
                for (int axis = 0; axis < 3; ++axis)
                {
                    add(index[axis], 1);
                    add(coordinate[axis], 1);
                }
                add(decay, 1);
                add(origin, windows);
                add(place, 3);
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

        /** Per coordinate, its phase at each of the tile's repetitions,
         *  [coordinate][repetition], real parts into @p re and imaginary parts
         *  into @p im: the phases one slot's repetitions are turned by lie
         *  together. A repetition's phases are the table's at its offset, or
         *  none where it has none. */
        template <typename Real>
        void tabulate_phases(const Tables<Real>& tables, const size_t* offsets, size_t n, Real* re, Real* im)
        {
            for (size_t k = 0; k < n; ++k)
                for (size_t r = 0; r < kTile; ++r)
                {
                    const bool held = offsets[r] != kNone;
                    re[k * kTile + r] = held ? tables.values[offsets[r] + k] : Real(1);
                    im[k * kTile + r] = held ? tables.values[offsets[r] + n + k] : Real(0);
                }
        }

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
            /** Per window and axis, [w * 3 + axis]: how the tile's
             *  phase-encoding phases are found; where tabulated, the phase per
             *  coordinate and repetition, [coordinate][repetition], in
             *  tables at table_at, real parts then imaginary. */
            std::vector<Encoding> encoding;
            std::vector<size_t> table_at;
            std::vector<const Real*> table_re, table_im;
            AlignedVector<Real> tables;
            /** Per window, axis and repetition, [(w * 3 + axis) * kTile + r]:
             *  -2 pi times the area, for a phase computed per slot. */
            std::vector<double> angle;
            /** Per window, the grid points (none for a window of one sample)
             *  and the start of its part of a worker's grid. */
            std::vector<size_t> cells, region;
            /** Per window, whether it is turned; and per window, axis and
             *  repetition, [(w * 3 + axis) * kTile + r], how much the area
             *  from one sample to the next exceeds the window's, in 1/m. */
            std::vector<unsigned char> turned;
            std::vector<double> delta;
            /** Per turned window, its transform's polynomials(), [w][power]
             *  [tap], powers of each. */
            size_t powers = 0;
            std::vector<Real> polynomials;
            /** Whether the repetitions leave net areas: their phases are then
             *  found as a set of encodings after the windows'. */
            bool netted = false;
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
            /** The workers' grids, region.back() values each, whether each
             *  carried slots through the tile, and the T2 classes from
             *  first_class to before last_class its slots belong to: the
             *  slots are in order of their class. */
            Real* grids = nullptr;
            std::vector<unsigned char> ran;
            std::vector<size_t> first_class, last_class;
        };

        /** A vector of @p L values: one per slot of a pack, or one per
         *  repetition of a part of a tile. */
        template <typename Real, size_t L>
        struct Lanes;

        template <typename Real>
        struct Lanes<Real, 1>
        {
            typedef Real Vector;
        };

#if defined(__GNUC__)
        template <typename Real, size_t L>
        struct Lanes
        {
            typedef Real Vector __attribute__((vector_size(L * sizeof(Real)), aligned(sizeof(Real)), may_alias));
            typedef std::conditional_t<sizeof(Real) == 4, int32_t, int64_t> Scalar;
            typedef Scalar Index __attribute__((vector_size(L * sizeof(Real))));
        };

        /** Rows @p a and @p b of a 2x2 matrix of H x H blocks of lanes, the
         *  blocks off its diagonal exchanged. */
        template <typename Real, size_t L, size_t H, size_t... J>
        BLOCH_INLINE void exchange(
            typename Lanes<Real, L>::Vector& a, typename Lanes<Real, L>::Vector& b, std::index_sequence<J...>)
        {
            using Vector = typename Lanes<Real, L>::Vector;
#if defined(__clang__)
            const Vector low = (Vector)__builtin_shufflevector(a, b, ((J & H) ? L + J - H : J)...);
            const Vector high = (Vector)__builtin_shufflevector(a, b, ((J & H) ? L + J : J + H)...);
#else
            using Index = typename Lanes<Real, L>::Index;
            using Scalar = typename Lanes<Real, L>::Scalar;
            const Vector low = __builtin_shuffle(a, b, Index{static_cast<Scalar>((J & H) ? L + J - H : J)...});
            const Vector high = __builtin_shuffle(a, b, Index{static_cast<Scalar>((J & H) ? L + J : J + H)...});
#endif
            a = low;
            b = high;
        }

        template <typename Real, size_t L, size_t H>
        BLOCH_INLINE void exchange_rows(typename Lanes<Real, L>::Vector* rows)
        {
            for (size_t i = 0; i < L; ++i)
                if ((i & H) == 0)
                    exchange<Real, L, H>(rows[i], rows[i + H], std::make_index_sequence<L>());
        }

        template <typename Real, size_t L, size_t... J>
        BLOCH_INLINE void turn_pairs(typename Lanes<Real, L>::Vector& a, std::index_sequence<J...>)
        {
            using Vector = typename Lanes<Real, L>::Vector;
            const Vector sign = {((J & 1) ? Real(1) : Real(-1))...};
#if defined(__clang__)
            a = (Vector)__builtin_shufflevector(a, a, (J ^ 1)...) * sign;
#else
            using Index = typename Lanes<Real, L>::Index;
            using Scalar = typename Lanes<Real, L>::Scalar;
            a = __builtin_shuffle(a, Index{static_cast<Scalar>(J ^ 1)...}) * sign;
#endif
        }
#endif

        /** Each pair of lanes of @p a, a complex number laid out Re, Im,
         *  times i, in place. */
        template <typename Real, size_t L>
        BLOCH_INLINE void times_i(typename Lanes<Real, L>::Vector& a)
        {
#if defined(__GNUC__)
            if constexpr (L > 1)
                turn_pairs<Real, L>(a, std::make_index_sequence<L>());
#else
            (void)a;
#endif
        }

        /** Transpose @p L rows of @p L lanes in place. */
        template <typename Real, size_t L>
        BLOCH_INLINE void transpose(typename Lanes<Real, L>::Vector* rows)
        {
#if defined(__GNUC__)
            if constexpr (L >= 16)
                exchange_rows<Real, L, 8>(rows);
            if constexpr (L >= 8)
                exchange_rows<Real, L, 4>(rows);
            if constexpr (L >= 4)
                exchange_rows<Real, L, 2>(rows);
            if constexpr (L >= 2)
                exchange_rows<Real, L, 1>(rows);
#else
            (void)rows;
#endif
        }

        /** Packs a worker carries through a tile together: the repetitions of
         *  one pack follow from each other, those of different packs do not,
         *  so that several are computed at once. */
        constexpr size_t kInterleaved = 4;

        template <typename Real, size_t L>
        BLOCH_INLINE typename Lanes<Real, L>::Vector& lanes_at(Real* pack, size_t value)
        {
            return *reinterpret_cast<typename Lanes<Real, L>::Vector*>(pack + value * L);
        }

        /** Pack @p p's coefficients at a window whose map rows start at value
         *  @p u, from its magnetisation @p m at the repetition's start. */
        template <typename Real, size_t L, bool Offsets>
        BLOCH_INLINE void window_coefficients(
            Real* p,
            size_t u,
            const typename Lanes<Real, L>::Vector* m,
            typename Lanes<Real, L>::Vector& re,
            typename Lanes<Real, L>::Vector& im)
        {
            re = lanes_at<Real, L>(p, u) * m[0] + lanes_at<Real, L>(p, u + 1) * m[1] + lanes_at<Real, L>(p, u + 2) * m[2];
            im = lanes_at<Real, L>(p, u + 3) * m[0] + lanes_at<Real, L>(p, u + 4) * m[1] +
                lanes_at<Real, L>(p, u + 5) * m[2];
            if (Offsets)
            {
                re += lanes_at<Real, L>(p, u + 6);
                im += lanes_at<Real, L>(p, u + 7);
            }
        }

        /** Pack @p p's magnetisation @p m carried through repetition @p r by
         *  its map, and turned into the next repetition's frame: by @p c and
         *  @p sn, or where @p Netted by each slot's own turn in @p nets. */
        template <typename Real, size_t L, bool Offsets, bool Netted>
        BLOCH_INLINE void carry_pack(Real* p, typename Lanes<Real, L>::Vector* m, Real c, Real sn, Real* nets, size_t r)
        {
            using Vector = typename Lanes<Real, L>::Vector;
            constexpr size_t T = kTile;
            Vector nx = lanes_at<Real, L>(p, 3) * m[0] + lanes_at<Real, L>(p, 4) * m[1] + lanes_at<Real, L>(p, 5) * m[2];
            Vector ny = lanes_at<Real, L>(p, 6) * m[0] + lanes_at<Real, L>(p, 7) * m[1] + lanes_at<Real, L>(p, 8) * m[2];
            Vector nz = lanes_at<Real, L>(p, 9) * m[0] + lanes_at<Real, L>(p, 10) * m[1] + lanes_at<Real, L>(p, 11) * m[2];
            if (Offsets)
            {
                const Vector bx = nx + lanes_at<Real, L>(p, 12);
                const Vector by = ny + lanes_at<Real, L>(p, 13);
                nz += lanes_at<Real, L>(p, 14);
                if (Netted)
                {
                    /* Each slot's own turn, its net area's phase in it. */
                    const Vector cg = lanes_at<Real, L>(nets, r);
                    const Vector sg = lanes_at<Real, L>(nets + T * L, r);
                    nx = cg * bx - sg * by;
                    ny = sg * bx + cg * by;
                }
                else
                {
                    nx = c * bx - sn * by;
                    ny = sn * bx + c * by;
                }
            }
            m[0] = nx;
            m[1] = ny;
            m[2] = nz;
        }

        /**
         * @p G packs' slots through the tile's @p count repetitions: each
         * repetition's coefficient at each window's first sample, u . m plus v
         * with the constant terms, and the magnetisation at the start of the
         * repetition after the tile. Pack g's coefficients are left in @p q
         * from vector 2 g windows kTile on, [window][Re, Im][repetition], zero
         * past the tile's repetitions and transposed in blocks of L
         * repetitions: row l of block b holds slot l's coefficients of
         * repetitions b L to b L + L - 1.
         */
        template <typename Real, size_t L, bool Offsets, size_t G, bool Netted = false>
        BLOCH_INLINE void carry_packs(
            Real* const* packs,
            size_t count,
            size_t windows,
            size_t u_at,
            size_t u_width,
            const Real* turn_cos,
            const Real* turn_sin,
            typename Lanes<Real, L>::Vector* q,
            Real* nets = nullptr)
        {
            using Vector = typename Lanes<Real, L>::Vector;
            constexpr size_t T = kTile;
            Real* pack[G];
            Vector m[G][3];
            for (size_t g = 0; g < G; ++g)
            {
                pack[g] = packs[g];
                for (size_t k = 0; k < 3; ++k)
                    m[g][k] = lanes_at<Real, L>(pack[g], k);
            }
            for (size_t r = 0; r < count; ++r)
            {
                for (size_t w = 0; w < windows; ++w)
                    for (size_t g = 0; g < G; ++g)
                        window_coefficients<Real, L, Offsets>(
                            pack[g],
                            u_at + w * u_width,
                            m[g],
                            q[((g * windows + w) * 2) * T + r],
                            q[((g * windows + w) * 2 + 1) * T + r]);
                for (size_t g = 0; g < G; ++g)
                    carry_pack<Real, L, Offsets, Netted>(
                        pack[g], m[g], turn_cos[r], turn_sin[r], Netted ? nets + g * 2 * T * L : nullptr, r);
            }
            for (size_t g = 0; g < G; ++g)
                for (size_t k = 0; k < 3; ++k)
                    lanes_at<Real, L>(pack[g], k) = m[g][k];
            for (size_t row = 0; row < 2 * G * windows; ++row)
            {
                for (size_t r = count; r < T; ++r)
                    q[row * T + r] = Vector{};
                for (size_t b = 0; b < T; b += L)
                    transpose<Real, L>(q + row * T + b);
            }
        }

        /** How a window's phase-encoding phases are found for a slot: per
         *  axis tabulated, its table of each coordinate's phase at each of the
         *  tile's repetitions and the slots' lattice indices; per axis
         *  computed, -2 pi times each repetition's area and the slots'
         *  coordinates. */
        template <typename Real>
        struct Encodings
        {
            size_t tabulated = 0;
            const Real* re[3] = {};
            const Real* im[3] = {};
            const uint32_t* index[3] = {};
            size_t computed = 0;
            const double* angle[3] = {};
            const Real* coordinate[3] = {};
        };

        template <typename Real>
        Encodings<Real> encodings(const Tile<Real>& tile, const Slots<Real>& s, size_t w)
        {
            Encodings<Real> out;
            for (size_t axis = 0; axis < 3; ++axis)
            {
                const size_t at = w * 3 + axis;
                if (tile.encoding[at] == Encoding::tabulated)
                {
                    out.re[out.tabulated] = tile.table_re[at];
                    out.im[out.tabulated] = tile.table_im[at];
                    out.index[out.tabulated++] = s.index[axis].data();
                }
                else if (tile.encoding[at] == Encoding::computed)
                {
                    out.angle[out.computed] = &tile.angle[at * kTile];
                    out.coordinate[out.computed++] = s.coordinate[axis].data();
                }
            }
            return out;
        }

        /** A slot's coefficients of a tile, @p er and @p ei, turned by its
         *  phase-encoding phase along each axis. */
        template <typename Real, size_t L>
        BLOCH_INLINE void encode_slot(
            const Encodings<Real>& encoding,
            size_t count,
            size_t slot,
            typename Lanes<Real, L>::Vector* er,
            typename Lanes<Real, L>::Vector* ei)
        {
            using Vector = typename Lanes<Real, L>::Vector;
            constexpr size_t P = kTile / L;
            for (size_t n = 0; n < encoding.tabulated; ++n)
            {
                const size_t row = static_cast<size_t>(encoding.index[n][slot]) * kTile;
                const Vector* cr = reinterpret_cast<const Vector*>(encoding.re[n] + row);
                const Vector* ci = reinterpret_cast<const Vector*>(encoding.im[n] + row);
                for (size_t b = 0; b < P; ++b)
                {
                    const Vector re = er[b] * cr[b] - ei[b] * ci[b];
                    ei[b] = er[b] * ci[b] + ei[b] * cr[b];
                    er[b] = re;
                }
            }
            for (size_t n = 0; n < encoding.computed; ++n)
            {
                /* The tile's phases in a table of their own, so that the
                 * coefficients are not written out to be turned one value at
                 * a time. */
                alignas(kVectorBytes) Real cr[kTile], ci[kTile];
                const double x = static_cast<double>(encoding.coordinate[n][slot]);
                for (size_t r = 0; r < kTile; ++r)
                {
                    const double phase = r < count ? encoding.angle[n][r] * x : 0.0;
                    cr[r] = static_cast<Real>(std::cos(phase));
                    ci[r] = static_cast<Real>(std::sin(phase));
                }
                for (size_t b = 0; b < P; ++b)
                {
                    Vector c, sn;
                    std::memcpy(&c, cr + b * L, sizeof(Vector));
                    std::memcpy(&sn, ci + b * L, sizeof(Vector));
                    const Vector re = er[b] * c - ei[b] * sn;
                    ei[b] = er[b] * sn + ei[b] * c;
                    er[b] = re;
                }
            }
        }

        /** The coefficients of a run of slots that spread onto the same grid
         *  points, @p e, [slot][Re, Im][part], times each coil's factor and
         *  each tap's weight, summed over the run and added onto the grid
         *  points from @p row on, a coil's @p per_coil values apart; onto one
         *  point, unweighted, without taps. Slot l's factor of coil k is at
         *  factor[l * factor_step + 2 k] and its weight of tap j at
         *  weight[l * weight_step + j]; @p v holds as many values as @p e. */
        template <typename Real, size_t L>
        BLOCH_INLINE void spread_run(
            const typename Lanes<Real, L>::Vector* e,
            size_t count,
            const Real* factor,
            size_t factor_step,
            const Real* weight,
            size_t weight_step,
            size_t coils,
            size_t taps,
            size_t per_coil,
            Real* row,
            typename Lanes<Real, L>::Vector* v)
        {
            using Vector = typename Lanes<Real, L>::Vector;
            constexpr size_t T = kTile;
            constexpr size_t P = T / L;
            for (size_t k = 0; k < coils; ++k)
            {
                for (size_t l = 0; l < count; ++l)
                {
                    const Real fr = factor[l * factor_step + 2 * k];
                    const Real fi = factor[l * factor_step + 2 * k + 1];
                    const Vector* er = e + 2 * l * P;
                    const Vector* ei = er + P;
                    for (size_t b = 0; b < P; ++b)
                    {
                        v[2 * l * P + b] = fr * er[b] - fi * ei[b];
                        v[(2 * l + 1) * P + b] = fr * ei[b] + fi * er[b];
                    }
                }
                Real* point = row + k * per_coil;
                for (size_t j = 0; j < std::max<size_t>(taps, 1); ++j)
                {
                    /* Two sums, for two chains of additions at once. */
                    Vector ar[P] = {}, ai[P] = {}, br[P] = {}, bi[P] = {};
                    size_t l = 0;
                    for (; l + 1 < count; l += 2)
                    {
                        const Real w0 = taps == 0 ? Real(1) : weight[l * weight_step + j];
                        const Real w1 = taps == 0 ? Real(1) : weight[(l + 1) * weight_step + j];
                        for (size_t b = 0; b < P; ++b)
                        {
                            ar[b] += w0 * v[2 * l * P + b];
                            ai[b] += w0 * v[(2 * l + 1) * P + b];
                            br[b] += w1 * v[(2 * l + 2) * P + b];
                            bi[b] += w1 * v[(2 * l + 3) * P + b];
                        }
                    }
                    if (l < count)
                    {
                        const Real w0 = taps == 0 ? Real(1) : weight[l * weight_step + j];
                        for (size_t b = 0; b < P; ++b)
                        {
                            ar[b] += w0 * v[2 * l * P + b];
                            ai[b] += w0 * v[(2 * l + 1) * P + b];
                        }
                    }
                    Vector* re = reinterpret_cast<Vector*>(point + j * 2 * T);
                    Vector* im = reinterpret_cast<Vector*>(point + j * 2 * T + T);
                    for (size_t b = 0; b < P; ++b)
                    {
                        re[b] += ar[b] + br[b];
                        im[b] += ai[b] + bi[b];
                    }
                }
            }
        }

        /** Zero the transients of the pack's first @p lanes slots at or below
         *  their limit; return how many were not zero. */
        template <typename Real>
        BLOCH_INLINE size_t drop_pack(const Slots<Real>& s, Real* pack, size_t lanes)
        {
            const size_t L = s.lanes;
            Real* m0 = pack;
            Real* m1 = pack + L;
            Real* m2 = pack + 2 * L;
            const Real* limit = pack + s.limit_at * L;
            size_t dropped = 0;
            for (size_t l = 0; l < lanes; ++l)
            {
                const Real size = m0[l] * m0[l] + m1[l] * m1[l] + m2[l] * m2[l];
                if (size > Real(0) && !(size > limit[l]))
                {
                    m0[l] = m1[l] = m2[l] = Real(0);
                    ++dropped;
                }
            }
            return dropped;
        }

        /** carry_packs over @p group packs, at most kInterleaved: together
         *  where there are that many, one at a time otherwise; each slot
         *  turned by its own turns from @p nets, as net_turns() leaves them,
         *  where given. */
        template <typename Real, size_t L, bool Offsets>
        BLOCH_INLINE void carry_group(
            Real* const* packs,
            size_t group,
            size_t count,
            const Slots<Real>& s,
            size_t windows,
            const Real* turn_cos,
            const Real* turn_sin,
            typename Lanes<Real, L>::Vector* q,
            Real* nets)
        {
            const size_t u_at = s.u_at;
            const size_t u_width = s.u_width;
            if (group == kInterleaved && nets != nullptr)
                carry_packs<Real, L, Offsets, kInterleaved, true>(
                    packs, count, windows, u_at, u_width, turn_cos, turn_sin, q, nets);
            else if (group == kInterleaved)
                carry_packs<Real, L, Offsets, kInterleaved>(packs, count, windows, u_at, u_width, turn_cos, turn_sin, q);
            for (size_t g = 0; g < group && group != kInterleaved; ++g)
            {
                auto* own = q + g * 2 * windows * kTile;
                if (nets != nullptr)
                    carry_packs<Real, L, Offsets, 1, true>(
                        packs + g, count, windows, u_at, u_width, turn_cos, turn_sin, own, nets + g * 2 * kTile * L);
                else
                    carry_packs<Real, L, Offsets, 1>(packs + g, count, windows, u_at, u_width, turn_cos, turn_sin, own);
            }
        }

        /** Each slot's turn from each repetition's frame to the next one's,
         *  the phase of the repetition's net area included, for the @p group
         *  packs from @p p on: [pack][cos, sin][repetition][lane] in
         *  @p turns. */
        template <typename Real, size_t L>
        BLOCH_INLINE void net_turns(
            const Tile<Real>& tile, const Encodings<Real>& net, const Slots<Real>& s, size_t p, size_t group, Real* turns)
        {
            using Vector = typename Lanes<Real, L>::Vector;
            constexpr size_t T = kTile;
            constexpr size_t P = T / L;
            for (size_t g = 0; g < group; ++g)
            {
                Real* cosines = turns + g * 2 * T * L;
                Real* sines = cosines + T * L;
                const size_t lanes = std::min(L, s.size - (p + g) * L);
                for (size_t l = 0; l < L; ++l)
                {
                    Vector er[P], ei[P];
                    for (size_t b = 0; b < P; ++b)
                    {
                        er[b] = Vector{} + Real(1);
                        ei[b] = Vector{};
                    }
                    if (l < lanes)
                        encode_slot<Real, L>(net, tile.count, (p + g) * L + l, er, ei);
                    Real re[T], im[T];
                    std::memcpy(re, er, sizeof(re));
                    std::memcpy(im, ei, sizeof(im));
                    for (size_t r = 0; r < T; ++r)
                    {
                        cosines[r * L + l] = tile.turn_cos[r] * re[r] - tile.turn_sin[r] * im[r];
                        sines[r * L + l] = tile.turn_sin[r] * re[r] + tile.turn_cos[r] * im[r];
                    }
                }
            }
        }

        /** The coefficients of pack @p p's first @p lanes slots in one
         *  window, @p coefficients as carry_packs leaves them, turned by each
         *  slot's phase-encoding phase into @p e, [slot][Re, Im][part]. */
        template <typename Real, size_t L>
        BLOCH_INLINE void encode_pack(
            const Encodings<Real>& encoding,
            size_t count,
            size_t p,
            size_t lanes,
            const typename Lanes<Real, L>::Vector* coefficients,
            typename Lanes<Real, L>::Vector* e)
        {
            using Vector = typename Lanes<Real, L>::Vector;
            constexpr size_t P = kTile / L;
            for (size_t l = 0; l < lanes; ++l)
            {
                Vector er[P], ei[P];
                for (size_t b = 0; b < P; ++b)
                {
                    er[b] = coefficients[b * L + l];
                    ei[b] = coefficients[kTile + b * L + l];
                }
                encode_slot<Real, L>(encoding, count, p * L + l, er, ei);
                for (size_t b = 0; b < P; ++b)
                {
                    e[2 * l * P + b] = er[b];
                    e[(2 * l + 1) * P + b] = ei[b];
                }
            }
        }

        /** Where slot @p slot spreads in window @p w's part of a grid: at the
         *  first coil of its T2 class and its first grid point; at the part's
         *  start for a window of one sample. */
        template <typename Real>
        BLOCH_INLINE size_t grid_point(const Tile<Real>& tile, const Slots<Real>& s, size_t slot, size_t w)
        {
            const size_t cells = tile.cells[w];
            if (cells == 0)
                return 0;
            return (s.decay[slot] * tile.coils * (cells + tile.taps) + s.start[slot * tile.windows + w]) * 2 * kTile;
        }

        /** Pack @p p's first @p lanes slots' encoded coefficients of window
         *  @p w, @p e as encode_pack leaves them, spread onto @p grid a run of
         *  slots on the same grid points at a time. */
        template <typename Real, size_t L>
        BLOCH_INLINE void spread_pack(
            const Tile<Real>& tile,
            const Slots<Real>& s,
            size_t w,
            size_t p,
            size_t lanes,
            const typename Lanes<Real, L>::Vector* e,
            Real* grid,
            typename Lanes<Real, L>::Vector* v)
        {
            constexpr size_t P = kTile / L;
            const size_t W = tile.windows;
            const size_t C = tile.coils;
            const size_t taps = tile.taps;
            const size_t cells = tile.cells[w];
            size_t point[L];
            for (size_t l = 0; l < lanes; ++l)
                point[l] = grid_point(tile, s, p * L + l, w);
            const size_t per_coil = cells == 0 ? 2 * kTile : (cells + taps) * 2 * kTile;
            for (size_t l0 = 0; l0 < lanes;)
            {
                size_t l1 = l0 + 1;
                while (l1 < lanes && point[l1] == point[l0])
                    ++l1;
                const size_t slot = p * L + l0;
                spread_run<Real, L>(
                    e + 2 * l0 * P,
                    l1 - l0,
                    &s.factor[(slot * W + w) * C * 2],
                    W * C * 2,
                    taps == 0 ? nullptr : &s.weight[(slot * W + w) * taps],
                    W * taps,
                    C,
                    cells == 0 ? 0 : taps,
                    per_coil,
                    grid + tile.region[w] + point[l0],
                    v);
                l0 = l1;
            }
        }

        /** Values a turned window's spreading works in: two per coil, in a
         *  whole number of kVectorBytes. */
        template <typename Real>
        size_t turned_values(const Tile<Real>& tile)
        {
            constexpr size_t vector = kVectorBytes / sizeof(Real);
            return (2 * tile.coils + vector - 1) / vector * vector;
        }

        /** Each of @p values factors, pairs laid out Re, Im, times i, into
         *  @p swapped: @p chunks whole vectors, then one pair at a time. */
        template <typename Real, size_t L>
        BLOCH_INLINE void factors_times_i(const Real* factor, size_t values, size_t chunks, Real* swapped)
        {
            using Vector = typename Lanes<Real, L>::Vector;
            const Vector* f = reinterpret_cast<const Vector*>(factor);
            Vector* g = reinterpret_cast<Vector*>(swapped);
            for (size_t k = 0; k < chunks; ++k)
            {
                g[k] = f[k];
                times_i<Real, L>(g[k]);
            }
            for (size_t k = chunks * L; k < values; k += 2)
            {
                swapped[k] = -factor[k + 1];
                swapped[k + 1] = factor[k];
            }
        }

        /** Where a slot at @p place, whose window's first sample lies at
         *  @p origin, is spread at each of the tile's repetitions: its first
         *  grid point and its place in its interval, from the repetitions'
         *  steps @p delta, [axis][repetition]. */
        template <typename Real>
        BLOCH_INLINE void locate_repetitions(
            const Nufft& transform, const double* delta, const double* place, double origin, double* first, Real* offset)
        {
            const double x = place[0];
            const double y = place[1];
            const double z = place[2];
            for (size_t r = 0; r < kTile; ++r)
            {
                double at = 0.0;
                first[r] = transform.locate(origin + delta[r] * x + delta[kTile + r] * y + delta[2 * kTile + r] * z, at);
                offset[r] = static_cast<Real>(at);
            }
        }

        /** Each tap's weights at the tile's repetitions into @p sum,
         *  [tap][repetition], by Horner's rule over their places @p t, the
         *  taps side by side. */
        template <typename Real, size_t L>
        BLOCH_INLINE void weigh_repetitions(
            const Real* polynomial,
            size_t powers,
            size_t taps,
            const typename Lanes<Real, L>::Vector* t,
            typename Lanes<Real, L>::Vector* sum)
        {
            using Vector = typename Lanes<Real, L>::Vector;
            constexpr size_t P = kTile / L;
            for (size_t j = 0; j < taps; ++j)
                for (size_t q = 0; q < P; ++q)
                    sum[j * P + q] = Vector{} + polynomial[(powers - 1) * taps + j];
            for (size_t m = powers - 1; m-- > 0;)
                for (size_t j = 0; j < taps; ++j)
                {
                    const Real c = polynomial[m * taps + j];
                    for (size_t q = 0; q < P; ++q)
                        sum[j * P + q] = sum[j * P + q] * t[q] + c;
                }
        }

        /** A coefficient @p a + i @p b times each factor, Re c f + Im c (i f),
         *  spread onto the @p taps grid points from @p point with @p weight
         *  each: @p chunks whole vectors of the values, then one at a time. */
        template <typename Real, size_t L>
        BLOCH_INLINE void spread_repetition(
            Real a,
            Real b,
            const Real* factor,
            const Real* swapped,
            size_t values,
            size_t chunks,
            const Real* weight,
            size_t taps,
            Real* point)
        {
            using Vector = typename Lanes<Real, L>::Vector;
            const Vector* f = reinterpret_cast<const Vector*>(factor);
            const Vector* g = reinterpret_cast<const Vector*>(swapped);
            for (size_t k = 0; k < chunks; ++k)
            {
                const Vector v = a * f[k] + b * g[k];
                for (size_t j = 0; j < taps; ++j)
                    *reinterpret_cast<Vector*>(point + j * values + k * L) += weight[j] * v;
            }
            for (size_t k = chunks * L; k < values; ++k)
            {
                const Real v = a * factor[k] + b * swapped[k];
                for (size_t j = 0; j < taps; ++j)
                    point[j * values + k] += weight[j] * v;
            }
        }

        /** Pack @p p's first @p lanes slots' encoded coefficients of turned
         *  window @p w, @p e as encode_pack leaves them, spread onto @p grid
         *  a slot and a repetition at a time, each repetition along its own
         *  direction: the window's part of the grid is [T2 class][repetition]
         *  [grid point][coil][Re, Im]. @p work holds two values per coil. */
        template <typename Real, size_t L>
        BLOCH_INLINE void spread_turned(
            const Tile<Real>& tile,
            const Slots<Real>& s,
            size_t w,
            size_t p,
            size_t lanes,
            const typename Lanes<Real, L>::Vector* e,
            Real* grid,
            Real* work)
        {
            using Vector = typename Lanes<Real, L>::Vector;
            constexpr size_t P = kTile / L;
            const size_t W = tile.windows;
            const size_t values = 2 * tile.coils;
            /* The values of whole vectors, then the rest one at a time. */
            const size_t chunks = L > 1 ? values / L : 0;
            const size_t taps = tile.taps;
            const size_t points = tile.cells[w] + taps;
            const Real* polynomial = &tile.polynomials[w * tile.powers * taps];
            alignas(64) double first[kTile];
            alignas(64) Real offset[kTile];
            alignas(64) Real weights[Nufft::kWidest * kTile];
            for (size_t l = 0; l < lanes; ++l)
            {
                const size_t slot = p * L + l;
                const Real* factor = &s.factor[(slot * W + w) * values];
                factors_times_i<Real, L>(factor, values, chunks, work);
                locate_repetitions<Real>(
                    *tile.transform[w], &tile.delta[w * 3 * kTile], &s.place[3 * slot], s.origin[slot * W + w], first, offset);
                weigh_repetitions<Real, L>(
                    polynomial, tile.powers, taps, reinterpret_cast<const Vector*>(offset), reinterpret_cast<Vector*>(weights));
                const Real* re = reinterpret_cast<const Real*>(e + 2 * l * P);
                const Real* im = reinterpret_cast<const Real*>(e + (2 * l + 1) * P);
                Real* rows = grid + tile.region[w] + s.decay[slot] * kTile * points * values;
                for (size_t r = 0; r < tile.count; ++r)
                {
                    Real weight[Nufft::kWidest];
                    for (size_t j = 0; j < taps; ++j)
                        weight[j] = weights[j * kTile + r];
                    spread_repetition<Real, L>(
                        re[r],
                        im[r],
                        factor,
                        work,
                        values,
                        chunks,
                        weight,
                        taps,
                        rows + (r * points + static_cast<size_t>(first[r])) * values);
                }
            }
        }

        /** Carry packs @p first to @p last through the tile: each slot's
         *  coefficients spread onto @p grid, per window onto the grid points
         *  of its T2 and coil or summed without a kernel for a window of one
         *  sample, and the magnetisation at the start of the repetition after
         *  the tile; drop the transients at or below their limit where the
         *  tile drops them and return how many. */
        template <typename Real, size_t L, bool Offsets>
        BLOCH_INLINE size_t carry(
            const Tile<Real>& tile, Slots<Real>& s, size_t first, size_t last, Real* grid, Real* scratch)
        {
            using Vector = typename Lanes<Real, L>::Vector;
            constexpr size_t T = kTile;
            constexpr size_t G = kInterleaved;
            const size_t W = tile.windows;
            Real turn_cos[kTile], turn_sin[kTile];
            std::copy(tile.turn_cos, tile.turn_cos + kTile, turn_cos);
            std::copy(tile.turn_sin, tile.turn_sin + kTile, turn_sin);
            std::vector<Encodings<Real>> encoding;
            for (size_t w = 0; w < W; ++w)
                encoding.push_back(encodings(tile, s, w));
            const Encodings<Real> net = tile.netted ? encodings(tile, s, W) : Encodings<Real>{};
            /* The packs' coefficients, [pack][window][Re, Im][repetition];
             * one window's, encoded, [slot][Re, Im][part]; the latter times a
             * coil's factor; and what spreading a turned window works in. */
            Vector* q = reinterpret_cast<Vector*>(scratch);
            Vector* e = q + G * 2 * W * T;
            Vector* v = e + 2 * T;
            Real* work = reinterpret_cast<Real*>(v + 2 * T);
            Real* nets = tile.netted ? work + turned_values(tile) : nullptr;
            size_t dropped = 0;
            for (size_t p = first; p < last;)
            {
                const size_t group = std::min(G, last - p);
                Real* packs[G];
                for (size_t g = 0; g < group; ++g)
                    packs[g] = s.pack.data() + (p + g) * s.width * L;
                if (nets != nullptr)
                    net_turns<Real, L>(tile, net, s, p, group, nets);
                carry_group<Real, L, Offsets>(packs, group, tile.count, s, W, turn_cos, turn_sin, q, nets);
                for (size_t g = 0; g < group; ++g, ++p)
                {
                    const size_t lanes = std::min(L, s.size - p * L);
                    for (size_t w = 0; w < W; ++w)
                    {
                        encode_pack<Real, L>(encoding[w], tile.count, p, lanes, q + (g * W + w) * 2 * T, e);
                        if (tile.turned[w])
                            spread_turned<Real, L>(tile, s, w, p, lanes, e, grid, work);
                        else
                            spread_pack<Real, L>(tile, s, w, p, lanes, e, grid, v);
                    }
                    if (tile.drop)
                        dropped += drop_pack(s, packs[g], lanes);
                }
            }
            return dropped;
        }

        template <typename Real>
        using CarryRange = size_t (*)(const Tile<Real>&, Slots<Real>&, size_t, size_t, Real*, Real*);

        template <typename Real, size_t L, bool Offsets>
        size_t carry_plain(const Tile<Real>& tile, Slots<Real>& s, size_t first, size_t last, Real* grid, Real* scratch)
        {
            return carry<Real, L, Offsets>(tile, s, first, last, grid, scratch);
        }

#if defined(BLOCH_X86_64) && defined(__GNUC__)
        template <typename Real, size_t L, bool Offsets>
        BLOCH_AVX2 size_t
            carry_avx2(const Tile<Real>& tile, Slots<Real>& s, size_t first, size_t last, Real* grid, Real* scratch)
        {
            return carry<Real, L, Offsets>(tile, s, first, last, grid, scratch);
        }

        template <typename Real, size_t L, bool Offsets>
        BLOCH_AVX512 size_t
            carry_avx512(const Tile<Real>& tile, Slots<Real>& s, size_t first, size_t last, Real* grid, Real* scratch)
        {
            return carry<Real, L, Offsets>(tile, s, first, last, grid, scratch);
        }
#endif

        /** The carry this processor runs fastest, and the lanes of the packs
         *  it reads. */
        template <typename Real>
        struct Carry
        {
            CarryRange<Real> range;
            size_t lanes;
        };

        template <typename Real>
        Carry<Real> fastest_carry(bool offsets)
        {
#if defined(BLOCH_X86_64) && defined(__GNUC__)
            static const bool widest = avx512f();
            static const bool wide = avx2_and_fma();
            constexpr size_t Z = 64 / sizeof(Real);
            constexpr size_t Y = 32 / sizeof(Real);
            if (widest)
                return {offsets ? carry_avx512<Real, Z, true> : carry_avx512<Real, Z, false>, Z};
            if (wide)
                return {offsets ? carry_avx2<Real, Y, true> : carry_avx2<Real, Y, false>, Y};
#endif
#if defined(__GNUC__)
            constexpr size_t X = 16 / sizeof(Real);
            return {offsets ? carry_plain<Real, X, true> : carry_plain<Real, X, false>, X};
#else
            return {offsets ? carry_plain<Real, 1, true> : carry_plain<Real, 1, false>, 1};
#endif
        }

        /** Values of a worker's scratch: carry()'s coefficients, what a turned
         *  window's spreading works in, and the slots' turns where the
         *  repetitions leave net areas. */
        template <typename Real>
        size_t scratch_per_worker(const Tile<Real>& tile, size_t lanes)
        {
            const size_t nets = tile.netted ? kInterleaved * 2 * kTile * lanes : 0;
            return (kInterleaved * tile.windows + 2) * 2 * kTile * lanes + turned_values(tile) + nets;
        }

        /** Carry every pack through the tile, each worker onto its own grid,
         *  and drop the transients at or below their limit where the tile
         *  drops them; return how many were dropped. */
        template <typename Real>
        size_t carry_tile(Tile<Real>& tile, Slots<Real>& slots, CarryRange<Real> range, Real* scratch, size_t threads)
        {
            const size_t grid_size = tile.region.back();
            const size_t scratch_size = scratch_per_worker(tile, slots.lanes);
            const size_t packs = slots.packs();
            const size_t least = std::max<size_t>(1, kLeast / slots.lanes);
            const size_t workers = workers_for(packs, threads, least);
            tile.ran.assign(workers, 0);
            tile.first_class.assign(workers, 0);
            tile.last_class.assign(workers, 0);
            std::atomic<size_t> dropped{0};
            parallel(packs, threads, least, [&](size_t worker, size_t begin, size_t end) {
                const size_t first = slots.decay[begin * slots.lanes];
                const size_t last = slots.decay[std::min(slots.size, end * slots.lanes) - 1] + 1;
                Real* grid = tile.grids + worker * grid_size;
                /* Only the rows of the worker's classes are spread onto, and
                 * only they are read. */
                for (size_t w = 0; w < tile.windows; ++w)
                {
                    Real* part = grid + tile.region[w];
                    const size_t per_class = tile.coils * (tile.cells[w] + tile.taps) * 2 * kTile;
                    if (tile.cells[w] == 0)
                        std::fill(part, grid + tile.region[w + 1], Real(0));
                    else
                        std::fill(part + first * per_class, part + last * per_class, Real(0));
                }
                dropped += range(tile, slots, begin, end, grid, scratch + worker * scratch_size);
                tile.first_class[worker] = first;
                tile.last_class[worker] = last;
                tile.ran[worker] = 1;
            });
            return dropped.load();
        }

        /** Move the values of @p column of the slots @p keep marks to the
         *  front, in their order. */
        template <size_t Bytes>
        void compact_column(const Column& column, const std::vector<unsigned char>& keep)
        {
            const size_t width = Bytes == 0 ? column.width : Bytes;
            if (column.lanes == 0)
            {
                unsigned char* to = column.data;
                const unsigned char* from = column.data;
                for (size_t i = 0; i < keep.size(); ++i, from += width)
                    if (keep[i])
                    {
                        if (to != from)
                            std::memcpy(to, from, width);
                        to += width;
                    }
                return;
            }
            /* Lane by lane through the packs, reading and writing alike. */
            unsigned char* to_pack = column.data;
            size_t to_lane = 0;
            const unsigned char* pack = column.data;
            for (size_t i = 0; i < keep.size(); pack += column.stride)
                for (size_t lane = 0; lane < column.lanes && i < keep.size(); ++lane, ++i)
                    if (keep[i])
                    {
                        unsigned char* to = to_pack + to_lane * width;
                        const unsigned char* from = pack + lane * width;
                        if (to != from)
                            std::memcpy(to, from, width);
                        if (++to_lane == column.lanes)
                        {
                            to_lane = 0;
                            to_pack += column.stride;
                        }
                    }
        }

        /** Keep only the slots whose transient is above its limit, in their
         *  order, and zero the lanes of the last pack past them. */
        template <typename Real>
        void compact(Slots<Real>& s, size_t threads)
        {
            std::vector<unsigned char> keep(s.size);
            const size_t lanes = s.lanes;
            parallel(s.packs(), threads, std::max<size_t>(1, kLeast / lanes), [&](size_t, size_t begin, size_t end) {
                for (size_t p = begin; p < end; ++p)
                {
                    const Real* pack = s.pack.data() + p * s.width * lanes;
                    const Real* limit = pack + s.limit_at * lanes;
                    for (size_t l = 0; l < lanes && p * lanes + l < s.size; ++l)
                    {
                        const Real x = pack[l], y = pack[lanes + l], z = pack[2 * lanes + l];
                        keep[p * lanes + l] = x * x + y * y + z * z > limit[l];
                    }
                }
            });
            const std::vector<Column> columns = s.columns();
            parallel(columns.size(), threads, 1, [&](size_t, size_t begin, size_t end) {
                for (size_t c = begin; c < end; ++c)
                    switch (columns[c].width)
                    {
                    case 4:
                        compact_column<4>(columns[c], keep);
                        break;
                    case 8:
                        compact_column<8>(columns[c], keep);
                        break;
                    default:
                        compact_column<0>(columns[c], keep);
                    }
            });
            s.size = static_cast<size_t>(std::count(keep.begin(), keep.end(), 1));
            for (size_t slot = s.size; slot < s.packs() * s.lanes; ++slot)
                for (size_t at = 0; at < s.width; ++at)
                    s.value(slot, at) = Real(0);
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
                if (!tile.ran[worker] || d < tile.first_class[worker] || d >= tile.last_class[worker])
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

        /** The workers' grids of turned window @p w, T2 class @p d and
         *  repetition @p r, summed and folded onto the window's grid points,
         *  per coil into @p columns, [coil][grid point]; false where no worker
         *  spread onto them. */
        template <typename Real>
        bool fold_turned(const Tile<Real>& tile, size_t w, size_t d, size_t r, std::vector<std::complex<double>>& columns)
        {
            const size_t cells = tile.cells[w];
            const size_t points = cells + tile.taps;
            const size_t coils = tile.coils;
            const size_t grid_size = tile.region.back();
            std::fill(columns.begin(), columns.end(), std::complex<double>(0.0, 0.0));
            bool any = false;
            for (size_t worker = 0; worker < tile.ran.size(); ++worker)
            {
                if (!tile.ran[worker] || d < tile.first_class[worker] || d >= tile.last_class[worker])
                    continue;
                any = true;
                const Real* point =
                    tile.grids + worker * grid_size + tile.region[w] + (d * kTile + r) * points * 2 * coils;
                for (size_t g = 0; g < points; ++g, point += 2 * coils)
                {
                    const size_t at = g < cells ? g : g - cells;
                    for (size_t k = 0; k < coils; ++k)
                        columns[k * cells + at] += std::complex<double>(point[2 * k], point[2 * k + 1]);
                }
            }
            return any;
        }

        /** Turned window @p w's samples per repetition of the tile and coil,
         *  as read_transformed() reads a window, a T2 class and a repetition
         *  at a time. */
        template <typename Real>
        void read_turned(
            const Tile<Real>& tile,
            size_t w,
            size_t threads,
            std::vector<std::complex<double>>& partial,
            std::complex<double>* out)
        {
            const size_t samples = tile.samples[w];
            const size_t coils = tile.coils;
            const size_t cells = tile.cells[w];
            partial.assign(tile.classes * coils * kTile * samples, std::complex<double>(0.0, 0.0));
            parallel(tile.classes * tile.count, threads, 1, [&](size_t, size_t from, size_t to) {
                std::vector<std::complex<double>> columns(coils * cells);
                for (size_t item = from; item < to; ++item)
                {
                    const size_t d = item / tile.count;
                    const size_t r = item % tile.count;
                    if (!fold_turned(tile, w, d, r, columns))
                        continue;
                    const double* decay = &tile.decay[w][d * samples];
                    for (size_t k = 0; k < coils; ++k)
                    {
                        std::complex<double>* sample = &partial[((d * coils + k) * kTile + r) * samples];
                        tile.transform[w]->finish(&columns[k * cells], sample);
                        for (size_t j = 0; j < samples; ++j)
                            sample[j] *= decay[j];
                    }
                }
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
        /** Per worker, its grid; and its pack's coefficients. */
        AlignedVector<float> single_grids, single_scratch;
        AlignedVector<double> double_grids, double_scratch;
        Tables<float> single_tables;
        Tables<double> double_tables;
        /** Slots dropped since the last compaction. */
        size_t dropped = 0;

        template <typename Real>
        Slots<Real>& slots();
        template <typename Real>
        Tables<Real>& tables();
        template <typename Real>
        AlignedVector<Real>& grids();
        template <typename Real>
        AlignedVector<Real>& scratch();
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
    AlignedVector<float>& Repetitions::Set::grids<float>()
    {
        return single_grids;
    }
    template <>
    AlignedVector<double>& Repetitions::Set::grids<double>()
    {
        return double_grids;
    }
    template <>
    AlignedVector<float>& Repetitions::Set::scratch<float>()
    {
        return single_scratch;
    }
    template <>
    AlignedVector<double>& Repetitions::Set::scratch<double>()
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
        std::vector<double> readouts,
        std::vector<double> nets,
        double tolerance)
        : isochromats_(isochromats),
          blocks_(std::move(blocks)),
          phases_(std::move(phases)),
          adc_phases_(std::move(adc_phases)),
          areas_(std::move(areas)),
          nets_(std::move(nets)),
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
        if (!nets_.empty() && nets_.size() != phases_.size() * 3)
            throw std::invalid_argument("the repetitions need three net areas each, or none");
        if (std::all_of(nets_.begin(), nets_.end(), [](double area) { return area == 0.0; }))
            nets_.clear();
        const size_t width = Nufft::width_for(tolerance_);
        for (const Window& window : windows_)
        {
            transforms_.push_back(window.samples > 1 ? std::make_unique<Nufft>(window.samples, width) : nullptr);
            receivers_.push_back(phasors(window.receiver));
        }
        turn_windows(readouts);
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

    void Repetitions::turn_windows(const std::vector<double>& readouts)
    {
        const size_t W = windows_.size();
        if (!readouts.empty() && readouts.size() != phases_.size() * W * 3)
            throw std::invalid_argument(
                "the repetitions need three readout gradients per ADC window, " + std::to_string(W) +
                " per repetition, or none");
        readouts_.assign(phases_.size() * W * 3, 0.0);
        for (size_t at = 0; at < readouts.size(); ++at)
        {
            Window& window = windows_[at / 3 % W];
            if (transforms_[at / 3 % W] == nullptr || readouts[at] == 0.0)
                continue;
            window.turned = true;
            readouts_[at] = readouts[at] * window.step;
        }
        /* A turned sample's spreading factor turns as a phase encoding of
         * centre() times its change of step would: it joins the area. */
        for (size_t at = 0; at < readouts_.size(); ++at)
            if (readouts_[at] != 0.0)
                areas_[at] += static_cast<double>(transforms_[at / 3 % W]->centre()) * readouts_[at];
    }

    bool Repetitions::turned() const
    {
        return std::any_of(windows_.begin(), windows_.end(), [](const Window& window) { return window.turned; });
    }

    double Repetitions::encoding_area(size_t n, size_t at) const
    {
        const size_t W = windows_.size();
        return at < W * 3 ? areas_[n * W * 3 + at] : nets_[n * 3 + at - W * 3];
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
            for (size_t k = static_cast<size_t>(axis); k < nets_.size() && !encoded_[axis]; k += 3)
                encoded_[axis] = nets_[k] != 0.0;
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
        if (turned())
            return spatial_order(chosen);
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

    std::vector<uint32_t> Repetitions::spatial_order(const std::vector<uint32_t>& chosen) const
    {
        const Isochromats& s = isochromats_;
        const IsochromatProperties& p = s.properties_;
        const std::vector<double>* axes[3] = {&p.x, &p.y, &p.z};
        double low[3], scale[3];
        for (int axis = 0; axis < 3; ++axis)
        {
            const auto range = std::minmax_element(axes[axis]->begin(), axes[axis]->end());
            low[axis] = range.first == axes[axis]->end() ? 0.0 : *range.first;
            const double span = range.first == axes[axis]->end() ? 0.0 : *range.second - low[axis];
            scale[axis] = span > 0.0 ? static_cast<double>(kCurveCells - 1) / span : 0.0;
        }
        std::vector<uint64_t> key(chosen.size());
        parallel(chosen.size(), s.threads_, kLeast, [&](size_t, size_t begin, size_t end) {
            for (size_t n = begin; n < end; ++n)
            {
                const size_t i = chosen[n];
                uint64_t code = 0;
                for (int axis = 0; axis < 3; ++axis)
                    code |= interleave(static_cast<uint32_t>(((*axes[axis])[i] - low[axis]) * scale[axis])) << axis;
                key[n] = (static_cast<uint64_t>(s.decay_of_[i]) << 32) | code;
            }
        });
        std::vector<uint32_t> order(chosen.size());
        std::iota(order.begin(), order.end(), 0u);
        std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return key[a] < key[b]; });
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
        slots.turned = turned();
        Encoding along[3];
        for (int axis = 0; axis < 3; ++axis)
            along[axis] = !encoded_[axis] ? Encoding::none
                : lattice_[axis].tabulated ? Encoding::tabulated
                                           : Encoding::computed;
        slots.allocate(order, fastest_carry<Real>(slots.offsets).lanes, along, divided_ && tolerance_ > 0.0);
        const size_t lanes = slots.lanes;
        /* Whole packs apiece, so that no two workers write one pack. */
        parallel(slots.packs(), s.threads_, std::max<size_t>(1, kLeast / lanes), [&](size_t, size_t begin, size_t end) {
            std::vector<double> weights(Nufft::width_of());
            for (size_t n = begin * lanes; n < std::min(order.size(), end * lanes); ++n)
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
        for (size_t k = 0; k < 3; ++k)
            slots.value(n, k) = static_cast<Real>(m_[3 * i + k]);
        for (size_t k = 0; k < 9; ++k)
            slots.value(n, 3 + k) = static_cast<Real>(a_[9 * i + k]);
        for (size_t k = 0; k < 3 && slots.offsets; ++k)
            slots.value(n, 12 + k) = static_cast<Real>(b_[3 * i + k]);
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
        for (int axis = 0; axis < 3 && slots.turned; ++axis)
            slots.place[3 * n + axis] = (*coordinates[axis])[i];
        const double limit = tolerance_ * p.proton_density[i];
        if (slots.limits)
            slots.value(n, slots.limit_at) = static_cast<Real>(limit * limit);
    }

    template <typename SlotsOf>
    void Repetitions::fill_window(SlotsOf& slots, size_t w, size_t n, size_t i, double* weights) const
    {
        using Real = typename SlotsOf::Value;
        const Isochromats& s = isochromats_;
        const Window& window = windows_[w];
        const size_t W = windows_.size();
        const size_t u = slots.u_at + w * slots.u_width;
        for (size_t k = 0; k < 3; ++k)
        {
            slots.value(n, u + k) = static_cast<Real>(window.u[3 * i + k].real());
            slots.value(n, u + 3 + k) = static_cast<Real>(window.u[3 * i + k].imag());
        }
        if (slots.offsets)
        {
            slots.value(n, u + 6) = static_cast<Real>(window.v[i].real());
            slots.value(n, u + 7) = static_cast<Real>(window.v[i].imag());
        }
        std::complex<double> shift(1.0, 0.0);
        if (slots.turned)
            slots.origin[n * W + w] = window_phase(window, i);
        if (transforms_[w] != nullptr)
        {
            slots.start[n * W + w] =
                static_cast<uint32_t>(transforms_[w]->spread(window_phase(window, i), weights, shift));
            for (size_t j = 0; j < slots.taps; ++j)
                slots.weight[(n * W + w) * slots.taps + j] = static_cast<Real>(weights[j]);
        }
        const bool sensitivities = !s.receive_re_.empty();
        for (size_t k = 0; k < slots.coils; ++k)
        {
            const std::complex<double> receive = sensitivities
                ? std::complex<double>(s.receive_re_[k * s.count_ + i], s.receive_im_[k * s.count_ + i])
                : std::complex<double>(1.0, 0.0);
            const std::complex<double> factor = receive * shift;
            slots.factor[((n * W + w) * slots.coils + k) * 2] = static_cast<Real>(factor.real());
            slots.factor[((n * W + w) * slots.coils + k) * 2 + 1] = static_cast<Real>(factor.imag());
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
        AlignedVector<Real>& grids = set_->template grids<Real>();
        AlignedVector<Real>& scratch = set_->template scratch<Real>();
        grids.resize(threads * tile.region.back());
        scratch.resize(threads * scratch_per_worker(tile, slots.lanes));
        tile.grids = grids.data();
        const CarryRange<Real> range = fastest_carry<Real>(slots.offsets).range;
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
                else if (tile.turned[w])
                    read_turned(tile, w, threads, partial, out);
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
        using Real = typename TileOf::Value;
        const Isochromats& s = isochromats_;
        const size_t W = windows_.size();
        tile.windows = W;
        tile.coils = s.coils_;
        tile.taps = taps;
        tile.classes = s.decays_.size();
        tile.drop = divided_ && tolerance_ > 0.0;
        tile.length = samples_;
        tile.netted = !nets_.empty();
        /* The windows' encodings, then the net areas' as one more set. */
        const size_t sets = W + (tile.netted ? 1 : 0);
        tile.encoding.assign(sets * 3, Encoding::none);
        tile.table_at.assign(sets * 3, kNone);
        tile.table_re.assign(sets * 3, nullptr);
        tile.table_im.assign(sets * 3, nullptr);
        tile.angle.assign(sets * 3 * kTile, 0.0);
        size_t tables = 0;
        for (size_t at = 0; at < sets * 3; ++at)
            if (encoded_[at % 3] && lattice_[at % 3].tabulated)
            {
                tile.table_at[at] = tables;
                tables += 2 * lattice_[at % 3].values.size() * kTile;
            }
        tile.tables.assign(tables, 0);
        tile.region.assign(1, 0);
        tile.turned.clear();
        tile.delta.assign(W * kTile * 3, 0.0);
        for (size_t w = 0; w < W; ++w)
        {
            const Nufft* transform = transforms_[w].get();
            tile.turned.push_back(windows_[w].turned ? 1 : 0);
            const size_t cells = transform != nullptr ? transform->grid() : 0;
            const size_t points = cells > 0 ? tile.classes * tile.coils * (cells + taps) : tile.coils;
            tile.cells.push_back(cells);
            tile.samples.push_back(windows_[w].samples);
            tile.offset.push_back(windows_[w].offset);
            tile.transform.push_back(transform);
            tile.decay.push_back(decays(windows_[w]));
            tile.region.push_back(tile.region.back() + points * 2 * kTile);
        }
        tile.powers = 0;
        for (size_t w = 0; w < W; ++w)
            if (tile.turned[w])
                tile.powers = tile.transform[w]->degree() + 1;
        tile.polynomials.assign(W * tile.powers * taps, Real(0));
        for (size_t w = 0; w < W; ++w)
            if (tile.turned[w])
                std::transform(
                    tile.transform[w]->polynomials().begin(),
                    tile.transform[w]->polynomials().end(),
                    tile.polynomials.begin() + static_cast<std::ptrdiff_t>(w * tile.powers * taps),
                    [](double c) { return static_cast<Real>(c); });
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
        const size_t W = windows_.size();
        for (size_t at = 0; at < W * 3 * kTile; ++at)
        {
            const size_t w = at / (3 * kTile);
            const size_t r = at % kTile;
            tile.delta[at] = r < tile.count ? readouts_[((first + r) * W + w) * 3 + at / kTile % 3] : 0.0;
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
        size_t offsets[kTile];
        for (size_t at = 0; at < tile.encoding.size(); ++at)
        {
            const int axis = static_cast<int>(at % 3);
            bool any = false;
            for (size_t r = 0; r < kTile; ++r)
            {
                const double area = r < tile.count ? encoding_area(first + r, at) : 0.0;
                tile.angle[at * kTile + r] = -kTwoPi * area;
                any = any || area != 0.0;
            }
            tile.encoding[at] = !any ? Encoding::none
                : lattice_[axis].tabulated ? Encoding::tabulated
                                           : Encoding::computed;
            if (tile.encoding[at] != Encoding::tabulated)
                continue;
            const std::vector<double>& values = lattice_[axis].values;
            for (size_t r = 0; r < kTile; ++r)
            {
                const double area = r < tile.count ? encoding_area(first + r, at) : 0.0;
                offsets[r] = area != 0.0 ? tables.find(axis, area, values) : kNone;
            }
            Real* re = &tile.tables[tile.table_at[at]];
            Real* im = re + values.size() * kTile;
            tabulate_phases(tables, offsets, values.size(), re, im);
            tile.table_re[at] = re;
            tile.table_im[at] = im;
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
        if (next_ >= phases_.size() || turned() || !nets_.empty())
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
                double x = slots.value(slot, 0), y = slots.value(slot, 1), z = slots.value(slot, 2);
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
