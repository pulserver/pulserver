/**
 * @file moments.cpp
 * @brief The b-tensor and gradient moments of every shot.  See moments.hpp.
 *
 * One forward pass over the blocks, carrying k in two halves -- what the
 * console's prescription will turn, and what `NOROT` exempts from it -- and
 * integrating the quadratic forms of both, plus their cross term, over each
 * shot's excitation-to-echo window.
 *
 * A note on cost, because the shape of this invites a memo that is not worth
 * writing: the pass is already linear in the number of gradient breakpoints,
 * because each shot's integral touches only its own.  Computing a canonical
 * TR once and reusing it through per-instance rotations and amplitudes costs
 * the same and assumes shots share a template, which a free-waveform or
 * variable-b design breaks.  What that memo was really wanted for -- a short
 * table instead of one tensor per shot -- is done at the end, on the computed
 * values, where it needs no such assumption.
 */

#include "pulseq/moments.hpp"

#include "pulseq/kspace.hpp"
#include "pulseq/sequence.hpp"
#include "waveform.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <vector>

namespace pulseq
{

    namespace
    {
        using detail::Piecewise;
        using detail::unit_waveform_analysis;

        constexpr double TWO_PI = 6.283185307179586476925286766559;

        /* Times that come from different arithmetic still mean the same
         * instant; a block boundary and a pulse centre landing on it must not
         * become two nodes a picosecond apart. */
        constexpr double TIME_EPS = 1e-12;

        const Matrix3 IDENTITY{{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};

        /**
         * (w, x, y, z) to a row-major matrix.
         *
         * The same construction as `pulseg__quaternion_to_matrix()` and as
         * `KTrajPlan::build_rotations()` in ktraj.cpp -- normalised first,
         * degenerate rows falling back to the identity.  Kept literally in
         * step with them: two rotation conventions in one repository is a
         * silent wrong-direction bug, and `tests/cpptests/test_moments.cpp`
         * pins this against the k the core produces for the same block.
         */
        Matrix3 quaternion_to_matrix(const double* q)
        {
            double w = q[0], x = q[1], y = q[2], z = q[3];
            const double norm = std::sqrt(w * w + x * x + y * y + z * z);
            if (!(norm > 1.0e-9))
                return IDENTITY;

            const double inv = 1.0 / norm;
            w *= inv;
            x *= inv;
            y *= inv;
            z *= inv;

            const double xx = x * x, yy = y * y, zz = z * z;
            const double xy = x * y, xz = x * z, yz = y * z;
            const double wx = w * x, wy = w * y, wz = w * z;

            return Matrix3{
                {1.0 - 2.0 * (yy + zz),
                 2.0 * (xy - wz),
                 2.0 * (xz + wy),
                 2.0 * (xy + wz),
                 1.0 - 2.0 * (xx + zz),
                 2.0 * (yz - wx),
                 2.0 * (xz - wy),
                 2.0 * (yz + wx),
                 1.0 - 2.0 * (xx + yy)}};
        }

        /**
         * The waveform's two endpoint values over `[t0, t1]`, zero when the
         * piece lies outside its extent.
         *
         * Decided on the midpoint, and it has to be: `value_at` holds the end
         * values outside the waveform, so asking it at exactly the last
         * breakpoint of a gradient that ends mid-block would report the final
         * sample and integrate a lobe that is not played.  Every breakpoint is
         * a node of the grid, so a piece is wholly inside or wholly outside
         * and the midpoint answers for all of it.
         */
        void piece_values(const Piecewise& p, double t0, double t1, double* lo, double* hi)
        {
            *lo = 0.0;
            *hi = 0.0;
            if (p.empty())
                return;
            const double mid = 0.5 * (t0 + t1);
            if (mid < p.t.front() || mid > p.t.back())
                return;
            *lo = p.value_at(t0);
            *hi = p.value_at(t1);
        }

        /** What a block does to k and to the prescription. */
        struct BlockState
        {
            Piecewise wave[3];
            double amplitude[3] = {0.0, 0.0, 0.0};
            Matrix3 rotation = IDENTITY;
            bool rotated = false;
        };

        enum class Marker
        {
            Excitation,
            Refocusing,
            Echo
        };

        struct Event
        {
            double t;
            Marker kind;
        };

        /**
         * The instants that break k, in playing order.
         *
         * All three come from the trajectory the core already computed: the
         * excitation and refocusing centres it inventoried, and the echo,
         * which is the sample of each readout it found nearest k-space zero.
         */
        std::vector<Event> gather_events(const KSpace& ks)
        {
            std::vector<Event> events;
            events.reserve(ks.excitations.size() + ks.refocusings.size() + ks.readouts.size());

            for (const RfEventTiming& rf : ks.excitations)
                events.push_back({rf.t, Marker::Excitation});
            for (const RfEventTiming& rf : ks.refocusings)
                events.push_back({rf.t, Marker::Refocusing});
            for (const Readout& ro : ks.readouts)
            {
                if (ro.center_sample >= 0 && ro.center_sample < ro.num_samples)
                    events.push_back({ro.t0 + ro.center_sample * ro.dwell, Marker::Echo});
            }

            std::stable_sort(
                events.begin(),
                events.end(),
                [](const Event& a, const Event& b) { return a.t < b.t; });
            return events;
        }

        /** One shot: when it starts, and when its echo closes it. */
        struct Window
        {
            double start = 0.0;
            double echo = std::numeric_limits<double>::quiet_NaN();
        };

        /**
         * Each excitation's integration window.
         *
         * The echo is the first measured echo centre in the interval.  Only
         * when there is none -- a design being checked before its readout is
         * attached -- does this fall back to MATLAB's `2*t_refocusing -
         * t_excitation`, which is the formula `calcMomentsBtensor` carries its
         * own `TODO: fixme for double-refocused sequences` beside.  The
         * fallback is taken only when it lands inside the shot's own
         * interval: a formula that puts the echo after the next excitation,
         * or after the sequence ends, has not found an echo -- it has
         * produced a number, and a NaN the caller can see is worth more.
         */
        std::vector<Window> shot_windows(const std::vector<Event>& events, double sequence_end)
        {
            std::vector<Window> windows;
            for (size_t i = 0; i < events.size(); ++i)
            {
                if (events[i].kind != Marker::Excitation)
                    continue;

                Window w;
                w.start = events[i].t;

                const double limit = [&]
                {
                    for (size_t j = i + 1; j < events.size(); ++j)
                    {
                        if (events[j].kind == Marker::Excitation)
                            return events[j].t;
                    }
                    return sequence_end;
                }();

                double refocusing = std::numeric_limits<double>::quiet_NaN();
                for (size_t j = i + 1; j < events.size() && events[j].t <= limit; ++j)
                {
                    if (events[j].kind == Marker::Echo)
                    {
                        w.echo = events[j].t;
                        break;
                    }
                    if (events[j].kind == Marker::Refocusing && !(refocusing == refocusing))
                        refocusing = events[j].t;
                }
                if (!(w.echo == w.echo) && refocusing == refocusing)
                {
                    const double candidate = 2.0 * refocusing - w.start;
                    if (candidate > w.start && candidate <= limit)
                        w.echo = candidate;
                }

                windows.push_back(w);
            }
            return windows;
        }

        /**
         * `integral(x_i y_j)` over a piece, where both are quadratic in the
         * local time.
         *
         * `x` is `xc[0] + xc[1] t + xc[2] t^2` per axis, and likewise `y`, so
         * the product is a quartic whose integral is a sum of nine terms --
         * exact, and independent of any step size.  This is the whole reason
         * the answer does not depend on a raster.
         */
        void accumulate_product(
            Matrix3& into,
            const double xc[3][3],
            const double yc[3][3],
            double width)
        {
            double power[5];
            power[0] = width;
            for (int n = 1; n < 5; ++n)
                power[n] = power[n - 1] * width;

            /* integral of t^(p+q) over [0, w] == w^(p+q+1) / (p+q+1) */
            double weight[3][3];
            for (int p = 0; p < 3; ++p)
                for (int q = 0; q < 3; ++q)
                    weight[p][q] = power[p + q] / static_cast<double>(p + q + 1);

            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    double sum = 0.0;
                    for (int p = 0; p < 3; ++p)
                    {
                        for (int q = 0; q < 3; ++q)
                            sum += xc[i][p] * yc[j][q] * weight[p][q];
                    }
                    into[static_cast<size_t>(i * 3 + j)] += sum;
                }
            }
        }

        /**
         * `integral(g (offset + t)^order)` over a piece, per axis.
         *
         * `g` is `a + b t`, and `(offset + t)^order` binomially expanded, so
         * again every term is closed form.  `offset` is measured from the
         * excitation, which is where MATLAB measures a moment from.
         */
        void accumulate_moment(
            std::array<double, 3>& into,
            const double a[3],
            const double b[3],
            double offset,
            double width,
            int order)
        {
            static const double binomial[4][4] =
                {{1, 0, 0, 0}, {1, 1, 0, 0}, {1, 2, 1, 0}, {1, 3, 3, 1}};

            for (int p = 0; p <= order; ++p)
            {
                double lead = binomial[order][p];
                for (int n = 0; n < order - p; ++n)
                    lead *= offset;

                double w_a = lead;
                for (int n = 0; n <= p; ++n)
                    w_a *= width;
                w_a /= static_cast<double>(p + 1);

                double w_b = lead;
                for (int n = 0; n <= p + 1; ++n)
                    w_b *= width;
                w_b /= static_cast<double>(p + 2);

                for (int axis = 0; axis < 3; ++axis)
                    into[static_cast<size_t>(axis)] += a[axis] * w_a + b[axis] * w_b;
            }
        }

        /** Every block's waveforms, rotation and `NOROT` state, in range. */
        void resolve_blocks(
            const Sequence& seq,
            int first,
            int last,
            std::vector<BlockState>& states,
            std::vector<char>& norot)
        {
            const int rotation_type = seq.find_extension_type_id("ROTATIONS");
            const int labelset_type = seq.find_extension_type_id("LABELSET");
            const int labelinc_type = seq.find_extension_type_id("LABELINC");
            const int norot_label = seq.find_label_id("NOROT");

            const int32_t* events = seq.block_events();

            /*
             * `NOROT` is sticky, the way pulseg_dedup.c carries it: a LABELSET
             * holds until the next one, so a module that turns it on turns it
             * off again explicitly.  That means the state has to be built from
             * block 1 even when the window starts later -- reading a range in
             * isolation would lose whichever SET preceded it.
             */
            int current = 0;
            states.resize(static_cast<size_t>(last - first + 1));
            norot.resize(static_cast<size_t>(last - first + 1));

            for (int b = 1; b <= last; ++b)
            {
                const int32_t head = events[static_cast<size_t>(b - 1) * BLOCK_WIDTH + 5];
                Matrix3 rotation = IDENTITY;
                bool rotated = false;

                for (int32_t link = head; link != 0;)
                {
                    const int32_t* row = seq.extensions_library().row(link);
                    const int32_t type = row[0];
                    const int32_t ref = row[1];
                    if (ref != 0)
                    {
                        if (norot_label != 0 && labelset_type != 0 && type == labelset_type)
                        {
                            const int32_t* entry = seq.label_set_library().row(ref);
                            if (entry[1] == norot_label)
                                current = entry[0];
                        }
                        else if (norot_label != 0 && labelinc_type != 0 && type == labelinc_type)
                        {
                            const int32_t* entry = seq.label_inc_library().row(ref);
                            if (entry[1] == norot_label)
                                current += entry[0];
                        }
                        else if (rotation_type != 0 && type == rotation_type)
                        {
                            rotation = quaternion_to_matrix(seq.rotation_library().row(ref));
                            rotated = true;
                        }
                    }
                    link = row[2];
                }

                if (b < first)
                    continue;

                const size_t at = static_cast<size_t>(b - first);
                norot[at] = current != 0 ? 1 : 0;
                states[at].rotation = rotation;
                states[at].rotated = rotated;

                const int32_t* row = events + static_cast<size_t>(b - 1) * BLOCK_WIDTH;
                for (int a = 0; a < 3; ++a)
                    states[at].wave[a] =
                        unit_waveform_analysis(seq, row[1 + a], &states[at].amplitude[a]);
            }
        }

        /** The largest magnitude anywhere in @p parts, for the table's scale. */
        double peak_of(const std::vector<BTensorParts>& parts)
        {
            double peak = 0.0;
            for (const BTensorParts& p : parts)
            {
                for (int i = 0; i < 9; ++i)
                {
                    peak = std::max(peak, std::fabs(p.fixed[static_cast<size_t>(i)]));
                    peak = std::max(peak, std::fabs(p.rotatable[static_cast<size_t>(i)]));
                    peak = std::max(peak, std::fabs(p.cross[static_cast<size_t>(i)]));
                }
            }
            return peak > 0.0 ? peak : 1.0;
        }

        /**
         * A shot's three matrices, rounded so two of them compare equal.
         *
         * Rounded against the **scan's** largest element, not the shot's own.
         * Normalising each shot by its own peak would make the key
         * scale-invariant, and two directions that differ only in b-value --
         * which is most of a multi-shell acquisition -- would collapse onto
         * one row.
         */
        std::vector<int64_t> table_key(const BTensorParts& parts, double scale, int digits)
        {
            const double* blocks[3] = {
                parts.fixed.data(),
                parts.rotatable.data(),
                parts.cross.data()};

            double unit = 1.0;
            for (int i = 0; i < digits; ++i)
                unit *= 10.0;

            std::vector<int64_t> key;
            key.reserve(27);
            for (int block = 0; block < 3; ++block)
            {
                for (int i = 0; i < 9; ++i)
                    key.push_back(
                        static_cast<int64_t>(std::llround(blocks[block][i] / scale * unit)));
            }
            return key;
        }

    } // namespace

    const Matrix3& identity3()
    {
        return IDENTITY;
    }

    Matrix3 compose_btensor(const BTensorParts& parts, const Matrix3& rotation)
    {
        const double* R = rotation.data();

        /* T = R * B_rot * R^T, and X = C * R^T, both written out rather than
         * routed through a matrix type this library does not have. */
        double RB[9] = {0};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    RB[i * 3 + j] += R[i * 3 + k] * parts.rotatable[static_cast<size_t>(k * 3 + j)];

        Matrix3 out{};
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                double turned = 0.0;
                double crossed = 0.0;
                for (int k = 0; k < 3; ++k)
                {
                    turned += RB[i * 3 + k] * R[j * 3 + k];
                    /* C R^T contributes (i,j); its transpose contributes (j,i). */
                    crossed += parts.cross[static_cast<size_t>(i * 3 + k)] * R[j * 3 + k] +
                        R[i * 3 + k] * parts.cross[static_cast<size_t>(j * 3 + k)];
                }
                out[static_cast<size_t>(i * 3 + j)] =
                    parts.fixed[static_cast<size_t>(i * 3 + j)] + crossed + turned;
            }
        }
        return out;
    }

    Moments calc_moments(const Sequence& seq, const MomentsOptions& options)
    {
        Moments out;

        const int blocks = seq.num_blocks();
        const int first = std::max(1, options.first_block);
        const int last = options.last_block > 0 ? std::min(options.last_block, blocks) : blocks;
        if (blocks == 0 || first > last)
            return out;

        /* The echoes.  Reusing a KSpace the caller already has matters: the
         * bridge into the C core is a serialise and a parse, which is the
         * expensive half of this whole call on a small sequence. */
        KSpace owned;
        const KSpace* ks = options.kspace;
        if (ks == nullptr)
        {
            KSpaceOptions ko;
            ko.first_block = first;
            ko.last_block = last;
            ko.derive_center_sample = true;
            /* Nothing here reads a sample, and materialising them is what
             * makes a trajectory large. */
            ko.materialize_samples = false;
            owned = calculate_kspace(seq, ko);
            ks = &owned;
        }

        const double* durations = seq.block_durations();
        double window_end = 0.0;
        for (int b = first; b <= last; ++b)
            window_end += durations[b - 1];

        const std::vector<Event> events = gather_events(*ks);
        const std::vector<Window> windows = shot_windows(events, window_end);
        if (windows.empty())
            return out;

        std::vector<BlockState> states;
        std::vector<char> norot;
        resolve_blocks(seq, first, last, states, norot);

        const int orders =
            (options.calc_m1 ? 1 : 0) + (options.calc_m2 ? 1 : 0) + (options.calc_m3 ? 1 : 0);

        const size_t shots = windows.size();
        std::vector<BTensorParts> b_parts(shots);
        std::vector<std::array<double, 3>> moment[3];
        for (int n = 0; n < 3; ++n)
            moment[n].assign(shots, {{0.0, 0.0, 0.0}});

        /*
         * A fallback echo is not an event -- it is arithmetic on one -- so it
         * has to be added to the grid separately or a shot would end
         * mid-piece.  Sorted with an advancing index, because scanning every
         * window at every block would make this quadratic in the scan.
         */
        std::vector<double> extra_edges;
        for (const Window& w : windows)
        {
            if (w.echo == w.echo)
                extra_edges.push_back(w.echo);
        }
        std::sort(extra_edges.begin(), extra_edges.end());
        size_t next_edge = 0;

        /* k, split by what the console's prescription will turn. */
        double k_fixed[3] = {0.0, 0.0, 0.0};
        double k_rot[3] = {0.0, 0.0, 0.0};

        /*
         * The refocusing sign.  A refocusing pulse conjugates the phase, which
         * is the same as playing every gradient after it with the opposite
         * sign -- so rather than negating k at the pulse, the *effective*
         * gradient carries the flip and k is a plain integral that resets at
         * each excitation.  One mechanism instead of two, and it is what makes
         * the moments right: `B` is quadratic in k and cannot tell the two
         * apart, but a moment is linear in the gradient and very much can.
         */
        double sign = 1.0;

        size_t next_event = 0;
        size_t shot = 0;

        std::vector<double> nodes;
        double t_block = 0.0;
        for (int b = first; b <= last; ++b)
        {
            const size_t at = static_cast<size_t>(b - first);
            const BlockState& state = states[at];
            const bool is_fixed = norot[at] != 0;
            const double duration = durations[b - 1];
            const double t_end = t_block + duration;

            /* The grid: the block's own ends, every waveform breakpoint in it,
             * and every instant that breaks k or opens or closes a shot.  A
             * piece therefore lies wholly inside or wholly outside each
             * waveform, so a gradient that has finished is silent rather than
             * held at its last sample. */
            nodes.clear();
            nodes.push_back(t_block);
            nodes.push_back(t_end);
            for (int a = 0; a < 3; ++a)
            {
                for (double x : state.wave[a].t)
                {
                    const double abs = t_block + x;
                    if (abs > t_block && abs < t_end)
                        nodes.push_back(abs);
                }
            }
            for (size_t e = next_event; e < events.size() && events[e].t <= t_end + TIME_EPS; ++e)
            {
                if (events[e].t > t_block + TIME_EPS)
                    nodes.push_back(events[e].t);
            }
            while (next_edge < extra_edges.size() && extra_edges[next_edge] < t_block - TIME_EPS)
                ++next_edge;
            for (size_t e = next_edge; e < extra_edges.size() && extra_edges[e] <= t_end + TIME_EPS;
                 ++e)
            {
                if (extra_edges[e] > t_block + TIME_EPS)
                    nodes.push_back(extra_edges[e]);
            }
            std::sort(nodes.begin(), nodes.end());
            nodes.erase(
                std::unique(
                    nodes.begin(),
                    nodes.end(),
                    [](double x, double y) { return std::fabs(x - y) <= TIME_EPS; }),
                nodes.end());

            for (size_t n = 0; n + 1 < nodes.size(); ++n)
            {
                const double t0 = nodes[n];
                const double t1 = nodes[n + 1];

                /* Markers land on a node, and are applied before the piece
                 * that starts there so a crusher on either side of a
                 * refocusing pulse ends up on the correct side of the sign
                 * flip. */
                while (next_event < events.size() && events[next_event].t <= t0 + TIME_EPS)
                {
                    switch (events[next_event].kind)
                    {
                    case Marker::Excitation:
                        for (int a = 0; a < 3; ++a)
                        {
                            k_fixed[a] = 0.0;
                            k_rot[a] = 0.0;
                        }
                        sign = 1.0;
                        break;
                    case Marker::Refocusing:
                        sign = -sign;
                        break;
                    case Marker::Echo:
                        break;
                    }
                    ++next_event;
                }

                /*
                 * Which shot this piece belongs to.  Every window edge is a
                 * node, so a piece never straddles one and its start decides
                 * for all of it.  A shot is done with when its echo has been
                 * reached, or -- having no echo at all, so contributing
                 * nothing -- when the next shot has started.
                 */
                while (shot < shots)
                {
                    const Window& w = windows[shot];
                    const bool has_echo = w.echo == w.echo;
                    if (has_echo && t0 < w.echo - TIME_EPS)
                        break;
                    if (!has_echo && (shot + 1 >= shots || t0 < windows[shot + 1].start - TIME_EPS))
                        break;
                    ++shot;
                }
                const bool inside = shot < shots && windows[shot].echo == windows[shot].echo &&
                    windows[shot].echo > windows[shot].start &&
                    t0 >= windows[shot].start - TIME_EPS;

                const double width = t1 - t0;
                if (width <= 0.0)
                    continue;

                /* The gradient over this piece: linear, so its two endpoint
                 * values are the whole story. */
                double slope[3];
                double start[3];
                for (int a = 0; a < 3; ++a)
                {
                    double lo = 0.0;
                    double hi = 0.0;
                    piece_values(state.wave[a], t0 - t_block, t1 - t_block, &lo, &hi);
                    start[a] = state.amplitude[a] * lo;
                    slope[a] = (state.amplitude[a] * hi - start[a]) / width;
                }

                double g0[3] = {0.0, 0.0, 0.0};
                double g1[3] = {0.0, 0.0, 0.0};
                if (state.rotated)
                {
                    for (int i = 0; i < 3; ++i)
                    {
                        for (int j = 0; j < 3; ++j)
                        {
                            g0[i] += state.rotation[static_cast<size_t>(i * 3 + j)] * start[j];
                            g1[i] += state.rotation[static_cast<size_t>(i * 3 + j)] * slope[j];
                        }
                    }
                }
                else
                {
                    for (int a = 0; a < 3; ++a)
                    {
                        g0[a] = start[a];
                        g1[a] = slope[a];
                    }
                }
                for (int a = 0; a < 3; ++a)
                {
                    g0[a] *= sign;
                    g1[a] *= sign;
                }

                /* k over the piece, as coefficients of 1, t, t^2. */
                double cf[3][3];
                double cr[3][3];
                for (int a = 0; a < 3; ++a)
                {
                    cf[a][0] = k_fixed[a];
                    cr[a][0] = k_rot[a];
                    cf[a][1] = is_fixed ? g0[a] : 0.0;
                    cr[a][1] = is_fixed ? 0.0 : g0[a];
                    cf[a][2] = is_fixed ? 0.5 * g1[a] : 0.0;
                    cr[a][2] = is_fixed ? 0.0 : 0.5 * g1[a];
                }

                if (inside && options.calc_b)
                {
                    BTensorParts& parts = b_parts[shot];
                    accumulate_product(parts.fixed, cf, cf, width);
                    accumulate_product(parts.rotatable, cr, cr, width);
                    accumulate_product(parts.cross, cf, cr, width);
                }
                if (inside && orders > 0)
                {
                    const double offset = t0 - windows[shot].start;
                    if (options.calc_m1)
                        accumulate_moment(moment[0][shot], g0, g1, offset, width, 1);
                    if (options.calc_m2)
                        accumulate_moment(moment[1][shot], g0, g1, offset, width, 2);
                    if (options.calc_m3)
                        accumulate_moment(moment[2][shot], g0, g1, offset, width, 3);
                }

                for (int a = 0; a < 3; ++a)
                {
                    const double step = g0[a] * width + 0.5 * g1[a] * width * width;
                    if (is_fixed)
                        k_fixed[a] += step;
                    else
                        k_rot[a] += step;
                }
            }

            t_block = t_end;
        }

        /* Units.  k is in 1/m, so q == 2*pi*k is rad/m and B is s/m^2; a
         * moment is 2*pi times the gradient integral. */
        const double b_scale = TWO_PI * TWO_PI;
        for (size_t s = 0; s < shots; ++s)
        {
            for (int i = 0; i < 9; ++i)
            {
                b_parts[s].fixed[static_cast<size_t>(i)] *= b_scale;
                b_parts[s].rotatable[static_cast<size_t>(i)] *= b_scale;
                b_parts[s].cross[static_cast<size_t>(i)] *= b_scale;
            }
            for (int n = 0; n < 3; ++n)
                for (int a = 0; a < 3; ++a)
                    moment[n][s][static_cast<size_t>(a)] *= TWO_PI;
        }

        const size_t skip = std::min(static_cast<size_t>(std::max(0, options.n_dummy)), shots);
        const size_t kept = shots - skip;

        out.t_excitation.reserve(kept);
        out.t_echo.reserve(kept);
        for (size_t s = skip; s < shots; ++s)
        {
            out.t_excitation.push_back(windows[s].start);
            out.t_echo.push_back(windows[s].echo);
        }
        if (options.calc_b)
            out.b.assign(b_parts.begin() + static_cast<long>(skip), b_parts.end());
        if (options.calc_m1)
            out.m1.assign(moment[0].begin() + static_cast<long>(skip), moment[0].end());
        if (options.calc_m2)
            out.m2.assign(moment[1].begin() + static_cast<long>(skip), moment[1].end());
        if (options.calc_m3)
            out.m3.assign(moment[2].begin() + static_cast<long>(skip), moment[2].end());

        out.table_index.assign(kept, -1);
        if (options.calc_b)
        {
            std::map<std::vector<int64_t>, int> seen;
            const double scale = peak_of(out.b);
            for (size_t s = 0; s < kept; ++s)
            {
                const std::vector<int64_t> key = table_key(out.b[s], scale, options.table_digits);
                auto it = seen.find(key);
                if (it == seen.end())
                {
                    it = seen.emplace(key, static_cast<int>(out.table.size())).first;
                    out.table.push_back(out.b[s]);
                }
                out.table_index[s] = it->second;
            }
        }

        return out;
    }

} // namespace pulseq
