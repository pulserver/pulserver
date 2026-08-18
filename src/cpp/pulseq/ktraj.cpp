/**
 * @file ktraj.cpp
 * @brief Absolute k at every ADC sample, and where each readout's echo is.
 *
 * See ktraj.hpp for what this is for and why it is not a cumulative sum over a
 * materialised waveform.  The shape of the implementation is:
 *
 *   1. collapse the gradient and ADC libraries onto their distinct rows
 *      (`canonicalise_events`), and integrate each distinct gradient once, at
 *      unit amplitude, into a piecewise-linear breakpoint list with a running
 *      integral (`build_waves`, over `detail::Piecewise` from waveform.hpp);
 *   2. walk the blocks once, accumulating k and recording where each block
 *      starts (`walk_blocks`);
 *   3. per readout, emit `origin + amplitude * cumulative(t)` per axis, then
 *      rotate (`samples`);
 *   4. two memoized sweeps to derive `center_sample` (`derive_centers`).
 *
 * Every accumulation is `double`: k is a running integral over an entire scan,
 * so a narrower accumulator drifts in a way a narrower *result* does not.
 */

#include "ktraj.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace pulseq
{
    namespace detail
    {
        namespace
        {
            /** [RF]: ampl mag phase time center delay freqPPM phasePPM freq phase */
            constexpr int RF_CENTER = 4;
            constexpr int RF_DELAY = 5;
            constexpr int RF_FREQ_PPM = 6;
            constexpr int RF_PHASE_PPM = 7;
            constexpr int RF_FREQ = 8;
            constexpr int RF_PHASE = 9;

            /** [ADC]: num dwell delay freqPPM phasePPM freq phase phase_shape */
            constexpr int ADC_NUM = 0;
            constexpr int ADC_DWELL = 1;
            constexpr int ADC_DELAY = 2;

            /** Block table columns: rf, gx, gy, gz, adc, ext. */
            constexpr int BLOCK_RF = 0;
            constexpr int BLOCK_GX = 1;
            constexpr int BLOCK_ADC = 4;

            constexpr double TWO_PI = 6.283185307179586476925286766559;

            /**
             * What a pulse does to k, or Undeclared if the sequence does not
             * say.
             *
             * "Excitation" resets k, "refocusing" negates it, and inversion,
             * saturation, preparation and other leave it alone.
             */
            enum class RfRole
            {
                Neutral,
                Excitation,
                Refocusing,
                Undeclared
            };

            RfRole rf_role(char use)
            {
                switch (use)
                {
                case 'e':
                    return RfRole::Excitation;
                case 'r':
                    return RfRole::Refocusing;
                case '\0':
                case 'u':
                    return RfRole::Undeclared;
                default:
                    return RfRole::Neutral;
                }
            }

            /**
             * B0 for the ppm terms on RF rows.
             *
             * gamma * B0 is a system property, not a sequence one, so it can
             * only arrive through `[DEFINITIONS]`.  When it does not, the ppm
             * contribution is taken as zero -- which is exact for every file
             * that leaves those columns at 0, i.e. everything written before
             * Pulseq 1.5 and most since.
             */
            double gamma_b0(const Sequence& seq)
            {
                const Definition* b0 = seq.definition("B0");
                if (b0 == nullptr)
                    return 0.0;

                double tesla = 0.0;
                if (!b0->numbers().empty())
                    tesla = b0->numbers()[0];
                else if (!b0->text().empty())
                    tesla = std::atof(b0->text().c_str());

                /* Pulseq writes B0 in tesla; gamma is the proton value, Hz/T. */
                return tesla * 42576000.0;
            }

        } // namespace

        /* ============================================================== */
        /*  The repeat key                                                */
        /* ============================================================== */

        bool KTrajPlan::RepeatKey::operator==(const RepeatKey& other) const
        {
            return grad == other.grad && adc == other.adc && rotation == other.rotation &&
                swept == other.swept && k0[0] == other.k0[0] && k0[1] == other.k0[1] &&
                k0[2] == other.k0[2];
        }

        size_t KTrajPlan::RepeatKeyHash::operator()(const RepeatKey& key) const
        {
            /*
             * Hashing the bytes of the doubles is sound only because the
             * comparison above is exact too: equal doubles have equal bytes,
             * apart from the one pair that does not -- +0.0 and -0.0 -- which
             * hash_double folds together.  A NaN would hash fine and never
             * compare equal, costing a missed hit and nothing else.
             */
            size_t h = HASH_SEED;
            hash_int(h, key.grad[0]);
            hash_int(h, key.grad[1]);
            hash_int(h, key.grad[2]);
            hash_int(h, key.adc);
            hash_int(h, key.rotation);
            hash_int(h, key.swept);
            for (int a = 0; a < 3; ++a)
                hash_double(h, key.k0[static_cast<size_t>(a)]);
            return h;
        }

        /* ============================================================== */
        /*  Waves                                                         */
        /* ============================================================== */

        const KTrajPlan::Wave* KTrajPlan::wave_for(int grad_id) const
        {
            if (grad_id <= 0 || grad_id >= static_cast<int>(grad_wave_.size()))
                return nullptr;
            const int row = grad_wave_[static_cast<size_t>(grad_id)];
            if (row < 0)
                return nullptr;
            const Wave& w = waves_[static_cast<size_t>(row)];
            return w.built ? &w : nullptr;
        }

        namespace
        {
            /**
             * Build the second antiderivative, once, on demand.
             *
             * Exact rather than approximated: on a segment the waveform is
             * linear, so its integral is quadratic and the integral of that is
             * cubic, and a cubic has a closed form.  Approximating here would
             * defeat the purpose -- the whole correction being computed is
             * smaller than one sample step.
             */
            std::vector<double> build_second(const Piecewise& p)
            {
                std::vector<double> c2(p.t.size(), 0.0);
                for (size_t i = 1; i < p.t.size(); ++i)
                {
                    const double span = p.t[i] - p.t[i - 1];
                    const double slope = (span > 0.0) ? (p.v[i] - p.v[i - 1]) / span : 0.0;
                    c2[i] = c2[i - 1] + p.c[i - 1] * span + p.v[i - 1] * span * span / 2.0 +
                        slope * span * span * span / 6.0;
                }
                return c2;
            }

            /** The integral of the cumulative, from `t[0]` to @p x. */
            double second_at(const Piecewise& p, const std::vector<double>& c2, double x)
            {
                if (p.empty() || c2.empty())
                    return 0.0;
                if (x <= p.t.front())
                    return 0.0;
                if (x >= p.t.back())
                {
                    /* Past the end the waveform is silent, so the cumulative
                     * holds at its total and the integral of it grows
                     * linearly. */
                    return c2.back() + p.c.back() * (x - p.t.back());
                }

                const size_t hi =
                    static_cast<size_t>(std::upper_bound(p.t.begin(), p.t.end(), x) - p.t.begin());
                const size_t lo = hi - 1;
                const double span = p.t[hi] - p.t[lo];
                const double slope = (span > 0.0) ? (p.v[hi] - p.v[lo]) / span : 0.0;
                const double u = x - p.t[lo];
                return c2[lo] + p.c[lo] * u + p.v[lo] * u * u / 2.0 + slope * u * u * u / 6.0;
            }
        } // namespace

        /**
         * The cumulative averaged over one dwell window centred on @p x.
         *
         * An ADC sample is not an instant: it integrates signal for a whole
         * dwell, so the coordinate it belongs to is k averaged across that
         * window rather than k at its midpoint.  For a smooth gradient the
         * difference is `(dwell^2 / 24) * dg/dt` -- identically zero while the
         * gradient is flat, and a real systematic offset on a ramp-sampled
         * readout.
         */
        namespace
        {
            double mean_cumulative(
                const Piecewise& p,
                const std::vector<double>& c2,
                double x,
                double dwell)
            {
                if (p.empty() || c2.empty() || !(dwell > 0.0))
                    return p.cumulative_at(x);
                return (second_at(p, c2, x + 0.5 * dwell) - second_at(p, c2, x - 0.5 * dwell)) /
                    dwell;
            }
        } // namespace

        /*
         * Integrate each distinct gradient row once.
         *
         * `grad_canon_` has already collapsed the ids that mean the same
         * thing, so only a representative is integrated and every id that
         * shares its row points at the same wave.
         */
        void KTrajPlan::build_waves()
        {
            const int count = seq_.num_gradients();
            grad_wave_.assign(static_cast<size_t>(count) + 1, -1);

            for (int id = 1; id <= count; ++id)
            {
                const int canonical = grad_canon_[static_cast<size_t>(id)];
                if (canonical != id)
                {
                    grad_wave_[static_cast<size_t>(id)] =
                        grad_wave_[static_cast<size_t>(canonical)];
                    continue;
                }

                Wave w;
                w.p = unit_waveform(seq_, id, &w.amplitude);
                /* A row with a broken shape id is not an error until something
                 * asks for it -- leave it unbuilt and let the block walk treat
                 * it as absent. */
                w.built = !w.p.empty();
                if (w.built && opts_.sample_window_average)
                    w.c2 = build_second(w.p);

                grad_wave_[static_cast<size_t>(id)] = static_cast<int>(waves_.size());
                waves_.push_back(std::move(w));
            }
        }

        /* ============================================================== */
        /*  Rotations                                                     */
        /* ============================================================== */

        /*
         * Convert every rotation quaternion to a matrix.
         *
         * The same construction as `pulseg__quaternion_to_matrix()` and as
         * `quaternion_to_matrix()` in moments.cpp: (w, x, y, z), normalised
         * first, degenerate rows falling back to the identity.  Kept literally
         * in step with them -- two rotation conventions in one repository
         * would be a silent wrong-image bug.
         */
        void KTrajPlan::build_rotations()
        {
            /* Resolved once: the name-to-id lookup is a map keyed by string,
             * and the chain walk that needs it runs per readout. */
            rotation_type_ = seq_.find_extension_type_id("ROTATIONS");

            num_rot_ = seq_.rotation_library().size();
            if (num_rot_ <= 0)
            {
                num_rot_ = 0;
                return;
            }

            rot_.resize(static_cast<size_t>(num_rot_) * 9);
            for (int i = 0; i < num_rot_; ++i)
            {
                const double* q = seq_.rotation_library().row(i + 1);
                double* m = rot_.data() + static_cast<size_t>(i) * 9;
                double w = q[0], x = q[1], y = q[2], z = q[3];
                const double norm = std::sqrt(w * w + x * x + y * y + z * z);

                if (!(norm > 1.0e-9))
                {
                    m[0] = 1.0;
                    m[1] = 0.0;
                    m[2] = 0.0;
                    m[3] = 0.0;
                    m[4] = 1.0;
                    m[5] = 0.0;
                    m[6] = 0.0;
                    m[7] = 0.0;
                    m[8] = 1.0;
                    continue;
                }

                const double inv = 1.0 / norm;
                w *= inv;
                x *= inv;
                y *= inv;
                z *= inv;

                const double xx = x * x, yy = y * y, zz = z * z;
                const double xy = x * y, xz = x * z, yz = y * z;
                const double wx = w * x, wy = w * y, wz = w * z;

                m[0] = 1.0 - 2.0 * (yy + zz);
                m[1] = 2.0 * (xy - wz);
                m[2] = 2.0 * (xz + wy);
                m[3] = 2.0 * (xy + wz);
                m[4] = 1.0 - 2.0 * (xx + zz);
                m[5] = 2.0 * (yz - wx);
                m[6] = 2.0 * (xz - wy);
                m[7] = 2.0 * (yz + wx);
                m[8] = 1.0 - 2.0 * (xx + yy);
            }
        }

        const double* KTrajPlan::rotation_for(int rotation_id) const
        {
            if (rotation_id < 0 || rotation_id >= num_rot_)
                return nullptr;
            return rot_.data() + static_cast<size_t>(rotation_id) * 9;
        }

        void KTrajPlan::rotate(const double* m, double* xyz) const
        {
            const double x = xyz[0], y = xyz[1], z = xyz[2];
            xyz[0] = m[0] * x + m[1] * y + m[2] * z;
            xyz[1] = m[3] * x + m[4] * y + m[5] * z;
            xyz[2] = m[6] * x + m[7] * y + m[8] * z;
        }

        /*
         * The 0-based rotation id block @p block_index (1-based) carries, or
         * -1.  Label, trigger and shim links on the same chain are walked past.
         */
        int KTrajPlan::block_rotation_id(int block_index) const
        {
            if (rotation_type_ == 0 || num_rot_ <= 0)
                return -1;

            const int32_t head =
                seq_.block_events()[static_cast<size_t>(block_index - 1) * BLOCK_WIDTH + 5];
            const int32_t links = seq_.extensions_library().size();
            int found = -1;
            for (int32_t link = head; link > 0 && link <= links;)
            {
                const int32_t* row = seq_.extensions_library().row(link);
                if (row[0] == rotation_type_ && row[1] > 0 && row[1] <= num_rot_)
                    found = row[1] - 1;
                link = row[2];
            }
            return found;
        }

        /* ============================================================== */
        /*  Canonical event ids                                           */
        /* ============================================================== */

        void KTrajPlan::canonicalise_events()
        {
            /*
             * Every column of a gradient row matters, and so does which table
             * it lives in: a trapezoid and an arbitrary gradient numbered into
             * the same id space are never the same event.
             *
             * The rows are hashed and compared where they lie rather than
             * copied into keys.  A scan built without deduplication holds one
             * gradient row per block per axis -- four million of them at
             * protocol scale -- and materialising a key for each would cost
             * more than the whole pass.
             */
            const int num_grads = seq_.num_gradients();
            const auto grad_row_of = [this](int id, int* width) -> const double*
            {
                switch (seq_.grad_kind(id))
                {
                case GradKind::Trap:
                    *width = TRAP_WIDTH;
                    return seq_.trap_library().row(seq_.grad_row(id));
                case GradKind::Arbitrary:
                    *width = ARB_WIDTH;
                    return seq_.arb_library().row(seq_.grad_row(id));
                default:
                    *width = 0;
                    return nullptr;
                }
            };

            grad_canon_ = canonical_ids(
                num_grads,
                [&](int id)
                {
                    int width = 0;
                    const double* row = grad_row_of(id, &width);
                    size_t h = HASH_SEED;
                    hash_int(h, width);
                    for (int c = 0; c < width; ++c)
                        hash_double(h, row[c]);
                    return h;
                },
                [&](int a, int b)
                {
                    int wa = 0, wb = 0;
                    const double* ra = grad_row_of(a, &wa);
                    const double* rb = grad_row_of(b, &wb);
                    if (wa != wb)
                        return false;
                    for (int c = 0; c < wa; ++c)
                        if (ra[c] != rb[c])
                            return false;
                    return true;
                });

            /* For an ADC only the three columns that decide when the samples
             * land: its frequency and phase offsets are receiver settings that
             * move no gradient, yet RF spoiling gives every line a different
             * phase and so a different id. */
            adc_canon_ = canonical_ids(
                seq_.adc_library().size(),
                [this](int id)
                {
                    const double* row = seq_.adc_library().row(id);
                    size_t h = HASH_SEED;
                    for (int c = 0; c < 3; ++c)
                        hash_double(h, row[c]);
                    return h;
                },
                [this](int a, int b)
                {
                    const double* ra = seq_.adc_library().row(a);
                    const double* rb = seq_.adc_library().row(b);
                    return ra[0] == rb[0] && ra[1] == rb[1] && ra[2] == rb[2];
                });
        }

        /* ============================================================== */
        /*  The block walk                                                */
        /* ============================================================== */

        /*
         * One pass over the blocks: accumulate k, record origins and start
         * times, and inventory the ADC and RF events.
         *
         * A block whose RF resets or reverses k is split at the pulse centre,
         * so crushers on either side of a refocusing pulse -- the ordinary
         * case -- land on the correct side of the sign flip.
         */
        void KTrajPlan::walk_blocks()
        {
            const int32_t* events = seq_.block_events();
            const double* durations = seq_.block_durations();
            const int rf_rows = seq_.rf_library().size();
            const int adc_rows = seq_.adc_library().size();
            const double gb0 = gamma_b0(seq_);

            /* Pass A: count, so the arrays are exact rather than grown. */
            int num_adc = 0, num_exc = 0, num_ref = 0;
            for (int b = first_block_; b <= last_block_; ++b)
            {
                const int32_t* row = events + static_cast<size_t>(b) * BLOCK_WIDTH;
                if (row[BLOCK_ADC] > 0 && row[BLOCK_ADC] <= adc_rows)
                    num_adc++;
                if (row[BLOCK_RF] > 0 && row[BLOCK_RF] <= rf_rows)
                {
                    const RfRole role =
                        rf_role(seq_.rf_uses()[static_cast<size_t>(row[BLOCK_RF]) - 1]);
                    /*
                     * An undeclared use is refused rather than guessed at, and
                     * refused here so the failure comes before a single
                     * allocation.  PyPulseq and Pulseq both read one as an
                     * excitation, and that guess is load-bearing: an
                     * excitation resets k to zero, so getting it wrong means k
                     * accumulates across the whole scan instead of restarting
                     * each TR.  Measured on a 32-line GRE whose pulses carried
                     * no use: the origin of the last block came out 2.8e4 1/m
                     * against a k-max of the same size.  A number that wrong,
                     * arrived at silently, is worse than no number -- and the
                     * caller who wrote the sequence is the one who knows what
                     * the pulse is for.
                     */
                    if (role == RfRole::Undeclared)
                        throw std::runtime_error(
                            "calculate_kspace: the RF pulse in block " + std::to_string(b + 1) +
                            " declares no use, so what it does to k is unknown");
                    if (role == RfRole::Excitation)
                        num_exc++;
                    else if (role == RfRole::Refocusing)
                        num_ref++;
                }
            }

            readouts_.resize(static_cast<size_t>(num_adc));
            keys_.resize(static_cast<size_t>(num_adc));
            adc_offset_.resize(static_cast<size_t>(num_adc));
            adc_dwell_.resize(static_cast<size_t>(num_adc));
            excitations_.resize(static_cast<size_t>(num_exc));
            refocusings_.resize(static_cast<size_t>(num_ref));

            origin_.assign(static_cast<size_t>(num_range_blocks_) * 3, 0.0);
            t_block_.assign(static_cast<size_t>(num_range_blocks_), 0.0);

            /* Pass B: the walk itself. */
            double k[3] = {0.0, 0.0, 0.0};
            double t = 0.0;
            int i_adc = 0, i_exc = 0, i_ref = 0;

            for (int b = first_block_; b <= last_block_; ++b)
            {
                const int32_t* row = events + static_cast<size_t>(b) * BLOCK_WIDTH;
                const size_t slot = static_cast<size_t>(b - first_block_);

                t_block_[slot] = t;
                for (int a = 0; a < 3; ++a)
                    origin_[slot * 3 + static_cast<size_t>(a)] = k[a];

                int grad_id[3];
                const Wave* wave[3];
                for (int a = 0; a < 3; ++a)
                {
                    grad_id[a] = static_cast<int>(row[BLOCK_GX + a]);
                    wave[a] = wave_for(grad_id[a]);
                }

                const double duration = durations[static_cast<size_t>(b)];

                RfRole role = RfRole::Neutral;
                double t_center = 0.0;
                if (row[BLOCK_RF] > 0 && row[BLOCK_RF] <= rf_rows)
                {
                    const double* rf = seq_.rf_library().row(static_cast<int>(row[BLOCK_RF]));
                    role = rf_role(seq_.rf_uses()[static_cast<size_t>(row[BLOCK_RF]) - 1]);
                    t_center = rf[RF_DELAY] + rf[RF_CENTER];

                    if (role != RfRole::Neutral)
                    {
                        KTrajRf& event = (role == RfRole::Excitation)
                            ? excitations_[static_cast<size_t>(i_exc)]
                            : refocusings_[static_cast<size_t>(i_ref)];
                        const double freq = rf[RF_FREQ] + rf[RF_FREQ_PPM] * 1.0e-6 * gb0;
                        const double phase = rf[RF_PHASE] + rf[RF_PHASE_PPM] * 1.0e-6 * gb0;

                        event.block_index = b + 1;
                        event.t = t + t_center;
                        event.freq_offset = freq;
                        /* PyPulseq references the phase to the pulse centre. */
                        event.phase_offset = phase + TWO_PI * freq * rf[RF_CENTER];
                        for (int a = 0; a < 3; ++a)
                            event.g[static_cast<size_t>(a)] =
                                wave[a] ? wave[a]->amplitude * wave[a]->p.value_at(t_center) : 0.0;

                        if (role == RfRole::Excitation)
                            i_exc++;
                        else
                            i_ref++;
                    }
                }

                if (row[BLOCK_ADC] > 0 && row[BLOCK_ADC] <= adc_rows)
                {
                    const int adc_id = static_cast<int>(row[BLOCK_ADC]);
                    const double* adc = seq_.adc_library().row(adc_id);
                    KTrajReadout& ro = readouts_[static_cast<size_t>(i_adc)];
                    RepeatKey& key = keys_[static_cast<size_t>(i_adc)];
                    const double dwell = adc[ADC_DWELL];
                    const double delay = adc[ADC_DELAY];
                    const int nsamples = static_cast<int>(adc[ADC_NUM]);
                    const int rotation_index = block_rotation_id(b + 1);

                    /* Sample centres, which is where Pulseq puts them. */
                    const double win0 = delay + 0.5 * dwell;
                    const double win1 = delay + (static_cast<double>(nsamples) - 0.5) * dwell;

                    adc_offset_[static_cast<size_t>(i_adc)] = win0;
                    adc_dwell_[static_cast<size_t>(i_adc)] = dwell;

                    ro.block_index = b + 1;
                    ro.num_samples = nsamples;
                    ro.t0 = t + win0;
                    ro.dwell = dwell;
                    ro.center_sample = -1;
                    ro.rotation_id = rotation_index;
                    for (int a = 0; a < 3; ++a)
                        ro.k0[static_cast<size_t>(a)] = k[a];

                    /*
                     * Constant axes, in the frame k is reported in.
                     *
                     * A rotated axis is constant only if every logical axis
                     * feeding it is.  Cancellation between two moving axes
                     * could in principle leave a rotated one still, but that
                     * is a coincidence rather than a design, and the error is
                     * in the safe direction: samples stored that need not have
                     * been.
                     */
                    {
                        const double* m =
                            opts_.apply_rotation ? rotation_for(rotation_index) : nullptr;
                        bool flat[3];
                        for (int a = 0; a < 3; ++a)
                            flat[a] = !wave[a] || is_flat(wave[a]->p, win0, win1);

                        ro.constant_axes = 0u;
                        for (int a = 0; a < 3; ++a)
                        {
                            bool constant = true;
                            if (m != nullptr)
                            {
                                for (int s = 0; s < 3; ++s)
                                {
                                    if (std::fabs(m[a * 3 + s]) > 1.0e-12 && !flat[s])
                                        constant = false;
                                }
                            }
                            else
                            {
                                constant = flat[a];
                            }
                            if (constant)
                                ro.constant_axes |= 1u << a;
                        }
                    }

                    /* Canonical ids, so two spellings of the same readout
                     * share a key.  The waveform behind a canonical id is
                     * identical to the one behind the id it replaced, so
                     * lookups stay valid. */
                    for (int a = 0; a < 3; ++a)
                    {
                        key.grad[static_cast<size_t>(a)] =
                            (grad_id[a] > 0 && grad_id[a] < static_cast<int>(grad_canon_.size()))
                            ? grad_canon_[static_cast<size_t>(grad_id[a])]
                            : grad_id[a];
                    }
                    key.adc = (adc_id < static_cast<int>(adc_canon_.size()))
                        ? adc_canon_[static_cast<size_t>(adc_id)]
                        : adc_id;
                    key.rotation = rotation_index;
                    i_adc++;
                }

                /* Accumulate this block's moment into k. */
                if (role != RfRole::Neutral)
                {
                    for (int a = 0; a < 3; ++a)
                    {
                        if (!wave[a])
                        {
                            k[a] = (role == RfRole::Excitation) ? 0.0 : -k[a];
                            continue;
                        }
                        const double before = wave[a]->p.cumulative_at(t_center);
                        double mid = k[a] + wave[a]->amplitude * before;
                        mid = (role == RfRole::Excitation) ? 0.0 : -mid;
                        k[a] = mid + wave[a]->amplitude * (wave[a]->p.total() - before);
                    }
                }
                else
                {
                    for (int a = 0; a < 3; ++a)
                    {
                        if (wave[a])
                            k[a] += wave[a]->amplitude * wave[a]->p.total();
                    }
                }

                /* A background gradient runs for the whole block regardless of
                 * what the block drives, so it is added here rather than
                 * folded into a wave that may not exist. */
                for (int a = 0; a < 3; ++a)
                {
                    if (opts_.gradient_offset[static_cast<size_t>(a)] != 0.0)
                        k[a] += opts_.gradient_offset[static_cast<size_t>(a)] * duration;
                }

                t += duration;
            }

            total_duration_ = t;
        }

        /* ============================================================== */
        /*  Sampling one readout                                          */
        /* ============================================================== */

        /*
         * The gradient time base carries `trajectory_delay`: sampling the
         * cumulative at (t - delay) is what shifts the gradient relative to
         * the ADC without moving the ADC, which is the distinction the option
         * exists to make.
         */
        void KTrajPlan::samples(int index, double* out_k, bool include_origin) const
        {
            const KTrajReadout& ro = readouts_[static_cast<size_t>(index)];
            const RepeatKey& key = keys_[static_cast<size_t>(index)];
            const int n = ro.num_samples;
            const double win_start = adc_offset_[static_cast<size_t>(index)];
            const double dwell = adc_dwell_[static_cast<size_t>(index)];

            for (int a = 0; a < 3; ++a)
            {
                const Wave* w = wave_for(key.grad[static_cast<size_t>(a)]);
                const double origin = include_origin ? ro.k0[static_cast<size_t>(a)] : 0.0;
                const double amplitude = w ? w->amplitude : 0.0;
                const double shift = opts_.trajectory_delay[static_cast<size_t>(a)];
                const double offset = opts_.gradient_offset[static_cast<size_t>(a)];
                double* dst = out_k + static_cast<size_t>(a) * static_cast<size_t>(n);

                for (int i = 0; i < n; ++i)
                {
                    const double tt = win_start + static_cast<double>(i) * dwell;
                    double v = origin;
                    if (w)
                    {
                        v += amplitude *
                            (opts_.sample_window_average
                                 ? mean_cumulative(w->p, w->c2, tt - shift, dwell)
                                 : w->p.cumulative_at(tt - shift));
                    }
                    if (offset != 0.0)
                        v += offset * tt;
                    dst[i] = v;
                }
            }

            if (opts_.apply_rotation)
            {
                const double* m = rotation_for(key.rotation);
                if (m != nullptr)
                {
                    for (int i = 0; i < n; ++i)
                    {
                        double xyz[3] = {out_k[i], out_k[n + i], out_k[2 * n + i]};
                        rotate(m, xyz);
                        out_k[i] = xyz[0];
                        out_k[n + i] = xyz[1];
                        out_k[2 * n + i] = xyz[2];
                    }
                }
            }
        }

        /* The same arithmetic for a single sample, so a caller that wants one
         * number does not pay for the whole readout. */
        std::array<double, 3> KTrajPlan::sample_k(int index, int sample) const
        {
            const KTrajReadout& ro = readouts_[static_cast<size_t>(index)];
            const RepeatKey& key = keys_[static_cast<size_t>(index)];
            const double dwell = adc_dwell_[static_cast<size_t>(index)];
            const double tt =
                adc_offset_[static_cast<size_t>(index)] + static_cast<double>(sample) * dwell;

            std::array<double, 3> out{{0.0, 0.0, 0.0}};
            for (int a = 0; a < 3; ++a)
            {
                const Wave* w = wave_for(key.grad[static_cast<size_t>(a)]);
                double v = ro.k0[static_cast<size_t>(a)];
                if (w)
                {
                    const double at = tt - opts_.trajectory_delay[static_cast<size_t>(a)];
                    v += w->amplitude *
                        (opts_.sample_window_average ? mean_cumulative(w->p, w->c2, at, dwell)
                                                     : w->p.cumulative_at(at));
                }
                if (opts_.gradient_offset[static_cast<size_t>(a)] != 0.0)
                    v += opts_.gradient_offset[static_cast<size_t>(a)] * tt;
                out[static_cast<size_t>(a)] = v;
            }

            if (opts_.apply_rotation)
            {
                const double* m = rotation_for(key.rotation);
                if (m != nullptr)
                    rotate(m, out.data());
            }
            return out;
        }

        std::array<double, 3> KTrajPlan::gradient_at(int index, double t_in_block) const
        {
            const RepeatKey& key = keys_[static_cast<size_t>(index)];

            std::array<double, 3> out{{0.0, 0.0, 0.0}};
            for (int a = 0; a < 3; ++a)
            {
                const Wave* w = wave_for(key.grad[static_cast<size_t>(a)]);
                const double shift = opts_.trajectory_delay[static_cast<size_t>(a)];
                out[static_cast<size_t>(a)] =
                    w ? w->amplitude * w->p.value_at(t_in_block - shift) : 0.0;
                out[static_cast<size_t>(a)] += opts_.gradient_offset[static_cast<size_t>(a)];
            }

            if (opts_.apply_rotation)
            {
                const double* m = rotation_for(key.rotation);
                if (m != nullptr)
                    rotate(m, out.data());
            }
            return out;
        }

        std::array<double, 3> KTrajPlan::readout_origin(int index) const
        {
            const KTrajReadout& ro = readouts_[static_cast<size_t>(index)];
            std::array<double, 3> p = ro.k0;
            if (opts_.apply_rotation)
            {
                const double* m = rotation_for(keys_[static_cast<size_t>(index)].rotation);
                if (m != nullptr)
                    rotate(m, p.data());
            }
            return p;
        }

        /* ============================================================== */
        /*  Deriving the echo                                             */
        /* ============================================================== */

        namespace
        {
            /**
             * Mean distance between consecutive samples of a readout, over the
             * swept axes.
             *
             * The scale every "is this a tie" question is asked against.  A
             * readout's own step is the only meaningful unit here: k is in 1/m
             * and the field of view is the sequence's business, so an absolute
             * tolerance would mean something different for every scan.
             */
            double sample_step(const double* buf, int n, unsigned int swept)
            {
                double total = 0.0;
                int steps = 0;
                for (int j = 1; j < n; ++j)
                {
                    double sq = 0.0;
                    for (int a = 0; a < 3; ++a)
                    {
                        if (!(swept & (1u << a)))
                            continue;
                        const double dv = buf[a * n + j] - buf[a * n + (j - 1)];
                        sq += dv * dv;
                    }
                    total += std::sqrt(sq);
                    steps++;
                }
                return (steps > 0) ? total / static_cast<double>(steps) : 0.0;
            }

            /** Whether k advances along the readout: its dominant displacement. */
            bool sweeps_forward(const double* buf, int n, unsigned int swept)
            {
                if (n < 2)
                    return true;
                double best = 0.0;
                for (int a = 0; a < 3; ++a)
                {
                    if (!(swept & (1u << a)))
                        continue;
                    const double d = buf[a * n + (n - 1)] - buf[a * n];
                    if (std::fabs(d) > std::fabs(best))
                        best = d;
                }
                return best >= 0.0;
            }

            /**
             * Which axes a readout sweeps, from the origin-free samples.
             *
             * Judged against the largest range across the three axes rather
             * than against an absolute number, because k is in 1/m and a
             * sequence's scale is its own business.  A rotation smears a
             * nominally idle axis by rounding alone, and the relative
             * threshold is what keeps that from reading as a sweep.
             */
            unsigned int swept_axes(const double* v, int n)
            {
                double lo[3], hi[3], widest = 0.0;
                for (int a = 0; a < 3; ++a)
                {
                    lo[a] = v[a * n];
                    hi[a] = v[a * n];
                    for (int j = 1; j < n; ++j)
                    {
                        lo[a] = std::min(lo[a], v[a * n + j]);
                        hi[a] = std::max(hi[a], v[a * n + j]);
                    }
                    widest = std::max(widest, hi[a] - lo[a]);
                }

                unsigned int mask = 0u;
                for (int a = 0; a < 3; ++a)
                {
                    if (hi[a] - lo[a] > 1.0e-9 * widest && hi[a] > lo[a])
                        mask |= 1u << a;
                }
                return mask;
            }
        } // namespace

        /*
         * ### Why the repeat key drops part of the origin
         *
         * The obvious key -- gradient ids, ADC, rotation and the whole entry
         * k -- is correct and almost never fires.  Measured on the fixtures, it
         * hit 0% of readouts everywhere, because in a Cartesian scan every line
         * enters with a different phase-encode prewinder and so no two entry
         * vectors agree.  A memo that never hits is not an optimisation.
         *
         * The origin does not have to be in the key in full.  Writing p for the
         * rotated entry k, v(j) for the rotated sweep the readout adds, and c
         * for the scan's k centre:
         *
         *     |p + v(j) - c|^2  =  sum_a (p_a + v_a(j) - c_a)^2
         *
         * and for an axis where v_a(j) does not change with j, that term is the
         * same for every sample -- it shifts the whole curve and cannot move
         * the minimum.  So the answer depends on p only through the axes the
         * readout actually sweeps.  For a Cartesian readout that is the readout
         * axis alone, and its prewinder is identical on every line, so all
         * lines share a key and the memo fires on essentially all of them.
         * That is the entire speed argument, and it is exact, not an
         * approximation.
         *
         * ### The two sweeps
         *
         * The same split makes the global k centre cheap.  Sweep 1 needs
         * min over readouts and samples of |p + v(j)|^2, which separates:
         *
         *     |p + v(j)|^2  =  sum_{a static} (p_a + v_a)^2
         *                    + sum_{a swept} (p_a + v_a(j))^2
         *
         * so each group pays one scan over its samples and each readout pays a
         * constant.  Sweep 2 memoizes on the key directly.
         */
        void KTrajPlan::derive_centers()
        {
            const int n_ro = static_cast<int>(readouts_.size());
            if (n_ro <= 0)
                return;

            int longest = 0;
            for (const KTrajReadout& ro : readouts_)
                longest = std::max(longest, ro.num_samples);
            if (longest <= 0)
                return;

            std::vector<double> buf(static_cast<size_t>(longest) * 3, 0.0);
            std::vector<std::array<double, 3>> origins(static_cast<size_t>(n_ro));
            std::vector<Group> groups(static_cast<size_t>(n_ro));
            std::vector<RepeatKey> mkeys(static_cast<size_t>(n_ro));

            for (int i = 0; i < n_ro; ++i)
                origins[static_cast<size_t>(i)] = readout_origin(i);

            /* For each readout, the index of the first readout sharing its
             * key.  Open addressing rather than a scan over everything already
             * seen, which on a million readouts would cost more than the
             * per-sample work the memo exists to avoid. */
            const auto group_by_key = [n_ro](const std::vector<RepeatKey>& keys)
            {
                /* canonical_ids works in the libraries' 1-based numbering; a
                 * readout index is 0-based, so it shifts either way. */
                const std::vector<int> canon = canonical_ids(
                    n_ro,
                    [&](int id) { return RepeatKeyHash{}(keys[static_cast<size_t>(id - 1)]); },
                    [&](int a, int b) {
                        return keys[static_cast<size_t>(a - 1)] == keys[static_cast<size_t>(b - 1)];
                    });

                std::vector<int> first(static_cast<size_t>(n_ro));
                for (int i = 0; i < n_ro; ++i)
                    first[static_cast<size_t>(i)] = canon[static_cast<size_t>(i) + 1] - 1;
                return first;
            };

            /* ---- Phase A: group by shape alone, to learn which axes are
             * swept.  This has to come first: the swept set decides which
             * parts of the origin belong in the full key, and it is a property
             * of the gradients and the rotation only, so one representative
             * per shape answers for all of them. */
            for (int i = 0; i < n_ro; ++i)
            {
                mkeys[static_cast<size_t>(i)] = keys_[static_cast<size_t>(i)];
                mkeys[static_cast<size_t>(i)].k0 = {{0.0, 0.0, 0.0}};
            }
            const std::vector<int> shape_first = group_by_key(mkeys);

            for (int i = 0; i < n_ro; ++i)
            {
                const int nn = readouts_[static_cast<size_t>(i)].num_samples;
                if (shape_first[static_cast<size_t>(i)] != i || nn <= 0)
                    continue;
                samples(i, buf.data(), false);
                groups[static_cast<size_t>(i)].swept = swept_axes(buf.data(), nn);
            }

            /* ---- Phase B: the full key -- shape plus the origin on swept
             * axes only.  Axes the readout does not sweep are left at zero,
             * which is what lets every phase-encode line of a Cartesian scan
             * share one key. */
            for (int i = 0; i < n_ro; ++i)
            {
                const unsigned int swept =
                    groups[static_cast<size_t>(shape_first[static_cast<size_t>(i)])].swept;
                RepeatKey& key = mkeys[static_cast<size_t>(i)];
                key = keys_[static_cast<size_t>(i)];
                for (int a = 0; a < 3; ++a)
                {
                    if (swept & (1u << a))
                    {
                        key.k0[static_cast<size_t>(a)] =
                            origins[static_cast<size_t>(i)][static_cast<size_t>(a)];
                    }
                    else
                    {
                        /* Contributes the same constant to every sample, so
                         * neither the gradient nor the entry point can change
                         * the answer. */
                        key.k0[static_cast<size_t>(a)] = 0.0;
                        key.grad[static_cast<size_t>(a)] = 0;
                    }
                }
                key.swept = swept;
            }
            const std::vector<int> key_first = group_by_key(mkeys);

            /* ---- Phase C: per group, the swept-axis minimum. */
            for (int i = 0; i < n_ro; ++i)
            {
                const int nn = readouts_[static_cast<size_t>(i)].num_samples;
                if (key_first[static_cast<size_t>(i)] != i || nn <= 0)
                    continue;

                Group& g = groups[static_cast<size_t>(i)];
                g.swept = mkeys[static_cast<size_t>(i)].swept;

                samples(i, buf.data(), true);
                g.tie_tol = 1.0e-2 * sample_step(buf.data(), nn, g.swept);
                g.forward = sweeps_forward(buf.data(), nn, g.swept);
                g.var_min = 0.0;
                g.var_argmin = 0;

                /*
                 * Distances, and a tie resolves along the direction of travel
                 * -- the same rule sweep 2 applies, and it has to be the same
                 * rule.  A symmetric readout's two central samples are exactly
                 * equidistant from the origin, so a strict comparison here
                 * would take the earlier one while sweep 2 took the later, and
                 * the whole scan's echo index would come out one low.
                 */
                for (int j = 0; j < nn; ++j)
                {
                    double d = 0.0;
                    for (int a = 0; a < 3; ++a)
                    {
                        if (g.swept & (1u << a))
                            d += buf[a * nn + j] * buf[a * nn + j];
                    }
                    d = std::sqrt(d);

                    if (j == 0 || d < g.var_min - g.tie_tol)
                    {
                        g.var_min = d;
                        g.var_argmin = j;
                    }
                    else if (d <= g.var_min + g.tie_tol)
                    {
                        if (d < g.var_min)
                            g.var_min = d;
                        if (g.forward)
                            g.var_argmin = j;
                    }
                }
            }

            /* ---- Sweep 1: the scan's closest approach to k = 0.
             *
             * The static part is read off sample 0, where by definition it
             * stays for the whole readout -- a group now spans readouts whose
             * static-axis gradients differ, since those were dropped from the
             * key.
             *
             * A tie here is between whole readouts, and it is resolved in
             * favour of a **forward** one.  Which polarity wins no longer
             * decides anyone's echo index -- the tie rule in phase C is
             * geometric, so both polarities of an EPI already agree that the
             * centre lies at the same *point* in k.  What this still decides is
             * which readout is reported as `central_readout`, and a forward one
             * is the readout whose sample indices need no mirroring.
             */
            double best_total = 0.0;
            int best_readout = -1;
            for (int i = 0; i < n_ro; ++i)
            {
                const int nn = readouts_[static_cast<size_t>(i)].num_samples;
                if (nn <= 0)
                    continue;

                const Group& g = groups[static_cast<size_t>(key_first[static_cast<size_t>(i)])];
                const std::array<double, 3> first_sample = sample_k(i, 0);
                double total = g.var_min * g.var_min;
                for (int a = 0; a < 3; ++a)
                {
                    if (!(g.swept & (1u << a)))
                        total += first_sample[static_cast<size_t>(a)] *
                            first_sample[static_cast<size_t>(a)];
                }
                total = std::sqrt(total);

                if (best_readout < 0)
                {
                    best_total = total;
                    best_readout = i;
                    continue;
                }

                const Group& incumbent =
                    groups[static_cast<size_t>(key_first[static_cast<size_t>(best_readout)])];
                const double tol = std::max(g.tie_tol, incumbent.tie_tol);

                if (total < best_total - tol)
                {
                    best_total = total;
                    best_readout = i;
                }
                else if (total <= best_total + tol)
                {
                    if (total < best_total)
                        best_total = total;
                    if (g.forward && !incumbent.forward)
                        best_readout = i;
                }
            }

            if (best_readout >= 0)
            {
                const Group& g =
                    groups[static_cast<size_t>(key_first[static_cast<size_t>(best_readout)])];
                k_center_ = sample_k(best_readout, g.var_argmin);
                central_readout_ = best_readout;
            }

            /* ---- Sweep 2: each readout's sample closest to that point. */
            for (int i = 0; i < n_ro; ++i)
            {
                const int nn = readouts_[static_cast<size_t>(i)].num_samples;
                if (nn <= 0)
                    continue;

                const int gi = key_first[static_cast<size_t>(i)];
                Group& g = groups[static_cast<size_t>(gi)];

                if (g.center == -2)
                {
                    /*
                     * A readout whose k does not move has no echo -- every
                     * sample is the same distance from the centre, so there is
                     * no sample that is "the" crossing.  A noise scan and a
                     * free-induction readout are both like this.
                     *
                     * It stays -1 rather than becoming 0 or num_samples/2.
                     * Inventing one here would be indistinguishable from a
                     * derived answer at every layer above, and a wrong echo
                     * index is a silently mis-reconstructed image.
                     */
                    if (g.swept == 0u)
                    {
                        g.center = -1;
                        readouts_[static_cast<size_t>(i)].center_sample = -1;
                        continue;
                    }

                    samples(gi, buf.data(), true);

                    /*
                     * A fully sampled symmetric readout straddles the origin:
                     * its two central samples sit at -dk/2 and +dk/2 and are
                     * the same distance from it.  Which of them wins would
                     * otherwise be decided by whatever asymmetry survives the
                     * file's six printed digits -- in one fixture the margin is
                     * 2e-4 out of a step of 4.5, or 0.005%.  Letting that
                     * decide would make the echo index a property of a rounding
                     * error.
                     *
                     * So anything inside the group's tolerance counts as a tie,
                     * and a tie resolves towards **increasing k along the
                     * direction of travel**: the later sample on a forward
                     * readout, the earlier one on a reverse one.
                     *
                     * Stated as an index rule it would be two different rules;
                     * stated geometrically it is one, and it is the one that
                     * survives an EPI.  A bipolar train's two polarities are
                     * mirror images, so their nearest samples are *different
                     * points*: with N = 128 the forward echo is sample 64 and
                     * the reverse echo is sample 63, and a reconstruction that
                     * mirrors the reverse lines -- which is what the REV label
                     * asks it to do -- finds both at 128 - 1 - 63 = 64.
                     */
                    int best = 0;
                    double best_d = 0.0;
                    for (int j = 0; j < nn; ++j)
                    {
                        double d = 0.0;
                        for (int a = 0; a < 3; ++a)
                        {
                            /* Axes the readout does not sweep shift every
                             * sample by the same amount, so they cannot move
                             * the minimum. */
                            if (!(g.swept & (1u << a)))
                                continue;
                            const double v = buf[a * nn + j] - k_center_[static_cast<size_t>(a)];
                            d += v * v;
                        }
                        d = std::sqrt(d); /* distances, so the tolerance means something */

                        if (j == 0 || d < best_d - g.tie_tol)
                        {
                            best_d = d;
                            best = j;
                        }
                        else if (d <= best_d + g.tie_tol)
                        {
                            /* Keep the smaller distance either way, so a long
                             * near-flat run cannot walk the answer along one
                             * tolerance at a time. */
                            if (d < best_d)
                                best_d = d;
                            if (g.forward)
                                best = j;
                        }
                    }
                    g.center = best;
                }

                KTrajReadout& ro = readouts_[static_cast<size_t>(i)];
                ro.center_sample = g.center;

                /* No echo, no gradient at the echo and no k at it -- both stay
                 * zero rather than being sampled at a made-up time. */
                if (g.center >= 0)
                {
                    const double t_in_block = adc_offset_[static_cast<size_t>(i)] +
                        static_cast<double>(g.center) * adc_dwell_[static_cast<size_t>(i)];
                    ro.g_echo = gradient_at(i, t_in_block);
                    /* Per readout, not per group: members of a group share a
                     * shape and their swept-axis entry point, but the axes
                     * dropped from the key are exactly the ones whose constant
                     * offset differs -- which is the whole phase-encode table. */
                    ro.k_echo = sample_k(i, g.center);
                }
            }

            /* The hit rate is worth having: a scan that should repeat and does
             * not is a wrong key, not a slow run, and the caller cannot see
             * that otherwise. */
            key_groups_ = 0;
            for (int i = 0; i < n_ro; ++i)
            {
                if (key_first[static_cast<size_t>(i)] == i)
                    key_groups_++;
            }
        }

        /* ============================================================== */
        /*  Construction and the public queries                           */
        /* ============================================================== */

        KTrajPlan::KTrajPlan(const Sequence& seq, const KTrajOptions& options)
            : seq_(seq), opts_(options)
        {
            if (seq.num_blocks() <= 0)
                throw std::runtime_error("calculate_kspace: the sequence has no blocks");

            first_block_ = (options.first_block > 0) ? options.first_block - 1 : 0;
            last_block_ = (options.last_block > 0) ? options.last_block - 1 : seq.num_blocks() - 1;
            last_block_ = std::min(last_block_, seq.num_blocks() - 1);
            if (first_block_ > last_block_)
                throw std::runtime_error("calculate_kspace: the block range is empty");
            num_range_blocks_ = last_block_ - first_block_ + 1;

            build_rotations();
            canonicalise_events();
            build_waves();
            walk_blocks();
            if (opts_.derive_center_sample)
                derive_centers();
        }

        void KTrajPlan::readout_k(int index, double* out_k) const
        {
            if (index < 0 || index >= static_cast<int>(readouts_.size()) || out_k == nullptr)
                return;
            if (readouts_[static_cast<size_t>(index)].num_samples <= 0)
                return;
            samples(index, out_k, true);
        }

        std::array<double, 3> KTrajPlan::block_origin(int block_index) const
        {
            const int slot = block_index - 1 - first_block_;
            if (slot < 0 || slot >= num_range_blocks_)
                return {{0.0, 0.0, 0.0}};
            return {
                {origin_[static_cast<size_t>(slot) * 3],
                 origin_[static_cast<size_t>(slot) * 3 + 1],
                 origin_[static_cast<size_t>(slot) * 3 + 2]}};
        }

        void KTrajPlan::at_times(const double* times, int count, double* out_k) const
        {
            const int32_t* events = seq_.block_events();

            for (int i = 0; i < count; ++i)
            {
                double t = times[i];
                t = std::max(t, 0.0);
                t = std::min(t, total_duration_);

                /* The block containing @p t.  A zero-duration block shares its
                 * start time with the one after it; the ends are answered
                 * before the search so a run of them at t = 0 resolves to the
                 * first, not the last. */
                size_t slot = t_block_.size() - 1;
                if (t <= t_block_.front())
                {
                    slot = 0;
                }
                else if (t < t_block_.back())
                {
                    slot = static_cast<size_t>(
                               std::upper_bound(t_block_.begin(), t_block_.end(), t) -
                               t_block_.begin()) -
                        1;
                }
                const int b = first_block_ + static_cast<int>(slot);
                const double local = t - t_block_[slot];
                const int32_t* row = events + static_cast<size_t>(b) * BLOCK_WIDTH;

                double k[3];
                for (int a = 0; a < 3; ++a)
                {
                    const Wave* w = wave_for(static_cast<int>(row[BLOCK_GX + a]));
                    k[a] = origin_[slot * 3 + static_cast<size_t>(a)];
                    if (w)
                        k[a] += w->amplitude *
                            w->p.cumulative_at(
                                local - opts_.trajectory_delay[static_cast<size_t>(a)]);
                    if (opts_.gradient_offset[static_cast<size_t>(a)] != 0.0)
                        k[a] += opts_.gradient_offset[static_cast<size_t>(a)] * local;
                }

                if (opts_.apply_rotation)
                {
                    const double* m = rotation_for(block_rotation_id(b + 1));
                    if (m != nullptr)
                        rotate(m, k);
                }

                for (int a = 0; a < 3; ++a)
                    out_k
                        [static_cast<size_t>(a) * static_cast<size_t>(count) +
                         static_cast<size_t>(i)] = k[a];
            }
        }

        std::vector<double> KTrajPlan::breakpoints() const
        {
            const int32_t* events = seq_.block_events();

            /* Every block contributes its own start plus each axis's
             * breakpoints, and the sequence contributes its end. */
            size_t capacity = static_cast<size_t>(num_range_blocks_) + 1;
            for (int slot = 0; slot < num_range_blocks_; ++slot)
            {
                const int32_t* row =
                    events + static_cast<size_t>(first_block_ + slot) * BLOCK_WIDTH;
                for (int a = 0; a < 3; ++a)
                {
                    const Wave* w = wave_for(static_cast<int>(row[BLOCK_GX + a]));
                    if (w)
                        capacity += w->p.t.size();
                }
            }

            std::vector<double> all;
            all.reserve(capacity);
            for (int slot = 0; slot < num_range_blocks_; ++slot)
            {
                const int32_t* row =
                    events + static_cast<size_t>(first_block_ + slot) * BLOCK_WIDTH;
                const double base = t_block_[static_cast<size_t>(slot)];

                all.push_back(base);
                for (int a = 0; a < 3; ++a)
                {
                    const Wave* w = wave_for(static_cast<int>(row[BLOCK_GX + a]));
                    if (!w)
                        continue;
                    for (size_t i = 0; i < w->p.t.size(); ++i)
                        all.push_back(base + w->p.t[i]);
                }
            }
            all.push_back(total_duration_);

            std::sort(all.begin(), all.end());

            /* Merge points closer than a tenth of a nanosecond.  Pulseq's own
             * calculateKspacePP rounds to the same 0.1 ns before its unique(),
             * for the same reason: two breakpoints that differ only by the last
             * bit of a printed decimal are one breakpoint. */
            constexpr double tol = 1.0e-10;
            size_t unique = 0;
            for (size_t i = 0; i < all.size(); ++i)
            {
                if (unique == 0 || all[i] - all[unique - 1] > tol)
                    all[unique++] = all[i];
            }
            all.resize(unique);
            return all;
        }

    } // namespace detail
} // namespace pulseq
