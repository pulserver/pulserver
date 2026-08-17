/**
 * @file ktraj.hpp
 * @brief Absolute k at every ADC sample of a `pulseq::Sequence`, and where each
 *        readout's echo is.
 *
 * Library-private, like waveform.hpp: `kspace.cpp` is the only consumer and
 * `pulseq::calculate_kspace` is the surface everything else uses.  What is
 * here is the arithmetic, kept apart from the marshalling so neither file has
 * to be read to follow the other.
 *
 * ### What this is for
 *
 * Non-Cartesian recon needs an absolute k for each acquired sample, and it
 * needs to know which sample of each readout is the echo.  Neither is written
 * into a `.seq`: nothing in the format marks the echo sample, and PyPulseq's
 * `calculate_kspace` does not compute an echo position at all.  So this
 * derives both, from nothing but the sequence.
 *
 * ### Why it is not just cumulative-sum-the-waveform
 *
 * A gradient is held as a **normalised shape beside a scalar amplitude**, and
 * a scan replays a handful of distinct shapes a great many times.  Integrating
 * the shape library once -- rather than integrating a materialised waveform
 * per block -- makes the cost scale with the number of distinct gradients
 * instead of with the length of the scan.  On top of that, blocks repeat: see
 * @ref KTrajPlan for the repeat key.
 *
 * ### Frames
 *
 * By default a rotation extension is resolved into the samples, so k comes
 * back in the **physical** frame -- what a reconstruction needs, and what
 * PyPulseq and Pulseq's MATLAB `calculateKspacePP` both return.  Clear
 * `apply_rotation` for the logical frame, which is what the FOV-shift path
 * asks for: the phase `dr . k` is invariant when both are rotated, so that
 * path must never see a resolved rotation.  The two are different contracts,
 * not an inconsistency.
 */

#ifndef PULSEQ_CXX_KTRAJ_HPP
#define PULSEQ_CXX_KTRAJ_HPP

#include "pulseq/sequence.hpp"
#include "waveform.hpp"

#include <array>
#include <vector>

namespace pulseq
{
    namespace detail
    {

        /** What to compute, and in which frame. */
        struct KTrajOptions
        {
            /**
             * Per-axis gradient timing compensation, seconds.  Shifts the
             * gradient time base only -- not the ADC sample times and not the
             * RF event times, which are intrinsically synchronised with each
             * other.  Applying it to the ADC instead is the mistake Pulseq's
             * own `calculateKspacePP` documents having made and reverted.
             */
            std::array<double, 3> trajectory_delay{{0.0, 0.0, 0.0}};

            /** Per-axis background gradient, Hz/m, added before integration. */
            std::array<double, 3> gradient_offset{{0.0, 0.0, 0.0}};

            /** 1-based inclusive; 0 for @c last_block means "to the end". */
            int first_block = 1;
            int last_block = 0;

            /** Resolve rotation extensions into k (the physical frame). */
            bool apply_rotation = true;

            /**
             * Give each sample the k averaged over its dwell window rather
             * than k at the window's midpoint.
             *
             * An ADC sample integrates signal for a whole dwell, so the
             * average is the coordinate it physically belongs to; the midpoint
             * is a first-order stand-in for it.  The two differ by
             * `(dwell^2 / 24) * dg/dt`, which is **exactly zero while the
             * gradient is flat** -- so an ordinary Cartesian readout is
             * unaffected and a ramp-sampled one is not.
             *
             * Off by default, and deliberately so: PyPulseq, MRpro and
             * mri-nufft all sample at the midpoint.
             */
            bool sample_window_average = false;

            /**
             * Derive each readout's `center_sample`.  Costs two extra sweeps
             * over the readouts, so a caller that only wants k can decline it.
             */
            bool derive_center_sample = true;
        };

        /** One ADC event and where it sits in k-space. */
        struct KTrajReadout
        {
            int block_index = 0; /**< 1-based block carrying the ADC     */
            int num_samples = 0;
            double t0 = 0.0;    /**< time of the first sample centre, s */
            double dwell = 0.0; /**< s                                  */
            /** k at the block start, 1/m, logical frame. */
            std::array<double, 3> k0{{0.0, 0.0, 0.0}};
            /** Gradient at the echo sample, Hz/m; zero when there is none. */
            std::array<double, 3> g_echo{{0.0, 0.0, 0.0}};
            /**
             * Absolute k at the echo sample, 1/m, in the output frame; zero
             * when there is no echo.
             *
             * Carried per readout because it is what an encoding-label
             * analysis actually reads: reducing a scan to one point per
             * readout is the step that makes such an analysis cost
             * O(readouts) instead of O(samples).
             */
            std::array<double, 3> k_echo{{0.0, 0.0, 0.0}};
            /**
             * Index of the sample closest to the k-space centre, or -1 when it
             * was not derived.  See @ref KTrajPlan for the rule.
             *
             * It is an index into *this* readout as acquired, so on a bipolar
             * train the two polarities report different numbers -- 64 and 63
             * on a 128-sample readout -- because their nearest samples are the
             * same point in k reached from opposite ends.  Mirroring the
             * reverse lines, which is what their REV label asks for, brings
             * both to 64.
             */
            int center_sample = -1;
            /**
             * Bit i set when axis i's k moves at a constant rate across the
             * ADC window, **in the frame k is reported in**.
             *
             * With `apply_rotation` set this is the physical answer, which is
             * not the logical one: a radial spoke is a flat gradient on one
             * logical axis plus a per-shot rotation, so it is logically as
             * constant as a Cartesian line while physically sweeping a line at
             * an arbitrary angle.  Handing back the logical answer beside a
             * rotated k would invite the shortcut that a constant gradient
             * needs no trajectory at all -- true for Cartesian, where the
             * recon indexes the line with an encoding counter and a frequency
             * offset, and false for radial, which has no such counter.
             */
            unsigned int constant_axes = 0u;
            /**
             * 0-based row of the rotation library this readout's block
             * carries, or -1 for none.
             *
             * Present so a spoke can be stored as a line plus an angle rather
             * than as samples; see @c constant_axes.
             */
            int rotation_id = -1;
        };

        /** One RF pulse that moves k, and the gradient it sits on. */
        struct KTrajRf
        {
            int block_index = 0;      /**< 1-based                          */
            double t = 0.0;           /**< pulse centre, s from range start */
            double freq_offset = 0.0; /**< Hz, including the ppm term       */
            double phase_offset = 0.0; /**< rad, referenced as PyPulseq does */
            std::array<double, 3> g{{0.0, 0.0, 0.0}}; /**< at the centre, Hz/m */
        };

        /**
         * The integrated gradient library plus per-block k origins.
         *
         * Constructing one is the expensive half of the work: it walks every
         * block once and integrates every distinct gradient once.  Reading
         * samples out of it afterwards is cheap.
         *
         * ### The repeat key
         *
         * With `derive_center_sample` set the constructor sweeps the readouts
         * twice: once to find the sample of the whole scan closest to the
         * origin, and once to set each readout's `center_sample` to its own
         * sample closest to *that* point.  Referencing the global closest
         * approach rather than a hard zero is what keeps the answer meaningful
         * for a readout that never crosses the origin -- partial Fourier, an
         * off-centre spiral interleaf -- where the nearest-to-zero sample
         * would just be an endpoint.  It is the rule Pulseq's `autoLabel.m`
         * uses, for the same reason.
         *
         * Both sweeps memoize on the block's **repeat key**: the event-id
         * tuple (gx, gy, gz, adc, rotation) plus the entry k on the axes the
         * readout sweeps.  Two blocks agreeing on all of it produce the same
         * samples, so the echo search runs once per distinct readout rather
         * than once per TR.  The rotation id has to be in the key: gradient
         * ids alone are shape-and-amplitude only, and two blocks that differ
         * solely by a per-slice rotation would otherwise collide.
         *
         * @throws std::runtime_error on an empty or inverted block range, or
         * on an RF pulse that declares no `use` -- see walk_blocks() in
         * ktraj.cpp for why that one is refused rather than guessed at.
         */
        class KTrajPlan
        {
        public:
            KTrajPlan(const Sequence& seq, const KTrajOptions& options);

            const std::vector<KTrajReadout>& readouts() const { return readouts_; }
            const std::vector<KTrajRf>& excitations() const { return excitations_; }
            const std::vector<KTrajRf>& refocusings() const { return refocusings_; }

            /** Sum of the block durations in range, seconds. */
            double total_duration() const { return total_duration_; }

            /**
             * Distinct repeat keys among the readouts, or 0 if centres were
             * not derived.
             *
             * `1 - key_groups / readouts.size()` is the fraction of readouts
             * the memo answered without a search.  Worth looking at: a scan
             * that ought to repeat and reports one group per readout has a key
             * that is wrong for it, which is a correctness smell rather than
             * merely a slow run.
             */
            int key_groups() const { return key_groups_; }

            /** The scan's closest approach to k = 0, 1/m. */
            const std::array<double, 3>& k_center() const { return k_center_; }

            /**
             * Index of the readout that achieves that closest approach, or -1.
             *
             * The scan's reference profile: the one readout guaranteed to pass
             * through the k-space centre, so its sample spacing defines the
             * encoding step and its shape says whether sampling is uniform.
             */
            int central_readout() const { return central_readout_; }

            /**
             * Absolute k for readout @p index, into a caller-supplied buffer
             * of `3 * num_samples`, written axis-major -- x, then y, then z.
             */
            void readout_k(int index, double* out_k) const;

            /**
             * k at the start of block @p block_index (1-based), 1/m.
             *
             * Always the logical frame -- a rotation belongs to the block that
             * carries it, and an origin is where the previous block left off.
             * Out-of-range indices return the origin, which is what a caller
             * asking outside its own range means.
             */
            std::array<double, 3> block_origin(int block_index) const;

            /**
             * Every block's origin at once, flat and axis-minor: block `slot`
             * of the plan's range occupies `[3 * slot, 3 * slot + 3)`.
             *
             * Handed out by reference because on a protocol-scale scan this is
             * fifty megabytes, and the caller that wants all of it -- the FOV
             * shift -- would otherwise copy it twice on the way.
             */
            const std::vector<double>& block_origins() const { return origin_; }

            /** The 0-based block the plan's range starts at. */
            int first_block() const { return first_block_; }

            /**
             * Event id -> the first id whose row means the same thing, for
             * gradients and for the ADC columns that decide when its samples
             * land.  Indexed from 1; entry 0 is unused.
             *
             * Handed out because the plan has already paid for them and a
             * caller walking the same blocks -- the FOV shift, which builds a
             * plan for the origins anyway -- would otherwise collapse the same
             * libraries a second time to key its own work.
             */
            const std::vector<int>& grad_canon() const { return grad_canon_; }
            const std::vector<int>& adc_canon() const { return adc_canon_; }

            /**
             * Every time at which the trajectory can change slope: the sorted,
             * de-duplicated union of block boundaries and gradient breakpoints
             * over the range, in seconds.
             *
             * Between two consecutive entries k is a single quadratic, so a
             * consumer that samples here and interpolates linearly draws the
             * trajectory with no more error than the drawing itself
             * introduces.  A ramp of any length costs two points.
             */
            std::vector<double> breakpoints() const;

            /**
             * Absolute k at arbitrary times, 1/m, axis-major (`3 * count`).
             *
             * Times are seconds from the start of the block range and need not
             * be sorted or lie on any raster.  A time outside the range clamps
             * to the nearest end, which is what makes a plot's closing sample
             * well defined.
             */
            void at_times(const double* times, int count, double* out_k) const;

        private:
            /** A gradient integrated once, at unit amplitude. */
            struct Wave
            {
                Piecewise p;
                /**
                 * The integral of `p.c` -- the *second* antiderivative.
                 *
                 * Only built for window-averaged sampling, which is the whole
                 * reason it exists: averaging k over a dwell means integrating
                 * k, and k is already an integral.
                 */
                std::vector<double> c2;
                double amplitude = 0.0;
                bool built = false;
            };

            /**
             * The memo key for a readout.  Two readouts agreeing on all of it
             * produce bit-identical samples and the same echo index.
             *
             * The entry k is compared **exactly**.  An earlier version rounded
             * it onto a grid, on the theory that two entry points closer than
             * a fraction of the sampling step could not pick different
             * samples.  That is false near a tie: in fse_2d the two
             * alternating readouts enter 0.0055 apart, well inside a 0.078
             * quantum, yet their echoes really are samples 127 and 128 -- both
             * equidistant from the origin, on opposite sides.  Exact
             * comparison cannot merge them, and it costs nothing in hit rate:
             * an excitation resets k to zero, so every TR reaches its readout
             * by summing the same moments from the same reset and arrives at a
             * bit-identical entry point.
             */
            struct RepeatKey
            {
                /**
                 * Gradient ids per axis, ADC id, rotation id.
                 *
                 * An axis the readout does not sweep is zeroed out of the
                 * *memo* copy of this key -- its gradient contributes the same
                 * constant to every sample and cannot move the echo.  That is
                 * not a detail: in a blipped Cartesian scan every line carries
                 * its own phase-encode gradient id, so keeping the id would
                 * give one group per line and no memo at all.  The plan's own
                 * copy always keeps the true ids; `samples()` reads them to
                 * find the waveforms.
                 */
                std::array<int, 3> grad{{0, 0, 0}};
                int adc = 0;
                int rotation = -1;
                /**
                 * Which axes the readout sweeps, carried rather than assumed.
                 *
                 * Zeroing a static axis out of `grad` means two readouts can
                 * agree on the surviving fields while having disagreed about
                 * which axes were static in the first place.  Including the
                 * mask makes that impossible instead of merely unlikely.
                 */
                unsigned int swept = 0u;
                std::array<double, 3> k0{{0.0, 0.0, 0.0}};

                bool operator==(const RepeatKey& other) const;
            };

            struct RepeatKeyHash
            {
                size_t operator()(const RepeatKey& key) const;
            };

            /** What a group of readouts sharing a repeat key has in common. */
            struct Group
            {
                unsigned int swept = 0u; /**< axes whose k actually changes   */
                double var_min = 0.0;    /**< min over swept axes of |k(j)|   */
                int var_argmin = 0;      /**< the j attaining it              */
                double tie_tol = 0.0;    /**< 1% of this readout's own step   */
                /**
                 * Whether k moves forwards through this readout: the sign of
                 * the dominant component of (last sample - first sample).  It
                 * is the same quantity a REV label carries, and it is what
                 * turns "resolve a tie towards increasing k" into an index.
                 */
                bool forward = true;
                /** Sweep-2 answer: index, -1 for no echo, -2 for not yet. */
                int center = -2;
            };

            const Wave* wave_for(int grad_id) const;
            const double* rotation_for(int rotation_id) const;
            int block_rotation_id(int block_index) const;
            void rotate(const double* m, double* xyz) const;

            void build_waves();
            void build_rotations();
            void canonicalise_events();
            void walk_blocks();
            void derive_centers();

            /** Absolute k for one readout, axis-major, into @p out. */
            void samples(int index, double* out, bool include_origin) const;
            /** Absolute k at one sample of one readout, in the output frame. */
            std::array<double, 3> sample_k(int index, int sample) const;
            /** The gradient vector at @p t_in_block, Hz/m, in the output frame. */
            std::array<double, 3> gradient_at(int index, double t_in_block) const;
            /** The rotated entry k of a readout. */
            std::array<double, 3> readout_origin(int index) const;

            const Sequence& seq_;
            KTrajOptions opts_;

            int first_block_ = 0; /**< 0-based inclusive */
            int last_block_ = 0;  /**< 0-based inclusive */
            int num_range_blocks_ = 0;

            /**
             * One per *distinct* gradient row, not one per id.
             *
             * That distinction is the whole cost argument, and it only holds
             * because the library is not deduplicated until the sequence is
             * written: a scan that plays one spiral arm a million times holds
             * a million identical rows, and integrating each would make the
             * pass scale with the scan rather than with its vocabulary.
             * `grad_wave_` is how an id reaches its row.
             */
            std::vector<Wave> waves_;
            std::vector<double> rot_; /**< 9 per rotation library row */
            int num_rot_ = 0;
            int rotation_type_ = 0; /**< the ROTATIONS extension's own id */

            /** Event id -> first equivalent event id; see canonicalise_events. */
            std::vector<int> grad_canon_;
            std::vector<int> adc_canon_;
            /** Gradient id -> its row in @ref waves_, or -1 for none. */
            std::vector<int> grad_wave_;

            /** Per block in range: k at its start (logical) and its start time. */
            std::vector<double> origin_;
            std::vector<double> t_block_;

            std::vector<KTrajReadout> readouts_;
            std::vector<RepeatKey> keys_;
            /**
             * Each readout's first-sample time and dwell, relative to its own
             * block.  Not recoverable from the readout: its `t0` is an
             * absolute time, and reconstructing the in-block offset as
             * `t0 - block_start` loses the small number to cancellation.
             */
            std::vector<double> adc_offset_;
            std::vector<double> adc_dwell_;

            std::vector<KTrajRf> excitations_;
            std::vector<KTrajRf> refocusings_;

            double total_duration_ = 0.0;
            std::array<double, 3> k_center_{{0.0, 0.0, 0.0}};
            int central_readout_ = -1;
            int key_groups_ = 0;
        };

    }  // namespace detail
}  // namespace pulseq

#endif /* PULSEQ_CXX_KTRAJ_HPP */
