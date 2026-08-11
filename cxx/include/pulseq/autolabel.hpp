/**
 * @file autolabel.hpp
 * @brief Recovering a sequence's encoding counters from its own trajectory.
 *
 * A `.seq` this project did not author usually carries no `LABELSET`
 * extensions, so nothing downstream knows which line, partition, slice or
 * repetition an acquisition belongs to.  Everything needed to say so is
 * nevertheless present -- it is written in where the readouts sit in k-space --
 * and this recovers it.
 *
 * ### Relation to Pulseq's autoLabel.m
 *
 * `refcode/pulseq/matlab/+mr/@Sequence/autoLabel.m` defines what these labels
 * *mean*, and that definition is followed here.  None of its control flow is:
 * it is O(total ADC samples) three times over, and each of those passes is
 * avoidable.
 *
 * | autoLabel.m | what it costs | here |
 * |---|---|---|
 * | `calculateKspacePP` over the scan | dense k at every breakpoint | never built |
 * | per-readout echo search | every sample of every readout | memoized per repeat key in the core |
 * | `ppval(gw_pp, ...)` per readout | re-evaluates a spline | `Readout::g_echo`, a by-product of the walk |
 * | `isCartesianReadout` scan | every readout against the first | three numbers per readout |
 *
 * What is left is a sort over readouts and one pass to count repetitions --
 * O(R log R) on R readouts with no sample ever touched.  The one readout whose
 * samples *are* read is `KSpace::k_central`, because the sampling pattern
 * cannot be judged from a single point; that is one readout, not R.
 *
 * Consequently @ref detect_labels takes a `KSpace` built with
 * `materialize_samples` off, and @ref auto_label builds one that way.
 *
 * ### Where this deliberately differs
 *
 * - **`kSpaceCenterLine` / `kSpaceCenterPartition` are indexed correctly.**
 *   `autoLabel.m:389` reads `labels.LIN(cCentralReadout)`, but when a noise
 *   scan is present `labels.LIN` has been padded at the front while
 *   `cCentralReadout` counts from the first *signal* readout -- so it reads a
 *   value that belongs to another readout.  Here both count the same thing.
 * - **The repetition counter is a hash map, not a dense array.**  MATLAB
 *   allocates `prod(kindex_end)` slots per slice; on a 3D scan with a large
 *   index range that is an allocation proportional to the bounding box of
 *   k-space rather than to the samples in it.
 * - **`SLC` is a geometric index.**  Slices are ranked by position, so
 *   `SlicePositions[SLC]` is where slice `SLC` is and an interleaved
 *   acquisition does not hand the recon a shuffled stack.  MATLAB computes
 *   this ordering (`sliceCountersSorted`) and then labels with the
 *   acquisition-order one.  Nothing is assumed about the prescription: gaps,
 *   uneven spacing and arbitrary orderings all come out of the frequency
 *   offsets themselves.
 * - **Slice positions group within a tolerance** rather than by exact
 *   equality, so two excitations of one slice cannot land in separate groups
 *   over a last-bit difference.  They are also drawn from the excitations that
 *   were actually acquired, so a dummy TR cannot add a slice the recon then
 *   allocates and never fills.
 * - **`kSpaceCenterSample` is quoted after mirroring.**  A bipolar train's two
 *   polarities put their echo one sample apart, so a single number for the
 *   scan is only true in one frame; the frame chosen is the one a
 *   reconstruction reads it in, once `REV` has been honoured.
 * - **Dimensions the trajectory cannot see are named by the caller**, not
 *   guessed -- see @ref AutoLabelOptions::repeat_dims.
 * - **`aux.kReadout` is not produced.**  It exists in MATLAB to draw the
 *   bipolar-alignment figure and is not among the fields that become sequence
 *   definitions.
 * - **`SliceThickness` and `SliceGap` are measured, never guessed.**  Both
 *   come from the RF pulse's spectral bandwidth over the slice-select
 *   amplitude, through `pulseq_rf.h` -- the same estimator pulseg's RF
 *   statistics now use, lifted to the pulseq level so there is one of it.
 *   Where MATLAB has no bandwidth it reports nothing; where this has none it
 *   reports nothing either, rather than substituting the `3.12 / duration`
 *   rule of thumb, which for a hard pulse is out by a factor of 2.6.
 *
 * Non-Cartesian sequences are refused, exactly as MATLAB refuses them
 * (`autoLabel.m:274`): the labels being derived are Cartesian encoding
 * counters, and there is no honest value for them when the readouts do not
 * share a direction.
 */

#ifndef PULSEQ_CXX_AUTOLABEL_HPP
#define PULSEQ_CXX_AUTOLABEL_HPP

#include <array>
#include <string>
#include <utility>
#include <vector>

#include "pulseq/kspace.hpp"

namespace pulseq
{

    class Sequence;

    /**
     * One dimension the trajectory cannot see, named by the caller.
     *
     * Where a readout sits in k says which line, partition and slice it
     * belongs to.  It says nothing about which echo of a multi-echo train it
     * is, which frame of a time series, which saturation state of a
     * magnetisation-transfer pair, or which average -- all of those revisit the
     * *same* k-space position, and by the trajectory alone they are
     * indistinguishable.  They are exactly what the repetition counter lumps
     * together: a scan with two echoes and ten frames gives `REP` running
     * 0..19, which is arithmetic rather than an encoding.
     *
     * A name splits that count back apart.  Only the names are needed --
     * `{"REP", "ECO"}`, **outermost loop first** -- because the *shape* of the
     * nest is written in the acquisition order and can be read back:
     * @ref detect_labels measures how far apart a k-space position's repeat
     * visits are, and a dimension nested inside the k-space loop revisits
     * after a short stride where one outside it revisits after a whole pass.
     * Two echoes inside a TR then ten frames over the scan produces strides
     * `1, 1, ..., L, 1, 1, ..., L` and there is only one nest that makes.
     *
     * ### What it refuses
     *
     * The reading is deliberately narrow, because a wrong split is not a
     * degraded answer -- it puts two different acquisitions in one slot.  It
     * requires the repeats to form a **rectangle**: every k-space position
     * visited the same number of times, in the same pattern.  A scan where
     * some positions are revisited and others are not has no nest to read
     * (three navigators on one line are not a dimension), and that raises
     * rather than resolving to something plausible.  So does a product that
     * does not match the repeats found, and so does a size that contradicts
     * one you gave explicitly.
     *
     * With one name there is nothing to read: it takes the whole count, and
     * the answer is the old `REP` renamed.
     */
    struct RepeatDim
    {
        /**
         * A label name -- `ECO`, `SET`, `PHS`, `AVG`, `REP`, `SEG` -- or any
         * other; a name Pulseq does not define is appended to the sequence's
         * own table, as `Sequence::label_id` does for a research label.
         *
         * The counters this derives for itself (`NOISE`, `SLC`, `REV`, `LIN`,
         * `PAR`) are refused: they mean something already.
         */
        std::string name;

        /**
         * How many values it takes, or **0 to read it from the acquisition
         * order** -- which is the ordinary case and what a bare name means.
         *
         * Give a number only to pin one down: it is then checked against what
         * was read rather than trusted, so a disagreement is reported instead
         * of silently preferring one of the two.
         */
        int size = 0;
    };

    /** How the trajectory is read before the counters are derived. */
    struct AutoLabelOptions
    {
        /** 1-based inclusive; 0 for `last_block` means "to the end". */
        int first_block = 1;
        int last_block = 0;

        /** Negate k, the slice positions and the gradients on these axes. */
        std::array<bool, 3> reflect{{false, false, false}};

        /**
         * Permutation applied to the axes, as source indices: `{1, 0, 2}`
         * swaps x and y.  Applied after `reflect`, matching MATLAB's order.
         */
        std::array<int, 3> reorder{{0, 1, 2}};

        /** Per-axis gradient timing compensation, seconds. */
        std::array<double, 3> trajectory_delay{{0.0, 0.0, 0.0}};

        /**
         * How close two readout directions must be to count as the same
         * one, after sign normalisation.  MATLAB's threshold, and its
         * comment questioning it, are both inherited: it is compared against
         * a difference of gradient amplitudes in Hz/m, so it is really an
         * exact-equality test with room for rounding.
         */
        double cartesian_tolerance = 1e-6;

        /**
         * How the repetition count decomposes, **outermost loop first**; see
         * @ref RepeatDim.
         *
         * Empty -- the default -- leaves it as a single `REP`, which is what
         * `autoLabel.m` produces and is correct for a scan that really does
         * just repeat.
         */
        std::vector<RepeatDim> repeat_dims;

        /**
         * Counters to leave alone: derived neither into the result nor onto
         * the sequence.
         *
         * For the sequence that labelled some of its own axes as it was built
         * and wants the geometric ones filled in around them.  Labels this
         * does not derive at all -- `ECO`, `SET`, `AVG`, anything custom --
         * already survive an @ref apply_labels pass untouched, so they need no
         * mention here.  `REP` is the one that needs it: it is derived by
         * default, and a design loop that separated its own contrasts or
         * frames has already said something better than "these are repeats".
         *
         * Names must be counters this actually derives (`NOISE`, `SLC`,
         * `REV`, `LIN`, `PAR`, `REP`); anything else is a typo rather than a
         * no-op, and is refused.
         *
         * A definition that only means something in terms of a skipped
         * counter goes with it -- `kSpaceCenterLine` without `LIN` would
         * describe an index nothing wrote.
         */
        std::vector<std::string> skip;
    };

    /**
     * One counter's value at every ADC in range.
     *
     * A label is present only when it varies -- a single-slice scan has no
     * `SLC` -- which is the same rule `autoLabel.m` applies and the reason a
     * caller has to ask whether a field is there before reading it.
     */
    struct AutoLabels
    {
        /** 1-based block index of each ADC, in acquisition order. */
        std::vector<int> adc_block;

        /** Empty when the label does not vary; otherwise one value per ADC. */
        std::vector<int> noise;
        std::vector<int> slc;
        std::vector<int> rev;
        std::vector<int> lin;
        std::vector<int> par;

        /**
         * The repetition counter, undivided.  Empty when
         * @ref AutoLabelOptions::repeat_dims named the dimensions it stands
         * for, in which case they are in @ref named instead.
         */
        std::vector<int> rep;

        /**
         * The declared dimensions, in declaration order, under the names the
         * caller gave them.  Same emptiness rule as the rest: a dimension that
         * never leaves zero is not written.
         */
        std::vector<std::pair<std::string, std::vector<int>>> named;

        /** The labels present, in the order they are written. */
        std::vector<std::pair<std::string, const std::vector<int>*>> present() const;
    };

    /** The `[DEFINITIONS]` half of the answer. */
    struct AutoLabelAux
    {
        bool has_center_line = false;
        int center_line = 0;
        bool has_center_partition = false;
        int center_partition = 0;

        /**
         * Echo sample of the readout through the k-space centre, 0-based,
         * in the frame that follows mirroring the reversed lines.
         *
         * One number has to stand for a scan whose polarities disagree about
         * it by one sample, and this is the frame the disagreement vanishes
         * in: a fully sampled 128-point readout reports 64 whichever polarity
         * happens to pass closest to the origin.
         */
        bool has_center_sample = false;
        int center_sample = 0;

        /**
         * Slice offsets in metres, **ascending**, so `slice_positions[SLC]` is
         * where slice `SLC` sits.  Empty unless there is more than one.
         */
        std::vector<double> slice_positions;

        /**
         * Excited slab thickness in metres: the pulse's spectral bandwidth
         * over the slice-select gradient amplitude.
         *
         * Absent when the excitation carries no gradient (a hard pulse is not
         * slice-selective) or when the spectrum is not measurable.  Never
         * substituted with the `3.12 / duration` rule of thumb -- an
         * unmeasured value and a measured one must not look the same here.
         */
        bool has_slice_thickness = false;
        double slice_thickness = 0.0;

        /**
         * Space between two adjacent slices, metres: their closest spacing
         * less one thickness.  Negative for overlapping slices, which is a
         * real prescription and reported as one.
         */
        bool has_slice_gap = false;
        double slice_gap = 0.0;

        /**
         * `[rise, flat, fall, adc_delay - grad_delay, adc_duration]`, seconds.
         *
         * Only when the central readout samples non-uniformly along k *and*
         * sits on a trapezoid -- that is, ramp sampling, where a gridder needs
         * the ramp to undo it.
         */
        bool has_gridding = false;
        std::array<double, 5> trapezoid_gridding{{0.0, 0.0, 0.0, 0.0, 0.0}};
        int target_gridded_samples = 0;
    };

    struct AutoLabelResult
    {
        AutoLabels labels;
        AutoLabelAux aux;

        /** Distinct repeat keys over the readouts; see `KSpace::key_groups`. */
        int key_groups = 0;
        int num_readouts = 0;
    };

    /**
     * Derive the counters from a trajectory that has already been computed.
     *
     * @p ks must carry `center_sample` (so `derive_center_sample` was on) and
     * `k_central`; `k_adc` is not read.  @p seq is consulted only for the
     * gridding parameters, which are a property of the events rather than of
     * the trajectory.
     *
     * @throws std::runtime_error when the readouts do not share a direction.
     */
    AutoLabelResult detect_labels(const KSpace& ks, const Sequence& seq,
                                  const AutoLabelOptions& options = {});

    /**
     * Write @p labels onto @p seq as `LABELSET` extensions, and @p aux into
     * `[DEFINITIONS]`.
     *
     * A label is emitted only where its value differs from the previous ADC,
     * which is what `LABELSET` semantics mean and what keeps the extension
     * library small.  Existing extensions on a block are preserved: the new
     * links are prepended to whatever chain the block already had, except that
     * a `LABELSET` for a counter being written here is dropped first, so
     * running this twice does not leave two of them.
     */
    void apply_labels(Sequence& seq, const AutoLabels& labels, const AutoLabelAux& aux);

    /**
     * Compute the trajectory, derive the counters, and apply them.
     *
     * @param apply false to detect only, leaving @p seq untouched.
     */
    AutoLabelResult auto_label(Sequence& seq, const AutoLabelOptions& options = {},
                               bool apply = true);

}  // namespace pulseq

#endif /* PULSEQ_CXX_AUTOLABEL_HPP */
