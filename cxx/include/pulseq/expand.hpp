/**
 * @file expand.hpp
 * @brief Playing a sequence more than once, resolved into the block table.
 *
 * A Pulseq file describes one pass.  Playing it N times is left to the
 * interpreter, which reads a repeat count from somewhere outside the file --
 * on a GE scanner, the `opnex` knob -- and replays the block table, using the
 * `ONCE` flag to decide what belongs to a single pass and what belongs to the
 * whole scan:
 *
 * | `ONCE` | plays |
 * |---|---|
 * | `1` | on the first repetition only -- preparation, a startup transient |
 * | `0` | on every repetition -- the body of the scan |
 * | `2` | on the last repetition only -- cooldown, a ramp-down |
 *
 * That is a small amount of arithmetic in exchange for a sequence whose
 * playing order is not written anywhere in it.  This does the arithmetic here
 * instead, and writes the answer down: after @ref expand_repeats the block
 * table *is* the scan, every repetition is present in the order it plays, and
 * nothing downstream has to be told how many times to read it.
 *
 * ### What it costs and what it buys
 *
 * The block table grows by roughly the repeat count -- and only the block
 * table.  Every event library is untouched, because a repetition plays the
 * *same* events: a hundred-thousand-block scan repeated three times is three
 * hundred thousand rows of six integers and a double, and not one extra
 * gradient.  Deduplication has nothing left to find, so it need not be run
 * again.
 *
 * What it buys is that the file stops depending on a number that is not in it.
 * A `.seq` written this way plays identically under any interpreter, including
 * one that has never heard of `ONCE`, and the repetition index is a label a
 * reconstruction can sort by rather than something the interpreter has to
 * synthesise on the way past.
 *
 * Whether a sequence carries its repetitions is the design's decision and is
 * stated by whether this ran at all: a subsequence that should be acquired
 * once -- an EPI calibration, say -- simply is not expanded.  Nothing is
 * written into the file to say so, because the file already shows it.
 *
 * ### On removing the flags
 *
 * @ref ExpandOptions::strip_once drops the `ONCE` labels once they have been
 * resolved: the flags would otherwise say something that is no longer true
 * of the table they sit in.  A Pulserver interpreter gives `ONCE` no
 * structural meaning -- its segmentation and TR detection read only the
 * block content -- so nothing downstream depends on keeping them.
 */

#ifndef PULSEQ_CXX_EXPAND_HPP
#define PULSEQ_CXX_EXPAND_HPP

#include <string>

namespace pulseq
{

    class Sequence;

    /** How @ref expand_repeats writes the repetitions down. */
    struct ExpandOptions
    {
        /**
         * The counter stamped with the repetition index, or empty for none.
         *
         * `AVG` by default, because the repetition an interpreter adds is a
         * signal average: the same acquisition, sampled again.  `REP` is the
         * frame counter of a dynamic series and means something else, and a
         * design that acquires frames should be labelling them itself.
         *
         * Emitted where it changes, as `LABELSET` semantics mean: one
         * extension per repetition, not one per block.  If the sequence
         * already writes this counter the expansion is **refused** -- two
         * meanings on one counter is not something to resolve quietly.
         */
        std::string label = "AVG";

        /** Drop the `ONCE` flags once resolved.  See the file comment. */
        bool strip_once = true;
    };

    /** What the expansion found and what it wrote. */
    struct ExpandResult
    {
        int repeats = 1;
        int blocks_before = 0;
        int blocks_after = 0;

        /** The `ONCE == 1`, `ONCE == 0` and `ONCE == 2` block counts, per pass. */
        int prep_blocks = 0;
        int body_blocks = 0;
        int cooldown_blocks = 0;
    };

    /**
     * Play @p seq @p repeats times, written into its block table.
     *
     * Each repetition emits the blocks that qualify for it, **in file order**
     * -- the flags are read as a per-block property rather than as section
     * delimiters, so a sequence whose preparation is not one contiguous run at
     * the front expands the same way an interpreter would play it.
     *
     * `repeats == 1` is not a no-op: it resolves the flags and strips them if
     * asked, leaving a file whose block table is the whole of what plays.
     *
     * @throws std::invalid_argument if @p repeats is below 1.
     * @throws std::runtime_error if a block carries an `ONCE` value outside
     *         `{0, 1, 2}`, if @p repeats exceeds 1 with every block marked
     *         `ONCE` (there is no body to repeat), or if the sequence already
     *         writes @ref ExpandOptions::label.
     */
    ExpandResult expand_repeats(Sequence& seq, int repeats, const ExpandOptions& options = {});

}  // namespace pulseq

#endif /* PULSEQ_CXX_EXPAND_HPP */
