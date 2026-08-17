/**
 * Tests for pulseq::expand_repeats -- the repetitions an interpreter would
 * play, written into the block table instead.
 *
 * The sequences here are built rather than loaded, because what is under test
 * is the block *order*, and a fixture would only assert it against itself.  A
 * block's identity is carried in its duration, which is unique per block and
 * survives every operation this performs, so the assertions read as the order
 * of durations a scanner would see.
 *
 * The load-bearing property is that the expansion is what the interpreter's
 * own rule produces: `ONCE == 1` on the first pass, `ONCE == 2` on the last,
 * `ONCE == 0` on all of them.  Everything else -- the labels stripped, the
 * counter stamped, the definition written -- follows from having done that.
 */

#include <gtest/gtest.h>

#include "pulseq/expand.hpp"
#include "pulseq/sequence.hpp"

#include <map>
#include <string>
#include <vector>

namespace
{

    /** A block carrying nothing but a duration and, optionally, one label. */
    int add_labelled(pulseq::Sequence& seq, double duration, const std::string& label, int value)
    {
        pulseq::Block block;
        block.duration = duration;
        if (!label.empty())
        {
            const int row = seq.register_label_set(static_cast<int32_t>(value),
                                                   static_cast<int32_t>(seq.label_id(label)));
            block.ext = static_cast<int32_t>(seq.chain_extension(
                static_cast<int32_t>(seq.extension_type_id("LABELSET")),
                static_cast<int32_t>(row), 0));
        }
        return seq.add_block(block);
    }

    /** The durations in play order -- the sequence as a scanner would meet it. */
    std::vector<double> order(const pulseq::Sequence& seq)
    {
        std::vector<double> out;
        for (int b = 1; b <= seq.num_blocks(); ++b)
            out.push_back(seq.get_block(b).duration);
        return out;
    }

    /** Every label in force at each block, replayed as a reader replays them. */
    std::vector<std::map<std::string, int>> replay(const pulseq::Sequence& seq)
    {
        const int labelset = seq.find_extension_type_id("LABELSET");
        std::vector<std::map<std::string, int>> out;
        std::map<std::string, int> state;
        for (int b = 1; b <= seq.num_blocks(); ++b)
        {
            for (int32_t link = seq.get_block(b).ext; link != 0;)
            {
                const int32_t* row = seq.extensions_library().row(link);
                if (labelset != 0 && row[0] == labelset && row[1] != 0)
                {
                    const int32_t* label = seq.label_set_library().row(row[1]);
                    state[seq.label_name(label[1])] = label[0];
                }
                link = row[2];
            }
            out.push_back(state);
        }
        return out;
    }

    /**
     * Preparation, two body blocks, cooldown.  Durations 1, 2, 3, 4 ms, so
     * "which block" and "which duration" are the same question.
     */
    pulseq::Sequence prep_body_cooldown()
    {
        pulseq::Sequence seq;
        add_labelled(seq, 1e-3, "ONCE", 1);
        add_labelled(seq, 2e-3, "ONCE", 0);
        add_labelled(seq, 3e-3, "", 0);
        add_labelled(seq, 4e-3, "ONCE", 2);
        return seq;
    }

}  // namespace

TEST(PulseqExpand, ThePreparationLeadsAndTheCooldownTrails)
{
    pulseq::Sequence seq = prep_body_cooldown();
    const pulseq::ExpandResult report = pulseq::expand_repeats(seq, 3);

    EXPECT_EQ(report.prep_blocks, 1);
    EXPECT_EQ(report.body_blocks, 2);
    EXPECT_EQ(report.cooldown_blocks, 1);
    EXPECT_EQ(report.blocks_before, 4);
    EXPECT_EQ(report.blocks_after, 1 + 3 * 2 + 1);

    /* 1 once, then the body three times, then 4 once. */
    const std::vector<double> expected{1e-3, 2e-3, 3e-3, 2e-3, 3e-3, 2e-3, 3e-3, 4e-3};
    EXPECT_EQ(order(seq), expected);
}

TEST(PulseqExpand, ASingleRepeatKeepsBothEnds)
{
    /* One pass is both the first and the last, so nothing is dropped -- the
     * table is unchanged apart from the flags being resolved away. */
    pulseq::Sequence seq = prep_body_cooldown();
    const pulseq::ExpandResult report = pulseq::expand_repeats(seq, 1);

    EXPECT_EQ(report.blocks_after, 4);
    const std::vector<double> expected{1e-3, 2e-3, 3e-3, 4e-3};
    EXPECT_EQ(order(seq), expected);
}

TEST(PulseqExpand, TheOnceFlagsAreGoneAndTheAverageCounterIsThere)
{
    pulseq::Sequence seq = prep_body_cooldown();
    pulseq::expand_repeats(seq, 3);

    const std::vector<std::map<std::string, int>> labels = replay(seq);
    ASSERT_EQ(labels.size(), 8u);
    for (size_t i = 0; i < labels.size(); ++i)
        EXPECT_EQ(labels[i].count("ONCE"), 0u) << "ONCE survived at block " << i;

    /* AVG counts the repetition, and the preparation belongs to the first --
     * it is played once, at the front, and its data is not a separate average. */
    const int expected_avg[8] = {0, 0, 0, 1, 1, 2, 2, 2};
    for (size_t i = 0; i < labels.size(); ++i)
    {
        const std::map<std::string, int>::const_iterator it = labels[i].find("AVG");
        const int value = (it == labels[i].end()) ? 0 : it->second;
        EXPECT_EQ(value, expected_avg[i]) << "at block " << i;
    }
}

TEST(PulseqExpand, KeepingTheFlagsLeavesAScanShapedLikeOnePass)
{
    /* The reason `strip_once = false` is offered: the expanded table is 1 at
     * the front, 0 through the middle and 2 at the end, which is exactly what
     * a single-pass sequence looks like -- so an interpreter that reads the
     * preparation and cooldown boundaries off these flags still finds them. */
    pulseq::Sequence seq = prep_body_cooldown();
    pulseq::ExpandOptions options;
    options.strip_once = false;
    pulseq::expand_repeats(seq, 3, options);

    const std::vector<std::map<std::string, int>> labels = replay(seq);
    ASSERT_EQ(labels.size(), 8u);
    const int expected_once[8] = {1, 0, 0, 0, 0, 0, 0, 2};
    for (size_t i = 0; i < labels.size(); ++i)
        EXPECT_EQ(labels[i].at("ONCE"), expected_once[i]) << "at block " << i;
}

TEST(PulseqExpand, ASequenceWithNoFlagsRepeatsWhole)
{
    pulseq::Sequence seq;
    add_labelled(seq, 1e-3, "", 0);
    add_labelled(seq, 2e-3, "", 0);
    const pulseq::ExpandResult report = pulseq::expand_repeats(seq, 2);

    EXPECT_EQ(report.prep_blocks, 0);
    EXPECT_EQ(report.body_blocks, 2);
    const std::vector<double> expected{1e-3, 2e-3, 1e-3, 2e-3};
    EXPECT_EQ(order(seq), expected);
}

TEST(PulseqExpand, TheFlagsAreReadPerBlockNotAsSections)
{
    /* Preparation that is not one contiguous run at the front.  An
     * interpreter plays the block list and asks each block whether it
     * qualifies, so a design that reopens its ONCE section midway expands the
     * way it would play rather than the way a section-shaped reading would
     * guess. */
    pulseq::Sequence seq;
    add_labelled(seq, 1e-3, "ONCE", 1);
    add_labelled(seq, 2e-3, "ONCE", 0);
    add_labelled(seq, 3e-3, "ONCE", 1);
    add_labelled(seq, 4e-3, "ONCE", 0);
    pulseq::expand_repeats(seq, 2);

    const std::vector<double> expected{1e-3, 2e-3, 3e-3, 4e-3, 2e-3, 4e-3};
    EXPECT_EQ(order(seq), expected);
}

TEST(PulseqExpand, OtherExtensionsOnAStrippedBlockSurvive)
{
    /* A rotation sharing a block with an ONCE label is not this pass's to
     * discard, and neither is a counter the design set for itself. */
    pulseq::Sequence seq;
    const double rotation[pulseq::ROTATION_WIDTH] = {0.0, 0.0, 0.0, 1.0};
    const int rotation_id = seq.register_rotation(rotation);

    pulseq::Block block;
    block.duration = 1e-3;
    int32_t head = static_cast<int32_t>(seq.chain_extension(
        static_cast<int32_t>(seq.extension_type_id("ROTATIONS")),
        static_cast<int32_t>(rotation_id), 0));
    head = static_cast<int32_t>(seq.chain_extension(
        static_cast<int32_t>(seq.extension_type_id("LABELSET")),
        static_cast<int32_t>(seq.register_label_set(7, static_cast<int32_t>(seq.label_id("ECO")))),
        head));
    head = static_cast<int32_t>(seq.chain_extension(
        static_cast<int32_t>(seq.extension_type_id("LABELSET")),
        static_cast<int32_t>(seq.register_label_set(1, static_cast<int32_t>(seq.label_id("ONCE")))),
        head));
    block.ext = head;
    seq.add_block(block);
    add_labelled(seq, 2e-3, "ONCE", 0);

    pulseq::expand_repeats(seq, 2);

    const int rotations = seq.find_extension_type_id("ROTATIONS");
    int found_rotations = 0;
    for (int32_t link = seq.get_block(1).ext; link != 0;)
    {
        const int32_t* row = seq.extensions_library().row(link);
        if (row[0] == rotations)
            ++found_rotations;
        link = row[2];
    }
    EXPECT_EQ(found_rotations, 1);
    EXPECT_EQ(replay(seq)[0].at("ECO"), 7);
    EXPECT_EQ(replay(seq)[0].count("ONCE"), 0u);
}

/*
 * Expanding says nothing about itself in `[DEFINITIONS]`.
 *
 * Whether a sequence carries its own repetitions is the design's decision and
 * the block table already shows it: a subsequence meant to be acquired once
 * is simply not expanded. A definition claiming it would be a second place
 * for the same fact to be wrong.
 */
TEST(PulseqExpand, ExpandingWritesNoDefinition)
{
    pulseq::Sequence seq = prep_body_cooldown();
    const size_t before = seq.definitions().size();

    pulseq::expand_repeats(seq, 3);

    EXPECT_EQ(seq.definitions().size(), before);
    EXPECT_EQ(seq.definition("IgnoreAverages"), nullptr);
}

TEST(PulseqExpand, ACounterTheSequenceAlreadyWritesIsRefused)
{
    /* Two meanings on one counter is not something to resolve quietly: the
     * design said what its AVG axis was, and this would overwrite it. */
    pulseq::Sequence seq;
    add_labelled(seq, 1e-3, "AVG", 0);
    add_labelled(seq, 2e-3, "AVG", 1);
    EXPECT_THROW(pulseq::expand_repeats(seq, 2), std::runtime_error);

    /* Under another name there is no collision, and the design's own AVG is
     * left exactly as it was. */
    pulseq::Sequence other;
    add_labelled(other, 1e-3, "AVG", 0);
    add_labelled(other, 2e-3, "AVG", 1);
    pulseq::ExpandOptions options;
    options.label = "REP";
    EXPECT_NO_THROW(pulseq::expand_repeats(other, 2, options));
    const std::vector<std::map<std::string, int>> labels = replay(other);
    EXPECT_EQ(labels[0].at("AVG"), 0);
    EXPECT_EQ(labels[3].at("AVG"), 1);
    EXPECT_EQ(labels[3].at("REP"), 1);
}

TEST(PulseqExpand, AnEmptyLabelStampsNothing)
{
    pulseq::Sequence seq;
    add_labelled(seq, 1e-3, "AVG", 4);
    pulseq::ExpandOptions options;
    options.label = "";
    EXPECT_NO_THROW(pulseq::expand_repeats(seq, 3, options));
    EXPECT_EQ(seq.num_blocks(), 3);
    /* The design's own AVG is untouched throughout. */
    const std::vector<std::map<std::string, int>> labels = replay(seq);
    for (size_t i = 0; i < labels.size(); ++i)
        EXPECT_EQ(labels[i].at("AVG"), 4);
}

TEST(PulseqExpand, NothingToRepeatIsRefused)
{
    pulseq::Sequence seq;
    add_labelled(seq, 1e-3, "ONCE", 1);
    add_labelled(seq, 2e-3, "ONCE", 2);
    EXPECT_THROW(pulseq::expand_repeats(seq, 2), std::runtime_error);
    /* One pass is still meaningful: there is nothing to drop. */
    EXPECT_NO_THROW(pulseq::expand_repeats(seq, 1));
    EXPECT_EQ(seq.num_blocks(), 2);
}

TEST(PulseqExpand, AFourthOnceStateIsRefused)
{
    pulseq::Sequence seq;
    add_labelled(seq, 1e-3, "ONCE", 3);
    EXPECT_THROW(pulseq::expand_repeats(seq, 2), std::runtime_error);
}

TEST(PulseqExpand, FewerThanOneRepeatIsRefused)
{
    pulseq::Sequence seq = prep_body_cooldown();
    EXPECT_THROW(pulseq::expand_repeats(seq, 0), std::invalid_argument);
}

TEST(PulseqExpand, TheEventLibrariesDoNotGrow)
{
    /* The whole point of doing this on the block table: a repetition plays the
     * same events, so the file grows by block rows and by nothing else. */
    pulseq::Sequence seq;
    const double trap[pulseq::TRAP_WIDTH] = {1000.0, 100e-6, 500e-6, 100e-6, 0.0};
    const int gx = seq.register_trap(trap);
    for (int i = 0; i < 4; ++i)
    {
        pulseq::Block block;
        block.duration = 700e-6;
        block.gx = static_cast<int32_t>(gx);
        seq.add_block(block);
    }
    const int traps_before = seq.trap_library().size();
    const int shapes_before = seq.shape_library().size();

    pulseq::expand_repeats(seq, 10);

    EXPECT_EQ(seq.num_blocks(), 40);
    EXPECT_EQ(seq.trap_library().size(), traps_before);
    EXPECT_EQ(seq.shape_library().size(), shapes_before);
    /* One LABELSET row per repetition index that is written -- nine, since the
     * first repetition is the default value and writes nothing. */
    EXPECT_EQ(seq.label_set_library().size(), 9);
}
