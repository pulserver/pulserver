/**
 * Unit tests for Sequence::remove_duplicates().
 *
 * Deduplication is the pass that decides the file, and it is the one pass
 * whose mistakes are invisible in the output: a library that failed to
 * collapse only makes a bigger `.seq`, and a library that collapsed too far
 * makes a wrong one that still parses.  So the properties pinned here are the
 * ones neither a round trip nor a byte comparison would catch -- what counts
 * as the same event, what the surviving row holds, and whether everything
 * pointing at a renumbered row followed it.
 */

#include <gtest/gtest.h>

#include "pulseq/read.hpp"
#include "pulseq/sequence.hpp"
#include "pulseq/write.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

using pulseq::Block;
using pulseq::GradKind;
using pulseq::Sequence;
using pulseq::SoftDelay;

namespace
{
std::array<double, pulseq::TRAP_WIDTH> trap(double amplitude, double delay = 0.0)
{
    return {amplitude, 100e-6, 500e-6, 100e-6, delay};
}

std::array<double, pulseq::ARB_WIDTH> arb(double amplitude, double shape_id)
{
    return {amplitude, 0.0, 0.0, shape_id, 0.0, 0.0};
}

std::array<double, pulseq::ADC_WIDTH> adc(double samples, double phase_shape = 0.0)
{
    return {samples, 1e-5, 0.0, 0.0, 0.0, 0.0, 0.0, phase_shape};
}

/** A block that plays gx and nothing else. */
Block on_x(int32_t grad, double duration = 1e-3)
{
    Block block;
    block.gx = grad;
    block.duration = duration;
    return block;
}
} // namespace

/* ================================================================== */
/*  What counts as the same event                                     */
/* ================================================================== */

TEST(PulseqDedup, IdenticalRowsCollapseAndBlocksFollow)
{
    Sequence seq;
    const auto row = trap(1000.0);
    const int a = seq.register_trap(row.data());
    const int b = seq.register_trap(row.data());
    const int c = seq.register_trap(row.data());
    seq.add_block(on_x(a));
    seq.add_block(on_x(b));
    seq.add_block(on_x(c));

    seq.remove_duplicates();

    EXPECT_EQ(seq.trap_library().size(), 1);
    EXPECT_EQ(seq.num_gradients(), 1);
    EXPECT_EQ(seq.get_block(1).gx, 1);
    EXPECT_EQ(seq.get_block(2).gx, 1);
    EXPECT_EQ(seq.get_block(3).gx, 1);
}

TEST(PulseqDedup, RowsAgreeingOnlyBeyondTheFilesPrecisionAreOneRow)
{
    // A trapezoid amplitude is written `%12g` -- six significant digits -- so
    // a difference in the seventh could not have survived being written and
    // must not survive here either.
    Sequence seq;
    const auto coarse = trap(1000.0);
    const auto finer = trap(1000.0000001);
    seq.add_block(on_x(seq.register_trap(coarse.data())));
    seq.add_block(on_x(seq.register_trap(finer.data())));

    seq.remove_duplicates();

    EXPECT_EQ(seq.trap_library().size(), 1);
    EXPECT_EQ(seq.get_block(2).gx, seq.get_block(1).gx);
}

TEST(PulseqDedup, RowsTheFileCouldTellApartStayApart)
{
    Sequence seq;
    const auto one = trap(1000.0);
    const auto other = trap(1001.0);
    seq.add_block(on_x(seq.register_trap(one.data())));
    seq.add_block(on_x(seq.register_trap(other.data())));

    seq.remove_duplicates();

    EXPECT_EQ(seq.trap_library().size(), 2);
    EXPECT_EQ(seq.get_block(1).gx, 1);
    EXPECT_EQ(seq.get_block(2).gx, 2);
}

TEST(PulseqDedup, TheSurvivingRowIsTheRoundedOne)
{
    // Not the original.  Were the original kept, writing the sequence and
    // reading it back would give different numbers from the ones deduplication
    // decided were the same, and the second write would not match the first.
    Sequence seq;
    const auto row = trap(1000.0000001);
    seq.add_block(on_x(seq.register_trap(row.data())));

    seq.remove_duplicates();

    ASSERT_EQ(seq.trap_library().size(), 1);
    EXPECT_DOUBLE_EQ(seq.trap_library().row(1)[0], 1000.0);
}

TEST(PulseqDedup, NegativeZeroIsFoldedOntoZero)
{
    // The same number, differing only in the bits a hash looks at: a phase
    // encode scaled by minus nothing is not an event of its own.
    Sequence seq;
    const auto positive = trap(0.0);
    const auto negative = trap(-0.0);
    seq.add_block(on_x(seq.register_trap(positive.data())));
    seq.add_block(on_x(seq.register_trap(negative.data())));

    seq.remove_duplicates();

    EXPECT_EQ(seq.trap_library().size(), 1);
    EXPECT_FALSE(std::signbit(seq.trap_library().row(1)[0]));
}

TEST(PulseqDedup, DelaysCollapseAtTheMicrosecondTheFileWrites)
{
    // The ramp and delay columns are written as whole microseconds, so they
    // round to decimal places in seconds rather than to significant digits.
    Sequence seq;
    const auto on_raster = trap(1000.0, 20e-6);
    const auto a_hair_off = trap(1000.0, 20e-6 + 1e-12);
    const auto a_step_off = trap(1000.0, 21e-6);
    seq.add_block(on_x(seq.register_trap(on_raster.data())));
    seq.add_block(on_x(seq.register_trap(a_hair_off.data())));
    seq.add_block(on_x(seq.register_trap(a_step_off.data())));

    seq.remove_duplicates();

    EXPECT_EQ(seq.trap_library().size(), 2);
    EXPECT_EQ(seq.get_block(2).gx, seq.get_block(1).gx);
    EXPECT_NE(seq.get_block(3).gx, seq.get_block(1).gx);
}

TEST(PulseqDedup, RfUseSeparatesOtherwiseIdenticalPulses)
{
    // `use` decides identity but is stored beside the row, so it has to be
    // keyed on: an excitation and a refocusing pulse of the same shape and
    // amplitude are two events.
    Sequence seq;
    std::array<double, pulseq::RF_WIDTH> row{};
    row[0] = 1000.0;

    Block first;
    first.rf = seq.register_rf(row.data(), 'e');
    first.duration = 1e-3;
    Block second;
    second.rf = seq.register_rf(row.data(), 'r');
    second.duration = 1e-3;
    Block third;
    third.rf = seq.register_rf(row.data(), 'e');
    third.duration = 1e-3;
    seq.add_block(first);
    seq.add_block(second);
    seq.add_block(third);

    seq.remove_duplicates();

    ASSERT_EQ(seq.rf_library().size(), 2);
    ASSERT_EQ(seq.rf_uses().size(), 2u);
    EXPECT_EQ(seq.rf_uses()[0], 'e');
    EXPECT_EQ(seq.rf_uses()[1], 'r');
    EXPECT_EQ(seq.get_block(1).rf, 1);
    EXPECT_EQ(seq.get_block(2).rf, 2);
    EXPECT_EQ(seq.get_block(3).rf, 1);
}

TEST(PulseqDedup, AdcPhaseModulationsFollowTheirShapes)
{
    Sequence seq;
    const std::vector<double> flat{0.0, 0.0, 0.0};
    const std::vector<double> ramp{0.0, 1.0, 0.0};
    const double first = static_cast<double>(seq.register_shape(3, flat.data(), 3));
    const double second = static_cast<double>(seq.register_shape(3, ramp.data(), 3));
    const double duplicate = static_cast<double>(seq.register_shape(3, ramp.data(), 3));

    const auto plain = adc(64, first);
    const auto modulated = adc(64, second);
    const auto same_again = adc(64, duplicate);
    Block a;
    a.adc = seq.register_adc(plain.data());
    a.duration = 1e-3;
    Block b;
    b.adc = seq.register_adc(modulated.data());
    b.duration = 1e-3;
    Block c;
    c.adc = seq.register_adc(same_again.data());
    c.duration = 1e-3;
    seq.add_block(a);
    seq.add_block(b);
    seq.add_block(c);

    seq.remove_duplicates();

    EXPECT_EQ(seq.shape_library().size(), 2);
    ASSERT_EQ(seq.adc_library().size(), 2);
    EXPECT_DOUBLE_EQ(seq.adc_library().row(1)[7], 1.0);
    EXPECT_DOUBLE_EQ(seq.adc_library().row(2)[7], 2.0);
    EXPECT_EQ(seq.get_block(3).adc, seq.get_block(2).adc);
}

/* ================================================================== */
/*  Renumbering                                                       */
/* ================================================================== */

TEST(PulseqDedup, ShapesCollapseAndTheEventsNamingThemFollow)
{
    Sequence seq;
    const std::vector<double> samples{0.0, 1.0, 0.0};
    const int one = seq.register_shape(3, samples.data(), 3);
    const int two = seq.register_shape(3, samples.data(), 3);
    ASSERT_NE(one, two);

    // Two gradients differing only in which copy of the shape they name.
    const auto first = arb(1000.0, static_cast<double>(one));
    const auto second = arb(1000.0, static_cast<double>(two));
    seq.add_block(on_x(seq.register_arbitrary(first.data())));
    seq.add_block(on_x(seq.register_arbitrary(second.data())));

    seq.remove_duplicates();

    EXPECT_EQ(seq.shape_library().size(), 1);
    ASSERT_EQ(seq.arb_library().size(), 1);
    EXPECT_DOUBLE_EQ(seq.arb_library().row(1)[3], 1.0);
    EXPECT_EQ(seq.get_block(2).gx, seq.get_block(1).gx);
}

TEST(PulseqDedup, ShapesThatDecompressToDifferentLengthsStayApart)
{
    // The same compressed run can stand for two waveforms of different length,
    // so the sample count is part of what makes a shape a shape.
    Sequence seq;
    const std::vector<double> samples{1.0, 0.0, 10.0};
    const auto shorter =
        arb(1000.0, static_cast<double>(seq.register_shape(12, samples.data(), 3)));
    const auto longer = arb(1000.0, static_cast<double>(seq.register_shape(20, samples.data(), 3)));
    seq.add_block(on_x(seq.register_arbitrary(shorter.data())));
    seq.add_block(on_x(seq.register_arbitrary(longer.data())));

    seq.remove_duplicates();

    EXPECT_EQ(seq.shape_library().size(), 2);
    EXPECT_EQ(seq.arb_library().size(), 2);
}

TEST(PulseqDedup, GradientsKeepOneNumberingAcrossBothTables)
{
    // Trapezoids and waveforms deduplicate apart -- they cannot be equal, and
    // their rows are not the same width -- but come out numbered together, in
    // order of first appearance, because that is what the file carries.
    Sequence seq;
    const std::vector<double> samples{0.0, 1.0, 0.0};
    const double shape = static_cast<double>(seq.register_shape(3, samples.data(), 3));

    const auto t1 = trap(1000.0);
    const auto a1 = arb(2000.0, shape);
    const auto t2 = trap(3000.0);

    const int first = seq.register_trap(t1.data());
    const int second = seq.register_arbitrary(a1.data());
    const int third = seq.register_trap(t2.data());
    const int again = seq.register_arbitrary(a1.data());
    seq.add_block(on_x(first));
    seq.add_block(on_x(second));
    seq.add_block(on_x(third));
    seq.add_block(on_x(again));

    seq.remove_duplicates();

    EXPECT_EQ(seq.num_gradients(), 3);
    EXPECT_EQ(seq.trap_library().size(), 2);
    EXPECT_EQ(seq.arb_library().size(), 1);

    EXPECT_EQ(seq.get_block(1).gx, 1);
    EXPECT_EQ(seq.get_block(2).gx, 2);
    EXPECT_EQ(seq.get_block(3).gx, 3);
    EXPECT_EQ(seq.get_block(4).gx, 2);

    EXPECT_EQ(seq.grad_kind(1), GradKind::Trap);
    EXPECT_EQ(seq.grad_kind(2), GradKind::Arbitrary);
    EXPECT_EQ(seq.grad_kind(3), GradKind::Trap);
    EXPECT_DOUBLE_EQ(seq.trap_library().row(seq.grad_row(3))[0], 3000.0);
    EXPECT_DOUBLE_EQ(seq.arb_library().row(seq.grad_row(2))[0], 2000.0);
}

TEST(PulseqDedup, EveryAxisOfTheBlockTableIsRenumbered)
{
    Sequence seq;
    const auto x = trap(1000.0);
    const auto y = trap(2000.0);
    const int gx = seq.register_trap(x.data());
    const int gy = seq.register_trap(y.data());
    const int gx_again = seq.register_trap(x.data());
    const int gz = seq.register_trap(y.data());

    Block block;
    block.gx = gx_again;
    block.gy = gy;
    block.gz = gz;
    block.duration = 1e-3;
    seq.add_block(on_x(gx));
    seq.add_block(block);

    seq.remove_duplicates();

    ASSERT_EQ(seq.num_gradients(), 2);
    const Block after = seq.get_block(2);
    EXPECT_EQ(after.gx, 1);
    EXPECT_EQ(after.gy, 2);
    EXPECT_EQ(after.gz, 2);
}

TEST(PulseqDedup, NoEventStaysNoEvent)
{
    Sequence seq;
    const auto row = trap(1000.0);
    seq.add_block(on_x(seq.register_trap(row.data())));
    Block empty;
    empty.duration = 1e-3;
    seq.add_block(empty);

    seq.remove_duplicates();

    const Block after = seq.get_block(2);
    EXPECT_EQ(after.rf, 0);
    EXPECT_EQ(after.gx, 0);
    EXPECT_EQ(after.gy, 0);
    EXPECT_EQ(after.gz, 0);
    EXPECT_EQ(after.adc, 0);
    EXPECT_EQ(after.ext, 0);
}

TEST(PulseqDedup, ABlockNamingAnEventThatWasNeverRegisteredIsRejected)
{
    Sequence seq;
    seq.add_block(on_x(7));
    EXPECT_THROW(seq.remove_duplicates(), std::out_of_range);
}

/* ================================================================== */
/*  Extensions                                                        */
/* ================================================================== */

TEST(PulseqDedup, ChainsSharedByEveryBlockCollapseToOne)
{
    Sequence seq;
    const int labelset = seq.extension_type_id("LABELSET");
    const int lin = seq.label_id("LIN");

    // Three blocks setting the same label to the same value, each having
    // registered its own row for it -- which is what add_block does.
    for (int i = 0; i < 3; ++i)
    {
        Block block;
        block.duration = 1e-3;
        const int row = seq.register_label_set(4, lin);
        block.ext = seq.chain_extension(labelset, row, 0);
        seq.add_block(block);
    }

    seq.remove_duplicates();

    EXPECT_EQ(seq.label_set_library().size(), 1);
    EXPECT_EQ(seq.extensions_library().size(), 1);
    EXPECT_EQ(seq.get_block(1).ext, 1);
    EXPECT_EQ(seq.get_block(2).ext, 1);
    EXPECT_EQ(seq.get_block(3).ext, 1);
}

TEST(PulseqDedup, ATwoLinkChainKeepsItsOrderAndItsReferences)
{
    Sequence seq;
    const int labelset = seq.extension_type_id("LABELSET");
    const int trigger_type = seq.extension_type_id("TRIGGERS");
    const int lin = seq.label_id("LIN");
    const std::array<double, pulseq::TRIGGER_WIDTH> trig{1, 2, 0.0, 100e-6};

    // A chain is built tail first, so the head is the last link added.
    const int label_row = seq.register_label_set(7, lin);
    const int trigger_row = seq.register_trigger(trig.data());
    const int tail = seq.chain_extension(labelset, label_row, 0);
    const int head = seq.chain_extension(trigger_type, trigger_row, tail);

    Block block;
    block.duration = 1e-3;
    block.ext = head;
    seq.add_block(block);

    seq.remove_duplicates();

    ASSERT_EQ(seq.extensions_library().size(), 2);
    const int32_t first = seq.get_block(1).ext;
    const int32_t *head_row = seq.extensions_library().row(first);
    EXPECT_EQ(head_row[0], trigger_type);
    EXPECT_EQ(head_row[1], 1);
    ASSERT_NE(head_row[2], 0);
    const int32_t *tail_row = seq.extensions_library().row(head_row[2]);
    EXPECT_EQ(tail_row[0], labelset);
    EXPECT_EQ(tail_row[1], 1);
    EXPECT_EQ(tail_row[2], 0);
}

TEST(PulseqDedup, AppendedChainsCollapseJustLikeSearchedOnes)
{
    // `append_extension` skips the lookup `chain_extension` does, on the
    // grounds that deduplication is going to work the answer out anyway. The
    // two therefore have to end in the same place, or the fast path would be
    // writing a different file from the careful one.
    Sequence appended;
    Sequence searched;
    for (Sequence *seq : {&appended, &searched})
    {
        const int labelset = seq->extension_type_id("LABELSET");
        const int lin = seq->label_id("LIN");
        for (int i = 0; i < 4; ++i)
        {
            Block block;
            block.duration = 1e-3;
            const int row = seq->register_label_set(i % 2, lin);
            block.ext = (seq == &appended) ? seq->append_extension(labelset, row, 0)
                                           : seq->chain_extension(labelset, row, 0);
            seq->add_block(block);
        }
        seq->remove_duplicates();
    }

    ASSERT_EQ(appended.extensions_library().size(), searched.extensions_library().size());
    EXPECT_EQ(appended.label_set_library().size(), searched.label_set_library().size());
    for (int id = 1; id <= appended.extensions_library().size(); ++id)
    {
        const int32_t *a = appended.extensions_library().row(id);
        const int32_t *b = searched.extensions_library().row(id);
        for (int c = 0; c < pulseq::EXTENSION_WIDTH; ++c)
            EXPECT_EQ(a[c], b[c]) << "chain " << id << " column " << c;
    }
    for (int i = 1; i <= appended.num_blocks(); ++i)
        EXPECT_EQ(appended.get_block(i).ext, searched.get_block(i).ext) << "block " << i;
}

TEST(PulseqDedup, RotationsAndShimsAreRenumberedThroughTheirChains)
{
    Sequence seq;
    const int rotations = seq.extension_type_id("ROTATIONS");
    const int shims = seq.extension_type_id("RF_SHIMS");
    const std::array<double, pulseq::ROTATION_WIDTH> quaternion{1.0, 0.0, 0.0, 0.0};
    const std::vector<double> shim{1.0, 0.0, 1.0, 0.0};

    for (int i = 0; i < 2; ++i)
    {
        Block block;
        block.duration = 1e-3;
        const int rot = seq.register_rotation(quaternion.data());
        const int sh = seq.register_rf_shim(shim.data(), 4);
        const int tail = seq.chain_extension(shims, sh, 0);
        block.ext = seq.chain_extension(rotations, rot, tail);
        seq.add_block(block);
    }

    seq.remove_duplicates();

    EXPECT_EQ(seq.rotation_library().size(), 1);
    EXPECT_EQ(seq.rf_shim_library().size(), 1);
    EXPECT_EQ(seq.extensions_library().size(), 2);
    EXPECT_EQ(seq.get_block(2).ext, seq.get_block(1).ext);
}

TEST(PulseqDedup, SoftDelaysAreMadeUniqueByWhatTheySay)
{
    // A soft delay ends in a hint string, so two rows agreeing on every number
    // but naming different delays are two rows.
    Sequence seq;
    const int delays = seq.extension_type_id("DELAYS");
    SoftDelay row;
    row.num = 0;
    row.offset = 1e-3;
    row.factor = 1.0;
    row.hint = "TE";

    Block first;
    first.duration = 1e-3;
    first.ext = seq.chain_extension(delays, seq.register_soft_delay(row), 0);
    seq.add_block(first);

    Block second;
    second.duration = 1e-3;
    second.ext = seq.chain_extension(delays, seq.register_soft_delay(row), 0);
    seq.add_block(second);

    row.hint = "TR";
    Block third;
    third.duration = 1e-3;
    third.ext = seq.chain_extension(delays, seq.register_soft_delay(row), 0);
    seq.add_block(third);

    seq.remove_duplicates();

    ASSERT_EQ(seq.soft_delay_library().size(), 2u);
    EXPECT_EQ(seq.soft_delay_library()[0].hint, "TE");
    EXPECT_EQ(seq.soft_delay_library()[1].hint, "TR");
    EXPECT_EQ(seq.get_block(2).ext, seq.get_block(1).ext);
    EXPECT_NE(seq.get_block(3).ext, seq.get_block(1).ext);
}

/* ================================================================== */
/*  Idempotence                                                       */
/* ================================================================== */

TEST(PulseqDedup, RunningItTwiceChangesNothingTheSecondTime)
{
    // The rounding is what makes this true: the surviving row is already at
    // the precision the second pass would round it to, so nothing moves.
    Sequence seq;
    const std::vector<double> samples{0.0, 1.0, 0.0};
    const double shape = static_cast<double>(seq.register_shape(3, samples.data(), 3));
    const int labelset = seq.extension_type_id("LABELSET");
    const int lin = seq.label_id("LIN");

    for (int i = 0; i < 5; ++i)
    {
        const auto waveform = arb(1000.0 + (i % 2), shape);
        const auto plateau = trap(2000.0);
        const auto sampling = adc(64);
        Block block;
        block.gx = seq.register_arbitrary(waveform.data());
        block.gy = seq.register_trap(plateau.data());
        block.adc = seq.register_adc(sampling.data());
        block.ext = seq.chain_extension(labelset, seq.register_label_set(i, lin), 0);
        block.duration = 1e-3;
        seq.add_block(block);
    }

    seq.remove_duplicates();
    const Sequence once = seq;
    seq.remove_duplicates();

    ASSERT_EQ(seq.num_blocks(), once.num_blocks());
    EXPECT_EQ(seq.arb_library().size(), once.arb_library().size());
    EXPECT_EQ(seq.trap_library().size(), once.trap_library().size());
    EXPECT_EQ(seq.adc_library().size(), once.adc_library().size());
    EXPECT_EQ(seq.label_set_library().size(), once.label_set_library().size());
    EXPECT_EQ(seq.extensions_library().size(), once.extensions_library().size());
    EXPECT_EQ(seq.num_gradients(), once.num_gradients());
    for (int b = 1; b <= seq.num_blocks(); ++b)
    {
        const Block now = seq.get_block(b);
        const Block before = once.get_block(b);
        EXPECT_EQ(now.gx, before.gx) << "block " << b;
        EXPECT_EQ(now.gy, before.gy) << "block " << b;
        EXPECT_EQ(now.adc, before.adc) << "block " << b;
        EXPECT_EQ(now.ext, before.ext) << "block " << b;
    }
}

TEST(PulseqDedup, AnEmptySequenceSurvivesIt)
{
    Sequence seq;
    seq.remove_duplicates();
    EXPECT_EQ(seq.num_blocks(), 0);
    EXPECT_EQ(seq.num_gradients(), 0);
}

/* ================================================================== */
/*  Over the whole fixture corpus                                     */
/* ================================================================== */
/*
 * Real sequences, of which several were written without deduplicating and so
 * give this pass genuine work to do -- one 32x32 GRE arrives with 192
 * trapezoid rows and leaves with 37.
 */

namespace
{
namespace fs = std::filesystem;

std::vector<std::string> fixtures()
{
    std::vector<std::string> paths;
    for (const auto &entry : fs::directory_iterator(PULSEQ_FIXTURES_DIR))
    {
        if (entry.path().extension() == ".seq")
            paths.push_back(entry.path().string());
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

/**
     * Everything a block plays, resolved through the libraries.
     *
     * Ids mean nothing across a renumbering, so nothing here is an id: a
     * gradient is its kind and its numbers, a shape is its samples, an
     * extension chain is a list of type names and specifications.  Two of
     * these compare equal exactly when the two blocks play the same thing,
     * which is what deduplication is not allowed to change.
     */
struct Played
{
    std::vector<double> values;
    std::vector<std::string> names;
};

void push_shape(const Sequence &seq, double id, Played &out)
{
    const int shape = static_cast<int>(std::lround(id));
    if (shape < 1)
    {
        // 0 is "no shape" and -1 is the half-raster flag; neither is an id.
        out.values.push_back(id);
        return;
    }
    out.values.push_back(-2.0); // a marker, so a shape is never read as a number
    out.values.push_back(seq.shape_library().num_uncompressed(shape));
    const int count = seq.shape_library().num_compressed(shape);
    out.values.push_back(count);
    const double *samples = seq.shape_library().samples(shape);
    out.values.insert(out.values.end(), samples, samples + count);
}

void push_row(const double *row, int width, Played &out)
{
    out.values.insert(out.values.end(), row, row + width);
}

Played played(const Sequence &seq, int index)
{
    const Block block = seq.get_block(index);
    Played out;
    out.values.push_back(block.duration);

    if (block.rf)
    {
        const double *row = seq.rf_library().row(block.rf);
        out.values.push_back(row[0]);
        for (int c = 1; c <= 3; ++c)
            push_shape(seq, row[c], out);
        for (int c = 4; c < pulseq::RF_WIDTH; ++c)
            out.values.push_back(row[c]);
        out.names.push_back(std::string(1, seq.rf_uses()[static_cast<size_t>(block.rf) - 1]));
    }

    const int32_t axes[3] = {block.gx, block.gy, block.gz};
    for (int32_t grad : axes)
    {
        if (!grad)
        {
            out.values.push_back(0.0);
            continue;
        }
        if (seq.grad_kind(grad) == GradKind::Trap)
        {
            out.values.push_back(1.0);
            push_row(seq.trap_library().row(seq.grad_row(grad)), pulseq::TRAP_WIDTH, out);
        }
        else
        {
            out.values.push_back(2.0);
            const double *row = seq.arb_library().row(seq.grad_row(grad));
            out.values.push_back(row[0]);
            out.values.push_back(row[1]);
            out.values.push_back(row[2]);
            push_shape(seq, row[3], out);
            push_shape(seq, row[4], out);
            out.values.push_back(row[5]);
        }
    }

    if (block.adc)
    {
        const double *row = seq.adc_library().row(block.adc);
        for (int c = 0; c < 7; ++c)
            out.values.push_back(row[c]);
        push_shape(seq, row[7], out);
    }

    // The chain, head first, as the file stores it.
    for (int32_t node = block.ext; node;)
    {
        const int32_t *link = seq.extensions_library().row(node);
        const std::string &type = seq.extension_type_name(link[0]);
        out.names.push_back(type);
        const int ref = link[1];
        if (type == "TRIGGERS")
            push_row(seq.trigger_library().row(ref), pulseq::TRIGGER_WIDTH, out);
        else if (type == "ROTATIONS")
            push_row(seq.rotation_library().row(ref), pulseq::ROTATION_WIDTH, out);
        else if (type == "LABELSET" || type == "LABELINC")
        {
            const pulseq::IntTable &table =
                type == "LABELSET" ? seq.label_set_library() : seq.label_inc_library();
            const int32_t *label = table.row(ref);
            out.values.push_back(label[0]);
            out.names.push_back(seq.label_name(label[1]));
        }
        else if (type == "RF_SHIMS")
        {
            const int count = seq.rf_shim_library().length(ref);
            out.values.push_back(count);
            push_row(seq.rf_shim_library().row(ref), count, out);
        }
        else if (type == "DELAYS")
        {
            const SoftDelay &delay = seq.soft_delay_library()[static_cast<size_t>(ref) - 1];
            out.values.push_back(delay.num);
            out.values.push_back(delay.offset);
            out.values.push_back(delay.factor);
            out.names.push_back(delay.hint);
        }
        node = link[2];
    }
    return out;
}

class PulseqDedupFixture : public testing::TestWithParam<std::string>
{
};

TEST_P(PulseqDedupFixture, EveryBlockStillPlaysTheSameEvents)
{
    // The property the whole pass exists to preserve, and the one that a
    // renumbering bug breaks while still writing a perfectly valid file:
    // collapse the vocabulary as far as you like, but every block has to
    // come out playing what it played.
    //
    // Compared to a tolerance rather than exactly, because deduplication
    // deliberately rounds to what the file records -- but to a tolerance
    // far tighter than any confusion of one event for another could hide
    // in.
    if (GetParam().find("corrupted") != std::string::npos)
        GTEST_SKIP() << "deliberately broken fixture; rejection is tested below";

    Sequence seq = pulseq::read_file(GetParam());
    const int blocks = seq.num_blocks();
    std::vector<Played> before;
    before.reserve(static_cast<size_t>(blocks));
    for (int b = 1; b <= blocks; ++b)
        before.push_back(played(seq, b));

    seq.remove_duplicates();

    ASSERT_EQ(seq.num_blocks(), blocks);
    for (int b = 1; b <= blocks; ++b)
    {
        const Played after = played(seq, b);
        const Played &was = before[static_cast<size_t>(b) - 1];
        ASSERT_EQ(after.values.size(), was.values.size()) << "block " << b << " shape changed";
        ASSERT_EQ(after.names, was.names) << "block " << b << " names changed";
        for (size_t i = 0; i < was.values.size(); ++i)
            ASSERT_NEAR(
                after.values[i],
                was.values[i],
                1e-5 * std::max(1.0, std::fabs(was.values[i])))
                << "block " << b << " value " << i;
    }
}

TEST_P(PulseqDedupFixture, DeduplicationIsAFixedPoint)
{
    // Not "changes nothing" -- several fixtures were written without
    // deduplicating -- but "settles": once collapsed, running it again has
    // to find nothing.  Rounding a row already at the file's precision has
    // to leave it alone, or the pass would keep moving, and every
    // cross-reference has to survive being rewritten twice.
    if (GetParam().find("corrupted") != std::string::npos)
        GTEST_SKIP() << "deliberately broken fixture; rejection is tested below";

    Sequence seq = pulseq::read_file(GetParam());
    seq.remove_duplicates();
    const std::string once = pulseq::write_text(seq, false);

    seq.remove_duplicates();
    const std::string twice = pulseq::write_text(seq, false);

    EXPECT_EQ(once.size(), twice.size());
    EXPECT_TRUE(once == twice) << "a second deduplication moved something";
}

TEST_P(PulseqDedupFixture, TheScanItselfIsUntouched)
{
    if (GetParam().find("corrupted") != std::string::npos)
        GTEST_SKIP() << "deliberately broken fixture; rejection is tested below";

    Sequence seq = pulseq::read_file(GetParam());
    const int blocks = seq.num_blocks();
    const double duration = seq.duration();
    const std::vector<double> durations(seq.block_durations(), seq.block_durations() + blocks);

    seq.remove_duplicates();

    ASSERT_EQ(seq.num_blocks(), blocks);
    EXPECT_DOUBLE_EQ(seq.duration(), duration);
    for (int i = 0; i < blocks; ++i)
        EXPECT_DOUBLE_EQ(seq.block_durations()[i], durations[static_cast<size_t>(i)])
            << "block " << (i + 1);

    // Nothing may point past the end of what it points into.
    const int32_t *events = seq.block_events();
    for (int i = 0; i < blocks; ++i)
    {
        const int32_t *row = events + static_cast<size_t>(i) * pulseq::BLOCK_WIDTH;
        EXPECT_LE(row[0], seq.rf_library().size()) << "block " << (i + 1) << " rf";
        for (int axis = 1; axis <= 3; ++axis)
            EXPECT_LE(row[axis], seq.num_gradients()) << "block " << (i + 1) << " gradient";
        EXPECT_LE(row[4], seq.adc_library().size()) << "block " << (i + 1) << " adc";
        EXPECT_LE(row[5], seq.extensions_library().size()) << "block " << (i + 1) << " ext";
    }
}

TEST(PulseqDedupCorpus, ABlockPointingAtAChainThatIsNotThereIsRejected)
{
    // The corrupted fixture is a copy of the corpus gre_2d whose first
    // block names a dangling extension chain id.  Reading and writing
    // both carry it through -- neither resolves the reference -- so
    // deduplication, which does, is where it has to be noticed.
    const std::string path = std::string(PULSEQ_FIXTURES_DIR) + "/gre_2d_corrupted.seq";
    Sequence seq = pulseq::read_file(path);
    try
    {
        seq.remove_duplicates();
        FAIL() << "a dangling extension reference went unnoticed";
    }
    catch (const std::out_of_range &error)
    {
        const std::string message = error.what();
        EXPECT_NE(message.find("block 1"), std::string::npos) << message;
        EXPECT_NE(message.find("extension chain"), std::string::npos) << message;
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllFixtures,
    PulseqDedupFixture,
    testing::ValuesIn(fixtures()),
    [](const testing::TestParamInfo<std::string> &info)
    {
        std::string name = fs::path(info.param).stem().string();
        for (char &c : name)
        {
            if (!std::isalnum(static_cast<unsigned char>(c)))
                c = '_';
        }
        return name;
    });

} // namespace

/* ================================================================== */
/*  The deduplicated claim                                             */
/* ================================================================== */

namespace
{
/** A sequence with two identical readouts, so there is something to collapse. */
pulseq::Sequence twin_readouts()
{
    pulseq::Sequence seq;
    seq.set_rasters(1e-6, 10e-6, 100e-9, 10e-6);

    const double trap[pulseq::TRAP_WIDTH] = {1.0e5, 100e-6, 1000e-6, 100e-6, 0.0};
    const double adc[pulseq::ADC_WIDTH] = {64.0, 10e-6, 100e-6, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (int i = 0; i < 2; ++i)
    {
        pulseq::Block block;
        block.gx = seq.register_trap(trap);
        block.adc = seq.register_adc(adc);
        block.duration = 1200e-6;
        seq.add_block(block);
    }
    return seq;
}
} // namespace

TEST(PulseqDedupClaim, IsFalseUntilTheSequenceHasBeenCollapsed)
{
    pulseq::Sequence seq = twin_readouts();
    EXPECT_FALSE(seq.deduplicated());
    EXPECT_EQ(seq.trap_library().size(), 2);

    seq.remove_duplicates();
    EXPECT_TRUE(seq.deduplicated());
    EXPECT_EQ(seq.trap_library().size(), 1);
}

/* A second pass has nothing to find, which is what makes the claim worth
 * keeping: the ordinary path deduplicates once and then writes. */
TEST(PulseqDedupClaim, SurvivesASecondPassAndTheSecondPassChangesNothing)
{
    pulseq::Sequence seq = twin_readouts();
    seq.remove_duplicates();
    const std::string once = pulseq::write_text(seq, false);

    seq.remove_duplicates();
    EXPECT_TRUE(seq.deduplicated());
    EXPECT_EQ(pulseq::write_text(seq, false), once);
}

/* Registering anything gives up the claim -- the new row may be a duplicate. */
TEST(PulseqDedupClaim, RegisteringGivesItUp)
{
    pulseq::Sequence seq = twin_readouts();
    seq.remove_duplicates();
    ASSERT_TRUE(seq.deduplicated());

    const double trap[pulseq::TRAP_WIDTH] = {1.0e5, 100e-6, 1000e-6, 100e-6, 0.0};
    seq.register_trap(trap);
    EXPECT_FALSE(seq.deduplicated());
}

/*
 * A writer only reads, so it must not cost a caller the claim.
 *
 * It does stamp `TotalDuration` and publish the rasters, and it compresses any
 * shape still held raw -- and that last one *can* make two shapes into one, so
 * it is the one thing here that legitimately gives the claim up.  A sequence
 * whose shapes are already encoded keeps it.
 */
TEST(PulseqDedupClaim, WritingASequenceWithEncodedShapesKeepsIt)
{
    pulseq::Sequence seq = twin_readouts();
    seq.remove_duplicates();
    seq.compress_shapes();
    ASSERT_TRUE(seq.deduplicated()) << "no shape was held raw, so nothing changed";

    pulseq::write_text(seq, false);
    EXPECT_TRUE(seq.deduplicated());

    pulseq::write_binary(seq);
    EXPECT_TRUE(seq.deduplicated());
}

/* Compressing a raw shape can collapse two rows into one, so it gives it up. */
TEST(PulseqDedupClaim, CompressingARawShapeGivesItUp)
{
    pulseq::Sequence seq = twin_readouts();
    const std::vector<double> wave(64, 0.5);
    seq.register_raw_shape(wave.data(), static_cast<int>(wave.size()));
    seq.remove_duplicates();
    ASSERT_TRUE(seq.deduplicated());

    seq.compress_shapes();
    EXPECT_FALSE(seq.deduplicated());
}

/* A copy carries the claim: it is a property of the contents, not the object. */
TEST(PulseqDedupClaim, ACopyCarriesIt)
{
    pulseq::Sequence seq = twin_readouts();
    seq.remove_duplicates();

    const pulseq::Sequence copy = seq;
    EXPECT_TRUE(copy.deduplicated());

    pulseq::Sequence fresh = twin_readouts();
    EXPECT_FALSE(fresh.deduplicated());
}
