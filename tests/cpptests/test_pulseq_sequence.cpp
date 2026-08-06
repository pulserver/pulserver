/**
 * Unit tests for pulseq::Sequence -- the storage model, not the file formats.
 *
 * The properties worth pinning down here are the ones a written file depends
 * on: ids handed out in append order, gradients numbered in one sequence
 * across the two tables they are stored in, and extension chains collapsing
 * when they repeat.  Everything downstream assumes those, and none of them is
 * visible from a `.seq` until something has already gone wrong in it.
 */

#include <gtest/gtest.h>

#include "pulseq/sequence.hpp"

#include <array>
#include <vector>

using pulseq::Block;
using pulseq::Definition;
using pulseq::GradKind;
using pulseq::Sequence;
using pulseq::SoftDelay;

namespace
{
    std::array<double, pulseq::TRAP_WIDTH> trap(double amplitude)
    {
        return {amplitude, 100e-6, 500e-6, 100e-6, 0.0};
    }

    std::array<double, pulseq::ARB_WIDTH> arb(double amplitude, double shape_id)
    {
        return {amplitude, 0.0, 0.0, shape_id, 0.0, 0.0};
    }
}  // namespace

/* ================================================================== */
/*  Tables and ids                                                    */
/* ================================================================== */

TEST(PulseqSequence, LibraryIdsAreOneBasedAndInAppendOrder)
{
    Sequence seq;
    const std::array<double, pulseq::ADC_WIDTH> a{64, 1e-5, 0, 0, 0, 0, 0, 0};
    const std::array<double, pulseq::ADC_WIDTH> b{128, 1e-5, 0, 0, 0, 0, 0, 0};

    EXPECT_EQ(seq.register_adc(a.data()), 1);
    EXPECT_EQ(seq.register_adc(b.data()), 2);
    EXPECT_EQ(seq.adc_library().size(), 2);
    EXPECT_DOUBLE_EQ(seq.adc_library().row(2)[0], 128.0);
}

TEST(PulseqSequence, RfKeepsItsUseTagAlongsideTheRow)
{
    Sequence seq;
    std::array<double, pulseq::RF_WIDTH> row{};
    row[0] = 1234.5;

    EXPECT_EQ(seq.register_rf(row.data(), 'e'), 1);
    row[0] = 99.0;
    EXPECT_EQ(seq.register_rf(row.data(), 'r'), 2);

    ASSERT_EQ(seq.rf_uses().size(), 2u);
    EXPECT_EQ(seq.rf_uses()[0], 'e');
    EXPECT_EQ(seq.rf_uses()[1], 'r');
    EXPECT_DOUBLE_EQ(seq.rf_library().row(1)[0], 1234.5);
}

/* ================================================================== */
/*  Gradients: one numbering, two tables                              */
/* ================================================================== */

TEST(PulseqSequence, GradientIdsInterleaveAcrossTrapAndArbitrary)
{
    Sequence seq;
    const auto t1 = trap(1000.0);
    const auto a1 = arb(2000.0, 1);
    const auto t2 = trap(3000.0);

    // This is the property the file format needs: a scan mixing the two kinds
    // numbers them in one sequence, so a trapezoid really can be 1 and 3 with
    // an arbitrary at 2.  Storing them apart must not change that.
    EXPECT_EQ(seq.register_trap(t1.data()), 1);
    EXPECT_EQ(seq.register_arbitrary(a1.data()), 2);
    EXPECT_EQ(seq.register_trap(t2.data()), 3);
    EXPECT_EQ(seq.num_gradients(), 3);

    EXPECT_EQ(seq.grad_kind(1), GradKind::Trap);
    EXPECT_EQ(seq.grad_kind(2), GradKind::Arbitrary);
    EXPECT_EQ(seq.grad_kind(3), GradKind::Trap);

    EXPECT_EQ(seq.grad_row(1), 1);
    EXPECT_EQ(seq.grad_row(2), 1);  // first row of the *arbitrary* table
    EXPECT_EQ(seq.grad_row(3), 2);  // second row of the trapezoid table

    EXPECT_DOUBLE_EQ(seq.trap_library().row(seq.grad_row(3))[0], 3000.0);
    EXPECT_DOUBLE_EQ(seq.arb_library().row(seq.grad_row(2))[0], 2000.0);
}

TEST(PulseqSequence, GradientSlotsCarrySignAsTheKind)
{
    Sequence seq;
    const auto t = trap(1.0);
    const auto a = arb(2.0, 1);
    seq.register_trap(t.data());
    seq.register_arbitrary(a.data());

    EXPECT_EQ(seq.grad_slots()[0], 1);
    EXPECT_EQ(seq.grad_slots()[1], -1);
}

TEST(PulseqSequence, UnknownGradientIdIsNone)
{
    Sequence seq;
    EXPECT_EQ(seq.grad_kind(0), GradKind::None);
    EXPECT_EQ(seq.grad_kind(1), GradKind::None);
    EXPECT_EQ(seq.grad_row(7), 0);
}

/* ================================================================== */
/*  Blocks                                                            */
/* ================================================================== */

TEST(PulseqSequence, BlocksRoundTrip)
{
    Sequence seq;
    Block block;
    block.rf = 1;
    block.gx = 2;
    block.adc = 3;
    block.ext = 4;
    block.duration = 5e-3;

    EXPECT_EQ(seq.add_block(block), 1);
    EXPECT_EQ(seq.num_blocks(), 1);

    const Block back = seq.get_block(1);
    EXPECT_EQ(back.rf, 1);
    EXPECT_EQ(back.gx, 2);
    EXPECT_EQ(back.gy, 0);
    EXPECT_EQ(back.gz, 0);
    EXPECT_EQ(back.adc, 3);
    EXPECT_EQ(back.ext, 4);
    EXPECT_DOUBLE_EQ(back.duration, 5e-3);
}

TEST(PulseqSequence, SetBlockOverwritesInPlace)
{
    Sequence seq;
    Block block;
    block.duration = 1e-3;
    seq.add_block(block);
    seq.add_block(block);

    Block replacement;
    replacement.gz = 9;
    replacement.duration = 2e-3;
    seq.set_block(1, replacement);

    EXPECT_EQ(seq.get_block(1).gz, 9);
    EXPECT_DOUBLE_EQ(seq.get_block(1).duration, 2e-3);
    EXPECT_EQ(seq.get_block(2).gz, 0);
    EXPECT_EQ(seq.num_blocks(), 2);
}

TEST(PulseqSequence, BlockIndicesAreCheckedAndOneBased)
{
    Sequence seq;
    Block block;
    seq.add_block(block);

    EXPECT_THROW(seq.get_block(0), std::out_of_range);
    EXPECT_THROW(seq.get_block(2), std::out_of_range);
    EXPECT_THROW(seq.set_block(0, block), std::out_of_range);
    EXPECT_NO_THROW(seq.get_block(1));
}

TEST(PulseqSequence, BulkBlockLoadMatchesAppending)
{
    // A composed scan arrives as columns; loading them must land exactly where
    // the same blocks appended one at a time would have landed.
    const std::vector<int32_t> events{1, 0, 0, 0, 2, 0, 3, 4, 0, 0, 0, 5};
    const std::vector<double> durations{1e-3, 2e-3};

    Sequence bulk;
    bulk.set_blocks(events.data(), durations.data(), 2);

    Sequence appended;
    Block first;
    first.rf = 1;
    first.adc = 2;
    first.duration = 1e-3;
    Block second;
    second.rf = 3;
    second.gx = 4;
    second.ext = 5;
    second.duration = 2e-3;
    appended.add_block(first);
    appended.add_block(second);

    ASSERT_EQ(bulk.num_blocks(), appended.num_blocks());
    for (int i = 1; i <= bulk.num_blocks(); ++i)
    {
        const Block a = bulk.get_block(i);
        const Block b = appended.get_block(i);
        EXPECT_EQ(a.rf, b.rf);
        EXPECT_EQ(a.gx, b.gx);
        EXPECT_EQ(a.adc, b.adc);
        EXPECT_EQ(a.ext, b.ext);
        EXPECT_DOUBLE_EQ(a.duration, b.duration);
    }
}

TEST(PulseqSequence, DurationSumsTheBlocks)
{
    Sequence seq;
    Block block;
    block.duration = 1.5e-3;
    seq.add_block(block);
    seq.add_block(block);
    seq.add_block(block);
    EXPECT_NEAR(seq.duration(), 4.5e-3, 1e-15);
}

/* ================================================================== */
/*  Extension chains                                                  */
/* ================================================================== */

TEST(PulseqSequence, RepeatedChainsCollapseToOneRow)
{
    // Every readout of a scan carries the same pair of labels, so this library
    // is scan-length before deduplication and three rows after it.  Collapsing
    // as the chain is built is what keeps it from ever being scan-length.
    Sequence seq;
    const int first = seq.chain_extension(1, 5, 0);
    const int again = seq.chain_extension(1, 5, 0);
    const int other = seq.chain_extension(1, 6, 0);

    EXPECT_EQ(first, 1);
    EXPECT_EQ(again, 1);
    EXPECT_EQ(other, 2);
    EXPECT_EQ(seq.extensions_library().size(), 2);
}

TEST(PulseqSequence, ChainsOfDifferentTailsStayApart)
{
    Sequence seq;
    const int head = seq.chain_extension(1, 5, 0);
    const int stacked = seq.chain_extension(2, 7, head);
    EXPECT_NE(head, stacked);
    EXPECT_EQ(seq.extensions_library().row(stacked)[2], head);
}

/* ================================================================== */
/*  Extension type registry                                           */
/* ================================================================== */

TEST(PulseqSequence, ExtensionTypeIdsAreAssignedOnFirstUse)
{
    Sequence seq;
    EXPECT_EQ(seq.find_extension_type_id("LABELSET"), 0);
    EXPECT_EQ(seq.extension_type_id("LABELSET"), 1);
    EXPECT_EQ(seq.extension_type_id("ROTATIONS"), 2);
    EXPECT_EQ(seq.extension_type_id("LABELSET"), 1);
    EXPECT_EQ(seq.extension_type_name(2), "ROTATIONS");
}

TEST(PulseqSequence, ReadingAFileCanForceTheNumbering)
{
    // A file declares its own mapping, and it need not match the order the
    // sections appear in -- so setting an id has to win over first use.
    Sequence seq;
    seq.extension_type_id("LABELSET");  // would be 1
    seq.set_extension_type_id("LABELSET", 4);

    EXPECT_EQ(seq.find_extension_type_id("LABELSET"), 4);
    EXPECT_EQ(seq.extension_type_name(4), "LABELSET");
    EXPECT_TRUE(seq.extension_type_name(1).empty());
}

/* ================================================================== */
/*  Ragged libraries                                                  */
/* ================================================================== */

TEST(PulseqSequence, ShimVectorsMayDifferInLength)
{
    Sequence seq;
    const std::vector<double> two{1.0, 0.0, 1.0, 3.14};
    const std::vector<double> four{1, 0, 1, 0, 1, 0, 1, 0};

    EXPECT_EQ(seq.register_rf_shim(two.data(), 4), 1);
    EXPECT_EQ(seq.register_rf_shim(four.data(), 8), 2);
    EXPECT_EQ(seq.rf_shim_library().length(1), 4);
    EXPECT_EQ(seq.rf_shim_library().length(2), 8);
    EXPECT_DOUBLE_EQ(seq.rf_shim_library().row(1)[3], 3.14);
    EXPECT_DOUBLE_EQ(seq.rf_shim_library().row(2)[0], 1.0);
}

TEST(PulseqSequence, ShapesKeepTheirCompressedFormAndUncompressedLength)
{
    Sequence seq;
    // A run-length encoded shape: the stored samples are shorter than what
    // they decompress to, and both numbers have to survive.
    const std::vector<double> compressed{0.0, 0.1, 0.1, 8.0};
    EXPECT_EQ(seq.register_shape(11, compressed.data(), 4), 1);

    EXPECT_EQ(seq.shape_library().num_uncompressed(1), 11);
    EXPECT_EQ(seq.shape_library().num_compressed(1), 4);
    EXPECT_DOUBLE_EQ(seq.shape_library().samples(1)[3], 8.0);
}

TEST(PulseqSequence, SoftDelaysCarryTheirHintName)
{
    Sequence seq;
    SoftDelay row;
    row.num = 2;
    row.offset = -1e-3;
    row.factor = 1.0;
    row.hint = "TE";

    EXPECT_EQ(seq.register_soft_delay(row), 1);
    EXPECT_EQ(seq.soft_delay_library()[0].hint, "TE");
    EXPECT_DOUBLE_EQ(seq.soft_delay_library()[0].offset, -1e-3);
}

/* ================================================================== */
/*  Definitions                                                       */
/* ================================================================== */

TEST(PulseqSequence, DefinitionsRememberWhetherTheyAreWholeNumbers)
{
    // The text writer formats these differently and the binary format tags
    // them, so an integer definition read back as a real would change the file.
    Sequence seq;
    seq.set_definition("Name", Definition(std::string("gre")));
    seq.set_definition("FOV", Definition(std::vector<double>{0.25, 0.25, 0.005}));
    seq.set_definition("Matrix", Definition::integers({128, 128, 1}));

    ASSERT_NE(seq.definition("Name"), nullptr);
    EXPECT_EQ(seq.definition("Name")->kind(), Definition::Kind::Text);
    EXPECT_EQ(seq.definition("Name")->text(), "gre");
    EXPECT_EQ(seq.definition("FOV")->kind(), Definition::Kind::Real);
    EXPECT_EQ(seq.definition("Matrix")->kind(), Definition::Kind::Int);
    EXPECT_EQ(seq.definition("Missing"), nullptr);
}

TEST(PulseqSequence, DefinitionsIterateInSortedOrder)
{
    // Both writers emit definitions sorted by key; holding them sorted is what
    // makes that free rather than a copy at write time.
    Sequence seq;
    seq.set_definition("Zed", Definition(1.0));
    seq.set_definition("Alpha", Definition(2.0));
    seq.set_definition("Middle", Definition(3.0));

    std::vector<std::string> keys;
    for (const auto& entry : seq.definitions())
        keys.push_back(entry.first);
    EXPECT_EQ(keys, (std::vector<std::string>{"Alpha", "Middle", "Zed"}));
}
