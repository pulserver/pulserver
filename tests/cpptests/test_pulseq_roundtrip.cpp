/**
 * Reading and writing, checked against themselves over the whole fixture corpus.
 *
 * The strongest statement available without shipping a second copy of every
 * fixture is that the pair is a fixed point: whatever a file means, writing it
 * out and reading it back has to mean the same thing, and writing *that* has to
 * produce the same bytes.  A reader that drops a field, a writer that invents
 * one, a unit conversion applied once too often -- all of them break this, and
 * none of them needs a hand-written expectation to catch.
 *
 * Cross-format agreement is the second half: the text and binary writers are
 * separate code, and a sequence stored either way has to come back the same.
 * (Not bit for bit -- the text format prints shape samples to nine significant
 * figures where the binary format carries float32 -- so amplitudes are compared
 * to a tolerance and structure exactly.)
 *
 * The fixtures mix writer-produced files and hand-specified event-table
 * text, so the *first* write generally differs from the file on disk: it is
 * this writer's rendering of that sequence, not a reproduction of the
 * original bytes.  Byte-equality against a reference writer is checked on the Python
 * side, where a reference writer exists to compare against.
 */

#include <gtest/gtest.h>

#include "pulseq/read.hpp"
#include "pulseq/sequence.hpp"
#include "pulseq/write.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

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

/** Write @p payload to a scratch file and read it back. */
pulseq::Sequence reread(const std::string &payload, const char *suffix)
{
    const fs::path scratch =
        fs::temp_directory_path() / (std::string("pulseqpp_roundtrip") + suffix);
    {
        std::ofstream out(scratch, std::ios::binary);
        out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    }
    pulseq::Sequence seq = pulseq::read_file(scratch.string());
    fs::remove(scratch);
    return seq;
}

class PulseqFixture : public testing::TestWithParam<std::string>
{
};

TEST_P(PulseqFixture, TextWriteIsAFixedPoint)
{
    pulseq::Sequence first = pulseq::read_file(GetParam());
    const std::string once = pulseq::write_text(first, false);

    pulseq::Sequence second = reread(once, ".seq");
    const std::string twice = pulseq::write_text(second, false);

    EXPECT_EQ(once.size(), twice.size());
    EXPECT_TRUE(once == twice) << "second write differs from the first";
}

TEST_P(PulseqFixture, BinaryWriteIsAFixedPoint)
{
    pulseq::Sequence first = pulseq::read_file(GetParam());
    const std::string once = pulseq::write_binary(first);

    pulseq::Sequence second = reread(once, ".bin");
    const std::string twice = pulseq::write_binary(second);

    EXPECT_EQ(once.size(), twice.size());
    EXPECT_TRUE(once == twice) << "second binary write differs from the first";
}

TEST_P(PulseqFixture, TheTwoFormatsCarryTheSameSequence)
{
    pulseq::Sequence source = pulseq::read_file(GetParam());
    const std::string as_text = pulseq::write_text(source, false);
    const std::string as_binary = pulseq::write_binary(source);

    pulseq::Sequence from_text = reread(as_text, ".seq");
    pulseq::Sequence from_binary = reread(as_binary, ".bin");

    ASSERT_EQ(from_text.num_blocks(), from_binary.num_blocks());
    ASSERT_EQ(from_text.num_gradients(), from_binary.num_gradients());
    ASSERT_EQ(from_text.rf_library().size(), from_binary.rf_library().size());
    ASSERT_EQ(from_text.adc_library().size(), from_binary.adc_library().size());
    ASSERT_EQ(from_text.shape_library().size(), from_binary.shape_library().size());
    ASSERT_EQ(from_text.extensions_library().size(), from_binary.extensions_library().size());
    ASSERT_EQ(from_text.rotation_library().size(), from_binary.rotation_library().size());

    // Block tables are ids and must agree exactly.
    const int32_t *text_blocks = from_text.block_events();
    const int32_t *binary_blocks = from_binary.block_events();
    for (int i = 0; i < from_text.num_blocks() * pulseq::BLOCK_WIDTH; ++i)
        ASSERT_EQ(text_blocks[i], binary_blocks[i]) << "block column " << i;

    for (int i = 0; i < from_text.num_blocks(); ++i)
    {
        EXPECT_NEAR(from_text.block_durations()[i], from_binary.block_durations()[i], 1e-12)
            << "block " << (i + 1) << " duration";
    }

    // Amplitudes: the text format prints nine significant figures, the
    // binary format carries float32 shape samples, so this is a comparison
    // to tolerance by construction rather than by concession.
    for (int id = 1; id <= from_text.rf_library().size(); ++id)
    {
        const double *a = from_text.rf_library().row(id);
        const double *b = from_binary.rf_library().row(id);
        for (int c = 0; c < pulseq::RF_WIDTH; ++c)
            EXPECT_NEAR(a[c], b[c], 1e-6 * std::max(1.0, std::fabs(a[c])))
                << "rf " << id << " column " << c;
    }

    for (int id = 1; id <= from_text.shape_library().size(); ++id)
    {
        ASSERT_EQ(
            from_text.shape_library().num_uncompressed(id),
            from_binary.shape_library().num_uncompressed(id));
        ASSERT_EQ(
            from_text.shape_library().num_compressed(id),
            from_binary.shape_library().num_compressed(id));
        const double *a = from_text.shape_library().samples(id);
        const double *b = from_binary.shape_library().samples(id);
        for (int i = 0; i < from_text.shape_library().num_compressed(id); ++i)
            EXPECT_NEAR(a[i], b[i], 1e-6 * std::max(1.0, std::fabs(a[i])))
                << "shape " << id << " sample " << i;
    }
}

TEST_P(PulseqFixture, ReadingProducesSomethingToWrite)
{
    // A fixture that silently read as an empty sequence would pass every
    // comparison above.
    pulseq::Sequence seq = pulseq::read_file(GetParam());
    EXPECT_GT(seq.num_blocks(), 0);
    EXPECT_GT(seq.duration(), 0.0);
}

/* ============================================================== */
/*  Rows longer than the reader's line buffer                     */
/* ============================================================== */
/*
     * Two sections have rows whose length is set by the prescription rather
     * than by the format: `[DEFINITIONS]` writes `SlicePositions` as one value
     * per slice, and `[RF_SHIMS]` two per transmit channel.  Everything else
     * is fixed-width or one value per line.
     *
     * The text reader reads with `fgets` into a 256-byte buffer, which does
     * not fail on a longer row -- it returns the first 255 bytes and hands the
     * rest back as the next line.  A 20-slice `SlicePositions` used to come
     * back holding 18 values, with the remainder parsed as a definition named
     * after whichever digits the cut landed between, and nothing anywhere
     * raised.  The counts below straddle that: 16 slices fit in the old
     * buffer, 20 did not.
     *
     * No fixture in the corpus is wide enough to reach this, which is why
     * these build their sequences instead.
     */

/** @p n slice positions, evenly spaced, at a spacing that does not round. */
std::vector<double> slice_positions(int n)
{
    std::vector<double> out;
    out.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        out.push_back((i - (n - 1) / 2.0) * 5.1234567e-3);
    return out;
}

class WideRowFixture : public testing::TestWithParam<int>
{
};

TEST_P(WideRowFixture, SlicePositionsSurviveTheTextFormat)
{
    const std::vector<double> expected = slice_positions(GetParam());

    pulseq::Sequence seq;
    pulseq::Block block;
    block.duration = 1e-3;
    seq.add_block(block);
    seq.set_definition("SlicePositions", pulseq::Definition(expected));

    const pulseq::Sequence back = reread(pulseq::write_text(seq, false), ".seq");
    const pulseq::Definition *got = back.definition("SlicePositions");
    ASSERT_NE(got, nullptr);
    ASSERT_EQ(got->numbers().size(), expected.size());
    /* Definitions are written at `%0.9g`, so nine significant figures is
         * all a round trip can promise -- and the binary format carries the
         * same strings, so it promises exactly as much.  A mantissa leading
         * with 1 spends most of a decade before the first digit changes, which
         * is why the relative bound is 1e-8 and not 1e-9. */
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_NEAR(got->numbers()[i], expected[i], 1e-8 * std::fabs(expected[i]) + 1e-15)
            << "slice " << i;
}

TEST_P(WideRowFixture, SlicePositionsSurviveTheBinaryFormat)
{
    const std::vector<double> expected = slice_positions(GetParam());

    pulseq::Sequence seq;
    pulseq::Block block;
    block.duration = 1e-3;
    seq.add_block(block);
    seq.set_definition("SlicePositions", pulseq::Definition(expected));

    const pulseq::Sequence back = reread(pulseq::write_binary(seq), ".bin");
    const pulseq::Definition *got = back.definition("SlicePositions");
    ASSERT_NE(got, nullptr);
    ASSERT_EQ(got->numbers().size(), expected.size());
    /* Definitions are written at `%0.9g`, so nine significant figures is
         * all a round trip can promise -- and the binary format carries the
         * same strings, so it promises exactly as much.  A mantissa leading
         * with 1 spends most of a decade before the first digit changes, which
         * is why the relative bound is 1e-8 and not 1e-9. */
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_NEAR(got->numbers()[i], expected[i], 1e-8 * std::fabs(expected[i]) + 1e-15)
            << "slice " << i;
}

TEST_P(WideRowFixture, ALongDefinitionDoesNotBecomeTwo)
{
    /* The tail of a split row parses as a definition of its own, named
         * after a fragment of a number.  Nothing else in the file can produce
         * a key that is a number, so that is the whole test. */
    pulseq::Sequence seq;
    pulseq::Block block;
    block.duration = 1e-3;
    seq.add_block(block);
    seq.set_definition("SlicePositions", pulseq::Definition(slice_positions(GetParam())));

    const pulseq::Sequence back = reread(pulseq::write_text(seq, false), ".seq");
    for (const auto &entry : back.definitions())
    {
        char *end = nullptr;
        std::strtod(entry.first.c_str(), &end);
        EXPECT_FALSE(end != nullptr && *end == '\0' && !entry.first.empty())
            << "definition '" << entry.first << "' is a number, so it is a row fragment";
    }
}

INSTANTIATE_TEST_SUITE_P(SliceCounts, WideRowFixture, testing::Values(1, 8, 16, 20, 32, 64, 200));

/** @p channels complex weights, as `[magnitude, phase]` pairs. */
std::vector<double> shim_weights(int channels)
{
    std::vector<double> out;
    out.reserve(static_cast<size_t>(2 * channels));
    for (int c = 0; c < channels; ++c)
    {
        out.push_back(0.11111111 + 0.88888888 * c / (channels > 1 ? channels - 1 : 1));
        out.push_back(-3.14159265 + 6.2831853 * c / (channels > 1 ? channels - 1 : 1));
    }
    return out;
}

class ShimWidthFixture : public testing::TestWithParam<int>
{
};

TEST_P(ShimWidthFixture, EveryChannelSurvivesBothFormats)
{
    const int channels = GetParam();
    const std::vector<double> expected = shim_weights(channels);

    pulseq::Sequence seq;
    const int shim = seq.register_rf_shim(expected.data(), static_cast<int>(expected.size()));
    pulseq::Block block;
    block.duration = 1e-3;
    block.ext = static_cast<int32_t>(seq.chain_extension(
        static_cast<int32_t>(seq.extension_type_id("RF_SHIMS")),
        static_cast<int32_t>(shim),
        0));
    seq.add_block(block);

    const pulseq::Sequence from_text = reread(pulseq::write_text(seq, false), ".seq");
    const pulseq::Sequence from_binary = reread(pulseq::write_binary(seq), ".bin");

    ASSERT_EQ(from_text.rf_shim_library().length(1), static_cast<int>(expected.size()));
    ASSERT_EQ(from_binary.rf_shim_library().length(1), static_cast<int>(expected.size()));
    for (size_t i = 0; i < expected.size(); ++i)
    {
        /* Shim rows are written at plain `%g` -- six significant figures,
             * which for a phase in radians is about 5e-6. */
        const double tol = 1e-5 * std::fabs(expected[i]) + 1e-12;
        EXPECT_NEAR(from_text.rf_shim_library().row(1)[i], expected[i], tol) << "text, value " << i;
        EXPECT_NEAR(from_binary.rf_shim_library().row(1)[i], expected[i], tol)
            << "binary, value " << i;
    }
}

/* 8 channels already exceeds the old buffer once written out; 64 is the
     * reader's compile-time ceiling, and past it both formats refuse. */
INSTANTIATE_TEST_SUITE_P(ChannelCounts, ShimWidthFixture, testing::Values(2, 8, 16, 32, 64));

INSTANTIATE_TEST_SUITE_P(
    AllFixtures,
    PulseqFixture,
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
