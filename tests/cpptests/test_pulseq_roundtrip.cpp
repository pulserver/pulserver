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
 * The fixtures were written by MATLAB and by several versions of PyPulseq, so
 * the *first* write generally differs from the file on disk: it is this
 * writer's rendering of that sequence, not a reproduction of the original
 * bytes.  Byte-equality against a reference writer is checked on the Python
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
        for (const auto& entry : fs::directory_iterator(PULSEQ_FIXTURES_DIR))
        {
            if (entry.path().extension() == ".seq")
                paths.push_back(entry.path().string());
        }
        std::sort(paths.begin(), paths.end());
        return paths;
    }

    /** Write @p payload to a scratch file and read it back. */
    pulseq::Sequence reread(const std::string& payload, const char* suffix)
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
        ASSERT_EQ(from_text.extensions_library().size(),
                  from_binary.extensions_library().size());
        ASSERT_EQ(from_text.rotation_library().size(), from_binary.rotation_library().size());

        // Block tables are ids and must agree exactly.
        const int32_t* text_blocks = from_text.block_events();
        const int32_t* binary_blocks = from_binary.block_events();
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
            const double* a = from_text.rf_library().row(id);
            const double* b = from_binary.rf_library().row(id);
            for (int c = 0; c < pulseq::RF_WIDTH; ++c)
                EXPECT_NEAR(a[c], b[c], 1e-6 * std::max(1.0, std::fabs(a[c])))
                    << "rf " << id << " column " << c;
        }

        for (int id = 1; id <= from_text.shape_library().size(); ++id)
        {
            ASSERT_EQ(from_text.shape_library().num_uncompressed(id),
                      from_binary.shape_library().num_uncompressed(id));
            ASSERT_EQ(from_text.shape_library().num_compressed(id),
                      from_binary.shape_library().num_compressed(id));
            const double* a = from_text.shape_library().samples(id);
            const double* b = from_binary.shape_library().samples(id);
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

    INSTANTIATE_TEST_SUITE_P(
        AllFixtures,
        PulseqFixture,
        testing::ValuesIn(fixtures()),
        [](const testing::TestParamInfo<std::string>& info) {
            std::string name = fs::path(info.param).stem().string();
            for (char& c : name)
            {
                if (!std::isalnum(static_cast<unsigned char>(c)))
                    c = '_';
            }
            return name;
        });

}  // namespace
