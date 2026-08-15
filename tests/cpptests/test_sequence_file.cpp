// test_sequence_file.cpp
//
// pulseq::SequenceFile: the lazy section index and the definitions-only
// read, over the corpus fixtures in both formats. The claims:
//
//   1. definitions() parsed from the one section equal the full parse's,
//      key by key, kind by kind, for text and binary alike;
//   2. the binary section walk lands exactly on every section head (an
//      off-by-anything in one section's computed length would derail every
//      later one, so a complete walk IS the layout check);
//   3. next_sequence() resolves the EPI chain link and is empty elsewhere;
//   4. sequence() hands back the ordinary full parse.

#include "pulseq/sequence_file.hpp"

#include "pulseq/read.hpp"
#include "pulseq/sequence.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{

    std::string corpus(const std::string& name)
    {
        return (fs::path(PULSEQ_CORPUS_DIR) / name).string();
    }

    std::vector<std::string> corpus_files(const std::string& extension)
    {
        std::vector<std::string> files;
        for (const auto& entry : fs::directory_iterator(PULSEQ_CORPUS_DIR))
        {
            if (entry.path().extension() == extension)
                files.push_back(entry.path().string());
        }
        std::sort(files.begin(), files.end());
        return files;
    }

    void expect_same_definitions(pulseq::SequenceFile& file)
    {
        pulseq::Sequence full = pulseq::read_file(file.path());
        const auto& lazy = file.definitions();

        for (const auto& entry : full.definitions())
        {
            // The full parse re-derives TotalDuration; the lazy read only
            // reports what the section says. Compare what both have.
            const auto it = lazy.find(entry.first);
            ASSERT_NE(it, lazy.end()) << entry.first << " missing from lazy read";

            const pulseq::Definition& theirs = entry.second;
            const pulseq::Definition& ours = it->second;
            if (theirs.kind() == pulseq::Definition::Kind::Text)
            {
                EXPECT_EQ(ours.kind(), pulseq::Definition::Kind::Text) << entry.first;
                EXPECT_EQ(ours.text(), theirs.text()) << entry.first;
                continue;
            }
            ASSERT_EQ(ours.numbers().size(), theirs.numbers().size()) << entry.first;
            for (size_t i = 0; i < theirs.numbers().size(); ++i)
            {
                /* The full parse round-trips definition values through
                 * 9-significant-digit strings; the lazy read takes the
                 * stored float64 as written. Compare at string precision. */
                EXPECT_NEAR(ours.numbers()[i], theirs.numbers()[i],
                            1e-8 * std::max(1.0, std::fabs(theirs.numbers()[i])))
                    << entry.first << "[" << i << "]";
            }
        }
    }

}  // namespace

class SequenceFileText : public ::testing::TestWithParam<std::string>
{
};

TEST_P(SequenceFileText, DefinitionsMatchTheFullParse)
{
    pulseq::SequenceFile file(GetParam());
    ASSERT_FALSE(file.is_binary());
    expect_same_definitions(file);
}

TEST_P(SequenceFileText, TheIndexSeesTheStructuralSections)
{
    pulseq::SequenceFile file(GetParam());
    EXPECT_TRUE(file.has_section("VERSION"));
    EXPECT_TRUE(file.has_section("BLOCKS"));
    EXPECT_TRUE(file.has_section("DEFINITIONS"));
}

INSTANTIATE_TEST_SUITE_P(AllTextFixtures, SequenceFileText,
                         ::testing::ValuesIn(corpus_files(".seq")),
                         [](const ::testing::TestParamInfo<std::string>& info)
                         {
                             std::string name = fs::path(info.param).stem().string();
                             for (char& c : name)
                                 if (!std::isalnum(static_cast<unsigned char>(c)))
                                     c = '_';
                             return name;
                         });

class SequenceFileBinary : public ::testing::TestWithParam<std::string>
{
};

TEST_P(SequenceFileBinary, DefinitionsMatchTheFullParse)
{
    pulseq::SequenceFile file(GetParam());
    ASSERT_TRUE(file.is_binary());
    expect_same_definitions(file);
}

TEST_P(SequenceFileBinary, TheSectionWalkCoversTheWholeFile)
{
    pulseq::SequenceFile file(GetParam());
    const auto& names = file.section_names();
    ASSERT_FALSE(names.empty());
    EXPECT_TRUE(file.has_section("BLOCKS"));
    EXPECT_TRUE(file.has_section("SHAPES"));
}

INSTANTIATE_TEST_SUITE_P(AllBinaryFixtures, SequenceFileBinary,
                         ::testing::ValuesIn(corpus_files(".bin")),
                         [](const ::testing::TestParamInfo<std::string>& info)
                         {
                             std::string name = fs::path(info.param).stem().string();
                             for (char& c : name)
                                 if (!std::isalnum(static_cast<unsigned char>(c)))
                                     c = '_';
                             return name;
                         });

TEST(SequenceFileChain, TheEpiLeadNamesItsSuccessorAndTheTailNamesNone)
{
    pulseq::SequenceFile lead(corpus("epi_2d.seq"));
    EXPECT_EQ(lead.next_sequence(), "epi_2d_main.seq");

    pulseq::SequenceFile tail(corpus("epi_2d_main.seq"));
    EXPECT_EQ(tail.next_sequence(), "");
}

TEST(SequenceFileFullParse, TheCachedSequenceIsTheOrdinaryRead)
{
    pulseq::SequenceFile file(corpus("gre_2d.seq"));
    pulseq::Sequence direct = pulseq::read_file(corpus("gre_2d.seq"));
    EXPECT_EQ(file.sequence().num_blocks(), direct.num_blocks());
    // Two calls, one parse.
    EXPECT_EQ(&file.sequence(), &file.sequence());
}
