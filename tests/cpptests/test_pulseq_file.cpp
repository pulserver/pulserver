/**
 * @file test_pulseq_file.cpp
 * @brief Smoke test for the pulseq::File / pulseq::FileSet C++ wrapper
 * (cxx/include/pulseq/file.hpp) against a real .seq fixture.
 */

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "pulseq.hpp"

#ifndef PULSEQ_FILE_FIXTURES_DIR
#define PULSEQ_FILE_FIXTURES_DIR "."
#endif

namespace
{

std::string fixture_path(const char *name)
{
    return std::string(PULSEQ_FILE_FIXTURES_DIR) + "/" + name;
}

} // namespace

TEST(PulseqFile, ReadsBlocksAndDefinitions)
{
    pulseq::Raster raster;
    raster.rf_us = 2.0f;
    raster.grad_us = 20.0f;
    raster.block_us = 20.0f;

    pulseq::File file = pulseq::File::read(fixture_path("gre_2d.seq"), raster);

    EXPECT_GT(file.num_blocks(), 0);
    EXPECT_GT(file.version_combined(), 0);

    // pulseq_parse.c converts [DEFINITIONS] FOV from meters (the .seq wire
    // format) to centimeters at parse time -- see pulseq_parse.c's "FOV" key
    // handling -- so the fixture's "FOV 0.22 0.22 0.005" (meters) becomes 22.
    const pulseq_reserved_definitions &defs = file.reserved_definitions();
    EXPECT_NEAR(defs.fov[0], 22.0f, 1e-4f);

    pulseq_raw_block raw = file.raw_block_content_ids(0);
    EXPECT_GE(raw.block_duration, 0);

    pulseq_raw_extension ext = file.raw_extension(raw);
    (void)ext; // resolves without throwing; contents depend on the fixture
}

TEST(PulseqFile, RejectsMissingFile)
{
    EXPECT_THROW(pulseq::File::read(fixture_path("does_not_exist.seq")), pulseq::Error);
}

TEST(PulseqFile, DecompressesUncompressedShape)
{
    const float samples[3] = {0.0f, 0.5f, 1.0f};
    pulseq_shape encoded = PULSEQ_SHAPE_INIT;
    encoded.num_uncompressed_samples = 3;
    encoded.num_samples = 3;
    encoded.samples = const_cast<float *>(samples);

    pulseq::Shape decoded = pulseq::decompress_shape(encoded, 2.0f);

    ASSERT_EQ(decoded.samples.size(), 3u);
    EXPECT_NEAR(decoded.samples[0], 0.0f, 1e-6f);
    EXPECT_NEAR(decoded.samples[1], 1.0f, 1e-6f);
    EXPECT_NEAR(decoded.samples[2], 2.0f, 1e-6f);
}

TEST(PulseqFile, NameTables)
{
    pulseq::LabelId label_id{};
    ASSERT_TRUE(pulseq::label_id_for_name("PMC", &label_id));
    EXPECT_EQ(label_id, pulseq::LabelId::Pmc);

    pulseq::HintId hint_id{};
    ASSERT_TRUE(pulseq::hint_id_for_name("TR", &hint_id));
    EXPECT_EQ(hint_id, pulseq::HintId::Tr);

    EXPECT_FALSE(pulseq::label_id_for_name("NOT_A_LABEL", &label_id));
}
