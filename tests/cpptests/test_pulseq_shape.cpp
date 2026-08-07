/**
 * Unit tests for Pulseq's shape codec -- compress_shape and its inverse.
 *
 * The encoder was already relied on by both writers; the decoder is newer, and
 * the property that matters is that the two are inverses over the waveforms
 * that actually occur: constant runs, linear ramps, and shapes too irregular
 * to compress at all (which the encoder leaves alone, and which the decoder
 * therefore has to recognise and pass through).
 *
 * The decoder is a port of pulseq_decompress_shape in the C reader, so the
 * cases below are also what keeps the two implementations from drifting.
 */

#include <gtest/gtest.h>

#include "pulseq/shape.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace
{
    /* Encode then decode, and say how far the round trip moved each sample. */
    double round_trip_error(const std::vector<double>& samples, bool* out_compressed = nullptr)
    {
        const std::vector<double> packed =
            pulseq::compress_shape(samples.data(), static_cast<int>(samples.size()));
        if (out_compressed)
            *out_compressed = packed.size() < samples.size();

        const std::vector<double> back = pulseq::decompress_shape(
            packed.data(), static_cast<int>(packed.size()), static_cast<int>(samples.size()));

        EXPECT_EQ(back.size(), samples.size());
        if (back.size() != samples.size())
            return std::numeric_limits<double>::infinity();

        double worst = 0.0;
        for (size_t i = 0; i < samples.size(); ++i)
            worst = std::max(worst, std::fabs(back[i] - samples[i]));
        return worst;
    }
}  // namespace

/* The codec quantises the derivative onto a 1e-7 grid, so that is the scale
 * the round trip is allowed to move a sample by. */
constexpr double kQuantTolerance = 1e-6;

TEST(PulseqShapeCodec, ConstantWaveformRoundTrips)
{
    const std::vector<double> samples(256, 0.75);
    bool compressed = false;
    EXPECT_LT(round_trip_error(samples, &compressed), kQuantTolerance);
    EXPECT_TRUE(compressed) << "a constant run is the case the encoding exists for";
}

TEST(PulseqShapeCodec, LinearRampRoundTrips)
{
    /* The step is an exact multiple of the 1e-7 quantisation grid.  That is
     * not a detail of the test: a ramp whose step is not (1/255, say) picks up
     * the encoder's correction term unevenly, its quantised derivative
     * alternates between two adjacent integers, and the run-length encoding
     * has nothing to collapse.  Real gradient ramps land on the raster the
     * same way, which is why the flat-topped shapes below are the ones that
     * compress unconditionally. */
    std::vector<double> samples(512);
    for (size_t i = 0; i < samples.size(); ++i)
        samples[i] = static_cast<double>(i) * 0.001;

    bool compressed = false;
    EXPECT_LT(round_trip_error(samples, &compressed), kQuantTolerance);
    EXPECT_TRUE(compressed) << "a ramp differentiates to a constant";
}

TEST(PulseqShapeCodec, RampOffTheQuantisationGridStillRoundTrips)
{
    /* The companion to the above: it does not compress, but it must still come
     * back within the quantisation tolerance. */
    std::vector<double> samples(512);
    for (size_t i = 0; i < samples.size(); ++i)
        samples[i] = static_cast<double>(i) / static_cast<double>(samples.size() - 1);

    bool compressed = true;
    EXPECT_LT(round_trip_error(samples, &compressed), kQuantTolerance);
    EXPECT_FALSE(compressed);
}

TEST(PulseqShapeCodec, TrapezoidRoundTrips)
{
    std::vector<double> samples;
    for (int i = 0; i < 50; ++i)
        samples.push_back(static_cast<double>(i) / 49.0);
    for (int i = 0; i < 200; ++i)
        samples.push_back(1.0);
    for (int i = 0; i < 50; ++i)
        samples.push_back(1.0 - static_cast<double>(i) / 49.0);

    bool compressed = false;
    EXPECT_LT(round_trip_error(samples, &compressed), kQuantTolerance);
    EXPECT_TRUE(compressed);
}

TEST(PulseqShapeCodec, IrregularWaveformIsPassedThroughUnchanged)
{
    /* Deterministic but with no repeated derivative, so compress_shape's
     * "keep the original if it is no longer" rule keeps the original. */
    std::vector<double> samples(64);
    for (size_t i = 0; i < samples.size(); ++i)
        samples[i] = std::sin(static_cast<double>(i) * 1.7) * std::cos(static_cast<double>(i) * 0.3);

    bool compressed = true;
    EXPECT_EQ(round_trip_error(samples, &compressed), 0.0)
        << "an uncompressed entry must survive exactly, not approximately";
    EXPECT_FALSE(compressed);
}

TEST(PulseqShapeCodec, ShortShapesAreNeverCompressed)
{
    const std::vector<double> samples{0.1, 0.2, 0.3, 0.4};
    bool compressed = true;
    EXPECT_EQ(round_trip_error(samples, &compressed), 0.0);
    EXPECT_FALSE(compressed);
}

TEST(PulseqShapeCodec, SingleSampleRoundTrips)
{
    const std::vector<double> samples{0.42};
    EXPECT_EQ(round_trip_error(samples), 0.0);
}

/* ================================================================== */
/*  Malformed input                                                    */
/* ================================================================== */

TEST(PulseqShapeCodec, RunLengthPastTheEndThrows)
{
    /* A repeated pair with no count after it. */
    const std::vector<double> packed{0.5, 0.5};
    EXPECT_THROW(
        pulseq::decompress_shape(packed.data(), static_cast<int>(packed.size()), 10),
        std::runtime_error);
}

TEST(PulseqShapeCodec, FractionalRunLengthThrows)
{
    const std::vector<double> packed{0.5, 0.5, 2.5, 0.1};
    EXPECT_THROW(
        pulseq::decompress_shape(packed.data(), static_cast<int>(packed.size()), 10),
        std::runtime_error);
}

TEST(PulseqShapeCodec, RunOverrunningTheSampleCountThrows)
{
    /* Claims 102 repeats into a shape declared as 8 samples. */
    const std::vector<double> packed{0.5, 0.5, 100.0, 0.1};
    EXPECT_THROW(
        pulseq::decompress_shape(packed.data(), static_cast<int>(packed.size()), 8),
        std::runtime_error);
}

TEST(PulseqShapeCodec, EmptyShapeIsEmpty)
{
    const std::vector<double> packed{1.0};
    EXPECT_TRUE(pulseq::decompress_shape(packed.data(), 1, 0).empty());
    EXPECT_TRUE(pulseq::decompress_shape(packed.data(), 0, 4).empty());
}
