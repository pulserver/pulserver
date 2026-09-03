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
#include "pulseq/sequence.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace
{
/* Encode then decode, and say how far the round trip moved each sample. */
double round_trip_error(const std::vector<double> &samples, bool *out_compressed = nullptr)
{
    const std::vector<double> packed =
        pulseq::compress_shape(samples.data(), static_cast<int>(samples.size()));
    if (out_compressed)
        *out_compressed = packed.size() < samples.size();

    const std::vector<double> back = pulseq::decompress_shape(
        packed.data(),
        static_cast<int>(packed.size()),
        static_cast<int>(samples.size()));

    EXPECT_EQ(back.size(), samples.size());
    if (back.size() != samples.size())
        return std::numeric_limits<double>::infinity();

    double worst = 0.0;
    for (size_t i = 0; i < samples.size(); ++i)
        worst = std::max(worst, std::fabs(back[i] - samples[i]));
    return worst;
}
} // namespace

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
        samples[i] =
            std::sin(static_cast<double>(i) * 1.7) * std::cos(static_cast<double>(i) * 0.3);

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

namespace
{
std::vector<double> irregular(int count, double seed)
{
    std::vector<double> out(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i)
        out[static_cast<size_t>(i)] = std::sin(seed * (i + 1)) + 1e-3 * (i % 7);
    return out;
}
} // namespace

TEST(PulseqShapeLibrary, CompressLeavesAnIncompressibleLibraryAsItStands)
{
    pulseq::ShapeLibrary lib;
    const std::vector<double> a = irregular(300, 0.37);
    const std::vector<double> b = irregular(200, 1.91);
    lib.append_raw(a.data(), static_cast<int>(a.size()));
    lib.append_raw(b.data(), static_cast<int>(b.size()));

    EXPECT_FALSE(lib.compress()) << "no row shrank, so no row changed";
    EXPECT_TRUE(lib.is_compressed(1));
    EXPECT_TRUE(lib.is_compressed(2));
    ASSERT_EQ(lib.num_compressed(1), static_cast<int>(a.size()));
    ASSERT_EQ(lib.num_compressed(2), static_cast<int>(b.size()));
    EXPECT_EQ(std::vector<double>(lib.samples(1), lib.samples(1) + a.size()), a);
    EXPECT_EQ(std::vector<double>(lib.samples(2), lib.samples(2) + b.size()), b);
    EXPECT_FALSE(lib.compress()) << "idempotent";
}

TEST(PulseqShapeLibrary, CompressRebuildsAMixedLibraryRowForRowInIdOrder)
{
    pulseq::ShapeLibrary lib;
    const std::vector<double> flat(128, 0.25);
    const std::vector<double> rough = irregular(96, 0.71);
    const std::vector<double> ramp = []
    {
        std::vector<double> r(64);
        for (size_t i = 0; i < r.size(); ++i)
            r[i] = 1e-3 * static_cast<double>(i);
        return r;
    }();
    const std::vector<double> ramp_packed =
        pulseq::compress_shape(ramp.data(), static_cast<int>(ramp.size()));
    lib.append_raw(flat.data(), static_cast<int>(flat.size()));
    lib.append_raw(rough.data(), static_cast<int>(rough.size()));
    lib.append(
        static_cast<int>(ramp.size()),
        ramp_packed.data(),
        static_cast<int>(ramp_packed.size()));

    EXPECT_TRUE(lib.compress());
    ASSERT_EQ(lib.size(), 3);
    const std::vector<double> flat_packed =
        pulseq::compress_shape(flat.data(), static_cast<int>(flat.size()));
    ASSERT_EQ(lib.num_compressed(1), static_cast<int>(flat_packed.size()));
    EXPECT_EQ(
        std::vector<double>(lib.samples(1), lib.samples(1) + flat_packed.size()),
        flat_packed);
    ASSERT_EQ(lib.num_compressed(2), static_cast<int>(rough.size()));
    EXPECT_EQ(std::vector<double>(lib.samples(2), lib.samples(2) + rough.size()), rough);
    ASSERT_EQ(lib.num_compressed(3), static_cast<int>(ramp_packed.size()));
    EXPECT_EQ(
        std::vector<double>(lib.samples(3), lib.samples(3) + ramp_packed.size()),
        ramp_packed);
    for (int id = 1; id <= 3; ++id)
        EXPECT_TRUE(lib.is_compressed(id));
    EXPECT_FALSE(lib.compress()) << "idempotent";
}

TEST(PulseqShapeLibrary, KeepFirstAppearancesDropsRepeatsInPlaceAndRenumbersDensely)
{
    pulseq::ShapeLibrary lib;
    const std::vector<double> a = irregular(50, 0.13);
    const std::vector<double> b = irregular(70, 0.29);
    const std::vector<double> c = irregular(30, 0.41);
    const std::vector<double> b_packed =
        pulseq::compress_shape(b.data(), static_cast<int>(b.size()));
    lib.append_raw(a.data(), static_cast<int>(a.size()));                                       // 1
    lib.append(static_cast<int>(b.size()), b_packed.data(), static_cast<int>(b_packed.size())); // 2
    lib.append_raw(a.data(), static_cast<int>(a.size())); // 3 = 1
    lib.append_raw(c.data(), static_cast<int>(c.size())); // 4
    lib.append(
        static_cast<int>(b.size()),
        b_packed.data(),
        static_cast<int>(b_packed.size())); // 5 = 2

    const std::vector<int32_t> first{0, 1, 2, 1, 4, 2};
    const std::vector<int32_t> renumbered = lib.keep_first_appearances(first);

    ASSERT_EQ(lib.size(), 3);
    EXPECT_EQ(renumbered, (std::vector<int32_t>{0, 1, 2, 1, 3, 2}));
    ASSERT_EQ(lib.num_compressed(1), static_cast<int>(a.size()));
    EXPECT_EQ(std::vector<double>(lib.samples(1), lib.samples(1) + a.size()), a);
    EXPECT_FALSE(lib.is_compressed(1));
    ASSERT_EQ(lib.num_compressed(2), static_cast<int>(b_packed.size()));
    EXPECT_EQ(std::vector<double>(lib.samples(2), lib.samples(2) + b_packed.size()), b_packed);
    EXPECT_EQ(lib.num_uncompressed(2), static_cast<int>(b.size()));
    EXPECT_TRUE(lib.is_compressed(2));
    ASSERT_EQ(lib.num_compressed(3), static_cast<int>(c.size()));
    EXPECT_EQ(std::vector<double>(lib.samples(3), lib.samples(3) + c.size()), c);
    EXPECT_FALSE(lib.is_compressed(3));
}

TEST(PulseqShapeLibrary, KeepFirstAppearancesOfADistinctLibraryChangesNothing)
{
    pulseq::ShapeLibrary lib;
    const std::vector<double> a = irregular(40, 0.53);
    const std::vector<double> b = irregular(60, 0.67);
    lib.append_raw(a.data(), static_cast<int>(a.size()));
    lib.append_raw(b.data(), static_cast<int>(b.size()));
    const std::vector<int32_t> renumbered = lib.keep_first_appearances({0, 1, 2});
    EXPECT_EQ(renumbered, (std::vector<int32_t>{0, 1, 2}));
    ASSERT_EQ(lib.size(), 2);
    EXPECT_EQ(std::vector<double>(lib.samples(1), lib.samples(1) + a.size()), a);
    EXPECT_EQ(std::vector<double>(lib.samples(2), lib.samples(2) + b.size()), b);
}

/* The library row a divided append stores is the row an append of the
 * already-divided samples stores, bit for bit, with the same statistics. */
TEST(PulseqShapeLibrary, DividedAppendStoresTheDividedRow)
{
    std::vector<double> samples(4096);
    for (size_t i = 0; i < samples.size(); ++i)
        samples[i] = -0.37 * std::sin(0.013 * static_cast<double>(i)) *
            (1.0 + 0.001 * static_cast<double>(i % 7));
    double peak = 0.0;
    for (double v : samples)
        peak = std::max(peak, std::fabs(v));
    peak = -peak; /* the first nonzero sample is negative */
    std::vector<double> divided(samples.size());
    for (size_t i = 0; i < samples.size(); ++i)
        divided[i] = samples[i] / peak;

    pulseq::ShapeLibrary by_hand, fused;
    const int a = by_hand.append_raw(divided.data(), static_cast<int>(divided.size()));
    const int b = fused.append_raw_divided(samples.data(), static_cast<int>(samples.size()), peak);
    ASSERT_EQ(a, b);
    ASSERT_EQ(by_hand.num_uncompressed(a), fused.num_uncompressed(b));
    const double *ra = by_hand.samples(a);
    const double *rb = fused.samples(b);
    for (int i = 0; i < by_hand.num_uncompressed(a); ++i)
        ASSERT_EQ(ra[i], rb[i]) << "sample " << i;
    double fa, la, pa, fb, lb, pb;
    by_hand.edge_stats(a, &fa, &la, &pa);
    fused.edge_stats(b, &fb, &lb, &pb);
    EXPECT_EQ(fa, fb);
    EXPECT_EQ(la, lb);
    EXPECT_EQ(pa, pb);
    EXPECT_EQ(pb, 1.0);
}

TEST(PulseqShapeLibrary, DividedAppendWithAUnitOrZeroDivisorIsAPlainAppend)
{
    const std::vector<double> samples = {0.0, 0.25, -0.5, 1.0, 0.75, 0.0};
    pulseq::ShapeLibrary plain, unit, zero;
    plain.append_raw(samples.data(), 6);
    unit.append_raw_divided(samples.data(), 6, 1.0);
    zero.append_raw_divided(samples.data(), 6, 0.0);
    for (int i = 0; i < 6; ++i)
    {
        EXPECT_EQ(plain.samples(1)[i], unit.samples(1)[i]);
        EXPECT_EQ(plain.samples(1)[i], zero.samples(1)[i]);
    }
    double f, l, p;
    unit.edge_stats(1, &f, &l, &p);
    EXPECT_EQ(p, 1.0);
}

/* Rows keep their bytes across chunk boundaries however the chunks are
 * backed: the table is read back row by row after it has spilled into a
 * second chunk. */
TEST(PulseqRaggedTable, RowsSurviveASecondChunk)
{
    pulseq::ShapeLibrary lib;
    std::vector<double> row(1 << 16);
    const int rows = 80; /* 80 x 512 KB > one 32 MB chunk */
    for (int r = 0; r < rows; ++r)
    {
        for (size_t i = 0; i < row.size(); ++i)
            row[i] = static_cast<double>(r) + 1e-3 * static_cast<double>(i);
        lib.append_raw(row.data(), static_cast<int>(row.size()));
    }
    for (int r = 0; r < rows; ++r)
    {
        const double *p = lib.samples(r + 1);
        ASSERT_EQ(lib.num_uncompressed(r + 1), static_cast<int>(row.size()));
        EXPECT_EQ(p[0], static_cast<double>(r));
        EXPECT_EQ(
            p[row.size() - 1],
            static_cast<double>(r) + 1e-3 * static_cast<double>(row.size() - 1));
    }
}
