/**
 * Unit tests for pulseq::base_trajectory -- the k-space a block's ADC window
 * traverses, with the amplitude left outside it.
 *
 * The properties worth pinning down are the ones the FOV-shift design rests
 * on: that the base is the same array for two instances that differ only in
 * amplitude (which is what makes it deduplicate), that scaling it by the
 * amplitude reproduces real k, that a flat axis is reported as flat (so a
 * consumer knows to take its DC from the encoding counters instead), and that
 * the whole-block moment is the exact area -- because a caller accumulates the
 * DC across a TR out of those.
 */

#include <gtest/gtest.h>

#include "pulseq/read.hpp"
#include "pulseq/trajectory.hpp"
#include "pulseq/write.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <utility>

using pulseq::Block;
using pulseq::BaseTrajectory;
using pulseq::Sequence;

namespace
{
    /* A trapezoid: 100 us up, `flat` flat, 100 us down, no delay. */
    std::array<double, pulseq::TRAP_WIDTH> trap(double amplitude, double flat)
    {
        return {amplitude, 100e-6, flat, 100e-6, 0.0};
    }

    std::array<double, pulseq::ADC_WIDTH> adc(double num_samples, double dwell, double delay)
    {
        return {num_samples, dwell, delay, 0, 0, 0, 0, 0};
    }

    /* A block whose gx is a trapezoid and whose ADC sits on its flat top. */
    int add_readout(Sequence& seq, double amplitude, double flat, double dwell, int samples)
    {
        const auto g = trap(amplitude, flat);
        const auto a = adc(samples, dwell, 100e-6);

        Block b;
        b.gx = seq.register_trap(g.data());
        b.adc = seq.register_adc(a.data());
        b.duration = 100e-6 + flat + 100e-6;
        return seq.add_block(b);
    }

    /* The trapezoid row block 1's gx points at. */
    int seq_first_trap(const Sequence& seq)
    {
        return seq.grad_row(seq.get_block(1).gx);
    }
}  // namespace

/* ================================================================== */
/*  Instance independence -- the point of the whole thing              */
/* ================================================================== */

TEST(PulseqTrajectory, BaseIsIdenticalAcrossAmplitudes)
{
    Sequence seq;
    const int weak = add_readout(seq, 1000.0, 640e-6, 10e-6, 64);
    const int strong = add_readout(seq, -75000.0, 640e-6, 10e-6, 64);

    const BaseTrajectory a = pulseq::base_trajectory(seq, weak);
    const BaseTrajectory b = pulseq::base_trajectory(seq, strong);

    ASSERT_TRUE(a.has_adc);
    ASSERT_TRUE(b.has_adc);
    ASSERT_EQ(a.axis[0].base.size(), b.axis[0].base.size());

    /* Bit-identical, not merely close: this is what a shape library
     * deduplicates on. */
    for (size_t i = 0; i < a.axis[0].base.size(); ++i)
        EXPECT_DOUBLE_EQ(a.axis[0].base[i], b.axis[0].base[i]) << "sample " << i;

    EXPECT_DOUBLE_EQ(a.axis[0].amplitude, 1000.0);
    EXPECT_DOUBLE_EQ(b.axis[0].amplitude, -75000.0);
}

TEST(PulseqTrajectory, AmplitudeTimesBaseIsRealKSpace)
{
    Sequence seq;
    const int blk = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);

    const BaseTrajectory t = pulseq::base_trajectory(seq, blk);
    ASSERT_TRUE(t.has_adc);

    /* The ADC starts at 100 us, exactly when the ramp ends, so by sample i
     * the unit waveform has swept the whole ramp (area 0.5 * 100us) plus
     * (i + 0.5) dwells of flat top. */
    const double ramp_area = 0.5 * 100e-6;
    for (int i = 0; i < t.num_samples; ++i)
    {
        const double expected_base = ramp_area + (static_cast<double>(i) + 0.5) * 10e-6;
        const double k = t.axis[0].amplitude * t.axis[0].base[static_cast<size_t>(i)];
        EXPECT_NEAR(k, 50000.0 * expected_base, 1e-9) << "sample " << i;
    }
}

/* ================================================================== */
/*  Flatness -- the trivial-shot classifier                            */
/* ================================================================== */

TEST(PulseqTrajectory, AxisFlatAcrossTheWindowIsReportedConstant)
{
    Sequence seq;
    /* ADC entirely on the flat top. */
    const int blk = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);

    const BaseTrajectory t = pulseq::base_trajectory(seq, blk);
    EXPECT_TRUE(t.axis[0].constant);
    /* Still integrated: a flat axis accumulates k, it just carries no shape. */
    EXPECT_GT(t.axis[0].base.back(), t.axis[0].base.front());
}

TEST(PulseqTrajectory, RampInsideTheWindowIsNotConstant)
{
    Sequence seq;
    const auto g = trap(50000.0, 640e-6);
    const auto a = adc(64, 10e-6, 0.0);  /* starts at t=0, over the ramp */

    Block b;
    b.gx = seq.register_trap(g.data());
    b.adc = seq.register_adc(a.data());
    b.duration = 840e-6;
    const int blk = seq.add_block(b);

    EXPECT_FALSE(pulseq::base_trajectory(seq, blk).axis[0].constant);
}

TEST(PulseqTrajectory, AbsentAxesAreEmptyAndConstant)
{
    Sequence seq;
    const int blk = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);

    const BaseTrajectory t = pulseq::base_trajectory(seq, blk);
    for (int axis : {1, 2})
    {
        EXPECT_FALSE(t.axis[axis].present) << "axis " << axis;
        EXPECT_TRUE(t.axis[axis].base.empty()) << "axis " << axis;
        EXPECT_TRUE(t.axis[axis].constant) << "axis " << axis;
        EXPECT_DOUBLE_EQ(t.axis[axis].moment, 0.0) << "axis " << axis;
    }
}

/* ================================================================== */
/*  Moments -- what a caller accumulates the DC out of                 */
/* ================================================================== */

TEST(PulseqTrajectory, WholeBlockMomentIsTheTrapezoidArea)
{
    Sequence seq;
    const auto g = trap(30000.0, 500e-6);
    Block b;
    b.gx = seq.register_trap(g.data());
    b.duration = 700e-6;
    seq.add_block(b);

    /* Unit trapezoid: flat + one ramp (the two half-ramps together). */
    const double expected = 500e-6 + 100e-6;
    EXPECT_NEAR(pulseq::gradient_moment(seq, seq.get_block(1).gx), expected, 1e-15);

    /* And k after a prewinder is amplitude * moment -- the DC a later block
     * inherits. */
    EXPECT_NEAR(30000.0 * expected, 18.0, 1e-12);
}

TEST(PulseqTrajectory, MomentOfAnAbsentGradientIsZero)
{
    Sequence seq;
    EXPECT_DOUBLE_EQ(pulseq::gradient_moment(seq, 0), 0.0);
}

/* ================================================================== */
/*  Arbitrary gradients                                                */
/* ================================================================== */

TEST(PulseqTrajectory, ArbitraryBaseIntegratesTheNormalisedShape)
{
    Sequence seq;
    seq.set_rasters(1e-6, 10e-6, 100e-9, 10e-6);

    /* A flat normalised waveform of 1.0: its integral is just elapsed time. */
    const std::vector<double> samples(8, 1.0);
    const int shape =
        seq.register_shape(static_cast<int>(samples.size()), samples.data(),
                           static_cast<int>(samples.size()));

    std::array<double, pulseq::ARB_WIDTH> g{};
    g[0] = 20000.0;                        /* amplitude */
    g[1] = 20000.0;                        /* first     */
    g[2] = 20000.0;                        /* last      */
    g[3] = static_cast<double>(shape);     /* amp shape */
    g[4] = 0.0;                            /* time shape: default raster */
    g[5] = 0.0;                            /* delay     */

    const auto a = adc(4, 10e-6, 0.0);
    Block b;
    b.gx = seq.register_arbitrary(g.data());
    b.adc = seq.register_adc(a.data());
    b.duration = 80e-6;
    const int blk = seq.add_block(b);

    const BaseTrajectory t = pulseq::base_trajectory(seq, blk);
    ASSERT_TRUE(t.has_adc);
    EXPECT_TRUE(t.axis[0].constant);

    for (int i = 0; i < t.num_samples; ++i)
    {
        const double t_i = (static_cast<double>(i) + 0.5) * 10e-6;
        EXPECT_NEAR(t.axis[0].base[static_cast<size_t>(i)], t_i, 1e-15) << "sample " << i;
    }
    EXPECT_NEAR(t.axis[0].moment, 80e-6, 1e-15);
}

TEST(PulseqTrajectory, ArbitraryBaseIsAmplitudeIndependent)
{
    Sequence seq;
    seq.set_rasters(1e-6, 10e-6, 100e-9, 10e-6);

    /* A ramp, so the base actually has structure to compare. */
    std::vector<double> samples(16);
    for (size_t i = 0; i < samples.size(); ++i)
        samples[i] = static_cast<double>(i) / static_cast<double>(samples.size() - 1);
    const int shape =
        seq.register_shape(static_cast<int>(samples.size()), samples.data(),
                           static_cast<int>(samples.size()));

    auto make = [&](double amplitude)
    {
        std::array<double, pulseq::ARB_WIDTH> g{};
        g[0] = amplitude;
        g[1] = 0.0;
        g[2] = amplitude;
        g[3] = static_cast<double>(shape);
        g[4] = 0.0;
        g[5] = 0.0;

        const auto a = adc(8, 20e-6, 0.0);
        Block b;
        b.gx = seq.register_arbitrary(g.data());
        b.adc = seq.register_adc(a.data());
        b.duration = 160e-6;
        return seq.add_block(b);
    };

    const BaseTrajectory lo = pulseq::base_trajectory(seq, make(500.0));
    const BaseTrajectory hi = pulseq::base_trajectory(seq, make(-40000.0));

    ASSERT_EQ(lo.axis[0].base.size(), hi.axis[0].base.size());
    for (size_t i = 0; i < lo.axis[0].base.size(); ++i)
        EXPECT_DOUBLE_EQ(lo.axis[0].base[i], hi.axis[0].base[i]) << "sample " << i;

    EXPECT_FALSE(lo.axis[0].constant);
    EXPECT_DOUBLE_EQ(lo.axis[0].moment, hi.axis[0].moment);
}

TEST(PulseqTrajectory, CompressedShapesGiveTheSameBaseAsRawOnes)
{
    /* A sequence read back from a file holds its shapes encoded, so the
     * encoded path has to agree with the raw one it will be compared against. */
    Sequence raw;
    Sequence packed;
    for (Sequence* seq : {&raw, &packed})
        seq->set_rasters(1e-6, 10e-6, 100e-9, 10e-6);

    /* A flat-topped waveform, which is what compress_shape is built for and
     * what a readout gradient actually looks like.  The ramp steps are exact
     * multiples of the codec's 1e-7 grid so the ramps collapse too. */
    std::vector<double> samples;
    for (int i = 0; i < 32; ++i)
        samples.push_back(static_cast<double>(i) * 0.03125);
    for (int i = 0; i < 192; ++i)
        samples.push_back(1.0);
    for (int i = 0; i < 32; ++i)
        samples.push_back(1.0 - static_cast<double>(i) * 0.03125);
    ASSERT_EQ(samples.size(), 256u);

    auto build = [&](Sequence& seq)
    {
        const int shape = seq.register_raw_shape(samples.data(), static_cast<int>(samples.size()));
        std::array<double, pulseq::ARB_WIDTH> g{};
        g[0] = 12345.0;
        g[1] = 0.0;
        g[2] = 12345.0;
        g[3] = static_cast<double>(shape);
        g[4] = 0.0;
        g[5] = 0.0;

        const auto a = adc(128, 20e-6, 0.0);
        Block b;
        b.gx = seq.register_arbitrary(g.data());
        b.adc = seq.register_adc(a.data());
        b.duration = 2560e-6;
        return seq.add_block(b);
    };

    const int raw_blk = build(raw);
    const int packed_blk = build(packed);
    packed.compress_shapes();
    ASSERT_TRUE(packed.shape_library().is_compressed(1));
    ASSERT_LT(packed.shape_library().num_compressed(1), 256)
        << "this fixture is meant to exercise the encoded path";

    const BaseTrajectory a = pulseq::base_trajectory(raw, raw_blk);
    const BaseTrajectory b = pulseq::base_trajectory(packed, packed_blk);

    ASSERT_EQ(a.axis[0].base.size(), b.axis[0].base.size());
    ASSERT_FALSE(a.axis[0].base.empty());
    for (size_t i = 0; i < a.axis[0].base.size(); ++i)
    {
        /* The codec quantises the derivative, so agreement is to that grid
         * scaled by the integration window, not bit-for-bit. */
        EXPECT_NEAR(a.axis[0].base[i], b.axis[0].base[i], 1e-12) << "sample " << i;
    }
    EXPECT_NEAR(a.axis[0].moment, b.axis[0].moment, 1e-12);
    EXPECT_EQ(a.axis[0].constant, b.axis[0].constant);
}

/* ================================================================== */
/*  Blocks without a readout                                           */
/* ================================================================== */

TEST(PulseqTrajectory, BlockWithoutAdcReportsNoAdcButStillCarriesMoments)
{
    Sequence seq;
    const auto g = trap(30000.0, 500e-6);
    Block b;
    b.gy = seq.register_trap(g.data());
    b.duration = 700e-6;
    const int blk = seq.add_block(b);

    const BaseTrajectory t = pulseq::base_trajectory(seq, blk);
    EXPECT_FALSE(t.has_adc);
    EXPECT_TRUE(t.axis[1].present);
    EXPECT_NEAR(t.axis[1].moment, 600e-6, 1e-15);
    EXPECT_TRUE(t.axis[1].base.empty());
}

TEST(PulseqTrajectory, OutOfRangeBlockThrows)
{
    Sequence seq;
    EXPECT_THROW(pulseq::base_trajectory(seq, 1), std::out_of_range);
    EXPECT_THROW(pulseq::base_trajectory(seq, 0), std::out_of_range);
}

/* ================================================================== */
/*  Absolute k                                                         */
/* ================================================================== */

namespace
{
    /* An RF row: amplitude, mag shape, phase shape, time shape, center, delay. */
    std::array<double, pulseq::RF_WIDTH> rf_row(int mag_shape, int phase_shape, double center,
                                                double delay)
    {
        std::array<double, pulseq::RF_WIDTH> r{};
        r[0] = 1000.0;
        r[1] = static_cast<double>(mag_shape);
        r[2] = static_cast<double>(phase_shape);
        r[3] = 0.0;
        r[4] = center;
        r[5] = delay;
        return r;
    }

    /* Add a hard pulse of `use`, returning the block index. */
    int add_pulse(Sequence& seq, char use, double duration)
    {
        const std::vector<double> mag(8, 1.0);
        const std::vector<double> phase(8, 0.0);
        const int m = seq.register_raw_shape(mag.data(), 8);
        const int p = seq.register_raw_shape(phase.data(), 8);

        const auto r = rf_row(m, p, 0.5 * duration, 0.0);
        Block b;
        b.rf = seq.register_rf(r.data(), use);
        b.duration = duration;
        return seq.add_block(b);
    }

    /* A gradient-only block, returning its index. */
    int add_gradient(Sequence& seq, double amplitude, double flat)
    {
        const auto g = trap(amplitude, flat);
        Block b;
        b.gx = seq.register_trap(g.data());
        b.duration = 100e-6 + flat + 100e-6;
        return seq.add_block(b);
    }
}  // namespace

TEST(PulseqKOrigins, GradientsAccumulate)
{
    Sequence seq;
    add_gradient(seq, 30000.0, 500e-6);  // moment 600e-6 * 30000 = 18 /m
    add_gradient(seq, 30000.0, 500e-6);

    const auto origins = pulseq::block_k_origins(seq);
    ASSERT_EQ(origins.size(), 3u);
    EXPECT_NEAR(origins[1][0], 0.0, 1e-12);
    EXPECT_NEAR(origins[2][0], 18.0, 1e-9);
}

TEST(PulseqKOrigins, ExcitationResetsK)
{
    Sequence seq;
    add_gradient(seq, 30000.0, 500e-6);
    add_pulse(seq, 'e', 200e-6);
    add_gradient(seq, 30000.0, 500e-6);

    const auto origins = pulseq::block_k_origins(seq);
    EXPECT_NEAR(origins[2][0], 18.0, 1e-9) << "k before the excitation";
    EXPECT_NEAR(origins[3][0], 0.0, 1e-12) << "the excitation resets it";
}

TEST(PulseqKOrigins, RefocusingNegatesK)
{
    Sequence seq;
    add_gradient(seq, 30000.0, 500e-6);   // k = +18
    add_pulse(seq, 'r', 200e-6);          // k = -18
    add_gradient(seq, 30000.0, 500e-6);   // k = 0 -- the echo

    const auto origins = pulseq::block_k_origins(seq);
    EXPECT_NEAR(origins[2][0], 18.0, 1e-9);
    EXPECT_NEAR(origins[3][0], -18.0, 1e-9) << "a refocusing pulse flips the sign";

    /* The point of the sign flip: the spin echo lands back at the origin. */
    Sequence check = seq;
    add_gradient(check, 0.0, 0.0);
    const auto after = pulseq::block_k_origins(check);
    EXPECT_NEAR(after[4][0], 0.0, 1e-9) << "the echo returns to k = 0";
}

TEST(PulseqKOrigins, InversionAndSaturationLeaveKAlone)
{
    for (char use : {'i', 's', 'p'})
    {
        Sequence seq;
        add_gradient(seq, 30000.0, 500e-6);
        add_pulse(seq, use, 200e-6);

        const auto origins = pulseq::block_k_origins(seq);
        EXPECT_NEAR(origins[2][0], 18.0, 1e-9) << "use '" << use << "'";
    }
}

/*
 * A pulse that does not say what it is for is refused.
 *
 * This used to answer "it leaves k alone", which is the one guess that costs
 * the most when it is wrong: an excitation resets k, so reading it as neutral
 * means k never restarts and accumulates over the whole scan. Measured on a
 * 32-line GRE whose rows carried no use, before that fixture was corrected:
 * the last block's origin was 2.8e4 1/m against a k-max of the same size --
 * a 98% error in the number TransformFOV builds its phase on, arrived at
 * without a word.
 *
 * PyPulseq and Pulseq both guess "excitation" here. Refusing is the stricter
 * reading and the one the `use` field exists to make possible.
 */
TEST(PulseqKOrigins, AnUndeclaredUseIsRefusedRatherThanGuessed)
{
    Sequence seq;
    add_gradient(seq, 30000.0, 500e-6);
    add_pulse(seq, 'u', 200e-6);

    EXPECT_THROW(pulseq::block_k_origins(seq), std::runtime_error);
}

TEST(PulseqKOrigins, CrushersSplitAroundTheRefocusingCentre)
{
    /* A refocusing pulse between two crushers: the first is negated with the
     * k before it, the second is not.  Symmetric crushers therefore cancel,
     * which is the whole reason they are drawn that way. */
    Sequence seq;
    seq.set_rasters(1e-6, 10e-6, 100e-9, 10e-6);

    const std::vector<double> mag(8, 1.0);
    const std::vector<double> phase(8, 0.0);
    const int m = seq.register_raw_shape(mag.data(), 8);
    const int p = seq.register_raw_shape(phase.data(), 8);

    /* Gradient spans the block; the pulse sits at its centre. */
    const auto g = trap(20000.0, 800e-6);       // total moment 900e-6 * 20000 = 18
    const auto r = rf_row(m, p, 500e-6, 0.0);   // centre at 500 us, mid flat-top

    Block b;
    b.gx = seq.register_trap(g.data());
    b.rf = seq.register_rf(r.data(), 'r');
    b.duration = 1000e-6;
    seq.add_block(b);

    const auto origins = pulseq::block_k_origins(seq);

    /* Before the centre: 100us ramp (area 50us) + 400us flat = 450us * 20000 = 9.
     * Negated to -9, then the remaining 450us * 20000 = +9 brings it to 0. */
    EXPECT_NEAR(origins[1][0], 0.0, 1e-12);
    Sequence tail = seq;
    add_gradient(tail, 0.0, 0.0);
    EXPECT_NEAR(pulseq::block_k_origins(tail)[2][0], 0.0, 1e-9)
        << "a symmetric crusher pair around a refocusing pulse cancels";
}

TEST(PulseqAbsoluteTrajectory, IsOriginPlusAmplitudeTimesBase)
{
    Sequence seq;
    add_gradient(seq, 30000.0, 500e-6);  // prewinder: k = 18
    const int readout = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);

    const auto origins = pulseq::block_k_origins(seq);
    const auto k = pulseq::absolute_trajectory(seq, readout, origins[readout]);
    const BaseTrajectory base = pulseq::base_trajectory(seq, readout);

    ASSERT_EQ(k[0].size(), static_cast<size_t>(base.num_samples));
    for (int i = 0; i < base.num_samples; ++i)
    {
        EXPECT_NEAR(k[0][static_cast<size_t>(i)],
                    18.0 + 50000.0 * base.axis[0].base[static_cast<size_t>(i)], 1e-9)
            << "sample " << i;
    }
    EXPECT_TRUE(k[1].empty());
    EXPECT_TRUE(k[2].empty());
}

/* ================================================================== */
/*  FOV shift                                                          */
/* ================================================================== */

TEST(PulseqFovShift, ConstantGradientCostsTwoScalarsAndNoShape)
{
    /* The compactness claim: a Cartesian readout under a flat gradient needs
     * a frequency and a phase, and no per-sample residual at all. */
    Sequence seq;
    const int blk = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);

    const int shapes_before = seq.shape_library().size();
    pulseq::apply_fov_shift(seq, {0.01, 0.0, 0.0}, pulseq::FovShiftScope::RfAndAdc);

    const double* row = seq.adc_library().row(seq.get_block(blk).adc);
    EXPECT_NE(row[5], 0.0) << "a frequency offset";
    EXPECT_EQ(row[7], 0.0) << "and no residual shape";
    EXPECT_EQ(seq.shape_library().size(), shapes_before);

    /* freq = shift . g = 0.01 m * 50000 Hz/m */
    EXPECT_NEAR(row[5], 500.0, 1e-9);
}

TEST(PulseqFovShift, VaryingGradientNeedsAResidual)
{
    Sequence seq;
    const auto g = trap(50000.0, 640e-6);
    const auto a = adc(64, 10e-6, 0.0);  // starts on the ramp

    Block b;
    b.gx = seq.register_trap(g.data());
    b.adc = seq.register_adc(a.data());
    b.duration = 840e-6;
    const int blk = seq.add_block(b);

    pulseq::apply_fov_shift(seq, {0.01, 0.0, 0.0}, pulseq::FovShiftScope::RfAndAdc);

    EXPECT_NE(seq.adc_library().row(seq.get_block(blk).adc)[7], 0.0)
        << "a ramp cannot be expressed as a frequency offset";
}

TEST(PulseqFovShift, ReconstructedPhaseMatchesTheShiftTimesK)
{
    /* Frequency + phase + residual must add back up to 2*pi*dr.k -- that is
     * the whole contract of the decomposition. */
    Sequence seq;
    const auto g = trap(50000.0, 640e-6);
    const auto a = adc(64, 10e-6, 0.0);

    Block b;
    b.gx = seq.register_trap(g.data());
    b.adc = seq.register_adc(a.data());
    b.duration = 840e-6;
    const int blk = seq.add_block(b);

    const auto origins = pulseq::block_k_origins(seq);
    const auto k = pulseq::absolute_trajectory(seq, blk, origins[blk]);

    constexpr double shift = 0.013;
    pulseq::apply_fov_shift(seq, {shift, 0.0, 0.0}, pulseq::FovShiftScope::RfAndAdc);

    const double* row = seq.adc_library().row(seq.get_block(blk).adc);
    const int n = static_cast<int>(row[0]);
    const double dwell = row[1];
    const double delay = row[2];
    const double freq = row[5];
    const double phase = row[6];
    const int shape_id = static_cast<int>(row[7]);
    ASSERT_GT(shape_id, 0);

    const auto& shapes = seq.shape_library();
    std::vector<double> residual(shapes.samples(shape_id),
                                 shapes.samples(shape_id) + shapes.num_uncompressed(shape_id));

    constexpr double two_pi = 6.283185307179586476925286766559;
    for (int i = 0; i < n; ++i)
    {
        const double t = delay + (static_cast<double>(i) + 0.5) * dwell;
        const double rebuilt =
            two_pi * freq * (t - delay) + phase + residual[static_cast<size_t>(i)];
        EXPECT_NEAR(rebuilt, two_pi * shift * k[0][static_cast<size_t>(i)], 1e-9)
            << "sample " << i;
    }
}

namespace
{
    /* An 8-sample hard pulse under a slice-select trapezoid.  `delay` places
     * it: 200 us puts every sample on the flat top, 0 puts it on the ramp. */
    int add_selective_pulse(Sequence& seq, double delay, int phase_shape_or_zero)
    {
        seq.set_rasters(1e-6, 10e-6, 100e-9, 10e-6);

        const std::vector<double> mag(8, 1.0);
        const int m = seq.register_raw_shape(mag.data(), 8);
        int p = phase_shape_or_zero;
        if (p < 0)
        {
            const std::vector<double> phase(8, 0.0);
            p = seq.register_raw_shape(phase.data(), 8);
        }

        const auto g = trap(20000.0, 800e-6);
        const auto r = rf_row(m, p, 4e-6, delay);

        Block b;
        b.gx = seq.register_trap(g.data());
        b.rf = seq.register_rf(r.data(), 'e');
        b.duration = 1000e-6;
        return seq.add_block(b);
    }
}  // namespace

TEST(PulseqFovShift, RfUnderAConstantGradientCostsTwoScalarsAndNoShape)
{
    Sequence seq;
    const int blk = add_selective_pulse(seq, 200e-6, -1);
    const double* before = seq.rf_library().row(seq.get_block(blk).rf);
    const int phase_id = static_cast<int>(before[2]);
    const int shapes_before = seq.shape_library().size();

    pulseq::apply_fov_shift(seq, {0.01, 0.0, 0.0}, pulseq::FovShiftScope::RfOnly);

    const double* row = seq.rf_library().row(seq.get_block(blk).rf);
    EXPECT_EQ(static_cast<int>(row[2]), phase_id) << "the phase shape is untouched";
    EXPECT_EQ(seq.shape_library().size(), shapes_before);

    /* freq = shift . g = 0.01 m * 20000 Hz/m, and the phase backs the centre
     * out so the pulse's effective phase there is unchanged. */
    constexpr double two_pi = 6.283185307179586476925286766559;
    EXPECT_NEAR(row[8], 200.0, 1e-9);
    EXPECT_NEAR(row[9], -two_pi * 200.0 * 4e-6, 1e-12);
}

TEST(PulseqFovShift, RfScalarsReproduceTheBakedPhaseExactly)
{
    /* The pair must give every sample the phase the shape path would have
     * baked: `phase + 2*pi*freq*t`, t from the shape's start, equal to
     * `2*pi * dr . (k(t) - k(centre))`. */
    Sequence seq;
    const int blk = add_selective_pulse(seq, 200e-6, -1);

    pulseq::apply_fov_shift(seq, {0.01, 0.0, 0.0}, pulseq::FovShiftScope::RfOnly);

    const double* row = seq.rf_library().row(seq.get_block(blk).rf);
    constexpr double two_pi = 6.283185307179586476925286766559;
    constexpr double shift = 0.01;

    /* k under trap(20000, 800e-6) at a time on the flat top, from block
     * start: full ramp area plus the flat run so far. */
    const auto k_at = [](double t) {
        return 20000.0 * (0.5 * 100e-6 + (t - 100e-6));
    };
    for (int i = 0; i < 8; ++i)
    {
        const double tau = (static_cast<double>(i) + 0.5) * 1e-6;  // from shape start
        const double from_scalars = row[9] + two_pi * row[8] * tau;
        const double baked = two_pi * shift * (k_at(200e-6 + tau) - k_at(200e-6 + 4e-6));
        EXPECT_NEAR(from_scalars, baked, 1e-9) << "sample " << i;
    }
}

TEST(PulseqFovShift, RfOverlappingTheRampStillBakesAShape)
{
    Sequence seq;
    const int blk = add_selective_pulse(seq, 0.0, -1);
    const double* before = seq.rf_library().row(seq.get_block(blk).rf);
    const int phase_id = static_cast<int>(before[2]);

    pulseq::apply_fov_shift(seq, {0.01, 0.0, 0.0}, pulseq::FovShiftScope::RfOnly);

    const double* row = seq.rf_library().row(seq.get_block(blk).rf);
    EXPECT_NE(static_cast<int>(row[2]), phase_id) << "a varying gradient needs the shape";
    EXPECT_EQ(row[8], 0.0);

    /* The pulse centre is at 4 us, inside the first sample, and the phase is
     * referenced there -- so the shape passes through ~0 near the start and
     * varies away from it. */
    const double* samples = seq.shape_library().samples(static_cast<int>(row[2]));
    EXPECT_NEAR(samples[0], 0.0, 2e-3);
    EXPECT_NE(samples[7], samples[0]);
}

TEST(PulseqFovShift, RepeatedRfContextsDedupOntoOneRow)
{
    /* Two blocks, same pulse and gradient, different k origins: because the
     * centre's share is subtracted, the scalar pair is origin-independent and
     * both blocks land on one new rf row -- the footprint claim. */
    Sequence seq;
    seq.set_rasters(1e-6, 10e-6, 100e-9, 10e-6);

    const std::vector<double> mag(8, 1.0);
    const std::vector<double> phase(8, 0.0);
    const int m = seq.register_raw_shape(mag.data(), 8);
    const int p = seq.register_raw_shape(phase.data(), 8);
    const auto g = trap(20000.0, 800e-6);
    const auto r = rf_row(m, p, 4e-6, 200e-6);

    Block b;
    b.gx = seq.register_trap(g.data());
    b.rf = seq.register_rf(r.data(), 'i');  // leaves k alone, so origins differ
    b.duration = 1000e-6;
    const int first = seq.add_block(b);
    const int second = seq.add_block(b);

    const int shapes_before = seq.shape_library().size();

    pulseq::apply_fov_shift(seq, {0.01, 0.0, 0.0}, pulseq::FovShiftScope::RfOnly);

    EXPECT_EQ(seq.shape_library().size(), shapes_before);

    /* Registration appends; the collapse is remove_duplicates' job, and the
     * pair being origin-independent is what makes the rows equal to collapse. */
    seq.remove_duplicates();
    EXPECT_EQ(seq.get_block(first).rf, seq.get_block(second).rf);
}

TEST(PulseqFovShift, AnRfWithoutAPhaseShapeStillGetsTheScalars)
{
    Sequence seq;
    const int blk = add_selective_pulse(seq, 200e-6, 0);

    pulseq::apply_fov_shift(seq, {0.01, 0.0, 0.0}, pulseq::FovShiftScope::RfOnly);

    const double* row = seq.rf_library().row(seq.get_block(blk).rf);
    EXPECT_EQ(static_cast<int>(row[2]), 0) << "still no phase shape";
    EXPECT_NEAR(row[8], 200.0, 1e-9);
}

TEST(PulseqFovShift, RfOnlyLeavesTheAdcUntouched)
{
    Sequence seq;
    const int blk = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);
    const int32_t before = seq.get_block(blk).adc;

    pulseq::apply_fov_shift(seq, {0.01, 0.0, 0.0}, pulseq::FovShiftScope::RfOnly);

    EXPECT_EQ(seq.get_block(blk).adc, before);
    EXPECT_EQ(seq.adc_library().row(before)[5], 0.0);
}

TEST(PulseqFovShift, ServerScopeBakesACartesianReadoutLikeNative)
{
    Sequence server;
    const int blk = add_readout(server, 50000.0, 640e-6, 10e-6, 64);
    Sequence native;
    add_readout(native, 50000.0, 640e-6, 10e-6, 64);

    pulseq::apply_fov_shift(server, {0.01, 0.0, 0.0}, pulseq::FovShiftScope::Server);
    pulseq::apply_fov_shift(native, {0.01, 0.0, 0.0}, pulseq::FovShiftScope::RfAndAdc);

    const double* s = server.adc_library().row(server.get_block(blk).adc);
    const double* n = native.adc_library().row(native.get_block(blk).adc);
    for (int c = 0; c < pulseq::ADC_WIDTH; ++c)
        EXPECT_DOUBLE_EQ(s[c], n[c]) << "column " << c;
    EXPECT_NEAR(s[5], 500.0, 1e-9);
    EXPECT_EQ(s[7], 0.0) << "no phase modulation of any kind";
}

TEST(PulseqFovShift, ServerScopeDefersARotatedReadout)
{
    Sequence seq;
    const int blk = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);
    {
        const int rotations = seq.extension_type_id("ROTATIONS");
        const double q[4] = {1.0, 0.0, 0.0, 0.0};
        Block b = seq.get_block(blk);
        b.ext = seq.chain_extension(rotations, seq.register_rotation(q), b.ext);
        seq.set_block(blk, b);
    }
    const int32_t adc_before = seq.get_block(blk).adc;

    pulseq::apply_fov_shift(seq, {0.01, 0.0, 0.0}, pulseq::FovShiftScope::Server);

    /* Under a rotation the shift phase is orientation-dependent, so the
     * consumer of the base trajectory applies it -- nothing is baked here. */
    EXPECT_EQ(seq.get_block(blk).adc, adc_before);
    EXPECT_DOUBLE_EQ(seq.adc_library().row(adc_before)[5], 0.0);
}

TEST(PulseqFovShift, ServerScopeDefersARampSampledReadout)
{
    Sequence seq;
    const auto g = trap(50000.0, 640e-6);
    const auto a = adc(64, 10e-6, 0.0);  // starts on the ramp

    Block b;
    b.gx = seq.register_trap(g.data());
    b.adc = seq.register_adc(a.data());
    b.duration = 840e-6;
    const int blk = seq.add_block(b);
    const int32_t adc_before = seq.get_block(blk).adc;

    pulseq::apply_fov_shift(seq, {0.01, 0.0, 0.0}, pulseq::FovShiftScope::Server);

    EXPECT_EQ(seq.get_block(blk).adc, adc_before);
    EXPECT_DOUBLE_EQ(seq.adc_library().row(adc_before)[5], 0.0)
        << "a varying gradient defers with the rotated ones";
}

TEST(PulseqFovShift, ZeroShiftIsANoOp)
{
    Sequence seq;
    const int blk = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);
    const int32_t adc_before = seq.get_block(blk).adc;
    const int shapes_before = seq.shape_library().size();

    pulseq::apply_fov_shift(seq, {0.0, 0.0, 0.0}, pulseq::FovShiftScope::RfAndAdc);

    EXPECT_EQ(seq.get_block(blk).adc, adc_before);
    EXPECT_EQ(seq.shape_library().size(), shapes_before);
}

TEST(PulseqFovShift, ShiftAlongAnAxisWithNoGradientDoesNothing)
{
    Sequence seq;
    const int blk = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);  // gx only

    pulseq::apply_fov_shift(seq, {0.0, 0.02, 0.0}, pulseq::FovShiftScope::RfAndAdc);

    const double* row = seq.adc_library().row(seq.get_block(blk).adc);
    EXPECT_DOUBLE_EQ(row[5], 0.0);
    EXPECT_DOUBLE_EQ(row[6], 0.0);
}

/* ================================================================== */
/*  The block range                                                    */
/* ================================================================== */

TEST(PulseqFovShift, ABlockRangeLeavesTheBlocksOutsideItAlone)
{
    Sequence seq;
    const int first = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);
    const int second = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);
    const int third = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);
    const int32_t before = seq.get_block(third).adc;

    pulseq::apply_fov_shift(seq, {0.01, 0.0, 0.0}, pulseq::FovShiftScope::RfAndAdc, second,
                            second);

    /* The one asked for moved; neither of its neighbours did, and they still
     * point at the row they always did. */
    EXPECT_NE(seq.adc_library().row(seq.get_block(second).adc)[5], 0.0);
    EXPECT_EQ(seq.get_block(third).adc, before);
    EXPECT_DOUBLE_EQ(seq.adc_library().row(seq.get_block(first).adc)[5], 0.0);
    EXPECT_DOUBLE_EQ(seq.adc_library().row(seq.get_block(third).adc)[5], 0.0);
}

TEST(PulseqFovShift, ARangedShiftGivesTheSamePhaseAsShiftingEverything)
{
    /* The point of accumulating k from block 1 regardless of the range: a
     * readout's phase must not depend on where the caller drew the boundary. */
    const auto phase_of = [](int first, int last) {
        Sequence seq;
        add_readout(seq, 50000.0, 640e-6, 10e-6, 64);
        const int probe = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);
        pulseq::apply_fov_shift(seq, {0.01, 0.0, 0.0}, pulseq::FovShiftScope::RfAndAdc, first,
                                last);
        return seq.adc_library().row(seq.get_block(probe).adc)[6];
    };

    EXPECT_DOUBLE_EQ(phase_of(2, 2), phase_of(1, 0));
}

TEST(PulseqFovShift, LastOfZeroMeansToTheEnd)
{
    Sequence seq;
    add_readout(seq, 50000.0, 640e-6, 10e-6, 64);
    const int last_block = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);

    pulseq::apply_fov_shift(seq, {0.01, 0.0, 0.0}, pulseq::FovShiftScope::RfAndAdc, 2, 0);

    EXPECT_NE(seq.adc_library().row(seq.get_block(last_block).adc)[5], 0.0);
}

/* ================================================================== */
/*  FOV scale                                                          */
/* ================================================================== */

TEST(PulseqFovScale, ScalesTheAmplitudeAndRegistersNoShape)
{
    Sequence seq;
    const int blk = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);
    const int shapes_before = seq.shape_library().size();

    pulseq::apply_fov_scale(seq, {0.5, 1.0, 1.0});

    const double* row = seq.trap_library().row(seq.grad_row(seq.get_block(blk).gx));
    EXPECT_DOUBLE_EQ(row[0], 25000.0);
    /* The timings are untouched -- a scaled gradient is the same shape. */
    EXPECT_DOUBLE_EQ(row[1], 100e-6);
    EXPECT_DOUBLE_EQ(row[2], 640e-6);
    EXPECT_DOUBLE_EQ(row[3], 100e-6);
    EXPECT_EQ(seq.shape_library().size(), shapes_before);
}

TEST(PulseqFovScale, ScalesAnArbitraryGradientWithoutTouchingItsWaveform)
{
    Sequence seq;
    const std::vector<double> wave{0.0, 0.5, 1.0, 0.5, 0.0};
    const int shape = seq.register_shape(static_cast<int>(wave.size()), wave.data(),
                                         static_cast<int>(wave.size()));
    /* amplitude, first, last, amp_shape, time_shape, delay */
    const std::array<double, pulseq::ARB_WIDTH> arb{10000.0, 100.0, 200.0,
                                                    static_cast<double>(shape), 0.0, 0.0};
    Block b;
    b.gx = seq.register_arbitrary(arb.data());
    b.duration = 50e-6;
    const int blk = seq.add_block(b);

    pulseq::apply_fov_scale(seq, {2.0, 1.0, 1.0});

    const double* row = seq.arb_library().row(seq.grad_row(seq.get_block(blk).gx));
    EXPECT_DOUBLE_EQ(row[0], 20000.0);
    /* `first` and `last` are absolute, so they scale with the amplitude. */
    EXPECT_DOUBLE_EQ(row[1], 200.0);
    EXPECT_DOUBLE_EQ(row[2], 400.0);
    /* The waveform is shared, not copied. */
    EXPECT_EQ(static_cast<int>(row[3]), shape);
}

TEST(PulseqFovScale, ZeroFlattensAnAxisAndOtherAxesAreUntouched)
{
    Sequence seq;
    const auto g = trap(50000.0, 640e-6);
    Block b;
    b.gx = seq.register_trap(g.data());
    b.gy = seq.register_trap(g.data());
    b.duration = 840e-6;
    const int blk = seq.add_block(b);

    pulseq::apply_fov_scale(seq, {0.0, 1.0, 1.0});

    const Block after = seq.get_block(blk);
    EXPECT_DOUBLE_EQ(seq.trap_library().row(seq.grad_row(after.gx))[0], 0.0);
    EXPECT_DOUBLE_EQ(seq.trap_library().row(seq.grad_row(after.gy))[0], 50000.0);
}

TEST(PulseqFovScale, HonoursTheBlockRangeAndUnitScaleIsANoOp)
{
    Sequence seq;
    const int first = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);
    const int second = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);
    const int32_t untouched = seq.get_block(first).gx;

    pulseq::apply_fov_scale(seq, {0.5, 1.0, 1.0}, second, second);

    EXPECT_EQ(seq.get_block(first).gx, untouched);
    EXPECT_DOUBLE_EQ(seq.trap_library().row(seq.grad_row(seq.get_block(first).gx))[0], 50000.0);
    EXPECT_DOUBLE_EQ(seq.trap_library().row(seq.grad_row(seq.get_block(second).gx))[0], 25000.0);

    const int32_t before = seq.get_block(second).gx;
    pulseq::apply_fov_scale(seq, {1.0, 1.0, 1.0});
    EXPECT_EQ(seq.get_block(second).gx, before);
}

/* ================================================================== */
/*  Copying                                                            */
/* ================================================================== */

TEST(PulseqSequenceCopy, TheCopyAndTheOriginalDivergeIndependently)
{
    Sequence original;
    add_readout(original, 50000.0, 640e-6, 10e-6, 64);
    original.set_definition("Name", pulseq::Definition(std::string("original")));

    Sequence copy = original;

    EXPECT_EQ(copy.num_blocks(), 1);
    EXPECT_EQ(copy.definition("Name")->text(), "original");

    /* Growing one leaves the other where it was ... */
    add_readout(copy, 50000.0, 640e-6, 10e-6, 64);
    EXPECT_EQ(copy.num_blocks(), 2);
    EXPECT_EQ(original.num_blocks(), 1);

    add_readout(original, 50000.0, 640e-6, 10e-6, 64);
    add_readout(original, 50000.0, 640e-6, 10e-6, 64);
    EXPECT_EQ(original.num_blocks(), 3);
    EXPECT_EQ(copy.num_blocks(), 2);

    /* ... and so does transforming it. */
    pulseq::apply_fov_scale(copy, {0.5, 1.0, 1.0});
    EXPECT_DOUBLE_EQ(copy.trap_library().row(seq_first_trap(copy))[0], 25000.0);
    EXPECT_DOUBLE_EQ(original.trap_library().row(seq_first_trap(original))[0], 50000.0);
}

/* ================================================================== */
/*  Carrying the base trajectory in a .seq                             */
/* ================================================================== */

namespace
{
    /* Attach a ROTATIONS extension, making the block one whose readout the
     * consumer has to finish -- the kind that stores a base trajectory. */
    void rotate_block(Sequence& seq, int blk)
    {
        const int rotations = seq.extension_type_id("ROTATIONS");
        const double q[4] = {1.0, 0.0, 0.0, 0.0};
        const int rot = seq.register_rotation(q);
        Block b = seq.get_block(blk);
        b.ext = seq.chain_extension(rotations, rot, b.ext);
        seq.set_block(blk, b);
    }
}  // namespace

TEST(PulseqBaseTrajectoryCarrier, MarkerSaysTheFileCarriesThem)
{
    Sequence seq;
    rotate_block(seq, add_readout(seq, 50000.0, 640e-6, 10e-6, 64));

    EXPECT_FALSE(pulseq::has_base_trajectory(seq));
    pulseq::attach_base_trajectory(seq);
    EXPECT_TRUE(pulseq::has_base_trajectory(seq));

    /* A native file must never be mistaken for a server-mode one. */
    Sequence native;
    add_readout(native, 50000.0, 640e-6, 10e-6, 64);
    EXPECT_FALSE(pulseq::has_base_trajectory(native));
}

TEST(PulseqBaseTrajectoryCarrier, CartesianUnrotatedReadoutStoresNothing)
{
    /* Its shift is two scalars on the ADC row and its k comes from the
     * encoding counters, so storing a trajectory would be pure weight --
     * and with nothing stored, the marker must stay off the file. */
    Sequence seq;
    const int blk = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);
    const int shapes_before = seq.shape_library().size();

    pulseq::attach_base_trajectory(seq);

    EXPECT_EQ(static_cast<int>(seq.adc_library().row(seq.get_block(blk).adc)[7]), 0);
    EXPECT_EQ(seq.shape_library().size(), shapes_before);
    EXPECT_FALSE(pulseq::has_base_trajectory(seq));
}

TEST(PulseqBaseTrajectoryCarrier, OnlyTheDeferringReadoutStores)
{
    Sequence seq;
    const int cartesian = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);
    const int rotated = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);
    rotate_block(seq, rotated);

    pulseq::attach_base_trajectory(seq);

    EXPECT_EQ(static_cast<int>(seq.adc_library().row(seq.get_block(cartesian).adc)[7]), 0);
    EXPECT_GT(static_cast<int>(seq.adc_library().row(seq.get_block(rotated).adc)[7]), 0);
    EXPECT_TRUE(pulseq::has_base_trajectory(seq));
}

TEST(PulseqBaseTrajectoryCarrier, RoundTripsThroughTheAdcRow)
{
    Sequence seq;
    seq.set_rasters(1e-6, 10e-6, 100e-9, 10e-6);

    /* Two axes active, one absent, so the packing of all three is exercised. */
    const auto gx = trap(50000.0, 640e-6);
    const auto gy = trap(-12000.0, 640e-6);
    const auto a = adc(64, 10e-6, 100e-6);

    Block b;
    b.gx = seq.register_trap(gx.data());
    b.gy = seq.register_trap(gy.data());
    b.adc = seq.register_adc(a.data());
    b.duration = 840e-6;
    const int blk = seq.add_block(b);
    rotate_block(seq, blk);

    const BaseTrajectory before = pulseq::base_trajectory(seq, blk);
    pulseq::attach_base_trajectory(seq);
    const BaseTrajectory after = pulseq::read_base_trajectory(seq, blk);

    ASSERT_TRUE(after.has_adc);
    EXPECT_EQ(after.num_samples, before.num_samples);
    EXPECT_DOUBLE_EQ(after.dwell, before.dwell);

    for (int axis = 0; axis < 3; ++axis)
    {
        EXPECT_EQ(after.axis[axis].present, before.axis[axis].present) << "axis " << axis;
        EXPECT_DOUBLE_EQ(after.axis[axis].amplitude, before.axis[axis].amplitude)
            << "axis " << axis;
        ASSERT_EQ(after.axis[axis].base.size(), before.axis[axis].base.size())
            << "axis " << axis;
        for (size_t i = 0; i < before.axis[axis].base.size(); ++i)
            EXPECT_NEAR(after.axis[axis].base[i], before.axis[axis].base[i], 1e-15)
                << "axis " << axis << " sample " << i;
    }
}

TEST(PulseqBaseTrajectoryCarrier, StoredSamplesRescaleToRealKSpace)
{
    Sequence seq;
    const int blk = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);
    rotate_block(seq, blk);

    const BaseTrajectory truth = pulseq::base_trajectory(seq, blk);
    pulseq::attach_base_trajectory(seq);

    /* The consumer's formula, spelled out from the ADC row alone:
     *     k = amplitude * num_samples * dwell * stored                */
    const Block block = seq.get_block(blk);
    const double* row = seq.adc_library().row(block.adc);
    const int num_samples = static_cast<int>(row[0]);
    const double dwell = row[1];
    const int shape_id = static_cast<int>(row[7]);
    ASSERT_GT(shape_id, 0);

    const auto& shapes = seq.shape_library();
    ASSERT_EQ(shapes.num_uncompressed(shape_id), 3 * num_samples);
    const double* stored = shapes.samples(shape_id);

    const double amplitude = truth.axis[0].amplitude;
    for (int i = 0; i < num_samples; ++i)
    {
        const double k = amplitude * num_samples * dwell * stored[i];
        EXPECT_NEAR(k, amplitude * truth.axis[0].base[static_cast<size_t>(i)], 1e-12)
            << "sample " << i;
    }
}

TEST(PulseqBaseTrajectoryCarrier, AxesAreAxisMajorAndAbsentOnesAreFlatZero)
{
    Sequence seq;
    const int blk = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);
    rotate_block(seq, blk);
    pulseq::attach_base_trajectory(seq);

    const Block block = seq.get_block(blk);
    const int shape_id = static_cast<int>(seq.adc_library().row(block.adc)[7]);
    const double* stored = seq.shape_library().samples(shape_id);

    /* x runs first and varies; y and z follow, flat at zero. */
    EXPECT_NE(stored[0], stored[63]);
    for (int i = 64; i < 192; ++i)
        EXPECT_DOUBLE_EQ(stored[i], 0.0) << "sample " << i;
}

TEST(PulseqBaseTrajectoryCarrier, ReadBackDerivesConstantFromTheSamples)
{
    Sequence seq;
    /* ADC on the flat top: k still climbs, so the axis is NOT constant... */
    const int flat_top = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);
    rotate_block(seq, flat_top);
    pulseq::attach_base_trajectory(seq);

    const BaseTrajectory t = pulseq::read_base_trajectory(seq, flat_top);
    EXPECT_FALSE(t.axis[0].constant) << "k sweeps under a flat gradient";
    /* ...while the axes with no gradient at all are flat, which is the signal
     * a consumer takes its DC from the encoding counters on. */
    EXPECT_TRUE(t.axis[1].constant);
    EXPECT_TRUE(t.axis[2].constant);
}

TEST(PulseqBaseTrajectoryCarrier, ReadbackIsEmptyWithoutAStoredShape)
{
    Sequence seq;
    const int blk = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);

    /* Never attached: nothing to read. */
    EXPECT_FALSE(pulseq::read_base_trajectory(seq, blk).has_adc);
}

TEST(PulseqBaseTrajectoryCarrier, ReadoutsDifferingOnlyInAmplitudeShareOneShape)
{
    /* The property the whole design rests on: a phase-encode table of N lines
     * costs one base trajectory, not N. */
    Sequence seq;
    seq.set_rasters(1e-6, 10e-6, 100e-9, 10e-6);

    constexpr int lines = 64;
    for (int line = 0; line < lines; ++line)
    {
        const auto gx = trap(50000.0, 640e-6);
        /* A phase encode that differs per line, and a readout that does not. */
        const auto gy = trap(-30000.0 + 1000.0 * line, 200e-6);
        const auto a = adc(64, 10e-6, 100e-6);

        Block b;
        b.gx = seq.register_trap(gx.data());
        b.gy = seq.register_trap(gy.data());
        b.adc = seq.register_adc(a.data());
        b.duration = 840e-6;
        rotate_block(seq, seq.add_block(b));
    }

    pulseq::attach_base_trajectory(seq);
    seq.remove_duplicates();

    /* Every line traverses the same k-space shape, so one shape survives for
     * all of them -- plus whatever the gradients themselves needed (none here:
     * trapezoids carry no shapes). */
    EXPECT_EQ(seq.shape_library().size(), 1)
        << lines << " readouts collapsed to " << seq.shape_library().size() << " shapes";
    EXPECT_EQ(seq.adc_library().size(), 1);
}

TEST(PulseqBaseTrajectoryCarrier, DistinctTrajectoriesKeepDistinctShapes)
{
    /* The converse: genuinely different readouts must not be collapsed. */
    Sequence seq;
    seq.set_rasters(1e-6, 10e-6, 100e-9, 10e-6);

    /* Same amplitude, different flat times -- so different k sweeps. */
    rotate_block(seq, add_readout(seq, 50000.0, 640e-6, 10e-6, 64));
    rotate_block(seq, add_readout(seq, 50000.0, 640e-6, 20e-6, 64));

    pulseq::attach_base_trajectory(seq);
    seq.remove_duplicates();

    EXPECT_EQ(seq.shape_library().size(), 2);
}

TEST(PulseqBaseTrajectoryCarrier, SharedAdcRowIsSplitWhenItsUsersDisagree)
{
    /* One ADC definition reused by two blocks that traverse different
     * k-space.  The row cannot describe both, so it has to be split. */
    Sequence seq;
    seq.set_rasters(1e-6, 10e-6, 100e-9, 10e-6);

    const auto a = adc(64, 10e-6, 100e-6);
    const int32_t shared = seq.register_adc(a.data());

    const auto slow = trap(10000.0, 640e-6);
    const auto fast = trap(10000.0, 640e-6);
    (void)fast;

    Block one;
    one.gx = seq.register_trap(slow.data());
    one.adc = shared;
    one.duration = 840e-6;
    const int blk_one = seq.add_block(one);

    Block two;
    /* Same amplitude, different ramp -- so a different k sweep under the
     * same ADC. */
    const std::array<double, pulseq::TRAP_WIDTH> steep{10000.0, 20e-6, 800e-6, 20e-6, 0.0};
    two.gx = seq.register_trap(steep.data());
    two.adc = shared;
    two.duration = 840e-6;
    const int blk_two = seq.add_block(two);
    rotate_block(seq, blk_one);
    rotate_block(seq, blk_two);

    pulseq::attach_base_trajectory(seq);

    const int32_t adc_one = seq.get_block(blk_one).adc;
    const int32_t adc_two = seq.get_block(blk_two).adc;
    EXPECT_NE(adc_one, adc_two) << "the shared row had to be split";

    /* And each block still reads back its own trajectory. */
    const BaseTrajectory truth_one = pulseq::base_trajectory(seq, blk_one);
    const BaseTrajectory back_one = pulseq::read_base_trajectory(seq, blk_one);
    const BaseTrajectory truth_two = pulseq::base_trajectory(seq, blk_two);
    const BaseTrajectory back_two = pulseq::read_base_trajectory(seq, blk_two);

    ASSERT_TRUE(back_one.has_adc);
    ASSERT_TRUE(back_two.has_adc);
    for (int i = 0; i < truth_one.num_samples; ++i)
    {
        EXPECT_NEAR(back_one.axis[0].base[static_cast<size_t>(i)],
                    truth_one.axis[0].base[static_cast<size_t>(i)], 1e-15);
        EXPECT_NEAR(back_two.axis[0].base[static_cast<size_t>(i)],
                    truth_two.axis[0].base[static_cast<size_t>(i)], 1e-15);
    }
    EXPECT_NE(back_one.axis[0].base.front(), back_two.axis[0].base.front());
}

TEST(PulseqBaseTrajectoryCarrier, SurvivesAWriteAndReadOfTheFile)
{
    /* The point of putting it in the .seq at all: it has to come back out. */
    Sequence seq;
    seq.set_rasters(1e-6, 10e-6, 100e-9, 10e-6);

    const auto gx = trap(50000.0, 640e-6);
    const auto gy = trap(-12000.0, 640e-6);
    const auto a = adc(64, 10e-6, 100e-6);

    Block b;
    b.gx = seq.register_trap(gx.data());
    b.gy = seq.register_trap(gy.data());
    b.adc = seq.register_adc(a.data());
    b.duration = 840e-6;
    const int blk = seq.add_block(b);
    rotate_block(seq, blk);

    pulseq::attach_base_trajectory(seq);
    const BaseTrajectory truth = pulseq::base_trajectory(seq, blk);

    const std::string text = pulseq::write_text(seq, false);

    const std::filesystem::path scratch =
        std::filesystem::temp_directory_path() / "pulseqpp_base_traj.seq";
    {
        std::ofstream out(scratch, std::ios::binary);
        out << text;
    }
    Sequence reloaded = pulseq::read_file(scratch.string());
    std::filesystem::remove(scratch);

    EXPECT_TRUE(pulseq::has_base_trajectory(reloaded))
        << "the marker has to survive, or the file is silently ambiguous";

    const BaseTrajectory back = pulseq::read_base_trajectory(reloaded, blk);
    ASSERT_TRUE(back.has_adc);
    ASSERT_EQ(back.num_samples, truth.num_samples);

    for (int axis = 0; axis < 2; ++axis)
    {
        ASSERT_EQ(back.axis[axis].base.size(), truth.axis[axis].base.size())
            << "axis " << axis;
        for (size_t i = 0; i < truth.axis[axis].base.size(); ++i)
        {
            /* The text writer prints shapes at finite precision, so agreement
             * is to the file's own resolution, not to the bit. */
            EXPECT_NEAR(back.axis[axis].base[i], truth.axis[axis].base[i], 1e-11)
                << "axis " << axis << " sample " << i;
        }
    }
}

TEST(PulseqBaseTrajectoryCarrier, BlocksWithoutAdcAreLeftAlone)
{
    Sequence seq;
    const auto g = trap(30000.0, 500e-6);
    Block b;
    b.gx = seq.register_trap(g.data());
    b.duration = 700e-6;
    const int blk = seq.add_block(b);

    pulseq::attach_base_trajectory(seq);

    EXPECT_EQ(seq.get_block(blk).adc, 0);
    EXPECT_EQ(seq.shape_library().size(), 0);
    EXPECT_FALSE(pulseq::read_base_trajectory(seq, blk).has_adc);
}

/* ================================================================== */
/*  The labels that exempt a block                                     */
/* ================================================================== */

namespace
{
    /** A block carrying `label = value` as a LABELSET extension. */
    int add_flagged(Sequence& seq, const std::string& label, int32_t value)
    {
        const int blk = add_readout(seq, 50000.0, 640e-6, 10e-6, 64);
        Block b = seq.get_block(blk);
        b.ext = seq.chain_extension(seq.extension_type_id("LABELSET"),
                                    seq.register_label_set(value, seq.label_id(label)), 0);
        seq.set_block(blk, b);
        return blk;
    }

    using Runs = std::vector<std::pair<int, int>>;
}  // namespace

/*
 * The exemption is sticky, and it starts at the block that sets it.
 *
 * Both are MATLAB's behaviour, and both are what the runs have to express:
 * a flag set in block 2 exempts 2 onwards, so the surviving run is 1..1.
 */
TEST(PulseqLabelGates, AFlagExemptsTheBlockThatSetsItAndEveryBlockAfter)
{
    Sequence seq;
    add_readout(seq, 50000.0, 640e-6, 10e-6, 64);
    add_flagged(seq, "NOPOS", 1);
    add_readout(seq, 50000.0, 640e-6, 10e-6, 64);

    const auto runs = pulseq::label_gate_runs(seq, {"NOPOS"});
    ASSERT_EQ(runs.size(), 1u);
    EXPECT_EQ(runs[0], Runs({{1, 1}}));
}

/* Clearing it lets the transformation resume, which is a second run. */
TEST(PulseqLabelGates, ClearingAFlagOpensASecondRun)
{
    Sequence seq;
    add_flagged(seq, "NOPOS", 1);
    add_readout(seq, 50000.0, 640e-6, 10e-6, 64);
    add_flagged(seq, "NOPOS", 0);
    add_readout(seq, 50000.0, 640e-6, 10e-6, 64);

    const auto runs = pulseq::label_gate_runs(seq, {"NOPOS"});
    ASSERT_EQ(runs.size(), 1u);
    EXPECT_EQ(runs[0], Runs({{3, 4}}));
}

/*
 * Each label is read independently, in one pass.
 *
 * The reason the call takes a list at all: the three exemptions gate three
 * different transforms and a scan is walked once for all of them.
 */
TEST(PulseqLabelGates, LabelsAreIndependentAndAnUnknownOneExemptsNothing)
{
    Sequence seq;
    add_readout(seq, 50000.0, 640e-6, 10e-6, 64);
    add_flagged(seq, "NOSCL", 1);

    const auto runs = pulseq::label_gate_runs(seq, {"NOSCL", "NOPOS", "NOROT"});
    ASSERT_EQ(runs.size(), 3u);
    EXPECT_EQ(runs[0], Runs({{1, 1}}));
    /* Never set, so never exempt -- the whole range is one run. */
    EXPECT_EQ(runs[1], Runs({{1, 2}}));
    EXPECT_EQ(runs[2], Runs({{1, 2}}));
}

/*
 * State starts clear at `first`, not at block 1.
 *
 * A windowed transform resets, which is what `applyToSeq` does: the flag set
 * in block 1 does not reach a range that starts at block 2.
 */
TEST(PulseqLabelGates, StateStartsClearAtTheRangeRatherThanAtBlockOne)
{
    Sequence seq;
    add_flagged(seq, "NOPOS", 1);
    add_readout(seq, 50000.0, 640e-6, 10e-6, 64);
    add_readout(seq, 50000.0, 640e-6, 10e-6, 64);

    const auto whole = pulseq::label_gate_runs(seq, {"NOPOS"});
    EXPECT_TRUE(whole[0].empty()) << "the flag is set in block 1 and never cleared";

    const auto windowed = pulseq::label_gate_runs(seq, {"NOPOS"}, 2, 3);
    EXPECT_EQ(windowed[0], Runs({{2, 3}}));
}

/* A sequence that flags nothing is one run, which is the ordinary case. */
TEST(PulseqLabelGates, NoFlagsAtAllIsOneRun)
{
    Sequence seq;
    for (int i = 0; i < 5; ++i)
        add_readout(seq, 50000.0, 640e-6, 10e-6, 64);

    const auto runs = pulseq::label_gate_runs(seq, {"NOPOS"});
    EXPECT_EQ(runs[0], Runs({{1, 5}}));
}
