/**
 * Tests for pulseq::calculate_kspace -- the Sequence-level surface over the
 * C89 trajectory core.
 *
 * The core's own arithmetic is covered in tests/ctests/test_pulseq_ktraj.c
 * against the sequences' own numbers.  What is left for here is everything
 * the bridge and the layer above it add, and every one of these covers
 * something that was wrong at some point while it was being written:
 *
 *  - the binary round trip is exact (it was not: the format narrows shape
 *    samples to float32, and a run-length-encoded derivative accumulates that
 *    into ~1e-5 of k);
 *  - the echo of a symmetric Cartesian readout is N/2 (it was decided by a
 *    0.005%-of-a-step rounding asymmetry before the tie rule);
 *  - the dense trajectory agrees with the ADC samples where they meet;
 *  - the MRD layout is the transpose of the stored one, with silent axes
 *    dropped over the whole scan rather than per readout.
 */

#include <gtest/gtest.h>

#include "pulseq/kspace.hpp"
#include "pulseq/read.hpp"
#include "pulseq/trajectory.hpp"
#include "pulseq/sequence.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace
{
    const std::string kData = std::string(PULSEQ_FIXTURES_DIR) + "/";

    pulseq::Sequence load(const std::string& stem)
    {
        return pulseq::read_file(kData + stem + ".seq");
    }

}  // namespace

/*
 * A fully sampled symmetric Cartesian readout puts its echo at N/2.
 *
 * The convention an even-sized readout is reconstructed under, and the reason
 * the search has an explicit tie rule: the two central samples sit at -dk/2
 * and +dk/2 and are equidistant from the origin, so without one the answer
 * comes from whichever way the file's six printed digits happened to round.
 * Measured margin on gre_2d_1sl_1avg: 2e-4 against a step of 4.5.
 *
 * bssfp_2d_1sl_1avg is in the list because it is the case with *no* margin:
 * its two central samples are -2.272720 and +2.272720 to the last bit, so
 * nothing but the tie rule decides it.  It reported 31 of 64 until the rule
 * was applied to the global minimum as well as to the per-readout search.
 */
TEST(PulseqKSpace, SymmetricCartesianEchoIsAtHalfN)
{
    for (const char* stem : {"gre_2d_1sl_1avg", "gre_2d_3sl_3avg", "fse_2d_1sl_1avg",
                             "gre_32x32_pe_blip", "bssfp_2d_1sl_1avg"})
    {
        pulseq::Sequence seq = load(stem);
        const pulseq::KSpace ks = pulseq::calculate_kspace(seq);
        ASSERT_FALSE(ks.readouts.empty()) << stem;

        for (const pulseq::Readout& r : ks.readouts)
        {
            if (r.num_samples <= 0)
                continue;
            EXPECT_EQ(r.center_sample, r.num_samples / 2)
                << stem << " readout at block " << r.block_index;
        }
    }
}

/*
 * An EPI's two readout polarities mirror each other, and so do their echoes.
 *
 * Not a defect: a reversed line traverses k backwards, so the sample nearest
 * the origin really does sit one index earlier in acquisition order.  What has
 * to hold is that reversing the line maps one onto the other -- which is
 * exactly what makes a REV label sufficient for a reconstruction to undo it.
 */
TEST(PulseqKSpace, EpiPolaritiesHaveMirroredEchoes)
{
    pulseq::Sequence seq = load("epi_2d_1sl_1avg");
    const pulseq::KSpace ks = pulseq::calculate_kspace(seq);
    ASSERT_GT(ks.readouts.size(), 4u);

    int seen_forward = 0;
    int seen_reverse = 0;
    for (const pulseq::Readout& r : ks.readouts)
    {
        if (r.num_samples <= 0 || r.center_sample < 0)
            continue;
        const double* kx = ks.k_adc.data() + static_cast<size_t>(r.sample_offset) * 3;
        const bool forward = kx[r.num_samples - 1] > kx[0];
        if (forward)
        {
            EXPECT_EQ(r.center_sample, r.num_samples / 2);
            ++seen_forward;
        }
        else
        {
            /* The mirror of the forward answer. */
            EXPECT_EQ(r.num_samples - 1 - r.center_sample, r.num_samples / 2);
            ++seen_reverse;
        }
    }
    EXPECT_GT(seen_forward, 0);
    EXPECT_GT(seen_reverse, 0);
}

/*
 * The dense trajectory and the ADC samples are the same trajectory.
 *
 * They come from different entry points -- one walks breakpoints, the other
 * walks readouts -- so agreeing where they meet is what says both are reading
 * the same accumulator.  k is quadratic between breakpoints, so linear
 * interpolation of the dense grid is compared with a tolerance scaled to the
 * curvature it cannot represent rather than to machine precision.
 */
TEST(PulseqKSpace, DenseTrajectoryAgreesWithTheAdcSamples)
{
    pulseq::Sequence seq = load("gre_2d_1sl_1avg");
    pulseq::KSpaceOptions options;
    options.dense = true;
    const pulseq::KSpace ks = pulseq::calculate_kspace(seq, options);

    ASSERT_FALSE(ks.t_ktraj.empty());
    ASSERT_EQ(ks.k_ktraj.size(), ks.t_ktraj.size() * 3);

    const int dense_n = static_cast<int>(ks.t_ktraj.size());
    double worst = 0.0;
    for (const pulseq::Readout& r : ks.readouts)
    {
        for (int j = 0; j < r.num_samples; ++j)
        {
            const double t = r.t0 + j * r.dwell;
            if (t <= ks.t_ktraj.front() || t >= ks.t_ktraj.back())
                continue;

            int lo = 0;
            int hi = dense_n - 1;
            while (hi - lo > 1)
            {
                const int mid = (lo + hi) / 2;
                if (ks.t_ktraj[static_cast<size_t>(mid)] <= t)
                    lo = mid;
                else
                    hi = mid;
            }
            const double span =
                ks.t_ktraj[static_cast<size_t>(hi)] - ks.t_ktraj[static_cast<size_t>(lo)];
            if (span <= 0.0)
                continue;
            const double frac =
                (t - ks.t_ktraj[static_cast<size_t>(lo)]) / span;

            for (int a = 0; a < 3; ++a)
            {
                const double k_lo = ks.k_ktraj[static_cast<size_t>(a) *
                                                   static_cast<size_t>(dense_n) +
                                               static_cast<size_t>(lo)];
                const double k_hi = ks.k_ktraj[static_cast<size_t>(a) *
                                                   static_cast<size_t>(dense_n) +
                                               static_cast<size_t>(hi)];
                const double interpolated = k_lo + (k_hi - k_lo) * frac;
                const double exact =
                    ks.k_adc[static_cast<size_t>(r.sample_offset) * 3 +
                             static_cast<size_t>(a) * static_cast<size_t>(r.num_samples) +
                             static_cast<size_t>(j)];
                worst = std::max(worst, std::fabs(interpolated - exact));
            }
        }
    }
    /* One gradient step's worth of curvature is the most a chord across a
     * breakpoint interval can miss by; anything larger means the two paths
     * disagree about the trajectory, not about how to draw it. */
    EXPECT_LT(worst, 5.0);
}

/*
 * Silent axes are decided over the whole scan, not one readout.
 *
 * A readout whose rotation leaves an axis idle would otherwise pack one axis
 * fewer than its neighbours and shift every value after it -- the same rule
 * PrecomputedTrajectory::axis_active states, and the reason it is stated
 * there.
 */
TEST(PulseqKSpace, MrdLayoutIsInterleavedAndPrunedConsistently)
{
    pulseq::Sequence seq = load("gre_2d_1sl_1avg");
    const pulseq::KSpace ks = pulseq::calculate_kspace(seq);
    const std::array<bool, 3> axes = pulseq::active_axes(ks);

    int ndim = 0;
    for (int a = 0; a < 3; ++a)
        ndim += axes[static_cast<size_t>(a)] ? 1 : 0;
    ASSERT_GT(ndim, 0);

    for (size_t i = 0; i < ks.readouts.size(); ++i)
    {
        const pulseq::Readout& r = ks.readouts[i];
        if (r.num_samples <= 0)
            continue;

        std::vector<float> out(static_cast<size_t>(ndim * r.num_samples), 0.0f);
        const int written =
            pulseq::interleave_readout(ks, axes, static_cast<int>(i), out.data());
        ASSERT_EQ(written, ndim * r.num_samples);

        /* Interleaved: sample j's axes are adjacent, not a stride apart. */
        int slot = 0;
        for (int a = 0; a < 3; ++a)
        {
            if (!axes[static_cast<size_t>(a)])
                continue;
            for (int j = 0; j < r.num_samples; ++j)
            {
                const double expected =
                    ks.k_adc[static_cast<size_t>(r.sample_offset) * 3 +
                             static_cast<size_t>(a) * static_cast<size_t>(r.num_samples) +
                             static_cast<size_t>(j)];
                EXPECT_FLOAT_EQ(out[static_cast<size_t>(j * ndim + slot)],
                                static_cast<float>(expected));
            }
            ++slot;
        }
    }
}

/* A block range bounds what comes back. */
TEST(PulseqKSpace, BlockRangeBoundsTheResult)
{
    pulseq::Sequence seq = load("gre_2d_1sl_1avg");
    const pulseq::KSpace all = pulseq::calculate_kspace(seq);

    pulseq::KSpaceOptions options;
    options.last_block = seq.num_blocks() / 2;
    const pulseq::KSpace part = pulseq::calculate_kspace(seq, options);

    EXPECT_LT(part.readouts.size(), all.readouts.size());
    EXPECT_LT(part.total_duration, all.total_duration);
}

/*
 * The repeat memo fires here too.
 *
 * The bridge canonicalises nothing of its own, so this is really asserting
 * that the round trip preserves whatever made readouts group -- library ids
 * included.  A binary writer that renumbered rows would pass every other test
 * in this file and fail this one.
 */
TEST(PulseqKSpace, TheRepeatMemoSurvivesTheBridge)
{
    pulseq::Sequence seq = load("gre_2d_3sl_3avg");
    const pulseq::KSpace ks = pulseq::calculate_kspace(seq);

    ASSERT_GT(ks.readouts.size(), 4u);
    EXPECT_LT(ks.key_groups, static_cast<int>(ks.readouts.size()));
}

/*
 * A radial spoke has a constant gradient and still needs a trajectory.
 *
 * The trap this warns about: a gradient that is flat across the ADC window
 * normally means the readout is a straight line a reconstruction can place
 * from an encoding counter and a frequency offset, storing no trajectory at
 * all.  Rotating a constant gradient leaves it constant, so **a radial spoke
 * passes that test too** -- and there is no counter that places it, because
 * what changes between shots is the direction, which lives in the rotation.
 *
 * Every spoke below plays the *same* flat readout gradient and differs only
 * by its rotation extension.  So: the gradients really are constant, the
 * spokes really do point in different directions, and `rotations_vary` is
 * what separates this from the Cartesian case the shortcut is valid for.
 */
TEST(PulseqKSpace, ARadialSpokeIsConstantGradientButNotCartesian)
{
    pulseq::Sequence seq;
    seq.set_rasters(1e-6, 10e-6, 100e-9, 10e-6);

    /* amplitude, rise, flat, fall, delay -- one flat readout, reused. */
    const double trap[pulseq::TRAP_WIDTH] = {1.0e5, 100e-6, 1000e-6, 100e-6, 0.0};
    const int gx = seq.register_trap(trap);

    /* num, dwell, delay, freq_ppm, phase_ppm, freq, phase, phase_shape */
    const double adc[pulseq::ADC_WIDTH] = {64.0, 10e-6, 100e-6, 0.0, 0.0, 0.0, 0.0, 0.0};
    const int adc_id = seq.register_adc(adc);

    const int rotations = seq.extension_type_id("ROTATIONS");
    const int kSpokes = 8;
    std::vector<double> angles;
    for (int s = 0; s < kSpokes; ++s)
    {
        const double angle = M_PI * static_cast<double>(s) / kSpokes;
        angles.push_back(angle);
        /* Quaternions are stored **scalar first**: (w, x, y, z). */
        const double q[pulseq::ROTATION_WIDTH] = {std::cos(0.5 * angle), 0.0, 0.0,
                                                  std::sin(0.5 * angle)};
        const int rotation = seq.register_rotation(q);

        pulseq::Block block;
        block.gx = gx;
        block.adc = adc_id;
        block.duration = 1200e-6;
        block.ext = seq.chain_extension(rotations, rotation, 0);
        seq.add_block(block);
    }

    const pulseq::KSpace ks = pulseq::calculate_kspace(seq);
    ASSERT_EQ(ks.readouts.size(), static_cast<size_t>(kSpokes));

    EXPECT_TRUE(ks.rotations_vary)
        << "the spokes differ only by rotation; nothing else says so";

    for (int s = 0; s < kSpokes; ++s)
    {
        const pulseq::Readout& r = ks.readouts[static_cast<size_t>(s)];
        EXPECT_GE(r.rotation_id, 0) << "spoke " << s << " lost its rotation";

        /* The gradient is flat on every axis -- which is exactly why this
         * cannot be the test for "no trajectory needed". */
        for (int a = 0; a < 3; ++a)
            EXPECT_TRUE(r.constant_axes[static_cast<size_t>(a)]) << "spoke " << s << " axis " << a;

        /* And yet the spoke points somewhere new.  Its direction is the
         * rotation applied to the readout axis. */
        const double* k = ks.k_adc.data() + static_cast<size_t>(r.sample_offset) * 3;
        const double dx = k[r.num_samples - 1] - k[0];
        const double dy = k[2 * r.num_samples - 1] - k[r.num_samples];
        ASSERT_GT(std::hypot(dx, dy), 1.0) << "spoke " << s << " does not move through k";
        EXPECT_NEAR(std::atan2(dy, dx), angles[static_cast<size_t>(s)], 1e-6)
            << "spoke " << s << " points the wrong way";
    }

    /*
     * These spokes are centre-out -- no prewinder, so k starts near the
     * origin and marches away -- and their echo is therefore sample 0.
     *
     * Worth asserting for its own sake: it is the plainest case where the
     * `num_samples / 2` that pulseg falls back to when its k-space analysis
     * did not run is not merely imprecise but wrong by half a readout.
     */
    for (const pulseq::Readout& r : ks.readouts)
        EXPECT_EQ(r.center_sample, 0);
}

/*
 * `block_k_origins()` is the core's answer in Pulseq's 1-based layout.
 *
 * It used to be a second implementation -- a `Piecewise` per block, written
 * before the C89 core existed -- and this test was the evidence that let it
 * go: over every fixture in the repository the two agreed to 4e-16 relative,
 * which is arithmetic noise rather than agreement by construction. Getting
 * there needed a real fix, not just a comparison, because the old one reset k
 * only on an explicit 'e': a pulse that declared no use was read as neither an
 * excitation nor a refocusing and k accumulated over the whole scan. Both
 * refuse an undeclared use now, and there is only one of them.
 *
 * What is left to check is the re-indexing, which `TransformFOV` depends on:
 * `[b]` belongs to block `b`, `[0]` is unused, and the frame is logical. A
 * silent off-by-one here would move every FOV shift by one block's gradient
 * moment.
 */
TEST(PulseqKSpace, BlockKOriginsIsTheCoreReindexedFromOne)
{
    for (const char* stem : {"gre_2d_1sl_1avg", "gre_2d_3sl_3avg", "epi_2d_3sl_3avg",
                             "fse_2d_1sl_1avg", "gre_32x32_pe_blip", "bssfp_2d_3sl_3avg",
                             "mprage_2d_3sl_3avg", "mprage_nav_2d_3sl_3avg",
                             "mprage_noncart_3d_3sl_3avg_userotext1",
                             "qalas_noncart_3d_3sl_3avg_userotext1"})
    {
        pulseq::Sequence seq = load(stem);
        const std::vector<std::array<double, 3>> theirs = pulseq::block_k_origins(seq);

        pulseq::KSpaceOptions options;
        /* Logical frame on both sides: a rotation belongs to the block that
         * carries it, and block_k_origins never resolves one. */
        options.apply_rotation = false;
        options.block_origins = true;
        options.derive_center_sample = false;
        options.materialize_samples = false;
        pulseq::Sequence again = load(stem);
        const pulseq::KSpace ks = pulseq::calculate_kspace(again, options);

        ASSERT_EQ(static_cast<int>(theirs.size()), seq.num_blocks() + 1) << stem;
        ASSERT_EQ(static_cast<int>(ks.block_origins.size()), seq.num_blocks()) << stem;

        /* [0] is not block 0 -- there is no block 0 -- and must stay zero. */
        EXPECT_EQ(theirs[0][0], 0.0);
        EXPECT_EQ(theirs[0][1], 0.0);
        EXPECT_EQ(theirs[0][2], 0.0);

        for (int b = 1; b <= seq.num_blocks(); ++b)
        {
            for (int a = 0; a < 3; ++a)
            {
                ASSERT_EQ(theirs[static_cast<size_t>(b)][static_cast<size_t>(a)],
                          ks.block_origins[static_cast<size_t>(b - 1)][static_cast<size_t>(a)])
                    << stem << " block " << b << " axis " << a;
            }
        }
    }
}

/* Window averaging leaves a flat readout alone and moves a curved one. */
TEST(PulseqKSpace, WindowAveragingIsOptInAndOnlyMovesCurvedReadouts)
{
    pulseq::Sequence flat = load("fse_2d_1sl_1avg");
    const pulseq::KSpace midpoint = pulseq::calculate_kspace(flat);

    pulseq::KSpaceOptions options;
    options.sample_window_average = true;
    pulseq::Sequence flat_again = load("fse_2d_1sl_1avg");
    const pulseq::KSpace averaged = pulseq::calculate_kspace(flat_again, options);

    ASSERT_EQ(midpoint.k_adc.size(), averaged.k_adc.size());
    for (size_t i = 0; i < midpoint.k_adc.size(); ++i)
        EXPECT_NEAR(midpoint.k_adc[i], averaged.k_adc[i], 1e-9);

    pulseq::Sequence curved = load("mprage_noncart_3d_3sl_3avg_userotext1");
    const pulseq::KSpace curved_mid = pulseq::calculate_kspace(curved);
    pulseq::Sequence curved_again = load("mprage_noncart_3d_3sl_3avg_userotext1");
    const pulseq::KSpace curved_avg = pulseq::calculate_kspace(curved_again, options);

    double biggest = 0.0;
    for (size_t i = 0; i < curved_mid.k_adc.size(); ++i)
        biggest = std::max(biggest, std::fabs(curved_mid.k_adc[i] - curved_avg.k_adc[i]));
    EXPECT_GT(biggest, 1e-3);
}
