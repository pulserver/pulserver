/**
 * The trajectory core, against properties the sequences state for themselves.
 *
 * The oracle question was the interesting one here.  PyPulseq is the obvious
 * reference and is used for exactly that in tests/python, but it cannot read a
 * ROTATIONS extension at all, and on an arbitrary gradient carrying an explicit
 * time shape its own gradient spline does not reproduce its own waveform (see
 * the header comment on FlatArbitraryGradientIsExact).  So the assertions below
 * are against the sequence's own numbers instead: a gradient the file declares
 * flat integrates to a straight line with a known slope, a refocusing pulse
 * negates k, a rotation composes with the file's own quaternion, and the echo
 * the fast path reports is the one an exhaustive search finds.  None of that
 * needs a second implementation to check against.
 *
 * Two of these exist because of bugs they caught: CentersAreMinimal covers a
 * shortcut in the echo search that was once wrong at a near-tie, and
 * TheRepeatMemoFires exists because every wrong version of the memo key still
 * produced right answers -- it just never fired, and nothing else would have
 * noticed.
 *
 * What the *bridge* above the core adds is covered in test_pulseq_kspace.cpp.
 */

#include <gtest/gtest.h>

#include "pulseq/kspace.hpp"
#include "pulseq/read.hpp"
#include "pulseq/sequence.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace
{
    const std::string kData = std::string(PULSEQ_FIXTURES_DIR) + "/";
    const std::string kCorpus = std::string(PULSEQ_CORPUS_DIR) + "/";

    pulseq::Sequence load(const std::string& stem)
    {
        return pulseq::read_file(kData + stem + ".seq");
    }

    pulseq::Sequence load_corpus(const std::string& stem)
    {
        return pulseq::read_file(kCorpus + stem + ".seq");
    }

    /**
     * What two k samples can differ by from rounding alone.
     *
     * k is a running integral over a whole scan reaching a few hundred 1/m, so
     * a handful of double epsilons is what the accumulation itself costs.  The
     * floor keeps the bound meaningful where k passes through zero.
     */
    double tol(double magnitude)
    {
        return std::max(4.0 * 2.3e-16 * magnitude, 1e-12);
    }

    /** Axis @p a, sample @p j of readout @p r. */
    double sample(const pulseq::KSpace& ks, const pulseq::Readout& r, int a, int j)
    {
        return ks.k_adc[static_cast<size_t>(r.sample_offset) * 3 +
                        static_cast<size_t>(a) * static_cast<size_t>(r.num_samples) +
                        static_cast<size_t>(j)];
    }

}  // namespace

/*
 * A gradient the file declares constant integrates to a straight line.
 *
 * fse_2d's readout is a flat 227273 Hz/m carried as a two-sample shape beside
 * a two-sample time shape, so consecutive samples must differ by exactly
 * amplitude * dwell, with no drift from the first pair to the last.
 *
 * This is the case where PyPulseq's own answer is wrong: its gradient spline
 * ramps across the readout, so its k step grows and lands 2.1e-6 low.
 * Asserting against PyPulseq here would have meant encoding its error as the
 * expectation; asserting against the file's own numbers does not.
 */
TEST(PulseqKTraj, FlatArbitraryGradientIsExact)
{
    const pulseq::Sequence seq = load_corpus("fse_2d");
    const pulseq::KSpace ks = pulseq::calculate_kspace(seq);
    ASSERT_FALSE(ks.readouts.empty());

    const pulseq::Readout& r = ks.readouts[0];
    ASSERT_EQ(r.num_samples, 32);

    const double expected_step = 227273.0 * r.dwell;
    for (int j = 1; j < r.num_samples; ++j)
    {
        const double a = sample(ks, r, 0, j);
        const double b = sample(ks, r, 0, j - 1);
        EXPECT_NEAR(a - b, expected_step, tol(std::max(std::fabs(a), std::fabs(b))))
            << "readout k step drifts from amplitude * dwell at sample " << j;
    }
}

/*
 * A refocusing pulse negates k.
 *
 * fse_2d is a spin-echo train, so between one readout and the next there is a
 * refocusing pulse.  What is asserted is the invariant that makes an echo
 * train walk back through the origin at all: every readout starts on the
 * negative side of k and ends on the positive side, which cannot happen if the
 * sign flip is dropped.
 */
TEST(PulseqKTraj, RefocusingNegatesK)
{
    const pulseq::Sequence seq = load_corpus("fse_2d");
    const pulseq::KSpace ks = pulseq::calculate_kspace(seq);
    ASSERT_FALSE(ks.refocusings.empty()) << "fixture has no refocusing pulses";

    for (const pulseq::Readout& r : ks.readouts)
    {
        EXPECT_LT(sample(ks, r, 0, 0), 0.0) << "readout at block " << r.block_index;
        EXPECT_GT(sample(ks, r, 0, r.num_samples - 1), 0.0)
            << "readout at block " << r.block_index;
    }
}

namespace
{
    /*
     * k_center is the scan's closest approach to the origin, and every
     * center_sample is its readout's closest sample to it -- checked against
     * the slow, obvious, unmemoized search.
     *
     * Both answers are minima *up to a tie*, and the core resolves ties by
     * convention rather than by whichever side rounding fell on.  So the
     * assertions are "no worse than the true minimum by more than one tie",
     * which is the contract; asserting the bare minimum would be asserting the
     * rounding.  Where it bites, measured: epi_2d_main, whose two readout
     * polarities approach the origin equally closely.
     */
    void check_centers_are_minimal(const pulseq::Sequence& seq, const std::string& name)
    {
        const pulseq::KSpace ks = pulseq::calculate_kspace(seq);
        ASSERT_FALSE(ks.readouts.empty()) << name;

        /* The tie tolerance the core works to: 1% of a sampling step. */
        double tie = 0.0;
        for (const pulseq::Readout& r : ks.readouts)
        {
            if (r.num_samples < 2)
                continue;
            double total = 0.0;
            for (int j = 1; j < r.num_samples; ++j)
            {
                double d = 0.0;
                for (int a = 0; a < 3; ++a)
                {
                    const double v = sample(ks, r, a, j) - sample(ks, r, a, j - 1);
                    d += v * v;
                }
                total += std::sqrt(d);
            }
            tie = std::max(tie, 1.0e-2 * total / static_cast<double>(r.num_samples - 1));
        }

        double global_best = 0.0;
        bool have_global = false;
        for (const pulseq::Readout& r : ks.readouts)
        {
            for (int j = 0; j < r.num_samples; ++j)
            {
                double d = 0.0;
                for (int a = 0; a < 3; ++a)
                    d += sample(ks, r, a, j) * sample(ks, r, a, j);
                if (!have_global || d < global_best)
                {
                    global_best = d;
                    have_global = true;
                }
            }
        }

        {
            double d = 0.0;
            for (int a = 0; a < 3; ++a)
                d += ks.k_center[static_cast<size_t>(a)] * ks.k_center[static_cast<size_t>(a)];
            /* Compared as distances, not squared ones: the tolerance bounds a
             * stored k, and squaring it would not mean anything. */
            EXPECT_LE(std::sqrt(d), std::sqrt(global_best) + tie + tol(std::sqrt(global_best) + 1.0))
                << name << ": k_center is not the scan's closest approach to the origin";
        }

        for (const pulseq::Readout& r : ks.readouts)
        {
            ASSERT_GE(r.center_sample, 0) << name << " block " << r.block_index;
            ASSERT_LT(r.center_sample, r.num_samples) << name << " block " << r.block_index;

            double best = 0.0;
            double chosen = 0.0;
            for (int j = 0; j < r.num_samples; ++j)
            {
                double d = 0.0;
                for (int a = 0; a < 3; ++a)
                {
                    const double v = sample(ks, r, a, j) - ks.k_center[static_cast<size_t>(a)];
                    d += v * v;
                }
                if (j == 0 || d < best)
                    best = d;
                if (j == r.center_sample)
                    chosen = d;
            }
            /* Compared as distances, not as indices: two samples equidistant
             * from the centre are both right answers, and which one wins is
             * not a property worth freezing. */
            EXPECT_LE(std::sqrt(chosen), std::sqrt(best) + tie + tol(std::sqrt(best) + 1.0))
                << name << ": center_sample is not the closest sample to the k centre, block "
                << r.block_index;
        }
    }
}  // namespace

/*
 * The echo is derived, and the shortcut that derives it is exact.
 *
 * The load-bearing test of the file.  center_sample is computed by splitting
 * the search into the axes a readout sweeps and the ones it does not, and
 * memoizing the first part across repeats -- so the assertion is against the
 * answer the slow unmemoized search gives, over fixtures chosen to exercise
 * the split: Cartesian, an echo train, a blipped EPI, and a rotated
 * non-Cartesian one.
 */
TEST(PulseqKTraj, CentersAreMinimal)
{
    check_centers_are_minimal(load_corpus("gre_2d"), "gre_2d");
    check_centers_are_minimal(load_corpus("gre_2d_3sl"), "gre_2d_3sl");
    check_centers_are_minimal(load_corpus("epi_2d_main"), "epi_2d_main");
    check_centers_are_minimal(load_corpus("fse_2d"), "fse_2d");
    check_centers_are_minimal(load("gre_32x32_pe_blip"), "gre_32x32_pe_blip");
    check_centers_are_minimal(load_corpus("gre_stack_of_stars_3d"), "gre_stack_of_stars_3d");
}

/*
 * Disabling the derivation costs the answer, not the trajectory.
 *
 * Guards the opt-out actually being an opt-out: k must be untouched and only
 * center_sample must go away.
 */
TEST(PulseqKTraj, CenterSampleCanBeDeclined)
{
    const pulseq::Sequence seq = load_corpus("gre_2d");
    const pulseq::KSpace with = pulseq::calculate_kspace(seq);

    pulseq::KSpaceOptions options;
    options.derive_center_sample = false;
    const pulseq::KSpace without = pulseq::calculate_kspace(seq, options);

    ASSERT_EQ(with.total_samples, without.total_samples);
    for (size_t i = 0; i < with.k_adc.size(); ++i)
        ASSERT_EQ(with.k_adc[i], without.k_adc[i]) << "declining centers changed k at " << i;
    for (const pulseq::Readout& r : without.readouts)
        EXPECT_EQ(r.center_sample, -1);
}

/*
 * A rotation extension composes into k, and declining it leaves the logical
 * frame behind.
 *
 * PyPulseq cannot read a ROTATIONS extension, so this is checked against the
 * fixture's own rotation instead: the physical answer has to differ from the
 * logical one on a fixture that carries rotations at all.
 */
TEST(PulseqKTraj, RotationExtensionComposes)
{
    const pulseq::Sequence seq = load_corpus("gre_stack_of_stars_3d");
    ASSERT_GT(seq.rotation_library().size(), 0) << "fixture carries no rotations";

    const pulseq::KSpace physical = pulseq::calculate_kspace(seq);

    pulseq::KSpaceOptions options;
    options.apply_rotation = false;
    const pulseq::KSpace logical = pulseq::calculate_kspace(seq, options);

    ASSERT_EQ(physical.readouts.size(), logical.readouts.size());

    bool differs = false;
    for (size_t i = 0; i < physical.readouts.size() && !differs; ++i)
    {
        const pulseq::Readout& r = physical.readouts[i];
        for (int j = 0; j < r.num_samples; ++j)
        {
            const double lx = sample(logical, logical.readouts[i], 0, j);
            const double ly = sample(logical, logical.readouts[i], 1, j);
            const double lz = sample(logical, logical.readouts[i], 2, j);
            const double px = sample(physical, r, 0, j);
            if (std::fabs(px - lx) >
                1e-6 * (std::fabs(lx) + std::fabs(ly) + std::fabs(lz) + 1.0))
            {
                differs = true;
                break;
            }
        }
    }
    EXPECT_TRUE(differs) << "rotation made no difference on a rotated fixture";
}

namespace
{
    /*
     * The floors below are floors, not measurements to keep in step with. They
     * sit well under what the fixtures currently reach (gre 87%, epi 98%, fse
     * 87%, the blipped one 96%) so ordinary changes do not trip them, while a
     * key that stops grouping does.
     */
    void check_memo_fires(const pulseq::Sequence& seq, int max_groups, const std::string& name)
    {
        const pulseq::KSpace ks = pulseq::calculate_kspace(seq);
        ASSERT_GT(static_cast<int>(ks.readouts.size()), max_groups)
            << name << " is too small to say anything about grouping";
        EXPECT_LE(ks.key_groups, max_groups)
            << name << ": the repeat memo grouped fewer readouts than it should have";
    }
}  // namespace

/*
 * The repeat memo actually fires.
 *
 * Correctness is covered above; this covers the thing correctness cannot see.
 * A memo that never hits is not slow, it is *pointless*, and every version of
 * this key that was too strict still produced right answers -- the first one
 * hit 0% of readouts on every fixture and nothing failed.
 */
TEST(PulseqKTraj, TheRepeatMemoFires)
{
    /* One readout shape, many phase-encode lines. */
    check_memo_fires(load_corpus("gre_2d"), 2, "gre_2d");
    check_memo_fires(load_corpus("gre_2d_3sl"), 3, "gre_2d_3sl");
    /* Two alternating readout polarities. */
    check_memo_fires(load_corpus("epi_2d_main"), 4, "epi_2d_main");
    check_memo_fires(load_corpus("fse_3d"), 4, "fse_3d");
    /* Library deduplication switched off: 32 ids for one readout. */
    check_memo_fires(load("gre_32x32_pe_blip"), 3, "gre_32x32_pe_blip");
}

/*
 * A block range bounds what is reported, and k restarts inside it.
 *
 * Unlike pulseq::apply_fov_shift, which accumulates from block 1 whatever
 * range it is asked to modify, this really does start at the range -- a caller
 * asking for a window is asking what that window traverses.
 */
TEST(PulseqKTraj, BlockRangeBoundsTheWalk)
{
    const pulseq::Sequence seq = load_corpus("gre_2d");
    const pulseq::KSpace all = pulseq::calculate_kspace(seq);

    pulseq::KSpaceOptions options;
    options.last_block = seq.num_blocks() / 2;
    const pulseq::KSpace part = pulseq::calculate_kspace(seq, options);

    EXPECT_LT(part.readouts.size(), all.readouts.size());
    EXPECT_LT(part.total_duration, all.total_duration);
}

/*
 * A gradient offset is a background gradient, so it moves k with time even
 * where the sequence drives nothing.
 */
TEST(PulseqKTraj, GradientOffsetMovesK)
{
    const pulseq::Sequence seq = load_corpus("gre_2d");
    const pulseq::KSpace plain = pulseq::calculate_kspace(seq);

    pulseq::KSpaceOptions options;
    options.gradient_offset[1] = 1000.0;
    const pulseq::KSpace offset = pulseq::calculate_kspace(seq, options);

    ASSERT_FALSE(plain.readouts.empty());
    EXPECT_GT(std::fabs(sample(offset, offset.readouts[0], 1, 0) -
                        sample(plain, plain.readouts[0], 1, 0)),
              1e-6)
        << "gradient offset did not move ky";
}

/* An RF pulse with no declared use is refused rather than guessed at. */
TEST(PulseqKTraj, UndeclaredRfUseIsRefused)
{
    pulseq::Sequence seq;
    seq.set_rasters(1e-6, 10e-6, 100e-9, 10e-6);

    /* amplitude, mag, phase, time, center, delay, freq_ppm, phase_ppm, freq, phase */
    const double rf[pulseq::RF_WIDTH] = {100.0, 0.0, 0.0, 0.0, 500e-6,
                                         0.0,   0.0, 0.0, 0.0, 0.0};

    pulseq::Block block;
    block.rf = seq.register_rf(rf, 'u');
    block.duration = 1000e-6;
    seq.add_block(block);

    EXPECT_THROW(pulseq::calculate_kspace(seq), std::runtime_error);

    /* Declared, and it goes through. */
    pulseq::Sequence declared;
    declared.set_rasters(1e-6, 10e-6, 100e-9, 10e-6);
    pulseq::Block ok;
    ok.rf = declared.register_rf(rf, 'e');
    ok.duration = 1000e-6;
    declared.add_block(ok);
    EXPECT_EQ(pulseq::calculate_kspace(declared).excitations.size(), 1u);
}

/* An empty or inverted block range is refused rather than silently clamped. */
TEST(PulseqKTraj, AnInvertedBlockRangeIsRefused)
{
    const pulseq::Sequence seq = load_corpus("gre_2d");

    pulseq::KSpaceOptions options;
    options.first_block = 10;
    options.last_block = 2;
    EXPECT_THROW(pulseq::calculate_kspace(seq, options), std::runtime_error);

    EXPECT_THROW(pulseq::calculate_kspace(pulseq::Sequence{}), std::runtime_error);
}

/*
 * Window-averaged sampling changes nothing on a flat gradient.
 *
 * Which is the whole reason it is safe to offer.  A sample's true coordinate is
 * k averaged over its dwell, and the average of a straight line over a
 * symmetric window is its midpoint -- so on a constant gradient, where k is
 * exactly linear in time, the correction is identically zero.  Anything that
 * did move would mean the second antiderivative is wrong, and that is much
 * easier to see here than in the ramp case where the correction is supposed to
 * be small but non-zero.
 */
TEST(PulseqKTraj, WindowAverageIsANoOpOnAFlatGradient)
{
    const pulseq::Sequence seq = load_corpus("fse_2d");
    const pulseq::KSpace midpoint = pulseq::calculate_kspace(seq);

    pulseq::KSpaceOptions options;
    options.sample_window_average = true;
    const pulseq::KSpace averaged = pulseq::calculate_kspace(seq, options);

    ASSERT_EQ(midpoint.readouts.size(), averaged.readouts.size());
    const pulseq::Readout& r = midpoint.readouts[0];
    for (int j = 0; j < r.num_samples; ++j)
    {
        const double a = sample(midpoint, r, 0, j);
        const double b = sample(averaged, averaged.readouts[0], 0, j);
        EXPECT_NEAR(a, b, tol(std::fabs(a) + 1.0))
            << "window averaging moved a sample on a flat gradient";
    }
}

/*
 * Where the gradient does vary, the correction is right.
 *
 * "Right" needs an oracle, and the obvious one -- that the correction should
 * equal (dwell^2/24) * dg/dt -- turns out not to apply to any fixture here.
 * That identity holds only while the gradient is linear across the whole
 * window, and with an 80 us dwell on a 20 us raster every window spans four
 * gradient segments.  Measured: zero samples in the fixture set satisfy it.
 *
 * So the oracle is built out of the midpoint path instead, which is already
 * checked against PyPulseq to 1e-12.  `trajectory_delay` shifts the gradient
 * time base, so a run with delay d reports k(t_j - d) at every sample; sweeping
 * d across one dwell samples k *inside* each window, and Simpson over those
 * samples is the window average computed a completely different way -- by
 * quadrature over many evaluations rather than in closed form from a second
 * antiderivative.
 *
 * The spiral is used because it is the only fixture whose readout gradient
 * actually varies: everywhere else the ADC sits on a flat top and the
 * correction is identically zero, which the previous test covers.
 */
TEST(PulseqKTraj, WindowAverageMatchesAnIndependentQuadrature)
{
    /* Even, so Simpson's rule applies; 64 is far past convergence here. */
    constexpr int kSubsamples = 64;

    const pulseq::Sequence seq = load_corpus("gre_spiral_2d");

    pulseq::KSpaceOptions windowed_options;
    windowed_options.sample_window_average = true;
    const pulseq::KSpace windowed = pulseq::calculate_kspace(seq, windowed_options);
    ASSERT_FALSE(windowed.readouts.empty());

    const double dwell = windowed.readouts[0].dwell;
    std::vector<pulseq::KSpace> sub;
    sub.reserve(kSubsamples + 1);
    for (int m = 0; m <= kSubsamples; ++m)
    {
        pulseq::KSpaceOptions options;
        const double shift =
            0.5 * dwell - dwell * static_cast<double>(m) / static_cast<double>(kSubsamples);
        options.trajectory_delay = {{shift, shift, shift}};
        sub.push_back(pulseq::calculate_kspace(seq, options));
    }

    bool moved = false;
    for (size_t i = 0; i < windowed.readouts.size(); ++i)
    {
        const pulseq::Readout& r = windowed.readouts[i];
        for (int a = 0; a < 3; ++a)
        {
            for (int j = 0; j < r.num_samples; ++j)
            {
                double acc = 0.0;
                for (int m = 0; m <= kSubsamples; ++m)
                {
                    const double w =
                        (m == 0 || m == kSubsamples) ? 1.0 : ((m % 2) ? 4.0 : 2.0);
                    acc += w * sample(sub[static_cast<size_t>(m)],
                                      sub[static_cast<size_t>(m)].readouts[i], a, j);
                }
                const double reference = acc / (3.0 * static_cast<double>(kSubsamples));
                const double got = sample(windowed, r, a, j);

                const double midpoint = sample(sub[kSubsamples / 2],
                                               sub[kSubsamples / 2].readouts[i], a, j);
                if (std::fabs(got - midpoint) > 1e-3)
                    moved = true;
                ASSERT_NEAR(got, reference, 1e-6 * (std::fabs(reference) + 1.0))
                    << "closed-form window average disagrees with quadrature";
            }
        }
    }
    EXPECT_TRUE(moved) << "window averaging changed nothing on a varying gradient";
}

/*
 * A centre-out readout has its echo at sample 0, not at the middle.
 *
 * The plainest statement of what this replaces: a design-time anchor that falls
 * back to `num_samples / 2` reports 32 of 64 for a shot that starts at the
 * centre of k-space and marches outwards -- half a readout of error, in the
 * index a reconstruction centres its FFT on.
 */
TEST(PulseqKTraj, ACentreOutReadoutPutsItsEchoAtTheFirstSample)
{
    const pulseq::Sequence seq = load_corpus("zte_3d");
    const pulseq::KSpace ks = pulseq::calculate_kspace(seq);
    ASSERT_FALSE(ks.readouts.empty());

    for (const pulseq::Readout& r : ks.readouts)
    {
        EXPECT_EQ(r.center_sample, 0) << "block " << r.block_index;

        /* And it is the first sample because it is nearest, not because the
         * search stopped early: distance grows along the shot. */
        double first = 0.0;
        double middle = 0.0;
        for (int a = 0; a < 3; ++a)
        {
            const double v0 = sample(ks, r, a, 0) - ks.k_center[static_cast<size_t>(a)];
            const double vm =
                sample(ks, r, a, r.num_samples / 2) - ks.k_center[static_cast<size_t>(a)];
            first += v0 * v0;
            middle += vm * vm;
        }
        EXPECT_GT(middle, 4.0 * first)
            << "the middle sample is not markedly further out than the first";

        for (int j = 2; j < r.num_samples; ++j)
        {
            double prev = 0.0;
            double here = 0.0;
            for (int a = 0; a < 3; ++a)
            {
                const double p = sample(ks, r, a, j - 1) - ks.k_center[static_cast<size_t>(a)];
                const double h = sample(ks, r, a, j) - ks.k_center[static_cast<size_t>(a)];
                prev += p * p;
                here += h * h;
            }
            ASSERT_GE(here, prev) << "a centre-out shot turned back towards the centre";
        }
    }
}

/*
 * A bipolar train's two polarities put the echo on different samples, and
 * mirroring the reverse ones brings them back together.
 *
 * This is the property that makes an EPI reconstructable.  The two polarities
 * are mirror images, so their nearest samples to the origin are the *same
 * point* reached from opposite ends: with 128 samples the forward readout finds
 * it at 64 and the reverse one at 63.  A rule that resolved the tie by index
 * rather than by direction would report 64 for both, and mirroring -- which is
 * what the reverse readout's REV label asks a reconstruction to do -- would
 * then land half the lines on 63.  Half a k-space one sample off centre is a
 * linear phase ramp across the image.
 */
TEST(PulseqKTraj, ABipolarTrainPutsEveryEchoAtOnePlaceOnceMirrored)
{
    const pulseq::Sequence seq = load_corpus("epi_2d_main");
    const pulseq::KSpace ks = pulseq::calculate_kspace(seq);
    ASSERT_GT(ks.readouts.size(), 1u);

    int mirrored = -1;
    bool seen_forward = false;
    bool seen_reverse = false;

    for (const pulseq::Readout& r : ks.readouts)
    {
        if (r.num_samples < 2 || r.center_sample < 0)
            continue;

        /* Direction of travel: the dominant component of the displacement from
         * the first sample to the last.  The same quantity a REV label
         * carries, worked out here from k alone. */
        double travel = 0.0;
        for (int a = 0; a < 3; ++a)
        {
            const double d = sample(ks, r, a, r.num_samples - 1) - sample(ks, r, a, 0);
            if (std::fabs(d) > std::fabs(travel))
                travel = d;
        }

        int here = 0;
        if (travel >= 0.0)
        {
            seen_forward = true;
            here = r.center_sample;
        }
        else
        {
            seen_reverse = true;
            here = r.num_samples - 1 - r.center_sample;
        }

        if (mirrored < 0)
            mirrored = here;
        ASSERT_EQ(mirrored, here) << "block " << r.block_index;
    }

    EXPECT_TRUE(seen_forward && seen_reverse)
        << "the fixture is not bipolar, so this proves nothing";
    /* 32 samples spanning the origin: the echo belongs at N/2. */
    EXPECT_EQ(mirrored, 16);
}
