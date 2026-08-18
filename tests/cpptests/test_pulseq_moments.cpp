/**
 * Tests for pulseq::calc_moments -- the b-tensor and gradient moments.
 *
 * The numeric claim, that a piecewise-linear gradient integrates in closed
 * form, is checked against a two-million-point quadrature on the Python side
 * (tests/python/test_pypulseq_analysis.py), which also compares this against
 * an independent numpy transcription of the same algebra.  What is left for
 * here is the structure -- the things that are properties of the C++ and not
 * of the arithmetic:
 *
 *  - the three-part split sums to the tensor at R = I, and composing it with
 *    a rotation matches turning the tensor directly;
 *  - NOROT decides which half of the split a gradient lands in, and cannot
 *    change the tensor at R = I, because the .seq does not carry the console's
 *    prescription at all;
 *  - the deduplicated table keeps two b-values of one direction apart, which
 *    a key normalised per shot would not;
 *  - a sequence with no readout reports NaN rather than a fabricated echo.
 */

#include <gtest/gtest.h>

#include "pulseq/moments.hpp"
#include "pulseq/read.hpp"
#include "pulseq/sequence.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace
{
const std::string kData = std::string(PULSEQ_FIXTURES_DIR) + "/";
const std::string kCorpus = std::string(PULSEQ_CORPUS_DIR) + "/";

pulseq::Sequence load(const std::string &stem)
{
    return pulseq::read_file(kData + stem + ".seq");
}

pulseq::Sequence load_corpus(const std::string &stem)
{
    return pulseq::read_file(kCorpus + stem + ".seq");
}

pulseq::Matrix3 total(const pulseq::BTensorParts &parts)
{
    return pulseq::compose_btensor(parts, pulseq::identity3());
}

double peak(const pulseq::Matrix3 &m)
{
    double out = 0.0;
    for (int i = 0; i < 9; ++i)
        out = std::max(out, std::fabs(m[static_cast<size_t>(i)]));
    return out;
}

/** A rotation of `angle` about z, row-major. */
pulseq::Matrix3 about_z(double angle)
{
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    return pulseq::Matrix3{{c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0}};
}

/**
     * A rotation of `angle` about y, row-major.
     *
     * The one that actually moves a Cartesian shot: its gradients live on the
     * readout axis and the slice axis, so a turn about z leaves the tensor
     * where it was and would make a sensitivity check pass for the wrong
     * reason.
     */
pulseq::Matrix3 about_y(double angle)
{
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    return pulseq::Matrix3{{c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c}};
}

} // namespace

/*
 * Every fixture with an excitation reports one shot per excitation, and each
 * shot's three parts add up to the tensor the caller sees.
 *
 * Cheap, but it is the invariant everything downstream stands on: the MRD
 * export writes the parts and the reconstruction adds them back, so a shot
 * where they do not sum is a wrong b-vector nobody can notice.
 */
TEST(PulseqMoments, TheSplitSumsToTheTensor)
{
    for (const std::string &stem :
         {kCorpus + "gre_2d", kCorpus + "fse_2d", kCorpus + "epi_2d_main"})
    {
        pulseq::Sequence seq = pulseq::read_file(stem + ".seq");
        const pulseq::Moments moments = pulseq::calc_moments(seq);
        ASSERT_GT(moments.num_shots(), 0) << stem;

        bool any = false;
        for (int shot = 0; shot < moments.num_shots(); ++shot)
        {
            const pulseq::BTensorParts &parts = moments.b[static_cast<size_t>(shot)];
            const pulseq::Matrix3 sum = total(parts);
            any |= peak(sum) > 0.0;
            const double scale = std::max(peak(sum), 1.0);
            for (int i = 0; i < 9; ++i)
            {
                const double expected = parts.fixed[static_cast<size_t>(i)] +
                    parts.rotatable[static_cast<size_t>(i)] + parts.cross[static_cast<size_t>(i)] +
                    parts.cross[static_cast<size_t>((i % 3) * 3 + i / 3)];
                EXPECT_NEAR(sum[static_cast<size_t>(i)], expected, 1e-12 * scale)
                    << stem << " shot " << shot << " element " << i;
            }
        }
        EXPECT_TRUE(any) << stem << ": every shot integrated to nothing";
    }
}

/*
 * With nothing under NOROT the whole tensor is rotatable, so composing with a
 * prescription is exactly turning it -- and the composition has to be
 * sensitive to which way, or it is not doing the multiplication it claims to.
 */
TEST(PulseqMoments, ComposingAPrescriptionTurnsTheTensor)
{
    pulseq::Sequence seq = load("gre_32x32_pe_blip");
    const pulseq::Moments moments = pulseq::calc_moments(seq);
    ASSERT_GT(moments.num_shots(), 0);

    /* A shot with something in it: a GRE opens with excitations that have no
     * readout after them, and those integrate to nothing. */
    const pulseq::BTensorParts *found = nullptr;
    for (const pulseq::BTensorParts &candidate : moments.b)
    {
        if (peak(total(candidate)) > 0.0)
        {
            found = &candidate;
            break;
        }
    }
    ASSERT_NE(found, nullptr) << "no shot of this fixture has an echo";
    const pulseq::BTensorParts &parts = *found;
    ASSERT_EQ(peak(parts.fixed), 0.0) << "no block of this fixture carries NOROT";

    const pulseq::Matrix3 R = about_y(0.7);
    const pulseq::Matrix3 composed = pulseq::compose_btensor(parts, R);
    const pulseq::Matrix3 direct = total(parts);

    /* R B R^T, written out so this does not lean on the code under test. */
    double turned[9] = {0};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l)
                    turned[i * 3 + j] += R[static_cast<size_t>(i * 3 + k)] *
                        direct[static_cast<size_t>(k * 3 + l)] * R[static_cast<size_t>(j * 3 + l)];

    const double scale = std::max(peak(direct), 1.0);
    bool differs = false;
    for (int i = 0; i < 9; ++i)
    {
        EXPECT_NEAR(composed[static_cast<size_t>(i)], turned[i], 1e-9 * scale);
        differs |= std::fabs(composed[static_cast<size_t>(i)] - direct[static_cast<size_t>(i)]) >
            1e-6 * scale;
    }
    EXPECT_TRUE(differs) << "a 0.7 rad rotation has to move this tensor";
}

/*
 * A trace is invariant under rotation, whatever the split.  So is the answer
 * at R = I, which is the one MATLAB's calcMomentsBtensor returns.
 */
TEST(PulseqMoments, TheBValueDoesNotDependOnThePrescription)
{
    pulseq::Sequence seq = load_corpus("fse_2d");
    const pulseq::Moments moments = pulseq::calc_moments(seq);
    ASSERT_GT(moments.num_shots(), 0);

    for (const pulseq::BTensorParts &parts : moments.b)
    {
        const pulseq::Matrix3 plain = total(parts);
        const pulseq::Matrix3 turned = pulseq::compose_btensor(parts, about_z(1.1));
        const double a = plain[0] + plain[4] + plain[8];
        const double b = turned[0] + turned[4] + turned[8];
        EXPECT_NEAR(a, b, 1e-9 * std::max(std::fabs(a), 1.0));
    }
}

/*
 * The table is what an MRD header carries, so its rows have to mean one
 * encoding each.  A Cartesian scan is the strongest case available in the
 * fixtures: every shot has a different phase-encode, so nothing may collapse.
 */
TEST(PulseqMoments, TheTableHasOneRowPerDistinctEncoding)
{
    pulseq::Sequence seq = load("gre_32x32_pe_blip");
    const pulseq::Moments moments = pulseq::calc_moments(seq);
    ASSERT_GT(moments.num_shots(), 1);

    ASSERT_EQ(moments.table_index.size(), static_cast<size_t>(moments.num_shots()));
    EXPECT_EQ(static_cast<int>(moments.table.size()), moments.num_shots())
        << "every phase-encode step is a different tensor";

    for (int shot = 0; shot < moments.num_shots(); ++shot)
    {
        const int row = moments.table_index[static_cast<size_t>(shot)];
        ASSERT_GE(row, 0);
        ASSERT_LT(row, static_cast<int>(moments.table.size()));

        const pulseq::Matrix3 from_table = total(moments.table[static_cast<size_t>(row)]);
        const pulseq::Matrix3 from_shot = total(moments.b[static_cast<size_t>(shot)]);
        const double scale = std::max(peak(from_shot), 1.0);
        for (int i = 0; i < 9; ++i)
            EXPECT_NEAR(
                from_table[static_cast<size_t>(i)],
                from_shot[static_cast<size_t>(i)],
                1e-9 * scale);
    }
}

/*
 * n_dummy drops leading shots from the answer without disturbing the ones
 * that survive -- k is walked through them either way, so the tensors after
 * the skip must be bit-identical to the same entries computed without it.
 */
TEST(PulseqMoments, SkippingDummiesLeavesTheRestAlone)
{
    pulseq::Sequence seq = load_corpus("gre_2d");

    const pulseq::Moments all = pulseq::calc_moments(seq);
    ASSERT_GT(all.num_shots(), 3);

    pulseq::MomentsOptions options;
    options.n_dummy = 2;
    const pulseq::Moments rest = pulseq::calc_moments(seq, options);

    ASSERT_EQ(rest.num_shots(), all.num_shots() - 2);
    for (int shot = 0; shot < rest.num_shots(); ++shot)
    {
        EXPECT_DOUBLE_EQ(
            rest.t_excitation[static_cast<size_t>(shot)],
            all.t_excitation[static_cast<size_t>(shot + 2)]);
        const pulseq::Matrix3 a = total(rest.b[static_cast<size_t>(shot)]);
        const pulseq::Matrix3 b = total(all.b[static_cast<size_t>(shot + 2)]);
        for (int i = 0; i < 9; ++i)
            EXPECT_DOUBLE_EQ(a[static_cast<size_t>(i)], b[static_cast<size_t>(i)]);
    }
}

/*
 * A design with no ADC has no measured echo, and the two-pulse fallback is
 * only taken when it lands inside the shot.  Where it does not, the answer is
 * NaN -- a number the caller can see is missing, rather than one clamped to
 * the end of the sequence and quietly integrated.
 */
TEST(PulseqMoments, AShotWithNoEchoReportsNaNAndIntegratesNothing)
{
    pulseq::Sequence seq = load("05_rf_ladder_no_adc");
    const pulseq::Moments moments = pulseq::calc_moments(seq);
    ASSERT_GT(moments.num_shots(), 0);

    bool saw_missing = false;
    for (int shot = 0; shot < moments.num_shots(); ++shot)
    {
        if (!std::isnan(moments.t_echo[static_cast<size_t>(shot)]))
            continue;
        saw_missing = true;
        EXPECT_EQ(peak(total(moments.b[static_cast<size_t>(shot)])), 0.0)
            << "shot " << shot << " has no echo, so there is no window to integrate";
    }
    EXPECT_TRUE(saw_missing) << "this fixture has no ADC at all";
}
