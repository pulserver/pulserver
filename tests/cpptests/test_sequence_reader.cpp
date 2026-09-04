// test_sequence_reader.cpp
//
// The recon-side reader, end to end over the corpus: read_sequence_files
// populates a SequenceCache from the seqfile chain and the format-agnostic
// consumers compose trajectories from it. The invariants pinned here were
// proven equivalent to the retired cache pipeline product-by-product before
// that pipeline was deleted; a regression against any of them is a change
// in what a reconstruction receives.

#include "sequence_cache.h"
#include "sequence_file_reader.h"

#include "pulseq/expand.hpp"
#include "pulseq/kspace.hpp"
#include "pulseq/read.hpp"
#include "pulseq/sequence_file.hpp"
#include "pulseq/write.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using mrdserver::SequenceCache;

namespace
{

std::string corpus(const std::string &name)
{
    return (fs::path(PULSEQ_CORPUS_DIR) / name).string();
}

constexpr uint64_t kNavFlag = 1ULL << 22;
constexpr uint64_t kLastFlag = 1ULL << 24;
constexpr uint64_t kFirstInLine = 1ULL << 0;
constexpr uint64_t kLastInLine = 1ULL << 1;
constexpr uint64_t kLastInPartition = 1ULL << 3;
constexpr uint64_t kFirstInAverage = 1ULL << 4;
constexpr uint64_t kLastInAverage = 1ULL << 5;
constexpr uint64_t kFirstInSlice = 1ULL << 6;
constexpr uint64_t kLastInSlice = 1ULL << 7;

/** A corpus sequence with its repetitions written into the block table, as
 *  the design side hands one to the scanner.  Returns the path it wrote. */
std::string expanded_copy(const std::string &name, int repeats, const std::string &out_path)
{
    pulseq::Sequence seq = pulseq::read_file(corpus(name));
    pulseq::expand_repeats(seq, repeats);
    std::ofstream(out_path) << pulseq::write_text(seq, false);
    return out_path;
}

/** The positions carrying `flag`. */
std::set<size_t> flagged(const SequenceCache &cache, uint64_t flag)
{
    std::set<size_t> out;
    for (size_t i = 0; i < cache.table.size(); ++i)
        if (cache.table[i].flags & flag)
            out.insert(i);
    return out;
}

} // namespace

class SequenceReader : public ::testing::TestWithParam<const char *>
{
};

TEST_P(SequenceReader, ReadsDeterministically)
{
    const std::string path = corpus(std::string(GetParam()) + ".seq");
    const SequenceCache first = mrdserver::read_sequence_files(path);
    const SequenceCache second = mrdserver::read_sequence_files(path);

    ASSERT_EQ(second.table.size(), first.table.size());
    for (size_t i = 0; i < first.table.size(); ++i)
    {
        EXPECT_EQ(second.table[i].lin, first.table[i].lin) << i;
        EXPECT_EQ(second.table[i].flags, first.table[i].flags) << i;
        EXPECT_EQ(second.table[i].kx_shot_id, first.table[i].kx_shot_id) << i;
    }
    EXPECT_EQ(second.kshots.size(), first.kshots.size());
    EXPECT_EQ(second.seq_descs.size(), first.seq_descs.size());
}

TEST_P(SequenceReader, TheCacheIsInternallyConsistent)
{
    const SequenceCache cache =
        mrdserver::read_sequence_files(corpus(std::string(GetParam()) + ".seq"));

    ASSERT_FALSE(cache.table.size() == 0);
    ASSERT_FALSE(cache.encoding_spaces.empty());

    // Exactly one final acquisition carries LAST_IN_MEASUREMENT.
    int last_count = 0;
    for (const auto &entry : cache.table)
    {
        last_count += (entry.flags & kLastFlag) ? 1 : 0;
        ASSERT_GE(entry.encoding_space_ref, 0);
        ASSERT_LT(entry.encoding_space_ref, static_cast<int>(cache.encoding_spaces.size()));
        // Every shot id resolves, and NAV routes to the navigator space.
        for (const int shot : {entry.kx_shot_id, entry.ky_shot_id, entry.kz_shot_id})
            ASSERT_LT(shot, static_cast<int>(cache.kshots.size()));
        if (entry.flags & kNavFlag)
            EXPECT_EQ(cache.encoding_spaces[entry.encoding_space_ref].geometry_tag, 1);
    }
    EXPECT_EQ(last_count, 1);
    EXPECT_TRUE((cache.table.back().flags & kLastFlag) != 0);

    // Label limits bound what the table holds.
    for (const auto &entry : cache.table)
    {
        const auto &limits = cache.encoding_spaces[entry.encoding_space_ref].label_limits;
        EXPECT_GE(entry.lin, limits.lin.min);
        EXPECT_LE(entry.lin, limits.lin.max);
        EXPECT_GE(entry.slc, limits.slc.min);
        EXPECT_LE(entry.slc, limits.slc.max);
    }

    // The corpus files declare TRSize, so the description is derived: one
    // row per structural-TR block, roles either one unique centre or a tie.
    ASSERT_TRUE(cache.has_seq_desc);
    for (const auto &sd : cache.seq_descs)
    {
        ASSERT_FALSE(sd.events.empty());
        EXPECT_GT(sd.tr_duration_us, 0.0f);
        int echo_centers = 0;
        for (const auto &row : sd.events)
        {
            if (row.type == mrdserver::SEQ_EVENT_ADC &&
                row.adc_role() == mrdserver::ADC_ROLE_ECHO_CENTER)
                ++echo_centers;
        }
        EXPECT_LE(echo_centers, 1);
        for (const auto &def : sd.rf_defs)
        {
            EXPECT_GE(def.num_bands, 1);
            EXPECT_GT(def.total_b1sq_power, 0.0f);
            EXPECT_FALSE(def.mag.samples.empty());
        }
    }
}

TEST_P(SequenceReader, ResidentAndMaterializedReadoutsAgree)
{
    const SequenceCache cache =
        mrdserver::read_sequence_files(corpus(std::string(GetParam()) + ".seq"));

    const auto resident = mrdserver::pre_compute_trajectories(cache);
    // A budget of zero floats forces the on-demand path.
    const auto deferred = mrdserver::pre_compute_trajectories(cache, 0);
    ASSERT_EQ(deferred.size(), resident.size());

    for (size_t es = 0; es < resident.size(); ++es)
    {
        const auto &a = resident[es];
        const auto &b = deferred[es];
        ASSERT_EQ(b.ndim, a.ndim) << "es " << es;
        if (a.ndim == 0)
            continue;
        ASSERT_EQ(b.num_samples, a.num_samples);
        ASSERT_EQ(b.num_readouts, a.num_readouts);

        std::vector<float> row_a(static_cast<size_t>(a.ndim) * a.num_samples);
        std::vector<float> row_b(row_a.size());
        for (int r = 0; r < a.num_readouts; ++r)
        {
            ASSERT_TRUE(
                mrdserver::materialize_readout(cache, a, static_cast<int>(es), r, row_a.data()));
            ASSERT_TRUE(
                mrdserver::materialize_readout(cache, b, static_cast<int>(es), r, row_b.data()));
            for (size_t s = 0; s < row_a.size(); ++s)
                ASSERT_EQ(row_b[s], row_a[s]) << "es " << es << " readout " << r << " sample " << s;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    Corpus,
    SequenceReader,
    ::testing::Values(
        "gre_2d",
        "gre_2d_3sl",
        "se_2d",
        "fse_2d",
        "bssfp_2d",
        "gre_radial_2d",
        "gre_spiral_2d",
        "gre_stack_of_stars_3d",
        "se_propeller_2d",
        "mprage_stack_of_spirals_3d",
        "zte_3d"));

TEST(SequenceReaderBoundaries, EachSliceOpensAndClosesAtItsOwnAcquisitions)
{
    const SequenceCache cache = mrdserver::read_sequence_files(corpus("gre_2d_3sl.seq"));

    std::map<int, size_t> first_of, last_of;
    for (size_t i = 0; i < cache.table.size(); ++i)
    {
        first_of.emplace(cache.table[i].slc, i);
        last_of[cache.table[i].slc] = i;
    }
    ASSERT_EQ(first_of.size(), 3u);

    std::set<size_t> expect_first, expect_last;
    for (const auto &entry : first_of)
        expect_first.insert(entry.second);
    for (const auto &entry : last_of)
        expect_last.insert(entry.second);

    EXPECT_EQ(flagged(cache, kFirstInSlice), expect_first);
    EXPECT_EQ(flagged(cache, kLastInSlice), expect_last);
}

TEST(SequenceReaderBoundaries, AnEncodingCounterIsBoundedInsideEachFrame)
{
    // A line belongs to one image, so its boundary is read within the frame
    // counters: every acquisition is the last of its own line in its own
    // slice, and there are as many closures as there are acquisitions.
    const SequenceCache cache = mrdserver::read_sequence_files(corpus("gre_2d_3sl.seq"));

    std::set<std::pair<int, int>> pairs;
    for (const auto &entry : cache.table)
        pairs.insert({entry.slc, entry.lin});
    ASSERT_EQ(pairs.size(), cache.table.size());

    EXPECT_EQ(flagged(cache, kLastInLine).size(), cache.table.size());
    EXPECT_EQ(flagged(cache, kFirstInLine).size(), cache.table.size());
}

TEST(SequenceReaderBoundaries, ACounterTheScanNeverWritesIsNeverBounded)
{
    // A 2D scan has no partition axis, so nothing closes one.
    const SequenceCache cache = mrdserver::read_sequence_files(corpus("gre_2d.seq"));

    for (const auto &entry : cache.table)
        ASSERT_EQ(entry.par, 0);
    EXPECT_TRUE(flagged(cache, kLastInPartition).empty());
    EXPECT_FALSE(flagged(cache, kLastInLine).empty());
}

TEST(SequenceReaderBoundaries, AnAverageIsAFrameCounterLikeTheOthers)
{
    // AVG carries the repetition index the design side stamped and nothing
    // else. A repetition is a whole image, so its boundary is read within the
    // other frame counters: an average closes once per slice, where that
    // slice's share of the repetition ends -- and a scan of one average closes
    // none, because there was no repetition to bound.
    const std::string thrice_path =
        (fs::temp_directory_path() / "reader_avg_boundaries_x3.seq").string();
    const SequenceCache once = mrdserver::read_sequence_files(corpus("gre_2d_3sl.seq"));
    const SequenceCache thrice =
        mrdserver::read_sequence_files(expanded_copy("gre_2d_3sl.seq", 3, thrice_path));

    EXPECT_TRUE(flagged(once, kLastInAverage).empty());
    EXPECT_EQ(flagged(thrice, kLastInSlice).size(), 3 * flagged(once, kLastInSlice).size());

    std::map<std::pair<int, int>, size_t> first_of, last_of;
    for (size_t i = 0; i < thrice.table.size(); ++i)
    {
        const std::pair<int, int> unit{thrice.table[i].slc, thrice.table[i].avg};
        first_of.emplace(unit, i);
        last_of[unit] = i;
    }
    ASSERT_EQ(last_of.size(), 9u);

    std::set<size_t> expect_first, expect_last;
    for (const auto &entry : first_of)
        expect_first.insert(entry.second);
    for (const auto &entry : last_of)
        expect_last.insert(entry.second);

    EXPECT_EQ(flagged(thrice, kFirstInAverage), expect_first);
    EXPECT_EQ(flagged(thrice, kLastInAverage), expect_last);

    fs::remove(thrice_path);
}

/*
 * The cache holds k in the logical gradient frame with the rotation left as an
 * id; an acquisition has to say where the sample physically sits. So the
 * rotation is composed when the acquisition is written -- and before the
 * trailing-zero axes are pruned, because a rotation mixes axes and an axis
 * pruned first is one the mixing can no longer reach.
 */
TEST(SequenceReaderRotation, AnAcquisitionCarriesTheRotatedTrajectory)
{
    const SequenceCache cache = mrdserver::read_sequence_files(corpus("gre_radial_2d.seq"));
    const auto trajectories = mrdserver::pre_compute_trajectories(cache);

    // Readout index within its encoding space, the way the emitter builds it.
    std::vector<int> readout_index_in_es(cache.table.size(), -1);
    std::vector<int> seen(cache.encoding_spaces.size(), 0);
    for (size_t t = 0; t < cache.table.size(); ++t)
    {
        const int es = cache.table[t].encoding_space_ref;
        if (es >= 0 && es < static_cast<int>(seen.size()))
            readout_index_in_es[t] = seen[es]++;
    }

    // A spoke whose rotation actually turns something. The id alone is not
    // enough: a radial set's first spoke carries a rotation event whose matrix
    // is (near) identity, and it would pass a rotate-then-compare test that
    // never rotated anything.
    int rotated = -1;
    for (size_t t = 0; t < cache.table.size() && rotated < 0; ++t)
    {
        const int id = cache.table[t].rotation_id;
        if (id <= 0 || id >= static_cast<int>(cache.rotations.size()))
            continue;
        const std::array<float, 9> &R = cache.rotations[static_cast<size_t>(id)];
        for (int e = 0; e < 9; ++e)
        {
            const float identity = (e % 4 == 0) ? 1.0f : 0.0f;
            if (std::fabs(R[static_cast<size_t>(e)] - identity) > 1e-3f)
            {
                rotated = static_cast<int>(t);
                break;
            }
        }
    }
    ASSERT_GE(rotated, 0) << "the radial fixture carries no non-identity rotation";

    const auto &entry = cache.table[static_cast<size_t>(rotated)];
    const auto &pt = trajectories[static_cast<size_t>(entry.encoding_space_ref)];
    ASSERT_GT(pt.ndim, 0);

    // What the cache holds, unrotated, back on its real axes.
    std::vector<float> packed(static_cast<size_t>(pt.ndim) * pt.num_samples);
    ASSERT_TRUE(mrdserver::materialize_readout(
        cache,
        pt,
        entry.encoding_space_ref,
        readout_index_in_es[static_cast<size_t>(rotated)],
        packed.data()));

    std::vector<float> logical(3 * static_cast<size_t>(pt.num_samples), 0.0f);
    int col = 0;
    for (int axis = 0; axis < 3; ++axis)
    {
        if (!pt.axis_active[axis])
            continue;
        for (int s = 0; s < pt.num_samples; ++s)
            logical[static_cast<size_t>(axis) * pt.num_samples + s] =
                packed[static_cast<size_t>(s) * pt.ndim + col];
        ++col;
    }

    ISMRMRD::Acquisition acq;
    acq.resize(static_cast<uint16_t>(pt.num_samples), 1, 0);
    mrdserver::enrich_ismrmrd_acquisition(
        acq,
        rotated,
        1u,
        0.0f,
        cache,
        trajectories,
        readout_index_in_es);

    const int ndim = static_cast<int>(acq.trajectory_dimensions());
    ASSERT_GT(ndim, 0);
    const float *k = acq.getTrajPtr();
    const std::array<float, 9> &R = cache.rotations[static_cast<size_t>(entry.rotation_id)];

    double worst = 0.0, scale = 0.0;
    for (int s = 0; s < pt.num_samples; ++s)
        for (int a = 0; a < ndim; ++a)
        {
            double expected = 0.0;
            for (int d = 0; d < 3; ++d)
                expected += R[static_cast<size_t>(a) * 3 + d] *
                    logical[static_cast<size_t>(d) * pt.num_samples + s];
            worst = std::max(worst, std::fabs(static_cast<double>(k[s * ndim + a]) - expected));
            scale = std::max(scale, std::fabs(expected));
        }
    ASSERT_GT(scale, 0.0) << "the rotated spoke came back identically zero";
    EXPECT_LT(worst, 1e-3 * scale);

    // And it is genuinely a rotation, not a copy: at least one sample moved.
    double moved = 0.0;
    for (int s = 0; s < pt.num_samples; ++s)
        for (int a = 0; a < ndim; ++a)
            moved = std::max(
                moved,
                std::fabs(
                    static_cast<double>(k[s * ndim + a]) -
                    static_cast<double>(logical[static_cast<size_t>(a) * pt.num_samples + s])));
    EXPECT_GT(moved, 1e-6 * scale);
}

/*
 * The reader reads the trajectory the design side wrote into the seqfile
 * instead of integrating the gradients again. Those are two routes to one
 * number, so they are held equal here -- and the equality is what makes
 * dropping the second route safe.
 *
 * The other half of this test is that the trajectory exists at all. When the
 * reader was first switched over against a corpus that carried no stored
 * bases, every non-Cartesian sequence came back with none, and 626 of 627
 * tests still passed: nothing else in the suite looks at trajectory content.
 */
class SequenceReaderNonCartesian : public ::testing::TestWithParam<const char *>
{
};

TEST_P(SequenceReaderNonCartesian, TheStoredBaseComposesToWhatTheGradientsIntegrateTo)
{
    const std::string name = GetParam();
    const SequenceCache cache = mrdserver::read_sequence_files(corpus(name + ".seq"));
    const auto trajectories = mrdserver::pre_compute_trajectories(cache);

    bool any_trajectory = false;
    for (const auto &pt : trajectories)
        any_trajectory = any_trajectory || pt.ndim > 0;
    ASSERT_TRUE(any_trajectory) << name << " came back with no trajectory on any encoding space";

    // The reference: integrate the gradients, logical frame, as the design
    // side did when it wrote the base.
    pulseq::SequenceFile file(corpus(name + ".seq"));
    pulseq::Sequence &seq = file.sequence();
    pulseq::KSpaceOptions options;
    options.apply_rotation = false;
    const pulseq::KSpace ks = pulseq::calculate_kspace(seq, options);
    ASSERT_EQ(cache.table.size(), ks.readouts.size()) << "single-subsequence fixture expected";

    std::vector<int> readout_index_in_es(cache.table.size(), -1);
    std::vector<int> seen(cache.encoding_spaces.size(), 0);
    for (size_t t = 0; t < cache.table.size(); ++t)
    {
        const int es = cache.table[t].encoding_space_ref;
        if (es >= 0 && es < static_cast<int>(seen.size()))
            readout_index_in_es[t] = seen[es]++;
    }

    double worst = 0.0, scale = 0.0;
    size_t compared = 0;
    for (size_t t = 0; t < cache.table.size(); ++t)
    {
        const int es = cache.table[t].encoding_space_ref;
        if (es < 0 || es >= static_cast<int>(trajectories.size()))
            continue;
        const auto &pt = trajectories[static_cast<size_t>(es)];
        if (pt.ndim <= 0)
            continue;

        std::vector<float> packed(static_cast<size_t>(pt.ndim) * pt.num_samples);
        ASSERT_TRUE(
            mrdserver::materialize_readout(cache, pt, es, readout_index_in_es[t], packed.data()));

        const pulseq::Readout &readout = ks.readouts[t];
        const double *reference = ks.k_adc.data() + static_cast<size_t>(readout.sample_offset) * 3;

        int col = 0;
        for (int axis = 0; axis < 3; ++axis)
        {
            if (!pt.axis_active[axis])
                continue;
            for (int sample = 0; sample < pt.num_samples && sample < readout.num_samples; ++sample)
            {
                const double expected =
                    reference[static_cast<size_t>(axis) * readout.num_samples + sample];
                const double got =
                    static_cast<double>(packed[static_cast<size_t>(sample) * pt.ndim + col]);
                worst = std::max(worst, std::fabs(got - expected));
                scale = std::max(scale, std::fabs(expected));
                ++compared;
            }
            ++col;
        }
    }

    ASSERT_GT(compared, 0u) << "no readout of " << name << " carried a trajectory";
    ASSERT_GT(scale, 0.0) << name << "'s trajectory is identically zero";
    EXPECT_LT(worst, 1e-3 * scale) << "worst " << worst << " against scale " << scale;
}

INSTANTIATE_TEST_SUITE_P(
    Corpus,
    SequenceReaderNonCartesian,
    ::testing::Values("gre_spiral_2d", "gre_radial_2d", "gre_stack_of_stars_3d", "gre_wave_3d"));

TEST(SequenceReaderChain, TheEpiChainBecomesOneCacheOfSubsequences)
{
    const SequenceCache cache = mrdserver::read_sequence_files(corpus("epi_2d.seq"));

    ASSERT_GE(cache.definitions_by_subseq.size(), 2u);
    std::set<int> subseqs;
    for (const auto &es : cache.encoding_spaces)
        subseqs.insert(es.subseq_idx);
    EXPECT_EQ(subseqs.size(), cache.definitions_by_subseq.size());

    // The navigator labels split an encoding space off somewhere.
    bool has_nav_space = false;
    for (const auto &es : cache.encoding_spaces)
        has_nav_space = has_nav_space || es.geometry_tag == 1;
    EXPECT_TRUE(has_nav_space);
}

TEST(SequenceReaderAverages, TheTableIsWhatTheFileSays)
{
    // The reader tiles nothing: repetitions reach it already written into the
    // block table, so a file expanded three times has three times the rows and
    // the AVG counter the expansion stamped.
    const std::string thrice_path =
        (fs::temp_directory_path() / "reader_avg_table_x3.seq").string();
    const SequenceCache once = mrdserver::read_sequence_files(corpus("gre_2d.seq"));
    const SequenceCache thrice =
        mrdserver::read_sequence_files(expanded_copy("gre_2d.seq", 3, thrice_path));

    ASSERT_EQ(thrice.table.size(), once.table.size() * 3);
    for (size_t i = 0; i < thrice.table.size(); ++i)
    {
        const auto &entry = thrice.table[i];
        EXPECT_EQ(entry.avg, static_cast<int>(i / once.table.size()));
        // REP passes through from the file's labels, untouched.
        EXPECT_EQ(entry.rep, once.table[i % once.table.size()].rep);
    }
    EXPECT_EQ(thrice.table.back().flags & kLastFlag, kLastFlag);
    EXPECT_EQ(once.table.back().flags & kLastFlag, kLastFlag);

    fs::remove(thrice_path);
}

/*
 * A wave-encoded Cartesian scan is Cartesian in its counters and not in its
 * readout: the corkscrew sweeps ky and kz across the ADC window, so the
 * readout stores a base trajectory the way a spiral's does, and that base is
 * what a reconstruction reads the corkscrew off. That the base is faithful is
 * held by SequenceReaderNonCartesian, which this fixture is also a case of.
 *
 * What is held here is what a reconstruction sorts by. The fixture carries
 * both passes -- the wave-free autocalibration rectangle and the wave-encoded
 * train -- and telling them apart is the first thing the reconstruction does.
 */
TEST(SequenceReaderWave, TheCorkscrewReachesTheAcquisition)
{
    constexpr uint64_t kParallelCalibration = 1ULL << 19;

    const SequenceCache cache = mrdserver::read_sequence_files(corpus("gre_wave_3d.seq"));
    const auto trajectories = mrdserver::pre_compute_trajectories(cache);

    std::vector<int> readout_index_in_es(cache.table.size(), -1);
    std::vector<int> seen(cache.encoding_spaces.size(), 0);
    for (size_t t = 0; t < cache.table.size(); ++t)
    {
        const int es = cache.table[t].encoding_space_ref;
        if (es >= 0 && es < static_cast<int>(seen.size()))
            readout_index_in_es[t] = seen[es]++;
    }

    /* How one axis moved within the readout, against where it started -- which
     * is where that line's own phase encode put it. */
    const auto excursion = [](const float *k, int ndim, int samples, int axis)
    {
        std::vector<double> out(static_cast<size_t>(samples));
        const double start = static_cast<double>(k[axis]);
        for (int s = 0; s < samples; ++s)
            out[static_cast<size_t>(s)] = static_cast<double>(k[s * ndim + axis]) - start;
        return out;
    };
    const auto peak = [](const std::vector<double> &values)
    {
        double out = 0.0;
        for (double value : values)
            out = std::max(out, std::fabs(value));
        return out;
    };

    std::vector<double> corkscrew[2];
    size_t waved = 0, flat = 0;

    for (size_t t = 0; t < cache.table.size(); ++t)
    {
        const int es = cache.table[t].encoding_space_ref;
        ASSERT_GE(es, 0);
        const auto &pt = trajectories[static_cast<size_t>(es)];
        ASSERT_GT(pt.ndim, 0) << "readout " << t << " belongs to a space with no trajectory";

        ISMRMRD::Acquisition acq;
        acq.resize(static_cast<uint16_t>(pt.num_samples), 1, 0);
        mrdserver::enrich_ismrmrd_acquisition(
            acq,
            static_cast<int>(t),
            1u,
            0.0f,
            cache,
            trajectories,
            readout_index_in_es);

        const int ndim = static_cast<int>(acq.trajectory_dimensions());
        ASSERT_EQ(ndim, 3) << "readout " << t << " came back with " << ndim
                           << " dimensions; a wave-encoded slab traverses all three";
        const float *k = acq.getTrajPtr();

        const std::vector<double> moved[2] = {
            excursion(k, ndim, pt.num_samples, 1),
            excursion(k, ndim, pt.num_samples, 2)};
        const bool sweeps = peak(moved[0]) > 0.0 || peak(moved[1]) > 0.0;
        const bool calibration = (cache.table[t].flags & kParallelCalibration) != 0;

        /* Which pass a readout belongs to is a flag, and what it played is a
         * trajectory. A reconstruction reads the first and trusts the second,
         * so the two have to say the same thing. */
        EXPECT_EQ(sweeps, !calibration)
            << "readout " << t << (sweeps ? " swept" : " stood still") << " but is flagged "
            << (calibration ? "calibration" : "imaging");

        if (!sweeps)
        {
            ++flat;
            continue;
        }
        ++waved;

        /* One corkscrew for the whole scan: the reconstruction reads the
         * point-spread function off a single readout and applies it to every
         * line, so every line has to have played the same one. */
        for (int axis = 0; axis < 2; ++axis)
        {
            if (corkscrew[axis].empty())
            {
                corkscrew[axis] = moved[axis];
                continue;
            }
            const double tolerance = 1e-3 * peak(corkscrew[axis]);
            for (int s = 0; s < pt.num_samples; ++s)
                ASSERT_NEAR(
                    moved[axis][static_cast<size_t>(s)],
                    corkscrew[axis][static_cast<size_t>(s)],
                    tolerance)
                    << "readout " << t << " played a different corkscrew on axis " << (axis + 1)
                    << " at sample " << s;
        }
    }

    ASSERT_GT(waved, 0u) << "no readout carried a corkscrew";
    ASSERT_GT(flat, 0u) << "the wave-free autocalibration rectangle is not in this fixture";
    EXPECT_GT(peak(corkscrew[0]), 0.0);
    EXPECT_GT(peak(corkscrew[1]), 0.0);
}

/*
 * A ramp-sampled EPI train advances k at a rate that changes along the
 * readout, so its samples are not on the grid and a reconstruction has to put
 * them back on it. Where they fell is what the acquisition's trajectory says,
 * and a readout that carries none is one a regridder reads as uniform and
 * passes straight through -- so the failure this pins is a silent one.
 *
 * Every sequence of the chain, not only the imaging: the coil calibration and
 * the phase navigator play the same train, and a reconstruction regrids what
 * it calibrates from too.
 */
class SequenceReaderRampSampled : public ::testing::TestWithParam<const char *>
{
};

TEST_P(SequenceReaderRampSampled, EveryReadoutCarriesWhereItsSamplesFell)
{
    const std::string name = GetParam();
    const SequenceCache cache = mrdserver::read_sequence_files(corpus(name + ".seq"));
    const auto trajectories = mrdserver::pre_compute_trajectories(cache);
    ASSERT_FALSE(cache.table.empty());

    std::vector<int> readout_index_in_es(cache.table.size(), -1);
    std::vector<int> seen(cache.encoding_spaces.size(), 0);
    for (size_t t = 0; t < cache.table.size(); ++t)
    {
        const int es = cache.table[t].encoding_space_ref;
        if (es >= 0 && es < static_cast<int>(seen.size()))
            readout_index_in_es[t] = seen[es]++;
    }

    size_t uneven = 0;
    for (size_t t = 0; t < cache.table.size(); ++t)
    {
        const int es = cache.table[t].encoding_space_ref;
        ASSERT_GE(es, 0);
        const auto &pt = trajectories[static_cast<size_t>(es)];
        ASSERT_GT(pt.ndim, 0) << name << ": readout " << t << " is in encoding space " << es
                              << ", which carries no trajectory at all";

        ISMRMRD::Acquisition acq;
        acq.resize(static_cast<uint16_t>(pt.num_samples), 1, 0);
        mrdserver::enrich_ismrmrd_acquisition(
            acq,
            static_cast<int>(t),
            1u,
            0.0f,
            cache,
            trajectories,
            readout_index_in_es);

        const int ndim = static_cast<int>(acq.trajectory_dimensions());
        ASSERT_GT(ndim, 0) << name << ": readout " << t << " carries no trajectory, so a "
                           << "regridder would read it as uniform and pass it through";

        /* The readout axis, sample to sample. Ramp sampling is exactly the
         * step not being constant, so a train whose steps are all equal is
         * one that waited for its plateau. */
        double smallest = 0.0, largest = 0.0;
        for (int s = 1; s < pt.num_samples; ++s)
        {
            const double step = std::fabs(
                static_cast<double>(acq.getTrajPtr()[s * ndim]) -
                static_cast<double>(acq.getTrajPtr()[(s - 1) * ndim]));
            if (s == 1)
                smallest = largest = step;
            smallest = std::min(smallest, step);
            largest = std::max(largest, step);
        }
        if (largest > 0.0 && smallest < 0.99 * largest)
            ++uneven;
    }

    EXPECT_GT(uneven, 0u) << name << " advances k evenly along every readout, so nothing here "
                          << "is ramp sampled and this fixture no longer covers the case";
}

INSTANTIATE_TEST_SUITE_P(Corpus, SequenceReaderRampSampled, ::testing::Values("epi_2d", "epi_3d"));

/* ------------------------------------------------------------------ *
 * Gradient-coil coefficients in the header.
 *
 * The correction runs where the reconstruction runs, which is not the
 * scanner, so the coil description travels in the header rather than a path
 * to the file it came from. The syntax is the scanner's own gw_coils.dat,
 * because that is what pulserver.recon.GradientCoefficients.from_file reads.
 * ------------------------------------------------------------------ */

namespace
{
std::string user_parameter(const ISMRMRD::IsmrmrdHeader &hdr, const std::string &name)
{
    if (!hdr.userParameters)
        return {};
    for (const auto &p : hdr.userParameters->userParameterString)
    {
        if (p.name == name)
            return p.value;
    }
    return {};
}

std::map<std::string, double> parse_dat(const std::string &payload)
{
    std::map<std::string, double> values;
    std::istringstream stream(payload);
    std::string key;
    double value = 0.0;
    while (stream >> key >> value)
        values[key] = value;
    return values;
}
} // namespace

TEST(GradwarpCoefficients, TheCoilDescriptionTravelsAsTheScannerStatesIt)
{
    std::vector<float> scales(30, 0.0f);
    scales[2] = -1.674470e-4f;  /* SCALEX3 */
    scales[14] = -8.702937e-8f; /* SCALEY5 */
    scales[22] = -1.136898e-4f; /* SCALEZ3 */

    ISMRMRD::IsmrmrdHeader hdr;
    mrdserver::add_gradwarp_coefficients(hdr, 1, scales, 0.0f);

    const auto values = parse_dat(user_parameter(hdr, "gradient_coefficients"));
    ASSERT_EQ(values.size(), 32u) << "the payload must state every coefficient, the "
                                  << "gradwarp type and the delta";
    EXPECT_EQ(values.at("GRADWARPTYPE"), 1.0);
    EXPECT_EQ(values.at("DELTA"), 0.0);
    for (std::size_t i = 0; i < scales.size(); ++i)
    {
        std::ostringstream key;
        key << "SCALE"
            << "XYZ"[i / 10] << (i % 10 + 1);
        EXPECT_FLOAT_EQ(static_cast<float>(values.at(key.str())), scales[i]) << key.str();
    }
}

TEST(GradwarpCoefficients, ACoilDescriptionThatIsNotThirtyValuesIsRefused)
{
    ISMRMRD::IsmrmrdHeader hdr;
    EXPECT_THROW(
        mrdserver::add_gradwarp_coefficients(hdr, 1, std::vector<float>(29, 0.0f), 0.0f),
        std::invalid_argument);
}

TEST(GradwarpCoefficients, ATypeThisBuildCannotCorrectIsCarriedRatherThanDropped)
{
    /* Refusing here would leave the reconstruction with no way to say why it
     * did not unwarp. It refuses, on the far side, where the correction is. */
    ISMRMRD::IsmrmrdHeader hdr;
    mrdserver::add_gradwarp_coefficients(hdr, 3, std::vector<float>(30, 0.0f), 0.25f);

    const auto values = parse_dat(user_parameter(hdr, "gradient_coefficients"));
    EXPECT_EQ(values.at("GRADWARPTYPE"), 3.0);
    EXPECT_FLOAT_EQ(static_cast<float>(values.at("DELTA")), 0.25f);
}

TEST(SequenceParameters, EveryEchoTimeReachesTheHeader)
{
    /* MRD's TE is a vector because a scan reads several echoes, and their
     * spacing is what a field map is derived from: a header that states only
     * the shortest cannot be corrected against. */
    const SequenceCache cache = mrdserver::read_sequence_files(corpus("gre_multiecho_2d.seq"));

    ISMRMRD::IsmrmrdHeader hdr;
    mrdserver::enrich_ismrmrd_header(hdr, cache);

    ASSERT_TRUE(hdr.sequenceParameters.is_present());
    const auto &te = hdr.sequenceParameters->TE;
    ASSERT_TRUE(te.is_present());
    EXPECT_EQ(te->size(), 3u);
    for (size_t i = 1; i < te->size(); ++i)
        EXPECT_LT((*te)[i - 1], (*te)[i]) << i;

    /* TR reduces to one representative, and TE[0] is still the shortest. */
    ASSERT_TRUE(hdr.sequenceParameters->TR.is_present());
    EXPECT_EQ(hdr.sequenceParameters->TR->size(), 1u);
}
