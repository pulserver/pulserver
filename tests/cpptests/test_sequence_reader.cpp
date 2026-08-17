// test_sequence_reader.cpp
//
// The recon-side reader, end to end over the corpus: read_sequence_files
// populates a SequenceCache from the seqfile chain and the format-agnostic
// consumers compose trajectories from it. The invariants pinned here were
// proven equivalent to the retired cache pipeline product-by-product before
// that pipeline was deleted; a regression against any of them is a change
// in what a reconstruction receives.

#include "sequence_file_reader.h"

#include "pulseq/sequence_file.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using mrdserver::SequenceCache;

namespace
{

    std::string corpus(const std::string& name)
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

    /** The positions carrying `flag`. */
    std::set<size_t> flagged(const SequenceCache& cache, uint64_t flag)
    {
        std::set<size_t> out;
        for (size_t i = 0; i < cache.table.size(); ++i)
            if (cache.table[i].flags & flag)
                out.insert(i);
        return out;
    }

}  // namespace

class SequenceReader : public ::testing::TestWithParam<const char*>
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
    for (const auto& entry : cache.table)
    {
        last_count += (entry.flags & kLastFlag) ? 1 : 0;
        ASSERT_GE(entry.encoding_space_ref, 0);
        ASSERT_LT(entry.encoding_space_ref,
                  static_cast<int>(cache.encoding_spaces.size()));
        // Every shot id resolves, and NAV routes to the navigator space.
        for (const int shot : {entry.kx_shot_id, entry.ky_shot_id, entry.kz_shot_id})
            ASSERT_LT(shot, static_cast<int>(cache.kshots.size()));
        if (entry.flags & kNavFlag)
            EXPECT_EQ(cache.encoding_spaces[entry.encoding_space_ref].geometry_tag, 1);
    }
    EXPECT_EQ(last_count, 1);
    EXPECT_TRUE((cache.table.back().flags & kLastFlag) != 0);

    // Label limits bound what the table holds.
    for (const auto& entry : cache.table)
    {
        const auto& limits =
            cache.encoding_spaces[entry.encoding_space_ref].label_limits;
        EXPECT_GE(entry.lin, limits.lin.min);
        EXPECT_LE(entry.lin, limits.lin.max);
        EXPECT_GE(entry.slc, limits.slc.min);
        EXPECT_LE(entry.slc, limits.slc.max);
    }

    // The corpus files declare TRSize, so the description is derived: one
    // row per structural-TR block, roles either one unique centre or a tie.
    ASSERT_TRUE(cache.has_seq_desc);
    for (const auto& sd : cache.seq_descs)
    {
        ASSERT_FALSE(sd.events.empty());
        EXPECT_GT(sd.tr_duration_us, 0.0f);
        int echo_centers = 0;
        for (const auto& row : sd.events)
        {
            if (row.type == mrdserver::SEQ_EVENT_ADC &&
                row.adc_role() == mrdserver::ADC_ROLE_ECHO_CENTER)
                ++echo_centers;
        }
        EXPECT_LE(echo_centers, 1);
        for (const auto& def : sd.rf_defs)
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
        const auto& a = resident[es];
        const auto& b = deferred[es];
        ASSERT_EQ(b.ndim, a.ndim) << "es " << es;
        if (a.ndim == 0)
            continue;
        ASSERT_EQ(b.num_samples, a.num_samples);
        ASSERT_EQ(b.num_readouts, a.num_readouts);

        std::vector<float> row_a(static_cast<size_t>(a.ndim) * a.num_samples);
        std::vector<float> row_b(row_a.size());
        for (int r = 0; r < a.num_readouts; ++r)
        {
            ASSERT_TRUE(mrdserver::materialize_readout(
                cache, a, static_cast<int>(es), r, row_a.data()));
            ASSERT_TRUE(mrdserver::materialize_readout(
                cache, b, static_cast<int>(es), r, row_b.data()));
            for (size_t s = 0; s < row_a.size(); ++s)
                ASSERT_EQ(row_b[s], row_a[s])
                    << "es " << es << " readout " << r << " sample " << s;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Corpus, SequenceReader,
                         ::testing::Values("gre_2d", "gre_2d_3sl", "se_2d", "fse_2d",
                                           "bssfp_2d", "gre_radial_2d", "gre_spiral_2d",
                                           "gre_stack_of_stars_3d", "se_propeller_2d",
                                           "mprage_stack_of_spirals_3d", "zte_3d"));

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
    for (const auto& entry : first_of)
        expect_first.insert(entry.second);
    for (const auto& entry : last_of)
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
    for (const auto& entry : cache.table)
        pairs.insert({entry.slc, entry.lin});
    ASSERT_EQ(pairs.size(), cache.table.size());

    EXPECT_EQ(flagged(cache, kLastInLine).size(), cache.table.size());
    EXPECT_EQ(flagged(cache, kFirstInLine).size(), cache.table.size());
}

TEST(SequenceReaderBoundaries, ACounterTheScanNeverWritesIsNeverBounded)
{
    // A 2D scan has no partition axis, so nothing closes one.
    const SequenceCache cache = mrdserver::read_sequence_files(corpus("gre_2d.seq"));

    for (const auto& entry : cache.table)
        ASSERT_EQ(entry.par, 0);
    EXPECT_TRUE(flagged(cache, kLastInPartition).empty());
    EXPECT_FALSE(flagged(cache, kLastInLine).empty());
}

TEST(SequenceReaderBoundaries, AnAverageIsAFrameCounterLikeTheOthers)
{
    // AVG carries the NEX replay index and nothing else. A replay is a whole
    // image, so its boundary is read within the other frame counters: an
    // average closes once per slice, where that slice's share of the replay
    // ends -- and a scan of one average closes none, because there was no
    // replay to bound.
    const std::string path = corpus("gre_2d_3sl.seq");
    const SequenceCache once = mrdserver::read_sequence_files(path, 1);
    const SequenceCache thrice = mrdserver::read_sequence_files(path, 3);

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
    for (const auto& entry : first_of)
        expect_first.insert(entry.second);
    for (const auto& entry : last_of)
        expect_last.insert(entry.second);

    EXPECT_EQ(flagged(thrice, kFirstInAverage), expect_first);
    EXPECT_EQ(flagged(thrice, kLastInAverage), expect_last);
}

TEST(SequenceReaderChain, TheEpiChainBecomesOneCacheOfSubsequences)
{
    const SequenceCache cache = mrdserver::read_sequence_files(corpus("epi_2d.seq"));

    ASSERT_GE(cache.definitions_by_subseq.size(), 2u);
    std::set<int> subseqs;
    for (const auto& es : cache.encoding_spaces)
        subseqs.insert(es.subseq_idx);
    EXPECT_EQ(subseqs.size(), cache.definitions_by_subseq.size());

    // The navigator labels split an encoding space off somewhere.
    bool has_nav_space = false;
    for (const auto& es : cache.encoding_spaces)
        has_nav_space = has_nav_space || es.geometry_tag == 1;
    EXPECT_TRUE(has_nav_space);
}

TEST(SequenceReaderAverages, TheAvgCounterCarriesTheReplayIndex)
{
    const std::string path = corpus("gre_2d.seq");
    const SequenceCache once = mrdserver::read_sequence_files(path, 1);
    const SequenceCache thrice = mrdserver::read_sequence_files(path, 3);

    ASSERT_EQ(thrice.table.size(), once.table.size() * 3);
    for (size_t i = 0; i < thrice.table.size(); ++i)
    {
        const auto& entry = thrice.table[i];
        EXPECT_EQ(entry.avg, static_cast<int>(i / once.table.size()));
        // REP passes through from the file's labels, untouched by NEX.
        EXPECT_EQ(entry.rep, once.table[i % once.table.size()].rep);
    }
    EXPECT_EQ(thrice.table.back().flags & kLastFlag, kLastFlag);
    EXPECT_EQ(once.table.back().flags & kLastFlag, kLastFlag);
}
