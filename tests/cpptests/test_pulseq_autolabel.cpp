/**
 * Tests for pulseq::auto_label -- the encoding counters a `.seq` does not
 * carry, recovered from where its readouts sit in k-space.
 *
 * There is no second implementation to check against here (`autoLabel.m` runs
 * under MATLAB and is a validation run, not a dependency), so the assertions
 * are against properties the sequences state for themselves: a fully sampled
 * Cartesian scan visits every line once, its centre line is at N/2, an EPI's
 * navigators repeat one line, and a scan whose readouts do not share a
 * direction has no Cartesian counters at all.
 *
 * The round trip is the load-bearing one: labels are read back off the
 * sequence by walking its extension chains, which is what a reconstruction
 * does, and a `SET` written only where the value changes has to reproduce the
 * whole evolution.
 */

#include <gtest/gtest.h>

#include "pulseq/autolabel.hpp"
#include "pulseq/read.hpp"
#include "pulseq/sequence.hpp"

#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace
{
    const std::string kData = std::string(PULSEQ_FIXTURES_DIR) + "/";

    pulseq::Sequence load(const std::string& stem)
    {
        return pulseq::read_file(kData + stem + ".seq");
    }

    /**
     * Replay the sequence's own LABELSET extensions, the way a reader does.
     *
     * Values persist from the block that sets them until something sets them
     * again -- that is what makes writing only the changes lossless -- so this
     * carries a running state across every block and samples it at each ADC.
     */
    std::map<std::string, std::vector<int>> replay_labels(const pulseq::Sequence& seq)
    {
        const int labelset = seq.find_extension_type_id("LABELSET");
        std::map<std::string, std::vector<int>> out;
        std::map<std::string, int> state;

        /* Every counter the file mentions starts at zero, including one whose
         * first SET is partway through -- a scan whose LIN begins at 0 writes
         * nothing on its first readout, and the value there is still 0. */
        for (int id = 1; id <= seq.label_set_library().size(); ++id)
            state[seq.label_name(seq.label_set_library().row(id)[1])] = 0;

        for (int b = 1; b <= seq.num_blocks(); ++b)
        {
            const pulseq::Block block = seq.get_block(b);
            for (int32_t link = block.ext; link != 0;)
            {
                const int32_t* row = seq.extensions_library().row(link);
                if (labelset != 0 && row[0] == labelset && row[1] != 0)
                {
                    const int32_t* label = seq.label_set_library().row(row[1]);
                    state[seq.label_name(label[1])] = label[0];
                }
                link = row[2];
            }
            if (block.adc == 0)
                continue;
            for (std::map<std::string, int>::const_iterator it = state.begin();
                 it != state.end(); ++it)
                out[it->first].push_back(it->second);
        }
        return out;
    }

}  // namespace

/*
 * A fully sampled Cartesian scan visits each line once, and the centre is N/2.
 *
 * gre_2d_1sl_1avg has eight phase-encode steps acquired in order, so the
 * counter is the identity and `kSpaceCenterLine` is 4 -- the same convention
 * the echo sample follows, and the reason the core has a tie rule at all.
 */
TEST(PulseqAutoLabel, CartesianLinesAreConsecutiveAndCentredAtHalfN)
{
    pulseq::Sequence seq = load("gre_2d_1sl_1avg");
    const pulseq::AutoLabelResult r = pulseq::auto_label(seq, {}, false);

    ASSERT_EQ(r.labels.lin.size(), 8u);
    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(r.labels.lin[static_cast<size_t>(i)], i);

    EXPECT_TRUE(r.labels.slc.empty()) << "a single-slice scan has no slice counter";
    EXPECT_TRUE(r.labels.rev.empty()) << "a single-polarity scan has no REV";
    EXPECT_TRUE(r.labels.rep.empty()) << "nothing is acquired twice";

    ASSERT_TRUE(r.aux.has_center_line);
    EXPECT_EQ(r.aux.center_line, 4);
    ASSERT_TRUE(r.aux.has_center_sample);
    EXPECT_EQ(r.aux.center_sample, 32);
}

/*
 * Slices are counted by position, and dummy TRs do not become slices.
 *
 * gre_2d_3sl_3avg plays five dummy excitations at 0 mm before acquiring, and
 * then acquires -5, 0, +5.  `autoLabel.m` uniques over every excitation, so
 * those dummies claim slice 0 and the first acquired slice comes out as 1.
 * Here the table is built from what was acquired, so every entry of
 * SlicePositions is a slice that exists -- which is what a reconstruction
 * allocating one slot per position depends on.
 *
 * This fixture acquires in ascending order, so it cannot tell position
 * ordering from acquisition ordering; the test below it does that.
 */
TEST(PulseqAutoLabel, SliceCountersIgnoreDummyExcitations)
{
    pulseq::Sequence seq = load("gre_2d_3sl_3avg");
    const pulseq::AutoLabelResult r = pulseq::auto_label(seq, {}, false);

    ASSERT_EQ(r.aux.slice_positions.size(), 3u);
    EXPECT_NEAR(r.aux.slice_positions[0], -5e-3, 1e-9);
    EXPECT_NEAR(r.aux.slice_positions[1], 0.0, 1e-9);
    EXPECT_NEAR(r.aux.slice_positions[2], +5e-3, 1e-9);

    ASSERT_EQ(r.labels.slc.size(), 24u);
    EXPECT_EQ(r.labels.slc[0], 0) << "the first acquisition must be slice 0";
    int highest = 0;
    for (size_t i = 0; i < r.labels.slc.size(); ++i)
    {
        EXPECT_GE(r.labels.slc[i], 0);
        EXPECT_LT(r.labels.slc[i], 3);
        highest = std::max(highest, r.labels.slc[i]);
    }
    EXPECT_EQ(highest, 2) << "counters must be contiguous over the positions";

    /* Slice-inner, line-outer: three acquisitions of a line, then the next. */
    ASSERT_EQ(r.labels.lin.size(), 24u);
    for (int i = 0; i < 24; ++i)
        EXPECT_EQ(r.labels.lin[static_cast<size_t>(i)], i / 3);
}

/*
 * SLC says where a slice is, not when it was acquired.
 *
 * Every fixture in the repository acquires its slices in ascending order, so
 * none of them can tell the two apart.  This one cannot be mistaken: five
 * slices at unevenly spaced positions, acquired interleaved.  Prescribed at
 * -10, -3, 0, +5 and +12 mm and played in the order 0, 2, 4, 1, 3, so the
 * acquisition visits -10, 0, +12, -3, +5.
 *
 * Numbering by arrival would call those 0, 1, 2, 3, 4 and hand a
 * reconstruction a stack shuffled into acquisition order; the slice a
 * radiologist sees between -10 and +12 mm would be the one acquired second.
 * Numbering by position gives 0, 2, 4, 1, 3, which is where they are.
 *
 * The uneven spacing is deliberate too: nothing here may assume a constant
 * step, so the positions are ranked rather than divided by one.
 */
TEST(PulseqAutoLabel, SliceCountersRankByPositionNotByArrival)
{
    /* Ascending, so the index into this array *is* the expected SLC. */
    const double kPositions[5] = {-10e-3, -3e-3, 0.0, +5e-3, +12e-3};
    const int kOrder[5] = {0, 2, 4, 1, 3};
    const int kSlices = 5;
    const int kSamples = 16;
    const double kReadout = 1e5;    /* Hz/m */
    const double kSliceSelect = 2e5; /* Hz/m */

    pulseq::Sequence seq;
    seq.set_rasters(1e-6, 10e-6, 100e-9, 10e-6);

    std::vector<double> magnitude(100, 1.0);
    const int mag_shape = seq.register_raw_shape(magnitude.data(), 100);

    const double slice_grad[pulseq::TRAP_WIDTH] = {kSliceSelect, 100e-6, 100e-6, 100e-6, 0.0};
    const double read[pulseq::TRAP_WIDTH] = {kReadout, 100e-6, 200e-6, 100e-6, 0.0};
    const double prewind[pulseq::TRAP_WIDTH] = {-1.1 * kReadout, 100e-6, 100e-6, 100e-6, 0.0};
    const int gz_slice = seq.register_trap(slice_grad);
    const int gx_read = seq.register_trap(read);
    const int gx_prewind = seq.register_trap(prewind);

    const double adc[pulseq::ADC_WIDTH] = {static_cast<double>(kSamples), 10e-6, 100e-6,
                                           0.0, 0.0, 0.0, 0.0, 0.0};
    const int adc_id = seq.register_adc(adc);

    for (int visit = 0; visit < kSlices; ++visit)
    {
        const double position = kPositions[kOrder[visit]];

        /* What selects a slice: the frequency the gradient puts it at. */
        double rf[pulseq::RF_WIDTH] = {1000.0, static_cast<double>(mag_shape), 0.0, 0.0,
                                       50e-6,  50e-6,                         0.0, 0.0,
                                       0.0,    0.0};
        rf[8] = kSliceSelect * position;  /* freq */

        pulseq::Block excite;
        excite.rf = seq.register_rf(rf, 'e');
        excite.gz = gz_slice;
        excite.duration = 300e-6;
        seq.add_block(excite);

        pulseq::Block encode;
        encode.gx = gx_prewind;
        encode.duration = 300e-6;
        seq.add_block(encode);

        pulseq::Block readout;
        readout.gx = gx_read;
        readout.adc = adc_id;
        readout.duration = 400e-6;
        seq.add_block(readout);
    }

    const pulseq::AutoLabelResult r = pulseq::auto_label(seq, {}, false);
    ASSERT_EQ(r.num_readouts, kSlices);

    ASSERT_EQ(r.aux.slice_positions.size(), static_cast<size_t>(kSlices));
    for (int s = 0; s < kSlices; ++s)
        EXPECT_NEAR(r.aux.slice_positions[static_cast<size_t>(s)], kPositions[s], 1e-9)
            << "SlicePositions[" << s << "]";

    ASSERT_EQ(r.labels.slc.size(), static_cast<size_t>(kSlices));
    for (int visit = 0; visit < kSlices; ++visit)
    {
        EXPECT_EQ(r.labels.slc[static_cast<size_t>(visit)], kOrder[visit])
            << "acquisition " << visit << " is the slice at "
            << kPositions[kOrder[visit]] * 1e3 << " mm";
    }

    /* And SlicePositions[SLC] really is where that acquisition was: the two
     * halves have to agree or neither is usable. */
    for (int visit = 0; visit < kSlices; ++visit)
        EXPECT_NEAR(
            r.aux.slice_positions[static_cast<size_t>(r.labels.slc[static_cast<size_t>(visit)])],
            kPositions[kOrder[visit]], 1e-9);
}

/*
 * An EPI's navigators repeat one line; its train alternates polarity.
 *
 * The three leading readouts carry no phase encoding, so they land on the same
 * line and are distinguished only by REP -- exactly what they are for.  REV
 * then alternates through the train, and the centre line is at N/2 like any
 * other Cartesian scan.
 */
TEST(PulseqAutoLabel, EpiNavigatorsRepeatOneLineAndThePolarityAlternates)
{
    pulseq::Sequence seq = load("epi_2d_1sl_1avg");
    const pulseq::AutoLabelResult r = pulseq::auto_label(seq, {}, false);

    ASSERT_GE(r.labels.lin.size(), 4u);
    ASSERT_EQ(r.labels.rep.size(), r.labels.lin.size());
    ASSERT_EQ(r.labels.rev.size(), r.labels.lin.size());

    EXPECT_EQ(r.labels.lin[0], r.labels.lin[1]);
    EXPECT_EQ(r.labels.lin[1], r.labels.lin[2]);
    EXPECT_EQ(r.labels.rep[0], 0);
    EXPECT_EQ(r.labels.rep[1], 1);
    EXPECT_EQ(r.labels.rep[2], 2);

    for (size_t i = 4; i < r.labels.rev.size(); ++i)
        EXPECT_NE(r.labels.rev[i], r.labels.rev[i - 1]) << "polarity at readout " << i;

    ASSERT_TRUE(r.aux.has_center_line);
    EXPECT_EQ(r.aux.center_line, 64);
}

/*
 * A bipolar train's echo indices differ by one, and the reported one is the
 * mirrored frame.
 *
 * The two polarities pass the same point in k from opposite ends, so their
 * nearest samples are different indices: 64 and 63 of 128.  A reconstruction
 * mirrors the reverse lines -- that is what REV asks it to do -- and both then
 * sit at 64, which is what a single `kSpaceCenterSample` for the scan has to
 * mean.  MATLAB quotes 63 here, the raw index of the reverse navigator it
 * happens to encounter first.
 */
TEST(PulseqAutoLabel, TheEpiEchoIndexIsQuotedAfterMirroring)
{
    pulseq::Sequence seq = load("epi_2d_1sl_1avg");

    pulseq::KSpaceOptions ko;
    ko.derive_center_sample = true;
    ko.materialize_samples = false;
    const pulseq::KSpace ks = pulseq::calculate_kspace(seq, ko);

    const pulseq::AutoLabelResult r = pulseq::detect_labels(ks, seq);
    ASSERT_EQ(r.labels.rev.size(), ks.readouts.size());

    bool seen_forward = false;
    bool seen_reverse = false;
    for (size_t i = 0; i < ks.readouts.size(); ++i)
    {
        const pulseq::Readout& ro = ks.readouts[i];
        ASSERT_GT(ro.num_samples, 0);
        const bool reversed = r.labels.rev[i] != 0;
        (reversed ? seen_reverse : seen_forward) = true;

        /* Each polarity finds it on its own sample... */
        EXPECT_EQ(ro.center_sample, reversed ? 63 : 64) << "readout " << i;
        /* ...and mirroring collapses them onto one. */
        EXPECT_EQ(reversed ? ro.num_samples - 1 - ro.center_sample : ro.center_sample, 64);
    }
    ASSERT_TRUE(seen_forward && seen_reverse) << "the fixture is not bipolar";

    ASSERT_TRUE(r.aux.has_center_sample);
    EXPECT_EQ(r.aux.center_sample, 64);
}

/*
 * Dimensions the trajectory cannot see are named by the caller, and the count
 * splits between them.
 *
 * The EPI's three leading navigators visit one line three times, so the
 * trajectory can say they are three visits and nothing more.  What they *are*
 * -- three echoes, three saturation states, three frames -- is not written
 * anywhere in k, and the only honest source is whoever built the sequence.
 */
TEST(PulseqAutoLabel, DeclaredRepeatDimensionsSplitTheRepeatCount)
{
    pulseq::Sequence seq = load("epi_2d_1sl_1avg");

    /* Undeclared: one counter running 0, 1, 2, as MATLAB produces. */
    const pulseq::AutoLabelResult plain = pulseq::auto_label(seq, {}, false);
    ASSERT_GE(plain.labels.rep.size(), 3u);
    EXPECT_EQ(plain.labels.rep[0], 0);
    EXPECT_EQ(plain.labels.rep[1], 1);
    EXPECT_EQ(plain.labels.rep[2], 2);

    /* Named: the same numbers, under the name they were given, and REP is no
     * longer standing in for something it is not. */
    {
        pulseq::AutoLabelOptions options;
        options.repeat_dims.push_back({"SET", 4});
        const pulseq::AutoLabelResult r = pulseq::auto_label(seq, options, false);
        EXPECT_TRUE(r.labels.rep.empty());
        ASSERT_EQ(r.labels.named.size(), 1u);
        EXPECT_EQ(r.labels.named[0].first, "SET");
        EXPECT_EQ(r.labels.named[0].second, plain.labels.rep);
    }

    /* Two of them, fastest first: three visits split as (SET, ECO) =
     * (0,0), (1,0), (0,1) -- mixed radix, exactly as declared. */
    {
        pulseq::AutoLabelOptions options;
        options.repeat_dims.push_back({"SET", 2});
        options.repeat_dims.push_back({"ECO", 0}); /* open-ended */
        const pulseq::AutoLabelResult r = pulseq::auto_label(seq, options, false);
        ASSERT_EQ(r.labels.named.size(), 2u);
        EXPECT_EQ(r.labels.named[0].first, "SET");
        EXPECT_EQ(r.labels.named[1].first, "ECO");
        EXPECT_EQ(r.labels.named[0].second[0], 0);
        EXPECT_EQ(r.labels.named[0].second[1], 1);
        EXPECT_EQ(r.labels.named[0].second[2], 0);
        EXPECT_EQ(r.labels.named[1].second[0], 0);
        EXPECT_EQ(r.labels.named[1].second[1], 0);
        EXPECT_EQ(r.labels.named[1].second[2], 1);
    }

    /* And they end up on the sequence under those names. */
    {
        pulseq::Sequence writable = load("epi_2d_1sl_1avg");
        pulseq::AutoLabelOptions options;
        options.repeat_dims.push_back({"SET", 4});
        pulseq::auto_label(writable, options, true);
        const std::map<std::string, std::vector<int>> replayed = replay_labels(writable);
        ASSERT_TRUE(replayed.count("SET") == 1) << "SET was never written";
        EXPECT_EQ(replayed.find("SET")->second, plain.labels.rep);
        EXPECT_EQ(replayed.count("REP"), 0u) << "REP was written as well as SET";
    }
}

/*
 * A declaration that cannot hold the repeats found is an error.
 *
 * Two declared slots and three acquisitions of one k-space position means
 * either the scan is not what was declared or a dimension was left out.
 * Wrapping the counter round with a modulo would produce labels that look
 * perfectly ordinary and put two different acquisitions in the same slot,
 * which a reconstruction would then average together.
 */
TEST(PulseqAutoLabel, RepeatDimensionsThatCannotHoldTheScanAreRefused)
{
    pulseq::Sequence seq = load("epi_2d_1sl_1avg");

    pulseq::AutoLabelOptions too_small;
    too_small.repeat_dims.push_back({"SET", 3});
    EXPECT_THROW(pulseq::auto_label(seq, too_small, false), std::runtime_error);

    /* A counter this derives for itself is not available to be redefined. */
    pulseq::AutoLabelOptions taken;
    taken.repeat_dims.push_back({"LIN", 3});
    EXPECT_THROW(pulseq::auto_label(seq, taken, false), std::runtime_error);

    pulseq::AutoLabelOptions twice;
    twice.repeat_dims.push_back({"SET", 1});
    twice.repeat_dims.push_back({"SET", 3});
    EXPECT_THROW(pulseq::auto_label(seq, twice, false), std::runtime_error);

    /* Open-ended anywhere but last: a dimension of unknown size cannot have a
     * faster one nested inside it, because there is no stride to divide by. */
    pulseq::AutoLabelOptions unbounded_first;
    unbounded_first.repeat_dims.push_back({"SET", 0});
    unbounded_first.repeat_dims.push_back({"ECO", 3});
    EXPECT_THROW(pulseq::auto_label(seq, unbounded_first, false), std::runtime_error);

    pulseq::AutoLabelOptions nameless;
    nameless.repeat_dims.push_back({"", 3});
    EXPECT_THROW(pulseq::auto_label(seq, nameless, false), std::runtime_error);
}

/*
 * A sequence whose readouts do not share a direction has no Cartesian
 * counters, and says so.
 *
 * The alternative would be to emit a LIN for a rotating spiral, which is a
 * number with no meaning that a reconstruction would nevertheless index with.
 */
TEST(PulseqAutoLabel, NonCartesianSequencesAreRefused)
{
    pulseq::Sequence seq = load("mprage_noncart_3d_3sl_3avg_userotext1");
    EXPECT_THROW(pulseq::auto_label(seq, {}, false), std::runtime_error);

    /* Also a navigated Cartesian scan: the navigator runs along another axis,
     * so the scan as a whole is not one Cartesian readout. */
    pulseq::Sequence nav = load("mprage_nav_2d_1sl_1avg");
    EXPECT_THROW(pulseq::auto_label(nav, {}, false), std::runtime_error);
}

/*
 * What is written is what is read back.
 *
 * A SET is emitted only where a counter changes, so recovering the evolution
 * means replaying the extensions across the whole scan.  This is the property
 * that makes the sparse encoding safe, and it is checked by reading the
 * sequence the way a reconstruction would rather than by inspecting what was
 * written.
 */
TEST(PulseqAutoLabel, WritingThenReadingReproducesTheEvolution)
{
    for (const char* stem : {"gre_2d_1sl_1avg", "gre_2d_3sl_3avg", "epi_2d_1sl_1avg",
                             "fse_2d_1sl_1avg", "gre_32x32_pe_blip"})
    {
        pulseq::Sequence seq = load(stem);
        const pulseq::AutoLabelResult r = pulseq::auto_label(seq, {}, true);

        const std::map<std::string, std::vector<int>> replayed = replay_labels(seq);
        const std::vector<std::pair<std::string, const std::vector<int>*>> expected =
            r.labels.present();
        ASSERT_FALSE(expected.empty()) << stem;

        for (size_t f = 0; f < expected.size(); ++f)
        {
            const std::map<std::string, std::vector<int>>::const_iterator it =
                replayed.find(expected[f].first);
            ASSERT_NE(it, replayed.end()) << stem << " lost " << expected[f].first;
            ASSERT_EQ(it->second.size(), expected[f].second->size()) << stem;
            for (size_t i = 0; i < it->second.size(); ++i)
                ASSERT_EQ(it->second[i], (*expected[f].second)[i])
                    << stem << " " << expected[f].first << " at acquisition " << i;
        }
    }
}

/*
 * Applying twice leaves the same sequence, not two sets of labels.
 *
 * A block would otherwise end up carrying two LABELSET links for one counter,
 * whose order in the chain would then decide the value.  The library must not
 * grow either: the rows are per distinct value, and a second run has no new
 * values to add.
 */
TEST(PulseqAutoLabel, ApplyingTwiceIsIdempotent)
{
    pulseq::Sequence seq = load("gre_2d_3sl_3avg");
    pulseq::auto_label(seq, {}, true);

    const int extensions = seq.extensions_library().size();
    const int label_rows = seq.label_set_library().size();
    const std::map<std::string, std::vector<int>> once = replay_labels(seq);

    pulseq::auto_label(seq, {}, true);

    EXPECT_EQ(seq.extensions_library().size(), extensions);
    EXPECT_EQ(seq.label_set_library().size(), label_rows);
    EXPECT_EQ(replay_labels(seq), once);
}

/*
 * An extension a block already carried is still there afterwards.
 *
 * Rewriting the block is how the labels get on, and the naive way to do it
 * drops whatever else was in the chain -- a rotation, a trigger.  MATLAB warns
 * about exactly this (`mr:fixmePreviousRotationExtension`); here the chain is
 * rebuilt around what it finds.
 */
TEST(PulseqAutoLabel, ExistingExtensionsOnAReadoutBlockSurvive)
{
    pulseq::Sequence seq = load("gre_2d_1sl_1avg");

    const int rotations = seq.extension_type_id("ROTATIONS");
    const double quaternion[4] = {0.0, 0.0, 0.0, 1.0};
    const int rotation = seq.register_rotation(quaternion);

    /* Put it on the first readout block. */
    const pulseq::AutoLabelResult before = pulseq::auto_label(seq, {}, false);
    ASSERT_FALSE(before.labels.adc_block.empty());
    const int target = before.labels.adc_block[0];
    {
        pulseq::Block block = seq.get_block(target);
        block.ext = seq.chain_extension(rotations, rotation, block.ext);
        seq.set_block(target, block);
    }

    pulseq::auto_label(seq, {}, true);

    bool found = false;
    const pulseq::Block block = seq.get_block(target);
    for (int32_t link = block.ext; link != 0;)
    {
        const int32_t* row = seq.extensions_library().row(link);
        if (row[0] == rotations && row[1] == rotation)
            found = true;
        link = row[2];
    }
    EXPECT_TRUE(found) << "the rotation extension was dropped";
}

/*
 * Detection reads one point per readout, not the samples.
 *
 * That is the performance claim of the whole file, and it is only true if the
 * answer does not depend on `k_adc` being there.  Turning it off must change
 * nothing -- if it ever does, something has started reading the samples.
 */
TEST(PulseqAutoLabel, TheAnswerDoesNotDependOnMaterialisedSamples)
{
    for (const char* stem : {"gre_2d_3sl_3avg", "epi_2d_1sl_1avg", "fse_2d_1sl_1avg"})
    {
        pulseq::Sequence dense = load(stem);
        pulseq::KSpaceOptions with;
        with.materialize_samples = true;
        const pulseq::KSpace ks_dense = pulseq::calculate_kspace(dense, with);
        EXPECT_FALSE(ks_dense.k_adc.empty()) << stem;

        pulseq::Sequence sparse = load(stem);
        pulseq::KSpaceOptions without;
        without.materialize_samples = false;
        const pulseq::KSpace ks_sparse = pulseq::calculate_kspace(sparse, without);
        EXPECT_TRUE(ks_sparse.k_adc.empty()) << stem;
        EXPECT_FALSE(ks_sparse.k_central.empty()) << stem;

        const pulseq::AutoLabelResult a = pulseq::detect_labels(ks_dense, dense);
        const pulseq::AutoLabelResult b = pulseq::detect_labels(ks_sparse, sparse);

        EXPECT_EQ(a.labels.lin, b.labels.lin) << stem;
        EXPECT_EQ(a.labels.par, b.labels.par) << stem;
        EXPECT_EQ(a.labels.slc, b.labels.slc) << stem;
        EXPECT_EQ(a.labels.rev, b.labels.rev) << stem;
        EXPECT_EQ(a.labels.rep, b.labels.rep) << stem;
        EXPECT_EQ(a.aux.center_line, b.aux.center_line) << stem;
        EXPECT_EQ(a.aux.center_sample, b.aux.center_sample) << stem;
    }
}

/*
 * Slice thickness is measured from the pulse, and it agrees with the geometry.
 *
 * gre_2d_3sl_3avg prescribes `FOV 0.22 0.22 0.015` over `NumSlices 3` -- 5 mm
 * a slice, contiguous -- and never records a thickness.  Recovering 5 mm from
 * the RF spectrum over the slice-select amplitude is the whole claim, and the
 * FOV is an independent statement of the answer: it comes from the
 * prescription, not from the pulse.
 *
 * The tolerance is 3%, which is the difference between a -6 dB width and the
 * nominal time-bandwidth product the pulse was designed to, plus the 10 Hz
 * spectral grid.  Measured: 4.950 mm.
 */
TEST(PulseqAutoLabel, SliceThicknessComesFromTheRfSpectrum)
{
    pulseq::Sequence seq = load("gre_2d_3sl_3avg");

    const pulseq::Definition* fov = seq.definition("FOV");
    const pulseq::Definition* slices = seq.definition("NumSlices");
    ASSERT_NE(fov, nullptr);
    ASSERT_NE(slices, nullptr);
    ASSERT_EQ(fov->numbers().size(), 3u);
    ASSERT_EQ(slices->numbers().size(), 1u);
    const double prescribed = fov->numbers()[2] / slices->numbers()[0];

    const pulseq::AutoLabelResult r = pulseq::auto_label(seq, {}, true);

    ASSERT_TRUE(r.aux.has_slice_thickness);
    EXPECT_NEAR(r.aux.slice_thickness, prescribed, 0.03 * prescribed);

    /* Contiguous slices: the gap is the spacing less the thickness, and the
     * spacing here is the thickness, so it is nearly nothing. */
    ASSERT_TRUE(r.aux.has_slice_gap);
    EXPECT_NEAR(r.aux.slice_gap, 0.0, 0.03 * prescribed);

    const pulseq::Definition* thickness = seq.definition("SliceThickness");
    ASSERT_NE(thickness, nullptr);
    EXPECT_EQ(thickness->kind(), pulseq::Definition::Kind::Real);
    ASSERT_EQ(thickness->numbers().size(), 1u);
    EXPECT_DOUBLE_EQ(thickness->numbers()[0], r.aux.slice_thickness);
    EXPECT_NE(seq.definition("SliceGap"), nullptr);
}

/*
 * A 3D slab-selective scan: LIN and PAR, no SLC, and a slab thickness.
 *
 * No fixture in the repository encodes a Cartesian third dimension, so the
 * `PAR` path and the slab case were untested until this.  It is built here
 * rather than read from a file precisely so the answer is known in advance:
 * every number below comes from the prescription, not from a previous run.
 *
 *  - four phase-encode steps inside three partitions, so LIN cycles and PAR
 *    steps once per cycle;
 *  - one slab, so there is no slice counter and no gap -- the axis that
 *    would have carried SLC in 2D is carrying PAR here, which is the
 *    distinction a 3D scan turns on;
 *  - a Hamming-windowed sinc of time-bandwidth 4 over 200 us is 20 kHz wide,
 *    and over a 2e5 Hz/m slab select that is a 100 mm slab.
 */
TEST(PulseqAutoLabel, AThreeDimensionalSlabGivesLinAndParAndASlabThickness)
{
    const int kLines = 4;
    const int kPartitions = 3;
    const int kSamples = 32;
    const double kReadout = 1e5;    /* Hz/m       */
    const double kDwell = 10e-6;    /* s          */
    const double kSlabSelect = 2e5; /* Hz/m       */
    const double kArea = 200e-6;    /* s, per unit amplitude of an encode trap */

    pulseq::Sequence seq;
    seq.set_rasters(1e-6, 10e-6, 100e-9, 10e-6);

    std::vector<double> magnitude(200);
    for (int i = 0; i < 200; ++i)
    {
        const double x = (i - 99.5) / 99.5 * 2.0;
        const double s = (std::fabs(x) < 1e-12) ? 1.0 : std::sin(M_PI * x) / (M_PI * x);
        magnitude[static_cast<size_t>(i)] = s * (0.54 + 0.46 * std::cos(M_PI * x / 2.0));
    }
    const int mag_shape = seq.register_raw_shape(magnitude.data(), 200);

    /* amplitude, mag_shape, phase_shape, time_shape, center, delay, ... */
    const double rf[pulseq::RF_WIDTH] = {1000.0, static_cast<double>(mag_shape), 0.0, 0.0,
                                         100e-6, 100e-6, 0.0, 0.0, 0.0, 0.0};
    const int excitation = seq.register_rf(rf, 'e');

    const double slab[pulseq::TRAP_WIDTH] = {kSlabSelect, 100e-6, 200e-6, 100e-6, 0.0};
    const double read[pulseq::TRAP_WIDTH] = {kReadout, 100e-6, 400e-6, 100e-6, 0.0};
    /* Puts the first sample at -(N/2 - 1/2) steps, so the readout straddles
     * the origin and its echo lands at N/2 like any symmetric one. */
    const double prewind[pulseq::TRAP_WIDTH] = {-1.05 * kReadout, 100e-6, 100e-6, 100e-6, 0.0};
    const int gz_slab = seq.register_trap(slab);
    const int gx_read = seq.register_trap(read);
    const int gx_prewind = seq.register_trap(prewind);

    const double adc[pulseq::ADC_WIDTH] = {static_cast<double>(kSamples), kDwell, 100e-6,
                                           0.0, 0.0, 0.0, 0.0, 0.0};
    const int adc_id = seq.register_adc(adc);

    /* The slab gradient still runs after the pulse centre; that moment has to
     * come back off before the readout or every partition sits on top of it. */
    const double rephase = -kSlabSelect * 150e-6 / kArea;

    for (int iz = 0; iz < kPartitions; ++iz)
    {
        for (int iy = 0; iy < kLines; ++iy)
        {
            pulseq::Block excite;
            excite.rf = excitation;
            excite.gz = gz_slab;
            excite.duration = 400e-6;
            seq.add_block(excite);

            /* One 1/m per step on both encoded axes. */
            const double gy[pulseq::TRAP_WIDTH] = {(iy - kLines / 2) / kArea, 100e-6, 100e-6,
                                                   100e-6, 0.0};
            const double gz[pulseq::TRAP_WIDTH] = {
                (iz - kPartitions / 2) / kArea + rephase, 100e-6, 100e-6, 100e-6, 0.0};
            pulseq::Block encode;
            encode.gx = gx_prewind;
            encode.gy = seq.register_trap(gy);
            encode.gz = seq.register_trap(gz);
            encode.duration = 300e-6;
            seq.add_block(encode);

            pulseq::Block readout;
            readout.gx = gx_read;
            readout.adc = adc_id;
            readout.duration = 600e-6;
            seq.add_block(readout);
        }
    }

    const pulseq::AutoLabelResult r = pulseq::auto_label(seq, {}, false);
    ASSERT_EQ(r.num_readouts, kLines * kPartitions);

    ASSERT_EQ(r.labels.lin.size(), static_cast<size_t>(kLines * kPartitions));
    ASSERT_EQ(r.labels.par.size(), r.labels.lin.size());
    for (int i = 0; i < kLines * kPartitions; ++i)
    {
        EXPECT_EQ(r.labels.lin[static_cast<size_t>(i)], i % kLines) << "readout " << i;
        EXPECT_EQ(r.labels.par[static_cast<size_t>(i)], i / kLines) << "readout " << i;
    }

    /* One slab: the counter a 2D scan would put here is carrying PAR. */
    EXPECT_TRUE(r.labels.slc.empty());
    EXPECT_TRUE(r.labels.rep.empty());
    EXPECT_TRUE(r.aux.slice_positions.empty());
    EXPECT_FALSE(r.aux.has_slice_gap);

    EXPECT_EQ(r.aux.center_line, kLines / 2);
    EXPECT_EQ(r.aux.center_partition, kPartitions / 2);
    EXPECT_EQ(r.aux.center_sample, kSamples / 2);

    /* time-bandwidth 4 over 200 us, over the slab-select amplitude. */
    ASSERT_TRUE(r.aux.has_slice_thickness);
    EXPECT_NEAR(r.aux.slice_thickness, (4.0 / 200e-6) / kSlabSelect, 0.03 * 0.1);
}

/*
 * A scan with one slice has a thickness but no gap.
 *
 * The gap needs two slices to be between; reporting a 0 there would be a
 * number a reconstruction could act on, and there is nothing to act on.
 */
TEST(PulseqAutoLabel, ASingleSliceHasNoGap)
{
    pulseq::Sequence seq = load("gre_2d_1sl_1avg");
    const pulseq::AutoLabelResult r = pulseq::auto_label(seq, {}, false);

    EXPECT_TRUE(r.aux.has_slice_thickness);
    EXPECT_FALSE(r.aux.has_slice_gap);
    EXPECT_TRUE(r.aux.slice_positions.empty());
}

/*
 * The counters go into `[DEFINITIONS]` as whole numbers.
 *
 * They are indices, and the text writer formats an integer definition
 * differently from a real one -- a `kSpaceCenterLine` of `4.0` is not what a
 * reader expects to parse.
 */
TEST(PulseqAutoLabel, AuxBecomesIntegerDefinitions)
{
    pulseq::Sequence seq = load("gre_2d_3sl_3avg");
    pulseq::auto_label(seq, {}, true);

    const pulseq::Definition* line = seq.definition("kSpaceCenterLine");
    ASSERT_NE(line, nullptr);
    EXPECT_EQ(line->kind(), pulseq::Definition::Kind::Int);
    ASSERT_EQ(line->numbers().size(), 1u);
    EXPECT_DOUBLE_EQ(line->numbers()[0], 4.0);

    const pulseq::Definition* sample = seq.definition("kSpaceCenterSample");
    ASSERT_NE(sample, nullptr);
    EXPECT_EQ(sample->kind(), pulseq::Definition::Kind::Int);

    const pulseq::Definition* slices = seq.definition("SlicePositions");
    ASSERT_NE(slices, nullptr);
    EXPECT_EQ(slices->kind(), pulseq::Definition::Kind::Real);
    EXPECT_EQ(slices->numbers().size(), 3u);
}
