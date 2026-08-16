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
 * gre_32x32_pe_blip visits each of its 32 phase-encode lines exactly once,
 * and `kSpaceCenterLine` is the mid-grid line -- the same convention the
 * echo sample follows, and the reason the core has a tie rule at all.
 */
TEST(PulseqAutoLabel, CartesianLinesAreConsecutiveAndCentredAtHalfN)
{
    pulseq::Sequence seq = load("gre_32x32_pe_blip");
    const pulseq::AutoLabelResult r = pulseq::auto_label(seq, {}, false);

    ASSERT_EQ(r.labels.lin.size(), 32u);
    std::vector<int> seen(32, 0);
    for (int i = 0; i < 32; ++i)
    {
        const int line = r.labels.lin[static_cast<size_t>(i)];
        ASSERT_GE(line, 0);
        ASSERT_LT(line, 32);
        ++seen[static_cast<size_t>(line)];
    }
    for (int line = 0; line < 32; ++line)
        EXPECT_EQ(seen[static_cast<size_t>(line)], 1) << "line " << line;

    EXPECT_TRUE(r.labels.slc.empty()) << "a single-slice scan has no slice counter";
    EXPECT_TRUE(r.labels.rev.empty()) << "a single-polarity scan has no REV";
    EXPECT_TRUE(r.labels.rep.empty()) << "nothing is acquired twice";

    ASSERT_TRUE(r.aux.has_center_line);
    EXPECT_EQ(r.aux.center_line, 16);
    ASSERT_TRUE(r.aux.has_center_sample);
    EXPECT_EQ(r.aux.center_sample, 16);
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
    pulseq::Sequence seq = load_corpus("gre_2d_3sl");
    const pulseq::AutoLabelResult r = pulseq::auto_label(seq, {}, false);

    ASSERT_EQ(r.aux.slice_positions.size(), 3u);
    EXPECT_NEAR(r.aux.slice_positions[0], -5e-3, 1e-7);
    EXPECT_NEAR(r.aux.slice_positions[1], 0.0, 1e-7);
    EXPECT_NEAR(r.aux.slice_positions[2], +5e-3, 1e-7);

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

namespace
{
    /** Positions and visiting order shared by the slice-sorting tests. */
    const double kSortPositions[5] = {-10e-3, -3e-3, 0.0, +5e-3, +12e-3};
    const int kSortOrder[5] = {0, 2, 4, 1, 3};

    /** The same five interleaved slices as the test above, one readout each. */
    pulseq::Sequence interleaved_slices()
    {
        const int kSamples = 16;
        const double kReadout = 1e5;
        const double kSliceSelect = 2e5;

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

        for (int visit = 0; visit < 5; ++visit)
        {
            double rf[pulseq::RF_WIDTH] = {1000.0, static_cast<double>(mag_shape), 0.0, 0.0,
                                           50e-6,  50e-6,                         0.0, 0.0,
                                           0.0,    0.0};
            rf[8] = kSliceSelect * kSortPositions[kSortOrder[visit]];

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
        return seq;
    }
}  // namespace

/*
 * `sortSlices` decides which slice gets which index -- and under every one of
 * the three, `SlicePositions[SLC]` is still where slice SLC is.  That
 * invariant is the point: a reconstruction reading the pair together cannot be
 * wrong, and only the numbering it sees changes.
 *
 * `acquisition` is MATLAB's default and this library's opt-in; the five
 * interleaved slices come back numbered by arrival, and the position table
 * comes back in arrival order to match.
 */
TEST(PulseqAutoLabel, AcquisitionSortingNumbersSlicesByArrival)
{
    pulseq::Sequence seq = interleaved_slices();
    pulseq::AutoLabelOptions options;
    options.sort_slices = pulseq::SliceSorting::Acquisition;
    const pulseq::AutoLabelResult r = pulseq::auto_label(seq, options, false);

    ASSERT_EQ(r.labels.slc.size(), 5u);
    for (int visit = 0; visit < 5; ++visit)
        EXPECT_EQ(r.labels.slc[static_cast<size_t>(visit)], visit) << "visit " << visit;

    ASSERT_EQ(r.aux.slice_positions.size(), 5u);
    for (int visit = 0; visit < 5; ++visit)
        EXPECT_NEAR(r.aux.slice_positions[static_cast<size_t>(visit)],
                    kSortPositions[kSortOrder[visit]], 1e-9);
}

TEST(PulseqAutoLabel, DescendingSortingReversesBothHalvesTogether)
{
    pulseq::Sequence seq = interleaved_slices();
    pulseq::AutoLabelOptions options;
    options.sort_slices = pulseq::SliceSorting::Descending;
    const pulseq::AutoLabelResult r = pulseq::auto_label(seq, options, false);

    ASSERT_EQ(r.aux.slice_positions.size(), 5u);
    for (int s = 0; s < 5; ++s)
        EXPECT_NEAR(r.aux.slice_positions[static_cast<size_t>(s)], kSortPositions[4 - s], 1e-9)
            << "SlicePositions[" << s << "]";

    ASSERT_EQ(r.labels.slc.size(), 5u);
    for (int visit = 0; visit < 5; ++visit)
    {
        EXPECT_EQ(r.labels.slc[static_cast<size_t>(visit)], 4 - kSortOrder[visit]);
        EXPECT_NEAR(r.aux.slice_positions[static_cast<size_t>(
                        r.labels.slc[static_cast<size_t>(visit)])],
                    kSortPositions[kSortOrder[visit]], 1e-9)
            << "the two halves disagree at visit " << visit;
    }
}

TEST(PulseqAutoLabel, TheSliceGapIsGeometryAndDoesNotFollowTheNumbering)
{
    /* Closest spacing is -3 to 0 mm, whichever end the counting starts from. */
    double gap[3] = {0.0, 0.0, 0.0};
    const pulseq::SliceSorting modes[3] = {pulseq::SliceSorting::Ascending,
                                           pulseq::SliceSorting::Descending,
                                           pulseq::SliceSorting::Acquisition};
    for (int m = 0; m < 3; ++m)
    {
        pulseq::Sequence seq = interleaved_slices();
        pulseq::AutoLabelOptions options;
        options.sort_slices = modes[m];
        const pulseq::AutoLabelResult r = pulseq::auto_label(seq, options, false);
        ASSERT_TRUE(r.aux.has_slice_gap) << "mode " << m;
        gap[m] = r.aux.slice_gap;
        EXPECT_GT(gap[m] + r.aux.slice_thickness, 0.0) << "spacing must be positive, mode " << m;
    }
    EXPECT_DOUBLE_EQ(gap[0], gap[1]);
    EXPECT_DOUBLE_EQ(gap[0], gap[2]);
}

/*
 * `mirrorFourier` reverses the encoding directions and leaves the slices
 * where they are.  That asymmetry is the whole content of the option: a
 * `reflect` on all three axes would turn the slice stack over with them.
 */
TEST(PulseqAutoLabel, MirrorFourierTurnsTheEncodingOverButNotTheSlices)
{
    pulseq::Sequence plain_seq = interleaved_slices();
    const pulseq::AutoLabelResult plain = pulseq::auto_label(plain_seq, {}, false);

    pulseq::Sequence mirrored_seq = interleaved_slices();
    pulseq::AutoLabelOptions options;
    options.mirror_fourier = true;
    const pulseq::AutoLabelResult mirrored = pulseq::auto_label(mirrored_seq, options, false);

    ASSERT_EQ(mirrored.labels.slc.size(), plain.labels.slc.size());
    for (size_t i = 0; i < plain.labels.slc.size(); ++i)
        EXPECT_EQ(mirrored.labels.slc[i], plain.labels.slc[i]) << "slice counter moved at " << i;
    ASSERT_EQ(mirrored.aux.slice_positions.size(), plain.aux.slice_positions.size());
    for (size_t s = 0; s < plain.aux.slice_positions.size(); ++s)
        EXPECT_NEAR(mirrored.aux.slice_positions[s], plain.aux.slice_positions[s], 1e-12);

    /* The readout polarity is what did turn over. */
    ASSERT_EQ(mirrored.aux.has_center_sample, plain.aux.has_center_sample);
    if (!plain.labels.rev.empty() && !mirrored.labels.rev.empty())
    {
        ASSERT_EQ(mirrored.labels.rev.size(), plain.labels.rev.size());
        for (size_t i = 0; i < plain.labels.rev.size(); ++i)
            EXPECT_NE(mirrored.labels.rev[i], plain.labels.rev[i]) << "polarity unchanged at " << i;
    }
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
    pulseq::Sequence seq = load_corpus("epi_2d");
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
    EXPECT_EQ(r.aux.center_line, 7);
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
    pulseq::Sequence seq = load_corpus("epi_2d");

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
        EXPECT_EQ(ro.center_sample, reversed ? 15 : 16) << "readout " << i;
        /* ...and mirroring collapses them onto one. */
        EXPECT_EQ(reversed ? ro.num_samples - 1 - ro.center_sample : ro.center_sample, 16);
    }
    ASSERT_TRUE(seen_forward && seen_reverse) << "the fixture is not bipolar";

    ASSERT_TRUE(r.aux.has_center_sample);
    EXPECT_EQ(r.aux.center_sample, 16);
}

/*
 * A single named dimension takes the repeat count, whatever shape it has.
 *
 * The EPI's three leading navigators visit one line three times, and the image
 * line through the centre visits it a fourth. The trajectory can say they are
 * four visits and nothing more; what they *are* is not written anywhere in k,
 * and the only source is whoever built the sequence. One name is always safe
 * here because there is nothing to work out -- it is the number REP would have
 * carried, under a name that means something.
 */
TEST(PulseqAutoLabel, OneNamedDimensionTakesTheRepeatCount)
{
    pulseq::Sequence seq = load_corpus("epi_2d");

    const pulseq::AutoLabelResult plain = pulseq::auto_label(seq, {}, false);
    ASSERT_GE(plain.labels.rep.size(), 3u);
    EXPECT_EQ(plain.labels.rep[0], 0);
    EXPECT_EQ(plain.labels.rep[1], 1);
    EXPECT_EQ(plain.labels.rep[2], 2);

    /* A bare name: no size, nothing inferred, REP renamed. */
    pulseq::AutoLabelOptions options;
    options.repeat_dims.push_back({"SET", 0});
    const pulseq::AutoLabelResult r = pulseq::auto_label(seq, options, false);
    EXPECT_TRUE(r.labels.rep.empty());
    ASSERT_EQ(r.labels.named.size(), 1u);
    EXPECT_EQ(r.labels.named[0].first, "SET");
    EXPECT_EQ(r.labels.named[0].second, plain.labels.rep);

    /* And it reaches the sequence under that name. */
    pulseq::Sequence writable = load_corpus("epi_2d");
    pulseq::auto_label(writable, options, true);
    const std::map<std::string, std::vector<int>> replayed = replay_labels(writable);
    ASSERT_EQ(replayed.count("SET"), 1u) << "SET was never written";
    EXPECT_EQ(replayed.find("SET")->second, plain.labels.rep);
    EXPECT_EQ(replayed.count("REP"), 0u) << "REP was written as well as SET";
}

/*
 * Two named dimensions, with their sizes read out of the acquisition order.
 *
 * Four lines, two echoes inside each TR, three frames over the whole scan --
 * so every line is acquired six times and nothing in k-space distinguishes
 * those six. What *does* distinguish them is when they happened: the two
 * echoes are consecutive acquisitions and the next frame is four lines away.
 * That is the whole signal, and it is enough to say which of the six each one
 * is without the caller counting anything.
 *
 * Built rather than read from a file so the answer is known from the
 * prescription: ECO must cycle 0,1 within a TR and REP must step once per
 * pass over the four lines.
 */
TEST(PulseqAutoLabel, RepeatDimensionSizesAreReadFromTheAcquisitionOrder)
{
    const int kLines = 4;
    const int kEchoes = 2;
    const int kFrames = 3;
    const int kSamples = 16;
    const double kReadout = 1e5;     /* Hz/m */
    const double kSliceSelect = 2e5; /* Hz/m */
    /* Every 100/100/100 us trapezoid integrates to amplitude * 200 us. */
    const double kArea = 200e-6;
    /* The readout is 100/200/100 us at kReadout, so k advances 13 /m between
     * the start of that block and the centre of the ADC window; the prewinder
     * cancels exactly that, putting the echo on sample 8 of 16.  After the
     * readout k has advanced 30 /m in all, so the gradient between two echoes
     * has to take back 30 for the second to repeat the first. */
    const double kPrewindArea = -13.0;
    const double kRewindArea = -30.0;

    pulseq::Sequence seq;
    seq.set_rasters(1e-6, 10e-6, 100e-9, 10e-6);

    std::vector<double> magnitude(100, 1.0);
    const int mag_shape = seq.register_raw_shape(magnitude.data(), 100);
    const double rf[pulseq::RF_WIDTH] = {1000.0, static_cast<double>(mag_shape), 0.0, 0.0,
                                         50e-6,  50e-6,                          0.0, 0.0,
                                         0.0,    0.0};
    const int excitation = seq.register_rf(rf, 'e');

    const double slice_grad[pulseq::TRAP_WIDTH] = {kSliceSelect, 100e-6, 100e-6, 100e-6, 0.0};
    const double read[pulseq::TRAP_WIDTH] = {kReadout, 100e-6, 200e-6, 100e-6, 0.0};
    const double prewind[pulseq::TRAP_WIDTH] = {kPrewindArea / kArea, 100e-6, 100e-6, 100e-6,
                                                0.0};
    /* Between echoes: take the readout back, so both echoes of a TR sit on
     * the same k-space position and are indistinguishable by trajectory. */
    const double rewind[pulseq::TRAP_WIDTH] = {kRewindArea / kArea, 100e-6, 100e-6, 100e-6, 0.0};
    const int gz_slice = seq.register_trap(slice_grad);
    const int gx_read = seq.register_trap(read);
    const int gx_prewind = seq.register_trap(prewind);
    const int gx_rewind = seq.register_trap(rewind);

    const double adc[pulseq::ADC_WIDTH] = {static_cast<double>(kSamples), 10e-6, 100e-6,
                                           0.0, 0.0, 0.0, 0.0, 0.0};
    const int adc_id = seq.register_adc(adc);

    for (int frame = 0; frame < kFrames; ++frame)
    {
        for (int line = 0; line < kLines; ++line)
        {
            pulseq::Block excite;
            excite.rf = excitation;
            excite.gz = gz_slice;
            excite.duration = 300e-6;
            seq.add_block(excite);

            const double gy[pulseq::TRAP_WIDTH] = {(line - kLines / 2) / kArea, 100e-6, 100e-6,
                                                   100e-6, 0.0};
            pulseq::Block encode;
            encode.gx = gx_prewind;
            encode.gy = seq.register_trap(gy);
            encode.duration = 300e-6;
            seq.add_block(encode);

            for (int echo = 0; echo < kEchoes; ++echo)
            {
                if (echo > 0)
                {
                    pulseq::Block between;
                    between.gx = gx_rewind;
                    between.duration = 300e-6;
                    seq.add_block(between);
                }
                pulseq::Block readout;
                readout.gx = gx_read;
                readout.adc = adc_id;
                readout.duration = 400e-6;
                seq.add_block(readout);
            }
        }
    }

    /* Undeclared: one counter running 0..5, which is arithmetic. */
    const pulseq::AutoLabelResult plain = pulseq::auto_label(seq, {}, false);
    ASSERT_EQ(plain.labels.rep.size(), static_cast<size_t>(kFrames * kLines * kEchoes));
    for (int i = 0; i < kFrames * kLines * kEchoes; ++i)
        EXPECT_EQ(plain.labels.rep[static_cast<size_t>(i)],
                  (i / (kLines * kEchoes)) * kEchoes + (i % kEchoes))
            << "acquisition " << i;

    /* Named, outermost first, with no sizes given at all. */
    pulseq::AutoLabelOptions options;
    options.repeat_dims.push_back({"REP", 0});
    options.repeat_dims.push_back({"ECO", 0});
    const pulseq::AutoLabelResult r = pulseq::auto_label(seq, options, false);

    EXPECT_TRUE(r.labels.rep.empty());
    ASSERT_EQ(r.labels.named.size(), 2u);
    EXPECT_EQ(r.labels.named[0].first, "REP");
    EXPECT_EQ(r.labels.named[1].first, "ECO");

    for (int i = 0; i < kFrames * kLines * kEchoes; ++i)
    {
        EXPECT_EQ(r.labels.named[0].second[static_cast<size_t>(i)], i / (kLines * kEchoes))
            << "REP at acquisition " << i;
        EXPECT_EQ(r.labels.named[1].second[static_cast<size_t>(i)], i % kEchoes)
            << "ECO at acquisition " << i;
    }

    /* Saying the sizes out loud agrees, and saying a wrong one is caught. */
    pulseq::AutoLabelOptions pinned;
    pinned.repeat_dims.push_back({"REP", kFrames});
    pinned.repeat_dims.push_back({"ECO", kEchoes});
    const pulseq::AutoLabelResult same = pulseq::auto_label(seq, pinned, false);
    EXPECT_EQ(same.labels.named[0].second, r.labels.named[0].second);
    EXPECT_EQ(same.labels.named[1].second, r.labels.named[1].second);

    pulseq::AutoLabelOptions wrong;
    wrong.repeat_dims.push_back({"REP", 0});
    wrong.repeat_dims.push_back({"ECO", 3});
    EXPECT_THROW(pulseq::auto_label(seq, wrong, false), std::runtime_error);
}

/*
 * Repeats that are not a rectangle have no nest in them, and say so.
 *
 * The EPI is the case: 127 of its 128 lines are acquired once and one is
 * acquired four times, because three of those are navigators. Splitting that
 * into two dimensions would put two different acquisitions in one slot, and a
 * reconstruction would average them together with every surrounding label
 * looking perfectly ordinary. So it raises, and the caller who knows better
 * than the evidence can still give the sizes outright.
 */
TEST(PulseqAutoLabel, RaggedRepeatsAreRefusedRatherThanSplitAnyway)
{
    pulseq::Sequence seq = load_corpus("epi_2d");

    pulseq::AutoLabelOptions two;
    two.repeat_dims.push_back({"REP", 0});
    two.repeat_dims.push_back({"ECO", 0});
    EXPECT_THROW(pulseq::auto_label(seq, two, false), std::runtime_error);

    /* Declared outright, it is arithmetic again and goes through. */
    pulseq::AutoLabelOptions declared;
    declared.repeat_dims.push_back({"REP", 2});
    declared.repeat_dims.push_back({"ECO", 2});
    const pulseq::AutoLabelResult r = pulseq::auto_label(seq, declared, false);
    ASSERT_EQ(r.labels.named.size(), 2u);
    EXPECT_EQ(r.labels.named[0].second[0], 0);
    EXPECT_EQ(r.labels.named[1].second[0], 0);
    EXPECT_EQ(r.labels.named[0].second[1], 0);
    EXPECT_EQ(r.labels.named[1].second[1], 1);
    EXPECT_EQ(r.labels.named[0].second[2], 1);
    EXPECT_EQ(r.labels.named[1].second[2], 0);
}

/*
 * A declaration that cannot hold the repeats found is an error.
 *
 * Wrapping the counter round with a modulo would produce labels that look
 * perfectly ordinary and put two different acquisitions in the same slot.
 */
TEST(PulseqAutoLabel, RepeatDimensionsThatCannotHoldTheScanAreRefused)
{
    pulseq::Sequence seq = load_corpus("epi_2d");

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

    pulseq::AutoLabelOptions negative;
    negative.repeat_dims.push_back({"SET", -1});
    EXPECT_THROW(pulseq::auto_label(seq, negative, false), std::runtime_error);

    pulseq::AutoLabelOptions nameless;
    nameless.repeat_dims.push_back({"", 3});
    EXPECT_THROW(pulseq::auto_label(seq, nameless, false), std::runtime_error);
}

/*
 * Labels the sequence set for itself survive, and REP can be left to it.
 *
 * The workflow this serves: a design loop stamps the axes only it knows --
 * which contrast, which frame, which saturation state -- and then one
 * auto_label pass fills in the geometric ones around them.
 *
 * Two halves. Labels this never derives (ECO here, and SET, AVG, anything
 * custom) come through an apply pass untouched already, because the extension
 * chain is rebuilt keeping every link that is not one of ours. REP is the
 * exception and the reason `skip` exists: it *is* derived by default, so a
 * loop that separated its own repeats would have that separation overwritten
 * by a bare count of revisits.
 */
TEST(PulseqAutoLabel, LabelsTheSequenceAlreadyCarriesSurviveAnAutoLabelPass)
{
    const int eco = 7;

    /* -- a label auto_label never derives: nothing to ask for -- */
    {
        pulseq::Sequence seq = load_corpus("gre_2d_3sl");
        const int labelset = seq.extension_type_id("LABELSET");
        const int eco_id = seq.label_id("ECO");
        for (int b = 1; b <= seq.num_blocks(); ++b)
        {
            pulseq::Block block = seq.get_block(b);
            if (block.adc == 0)
                continue;
            block.ext = static_cast<int32_t>(seq.chain_extension(
                labelset,
                static_cast<int32_t>(seq.register_label_set(eco, eco_id)),
                block.ext));
            seq.set_block(b, block);
        }

        pulseq::auto_label(seq, {}, true);

        const std::map<std::string, std::vector<int>> replayed = replay_labels(seq);
        ASSERT_EQ(replayed.count("ECO"), 1u) << "the sequence's own label was dropped";
        for (size_t i = 0; i < replayed.find("ECO")->second.size(); ++i)
            EXPECT_EQ(replayed.find("ECO")->second[i], eco);
        /* And the geometric ones were filled in around it. */
        EXPECT_EQ(replayed.count("SLC"), 1u);
        EXPECT_EQ(replayed.count("LIN"), 1u);
    }

    /* -- REP: derived by default, so it has to be handed back -- */
    {
        pulseq::Sequence seq = load_corpus("epi_2d");
        const int labelset = seq.extension_type_id("LABELSET");
        const int rep_id = seq.label_id("REP");
        const int mine = 3;
        for (int b = 1; b <= seq.num_blocks(); ++b)
        {
            pulseq::Block block = seq.get_block(b);
            if (block.adc == 0)
                continue;
            block.ext = static_cast<int32_t>(seq.chain_extension(
                labelset,
                static_cast<int32_t>(seq.register_label_set(mine, rep_id)),
                block.ext));
            seq.set_block(b, block);
        }

        pulseq::AutoLabelOptions options;
        options.skip.push_back("REP");
        const pulseq::AutoLabelResult r = pulseq::auto_label(seq, options, true);

        EXPECT_TRUE(r.labels.rep.empty()) << "REP was derived despite being skipped";
        EXPECT_FALSE(r.labels.lin.empty()) << "the geometric counters still come";

        const std::map<std::string, std::vector<int>> replayed = replay_labels(seq);
        ASSERT_EQ(replayed.count("REP"), 1u);
        for (size_t i = 0; i < replayed.find("REP")->second.size(); ++i)
            EXPECT_EQ(replayed.find("REP")->second[i], mine)
                << "the sequence's own REP was overwritten at acquisition " << i;
    }
}

/*
 * A skip that names something this does not derive is a typo, not a no-op.
 *
 * `skip={"ECO"}` reads as "leave my ECO alone", which is already true and
 * always was; accepting it silently would teach a caller that the list is
 * what protects their labels, and the day they misspell one of the six that
 * are derived they would get no warning at all.
 */
TEST(PulseqAutoLabel, SkippingSomethingNotDerivedIsRefused)
{
    pulseq::Sequence seq = load_corpus("gre_2d_3sl");

    pulseq::AutoLabelOptions not_ours;
    not_ours.skip.push_back("ECO");
    EXPECT_THROW(pulseq::auto_label(seq, not_ours, false), std::runtime_error);

    /* REP cannot be both handed back and named as a dimension to split. */
    pulseq::AutoLabelOptions both;
    both.skip.push_back("REP");
    both.repeat_dims.push_back({"REP", 0});
    EXPECT_THROW(pulseq::auto_label(seq, both, false), std::runtime_error);
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
    pulseq::Sequence seq = load_corpus("gre_stack_of_stars_3d");
    EXPECT_THROW(pulseq::auto_label(seq, {}, false), std::runtime_error);

    /* Also a navigated Cartesian scan: the navigator runs along another axis,
     * so the scan as a whole is not one Cartesian readout. */
    pulseq::Sequence nav = load("gre_nav_2d");
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
    for (const std::string& stem : {kCorpus + "gre_2d", kCorpus + "gre_2d_3sl",
                                    kCorpus + "epi_2d_main", kCorpus + "fse_2d",
                                    kData + "gre_32x32_pe_blip"})
    {
        pulseq::Sequence seq = pulseq::read_file(stem + ".seq");
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
    pulseq::Sequence seq = load_corpus("gre_2d_3sl");
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
    pulseq::Sequence seq = load_corpus("gre_2d");

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
    for (const std::string& stem : {kCorpus + "gre_2d_3sl", kCorpus + "epi_2d_main",
                                    kCorpus + "fse_2d"})
    {
        pulseq::Sequence dense = pulseq::read_file(stem + ".seq");
        pulseq::KSpaceOptions with;
        with.materialize_samples = true;
        const pulseq::KSpace ks_dense = pulseq::calculate_kspace(dense, with);
        EXPECT_FALSE(ks_dense.k_adc.empty()) << stem;

        pulseq::Sequence sparse = pulseq::read_file(stem + ".seq");
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
 * The tolerance is 5%, which covers the difference between a -6 dB width and the
 * spectral grid.  Measured: 4.834 mm against the 5 mm prescription.
 */
TEST(PulseqAutoLabel, SliceThicknessComesFromTheRfSpectrum)
{
    pulseq::Sequence seq = load_corpus("gre_2d_3sl");

    /* Three contiguous 5 mm slices: the z FOV divided by the slice count
     * is the prescription the spectral measurement must land near. */
    const pulseq::Definition* fov = seq.definition("FOV");
    ASSERT_NE(fov, nullptr);
    ASSERT_EQ(fov->numbers().size(), 3u);
    const double prescribed = fov->numbers()[2] / 3.0;

    const pulseq::AutoLabelResult r = pulseq::auto_label(seq, {}, true);

    ASSERT_TRUE(r.aux.has_slice_thickness);
    EXPECT_NEAR(r.aux.slice_thickness, prescribed, 0.05 * prescribed);

    /* Contiguous slices: the gap is the spacing less the thickness, and the
     * spacing here is the thickness, so it is nearly nothing. */
    ASSERT_TRUE(r.aux.has_slice_gap);
    EXPECT_NEAR(r.aux.slice_gap, 0.0, 0.05 * prescribed);

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
    pulseq::Sequence seq = load_corpus("gre_2d");
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
    pulseq::Sequence seq = load_corpus("gre_2d_3sl");
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
