/**
 * @file test_pulseq_adapter_equivalence.cpp
 * @brief Stage 3 Step 3.4: load a fixture .seq through the ExternalSequence
 * adapter (cxx/pulseq_adapter) and through pulseg's native C reader
 * (pulseg_pulseq_file_read), and assert the resulting collections agree:
 * block count, segment table, and gradient/RF waveform samples.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "ExternalSequence.h"
#include "collection.hpp"
#include "pulseq_adapter.h"

#ifndef PULSEQ_ADAPTER_FIXTURES_DIR
#define PULSEQ_ADAPTER_FIXTURES_DIR "."
#endif

namespace
{

    pulseg::Opts make_opts()
    {
        pulseg::Opts o;
        o.gamma_hz_per_t = 42577478.0f;
        o.b0_t = 3.0f;
        o.max_grad_hz_per_m = 42577478.0f * 0.040f;
        o.max_slew_hz_per_m_per_s = 42577478.0f * 170.0f;
        o.rf_raster_us = 2.0f;
        o.grad_raster_us = 20.0f;
        o.adc_raster_us = 2.0f;
        o.block_raster_us = 20.0f;
        return o;
    }

    std::string fixture_path(const char *name)
    {
        return std::string(PULSEQ_ADAPTER_FIXTURES_DIR) + "/" + name;
    }

} // namespace

TEST(PulseqAdapterEquivalence, FseFixtureMatchesNativeReader)
{
    const std::string path = fixture_path("fse_2d_1sl_3avg.seq");
    pulseg::Opts opts = make_opts();

    ExternalSequence ext;
    ASSERT_TRUE(ext.load(path)) << "ExternalSequence failed to load fixture";

    pulseg::Collection adapted(ext, opts, /*parse_labels=*/true, /*num_averages=*/1, path.c_str());
    pulseg::Collection native(path.c_str(), opts, /*cache_binary=*/false, /*verify_signature=*/false,
                               /*parse_labels=*/true, /*num_averages=*/1);

    // --- Collection-level info ---
    auto ci_a = adapted.collection_info();
    auto ci_n = native.collection_info();
    EXPECT_EQ(ci_a.num_subsequences, ci_n.num_subsequences);
    EXPECT_EQ(ci_a.num_segments, ci_n.num_segments);
    EXPECT_EQ(ci_a.total_readouts, ci_n.total_readouts);
    EXPECT_NEAR(ci_a.total_duration_us, ci_n.total_duration_us, 1e-2);

    // --- Subsequence / TR info ---
    auto si_a = adapted.subseq_info(0);
    auto si_n = native.subseq_info(0);
    EXPECT_EQ(si_a.num_trs, si_n.num_trs);
    EXPECT_EQ(si_a.tr_size, si_n.tr_size);
    EXPECT_EQ(si_a.num_unique_adcs, si_n.num_unique_adcs);
    EXPECT_EQ(si_a.num_unique_rf, si_n.num_unique_rf);
    EXPECT_NEAR(si_a.tr_duration_us, si_n.tr_duration_us, 1e-2);

    // --- Segment table (block count per segment) ---
    ASSERT_EQ(ci_a.num_segments, ci_n.num_segments);
    for (int seg = 0; seg < ci_a.num_segments; ++seg)
    {
        auto seg_a = adapted.segment_info(seg);
        auto seg_n = native.segment_info(seg);
        EXPECT_EQ(seg_a.num_blocks, seg_n.num_blocks) << "segment " << seg;
        EXPECT_EQ(seg_a.duration_us, seg_n.duration_us) << "segment " << seg;
        EXPECT_EQ(seg_a.has_trigger, seg_n.has_trigger) << "segment " << seg;
        // NOTE: is_nav is intentionally not compared here -- on this fixture
        // pulseg_get_segment_info()'s is_nav comes back as an apparently
        // uninitialized value on BOTH the adapter-built and the
        // natively-parsed collection (reproduces without the adapter at
        // all), so it is not something this adapter-equivalence test can
        // meaningfully attribute to the io/convert seam under test here.
    }

    // --- Waveform ("shape") checksum: full-TR gradient + RF samples ---
    auto tr_a = adapted.get_tr_waveforms(0, PULSEG_AMP_ACTUAL, 0);
    auto tr_n = native.get_tr_waveforms(0, PULSEG_AMP_ACTUAL, 0);

    ASSERT_EQ(tr_a.gx.amplitude.size(), tr_n.gx.amplitude.size());
    ASSERT_EQ(tr_a.gy.amplitude.size(), tr_n.gy.amplitude.size());
    ASSERT_EQ(tr_a.gz.amplitude.size(), tr_n.gz.amplitude.size());
    ASSERT_EQ(tr_a.rf_mag.amplitude.size(), tr_n.rf_mag.amplitude.size());
    ASSERT_EQ(tr_a.rf_phase.amplitude.size(), tr_n.rf_phase.amplitude.size());

    for (size_t i = 0; i < tr_a.gx.amplitude.size(); ++i)
        EXPECT_NEAR(tr_a.gx.amplitude[i], tr_n.gx.amplitude[i], 1e-1) << "gx[" << i << "]";
    for (size_t i = 0; i < tr_a.gy.amplitude.size(); ++i)
        EXPECT_NEAR(tr_a.gy.amplitude[i], tr_n.gy.amplitude[i], 1e-1) << "gy[" << i << "]";
    for (size_t i = 0; i < tr_a.gz.amplitude.size(); ++i)
        EXPECT_NEAR(tr_a.gz.amplitude[i], tr_n.gz.amplitude[i], 1e-1) << "gz[" << i << "]";
    for (size_t i = 0; i < tr_a.rf_mag.amplitude.size(); ++i)
        EXPECT_NEAR(tr_a.rf_mag.amplitude[i], tr_n.rf_mag.amplitude[i], 1e-1) << "rf_mag[" << i << "]";
    for (size_t i = 0; i < tr_a.rf_phase.amplitude.size(); ++i)
        EXPECT_NEAR(tr_a.rf_phase.amplitude[i], tr_n.rf_phase.amplitude[i], 1e-3) << "rf_phase[" << i << "]";

    EXPECT_NEAR(tr_a.total_duration_us, tr_n.total_duration_us, 1e-2);
}
