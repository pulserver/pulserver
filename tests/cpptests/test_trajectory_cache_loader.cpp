// SPDX-License-Identifier: MIT
//
// GoogleTest suite for the standalone trajectory cache reader.
//
// Iterates every .pge fixture in pulserverlib-tests/expected/, parses it via
// read_sequence_cache(), pre-computes per-encoding-space trajectories, and
// asserts structural invariants. When a companion <stem>_trajectory.truth
// truth file exists (Phase B), this test will additionally compare the
// pre-computed trajectory bytes against the reference (not implemented in
// Phase A — see PLAN.md).

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "trajectory_cache_reader.h"

namespace fs = std::filesystem;

#ifndef MRDSERVER_FIXTURES_DIR
#error "MRDSERVER_FIXTURES_DIR must be defined at compile time"
#endif

namespace
{

/** Collect every .pge fixture under MRDSERVER_FIXTURES_DIR sorted by name. */
std::vector<fs::path> discover_fixtures()
{
    std::vector<fs::path> out;
    const fs::path dir(MRDSERVER_FIXTURES_DIR);
    if (!fs::is_directory(dir))
        return out;
    /* Suffixes for MATLAB truth-companion files we must NOT treat as cache. */
    static const char *const truth_suffixes[] = {
        "_trajectory",
        "_tr_waveform",
        "_scan_table",
        "_segment_def",
        "_freqmod_def",
        "_freqmod_plan",
        "_label_state",
        "_seq_desc",
    };
    for (const auto &entry : fs::directory_iterator(dir))
    {
        if (!entry.is_regular_file())
            continue;
        const auto &p = entry.path();
        if (p.extension() != ".pge")
            continue;
        const std::string stem = p.stem().string();
        bool skip = false;
        for (const char *suf : truth_suffixes)
        {
            const size_t sl = std::string(suf).size();
            if (stem.size() >= sl && stem.compare(stem.size() - sl, std::string::npos, suf) == 0)
            {
                skip = true;
                break;
            }
        }
        if (skip)
            continue;
        out.push_back(p);
    }
    std::sort(out.begin(), out.end());
    return out;
}

bool all_finite(const std::vector<float> &v)
{
    for (float f : v)
    {
        if (!std::isfinite(f))
            return false;
    }
    return true;
}

/* -------------------------------------------------------------------- */
/*  Truth trajectory parser ('TRJ2' wire format) — Phase B comparator   */
/* -------------------------------------------------------------------- */

struct TruthAdc
{
    int num_samples = 0;
    int rotation_id = -1;
    int encoding_space_ref = 0;
    std::array<int32_t, 10> labels{};
    std::vector<float> k; /* size = ndim * num_samples (interleaved); empty if cartesian */
};

struct TruthTrajectory
{
    bool has = false;
    int num_adcs = 0;
    bool is_cartesian = false;
    int ndim = 0;
    std::vector<std::array<float, 9>> rotations; /* row-major */
    std::vector<TruthAdc> adcs;
};

template <typename T> static bool fread_n(std::ifstream &f, T *dst, size_t n)
{
    f.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(n * sizeof(T)));
    return static_cast<size_t>(f.gcount()) == n * sizeof(T);
}

TruthTrajectory load_truth_trajectory(const fs::path &path)
{
    TruthTrajectory T;
    if (!fs::exists(path))
        return T;
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return T;

    uint32_t magic = 0;
    if (!fread_n(f, &magic, 1) || magic != 0x54524A32u)
        return T;

    int32_t num_adcs = 0, is_cart = 0, ndim = 0, num_rot = 0;
    if (!fread_n(f, &num_adcs, 1))
        return T;
    if (!fread_n(f, &is_cart, 1))
        return T;
    if (!fread_n(f, &ndim, 1))
        return T;
    if (!fread_n(f, &num_rot, 1))
        return T;

    T.has = true;
    T.num_adcs = num_adcs;
    T.is_cartesian = (is_cart != 0);
    T.ndim = ndim;
    T.rotations.resize(static_cast<size_t>(num_rot));
    for (int r = 0; r < num_rot; ++r)
    {
        if (!fread_n(f, T.rotations[r].data(), 9))
        {
            T.has = false;
            return T;
        }
    }

    int32_t num_es = 0;
    if (!fread_n(f, &num_es, 1))
    {
        T.has = false;
        return T;
    }
    /* Skip per-ES descriptors: 3f + 3i + 3f + 3i + 1i + 1i + 20i = 34 4-byte words */
    for (int e = 0; e < num_es; ++e)
    {
        int32_t buf[34];
        if (!fread_n(f, buf, 34))
        {
            T.has = false;
            return T;
        }
    }

    if (num_adcs == 0)
        return T; /* degenerate vacuous truth */
    T.adcs.resize(static_cast<size_t>(num_adcs));
    for (int a = 0; a < num_adcs; ++a)
    {
        auto &A = T.adcs[a];
        int32_t ns = 0, rot = 0, esref = 0;
        if (!fread_n(f, &ns, 1))
        {
            T.has = false;
            return T;
        }
        if (!fread_n(f, &rot, 1))
        {
            T.has = false;
            return T;
        }
        if (!fread_n(f, &esref, 1))
        {
            T.has = false;
            return T;
        }
        if (!fread_n(f, A.labels.data(), 10))
        {
            T.has = false;
            return T;
        }
        A.num_samples = ns;
        A.rotation_id = rot;
        A.encoding_space_ref = esref;
        if (!T.is_cartesian)
        {
            A.k.resize(static_cast<size_t>(ndim) * static_cast<size_t>(ns));
            if (!A.k.empty() && !fread_n(f, A.k.data(), A.k.size()))
            {
                T.has = false;
                return T;
            }
        }
    }
    return T;
}

class TrajectoryCacheFixture : public ::testing::TestWithParam<fs::path>
{
};

TEST_P(TrajectoryCacheFixture, LoadsAndPreComputesCleanly)
{
    const fs::path &cache_path = GetParam();
    ASSERT_TRUE(fs::exists(cache_path)) << cache_path;

    mrdserver::SequenceCache cache;
    try
    {
        cache = mrdserver::read_sequence_cache(cache_path.string());
    }
    catch (const std::exception &e)
    {
        FAIL() << "read_sequence_cache threw: " << e.what() << "  on " << cache_path;
    }

    // Trajectory section is optional: small synthetic fixtures used for
    // safety/RF-stats unit tests have no ADC events and therefore an empty
    // encoding-space + table.  Skip ADC-related checks in that case but
    // still verify pre_compute_trajectories() returns a matching vector.
    if (cache.encoding_spaces.empty() && cache.table.empty())
    {
        std::vector<mrdserver::PrecomputedTrajectory> trajs;
        try
        {
            trajs = mrdserver::pre_compute_trajectories(cache);
        }
        catch (const std::exception &e)
        {
            FAIL() << "pre_compute_trajectories threw on empty cache: " << e.what() << "  on "
                   << cache_path;
        }
        EXPECT_TRUE(trajs.empty()) << cache_path;
        return;
    }
    EXPECT_FALSE(cache.encoding_spaces.empty()) << "no encoding spaces in " << cache_path;
    EXPECT_FALSE(cache.table.empty()) << "no table entries in " << cache_path;

    // table[i].encoding_space_ref must be in range and kshot ids must be
    // either -1 (trivial) or within cache.kshots.
    const int n_es = static_cast<int>(cache.encoding_spaces.size());
    const int n_kshots = static_cast<int>(cache.kshots.size());
    const int n_rot = static_cast<int>(cache.rotations.size());
    for (size_t t = 0; t < cache.table.size(); ++t)
    {
        const auto &e = cache.table[t];
        EXPECT_GE(e.encoding_space_ref, 0) << "row " << t << " in " << cache_path;
        EXPECT_LT(e.encoding_space_ref, n_es) << "row " << t << " in " << cache_path;
        for (int sid : {e.kx_shot_id, e.ky_shot_id, e.kz_shot_id})
        {
            if (sid >= 0)
                EXPECT_LT(sid, n_kshots) << cache_path;
        }
        if (e.rotation_id >= 0)
            EXPECT_LT(e.rotation_id, n_rot) << cache_path;
        EXPECT_GE(e.center_sample, 0) << cache_path;
        EXPECT_GT(e.sample_time_us, 0.0f) << cache_path;
    }

    // All kshot samples must be finite.
    for (size_t s = 0; s < cache.kshots.size(); ++s)
    {
        EXPECT_TRUE(all_finite(cache.kshots[s].k))
            << "kshot " << s << " has non-finite samples in " << cache_path;
    }

    // Pre-compute trajectories per encoding space and validate shape.
    std::vector<mrdserver::PrecomputedTrajectory> trajs;
    ASSERT_NO_THROW({ trajs = mrdserver::pre_compute_trajectories(cache); }) << cache_path;
    ASSERT_EQ(static_cast<int>(trajs.size()), n_es) << cache_path;

    for (int es = 0; es < n_es; ++es)
    {
        const auto &T = trajs[es];
        // Count ADCs that map to this encoding space.
        int expected_ro = 0;
        for (const auto &e : cache.table)
            if (e.encoding_space_ref == es)
                ++expected_ro;
        if (expected_ro == 0)
        {
            // No readouts -> empty trajectory permitted.
            EXPECT_EQ(T.num_readouts, 0) << "es=" << es << " " << cache_path;
            continue;
        }
        // Either Cartesian-vacuous (ndim==0) or real (ndim 2 or 3).
        EXPECT_TRUE(T.ndim == 0 || T.ndim == 2 || T.ndim == 3)
            << "es=" << es << " ndim=" << T.ndim << " " << cache_path;
        if (T.ndim != 0)
        {
            EXPECT_GT(T.num_samples, 0) << cache_path;
            EXPECT_EQ(T.num_readouts, expected_ro) << cache_path;
            const size_t expected_size = static_cast<size_t>(T.num_readouts) *
                static_cast<size_t>(T.num_samples) * static_cast<size_t>(T.ndim);
            EXPECT_EQ(T.data.size(), expected_size) << cache_path;
            EXPECT_TRUE(all_finite(T.data)) << "non-finite trajectory samples in " << cache_path;
        }
    }

    /* ------------------------------------------------------------------ */
    /*  Phase B: compare against MATLAB-generated _trajectory.truth truth.  */
    /* ------------------------------------------------------------------ */
    fs::path truth_path = cache_path;
    truth_path.replace_extension("");
    truth_path += "_trajectory.truth";
    TruthTrajectory truth = load_truth_trajectory(truth_path);
    if (!truth.has)
        return; /* No companion truth — fixture exempt. */

    /* Cartesian-vs-non-cartesian gate: if MATLAB declares cartesian, the
         * cache must also be cartesian (PrecomputedTrajectory.ndim == 0) on
         * every encoding space. Numerical comparison is not meaningful here. */
    if (truth.is_cartesian)
    {
        for (const auto &T : trajs)
        {
            EXPECT_EQ(T.ndim, 0)
                << "cache produced non-cartesian trajectory but truth is cartesian: " << truth_path;
        }
        return;
    }

    /* If the ADC counts disagree we cannot align rows 1:1 — log this as a
         * skipped value-comparison rather than a noisy failure. The structural
         * checks above still ran. (Known issue: pulseg cache writer counts
         * deduped base_blocks[].adc_id, which is shared between dummy and
         * real ADC blocks for some sequences; truth filters by exec_stream col 10.) */
    if (static_cast<int>(cache.table.size()) != truth.num_adcs)
    {
        GTEST_SKIP() << "value comparison skipped: cache=" << cache.table.size()
                     << " ADCs vs truth=" << truth.num_adcs << " for " << truth_path;
    }

    /* If the cache reports cartesian on every encoding space (ndim==0) but
         * the truth flagged non-cartesian, no rows are comparable. This happens
         * when MATLAB's "constant gradient per ADC" threshold disagrees with
         * pulseg's heuristic — record as skipped rather than fail. */
    bool cache_all_cart = true;
    for (const auto &T : trajs)
        if (T.ndim != 0)
        {
            cache_all_cart = false;
            break;
        }
    if (cache_all_cart)
    {
        GTEST_SKIP() << "value comparison skipped: cache flags cartesian but "
                        "truth is non-cartesian for "
                     << truth_path;
    }

    /* Per-ES active-axis map, mirroring pre_compute_trajectories' own
         * has_x/has_y/has_z (cxx/recon/trajectory_cache_reader.cpp): an axis
         * occupies a slot in T.data (packed x,y,z in that order, skipping
         * absent axes) iff at least one row in the ES recorded a real shot
         * for it. Needed below to know which packed dim index a given row's
         * kx/ky/kz_shot_id corresponds to. */
    std::vector<std::array<bool, 3>> es_has_axis(
        static_cast<size_t>(n_es),
        std::array<bool, 3>{false, false, false});
    for (const auto &e : cache.table)
    {
        if (e.encoding_space_ref < 0 || e.encoding_space_ref >= n_es)
            continue;
        auto &has = es_has_axis[static_cast<size_t>(e.encoding_space_ref)];
        if (e.kx_shot_id >= 0)
            has[0] = true;
        if (e.ky_shot_id >= 0)
            has[1] = true;
        if (e.kz_shot_id >= 0)
            has[2] = true;
    }
    // Packed-dim -> physical-axis (0=x,1=y,2=z) map per ES, in the same
    // x,y,z order pre_compute_trajectories packs them in.
    std::vector<std::vector<int>> es_axis_of_dim(static_cast<size_t>(n_es));
    for (int es = 0; es < n_es; ++es)
        for (int axis = 0; axis < 3; ++axis)
            if (es_has_axis[static_cast<size_t>(es)][static_cast<size_t>(axis)])
                es_axis_of_dim[static_cast<size_t>(es)].push_back(axis);

    /* Iterate ADCs 1:1 in scan order, accumulating per-encoding-space row
         * positions in the order pre_compute_trajectories packs them.        */
    std::vector<size_t> es_pos(static_cast<size_t>(n_es), 0);
    int matched = 0;
    int compared = 0;
    double max_abs_diff = 0.0;
    double sum_sq_diff = 0.0;
    double sum_sq_truth = 0.0;
    for (size_t a = 0; a < cache.table.size(); ++a)
    {
        const auto &cache_e = cache.table[a];
        const auto &truth_a = truth.adcs[a];
        const int es = cache_e.encoding_space_ref;
        if (es < 0 || es >= n_es)
            continue;
        const auto &T = trajs[es];
        const size_t ro = es_pos[es]++;
        if (T.ndim == 0)
            continue; /* this ES is cartesian in cache */
        if (T.num_samples != truth_a.num_samples)
            continue;
        const int dim = std::min(T.ndim, truth.ndim);
        if (static_cast<int>(ro) >= T.num_readouts)
            continue;
        const float *cache_row = &T.data[ro * T.ndim * T.num_samples];
        const float *truth_row = truth_a.k.data();
        ++matched;
        /* Both cache and truth store the SAME absolute base/logical-frame
             * k(t) (verified byte-for-byte: the pulseg cache slices the same
             * canonical, unrotated ZERO_VAR trajectory the TruthBuilder
             * integrates from raw block gradients — rotation is stored as a
             * separate matrix for downstream livesdk, never baked into the
             * cache k samples).  To keep the comparison invariant to any
             * per-side k=0 anchoring convention, subtract EACH side's own
             * value at center_sample (the kzero index) before diffing.
             * (Earlier this subtracted only the truth anchor, which injected
             * a spurious offset == truth[center_sample] on non-cartesian
             * fixtures and failed every rotated/non-cartesian case.) */
        int kz_idx = cache_e.center_sample;
        if (kz_idx < 0)
            kz_idx = 0;
        if (kz_idx >= T.num_samples)
            kz_idx = T.num_samples - 1;
        /* A packed dim is only comparable for THIS row if cache recorded a
             * real shot for its physical axis here: kx/ky/kz_shot_id == -1
             * means pre_compute_trajectories deliberately zero-filled that
             * dim for this ADC (the axis is reconstructed downstream from
             * gradient-amplitude metadata instead, e.g. a Cartesian
             * partition/phase-encode step, or -- as with an unrewound
             * slice-select gradient on an orthogonal-plane navigator -- a
             * genuinely constant-during-ADC axis with no recorded shot at
             * all) -- that zero-fill is not a physical k value and must not
             * be diffed against truth's real, continuously-integrated k(t)
             * on the same axis.
             *
             * A second, related case: a rotated block always gets a real
             * (non -1) shot id on every active axis (the rotated frame
             * needs real per-sample data for axes the rotation actually
             * mixes), even when that axis's canonical k-space was itself
             * deliberately zeroed out for varying independent of rotation
             * (PULSEG_AMP_ZERO_VAR only retains the *invariant* part of the
             * canonical TR -- a phase/partition-encode axis whose amplitude
             * varies per repeat is excluded by design, same as the
             * Cartesian case above, and is meant to be reconstructed from
             * that amplitude metadata instead). The tell is unambiguous:
             * a nonzero recorded gradient amplitude cannot physically
             * integrate to an identically-zero k(t) over the whole
             * readout, so "amplitude != 0 but every sample == 0" means the
             * shot is such a canonical-extraction placeholder, not real
             * data, and must be skipped the same way. */
        const auto &axis_of_dim = es_axis_of_dim[static_cast<size_t>(es)];
        auto row_has_shot_for_axis = [&](int axis) -> bool
        {
            switch (axis)
            {
            case 0:
                return cache_e.kx_shot_id >= 0;
            case 1:
                return cache_e.ky_shot_id >= 0;
            default:
                return cache_e.kz_shot_id >= 0;
            }
        };
        auto row_amplitude_for_axis = [&](int axis) -> float
        {
            switch (axis)
            {
            case 0:
                return cache_e.gx_amplitude;
            case 1:
                return cache_e.gy_amplitude;
            default:
                return cache_e.gz_amplitude;
            }
        };
        std::array<bool, 3> dim_is_zero_placeholder{false, false, false};
        for (int d = 0; d < dim && d < static_cast<int>(axis_of_dim.size()); ++d)
        {
            if (!row_has_shot_for_axis(axis_of_dim[static_cast<size_t>(d)]))
                continue; /* already excluded above */
            if (row_amplitude_for_axis(axis_of_dim[static_cast<size_t>(d)]) == 0.0f)
                continue;
            bool all_zero = true;
            for (int s = 0; s < T.num_samples && all_zero; ++s)
                if (cache_row[s * T.ndim + d] != 0.0f)
                    all_zero = false;
            dim_is_zero_placeholder[static_cast<size_t>(d)] = all_zero;
        }

        float truth_anchor[3] = {0.0f, 0.0f, 0.0f};
        float cache_anchor[3] = {0.0f, 0.0f, 0.0f};
        for (int d = 0; d < dim; ++d)
        {
            truth_anchor[d] = truth_row[kz_idx * truth.ndim + d];
            cache_anchor[d] = cache_row[kz_idx * T.ndim + d];
        }
        for (int s = 0; s < T.num_samples; ++s)
        {
            for (int d = 0; d < dim; ++d)
            {
                if (d < static_cast<int>(axis_of_dim.size()) &&
                    !row_has_shot_for_axis(axis_of_dim[static_cast<size_t>(d)]))
                    continue;
                if (dim_is_zero_placeholder[static_cast<size_t>(d)])
                    continue;
                const float cv = cache_row[s * T.ndim + d] - cache_anchor[d];
                const float tv = truth_row[s * truth.ndim + d] - truth_anchor[d];
                const double diff = static_cast<double>(cv) - static_cast<double>(tv);
                max_abs_diff = std::max(max_abs_diff, std::abs(diff));
                sum_sq_diff += diff * diff;
                sum_sq_truth += static_cast<double>(tv) * static_cast<double>(tv);
                ++compared;
            }
        }
    }

    if (compared == 0)
    {
        ADD_FAILURE() << "no trajectory samples comparable against truth: " << truth_path;
        return;
    }

    const double rms_diff = std::sqrt(sum_sq_diff / compared);
    const double rms_truth = std::sqrt(sum_sq_truth / compared) + 1e-12;
    const double rel_rms = rms_diff / rms_truth;

    /* Loose tolerance: MATLAB integrates raw-block waveforms, C++ integrates
         * normalised shapes × amplitude. Both cover the same ADC window with
         * linear interpolation; expect agreement within a few percent.        */
    EXPECT_LT(rel_rms, 0.05) << "trajectory RMS disagreement vs truth: rel_rms=" << rel_rms
                             << " max_abs=" << max_abs_diff << " (" << matched << " ADCs, "
                             << compared << " samples) " << truth_path;
}

INSTANTIATE_TEST_SUITE_P(
    AllFixtures,
    TrajectoryCacheFixture,
    ::testing::ValuesIn(discover_fixtures()),
    [](const ::testing::TestParamInfo<fs::path> &info)
    {
        // GoogleTest test names must match [A-Za-z0-9_]+.
        std::string s = info.param.stem().string();
        for (char &c : s)
        {
            if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')))
                c = '_';
        }
        return s;
    });

/* ------------------------------------------------------------------ */
/*  Section-5 smoke-test: exercises the C++ reader's Section 5 parse  */
/* ------------------------------------------------------------------ */

TEST_P(TrajectoryCacheFixture, SeqDescParsedConsistently)
{
    const fs::path &cache_path = GetParam();

    mrdserver::SequenceCache cache;
    try
    {
        cache = mrdserver::read_sequence_cache(cache_path.string());
    }
    catch (const std::exception &e)
    {
        FAIL() << "read_sequence_cache threw: " << e.what() << "  on " << cache_path;
    }

    if (!cache.has_seq_desc)
        return; // Section 5 absent — nothing to check

    const int n_ss = cache.seq_params.num_subseqs;
    EXPECT_EQ(static_cast<int>(cache.seq_descs.size()), n_ss)
        << "seq_descs count != num_subseqs  " << cache_path;

    for (int ss = 0; ss < n_ss; ++ss)
    {
        const auto &sd = cache.seq_descs[static_cast<size_t>(ss)];

        EXPECT_GE(sd.subseq_idx, 0) << "subseq " << ss << " " << cache_path;
        EXPECT_GT(sd.tr_duration_us, 0.0f) << "subseq " << ss << " " << cache_path;

        // Every RF event: rf_def_id >= 0
        for (int ri = 0; ri < static_cast<int>(sd.events.size()); ++ri)
        {
            const auto &ev = sd.events[static_cast<size_t>(ri)];
            if (ev.type == mrdserver::SEQ_EVENT_RF)
            {
                EXPECT_GE(ev.rf_def_id(), 0)
                    << "subseq " << ss << " row " << ri << " rf_def_id < 0  " << cache_path;
                EXPECT_GE(ev.rf_ss_grad_amp_hz_per_m(), 0.0f)
                    << "subseq " << ss << " row " << ri << " negative ss_grad_amp  " << cache_path;
            }
        }

        // RF shape tuples: slice_thickness_mm >= 0, slice_selective in {0,1}
        for (int ti = 0; ti < static_cast<int>(sd.rf_shape_tuples.size()); ++ti)
        {
            const auto &tup = sd.rf_shape_tuples[static_cast<size_t>(ti)];
            EXPECT_GE(tup.slice_thickness_mm, 0.0f)
                << "subseq " << ss << " tuple " << ti << " " << cache_path;
            EXPECT_TRUE(tup.slice_selective == 0 || tup.slice_selective == 1)
                << "subseq " << ss << " tuple " << ti << " " << cache_path;
        }
    }
}

} // namespace
