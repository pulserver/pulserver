/*
 * test_safety_grad.c -- gradient safety tests.
 *
 * Suite A: gradient amplitude / slew-rate limit violations (4 tests).
 * Suite B: gradient continuity checks (17 tests).
 */
#include "test_helpers.h"

#include "pulseg_pns_models.h"

/* ================================================================== */
/*  Shared data-driven helpers                                        */
/* ================================================================== */

static pulseg_opts s_opts;
static pulseg_diagnostic s_diag;

/**
 * Load a sequence, run check_safety with the current s_opts,
 * compare return code to expected_code.
 */
static void run_safety_check(const char *filename, int expected_code)
{
    pulseg_collection *coll = NULL;
    int rc;

    rc = load_seq(&coll, filename, &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_check_safety(
        coll,
        &s_diag,
        NULL,
        &s_opts,
        NULL, /* no forbidden bands */
        NULL,
        0.0f /* no PNS */);

    if (expected_code > 0)
    {
        mu_assert(PULSEG_SUCCEEDED(rc), "expected success but got failure");
    }
    else
    {
        mu_assert_int_eq(expected_code, rc);
    }

    pulseg_collection_free(coll);
}

/* ================================================================== */
/*  Continuity-class error detection                                  */
/* ================================================================== */

/**
 * Returns non-zero if @p rc is a gradient-continuity-class error:
 * either the segmentation check (nonzero start/end gradient at TR
 * boundaries) or the safety check (inter-block discontinuity).
 */
static int is_grad_continuity_error(int rc)
{
    return rc == PULSEG_ERR_SEG_NONZERO_START_GRAD || rc == PULSEG_ERR_SEG_NONZERO_END_GRAD ||
        rc == PULSEG_ERR_GRAD_DISCONTINUITY;
}

/**
 * Load a sequence and check gradient continuity.
 *
 * The library detects gradient continuity violations at two stages:
 *   1. Segmentation (during pulseg_read) — catches gradients whose
 *      first/last sample at TR boundaries exceeds max_slew * grad_raster.
 *   2. Safety check (check_grad_continuity) — catches inter-block
 *      gradient steps that exceed the same threshold.
 *
 * For "should-pass" cases the safety check may still fail for
 * non-continuity reasons (e.g. max-slew-rate on trapezoidal ramps that
 * were generated at pypulseq's max_slew); we only verify that
 * PULSEG_ERR_GRAD_DISCONTINUITY is *not* returned.
 *
 * @param filename    Basename of .seq in TEST_DATA_DIR.
 * @param should_pass 1 = sequence is gradient-continuous (no grad-class
 *                    error expected); 0 = a grad-class error is expected.
 */
static void run_continuity_check(const char *filename, int should_pass)
{
    pulseg_collection *coll = NULL;
    int rc;

    rc = load_seq(&coll, filename, &s_opts);

    if (PULSEG_FAILED(rc))
    {
        if (is_grad_continuity_error(rc))
        {
            /* Segmentation caught a gradient boundary issue */
            mu_assert(!should_pass, "continuous sequence rejected by grad boundary check");
            return;
        }
        /* Non-gradient load failure — unexpected for continuity tests */
        mu_assert(0, "load_seq failed with unexpected error");
        return;
    }

    /* Load succeeded — run full safety check */
    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_check_safety(coll, &s_diag, NULL, &s_opts, NULL, NULL, 0.0f);

    if (should_pass)
    {
        /* Sequence is continuous; safety may fail for non-continuity
         * reasons (e.g. slew rate) but must not be GRAD_DISCONTINUITY. */
        mu_assert(
            rc != PULSEG_ERR_GRAD_DISCONTINUITY,
            "expected no discontinuity but got GRAD_DISCONTINUITY");
    }
    else
    {
        mu_assert(rc == PULSEG_ERR_GRAD_DISCONTINUITY, "expected GRAD_DISCONTINUITY");
    }

    pulseg_collection_free(coll);
}

/* ================================================================== */
/*  Suite A — Gradient limit tests                                    */
/* ================================================================== */

/*
 * For amplitude tests: set max_grad tight, max_slew huge.
 * For slew tests: set max_slew tight, max_grad huge.
 * Non-tested limits are 1e10 so they never fire first.
 */

static void grad_limit_opts(float max_grad, float max_slew)
{
    pulseg_opts_init(&s_opts, GAMMA_HZ_PER_T, 3.0f, max_grad, max_slew, 1.0f, 10.0f, 0.1f, 10.0f);
}

MU_TEST(test_grad_amplitude_violation)
{
    grad_limit_opts(10.0f, 1e10f);
    run_safety_check("01_grad_amplitude_violation.seq", PULSEG_ERR_MAX_GRAD_EXCEEDED);
}

MU_TEST(test_slew_violation)
{
    grad_limit_opts(1e10f, 100.0f);
    run_safety_check("02_slew_violation.seq", PULSEG_ERR_MAX_SLEW_EXCEEDED);
}

MU_TEST(test_grad_rss_violation)
{
    grad_limit_opts(10.0f, 1e10f);
    run_safety_check("03_grad_rss_violation.seq", PULSEG_ERR_MAX_GRAD_EXCEEDED);
}

MU_TEST(test_slew_rss_violation)
{
    grad_limit_opts(1e10f, 100.0f);
    run_safety_check("04_slew_rss_violation.seq", PULSEG_ERR_MAX_SLEW_EXCEEDED);
}

MU_TEST_SUITE(suite_grad_limits)
{
    MU_RUN_TEST(test_grad_amplitude_violation);
    MU_RUN_TEST(test_slew_violation);
    MU_RUN_TEST(test_grad_rss_violation);
    MU_RUN_TEST(test_slew_rss_violation);
}

/* ================================================================== */
/*  Suite B — Gradient continuity tests (16 of 17 files)              */
/* ================================================================== */

MU_TEST(test_cont_01_ok_trap_extended_trap)
{
    run_continuity_check("01_ok_trap_extended_trap.seq", 1);
}

MU_TEST(test_cont_02_fail_trap_then_startshigh)
{
    run_continuity_check("02_fail_trap_then_startshigh.seq", 0);
}

MU_TEST(test_cont_03_fail_startshigh_first)
{
    run_continuity_check("03_fail_startshigh_first.seq", 0);
}

MU_TEST(test_cont_04_fail_delay_then_allhigh)
{
    run_continuity_check("04_fail_delay_then_allhigh.seq", 0);
}

MU_TEST(test_cont_05_ok_extended_with_delay)
{
    run_continuity_check("05_ok_extended_with_delay.seq", 1);
}

MU_TEST(test_cont_06_fail_delay_then_startshigh)
{
    run_continuity_check("06_fail_delay_then_startshigh.seq", 0);
}

MU_TEST(test_cont_07_fail_nonconnecting)
{
    run_continuity_check("07_fail_nonconnecting.seq", 0);
}

MU_TEST(test_cont_08_ok_rot_identity)
{
    run_continuity_check("08_ok_rot_identity.seq", 1);
}

MU_TEST(test_cont_09_fail_rot_identity)
{
    run_continuity_check("09_fail_rot_identity.seq", 0);
}

MU_TEST(test_cont_10_fail_rot_first_block)
{
    run_continuity_check("10_fail_rot_first_block.seq", 0);
}

MU_TEST(test_cont_11_fail_rot_allhigh)
{
    run_continuity_check("11_fail_rot_allhigh.seq", 0);
}

MU_TEST(test_cont_12_ok_rot_extended_delay)
{
    run_continuity_check("12_ok_rot_extended_delay.seq", 1);
}

MU_TEST(test_cont_13_fail_rot_delay_then_startshigh)
{
    run_continuity_check("13_fail_rot_delay_then_startshigh.seq", 0);
}

MU_TEST(test_cont_14_fail_rot_nonconnecting)
{
    run_continuity_check("14_fail_rot_nonconnecting.seq", 0);
}

MU_TEST(test_cont_15_ok_rot_same_rotation)
{
    run_continuity_check("15_ok_rot_same_rotation.seq", 1);
}

MU_TEST(test_cont_16_fail_rot_diff_rotation_1)
{
    run_continuity_check("16_fail_rot_diff_rotation_1.seq", 0);
}

MU_TEST(test_cont_17_fail_rot_diff_rotation_2)
{
    run_continuity_check("17_fail_rot_diff_rotation_2.seq", 0);
}

static void continuity_setup(void)
{
    default_opts_init(&s_opts);
}

MU_TEST_SUITE(suite_grad_continuity)
{
    MU_SUITE_CONFIGURE(continuity_setup, NULL);
    MU_RUN_TEST(test_cont_01_ok_trap_extended_trap);
    MU_RUN_TEST(test_cont_02_fail_trap_then_startshigh);
    MU_RUN_TEST(test_cont_03_fail_startshigh_first);
    MU_RUN_TEST(test_cont_04_fail_delay_then_allhigh);
    MU_RUN_TEST(test_cont_05_ok_extended_with_delay);
    MU_RUN_TEST(test_cont_06_fail_delay_then_startshigh);
    MU_RUN_TEST(test_cont_07_fail_nonconnecting);
    MU_RUN_TEST(test_cont_08_ok_rot_identity);
    MU_RUN_TEST(test_cont_09_fail_rot_identity);
    MU_RUN_TEST(test_cont_10_fail_rot_first_block);
    MU_RUN_TEST(test_cont_11_fail_rot_allhigh);
    MU_RUN_TEST(test_cont_12_ok_rot_extended_delay);
    MU_RUN_TEST(test_cont_13_fail_rot_delay_then_startshigh);
    MU_RUN_TEST(test_cont_14_fail_rot_nonconnecting);
    MU_RUN_TEST(test_cont_15_ok_rot_same_rotation);
    MU_RUN_TEST(test_cont_16_fail_rot_diff_rotation_1);
    MU_RUN_TEST(test_cont_17_fail_rot_diff_rotation_2);
}

/* ================================================================== */
/*  Suite C -- Canonical segment-sequence API (gradient safety path)  */
/* ================================================================== */

static void assert_canonical_sequence_matches_expected(
    const pulseg_collection *coll,
    int subseq_idx)
{
    const pulseg_sequence_descriptor *desc;
    int n_main;
    int ncanon, i;
    int *canon_ids = NULL;

    desc = &coll->descriptors[subseq_idx];
    n_main = desc->segment_table.num_main_segments;

    ncanon = pulseg_get_canonical_segment_sequence(coll, NULL, subseq_idx);
    mu_assert_int_eq(n_main, ncanon);

    if (ncanon > 0)
    {
        canon_ids = (int *)malloc((size_t)ncanon * sizeof(int));
        mu_assert(canon_ids != NULL, "malloc failed for canonical segment ids");
    }

    ncanon = pulseg_get_canonical_segment_sequence(coll, canon_ids, subseq_idx);
    mu_assert_int_eq(n_main, ncanon);

    for (i = 0; i < n_main; ++i)
        mu_assert_int_eq(desc->segment_table.main_segment_table[i], canon_ids[i]);

    free(canon_ids);
}

MU_TEST(test_canonical_segment_sequence_degenerate_main_only)
{
    pulseg_collection *coll = NULL;
    int rc;

    default_opts_init(&s_opts);
    rc = load_seq(&coll, "00_basic_rfstat.seq", &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

    assert_canonical_sequence_matches_expected(coll, 0);

    pulseg_collection_free(coll);
}

MU_TEST_SUITE(suite_grad_canonical_sequence)
{
    MU_RUN_TEST(test_canonical_segment_sequence_degenerate_main_only);
}

/* ================================================================== */
/*  Suite D — Mechanical resonance forbidden-frequency safety tests              */
/* ================================================================== */

/* Re-enabled for mechres Part 1 (F1 Option B): pulseg_check_safety now
 * drives the structural analysis with a real resolution/max-frequency
 * whenever forbidden bands are configured, instead of silently no-op'ing
 * (num_freq_bins==0). See REVIEW_safety_mech_resonance.md F1/F4/F11 and
 * PLAN_safety_mechres_fixes.md Part 1 step 1.4. */

/*
 * gre_opts_init's max_grad/max_slew (28 mT/m / 150 T/m/s) are tuned for
 * the segmentation-generator GRE/bSSFP fixtures and are exceeded by this
 * EPI readout's slew rate (unrelated to mech-resonance -- pulseg_check_safety
 * runs check_max_slew() before the per-subsequence mech-resonance loop, so a
 * tight slew limit here would fail every test in this suite before the
 * acoustic analysis is ever reached). Use generous grad/slew limits so this
 * suite exercises only the mech-resonance path.
 */
static TEST_MAYBE_UNUSED void mech_resonances_opts_init(pulseg_opts *opts)
{
    pulseg_opts_init(
        opts,
        GAMMA_HZ_PER_T,
        3.0f,
        GAMMA_HZ_PER_T * 0.080f, /* 80 mT/m -> Hz/m */
        GAMMA_HZ_PER_T * 400.0f, /* 400 T/m/s -> Hz/m/s */
        2.0f,
        20.0f,
        2.0f,
        20.0f);
}

/**
 * Load a sequence, run check_safety with the given forbidden bands,
 * compare return code to expected_code.
 *
 * expected_code > 0 means a passing (success) result is expected.
 * expected_code <= 0 means that specific error code is expected.
 */
static void run_mech_resonances_check(
    const char *filename,
    int num_bands,
    const pulseg_forbidden_band *bands,
    int expected_code)
{
    pulseg_collection *coll = NULL;
    pulseg_forbidden_band_list band_list = PULSEG_FORBIDDEN_BAND_LIST_INIT;
    int rc;

    rc = load_corpus_seq(&coll, filename, &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed for acoustic test");

    pulseg_diagnostic_init(&s_diag);
    band_list.count = num_bands;
    band_list.bands = bands;
    rc = pulseg_check_safety(coll, &s_diag, NULL, &s_opts, &band_list, NULL, 0.0f /* no PNS */);

    if (expected_code > 0)
    {
        mu_assert(PULSEG_SUCCEEDED(rc), "expected acoustic safety pass");
    }
    else
    {
        mu_assert_int_eq(expected_code, rc);
    }

    pulseg_collection_free(coll);
}

/*
 * EPI readout produces a peaked acoustic spectrum at 1/ESP and harmonics.
 * This fixture's readout train drives GX at ~1236 Hz (~10.4 mT/m).
 * Forbid a band spanning 800–2500 Hz at zero amplitude and expect an
 * acoustic violation.
 */
MU_TEST(test_epi_forbidden_readout_peak)
{
    pulseg_forbidden_band bands[2];

    /* Band 1: 800–2500 Hz — covers 1/ESP fundamental */
    bands[0].freq_min_hz = 800.0f;
    bands[0].freq_max_hz = 2500.0f;
    bands[0].max_amplitude_hz_per_m = 0.0f;

    /* Band 2: 2500–4500 Hz — covers 2nd harmonic */
    bands[1].freq_min_hz = 2500.0f;
    bands[1].freq_max_hz = 4500.0f;
    bands[1].max_amplitude_hz_per_m = 0.0f;

    run_mech_resonances_check("epi_2d_main.seq", 2, bands, PULSEG_ERR_MECH_RESONANCES_VIOLATION);
}

/*
 * Same sequence/bands as test_epi_forbidden_readout_peak, but with the
 * band amplitude limit set far above anything the readout train can
 * produce -> must pass (F1 gate now actually runs the analysis and
 * correctly clears it, rather than trivially no-op-passing).
 */
MU_TEST(test_epi_forbidden_band_amplitude_above_train_passes)
{
    pulseg_forbidden_band bands[2];

    bands[0].freq_min_hz = 800.0f;
    bands[0].freq_max_hz = 2500.0f;
    bands[0].max_amplitude_hz_per_m = 1.0e9f;

    bands[1].freq_min_hz = 2500.0f;
    bands[1].freq_max_hz = 4500.0f;
    bands[1].max_amplitude_hz_per_m = 1.0e9f;

    run_mech_resonances_check("epi_2d_main.seq", 2, bands, 1 /* pass */);
}

/*
 * No forbidden bands configured -> skip path (mr_target_res_hz/
 * mr_max_freq_hz stay 0,0; num_freq_bins==0; sa_check_structural_violations
 * returns SUCCESS before building the evaluation grid). Must pass
 * regardless of what the sequence actually contains.
 */
MU_TEST(test_epi_no_bands_skips_mech_resonance_check)
{
    run_mech_resonances_check("epi_2d_main.seq", 0, NULL, 1 /* pass */);
}

/*
 * No cross-band masking: each forbidden band is evaluated independently on its
 * own in-band spectral lines, so the strong off-band EPI readout (~1236 Hz)
 * cannot influence a band elsewhere. This band brackets the first TR harmonic
 * (f1 = 1/TR ~= 96.5 Hz for this fixture's TR = 10.36 ms).
 *
 * A_eq criterion: the drive at the TR fundamental is the sequence's slow
 * envelope, ~3.2 mT/m here -- below the readout-scale floor
 * eps = k*G_max (k=0.08 -> ~6.4 mT/m at G_max=80 mT/m). It is NOT a
 * readout-scale sustained line, so it PASSES: every periodic gradient
 * sprinkles weak harmonics into any band wider than its comb spacing, and
 * only readout-scale lines count.
 */
MU_TEST(test_epi_tr_fundamental_below_readout_floor_passes)
{
    pulseg_forbidden_band band;

    band.freq_min_hz = 90.0f;
    band.freq_max_hz = 103.0f;
    band.max_amplitude_hz_per_m = 0.0f;

    run_mech_resonances_check("epi_2d_main.seq", 1, &band, 1 /* pass */);
}

/*
 * A_eq guard + eps behaviour on split bands (this fixture's dominant EPI
 * readout line is ~1236 Hz at ~10.4 mT/m on GX). With G_max=80 mT/m the
 * readout floor is eps = k*G_max ~= 6.4 mT/m.
 *
 *  - band_lo [1500,2350]: the 1236 Hz readout line sits ~264 Hz below the
 *    edge but within the frequency guard (HWHM = min_band_width/2 = 425 Hz),
 *    so it counts at full amplitude > eps -> VIOLATION. (Covers the guard
 *    picking up a strong line just outside a band edge.)
 *  - band_hi [2600,3450]: the readout fundamental is out of guard range and
 *    only weak harmonics (< 1 mT/m) are in-band, below eps -> PASS. (Weak
 *    harmonics in a wide band are not readout-scale lines. This verdict is
 *    G_max-dependent by design -- a coil with a smaller G_max, hence smaller
 *    eps, would flag it.)
 */
MU_TEST(test_epi_guard_catches_edge_line_weak_harmonic_passes)
{
    pulseg_forbidden_band band_lo, band_hi;

    band_lo.freq_min_hz = 1500.0f;
    band_lo.freq_max_hz = 2350.0f;
    band_lo.max_amplitude_hz_per_m = 0.0f;

    band_hi.freq_min_hz = 2600.0f;
    band_hi.freq_max_hz = 3450.0f;
    band_hi.max_amplitude_hz_per_m = 0.0f;

    run_mech_resonances_check("epi_2d_main.seq", 1, &band_lo, PULSEG_ERR_MECH_RESONANCES_VIOLATION);
    run_mech_resonances_check("epi_2d_main.seq", 1, &band_hi, 1 /* pass */);
}

static void mech_resonances_setup(void)
{
    mech_resonances_opts_init(&s_opts);
}

MU_TEST_SUITE(suite_mech_resonances_safety)
{
    MU_SUITE_CONFIGURE(mech_resonances_setup, NULL);
    MU_RUN_TEST(test_epi_forbidden_readout_peak);
    MU_RUN_TEST(test_epi_forbidden_band_amplitude_above_train_passes);
    MU_RUN_TEST(test_epi_no_bands_skips_mech_resonance_check);
    MU_RUN_TEST(test_epi_tr_fundamental_below_readout_floor_passes);
    MU_RUN_TEST(test_epi_guard_catches_edge_line_weak_harmonic_passes);
}

/* ================================================================== */
/*  Suite E - Memoized PNS equivalence                                */
/* ================================================================== */

/*
 * A model that publishes its kernel (pulseg_pns_model::kernel) lets the
 * safety core assemble the response from one convolution per distinct
 * gradient shape instead of one transform over the whole canonical window
 * (pulseg_pns_memo.c). The two routes must agree: that is the entire
 * contract, and it is what these tests pin down.
 *
 * They cannot agree bit for bit. The memoized route factors each block's
 * amplitude out of its shape, so it computes a_i * (unit sample) where the
 * exact route stores (a_i * unit) -- a difference of up to an ulp per
 * sample. The tolerance below is a few hundred times float epsilon on the
 * peak, which is the number the gate actually compares against a threshold;
 * measured agreement across the zoo sits near 5e-7 relative.
 */

#define PNS_MEMO_TEST_RTOL 1e-5

typedef struct
{
    float chronaxie_us;
    float rheobase;
    float alpha;
} pns_test_ctx;

/* Each tap is the bin integral of c/(c+tau)^2 over its own sample interval,
 * taken here through the antiderivative -c/(c+tau) evaluated at the interval's
 * ends, in double precision. The shipped model computes the same integral in
 * closed product form; two routes to one integral is what keeps this reference
 * independent below the formula. */
static int pns_test_build_kernel(const pns_test_ctx *c, float dt_us, float **out, int *len)
{
    double c_s, dt_s, s_min, lo, hi;
    float *k;
    int i, n;

    if (!c || dt_us <= 0.0f || c->chronaxie_us <= 0.0f || c->rheobase <= 0.0f || c->alpha <= 0.0f)
        return PULSEG_ERR_PNS_INVALID_PARAMS;
    c_s = (double)c->chronaxie_us * 1e-6;
    dt_s = (double)dt_us * 1e-6;
    s_min = (double)c->rheobase / (double)c->alpha;
    n = (int)(20.0 * c_s / dt_s) + 1;
    k = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    if (!k)
        return PULSEG_ERR_ALLOC_FAILED;
    for (i = 0; i < n; ++i)
    {
        lo = c_s / (c_s + (double)i * dt_s);
        hi = c_s / (c_s + (double)(i + 1) * dt_s);
        k[i] = (float)((lo - hi) / s_min);
    }
    *out = k;
    *len = n;
    return PULSEG_SUCCESS;
}

static int pns_test_required_padding(void *ctx, float dt_us)
{
    float *k = NULL;
    int n = 0;
    int rc = pns_test_build_kernel((const pns_test_ctx *)ctx, dt_us, &k, &n);
    if (PULSEG_FAILED(rc))
        return rc;
    PULSEG_FREE(k);
    return n;
}

static int pns_test_evaluate(
    void *ctx,
    const float *dx,
    const float *dy,
    const float *dz,
    int n,
    float dt_us,
    float *ox,
    float *oy,
    float *oz)
{
    float *k = NULL;
    int kl = 0, i, rc;

    rc = pns_test_build_kernel((const pns_test_ctx *)ctx, dt_us, &k, &kl);
    if (PULSEG_FAILED(rc))
        return rc;
    rc = pulseg__calc_convolution_fft(ox, dx, n, k, kl);
    if (!PULSEG_FAILED(rc))
        rc = pulseg__calc_convolution_fft(oy, dy, n, k, kl);
    if (!PULSEG_FAILED(rc))
        rc = pulseg__calc_convolution_fft(oz, dz, n, k, kl);
    if (!PULSEG_FAILED(rc))
        for (i = 0; i < n; ++i)
        {
            ox[i] *= 100.0f;
            oy[i] *= 100.0f;
            oz[i] *= 100.0f;
        }
    PULSEG_FREE(k);
    return rc;
}

static int pns_test_kernel(void *ctx, float dt_us, float **k, int *len, float *scale)
{
    int rc = pns_test_build_kernel((const pns_test_ctx *)ctx, dt_us, k, len);
    if (PULSEG_FAILED(rc))
        return rc;
    *scale = 100.0f;
    return PULSEG_SUCCESS;
}

/* Peak combined PNS over a result, i.e. what the gate thresholds. */
static double pns_peak(const pulseg_pns_result *r)
{
    double best = 0.0, v;
    int i;

    for (i = 0; i < r->num_samples; ++i)
    {
        v = sqrt(
            (double)r->slew_x_hz_per_m_per_s[i] * r->slew_x_hz_per_m_per_s[i] +
            (double)r->slew_y_hz_per_m_per_s[i] * r->slew_y_hz_per_m_per_s[i] +
            (double)r->slew_z_hz_per_m_per_s[i] * r->slew_z_hz_per_m_per_s[i]);
        if (v > best)
            best = v;
    }
    return best;
}

/* Run both routes over one fixture and compare. */
static void run_pns_memo_equivalence(const char *filename)
{
    pns_test_ctx ctx = {360.0f, 4.25e8f, 0.333f};
    pulseg_pns_model exact_model = {&ctx, pns_test_required_padding, pns_test_evaluate, NULL};
    pulseg_pns_model memo_model =
        {&ctx, pns_test_required_padding, pns_test_evaluate, pns_test_kernel};
    pulseg_collection *coll = NULL;
    pulseg_pns_result exact, memo;
    double peak_exact, peak_memo;
    int rc;

    rc = load_corpus_seq(&coll, filename, &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed for PNS memo test");

    memset(&exact, 0, sizeof(exact));
    memset(&memo, 0, sizeof(memo));

    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_calc_pns(coll, &exact, &s_diag, NULL, 0, 0, &s_opts, &exact_model);
    mu_assert(PULSEG_SUCCEEDED(rc), "exact PNS evaluation failed");

    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_calc_pns(coll, &memo, &s_diag, NULL, 0, 0, &s_opts, &memo_model);
    mu_assert(PULSEG_SUCCEEDED(rc), "memoized PNS evaluation failed");

    mu_assert_int_eq(exact.num_samples, memo.num_samples);
    mu_assert(exact.num_samples > 0, "PNS produced no samples");

    peak_exact = pns_peak(&exact);
    peak_memo = pns_peak(&memo);
    mu_assert(peak_exact > 0.0, "exact PNS peak is zero: fixture exercises nothing");
    mu_assert(
        fabs(peak_exact - peak_memo) <= PNS_MEMO_TEST_RTOL * peak_exact,
        "memoized PNS peak disagrees with the exact peak");

    /* Every sample, not only the peak. The window the verdict is read from is
     * followed by the model's own memory, filled by wrapping the repetition,
     * and a route that gets that region wrong still agrees on a peak that
     * happens to fall inside the window. */
    {
        const float *m_axis[3];
        const float *e_axis[3];
        double worst = 0.0, scale = 0.0, d;
        int ax, i;

        m_axis[0] = memo.slew_x_hz_per_m_per_s;
        m_axis[1] = memo.slew_y_hz_per_m_per_s;
        m_axis[2] = memo.slew_z_hz_per_m_per_s;
        e_axis[0] = exact.slew_x_hz_per_m_per_s;
        e_axis[1] = exact.slew_y_hz_per_m_per_s;
        e_axis[2] = exact.slew_z_hz_per_m_per_s;

        for (ax = 0; ax < 3; ++ax)
        {
            for (i = 0; i < exact.num_samples; ++i)
            {
                d = fabs((double)m_axis[ax][i] - (double)e_axis[ax][i]);
                if (d > worst)
                    worst = d;
                if (fabs((double)e_axis[ax][i]) > scale)
                    scale = fabs((double)e_axis[ax][i]);
            }
        }
        mu_assert(scale > 0.0, "exact PNS response is identically zero");
        mu_assert(
            worst <= PNS_MEMO_TEST_RTOL * scale,
            "memoized PNS response disagrees with the exact response");
    }

    pulseg_pns_result_free(&exact);
    pulseg_pns_result_free(&memo);
    pulseg_collection_free(coll);
}

MU_TEST(test_pns_memo_matches_exact_gre)
{
    gre_opts_init(&s_opts);
    run_pns_memo_equivalence("gre_2d_3sl.seq");
}

MU_TEST(test_pns_memo_matches_exact_epi)
{
    mech_resonances_opts_init(&s_opts);
    run_pns_memo_equivalence("epi_2d_main.seq");
}

MU_TEST(test_pns_memo_matches_exact_fse)
{
    mech_resonances_opts_init(&s_opts);
    run_pns_memo_equivalence("fse_2d.seq");
}

MU_TEST(test_pns_memo_matches_exact_mprage)
{
    mech_resonances_opts_init(&s_opts);
    run_pns_memo_equivalence("mprage_3d.seq");
}

/* Arbitrary (uniformly rastered) gradients rather than trapezoids: the
 * templates come from shape samples, not corner points. */
MU_TEST(test_pns_memo_matches_exact_noncart)
{
    mech_resonances_opts_init(&s_opts);
    run_pns_memo_equivalence("gre_stack_of_stars_3d.seq");
}

/* A repetition shorter than the nerve model's memory, so the padding that
 * warms the filter wraps the window more than once. A route that replays the
 * wrap a fixed number of times agrees on everything up to the point where the
 * second wrap would start. */
MU_TEST(test_pns_memo_matches_exact_short_window)
{
    mech_resonances_opts_init(&s_opts);
    run_pns_memo_equivalence("gre_stack_of_spirals_3d.seq");
}

/* A live rotation on the canonical TR: the same shape reaches two physical
 * axes turned by different amounts, so slices sharing a waveform identity are
 * not scalar multiples of one another. This is the fixture that distinguishes
 * a memo key sufficient in the physical frame from one that is not. */
MU_TEST(test_pns_memo_matches_exact_rotated)
{
    mech_resonances_opts_init(&s_opts);
    run_pns_memo_equivalence("zte_3d.seq");
}

MU_TEST(test_pns_memo_matches_exact_radial)
{
    mech_resonances_opts_init(&s_opts);
    run_pns_memo_equivalence("gre_radial_2d.seq");
}

/* A model that does not publish a kernel must never take the memoized
 * route -- it may not be a linear filter at all. Exercised implicitly by
 * every other PNS test here, and explicitly by comparing a no-kernel model
 * against itself for byte equality. */
MU_TEST(test_pns_no_kernel_is_deterministic_exact_path)
{
    pns_test_ctx ctx = {360.0f, 4.25e8f, 0.333f};
    pulseg_pns_model model = {&ctx, pns_test_required_padding, pns_test_evaluate, NULL};
    pulseg_collection *coll = NULL;
    pulseg_pns_result a, b;
    int rc, i, differing = 0;

    mech_resonances_opts_init(&s_opts);
    rc = load_corpus_seq(&coll, "epi_2d_main.seq", &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_calc_pns(coll, &a, &s_diag, NULL, 0, 0, &s_opts, &model);
    mu_assert(PULSEG_SUCCEEDED(rc), "first exact PNS evaluation failed");
    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_calc_pns(coll, &b, &s_diag, NULL, 0, 0, &s_opts, &model);
    mu_assert(PULSEG_SUCCEEDED(rc), "second exact PNS evaluation failed");

    mu_assert_int_eq(a.num_samples, b.num_samples);
    for (i = 0; i < a.num_samples; ++i)
        if (a.slew_x_hz_per_m_per_s[i] != b.slew_x_hz_per_m_per_s[i] ||
            a.slew_y_hz_per_m_per_s[i] != b.slew_y_hz_per_m_per_s[i] ||
            a.slew_z_hz_per_m_per_s[i] != b.slew_z_hz_per_m_per_s[i])
            differing++;
    mu_assert_int_eq(0, differing);

    pulseg_pns_result_free(&a);
    pulseg_pns_result_free(&b);
    pulseg_collection_free(coll);
}

/* The shipped Irnich model (pulseg_pns_irnich.c) against this suite's own
 * implementation of the same published form. The two are independent below
 * the formula -- the shipped one convolves directly, this one through
 * pulseg__calc_convolution_fft -- so agreement is evidence about the model,
 * where memo-versus-exact above is evidence about the machinery. */
MU_TEST(test_shipped_irnich_matches_an_independent_implementation)
{
    pns_test_ctx ctx = {360.0f, 4.25e8f, 0.333f};
    pulseg_pns_model reference = {&ctx, pns_test_required_padding, pns_test_evaluate, NULL};
    pulseg_pns_irnich shipped_ctx;
    pulseg_pns_model shipped;
    pulseg_collection *coll = NULL;
    pulseg_pns_result a, b;
    double peak_a, peak_b;
    int rc;

    pulseg_pns_irnich_init(&shipped, &shipped_ctx, 360.0f, 4.25e8f, 0.333f);

    mech_resonances_opts_init(&s_opts);
    rc = load_corpus_seq(&coll, "epi_2d_main.seq", &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_calc_pns(coll, &a, &s_diag, NULL, 0, 0, &s_opts, &reference);
    mu_assert(PULSEG_SUCCEEDED(rc), "reference PNS evaluation failed");
    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_calc_pns(coll, &b, &s_diag, NULL, 0, 0, &s_opts, &shipped);
    mu_assert(PULSEG_SUCCEEDED(rc), "shipped Irnich PNS evaluation failed");

    mu_assert_int_eq(a.num_samples, b.num_samples);
    peak_a = pns_peak(&a);
    peak_b = pns_peak(&b);
    mu_assert(peak_a > 0.0, "reference PNS peak is zero: fixture exercises nothing");
    mu_assert(
        fabs(peak_a - peak_b) <= PNS_MEMO_TEST_RTOL * peak_a,
        "shipped Irnich model disagrees with the independent implementation");

    pulseg_pns_result_free(&a);
    pulseg_pns_result_free(&b);
    pulseg_collection_free(coll);
}

/* The response is a property of the waveform, not of the raster it is
 * evaluated on. Both rasters below see the *same* G(t) -- closed-form, so
 * dG/dt on each grid is the exact bin difference the engine's extraction
 * produces, with no resampling anywhere -- and the peak they report must
 * agree. A kernel whose taps are point samples of c/(c+tau)^2 rather than
 * its bin integrals fails this by ~0.5% at these rasters. */

#define PNS_RS_SINE_HZ 137.0
#define PNS_RS_SINE_AMP 2.0e6
#define PNS_RS_TRI_PERIOD_S 2.7e-3
#define PNS_RS_TRI_AMP 1.5e6
#define PNS_RS_DURATION_S 40.0e-3

static double pns_rs_g_sine(double t)
{
    return PNS_RS_SINE_AMP * sin(2.0 * 3.14159265358979323846 * PNS_RS_SINE_HZ * t);
}

static double pns_rs_g_triangle(double t)
{
    double s = fmod(t, PNS_RS_TRI_PERIOD_S);
    double half = 0.5 * PNS_RS_TRI_PERIOD_S;
    if (s <= half)
        return PNS_RS_TRI_AMP * (s / half);
    return PNS_RS_TRI_AMP * (2.0 - s / half);
}

/* Peak RSS of the shipped Irnich response to the two waveforms above, on one
 * raster. Returns a negative value on any failure. */
static double pns_rs_peak_at(float dt_us)
{
    pulseg_pns_irnich ctx;
    pulseg_pns_model model;
    float *gx, *gy, *gz, *ox, *oy, *oz;
    double dt_s, t0, t1, v, peak;
    int i, n, rc;

    dt_s = (double)dt_us * 1e-6;
    n = (int)(PNS_RS_DURATION_S / dt_s);
    gx = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    gy = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    gz = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    ox = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    oy = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    oz = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    if (!gx || !gy || !gz || !ox || !oy || !oz)
    {
        peak = -1.0;
        goto done;
    }

    for (i = 0; i < n; ++i)
    {
        t0 = (double)i * dt_s;
        t1 = t0 + dt_s;
        gx[i] = (float)((pns_rs_g_sine(t1) - pns_rs_g_sine(t0)) / dt_s);
        gy[i] = (float)((pns_rs_g_triangle(t1) - pns_rs_g_triangle(t0)) / dt_s);
        gz[i] = 0.0f;
    }

    pulseg_pns_irnich_init(&model, &ctx, 360.0f, 4.25e8f, 0.333f);
    rc = model.evaluate(model.ctx, gx, gy, gz, n, dt_us, ox, oy, oz);
    if (PULSEG_FAILED(rc))
    {
        peak = -1.0;
        goto done;
    }

    peak = 0.0;
    for (i = 0; i < n; ++i)
    {
        v = sqrt((double)ox[i] * ox[i] + (double)oy[i] * oy[i] + (double)oz[i] * oz[i]);
        if (v > peak)
            peak = v;
    }

done:
    if (gx)
        PULSEG_FREE(gx);
    if (gy)
        PULSEG_FREE(gy);
    if (gz)
        PULSEG_FREE(gz);
    if (ox)
        PULSEG_FREE(ox);
    if (oy)
        PULSEG_FREE(oy);
    if (oz)
        PULSEG_FREE(oz);
    return peak;
}

/* The exact response peak of the whole scan, every block at the amplitude
 * it actually plays, evaluated cold from the scan's first sample -- the
 * ground truth the occurrence score must bound. Returns a negative value on
 * failure. */
static double pns_score_exact_scan_peak(
    pulseg_collection *coll,
    const pulseg_pns_model *model,
    float gamma)
{
    pulseg_check_plan *plan = NULL;
    const pulseg__uniform_grad_waveforms *uw;
    const pulseg_sequence_descriptor *desc;
    float *dgdt[3], *out[3];
    const float *wavep[3];
    double peak, ss, inv;
    int n, i, a, rc;

    desc = &coll->descriptors[0];
    peak = -1.0;
    for (a = 0; a < 3; ++a)
    {
        dgdt[a] = NULL;
        out[a] = NULL;
    }

    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_check_plan_create(&plan, &s_diag, coll, NULL);
    if (PULSEG_FAILED(rc))
        return -1.0;
    rc = pulseg__plan_waveforms(
        plan,
        &uw,
        &s_diag,
        desc,
        0,
        0,
        desc->num_blocks,
        PULSEG_AMP_ACTUAL,
        NULL,
        0);
    if (PULSEG_FAILED(rc) || uw->num_samples < 2)
        goto done;

    n = uw->num_samples;
    wavep[0] = uw->gx;
    wavep[1] = uw->gy;
    wavep[2] = uw->gz;
    inv = 1.0 / ((double)gamma * ((double)uw->raster_us * 1e-6));
    for (a = 0; a < 3; ++a)
    {
        dgdt[a] = (float *)PULSEG_ALLOC((size_t)(n + 1) * sizeof(float));
        out[a] = (float *)PULSEG_ALLOC((size_t)(n + 1) * sizeof(float));
        if (!dgdt[a] || !out[a])
            goto done;
        dgdt[a][0] = (float)((double)wavep[a][0] * inv);
        for (i = 1; i < n; ++i)
            dgdt[a][i] = (float)(((double)wavep[a][i] - (double)wavep[a][i - 1]) * inv);
        dgdt[a][n] = (float)(-(double)wavep[a][n - 1] * inv);
    }

    rc = model->evaluate(
        model->ctx,
        dgdt[0],
        dgdt[1],
        dgdt[2],
        n + 1,
        uw->raster_us,
        out[0],
        out[1],
        out[2]);
    if (PULSEG_FAILED(rc))
        goto done;

    peak = 0.0;
    for (i = 0; i < n + 1; ++i)
    {
        ss = sqrt(
            (double)out[0][i] * out[0][i] + (double)out[1][i] * out[1][i] +
            (double)out[2][i] * out[2][i]);
        if (ss > peak)
            peak = ss;
    }

done:
    for (a = 0; a < 3; ++a)
    {
        if (dgdt[a])
            PULSEG_FREE(dgdt[a]);
        if (out[a])
            PULSEG_FREE(out[a]);
    }
    pulseg_check_plan_destroy(plan);
    return peak;
}

/* The occurrence score is an upper bound on the exact peak, fixture by
 * fixture; the 2e-3 allowance is the raster-invariance bound between the
 * score's shape-raster templates and the extraction's half-raster stream,
 * far below the l1 gap the score carries by construction. */
static void run_pns_score_soundness(const char *filename)
{
    pulseg_pns_irnich ctx;
    pulseg_pns_model model;
    pulseg_collection *coll = NULL;
    pulseg__pns_score *score = NULL;
    float score_max;
    double exact;
    int rc, declined;

    mech_resonances_opts_init(&s_opts);
    rc = load_corpus_seq(&coll, filename, &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_corpus_seq failed");

    pulseg_pns_irnich_init(&model, &ctx, 360.0f, 4.25e8f, 0.333f);

    rc = pulseg__pns_score_build(
        &score,
        &coll->descriptors[0],
        &model,
        s_opts.gamma_hz_per_t,
        &declined,
        NULL);
    mu_assert(PULSEG_SUCCEEDED(rc), "score build failed");
    mu_assert(!declined && score != NULL, "score declined a corpus fixture");

    rc = pulseg__pns_score_evaluate(score, &score_max, NULL, NULL);
    mu_assert(PULSEG_SUCCEEDED(rc), "score evaluation failed");

    exact = pns_score_exact_scan_peak(coll, &model, s_opts.gamma_hz_per_t);
    mu_assert(exact > 0.0, "exact scan evaluation failed");

    printf(
        "    score %s: bound %.6g, exact %.6g, gap %.3fx\n",
        filename,
        (double)score_max,
        exact,
        (double)score_max / exact);
    mu_assert(
        (double)score_max >= exact * (1.0 - 2.0e-3),
        "the occurrence score fell below the exact scan peak");

    pulseg__pns_score_free(score);
    pulseg_collection_free(coll);
}

/* The score path's verdict against the ground truth, bracketing the true
 * scan peak from both sides. Below the peak the exact assembly must find
 * the violation the bound flagged; above it the bound either clears the
 * threshold outright or the assembly clears what it flagged. Either way the
 * verdict is the truth's, at one percent from the peak. */
static void run_pns_score_verdict_brackets(const char *filename)
{
    static const double factors[4] = {0.5, 0.99, 1.01, 2.0};
    pulseg_pns_irnich ctx;
    pulseg_pns_model model;
    pulseg_collection *coll = NULL;
    pulseg_check_plan *plan = NULL;
    double exact, thr;
    int rc, declined, f, expect_refusal;

    mech_resonances_opts_init(&s_opts);
    rc = load_corpus_seq(&coll, filename, &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_corpus_seq failed");
    pulseg_pns_irnich_init(&model, &ctx, 360.0f, 4.25e8f, 0.333f);

    exact = pns_score_exact_scan_peak(coll, &model, s_opts.gamma_hz_per_t);
    mu_assert(exact > 0.0, "exact scan evaluation failed");

    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_check_plan_create(&plan, &s_diag, coll, NULL);
    mu_assert(PULSEG_SUCCEEDED(rc), "plan creation failed");

    for (f = 0; f < 4; ++f)
    {
        thr = exact * factors[f];
        expect_refusal = (factors[f] < 1.0);
        declined = 0;
        pulseg_diagnostic_init(&s_diag);
        rc = pulseg__pns_score_check(
            plan,
            &s_diag,
            &coll->descriptors[0],
            0,
            &model,
            s_opts.gamma_hz_per_t,
            (float)thr,
            &declined,
            NULL,
            NULL);
        mu_assert(!declined, "score declined a corpus fixture");
        if (expect_refusal)
        {
            mu_assert(
                rc == PULSEG_ERR_PNS_THRESHOLD_EXCEEDED,
                "score path passed a scan whose true peak exceeds the threshold");
            mu_assert(
                s_diag.code == PULSEG_ERR_PNS_THRESHOLD_EXCEEDED,
                "refusal left no diagnostic");
        }
        else
        {
            mu_assert(
                PULSEG_SUCCEEDED(rc),
                "score path refused a scan whose true peak is under the threshold");
        }
    }

    pulseg_check_plan_destroy(plan);
    pulseg_collection_free(coll);
}

MU_TEST(test_the_score_path_verdict_is_the_truths_gre)
{
    run_pns_score_verdict_brackets("gre_2d.seq");
}

MU_TEST(test_the_score_path_verdict_is_the_truths_fse)
{
    run_pns_score_verdict_brackets("fse_2d.seq");
}

MU_TEST(test_the_score_path_verdict_is_the_truths_epi)
{
    run_pns_score_verdict_brackets("epi_2d.seq");
}

MU_TEST(test_the_score_path_verdict_is_the_truths_spiral)
{
    run_pns_score_verdict_brackets("gre_spiral_2d.seq");
}

MU_TEST(test_the_score_bounds_the_exact_scan_peak_gre)
{
    run_pns_score_soundness("gre_2d.seq");
}

MU_TEST(test_the_score_bounds_the_exact_scan_peak_gre_3sl)
{
    run_pns_score_soundness("gre_2d_3sl.seq");
}

MU_TEST(test_the_score_bounds_the_exact_scan_peak_fse)
{
    run_pns_score_soundness("fse_2d.seq");
}

MU_TEST(test_the_score_bounds_the_exact_scan_peak_epi)
{
    run_pns_score_soundness("epi_2d.seq");
}

MU_TEST(test_the_score_bounds_the_exact_scan_peak_spiral)
{
    run_pns_score_soundness("gre_spiral_2d.seq");
}

MU_TEST(test_the_score_bounds_the_exact_scan_peak_radial)
{
    run_pns_score_soundness("gre_radial_2d.seq");
}

MU_TEST(test_the_response_does_not_move_when_the_raster_is_halved)
{
    double coarse = pns_rs_peak_at(4.0f);
    double fine = pns_rs_peak_at(2.0f);

    mu_assert(coarse > 0.0, "coarse-raster evaluation failed");
    mu_assert(fine > 0.0, "fine-raster evaluation failed");
    mu_assert(
        fabs(coarse - fine) <= 1.0e-3 * coarse,
        "the Irnich peak depends on the evaluation raster");
}

MU_TEST_SUITE(suite_pns_memoization)
{
    MU_RUN_TEST(test_shipped_irnich_matches_an_independent_implementation);
    MU_RUN_TEST(test_the_response_does_not_move_when_the_raster_is_halved);
    MU_RUN_TEST(test_the_score_path_verdict_is_the_truths_gre);
    MU_RUN_TEST(test_the_score_path_verdict_is_the_truths_fse);
    MU_RUN_TEST(test_the_score_path_verdict_is_the_truths_epi);
    MU_RUN_TEST(test_the_score_path_verdict_is_the_truths_spiral);
    MU_RUN_TEST(test_the_score_bounds_the_exact_scan_peak_gre);
    MU_RUN_TEST(test_the_score_bounds_the_exact_scan_peak_gre_3sl);
    MU_RUN_TEST(test_the_score_bounds_the_exact_scan_peak_fse);
    MU_RUN_TEST(test_the_score_bounds_the_exact_scan_peak_epi);
    MU_RUN_TEST(test_the_score_bounds_the_exact_scan_peak_spiral);
    MU_RUN_TEST(test_the_score_bounds_the_exact_scan_peak_radial);
    MU_RUN_TEST(test_pns_memo_matches_exact_gre);
    MU_RUN_TEST(test_pns_memo_matches_exact_epi);
    MU_RUN_TEST(test_pns_memo_matches_exact_fse);
    MU_RUN_TEST(test_pns_memo_matches_exact_mprage);
    MU_RUN_TEST(test_pns_memo_matches_exact_noncart);
    MU_RUN_TEST(test_pns_memo_matches_exact_short_window);
    MU_RUN_TEST(test_pns_memo_matches_exact_rotated);
    MU_RUN_TEST(test_pns_memo_matches_exact_radial);
    MU_RUN_TEST(test_pns_no_kernel_is_deterministic_exact_path);
}

/* ================================================================== */
/*  Entry point                                                       */
/* ================================================================== */

int test_safety_grad_main(void)
{
    minunit_run = 0;
    minunit_fail = 0;
    minunit_assert = 0;
    minunit_status = 0;
    minunit_real_timer = 0;
    minunit_proc_timer = 0;

    MU_RUN_SUITE(suite_grad_limits);
    MU_RUN_SUITE(suite_grad_continuity);
    MU_RUN_SUITE(suite_grad_canonical_sequence);
    MU_RUN_SUITE(suite_mech_resonances_safety);
    MU_RUN_SUITE(suite_pns_memoization);
    MU_REPORT();
    return MU_EXIT_CODE;
}
