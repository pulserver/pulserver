/*
 * test_safety_grad.c -- gradient safety tests.
 *
 * Suite A: gradient amplitude / slew-rate limit violations (4 tests).
 * Suite B: gradient continuity checks (17 tests).
 */
#include "test_helpers.h"

#include "pulseg_pns_models.h"

/* One-grid windows for the scan-window probes. */
static const double w10k[1] = {10000.0};
static const double w5k[1] = {5000.0};

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
    bands[0].max_amplitude_hz_per_m = 0.010f * GAMMA_HZ_PER_T; /* the tooth sustains 10.8 mT/m */
    bands[0].axis_mask = 0;

    /* Band 2: 2500–4500 Hz — covers 2nd harmonic */
    bands[1].freq_min_hz = 2500.0f;
    bands[1].freq_max_hz = 4500.0f;
    bands[1].max_amplitude_hz_per_m = 0.0f;
    bands[1].axis_mask = 0;

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
    bands[0].axis_mask = 0;

    bands[1].freq_min_hz = 2500.0f;
    bands[1].freq_max_hz = 4500.0f;
    bands[1].max_amplitude_hz_per_m = 1.0e9f;
    bands[1].axis_mask = 0;

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
    band.axis_mask = 0;

    run_mech_resonances_check("epi_2d_main.seq", 1, &band, 1 /* pass */);
}

/*
 * A band is read where it stands: this fixture's readout tooth sits at
 * 1243 Hz, 257 Hz below a band starting at 1500 Hz, and the window's own
 * 50 Hz width is the only widening there is, so neither band sees it: the
 * lower reads a 1.8 mT/m companion, the upper weak harmonics under 1 mT/m,
 * both far under a zero band's floor.
 */
MU_TEST(test_a_line_outside_a_band_is_not_refused)
{
    pulseg_forbidden_band band_lo, band_hi;

    band_lo.freq_min_hz = 1500.0f;
    band_lo.freq_max_hz = 2350.0f;
    band_lo.max_amplitude_hz_per_m = 0.0f;
    band_lo.axis_mask = 0;

    band_hi.freq_min_hz = 2600.0f;
    band_hi.freq_max_hz = 3450.0f;
    band_hi.max_amplitude_hz_per_m = 0.0f;
    band_hi.axis_mask = 0;

    run_mech_resonances_check("epi_2d_main.seq", 1, &band_lo, 1 /* pass */);
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
    MU_RUN_TEST(test_a_line_outside_a_band_is_not_refused);
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
static double pns_reference_scan_peak(
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

/* The score path's verdict against the ground truth, bracketing the true
 * scan peak from both sides. Below the peak the exact assembly must find
 * the violation the bound flagged; above it the bound either clears the
 * threshold outright or the assembly clears what it flagged. Either way the
 * verdict is the truth's, at one percent from the peak. */

/* The slide prices earlier blocks by how long ago they ended, so its
 * timeline has to be the scan's: block spans laid end to end, closing on
 * the descriptor's total duration. */

/* A loop runner that deals the range out in three uneven pieces, in
 * reverse: what a host's thread pool may do, without the threads. */
static void chunked_in_reverse(
    void *ctx,
    int count,
    void (*body)(void *arg, int begin, int end),
    void *arg)
{
    int a = count / 3, b = (2 * count) / 3;
    (void)ctx;
    body(arg, b, count);
    body(arg, a, b);
    body(arg, 0, a);
}

/* The exact-element score does not depend on how its loop is dealt out. */

/* A gradient that runs on across blocks is charged its slew, not a step
 * at every seam: zte_3d's readouts stay near full amplitude from block to
 * block, and the score prices them within a factor two of the scan. */

/* The acoustic tables are built one distinct waveform per slot, so dealing
 * the slots out to a pool changes nothing a candidate reads. */
MU_TEST(test_the_acoustic_candidates_are_the_same_however_the_loop_is_dealt)
{
    pulseg_collection *coll = NULL;
    pulseg_mech_resonances_spectra plain = PULSEG_MECH_RESONANCES_SPECTRA_INIT;
    pulseg_mech_resonances_spectra dealt = PULSEG_MECH_RESONANCES_SPECTRA_INIT;
    pulseg_mech_resonances_request request = PULSEG_MECH_RESONANCES_REQUEST_INIT;
    pulseg_forbidden_band bands[2];
    pulseg_opts sequential, pooled;
    int rc, i;

    mech_resonances_opts_init(&s_opts);
    rc = load_corpus_seq(&coll, "zte_3d.seq", &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_corpus_seq failed");
    bands[0].freq_min_hz = 550.0f;
    bands[0].freq_max_hz = 650.0f;
    bands[0].max_amplitude_hz_per_m = 1.0e12f;
    bands[0].axis_mask = 0;
    bands[1].freq_min_hz = 1100.0f;
    bands[1].freq_max_hz = 1250.0f;
    bands[1].max_amplitude_hz_per_m = 1.0e12f;
    bands[1].axis_mask = 0;
    request.bands.count = 2;
    request.bands.bands = bands;

    sequential = s_opts;
    sequential.parallel_for_fn = NULL;
    pooled = s_opts;
    pooled.parallel_for_fn = chunked_in_reverse;
    rc = pulseg_calc_mech_resonances(coll, &plain, &s_diag, NULL, &sequential, &request);
    mu_assert(PULSEG_SUCCEEDED(rc), "sequential acoustic analysis failed");
    rc = pulseg_calc_mech_resonances(coll, &dealt, &s_diag, NULL, &pooled, &request);
    mu_assert(PULSEG_SUCCEEDED(rc), "dealt acoustic analysis failed");

    mu_assert(plain.num_candidates == dealt.num_candidates, "candidate count differs");
    mu_assert(plain.num_candidates > 0, "no candidates on a fixture with forbidden bands");
    for (i = 0; i < plain.num_candidates; ++i)
    {
        mu_assert(
            plain.candidate_freqs[i] == dealt.candidate_freqs[i],
            "a candidate frequency moved");
        mu_assert(
            plain.candidate_grad_amps[i] == dealt.candidate_grad_amps[i],
            "a candidate amplitude moved");
        mu_assert(
            plain.candidate_violations[i] == dealt.candidate_violations[i],
            "a candidate verdict moved");
        if (plain.candidate_amps_gx && dealt.candidate_amps_gx)
            mu_assert(
                plain.candidate_amps_gx[i] == dealt.candidate_amps_gx[i],
                "an x amplitude moved");
        if (plain.candidate_amps_gy && dealt.candidate_amps_gy)
            mu_assert(
                plain.candidate_amps_gy[i] == dealt.candidate_amps_gy[i],
                "a y amplitude moved");
        if (plain.candidate_amps_gz && dealt.candidate_amps_gz)
            mu_assert(
                plain.candidate_amps_gz[i] == dealt.candidate_amps_gz[i],
                "a z amplitude moved");
    }
    pulseg_mech_resonances_spectra_free(&plain);
    pulseg_mech_resonances_spectra_free(&dealt);
    pulseg_collection_free(coll);
}

/* The scan-window amplitude at f by the criterion's own definition, from
 * the rendered scan: a window opens at every block start, takes whole the
 * blocks that start inside it, and the amplitude is sustained over the span
 * those blocks cover (never less than the window). What the probe computes
 * from event transforms, taken from the samples instead. */
static double scan_window_brute_force(
    pulseg_collection *coll,
    int axis,
    double f_hz,
    double window_us)
{
    pulseg_check_plan *plan = NULL;
    const pulseg__uniform_grad_waveforms *uw;
    const pulseg_sequence_descriptor *desc = &coll->descriptors[0];
    const float *g;
    double *starts, *ends;
    double best, omega, t, sre, sim, t0;
    int n, b, i, j, k, rc, s0, s1;

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
    {
        pulseg_check_plan_destroy(plan);
        return -1.0;
    }
    g = (axis == 0) ? uw->gx : (axis == 1) ? uw->gy : uw->gz;
    n = uw->num_samples;
    starts = (double *)PULSEG_ALLOC((size_t)desc->num_blocks * sizeof(double));
    ends = (double *)PULSEG_ALLOC((size_t)desc->num_blocks * sizeof(double));
    t = 0.0;
    for (b = 0; b < desc->num_blocks; ++b)
    {
        starts[b] = t;
        t += (double)desc->base_blocks[desc->block_table[b].id].duration_us;
        ends[b] = t;
    }
    omega = 2.0 * M_PI * f_hz * 1.0e-6 * (double)uw->raster_us;
    best = 0.0;
    for (i = 0; i < desc->num_blocks; ++i)
    {
        t0 = starts[i];
        j = i;
        while (j + 1 < desc->num_blocks && starts[j + 1] < t0 + window_us)
            ++j;
        s0 = (int)(t0 / (double)uw->raster_us);
        s1 = (int)(ends[j] / (double)uw->raster_us);
        if (s1 > n)
            s1 = n;
        sre = 0.0;
        sim = 0.0;
        for (k = s0; k < s1; ++k)
        {
            t = omega * (double)k;
            sre += (double)g[k] * cos(t);
            sim -= (double)g[k] * sin(t);
        }
        sre *= (double)uw->raster_us;
        sim *= (double)uw->raster_us;
        t = 2.0 / (window_us * 1.0e-6) * sqrt(sre * sre + sim * sim) * 1.0e-6;
        if (t > best)
            best = t;
    }
    PULSEG_FREE(starts);
    PULSEG_FREE(ends);
    pulseg_check_plan_destroy(plan);
    return best;
}

/* With every repetition the same and the window a whole number of TRs, the
 * scan-window amplitude at a TR harmonic is the periodic line amplitude
 * (2 / T_TR) |S_TR| -- the brute-force transform of the rendered scan over
 * that window says the same number. */
MU_TEST(test_the_scan_window_reduces_to_the_periodic_line_on_a_repeated_tr)
{
    pulseg_collection *coll = NULL;
    const pulseg_sequence_descriptor *desc;
    pulseg__mech_scan_grid grids[3];
    double window_us, t_tr_us, brute;
    double windows[3];
    float amp_x[3], amp_y[3], amp_z[3];
    int rc, k, reps;

    mech_resonances_opts_init(&s_opts);
    rc = load_corpus_seq(&coll, "gre_2d.seq", &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_corpus_seq failed");
    desc = &coll->descriptors[0];
    t_tr_us = (double)desc->tr_descriptor.tr_duration_us;
    reps = 4;
    window_us = (double)reps * t_tr_us - 0.5;
    windows[0] = windows[1] = windows[2] = window_us;
    for (k = 0; k < 3; ++k)
    {
        grids[k].f0_hz = (double)(k + 3) * 1.0e6 / t_tr_us;
        grids[k].df_hz = 0.0;
        grids[k].count = 1;
    }
    rc = pulseg__mech_scan_window_probe(
        desc,
        grids,
        3,
        windows,
        0,
        NULL,
        NULL,
        amp_x,
        amp_y,
        amp_z,
        NULL);
    mu_assert(PULSEG_SUCCEEDED(rc), "scan-window probe failed");
    for (k = 0; k < 3; ++k)
    {
        brute = scan_window_brute_force(coll, 0, grids[k].f0_hz, window_us);
        mu_assert(brute >= 0.0, "brute-force window evaluation failed");
        printf(
            "    scan window gre_2d harmonic %d: probe %.6g, rendered %.6g\n",
            k + 3,
            (double)amp_x[k],
            brute);
        mu_assert(
            fabs((double)amp_x[k] - brute) <= 0.02 * brute + 1.0e-9,
            "the scan window disagrees with the rendered scan at a TR harmonic");
    }
    pulseg_collection_free(coll);
}

/* On a scan whose repetitions differ, the same quantity from the samples. */
MU_TEST(test_the_scan_window_matches_the_rendered_scan_on_distinct_repetitions)
{
    pulseg_collection *coll = NULL;
    const pulseg_sequence_descriptor *desc;
    pulseg__mech_scan_grid grids[2];
    double window_us, t_tr_us, brute;
    double windows[3];
    float amp_x[2], amp_y[2], amp_z[2];
    int rc, k;

    mech_resonances_opts_init(&s_opts);
    rc = load_corpus_seq(&coll, "zte_3d.seq", &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_corpus_seq failed");
    desc = &coll->descriptors[0];
    t_tr_us = (double)desc->tr_descriptor.tr_duration_us;
    window_us = 3.0 * t_tr_us - 0.5;
    windows[0] = windows[1] = windows[2] = window_us;
    grids[0].f0_hz = 2.0e6 / t_tr_us;
    grids[1].f0_hz = 2.37e6 / t_tr_us;
    for (k = 0; k < 2; ++k)
    {
        grids[k].df_hz = 0.0;
        grids[k].count = 1;
    }
    rc = pulseg__mech_scan_window_probe(
        desc,
        grids,
        2,
        windows,
        0,
        NULL,
        NULL,
        amp_x,
        amp_y,
        amp_z,
        NULL);
    mu_assert(PULSEG_SUCCEEDED(rc), "scan-window probe failed");
    for (k = 0; k < 2; ++k)
    {
        brute = scan_window_brute_force(coll, 2, grids[k].f0_hz, window_us);
        mu_assert(brute >= 0.0, "brute-force window evaluation failed");
        printf(
            "    scan window zte_3d f=%.1f Hz: probe %.6g, rendered %.6g\n",
            grids[k].f0_hz,
            (double)amp_z[k],
            brute);
        mu_assert(
            fabs((double)amp_z[k] - brute) <= 0.03 * brute + 1.0e-9,
            "the scan window disagrees with the rendered scan");
    }
    pulseg_collection_free(coll);
}

/* On the written-out arms, the FFT records against the rendered scan. */
MU_TEST(test_the_scan_window_matches_the_rendered_scan_on_written_out_arms)
{
    pulseg_collection *coll = NULL;
    pulseg__mech_scan_grid grids[3];
    float amp_x[3], amp_y[3], amp_z[3];
    double brute, window_us = 10000.0;
    double windows[3] = {10000.0, 10000.0, 10000.0};
    int rc, k;

    mech_resonances_opts_init(&s_opts);
    rc = load_seq(&coll, "arms_scan.seq", &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");
    grids[0].f0_hz = 590.0;
    grids[1].f0_hz = 1180.0;
    grids[2].f0_hz = 613.7;
    for (k = 0; k < 3; ++k)
    {
        grids[k].df_hz = 0.0;
        grids[k].count = 1;
    }
    rc = pulseg__mech_scan_window_probe(
        &coll->descriptors[0],
        grids,
        3,
        windows,
        0,
        NULL,
        NULL,
        amp_x,
        amp_y,
        amp_z,
        NULL);
    mu_assert(PULSEG_SUCCEEDED(rc), "scan-window probe failed");
    for (k = 0; k < 3; ++k)
    {
        brute = scan_window_brute_force(coll, 1, grids[k].f0_hz, window_us);
        mu_assert(brute >= 0.0, "brute-force window evaluation failed");
        printf(
            "    scan window arms_scan f=%.1f Hz: probe %.6g, rendered %.6g\n",
            grids[k].f0_hz,
            (double)amp_y[k],
            brute);
        mu_assert(
            fabs((double)amp_y[k] - brute) <= 0.02 * brute + 1.0e-9,
            "the scan window disagrees with the rendered arms");
    }
    pulseg_collection_free(coll);
}

MU_TEST(test_the_scan_window_is_the_same_however_the_loop_is_dealt)
{
    pulseg_collection *coll = NULL;
    pulseg__mech_scan_grid grid;
    float a1[5], b1[5], c1[5], a2[5], b2[5], c2[5];
    int rc, k;

    mech_resonances_opts_init(&s_opts);
    rc = load_corpus_seq(&coll, "zte_3d.seq", &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_corpus_seq failed");
    grid.f0_hz = 550.0;
    grid.df_hz = 25.0;
    grid.count = 5;
    rc = pulseg__mech_scan_window_probe(
        &coll->descriptors[0],
        &grid,
        1,
        w10k,
        0,
        NULL,
        NULL,
        a1,
        b1,
        c1,
        NULL);
    mu_assert(PULSEG_SUCCEEDED(rc), "sequential probe failed");
    rc = pulseg__mech_scan_window_probe(
        &coll->descriptors[0],
        &grid,
        1,
        w10k,
        0,
        chunked_in_reverse,
        NULL,
        a2,
        b2,
        c2,
        NULL);
    mu_assert(PULSEG_SUCCEEDED(rc), "dealt probe failed");
    for (k = 0; k < 5; ++k)
        mu_assert(
            a1[k] == a2[k] && b1[k] == b2[k] && c1[k] == c2[k],
            "a scan-window amplitude depends on how the loop was dealt");
    pulseg_collection_free(coll);
}

/* On a scan of long arbitrary waveforms, every waveform's FFT record
 * interpolated onto the fine grid against every transform summed directly:
 * never below it, and within the guard of it. */
MU_TEST(test_the_arm_spectrum_by_fft_matches_the_direct_transform)
{
    pulseg_collection *coll = NULL;
    pulseg__mech_scan_grid grid;
    float *f[3], *d[3];
    double worst = 0.0, worst_below = 0.0;
    int rc, k, ax, n = 401;

    mech_resonances_opts_init(&s_opts);
    rc = load_seq(&coll, "arms_scan.seq", &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");
    grid.f0_hz = 550.0;
    grid.df_hz = 0.25;
    grid.count = n;
    for (ax = 0; ax < 3; ++ax)
    {
        f[ax] = (float *)malloc((size_t)n * sizeof(float));
        d[ax] = (float *)malloc((size_t)n * sizeof(float));
    }
    rc = pulseg__mech_scan_window_probe(
        &coll->descriptors[0],
        &grid,
        1,
        w10k,
        0,
        NULL,
        NULL,
        f[0],
        f[1],
        f[2],
        NULL);
    mu_assert(PULSEG_SUCCEEDED(rc), "FFT-record probe failed");
    rc = pulseg__mech_scan_window_probe(
        &coll->descriptors[0],
        &grid,
        1,
        w10k,
        PULSEG__MECH_SCAN_DIRECT,
        NULL,
        NULL,
        d[0],
        d[1],
        d[2],
        NULL);
    mu_assert(PULSEG_SUCCEEDED(rc), "direct probe failed");
    for (ax = 0; ax < 3; ++ax)
        for (k = 0; k < n; ++k)
        {
            double a = (double)f[ax][k], b = (double)d[ax][k];
            double rel = fabs(a - b) / (b + 1.0e-12);
            double below = (b - a) / (b + 1.0e-12);
            if (rel > worst)
                worst = rel;
            if (below > worst_below)
                worst_below = below;
        }
    printf(
        "    arm spectrum by FFT vs direct on arms_scan: worst |rel| %.2e, worst below %.2e\n",
        worst,
        worst_below);
    mu_assert(worst > 0.0, "the FFT record path was not exercised");
    mu_assert(worst <= 2.0e-3, "the interpolated spectrum strays from the direct transform");
    mu_assert(worst_below <= 1.0e-5, "the interpolated spectrum falls below the direct transform");
    for (ax = 0; ax < 3; ++ax)
    {
        free(f[ax]);
        free(d[ax]);
    }
    pulseg_collection_free(coll);
}

/* The kernel's truncation constant: the sup over the support a 2x
 * oversampled record can have (|t| <= 1/4 of a bin period) and every
 * fractional position of |sum over the taps of phi(x_j) e^{i 2 pi x_j t} - 1|,
 * the factor by which interpolation from the taps departs from the exact
 * transform, per unit of the waveform's L1. */
MU_TEST(test_the_scan_window_kernel_constant_bounds_its_truncation)
{
    const int nt = 201, nf = 64, K = 8;
    double worst = 0.0;
    int it, ifr, tap;

    for (it = 0; it < nt; ++it)
    {
        double t = -0.25 + 0.5 * (double)it / (double)(nt - 1);
        for (ifr = 0; ifr < nf; ++ifr)
        {
            double frac = (double)ifr / (double)nf;
            double re = 0.0, im = 0.0, err;
            int jl = -K + 1;
            for (tap = 0; tap < 2 * K; ++tap)
            {
                double x = frac - (double)(jl + tap);
                double w = pulseg__mech_scan_kernel(x);
                re += w * cos(2.0 * M_PI * x * t);
                im += w * sin(2.0 * M_PI * x * t);
            }
            err = sqrt((re - 1.0) * (re - 1.0) + im * im);
            if (err > worst)
                worst = err;
        }
    }
    printf(
        "    scan window kernel truncation: sup %.3e, pinned %.3e\n",
        worst,
        pulseg__mech_scan_kernel_e());
    mu_assert(pulseg__mech_scan_kernel(0.0) == 1.0, "the kernel is not 1 at its centre");
    mu_assert(pulseg__mech_scan_kernel(8.0) == 0.0, "the kernel does not vanish at its half-width");
    mu_assert(
        worst <= pulseg__mech_scan_kernel_e(),
        "the pinned kernel constant is below its truncation error");
}

/* Probing two grids together reads what probing each alone reads: the
 * records and the window pass are shared, the readings are not mixed. */
MU_TEST(test_each_band_is_probed_with_its_own_window_as_if_alone)
{
    pulseg_collection *coll = NULL;
    pulseg__mech_scan_grid grids[2];
    double windows[2];
    double spans[2], span_one[1];
    float together[3][37], alone[3][21], alone2[3][16];
    int rc, k, ax;

    mech_resonances_opts_init(&s_opts);
    rc = load_seq(&coll, "arms_scan.seq", &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");
    windows[0] = 10000.0;
    windows[1] = 10000.0;
    grids[0].f0_hz = 550.0;
    grids[0].df_hz = 5.0;
    grids[0].count = 21;
    grids[1].f0_hz = 1100.0;
    grids[1].df_hz = 10.0;
    grids[1].count = 16;
    rc = pulseg__mech_scan_window_probe(
        &coll->descriptors[0],
        grids,
        2,
        windows,
        0,
        NULL,
        NULL,
        together[0],
        together[1],
        together[2],
        spans);
    mu_assert(PULSEG_SUCCEEDED(rc), "probe of both grids failed");
    rc = pulseg__mech_scan_window_probe(
        &coll->descriptors[0],
        &grids[0],
        1,
        &windows[0],
        0,
        NULL,
        NULL,
        alone[0],
        alone[1],
        alone[2],
        span_one);
    mu_assert(PULSEG_SUCCEEDED(rc), "probe of the first grid failed");
    mu_assert(span_one[0] == spans[0], "the first grid's span is its own");
    for (ax = 0; ax < 3; ++ax)
        for (k = 0; k < 21; ++k)
            mu_assert(
                together[ax][k] == alone[ax][k],
                "first grid differs when probed with another");
    rc = pulseg__mech_scan_window_probe(
        &coll->descriptors[0],
        &grids[1],
        1,
        &windows[1],
        0,
        NULL,
        NULL,
        alone2[0],
        alone2[1],
        alone2[2],
        span_one);
    mu_assert(PULSEG_SUCCEEDED(rc), "probe of the second grid failed");
    mu_assert(span_one[0] == spans[1], "the second grid's span is its own");
    for (ax = 0; ax < 3; ++ax)
        for (k = 0; k < 16; ++k)
            mu_assert(
                together[ax][21 + k] == alone2[ax][k],
                "second grid differs when probed with another");
    pulseg_collection_free(coll);
}

/* The prescription is the scanner's own rotation. With x carried onto z, a
 * band watched on z alone sees the arms the scan plays on x, an identity
 * prescription is no prescription, and the descriptor is handed back as it
 * was. */
MU_TEST(test_the_prescription_carries_the_drive_onto_the_axis_the_band_watches)
{
    static const float swap_xz[9] = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    static const float identity[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    pulseg_collection *coll = NULL;
    pulseg_forbidden_band_list list = PULSEG_FORBIDDEN_BAND_LIST_INIT;
    pulseg_forbidden_band band;
    pulseg_opts oblique;
    int rc, n_rot, id0, plain_rc;

    band.freq_min_hz = 550.0f;
    band.freq_max_hz = 650.0f;
    band.max_amplitude_hz_per_m = 1000.0f;
    band.axis_mask = 4;
    mech_resonances_opts_init(&s_opts);
    rc = load_seq(&coll, "arms_scan.seq", &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");
    list.count = 1;
    list.bands = &band;
    n_rot = coll->descriptors[0].num_rotations;
    id0 = coll->descriptors[0].block_table[0].rotation_id;
    pulseg_diagnostic_init(&s_diag);
    plain_rc = pulseg_check_mech_resonances(coll, &s_diag, NULL, &s_opts, &list);
    mu_assert(PULSEG_SUCCEEDED(plain_rc), "in the design frame nothing plays on z");
    oblique = s_opts;
    memcpy(oblique.prescription_rotation, swap_xz, sizeof(swap_xz));
    oblique.has_prescription_rotation = 1;
    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_check_mech_resonances(coll, &s_diag, NULL, &oblique, &list);
    mu_assert_int_eq(PULSEG_ERR_MECH_RESONANCES_VIOLATION, rc);
    mu_assert(strstr(s_diag.message, "ax=2") != NULL, "the violation is on physical z");
    mu_assert(coll->descriptors[0].num_rotations == n_rot, "the rotation table is handed back");
    mu_assert(
        coll->descriptors[0].block_table[0].rotation_id == id0,
        "the block ids are handed back");
    mu_assert(coll->descriptors[0].prescription_depth == 0, "no frame stays installed");
    memcpy(oblique.prescription_rotation, identity, sizeof(identity));
    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_check_mech_resonances(coll, &s_diag, NULL, &oblique, &list);
    mu_assert_int_eq(plain_rc, rc);
    pulseg_collection_free(coll);
}

/* A band names the physical axes it applies to: the axis that drives the
 * band refuses the scan alone or unnamed, and an axis the arms never play
 * cannot refuse it. */
MU_TEST(test_a_band_refuses_only_on_the_axes_it_names)
{
    pulseg_collection *coll = NULL;
    pulseg_forbidden_band_list list = PULSEG_FORBIDDEN_BAND_LIST_INIT;
    pulseg_forbidden_band band;
    const char *p;
    int rc, ax;

    band.freq_min_hz = 550.0f;
    band.freq_max_hz = 650.0f;
    band.max_amplitude_hz_per_m = 1000.0f;
    band.axis_mask = 0;
    mech_resonances_opts_init(&s_opts);
    rc = load_seq(&coll, "arms_scan.seq", &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");
    list.count = 1;
    list.bands = &band;
    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_check_mech_resonances(coll, &s_diag, NULL, &s_opts, &list);
    mu_assert_int_eq(PULSEG_ERR_MECH_RESONANCES_VIOLATION, rc);
    p = strstr(s_diag.message, "ax=");
    mu_assert(p != NULL, "the diagnostic names the offending axis");
    ax = p[3] - '0';
    mu_assert(ax == 0 || ax == 1, "the arms play on x and y");
    band.axis_mask = 1 << ax;
    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_check_mech_resonances(coll, &s_diag, NULL, &s_opts, &list);
    mu_assert_int_eq(PULSEG_ERR_MECH_RESONANCES_VIOLATION, rc);
    band.axis_mask = 4;
    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_check_mech_resonances(coll, &s_diag, NULL, &s_opts, &list);
    mu_assert(PULSEG_SUCCEEDED(rc), "a band on z cannot refuse arms played on x and y");
    pulseg_collection_free(coll);
}

/* A raw shape pointed at in the descriptor gives the same amplitudes as the
 * same shape copied: the probe with records (borrowing) against the direct
 * probe (copying) on a scan whose long waveforms are stored raw. */
MU_TEST(test_a_borrowed_waveform_is_the_copied_one)
{
    pulseg_collection *coll = NULL;
    pulseg__mech_scan_grid grid;
    float f[3][9], d[3][9];
    int rc, k, ax, raw = 0, s;

    mech_resonances_opts_init(&s_opts);
    rc = load_seq(&coll, "arms_scan.seq", &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");
    for (s = 0; s < coll->descriptors[0].num_shapes; ++s)
    {
        const pulseq_shape *sh = &coll->descriptors[0].shapes[s];
        if (sh->num_samples == sh->num_uncompressed_samples && sh->num_uncompressed_samples >= 64)
            ++raw;
    }
    printf("    arms_scan raw shapes of 64+ samples: %d\n", raw);
    mu_assert(raw >= 16, "the arms are not stored raw");
    grid.f0_hz = 400.0;
    grid.df_hz = 50.0;
    grid.count = 9;
    rc = pulseg__mech_scan_window_probe(
        &coll->descriptors[0],
        &grid,
        1,
        w5k,
        0,
        NULL,
        NULL,
        f[0],
        f[1],
        f[2],
        NULL);
    mu_assert(PULSEG_SUCCEEDED(rc), "borrowing probe failed");
    rc = pulseg__mech_scan_window_probe(
        &coll->descriptors[0],
        &grid,
        1,
        w5k,
        PULSEG__MECH_SCAN_DIRECT,
        NULL,
        NULL,
        d[0],
        d[1],
        d[2],
        NULL);
    mu_assert(PULSEG_SUCCEEDED(rc), "copying probe failed");
    for (ax = 0; ax < 3; ++ax)
        for (k = 0; k < 9; ++k)
            mu_assert(
                fabs((double)f[ax][k] - (double)d[ax][k]) <= 2.0e-3 * (double)d[ax][k] + 1.0e-9,
                "a borrowed waveform is priced differently from its copy");
    pulseg_collection_free(coll);
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

/* The exact scan against the scan convolved whole by the model's own
 * evaluator: the same peak, however the chunks are dealt. */
static void run_pns_exact_scan_equals_reference(const char *filename, int corpus)
{
    pulseg_collection *coll = NULL;
    pulseg_check_plan *plan = NULL;
    pulseg_pns_irnich ctx;
    pulseg_pns_model model;
    double reference, exact, dealt;
    int rc, block, block2;

    mech_resonances_opts_init(&s_opts);
    rc = corpus ? load_corpus_seq(&coll, filename, &s_opts) : load_seq(&coll, filename, &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load failed");
    pulseg_pns_irnich_init(&model, &ctx, 360.0f, 4.25e8f, 0.333f);
    reference = pns_reference_scan_peak(coll, &model, s_opts.gamma_hz_per_t);
    mu_assert(reference > 0.0, "reference peak failed");
    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_check_plan_create(&plan, &s_diag, coll, NULL);
    mu_assert(PULSEG_SUCCEEDED(rc), "plan failed");
    rc = pulseg__pns_exact_scan_peak(
        plan,
        &s_diag,
        &coll->descriptors[0],
        0,
        &model,
        NULL,
        NULL,
        s_opts.gamma_hz_per_t,
        &exact,
        &block);
    mu_assert(PULSEG_SUCCEEDED(rc), "exact scan failed");
    rc = pulseg__pns_exact_scan_peak(
        plan,
        &s_diag,
        &coll->descriptors[0],
        0,
        &model,
        chunked_in_reverse,
        NULL,
        s_opts.gamma_hz_per_t,
        &dealt,
        &block2);
    mu_assert(PULSEG_SUCCEEDED(rc), "dealt exact scan failed");
    printf(
        "    exact scan %s: reference %.6g%%, exact %.6g%% (block %d)\n",
        filename,
        reference,
        exact,
        block);
    mu_assert(
        exact == dealt && block == block2,
        "the exact scan depends on how the chunks are dealt");
    mu_assert(
        exact >= reference * (1.0 - 1.0e-4),
        "the exact scan falls below the scan convolved whole");
    mu_assert(
        exact <= reference * (1.0 + 3.0e-3),
        "the exact scan strays above the scan convolved whole");
    pulseg_check_plan_destroy(plan);
    pulseg_collection_free(coll);
}

MU_TEST(test_the_exact_scan_equals_the_scan_convolved_whole_gre)
{
    run_pns_exact_scan_equals_reference("gre_2d.seq", 1);
}
MU_TEST(test_the_exact_scan_equals_the_scan_convolved_whole_epi)
{
    run_pns_exact_scan_equals_reference("epi_2d.seq", 1);
}
MU_TEST(test_the_exact_scan_equals_the_scan_convolved_whole_fse)
{
    run_pns_exact_scan_equals_reference("fse_2d.seq", 1);
}
MU_TEST(test_the_exact_scan_equals_the_scan_convolved_whole_spiral)
{
    run_pns_exact_scan_equals_reference("gre_spiral_2d.seq", 1);
}
MU_TEST(test_the_exact_scan_equals_the_scan_convolved_whole_radial)
{
    run_pns_exact_scan_equals_reference("gre_radial_2d.seq", 1);
}
MU_TEST(test_the_exact_scan_equals_the_scan_convolved_whole_zte)
{
    run_pns_exact_scan_equals_reference("zte_3d.seq", 1);
}
MU_TEST(test_the_exact_scan_equals_the_scan_convolved_whole_written_out_arms)
{
    run_pns_exact_scan_equals_reference("arms_scan.seq", 0);
}

MU_TEST_SUITE(suite_pns_memoization)
{
    MU_RUN_TEST(test_the_exact_scan_equals_the_scan_convolved_whole_gre);
    MU_RUN_TEST(test_the_exact_scan_equals_the_scan_convolved_whole_epi);
    MU_RUN_TEST(test_the_exact_scan_equals_the_scan_convolved_whole_fse);
    MU_RUN_TEST(test_the_exact_scan_equals_the_scan_convolved_whole_spiral);
    MU_RUN_TEST(test_the_exact_scan_equals_the_scan_convolved_whole_radial);
    MU_RUN_TEST(test_the_exact_scan_equals_the_scan_convolved_whole_zte);
    MU_RUN_TEST(test_the_exact_scan_equals_the_scan_convolved_whole_written_out_arms);
    MU_RUN_TEST(test_shipped_irnich_matches_an_independent_implementation);
    MU_RUN_TEST(test_the_response_does_not_move_when_the_raster_is_halved);
    MU_RUN_TEST(test_the_acoustic_candidates_are_the_same_however_the_loop_is_dealt);
    MU_RUN_TEST(test_the_scan_window_reduces_to_the_periodic_line_on_a_repeated_tr);
    MU_RUN_TEST(test_the_scan_window_matches_the_rendered_scan_on_distinct_repetitions);
    MU_RUN_TEST(test_the_scan_window_is_the_same_however_the_loop_is_dealt);
    MU_RUN_TEST(test_the_scan_window_matches_the_rendered_scan_on_written_out_arms);
    MU_RUN_TEST(test_the_arm_spectrum_by_fft_matches_the_direct_transform);
    MU_RUN_TEST(test_the_scan_window_kernel_constant_bounds_its_truncation);
    MU_RUN_TEST(test_a_borrowed_waveform_is_the_copied_one);
    MU_RUN_TEST(test_each_band_is_probed_with_its_own_window_as_if_alone);
    MU_RUN_TEST(test_a_band_refuses_only_on_the_axes_it_names);
    MU_RUN_TEST(test_the_prescription_carries_the_drive_onto_the_axis_the_band_watches);
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
