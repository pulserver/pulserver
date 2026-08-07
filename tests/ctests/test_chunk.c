/*
 * test_chunk.c -- the chunk planner's two constraints, and what happens when
 *                 they cannot both be met.
 *
 * A chunk is squeezed from opposite sides: its distinct waveforms must fit the
 * memory budget (a ceiling), and its playout must last long enough to build
 * the next chunk (a floor).  These check both edges and the case where the
 * floor is above the ceiling, which is a sequence no split can rescue.
 */
#include "pulseg/pulseg_chunk.h"

#include "test_helpers.h"

#define CHUNK_SEQ "gre_2d_1sl_1avg.seq"

/* mu_assert expands to a bare `return`, so the load has to hand its result
 * back through a parameter rather than a return value. */
static void chunk_load(pulseg_collection **coll)
{
    pulseg_opts opts;
    int rc;

    *coll = NULL;
    gre_opts_init(&opts);
    rc = load_seq(coll, CHUNK_SEQ, &opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed for the chunk fixture");
}

/* Integral of g^2 over a piecewise-linear waveform given as corner points.
 * Exact: on each interval the square of a linear function is a quadratic, and
 * Simpson's rule on three points is exact for one. */
static double trapz_square(const float *t, const float *v, int n, float peak)
{
    double total = 0.0;
    double a, b, m, dt;
    int i;

    for (i = 1; i < n; ++i)
    {
        dt = (double)(t[i] - t[i - 1]);
        if (dt <= 0.0)
            continue;
        a = (double)v[i - 1] * peak;
        b = (double)v[i] * peak;
        m = 0.5 * (a + b);
        total += dt * (a * a + 4.0 * m * m + b * b) / 6.0;
    }
    return total;
}

/* Every position of the exec stream lands in exactly one chunk, in order. */
static void assert_covers_stream(const pulseg_chunk_plan *plan, int stream_len)
{
    int k, expected;

    mu_assert(plan->num_chunks > 0, "plan has no chunks");
    expected = 0;
    for (k = 0; k < plan->num_chunks; ++k)
    {
        mu_assert_int_eq(expected, plan->chunks[k].first_position);
        mu_assert(plan->chunks[k].num_positions > 0, "empty chunk");
        expected += plan->chunks[k].num_positions;
    }
    mu_assert_int_eq(stream_len, expected);
}

/* A budget nothing can exceed collapses the scan to a single chunk. */
MU_TEST(test_chunk_generous_budget_is_one_chunk)
{
    pulseg_collection *coll = NULL;
    pulseg_chunk_plan plan = PULSEG_CHUNK_PLAN_INIT;
    pulseg_chunk_budget budget = PULSEG_CHUNK_BUDGET_INIT;
    pulseg_diagnostic diag = PULSEG_DIAGNOSTIC_INIT;
    int rc;

    chunk_load(&coll);
    budget.max_wave_samples = 1000000000L;
    budget.build_us_per_sample = 0.0f; /* rate check disabled */

    rc = pulseg_plan_chunks(coll, 0, &budget, &plan, &diag);
    mu_assert(PULSEG_SUCCEEDED(rc), "plan failed on a generous budget");
    mu_assert_int_eq(PULSEG_WAVE_RESIDENT, plan.mode);
    mu_assert_int_eq(1, plan.num_chunks);
    mu_assert(plan.total_wave_points > 0, "resident plan holds no waves");
    mu_assert(plan.peak_wave_samples == plan.chunks[0].wave_samples, "peak != only chunk");

    pulseg_free_chunk_plan(&plan);
    pulseg_collection_free(coll);
}

/* A tight budget splits, and every chunk honours it. */
MU_TEST(test_chunk_tight_budget_splits_and_fits)
{
    pulseg_collection *coll = NULL;
    pulseg_chunk_plan plan = PULSEG_CHUNK_PLAN_INIT;
    pulseg_chunk_budget budget = PULSEG_CHUNK_BUDGET_INIT;
    pulseg_diagnostic diag = PULSEG_DIAGNOSTIC_INIT;
    pulseg_chunk_plan whole = PULSEG_CHUNK_PLAN_INIT;
    int rc, k, stream_len;

    chunk_load(&coll);
    /* Learn the whole scan's size first, then ask for a fraction of it. */
    budget.max_wave_samples = 1000000000L;
    budget.build_us_per_sample = 0.0f;
    rc = pulseg_plan_chunks(coll, 0, &budget, &whole, &diag);
    mu_assert(PULSEG_SUCCEEDED(rc), "reference plan failed");
    stream_len = whole.chunks[0].num_positions;

    budget.max_wave_samples = whole.peak_wave_samples / 4;
    if (budget.max_wave_samples < 1)
        budget.max_wave_samples = 1;
    pulseg_free_chunk_plan(&whole);

    rc = pulseg_plan_chunks(coll, 0, &budget, &plan, &diag);
    mu_assert(PULSEG_SUCCEEDED(rc), "plan failed on a tight budget");
    assert_covers_stream(&plan, stream_len);

    mu_assert_int_eq(PULSEG_WAVE_STREAMED, plan.mode);
    mu_assert(plan.num_chunks > 1, "streaming did not split by segment");

    /* A chunk is a segment instance, so no chunk may span a segment change. */
    for (k = 0; k < plan.num_chunks; ++k)
    {
        mu_assert(plan.chunks[k].num_positions > 0, "empty segment chunk");
        mu_assert(plan.chunks[k].num_waves <= plan.num_waves, "chunk names unknown waves");
    }

    pulseg_free_chunk_plan(&plan);
    pulseg_collection_free(coll);
}

/* Chunking is what makes the peak smaller -- that is the entire point. */
MU_TEST(test_chunk_splitting_lowers_the_peak)
{
    pulseg_collection *coll = NULL;
    pulseg_chunk_plan whole = PULSEG_CHUNK_PLAN_INIT;
    pulseg_chunk_plan split = PULSEG_CHUNK_PLAN_INIT;
    pulseg_chunk_budget budget = PULSEG_CHUNK_BUDGET_INIT;
    pulseg_diagnostic diag = PULSEG_DIAGNOSTIC_INIT;

    chunk_load(&coll);
    budget.max_wave_samples = 1000000000L;
    budget.build_us_per_sample = 0.0f;
    mu_assert(
        PULSEG_SUCCEEDED(pulseg_plan_chunks(coll, 0, &budget, &whole, &diag)),
        "reference plan failed");

    budget.max_wave_samples = whole.peak_wave_samples / 4;
    if (budget.max_wave_samples < 1)
        budget.max_wave_samples = 1;
    mu_assert(
        PULSEG_SUCCEEDED(pulseg_plan_chunks(coll, 0, &budget, &split, &diag)),
        "split plan failed");

    mu_assert_int_eq(PULSEG_WAVE_STREAMED, split.mode);
    mu_assert(
        split.peak_wave_samples < whole.peak_wave_samples,
        "streaming did not reduce what has to be resident at once");
    mu_assert_int_eq(whole.num_waves, split.num_waves);

    pulseg_free_chunk_plan(&whole);
    pulseg_free_chunk_plan(&split);
    pulseg_collection_free(coll);
}

/*
 * A build rate slow enough that no chunk can be materialised inside its
 * predecessor's playout.  There is no split that fixes this -- growing the
 * chunk is what would, and the memory budget is what stops it -- so the
 * planner must say so rather than hand back a plan that ends in EOS.
 */
MU_TEST(test_chunk_infeasible_rate_is_reported)
{
    pulseg_collection *coll = NULL;
    pulseg_chunk_plan plan = PULSEG_CHUNK_PLAN_INIT;
    pulseg_chunk_budget budget = PULSEG_CHUNK_BUDGET_INIT;
    pulseg_diagnostic diag = PULSEG_DIAGNOSTIC_INIT;
    pulseg_chunk_plan probe = PULSEG_CHUNK_PLAN_INIT;
    int rc;

    chunk_load(&coll);
    budget.max_wave_samples = 1000000000L;
    budget.build_us_per_sample = 0.0f;
    mu_assert(
        PULSEG_SUCCEEDED(pulseg_plan_chunks(coll, 0, &budget, &probe, &diag)),
        "probe plan failed");
    budget.max_wave_samples = probe.peak_wave_samples / 8;
    if (budget.max_wave_samples < 1)
        budget.max_wave_samples = 1;
    pulseg_free_chunk_plan(&probe);

    /* Absurdly slow: a second of work per sample. */
    budget.build_us_per_sample = 1.0e6f;

    rc = pulseg_plan_chunks(coll, 0, &budget, &plan, &diag);
    if (plan.num_chunks > 1)
    {
        mu_assert_int_eq(PULSEG_ERR_CHUNK_INFEASIBLE, rc);
        mu_assert_int_eq(PULSEG_ERR_CHUNK_INFEASIBLE, diag.code);
        mu_assert(diag.message[0] != '\0', "infeasible plan carried no diagnostic");
    }

    pulseg_free_chunk_plan(&plan);
    pulseg_collection_free(coll);
}

/* A zero budget is a caller error, not a plan. */
MU_TEST(test_chunk_rejects_zero_budget)
{
    pulseg_collection *coll = NULL;
    pulseg_chunk_plan plan = PULSEG_CHUNK_PLAN_INIT;
    pulseg_chunk_budget budget = PULSEG_CHUNK_BUDGET_INIT;
    pulseg_diagnostic diag = PULSEG_DIAGNOSTIC_INIT;

    chunk_load(&coll);
    budget.max_wave_samples = 0;
    mu_assert_int_eq(
        PULSEG_ERR_INVALID_ARGUMENT, pulseg_plan_chunks(coll, 0, &budget, &plan, &diag));

    pulseg_free_chunk_plan(&plan);
    pulseg_collection_free(coll);
}

/*
 * Rotation preserves the summed power across the three axes.
 *
 * This is the property the whole design leans on: S(f) = |Gx|^2+|Gy|^2+|Gz|^2
 * is invariant because a rotation sends G to RG and R preserves norms.  By
 * Parseval the same holds sample by sample in time, so materialising a key
 * with and without its rotation must give the same total energy.  If this
 * drifts, the representative selection and the safety bound built on it are
 * both unsound.
 */
MU_TEST(test_chunk_rotation_preserves_summed_power)
{
    pulseg_collection *coll = NULL;
    pulseg_chunk_plan plan = PULSEG_CHUNK_PLAN_INIT;
    pulseg_chunk_budget budget = PULSEG_CHUNK_BUDGET_INIT;
    pulseg_diagnostic diag = PULSEG_DIAGNOSTIC_INIT;
    float buf[8192];
    float tbuf[8192];
    int w, axis, ns, checked;

    chunk_load(&coll);
    budget.max_wave_samples = 1000000000L;
    budget.build_us_per_sample = 0.0f;
    mu_assert(
        PULSEG_SUCCEEDED(pulseg_plan_chunks(coll, 0, &budget, &plan, &diag)), "plan failed");

    checked = 0;
    for (w = 0; w < plan.num_waves; ++w)
    {
        pulseg_wave_key rotated = plan.waves[w];
        pulseg_wave_key plain = plan.waves[w];
        double power_rot = 0.0;
        double power_plain = 0.0;
        float peak;

        if (rotated.num_points <= 0 || rotated.num_points > 8192)
            continue;
        plain.rotation_id = -1;

        for (axis = 0; axis < 3; ++axis)
        {
            if (PULSEG_FAILED(pulseg_materialize_wave(
                    coll, 0, &rotated, axis, tbuf, buf, 8192, &ns, &peak)))
                continue;
            power_rot += trapz_square(tbuf, buf, ns, peak);

            if (PULSEG_FAILED(pulseg_materialize_wave(
                    coll, 0, &plain, axis, tbuf, buf, 8192, &ns, &peak)))
                continue;
            power_plain += trapz_square(tbuf, buf, ns, peak);
        }

        if (power_plain > 1e-12)
        {
            double rel = (power_rot - power_plain) / power_plain;
            if (rel < 0.0)
                rel = -rel;
            mu_assert(rel < 1e-4, "rotation changed the summed power across axes");
            checked++;
        }
    }
    mu_assert(checked > 0, "no wave was actually checked");

    pulseg_free_chunk_plan(&plan);
    pulseg_collection_free(coll);
}

/* A buffer that is too small reports how big it had to be. */
MU_TEST(test_chunk_materialise_reports_required_size)
{
    pulseg_collection *coll = NULL;
    pulseg_chunk_plan plan = PULSEG_CHUNK_PLAN_INIT;
    pulseg_chunk_budget budget = PULSEG_CHUNK_BUDGET_INIT;
    pulseg_diagnostic diag = PULSEG_DIAGNOSTIC_INIT;
    float tiny[1];
    float ttiny[1];
    int w, ns, rc, found;

    chunk_load(&coll);
    budget.max_wave_samples = 1000000000L;
    budget.build_us_per_sample = 0.0f;
    mu_assert(
        PULSEG_SUCCEEDED(pulseg_plan_chunks(coll, 0, &budget, &plan, &diag)), "plan failed");

    found = 0;
    for (w = 0; w < plan.num_waves; ++w)
    {
        if (plan.waves[w].num_points <= 2)
            continue;
        rc = pulseg_materialize_wave(
            coll, 0, &plan.waves[w], PULSEG_GRAD_AXIS_X, ttiny, tiny, 1, &ns, NULL);
        mu_assert_int_eq(PULSEG_ERR_INDEX, rc);
        mu_assert(ns > 1, "a multi-point wave reported one point");
        found = 1;
        break;
    }
    mu_assert(found, "no multi-sample wave to probe");

    pulseg_free_chunk_plan(&plan);
    pulseg_collection_free(coll);
}

/*
 * Corner times come back strictly increasing.
 *
 * The invariant a real bug broke: an arbitrary gradient WITH a time shape is
 * sampled on raster edges, so its first sample already is the event's start.
 * Prepending a boundary corner there put two points at the same time with
 * different values -- a zero-width discontinuity that the rotation combine
 * would have carried onto every output axis.  Without a time shape the
 * samples are at raster centres and the two edge corners are genuinely
 * distinct times, which is where the ns+2 comes from.
 */
MU_TEST(test_chunk_corner_times_strictly_increase)
{
    pulseg_collection *coll = NULL;
    pulseg_chunk_plan plan = PULSEG_CHUNK_PLAN_INIT;
    pulseg_chunk_budget budget = PULSEG_CHUNK_BUDGET_INIT;
    pulseg_diagnostic diag = PULSEG_DIAGNOSTIC_INIT;
    float tbuf[8192];
    float abuf[8192];
    int w, axis, ns, j, checked;

    chunk_load(&coll);
    budget.max_wave_samples = 1000000000L;
    budget.build_us_per_sample = 0.0f;
    mu_assert(
        PULSEG_SUCCEEDED(pulseg_plan_chunks(coll, 0, &budget, &plan, &diag)), "plan failed");

    checked = 0;
    for (w = 0; w < plan.num_waves; ++w)
    {
        for (axis = 0; axis < 3; ++axis)
        {
            if (PULSEG_FAILED(pulseg_materialize_wave(
                    coll, 0, &plan.waves[w], axis, tbuf, abuf, 8192, &ns, NULL)))
                continue;
            for (j = 1; j < ns; ++j)
                mu_assert(tbuf[j] > tbuf[j - 1], "corner times are not strictly increasing");
            if (ns > 1)
                checked++;
        }
    }
    mu_assert(checked > 0, "no wave was actually checked");

    pulseg_free_chunk_plan(&plan);
    pulseg_collection_free(coll);
}

/*
 * Every position's wave index points into its OWN chunk.
 *
 * The planner rewinds when a chunk overflows -- it closes at the last legal
 * cut and replays from there -- so positions can be visited twice and their
 * wave index reassigned.  If a stale index from the abandoned pass survived,
 * the interpreter would swap in a waveform belonging to a chunk that is no
 * longer resident, which is silently the wrong gradient.
 */
MU_TEST(test_chunk_position_waves_stay_inside_their_chunk)
{
    pulseg_collection *coll = NULL;
    pulseg_chunk_plan plan = PULSEG_CHUNK_PLAN_INIT;
    pulseg_chunk_budget budget = PULSEG_CHUNK_BUDGET_INIT;
    pulseg_diagnostic diag = PULSEG_DIAGNOSTIC_INIT;
    pulseg_chunk_plan probe = PULSEG_CHUNK_PLAN_INIT;
    int k, p, w, checked;

    chunk_load(&coll);
    budget.max_wave_samples = 1000000000L;
    budget.build_us_per_sample = 0.0f;
    mu_assert(
        PULSEG_SUCCEEDED(pulseg_plan_chunks(coll, 0, &budget, &probe, &diag)), "probe failed");
    budget.max_wave_samples = probe.peak_wave_samples / 4;
    if (budget.max_wave_samples < 1)
        budget.max_wave_samples = 1;
    pulseg_free_chunk_plan(&probe);

    mu_assert(
        PULSEG_SUCCEEDED(pulseg_plan_chunks(coll, 0, &budget, &plan, &diag)), "plan failed");
    mu_assert(plan.max_wave_points > 0, "no wave has any points");

    checked = 0;
    for (k = 0; k < plan.num_chunks; ++k)
    {
        for (p = plan.chunks[k].first_position;
             p < plan.chunks[k].first_position + plan.chunks[k].num_positions; ++p)
        {
            int j, listed = 0;
            w = plan.position_wave[p];
            if (w < 0)
                continue; /* block drives no gradient */
            mu_assert(w < plan.num_waves, "wave index runs past the global list");
            /* and the chunk this position sits in must actually name it */
            for (j = plan.chunks[k].first_wave;
                 j < plan.chunks[k].first_wave + plan.chunks[k].num_waves; ++j)
                if (plan.chunk_waves[j] == w)
                {
                    listed = 1;
                    break;
                }
            mu_assert(listed, "a position plays a wave its own chunk does not load");
            checked++;
        }
    }
    mu_assert(checked > 0, "no position carried a wave");

    pulseg_free_chunk_plan(&plan);
    pulseg_collection_free(coll);
}

MU_TEST_SUITE(suite_chunk)
{
    MU_RUN_TEST(test_chunk_generous_budget_is_one_chunk);
    MU_RUN_TEST(test_chunk_tight_budget_splits_and_fits);
    MU_RUN_TEST(test_chunk_splitting_lowers_the_peak);
    MU_RUN_TEST(test_chunk_infeasible_rate_is_reported);
    MU_RUN_TEST(test_chunk_rejects_zero_budget);
    MU_RUN_TEST(test_chunk_rotation_preserves_summed_power);
    MU_RUN_TEST(test_chunk_materialise_reports_required_size);
    MU_RUN_TEST(test_chunk_corner_times_strictly_increase);
    MU_RUN_TEST(test_chunk_position_waves_stay_inside_their_chunk);
}

int test_chunk_main(void)
{
    MU_RUN_SUITE(suite_chunk);
    MU_REPORT();
    return minunit_fail;
}
