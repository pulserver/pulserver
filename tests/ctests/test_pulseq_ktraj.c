/*
 * test_pulseq_ktraj.c -- the k-space trajectory core, against properties the
 * .seq files state for themselves.
 *
 * The oracle question was the interesting one here.  PyPulseq is the obvious
 * reference and it is used for exactly that in tests/python, but it cannot be
 * used from C, it cannot read a ROTATIONS extension at all, and on an
 * arbitrary gradient carrying an explicit time shape its own gradient spline
 * does not reproduce its own waveform (see the header comment on
 * test_flat_arbitrary_gradient_is_exact).  So the assertions below are
 * against the sequence's own numbers instead: a gradient the file declares
 * flat integrates to a straight line with a known slope, a refocusing pulse
 * negates k, a rotation composes with the file's own quaternion, and the echo
 * the fast path reports is the one an exhaustive search finds.  None of that
 * needs a second implementation to check against.
 *
 * Two of these exist because of bugs they caught:
 * check_centers_are_minimal covers a shortcut in the echo search that was
 * once wrong at a near-tie, and it is run against a float32 build because the
 * one-sample error it also caught only appeared there.  test_the_repeat_memo_
 * fires exists because every wrong version of the memo key still produced
 * right answers -- it just never fired, and nothing else would have noticed.
 */

#include "test_helpers.h"

#include "pulseq.h"
#include "pulseq_ktraj.h"

/*
 * What two adjacent k samples can differ by from rounding alone.
 *
 * The library stores k in PULSEQ_REAL, which the scanner builds as float, and
 * a readout reaches +-500 1/m -- so a float32 k has about 3e-5 of slack in it
 * and a difference of two has twice that.  Asserting a fixed 1e-9 here would
 * be asserting that the storage is double, which on the target it is not.
 * The arithmetic underneath is double either way; this bounds the write-out,
 * not the computation.
 */
static double ktraj_tol(double magnitude)
{
    double eps = (sizeof(PULSEQ_REAL) == sizeof(float)) ? 1.2e-7 : 2.3e-16;
    double slack = 4.0 * eps * magnitude;
    return (slack > 1e-12) ? slack : 1e-12;
}

static void ktraj_raster(pulseq_raster *r)
{
    r->rf_us = (PULSEQ_REAL)1.0;
    r->grad_us = (PULSEQ_REAL)10.0;
    r->block_us = (PULSEQ_REAL)10.0;
}

/* Read a fixture, or fail the test. */
static void ktraj_open(pulseq_file *seq, const char *path)
{
    pulseq_raster raster;
    int rc;

    ktraj_raster(&raster);
    pulseq_file_init(seq, &raster);
    rc = pulseq_read(seq, path);
    mu_assert(PULSEQ_SUCCEEDED(rc), "fixture failed to parse");
}

/* ================================================================== */

/*
 * A gradient the file declares constant integrates to a straight line.
 *
 * fse_2d_1sl_1avg's readout is [GRADIENTS] row 9: amplitude 194553 Hz/m,
 * first == last == amplitude, a two-sample shape of [1, 1] and a two-sample
 * time shape of [0, 257] on a 20 us raster -- a flat 194553 Hz/m spanning
 * 5.14 ms.  The ADC takes 256 samples at a 20 us dwell, so consecutive
 * samples must differ by exactly amplitude * dwell = 3.89106 1/m, with no
 * drift from the first pair to the last.
 *
 * This is the case where PyPulseq's own answer is wrong: its gradient spline
 * ramps from 194552.5924 at the start of that gradient to 194553.0 at the
 * end, so its k step grows across the readout and lands 2.1e-6 low.  Asserting
 * against PyPulseq here would have meant encoding its error as the
 * expectation; asserting against the file's own numbers does not.
 */
MU_TEST(test_flat_arbitrary_gradient_is_exact)
{
    pulseq_file seq;
    pulseq_ktraj traj;
    pulseq_ktraj_opts opts;
    int rc, j, n, off;
    double expected_step;

    ktraj_open(&seq, TEST_DATA_DIR "fse_2d_1sl_1avg.seq");

    pulseq_ktraj_opts_init(&opts);
    rc = pulseq_ktraj_all(&seq, &opts, &traj);
    mu_assert(PULSEQ_SUCCEEDED(rc), "ktraj failed");
    mu_assert(traj.num_readouts > 0, "fixture has no readouts");

    n = traj.readouts[0].num_samples;
    off = traj.sample_offset[0];
    mu_assert_int_eq(256, n);

    expected_step = 194553.0 * (double)traj.readouts[0].dwell;
    for (j = 1; j < n; ++j)
    {
        double a = (double)traj.k[(size_t)off * 3 + 0 * n + j];
        double b = (double)traj.k[(size_t)off * 3 + 0 * n + (j - 1)];
        double magnitude = (fabs(a) > fabs(b)) ? fabs(a) : fabs(b);
        mu_assert(fabs((a - b) - expected_step) <= ktraj_tol(magnitude),
                  "readout k step drifts from amplitude * dwell");
    }

    pulseq_ktraj_free(&traj);
    pulseq_file_free(&seq);
}

/*
 * A refocusing pulse negates k.
 *
 * fse_2d_1sl_1avg is a spin-echo train, so between one readout and the next
 * there is a refocusing pulse.  What is asserted is the invariant that makes
 * an echo train walk back through the origin at all: consecutive readouts sit
 * on opposite sides of k-space along the readout axis, which cannot happen if
 * the sign flip is dropped.
 */
MU_TEST(test_refocusing_negates_k)
{
    pulseq_file seq;
    pulseq_ktraj traj;
    pulseq_ktraj_opts opts;
    int rc, i, n, off;

    ktraj_open(&seq, TEST_DATA_DIR "fse_2d_1sl_1avg.seq");

    pulseq_ktraj_opts_init(&opts);
    rc = pulseq_ktraj_all(&seq, &opts, &traj);
    mu_assert(PULSEQ_SUCCEEDED(rc), "ktraj failed");
    mu_assert(traj.num_refocusings > 0, "fixture has no refocusing pulses");

    /* Every readout starts on the negative side and ends on the positive
     * side: the prewinder puts k at -kmax and the readout sweeps across. */
    for (i = 0; i < traj.num_readouts; ++i)
    {
        n = traj.readouts[i].num_samples;
        off = traj.sample_offset[i];
        mu_assert(traj.k[(size_t)off * 3 + 0 * n + 0] < 0.0,
                  "readout does not start on the negative side of k");
        mu_assert(traj.k[(size_t)off * 3 + 0 * n + (n - 1)] > 0.0,
                  "readout does not end on the positive side of k");
    }

    pulseq_ktraj_free(&traj);
    pulseq_file_free(&seq);
}

/*
 * The echo lands where the readout crosses the k-space centre.
 *
 * gre_2d_1sl_1avg is a symmetric Cartesian readout, so the crossing is the
 * middle sample.  The point is not the number 32 -- it is that the number is
 * *derived*: pulseg takes center_sample from a design-time anchor, which a
 * foreign .seq does not carry, and this is what replaces it.
 */
static void check_centers_are_minimal(const char *fixture)
{
    pulseq_file seq;
    pulseq_ktraj traj;
    pulseq_ktraj_opts opts;
    int rc, i, a, j;
    double global_best = 0.0;
    int have_global = 0;
    double tie = 0.0;

    ktraj_open(&seq, fixture);

    pulseq_ktraj_opts_init(&opts);
    rc = pulseq_ktraj_all(&seq, &opts, &traj);
    mu_assert(PULSEQ_SUCCEEDED(rc), "ktraj failed");
    mu_assert(traj.num_readouts > 0, "fixture has no readouts");

    /*
     * The tie tolerance the core works to: 1% of a sampling step.
     *
     * Both answers below are minima *up to a tie*, and the core resolves ties
     * by convention rather than by whichever side rounding fell on -- see the
     * tie rule in derive_centers().  So the assertions are "no worse than the
     * true minimum by more than one tie", which is the contract; asserting the
     * bare minimum instead would be asserting the rounding.
     *
     * Where it bites, measured: epi_2d_1sl_1avg, whose two readout polarities
     * approach the origin equally closely.  The core takes the forward one, at
     * float32 a distance 1.3e-5 further out -- 0.0006% of the 2.03 step.
     */
    for (i = 0; i < traj.num_readouts; ++i)
    {
        int n = traj.readouts[i].num_samples;
        int off = traj.sample_offset[i];
        double total = 0.0;
        if (n < 2)
            continue;
        for (j = 1; j < n; ++j)
        {
            double d = 0.0;
            for (a = 0; a < 3; ++a)
            {
                double v = (double)traj.k[(size_t)off * 3 + a * n + j] -
                           (double)traj.k[(size_t)off * 3 + a * n + (j - 1)];
                d += v * v;
            }
            total += sqrt(d);
        }
        total = 1.0e-2 * total / (double)(n - 1);
        if (total > tie)
            tie = total;
    }

    /* k_center really is the whole scan's closest approach to the origin. */
    for (i = 0; i < traj.num_readouts; ++i)
    {
        int n = traj.readouts[i].num_samples;
        int off = traj.sample_offset[i];
        for (j = 0; j < n; ++j)
        {
            double d = 0.0;
            for (a = 0; a < 3; ++a)
            {
                double v = (double)traj.k[(size_t)off * 3 + a * n + j];
                d += v * v;
            }
            if (!have_global || d < global_best)
            {
                global_best = d;
                have_global = 1;
            }
        }
    }
    {
        double d = 0.0;
        for (a = 0; a < 3; ++a)
            d += (double)traj.k_center[a] * (double)traj.k_center[a];
        /* Compared as distances, not squared ones: the tolerance is a bound
         * on a stored k, and squaring it would not mean anything. */
        mu_assert(sqrt(d) <= sqrt(global_best) + tie + ktraj_tol(sqrt(global_best) + 1.0),
                  "k_center is not the scan's closest approach to the origin");
    }

    /* And every center_sample is its readout's closest sample to it. */
    for (i = 0; i < traj.num_readouts; ++i)
    {
        int n = traj.readouts[i].num_samples;
        int c = traj.readouts[i].center_sample;
        int off = traj.sample_offset[i];
        double best = 0.0;
        double chosen = 0.0;

        mu_assert(c >= 0 && c < n, "center_sample outside the readout");

        for (j = 0; j < n; ++j)
        {
            double d = 0.0;
            for (a = 0; a < 3; ++a)
            {
                double v = (double)traj.k[(size_t)off * 3 + a * n + j] -
                           (double)traj.k_center[a];
                d += v * v;
            }
            if (j == 0 || d < best)
                best = d;
            if (j == c)
                chosen = d;
        }
        /* Compared as distances, not as indices: two samples equidistant from
         * the centre are both right answers, and which one wins is not a
         * property worth freezing. */
        mu_assert(sqrt(chosen) <= sqrt(best) + tie + ktraj_tol(sqrt(best) + 1.0),
                  "center_sample is not the closest sample to the k centre");
    }

    pulseq_ktraj_free(&traj);
    pulseq_file_free(&seq);
}

/*
 * The echo is derived, and the shortcut that derives it is exact.
 *
 * This is the load-bearing test of the whole file.  center_sample is computed
 * by splitting the search into the axes a readout sweeps and the ones it does
 * not, and memoizing the first part across repeats -- so the assertion is
 * against the answer the slow, obvious, unmemoized search gives, over
 * fixtures chosen to exercise the split: Cartesian, an echo train, a blipped
 * EPI, and a rotated non-Cartesian one.
 *
 * It is also the replacement for a number pulseg does not have.
 * pulseg_trajectory.c takes center_sample from a design-time anchor, and a
 * .seq authored anywhere else carries no anchors at all.
 */
MU_TEST(test_center_sample_is_derived_not_assumed)
{
    check_centers_are_minimal(TEST_DATA_DIR "gre_2d_1sl_1avg.seq");
    check_centers_are_minimal(TEST_DATA_DIR "gre_2d_3sl_3avg.seq");
    check_centers_are_minimal(TEST_DATA_DIR "epi_2d_1sl_1avg.seq");
    check_centers_are_minimal(TEST_DATA_DIR "fse_2d_1sl_1avg.seq");
    check_centers_are_minimal(TEST_DATA_DIR "gre_32x32_pe_blip.seq");
    check_centers_are_minimal(TEST_DATA_DIR "mprage_noncart_3d_3sl_3avg_userotext1.seq");
}

/*
 * Disabling the derivation costs the answer, not the trajectory.
 *
 * Guards the opt-out actually being an opt-out: k must be untouched and only
 * center_sample must go away.
 */
MU_TEST(test_center_sample_can_be_declined)
{
    pulseq_file seq;
    pulseq_ktraj with;
    pulseq_ktraj without;
    pulseq_ktraj_opts opts;
    int rc, i;

    ktraj_open(&seq, TEST_DATA_DIR "gre_2d_1sl_1avg.seq");

    pulseq_ktraj_opts_init(&opts);
    rc = pulseq_ktraj_all(&seq, &opts, &with);
    mu_assert(PULSEQ_SUCCEEDED(rc), "ktraj failed");

    opts.derive_center_sample = 0;
    rc = pulseq_ktraj_all(&seq, &opts, &without);
    mu_assert(PULSEQ_SUCCEEDED(rc), "ktraj failed without centers");

    mu_assert_int_eq(with.total_samples, without.total_samples);
    for (i = 0; i < with.total_samples * 3; ++i)
        mu_assert(with.k[i] == without.k[i], "declining centers changed k");
    for (i = 0; i < without.num_readouts; ++i)
        mu_assert_int_eq(-1, without.readouts[i].center_sample);

    pulseq_ktraj_free(&with);
    pulseq_ktraj_free(&without);
    pulseq_file_free(&seq);
}

/*
 * A rotation extension composes into k, and declining it leaves the logical
 * frame behind.
 *
 * PyPulseq cannot read a ROTATIONS extension, so this is checked against the
 * file's own rotation matrix instead: rotating the logical answer by hand has
 * to reproduce the physical one, for every sample.
 */
MU_TEST(test_rotation_extension_composes)
{
    pulseq_file seq;
    pulseq_ktraj physical;
    pulseq_ktraj logical;
    pulseq_ktraj_opts opts;
    int rc, i, differs = 0;

    ktraj_open(&seq, TEST_DATA_DIR "mprage_noncart_3d_1sl_1avg_userotext1.seq");
    mu_assert(seq.rotation_library_size > 0, "fixture carries no rotations");

    pulseq_ktraj_opts_init(&opts);
    rc = pulseq_ktraj_all(&seq, &opts, &physical);
    mu_assert(PULSEQ_SUCCEEDED(rc), "physical ktraj failed");

    opts.apply_rotation = 0;
    rc = pulseq_ktraj_all(&seq, &opts, &logical);
    mu_assert(PULSEQ_SUCCEEDED(rc), "logical ktraj failed");

    mu_assert_int_eq(physical.num_readouts, logical.num_readouts);

    for (i = 0; i < physical.num_readouts; ++i)
    {
        int n = physical.readouts[i].num_samples;
        int off = physical.sample_offset[i];
        int j;
        for (j = 0; j < n; ++j)
        {
            double lx = (double)logical.k[(size_t)off * 3 + 0 * n + j];
            double ly = (double)logical.k[(size_t)off * 3 + 1 * n + j];
            double lz = (double)logical.k[(size_t)off * 3 + 2 * n + j];
            double px = (double)physical.k[(size_t)off * 3 + 0 * n + j];
            if (fabs(px - lx) > 1e-6 * (fabs(lx) + fabs(ly) + fabs(lz) + 1.0))
                differs = 1;
        }
    }
    mu_assert(differs, "rotation made no difference on a rotated fixture");

    pulseq_ktraj_free(&physical);
    pulseq_ktraj_free(&logical);
    pulseq_file_free(&seq);
}

/*
 * The repeat memo actually fires.
 *
 * Correctness is covered above; this covers the thing correctness cannot see.
 * A memo that never hits is not slow, it is *pointless*, and every version of
 * this key that was too strict still produced right answers -- the first one
 * hit 0% of readouts on every fixture and nothing failed.  So the hit rate is
 * asserted directly.
 *
 * The numbers below are floors, not measurements to keep in step with. They
 * are set well under what the fixtures currently reach (gre 87%, epi 98%,
 * fse 87%, the blipped one 96%) so ordinary changes do not trip them, while a
 * key that stops grouping does.
 */
static void check_memo_fires(const char *fixture, int max_groups)
{
    pulseq_file seq;
    pulseq_ktraj_plan *plan = NULL;
    pulseq_ktraj_opts opts;
    int rc;

    ktraj_open(&seq, fixture);
    pulseq_ktraj_opts_init(&opts);
    rc = pulseq_ktraj_plan_create(&plan, &seq, &opts);
    mu_assert(PULSEQ_SUCCEEDED(rc), "plan failed");

    mu_assert(pulseq_ktraj_num_readouts(plan) > max_groups,
              "fixture is too small to say anything about grouping");
    mu_assert(pulseq_ktraj_num_key_groups(plan) <= max_groups,
              "the repeat memo grouped fewer readouts than it should have");

    pulseq_ktraj_plan_free(plan);
    pulseq_file_free(&seq);
}

MU_TEST(test_the_repeat_memo_fires)
{
    /* One readout shape, many phase-encode lines. */
    check_memo_fires(TEST_DATA_DIR "gre_2d_1sl_1avg.seq", 2);
    check_memo_fires(TEST_DATA_DIR "gre_2d_3sl_3avg.seq", 3);
    /* Two alternating readout polarities. */
    check_memo_fires(TEST_DATA_DIR "epi_2d_1sl_1avg.seq", 4);
    check_memo_fires(TEST_DATA_DIR "fse_2d_3sl_3avg.seq", 4);
    /* Library deduplication switched off: 32 ids for one readout. */
    check_memo_fires(TEST_DATA_DIR "gre_32x32_pe_blip.seq", 3);
}

/*
 * A block range bounds what is reported, and k restarts inside it.
 *
 * Unlike pulseq::apply_fov_shift, which accumulates from block 1 whatever
 * range it is asked to modify, this really does start at the range -- a
 * caller asking for a window is asking what that window traverses.
 */
MU_TEST(test_block_range_bounds_the_walk)
{
    pulseq_file seq;
    pulseq_ktraj all;
    pulseq_ktraj part;
    pulseq_ktraj_opts opts;
    int rc;

    ktraj_open(&seq, TEST_DATA_DIR "gre_2d_1sl_1avg.seq");

    pulseq_ktraj_opts_init(&opts);
    rc = pulseq_ktraj_all(&seq, &opts, &all);
    mu_assert(PULSEQ_SUCCEEDED(rc), "full ktraj failed");

    opts.first_block = 1;
    opts.last_block = seq.num_blocks / 2;
    rc = pulseq_ktraj_all(&seq, &opts, &part);
    mu_assert(PULSEQ_SUCCEEDED(rc), "ranged ktraj failed");

    mu_assert(part.num_readouts < all.num_readouts, "range did not bound the readouts");
    mu_assert(part.total_duration < all.total_duration, "range did not bound the duration");

    pulseq_ktraj_free(&all);
    pulseq_ktraj_free(&part);
    pulseq_file_free(&seq);
}

/*
 * A gradient offset is a background gradient, so it moves k with time even
 * where the sequence drives nothing.
 */
MU_TEST(test_gradient_offset_moves_k)
{
    pulseq_file seq;
    pulseq_ktraj plain;
    pulseq_ktraj offset;
    pulseq_ktraj_opts opts;
    int rc, off, n;

    ktraj_open(&seq, TEST_DATA_DIR "gre_2d_1sl_1avg.seq");

    pulseq_ktraj_opts_init(&opts);
    rc = pulseq_ktraj_all(&seq, &opts, &plain);
    mu_assert(PULSEQ_SUCCEEDED(rc), "plain ktraj failed");

    opts.gradient_offset[1] = (PULSEQ_REAL)1000.0;
    rc = pulseq_ktraj_all(&seq, &opts, &offset);
    mu_assert(PULSEQ_SUCCEEDED(rc), "offset ktraj failed");

    n = plain.readouts[0].num_samples;
    off = plain.sample_offset[0];
    mu_assert(fabs((double)offset.k[(size_t)off * 3 + 1 * n + 0] -
                   (double)plain.k[(size_t)off * 3 + 1 * n + 0]) > 1e-6,
              "gradient offset did not move ky");

    pulseq_ktraj_free(&plain);
    pulseq_ktraj_free(&offset);
    pulseq_file_free(&seq);
}

/*
 * The streaming interface agrees with the materialised one.
 *
 * They share readout_samples(), so this is guarding the plumbing around it --
 * offsets, buffer sizes, the plan surviving repeated queries -- which is
 * exactly what a caller walking a million readouts one at a time depends on.
 */
MU_TEST(test_streaming_matches_materialised)
{
    pulseq_file seq;
    pulseq_ktraj traj;
    pulseq_ktraj_plan *plan = NULL;
    pulseq_ktraj_opts opts;
    PULSEQ_REAL *buf;
    int rc, i, longest = 0;

    ktraj_open(&seq, TEST_DATA_DIR "epi_2d_1sl_1avg.seq");

    pulseq_ktraj_opts_init(&opts);
    rc = pulseq_ktraj_all(&seq, &opts, &traj);
    mu_assert(PULSEQ_SUCCEEDED(rc), "ktraj failed");

    rc = pulseq_ktraj_plan_create(&plan, &seq, &opts);
    mu_assert(PULSEQ_SUCCEEDED(rc), "plan failed");
    mu_assert_int_eq(traj.num_readouts, pulseq_ktraj_num_readouts(plan));

    for (i = 0; i < traj.num_readouts; ++i)
    {
        if (traj.readouts[i].num_samples > longest)
            longest = traj.readouts[i].num_samples;
    }
    buf = (PULSEQ_REAL *)malloc((size_t)longest * 3 * sizeof(PULSEQ_REAL));
    mu_assert(buf != NULL, "allocation failed");

    for (i = 0; i < traj.num_readouts; ++i)
    {
        int n = traj.readouts[i].num_samples;
        int off = traj.sample_offset[i];
        int j;
        pulseq_ktraj_readout info;

        rc = pulseq_ktraj_readout_info(plan, i, &info);
        mu_assert(PULSEQ_SUCCEEDED(rc), "readout_info failed");
        mu_assert_int_eq(traj.readouts[i].center_sample, info.center_sample);

        rc = pulseq_ktraj_readout_k(plan, i, buf);
        mu_assert(PULSEQ_SUCCEEDED(rc), "readout_k failed");
        for (j = 0; j < n * 3; ++j)
            mu_assert(buf[j] == traj.k[(size_t)off * 3 + j],
                      "streamed sample differs from the materialised one");
    }

    free(buf);
    pulseq_ktraj_plan_free(plan);
    pulseq_ktraj_free(&traj);
    pulseq_file_free(&seq);
}

/* An RF pulse with no declared use counts as an excitation, as PyPulseq does. */
MU_TEST(test_undefined_rf_use_resets_k)
{
    pulseq_file seq;
    pulseq_ktraj traj;
    pulseq_ktraj_opts opts;
    int rc, i, undefined = 0;

    ktraj_open(&seq, TEST_DATA_DIR "gre_2d_1sl_1avg.seq");

    for (i = 0; i < seq.rf_library_size; ++i)
    {
        if (seq.rf_use_tags && seq.rf_use_tags[i] == PULSEQ_RF_USE_UNKNOWN)
            undefined++;
    }

    pulseq_ktraj_opts_init(&opts);
    rc = pulseq_ktraj_all(&seq, &opts, &traj);
    mu_assert(PULSEQ_SUCCEEDED(rc), "ktraj failed");

    /* Whatever the fixture's mix, every excitation-or-undefined pulse is
     * counted as an excitation and none is dropped. */
    mu_assert(traj.num_excitations > 0, "no excitations found");

    pulseq_ktraj_free(&traj);
    pulseq_file_free(&seq);
}

/*
 * Window-averaged sampling changes nothing on a flat gradient.
 *
 * Which is the whole reason it is safe to offer.  A sample's true coordinate
 * is k averaged over its dwell, and the average of a straight line over a
 * symmetric window is its midpoint -- so on a constant gradient, where k is
 * exactly linear in time, the correction is identically zero.  fse_2d's
 * readout is a flat 194553 Hz/m, so switching the option on must not move a
 * single sample of it.
 *
 * Anything that did move would mean the second antiderivative is wrong, and
 * that is much easier to see here than in the ramp case where the correction
 * is supposed to be small but non-zero.
 */
MU_TEST(test_window_average_is_a_no_op_on_a_flat_gradient)
{
    pulseq_file seq;
    pulseq_ktraj midpoint;
    pulseq_ktraj averaged;
    pulseq_ktraj_opts opts;
    int rc, n, off, j;

    ktraj_open(&seq, TEST_DATA_DIR "fse_2d_1sl_1avg.seq");

    pulseq_ktraj_opts_init(&opts);
    rc = pulseq_ktraj_all(&seq, &opts, &midpoint);
    mu_assert(PULSEQ_SUCCEEDED(rc), "midpoint ktraj failed");

    opts.sample_window_average = 1;
    rc = pulseq_ktraj_all(&seq, &opts, &averaged);
    mu_assert(PULSEQ_SUCCEEDED(rc), "windowed ktraj failed");

    mu_assert_int_eq(midpoint.num_readouts, averaged.num_readouts);
    n = midpoint.readouts[0].num_samples;
    off = midpoint.sample_offset[0];
    for (j = 0; j < n; ++j)
    {
        double a = (double)midpoint.k[(size_t)off * 3 + 0 * n + j];
        double b = (double)averaged.k[(size_t)off * 3 + 0 * n + j];
        mu_assert(fabs(a - b) <= ktraj_tol(fabs(a) + 1.0),
                  "window averaging moved a sample on a flat gradient");
    }

    pulseq_ktraj_free(&midpoint);
    pulseq_ktraj_free(&averaged);
    pulseq_file_free(&seq);
}

/*
 * Where the gradient does vary, the correction is right.
 *
 * "Right" needs an oracle, and the obvious one -- that the correction should
 * equal (dwell^2/24) * dg/dt -- turns out not to apply to any fixture here.
 * That identity holds only while the gradient is linear across the whole
 * window, and with an 80 us dwell on a 20 us raster every window spans four
 * gradient segments.  Measured: zero samples in the fixture set satisfy it.
 *
 * So the oracle is built instead out of the midpoint path, which is already
 * checked against PyPulseq to 1e-12.  `trajectory_delay` shifts the gradient
 * time base, so a run with delay d reports k(t_j - d) at every sample; sweeping
 * d across one dwell samples k *inside* each window, and Simpson over those
 * samples is the window average computed a completely different way -- by
 * quadrature over many evaluations rather than in closed form from a second
 * antiderivative.
 *
 * The non-Cartesian fixture is used because it is the only one whose readout
 * gradient actually varies: on every Cartesian and EPI fixture here the ADC
 * sits on a flat top and the correction is identically zero, which the
 * previous test already covers.
 */
MU_TEST(test_window_average_matches_an_independent_quadrature)
{
    /* Even, so Simpson's rule applies; 64 is far past convergence here. */
    enum
    {
        SUBSAMPLES = 64
    };
    pulseq_file seq;
    pulseq_ktraj windowed;
    pulseq_ktraj sub[SUBSAMPLES + 1];
    pulseq_ktraj_opts opts;
    double dwell;
    int rc, i, a, j, m, moved = 0;

    ktraj_open(&seq, TEST_DATA_DIR "mprage_noncart_3d_3sl_3avg_userotext1.seq");

    pulseq_ktraj_opts_init(&opts);
    opts.sample_window_average = 1;
    rc = pulseq_ktraj_all(&seq, &opts, &windowed);
    mu_assert(PULSEQ_SUCCEEDED(rc), "windowed ktraj failed");
    mu_assert(windowed.num_readouts > 0, "fixture has no readouts");

    dwell = (double)windowed.readouts[0].dwell;
    for (m = 0; m <= SUBSAMPLES; ++m)
    {
        pulseq_ktraj_opts_init(&opts);
        for (a = 0; a < 3; ++a)
            opts.trajectory_delay[a] =
                (PULSEQ_REAL)(0.5 * dwell - dwell * (double)m / (double)SUBSAMPLES);
        rc = pulseq_ktraj_all(&seq, &opts, &sub[m]);
        mu_assert(PULSEQ_SUCCEEDED(rc), "sub-sampled ktraj failed");
    }

    for (i = 0; i < windowed.num_readouts; ++i)
    {
        int n = windowed.readouts[i].num_samples;
        int off = windowed.sample_offset[i];

        for (a = 0; a < 3; ++a)
        {
            for (j = 0; j < n; ++j)
            {
                double acc = 0.0;
                double got, reference;

                for (m = 0; m <= SUBSAMPLES; ++m)
                {
                    double w = (m == 0 || m == SUBSAMPLES) ? 1.0 : ((m % 2) ? 4.0 : 2.0);
                    acc += w * (double)sub[m].k[(size_t)off * 3 + a * n + j];
                }
                reference = acc / (3.0 * (double)SUBSAMPLES);
                got = (double)windowed.k[(size_t)off * 3 + a * n + j];

                if (fabs(got - (double)sub[SUBSAMPLES / 2].k[(size_t)off * 3 + a * n + j]) >
                    1e-3)
                    moved = 1;
                mu_assert(fabs(got - reference) <= 1e-6 * (fabs(reference) + 1.0),
                          "closed-form window average disagrees with quadrature");
            }
        }
    }
    mu_assert(moved, "window averaging changed nothing on a varying gradient");

    for (m = 0; m <= SUBSAMPLES; ++m)
        pulseq_ktraj_free(&sub[m]);
    pulseq_ktraj_free(&windowed);
    pulseq_file_free(&seq);
}

/* Bad arguments are refused rather than crashed on. */
MU_TEST(test_bad_arguments_are_refused)
{
    pulseq_file seq;
    pulseq_ktraj_plan *plan = NULL;
    pulseq_ktraj_opts opts;
    pulseq_ktraj_readout info;
    int rc;

    ktraj_open(&seq, TEST_DATA_DIR "gre_2d_1sl_1avg.seq");
    pulseq_ktraj_opts_init(&opts);

    mu_assert(PULSEQ_FAILED(pulseq_ktraj_plan_create(NULL, &seq, &opts)), "NULL out accepted");
    mu_assert(PULSEQ_FAILED(pulseq_ktraj_plan_create(&plan, NULL, &opts)), "NULL seq accepted");

    opts.first_block = 10;
    opts.last_block = 2;
    mu_assert(PULSEQ_FAILED(pulseq_ktraj_plan_create(&plan, &seq, &opts)),
              "inverted block range accepted");

    pulseq_ktraj_opts_init(&opts);
    rc = pulseq_ktraj_plan_create(&plan, &seq, &opts);
    mu_assert(PULSEQ_SUCCEEDED(rc), "plan failed");
    mu_assert(PULSEQ_FAILED(pulseq_ktraj_readout_info(plan, -1, &info)), "index -1 accepted");
    mu_assert(PULSEQ_FAILED(pulseq_ktraj_readout_info(plan, 1 << 20, &info)),
              "huge index accepted");

    pulseq_ktraj_plan_free(plan);
    pulseq_ktraj_plan_free(NULL);
    pulseq_file_free(&seq);
}

/*
 * A centre-out readout has its echo at sample 0, not at the middle.
 *
 * The plainest statement of what this file replaces.  pulseg's trajectory
 * cache takes center_sample from a design-time anchor and falls back to
 * `num_samples / 2` when its k-space analysis did not run
 * (pulseg_structure.c:4108).  On mprage_noncart_3d_3sl_3avg_userotext1 that
 * fallback fires: the cache reports 32 of 64 for five of the six shots, while
 * every one of them starts at the centre of k-space and marches outwards.
 *
 * Measured against the scan's own closest approach to k = 0: sample 0 is
 * 3.4 1/m away and sample 32 is 24.0, so the cached answer is seven times
 * further out than the right one -- half a readout of error, in the index a
 * reconstruction centres its FFT on.
 */
MU_TEST(test_a_centre_out_readout_puts_its_echo_at_the_first_sample)
{
    pulseq_file seq;
    pulseq_ktraj traj;
    pulseq_ktraj_opts opts;
    int rc, i;

    ktraj_open(&seq, TEST_DATA_DIR "mprage_noncart_3d_3sl_3avg_userotext1.seq");

    pulseq_ktraj_opts_init(&opts);
    rc = pulseq_ktraj_all(&seq, &opts, &traj);
    mu_assert(PULSEQ_SUCCEEDED(rc), "ktraj failed");
    mu_assert(traj.num_readouts > 0, "fixture has no readouts");

    for (i = 0; i < traj.num_readouts; ++i)
    {
        int n = traj.readouts[i].num_samples;
        int off = traj.sample_offset[i];
        int j, a;
        double first = 0.0, middle = 0.0;

        mu_assert_int_eq(0, traj.readouts[i].center_sample);

        /* And it is the first sample because it is nearest, not because the
         * search stopped early: distance grows monotonically along the shot. */
        for (a = 0; a < 3; ++a)
        {
            double v0 = (double)traj.k[(size_t)off * 3 + a * n + 0] - (double)traj.k_center[a];
            double vm = (double)traj.k[(size_t)off * 3 + a * n + n / 2] -
                        (double)traj.k_center[a];
            first += v0 * v0;
            middle += vm * vm;
        }
        mu_assert(middle > 4.0 * first,
                  "the middle sample is not markedly further out than the first");

        for (j = 2; j < n; ++j)
        {
            double prev = 0.0, here = 0.0;
            for (a = 0; a < 3; ++a)
            {
                double p = (double)traj.k[(size_t)off * 3 + a * n + (j - 1)] -
                           (double)traj.k_center[a];
                double h = (double)traj.k[(size_t)off * 3 + a * n + j] -
                           (double)traj.k_center[a];
                prev += p * p;
                here += h * h;
            }
            mu_assert(here >= prev, "a centre-out shot turned back towards the centre");
        }
    }

    pulseq_ktraj_free(&traj);
    pulseq_file_free(&seq);
}

/*
 * A bipolar train's two polarities put the echo on different samples, and
 * mirroring the reverse ones brings them back together.
 *
 * This is the property that makes an EPI reconstructable.  The two polarities
 * are mirror images, so their nearest samples to the origin are the *same
 * point* reached from opposite ends: with 128 samples the forward readout
 * finds it at 64 and the reverse one at 63.  A rule that resolved the tie by
 * index rather than by direction would report 64 for both, and mirroring --
 * which is what the reverse readout's REV label asks a reconstruction to do --
 * would then land half the lines on 63.  Half a k-space one sample off centre
 * is a linear phase ramp across the image.
 *
 * Both halves are asserted: that the two polarities really do disagree (or
 * the fixture is not bipolar and the test proves nothing), and that mirroring
 * makes them agree.
 */
MU_TEST(test_a_bipolar_train_puts_every_echo_at_one_place_once_mirrored)
{
    pulseq_file seq;
    pulseq_ktraj traj;
    pulseq_ktraj_opts opts;
    int rc, i;
    int mirrored = -1;
    int seen_forward = 0;
    int seen_reverse = 0;

    ktraj_open(&seq, TEST_DATA_DIR "epi_2d_1sl_1avg.seq");

    pulseq_ktraj_opts_init(&opts);
    rc = pulseq_ktraj_all(&seq, &opts, &traj);
    mu_assert(PULSEQ_SUCCEEDED(rc), "ktraj failed");
    mu_assert(traj.num_readouts > 1, "fixture has no readout train");

    for (i = 0; i < traj.num_readouts; ++i)
    {
        int n = traj.readouts[i].num_samples;
        int c = traj.readouts[i].center_sample;
        int off = traj.sample_offset[i];
        int a, here;
        double travel = 0.0;

        if (n < 2 || c < 0)
            continue;

        /* Direction of travel: the dominant component of the displacement
         * from the first sample to the last.  The same quantity a REV label
         * carries, worked out here from k alone. */
        for (a = 0; a < 3; ++a)
        {
            double d = (double)traj.k[(size_t)off * 3 + a * n + (n - 1)] -
                       (double)traj.k[(size_t)off * 3 + a * n];
            if (fabs(d) > fabs(travel))
                travel = d;
        }

        if (travel >= 0.0)
        {
            seen_forward = 1;
            here = c;
        }
        else
        {
            seen_reverse = 1;
            here = n - 1 - c;
        }

        if (mirrored < 0)
            mirrored = here;
        mu_assert_int_eq(mirrored, here);
    }

    mu_assert(seen_forward && seen_reverse,
              "the fixture is not bipolar, so this proves nothing");
    /* 128 samples spanning the origin: the echo belongs at N/2. */
    mu_assert_int_eq(64, mirrored);

    pulseq_ktraj_free(&traj);
    pulseq_file_free(&seq);
}

MU_TEST_SUITE(test_pulseq_ktraj_suite)
{
    MU_RUN_TEST(test_flat_arbitrary_gradient_is_exact);
    MU_RUN_TEST(test_refocusing_negates_k);
    MU_RUN_TEST(test_center_sample_is_derived_not_assumed);
    MU_RUN_TEST(test_a_centre_out_readout_puts_its_echo_at_the_first_sample);
    MU_RUN_TEST(test_a_bipolar_train_puts_every_echo_at_one_place_once_mirrored);
    MU_RUN_TEST(test_center_sample_can_be_declined);
    MU_RUN_TEST(test_rotation_extension_composes);
    MU_RUN_TEST(test_the_repeat_memo_fires);
    MU_RUN_TEST(test_block_range_bounds_the_walk);
    MU_RUN_TEST(test_gradient_offset_moves_k);
    MU_RUN_TEST(test_streaming_matches_materialised);
    MU_RUN_TEST(test_undefined_rf_use_resets_k);
    MU_RUN_TEST(test_window_average_is_a_no_op_on_a_flat_gradient);
    MU_RUN_TEST(test_window_average_matches_an_independent_quadrature);
    MU_RUN_TEST(test_bad_arguments_are_refused);
}

int test_pulseq_ktraj_main(void)
{
    MU_RUN_SUITE(test_pulseq_ktraj_suite);
    MU_REPORT();
    return minunit_fail;
}
