/*
 * test_corner_points.c -- joined gradient corner-point stream.
 *
 * The corner stream is the whole interface the vendor gradient-heating
 * model sees, so these tests check the stream itself rather than anything
 * derived from it.
 */
#include "test_helpers.h"

#include "pulseg_waveforms.h"

static pulseg_opts s_opts;

/* Sequences whose canonical TR carries no live rotation.  Declaring
 * ROTATION events is not enough: radial, spiral, stack-of-stars and
 * propeller all reference an identity rotation within a single shot, so
 * they exercise the unrotated path however many rotations they define. */
static const char *const PLAIN_SEQS[] = {
    "gre_2d.seq",
    "gre_3d.seq",
    "se_2d.seq",
    "fse_2d.seq",
    "epi_2d.seq",
    "mprage_3d.seq",
    "gre_radial_2d.seq",
    "gre_spiral_2d.seq",
    "gre_stack_of_stars_3d.seq",
    "se_propeller_2d.seq"};

/* Every block of a ZTE TR carries a non-identity rotation, which is what
 * makes it the sequence that exercises the rotation pass. */
static const char *const ROTATED_SEQS[] = {"zte_3d.seq"};

#define N_PLAIN (int)(sizeof(PLAIN_SEQS) / sizeof(PLAIN_SEQS[0]))
#define N_ROTATED (int)(sizeof(ROTATED_SEQS) / sizeof(ROTATED_SEQS[0]))

static float blk_dur(const pulseg_sequence_descriptor *desc, int blk)
{
    const pulseg_block_table_element *bte = &desc->block_table[blk];
    const pulseg_base_block *bdef = &desc->base_blocks[bte->id];

    return (bte->duration_us >= 0) ? (float)bte->duration_us : (float)bdef->duration_us;
}

static int blk_is_delay(const pulseg_sequence_descriptor *desc, int blk)
{
    const pulseg_block_table_element *bte = &desc->block_table[blk];

    return bte->gx_id < 0 && bte->gy_id < 0 && bte->gz_id < 0 && bte->rf_id < 0 &&
           bte->adc_id < 0;
}

static float tr_duration_us(const pulseg_sequence_descriptor *desc)
{
    float total, best, d;
    int tr_size, num_trs, p, t, blk;

    tr_size = desc->tr_descriptor.tr_size;
    num_trs = desc->num_blocks / tr_size;
    if (num_trs < 1)
        num_trs = 1;

    /* States the rule independently: every delay position at its own
     * shortest instance, every other position as it stands. */
    total = 0.0f;
    for (p = 0; p < tr_size; ++p)
    {
        if (!blk_is_delay(desc, p))
        {
            total += blk_dur(desc, p);
            continue;
        }
        best = blk_dur(desc, p);
        for (t = 1; t < num_trs; ++t)
        {
            blk = t * tr_size + p;
            if (blk >= desc->num_blocks)
                break;
            d = blk_dur(desc, blk);
            if (d < best)
                best = d;
        }
        total += best;
    }
    return total;
}

/* The stream must span the whole TR: a pure-delay block contributes its
 * duration as idle, which is the property the heating model depends on. */
static void check_spans_tr(const char *name)
{
    pulseg_collection *coll = NULL;
    pulseg_corner_point_stream s = PULSEG_CORNER_POINT_STREAM_INIT;
    pulseg_diagnostic diag;
    float expected, got;
    int rc;

    rc = load_corpus_seq(&coll, name, &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_corpus_seq failed");

    rc = pulseg_get_tr_corner_points(coll, &s, &diag, 0);
    mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_tr_corner_points failed");
    mu_assert(s.num_points > 1, "expected a non-degenerate stream");

    expected = tr_duration_us(&coll->descriptors[0]);
    got = s.time_us[s.num_points - 1];
    mu_assert_float_near("stream must reach the end of the TR", expected, got, 1.0f);
    mu_assert(s.time_us[0] >= 0.0f, "stream must start at or after t=0");

    pulseg_corner_point_stream_free(&s);
    pulseg_collection_free(coll);
}

/* Times must be strictly increasing: the union collapses coincident
 * breakpoints, and a repeated time would be a zero-width interval. */
static void check_strictly_increasing(const char *name)
{
    pulseg_collection *coll = NULL;
    pulseg_corner_point_stream s = PULSEG_CORNER_POINT_STREAM_INIT;
    pulseg_diagnostic diag;
    int rc, i;

    rc = load_corpus_seq(&coll, name, &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_corpus_seq failed");
    rc = pulseg_get_tr_corner_points(coll, &s, &diag, 0);
    mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_tr_corner_points failed");

    for (i = 1; i < s.num_points; ++i)
        mu_assert(s.time_us[i] > s.time_us[i - 1], "corner times must strictly increase");

    pulseg_corner_point_stream_free(&s);
    pulseg_collection_free(coll);
}

/* The stream must agree with the rasterised path.  The two reach the same
 * waveform by different routes -- one rotates per raster sample, the other
 * rotates at the corners -- so agreement exercises both the join and the
 * rotation. */
/* pulseg_get_tr_waveforms renders TR 0, so it is a valid reference only
 * where the composed stream reduces to TR 0 -- that is, a sequence with a
 * single TR instance. */
static int has_single_tr_instance(const pulseg_collection *coll)
{
    const pulseg_sequence_descriptor *d = &coll->descriptors[0];

    return d->num_blocks / d->tr_descriptor.tr_size <= 1;
}

static void check_matches_uniform(const char *name)
{
    pulseg_collection *coll = NULL;
    pulseg_corner_point_stream s = PULSEG_CORNER_POINT_STREAM_INIT;
    pulseg_tr_waveforms w;
    pulseg_diagnostic diag;
    float *resampled;
    float tol;
    int rc, i, n;

    memset(&w, 0, sizeof(w));

    rc = load_corpus_seq(&coll, name, &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_corpus_seq failed");

    if (!has_single_tr_instance(coll))
    {
        pulseg_collection_free(coll);
        return;
    }

    rc = pulseg_get_tr_corner_points(coll, &s, &diag, 0);
    mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_tr_corner_points failed");

    rc = pulseg_get_tr_waveforms(coll, &w, &diag, 0, PULSEG_AMP_ACTUAL, 0, 0);
    mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_tr_waveforms failed");

    n = w.gx.num_samples;
    mu_assert(n > 0, "expected uniform samples");

    resampled = (float *)malloc((size_t)n * sizeof(float));
    mu_assert(resampled != NULL, "alloc failed");

    /* 1 Hz/m against gradients of order 1e6 Hz/m. */
    tol = 1.0f;

    pulseg__interp1_linear(resampled, w.gx.time_us, n, s.time_us, s.gx_hz_per_m, s.num_points);
    for (i = 0; i < n; ++i)
        mu_assert_float_near("gx must match the rasterised path", w.gx.amplitude[i], resampled[i], tol);

    pulseg__interp1_linear(resampled, w.gy.time_us, n, s.time_us, s.gy_hz_per_m, s.num_points);
    for (i = 0; i < n; ++i)
        mu_assert_float_near("gy must match the rasterised path", w.gy.amplitude[i], resampled[i], tol);

    pulseg__interp1_linear(resampled, w.gz.time_us, n, s.time_us, s.gz_hz_per_m, s.num_points);
    for (i = 0; i < n; ++i)
        mu_assert_float_near("gz must match the rasterised path", w.gz.amplitude[i], resampled[i], tol);

    free(resampled);
    pulseg_tr_waveforms_free(&w);
    pulseg_corner_point_stream_free(&s);
    pulseg_collection_free(coll);
}

/* Point count must follow the sequence, not the TR duration -- that is the
 * whole reason for a corner stream rather than a raster. */
static void check_not_a_raster(const char *name)
{
    pulseg_collection *coll = NULL;
    pulseg_corner_point_stream s = PULSEG_CORNER_POINT_STREAM_INIT;
    pulseg_diagnostic diag;
    float raster_points;
    int rc;

    rc = load_corpus_seq(&coll, name, &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_corpus_seq failed");
    rc = pulseg_get_tr_corner_points(coll, &s, &diag, 0);
    mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_tr_corner_points failed");

    raster_points =
        tr_duration_us(&coll->descriptors[0]) / (0.5f * coll->descriptors[0].grad_raster_us);
    mu_assert((float)s.num_points <= raster_points, "stream must not exceed a raster of the TR");

    pulseg_corner_point_stream_free(&s);
    pulseg_collection_free(coll);
}

MU_TEST(test_corner_stream_spans_whole_tr)
{
    int i;
    for (i = 0; i < N_PLAIN; ++i)
        check_spans_tr(PLAIN_SEQS[i]);
    for (i = 0; i < N_ROTATED; ++i)
        check_spans_tr(ROTATED_SEQS[i]);
}

MU_TEST(test_corner_times_strictly_increase)
{
    int i;
    for (i = 0; i < N_PLAIN; ++i)
        check_strictly_increasing(PLAIN_SEQS[i]);
    for (i = 0; i < N_ROTATED; ++i)
        check_strictly_increasing(ROTATED_SEQS[i]);
}

MU_TEST(test_corner_stream_matches_uniform_path)
{
    int i;
    for (i = 0; i < N_PLAIN; ++i)
        check_matches_uniform(PLAIN_SEQS[i]);
}

MU_TEST(test_corner_stream_matches_uniform_path_rotated)
{
    int i;
    for (i = 0; i < N_ROTATED; ++i)
        check_matches_uniform(ROTATED_SEQS[i]);
}

MU_TEST(test_corner_stream_is_not_a_raster)
{
    int i;
    for (i = 0; i < N_PLAIN; ++i)
        check_not_a_raster(PLAIN_SEQS[i]);
    for (i = 0; i < N_ROTATED; ++i)
        check_not_a_raster(ROTATED_SEQS[i]);
}

MU_TEST_SUITE(test_corner_points_suite)
{
    pulseg_opts_init(
        &s_opts, GAMMA_HZ_PER_T, 3.0f, 1.0e7f, 1.0e11f, 1.0f, 10.0f, 0.1f, 10.0f);
    MU_RUN_TEST(test_corner_stream_spans_whole_tr);
    MU_RUN_TEST(test_corner_times_strictly_increase);
    MU_RUN_TEST(test_corner_stream_matches_uniform_path);
    MU_RUN_TEST(test_corner_stream_matches_uniform_path_rotated);
    MU_RUN_TEST(test_corner_stream_is_not_a_raster);
}

int test_corner_points_main(void)
{
    minunit_run = 0;
    minunit_fail = 0;
    minunit_assert = 0;
    minunit_status = 0;
    minunit_real_timer = 0;
    minunit_proc_timer = 0;

    MU_RUN_SUITE(test_corner_points_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
