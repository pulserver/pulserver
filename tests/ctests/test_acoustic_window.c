/*
 * test_acoustic_window.c -- the window pulseg_get_tr_gradient_waveforms hands
 *                           the acoustic pipeline.
 *
 * Sound pressure is read off one window per subsequence and the verdict is
 * taken to hold for the scan, so the window has to stand for the repetitions
 * it replaces rather than be one of them.  These tests state that as a
 * property of the waveform: at every sample, on every axis, the window drives
 * the coil at least as hard as any repetition it covers.
 */
#include "test_helpers.h"

#include "pulseg_waveforms.h"

static pulseg_opts s_opts;

/* Families whose repetitions differ, which is the only case where the
 * distinction between a window and a repetition is observable: a phase
 * encode steps, a flip angle ramps, a spiral arm turns. */
static const char *const VARYING_SEQS[] = {
    "gre_2d.seq",
    "gre_3d.seq",
    "fse_2d.seq",
    "epi_2d.seq",
    "mprage_3d.seq",
    "gre_radial_2d.seq",
    "gre_stack_of_spirals_3d.seq"};

#define N_VARYING (int)(sizeof(VARYING_SEQS) / sizeof(VARYING_SEQS[0]))

/* Larger than the float noise of two renderings of the same corner, and far
 * smaller than any encode step: an amplitude that exceeds the window by this
 * much is a repetition the window failed to cover. */
#define WINDOW_TOL_HZ_PER_M 1.0f

static int num_tr_instances(const pulseg_sequence_descriptor *desc)
{
    int tr_size = desc->tr_descriptor.tr_size;
    int n;

    if (tr_size <= 0)
        return 0;
    n = desc->num_blocks / tr_size;
    return (n > 0) ? n : 1;
}

/* Every repetition of the subsequence, rendered as it is actually played. */
static void check_window_bounds_every_repetition(const char *name)
{
    pulseg_collection *coll = NULL;
    pulseg_subseq_info sinfo = PULSEG_SUBSEQ_INFO_INIT;
    pulseg_tr_gradient_waveforms wf;
    pulseg__uniform_grad_waveforms inst;
    pulseg_diagnostic diag;
    const pulseg_sequence_descriptor *desc;
    int rc, ct, t, i, tr_size, ntr, covered;
    float w, a;

    rc = load_corpus_seq(&coll, name, &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_corpus_seq failed");
    desc = &coll->descriptors[0];
    tr_size = desc->tr_descriptor.tr_size;
    ntr = num_tr_instances(desc);

    rc = pulseg_get_subseq_info(coll, &sinfo, 0);
    mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_subseq_info failed");
    mu_assert(sinfo.num_canonical_trs >= 1, "a subsequence has at least one window");

    covered = 0;
    for (ct = 0; ct < sinfo.num_canonical_trs; ++ct)
    {
        wf = (pulseg_tr_gradient_waveforms)PULSEG_TR_GRADIENT_WAVEFORMS_INIT;
        rc = pulseg_get_tr_gradient_waveforms(coll, &wf, &diag, 0, ct);
        mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_tr_gradient_waveforms failed");

        for (t = 0; t < ntr; ++t)
        {
            memset(&inst, 0, sizeof(inst));
            rc = pulseg__get_gradient_waveforms_range(
                desc,
                &inst,
                &diag,
                t * tr_size,
                tr_size,
                PULSEG_AMP_ACTUAL,
                NULL,
                0,
                NULL);
            if (PULSEG_FAILED(rc))
                continue;
            if (inst.num_samples != wf.gx.num_samples)
            {
                /* A different set of waveforms: some other window covers it. */
                pulseg__uniform_grad_waveforms_free(&inst);
                continue;
            }
            for (i = 0; i < inst.num_samples; ++i)
            {
                w = (float)fabs((double)wf.gx.amplitude_hz_per_m[i]);
                a = (float)fabs((double)inst.gx[i]);
                mu_assert(
                    a <= w + WINDOW_TOL_HZ_PER_M,
                    "a repetition drove Gx harder than the window");
                w = (float)fabs((double)wf.gy.amplitude_hz_per_m[i]);
                a = (float)fabs((double)inst.gy[i]);
                mu_assert(
                    a <= w + WINDOW_TOL_HZ_PER_M,
                    "a repetition drove Gy harder than the window");
                w = (float)fabs((double)wf.gz.amplitude_hz_per_m[i]);
                a = (float)fabs((double)inst.gz[i]);
                mu_assert(
                    a <= w + WINDOW_TOL_HZ_PER_M,
                    "a repetition drove Gz harder than the window");
            }
            covered++;
            pulseg__uniform_grad_waveforms_free(&inst);
        }
        pulseg_tr_gradient_waveforms_free(&wf);
    }

    mu_assert(covered > 0, "no repetition was comparable with any window");
    pulseg_collection_free(coll);
}

static void test_the_acoustic_window_bounds_every_repetition_it_stands_for(void)
{
    int i;

    for (i = 0; i < N_VARYING; ++i)
        check_window_bounds_every_repetition(VARYING_SEQS[i]);
}

/* The property above is satisfied vacuously by a window that happens to equal
 * repetition 0 on a sequence where nothing varies.  A phase encode steps, so
 * on a Cartesian gradient echo the window must differ from the first
 * repetition somewhere -- otherwise the bound is being read off one shot. */
static void test_the_window_is_not_the_first_repetition(void)
{
    pulseg_collection *coll = NULL;
    pulseg_tr_gradient_waveforms wf = PULSEG_TR_GRADIENT_WAVEFORMS_INIT;
    pulseg__uniform_grad_waveforms first;
    pulseg_diagnostic diag;
    const pulseg_sequence_descriptor *desc;
    int rc, i, differs;

    rc = load_corpus_seq(&coll, "gre_2d.seq", &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_corpus_seq failed");
    desc = &coll->descriptors[0];

    rc = pulseg_get_tr_gradient_waveforms(coll, &wf, &diag, 0, 0);
    mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_tr_gradient_waveforms failed");

    memset(&first, 0, sizeof(first));
    rc = pulseg__get_gradient_waveforms_range(
        desc,
        &first,
        &diag,
        0,
        desc->tr_descriptor.tr_size,
        PULSEG_AMP_ACTUAL,
        NULL,
        0,
        NULL);
    mu_assert(PULSEG_SUCCEEDED(rc), "rendering the first repetition failed");
    mu_assert(first.num_samples == wf.gx.num_samples, "the two renderings must share a raster");

    differs = 0;
    for (i = 0; i < first.num_samples; ++i)
    {
        if ((float)fabs((double)(first.gy[i] - wf.gy.amplitude_hz_per_m[i])) > WINDOW_TOL_HZ_PER_M)
            differs = 1;
    }
    mu_assert(differs, "the window equals the first repetition on a stepping phase encode");

    pulseg__uniform_grad_waveforms_free(&first);
    pulseg_tr_gradient_waveforms_free(&wf);
    pulseg_collection_free(coll);
}

/* The count and the index are one contract: a caller loops to
 * num_canonical_trs, so exactly that many indices have to resolve. */
static void test_the_window_count_is_the_index_range(void)
{
    pulseg_collection *coll = NULL;
    pulseg_subseq_info sinfo = PULSEG_SUBSEQ_INFO_INIT;
    pulseg_tr_gradient_waveforms wf;
    pulseg_diagnostic diag;
    int rc, ct;

    rc = load_corpus_seq(&coll, "gre_stack_of_spirals_3d.seq", &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_corpus_seq failed");
    rc = pulseg_get_subseq_info(coll, &sinfo, 0);
    mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_get_subseq_info failed");

    for (ct = 0; ct < sinfo.num_canonical_trs; ++ct)
    {
        wf = (pulseg_tr_gradient_waveforms)PULSEG_TR_GRADIENT_WAVEFORMS_INIT;
        rc = pulseg_get_tr_gradient_waveforms(coll, &wf, &diag, 0, ct);
        mu_assert(PULSEG_SUCCEEDED(rc), "an index below the count must resolve");
        mu_assert(wf.gx.num_samples > 1, "a window must carry a waveform");
        pulseg_tr_gradient_waveforms_free(&wf);
    }

    wf = (pulseg_tr_gradient_waveforms)PULSEG_TR_GRADIENT_WAVEFORMS_INIT;
    rc = pulseg_get_tr_gradient_waveforms(coll, &wf, &diag, 0, sinfo.num_canonical_trs);
    mu_assert(PULSEG_FAILED(rc), "an index past the count must be refused");

    pulseg_collection_free(coll);
}

MU_TEST_SUITE(test_acoustic_window_suite)
{
    pulseg_opts_init(&s_opts, GAMMA_HZ_PER_T, 3.0f, 1.0e7f, 1.0e11f, 1.0f, 10.0f, 0.1f, 10.0f);
    MU_RUN_TEST(test_the_acoustic_window_bounds_every_repetition_it_stands_for);
    MU_RUN_TEST(test_the_window_is_not_the_first_repetition);
    MU_RUN_TEST(test_the_window_count_is_the_index_range);
}

int test_acoustic_window_main(void)
{
    minunit_run = 0;
    minunit_fail = 0;
    minunit_assert = 0;
    minunit_status = 0;
    minunit_real_timer = 0;
    minunit_proc_timer = 0;

    MU_RUN_SUITE(test_acoustic_window_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
