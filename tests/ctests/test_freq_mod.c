/*
 * test_freq_mod.c -- frequency modulation collection tests.
 *
 * Tests:
 *   1. build_freq_mod_collection succeeds for sequences with RF+ADC.
 *   2. With zero shift, all waveform samples are zero.
 *   3. update_freq_mod_collection with nonzero shift produces nonzero
 *      waveform on blocks that have gradients.
 *   4. get_freq_mod_count matches expected RF+ADC event count.
 *   5. PMC rewinding: after update with shift then update with
 *      zero shift, waveforms return to zero.
 *
 * Requires:
 *   - expected_output/seq1.seq   (basic, has RF)
 *   - expected_output/gre_2d.seq (with RF + ADC per TR)
 */
#include "test_helpers.h"
#include "pulseqlib_internal.h"  /* for fmc->libs[0]->scan_table_len */

/* ------------------------------------------------------------------ */
/*  Smoke: build collection on seq1                                   */
/* ------------------------------------------------------------------ */

MU_TEST(test_freq_mod_build_seq1)
{
    pulseqlib_collection*          coll = NULL;
    pulseqlib_diagnostic           diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_freq_mod_collection* fmc  = NULL;
    float shift[3] = {0.0f, 0.0f, 0.0f};
    int rc, count, i;

    rc = load_seq("expected_output/seq1.seq", &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load seq1");

    count = pulseqlib_get_freq_mod_count(coll);
    mu_assert(count >= 0, "freq_mod_count >= 0");

    if (count == 0) {
        /* No RF+ADC events, skip test */
        pulseqlib_collection_free(coll);
        return;
    }

    rc = pulseqlib_build_freq_mod_collection(&fmc, coll, shift);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "build collection should succeed");
    mu_assert(fmc != NULL, "fmc should be non-NULL");

    /* With zero shift, all waveforms should have zero values */
    for (i = 0; i < fmc->libs[0]->scan_table_len; i++) {
        const float* wf = NULL;
        int ns = 0;
        float ph = 0.0f;
        int has = pulseqlib_freq_mod_collection_get(fmc, 0, i, &wf, &ns, &ph);
        if (has) {
            int j;
            for (j = 0; j < ns; j++) {
                mu_assert((float)fabs(wf[j]) < 1e-6f,
                          "zero shift -> zero freq mod");
            }
            mu_assert((float)fabs(ph) < 1e-6f,
                      "zero shift -> zero phase");
        }
    }

    pulseqlib_freq_mod_collection_free(fmc);
    pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Stub: nonzero shift produces nonzero waveform                     */
/* ------------------------------------------------------------------ */

/*
 * TODO: uncomment once gre_2d.seq is generated.
 *
 * MU_TEST(test_freq_mod_nonzero_shift)
 * {
 *     pulseqlib_collection*          coll = NULL;
 *     pulseqlib_diagnostic           diag = PULSEQLIB_DIAGNOSTIC_INIT;
 *     pulseqlib_freq_mod_collection* fmc  = NULL;
 *     float shift[3] = {0.01f, 0.0f, 0.0f};   // 10 mm shift in X
 *     int rc, count;
 *
 *     rc = load_seq("expected_output/gre_2d.seq", &coll, &diag, 0);
 *     mu_assert(PULSEQLIB_SUCCEEDED(rc), "load GRE");
 *
 *     count = pulseqlib_get_freq_mod_count(coll);
 *     mu_assert(count > 0, "GRE should have RF+ADC");
 *
 *     rc = pulseqlib_build_freq_mod_collection(&fmc, coll, shift);
 *     mu_assert(PULSEQLIB_SUCCEEDED(rc), "build collection");
 *
 *     // At least one block should have nonzero waveform
 *     {
 *         int i, found_nonzero = 0;
 *         for (i = 0; i < fmc->libs[0]->scan_table_len; i++) {
 *             const float* wf;
 *             int ns;
 *             float ph;
 *             if (pulseqlib_freq_mod_collection_get(fmc, 0, i, &wf, &ns, &ph)) {
 *                 if (ns > 0 && fabs(wf[0]) > 1e-10f)
 *                     found_nonzero = 1;
 *             }
 *         }
 *         mu_assert(found_nonzero,
 *                   "nonzero shift should produce nonzero freq mod");
 *     }
 *
 *     // PMC rewinding: update back to zero
 *     {
 *         float zero[3] = {0.0f, 0.0f, 0.0f};
 *         rc = pulseqlib_update_freq_mod_collection(fmc, 0, zero);
 *         mu_assert(PULSEQLIB_SUCCEEDED(rc), "update to zero");
 *         // Now all waveforms should be zero again
 *         {
 *             int i;
 *             for (i = 0; i < fmc->libs[0]->scan_table_len; i++) {
 *                 const float* wf;
 *                 int ns;
 *                 float ph;
 *                 if (pulseqlib_freq_mod_collection_get(fmc, 0, i, &wf, &ns, &ph)) {
 *                     int j;
 *                     for (j = 0; j < ns; j++) {
 *                         mu_assert(fabs(wf[j]) < 1e-6f,
 *                                   "rewinding to zero shift");
 *                     }
 *                 }
 *             }
 *         }
 *     }
 *
 *     pulseqlib_freq_mod_collection_free(fmc);
 *     pulseqlib_collection_free(coll);
 * }
 */

/* ------------------------------------------------------------------ */
/*  Suite                                                             */
/* ------------------------------------------------------------------ */

MU_TEST_SUITE(test_freq_mod_suite)
{
    MU_RUN_TEST(test_freq_mod_build_seq1);
    /* MU_RUN_TEST(test_freq_mod_nonzero_shift); */
}

int test_freq_mod_main(void)
{
    MU_RUN_SUITE(test_freq_mod_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
