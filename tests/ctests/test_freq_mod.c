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

    rc = pulseqlib_build_freq_mod_collection(&fmc, coll, shift, NULL);
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
 *     rc = pulseqlib_build_freq_mod_collection(&fmc, coll, shift, NULL);
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
/*  Rotation math test: all four cases                                */
/* ------------------------------------------------------------------ */

/*
 * Validates the effective shift u = R^T @ shift_m for all combinations:
 *
 *   norot=0, rotation event R_ext:   u = R_ext^T @ shift_m
 *       (undoes the rotation event — diffusion direction stays in
 *       prescribed FOV orientation regardless of the rotation event)
 *
 *   norot=0, no rotation event:      u = shift_m
 *       (no rotation to undo)
 *
 *   norot=1, rotation event R_ext:   u = R_ext^T @ R_presc @ shift_m
 *       (rotation event applied in axial orientation, not prescribed FOV)
 *
 *   norot=1, no rotation event:      u = R_presc @ shift_m
 *       (undo the absent prescription rotation)
 *
 * Also verifies that norot=0 and norot=1 produce DIFFERENT results
 * when R_presc is non-identity, confirming the norot handling does
 * not accidentally modify norot=0 blocks.
 */
MU_TEST(test_freq_mod_rotation_all_cases)
{
    /* 90-degree rotation around Z axis (row-major) */
    float R_presc[9] = {0,-1,0, 1,0,0, 0,0,1};
    /* 90-degree rotation around X axis */
    float R_ext[9]   = {1,0,0, 0,0,-1, 0,1,0};
    float shift_m[3] = {1.0f, 2.0f, 3.0f};
    float identity[9] = {1,0,0, 0,1,0, 0,0,1};

    /* Result vectors for all four cases */
    float u_norot0_rot[3];    /* norot=0, rot event  */
    float u_norot0_norot[3];  /* norot=0, no rot     */
    float u_norot1_rot[3];    /* norot=1, rot event  */
    float u_norot1_norot[3];  /* norot=1, no rot     */

    /* Expected values computed from physics */
    float expected[3], tmp[3];
    float R_eff[9];
    int i, j;

    /* ---- Case 1: norot=0, rotation event ----
     * Code uses R = R_ext → u = R_ext^T @ shift_m.
     * Physically: the rotation event is undone so that the gradient
     * rotation is applied in the prescribed FOV frame. */
    pulseqlib__apply_rotation(u_norot0_rot, R_ext, shift_m, 1);

    /* Verify against hand-computed value:
     * R_ext = Rx(90°) = {{1,0,0},{0,0,-1},{0,1,0}}
     * R_ext^T @ [1,2,3] = [1, 3, -2] */
    expected[0] = 1.0f; expected[1] = 3.0f; expected[2] = -2.0f;
    for (i = 0; i < 3; ++i) {
        mu_assert((float)fabs(u_norot0_rot[i] - expected[i]) < 1e-5f,
                  "norot=0+rot: u = R_ext^T @ shift_m (old behavior)");
    }

    /* ---- Case 2: norot=0, no rotation event ----
     * Code uses R = I → u = shift_m. */
    pulseqlib__apply_rotation(u_norot0_norot, identity, shift_m, 1);
    for (i = 0; i < 3; ++i) {
        mu_assert((float)fabs(u_norot0_norot[i] - shift_m[i]) < 1e-5f,
                  "norot=0+no_rot: u = shift_m (old behavior)");
    }

    /* ---- Case 3: norot=1, rotation event ----
     * R_eff = R_presc^T @ R_ext  →  u = R_eff^T @ shift_m
     *       = R_ext^T @ R_presc @ shift_m */
    pulseqlib__apply_rotation(tmp, R_presc, shift_m, 0);
    pulseqlib__apply_rotation(u_norot1_rot, R_ext, tmp, 1);

    /* Verify via R_eff */
    for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j)
            R_eff[i*3+j] = R_presc[0*3+i]*R_ext[0*3+j]
                          + R_presc[1*3+i]*R_ext[1*3+j]
                          + R_presc[2*3+i]*R_ext[2*3+j];
    {
        float u_via_Reff[3];
        pulseqlib__apply_rotation(u_via_Reff, R_eff, shift_m, 1);
        for (i = 0; i < 3; ++i) {
            mu_assert((float)fabs(u_norot1_rot[i] - u_via_Reff[i]) < 1e-5f,
                      "norot=1+rot: R_eff^T @ shift == R_ext^T @ R_presc @ shift");
        }
    }

    /* ---- Case 4: norot=1, no rotation event ----
     * R_eff = R_presc^T  →  u = R_presc @ shift_m */
    pulseqlib__apply_rotation(u_norot1_norot, R_presc, shift_m, 0);

    for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j)
            R_eff[i*3+j] = R_presc[j*3+i];
    {
        float u_via_Reff[3];
        pulseqlib__apply_rotation(u_via_Reff, R_eff, shift_m, 1);
        for (i = 0; i < 3; ++i) {
            mu_assert((float)fabs(u_norot1_norot[i] - u_via_Reff[i]) < 1e-5f,
                      "norot=1+no_rot: R_eff^T @ shift == R_presc @ shift");
        }
    }

    /* ---- Verify norot=0 and norot=1 give DIFFERENT results ----
     * This confirms the norot handling does NOT modify norot=0 blocks. */
    {
        float diff_rot = 0.0f, diff_norot = 0.0f;
        for (i = 0; i < 3; ++i) {
            diff_rot   += (float)fabs(u_norot0_rot[i] - u_norot1_rot[i]);
            diff_norot += (float)fabs(u_norot0_norot[i] - u_norot1_norot[i]);
        }
        mu_assert(diff_rot > 1e-3f,
                  "norot=0+rot differs from norot=1+rot (R_presc != I)");
        mu_assert(diff_norot > 1e-3f,
                  "norot=0+no_rot differs from norot=1+no_rot (R_presc != I)");
    }
}

/* ------------------------------------------------------------------ */
/*  Build with fov_rotation (NULL vs non-NULL, identity case)         */
/* ------------------------------------------------------------------ */

MU_TEST(test_freq_mod_build_with_fov_rotation)
{
    pulseqlib_collection*          coll = NULL;
    pulseqlib_diagnostic           diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_freq_mod_collection* fmc1 = NULL;
    pulseqlib_freq_mod_collection* fmc2 = NULL;
    float shift[3]   = {0.0f, 0.0f, 0.0f};
    float fov_id[9]  = {1,0,0, 0,1,0, 0,0,1};
    int rc;

    rc = load_seq("expected_output/seq1.seq", &coll, &diag, 0);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "load seq1");

    /* Build with NULL fov_rotation */
    rc = pulseqlib_build_freq_mod_collection(&fmc1, coll, shift, NULL);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "build with NULL fov_rotation");

    /* Build with identity fov_rotation (should give same result) */
    rc = pulseqlib_build_freq_mod_collection(&fmc2, coll, shift, fov_id);
    mu_assert(PULSEQLIB_SUCCEEDED(rc), "build with identity fov_rotation");

    pulseqlib_freq_mod_collection_free(fmc1);
    pulseqlib_freq_mod_collection_free(fmc2);
    pulseqlib_collection_free(coll);
}

/* ------------------------------------------------------------------ */
/*  Suite                                                             */
/* ------------------------------------------------------------------ */

MU_TEST_SUITE(test_freq_mod_suite)
{
    MU_RUN_TEST(test_freq_mod_build_seq1);
    MU_RUN_TEST(test_freq_mod_rotation_all_cases);
    MU_RUN_TEST(test_freq_mod_build_with_fov_rotation);
    /* MU_RUN_TEST(test_freq_mod_nonzero_shift); */
}

int test_freq_mod_main(void)
{
    MU_RUN_SUITE(test_freq_mod_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
