/*
 * test_raster.c -- raster alignment of event times against a scanner's grid.
 *
 * Every fixture declares RF 1 us and gradient/block 10 us and is played
 * against a system twice as coarse, so the declared-versus-system grid check
 * passes and what these assert is pulseg_check_raster_alignment() alone.
 */
#include "test_helpers.h"

static pulseg_opts s_opts;
static pulseg_diagnostic s_diag;

/** RF 2 us, gradient 20 us, ADC 0.2 us, block 20 us. */
static void coarse_opts_init(pulseg_opts *opts)
{
    pulseg_opts_init(opts, GAMMA_HZ_PER_T, 3.0f, 1.0e6f, 1.0e9f, 2.0f, 20.0f, 0.2f, 20.0f);
}

/**
 * Load @p filename against the coarse system and assert the alignment check
 * returns @p expected_code.
 */
static void run_alignment_check(const char *filename, int expected_code)
{
    pulseg_collection *coll = NULL;
    int rc;

    coarse_opts_init(&s_opts);
    rc = load_seq(&coll, filename, &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_check_raster_alignment(coll, &s_diag, &s_opts);
    mu_assert_int_eq(expected_code, rc);

    pulseg_collection_free(coll);
}

MU_TEST(test_a_sequence_on_a_coarser_system_raster_passes)
{
    run_alignment_check("20_raster_ok_on_coarse_system.seq", PULSEG_SUCCESS);
}

MU_TEST(test_an_rf_delay_off_the_system_raster_is_caught)
{
    run_alignment_check("21_raster_rf_delay_misaligned.seq", PULSEG_ERR_RASTER_ALIGNMENT);
}

MU_TEST(test_a_trapezoid_ramp_off_the_system_raster_is_caught)
{
    run_alignment_check("22_raster_trap_ramp_misaligned.seq", PULSEG_ERR_RASTER_ALIGNMENT);
}

MU_TEST(test_a_block_duration_off_the_system_raster_is_caught)
{
    run_alignment_check("23_raster_block_duration_misaligned.seq", PULSEG_ERR_RASTER_ALIGNMENT);
}

MU_TEST(test_an_event_ending_after_its_block_is_caught)
{
    run_alignment_check("24_raster_block_duration_overrun.seq", PULSEG_ERR_BLOCK_DURATION_OVERRUN);
}

/*
 * The grid check the collection already ran accepts a declared raster finer
 * than the system's, which is what leaves these times to catch: asserting it
 * passes the very file the alignment check rejects keeps the two questions
 * from being confused for one.
 */
MU_TEST(test_the_declared_grid_check_admits_what_alignment_rejects)
{
    pulseg_collection *coll = NULL;
    int rc;

    coarse_opts_init(&s_opts);
    rc = load_seq(&coll, "22_raster_trap_ramp_misaligned.seq", &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "a finer declared raster must survive the grid check");

    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_check_raster_alignment(coll, &s_diag, &s_opts);
    mu_assert_int_eq(PULSEG_ERR_RASTER_ALIGNMENT, rc);
    mu_assert(s_diag.message[0] != '\0', "a violation must name the offending value");

    pulseg_collection_free(coll);
}

/*
 * The same file against the raster it was written for. What the check reads is
 * the opts it is handed, which is what lets a design tool ask it offline.
 */
MU_TEST(test_alignment_is_judged_against_the_opts_it_is_given)
{
    pulseg_collection *coll = NULL;
    pulseg_opts design_opts;
    int rc;

    pulseg_opts_init(&design_opts, GAMMA_HZ_PER_T, 3.0f, 1.0e6f, 1.0e9f, 1.0f, 10.0f, 0.1f, 10.0f);
    rc = load_seq(&coll, "22_raster_trap_ramp_misaligned.seq", &design_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "load_seq failed");

    pulseg_diagnostic_init(&s_diag);
    rc = pulseg_check_raster_alignment(coll, &s_diag, &design_opts);
    mu_assert_int_eq(PULSEG_SUCCESS, rc);

    pulseg_collection_free(coll);
}

/*
 * A 1-based id names a row in another library, and a file is free to name one
 * that is not there. Every consumer indexes with what the reader hands it, so
 * refusing the file is the reader's job -- reading it and letting a consumer
 * walk off the end is what this asserts against.
 */
static void assert_refused(const char *filename)
{
    pulseg_collection *coll = NULL;
    int rc;

    coarse_opts_init(&s_opts);
    rc = load_seq(&coll, filename, &s_opts);
    mu_assert(PULSEG_FAILED(rc), "a dangling id must not read as a valid sequence");

    if (PULSEG_SUCCEEDED(rc))
        pulseg_collection_free(coll);
}

MU_TEST(test_a_block_naming_an_absent_event_is_refused)
{
    assert_refused("malformed/25_dangling_rf_id.seq");
}

MU_TEST(test_an_extension_naming_an_absent_rotation_is_refused)
{
    assert_refused("malformed/26_dangling_rotation_ref.seq");
}

MU_TEST(test_an_extension_chain_running_off_its_library_is_refused)
{
    assert_refused("malformed/27_dangling_extension_link.seq");
}

/*
 * The other side of the refusals above. An extension type this reader has no
 * name for carries a reference into a library it cannot identify, so it cannot
 * judge that reference either -- and never dereferences it. Refusing the file
 * would reject every vendor extension in existence.
 */
MU_TEST(test_an_extension_this_reader_does_not_know_is_kept)
{
    pulseg_collection *coll = NULL;
    int rc;

    coarse_opts_init(&s_opts);
    rc = load_seq(&coll, "28_unknown_extension_type.seq", &s_opts);
    mu_assert(PULSEG_SUCCEEDED(rc), "an unknown extension type must not fail the read");

    pulseg_collection_free(coll);
}

MU_TEST_SUITE(raster_suite)
{
    MU_RUN_TEST(test_a_sequence_on_a_coarser_system_raster_passes);
    MU_RUN_TEST(test_an_rf_delay_off_the_system_raster_is_caught);
    MU_RUN_TEST(test_a_trapezoid_ramp_off_the_system_raster_is_caught);
    MU_RUN_TEST(test_a_block_duration_off_the_system_raster_is_caught);
    MU_RUN_TEST(test_an_event_ending_after_its_block_is_caught);
    MU_RUN_TEST(test_the_declared_grid_check_admits_what_alignment_rejects);
    MU_RUN_TEST(test_alignment_is_judged_against_the_opts_it_is_given);
    MU_RUN_TEST(test_a_block_naming_an_absent_event_is_refused);
    MU_RUN_TEST(test_an_extension_naming_an_absent_rotation_is_refused);
    MU_RUN_TEST(test_an_extension_chain_running_off_its_library_is_refused);
    MU_RUN_TEST(test_an_extension_this_reader_does_not_know_is_kept);
}

int test_raster_main(void)
{
    MU_RUN_SUITE(raster_suite);
    MU_REPORT();
    return minunit_fail;
}
