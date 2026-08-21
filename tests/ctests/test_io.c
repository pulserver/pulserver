/*
 * test_io.c -- IO and cache behavior tests.
 *
 * Covers:
 *   - MD5 signature validation for .seq loading
 *   - Stage cache loaders (check/geninstructions/scanloop)
 *   - Cache clear API behavior
 */

#include "test_helpers.h"

static void build_seq_path(char *out_path, size_t out_size, const char *filename)
{
    (void)snprintf(out_path, out_size, "%s%s", TEST_CORPUS_DIR, filename);
}

static void build_cache_path(char *out_path, size_t out_size, const char *seq_path)
{
    const char *dot;
    const char *slash_fwd;
    const char *slash_back;

    (void)snprintf(out_path, out_size, "%s", seq_path);

    dot = strrchr(out_path, '.');
    slash_fwd = strrchr(out_path, '/');
    slash_back = strrchr(out_path, '\\');
    if (dot && dot > slash_fwd && dot > slash_back)
    {
        (void)snprintf((char *)dot, out_size - (size_t)(dot - out_path), ".pge");
    }
    else
    {
        size_t len = strlen(out_path);
        if (len + 4 < out_size)
            strcat(out_path, ".pge");
    }
}

static int cache_file_exists(const char *cache_path)
{
    FILE *f = fopen(cache_path, "rb");
    if (!f)
        return 0;
    fclose(f);
    return 1;
}

MU_TEST(test_signature_valid_gre)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    int rc;

    gre_opts_init(&opts);
    rc = load_corpus_seq_with_signature_check(&coll, "gre_2d.seq", &opts);

    mu_assert(PULSEG_SUCCEEDED(rc), "signature check failed on valid sequence");
    mu_assert(coll != NULL, "collection must be allocated on valid signature");

    pulseg_collection_free(coll);
}

MU_TEST(test_signature_mismatch_gre)
{
    pulseg_opts opts;
    pulseg_collection *coll = NULL;
    int rc;

    gre_opts_init(&opts);
    rc = load_seq_with_signature_check(&coll, "malformed/gre_2d_corrupted.seq", &opts);

    mu_assert_int_eq(PULSEG_ERR_SIGNATURE_MISMATCH, rc);
    mu_assert(coll == NULL, "collection must remain NULL on signature mismatch");
}

MU_TEST(test_cache_stage_loaders_and_clear)
{
    pulseg_opts opts;
    pulseg_diagnostic diag = PULSEG_DIAGNOSTIC_INIT;
    pulseg_collection *coll = NULL;
    pulseg_collection *stage_coll = NULL;
    pulseg_collection_info info = PULSEG_COLLECTION_INFO_INIT;
    char seq_path[512];
    char cache_path[512];
    int rc;

    gre_opts_init(&opts);
    build_seq_path(seq_path, sizeof(seq_path), "gre_2d.seq");
    build_cache_path(cache_path, sizeof(cache_path), seq_path);

    rc = pulseg_clear_cache(seq_path);
    mu_assert_int_eq(PULSEG_SUCCESS, rc);
    mu_assert(!cache_file_exists(cache_path), "cache file should be absent after clear");

    rc = pulseg_read(
        &coll,
        &diag,
        seq_path,
        &opts,
        1, /* cache_binary */
        1, /* verify_signature */
        0, /* parse_labels */
        1);
    mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_read failed with cache enabled");
    mu_assert(cache_file_exists(cache_path), "cache file should be created by pulseg_read");

    pulseg_collection_free(coll);
    coll = NULL;

    rc = pulseg_load_geninstructions_cache(&stage_coll, seq_path);
    mu_assert_int_eq(PULSEG_SUCCESS, rc);
    rc = pulseg_get_collection_info(stage_coll, &info);
    mu_assert(PULSEG_SUCCEEDED(rc), "get_collection_info failed after geninstructions cache load");
    mu_assert(
        info.num_subsequences > 0,
        "geninstructions cache must contain at least one subsequence");
    pulseg_collection_free(stage_coll);
    stage_coll = NULL;

    rc = pulseg_load_scanloop_cache(&stage_coll, seq_path);
    mu_assert_int_eq(PULSEG_SUCCESS, rc);
    pulseg_collection_free(stage_coll);
    stage_coll = NULL;

    rc = pulseg_clear_cache(seq_path);
    mu_assert_int_eq(PULSEG_SUCCESS, rc);
    mu_assert(!cache_file_exists(cache_path), "cache file should be removed by clear_cache");

    rc = pulseg_load_geninstructions_cache(&stage_coll, seq_path);
    mu_assert(
        PULSEG_FAILED(rc),
        "load_geninstructions_cache should fail when cache file is missing");
    mu_assert(stage_coll == NULL, "stage collection should stay NULL on cache load failure");
}

MU_TEST_SUITE(suite_io)
{
    MU_RUN_TEST(test_signature_valid_gre);
    MU_RUN_TEST(test_signature_mismatch_gre);
    MU_RUN_TEST(test_cache_stage_loaders_and_clear);
}

int test_io_main(void)
{
    minunit_run = 0;
    minunit_fail = 0;
    minunit_assert = 0;
    minunit_status = 0;
    minunit_real_timer = 0;
    minunit_proc_timer = 0;

    MU_RUN_SUITE(suite_io);
    MU_REPORT();
    return MU_EXIT_CODE;
}
