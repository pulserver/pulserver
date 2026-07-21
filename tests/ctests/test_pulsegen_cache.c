/*
 * test_pulsegen_cache.c -- the pulsegen cache load must be information-
 *                          complete without the per-instance tables.
 *
 * pulseg_load_geninstructions_cache() reads COMMON + SHAPES only: no
 * INSTANCES (block/rf/grad/adc tables), no SCANLOOP (execution stream).
 * Everything the pulsegen pass resolves per (segment, block position) is
 * frozen into the segment definitions' initial-state records at parse time.
 *
 * The gate: every getter the pulsegen pass calls must return, from that
 * partial load, exactly what it returns from a full in-memory parse.  If it
 * does not, pulsegen would build hardware images that disagree with the
 * waveforms the scan cursor plays -- the failure mode that previously forced
 * the loader to pull SCANLOOP in as well.
 */
#include "test_helpers.h"

/* Assert the two collections agree on every pulsegen-path query. */
static void assert_pulsegen_view_equal(
    const pulseg_collection *full,
    const pulseg_collection *cached)
{
    pulseg_collection_info ci_full = PULSEG_COLLECTION_INFO_INIT;
    pulseg_collection_info ci_cache = PULSEG_COLLECTION_INFO_INIT;
    int seg, blk, axis;

    mu_assert(PULSEG_SUCCEEDED(pulseg_get_collection_info(full, &ci_full)), "info(full)");
    mu_assert(PULSEG_SUCCEEDED(pulseg_get_collection_info(cached, &ci_cache)), "info(cached)");
    mu_assert_int_eq(ci_full.num_subsequences, ci_cache.num_subsequences);
    mu_assert_int_eq(ci_full.num_segments, ci_cache.num_segments);
    mu_assert_int_eq(ci_full.max_adc_samples, ci_cache.max_adc_samples);
    mu_assert_int_eq(ci_full.total_readouts, ci_cache.total_readouts);

    for (seg = 0; seg < ci_full.num_segments; ++seg)
    {
        pulseg_segment_info si_full = PULSEG_SEGMENT_INFO_INIT;
        pulseg_segment_info si_cache = PULSEG_SEGMENT_INFO_INIT;

        mu_assert(PULSEG_SUCCEEDED(pulseg_get_segment_info(full, &si_full, seg)), "seg(full)");
        mu_assert(PULSEG_SUCCEEDED(pulseg_get_segment_info(cached, &si_cache, seg)), "seg(cached)");
        mu_assert_int_eq(si_full.duration_us, si_cache.duration_us);
        mu_assert_int_eq(si_full.num_blocks, si_cache.num_blocks);
        mu_assert_int_eq(si_full.start_block, si_cache.start_block);
        mu_assert_int_eq(si_full.pure_delay, si_cache.pure_delay);
        mu_assert_int_eq(si_full.has_trigger, si_cache.has_trigger);
        mu_assert_int_eq(si_full.trigger_type, si_cache.trigger_type);
        mu_assert_int_eq(si_full.trigger_delay_us, si_cache.trigger_delay_us);
        mu_assert_int_eq(si_full.trigger_duration_us, si_cache.trigger_duration_us);
        mu_assert_int_eq(si_full.is_nav, si_cache.is_nav);
        mu_assert_int_eq(si_full.rf_adc_gap_us, si_cache.rf_adc_gap_us);
        mu_assert_int_eq(si_full.adc_adc_gap_us, si_cache.adc_adc_gap_us);

        for (blk = 0; blk < si_full.num_blocks; ++blk)
        {
            pulseg_block_info bi_full = PULSEG_BLOCK_INFO_INIT;
            pulseg_block_info bi_cache = PULSEG_BLOCK_INFO_INIT;
            int fm_full, fm_cache, ns_full, ns_cache;

            mu_assert(
                PULSEG_SUCCEEDED(pulseg_get_block_info(full, &bi_full, seg, blk)),
                "block(full)");
            mu_assert(
                PULSEG_SUCCEEDED(pulseg_get_block_info(cached, &bi_cache, seg, blk)),
                "block(cached)");
            mu_assert(
                memcmp(&bi_full, &bi_cache, sizeof(bi_full)) == 0,
                "block info differs between full parse and pulsegen cache load");

            mu_assert_float_near(
                "rf initial amplitude",
                pulseg_get_rf_initial_amplitude_hz(full, seg, blk),
                pulseg_get_rf_initial_amplitude_hz(cached, seg, blk),
                0.0f);

            fm_full = pulseg_block_needs_freq_mod(full, &ns_full, seg, blk);
            fm_cache = pulseg_block_needs_freq_mod(cached, &ns_cache, seg, blk);
            mu_assert_int_eq(fm_full, fm_cache);
            mu_assert_int_eq(ns_full, ns_cache);

            if (bi_full.has_adc)
            {
                pulseg_adc_def ad_full = PULSEG_ADC_DEF_INIT;
                pulseg_adc_def ad_cache = PULSEG_ADC_DEF_INIT;
                mu_assert(
                    PULSEG_SUCCEEDED(pulseg_get_adc_def(full, &ad_full, bi_full.adc_def_id)),
                    "adc(full)");
                mu_assert(
                    PULSEG_SUCCEEDED(pulseg_get_adc_def(cached, &ad_cache, bi_cache.adc_def_id)),
                    "adc(cached)");
                mu_assert_int_eq(ad_full.dwell_ns, ad_cache.dwell_ns);
                mu_assert_int_eq(ad_full.num_samples, ad_cache.num_samples);
            }

            for (axis = 0; axis < 3; ++axis)
            {
                mu_assert_int_eq(
                    pulseg_get_grad_initial_shot_id(full, seg, blk, axis),
                    pulseg_get_grad_initial_shot_id(cached, seg, blk, axis));
                mu_assert_float_near(
                    "grad initial amplitude",
                    pulseg_get_grad_initial_amplitude_hz_per_m(full, seg, blk, axis),
                    pulseg_get_grad_initial_amplitude_hz_per_m(cached, seg, blk, axis),
                    0.0f);

                if (bi_full.has_grad[axis])
                {
                    int shots_full = 0, shots_cache = 0;
                    int samples_full = 0, samples_cache = 0;
                    float **wf_full;
                    float **wf_cache;
                    int shot, s;

                    wf_full =
                        pulseg_get_grad_amplitude(full, &shots_full, &samples_full, seg, blk, axis);
                    wf_cache = pulseg_get_grad_amplitude(
                        cached,
                        &shots_cache,
                        &samples_cache,
                        seg,
                        blk,
                        axis);
                    mu_assert(wf_full != NULL, "gradient waveform missing (full)");
                    mu_assert(wf_cache != NULL, "gradient waveform missing (cached)");
                    mu_assert_int_eq(shots_full, shots_cache);
                    mu_assert_int_eq(samples_full, samples_cache);

                    for (shot = 0; shot < shots_full; ++shot)
                    {
                        if (!wf_full[shot] || !wf_cache[shot])
                        {
                            mu_assert(
                                wf_full[shot] == NULL && wf_cache[shot] == NULL,
                                "gradient shot present in only one of the two loads");
                            continue;
                        }
                        for (s = 0; s < samples_full; ++s)
                            mu_assert_float_near(
                                "gradient sample",
                                wf_full[shot][s],
                                wf_cache[shot][s],
                                0.0f);
                    }

                    for (shot = 0; shot < shots_full; ++shot)
                    {
                        if (wf_full[shot])
                            PULSEG_FREE(wf_full[shot]);
                        if (wf_cache[shot])
                            PULSEG_FREE(wf_cache[shot]);
                    }
                    PULSEG_FREE(wf_full);
                    PULSEG_FREE(wf_cache);
                }
            }
        }
    }
}

/* Parse @p filename twice: once fully in memory, once through the cache the
 * predownload pass writes, loaded with the pulsegen section set only. */
static void check_fixture(const char *filename)
{
    pulseg_opts opts;
    pulseg_diagnostic diag = PULSEG_DIAGNOSTIC_INIT;
    pulseg_collection *full = NULL;
    pulseg_collection *cached = NULL;
    char seq_path[512];
    int rc;

    default_opts_init(&opts);
    (void)snprintf(seq_path, sizeof(seq_path), "%s%s", TEST_DATA_DIR, filename);

    /* cache_binary = 1: writes the sectioned cache beside the .seq, at the
     * DEFAULT extension (.pseg) -- the committed fixtures use the GE .pge
     * extension, so the write and the clear below cannot touch them. */
    rc = pulseg_read(&full, &diag, seq_path, &opts, 1, 0, 0, 1);
    mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_read with cache_binary failed");

    rc = pulseg_load_geninstructions_cache(&cached, seq_path);
    mu_assert(PULSEG_SUCCEEDED(rc), "pulseg_load_geninstructions_cache failed");

    /* The load must NOT have pulled in the per-instance tables. */
    mu_assert_int_eq(0, cached->descriptors[0].num_blocks);
    mu_assert(cached->descriptors[0].block_table == NULL, "INSTANCES leaked into pulsegen load");
    mu_assert_int_eq(0, cached->descriptors[0].exec_stream_len);
    mu_assert(
        cached->descriptors[0].exec_stream_block_idx == NULL,
        "SCANLOOP leaked into pulsegen load");

    assert_pulsegen_view_equal(full, cached);

    pulseg_collection_free(cached);
    pulseg_collection_free(full);
    (void)pulseg_clear_cache(seq_path);
}

MU_TEST(test_pulsegen_cache_matches_full_parse_gre)
{
    check_fixture("gre_2d_3sl_3avg.seq");
}

MU_TEST(test_pulsegen_cache_matches_full_parse_epi)
{
    check_fixture("epi_2d_3sl_3avg.seq");
}

MU_TEST(test_pulsegen_cache_matches_full_parse_rotated)
{
    check_fixture("mprage_noncart_3d_3sl_3avg_userotext1.seq");
}

/* Cross-subsequence chain: the dedup remap is rebuilt from COMMON + SHAPES on
 * the pulsegen path, so the global segment count and every per-segment view
 * must match the full parse's. */
MU_TEST(test_pulsegen_cache_matches_full_parse_chained)
{
    check_fixture("dedup_gre_pair.seq");
}

MU_TEST_SUITE(suite_pulsegen_cache)
{
    MU_RUN_TEST(test_pulsegen_cache_matches_full_parse_gre);
    MU_RUN_TEST(test_pulsegen_cache_matches_full_parse_epi);
    MU_RUN_TEST(test_pulsegen_cache_matches_full_parse_rotated);
    MU_RUN_TEST(test_pulsegen_cache_matches_full_parse_chained);
}

int test_pulsegen_cache_main(void)
{
    minunit_run = 0;
    minunit_fail = 0;
    minunit_assert = 0;
    minunit_status = 0;
    minunit_real_timer = 0;
    minunit_proc_timer = 0;

    MU_RUN_SUITE(suite_pulsegen_cache);
    MU_REPORT();
    return MU_EXIT_CODE;
}
