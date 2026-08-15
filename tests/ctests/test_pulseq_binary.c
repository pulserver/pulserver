/*
 * test_pulseq_binary.c -- the binary reader against the text reader.
 *
 * One question, asked per fixture: reading the same sequence stored both
 * ways must build the same pulseq_file. Fixtures come in pairs (see
 * tests/utils/make_binary_fixtures.py) because a .seq read back and
 * rewritten is not byte-stable, so a stock .seq and a binary derived from
 * it would be two different sequences and the comparison would prove
 * nothing.
 *
 * Structural fields (sizes, ids, flags, shape lengths) are compared for
 * exact equality. Reals are compared exactly too: the binary format carries
 * more precision than the text it is written beside, but every value here
 * lands in a float in the end, and in practice that lands on the same float.
 * If a fixture ever legitimately diverges in the last bit, that is a real
 * finding and belongs in this file as an explicit tolerance, not as a
 * blanket one applied everywhere.
 */

#include "test_helpers.h"

#include "pulseq.h"

#define BIN_DIR TEST_ROOT_DIR "/tests/utils/expected/binary/"

static void bin_paths(char *text_path, char *bin_path, size_t n, const char *stem)
{
    (void)snprintf(text_path, n, "%s%s.seq", BIN_DIR, stem);
    (void)snprintf(bin_path, n, "%s%s.bin", BIN_DIR, stem);
}

/* Compare every row of two float libraries of the same shape. */
static int rows_equal(const float *a, const float *b, int rows, int cols)
{
    int i;
    if (rows > 0 && (!a || !b))
        return 0;
    for (i = 0; i < rows * cols; ++i)
    {
        if (a[i] != b[i])
            return 0;
    }
    return 1;
}

static void check_pair(const char *stem)
{
    char text_path[1024];
    char bin_path[1024];
    pulseq_file t;
    pulseq_file b;
    pulseq_raster raster;
    int rc;
    int i;
    int j;

    raster.rf_us = 2.0f;
    raster.grad_us = 10.0f;
    raster.block_us = 10.0f;

    bin_paths(text_path, bin_path, sizeof(text_path), stem);

    mu_assert(pulseq_file_is_binary(text_path) == 0, "text fixture detected as binary");
    mu_assert(pulseq_file_is_binary(bin_path) == 1, "binary fixture not detected");

    pulseq_file_init(&t, &raster);
    pulseq_file_init(&b, &raster);

    rc = pulseq_read(&t, text_path);
    mu_assert(PULSEQ_SUCCEEDED(rc), "text fixture failed to parse");
    /* pulseq_read dispatches on content, so this exercises the dispatch too. */
    rc = pulseq_read(&b, bin_path);
    mu_assert(PULSEQ_SUCCEEDED(rc), "binary fixture failed to parse");

    mu_assert_int_eq(t.version_combined, b.version_combined);
    mu_assert_int_eq(t.num_definitions, b.num_definitions);
    mu_assert_int_eq(t.num_blocks, b.num_blocks);
    mu_assert_int_eq(t.rf_library_size, b.rf_library_size);
    mu_assert_int_eq(t.grad_library_size, b.grad_library_size);
    mu_assert_int_eq(t.adc_library_size, b.adc_library_size);
    mu_assert_int_eq(t.shapes_library_size, b.shapes_library_size);
    mu_assert_int_eq(t.extensions_library_size, b.extensions_library_size);
    mu_assert_int_eq(t.trigger_library_size, b.trigger_library_size);
    mu_assert_int_eq(t.rotation_library_size, b.rotation_library_size);
    mu_assert_int_eq(t.labelset_library_size, b.labelset_library_size);
    mu_assert_int_eq(t.labelinc_library_size, b.labelinc_library_size);
    mu_assert_int_eq(t.soft_delay_library_size, b.soft_delay_library_size);
    mu_assert_int_eq(t.rf_shim_library_size, b.rf_shim_library_size);
    mu_assert_int_eq(t.extension_lut_size, b.extension_lut_size);

    for (i = 0; i < 8; ++i)
        mu_assert_int_eq(t.extension_map[i], b.extension_map[i]);
    for (i = 0; i < 22; ++i)
        mu_assert_int_eq(t.is_label_defined[i], b.is_label_defined[i]);
    for (i = 0; i < 8; ++i)
        mu_assert_int_eq(t.is_delay_defined[i], b.is_delay_defined[i]);

    /* The reserved definitions are what peek_scan_time and the chain walk
     * read, so they matter even when nothing else is touched. */
    mu_assert_float_near(
        "GradientRasterTime",
        t.reserved_definitions_library.gradient_raster_time,
        b.reserved_definitions_library.gradient_raster_time,
        0.0f);
    mu_assert_float_near(
        "RadiofrequencyRasterTime",
        t.reserved_definitions_library.radiofrequency_raster_time,
        b.reserved_definitions_library.radiofrequency_raster_time,
        0.0f);
    mu_assert_float_near(
        "BlockDurationRaster",
        t.reserved_definitions_library.block_duration_raster,
        b.reserved_definitions_library.block_duration_raster,
        0.0f);
    mu_assert_float_near(
        "TotalDuration",
        t.reserved_definitions_library.total_duration,
        b.reserved_definitions_library.total_duration,
        0.0f);
    mu_assert(
        strcmp(
            t.reserved_definitions_library.name,
            b.reserved_definitions_library.name) == 0,
        "Name definition differs");
    mu_assert(
        strcmp(
            t.reserved_definitions_library.next_sequence,
            b.reserved_definitions_library.next_sequence) == 0,
        "NextSequence definition differs");

    /* The generic definitions are exchanged as strings, so the two parses
     * have to agree on them as strings and not merely as numbers. */
    for (i = 0; i < t.num_definitions; ++i)
    {
        mu_assert(
            strcmp(t.definitions_library[i].name, b.definitions_library[i].name) == 0,
            "definition name differs");
        mu_assert_int_eq(
            t.definitions_library[i].value_size,
            b.definitions_library[i].value_size);
        for (j = 0; j < t.definitions_library[i].value_size; ++j)
        {
            mu_assert(
                strcmp(
                    t.definitions_library[i].value[j],
                    b.definitions_library[i].value[j]) == 0,
                "definition value differs");
        }
    }

    mu_assert(
        rows_equal((const float *)t.block_library, (const float *)b.block_library,
                   t.num_blocks, 7),
        "block library differs");
    mu_assert(
        rows_equal((const float *)t.rf_library, (const float *)b.rf_library,
                   t.rf_library_size, 10),
        "rf library differs");
    mu_assert(
        rows_equal((const float *)t.grad_library, (const float *)b.grad_library,
                   t.grad_library_size, 7),
        "grad library differs");
    mu_assert(
        rows_equal((const float *)t.adc_library, (const float *)b.adc_library,
                   t.adc_library_size, 8),
        "adc library differs");
    mu_assert(
        rows_equal((const float *)t.extensions_library, (const float *)b.extensions_library,
                   t.extensions_library_size, 3),
        "extension chain differs");
    mu_assert(
        rows_equal((const float *)t.trigger_library, (const float *)b.trigger_library,
                   t.trigger_library_size, 4),
        "trigger library differs");
    mu_assert(
        rows_equal((const float *)t.rotation_quaternion_library,
                   (const float *)b.rotation_quaternion_library,
                   t.rotation_library_size, 4),
        "rotation library differs");
    mu_assert(
        rows_equal((const float *)t.labelset_library, (const float *)b.labelset_library,
                   t.labelset_library_size, 2),
        "labelset library differs");
    mu_assert(
        rows_equal((const float *)t.labelinc_library, (const float *)b.labelinc_library,
                   t.labelinc_library_size, 2),
        "labelinc library differs");
    mu_assert(
        rows_equal((const float *)t.soft_delay_library, (const float *)b.soft_delay_library,
                   t.soft_delay_library_size, 4),
        "soft delay library differs");

    for (i = 0; i < t.rf_library_size; ++i)
    {
        mu_assert(t.rf_use_tags != NULL && b.rf_use_tags != NULL, "rf use tags missing");
        mu_assert_int_eq(t.rf_use_tags[i], b.rf_use_tags[i]);
    }

    for (i = 0; i < t.shapes_library_size; ++i)
    {
        mu_assert_int_eq(
            t.shapes_library[i].num_uncompressed_samples,
            b.shapes_library[i].num_uncompressed_samples);
        mu_assert_int_eq(t.shapes_library[i].num_samples, b.shapes_library[i].num_samples);
        mu_assert(
            rows_equal(t.shapes_library[i].samples, b.shapes_library[i].samples,
                       t.shapes_library[i].num_samples, 1),
            "shape samples differ");
    }

    for (i = 0; i < t.rf_shim_library_size; ++i)
    {
        mu_assert_int_eq(
            t.rf_shim_library[i].num_channels,
            b.rf_shim_library[i].num_channels);
        for (j = 0; j < 2 * t.rf_shim_library[i].num_channels; ++j)
        {
            mu_assert(
                t.rf_shim_library[i].values[j] == b.rf_shim_library[i].values[j],
                "rf shim values differ");
        }
    }

    pulseq_file_free(&t);
    pulseq_file_free(&b);
}

MU_TEST(test_binary_matches_text_basic)
{
    check_pair("basic");
}

MU_TEST(test_binary_matches_text_arbgrad)
{
    check_pair("arbgrad");
}

MU_TEST(test_binary_matches_text_rotations)
{
    check_pair("rotations");
}

MU_TEST(test_binary_matches_text_rfshims)
{
    check_pair("rfshims");
}

MU_TEST(test_binary_matches_text_labelset)
{
    check_pair("labelset");
}

MU_TEST(test_binary_matches_text_labelinc)
{
    check_pair("labelinc");
}

/*
 * Definitions come first in the file, so the definitions-only reader stops
 * as soon as it has them. This is what backs pulseg_peek_scan_time() in file
 * mode; the point of the test is that it agrees with the full read rather
 * than that it is fast.
 */
MU_TEST(test_binary_definitions_only_matches_full_read)
{
    char text_path[1024];
    char bin_path[1024];
    pulseq_file full;
    pulseq_file peek;
    pulseq_file text_peek;
    pulseq_raster raster;
    int rc;

    raster.rf_us = 2.0f;
    raster.grad_us = 10.0f;
    raster.block_us = 10.0f;

    bin_paths(text_path, bin_path, sizeof(text_path), "labelset");

    pulseq_file_init(&full, &raster);
    pulseq_file_init(&peek, &raster);
    pulseq_file_init(&text_peek, &raster);

    rc = pulseq_read(&full, bin_path);
    mu_assert(PULSEQ_SUCCEEDED(rc), "binary full read failed");
    /* Dispatches to the binary path on content, exactly as file mode does. */
    rc = pulseq_read_definitions_only(&peek, bin_path);
    mu_assert(PULSEQ_SUCCEEDED(rc), "binary definitions-only read failed");
    rc = pulseq_read_definitions_only(&text_peek, text_path);
    mu_assert(PULSEQ_SUCCEEDED(rc), "text definitions-only read failed");

    mu_assert_int_eq(full.num_definitions, peek.num_definitions);
    mu_assert_int_eq(full.version_combined, peek.version_combined);
    mu_assert_float_near(
        "TotalDuration (full vs peek)",
        full.reserved_definitions_library.total_duration,
        peek.reserved_definitions_library.total_duration,
        0.0f);
    /* And the peek agrees across formats, which is the property file mode
     * actually depends on. */
    mu_assert_int_eq(text_peek.num_definitions, peek.num_definitions);
    mu_assert_float_near(
        "TotalDuration (text vs binary peek)",
        text_peek.reserved_definitions_library.total_duration,
        peek.reserved_definitions_library.total_duration,
        0.0f);
    mu_assert_float_near(
        "BlockDurationRaster (text vs binary peek)",
        text_peek.reserved_definitions_library.block_duration_raster,
        peek.reserved_definitions_library.block_duration_raster,
        0.0f);

    /* A peek must not decode the libraries it skipped past. */
    mu_assert_int_eq(0, peek.num_blocks);
    mu_assert_int_eq(0, peek.rf_library_size);

    pulseq_file_free(&full);
    pulseq_file_free(&peek);
    pulseq_file_free(&text_peek);
}

/* A text file must not be mistaken for binary, or the other way round. */
MU_TEST(test_binary_detection_rejects_non_binary)
{
    mu_assert_int_eq(0, pulseq_file_is_binary(TEST_DATA_DIR "00_basic_rfstat.seq"));
    mu_assert_int_eq(0, pulseq_file_is_binary(BIN_DIR "does_not_exist.bin"));
}

MU_TEST_SUITE(test_pulseq_binary_suite)
{
    MU_RUN_TEST(test_binary_matches_text_basic);
    MU_RUN_TEST(test_binary_matches_text_arbgrad);
    MU_RUN_TEST(test_binary_matches_text_rotations);
    MU_RUN_TEST(test_binary_matches_text_rfshims);
    MU_RUN_TEST(test_binary_matches_text_labelset);
    MU_RUN_TEST(test_binary_matches_text_labelinc);
    MU_RUN_TEST(test_binary_definitions_only_matches_full_read);
    MU_RUN_TEST(test_binary_detection_rejects_non_binary);
}

int test_pulseq_binary_main(void)
{
    MU_RUN_SUITE(test_pulseq_binary_suite);
    MU_REPORT();
    return minunit_fail;
}
