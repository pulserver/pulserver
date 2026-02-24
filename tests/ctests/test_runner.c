/*
 * test_runner.c -- entry point for the pulseqlib unit test suite.
 *
 * Calls each test_*_main() which runs its own MU_RUN_SUITE / MU_REPORT.
 * A non-zero return from any suite indicates failure.
 */
#include "test_helpers.h"

int main(void)
{
    int failed = 0;

    printf("==== test_error ====\n");
    failed += test_error_main();

    printf("\n==== test_load ====\n");
    failed += test_load_main();

    printf("\n==== test_structure ====\n");
    failed += test_structure_main();

    printf("\n==== test_segments ====\n");
    failed += test_segments_main();

    printf("\n==== test_consistency ====\n");
    failed += test_consistency_main();

    printf("\n==== test_cursor ====\n");
    failed += test_cursor_main();

    printf("\n==== test_waveforms ====\n");
    failed += test_waveforms_main();

    printf("\n==== test_rf_stats ====\n");
    failed += test_rf_stats_main();

    printf("\n==== test_safety_grad ====\n");
    failed += test_safety_grad_main();

    printf("\n==== test_safety_acoustic ====\n");
    failed += test_safety_acoustic_main();

    printf("\n==== test_safety_pns ====\n");
    failed += test_safety_pns_main();

    printf("\n==== test_freq_mod ====\n");
    failed += test_freq_mod_main();

    printf("\n==== test_labels ====\n");
    failed += test_labels_main();

    printf("\n");
    if (failed)
        printf("OVERALL: %d test(s) FAILED\n", failed);
    else
        printf("OVERALL: All suites PASSED.\n");

    return failed ? 1 : 0;
}
