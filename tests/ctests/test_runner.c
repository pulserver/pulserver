
/*
 * test_runner.c -- entry point for the pulseg unit test suite.
 *
 * Calls each test_*_main() which runs its own MU_RUN_SUITE / MU_REPORT.
 * A non-zero return from any suite indicates failure.
 */



#include <stdio.h>

// Only register and call the suite, do not define test logic here
// Forward declaration
int test_safety_grad_main(void);
int test_rf_stats_main(void);
int test_sequences_main(void);
int test_protocol_main(void);
int test_recon_main(void);
int test_ptx_getters_main(void);
int test_trid_labels_main(void);
int test_block_instance_main(void);
int test_pulsegen_cache_main(void);
int test_pulseq_binary_main(void);
int test_pulseq_ktraj_main(void);
int test_pulseq_rf_main(void);
int test_czt_main(void);
int test_chunk_main(void);

int main(void)
{
    int failed = 0;

    printf("==== test_safety_grad ====\n");
    failed += test_safety_grad_main();

    printf("\n==== test_rf_stats ====\n");
    failed += test_rf_stats_main();

    printf("\n==== test_sequences ====\n");
    failed += test_sequences_main();

    printf("\n==== test_protocol ====\n");
    failed += test_protocol_main();

    printf("\n==== test_recon ====\n");
    failed += test_recon_main();

    printf("\n==== test_ptx_getters ====\n");
    failed += test_ptx_getters_main();

    printf("\n==== test_trid_labels ====\n");
    failed += test_trid_labels_main();

    printf("\n==== test_block_instance ====\n");
    failed += test_block_instance_main();

    printf("\n==== test_pulsegen_cache ====\n");
    failed += test_pulsegen_cache_main();

    printf("\n==== test_pulseq_binary ====\n");
    failed += test_pulseq_binary_main();

    printf("\n==== test_pulseq_ktraj ====\n");
    failed += test_pulseq_ktraj_main();

    printf("\n==== test_pulseq_rf ====\n");
    failed += test_pulseq_rf_main();

    printf("\n==== test_czt ====\n");
    failed += test_czt_main();

    printf("\n==== test_chunk ====\n");
    failed += test_chunk_main();

    // printf("\n==== test_io ====\n");
    // failed += test_io_main();

    printf("\n");
    if (failed)
        printf("OVERALL: %d test(s) FAILED\n", failed);
    else
        printf("OVERALL: All suites PASSED.\n");

    return failed ? 1 : 0;
}
