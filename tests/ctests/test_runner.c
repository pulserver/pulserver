
/*
 * test_runner.c -- entry point for the pulseqlib unit test suite.
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

    // printf("\n==== test_io ====\n");
    // failed += test_io_main();

    printf("\n");
    if (failed)
        printf("OVERALL: %d test(s) FAILED\n", failed);
    else
        printf("OVERALL: All suites PASSED.\n");

    return failed ? 1 : 0;
}
