/*
 * test_runner.c -- entry point for the pulseqlib unit test suite.
 *
 * Calls each test_*_main() which runs its own MU_RUN_SUITE / MU_REPORT.
 * A non-zero return from any suite indicates failure.
 *
 * NOTE: We deliberately avoid including minunit.h here because this
 * file does not define any MU_TEST / MU_TEST_SUITE, and the static
 * globals in minunit.h would trigger -Wunused-variable / -Wunused-function.
 */
#include <stdio.h>

/* Forward declarations -- defined in test_helpers.h and the test files */
int test_safety_grad_main(void);
int test_rf_stats_main(void);
int test_sequences_main(void);

int main(void)
{
    int failed = 0;

    printf("==== test_safety_grad ====\n");
    failed += test_safety_grad_main();

    printf("\n==== test_rf_stats ====\n");
    failed += test_rf_stats_main();

    printf("\n==== test_sequences ====\n");
    failed += test_sequences_main();

    printf("\n");
    if (failed)
        printf("OVERALL: %d test(s) FAILED\n", failed);
    else
        printf("OVERALL: All suites PASSED.\n");

    return failed ? 1 : 0;
}
