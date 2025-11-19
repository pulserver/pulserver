#include <stdio.h>

/* Test entry points */
int test_seqfile_main(void);
int test_block_main(void);

int main(void)
{
    int failed = 0;

    printf("Running seqfile tests...\n");
    failed += test_seqfile_main();

    printf("Running block tests...\n");
    failed += test_block_main();

    if (failed)
        printf("Some tests FAILED (%d)\n", failed);
    else
        printf("All tests PASSED.\n");

    return failed;
}
