#include <stdio.h>

/* Test entry points */
int test_seqfile_main(void);
int test_block_main(void);
int test_segments_main(void);
int test_concatenate_main(void);
int test_math_main(void);

int main(void)
{
    int failed = 0;

    printf("Running seqfile tests...\n");
    failed += test_seqfile_main();

    printf("Running block tests...\n");
    failed += test_block_main();

    printf("Running segment tests...\n");
    failed += test_segments_main();

    printf("Running concatenation tests...\n");
    failed += test_concatenate_main();

    printf("Running math tests...\n");
    failed += test_math_main();

    if (failed)
        printf("Some tests FAILED (%d)\n", failed);
    else
        printf("All tests PASSED.\n");

    return failed;
}
