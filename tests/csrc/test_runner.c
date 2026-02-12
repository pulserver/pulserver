#include <stdio.h>

/* Test entry points */

int main(void)
{
    int failed = 0;

    if (failed)
        printf("Some tests FAILED (%d)\n", failed);
    else
        printf("All tests PASSED.\n");

    return failed;
}
