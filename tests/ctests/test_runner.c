
/*
 * test_runner.c -- entry point for the pulseg unit test suite.
 *
 * With no arguments every suite runs. Arguments name the suites to run,
 * with or without the "test_" prefix; "--list" prints one suite name per
 * line and exits. Each suite is a test_*_main() running its own
 * MU_RUN_SUITE / MU_REPORT; a non-zero exit means a suite failed.
 */

#include <stdio.h>
#include <string.h>

/* Only register and call the suites, do not define test logic here. */
int test_safety_grad_main(void);
int test_rf_stats_main(void);
int test_sequences_main(void);
int test_protocol_main(void);
int test_ptx_getters_main(void);
int test_trid_labels_main(void);
int test_block_instance_main(void);
int test_pulsegen_cache_main(void);
int test_pulseq_binary_main(void);
int test_pulseq_ktraj_main(void);
int test_pulseq_rf_main(void);
int test_czt_main(void);
int test_chunk_main(void);

typedef struct
{
    const char *name;
    int (*run)(void);
} test_suite;

static const test_suite suites[] = {
    {"test_safety_grad", test_safety_grad_main},
    {"test_rf_stats", test_rf_stats_main},
    {"test_sequences", test_sequences_main},
    {"test_protocol", test_protocol_main},
    {"test_ptx_getters", test_ptx_getters_main},
    {"test_trid_labels", test_trid_labels_main},
    {"test_block_instance", test_block_instance_main},
    {"test_pulsegen_cache", test_pulsegen_cache_main},
    {"test_pulseq_binary", test_pulseq_binary_main},
    {"test_pulseq_ktraj", test_pulseq_ktraj_main},
    {"test_pulseq_rf", test_pulseq_rf_main},
    {"test_czt", test_czt_main},
    {"test_chunk", test_chunk_main},
};

#define NUM_SUITES (sizeof(suites) / sizeof(suites[0]))

static const test_suite *find_suite(const char *name)
{
    size_t i;

    for (i = 0; i < NUM_SUITES; i++)
    {
        if (strcmp(name, suites[i].name) == 0 || strcmp(name, suites[i].name + 5) == 0)
            return &suites[i];
    }
    return NULL;
}

static int run_suite(const test_suite *suite)
{
    printf("==== %s ====\n", suite->name);
    return suite->run();
}

int main(int argc, char **argv)
{
    int failed = 0;
    size_t i;
    int a;

    if (argc > 1 && strcmp(argv[1], "--list") == 0)
    {
        for (i = 0; i < NUM_SUITES; i++)
            printf("%s\n", suites[i].name);
        return 0;
    }

    if (argc > 1)
    {
        for (a = 1; a < argc; a++)
        {
            const test_suite *suite = find_suite(argv[a]);
            if (suite == NULL)
            {
                fprintf(stderr, "unknown suite: %s (--list prints them)\n", argv[a]);
                return 2;
            }
            failed += run_suite(suite);
            printf("\n");
        }
    }
    else
    {
        for (i = 0; i < NUM_SUITES; i++)
        {
            failed += run_suite(&suites[i]);
            printf("\n");
        }
    }

    if (failed)
        printf("OVERALL: %d test(s) FAILED\n", failed);
    else
        printf("OVERALL: All suites PASSED.\n");

    return failed ? 1 : 0;
}
