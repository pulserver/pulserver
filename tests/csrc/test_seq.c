#include "minunit.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "pulseqlib.h"
#include "pulseqlib_methods.h"

static pulseqlib_SeqFile* load_seq(char* filePath) {
    char seq_path[1024];
    pulseqlib_SeqFile* seq;
    pulseqlib_Opts opts;

    /* Allocate memory for the sequence file */
    seq = (pulseqlib_SeqFile*)ALLOC(sizeof(pulseqlib_SeqFile));
    if (!seq) return NULL;
    
    /* Create the full path to the sequence file */
    snprintf(seq_path, sizeof(seq_path), "%s/%s", TEST_ROOT_DIR, filePath);
    
    /* Initialize the sequence file structure */
    pulseqlib_optsInit(&opts, 3.0f, 40.0f, 150.0f, 1.0f, 10.0f, 0.1f, 10.0f);
    pulseqlib_seqFileInit(seq, &opts);
    pulseqlib_optsFree(&opts);

    /* Read the sequence data */
    pulseqlib_readSeq(seq, seq_path);
    return seq;
}

/* END UTILS */

MU_TEST(test_basic) {
    pulseqlib_SeqFile* seq = load_seq("expected_output/seq1.seq");
    mu_assert(seq != NULL, "Failed to load sequence file");
    mu_assert(seq->versionMajor == 1, "Sequence version major should be 1 (Pulseq v1.x.x)");
    mu_assert(seq->versionMinor == 5, "Sequence version minor should be 5 (Pulseq vx.5.x)");
    mu_assert(seq->versionRevision == 0, "Sequence version revision should be 0 (Pulseq vx.x.0)");
    mu_assert(seq->versionCombined == 1005000, "Sequence combined version should be 1005000 (Pulseq v1.5.0)");
    mu_assert(seq->numBlocks == 7, "Sequence should have exactly 7 blocks");
    pulseqlib_seqFileFree(seq);
}

MU_TEST(test_definitions) {
    pulseqlib_SeqFile* seq = load_seq("expected_output/seq1.seq");
    mu_assert(seq != NULL, "Failed to load sequence file");
    mu_assert(seq->numDefinitions == 5, "Sequence should have exactly 5 definitions");
    mu_assert(seq->definitionsLibrary != NULL, "Definitions library should not be NULL");
    
    mu_assert(strcmp(seq->definitionsLibrary[0].name, "AdcRasterTime") == 0, "First definition name should be 'AdcRasterTime'");
    mu_assert(seq->definitionsLibrary[0].valueSize == 1, "First definition value size should be 1");
    mu_assert(strcmp(seq->definitionsLibrary[0].value[0], "1e-07") == 0, "First definition value should be '1e-07'");

    mu_assert(strcmp(seq->definitionsLibrary[1].name, "BlockDurationRaster") == 0, "Second definition name should be 'BlockDurationRaster'");
    mu_assert(seq->definitionsLibrary[1].valueSize == 1, "Second definition value size should be 1");
    mu_assert(strcmp(seq->definitionsLibrary[1].value[0], "1e-05") == 0, "Second definition value should be '1e-05'");

    mu_assert(strcmp(seq->definitionsLibrary[2].name, "GradientRasterTime") == 0, "Third definition name should be 'GradientRasterTime'");
    mu_assert(seq->definitionsLibrary[2].valueSize == 1, "Third definition value size should be 1");
    mu_assert(strcmp(seq->definitionsLibrary[2].value[0], "1e-05") == 0, "Third definition value should be '1e-05'");

    mu_assert(strcmp(seq->definitionsLibrary[3].name, "RadiofrequencyRasterTime") == 0, "Fourth definition name should be 'RadiofrequencyRasterTime'");
    mu_assert(seq->definitionsLibrary[3].valueSize == 1, "Fourth definition value size should be 1");
    mu_assert(strcmp(seq->definitionsLibrary[3].value[0], "1e-06") == 0, "Fourth definition value should be '1e-06'");

    mu_assert(strcmp(seq->definitionsLibrary[4].name, "TotalDuration") == 0, "Last definition name should be 'TotalDuration'");
    mu_assert(seq->definitionsLibrary[4].valueSize == 1, "Last definition value size should be 1");
    mu_assert(strcmp(seq->definitionsLibrary[4].value[0], "0.0051") == 0, "Last definition value should be '0.0051'");
    
    pulseqlib_seqFileFree(seq);
}

MU_TEST_SUITE(test_seqfile_suite) {
    printf("Running test_basic...\n");
    MU_RUN_TEST(test_basic);
    printf("Running test_definitions...\n");
    MU_RUN_TEST(test_definitions);
}

int test_seqfile_main(void) {
    printf("Starting SeqFile test suite...\n");
    MU_RUN_SUITE(test_seqfile_suite);
    printf("Test SeqFile suite completed.\n");
    MU_REPORT();
    return MU_EXIT_CODE;
}