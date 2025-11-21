#include "minunit.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>

#include "pulseqlib.h"
#include "pulseqlib_methods.h"

/* Helper function to get a block (needed for shape loading test) */
static pulseqlib_SeqBlock* getBlock(pulseqlib_SeqFile* seq, int blockIndex, int parseExtensions) {
    pulseqlib_SeqBlock* block;

    /* Allocate memory for the sequence file */
    block = (pulseqlib_SeqBlock*)ALLOC(sizeof(pulseqlib_SeqBlock));
    if (!block) return NULL;

    /* Initialize the block with default values */
    pulseqlib_seqBlockInit(block);
    
    /* Get the block using the new API */
    pulseqlib_getBlock(seq, blockIndex, parseExtensions, block);
    return block;
}

/* Helper to clean up .shapes file */
static void clean_shapes(const char* filePath) {
    char shapes_path[1024];
    char* ext;
    snprintf(shapes_path, sizeof(shapes_path), "%s/%s", TEST_ROOT_DIR, filePath);
    
    /* Replace .seq with .shapes */
    ext = strrchr(shapes_path, '.');
    if (ext && strcmp(ext, ".seq") == 0) {
        strcpy(ext, ".shapes");
        remove(shapes_path);
    }
}

static pulseqlib_SeqFile* load_seq(char* filePath) {
    char seq_path[1024];
    pulseqlib_SystemParams* system;
    pulseqlib_SeqFile* seq;
    
    /* Allocate memory for the system parameters */
    system = (pulseqlib_SystemParams*)ALLOC(sizeof(pulseqlib_SystemParams));
    if (!system) return NULL;

    /* Initialize the system parameters structure with default values */
    pulseqlib_systemParamsInit(system, 
        /*B0=*/3.0f, /*max_grad=*/40.0f, /*max_slew=*/150.0f, 
        /*rf_raster=*/1.0f, /*grad_raster=*/10.0f, /*adc_raster=*/0.1f, /*block_raster=*/10.0f
    );

    /* Allocate memory for the sequence file */
    seq = (pulseqlib_SeqFile*)ALLOC(sizeof(pulseqlib_SeqFile));
    if (!seq) return NULL;
    
    /* Create the full path to the sequence file */
    snprintf(seq_path, sizeof(seq_path), "%s/%s", TEST_ROOT_DIR, filePath);
    
    /* Initialize the sequence file structure */
    pulseqlib_seqFileInit(seq_path, seq, system);
    if (!seq) {
        FREE(seq);
        return NULL;
    }
    
    /* Read the sequence data */
    pulseqlib_readSeq(seq, 1);
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

MU_TEST(test_shapes_io) {
    pulseqlib_SeqBlock* block;
    pulseqlib_SeqFile* seq;
    char shapes_path[1024];
    struct stat attr;
    time_t mtime_before;
    FILE* f_handle;
    pulseqlib_SeqBlock* block2;
    
    snprintf(shapes_path, sizeof(shapes_path), "%s/expected_output/seq2.shapes", TEST_ROOT_DIR);
    
    /* Ensure clean start */
    clean_shapes("expected_output/seq2.seq");
    
    /* 1. Load sequence (creates .shapes) */
    seq = load_seq("expected_output/seq2.seq");
    mu_assert(seq != NULL, "Sequence should load");
    
    /* Check initial state - should be closed until needed */
    mu_assert(seq->shapesLibrary.open == 0, "Shapes library should be closed initially");
    
    /* 2. Trigger shape loading */
    block = getBlock(seq, 0, 1); /* Block 0 has shapes */
    mu_assert(seq->shapesLibrary.open == 1, "Shapes library should be open after accessing block with shapes");
    mu_assert(seq->shapesLibrary.file != NULL, "File pointer should be valid");
    
    /* 3. Verify persistence of file handle */
    f_handle = seq->shapesLibrary.file;
    block2 = getBlock(seq, 1, 1);
    mu_assert(seq->shapesLibrary.file == f_handle, "File handle should be reused");
    
    pulseqlib_seqBlockFree(block);
    FREE(block);
    pulseqlib_seqBlockFree(block2);
    FREE(block2);
    
    /* 4. Close and check */
    /* We manually allocate/free here to check the struct state after pulseqlib_seqFileFree */
    pulseqlib_seqFileFree(seq);
    mu_assert(seq->shapesLibrary.open == 0, "Shapes library should be marked closed after free");
    FREE(seq);

    /* 5. Test Persistence (not recreated) */
    /* Get timestamp of existing .shapes */
    if (stat(shapes_path, &attr) == 0) {
        mtime_before = attr.st_mtime;
    } else {
        mu_fail("Could not stat .shapes file");
    }
    
    sleep(1); /* Ensure filesystem time resolution */
    
    /* Reload */
    seq = load_seq("expected_output/seq2.seq");
    
    /* Check timestamp again */
    if (stat(shapes_path, &attr) == 0) {
        mu_assert(attr.st_mtime == mtime_before, ".shapes file should not be recreated if it exists");
    }
    
    pulseqlib_seqFileFree(seq);
    FREE(seq);
    
    clean_shapes("expected_output/seq2.seq");
}

MU_TEST(test_interpolation_flag) {
    pulseqlib_SystemParams sys;
    char seq_path[1024];
    pulseqlib_SeqFile* seq = load_seq("expected_output/seq1.seq");
    
    snprintf(seq_path, sizeof(seq_path), "%s/expected_output/seq1.seq", TEST_ROOT_DIR);
    
    /* Case 1: Matching rasters -> interpolate = 0 */
    mu_assert(seq->interpolate == 0, "Interpolate flag should be 0 for matching rasters");
    pulseqlib_seqFileFree(seq);
    FREE(seq);
    
    /* Case 2: Mismatching rasters -> interpolate = 1 */
    /* Change RF raster to 2us */
    pulseqlib_systemParamsInit(&sys, 
        /*B0=*/3.0f, /*max_grad=*/40.0f, /*max_slew=*/150.0f, 
        /*rf_raster=*/2.0f, /*grad_raster=*/10.0f, /*adc_raster=*/0.1f, /*block_raster=*/10.0f
    );
    
    seq = (pulseqlib_SeqFile*)ALLOC(sizeof(pulseqlib_SeqFile));
    pulseqlib_seqFileInit(seq_path, seq, &sys);
    pulseqlib_readSeq(seq, 1);
    
    mu_assert(seq->interpolate == 1, "Interpolate flag should be 1 for mismatching rasters");
    pulseqlib_seqFileFree(seq);
    FREE(seq);
}

MU_TEST_SUITE(test_seqfile_suite) {
    printf("Running test_basic...\n");
    MU_RUN_TEST(test_basic);
    printf("Running test_definitions...\n");
    MU_RUN_TEST(test_definitions);
    printf("Running test_shapes_io...\n");
    MU_RUN_TEST(test_shapes_io);
    printf("Running test_interpolation_flag...\n");
    MU_RUN_TEST(test_interpolation_flag);
}

int test_seqfile_main(void) {
    printf("Starting SeqFile test suite...\n");
    MU_RUN_SUITE(test_seqfile_suite);
    printf("Test SeqFile suite completed.\n");
    MU_REPORT();
    return MU_EXIT_CODE;
}