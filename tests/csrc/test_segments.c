#include "minunit.h"

#include <stdio.h>
#include <stdlib.h>

#include "pulseqlib.h"
#include "pulseqlib_methods.h"
#include "pulserverlib.h"
#include "pulserverlib_methods.h"

static pulseqlib_SeqFile* load_seq(const char* relativePath) {
    char seq_path[1024];
    pulseqlib_SeqFile* seq;
    pulseqlib_Opts opts;

    seq = (pulseqlib_SeqFile*)malloc(sizeof(pulseqlib_SeqFile));
    if (!seq) {
        return NULL;
    }

    snprintf(seq_path, sizeof(seq_path), "%s/%s", TEST_ROOT_DIR, relativePath);

    pulseqlib_optsInit(&opts, 3.0f, 40.0f, 150.0f, 1.0f, 10.0f, 0.1f, 10.0f);
    pulseqlib_seqFileInit(seq, &opts);
    pulseqlib_optsFree(&opts);

    pulseqlib_readSeq(seq, seq_path);
    return seq;
}

MU_TEST(test_segment_layout_basic) {
    pulserverlib_SegmentLayout layout;
    pulserverlib_Status status;
    pulseqlib_SeqFile* seq;
    int expectedIndices[] = {0, 1, 2, 1, 2};
    int i;

    seq = load_seq("expected_output/segment.seq");
    mu_assert(seq != NULL, "Failed to load sequence file");

    status = pulserverlib_segmentLayoutInit(&layout, seq);
    mu_assert(status == PULSERVERLIB_STATUS_OK, "Segment layout initialization should succeed");

    mu_assert(layout.numSegments == 3, "There should be three unique segments");
    mu_assert(layout.segments[0].ID == 1, "Segment 0 ID should be 1");
    mu_assert(layout.segments[0].offsetBlock == 0, "Segment 0 should start at block 0");
    mu_assert(layout.segments[0].numBlocks == 2, "Segment 0 should span two blocks");

    mu_assert(layout.segments[1].ID == 4, "Segment 1 ID should be 4");
    mu_assert(layout.segments[1].offsetBlock == 2, "Segment 1 should start at block 2");
    mu_assert(layout.segments[1].numBlocks == 1, "Segment 1 should span one block");

    mu_assert(layout.segments[2].ID == 2, "Segment 2 ID should be 2");
    mu_assert(layout.segments[2].offsetBlock == 3, "Segment 2 should start at block 3");
    mu_assert(layout.segments[2].numBlocks == 1, "Segment 2 should span one block");

    mu_assert(layout.tr.ID == 1, "TR ID should be 1");
    mu_assert(layout.tr.numSegments == 5, "TR definition should list five segments");

    for (i = 0; i < layout.tr.numSegments; ++i) {
        mu_assert(layout.tr.segmentIndices[i] == expectedIndices[i], "Unexpected segment index in TR definition");
    }

    pulserverlib_segmentLayoutFree(&layout);
    pulseqlib_seqFileFree(seq);
}

MU_TEST(test_segment_layout_missing_trid) {
    pulserverlib_SegmentLayout layout;
    pulserverlib_Status status;
    pulseqlib_SeqFile* seq;

    seq = load_seq("expected_output/seq1.seq");
    mu_assert(seq != NULL, "Failed to load sequence file");

    status = pulserverlib_segmentLayoutInit(&layout, seq);
    mu_assert(status == PULSERVERLIB_STATUS_MISSING_TRID, "Missing TRID should be reported");

    pulserverlib_segmentLayoutFree(&layout);
    pulseqlib_seqFileFree(seq);
}

MU_TEST_SUITE(test_segments_suite) {
    MU_RUN_TEST(test_segment_layout_basic);
    MU_RUN_TEST(test_segment_layout_missing_trid);
}

int test_segments_main(void) {
    MU_RUN_SUITE(test_segments_suite);
    MU_REPORT();
    return MU_EXIT_CODE;
}
