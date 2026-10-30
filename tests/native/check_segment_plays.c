/*
 * Count the segment instances and the scan time of collections written by
 * hand, so that every case is spelled out: a block table, the segment each
 * stream position belongs to, and what the counts must be. Exits non-zero
 * and names the case on the first count that differs.
 *
 * A segment is a run of blocks that play together; a pure delay is a segment
 * of one block that has no event. Blocks: 0 and 1 carry events (290 and 94
 * us), 2 is a delay of 45536 us, 3 is a delay of 4 us.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pulseg.h"
#include "pulseg_internal.h"

#define NUM_BLOCKS 4
#define NUM_SEGMENTS 4

static const int BLOCK_US[NUM_BLOCKS] = {290, 94, 45536, 4};

/* Segment 0 plays blocks 0 and 1; 1 and 2 are the two delays; 3 is block 0
 * alone. */
static const int SEGMENT_BLOCKS[NUM_SEGMENTS] = {2, 1, 1, 1};
static const int SEGMENT_FIRST_BLOCK[NUM_SEGMENTS] = {0, 2, 3, 0};

static int failures = 0;

static void expect(const char *what, double want, double got)
{
    if (want != got)
    {
        fprintf(stderr, "FAIL %s: expected %.0f got %.0f\n", what, want, got);
        failures += 1;
    }
}

/* One subsequence from a list of block ids and the segment of each. A
 * duration_override of -1 leaves the block its own. */
typedef struct
{
    int length;
    const int *blocks;
    const int *segments;
    const int *override_us;
} stream;

static void build(pulseg_collection *coll, pulseg_sequence_descriptor *desc, const stream *s)
{
    int n, k, runs = 0;
    pulseg_base_block *base = (pulseg_base_block *)calloc(NUM_BLOCKS, sizeof(*base));
    pulseg_virtual_segment *seg =
        (pulseg_virtual_segment *)calloc(NUM_SEGMENTS, sizeof(*seg));
    pulseg_block_table_element *table =
        (pulseg_block_table_element *)calloc((size_t)s->length, sizeof(*table));
    pulseg_exec_run *run = (pulseg_exec_run *)calloc(1, sizeof(*run));
    int *run_start = (int *)calloc((size_t)s->length + 1, sizeof(int));
    int *run_id = (int *)calloc((size_t)s->length + 1, sizeof(int));

    memset(desc, 0, sizeof(*desc));
    for (n = 0; n < NUM_BLOCKS; ++n)
    {
        base[n].id = n;
        base[n].duration_us = BLOCK_US[n];
        base[n].rf_id = (n < 2) ? 0 : -1;
        base[n].gx_id = -1;
        base[n].gy_id = -1;
        base[n].gz_id = -1;
        base[n].adc_id = -1;
    }
    for (n = 0; n < NUM_SEGMENTS; ++n)
    {
        seg[n].num_blocks = SEGMENT_BLOCKS[n];
        seg[n].unique_block_indices = (int *)calloc((size_t)SEGMENT_BLOCKS[n], sizeof(int));
        seg[n].has_digitalout = (int *)calloc((size_t)SEGMENT_BLOCKS[n], sizeof(int));
        for (k = 0; k < SEGMENT_BLOCKS[n]; ++k)
            seg[n].unique_block_indices[k] = SEGMENT_FIRST_BLOCK[n] + k;
    }
    for (n = 0; n < s->length; ++n)
    {
        table[n].id = s->blocks[n];
        table[n].duration_us = s->override_us ? s->override_us[n] : -1;
        if (n == 0 || s->segments[n] != s->segments[n - 1])
        {
            run_start[runs] = n;
            run_id[runs] = s->segments[n];
            runs += 1;
        }
    }
    run_start[runs] = s->length;
    run[0].emit_start = 0;
    run[0].block_start = 0;
    run[0].length = s->length;

    desc->base_blocks = base;
    desc->block_table = table;
    desc->num_unique_segments = NUM_SEGMENTS;
    desc->segment_definitions = seg;
    desc->exec_stream_len = s->length;
    desc->num_exec_runs = 1;
    desc->exec_runs = run;
    desc->num_seg_runs = runs;
    desc->seg_period = s->length;
    desc->seg_run_start = run_start;
    desc->seg_run_id = run_id;

    memset(coll, 0, sizeof(*coll));
    coll->descriptors = desc;
    coll->num_subsequences = 1;
}

static void check(const char *what, const stream *s, int short_us, double total_us,
                  int instances, int short_delays)
{
    pulseg_collection coll;
    pulseg_sequence_descriptor desc;
    pulseg_scan_time_info time = PULSEG_SCAN_TIME_INFO_INIT;
    pulseg_segment_plays plays = PULSEG_SEGMENT_PLAYS_INIT;
    char label[96];

    build(&coll, &desc, s);
    pulseg_get_scan_time(&coll, &time);
    pulseg_get_segment_plays(&coll, short_us, &plays);
    sprintf(label, "%s: scan time", what);
    expect(label, total_us, (double)time.total_duration_us);
    sprintf(label, "%s: segment instances", what);
    expect(label, instances, plays.segment_instances);
    sprintf(label, "%s: short delays", what);
    expect(label, short_delays, plays.short_delays);
}

int main(void)
{
    /* A repetition is an event segment and a delay, 1000 times: the shape of
     * a gradient-echo scan that waits out its repetition time. */
    {
        enum { N = 1000, L = 3 * N };
        static int blocks[L], segments[L];
        stream s;
        int i;
        for (i = 0; i < N; ++i)
        {
            blocks[3 * i] = 0;     segments[3 * i] = 0;
            blocks[3 * i + 1] = 1; segments[3 * i + 1] = 0;
            blocks[3 * i + 2] = 2; segments[3 * i + 2] = 1;
        }
        s.length = L; s.blocks = blocks; s.segments = segments; s.override_us = NULL;
        check("event segment and long delay", &s, 4, N * (290.0 + 94 + 45536), 2 * N, 0);
    }

    /* The same with a 4 us delay in place of the long one: a delay the
     * segment before it can absorb. */
    {
        enum { N = 1000, L = 3 * N };
        static int blocks[L], segments[L];
        stream s;
        int i;
        for (i = 0; i < N; ++i)
        {
            blocks[3 * i] = 0;     segments[3 * i] = 0;
            blocks[3 * i + 1] = 1; segments[3 * i + 1] = 0;
            blocks[3 * i + 2] = 3; segments[3 * i + 2] = 2;
        }
        s.length = L; s.blocks = blocks; s.segments = segments; s.override_us = NULL;
        check("event segment and 4 us delay", &s, 4, N * (290.0 + 94 + 4), 2 * N, N);
        check("a limit below the delay", &s, 3, N * (290.0 + 94 + 4), 2 * N, 0);
    }

    /* A segment played twice in a row is two instances, and two repetitions
     * of one segment are one change of segment. */
    {
        enum { L = 4 };
        static int blocks[L] = {0, 1, 0, 1}, segments[L] = {0, 0, 0, 0};
        stream s;
        s.length = L; s.blocks = blocks; s.segments = segments; s.override_us = NULL;
        check("one segment twice in a row", &s, 4, 2 * (290.0 + 94), 2, 0);
    }

    /* A delay's own duration is the one in the block table where it has one:
     * 3 us folds into the segment before it, 100 us does not. */
    {
        enum { L = 6 };
        static int blocks[L] = {0, 1, 3, 0, 1, 3};
        static int segments[L] = {0, 0, 2, 0, 0, 2};
        static int override_us[L] = {-1, -1, 3, -1, -1, 100};
        stream s;
        s.length = L; s.blocks = blocks; s.segments = segments; s.override_us = override_us;
        check("a delay stretched in its instance", &s, 4, 2 * (290.0 + 94) + 3 + 100, 4, 1);
    }

    /* A delay after a delay is played whole. */
    {
        enum { L = 3 };
        static int blocks[L] = {2, 3, 0}, segments[L] = {1, 2, 3};
        stream s;
        s.length = L; s.blocks = blocks; s.segments = segments; s.override_us = NULL;
        check("a delay after a delay", &s, 4, 45536.0 + 4 + 290, 3, 0);
    }

    /* The sum of a long scan is the sum of its blocks to the microsecond: 20000
     * repetitions of five blocks come to 1.0e9 us, where a single-precision
     * running sum is 143 ms off. */
    {
        enum { N = 20000, L = 3 * N };
        static int blocks[L], segments[L];
        stream s;
        int i;
        for (i = 0; i < N; ++i)
        {
            blocks[3 * i] = 0;     segments[3 * i] = 0;
            blocks[3 * i + 1] = 1; segments[3 * i + 1] = 0;
            blocks[3 * i + 2] = 2; segments[3 * i + 2] = 1;
        }
        s.length = L; s.blocks = blocks; s.segments = segments; s.override_us = NULL;
        check("a scan of 1e9 us", &s, 4, (double)(float)(N * (290.0 + 94 + 45536)), 2 * N, 0);
    }

    if (failures)
        return 1;
    printf("ALL PASS check_segment_plays\n");
    return 0;
}
