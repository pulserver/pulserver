/*
 * Drive a scan loop whose backend asks for one segment instance again, and
 * print what it was given: one "block <segment> <instance> <position>" line
 * per block set, in order.  A replayed instance therefore prints its blocks
 * twice at the same instance number, because asking for it again means it
 * has not been played.  Compiled by tests/test_ir.py with the scanner's word
 * size and vendor.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pulseg.h"
#include "pulseg_cache.h"

typedef struct
{
    int replay_at;   /* the instance to ask for again, -1 for none */
    int replayed;    /* whether it has been asked for already */
} driver;

static int on_reserve(void *ctx, const pulseg_wave_plan *plan)
{
    (void)ctx;
    (void)plan;
    return PULSEG_SUCCESS;
}

static int on_load(void *ctx, const pulseg_wave_load *load)
{
    (void)ctx;
    (void)load;
    return PULSEG_SUCCESS;
}

static int on_block(
    void *ctx, const pulseg_playout_segment *segment, const pulseg_playout_block *block)
{
    (void)ctx;
    printf("block %d %d %d\n", segment->segment, segment->instance, block->position);
    return PULSEG_SUCCESS;
}

static int on_instance(void *ctx, const pulseg_playout_segment *segment)
{
    driver *d = (driver *)ctx;

    printf("played %d %d\n", segment->segment, segment->instance);
    if (!d->replayed && segment->instance == d->replay_at)
    {
        d->replayed = 1;
        printf("replay %d %d\n", segment->segment, segment->instance);
        return PULSEG_PLAYOUT_REPLAY;
    }
    return PULSEG_SUCCESS;
}

int main(int argc, char **argv)
{
    pulseg_collection *coll = NULL;
    pulseg_playout_backend backend;
    pulseg_playout_options options = PULSEG_PLAYOUT_OPTIONS_INIT;
    pulseg_wave_plan plan = PULSEG_WAVE_PLAN_INIT;
    pulseg_wave_budget budget = PULSEG_WAVE_BUDGET_INIT;
    pulseg_diagnostic diag = PULSEG_DIAGNOSTIC_INIT;
    driver d;
    int rc;

    if (argc < 3)
    {
        fprintf(stderr, "usage: %s <seq_path> <replay_at_instance>\n", argv[0]);
        return 2;
    }
    d.replay_at = atoi(argv[2]);
    d.replayed = 0;

    rc = pulseg_load_scanloop_cache(&coll, argv[1]);
    if (PULSEG_FAILED(rc))
    {
        fprintf(stderr, "load failed rc=%d\n", rc);
        return 1;
    }

    budget.max_samples = 1 << 20;
    budget.raster_us = 4.0f;
    budget.slots = 2;
    budget.headroom = 0.5f;
    rc = pulseg_get_wave_plan(coll, &budget, &plan, &diag);
    if (PULSEG_FAILED(rc))
    {
        fprintf(stderr, "wave plan failed rc=%d: %s\n", rc, diag.message);
        return 1;
    }

    memset(&backend, 0, sizeof(backend));
    backend.ctx = &d;
    backend.reserve_waves = on_reserve;
    backend.load_wave = on_load;
    backend.set_block = on_block;
    backend.play_instance = on_instance;
    options.prescan_subsequence = -1;

    rc = pulseg_playout_scan(coll, &plan, &backend, &options, &diag);
    if (PULSEG_FAILED(rc))
    {
        fprintf(stderr, "playout failed rc=%d: %s\n", rc, diag.message);
        return 1;
    }
    pulseg_free_wave_plan(&plan);
    pulseg_collection_free(coll);
    return 0;
}
