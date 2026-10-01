/*
 * Print the RF of a cache's canonical repetition as a scanner build of the
 * library reads it: one "rf <definition> <designed angle in degrees>" line per
 * RF-bearing block, the definition from pulseg_get_rf_event_array and the
 * angle from the definition's nominal one scaled by the amplitude this block
 * plays it at.  Compiled by tests/test_ir_rf.py with the scanner's word size
 * and vendor.
 */

#include <stdio.h>
#include <stdlib.h>

#include "pulseg.h"
#include "pulseg_cache.h"

int main(int argc, char **argv)
{
    pulseg_collection *coll = NULL;
    pulseg_rf_stats *stats = NULL;
    pulseg_rf_event *events = NULL;
    int found, named, i;

    if (argc != 3)
    {
        fprintf(stderr, "usage: %s CACHE SOURCE_SIZE\n", argv[0]);
        return 2;
    }
    coll = pulseg_collection_alloc();
    if (!coll)
        return 1;
    if (PULSEG_FAILED(pulseg_load_cache(coll, argv[1], atoi(argv[2]))))
    {
        pulseg_collection_free(coll);
        return 3;
    }
    found = pulseg_get_rf_array(coll, &stats, 0);
    named = pulseg_get_rf_event_array(coll, &events, 0);
    if (found < 0 || named != found)
    {
        if (stats) free(stats);
        if (events) free(events);
        pulseg_collection_free(coll);
        return 4;
    }
    for (i = 0; i < found; ++i)
    {
        const double nominal = stats[i].flip_angle_rad * 180.0 / 3.14159265358979323846;
        const double designed = (stats[i].base_amplitude_hz > 0.0f)
            ? nominal * stats[i].act_amplitude_hz / stats[i].base_amplitude_hz
            : nominal;
        printf("rf %d %.4f\n", events[i].rf_def_id, designed);
    }
    if (stats) free(stats);
    if (events) free(events);
    pulseg_collection_free(coll);
    return 0;
}
