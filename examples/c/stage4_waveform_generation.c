/*
 * stage4_waveform_generation.c -- materialise the distinct waveforms and fit
 * them in waveform memory.
 *
 * A scan of a million blocks does not contain a million waveforms. It
 * contains a handful, replayed at different amplitudes and rotations, and the
 * whole point of the structure derived at preparation is that this stage can
 * upload each distinct one once.
 *
 * The cache is read by section, so a stage pays only for what it needs. Which
 * section that is depends on the question. Rendering one waveform from its
 * definition needs the shared definitions and the shapes
 * (pulseg_load_geninstructions_cache). Deciding whether they all fit needs to
 * know how often each is played and in what order, which lives in the
 * execution stream, so chunk planning reads the scan-stage cache.
 *
 *   stage4_waveform_generation <scan.seq>
 */

#include <string.h>

#include "vendor.h"

#define MAX_POINTS 4096

int main(int argc, char **argv)
{
    pulseg_diagnostic diagnostic;
    pulseg_collection *scan = NULL;
    pulseg_chunk_budget budget = PULSEG_CHUNK_BUDGET_INIT;
    pulseg_chunk_plan plan;
    float times_us[MAX_POINTS];
    float amplitudes[MAX_POINTS];
    float peak;
    int code, w, axis, num_points;

    if (argc != 2)
    {
        (void)fprintf(stderr, "usage: %s <scan.seq>\n", argv[0]);
        return 2;
    }

    pulseg_diagnostic_init(&diagnostic);

    /* 1. Load what the planner reads
     * The execution stream is what says how often each waveform is played
     * and in what order, and the planner needs both. A stage that only
     * rendered definitions, with no planning, would take the cheaper
     * pulseg_load_geninstructions_cache instead. */
    code = pulseg_load_scanloop_cache(&scan, argv[1]);
    if (PULSEG_FAILED(code))
    {
        vendor_refuse(code, NULL);
        return 1;
    }
    vendor_log("cache: definitions, shapes and execution stream loaded");

    /* 2. Ask whether it all fits
     * The planner answers with one of two modes. RESIDENT means every
     * distinct waveform fits at once: upload them here and upload nothing
     * during the scan. STREAMED means it does not, and each segment has to be
     * materialised while its predecessor plays, which the budget below is
     * what decides. Most scans are RESIDENT. */
    budget.max_wave_samples = vendor_waveform_memory_samples() / 2;
    budget.build_us_per_sample = 0.05f; /* measured on the target */

    memset(&plan, 0, sizeof plan);
    code = pulseg_plan_chunks(scan, 0, &budget, &plan, &diagnostic);
    if (PULSEG_FAILED(code))
    {
        vendor_refuse(code, &diagnostic);
        pulseg_collection_free(scan);
        return 1;
    }
    vendor_log(
        "plan: %s, %d distinct waveform(s) in %d chunk(s)",
        plan.mode == PULSEG_WAVE_RESIDENT ? "resident" : "streamed",
        plan.num_waves,
        plan.num_chunks);

    /* 3. Materialise each one, once
     * A wave key names a shape per axis and the amplitude ratios between
     * them. Materialising it renders the corner points; the hardware then
     * replays that one waveform at whatever amplitude each instance asks
     * for, which is why the count here is the count of distinct shapes and
     * not of blocks. */
    for (w = 0; w < plan.num_waves; ++w)
    {
        for (axis = 0; axis < plan.waves[w].num_axes; ++axis)
        {
            code = pulseg_materialize_wave(
                scan,
                0,
                &plan.waves[w],
                axis,
                times_us,
                amplitudes,
                MAX_POINTS,
                &num_points,
                &peak);
            if (PULSEG_FAILED(code))
            {
                vendor_refuse(code, NULL);
                pulseg_free_chunk_plan(&plan);
                pulseg_collection_free(scan);
                return 1;
            }
            (void)vendor_load_waveform(axis, amplitudes, num_points);
        }
    }

    pulseg_free_chunk_plan(&plan);
    pulseg_collection_free(scan);
    return 0;
}
