/*
 * stage4_waveform_generation.cpp -- materialise the distinct waveforms and
 * fit them in waveform memory.
 *
 * C++ counterpart of examples/c/stage4_waveform_generation.c.
 *
 *   stage4_waveform_generation <scan.seq>
 */

#include "vendor.hpp"

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::fprintf(stderr, "usage: %s <scan.seq>\n", argv[0]);
        return 2;
    }

    try
    {
        /* 1. Load what the planner reads. The execution stream says how
         * often each waveform is played and in what order, and the planner
         * needs both; a stage that only rendered definitions would take
         * from_geninstructions_cache instead. */
        pulseg::Collection scan =
            pulseg::Collection::from_scanloop_cache(argv[1], vendor::system_limits());
        vendor::log("cache: definitions, shapes and execution stream loaded");

        /* 2. Ask whether it all fits. RESIDENT means every distinct waveform
         * fits at once and nothing is uploaded mid-scan; STREAMED means each
         * segment is materialised while its predecessor plays. */
        pulseg_chunk_budget budget = PULSEG_CHUNK_BUDGET_INIT;
        budget.max_wave_samples = vendor::waveform_memory_samples() / 2;
        budget.build_us_per_sample = 0.05f; /* measured on the target */

        pulseg::ChunkPlan plan(scan.handle(), 0, budget);
        std::printf(
            "plan: %s, %d distinct waveform(s) in %d chunk(s)\n",
            plan.mode() == PULSEG_WAVE_RESIDENT ? "resident" : "streamed",
            plan.num_waves(),
            plan.num_chunks());

        /* 3. Materialise each one, once. The hardware replays it at whatever
         * amplitude each instance asks for, so the count is the number of
         * distinct shapes and not of blocks. */
        for (int w = 0; w < plan.num_waves(); ++w)
            for (int axis = 0; axis < plan.wave(w).num_axes; ++axis)
                vendor::load_waveform(axis, plan.materialise(w, axis).amplitude);
    }
    catch (const pulseg::Error& error)
    {
        std::fprintf(stderr, "refused: %s\n", error.what());
        return 1;
    }
    return 0;
}
