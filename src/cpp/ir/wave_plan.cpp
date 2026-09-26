/**
 * @file wave_plan.cpp
 * @brief Where a playout holds the waves of a collection in its waveform
 *        memory, and whether it loads them in time.
 */

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstddef>
#include <vector>

extern "C"
{
#include "pulseg_internal.h"
#include "pulseg.h"
}

#include "pulseg_convert.h"

namespace
{

using Axes = std::array<long, 3>;

/* Region tables per subsequence or per segment. */
using Tables = std::vector<std::vector<pulseg_wave_region>>;

/* Give @p region an offset on each axis of @p axes, after what @p used holds
 * there, and -1 on the others. */
void place(pulseg_wave_region &region, int axes, Axes &used)
{
    for (int a = 0; a < 3; ++a)
    {
        region.offset[a] = -1;
        if (axes & (1 << a))
        {
            region.offset[a] = used[static_cast<size_t>(a)];
            used[static_cast<size_t>(a)] += region.samples;
        }
    }
}

int peak_axes(const pulseg_wave &wave)
{
    int axes = 0;
    for (int a = 0; a < 3; ++a)
        if (wave.peak[a] > 0.0f)
            axes |= 1 << a;
    return axes;
}

int driven_axes(const pulseg_wave_region &region)
{
    return (region.offset[0] >= 0) + (region.offset[1] >= 0) + (region.offset[2] >= 0);
}

/* RESIDENT: a region per wave of each subsequence, laid out in order. */
Tables lay_out_waves(const pulseg_collection *coll, float raster_us, Axes &used)
{
    Tables tables(static_cast<size_t>(std::max(coll->num_subsequences, 0)));
    for (int s = 0; s < coll->num_subsequences; ++s)
    {
        const pulseg_sequence_descriptor &desc = coll->descriptors[s];
        for (int w = 0; w < desc.num_waves && desc.waves; ++w)
        {
            pulseg_wave_region region{};
            pulseg__wave_cover(desc.waves[w].start_us, desc.waves[w].end_us, raster_us, &region);
            place(region, peak_axes(desc.waves[w]), used);
            tables[static_cast<size_t>(s)].push_back(region);
        }
    }
    return tables;
}

/* STREAMED: a ring of @p slots slots per position of each segment, empty
 * where the position plays no wave. */
int lay_out_slots(
    const pulseg_collection *coll,
    float raster_us,
    int slots,
    Tables &tables,
    Axes &used)
{
    tables.assign(static_cast<size_t>(std::max(coll->total_unique_segments, 0)), {});
    for (int g = 0; g < coll->total_unique_segments; ++g)
    {
        pulseg_segment_info seg = PULSEG_SEGMENT_INFO_INIT;
        int rc = pulseg_get_segment_info(coll, &seg, g);
        for (int b = 0; b < seg.num_blocks && PULSEG_SUCCEEDED(rc); ++b)
        {
            pulseg_block_info block = PULSEG_BLOCK_INFO_INIT;
            rc = pulseg_get_block_info(coll, &block, g, b);
            for (int k = 0; k < slots && PULSEG_SUCCEEDED(rc); ++k)
            {
                pulseg_wave_region region{};
                if (block.wave_points > 0)
                    pulseg__wave_cover(block.wave_start_us, block.wave_end_us, raster_us, &region);
                place(region, block.wave_points > 0 ? block.wave_axes : 0, used);
                tables[static_cast<size_t>(g)].push_back(region);
            }
        }
        if (PULSEG_FAILED(rc))
            return rc;
    }
    return PULSEG_SUCCESS;
}

/* PULSEG_WAVES_* the memory affords, or -1 when neither layout fits. */
int affordable_mode(const long resident[3], const long streamed[3], long max_samples)
{
    bool any = false, all_resident = true, all_streamed = true;
    for (int a = 0; a < 3; ++a)
    {
        any = any || resident[a] > 0;
        all_resident = all_resident && resident[a] <= max_samples;
        all_streamed = all_streamed && streamed[a] <= max_samples;
    }
    if (!any)
        return PULSEG_WAVES_NONE;
    if (all_resident)
        return PULSEG_WAVES_RESIDENT;
    return all_streamed ? PULSEG_WAVES_STREAMED : -1;
}

/* A segment instance of the scan, in play order across the chain. */
struct Instance
{
    int subsequence;
    int first_position; /* its first block's execution-stream position */
    int segment;        /* global */
    double duration_us;
};

/* Append the segment instances subsequence @p s plays to @p instances. */
void subsequence_instances(const pulseg_collection *coll, int s, std::vector<Instance> &instances)
{
    const pulseg_sequence_descriptor *desc = &coll->descriptors[s];
    int last = -1;
    int position = 0;
    for (int n = 0; n < desc->exec_stream_len; ++n)
    {
        const int local = pulseg__exec_seg_id(desc, n);
        if (local < 0 || local >= desc->num_unique_segments)
        {
            last = -1;
            continue;
        }
        const int blocks = desc->segment_definitions[local].num_blocks;
        position = (local == last && blocks > 0) ? (position + 1) % blocks : 0;
        last = local;
        if (position == 0)
            instances.push_back({s, n, pulseg__global_segment(coll, s, local), 0.0});
        const int block = pulseg__exec_block_idx(desc, n);
        if (block >= 0 && block < desc->num_blocks)
            instances.back().duration_us += pulseg__played_duration_us(desc, block);
    }
}

std::vector<Instance> scan_instances(const pulseg_collection *coll)
{
    std::vector<Instance> instances;
    for (int s = 0; s < coll->num_subsequences; ++s)
        subsequence_instances(coll, s, instances);
    return instances;
}

/* How long loading the waves of an instance of each segment takes: every slot
 * of a ring covers the same span, so the first speaks for the ring. */
std::vector<double> load_times(const Tables &slots, int per, float load_us_per_sample)
{
    std::vector<double> times;
    times.reserve(slots.size());
    for (const auto &positions : slots)
    {
        double total = 0.0;
        for (size_t b = 0; b < positions.size(); b += static_cast<size_t>(per))
            total += static_cast<double>(positions[b].samples) * driven_axes(positions[b]) *
                load_us_per_sample;
        times.push_back(total);
    }
    return times;
}

/* The loading of a streamed layout, on the playout's timeline scaled by the
 * headroom.  The scan starts with its first instance loaded.  The waves of
 * each later instance are loaded one instance after another, from once the
 * instance budget.slots - 1 before it has started; an instance's spare time
 * is how long before its start its loading ends.  The walk stops at the
 * first instance whose loading ends after it starts, where the playout would
 * wait. */
void check_loading(
    const pulseg_collection *coll,
    const Tables &slots,
    const pulseg_wave_budget &budget,
    pulseg_wave_plan &plan)
{
    const std::vector<Instance> instances = scan_instances(coll);
    const std::vector<double> load_us = load_times(slots, budget.slots, budget.load_us_per_sample);
    const size_t ahead = static_cast<size_t>(budget.slots - 1);
    std::vector<double> start(instances.size());
    double elapsed = 0.0;
    for (size_t i = 0; i < instances.size(); ++i)
    {
        start[i] = budget.headroom * elapsed;
        elapsed += instances[i].duration_us;
    }

    double least = FLT_MAX;
    double loaded = 0.0;
    plan.tightest_subseq = -1;
    plan.tightest_position = -1;
    for (size_t i = 1; i < instances.size(); ++i)
    {
        const size_t segment = static_cast<size_t>(instances[i].segment);
        const double earliest = i >= ahead ? start[i - ahead] : 0.0;
        loaded = std::max(loaded, earliest) + (segment < load_us.size() ? load_us[segment] : 0.0);
        const double spare = start[i] - loaded;
        if (spare < least)
        {
            least = spare;
            plan.tightest_subseq = instances[i].subsequence;
            plan.tightest_position = instances[i].first_position;
        }
        if (spare < 0.0)
            break;
    }
    plan.least_spare_us = plan.tightest_subseq < 0 ? 0.0f : static_cast<float>(least);
    plan.loading_checked = 1;
}

/* Hand @p tables of @p per regions per entry to C arrays the plan owns. */
int store_tables(const Tables &tables, int per, int **lengths, pulseg_wave_region ***regions)
{
    const size_t count = tables.size();
    *lengths = nullptr;
    *regions = nullptr;
    if (count == 0)
        return PULSEG_SUCCESS;
    *lengths = static_cast<int *>(PULSEG_ALLOC(count * sizeof(int)));
    *regions = static_cast<pulseg_wave_region **>(PULSEG_ALLOC(count * sizeof(pulseg_wave_region *)));
    if (!*lengths || !*regions)
        return PULSEG_ERR_ALLOC_FAILED;
    std::fill(*regions, *regions + count, nullptr);
    std::fill(*lengths, *lengths + count, 0);
    for (size_t i = 0; i < count; ++i)
    {
        const std::vector<pulseg_wave_region> &table = tables[i];
        if (table.empty())
            continue;
        (*regions)[i] = static_cast<pulseg_wave_region *>(
            PULSEG_ALLOC(table.size() * sizeof(pulseg_wave_region)));
        if (!(*regions)[i])
            return PULSEG_ERR_ALLOC_FAILED;
        std::copy(table.begin(), table.end(), (*regions)[i]);
        (*lengths)[i] = static_cast<int>(table.size()) / per;
    }
    return PULSEG_SUCCESS;
}

int wave_memory_shortfall(
    const pulseg_wave_plan &plan,
    const pulseg_wave_budget &budget,
    pulseg_diagnostic *diag)
{
    if (diag)
    {
        diag->code = PULSEG_ERR_WAVE_MEMORY;
        pulseg__diag_printf(
            diag,
            "waves take %ld, %ld, %ld samples at once and %ld, %ld, %ld in %d slots "
            "per position, against %ld per axis",
            plan.resident_samples[0], plan.resident_samples[1], plan.resident_samples[2],
            plan.streamed_samples[0], plan.streamed_samples[1], plan.streamed_samples[2],
            budget.slots, budget.max_samples);
    }
    return PULSEG_ERR_WAVE_MEMORY;
}

int wave_loading_shortfall(const pulseg_wave_plan &plan, pulseg_diagnostic *diag)
{
    if (diag)
    {
        diag->code = PULSEG_ERR_WAVE_LOADING;
        pulseg__diag_printf(
            diag,
            "subsequence %d, execution-stream position %d: loading its waves ends %.1f us "
            "after it starts",
            plan.tightest_subseq, plan.tightest_position, static_cast<double>(-plan.least_spare_us));
    }
    return PULSEG_ERR_WAVE_LOADING;
}

int plan_arguments(
    const pulseg_collection *coll,
    const pulseg_wave_budget *budget,
    const pulseg_wave_plan *plan)
{
    if (!coll || !budget || !plan)
        return PULSEG_ERR_NULL_POINTER;
    const bool valid = budget->raster_us > 0.0f && budget->max_samples >= 0 &&
        budget->load_us_per_sample >= 0.0f && budget->headroom > 0.0f && budget->slots >= 2;
    return valid ? PULSEG_SUCCESS : PULSEG_ERR_INVALID_ARGUMENT;
}

/* Both layouts of the waves into @p plan's tables and sizes, and the slots of
 * the streamed one into @p slots. */
int lay_out(
    const pulseg_collection *coll,
    const pulseg_wave_budget &budget,
    pulseg_wave_plan &plan,
    Tables &slots)
{
    Axes resident{0, 0, 0};
    Axes streamed{0, 0, 0};
    const Tables waves = lay_out_waves(coll, budget.raster_us, resident);
    int rc = lay_out_slots(coll, budget.raster_us, budget.slots, slots, streamed);
    if (PULSEG_SUCCEEDED(rc))
        rc = store_tables(waves, 1, &plan.num_waves, &plan.waves);
    if (PULSEG_SUCCEEDED(rc))
        rc = store_tables(slots, budget.slots, &plan.num_positions, &plan.slots);
    plan.num_subsequences = static_cast<int>(waves.size());
    plan.num_segments = static_cast<int>(slots.size());
    std::copy(resident.begin(), resident.end(), plan.resident_samples);
    std::copy(streamed.begin(), streamed.end(), plan.streamed_samples);
    return rc;
}

void adopt(pulseg_wave_plan &plan, int mode)
{
    const long *held = mode == PULSEG_WAVES_STREAMED ? plan.streamed_samples : plan.resident_samples;
    plan.mode = mode;
    std::copy(held, held + 3, plan.samples);
}

} // namespace

int pulseg_plan_waves(
    const pulseg_collection *coll,
    const pulseg_wave_budget *budget,
    pulseg_wave_plan *plan,
    pulseg_diagnostic *diag)
{
    static const pulseg_wave_plan empty = PULSEG_WAVE_PLAN_INIT;

    int rc = plan_arguments(coll, budget, plan);
    if (PULSEG_FAILED(rc))
        return rc;
    *plan = empty;
    plan->budget = *budget;
    Tables slots;
    rc = lay_out(coll, *budget, *plan, slots);
    if (PULSEG_FAILED(rc))
        return rc;

    const int mode = affordable_mode(plan->resident_samples, plan->streamed_samples, budget->max_samples);
    if (mode < 0)
        return wave_memory_shortfall(*plan, *budget, diag);
    adopt(*plan, mode);
    if (mode == PULSEG_WAVES_STREAMED && budget->load_us_per_sample > 0.0f)
        check_loading(coll, slots, *budget, *plan);
    if (plan->loading_checked && plan->least_spare_us < 0.0f)
        return wave_loading_shortfall(*plan, diag);
    return PULSEG_SUCCESS;
}

int pulseg_store_wave_plan(
    pulseg_collection *coll,
    const pulseg_wave_budget *budget,
    pulseg_diagnostic *diag)
{
    if (!coll)
        return PULSEG_ERR_NULL_POINTER;
    pulseg_free_wave_plan(&coll->wave_plan);
    return pulseg_plan_waves(coll, budget, &coll->wave_plan, diag);
}
