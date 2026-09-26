/**
 * @file waves.cpp
 * @brief The rotated waves of a subsequence: every distinct combination its
 *        rotated blocks play, the wave each block plays, and the length each
 *        segment position reserves for them.
 */

#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

extern "C"
{
#include "pulseg_internal.h"
#include "pulseg.h"
}

namespace
{

/* What makes two blocks play the same wave: the definitions and shapes of
 * their three axes, their rotation, and their amplitude ratios to
 * PULSEG_WAVE_RATIO_STEP. */
struct WaveKey
{
    int v[10];
    bool operator==(const WaveKey &other) const
    {
        return std::memcmp(v, other.v, sizeof(v)) == 0;
    }
};

struct WaveKeyHash
{
    std::size_t operator()(const WaveKey &key) const
    {
        std::size_t h = 1469598103934665603ull;
        for (int i = 0; i < 10; ++i)
        {
            h ^= static_cast<std::size_t>(static_cast<unsigned>(key.v[i]));
            h *= 1099511628211ull;
        }
        return h;
    }
};

/* The wave block-table entry @p bte plays, without its peaks and point count;
 * false when it drives no gradient. */
bool wave_of_block(
    const pulseg_sequence_descriptor *desc,
    const pulseg_block_table_element *bte,
    pulseg_wave &wave,
    WaveKey &key)
{
    const int ids[3] = {bte->gx_id, bte->gy_id, bte->gz_id};
    float amplitude[3] = {0.0f, 0.0f, 0.0f};
    bool driven = false;

    std::memset(&wave, 0, sizeof(wave));
    for (int d = 0; d < 3; ++d)
    {
        wave.grad_def[d] = -1;
        if (ids[d] < 0 || ids[d] >= desc->grad_table_size)
            continue;
        const pulseg_grad_table_element &element = desc->grad_table[ids[d]];
        if (element.id < 0 || element.id >= desc->num_unique_grads)
            continue;
        wave.grad_def[d] = element.id;
        wave.shape_id[d] = element.shape_id;
        amplitude[d] = element.amplitude;
        driven = true;
    }
    const float scale = pulseg__wave_scale(desc, bte);
    if (!driven || scale == 0.0f)
        return false;

    wave.rotation_id = -1;
    for (int i = 0; i < 9; ++i)
        wave.rotation[i] = (i % 4 == 0) ? 1.0f : 0.0f;
    if (bte->rotation_id >= 0 && bte->rotation_id < desc->num_rotations &&
        desc->rotation_matrices)
    {
        wave.rotation_id = bte->rotation_id;
        for (int i = 0; i < 9; ++i)
            wave.rotation[i] = desc->rotation_matrices[bte->rotation_id][i];
    }

    for (int d = 0; d < 3; ++d)
    {
        wave.ratio[d] = amplitude[d] / scale;
        key.v[d] = wave.grad_def[d];
        key.v[3 + d] = wave.shape_id[d];
        key.v[7 + d] = static_cast<int>(std::lround(wave.ratio[d] / PULSEG_WAVE_RATIO_STEP));
    }
    key.v[6] = wave.rotation_id;
    return true;
}

/* Fill the peak of each axis and the point count of @p wave. */
int measure_wave(const pulseg_sequence_descriptor *desc, pulseg_wave &wave)
{
    wave.num_points = 0;
    for (int axis = 0; axis < 3; ++axis)
    {
        int points = 0;
        float peak = 0.0f;
        const int rc = pulseg__wave_materialize(
            desc, &wave, axis, nullptr, nullptr, 0, &points, &peak);
        if (PULSEG_FAILED(rc))
            return rc;
        wave.peak[axis] = peak;
        if (points > wave.num_points)
            wave.num_points = points;
    }
    return PULSEG_SUCCESS;
}

} // namespace

/* Record, for a subsequence whose segments and execution stream are built,
 * the rotated wave of every block at a position whose blocks carry a
 * rotation, and at each such position the longest wave it plays.  Blocks at
 * other positions play their own shapes and get none. */
extern "C" int pulseg__build_waves(pulseg_sequence_descriptor *desc)
{
    if (desc->waves)
        PULSEG_FREE(desc->waves);
    if (desc->block_wave)
        PULSEG_FREE(desc->block_wave);
    desc->waves = nullptr;
    desc->block_wave = nullptr;
    desc->num_waves = 0;
    if (desc->num_blocks <= 0)
        return PULSEG_SUCCESS;

    desc->block_wave = static_cast<int *>(PULSEG_ALLOC((size_t)desc->num_blocks * sizeof(int)));
    if (!desc->block_wave)
        return PULSEG_ERR_ALLOC_FAILED;
    for (int i = 0; i < desc->num_blocks; ++i)
        desc->block_wave[i] = -1;
    for (int s = 0; s < desc->num_unique_segments; ++s)
    {
        pulseg_virtual_segment &seg = desc->segment_definitions[s];
        if (seg.initial_states)
            for (int b = 0; b < seg.num_blocks; ++b)
                seg.initial_states[b].wave_points = 0;
    }

    std::unordered_map<WaveKey, int, WaveKeyHash> index;
    std::vector<pulseg_wave> waves;
    int previous = -1;
    int position = 0;
    for (int n = 0; n < desc->exec_stream_len; ++n)
    {
        const int s = pulseg__exec_seg_id(desc, n);
        if (s < 0 || s >= desc->num_unique_segments)
        {
            previous = -1;
            continue;
        }
        pulseg_virtual_segment &seg = desc->segment_definitions[s];
        const int blocks = seg.num_blocks > 0 ? seg.num_blocks : 1;
        position = (s == previous) ? (position + 1) % blocks : 0;
        previous = s;
        if (!seg.has_rotation || !seg.has_rotation[position])
            continue;

        const int block = pulseg__exec_block_idx(desc, n);
        if (block < 0 || block >= desc->num_blocks)
            continue;
        if (desc->block_wave[block] < 0)
        {
            pulseg_wave wave;
            WaveKey key;
            if (!wave_of_block(desc, &desc->block_table[block], wave, key))
                continue;
            auto found = index.find(key);
            if (found == index.end())
            {
                if (!desc->structure_only)
                {
                    const int rc = measure_wave(desc, wave);
                    if (PULSEG_FAILED(rc))
                        return rc;
                }
                found = index.emplace(key, static_cast<int>(waves.size())).first;
                waves.push_back(wave);
            }
            desc->block_wave[block] = found->second;
        }
        if (seg.initial_states)
        {
            int &reserved = seg.initial_states[position].wave_points;
            const int points = waves[static_cast<size_t>(desc->block_wave[block])].num_points;
            if (points > reserved)
                reserved = points;
        }
    }

    if (!waves.empty())
    {
        desc->waves =
            static_cast<pulseg_wave *>(PULSEG_ALLOC(waves.size() * sizeof(pulseg_wave)));
        if (!desc->waves)
            return PULSEG_ERR_ALLOC_FAILED;
        std::memcpy(desc->waves, waves.data(), waves.size() * sizeof(pulseg_wave));
        desc->num_waves = static_cast<int>(waves.size());
    }
    return PULSEG_SUCCESS;
}
