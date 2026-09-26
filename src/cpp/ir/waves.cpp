/**
 * @file waves.cpp
 * @brief The rotated waves of a subsequence: every distinct combination its
 *        rotated blocks play, the wave each block plays, and the length each
 *        segment position reserves for them.
 */

#include <algorithm>
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
    if (!pulseg__block_combination(desc, bte, &wave))
        return false;
    for (int d = 0; d < 3; ++d)
    {
        key.v[d] = wave.grad_def[d];
        key.v[3 + d] = wave.shape_id[d];
        key.v[7 + d] = static_cast<int>(std::lround(wave.ratio[d] / PULSEG_WAVE_RATIO_STEP));
    }
    key.v[6] = wave.rotation_id;
    return true;
}

/* Fill the peak of each axis, the point count and the span of @p wave. */
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
    wave.start_us = 0.0f;
    wave.end_us = 0.0f;
    if (wave.num_points < 1)
        return PULSEG_SUCCESS;
    /* Every axis shares one grid, so one axis gives the span. */
    std::vector<float> time_us(static_cast<size_t>(wave.num_points));
    std::vector<float> amplitude(static_cast<size_t>(wave.num_points));
    int points = 0;
    const int rc = pulseg__wave_materialize(
        desc, &wave, 0, time_us.data(), amplitude.data(), wave.num_points, &points, nullptr);
    if (PULSEG_FAILED(rc))
        return rc;
    wave.start_us = time_us.front();
    wave.end_us = time_us[static_cast<size_t>(points - 1)];
    return PULSEG_SUCCESS;
}

/* Widen a position's record of the waves it plays by @p wave. */
void reserve(pulseg_block_initial_state &state, const pulseg_wave &wave)
{
    if (state.wave_points == 0 || wave.start_us < state.wave_start_us)
        state.wave_start_us = wave.start_us;
    if (state.wave_points == 0 || wave.end_us > state.wave_end_us)
        state.wave_end_us = wave.end_us;
    if (wave.num_points > state.wave_points)
        state.wave_points = wave.num_points;
    for (int axis = 0; axis < 3; ++axis)
        if (wave.peak[axis] > 0.0f)
            state.wave_axes |= 1 << axis;
}

/* Clear what an earlier build left, and give every block no wave. */
int reset_waves(pulseg_sequence_descriptor *desc)
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
    std::fill(desc->block_wave, desc->block_wave + desc->num_blocks, -1);
    for (int s = 0; s < desc->num_unique_segments; ++s)
    {
        pulseg_virtual_segment &seg = desc->segment_definitions[s];
        if (seg.initial_states)
            for (int b = 0; b < seg.num_blocks; ++b)
            {
                seg.initial_states[b].wave_points = 0;
                seg.initial_states[b].wave_start_us = 0.0f;
                seg.initial_states[b].wave_end_us = 0.0f;
                seg.initial_states[b].wave_axes = 0;
            }
    }
    return PULSEG_SUCCESS;
}

/* The distinct waves of a subsequence, in the order its blocks first play
 * them. */
class WaveTable
{
  public:
    explicit WaveTable(const pulseg_sequence_descriptor *desc) : desc_(desc) {}

    /* The index of the wave block-table entry @p block plays, -1 when it
     * drives no gradient; @p rc says whether measuring a new one failed. */
    int intern(int block, int &rc)
    {
        pulseg_wave wave;
        WaveKey key;
        if (!wave_of_block(desc_, &desc_->block_table[block], wave, key))
            return -1;
        const auto found = index_.find(key);
        if (found != index_.end())
            return found->second;
        if (!desc_->structure_only)
        {
            rc = measure_wave(desc_, wave);
            if (PULSEG_FAILED(rc))
                return -1;
        }
        const int added = static_cast<int>(waves_.size());
        index_.emplace(key, added);
        waves_.push_back(wave);
        return added;
    }

    const pulseg_wave &operator[](int w) const { return waves_[static_cast<size_t>(w)]; }

    /* Hand the waves to @p desc. */
    int store(pulseg_sequence_descriptor *desc) const
    {
        if (waves_.empty())
            return PULSEG_SUCCESS;
        desc->waves =
            static_cast<pulseg_wave *>(PULSEG_ALLOC(waves_.size() * sizeof(pulseg_wave)));
        if (!desc->waves)
            return PULSEG_ERR_ALLOC_FAILED;
        std::copy(waves_.begin(), waves_.end(), desc->waves);
        desc->num_waves = static_cast<int>(waves_.size());
        return PULSEG_SUCCESS;
    }

  private:
    const pulseg_sequence_descriptor *desc_;
    std::unordered_map<WaveKey, int, WaveKeyHash> index_;
    std::vector<pulseg_wave> waves_;
};

/* The position each execution-stream entry plays at in its segment
 * instance, entry by entry. */
class PositionWalk
{
  public:
    /* The block-table entry @p n plays where the blocks of its position carry
     * a rotation, with @p state that position's record, null when the segment
     * keeps none; -1 elsewhere. */
    int rotated_block(pulseg_sequence_descriptor *desc, int n, pulseg_block_initial_state *&state)
    {
        const int block = pulseg__exec_block_idx(desc, n);
        const bool rotated = advance(desc, pulseg__exec_seg_id(desc, n), state);
        return (rotated && block >= 0 && block < desc->num_blocks) ? block : -1;
    }

  private:
    bool advance(pulseg_sequence_descriptor *desc, int s, pulseg_block_initial_state *&state)
    {
        state = nullptr;
        if (s < 0 || s >= desc->num_unique_segments)
        {
            previous_ = -1;
            return false;
        }
        pulseg_virtual_segment &seg = desc->segment_definitions[s];
        const int blocks = seg.num_blocks > 0 ? seg.num_blocks : 1;
        position_ = (s == previous_) ? (position_ + 1) % blocks : 0;
        previous_ = s;
        if (!seg.has_rotation || !seg.has_rotation[position_])
            return false;
        if (seg.initial_states)
            state = &seg.initial_states[position_];
        return true;
    }

    int previous_ = -1;
    int position_ = 0;
};

} // namespace

/* Record, for a subsequence whose segments and execution stream are built,
 * the rotated wave of every block at a position whose blocks carry a
 * rotation, and at each such position the points, span and axes of the waves
 * it plays.  Blocks at other positions play their own shapes and get none. */
extern "C" int pulseg__build_waves(pulseg_sequence_descriptor *desc)
{
    int rc = reset_waves(desc);
    if (PULSEG_FAILED(rc) || desc->num_blocks <= 0)
        return rc;

    WaveTable waves(desc);
    PositionWalk walk;
    for (int n = 0; n < desc->exec_stream_len; ++n)
    {
        pulseg_block_initial_state *state = nullptr;
        const int block = walk.rotated_block(desc, n, state);
        if (block < 0)
            continue;
        if (desc->block_wave[block] < 0)
            desc->block_wave[block] = waves.intern(block, rc);
        if (PULSEG_FAILED(rc))
            return rc;
        if (state && desc->block_wave[block] >= 0)
            reserve(*state, waves[desc->block_wave[block]]);
    }
    return waves.store(desc);
}
