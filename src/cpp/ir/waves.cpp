/**
 * @file waves.cpp
 * @brief The waves of a subsequence: every distinct combination the blocks at
 *        its wave positions play, the wave each block plays, and the span
 *        each segment position reserves for them.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <utility>
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

/* A segment position: its segment, -1 outside every segment, and the index
 * of its block in the segment. */
struct Position
{
    int segment = -1;
    int index = 0;
};

/* The position each execution-stream entry plays at, entry by entry. */
class PositionWalk
{
  public:
    Position next(const pulseg_sequence_descriptor *desc, int n)
    {
        const int s = pulseg__exec_seg_id(desc, n);
        if (s < 0 || s >= desc->num_unique_segments)
        {
            previous_ = -1;
            return Position{};
        }
        const int blocks = std::max(desc->segment_definitions[s].num_blocks, 1);
        index_ = (s == previous_) ? (index_ + 1) % blocks : 0;
        previous_ = s;
        return Position{s, index_};
    }

  private:
    int previous_ = -1;
    int index_ = 0;
};

/* Per axis, the gradient definition and shape a block plays; -1 where it
 * drives none. */
using Events = std::array<std::array<int, 2>, 3>;

Events gradient_events(const pulseg_sequence_descriptor *desc, const pulseg_block_table_element &bte)
{
    const int ids[3] = {bte.gx_id, bte.gy_id, bte.gz_id};
    Events events;
    for (int d = 0; d < 3; ++d)
    {
        events[static_cast<size_t>(d)] = {-1, 0};
        if (ids[d] < 0 || ids[d] >= desc->grad_table_size)
            continue;
        const pulseg_grad_table_element &element = desc->grad_table[ids[d]];
        if (element.id >= 0 && element.id < desc->num_unique_grads)
            events[static_cast<size_t>(d)] = {element.id, element.shape_id};
    }
    return events;
}

/* Per segment position, whether a block there drives an axis with a gradient
 * definition or shape other than the one the position's initial state names:
 * the event a playout prepares for the position, which cannot follow it by
 * its amplitude.  An axis a block does not drive plays that event at zero. */
class ShapeVariation
{
  public:
    explicit ShapeVariation(const pulseg_sequence_descriptor *desc)
        : desc_(desc), varies_(static_cast<size_t>(std::max(desc->num_unique_segments, 0)))
    {
        for (size_t s = 0; s < varies_.size(); ++s)
            varies_[s].assign(
                static_cast<size_t>(std::max(desc->segment_definitions[s].num_blocks, 1)), 0);
    }

    void note(const Position &at, const Events &events)
    {
        const pulseg_virtual_segment &seg = desc_->segment_definitions[at.segment];
        if (!seg.initial_states || at.index >= seg.num_blocks)
            return;
        const pulseg_block_initial_state &prepared = seg.initial_states[at.index];
        for (int d = 0; d < 3; ++d)
        {
            const std::array<int, 2> &played = events[static_cast<size_t>(d)];
            if (played[0] >= 0 &&
                (played[0] != prepared.grad_def_id[d] || played[1] != prepared.grad_shape_id[d]))
                flag(at) = 1;
        }
    }

    bool varies(const Position &at) { return flag(at) != 0; }

  private:
    char &flag(const Position &at)
    {
        return varies_[static_cast<size_t>(at.segment)][static_cast<size_t>(at.index)];
    }

    const pulseg_sequence_descriptor *desc_;
    std::vector<std::vector<char>> varies_;
};

/* Whether the blocks at @p at play waves rather than the position's own
 * gradient events: where they carry a rotation, or drive an axis the events
 * cannot play. */
bool plays_waves(const pulseg_sequence_descriptor *desc, const Position &at, ShapeVariation &shapes)
{
    const pulseg_virtual_segment &seg = desc->segment_definitions[at.segment];
    return (seg.has_rotation && seg.has_rotation[at.index]) || shapes.varies(at);
}

/* The waves and the positions that play them, joined into groups that share
 * one span: every wave, and every position that plays one, covers the union
 * of the spans of the waves of its group.  The waves a position plays then
 * all cover the one interval, and so does every position a wave plays at. */
class SpanGroups
{
  public:
    explicit SpanGroups(const pulseg_sequence_descriptor *desc)
    {
        int total = 0;
        first_.reserve(static_cast<size_t>(std::max(desc->num_unique_segments, 0)));
        for (int s = 0; s < desc->num_unique_segments; ++s)
        {
            first_.push_back(total);
            total += std::max(desc->segment_definitions[s].num_blocks, 1);
        }
        positions_ = total;
    }

    void played(const Position &at, int wave)
    {
        edges_.emplace_back(node(at), wave);
    }

    /* Widen the spans of the waves of @p desc and of the positions that play
     * them to those of their groups. */
    void widen(pulseg_sequence_descriptor *desc)
    {
        parent_.resize(static_cast<size_t>(positions_ + desc->num_waves));
        for (size_t i = 0; i < parent_.size(); ++i)
            parent_[i] = static_cast<int>(i);
        for (const auto &edge : edges_)
            parent_[static_cast<size_t>(root(edge.first))] = root(positions_ + edge.second);
        std::vector<std::pair<float, float>> span(parent_.size(), {0.0f, 0.0f});
        std::vector<bool> spanned(parent_.size(), false);
        for (int w = 0; w < desc->num_waves; ++w)
        {
            const size_t r = static_cast<size_t>(root(positions_ + w));
            const pulseg_wave &wave = desc->waves[w];
            span[r] = spanned[r] ? std::make_pair(std::min(span[r].first, wave.start_us),
                                                  std::max(span[r].second, wave.end_us))
                                 : std::make_pair(wave.start_us, wave.end_us);
            spanned[r] = true;
        }
        for (int w = 0; w < desc->num_waves; ++w)
        {
            const auto &group = span[static_cast<size_t>(root(positions_ + w))];
            desc->waves[w].start_us = group.first;
            desc->waves[w].end_us = group.second;
        }
        for (int s = 0; s < desc->num_unique_segments; ++s)
            widen_positions(desc->segment_definitions[s], s, span);
    }

  private:
    int node(const Position &at) const
    {
        return first_[static_cast<size_t>(at.segment)] + at.index;
    }

    int root(int x)
    {
        while (parent_[static_cast<size_t>(x)] != x)
        {
            parent_[static_cast<size_t>(x)] =
                parent_[static_cast<size_t>(parent_[static_cast<size_t>(x)])];
            x = parent_[static_cast<size_t>(x)];
        }
        return x;
    }

    void widen_positions(
        pulseg_virtual_segment &seg,
        int s,
        const std::vector<std::pair<float, float>> &span)
    {
        if (!seg.initial_states)
            return;
        for (int b = 0; b < seg.num_blocks; ++b)
        {
            pulseg_block_initial_state &state = seg.initial_states[b];
            if (state.wave_points == 0)
                continue;
            const auto &group = span[static_cast<size_t>(root(node(Position{s, b})))];
            state.wave_start_us = group.first;
            state.wave_end_us = group.second;
        }
    }

    std::vector<int> first_;
    int positions_ = 0;
    std::vector<std::pair<int, int>> edges_;
    std::vector<int> parent_;
};

/* Note, entry by entry, the gradient events each position's blocks play. */
void note_shapes(const pulseg_sequence_descriptor *desc, ShapeVariation &shapes)
{
    PositionWalk walk;
    for (int n = 0; n < desc->exec_stream_len; ++n)
    {
        const Position at = walk.next(desc, n);
        const int block = pulseg__exec_block_idx(desc, n);
        if (at.segment >= 0 && block >= 0 && block < desc->num_blocks)
            shapes.note(at, gradient_events(desc, desc->block_table[block]));
    }
}

/* The block-table entry execution-stream entry @p n plays, at @p at, where
 * that position plays waves; -1 elsewhere. */
int wave_block(
    const pulseg_sequence_descriptor *desc,
    int n,
    const Position &at,
    ShapeVariation &shapes)
{
    const int block = pulseg__exec_block_idx(desc, n);
    if (at.segment < 0 || block < 0 || block >= desc->num_blocks || !plays_waves(desc, at, shapes))
        return -1;
    return block;
}

/* Give every block at a position that plays waves its wave, and each such
 * position its record of the waves it plays. */
int assign_waves(
    pulseg_sequence_descriptor *desc,
    ShapeVariation &shapes,
    WaveTable &waves,
    SpanGroups &groups)
{
    int rc = PULSEG_SUCCESS;
    PositionWalk walk;
    for (int n = 0; n < desc->exec_stream_len; ++n)
    {
        const Position at = walk.next(desc, n);
        const int block = wave_block(desc, n, at, shapes);
        if (block < 0)
            continue;
        if (desc->block_wave[block] < 0)
            desc->block_wave[block] = waves.intern(block, rc);
        if (PULSEG_FAILED(rc))
            return rc;
        const int w = desc->block_wave[block];
        if (w < 0)
            continue;
        pulseg_virtual_segment &seg = desc->segment_definitions[at.segment];
        if (seg.initial_states)
            reserve(seg.initial_states[at.index], waves[w]);
        groups.played(at, w);
    }
    return rc;
}

} // namespace

/* Record, for a subsequence whose segments and execution stream are built,
 * the wave of every block at a position that plays waves -- one whose blocks
 * carry a rotation, or drive an axis with a gradient definition or shape other
 * than the one the position's events are prepared with -- and at each such
 * position the points, span and axes of the waves it plays.  Blocks at other
 * positions play their position's events and get none. */
extern "C" int pulseg__build_waves(pulseg_sequence_descriptor *desc)
{
    int rc = reset_waves(desc);
    if (PULSEG_FAILED(rc) || desc->num_blocks <= 0)
        return rc;

    ShapeVariation shapes(desc);
    note_shapes(desc, shapes);
    WaveTable waves(desc);
    SpanGroups groups(desc);
    rc = assign_waves(desc, shapes, waves, groups);
    if (PULSEG_SUCCEEDED(rc))
        rc = waves.store(desc);
    if (PULSEG_SUCCEEDED(rc))
        groups.widen(desc);
    return rc;
}
