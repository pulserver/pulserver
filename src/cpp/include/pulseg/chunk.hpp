/**
 * @file chunk.hpp
 * @brief Chunk planning and waveform materialisation.
 *
 * C++ over pulseg_chunk.h. ChunkPlan owns the C plan; materialise() returns
 * the points by value.
 */

#ifndef PULSEG_CHUNK_HPP
#define PULSEG_CHUNK_HPP

#include <cstring>
#include <vector>

#include "pulseg.h"

#include "error.hpp"
#include "types.hpp"

namespace pulseg
{
    /** One materialised waveform: the corner points, and the peak. */
    struct MaterialisedWave
    {
        std::vector<float> time_us;
        std::vector<float> amplitude;
        float peak = 0.0f;
    };

    /**
     * How a subsequence's waveforms reach the hardware.
     *
     * Owns a pulseg_chunk_plan and releases it in the destructor. Move-only.
     *
     * The collection must carry the execution stream: the planner needs to
     * know how often each waveform is played and in what order.
     */
    class ChunkPlan
    {
    public:
        ChunkPlan(const pulseg_collection* coll, int subseq_idx, const pulseg_chunk_budget& budget)
            : coll_(coll), subseq_idx_(subseq_idx)
        {
            pulseg_diagnostic diag;
            pulseg_diagnostic_init(&diag);
            check(pulseg_plan_chunks(coll, subseq_idx, &budget, &plan_, &diag), diag);
        }

        ~ChunkPlan()
        {
            pulseg_free_chunk_plan(&plan_);
        }

        ChunkPlan(ChunkPlan&& o) noexcept
            : plan_(o.plan_), coll_(o.coll_), subseq_idx_(o.subseq_idx_)
        {
            std::memset(&o.plan_, 0, sizeof o.plan_);
        }
        ChunkPlan& operator=(ChunkPlan&& o) noexcept
        {
            if (this != &o)
            {
                pulseg_free_chunk_plan(&plan_);
                plan_ = o.plan_;
                coll_ = o.coll_;
                subseq_idx_ = o.subseq_idx_;
                std::memset(&o.plan_, 0, sizeof o.plan_);
            }
            return *this;
        }
        ChunkPlan(const ChunkPlan&) = delete;
        ChunkPlan& operator=(const ChunkPlan&) = delete;

        /** PULSEG_WAVE_RESIDENT or PULSEG_WAVE_STREAMED. */
        pulseg_wave_mode mode() const
        {
            return plan_.mode;
        }
        int num_chunks() const
        {
            return plan_.num_chunks;
        }
        int num_waves() const
        {
            return plan_.num_waves;
        }

        /** Every distinct waveform in the subsequence, once. */
        const pulseg_wave_key& wave(int index) const
        {
            return plan_.waves[index];
        }

        const pulseg_chunk& chunk(int index) const
        {
            return plan_.chunks[index];
        }

        /** Which wave each exec-stream position plays; -1 where none. */
        std::vector<int> position_waves() const
        {
            return std::vector<int>(
                plan_.position_wave,
                plan_.position_wave + plan_.position_count);
        }

        /** Render one axis of one wave. */
        MaterialisedWave materialise(int wave_index, int axis, int max_points = 8192) const
        {
            MaterialisedWave out;
            out.time_us.resize(max_points);
            out.amplitude.resize(max_points);
            int num_points = 0;

            check(pulseg_materialize_wave(
                coll_,
                subseq_idx_,
                &plan_.waves[wave_index],
                axis,
                out.time_us.data(),
                out.amplitude.data(),
                max_points,
                &num_points,
                &out.peak));

            out.time_us.resize(num_points);
            out.amplitude.resize(num_points);
            return out;
        }

        const pulseg_chunk_plan* handle() const
        {
            return &plan_;
        }

    private:
        pulseg_chunk_plan plan_ = PULSEG_CHUNK_PLAN_INIT;
        const pulseg_collection* coll_ = nullptr;
        int subseq_idx_ = 0;
    };
} // namespace pulseg

#endif // PULSEG_CHUNK_HPP
