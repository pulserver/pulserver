/**
 * @file repetition.cpp
 * @brief The gradients of the repetition of a subsequence that carries the
 *        most gradient energy, joined over its blocks as corner points.
 */

#include <algorithm>
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

/* The integral over block-table entry @p block of its squared gradient summed
 * over the axes, in (Hz/m)^2 s; its rotation leaves it unchanged. */
float block_energy(const pulseg_sequence_descriptor *desc, int block)
{
    const pulseg_block_table_element &bte = desc->block_table[block];
    const int ids[3] = {bte.gx_id, bte.gy_id, bte.gz_id};
    float energy = 0.0f;
    for (const int id : ids)
    {
        if (id < 0 || id >= desc->grad_table_size)
            continue;
        const pulseg_grad_table_element &element = desc->grad_table[id];
        if (element.id < 0 || element.id >= desc->num_unique_grads)
            continue;
        energy += element.amplitude * element.amplitude *
            pulseg__grad_instance_energy(
                desc, &desc->grad_definitions[element.id], element.shape_id);
    }
    return energy;
}

/* The execution-stream positions per repetition; the whole stream where the
 * subsequence does not repeat. */
int repetition_size(const pulseg_sequence_descriptor *desc)
{
    const int size = desc->tr_descriptor.tr_size;
    return (size > 0 && size < desc->exec_stream_len) ? size : desc->exec_stream_len;
}

float repetition_energy(const pulseg_sequence_descriptor *desc, int first, int count)
{
    float energy = 0.0f;
    for (int n = first; n < first + count && n < desc->exec_stream_len; ++n)
    {
        const int block = pulseg__exec_block_idx(desc, n);
        if (block >= 0 && block < desc->num_blocks)
            energy += block_energy(desc, block);
    }
    return energy;
}

/* The first execution-stream position of the repetition of most gradient
 * energy, the earliest of those that tie, and that energy. */
int heaviest_repetition(const pulseg_sequence_descriptor *desc, int size, float &energy)
{
    int heaviest = 0;
    energy = -1.0f;
    for (int first = 0; first < desc->exec_stream_len; first += size)
    {
        const float e = repetition_energy(desc, first, size);
        if (e > energy)
        {
            energy = e;
            heaviest = first;
        }
    }
    return heaviest;
}

/* Corner points on one timeline, all three axes at every point. */
class Corners
{
  public:
    /* Append a corner.  One at the time of the last replaces it: where one
     * block's gradient meets the next's, the later block's corner stands. */
    void point(float t, float gx, float gy, float gz)
    {
        if (time_.empty() || t > time_.back() + 1e-3f)
        {
            time_.push_back(t);
            gx_.push_back(gx);
            gy_.push_back(gy);
            gz_.push_back(gz);
            return;
        }
        time_.back() = t;
        gx_.back() = gx;
        gy_.back() = gy;
        gz_.back() = gz;
    }

    float last_time() const { return time_.back(); }

    /* Hand the corners to @p out, in arrays the library frees. */
    int store(pulseg_corner_point_stream *out) const
    {
        const size_t n = time_.size();
        out->time_us = copy(time_);
        out->gx_hz_per_m = copy(gx_);
        out->gy_hz_per_m = copy(gy_);
        out->gz_hz_per_m = copy(gz_);
        out->num_points = static_cast<int>(n);
        if (!out->time_us || !out->gx_hz_per_m || !out->gy_hz_per_m || !out->gz_hz_per_m)
            return PULSEG_ERR_ALLOC_FAILED;
        return PULSEG_SUCCESS;
    }

  private:
    static float *copy(const std::vector<float> &values)
    {
        float *out = static_cast<float *>(PULSEG_ALLOC(values.size() * sizeof(float)));
        if (out)
            std::copy(values.begin(), values.end(), out);
        return out;
    }

    std::vector<float> time_, gx_, gy_, gz_;
};

/* The corners block-table entry @p block plays, from @p start_us. */
int append_block(Corners &corners, const pulseg_sequence_descriptor *desc, int block, float start_us)
{
    float *time_us = nullptr;
    float *gradient[3] = {nullptr, nullptr, nullptr};
    int n = 0;
    const int rc = pulseg__block_gradients(desc, block, &time_us, gradient, &n);
    for (int i = 0; i < n && PULSEG_SUCCEEDED(rc); ++i)
        corners.point(start_us + time_us[i], gradient[0][i], gradient[1][i], gradient[2][i]);
    if (time_us)
        PULSEG_FREE(time_us);
    for (float *axis : gradient)
        if (axis)
            PULSEG_FREE(axis);
    return rc;
}

/* Join the blocks of the repetition of @p size positions from @p first, from
 * zero at its start, where no gradient plays there, to zero at its end, where
 * none plays up to it; its duration into @p duration_us. */
int join_repetition(
    Corners &corners,
    const pulseg_sequence_descriptor *desc,
    int first,
    int size,
    float &duration_us)
{
    corners.point(0.0f, 0.0f, 0.0f, 0.0f);
    duration_us = 0.0f;
    int rc = PULSEG_SUCCESS;
    for (int n = first; n < first + size && n < desc->exec_stream_len && PULSEG_SUCCEEDED(rc); ++n)
    {
        const int block = pulseg__exec_block_idx(desc, n);
        if (block < 0 || block >= desc->num_blocks)
            continue;
        rc = append_block(corners, desc, block, duration_us);
        duration_us += static_cast<float>(pulseg__played_duration_us(desc, block));
    }
    if (PULSEG_SUCCEEDED(rc) && corners.last_time() < duration_us - 1e-3f)
        corners.point(duration_us, 0.0f, 0.0f, 0.0f);
    return rc;
}

/* The gradients of the heaviest repetition of @p desc, whose execution stream
 * is built. */
int heaviest_repetition_gradients(
    const pulseg_sequence_descriptor *desc,
    pulseg_corner_point_stream *out)
{
    static const pulseg_corner_point_stream empty = PULSEG_CORNER_POINT_STREAM_INIT;

    if (!desc || !out)
        return PULSEG_ERR_NULL_POINTER;
    *out = empty;
    if (desc->exec_stream_len <= 0 || !desc->block_table)
        return PULSEG_ERR_INVALID_ARGUMENT;

    const int size = repetition_size(desc);
    out->first_position = heaviest_repetition(desc, size, out->energy);
    Corners corners;
    int rc = join_repetition(corners, desc, out->first_position, size, out->duration_us);
    if (PULSEG_SUCCEEDED(rc))
        rc = corners.store(out);
    if (PULSEG_FAILED(rc))
        pulseg_corner_point_stream_free(out);
    return rc;
}

} // namespace

int pulseg_store_repetitions(pulseg_collection *coll)
{
    if (!coll)
        return PULSEG_ERR_NULL_POINTER;
    for (int s = 0; s < coll->num_subsequences; ++s)
    {
        pulseg_sequence_descriptor *desc = &coll->descriptors[s];
        pulseg_corner_point_stream_free(&desc->repetition);
        if (desc->structure_only)
            continue;
        const int rc = heaviest_repetition_gradients(desc, &desc->repetition);
        if (PULSEG_FAILED(rc))
            return rc;
    }
    return PULSEG_SUCCESS;
}
