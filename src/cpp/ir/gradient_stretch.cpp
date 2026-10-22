/**
 * @file gradient_stretch.cpp
 * @brief The gradients of the stretches of a subsequence a gradient-heating
 *        model is evaluated on, joined over their blocks as corner points.
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

float position_energy(const pulseg_sequence_descriptor *desc, int n)
{
    const int block = pulseg__exec_block_idx(desc, n);
    return (block >= 0 && block < desc->num_blocks) ? block_energy(desc, block) : 0.0f;
}

/* The execution-stream positions per repetition; the whole stream where the
 * subsequence does not repeat. */
int repetition_size(const pulseg_sequence_descriptor *desc)
{
    const int size = desc->tr_descriptor.tr_size;
    return (size > 0 && size < desc->exec_stream_len) ? size : desc->exec_stream_len;
}

/* The first execution-stream position of the repetition of most gradient
 * energy, the earliest of those that tie. */
int heaviest_repetition(const pulseg_sequence_descriptor *desc)
{
    const int size = repetition_size(desc);
    int heaviest = 0;
    double most = -1.0;
    for (int first = 0; first < desc->exec_stream_len; first += size)
    {
        double e = 0.0;
        for (int n = first; n < first + size && n < desc->exec_stream_len; ++n)
            e += position_energy(desc, n);
        if (e > most)
        {
            most = e;
            heaviest = first;
        }
    }
    return heaviest;
}

/* The segment instances of the execution stream in play order, from one that
 * starts at the walk's first position.  Consecutive instances of one segment
 * part every num_blocks positions; a position without a segment is an
 * instance of its own. */
class InstanceWalk
{
  public:
    InstanceWalk(const pulseg_sequence_descriptor *desc, int from) : desc_(desc), next_(from) {}

    /* Step to the next instance; false past the stream's end. */
    bool advance()
    {
        if (next_ >= desc_->exec_stream_len)
            return false;
        first = next_;
        count = 0;
        duration_us = 0.0;
        energy = 0.0;
        const int local = pulseg__exec_seg_id(desc_, first);
        const int blocks = (local >= 0 && local < desc_->num_unique_segments)
            ? desc_->segment_definitions[local].num_blocks
            : 1;
        do
        {
            const int block = pulseg__exec_block_idx(desc_, next_);
            if (block >= 0 && block < desc_->num_blocks)
            {
                duration_us += static_cast<double>(pulseg__played_duration_us(desc_, block));
                energy += block_energy(desc_, block);
            }
            ++count;
            ++next_;
        } while (count < blocks && next_ < desc_->exec_stream_len &&
                 pulseg__exec_seg_id(desc_, next_) == local);
        return true;
    }

    int first = 0;
    int count = 0;
    double duration_us = 0.0;
    double energy = 0.0;

  private:
    const pulseg_sequence_descriptor *desc_;
    int next_;
};

/* A run of whole segment instances. */
struct Stretch
{
    int first = 0;
    int count = 0;
    double duration_us = 0.0;
    double energy = 0.0;
};

/* The instances from the one starting at @p from until they last
 * @p window_us, or to the stream's end where it comes first. */
Stretch stretch_from(const pulseg_sequence_descriptor *desc, int from, double window_us)
{
    Stretch s;
    s.first = from;
    InstanceWalk walk(desc, from);
    while (s.duration_us < window_us && walk.advance())
    {
        s.count += walk.count;
        s.duration_us += walk.duration_us;
        s.energy += walk.energy;
    }
    return s;
}

/* Of the stretches lasting @p window_us from an instance's start, the one of
 * most gradient energy, the earliest on a tie; the whole stream where it is
 * shorter.  Its end slides ahead of its start, each walk passing once. */
Stretch heaviest_stretch(const pulseg_sequence_descriptor *desc, double window_us)
{
    InstanceWalk lag(desc, 0), lead(desc, 0);
    double duration = 0.0, energy = 0.0, most = -1.0;
    int from = 0;
    while (lag.advance())
    {
        while (duration < window_us && lead.advance())
        {
            duration += lead.duration_us;
            energy += lead.energy;
        }
        if (duration < window_us)
            break;
        if (energy > most + 1e-9 * most)
        {
            most = energy;
            from = lag.first;
        }
        duration -= lag.duration_us;
        energy -= lag.energy;
    }
    return stretch_from(desc, from, window_us);
}

/* The first position of the instance holding position @p n. */
int instance_holding(const pulseg_sequence_descriptor *desc, int n)
{
    InstanceWalk walk(desc, 0);
    while (walk.advance())
        if (n < walk.first + walk.count)
            return walk.first;
    return 0;
}

/* Corner points on one timeline, all three axes at every point. */
class Corners
{
  public:
    /* Append a corner.  One at the time of the last replaces it: where one
     * block's gradient meets the next's, the later block's corner stands. */
    void point(double t_us, float gx, float gy, float gz)
    {
        const float t = static_cast<float>(t_us);
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
int append_block(Corners &corners, const pulseg_sequence_descriptor *desc, int block, double start_us)
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

/* The gradients of @p s into @p out: from zero at its start, where no
 * gradient plays there, to zero at its end, where none plays up to it. */
int join(const pulseg_sequence_descriptor *desc, const Stretch &s, pulseg_corner_point_stream *out)
{
    Corners corners;
    corners.point(0.0, 0.0f, 0.0f, 0.0f);
    double t = 0.0;
    int rc = PULSEG_SUCCESS;
    for (int n = s.first; n < s.first + s.count && PULSEG_SUCCEEDED(rc); ++n)
    {
        const int block = pulseg__exec_block_idx(desc, n);
        if (block < 0 || block >= desc->num_blocks)
            continue;
        rc = append_block(corners, desc, block, t);
        t += static_cast<double>(pulseg__played_duration_us(desc, block));
    }
    if (PULSEG_SUCCEEDED(rc) && corners.last_time() < static_cast<float>(t) - 1e-3f)
        corners.point(t, 0.0f, 0.0f, 0.0f);
    if (PULSEG_SUCCEEDED(rc))
        rc = corners.store(out);
    out->duration_us = static_cast<float>(t);
    out->first_position = s.first;
    out->num_positions = s.count;
    out->energy = static_cast<float>(s.energy);
    return rc;
}

/* The stretches of @p desc, whose execution stream is built, into
 * @p out[PULSEG_NUM_STRETCHES]. */
int store_stretches(const pulseg_sequence_descriptor *desc, pulseg_corner_point_stream *out)
{
    static const pulseg_corner_point_stream empty = PULSEG_CORNER_POINT_STREAM_INIT;

    for (int k = 0; k < PULSEG_NUM_STRETCHES; ++k)
        out[k] = empty;
    if (desc->exec_stream_len <= 0 || !desc->block_table)
        return PULSEG_ERR_INVALID_ARGUMENT;

    const Stretch chosen[PULSEG_NUM_STRETCHES] = {
        heaviest_stretch(desc, PULSEG_STRETCH_US),
        stretch_from(desc, instance_holding(desc, heaviest_repetition(desc)), PULSEG_STRETCH_US),
    };
    int rc = PULSEG_SUCCESS;
    for (int k = 0; k < PULSEG_NUM_STRETCHES && PULSEG_SUCCEEDED(rc); ++k)
        rc = join(desc, chosen[k], &out[k]);
    if (PULSEG_FAILED(rc))
        for (int k = 0; k < PULSEG_NUM_STRETCHES; ++k)
            pulseg_corner_point_stream_free(&out[k]);
    return rc;
}

} // namespace

int pulseg_store_gradient_stretches(pulseg_collection *coll)
{
    if (!coll)
        return PULSEG_ERR_NULL_POINTER;
    for (int s = 0; s < coll->num_subsequences; ++s)
    {
        pulseg_sequence_descriptor *desc = &coll->descriptors[s];
        for (pulseg_corner_point_stream &stretch : desc->stretches)
            pulseg_corner_point_stream_free(&stretch);
        if (desc->structure_only)
            continue;
        const int rc = store_stretches(desc, desc->stretches);
        if (PULSEG_FAILED(rc))
            return rc;
    }
    return PULSEG_SUCCESS;
}
