/**
 * @file playout.cpp
 * @brief A playout backend that plays nothing and records what both stages
 *        hand it, with a model of its waveform memory.
 */

#include "playout.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "native.hpp"

namespace native
{
namespace
{

/* A stretch of one axis's waveform memory, [begin, end). */
struct Span
{
    int axis;
    long begin;
    long end;
};

bool overlap(const Span &a, const Span &b)
{
    return a.axis == b.axis && a.begin < b.end && b.begin < a.end;
}

/* What the first stage prepares at each position: the gradient events it
 * plays at their amplitudes where it plays no wave, as corners at unit
 * amplitude, and the slots its waves play from where it does. */
class Prepared
{
  public:
    void add(
        const pulseg_collection *coll,
        int segment,
        int position,
        const pulseg_block_info &info,
        const pulseg_wave_region *slot)
    {
        segment_.push_back(segment);
        position_.push_back(position);
        for (int axis = 0; axis < 3; ++axis)
        {
            span_.push_back(static_cast<py::ssize_t>(time_us_.size()));
            if (!slot)
                event(coll, segment, position, axis, info);
            span_.push_back(static_cast<py::ssize_t>(time_us_.size()));
        }
        for (int h = 0; h < 2; ++h)
        {
            for (int axis = 0; axis < 3; ++axis)
                slot_offset_.push_back(slot ? slot[h].offset[axis] : -1);
            slot_samples_.push_back(slot ? slot[h].samples : 0);
            slot_start_us_.push_back(slot ? slot[h].start_us : 0.0f);
        }
    }

    py::dict result() const
    {
        const auto n = static_cast<py::ssize_t>(segment_.size());
        const auto corners = static_cast<py::ssize_t>(time_us_.size());
        py::dict out;
        out["segment"] = as_array(segment_, {n});
        out["position"] = as_array(position_, {n});
        out["event_time_us"] = as_array(time_us_, {corners});
        out["event_shape"] = as_array(shape_, {corners});
        out["event_span"] = as_array(span_, {n, 3, 2});
        out["slot_offset"] = as_array(slot_offset_, {n, 2, 3});
        out["slot_samples"] = as_array(slot_samples_, {n, 2});
        out["slot_start_us"] = as_array(slot_start_us_, {n, 2});
        return out;
    }

  private:
    void event(
        const pulseg_collection *coll,
        int segment,
        int position,
        int axis,
        const pulseg_block_info &info)
    {
        Channels shape;
        int samples = 0;
        shape.samples =
            pulseg_get_grad_amplitude(coll, &shape.count, &samples, segment, position, axis);
        if (!shape.samples || shape.count < 1 || !shape.samples[0] || samples <= 0)
            return;
        Waveform time;
        time.samples = pulseg_get_grad_time_us(coll, segment, position, axis);
        if (!time.samples)
            throw std::runtime_error("cannot read the gradient a position prepares");
        append_event(
            time.samples, shape.samples[0], samples,
            static_cast<float>(std::max(info.grad_delay_us[axis], 0)), 1.0f, time_us_, shape_);
    }

    std::vector<int> segment_, position_;
    std::vector<float> time_us_, shape_;
    std::vector<py::ssize_t> span_;
    std::vector<long> slot_offset_, slot_samples_;
    std::vector<float> slot_start_us_;
};

/* The registers the second stage sets on each block, in play order, and the
 * samples each block's wave reads from waveform memory. */
class Played
{
  public:
    void add(const pulseg_playout_segment &segment, const pulseg_playout_block &block)
    {
        const pulseg_block_instance &b = block.instance;
        const int where[] = {segment.subsequence, segment.segment, block.position,
                             segment.instance, segment.half, segment.rotate,
                             segment.await_trigger, segment.first_position};
        const int variable[] = {b.gx_variable, b.gy_variable, b.gz_variable};
        const float rf[] = {b.rf_amp_hz, b.rf_phase_rad, b.rf_freq_hz};
        const float adc[] = {b.adc_freq_hz, b.adc_phase_rad};
        const float gradient[] = {b.gx_amp_hz_per_m, b.gy_amp_hz_per_m, b.gz_amp_hz_per_m};
        where_.insert(where_.end(), std::begin(where), std::end(where));
        rf_.insert(rf_.end(), std::begin(rf), std::end(rf));
        adc_.insert(adc_.end(), std::begin(adc), std::end(adc));
        gradient_.insert(gradient_.end(), std::begin(gradient), std::end(gradient));
        variable_.insert(variable_.end(), std::begin(variable), std::end(variable));
        wave_amp_.insert(wave_amp_.end(), b.wave_amp_hz_per_m, b.wave_amp_hz_per_m + 3);
        flags_.push_back(b.duration_us);
        flags_.push_back(b.rf_shim_id);
        flags_.push_back(b.wave_id);
        flags_.push_back(b.adc_flag);
        flags_.push_back(b.digitalout_flag);
        for (int axis = 0; axis < 3; ++axis)
            region_offset_.push_back(block.wave ? block.wave->offset[axis] : -1);
        region_samples_.push_back(block.wave ? block.wave->samples : 0);
        region_start_us_.push_back(block.wave ? block.wave->start_us : 0.0f);
    }

    /* Append what one axis of the last block reads from @p memory. */
    long read(const std::vector<float> &memory, const pulseg_wave_region *region, int axis)
    {
        long unloaded = 0;
        read_span_.push_back(static_cast<py::ssize_t>(read_.size()));
        if (region && region->offset[axis] >= 0)
            for (long i = 0; i < region->samples; ++i)
            {
                const float value = memory[static_cast<size_t>(region->offset[axis] + i)];
                unloaded += std::isnan(value) ? 1 : 0;
                read_.push_back(value);
            }
        read_span_.push_back(static_cast<py::ssize_t>(read_.size()));
        return unloaded;
    }

    py::dict result() const
    {
        const auto n = static_cast<py::ssize_t>(region_samples_.size());
        py::dict out;
        static const char *const where[] = {"subsequence", "segment", "position", "instance",
                                            "half", "rotate", "await_trigger", "first_position"};
        static const char *const flags[] = {"duration_us", "rf_shim", "wave", "adc",
                                            "digitalout"};
        column(out, where, 8, where_);
        column(out, flags, 5, flags_);
        out["rf_amp_hz"] = strided(rf_, 3, 0);
        out["rf_phase_rad"] = strided(rf_, 3, 1);
        out["rf_freq_hz"] = strided(rf_, 3, 2);
        out["adc_freq_hz"] = strided(adc_, 2, 0);
        out["adc_phase_rad"] = strided(adc_, 2, 1);
        out["gradient_hz_per_m"] = as_array(gradient_, {n, 3});
        out["gradient_variable"] = as_array(variable_, {n, 3});
        out["wave_amp_hz_per_m"] = as_array(wave_amp_, {n, 3});
        out["wave_offset"] = as_array(region_offset_, {n, 3});
        out["wave_samples"] = as_array(region_samples_, {n});
        out["wave_start_us"] = as_array(region_start_us_, {n});
        out["wave_read"] = as_array(read_, {static_cast<py::ssize_t>(read_.size())});
        out["wave_read_span"] = as_array(read_span_, {n, 3, 2});
        return out;
    }

  private:
    /* Column @p c of each of the @p count-wide rows of @p values. */
    template <typename T>
    static py::array_t<T> strided(const std::vector<T> &values, size_t count, size_t c)
    {
        std::vector<T> picked;
        for (size_t i = c; i < values.size(); i += count)
            picked.push_back(values[i]);
        return as_array(picked, {static_cast<py::ssize_t>(picked.size())});
    }

    static void column(py::dict &out, const char *const *names, size_t count, const std::vector<int> &rows)
    {
        for (size_t c = 0; c < count; ++c)
            out[names[c]] = strided(rows, count, c);
    }

    std::vector<int> where_, flags_, variable_;
    std::vector<float> rf_, adc_, gradient_, wave_amp_;
    std::vector<long> region_offset_, region_samples_;
    std::vector<float> region_start_us_, read_;
    std::vector<py::ssize_t> read_span_;
};

/* The backend: a model of the waveform memory, and what the stages hand it. */
class Recorder
{
  public:
    explicit Recorder(const pulseg_collection *coll) : coll_(coll) {}

    int reserve(const pulseg_wave_plan &plan)
    {
        mode_ = plan.mode;
        for (int axis = 0; axis < 3; ++axis)
            memory_[static_cast<size_t>(axis)].assign(
                static_cast<size_t>(plan.samples[axis]), std::numeric_limits<float>::quiet_NaN());
        return PULSEG_SUCCESS;
    }

    int prepare(int segment, int position, const pulseg_block_info &info, const pulseg_wave_region *slot)
    {
        prepared_.add(coll_, segment, position, info, slot);
        return PULSEG_SUCCESS;
    }

    /* Write samples into memory, counting a write over what the instance in
     * play reads: loading must leave that alone. */
    int load(const pulseg_wave_load &load)
    {
        if (load.axis < 0 || load.axis > 2 || load.offset < 0 || load.count < 0)
            return PULSEG_ERR_INDEX;
        const auto axis = static_cast<size_t>(load.axis);
        if (static_cast<size_t>(load.offset + load.count) > memory_[axis].size())
            return PULSEG_ERR_INDEX;
        const Span written{load.axis, load.offset, load.offset + load.count};
        overwrites_ += std::count_if(
            playing_.begin(), playing_.end(),
            [&written](const Span &read) { return overlap(written, read); });
        std::copy(load.samples, load.samples + load.count, memory_[axis].begin() + load.offset);
        loads_ += 1;
        return PULSEG_SUCCESS;
    }

    int set(const pulseg_playout_segment &segment, const pulseg_playout_block &block)
    {
        played_.add(segment, block);
        for (int axis = 0; axis < 3; ++axis)
        {
            unloaded_ += played_.read(memory_[static_cast<size_t>(axis)], block.wave, axis);
            if (block.wave && block.wave->offset[axis] >= 0)
                setting_.push_back(Span{axis, block.wave->offset[axis],
                                        block.wave->offset[axis] + block.wave->samples});
        }
        return PULSEG_SUCCESS;
    }

    int play()
    {
        playing_.swap(setting_);
        setting_.clear();
        instances_ += 1;
        return PULSEG_SUCCESS;
    }

    py::dict result() const
    {
        static const char *const modes[] = {"none", "resident", "streamed"};
        py::dict out;
        out["mode"] = modes[mode_];
        out["memory_samples"] = py::make_tuple(memory_[0].size(), memory_[1].size(), memory_[2].size());
        out["positions"] = prepared_.result();
        out["blocks"] = played_.result();
        out["instances"] = instances_;
        out["loads"] = loads_;
        out["overwrites"] = overwrites_;
        out["unloaded"] = unloaded_;
        return out;
    }

  private:
    const pulseg_collection *coll_;
    std::array<std::vector<float>, 3> memory_;
    std::vector<Span> playing_; /* what the instance in play reads */
    std::vector<Span> setting_; /* what the instance being set reads */
    Prepared prepared_;
    Played played_;
    int mode_ = PULSEG_WAVES_NONE;
    long instances_ = 0;
    long loads_ = 0;
    long overwrites_ = 0;
    long unloaded_ = 0;
};

/* The recorder, and what it raised: an exception cannot cross the C library. */
struct Session
{
    explicit Session(const pulseg_collection *coll) : recorder(coll) {}

    Recorder recorder;
    std::exception_ptr failure;
};

template <typename Call>
int guarded(void *ctx, Call &&call)
{
    Session &session = *static_cast<Session *>(ctx);
    try
    {
        return call(session.recorder);
    }
    catch (...)
    {
        session.failure = std::current_exception();
        return PULSEG_ERR_INVALID_ARGUMENT;
    }
}

pulseg_playout_backend backend_for(Session &session)
{
    pulseg_playout_backend backend;
    std::memset(&backend, 0, sizeof(backend));
    backend.ctx = &session;
    backend.reserve_waves = [](void *ctx, const pulseg_wave_plan *plan)
    { return guarded(ctx, [plan](Recorder &r) { return r.reserve(*plan); }); };
    backend.prepare_block =
        [](void *ctx, int segment, int position, const pulseg_block_info *info,
           const pulseg_wave_region *slot)
    {
        return guarded(
            ctx, [=](Recorder &r) { return r.prepare(segment, position, *info, slot); });
    };
    backend.load_wave = [](void *ctx, const pulseg_wave_load *load)
    { return guarded(ctx, [load](Recorder &r) { return r.load(*load); }); };
    backend.set_block =
        [](void *ctx, const pulseg_playout_segment *segment, const pulseg_playout_block *block)
    { return guarded(ctx, [=](Recorder &r) { return r.set(*segment, *block); }); };
    backend.play_instance = [](void *ctx, const pulseg_playout_segment *)
    { return guarded(ctx, [](Recorder &r) { return r.play(); }); };
    return backend;
}

} // namespace

py::dict record_playout(
    pulseg_collection *coll,
    const pulseg_wave_budget &budget,
    const pulseg_playout_options &options)
{
    Session session(coll);
    const pulseg_playout_backend backend = backend_for(session);
    pulseg_diagnostic diag = PULSEG_DIAGNOSTIC_INIT;
    int rc = pulseg_playout_prepare(coll, &budget, &backend, &diag);
    if (PULSEG_SUCCEEDED(rc))
        rc = pulseg_playout_scan(coll, &budget, &backend, &options, &diag);
    if (session.failure)
        std::rethrow_exception(session.failure);
    if (PULSEG_FAILED(rc))
        raise_diagnosed(rc, diag);
    return session.recorder.result();
}

} // namespace native
