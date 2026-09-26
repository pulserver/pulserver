/**
 * @file playout.cpp
 * @brief A playout backend that plays nothing and records what both stages
 *        hand it, with a model of its waveform memory.
 */

#include "playout.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
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

/* What the first stage prepares at one segment position: where it plays no
 * wave, its gradient events at unit amplitude, as corners from the block's
 * start; where it does, the points its longest wave takes and the slots its
 * waves play from; and its RF pulse at unit amplitude and its readout. */
struct Position
{
    std::array<std::vector<float>, 3> time_us, shape;
    int wave_points = 0;
    std::vector<RfShape> rf;
    int rf_channels = 0;
    int rf_delay_us = 0;
    float rf_centre_us = std::numeric_limits<float>::quiet_NaN();
    int adc_def = -1;
    int adc_delay_us = 0;
    std::array<pulseg_wave_region, 2> slot{};
    bool waves = false;
};

void prepare_event(
    const pulseg_collection *coll,
    int segment,
    int position,
    int axis,
    const pulseg_block_info &info,
    Position &p)
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
    const auto a = static_cast<size_t>(axis);
    append_event(
        time.samples, shape.samples[0], samples,
        static_cast<float>(std::max(info.grad_delay_us[axis], 0)), 1.0f, p.time_us[a],
        p.shape[a]);
}

Position prepare_position(
    const pulseg_collection *coll,
    int segment,
    int position,
    const pulseg_block_info &info,
    const pulseg_wave_region *slot)
{
    Position p;
    for (int axis = 0; axis < 3 && !slot; ++axis)
        prepare_event(coll, segment, position, axis, info, p);
    if (slot)
    {
        p.slot = {slot[0], slot[1]};
        p.waves = true;
        p.wave_points = info.wave_points;
    }
    if (info.has_rf)
    {
        p.rf.emplace_back(coll, segment, position, info);
        p.rf_channels = info.rf_num_channels;
        p.rf_delay_us = info.rf_delay_us;
        p.rf_centre_us = recorded_rf_centre_us(coll, segment, position, info);
    }
    if (info.has_adc)
    {
        p.adc_def = info.adc_def_id;
        p.adc_delay_us = info.adc_delay_us;
    }
    return p;
}

/* One row per block, a column per quantity, each named as
 * pulserver.ir.playout returns it; the (blocks, 3) ones kept apart. */
class Columns
{
  public:
    template <typename T>
    void put(const std::string &name, T value)
    {
        column<T>(name).push_back(value);
    }

    template <typename T>
    void put3(const std::string &name, T x, T y, T z)
    {
        std::vector<T> &c = column<T>(name + "\n3");
        c.push_back(x);
        c.push_back(y);
        c.push_back(z);
    }

    void into(py::dict &out) const
    {
        fill(out, ints_);
        fill(out, longs_);
        fill(out, floats_);
    }

  private:
    template <typename T>
    std::vector<T> &column(const std::string &name);

    template <typename T>
    static void fill(py::dict &out, const std::map<std::string, std::vector<T>> &columns)
    {
        for (const auto &c : columns)
        {
            const auto split = c.first.find('\n');
            const auto n = static_cast<py::ssize_t>(c.second.size());
            if (split == std::string::npos)
                out[c.first.c_str()] = as_array(c.second, {n});
            else
                out[c.first.substr(0, split).c_str()] = as_array(c.second, {n / 3, 3});
        }
    }

    std::map<std::string, std::vector<int>> ints_;
    std::map<std::string, std::vector<long>> longs_;
    std::map<std::string, std::vector<float>> floats_;
};

template <>
std::vector<int> &Columns::column<int>(const std::string &name)
{
    return ints_[name];
}

template <>
std::vector<long> &Columns::column<long>(const std::string &name)
{
    return longs_[name];
}

template <>
std::vector<float> &Columns::column<float>(const std::string &name)
{
    return floats_[name];
}

/* The waveforms each block plays: its gradients as corners from its start,
 * its RF pulse and its readout's phase modulation, each with the start and
 * stop of the block's part.  A wave plays as the IR defines it, linear
 * between its points: how a playout's hardware plays the samples it loads is
 * the playout's. */
class Waveforms
{
  public:
    void gradients(
        const Position &p,
        int subsequence,
        const pulseg_block_instance &b,
        Waves &waves)
    {
        const float amplitude[] = {b.gx_amp_hz_per_m, b.gy_amp_hz_per_m, b.gz_amp_hz_per_m};
        for (int axis = 0; axis < 3; ++axis)
        {
            const auto a = static_cast<size_t>(axis);
            gradient_span_.push_back(static_cast<py::ssize_t>(gradient_time_.size()));
            if (b.wave_id >= 0)
                waves.append(
                    subsequence, b.wave_id, axis, b.wave_amp_hz_per_m[axis], p.wave_points,
                    gradient_time_, gradient_value_);
            else
                event(p.time_us[a], p.shape[a], amplitude[axis]);
            gradient_span_.push_back(static_cast<py::ssize_t>(gradient_time_.size()));
        }
    }

    void rf(const Position &p, float amplitude)
    {
        rf_span_.push_back(static_cast<py::ssize_t>(rf_time_.size()));
        for (const RfShape &pulse : p.rf)
            pulse.play(amplitude, rf_time_, rf_value_);
        rf_span_.push_back(static_cast<py::ssize_t>(rf_time_.size()));
    }

    void modulation(const pulseg_collection *coll)
    {
        modulation_span_.push_back(static_cast<py::ssize_t>(modulation_.size()));
        played_modulation(coll, modulation_);
        modulation_span_.push_back(static_cast<py::ssize_t>(modulation_.size()));
    }

    void into(py::dict &out, py::ssize_t n) const
    {
        const auto corners = static_cast<py::ssize_t>(gradient_time_.size());
        const auto rf_samples = static_cast<py::ssize_t>(rf_time_.size());
        out["gradient_time_us"] = as_array(gradient_time_, {corners});
        out["gradient_waveform_hz_per_m"] = as_array(gradient_value_, {corners});
        out["gradient_span"] = as_array(gradient_span_, {n, 3, 2});
        out["rf_time_us"] = as_array(rf_time_, {rf_samples});
        out["rf_waveform_hz"] = as_array(rf_value_, {rf_samples});
        out["rf_span"] = as_array(rf_span_, {n, 2});
        out["adc_phase_modulation_rad"] =
            as_array(modulation_, {static_cast<py::ssize_t>(modulation_.size())});
        out["adc_modulation_span"] = as_array(modulation_span_, {n, 2});
    }

  private:
    void event(const std::vector<float> &time_us, const std::vector<float> &shape, float amplitude)
    {
        gradient_time_.insert(gradient_time_.end(), time_us.begin(), time_us.end());
        std::transform(
            shape.begin(), shape.end(), std::back_inserter(gradient_value_),
            [amplitude](float s) { return amplitude * s; });
    }

    std::vector<float> gradient_time_, gradient_value_, rf_time_, modulation_;
    std::vector<std::complex<float>> rf_value_;
    std::vector<py::ssize_t> gradient_span_, rf_span_, modulation_span_;
};

/* The registers the second stage sets on a block. */
void put_registers(
    Columns &c,
    const pulseg_playout_segment &segment,
    const pulseg_playout_block &block)
{
    const pulseg_block_instance &b = block.instance;
    c.put("subsequence", segment.subsequence);
    c.put("segment", segment.segment);
    c.put("position", block.position);
    c.put("instance", segment.instance);
    c.put("half", segment.half);
    c.put("rotate", segment.rotate);
    c.put("await_trigger", segment.await_trigger);
    c.put("first_position", segment.first_position);
    c.put("duration_us", b.duration_us);
    c.put("rf_amp_hz", b.rf_amp_hz);
    c.put("rf_phase_rad", b.rf_phase_rad);
    c.put("rf_freq_hz", b.rf_freq_hz);
    c.put("rf_shim", b.rf_shim_id);
    c.put("rf_use", b.rf_use);
    c.put("adc", b.adc_flag);
    c.put("adc_freq_hz", b.adc_freq_hz);
    c.put("adc_phase_rad", b.adc_phase_rad);
    c.put("digitalout", b.digitalout_flag);
    c.put("wave", b.wave_id);
    c.put3("gradient_hz_per_m", b.gx_amp_hz_per_m, b.gy_amp_hz_per_m, b.gz_amp_hz_per_m);
    c.put3("gradient_variable", b.gx_variable, b.gy_variable, b.gz_variable);
    c.put3(
        "wave_amp_hz_per_m", b.wave_amp_hz_per_m[0], b.wave_amp_hz_per_m[1],
        b.wave_amp_hz_per_m[2]);
}

/* Where the block's wave plays from; offsets -1 and no samples without one. */
void put_region(Columns &c, const pulseg_wave_region *region)
{
    const long none = -1;
    c.put3(
        "wave_offset", region ? region->offset[0] : none, region ? region->offset[1] : none,
        region ? region->offset[2] : none);
    c.put("wave_samples", region ? region->samples : 0L);
    c.put("wave_start_us", region ? region->start_us : 0.0f);
}

/* What the block's position prepares for its RF pulse and readout; the
 * readout's window only where the block acquires. */
void put_prepared(
    Columns &c,
    const pulseg_collection *coll,
    const Position &p,
    const pulseg_block_instance &b)
{
    pulseg_adc_def window = PULSEG_ADC_DEF_INIT;
    if (b.adc_flag && p.adc_def >= 0)
        require(pulseg_get_adc_def(coll, &window, p.adc_def), "ADC definition");
    c.put("rf_delay_us", p.rf_delay_us);
    c.put("rf_channels", p.rf_channels);
    c.put("rf_center_us", p.rf_centre_us);
    c.put("adc_delay_us", b.adc_flag ? p.adc_delay_us : 0);
    c.put("adc_dwell_ns", window.dwell_ns);
    c.put("adc_samples", window.num_samples);
}

/* The prepared positions as pulserver.ir.playout returns them, one row per
 * position in segment order. */
class PreparedRows
{
  public:
    void add(int segment, int position, const Position &p)
    {
        segment_.push_back(segment);
        position_.push_back(position);
        for (size_t a = 0; a < 3; ++a)
        {
            span_.push_back(static_cast<py::ssize_t>(time_us_.size()));
            time_us_.insert(time_us_.end(), p.time_us[a].begin(), p.time_us[a].end());
            shape_.insert(shape_.end(), p.shape[a].begin(), p.shape[a].end());
            span_.push_back(static_cast<py::ssize_t>(time_us_.size()));
        }
        for (const pulseg_wave_region &slot : p.slot)
        {
            for (int a = 0; a < 3; ++a)
                slot_offset_.push_back(p.waves ? slot.offset[a] : -1);
            slot_samples_.push_back(p.waves ? slot.samples : 0);
            slot_start_us_.push_back(p.waves ? slot.start_us : 0.0f);
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
    std::vector<int> segment_, position_;
    std::vector<float> time_us_, shape_, slot_start_us_;
    std::vector<py::ssize_t> span_;
    std::vector<long> slot_offset_, slot_samples_;
};

py::dict positions_dict(const std::map<std::pair<int, int>, Position> &positions)
{
    PreparedRows rows;
    for (const auto &entry : positions)
        rows.add(entry.first.first, entry.first.second, entry.second);
    return rows.result();
}

/* The backend: a model of the waveform memory, what the first stage
 * prepares, and what the second sets and plays. */
class Recorder
{
  public:
    Recorder(const pulseg_collection *coll, bool waveforms)
        : coll_(coll), waveforms_(waveforms), waves_(coll)
    {
    }

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
        positions_[{segment, position}] = prepare_position(coll_, segment, position, info, slot);
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
        const auto found = positions_.find({segment.segment, block.position});
        if (found == positions_.end())
            return PULSEG_ERR_INDEX;
        put_registers(columns_, segment, block);
        put_region(columns_, block.wave);
        read(block.wave);
        blocks_ += 1;
        if (!waveforms_)
            return PULSEG_SUCCESS;
        put_prepared(columns_, coll_, found->second, block.instance);
        waveforms_played_.gradients(found->second, segment.subsequence, block.instance, waves_);
        waveforms_played_.rf(found->second, block.instance.rf_amp_hz);
        waveforms_played_.modulation(coll_);
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
        py::dict blocks;
        columns_.into(blocks);
        blocks["wave_read"] = as_array(read_, {static_cast<py::ssize_t>(read_.size())});
        blocks["wave_read_span"] = as_array(read_span_, {blocks_, 3, 2});
        if (waveforms_)
            waveforms_played_.into(blocks, blocks_);
        py::dict out;
        out["mode"] = modes[mode_];
        out["memory_samples"] =
            py::make_tuple(memory_[0].size(), memory_[1].size(), memory_[2].size());
        out["positions"] = positions_dict(positions_);
        out["blocks"] = blocks;
        out["instances"] = instances_;
        out["loads"] = loads_;
        out["overwrites"] = overwrites_;
        out["unloaded"] = unloaded_;
        return out;
    }

  private:
    /* Keep what each axis of the block's wave reads from memory, counting
     * samples nothing was loaded into. */
    void read(const pulseg_wave_region *region)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            read_span_.push_back(static_cast<py::ssize_t>(read_.size()));
            if (region && region->offset[axis] >= 0)
            {
                const auto from = memory_[static_cast<size_t>(axis)].begin() + region->offset[axis];
                read_.insert(read_.end(), from, from + region->samples);
                unloaded_ += std::count_if(
                    from, from + region->samples, [](float v) { return std::isnan(v); });
                setting_.push_back(
                    Span{axis, region->offset[axis], region->offset[axis] + region->samples});
            }
            read_span_.push_back(static_cast<py::ssize_t>(read_.size()));
        }
    }

    const pulseg_collection *coll_;
    bool waveforms_;
    Waves waves_;
    std::array<std::vector<float>, 3> memory_;
    std::vector<Span> playing_; /* what the instance in play reads */
    std::vector<Span> setting_; /* what the instance being set reads */
    std::map<std::pair<int, int>, Position> positions_;
    Columns columns_;
    Waveforms waveforms_played_;
    std::vector<float> read_;
    std::vector<py::ssize_t> read_span_;
    py::ssize_t blocks_ = 0;
    int mode_ = PULSEG_WAVES_NONE;
    long instances_ = 0;
    long loads_ = 0;
    long overwrites_ = 0;
    long unloaded_ = 0;
};

/* The recorder, and what it raised: an exception cannot cross the C library. */
struct Session
{
    Session(const pulseg_collection *coll, bool waveforms) : recorder(coll, waveforms) {}

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
    const pulseg_playout_options &options,
    bool waveforms)
{
    Session session(coll, waveforms);
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
