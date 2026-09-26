/**
 * @file module.cpp
 * @brief The compiled extension, bound as `pulserver._ext`.
 */

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "ir/from_libraries.hpp"
#include "native.hpp"
#include "playout.hpp"
#include "pulseg.h"
#include "pulseg_cache.h"
#include "pulseg_convert.h"
#include "pulseq.h"
#include "pulseq_file.h"

namespace py = pybind11;

namespace
{

using native::as_array;
using native::played_modulation;
using native::played_rf;
using native::recorded_rf_centre_us;
using native::require;
using native::Waveform;
using native::Waves;

struct CollectionFree
{
    void operator()(pulseg_collection *coll) const { pulseg_collection_free(coll); }
};

using Collection = std::unique_ptr<pulseg_collection, CollectionFree>;

pulseg_opts make_opts(
    float rf_raster_us,
    float grad_raster_us,
    float adc_raster_us,
    float block_raster_us,
    int vendor,
    const std::array<int, 3> &label_column_map,
    const std::string &cache_ext)
{
    pulseg_opts opts;
    std::memset(&opts, 0, sizeof(opts));
    pulseg_opts_init(&opts, rf_raster_us, grad_raster_us, adc_raster_us, block_raster_us);
    if (cache_ext.size() >= sizeof(opts.cache_ext))
        throw std::invalid_argument("cache extension '" + cache_ext + "' is too long");
    std::memcpy(opts.cache_ext, cache_ext.c_str(), cache_ext.size() + 1);
    opts.vendor = vendor;
    for (std::size_t i = 0; i < label_column_map.size(); ++i)
        opts.label_column_map[i] = label_column_map[i];
    return opts;
}

[[noreturn]] void raise_failure(int code, const pulseg_diagnostic &diag)
{
    char message[1024];
    pulseg_format_error(message, sizeof(message), code, &diag);
    throw std::invalid_argument(std::string(message) + " [error " + std::to_string(code) + "]");
}

/* The TRID-labelled groups of one subsequence, in first-seen order. */
py::list tr_groups(const pulseg_collection *coll, int subseq_idx)
{
    pulseg_tr_group *groups = NULL;
    const int count = pulseg_get_tr_groups(coll, &groups, subseq_idx);
    if (count < 0)
    {
        if (groups)
            PULSEG_FREE(groups);
        require(count, "TR groups");
    }
    py::list out;
    for (int i = 0; i < count; ++i)
    {
        py::dict entry;
        entry["trid"] = groups[i].trid;
        entry["num_instances"] = groups[i].num_instances;
        entry["one_instance_duration_us"] = groups[i].one_instance_duration_us;
        entry["total_duration_us"] = groups[i].total_duration_us;
        out.append(entry);
    }
    if (groups)
        PULSEG_FREE(groups);
    return out;
}

/* The label columns each readout of one subsequence records, in play order. */
py::list readout_labels(const pulseg_collection *coll, int subseq_idx, const pulseg_subseq_info &s)
{
    py::list out;
    std::vector<int> values((size_t)std::max(s.num_label_columns, 0));
    for (int i = 0; i < s.num_adc_occurrences; ++i)
    {
        require(pulseg_get_adc_label(coll, values.data(), subseq_idx, i), "readout labels");
        out.append(py::cast(values));
    }
    return out;
}

/* The flip angle and spectral statistics of each unique RF definition of one
 * subsequence. */
py::list rf_statistics(const pulseg_collection *coll, int subseq_idx, int num_unique_rf)
{
    py::list out;
    for (int i = 0; i < num_unique_rf; ++i)
    {
        pulseg_rf_stats stats;
        require(pulseg_get_rf_stats(coll, &stats, subseq_idx, i), "RF statistics");
        const int bands = stats.num_bands < PULSEG_MAX_BANDS ? stats.num_bands : PULSEG_MAX_BANDS;
        py::list offsets;
        for (int b = 0; b < bands; ++b)
            offsets.append(stats.band_freq_offsets_hz[b]);
        py::dict entry;
        entry["flip_angle_deg"] = stats.flip_angle_rad * 180.0 / M_PI;
        entry["bandwidth_hz"] = stats.bandwidth_hz;
        entry["num_bands"] = stats.num_bands;
        entry["band_freq_offsets_hz"] = offsets;
        entry["band_bandwidth_hz"] = stats.band_bandwidth_hz;
        entry["b1sq_integral_s"] = stats.total_b1sq_power;
        out.append(entry);
    }
    return out;
}

py::dict summarize(const pulseg_collection *coll)
{
    pulseg_collection_info info = PULSEG_COLLECTION_INFO_INIT;
    require(pulseg_get_collection_info(coll, &info), "collection info");

    py::list subsequences;
    for (int i = 0; i < info.num_subsequences; ++i)
    {
        pulseg_subseq_info s = PULSEG_SUBSEQ_INFO_INIT;
        require(pulseg_get_subseq_info(coll, &s, i), "subsequence info");
        py::dict entry;
        entry["tr_duration_us"] = s.tr_duration_us;
        entry["num_trs"] = s.num_trs;
        entry["tr_size"] = s.tr_size;
        entry["num_unique_adcs"] = s.num_unique_adcs;
        entry["num_unique_rf"] = s.num_unique_rf;
        entry["num_tr_instances"] = s.num_tr_instances;
        entry["rf_amplitude_variable"] = s.rf_amplitude_variable;
        entry["vop_sar_ratio"] = s.vop_sar_ratio;
        entry["vop_global_sar_ratio"] = s.vop_global_sar_ratio;
        entry["grad_raster_us"] = s.grad_raster_us;
        entry["tr_groups"] = tr_groups(coll, i);
        entry["rf"] = rf_statistics(coll, i, s.num_unique_rf);
        entry["readout_labels"] = readout_labels(coll, i, s);
        py::list waves;
        const int num_waves = pulseg_get_num_waves(coll, i);
        require(num_waves, "waves");
        for (int w = 0; w < num_waves; ++w)
        {
            int points = 0;
            float peak[3] = {0.0f, 0.0f, 0.0f};
            for (int axis = 0; axis < 3; ++axis)
                require(
                    pulseg_materialize_wave(
                        coll, i, w, axis, nullptr, nullptr, 0, &points, &peak[axis]),
                    "wave");
            py::dict wave;
            wave["points"] = points;
            wave["peak"] = py::make_tuple(peak[0], peak[1], peak[2]);
            waves.append(wave);
        }
        entry["waves"] = waves;
        subsequences.append(entry);
    }

    py::list segments;
    for (int i = 0; i < info.num_segments; ++i)
    {
        pulseg_segment_info g = PULSEG_SEGMENT_INFO_INIT;
        require(pulseg_get_segment_info(coll, &g, i), "segment info");
        py::dict entry;
        entry["duration_us"] = g.duration_us;
        entry["num_blocks"] = g.num_blocks;
        entry["start_block"] = g.start_block;
        entry["pure_delay"] = g.pure_delay;
        entry["has_trigger"] = g.has_trigger;
        entry["is_nav"] = g.is_nav;
        entry["rf_adc_gap_us"] = g.rf_adc_gap_us;
        entry["adc_adc_gap_us"] = g.adc_adc_gap_us;
        segments.append(entry);
    }

    py::dict result;
    result["num_subsequences"] = info.num_subsequences;
    result["num_segments"] = info.num_segments;
    result["max_adc_samples"] = info.max_adc_samples;
    result["total_readouts"] = info.total_readouts;
    result["total_duration_us"] = info.total_duration_us;
    result["subsequences"] = subsequences;
    result["segments"] = segments;
    return result;
}

/* Append one axis of the block at the cursor: its corners, timed from the
 * block's start, and the instance's amplitude times its normalised waveform.
 * The instance's shape is played in the waveform its segment position is
 * prepared with, so the two must hold as many samples. */
void played_gradient(
    const pulseg_collection *coll,
    int axis,
    float amplitude,
    const pulseg_block_info &b,
    std::vector<float> &times,
    std::vector<float> &values)
{
    Waveform shape, time;
    const int samples =
        pulseg_get_cursor_grad_waveform(coll, axis, &shape.samples, &time.samples);
    if (samples < 0)
        throw std::runtime_error("cannot read the gradient a block plays");
    if (samples != std::max(b.grad_num_samples[axis], 0))
        throw std::runtime_error(
            "a block plays a gradient of " + std::to_string(samples) +
            " samples at a segment position prepared for " +
            std::to_string(std::max(b.grad_num_samples[axis], 0)));
    if (samples == 0)
        return;
    native::append_event(
        time.samples, shape.samples, samples,
        static_cast<float>(std::max(b.grad_delay_us[axis], 0)), amplitude, times, values);
}

/* The amplitude a block plays each axis at: its gradient events', or, for a
 * block that plays a wave, the wave's, the rotation in it. */
std::array<float, 3> played_amplitudes(const pulseg_block_instance &block)
{
    if (block.wave_id >= 0)
        return {block.wave_amp_hz_per_m[0], block.wave_amp_hz_per_m[1], block.wave_amp_hz_per_m[2]};
    return {block.gx_amp_hz_per_m, block.gy_amp_hz_per_m, block.gz_amp_hz_per_m};
}

/* The gradient corners of every played block, in play order, with the start
 * and stop of each block's axes in them. */
struct PlayedGradients
{
    std::vector<float> time_us;
    std::vector<float> value;
    std::vector<py::ssize_t> span;

    void append(
        const pulseg_collection *coll,
        Waves &waves,
        int subsequence,
        const pulseg_block_instance &block,
        const pulseg_block_info &b,
        const std::array<float, 3> &amplitude)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            span.push_back(static_cast<py::ssize_t>(time_us.size()));
            if (block.wave_id >= 0)
                waves.append(
                    subsequence, block.wave_id, axis, amplitude[axis], b.wave_points, time_us,
                    value);
            else
                played_gradient(coll, axis, amplitude[axis], b, time_us, value);
            span.push_back(static_cast<py::ssize_t>(time_us.size()));
        }
    }
};

/* Every block the cursor plays, in play order, one entry per block in each
 * array; with waveforms, also the RF pulses and their timing, the gradients
 * and the ADC phase modulations. */
py::dict play(pulseg_collection *coll, bool waveforms)
{
    std::vector<int> subsequence, segment, duration_us, adc, trid, norot, nopos, rf_use;
    std::vector<int> rf_channels, rf_grad_constant, wave;
    std::vector<float> rf_amp, rf_freq, rf_phase, adc_freq, adc_phase, gradient;
    std::vector<float> rf_grad_level;
    std::vector<int> rf_delay_us, adc_delay_us, adc_dwell_ns, adc_samples;
    std::vector<float> rf_centre_us, rf_time, modulation;
    std::vector<std::complex<float>> rf_value;
    std::vector<py::ssize_t> rf_span, modulation_span;
    PlayedGradients gradients;
    Waves waves(coll);
    pulseg_cursor_reset(coll);
    pulseg_cursor_info info = PULSEG_CURSOR_INFO_INIT;
    int status;
    int position = 0;
    while ((status = pulseg_cursor_advance(coll, &info)) == PULSEG_CURSOR_BLOCK)
    {
        pulseg_block_instance block = PULSEG_BLOCK_INSTANCE_INIT;
        require(pulseg_get_block_instance(coll, &block), "block instance");
        position = info.segment_start ? 0 : position + 1;
        subsequence.push_back(info.subseq_idx);
        segment.push_back(info.segment_id);
        duration_us.push_back(block.duration_us);
        rf_amp.push_back(block.rf_amp_hz);
        rf_freq.push_back(block.rf_freq_hz);
        rf_phase.push_back(block.rf_phase_rad);
        rf_use.push_back(block.rf_use);
        const std::array<float, 3> amplitude = played_amplitudes(block);
        gradient.insert(gradient.end(), amplitude.begin(), amplitude.end());
        wave.push_back(block.wave_id);
        norot.push_back(block.norot_flag);
        nopos.push_back(block.nopos_flag);
        adc.push_back(block.adc_flag);
        adc_freq.push_back(block.adc_freq_hz);
        adc_phase.push_back(block.adc_phase_rad);
        trid.push_back(block.trid);

        pulseg_block_info b = PULSEG_BLOCK_INFO_INIT;
        require(pulseg_get_block_info(coll, &b, info.segment_id, position), "block info");
        pulseg_adc_def window = PULSEG_ADC_DEF_INIT;
        if (block.adc_flag)
            require(pulseg_get_adc_def(coll, &window, b.adc_def_id), "ADC definition");
        rf_delay_us.push_back(b.has_rf ? b.rf_delay_us : 0);
        rf_channels.push_back(b.has_rf ? b.rf_num_channels : 0);
        rf_grad_constant.push_back(b.rf_grad_constant);
        rf_grad_level.insert(rf_grad_level.end(), b.rf_grad_level, b.rf_grad_level + 3);
        adc_delay_us.push_back(block.adc_flag ? b.adc_delay_us : 0);
        adc_dwell_ns.push_back(window.dwell_ns);
        adc_samples.push_back(window.num_samples);
        if (!waveforms)
            continue;
        rf_centre_us.push_back(
            b.has_rf ? recorded_rf_centre_us(coll, info.segment_id, position, b)
                     : std::numeric_limits<float>::quiet_NaN());
        rf_span.push_back(static_cast<py::ssize_t>(rf_time.size()));
        if (b.has_rf)
            played_rf(coll, info.segment_id, position, block.rf_amp_hz, b, rf_time, rf_value);
        rf_span.push_back(static_cast<py::ssize_t>(rf_time.size()));
        gradients.append(coll, waves, info.subseq_idx, block, b, amplitude);
        modulation_span.push_back(static_cast<py::ssize_t>(modulation.size()));
        played_modulation(coll, modulation);
        modulation_span.push_back(static_cast<py::ssize_t>(modulation.size()));
    }
    require(status, "cursor");

    const auto count = static_cast<py::ssize_t>(duration_us.size());
    py::dict out;
    out["subsequence"] = as_array(subsequence, {count});
    out["segment"] = as_array(segment, {count});
    out["duration_us"] = as_array(duration_us, {count});
    out["rf_amp_hz"] = as_array(rf_amp, {count});
    out["rf_freq_hz"] = as_array(rf_freq, {count});
    out["rf_phase_rad"] = as_array(rf_phase, {count});
    out["rf_use"] = as_array(rf_use, {count});
    out["rf_delay_us"] = as_array(rf_delay_us, {count});
    out["rf_channels"] = as_array(rf_channels, {count});
    out["rf_grad_constant"] = as_array(rf_grad_constant, {count});
    out["rf_grad_level"] = as_array(rf_grad_level, {count, 3});
    out["gradient_hz_per_m"] = as_array(gradient, {count, 3});
    out["wave"] = as_array(wave, {count});
    out["norot"] = as_array(norot, {count});
    out["nopos"] = as_array(nopos, {count});
    out["adc"] = as_array(adc, {count});
    out["adc_freq_hz"] = as_array(adc_freq, {count});
    out["adc_phase_rad"] = as_array(adc_phase, {count});
    out["adc_delay_us"] = as_array(adc_delay_us, {count});
    out["adc_dwell_ns"] = as_array(adc_dwell_ns, {count});
    out["adc_samples"] = as_array(adc_samples, {count});
    out["trid"] = as_array(trid, {count});
    if (waveforms)
    {
        const auto corners = static_cast<py::ssize_t>(gradients.time_us.size());
        out["rf_center_us"] = as_array(rf_centre_us, {count});
        const auto rf_samples = static_cast<py::ssize_t>(rf_time.size());
        out["rf_time_us"] = as_array(rf_time, {rf_samples});
        out["rf_waveform_hz"] = as_array(rf_value, {rf_samples});
        out["rf_span"] = as_array(rf_span, {count, 2});
        out["gradient_time_us"] = as_array(gradients.time_us, {corners});
        out["gradient_waveform_hz_per_m"] = as_array(gradients.value, {corners});
        out["gradient_span"] = as_array(gradients.span, {count, 3, 2});
        out["adc_phase_modulation_rad"] =
            as_array(modulation, {static_cast<py::ssize_t>(modulation.size())});
        out["adc_modulation_span"] = as_array(modulation_span, {count, 2});
    }
    return out;
}

py::dict region_dict(const pulseg_wave_region &region)
{
    py::dict out;
    out["offset"] = py::make_tuple(region.offset[0], region.offset[1], region.offset[2]);
    out["samples"] = region.samples;
    out["start_us"] = region.start_us;
    return out;
}

/* A wave plan, released with it. */
struct WavePlan
{
    pulseg_wave_plan plan = PULSEG_WAVE_PLAN_INIT;

    WavePlan() = default;
    WavePlan(const WavePlan &) = delete;
    WavePlan &operator=(const WavePlan &) = delete;
    ~WavePlan() { pulseg_free_wave_plan(&plan); }
};

/* The waveform memory a playout on @p budget gives the waves. */
py::dict plan_waves(const pulseg_collection *coll, const pulseg_wave_budget &budget)
{
    WavePlan planned;
    const pulseg_wave_plan &plan = planned.plan;
    pulseg_diagnostic diag = PULSEG_DIAGNOSTIC_INIT;
    const int rc = pulseg_plan_waves(coll, &budget, &planned.plan, &diag);
    if (PULSEG_FAILED(rc))
        native::raise_diagnosed(rc, diag);

    static const char *const modes[] = {"none", "resident", "streamed"};
    py::dict out;
    out["mode"] = modes[plan.mode];
    out["samples"] = py::make_tuple(plan.samples[0], plan.samples[1], plan.samples[2]);
    out["resident_samples"] = py::make_tuple(
        plan.resident_samples[0], plan.resident_samples[1], plan.resident_samples[2]);
    out["streamed_samples"] = py::make_tuple(
        plan.streamed_samples[0], plan.streamed_samples[1], plan.streamed_samples[2]);
    py::list waves;
    for (int s = 0; s < plan.num_subsequences; ++s)
    {
        py::list regions;
        for (int w = 0; w < plan.num_waves[s]; ++w)
            regions.append(region_dict(plan.waves[s][w]));
        waves.append(regions);
    }
    out["waves"] = waves;
    py::list slots;
    for (int g = 0; g < plan.num_segments; ++g)
    {
        py::list positions;
        for (int b = 0; b < plan.num_positions[g]; ++b)
            positions.append(py::make_tuple(
                region_dict(plan.slots[g][2 * b]), region_dict(plan.slots[g][2 * b + 1])));
        slots.append(positions);
    }
    out["slots"] = slots;
    out["loading_checked"] = plan.loading_checked != 0;
    out["least_spare_us"] = plan.least_spare_us;
    out["tightest"] = plan.tightest_subseq < 0
        ? py::object(py::none())
        : py::object(py::make_tuple(plan.tightest_subseq, plan.tightest_position));
    return out;
}

/* A playout's waveform memory, gradient raster, load rate and headroom, as
 * pulserver.ir.WaveBudget holds them. */
using Budget = std::tuple<long, float, float, float>;

/* The budget @p given, or, without one, every wave held at once on the
 * gradient raster of the chain's first file. */
pulseg_wave_budget budget_for(const pulseg_collection *coll, const std::optional<Budget> &given)
{
    pulseg_wave_budget b = PULSEG_WAVE_BUDGET_INIT;
    if (given)
    {
        b.max_samples = std::get<0>(*given);
        b.raster_us = std::get<1>(*given);
        b.load_us_per_sample = std::get<2>(*given);
        b.headroom = std::get<3>(*given);
        return b;
    }
    pulseg_subseq_info first = PULSEG_SUBSEQ_INFO_INIT;
    require(pulseg_get_subseq_info(coll, &first, 0), "subsequence info");
    b.max_samples = std::numeric_limits<long>::max();
    b.raster_us = first.grad_raster_us;
    return b;
}

/* A corner-point stream, released with it. */
struct CornerPoints
{
    pulseg_corner_point_stream stream = PULSEG_CORNER_POINT_STREAM_INIT;

    CornerPoints() = default;
    CornerPoints(const CornerPoints &) = delete;
    CornerPoints &operator=(const CornerPoints &) = delete;
    ~CornerPoints() { pulseg_corner_point_stream_free(&stream); }
};

/* The gradients of subsequence @p subsequence's heaviest repetition, and,
 * with a positive @p raster_us, its samples on that raster. */
py::dict repetition_gradients(const pulseg_collection *coll, int subsequence, float raster_us)
{
    CornerPoints corners;
    const pulseg_corner_point_stream &s = corners.stream;
    pulseg_diagnostic diag = PULSEG_DIAGNOSTIC_INIT;
    const int rc = pulseg_get_tr_corner_points(coll, &corners.stream, &diag, subsequence);
    if (PULSEG_FAILED(rc))
        native::raise_diagnosed(rc, diag);

    const auto n = static_cast<py::ssize_t>(s.num_points);
    std::vector<float> gradient;
    gradient.reserve(static_cast<size_t>(3 * n));
    for (py::ssize_t i = 0; i < n; ++i)
    {
        gradient.push_back(s.gx_hz_per_m[i]);
        gradient.push_back(s.gy_hz_per_m[i]);
        gradient.push_back(s.gz_hz_per_m[i]);
    }
    py::dict out;
    out["first_position"] = s.first_position;
    out["duration_us"] = s.duration_us;
    out["energy"] = s.energy;
    out["time_us"] = as_array(std::vector<float>(s.time_us, s.time_us + n), {n});
    out["gradient_hz_per_m"] = as_array(gradient, {n, 3});
    if (raster_us <= 0.0f)
        return out;
    const auto m = static_cast<long>(std::ceil(s.duration_us / raster_us - 1e-3f));
    std::vector<float> samples(static_cast<size_t>(3 * (m > 0 ? m : 0)));
    for (int axis = 0; axis < 3 && m > 0; ++axis)
        require(
            pulseg_sample_corner_points(
                &s, axis, raster_us, m, samples.data() + static_cast<size_t>(axis * m)),
            "repetition sampling");
    out["samples_hz_per_m"] = as_array(samples, {3, static_cast<py::ssize_t>(m > 0 ? m : 0)});
    return out;
}

Collection load(const std::string &cache_path, int source_size)
{
    Collection coll(pulseg_collection_alloc());
    if (!coll)
        throw std::bad_alloc();
    if (PULSEG_FAILED(pulseg_load_cache(coll.get(), cache_path.c_str(), source_size)))
        throw std::invalid_argument("cannot load the cache " + cache_path);
    return coll;
}

/* Convert a chain of sequences, each given as the libraries it was read into. */
Collection convert(const py::list &chain, const pulseg_opts &opts)
{
    const int count = static_cast<int>(chain.size());
    if (count < 1)
        throw std::invalid_argument("a chain holds at least one subsequence");

    std::vector<pulseq_file> files((size_t)count);
    for (int i = 0; i < count; ++i)
        pulseq_file_init(&files[(size_t)i], nullptr);
    auto release = [&files]()
    {
        for (auto &file : files)
            pulseq_file_free(&file);
    };
    try
    {
        for (int i = 0; i < count; ++i)
            pulserver::build_pulseq_file(files[(size_t)i], chain[(size_t)i].cast<py::dict>());
    }
    catch (...)
    {
        release();
        throw;
    }

    Collection coll(pulseg_collection_alloc());
    if (!coll)
    {
        release();
        throw std::bad_alloc();
    }
    pulseg_diagnostic diag = PULSEG_DIAGNOSTIC_INIT;
    const int converted =
        pulseg_convert_collection(coll.get(), &diag, files.data(), count, &opts, 1);
    release();
    if (converted != count)
        raise_failure(diag.code, diag);
    return coll;
}

} // namespace

PYBIND11_MODULE(_ext, module)
{
    module.doc() = "Compiled scanner IR conversion for pulserver";

    module.def(
        "convert_libraries",
        [](const py::list &chain,
           const std::string &seq_path,
           float rf_raster_us,
           float grad_raster_us,
           float adc_raster_us,
           float block_raster_us,
           int vendor,
           const std::array<int, 3> &label_column_map,
           const std::string &cache_ext)
        {
            const pulseg_opts opts = make_opts(
                rf_raster_us,
                grad_raster_us,
                adc_raster_us,
                block_raster_us,
                vendor,
                label_column_map,
                cache_ext);
            const Collection coll = convert(chain, opts);
            if (PULSEG_FAILED(pulseg_save_cache(coll.get(), seq_path.c_str(), &opts)))
                throw std::invalid_argument("cannot write the cache beside " + seq_path);
        },
        "Segment a chain read into libraries and write its IR cache beside a sequence file.");

    module.def(
        "summary_from_libraries",
        [](const py::list &chain,
           float rf_raster_us,
           float grad_raster_us,
           float adc_raster_us,
           float block_raster_us,
           const std::array<int, 3> &label_column_map)
        {
            const pulseg_opts opts = make_opts(
                rf_raster_us,
                grad_raster_us,
                adc_raster_us,
                block_raster_us,
                0,
                label_column_map,
                PULSEG_CACHE_EXT_DEFAULT);
            return summarize(convert(chain, opts).get());
        },
        "Segment a chain read into libraries and return its summary, writing no cache.");

    module.def(
        "summary_from_cache",
        [](const std::string &cache_path, int source_size)
        { return summarize(load(cache_path, source_size).get()); },
        "The summary a written cache carries.");

    module.def(
        "play_cache",
        [](const std::string &cache_path, int source_size, bool waveforms)
        { return play(load(cache_path, source_size).get(), waveforms); },
        py::arg("cache_path"),
        py::arg("source_size"),
        py::arg("waveforms") = false,
        "Walk a written cache with the scanner's cursor, one entry per played block.");

    module.def(
        "plan_waves_from_cache",
        [](const std::string &cache_path,
           int source_size,
           long max_samples,
           float raster_us,
           float load_us_per_sample,
           float headroom)
        {
            pulseg_wave_budget budget = PULSEG_WAVE_BUDGET_INIT;
            budget.max_samples = max_samples;
            budget.raster_us = raster_us;
            budget.load_us_per_sample = load_us_per_sample;
            budget.headroom = headroom;
            return plan_waves(load(cache_path, source_size).get(), budget);
        },
        "Lay out a written cache's waves in a playout's waveform memory.");

    module.def(
        "playout_from_cache",
        [](const std::string &cache_path,
           int source_size,
           const std::optional<Budget> &budget,
           const std::array<int, 2> &prescan,
           bool waveforms)
        {
            const Collection coll = load(cache_path, source_size);
            pulseg_playout_options options = PULSEG_PLAYOUT_OPTIONS_INIT;
            options.prescan_subsequence = prescan[0];
            options.prescan_readouts = prescan[1];
            return native::record_playout(
                coll.get(), budget_for(coll.get(), budget), options, waveforms);
        },
        "Play a written cache's two stages over a backend that records them.");

    module.def(
        "repetition_gradients_from_cache",
        [](const std::string &cache_path, int source_size, int subsequence, float raster_us)
        { return repetition_gradients(load(cache_path, source_size).get(), subsequence, raster_us); },
        "The gradients of a written cache's heaviest repetition of one subsequence.");

    module.def(
        "sample_wave_from_cache",
        [](const std::string &cache_path,
           int source_size,
           int subsequence,
           int wave,
           int axis,
           float start_us,
           float raster_us,
           long samples)
        {
            const Collection coll = load(cache_path, source_size);
            std::vector<float> values(static_cast<size_t>(samples > 0 ? samples : 0));
            require(
                pulseg_sample_wave(
                    coll.get(), subsequence, wave, axis, start_us, raster_us, samples,
                    values.data()),
                "wave sampling");
            return as_array(values, {static_cast<py::ssize_t>(values.size())});
        },
        "One axis of a written cache's wave on a playout's raster.");
}
