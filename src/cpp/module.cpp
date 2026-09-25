/**
 * @file module.cpp
 * @brief The compiled extension, bound as `pulserver._ext`.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "ir/from_libraries.hpp"
#include "pulseg.h"
#include "pulseg_cache.h"
#include "pulseg_convert.h"
#include "pulseq.h"
#include "pulseq_file.h"

namespace py = pybind11;

namespace
{

struct CollectionFree
{
    void operator()(pulseg_collection *coll) const { pulseg_collection_free(coll); }
};

using Collection = std::unique_ptr<pulseg_collection, CollectionFree>;

pulseg_opts make_opts(
    float gamma_hz_per_t,
    float b0_t,
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
    pulseg_opts_init(
        &opts,
        gamma_hz_per_t,
        b0_t,
        rf_raster_us,
        grad_raster_us,
        adc_raster_us,
        block_raster_us);
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

void require(int code, const char *what)
{
    if (PULSEG_FAILED(code))
        throw std::invalid_argument(std::string(what) + " failed: error " + std::to_string(code));
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
        entry["tr_groups"] = tr_groups(coll, i);
        entry["rf"] = rf_statistics(coll, i, s.num_unique_rf);
        entry["readout_labels"] = readout_labels(coll, i, s);
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

template <typename T>
py::array_t<T> as_array(const std::vector<T> &values, std::vector<py::ssize_t> shape)
{
    py::array_t<T> out(shape);
    if (!values.empty())
        std::memcpy(out.mutable_data(), values.data(), values.size() * sizeof(T));
    return out;
}

/* A waveform the library allocated: freed with it, however the caller leaves. */
struct Waveform
{
    float *samples = nullptr;
    ~Waveform() { PULSEG_FREE(samples); }
};

/* The RF centre the cache records, from the block's start, in us. */
float recorded_rf_centre_us(
    const pulseg_collection *coll, int seg, int blk, const pulseg_block_info &b)
{
    const float isocentre = pulseg_get_rf_isocenter_us(coll, seg, blk);
    if (isocentre < 0.0f)
        return std::numeric_limits<float>::quiet_NaN();
    return isocentre - static_cast<float>(b.start_time_us);
}

/* Append one axis of the block at the cursor: its corners, timed from the
 * block's start, and the instance's amplitude times its normalised waveform.
 * A waveform on the gradient raster is sampled at the middle of each raster
 * interval; over the half intervals at its ends it holds its end values,
 * which keeps the area its samples give. */
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
    if (samples == 0)
        return;
    const float *t = time.samples;
    const float *w = shape.samples;
    const float delay = static_cast<float>(std::max(b.grad_delay_us[axis], 0));
    const bool centred =
        samples > 1 && t[0] > 0.0f && std::fabs(t[0] - 0.5f * (t[1] - t[0])) < 1e-3f;
    if (centred)
    {
        times.push_back(delay);
        values.push_back(amplitude * w[0]);
    }
    for (int i = 0; i < samples; ++i)
    {
        times.push_back(delay + t[i]);
        values.push_back(amplitude * w[i]);
    }
    if (centred)
    {
        times.push_back(delay + t[samples - 1] + 0.5f * (t[1] - t[0]));
        values.push_back(amplitude * w[samples - 1]);
    }
}

/* Append the phase modulation of the ADC the block at the cursor plays, one
 * phase per sample in radians; nothing when it carries none. */
void played_modulation(const pulseg_collection *coll, std::vector<float> &phases)
{
    Waveform phase;
    const int samples = pulseg_get_cursor_adc_phase_modulation(coll, &phase.samples);
    if (samples < 0)
        throw std::runtime_error("cannot read the phase modulation a readout plays");
    phases.insert(phases.end(), phase.samples, phase.samples + samples);
}

/* Every block the cursor plays, in play order, one entry per block in each
 * array; with waveforms, also the gradients, RF timing and ADC windows. */
py::dict play(pulseg_collection *coll, bool waveforms)
{
    std::vector<int> subsequence, segment, duration_us, adc, trid, norot, nopos, rf_use;
    std::vector<int> rf_channels;
    std::vector<float> rf_amp, rf_freq, rf_phase, adc_freq, adc_phase, gradient, rotation;
    std::vector<int> rf_delay_us, adc_delay_us, adc_dwell_ns, adc_samples;
    std::vector<float> rf_centre_us, grad_time, grad_value, modulation;
    std::vector<py::ssize_t> grad_span, modulation_span;
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
        const float amplitude[3] = {
            block.gx_amp_hz_per_m, block.gy_amp_hz_per_m, block.gz_amp_hz_per_m};
        gradient.insert(gradient.end(), amplitude, amplitude + 3);
        rotation.insert(rotation.end(), block.rotmat, block.rotmat + 9);
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
        adc_delay_us.push_back(block.adc_flag ? b.adc_delay_us : 0);
        adc_dwell_ns.push_back(window.dwell_ns);
        adc_samples.push_back(window.num_samples);
        if (!waveforms)
            continue;
        rf_centre_us.push_back(
            b.has_rf ? recorded_rf_centre_us(coll, info.segment_id, position, b)
                     : std::numeric_limits<float>::quiet_NaN());
        for (int axis = 0; axis < 3; ++axis)
        {
            grad_span.push_back(static_cast<py::ssize_t>(grad_time.size()));
            played_gradient(coll, axis, amplitude[axis], b, grad_time, grad_value);
            grad_span.push_back(static_cast<py::ssize_t>(grad_time.size()));
        }
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
    out["gradient_hz_per_m"] = as_array(gradient, {count, 3});
    out["rotation"] = as_array(rotation, {count, 3, 3});
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
        const auto corners = static_cast<py::ssize_t>(grad_time.size());
        out["rf_center_us"] = as_array(rf_centre_us, {count});
        out["gradient_time_us"] = as_array(grad_time, {corners});
        out["gradient_waveform_hz_per_m"] = as_array(grad_value, {corners});
        out["gradient_span"] = as_array(grad_span, {count, 3, 2});
        out["adc_phase_modulation_rad"] =
            as_array(modulation, {static_cast<py::ssize_t>(modulation.size())});
        out["adc_modulation_span"] = as_array(modulation_span, {count, 2});
    }
    return out;
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
           float gamma_hz_per_t,
           float b0_t,
           float rf_raster_us,
           float grad_raster_us,
           float adc_raster_us,
           float block_raster_us,
           int vendor,
           const std::array<int, 3> &label_column_map,
           const std::string &cache_ext)
        {
            const pulseg_opts opts = make_opts(
                gamma_hz_per_t,
                b0_t,
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
           float gamma_hz_per_t,
           float b0_t,
           float rf_raster_us,
           float grad_raster_us,
           float adc_raster_us,
           float block_raster_us,
           const std::array<int, 3> &label_column_map)
        {
            const pulseg_opts opts = make_opts(
                gamma_hz_per_t,
                b0_t,
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
        {
            Collection coll(pulseg_collection_alloc());
            if (!coll)
                throw std::bad_alloc();
            if (PULSEG_FAILED(pulseg_load_cache(coll.get(), cache_path.c_str(), source_size)))
                throw std::invalid_argument("cannot load the cache " + cache_path);
            return summarize(coll.get());
        },
        "The summary a written cache carries.");

    module.def(
        "play_cache",
        [](const std::string &cache_path, int source_size, bool waveforms)
        {
            Collection coll(pulseg_collection_alloc());
            if (!coll)
                throw std::bad_alloc();
            if (PULSEG_FAILED(pulseg_load_cache(coll.get(), cache_path.c_str(), source_size)))
                throw std::invalid_argument("cannot load the cache " + cache_path);
            return play(coll.get(), waveforms);
        },
        py::arg("cache_path"),
        py::arg("source_size"),
        py::arg("waveforms") = false,
        "Walk a written cache with the scanner's cursor, one entry per played block.");
}
