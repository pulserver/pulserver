/**
 * @file pulseg.cpp
 * @brief Thin pybind11 binding for pulseg C++11 interface.
 *
 * All docstrings belong in the Python wrapper layer.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "bindings.hpp"

#include "pulseg.hpp"
#include "pulseg_pns_models.h"
#include "pulseg.h"
#include "pulseg_types.h"

namespace py = pybind11;

// ─── Thin holder for the C++ Collection ─────────────────────────────

static void apply_label_column_map(pulseg::Opts& opts, const std::vector<int>& map)
{
    if (map.empty())
        return;
    if (map.size() != 3)
        throw std::invalid_argument("label_column_map must have exactly 3 entries");
    for (int i = 0; i < 3; ++i)
        opts.label_column_map[i] = map[static_cast<size_t>(i)];
}

class _PulseqCollection
{
public:
    _PulseqCollection(
        const py::list& seq_bytes_list,
        float gamma,
        float B0,
        float max_grad,
        float max_slew,
        float rf_raster_time,
        float grad_raster_time,
        float adc_raster_time,
        float block_duration_raster,
        bool parse_labels,
        int num_averages,
        const std::vector<int>& label_column_map)
    {
        pulseg::Opts opts;
        opts.gamma_hz_per_t = gamma;
        opts.b0_t = B0;
        opts.max_grad_hz_per_m = max_grad;
        opts.max_slew_hz_per_m_per_s = max_slew;
        opts.rf_raster_us = rf_raster_time * 1e6f;
        opts.grad_raster_us = grad_raster_time * 1e6f;
        opts.adc_raster_us = adc_raster_time * 1e6f;
        opts.block_raster_us = block_duration_raster * 1e6f;
        apply_label_column_map(opts, label_column_map);

        int n = static_cast<int>(seq_bytes_list.size());
        std::vector<std::string> buffers(n);
        std::vector<const char*> buf_ptrs(n);
        std::vector<int> buf_sizes(n);
        for (int i = 0; i < n; ++i)
        {
            buffers[i] = seq_bytes_list[i].cast<py::bytes>();
            buf_ptrs[i] = buffers[i].data();
            buf_sizes[i] = static_cast<int>(buffers[i].size());
        }

        coll_ = std::unique_ptr<pulseg::Collection>(new pulseg::Collection(
            buf_ptrs.data(),
            buf_sizes.data(),
            n,
            opts,
            parse_labels,
            num_averages));
        source_size_ = (n > 0) ? buf_sizes[0] : 0;
    }

    /** Load from a .seq file path, using the .pge cache when present. */
    _PulseqCollection(
        const std::string& file_path,
        float gamma,
        float B0,
        float max_grad,
        float max_slew,
        float rf_raster_time,
        float grad_raster_time,
        float adc_raster_time,
        float block_duration_raster,
        bool parse_labels,
        int num_averages,
        const std::vector<int>& label_column_map)
    {
        pulseg::Opts opts;
        opts.gamma_hz_per_t = gamma;
        opts.b0_t = B0;
        opts.max_grad_hz_per_m = max_grad;
        opts.max_slew_hz_per_m_per_s = max_slew;
        opts.rf_raster_us = rf_raster_time * 1e6f;
        opts.grad_raster_us = grad_raster_time * 1e6f;
        opts.adc_raster_us = adc_raster_time * 1e6f;
        opts.block_raster_us = block_duration_raster * 1e6f;
        apply_label_column_map(opts, label_column_map);

        coll_ = std::unique_ptr<pulseg::Collection>(new pulseg::Collection(
            file_path.c_str(),
            opts,
            /*cache_binary=*/true,
            /*verify_signature=*/false,
            parse_labels,
            num_averages));
        source_size_ = 0;
    }

    pulseg::Collection& coll()
    {
        return *coll_;
    }
    const pulseg::Collection& coll() const
    {
        return *coll_;
    }
    int source_size() const
    {
        return source_size_;
    }

private:
    std::unique_ptr<pulseg::Collection> coll_;
    int source_size_ = 0;
};

// ─── Thin conversion functions ──────────────────────────────────────

static py::dict _find_tr(_PulseqCollection& pc, int subsequence_idx = 0)
{
    const auto& c = pc.coll();
    auto si = c.subseq_info(subsequence_idx);
    py::dict out;
    out["tr_size"] = si.tr_size;
    out["num_trs"] = si.num_trs;
    out["tr_duration_us"] = si.tr_duration_us;
    out["num_averages"] = si.num_averages;
    out["num_canonical_trs"] = si.num_canonical_trs;
    out["num_tr_instances"] = si.num_tr_instances;
    return out;
}

static py::dict _calc_mech_resonances(
    _PulseqCollection& pc,
    int subsequence_idx,
    int canonical_tr_idx,
    float target_resolution_hz,
    float max_freq_hz,
    py::list py_bands,
    py::object peak_log10_threshold,
    py::object peak_norm_scale,
    py::object peak_eps,
    py::object peak_prominence,
    bool compress_trains)
{
    std::vector<pulseg::ForbiddenBand> bands;
    for (auto item : py_bands)
    {
        py::tuple t = item.cast<py::tuple>();
        pulseg::ForbiddenBand b;
        b.freq_min_hz = t[0].cast<float>();
        b.freq_max_hz = t[1].cast<float>();
        b.max_amplitude_hz_per_m = t[2].cast<float>();
        bands.push_back(b);
    }

    auto parse_optional_float = [](const py::object& obj) -> float
    {
        if (obj.is_none())
        {
            return std::numeric_limits<float>::quiet_NaN();
        }
        return obj.cast<float>();
    };

    float peak_log10_threshold_val = parse_optional_float(peak_log10_threshold);
    float peak_norm_scale_val = parse_optional_float(peak_norm_scale);
    float peak_eps_val = parse_optional_float(peak_eps);
    float peak_prominence_val = parse_optional_float(peak_prominence);

    auto sp = pc.coll().calc_mech_resonances(
        subsequence_idx,
        canonical_tr_idx,
        target_resolution_hz,
        max_freq_hz,
        bands,
        peak_log10_threshold_val,
        peak_norm_scale_val,
        peak_eps_val,
        peak_prominence_val,
        compress_trains);

    py::dict out;
    out["freq_min_hz"] = sp.freq_min_hz;
    out["freq_spacing_hz"] = sp.freq_spacing_hz;
    out["num_freq_bins"] = sp.num_freq_bins;

    out["spectrum_full_gx"] = sp.spectrum_full_gx;
    out["spectrum_full_gy"] = sp.spectrum_full_gy;
    out["spectrum_full_gz"] = sp.spectrum_full_gz;

    out["num_instances"] = sp.num_instances;

    out["num_analytical_peaks"] = sp.num_analytical_peaks;
    out["analytical_peak_freqs"] = sp.analytical_peak_freqs;
    out["analytical_peak_amp_gx"] = sp.analytical_peak_amp_gx;
    out["analytical_peak_amp_gy"] = sp.analytical_peak_amp_gy;
    out["analytical_peak_amp_gz"] = sp.analytical_peak_amp_gz;
    out["analytical_peak_phase_gx"] = sp.analytical_peak_phase_gx;
    out["analytical_peak_phase_gy"] = sp.analytical_peak_phase_gy;
    out["analytical_peak_phase_gz"] = sp.analytical_peak_phase_gz;
    out["analytical_peak_widths_hz"] = sp.analytical_peak_widths_hz;
    out["num_candidates"] = sp.num_candidates;
    out["candidate_freqs"] = sp.candidate_freqs;
    out["candidate_amps_gx"] = sp.candidate_amps_gx;
    out["candidate_amps_gy"] = sp.candidate_amps_gy;
    out["candidate_amps_gz"] = sp.candidate_amps_gz;
    out["candidate_grad_amps"] = sp.candidate_grad_amps;
    out["candidate_grad_amps_gx"] = sp.candidate_grad_amps_gx;
    out["candidate_grad_amps_gy"] = sp.candidate_grad_amps_gy;
    out["candidate_grad_amps_gz"] = sp.candidate_grad_amps_gz;
    out["candidate_violations"] = sp.candidate_violations;

    out["num_component_terms"] = sp.num_component_terms;
    out["component_freqs_hz"] = sp.component_freqs_hz;
    out["component_amps"] = sp.component_amps;
    out["component_phases_rad"] = sp.component_phases_rad;
    out["component_widths_hz"] = sp.component_widths_hz;
    out["component_axes"] = sp.component_axes;
    out["component_def_ids"] = sp.component_def_ids;
    out["component_contrib_ids"] = sp.component_contrib_ids;
    out["component_run_ids"] = sp.component_run_ids;

    out["num_surviving_freqs"] = sp.num_surviving_freqs;
    out["surviving_freqs_hz"] = sp.surviving_freqs_hz;

    out["num_envelope_bins"] = sp.num_envelope_bins;
    out["envelope_freqs_hz"] = sp.envelope_freqs_hz;
    out["envelope_amp_gx"] = sp.envelope_amp_gx;
    out["envelope_amp_gy"] = sp.envelope_amp_gy;
    out["envelope_amp_gz"] = sp.envelope_amp_gz;

    return out;
}

static py::dict _calc_pns(
    _PulseqCollection& pc,
    int subsequence_idx,
    int canonical_tr_idx,
    float chronaxie_us,
    float rheobase,
    float alpha)
{
    pulseg_pns_irnich ctx;
    pulseg_pns_model model;
    pulseg_pns_irnich_init(&model, &ctx, chronaxie_us, rheobase, alpha);

    auto r = pc.coll().calc_pns(subsequence_idx, canonical_tr_idx, model);

    py::dict out;
    out["num_samples"] = r.num_samples;
    out["slew_x"] = r.slew_x;
    out["slew_y"] = r.slew_y;
    out["slew_z"] = r.slew_z;
    return out;
}

/** One axis of a SAFE hardware table, as (a1, a2, a3, tau1, tau2, tau3,
 *  stim_limit, g_scale) -- the order pypulseq's asc_to_hw reports them. */
static pulseg_pns_safe_axis _safe_axis(const py::sequence& coeffs, const char* name)
{
    if (py::len(coeffs) != 8)
        throw std::invalid_argument(
            std::string("SAFE axis '") + name +
            "' needs 8 coefficients (a1, a2, a3, tau1_ms, tau2_ms, tau3_ms, stim_limit, g_scale)");

    pulseg_pns_safe_axis axis;
    axis.a1 = coeffs[0].cast<float>();
    axis.a2 = coeffs[1].cast<float>();
    axis.a3 = coeffs[2].cast<float>();
    axis.tau1_ms = coeffs[3].cast<float>();
    axis.tau2_ms = coeffs[4].cast<float>();
    axis.tau3_ms = coeffs[5].cast<float>();
    axis.stim_limit = coeffs[6].cast<float>();
    axis.g_scale = coeffs[7].cast<float>();
    return axis;
}

static py::dict _calc_pns_safe(
    _PulseqCollection& pc,
    int subsequence_idx,
    int canonical_tr_idx,
    const py::sequence& gx,
    const py::sequence& gy,
    const py::sequence& gz)
{
    pulseg_pns_safe ctx;
    pulseg_pns_model model;
    ctx.x = _safe_axis(gx, "x");
    ctx.y = _safe_axis(gy, "y");
    ctx.z = _safe_axis(gz, "z");
    pulseg_pns_safe_init(&model, &ctx);

    auto r = pc.coll().calc_pns(subsequence_idx, canonical_tr_idx, model);

    py::dict out;
    out["num_samples"] = r.num_samples;
    out["slew_x"] = r.slew_x;
    out["slew_y"] = r.slew_y;
    out["slew_z"] = r.slew_z;
    return out;
}

// ─── TR waveforms ──────────────────────────────────────────────────

/**
 * Native-timing waveforms of one TR: gradients, RF magnitude and phase, ADC
 * event descriptors and per-block metadata, all on one TR-relative time base.
 * `amplitude_mode` is PULSEG_AMP_MAX_POS (0), _ZERO_VAR (1) or _ACTUAL (2);
 * `tr_index` is read only by _ACTUAL.
 *
 * Everything pulseg_get_tr_waveforms fills is handed across, so that a viewer
 * draws the same canonical TR the safety checks judge rather than one rebuilt
 * from the timeline. RF is channel-major when num_rf_channels > 1.
 */
static py::dict _get_tr_waveforms(
    _PulseqCollection& pc,
    int subsequence_idx,
    int amplitude_mode,
    int tr_index,
    bool collapse_delays,
    int num_averages)
{
    const auto w = pc.coll().get_tr_waveforms(
        subsequence_idx,
        amplitude_mode,
        tr_index,
        collapse_delays,
        num_averages);

    auto axis = [](const pulseg::ChannelWaveform& ch)
    {
        py::dict d;
        d["time_us"] = ch.time_us;
        d["amplitude"] = ch.amplitude;
        return d;
    };

    py::list adc_events;
    for (const auto& a : w.adc_events)
    {
        py::dict d;
        d["onset_us"] = a.onset_us;
        d["duration_us"] = a.duration_us;
        d["num_samples"] = a.num_samples;
        d["freq_offset_hz"] = a.freq_offset_hz;
        d["phase_offset_rad"] = a.phase_offset_rad;
        adc_events.append(d);
    }

    py::list blocks;
    for (const auto& b : w.blocks)
    {
        py::dict d;
        d["start_us"] = b.start_us;
        d["duration_us"] = b.duration_us;
        d["segment_idx"] = b.segment_idx;
        d["rf_isocenter_us"] = b.rf_isocenter_us;
        blocks.append(d);
    }

    py::dict out;
    out["gx"] = axis(w.gx);
    out["gy"] = axis(w.gy);
    out["gz"] = axis(w.gz);
    out["num_rf_channels"] = w.num_rf_channels;
    out["rf_mag"] = axis(w.rf_mag);
    out["rf_phase"] = axis(w.rf_phase);
    out["adc_events"] = adc_events;
    out["blocks"] = blocks;
    out["total_duration_us"] = w.total_duration_us;
    return out;
}

/*
 * One subsequence's canonical-TR event table, plus the scan-global
 * parameters.
 *
 * The rows cross as flat arrays -- a type column, a timestamp column and an
 * (n, 7) params block -- rather than as a list of dicts: the table is one row
 * per block over a whole pass, and boxing seven floats per row would cost more
 * than everything else here. Their meaning is the C header's
 * (pulseg_types.h); the Python layer names the fields.
 */
static py::dict _get_sequence_description(_PulseqCollection& pc, int subsequence_idx)
{
    const auto desc = pc.coll().get_sequence_description(subsequence_idx);
    const auto count = static_cast<py::ssize_t>(desc.rows.size());

    py::array_t<int> types(count);
    py::array_t<float> timestamps(count);
    py::array_t<float> params({count, static_cast<py::ssize_t>(PULSEG_SEQ_EVENT_PARAMS)});

    auto type_view = types.mutable_unchecked<1>();
    auto stamp_view = timestamps.mutable_unchecked<1>();
    auto param_view = params.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < count; ++i)
    {
        const pulseg_seq_event& row = desc.rows[static_cast<size_t>(i)];
        type_view(i) = row.type;
        stamp_view(i) = row.timestamp_us;
        for (py::ssize_t j = 0; j < static_cast<py::ssize_t>(PULSEG_SEQ_EVENT_PARAMS); ++j)
            param_view(i, j) = row.params[j];
    }

    py::dict out;
    out["subseq_idx"] = desc.subseq_idx;
    out["tr_duration_us"] = desc.tr_duration_us;
    out["type"] = types;
    out["timestamp_us"] = timestamps;
    out["params"] = params;
    return out;
}

/*
 * The RF definitions an event table's rf_def_id column points into, walked
 * until the collection stops recognising an id. Shapes come across in the
 * file's units (normalised magnitude, phase in turns, times in us) -- the
 * Python layer is where they become physics.
 */
static py::list _get_rf_definitions(_PulseqCollection& pc, int subsequence_idx)
{
    py::list out;
    for (int rf_def_id = 0;; ++rf_def_id)
    {
        const auto shapes = pc.coll().get_rf_definition(subsequence_idx, rf_def_id);
        if (shapes.magnitude.empty())
            break;

        py::dict entry;
        entry["rf_def_id"] = rf_def_id;
        entry["num_channels"] = shapes.num_channels;
        entry["magnitude"] = shapes.magnitude;
        entry["phase_turns"] = shapes.phase_turns;
        entry["time_us"] = shapes.time_us;
        out.append(entry);
    }
    return out;
}

static py::dict _get_sequence_parameters(_PulseqCollection& pc)
{
    const auto p = pc.coll().get_sequence_parameters();
    py::dict out;
    out["min_te_us"] = p.min_te_us;
    out["min_tr_us"] = p.min_tr_us;
    out["max_tr_us"] = p.max_tr_us;
    out["max_flip_angle_deg"] = p.max_flip_angle_deg;
    out["total_scan_time_us"] = p.total_scan_time_us;
    out["num_subseqs"] = p.num_subseqs;
    return out;
}

// ─── Check functions ────────────────────────────────────────────────

static void _check_consistency(_PulseqCollection& pc)
{
    pc.coll().check_consistency();
}

static std::string _check_grad_continuity(_PulseqCollection& pc)
{
    return pc.coll().check_grad_continuity();
}

static void _check_safety(
    _PulseqCollection& pc,
    py::list py_bands,
    float stim_threshold,
    float decay_constant_us,
    float pns_threshold_percent,
    bool skip_pns)
{
    std::vector<pulseg::ForbiddenBand> bands;
    for (auto item : py_bands)
    {
        py::tuple t = item.cast<py::tuple>();
        pulseg::ForbiddenBand b;
        b.freq_min_hz = t[0].cast<float>();
        b.freq_max_hz = t[1].cast<float>();
        b.max_amplitude_hz_per_m = t[2].cast<float>();
        bands.push_back(b);
    }

    const pulseg_pns_model* pns_ptr = nullptr;
    pulseg_pns_irnich ctx;
    pulseg_pns_model model;
    if (!skip_pns)
    {
        /* alpha folded into stim_threshold, which is rheobase/alpha. */
        pulseg_pns_irnich_init(&model, &ctx, decay_constant_us, stim_threshold, 1.0f);
        pns_ptr = &model;
    }

    pc.coll().check_safety(bands, pns_ptr, pns_threshold_percent);
}

// ─── Segment layout ────────────────────────────────────────────────

/**
 * Per-segment layout of one subsequence, resolved to the *max-energy*
 * instance of every unique segment. All resolution logic lives in
 * pulseg::Collection::segments() (src/cpp/include/pulseg/collection.hpp); this
 * function only converts its result to Python types.
 */
static py::list _get_segments(_PulseqCollection& pc, int subseq_idx)
{
    py::list out;
    const auto segments = pc.coll().segments(subseq_idx);
    for (size_t s = 0; s < segments.size(); ++s)
    {
        const auto& seg = segments[s];

        py::list block_indices;
        for (int blk : seg.block_indices)
            block_indices.append(blk);

        py::dict d;
        d["index"] = static_cast<int>(s);
        d["global_index"] = seg.global_index;
        d["num_blocks"] = seg.num_blocks;
        d["start_block"] = seg.start_block;
        d["max_energy_start_block"] = seg.max_energy_start_block;
        d["from_max_energy_instance"] = seg.from_max_energy_instance;
        d["block_indices"] = block_indices;
        d["duration_us"] = seg.duration_us;
        d["pure_delay"] = seg.pure_delay;
        d["is_nav"] = seg.is_nav;
        d["has_trigger"] = seg.has_trigger;
        out.append(d);
    }
    return out;
}

// ─── Conformance projections ────────────────────────────────────────
//
// Read-only views of the two PulSeg IR classes the id-indexed layout
// factors apart: the virtual-segment content (spec 3.0/3.2, per-segment
// per-block event definitions with hydrated waveforms) and the instance
// table (spec 3.3, one resolved row per executed block).

static py::array_t<float> _own_float_array(float* data, int n)
{
    py::array_t<float> out(n);
    std::memcpy(out.mutable_data(), data, sizeof(float) * static_cast<size_t>(n));
    PULSEG_FREE(data);
    return out;
}

static py::array_t<float> _own_float_matrix(float** rows, int nrows, int ncols)
{
    py::array_t<float> out({nrows, ncols});
    for (int r = 0; r < nrows; ++r)
    {
        std::memcpy(out.mutable_data(r, 0), rows[r], sizeof(float) * static_cast<size_t>(ncols));
        PULSEG_FREE(rows[r]);
    }
    PULSEG_FREE(rows);
    return out;
}

static py::list _get_segment_blocks(_PulseqCollection& pc)
{
    const pulseg_collection* coll = pc.coll().handle();
    py::list segments;

    pulseg_collection_info cinfo = PULSEG_COLLECTION_INFO_INIT;
    if (!PULSEG_SUCCEEDED(pulseg_get_collection_info(coll, &cinfo)))
        throw std::runtime_error("pulseg_get_collection_info failed");
    const int num_segments = cinfo.num_segments;
    for (int s = 0; s < num_segments; ++s)
    {
        pulseg_segment_info segi = PULSEG_SEGMENT_INFO_INIT;
        if (!PULSEG_SUCCEEDED(pulseg_get_segment_info(coll, &segi, s)))
            throw std::runtime_error("pulseg_get_segment_info failed");

        py::dict seg;
        seg["duration_us"] = segi.duration_us;
        seg["num_blocks"] = segi.num_blocks;
        seg["start_block"] = segi.start_block;
        seg["pure_delay"] = segi.pure_delay;
        seg["is_nav"] = segi.is_nav;
        seg["has_trigger"] = segi.has_trigger;
        seg["trigger_type"] = segi.trigger_type;

        py::list blocks;
        for (int b = 0; b < segi.num_blocks; ++b)
        {
            pulseg_block_info bi = PULSEG_BLOCK_INFO_INIT;
            if (!PULSEG_SUCCEEDED(pulseg_get_block_info(coll, &bi, s, b)))
                throw std::runtime_error("pulseg_get_block_info failed");

            py::dict blk;
            blk["duration_us"] = bi.duration_us;
            blk["start_time_us"] = bi.start_time_us;
            blk["has_rf"] = bi.has_rf;
            blk["has_adc"] = bi.has_adc;
            blk["has_rotation"] = bi.has_rotation;
            blk["has_digitalout"] = bi.has_digitalout;
            blk["is_variable_delay"] = bi.is_variable_delay;
            blk["norot_flag"] = bi.norot_flag;
            blk["nopos_flag"] = bi.nopos_flag;
            blk["rf_grad_constant"] = bi.rf_grad_constant;

            if (bi.has_rf)
            {
                blk["rf_delay_us"] = bi.rf_delay_us;
                blk["rf_num_channels"] = bi.rf_num_channels;
                blk["rf_duration_us"] = bi.rf_duration_us;
                blk["rf_initial_amplitude_hz"] = pulseg_get_rf_initial_amplitude_hz(coll, s, b);
                blk["rf_max_amplitude_hz"] = pulseg_get_rf_max_amplitude_hz(coll, s, b);

                int nch = 0;
                int ns = 0;
                float** mag = pulseg_get_rf_magnitude(coll, &nch, &ns, s, b);
                if (mag != NULL)
                    blk["rf_magnitude"] = _own_float_matrix(mag, nch, ns);
                nch = 0;
                ns = 0;
                float** phase = pulseg_get_rf_phase(coll, &nch, &ns, s, b);
                if (phase != NULL)
                    blk["rf_phase"] = _own_float_matrix(phase, nch, ns);
                float* time = pulseg_get_rf_time_us(coll, s, b);
                if (time != NULL && ns > 0)
                    blk["rf_time_us"] = _own_float_array(time, ns);
                else if (time != NULL)
                    PULSEG_FREE(time);
            }

            py::list grads;
            for (int axis = 0; axis < 3; ++axis)
            {
                if (!bi.has_grad[axis])
                {
                    grads.append(py::none());
                    continue;
                }
                py::dict g;
                g["is_trapezoid"] = bi.grad_is_trapezoid[axis];
                g["delay_us"] = bi.grad_delay_us[axis];
                g["num_shots"] = bi.grad_num_shots[axis];
                g["initial_amplitude_hz_per_m"] =
                    pulseg_get_grad_initial_amplitude_hz_per_m(coll, s, b, axis);
                g["max_amplitude_hz_per_m"] =
                    pulseg_get_grad_max_amplitude_hz_per_m(coll, s, b, axis);

                int nshots = 0;
                int ns = 0;
                float** amp = pulseg_get_grad_amplitude(coll, &nshots, &ns, s, b, axis);
                if (amp != NULL)
                    g["amplitude"] = _own_float_matrix(amp, nshots, ns);
                float* time = pulseg_get_grad_time_us(coll, s, b, axis);
                if (time != NULL && ns > 0)
                    g["time_us"] = _own_float_array(time, ns);
                else if (time != NULL)
                    PULSEG_FREE(time);
                grads.append(g);
            }
            blk["grads"] = grads;

            if (bi.has_adc)
            {
                blk["adc_delay_us"] = bi.adc_delay_us;
                blk["adc_def_id"] = bi.adc_def_id;
            }
            if (bi.has_digitalout)
            {
                blk["digitalout_delay_us"] = bi.digitalout_delay_us;
                blk["digitalout_duration_us"] = bi.digitalout_duration_us;
                blk["digitalout_channel"] = bi.digitalout_channel;
            }
            blocks.append(blk);
        }
        seg["blocks"] = blocks;
        segments.append(seg);
    }
    return segments;
}

static py::dict _get_scan_table(_PulseqCollection& pc)
{
    pulseg_collection* coll = pc.coll().handle();

    std::vector<int> subseq_idx, scan_pos, segment_id, segment_start, segment_end;
    std::vector<int> duration_us, rf_shim_id, gx_shape, gy_shape, gz_shape;
    std::vector<int> norot, nopos, digitalout_flag, digitalout_channel, adc_flag, trid;
    std::vector<float> rf_amp, rf_freq, rf_phase;
    std::vector<float> gx_amp, gy_amp, gz_amp, adc_freq, adc_phase;
    std::vector<float> rotmat;

    pulseg_cursor_reset(coll);
    pulseg_cursor_info info;
    int rc;
    while ((rc = pulseg_cursor_advance(coll, &info)) == PULSEG_CURSOR_BLOCK)
    {
        pulseg_block_instance inst = PULSEG_BLOCK_INSTANCE_INIT;
        if (!PULSEG_SUCCEEDED(pulseg_get_block_instance(coll, &inst)))
            throw std::runtime_error("pulseg_get_block_instance failed");

        subseq_idx.push_back(info.subseq_idx);
        scan_pos.push_back(info.scan_pos);
        segment_id.push_back(info.segment_id);
        segment_start.push_back(info.segment_start);
        segment_end.push_back(info.segment_end);
        duration_us.push_back(inst.duration_us);
        rf_amp.push_back(inst.rf_amp_hz);
        rf_freq.push_back(inst.rf_freq_hz);
        rf_phase.push_back(inst.rf_phase_rad);
        rf_shim_id.push_back(inst.rf_shim_id);
        gx_amp.push_back(inst.gx_amp_hz_per_m);
        gy_amp.push_back(inst.gy_amp_hz_per_m);
        gz_amp.push_back(inst.gz_amp_hz_per_m);
        gx_shape.push_back(inst.gx_shape_id);
        gy_shape.push_back(inst.gy_shape_id);
        gz_shape.push_back(inst.gz_shape_id);
        norot.push_back(inst.norot_flag);
        nopos.push_back(inst.nopos_flag);
        digitalout_flag.push_back(inst.digitalout_flag);
        digitalout_channel.push_back(inst.digitalout_channel);
        adc_flag.push_back(inst.adc_flag);
        adc_freq.push_back(inst.adc_freq_hz);
        adc_phase.push_back(inst.adc_phase_rad);
        trid.push_back(inst.trid);
        rotmat.insert(rotmat.end(), inst.rotmat, inst.rotmat + 9);
    }
    pulseg_cursor_reset(coll);
    if (rc != PULSEG_CURSOR_DONE)
        throw std::runtime_error("cursor walk failed before the end of the stream");

    const int n = static_cast<int>(subseq_idx.size());
    auto ints = [n](const std::vector<int>& v)
    {
        py::array_t<int> out(n);
        std::memcpy(out.mutable_data(), v.data(), sizeof(int) * static_cast<size_t>(n));
        return out;
    };
    auto floats = [n](const std::vector<float>& v)
    {
        py::array_t<float> out(n);
        std::memcpy(out.mutable_data(), v.data(), sizeof(float) * static_cast<size_t>(n));
        return out;
    };

    py::dict out;
    out["subseq_idx"] = ints(subseq_idx);
    out["scan_pos"] = ints(scan_pos);
    out["segment_id"] = ints(segment_id);
    out["segment_start"] = ints(segment_start);
    out["segment_end"] = ints(segment_end);
    out["duration_us"] = ints(duration_us);
    out["rf_amp_hz"] = floats(rf_amp);
    out["rf_freq_hz"] = floats(rf_freq);
    out["rf_phase_rad"] = floats(rf_phase);
    out["rf_shim_id"] = ints(rf_shim_id);
    out["gx_amp_hz_per_m"] = floats(gx_amp);
    out["gy_amp_hz_per_m"] = floats(gy_amp);
    out["gz_amp_hz_per_m"] = floats(gz_amp);
    out["gx_shape_id"] = ints(gx_shape);
    out["gy_shape_id"] = ints(gy_shape);
    out["gz_shape_id"] = ints(gz_shape);
    out["norot_flag"] = ints(norot);
    out["nopos_flag"] = ints(nopos);
    out["digitalout_flag"] = ints(digitalout_flag);
    out["digitalout_channel"] = ints(digitalout_channel);
    out["adc_flag"] = ints(adc_flag);
    out["adc_freq_hz"] = floats(adc_freq);
    out["adc_phase_rad"] = floats(adc_phase);
    out["trid"] = ints(trid);
    {
        py::array_t<float> rm({n, 9});
        std::memcpy(rm.mutable_data(), rotmat.data(), sizeof(float) * static_cast<size_t>(n) * 9);
        out["rotmat"] = rm;
    }
    return out;
}

static py::array_t<int> _get_adc_labels(_PulseqCollection& pc, int subseq_idx)
{
    const pulseg_collection* coll = pc.coll().handle();

    pulseg_subseq_info sinfo = PULSEG_SUBSEQ_INFO_INIT;
    if (!PULSEG_SUCCEEDED(pulseg_get_subseq_info(coll, &sinfo, subseq_idx)))
        throw std::runtime_error("pulseg_get_subseq_info failed");

    const int rows = sinfo.num_adc_occurrences;
    const int cols = sinfo.num_label_columns;
    py::array_t<int> out({rows, cols});
    for (int r = 0; r < rows; ++r)
    {
        if (!PULSEG_SUCCEEDED(pulseg_get_adc_label(coll, out.mutable_data(r, 0), subseq_idx, r)))
            throw std::runtime_error("pulseg_get_adc_label failed");
    }
    return out;
}

// ─── Module ─────────────────────────────────────────────────────────

void pulserver_bind_pulseg(py::module_& m)
{
    py::class_<_PulseqCollection>(m, "_PulseqCollection")
        .def(
            py::init<
                py::list,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                bool,
                int,
                std::vector<int>>(),
            py::arg("seq_bytes_list"),
            py::arg("gamma"),
            py::arg("B0"),
            py::arg("max_grad"),
            py::arg("max_slew"),
            py::arg("rf_raster_time"),
            py::arg("grad_raster_time"),
            py::arg("adc_raster_time"),
            py::arg("block_duration_raster"),
            py::arg("parse_labels") = true,
            py::arg("num_averages") = 1,
            py::arg("label_column_map") = std::vector<int>())
        .def(
            py::init<
                std::string,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                bool,
                int,
                std::vector<int>>(),
            py::arg("file_path"),
            py::arg("gamma"),
            py::arg("B0"),
            py::arg("max_grad"),
            py::arg("max_slew"),
            py::arg("rf_raster_time"),
            py::arg("grad_raster_time"),
            py::arg("adc_raster_time"),
            py::arg("block_duration_raster"),
            py::arg("parse_labels") = true,
            py::arg("num_averages") = 1,
            py::arg("label_column_map") = std::vector<int>());

    m.def("_find_tr", &_find_tr, py::arg("collection"), py::arg("subsequence_idx") = 0);

    m.def(
        "_calc_mech_resonances",
        &_calc_mech_resonances,
        py::arg("collection"),
        py::arg("subsequence_idx") = 0,
        py::arg("canonical_tr_idx") = 0,
        py::arg("target_resolution_hz"),
        py::arg("max_freq_hz"),
        py::arg("forbidden_bands") = py::list(),
        py::arg("peak_log10_threshold") = py::none(),
        py::arg("peak_norm_scale") = py::none(),
        py::arg("peak_eps") = py::none(),
        py::arg("peak_prominence") = py::none(),
        py::arg("compress_trains") = true);

    m.def(
        "_calc_pns",
        &_calc_pns,
        py::arg("collection"),
        py::arg("subsequence_idx") = 0,
        py::arg("canonical_tr_idx") = 0,
        py::arg("chronaxie_us"),
        py::arg("rheobase"),
        py::arg("alpha"));

    m.def(
        "_calc_pns_safe",
        &_calc_pns_safe,
        py::arg("collection"),
        py::arg("subsequence_idx") = 0,
        py::arg("canonical_tr_idx") = 0,
        py::arg("gx"),
        py::arg("gy"),
        py::arg("gz"));

    m.def(
        "_get_tr_waveforms",
        &_get_tr_waveforms,
        py::arg("collection"),
        py::arg("subsequence_idx") = 0,
        py::arg("amplitude_mode") = 0,
        py::arg("tr_index") = 0,
        py::arg("collapse_delays") = false,
        py::arg("num_averages") = 0);

    m.def(
        "_get_sequence_description",
        &_get_sequence_description,
        py::arg("collection"),
        py::arg("subsequence_idx") = 0);

    m.def(
        "_get_rf_definitions",
        &_get_rf_definitions,
        py::arg("collection"),
        py::arg("subsequence_idx") = 0);

    m.def("_get_sequence_parameters", &_get_sequence_parameters, py::arg("collection"));

    m.def("_check_consistency", &_check_consistency, py::arg("collection"));

    m.def(
        "_check_safety",
        &_check_safety,
        py::arg("collection"),
        py::arg("forbidden_bands") = py::list(),
        py::arg("stim_threshold") = 0.0f,
        py::arg("decay_constant_us") = 0.0f,
        py::arg("pns_threshold_percent") = 100.0f,
        py::arg("skip_pns") = true);

    m.def("_check_grad_continuity", &_check_grad_continuity, py::arg("collection"));

    m.def("_get_segments", &_get_segments, py::arg("collection"), py::arg("subseq_idx") = 0);
    m.def("_get_segment_blocks", &_get_segment_blocks, py::arg("collection"));
    m.def("_get_scan_table", &_get_scan_table, py::arg("collection"));
    m.def("_get_adc_labels", &_get_adc_labels, py::arg("collection"), py::arg("subseq_idx") = 0);
}
