/**
 * @file _pulseg_wrapper.cpp
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

#include "pulseg.hpp"
#include "pulseg.h"
#include "pulseg_types.h"

namespace py = pybind11;

// ─── Thin holder for the C++ Collection ─────────────────────────────

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
        int num_averages)
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
        int num_averages)
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
    out["num_prep_blocks"] = si.num_prep_blocks;
    out["num_cooldown_blocks"] = si.num_cooldown_blocks;
    out["degenerate_prep"] = si.degenerate_prep;
    out["degenerate_cooldown"] = si.degenerate_cooldown;
    out["num_prep_trs"] = si.num_prep_trs;
    out["num_cooldown_trs"] = si.num_cooldown_trs;
    out["tr_duration_us"] = si.tr_duration_us;
    out["num_passes"] = si.num_passes;
    out["num_averages"] = si.num_averages;
    out["num_canonical_trs"] = si.num_canonical_trs;
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
    pulseg::PnsParams params;
    params.chronaxie_us = chronaxie_us;
    params.rheobase_hz_per_m_per_s = rheobase;
    params.alpha = alpha;

    auto r = pc.coll().calc_pns(subsequence_idx, canonical_tr_idx, params);

    py::dict out;
    out["num_samples"] = r.num_samples;
    out["slew_x"] = r.slew_x;
    out["slew_y"] = r.slew_y;
    out["slew_z"] = r.slew_z;
    return out;
}

/** One axis of a SAFE hardware table, as (a1, a2, a3, tau1, tau2, tau3,
 *  stim_limit, g_scale) -- the order pypulseq's asc_to_hw reports them. */
static pulseg::SafeParams::Axis _safe_axis(const py::sequence& coeffs, const char* name)
{
    if (py::len(coeffs) != 8)
        throw std::invalid_argument(
            std::string("SAFE axis '") + name +
            "' needs 8 coefficients (a1, a2, a3, tau1_ms, tau2_ms, tau3_ms, stim_limit, g_scale)");

    pulseg::SafeParams::Axis axis;
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
    pulseg::SafeParams params;
    params.x = _safe_axis(gx, "x");
    params.y = _safe_axis(gy, "y");
    params.z = _safe_axis(gz, "z");

    auto r = pc.coll().calc_pns(subsequence_idx, canonical_tr_idx, params);

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
        d["adc_kzero_us"] = b.adc_kzero_us;
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

    const pulseg::PnsParams* pns_ptr = nullptr;
    pulseg::PnsParams pns;
    if (!skip_pns)
    {
        pns.chronaxie_us = decay_constant_us;
        pns.rheobase_hz_per_m_per_s = stim_threshold; // rheobase/alpha combined
        pns.alpha = 1.0f;                             // folded into stim_threshold
        pns_ptr = &pns;
    }

    pc.coll().check_safety(bands, pns_ptr, pns_threshold_percent);
}

// ─── Segment layout ────────────────────────────────────────────────

/**
 * Per-segment layout of one subsequence, resolved to the *max-energy*
 * instance of every unique segment. All resolution logic lives in
 * pulseg::Collection::segments() (cxx/include/pulseg/collection.hpp); this
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

// ─── Module ─────────────────────────────────────────────────────────

PYBIND11_MODULE(_pulseg_wrapper, m)
{
    py::class_<_PulseqCollection>(m, "_PulseqCollection")
        .def(
            py::init<py::list, float, float, float, float, float, float, float, float, bool, int>(),
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
            py::arg("num_averages") = 1)
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
                int>(),
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
            py::arg("num_averages") = 1);

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

    m.def("_get_segments", &_get_segments, py::arg("collection"), py::arg("subseq_idx") = 0);
}
