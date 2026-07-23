/**
 * @file _pulseg_wrapper.cpp
 * @brief Thin pybind11 binding for pulseg C++11 interface.
 *
 * All docstrings belong in the Python wrapper layer.
 */

#include <pybind11/pybind11.h>
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
#include "pulseg_internal.h"

namespace py = pybind11;

// ─── Thin holder for the C++ Collection ─────────────────────────────

class _PulseqCollection
{
  public:
    _PulseqCollection(
        const py::list &seq_bytes_list,
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
        std::vector<const char *> buf_ptrs(n);
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
        const std::string &file_path,
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

    pulseg::Collection &coll()
    {
        return *coll_;
    }
    const pulseg::Collection &coll() const
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

static py::dict _find_tr(_PulseqCollection &pc, int subsequence_idx = 0)
{
    const auto &c = pc.coll();
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
    _PulseqCollection &pc,
    int subsequence_idx,
    int canonical_tr_idx,
    float target_resolution_hz,
    float max_freq_hz,
    py::list py_bands,
    py::object peak_log10_threshold,
    py::object peak_norm_scale,
    py::object peak_eps,
    py::object peak_prominence)
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

    auto parse_optional_float = [](const py::object &obj) -> float
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
        peak_prominence_val);

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
    _PulseqCollection &pc,
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

// ─── Check functions ────────────────────────────────────────────────

static void _check_consistency(_PulseqCollection &pc)
{
    pc.coll().check_consistency();
}

static void _check_safety(
    _PulseqCollection &pc,
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

    const pulseg::PnsParams *pns_ptr = nullptr;
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
 * instance of every unique segment.
 *
 * `max_energy_start_block` is a scan-table (exec stream) position, so it is
 * translated here into the 0-based .seq block indices that instance occupies;
 * Python only ever sees .seq block indices.  Falls back to the segment
 * definition's own `start_block` when the exec stream is unavailable (e.g. a
 * pulsegen cache load that never materialised the scan table).
 */
static py::list _get_segments(_PulseqCollection &pc, int subseq_idx)
{
    const pulseg_collection *coll_ptr = pc.coll().handle();
    if (!coll_ptr)
        throw std::runtime_error("pulseg collection is not initialised");
    if (subseq_idx < 0 || subseq_idx >= coll_ptr->num_subsequences)
        throw std::out_of_range("subseq_idx out of range");

    const pulseg_sequence_descriptor *desc = &coll_ptr->descriptors[subseq_idx];

    int seg_offset = 0;
    for (int i = 0; i < subseq_idx; ++i)
        seg_offset += coll_ptr->descriptors[i].num_unique_segments;

    py::list out;
    for (int s = 0; s < desc->num_unique_segments; ++s)
    {
        const pulseg_virtual_segment *seg = &desc->segment_definitions[s];
        int nb = seg->num_blocks;
        int start = seg->max_energy_start_block;

        py::list block_indices;
        bool from_exec = (start >= 0 && desc->exec_stream_block_idx != NULL &&
                          start + nb <= desc->exec_stream_len);
        for (int b = 0; b < nb; ++b)
        {
            int blk = from_exec ? desc->exec_stream_block_idx[start + b] : (seg->start_block + b);
            if (blk < 0 || blk >= desc->num_blocks)
            {
                block_indices = py::list();
                break;
            }
            block_indices.append(blk);
        }

        pulseg_segment_info info = PULSEG_SEGMENT_INFO_INIT;
        pulseg_get_segment_info(coll_ptr, &info, seg_offset + s);

        py::dict d;
        d["index"] = s;
        d["global_index"] = seg_offset + s;
        d["num_blocks"] = nb;
        d["start_block"] = seg->start_block;
        d["max_energy_start_block"] = start;
        d["from_max_energy_instance"] = from_exec;
        d["block_indices"] = block_indices;
        d["duration_us"] = info.duration_us;
        d["pure_delay"] = info.pure_delay;
        d["is_nav"] = info.is_nav;
        d["has_trigger"] = info.has_trigger;
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
        py::arg("peak_prominence") = py::none());

    m.def(
        "_calc_pns",
        &_calc_pns,
        py::arg("collection"),
        py::arg("subsequence_idx") = 0,
        py::arg("canonical_tr_idx") = 0,
        py::arg("chronaxie_us"),
        py::arg("rheobase"),
        py::arg("alpha"));

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
