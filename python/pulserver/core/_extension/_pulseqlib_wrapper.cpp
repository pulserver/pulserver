/**
 * @file _pulseqlib_wrapper.cpp
 * @brief Thin pybind11 binding for pulseqlib C++11 interface.
 *
 * All docstrings belong in the Python wrapper layer.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulseqlib.hpp"

namespace py = pybind11;

// ─── Thin holder for the C++ Collection ─────────────────────────────

class _PulseqCollection {
public:
    _PulseqCollection(const py::list& seq_bytes_list,
                      float gamma,
                      float B0,
                      float max_grad,
                      float max_slew,
                      float rf_raster_time,
                      float grad_raster_time,
                      float adc_raster_time,
                      float block_duration_raster,
                      bool  parse_labels,
                      int   num_averages)
    {
        pulseqlib::Opts opts;
        opts.gamma_hz_per_t          = gamma;
        opts.b0_t                    = B0;
        opts.max_grad_hz_per_m       = max_grad;
        opts.max_slew_hz_per_m_per_s = max_slew;
        opts.rf_raster_us            = rf_raster_time * 1e6f;
        opts.grad_raster_us          = grad_raster_time * 1e6f;
        opts.adc_raster_us           = adc_raster_time * 1e6f;
        opts.block_raster_us         = block_duration_raster * 1e6f;

        int n = static_cast<int>(seq_bytes_list.size());
        std::vector<std::string> buffers(n);
        std::vector<const char*> buf_ptrs(n);
        std::vector<int>         buf_sizes(n);
        for (int i = 0; i < n; ++i) {
            buffers[i]   = seq_bytes_list[i].cast<py::bytes>();
            buf_ptrs[i]  = buffers[i].data();
            buf_sizes[i] = static_cast<int>(buffers[i].size());
        }

        coll_ = std::unique_ptr<pulseqlib::Collection>(
            new pulseqlib::Collection(
                buf_ptrs.data(), buf_sizes.data(), n,
                opts, parse_labels, num_averages));
        source_size_ = (n > 0) ? buf_sizes[0] : 0;
    }

    pulseqlib::Collection& coll() { return *coll_; }
    const pulseqlib::Collection& coll() const { return *coll_; }
    int source_size() const { return source_size_; }

private:
    std::unique_ptr<pulseqlib::Collection> coll_;
    int source_size_ = 0;
};

// ─── Thin conversion functions ──────────────────────────────────────

static py::dict _find_tr(_PulseqCollection& pc, int subsequence_idx = 0) {
    const auto& c = pc.coll();
    auto si = c.subseq_info(subsequence_idx);
    py::dict out;
    out["tr_size"]              = si.tr_size;
    out["num_trs"]              = si.num_trs;
    out["num_prep_blocks"]      = si.num_prep_blocks;
    out["num_cooldown_blocks"]  = si.num_cooldown_blocks;
    out["degenerate_prep"]      = si.degenerate_prep;
    out["degenerate_cooldown"]  = si.degenerate_cooldown;
    out["num_prep_trs"]         = si.num_prep_trs;
    out["num_cooldown_trs"]     = si.num_cooldown_trs;
    out["tr_duration_us"]       = si.tr_duration_us;
    out["num_passes"]           = si.num_passes;
    return out;
}

static py::dict _get_tr_waveforms(
    _PulseqCollection& pc,
    int subsequence_idx,
    int amplitude_mode,
    int tr_index,
    bool include_prep,
    bool include_cooldown,
    bool collapse_delays)
{
    auto wf = pc.coll().get_tr_waveforms(
        subsequence_idx, amplitude_mode, tr_index, include_prep, include_cooldown,
        collapse_delays);
    py::dict out;

    auto ch_to_dict = [](const pulseqlib::ChannelWaveform& ch) -> py::dict {
        py::dict d;
        d["time_us"]   = ch.time_us;
        d["amplitude"] = ch.amplitude;
        return d;
    };
    out["gx"]       = ch_to_dict(wf.gx);
    out["gy"]       = ch_to_dict(wf.gy);
    out["gz"]       = ch_to_dict(wf.gz);
    out["rf_mag"]   = ch_to_dict(wf.rf_mag);
    out["rf_phase"] = ch_to_dict(wf.rf_phase);

    // ADC events
    py::list adc_list;
    for (const auto& a : wf.adc_events) {
        py::dict ad;
        ad["onset_us"]         = a.onset_us;
        ad["duration_us"]      = a.duration_us;
        ad["num_samples"]      = a.num_samples;
        ad["freq_offset_hz"]   = a.freq_offset_hz;
        ad["phase_offset_rad"] = a.phase_offset_rad;
        adc_list.append(ad);
    }
    out["adc_events"] = adc_list;

    // Block descriptors
    py::list blk_list;
    for (const auto& b : wf.blocks) {
        py::dict bd;
        bd["start_us"]    = b.start_us;
        bd["duration_us"] = b.duration_us;
        bd["segment_idx"] = b.segment_idx;
        blk_list.append(bd);
    }
    out["blocks"]             = blk_list;
    out["total_duration_us"]  = wf.total_duration_us;

    return out;
}

static py::dict _calc_acoustic_spectra(
    _PulseqCollection& pc,
    int subsequence_idx,
    int target_window_size,
    float target_resolution_hz,
    float max_freq_hz,
    py::list py_bands)
{
    std::vector<pulseqlib::ForbiddenBand> bands;
    for (auto item : py_bands) {
        py::tuple t = item.cast<py::tuple>();
        pulseqlib::ForbiddenBand b;
        b.freq_min_hz            = t[0].cast<float>();
        b.freq_max_hz            = t[1].cast<float>();
        b.max_amplitude_hz_per_m = t[2].cast<float>();
        bands.push_back(b);
    }

    auto sp = pc.coll().calc_acoustic_spectra(
        subsequence_idx, target_window_size, target_resolution_hz, max_freq_hz, bands);

    py::dict out;
    out["freq_min_hz"]       = sp.freq_min_hz;
    out["freq_spacing_hz"]   = sp.freq_spacing_hz;
    out["num_freq_bins"]     = sp.num_freq_bins;
    out["num_windows"]       = sp.num_windows;

    out["spectrogram_gx"]    = sp.spectrogram_gx;
    out["spectrogram_gy"]    = sp.spectrogram_gy;
    out["spectrogram_gz"]    = sp.spectrogram_gz;
    out["peaks_gx"]          = sp.peaks_gx;
    out["peaks_gy"]          = sp.peaks_gy;
    out["peaks_gz"]          = sp.peaks_gz;

    out["spectrum_full_gx"]  = sp.spectrum_full_gx;
    out["spectrum_full_gy"]  = sp.spectrum_full_gy;
    out["spectrum_full_gz"]  = sp.spectrum_full_gz;
    out["peaks_full_gx"]     = sp.peaks_full_gx;
    out["peaks_full_gy"]     = sp.peaks_full_gy;
    out["peaks_full_gz"]     = sp.peaks_full_gz;

    out["freq_spacing_seq_hz"] = sp.freq_spacing_seq_hz;
    out["num_freq_bins_seq"]   = sp.num_freq_bins_seq;
    if (sp.num_freq_bins_seq > 0) {
        out["spectrum_seq_gx"] = sp.spectrum_seq_gx;
        out["spectrum_seq_gy"] = sp.spectrum_seq_gy;
        out["spectrum_seq_gz"] = sp.spectrum_seq_gz;
        out["peaks_seq_gx"]    = sp.peaks_seq_gx;
        out["peaks_seq_gy"]    = sp.peaks_seq_gy;
        out["peaks_seq_gz"]    = sp.peaks_seq_gz;
    }
    return out;
}

static py::dict _calc_pns(
    _PulseqCollection& pc,
    int subsequence_idx,
    float chronaxie_us,
    float rheobase,
    float alpha)
{
    pulseqlib::PnsParams params;
    params.chronaxie_us            = chronaxie_us;
    params.rheobase_hz_per_m_per_s = rheobase;
    params.alpha                   = alpha;

    auto r = pc.coll().calc_pns(subsequence_idx, params);

    py::dict out;
    out["num_samples"] = r.num_samples;
    out["slew_x"]      = r.slew_x;
    out["slew_y"]      = r.slew_y;
    out["slew_z"]      = r.slew_z;
    return out;
}

// ─── Check functions ────────────────────────────────────────────────

static void _check_consistency(_PulseqCollection& pc) {
    pc.coll().check_consistency();
}

static void _check_safety(
    _PulseqCollection& pc,
    py::list py_bands,
    float stim_threshold,
    float decay_constant_us,
    float pns_threshold_percent,
    bool  skip_pns)
{
    std::vector<pulseqlib::ForbiddenBand> bands;
    for (auto item : py_bands) {
        py::tuple t = item.cast<py::tuple>();
        pulseqlib::ForbiddenBand b;
        b.freq_min_hz            = t[0].cast<float>();
        b.freq_max_hz            = t[1].cast<float>();
        b.max_amplitude_hz_per_m = t[2].cast<float>();
        bands.push_back(b);
    }

    const pulseqlib::PnsParams* pns_ptr = nullptr;
    pulseqlib::PnsParams pns;
    if (!skip_pns) {
        pns.chronaxie_us            = decay_constant_us;
        pns.rheobase_hz_per_m_per_s = stim_threshold;   // rheobase/alpha combined
        pns.alpha                   = 1.0f;             // folded into stim_threshold
        pns_ptr = &pns;
    }

    pc.coll().check_safety(bands, pns_ptr, pns_threshold_percent);
}

// ─── Report (collection + subseq + segment info) ───────────────────

static py::dict _get_report(_PulseqCollection& pc) {
    const auto& c = pc.coll();
    py::dict out;

    /* Collection-level */
    auto ci = c.collection_info();
    out["num_subsequences"]  = ci.num_subsequences;
    out["num_segments"]      = ci.num_segments;
    out["total_duration_us"] = ci.total_duration_us;

    /* Per-subsequence */
    py::list subseqs;
    for (int ss = 0; ss < ci.num_subsequences; ++ss) {
        auto si = c.subseq_info(ss);
        py::dict sd;
        sd["tr_size"]               = si.tr_size;
        sd["num_trs"]               = si.num_trs;
        sd["num_prep_blocks"]       = si.num_prep_blocks;
        sd["num_cooldown_blocks"]   = si.num_cooldown_blocks;
        sd["tr_duration_us"]        = si.tr_duration_us;
        sd["num_passes"]            = si.num_passes;
        sd["num_unique_segments"]   = si.num_prep_segments + si.num_main_segments + si.num_cooldown_segments;
        sd["segment_offset"]        = si.segment_offset;

        /* Unique segments in this subsequence */
        int seg_start = si.segment_offset;
        int seg_count = si.num_prep_segments + si.num_main_segments + si.num_cooldown_segments;
        py::list segs;
        for (int j = seg_start; j < seg_start + seg_count; ++j) {
            auto seg = c.segment_info(j);
            py::dict segd;
            segd["start_block"] = seg.start_block;
            segd["num_blocks"]  = seg.num_blocks;
            segs.append(segd);
        }
        sd["segments"] = segs;

        /* Segment tables */
        sd["prep_segment_table"]     = c.prep_segment_table(ss);
        sd["main_segment_table"]     = c.main_segment_table(ss);
        sd["cooldown_segment_table"] = c.cooldown_segment_table(ss);

        subseqs.append(sd);
    }
    out["subsequences"] = subseqs;
    return out;
}

// ─── Unique-block queries ────────────────────────────────────────────

static int _get_num_unique_blocks(_PulseqCollection& pc, int seq_idx) {
    return pc.coll().num_unique_blocks(seq_idx);
}

static int _get_unique_block_id(_PulseqCollection& pc, int seq_idx, int blk_def_idx) {
    return pc.coll().unique_block_id(seq_idx, blk_def_idx);
}

// ─── Module ─────────────────────────────────────────────────────────

PYBIND11_MODULE(_pulseqlib_wrapper, m) {
    py::class_<_PulseqCollection>(m, "_PulseqCollection")
        .def(py::init<py::list, float, float, float, float,
                       float, float, float, float, bool, int>(),
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
        ;

    m.def("_find_tr", &_find_tr,
          py::arg("collection"),
          py::arg("subsequence_idx") = 0);

    m.def("_get_tr_waveforms", &_get_tr_waveforms,
          py::arg("collection"),
          py::arg("subsequence_idx") = 0,
          py::arg("amplitude_mode") = 0,
          py::arg("tr_index") = 0,
          py::arg("include_prep") = false,
          py::arg("include_cooldown") = false,
          py::arg("collapse_delays") = false);

    m.def("_calc_acoustic_spectra", &_calc_acoustic_spectra,
          py::arg("collection"),
          py::arg("subsequence_idx") = 0,
          py::arg("target_window_size"),
          py::arg("target_resolution_hz"),
          py::arg("max_freq_hz"),
          py::arg("forbidden_bands") = py::list());

    m.def("_calc_pns", &_calc_pns,
          py::arg("collection"),
          py::arg("subsequence_idx") = 0,
          py::arg("chronaxie_us"),
          py::arg("rheobase"),
          py::arg("alpha"));

    m.def("_check_consistency", &_check_consistency,
          py::arg("collection"));

    m.def("_check_safety", &_check_safety,
          py::arg("collection"),
          py::arg("forbidden_bands") = py::list(),
          py::arg("stim_threshold") = 0.0f,
          py::arg("decay_constant_us") = 0.0f,
          py::arg("pns_threshold_percent") = 100.0f,
          py::arg("skip_pns") = true);

    m.def("_get_report", &_get_report,
          py::arg("collection"));

    m.def("_get_num_unique_blocks", &_get_num_unique_blocks,
          py::arg("collection"),
          py::arg("seq_idx"));

    m.def("_get_unique_block_id", &_get_unique_block_id,
          py::arg("collection"),
          py::arg("seq_idx"),
          py::arg("blk_def_idx"));

}
