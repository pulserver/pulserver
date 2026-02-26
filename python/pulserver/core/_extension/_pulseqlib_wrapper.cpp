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
    _PulseqCollection(const py::bytes& seq_bytes,
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

        std::string buffer = seq_bytes;
        const char* buf_ptr  = buffer.data();
        int         buf_size = static_cast<int>(buffer.size());

        coll_ = std::unique_ptr<pulseqlib::Collection>(
            new pulseqlib::Collection(
                &buf_ptr, &buf_size, 1, opts, parse_labels, num_averages));
        source_size_ = buf_size;
    }

    pulseqlib::Collection& coll() { return *coll_; }
    const pulseqlib::Collection& coll() const { return *coll_; }
    int source_size() const { return source_size_; }

private:
    std::unique_ptr<pulseqlib::Collection> coll_;
    int source_size_ = 0;
};

// ─── Thin conversion functions ──────────────────────────────────────

static py::dict _find_tr(_PulseqCollection& pc) {
    const auto& c = pc.coll();
    auto si = c.subseq_info(0);
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
    return out;
}

static py::dict _find_segments(_PulseqCollection& pc) {
    const auto& c = pc.coll();
    auto ci = c.collection_info();
    py::dict out;

    py::list segments;
    for (int i = 0; i < ci.num_segments; ++i) {
        auto seg = c.segment_info(i);
        py::dict segd;
        segd["start_block"] = seg.start_block;
        segd["num_blocks"]  = seg.num_blocks;
        segments.append(segd);
    }
    out["unique_segments"]        = segments;
    out["prep_segment_table"]     = c.prep_segment_table();
    out["main_segment_table"]     = c.main_segment_table();
    out["cooldown_segment_table"] = c.cooldown_segment_table();
    return out;
}

static py::dict _get_tr_gradient_waveforms(_PulseqCollection& pc) {
    auto wf = pc.coll().get_tr_gradient_waveforms();
    py::dict out;
    out["time_gx"]     = wf.gx.time_us;
    out["waveform_gx"] = wf.gx.amplitude_hz_per_m;
    out["time_gy"]     = wf.gy.time_us;
    out["waveform_gy"] = wf.gy.amplitude_hz_per_m;
    out["time_gz"]     = wf.gz.time_us;
    out["waveform_gz"] = wf.gz.amplitude_hz_per_m;
    return out;
}

static py::dict _get_tr_waveforms(
    _PulseqCollection& pc,
    int amplitude_mode,
    int tr_index,
    bool include_prep,
    bool include_cooldown)
{
    auto wf = pc.coll().get_tr_waveforms(
        0, amplitude_mode, tr_index, include_prep, include_cooldown);
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
    int target_window_size,
    float target_resolution_hz,
    float max_freq_hz,
    py::list py_bands)
{
    std::vector<pulseqlib::ForbiddenBand> bands;
    for (auto item : py_bands) {
        py::dict d = item.cast<py::dict>();
        pulseqlib::ForbiddenBand b;
        b.freq_min_hz            = d["freq_min_hz"].cast<float>();
        b.freq_max_hz            = d["freq_max_hz"].cast<float>();
        b.max_amplitude_hz_per_m = d["max_amplitude"].cast<float>();
        bands.push_back(b);
    }

    auto sp = pc.coll().calc_acoustic_spectra(
        0, target_window_size, target_resolution_hz, max_freq_hz, bands);

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
    float chronaxie_us,
    float rheobase,
    float alpha)
{
    pulseqlib::PnsParams params;
    params.chronaxie_us            = chronaxie_us;
    params.rheobase_hz_per_m_per_s = rheobase;
    params.alpha                   = alpha;

    auto r = pc.coll().calc_pns(0, params);

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
    float pns_chronaxie_us,
    float pns_rheobase,
    float pns_alpha,
    float pns_threshold_percent,
    bool  skip_pns)
{
    std::vector<pulseqlib::ForbiddenBand> bands;
    for (auto item : py_bands) {
        py::dict d = item.cast<py::dict>();
        pulseqlib::ForbiddenBand b;
        b.freq_min_hz            = d["freq_min_hz"].cast<float>();
        b.freq_max_hz            = d["freq_max_hz"].cast<float>();
        b.max_amplitude_hz_per_m = d["max_amplitude"].cast<float>();
        bands.push_back(b);
    }

    const pulseqlib::PnsParams* pns_ptr = nullptr;
    pulseqlib::PnsParams pns;
    if (!skip_pns) {
        pns.chronaxie_us            = pns_chronaxie_us;
        pns.rheobase_hz_per_m_per_s = pns_rheobase;
        pns.alpha                   = pns_alpha;
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

// ─── Block info ─────────────────────────────────────────────────────

static py::dict _get_block_info(_PulseqCollection& pc, int seg, int blk) {
    auto info = pc.coll().block_info(seg, blk);
    py::dict out;
    out["duration_us"]   = info.duration_us;
    out["start_time_us"] = info.start_time_us;

    /* Gradient per axis */
    for (int ax = 0; ax < 3; ++ax) {
        std::string prefix = std::string(1, "xyz"[ax]);
        out[(prefix + "_has_grad").c_str()]       = info.has_grad[ax] != 0;
        out[(prefix + "_is_trapezoid").c_str()]   = info.grad_is_trapezoid[ax] != 0;
        out[(prefix + "_grad_delay_us").c_str()]  = info.grad_delay_us[ax];
        out[(prefix + "_num_samples").c_str()]     = info.grad_num_samples[ax];
    }

    /* RF */
    out["has_rf"]           = info.has_rf != 0;
    out["rf_delay_us"]      = info.rf_delay_us;
    out["rf_num_samples"]   = info.rf_num_samples;

    /* ADC */
    out["has_adc"]          = info.has_adc != 0;
    out["adc_delay_us"]     = info.adc_delay_us;

    return out;
}

// ─── Cache functions ────────────────────────────────────────────────

static void _save_cache(_PulseqCollection& pc,
                        const std::string& path,
                        int source_size) {
    int sz = (source_size > 0) ? source_size : pc.source_size();
    pc.coll().save_cache(path, sz);
}

static void _load_cache(_PulseqCollection& pc,
                        const std::string& path) {
    pc.coll().load_cache(path, pc.source_size());
}

// ─── Module ─────────────────────────────────────────────────────────

PYBIND11_MODULE(_pulseqlib_wrapper, m) {
    py::class_<_PulseqCollection>(m, "_PulseqCollection")
        .def(py::init<py::bytes, float, float, float, float,
                       float, float, float, float, bool, int>(),
             py::arg("seq_bytes"),
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
          py::arg("collection"));

    m.def("_find_segments", &_find_segments,
          py::arg("collection"));

    m.def("_get_tr_gradient_waveforms", &_get_tr_gradient_waveforms,
          py::arg("collection"));

    m.def("_get_tr_waveforms", &_get_tr_waveforms,
          py::arg("collection"),
          py::arg("amplitude_mode") = 0,
          py::arg("tr_index") = 0,
          py::arg("include_prep") = false,
          py::arg("include_cooldown") = false);

    m.def("_calc_acoustic_spectra", &_calc_acoustic_spectra,
          py::arg("collection"),
          py::arg("target_window_size"),
          py::arg("target_resolution_hz"),
          py::arg("max_freq_hz"),
          py::arg("forbidden_bands") = py::list());

    m.def("_calc_pns", &_calc_pns,
          py::arg("collection"),
          py::arg("chronaxie_us"),
          py::arg("rheobase"),
          py::arg("alpha"));

    m.def("_check_consistency", &_check_consistency,
          py::arg("collection"));

    m.def("_check_safety", &_check_safety,
          py::arg("collection"),
          py::arg("forbidden_bands") = py::list(),
          py::arg("pns_chronaxie_us") = 0.0f,
          py::arg("pns_rheobase") = 0.0f,
          py::arg("pns_alpha") = 0.0f,
          py::arg("pns_threshold_percent") = 100.0f,
          py::arg("skip_pns") = true);

    m.def("_save_cache", &_save_cache,
          py::arg("collection"),
          py::arg("path"),
          py::arg("source_size"));

    m.def("_load_cache", &_load_cache,
          py::arg("collection"),
          py::arg("path"));

    m.def("_get_report", &_get_report,
          py::arg("collection"));

    m.def("_get_block_info", &_get_block_info,
          py::arg("collection"),
          py::arg("seg"),
          py::arg("blk"));
}
