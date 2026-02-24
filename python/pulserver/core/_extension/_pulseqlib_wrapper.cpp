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
    }

    pulseqlib::Collection& coll() { return *coll_; }
    const pulseqlib::Collection& coll() const { return *coll_; }

private:
    std::unique_ptr<pulseqlib::Collection> coll_;
};

// ─── Thin conversion functions ──────────────────────────────────────

static py::dict _find_tr(_PulseqCollection& pc) {
    const auto& c = pc.coll();
    py::dict out;
    out["tr_size"]              = c.tr_size();
    out["num_trs"]              = c.num_trs();
    out["num_prep_blocks"]      = c.num_prep_blocks();
    out["num_cooldown_blocks"]  = c.num_cooldown_blocks();
    out["degenerate_prep"]      = c.degenerate_prep();
    out["degenerate_cooldown"]  = c.degenerate_cooldown();
    out["num_prep_trs"]         = c.num_prep_trs();
    out["num_cooldown_trs"]     = c.num_cooldown_trs();
    out["tr_duration_us"]       = c.tr_duration_us();
    return out;
}

static py::dict _find_segments(_PulseqCollection& pc) {
    const auto& c = pc.coll();
    int nseg = c.num_segments();
    py::dict out;

    py::list segments;
    for (int i = 0; i < nseg; ++i) {
        py::dict seg;
        seg["start_block"] = c.segment_start_block(i);
        seg["num_blocks"]  = c.segment_num_blocks(i);
        segments.append(seg);
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
}
