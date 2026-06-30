/**
 * @file _pulseqlib_wrapper.cpp
 * @brief Thin pybind11 binding for pulseqlib C++11 interface.
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

#include "pulseqlib.hpp"
#include "pulseqlib_methods.h"
#include "pulseqlib_types.h"
#include "pulseqlib_internal.h"

namespace py = pybind11;

// ─── Thin holder for the C++ Collection ─────────────────────────────

class _PulseqCollection
{
public:
    _PulseqCollection(const py::list &seq_bytes_list,
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
        pulseqlib::Opts opts;
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

        coll_ = std::unique_ptr<pulseqlib::Collection>(
            new pulseqlib::Collection(
                buf_ptrs.data(), buf_sizes.data(), n,
                opts, parse_labels, num_averages));
        source_size_ = (n > 0) ? buf_sizes[0] : 0;
    }

    /** Load from a .seq file path, using the .pge cache when present. */
    _PulseqCollection(const std::string &file_path,
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
        pulseqlib::Opts opts;
        opts.gamma_hz_per_t = gamma;
        opts.b0_t = B0;
        opts.max_grad_hz_per_m = max_grad;
        opts.max_slew_hz_per_m_per_s = max_slew;
        opts.rf_raster_us = rf_raster_time * 1e6f;
        opts.grad_raster_us = grad_raster_time * 1e6f;
        opts.adc_raster_us = adc_raster_time * 1e6f;
        opts.block_raster_us = block_duration_raster * 1e6f;

        coll_ = std::unique_ptr<pulseqlib::Collection>(
            new pulseqlib::Collection(
                file_path.c_str(), opts,
                /*cache_binary=*/true,
                /*verify_signature=*/false,
                parse_labels, num_averages));
        source_size_ = 0;
    }

    pulseqlib::Collection &coll() { return *coll_; }
    const pulseqlib::Collection &coll() const { return *coll_; }
    int source_size() const { return source_size_; }

private:
    std::unique_ptr<pulseqlib::Collection> coll_;
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

static py::dict _get_tr_waveforms(
    _PulseqCollection &pc,
    int subsequence_idx,
    int amplitude_mode,
    int tr_index,
    bool collapse_delays,
    int num_averages)
{
    auto wf = pc.coll().get_tr_waveforms(
        subsequence_idx, amplitude_mode, tr_index, collapse_delays,
        num_averages);
    py::dict out;

    auto ch_to_dict = [](const pulseqlib::ChannelWaveform &ch) -> py::dict
    {
        py::dict d;
        d["time_us"] = ch.time_us;
        d["amplitude"] = ch.amplitude;
        return d;
    };
    out["gx"] = ch_to_dict(wf.gx);
    out["gy"] = ch_to_dict(wf.gy);
    out["gz"] = ch_to_dict(wf.gz);
    out["rf_mag"] = ch_to_dict(wf.rf_mag);
    out["rf_phase"] = ch_to_dict(wf.rf_phase);
    out["num_rf_channels"] = wf.num_rf_channels;

    // ADC events
    py::list adc_list;
    for (const auto &a : wf.adc_events)
    {
        py::dict ad;
        ad["onset_us"] = a.onset_us;
        ad["duration_us"] = a.duration_us;
        ad["num_samples"] = a.num_samples;
        ad["freq_offset_hz"] = a.freq_offset_hz;
        ad["phase_offset_rad"] = a.phase_offset_rad;
        adc_list.append(ad);
    }
    out["adc_events"] = adc_list;

    // Block descriptors
    py::list blk_list;
    for (const auto &b : wf.blocks)
    {
        py::dict bd;
        bd["start_us"] = b.start_us;
        bd["duration_us"] = b.duration_us;
        bd["segment_idx"] = b.segment_idx;
        bd["rf_isocenter_us"] = b.rf_isocenter_us;
        bd["adc_kzero_us"] = b.adc_kzero_us;
        blk_list.append(bd);
    }
    out["blocks"] = blk_list;
    out["total_duration_us"] = wf.total_duration_us;

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
    std::vector<pulseqlib::ForbiddenBand> bands;
    for (auto item : py_bands)
    {
        py::tuple t = item.cast<py::tuple>();
        pulseqlib::ForbiddenBand b;
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
    pulseqlib::PnsParams params;
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
    std::vector<pulseqlib::ForbiddenBand> bands;
    for (auto item : py_bands)
    {
        py::tuple t = item.cast<py::tuple>();
        pulseqlib::ForbiddenBand b;
        b.freq_min_hz = t[0].cast<float>();
        b.freq_max_hz = t[1].cast<float>();
        b.max_amplitude_hz_per_m = t[2].cast<float>();
        bands.push_back(b);
    }

    const pulseqlib::PnsParams *pns_ptr = nullptr;
    pulseqlib::PnsParams pns;
    if (!skip_pns)
    {
        pns.chronaxie_us = decay_constant_us;
        pns.rheobase_hz_per_m_per_s = stim_threshold; // rheobase/alpha combined
        pns.alpha = 1.0f;                             // folded into stim_threshold
        pns_ptr = &pns;
    }

    pc.coll().check_safety(bands, pns_ptr, pns_threshold_percent);
}

// ─── Report (collection + subseq + segment info) ───────────────────

static py::dict _get_report(_PulseqCollection &pc)
{
    const auto &c = pc.coll();
    py::dict out;

    /* Collection-level */
    auto ci = c.collection_info();
    out["num_subsequences"] = ci.num_subsequences;
    out["num_segments"] = ci.num_segments;
    out["total_duration_us"] = ci.total_duration_us;

    /* Per-subsequence */
    py::list subseqs;
    for (int ss = 0; ss < ci.num_subsequences; ++ss)
    {
        auto si = c.subseq_info(ss);
        py::dict sd;
        sd["tr_size"] = si.tr_size;
        sd["num_trs"] = si.num_trs;
        sd["num_prep_blocks"] = si.num_prep_blocks;
        sd["num_cooldown_blocks"] = si.num_cooldown_blocks;
        sd["tr_duration_us"] = si.tr_duration_us;
        sd["num_passes"] = si.num_passes;
        sd["num_unique_segments"] = si.num_prep_segments + si.num_main_segments + si.num_cooldown_segments;
        sd["segment_offset"] = si.segment_offset;

        /* Unique segments in this subsequence */
        int seg_start = si.segment_offset;
        int seg_count = si.num_prep_segments + si.num_main_segments + si.num_cooldown_segments;
        py::list segs;
        for (int j = seg_start; j < seg_start + seg_count; ++j)
        {
            auto seg = c.segment_info(j);
            py::dict segd;
            segd["start_block"] = seg.start_block;
            segd["num_blocks"] = seg.num_blocks;
            segs.append(segd);
        }
        sd["segments"] = segs;

        /* Segment tables */
        sd["prep_segment_table"] = c.prep_segment_table(ss);
        sd["main_segment_table"] = c.main_segment_table(ss);
        sd["cooldown_segment_table"] = c.cooldown_segment_table(ss);

        subseqs.append(sd);
    }
    out["subsequences"] = subseqs;
    return out;
}

// ─── Unique-block queries ────────────────────────────────────────────

static int _get_num_unique_blocks(_PulseqCollection &pc, int seq_idx)
{
    return pc.coll().num_unique_blocks(seq_idx);
}

static int _get_unique_block_id(_PulseqCollection &pc, int seq_idx, int blk_def_idx)
{
    return pc.coll().unique_block_id(seq_idx, blk_def_idx);
}

// ─── Sequence description / parameters ─────────────────────────────

static py::dict _get_sequence_parameters(_PulseqCollection &pc)
{
    pulseqlib_sequence_parameters sp;
    std::memset(&sp, 0, sizeof(sp));
    int rc = pulseqlib_get_sequence_parameters(&sp, pc.coll().handle());
    if (rc != PULSEQLIB_SUCCESS)
    {
        throw std::runtime_error(
            "pulseqlib_get_sequence_parameters failed: " + std::to_string(rc));
    }
    py::dict result;
    result["min_te_us"] = sp.min_te_us;
    result["min_tr_us"] = sp.min_tr_us;
    result["max_tr_us"] = sp.max_tr_us;
    result["max_flip_angle_deg"] = sp.max_flip_angle_deg * (180.0f / static_cast<float>(M_PI));
    result["total_scan_time_us"] = sp.total_scan_time_us;
    result["num_subseqs"] = sp.num_subseqs;
    return result;
}

static py::dict _get_sequence_description(_PulseqCollection &pc, int subseq_idx)
{
    pulseqlib_sequence_description sd;
    std::memset(&sd, 0, sizeof(sd));
    int rc = pulseqlib_get_sequence_description(&sd, pc.coll().handle(), subseq_idx);
    if (rc != PULSEQLIB_SUCCESS)
    {
        pulseqlib_free_sequence_description(&sd);
        throw std::runtime_error(
            "pulseqlib_get_sequence_description failed: " + std::to_string(rc));
    }

    py::dict result;
    result["subseq_idx"] = sd.subseq_idx;
    result["tr_duration_us"] = sd.tr_duration_us;

    // ── Compact event table (one event per pass block) ─────────────
    // params[0] = timestamp_us; params[1..N] = type-specific event params.
    py::list events;
    for (int i = 0; i < sd.num_rows; ++i)
    {
        const pulseqlib_seq_event *row = &sd.rows[i];
        std::vector<float> p;
        p.reserve(1 + PULSEQLIB_SEQ_EVENT_PARAMS);
        p.push_back(row->timestamp_us);
        p.insert(p.end(), row->params, row->params + PULSEQLIB_SEQ_EVENT_PARAMS);
        py::dict rd;
        rd["type"] = row->type;
        rd["params"] = p;
        events.append(rd);
    }
    result["events"] = events;

    // ── RF shape tuples (one per unique RF definition) ─────────────
    // Build a map: rf_def_idx -> (global_seg_idx, blk_pos) for waveform retrieval.
    // Only need any single occurrence of each unique RF.
    {
        const pulseqlib_collection *coll_ptr = pc.coll().handle();
        if (subseq_idx >= 0 && subseq_idx < coll_ptr->num_subsequences)
        {
            const pulseqlib_sequence_descriptor *desc =
                &coll_ptr->descriptors[subseq_idx];

            // Compute global segment offset for this subsequence
            int seg_offset = 0;
            for (int i = 0; i < subseq_idx; ++i)
                seg_offset += coll_ptr->descriptors[i].num_unique_segments;

            // For each unique RF definition, find the first (seg, blk) that uses it
            std::vector<int> rf_seg(desc->num_unique_rfs, -1);
            std::vector<int> rf_blk(desc->num_unique_rfs, -1);
            for (int si = 0; si < desc->segment_table.num_unique_segments; ++si)
            {
                const pulseqlib_tr_segment *seg = &desc->segment_definitions[si];
                for (int bi = 0; bi < seg->num_blocks; ++bi)
                {
                    int bdef_idx = seg->unique_block_indices[bi];
                    int rf_id = desc->block_definitions[bdef_idx].rf_id;
                    if (rf_id >= 0 && rf_id < desc->num_unique_rfs &&
                        rf_seg[rf_id] < 0)
                    {
                        rf_seg[rf_id] = seg_offset + si;
                        rf_blk[rf_id] = bi;
                    }
                }
            }

            py::list rft_list;
            for (int rf_id = 0; rf_id < desc->num_unique_rfs; ++rf_id)
            {
                if (rf_seg[rf_id] < 0)
                    continue; // RF definition exists but not used in any segment block

                pulseqlib_rf_stats stats;
                int rc2 = pulseqlib_get_rf_stats(
                    coll_ptr, &stats, subseq_idx, rf_id);
                if (rc2 != PULSEQLIB_SUCCESS)
                    continue;

                // Magnitude waveform
                int nch = 0, npts = 0;
                float **mag_arr = pulseqlib_get_rf_magnitude(
                    coll_ptr, &nch, &npts, rf_seg[rf_id], rf_blk[rf_id]);

                // Phase waveform
                int nch_ph = 0, npts_ph = 0;
                float **phase_arr = pulseqlib_get_rf_phase(
                    coll_ptr, &nch_ph, &npts_ph, rf_seg[rf_id], rf_blk[rf_id]);

                // Time axis
                float *time_arr = pulseqlib_get_rf_time_us(
                    coll_ptr, rf_seg[rf_id], rf_blk[rf_id]);

                // Flatten magnitude: [ch0_s0, ch0_s1, ..., ch1_s0, ...]
                std::vector<float> mag_flat, phase_flat, time_flat;
                if (mag_arr && nch > 0)
                {
                    for (int c = 0; c < nch; ++c)
                    {
                        if (npts > 0)
                        {
                            mag_flat.insert(mag_flat.end(),
                                            mag_arr[c], mag_arr[c] + npts);
                        }
                        PULSEQLIB_FREE(mag_arr[c]);
                    }
                    PULSEQLIB_FREE(mag_arr);
                }
                if (phase_arr && nch_ph > 0)
                {
                    for (int c = 0; c < nch_ph; ++c)
                    {
                        if (npts_ph > 0)
                        {
                            phase_flat.insert(phase_flat.end(),
                                              phase_arr[c], phase_arr[c] + npts_ph);
                        }
                        PULSEQLIB_FREE(phase_arr[c]);
                    }
                    PULSEQLIB_FREE(phase_arr);
                }
                if (time_arr && npts > 0)
                {
                    time_flat.assign(time_arr, time_arr + npts);
                    PULSEQLIB_FREE(time_arr);
                }

                // band_freq_offsets: trim to num_bands entries
                int nb = (stats.num_bands > 0) ? stats.num_bands : 1;
                std::vector<float> freq_offsets(
                    stats.band_freq_offsets_hz,
                    stats.band_freq_offsets_hz + nb);

                float rf_raster_us_val = desc->rf_raster_us;

                py::dict t;
                t["tuple_id"] = rf_id;
                t["N_tx"] = (nch > 0) ? nch : 1;
                t["N_samples"] = npts;
                t["rf_raster_us"] = rf_raster_us_val;
                t["num_bands"] = stats.num_bands;
                t["band_freq_offsets_hz"] = freq_offsets;
                t["band_bandwidth_hz"] = stats.band_bandwidth_hz;
                t["total_b1sq_power"] = stats.total_b1sq_power;
                t["mag"] = mag_flat;
                t["phase"] = phase_flat;
                t["time"] = time_flat;
                rft_list.append(t);
            }
            result["rf_shape_tuples"] = rft_list;
        }
    }

    pulseqlib_free_sequence_description(&sd);
    return result;
}

// ─── Module ─────────────────────────────────────────────────────────

PYBIND11_MODULE(_pulseqlib_wrapper, m)
{
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
        .def(py::init<std::string, float, float, float, float,
                      float, float, float, float, bool, int>(),
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

    m.def("_find_tr", &_find_tr,
          py::arg("collection"),
          py::arg("subsequence_idx") = 0);

    m.def("_get_tr_waveforms", &_get_tr_waveforms,
          py::arg("collection"),
          py::arg("subsequence_idx") = 0,
          py::arg("amplitude_mode") = 0,
          py::arg("tr_index") = 0,
          py::arg("collapse_delays") = false,
          py::arg("num_averages") = 0);

    m.def("_calc_mech_resonances", &_calc_mech_resonances,
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

    m.def("_calc_pns", &_calc_pns,
          py::arg("collection"),
          py::arg("subsequence_idx") = 0,
          py::arg("canonical_tr_idx") = 0,
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

    m.def("_get_sequence_parameters", &_get_sequence_parameters,
          py::arg("collection"));

    m.def("_get_sequence_description", &_get_sequence_description,
          py::arg("collection"),
          py::arg("subseq_idx") = 0);
}
