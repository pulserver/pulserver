/**
 * @file collection.hpp
 * @brief RAII C++11 wrapper around pulseg_collection.
 */

#ifndef PULSEG_COLLECTION_HPP
#define PULSEG_COLLECTION_HPP

#include <cstring>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "pulseg.h"
#include "pulseg_types.h"

#include "error.hpp"
#include "types.hpp"

namespace pulseg
{
    /**
     * Owning wrapper around a pulseg_collection* with RAII lifetime.
     *
     * Movable, not copyable.
     */
    class Collection
    {
    public:
        // ── Construction / lifetime ──────────────────────────────────

        /** Load from one or more in-memory .seq buffers. */
        Collection(const char *const *buffers,
                   const int *sizes,
                   int num_buffers,
                   const Opts &opts,
                   bool parse_labels = true,
                   int num_averages = 1)
        {
            pulseg_opts copts = opts.to_c();
            pulseg_diagnostic diag;
            pulseg_diagnostic_init(&diag);
            int code = pulseg_read_from_buffers(
                &coll_, &diag, buffers, sizes, num_buffers,
                &copts, parse_labels ? 1 : 0, num_averages);
            check(code, diag);
            opts_ = copts;
        }

        /** Load from a .seq file on disk. */
        Collection(const char *file_path,
                   const Opts &opts,
                   bool cache_binary = false,
                   bool verify_signature = false,
                   bool parse_labels = true,
                   int num_averages = 1)
        {
            pulseg_opts copts = opts.to_c();
            pulseg_diagnostic diag;
            pulseg_diagnostic_init(&diag);
            int code = pulseg_read(
                &coll_, &diag, file_path, &copts,
                cache_binary ? 1 : 0,
                verify_signature ? 1 : 0,
                parse_labels ? 1 : 0,
                num_averages);
            check(code, diag);
            opts_ = copts;
        }

        ~Collection()
        {
            if (coll_)
            {
                pulseg_collection_free(coll_);
                coll_ = nullptr;
            }
        }

        // Move-only
        Collection(Collection &&o) noexcept : coll_(o.coll_), opts_(o.opts_)
        {
            o.coll_ = nullptr;
        }
        Collection &operator=(Collection &&o) noexcept
        {
            if (this != &o)
            {
                if (coll_)
                    pulseg_collection_free(coll_);
                coll_ = o.coll_;
                opts_ = o.opts_;
                o.coll_ = nullptr;
            }
            return *this;
        }
        Collection(const Collection &) = delete;
        Collection &operator=(const Collection &) = delete;

        // ── Raw handle (for advanced use) ────────────────────────────

        pulseg_collection *handle() { return coll_; }
        const pulseg_collection *handle() const { return coll_; }
        const pulseg_opts &opts() const { return opts_; }

        // ── Cache (serialization / deserialization) ──────────────────

        /** Save collection to a binary cache file. */
        void save_cache(const std::string &path, int source_size) const
        {
            check(pulseg_save_cache(coll_, path.c_str(), source_size));
        }

        /** Load collection from a binary cache file (mutates this). */
        void load_cache(const std::string &path, int source_size)
        {
            check(pulseg_load_cache(coll_, path.c_str(), source_size));
        }

        // ── Batch info queries ──────────────────────────────────────

        pulseg_collection_info collection_info() const
        {
            pulseg_collection_info info = PULSEG_COLLECTION_INFO_INIT;
            check(pulseg_get_collection_info(coll_, &info));
            return info;
        }

        pulseg_subseq_info subseq_info(int ss = 0) const
        {
            pulseg_subseq_info info = PULSEG_SUBSEQ_INFO_INIT;
            check(pulseg_get_subseq_info(coll_, &info, ss));
            return info;
        }

        pulseg_segment_info segment_info(int seg) const
        {
            pulseg_segment_info info = PULSEG_SEGMENT_INFO_INIT;
            check(pulseg_get_segment_info(coll_, &info, seg));
            return info;
        }

        pulseg_block_info block_info(int seg, int blk) const
        {
            pulseg_block_info info = PULSEG_BLOCK_INFO_INIT;
            check(pulseg_get_block_info(coll_, &info, seg, blk));
            return info;
        }

        pulseg_adc_def adc_def(int adc_idx) const
        {
            pulseg_adc_def def = PULSEG_ADC_DEF_INIT;
            check(pulseg_get_adc_def(coll_, &def, adc_idx));
            return def;
        }

        pulseg_rf_shim_def rf_shim_def(int subseq_idx, int shim_idx) const
        {
            pulseg_rf_shim_def def = PULSEG_RF_SHIM_DEF_INIT;
            check(pulseg_get_rf_shim_def(coll_, &def, subseq_idx, shim_idx));
            return def;
        }

        int num_rf_shims(int subseq_idx) const
        {
            int n = pulseg_get_num_rf_shims(coll_, subseq_idx);
            if (n < 0)
                check(n);
            return n;
        }

        // ── Scan time ────────────────────────────────────────────────

        ScanTimeInfo get_scan_time(int num_reps) const
        {
            pulseg_scan_time_info cinfo = PULSEG_SCAN_TIME_INFO_INIT;
            check(pulseg_get_scan_time(coll_, &cinfo, num_reps));
            return ScanTimeInfo::from_c(cinfo);
        }

        // ── Consistency check ────────────────────────────────────────

        void check_consistency() const
        {
            pulseg_diagnostic diag;
            pulseg_diagnostic_init(&diag);
            check(pulseg_check_consistency(coll_, &diag), diag);
        }

        // ── Segment tables ───────────────────────────────────────────

        std::vector<int> prep_segment_table(int ss = 0) const
        {
            pulseg_subseq_info si = subseq_info(ss);
            std::vector<int> ids(si.num_prep_segments);
            if (si.num_prep_segments > 0)
                pulseg_get_prep_segment_table(coll_, ss, ids.data());
            return ids;
        }
        std::vector<int> main_segment_table(int ss = 0) const
        {
            pulseg_subseq_info si = subseq_info(ss);
            std::vector<int> ids(si.num_main_segments);
            if (si.num_main_segments > 0)
                pulseg_get_main_segment_table(coll_, ss, ids.data());
            return ids;
        }
        std::vector<int> cooldown_segment_table(int ss = 0) const
        {
            pulseg_subseq_info si = subseq_info(ss);
            std::vector<int> ids(si.num_cooldown_segments);
            if (si.num_cooldown_segments > 0)
                pulseg_get_cooldown_segment_table(coll_, ss, ids.data());
            return ids;
        }

        // ── RF queries (waveform access – still individual) ─────────

        RfStats get_rf_stats(int ss, int rf_idx) const
        {
            pulseg_rf_stats cstats = PULSEG_RF_STATS_INIT;
            check(pulseg_get_rf_stats(coll_, &cstats, ss, rf_idx));
            return RfStats::from_c(cstats);
        }

        std::vector<int> tr_rf_ids(int ss = 0) const
        {
            pulseg_subseq_info si = subseq_info(ss);
            std::vector<int> ids(si.tr_size, -1);
            pulseg_get_tr_rf_ids(coll_, ids.data(), ss);
            return ids;
        }

        // ── Gradient waveform queries (still individual) ─────────────

        float grad_initial_amplitude(int seg, int blk, int axis) const
        {
            return pulseg_get_grad_initial_amplitude_hz_per_m(coll_, seg, blk, axis);
        }
        int grad_initial_shot_id(int seg, int blk, int axis) const
        {
            return pulseg_get_grad_initial_shot_id(coll_, seg, blk, axis);
        }

        // ── Label queries ────────────────────────────────────────────

        LabelLimits get_label_limits(int ss = 0) const
        {
            pulseg_label_limits cl;
            check(pulseg_get_label_limits(coll_, &cl, ss));
            return LabelLimits::from_c(cl);
        }

        std::vector<int> get_adc_label(int ss, int occurrence) const
        {
            pulseg_subseq_info si = subseq_info(ss);
            std::vector<int> vals(si.num_label_columns);
            check(pulseg_get_adc_label(coll_, vals.data(), ss, occurrence));
            return vals;
        }

        // ── TR gradient waveforms ────────────────────────────────────

        TrGradientWaveforms get_tr_gradient_waveforms(int subseq_idx = 0, int canonical_tr_idx = 0) const
        {
            pulseg_tr_gradient_waveforms cw = PULSEG_TR_GRADIENT_WAVEFORMS_INIT;
            pulseg_diagnostic diag;
            pulseg_diagnostic_init(&diag);
            int code = pulseg_get_tr_gradient_waveforms(coll_, &cw, &diag, subseq_idx, canonical_tr_idx);
            check(code, diag);

            TrGradientWaveforms w;
            auto copy_axis = [](GradAxisWaveform &dst, const pulseg_grad_axis_waveform &src)
            {
                dst.time_us.assign(src.time_us, src.time_us + src.num_samples);
                dst.amplitude_hz_per_m.assign(src.amplitude_hz_per_m, src.amplitude_hz_per_m + src.num_samples);
                dst.seg_label.resize(src.num_samples);
                for (int i = 0; i < src.num_samples; ++i)
                    dst.seg_label[i] = static_cast<float>(src.seg_label[i]);
            };
            copy_axis(w.gx, cw.gx);
            copy_axis(w.gy, cw.gy);
            copy_axis(w.gz, cw.gz);

            pulseg_tr_gradient_waveforms_free(&cw);
            return w;
        }

        // ── Native-timing TR waveforms (for plotting) ────────────────

        TrWaveforms get_tr_waveforms(
            int ss = 0,
            int amplitude_mode = PULSEG_AMP_MAX_POS,
            int tr_index = 0,
            bool collapse_delays = false,
            int num_averages = 0) const
        {
            pulseg_tr_waveforms cw;
            pulseg_diagnostic diag;
            memset(&cw, 0, sizeof(cw));
            pulseg_diagnostic_init(&diag);

            int code = pulseg_get_tr_waveforms(coll_,
                                                  &cw,
                                                  &diag,
                                                  ss,
                                                  amplitude_mode,
                                                  tr_index,
                                                  collapse_delays ? 1 : 0,
                                                  num_averages);
            check(code, diag);

            TrWaveforms w;
            auto copy_ch = [](ChannelWaveform &dst, const pulseg_channel_waveform &src)
            {
                if (src.num_samples > 0 && src.time_us && src.amplitude)
                {
                    dst.time_us.assign(src.time_us, src.time_us + src.num_samples);
                    dst.amplitude.assign(src.amplitude, src.amplitude + src.num_samples);
                }
            };
            copy_ch(w.gx, cw.gx);
            copy_ch(w.gy, cw.gy);
            copy_ch(w.gz, cw.gz);
            copy_ch(w.rf_mag, cw.rf_mag);
            copy_ch(w.rf_phase, cw.rf_phase);
            w.num_rf_channels = cw.num_rf_channels;

            w.adc_events.resize(static_cast<size_t>(cw.num_adc_events));
            for (int i = 0; i < cw.num_adc_events; ++i)
            {
                AdcEvent &a = w.adc_events[static_cast<size_t>(i)];
                a.onset_us = cw.adc_events[i].onset_us;
                a.duration_us = cw.adc_events[i].duration_us;
                a.num_samples = cw.adc_events[i].num_samples;
                a.freq_offset_hz = cw.adc_events[i].freq_offset_hz;
                a.phase_offset_rad = cw.adc_events[i].phase_offset_rad;
            }

            w.blocks.resize(static_cast<size_t>(cw.num_blocks));
            for (int i = 0; i < cw.num_blocks; ++i)
            {
                TrBlockDescriptor &b = w.blocks[static_cast<size_t>(i)];
                b.start_us = cw.blocks[i].start_us;
                b.duration_us = cw.blocks[i].duration_us;
                b.segment_idx = cw.blocks[i].segment_idx;
                b.rf_isocenter_us = cw.blocks[i].rf_isocenter_us;
                b.adc_kzero_us = cw.blocks[i].adc_kzero_us;
            }

            w.total_duration_us = cw.total_duration_us;

            pulseg_tr_waveforms_free(&cw);
            return w;
        }

        // ── Mechanical resonances spectra ─────────────────────────────────────────

        MechResonancesSpectra calc_mech_resonances(
            int ss,
            int canonical_tr_idx,
            float target_resolution_hz,
            float max_freq_hz,
            const std::vector<ForbiddenBand> &bands = {},
            float peak_log10_threshold = std::numeric_limits<float>::quiet_NaN(),
            float peak_norm_scale = std::numeric_limits<float>::quiet_NaN(),
            float peak_eps = std::numeric_limits<float>::quiet_NaN(),
            float peak_prominence = std::numeric_limits<float>::quiet_NaN()) const
        {
            std::vector<pulseg_forbidden_band> cbands(bands.size());
            for (size_t i = 0; i < bands.size(); ++i)
                cbands[i] = bands[i].to_c();

            pulseg_opts run_opts = opts_;
            if (!std::isnan(peak_log10_threshold))
                run_opts.peak_log10_threshold = peak_log10_threshold;
            if (!std::isnan(peak_norm_scale))
                run_opts.peak_norm_scale = peak_norm_scale;
            if (!std::isnan(peak_eps))
                run_opts.peak_eps = peak_eps;
            if (!std::isnan(peak_prominence))
                run_opts.peak_prominence = peak_prominence;

            pulseg_mech_resonances_spectra cs = PULSEG_MECH_RESONANCES_SPECTRA_INIT;
            pulseg_diagnostic diag;
            pulseg_diagnostic_init(&diag);
            int code = pulseg_calc_mech_resonances(coll_,
                                                      &cs,
                                                      &diag,
                                                      ss,
                                                      canonical_tr_idx,
                                                      &run_opts,
                                                      target_resolution_hz,
                                                      max_freq_hz,
                                                      static_cast<int>(cbands.size()),
                                                      cbands.empty() ? nullptr : cbands.data());
            check(code, diag);

            MechResonancesSpectra a;
            a.freq_min_hz = cs.freq_min_hz;
            a.freq_spacing_hz = cs.freq_spacing_hz;
            a.num_freq_bins = cs.num_freq_bins;

            auto assign_f = [](std::vector<float> &v, const float *p, int n)
            { if (p) v.assign(p, p + n); };
            auto assign_i = [](std::vector<int> &v, const int *p, int n)
            { if (p) v.assign(p, p + n); };

            assign_f(a.spectrum_full_gx, cs.spectrum_full_gx, cs.num_freq_bins);
            assign_f(a.spectrum_full_gy, cs.spectrum_full_gy, cs.num_freq_bins);
            assign_f(a.spectrum_full_gz, cs.spectrum_full_gz, cs.num_freq_bins);

            a.num_instances = cs.num_instances;

            a.num_analytical_peaks = cs.num_analytical_peaks;
            assign_f(a.analytical_peak_freqs, cs.analytical_peak_freqs, cs.num_analytical_peaks);
            assign_f(a.analytical_peak_amp_gx, cs.analytical_peak_amp_gx, cs.num_analytical_peaks);
            assign_f(a.analytical_peak_amp_gy, cs.analytical_peak_amp_gy, cs.num_analytical_peaks);
            assign_f(a.analytical_peak_amp_gz, cs.analytical_peak_amp_gz, cs.num_analytical_peaks);
            assign_f(a.analytical_peak_phase_gx, cs.analytical_peak_phase_gx, cs.num_analytical_peaks);
            assign_f(a.analytical_peak_phase_gy, cs.analytical_peak_phase_gy, cs.num_analytical_peaks);
            assign_f(a.analytical_peak_phase_gz, cs.analytical_peak_phase_gz, cs.num_analytical_peaks);
            assign_f(a.analytical_peak_widths_hz, cs.analytical_peak_widths_hz, cs.num_analytical_peaks);
            a.num_candidates = cs.num_candidates;
            assign_f(a.candidate_freqs, cs.candidate_freqs, cs.num_candidates);
            assign_f(a.candidate_amps_gx, cs.candidate_amps_gx, cs.num_candidates);
            assign_f(a.candidate_amps_gy, cs.candidate_amps_gy, cs.num_candidates);
            assign_f(a.candidate_amps_gz, cs.candidate_amps_gz, cs.num_candidates);
            assign_f(a.candidate_grad_amps, cs.candidate_grad_amps, cs.num_candidates);
            assign_f(a.candidate_grad_amps_gx, cs.candidate_grad_amps_gx, cs.num_candidates);
            assign_f(a.candidate_grad_amps_gy, cs.candidate_grad_amps_gy, cs.num_candidates);
            assign_f(a.candidate_grad_amps_gz, cs.candidate_grad_amps_gz, cs.num_candidates);
            assign_i(a.candidate_violations, cs.candidate_violations, cs.num_candidates);

            a.num_component_terms = cs.num_component_terms;
            assign_f(a.component_freqs_hz, cs.component_freqs_hz, cs.num_component_terms);
            assign_f(a.component_amps, cs.component_amps, cs.num_component_terms);
            assign_f(a.component_phases_rad, cs.component_phases_rad, cs.num_component_terms);
            assign_f(a.component_widths_hz, cs.component_widths_hz, cs.num_component_terms);
            assign_i(a.component_axes, cs.component_axes, cs.num_component_terms);
            assign_i(a.component_def_ids, cs.component_def_ids, cs.num_component_terms);
            assign_i(a.component_contrib_ids, cs.component_contrib_ids, cs.num_component_terms);
            assign_i(a.component_run_ids, cs.component_run_ids, cs.num_component_terms);

            a.num_surviving_freqs = cs.num_surviving_freqs;
            assign_f(a.surviving_freqs_hz, cs.surviving_freqs_hz, cs.num_surviving_freqs);

            pulseg_mech_resonances_spectra_free(&cs);
            return a;
        }

        // ── PNS computation ──────────────────────────────────────────

        PnsResult calc_pns(int ss, int canonical_tr_idx, const PnsParams &params) const
        {
            pulseg_pns_params cp = params.to_c();
            pulseg_pns_result cr = PULSEG_PNS_RESULT_INIT;
            pulseg_diagnostic diag;
            pulseg_diagnostic_init(&diag);
            int code = pulseg_calc_pns(coll_, &cr, &diag, ss, canonical_tr_idx, &opts_, &cp);
            check(code, diag);

            PnsResult r;
            r.num_samples = cr.num_samples;
            if (cr.slew_x_hz_per_m_per_s)
                r.slew_x.assign(cr.slew_x_hz_per_m_per_s, cr.slew_x_hz_per_m_per_s + cr.num_samples);
            if (cr.slew_y_hz_per_m_per_s)
                r.slew_y.assign(cr.slew_y_hz_per_m_per_s, cr.slew_y_hz_per_m_per_s + cr.num_samples);
            if (cr.slew_z_hz_per_m_per_s)
                r.slew_z.assign(cr.slew_z_hz_per_m_per_s, cr.slew_z_hz_per_m_per_s + cr.num_samples);

            pulseg_pns_result_free(&cr);
            return r;
        }

        // ── Safety check ─────────────────────────────────────────────

        void check_safety(
            const std::vector<ForbiddenBand> &bands = {},
            const PnsParams *pns_params = nullptr,
            float pns_threshold_percent = 100.0f) const
        {
            std::vector<pulseg_forbidden_band> cbands(bands.size());
            for (size_t i = 0; i < bands.size(); ++i)
                cbands[i] = bands[i].to_c();

            pulseg_pns_params cp;
            const pulseg_pns_params *cpp = nullptr;
            if (pns_params)
            {
                cp = pns_params->to_c();
                cpp = &cp;
            }

            pulseg_diagnostic diag;
            pulseg_diagnostic_init(&diag);
            // Note: check_safety takes non-const coll for cursor dry-run
            int code = pulseg_check_safety(
                coll_, &diag, &opts_,
                static_cast<int>(cbands.size()),
                cbands.empty() ? nullptr : cbands.data(),
                cpp, pns_threshold_percent);
            check(code, diag);
        }

        // ── Block cursor ─────────────────────────────────────────────

        int cursor_next() { return pulseg_cursor_next(coll_); }
        void cursor_rewind() { pulseg_cursor_rewind(coll_); }

        BlockInstance get_block_instance() const
        {
            pulseg_block_instance ci = PULSEG_BLOCK_INSTANCE_INIT;
            check(pulseg_get_block_instance(coll_, &ci));
            return BlockInstance::from_c(ci);
        }

        // ── Unique-block and segment-block queries ───────────────────

        /** Number of unique block definitions in subsequence @p ss. */
        int num_unique_blocks(int ss = 0) const
        {
            int n = pulseg_get_num_unique_blocks(coll_, ss);
            if (n < 0)
                check(n);
            return n;
        }

        /** 1-based .seq block ID for the @p blk_def_idx-th unique block. */
        int unique_block_id(int ss, int blk_def_idx) const
        {
            int id = pulseg_get_unique_block_id(coll_, ss, blk_def_idx);
            if (id < 0)
                check(id);
            return id;
        }

        /** Unique-block-definition indices for a global segment. */
        std::vector<int> segment_block_def_indices(int seg_idx) const
        {
            auto si = segment_info(seg_idx);
            std::vector<int> ids(si.num_blocks);
            if (si.num_blocks > 0)
            {
                int rc = pulseg_get_segment_block_def_indices(coll_, seg_idx, ids.data());
                if (rc < 0)
                    check(rc);
            }
            return ids;
        }

        // ── Frequency modulation ─────────────────────────────────────

        int freq_mod_count() const
        {
            return pulseg_get_freq_mod_count(coll_);
        }
        int freq_mod_count_tr(int tr_type, int tr_index) const
        {
            return pulseg_get_freq_mod_count_tr(coll_, tr_type, tr_index);
        }

    private:
        pulseg_collection *coll_ = nullptr;
        pulseg_opts opts_ = PULSEG_OPTS_INIT;
    };

} // namespace pulseg

#endif // PULSEG_COLLECTION_HPP
