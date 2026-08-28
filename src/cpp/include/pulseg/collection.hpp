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
     * Owning wrapper around a pulseg_check_plan*.
     *
     * The preprocessing the PNS and mechanical-resonance checks would
     * otherwise each pay for. Hand the same plan to several questions about
     * one scan and the work behind them happens once; hand none and each
     * call keeps its own, which is what the defaulted arguments below do.
     *
     * A plan must not outlive the Collection it was made from.
     */
    class CheckPlan
    {
    public:
        CheckPlan(const pulseg_collection* coll, int cache_budget_kb = 0)
        {
            pulseg_check_plan_config config = PULSEG_CHECK_PLAN_CONFIG_INIT;
            config.cache_budget_kb = cache_budget_kb;
            pulseg_diagnostic diag;
            pulseg_diagnostic_init(&diag);
            check(pulseg_check_plan_create(&plan_, &diag, coll, &config), diag);
        }

        ~CheckPlan()
        {
            pulseg_check_plan_destroy(plan_);
        }

        CheckPlan(CheckPlan&& o) noexcept : plan_(o.plan_)
        {
            o.plan_ = nullptr;
        }
        CheckPlan& operator=(CheckPlan&& o) noexcept
        {
            if (this != &o)
            {
                pulseg_check_plan_destroy(plan_);
                plan_ = o.plan_;
                o.plan_ = nullptr;
            }
            return *this;
        }
        CheckPlan(const CheckPlan&) = delete;
        CheckPlan& operator=(const CheckPlan&) = delete;

        pulseg_check_plan* handle() const
        {
            return plan_;
        }

    private:
        pulseg_check_plan* plan_ = nullptr;
    };

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
        Collection(
            const char* const* buffers,
            const int* sizes,
            int num_buffers,
            const Opts& opts,
            bool parse_labels = true)
        {
            pulseg_opts copts = opts.to_c();
            pulseg_diagnostic diag;
            pulseg_diagnostic_init(&diag);
            int code = pulseg_read_from_buffers(
                &coll_,
                &diag,
                buffers,
                sizes,
                num_buffers,
                &copts,
                parse_labels ? 1 : 0);
            check(code, diag);
            opts_ = copts;
        }

        /** Load from a .seq file on disk. */
        Collection(
            const char* file_path,
            const Opts& opts,
            bool cache_binary = false,
            bool verify_signature = false,
            bool parse_labels = true)
        {
            pulseg_opts copts = opts.to_c();
            pulseg_diagnostic diag;
            pulseg_diagnostic_init(&diag);
            int code = pulseg_read(
                &coll_,
                &diag,
                file_path,
                &copts,
                cache_binary ? 1 : 0,
                verify_signature ? 1 : 0,
                parse_labels ? 1 : 0);
            check(code, diag);
            opts_ = copts;
        }

        /** Adopt an already-loaded C collection. Takes ownership. */
        Collection(pulseg_collection* coll, const pulseg_opts& opts) : coll_(coll), opts_(opts)
        {
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
        Collection(Collection&& o) noexcept : coll_(o.coll_), opts_(o.opts_)
        {
            o.coll_ = nullptr;
        }
        Collection& operator=(Collection&& o) noexcept
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
        Collection(const Collection&) = delete;
        Collection& operator=(const Collection&) = delete;

        // ── Raw handle (for advanced use) ────────────────────────────

        pulseg_collection* handle()
        {
            return coll_;
        }
        const pulseg_collection* handle() const
        {
            return coll_;
        }
        const pulseg_opts& opts() const
        {
            return opts_;
        }

        // ── Cache (serialization / deserialization) ──────────────────

        /**
         * Load the COMMON and SHAPES sections of the cache beside `seq_path`.
         *
         * Neither the per-instance tables nor the execution stream, which are
         * the sections that scale with the length of the scan.
         */
        static Collection from_geninstructions_cache(const std::string& seq_path, const Opts& opts)
        {
            pulseg_collection* coll = nullptr;
            check(pulseg_load_geninstructions_cache(&coll, seq_path.c_str()));
            return Collection(coll, opts.to_c());
        }

        /**
         * Load the COMMON, INSTANCES, ROTATIONS, SHAPES and SCANLOOP sections
         * of the cache beside `seq_path`.
         */
        static Collection from_scanloop_cache(const std::string& seq_path, const Opts& opts)
        {
            pulseg_collection* coll = nullptr;
            check(pulseg_load_scanloop_cache(&coll, seq_path.c_str()));
            return Collection(coll, opts.to_c());
        }

        /**
         * Write the cache beside the .seq the collection was read from.
         *
         * The cache path comes from `seq_path` and `opts.cache_ext`, and the
         * integrity size is read off the .seq itself -- so this pairs with
         * the loaders, which find the cache the same way.
         */
        void save_cache(const std::string& seq_path)
        {
            check(pulseg_save_cache(coll_, seq_path.c_str(), &opts_));
        }

        /**
         * Save to an explicitly named cache file, for a caller keeping its
         * cache somewhere other than beside the sequence.
         */
        void save_cache_to_path(const std::string& path, int source_size) const
        {
            check(pulseg_save_cache_to_path(coll_, path.c_str(), source_size));
        }

        /** Load collection from a binary cache file (mutates this). */
        void load_cache(const std::string& path, int source_size)
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

        ScanTimeInfo get_scan_time() const
        {
            pulseg_scan_time_info cinfo = PULSEG_SCAN_TIME_INFO_INIT;
            check(pulseg_get_scan_time(coll_, &cinfo));
            return ScanTimeInfo::from_c(cinfo);
        }

        // ── Consistency check ────────────────────────────────────────

        void check_consistency() const
        {
            pulseg_diagnostic diag;
            pulseg_diagnostic_init(&diag);
            check(pulseg_check_consistency(coll_, &diag), diag);
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
        int grad_initial_shape_id(int seg, int blk, int axis) const
        {
            return pulseg_get_grad_initial_shape_id(coll_, seg, blk, axis);
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

        TrGradientWaveforms get_tr_gradient_waveforms(int subseq_idx = 0, int canonical_tr_idx = 0)
            const
        {
            pulseg_tr_gradient_waveforms cw = PULSEG_TR_GRADIENT_WAVEFORMS_INIT;
            pulseg_diagnostic diag;
            pulseg_diagnostic_init(&diag);
            int code =
                pulseg_get_tr_gradient_waveforms(coll_, &cw, &diag, subseq_idx, canonical_tr_idx);
            check(code, diag);

            TrGradientWaveforms w;
            auto copy_axis = [](GradAxisWaveform& dst, const pulseg_grad_axis_waveform& src)
            {
                dst.time_us.assign(src.time_us, src.time_us + src.num_samples);
                dst.amplitude_hz_per_m.assign(
                    src.amplitude_hz_per_m,
                    src.amplitude_hz_per_m + src.num_samples);
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

        // ── Sequence description (state-machine event rows) ──────────

        /**
         * One subsequence's canonical-TR event table: one row per block,
         * carrying RF use / amplitude / phase / frequency, ADC role and
         * k-space-zero timing, or nothing for a block that plays neither.
         *
         * This is the same table the .pge cache's SEQDESC section stores and
         * the reconstruction reads back, produced here from a loaded
         * collection rather than from a file -- so a sequence can be described
         * before it has ever been written, let alone run.
         */
        SequenceDescription get_sequence_description(int ss = 0) const
        {
            pulseg_sequence_description desc;
            memset(&desc, 0, sizeof(desc));
            check(pulseg_get_sequence_description(&desc, coll_, ss));

            SequenceDescription out;
            out.subseq_idx = desc.subseq_idx;
            out.tr_duration_us = desc.tr_duration_us;
            if (desc.num_rows > 0 && desc.rows)
                out.rows.assign(desc.rows, desc.rows + desc.num_rows);

            pulseg_sequence_description_free(&desc);
            return out;
        }

        /**
         * One RF definition's decompressed shapes, keyed by the rf_def_id an
         * RF row carries in params[0].
         *
         * Units are the file's, not physics': the magnitude is normalised to
         * a peak of about 1 and scales by the row's amplitude, the phase is in
         * *turns* rather than radians (Pulseq's shape convention), and the
         * time points, when the definition has any, are in microseconds. The
         * getters hand back exactly what the shape library holds -- converting
         * here would put a second convention in play.
         */
        RfDefinitionShapes get_rf_definition(int subseq_idx, int rf_def_id) const
        {
            RfDefinitionShapes out;
            int channels = 0;
            int samples = 0;

            float** magnitude =
                pulseg_get_rf_def_magnitude(coll_, &channels, &samples, subseq_idx, rf_def_id);
            if (magnitude && channels > 0 && samples > 0)
            {
                out.num_channels = channels;
                out.magnitude.assign(magnitude[0], magnitude[0] + channels * samples);
            }

            channels = samples = 0;
            float** phase =
                pulseg_get_rf_def_phase(coll_, &channels, &samples, subseq_idx, rf_def_id);
            if (phase && channels > 0 && samples > 0)
                out.phase_turns.assign(phase[0], phase[0] + channels * samples);

            samples = 0;
            float* times = pulseg_get_rf_def_time(coll_, &samples, subseq_idx, rf_def_id);
            if (times)
            {
                if (samples > 0)
                    out.time_us.assign(times, times + samples);
                PULSEG_FREE(times);
            }
            return out;
        }

        /** Scan-global parameters aggregated over every loaded subsequence. */
        pulseg_sequence_parameters get_sequence_parameters() const
        {
            pulseg_sequence_parameters params;
            memset(&params, 0, sizeof(params));
            check(pulseg_get_sequence_parameters(&params, coll_));
            return params;
        }

        // ── Native-timing TR waveforms (for plotting) ────────────────

        TrWaveforms get_tr_waveforms(
            int ss = 0,
            int amplitude_mode = PULSEG_AMP_MAX_POS,
            int tr_index = 0,
            bool collapse_delays = false) const
        {
            pulseg_tr_waveforms cw;
            pulseg_diagnostic diag;
            memset(&cw, 0, sizeof(cw));
            pulseg_diagnostic_init(&diag);

            int code = pulseg_get_tr_waveforms(
                coll_,
                &cw,
                &diag,
                ss,
                amplitude_mode,
                tr_index,
                collapse_delays ? 1 : 0);
            check(code, diag);

            TrWaveforms w;
            auto copy_ch = [](ChannelWaveform& dst, const pulseg_channel_waveform& src)
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
                AdcEvent& a = w.adc_events[static_cast<size_t>(i)];
                a.onset_us = cw.adc_events[i].onset_us;
                a.duration_us = cw.adc_events[i].duration_us;
                a.num_samples = cw.adc_events[i].num_samples;
                a.freq_offset_hz = cw.adc_events[i].freq_offset_hz;
                a.phase_offset_rad = cw.adc_events[i].phase_offset_rad;
            }

            w.blocks.resize(static_cast<size_t>(cw.num_blocks));
            for (int i = 0; i < cw.num_blocks; ++i)
            {
                TrBlockDescriptor& b = w.blocks[static_cast<size_t>(i)];
                b.start_us = cw.blocks[i].start_us;
                b.duration_us = cw.blocks[i].duration_us;
                b.segment_idx = cw.blocks[i].segment_idx;
                b.rf_isocenter_us = cw.blocks[i].rf_isocenter_us;
            }

            w.total_duration_us = cw.total_duration_us;

            pulseg_tr_waveforms_free(&cw);
            return w;
        }

        // ── Mechanical resonances spectra ─────────────────────────────────────────

        MechResonancesSpectra calc_mech_resonances(
            int ss,
            int canonical_tr_idx,
            int amplitude_mode,
            float target_resolution_hz,
            float max_freq_hz,
            const std::vector<ForbiddenBand>& bands = {},
            float peak_log10_threshold = std::numeric_limits<float>::quiet_NaN(),
            float peak_norm_scale = std::numeric_limits<float>::quiet_NaN(),
            float peak_eps = std::numeric_limits<float>::quiet_NaN(),
            float peak_prominence = std::numeric_limits<float>::quiet_NaN(),
            bool compress_trains = true,
            CheckPlan* plan = nullptr) const
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
            pulseg_mech_resonances_request request = PULSEG_MECH_RESONANCES_REQUEST_INIT;
            request.subseq_idx = ss;
            request.canonical_tr_idx = canonical_tr_idx;
            request.amplitude_mode = amplitude_mode;
            request.target_resolution_hz = target_resolution_hz;
            request.max_freq_hz = max_freq_hz;
            request.bands.count = static_cast<int>(cbands.size());
            request.bands.bands = cbands.empty() ? nullptr : cbands.data();
            request.compress_trains = compress_trains ? 1 : 0;

            int code = pulseg_calc_mech_resonances(
                coll_,
                &cs,
                &diag,
                plan ? plan->handle() : nullptr,
                &run_opts,
                &request);
            check(code, diag);

            MechResonancesSpectra a;
            a.freq_min_hz = cs.freq_min_hz;
            a.freq_spacing_hz = cs.freq_spacing_hz;
            a.num_freq_bins = cs.num_freq_bins;

            auto assign_f = [](std::vector<float>& v, const float* p, int n)
            {
                if (p)
                    v.assign(p, p + n);
            };
            auto assign_i = [](std::vector<int>& v, const int* p, int n)
            {
                if (p)
                    v.assign(p, p + n);
            };

            assign_f(a.spectrum_full_gx, cs.spectrum_full_gx, cs.num_freq_bins);
            assign_f(a.spectrum_full_gy, cs.spectrum_full_gy, cs.num_freq_bins);
            assign_f(a.spectrum_full_gz, cs.spectrum_full_gz, cs.num_freq_bins);

            a.num_instances = cs.num_instances;

            a.num_analytical_peaks = cs.num_analytical_peaks;
            assign_f(a.analytical_peak_freqs, cs.analytical_peak_freqs, cs.num_analytical_peaks);
            assign_f(a.analytical_peak_amp_gx, cs.analytical_peak_amp_gx, cs.num_analytical_peaks);
            assign_f(a.analytical_peak_amp_gy, cs.analytical_peak_amp_gy, cs.num_analytical_peaks);
            assign_f(a.analytical_peak_amp_gz, cs.analytical_peak_amp_gz, cs.num_analytical_peaks);
            assign_f(
                a.analytical_peak_phase_gx,
                cs.analytical_peak_phase_gx,
                cs.num_analytical_peaks);
            assign_f(
                a.analytical_peak_phase_gy,
                cs.analytical_peak_phase_gy,
                cs.num_analytical_peaks);
            assign_f(
                a.analytical_peak_phase_gz,
                cs.analytical_peak_phase_gz,
                cs.num_analytical_peaks);
            assign_f(
                a.analytical_peak_widths_hz,
                cs.analytical_peak_widths_hz,
                cs.num_analytical_peaks);
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

            a.num_envelope_bins = cs.num_envelope_bins;
            assign_f(a.envelope_freqs_hz, cs.envelope_freqs_hz, cs.num_envelope_bins);
            assign_f(a.envelope_amp_gx, cs.envelope_amp_gx, cs.num_envelope_bins);
            assign_f(a.envelope_amp_gy, cs.envelope_amp_gy, cs.num_envelope_bins);
            assign_f(a.envelope_amp_gz, cs.envelope_amp_gz, cs.num_envelope_bins);

            pulseg_mech_resonances_spectra_free(&cs);
            return a;
        }

        // ── PNS computation ──────────────────────────────────────────

        /* Takes the C model rather than a model type: `pulseg_pns_model` is
         * the interface every model implements -- the published Irnich and
         * SAFE (pulseg_pns_models.h) and a vendor's own alike. */
        PnsResult calc_pns(
            int ss,
            int canonical_tr_idx,
            const pulseg_pns_model& model,
            CheckPlan* plan = nullptr) const
        {
            pulseg_pns_result cr = PULSEG_PNS_RESULT_INIT;
            pulseg_diagnostic diag;
            pulseg_diagnostic_init(&diag);
            int code = pulseg_calc_pns(
                coll_,
                &cr,
                &diag,
                plan ? plan->handle() : nullptr,
                ss,
                canonical_tr_idx,
                &opts_,
                &model);
            check(code, diag);

            PnsResult r;
            r.num_samples = cr.num_samples;
            r.worst_group = cr.worst_group;
            if (cr.slew_x_hz_per_m_per_s)
                r.slew_x.assign(
                    cr.slew_x_hz_per_m_per_s,
                    cr.slew_x_hz_per_m_per_s + cr.num_samples);
            if (cr.slew_y_hz_per_m_per_s)
                r.slew_y.assign(
                    cr.slew_y_hz_per_m_per_s,
                    cr.slew_y_hz_per_m_per_s + cr.num_samples);
            if (cr.slew_z_hz_per_m_per_s)
                r.slew_z.assign(
                    cr.slew_z_hz_per_m_per_s,
                    cr.slew_z_hz_per_m_per_s + cr.num_samples);

            pulseg_pns_result_free(&cr);
            return r;
        }

        // ── Safety check ─────────────────────────────────────────────

        void check_safety(
            const std::vector<ForbiddenBand>& bands = {},
            const pulseg_pns_model* pns_model = nullptr,
            float pns_threshold_percent = 100.0f,
            CheckPlan* plan = nullptr) const
        {
            std::vector<pulseg_forbidden_band> cbands(bands.size());
            for (size_t i = 0; i < bands.size(); ++i)
                cbands[i] = bands[i].to_c();

            pulseg_diagnostic diag;
            pulseg_diagnostic_init(&diag);
            // Note: check_safety takes non-const coll for cursor dry-run
            pulseg_forbidden_band_list band_list = PULSEG_FORBIDDEN_BAND_LIST_INIT;
            band_list.count = static_cast<int>(cbands.size());
            band_list.bands = cbands.empty() ? nullptr : cbands.data();

            int code = pulseg_check_safety(
                coll_,
                &diag,
                plan ? plan->handle() : nullptr,
                &opts_,
                &band_list,
                pns_model,
                pns_threshold_percent);
            check(code, diag);
        }

        /**
         * Gradient amplitude alone.
         *
         * Each check is also its own entry point, because a platform that
         * enforces some of them in hardware wants only the rest. None of them
         * assumes another has run.
         */
        void check_max_grad() const
        {
            pulseg_diagnostic diag;
            pulseg_diagnostic_init(&diag);
            check(pulseg_check_max_grad(coll_, &diag, &opts_), diag);
        }

        /** Gradient slew alone: what one event demands, not the step between two. */
        void check_max_slew() const
        {
            pulseg_diagnostic diag;
            pulseg_diagnostic_init(&diag);
            check(pulseg_check_max_slew(coll_, &diag, &opts_), diag);
        }

        /** Every event time on the raster it is played on. */
        void check_raster_alignment() const
        {
            pulseg_diagnostic diag;
            pulseg_diagnostic_init(&diag);
            check(pulseg_check_raster_alignment(coll_, &diag, &opts_), diag);
        }

        /** The PNS response against a threshold, using the injected model. */
        void check_pns(
            const pulseg_pns_model& model,
            float threshold_percent = 100.0f,
            CheckPlan* plan = nullptr) const
        {
            pulseg_diagnostic diag;
            pulseg_diagnostic_init(&diag);
            check(
                pulseg_check_pns(
                    coll_,
                    &diag,
                    plan ? plan->handle() : nullptr,
                    &opts_,
                    &model,
                    threshold_percent),
                diag);
        }

        /** No gradient harmonic inside a forbidden acoustic band. */
        void check_mech_resonances(
            const std::vector<ForbiddenBand>& bands,
            CheckPlan* plan = nullptr) const
        {
            std::vector<pulseg_forbidden_band> cbands(bands.size());
            for (size_t i = 0; i < bands.size(); ++i)
                cbands[i] = bands[i].to_c();

            pulseg_forbidden_band_list band_list = PULSEG_FORBIDDEN_BAND_LIST_INIT;
            band_list.count = static_cast<int>(cbands.size());
            band_list.bands = cbands.empty() ? nullptr : cbands.data();

            pulseg_diagnostic diag;
            pulseg_diagnostic_init(&diag);
            check(
                pulseg_check_mech_resonances(
                    coll_,
                    &diag,
                    plan ? plan->handle() : nullptr,
                    &opts_,
                    &band_list),
                diag);
        }

        /**
         * Gradient continuity alone, without a PNS model or a band table.
         *
         * Returns the diagnostic message rather than throwing on violation,
         * so a caller can report every question it asked instead of stopping
         * at the first. Empty means continuous.
         */
        std::string check_grad_continuity() const
        {
            pulseg_diagnostic diag;
            pulseg_diagnostic_init(&diag);
            const int code = pulseg_check_grad_continuity(coll_, &diag, &opts_);
            if (code == PULSEG_ERR_GRAD_DISCONTINUITY)
                return diag.message[0] ? diag.message : "gradient discontinuity";
            check(code, diag);
            return std::string();
        }

        // ── Block cursor ─────────────────────────────────────────────

        /** Advance one block. Returns PULSEG_CURSOR_BLOCK or PULSEG_CURSOR_DONE. */
        int cursor_next()
        {
            return pulseg_cursor_next(coll_);
        }

        /** Advance one block and report where it landed. */
        int cursor_advance(pulseg_cursor_info& info)
        {
            return pulseg_cursor_advance(coll_, &info);
        }

        /** Where the cursor rests now. */
        pulseg_cursor_info cursor_info() const
        {
            pulseg_cursor_info info = PULSEG_CURSOR_INFO_INIT;
            check(pulseg_cursor_get_info(coll_, &info));
            return info;
        }

        /** Back to before the first block. */
        void cursor_rewind()
        {
            pulseg_cursor_rewind(coll_);
        }

        /** Remember the current position, for a later cursor_reset(). */
        void cursor_mark()
        {
            pulseg_cursor_mark(coll_);
        }

        /** Return to the last cursor_mark(). */
        void cursor_reset()
        {
            pulseg_cursor_reset(coll_);
        }

        BlockInstance get_block_instance() const
        {
            pulseg_block_instance ci = PULSEG_BLOCK_INSTANCE_INIT;
            check(pulseg_get_block_instance(coll_, &ci));
            return BlockInstance::from_c(ci);
        }

        /** Resolved per-instance view at @p pos in subsequence @p ss, read
     *  independently of the cursor (PulSeg SegmentInstance, spec 3.3). */
        BlockInstance get_block_instance_at(int ss, int pos) const
        {
            pulseg_block_instance ci = PULSEG_BLOCK_INSTANCE_INIT;
            check(pulseg_get_block_instance_at(coll_, &ci, ss, pos));
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
                int rc = pulseg_get_segment_block_def_indices(coll_, ids.data(), seg_idx);
                if (rc < 0)
                    check(rc);
            }
            return ids;
        }

        // ── Segment layout ────────────────────────────────────────────

        /**
     * Per-segment layout of subsequence @p subseq_idx, resolved to the
     * *max-energy* instance of every (subsequence-local, pre-dedup)
     * unique segment.
     *
     * `block_indices` is a scan-table (exec stream) position translated
     * into the 0-based .seq block indices that instance occupies; falls
     * back to the segment definition's own `start_block` when the exec
     * stream is unavailable (e.g. a pulsegen cache load that never
     * materialised the scan table).
     */
        std::vector<SegmentLayout> segments(int subseq_idx = 0) const
        {
            // local_seg_idx below indexes the subsequence's *deduplicated*
            // segment definitions (see resolve_subseq_local_segment_global_id()
            // in pulseg_getters.c), not the pre-dedup prep+main+cooldown
            // position count -- a subsequence with repeated TR positions
            // (e.g. a partition loop) has strictly fewer unique segments than
            // positions.
            int num_local = pulseg_get_num_unique_segments(coll_, subseq_idx);
            if (num_local < 0)
                check(num_local);

            std::vector<SegmentLayout> out;
            out.reserve(static_cast<size_t>(num_local));
            for (int local = 0; local < num_local; ++local)
            {
                pulseg_segment_layout layout = PULSEG_SEGMENT_LAYOUT_INIT;
                check(pulseg_get_subseq_segment_layout(coll_, &layout, subseq_idx, local));

                SegmentLayout entry;
                entry.global_index = layout.global_index;
                entry.num_blocks = layout.num_blocks;
                entry.start_block = layout.start_block;
                entry.max_energy_start_block = layout.max_energy_start_block;
                entry.from_max_energy_instance = layout.from_max_energy_instance != 0;

                entry.block_indices.resize(static_cast<size_t>(layout.num_blocks));
                int written = pulseg_get_subseq_segment_block_indices(
                    coll_,
                    entry.block_indices.empty() ? nullptr : entry.block_indices.data(),
                    subseq_idx,
                    local);
                if (written != layout.num_blocks)
                    entry.block_indices.clear();

                pulseg_segment_info seg_info = segment_info(layout.global_index);
                entry.duration_us = seg_info.duration_us;
                entry.pure_delay = seg_info.pure_delay != 0;
                entry.is_nav = seg_info.is_nav != 0;
                entry.has_trigger = seg_info.has_trigger != 0;

                out.push_back(std::move(entry));
            }
            return out;
        }

    private:
        pulseg_collection* coll_ = nullptr;
        pulseg_opts opts_ = PULSEG_OPTS_INIT;
    };

} // namespace pulseg

#endif // PULSEG_COLLECTION_HPP
