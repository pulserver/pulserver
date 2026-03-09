/**
 * @file collection.hpp
 * @brief RAII C++11 wrapper around pulseqlib_collection.
 */

#ifndef PULSEQLIB_COLLECTION_HPP
#define PULSEQLIB_COLLECTION_HPP

#include <cstring>
#include <string>
#include <vector>

#include "pulseqlib_methods.h"
#include "pulseqlib_types.h"

#include "error.hpp"
#include "types.hpp"

namespace pulseqlib {

/**
 * Owning wrapper around a pulseqlib_collection* with RAII lifetime.
 *
 * Movable, not copyable.
 */
class Collection {
public:
    // ── Construction / lifetime ──────────────────────────────────

    /** Load from one or more in-memory .seq buffers. */
    Collection(const char* const* buffers,
               const int*         sizes,
               int                num_buffers,
               const Opts&        opts,
               bool               parse_labels = true,
               int                num_averages = 1)
    {
        pulseqlib_opts copts = opts.to_c();
        pulseqlib_diagnostic diag;
        pulseqlib_diagnostic_init(&diag);
        int code = pulseqlib_read_from_buffers(
            &coll_, &diag, buffers, sizes, num_buffers,
            &copts, parse_labels ? 1 : 0, num_averages);
        check(code, diag);
        opts_ = copts;
    }

    /** Load from a .seq file on disk. */
    Collection(const char*  file_path,
               const Opts&  opts,
               bool         cache_binary     = false,
               bool         verify_signature = false,
               bool         parse_labels     = true,
               int          num_averages     = 1)
    {
        pulseqlib_opts copts = opts.to_c();
        pulseqlib_diagnostic diag;
        pulseqlib_diagnostic_init(&diag);
        int code = pulseqlib_read(
            &coll_, &diag, file_path, &copts,
            cache_binary ? 1 : 0,
            verify_signature ? 1 : 0,
            parse_labels ? 1 : 0,
            num_averages);
        check(code, diag);
        opts_ = copts;
    }

    ~Collection() {
        if (coll_) {
            pulseqlib_collection_free(coll_);
            coll_ = nullptr;
        }
    }

    // Move-only
    Collection(Collection&& o) noexcept : coll_(o.coll_), opts_(o.opts_) {
        o.coll_ = nullptr;
    }
    Collection& operator=(Collection&& o) noexcept {
        if (this != &o) {
            if (coll_) pulseqlib_collection_free(coll_);
            coll_ = o.coll_;
            opts_ = o.opts_;
            o.coll_ = nullptr;
        }
        return *this;
    }
    Collection(const Collection&) = delete;
    Collection& operator=(const Collection&) = delete;

    // ── Raw handle (for advanced use) ────────────────────────────

    pulseqlib_collection*       handle()       { return coll_; }
    const pulseqlib_collection* handle() const { return coll_; }
    const pulseqlib_opts&       opts()   const { return opts_;  }

    // ── Cache (serialization / deserialization) ──────────────────

    /** Save collection to a binary cache file. */
    void save_cache(const std::string& path, int source_size) const {
        check(pulseqlib_save_cache(coll_, path.c_str(), source_size));
    }

    /** Load collection from a binary cache file (mutates this). */
    void load_cache(const std::string& path, int source_size) {
        check(pulseqlib_load_cache(coll_, path.c_str(), source_size));
    }

    // ── Batch info queries ──────────────────────────────────────

    pulseqlib_collection_info collection_info() const {
        pulseqlib_collection_info info = PULSEQLIB_COLLECTION_INFO_INIT;
        check(pulseqlib_get_collection_info(coll_, &info));
        return info;
    }

    pulseqlib_subseq_info subseq_info(int ss = 0) const {
        pulseqlib_subseq_info info = PULSEQLIB_SUBSEQ_INFO_INIT;
        check(pulseqlib_get_subseq_info(coll_, ss, &info));
        return info;
    }

    pulseqlib_segment_info segment_info(int seg) const {
        pulseqlib_segment_info info = PULSEQLIB_SEGMENT_INFO_INIT;
        check(pulseqlib_get_segment_info(coll_, seg, &info));
        return info;
    }

    pulseqlib_block_info block_info(int seg, int blk) const {
        pulseqlib_block_info info = PULSEQLIB_BLOCK_INFO_INIT;
        check(pulseqlib_get_block_info(coll_, seg, blk, &info));
        return info;
    }

    pulseqlib_adc_def adc_def(int adc_idx) const {
        pulseqlib_adc_def def = PULSEQLIB_ADC_DEF_INIT;
        check(pulseqlib_get_adc_def(coll_, adc_idx, &def));
        return def;
    }

    // ── Scan time ────────────────────────────────────────────────

    ScanTimeInfo get_scan_time(int num_reps) const {
        pulseqlib_scan_time_info cinfo = PULSEQLIB_SCAN_TIME_INFO_INIT;
        check(pulseqlib_get_scan_time(coll_, num_reps, &cinfo));
        return ScanTimeInfo::from_c(cinfo);
    }

    // ── Consistency check ────────────────────────────────────────

    void check_consistency() const {
        pulseqlib_diagnostic diag;
        pulseqlib_diagnostic_init(&diag);
        check(pulseqlib_check_consistency(coll_, &diag), diag);
    }

    // ── Segment tables ───────────────────────────────────────────

    std::vector<int> prep_segment_table(int ss = 0) const {
        pulseqlib_subseq_info si = subseq_info(ss);
        std::vector<int> ids(si.num_prep_segments);
        if (si.num_prep_segments > 0)
            pulseqlib_get_prep_segment_table(coll_, ss, ids.data());
        return ids;
    }
    std::vector<int> main_segment_table(int ss = 0) const {
        pulseqlib_subseq_info si = subseq_info(ss);
        std::vector<int> ids(si.num_main_segments);
        if (si.num_main_segments > 0)
            pulseqlib_get_main_segment_table(coll_, ss, ids.data());
        return ids;
    }
    std::vector<int> cooldown_segment_table(int ss = 0) const {
        pulseqlib_subseq_info si = subseq_info(ss);
        std::vector<int> ids(si.num_cooldown_segments);
        if (si.num_cooldown_segments > 0)
            pulseqlib_get_cooldown_segment_table(coll_, ss, ids.data());
        return ids;
    }

    // ── RF queries (waveform access – still individual) ─────────

    RfStats get_rf_stats(int ss, int rf_idx) const {
        pulseqlib_rf_stats cstats = PULSEQLIB_RF_STATS_INIT;
        check(pulseqlib_get_rf_stats(coll_, &cstats, ss, rf_idx));
        return RfStats::from_c(cstats);
    }

    std::vector<int> tr_rf_ids(int ss = 0) const {
        pulseqlib_subseq_info si = subseq_info(ss);
        std::vector<int> ids(si.tr_size, -1);
        pulseqlib_get_tr_rf_ids(coll_, ids.data(), ss);
        return ids;
    }

    // ── Gradient waveform queries (still individual) ─────────────

    float grad_initial_amplitude(int seg, int blk, int axis) const {
        return pulseqlib_get_grad_initial_amplitude_hz_per_m(coll_, seg, blk, axis);
    }
    int grad_initial_shot_id(int seg, int blk, int axis) const {
        return pulseqlib_get_grad_initial_shot_id(coll_, seg, blk, axis);
    }

    // ── Label queries ────────────────────────────────────────────

    LabelLimits get_label_limits(int ss = 0) const {
        pulseqlib_label_limits cl;
        check(pulseqlib_get_label_limits(coll_, ss, &cl));
        return LabelLimits::from_c(cl);
    }

    std::vector<int> get_adc_label(int ss, int occurrence) const {
        pulseqlib_subseq_info si = subseq_info(ss);
        std::vector<int> vals(si.num_label_columns);
        check(pulseqlib_get_adc_label(coll_, ss, occurrence, vals.data()));
        return vals;
    }

    // ── TR gradient waveforms ────────────────────────────────────

    TrGradientWaveforms get_tr_gradient_waveforms(int ss = 0) const {
        pulseqlib_tr_gradient_waveforms cw = PULSEQLIB_TR_GRADIENT_WAVEFORMS_INIT;
        pulseqlib_diagnostic diag;
        pulseqlib_diagnostic_init(&diag);
        int code = pulseqlib_get_tr_gradient_waveforms(coll_, ss, &cw, &diag);
        check(code, diag);

        TrGradientWaveforms w;
        auto copy_axis = [](GradAxisWaveform& dst, const pulseqlib_grad_axis_waveform& src) {
            dst.time_us.assign(src.time_us, src.time_us + src.num_samples);
            dst.amplitude_hz_per_m.assign(src.amplitude_hz_per_m, src.amplitude_hz_per_m + src.num_samples);
            dst.seg_label.resize(src.num_samples);
            for (int i = 0; i < src.num_samples; ++i)
                dst.seg_label[i] = static_cast<float>(src.seg_label[i]);
        };
        copy_axis(w.gx, cw.gx);
        copy_axis(w.gy, cw.gy);
        copy_axis(w.gz, cw.gz);

        pulseqlib_tr_gradient_waveforms_free(&cw);
        return w;
    }

    // ── Native-timing TR waveforms (for plotting) ────────────────

    TrWaveforms get_tr_waveforms(
        int ss                = 0,
        int amplitude_mode    = PULSEQLIB_AMP_MAX_POS,
        int tr_index          = 0,
        bool include_prep     = false,
        bool include_cooldown = false,
        bool collapse_delays  = false) const
    {
        pulseqlib_tr_waveforms cw;
        pulseqlib_diagnostic diag;
        memset(&cw, 0, sizeof(cw));
        pulseqlib_diagnostic_init(&diag);

        int code = pulseqlib_get_tr_waveforms(
            coll_, ss, amplitude_mode, tr_index,
            include_prep ? 1 : 0, include_cooldown ? 1 : 0,
            collapse_delays ? 1 : 0,
            &cw, &diag);
        check(code, diag);

        TrWaveforms w;
        auto copy_ch = [](ChannelWaveform& dst, const pulseqlib_channel_waveform& src) {
            if (src.num_samples > 0 && src.time_us && src.amplitude) {
                dst.time_us.assign(src.time_us, src.time_us + src.num_samples);
                dst.amplitude.assign(src.amplitude, src.amplitude + src.num_samples);
            }
        };
        copy_ch(w.gx,       cw.gx);
        copy_ch(w.gy,       cw.gy);
        copy_ch(w.gz,       cw.gz);
        copy_ch(w.rf_mag,   cw.rf_mag);
        copy_ch(w.rf_phase, cw.rf_phase);

        w.adc_events.resize(static_cast<size_t>(cw.num_adc_events));
        for (int i = 0; i < cw.num_adc_events; ++i) {
            AdcEvent& a = w.adc_events[static_cast<size_t>(i)];
            a.onset_us         = cw.adc_events[i].onset_us;
            a.duration_us      = cw.adc_events[i].duration_us;
            a.num_samples      = cw.adc_events[i].num_samples;
            a.freq_offset_hz   = cw.adc_events[i].freq_offset_hz;
            a.phase_offset_rad = cw.adc_events[i].phase_offset_rad;
        }

        w.blocks.resize(static_cast<size_t>(cw.num_blocks));
        for (int i = 0; i < cw.num_blocks; ++i) {
            TrBlockDescriptor& b = w.blocks[static_cast<size_t>(i)];
            b.start_us    = cw.blocks[i].start_us;
            b.duration_us = cw.blocks[i].duration_us;
            b.segment_idx = cw.blocks[i].segment_idx;
        }

        w.total_duration_us = cw.total_duration_us;

        pulseqlib_tr_waveforms_free(&cw);
        return w;
    }

    // ── Acoustic spectra ─────────────────────────────────────────

    AcousticSpectra calc_acoustic_spectra(
        int ss,
        int target_window_size,
        float target_resolution_hz,
        float max_freq_hz,
        const std::vector<ForbiddenBand>& bands = {}) const
    {
        std::vector<pulseqlib_forbidden_band> cbands(bands.size());
        for (size_t i = 0; i < bands.size(); ++i)
            cbands[i] = bands[i].to_c();

        pulseqlib_acoustic_spectra cs = PULSEQLIB_ACOUSTIC_SPECTRA_INIT;
        pulseqlib_diagnostic diag;
        pulseqlib_diagnostic_init(&diag);
        int code = pulseqlib_calc_acoustic_spectra(
            &cs, &diag, coll_, ss, &opts_,
            target_window_size, target_resolution_hz, max_freq_hz,
            static_cast<int>(cbands.size()),
            cbands.empty() ? nullptr : cbands.data());
        check(code, diag);

        AcousticSpectra a;
        a.freq_min_hz     = cs.freq_min_hz;
        a.freq_spacing_hz = cs.freq_spacing_hz;
        a.num_freq_bins   = cs.num_freq_bins;
        a.num_windows     = cs.num_windows;

        int total = cs.num_windows * cs.num_freq_bins;
        auto assign_f = [](std::vector<float>& v, const float* p, int n) { if (p) v.assign(p, p + n); };
        auto assign_i = [](std::vector<int>&   v, const int*   p, int n) { if (p) v.assign(p, p + n); };

        assign_f(a.spectrogram_gx, cs.spectrogram_gx, total);
        assign_f(a.spectrogram_gy, cs.spectrogram_gy, total);
        assign_f(a.spectrogram_gz, cs.spectrogram_gz, total);
        assign_i(a.peaks_gx, cs.peaks_gx, total);
        assign_i(a.peaks_gy, cs.peaks_gy, total);
        assign_i(a.peaks_gz, cs.peaks_gz, total);

        assign_f(a.spectrum_full_gx, cs.spectrum_full_gx, cs.num_freq_bins);
        assign_f(a.spectrum_full_gy, cs.spectrum_full_gy, cs.num_freq_bins);
        assign_f(a.spectrum_full_gz, cs.spectrum_full_gz, cs.num_freq_bins);
        assign_i(a.peaks_full_gx,    cs.peaks_full_gx,    cs.num_freq_bins);
        assign_i(a.peaks_full_gy,    cs.peaks_full_gy,    cs.num_freq_bins);
        assign_i(a.peaks_full_gz,    cs.peaks_full_gz,    cs.num_freq_bins);

        a.freq_spacing_seq_hz = cs.freq_spacing_seq_hz;
        a.num_freq_bins_seq   = cs.num_freq_bins_seq;
        if (cs.num_freq_bins_seq > 0) {
            assign_f(a.spectrum_seq_gx, cs.spectrum_seq_gx, cs.num_freq_bins_seq);
            assign_f(a.spectrum_seq_gy, cs.spectrum_seq_gy, cs.num_freq_bins_seq);
            assign_f(a.spectrum_seq_gz, cs.spectrum_seq_gz, cs.num_freq_bins_seq);
            assign_i(a.peaks_seq_gx,    cs.peaks_seq_gx,    cs.num_freq_bins_seq);
            assign_i(a.peaks_seq_gy,    cs.peaks_seq_gy,    cs.num_freq_bins_seq);
            assign_i(a.peaks_seq_gz,    cs.peaks_seq_gz,    cs.num_freq_bins_seq);
        }

        pulseqlib_acoustic_spectra_free(&cs);
        return a;
    }

    // ── PNS computation ──────────────────────────────────────────

    PnsResult calc_pns(int ss, const PnsParams& params) const {
        pulseqlib_pns_params cp = params.to_c();
        pulseqlib_pns_result cr = PULSEQLIB_PNS_RESULT_INIT;
        pulseqlib_diagnostic diag;
        pulseqlib_diagnostic_init(&diag);
        int code = pulseqlib_calc_pns(&cr, &diag, coll_, ss, &opts_, &cp);
        check(code, diag);

        PnsResult r;
        r.num_samples = cr.num_samples;
        if (cr.slew_x_hz_per_m_per_s)
            r.slew_x.assign(cr.slew_x_hz_per_m_per_s, cr.slew_x_hz_per_m_per_s + cr.num_samples);
        if (cr.slew_y_hz_per_m_per_s)
            r.slew_y.assign(cr.slew_y_hz_per_m_per_s, cr.slew_y_hz_per_m_per_s + cr.num_samples);
        if (cr.slew_z_hz_per_m_per_s)
            r.slew_z.assign(cr.slew_z_hz_per_m_per_s, cr.slew_z_hz_per_m_per_s + cr.num_samples);

        pulseqlib_pns_result_free(&cr);
        return r;
    }

    // ── Safety check ─────────────────────────────────────────────

    void check_safety(
        const std::vector<ForbiddenBand>& bands = {},
        const PnsParams* pns_params = nullptr,
        float pns_threshold_percent = 100.0f) const
    {
        std::vector<pulseqlib_forbidden_band> cbands(bands.size());
        for (size_t i = 0; i < bands.size(); ++i)
            cbands[i] = bands[i].to_c();

        pulseqlib_pns_params cp;
        const pulseqlib_pns_params* cpp = nullptr;
        if (pns_params) {
            cp  = pns_params->to_c();
            cpp = &cp;
        }

        pulseqlib_diagnostic diag;
        pulseqlib_diagnostic_init(&diag);
        // Note: check_safety takes non-const coll for cursor dry-run
        int code = pulseqlib_check_safety(
            coll_, &diag, &opts_,
            static_cast<int>(cbands.size()),
            cbands.empty() ? nullptr : cbands.data(),
            cpp, pns_threshold_percent);
        check(code, diag);
    }

    // ── Block cursor ─────────────────────────────────────────────

    int  cursor_next()  { return pulseqlib_cursor_next(coll_); }
    void cursor_reset() { pulseqlib_cursor_reset(coll_); }

    BlockInstance get_block_instance() const {
        pulseqlib_block_instance ci = PULSEQLIB_BLOCK_INSTANCE_INIT;
        check(pulseqlib_get_block_instance(coll_, &ci));
        return BlockInstance::from_c(ci);
    }

    // ── Unique-block and segment-block queries ───────────────────

    /** Number of unique block definitions in subsequence @p ss. */
    int num_unique_blocks(int ss = 0) const {
        int n = pulseqlib_get_num_unique_blocks(coll_, ss);
        if (n < 0) check(n);
        return n;
    }

    /** 1-based .seq block ID for the @p blk_def_idx-th unique block. */
    int unique_block_id(int ss, int blk_def_idx) const {
        int id = pulseqlib_get_unique_block_id(coll_, ss, blk_def_idx);
        if (id < 0) check(id);
        return id;
    }

    /** Unique-block-definition indices for a global segment. */
    std::vector<int> segment_block_def_indices(int seg_idx) const {
        auto si = segment_info(seg_idx);
        std::vector<int> ids(si.num_blocks);
        if (si.num_blocks > 0) {
            int rc = pulseqlib_get_segment_block_def_indices(coll_, seg_idx, ids.data());
            if (rc < 0) check(rc);
        }
        return ids;
    }

    // ── Frequency modulation ─────────────────────────────────────

    int freq_mod_count() const {
        return pulseqlib_get_freq_mod_count(coll_);
    }
    int freq_mod_count_tr(int tr_type, int tr_index) const {
        return pulseqlib_get_freq_mod_count_tr(coll_, tr_type, tr_index);
    }

private:
    pulseqlib_collection* coll_ = nullptr;
    pulseqlib_opts        opts_ = PULSEQLIB_OPTS_INIT;
};

} // namespace pulseqlib

#endif // PULSEQLIB_COLLECTION_HPP
