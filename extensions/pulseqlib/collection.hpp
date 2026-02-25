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

    // ── Subsequence info ─────────────────────────────────────────

    int   num_subsequences()               const { return pulseqlib_get_num_subsequences(coll_); }
    float tr_duration_us(int ss = 0)       const { return pulseqlib_get_tr_duration_us(coll_, ss); }
    int   num_trs(int ss = 0)              const { return pulseqlib_get_num_trs(coll_, ss); }
    int   tr_size(int ss = 0)              const { return pulseqlib_get_tr_size(coll_, ss); }
    int   num_unique_adcs(int ss = 0)      const { return pulseqlib_get_num_unique_adcs(coll_, ss); }
    bool  is_pmc_enabled(int ss = 0)        const { return pulseqlib_is_pmc_enabled(coll_, ss) != 0; }
    int   subseq_segment_offset(int ss = 0) const { return pulseqlib_get_subseq_segment_offset(coll_, ss); }
    float total_duration_us()              const { return pulseqlib_get_total_duration_us(coll_); }

    // ── TR structure ─────────────────────────────────────────────

    int  num_prep_blocks(int ss = 0)       const { return pulseqlib_get_num_prep_blocks(coll_, ss); }
    int  num_cooldown_blocks(int ss = 0)   const { return pulseqlib_get_num_cooldown_blocks(coll_, ss); }
    bool degenerate_prep(int ss = 0)       const { return pulseqlib_get_degenerate_prep(coll_, ss) != 0; }
    bool degenerate_cooldown(int ss = 0)   const { return pulseqlib_get_degenerate_cooldown(coll_, ss) != 0; }
    int  num_prep_trs(int ss = 0)          const { return pulseqlib_get_num_prep_trs(coll_, ss); }
    int  num_cooldown_trs(int ss = 0)      const { return pulseqlib_get_num_cooldown_trs(coll_, ss); }

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

    // ── Segment info ─────────────────────────────────────────────

    int  num_segments()                         const { return pulseqlib_get_num_segments(coll_); }
    int  segment_duration_us(int seg)           const { return pulseqlib_get_segment_duration_us(coll_, seg); }
    bool is_segment_pure_delay(int seg)         const { return pulseqlib_is_segment_pure_delay(coll_, seg) == 1; }
    int  segment_num_blocks(int seg)            const { return pulseqlib_get_segment_num_blocks(coll_, seg); }
    int  segment_start_block(int seg)           const { return pulseqlib_get_segment_start_block(coll_, seg); }
    int  segment_num_kzero_crossings(int seg)   const { return pulseqlib_get_segment_num_kzero_crossings(coll_, seg); }

    // ── Segment tables ───────────────────────────────────────────

    int num_prep_segments(int ss = 0) const {
        return pulseqlib_get_num_prep_segments(coll_, ss);
    }
    int num_main_segments(int ss = 0) const {
        return pulseqlib_get_num_main_segments(coll_, ss);
    }
    int num_cooldown_segments(int ss = 0) const {
        return pulseqlib_get_num_cooldown_segments(coll_, ss);
    }

    std::vector<int> prep_segment_table(int ss = 0) const {
        int n = num_prep_segments(ss);
        std::vector<int> ids(n);
        if (n > 0) pulseqlib_get_prep_segment_table(coll_, ss, ids.data());
        return ids;
    }
    std::vector<int> main_segment_table(int ss = 0) const {
        int n = num_main_segments(ss);
        std::vector<int> ids(n);
        if (n > 0) pulseqlib_get_main_segment_table(coll_, ss, ids.data());
        return ids;
    }
    std::vector<int> cooldown_segment_table(int ss = 0) const {
        int n = num_cooldown_segments(ss);
        std::vector<int> ids(n);
        if (n > 0) pulseqlib_get_cooldown_segment_table(coll_, ss, ids.data());
        return ids;
    }

    // ── Block-level queries ──────────────────────────────────────

    int block_start_time_us(int seg, int blk)    const { return pulseqlib_get_block_start_time_us(coll_, seg, blk); }
    int block_duration_us(int seg, int blk)      const { return pulseqlib_get_block_duration_us(coll_, seg, blk); }

    // ── RF queries ───────────────────────────────────────────────

    int  num_unique_rf(int ss = 0)               const { return pulseqlib_get_num_unique_rf(coll_, ss); }
    bool block_has_rf(int seg, int blk)          const { return pulseqlib_block_has_rf(coll_, seg, blk) == 1; }
    int  rf_num_samples(int seg, int blk)        const { return pulseqlib_get_rf_num_samples(coll_, seg, blk); }
    int  rf_num_channels(int seg, int blk)       const { return pulseqlib_get_rf_num_channels(coll_, seg, blk); }
    int  rf_delay_us(int seg, int blk)           const { return pulseqlib_get_rf_delay_us(coll_, seg, blk); }

    RfStats get_rf_stats(int ss, int rf_idx) const {
        pulseqlib_rf_stats cstats = PULSEQLIB_RF_STATS_INIT;
        check(pulseqlib_get_rf_stats(coll_, &cstats, ss, rf_idx));
        return RfStats::from_c(cstats);
    }

    float rf_base_amplitude_hz(int ss, int rf_idx) const {
        return pulseqlib_get_rf_base_amplitude_hz(coll_, ss, rf_idx);
    }

    std::vector<int> tr_rf_ids(int ss = 0) const {
        int sz = tr_size(ss);
        std::vector<int> ids(sz, -1);
        pulseqlib_get_tr_rf_ids(coll_, ids.data(), ss);
        return ids;
    }

    // ── Gradient queries ─────────────────────────────────────────

    bool block_has_grad(int seg, int blk, int axis)         const { return pulseqlib_block_has_grad(coll_, seg, blk, axis) == 1; }
    bool block_grad_is_trapezoid(int seg, int blk, int axis) const { return pulseqlib_block_grad_is_trapezoid(coll_, seg, blk, axis) == 1; }
    int  grad_num_samples(int seg, int blk, int axis)        const { return pulseqlib_get_grad_num_samples(coll_, seg, blk, axis); }
    int  grad_num_shots(int seg, int blk, int axis)          const { return pulseqlib_get_grad_num_shots(coll_, seg, blk, axis); }
    int  grad_delay_us(int seg, int blk, int axis)           const { return pulseqlib_get_grad_delay_us(coll_, seg, blk, axis); }

    float grad_initial_amplitude(int seg, int blk, int axis) const {
        return pulseqlib_get_grad_initial_amplitude_hz_per_m(coll_, seg, blk, axis);
    }
    int grad_initial_shot_id(int seg, int blk, int axis) const {
        return pulseqlib_get_grad_initial_shot_id(coll_, seg, blk, axis);
    }

    // ── ADC queries ──────────────────────────────────────────────

    int  max_adc_samples()                      const { return pulseqlib_get_max_adc_samples(coll_); }
    int  adc_dwell_us(int adc_idx)              const { return pulseqlib_get_adc_dwell_us(coll_, adc_idx); }
    int  adc_num_samples(int adc_idx)           const { return pulseqlib_get_adc_num_samples(coll_, adc_idx); }
    bool block_has_adc(int seg, int blk)        const { return pulseqlib_block_has_adc(coll_, seg, blk) == 1; }
    int  adc_delay_us(int seg, int blk)         const { return pulseqlib_get_adc_delay_us(coll_, seg, blk); }
    int  adc_library_index(int seg, int blk)    const { return pulseqlib_get_adc_library_index(coll_, seg, blk); }

    // ── Flow control queries ─────────────────────────────────────

    bool block_has_digitalout(int seg, int blk)  const { return pulseqlib_block_has_digitalout(coll_, seg, blk) == 1; }
    int  digitalout_delay_us(int seg, int blk)   const { return pulseqlib_get_digitalout_delay_us(coll_, seg, blk); }
    int  digitalout_duration_us(int seg, int blk)const { return pulseqlib_get_digitalout_duration_us(coll_, seg, blk); }
    bool segment_has_trigger(int seg)            const { return pulseqlib_segment_has_trigger(coll_, seg) == 1; }
    int  segment_trigger_delay_us(int seg)       const { return pulseqlib_get_segment_trigger_delay_us(coll_, seg); }
    int  segment_trigger_duration_us(int seg)    const { return pulseqlib_get_segment_trigger_duration_us(coll_, seg); }
    bool segment_is_nav(int seg)                 const { return pulseqlib_segment_is_nav(coll_, seg) == 1; }
    bool block_has_rotation(int seg, int blk)    const { return pulseqlib_block_has_rotation(coll_, seg, blk) == 1; }
    bool block_has_norot(int seg, int blk)      const { return pulseqlib_block_has_norot(coll_, seg, blk) == 1; }
    bool block_has_nopos(int seg, int blk)      const { return pulseqlib_block_has_nopos(coll_, seg, blk) == 1; }

    // ── Label queries ────────────────────────────────────────────

    LabelLimits get_label_limits(int ss = 0) const {
        pulseqlib_label_limits cl;
        check(pulseqlib_get_label_limits(coll_, ss, &cl));
        return LabelLimits::from_c(cl);
    }

    int num_adc_occurrences(int ss = 0) const {
        return pulseqlib_get_num_adc_occurrences(coll_, ss);
    }
    int num_label_columns(int ss = 0) const {
        return pulseqlib_get_num_label_columns(coll_, ss);
    }

    std::vector<int> get_adc_label(int ss, int occurrence) const {
        int ncols = num_label_columns(ss);
        std::vector<int> vals(ncols);
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
