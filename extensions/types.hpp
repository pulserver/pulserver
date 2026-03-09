/**
 * @file types.hpp
 * @brief C++11 value types wrapping pulseqlib C structs.
 */

#ifndef PULSEQLIB_TYPES_HPP
#define PULSEQLIB_TYPES_HPP

#include <vector>

#include "pulseqlib_types.h"

namespace pulseqlib {

// ── Scanner options ──────────────────────────────────────────────────

struct Opts {
    float gamma_hz_per_t         = 0.0f;
    float b0_t                   = 0.0f;
    float max_grad_hz_per_m      = 0.0f;
    float max_slew_hz_per_m_per_s = 0.0f;
    float rf_raster_us           = 0.0f;
    float grad_raster_us         = 0.0f;
    float adc_raster_us          = 0.0f;
    float block_raster_us        = 0.0f;

    /** Convert to the C struct, calling pulseqlib_opts_init. */
    pulseqlib_opts to_c() const {
        pulseqlib_opts o;
        pulseqlib_opts_init(&o,
            gamma_hz_per_t, b0_t,
            max_grad_hz_per_m, max_slew_hz_per_m_per_s,
            rf_raster_us, grad_raster_us, adc_raster_us, block_raster_us);
        return o;
    }
};

// ── Forbidden acoustic band ─────────────────────────────────────────

struct ForbiddenBand {
    float freq_min_hz            = 0.0f;
    float freq_max_hz            = 0.0f;
    float max_amplitude_hz_per_m = 0.0f;

    pulseqlib_forbidden_band to_c() const {
        pulseqlib_forbidden_band b;
        b.freq_min_hz            = freq_min_hz;
        b.freq_max_hz            = freq_max_hz;
        b.max_amplitude_hz_per_m = max_amplitude_hz_per_m;
        return b;
    }
};

// ── PNS parameters ──────────────────────────────────────────────────

struct PnsParams {
    int   vendor                   = 0;
    float chronaxie_us             = 0.0f;
    float rheobase_hz_per_m_per_s  = 0.0f;
    float alpha                    = 1.0f;

    pulseqlib_pns_params to_c() const {
        pulseqlib_pns_params p;
        p.vendor                  = vendor;
        p.chronaxie_us            = chronaxie_us;
        p.rheobase_hz_per_m_per_s = rheobase_hz_per_m_per_s;
        p.alpha                   = alpha;
        return p;
    }
};

// ── Scan-time info ──────────────────────────────────────────────────

struct ScanTimeInfo {
    float total_duration_us        = 0.0f;
    int   total_segment_boundaries = 0;

    static ScanTimeInfo from_c(const pulseqlib_scan_time_info& c) {
        ScanTimeInfo s;
        s.total_duration_us        = c.total_duration_us;
        s.total_segment_boundaries = c.total_segment_boundaries;
        return s;
    }
};

// ── RF statistics ───────────────────────────────────────────────────

struct RfStats {
    float flip_angle_deg   = 0.0f;
    float area             = 0.0f;
    float abs_width        = 0.0f;
    float eff_width        = 0.0f;
    float duty_cycle       = 0.0f;
    float max_pulse_width  = 0.0f;
    float duration_us      = 0.0f;
    int   isodelay_us      = 0;
    float bandwidth_hz     = 0.0f;
    float base_amplitude_hz = 0.0f;
    int   num_samples      = 0;

    static RfStats from_c(const pulseqlib_rf_stats& c) {
        RfStats s;
        s.flip_angle_deg   = c.flip_angle_deg;
        s.area             = c.area;
        s.abs_width        = c.abs_width;
        s.eff_width        = c.eff_width;
        s.duty_cycle       = c.duty_cycle;
        s.max_pulse_width  = c.max_pulse_width;
        s.duration_us      = c.duration_us;
        s.isodelay_us      = c.isodelay_us;
        s.bandwidth_hz     = c.bandwidth_hz;
        s.base_amplitude_hz = c.base_amplitude_hz;
        s.num_samples      = c.num_samples;
        return s;
    }
};

// ── Label limits ────────────────────────────────────────────────────

struct LabelLimit {
    int min;
    int max;
    LabelLimit() : min(0), max(0) {}
    LabelLimit(int mn, int mx) : min(mn), max(mx) {}
};

struct LabelLimits {
    LabelLimit slc, phs, rep, avg, seg, set, eco, par, lin, acq;

    static LabelLimits from_c(const pulseqlib_label_limits& c) {
        LabelLimits l;
        l.slc = LabelLimit(c.slc.min, c.slc.max);
        l.phs = LabelLimit(c.phs.min, c.phs.max);
        l.rep = LabelLimit(c.rep.min, c.rep.max);
        l.avg = LabelLimit(c.avg.min, c.avg.max);
        l.seg = LabelLimit(c.seg.min, c.seg.max);
        l.set = LabelLimit(c.set.min, c.set.max);
        l.eco = LabelLimit(c.eco.min, c.eco.max);
        l.par = LabelLimit(c.par.min, c.par.max);
        l.lin = LabelLimit(c.lin.min, c.lin.max);
        l.acq = LabelLimit(c.acq.min, c.acq.max);
        return l;
    }
};

// ── Gradient axis waveform ──────────────────────────────────────────

struct GradAxisWaveform {
    std::vector<float> time_us;
    std::vector<float> amplitude_hz_per_m;
    std::vector<float> seg_label;  // stored as float for uniformity
};

struct TrGradientWaveforms {
    GradAxisWaveform gx;
    GradAxisWaveform gy;
    GradAxisWaveform gz;
};

// ── Native-timing TR waveforms (for plotting) ──────────────────────

struct ChannelWaveform {
    std::vector<float> time_us;
    std::vector<float> amplitude;
};

struct AdcEvent {
    float onset_us          = 0.0f;
    float duration_us       = 0.0f;
    int   num_samples       = 0;
    float freq_offset_hz    = 0.0f;
    float phase_offset_rad  = 0.0f;
};

struct TrBlockDescriptor {
    float start_us      = 0.0f;
    float duration_us   = 0.0f;
    int   segment_idx   = -1;
};

struct TrWaveforms {
    ChannelWaveform gx;         // Hz/m
    ChannelWaveform gy;         // Hz/m
    ChannelWaveform gz;         // Hz/m
    ChannelWaveform rf_mag;     // Hz
    ChannelWaveform rf_phase;   // rad
    std::vector<AdcEvent>        adc_events;
    std::vector<TrBlockDescriptor> blocks;
    float total_duration_us = 0.0f;
};

// ── Acoustic spectra ────────────────────────────────────────────────

struct AcousticSpectra {
    float freq_min_hz       = 0.0f;
    float freq_spacing_hz   = 0.0f;
    int   num_freq_bins     = 0;
    int   num_windows       = 0;

    std::vector<float> spectrogram_gx;
    std::vector<float> spectrogram_gy;
    std::vector<float> spectrogram_gz;
    std::vector<int>   peaks_gx;
    std::vector<int>   peaks_gy;
    std::vector<int>   peaks_gz;

    std::vector<float> spectrum_full_gx;
    std::vector<float> spectrum_full_gy;
    std::vector<float> spectrum_full_gz;
    std::vector<int>   peaks_full_gx;
    std::vector<int>   peaks_full_gy;
    std::vector<int>   peaks_full_gz;

    float              freq_spacing_seq_hz = 0.0f;
    int                num_freq_bins_seq   = 0;
    std::vector<float> spectrum_seq_gx;
    std::vector<float> spectrum_seq_gy;
    std::vector<float> spectrum_seq_gz;
    std::vector<int>   peaks_seq_gx;
    std::vector<int>   peaks_seq_gy;
    std::vector<int>   peaks_seq_gz;
};

// ── PNS result ──────────────────────────────────────────────────────

struct PnsResult {
    int num_samples = 0;
    std::vector<float> slew_x;
    std::vector<float> slew_y;
    std::vector<float> slew_z;
};

// ── Block instance (cursor output) ──────────────────────────────────

struct BlockInstance {
    int   duration_us    = 0;
    float rf_amp_hz      = 0.0f;
    float rf_freq_hz     = 0.0f;
    float rf_phase_rad   = 0.0f;
    int   rf_shim_id     = -1;
    float gx_amp         = 0.0f;
    float gy_amp         = 0.0f;
    float gz_amp         = 0.0f;
    int   gx_shot_idx    = 0;
    int   gy_shot_idx    = 0;
    int   gz_shot_idx    = 0;
    float rotmat[9]      = {1,0,0, 0,1,0, 0,0,1};
    int   norot_flag     = 0;
    int   nopos_flag     = 0;
    int   digitalout_flag = 0;
    int   adc_flag       = 0;
    float adc_freq_hz    = 0.0f;
    float adc_phase_rad  = 0.0f;

    static BlockInstance from_c(const pulseqlib_block_instance& c) {
        BlockInstance b;
        b.duration_us  = c.duration_us;
        b.rf_amp_hz    = c.rf_amp_hz;
        b.rf_freq_hz   = c.rf_freq_hz;
        b.rf_phase_rad = c.rf_phase_rad;
        b.rf_shim_id   = c.rf_shim_id;
        b.gx_amp       = c.gx_amp_hz_per_m;
        b.gy_amp       = c.gy_amp_hz_per_m;
        b.gz_amp       = c.gz_amp_hz_per_m;
        b.gx_shot_idx  = c.gx_shot_idx;
        b.gy_shot_idx  = c.gy_shot_idx;
        b.gz_shot_idx  = c.gz_shot_idx;
        for (int i = 0; i < 9; ++i) b.rotmat[i] = c.rotmat[i];
        b.norot_flag   = c.norot_flag;
        b.nopos_flag   = c.nopos_flag;
        b.digitalout_flag = c.digitalout_flag;
        b.adc_flag     = c.adc_flag;
        b.adc_freq_hz  = c.adc_freq_hz;
        b.adc_phase_rad= c.adc_phase_rad;
        return b;
    }
};

} // namespace pulseqlib

#endif // PULSEQLIB_TYPES_HPP
