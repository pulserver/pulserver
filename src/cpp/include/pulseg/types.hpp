/**
 * @file types.hpp
 * @brief C++11 value types wrapping pulseg C structs.
 */

#ifndef PULSEG_TYPES_HPP
#define PULSEG_TYPES_HPP

#include <cmath>
#include <vector>

#include "pulseg_types.h"

namespace pulseg
{

    // ── Scanner options ──────────────────────────────────────────────────

    struct Opts
    {
        float gamma_hz_per_t = 0.0f;
        float b0_t = 0.0f;
        float max_grad_hz_per_m = 0.0f;
        float max_slew_hz_per_m_per_s = 0.0f;
        float rf_raster_us = 0.0f;
        float grad_raster_us = 0.0f;
        float adc_raster_us = 0.0f;
        float block_raster_us = 0.0f;
        float peak_log10_threshold = PULSEG_PEAK_LOG10_THRESHOLD_DEFAULT;
        float peak_norm_scale = PULSEG_PEAK_NORM_SCALE_DEFAULT;
        float peak_eps = PULSEG_PEAK_EPS_DEFAULT;

        /** Which label-state indices fill the 3 ADC label-table columns
         *  (0=SLC 1=PHS 2=REP 3=AVG 4=SEG 5=SET 6=ECO 7=PAR 8=LIN 9=ACQ).
         *  The GE console injects {8, 0, 6} = [LIN, SLC, ECO]. */
        int label_column_map[3] = {0, 1, 2};

        /** Convert to the C struct, calling pulseg_opts_init. */
        pulseg_opts to_c() const
        {
            pulseg_opts o;
            pulseg_opts_init_full(
                &o,
                gamma_hz_per_t,
                b0_t,
                max_grad_hz_per_m,
                max_slew_hz_per_m_per_s,
                rf_raster_us,
                grad_raster_us,
                adc_raster_us,
                block_raster_us,
                peak_log10_threshold,
                peak_norm_scale,
                peak_eps);
            o.label_column_map[0] = label_column_map[0];
            o.label_column_map[1] = label_column_map[1];
            o.label_column_map[2] = label_column_map[2];
            return o;
        }
    };

    // ── Forbidden mechanical resonance band ─────────────────────────────────────────

    struct ForbiddenBand
    {
        float freq_min_hz = 0.0f;
        float freq_max_hz = 0.0f;
        float max_amplitude_hz_per_m = 0.0f;

        pulseg_forbidden_band to_c() const
        {
            pulseg_forbidden_band b;
            b.freq_min_hz = freq_min_hz;
            b.freq_max_hz = freq_max_hz;
            b.max_amplitude_hz_per_m = max_amplitude_hz_per_m;
            return b;
        }
    };

    // ── PNS parameters (generic exponential model) ──────────────────────
    //
    // pulseg no longer bundles a vendor PNS model (see pulseg_pns_model in
    // pulseg_types.h): the functional form differs per vendor (GE's
    // exponential kernel vs. e.g. Siemens SAFE's nonlinear multi-stage
    // filter) and only an evaluator interface, not a sampled kernel, can
    // represent both. PnsParams below is a generic, vendor-neutral
    // convenience implementation of the chronaxie/rheobase/alpha exponential
    // form -- compiled into this header, not into libpulseg -- so wrapper
    // users can still plot/check PNS with explicit parameters of their own
    // choosing. It is NOT the GE-calibrated model used on-scanner (that lives
    // in the private pulserver-interpreter's src_gelib/).
    struct PnsParams
    {
        float chronaxie_us = 0.0f;
        float rheobase_hz_per_m_per_s = 0.0f;
        float alpha = 1.0f;

        /** Build a pulseg_pns_model view onto this instance (borrows *this; keep alive
     *  for the duration of the calc_pns/check_safety call). Only reads chronaxie/
     *  rheobase/alpha, so a const instance may safely be adapted. */
        pulseg_pns_model to_c() const
        {
            pulseg_pns_model m;
            m.ctx = const_cast<PnsParams*>(this);
            m.required_padding = &PnsParams::required_padding_cb;
            m.evaluate = &PnsParams::evaluate_cb;
            /* Must be set: pulseg_pns_model is a plain C struct, so `m` is
             * uninitialised here, and calc_pns_from_uniform branches on
             * model->kernel -- leaving it would read (and potentially call)
             * whatever was on the stack. This model publishes no kernel, so
             * the core takes its exact full-waveform path. */
            m.kernel = NULL;
            return m;
        }

    private:
        static constexpr float kKernelDurationFactor = 20.0f;

        /* Exponential kernel k[i] = (dt/s_min) * c / (c+tau)^2, i=0..n-1. */
        std::vector<float> build_kernel(float dt_us) const
        {
            float c_s = chronaxie_us * 1e-6f;
            float dt_s = dt_us * 1e-6f;
            float s_min = rheobase_hz_per_m_per_s / alpha;
            int n = static_cast<int>(kKernelDurationFactor * c_s / dt_s) + 1;
            std::vector<float> k(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i)
            {
                float tau = static_cast<float>(i) * dt_s;
                float denom = (c_s + tau) * (c_s + tau);
                k[static_cast<size_t>(i)] = (dt_s / s_min) * (c_s / denom);
            }
            return k;
        }

        /* Causal FIR application: out[i] = sum_{k=0}^{min(i,klen-1)} kernel[k]*sig[i-k]. */
        static void convolve_causal(
            float* out,
            const float* sig,
            int n,
            const std::vector<float>& kernel)
        {
            int klen = static_cast<int>(kernel.size());
            for (int i = 0; i < n; ++i)
            {
                float acc = 0.0f;
                int kmax = (i < klen - 1) ? i : (klen - 1);
                for (int k = 0; k <= kmax; ++k)
                    acc += kernel[static_cast<size_t>(k)] * sig[i - k];
                out[i] = acc * 100.0f;
            }
        }

        static int required_padding_cb(void* ctx, float dt_us)
        {
            auto* self = static_cast<PnsParams*>(ctx);
            return static_cast<int>(self->build_kernel(dt_us).size());
        }

        static int evaluate_cb(
            void* ctx,
            const float* dgdt_x,
            const float* dgdt_y,
            const float* dgdt_z,
            int n,
            float dt_us,
            float* out_x,
            float* out_y,
            float* out_z)
        {
            auto* self = static_cast<PnsParams*>(ctx);
            std::vector<float> kernel = self->build_kernel(dt_us);
            convolve_causal(out_x, dgdt_x, n, kernel);
            convolve_causal(out_y, dgdt_y, n, kernel);
            convolve_causal(out_z, dgdt_z, n, kernel);
            return 0; /* PULSEG_SUCCESS */
        }
    };

    // ── SAFE PNS parameters (Siemens' three-branch model) ───────────────
    //
    // The second vendor-neutral convenience model beside PnsParams above, and
    // header-only for the same reason. SAFE is Hebrank & Gebhardt's published
    // abstract (ISMRM 2000, No. 2007), coded by Witzel and Szczepankiewicz and
    // shipped in PyPulseq as safe_pns_prediction.py; only the per-scanner
    // coefficients below are proprietary, and none are distributed here.
    //
    // Deliberately leaves pulseg_pns_model::kernel NULL. Publishing a kernel
    // asserts the model is exactly a convolution, and this one is not: the
    // first and third branches rectify *after* their lowpass and the second
    // rectifies *before* it, so pulseg_pns_memo.c's per-shape assembly would
    // be wrong rather than merely slow. The core then always takes its exact
    // full-waveform path, which costs roughly what PnsParams costs before
    // memoization -- an order of magnitude more than the memoized Irnich
    // model on a long canonical TR.
    struct SafeParams
    {
        // Per-axis coefficients, in the units the .asc tables state them:
        // time constants in ms, stim_limit in T/m/s, a1..a3 and g_scale
        // dimensionless.
        struct Axis
        {
            float a1 = 0.0f;
            float a2 = 0.0f;
            float a3 = 0.0f;
            float tau1_ms = 0.0f;
            float tau2_ms = 0.0f;
            float tau3_ms = 0.0f;
            float stim_limit = 0.0f;
            float g_scale = 0.0f;
        };

        Axis x;
        Axis y;
        Axis z;

        /** Build a pulseg_pns_model view onto this instance (borrows *this; keep alive
     *  for the duration of the calc_pns/check_safety call). */
        pulseg_pns_model to_c() const
        {
            pulseg_pns_model m;
            m.ctx = const_cast<SafeParams*>(this);
            m.required_padding = &SafeParams::required_padding_cb;
            m.evaluate = &SafeParams::evaluate_cb;
            m.kernel = NULL; /* not LTI -- see the comment above */
            return m;
        }

    private:
        /* safe_gwf_to_pns pads by 4x the longest time constant before the
         * waveform and 4x after it; the core only appends, and only needs
         * enough history for the slowest branch to have settled. */
        static constexpr float kPaddingTimeConstants = 4.0f;

        float longest_tau_ms() const
        {
            float longest = x.tau1_ms;
            const float taus[9] = {
                x.tau1_ms,
                x.tau2_ms,
                x.tau3_ms,
                y.tau1_ms,
                y.tau2_ms,
                y.tau3_ms,
                z.tau1_ms,
                z.tau2_ms,
                z.tau3_ms};
            for (int i = 0; i < 9; ++i)
                if (taus[i] > longest)
                    longest = taus[i];
            return longest;
        }

        /* One-pole RC lowpass, tau and dt in the same unit. Mirrors
         * safe_tau_lowpass: the leading alpha applies to the first sample
         * too, which is the 230206 correction upstream carries. */
        static void lowpass(
            std::vector<float>& out,
            const float* signal,
            int n,
            float tau,
            float dt)
        {
            const float alpha = dt / (tau + dt);
            float state = 0.0f;
            out.resize(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i)
            {
                state = alpha * signal[i] + (1.0f - alpha) * state;
                out[static_cast<size_t>(i)] = state;
            }
        }

        /* stim = (a1|LP(s,tau1)| + a2 LP(|s|,tau2) + a3|LP(s,tau3)|)
         *        / stim_limit * g_scale * 100, s in T/m/s. */
        static void evaluate_axis(
            float* out,
            const float* dgdt,
            int n,
            const Axis& axis,
            float dt_ms)
        {
            std::vector<float> rectified(static_cast<size_t>(n));
            std::vector<float> branch;
            int i;

            for (i = 0; i < n; ++i)
                rectified[static_cast<size_t>(i)] = std::fabs(dgdt[i]);

            lowpass(branch, dgdt, n, axis.tau1_ms, dt_ms);
            for (i = 0; i < n; ++i)
                out[i] = axis.a1 * std::fabs(branch[static_cast<size_t>(i)]);

            lowpass(branch, rectified.data(), n, axis.tau2_ms, dt_ms);
            for (i = 0; i < n; ++i)
                out[i] += axis.a2 * branch[static_cast<size_t>(i)];

            lowpass(branch, dgdt, n, axis.tau3_ms, dt_ms);
            for (i = 0; i < n; ++i)
                out[i] += axis.a3 * std::fabs(branch[static_cast<size_t>(i)]);

            const float scale =
                (axis.stim_limit > 0.0f) ? (axis.g_scale * 100.0f / axis.stim_limit) : 0.0f;
            for (i = 0; i < n; ++i)
                out[i] *= scale;
        }

        static int required_padding_cb(void* ctx, float dt_us)
        {
            const SafeParams* self = static_cast<const SafeParams*>(ctx);
            const float dt_ms = dt_us * 1e-3f;
            if (dt_ms <= 0.0f)
                return 0;
            return static_cast<int>(kPaddingTimeConstants * self->longest_tau_ms() / dt_ms) + 1;
        }

        static int evaluate_cb(
            void* ctx,
            const float* dgdt_x,
            const float* dgdt_y,
            const float* dgdt_z,
            int n,
            float dt_us,
            float* out_x,
            float* out_y,
            float* out_z)
        {
            const SafeParams* self = static_cast<const SafeParams*>(ctx);
            const float dt_ms = dt_us * 1e-3f;
            evaluate_axis(out_x, dgdt_x, n, self->x, dt_ms);
            evaluate_axis(out_y, dgdt_y, n, self->y, dt_ms);
            evaluate_axis(out_z, dgdt_z, n, self->z, dt_ms);
            return 0; /* PULSEG_SUCCESS */
        }
    };

    // ── Scan-time info ──────────────────────────────────────────────────

    struct ScanTimeInfo
    {
        float total_duration_us = 0.0f;
        int total_segment_boundaries = 0;

        static ScanTimeInfo from_c(const pulseg_scan_time_info& c)
        {
            ScanTimeInfo s;
            s.total_duration_us = c.total_duration_us;
            s.total_segment_boundaries = c.total_segment_boundaries;
            return s;
        }
    };

    // ── RF statistics ───────────────────────────────────────────────────

    struct RfStats
    {
        float flip_angle_rad = 0.0f;
        float area = 0.0f;
        /** Vendor-specific envelope stats (D8-A); 0 unless opts.vendor_rf_stats_fn
     *  was wired before loading. Meaning is vendor-defined. */
        float vendor_stat[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float duration_us = 0.0f;
        int isodelay_us = 0;
        float bandwidth_hz = 0.0f;
        float base_amplitude_hz = 0.0f;
        int num_samples = 0;

        static RfStats from_c(const pulseg_rf_stats& c)
        {
            RfStats s;
            s.flip_angle_rad = c.flip_angle_rad;
            s.area = c.area;
            s.vendor_stat[0] = c.vendor_stat[0];
            s.vendor_stat[1] = c.vendor_stat[1];
            s.vendor_stat[2] = c.vendor_stat[2];
            s.vendor_stat[3] = c.vendor_stat[3];
            s.duration_us = c.duration_us;
            s.isodelay_us = c.isodelay_us;
            s.bandwidth_hz = c.bandwidth_hz;
            s.base_amplitude_hz = c.base_amplitude_hz;
            s.num_samples = c.num_samples;
            return s;
        }
    };

    // ── Label limits ────────────────────────────────────────────────────

    struct LabelLimit
    {
        int min;
        int max;
        LabelLimit() : min(0), max(0)
        {
        }
        LabelLimit(int mn, int mx) : min(mn), max(mx)
        {
        }
    };

    struct LabelLimits
    {
        LabelLimit slc, phs, rep, avg, seg, set, eco, par, lin, acq;

        static LabelLimits from_c(const pulseg_label_limits& c)
        {
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

    struct GradAxisWaveform
    {
        std::vector<float> time_us;
        std::vector<float> amplitude_hz_per_m;
        std::vector<float> seg_label; // stored as float for uniformity
    };

    struct TrGradientWaveforms
    {
        GradAxisWaveform gx;
        GradAxisWaveform gy;
        GradAxisWaveform gz;
    };

    // ── Native-timing TR waveforms (for plotting) ──────────────────────

    struct ChannelWaveform
    {
        std::vector<float> time_us;
        std::vector<float> amplitude;
    };

    struct AdcEvent
    {
        float onset_us = 0.0f;
        float duration_us = 0.0f;
        int num_samples = 0;
        float freq_offset_hz = 0.0f;
        float phase_offset_rad = 0.0f;
    };

    struct TrBlockDescriptor
    {
        float start_us = 0.0f;
        float duration_us = 0.0f;
        int segment_idx = -1;
        float rf_isocenter_us = -1.0f;
    };

    struct TrWaveforms
    {
        ChannelWaveform gx;       // Hz/m
        ChannelWaveform gy;       // Hz/m
        ChannelWaveform gz;       // Hz/m
        int num_rf_channels = 1;  // 1 for single-Tx, nch for pTx
        ChannelWaveform rf_mag;   // Hz  (channel-major when num_rf_channels > 1)
        ChannelWaveform rf_phase; // rad (channel-major when num_rf_channels > 1)
        std::vector<AdcEvent> adc_events;
        std::vector<TrBlockDescriptor> blocks;
        float total_duration_us = 0.0f;
    };

    // ── Sequence description (state-machine event rows) ──────────────────────────────

    /**
     * One subsequence's canonical-TR event table, as
     * pulseg_get_sequence_description() builds it.
     *
     * The rows cross as the C row type rather than being unpacked here: it is
     * a POD of a tag, a timestamp and seven floats whose meaning depends on
     * the tag (see pulseg_types.h), and re-describing that layout in C++
     * would give it a second place to drift.
     */
    struct SequenceDescription
    {
        int subseq_idx = 0;
        float tr_duration_us = 0.0f;
        std::vector<pulseg_seq_event> rows;
    };

    /**
     * One RF definition's decompressed shapes, in the file's own units:
     * magnitude normalised to a peak near 1, phase in turns, times in us.
     * Channel-major when num_channels > 1.  Empty vectors mean the definition
     * carries no such shape (a common phase, or a uniform time raster).
     */
    struct RfDefinitionShapes
    {
        int num_channels = 1;
        std::vector<float> magnitude;
        std::vector<float> phase_turns;
        std::vector<float> time_us;
    };

    // ── Mechanical resonances spectra ────────────────────────────────────────────────

    struct MechResonancesSpectra
    {
        float freq_min_hz = 0.0f;
        float freq_spacing_hz = 0.0f;
        int num_freq_bins = 0;

        std::vector<float> spectrum_full_gx;
        std::vector<float> spectrum_full_gy;
        std::vector<float> spectrum_full_gz;

        int num_instances = 0;

        /* -- analytical structural spectrum (sparse TR-harmonic grid) -- */
        int num_analytical_peaks = 0;
        std::vector<float> analytical_peak_freqs;
        std::vector<float> analytical_peak_amp_gx;
        std::vector<float> analytical_peak_amp_gy;
        std::vector<float> analytical_peak_amp_gz;
        std::vector<float> analytical_peak_phase_gx;
        std::vector<float> analytical_peak_phase_gy;
        std::vector<float> analytical_peak_phase_gz;
        std::vector<float> analytical_peak_widths_hz;

        /* -- structural candidate frequencies (shared cross-axis) -- */
        int num_candidates = 0;
        std::vector<float> candidate_freqs;
        std::vector<float> candidate_amps_gx;
        std::vector<float> candidate_amps_gy;
        std::vector<float> candidate_amps_gz;
        std::vector<float> candidate_grad_amps;
        std::vector<float> candidate_grad_amps_gx;
        std::vector<float> candidate_grad_amps_gy;
        std::vector<float> candidate_grad_amps_gz;
        std::vector<int> candidate_violations;

        int num_component_terms = 0;
        std::vector<float> component_freqs_hz;
        std::vector<float> component_amps;
        std::vector<float> component_phases_rad;
        std::vector<float> component_widths_hz;
        std::vector<int> component_axes;
        std::vector<int> component_def_ids;
        std::vector<int> component_contrib_ids;
        std::vector<int> component_run_ids;

        int num_surviving_freqs = 0;
        std::vector<float> surviving_freqs_hz;

        /* -- dense analytic envelope (display-only; plotting API only) --
     * Same S_ax(f) transform as analytical_peak_*, evaluated on a uniform
     * grid instead of only at TR harmonics -- see pulseg_types.h. Empty
     * unless the caller requested a display grid (target_resolution_hz>0);
     * never populated on the pulseg_check_safety (PSD) path. */
        int num_envelope_bins = 0;
        std::vector<float> envelope_freqs_hz;
        std::vector<float> envelope_amp_gx;
        std::vector<float> envelope_amp_gy;
        std::vector<float> envelope_amp_gz;
    };

    // ── PNS result ──────────────────────────────────────────────────────

    struct PnsResult
    {
        int num_samples = 0;
        std::vector<float> slew_x;
        std::vector<float> slew_y;
        std::vector<float> slew_z;
    };

    // ── Block instance (cursor output) ──────────────────────────────────

    struct BlockInstance
    {
        int duration_us = 0;
        float rf_amp_hz = 0.0f;
        float rf_freq_hz = 0.0f;
        float rf_phase_rad = 0.0f;
        int rf_shim_id = -1;
        float gx_amp = 0.0f;
        float gy_amp = 0.0f;
        float gz_amp = 0.0f;
        int gx_shape_id = 0;
        int gy_shape_id = 0;
        int gz_shape_id = 0;
        float rotmat[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        int norot_flag = 0;
        int nopos_flag = 0;
        int digitalout_flag = 0;
        int adc_flag = 0;
        float adc_freq_hz = 0.0f;
        float adc_phase_rad = 0.0f;

        static BlockInstance from_c(const pulseg_block_instance& c)
        {
            BlockInstance b;
            b.duration_us = c.duration_us;
            b.rf_amp_hz = c.rf_amp_hz;
            b.rf_freq_hz = c.rf_freq_hz;
            b.rf_phase_rad = c.rf_phase_rad;
            b.rf_shim_id = c.rf_shim_id;
            b.gx_amp = c.gx_amp_hz_per_m;
            b.gy_amp = c.gy_amp_hz_per_m;
            b.gz_amp = c.gz_amp_hz_per_m;
            b.gx_shape_id = c.gx_shape_id;
            b.gy_shape_id = c.gy_shape_id;
            b.gz_shape_id = c.gz_shape_id;
            for (int i = 0; i < 9; ++i)
                b.rotmat[i] = c.rotmat[i];
            b.norot_flag = c.norot_flag;
            b.nopos_flag = c.nopos_flag;
            b.digitalout_flag = c.digitalout_flag;
            b.adc_flag = c.adc_flag;
            b.adc_freq_hz = c.adc_freq_hz;
            b.adc_phase_rad = c.adc_phase_rad;
            return b;
        }
    };

    // ── Segment layout (subsequence-local segment, resolved to a scan
    //    instance) ───────────────────────────────────────────────────────

    /**
 * Resolved position/layout of one subsequence-local unique segment, plus
 * the segment metadata already exposed by Collection::segment_info().
 * See Collection::segments().
 */
    struct SegmentLayout
    {
        int global_index = -1; ///< deduplicated global segment id
        int num_blocks = 0;
        int start_block = 0; ///< segment definition's own start block
        int max_energy_start_block =
            -1; ///< start block of the max-energy scan instance, -1 if none
        bool from_max_energy_instance = false;
        std::vector<int> block_indices; ///< resolved .seq block indices; empty if unresolved
        int duration_us = 0;
        bool pure_delay = false;
        bool is_nav = false;
        bool has_trigger = false;
    };

} // namespace pulseg

#endif // PULSEG_TYPES_HPP
