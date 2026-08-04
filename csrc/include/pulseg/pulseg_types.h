/**
 * @file pulseg_types.h
 * @brief Public type definitions, error codes, and initializer macros.
 *
 * All types intended for consumption by calling code are defined here.
 * Internal / opaque types live in pulseg_internal.h.
 *
 * Naming conventions:
 *   - Physical quantities carry unit suffixes: _us, _hz, _hz_per_m, etc.
 *   - All identifiers are snake_case.
 *   - INIT macros are C89 and C++ compatible (positional, no designated init).
 */

#ifndef PULSEG_TYPES_H
#define PULSEG_TYPES_H

#include "pulseq_types.h"

#include "pulseg_config.h"
#include "pulseg_errors.h"

/* ================================================================== */
/*  Gradient axes                                                     */
/* ================================================================== */
#define PULSEG_GRAD_AXIS_X 0
#define PULSEG_GRAD_AXIS_Y 1
#define PULSEG_GRAD_AXIS_Z 2

/* ================================================================== */
/*  RF use codes (pulseg_seq_event.params[1] for RF rows).            */
/*  Aliases of the raw Pulseq RF library's trailing e/r/i/s use tag,  */
/*  which the pulseq module owns.                                     */
/* ================================================================== */
#define PULSEG_RF_USE_UNKNOWN PULSEQ_RF_USE_UNKNOWN
#define PULSEG_RF_USE_EXCITATION PULSEQ_RF_USE_EXCITATION
#define PULSEG_RF_USE_REFOCUSING PULSEQ_RF_USE_REFOCUSING
#define PULSEG_RF_USE_INVERSION PULSEQ_RF_USE_INVERSION
#define PULSEG_RF_USE_SATURATION PULSEQ_RF_USE_SATURATION

/* ================================================================== */
/*  Error codes                                                       */
/* ================================================================== */

/** @defgroup errcodes Error codes
 *  Every public function returns a plain int:
 *    positive  = success (PULSEG_OK)
 *    negative  = failure
 *
 *  On failure the caller should read the diagnostic message string
 *  (filled by every function that accepts a pulseg_diagnostic*)
 *  and pass it to the vendor error-reporting routine.  Specific
 *  negative values are library-internal and must NOT be matched by
 *  consumers.
 *  @{ */

#define PULSEG_SUCCESS 1

#define PULSEG_SUCCEEDED(code) ((code) > 0)
#define PULSEG_FAILED(code) ((code) < 0)

/** @} */

/* ================================================================== */
/*  Cursor states                                                     */
/* ================================================================== */
#define PULSEG_CURSOR_BLOCK 0
#define PULSEG_CURSOR_DONE 1

/* ================================================================== */
/*  Max-size constants                                                */
/* ================================================================== */
#define PULSEG_MAX_GRAD_SHOTS 16
#define PULSEG_DIAG_MSG_LEN 256

/* ================================================================== */
/*  Diagnostic                                                        */
/* ================================================================== */

/**
 * @brief Diagnostic info returned by library functions on failure.
 *
 * On error, @c code is set to a negative PULSEG_ERR_* value and
 * @c message contains a human-readable description (may include
 * offending block index, axis, amplitude, etc.).
 */
typedef struct pulseg_diagnostic
{
    int code;
    char message[PULSEG_DIAG_MSG_LEN];
} pulseg_diagnostic;

/* clang-format off */
#define PULSEG_DIAGNOSTIC_INIT {PULSEG_SUCCESS, { '\0' }}
/* clang-format on */

/* ================================================================== */
/*  Shape (RLE-decompressible waveform)                               */
/*                                                                    */
/*  Owned by the pulseq module (pulseq_types.h) -- the IR keeps Pulseq */
/*  RLE shapes as its waveform store, so the raw shape library entry   */
/*  and the descriptor's shape storage are deliberately the same type. */
/*  pulseg_shape_arbitrary remains available as a compatibility alias. */
/* ================================================================== */

typedef pulseq_shape pulseg_shape_arbitrary;

/* ================================================================== */
/*  RF envelope view (for the vendor RF-stats callback)               */
/* ================================================================== */

/**
 * @brief Read-only view of a uniform-raster RF envelope, handed to an
 * optional vendor callback (@c pulseg_opts.vendor_rf_stats_fn) so it can
 * compute vendor-specific envelope statistics without owning any of the
 * dedup-time buffers.
 */
typedef struct pulseg_rf_view
{
    const float *mag;     /**< |B1(t)| envelope, normalised, length n */
    const float *phase;   /**< phase (rad), length n                  */
    int n;                /**< sample count                           */
    float dt_us;          /**< uniform raster period (us)             */
    float duration_us;    /**< RF event duration (us)                 */
    float tr_duration_us; /**< enclosing TR duration (us); 0 if unknown at
                                 dedup time                            */
} pulseg_rf_view;

/* ================================================================== */
/*  System options                                                    */
/* ================================================================== */

/**
 * @brief Scanner hardware limits and raster times.
 *
 * All raster times are in microseconds.  Gradient / slew limits use
 * internal Pulseq units (Hz/m and Hz/m/s respectively).
 */
typedef struct pulseg_opts
{
    int vendor;                    /**< PULSEG_VENDOR_* constant       */
    float gamma_hz_per_t;          /**< gyromagnetic ratio  (Hz / T)      */
    float b0_t;                    /**< static field strength (T)         */
    float max_grad_hz_per_m;       /**< gradient amplitude limit (Hz / m) */
    float max_slew_hz_per_m_per_s; /**< slew rate limit (Hz / m / s)      */
    float rf_raster_us;            /**< RF sample raster (us)             */
    float grad_raster_us;          /**< gradient sample raster (us)       */
    float adc_raster_us;           /**< ADC dwell raster (us)             */
    float block_raster_us;         /**< block duration raster (us)        */
    float peak_log10_threshold;    /**< resonance detector log10 threshold */
    float peak_norm_scale;         /**< resonance detector normalization   */
    float peak_eps;                /**< resonance detector epsilon         */
    float peak_prominence;         /**< resonance detector min prominence  */

    /** Optional vendor RF envelope-stats callback. NULL -> the four
     *  pulseg_rf_stats.vendor_stat[] slots are left at 0. */
    int (*vendor_rf_stats_fn)(void *ctx, const pulseg_rf_view *rf, float out_stat[4]);
    void *vendor_rf_stats_ctx;

    /**
     * @brief Which Pulseq label fills output column 0/1/2 of the 3-column
     * ADC label table. Values are Pulseq label *state-array* indices:
     * 0=SLC, 1=PHS, 2=REP, 3=AVG, 4=SEG, 5=SET, 6=ECO, 7=PAR, 8=LIN, 9=ACQ.
     * Example (GE convention): {8, 0, 6} = [LIN, SLC, ECO]. Public default
     * is the identity {0, 1, 2} = [SLC, PHS, REP]; vendor layers override
     * this before parsing (see pulserver_ge_config.h in the private
     * pulserver-interpreter for the GE values).
     */
    int label_column_map[3];

    /** Binary cache file extension, including the dot. Default
     *  ".pseg"; GE overrides to ".pge" (see pulserver_ge_config.h). Only
     *  the main pulseg_read()/pulseg__write_cache() path honors this;
     *  standalone cache utilities (pulseg_load_cache, pulseg_clear_cache,
     *  etc.) that run before any collection exists always use the public
     *  default. */
    char cache_ext[PULSEG_CACHE_EXT_MAX];

    /** Optional opaque vendor cache section. Writer emits a section
     *  only when set; GE leaves this unused. ctx/buf ownership: the
     *  callback allocates *out_buf via PULSEG_ALLOC; the cache writer
     *  frees it after use. */
    int (*vendor_section_write_fn)(void *ctx, unsigned char **out_buf, int *out_len);
    void *vendor_section_ctx;

    /**
     * @brief Accept RF amplitude that varies across canonical TR instances.
     *
     * Default 1. When set, a subsequence whose positional RF amplitude
     * pattern differs between TR instances (or passes) is accepted instead
     * of returning @c PULSEG_ERR_CONSISTENCY_RF_PERIODIC; the descriptor's
     * @c rf_amplitude_variable flag is raised and the RF safety model is
     * built from the *positional-max envelope* rather than one canonical
     * instance -- see pulseg_get_rf_array().
     *
     * RF *shim* pattern variation stays rejected regardless of this flag:
     * VOP SAR with changing shim vectors is not order-monotone in any
     * per-position scalar, so no envelope dominates it.
     *
     * Set to 0 to restore the strict periodicity gate.
     */
    int allow_variable_rf_amplitude;
} pulseg_opts;

/* clang-format off */
#define PULSEG_OPTS_INIT \
    { \
    0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, \
    PULSEG_PEAK_LOG10_THRESHOLD_DEFAULT, PULSEG_PEAK_NORM_SCALE_DEFAULT, \
    PULSEG_PEAK_EPS_DEFAULT, PULSEG_PEAK_PROMINENCE_DEFAULT, NULL, NULL, {0, 1, 2}, \
    PULSEG_CACHE_EXT_DEFAULT, NULL, NULL, 1 \
    }
/* clang-format on */

/* ================================================================== */
/*  RF statistics                                                     */
/* ================================================================== */

/** Maximum simultaneous frequency bands detectable in a multiband RF pulse. */
#define PULSEG_MAX_BANDS 8

/**
 * @brief Per-RF-definition statistics (always available).
 */
typedef struct pulseg_rf_stats
{
    float flip_angle_rad;   /**< nominal flip angle (radians)           */
    float act_amplitude_hz; /**< actual |gamma*B1| amplitude (Hz). When the
                              *  descriptor's rf_amplitude_variable flag is
                              *  set, this is the REAL amplitude of the
                              *  worst-B1rms TR instance at this position
                              *  (not a synthetic per-position envelope) --
                              *  feeds time-averaged SAR / amplifier-duty
                              *  consumers (GE minseqrfamp/maxsar). See
                              *  peak_amplitude_hz for the peak-dominant
                              *  counterpart. */
    float peak_amplitude_hz; /**< positional-max |gamma*B1| amplitude (Hz)
                               *  across every TR instance at this position
                               *  -- for peak-only consumers (GE peakB1())
                               *  that need per-position dominance across
                               *  ALL instances, which act_amplitude_hz no
                               *  longer guarantees once it tracks a single
                               *  real worst-B1rms instance. Equal to
                               *  act_amplitude_hz for periodic sequences. */
    float area;             /**< integral of |B1(t)| dt  (a.u.)        */
    /** Vendor-specific envelope statistics, filled by the optional
     *  pulseg_opts.vendor_rf_stats_fn callback; all 0 when unset. Meaning
     *  is vendor-defined -- e.g. GE's abswidth/effwidth/dtycyc/maxpw live
     *  in src_gelib/pulserver_ge_rf_stats.h as PULSERVER_GE_RF_* accessors. */
    float vendor_stat[4];
    float duration_us;       /**< total RF event duration (us)          */
    int isodelay_us;         /**< isodelay from center to echo (us)     */
    float bandwidth_hz;      /**< estimated bandwidth (Hz, via FFT)     */
    float base_amplitude_hz; /**< base (nominal) peak |gamma*B1| (Hz)   */
    int num_samples;         /**< waveform sample count                 */
    int num_instances;       /**< repetition count for this RF pulse    */
    /* --- multiband / power fields (appended; do not reorder above) --- */
    int num_bands; /**< number of simultaneous frequency bands (>=1) */
    float band_freq_offsets_hz
        [PULSEG_MAX_BANDS];  /**< per-band center offsets relative to carrier (Hz) */
    float band_bandwidth_hz; /**< per-band bandwidth (Hz) */
    float total_b1sq_power;  /**< integral |B1(t)|^2 dt normalised (a.u.) */
    /* --- vendor tag (appended; identifies the meaning of the
     *     vendor-specific interpretation of the fields above; for new
     *     vendor variants, a sibling struct may be added later and
     *     selected via this field) ---                                 */
    int vendor; /**< PULSEG_VENDOR_* constant (0 = unspecified -> GEHC for back-compat) */
    /* --- safety-group module label (appended; do not reorder above) --- */
    int module_id; /**< sticky MODULE label id of the originating block, 0 = ungrouped */
} pulseg_rf_stats;

/* clang-format off */
#define PULSEG_RF_STATS_INIT \
    { \
    0.0f, 0.0f, 0.0f, 0.0f, {0.0f}, 0.0f, 0, 0.0f, 0.0f, 0, 0, 1, {0.0f}, 0.0f, 0.0f, 0, 0 \
    }
/* clang-format on */

/**
 * @brief One entry per distinct MODULE-labeled group in a subsequence,
 * as identified/verified/deduplicated by pulseg_get_modules() from the
 * materialized scan table. See pulseg_get_modules() for the identify ->
 * verify-structural-identity -> dedup algorithm.
 */
typedef struct pulseg_module
{
    int module_id;                /**< sticky MODULE label id (>=1)      */
    int one_instance_duration_us; /**< duration of the validated reference occurrence */
    int total_duration_us;        /**< num_instances * one_instance_duration_us */
    int num_instances;            /**< count of structurally-identical occurrences */
} pulseg_module;

/* ================================================================== */
/*  TR region selectors (for freq-mod plan)                           */
/* ================================================================== */
#define PULSEG_TR_REGION_PREP 0
#define PULSEG_TR_REGION_MAIN 1
#define PULSEG_TR_REGION_COOLDOWN 2

/* ================================================================== */
/*  Frequency modulation collection                                   */
/* ================================================================== */

/**
 * @brief Opaque handle to frequency modulation data for all subsequences.
 *
 * Wraps per-subsequence libraries into a single object.  The entire
 * collection is built, cached, and freed as a unit.
 *
 * Created by pulseg_build_freq_mod_collection() or
 * pulseg_freq_mod_collection_read_cache(), queried via
 * pulseg_freq_mod_collection_get(), freed by
 * pulseg_freq_mod_collection_free().
 */
typedef struct pulseg_freq_mod_collection pulseg_freq_mod_collection;

/* ================================================================== */
/*  Opaque collection handle                                          */
/* ================================================================== */

/**
 * @brief Opaque handle to a loaded Pulseq sequence collection.
 *
 * Created by pulseg_read() or pulseg_read_from_buffers().
 * All getter functions take a const pointer to this type.
 * Freed by pulseg_collection_free().
 */
typedef struct pulseg_collection pulseg_collection;

/* ================================================================== */
/*  Per-axis gradient waveform (for plotting)                         */
/* ================================================================== */

/**
 * @brief Single-axis gradient waveform with per-sample segment label.
 *
 * Each element i represents a time-point with amplitude and the
 * segment it belongs to.  The time array is NOT interpolated to a
 * uniform raster -- it follows the native event timing.
 */
typedef struct pulseg_grad_axis_waveform
{
    int num_samples;           /**< number of time-points          */
    float *time_us;            /**< time of each sample (us)       */
    float *amplitude_hz_per_m; /**< gradient amplitude (Hz / m)    */
    int *seg_label;            /**< segment index for each sample  */
} pulseg_grad_axis_waveform;

/* clang-format off */
#define PULSEG_GRAD_AXIS_WAVEFORM_INIT {0, NULL, NULL, NULL}
/* clang-format on */

/**
 * @brief Per-TR gradient waveforms for all three axes.
 *
 * Used for gradient-shape plotting in the wrapper.  Each axis carries
 * its own time base (not interpolated to a common raster).
 */
typedef struct pulseg_tr_gradient_waveforms
{
    pulseg_grad_axis_waveform gx;
    pulseg_grad_axis_waveform gy;
    pulseg_grad_axis_waveform gz;
} pulseg_tr_gradient_waveforms;

/* clang-format off */
#define PULSEG_TR_GRADIENT_WAVEFORMS_INIT \
    { \
    PULSEG_GRAD_AXIS_WAVEFORM_INIT, PULSEG_GRAD_AXIS_WAVEFORM_INIT, \
    PULSEG_GRAD_AXIS_WAVEFORM_INIT \
    }
/* clang-format on */

/* ================================================================== */
/*  Native-timing TR waveforms (for plotting)                        */
/* ================================================================== */

/** @brief Amplitude modes for pulseg_get_tr_waveforms. */
#define PULSEG_AMP_MAX_POS 0  /**< Position-max (safety worst case) */
#define PULSEG_AMP_ZERO_VAR 1 /**< Zero variable-amplitude gradients, keep constant ones */
#define PULSEG_AMP_ACTUAL 2   /**< Actual signed amplitude for given TR */

/**
 * @brief Single-channel waveform with native (non-uniform) timing.
 *
 * Both arrays have @c num_samples elements.  Units depend on the
 * channel:  Hz/m for gradients, Hz for RF magnitude, radians for
 * RF phase.
 */
typedef struct pulseg_channel_waveform
{
    int num_samples;
    float *time_us;   /**< [num_samples] */
    float *amplitude; /**< [num_samples] */
} pulseg_channel_waveform;

/* clang-format off */
#define PULSEG_CHANNEL_WAVEFORM_INIT {0, NULL, NULL}
/* clang-format on */

/**
 * @brief ADC event descriptor within a TR.
 */
typedef struct pulseg_adc_event
{
    float onset_us;         /**< start time within TR (us)         */
    float duration_us;      /**< num_samples * dwell_time (us)     */
    int num_samples;        /**< number of ADC samples             */
    float freq_offset_hz;   /**< per-instance freq offset (Hz)     */
    float phase_offset_rad; /**< per-instance phase offset (rad)   */
} pulseg_adc_event;

/**
 * @brief Per-block metadata within a TR.
 */
typedef struct pulseg_tr_block_descriptor
{
    float start_us;        /**< block start time within TR (us)   */
    float duration_us;     /**< block duration (us)               */
    int segment_idx;       /**< segment index, or -1 (prep/cooldown) */
    float rf_isocenter_us; /**< RF isocenter time within TR (us), or -1.0 */
    float adc_kzero_us;    /**< ADC k=0 time within TR (us), or -1.0 */
} pulseg_tr_block_descriptor;

/**
 * @brief Complete native-timing TR waveforms for plotting.
 *
 * Each channel carries its own time base.  Gradient channels
 * preserve native timing (trap corner-points, arb raster samples).
 * RF channels use the RF raster.  ADC events are descriptors only.
 *
 * Block descriptors provide timing and segment assignment for
 * drawing block/segment boundaries.
 */
typedef struct pulseg_tr_waveforms
{
    /* Gradient channels (Hz/m) */
    pulseg_channel_waveform gx;
    pulseg_channel_waveform gy;
    pulseg_channel_waveform gz;

    /* RF channels.  num_rf_channels == 1 for single-Tx.  For pTx
     * (num_rf_channels > 1), rf_mag.amplitude and rf_phase.amplitude
     * are channel-major flat arrays: ch0[0..npts-1], ch1[0..npts-1], ...
     * rf_mag.num_samples == num_rf_channels * npts_per_channel.       */
    int num_rf_channels;              /**< 1 for single-Tx, nch for pTx */
    pulseg_channel_waveform rf_mag;   /**< amplitude in Hz           */
    pulseg_channel_waveform rf_phase; /**< amplitude in rad          */

    /* ADC events */
    int num_adc_events;
    pulseg_adc_event *adc_events;

    /* Block-level metadata */
    int num_blocks;
    pulseg_tr_block_descriptor *blocks;

    /* Total duration */
    float total_duration_us;
} pulseg_tr_waveforms;

/* ================================================================== */
/*  Mechanical resonances spectra (for plotting)                      */
/* ================================================================== */

/**
 * @brief Mechanical resonances spectral data for wrapper-side plotting.
 *
 * Frequency axes are specified by (min, spacing, num_bins) so
 * the caller can reconstruct: freq[k] = freq_min_hz + k * freq_spacing_hz.
 *
 * The canonical mechanical-resonance verdict is provided by the
 * structural-analysis arrays (analytical_*, candidate_*, component_*,
 * surviving_*).  spectrum_full_g{x,y,z} are display-only full-TR
 * magnitude spectra.
 */
typedef struct pulseg_mech_resonances_spectra
{
    /* -- full TR spectrum (display-only) --------------------------- */
    float freq_min_hz;       /**< lowest frequency bin (Hz)         */
    float freq_spacing_hz;   /**< bin width (Hz)                    */
    int num_freq_bins;       /**< frequency bins                    */
    float *spectrum_full_gx; /**< [num_freq_bins]                   */
    float *spectrum_full_gy;
    float *spectrum_full_gz;

    /* -- repetition info ------------------------------------------- */
    int num_instances; /**< TR repetition count (for display) */

    /* -- analytical structural spectrum (sparse TR-harmonic grid) -- */
    int num_analytical_peaks;      /**< evaluated harmonic count        */
    float *analytical_peak_freqs;  /**< [num_analytical_peaks] (Hz)     */
    float *analytical_peak_amp_gx; /**< [num_analytical_peaks] |S_gx|   */
    float *analytical_peak_amp_gy;
    float *analytical_peak_amp_gz;
    float *analytical_peak_phase_gx; /**< [num_analytical_peaks] arg(S_gx) (rad) */
    float *analytical_peak_phase_gy;
    float *analytical_peak_phase_gz;
    float *analytical_peak_widths_hz; /**< [num_analytical_peaks] FWHM (Hz) */

    /* -- structural candidate frequencies (shared cross-axis) ----- */
    int num_candidates;       /**< candidate count (shared)         */
    float *candidate_freqs;   /**< [num_candidates] (Hz)            */
    float *candidate_amps_gx; /**< per-axis analytical amplitudes   */
    float *candidate_amps_gy;
    float *candidate_amps_gz;
    float *candidate_grad_amps;    /**< max time-domain grad amp (Hz/m)  */
    float *candidate_grad_amps_gx; /**< per-axis contributing grad amp (Hz/m) */
    float *candidate_grad_amps_gy;
    float *candidate_grad_amps_gz;
    int *candidate_violations; /**< 1 = violates a band              */

    /* -- component-level sparse analytical terms ------------------ */
    int num_component_terms;     /**< number of sparse component terms */
    float *component_freqs_hz;   /**< [num_component_terms] term center (Hz) */
    float *component_amps;       /**< [num_component_terms] |term| (Hz/m) */
    float *component_phases_rad; /**< [num_component_terms] arg(term) (rad) */
    float *component_widths_hz;  /**< [num_component_terms] FWHM (Hz) */
    int *component_axes;         /**< [num_component_terms] 0=gx,1=gy,2=gz */
    int *component_def_ids;      /**< [num_component_terms] grad def id */
    int *component_contrib_ids;  /**< [num_component_terms] axis-local contrib id */
    int *component_run_ids;      /**< [num_component_terms] run index within contrib */

    /* -- surviving sparse peak positions (positions only) --------- */
    int num_surviving_freqs;   /**< surviving candidate frequency count */
    float *surviving_freqs_hz; /**< [num_surviving_freqs] (Hz) */

    /* -- dense analytic envelope (display-only; plotting API only) ---
     * The SAME closed-form S_ax(f) transform as analytical_peak_*, evaluated
     * on a dense uniform grid (spectrum_full's freq_min_hz/freq_spacing_hz)
     * instead of only at TR harmonics k/T_TR.  Because analytical_peak_* is
     * literally this function sampled at k/T_TR, this array passes exactly
     * through every analytical_peak_* point -- a true matched envelope, not
     * an interpolation and not a separately-windowed/normalised FFT.
     * Never populated on the pulseg_check_safety (PSD) path -- see
     * calc_mech_resonances_from_uniform's compute_dense_envelope gate. */
    int num_envelope_bins;    /**< dense envelope sample count (0 = not computed) */
    float *envelope_freqs_hz; /**< [num_envelope_bins] (Hz), uniform grid    */
    float *envelope_amp_gx;   /**< [num_envelope_bins] A_eq(f) = (2/T_TR)|S_gx(f)| (Hz/m) */
    float *envelope_amp_gy;
    float *envelope_amp_gz;
} pulseg_mech_resonances_spectra;

/* clang-format off */
#define PULSEG_MECH_RESONANCES_SPECTRA_INIT \
    { \
    /* freq_min_hz, freq_spacing_hz, num_freq_bins */ 0.0f, 0.0f, 0, /* \
    spectrum_full_gx/gy/gz */ NULL, NULL, NULL, /* num_instances */ 0, /* \
    num_analytical_peaks, analytical_peak_freqs, amp_gx/gy/gz, phase_gx/gy/gz, widths */ \
    0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, /* num_candidates, \
    candidate_freqs, amps_gx/gy/gz, grad_amps, grad_amps_gx/gy/gz, violations */ 0, \
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, /* num_component_terms, \
    component_{freqs,amps,phases,widths,axes,def_ids,contrib_ids,run_ids} */ 0, NULL, \
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, /* num_surviving_freqs, surviving_freqs_hz \
    */ 0, NULL, /* num_envelope_bins, envelope_freqs_hz, amp_gx/gy/gz */ 0, NULL, NULL, \
    NULL \
    }
/* clang-format on */

/* ================================================================== */
/*  Forbidden frequency band (for mechanical resonance check)         */
/* ================================================================== */

/**
 * @brief A forbidden mechanical resonance frequency band.
 *
 * @c max_amplitude_hz_per_m is the maximum allowed gradient spectral
 * amplitude (in Hz / m) within the band [freq_min_hz, freq_max_hz].
 */
typedef struct pulseg_forbidden_band
{
    float freq_min_hz;            /**< lower band edge (Hz)          */
    float freq_max_hz;            /**< upper band edge (Hz)          */
    float max_amplitude_hz_per_m; /**< max spectral amplitude (Hz/m) */
} pulseg_forbidden_band;

/* ================================================================== */
/*  PNS evaluator (vendor-pluggable model)                            */
/* ================================================================== */

/**
 * @brief Vendor-pluggable PNS model.
 *
 * The public library owns the vendor-neutral half of PNS evaluation
 * (canonical-TR selection, uniform-raster dG/dt extraction, combined
 * sqrt(x^2+y^2+z^2), result marshalling). The model half -- the actual
 * stimulation-threshold functional form (e.g. GE's rheobase-chronaxie
 * (Irnich/den Boer) `c/(c+tau)^2` kernel, or Siemens SAFE's nonlinear
 * multi-stage filter) -- is injected through this struct. Only an evaluator
 * interface (not a sampled-kernel API) can represent both forms.
 *
 * Calling convention (enforced by pulseg_calc_pns / pulseg_check_safety,
 * not by the model): before differentiating the uniform-raster gradient
 * waveforms, the safety core calls @c required_padding(ctx, dt_us) to
 * learn how many extra circularly-wrapped samples the model needs
 * appended so its filter sees a fully "warmed up" history (0 if none).
 * It then calls @c evaluate() once with the resulting dG/dt arrays,
 * all of length @c n (already including that padding) -- @c evaluate
 * must return exactly @c n output samples per axis.
 */
typedef struct pulseg_pns_model
{
    void *ctx; /**< opaque model state (vendor-owned) */

    /**
     * @brief Report how many extra circular-wrap dG/dt samples this
     * model needs appended before it is called, for a given raster.
     * @param ctx    Opaque model state.
     * @param dt_us  Gradient raster period (us).
     * @return Number of extra samples (>= 0).
     */
    int (*required_padding)(void *ctx, float dt_us);

    /**
     * @brief Evaluate the model on uniform-raster dG/dt waveforms.
     * @param ctx     Opaque model state.
     * @param dgdt_x  Per-axis dG/dt, X (Hz/m/s), length n.
     * @param dgdt_y  Per-axis dG/dt, Y (Hz/m/s), length n.
     * @param dgdt_z  Per-axis dG/dt, Z (Hz/m/s), length n.
     * @param n       Number of samples (same length for all in/out arrays).
     * @param dt_us   Gradient raster period (us).
     * @param out_x   Receives per-axis result, X (% of threshold), length n.
     * @param out_y   Receives per-axis result, Y (% of threshold), length n.
     * @param out_z   Receives per-axis result, Z (% of threshold), length n.
     * @return PULSEG_SUCCESS on success, negative error code on failure.
     */
    int (*evaluate)(
        void *ctx,
        const float *dgdt_x,
        const float *dgdt_y,
        const float *dgdt_z,
        int n,
        float dt_us,
        float *out_x,
        float *out_y,
        float *out_z);

    /**
     * @brief Optional: expose this model as a linear time-invariant
     * filter by handing back its impulse response.
     *
     * Setting this field is a claim about the model, not just a
     * convenience: it asserts that @c evaluate is exactly the discrete
     * convolution of each dG/dt axis with the returned kernel, followed
     * by a per-sample scaling this call reports in @c out_scale. A model
     * with any nonlinearity after the filter (thresholding, rectifying,
     * cross-axis coupling) must leave this NULL.
     *
     * When it is set, the safety core may take a faster route to the
     * same numbers: because convolution is linear, it can convolve each
     * distinct gradient shape in the sequence once and then accumulate
     * scaled, time-shifted copies, instead of convolving a waveform that
     * spans the whole canonical TR. On a long TR built from a handful of
     * repeated shapes that is one to two orders of magnitude cheaper.
     * Leaving this NULL is always safe -- the core falls back to
     * @c evaluate over the full waveform.
     *
     * The kernel stays vendor-owned: the core treats it as opaque data
     * and never inspects its shape or the constants behind it.
     *
     * @param ctx         Opaque model state.
     * @param dt_us       Gradient raster period (us).
     * @param out_kernel  Receives a newly allocated kernel; the core
     *                    frees it with PULSEG_FREE.
     * @param out_len     Receives the kernel length (> 0).
     * @param out_scale   Receives the per-sample output scaling
     *                    @c evaluate applies after convolving (1.0 if
     *                    none) -- e.g. 100 for a model reporting
     *                    percent-of-threshold.
     * @return PULSEG_SUCCESS on success, negative error code on failure.
     */
    int (*kernel)(
        void *ctx,
        float dt_us,
        float **out_kernel,
        int *out_len,
        float *out_scale);
} pulseg_pns_model;

/* clang-format off */
#define PULSEG_PNS_MODEL_INIT {NULL, NULL, NULL, NULL}
/* clang-format on */

/* ================================================================== */
/*  PNS result (for plotting)                                         */
/* ================================================================== */

/**
 * @brief Convolved slew-rate waveforms per axis.
 *
 * The wrapper can compute combined PNS = sqrt(x^2+y^2+z^2) and the
 * percentage per the injected model's threshold normalization.  This
 * avoids duplicating model logic across languages.
 */
typedef struct pulseg_pns_result
{
    int num_samples;
    float *slew_x_hz_per_m_per_s; /**< convolved dG/dt on X (Hz/m/s) */
    float *slew_y_hz_per_m_per_s; /**< convolved dG/dt on Y (Hz/m/s) */
    float *slew_z_hz_per_m_per_s; /**< convolved dG/dt on Z (Hz/m/s) */
} pulseg_pns_result;

/* clang-format off */
#define PULSEG_PNS_RESULT_INIT {0, NULL, NULL, NULL}
/* clang-format on */

/* ================================================================== */
/*  Label limits                                                      */
/*  pulseq_label_limit (the per-label min/max pair) is owned by the    */
/*  pulseq module; pulseg groups one per Pulseq label below.           */
/* ================================================================== */

typedef pulseq_label_limit pulseg_label_limit;

/** @brief Observed [min, max] of every Pulseq counter label in a subsequence. */
typedef struct pulseg_label_limits
{
    pulseq_label_limit slc;
    pulseq_label_limit phs;
    pulseq_label_limit rep;
    pulseq_label_limit avg;
    pulseq_label_limit seg;
    pulseq_label_limit set;
    pulseq_label_limit eco;
    pulseq_label_limit par;
    pulseq_label_limit lin;
    pulseq_label_limit acq;
} pulseg_label_limits;

/* ================================================================== */
/*  Block instance (cursor output)                                    */
/* ================================================================== */

/**
 * @brief Resolved block data for the current cursor position.
 *
 * Returned by pulseg_get_block_instance().  Amplitudes are in
 * Pulseq native units (Hz for RF, Hz/m for gradients).
 */
typedef struct pulseg_block_instance
{
    int duration_us; /**< block duration (us)                */

    /* RF */
    float rf_amp_hz;    /**< RF amplitude (Hz, = gamma*B1)     */
    float rf_freq_hz;   /**< RF frequency offset (Hz)          */
    float rf_phase_rad; /**< RF phase offset (rad)             */
    int rf_shim_id;     /**< RF shim definition index (-1=none)*/

    /* Gradients */
    float gx_amp_hz_per_m; /**< GX amplitude (Hz / m)             */
    float gy_amp_hz_per_m; /**< GY amplitude (Hz / m)             */
    float gz_amp_hz_per_m; /**< GZ amplitude (Hz / m)             */
    int gx_shot_idx;       /**< GX multi-shot index               */
    int gy_shot_idx;       /**< GY multi-shot index               */
    int gz_shot_idx;       /**< GZ multi-shot index               */
    int gx_variable;       /**< 1 if GX amplitude varies across TRs */
    int gy_variable;       /**< 1 if GY amplitude varies across TRs */
    int gz_variable;       /**< 1 if GZ amplitude varies across TRs */

    /* Rotation */
    float rotmat[9]; /**< 3x3 rotation matrix (row-major)   */
    int norot_flag;  /**< 1 = skip rotation for this block  */
    int nopos_flag;  /**< 1 = skip repositioning            */

    /* Digital output */
    int digitalout_flag;    /**< 1 = digital output event present  */
    int digitalout_channel; /**< trigger channel, -1 if absent     */

    /* ADC */
    int adc_flag;        /**< 1 = ADC acquisition active        */
    float adc_freq_hz;   /**< ADC frequency offset (Hz)         */
    float adc_phase_rad; /**< ADC phase offset (rad)            */

    /* Safety-group module label (sticky pulseq MODULE, 0 = ungrouped) */
    int module_id;
} pulseg_block_instance;

/* clang-format off */
#define PULSEG_BLOCK_INSTANCE_INIT \
    { \
    0, 0.0f, 0.0f, 0.0f, -1, 0.0f, 0.0f, 0.0f, 0, 0, 0, 0, 0, 0, {1, 0, 0, 0, 1, 0, 0, \
    0, 1}, 0, 0, 0, -1, 0, 0.0f, 0.0f, 0 \
    }
/* clang-format on */

/* ================================================================== */
/*  Cursor info                                                       */
/* ================================================================== */

/**
 * @brief Position and context metadata for the current cursor block.
 *
 * Returned by pulseg_cursor_get_info() after a successful
 * pulseg_cursor_next() call.
 */
typedef struct pulseg_cursor_info
{
    int subseq_idx;    /**< current subsequence index                     */
    int scan_pos;      /**< scan-table position (for freq-mod lookup)     */
    int segment_id;    /**< current segment ID (global)                   */
    int segment_start; /**< 1 if first block of current segment           */
    int segment_end;   /**< 1 if last block of current segment            */
    int is_nav;        /**< 1 if current segment is a NAV segment         */
    int has_trigger;   /**< 1 if current segment has a trigger/digitalout */
    int tr_start;      /**< 1 if first block of a main-region TR          */
    int pmc;           /**< 1 if current subsequence has PMC enabled      */
} pulseg_cursor_info;

/* clang-format off */
#define PULSEG_CURSOR_INFO_INIT {0, 0, -1, 0, 0, 0, 0, 0, 0}
/* clang-format on */

/* ================================================================== */
/*  Scan-time query result                                            */
/* ================================================================== */

/**
 * @brief Scan-time summary.
 *
 * When returned by pulseg_peek_scan_time(), only
 * @c total_duration_us is populated (approximated from the
 * [DEFINITIONS] section, multiplied by @c num_reps with
 * per-subsequence @c IgnoreAverages clamping) and
 * @c total_segment_boundaries is left at 0.
 *
 * When computed from a fully-loaded collection via
 * pulseg_get_scan_time(), both fields are accurate and
 * account for prep/cooldown block durations, degenerate
 * TR folding, and the consumer-supplied @c num_reps.
 */
typedef struct pulseg_scan_time_info
{
    float total_duration_us;      /**< total sequence duration (us)  */
    int total_segment_boundaries; /**< total segment boundary count  */
} pulseg_scan_time_info;

/* clang-format off */
#define PULSEG_SCAN_TIME_INFO_INIT {0.0f, 0}
/* clang-format on */

/* ================================================================== */
/*  Collection info (replaces individual collection-level getters)    */
/* ================================================================== */

/**
 * @brief Summary information about a loaded collection.
 *
 * Returned by pulseg_get_collection_info().
 */
typedef struct pulseg_collection_info
{
    int num_subsequences;    /**< number of subsequences              */
    int num_segments;        /**< total unique segments               */
    int max_adc_samples;     /**< max sample count across all ADCs    */
    int total_readouts;      /**< total ADC readout events            */
    float total_duration_us; /**< total sequence duration (us)        */
} pulseg_collection_info;

/* clang-format off */
#define PULSEG_COLLECTION_INFO_INIT {0, 0, 0, 0, 0.0f}
/* clang-format on */

/* ================================================================== */
/*  Subsequence info (replaces per-subsequence getters)               */
/* ================================================================== */

/**
 * @brief Metadata for a single subsequence.
 *
 * Returned by pulseg_get_subseq_info().
 */
typedef struct pulseg_subseq_info
{
    float tr_duration_us;      /**< TR duration (us)                    */
    int num_trs;               /**< number of TRs                       */
    int tr_size;               /**< blocks per TR                       */
    int num_prep_blocks;       /**< preparation blocks before first TR  */
    int num_cooldown_blocks;   /**< cooldown blocks after last TR       */
    int num_prep_trs;          /**< preparation TRs                     */
    int num_cooldown_trs;      /**< cooldown TRs                        */
    int degenerate_prep;       /**< 1 if prep == first TR               */
    int degenerate_cooldown;   /**< 1 if cooldown == last TR            */
    int num_unique_adcs;       /**< unique ADC definitions              */
    int num_unique_rf;         /**< unique RF definitions               */
    int pmc_enabled;           /**< 1 if PMC (prospective motion corr)  */
    int segment_offset;        /**< global segment index offset         */
    int num_prep_segments;     /**< segments in prep region             */
    int num_main_segments;     /**< segments in main TR region          */
    int num_cooldown_segments; /**< segments in cooldown region         */
    int num_adc_occurrences;   /**< ADC entries in label table          */
    int num_label_columns;     /**< label columns (vendor-dependent)    */
    int num_passes;            /**< number of inner-loop passes (>=1)   */
    int num_averages;          /**< number of averages (>=1)            */
    int num_canonical_trs;     /**< unique shot-ID combinations (>=1)   */
    int num_gain_cal_readouts; /**< calibration readouts for APS2 gain cal (pislquant) */
} pulseg_subseq_info;

/* clang-format off */
#define PULSEG_SUBSEQ_INFO_INIT \
    { \
    0.0f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0 \
    }
/* clang-format on */

/* ================================================================== */
/*  Segment info (replaces per-segment getters)                       */
/* ================================================================== */

/**
 * @brief Metadata for a single segment.
 *
 * Returned by pulseg_get_segment_info().
 */
typedef struct pulseg_segment_info
{
    int duration_us;         /**< total segment duration (us)         */
    int num_blocks;          /**< unique blocks in the segment        */
    int start_block;         /**< start block index in the sequence   */
    int pure_delay;          /**< 1 if segment is a bare delay        */
    int has_trigger;         /**< 1 if physio trigger attached        */
    int trigger_type;        /**< trigger type (1=output/TTL, 2=input/ECG), 0 if none */
    int trigger_delay_us;    /**< trigger delay (us), -1 if none      */
    int trigger_duration_us; /**< trigger duration (us), -1 if none   */
    int is_nav;              /**< 1 if navigator segment              */
    int rf_adc_gap_us;       /**< RF->ADC gap (us), -1 if no pair     */
    int adc_adc_gap_us;      /**< min ADC->ADC gap (us), -1 if < 2    */
} pulseg_segment_info;

/* clang-format off */
#define PULSEG_SEGMENT_INFO_INIT {0, 0, 0, 0, 0, 0, -1, -1, 0, -1, -1}
/* clang-format on */

/** Trigger type constants (public, matching internal definitions). */
#define PULSEG_TRIGGER_TYPE_OUTPUT 1 /**< TTL / digital output */
#define PULSEG_TRIGGER_TYPE_INPUT 2  /**< ECG / cardiac gating */

/* ================================================================== */
/*  Segment layout (subsequence-local segment, resolved to a scan     */
/*  instance -- replaces ad hoc consumer-side exec_stream walking)    */
/* ================================================================== */

/**
 * @brief Position/layout of one subsequence-local unique segment,
 * resolved to a concrete .seq block range.
 *
 * Returned by pulseg_get_subseq_segment_layout(). @c global_index is the
 * deduplicated cross-subsequence segment id (same space as every seg_idx
 * elsewhere in this API, e.g. pulseg_get_segment_info()); pair with
 * pulseg_get_subseq_segment_block_indices() for the resolved per-position
 * .seq block indices.
 */
typedef struct pulseg_segment_layout
{
    int global_index;             /**< deduplicated global segment id      */
    int num_blocks;               /**< blocks in the segment               */
    int start_block;              /**< segment definition's own start block
                                        (used when no scan-table instance
                                        is available)                      */
    int max_energy_start_block;   /**< start block of the max-energy scan
                                        instance, -1 if none                */
    int from_max_energy_instance; /**< 1 if pulseg_get_subseq_segment_-
                                        block_indices() resolved via the
                                        max-energy scan instance, 0 if it
                                        fell back to @c start_block         */
} pulseg_segment_layout;

/* clang-format off */
#define PULSEG_SEGMENT_LAYOUT_INIT {-1, 0, 0, -1, 0}
/* clang-format on */

/* ================================================================== */
/*  Block info (replaces per-block has/get accessor pairs)            */
/* ================================================================== */

/**
 * @brief Metadata for a single block within a segment.
 *
 * Returned by pulseg_get_block_info().
 * Waveform data is NOT included — use the dedicated waveform getters
 * (e.g.\ pulseg_get_grad_amplitude) keyed by the metadata here.
 */
typedef struct pulseg_block_info
{
    int duration_us;   /**< block duration (us)               */
    int start_time_us; /**< start time within segment (us)    */

    /* Gradient (per axis: [0]=X, [1]=Y, [2]=Z) */
    int has_grad[3];          /**< 1 if gradient present             */
    int grad_is_trapezoid[3]; /**< 1 if trapezoid (not arbitrary)    */
    int grad_delay_us[3];     /**< gradient delay (us), -1 if absent */
    int grad_num_shots[3];    /**< shot count, -1 if absent          */
    int grad_num_samples[3];  /**< sample count, -1 if absent        */

    /* RF */
    int has_rf;            /**< 1 if RF event present             */
    int rf_delay_us;       /**< RF delay (us), -1 if absent       */
    int rf_num_channels;   /**< Tx channel count, -1 if absent    */
    int rf_num_samples;    /**< samples per channel, -1 if absent */
    int rf_duration_us;    /**< RF duration (us) from last time-shape sample; -1 if absent */
    int rf_is_complex;     /**< 1 if phase shape exists           */
    int rf_uniform_raster; /**< 1 if time shape present           */

    /* ADC */
    int has_adc;      /**< 1 if ADC acquisition active       */
    int adc_delay_us; /**< ADC delay (us), -1 if absent      */
    int adc_def_id;   /**< global ADC library index, -1      */

    /* Digital output */
    int has_digitalout;         /**< 1 if digital output present       */
    int digitalout_delay_us;    /**< delay (us), -1 if absent          */
    int digitalout_duration_us; /**< duration (us), -1 if absent       */
    int digitalout_channel;     /**< trigger channel, -1 if absent     */

    /* Flags */
    int has_rotation;      /**< 1 if rotation event present       */
    int norot_flag;        /**< 1 if no-rotation override         */
    int nopos_flag;        /**< 1 if no-position override         */
    int has_freq_mod;      /**< 1 if frequency modulation present */
    int is_variable_delay; /**< 1 if a pure-delay block (no RF/grad/ADC): its
                            *   duration is runtime-adjustable via setperiod, so
                            *   two segments differing only in such a block's
                            *   duration share one segment definition. */
} pulseg_block_info;

/* clang-format off */
#define PULSEG_BLOCK_INFO_INIT \
    { \
    0, 0, {0, 0, 0}, {0, 0, 0}, {-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}, 0, -1, -1, -1, \
    -1, 0, 0, 0, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0 \
    }
/* clang-format on */

/* ================================================================== */
/*  ADC definition (replaces per-ADC getters)                         */
/* ================================================================== */

/**
 * @brief Information about a unique ADC definition.
 *
 * Returned by pulseg_get_adc_def().
 */
typedef struct pulseg_adc_def
{
    int dwell_ns;    /**< dwell time (ns)                     */
    int num_samples; /**< sample count                        */
} pulseg_adc_def;

/* clang-format off */
#define PULSEG_ADC_DEF_INIT {0, 0}
/* clang-format on */

/* ================================================================== */
/*  RF shim definition (parallel-transmit channel weights)            */
/* ================================================================== */

#define PULSEG_MAX_RF_SHIM_CHANNELS 64

/**
 * @brief Per-channel amplitude and phase weights for parallel transmit.
 *
 * Returned by pulseg_get_rf_shim_def().  The rf_shim_id field in
 * pulseg_block_instance is LOCAL to its subsequence (same convention as
 * rf_id, gx_id, etc.) — index 0 is the first shim of that subsequence.
 */
typedef struct pulseg_rf_shim_def
{
    int num_channels;                              /**< Tx channel count       */
    float magnitudes[PULSEG_MAX_RF_SHIM_CHANNELS]; /**< per-ch magnitude [0,1] */
    float phases[PULSEG_MAX_RF_SHIM_CHANNELS];     /**< per-ch phase (rad)     */
} pulseg_rf_shim_def;

/* clang-format off */
#define PULSEG_RF_SHIM_DEF_INIT {0, {0}, { 0 }}
/* clang-format on */

/* ================================================================== */
/*  RF event (per-occurrence identity, for pTx SAR accumulation)       */
/* ================================================================== */

/**
 * @brief Identity of one RF occurrence in the canonical TR, index-aligned
 * with pulseg_get_rf_array().
 *
 * Returned by pulseg_get_rf_event_array().
 */
typedef struct pulseg_rf_event
{
    int rf_def_id;      /**< local RF definition index within subsequence */
    float amplitude_hz; /**< per-event |amplitude| from rf_table (Hz)     */
    int rf_shim_id;     /**< local shim index, -1 if none                 */
    int num_channels;   /**< channels in the RF definition waveform (>=1) */
} pulseg_rf_event;

/* ================================================================== */
/*  K-space trajectory types                                          */
/* ================================================================== */

/** @brief Single k-space shot (one axis, ADC-sampled, k-zero centred). */
typedef struct pulseg_kshot
{
    int num_samples;
    float *k; /**< k-space values [num_samples], Hz·s/m */
} pulseg_kshot;

/** @brief Library of unique per-axis k-space shots. */
typedef struct pulseg_kshot_library
{
    int num_shots;
    pulseg_kshot *shots;
} pulseg_kshot_library;

/** @brief Per-ADC-event trajectory table entry. */
typedef struct pulseg_traj_table_entry
{
    int kx_shot_id;     /**< kshot index for X axis (-1 = trivial) */
    int ky_shot_id;     /**< kshot index for Y axis (-1 = trivial) */
    int kz_shot_id;     /**< kshot index for Z axis (-1 = trivial) */
    float gx_amplitude; /**< gradient amplitude for X (Hz/m)      */
    float gy_amplitude; /**< gradient amplitude for Y (Hz/m)      */
    float gz_amplitude; /**< gradient amplitude for Z (Hz/m)      */
    int rotation_id;    /**< index into rotation_matrices          */
    int slc, seg, rep, avg, set, eco, phs, lin, par, acq;
    unsigned long flags;    /**< ISMRMRD-compatible flag bitmask       */
    int center_sample;      /**< k-zero sample index within readout   */
    float sample_time_us;   /**< ADC dwell time in microseconds       */
    int encoding_space_ref; /**< encoding space index                  */
    int off;                /**< Pulseq LABELSET OFF flag (1=discard) */
} pulseg_traj_table_entry;

/** @brief Per-subsequence encoding-space descriptor.
 *
 * fov/matrix/nav_fov/nav_matrix dropped -- geometry is sourced
 * from the DEFINITIONS section (id 0) by subseq_idx, not duplicated here.
 * geometry_tag distinguishes the primary encoding space from a navigator
 * one sharing the same subsequence (DEFINITIONS' NavFOV/NavMatrix kv apply
 * when geometry_tag == 1). */
typedef struct pulseg_encoding_space
{
    int subseq_idx;                   /**< owning subsequence index              */
    int nav_subseq_offset;            /**< navigator subseq offset, 0 if none    */
    int geometry_tag;                 /**< 0 = primary, 1 = navigator            */
    pulseg_label_limits label_limits; /**< per-encoding-space label limits */
} pulseg_encoding_space;

/** @brief Complete trajectory description for a collection. */
typedef struct pulseg_trajectory
{
    pulseg_kshot_library kshots;
    int num_encoding_spaces;
    pulseg_encoding_space *encoding_spaces;
    int num_adc_events;
    pulseg_traj_table_entry *table;
    /* rotation-matrix library folded into TRAJECTORY itself
     * (copied from the owning descriptor's rotation_matrices[]) so the
     * recon reader is self-contained and never reads the PSD-internal
     * ROTATIONS section. table[].rotation_id indexes this array;
     * pulseg_merge_trajectory offsets it like the kshot ids. */
    int num_rotations;
    float (*rotation_matrices)[9];
} pulseg_trajectory;

/* ================================================================== */
/*  Sequence description (Section 5 — compact canonical-TR table)     */
/* ================================================================== */

/** ADC role codes for sequence-description rows. */
#define PULSEG_ADC_ROLE_NON_ACQUIRED 0 /**< navigator / PMC — skip in recon  */
#define PULSEG_ADC_ROLE_SINGLE 1       /**< single ADC in its echo group     */
#define PULSEG_ADC_ROLE_ECHO_CENTER 2  /**< nearest-to-k-zero ADC in group   */
#define PULSEG_ADC_ROLE_NON_CENTER 3   /**< other ADC in a multi-ADC group   */

/** Event type codes for Section 5 rows. */
#define PULSEG_SEQ_EVENT_OTHER 0 /**< pure wait / no RF or ADC event */
#define PULSEG_SEQ_EVENT_RF 1    /**< RF pulse                       */
#define PULSEG_SEQ_EVENT_ADC 2   /**< readout event                  */

/* Keep legacy alias so existing call-sites outside this repo still compile. */
#define PULSEG_SEQ_EVENT_WAIT PULSEG_SEQ_EVENT_OTHER

/* Number of float payload fields per row (same for all event types; unused
 * fields are zero-padded).                                                */
#define PULSEG_SEQ_EVENT_PARAMS 7

/**
 * @brief One row in the compact canonical-TR event table (Section 5).
 *
 * The table has one row per block in the average-expanded canonical TR
 * (the full pass: prep + main + cooldown).  One row per block regardless
 * of event type — blocks with both RF and ADC are represented by their RF
 * row (RF takes priority).
 *
 * Timestamp semantics (pass-relative, us):
 *   RF    — RF isocenter time
 *   ADC   — k-space-zero sample time
 *   OTHER — block start time
 *
 * Payload interpretation (params[] stored as float; integer IDs are cast):
 *   RF:    [0] rf_def_id (int)        index into per-subseq RF definition table
 *          [1] rf_use (int)           PULSEG_RF_USE_* code
 *          [2] act_amplitude_hz (f)   actual |gamma*B1| peak (Hz)
 *          [3] phase_offset_rad (f)   per-instance phase incl. ppm (rad)
 *          [4] freq_offset_hz (f)     per-instance freq incl. ppm (Hz)
 *          [5] rf_shim_id (int)       shim definition index, -1 if none
 *          [6] ss_grad_amp_hz_per_m (f) amplitude of slice-selection gradient
 *                                       (Hz/m); 0.0 if absent, non-trap, or
 *                                       more than one gradient axis is active
 *
 *   ADC:   [0] adc_role (int)         PULSEG_ADC_ROLE_*
 *          [1] phase_offset_rad (f)   per-instance ADC phase incl. ppm (rad)
 *          [2] echo (int/bool)         1 when this ADC position reaches
 *                                      k-space zero for at least one instance
 *          [3..6] = 0
 *
 *   OTHER: [0..6] = 0
 *
 * The C++ reader (trajectory_cache_reader) dedups the
 * (rf_def_id, rf_shim_id, ss_grad_amp_hz_per_m) triplets over all rows to
 * form a unique-tuple library. For each unique tuple it computes
 * slice_thickness_mm = bandwidth_hz / |ss_grad_amp_hz_per_m| * 1e3 and sets
 * slice_selective = (slice_thickness_mm < 10.0). Both are packed into the
 * per-subsequence RF waveform header streamed as an ISMRMRD Waveform.
 */
typedef struct pulseg_seq_event
{
    int type;           /**< PULSEG_SEQ_EVENT_{OTHER,RF,ADC}    */
    float timestamp_us; /**< anchor time (us, pass-relative)        */
    float params[PULSEG_SEQ_EVENT_PARAMS];
} pulseg_seq_event;

/**
 * @brief Scan-global sequence parameters, aggregated across all subsequences.
 */
typedef struct pulseg_sequence_parameters
{
    float min_te_us;
    float min_tr_us;
    float max_tr_us;
    float max_flip_angle_deg;
    float total_scan_time_us; /**< estimated total scan duration (us) */
    int num_subseqs;
    int reserved[3];
} pulseg_sequence_parameters;

/**
 * @brief Per-subsequence sequence description — compact canonical-TR table.
 *
 * @c rows is heap-allocated; freed by pulseg_sequence_description_free().
 * @c num_rows == number of blocks in the full pass (prep + main + cooldown).
 */
typedef struct pulseg_sequence_description
{
    int subseq_idx;
    float tr_duration_us; /**< full pass duration (us) */
    int num_rows;
    pulseg_seq_event *rows;
} pulseg_sequence_description;

#endif /* PULSEG_TYPES_H */
