/**
 * @file trajectory_cache_reader.h
 * @brief Standalone reader for pulseqlib binary cache trajectory data.
 *
 * No dependency on pulserverlib — reads the binary cache format directly.
 * Stage 1.5c/1.5d: the reader is now self-contained across three sections
 * only, and never walks the PSD-internal COMMON/ROTATIONS descriptor:
 *   - Section 0 (DEFINITIONS): per-subsequence generic [DEFINITIONS] kv
 *     (FOV/Matrix/NavFOV/NavMatrix/TR/TE/TI/FlipAngle/... as pulseq strings).
 *   - Section 6 (TRAJECTORY): kshot library, encoding spaces (geometry now
 *     sourced from DEFINITIONS by subseq_idx, not duplicated here), table,
 *     and a folded-in rotation-matrix library (table[].rotation_id indexes
 *     it directly; no separate ROTATIONS-section read).
 *   - Section 7 (SEQDESC): per-subsequence event lists + a per-subsequence
 *     RF-definition library (bandwidth/bands/b1sq + compressed mag/phase/
 *     time shapes), when present.
 *
 * Section 5 (FREQMOD / off-isocenter shift) is intentionally NOT parsed:
 * frequency modulation is applied PSD-side, so data arriving at the
 * recon are already centered. See pulserverlib-tests/SCHEMA.md.
 */

#ifndef TRAJECTORY_CACHE_READER_H
#define TRAJECTORY_CACHE_READER_H

#include <vector>
#include <array>
#include <string>
#include <map>
#include <cstdint>

#include "ismrmrd/ismrmrd.h"
#include "ismrmrd/waveform.h"
#include "ismrmrd/xml.h"

namespace mrdserver
{

    struct Kshot
    {
        std::vector<float> k; /**< k-space values [num_samples], Hz·s/m */
    };

    struct LabelLimit
    {
        int min, max;
    };

    struct EncodingSpace
    {
        /* Stage 1.5c: fov/matrix/nav_fov/nav_matrix dropped -- geometry is
         * sourced from SequenceCache::definitions_by_subseq[subseq_idx] by
         * key ("FOV"/"Matrix" when geometry_tag==0, "NavFOV"/"NavMatrix"
         * when geometry_tag==1), not duplicated here. */
        int subseq_idx;
        int nav_subseq_offset;
        int geometry_tag; /**< 0 = primary, 1 = navigator */
        struct
        {
            LabelLimit slc, phs, rep, avg, seg, set, eco, par, lin, acq;
        } label_limits;
    };

    struct TrajTableEntry
    {
        int kx_shot_id, ky_shot_id, kz_shot_id;
        float gx_amplitude, gy_amplitude, gz_amplitude;
        int rotation_id;
        int slc, seg, rep, avg, set, eco, phs, lin, par, acq;
        uint64_t flags;         /**< ISMRMRD-compatible flag bitmask    */
        int center_sample;      /**< k-zero sample index                */
        float sample_time_us;   /**< ADC dwell time (us)                */
        int encoding_space_ref; /**< encoding space index                */
        int32_t off = 0;        /**< Pulseq LABELSET OFF flag (1=discard) */
    };

    /* ------------------------------------------------------------------ */
    /*  Sequence-description structs (Section 5 — SEQUENCEDESCRIPTION)    */
    /* ------------------------------------------------------------------ */

    /** ADC role constants */
    enum AdcRole
    {
        ADC_ROLE_NON_ACQUIRED = 0,
        ADC_ROLE_SINGLE = 1,
        ADC_ROLE_ECHO_CENTER = 2,
        ADC_ROLE_NON_CENTER = 3,
    };

    /** Sequence event type */
    enum SeqEventType
    {
        SEQ_EVENT_WAIT = 0,
        SEQ_EVENT_RF = 1,
        SEQ_EVENT_ADC = 2,
    };

    /**
     * @brief Single TR event (WAIT / RF / ADC).
     * See pulseqlib_types.h pulseqlib_seq_event for param layout.
     *
     * RF params:
     *   params[0] = rf_def_id (int)
     *   params[1] = rf_use (int, PULSEQLIB_RF_USE_*)
     *   params[2] = act_amplitude_hz (float)
     *   params[3] = phase_offset_rad (float)
     *   params[4] = freq_offset_hz (float)
     *   params[5] = rf_shim_id (int, -1 if none)
     *   params[6] = ss_grad_amp_hz_per_m (float, 0 if no slice-select grad)
     *
     * ADC params:
     *   params[0] = adc_role (int, AdcRole)
     *   params[1] = phase_offset_rad (float)
     */
    struct SeqEvent
    {
        SeqEventType type;
        float timestamp_us;
        float params[7];

        /* Accessors for RF events */
        int rf_def_id() const { return (int)params[0]; }
        int rf_use() const { return (int)params[1]; }
        float rf_act_amplitude_hz() const { return params[2]; }
        float rf_phase_offset_rad() const { return params[3]; }
        float rf_freq_offset_hz() const { return params[4]; }
        int rf_shim_id() const { return (int)params[5]; }
        float rf_ss_grad_amp_hz_per_m() const { return params[6]; }

        /* Accessors for ADC events */
        AdcRole adc_role() const { return static_cast<AdcRole>((int)params[0]); }
        float adc_phase_offset_rad() const { return params[1]; }
    };

    /** @brief RF shape tuple: unique (rf_def_id, rf_shim_id, ss_grad_amp) triplet,
     * deduped over a subsequence's events. Carries the per-event slice-geometry
     * derivation (bandwidth_hz comes from the matching RfDef, Stage 1.5d). */
    struct RfShapeTuple
    {
        int tuple_id;
        int rf_def_id;
        int rf_shim_id;             /**< -1 if no shim applied */
        float ss_grad_amp_hz_per_m; /**< slice-selection gradient amplitude (Hz/m) */
        float slice_thickness_mm;   /**< 0 if ss_grad_amp == 0 or bandwidth unknown */
        int slice_selective;        /**< 1 if slice_thickness_mm < 10 mm, else 0 */
    };

    /** @brief One (still-compressed) RF waveform shape, copied verbatim from
     * SEQDESC. Gadgetron decompresses; the reader never does. */
    struct RfShapeSamples
    {
        int num_uncompressed = 0;
        std::vector<float> samples; /**< compressed (delta-RLE), as stored */
    };

    /** @brief Per-subsequence RF-definition library entry (Stage 1.5b/1.5d).
     * rf_def_id is the array index into this subsequence's library, matching
     * SeqEvent::rf_def_id() for RF rows. */
    struct RfDef
    {
        int rf_def_id = 0;
        float bandwidth_hz = 0.0f;
        int num_bands = 1;
        float band_freq_offsets_hz[8] = {0};
        float band_bandwidth_hz = 0.0f;
        float total_b1sq_power = 0.0f;
        RfShapeSamples mag;
        bool has_phase = false;
        RfShapeSamples phase;
        bool has_time = false;
        RfShapeSamples time;
    };

    /** @brief Per-subsequence sequence description (event list + RF libraries). */
    struct SequenceDescription
    {
        int subseq_idx = 0;
        float tr_duration_us = 0.0f;
        std::vector<RfDef> rf_defs;                 /**< per-rf_def_id library (Stage 1.5b) */
        std::vector<RfShapeTuple> rf_shape_tuples; /**< deduped (def,shim,ss_grad) triplets */
        std::vector<SeqEvent> events;              /**< TR events (RF/ADC/WAIT) */
    };

    /** @brief Scan-global sequence parameters. */
    struct SequenceParameters
    {
        float min_te_us = 0.0f;
        float min_tr_us = 0.0f;
        float max_tr_us = 0.0f;
        float max_flip_angle_deg = 0.0f;
        float total_scan_time_us = 0.0f;
        int num_subseqs = 0;
    };

    /* ------------------------------------------------------------------ */

    struct SequenceCache
    {
        std::vector<Kshot> kshots;
        std::vector<std::array<float, 9>> rotations;
        std::vector<EncodingSpace> encoding_spaces;
        std::vector<TrajTableEntry> table;
        // Concatenated merge of every subsequence's [DEFINITIONS] (duplicate keys
        // across subsequences accumulate, e.g. one TR value per subsequence).
        std::map<std::string, std::vector<std::string>> definitions;
        /* Per-subsequence [DEFINITIONS] kv, indexed by subseq_idx (Stage 1.5a/1.5d).
         * Source of per-encoding-space FOV/Matrix/NavFOV/NavMatrix geometry --
         * see EncodingSpace doc comment. */
        std::vector<std::map<std::string, std::vector<std::string>>> definitions_by_subseq;
        /* Section 6 — sequence description (populated when present) */
        std::vector<SequenceDescription> seq_descs;
        SequenceParameters seq_params;
        bool has_seq_desc = false;
    };

    /**
     * Pre-computed trajectory data for a single encoding space.
     * ndim == 0 means Cartesian (no trajectory attached to acquisitions).
     * Layout: [ndim × num_samples] interleaved, repeated for each readout.
     * i.e. data[readout * ndim * num_samples + sample * ndim + dim]
     */
    struct PrecomputedTrajectory
    {
        int ndim = 0;        /**< 0=Cartesian, 2 or 3 */
        int num_samples = 0; /**< ADC samples per readout */
        int num_readouts = 0;
        std::vector<float> data; /**< [num_readouts × num_samples × ndim] */
    };

    /**
     * @brief Read trajectory data from a pulseqlib binary cache file.
     *
     * @param cache_path  Path to the .bin cache file.
     * @return Populated SequenceCache; empty if file not found or
     *         trajectory section is missing.
     * @throws std::runtime_error on I/O or format errors.
     */
    SequenceCache read_sequence_cache(const std::string &cache_path);

    /**
     * @brief Pre-compute per-encoding-space trajectory arrays from a loaded cache.
     *
     * Applies kshot scaling (amplitude), rotation matrices, and axis pruning.
     * Returns one PrecomputedTrajectory per encoding space; ndim==0 entries are
     * Cartesian and need no trajectory attached to ISMRMRD acquisitions.
     *
     * @param cache  Populated SequenceCache (from read_sequence_cache).
     * @return Vector of PrecomputedTrajectory, indexed by encoding_space_ref.
     */
    std::vector<PrecomputedTrajectory> pre_compute_trajectories(const SequenceCache &cache);

    /**
     * @brief Enrich an ISMRMRD header with Pulseq sequence parameters and
     *        encoding geometry derived from a loaded trajectory cache.
     *
     * Overrides (not appends):
     *  - sequenceParameters: TR, TE, TI, flipAngle_deg from definitions
     *  - encoding[i].encodingLimits (i==0): kspace limits from label_limits
     *  - encoding[i].encodedSpace / reconSpace: FOV and matrix from each
     *    encoding space entry (1:1 mapping — navigator encoding spaces are
     *    already present as separate entries in cache.encoding_spaces at the
     *    indices determined at cache-write time).
     *
     * @param hdr    Header to modify in-place (already deserialised).
     * @param cache  Populated SequenceCache.
     */
    void enrich_ismrmrd_header(ISMRMRD::IsmrmrdHeader &hdr, const SequenceCache &cache);

    /**
     * @brief Add or update a UserParameterString entry in the ISMRMRD header.
     *
     * Creates hdr.userParameters if absent. If a UserParameterString with the
     * same `name` already exists, its `value` is overwritten; otherwise a new
     * entry is appended.
     *
     * @param hdr    Header to modify in-place.
     * @param name   Parameter name (e.g. "tensor_dat_path", "grad_coef_path").
     * @param value  Parameter string value.
     */
    void set_user_parameter_string(ISMRMRD::IsmrmrdHeader &hdr,
                                   const std::string &name,
                                   const std::string &value);

    /**
     * @brief Resolve and inject sequence-description path UserParameters.
     *
     * Adds:
     *  - "tensor_dat_path" = "<base>/tensor<tensor_index>.dat"  if tensor_index > 0
     *  - "grad_coef_path"  = "<base>/<grad_coil_name>.coef"     if mapping known
     *
     * Files are NOT opened or validated; the recon side handles I/O. The base
     * directory defaults to "/usr/g/bin" (the GE scanner research dir) and may
     * be overridden via the GADGETRON_RESOURCE_DIR environment variable.
     *
     * @param hdr            Header to modify in-place.
     * @param tensor_index   Diffusion tensor file index (rdb_hdr_user6); 0 = skip.
     * @param grad_coil_id   cfgradcoil value (rdb_hdr_user7); unknown ids are skipped.
     */
    void add_sequence_resource_paths(ISMRMRD::IsmrmrdHeader &hdr,
                                     int tensor_index,
                                     int grad_coil_id);

    /**
     * @brief Enrich an ISMRMRD Acquisition with trajectory cache metadata,
     *        pre-computed trajectory data, and scan-invariant header fields.
     *
     * All parameters that would require Orchestra SDK calls must be resolved
     * by the caller before invoking this function.
     *
     * Fills:
     *  - measurement_uid, patient_table_position (scan-invariant, from caller)
     *  - acquisition_time_stamp (ms since midnight, computed from system clock)
     *  - idx fields (lin, par, slc, avg, eco, phs, rep, set, seg)
     *  - flags, center_sample, sample_time_us, encoding_space_ref
     *  - trajectory data (memcpy from pre-computed array for the encoding space)
     *
     * @param acq                ISMRMRD Acquisition to modify in-place.
     * @param acquisition_index  Global readout counter (0-based).
     * @param measurement_uid    Stable per-scan identifier (e.g. exam<<16 ^ series).
     * @param table_position_z   Patient table S-axis isocenter (mm).
     * @param cache              Populated SequenceCache.
     * @param trajectories       Pre-computed trajectories (from pre_compute_trajectories).
     * @param readout_index_in_es Per-table-entry readout index within its encoding space.
     * @param physio_stamps      Optional array of 3 ms-since-midnight timestamps for the most
     *                           recent physiological trigger of each type
     *                           (index 0 = ECG, 1 = PPG, 2 = Respiratory).
     *                           Pass nullptr when physio is not enabled.
     */
    void enrich_ismrmrd_acquisition(
        ISMRMRD::Acquisition &acq,
        int acquisition_index,
        uint32_t measurement_uid,
        float table_position_z,
        const SequenceCache &cache,
        const std::vector<PrecomputedTrajectory> &trajectories,
        const std::vector<int> &readout_index_in_es,
        const uint32_t *physio_stamps = nullptr);

    /**
     * @brief Add WaveformInformation entries to an ISMRMRD header for the
     *        physiological signal types that will be sent.
     *
     * Call this in OnPrep after deserialization, before serializing and sending
     * the ISMRMRD header to Gadgetron.
     *
     * Waveform IDs follow the MRD standard:
     *   0 = ECG, 1 = Pulse Oximetry (PPG), 2 = Respiratory
     *
     * @param hdr         Header to modify in-place.
     * @param has_ecg     Include ECG waveform information.
     * @param has_ppg     Include pulse oximetry (PPG) waveform information.
     * @param has_resp    Include respiratory waveform information.
     */
    void add_waveform_information(ISMRMRD::IsmrmrdHeader &hdr,
                                  bool has_ecg, bool has_ppg, bool has_resp);

    /**
     * @brief Create a single ISMRMRD Waveform from one or more int16_t channel arrays.
     *
     * Each element of @p channels is a pointer to an array of @p num_samples int16_t
     * samples. Channels may differ in the waveform but must all have the same length.
     * int16_t samples are sign-extended to uint32_t in the output data array.
     *
     * ISMRMRD data layout: channel-major — all samples of channel 0, then channel 1, etc.
     *
     * @param waveform_id      MRD waveform type ID (0=ECG, 1=PPG, 2=Resp, 3/4=ext).
     * @param measurement_uid  Stable per-scan identifier.
     * @param scan_counter     Index of the next acquisition after this waveform.
     * @param time_stamp_ms    Start timestamp of the waveform (ms since midnight).
     * @param sample_time_us   Time between samples in microseconds.
     * @param channels         Vector of pointers to int16_t sample arrays.
     * @param num_samples      Number of samples per channel.
     * @return Populated ISMRMRD::Waveform ready to send.
     */
    ISMRMRD::Waveform make_physio_waveform(uint16_t waveform_id,
                                           uint32_t measurement_uid,
                                           uint32_t scan_counter,
                                           uint32_t time_stamp_ms,
                                           float sample_time_us,
                                           const std::vector<const int16_t *> &channels,
                                           uint16_t num_samples);

    /* ------------------------------------------------------------------ */
    /*  Section 6 waveform factory functions                              */
    /* ------------------------------------------------------------------ */

    /**
     * @brief Waveform IDs for the sequence-description custom waveforms.
     *
     * 999  — sequence-description header (one waveform per scan)
     * 1000 — per-TR event list (one per subsequence)
     * 1002 — RF shape tuples (one per subsequence)
     * 1005 — shim definitions (one per subsequence, omitted if empty)
     */
    constexpr uint16_t WAVEFORM_ID_SEQDESC_HEADER = 999;
    constexpr uint16_t WAVEFORM_ID_SEQDESC_EVENTS = 1000;
    constexpr uint16_t WAVEFORM_ID_SEQDESC_RF_SHAPES = 1002;
    constexpr uint16_t WAVEFORM_ID_SEQDESC_SHIMS = 1005;

    /**
     * @brief Create the sequence-description header waveform (ID 999).
     *
     * Contains: num_subseqs, min_te_us, min_tr_us, max_tr_us,
     *           max_flip_angle_deg, total_scan_time_us  (as uint32 bit-casts).
     */
    ISMRMRD::Waveform make_seqdesc_header_waveform(
        const SequenceCache &cache,
        uint32_t measurement_uid,
        uint32_t scan_counter);

    /**
     * @brief Create the event-list waveform (ID 1000) for one subsequence.
     *
     * Encodes all SeqEvent entries as float32 words packed into uint32 channels:
     *   channel 0: event type (0=WAIT, 1=RF, 2=ADC)
     *   channels 1-7: params[0..6]
     */
    ISMRMRD::Waveform make_seqdesc_events_waveform(
        const SequenceDescription &desc,
        uint32_t measurement_uid,
        uint32_t scan_counter);

    /**
     * @brief Create the RF-shape waveform (ID 1002) for one subsequence.
     *
     * Stage 1.5d: packs the REAL (still-compressed) mag/phase/time sample
     * arrays for every entry in desc.rf_defs (Gadgetron decompresses).
     * Header per rf_def (as uint32 words):
     *   rf_def_id, bandwidth_hz (float), num_bands,
     *   band_freq_offsets_hz[8] (float x8), band_bandwidth_hz (float),
     *   total_b1sq_power (float), mag_num_uncompressed, mag_num_samples,
     *   has_phase, [if has_phase: phase_num_uncompressed, phase_num_samples],
     *   has_time, [if has_time: time_num_uncompressed, time_num_samples]
     * Followed by: mag samples, then phase samples (if any), then time
     * samples (if any) -- all as float32 bit-cast into the uint32 payload.
     */
    ISMRMRD::Waveform make_seqdesc_rf_shapes_waveform(
        const SequenceDescription &desc,
        uint32_t measurement_uid,
        uint32_t scan_counter);

    /**
     * @brief Create the shim-definitions waveform (ID 1005) for one subsequence.
     *
     * Per shim: shim_id_local, N_ch, magnitudes[N_ch], phases[N_ch].
     * Returns an empty (zero-sample) waveform if no shims are defined.
     */
    ISMRMRD::Waveform make_seqdesc_shims_waveform(
        const SequenceDescription &desc,
        uint32_t measurement_uid,
        uint32_t scan_counter);

} // namespace mrdserver

#endif // TRAJECTORY_CACHE_READER_H
