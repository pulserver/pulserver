#ifndef SEQUENCE_CACHE_H
#define SEQUENCE_CACHE_H

/* The ISMRMRD-facing side of the sequence cache: header and acquisition
 * enrichment and the SEQDESC waveforms. The data model and the file
 * reader's types live in sequence_model.h, without ISMRMRD. */

#include "sequence_model.h"

#include "ismrmrd/ismrmrd.h"
#include "ismrmrd/waveform.h"
#include "ismrmrd/xml.h"

namespace mrdserver
{
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
    void enrich_ismrmrd_header(ISMRMRD::IsmrmrdHeader& hdr, const SequenceCache& cache);

    /**
     * @brief Add or update a UserParameterString entry in the ISMRMRD header.
     *
     * Creates hdr.userParameters if absent. If a UserParameterString with the
     * same `name` already exists, its `value` is overwritten; otherwise a new
     * entry is appended.
     *
     * @param hdr    Header to modify in-place.
     * @param name   Parameter name (e.g. "tensor_dat_path", "gradient_coefficients").
     * @param value  Parameter string value.
     */
    void set_user_parameter_string(
        ISMRMRD::IsmrmrdHeader& hdr,
        const std::string& name,
        const std::string& value);

    /**
     * @brief Copy the diffusion gradient table from the cache into the header.
     *
     * Adds `bTensorFixed`, `bTensorRotatable`, `bTensorCross` and
     * `bTensorAxis` as UserParameterStrings, taken **verbatim** from the
     * sequence's `[DEFINITIONS]` -- which is where
     * `Sequence.write_diffusion_definitions()` put them on the design side.
     * A part that was identically zero is not written and so is not copied;
     * `bTensorFixed` and `bTensorAxis` are always present when the sequence
     * carries a table at all, and their absence simply means it does not.
     *
     * The tensor is in three parts because the console's FOV rotation is not
     * in the `.seq`: `NOROT` exempts a block from it, which is what a
     * diffusion preparation does, while the imaging gradients of the same shot
     * do not. Composing them needs that rotation, and this deliberately does
     * **not** do it -- MRD already carries the orientation in the
     * acquisition's direction cosines, and one composition on the
     * reconstruction side, where it can be tested, beats a second convention
     * here. `pulserver.pypulseq.DiffusionTable.from_definitions` is the
     * reader.
     *
     * Called by @ref enrich_ismrmrd_header, so a caller that already uses that
     * needs no change.
     *
     * @param hdr    Header to modify in-place.
     * @param cache  Populated SequenceCache.
     */
    void add_diffusion_parameters(ISMRMRD::IsmrmrdHeader& hdr, const SequenceCache& cache);

    /**
     * @brief Inject the diffusion tensor table path as a UserParameter.
     *
     * Adds "tensor_dat_path" = "<base>/tensor<tensor_index>.dat" when
     * `tensor_index` is positive. The file is NOT opened or validated; the
     * recon side handles I/O. The base directory defaults to "/usr/g/bin"
     * (the GE scanner research dir) and may be overridden via the
     * GADGETRON_RESOURCE_DIR environment variable.
     *
     * @param hdr            Header to modify in-place.
     * @param tensor_index   Diffusion tensor file index (rdb_hdr_user2); 0 = skip.
     */
    void add_tensor_resource_path(ISMRMRD::IsmrmrdHeader& hdr, int tensor_index);

    /**
     * @brief Carry the gradient-coil spherical harmonics in the header.
     *
     * Writes the coefficients as UserParameterString "gradient_coefficients",
     * in the `GRADWARPTYPE` / `SCALE{X,Y,Z}{1..10}` / `DELTA` syntax of the
     * scanner's gw_coils.dat. The values travel rather than a path to them: a
     * reconstruction runs off the scanner's filesystem, where no path resolves.
     *
     * The key is vendor-neutral and the payload is not:
     * `pulserver.recon.GradientCoefficients.from_file` tells a GE table from a
     * Siemens one by its contents, so a site with a different console writes
     * its own syntax under the same name.
     *
     * Nothing is validated here. A gradwarp type this build cannot correct, or
     * a non-zero delta, is written as it was read and refused on the recon
     * side, where the correction that would be wrong is the one being asked
     * for.
     *
     * @param hdr             Header to modify in-place.
     * @param gradwarp_type   GRADWARPTYPE; 1 is spherical-harmonic.
     * @param scales          30 coefficients, X1..X10 then Y1..Y10 then Z1..Z10.
     * @param delta           DELTA term of the coil description.
     *
     * @throws std::invalid_argument if `scales` does not hold 30 entries.
     */
    void add_gradwarp_coefficients(
        ISMRMRD::IsmrmrdHeader& hdr,
        int gradwarp_type,
        const std::vector<float>& scales,
        float delta);

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
        ISMRMRD::Acquisition& acq,
        int acquisition_index,
        uint32_t measurement_uid,
        float table_position_z,
        const SequenceCache& cache,
        const std::vector<PrecomputedTrajectory>& trajectories,
        const std::vector<int>& readout_index_in_es,
        const uint32_t* physio_stamps = nullptr);

    /**
     * @brief Apply an off-isocentre FOV shift to an acquisition's data.
     *
     * A shift is a phase, `exp(+i 2*pi * dr . k)`, and this is where the
     * receive side of it happens. The spectrometer no longer synthesises a
     * modulation waveform for it: computing that required the physical-frame
     * gradient, which is what forced the interpreter to undo rotation
     * extensions and NOROT before it could produce anything.
     *
     * Call **after** enrich_ismrmrd_acquisition(), which is what puts the
     * trajectory on @p acq. The trajectory is in the LOGICAL frame, so
     * @p shift_m must be too -- and that is the point: `dr . k` is invariant
     * when both are rotated, so neither side needs to know the prescribed
     * orientation, any rotation extension, or PMC rotation.
     *
     * The sign matches what pulseq::apply_fov_shift bakes into a native-mode
     * `.seq`, so a sequence shifted here and one shifted at design time
     * reconstruct identically. Changing one without the other silently
     * mirrors the offset.
     *
     * Axes beyond the acquisition's trajectory dimensionality are ignored:
     * an axis that carries no k has no shift to contribute, and a Cartesian
     * phase-encode offset arrives through the encoding counters instead.
     *
     * No-op when @p shift_m is all zero, when @p acq carries no trajectory,
     * or when its trajectory length disagrees with its sample count.
     *
     * @param acq      Acquisition to modify in place, all channels.
     * @param shift_m  (dx, dy, dz) along the logical readout/phase/slice
     *                 axes, in metres.
     */
    void demodulate_fov_shift(ISMRMRD::Acquisition& acq, const float shift_m[3]);

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
    void add_waveform_information(
        ISMRMRD::IsmrmrdHeader& hdr,
        bool has_ecg,
        bool has_ppg,
        bool has_resp);

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
    ISMRMRD::Waveform make_physio_waveform(
        uint16_t waveform_id,
        uint32_t measurement_uid,
        uint32_t scan_counter,
        uint32_t time_stamp_ms,
        float sample_time_us,
        const std::vector<const int16_t*>& channels,
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
        const SequenceCache& cache,
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
        const SequenceDescription& desc,
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
        const SequenceDescription& desc,
        uint32_t measurement_uid,
        uint32_t scan_counter);

    /**
     * @brief Create the shim-definitions waveform (ID 1005) for one subsequence.
     *
     * Per shim: shim_id_local, N_ch, magnitudes[N_ch], phases[N_ch].
     * Returns an empty (zero-sample) waveform if no shims are defined.
     */
    ISMRMRD::Waveform make_seqdesc_shims_waveform(
        const SequenceDescription& desc,
        uint32_t measurement_uid,
        uint32_t scan_counter);

} // namespace mrdserver

#endif // SEQUENCE_CACHE_H
