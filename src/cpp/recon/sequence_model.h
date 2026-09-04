/**
 * @file sequence_cache.h
 * @brief The recon-side sequence model and its consumers.
 *
 * SequenceCache is the in-memory description a reconstruction works from:
 * the kshot library and per-acquisition table (logical-frame k, composed
 * with the rotation library downstream), the encoding spaces with their
 * label limits, per-subsequence [DEFINITIONS], and -- when the design side
 * declared TRSize -- the per-subsequence sequence description (SeqEvent
 * rows plus the RF-definition library, shapes still compressed).
 *
 * It is populated from the Pulseq seqfile chain by read_sequence_files()
 * (sequence_file_reader.h); everything declared below consumes the struct
 * and never touches a file.
 */

#ifndef SEQUENCE_MODEL_H
#define SEQUENCE_MODEL_H

#include <vector>
#include <array>
#include <string>
#include <map>
#include <cstdint>

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
        /**
         * k at the start of this readout's block, per axis, 1/m.
         *
         * A kshot holds the block's NORMALISED base, so what it costs is one
         * row per distinct *shape* rather than one per instance: interleaves
         * that differ only in gradient amplitude -- cones, floret -- share it,
         * as do those differing only in rotation. The two things that make a
         * readout its own is the amplitude triple and this origin, and both
         * are three floats.
         *
         * The origin is the moment accumulated since the last excitation, in
         * earlier blocks, and is not recoverable from the readout's own block.
         */
        float k_origin[3] = {0.0f, 0.0f, 0.0f};
        /**
         * 1 when this readout's off-isocentre FOV shift was left for the
         * consumer rather than baked into the sequence.
         *
         * A design that writes for a consumer bakes the shift as two scalars
         * where it can -- a Cartesian, unrotated readout -- and leaves it
         * alone where it cannot, storing the readout's base trajectory
         * instead. Only the second kind wants @ref demodulate_fov_shift; the
         * first already carries the phase, and applying it twice mirrors the
         * offset. A natively-shifted sequence sets this on nothing.
         */
        int32_t defers_fov_shift = 0;
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
     *
     * RF params:
     *   params[0] = rf_def_id (int)
     *   params[1] = rf_use (int, PULSEG_RF_USE_*)
     *   params[2] = act_amplitude_hz (float)
     *   params[3] = phase_offset_rad (float)
     *   params[4] = freq_offset_hz (float)
     *   params[5] = rf_shim_id (int, -1 if none)
     *   params[6] = ss_grad_amp_hz_per_m (float, 0 if no slice-select grad)
     *
     * ADC params:
     *   params[0] = adc_role (int, AdcRole)
     *   params[1] = phase_offset_rad (float)
     *   params[2] = echo (int/bool; any instance reaches k-space zero)
     */
    struct SeqEvent
    {
        SeqEventType type;
        float timestamp_us;
        float params[7];

        /* Accessors for RF events */
        int rf_def_id() const
        {
            return (int)params[0];
        }
        int rf_use() const
        {
            return (int)params[1];
        }
        float rf_act_amplitude_hz() const
        {
            return params[2];
        }
        float rf_phase_offset_rad() const
        {
            return params[3];
        }
        float rf_freq_offset_hz() const
        {
            return params[4];
        }
        int rf_shim_id() const
        {
            return (int)params[5];
        }
        float rf_ss_grad_amp_hz_per_m() const
        {
            return params[6];
        }

        /* Accessors for ADC events */
        AdcRole adc_role() const
        {
            return static_cast<AdcRole>((int)params[0]);
        }
        float adc_phase_offset_rad() const
        {
            return params[1];
        }
        bool adc_is_echo() const
        {
            return params[2] != 0.0f;
        }
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
        std::vector<RfDef> rf_defs;                /**< per-rf_def_id library (Stage 1.5b) */
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
     *
     * ndim reflects whether EACH axis is active for ANY readout in the
     * encoding space, not necessarily every readout: a readout whose
     * gradient rotation only mixes a subset of axes (e.g. a z-axis
     * rotation, which leaves the z gradient untouched) can legitimately
     * have an identically-zero row on an axis that is real data for other
     * readouts in the same space. This struct itself does not collapse
     * dimensionality per readout, only per encoding space -- but
     * enrich_ismrmrd_acquisition() does, trimming trailing all-zero axes
     * for the specific readout it writes, so the ISMRMRD acquisition (and
     * anything reading the file back) reflects that readout's real
     * dimensionality rather than carrying a spurious all-zero axis.
     * Consumers working from PrecomputedTrajectory directly (bypassing
     * enrich_ismrmrd_acquisition) must still do this pruning themselves.
     */
    struct PrecomputedTrajectory
    {
        int ndim = 0;        /**< 0=Cartesian, 2 or 3 */
        int num_samples = 0; /**< ADC samples per readout */
        int num_readouts = 0;

        /**
         * [num_readouts x num_samples x ndim], or EMPTY when `resident` is
         * false.  See below.
         */
        std::vector<float> data;

        /**
         * Whether `data` holds the whole encoding space.
         *
         * The same choice the interpreter makes for gradient waveforms, for
         * the same reason.  Most scans are a handful of base trajectories
         * replayed at different rotations, and holding them all costs
         * nothing -- so they are held, and a consumer reads `data` directly.
         *
         * An individually optimised trajectory is different: one distinct
         * readout per acquisition, at which point this array is the whole
         * k-space of the scan and can run to gigabytes.  Then `data` is left
         * empty and a consumer asks for one readout at a time with
         * @ref materialize_readout, which rebuilds it from the kshot library
         * at the cost of a copy.
         *
         * Nothing else about the struct changes, so a consumer that always
         * calls materialize_readout is correct in both modes.
         */
        bool resident = true;

        /**
         * Which of x, y, z this space packs, decided over EVERY readout.
         *
         * Not derivable from one readout: a readout whose rotation leaves an
         * axis silent would pack one axis fewer and shift every sample after
         * it.  Carried so the on-demand path packs identically to the
         * resident one.
         */
        bool axis_active[3] = {false, false, false};
    };

    /** Floats a single encoding space may hold before it stops being resident. */
    constexpr size_t DEFAULT_TRAJECTORY_BUDGET_FLOATS = 64u * 1024u * 1024u; /* 256 MB */

    /**
     * @brief Pre-compute per-encoding-space trajectory arrays from a loaded cache.
     *
     * Applies kshot scaling (amplitude), rotation matrices, and axis pruning.
     * Returns one PrecomputedTrajectory per encoding space; ndim==0 entries are
     * Cartesian and need no trajectory attached to ISMRMRD acquisitions.
     *
     * @param cache  Populated SequenceCache (from read_sequence_files).
     * @return Vector of PrecomputedTrajectory, indexed by encoding_space_ref.
     */
    std::vector<PrecomputedTrajectory> pre_compute_trajectories(
        const SequenceCache& cache,
        size_t budget_floats = DEFAULT_TRAJECTORY_BUDGET_FLOATS);

    /**
     * @brief Build one readout's trajectory, whether or not it is resident.
     *
     * Writes `ndim * num_samples` interleaved floats to @p out.  When the
     * encoding space is resident this copies out of `data`; when it is not,
     * it rebuilds the readout from the kshot library -- the same arithmetic
     * pre_compute_trajectories would have done, deferred.
     *
     * @param es_index  encoding-space index, matching the vector returned by
     *                  pre_compute_trajectories.
     * @return false when the space has no trajectory (Cartesian) or the
     *         readout is out of range, in which case @p out is untouched.
     */
    bool materialize_readout(
        const SequenceCache& cache,
        const PrecomputedTrajectory& traj,
        int es_index,
        int readout,
        float* out);

} // namespace mrdserver

#endif // SEQUENCE_MODEL_H
