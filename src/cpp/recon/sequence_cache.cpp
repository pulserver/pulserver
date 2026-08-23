/**
 * @file sequence_cache.cpp
 * @brief Format-agnostic consumers of a populated SequenceCache.
 *
 * Everything here reads the in-memory SequenceCache and never a file:
 * trajectory composition (pre_compute_trajectories / materialize_readout),
 * ISMRMRD header and acquisition enrichment, the off-isocentre phase
 * demodulation, and the physiological/SEQDESC waveform helpers. The cache
 * itself is populated from the Pulseq seqfile chain by
 * read_sequence_files() (sequence_file_reader.cpp).
 */

#include "sequence_cache.h"

#include <fstream>
#include <stdexcept>
#include <cmath>
#include <complex>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <sstream>
#include <algorithm>

#include "ismrmrd/ismrmrd.h"
#include "ismrmrd/version.h"
#include "ismrmrd/waveform.h"

namespace mrdserver
{

    namespace
    {
        /* The three logical-frame k rows of one table entry, composed from the
         * kshot library.
         *
         * A kshot is the block's NORMALISED base -- the shape, with the
         * amplitude divided out -- exactly as the seqfile stores it. So a
         * readout is its base plus two triples of scalars:
         *
         *     k_axis(t) = k_origin_axis + amplitude_axis * base_axis(t)
         *
         * which is what lets interleaves differing only in amplitude (cones,
         * floret) share one row, the way interleaves differing only in
         * rotation already did.
         *
         * Per-ADC rotation (entry.rotation_id) is intentionally NOT applied:
         * the cache stores k in the LOGICAL gradient frame, which is the frame
         * the design side works in and the one an FOV shift is invariant under.
         * enrich_ismrmrd_acquisition composes the rotation when it writes an
         * acquisition, which is the first point anything needs where a sample
         * physically sits.
         */
        void compose_entry_rows(
            const SequenceCache& cache,
            const TrajTableEntry& entry,
            int nsamples,
            float* px,
            float* py,
            float* pz)
        {
            const int shots[3] = {entry.kx_shot_id, entry.ky_shot_id, entry.kz_shot_id};
            const float amps[3] = {
                entry.gx_amplitude,
                entry.gy_amplitude,
                entry.gz_amplitude};
            float* dsts[3] = {px, py, pz};

            for (int axis = 0; axis < 3; ++axis)
            {
                float* dst = dsts[axis];
                const float origin = entry.k_origin[axis];
                for (int i = 0; i < nsamples; ++i)
                    dst[i] = origin;
                const int shot_id = shots[axis];
                if (shot_id < 0 || shot_id >= static_cast<int>(cache.kshots.size()))
                    continue;
                const auto& sk = cache.kshots[shot_id].k;
                for (int i = 0; i < std::min(nsamples, static_cast<int>(sk.size())); ++i)
                    dst[i] += amps[axis] * sk[i];
            }
        }

        /* Interleave the active axes of one readout into `dst`. */
        void pack_readout(
            const float* kx,
            const float* ky,
            const float* kz,
            int nsamples,
            int ndim,
            bool has_x,
            bool has_y,
            bool has_z,
            float* dst)
        {
            for (int s = 0; s < nsamples; ++s)
            {
                int d = 0;
                if (has_x)
                    dst[s * ndim + d++] = kx[s];
                if (has_y)
                    dst[s * ndim + d++] = ky[s];
                if (has_z)
                    dst[s * ndim + d++] = kz[s];
            }
        }
        /**
         * @brief Whether an encoding space's readouts carry k-space shots.
         *
         * Reads the first table entry of the space rather than composing anything:
         * every readout of one encoding space has the same trajectory structure.
         *
         * @param cache Populated SequenceCache.
         * @param es    Encoding space index.
         * @return true when the space samples a trajectory, false when Cartesian.
         */
        bool space_has_trajectory(const SequenceCache& cache, int es)
        {
            for (size_t t = 0; t < cache.table.size(); ++t)
            {
                const auto& entry = cache.table[t];
                if (entry.encoding_space_ref != es)
                    continue;
                const int shots[3] = {entry.kx_shot_id, entry.ky_shot_id, entry.kz_shot_id};
                for (int axis = 0; axis < 3; ++axis)
                {
                    if (shots[axis] >= 0 && shots[axis] < static_cast<int>(cache.kshots.size()) &&
                        !cache.kshots[shots[axis]].k.empty())
                        return true;
                }
                return false;
            }
            return false;
        }

    } // namespace

    std::vector<PrecomputedTrajectory> pre_compute_trajectories(
        const SequenceCache& cache,
        size_t budget_floats)
    {
        const int num_es = static_cast<int>(cache.encoding_spaces.size());
        std::vector<PrecomputedTrajectory> result(static_cast<size_t>(num_es));

        for (int es = 0; es < num_es; ++es)
        {
            // Collect ADC indices for this encoding space
            std::vector<int> adc_indices;
            for (int t = 0; t < static_cast<int>(cache.table.size()); ++t)
            {
                if (cache.table[t].encoding_space_ref == es)
                    adc_indices.push_back(t);
            }
            if (adc_indices.empty())
                continue;

            // Determine num_samples from the first ADC's kshot
            const auto& first = cache.table[adc_indices[0]];
            int nsamples = 0;
            if (first.kx_shot_id >= 0 && first.kx_shot_id < static_cast<int>(cache.kshots.size()))
                nsamples = static_cast<int>(cache.kshots[first.kx_shot_id].k.size());
            else if (
                first.ky_shot_id >= 0 && first.ky_shot_id < static_cast<int>(cache.kshots.size()))
                nsamples = static_cast<int>(cache.kshots[first.ky_shot_id].k.size());
            else if (
                first.kz_shot_id >= 0 && first.kz_shot_id < static_cast<int>(cache.kshots.size()))
                nsamples = static_cast<int>(cache.kshots[first.kz_shot_id].k.size());

            // The reader stores a real base shot for every active, rotated
            // axis instead of collapsing it to cartesian (-1), so
            // nsamples == 0 here means there is genuinely no trajectory
            // data for this encoding space.
            if (nsamples == 0)
                continue;

            const int num_ro = static_cast<int>(adc_indices.size());
            std::vector<float> kx_all(static_cast<size_t>(nsamples) * num_ro, 0.0f);
            std::vector<float> ky_all(static_cast<size_t>(nsamples) * num_ro, 0.0f);
            std::vector<float> kz_all(static_cast<size_t>(nsamples) * num_ro, 0.0f);

            for (int r = 0; r < num_ro; ++r)
            {
                const auto& entry = cache.table[adc_indices[r]];
                float* px = &kx_all[static_cast<size_t>(r) * nsamples];
                float* py = &ky_all[static_cast<size_t>(r) * nsamples];
                float* pz = &kz_all[static_cast<size_t>(r) * nsamples];

                compose_entry_rows(cache, entry, nsamples, px, py, pz);
            }

            auto is_zero = [](const std::vector<float>& v)
            {
                for (float f : v)
                    if (f != 0.0f)
                        return false;
                return true;
            };
            bool has_x = !is_zero(kx_all);
            bool has_y = !is_zero(ky_all);
            bool has_z = !is_zero(kz_all);
            int ndim = (has_x ? 1 : 0) + (has_y ? 1 : 0) + (has_z ? 1 : 0);
            if (ndim == 0)
                continue; // Cartesian

            auto& pt = result[es];
            pt.ndim = ndim;
            pt.num_samples = nsamples;
            pt.num_readouts = num_ro;
            pt.axis_active[0] = has_x;
            pt.axis_active[1] = has_y;
            pt.axis_active[2] = has_z;

            /* Hold the whole space only when it fits.  Past the budget this
             * array IS the scan's k-space -- one distinct readout per
             * acquisition for an individually optimised trajectory -- so it
             * is left empty and rebuilt a readout at a time on demand.  The
             * axis-activity decision above still stands: it was taken over
             * every readout, which is what makes ndim stable between the two
             * modes. */
            const size_t total = static_cast<size_t>(ndim) * static_cast<size_t>(nsamples) *
                static_cast<size_t>(num_ro);
            pt.resident = (total <= budget_floats);
            if (!pt.resident)
                continue;

            pt.data.resize(total);

            // Pack active axes: interleaved [ax0_s0, ax1_s0, ..., ax0_s1, ...]
            for (int r = 0; r < num_ro; ++r)
            {
                pack_readout(
                    &kx_all[static_cast<size_t>(r) * nsamples],
                    &ky_all[static_cast<size_t>(r) * nsamples],
                    &kz_all[static_cast<size_t>(r) * nsamples],
                    nsamples,
                    ndim,
                    has_x,
                    has_y,
                    has_z,
                    &pt.data[static_cast<size_t>(r) * ndim * nsamples]);
            }
        }

        return result;
    }

    bool materialize_readout(
        const SequenceCache& cache,
        const PrecomputedTrajectory& traj,
        int es_index,
        int readout,
        float* out)
    {
        if (!out || traj.ndim <= 0 || traj.num_samples <= 0)
            return false;
        if (readout < 0 || readout >= traj.num_readouts)
            return false;

        const size_t stride =
            static_cast<size_t>(traj.ndim) * static_cast<size_t>(traj.num_samples);

        if (traj.resident)
        {
            if (traj.data.size() < (static_cast<size_t>(readout) + 1) * stride)
                return false;
            std::memcpy(
                out,
                &traj.data[static_cast<size_t>(readout) * stride],
                stride * sizeof(float));
            return true;
        }

        /* Not resident: find this encoding space's readout-th table entry and
         * rebuild it.  The scan is walked rather than indexed because the
         * table interleaves encoding spaces, and the resident path counted
         * them the same way. */
        int seen = 0;
        const TrajTableEntry* entry = nullptr;
        for (const auto& e : cache.table)
        {
            if (e.encoding_space_ref != es_index)
                continue;
            if (seen == readout)
            {
                entry = &e;
                break;
            }
            ++seen;
        }
        if (!entry)
            return false;

        std::vector<float> kx(static_cast<size_t>(traj.num_samples));
        std::vector<float> ky(static_cast<size_t>(traj.num_samples));
        std::vector<float> kz(static_cast<size_t>(traj.num_samples));
        compose_entry_rows(cache, *entry, traj.num_samples, kx.data(), ky.data(), kz.data());

        /* The mask decided over every readout, not one inferred here. */
        pack_readout(
            kx.data(),
            ky.data(),
            kz.data(),
            traj.num_samples,
            traj.ndim,
            traj.axis_active[0],
            traj.axis_active[1],
            traj.axis_active[2],
            out);
        return true;
    }

    namespace
    {

        /* Map cfgradcoil identifier to the GE coef base name.
         * Reference values from EPIC pulserver.allcv.h:
         *   1=CRD, 2=Roemer, 101=HGC, 102=Vectra, 103=Permanent.
         * Add new entries as more gradient subsystems become relevant. */
        const char* cfgradcoil_to_coef_name(int id)
        {
            switch (id)
            {
            case 1:
                return "crd";
            case 2:
                return "roemer";
            case 101:
                return "hgc";
            case 102:
                return "vectra";
            case 103:
                return "permanent";
            default:
                return nullptr;
            }
        }

        std::string sequence_resource_base_dir()
        {
            const char* env = std::getenv("GADGETRON_RESOURCE_DIR");
            if (env && env[0] != '\0')
                return std::string(env);
            return std::string("/usr/g/bin");
        }

    } // anonymous namespace

    void set_user_parameter_string(
        ISMRMRD::IsmrmrdHeader& hdr,
        const std::string& name,
        const std::string& value)
    {
        if (!hdr.userParameters)
            hdr.userParameters = ISMRMRD::UserParameters();
        auto& strs = hdr.userParameters->userParameterString;
        for (auto& p : strs)
        {
            if (p.name == name)
            {
                p.value = value;
                return;
            }
        }
        ISMRMRD::UserParameterString p;
        p.name = name;
        p.value = value;
        strs.push_back(p);
    }

    void add_diffusion_parameters(ISMRMRD::IsmrmrdHeader& hdr, const SequenceCache& cache)
    {
        static const char* const KEYS[] =
            {"bTensorFixed", "bTensorRotatable", "bTensorCross", "bTensorAxis"};

        for (const char* key : KEYS)
        {
            auto it = cache.definitions.find(key);
            if (it == cache.definitions.end() || it->second.empty())
                continue;

            // Verbatim, joined the way the definition was written.  Nothing is
            // parsed and nothing is composed here on purpose: the b-tensor
            // comes in three parts precisely because the console's FOV
            // rotation is not in the .seq, and putting that rotation together
            // with them needs a convention -- which MRD already states, in the
            // acquisition's direction cosines.  A wrong b-vector is invisible
            // downstream, so the composition happens once, on the
            // reconstruction side, where it is tested.
            std::string joined;
            for (size_t i = 0; i < it->second.size(); ++i)
            {
                if (i)
                    joined += ' ';
                joined += it->second[i];
            }
            set_user_parameter_string(hdr, key, joined);
        }
    }

    void add_sequence_resource_paths(
        ISMRMRD::IsmrmrdHeader& hdr,
        int tensor_index,
        int grad_coil_id)
    {
        const std::string base = sequence_resource_base_dir();

        if (tensor_index > 0)
        {
            std::ostringstream oss;
            oss << base << "/tensor" << tensor_index << ".dat";
            set_user_parameter_string(hdr, "tensor_dat_path", oss.str());
        }

        if (const char* coef = cfgradcoil_to_coef_name(grad_coil_id))
        {
            std::ostringstream oss;
            oss << base << "/" << coef << ".coef";
            set_user_parameter_string(hdr, "grad_coef_path", oss.str());
        }
    }

    void enrich_ismrmrd_header(ISMRMRD::IsmrmrdHeader& hdr, const SequenceCache& cache)
    {
        add_diffusion_parameters(hdr, cache);

        // --- Sequence parameters from definitions ---
        {
            ISMRMRD::SequenceParameters seqp;
            auto get_floats = [&](const char* key) -> std::vector<float>
            {
                std::vector<float> out;
                auto it = cache.definitions.find(key);
                if (it != cache.definitions.end())
                {
                    for (const auto& sv : it->second)
                    {
                        try
                        {
                            out.push_back(std::stof(sv));
                        }
                        catch (...)
                        {
                        }
                    }
                }
                return out;
            };
            // Sequence parameters are NOT divided per encoding space in the ISMRMRD
            // header; for a multi-subsequence (multi-contrast) collection the merged
            // values are reduced to a single representative — min TR/TE/TI, max
            // flip angle. (Per-subsequence detail is carried separately by the
            // SEQDESC waveforms.) Each is emitted as a one-element vector.
            auto reduce_min = [](std::vector<float> vals) -> std::vector<float>
            {
                if (vals.empty())
                    return {};
                return {*std::min_element(vals.begin(), vals.end())};
            };
            auto reduce_max = [](std::vector<float> vals) -> std::vector<float>
            {
                if (vals.empty())
                    return {};
                return {*std::max_element(vals.begin(), vals.end())};
            };
            auto tr = reduce_min(get_floats("TR"));
            auto te = reduce_min(get_floats("TE"));
            auto ti = reduce_min(get_floats("TI"));
            auto fa = reduce_max(get_floats("FlipAngle"));
            if (!tr.empty())
                seqp.TR = tr;
            if (!te.empty())
                seqp.TE = te;
            if (!ti.empty())
                seqp.TI = ti;
            if (!fa.empty())
                seqp.flipAngle_deg = fa;
            // Only override if we actually have something to set
            if (seqp.TR || seqp.TE || seqp.TI || seqp.flipAngle_deg)
                hdr.sequenceParameters = seqp;
        }

        // --- Encoding limits, encodedSpace and reconSpace ---
        // cache.encoding_spaces is the authoritative flat list: normal and
        // navigator encoding spaces are already interleaved at the correct
        // indices (encoding_space_ref values in the table use these directly).
        // We map 1:1: hdr.encoding[i] ↔ cache.encoding_spaces[i].
        if (!cache.encoding_spaces.empty())
        {
            const int num_es = static_cast<int>(cache.encoding_spaces.size());
            if (static_cast<int>(hdr.encoding.size()) < num_es)
                hdr.encoding.resize(static_cast<size_t>(num_es));

            auto make_limit = [](const LabelLimit& ll)
            {
                ISMRMRD::Limit lim;
                lim.minimum = 0;
                lim.maximum = static_cast<uint16_t>(ll.max);
                lim.center = static_cast<uint16_t>(ll.max / 2);
                return lim;
            };

            // Stage 1.5c: FOV/Matrix are no longer duplicated on EncodingSpace --
            // read the verbatim pulseq [DEFINITIONS] strings for this ES's owning
            // subsequence (FOV/Matrix for the primary ES, NavFOV/NavMatrix for a
            // navigator ES, selected by geometry_tag). Pulseq FOV is in METERS
            // (e.g. "FOV 0.22 0.22 0.005"); ISMRMRD fieldOfView_mm wants mm.
            auto read_geometry =
                [&](int subseq_idx, int geometry_tag, float fov_mm[3], float matrix[3])
            {
                fov_mm[0] = fov_mm[1] = fov_mm[2] = 0.0f;
                matrix[0] = matrix[1] = matrix[2] = 0.0f;
                if (subseq_idx < 0 ||
                    subseq_idx >= static_cast<int>(cache.definitions_by_subseq.size()))
                    return;
                const auto& defs = cache.definitions_by_subseq[static_cast<size_t>(subseq_idx)];
                const char* fov_key = (geometry_tag == 1) ? "NavFOV" : "FOV";
                const char* matrix_key = (geometry_tag == 1) ? "NavMatrix" : "Matrix";
                auto fov_it = defs.find(fov_key);
                if (fov_it != defs.end() && fov_it->second.size() >= 3)
                {
                    for (int i = 0; i < 3; ++i)
                    {
                        try
                        {
                            fov_mm[i] = std::stof(fov_it->second[static_cast<size_t>(i)]) * 1000.0f;
                        }
                        catch (...)
                        {
                        }
                    }
                }
                auto mat_it = defs.find(matrix_key);
                if (mat_it != defs.end() && mat_it->second.size() >= 3)
                {
                    for (int i = 0; i < 3; ++i)
                    {
                        try
                        {
                            matrix[i] = std::stof(mat_it->second[static_cast<size_t>(i)]);
                        }
                        catch (...)
                        {
                        }
                    }
                }
            };

            // encodedSpace, reconSpace and encodingLimits: one entry per encoding space
            for (int es = 0; es < num_es; ++es)
            {
                const auto& ces = cache.encoding_spaces[es];

                float fov_mm[3], matrix[3];
                read_geometry(ces.subseq_idx, ces.geometry_tag, fov_mm, matrix);

                ISMRMRD::EncodingSpace space;
                space.matrixSize.x = static_cast<uint16_t>(matrix[0]);
                space.matrixSize.y = static_cast<uint16_t>(matrix[1]);
                space.matrixSize.z = static_cast<uint16_t>(matrix[2]);
                space.fieldOfView_mm.x = fov_mm[0];
                space.fieldOfView_mm.y = fov_mm[1];
                space.fieldOfView_mm.z = fov_mm[2];
                hdr.encoding[es].encodedSpace = space;
                hdr.encoding[es].reconSpace = space;

                // A readout with k-space shots attached samples something
                // other than a Cartesian grid, and that distinction is not
                // recoverable downstream: on a Cartesian space the phase-encode
                // limit undersamples its own grid, while on a non-Cartesian one
                // it counts the views themselves. A reconstruction sizing a
                // buffer has to know which it is holding.
                hdr.encoding[es].trajectory = space_has_trajectory(cache, es)
                    ? ISMRMRD::TrajectoryType::OTHER
                    : ISMRMRD::TrajectoryType::CARTESIAN;

                const auto& ll = ces.label_limits;
                auto& enc = hdr.encoding[es].encodingLimits;
                enc.kspace_encoding_step_1 = make_limit(ll.lin);
                enc.kspace_encoding_step_2 = make_limit(ll.par);
                enc.slice = make_limit(ll.slc);
                enc.average = make_limit(ll.avg);
                enc.contrast = make_limit(ll.eco);
                enc.phase = make_limit(ll.phs);
                enc.repetition = make_limit(ll.rep);
                enc.set = make_limit(ll.set);
                enc.segment = make_limit(ll.seg);
            }
        }
    }

    void enrich_ismrmrd_acquisition(
        ISMRMRD::Acquisition& acq,
        int acquisition_index,
        uint32_t measurement_uid,
        float table_position_z,
        const SequenceCache& cache,
        const std::vector<PrecomputedTrajectory>& trajectories,
        const std::vector<int>& readout_index_in_es,
        const uint32_t* physio_stamps)
    {
        // Scan-invariant fields not correctly populated by the GE converter
        acq.measurement_uid() = measurement_uid;
        acq.patient_table_position()[0] = 0.0f;
        acq.patient_table_position()[1] = 0.0f;
        acq.patient_table_position()[2] = table_position_z;

        // acquisition_time_stamp: ISMRMRD convention is ms since midnight;
        // the GE converter sets time(NULL) (seconds since epoch)
        time_t now = time(nullptr);
        struct tm* t = localtime(&now);
        acq.acquisition_time_stamp() =
            static_cast<uint32_t>((t->tm_hour * 3600 + t->tm_min * 60 + t->tm_sec) * 1000);

        // Physiological trigger timestamps: ms since midnight for each trigger type
        // (0=ECG, 1=PPG, 2=Respiratory).  Always zero-initialise so scans without
        // physio recording leave a well-defined default value.
        using namespace ISMRMRD; // bring enum values into scope
        for (int i = 0; i < ISMRMRD_PHYS_STAMPS; ++i)
            acq.physiology_time_stamp()[i] = (physio_stamps != nullptr) ? physio_stamps[i] : 0u;

        // Trajectory cache metadata and pre-computed trajectory
        if (cache.table.empty() || acquisition_index < 0 ||
            acquisition_index >= static_cast<int>(cache.table.size()))
            return;

        const auto& entry = cache.table[acquisition_index];

        auto& idx = acq.idx();
        idx.kspace_encode_step_1 = static_cast<uint16_t>(entry.lin);
        idx.kspace_encode_step_2 = static_cast<uint16_t>(entry.par);
        idx.slice = static_cast<uint16_t>(entry.slc);
        idx.average = static_cast<uint16_t>(entry.avg);
        idx.contrast = static_cast<uint16_t>(entry.eco);
        idx.phase = static_cast<uint16_t>(entry.phs);
        idx.repetition = static_cast<uint16_t>(entry.rep);
        idx.set = static_cast<uint16_t>(entry.set);
        idx.segment = static_cast<uint16_t>(entry.seg);

        const_cast<ISMRMRD::AcquisitionHeader&>(acq.getHead()).flags = entry.flags;

        /*
         * A negative echo index means the cache could not derive one -- the
         * readout's k does not move, so there is no sample that is the
         * crossing.  It is refused rather than cast: `uint16_t(-1)` is 65535,
         * a number a reconstruction would index k-space with.
         *
         * Reaching here means the sequence's canonical trajectory was flat
         * across the ADC window, which is a gap in the cache rather than in
         * the sequence.  The old code could not tell: it published
         * num_samples / 2 whether or not anything had been derived.
         */
        if (entry.center_sample < 0)
            throw std::runtime_error(
                "trajectory cache: acquisition " + std::to_string(acquisition_index) +
                " carries no derived echo index, so there is nothing to publish as "
                "centre sample. The sequence's canonical k-space is flat across this "
                "readout.");
        acq.center_sample() = static_cast<uint16_t>(entry.center_sample);
        acq.sample_time_us() = entry.sample_time_us;
        acq.encoding_space_ref() = static_cast<uint16_t>(entry.encoding_space_ref);

        const int es = entry.encoding_space_ref;

        if (es < 0 || es >= static_cast<int>(trajectories.size()))
            return;

        const auto& pt = trajectories[es];
        const int ro_idx = (acquisition_index < static_cast<int>(readout_index_in_es.size()))
            ? readout_index_in_es[acquisition_index]
            : -1;
        if (pt.ndim > 0 && ro_idx >= 0 && ro_idx < pt.num_readouts)
        {
            std::vector<float> packed(
                static_cast<size_t>(pt.ndim) * static_cast<size_t>(pt.num_samples));
            if (!materialize_readout(cache, pt, es, ro_idx, packed.data()))
                return;

            /* Back to three axes.
             *
             * The packed columns are the axes ACTIVE ACROSS THE ENCODING SPACE
             * (pt.axis_active), so column d is not axis d: a space that drives
             * only y and z packs two columns holding y and z. Anything that
             * multiplies the trajectory by something per-axis -- the rotation
             * below, the shift in demodulate_fov_shift -- has to be told which
             * axis it is looking at, and the only place that is known is here. */
            std::vector<float> k(3 * static_cast<size_t>(pt.num_samples), 0.0f);
            {
                int col = 0;
                for (int axis = 0; axis < 3; ++axis)
                {
                    if (!pt.axis_active[axis])
                        continue;
                    for (int sample = 0; sample < pt.num_samples; ++sample)
                        k[static_cast<size_t>(axis) * pt.num_samples + sample] =
                            packed[static_cast<size_t>(sample) * pt.ndim + col];
                    ++col;
                }
            }

            /* The rotation the block plays.
             *
             * The cache holds k in the LOGICAL gradient frame with the rotation
             * left as an id, because that is the frame the design side works in
             * and the one the shift is invariant under. An acquisition is not
             * that: what a reconstruction needs on `traj` is where the sample
             * actually sits, so the rotation is composed here, once, before
             * anything reads the result. */
            const int rot_id = entry.rotation_id;
            if (rot_id > 0 && rot_id < static_cast<int>(cache.rotations.size()))
            {
                const std::array<float, 9>& R = cache.rotations[rot_id];
                std::vector<float> rotated(k.size(), 0.0f);
                for (int sample = 0; sample < pt.num_samples; ++sample)
                {
                    for (int out_axis = 0; out_axis < 3; ++out_axis)
                    {
                        float acc = 0.0f;
                        for (int in_axis = 0; in_axis < 3; ++in_axis)
                            acc += R[static_cast<size_t>(out_axis) * 3 + in_axis] *
                                k[static_cast<size_t>(in_axis) * pt.num_samples + sample];
                        rotated[static_cast<size_t>(out_axis) * pt.num_samples + sample] = acc;
                    }
                }
                k.swap(rotated);
            }

            /* Trailing all-zero axes go, so the acquisition states this
             * readout's real dimensionality rather than the encoding space's.
             * After the rotation, never before it: a rotation mixes axes, and
             * an axis pruned first is one the mixing can no longer reach. */
            int effective_ndim = 3;
            while (effective_ndim > 0)
            {
                bool axis_all_zero = true;
                for (int sample = 0; sample < pt.num_samples && axis_all_zero; ++sample)
                    if (k[static_cast<size_t>(effective_ndim - 1) * pt.num_samples + sample] !=
                        0.0f)
                        axis_all_zero = false;
                if (!axis_all_zero)
                    break;
                --effective_ndim;
            }

            if (effective_ndim > 0)
            {
                acq.resize(acq.number_of_samples(), acq.active_channels(), effective_ndim);
                float* dst = acq.getTrajPtr();
                for (int sample = 0; sample < pt.num_samples; ++sample)
                    for (int d = 0; d < effective_ndim; ++d)
                        dst[static_cast<size_t>(sample) * effective_ndim + d] =
                            k[static_cast<size_t>(d) * pt.num_samples + sample];
            }
        }
    }

    void demodulate_fov_shift(ISMRMRD::Acquisition& acq, const float shift_m[3])
    {
        if (shift_m == nullptr)
            return;
        if (shift_m[0] == 0.0f && shift_m[1] == 0.0f && shift_m[2] == 0.0f)
            return;

        const int ndim = static_cast<int>(acq.trajectory_dimensions());
        const int nsamples = static_cast<int>(acq.number_of_samples());
        const int nchannels = static_cast<int>(acq.active_channels());
        if (ndim <= 0 || nsamples <= 0 || nchannels <= 0)
            return;

        const float* k = acq.getTrajPtr();
        if (k == nullptr)
            return;

        // Trailing all-zero axes were trimmed when the trajectory was attached,
        // so a shift along an axis this readout does not traverse simply has no
        // k to multiply -- which is correct, not a dropped term: a constant
        // gradient's contribution reaches the recon through the encoding
        // counters, not through the trajectory.
        const int axes = std::min(ndim, 3);

        static constexpr double kTwoPi = 6.283185307179586476925286766559;

        for (int s = 0; s < nsamples; ++s)
        {
            double cycles = 0.0;
            for (int d = 0; d < axes; ++d)
                cycles += static_cast<double>(shift_m[d]) *
                    static_cast<double>(k[static_cast<size_t>(s) * ndim + d]);

            const double phi = kTwoPi * cycles;
            const std::complex<float> rotor(
                static_cast<float>(std::cos(phi)),
                static_cast<float>(std::sin(phi)));

            for (int c = 0; c < nchannels; ++c)
                acq.data(s, c) *= rotor;
        }
    }

    void add_waveform_information(
        ISMRMRD::IsmrmrdHeader& hdr,
        bool has_ecg,
        bool has_ppg,
        bool has_resp)
    {
        struct Desc
        {
            bool enabled;
            ISMRMRD::WaveformType type;
            const char* name;
        };
        const Desc descs[] = {
            {has_ecg, ISMRMRD::WaveformType::ECG, "ECG"},
            {has_ppg, ISMRMRD::WaveformType::PULSE, "PPG"},
            {has_resp, ISMRMRD::WaveformType::RESPIRATORY, "Respiratory"},
        };
        for (const auto& d : descs)
        {
            if (!d.enabled)
                continue;
            ISMRMRD::WaveformInformation info;
            info.waveformName = d.name;
            info.waveformType = d.type;
            hdr.waveformInformation.push_back(info);
        }
    }

    ISMRMRD::Waveform make_physio_waveform(
        uint16_t waveform_id,
        uint32_t measurement_uid,
        uint32_t scan_counter,
        uint32_t time_stamp_ms,
        float sample_time_us,
        const std::vector<const int16_t*>& channels,
        uint16_t num_samples)
    {
        const uint16_t num_channels = static_cast<uint16_t>(channels.size());
        ISMRMRD::Waveform wav(num_samples, num_channels);
        wav.head.version = ISMRMRD_VERSION_MAJOR;
        wav.head.measurement_uid = measurement_uid;
        wav.head.scan_counter = scan_counter;
        wav.head.time_stamp = time_stamp_ms;
        wav.head.number_of_samples = num_samples;
        wav.head.channels = num_channels;
        wav.head.sample_time_us = sample_time_us;
        wav.head.waveform_id = waveform_id;

        // Channel-major layout: all samples of ch0, then ch1, etc.
        // int16_t sign-extended to uint32_t
        uint32_t* dst = wav.begin_data();
        for (uint16_t ch = 0; ch < num_channels; ++ch)
        {
            const int16_t* src = channels[ch];
            for (uint16_t s = 0; s < num_samples; ++s)
            {
                dst[static_cast<size_t>(ch) * num_samples + s] =
                    static_cast<uint32_t>(static_cast<int32_t>(src[s]));
            }
        }
        return wav;
    }

    // ---------- Helper: build a float-payload waveform ----------
    // All values are stored as bit-casts of float32 into uint32 channels.
    // num_channels = 1 (all data serialised as a flat uint32 stream).
    static ISMRMRD::Waveform make_float_payload_waveform(
        uint16_t waveform_id,
        uint32_t measurement_uid,
        uint32_t scan_counter,
        const std::vector<uint32_t>& payload)
    {
        const auto n = static_cast<uint16_t>(payload.size() > 65535u ? 65535u : payload.size());
        ISMRMRD::Waveform wav(n, 1);
        wav.head.version = ISMRMRD_VERSION_MAJOR;
        wav.head.measurement_uid = measurement_uid;
        wav.head.scan_counter = scan_counter;
        wav.head.time_stamp = 0;
        wav.head.number_of_samples = n;
        wav.head.channels = 1;
        wav.head.sample_time_us = 1.0f;
        wav.head.waveform_id = waveform_id;
        std::memcpy(wav.begin_data(), payload.data(), n * sizeof(uint32_t));
        return wav;
    }

    static uint32_t f2u(float v)
    {
        uint32_t u;
        std::memcpy(&u, &v, sizeof(u));
        return u;
    }

    // ================================================================
    // Sequence-description waveform factory functions
    // ================================================================

    ISMRMRD::Waveform make_seqdesc_header_waveform(
        const SequenceCache& cache,
        uint32_t measurement_uid,
        uint32_t scan_counter)
    {
        const auto& sp = cache.seq_params;
        std::vector<uint32_t> p;
        p.reserve(8);
        p.push_back(static_cast<uint32_t>(sp.num_subseqs));
        p.push_back(f2u(sp.min_te_us));
        p.push_back(f2u(sp.min_tr_us));
        p.push_back(f2u(sp.max_tr_us));
        p.push_back(f2u(sp.max_flip_angle_deg));
        p.push_back(f2u(sp.total_scan_time_us));
        return make_float_payload_waveform(
            WAVEFORM_ID_SEQDESC_HEADER,
            measurement_uid,
            scan_counter,
            p);
    }

    ISMRMRD::Waveform make_seqdesc_events_waveform(
        const SequenceDescription& desc,
        uint32_t measurement_uid,
        uint32_t scan_counter)
    {
        // Header: subseq_idx, tr_duration_us, num_events
        // Per event: type (int as uint32), timestamp_us (float), params[7] (float x7)
        // = 9 words per event
        std::vector<uint32_t> p;
        p.reserve(3 + desc.events.size() * 9);
        p.push_back(static_cast<uint32_t>(desc.subseq_idx));
        p.push_back(f2u(desc.tr_duration_us));
        p.push_back(static_cast<uint32_t>(desc.events.size()));
        for (const auto& ev : desc.events)
        {
            p.push_back(static_cast<uint32_t>(ev.type));
            p.push_back(f2u(ev.timestamp_us));
            for (int i = 0; i < 7; ++i)
                p.push_back(f2u(ev.params[i]));
        }
        return make_float_payload_waveform(
            WAVEFORM_ID_SEQDESC_EVENTS,
            measurement_uid,
            scan_counter,
            p);
    }

    ISMRMRD::Waveform make_seqdesc_rf_shapes_waveform(
        const SequenceDescription& desc,
        uint32_t measurement_uid,
        uint32_t scan_counter)
    {
        // Stage 1.5d: emit the REAL (still-compressed) per-rf_def shapes from
        // desc.rf_defs -- see the header doc comment for the exact wire format.
        std::vector<uint32_t> p;
        p.push_back(static_cast<uint32_t>(desc.subseq_idx));
        p.push_back(static_cast<uint32_t>(desc.rf_defs.size()));

        auto push_shape_header = [&](const RfShapeSamples& shape)
        {
            p.push_back(static_cast<uint32_t>(shape.num_uncompressed));
            p.push_back(static_cast<uint32_t>(shape.samples.size()));
        };
        auto push_shape_samples = [&](const RfShapeSamples& shape)
        {
            for (float v : shape.samples)
                p.push_back(f2u(v));
        };

        for (const auto& def : desc.rf_defs)
        {
            p.push_back(static_cast<uint32_t>(def.rf_def_id));
            p.push_back(f2u(def.bandwidth_hz));
            p.push_back(static_cast<uint32_t>(def.num_bands));
            for (int b = 0; b < 8; ++b)
                p.push_back(f2u(def.band_freq_offsets_hz[b]));
            p.push_back(f2u(def.band_bandwidth_hz));
            p.push_back(f2u(def.total_b1sq_power));

            push_shape_header(def.mag);
            p.push_back(static_cast<uint32_t>(def.has_phase ? 1 : 0));
            if (def.has_phase)
                push_shape_header(def.phase);
            p.push_back(static_cast<uint32_t>(def.has_time ? 1 : 0));
            if (def.has_time)
                push_shape_header(def.time);

            push_shape_samples(def.mag);
            if (def.has_phase)
                push_shape_samples(def.phase);
            if (def.has_time)
                push_shape_samples(def.time);
        }
        return make_float_payload_waveform(
            WAVEFORM_ID_SEQDESC_RF_SHAPES,
            measurement_uid,
            scan_counter,
            p);
    }

    ISMRMRD::Waveform make_seqdesc_shims_waveform(
        const SequenceDescription& desc,
        uint32_t measurement_uid,
        uint32_t scan_counter)
    {
        // Shim definitions are no longer stored in Section 5 of the cache.
        // Emit a minimal waveform with just the subseq header and zero shim count.
        std::vector<uint32_t> p;
        p.push_back(static_cast<uint32_t>(desc.subseq_idx));
        p.push_back(0u); // num_shims = 0
        return make_float_payload_waveform(
            WAVEFORM_ID_SEQDESC_SHIMS,
            measurement_uid,
            scan_counter,
            p);
    }

} // namespace mrdserver
