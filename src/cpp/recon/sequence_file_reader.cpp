/**
 * @file sequence_file_reader.cpp
 * @brief SequenceCache populated from the Pulseq seqfile chain.
 *
 * One pulseq::SequenceFile per subsequence, walked along NextSequence.
 * Per subsequence, the trajectory table is built from the sequence itself:
 *
 *   - k in the LOGICAL gradient frame (rotation extensions are NOT resolved
 *     into the samples; the rotation-matrix library rides beside the table
 *     and is composed downstream, exactly the frame contract
 *     pre_compute_trajectories documents);
 *   - one kshot per active axis of a readout, content-deduplicated; an axis
 *     collapses to Cartesian (-1) only when its gradient is constant over
 *     the ADC window AND the block carries no rotation -- a rotated constant
 *     gradient is a real spoke, not a counter-placeable line;
 *   - labels evaluated by walking each block's LABELSET/LABELINC chains in
 *     stream order; NAV/REV/REF/IMA/NOISE map to their ISMRMRD flag bits,
 *     and `rep` carries the NEX average index, the table tiled once per
 *     average;
 *   - the navigator flag splits each subsequence into up to two encoding
 *     spaces (geometry_tag 1 = navigator), with per-space label limits.
 */

#include "sequence_file_reader.h"

#include "pulseq/kspace.hpp"
#include "pulseq/trajectory.hpp"
#include "pulseq/sequence.hpp"
#include "pulseq/sequence_file.hpp"
#include "pulseq/shape.hpp"

extern "C"
{
#include "pulseq_rf.h"
}

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

namespace mrdserver
{

    namespace
    {

        constexpr uint64_t FLAG_NAV = 1ULL << 22;   /* ACQ_IS_NAVIGATION_DATA (23)   */
        constexpr uint64_t FLAG_REV = 1ULL << 21;   /* ACQ_IS_REVERSE (22)           */
        constexpr uint64_t FLAG_REF = 1ULL << 19;   /* ACQ_IS_PARALLEL_CALIBRATION   */
        constexpr uint64_t FLAG_IMA = 1ULL << 20;   /* ..._AND_IMAGING (21)          */
        constexpr uint64_t FLAG_NOISE = 1ULL << 18; /* ACQ_IS_NOISE_MEASUREMENT (19) */
        constexpr uint64_t FLAG_LAST_IN_MEASUREMENT = 1ULL << 24;

        std::string format_number(double value)
        {
            char buffer[64];
            std::snprintf(buffer, sizeof(buffer), "%0.9g", value);
            return buffer;
        }

        std::map<std::string, std::vector<std::string>> definitions_as_strings(
            const std::map<std::string, pulseq::Definition>& defs)
        {
            std::map<std::string, std::vector<std::string>> out;
            for (const auto& entry : defs)
            {
                std::vector<std::string>& values = out[entry.first];
                if (entry.second.kind() == pulseq::Definition::Kind::Text)
                {
                    values.push_back(entry.second.text());
                    continue;
                }
                for (double number : entry.second.numbers())
                    values.push_back(format_number(number));
            }
            return out;
        }

        /** Scalar-first quaternion to a row-major 3x3, the frame consumers use. */
        std::array<float, 9> quaternion_to_matrix(const double* q)
        {
            const double w = q[0], x = q[1], y = q[2], z = q[3];
            return {
                static_cast<float>(1 - 2 * (y * y + z * z)),
                static_cast<float>(2 * (x * y - w * z)),
                static_cast<float>(2 * (x * z + w * y)),
                static_cast<float>(2 * (x * y + w * z)),
                static_cast<float>(1 - 2 * (x * x + z * z)),
                static_cast<float>(2 * (y * z - w * x)),
                static_cast<float>(2 * (x * z - w * y)),
                static_cast<float>(2 * (y * z + w * x)),
                static_cast<float>(1 - 2 * (x * x + y * y)),
            };
        }

        /** Per-ADC label state, walked off the extension chains in order. */
        struct LabelState
        {
            std::map<std::string, int> values;

            int get(const char* name) const
            {
                const auto it = values.find(name);
                return it == values.end() ? 0 : it->second;
            }
        };

        std::vector<LabelState> label_states_per_adc(pulseq::Sequence& seq)
        {
            const int labelset = seq.find_extension_type_id("LABELSET");
            const int labelinc = seq.find_extension_type_id("LABELINC");

            std::vector<LabelState> out;
            LabelState state;
            for (int index = 1; index <= seq.num_blocks(); ++index)
            {
                const pulseq::Block block = seq.get_block(index);
                int link = block.ext;
                while (link)
                {
                    const int32_t* row = seq.extensions_library().row(link);
                    const int type = row[0];
                    const int ref = row[1];
                    if (type == labelset && labelset != 0)
                    {
                        const int32_t* entry = seq.label_set_library().row(ref);
                        state.values[seq.label_name(entry[1])] = entry[0];
                    }
                    else if (type == labelinc && labelinc != 0)
                    {
                        const int32_t* entry = seq.label_inc_library().row(ref);
                        state.values[seq.label_name(entry[1])] += entry[0];
                    }
                    link = row[2];
                }
                if (block.adc != 0)
                    out.push_back(state);
            }
            return out;
        }

        /** Every counter a sequence writes, over the whole scan.
         *
         * A label absent here was never set, so it has no unit to bound and
         * gets no boundary flag -- as opposed to one written and left at zero,
         * which is a single unit that still begins and ends. */
        std::set<std::string> written_labels(const std::vector<LabelState>& states)
        {
            std::set<std::string> names;
            for (const LabelState& state : states)
                for (const auto& entry : state.values)
                    names.insert(entry.first);
            return names;
        }

        /** One counter and the ISMRMRD flag bits that bound it.
         *
         * The bit is the `ISMRMRD_ACQ_*` enum value less one, the same
         * convention as the flag constants above. */
        struct BoundaryCounter
        {
            const char* name;
            int TrajTableEntry::*field;
            uint64_t first_bit;
            uint64_t last_bit;
            bool is_frame; /**< names a whole image rather than a place in one */
        };

        constexpr BoundaryCounter BOUNDARY_COUNTERS[] = {
            {"SLC", &TrajTableEntry::slc, 1ULL << 6, 1ULL << 7, true},
            {"ECO", &TrajTableEntry::eco, 1ULL << 8, 1ULL << 9, true},
            {"PHS", &TrajTableEntry::phs, 1ULL << 10, 1ULL << 11, true},
            {"REP", &TrajTableEntry::rep, 1ULL << 12, 1ULL << 13, true},
            {"AVG", &TrajTableEntry::avg, 1ULL << 4, 1ULL << 5, true},
            {"SET", &TrajTableEntry::set, 1ULL << 14, 1ULL << 15, true},
            {"LIN", &TrajTableEntry::lin, 1ULL << 0, 1ULL << 1, false},
            {"PAR", &TrajTableEntry::par, 1ULL << 2, 1ULL << 3, false},
            {"SEG", &TrajTableEntry::seg, 1ULL << 16, 1ULL << 17, false},
        };

        /** Set the first/last boundary flag of every counter the scan writes.
         *
         * A frame -- one image -- is a setting of every frame counter at once,
         * and it is complete when every encoding position inside it has
         * arrived. So a frame counter's boundary is read within the other
         * frame counters, and an encoding counter's within all of them; inside
         * that group the first and last occurrence of each value are the
         * boundary. Encoding spaces are bounded separately, so a navigator
         * never closes an imaging slice.
         *
         * Reading the boundary off the counter's extent instead would fire
         * `LAST_IN_SLICE` on every acquisition of the highest-numbered slice
         * and on none of the others.
         *
         * The pulseq interpreter tracks the same boundary by remembering the
         * last acquisition of each slice and matching on its whole counter
         * tuple less REP, which agrees with this wherever the remaining frame
         * counters do not vary -- the usual case. Where they do, a scan of
         * several echoes say, that match lands on whichever echo happened to
         * be played last and this closes the slice once per echo, which is
         * the unit a reconstruction buffers.
         */
        void set_boundary_flags(
            std::vector<TrajTableEntry>& table,
            const std::set<std::string>& written)
        {
            std::vector<int> key;
            for (const BoundaryCounter& counter : BOUNDARY_COUNTERS)
            {
                if (written.count(counter.name) == 0)
                    continue;

                std::vector<const BoundaryCounter*> enclosing;
                for (const BoundaryCounter& other : BOUNDARY_COUNTERS)
                    if (other.is_frame && &other != &counter && written.count(other.name))
                        enclosing.push_back(&other);

                std::map<std::vector<int>, size_t> first_at, last_at;
                for (size_t i = 0; i < table.size(); ++i)
                {
                    key.clear();
                    key.push_back(table[i].encoding_space_ref);
                    for (const BoundaryCounter* other : enclosing)
                        key.push_back(table[i].*(other->field));
                    key.push_back(table[i].*(counter.field));
                    first_at.emplace(key, i);
                    last_at[key] = i;
                }

                for (const auto& entry : first_at)
                    table[entry.second].flags |= counter.first_bit;
                for (const auto& entry : last_at)
                    table[entry.second].flags |= counter.last_bit;
            }
        }

        /** The gradient's value at time `t` (seconds) from block start, Hz/m.
         * Trapezoids evaluate analytically; arbitrary/extended gradients by
         * linear interpolation of their shape. Zero outside the waveform. */
        double grad_value_at(const pulseq::Sequence& seq, int id, double t)
        {
            if (id == 0)
                return 0.0;
            const int row_index = seq.grad_row(id);
            if (seq.grad_kind(id) == pulseq::GradKind::Trap)
            {
                const double* d = seq.trap_library().row(row_index);
                const double amp = d[0], rise = d[1], flat = d[2], fall = d[3], delay = d[4];
                const double local = t - delay;
                if (local <= 0.0 || local >= rise + flat + fall)
                    return 0.0;
                if (local < rise)
                    return amp * (rise > 0.0 ? local / rise : 1.0);
                if (local <= rise + flat)
                    return amp;
                return amp * (fall > 0.0 ? (rise + flat + fall - local) / fall : 0.0);
            }

            const double* d = seq.arb_library().row(row_index);
            const double amp = d[0], delay = d[5];
            const int shape_id = static_cast<int>(d[3]);
            const int time_id = static_cast<int>(d[4]);
            if (shape_id <= 0)
                return 0.0;
            const std::vector<double> wave = pulseq::decompress_shape(
                seq.shape_library().samples(shape_id),
                seq.shape_library().num_compressed(shape_id),
                seq.shape_library().num_uncompressed(shape_id));
            if (wave.empty())
                return 0.0;
            const double raster = seq.grad_raster_time();
            std::vector<double> tt;
            if (time_id > 0)
            {
                tt = pulseq::decompress_shape(
                    seq.shape_library().samples(time_id),
                    seq.shape_library().num_compressed(time_id),
                    seq.shape_library().num_uncompressed(time_id));
                for (double& tick : tt)
                    tick *= raster;
            }
            else
            {
                tt.resize(wave.size());
                for (size_t i = 0; i < wave.size(); ++i)
                    tt[i] = (static_cast<double>(i) + 0.5) * raster;
            }
            const double local = t - delay;
            if (local <= tt.front())
                return local < 0.0 ? 0.0 : amp * wave.front();
            if (local >= tt.back())
                return 0.0;
            size_t j = 0;
            while (j + 1 < tt.size() && tt[j + 1] < local)
                ++j;
            const double span = tt[j + 1] - tt[j];
            const double w = span > 0.0 ? (local - tt[j]) / span : 0.0;
            return amp * (wave[j] + w * (wave[j + 1] - wave[j]));
        }

        const std::array<double, 3> kNoOrigin{{0.0, 0.0, 0.0}};

        /** Content-deduplicating kshot library builder. */
        struct KshotLibrary
        {
            std::vector<Kshot> shots;
            std::map<std::vector<float>, int> index;

            int add(std::vector<float> k)
            {
                const auto it = index.find(k);
                if (it != index.end())
                    return it->second;
                const int id = static_cast<int>(shots.size());
                index.emplace(k, id);
                shots.push_back(Kshot{std::move(k)});
                return id;
            }
        };

        struct SubseqTrajectory
        {
            std::vector<Kshot> kshots;
            std::vector<std::array<float, 9>> rotations;
            std::vector<EncodingSpace> encoding_spaces;
            std::vector<TrajTableEntry> table;
        };

        void fill_label_limits(
            const std::vector<TrajTableEntry>& table,
            int es_local,
            EncodingSpace* es)
        {
            bool first = true;
            for (const TrajTableEntry& entry : table)
            {
                if (entry.encoding_space_ref != es_local)
                    continue;
                auto take = [&](LabelLimit& limit, int value)
                {
                    if (first)
                        limit.min = limit.max = value;
                    else
                    {
                        limit.min = std::min(limit.min, value);
                        limit.max = std::max(limit.max, value);
                    }
                };
                take(es->label_limits.slc, entry.slc);
                take(es->label_limits.phs, entry.phs);
                take(es->label_limits.rep, entry.rep);
                take(es->label_limits.avg, entry.avg);
                take(es->label_limits.seg, entry.seg);
                take(es->label_limits.set, entry.set);
                take(es->label_limits.eco, entry.eco);
                take(es->label_limits.par, entry.par);
                take(es->label_limits.lin, entry.lin);
                take(es->label_limits.acq, entry.acq);
                first = false;
            }
        }

        SubseqTrajectory build_subseq_trajectory(
            pulseq::Sequence& seq,
            const pulseq::KSpace& ks,
            int subseq_idx)
        {
            SubseqTrajectory out;

            /* Rotation library: identity is index 0 so an unrotated entry can
             * say so with rotation_id 0; the file's ids follow at the same
             * offsets they had. */
            out.rotations.push_back({1, 0, 0, 0, 1, 0, 0, 0, 1});
            for (int id = 1; id <= seq.rotation_library().size(); ++id)
                out.rotations.push_back(quaternion_to_matrix(seq.rotation_library().row(id)));

            const std::vector<LabelState> labels = label_states_per_adc(seq);
            if (labels.size() != ks.readouts.size())
                throw std::runtime_error("label walk and k-space walk disagree on ADC count");

            const int rotations_id_type = seq.find_extension_type_id("ROTATIONS");

            /* A file written for a consumer says so, and then a readout that
             * stored a base trajectory is one whose shift was left for us. */
            const bool shift_deferred_anywhere = pulseq::has_base_trajectory(seq);

            /* k at the start of every block, logical frame.  The base a
             * readout stores is its own block's normalised moment and carries
             * no history, so the accumulated moment has to come from here. */
            const std::vector<std::array<double, 3>> origins = pulseq::block_k_origins(seq);

            KshotLibrary library;
            out.table.reserve(ks.readouts.size());

            for (size_t r = 0; r < ks.readouts.size(); ++r)
            {
                const pulseq::Readout& readout = ks.readouts[r];
                const int nsamples = readout.num_samples;
                const pulseq::Block block = seq.get_block(readout.block_index);

                /* The trajectory this readout samples, as the FILE states it.
                 *
                 * The design side already integrated the gradients and wrote
                 * the result into the ADC's phase_modulation, normalised so
                 * one row serves every instance that differs by amplitude.
                 * Re-integrating it here would be the same arithmetic done a
                 * second time, and would throw that sharing away.
                 *
                 * A readout with no stored base gets no trajectory at all:
                 * that is a Cartesian line, which the encoding counters
                 * locate, and an MRD acquisition is entitled to carry none. */
                const pulseq::BaseTrajectory stored =
                    pulseq::read_base_trajectory(seq, readout.block_index);

                /* Axis-major within the readout: x, then y, then z. */
                std::array<std::vector<float>, 3> rows;
                if (stored.has_adc)
                {
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        rows[axis].assign(static_cast<size_t>(nsamples), 0.0f);
                        const std::vector<double>& b = stored.axis[axis].base;
                        for (int i = 0; i < nsamples && i < static_cast<int>(b.size()); ++i)
                            rows[axis][static_cast<size_t>(i)] = static_cast<float>(b[static_cast<size_t>(i)]);
                    }
                }

                int rotation_id = 0;
                {
                    int link = block.ext;
                    while (link)
                    {
                        const int32_t* row = seq.extensions_library().row(link);
                        if (row[0] == rotations_id_type && rotations_id_type != 0)
                        {
                            rotation_id = row[1];
                            break;
                        }
                        link = row[2];
                    }
                }

                TrajTableEntry entry{};
                if (shift_deferred_anywhere && block.adc > 0)
                {
                    const double* adc_row = seq.adc_library().row(static_cast<int>(block.adc));
                    entry.defers_fov_shift = (adc_row[7] != 0.0) ? 1 : 0;
                }
                int shot_ids[3] = {-1, -1, -1};
                float* amplitudes[3] = {
                    &entry.gx_amplitude,
                    &entry.gy_amplitude,
                    &entry.gz_amplitude};
                if (stored.has_adc)
                {
                    const std::array<double, 3>& origin =
                        (readout.block_index >= 0 &&
                         readout.block_index < static_cast<int>(origins.size()))
                        ? origins[static_cast<size_t>(readout.block_index)]
                        : kNoOrigin;
                    for (int axis = 0; axis < 3; ++axis)
                    {
                        /* Every axis, including one the block does not drive:
                         * its k is constant at the origin across the window,
                         * and a rotation downstream mixes it into the others.
                         * The zero row it interns is one row for the whole
                         * scan. */
                        *amplitudes[axis] = static_cast<float>(stored.axis[axis].amplitude);
                        shot_ids[axis] = library.add(rows[axis]);
                        entry.k_origin[axis] = static_cast<float>(origin[static_cast<size_t>(axis)]);
                    }
                }
                entry.kx_shot_id = shot_ids[0];
                entry.ky_shot_id = shot_ids[1];
                entry.kz_shot_id = shot_ids[2];
                entry.rotation_id = rotation_id;
                entry.center_sample = readout.center_sample;
                entry.sample_time_us = static_cast<float>(readout.dwell * 1e6);

                const LabelState& state = labels[r];
                entry.lin = state.get("LIN");
                entry.slc = state.get("SLC");
                entry.eco = state.get("ECO");
                entry.rep = state.get("REP");
                entry.phs = state.get("PHS");
                entry.set = state.get("SET");
                entry.seg = state.get("SEG");
                entry.avg = state.get("AVG");
                entry.par = state.get("PAR");
                entry.acq = state.get("ACQ");
                entry.off = state.get("OFF") ? 1 : 0;

                entry.flags = 0;
                if (state.get("NAV"))
                    entry.flags |= FLAG_NAV;
                if (state.get("REV"))
                    entry.flags |= FLAG_REV;
                if (state.get("REF"))
                    entry.flags |= FLAG_REF;
                if (state.get("IMA"))
                    entry.flags |= FLAG_IMA;
                if (state.get("NOISE"))
                    entry.flags |= FLAG_NOISE;

                entry.encoding_space_ref = (entry.flags & FLAG_NAV) ? 1 : 0;
                out.table.push_back(entry);
            }

            out.kshots = std::move(library.shots);

            if (!out.table.empty())
                out.table.back().flags |= FLAG_LAST_IN_MEASUREMENT;

            set_boundary_flags(out.table, written_labels(labels));

            bool has_nav = false;
            for (const TrajTableEntry& entry : out.table)
                has_nav = has_nav || (entry.flags & FLAG_NAV) != 0;

            EncodingSpace primary;
            primary.subseq_idx = subseq_idx;
            primary.nav_subseq_offset = has_nav ? 1 : 0;
            primary.geometry_tag = 0;
            fill_label_limits(out.table, 0, &primary);
            out.encoding_spaces.push_back(primary);
            if (has_nav)
            {
                EncodingSpace nav;
                nav.subseq_idx = subseq_idx;
                nav.nav_subseq_offset = 0;
                nav.geometry_tag = 1;
                fill_label_limits(out.table, 1, &nav);
                out.encoding_spaces.push_back(nav);
            }

            return out;
        }

        void merge_subseq(SequenceCache& cache, SubseqTrajectory&& one)
        {
            const int kshot_base = static_cast<int>(cache.kshots.size());
            const int rotation_base = static_cast<int>(cache.rotations.size());
            const int es_base = static_cast<int>(cache.encoding_spaces.size());

            for (Kshot& shot : one.kshots)
                cache.kshots.push_back(std::move(shot));
            for (const auto& matrix : one.rotations)
                cache.rotations.push_back(matrix);
            for (const EncodingSpace& es : one.encoding_spaces)
                cache.encoding_spaces.push_back(es);

            for (TrajTableEntry entry : one.table)
            {
                if (entry.kx_shot_id >= 0)
                    entry.kx_shot_id += kshot_base;
                if (entry.ky_shot_id >= 0)
                    entry.ky_shot_id += kshot_base;
                if (entry.kz_shot_id >= 0)
                    entry.kz_shot_id += kshot_base;
                entry.rotation_id += rotation_base;
                entry.encoding_space_ref += es_base;
                cache.table.push_back(entry);
            }
        }

        /* ============================================================== */
        /*  Sequence description (SeqEvent rows + RF-def library)         */
        /*                                                                */
        /*  Derived only under a TRSize declaration: the design side       */
        /*  states the structural TR, the reader locates the first window  */
        /*  whose block pattern repeats (or ends the file) and verifies    */
        /*  it, and one row is emitted per window block. Without the       */
        /*  declaration the description is skipped -- it feeds subspace    */
        /*  reconstruction and parameter fitting, never basic imaging.     */
        /* ============================================================== */

        /** The normalized complex envelope of one RF definition, on its own
         * time grid and resampled to the RF raster. */
        struct RfEnvelope
        {
            std::vector<float> re_uniform, im_uniform;
            float raster_us = 0.0f;
            float duration_us = 0.0f;
            float center_us = 0.0f;
        };

        RfEnvelope build_rf_envelope(const pulseq::Sequence& seq, int rf_id)
        {
            const double* row = seq.rf_library().row(rf_id);
            const int mag_id = static_cast<int>(row[1]);
            const int phase_id = static_cast<int>(row[2]);
            const int time_id = static_cast<int>(row[3]);

            const auto decompress = [&](int shape_id) -> std::vector<double>
            {
                if (shape_id <= 0)
                    return {};
                return pulseq::decompress_shape(
                    seq.shape_library().samples(shape_id),
                    seq.shape_library().num_compressed(shape_id),
                    seq.shape_library().num_uncompressed(shape_id));
            };

            const std::vector<double> mag = decompress(mag_id);
            const std::vector<double> phase = decompress(phase_id);
            const std::vector<double> time = decompress(time_id);
            const size_t n = mag.size();

            RfEnvelope env;
            env.raster_us = static_cast<float>(seq.rf_raster_time() * 1e6);
            if (n == 0)
                return env;

            std::vector<float> t_us(n), re(n), im(n);
            for (size_t i = 0; i < n; ++i)
            {
                t_us[i] = time.size() == n ? static_cast<float>(time[i] * env.raster_us)
                                           : (static_cast<float>(i) + 0.5f) * env.raster_us;
                const double turns = phase.size() == n ? phase[i] : 0.0;
                re[i] = static_cast<float>(mag[i] * std::cos(2.0 * M_PI * turns));
                im[i] = static_cast<float>(mag[i] * std::sin(2.0 * M_PI * turns));
            }
            env.duration_us = t_us.back();
            env.center_us = static_cast<float>(row[4] * 1e6);

            /* Resample onto the uniform raster grid, linearly. */
            const int n_uniform =
                std::max(2, static_cast<int>(std::ceil(env.duration_us / env.raster_us)));
            env.re_uniform.resize(static_cast<size_t>(n_uniform));
            env.im_uniform.resize(static_cast<size_t>(n_uniform));
            size_t j = 0;
            for (int i = 0; i < n_uniform; ++i)
            {
                const float t = static_cast<float>(i) * env.raster_us;
                while (j + 1 < n && t_us[j + 1] < t)
                    ++j;
                if (t <= t_us.front())
                {
                    env.re_uniform[i] = re.front();
                    env.im_uniform[i] = im.front();
                }
                else if (t >= t_us.back())
                {
                    env.re_uniform[i] = re.back();
                    env.im_uniform[i] = im.back();
                }
                else
                {
                    const float span = t_us[j + 1] - t_us[j];
                    const float w = span > 0.0f ? (t - t_us[j]) / span : 0.0f;
                    env.re_uniform[i] = re[j] + w * (re[j + 1] - re[j]);
                    env.im_uniform[i] = im[j] + w * (im[j + 1] - im[j]);
                }
            }
            return env;
        }

        RfShapeSamples copy_compressed_shape(const pulseq::Sequence& seq, int shape_id)
        {
            RfShapeSamples out;
            if (shape_id <= 0)
                return out;
            out.num_uncompressed = seq.shape_library().num_uncompressed(shape_id);
            const int count = seq.shape_library().num_compressed(shape_id);
            const double* samples = seq.shape_library().samples(shape_id);
            out.samples.assign(samples, samples + count);
            return out;
        }

        /** Bandwidth, bands and b1sq of one definition, off the same spectrum
         * machinery and the same thresholds the interpreter uses. */
        void fill_rf_stats(RfDef* def, const RfEnvelope& env)
        {
            const size_t n = env.re_uniform.size();
            if (n == 0)
                return;

            double sum_sq = 0.0;
            for (size_t i = 0; i < n; ++i)
                sum_sq += static_cast<double>(env.re_uniform[i]) * env.re_uniform[i] +
                    static_cast<double>(env.im_uniform[i]) * env.im_uniform[i];
            def->total_b1sq_power = static_cast<float>(sum_sq * env.raster_us * 1e-6);

            pulseq_rf_spectrum* plan = nullptr;
            if (pulseq_rf_spectrum_create(&plan, env.raster_us, 0.0f) <= 0 || plan == nullptr)
                return;

            std::vector<float> t_us(n);
            for (size_t i = 0; i < n; ++i)
                t_us[i] = static_cast<float>(i) * env.raster_us;

            if (pulseq_rf_spectrum_run(
                    plan,
                    env.re_uniform.data(),
                    env.im_uniform.data(),
                    t_us.data(),
                    static_cast<int>(n),
                    env.center_us) > 0)
            {
                def->bandwidth_hz = pulseq_rf_bandwidth(plan, 0.0f, nullptr);
                def->band_bandwidth_hz = def->bandwidth_hz;

                /* Multiband split: contiguous runs above 0.3 x spectral peak,
                 * one centroid frequency per run. */
                const int size = pulseq_rf_spectrum_size(plan);
                const float* freq = pulseq_rf_spectrum_freq(plan);
                const float* sre = pulseq_rf_spectrum_re(plan);
                const float* sim = pulseq_rf_spectrum_im(plan);
                float peak = 0.0f;
                std::vector<float> magnitude(static_cast<size_t>(size));
                for (int i = 0; i < size; ++i)
                {
                    magnitude[i] = std::sqrt(sre[i] * sre[i] + sim[i] * sim[i]);
                    peak = std::max(peak, magnitude[i]);
                }
                if (peak > 1e-9f)
                {
                    const float threshold = 0.3f * peak;
                    int bands = 0;
                    bool in_band = false;
                    double wsum = 0.0, msum = 0.0;
                    for (int i = 0; i <= size; ++i)
                    {
                        const float m = i < size ? magnitude[i] : 0.0f;
                        if (m >= threshold)
                        {
                            wsum += static_cast<double>(freq[i]) * m;
                            msum += m;
                            in_band = true;
                        }
                        else if (in_band)
                        {
                            if (bands < 8)
                                def->band_freq_offsets_hz[bands] =
                                    msum > 0.0 ? static_cast<float>(wsum / msum) : 0.0f;
                            ++bands;
                            wsum = msum = 0.0;
                            in_band = false;
                        }
                    }
                    if (bands >= 1)
                        def->num_bands = bands;
                }
            }
            pulseq_rf_spectrum_free(plan);
        }

        /** A block's structural signature: shapes and timing, free of the
         * per-instance quantities (amplitudes, phase/frequency offsets,
         * rotations, labels) that legitimately vary between TRs -- RF
         * spoiling and phase alternation register one RF library row per
         * value, so raw event ids do not repeat where the structure does. */
        void append_block_signature(
            const pulseq::Sequence& seq,
            int block_index,
            std::vector<int64_t>* out)
        {
            const pulseq::Block block = seq.get_block(block_index);
            out->push_back(
                static_cast<int64_t>(std::llround(block.duration / seq.block_duration_raster())));

            if (block.rf != 0)
            {
                const double* rf_row = seq.rf_library().row(block.rf);
                out->push_back(static_cast<int64_t>(rf_row[1])); /* mag shape   */
                out->push_back(static_cast<int64_t>(rf_row[2])); /* phase shape */
                out->push_back(static_cast<int64_t>(rf_row[3])); /* time shape  */
                out->push_back(static_cast<int64_t>(std::llround(rf_row[5] * 1e9)));
            }
            else
            {
                out->insert(out->end(), 4, 0);
            }

            for (const int grad_id : {block.gx, block.gy, block.gz})
            {
                if (grad_id == 0)
                {
                    out->insert(out->end(), 4, 0);
                    continue;
                }
                const double* row =
                    (seq.grad_kind(grad_id) == pulseq::GradKind::Trap ? seq.trap_library()
                                                                      : seq.arb_library())
                        .row(seq.grad_row(grad_id));
                if (seq.grad_kind(grad_id) == pulseq::GradKind::Trap)
                {
                    out->push_back(1);
                    out->push_back(static_cast<int64_t>(std::llround(row[1] * 1e9)));
                    out->push_back(static_cast<int64_t>(std::llround(row[2] * 1e9)));
                    out->push_back(static_cast<int64_t>(std::llround(row[3] * 1e9)));
                }
                else
                {
                    out->push_back(2);
                    out->push_back(static_cast<int64_t>(row[3])); /* shape id  */
                    out->push_back(static_cast<int64_t>(row[4])); /* time id   */
                    out->push_back(static_cast<int64_t>(std::llround(row[5] * 1e9)));
                }
            }

            out->push_back(static_cast<int64_t>(block.adc));
        }

        /** The first block index (0-based) of a window of `tr_size` blocks
         * whose structural signature repeats to the next window or ends the
         * file. -1 when no such window exists. */
        int locate_tr_window(const pulseq::Sequence& seq, int tr_size)
        {
            const int n_blocks = seq.num_blocks();
            if (tr_size <= 0 || tr_size > n_blocks)
                return -1;

            std::vector<std::vector<int64_t>> signatures(static_cast<size_t>(n_blocks));
            for (int i = 0; i < n_blocks; ++i)
                append_block_signature(seq, i + 1, &signatures[i]);

            const auto windows_equal = [&](int a, int b)
            {
                for (int i = 0; i < tr_size; ++i)
                    if (signatures[a + i] != signatures[b + i])
                        return false;
                return true;
            };

            const auto window_has_adc = [&](int start)
            {
                for (int i = 0; i < tr_size; ++i)
                    if (seq.get_block(start + 1 + i).adc != 0)
                        return true;
                return false;
            };

            int start = -1;
            for (int candidate = 0; candidate + tr_size <= n_blocks; ++candidate)
            {
                const bool repeats = candidate + 2 * tr_size <= n_blocks
                    ? windows_equal(candidate, candidate + tr_size)
                    : candidate + tr_size == n_blocks;
                if (repeats)
                {
                    start = candidate;
                    break;
                }
            }
            if (start < 0)
                return -1;

            /* Prefer a window that acquires -- a dummy TR repeats just as
             * well but describes nothing a reconstruction can use -- and
             * skip forward by WHOLE TRs so the window's phase (which block
             * of the pattern comes first) stays put. */
            while (!window_has_adc(start) && start + 2 * tr_size <= n_blocks)
                start += tr_size;
            return start;
        }

        char rf_use_tag(const pulseq::Sequence& seq, int rf_id)
        {
            const auto& uses = seq.rf_uses();
            const size_t index = static_cast<size_t>(rf_id) - 1;
            return index < uses.size() && uses[index] ? uses[index] : 'u';
        }

        int rf_use_code(char use)
        {
            /* PULSEQ_RF_USE_* order: the initials Pulseq writes. */
            static const char kUseChars[] = {'u', 'e', 'r', 'i', 's', 'p', 'o'};
            for (int i = 0; i < static_cast<int>(sizeof(kUseChars)); ++i)
                if (kUseChars[i] == use)
                    return i;
            return 0;
        }

        SequenceDescription derive_sequence_description(
            pulseq::Sequence& seq,
            const pulseq::KSpace& ks,
            int subseq_idx,
            int tr_size)
        {
            SequenceDescription sd;
            sd.subseq_idx = subseq_idx;

            const int start = locate_tr_window(seq, tr_size);
            if (start < 0)
                return sd;

            /* RF definitions are shapes: library rows differing only in
             * amplitude, phase or frequency (RF spoiling makes one row per
             * increment) collapse onto one definition, in first-use order. */
            std::map<std::array<int, 3>, int> def_by_shapes;
            std::vector<int> def_of_rf_id(static_cast<size_t>(seq.rf_library().size()) + 1, -1);
            std::vector<int> representative_rf_id;
            for (int id = 1; id <= seq.rf_library().size(); ++id)
            {
                const double* rf_row = seq.rf_library().row(id);
                const std::array<int, 3> key = {
                    static_cast<int>(rf_row[1]),
                    static_cast<int>(rf_row[2]),
                    static_cast<int>(rf_row[3])};
                const auto found = def_by_shapes.find(key);
                if (found == def_by_shapes.end())
                {
                    const int def_id = static_cast<int>(representative_rf_id.size());
                    def_by_shapes.emplace(key, def_id);
                    def_of_rf_id[id] = def_id;
                    representative_rf_id.push_back(id);
                }
                else
                {
                    def_of_rf_id[id] = found->second;
                }
            }

            /* Absolute block start times, seconds. */
            std::vector<double> block_start(static_cast<size_t>(seq.num_blocks()) + 1, 0.0);
            for (int i = 0; i < seq.num_blocks(); ++i)
                block_start[i + 1] = block_start[i] + seq.get_block(i + 1).duration;
            const double window_t0 = block_start[start];
            sd.tr_duration_us =
                static_cast<float>((block_start[start + tr_size] - window_t0) * 1e6);

            /* Per-readout facts, and the echo flag reduced over every
             * instance at the same window offset. The criterion is relative:
             * a position bears an echo when some instance of it reaches the
             * scan's own closest approach to k = 0 -- a finite phase-encode
             * offset stays out, accumulated float error stays in. */
            double k_max = 0.0;
            double floor = 1e30;
            std::map<int, double> min_k_by_offset;
            std::map<int, const pulseq::Readout*> readout_by_block;
            for (const pulseq::Readout& readout : ks.readouts)
            {
                readout_by_block[readout.block_index] = &readout;
                const int offset = (readout.block_index - 1 - start) % tr_size;
                const double k_norm =
                    std::hypot(readout.k_echo[0], std::hypot(readout.k_echo[1], readout.k_echo[2]));
                for (int axis = 0; axis < 3; ++axis)
                    k_max = std::max(k_max, std::abs(readout.k_echo[axis]));
                floor = std::min(floor, k_norm);
                const auto seen = min_k_by_offset.find(offset);
                if (seen == min_k_by_offset.end() || k_norm < seen->second)
                    min_k_by_offset[offset] = k_norm;
            }
            if (floor >= 1e30)
                floor = 0.0;
            const double echo_threshold = floor + 1e-5 * k_max + 1e-6;
            std::map<int, bool> echo_by_offset;
            for (const auto& entry : min_k_by_offset)
                echo_by_offset[entry.first] = entry.second <= echo_threshold;

            /* ADC roles are STRUCTURAL: centrality is judged along the axes
             * the readout itself sweeps, ignoring offsets carried in from
             * encodes -- a CPMG echo is central whatever line it encodes,
             * because its own sweep crosses zero. Reduced over instances,
             * classified as the interpreter classifies: every acquisition
             * starts SINGLE; with several, the positions at the window's
             * minimum (within 1e-3 of its maximum) are the centre --
             * ECHO_CENTER when unique, SINGLE when tied (a CPMG train's
             * every echo, a PROPELLER's every blade) -- rest NON_CENTER. */
            std::map<int, double> role_norm_by_offset;
            for (const pulseq::Readout& readout : ks.readouts)
            {
                const int offset = (readout.block_index - 1 - start) % tr_size;
                const double* base =
                    ks.k_adc.data() + static_cast<size_t>(readout.sample_offset) * 3;
                double norm_sq = 0.0;
                for (int axis = 0; axis < 3; ++axis)
                {
                    const double* row = base + axis * readout.num_samples;
                    double peak = 0.0;
                    for (int sample = 0; sample < readout.num_samples; ++sample)
                        peak = std::max(peak, std::abs(row[sample]));
                    const double sweep = std::abs(row[readout.num_samples - 1] - row[0]);
                    if (sweep > 1e-6 * std::max(1.0, peak))
                        norm_sq += readout.k_echo[axis] * readout.k_echo[axis];
                }
                const double norm = std::sqrt(norm_sq);
                const auto seen = role_norm_by_offset.find(offset);
                if (seen == role_norm_by_offset.end() || norm < seen->second)
                    role_norm_by_offset[offset] = norm;
            }

            std::vector<int> role_of(static_cast<size_t>(tr_size), ADC_ROLE_SINGLE);
            {
                std::vector<int> offsets;
                std::vector<double> k_at_echo;
                for (int i = 0; i < tr_size; ++i)
                {
                    const pulseq::Block block = seq.get_block(start + 1 + i);
                    if (block.adc == 0)
                        continue;
                    const auto found = role_norm_by_offset.find(i);
                    if (found == role_norm_by_offset.end())
                        continue;
                    offsets.push_back(i);
                    k_at_echo.push_back(found->second);
                }
                if (offsets.size() > 1)
                {
                    double lo = 1e30, hi = 0.0;
                    for (double k : k_at_echo)
                    {
                        lo = std::min(lo, k);
                        hi = std::max(hi, k);
                    }
                    if (hi > 0.0)
                    {
                        const double threshold = lo + 1e-3 * hi;
                        int centred = 0;
                        for (double k : k_at_echo)
                            centred += k <= threshold ? 1 : 0;
                        const int centre_role =
                            centred == 1 ? ADC_ROLE_ECHO_CENTER : ADC_ROLE_SINGLE;
                        for (size_t j = 0; j < offsets.size(); ++j)
                            role_of[offsets[j]] =
                                k_at_echo[j] <= threshold ? centre_role : ADC_ROLE_NON_CENTER;
                    }
                }
            }

            sd.events.reserve(static_cast<size_t>(tr_size));
            for (int i = 0; i < tr_size; ++i)
            {
                const int block_index = start + 1 + i;
                const pulseq::Block block = seq.get_block(block_index);
                SeqEvent row{};
                const double t_block_us = (block_start[block_index - 1] - window_t0) * 1e6;

                if (block.rf != 0)
                {
                    const double* rf_row = seq.rf_library().row(block.rf);
                    row.type = SEQ_EVENT_RF;
                    /* The RF isocenter: delay + the pulse's own centre. */
                    row.timestamp_us =
                        static_cast<float>(t_block_us + (rf_row[5] + rf_row[4]) * 1e6);
                    row.params[0] = static_cast<float>(def_of_rf_id[block.rf]);
                    row.params[1] = static_cast<float>(rf_use_code(rf_use_tag(seq, block.rf)));
                    row.params[2] = static_cast<float>(rf_row[0]); /* amplitude, Hz */
                    row.params[3] = static_cast<float>(rf_row[9]); /* phase, rad */
                    row.params[4] = static_cast<float>(rf_row[8]); /* freq, Hz */

                    /* The shim rides the extension chain. -1 when absent. */
                    int shim_id = -1;
                    const int shims_type = seq.find_extension_type_id("RF_SHIMS");
                    int link = block.ext;
                    while (link && shims_type)
                    {
                        const int32_t* ext = seq.extensions_library().row(link);
                        if (ext[0] == shims_type)
                        {
                            shim_id = ext[1] - 1;
                            break;
                        }
                        link = ext[2];
                    }
                    row.params[5] = static_cast<float>(shim_id);

                    /* Slice-selection amplitude: the gradient under the RF
                     * center, when exactly one axis carries one -- trapezoid
                     * or composite (a crush-slab-crush extended trapezoid's
                     * plateau reads the same way). */
                    const double t_center = rf_row[5] + rf_row[4];
                    float ss_amp = 0.0f;
                    int active = 0;
                    for (const int grad_id : {block.gx, block.gy, block.gz})
                    {
                        const double value = grad_value_at(seq, grad_id, t_center);
                        if (std::abs(value) > 1e-3)
                        {
                            ++active;
                            ss_amp = std::abs(static_cast<float>(value));
                        }
                    }
                    row.params[6] = active == 1 ? ss_amp : 0.0f;
                }
                else if (block.adc != 0)
                {
                    const auto found = readout_by_block.find(block_index);
                    const pulseq::Readout* readout =
                        found != readout_by_block.end() ? found->second : nullptr;
                    row.type = SEQ_EVENT_ADC;
                    const bool echo = echo_by_offset.count(i) && echo_by_offset[i];
                    if (readout != nullptr && readout->center_sample >= 0)
                    {
                        /* The echo anchor is the START of the k = 0 sample:
                         * t0 is the first sample's CENTER, so half a dwell
                         * comes off. */
                        row.timestamp_us = static_cast<float>(
                            (readout->t0 - window_t0 +
                             (readout->center_sample - 0.5) * readout->dwell) *
                            1e6);
                    }
                    else
                    {
                        row.timestamp_us = static_cast<float>(t_block_us);
                    }
                    row.params[0] = static_cast<float>(role_of[i]);
                    row.params[1] = static_cast<float>(seq.adc_library().row(block.adc)[4]);
                    row.params[2] = echo ? 1.0f : 0.0f;
                }
                else
                {
                    row.type = SEQ_EVENT_WAIT;
                    row.timestamp_us = static_cast<float>(t_block_us);
                }
                sd.events.push_back(row);
            }

            /* The RF-def library: one entry per unique shape triplet, in
             * first-use order, matching the rows' rf_def_id. */
            sd.rf_defs.reserve(representative_rf_id.size());
            for (size_t d = 0; d < representative_rf_id.size(); ++d)
            {
                const int id = representative_rf_id[d];
                const double* rf_row = seq.rf_library().row(id);
                RfDef def;
                def.rf_def_id = static_cast<int>(d);
                const RfEnvelope env = build_rf_envelope(seq, id);
                fill_rf_stats(&def, env);
                def.mag = copy_compressed_shape(seq, static_cast<int>(rf_row[1]));
                def.has_phase = static_cast<int>(rf_row[2]) > 0;
                if (def.has_phase)
                    def.phase = copy_compressed_shape(seq, static_cast<int>(rf_row[2]));
                def.has_time = static_cast<int>(rf_row[3]) > 0;
                if (def.has_time)
                    def.time = copy_compressed_shape(seq, static_cast<int>(rf_row[3]));
                sd.rf_defs.push_back(std::move(def));
            }

            /* Deduplicated (def, shim, ss-grad) triplets with the slice
             * geometry each implies. */
            for (const SeqEvent& row : sd.events)
            {
                if (row.type != SEQ_EVENT_RF)
                    continue;
                const int def_id = row.rf_def_id();
                const int shim_id = row.rf_shim_id();
                const float ss = row.rf_ss_grad_amp_hz_per_m();
                bool seen = false;
                for (const RfShapeTuple& tuple : sd.rf_shape_tuples)
                    seen = seen ||
                        (tuple.rf_def_id == def_id && tuple.rf_shim_id == shim_id &&
                         tuple.ss_grad_amp_hz_per_m == ss);
                if (seen)
                    continue;
                RfShapeTuple tuple;
                tuple.tuple_id = static_cast<int>(sd.rf_shape_tuples.size());
                tuple.rf_def_id = def_id;
                tuple.rf_shim_id = shim_id;
                tuple.ss_grad_amp_hz_per_m = ss;
                const float bandwidth = def_id >= 0 && def_id < static_cast<int>(sd.rf_defs.size())
                    ? sd.rf_defs[def_id].bandwidth_hz
                    : 0.0f;
                tuple.slice_thickness_mm =
                    ss > 0.0f && bandwidth > 0.0f ? bandwidth / ss * 1e3f : 0.0f;
                tuple.slice_selective =
                    tuple.slice_thickness_mm > 0.0f && tuple.slice_thickness_mm < 10.0f;
                sd.rf_shape_tuples.push_back(tuple);
            }

            return sd;
        }

        float definition_number(const SequenceCache& cache, const char* key, float fallback = 0.0f)
        {
            const auto it = cache.definitions.find(key);
            if (it == cache.definitions.end() || it->second.empty())
                return fallback;
            return std::strtof(it->second.front().c_str(), nullptr);
        }

        bool file_exists(const std::string& path)
        {
            std::FILE* f = std::fopen(path.c_str(), "rb");
            if (!f)
                return false;
            std::fclose(f);
            return true;
        }

        /* A successor is named relative to the file that names it. */
        std::string sibling_of(const std::string& path, const std::string& name)
        {
            const std::size_t cut = path.find_last_of("/\\");
            if (cut == std::string::npos)
                return name;
            return path.substr(0, cut + 1) + name;
        }

    } // namespace

    SequenceCache read_sequence_files(const std::string& lead_path)
    {
        SequenceCache cache;

        std::vector<std::string> chain;
        std::string cursor = lead_path;
        while (!cursor.empty())
        {
            if (!file_exists(cursor))
                throw std::runtime_error("sequence chain link not found: " + cursor);
            chain.push_back(cursor);

            pulseq::SequenceFile link(cursor);
            const std::string next = link.next_sequence();
            cursor = next.empty() ? "" : sibling_of(cursor, next);
            if (chain.size() > 64)
                throw std::runtime_error("NextSequence chain does not terminate");
        }

        double total_duration = 0.0;
        float min_te = 0.0f, min_tr = 0.0f, max_tr = 0.0f, max_flip = 0.0f;

        for (size_t s = 0; s < chain.size(); ++s)
        {
            pulseq::SequenceFile file(chain[s]);
            auto defs = definitions_as_strings(file.definitions());

            int tr_size = 0;
            const auto declared = defs.find("TRSize");
            if (declared != defs.end() && !declared->second.empty())
                tr_size = static_cast<int>(std::strtod(declared->second.front().c_str(), nullptr));

            for (const auto& kv : defs)
            {
                auto& merged = cache.definitions[kv.first];
                merged.insert(merged.end(), kv.second.begin(), kv.second.end());
            }
            cache.definitions_by_subseq.push_back(std::move(defs));

            pulseq::Sequence& seq = file.sequence();
            pulseq::KSpaceOptions options;
            options.apply_rotation = false; /* the LOGICAL frame */
            const pulseq::KSpace ks = pulseq::calculate_kspace(seq, options);

            merge_subseq(cache, build_subseq_trajectory(seq, ks, static_cast<int>(s)));

            if (tr_size > 0)
            {
                SequenceDescription sd =
                    derive_sequence_description(seq, ks, static_cast<int>(s), tr_size);
                if (!sd.events.empty())
                    cache.seq_descs.push_back(std::move(sd));
            }

            total_duration += seq.duration();
        }
        cache.has_seq_desc = !cache.seq_descs.empty();

        min_te = definition_number(cache, "TE");
        min_tr = max_tr = definition_number(cache, "TR");
        const auto tr_values = cache.definitions.find("TR");
        if (tr_values != cache.definitions.end())
        {
            for (const std::string& value : tr_values->second)
            {
                const float tr = std::strtof(value.c_str(), nullptr);
                min_tr = std::min(min_tr, tr);
                max_tr = std::max(max_tr, tr);
            }
        }
        max_flip = definition_number(cache, "FlipAngle");

        cache.seq_params.min_te_us = min_te * 1e6f;
        cache.seq_params.min_tr_us = min_tr * 1e6f;
        cache.seq_params.max_tr_us = max_tr * 1e6f;
        cache.seq_params.max_flip_angle_deg = max_flip;
        cache.seq_params.total_scan_time_us = static_cast<float>(total_duration * 1e6);
        cache.seq_params.num_subseqs = static_cast<int>(chain.size());

        return cache;
    }

} // namespace mrdserver
