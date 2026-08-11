/**
 * @file kspace.cpp
 * @brief Running the C89 trajectory core over a pulseq::Sequence.
 *
 * See kspace.hpp for why this converts rather than reimplements.
 *
 * @warning raw64.hpp must come first and must not be joined by the ordinary
 * pulseq headers in this translation unit -- they share include guards, so
 * whichever wins silently turns the other into a no-op and calls compile
 * against the wrong struct layout.  The Sequence side is reached through
 * kspace_sequence.cpp instead, which knows nothing about raw64.
 */

#include "pulseq/raw64.hpp"

#include "pulseq/kspace.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

#if defined(__unix__) || defined(__APPLE__)
#define PULSEQ_HAVE_FMEMOPEN 1
#endif

namespace pulseq
{
    namespace
    {
        namespace raw = ::pulseq::raw64;

        /** A raw64::pulseq_file that frees itself. */
        class RawFile
        {
          public:
            RawFile() { raw::pulseq_file_init(&file_, nullptr); }
            ~RawFile() { raw::pulseq_file_free(&file_); }
            RawFile(const RawFile&) = delete;
            RawFile& operator=(const RawFile&) = delete;

            raw::pulseq_file* get() { return &file_; }

          private:
            raw::pulseq_file file_{};
        };

        /**
         * Parse an in-memory binary Pulseq file.
         *
         * fmemopen keeps it in memory where it belongs; tmpfile() is the
         * fallback for platforms without it, and is still correct -- just
         * slower and dependent on a writable temp directory.
         */
        void parse_binary(const std::string& bytes, raw::pulseq_file* out)
        {
            FILE* handle = nullptr;
#ifdef PULSEQ_HAVE_FMEMOPEN
            handle = fmemopen(const_cast<char*>(bytes.data()), bytes.size(), "rb");
#else
            handle = std::tmpfile();
            if (handle != nullptr)
            {
                if (std::fwrite(bytes.data(), 1, bytes.size(), handle) != bytes.size())
                {
                    std::fclose(handle);
                    handle = nullptr;
                }
                else
                {
                    std::rewind(handle);
                }
            }
#endif
            if (handle == nullptr)
                throw std::runtime_error("calculate_kspace: could not open an in-memory buffer");

            const int rc = raw::pulseq_read_binary_from_buffer(out, handle);
            std::fclose(handle);
            if (rc < 0)
                throw std::runtime_error("calculate_kspace: re-reading the sequence failed (" +
                                         std::to_string(rc) + ")");
        }

        /*
         * Put the double-precision shape samples back over the float32 ones
         * the binary round trip produced.
         *
         * The only field the format narrows, and the one it can least afford
         * to: shapes are run-length encoded as derivatives, so decompression
         * is a cumulative sum and a per-sample error of 3e-8 becomes 1e-5 in k
         * by the end of a long readout.  Everything else -- amplitudes,
         * times, rotations, block ids -- comes back bit-identical, verified by
         * comparing the two parses field by field.
         *
         * Sizes are checked rather than assumed: a mismatch would mean the
         * writer and the parser disagree about the library, and silently
         * writing into the wrong row is worse than the precision it fixes.
         */
        void restore_shapes(raw::pulseq_file* file, const double* samples, const int* offsets,
                            int count)
        {
            if (count == 0)
                return;
            if (file->shapes_library_size != count)
                throw std::runtime_error(
                    "calculate_kspace: shape library changed size across the bridge");

            for (int i = 0; i < count; ++i)
            {
                const int n = offsets[i + 1] - offsets[i];
                if (file->shapes_library[i].num_samples != n)
                    throw std::runtime_error(
                        "calculate_kspace: shape " + std::to_string(i + 1) +
                        " changed length across the bridge");
                for (int j = 0; j < n; ++j)
                    file->shapes_library[i].samples[j] = samples[offsets[i] + j];
            }
        }

        void copy_rf(const raw::pulseq_ktraj_rf& src, RfEventTiming& dst)
        {
            dst.block_index = src.block_index;
            dst.t = src.t;
            dst.freq_offset = src.freq_offset;
            dst.phase_offset = src.phase_offset;
            for (int a = 0; a < 3; ++a)
                dst.g[static_cast<size_t>(a)] = src.g[a];
        }

    }  // namespace

    /**
     * The half of calculate_kspace that speaks raw64.
     *
     * Declared in kspace_sequence.cpp, which owns the Sequence side; the two
     * meet only through this signature and a std::string of bytes, which is
     * what keeps the two header sets in separate translation units.
     */
    KSpace kspace_from_binary(const std::string& bytes, const double* shape_samples,
                              const int* shape_offsets, int shape_count,
                              const KSpaceOptions& options)
    {
        RawFile raw_file;
        parse_binary(bytes, raw_file.get());
        const int self_num_blocks = raw_file.get()->num_blocks;
        restore_shapes(raw_file.get(), shape_samples, shape_offsets, shape_count);

        raw::pulseq_ktraj_opts opts;
        raw::pulseq_ktraj_opts_init(&opts);
        for (int a = 0; a < 3; ++a)
        {
            opts.trajectory_delay[a] = options.trajectory_delay[static_cast<size_t>(a)];
            opts.gradient_offset[a] = options.gradient_offset[static_cast<size_t>(a)];
        }
        opts.first_block = options.first_block;
        opts.last_block = options.last_block;
        opts.apply_rotation = options.apply_rotation ? 1 : 0;
        opts.sample_window_average = options.sample_window_average ? 1 : 0;
        opts.derive_center_sample = options.derive_center_sample ? 1 : 0;

        raw::pulseq_ktraj_plan* plan = nullptr;
        const int rc = raw::pulseq_ktraj_plan_create(&plan, raw_file.get(), &opts);
        if (rc < 0)
            throw std::runtime_error("calculate_kspace: the trajectory core failed (" +
                                     std::to_string(rc) + ")");

        KSpace out;
        try
        {
            const int num_readouts = raw::pulseq_ktraj_num_readouts(plan);
            out.readouts.resize(static_cast<size_t>(num_readouts));
            out.key_groups = raw::pulseq_ktraj_num_key_groups(plan);
            out.total_duration = raw::pulseq_ktraj_total_duration(plan);
            {
                double centre[3] = {0.0, 0.0, 0.0};
                raw::pulseq_ktraj_center(plan, centre);
                for (int a = 0; a < 3; ++a)
                    out.k_center[static_cast<size_t>(a)] = centre[a];
            }

            int total = 0;
            for (int i = 0; i < num_readouts; ++i)
            {
                raw::pulseq_ktraj_readout info;
                raw::pulseq_ktraj_readout_info(plan, i, &info);
                Readout& r = out.readouts[static_cast<size_t>(i)];
                r.block_index = info.block_index;
                r.num_samples = info.num_samples;
                r.t0 = info.t0;
                r.dwell = info.dwell;
                r.center_sample = info.center_sample;
                r.rotation_id = info.rotation_id;
                for (int a = 0; a < 3; ++a)
                {
                    r.g_echo[static_cast<size_t>(a)] = info.g_echo[a];
                    r.k_echo[static_cast<size_t>(a)] = info.k_echo[a];
                    r.constant_axes[static_cast<size_t>(a)] =
                        (info.constant_axes & (1u << a)) != 0u;
                }
                r.sample_offset = total;
                total += (info.num_samples > 0) ? info.num_samples : 0;
            }
            out.total_samples = total;

            /* One rotation shared by every readout is a change of frame; more
             * than one is an encoding.  See KSpace::rotations_vary. */
            for (int i = 1; i < num_readouts; ++i)
            {
                if (out.readouts[static_cast<size_t>(i)].rotation_id !=
                    out.readouts[0].rotation_id)
                {
                    out.rotations_vary = true;
                    break;
                }
            }

            if (options.materialize_samples)
            {
                out.k_adc.resize(static_cast<size_t>(total) * 3);
                out.t_adc.resize(static_cast<size_t>(total));
                for (int i = 0; i < num_readouts; ++i)
                {
                    const Readout& r = out.readouts[static_cast<size_t>(i)];
                    if (r.num_samples <= 0)
                        continue;
                    raw::pulseq_ktraj_readout_k(
                        plan, i, out.k_adc.data() + static_cast<size_t>(r.sample_offset) * 3);
                    for (int j = 0; j < r.num_samples; ++j)
                        out.t_adc[static_cast<size_t>(r.sample_offset + j)] =
                            r.t0 + static_cast<double>(j) * r.dwell;
                }
            }

            if (options.block_origins)
            {
                const int first = (options.first_block > 0) ? options.first_block : 1;
                const int last =
                    (options.last_block > 0) ? options.last_block : self_num_blocks;
                out.block_origins.resize(
                    static_cast<size_t>((last >= first) ? (last - first + 1) : 0));
                for (int b = first; b <= last; ++b)
                {
                    double k[3] = {0.0, 0.0, 0.0};
                    raw::pulseq_ktraj_block_origin(plan, b, k);
                    for (int a = 0; a < 3; ++a)
                        out.block_origins[static_cast<size_t>(b - first)]
                                         [static_cast<size_t>(a)] = k[a];
                }
            }

            /* The reference profile, always -- one readout, and the only one
             * an analysis of the sampling pattern has to look inside. */
            out.central_readout = raw::pulseq_ktraj_central_readout(plan);
            if (out.central_readout >= 0 && out.central_readout < num_readouts)
            {
                const Readout& r = out.readouts[static_cast<size_t>(out.central_readout)];
                if (r.num_samples > 0)
                {
                    out.k_central.resize(static_cast<size_t>(r.num_samples) * 3);
                    raw::pulseq_ktraj_readout_k(plan, out.central_readout, out.k_central.data());
                }
            }

            const int num_exc = raw::pulseq_ktraj_num_excitations(plan);
            out.excitations.resize(static_cast<size_t>(num_exc));
            for (int i = 0; i < num_exc; ++i)
            {
                raw::pulseq_ktraj_rf rf;
                raw::pulseq_ktraj_excitation_info(plan, i, &rf);
                copy_rf(rf, out.excitations[static_cast<size_t>(i)]);
            }

            const int num_ref = raw::pulseq_ktraj_num_refocusings(plan);
            out.refocusings.resize(static_cast<size_t>(num_ref));
            for (int i = 0; i < num_ref; ++i)
            {
                raw::pulseq_ktraj_rf rf;
                raw::pulseq_ktraj_refocusing_info(plan, i, &rf);
                copy_rf(rf, out.refocusings[static_cast<size_t>(i)]);
            }

            /* Slice positions.
             *
             * offset / gradient is a length: the frequency offset that
             * selects a slice is the gradient times the distance to it.  An
             * axis with no gradient gives no information about position, and
             * Pulseq's own calculateKspacePP reports 0 there rather than a
             * NaN, so this does too. */
            out.slice_pos.resize(static_cast<size_t>(num_exc));
            out.t_slice_pos.resize(static_cast<size_t>(num_exc));
            for (int i = 0; i < num_exc; ++i)
            {
                const RfEventTiming& rf = out.excitations[static_cast<size_t>(i)];
                out.t_slice_pos[static_cast<size_t>(i)] = rf.t;
                for (int a = 0; a < 3; ++a)
                {
                    const double g = rf.g[static_cast<size_t>(a)];
                    out.slice_pos[static_cast<size_t>(i)][static_cast<size_t>(a)] =
                        (g != 0.0) ? rf.freq_offset / g : 0.0;
                }
            }
        }
        catch (...)
        {
            raw::pulseq_ktraj_plan_free(plan);
            throw;
        }
        raw::pulseq_ktraj_plan_free(plan);
        return out;
    }

    void build_dense_from_binary(const std::string& bytes, const double* shape_samples,
                                 const int* shape_offsets, int shape_count,
                                 const KSpaceOptions& options, KSpace& out)
    {
        RawFile raw_file;
        parse_binary(bytes, raw_file.get());
        restore_shapes(raw_file.get(), shape_samples, shape_offsets, shape_count);

        raw::pulseq_ktraj_opts opts;
        raw::pulseq_ktraj_opts_init(&opts);
        for (int a = 0; a < 3; ++a)
        {
            opts.trajectory_delay[a] = options.trajectory_delay[static_cast<size_t>(a)];
            opts.gradient_offset[a] = options.gradient_offset[static_cast<size_t>(a)];
        }
        opts.first_block = options.first_block;
        opts.last_block = options.last_block;
        opts.apply_rotation = options.apply_rotation ? 1 : 0;
        /* The dense grid is for drawing, and the echo search is the expensive
         * half of a plan -- there is nothing here that wants it. */
        opts.derive_center_sample = 0;

        raw::pulseq_ktraj_plan* plan = nullptr;
        const int rc = raw::pulseq_ktraj_plan_create(&plan, raw_file.get(), &opts);
        if (rc < 0)
            throw std::runtime_error("calculate_kspace: the dense pass failed (" +
                                     std::to_string(rc) + ")");

        try
        {
            const int count = raw::pulseq_ktraj_breakpoints(plan, nullptr, 0);
            if (count < 0)
                throw std::runtime_error("calculate_kspace: breakpoints failed");

            out.t_ktraj.resize(static_cast<size_t>(count));
            raw::pulseq_ktraj_breakpoints(plan, out.t_ktraj.data(), count);

            out.k_ktraj.resize(static_cast<size_t>(count) * 3);
            raw::pulseq_ktraj_at_times(plan, out.t_ktraj.data(), count, out.k_ktraj.data());
        }
        catch (...)
        {
            raw::pulseq_ktraj_plan_free(plan);
            throw;
        }
        raw::pulseq_ktraj_plan_free(plan);
    }

}  // namespace pulseq
