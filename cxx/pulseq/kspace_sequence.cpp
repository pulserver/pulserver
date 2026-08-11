/**
 * @file kspace_sequence.cpp
 * @brief The Sequence half of calculate_kspace, and the MRD layout helpers.
 *
 * Split from kspace.cpp for a mechanical reason: that file includes
 * raw64.hpp, which forbids the ordinary pulseq headers in the same
 * translation unit because they share include guards.  This one includes
 * sequence.hpp and knows nothing about raw64; the two meet across one
 * function taking a std::string of bytes.
 */

#include "pulseq/kspace.hpp"

#include "pulseq/sequence.hpp"
#include "pulseq/write.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace pulseq
{

    /* Both defined in kspace.cpp, which owns the raw64 side.  The shapes
     * cross as plain doubles because the binary format narrows them to float32
     * on the way; see the file comment on kspace.hpp. */
    KSpace kspace_from_binary(const std::string& bytes, const double* shape_samples,
                              const int* shape_offsets, int shape_count,
                              const KSpaceOptions& options);

    /**
     * The dense trajectory, on the breakpoint grid.  Also in kspace.cpp,
     * because it needs the core.
     *
     * Deliberately not on the gradient raster: between two breakpoints k is
     * one quadratic, so the breakpoints are the smallest grid that loses
     * nothing structural.  A five-millisecond flat top costs two points
     * rather than five hundred, and a ramp costs two either way -- which is
     * what keeps this affordable on a scan long enough for the dense
     * trajectory to be a question at all.
     */
    void build_dense_from_binary(const std::string& bytes, const double* shape_samples,
                                 const int* shape_offsets, int shape_count,
                                 const KSpaceOptions& options, KSpace& out);

    KSpace calculate_kspace(Sequence& seq, const KSpaceOptions& options)
    {
        /* Non-const because write_binary stamps TotalDuration first; that is
         * upstream's behaviour and the reason the signature says so. */
        const std::string bytes = write_binary(seq);

        /* The compressed samples exactly as the sequence holds them, flattened
         * with a one-past-the-end offset so a shape's length is a subtraction. */
        const int count = seq.shape_library().size();
        std::vector<int> offsets(static_cast<size_t>(count) + 1, 0);
        for (int id = 1; id <= count; ++id)
            offsets[static_cast<size_t>(id)] =
                offsets[static_cast<size_t>(id) - 1] + seq.shape_library().num_compressed(id);

        std::vector<double> samples(static_cast<size_t>(offsets[static_cast<size_t>(count)]));
        for (int id = 1; id <= count; ++id)
        {
            const double* src = seq.shape_library().samples(id);
            const int n = seq.shape_library().num_compressed(id);
            for (int j = 0; j < n; ++j)
                samples[static_cast<size_t>(offsets[static_cast<size_t>(id) - 1] + j)] = src[j];
        }

        KSpace out = kspace_from_binary(bytes, samples.data(), offsets.data(), count, options);
        if (options.dense)
            build_dense_from_binary(bytes, samples.data(), offsets.data(), count, options, out);
        return out;
    }

    std::array<bool, 3> active_axes(const KSpace& ks, double tolerance)
    {
        std::array<bool, 3> active{{false, false, false}};

        for (const Readout& r : ks.readouts)
        {
            if (r.num_samples <= 0)
                continue;
            for (int a = 0; a < 3; ++a)
            {
                if (active[static_cast<size_t>(a)])
                    continue;
                const double* k = ks.k_adc.data() +
                                  static_cast<size_t>(r.sample_offset) * 3 +
                                  static_cast<size_t>(a) * static_cast<size_t>(r.num_samples);
                for (int j = 0; j < r.num_samples; ++j)
                {
                    if (std::fabs(k[j]) > tolerance)
                    {
                        active[static_cast<size_t>(a)] = true;
                        break;
                    }
                }
            }
        }
        return active;
    }

    int interleave_readout(const KSpace& ks, const std::array<bool, 3>& axes, int readout,
                           float* out)
    {
        if (readout < 0 || readout >= static_cast<int>(ks.readouts.size()) || out == nullptr)
            return 0;

        const Readout& r = ks.readouts[static_cast<size_t>(readout)];
        if (r.num_samples <= 0)
            return 0;

        int ndim = 0;
        for (int a = 0; a < 3; ++a)
            ndim += axes[static_cast<size_t>(a)] ? 1 : 0;
        if (ndim == 0)
            return 0;

        const double* base = ks.k_adc.data() + static_cast<size_t>(r.sample_offset) * 3;
        int written = 0;
        for (int j = 0; j < r.num_samples; ++j)
        {
            for (int a = 0; a < 3; ++a)
            {
                if (!axes[static_cast<size_t>(a)])
                    continue;
                out[written++] = static_cast<float>(
                    base[static_cast<size_t>(a) * static_cast<size_t>(r.num_samples) +
                         static_cast<size_t>(j)]);
            }
        }
        return written;
    }

}  // namespace pulseq
