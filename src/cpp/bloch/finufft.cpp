/**
 * @file finufft.cpp
 * @brief FINUFFT's plan interface through the entry points the host hands
 *        over.  See finufft.hpp.
 */

#include "bloch/finufft.hpp"

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

namespace bloch
{

    namespace
    {

        /** The plan entry points of one precision, whose tolerance is of
         *  that precision too. */
        template <typename Real>
        struct Precision
        {
            using MakePlan = int (*)(int, int, int64_t*, int, int, Real, void**, void*);
            using SetPoints = int (*)(void*, int64_t, Real*, Real*, Real*, int64_t, Real*, Real*, Real*);
            using Execute = int (*)(void*, std::complex<Real>*, std::complex<Real>*);
            using Destroy = int (*)(void*);

            MakePlan makeplan = nullptr;
            SetPoints setpts = nullptr;
            Execute execute = nullptr;
            Destroy destroy = nullptr;

            void take(const FinufftPlans& given)
            {
                makeplan = reinterpret_cast<MakePlan>(given.makeplan);
                setpts = reinterpret_cast<SetPoints>(given.setpts);
                execute = reinterpret_cast<Execute>(given.execute);
                destroy = reinterpret_cast<Destroy>(given.destroy);
            }
        };

        using DefaultOptions = void (*)(void*);

        /** FINUFFT's return codes above this are errors; this one warns that
         *  the tolerance asked for is below what it reaches. */
        constexpr int kWarnings = 1;

        /** The least tolerance a single-precision plan is made for: single
         *  precision's own rounding stays below a fifth of it, relative to the
         *  sum of the magnitudes of the modes. */
        constexpr double kSinglePrecision = 2e-6;

        /** FFTW's planning flags: FFTW_MEASURE, FFTW_ESTIMATE. */
        constexpr int kMeasure = 0;
        constexpr int kEstimate = 1 << 6;

        struct Table
        {
            Precision<double> doubles;
            Precision<float> singles;
            DefaultOptions default_opts = nullptr;
            FinufftOptions options;
        };

        Table table;
        std::mutex table_mutex;

        Table held_table()
        {
            const std::lock_guard<std::mutex> held(table_mutex);
            return table;
        }

        /** The upsampling of FINUFFT's grid: a quarter over where its kernel
         *  reaches the tolerance so, twice otherwise. */
        double upsampling_for(double tolerance)
        {
            return tolerance >= 1e-9 ? 1.25 : 2.0;
        }

        bool complete(const FinufftPlans& plans)
        {
            return plans.makeplan != nullptr && plans.setpts != nullptr && plans.execute != nullptr &&
                plans.destroy != nullptr;
        }

        bool inside(size_t at, size_t bytes, size_t size)
        {
            return at + bytes <= size;
        }

        void check(int status, const char* what)
        {
            if (status > kWarnings)
                throw std::runtime_error(std::string("FINUFFT refused to ") + what + ": status " + std::to_string(status));
        }

    } // namespace

    bool use_finufft(
        const FinufftPlans& doubles,
        const FinufftPlans& singles,
        void* default_opts,
        const FinufftOptions& options)
    {
        if (!complete(doubles) || !complete(singles) || default_opts == nullptr)
            return false;
        if (options.size < 16 || options.size > 4096 || !inside(options.threads_at, sizeof(int), options.size) ||
            !inside(options.fftw_at, sizeof(int), options.size) ||
            !inside(options.upsampling_at, sizeof(double), options.size) ||
            !inside(options.warnings_at, sizeof(int), options.size))
            return false;
        const std::lock_guard<std::mutex> held(table_mutex);
        table.doubles.take(doubles);
        table.singles.take(singles);
        table.default_opts = reinterpret_cast<DefaultOptions>(default_opts);
        table.options = options;
        return true;
    }

    bool finufft_ready()
    {
        const std::lock_guard<std::mutex> held(table_mutex);
        return table.doubles.makeplan != nullptr;
    }

    bool LatticeTransform::single_for(double tolerance)
    {
        return tolerance >= kSinglePrecision;
    }

    LatticeTransform::LatticeTransform(
        int dimensions, const int64_t modes[3], int vectors, double tolerance, int threads)
        : dimensions_(dimensions), modes_{modes[0], dimensions > 1 ? modes[1] : 1, dimensions > 2 ? modes[2] : 1},
          vectors_(vectors), tolerance_(tolerance), single_(single_for(tolerance))
    {
        if (dimensions < 1 || dimensions > 3 || vectors < 1)
            throw std::invalid_argument("a lattice transform takes 1 to 3 dimensions and one vector or more");
        const Table held = held_table();
        if (held.default_opts == nullptr)
            throw std::runtime_error("FINUFFT has not been handed to the engine");
        std::vector<unsigned char> options(held.options.size, 0);
        held.default_opts(options.data());
        const int quiet = 0;
        const int64_t grid = modes_[0] * modes_[1] * modes_[2];
        const int fftw = grid <= kMeasuredModes && grid * vectors_ >= kMeasuredWork ? kMeasure : kEstimate;
        const double upsampling = upsampling_for(tolerance);
        std::memcpy(options.data() + held.options.threads_at, &threads, sizeof(int));
        std::memcpy(options.data() + held.options.fftw_at, &fftw, sizeof(int));
        std::memcpy(options.data() + held.options.upsampling_at, &upsampling, sizeof(double));
        std::memcpy(options.data() + held.options.warnings_at, &quiet, sizeof(int));
        /* Type 2, exp(-i m . x). */
        const int made = single_
            ? held.singles.makeplan(2, dimensions_, modes_, -1, vectors_, static_cast<float>(tolerance_), &plan_, options.data())
            : held.doubles.makeplan(2, dimensions_, modes_, -1, vectors_, tolerance_, &plan_, options.data());
        check(made, "plan");
    }

    LatticeTransform::~LatticeTransform()
    {
        if (plan_ == nullptr)
            return;
        const Table held = held_table();
        if (single_)
            held.singles.destroy(plan_);
        else
            held.doubles.destroy(plan_);
    }

    void LatticeTransform::points(int64_t count, const double* x, const double* y, const double* z)
    {
        const double* given[3] = {x, y, z};
        const Table held = held_table();
        if (single_)
        {
            float* at[3] = {nullptr, nullptr, nullptr};
            for (int d = 0; d < dimensions_; ++d)
            {
                single_coordinates_[d].assign(given[d], given[d] + count);
                at[d] = single_coordinates_[d].data();
            }
            check(held.singles.setpts(plan_, count, at[0], at[1], at[2], 0, nullptr, nullptr, nullptr), "take the points");
            return;
        }
        double* at[3] = {nullptr, nullptr, nullptr};
        for (int d = 0; d < dimensions_; ++d)
        {
            coordinates_[d].assign(given[d], given[d] + count);
            at[d] = coordinates_[d].data();
        }
        check(held.doubles.setpts(plan_, count, at[0], at[1], at[2], 0, nullptr, nullptr, nullptr), "take the points");
    }

    /* Type 2 reads the modes, FINUFFT's f, and writes the sums, its c. */

    void LatticeTransform::execute(std::complex<double>* modes, std::complex<double>* sums)
    {
        if (single_)
            throw std::logic_error("a single-precision lattice transform takes single-precision modes");
        check(held_table().doubles.execute(plan_, sums, modes), "transform");
    }

    void LatticeTransform::execute(std::complex<float>* modes, std::complex<float>* sums)
    {
        if (!single_)
            throw std::logic_error("a double-precision lattice transform takes double-precision modes");
        check(held_table().singles.execute(plan_, sums, modes), "transform");
    }

} // namespace bloch
