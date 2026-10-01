/**
 * @file finufft.hpp
 * @brief FINUFFT's plan interface, reached through the entry points of the
 *        finufft wheel's library, which the host hands over.
 */

#ifndef PULSERVER_BLOCH_FINUFFT_HPP
#define PULSERVER_BLOCH_FINUFFT_HPP

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace bloch
{

    /** FINUFFT's plan entry points in one precision: its makeplan, setpts,
     *  execute and destroy. */
    struct FinufftPlans
    {
        void* makeplan = nullptr;
        void* setpts = nullptr;
        void* execute = nullptr;
        void* destroy = nullptr;
    };

    /** The size of FINUFFT's options struct and the byte offsets of the
     *  fields set here: the thread count, the FFTW planning flags, the
     *  upsampling and the switch of its warnings. */
    struct FinufftOptions
    {
        size_t size = 0;
        size_t threads_at = 0;
        size_t fftw_at = 0;
        size_t upsampling_at = 0;
        size_t warnings_at = 0;
    };

    /**
     * Take FINUFFT's entry points -- the finufft_ plans in double precision,
     * the finufftf_ plans in single, and finufft_default_opts -- and the
     * layout of its options, read by the host from the package the library
     * came from, so that a release that moves a field is read where it is.
     * Return whether they were taken: every entry point given and every field
     * inside the struct.
     */
    bool use_finufft(
        const FinufftPlans& doubles,
        const FinufftPlans& singles,
        void* default_opts,
        const FinufftOptions& options);

    /** Whether use_finufft() has taken FINUFFT's entry points. */
    bool finufft_ready();

    /**
     * A FINUFFT plan of type 2: sums f(x) = sum_m f_m exp(-i m . x) over the
     * modes m of a grid of up to three dimensions, the first fastest and
     * each running from -modes / 2, at points x in radians, for several
     * vectors of modes at once.
     *
     * The plan works in single precision where the tolerance lies above
     * what single precision reaches, in double otherwise. FFTW plans the
     * FFTs by timing them where an execute() transforms at least
     * kMeasuredWork modes, over its vectors, on a grid of at most
     * kMeasuredModes, and by estimating their cost otherwise: timing them
     * takes about a second on a grid of 2^15 modes.
     */
    class LatticeTransform
    {
    public:
        static constexpr int64_t kMeasuredWork = int64_t(1) << 21;
        static constexpr int64_t kMeasuredModes = int64_t(1) << 18;

        /** Whether a plan for @p tolerance works in single precision. */
        static bool single_for(double tolerance);

        /**
         * @param dimensions  1, 2 or 3.
         * @param modes       Modes along each dimension.
         * @param vectors     Vectors transformed by each execute().
         * @param tolerance   FINUFFT's tolerance.
         * @param threads     Threads FINUFFT runs on; 0 for its own choice.
         * @throws std::runtime_error where FINUFFT is not ready or refuses the plan.
         */
        LatticeTransform(int dimensions, const int64_t modes[3], int vectors, double tolerance, int threads);
        ~LatticeTransform();
        LatticeTransform(const LatticeTransform&) = delete;
        LatticeTransform& operator=(const LatticeTransform&) = delete;

        int dimensions() const
        {
            return dimensions_;
        }
        int vectors() const
        {
            return vectors_;
        }
        const int64_t* modes() const
        {
            return modes_;
        }
        double tolerance() const
        {
            return tolerance_;
        }
        /** Whether execute() takes and writes single-precision values. */
        bool single() const
        {
            return single_;
        }

        /** Sum at the @p count points whose coordinates along each dimension
         *  are @p x, @p y and @p z, those beyond dimensions() unread. */
        void points(int64_t count, const double* x, const double* y, const double* z);

        /** Write the sums of each of vectors() vectors of modes, @p modes
         *  vector-major, at each point to @p sums, vector-major, in the
         *  plan's precision.
         *  @throws std::logic_error in the other precision. */
        void execute(std::complex<double>* modes, std::complex<double>* sums);
        void execute(std::complex<float>* modes, std::complex<float>* sums);

    private:
        int dimensions_;
        int64_t modes_[3];
        int vectors_;
        double tolerance_;
        bool single_;
        void* plan_ = nullptr;
        /** The points' coordinates, which FINUFFT reads at every execute(). */
        std::vector<double> coordinates_[3];
        std::vector<float> single_coordinates_[3];
    };

} // namespace bloch

#endif /* PULSERVER_BLOCH_FINUFFT_HPP */
