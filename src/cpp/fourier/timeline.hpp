/**
 * @file timeline.hpp
 * @brief The gradients a playout's blocks play, and the moments the Fourier
 *        engine's timeline reads off them.
 */

#ifndef PULSERVER_FOURIER_TIMELINE_HPP
#define PULSERVER_FOURIER_TIMELINE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fourier
{

    /**
     * Each block's gradient along each axis, linear between its corners and
     * zero outside them, and the moment of the gradients from the start of
     * the scan along the logical axes: a block that plays in the physical
     * frame, as one labelled NOROT does, is turned back by the prescription's
     * rotation.
     */
    class GradientTable
    {
      public:
        /**
         * @param time_us   corner times from their block's start, in µs
         * @param value     corner values, in Hz/m
         * @param span      (blocks, 3, 2) each block's corners along each axis
         * @param starts_us (blocks + 1) each block's start, and the scan's end
         * @param physical  (blocks) whether a block plays in the physical frame
         * @param rotation  row-major logical-to-physical rotation
         */
        GradientTable(
            std::vector<double> time_us,
            std::vector<double> value,
            std::vector<int64_t> span,
            std::vector<double> starts_us,
            std::vector<uint8_t> physical,
            const std::array<double, 9>& rotation);

        size_t blocks() const { return physical_.size(); }

        /** (blocks + 1, 3) the moment at each block's start, and the scan's end. */
        const std::vector<double>& start_moments() const { return start_moment_; }

        /** Write the moment at @p since_us into each @p block, (n, 3) in 1/m. */
        void moment(const int64_t* block, const double* since_us, size_t n, double* out) const;

        /** Write the gradient at @p since_us into each @p block, (n, 3) in Hz/m, in the
         *  frame the block plays in. */
        void value(const int64_t* block, const double* since_us, size_t n, double* out) const;

      private:
        /** Moment of axis @p axis of @p block from its start to @p since_us, in 1/m. */
        double within(size_t block, int axis, double since_us) const;

        std::vector<double> time_, value_, starts_, start_moment_;
        std::vector<int64_t> span_;
        std::vector<uint8_t> physical_;
        std::array<double, 9> rotation_;
    };

    /**
     * The moment k is measured from, and the time precession is measured from,
     * after each pulse in play order, (n + 1) with the state before the first:
     * an excitation sets both to its own, a refocusing pulse mirrors both
     * about its own, and a new file of a chain forgets them, as NaN.
     */
    void origins(
        const double* moment,
        const double* time_us,
        const int64_t* use,
        const int64_t* file,
        size_t n,
        int64_t excitation,
        int64_t refocusing,
        double* origins,
        double* precession);

} // namespace fourier

#endif /* PULSERVER_FOURIER_TIMELINE_HPP */
