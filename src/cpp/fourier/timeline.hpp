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

        /** When @p block starts, in µs from the start of the scan. */
        double start_of(size_t block) const { return starts_[block]; }

        /** (blocks + 1, 3) the moment at each block's start, and the scan's end. */
        const std::vector<double>& start_moments() const { return start_moment_; }

        /** Write the moment at @p since_us into each @p block, (n, 3) in 1/m. */
        void moment(const int64_t* block, const double* since_us, size_t n, double* out) const;

        /** Write the moment at @p since_us into @p block, in 1/m. */
        void moment_at(size_t block, double since_us, double out[3]) const;

        /** Write the gradient at @p since_us into each @p block, (n, 3) in Hz/m, in the
         *  frame the block plays in. */
        void value(const int64_t* block, const double* since_us, size_t n, double* out) const;

        /**
         * Write into @p out the b-value at each of @p echo_us, in s/m²:
         * (2 pi)² times the integral of |k|² from the last excitation, k
         * measured from the origin each pulse leaves, (p + 1, 3) as origins()
         * writes them. Zero where no excitation precedes the echo in its file.
         */
        void b_values(
            const double* pulse_us,
            const uint8_t* excites,
            const double* origins,
            size_t pulses,
            const double* echo_us,
            size_t n,
            double* out) const;

      private:
        /** Gradient of axis @p axis of @p block at @p since_us, strictly between corners, in Hz/m. */
        double gradient_at(size_t block, int axis, double since_us) const;

        /** Moment of axis @p axis of @p block from its start to @p since_us, in 1/m. */
        double within(size_t block, int axis, double since_us) const;

        std::vector<double> time_, value_, starts_, start_moment_;
        std::vector<int64_t> span_;
        std::vector<uint8_t> physical_;
        std::array<double, 9> rotation_;
    };

    /** What a timeline knows of its pulses when it reads its readouts. */
    struct Pulses
    {
        const double* time_us;      /**< (p) in play order */
        const int64_t* file;        /**< (p) the file of a chain each plays in */
        const uint8_t* refocusing;  /**< (p) whether each refocuses */
        const double* interval;     /**< (p, 3) the moment from each to the next */
        const double* origins;      /**< (p + 1, 3) k's origin after each, NaN before any */
        size_t count;
    };

    /** What a readout's samples are read for: its block and file, how many
     *  samples it takes, their spacing and the first one's start, in µs. */
    struct Readout
    {
        int64_t block;
        int64_t file;
        int64_t samples;
        double dwell_us;
        double delay_us;
    };

    /** What is read off a readout: the sample nearest the centre of k-space
     *  and its time, the widest |k| along each axis, the pathway it reads and
     *  the moment the interval it lies in winds. */
    struct Reading
    {
        int64_t echo;
        double echo_us;
        double reach[3];
        int64_t pathway;
        double winding[3];
    };

    /**
     * Read each readout off @p coarse of its samples, evenly spaced, and its
     * last: the pathway is the one of @p pathways whose k, shifted by its
     * multiple of the winding, passes nearest the centre, a later one only by
     * half the winding; the echo is the nearest sample within a step of the
     * nearest of those.
     */
    void read(
        const GradientTable& table,
        const Pulses& pulses,
        const Readout* readouts,
        size_t count,
        int64_t coarse,
        int64_t pathways,
        Reading* out);

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
