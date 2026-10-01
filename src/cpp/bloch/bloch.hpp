/**
 * @file bloch.hpp
 * @brief Isochromats whose magnetisation the Bloch equation carries from one
 *        block to the next.
 *
 * The magnetisation turns about the field b = (Re b1, Im b1, bz), in Hz, in
 * the frame rotating at the reference frequency, as the magnetic moment of a
 * nucleus of positive gyromagnetic ratio precesses: dM/dt = 2 pi M x b, a
 * clockwise turn seen from the tip of b. Free precession, with a
 * piecewise-linear gradient, is integrated exactly. An RF pulse is a sequence
 * of steps holding b1 constant, each a rotation about the step's mean field
 * between two half steps of relaxation.
 */

#ifndef PULSERVER_BLOCH_BLOCH_HPP
#define PULSERVER_BLOCH_BLOCH_HPP

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <vector>

namespace bloch
{

    class GradientAreas;
    class LatticeTransform;
    class Nufft;
    class Repetitions;
    struct PulseGradient;

    /** What each isochromat is; entry i of every vector describes isochromat i. */
    struct IsochromatProperties
    {
        /** Position, in m. */
        std::vector<double> x, y, z;
        /** Equilibrium longitudinal magnetisation. */
        std::vector<double> proton_density;
        /** Relaxation times, in s; infinity for none. */
        std::vector<double> t1, t2;
        /** Precession frequency at rest, in Hz from the reference frequency. */
        std::vector<double> off_resonance;
        /** Transmit sensitivities, isochromat-major, `transmit_channels` per
         *  isochromat. None: one channel of unit sensitivity. */
        size_t transmit_channels = 0;
        std::vector<std::complex<double>> transmit;
        /** Receive sensitivities, isochromat-major, `coils` per isochromat,
         *  read once, while the isochromats are constructed, into their own
         *  coil-major layout; the caller keeps them until then. None: one
         *  coil of unit sensitivity. */
        size_t coils = 0;
        const std::complex<double>* receive = nullptr;
    };

    /**
     * A window under a changing gradient whose isochromats lie on a lattice
     * along each axis its k moves along, as the engine hands it to a device
     * that sums it onto the lattice and transforms it to the samples in its
     * place. Every array is the engine's own, valid during the call.
     */
    struct LatticeWindowRead
    {
        /** The lattice axes, a bit per axis, the points along each, the
         *  lowest axis first and fastest, and the points in all. */
        unsigned axes = 0;
        int dimensions = 0;
        const int64_t* modes = nullptr;
        size_t points = 0;
        /** The isochromats in order of their point on the lattice, each
         *  point's first position in that order, one more than the points;
         *  and, in that order, each isochromat's off-resonance, in Hz, and
         *  T2 class. */
        size_t isochromats = 0;
        const uint32_t* order = nullptr;
        const uint32_t* starts = nullptr;
        const double* off_resonance = nullptr;
        const uint32_t* decay_of = nullptr;
        /** The receive sensitivities, coil-major in the isochromats' own
         *  order; none for one coil of unit sensitivity. */
        size_t coils = 0;
        const double* receive_re = nullptr;
        const double* receive_im = nullptr;
        /** The transverse magnetisation at the first sample, in the
         *  isochromats' own order. */
        const double* mx = nullptr;
        const double* my = nullptr;
        /** The window's middle off-resonance, in Hz; its Chebyshev points,
         *  in s from the first sample; each T2 class's decay at them
         *  relative to the window's middle rate, [class][point]; and the
         *  Lagrange basis at each sample, [sample][point]. */
        double frequency = 0.0;
        size_t segments = 0;
        const double* nodes = nullptr;
        size_t decays = 0;
        const double* decay = nullptr;
        size_t samples = 0;
        const double* basis = nullptr;
        /** Each sample's coordinate along each lattice axis, in radians in
         *  [-pi, pi), lowest axis first. */
        const double* x[3] = {nullptr, nullptr, nullptr};
        /** The tolerance the transform is planned to, and whether it is
         *  computed in single precision. */
        double tolerance = 0.0;
        bool single = false;
        /** Each coil's sum over the Chebyshev points of the basis times the
         *  transform of its lattice sums, [coil][sample], zero on entry and
         *  held until the read is finished. */
        std::complex<double>* out = nullptr;
    };

    /**
     * Reads windows on the lattice in the engine's place. @c read starts
     * reading a window and returns true, or declines it and returns false;
     * the engine then takes the isochromats to the window's last sample
     * while the device reads, and calls @c finish, which returns once
     * @c out holds the window. Arrays other than @c out are read before
     * @c read returns.
     */
    struct LatticeDevice
    {
        std::function<bool(const LatticeWindowRead&)> read;
        std::function<void()> finish;
    };

    /** The events one block plays, timed in s from the block's start. */
    struct BlockEvents
    {
        double duration = 0.0;

        /** Per axis, the corners of a piecewise-linear gradient in Hz/m,
         *  zero outside them. */
        std::array<const double*, 3> gradient_times{};
        std::array<const double*, 3> gradient_values{};
        std::array<size_t, 3> gradient_corners{};

        /**
         * The transverse field of an RF pulse, in Hz: `rf_channels` rows of
         * `rf_steps` values, channel-major, each held for `rf_step` from
         * `rf_start` on. Without transmit sensitivities the channels are
         * summed; with them, the channels are theirs.
         */
        double rf_start = 0.0;
        double rf_step = 0.0;
        size_t rf_steps = 0;
        size_t rf_channels = 0;
        const std::complex<double>* rf = nullptr;

        /** Increasing ADC sample times, none inside the RF pulse. */
        const double* adc_times = nullptr;
        size_t adc_samples = 0;
    };

    /**
     * A set of isochromats and their magnetisation.
     *
     * Free precession is applied when the magnetisation is next needed -- by
     * an RF pulse, an ADC sample or a read -- so blocks without either cost
     * nothing per isochromat. A pulse that differs from an earlier one by its
     * phase alone, under the same gradient, applies the earlier pulse's maps
     * turned about z by that phase, which is exact. A pulse played under no
     * gradient or one held throughout is computed on a grid of the field an
     * isochromat sees during it, where that costs fewer maps than one per
     * group, and interpolated: without transmit sensitivities, or with every
     * channel playing one waveform times a weight of its own, so that an
     * isochromat's transmit field is that waveform times one complex drive,
     * whose magnitude is a second axis of the grid and whose phase turns the
     * map about z. An ADC window under a gradient held throughout it is read
     * by a non-uniform FFT of each T2's isochromats, to within about 1e-13 of
     * the sum of their transverse magnetisations' magnitudes, where that
     * costs less than turning every isochromat at every sample. A window
     * under any other gradient is read by FINUFFT, where the host has handed
     * it over (use_finufft()), if its k moves along axes on which the
     * isochromats lie on a lattice and that costs less: each isochromat's
     * value is summed onto its lattice point for each coil and each of a few
     * Chebyshev points across the window, between which its decay and
     * precession are interpolated, and the lattice is transformed to each
     * sample's k, to within about 1e-11 of the sum of the magnitudes of the
     * terms each sample sums. Either transform holds to a tolerance instead
     * where play() is given one. An ADC
     * sample's coil sums are formed in partial sums over blocks of coils, with
     * AVX2 and FMA or AVX-512 where the processor has them, so their rounding
     * depends on the processor and the number of threads.
     */
    class Isochromats
    {
    public:
        /**
         * @param properties  What each isochromat is; every vector as long as
         *                    the positions, or empty for its default.
         * @param threads     Worker threads; 0 for every core.
         * @throws std::invalid_argument on vectors of the wrong length.
         */
        explicit Isochromats(IsochromatProperties properties, size_t threads = 0);
        ~Isochromats();
        Isochromats(const Isochromats&) = delete;
        Isochromats& operator=(const Isochromats&) = delete;

        size_t size() const
        {
            return count_;
        }
        size_t coils() const
        {
            return coils_;
        }
        size_t transmit_channels() const
        {
            return properties_.transmit_channels;
        }

        /** The time played since construction or the last reset, in s. */
        double elapsed() const
        {
            const std::lock_guard<std::mutex> held(mutex_);
            return elapsed_;
        }

        /** ADC windows read on a lattice since construction. */
        size_t lattice_windows() const
        {
            const std::lock_guard<std::mutex> held(mutex_);
            return lattice_windows_;
        }

        /** Of those, the windows a device read. */
        size_t device_windows() const
        {
            const std::lock_guard<std::mutex> held(mutex_);
            return device_windows_;
        }

        /** Offer every window read on the lattice to @p device before the
         *  engine reads it; an empty one offers none. */
        void use_lattice_device(LatticeDevice device);

        /** Put every isochromat at equilibrium, along +z, and the clock at zero. */
        void reset();

        /** Write the magnetisation, (size(), 3) row-major, to @p into. */
        void magnetization(double* into);

        /** Replace the magnetisation with (size(), 3) row-major @p from. */
        void set_magnetization(const double* from);

        /**
         * Play @p block, writing what each coil receives at each ADC sample to
         * @p signal, coil-major: signal[c * adc_samples + k], the sum over the
         * isochromats of the receive sensitivity times Mx + i My.
         *
         * An ADC window read by a transform is read to within @p tolerance of
         * the sum of the magnitudes of the terms each sample sums, where it is
         * above zero.
         *
         * @throws std::invalid_argument on events the engine cannot play: an
         *         ADC sample inside the RF pulse, times out of order, or RF
         *         channels that differ in number from the transmit
         *         sensitivities.
         */
        void play(const BlockEvents& block, std::complex<double>* signal, double tolerance = 0.0);

    private:
        friend class Repetitions;

        /** Isochromats that see one field during an RF pulse. */
        struct Grouping
        {
            int mode = 0;
            double direction[3] = {0.0, 0.0, 0.0};
            std::vector<uint32_t> group_of;
            std::vector<uint32_t> representative;
        };

        /** A pulse's affine maps, 12 per group of its grouping, kept for
         *  later pulses that differ from it by a phase alone. */
        struct HeldPulse
        {
            int mode = 0;
            double direction[3] = {0.0, 0.0, 0.0};
            double step = 0.0;
            size_t channels = 0;
            /** Per step, the gradient area in 1/m. */
            std::vector<double> delta;
            /** The transverse field, channel-major, in Hz. */
            std::vector<std::complex<double>> rf;
            std::vector<double> maps;

            size_t bytes() const
            {
                return delta.size() * sizeof(double) + rf.size() * sizeof(std::complex<double>) +
                    maps.size() * sizeof(double);
            }
        };

        void check();
        void lay_out_receive();
        void classify();
        void flush();
        void advance(const GradientAreas& areas, double& now, double to);
        void excite(const BlockEvents& block, const GradientAreas& areas);
        /** The held pulse @p block is under @p gradient but for a phase,
         *  written to @p turn, or held_.end(). */
        std::list<HeldPulse>::iterator find_held(
            const BlockEvents& block, const PulseGradient& gradient, std::complex<double>& turn);
        /** A held pulse of @p block under @p gradient, its maps not yet filled. */
        HeldPulse held_pulse(const BlockEvents& block, const PulseGradient& gradient, const Grouping& groups) const;
        /** Hold @p made, the most recently played, letting the least recent
         *  others go beyond kHeldPulses or kHeldBytes. */
        void keep_held(HeldPulse made);
        /** Compute the pulse's map for each group of @p groups, and hold them. */
        void hold(const BlockEvents& block, const PulseGradient& gradient, const Grouping& groups);
        /** Apply @p maps, one per group of @p groups, turned about z by
         *  @p turn: conj(turn) before them and turn after. */
        void apply(const std::vector<double>& maps, const Grouping& groups, std::complex<double> turn);
        /** Apply the affine map @p map to isochromat @p i, turned about z by
         *  @p before before it and @p after after it. */
        void apply(size_t i, const double* map, std::complex<double> before, std::complex<double> after);
        /** Play @p block without reading its ADC samples, writing each
         *  isochromat's Mx + i My at the first of them to @p first_sample
         *  where it has any; they must all lie on one side of its pulse. */
        void play_quietly(const BlockEvents& block, std::complex<double>* first_sample);
        /** The gradient area, in 1/m, and the time, in s, from each of
         *  @p block's ADC samples to the next, into @p area and @p step;
         *  whether they are one increment throughout. */
        bool window_steps(const BlockEvents& block, double area[3], double& step) const;
        void acquire(
            const BlockEvents& block,
            const GradientAreas& areas,
            size_t first,
            size_t last,
            std::complex<double>* signal,
            double tolerance);
        /** Read @p samples samples, @p area in 1/m and @p step in s apart,
         *  @p span in s from first to last, into signal[c * stride + k] by the
         *  non-uniform FFT of kernel @p width, unless reading them one by one
         *  costs less; return whether it did. */
        bool read_transformed(
            const double area[3],
            double step,
            size_t samples,
            double span,
            size_t width,
            std::complex<double>* signal,
            size_t stride);
        /** Whether the transform of kernel @p width reads a window of
         *  @p samples samples for less than reading it sample by sample,
         *  within its memory. */
        bool transform_pays(size_t samples, size_t width) const;
        /** Spread every isochromat's term onto a grid per T2 and coil, the
         *  coils of a point together, and leave it as it stands at the
         *  window's last sample; return the grids. */
        std::vector<std::complex<double>> spread_window(
            const Nufft& transform, const double area[3], double step, double span);
        /** Transform the @p spread grids and write each coil's samples, the
         *  sum over T2s of each T2's decay times its transform. */
        void finish_window(
            const Nufft& transform,
            double step,
            const std::vector<std::complex<double>>& spread,
            std::complex<double>* signal,
            size_t stride);
        /** The transform of windows of @p samples samples by a kernel
         *  @p width grid points wide. */
        const Nufft& window_transform(size_t samples, size_t width);

        /** The isochromats' positions along one axis as whole multiples of a
         *  spacing from the least of them, where they are. */
        struct Lattice
        {
            /** 0 where the positions are not such multiples. */
            double spacing = 0.0;
            double origin = 0.0;
            size_t points = 0;
            /** The largest distance of a position from its point, in m. */
            double deviation = 0.0;
            /** Each isochromat's point. */
            std::vector<uint32_t> index;
        };
        /** The isochromats in order of their point on the lattice of the
         *  axes in @p axes, a bit per axis, the lowest axis's index fastest,
         *  where each point's start in that order, one more than the
         *  points, and the most isochromats at one point; and, in that
         *  order, each isochromat's off-resonance and T2 and, from the first
         *  window summed in single precision on, its sensitivities, the
         *  coils of an isochromat together. */
        struct LatticeOrder
        {
            unsigned axes = 0;
            std::vector<uint32_t> order;
            std::vector<uint32_t> starts;
            size_t fullest = 0;
            std::vector<double> off_resonance;
            std::vector<uint32_t> decay_of;
            std::vector<std::complex<float>> receive;
        };
        /** The lattice of the positions along @p axis, found on first use. */
        const Lattice& lattice(int axis);
        LatticeOrder& lattice_order(unsigned axes);
        /** Chebyshev points across a window of @p span s at which each
         *  isochromat's decay and precession, interpolated between them,
         *  hold to within @p error of their magnitude throughout the window;
         *  0 where more than kMostSegments would be needed. */
        size_t segments_for(double span, double error);
        /**
         * Read the window whose samples are @p area in 1/m and @p time in s
         * after the one before them, three and one per sample, into
         * signal[c * stride + k] by FINUFFT from the lattice of the axes its k
         * moves along, to within @p error of the sum of the magnitudes of the
         * terms each sample sums, and leave the isochromats as they stand at
         * its last sample; unless FINUFFT is not ready, the isochromats lie on
         * no such lattice, or reading it sample by sample costs less. Return
         * whether it did.
         */
        bool read_on_lattice(
            const std::vector<double>& area,
            const std::vector<double>& time,
            size_t samples,
            double error,
            std::complex<double>* signal,
            size_t stride);

        /** A window read on a lattice. */
        struct LatticeWindow
        {
            /** Each sample's k from the first, in 1/m, three per sample, and
             *  time from it, in s. */
            std::vector<double> k;
            std::vector<double> time;
            double span = 0.0;
            /** The axes the window's k moves along, the lowest first, and
             *  their lattice's points along each. */
            int along[3] = {0, 0, 0};
            int axes = 0;
            unsigned mask = 0;
            int64_t modes[3] = {1, 1, 1};
            size_t points = 1;
            /** Per sample, the phase in cycles, over -2 pi, of the lattice's
             *  centre along the axes the window's k moves along and of the
             *  positions' middle along the others. */
            std::vector<double> centre;
            /** The middle of the isochromats' rates: 1/T2, in 1/s, and
             *  off-resonance, in Hz. */
            double rate = 0.0;
            double frequency = 0.0;
        };
        /** Chebyshev points across a window, and the Lagrange basis at each
         *  sample, [sample][point]. */
        struct Segments
        {
            size_t count = 0;
            std::vector<double> nodes;
            std::vector<double> basis;
        };
        /** Fill @p window for the samples @p area and @p time apart; false
         *  where its k moves along no axis or along one the isochromats do not
         *  lie on a lattice of, to within @p error of the sum of the
         *  magnitudes of the terms. */
        bool lattice_window(
            const std::vector<double>& area,
            const std::vector<double>& time,
            size_t samples,
            double error,
            LatticeWindow& window);
        /** Sum each isochromat's value onto its lattice point, for each of
         *  @p coils coils from @p first_coil on and each of the Chebyshev
         *  points, into @p sums, [coil][point][lattice point]. */
        template <typename Real>
        void sum_onto_lattice(
            const LatticeWindow& window,
            const Segments& segments,
            size_t first_coil,
            size_t coils,
            std::complex<Real>* sums);
        /** Add each coil's sums at the samples of @p window, at the points
         *  @p x along its axes in radians, to @p out, [coil][sample], by
         *  lattice transforms in @p Real to within @p tolerance of the sum of
         *  the magnitudes of the modes, as many coils at once as fit. */
        template <typename Real>
        void transform_lattice(
            const LatticeWindow& window,
            const Segments& segments,
            const std::vector<std::vector<double>>& x,
            double tolerance,
            std::vector<std::complex<double>>& out);
        /** Hand the window to the lattice device; whether it read it. */
        bool read_on_device(
            const LatticeWindow& window,
            const Segments& segments,
            const std::vector<std::vector<double>>& x,
            double tolerance,
            std::vector<std::complex<double>>& out);
        /** Worker @p worker's lattice transform of @p modes along @p axes
         *  axes, of @p vectors vectors at once, to within @p tolerance. */
        LatticeTransform& lattice_transform(
            int axes, const int64_t modes[3], int vectors, double tolerance, size_t worker);
        /** Transform @p vectors vectors of @p modes, one after another, to
         *  the samples of @p window at the points @p x, into @p sums, a
         *  share of the vectors per worker. */
        template <typename Real>
        void transform_vectors(
            const LatticeWindow& window,
            const std::vector<std::vector<double>>& x,
            double tolerance,
            size_t vectors,
            std::complex<Real>* modes,
            std::complex<Real>* sums);
        /** Leave each isochromat as it stands at the last sample of
         *  @p window. */
        void settle_after(const LatticeWindow& window);
        const Grouping& grouping(int mode, const double direction[3]);

        IsochromatProperties properties_;
        size_t count_ = 0;
        size_t coils_ = 1;
        size_t threads_ = 1;

        std::vector<double> mx_, my_, mz_;
        /** Receive sensitivities, coil-major: entry c * size() + i. */
        std::vector<double> receive_re_, receive_im_;
        /** Classes of isochromats equal in everything but position. */
        std::vector<uint32_t> class_of_;

        /** Free precession not yet applied: gradient area in 1/m, and time. */
        double pending_area_[3] = {0.0, 0.0, 0.0};
        double pending_time_ = 0.0;
        double elapsed_ = 0.0;

        std::vector<Grouping> groupings_;
        /** Pulses held for reuse, the most recently played last. */
        std::list<HeldPulse> held_;
        size_t held_bytes_ = 0;

        /** A pulse's affine maps on a grid of the field an isochromat of one
         *  T1 and T2 sees throughout it, and of the magnitude of the drive it
         *  sees the pulse's waveform at, kept for later pulses that differ
         *  from it by a phase alone. Point (row, column) lies at a field of
         *  column * spacing, in Hz, and a drive of row * drive_spacing, or of
         *  one where drive_spacing is 0 and row is 0. */
        struct PulseTable
        {
            double step = 0.0;
            double t1 = 0.0;
            double t2 = 0.0;
            double spacing = 0.0;
            double drive_spacing = 0.0;
            /** The waveform, in Hz per unit of drive. */
            std::vector<std::complex<double>> rf;
            /** The first row and column held, and how many, then 12 values
             *  per point, row by row, in the frame the isochromat's own
             *  precession over half the pulse turns on either side. */
            long long first_row = 0;
            long long first = 0;
            long long rows = 0;
            long long columns = 0;
            std::vector<double> maps;

            size_t bytes() const
            {
                return rf.size() * sizeof(std::complex<double>) + maps.size() * sizeof(double);
            }
        };

        /** Hold the pulse's map for each group of @p groups from its tables,
         *  made or extended as needed, under a gradient held at @p along, in
         *  Hz/m, unless their new points outnumber the groups; return whether
         *  it did. */
        bool hold_on_grid(const BlockEvents& block, const PulseGradient& gradient, double along, const Grouping& groups);
        /** The table of @p rf, @p step, @p spacing and @p drive_spacing for
         *  the class relaxing as @p relaxation, but for a phase written to
         *  @p turn, or tables_.end(). */
        std::list<PulseTable>::iterator find_table(
            const std::array<double, 2>& relaxation,
            double step,
            double spacing,
            double drive_spacing,
            const std::vector<std::complex<double>>& rf,
            std::complex<double>& turn);
        /** Compute the points of rows @p bounds[0] to @p bounds[1] and
         *  columns @p bounds[2] to @p bounds[3] that @p table lacks, holding
         *  the smallest rectangle of points that covers both. */
        void extend(PulseTable& table, const std::array<long long, 4>& bounds);
        /** Write the map of each group, of class @p class_of, from its
         *  class's @p table at its @p field and the magnitude of its
         *  @p drive, turned by the class's @p turn and the drive's phase, to
         *  @p maps; without a drive, at a drive of one. */
        void table_maps(
            const std::vector<double>& field,
            const std::vector<std::complex<double>>& drive,
            const std::vector<uint32_t>& class_of,
            const std::vector<std::list<PulseTable>::iterator>& table,
            const std::vector<std::complex<double>>& turn,
            double duration,
            std::vector<double>& maps);
        /** Make the @p played tables the most recent, and let the least
         *  recent others go beyond kPulseTables or kTableBytes. */
        void keep(const std::vector<std::list<PulseTable>::iterator>& played);

        /** Classes of isochromats equal in T1 and T2, and each class's. */
        std::vector<uint32_t> relaxation_of_;
        std::vector<std::array<double, 2>> relaxations_;
        /** Tables held for reuse, the most recently played last. */
        std::list<PulseTable> tables_;
        size_t table_bytes_ = 0;

        /** Classes of isochromats equal in T2, and each class's rate, in 1/s. */
        std::vector<uint32_t> decay_of_;
        std::vector<double> decays_;
        /** Transforms of ADC windows, the most recently read last. */
        std::list<std::unique_ptr<Nufft>> transforms_;

        std::array<std::unique_ptr<Lattice>, 3> lattices_;
        std::vector<std::unique_ptr<LatticeOrder>> lattice_orders_;
        /** Lattice transforms, each a worker's, the most recently used
         *  last. */
        struct LatticeTransformHeld
        {
            size_t worker;
            std::unique_ptr<LatticeTransform> plan;
        };
        std::list<LatticeTransformHeld> lattice_transforms_;
        /** Chebyshev points found for a window's span and error. */
        struct Segmentation
        {
            double span;
            double error;
            size_t count;
        };
        std::vector<Segmentation> segmentations_;
        size_t lattice_windows_ = 0;
        size_t device_windows_ = 0;
        LatticeDevice lattice_device_;
        /** The lattice's sums and their transforms at the samples, in
         *  either precision, kept from one window to the next. */
        std::vector<std::complex<double>> lattice_sums_, lattice_values_;
        std::vector<std::complex<float>> single_sums_, single_values_;

        /** Held by every call that reads or changes the magnetisation. */
        mutable std::mutex mutex_;
    };

} // namespace bloch

#endif /* PULSERVER_BLOCH_BLOCH_HPP */
