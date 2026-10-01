/**
 * @file repetitions.hpp
 * @brief Repetitions of a sequence of blocks, played on isochromats from the
 *        affine map one repetition applies to each isochromat.
 */

#ifndef PULSERVER_BLOCH_REPETITIONS_HPP
#define PULSERVER_BLOCH_REPETITIONS_HPP

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "bloch/bloch.hpp"
#include "bloch/nufft.hpp"

namespace bloch
{

    /** One block's events, owning what BlockEvents points to. */
    struct OwnedBlock
    {
        double duration = 0.0;
        std::vector<double> gradient_times[3];
        std::vector<double> gradient_values[3];
        double rf_start = 0.0;
        double rf_step = 0.0;
        size_t rf_steps = 0;
        size_t rf_channels = 0;
        std::vector<std::complex<double>> rf;
        std::vector<double> adc_times;
        /** The phase, in rad, each ADC sample is multiplied by exp(i phase)
         *  with; one per sample. */
        std::vector<double> receiver;

        BlockEvents events() const;
    };

    /**
     * Repetitions of a sequence of blocks, played on isochromats.
     *
     * Repetition n plays the blocks with each RF pulse's field turned about z
     * by -phases[n], as a phase offset larger by phases[n] turns it, and with
     * gradients that differ from the blocks' by a waveform zero during every
     * pulse and held through every ADC window, at readouts[n][w] in Hz/m
     * through the repetition's w-th window, whose area at that window's first
     * sample is areas[n][w], in 1/m, by the start of any pulse zero, and over
     * the repetition nets[n]. Its samples are demodulated with each window's
     * phase larger by adc_phases[n].
     *
     * One repetition applies an affine map to each isochromat's
     * magnetisation, and another to its transverse magnetisation at each
     * window's first sample: four plays of the blocks, from zero and from a
     * unit magnetisation along each axis, give both. A repetition turned about
     * z applies the maps turned alike, and a waveform of no area at any pulse
     * turns the transverse magnetisation by its area at the sample, and at the
     * repetition's end by its net area, and nothing else, so every repetition
     * follows from the four plays. Each window is
     * read by the non-uniform FFT the isochromats read a window under a held
     * gradient with, and must be one; a window whose readouts are not zero is
     * turned, read along each repetition's own gradient.
     *
     * A first block that plays its pulse under a gradient held through
     * the block, larger by pulse_gradients[n] than the block's own in
     * repetition n, applies a map that differs from one repetition to the
     * next by more than a turn about z: each isochromat's is read off the
     * pulse's tables at the field it sees under that repetition's gradient,
     * with the block's free precession at that field before and after the
     * pulse. The four plays then play the blocks after the first, and the
     * areas and nets are what the waveform leaves over them.
     *
     * The isochromats hold the magnetisation at the start of the next
     * repetition to be played. Blocks played on them in between break the
     * repetitions, unless resume() takes what they leave as the start of
     * the next.
     *
     * The first play() moves the maps into the arrays the repetitions are
     * played from and frees them: split() and column_sums() come before it.
     * It also offers those arrays to the isochromats' run device; a device
     * that takes them carries them through every repetition and spreads them
     * onto the windows' grids, which the repetitions then read as they read
     * their own.
     */
    class Repetitions
    {
    public:
        /**
         * Play the four plays of @p blocks, leaving the isochromats as they
         * stand. A @p tolerance above zero, relative to the sum of the
         * proton densities, lets play() carry the magnetisation in single
         * precision from 1e-4 on and read the windows by a narrower kernel,
         * and split() drop transients below it.
         *
         * @throws std::invalid_argument if the phases, ADC phases, areas,
         *         readouts, nets and pulse gradients, the last three possibly
         *         none, are not one per repetition, a window is not read
         *         under a held gradient, a block is one the isochromats
         *         cannot play, or, with pulse gradients, the first block
         *         plays no pulse, reads a window or holds no gradient through
         *         it, or its pulse cannot be read off tables: its channels
         *         play more than one waveform, or its tables would take more
         *         points than the isochromats over every repetition.
         */
        Repetitions(
            Isochromats& isochromats,
            std::vector<OwnedBlock> blocks,
            std::vector<double> phases,
            std::vector<double> adc_phases,
            std::vector<double> areas,
            std::vector<double> readouts,
            std::vector<double> nets,
            std::vector<double> pulse_gradients,
            double tolerance = 0.0);
        ~Repetitions();
        Repetitions(const Repetitions&) = delete;
        Repetitions& operator=(const Repetitions&) = delete;

        /** Repetitions in all. */
        size_t count() const
        {
            return phases_.size();
        }
        /** Repetitions played. */
        size_t played() const
        {
            return next_;
        }
        /** Receive coils. */
        size_t coils() const
        {
            return isochromats_.coils();
        }
        /** ADC windows per repetition. */
        size_t windows() const
        {
            return windows_.size();
        }
        /** ADC samples per repetition. */
        size_t samples() const
        {
            return samples_;
        }

        /**
         * Split each isochromat's magnetisation into its fixed point under the
         * repetitions and a transient about it, and from then on carry only
         * transients larger than the tolerance times the proton density:
         * play() then writes the transients' samples alone, and
         * column_sums() gives what the fixed points send.
         * Return false, splitting nothing, unless the pulses turn by one step
         * each, no window is turned, the repetitions leave no net area and
         * the first block's pulse plays under the block's own gradient in
         * every one. A tolerance of zero drops no transient.
         *
         * @throws std::logic_error after the first play().
         */
        bool split();
        /** Whether split() split the magnetisation. */
        bool divided() const
        {
            return divided_;
        }
        /** Isochromats whose transient is carried. */
        size_t carried() const;
        /**
         * The fixed points' samples of window @p window summed over each
         * column of isochromats that share their coordinates along @p axes,
         * each axis tabulated (see lattice()): (columns, coils, samples),
         * column-major, the first axis's index slowest. Each isochromat
         * contributes each coil's sensitivity times its transverse
         * magnetisation at the window's first sample in its fixed point, in
         * the frame of the repetition's pulses, times exp(-2 pi i encoding .
         * r); and at sample k that times exp(-k step (1 / T2 + 2 pi i
         * off-resonance) - 2 pi i k area . r), area and step the window's.
         *
         * Written to @p out, which holds the product of the axes' lattice()
         * sizes times coils() times window_samples() values.
         *
         * @throws std::invalid_argument if an axis is not tabulated.
         */
        void column_sums(
            size_t window, const std::vector<int>& axes, const double encoding[3], std::complex<double>* out) const;
        /** The distinct coordinates, in m, ascending, of an axis some
         *  window's phase encoding runs along, where there are few enough of
         *  them to tabulate a phase per coordinate; empty otherwise. */
        const std::vector<double>& lattice(int axis) const
        {
            return lattice_[axis].values;
        }
        size_t window_samples(size_t window) const
        {
            return windows_[window].samples;
        }
        /** The largest magnitude of any isochromat's coordinate, in m. */
        double reach() const;

        /**
         * Play the next @p count repetitions, writing each repetition's
         * samples, coil-major, to @p signal: signal[(r * coils + c) *
         * samples() + k], and leave the isochromats at the start of the
         * repetition after them.
         *
         * @throws std::invalid_argument if fewer than @p count remain.
         */
        void play(size_t count, std::complex<double>* signal);

        /**
         * Take the magnetisation the isochromats hold, after blocks played on
         * them since the last play(), as that at the start of the next
         * repetition.
         *
         * @throws std::logic_error after split().
         */
        void resume();

    private:
        struct Set;

        /** A block's ADC window as the repetitions read it. */
        struct Window
        {
            size_t block = 0;
            size_t samples = 0;
            /** First sample of this window among the repetition's samples. */
            size_t offset = 0;
            /** Gradient area, in 1/m, and time, in s, from one sample to the
             *  next. */
            double area[3] = {0.0, 0.0, 0.0};
            double step = 0.0;
            /** Whether the repetitions read it along gradients of their
             *  own. */
            bool turned = false;
            std::vector<double> receiver;
            /** Mx + i My at the first sample: u . m + v, m the magnetisation
             *  at the repetition's start. */
            std::vector<std::complex<double>> u;
            std::vector<std::complex<double>> v;
        };

        /** Each isochromat's coordinate along one axis, as an index into the
         *  distinct coordinates, where they are few enough that a
         *  phase-encoding phase per coordinate costs less than one per
         *  isochromat. */
        struct Lattice
        {
            bool tabulated = false;
            std::vector<double> values;
            std::vector<uint32_t> index;
        };

        /** The field's turn of repetition @p n, in rad. */
        double turn(size_t n) const
        {
            return -phases_[n];
        }
        /** Read each block's ADC window, and the repetition's duration. */
        void read_windows();
        /** Take @p readouts as each turned window's steps, and turn the
         *  windows whose readouts are not zero. */
        void turn_windows(const std::vector<double>& readouts);
        bool turned() const;
        /** Check one ADC phase per repetition, and three pulse gradients
         *  and net areas each or none; forget those that are all zero. */
        void check_repetitions();
        /** Whether the first block's pulse is read off tables. */
        bool pulsed() const
        {
            return !pulse_gradients_.empty();
        }
        /** Check the first block plays a pulse under a gradient held through
         *  it, and reads no window; keep that gradient. */
        void hold_pulse();
        /** Make the tables the first block's pulse is read from. */
        void tabulate_pulse();
        /** Phase-encoding area @p at of repetition @p n: [window][axis] of
         *  its windows' areas, then [axis] of its net area. */
        double encoding_area(size_t n, size_t at) const;
        /** The four plays: each isochromat's maps. */
        void play_maps();
        /** Play the blocks from no magnetisation, @p column -1, or from a
         *  unit magnetisation along axis @p column, keeping the transverse
         *  magnetisation at each window's first sample in @p first. */
        void play_from(int column, std::vector<std::vector<std::complex<double>>>& first);
        /** Keep what the play from @p column gives: b and v, or a column of
         *  A and of u. */
        void store_map(int column, const std::vector<std::vector<std::complex<double>>>& first);
        /** Find the axes a phase encoding runs along, and tabulate the
         *  coordinates along each. */
        void tabulate_encoded();
        /** The phase, in cycles, isochromat @p i turns by from one sample of
         *  @p window to the next. */
        double window_phase(const Window& window, size_t i) const;
        /** Whether isochromat @p i's transient is above the tolerance times
         *  its proton density. */
        bool transient_above(size_t i) const;
        /** The isochromats carried, in order of T2 and of the first window's
         *  first grid point, which keeps a worker's spreading on a few grid
         *  points at a time: after a split, those whose transient is above
         *  the limit. */
        std::vector<uint32_t> carried_order() const;
        /** @p chosen in order of T2 and along a Z-order curve through the
         *  isochromats' coordinates: neighbours spread onto nearby grid
         *  points whatever the direction a turned window is read along. */
        std::vector<uint32_t> spatial_order(const std::vector<uint32_t>& chosen) const;
        /** Move the maps into the set played from, and free them. */
        template <typename Real>
        void gather();
        /** Write isochromat @p i to slot @p n of @p slots. */
        template <typename Slots>
        void fill_slot(Slots& slots, size_t n, size_t i, double* weights) const;
        /** Write the tables the first block's pulse is read off, per class,
         *  to @p slots. */
        template <typename Slots>
        void table_slots(Slots& slots) const;
        /** Write what reading isochromat @p i's map of the first block off
         *  the tables takes to slot @p n of @p slots. */
        template <typename Slots>
        void fill_pulse(Slots& slots, size_t n, size_t i) const;
        /** Write what window @p w reads of isochromat @p i to slot @p n. */
        template <typename Slots>
        void fill_window(Slots& slots, size_t w, size_t n, size_t i, double* weights) const;
        void release_maps();
        /** Each T2's decay from @p window's first sample to each. */
        std::vector<double> decays(const Window& window) const;
        template <typename Real>
        void play_tiles(size_t count, std::complex<double>* signal);
        /** Offer the carried slots to the run device, whose grids are laid
         *  out as @p tile's; whether it took them. */
        template <typename Tile>
        bool offer_device(const Tile& tile);
        /** The tile carried and spread on the run device onto the first of
         *  its grids, read as one worker's; the transients dropped. */
        template <typename Tile>
        size_t carry_on_device(Tile& tile);
        /** What reading the windows of a tile reads, and its arrays sized. */
        template <typename Tile>
        void plan_tile(Tile& tile, size_t taps) const;
        /** The turns and phase-encoding phases of the tile of repetitions
         *  from @p first on. */
        template <typename Tile>
        void prepare_tile(Tile& tile, size_t first);
        template <typename Tile>
        void encode_tile(Tile& tile, size_t first);
        /** Multiply the samples of @p count repetitions from @p first on by
         *  their pulses' turn, ADC phase and receiver phase. */
        void demodulate(size_t first, size_t count, std::complex<double>* out) const;
        /** The mean step between the turns of the repetitions from the next
         *  on, in @p step; false unless each step lies within
         *  kStepTolerance of it. */
        bool mean_step(double& step) const;
        /** The isochromats of each column along @p axes, in @p members, and
         *  where each column's start among them, one offset more than the
         *  columns. */
        std::vector<size_t> columns_along(const std::vector<int>& axes, std::vector<uint32_t>& members) const;
        /** Write the magnetisation at the start of repetition @p n, in the
         *  laboratory frame, to the isochromats. */
        template <typename Real>
        void settle(size_t n);
        /** Write the isochromats' magnetisation, turned into the frame of
         *  the next repetition's pulses, of cosine @p c and sine @p sn, to
         *  the slots. */
        template <typename Real>
        void resume_slots(double c, double sn);
        /** Fetch the magnetisation of the isochromat of @p slot, where it
         *  is before @p end, ahead of a read or, with @p Write 1, a write. */
        template <int Write, typename SlotsOf>
        void fetch_isochromat(const SlotsOf& slots, size_t slot, size_t end) const;
        /** Hand the run device the slots' magnetisation to take as its own
         *  where @p load, or have it write its own back into the slots. */
        template <typename Real>
        void exchange_state(bool load);

        Isochromats& isochromats_;
        std::vector<OwnedBlock> blocks_;
        std::vector<double> phases_;
        std::vector<double> adc_phases_;
        std::vector<double> areas_;
        /** Per repetition and window, [(n * windows + w) * 3 + axis], how
         *  much the gradient area from one sample to the next exceeds the
         *  window's, in 1/m. */
        std::vector<double> readouts_;
        /** Per repetition and axis, the area it leaves, in 1/m; empty where
         *  none leaves any. */
        std::vector<double> nets_;
        /** Per repetition and axis, how much the gradient held through the
         *  first block exceeds the block's own, in Hz/m; empty where the
         *  first block's pulse is played as the other blocks' are. */
        std::vector<double> pulse_gradients_;
        /** The first block's own gradient, held through it, in Hz/m, and
         *  the tables its pulse is read off. */
        double held_[3] = {0.0, 0.0, 0.0};
        Isochromats::RunTables pulse_tables_;
        std::vector<Window> windows_;
        size_t samples_ = 0;
        double duration_ = 0.0;
        /** Per isochromat, the repetition's map M -> A M + b, A row-major. */
        std::vector<double> a_;
        std::vector<double> b_;
        /** Per isochromat, the magnetisation at the start of the next
         *  repetition, in the frame its pulses' turn turns. */
        std::vector<double> m_;
        size_t next_ = 0;
        Lattice lattice_[3];
        /** Whether any window's phase encoding runs along each axis. */
        bool encoded_[3] = {false, false, false};
        /** Whether m_ holds transients about the fixed points fixed_ and a_
         *  holds the map turned by the step. */
        bool divided_ = false;
        double tolerance_ = 0.0;
        double step_ = 0.0;
        size_t split_at_ = 0;
        std::vector<double> fixed_;
        /** Each window's transform, by the kernel the tolerance allows. */
        std::vector<std::unique_ptr<Nufft>> transforms_;
        /** Each window's e^(i receiver phase) per sample. */
        std::vector<std::vector<std::complex<double>>> receivers_;
        /** The isochromats carried, from the first play() on. */
        std::unique_ptr<Set> set_;
        /** This run, among every engine's and run's identities. */
        size_t run_ = 0;
    };

} // namespace bloch

#endif /* PULSERVER_BLOCH_REPETITIONS_HPP */
