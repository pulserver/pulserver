/**
 * @file events.hpp
 * @brief The events a block is made of, held the way a sequence wants them.
 *
 * PyPulseq's `make_*` functions hand back a `SimpleNamespace`, and reading one
 * costs a dictionary lookup per field.  That is fine once; it is not fine two
 * million times, which is what building a three-dimensional protocol does.
 * These are the same events with the fields in slots.
 *
 * ### Waveforms are stored normalised
 *
 * An RF pulse is kept as a magnitude in 0..1, a phase in turns, and one scalar
 * amplitude; an arbitrary gradient as a normalised waveform and one scalar.
 * That is exactly how the file stores them -- the `[SHAPES]` entry is
 * normalised and the amplitude lives in the event row -- so nothing is being
 * transformed for our convenience.
 *
 * What it buys is that **scaling a pulse is one number**.  A variable flip
 * angle train is the same magnitude shape at a hundred amplitudes, and with
 * this layout it registers that shape once and writes a hundred rows that
 * differ in a single column.  Scaling the samples instead would make a hundred
 * shapes that deduplication then has to discover are related, having already
 * paid to store them.
 *
 * ### Registered ids ride along
 *
 * Once a waveform has been registered with a sequence, its shape ids are kept
 * here so the next block that plays it does not go looking again -- which is
 * what Pulseq's own `registerGradEvent` does with `event.shapeIDs`.  The
 * difference is the `owner` stamp: ids are only reused for the sequence that
 * issued them, so the same event played into a second sequence registers
 * afresh instead of pointing at a shape that sequence has never heard of.
 */

#ifndef PULSEQ_CXX_EVENTS_HPP
#define PULSEQ_CXX_EVENTS_HPP

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace pulseq
{

    /** What each event is, so one cast identifies any of them. */
    enum class EventKind
    {
        Rf,
        Trap,
        Grad,
        Adc,
        Label,
        Trigger,
        Rotation,
        SoftDelay,
        Delay
    };

    /**
     * The common head of every event.
     *
     * Present so that unpacking a block is one cast and a switch rather than a
     * chain of type tests: a block carrying seven events would otherwise ask
     * "is this a trapezoid?" some thirty times.
     */
    struct Event
    {
        explicit Event(EventKind of) : kind(of) {}
        EventKind kind;
    };

    /** What a waveform was registered as, and with whom. */
    struct Registration
    {
        int32_t owner = 0;
        int32_t shape = 0;
        int32_t time_shape = 0;

        bool known_to(int32_t sequence) const { return owner == sequence && owner != 0; }
        void note(int32_t sequence, int32_t shape_id, int32_t time_shape_id)
        {
            owner = sequence;
            shape = shape_id;
            time_shape = time_shape_id;
        }
    };

    /** Which axis a gradient drives. */
    enum class Axis
    {
        X = 0,
        Y = 1,
        Z = 2
    };

    struct RfEvent : Event
    {
        RfEvent() : Event(EventKind::Rf) {}

        /** Peak amplitude in Hz.  The only thing scaling a pulse touches. */
        double amplitude = 0.0;
        /** Magnitude in 0..1 and phase in turns -- the file's own units. */
        std::vector<double> magnitude;
        std::vector<double> phase;
        /** Sample times in seconds; empty means the RF raster. */
        std::vector<double> t;

        double shape_dur = 0.0;
        double center = 0.0;
        double delay = 0.0;
        double freq_offset = 0.0;
        double phase_offset = 0.0;
        double freq_ppm = 0.0;
        double phase_ppm = 0.0;
        double dead_time = 0.0;
        double ringdown_time = 0.0;
        /** Pulseq's word ("excitation", ...), kept whole so the event reads
         *  back as PyPulseq built it.  The file carries the initial. */
        std::string use = "undefined";

        char use_code() const { return use.empty() ? 'u' : use[0]; }

        /** Magnitude, phase and time shapes; `shape` is the magnitude. */
        Registration registered;
        int32_t phase_shape = 0;

        double duration() const { return shape_dur + delay + ringdown_time; }
    };

    struct TrapEvent : Event
    {
        TrapEvent() : Event(EventKind::Trap) {}

        double amplitude = 0.0;
        double rise_time = 0.0;
        double flat_time = 0.0;
        double fall_time = 0.0;
        double delay = 0.0;
        Axis axis = Axis::X;

        double duration() const { return delay + rise_time + flat_time + fall_time; }
    };

    struct GradEvent : Event
    {
        GradEvent() : Event(EventKind::Grad) {}

        /** Peak amplitude in Hz/m, signed by the first non-zero sample. */
        double amplitude = 0.0;
        /** The waveform normalised by that amplitude. */
        std::vector<double> waveform;
        /** Sample times in seconds; empty means the gradient raster. */
        std::vector<double> tt;

        double first = 0.0;
        double last = 0.0;
        double delay = 0.0;
        double last_time = 0.0;  /**< tt.back(), kept so duration() is O(1). */
        Axis axis = Axis::X;

        Registration registered;
    };

    struct AdcEvent : Event
    {
        AdcEvent() : Event(EventKind::Adc) {}

        double num_samples = 0.0;
        double dwell = 0.0;
        double delay = 0.0;
        double freq_offset = 0.0;
        double phase_offset = 0.0;
        double freq_ppm = 0.0;
        double phase_ppm = 0.0;
        double dead_time = 0.0;
        std::vector<double> phase_modulation;

        Registration registered;

        double duration() const { return delay + num_samples * dwell + dead_time; }
    };

    struct LabelEvent : Event
    {
        LabelEvent() : Event(EventKind::Label) {}

        int32_t value = 0;
        std::string label;
        bool setting = true;  /**< false for an increment. */
        /** The id this sequence gave `label`, cached like a shape id. */
        int32_t owner = 0;
        int32_t label_id = 0;
    };

    struct TriggerEvent : Event
    {
        TriggerEvent() : Event(EventKind::Trigger) {}

        /** 1 for an output, 2 for a trigger -- Pulseq's own numbering.
         *  Not called `kind`: that name belongs to Event. */
        double control = 2.0;
        double channel = 1.0;
        double delay = 0.0;
        double duration_s = 0.0;
    };

    struct RotationEvent : Event
    {
        RotationEvent() : Event(EventKind::Rotation) {}

        std::array<double, 4> quaternion{{1.0, 0.0, 0.0, 0.0}};
    };

    struct SoftDelayEvent : Event
    {
        SoftDelayEvent() : Event(EventKind::SoftDelay) {}

        int32_t num = 0;
        double offset = 0.0;
        double factor = 1.0;
        double default_duration = 0.0;
        std::string hint;
    };

    struct DelayEvent : Event
    {
        DelayEvent() : Event(EventKind::Delay) {}

        double delay = 0.0;
    };

}  // namespace pulseq

#endif /* PULSEQ_CXX_EVENTS_HPP */
