/**
 * @file _pulseqpp_events.h
 * @brief `add_block(*events)`: PyPulseq event objects straight into the C++
 *        libraries, with no sequence object in between.
 *
 * This is the ergonomic path -- `seq.add_block(rf, gx, gy, gz, adc, lin, par)`
 * with the objects `make_trapezoid` and friends hand back -- and it is one
 * call per block rather than one per event, because everything between the
 * objects and the libraries is done here.
 *
 * The one cost that cannot be moved is that the events *are* Python objects:
 * a trapezoid is five attribute reads whoever does them.  They are done
 * against the instance dictionary with interned keys rather than through
 * `getattr`, which skips the descriptor protocol and the temporary each
 * lookup would otherwise build -- worth roughly a third of the extraction on
 * a block carrying seven events.
 *
 * Nothing is checked against a library on the way in and no waveform is
 * compressed: a scan built this way registers a row per use and a shape per
 * shot, and `remove_duplicates` followed by the writers' prewrite pass turns
 * that into the file.  Searching per event would make building quadratic in
 * the thing that is already the largest.
 */

#ifndef PULSERVER_PULSEQPP_EVENTS_H
#define PULSERVER_PULSEQPP_EVENTS_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "_pulseqpp_eventtypes.h"
#include "pulseq/events.hpp"
#include "pulseq/sequence.hpp"

#include <cmath>
#include <algorithm>
#include <complex>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <string>
#include <vector>

namespace pulseqpp_events
{
    namespace py = pybind11;

    /**
     * A sequence, plus the one thing the "register everything, sort it out
     * later" design cannot do blindly.
     *
     * Event rows are small -- a trapezoid is five doubles -- so appending one
     * per use and collapsing them at the end costs a few tens of bytes per
     * block and is plainly the right trade.  A *shape* is not small.  A spiral
     * arm is several thousand samples, and a stack-of-spirals protocol plays
     * half a million of them: registering the waveform again on every shot
     * would cost twenty-five gigabytes before deduplication ever ran, to end
     * up with the thousand distinct arms the scan actually has.
     *
     * So shapes -- and only shapes -- are remembered by the identity of the
     * array they came from.  Not by their contents: hashing a waveform per
     * shot is the cost this design exists to avoid.  A caller that reuses an
     * array registers its shape once; one that builds a new array every time
     * gets a new shape, which is exactly right, because it really is telling
     * us it made a new waveform.
     *
     * The arrays are held so their addresses cannot be reused by something
     * else once they are collected.
     */
    /**
     * What a waveform resolves to: its shapes, and the amplitude the row
     * carries separately.
     *
     * The amplitude is remembered along with the shapes because it is derived
     * from the same samples.  Recomputing it per use is not free -- it is a
     * pass over the whole waveform -- and on a spiral protocol that pass, run
     * twice per shot for half a million shots, was an order of magnitude more
     * than everything else in the build put together.
     */
    struct Registered
    {
        int32_t shape = 0;
        int32_t time_shape = 0;
        double amplitude = 0.0;
    };

    struct BoundSequence : pulseq::Sequence
    {
        /**
         * Which sequence this is, so an event can tell whether the shape ids
         * it is carrying were issued here.  Serial rather than the address,
         * because addresses are reused and a stale id is worse than no id.
         */
        BoundSequence()
        {
            static int32_t issued = 0;
            serial = ++issued;
        }
        int32_t serial = 0;

        std::unordered_map<PyObject*, Registered> shape_ids;

        /**
         * Quaternions, by the identity of the rotation object they came from.
         *
         * Not to save the four doubles -- those are registered per use like
         * any other row, and deduplication collapses them -- but to save
         * *asking* for them.  `as_quat()` is a call into SciPy that builds an
         * array, and a non-Cartesian scan asks a rotation object the same
         * question once per shot.  A stack of spirals with a thousand
         * orientations asks it half a million times.
         */
        std::unordered_map<PyObject*, std::array<double, pulseq::ROTATION_WIDTH>> quaternions;
        std::vector<py::object> pinned;

        void remember(PyObject* source, const Registered& what)
        {
            shape_ids.emplace(source, what);
            pinned.push_back(py::reinterpret_borrow<py::object>(source));
        }

        /** The shape id for @p source, registering its samples on first sight. */
        int shape_for(PyObject* source, const double* samples, int count)
        {
            auto found = shape_ids.find(source);
            if (found != shape_ids.end())
                return found->second.shape;
            Registered made;
            made.shape = static_cast<int32_t>(register_raw_shape(samples, count));
            remember(source, made);
            return made.shape;
        }
    };

    /** An interned attribute name, made once and never released. */
    inline PyObject* key(const char* name)
    {
        PyObject* interned = PyUnicode_InternFromString(name);
        return interned;  // deliberately leaked: one per name, for the process
    }

    /**
     * The instance dictionary of @p object.
     *
     * One attribute lookup per event, and every field after it is a dictionary
     * hit with an interned key -- which skips the descriptor protocol that
     * `getattr` walks, and is what makes reading seven events off a block cost
     * a fraction of what it otherwise would.
     */
    inline py::object fields(const py::handle& object)
    {
        PyObject* dict = PyObject_GetAttrString(object.ptr(), "__dict__");
        if (!dict)
        {
            PyErr_Clear();
            throw std::invalid_argument(
                "add_block() takes PyPulseq event objects; this one has no attributes");
        }
        return py::reinterpret_steal<py::object>(dict);
    }

    /** Borrowed lookup; null when absent.  Never raises. */
    inline PyObject* field(PyObject* dict, PyObject* name)
    {
        return dict ? PyDict_GetItemWithError(dict, name) : nullptr;
    }

    inline double as_double(PyObject* value, double fallback = 0.0)
    {
        if (!value)
            return fallback;
        const double out = PyFloat_AsDouble(value);
        if (out == -1.0 && PyErr_Occurred())
        {
            PyErr_Clear();
            return fallback;
        }
        return out;
    }

    /** A contiguous float64 view of an array attribute, without copying it. */
    struct Samples
    {
        py::array_t<double, py::array::c_style | py::array::forcecast> held;
        const double* data = nullptr;
        int count = 0;

        explicit Samples(PyObject* value)
        {
            if (!value)
                return;
            held = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(
                py::reinterpret_borrow<py::object>(value));
            data = held.data();
            count = static_cast<int>(held.size());
        }
    };

    /* ================================================================== */
    /*  Field names                                                       */
    /* ================================================================== */

    struct Names
    {
        PyObject* type = key("type");
        PyObject* amplitude = key("amplitude");
        PyObject* rise_time = key("rise_time");
        PyObject* flat_time = key("flat_time");
        PyObject* fall_time = key("fall_time");
        PyObject* delay = key("delay");
        PyObject* waveform = key("waveform");
        PyObject* tt = key("tt");
        PyObject* first = key("first");
        PyObject* last = key("last");
        PyObject* signal = key("signal");
        PyObject* t = key("t");
        PyObject* shape_dur = key("shape_dur");
        PyObject* center = key("center");
        PyObject* freq_offset = key("freq_offset");
        PyObject* phase_offset = key("phase_offset");
        PyObject* freq_ppm = key("freq_ppm");
        PyObject* phase_ppm = key("phase_ppm");
        PyObject* ringdown_time = key("ringdown_time");
        PyObject* dead_time = key("dead_time");
        PyObject* use = key("use");
        PyObject* num_samples = key("num_samples");
        PyObject* dwell = key("dwell");
        PyObject* phase_modulation = key("phase_modulation");
        PyObject* label = key("label");
        PyObject* value = key("value");
        PyObject* channel = key("channel");
        PyObject* duration = key("duration");
        PyObject* rot_quaternion = key("rot_quaternion");
        PyObject* shim_vector = key("shim_vector");
        PyObject* numID = key("numID");
        PyObject* offset = key("offset");
        PyObject* factor = key("factor");
        PyObject* hint = key("hint");
        PyObject* default_duration = key("default_duration");
    };

    inline const Names& names()
    {
        static const Names singleton;
        return singleton;
    }

    /* ================================================================== */
    /*  Per-event handlers                                                */
    /* ================================================================== */

    /**
     * The magnitude and phase shapes of an RF pulse, registered raw.
     *
     * `signal` is complex, and the file wants it as a normalised magnitude and
     * a phase in turns -- so this is the one place a waveform is transformed
     * rather than passed through.  The time shape is registered only when the
     * samples are not on the RF raster, which is what PyPulseq decides too.
     */
    inline double register_rf_shapes(BoundSequence& seq, PyObject* dict, double rf_raster,
                                     int32_t shapes[3])
    {
        const Names& n = names();
        PyObject* signal_object = field(dict, n.signal);

        // Answered without looking at a single sample when the pulse has been
        // seen before, which on a scan that plays one excitation per TR is
        // every time but the first.
        auto known = seq.shape_ids.find(signal_object);
        if (known != seq.shape_ids.end())
        {
            shapes[0] = known->second.shape;
            shapes[1] = static_cast<int32_t>(shapes[0] + 1);
            shapes[2] = known->second.time_shape;
            return known->second.amplitude;
        }

        auto signal = py::cast<py::array_t<std::complex<double>,
                                           py::array::c_style | py::array::forcecast>>(
            py::reinterpret_borrow<py::object>(signal_object));
        const std::complex<double>* values = signal.data();
        const int count = static_cast<int>(signal.size());

        std::vector<double> magnitude(static_cast<size_t>(count));
        std::vector<double> phase(static_cast<size_t>(count));
        double amplitude = 0.0;
        for (int i = 0; i < count; ++i)
        {
            const double m = std::abs(values[i]);
            if (m > amplitude)
                amplitude = m;
        }
        for (int i = 0; i < count; ++i)
        {
            magnitude[static_cast<size_t>(i)] =
                amplitude > 0.0 ? std::abs(values[i]) / amplitude : 0.0;
            double angle = std::arg(values[i]);
            if (angle < 0.0)
                angle += 2.0 * M_PI;
            phase[static_cast<size_t>(i)] = angle / (2.0 * M_PI);
        }

        // Magnitude and phase are both functions of the signal, so they are
        // registered together and the pair is contiguous.
        shapes[0] = static_cast<int32_t>(seq.register_raw_shape(magnitude.data(), count));
        shapes[1] = static_cast<int32_t>(seq.register_raw_shape(phase.data(), count));
        shapes[2] = 0;

        const Samples times(field(dict, n.t));
        bool on_raster = times.count == count;
        for (int i = 0; on_raster && i < times.count; ++i)
        {
            if (std::floor(times.data[i] / rf_raster) != static_cast<double>(i))
                on_raster = false;
        }
        if (!on_raster && times.count)
        {
            std::vector<double> ticks(static_cast<size_t>(times.count));
            for (int i = 0; i < times.count; ++i)
                ticks[static_cast<size_t>(i)] = times.data[i] / rf_raster;
            shapes[2] = static_cast<int32_t>(
                seq.register_raw_shape(ticks.data(), times.count));
        }

        Registered made;
        made.shape = shapes[0];
        made.time_shape = shapes[2];
        made.amplitude = amplitude;
        seq.remember(signal_object, made);
        return amplitude;
    }

    /**
     * The amplitude and shapes of an arbitrary gradient, registered raw.
     *
     * The time column is 0 for the standard raster and -1 for the half-raster
     * variant, and only a genuinely irregular grid becomes a shape.  PyPulseq
     * decides this from the *compressed* timing; the same question is asked
     * here of the samples themselves, which is the same answer without
     * compressing a waveform to find it out.
     */
    inline double register_grad_shapes(BoundSequence& seq, PyObject* dict, double grad_raster,
                                       int32_t shapes[2])
    {
        const Names& n = names();
        PyObject* waveform_object = field(dict, n.waveform);

        auto known = seq.shape_ids.find(waveform_object);
        if (known != seq.shape_ids.end())
        {
            shapes[0] = known->second.shape;
            shapes[1] = known->second.time_shape;
            return known->second.amplitude;
        }

        const Samples wave(waveform_object);
        const Samples times(field(dict, n.tt));

        double amplitude = 0.0;
        for (int i = 0; i < wave.count; ++i)
        {
            const double m = std::fabs(wave.data[i]);
            if (m > amplitude)
                amplitude = m;
        }
        if (amplitude > 0.0)
        {
            for (int i = 0; i < wave.count; ++i)
            {
                if (wave.data[i] != 0.0)
                {
                    if (wave.data[i] < 0.0)
                        amplitude = -amplitude;
                    break;
                }
            }
        }

        std::vector<double> normalised(static_cast<size_t>(wave.count));
        for (int i = 0; i < wave.count; ++i)
            normalised[static_cast<size_t>(i)] =
                amplitude != 0.0 ? wave.data[i] / amplitude : wave.data[i];
        shapes[0] = static_cast<int32_t>(
            seq.register_raw_shape(normalised.data(), wave.count));

        // Standard raster: sample i sits at (i + 1/2) rasters.
        // Half raster: sample i sits at (i + 1) half-rasters.
        shapes[1] = 0;
        bool standard = times.count == wave.count;
        bool half = times.count == wave.count;
        for (int i = 0; (standard || half) && i < times.count; ++i)
        {
            const double ticks = times.data[i] / grad_raster;
            if (std::fabs(ticks - (i + 0.5)) > 1e-9)
                standard = false;
            if (std::fabs(ticks - 0.5 * (i + 1)) > 1e-9)
                half = false;
        }
        if (half)
        {
            shapes[1] = -1;
        }
        else if (!standard && times.count)
        {
            std::vector<double> ticks(static_cast<size_t>(times.count));
            for (int i = 0; i < times.count; ++i)
                ticks[static_cast<size_t>(i)] = times.data[i] / grad_raster;
            shapes[1] = static_cast<int32_t>(
                seq.register_raw_shape(ticks.data(), times.count));
        }

        Registered made;
        made.shape = shapes[0];
        made.time_shape = shapes[1];
        made.amplitude = amplitude;
        seq.remember(waveform_object, made);
        return amplitude;
    }

    /** Which gradient axis an event names, or -1. */
    inline int axis_of(PyObject* dict)
    {
        static PyObject* channel_key = key("channel");
        PyObject* value = field(dict, channel_key);
        if (!value || !PyUnicode_Check(value))
            return -1;
        const char* name = PyUnicode_AsUTF8(value);
        if (!name || !name[0])
            return -1;
        switch (name[0])
        {
            case 'x': return 0;
            case 'y': return 1;
            case 'z': return 2;
            default: return -1;
        }
    }



    /* ================================================================== */
    /*  The fast path: events that are already unpacked                   */
    /* ================================================================== */

    /** Register a typed RF event's shapes, unless this sequence already has them. */
    inline void warm_rf(BoundSequence& seq, pulseq::RfEvent& e)
    {
        if (e.registered.known_to(seq.serial))
            return;

        const int count = static_cast<int>(e.magnitude.size());
        const int32_t magnitude =
            static_cast<int32_t>(seq.register_raw_shape(e.magnitude.data(), count));
        e.phase_shape = static_cast<int32_t>(
            seq.register_raw_shape(e.phase.data(), static_cast<int>(e.phase.size())));

        int32_t time_shape = 0;
        if (!e.t.empty())
        {
            const double raster = seq.rf_raster_time();
            bool on_raster = static_cast<int>(e.t.size()) == count;
            for (int i = 0; on_raster && i < count; ++i)
            {
                if (std::floor(e.t[static_cast<size_t>(i)] / raster) != static_cast<double>(i))
                    on_raster = false;
            }
            if (!on_raster)
            {
                std::vector<double> ticks(e.t.size());
                for (size_t i = 0; i < e.t.size(); ++i)
                    ticks[i] = e.t[i] / raster;
                time_shape = static_cast<int32_t>(
                    seq.register_raw_shape(ticks.data(), static_cast<int>(ticks.size())));
            }
        }
        e.registered.note(seq.serial, magnitude, time_shape);
    }

    /** Register a typed RF event, reusing its shapes if this sequence made them. */
    inline void take_rf(BoundSequence& seq, pulseq::RfEvent& e, pulseq::Block& block,
                        double& duration)
    {
        warm_rf(seq, e);

        const double row[pulseq::RF_WIDTH] = {
            e.amplitude,
            static_cast<double>(e.registered.shape),
            static_cast<double>(e.phase_shape),
            static_cast<double>(e.registered.time_shape),
            e.center,
            e.delay,
            e.freq_ppm,
            e.phase_ppm,
            e.freq_offset,
            e.phase_offset};
        block.rf = seq.register_rf(row, e.use_code());
        duration = std::max(duration, e.duration());
    }

    /** Register a typed gradient's shapes, unless this sequence already has them. */
    inline void warm_grad(BoundSequence& seq, pulseq::GradEvent& e)
    {
        if (e.registered.known_to(seq.serial))
            return;

        const int count = static_cast<int>(e.waveform.size());
        const int32_t shape =
            static_cast<int32_t>(seq.register_raw_shape(e.waveform.data(), count));

        // 0 is the standard raster, -1 the half-raster variant, and only
        // a grid that is neither becomes a shape of its own.
        int32_t time_shape = 0;
        if (!e.tt.empty())
        {
            const double raster = seq.grad_raster_time();
            bool standard = static_cast<int>(e.tt.size()) == count;
            bool half = standard;
            for (int i = 0; (standard || half) && i < count; ++i)
            {
                const double ticks = e.tt[static_cast<size_t>(i)] / raster;
                if (std::fabs(ticks - (i + 0.5)) > 1e-9)
                    standard = false;
                if (std::fabs(ticks - 0.5 * (i + 1)) > 1e-9)
                    half = false;
            }
            if (half)
            {
                time_shape = -1;
            }
            else if (!standard)
            {
                std::vector<double> ticks(e.tt.size());
                for (size_t i = 0; i < e.tt.size(); ++i)
                    ticks[i] = e.tt[i] / raster;
                time_shape = static_cast<int32_t>(
                    seq.register_raw_shape(ticks.data(), static_cast<int>(ticks.size())));
            }
        }
        e.registered.note(seq.serial, shape, time_shape);
    }

    inline void take_grad(BoundSequence& seq, pulseq::GradEvent& e, pulseq::Block& block,
                          double& duration)
    {
        warm_grad(seq, e);

        const double row[pulseq::ARB_WIDTH] = {
            e.amplitude,
            e.first,
            e.last,
            static_cast<double>(e.registered.shape),
            static_cast<double>(e.registered.time_shape),
            e.delay};
        const int id = seq.register_arbitrary(row);
        switch (e.axis)
        {
            case pulseq::Axis::X: block.gx = id; break;
            case pulseq::Axis::Y: block.gy = id; break;
            default: block.gz = id; break;
        }
        duration = std::max(duration,
                            e.delay + std::ceil(e.last_time / seq.grad_raster_time() - 1e-10) *
                                          seq.grad_raster_time());
    }

    /** Register a typed ADC's phase-modulation shape, if it has one. */
    inline void warm_adc(BoundSequence& seq, pulseq::AdcEvent& e)
    {
        if (e.phase_modulation.empty() || e.registered.known_to(seq.serial))
            return;
        const int32_t shape = static_cast<int32_t>(seq.register_raw_shape(
            e.phase_modulation.data(), static_cast<int>(e.phase_modulation.size())));
        e.registered.note(seq.serial, shape, 0);
    }

    inline void take_adc(BoundSequence& seq, pulseq::AdcEvent& e, pulseq::Block& block,
                         double& duration)
    {
        warm_adc(seq, e);
        const double row[pulseq::ADC_WIDTH] = {
            e.num_samples,
            e.dwell,
            // The later of the two, not their sum: Pulseq lets the dead time
            // and the delay overlap.
            std::max(e.delay, e.dead_time),
            e.freq_ppm,
            e.phase_ppm,
            e.freq_offset,
            e.phase_offset,
            e.phase_modulation.empty() ? 0.0 : static_cast<double>(e.registered.shape)};
        block.adc = seq.register_adc(row);
        duration = std::max(duration, e.duration());
    }

    /* ================================================================== */
    /*  The block                                                         */
    /* ================================================================== */

    /**
     * Register every event of one block and return the block that plays them.
     *
     * The block's duration is the longest thing in it, rounded up onto the
     * block raster -- which is what PyPulseq computes, and what a caller who
     * passes a bare `make_delay` is asking for directly.
     *
     * Whether the block is then appended or written over an existing one is
     * the caller's business; the registration is the same either way.
     */
    inline pulseq::Block build_block(BoundSequence& seq, const py::args& events)
    {
        const Names& n = names();
        const double rf_raster = seq.rf_raster_time();
        const double grad_raster = seq.grad_raster_time();
        const double block_raster = seq.block_duration_raster();

        pulseq::Block block;
        double duration = 0.0;
        // Chain links, in the order given; the first listed ends up the head.
        int32_t chain[8][2];
        int chained = 0;

        for (const py::handle& event : events)
        {
            // Already unpacked?  One comparison, one offset and a switch,
            // rather than a dictionary lookup per field.
            if (pulseqpp_types::is_event(event.ptr()))
            {
                pulseq::Event* base = pulseqpp_types::event_of(event.ptr());
                switch (base->kind)
                {
                    case pulseq::EventKind::Trap:
                    {
                        auto& e = *static_cast<pulseq::TrapEvent*>(base);
                        const double row[pulseq::TRAP_WIDTH] = {e.amplitude, e.rise_time,
                                                                e.flat_time, e.fall_time,
                                                                e.delay};
                        const int id = seq.register_trap(row);
                        switch (e.axis)
                        {
                            case pulseq::Axis::X: block.gx = id; break;
                            case pulseq::Axis::Y: block.gy = id; break;
                            default: block.gz = id; break;
                        }
                        duration = std::max(duration, e.duration());
                        break;
                    }
                    case pulseq::EventKind::Rf:
                        take_rf(seq, *static_cast<pulseq::RfEvent*>(base), block, duration);
                        break;
                    case pulseq::EventKind::Grad:
                        take_grad(seq, *static_cast<pulseq::GradEvent*>(base), block, duration);
                        break;
                    case pulseq::EventKind::Adc:
                        take_adc(seq, *static_cast<pulseq::AdcEvent*>(base), block, duration);
                        break;
                    case pulseq::EventKind::Label:
                    {
                        auto& e = *static_cast<pulseq::LabelEvent*>(base);
                        if (e.owner != seq.serial)
                        {
                            e.label_id = static_cast<int32_t>(seq.label_id(e.label));
                            e.owner = seq.serial;
                        }
                        const int ref = e.setting ? seq.register_label_set(e.value, e.label_id)
                                                  : seq.register_label_inc(e.value, e.label_id);
                        if (chained == 8)
                            throw std::invalid_argument(
                                "a block carries at most eight extensions");
                        chain[chained][0] = static_cast<int32_t>(
                            seq.extension_type_id(e.setting ? "LABELSET" : "LABELINC"));
                        chain[chained][1] = static_cast<int32_t>(ref);
                        ++chained;
                        break;
                    }
                    case pulseq::EventKind::Trigger:
                    {
                        auto& e = *static_cast<pulseq::TriggerEvent*>(base);
                        const double row[pulseq::TRIGGER_WIDTH] = {e.control, e.channel, e.delay,
                                                                   e.duration_s};
                        if (chained == 8)
                            throw std::invalid_argument(
                                "a block carries at most eight extensions");
                        chain[chained][0] =
                            static_cast<int32_t>(seq.extension_type_id("TRIGGERS"));
                        chain[chained][1] = static_cast<int32_t>(seq.register_trigger(row));
                        ++chained;
                        duration = std::max(duration, e.delay + e.duration_s);
                        break;
                    }
                    case pulseq::EventKind::Rotation:
                    {
                        auto& e = *static_cast<pulseq::RotationEvent*>(base);
                        if (chained == 8)
                            throw std::invalid_argument(
                                "a block carries at most eight extensions");
                        chain[chained][0] =
                            static_cast<int32_t>(seq.extension_type_id("ROTATIONS"));
                        chain[chained][1] =
                            static_cast<int32_t>(seq.register_rotation(e.quaternion.data()));
                        ++chained;
                        break;
                    }
                    case pulseq::EventKind::SoftDelay:
                    {
                        auto& e = *static_cast<pulseq::SoftDelayEvent*>(base);
                        pulseq::SoftDelay row;
                        row.num = e.num;
                        row.offset = e.offset;
                        row.factor = e.factor;
                        row.hint = e.hint;
                        if (chained == 8)
                            throw std::invalid_argument(
                                "a block carries at most eight extensions");
                        chain[chained][0] = static_cast<int32_t>(seq.extension_type_id("DELAYS"));
                        chain[chained][1] =
                            static_cast<int32_t>(seq.register_soft_delay(row));
                        ++chained;
                        duration = std::max(duration, e.default_duration);
                        break;
                    }
                    case pulseq::EventKind::Delay:
                        duration = std::max(duration,
                                            static_cast<pulseq::DelayEvent*>(base)->delay);
                        break;
                }
                continue;
            }

            const py::object holder = fields(event);
            PyObject* dict = holder.ptr();
            PyObject* kind_value = field(dict, n.type);
            if (!kind_value || !PyUnicode_Check(kind_value))
                throw std::invalid_argument("add_block() event has no `type`");
            const char* kind = PyUnicode_AsUTF8(kind_value);

            if (std::strcmp(kind, "trap") == 0)
            {
                const double delay = as_double(field(dict, n.delay));
                const double rise = as_double(field(dict, n.rise_time));
                const double flat = as_double(field(dict, n.flat_time));
                const double fall = as_double(field(dict, n.fall_time));
                const double row[pulseq::TRAP_WIDTH] = {
                    as_double(field(dict, n.amplitude)), rise, flat, fall, delay};
                const int id = seq.register_trap(row);
                switch (axis_of(dict))
                {
                    case 0: block.gx = id; break;
                    case 1: block.gy = id; break;
                    case 2: block.gz = id; break;
                    default: throw std::invalid_argument("gradient event has no channel");
                }
                duration = std::max(duration, delay + rise + flat + fall);
            }
            else if (std::strcmp(kind, "grad") == 0)
            {
                int32_t shapes[2];
                const double amplitude = register_grad_shapes(seq, dict, grad_raster, shapes);
                const double delay = as_double(field(dict, n.delay));
                const double row[pulseq::ARB_WIDTH] = {
                    amplitude,
                    as_double(field(dict, n.first)),
                    as_double(field(dict, n.last)),
                    static_cast<double>(shapes[0]),
                    static_cast<double>(shapes[1]),
                    delay};
                const int id = seq.register_arbitrary(row);
                switch (axis_of(dict))
                {
                    case 0: block.gx = id; break;
                    case 1: block.gy = id; break;
                    case 2: block.gz = id; break;
                    default: throw std::invalid_argument("gradient event has no channel");
                }
                const Samples times(field(dict, n.tt));
                if (times.count)
                {
                    const double last = times.data[times.count - 1];
                    duration = std::max(
                        duration,
                        delay + std::ceil(last / grad_raster - 1e-10) * grad_raster);
                }
            }
            else if (std::strcmp(kind, "rf") == 0)
            {
                int32_t shapes[3];
                const double amplitude = register_rf_shapes(seq, dict, rf_raster, shapes);
                const double delay = as_double(field(dict, n.delay));
                const double row[pulseq::RF_WIDTH] = {
                    amplitude,
                    static_cast<double>(shapes[0]),
                    static_cast<double>(shapes[1]),
                    static_cast<double>(shapes[2]),
                    as_double(field(dict, n.center)),
                    delay,
                    as_double(field(dict, n.freq_ppm)),
                    as_double(field(dict, n.phase_ppm)),
                    as_double(field(dict, n.freq_offset)),
                    as_double(field(dict, n.phase_offset))};
                char use = 'u';
                PyObject* use_value = field(dict, n.use);
                if (use_value && PyUnicode_Check(use_value))
                {
                    const char* text = PyUnicode_AsUTF8(use_value);
                    if (text && text[0])
                        use = text[0];
                }
                block.rf = seq.register_rf(row, use);
                duration = std::max(duration, as_double(field(dict, n.shape_dur)) + delay +
                                                  as_double(field(dict, n.ringdown_time)));
            }
            else if (std::strcmp(kind, "adc") == 0)
            {
                const double delay = as_double(field(dict, n.delay));
                const double num = as_double(field(dict, n.num_samples));
                const double dwell = as_double(field(dict, n.dwell));
                double phase_shape = 0.0;
                PyObject* modulation = field(dict, n.phase_modulation);
                if (modulation && modulation != Py_None)
                {
                    const Samples samples(modulation);
                    if (samples.count)
                        phase_shape = static_cast<double>(
                            seq.shape_for(modulation, samples.data, samples.count));
                }
                // Pulseq stores the later of the two rather than their sum:
                // the dead time and the delay are allowed to overlap.  Both
                // the MATLAB toolbox and PyPulseq do this, and a file written
                // with the plain delay reads back as a different ADC.
                const double dead = as_double(field(dict, n.dead_time));
                const double row[pulseq::ADC_WIDTH] = {
                    num,
                    dwell,
                    std::max(delay, dead),
                    as_double(field(dict, n.freq_ppm)),
                    as_double(field(dict, n.phase_ppm)),
                    as_double(field(dict, n.freq_offset)),
                    as_double(field(dict, n.phase_offset)),
                    phase_shape};
                block.adc = seq.register_adc(row);
                duration = std::max(duration, delay + num * dwell + dead);
            }
            else if (std::strcmp(kind, "labelset") == 0 || std::strcmp(kind, "labelinc") == 0)
            {
                PyObject* name_value = field(dict, n.label);
                if (!name_value || !PyUnicode_Check(name_value))
                    throw std::invalid_argument("label event has no `label`");
                const int label = seq.label_id(PyUnicode_AsUTF8(name_value));
                const int32_t value =
                    static_cast<int32_t>(std::lround(as_double(field(dict, n.value))));
                const bool setting = kind[5] == 's';
                const int ref = setting ? seq.register_label_set(value, label)
                                        : seq.register_label_inc(value, label);
                if (chained == 8)
                    throw std::invalid_argument("a block carries at most eight extensions");
                chain[chained][0] =
                    static_cast<int32_t>(seq.extension_type_id(setting ? "LABELSET" : "LABELINC"));
                chain[chained][1] = static_cast<int32_t>(ref);
                ++chained;
            }
            else if (std::strcmp(kind, "trigger") == 0 || std::strcmp(kind, "output") == 0)
            {
                const double delay = as_double(field(dict, n.delay));
                const double length = as_double(field(dict, n.duration));
                const bool is_trigger = kind[0] == 't';
                double channel = 1.0;
                PyObject* channel_value = field(dict, n.channel);
                if (channel_value && PyUnicode_Check(channel_value))
                {
                    const std::string text = PyUnicode_AsUTF8(channel_value);
                    if (is_trigger)
                        channel = text == "physio2" ? 2.0 : 1.0;
                    else
                        channel = text == "osc1" ? 2.0 : (text == "ext1" ? 3.0 : 1.0);
                }
                const double row[pulseq::TRIGGER_WIDTH] = {is_trigger ? 2.0 : 1.0, channel,
                                                           delay, length};
                if (chained == 8)
                    throw std::invalid_argument("a block carries at most eight extensions");
                chain[chained][0] = static_cast<int32_t>(seq.extension_type_id("TRIGGERS"));
                chain[chained][1] = static_cast<int32_t>(seq.register_trigger(row));
                ++chained;
                duration = std::max(duration, delay + length);
            }
            else if (std::strcmp(kind, "rot3D") == 0)
            {
                // The cheap way to play a non-Cartesian trajectory: one base
                // waveform, and four numbers per shot saying where it points.
                PyObject* rotation = field(dict, n.rot_quaternion);
                if (!rotation)
                    throw std::invalid_argument("rotation event has no `rot_quaternion`");
                auto cached = seq.quaternions.find(rotation);
                if (cached == seq.quaternions.end())
                {
                    py::object quaternion = py::reinterpret_borrow<py::object>(rotation).attr(
                        "as_quat")(py::arg("canonical") = true, py::arg("scalar_first") = true);
                    auto values =
                        py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(
                            quaternion);
                    if (values.size() != pulseq::ROTATION_WIDTH)
                        throw std::invalid_argument("a rotation is four numbers");
                    std::array<double, pulseq::ROTATION_WIDTH> row;
                    std::memcpy(row.data(), values.data(),
                                sizeof(double) * pulseq::ROTATION_WIDTH);
                    double norm = 0.0;
                    for (double component : row)
                        norm += component * component;
                    if (std::fabs(1.0 - norm) > 1e-6)
                        throw std::invalid_argument(
                            "rotation quaternion is not a unit quaternion");
                    cached = seq.quaternions.emplace(rotation, row).first;
                    seq.pinned.push_back(py::reinterpret_borrow<py::object>(rotation));
                }
                if (chained == 8)
                    throw std::invalid_argument("a block carries at most eight extensions");
                chain[chained][0] = static_cast<int32_t>(seq.extension_type_id("ROTATIONS"));
                chain[chained][1] =
                    static_cast<int32_t>(seq.register_rotation(cached->second.data()));
                ++chained;
            }
            else if (std::strcmp(kind, "rf_shim") == 0)
            {
                auto shim = py::cast<py::array_t<std::complex<double>,
                                                 py::array::c_style | py::array::forcecast>>(
                    py::reinterpret_borrow<py::object>(field(dict, n.shim_vector)));
                const std::complex<double>* channels = shim.data();
                const int count = static_cast<int>(shim.size());
                // Stored as magnitude and phase per channel, interleaved, which
                // is how the file carries it.
                std::vector<double> values(static_cast<size_t>(count) * 2);
                for (int i = 0; i < count; ++i)
                {
                    values[static_cast<size_t>(2 * i)] = std::abs(channels[i]);
                    values[static_cast<size_t>(2 * i + 1)] = std::arg(channels[i]);
                }
                if (chained == 8)
                    throw std::invalid_argument("a block carries at most eight extensions");
                chain[chained][0] = static_cast<int32_t>(seq.extension_type_id("RF_SHIMS"));
                chain[chained][1] = static_cast<int32_t>(
                    seq.register_rf_shim(values.data(), static_cast<int>(values.size())));
                ++chained;
            }
            else if (std::strcmp(kind, "soft_delay") == 0)
            {
                pulseq::SoftDelay row;
                row.num = static_cast<int32_t>(
                    std::lround(as_double(field(dict, n.numID))));
                row.offset = as_double(field(dict, n.offset));
                row.factor = as_double(field(dict, n.factor));
                PyObject* hint_value = field(dict, n.hint);
                if (hint_value && PyUnicode_Check(hint_value))
                    row.hint = PyUnicode_AsUTF8(hint_value);
                if (chained == 8)
                    throw std::invalid_argument("a block carries at most eight extensions");
                chain[chained][0] = static_cast<int32_t>(seq.extension_type_id("DELAYS"));
                chain[chained][1] = static_cast<int32_t>(seq.register_soft_delay(row));
                ++chained;
                // A soft delay's default is what the block lasts until the
                // console is told otherwise.
                duration = std::max(duration, as_double(field(dict, n.default_duration)));
            }
            else if (std::strcmp(kind, "delay") == 0)
            {
                duration = std::max(duration, as_double(field(dict, n.delay)));
            }
            else
            {
                throw std::invalid_argument(std::string("add_block() cannot take a `") + kind +
                                            "` event");
            }
        }

        // Built tail first, so walking the chain gives the events back in the
        // order they were passed.
        for (int i = chained - 1; i >= 0; --i)
            block.ext = static_cast<int32_t>(
                seq.append_extension(chain[i][0], chain[i][1], block.ext));

        if (block_raster > 0.0)
            duration = std::ceil(duration / block_raster - 1e-12) * block_raster;
        block.duration = duration;
        return block;
    }

    /* ================================================================== */
    /*  Pre-registration -- upstream's register_*_event                    */
    /* ================================================================== */

    /**
     * Register one event's *shapes* into @p seq without adding a block, and
     * report the ids.
     *
     * This is the half of upstream's `register_*_event` that means something
     * here.  The other half -- an event-library row id -- is deliberately not
     * produced: rows are appended per block and renumbered by
     * `remove_duplicates`, so there is no id to hand out that would still be
     * true afterwards.
     *
     * Doing it early is not a saving on its own; the same shapes would be
     * registered by the first `add_block` and memoized identically.  What it
     * buys is what upstream's API is for -- the cost lands here rather than on
     * whichever loop iteration happened to come first.
     *
     * Works for both event flavours: a slotted event memoizes on itself
     * (`Event::registered`, keyed by this sequence's serial), a namespace one
     * on the identity of the array behind it (`BoundSequence::shape_ids`).
     */
    inline std::vector<int32_t> warm_event(BoundSequence& seq, const py::handle& event)
    {
        if (pulseqpp_types::is_event(event.ptr()))
        {
            pulseq::Event* base = pulseqpp_types::event_of(event.ptr());
            switch (base->kind)
            {
                case pulseq::EventKind::Rf:
                {
                    auto& e = *static_cast<pulseq::RfEvent*>(base);
                    warm_rf(seq, e);
                    return {e.registered.shape, e.phase_shape, e.registered.time_shape};
                }
                case pulseq::EventKind::Grad:
                {
                    auto& e = *static_cast<pulseq::GradEvent*>(base);
                    warm_grad(seq, e);
                    return {e.registered.shape, e.registered.time_shape};
                }
                case pulseq::EventKind::Adc:
                {
                    auto& e = *static_cast<pulseq::AdcEvent*>(base);
                    warm_adc(seq, e);
                    return {e.registered.shape};
                }
                default:
                    // A trapezoid, label, trigger, rotation or delay carries no
                    // waveform, so there is nothing to warm.
                    return {};
            }
        }

        // A namespace event: which shapes it has is told by which fields it has.
        const Names& n = names();
        const py::object holder = fields(event);
        PyObject* dict = holder.ptr();

        if (field(dict, n.signal))
        {
            int32_t shapes[3] = {0, 0, 0};
            register_rf_shapes(seq, dict, seq.rf_raster_time(), shapes);
            return {shapes[0], shapes[1], shapes[2]};
        }
        if (field(dict, n.waveform))
        {
            int32_t shapes[2] = {0, 0};
            register_grad_shapes(seq, dict, seq.grad_raster_time(), shapes);
            return {shapes[0], shapes[1]};
        }
        return {};
    }

}  // namespace pulseqpp_events

#endif /* PULSERVER_PULSEQPP_EVENTS_H */
