/**
 * @file _pulseqpp_eventtypes.h
 * @brief Python bindings for pulseq's event classes.
 *
 * These are what `pulserver.pypulseq.make_*` hands back: the same events
 * PyPulseq builds, converted once into slots.  They quack like the
 * `SimpleNamespace` they replace -- `rf.signal` is the complex waveform at its
 * real amplitude, `grad.waveform` is the scaled samples -- so anything
 * downstream that reads an event keeps working.
 *
 * ### The point of them
 *
 * Reading a `SimpleNamespace` costs a dictionary lookup per field, and a
 * three-dimensional protocol reads tens of millions.  Here a field is an
 * offset.
 *
 * ### Setters do not validate
 *
 * Deliberately.  `make_*` checked the event when it built it; a loop that
 * moves a phase encode from one line to the next is not making a new
 * assertion about the hardware, and re-deriving one per block is precisely
 * the cost this exists to remove.  A caller that wants the checks back calls
 * `make_*` again.
 *
 * ### Scaling is one number
 *
 * `signal` and `waveform` are stored normalised beside a scalar amplitude, so
 * `rf.amplitude *= 0.5` is a single write and leaves the registered shape
 * valid.  Assigning to `signal` or `waveform` outright re-normalises and
 * drops the registration, because that really is a new waveform.
 *
 * ### Why these are hand-written CPython types
 *
 * A pybind11 `def_readwrite` is a property: reading one is a descriptor call
 * into a lambda, and it measured at 178 ns against a `SimpleNamespace`
 * field's 56 ns.  So the dictionary lookups removed from `add_block` came
 * straight back on the loop's own arithmetic, which touches an amplitude or a
 * phase offset several times per repetition.
 *
 * A `PyMemberDef` is not a call.  It is a type code and a byte offset, and
 * `PyMember_GetOne` reads the double out of the instance directly, so the C++
 * object has to live *inside* the `PyObject` rather than behind a pointer --
 * which is the one thing pybind11's instance layout will not do.  Hence
 * `Holder<T>`, nine static types, and `tp_members` for every scalar.  The
 * fields that are not scalars -- waveforms, names, the computed areas -- stay
 * accessor-shaped in `tp_getset`, where the cost is beside the work.
 */

#ifndef PULSERVER_PULSEQPP_EVENTTYPES_H
#define PULSERVER_PULSEQPP_EVENTTYPES_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pulseq/events.hpp"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulseqpp_types
{
    namespace py = pybind11;
    using namespace pulseq;

    /* ================================================================== */
    /*  Normalisation                                                     */
    /* ================================================================== */

    /** Split a complex RF waveform into an amplitude, a magnitude and a phase. */
    inline void normalise_rf(const std::complex<double>* values, int count, RfEvent& out)
    {
        out.magnitude.resize(static_cast<size_t>(count));
        out.phase.resize(static_cast<size_t>(count));
        double peak = 0.0;
        for (int i = 0; i < count; ++i)
        {
            const double m = std::abs(values[i]);
            if (m > peak)
                peak = m;
        }
        out.amplitude = peak;
        for (int i = 0; i < count; ++i)
        {
            out.magnitude[static_cast<size_t>(i)] = peak > 0.0 ? std::abs(values[i]) / peak : 0.0;
            double angle = std::arg(values[i]);
            if (angle < 0.0)
                angle += 2.0 * M_PI;
            out.phase[static_cast<size_t>(i)] = angle / (2.0 * M_PI);
        }
        out.registered = Registration{};
    }

    /**
     * Split a gradient waveform into an amplitude and a normalised shape.
     *
     * The amplitude carries the sign of the first non-zero sample, which is
     * Pulseq's convention and what keeps a waveform and its negation one
     * shape rather than two.
     */
    inline void normalise_grad(const double* values, int count, GradEvent& out)
    {
        double peak = 0.0;
        for (int i = 0; i < count; ++i)
        {
            const double m = std::fabs(values[i]);
            if (m > peak)
                peak = m;
        }
        if (peak > 0.0)
        {
            for (int i = 0; i < count; ++i)
            {
                if (values[i] != 0.0)
                {
                    if (values[i] < 0.0)
                        peak = -peak;
                    break;
                }
            }
        }
        out.amplitude = peak;
        out.waveform.resize(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i)
            out.waveform[static_cast<size_t>(i)] = peak != 0.0 ? values[i] / peak : values[i];
        out.registered = Registration{};
    }

    /* ================================================================== */
    /*  Duck-typed views                                                  */
    /* ================================================================== */

    inline py::array_t<std::complex<double>> rf_signal(const RfEvent& rf)
    {
        const size_t n = rf.magnitude.size();
        py::array_t<std::complex<double>> out(static_cast<py::ssize_t>(n));
        std::complex<double>* data = out.mutable_data();
        for (size_t i = 0; i < n; ++i)
        {
            const double turns = 2.0 * M_PI * rf.phase[i];
            data[i] = std::polar(rf.amplitude * rf.magnitude[i], turns);
        }
        return out;
    }

    inline py::array_t<double> grad_waveform(const GradEvent& g)
    {
        const size_t n = g.waveform.size();
        py::array_t<double> out(static_cast<py::ssize_t>(n));
        double* data = out.mutable_data();
        for (size_t i = 0; i < n; ++i)
            data[i] = g.amplitude * g.waveform[i];
        return out;
    }

    /** The sample times an event was built on, materialised on the raster. */
    inline py::array_t<double> raster_times(const std::vector<double>& stored, int count,
                                            double raster, double offset)
    {
        if (!stored.empty())
            return py::array_t<double>(static_cast<py::ssize_t>(stored.size()), stored.data());
        py::array_t<double> out(count);
        double* data = out.mutable_data();
        for (int i = 0; i < count; ++i)
            data[i] = (i + offset) * raster;
        return out;
    }

    inline std::vector<double> as_vector(const py::object& source)
    {
        if (source.is_none())
            return {};
        auto array =
            py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(source);
        const double* data = array.data();
        return std::vector<double>(data, data + array.size());
    }

    /* ================================================================== */
    /*  The instance layout                                               */
    /* ================================================================== */

    /**
     * A Python object with the event *in* it.
     *
     * Inline, not behind a pointer, because that is what makes a field a
     * `PyMemberDef`: the offset a member descriptor holds is measured from
     * the head of the instance, so there can be no indirection between them.
     */
    template <typename T>
    struct Holder
    {
        PyObject_HEAD
        T event;
    };

/* Every event derives from Event, so none of them is standard-layout and
 * offsetof is formally off-limits.  It is nevertheless exactly right for
 * single, non-virtual inheritance, which is what these are; bind() proves it
 * at import against a live object rather than trusting the compiler. */
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#endif

    /** Where the C++ object starts.  The same for all nine -- see bind(). */
    constexpr Py_ssize_t EVENT_OFFSET = static_cast<Py_ssize_t>(offsetof(Holder<RfEvent>, event));

    /** A field's offset from the head of the instance. */
#define PULSEQPP_FIELD(EventType, member) \
    (static_cast<Py_ssize_t>(offsetof(Holder<EventType>, event) + offsetof(EventType, member)))

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

    /** The base the nine event types share; none of them is subclassable. */
    extern PyTypeObject EventBaseType;

    /** Is this one of ours?  One dereference, because the nine are leaves. */
    inline bool is_event(PyObject* object)
    {
        return Py_TYPE(object)->tp_base == &EventBaseType;
    }

    inline Event* event_of(PyObject* object)
    {
        return reinterpret_cast<Event*>(reinterpret_cast<char*>(object) + EVENT_OFFSET);
    }

    void bind(py::module_& m);

}  // namespace pulseqpp_types

#endif /* PULSERVER_PULSEQPP_EVENTTYPES_H */
