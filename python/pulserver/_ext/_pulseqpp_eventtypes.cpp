/**
 * @file _pulseqpp_eventtypes.cpp
 * @brief The nine event types, hand-written.  See _pulseqpp_eventtypes.h for why.
 *
 * Scalars are `tp_members`, so reading one is a type code and an offset.
 * Waveforms, names and computed quantities are `tp_getset`, where an accessor
 * call is beside the point next to the array it builds.
 */

#include "_pulseqpp_eventtypes.h"

#include <cstring>
#include <new>

/* PULSEQPP_FIELD reaches a member through a base class, which is exactly
 * right here and formally off-limits; ready() proves each layout at import. */
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#endif

#if PY_VERSION_HEX < 0x030C0000
#include <structmember.h>
#define Py_T_DOUBLE T_DOUBLE
#define Py_T_INT T_INT
#define Py_T_BOOL T_BOOL
#endif

namespace pulseqpp_types
{

    PyTypeObject EventBaseType = {PyVarObject_HEAD_INIT(nullptr, 0)};

    namespace
    {
        static_assert(sizeof(int32_t) == sizeof(int), "Py_T_INT fields are int32_t");

        /** The event inside @p self.  The caller has already identified it. */
        template <typename T>
        inline T& unwrap(PyObject* self)
        {
            return reinterpret_cast<Holder<T>*>(self)->event;
        }

        /** A getter's body, with C++ exceptions turned into Python ones. */
        template <typename Make>
        inline PyObject* guarded(Make&& make)
        {
            try
            {
                return py::object(make()).release().ptr();
            }
            catch (py::error_already_set& raised)
            {
                raised.restore();
                return nullptr;
            }
            catch (const std::exception& raised)
            {
                PyErr_SetString(PyExc_ValueError, raised.what());
                return nullptr;
            }
        }

        /** A setter's body.  Deleting an event's field is not a thing. */
        template <typename Apply>
        inline int guarded_set(PyObject* value, Apply&& apply)
        {
            if (!value)
            {
                PyErr_SetString(PyExc_AttributeError, "an event field cannot be deleted");
                return -1;
            }
            try
            {
                apply(py::reinterpret_borrow<py::object>(value));
                return 0;
            }
            catch (py::error_already_set& raised)
            {
                raised.restore();
                return -1;
            }
            catch (const std::exception& raised)
            {
                PyErr_SetString(PyExc_ValueError, raised.what());
                return -1;
            }
        }

        /** An interned name, made once per call site and never released. */
        inline PyObject* interned(PyObject*& slot, const char* text)
        {
            if (!slot)
                slot = PyUnicode_InternFromString(text);
            return slot ? Py_NewRef(slot) : nullptr;
        }

        inline py::array_t<double> samples_of(const std::vector<double>& values)
        {
            return py::array_t<double>(static_cast<py::ssize_t>(values.size()), values.data());
        }

        Axis axis_from(const std::string& channel)
        {
            if (channel == "x")
                return Axis::X;
            if (channel == "y")
                return Axis::Y;
            if (channel == "z")
                return Axis::Z;
            throw std::invalid_argument("gradient channel must be 'x', 'y' or 'z'");
        }

        PyObject* axis_name(Axis axis)
        {
            static PyObject* names[3] = {nullptr, nullptr, nullptr};
            static const char* text[3] = {"x", "y", "z"};
            const int which = static_cast<int>(axis);
            return interned(names[which], text[which]);
        }

        std::array<double, 4> unit_quaternion(const py::object& value)
        {
            const std::vector<double> given = as_vector(value);
            if (given.size() != 4)
                throw std::invalid_argument("a rotation is four numbers");
            double norm = 0.0;
            for (double component : given)
                norm += component * component;
            if (std::fabs(1.0 - norm) > 1e-6)
                throw std::invalid_argument("rotation quaternion is not a unit quaternion");
            return {{given[0], given[1], given[2], given[3]}};
        }

        /* ============================================================== */
        /*  Lifetime                                                      */
        /* ============================================================== */

        template <typename T>
        PyObject* generic_new(PyTypeObject* type, PyObject*, PyObject*)
        {
            PyObject* self = type->tp_alloc(type, 0);
            if (!self)
                return nullptr;
            new (&reinterpret_cast<Holder<T>*>(self)->event) T();
            return self;
        }

        template <typename T>
        void generic_dealloc(PyObject* self)
        {
            reinterpret_cast<Holder<T>*>(self)->event.~T();
            Py_TYPE(self)->tp_free(self);
        }

        /* ============================================================== */
        /*  RF                                                            */
        /* ============================================================== */

        PyObject* rf_type(PyObject*, void*)
        {
            static PyObject* name = nullptr;
            return interned(name, "rf");
        }

        PyObject* rf_get_use(PyObject* self, void*)
        {
            const std::string& use = unwrap<RfEvent>(self).use;
            return PyUnicode_FromStringAndSize(use.data(), static_cast<Py_ssize_t>(use.size()));
        }

        int rf_set_use(PyObject* self, PyObject* value, void*)
        {
            return guarded_set(value, [&](const py::object& given) {
                unwrap<RfEvent>(self).use = given.cast<std::string>();
            });
        }

        PyObject* rf_get_signal(PyObject* self, void*)
        {
            return guarded([&] { return rf_signal(unwrap<RfEvent>(self)); });
        }

        int rf_set_signal(PyObject* self, PyObject* value, void*)
        {
            return guarded_set(value, [&](const py::object& given) {
                auto samples = py::cast<
                    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>>(
                    given);
                normalise_rf(samples.data(), static_cast<int>(samples.size()),
                             unwrap<RfEvent>(self));
            });
        }

        PyObject* rf_get_magnitude(PyObject* self, void*)
        {
            return guarded([&] { return samples_of(unwrap<RfEvent>(self).magnitude); });
        }

        PyObject* rf_get_phase(PyObject* self, void*)
        {
            return guarded([&] { return samples_of(unwrap<RfEvent>(self).phase); });
        }

        PyObject* rf_get_t(PyObject* self, void*)
        {
            return guarded([&] { return samples_of(unwrap<RfEvent>(self).t); });
        }

        int rf_set_t(PyObject* self, PyObject* value, void*)
        {
            return guarded_set(value, [&](const py::object& given) {
                RfEvent& event = unwrap<RfEvent>(self);
                event.t = as_vector(given);
                event.registered = Registration{};
            });
        }

        Py_ssize_t rf_length(PyObject* self)
        {
            return static_cast<Py_ssize_t>(unwrap<RfEvent>(self).magnitude.size());
        }

        PyMemberDef rf_members[] = {
            {"amplitude", Py_T_DOUBLE, PULSEQPP_FIELD(RfEvent, amplitude), 0, nullptr},
            {"shape_dur", Py_T_DOUBLE, PULSEQPP_FIELD(RfEvent, shape_dur), 0, nullptr},
            {"center", Py_T_DOUBLE, PULSEQPP_FIELD(RfEvent, center), 0, nullptr},
            {"delay", Py_T_DOUBLE, PULSEQPP_FIELD(RfEvent, delay), 0, nullptr},
            {"freq_offset", Py_T_DOUBLE, PULSEQPP_FIELD(RfEvent, freq_offset), 0, nullptr},
            {"phase_offset", Py_T_DOUBLE, PULSEQPP_FIELD(RfEvent, phase_offset), 0, nullptr},
            {"freq_ppm", Py_T_DOUBLE, PULSEQPP_FIELD(RfEvent, freq_ppm), 0, nullptr},
            {"phase_ppm", Py_T_DOUBLE, PULSEQPP_FIELD(RfEvent, phase_ppm), 0, nullptr},
            {"dead_time", Py_T_DOUBLE, PULSEQPP_FIELD(RfEvent, dead_time), 0, nullptr},
            {"ringdown_time", Py_T_DOUBLE, PULSEQPP_FIELD(RfEvent, ringdown_time), 0, nullptr},
            {nullptr, 0, 0, 0, nullptr}};

        PyGetSetDef rf_getset[] = {
            {"type", rf_type, nullptr, nullptr, nullptr},
            {"use", rf_get_use, rf_set_use, nullptr, nullptr},
            {"signal", rf_get_signal, rf_set_signal, nullptr, nullptr},
            {"magnitude", rf_get_magnitude, nullptr, nullptr, nullptr},
            {"phase", rf_get_phase, nullptr, nullptr, nullptr},
            {"t", rf_get_t, rf_set_t, nullptr, nullptr},
            {nullptr, nullptr, nullptr, nullptr, nullptr}};

        PySequenceMethods rf_sequence = {rf_length};

        /* ============================================================== */
        /*  Trapezoid                                                     */
        /* ============================================================== */

        PyObject* trap_type(PyObject*, void*)
        {
            static PyObject* name = nullptr;
            return interned(name, "trap");
        }

        PyObject* trap_get_channel(PyObject* self, void*)
        {
            return axis_name(unwrap<TrapEvent>(self).axis);
        }

        int trap_set_channel(PyObject* self, PyObject* value, void*)
        {
            return guarded_set(value, [&](const py::object& given) {
                unwrap<TrapEvent>(self).axis = axis_from(given.cast<std::string>());
            });
        }

        PyObject* trap_get_area(PyObject* self, void*)
        {
            const TrapEvent& event = unwrap<TrapEvent>(self);
            return PyFloat_FromDouble(event.amplitude *
                                      (event.flat_time + event.rise_time / 2.0 +
                                       event.fall_time / 2.0));
        }

        PyObject* trap_get_flat_area(PyObject* self, void*)
        {
            const TrapEvent& event = unwrap<TrapEvent>(self);
            return PyFloat_FromDouble(event.amplitude * event.flat_time);
        }

        PyMemberDef trap_members[] = {
            {"amplitude", Py_T_DOUBLE, PULSEQPP_FIELD(TrapEvent, amplitude), 0, nullptr},
            {"rise_time", Py_T_DOUBLE, PULSEQPP_FIELD(TrapEvent, rise_time), 0, nullptr},
            {"flat_time", Py_T_DOUBLE, PULSEQPP_FIELD(TrapEvent, flat_time), 0, nullptr},
            {"fall_time", Py_T_DOUBLE, PULSEQPP_FIELD(TrapEvent, fall_time), 0, nullptr},
            {"delay", Py_T_DOUBLE, PULSEQPP_FIELD(TrapEvent, delay), 0, nullptr},
            {nullptr, 0, 0, 0, nullptr}};

        PyGetSetDef trap_getset[] = {
            {"type", trap_type, nullptr, nullptr, nullptr},
            {"channel", trap_get_channel, trap_set_channel, nullptr, nullptr},
            {"area", trap_get_area, nullptr, nullptr, nullptr},
            {"flat_area", trap_get_flat_area, nullptr, nullptr, nullptr},
            {nullptr, nullptr, nullptr, nullptr, nullptr}};

        /* ============================================================== */
        /*  Arbitrary gradient                                            */
        /* ============================================================== */

        PyObject* grad_type(PyObject*, void*)
        {
            static PyObject* name = nullptr;
            return interned(name, "grad");
        }

        PyObject* grad_get_channel(PyObject* self, void*)
        {
            return axis_name(unwrap<GradEvent>(self).axis);
        }

        int grad_set_channel(PyObject* self, PyObject* value, void*)
        {
            return guarded_set(value, [&](const py::object& given) {
                unwrap<GradEvent>(self).axis = axis_from(given.cast<std::string>());
            });
        }

        PyObject* grad_get_waveform(PyObject* self, void*)
        {
            return guarded([&] { return grad_waveform(unwrap<GradEvent>(self)); });
        }

        int grad_set_waveform(PyObject* self, PyObject* value, void*)
        {
            return guarded_set(value, [&](const py::object& given) {
                auto samples =
                    py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(
                        given);
                normalise_grad(samples.data(), static_cast<int>(samples.size()),
                               unwrap<GradEvent>(self));
            });
        }

        PyObject* grad_get_shape(PyObject* self, void*)
        {
            return guarded([&] { return samples_of(unwrap<GradEvent>(self).waveform); });
        }

        PyObject* grad_get_tt(PyObject* self, void*)
        {
            return guarded([&] { return samples_of(unwrap<GradEvent>(self).tt); });
        }

        int grad_set_tt(PyObject* self, PyObject* value, void*)
        {
            return guarded_set(value, [&](const py::object& given) {
                GradEvent& event = unwrap<GradEvent>(self);
                event.tt = as_vector(given);
                event.last_time = event.tt.empty() ? 0.0 : event.tt.back();
                event.registered = Registration{};
            });
        }

        Py_ssize_t grad_length(PyObject* self)
        {
            return static_cast<Py_ssize_t>(unwrap<GradEvent>(self).waveform.size());
        }

        PyMemberDef grad_members[] = {
            {"amplitude", Py_T_DOUBLE, PULSEQPP_FIELD(GradEvent, amplitude), 0, nullptr},
            {"first", Py_T_DOUBLE, PULSEQPP_FIELD(GradEvent, first), 0, nullptr},
            {"last", Py_T_DOUBLE, PULSEQPP_FIELD(GradEvent, last), 0, nullptr},
            {"delay", Py_T_DOUBLE, PULSEQPP_FIELD(GradEvent, delay), 0, nullptr},
            {nullptr, 0, 0, 0, nullptr}};

        PyGetSetDef grad_getset[] = {
            {"type", grad_type, nullptr, nullptr, nullptr},
            {"channel", grad_get_channel, grad_set_channel, nullptr, nullptr},
            {"waveform", grad_get_waveform, grad_set_waveform, nullptr, nullptr},
            {"shape", grad_get_shape, nullptr, nullptr, nullptr},
            {"tt", grad_get_tt, grad_set_tt, nullptr, nullptr},
            {nullptr, nullptr, nullptr, nullptr, nullptr}};

        PySequenceMethods grad_sequence = {grad_length};

        /* ============================================================== */
        /*  ADC                                                           */
        /* ============================================================== */

        PyObject* adc_type(PyObject*, void*)
        {
            static PyObject* name = nullptr;
            return interned(name, "adc");
        }

        PyObject* adc_get_phase_modulation(PyObject* self, void*)
        {
            const AdcEvent& event = unwrap<AdcEvent>(self);
            if (event.phase_modulation.empty())
                Py_RETURN_NONE;
            return guarded([&] { return samples_of(event.phase_modulation); });
        }

        int adc_set_phase_modulation(PyObject* self, PyObject* value, void*)
        {
            return guarded_set(value, [&](const py::object& given) {
                AdcEvent& event = unwrap<AdcEvent>(self);
                event.phase_modulation = as_vector(given);
                event.registered = Registration{};
            });
        }

        PyMemberDef adc_members[] = {
            {"num_samples", Py_T_DOUBLE, PULSEQPP_FIELD(AdcEvent, num_samples), 0, nullptr},
            {"dwell", Py_T_DOUBLE, PULSEQPP_FIELD(AdcEvent, dwell), 0, nullptr},
            {"delay", Py_T_DOUBLE, PULSEQPP_FIELD(AdcEvent, delay), 0, nullptr},
            {"freq_offset", Py_T_DOUBLE, PULSEQPP_FIELD(AdcEvent, freq_offset), 0, nullptr},
            {"phase_offset", Py_T_DOUBLE, PULSEQPP_FIELD(AdcEvent, phase_offset), 0, nullptr},
            {"freq_ppm", Py_T_DOUBLE, PULSEQPP_FIELD(AdcEvent, freq_ppm), 0, nullptr},
            {"phase_ppm", Py_T_DOUBLE, PULSEQPP_FIELD(AdcEvent, phase_ppm), 0, nullptr},
            {"dead_time", Py_T_DOUBLE, PULSEQPP_FIELD(AdcEvent, dead_time), 0, nullptr},
            {nullptr, 0, 0, 0, nullptr}};

        PyGetSetDef adc_getset[] = {
            {"type", adc_type, nullptr, nullptr, nullptr},
            {"phase_modulation", adc_get_phase_modulation, adc_set_phase_modulation, nullptr,
             nullptr},
            {nullptr, nullptr, nullptr, nullptr, nullptr}};

        /* ============================================================== */
        /*  Label                                                         */
        /* ============================================================== */

        PyObject* label_type(PyObject* self, void*)
        {
            static PyObject* set_name = nullptr;
            static PyObject* inc_name = nullptr;
            return unwrap<LabelEvent>(self).setting ? interned(set_name, "labelset")
                                                    : interned(inc_name, "labelinc");
        }

        PyObject* label_get_label(PyObject* self, void*)
        {
            const std::string& label = unwrap<LabelEvent>(self).label;
            return PyUnicode_FromStringAndSize(label.data(),
                                               static_cast<Py_ssize_t>(label.size()));
        }

        int label_set_label(PyObject* self, PyObject* value, void*)
        {
            return guarded_set(value, [&](const py::object& given) {
                LabelEvent& event = unwrap<LabelEvent>(self);
                event.label = given.cast<std::string>();
                event.owner = 0;  // the cached id belonged to the old name
            });
        }

        PyMemberDef label_members[] = {
            {"value", Py_T_INT, PULSEQPP_FIELD(LabelEvent, value), 0, nullptr},
            {"setting", Py_T_BOOL, PULSEQPP_FIELD(LabelEvent, setting), 0, nullptr},
            {nullptr, 0, 0, 0, nullptr}};

        PyGetSetDef label_getset[] = {
            {"type", label_type, nullptr, nullptr, nullptr},
            {"label", label_get_label, label_set_label, nullptr, nullptr},
            {nullptr, nullptr, nullptr, nullptr, nullptr}};

        /* ============================================================== */
        /*  Trigger                                                       */
        /* ============================================================== */

        PyObject* trigger_type(PyObject* self, void*)
        {
            static PyObject* trigger_name = nullptr;
            static PyObject* output_name = nullptr;
            return unwrap<TriggerEvent>(self).control == 2.0 ? interned(trigger_name, "trigger")
                                                             : interned(output_name, "output");
        }

        PyMemberDef trigger_members[] = {
            {"control", Py_T_DOUBLE, PULSEQPP_FIELD(TriggerEvent, control), 0, nullptr},
            {"channel_code", Py_T_DOUBLE, PULSEQPP_FIELD(TriggerEvent, channel), 0, nullptr},
            {"delay", Py_T_DOUBLE, PULSEQPP_FIELD(TriggerEvent, delay), 0, nullptr},
            {"duration", Py_T_DOUBLE, PULSEQPP_FIELD(TriggerEvent, duration_s), 0, nullptr},
            {nullptr, 0, 0, 0, nullptr}};

        PyGetSetDef trigger_getset[] = {{"type", trigger_type, nullptr, nullptr, nullptr},
                                        {nullptr, nullptr, nullptr, nullptr, nullptr}};

        /* ============================================================== */
        /*  Rotation                                                      */
        /* ============================================================== */

        PyObject* rotation_type(PyObject*, void*)
        {
            static PyObject* name = nullptr;
            return interned(name, "rot3D");
        }

        PyObject* rotation_get_quaternion(PyObject* self, void*)
        {
            return guarded(
                [&] { return py::array_t<double>(4, unwrap<RotationEvent>(self).quaternion.data()); });
        }

        int rotation_set_quaternion(PyObject* self, PyObject* value, void*)
        {
            return guarded_set(value, [&](const py::object& given) {
                unwrap<RotationEvent>(self).quaternion = unit_quaternion(given);
            });
        }

        PyGetSetDef rotation_getset[] = {
            {"type", rotation_type, nullptr, nullptr, nullptr},
            {"quaternion", rotation_get_quaternion, rotation_set_quaternion, nullptr, nullptr},
            {nullptr, nullptr, nullptr, nullptr, nullptr}};

        /* ============================================================== */
        /*  Soft delay                                                    */
        /* ============================================================== */

        PyObject* soft_delay_type(PyObject*, void*)
        {
            static PyObject* name = nullptr;
            return interned(name, "soft_delay");
        }

        PyObject* soft_delay_get_hint(PyObject* self, void*)
        {
            const std::string& hint = unwrap<SoftDelayEvent>(self).hint;
            return PyUnicode_FromStringAndSize(hint.data(), static_cast<Py_ssize_t>(hint.size()));
        }

        int soft_delay_set_hint(PyObject* self, PyObject* value, void*)
        {
            return guarded_set(value, [&](const py::object& given) {
                unwrap<SoftDelayEvent>(self).hint = given.cast<std::string>();
            });
        }

        PyMemberDef soft_delay_members[] = {
            {"numID", Py_T_INT, PULSEQPP_FIELD(SoftDelayEvent, num), 0, nullptr},
            {"offset", Py_T_DOUBLE, PULSEQPP_FIELD(SoftDelayEvent, offset), 0, nullptr},
            {"factor", Py_T_DOUBLE, PULSEQPP_FIELD(SoftDelayEvent, factor), 0, nullptr},
            {"default_duration", Py_T_DOUBLE, PULSEQPP_FIELD(SoftDelayEvent, default_duration), 0,
             nullptr},
            {nullptr, 0, 0, 0, nullptr}};

        PyGetSetDef soft_delay_getset[] = {
            {"type", soft_delay_type, nullptr, nullptr, nullptr},
            {"hint", soft_delay_get_hint, soft_delay_set_hint, nullptr, nullptr},
            {nullptr, nullptr, nullptr, nullptr, nullptr}};

        /* ============================================================== */
        /*  Delay                                                         */
        /* ============================================================== */

        PyObject* delay_type(PyObject*, void*)
        {
            static PyObject* name = nullptr;
            return interned(name, "delay");
        }

        PyMemberDef delay_members[] = {
            {"delay", Py_T_DOUBLE, PULSEQPP_FIELD(DelayEvent, delay), 0, nullptr},
            {nullptr, 0, 0, 0, nullptr}};

        PyGetSetDef delay_getset[] = {{"type", delay_type, nullptr, nullptr, nullptr},
                                      {nullptr, nullptr, nullptr, nullptr, nullptr}};

        /* ============================================================== */
        /*  The types                                                     */
        /* ============================================================== */

        PyTypeObject RfType = {PyVarObject_HEAD_INIT(nullptr, 0)};
        PyTypeObject TrapType = {PyVarObject_HEAD_INIT(nullptr, 0)};
        PyTypeObject GradType = {PyVarObject_HEAD_INIT(nullptr, 0)};
        PyTypeObject AdcType = {PyVarObject_HEAD_INIT(nullptr, 0)};
        PyTypeObject LabelType = {PyVarObject_HEAD_INIT(nullptr, 0)};
        PyTypeObject TriggerType = {PyVarObject_HEAD_INIT(nullptr, 0)};
        PyTypeObject RotationType = {PyVarObject_HEAD_INIT(nullptr, 0)};
        PyTypeObject SoftDelayType = {PyVarObject_HEAD_INIT(nullptr, 0)};
        PyTypeObject DelayType = {PyVarObject_HEAD_INIT(nullptr, 0)};

        /**
         * Fill in and register one event type.
         *
         * Not `Py_TPFLAGS_BASETYPE`: the nine are leaves, which is what lets
         * `is_event` be a single comparison against `tp_base`.
         */
        template <typename T>
        void ready(py::module_& m, PyTypeObject& type, const char* name, PyMemberDef* members,
                   PyGetSetDef* getset, PySequenceMethods* sequence = nullptr)
        {
            static_assert(offsetof(Holder<T>, event) == EVENT_OFFSET,
                          "every event sits at the same offset in its holder");

            type.tp_name = name;
            type.tp_basicsize = sizeof(Holder<T>);
            type.tp_flags = Py_TPFLAGS_DEFAULT;
            type.tp_new = generic_new<T>;
            type.tp_dealloc = generic_dealloc<T>;
            type.tp_members = members;
            type.tp_getset = getset;
            type.tp_as_sequence = sequence;
            type.tp_base = &EventBaseType;
            if (PyType_Ready(&type) < 0)
                throw py::error_already_set();

            // The offsets above are taken through a base class, which is not
            // something the standard blesses.  One live object settles it.
            py::object probe = py::reinterpret_steal<py::object>(generic_new<T>(&type, nullptr, nullptr));
            if (!probe)
                throw py::error_already_set();
            if (reinterpret_cast<char*>(static_cast<Event*>(&unwrap<T>(probe.ptr()))) !=
                reinterpret_cast<char*>(probe.ptr()) + EVENT_OFFSET)
                throw std::runtime_error("event layout is not what PyMemberDef was told");

            const char* leaf = std::strrchr(name, '.');
            m.add_object(leaf ? leaf + 1 : name, py::reinterpret_borrow<py::object>(
                                                    reinterpret_cast<PyObject*>(&type)));
        }

        /** A fresh event of type @p T, for the converters below. */
        template <typename T>
        py::object fresh(PyTypeObject& type)
        {
            PyObject* made = generic_new<T>(&type, nullptr, nullptr);
            if (!made)
                throw py::error_already_set();
            return py::reinterpret_steal<py::object>(made);
        }

    }  // namespace

    void bind(py::module_& m)
    {
        static const char* base_name = "pulserver._ext._pulseqpp_wrapper.Event";
        EventBaseType.tp_name = base_name;
        EventBaseType.tp_basicsize = sizeof(PyObject);
        EventBaseType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        EventBaseType.tp_doc = "Base of the events `pulserver.pypulseq.make_*` returns.";
        if (PyType_Ready(&EventBaseType) < 0)
            throw py::error_already_set();
        m.add_object("Event",
                     py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject*>(
                         &EventBaseType)));

        ready<RfEvent>(m, RfType, "pulserver._ext._pulseqpp_wrapper.RfEvent", rf_members,
                       rf_getset, &rf_sequence);
        ready<TrapEvent>(m, TrapType, "pulserver._ext._pulseqpp_wrapper.TrapEvent", trap_members,
                         trap_getset);
        ready<GradEvent>(m, GradType, "pulserver._ext._pulseqpp_wrapper.GradEvent", grad_members,
                         grad_getset, &grad_sequence);
        ready<AdcEvent>(m, AdcType, "pulserver._ext._pulseqpp_wrapper.AdcEvent", adc_members,
                        adc_getset);
        ready<LabelEvent>(m, LabelType, "pulserver._ext._pulseqpp_wrapper.LabelEvent",
                          label_members, label_getset);
        ready<TriggerEvent>(m, TriggerType, "pulserver._ext._pulseqpp_wrapper.TriggerEvent",
                            trigger_members, trigger_getset);
        ready<RotationEvent>(m, RotationType, "pulserver._ext._pulseqpp_wrapper.RotationEvent",
                             nullptr, rotation_getset);
        ready<SoftDelayEvent>(m, SoftDelayType, "pulserver._ext._pulseqpp_wrapper.SoftDelayEvent",
                              soft_delay_members, soft_delay_getset);
        ready<DelayEvent>(m, DelayType, "pulserver._ext._pulseqpp_wrapper.DelayEvent",
                          delay_members, delay_getset);

        /* -- construction from PyPulseq's namespaces ----------------------- */
        //
        // One conversion, at the point `make_*` returns.  Everything after
        // that reads slots.

        m.def("_rf_from", [](const py::object& source) {
            py::object made = fresh<RfEvent>(RfType);
            RfEvent& e = unwrap<RfEvent>(made.ptr());
            auto signal = py::cast<
                py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>>(
                source.attr("signal"));
            normalise_rf(signal.data(), static_cast<int>(signal.size()), e);
            e.t = as_vector(source.attr("t"));
            e.shape_dur = source.attr("shape_dur").cast<double>();
            e.center = source.attr("center").cast<double>();
            e.delay = source.attr("delay").cast<double>();
            e.freq_offset = source.attr("freq_offset").cast<double>();
            e.phase_offset = source.attr("phase_offset").cast<double>();
            e.freq_ppm = source.attr("freq_ppm").cast<double>();
            e.phase_ppm = source.attr("phase_ppm").cast<double>();
            e.dead_time = source.attr("dead_time").cast<double>();
            e.ringdown_time = source.attr("ringdown_time").cast<double>();
            e.use = source.attr("use").cast<std::string>();
            return made;
        });

        m.def("_trap_from", [](const py::object& source) {
            py::object made = fresh<TrapEvent>(TrapType);
            TrapEvent& e = unwrap<TrapEvent>(made.ptr());
            e.amplitude = source.attr("amplitude").cast<double>();
            e.rise_time = source.attr("rise_time").cast<double>();
            e.flat_time = source.attr("flat_time").cast<double>();
            e.fall_time = source.attr("fall_time").cast<double>();
            e.delay = source.attr("delay").cast<double>();
            e.axis = axis_from(source.attr("channel").cast<std::string>());
            return made;
        });

        m.def("_grad_from", [](const py::object& source) {
            py::object made = fresh<GradEvent>(GradType);
            GradEvent& e = unwrap<GradEvent>(made.ptr());
            auto wave = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(
                source.attr("waveform"));
            normalise_grad(wave.data(), static_cast<int>(wave.size()), e);
            e.tt = as_vector(source.attr("tt"));
            e.last_time = e.tt.empty() ? 0.0 : e.tt.back();
            e.first = source.attr("first").cast<double>();
            e.last = source.attr("last").cast<double>();
            e.delay = source.attr("delay").cast<double>();
            e.axis = axis_from(source.attr("channel").cast<std::string>());
            return made;
        });

        m.def("_adc_from", [](const py::object& source) {
            py::object made = fresh<AdcEvent>(AdcType);
            AdcEvent& e = unwrap<AdcEvent>(made.ptr());
            e.num_samples = source.attr("num_samples").cast<double>();
            e.dwell = source.attr("dwell").cast<double>();
            e.delay = source.attr("delay").cast<double>();
            e.freq_offset = source.attr("freq_offset").cast<double>();
            e.phase_offset = source.attr("phase_offset").cast<double>();
            e.freq_ppm = source.attr("freq_ppm").cast<double>();
            e.phase_ppm = source.attr("phase_ppm").cast<double>();
            e.dead_time = source.attr("dead_time").cast<double>();
            if (py::hasattr(source, "phase_modulation"))
            {
                py::object modulation = source.attr("phase_modulation");
                if (!modulation.is_none())
                    e.phase_modulation = as_vector(modulation);
            }
            return made;
        });

        m.def("_label_from", [](const py::object& source) {
            py::object made = fresh<LabelEvent>(LabelType);
            LabelEvent& e = unwrap<LabelEvent>(made.ptr());
            e.value = static_cast<int32_t>(source.attr("value").cast<double>());
            e.label = source.attr("label").cast<std::string>();
            e.setting = source.attr("type").cast<std::string>() == "labelset";
            return made;
        });

        m.def("_trigger_from", [](const py::object& source) {
            py::object made = fresh<TriggerEvent>(TriggerType);
            TriggerEvent& e = unwrap<TriggerEvent>(made.ptr());
            const std::string kind = source.attr("type").cast<std::string>();
            const std::string channel = source.attr("channel").cast<std::string>();
            if (kind == "trigger")
            {
                e.control = 2.0;
                e.channel = channel == "physio2" ? 2.0 : 1.0;
            }
            else
            {
                e.control = 1.0;
                e.channel = channel == "osc1" ? 2.0 : (channel == "ext1" ? 3.0 : 1.0);
            }
            e.delay = source.attr("delay").cast<double>();
            e.duration_s = source.attr("duration").cast<double>();
            return made;
        });

        m.def("_rotation_from", [](const py::object& source) {
            py::object made = fresh<RotationEvent>(RotationType);
            unwrap<RotationEvent>(made.ptr()).quaternion =
                unit_quaternion(source.attr("rot_quaternion")
                                    .attr("as_quat")(py::arg("canonical") = true,
                                                     py::arg("scalar_first") = true));
            return made;
        });

        m.def("_soft_delay_from", [](const py::object& source) {
            py::object made = fresh<SoftDelayEvent>(SoftDelayType);
            SoftDelayEvent& e = unwrap<SoftDelayEvent>(made.ptr());
            e.num = static_cast<int32_t>(source.attr("numID").cast<double>());
            e.offset = source.attr("offset").cast<double>();
            e.factor = source.attr("factor").cast<double>();
            e.hint = source.attr("hint").cast<std::string>();
            if (py::hasattr(source, "default_duration"))
            {
                py::object value = source.attr("default_duration");
                if (!value.is_none())
                    e.default_duration = value.cast<double>();
            }
            return made;
        });

        m.def("_delay_from", [](const py::object& source) {
            py::object made = fresh<DelayEvent>(DelayType);
            unwrap<DelayEvent>(made.ptr()).delay = source.attr("delay").cast<double>();
            return made;
        });
    }

}  // namespace pulseqpp_types
