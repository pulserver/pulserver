/**
 * @file _pulseqpp_wrapper.cpp
 * @brief Python binding for pulseq::Sequence.  Private, and deliberately thin.
 *
 * Nothing here has an opinion about the API.  Argument names, defaults,
 * docstrings and type hints belong to the Python wrapper
 * (pulserver/pypulseq/_sequence.py); this file only moves numbers across, and
 * the docstrings below are one line each because a second copy of the real
 * documentation would be a second thing to keep true.
 *
 * The library tables cross as whole arrays rather than row by row.  A composed
 * scan already holds its libraries as dense NumPy arrays -- that is what it was
 * built as -- so handing them over is a memcpy, where a per-row loop would be
 * millions of round trips through the interpreter to arrive at the same bytes.
 *
 * Naming: `_pulseqpp_`, not `_pulseq_`, because `_pulseg_wrapper` already
 * exists one letter away and the two do entirely different jobs.
 */

#include <array>
#include <map>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "_pulseqpp_eventtypes.h"
#include "_pulseqpp_events.h"

#include "pulseq/autolabel.hpp"
#include "pulseq/expand.hpp"
#include "pulseq/kspace.hpp"
#include "pulseq/moments.hpp"
#include "pulseq/read.hpp"
#include "pulseq/sequence.hpp"
#include "pulseq/trajectory.hpp"
#include "pulseq/write.hpp"

#include <cstring>
#include <memory>
#include <stdexcept>

namespace py = pybind11;

// The class Python sees is pulseq::Sequence plus the shape identity cache the
// event path needs; see _pulseqpp_events.h.  Everything below is written
// against the base and works on either.
using Sequence = pulseqpp_events::BoundSequence;

namespace
{

    /** A contiguous (N, width) float64 view, or an error naming the offender. */
    py::array_t<double, py::array::c_style | py::array::forcecast>
    as_matrix(const py::object& source, int width, const char* what)
    {
        auto array = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(source);
        if (array.ndim() != 2 || array.shape(1) != width)
        {
            throw std::invalid_argument(
                std::string(what) + " must have shape (N, " + std::to_string(width) + ")");
        }
        return array;
    }

    py::array_t<int32_t, py::array::c_style | py::array::forcecast>
    as_int_matrix(const py::object& source, int width, const char* what)
    {
        auto array =
            py::cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>(source);
        if (array.ndim() != 2 || array.shape(1) != width)
        {
            throw std::invalid_argument(
                std::string(what) + " must have shape (N, " + std::to_string(width) + ")");
        }
        return array;
    }

    /** Replace a fixed-width table wholesale. */
    void fill_table(pulseq::Table& table, const py::object& source, const char* what)
    {
        auto array = as_matrix(source, table.width(), what);
        const auto rows = static_cast<int>(array.shape(0));
        table.resize(rows);
        if (rows)
            std::memcpy(table.data(), array.data(), sizeof(double) * rows * table.width());
    }

    void fill_int_table(pulseq::IntTable& table, const py::object& source, const char* what)
    {
        auto array = as_int_matrix(source, table.width(), what);
        const auto rows = static_cast<int>(array.shape(0));
        table.resize(rows);
        if (rows)
            std::memcpy(table.data(), array.data(), sizeof(int32_t) * rows * table.width());
    }

    /* -- definitions -------------------------------------------------- */
    //
    // Deliberately two entry points rather than one that sniffs the type.
    // Deciding whether a definition is text, whole numbers or reals means
    // knowing about `str`, Python ints, NumPy scalars, NumPy arrays and the
    // empty case -- which is Python's job, done in Python, where getting it
    // wrong raises instead of corrupting the heap.

    /**
     * `add_block(*events)`, called without building a tuple.
     *
     * The one call a design loop makes per block, so it is the one place
     * where pybind11's argument handling is worth going around: METH_FASTCALL
     * hands the arguments over as a C array, which is what `build_block`
     * wanted in the first place.
     */
    PyObject* add_block_fast(PyObject* self, PyObject* const* args, Py_ssize_t nargs)
    {
        try
        {
            Sequence& seq = py::cast<Sequence&>(py::handle(self));
            const int index = seq.add_block(pulseqpp_events::build_block(seq, args, nargs));
            return PyLong_FromLong(index);
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

    PyMethodDef add_block_fast_def = {
        "add_block_events",
        reinterpret_cast<PyCFunction>(reinterpret_cast<void*>(add_block_fast)), METH_FASTCALL,
        PyDoc_STR("add_block_events(*events) -> int")};

    py::object definition_to(const pulseq::Definition& def)
    {
        if (def.kind() == pulseq::Definition::Kind::Text)
            return py::str(def.text());

        py::list values;
        for (double value : def.numbers())
        {
            if (def.kind() == pulseq::Definition::Kind::Int)
                values.append(py::int_(static_cast<long long>(value)));
            else
                values.append(py::float_(value));
        }
        return values;
    }

}  // namespace

PYBIND11_MODULE(_pulseqpp_wrapper, m)
{
    m.doc() = "Private binding for pulseq::Sequence. See pulserver.pypulseq.";

    pulseqpp_types::bind(m);

    auto sequence_class = py::class_<Sequence>(m, "Sequence")
        .def(py::init<>())

        /* -- copying ---------------------------------------------------- */
        //
        // Everything the sequence proper holds is a plain container, so this
        // is one allocation and one memcpy per library (see sequence.hpp).
        //
        // What is NOT copied is the event path's caches: the new object gets
        // a fresh `serial` and empty `shape_ids`/`quaternions`.  Those are
        // memoized by the identity of a Python object, and an event carrying
        // shape ids issued to the original must not have them believed here --
        // `remove_duplicates()` renumbers, so the two sequences' ids diverge
        // the moment either is deduplicated.  A copy re-registers on first
        // use, which costs a pass over a waveform and cannot be wrong.
        .def("copy",
             [](const Sequence& self) {
                 auto made = std::make_unique<Sequence>();
                 static_cast<pulseq::Sequence&>(*made) =
                     static_cast<const pulseq::Sequence&>(self);
                 return made;
             })

        /* -- version and rasters -------------------------------------- */
        .def_property_readonly("version_major", &Sequence::version_major)
        .def_property_readonly("version_minor", &Sequence::version_minor)
        .def_property_readonly("version_revision", &Sequence::version_revision)
        .def("set_version", &Sequence::set_version)
        .def_property_readonly("rf_raster_time", &Sequence::rf_raster_time)
        .def_property_readonly("grad_raster_time", &Sequence::grad_raster_time)
        .def_property_readonly("adc_raster_time", &Sequence::adc_raster_time)
        .def_property_readonly("block_duration_raster", &Sequence::block_duration_raster)
        .def("set_rasters", &Sequence::set_rasters)

        /* -- definitions ---------------------------------------------- */
        .def("set_definition_text",
             [](Sequence& self, const std::string& key, const std::string& value) {
                 self.set_definition(key, pulseq::Definition(value));
             })
        .def("set_definition_numbers",
             [](Sequence& self, const std::string& key, const std::vector<double>& values,
                bool integers) {
                 self.set_definition(key, integers
                                              ? pulseq::Definition::integers(values)
                                              : pulseq::Definition(values));
             })
        .def("get_definition",
             [](const Sequence& self, const std::string& key) -> py::object {
                 const pulseq::Definition* def = self.definition(key);
                 return def ? definition_to(*def) : py::none();
             })
        .def("definitions",
             [](const Sequence& self) {
                 py::dict out;
                 for (const auto& entry : self.definitions())
                     out[py::str(entry.first)] = definition_to(entry.second);
                 return out;
             })

        /* -- registries ------------------------------------------------ */
        .def("extension_type_id", &Sequence::extension_type_id)
        .def("find_extension_type_id", &Sequence::find_extension_type_id)
        .def("set_extension_type_id", &Sequence::set_extension_type_id)
        .def("extension_type_name", &Sequence::extension_type_name)
        .def("label_id", &Sequence::label_id)
        .def("find_label_id", &Sequence::find_label_id)
        .def("label_name", &Sequence::label_name)
        .def("is_custom_label", &Sequence::is_custom_label)
        .def_static("builtin_labels", &Sequence::builtin_labels)

        /* -- bulk library loading -------------------------------------- */
        //
        // One call per library, each taking the whole thing.  This is the path
        // a composed scan uses: its libraries are already dense arrays, so the
        // transfer is a memcpy rather than millions of interpreter round trips.
        .def("set_rf",
             [](Sequence& self, const py::object& rows, const std::string& uses) {
                 fill_table(self.rf_library(), rows, "rf");
                 if (static_cast<int>(uses.size()) != self.rf_library().size())
                     throw std::invalid_argument("one use character per RF row is required");
                 self.rf_uses().assign(uses.begin(), uses.end());
             })
        .def("set_gradients",
             [](Sequence& self, const py::object& traps, const py::object& arbs,
                const py::object& slots) {
                 // The two tables plus the map from the shared id onto them.
                 // A positive slot is a trapezoid row, a negative one an
                 // arbitrary row, both 1-based.
                 fill_table(self.trap_library(), traps, "traps");
                 fill_table(self.arb_library(), arbs, "arbitrary gradients");
                 auto array =
                     py::cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>(
                         slots);
                 if (array.ndim() != 1)
                     throw std::invalid_argument("gradient slots must be one-dimensional");
                 self.set_grad_slots(array.data(), static_cast<int>(array.shape(0)));
             })
        .def("set_adc",
             [](Sequence& self, const py::object& rows) {
                 fill_table(self.adc_library(), rows, "adc");
             })
        .def("set_triggers",
             [](Sequence& self, const py::object& rows) {
                 fill_table(self.trigger_library(), rows, "triggers");
             })
        .def("set_rotations",
             [](Sequence& self, const py::object& rows) {
                 fill_table(self.rotation_library(), rows, "rotations");
             })
        .def("set_extensions",
             [](Sequence& self, const py::object& rows) {
                 fill_int_table(self.extensions_library(), rows, "extension chains");
             })
        .def("set_label_set",
             [](Sequence& self, const py::object& rows) {
                 fill_int_table(self.label_set_library(), rows, "label set");
             })
        .def("set_label_inc",
             [](Sequence& self, const py::object& rows) {
                 fill_int_table(self.label_inc_library(), rows, "label inc");
             })
        .def("set_shapes",
             [](Sequence& self, const py::object& lengths, const py::object& starts,
                const py::object& samples) {
                 auto n = py::cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>(
                     lengths);
                 auto s = py::cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>(
                     starts);
                 auto d =
                     py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(
                         samples);
                 self.set_shapes(n.data(), static_cast<int>(n.shape(0)), s.data(), d.data());
             })
        .def("set_rf_shims",
             [](Sequence& self, const py::object& starts, const py::object& values) {
                 auto s = py::cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>(
                     starts);
                 auto d =
                     py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(
                         values);
                 self.set_rf_shims(s.data(), static_cast<int>(s.shape(0)) - 1, d.data());
             })
        .def("set_soft_delays",
             [](Sequence& self, const std::vector<int32_t>& nums,
                const std::vector<double>& offsets, const std::vector<double>& factors,
                const std::vector<std::string>& hints) {
                 self.soft_delay_library().clear();
                 for (size_t i = 0; i < nums.size(); ++i)
                 {
                     pulseq::SoftDelay row;
                     row.num = nums[i];
                     row.offset = offsets[i];
                     row.factor = factors[i];
                     row.hint = hints[i];
                     self.register_soft_delay(row);
                 }
             })

        /* -- blocks ----------------------------------------------------- */
        .def("set_blocks",
             [](Sequence& self, const py::array& events, const py::object& durations) {
                 // Six columns or Pulseq's seven -- the extra leading one is
                 // the legacy delay id, which is not written.  Taking the wide
                 // table as it stands means the narrowing happens in the copy
                 // this makes anyway, rather than in a NumPy slice that would
                 // pass over sixty megabytes to arrive at the same rows.
                 if (events.ndim() != 2)
                     throw std::invalid_argument("the block table must be two-dimensional");
                 const auto columns = static_cast<int>(events.shape(1));
                 if (columns != pulseq::BLOCK_WIDTH && columns != pulseq::BLOCK_WIDTH + 1)
                 {
                     throw std::invalid_argument(
                         "the block table must have 6 or 7 columns, not " +
                         std::to_string(columns));
                 }

                 auto d =
                     py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(
                         durations);
                 const auto rows = static_cast<int>(events.shape(0));
                 if (d.ndim() != 1 || static_cast<int>(d.shape(0)) != rows)
                     throw std::invalid_argument("one duration per block is required");

                 if (columns == pulseq::BLOCK_WIDTH)
                 {
                     auto e = as_int_matrix(events, pulseq::BLOCK_WIDTH, "block events");
                     self.set_blocks(e.data(), d.data(), rows);
                     return;
                 }

                 auto wide = py::cast<
                     py::array_t<int32_t, py::array::c_style | py::array::forcecast>>(events);
                 std::vector<int32_t> narrow(static_cast<size_t>(rows) * pulseq::BLOCK_WIDTH);
                 const int32_t* source = wide.data();
                 for (int row = 0; row < rows; ++row)
                 {
                     std::memcpy(narrow.data() + static_cast<size_t>(row) * pulseq::BLOCK_WIDTH,
                                 source + static_cast<size_t>(row) * (pulseq::BLOCK_WIDTH + 1) + 1,
                                 sizeof(int32_t) * pulseq::BLOCK_WIDTH);
                 }
                 self.set_blocks(narrow.data(), d.data(), rows);
             })
        /* -- registration, one event at a time ---------------------------- */
        /*
         * The counterpart to the bulk setters above: a sequence built block by
         * block registers each event as it comes.  Rows arrive as sequences of
         * numbers rather than arrays -- a caller here is holding six doubles,
         * not a matrix, and making a NumPy array per event would cost more
         * than the call.
         */
        .def("register_rf",
             [](Sequence& self, const std::array<double, pulseq::RF_WIDTH>& row, char use) {
                 return self.register_rf(row.data(), use);
             })
        .def("register_trap",
             [](Sequence& self, const std::array<double, pulseq::TRAP_WIDTH>& row) {
                 return self.register_trap(row.data());
             })
        .def("register_arbitrary",
             [](Sequence& self, const std::array<double, pulseq::ARB_WIDTH>& row) {
                 return self.register_arbitrary(row.data());
             })
        .def("register_adc",
             [](Sequence& self, const std::array<double, pulseq::ADC_WIDTH>& row) {
                 return self.register_adc(row.data());
             })
        .def("register_trigger",
             [](Sequence& self, const std::array<double, pulseq::TRIGGER_WIDTH>& row) {
                 return self.register_trigger(row.data());
             })
        .def("register_rotation",
             [](Sequence& self, const std::array<double, pulseq::ROTATION_WIDTH>& row) {
                 return self.register_rotation(row.data());
             })
        .def("register_label_set", &Sequence::register_label_set)
        .def("register_label_inc", &Sequence::register_label_inc)
        .def("register_rf_shim",
             [](Sequence& self, const std::vector<double>& values) {
                 return self.register_rf_shim(values.data(), static_cast<int>(values.size()));
             })
        .def("register_soft_delay",
             [](Sequence& self, int32_t num, double offset, double factor,
                const std::string& hint) {
                 pulseq::SoftDelay row;
                 row.num = num;
                 row.offset = offset;
                 row.factor = factor;
                 row.hint = hint;
                 return self.register_soft_delay(row);
             })
        .def("register_shape",
             [](Sequence& self, int num_uncompressed,
                py::array_t<double, py::array::c_style | py::array::forcecast> samples) {
                 const py::buffer_info info = samples.request();
                 return self.register_shape(num_uncompressed,
                                            static_cast<const double*>(info.ptr),
                                            static_cast<int>(info.size));
             })
        .def("chain_extension", &Sequence::chain_extension)
        .def("append_extension", &Sequence::append_extension)

        /* -- the ergonomic path ------------------------------------------ */
        //
        // `add_block(*events)` with the objects PyPulseq's `make_*` return.
        // Every event is unpacked and registered here, so a block costs one
        // call rather than one per event.  See _pulseqpp_events.h.  It is
        // attached below rather than here, because it is METH_FASTCALL.
        //
        // The same registration, written over a block that already exists.
        // Rows the old block referred to are left where they are: they may be
        // shared, and deduplication is what decides what survives.
        .def("set_block_events",
             [](Sequence& self, int index, const py::args& events) {
                 PyObject* const* items = &PyTuple_GET_ITEM(events.ptr(), 0);
                 self.set_block(index,
                                pulseqpp_events::build_block(self, items, PyTuple_GET_SIZE(events.ptr())));
             })
        // Register one event's shapes without adding a block; returns the
        // shape ids.  Backs Sequence.register_*_event -- see _pulseqpp_events.h.
        .def("warm_event",
             [](Sequence& self, const py::handle& event) {
                 return pulseqpp_events::warm_event(self, event);
             })

        // Measurement only: takes events and does nothing with them, so the
        // difference against add_block_events is what unpacking and
        // registering them costs.  Deliberately a py::args binding, which is
        // also what it measures the cost of.
        .def("_bench_noop", [](Sequence&, const py::args& events) {
            return static_cast<int>(events.size());
        })
        .def("compress_shapes", &Sequence::compress_shapes,
             py::call_guard<py::gil_scoped_release>())
        .def("register_raw_shape",
             [](Sequence& self,
                py::array_t<double, py::array::c_style | py::array::forcecast> samples) {
                 const py::buffer_info info = samples.request();
                 return self.register_raw_shape(static_cast<const double*>(info.ptr),
                                                static_cast<int>(info.size));
             })

        .def("num_blocks", &Sequence::num_blocks)
        .def("duration", &Sequence::duration)
        .def("add_block",
             [](Sequence& self, int32_t rf, int32_t gx, int32_t gy, int32_t gz, int32_t adc,
                int32_t ext, double duration) {
                 pulseq::Block block{rf, gx, gy, gz, adc, ext, duration};
                 return self.add_block(block);
             })
        .def("set_block",
             [](Sequence& self, int index, int32_t rf, int32_t gx, int32_t gy, int32_t gz,
                int32_t adc, int32_t ext, double duration) {
                 pulseq::Block block{rf, gx, gy, gz, adc, ext, duration};
                 self.set_block(index, block);
             })
        .def("get_block",
             [](const Sequence& self, int index) {
                 const pulseq::Block block = self.get_block(index);
                 return py::make_tuple(block.rf, block.gx, block.gy, block.gz, block.adc,
                                       block.ext, block.duration);
             })
        .def("block_events",
             [](Sequence& self) {
                 // A view, not a copy: the caller reads columns out of it and
                 // the sequence outlives the call.
                 return py::array_t<int32_t>(
                     {self.num_blocks(), pulseq::BLOCK_WIDTH}, self.block_events(),
                     py::cast(&self, py::return_value_policy::reference));
             })
        .def("block_durations",
             [](Sequence& self) {
                 return py::array_t<double>({self.num_blocks()}, self.block_durations(),
                                            py::cast(&self, py::return_value_policy::reference));
             })
        /* Soft delays rewrite a block's duration and nothing else about it,
           so they get a scalar setter rather than a rebuild of the block. */
        .def("set_block_duration",
             [](Sequence& self, int index, double seconds) {
                 if (index < 1 || index > self.num_blocks())
                     throw py::index_error("block index out of range");
                 self.block_durations()[index - 1] = seconds;
             },
             py::arg("index"), py::arg("seconds"))

        /* -- library sizes, for the Python side to report --------------- */
        .def("num_gradients", &Sequence::num_gradients)
        .def("num_rf", [](const Sequence& s) { return s.rf_library().size(); })
        .def("num_adc", [](const Sequence& s) { return s.adc_library().size(); })
        .def("num_shapes", [](const Sequence& s) { return s.shape_library().size(); })
        .def("num_extensions", [](const Sequence& s) { return s.extensions_library().size(); })
        .def("num_rotations", [](const Sequence& s) { return s.rotation_library().size(); })
        .def("num_rf_shims", [](const Sequence& s) { return s.rf_shim_library().size(); })
        .def("num_triggers", [](const Sequence& s) { return s.trigger_library().size(); })
        .def("num_label_set", [](const Sequence& s) { return s.label_set_library().size(); })
        .def("num_label_inc", [](const Sequence& s) { return s.label_inc_library().size(); })
        .def("num_soft_delays",
             [](const Sequence& s) { return static_cast<int>(s.soft_delay_library().size()); })

        /* -- library readback, for get_block and the native window ------ */
        .def("rf_row", [](const Sequence& s,
                          int id) { return py::array_t<double>(pulseq::RF_WIDTH, s.rf_library().row(id)); })
        .def("rf_use", [](const Sequence& s, int id) {
            return std::string(1, s.rf_uses()[static_cast<size_t>(id) - 1]);
        })
        .def("grad_kind",
             [](const Sequence& s, int id) {
                 switch (s.grad_kind(id))
                 {
                 case pulseq::GradKind::Trap:
                     return "trap";
                 case pulseq::GradKind::Arbitrary:
                     return "grad";
                 default:
                     return "";
                 }
             })
        .def("grad_row",
             [](const Sequence& s, int id) {
                 const int row = s.grad_row(id);
                 if (s.grad_kind(id) == pulseq::GradKind::Trap)
                     return py::array_t<double>(pulseq::TRAP_WIDTH, s.trap_library().row(row));
                 return py::array_t<double>(pulseq::ARB_WIDTH, s.arb_library().row(row));
             })
        .def("adc_row", [](const Sequence& s, int id) {
            return py::array_t<double>(pulseq::ADC_WIDTH, s.adc_library().row(id));
        })
        .def("extension_row",
             [](const Sequence& s, int id) {
                 const int32_t* row = s.extensions_library().row(id);
                 return py::make_tuple(row[0], row[1], row[2]);
             })
        .def("rotation_row", [](const Sequence& s, int id) {
            return py::array_t<double>(pulseq::ROTATION_WIDTH, s.rotation_library().row(id));
        })
        .def("rf_shim_row",
             [](const Sequence& s, int id) {
                 return py::array_t<double>(s.rf_shim_library().length(id),
                                            s.rf_shim_library().row(id));
             })
        .def(
            "label_evolution",
            [](const Sequence& s, int first, int last, const std::string& mode) {
                /* Replay the sticky label state across a block range, in C++.
                 * The Python equivalent rebuilt every block as a namespace to
                 * read two integers off its extension chain, which is a walk
                 * the size of the scan for an answer the size of the ADC
                 * count. */
                const int labelset = s.find_extension_type_id("LABELSET");
                const int labelinc = s.find_extension_type_id("LABELINC");
                const bool per_adc = (mode == "adc");
                const bool per_block = (mode == "blocks");
                const bool per_label = (mode == "label");

                std::map<int, int> state;
                std::map<int, std::vector<int>> evolution;
                std::vector<int> order;
                int steps = 0;

                auto record = [&]() {
                    for (const auto& entry : state)
                    {
                        std::vector<int>& column = evolution[entry.first];
                        if (column.empty())
                            order.push_back(entry.first);
                        column.resize(static_cast<size_t>(steps), 0);
                        column.push_back(entry.second);
                    }
                    ++steps;
                };

                for (int index = first; index <= last; ++index)
                {
                    const int32_t* block =
                        s.block_events() + static_cast<size_t>(index - 1) * pulseq::BLOCK_WIDTH;
                    bool touched = false;
                    int32_t link = block[5]; /* ext */
                    while (link > 0)
                    {
                        const int32_t* ext = s.extensions_library().row(link);
                        if (ext[0] == labelset && ext[1] > 0)
                        {
                            const int32_t* row = s.label_set_library().row(ext[1]);
                            state[row[1]] = row[0];
                            touched = true;
                        }
                        else if (ext[0] == labelinc && ext[1] > 0)
                        {
                            const int32_t* row = s.label_inc_library().row(ext[1]);
                            state[row[1]] += row[0];
                            touched = true;
                        }
                        link = ext[2];
                    }
                    if (per_block || (per_label && touched) ||
                        (per_adc && block[4] > 0)) /* adc */
                        record();
                }

                py::dict out;
                if (steps == 0)
                {
                    for (const auto& entry : state)
                        out[py::str(s.label_name(entry.first))] = entry.second;
                    return out;
                }
                for (int id : order)
                {
                    std::vector<int>& column = evolution[id];
                    column.resize(static_cast<size_t>(steps), 0);
                    out[py::str(s.label_name(id))] =
                        py::array_t<int>(steps, column.data());
                }
                return out;
            },
            py::arg("first"),
            py::arg("last"),
            py::arg("mode"))
        // Does any LABELSET row write this label?  A library holds one row
        // per *use* until deduplication runs, so asking from Python would be
        // a call per shot on a sequence that has not been collapsed yet.
        .def("label_set_writes",
             [](const Sequence& s, int32_t label_id) {
                 const auto& library = s.label_set_library();
                 for (int id = 1; id <= library.size(); ++id)
                     if (library.row(id)[1] == label_id)
                         return true;
                 return false;
             })
        .def("label_set_row",
             [](const Sequence& s, int id) {
                 const int32_t* row = s.label_set_library().row(id);
                 return py::make_tuple(row[0], row[1]);
             })
        .def("label_inc_row",
             [](const Sequence& s, int id) {
                 const int32_t* row = s.label_inc_library().row(id);
                 return py::make_tuple(row[0], row[1]);
             })
        .def("trigger_row", [](const Sequence& s, int id) {
            return py::array_t<double>(pulseq::TRIGGER_WIDTH, s.trigger_library().row(id));
        })
        .def("soft_delay_row",
             [](const Sequence& s, int id) {
                 const pulseq::SoftDelay& row = s.soft_delay_library()[static_cast<size_t>(id) - 1];
                 return py::make_tuple(row.num, row.offset, row.factor, row.hint);
             })
        .def("shape_row",
             [](const Sequence& s, int id) {
                 return py::make_tuple(
                     s.shape_library().num_uncompressed(id),
                     py::array_t<double>(s.shape_library().num_compressed(id),
                                         s.shape_library().samples(id)));
             })

        // Nothing in the pass touches Python, and on a scan built block by
        // block it is the longest call in the binding.
        .def("remove_duplicates", &Sequence::remove_duplicates,
             py::call_guard<py::gil_scoped_release>())

        // Repetitions resolved into the block table.  Scan-length like the
        // pass above it, and touching Python no more than that one does.
        .def("expand_repeats",
             [](Sequence& self, int repeats, const std::string& label, bool strip_once,
                bool set_ignore_averages) {
                 pulseq::ExpandOptions options;
                 options.label = label;
                 options.strip_once = strip_once;
                 options.set_ignore_averages = set_ignore_averages;

                 pulseq::ExpandResult r;
                 {
                     py::gil_scoped_release unlocked;
                     r = pulseq::expand_repeats(self, repeats, options);
                 }

                 py::dict out;
                 out["repeats"] = r.repeats;
                 out["blocks_before"] = r.blocks_before;
                 out["blocks_after"] = r.blocks_after;
                 out["prep_blocks"] = r.prep_blocks;
                 out["body_blocks"] = r.body_blocks;
                 out["cooldown_blocks"] = r.cooldown_blocks;
                 return out;
             })

        /* -- files ------------------------------------------------------ */
        .def("write_text",
             [](Sequence& self, bool create_signature) {
                 return py::bytes(pulseq::write_text(self, create_signature));
             })
        .def("write_binary",
             [](Sequence& self) { return py::bytes(pulseq::write_binary(self)); })
        .def("required_revision", [](const Sequence& s) { return pulseq::required_revision(s); })

        /* -- FOV positioning -------------------------------------------- */
        //
        // The shift is in LOGICAL coordinates, which is what lets it ignore
        // rotation entirely: `dr . k` is invariant when both are rotated.  The
        // caller converts a prescribed physical offset once, with
        // `dr_logical = R.T @ dr_physical`.
        // `first`/`last` are 1-based inclusive and bound only which blocks are
        // modified; `last=0` means "to the end".  Absolute k is accumulated
        // from block 1 either way.
        // `scope` picks who finishes the ADC side: "native" bakes every
        // readout, "server" bakes only the Cartesian unrotated ones and
        // leaves the rest to the consumer of the base trajectory, "rf_only"
        // bakes none.
        .def(
            "apply_fov_shift",
            [](Sequence& self, double dx, double dy, double dz, const std::string& scope,
               int first, int last) {
                pulseq::FovShiftScope value;
                if (scope == "native")
                    value = pulseq::FovShiftScope::RfAndAdc;
                else if (scope == "server")
                    value = pulseq::FovShiftScope::Server;
                else if (scope == "rf_only")
                    value = pulseq::FovShiftScope::RfOnly;
                else
                    throw py::value_error(
                        "apply_fov_shift(): scope must be 'native', 'server' or "
                        "'rf_only', got '" +
                        scope + "'");
                pulseq::apply_fov_shift(self, {dx, dy, dz}, value, first, last);
            },
            py::arg("dx"), py::arg("dy"), py::arg("dz"), py::arg("scope") = "native",
            py::arg("first") = 1, py::arg("last") = 0,
            py::call_guard<py::gil_scoped_release>())

        // Per-axis gradient scaling over a block range.  Multiplies the
        // amplitude a gradient row carries; the shape it points at is left
        // alone, so this registers no shapes.
        .def(
            "apply_fov_scale",
            [](Sequence& self, double sx, double sy, double sz, int first, int last) {
                pulseq::apply_fov_scale(self, {sx, sy, sz}, first, last);
            },
            py::arg("sx"), py::arg("sy"), py::arg("sz"), py::arg("first") = 1,
            py::arg("last") = 0, py::call_guard<py::gil_scoped_release>())

        // Server mode: store each readout's base trajectory in its ADC's
        // phase_modulation, for a consumer of ours to rescale and rotate.
        .def(
            "attach_base_trajectory",
            [](Sequence& self) { pulseq::attach_base_trajectory(self); },
            py::call_guard<py::gil_scoped_release>())
        .def("has_base_trajectory",
             [](const Sequence& s) { return pulseq::has_base_trajectory(s); })

        .def("block_k_origins",
             [](Sequence& s) {
                 const std::vector<std::array<double, 3>> origins = pulseq::block_k_origins(s);
                 py::array_t<double> out({static_cast<int>(origins.size()), 3});
                 double* dst = static_cast<double*>(out.request().ptr);
                 for (size_t b = 0; b < origins.size(); ++b)
                     for (int a = 0; a < 3; ++a)
                         dst[b * 3 + static_cast<size_t>(a)] = origins[b][static_cast<size_t>(a)];
                 return out;
             })

        // (3, num_samples) absolute k for one readout, or None where the block
        // has no ADC.  Axes the block does not drive come back as zeros.
        .def("absolute_trajectory",
             [](const Sequence& s, int block_index, double kx, double ky, double kz)
                 -> py::object {
                 const std::array<std::vector<double>, 3> k =
                     pulseq::absolute_trajectory(s, block_index, {kx, ky, kz});

                 size_t n = 0;
                 for (int a = 0; a < 3; ++a)
                     n = std::max(n, k[static_cast<size_t>(a)].size());
                 if (n == 0)
                     return py::none();

                 py::array_t<double> out({3, static_cast<int>(n)});
                 double* dst = static_cast<double*>(out.request().ptr);
                 std::fill(dst, dst + 3 * n, 0.0);
                 for (int a = 0; a < 3; ++a)
                 {
                     const std::vector<double>& src = k[static_cast<size_t>(a)];
                     for (size_t i = 0; i < src.size(); ++i)
                         dst[static_cast<size_t>(a) * n + i] = src[i];
                 }
                 return out;
             })

        // The k-space trajectory.  Returns a dict rather than a tuple: the
        // Python wrapper assembles upstream's five-tuple from it and would
        // otherwise have to remember a positional order.
        .def(
            "calculate_kspace",
            [](Sequence& self, std::array<double, 3> trajectory_delay,
               std::array<double, 3> gradient_offset, int first_block, int last_block,
               bool apply_rotation, bool sample_window_average, bool dense) {
                pulseq::KSpaceOptions options;
                options.trajectory_delay = trajectory_delay;
                options.gradient_offset = gradient_offset;
                options.first_block = first_block;
                options.last_block = last_block;
                options.apply_rotation = apply_rotation;
                options.sample_window_average = sample_window_average;
                options.dense = dense;

                pulseq::KSpace ks;
                {
                    py::gil_scoped_release unlocked;
                    ks = pulseq::calculate_kspace(self, options);
                }

                const auto rf_times = [](const std::vector<pulseq::RfEventTiming>& events) {
                    py::array_t<double> out(static_cast<int>(events.size()));
                    double* dst = static_cast<double*>(out.request().ptr);
                    for (size_t i = 0; i < events.size(); ++i)
                        dst[i] = events[i].t;
                    return out;
                };

                // (3, n) throughout: upstream's orientation, and the one the
                // stored layout already is.
                py::array_t<double> k_adc({3, ks.total_samples});
                {
                    double* dst = static_cast<double*>(k_adc.request().ptr);
                    const size_t n = static_cast<size_t>(ks.total_samples);
                    for (const pulseq::Readout& r : ks.readouts)
                    {
                        if (r.num_samples <= 0)
                            continue;
                        const double* src =
                            ks.k_adc.data() + static_cast<size_t>(r.sample_offset) * 3;
                        for (int a = 0; a < 3; ++a)
                            std::memcpy(dst + static_cast<size_t>(a) * n +
                                            static_cast<size_t>(r.sample_offset),
                                        src + static_cast<size_t>(a) *
                                                  static_cast<size_t>(r.num_samples),
                                        static_cast<size_t>(r.num_samples) * sizeof(double));
                    }
                }

                const int dense_n = static_cast<int>(ks.t_ktraj.size());
                py::array_t<double> k_traj({3, dense_n});
                if (dense_n > 0)
                    std::memcpy(k_traj.request().ptr, ks.k_ktraj.data(),
                                ks.k_ktraj.size() * sizeof(double));

                py::array_t<double> slice_pos({3, static_cast<int>(ks.slice_pos.size())});
                {
                    double* dst = static_cast<double*>(slice_pos.request().ptr);
                    const size_t n = ks.slice_pos.size();
                    for (size_t i = 0; i < n; ++i)
                        for (int a = 0; a < 3; ++a)
                            dst[static_cast<size_t>(a) * n + i] =
                                ks.slice_pos[i][static_cast<size_t>(a)];
                }

                py::array_t<int> readout_block(static_cast<int>(ks.readouts.size()));
                py::array_t<int> readout_samples(static_cast<int>(ks.readouts.size()));
                py::array_t<int> readout_center(static_cast<int>(ks.readouts.size()));
                py::array_t<int> readout_rotation(static_cast<int>(ks.readouts.size()));
                {
                    int* b = static_cast<int*>(readout_block.request().ptr);
                    int* n = static_cast<int*>(readout_samples.request().ptr);
                    int* c = static_cast<int*>(readout_center.request().ptr);
                    int* r = static_cast<int*>(readout_rotation.request().ptr);
                    for (size_t i = 0; i < ks.readouts.size(); ++i)
                    {
                        b[i] = ks.readouts[i].block_index;
                        n[i] = ks.readouts[i].num_samples;
                        c[i] = ks.readouts[i].center_sample;
                        r[i] = ks.readouts[i].rotation_id;
                    }
                }

                py::dict out;
                out["k_adc"] = k_adc;
                out["t_adc"] = py::array_t<double>(static_cast<int>(ks.t_adc.size()),
                                                   ks.t_adc.data());
                out["k_traj"] = k_traj;
                out["t_ktraj"] = py::array_t<double>(dense_n, ks.t_ktraj.data());
                out["t_excitation"] = rf_times(ks.excitations);
                out["t_refocusing"] = rf_times(ks.refocusings);
                out["slice_pos"] = slice_pos;
                out["t_slice_pos"] = py::array_t<double>(
                    static_cast<int>(ks.t_slice_pos.size()), ks.t_slice_pos.data());
                out["readout_block"] = readout_block;
                out["readout_samples"] = readout_samples;
                out["readout_center_sample"] = readout_center;
                out["readout_rotation"] = readout_rotation;
                out["k_center"] = py::make_tuple(ks.k_center[0], ks.k_center[1], ks.k_center[2]);
                out["central_readout"] = ks.central_readout;
                out["rotations_vary"] = ks.rotations_vary;
                out["key_groups"] = ks.key_groups;
                out["total_duration"] = ks.total_duration;
                return out;
            })

        // The b-tensor and gradient moments, one entry per excitation.  A
        // dict again, and for the same reason -- and here the shape is not
        // upstream's at all: the tensor comes back in three parts, because
        // NOROT splits a shot into what the console's prescription turns and
        // what it does not.  See moments.hpp.
        .def(
            "calc_moments",
            [](Sequence& self, bool calc_b, bool calc_m1, bool calc_m2, bool calc_m3,
               int n_dummy, int first_block, int last_block) {
                pulseq::MomentsOptions options;
                options.calc_b = calc_b;
                options.calc_m1 = calc_m1;
                options.calc_m2 = calc_m2;
                options.calc_m3 = calc_m3;
                options.n_dummy = n_dummy;
                options.first_block = first_block;
                options.last_block = last_block;

                pulseq::Moments m;
                {
                    py::gil_scoped_release unlocked;
                    m = pulseq::calc_moments(self, options);
                }

                const int shots = m.num_shots();

                const auto tensors = [](const std::vector<pulseq::BTensorParts>& src,
                                        pulseq::Matrix3 pulseq::BTensorParts::*part) {
                    py::array_t<double> out({static_cast<int>(src.size()), 3, 3});
                    double* dst = static_cast<double*>(out.request().ptr);
                    for (size_t i = 0; i < src.size(); ++i)
                        std::memcpy(dst + i * 9, (src[i].*part).data(), 9 * sizeof(double));
                    return out;
                };
                const auto vectors = [](const std::vector<std::array<double, 3>>& src) {
                    py::array_t<double> out({static_cast<int>(src.size()), 3});
                    double* dst = static_cast<double*>(out.request().ptr);
                    for (size_t i = 0; i < src.size(); ++i)
                        std::memcpy(dst + i * 3, src[i].data(), 3 * sizeof(double));
                    return out;
                };

                py::dict out;
                out["t_excitation"] = py::array_t<double>(shots, m.t_excitation.data());
                out["t_echo"] = py::array_t<double>(shots, m.t_echo.data());
                out["b_fixed"] = tensors(m.b, &pulseq::BTensorParts::fixed);
                out["b_rotatable"] = tensors(m.b, &pulseq::BTensorParts::rotatable);
                out["b_cross"] = tensors(m.b, &pulseq::BTensorParts::cross);
                out["m1"] = vectors(m.m1);
                out["m2"] = vectors(m.m2);
                out["m3"] = vectors(m.m3);
                out["table_fixed"] = tensors(m.table, &pulseq::BTensorParts::fixed);
                out["table_rotatable"] = tensors(m.table, &pulseq::BTensorParts::rotatable);
                out["table_cross"] = tensors(m.table, &pulseq::BTensorParts::cross);
                out["table_index"] = py::array_t<int>(static_cast<int>(m.table_index.size()),
                                                      m.table_index.data());
                return out;
            })

        // Encoding counters derived from the trajectory.  Also a dict; the
        // wrapper turns it into the labels/aux pair.
        .def("auto_label",
             [](Sequence& self, int first_block, int last_block,
                std::array<bool, 3> reflect, std::array<int, 3> reorder,
                std::array<double, 3> trajectory_delay, bool apply,
                const std::vector<std::pair<std::string, int>>& repeat_dims,
                const std::vector<std::string>& skip, bool mirror_fourier,
                const std::string& sort_slices) {
                 pulseq::AutoLabelOptions options;
                 options.first_block = first_block;
                 options.last_block = last_block;
                 options.reflect = reflect;
                 options.reorder = reorder;
                 options.trajectory_delay = trajectory_delay;
                 options.mirror_fourier = mirror_fourier;
                 if (sort_slices == "acquisition")
                     options.sort_slices = pulseq::SliceSorting::Acquisition;
                 else if (sort_slices == "descending")
                     options.sort_slices = pulseq::SliceSorting::Descending;
                 else
                     options.sort_slices = pulseq::SliceSorting::Ascending;
                 for (const auto& dim : repeat_dims)
                 {
                     pulseq::RepeatDim d;
                     d.name = dim.first;
                     d.size = dim.second;
                     options.repeat_dims.push_back(d);
                 }
                 options.skip = skip;

                 pulseq::AutoLabelResult r;
                 {
                     py::gil_scoped_release unlocked;
                     r = pulseq::auto_label(self, options, apply);
                 }

                 py::dict labels;
                 for (const auto& entry : r.labels.present())
                     labels[py::str(entry.first)] = py::array_t<int>(
                         static_cast<int>(entry.second->size()), entry.second->data());

                 py::dict aux;
                 if (r.aux.has_center_line)
                     aux["kSpaceCenterLine"] = r.aux.center_line;
                 if (r.aux.has_center_partition)
                     aux["kSpaceCenterPartition"] = r.aux.center_partition;
                 if (r.aux.has_center_sample)
                     aux["kSpaceCenterSample"] = r.aux.center_sample;
                 if (!r.aux.slice_positions.empty())
                     aux["SlicePositions"] = py::array_t<double>(
                         static_cast<int>(r.aux.slice_positions.size()),
                         r.aux.slice_positions.data());
                 if (r.aux.has_slice_thickness)
                     aux["SliceThickness"] = r.aux.slice_thickness;
                 if (r.aux.has_slice_gap)
                     aux["SliceGap"] = r.aux.slice_gap;
                 if (r.aux.has_gridding)
                 {
                     aux["TrapezoidGriddingParameters"] =
                         py::array_t<double>(5, r.aux.trapezoid_gridding.data());
                     aux["TargetGriddedSamples"] = r.aux.target_gridded_samples;
                 }

                 py::dict out;
                 out["labels"] = labels;
                 out["aux"] = aux;
                 out["adc_block"] = py::array_t<int>(
                     static_cast<int>(r.labels.adc_block.size()), r.labels.adc_block.data());
                 out["key_groups"] = r.key_groups;
                 out["num_readouts"] = r.num_readouts;
                 return out;
             })

        // Labels computed elsewhere, applied here -- autoLabel.m's
        // 'useLabels'/'useAux'.  The six derived counters go into their own
        // fields so the emission order is the one detection would have used;
        // anything else keeps the caller's order after them.
        .def("apply_labels",
             [](Sequence& self, const std::vector<int>& adc_block,
                const std::vector<std::pair<std::string, std::vector<int>>>& labels,
                const py::dict& aux) {
                 pulseq::AutoLabels out;
                 out.adc_block = adc_block;
                 for (const auto& entry : labels)
                 {
                     if (entry.first == "NOISE")
                         out.noise = entry.second;
                     else if (entry.first == "SLC")
                         out.slc = entry.second;
                     else if (entry.first == "REV")
                         out.rev = entry.second;
                     else if (entry.first == "LIN")
                         out.lin = entry.second;
                     else if (entry.first == "PAR")
                         out.par = entry.second;
                     else if (entry.first == "REP")
                         out.rep = entry.second;
                     else
                         out.named.emplace_back(entry.first, entry.second);
                 }

                 pulseq::AutoLabelAux a;
                 if (aux.contains("kSpaceCenterLine"))
                 {
                     a.has_center_line = true;
                     a.center_line = aux["kSpaceCenterLine"].cast<int>();
                 }
                 if (aux.contains("kSpaceCenterPartition"))
                 {
                     a.has_center_partition = true;
                     a.center_partition = aux["kSpaceCenterPartition"].cast<int>();
                 }
                 if (aux.contains("kSpaceCenterSample"))
                 {
                     a.has_center_sample = true;
                     a.center_sample = aux["kSpaceCenterSample"].cast<int>();
                 }
                 if (aux.contains("SlicePositions"))
                     a.slice_positions = aux["SlicePositions"].cast<std::vector<double>>();
                 if (aux.contains("SliceThickness"))
                 {
                     a.has_slice_thickness = true;
                     a.slice_thickness = aux["SliceThickness"].cast<double>();
                 }
                 if (aux.contains("SliceGap"))
                 {
                     a.has_slice_gap = true;
                     a.slice_gap = aux["SliceGap"].cast<double>();
                 }
                 if (aux.contains("TrapezoidGriddingParameters"))
                 {
                     const std::vector<double> g =
                         aux["TrapezoidGriddingParameters"].cast<std::vector<double>>();
                     if (g.size() != 5)
                         throw std::runtime_error(
                             "apply_labels: TrapezoidGriddingParameters needs 5 values");
                     a.has_gridding = true;
                     for (size_t i = 0; i < 5; ++i)
                         a.trapezoid_gridding[i] = g[i];
                     if (aux.contains("TargetGriddedSamples"))
                         a.target_gridded_samples = aux["TargetGriddedSamples"].cast<int>();
                 }

                 py::gil_scoped_release unlocked;
                 pulseq::apply_labels(self, out, a);
             });

    {
        PyTypeObject* type = reinterpret_cast<PyTypeObject*>(sequence_class.ptr());
        py::object descriptor =
            py::reinterpret_steal<py::object>(PyDescr_NewMethod(type, &add_block_fast_def));
        if (!descriptor)
            throw py::error_already_set();
        sequence_class.attr("add_block_events") = descriptor;
    }

    m.def(
        "read_file",
        [](const std::string& path) {
            Sequence seq;
            static_cast<pulseq::Sequence&>(seq) = pulseq::read_file(path);
            return seq;
        },
        "Read a .seq or binary Pulseq file.");
}
