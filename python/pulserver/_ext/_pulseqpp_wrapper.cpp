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
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "_pulseqpp_eventtypes.h"
#include "_pulseqpp_events.h"

#include "pulseq/read.hpp"
#include "pulseq/sequence.hpp"
#include "pulseq/write.hpp"

#include <cstring>
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

    py::class_<Sequence>(m, "Sequence")
        .def(py::init<>())

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
        // call rather than one per event.  See _pulseqpp_events.h.
        .def("add_block_events",
             [](Sequence& self, const py::args& events) {
                 return self.add_block(pulseqpp_events::build_block(self, events));
             })
        // The same registration, written over a block that already exists.
        // Rows the old block referred to are left where they are: they may be
        // shared, and deduplication is what decides what survives.
        .def("set_block_events",
             [](Sequence& self, int index, const py::args& events) {
                 self.set_block(index, pulseqpp_events::build_block(self, events));
             })
        // Measurement only: the same call signature as add_block_events with
        // none of the work, so the difference is exactly what unpacking the
        // event objects costs.
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

        /* -- files ------------------------------------------------------ */
        .def("write_text",
             [](Sequence& self, bool create_signature) {
                 return py::bytes(pulseq::write_text(self, create_signature));
             })
        .def("write_binary",
             [](Sequence& self) { return py::bytes(pulseq::write_binary(self)); })
        .def("required_revision", [](const Sequence& s) { return pulseq::required_revision(s); });

    m.def(
        "read_file",
        [](const std::string& path) {
            Sequence seq;
            static_cast<pulseq::Sequence&>(seq) = pulseq::read_file(path);
            return seq;
        },
        "Read a .seq or binary Pulseq file.");
}
