/**
 * @file from_libraries.cpp
 * @brief A pulseq_file built from the libraries pypulseqpp was read into.
 */

#include "from_libraries.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace pulserver
{
namespace
{

using Rows = py::array_t<double, py::array::c_style | py::array::forcecast>;

/* An (n, width) array as a freshly allocated C row array; null when empty. */
template <int Width>
PULSEQ_REAL (*rows(const py::object &value, int &count))[Width]
{
    const Rows array = value.cast<Rows>();
    if (array.ndim() != 2 || (array.shape(0) > 0 && array.shape(1) != Width))
        throw std::invalid_argument(
            "expected an (n, " + std::to_string(Width) + ") array");
    count = static_cast<int>(array.shape(0));
    if (count == 0)
        return nullptr;
    auto *out = static_cast<PULSEQ_REAL (*)[Width]>(
        PULSEQ_ALLOC(sizeof(PULSEQ_REAL) * (size_t)count * Width));
    if (!out)
        throw std::bad_alloc();
    const double *data = array.data();
    for (int i = 0; i < count * Width; ++i)
        (&out[0][0])[i] = static_cast<PULSEQ_REAL>(data[i]);
    return out;
}

/* An (n, width) integer array as a freshly allocated C row array; null when
 * empty. */
template <int Width>
int (*integer_rows(const py::object &value, int &count))[Width]
{
    const auto array = value.cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
    if (array.ndim() != 2 || (array.shape(0) > 0 && array.shape(1) != Width))
        throw std::invalid_argument(
            "expected an (n, " + std::to_string(Width) + ") integer array");
    count = static_cast<int>(array.shape(0));
    if (count == 0)
        return nullptr;
    auto *out = static_cast<int (*)[Width]>(PULSEQ_ALLOC(sizeof(int) * (size_t)count * Width));
    if (!out)
        throw std::bad_alloc();
    std::memcpy(out, array.data(), sizeof(int) * (size_t)count * Width);
    return out;
}

int *integers(const py::object &value, int count)
{
    const auto array = value.cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
    if (static_cast<int>(array.size()) != count)
        throw std::invalid_argument("an integer column of the wrong length");
    if (count == 0)
        return nullptr;
    auto *out = static_cast<int *>(PULSEQ_ALLOC(sizeof(int) * (size_t)count));
    if (!out)
        throw std::bad_alloc();
    std::memcpy(out, array.data(), sizeof(int) * (size_t)count);
    return out;
}

PULSEQ_REAL *reals(const py::object &value, int count)
{
    const auto array = value.cast<Rows>();
    if (static_cast<int>(array.size()) != count)
        throw std::invalid_argument("a real column of the wrong length");
    if (count == 0)
        return nullptr;
    auto *out = static_cast<PULSEQ_REAL *>(PULSEQ_ALLOC(sizeof(PULSEQ_REAL) * (size_t)count));
    if (!out)
        throw std::bad_alloc();
    const double *data = array.data();
    for (int i = 0; i < count; ++i)
        out[i] = static_cast<PULSEQ_REAL>(data[i]);
    return out;
}

void copy_name(char *destination, size_t size, const std::string &name)
{
    const size_t length = name.size() < size - 1 ? name.size() : size - 1;
    std::memcpy(destination, name.c_str(), length);
    destination[length] = '\0';
}

void triple(PULSEQ_REAL *destination, const py::object &value)
{
    const auto values = value.cast<std::vector<double>>();
    for (size_t i = 0; i < 3 && i < values.size(); ++i)
        destination[i] = static_cast<PULSEQ_REAL>(values[i]);
}

void build_definitions(pulseq_file &seq, const py::dict &definitions)
{
    seq.num_definitions = static_cast<int>(definitions.size());
    seq.is_definitions_library_parsed = 1;
    if (seq.num_definitions == 0)
        return;
    seq.definitions_library = static_cast<pulseq_definition *>(
        PULSEQ_ALLOC(sizeof(pulseq_definition) * (size_t)seq.num_definitions));
    if (!seq.definitions_library)
        throw std::bad_alloc();
    std::memset(
        seq.definitions_library, 0, sizeof(pulseq_definition) * (size_t)seq.num_definitions);

    int index = 0;
    for (const auto &item : definitions)
    {
        pulseq_definition &definition = seq.definitions_library[index++];
        copy_name(
            definition.name, sizeof(definition.name), item.first.cast<std::string>());
        const auto values = item.second.cast<std::vector<std::string>>();
        definition.value_size = static_cast<int>(values.size());
        if (definition.value_size == 0)
            continue;
        definition.value =
            static_cast<char **>(PULSEQ_ALLOC(sizeof(char *) * (size_t)definition.value_size));
        if (!definition.value)
            throw std::bad_alloc();
        for (int i = 0; i < definition.value_size; ++i)
        {
            definition.value[i] = static_cast<char *>(PULSEQ_ALLOC(values[i].size() + 1));
            if (!definition.value[i])
                throw std::bad_alloc();
            std::memcpy(definition.value[i], values[i].c_str(), values[i].size() + 1);
        }
    }
}

void build_shapes(pulseq_file &seq, const py::list &shapes)
{
    seq.shapes_library_size = static_cast<int>(shapes.size());
    seq.is_shapes_library_parsed = 1;
    if (seq.shapes_library_size == 0)
        return;
    seq.shapes_library = static_cast<pulseq_shape *>(
        PULSEQ_ALLOC(sizeof(pulseq_shape) * (size_t)seq.shapes_library_size));
    if (!seq.shapes_library)
        throw std::bad_alloc();
    std::memset(seq.shapes_library, 0, sizeof(pulseq_shape) * (size_t)seq.shapes_library_size);

    for (int i = 0; i < seq.shapes_library_size; ++i)
    {
        const auto entry = shapes[(size_t)i].cast<py::tuple>();
        const auto samples =
            entry[1].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        pulseq_shape &shape = seq.shapes_library[i];
        shape.num_uncompressed_samples = entry[0].cast<int>();
        shape.num_samples = static_cast<int>(samples.size());
        if (shape.num_samples == 0)
            continue;
        shape.samples = static_cast<PULSEQ_REAL *>(
            PULSEQ_ALLOC(sizeof(PULSEQ_REAL) * (size_t)shape.num_samples));
        if (!shape.samples)
            throw std::bad_alloc();
        for (int j = 0; j < shape.num_samples; ++j)
            shape.samples[j] = static_cast<PULSEQ_REAL>(samples.data()[j]);
    }
}

void build_rf_shims(pulseq_file &seq, const py::list &shims)
{
    seq.rf_shim_library_size = static_cast<int>(shims.size());
    if (seq.rf_shim_library_size == 0)
        return;
    seq.rf_shim_library = static_cast<pulseq_rf_shim_entry *>(
        PULSEQ_ALLOC(sizeof(pulseq_rf_shim_entry) * (size_t)seq.rf_shim_library_size));
    if (!seq.rf_shim_library)
        throw std::bad_alloc();
    std::memset(
        seq.rf_shim_library, 0, sizeof(pulseq_rf_shim_entry) * (size_t)seq.rf_shim_library_size);

    for (int i = 0; i < seq.rf_shim_library_size; ++i)
    {
        const auto values =
            shims[(size_t)i].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        pulseq_rf_shim_entry &entry = seq.rf_shim_library[i];
        entry.num_channels = static_cast<int>(values.size()) / 2;
        if (entry.num_channels > PULSEQ_MAX_RF_SHIM_CHANNELS)
            throw std::invalid_argument("an RF shim of more channels than the format holds");
        for (int j = 0; j < 2 * entry.num_channels; ++j)
            entry.values[j] = static_cast<PULSEQ_REAL>(values.data()[j]);
    }
}

void build_extensions(pulseq_file &seq, const py::dict &libraries)
{
    seq.extensions_library = rows<3>(libraries["extensions"], seq.extensions_library_size);
    seq.trigger_library = rows<4>(libraries["triggers"], seq.trigger_library_size);
    seq.rotation_quaternion_library =
        rows<4>(libraries["rotations"], seq.rotation_library_size);
    seq.labelset_library = rows<2>(libraries["labelset"], seq.labelset_library_size);
    seq.labelinc_library = rows<2>(libraries["labelinc"], seq.labelinc_library_size);
    seq.soft_delay_library = rows<4>(libraries["soft_delays"], seq.soft_delay_library_size);
    build_rf_shims(seq, libraries["rf_shims"].cast<py::list>());

    const auto map = libraries["extension_map"].cast<std::vector<int>>();
    for (size_t i = 0; i < 8 && i < map.size(); ++i)
    {
        seq.extension_map[i] = map[i];
        if (seq.extension_lut_size < map[i])
            seq.extension_lut_size = map[i];
    }
    if (seq.extension_lut_size > 0)
    {
        seq.extension_lut =
            static_cast<int *>(PULSEQ_ALLOC(sizeof(int) * (size_t)(seq.extension_lut_size + 1)));
        if (!seq.extension_lut)
            throw std::bad_alloc();
        for (int i = 0; i <= seq.extension_lut_size; ++i)
            seq.extension_lut[i] = PULSEQ_EXT_UNKNOWN;
        for (int i = 0; i < 8; ++i)
            if (seq.extension_map[i] > 0)
                seq.extension_lut[seq.extension_map[i]] = i;
    }
    seq.is_extensions_library_parsed = 1;
}

using Integers = py::array_t<int32_t, py::array::forcecast>;
using Reals = py::array_t<double, py::array::forcecast>;

/* Old library id to the id of the played rows, numbered from 1 in id order;
 * 0 for a row no block plays. */
std::vector<int32_t> id_map(const py::object &value)
{
    const auto array = value.cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
    return std::vector<int32_t>(array.data(), array.data() + array.size());
}

int32_t mapped(const std::vector<int32_t> &map, int32_t id)
{
    if (id < 0 || static_cast<size_t>(id) >= map.size())
        throw std::invalid_argument("a block names an event its library does not hold");
    return map[static_cast<size_t>(id)];
}

int played_count(const std::vector<int32_t> &map)
{
    int count = 0;
    for (const int32_t id : map)
        count = id > count ? id : count;
    return count;
}

/* The definition of each played row, numbered from 0 in order of first
 * appearance; every row when no map is given. */
int *dense_definitions(const py::object &value, const std::vector<int32_t> *map, int count)
{
    const auto raw = value.cast<Integers>().unchecked<1>();
    if (count == 0)
        return nullptr;
    auto *out = static_cast<int *>(PULSEQ_ALLOC(sizeof(int) * (size_t)count));
    if (!out)
        throw std::bad_alloc();
    std::vector<int> rank;
    int next = 0;
    int row = 0;
    for (py::ssize_t old = 0; old < raw.shape(0) && row < count; ++old)
    {
        if (map && mapped(*map, static_cast<int32_t>(old + 1)) == 0)
            continue;
        const int32_t definition = raw(old);
        if (definition < 0)
            throw std::invalid_argument("a negative event definition");
        if (static_cast<size_t>(definition) >= rank.size())
            rank.resize(static_cast<size_t>(definition) + 1, -1);
        if (rank[static_cast<size_t>(definition)] < 0)
            rank[static_cast<size_t>(definition)] = next++;
        out[row++] = rank[static_cast<size_t>(definition)];
    }
    if (row != count)
        throw std::invalid_argument("a definition column of the wrong length");
    return out;
}

/* The block table from pypulseqpp's event ids, renumbered to the played
 * rows, and durations in block rasters. */
void build_blocks(
    pulseq_file &seq,
    const py::dict &libraries,
    const std::vector<int32_t> &rf_map,
    const std::vector<int32_t> &grad_map,
    const std::vector<int32_t> &adc_map)
{
    const auto events = libraries["block_events"].cast<Integers>().unchecked<2>();
    const auto durations = libraries["block_durations"].cast<Reals>().unchecked<1>();
    const double raster = libraries["block_duration_raster"].cast<double>();
    if (events.shape(0) > 0 && events.shape(1) != 6)
        throw std::invalid_argument("expected an (n, 6) block event table");
    if (durations.shape(0) != events.shape(0))
        throw std::invalid_argument("a block duration column of the wrong length");
    seq.num_blocks = static_cast<int>(events.shape(0));
    if (seq.num_blocks == 0)
        return;
    seq.block_library = static_cast<PULSEQ_REAL (*)[7]>(
        PULSEQ_ALLOC(sizeof(PULSEQ_REAL) * (size_t)seq.num_blocks * 7));
    if (!seq.block_library)
        throw std::bad_alloc();
    for (int i = 0; i < seq.num_blocks; ++i)
    {
        PULSEQ_REAL *row = seq.block_library[i];
        row[0] = static_cast<PULSEQ_REAL>(std::rint(durations(i) / raster));
        row[1] = static_cast<PULSEQ_REAL>(mapped(rf_map, events(i, 0)));
        for (int axis = 0; axis < 3; ++axis)
            row[2 + axis] = static_cast<PULSEQ_REAL>(mapped(grad_map, events(i, 1 + axis)));
        row[5] = static_cast<PULSEQ_REAL>(mapped(adc_map, events(i, 4)));
        row[6] = static_cast<PULSEQ_REAL>(events(i, 5));
    }
}

/* The played ADC rows: sample count, dwell in ns, delay in µs, zero ppm
 * offsets, the absolute frequency (Hz) and phase (rad) offsets, and no
 * phase modulation. */
void build_adc(pulseq_file &seq, const py::dict &libraries, const std::vector<int32_t> &adc_map)
{
    const auto raw = libraries["adc"].cast<Reals>().unchecked<2>();
    const auto offsets = libraries["adc_offsets"].cast<Reals>().unchecked<2>();
    const auto sizes = libraries["shape_sizes"].cast<std::vector<int>>();
    seq.adc_library_size = played_count(adc_map);
    if (seq.adc_library_size == 0)
        return;
    if (raw.shape(1) != 8 || offsets.shape(0) != raw.shape(0))
        throw std::invalid_argument("expected an (n, 8) ADC table and one offset pair per row");
    seq.adc_library = static_cast<PULSEQ_REAL (*)[8]>(
        PULSEQ_ALLOC(sizeof(PULSEQ_REAL) * (size_t)seq.adc_library_size * 8));
    if (!seq.adc_library)
        throw std::bad_alloc();
    for (py::ssize_t old = 0; old < raw.shape(0); ++old)
    {
        const int32_t id = static_cast<size_t>(old + 1) < adc_map.size()
                               ? adc_map[static_cast<size_t>(old + 1)]
                               : 0;
        if (id == 0)
            continue;
        const long modulation = std::lround(raw(old, 7));
        if (modulation > 0 && static_cast<size_t>(modulation) <= sizes.size() &&
            sizes[static_cast<size_t>(modulation - 1)] > 0 &&
            sizes[static_cast<size_t>(modulation - 1)] != raw(old, 0))
            throw std::invalid_argument(
                "ADC " + std::to_string(old + 1) + " acquires " +
                std::to_string(std::lround(raw(old, 0))) + " samples but its phase modulation has " +
                std::to_string(sizes[static_cast<size_t>(modulation - 1)]));
        PULSEQ_REAL *row = seq.adc_library[id - 1];
        row[0] = static_cast<PULSEQ_REAL>(raw(old, 0));
        row[1] = static_cast<PULSEQ_REAL>(std::rint(raw(old, 1) * 1e9));
        row[2] = static_cast<PULSEQ_REAL>(std::rint(raw(old, 2) * 1e6));
        row[3] = 0;
        row[4] = 0;
        row[5] = static_cast<PULSEQ_REAL>(offsets(old, 0));
        row[6] = static_cast<PULSEQ_REAL>(offsets(old, 1));
        row[7] = 0;
    }
}

/* A per-block label column, or one value for every block. */
struct Column
{
    Integers array;
    int value(int block) const
    {
        return array.size() == 1 ? array.data()[0] : array.data()[block];
    }
};

std::vector<Column> label_columns(const py::object &value, int count)
{
    std::vector<Column> columns;
    for (const auto &item : value.cast<py::list>())
    {
        Column column{py::array_t<int32_t, py::array::c_style | py::array::forcecast>::ensure(item)};
        if (!column.array || (column.array.size() != 1 && column.array.size() != count))
            throw std::invalid_argument("a label column of the wrong length");
        columns.push_back(std::move(column));
    }
    return columns;
}

/* Per block, the rotation and shim rows from 0, the flags in force, where a
 * TRID is set and the gradient under its RF pulse; per readout, the labels. */
void build_block_states(pulseq_file &seq, const py::dict &libraries)
{
    const int count = seq.num_blocks;
    const auto events = libraries["block_events"].cast<Integers>().unchecked<2>();
    seq.block_rotations = integers(libraries["block_rotations"], count);
    seq.block_shims = integers(libraries["block_shims"], count);
    for (int i = 0; i < count; ++i)
    {
        seq.block_rotations[i] -= 1;
        seq.block_shims[i] -= 1;
    }

    const auto allocate = [](size_t bytes) {
        void *memory = PULSEQ_ALLOC(bytes);
        if (!memory)
            throw std::bad_alloc();
        std::memset(memory, 0, bytes);
        return memory;
    };
    seq.block_trid_set = static_cast<int *>(allocate(sizeof(int) * (size_t)count));
    for (const int block : libraries["trid_blocks"].cast<std::vector<int>>())
        if (block >= 1 && block <= count)
            seq.block_trid_set[block - 1] = 1;

    seq.block_rf_steady = static_cast<int *>(allocate(sizeof(int) * (size_t)count));
    seq.block_rf_gradient =
        static_cast<PULSEQ_REAL (*)[3]>(allocate(sizeof(PULSEQ_REAL) * (size_t)count * 3));
    const auto pulsed = libraries["rf_pulsed"].cast<Integers>().unchecked<1>();
    const auto steady = libraries["rf_steady"].cast<py::array_t<bool, py::array::forcecast>>().unchecked<2>();
    const auto gradient = libraries["rf_gradient"].cast<Reals>().unchecked<2>();
    for (py::ssize_t p = 0; p < pulsed.shape(0); ++p)
    {
        const int block = pulsed(p) - 1;
        if (block < 0 || block >= count)
            throw std::invalid_argument("an RF gradient for a block the sequence does not hold");
        seq.block_rf_steady[block] = steady(p, 0) && steady(p, 1) && steady(p, 2);
        for (int axis = 0; axis < 3; ++axis)
            seq.block_rf_gradient[block][axis] = static_cast<PULSEQ_REAL>(gradient(p, axis));
    }

    const auto flags = label_columns(libraries["flag_labels"], count);
    const auto readout = label_columns(libraries["readout_labels"], count);
    if (flags.size() != PULSEQ_BLOCK_FLAG_WIDTH || readout.size() != PULSEQ_ADC_LABEL_WIDTH)
        throw std::invalid_argument("a label state of the wrong width");
    seq.block_flags = static_cast<int (*)[PULSEQ_BLOCK_FLAG_WIDTH]>(
        allocate(sizeof(int) * (size_t)count * PULSEQ_BLOCK_FLAG_WIDTH));
    seq.num_adc_labels = 0;
    for (int i = 0; i < count; ++i)
    {
        for (int f = 0; f < PULSEQ_BLOCK_FLAG_WIDTH; ++f)
            seq.block_flags[i][f] = flags[(size_t)f].value(i);
        seq.num_adc_labels += events(i, 4) > 0;
    }
    if (seq.num_adc_labels == 0)
        return;
    seq.adc_labels = static_cast<int (*)[PULSEQ_ADC_LABEL_WIDTH]>(
        allocate(sizeof(int) * (size_t)seq.num_adc_labels * PULSEQ_ADC_LABEL_WIDTH));
    int row = 0;
    for (int i = 0; i < count; ++i)
    {
        if (events(i, 4) <= 0)
            continue;
        for (int l = 0; l < PULSEQ_ADC_LABEL_WIDTH; ++l)
            seq.adc_labels[row][l] = readout[(size_t)l].value(i);
        seq.adc_labels[row][PULSEQ_ADC_LABEL_WIDTH - 1] =
            seq.adc_labels[row][PULSEQ_ADC_LABEL_WIDTH - 1] != 0;
        ++row;
    }
}

} // namespace

void build_pulseq_file(pulseq_file &seq, const py::dict &libraries)
{
    const auto version = libraries["version"].cast<std::vector<int>>();
    seq.version_major = version.at(0);
    seq.version_minor = version.at(1);
    seq.version_revision = version.at(2);
    seq.version_combined =
        seq.version_major * 1000000 + seq.version_minor * 1000 + seq.version_revision;
    seq.is_version_parsed = 1;

    const auto rasters = libraries["rasters"].cast<std::vector<double>>();
    pulseq_reserved_definitions &reserved = seq.reserved_definitions_library;
    reserved.radiofrequency_raster_time = static_cast<PULSEQ_REAL>(rasters.at(0));
    reserved.gradient_raster_time = static_cast<PULSEQ_REAL>(rasters.at(1));
    reserved.adc_raster_time = static_cast<PULSEQ_REAL>(rasters.at(2));
    reserved.block_duration_raster = static_cast<PULSEQ_REAL>(rasters.at(3));

    const auto declared = libraries["reserved"].cast<py::dict>();
    triple(reserved.fov, declared["fov"]);
    triple(reserved.matrix, declared["matrix"]);
    triple(reserved.nav_fov, declared["nav_fov"]);
    triple(reserved.nav_matrix, declared["nav_matrix"]);
    reserved.total_duration = static_cast<PULSEQ_REAL>(declared["total_duration"].cast<double>());
    reserved.enable_pmc = declared["enable_pmc"].cast<int>();
    reserved.num_gain_cal_readouts = declared["num_gain_cal_readouts"].cast<int>();
    reserved.enable_sar_burst_mode = declared["enable_sar_burst_mode"].cast<int>();
    reserved.vop_sar_ratio = static_cast<PULSEQ_REAL>(declared["vop_sar_ratio"].cast<double>());
    reserved.vop_global_sar_ratio =
        static_cast<PULSEQ_REAL>(declared["vop_global_sar_ratio"].cast<double>());
    reserved.repetition_size = declared["repetition_size"].cast<int>();
    copy_name(reserved.name, sizeof(reserved.name), declared["name"].cast<std::string>());
    copy_name(
        reserved.next_sequence,
        sizeof(reserved.next_sequence),
        declared["next_sequence"].cast<std::string>());

    build_definitions(seq, libraries["definitions"].cast<py::dict>());

    const auto rf_map = id_map(libraries["rf_map"]);
    const auto grad_map = id_map(libraries["grad_map"]);
    const auto adc_map = id_map(libraries["adc_map"]);
    build_blocks(seq, libraries, rf_map, grad_map, adc_map);
    seq.block_ids = static_cast<int *>(PULSEQ_ALLOC(sizeof(int) * (size_t)(seq.num_blocks + 1)));
    if (!seq.block_ids)
        throw std::bad_alloc();
    for (int i = 0; i < seq.num_blocks; ++i)
        seq.block_ids[i] = i + 1;
    seq.is_block_library_parsed = 1;
    build_block_states(seq, libraries);

    seq.rf_library = rows<10>(libraries["rf"], seq.rf_library_size);
    seq.rf_use_tags = integers(libraries["rf_use"], seq.rf_library_size);
    seq.rf_flip_deg = reals(libraries["rf_flip_deg"], seq.rf_library_size);
    seq.rf_channels = integers(libraries["rf_channels"], seq.rf_library_size);
    seq.rf_b1sq_integral = reals(libraries["rf_b1sq_integral"], seq.rf_library_size);
    seq.is_rf_library_parsed = 1;
    {
        int spectra = 0;
        seq.rf_spectra =
            rows<PULSEQ_RF_SPECTRUM_WIDTH>(libraries["rf_spectra"], spectra);
        if (spectra != seq.rf_library_size)
            throw std::invalid_argument("an RF spectrum table of the wrong length");
    }

    seq.grad_library = rows<7>(libraries["grad"], seq.grad_library_size);
    seq.is_grad_library_parsed = 1;
    {
        int measured = 0;
        seq.grad_statistics = rows<3>(libraries["grad_statistics"], measured);
        if (measured != seq.grad_library_size)
            throw std::invalid_argument("a gradient statistics table of the wrong length");
    }

    build_adc(seq, libraries, adc_map);
    seq.is_adc_library_parsed = 1;

    seq.rf_definitions = dense_definitions(libraries["rf_definitions"], &rf_map, seq.rf_library_size);
    seq.grad_definitions =
        dense_definitions(libraries["grad_definitions"], &grad_map, seq.grad_library_size);
    seq.adc_definitions =
        dense_definitions(libraries["adc_definitions"], &adc_map, seq.adc_library_size);
    seq.block_definitions = dense_definitions(libraries["block_definitions"], nullptr, seq.num_blocks);

    build_extensions(seq, libraries);
    build_shapes(seq, libraries["shapes"].cast<py::list>());
}

} // namespace pulserver
