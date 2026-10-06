/**
 * @file from_libraries.cpp
 * @brief A pulseq_file built from the libraries pypulseqpp was read into.
 */

#include "from_libraries.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdlib>
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
using Held = std::vector<py::object>;

/* A C-contiguous view of an array the conversion reads in place, kept alive
 * in held; a converted copy only when the caller's array is not one. */
template <typename T>
const T *borrowed(const py::object &value, Held &held, py::ssize_t &size)
{
    auto array = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(value);
    if (!array)
        throw std::invalid_argument("expected a numeric array");
    size = array.size();
    held.push_back(array);
    return array.data();
}

/* Old library id to the id of the played rows, numbered from 1 in id order;
 * 0 for a row no block plays. */
struct Map
{
    const int32_t *ids;
    py::ssize_t size;

    int32_t operator()(int32_t id) const
    {
        if (id < 0 || id >= size)
            throw std::invalid_argument("a block names an event its library does not hold");
        return ids[id];
    }

    int played() const
    {
        int count = 0;
        for (py::ssize_t i = 0; i < size; ++i)
            count = ids[i] > count ? ids[i] : count;
        return count;
    }
};

Map id_map(const py::object &value, Held &held)
{
    Map map{};
    map.ids = borrowed<int32_t>(value, held, map.size);
    return map;
}

/* The definition of each played row, numbered from 0 in order of first
 * appearance; every row when no map is given. */
int *dense_definitions(const py::object &value, const Map *map, int count)
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
        if (map && (*map)(static_cast<int32_t>(old + 1)) == 0)
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

/* pypulseqpp's block events and durations, read in place through the maps to
 * the played rows. */
void build_blocks(
    pulseq_file &seq,
    const py::dict &libraries,
    const Map &rf_map,
    const Map &grad_map,
    const Map &adc_map,
    Held &held)
{
    py::ssize_t events_size = 0, durations_size = 0;
    const int32_t *events = borrowed<int32_t>(libraries["block_events"], held, events_size);
    const double *durations = borrowed<double>(libraries["block_durations"], held, durations_size);
    if (events_size != 6 * durations_size)
        throw std::invalid_argument("expected an (n, 6) block event table and one duration per block");
    seq.num_blocks = static_cast<int>(durations_size);
    for (py::ssize_t i = 0; i < durations_size; ++i)
    {
        const int32_t *row = events + 6 * i;
        rf_map(row[0]);
        for (int axis = 0; axis < 3; ++axis)
            grad_map(row[1 + axis]);
        adc_map(row[4]);
    }
    seq.block_events = reinterpret_cast<const int (*)[6]>(events);
    seq.block_durations = durations;
    seq.block_duration_raster = libraries["block_duration_raster"].cast<double>();
    seq.rf_map = rf_map.ids;
    seq.rf_map_size = static_cast<int>(rf_map.size);
    seq.grad_map = grad_map.ids;
    seq.grad_map_size = static_cast<int>(grad_map.size);
    seq.adc_map = adc_map.ids;
    seq.adc_map_size = static_cast<int>(adc_map.size);
}

/* The played ADC rows and their absolute frequency (Hz) and phase (rad)
 * offsets, read in place; a phase modulation must hold one phase per
 * sample. */
void build_adc(pulseq_file &seq, const py::dict &libraries, const Map &adc_map, Held &held)
{
    py::ssize_t raw_size = 0, offsets_size = 0;
    const double *raw = borrowed<double>(libraries["adc"], held, raw_size);
    const double *offsets = borrowed<double>(libraries["adc_offsets"], held, offsets_size);
    const auto sizes = libraries["shape_sizes"].cast<std::vector<int>>();
    seq.adc_library_size = adc_map.played();
    if (seq.adc_library_size == 0)
        return;
    const py::ssize_t rows = raw_size / 8;
    if (raw_size != 8 * rows || offsets_size != 2 * rows)
        throw std::invalid_argument("expected an (n, 8) ADC table and one offset pair per row");
    bool identity = rows == seq.adc_library_size;
    for (py::ssize_t old = 0; old < rows; ++old)
    {
        const int32_t id = old + 1 < adc_map.size ? adc_map.ids[old + 1] : 0;
        identity = identity && id == old + 1;
        if (id == 0)
            continue;
        const double *row = raw + 8 * old;
        const long modulation = std::lround(row[7]);
        if (modulation > 0 && static_cast<size_t>(modulation) <= sizes.size() &&
            sizes[static_cast<size_t>(modulation - 1)] > 0 &&
            sizes[static_cast<size_t>(modulation - 1)] != row[0])
            throw std::invalid_argument(
                "ADC " + std::to_string(old + 1) + " acquires " +
                std::to_string(std::lround(row[0])) + " samples but its phase modulation has " +
                std::to_string(sizes[static_cast<size_t>(modulation - 1)]));
    }
    seq.adc_rows = reinterpret_cast<const double (*)[8]>(raw);
    seq.adc_offsets = reinterpret_cast<const double (*)[2]>(offsets);
    if (identity)
        return;
    seq.adc_rows_of = static_cast<int *>(PULSEQ_ALLOC(sizeof(int) * (size_t)seq.adc_library_size));
    if (!seq.adc_rows_of)
        throw std::bad_alloc();
    for (py::ssize_t old = 0; old < rows; ++old)
    {
        const int32_t id = old + 1 < adc_map.size ? adc_map.ids[old + 1] : 0;
        if (id > 0)
            seq.adc_rows_of[id - 1] = static_cast<int>(old);
    }
}

/* Per-block label columns, each one value per block or one for every block,
 * read in place. */
void label_columns(const py::object &value, int count, pulseq_int_column *columns, size_t width, Held &held)
{
    const auto items = value.cast<py::list>();
    if (items.size() != width)
        throw std::invalid_argument("a label state of the wrong width");
    for (size_t i = 0; i < width; ++i)
    {
        py::ssize_t size = 0;
        columns[i].data = borrowed<int32_t>(items[i], held, size);
        if (size != 1 && size != count)
            throw std::invalid_argument("a label column of the wrong length");
        columns[i].stride = size == 1 ? 0 : 1;
    }
}

/* Zeroed by calloc, which leaves the pages of a large array untouched until
 * written: most of a per-block array set only at RF pulses never is. The
 * host build frees with PULSEQ_FREE, which is free. */
template <typename T> T *zeroed(size_t count)
{
    void *memory = std::calloc(count, sizeof(T));
    if (!memory)
        throw std::bad_alloc();
    return static_cast<T *>(memory);
}

/* The rotation and shim rows, read in place, and the blocks setting a TRID. */
void build_block_groups(pulseq_file &seq, const py::dict &libraries, Held &held)
{
    const int count = seq.num_blocks;
    py::ssize_t rotations = 0, shims = 0;
    seq.block_rotations = borrowed<int32_t>(libraries["block_rotations"], held, rotations);
    seq.block_shims = borrowed<int32_t>(libraries["block_shims"], held, shims);
    if (rotations != count || shims != count)
        throw std::invalid_argument("an integer column of the wrong length");
    seq.block_trid_set = zeroed<int>((size_t)count);
    for (const int block : libraries["trid_blocks"].cast<std::vector<int>>())
        if (block >= 1 && block <= count)
            seq.block_trid_set[block - 1] = 1;
}

/* The gradient under each RF pulse, and whether it is steady there. */
void build_rf_gradients(pulseq_file &seq, const py::dict &libraries)
{
    const int count = seq.num_blocks;
    seq.block_rf_steady = zeroed<int>((size_t)count);
    seq.block_rf_gradient = reinterpret_cast<PULSEQ_REAL (*)[3]>(zeroed<PULSEQ_REAL>((size_t)count * 3));
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
}

} // namespace

void build_pulseq_file(pulseq_file &seq, const py::dict &libraries, Held &held)
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
    reserved.spl_peak_db = static_cast<PULSEQ_REAL>(declared["spl_peak_db"].cast<double>());
    reserved.spl_average_dba =
        static_cast<PULSEQ_REAL>(declared["spl_average_dba"].cast<double>());
    reserved.repetition_size = declared["repetition_size"].cast<int>();
    copy_name(reserved.name, sizeof(reserved.name), declared["name"].cast<std::string>());
    copy_name(
        reserved.next_sequence,
        sizeof(reserved.next_sequence),
        declared["next_sequence"].cast<std::string>());

    build_definitions(seq, libraries["definitions"].cast<py::dict>());

    const Map rf_map = id_map(libraries["rf_map"], held);
    const Map grad_map = id_map(libraries["grad_map"], held);
    const Map adc_map = id_map(libraries["adc_map"], held);
    build_blocks(seq, libraries, rf_map, grad_map, adc_map, held);
    seq.is_block_library_parsed = 1;
    build_block_groups(seq, libraries, held);
    build_rf_gradients(seq, libraries);
    label_columns(libraries["flag_labels"], seq.num_blocks, seq.block_flags, PULSEQ_BLOCK_FLAG_WIDTH, held);
    label_columns(
        libraries["readout_labels"], seq.num_blocks, seq.adc_labels, PULSEQ_ADC_LABEL_WIDTH, held);

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

    build_adc(seq, libraries, adc_map, held);
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
