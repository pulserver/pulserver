/**
 * @file from_libraries.cpp
 * @brief A pulseq_file built from the libraries pypulseqpp was read into.
 */

#include "from_libraries.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

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
    copy_name(reserved.name, sizeof(reserved.name), declared["name"].cast<std::string>());
    copy_name(
        reserved.next_sequence,
        sizeof(reserved.next_sequence),
        declared["next_sequence"].cast<std::string>());

    build_definitions(seq, libraries["definitions"].cast<py::dict>());

    seq.block_library = rows<7>(libraries["blocks"], seq.num_blocks);
    seq.block_ids = static_cast<int *>(PULSEQ_ALLOC(sizeof(int) * (size_t)(seq.num_blocks + 1)));
    if (!seq.block_ids)
        throw std::bad_alloc();
    for (int i = 0; i < seq.num_blocks; ++i)
        seq.block_ids[i] = i + 1;
    seq.is_block_library_parsed = 1;

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

    seq.adc_library = rows<8>(libraries["adc"], seq.adc_library_size);
    seq.is_adc_library_parsed = 1;

    build_extensions(seq, libraries);
    build_shapes(seq, libraries["shapes"].cast<py::list>());
}

} // namespace pulserver
