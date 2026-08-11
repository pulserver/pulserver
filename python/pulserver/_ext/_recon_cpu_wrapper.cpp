#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <thread>
#include <vector>

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define PULSERVER_X86_TARGETS 1
#else
#define PULSERVER_X86_TARGETS 0
#endif

namespace py = pybind11;

namespace
{

    using complex64 = std::complex<float>;

    std::size_t packed_rank(std::size_t entries)
    {
        std::size_t rank = 0;
        while (rank * (rank + 1) / 2 < entries)
        {
            ++rank;
        }
        if (rank * (rank + 1) / 2 != entries)
        {
            throw std::invalid_argument("values first dimension is not a packed upper triangle");
        }
        return rank;
    }

    std::size_t location_thread_count(std::size_t locations)
    {
        // The matrix rank is deliberately not parallelized: every worker owns a
        // contiguous range of independent spatial-frequency locations.  A fairly
        // coarse grain amortizes thread creation for the small K encountered by
        // subspace and off-resonance models.
        constexpr std::size_t minimum_locations_per_thread = 16384;
        const std::size_t available = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        const std::size_t useful = std::max<std::size_t>(
            1,
            (locations + minimum_locations_per_thread - 1) / minimum_locations_per_thread);
        return std::min(available, useful);
    }

    std::size_t sample_thread_count(std::size_t batches, std::size_t locations)
    {
        constexpr std::size_t minimum_samples_per_thread = 16384;
        const std::size_t available = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        const std::size_t samples = batches * locations;
        const std::size_t useful = std::max<std::size_t>(
            1,
            (samples + minimum_samples_per_thread - 1) / minimum_samples_per_thread);
        return std::min(available, useful);
    }

    template <typename Worker>
    void parallel_samples(std::size_t batches, std::size_t locations, Worker worker)
    {
        const std::size_t workers = sample_thread_count(batches, locations);
        if (workers == 1)
        {
            worker(0, batches, 0, locations);
            return;
        }
        std::vector<std::thread> threads;
        threads.reserve(workers - 1);
        if (batches >= workers)
        {
            for (std::size_t index = 1; index < workers; ++index)
            {
                const std::size_t begin = index * batches / workers;
                const std::size_t end = (index + 1) * batches / workers;
                threads.emplace_back(worker, begin, end, 0, locations);
            }
            worker(0, batches / workers, 0, locations);
        }
        else
        {
            for (std::size_t index = 1; index < workers; ++index)
            {
                const std::size_t begin = index * locations / workers;
                const std::size_t end = (index + 1) * locations / workers;
                threads.emplace_back(worker, 0, batches, begin, end);
            }
            worker(0, batches, 0, locations / workers);
        }
        for (auto& thread : threads)
        {
            thread.join();
        }
    }

    void clear_output(
        complex64* output,
        std::size_t batches,
        std::size_t rank,
        std::size_t locations,
        std::size_t begin,
        std::size_t end)
    {
        for (std::size_t batch = 0; batch < batches; ++batch)
        {
            for (std::size_t coefficient = 0; coefficient < rank; ++coefficient)
            {
                complex64* row = output + (batch * rank + coefficient) * locations;
                std::fill(row + begin, row + end, complex64{});
            }
        }
    }

    template <typename Weight>
    void scalar_matvec(
        const Weight* values,
        const complex64* input,
        complex64* output,
        std::size_t batches,
        std::size_t rank,
        std::size_t locations,
        std::size_t begin,
        std::size_t end)
    {
        clear_output(output, batches, rank, locations, begin, end);
        std::size_t packed = 0;
        for (std::size_t row = 0; row < rank; ++row)
        {
            for (std::size_t column = row; column < rank; ++column, ++packed)
            {
                const Weight* weight = values + packed * locations;
                for (std::size_t batch = 0; batch < batches; ++batch)
                {
                    const complex64* column_input = input + (batch * rank + column) * locations;
                    complex64* row_output = output + (batch * rank + row) * locations;
                    for (std::size_t location = begin; location < end; ++location)
                    {
                        row_output[location] +=
                            static_cast<complex64>(weight[location]) * column_input[location];
                    }
                    if (row == column)
                    {
                        continue;
                    }
                    const complex64* row_input = input + (batch * rank + row) * locations;
                    complex64* column_output = output + (batch * rank + column) * locations;
                    for (std::size_t location = begin; location < end; ++location)
                    {
                        column_output[location] +=
                            std::conj(static_cast<complex64>(weight[location])) *
                            row_input[location];
                    }
                }
            }
        }
    }

#if PULSERVER_X86_TARGETS
    bool supports_avx512()
    {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512f");
    }

    bool supports_avx2()
    {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2");
    }

    __attribute__((target("avx2"))) void avx2_real_matvec(
        const float* values,
        const complex64* input,
        complex64* output,
        std::size_t batches,
        std::size_t rank,
        std::size_t locations,
        std::size_t begin,
        std::size_t end)
    {
        clear_output(output, batches, rank, locations, begin, end);
        std::size_t packed = 0;
        for (std::size_t row = 0; row < rank; ++row)
        {
            for (std::size_t column = row; column < rank; ++column, ++packed)
            {
                const float* weight = values + packed * locations;
                for (std::size_t batch = 0; batch < batches; ++batch)
                {
                    const float* column_input =
                        reinterpret_cast<const float*>(input + (batch * rank + column) * locations);
                    float* row_output =
                        reinterpret_cast<float*>(output + (batch * rank + row) * locations);
                    std::size_t location = begin;
                    for (; location + 4 <= end; location += 4)
                    {
                        const __m128 weights4 = _mm_loadu_ps(weight + location);
                        const __m128 low = _mm_unpacklo_ps(weights4, weights4);
                        const __m128 high = _mm_unpackhi_ps(weights4, weights4);
                        __m256 weights8 = _mm256_castps128_ps256(low);
                        weights8 = _mm256_insertf128_ps(weights8, high, 1);
                        const __m256 source = _mm256_loadu_ps(column_input + 2 * location);
                        const __m256 destination = _mm256_loadu_ps(row_output + 2 * location);
                        _mm256_storeu_ps(
                            row_output + 2 * location,
                            _mm256_add_ps(destination, _mm256_mul_ps(weights8, source)));
                    }
                    for (; location < end; ++location)
                    {
                        reinterpret_cast<complex64*>(row_output)[location] += weight[location] *
                            reinterpret_cast<const complex64*>(column_input)[location];
                    }
                    if (row == column)
                    {
                        continue;
                    }
                    const float* row_input =
                        reinterpret_cast<const float*>(input + (batch * rank + row) * locations);
                    float* column_output =
                        reinterpret_cast<float*>(output + (batch * rank + column) * locations);
                    location = begin;
                    for (; location + 4 <= end; location += 4)
                    {
                        const __m128 weights4 = _mm_loadu_ps(weight + location);
                        const __m128 low = _mm_unpacklo_ps(weights4, weights4);
                        const __m128 high = _mm_unpackhi_ps(weights4, weights4);
                        __m256 weights8 = _mm256_castps128_ps256(low);
                        weights8 = _mm256_insertf128_ps(weights8, high, 1);
                        const __m256 source = _mm256_loadu_ps(row_input + 2 * location);
                        const __m256 destination = _mm256_loadu_ps(column_output + 2 * location);
                        _mm256_storeu_ps(
                            column_output + 2 * location,
                            _mm256_add_ps(destination, _mm256_mul_ps(weights8, source)));
                    }
                    for (; location < end; ++location)
                    {
                        reinterpret_cast<complex64*>(column_output)[location] += weight[location] *
                            reinterpret_cast<const complex64*>(row_input)[location];
                    }
                }
            }
        }
    }

    __attribute__((target("avx2"))) inline __m256 complex_product_avx2(
        __m256 weights,
        __m256 source,
        bool conjugate)
    {
        const __m256 real = _mm256_moveldup_ps(weights);
        const __m256 imaginary = _mm256_movehdup_ps(weights);
        const __m256 swapped = _mm256_permute_ps(source, 0xB1);
        __m256 cross = _mm256_mul_ps(imaginary, swapped);
        const __m256 signs = conjugate
            ? _mm256_setr_ps(1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F)
            : _mm256_setr_ps(-1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F);
        cross = _mm256_mul_ps(cross, signs);
        return _mm256_add_ps(_mm256_mul_ps(real, source), cross);
    }

    __attribute__((target("avx2"))) void avx2_complex_matvec(
        const complex64* values,
        const complex64* input,
        complex64* output,
        std::size_t batches,
        std::size_t rank,
        std::size_t locations,
        std::size_t begin,
        std::size_t end)
    {
        clear_output(output, batches, rank, locations, begin, end);
        std::size_t packed = 0;
        for (std::size_t row = 0; row < rank; ++row)
        {
            for (std::size_t column = row; column < rank; ++column, ++packed)
            {
                const float* weight = reinterpret_cast<const float*>(values + packed * locations);
                for (std::size_t batch = 0; batch < batches; ++batch)
                {
                    const float* column_input =
                        reinterpret_cast<const float*>(input + (batch * rank + column) * locations);
                    float* row_output =
                        reinterpret_cast<float*>(output + (batch * rank + row) * locations);
                    std::size_t location = begin;
                    for (; location + 4 <= end; location += 4)
                    {
                        const __m256 product = complex_product_avx2(
                            _mm256_loadu_ps(weight + 2 * location),
                            _mm256_loadu_ps(column_input + 2 * location),
                            false);
                        _mm256_storeu_ps(
                            row_output + 2 * location,
                            _mm256_add_ps(_mm256_loadu_ps(row_output + 2 * location), product));
                    }
                    for (; location < end; ++location)
                    {
                        reinterpret_cast<complex64*>(row_output)[location] +=
                            values[packed * locations + location] *
                            reinterpret_cast<const complex64*>(column_input)[location];
                    }
                    if (row == column)
                    {
                        continue;
                    }
                    const float* row_input =
                        reinterpret_cast<const float*>(input + (batch * rank + row) * locations);
                    float* column_output =
                        reinterpret_cast<float*>(output + (batch * rank + column) * locations);
                    location = begin;
                    for (; location + 4 <= end; location += 4)
                    {
                        const __m256 product = complex_product_avx2(
                            _mm256_loadu_ps(weight + 2 * location),
                            _mm256_loadu_ps(row_input + 2 * location),
                            true);
                        _mm256_storeu_ps(
                            column_output + 2 * location,
                            _mm256_add_ps(_mm256_loadu_ps(column_output + 2 * location), product));
                    }
                    for (; location < end; ++location)
                    {
                        reinterpret_cast<complex64*>(column_output)[location] +=
                            std::conj(values[packed * locations + location]) *
                            reinterpret_cast<const complex64*>(row_input)[location];
                    }
                }
            }
        }
    }

    __attribute__((target("avx512f"))) inline __m512 duplicate_real_weights_avx512(
        const float* weights)
    {
        const __m512 loaded = _mm512_maskz_loadu_ps(0x00FF, weights);
        const __m512i indices = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
        return _mm512_permutexvar_ps(indices, loaded);
    }

    __attribute__((target("avx512f"))) void avx512_real_matvec(
        const float* values,
        const complex64* input,
        complex64* output,
        std::size_t batches,
        std::size_t rank,
        std::size_t locations,
        std::size_t begin,
        std::size_t end)
    {
        clear_output(output, batches, rank, locations, begin, end);
        std::size_t packed = 0;
        for (std::size_t row = 0; row < rank; ++row)
        {
            for (std::size_t column = row; column < rank; ++column, ++packed)
            {
                const float* weight = values + packed * locations;
                for (std::size_t batch = 0; batch < batches; ++batch)
                {
                    const float* column_input =
                        reinterpret_cast<const float*>(input + (batch * rank + column) * locations);
                    float* row_output =
                        reinterpret_cast<float*>(output + (batch * rank + row) * locations);
                    std::size_t location = begin;
                    for (; location + 8 <= end; location += 8)
                    {
                        const __m512 weights = duplicate_real_weights_avx512(weight + location);
                        const __m512 source = _mm512_loadu_ps(column_input + 2 * location);
                        const __m512 destination = _mm512_loadu_ps(row_output + 2 * location);
                        _mm512_storeu_ps(
                            row_output + 2 * location,
                            _mm512_add_ps(destination, _mm512_mul_ps(weights, source)));
                    }
                    for (; location < end; ++location)
                    {
                        reinterpret_cast<complex64*>(row_output)[location] += weight[location] *
                            reinterpret_cast<const complex64*>(column_input)[location];
                    }
                    if (row == column)
                    {
                        continue;
                    }
                    const float* row_input =
                        reinterpret_cast<const float*>(input + (batch * rank + row) * locations);
                    float* column_output =
                        reinterpret_cast<float*>(output + (batch * rank + column) * locations);
                    location = begin;
                    for (; location + 8 <= end; location += 8)
                    {
                        const __m512 weights = duplicate_real_weights_avx512(weight + location);
                        const __m512 source = _mm512_loadu_ps(row_input + 2 * location);
                        const __m512 destination = _mm512_loadu_ps(column_output + 2 * location);
                        _mm512_storeu_ps(
                            column_output + 2 * location,
                            _mm512_add_ps(destination, _mm512_mul_ps(weights, source)));
                    }
                    for (; location < end; ++location)
                    {
                        reinterpret_cast<complex64*>(column_output)[location] += weight[location] *
                            reinterpret_cast<const complex64*>(row_input)[location];
                    }
                }
            }
        }
    }

    __attribute__((target("avx512f"))) inline __m512 complex_product_avx512(
        __m512 weights,
        __m512 source,
        bool conjugate)
    {
        const __m512 real = _mm512_moveldup_ps(weights);
        const __m512 imaginary = _mm512_movehdup_ps(weights);
        const __m512 swapped = _mm512_permute_ps(source, 0xB1);
        __m512 cross = _mm512_mul_ps(imaginary, swapped);
        const __m512 signs = conjugate ? _mm512_setr_ps(
                                             1.0F,
                                             -1.0F,
                                             1.0F,
                                             -1.0F,
                                             1.0F,
                                             -1.0F,
                                             1.0F,
                                             -1.0F,
                                             1.0F,
                                             -1.0F,
                                             1.0F,
                                             -1.0F,
                                             1.0F,
                                             -1.0F,
                                             1.0F,
                                             -1.0F)
                                       : _mm512_setr_ps(
                                             -1.0F,
                                             1.0F,
                                             -1.0F,
                                             1.0F,
                                             -1.0F,
                                             1.0F,
                                             -1.0F,
                                             1.0F,
                                             -1.0F,
                                             1.0F,
                                             -1.0F,
                                             1.0F,
                                             -1.0F,
                                             1.0F,
                                             -1.0F,
                                             1.0F);
        cross = _mm512_mul_ps(cross, signs);
        return _mm512_add_ps(_mm512_mul_ps(real, source), cross);
    }

    __attribute__((target("avx512f"))) void avx512_complex_matvec(
        const complex64* values,
        const complex64* input,
        complex64* output,
        std::size_t batches,
        std::size_t rank,
        std::size_t locations,
        std::size_t begin,
        std::size_t end)
    {
        clear_output(output, batches, rank, locations, begin, end);
        std::size_t packed = 0;
        for (std::size_t row = 0; row < rank; ++row)
        {
            for (std::size_t column = row; column < rank; ++column, ++packed)
            {
                const float* weight = reinterpret_cast<const float*>(values + packed * locations);
                for (std::size_t batch = 0; batch < batches; ++batch)
                {
                    const float* column_input =
                        reinterpret_cast<const float*>(input + (batch * rank + column) * locations);
                    float* row_output =
                        reinterpret_cast<float*>(output + (batch * rank + row) * locations);
                    std::size_t location = begin;
                    for (; location + 8 <= end; location += 8)
                    {
                        const __m512 product = complex_product_avx512(
                            _mm512_loadu_ps(weight + 2 * location),
                            _mm512_loadu_ps(column_input + 2 * location),
                            false);
                        _mm512_storeu_ps(
                            row_output + 2 * location,
                            _mm512_add_ps(_mm512_loadu_ps(row_output + 2 * location), product));
                    }
                    for (; location < end; ++location)
                    {
                        reinterpret_cast<complex64*>(row_output)[location] +=
                            values[packed * locations + location] *
                            reinterpret_cast<const complex64*>(column_input)[location];
                    }
                    if (row == column)
                    {
                        continue;
                    }
                    const float* row_input =
                        reinterpret_cast<const float*>(input + (batch * rank + row) * locations);
                    float* column_output =
                        reinterpret_cast<float*>(output + (batch * rank + column) * locations);
                    location = begin;
                    for (; location + 8 <= end; location += 8)
                    {
                        const __m512 product = complex_product_avx512(
                            _mm512_loadu_ps(weight + 2 * location),
                            _mm512_loadu_ps(row_input + 2 * location),
                            true);
                        _mm512_storeu_ps(
                            column_output + 2 * location,
                            _mm512_add_ps(_mm512_loadu_ps(column_output + 2 * location), product));
                    }
                    for (; location < end; ++location)
                    {
                        reinterpret_cast<complex64*>(column_output)[location] +=
                            std::conj(values[packed * locations + location]) *
                            reinterpret_cast<const complex64*>(row_input)[location];
                    }
                }
            }
        }
    }
#else
    bool supports_avx512()
    {
        return false;
    }
    bool supports_avx2()
    {
        return false;
    }
#endif

    void packed_hermitian_matvec(py::array values, py::array spectrum, py::array output)
    {
        if (values.ndim() != 2 || spectrum.ndim() != 3 || output.ndim() != 3)
        {
            throw std::invalid_argument("expected values (packed, locations) and spectrum/output "
                                        "(batch, rank, locations)");
        }
        if (!py::dtype::of<complex64>().is(spectrum.dtype()) ||
            !py::dtype::of<complex64>().is(output.dtype()))
        {
            throw std::invalid_argument("spectrum and output must be complex64");
        }
        if (!(spectrum.flags() & py::array::c_style) || !(output.flags() & py::array::c_style) ||
            !(values.flags() & py::array::c_style))
        {
            throw std::invalid_argument("all arrays must be C-contiguous");
        }
        const std::size_t batches = static_cast<std::size_t>(spectrum.shape(0));
        const std::size_t rank = packed_rank(static_cast<std::size_t>(values.shape(0)));
        const std::size_t locations = static_cast<std::size_t>(spectrum.shape(2));
        if (static_cast<std::size_t>(spectrum.shape(1)) != rank ||
            static_cast<std::size_t>(values.shape(1)) != locations ||
            output.shape(0) != spectrum.shape(0) || output.shape(1) != spectrum.shape(1) ||
            output.shape(2) != spectrum.shape(2))
        {
            throw std::invalid_argument("packed matrix and spectrum shapes disagree");
        }

        const bool real_values = py::dtype::of<float>().is(values.dtype());
        const bool complex_values = py::dtype::of<complex64>().is(values.dtype());
        if (!real_values && !complex_values)
        {
            throw std::invalid_argument("values must be float32 or complex64");
        }
        const auto* input = static_cast<const complex64*>(spectrum.data());
        auto* result = static_cast<complex64*>(output.mutable_data());
        const auto* real = real_values ? static_cast<const float*>(values.data()) : nullptr;
        const auto* complex =
            complex_values ? static_cast<const complex64*>(values.data()) : nullptr;
        py::gil_scoped_release release;
        parallel_samples(
            batches,
            locations,
            [&](std::size_t batch_begin,
                std::size_t batch_end,
                std::size_t begin,
                std::size_t end)
            {
                const std::size_t worker_batches = batch_end - batch_begin;
                const complex64* worker_input = input + batch_begin * rank * locations;
                complex64* worker_result = result + batch_begin * rank * locations;
#if PULSERVER_X86_TARGETS
                if (supports_avx512())
                {
                    if (real_values)
                    {
                        avx512_real_matvec(
                            real,
                            worker_input,
                            worker_result,
                            worker_batches,
                            rank,
                            locations,
                            begin,
                            end);
                    }
                    else
                    {
                        avx512_complex_matvec(
                            complex,
                            worker_input,
                            worker_result,
                            worker_batches,
                            rank,
                            locations,
                            begin,
                            end);
                    }
                    return;
                }
                if (supports_avx2())
                {
                    if (real_values)
                    {
                        avx2_real_matvec(
                            real,
                            worker_input,
                            worker_result,
                            worker_batches,
                            rank,
                            locations,
                            begin,
                            end);
                    }
                    else
                    {
                        avx2_complex_matvec(
                            complex,
                            worker_input,
                            worker_result,
                            worker_batches,
                            rank,
                            locations,
                            begin,
                            end);
                    }
                    return;
                }
#endif
                if (real_values)
                {
                    scalar_matvec(
                        real,
                        worker_input,
                        worker_result,
                        worker_batches,
                        rank,
                        locations,
                        begin,
                        end);
                }
                else
                {
                    scalar_matvec(
                        complex,
                        worker_input,
                        worker_result,
                        worker_batches,
                        rank,
                        locations,
                        begin,
                        end);
                }
            });
    }

    void wave_psf(py::array phase, py::array y_axis, py::array z_axis, py::array output)
    {
        if (phase.ndim() != 3 || y_axis.ndim() != 1 || z_axis.ndim() != 1 || output.ndim() != 4)
        {
            throw std::invalid_argument(
                "expected phase (models, 2, readout), axes (phase)/(partition), and output "
                "(models, readout, phase, partition)");
        }
        if (!py::dtype::of<float>().is(phase.dtype()) ||
            !py::dtype::of<float>().is(y_axis.dtype()) ||
            !py::dtype::of<float>().is(z_axis.dtype()) ||
            !py::dtype::of<complex64>().is(output.dtype()))
        {
            throw std::invalid_argument("phase/axes must be float32 and output must be complex64");
        }
        if (!(phase.flags() & py::array::c_style) || !(y_axis.flags() & py::array::c_style) ||
            !(z_axis.flags() & py::array::c_style) || !(output.flags() & py::array::c_style))
        {
            throw std::invalid_argument("all arrays must be C-contiguous");
        }

        const std::size_t models = static_cast<std::size_t>(phase.shape(0));
        const std::size_t readout = static_cast<std::size_t>(phase.shape(2));
        const std::size_t phase_size = static_cast<std::size_t>(y_axis.shape(0));
        const std::size_t partition_size = static_cast<std::size_t>(z_axis.shape(0));
        if (phase.shape(1) != 2 || output.shape(0) != phase.shape(0) ||
            output.shape(1) != phase.shape(2) || output.shape(2) != y_axis.shape(0) ||
            output.shape(3) != z_axis.shape(0))
        {
            throw std::invalid_argument("phase, axes, and output shapes disagree");
        }

        const auto* phase_values = static_cast<const float*>(phase.data());
        const auto* y_values = static_cast<const float*>(y_axis.data());
        const auto* z_values = static_cast<const float*>(z_axis.data());
        auto* result = static_cast<complex64*>(output.mutable_data());
        const std::size_t rows = models * readout * phase_size;
        py::gil_scoped_release release;
        parallel_samples(
            1,
            rows,
            [&](std::size_t, std::size_t, std::size_t begin, std::size_t end)
            {
                for (std::size_t row = begin; row < end; ++row)
                {
                    const std::size_t y_index = row % phase_size;
                    const std::size_t model_readout = row / phase_size;
                    const std::size_t readout_index = model_readout % readout;
                    const std::size_t model = model_readout / readout;
                    const float phase_y = phase_values[(model * 2) * readout + readout_index];
                    const float phase_z = phase_values[(model * 2 + 1) * readout + readout_index];
                    complex64* output_row = result + row * partition_size;
                    const float y_phase = phase_y * y_values[y_index];
                    for (std::size_t z_index = 0; z_index < partition_size; ++z_index)
                    {
                        const float angle = -(y_phase + phase_z * z_values[z_index]);
                        output_row[z_index] = complex64(std::cos(angle), std::sin(angle));
                    }
                }
            });
    }

} // namespace

PYBIND11_MODULE(_recon_cpu_wrapper, module)
{
    module.doc() = "Precompiled CPU kernels for Pulserver reconstruction";
    module.def(
        "packed_hermitian_matvec",
        &packed_hermitian_matvec,
        py::arg("values"),
        py::arg("spectrum"),
        py::arg("output"));
    module.def(
        "simd_level",
        []()
        {
            if (supports_avx512())
            {
                return "avx512";
            }
            return supports_avx2() ? "avx2" : "scalar";
        });
    module.def(
        "wave_psf",
        &wave_psf,
        py::arg("phase"),
        py::arg("y_axis"),
        py::arg("z_axis"),
        py::arg("output"));
    module.def("thread_count", &location_thread_count, py::arg("locations"));
    module.def(
        "sample_thread_count",
        &sample_thread_count,
        py::arg("batches"),
        py::arg("locations"));
}
