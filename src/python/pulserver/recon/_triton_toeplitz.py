"""Optional Triton kernels for compact Toeplitz multiplication."""

from __future__ import annotations

from typing import Any

import triton
import triton.language as tl


_DIRECT_CONFIGS = [
    triton.Config({"block": 64}, num_warps=2),
    triton.Config({"block": 128}, num_warps=4),
    triton.Config({"block": 256}, num_warps=4),
    triton.Config({"block": 512}, num_warps=8),
]
_DIRECT_CHOICES: dict[tuple[Any, ...], str] = {}


@triton.autotune(configs=_DIRECT_CONFIGS, key=["rank"])
@triton.jit
def _packed_real_matvec_direct_reuse(
    output_ptr,
    spectrum_ptr,
    packed_ptr,
    indices_ptr,
    packed_stride: tl.constexpr,
    n_grid: tl.constexpr,
    n_locations: tl.constexpr,
    rank: tl.constexpr,
    block: tl.constexpr,
):
    """Direct-bank kernel retaining every output accumulator per tile."""
    locations = tl.program_id(0) * block + tl.arange(0, block)
    batch = tl.program_id(1)
    mask = locations < n_locations
    spatial_index = tl.load(indices_ptr + locations, mask=mask, other=0).to(tl.int64)
    output_real = ()
    output_imaginary = ()
    for output_coefficient in tl.static_range(0, rank):
        real = tl.zeros((block,), dtype=tl.float32)
        imaginary = tl.zeros((block,), dtype=tl.float32)
        for input_coefficient in tl.static_range(0, rank):
            spectrum_index = (batch * rank + input_coefficient) * n_grid + spatial_index
            value_real = tl.load(
                spectrum_ptr + 2 * spectrum_index,
                mask=mask,
                other=0.0,
            )
            value_imaginary = tl.load(
                spectrum_ptr + 2 * spectrum_index + 1,
                mask=mask,
                other=0.0,
            )
            lower = min(output_coefficient, input_coefficient)
            upper = max(output_coefficient, input_coefficient)
            packed_index = lower * rank - lower * (lower - 1) // 2 + upper - lower
            weight = tl.load(
                packed_ptr + packed_index * packed_stride + locations,
                mask=mask,
                other=0.0,
            )
            real += weight * value_real
            imaginary += weight * value_imaginary
        output_real += (real,)
        output_imaginary += (imaginary,)
    for output_coefficient in tl.static_range(0, rank):
        output_index = (batch * rank + output_coefficient) * n_grid + spatial_index
        tl.store(
            output_ptr + 2 * output_index,
            output_real[output_coefficient],
            mask=mask,
        )
        tl.store(
            output_ptr + 2 * output_index + 1,
            output_imaginary[output_coefficient],
            mask=mask,
        )


@triton.autotune(configs=_DIRECT_CONFIGS, key=["rank"])
@triton.jit
def _packed_complex_matvec_direct_reuse(
    output_ptr,
    spectrum_ptr,
    packed_ptr,
    indices_ptr,
    packed_row_stride: tl.constexpr,
    packed_location_stride: tl.constexpr,
    n_grid: tl.constexpr,
    n_locations: tl.constexpr,
    rank: tl.constexpr,
    block: tl.constexpr,
):
    """Complex-Hermitian direct-bank kernel with cross-output reuse."""
    locations = tl.program_id(0) * block + tl.arange(0, block)
    batch = tl.program_id(1)
    mask = locations < n_locations
    spatial_index = tl.load(indices_ptr + locations, mask=mask, other=0).to(tl.int64)
    output_real = ()
    output_imaginary = ()
    for output_coefficient in tl.static_range(0, rank):
        real = tl.zeros((block,), dtype=tl.float32)
        imaginary = tl.zeros((block,), dtype=tl.float32)
        for input_coefficient in tl.static_range(0, rank):
            spectrum_index = (batch * rank + input_coefficient) * n_grid + spatial_index
            value_real = tl.load(
                spectrum_ptr + 2 * spectrum_index,
                mask=mask,
                other=0.0,
            )
            value_imaginary = tl.load(
                spectrum_ptr + 2 * spectrum_index + 1,
                mask=mask,
                other=0.0,
            )
            lower = min(output_coefficient, input_coefficient)
            upper = max(output_coefficient, input_coefficient)
            packed_index = lower * rank - lower * (lower - 1) // 2 + upper - lower
            weight_index = (
                packed_index * packed_row_stride + locations * packed_location_stride
            )
            weight_real = tl.load(
                packed_ptr + weight_index,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            weight_imaginary = tl.load(
                packed_ptr + weight_index + 1,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            if output_coefficient > input_coefficient:
                weight_imaginary = -weight_imaginary
            real += weight_real * value_real - weight_imaginary * value_imaginary
            imaginary += weight_real * value_imaginary + weight_imaginary * value_real
        output_real += (real,)
        output_imaginary += (imaginary,)
    for output_coefficient in tl.static_range(0, rank):
        output_index = (batch * rank + output_coefficient) * n_grid + spatial_index
        tl.store(
            output_ptr + 2 * output_index,
            output_real[output_coefficient],
            mask=mask,
        )
        tl.store(
            output_ptr + 2 * output_index + 1,
            output_imaginary[output_coefficient],
            mask=mask,
        )


@triton.autotune(configs=_DIRECT_CONFIGS, key=["rank"])
@triton.jit
def _packed_real_matvec_direct(
    output_ptr,
    spectrum_ptr,
    packed_ptr,
    indices_ptr,
    packed_stride: tl.constexpr,
    n_grid: tl.constexpr,
    n_locations: tl.constexpr,
    rank: tl.constexpr,
    block: tl.constexpr,
):
    """Read and write full padded coefficient banks at retained locations."""
    locations = tl.program_id(0) * block + tl.arange(0, block)
    batch_output = tl.program_id(1)
    mask = locations < n_locations
    spatial_index = tl.load(indices_ptr + locations, mask=mask, other=0).to(tl.int64)
    batch = batch_output // rank
    output_coefficient = batch_output - batch * rank
    real = tl.zeros((block,), dtype=tl.float32)
    imaginary = tl.zeros((block,), dtype=tl.float32)
    for input_coefficient in tl.static_range(0, rank):
        spectrum_index = (batch * rank + input_coefficient) * n_grid + spatial_index
        value_real = tl.load(
            spectrum_ptr + 2 * spectrum_index,
            mask=mask,
            other=0.0,
        )
        value_imaginary = tl.load(
            spectrum_ptr + 2 * spectrum_index + 1,
            mask=mask,
            other=0.0,
        )
        lower = tl.minimum(output_coefficient, input_coefficient)
        upper = tl.maximum(output_coefficient, input_coefficient)
        packed_index = lower * rank - lower * (lower - 1) // 2 + upper - lower
        weight = tl.load(
            packed_ptr + packed_index * packed_stride + locations,
            mask=mask,
            other=0.0,
        )
        real += weight * value_real
        imaginary += weight * value_imaginary
    output_index = batch_output * n_grid + spatial_index
    tl.store(output_ptr + 2 * output_index, real, mask=mask)
    tl.store(output_ptr + 2 * output_index + 1, imaginary, mask=mask)


@triton.autotune(configs=_DIRECT_CONFIGS, key=["rank"])
@triton.jit
def _packed_complex_matvec_direct(
    output_ptr,
    spectrum_ptr,
    packed_ptr,
    indices_ptr,
    packed_row_stride: tl.constexpr,
    packed_location_stride: tl.constexpr,
    n_grid: tl.constexpr,
    n_locations: tl.constexpr,
    rank: tl.constexpr,
    block: tl.constexpr,
):
    """Complex-Hermitian counterpart of ``_packed_real_matvec_direct``."""
    locations = tl.program_id(0) * block + tl.arange(0, block)
    batch_output = tl.program_id(1)
    mask = locations < n_locations
    spatial_index = tl.load(indices_ptr + locations, mask=mask, other=0).to(tl.int64)
    batch = batch_output // rank
    output_coefficient = batch_output - batch * rank
    real = tl.zeros((block,), dtype=tl.float32)
    imaginary = tl.zeros((block,), dtype=tl.float32)
    for input_coefficient in tl.static_range(0, rank):
        spectrum_index = (batch * rank + input_coefficient) * n_grid + spatial_index
        value_real = tl.load(
            spectrum_ptr + 2 * spectrum_index,
            mask=mask,
            other=0.0,
        )
        value_imaginary = tl.load(
            spectrum_ptr + 2 * spectrum_index + 1,
            mask=mask,
            other=0.0,
        )
        lower = tl.minimum(output_coefficient, input_coefficient)
        upper = tl.maximum(output_coefficient, input_coefficient)
        packed_index = lower * rank - lower * (lower - 1) // 2 + upper - lower
        weight_index = (
            packed_index * packed_row_stride + locations * packed_location_stride
        )
        weight_real = tl.load(
            packed_ptr + weight_index,
            mask=mask,
            other=0.0,
        )
        weight_imaginary = tl.load(
            packed_ptr + weight_index + 1,
            mask=mask,
            other=0.0,
        )
        weight_imaginary = tl.where(
            output_coefficient > input_coefficient,
            -weight_imaginary,
            weight_imaginary,
        )
        real += weight_real * value_real - weight_imaginary * value_imaginary
        imaginary += weight_real * value_imaginary + weight_imaginary * value_real
    output_index = batch_output * n_grid + spatial_index
    tl.store(output_ptr + 2 * output_index, real, mask=mask)
    tl.store(output_ptr + 2 * output_index + 1, imaginary, mask=mask)


@triton.jit
def _packed_real_matvec_inplace(
    supported_ptr,
    packed_ptr,
    packed_stride: tl.constexpr,
    n_locations: tl.constexpr,
    location_offset: tl.constexpr,
    count: tl.constexpr,
    rank: tl.constexpr,
    block: tl.constexpr,
):
    locations = tl.program_id(0) * block + tl.arange(0, block)
    batch = tl.program_id(1)
    mask = locations < count
    absolute = location_offset + locations
    output_real = ()
    output_imaginary = ()
    for output_coefficient in tl.static_range(0, rank):
        real = tl.zeros((block,), dtype=tl.float32)
        imaginary = tl.zeros((block,), dtype=tl.float32)
        for input_coefficient in tl.static_range(0, rank):
            spectrum_index = (batch * rank + input_coefficient) * n_locations + absolute
            value_real = tl.load(
                supported_ptr + 2 * spectrum_index,
                mask=mask,
                other=0.0,
            )
            value_imaginary = tl.load(
                supported_ptr + 2 * spectrum_index + 1,
                mask=mask,
                other=0.0,
            )
            lower = min(output_coefficient, input_coefficient)
            upper = max(output_coefficient, input_coefficient)
            packed_index = lower * rank - lower * (lower - 1) // 2 + upper - lower
            weight = tl.load(
                packed_ptr + packed_index * packed_stride + locations,
                mask=mask,
                other=0.0,
            )
            real += weight * value_real
            imaginary += weight * value_imaginary
        output_real += (real,)
        output_imaginary += (imaginary,)
    for output_coefficient in tl.static_range(0, rank):
        output_index = (batch * rank + output_coefficient) * n_locations + absolute
        tl.store(
            supported_ptr + 2 * output_index,
            output_real[output_coefficient],
            mask=mask,
        )
        tl.store(
            supported_ptr + 2 * output_index + 1,
            output_imaginary[output_coefficient],
            mask=mask,
        )


@triton.jit
def _packed_complex_matvec_inplace(
    supported_ptr,
    packed_ptr,
    packed_row_stride: tl.constexpr,
    packed_location_stride: tl.constexpr,
    n_locations: tl.constexpr,
    location_offset: tl.constexpr,
    count: tl.constexpr,
    rank: tl.constexpr,
    block: tl.constexpr,
):
    locations = tl.program_id(0) * block + tl.arange(0, block)
    batch = tl.program_id(1)
    mask = locations < count
    absolute = location_offset + locations
    output_real = ()
    output_imaginary = ()
    for output_coefficient in tl.static_range(0, rank):
        real = tl.zeros((block,), dtype=tl.float32)
        imaginary = tl.zeros((block,), dtype=tl.float32)
        for input_coefficient in tl.static_range(0, rank):
            spectrum_index = (batch * rank + input_coefficient) * n_locations + absolute
            value_real = tl.load(
                supported_ptr + 2 * spectrum_index,
                mask=mask,
                other=0.0,
            )
            value_imaginary = tl.load(
                supported_ptr + 2 * spectrum_index + 1,
                mask=mask,
                other=0.0,
            )
            lower = min(output_coefficient, input_coefficient)
            upper = max(output_coefficient, input_coefficient)
            packed_index = lower * rank - lower * (lower - 1) // 2 + upper - lower
            weight_index = (
                packed_index * packed_row_stride + locations * packed_location_stride
            )
            weight_real = tl.load(
                packed_ptr + weight_index,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            weight_imaginary = tl.load(
                packed_ptr + weight_index + 1,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            if output_coefficient > input_coefficient:
                weight_imaginary = -weight_imaginary
            real += weight_real * value_real - weight_imaginary * value_imaginary
            imaginary += weight_real * value_imaginary + weight_imaginary * value_real
        output_real += (real,)
        output_imaginary += (imaginary,)
    for output_coefficient in tl.static_range(0, rank):
        output_index = (batch * rank + output_coefficient) * n_locations + absolute
        tl.store(
            supported_ptr + 2 * output_index,
            output_real[output_coefficient],
            mask=mask,
        )
        tl.store(
            supported_ptr + 2 * output_index + 1,
            output_imaginary[output_coefficient],
            mask=mask,
        )


def packed_real_matvec_inplace(
    supported: Any,
    packed: Any,
    *,
    location_start: int,
    count: int,
) -> None:
    """Apply packed real Hermitian matrices to complex spectra in place."""
    import torch

    block = 256
    grid = (
        triton.cdiv(count, block),
        supported.shape[0],
    )
    _packed_real_matvec_inplace[grid](
        supported.view(torch.float32),
        packed,
        packed_stride=packed.stride(0),
        n_locations=supported.shape[-1],
        location_offset=location_start,
        count=count,
        rank=supported.shape[1],
        block=block,
    )


def packed_complex_matvec_inplace(
    supported: Any,
    packed: Any,
    *,
    location_start: int,
    count: int,
) -> None:
    """Apply packed complex Hermitian matrices to spectra in place."""
    import torch

    block = 256
    grid = (
        triton.cdiv(count, block),
        supported.shape[0],
    )
    storage, row_stride, location_stride = _complex_storage(packed)
    _packed_complex_matvec_inplace[grid](
        supported.view(torch.float32),
        storage,
        packed_row_stride=row_stride,
        packed_location_stride=location_stride,
        n_locations=supported.shape[-1],
        location_offset=location_start,
        count=count,
        rank=supported.shape[1],
        block=block,
    )


def packed_real_matvec_direct(
    output: Any,
    spectrum: Any,
    packed: Any,
    indices: Any,
) -> str:
    """Apply real packed matrices and return the autotuned algorithm name."""
    import torch

    def reuse() -> None:
        grid = lambda meta: (
            triton.cdiv(indices.numel(), meta["block"]),
            spectrum.shape[0],
        )
        _packed_real_matvec_direct_reuse[grid](
            output.view(torch.float32),
            spectrum.view(torch.float32),
            packed,
            indices,
            packed_stride=packed.stride(0),
            n_grid=spectrum[0, 0].numel(),
            n_locations=indices.numel(),
            rank=spectrum.shape[1],
        )

    def independent() -> None:
        grid = lambda meta: (
            triton.cdiv(indices.numel(), meta["block"]),
            spectrum.shape[0] * spectrum.shape[1],
        )
        _packed_real_matvec_direct[grid](
            output.view(torch.float32),
            spectrum.view(torch.float32),
            packed,
            indices,
            packed_stride=packed.stride(0),
            n_grid=spectrum[0, 0].numel(),
            n_locations=indices.numel(),
            rank=spectrum.shape[1],
        )

    return _autotuned_direct(
        spectrum,
        packed,
        indices,
        {"reuse": reuse, "independent": independent},
    )


def packed_complex_matvec_direct(
    output: Any,
    spectrum: Any,
    packed: Any,
    indices: Any,
) -> str:
    """Apply complex packed matrices and return the autotuned algorithm name."""
    import torch

    storage, row_stride, location_stride = _complex_storage(packed)

    def reuse() -> None:
        grid = lambda meta: (
            triton.cdiv(indices.numel(), meta["block"]),
            spectrum.shape[0],
        )
        _packed_complex_matvec_direct_reuse[grid](
            output.view(torch.float32),
            spectrum.view(torch.float32),
            storage,
            indices,
            packed_row_stride=row_stride,
            packed_location_stride=location_stride,
            n_grid=spectrum[0, 0].numel(),
            n_locations=indices.numel(),
            rank=spectrum.shape[1],
        )

    def independent() -> None:
        grid = lambda meta: (
            triton.cdiv(indices.numel(), meta["block"]),
            spectrum.shape[0] * spectrum.shape[1],
        )
        _packed_complex_matvec_direct[grid](
            output.view(torch.float32),
            spectrum.view(torch.float32),
            storage,
            indices,
            packed_row_stride=row_stride,
            packed_location_stride=location_stride,
            n_grid=spectrum[0, 0].numel(),
            n_locations=indices.numel(),
            rank=spectrum.shape[1],
        )

    return _autotuned_direct(
        spectrum,
        packed,
        indices,
        {"reuse": reuse, "independent": independent},
    )


def _complex_storage(packed: Any) -> tuple[Any, int, int]:
    """Expose interleaved complex components to Triton.

    Two storage forms reach here and they interleave identically: a complex64
    transfer, whose real and imaginary words already alternate, and the paired
    BF16 form a narrowed complex transfer takes. Both are read as a stream of
    real words, so the kernels differ only in the width of the word they load.
    """
    import torch

    if packed.dtype == torch.complex64 and packed.ndim == 2:
        return packed.view(torch.float32), packed.stride(0) * 2, 2
    if packed.dtype == torch.bfloat16 and packed.ndim == 3 and packed.shape[-1] == 2:
        return packed.reshape(packed.shape[0], -1), packed.stride(0), packed.stride(1)
    raise TypeError(
        "packed complex CUDA kernels require complex64 or paired bfloat16 storage"
    )


def _autotuned_direct(
    spectrum: Any,
    packed: Any,
    indices: Any,
    candidates: dict[str, Any],
) -> str:
    """Benchmark direct-kernel organizations once per device/problem class."""
    import torch

    key = (
        spectrum.device,
        spectrum.shape[0],
        spectrum.shape[1],
        spectrum[0, 0].numel(),
        indices.numel(),
        packed.dtype,
        packed.is_complex(),
        packed.ndim,
    )
    selected = _DIRECT_CHOICES.get(key)
    if selected is None:
        timings = {}
        for name, candidate in candidates.items():
            candidate()
            torch.cuda.synchronize(spectrum.device)
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(3):
                candidate()
            stop.record()
            stop.synchronize()
            timings[name] = start.elapsed_time(stop)
        selected = min(timings, key=timings.__getitem__)
        _DIRECT_CHOICES[key] = selected
    candidates[selected]()
    return selected
