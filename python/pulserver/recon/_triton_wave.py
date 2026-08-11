"""Optional Triton kernel for Wave point-spread-function synthesis."""

from __future__ import annotations

__all__: list[str] = []

from typing import Any

import triton
import triton.language as tl


@triton.jit
def _wave_psf(
    output_ptr,
    phase_ptr,
    y_axis_ptr,
    z_axis_ptr,
    count: tl.constexpr,
    readout: tl.constexpr,
    phase_size: tl.constexpr,
    partition_size: tl.constexpr,
    block: tl.constexpr,
):
    """Synthesize independent complex64 PSF samples without angle storage."""
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < count
    z_index = offsets % partition_size
    model_readout_phase = offsets // partition_size
    y_index = model_readout_phase % phase_size
    model_readout = model_readout_phase // phase_size
    readout_index = model_readout % readout
    model = model_readout // readout
    phase_y = tl.load(
        phase_ptr + (model * 2) * readout + readout_index,
        mask=mask,
        other=0.0,
    )
    phase_z = tl.load(
        phase_ptr + (model * 2 + 1) * readout + readout_index,
        mask=mask,
        other=0.0,
    )
    y = tl.load(y_axis_ptr + y_index, mask=mask, other=0.0)
    z = tl.load(z_axis_ptr + z_index, mask=mask, other=0.0)
    angle = -(phase_y * y + phase_z * z)
    tl.store(output_ptr + 2 * offsets, tl.cos(angle), mask=mask)
    tl.store(output_ptr + 2 * offsets + 1, tl.sin(angle), mask=mask)


def wave_psf(
    phase: Any,
    y_axis: Any,
    z_axis: Any,
    output: Any,
) -> None:
    """Fill a contiguous complex64 Wave PSF on CUDA."""
    import torch

    count = output.numel()
    block = 256
    _wave_psf[(triton.cdiv(count, block),)](
        output.view(torch.float32),
        phase,
        y_axis,
        z_axis,
        count=count,
        readout=phase.shape[-1],
        phase_size=y_axis.numel(),
        partition_size=z_axis.numel(),
        block=block,
    )
