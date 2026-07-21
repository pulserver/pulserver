"""Slice/SMS acquisition grouping and frequency-offset helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ordering


@dataclass(frozen=True)
class SliceGroup:
    group_index: int
    slice_indices: tuple[int, ...]
    positions_m: np.ndarray

    def __post_init__(self):
        positions = np.array(self.positions_m, dtype=float, copy=True)
        if positions.ndim != 1 or len(positions) != len(self.slice_indices) or not np.all(np.isfinite(positions)):
            raise ValueError("positions_m must be a finite 1D value per slice")
        positions.setflags(write=False)
        object.__setattr__(self, "positions_m", positions)

    def frequency_offsets_hz(self, gradient_hz_per_m: float) -> np.ndarray:
        gradient_hz_per_m = float(gradient_hz_per_m)
        if not np.isfinite(gradient_hz_per_m):
            raise ValueError("gradient_hz_per_m must be finite")
        return self.positions_m * gradient_hz_per_m


def slice_groups(num_slices, spacing_m, *, order="interleaved", sms_factor=1):
    num_slices, sms_factor = int(num_slices), int(sms_factor)
    spacing_m = float(spacing_m)
    if num_slices <= 0 or sms_factor <= 0 or num_slices % sms_factor:
        raise ValueError("num_slices must be positive and divisible by sms_factor")
    if not np.isfinite(spacing_m) or spacing_m == 0:
        raise ValueError("spacing_m must be finite and nonzero")
    n_groups = num_slices // sms_factor
    functions = {
        "sequential": ordering.sequential,
        "reverse": ordering.reverse,
        "interleaved": ordering.interleaved,
        "center_out": ordering.center_out,
        "outside_in": ordering.outside_in,
    }
    if order not in functions:
        raise ValueError(f"unknown slice order {order!r}")
    positions = (np.arange(num_slices) - (num_slices - 1) / 2.0) * spacing_m
    groups = []
    for group_index in functions[order](n_groups):
        members = tuple(int(group_index + band * n_groups) for band in range(sms_factor))
        groups.append(SliceGroup(int(group_index), members, positions[list(members)]))
    return tuple(groups)
