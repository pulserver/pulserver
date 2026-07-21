"""Core sampling-plan data model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _readonly_copy(value, *, dtype=None):
    out = np.array(value, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class SamplingPattern(Sequence[np.ndarray]):
    """Sampled support plus its shot-by-shot acquisition order.

    ``support`` contains Cartesian coordinates or non-Cartesian tilts.
    Each entry of ``order`` contains indices into that support for one shot.
    Variable train lengths are intentionally supported.
    """

    support: np.ndarray
    order: tuple[np.ndarray, ...]
    mask: np.ndarray | None = None

    def __post_init__(self):
        support = np.asarray(self.support)
        if support.ndim == 0:
            support = support.reshape(1, 1)
        elif support.ndim == 1:
            support = support.reshape(-1, 1)
        elif support.ndim != 2:
            raise ValueError("support must be scalar, 1D, or 2D")
        if not np.issubdtype(support.dtype, np.number):
            raise TypeError("support must be numeric")
        if not np.all(np.isfinite(support)):
            raise ValueError("support must contain only finite values")
        support = _readonly_copy(support)

        order = tuple(self.order)
        normalized = []
        for shot in order:
            indices = np.asarray(shot)
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                raise TypeError("each order entry must be a one-dimensional integer array")
            indices = _readonly_copy(indices, dtype=np.intp)
            if indices.size and (indices.min() < 0 or indices.max() >= len(support)):
                raise IndexError("order contains an out-of-range support index")
            normalized.append(indices)

        mask = None
        if self.mask is not None:
            mask = _readonly_copy(self.mask, dtype=bool)
            if int(np.count_nonzero(mask)) != len(support):
                raise ValueError("mask sample count must equal support length")

        object.__setattr__(self, "support", support)
        object.__setattr__(self, "order", tuple(normalized))
        object.__setattr__(self, "mask", mask)

    @property
    def n_shots(self) -> int:
        return len(self.order)

    @property
    def n_samples(self) -> int:
        return sum(len(shot) for shot in self.order)

    @property
    def inner_lengths(self) -> tuple[int, ...]:
        return tuple(len(shot) for shot in self.order)

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, shot):
        if isinstance(shot, slice):
            return tuple(self.support[idx] for idx in self.order[shot])
        return self.support[self.order[shot]]

    def shot_indices(self, shot: int) -> np.ndarray:
        return self.order[shot]

    def flatten(self) -> np.ndarray:
        if not self.order:
            return np.empty((0, self.support.shape[1]), dtype=self.support.dtype)
        return self.support[np.concatenate(self.order)]

    def relative(self, shot: int) -> np.ndarray:
        values = self[shot]
        if not len(values):
            return values.copy()
        return values - values[0]

    def increments(self, shot: int) -> np.ndarray:
        return np.diff(self[shot], axis=0)
