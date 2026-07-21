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

    Every sampling helper in :mod:`pulserver.pypulseq` returns one of these.
    *Where* k-space is sampled and *when* each location is acquired are kept
    separate: ``support`` lists the distinct sampled locations once, and each
    entry of ``order`` indexes into it for one shot. Trains of unequal length
    are supported by construction.

    A pattern is an immutable :class:`~collections.abc.Sequence` of shots:
    ``len(pattern)`` is the shot count and ``pattern[i]`` returns shot ``i``'s
    coordinates, already in acquisition order — so a sequence loop reads as
    ``for shot in pattern:``.

    Parameters
    ----------
    support : array_like
        Shape ``(n_points, D)``: Cartesian ``(ky[, kz])`` indices, spoke
        angles, or unit direction vectors. 1D input is treated as ``(n, 1)``.
    order : tuple of numpy.ndarray
        One integer index array per shot, indexing into ``support``.
    mask : numpy.ndarray, optional
        Cartesian boolean mask, when the support came from one. Its nonzero
        count must equal ``len(support)``.

    Attributes
    ----------
    n_shots : int
        Number of shots — same as ``len(pattern)``.
    n_samples : int
        Total sampled points across all shots (counting repeats).
    inner_lengths : tuple of int
        Train length of each shot.

    Examples
    --------
    >>> from pulserver.pypulseq import radial_2d
    >>> pattern = radial_2d(8, segment_length=4)
    >>> pattern.n_shots, pattern.inner_lengths
    (2, (4, 4))
    >>> pattern[0].shape
    (4, 1)

    Relative steps for a blipped train, instead of absolute coordinates:

    >>> from pulserver.pypulseq import from_relative_shifts
    >>> plan = from_relative_shifts([[0, 0]], [[0, 0], [2, 1]], shape=(8, 4))
    >>> plan.relative(0)
    array([[0, 0],
           [2, 1]])
    >>> plan.increments(0)
    array([[2, 1]])
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
        """Number of shots in the plan."""
        return len(self.order)

    @property
    def n_samples(self) -> int:
        """Total number of acquired points, summed over shots."""
        return sum(len(shot) for shot in self.order)

    @property
    def inner_lengths(self) -> tuple[int, ...]:
        """Train length of each shot."""
        return tuple(len(shot) for shot in self.order)

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, shot):
        if isinstance(shot, slice):
            return tuple(self.support[idx] for idx in self.order[shot])
        return self.support[self.order[shot]]

    def shot_indices(self, shot: int) -> np.ndarray:
        """Return shot ``shot``'s indices into ``support`` (not the values)."""
        return self.order[shot]

    def flatten(self) -> np.ndarray:
        """Return every acquired coordinate, concatenated in acquisition order."""
        if not self.order:
            return np.empty((0, self.support.shape[1]), dtype=self.support.dtype)
        return self.support[np.concatenate(self.order)]

    def relative(self, shot: int) -> np.ndarray:
        """Return shot ``shot``'s coordinates relative to its own first point.

        The prewinder-relative view: what the gradients must reach once the
        shot's start has been played.
        """
        values = self[shot]
        if not len(values):
            return values.copy()
        return values - values[0]

    def increments(self, shot: int) -> np.ndarray:
        """Return the step between consecutive points of shot ``shot``.

        One entry shorter than the shot: these are the blip areas to play
        between echoes.
        """
        return np.diff(self[shot], axis=0)
