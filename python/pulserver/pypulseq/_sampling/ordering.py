"""Reusable acquisition-order helpers."""

from __future__ import annotations

import numpy as np


def _count(n):
    n = int(n)
    if n < 0:
        raise ValueError("n must be nonnegative")
    return n


def sequential(n: int) -> np.ndarray:
    return np.arange(_count(n), dtype=np.intp)


def reverse(n: int) -> np.ndarray:
    return sequential(n)[::-1].copy()


def interleaved(n: int) -> np.ndarray:
    values = sequential(n)
    return np.concatenate((values[::2], values[1::2]))


def center_out(n: int) -> np.ndarray:
    n = _count(n)
    center = (n - 1) / 2.0
    return np.asarray(sorted(range(n), key=lambda index: (abs(index - center), index)), dtype=np.intp)


def outside_in(n: int) -> np.ndarray:
    return center_out(n)[::-1].copy()


def random_order(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).permutation(_count(n))


def calc_chunk_indices(indices: list[int], size: int) -> list[list[int]]:
    """Split a flat index list into consecutive chunks of at most ``size``.

    The generic shot/echo-train splitter: ``indices`` is the acquisition order
    and ``size`` the inner train length. A trailing partial chunk is kept as is
    rather than padded or dropped.

    Parameters
    ----------
    indices : list of int
        Indices in acquisition order.
    size : int
        Maximum chunk length; values below 1 are clamped to 1.

    Returns
    -------
    list of list of int
        Consecutive chunks, in order.

    Examples
    --------
    >>> from pulserver.pypulseq._sampling import calc_chunk_indices
    >>> calc_chunk_indices([0, 1, 2, 3, 4], 2)
    [[0, 1], [2, 3], [4]]
    """
    size = max(1, int(size))
    return [indices[i : i + size] for i in range(0, len(indices), size)]
