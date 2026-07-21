"""Reusable acquisition-order helpers."""

from __future__ import annotations

from itertools import product

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


def make_outer_product(**dimensions) -> tuple[dict[str, object], ...]:
    """Enumerate every combination of the named outer loop dimensions.

    The loop nest a plugin writes by hand — frames, slice groups, averages —
    flattened into one iterable of keyword dictionaries. The **first** keyword
    varies slowest and the last varies fastest, matching the order the loops
    would have been written in.

    Parameters
    ----------
    **dimensions
        One iterable per outer dimension, named after that dimension.

    Returns
    -------
    tuple of dict
        One dictionary per combination, keyed by the dimension names.

    Examples
    --------
    >>> from pulserver.pypulseq import make_outer_product
    >>> combinations = make_outer_product(frame=range(2), slice_index=range(3))
    >>> len(combinations)
    6
    >>> combinations[0]
    {'frame': 0, 'slice_index': 0}
    >>> combinations[1]
    {'frame': 0, 'slice_index': 1}

    Drive a sequence loop directly with it::

        for outer in make_outer_product(frame=range(20), group=make_slice_groups(8, 3e-3)):
            readout(seq, **outer)
    """
    names = tuple(dimensions)
    values = tuple(tuple(value) for value in dimensions.values())
    return tuple(dict(zip(names, combination, strict=True)) for combination in product(*values))


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
    >>> from pulserver.pypulseq import calc_chunk_indices
    >>> calc_chunk_indices([0, 1, 2, 3, 4], 2)
    [[0, 1], [2, 3], [4]]
    """
    size = max(1, int(size))
    return [indices[i : i + size] for i in range(0, len(indices), size)]


def make_linear_order(n: int, etl: int) -> list[list[int]]:
    """Split ``n`` sequential views into echo trains of length ``etl``.

    The simplest FSE/segmented ordering: views are acquired in increasing
    index order and cut into consecutive trains. Equivalent to
    ``calc_chunk_indices(range(n), etl)``.

    Parameters
    ----------
    n : int
        Number of views.
    etl : int
        Echo-train (or segment) length.

    Returns
    -------
    list of list of int
        One list of view indices per shot.

    Examples
    --------
    >>> from pulserver.pypulseq import make_linear_order
    >>> make_linear_order(6, 3)
    [[0, 1, 2], [3, 4, 5]]

    See Also
    --------
    make_fse_linear_order : same idea over a 2D (ky, kz) point set.
    """
    return calc_chunk_indices(list(range(n)), etl)


def make_outer_inner_order(outer_indices: list[int], inner_len: int) -> list[list[tuple[int, int]]]:
    """Pair every outer index with the full inner range, as one shot each.

    Used when the inner loop is played out in full within one shot — an MPRAGE
    segment or a multi-echo train — so the shot list is the outer loop and each
    shot carries the complete inner sweep.

    Parameters
    ----------
    outer_indices : list of int
        Outer-loop indices, in acquisition order.
    inner_len : int
        Length of the inner loop.

    Returns
    -------
    list of list of tuple
        One shot per outer index; each entry an ``(outer, inner)`` pair.

    Examples
    --------
    >>> from pulserver.pypulseq import make_outer_inner_order
    >>> make_outer_inner_order([5, 6], 2)
    [[(5, 0), (5, 1)], [(6, 0), (6, 1)]]
    """
    return [[(outer, inner) for inner in range(inner_len)] for outer in outer_indices]
