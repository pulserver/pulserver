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


_TRAVERSALS = {
    "sequential": sequential,
    "reverse": reverse,
    "interleaved": interleaved,
    "center_out": center_out,
    "outside_in": outside_in,
}


def calc_traversal_order(n: int, order: str = "sequential", *, seed: int = 0) -> np.ndarray:
    """Return the order in which to visit ``n`` positions of a one-dimensional loop.

    The ordering primitive behind any outer loop that is a plain count —
    3D partitions, slices, inversion times, diffusion directions, dynamic
    frames. Each scheme is just a permutation of ``range(n)``, so the loop
    stays yours: iterate the result instead of ``range(n)``.

    Parameters
    ----------
    n : int
        Number of positions.
    order : {'sequential', 'reverse', 'interleaved', 'center_out', 'outside_in', 'random'}, optional
        Traversal scheme. ``'interleaved'`` takes even positions then odd ones
        (the standard multi-slice choice, maximising the gap between
        neighbours); ``'center_out'`` starts at the middle, which is the usual
        choice for a partition loop whose contrast is set by the first echoes.
    seed : int, optional
        Seed for ``order='random'``.

    Returns
    -------
    numpy.ndarray
        Integer permutation of ``range(n)``.

    Raises
    ------
    ValueError
        If ``n`` is negative or ``order`` is unknown.

    Examples
    --------
    >>> from pulserver.design._lowlevel import calc_traversal_order
    >>> calc_traversal_order(6, "interleaved").tolist()
    [0, 2, 4, 1, 3, 5]
    >>> calc_traversal_order(5, "center_out").tolist()
    [2, 1, 3, 0, 4]

    Drive a stack-of-spirals partition loop centre-out, with the in-plane
    angle advancing continuously across the whole scan::

        rotations = iter(design.make_noncartesian_2d_sampling(matrix, views=nz * nviews).to_rotations())
        for par in _lowlevel.calc_traversal_order(nz, "center_out"):
            for view in range(nviews):
                readout.set_state(lin_idx=view, par_idx=int(par), rotation=next(rotations))
                for _block in readout:
                    seq.add_block(*_block)
    See Also
    --------
    pulserver.design.make_slice_loop : slice/SMS grouping built on this.
    """
    if order == "random":
        return random_order(n, seed)
    try:
        return _TRAVERSALS[order](n)
    except KeyError:
        raise ValueError(
            f"unknown order {order!r}; expected one of {', '.join(sorted((*_TRAVERSALS, 'random')))}"
        ) from None


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
    >>> from pulserver.design._lowlevel import calc_chunk_indices
    >>> calc_chunk_indices([0, 1, 2, 3, 4], 2)
    [[0, 1], [2, 3], [4]]
    """
    size = max(1, int(size))
    return [indices[i : i + size] for i in range(0, len(indices), size)]
