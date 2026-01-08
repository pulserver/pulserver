"""
Pulseq ordering functions.

This module contains low-level helper functions sorting sampling masks
to achieve desired acquisition orders.

"""

__all__ = [
    'make_centerout_ordering_1d',
    'make_centerout_ordering_2d',
    'make_interleaved_ordering_1d',
    'make_radial_ordering_2d',
    'make_random_ordering_1d',
    'make_random_ordering_2d',
    'make_spiral_ordering_2d',
]

import numpy as np
from numpy.typing import NDArray


def make_interleaved_ordering_1d(n1: int, ngroups: int) -> NDArray[int]:
    """
    Create an 1D interleaved ordering array.

    Parameters
    ----------
    n1 : int
        Number of elements in the 1D array to be sorted.
    ngroups : int
        Target number of interleaved groups.

    Returns
    -------
    NDArray[int]
        Ordering array to perform interleaving, of shape ``(n1,)``.

    Examples
    --------
    To perform even-odd interleaving, we can do as follows.

    First, we import ``pulserver.design``:

    >>> import pulserver.design as pd

    Suppose we have ``10`` slices:

    >>> nslices = 10

    Now, sorting array for our set of slices can be created as:

    >>> ordering = pd.make_interleaved_ordering_1d(nslices, ngroups=2)
    >>> print(ordering)
    [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]

    """
    indexes = []
    ax1 = np.arange(n1)
    for n in range(ngroups):
        indexes.append(ax1[n::ngroups])
    return np.concatenate(indexes)


def make_centerout_ordering_1d(n1: int) -> NDArray[int]:
    """
    Create an 1D center-out ordering array.

    Parameters
    ----------
    n1 : int
        Number of elements in the 1D array to be sorted.

    Returns
    -------
    NDArray[int]
        Ordering array to perform 1D center-out sorting, of shape ``(n1,)``.

    Examples
    --------
    To perform 1D center-out sorting, we can do as follows.

    First, we import ``pulserver.design``:

    >>> import pulserver.design as pd

    Suppose we have ``10`` phase encoding steps:

    >>> nphase_enc = 10

    Now, sorting array for our set of phase encoding amplitudes can be created as:

    >>> ordering = pd.make_centerout_ordering_1d(nslices)
    >>> print(ordering)
    [5 4 6 3 7 2 8 1 9 0]

    """
    ax1 = np.arange(n1)
    order = np.argsort(np.abs(ax1 - n1 // 2))
    return ax1[order]


def make_random_ordering_1d(n1: int) -> NDArray[int]:
    """
    Create an 1D random ordering array.

    Parameters
    ----------
    n1 : int
        Number of elements in the 1D array to be sorted.

    Returns
    -------
    NDArray[int]
        Ordering array to perform 1D random sorting, of shape ``(n1,)``.

    """
    rng = np.random.default_rng()
    return rng.permutation(n1)


def make_radial_ordering_2d(
    n1: int,
    n2: int,
    inc: float,
    theta0: float = 0.0,
    prune: bool = True,
) -> NDArray[int]:
    """
    Create a 2D radial ordering array.

    Parameters
    ----------
    n1 : int
        Matrix size along radial direction. Usually, ``nx = ny = n1``.
    n2 : int
        Number of radial projections.
    inc : float
        Angular increment in ``[rad]``.
    theta0 : float, optional
        Angular offset in ``[rad]``. The default is ``0.0``.
    prune : bool, optional
        If ``True``, remove duplicates from each spoke and make sure
        final sampling pattern has equal number of samples for each spoke.
        The default is ``True``.

    Returns
    -------
    NDArray[int]
        Ordering array to perform 2D radial sorting, of shape ``(n1, n2)``.

    Examples
    --------
    To perform 2D radial sorting, we can do as follows.

    First, we import ``pulserver.design``:

    >>> import numpy as np
    >>> import pulserver.design as pd

    Suppose we have ``(8, 8)`` encoding matrix size (e.g., in ``(ky, kz)`` plane),
    and we want to acquire k-space samples arranging ``kx`` shots in two orthogonal
    ``(ky, kz)`` lines (one parallel to ``ky``, the other to ``kz``):

    >>> nencodes = 8
    >>> nspokes = 2
    >>> increment = np.deg2rad(90.0)

    >>> ordering = pd.make_radial_ordering_2d(nencodes, nspokes, increment)
    >>> print(ordering.T)
    [[ 4 12 20 28 36 44 52 60]
     [32 33 34 35 36 37 38 39]]

    First line represent the flattened indexes for the ``x`` axis of a ``(8, 8)``
    matrix; the second the indexes for the ``y`` axis of the same matrix.

    """
    ax1 = np.arange(n1) - n1 // 2
    ax2 = np.exp(1j * (np.arange(n2) * inc + theta0))
    grid = ax1[:, None] * ax2[None, :]
    grid = np.stack((np.round(grid.real), np.round(grid.imag))).astype(int) + n1 // 2
    grid = np.clip(grid, 0, n1 - 1)
    indexes = np.ravel_multi_index(grid, (n1, n1))
    if prune:
        return _prune_sampling(indexes)
    return indexes


def make_centerout_ordering_2d(
    n1: int,
    n2: int,
    inc: float,
    theta0: float = 0.0,
    prune: bool = True,
) -> NDArray[int]:
    """
    Create a 2D center-out ordering array.

    Parameters
    ----------
    n1 : int
        Matrix size along radial direction. Usually, ``nx = ny = n1``.
    n2 : int
        Number of center-out projections.
    inc : float
        Angular increment in ``[rad]``.
    theta0 : float, optional
        Angular offset in ``[rad]``. The default is ``0.0``.
    prune : bool, optional
        If ``True``, remove duplicates from each spoke and make sure
        final sampling pattern has equal number of samples for each spoke.
        The default is ``True``.

    Returns
    -------
    NDArray[int]
        Ordering array to perform 2D center-out sorting, of shape ``(n1 // 2, n2)``.

    Examples
    --------
    To perform 2D center-out sorting, we can do as follows.

    First, we import ``pulserver.design``:

    >>> import numpy as np
    >>> import pulserver.design as pd

    Suppose we have ``(8, 8)`` encoding matrix size (e.g., in ``(ky, kz)`` plane),
    and we want to acquire k-space samples arranging ``kx`` shots in four orthogonal
    ``(ky, kz)`` lines, one for each semi-axis of the k-space plane:

    >>> nencodes = 8
    >>> nspokes = 4
    >>> increment = np.deg2rad(90.0)

    >>> ordering = pd.make_centerout_ordering_2d(nencodes, nspokes, increment)
    >>> print(ordering.T)
    [[36 44 52 60]
     [36 37 38 39]
     [36 28 20 12]
     [36 35 34 33]]

    First line represent the flattened indexes for the positive ``x`` axis
    of a ``(8, 8)`` matrix, the second the indexes for the positive ``y``,
    third is negative ``x`` axis and last is negative ``y`` axis.

    """
    ax1 = np.arange(n1 // 2)
    ax2 = np.exp(1j * (np.arange(n2) * inc + theta0))
    grid = ax1[:, None] * ax2[None, :]
    grid = np.stack((np.round(grid.real), np.round(grid.imag))).astype(int) + n1 // 2
    grid = np.clip(grid, 0, n1 - 1)
    indexes = np.ravel_multi_index(grid, (n1, n1))
    if prune:
        return _prune_sampling(indexes)
    return indexes


def make_spiral_ordering_2d(
    n1: int, n2: int, inc: float, prune: bool = True
) -> NDArray[int]:
    """
    Create a 2D spiral ordering array.

    Parameters
    ----------
    n1 : int
        Matrix size along radial direction. Usually, ``nx = ny = n1``.
    n2 : int
        Number of spiral interleaves.
    inc : float
        Angular increment in ``[rad]``.
    theta0 : float, optional
        Initial angle in ``[rad]``. The default is ``0.0``.
    prune : bool, optional
        If ``True``, remove duplicates from each interleaf and make sure
        final sampling pattern has equal number of samples for each interleaf.
        The default is ``True``.

    Returns
    -------
    NDArray[int]
        Ordering array to perform 2D spiral sorting, of shape ``(npts, n2)``,
        with ``npts = 0.5 * np.pi * n1**2 / n2``

    Examples
    --------
    To perform 2D spiral sorting, we can do as follows.

    First, we import ``pulserver.design``:

    >>> import numpy as np
    >>> import pulserver.design as pd

    Suppose we have ``(8, 8)`` encoding matrix size (e.g., in ``(ky, kz)`` plane),
    and we want to arrange the samples in 2 Cartesian spiral shots,
    with linear increment:

    >>> nencodes = 8
    >>> nspokes = 2
    >>> increment = np.deg2rad(180.0)

    >>> ordering = pd.make_spiral_ordering_2d(nencodes, nspokes, increment)
    >>> print(ordering.T)
    [[36 52 53 45 46 38 30 29 21 20 19 27 26 34 42 43 51]
     [36 28 27 35 43 44 45 37 29 12 11 18 25 33 41 50 59]]

    First line represent the flattened indexes for first spiral interleaf,
    while the second represents the flattened indexes for second spiral interleaf.
    Together, the two interleaves produce a fully sampled ``(ky, kz)`` k-space.

    """
    indexes = []
    n20 = int(np.ceil(np.pi * n1))  # enforce multiple of target n interleaves
    inc0 = np.deg2rad(360.0 / n20)
    for n in range(n2):
        _inc = n * inc % (2 * np.pi)
        _indexes = make_centerout_ordering_2d(n1, n20, inc0, _inc, False).T
        _indexes = np.concatenate((_indexes[:, [0]], _indexes[:, n::n2]), axis=-1).T
        indexes.append(_indexes.ravel())

    # # Rearrange in spiral order
    indexes = np.stack(indexes, axis=-1)

    if prune:
        return _prune_sampling(indexes)

    return indexes


def make_random_ordering_2d(n1: int, n2: int) -> NDArray[int]:
    """
    Create a 2D random ordering array.

    Parameters
    ----------
    n1 : int
        Matrix size along first direction.
    n2 : int
        Matrix size along second direction.

    Returns
    -------
    NDArray[int]
        Ordering array to perform 2D random sorting, of shape ``(n1, n2)``.

    Examples
    --------
    To perform 2D random sorting, we can do as follows.

    First, we import ``pulserver.design``:

    >>> import numpy as np
    >>> import pulserver.design as pd

    Suppose we have ``(8, 8)`` encoding matrix size (e.g., in ``(ky, kz)`` plane),
    and we want to randomly permute its elements:

    >>> ny, nz = 8, 8
    >>> encoding = np.random.rand(ny, nz)
    >>> ordering = pd.make_random_ordering_2d(8, 8)
    >>> print(ordering.shape)
    (8, 8)

    Now, to apply the ordering, we can do as follows:

    >>> encoding = encoding.ravel()[ordering]

    """
    rng = np.random.default_rng()
    return rng.permutation(n1 * n2).reshape(n1, n2)


# %% Internal helpers
def _prune_sampling(samples: NDArray[int]) -> NDArray[int]:
    """Prune sampling to remove duplicated indexes."""
    uniq_cols = [
        np.unique(samples[:, j], return_index=True) for j in range(samples.shape[1])
    ]
    uniq_cols_sorted = [col[np.argsort(idx)] for col, idx in uniq_cols]
    m_min = min(len(col) for col in uniq_cols_sorted)
    return np.column_stack([col[:m_min] for col in uniq_cols_sorted])
