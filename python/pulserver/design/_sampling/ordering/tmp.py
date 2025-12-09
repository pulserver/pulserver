
import numpy as np

from numpy.typing import NDArray

def make_linear_coords(n1: int) -> np.ndarray:
    return np.arange(n1)

def make_cartesian_grid(n1: int, n2: int) -> np.ndarray:
    grid = np.mgrid[:n1, :n2]
    return np.ravel_multi_index(grid, (n1, n2))

# %% 1D orderings
def make_interleaved_ordering_1d(n1: int, ngroups: int) -> NDArray:
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
    NDArray
        Ordering array to perform interleaving.
        
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
    

def make_centerout_ordering_1d(n1: int) -> NDArray:
    """
    Create an 1D center-out ordering array.

    Parameters
    ----------
    n1 : int
        Number of elements in the 1D array to be sorted.

    Returns
    -------
    NDArray
        Ordering array to perform 1D center-out sorting.
        
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
    
# %% 2D orderings
def make_radial_ordering_2d(n1: int, n2: int, inc: float, theta0: float = 0.0) -> NDArray:
    """
    Create a 2D radial ordering array.

    Parameters
    ----------
    n1 : int
        Matrix size along radial direction. Usually, ``nx = ny = n1``.
    n2 : int
        Number of radial projections.
    inc : float
        Radial increment in ``[rad``].
    theta0 : float, optional
        Initial angle in ``[rad]``. The default is ``0.0``.

    Returns
    -------
    NDArray
        Ordering array to perform 2D radial sorting.
        
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
    >> increment = np.deg2rad(90.0)
    
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
    grid = np.clip(grid, 0, n1-1)
    return np.ravel_multi_index(grid, (n1, n1))

def make_centerout_ordering_2d(n1: int, n2: int, inc: float, theta0: float = 0.0) -> np.ndarray:
    """
    Create a 2D center-out ordering array.

    Parameters
    ----------
    n1 : int
        Matrix size along radial direction. Usually, ``nx = ny = n1``.
    n2 : int
        Number of center-out projections.
    inc : float
        Radial increment in ``[rad``].
    theta0 : float, optional
        Initial angle in ``[rad]``. The default is ``0.0``.

    Returns
    -------
    NDArray
        Ordering array to perform 2D center-out sorting.
        
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
    >> increment = np.deg2rad(90.0)
    
    >>> ordering = pd.make_centerout_ordering_2d(nencodes, nspokes, increment)
    >>> print(ordering.T)
    [[ 4 12 20 28 36 44 52 60]
     [32 33 34 35 36 37 38 39]]
    
    First line represent the flattened indexes for the ``x`` axis of a ``(8, 8)``
    matrix; the second the indexes for the ``y`` axis of the same matrix.

    """
    ax1 = np.arange(n1 // 2)
    ax2 = np.exp(1j * (np.arange(n2) * inc + theta0))
    grid = ax1[:, None] * ax2[None, :] 
    grid = np.stack((np.round(grid.real), np.round(grid.imag))).astype(int) + n1 // 2
    grid = np.clip(grid, 0, n1-1)
    return np.ravel_multi_index(grid, (n1, n1))

def make_spiral_grid(n1: int, n2: int, inc: float) -> np.ndarray:
    indexes = []
    n20 = int(np.ceil(np.pi * n1 / n2)) # enforce multiple of target n interleaves
    inc0 = np.deg2rad(360.0 / n20)
    for n in range(n2):
        _inc = n * inc % (2 * np.pi)
        _indexes = make_centerout_ordering_2d(n1, n20, inc0, _inc)
        indexes.append(_indexes.ravel())
    
    # # Rearrange in spiral order
    indexes = np.stack(indexes, axis=-1)
    
    return indexes
