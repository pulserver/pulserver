"""
Pulseq sampling functions.

This module contains low-level helper functions to create sampling masks.

"""

from __future__ import annotations

__all__ = [
    'make_caipirinha_sampling',
    'make_partial_fourier_sampling',
    'make_poisson_disk_sampling',
    'make_regular_sampling',
]

import logging
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ._extension._cpd_wrapper import gen_udcpd as _gen_udcpd_bind
from ._extension._cpd_wrapper import gen_vdcpd as _gen_vdcpd_bind

ShapeStr = Literal['CROSS', 'L1_BALL', 'L2_BALL', 'CONES', 'PLANE_AND_CONES']
ShapeLike = str | ShapeStr

_SHAPE_MAP = {
    'CROSS': 0,
    'L1_BALL': 1,
    'L2_BALL': 2,
    'CONES': 3,
    'PLANE_AND_CONES': 4,
}


def require(condition, msg):
    if not condition:
        logging.error(msg)
        raise ValueError(msg)


def make_regular_sampling(
    shape: int,
    accel: int,
    calib: int | None = None,
) -> NDArray[int]:
    """
    Generate regular sampling pattern for GRAPPA/ARC accelerated acquisition.

    Can be used for 2D imaging (i.e., ``ky``).

    Parameters
    ----------
    shape : int | tuple[int]
        Image shape along phase encoding dim ``ny``.
    accel : int, optional
        Target acceleration factor along phase encoding dim ``Ry``.
        Must be ``>= 1``. The default is ``1`` (no acceleration).
    calib : int | None = None, optional
        Image shape along phase encoding dim ``cy``.
        The default is ``None`` (no calibration).

    Returns
    -------
    mask : np.ndarray
        Regular-grid sampling mask of shape ``(ny,)``.

    Examples
    --------
    Basic regular sampling without calibration:

    >>> import numpy as np
    >>> from sampling import make_regular_sampling
    >>> make_regular_sampling(12, 3)
    array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])

    Regular sampling with a small calibration region (cy=4)
    centered in k-space (for shape=12):

    >>> make_regular_sampling(12, 3, calib=4)
    array([1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0])


    """
    require(accel >= 1, f'Ky acceleration must be >= 1, got {accel}')

    # Build mask
    mask = np.zeros(shape, dtype=int)
    mask[::accel] = 1

    # Calib
    if calib is not None:
        mask[shape // 2 - calib // 2 : shape // 2 + calib // 2] = 1

    return mask


def make_caipirinha_sampling(
    shape: int | tuple[int],
    accel: int | tuple[int] = 1.0,
    calib: int | tuple[int] | None = None,
    shift: int = 0,
    elliptical: bool = True,
) -> NDArray[int]:
    """
    Generate regular sampling pattern for CAIPIRINHA accelerated acquisition.

    Can be used for 3D imaging (i.e., ``ky, kz``) or 2D+t (i.e.g, ``ky, t``).

    Parameters
    ----------
    shape : int | tuple[int]
        Image shape along phase encoding dims ``(ny, nz)``.
        If scalar, assume equal size for ``y`` and ``z`` axes.
    accel : int | tuple[int], optional
        Target acceleration factor along phase encoding dims ``(Ry, Rz)``.
        Must be ``>= 1``. If scalar, assume acceleration over ``y``
        only. The default is ``1`` (no acceleration).
    calib : int | tuple[int], optional
        Image shape along phase encoding dims ``(cy, cz)``.
        If scalar, assume equal size for ``y`` and ``z`` axes.
        The default is ``None`` (no calibration).
    shift : int, optional
        Caipirinha shift. The default is ``0`` (standard PI sampling).
    elliptical : bool, optional
        Toggle whether to crop corners of k-space (elliptical sampling).
        The default is ``True``.

    Returns
    -------
    mask : NDArray[int]
        Regular-grid sampling mask of shape ``(ny, nz)``.

    Examples
    --------
    Basic CAIPIRINHA sampling on a square 8x8 grid with R=(2,2):

    >>> import numpy as np
    >>> from sampling import make_caipirinha_sampling
    >>> mask = make_caipirinha_sampling((8, 8), accel=(2, 2), shift=0, elliptical=False)
    >>> mask.shape
    (8, 8)
    >>> mask.sum()
    16

    Use CAIPIRINHA with a shift (shift=1) and a small calibration region:

    >>> mask_shifted = make_caipirinha_sampling(
            (8, 8), accel=(2, 2), shift=1, calib=2, elliptical=False
        )
    >>> mask_shifted.shape
    (8, 8)
    >>> # calibration region is inserted at the center
    >>> mask_shifted[3:5, 3:5]
    array([[1, 1],
           [1, 1]])

    """
    if np.isscalar(shape):
        shape = [shape, shape]
    if np.isscalar(accel):
        accel = [accel, 1]

    # Cast tuple to lists
    shape = list(shape)
    accel = list(accel)

    # Validate input
    require(accel[0] >= 1, f'Ky acceleration must be >= 1, got {accel[0]}')
    require(accel[1] >= 1, f'Kz acceleration must be >= 1, got {accel[1]}')
    require(shift >= 0, f'CAPIRINHA shift must be positive, got {shift}')
    require(
        shift <= accel[1] - 1, f'CAPIRINHA shift must be lower than Rz, got {shift}'
    )

    # Define elliptical grid
    nz, ny = shape
    z, y = np.mgrid[:nz, :ny]
    y, z = abs(y - shape[-1] // 2), abs(z - shape[-2] // 2)
    r = np.sqrt((y / shape[-1]) ** 2 + (z / shape[-2]) ** 2) < 0.5

    if np.all(np.asarray(accel) == 1):
        return np.ones(shape, dtype=int)

    # Build mask
    rows, cols = np.mgrid[:nz, :ny]
    mask = (rows % accel[0] == 0) & (cols % accel[1] == 0)

    # CAPIRINHA shift
    if shift > 0:
        padsize0 = int(np.ceil(mask.shape[0] / accel[0]) * accel[0] - mask.shape[0])
        mask = np.pad(mask, ((0, padsize0), (0, 0)))
        nzp0, _ = mask.shape
        mask = mask.reshape(nzp0 // accel[0], accel[0], ny)
        mask = mask.reshape(nzp0 // accel[0], accel[0] * ny)
        padsize1 = int(np.ceil(mask.shape[0] / accel[1]) * accel[1] - mask.shape[0])
        mask = np.pad(mask, ((0, padsize1), (0, 0)))
        nzp1, _ = mask.shape
        mask = mask.reshape(nzp1 // accel[1], accel[1], accel[0] * ny)
        for n in range(1, mask.shape[1]):
            actshift = n * shift
            mask[:, n, :] = np.roll(mask[:, n, :], actshift)
        mask = mask.reshape(nzp1, accel[0] * ny)
        mask = mask[:nzp0, :]
        mask = mask.reshape(nzp0 // accel[0], accel[0], ny)
        mask = mask.reshape(nzp0, ny)
        mask = mask[:nz, :]

    # Re-insert calibration region
    if calib is not None:
        if np.isscalar(calib):
            calib = [calib, calib]
        calib = list(calib)
        calib.reverse()
        mask[
            shape[0] // 2 - calib[0] // 2 : shape[0] // 2 + calib[0] // 2,
            shape[1] // 2 - calib[1] // 2 : shape[1] // 2 + calib[1] // 2,
        ] = 1

    # Crop corners
    if elliptical:
        mask *= r

    return mask  # (ny, nz)


def make_partial_fourier_sampling(shape: int, undersampling: float) -> NDArray[int]:
    """
    Generate sampling pattern for Partial Fourier accelerated acquisition.

    Parameters
    ----------
    shape : int
        Image shape along partial fourier axis.
    undersampling : float
        Target undersampling factor.
        Must be > 0.5 (suggested > 0.7) and <= 1 (=1: no PF).

    Returns
    -------
    NDArray[int]
        Regular-grid sampling mask of shape ``(shape,)``.

    Examples
    --------
    Basic partial-fourier sampling: for shape=16 and undersampling=0.75,
    we keep the first floor(16*0.75)=12 lines and zero the remainder:

    >>> from sampling import make_partial_fourier_sampling
    >>> mask = make_partial_fourier_sampling(16, 0.75)
    >>> mask.sum()
    12
    >>> mask[-1]
    0

    """
    require(
        0.5 <= undersampling <= 1,
        f'Undersampling must between 0.5 and 1 (inclusive), got {undersampling}',
    )
    if undersampling < 0.75:
        logging.warning(
            f'Undersampling factor = {undersampling} < 0.75 - phase errors will'
            ' likely occur.'
        )

    # Generate mask
    mask = np.ones(shape, dtype=int)

    # Cut mask
    edge = np.floor(np.asarray(shape) * np.asarray(undersampling))
    edge = int(edge)
    mask[edge:] = 0

    return mask


def make_poisson_disk_sampling(
    shape: int | tuple[int, int],
    accel: float | tuple[float] = 1.0,
    calib: int | tuple[int, int] | None = None,
    elliptical: bool = True,
    nt: int = 1,
    vd_exp: float = 1.0,
    shape_opt: ShapeLike = 'L2_BALL',
    mindist_scaling: float = 1.0,
    C: float = 1.0,
    fov_ratio: float = 1.0,
    Rmax: float | None = None,
    vardens: bool = True,
    verbose: bool = False,
) -> NDArray[int]:
    """
    Generate a (variable-density) Poisson-disc sampling mask using CPD.

    Parameters
    ----------
    shape : int | tuple[int, int]
        Image shape along phase-encode dims ``(ny, nz)``.
    accel : float | tuple[float, float], optional
        Target acceleration ``(Ry, Rz) >= 1``. Default ``1``.
    calib : int | tuple[int, int], optional
        ACS box size ``(cy, cz)`` centered in k-space. If ``None``, no ACS.
    elliptical : bool, optional
        Toggle whether to crop corners of k-space (elliptical sampling).
        The default is ``True``.
    nt : int, optional
        Number of complementary temporal phases. The default is ``1``.
    vd_exp : float, optional
        Variable-density falloff exponent. Default ``1.0``.
    shape_opt : {'CROSS','L1_BALL','L2_BALL','CONES','PLANE_AND_CONES'}, optional
        CPD min-distance shape. Default ``'L2_BALL'``.
    mindist_scaling : float, optional
        CPD min-distance scaling. Default ``1.0``.
    C : float, optional
        CPD dt_min / dky_min balance. For nt=1 it has little effect. Default ``1.0``.
    fov_ratio : float, optional
        Ratio between FOVy and FOVz. The default is ``1.0``.
    vardens : bool, optional
        Toggle between uniform density (``False``) and variable density (``True``).
        The default is ``True``.
    verbose : bool, optional
        Verbosity flag. The default is ``False``.

    Returns
    -------
    mask : np.ndarray
        Binary mask of shape ``(ny, nz, nt)`` (squeezed from CPD output).

    Examples
    --------
    Typical variable-density CPD sampling on a 64x64 k-space with a central ACS:

    >>> from sampling import make_poisson_disk_sampling
    >>> # variable-density CPD with center calibration region
    >>> mask = make_poisson_disk_sampling(
            (64, 64), accel=(2, 2), calib=20, nt=1, vardens=True
        )
    >>> mask.shape
    (64, 64, 1)
    >>> set(np.unique(mask).tolist()) <= {0, 1}
    True
    >>> # center calibration block is all ones
    >>> cy = 20
    >>> cz = 20
    >>> center_block = mask[32 - cy//2 : 32 + cy//2, 32 - cz//2 : 32 + cz//2, 0]
    >>> center_block.sum() == cy * cz
    True

    Uniform-density (UD-CPD-like) example and temporal phases:

    >>> # uniform density generation (vardens=False) and multiple temporal phases
    >>> m_t = make_poisson_disk_sampling((32, 32), accel=(4, 1), nt=3, vardens=False)
    >>> m_t.shape
    (32, 32, 3)
    >>> # logical OR across temporal phases yields a 2D sampling coverage map
    >>> combined = np.sum(m_t, axis=2) > 0
    >>> combined.dtype == bool
    True

    Notes
    -----
    Based on https://github.com/evanlev/cpd/tree/master

    """
    if np.isscalar(shape):
        shape = [shape, shape]

    # Build feasible region
    feasible_points = make_caipirinha_sampling(shape, accel)

    # Define elliptical grid
    if elliptical:
        nz, ny = shape
        z, y = np.mgrid[:nz, :ny]
        y, z = abs(y - shape[-1] // 2), abs(z - shape[-2] // 2)
        r = np.sqrt((y / shape[-1]) ** 2 + (z / shape[-2]) ** 2) < 0.5
        feasible_points *= r

    # ACS region: exclude from CPD, add back afterward
    if calib is not None:
        # broadcast
        if np.isscalar(calib):
            calib = [calib, calib]

        # cast tuple to list
        calib = list(calib)

        # reverse (cz, cy)
        calib.reverse()

        feasible_points[
            shape[0] // 2 - calib[0] // 2 : shape[0] // 2 + calib[0] // 2,
            shape[1] // 2 - calib[1] // 2 : shape[1] // 2 + calib[1] // 2,
        ] = 0

    if vardens:
        if Rmax is None:
            Rmax = np.prod(accel).item()
        mask = _gen_vdcpd(
            nt,
            Rmax,
            vd_exp,
            fov_ratio,
            feasible_points,
            shape_opt,
            verbose,
            C,
            mindist_scaling,
        )
    else:
        mask = _gen_udcpd(
            nt, fov_ratio, feasible_points, shape_opt, verbose, C, mindist_scaling
        )

    # Re-insert calibration region
    if calib is not None:
        mask[
            shape[0] // 2 - calib[0] // 2 : shape[0] // 2 + calib[0] // 2,
            shape[1] // 2 - calib[1] // 2 : shape[1] // 2 + calib[1] // 2,
            ...,
        ] = 1

    return mask


# %% Internal helpers
def _normalize_shape(shape_opt: str) -> int:
    """Transform SHAPE into shape code."""
    if isinstance(shape_opt, str):
        key = shape_opt.strip().upper()
        if key in _SHAPE_MAP:
            return _SHAPE_MAP[key]
    raise ValueError(
        f"Unknown shape '{shape_opt}'. Expected one of: {', '.join(_SHAPE_MAP.keys())}"
    )


def _ensure_feasible_array(feasible_points: NDArray[int]) -> NDArray[float]:
    """Transform input array in Fortran-ordered floating array."""
    arr = np.asarray(feasible_points)
    if arr.ndim != 2:
        raise ValueError('feasiblePoints must be a 2D array (ny x nz)')
    return np.asfortranarray(arr.astype(np.float64, copy=False))


def _gen_udcpd(
    num_masks: int,
    alph: float,
    feasible_points: NDArray[int],
    shape_opt: str = "cones",  # noqa
    verbose: int = 1,
    C: float = 1.0,
    mindist_scaling: float = 1.0,
) -> NDArray[int]:
    """Run compiled Uniform Density Poisson mask generation routine."""
    shape_code = _normalize_shape(shape_opt)
    feasible_f = _ensure_feasible_array(feasible_points)
    return _gen_udcpd_bind(
        int(num_masks),
        float(alph),
        feasible_f,
        int(shape_code),
        int(verbose),
        float(C),
        float(mindist_scaling),
    )


def _gen_vdcpd(
    num_masks: int,
    Rmax: float,
    vd_exp: float,
    alph: float,
    feasible_points: NDArray[int],
    shape_opt: str = "cones",  # noqa
    verbose: int = 1,
    C: float = 1.0,
    mindist_scaling: float = 1.0,
) -> NDArray[int]:
    """Run compiled Variable Density Poisson mask generation routine."""
    shape_code = _normalize_shape(shape_opt)
    feasible_f = _ensure_feasible_array(feasible_points)
    return _gen_vdcpd_bind(
        int(num_masks),
        float(alph),
        feasible_f,
        int(shape_code),
        int(verbose),
        float(C),
        float(mindist_scaling),
        float(Rmax),
        float(vd_exp),
    )
