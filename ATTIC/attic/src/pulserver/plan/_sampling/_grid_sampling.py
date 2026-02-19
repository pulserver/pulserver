"""
Regular grid undersampling pattern for Cartesian imaging.

This can be used e.g., for SENSE/GRAPPA and CAIPIRINHA acceleration.

"""

__all__ = ["grid_sampling2D", "grid_sampling3D"]


import numpy as np


def grid_sampling2D(
    shape: int,
    accel: int,
    calib: int = None,
) -> np.ndarray:
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

    """
    if accel < 1:
        raise ValueError(f"Ky acceleration must be greater than 1, got {accel}")

    # build mask
    mask = np.zeros(shape, dtype=bool)
    mask[::accel] = True

    # calib
    if calib is not None:
        mask[shape // 2 - calib // 2 : shape // 2 + calib // 2] = True

    return mask


def grid_sampling3D(
    shape: int | tuple[int],
    accel: int | tuple[int] = 1.0,
    calib: int | tuple[int] | None = None,
    shift: int = 0,
    crop_corners: bool = True,
) -> np.ndarray:
    """
    Generate regular sampling pattern for SENSE/GRAPPA or CAIPIRINHA accelerated acquisition.

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
    crop_corner : bool, optional
        Toggle whether to crop corners of k-space (elliptical sampling).
        The default is ``True``.

    Returns
    -------
    mask : np.ndarray
        Regular-grid sampling mask of shape ``(nz, ny)``.

    """
    if np.isscalar(shape):
        # assume square matrix (ky, kz)
        shape = [shape, shape]
    if np.isscalar(accel):
        # assume acceleration along a single axis
        accel = [accel, 1]

    # cast tuple to lists
    shape = list(shape)
    accel = list(accel)

    # define elliptical grid
    nz, ny = shape
    z, y = np.mgrid[:nz, :ny]
    y, z = abs(y - shape[-1] // 2), abs(z - shape[-2] // 2)
    r = np.sqrt((y / shape[-1]) ** 2 + (z / shape[-2]) ** 2) < 0.5

    # check
    # if accel[0] < 1:
    #     raise ValueError(f"Ky acceleration must be >= 1, got {accel[0]}")
    # if accel[1] < 1:
    #     raise ValueError(f"Kz acceleration must be >= 1, got {accel[1]}")
    if np.all(np.asarray(accel) == 1):
        return np.ones(shape, dtype=bool)

    if shift < 0:
        raise ValueError(f"CAPIRINHA shift must be positive, got {shift}")
    if shift > accel[1] - 1:
        raise ValueError(f"CAPIRINHA shift must be lower than Rz, got {shift}")

    # build mask
    rows, cols = np.mgrid[:nz, :ny]
    mask = (rows % accel[0] == 0) & (cols % accel[1] == 0)

    # CAPIRINHA shift
    if shift > 0:
        # first pad
        padsize0 = int(np.ceil(mask.shape[0] / accel[0]) * accel[0] - mask.shape[0])
        mask = np.pad(mask, ((0, padsize0), (0, 0)))
        nzp0, _ = mask.shape

        # first reshape
        mask = mask.reshape(nzp0 // accel[0], accel[0], ny)
        mask = mask.reshape(nzp0 // accel[0], accel[0] * ny)

        # second pad
        padsize1 = int(np.ceil(mask.shape[0] / accel[1]) * accel[1] - mask.shape[0])
        mask = np.pad(mask, ((0, padsize1), (0, 0)))
        nzp1, _ = mask.shape

        # second reshape
        mask = mask.reshape(nzp1 // accel[1], accel[1], accel[0] * ny)

        # perform shift
        for n in range(1, mask.shape[1]):
            actshift = n * shift
            mask[:, n, :] = np.roll(mask[:, n, :], actshift)

        # first reshape back
        mask = mask.reshape(nzp1, accel[0] * ny)
        mask = mask[:nzp0, :]

        # second reshape back
        mask = mask.reshape(nzp0 // accel[0], accel[0], ny)
        mask = mask.reshape(nzp0, ny)
        mask = mask[:nz, :]

    # re-insert calibration region
    if calib is not None:
        # broadcast
        if np.isscalar(calib):
            calib = [calib, calib]

        # cast tuple to list
        calib = list(calib)

        # reverse (cz, cy)
        calib.reverse()

        mask[
            shape[0] // 2 - calib[0] // 2 : shape[0] // 2 + calib[0] // 2,
            shape[1] // 2 - calib[1] // 2 : shape[1] // 2 + calib[1] // 2,
        ] = 1

    # crop corners
    if crop_corners:
        mask *= r

    return mask  # (ny, nz)
