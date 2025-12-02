"""
Regular grid undersampling pattern for Cartesian imaging.

This can be used e.g., for Compressed Sensing acceleration.

"""

__all__ = ["poisson_sampling3D"]


import warnings


import numpy as np
import numba as nb


def poisson_sampling3D(
    shape: int | tuple[int],
    accel: float | tuple[float] = 1.0,
    calib: int | tuple[int] | None = None,
    crop_corner: bool = True,
    seed: int = 0,
    max_attempts: int = 30,
    tol: float = 0.1,
) -> np.ndarray:
    """
    Generate variable-density Poisson-disc sampling pattern.

    The function generates a variable density Poisson-disc sampling
    mask with density proportional to :math:`1 / (1 + s |r|)`,
    where :math:`r` represents the k-space radius, and :math:`s`
    represents the slope. A binary search is performed on the slope :math:`s`
    such that the resulting acceleration factor is close to the
    prescribed acceleration factor `accel`. The parameter `tol`
    determines how much they can deviate.

    Parameters
    ----------
    shape : int | tuple[int]
        Image shape along phase encoding dims ``(ny, nz)``.
        If scalar, assume equal size for ``y`` and ``z`` axes.
    accel : float | Tuple[float], optional
        Target acceleration factor along phase encoding dims ``(Ry, Rz)``.
        Must be ``>= 1``. If scalar, assume acceleration over ``y``
        only. The default is ``1`` (no acceleration).
    calib : int | tuple[int], optional
        Image shape along phase encoding dims ``(cy, cz)``.
        If scalar, assume equal size for ``y`` and ``z`` axes.
        The default is ``None`` (no calibration).
    crop_corner : bool, optional
        Toggle whether to crop corners of k-space (elliptical sampling).
        The default is ``True``.
    seed : int, optional
        Random seed. The default is ``0``.
    max_attempts : float, optional
        Maximum number of samples to reject in Poisson disc calculation.
        The default is ``30``.
    tol : float. optional
        Tolerance for how much the resulting acceleration can
        deviate form ``accel``. The default is ``0.1``.

    Returns
    -------
    mask : np.ndarray
        Poisson-disc sampling mask of shape ``(ny, nz)``.
    R : float
        Actual undersampling factor.

    References
    ----------
    Bridson, Robert. "Fast Poisson disk sampling in arbitrary dimensions."
    SIGGRAPH sketches. 2007.

    """
    if np.isscalar(shape):
        # assume square matrix (ky, kz)
        shape = [shape, shape, 1]
    shape = list(shape)
    if len(shape) == 2:
        shape = shape + [1]

    # cast tuple to lists
    shape = list(shape)

    if calib is not None:
        if np.isscalar(calib):
            calib = [calib, calib]

        # cast tuple to lists
        calib = list(calib)

        # find actual calibration size
        if shape[-1] > 1:
            calib = max(calib)
            calib = int(np.ceil(calib / shape[-1]) * shape[-1])
            calib = [calib, calib]

        # reverse (cz, cy)
        calib.reverse()

    # if accel < 1:
    #     raise ValueError(f"accel must be greater than 1, got {accel}")
    if accel == 1:
        return np.ones(shape, dtype=bool).squeeze()

    if seed is not None:
        rand_state = np.random.get_state()

    # define elliptical grid
    ny, nz, nt = shape
    z, y = np.mgrid[:nz, :ny]
    y, z = abs(y - shape[0] / 2), abs(z - shape[1] / 2)
    rdisk = 2 * np.sqrt((y / shape[0]) ** 2 + (z / shape[1]) ** 2)
    if nt == 1:
        r = rdisk[None, ...]
    else:
        t, z, y = np.mgrid[:nt, :nz, :ny]
        y, z, t = abs(y - shape[0] / 2), abs(z - shape[1] / 2), abs(t - shape[2] / 2)
        r = 2 * np.sqrt((y / shape[0]) ** 2 + (z / shape[1]) ** 2 + (t / shape[2]) ** 2)

    # calculate mask
    slope_max = max(ny, nz, nt)
    slope_min = 0
    while slope_min < slope_max:
        slope = (slope_max + slope_min) / 2
        radius_y = np.clip((1 + r * slope) * ny / max(ny, nz, nt), 1, None)
        radius_z = np.clip((1 + r * slope) * nz / max(ny, nz, nt), 1, None)
        radius_t = np.clip((1 + r * slope) * nt / max(ny, nz, nt), 1, None)
        mask = _poisson(
            shape[0],
            shape[1],
            shape[2],
            radius_y,
            radius_z,
            radius_t,
            max_attempts,
            seed,
        )

        # re-insert calibration region
        mask = _insert_calibration(mask, calib)

        if crop_corner:
            mask *= rdisk < 1

        actual_accel = np.prod(shape) / np.sum(mask)

        if abs(actual_accel - accel) < tol:
            break
        if actual_accel < accel:
            slope_min = slope
        else:
            slope_max = slope

    if abs(actual_accel - accel) >= tol:
        warnings.warn(
            f"Cannot generate mask to satisfy accel={accel}"
            f" - actual acceleration will be {actual_accel}"
        )

    # prepare for output
    mask = mask.reshape(shape[2], shape[1], shape[0]).squeeze()

    if seed is not None:
        np.random.set_state(rand_state)

    return mask.T  # , actual_accel


# %% local utils
@nb.njit(cache=True, fastmath=True)  # pragma: no cover
def _poisson(ny, nz, nt, radius_z, radius_y, radius_t, max_attempts, seed=None):
    mask = np.zeros((nt, nz, ny), dtype=np.int32)

    if seed is not None:
        np.random.seed(int(seed))

    # initialize active list
    pys = np.empty(ny * nz * nt, np.int32)
    pzs = np.empty(ny * nz * nt, np.int32)
    pts = np.empty(ny * nz * nt, np.int32)
    pys[0] = np.random.randint(0, ny)
    pzs[0] = np.random.randint(0, nz)
    pts[0] = np.random.randint(0, nt)
    num_actives = 1

    while num_actives > 0:
        i = np.random.randint(0, num_actives)
        py = pys[i]
        pz = pzs[i]
        pt = pts[i]
        ry = radius_y[pt, pz, py]
        rz = radius_z[pt, pz, py]
        rt = radius_t[pt, pz, py]

        # Attempt to generate point
        done = False
        k = 0
        while not done and k < max_attempts:
            # Generate point randomly from r and 2 * r
            v = (np.random.random() * 3 + 1) ** 0.5
            phi = 2 * np.pi * np.random.random()
            theta = np.arccos(np.random.random() * 2 - 1)

            qy = py + v * ry * np.cos(phi) * np.sin(theta)
            qz = pz + v * rz * np.sin(phi) * np.sin(theta)
            qt = pt + v * rt * np.cos(theta)

            # Reject if outside grid or close to other points
            if qy >= 0 and qy < ny and qz >= 0 and qz < nz and qt >= 0 and qt < nt:
                starty = max(int(qy - ry), 0)
                endy = min(int(qy + ry + 1), ny)
                startz = max(int(qz - rz), 0)
                endz = min(int(qz + rz + 1), nz)
                startt = max(int(qt - rt), 0)
                endt = min(int(qt + rt + 1), nt)

                done = True
                for y in range(starty, endy):
                    for z in range(startz, endz):
                        for t in range(startt, endt):
                            if mask[t, z, y] == 1 and (
                                ((qy - y) / (radius_y[t, z, y])) ** 2
                                + ((qz - z) / (radius_z[t, z, y])) ** 2
                                + ((qt - t) / (radius_t[t, z, y])) ** 2
                                < 1
                            ):
                                done = False
                                break

            k += 1

        # Add point if done else remove from active list
        if done:
            pys[num_actives] = qy
            pzs[num_actives] = qz
            pts[num_actives] = qt
            mask[int(qt), int(qz), int(qy)] = 1
            num_actives += 1
        else:
            pys[i] = pys[num_actives - 1]
            pzs[i] = pzs[num_actives - 1]
            pts[i] = pts[num_actives - 1]
            num_actives -= 1

    return mask


def _insert_calibration(mask, calib):
    shape = mask.shape
    if calib is not None:
        calib_mask = np.zeros(shape[1:], dtype=int)

        # find center and edges
        y0, z0 = shape[1] // 2, shape[2] // 2
        dy, dz = calib[0] // 2, calib[1] // 2
        calib_mask[y0 - dy : y0 + dy, z0 - dz : z0 + dz] = 1

        # find indices and fill mask
        idx = np.where(calib_mask)
        idx = [i.reshape(shape[0], int(i.shape[0] / shape[0])) for i in idx]
        idx = nb.typed.List(idx)
        _fill_mask(mask, idx)

    return mask


@nb.njit(cache=True)
def _fill_mask(mask, idx):
    nframes = mask.shape[0]
    npts = idx[0].shape[-1]
    for n in range(nframes):
        for i in range(npts):
            mask[n, idx[0][n, i], idx[1][n, i]] = 1
