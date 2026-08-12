"""The four undersampling mask modes, as one family."""

from __future__ import annotations

__all__ = ["make_uniform_mask"]

import numpy as np


def make_uniform_mask(shape, acceleration, calibration=0) -> np.ndarray:
    """Regular undersampling with a fully sampled centre.

    The fourth mode beside CAIPIRINHA, random and Poisson-disc, and the only one
    that works on a single phase-encode axis.

    Parameters
    ----------
    shape : int or tuple of int
        Phase-encode extents: one value for a 2D acquisition's single encoded
        axis, two for a 3D acquisition's ``(ny, nz)``.
    acceleration : int or tuple of int
        Undersampling factor, per axis or one for all.
    calibration : int or tuple of int, optional
        Fully sampled centred region, per axis or one for all.

    Returns
    -------
    numpy.ndarray
        Boolean mask of ``shape``.

    Raises
    ------
    ValueError
        If the extents, factors and calibration sizes do not agree, or any is
        out of range.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> mask = pp.make_uniform_mask(64, 2, calibration=8)
    >>> int(mask.sum())
    36

    Two accelerated axes multiply:

    >>> mask = pp.make_uniform_mask((64, 32), (2, 2))
    >>> mask.shape, int(mask.sum())
    ((64, 32), 512)

    See Also
    --------
    make_poisson_disc_mask : incoherent but locally uniform alternative.
    make_caipirinha_mask : shifted regular sampling over two encoded axes.
    make_random_mask : variable-density random sampling.
    """
    shape = _per_axis(shape, None, "shape")
    acceleration = _per_axis(acceleration, len(shape), "acceleration")
    calibration = _per_axis(calibration, len(shape), "calibration")
    if any(n < 1 for n in shape):
        raise ValueError("shape entries must be >= 1")
    if any(r < 1 for r in acceleration):
        raise ValueError("acceleration entries must be >= 1")
    if any(c < 0 for c in calibration):
        raise ValueError("calibration entries must be >= 0")

    mask = np.zeros(shape, dtype=bool)
    grids = np.ix_(*[np.arange(0, n, r) for n, r in zip(shape, acceleration, strict=True)])
    mask[grids] = True

    centred = []
    for n, c in zip(shape, calibration, strict=True):
        start = max(0, n // 2 - c // 2)
        centred.append(np.arange(start, min(n, start + c)))
    if all(len(axis) for axis in centred):
        mask[np.ix_(*centred)] = True
    return mask


def _per_axis(value, count: int | None, name: str) -> tuple[int, ...]:
    """``value`` as one integer per axis, broadcasting a scalar."""
    if np.isscalar(value):
        return (int(value),) if count is None else (int(value),) * count
    values = tuple(int(v) for v in value)
    if count is not None and len(values) != count:
        raise ValueError(f"{name} must be a scalar or one value per axis")
    return values
