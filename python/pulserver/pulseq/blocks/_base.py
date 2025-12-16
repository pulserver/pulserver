"""
"""

__all__ = []

from types import SimpleNamespace

import numpy as np
from numpy.typing import NDArray


def calc_kspace_line_jump(
    fov: float,
    indexes: NDArray[int],
) -> tuple[float, float, float, NDArray[float]]:
    """
    Compute readout-intrinsic k-space parameters for EPI-like phase encoding
    along a single phase-encoding axis.

    This utility analyzes the k-space sampling order within a single readout
    (shot) and returns quantities that depend *only* on the readout trajectory,
    independent of segment (blade) position. All returned k-space offsets are
    expressed relative to the center of the sampled k-space band.

    The results are intended to be combined downstream with a segment-dependent
    k-space offset to synthesize the actual prewinder and rewinder areas using
    a single maximum-area trapezoid scaled in amplitude.

    Parameters
    ----------
    fov : float
        Target field-of-view along the phase-encoding axis, in meters.
        The corresponding k-space sampling interval is ``1 / fov``.

    indexes : NDArray[int]
        Sequence of k-space sample indices acquired during the readout
        along the given phase-encoding axis. The indices refer to positions
        within a k-space band and may be arbitrarily ordered (e.g. non-
        monotonic, permuted, or locally incoherent).

    Returns
    -------
    max_blip_area : float
        Maximum k-space displacement between any two consecutive readout
        samples, expressed in k-space units (``[1/m]``). This value defines
        the area of the base phase-encoding blip trapezoid.

    kp_rel_start : float
        Relative k-space position of the first readout sample with respect
        to the center of the sampled k-space band, in ``[1/m]`` units.
        This quantity must be added to the segment-dependent band offset
        to obtain the actual prewinder area.

    kp_rel_end : float
        Relative k-space position of the last readout sample with respect
        to the center of the sampled k-space band, in ``[1/m]`` units.
        This quantity must be added to the segment-dependent band offset
        to obtain the actual rewinder area.

    scaling : NDArray[float]
        Signed scaling factors for each intra-readout phase-encoding blip.
        Each entry represents the fraction of ``max_blip_area`` required
        to move from one readout sample to the next.

    Examples
    --------
    >>> fov = 0.22  # meters
    >>> indexes = np.array([4, 3, 2, 1, 0, 1, 2])
    >>> max_blip_k, ky0, kyN, scale = calc_kspace_line_jump(fov, indexes)
    >>> max_blip_k
    4.545454545454546
    >>> ky0, kyN
    (4.545454545454546, -4.545454545454546)

    Notes
    -----
    - The center of the k-space band is defined as the midpoint between the
      minimum and maximum indices, and may be fractional.
    - No assumptions are made about monotonicity or symmetry of the sampling
      order.
    - The returned quantities are readout-intrinsic and do not encode any
      information about segment or blade placement in k-space.

    """
    delta_k = 1.0 / fov

    # Readout-intrinsic band center (index space)
    center = 0.5 * (indexes.min() + indexes.max())

    # Relative ky positions of first and last readout
    kp_rel_start = (indexes[0] - center) * delta_k
    kp_rel_end = (indexes[-1] - center) * delta_k

    # Intra-readout jumps
    jumps = np.diff(indexes)
    max_jump = np.abs(jumps).max() if jumps.size else 0

    max_blip_area = max_jump * delta_k

    scaling = jumps / max_jump if max_jump > 0 else np.zeros_like(jumps)

    return max_blip_area, kp_rel_start, kp_rel_end, scaling


def calc_kspace_band_jump(
    fov: float,
    npix: int,
    band_width: int = 1,
    kp_rel_start: float = 0.0,
    kp_rel_end: float = 0.0,
) -> tuple[float, SimpleNamespace]:
    """
    Compute k-space band parameters for prewinder/rewinder scaling along a
    phase-encoding axis.

    This function computes the maximum absolute k-space offset across all
    bands and the scaling factors for prewinder and rewinder gradients
    for each band center line. DC (k-space center) is explicitly included
    as the central index, even if the band is asymmetric.

    Parameters
    ----------
    fov : float
        Field-of-view along the phase-encoding axis ``[m]``.
    npix : int
        Number of phase-encoding pixels along the axis.
    band_width : int, optional
        Width of each band (number of lines), by default ``1``.
    kp_rel_start : float, optional
        Readout-intrinsic offset of the first sample relative to band center
        ``[1/m]``, by default ``0.0``.
    kp_rel_end : float, optional
        Readout-intrinsic offset of the last sample relative to band center
        ``[1/m]``, by default ``0.0``.

    Returns
    -------
    kp_area_max : float
        Maximum absolute k-space offset among all bands ``[1/m]``. This defines
        the base trapezoid area for prewinder/rewinder.
    scaling : SimpleNamespace
        Named tuple with scaling factors for each band center line:
        - pre : ``NDArray[float]``, prewinder scaling factors ``[-1, 1]``
        - rew : ``NDArray[float]``, rewinder scaling factors ``[-1, 1]``

    Notes
    -----
    - The function assumes phase-encoding samples are numbered from 0 to npix-1.
    - All returned offsets are relative to DC (k-space center) explicitly
      included in the band.
    - Scaling factors are relative to the maximum absolute k-space area,
      so they can be directly applied to a single maximum-area trapezoid.

    Examples
    --------
    >>> fov = 0.22  # meters
    >>> npix = 7
    >>> band_width = 2
    >>> kp_rel_start = 0.0
    >>> kp_rel_end = 0.0
    >>> kp_area_max, scaling = calc_kspace_band_jump(fov, npix, band_width, kp_rel_start, kp_rel_end)
    >>> kp_area_max
    0.3181818181818182
    >>> scaling.pre
    array([-1.        , -0.71428571, -0.42857143,  0.        ])
    >>> scaling.rew
    array([ 1.        ,  0.71428571,  0.42857143, -0.        ])

    """
    delta_k = 1.0 / fov
    kp_area = npix * delta_k

    # Handle single-pixel case safely
    if npix == 1:
        pre_scaling = np.array([1.0])
        rew_scaling = np.array([-1.0])
        kp_area_max = (
            abs(kp_rel_start)
            if abs(kp_rel_start) > abs(kp_rel_end)
            else abs(kp_rel_end)
        )
        return kp_area_max, SimpleNamespace(pre=pre_scaling, rew=rew_scaling)

    # Relative positions of pixels w.r.t. DC
    scalings = (np.arange(npix) - npix // 2) / npix

    # Center indices of each band
    band_centers = np.arange(0, npix, band_width) + band_width // 2

    # Ensure indices do not exceed npix-1
    band_centers = np.clip(band_centers, 0, npix - 1)

    # Area of first and last band center relative to readout
    kp_area_start = scalings[band_centers[0]] * kp_area + kp_rel_start
    kp_area_end = scalings[band_centers[-1]] * kp_area + kp_rel_end

    # Maximum absolute area for trapezoid sizing
    kp_area_max = max(abs(kp_area_start), abs(kp_area_end), 1e-12)

    # Scaling factors for each band
    pre_scaling = (scalings[band_centers] * kp_area + kp_rel_start) / kp_area_max
    rew_scaling = -(scalings[band_centers] * kp_area + kp_rel_end) / kp_area_max

    return kp_area_max, SimpleNamespace(pre=pre_scaling, rew=rew_scaling)


def calc_readout_area(
    fov: float,
    npix: int,
):
    delta_k = 1.0 / fov
    npix * delta_k
