"""Rotation angle generators."""

__all__ = ["generate_tilt_angles"]


import numpy as np


from mrinufft.trajectories.utils import initialize_tilt


def generate_tilt_angles(
    n_angles: int, tilt: float | str, n_partitions: int | None = None
) -> np.ndarray:
    r"""Initialize the tilt angle.

    Parameters
    ----------
    n_angles : int
        Length of the generated tilt angle list.
    tilt : str or float
        Tilt angle in ``[rad]`` or name of the tilt.
    n_partitions : int, optional
        Number of partitions of the unit circle.
        If not provided, assume it is equal to ``n_angles``.

    Returns
    -------
    np.ndarray
        Tilt angle list in ``[rad]``.

    Notes
    -----
    The following values are accepted for the tilt name, with :math:`N` the number of
    partitions:

    - ``"none"``: no tilt
    - ``"uniform"``: uniform tilt: 2:math:`\pi / N`
    - ``"intergaps"``: :math:`\pi/2N`
    - ``"inverted"``: inverted tilt :math:`\pi/N + \pi`
    - ``"golden"``: tilt of the golden angle :math:`\pi(3-\sqrt{5})`
    - ``"mri golden"``: tilt of the golden angle used in MRI :math:`\pi(\sqrt{5}-1)/2`

    """
    if n_partitions is None:
        n_partitions = n_angles
    dtheta = initialize_tilt(tilt, n_partitions)
    return dtheta * np.arange(n_angles) % (2 * np.pi)
