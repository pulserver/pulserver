"""3D Non Cartesian sampling encoding plan generation."""

__all__ = ["noncartesian3D"]


import numpy as np


from mrinufft.trajectories.maths import Rz, Rx


from . import _ordering
from ._iterators import NonCartesian3DIterator
from ._sampling import generate_tilt_angles


def noncartesian3D(
    n_views_plane: int,
    n_views_angular: int,
    Rplane: float = 1.0,
    Rangular: float = 1.0,
    angular_order: str = "mri-golden",
    dummy_shots: int = 0,
):
    r"""
    Generate encoding table for 3D on Cartesian imaging.

    This supports regular undersampling for Parallel Imaging acceleration.

    Parameters
    ----------
    n_views_plane : int
        Number of readouts to sample a plane.
    n_views_angular : int
        Number of encoding planes.
    Rplane : float, optional
        In-plane undersampling factor. The default is ``1.0`` (no acceleration).
    Rangular : float, optional
        Angular undersampling factor. The default is ``1.0`` (no acceleration).
    angular_order : str, optional
        Tilt angle in ``[rad]`` or name of the tilt. The default is ``"mri-golden"``.
    dummy_shots : int, optional
        Number of dummy shots at the beginning of the scan loop.
        The default is ``0``.

    Notes
    -----
    The following values are accepted for the tilt name, with :math:`N` the number of
    partitions:

    - ``"none"``: no tilt
    - ``"uniform"``: uniform tilt: 2:math:`\pi / N`
    - ``"intergaps"``: :math:`\pi/2N`
    - ``"inverted"``: inverted tilt :math:`\pi/N + \pi`
    - ``"golden"``: tilt of the golden angle :math:`\pi(3-\sqrt{5})`
    - ``"mri-golden"``: tilt of the golden angle used in MRI :math:`\pi(\sqrt{5}-1)/2`

    Returns
    -------
    NonCartesian3DIterator : object
        Iterator to keep trace of the shot index. Used to retrieve
        gradient amplitude and data labeling at a specific point during
        the scan loop.
    rotmat : tuple
        Stack of rotation matrices of shape ``(nviews // R, 3, 3)``,
        with ``nviews = n_views_plane * n_views_angular`` and ``R = Rplane * Rtheta``.

    """
    # Compute view angle for each readout within the plane
    plane_tilt = generate_tilt_angles(int(n_views_plane // Rplane), "uniform")
    angular_tilt = generate_tilt_angles(int(n_views_angular // Rangular), angular_order)

    # Combine
    _tilt = np.meshgrid(plane_tilt, angular_tilt, indexing="xy")
    plane_tilt, angular_tilt = _tilt

    # Generate rotation matrices
    _plane_rotmat = [Rz(theta) for theta in plane_tilt.ravel()]
    _plane_rotmat = np.stack(_plane_rotmat, axis=0)

    _angular_rotmat = [Rx(theta) for theta in angular_tilt.ravel()]
    _angular_rotmat = np.stack(_plane_rotmat, axis=0)

    rotmat = _angular_rotmat @ _plane_rotmat
    view_labels = np.arange(rotmat.shape[0])

    # Generate encoding iterator
    return (
        NonCartesian3DIterator(
            (rotmat, view_labels.astype(int)),
            dummy_shots,
        ),
        rotmat,
    )
