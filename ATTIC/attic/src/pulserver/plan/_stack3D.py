"""3D Stack-of- Non Cartesian sampling encoding plan generation."""

__all__ = ["stack3D"]


import numpy as np


from mrinufft.trajectories.maths import Rz as _Rz


from . import _ordering
from ._iterators import Stack3DIterator
from ._sampling import generate_tilt_angles, grid_sampling2D, partial_fourier


def stack3D(
    n_views: int,
    nz: int,
    Rtheta: float = 1.0,
    Rz: int = 1,
    Rpf: float = 1.0,
    calib: int | None = None,
    view_order: str = "mri-golden",
    slice_order: str = "interleaved",
    view_loop_position: str = "inner",
    dummy_shots: int = 0,
):
    r"""
    Generate encoding table for 3D Stack-of Non Cartesian imaging.

    This supports regular undersampling for Parallel Imaging acceleration.

    User can switch between sequential view ordering (i.e., acquire all
    views before moving to next slice) and interleaved (i.e., acquire
    all the slices from the same view before moving to the next).

    Also supports different slice orderings (sequential, interleaved).

    Parameters
    ----------
    n_views : int
        Number of readouts.
    nz : int
        Number of slice encoding lines.
    Rtheta : float, optional
        Angular undersampling factor. The default is ``1.0`` (no acceleration).
    Rz : float, optional
        Parallel Imaging acceleration along z. The default is ``1.0`` (no acceleration).
    Rpf : float, optional
        Partial Fourier acceleration. The default is ``1.0`` (no acceleration).
        Must be > 0.5 (suggested > 0.7) and <= 1 (=1: no PF).
    calib : int | None = None, optional
        Image shape along slice encoding dim ``cz``.
        The default is ``None`` (no calibration).
    view_order : str, optional
        Tilt angle in ``[rad]`` or name of the tilt. The default is ``"mri-golden"``.
    slice_order : str, optional
        Slice ordering. Can be either ``"sequential"``, ``"interleaved"`` or ``"center-out"``.
        If ``"sequential"``, acquire all slices in sequential order. If
        ``"interleaved"``, acquire all odd slices sequentially before moving to even.
        If ``"center-out"``, acquire all symmetrically with respect to center..
        The default is ``"interleaved"``.
    view_loop_position : str, optional
        Phase encoding loop position. Can be either ``"inner"``
        or ``"outer"``. If ``"inner"``, it acquires all the views for a slice
        before moving to the next. If ``"outer"``, acquire all slices before
        moving to next view. The default is ``"inner"``.
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
    Stack3DIterator : object
        Iterator to keep trace of the shot index. Used to retrieve
        gradient amplitude and data labeling at a specific point during
        the scan loop.
    sampling_pattern : tuple
        Tuple of ``(rotmat, sampling_pattern)`` ndarrays.
        Here, ``rotmat`` is the stack of rotation matrices of shape ``(nviews // Rtheta, 3, 3)``,
        while ``sampling pattern`` is the slice encoding sampling pattern
        of shape ``(nz,)``.

    """
    # Compute RF frequency offsets for each slice (in [Hz/m])
    slice_encoding_scaling = ((np.arange(nz)) - (nz / 2)) / nz
    slice_encoding_labels = np.arange(nz)

    # Reorder slices
    if slice_order == "sequential":
        pass
    elif slice_order == "interleaved":
        slice_encoding_scaling = _ordering.interleaved(slice_encoding_scaling)
        slice_encoding_labels = _ordering.interleaved(slice_encoding_labels)
    elif slice_order == "center-out":
        slice_encoding_scaling = _ordering.center_out(slice_encoding_scaling)
        slice_encoding_labels = _ordering.center_out(slice_encoding_labels)
    else:
        raise ValueError(
            f"Unrecognized slice order: {slice_order} - must be either 'sequential', 'interleaved' or 'center-out'."
        )

    # Compute sampling mask for phase encoding
    sampling_pattern = grid_sampling2D(nz, Rz, calib) * partial_fourier(nz, Rpf)

    # Compute view angle for each readout
    view_tilt = generate_tilt_angles(int(n_views // Rtheta), view_order)
    view_labels = np.arange(int(n_views // Rtheta))

    # Generate rotation matrices
    rotmat = [_Rz(theta) for theta in view_tilt]
    rotmat = np.stack(rotmat, axis=0)

    # Generate encoding iterator
    return (
        Stack3DIterator(
            (rotmat, view_labels.astype(int)),
            (slice_encoding_scaling, slice_encoding_labels.astype(int)),
            view_loop_position,
            dummy_shots,
        ),
        (rotmat, sampling_pattern),
    )
