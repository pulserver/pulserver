"""2D Non Cartesian sampling encoding plan generation."""

__all__ = ["noncartesian2D"]


from types import SimpleNamespace


import numpy as np


from mrinufft.trajectories.maths import Rz


from . import _ordering
from ._iterators import NonCartesian2DIterator
from ._sampling import generate_tilt_angles


def noncartesian2D(
    g_slice_select: SimpleNamespace,
    slice_thickness: float,
    n_views: int,
    n_slices: int,
    slice_gap: float = 0.0,
    Rtheta: float = 1.0,
    view_order: str = "mri-golden",
    slice_order: str = "interleaved",
    view_loop_position: str = "inner",
    dummy_shots: int = 0,
):
    r"""
    Generate encoding table for 2D Non Cartesian imaging.

    This supports regular undersampling for Parallel Imaging acceleration.

    User can switch between sequential view ordering (i.e., acquire all
    views before moving to next slice) and interleaved (i.e., acquire
    all the slices from the same view before moving to the next).

    Also supports different slice orderings (sequential, interleaved).

    Parameters
    ----------
    g_slice_select : SimpleNamespace
        PyPulseq slice selection event.
    slice_thickness : float
        Slice thickness in ``[mm]``.
    n_views : int
        Number of readouts.
    n_slices : int
        Number of slices.
    slice_gap : float
        Slice gap in ``[mm]``.
    Rtheta : float, optional
        Angular undersampling factor. The default is ``1.0`` (no acceleration).
    view_order : str, optional
        Tilt angle in ``[rad]`` or name of the tilt. The default is ``"mri-golden"``.
    slice_order : str, optional
        Slice ordering. Can be either ``"sequential"`` or ``"interleaved"``.
        If ``"sequential"``, acquire all slices in sequential order. If
        ``"interleaved"``, acquire all odd slices sequentially before moving to even.
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
    The effective slice separation is given by the sum of ``slice_thickness``
    and ``slice_gap``. For ``slice_gap == 0``, slices are contiguous.

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
    NonCartesian2DIterator : object
        Iterator to keep trace of the shot index. Used to retrieve
        gradient amplitude and data labeling at a specific point during
        the scan loop.
    rotmat : np.ndarray
        Stack of rotation matrices of shape ``(nviews // Rtheta, 3, 3)``.

    """
    # Compute RF frequency offsets for each slice (in [Hz/m])
    slice_coverage = (
        n_slices * (slice_thickness + slice_gap) * 1e-3
    )  # total z coverage in [m]
    if n_slices != 1:
        slice_freq_offset = (
            np.linspace(-slice_coverage / 2, slice_coverage / 2, n_slices)
            * g_slice_select.amplitude
        )
    else:
        slice_freq_offset = np.asarray([0.0])
    slice_labels = np.arange(n_slices)

    # Reorder slices
    if slice_order == "sequential":
        pass
    elif slice_order == "interleaved":
        slice_freq_offset = _ordering.interleaved(slice_freq_offset)
        slice_labels = _ordering.interleaved(slice_labels)
    else:
        raise ValueError(
            f"Unrecognized slice order: {slice_order} - must be either 'sequential' or 'interleaved'."
        )

    # Compute view angle for each readout
    view_tilt = generate_tilt_angles(int(n_views // Rtheta), view_order)
    view_labels = np.arange(int(n_views // Rtheta))

    # Generate rotation matrices
    rotmat = [Rz(theta) for theta in view_tilt]
    rotmat = np.stack(rotmat, axis=0)

    # Generate encoding iterator
    return (
        NonCartesian2DIterator(
            (rotmat, view_labels.astype(int)),
            (slice_freq_offset, slice_labels.astype(int)),
            view_loop_position,
            dummy_shots,
        ),
        rotmat,
    )
