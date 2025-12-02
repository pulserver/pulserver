"""3D Cartesian sampling encoding plan generation."""

__all__ = ["cartesian3D"]


import numpy as np


from . import _ordering
from ._iterators import sampling2labels, Cartesian3DIterator
from ._sampling import grid_sampling3D, partial_fourier, poisson_sampling3D


def cartesian3D(
    ny: int,
    nz: int,
    Ry: int = 1,
    Rz: int = 1,
    shift: int = 0,
    Rp: float = 1.0,
    Rpf: float = 1.0,
    calib: int | tuple[int] | None = None,
    view_order: str = "sequential",
    dummy_shots: int = 0,
    crop_corner: bool = True,
    seed: int = 0,
    max_attempts: int = 30,
    tol: float = 0.1,
):
    """
    Generate encoding table for 3D Cartesian imaging.

    This supports regular undersampling for Parallel Imaging (or CAIPIRINHA)
    and Partial Fourier acceleration, as well as Poisson Disk undersampling (Compressed Sensing).

    User can switch between sequential and center-out view ordering.

    Parameters
    ----------
    ny : int
        Number of phase encoding lines.
    nz : int
        Number of slice encoding lines.
    Ry : int, optional
        Parallel Imaging acceleration along y. The default is ``1`` (no acceleration).
    Rz : int, optional
        Parallel Imaging acceleration along z. The default is ``1`` (no acceleration).
    shift : int, optional
        Caipirinha shift. The default is ``0`` (standard PI sampling).
    Rp : float, optional
        Poisson acceleration for Compressed Sensing. The default is ``1.0`` (no acceleration).
    Rpf : float, optional
        Partial Fourier acceleration.
        Must be > 0.5 (suggested > 0.7) and <= 1 (=1: no PF).  The default is ``1.0`` (no acceleration).
    calib : int | tuple[int], optional
        Image shape along phase encoding dims ``(cy, cz)``.
        If scalar, assume equal size for ``y`` and ``z`` axes.
        The default is ``None`` (no calibration).
    view_order : str, optional
        View ordering. Can be either ``"sequential"`` or ``"center-out"``.
        If ``"sequential"``, acquire all views in sequential order. If
        ``"center-out"``, acquisition ordering is symmetrical with respect to center.
        The default is ``"sequential"``.
    dummy_shots : int, optional
        Number of dummy shots at the beginning of the scan loop.
        The default is ``0``.
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
    Cartesian3DIterator : object
        Iterator to keep trace of the shot index. Used to retrieve
        gradient amplitude and data labeling at a specific point during
        the scan loop.
    tilt_angles : np.ndarray
        Rotation angles of shape ``(nviews // Rtheta,)``.

    """
    # Compute phase encoding gradient scaling for each phase encoding step (from -0.5 to 0.5)
    phase_encoding_scaling = ((np.arange(ny)) - (ny // 2)) / ny
    slice_encoding_scaling = ((np.arange(nz)) - (nz // 2)) / nz

    # Reorder views
    if view_order == "sequential":
        phase_encoding_ordering = np.arange(ny)
        slice_encoding_ordering = np.arange(nz)
    elif view_order == "center-out":
        phase_encoding_ordering = _ordering.center_out(np.arange(ny))
        slice_encoding_ordering = _ordering.center_out(np.arange(nz))
    else:
        raise ValueError(
            f"Unrecognized view order: {view_order} - must be either 'sequential' or 'center-out'."
        )

    # Combine scaling
    encoding_scaling = np.meshgrid(
        phase_encoding_scaling,
        slice_encoding_scaling,
        indexing="xy",
    )
    encoding_scaling = np.stack(encoding_scaling, axis=0)

    # Compute sampling mask for phase encoding
    sampling_pattern = (
        grid_sampling3D((ny, nz), (Ry, Rz), calib, shift, crop_corner)
        * poisson_sampling3D((ny, nz), Rp, calib, crop_corner, seed, max_attempts, tol)
        * partial_fourier(nz, Rpf)
    )

    # Initialize labels
    encoding_labels = np.indices(sampling_pattern.T.shape)

    # Apply undersampling
    mask = np.where(sampling_pattern.T == 0, np.nan, sampling_pattern.T)
    encoding_scaling = np.stack([mask * enc for enc in encoding_scaling], axis=0)
    encoding_labels = np.stack([mask * enc for enc in encoding_labels], axis=0)

    # Sort coordinates
    encoding_scaling[0] = encoding_scaling[0][:, phase_encoding_ordering]
    encoding_scaling[1] = encoding_scaling[1][slice_encoding_ordering, :]
    encoding_labels[0] = encoding_labels[0][:, phase_encoding_ordering]
    encoding_labels[1] = encoding_labels[1][slice_encoding_ordering, :]

    # Filter out unsampled locations
    encoding_scaling = np.stack(
        [enc[np.logical_not(np.isnan(enc))] for enc in encoding_scaling], axis=0
    )
    encoding_labels = np.stack(
        [enc[np.logical_not(np.isnan(enc))] for enc in encoding_labels], axis=0
    )

    # Unpack
    phase_encoding_scaling = encoding_scaling[0]
    slice_encoding_scaling = encoding_scaling[1]

    phase_encoding_labels = encoding_labels[1].astype(int)
    slice_encoding_labels = encoding_labels[0].astype(int)

    # Generate encoding iterator
    return (
        Cartesian3DIterator(
            (phase_encoding_scaling, phase_encoding_labels),
            (slice_encoding_scaling, slice_encoding_labels),
            dummy_shots,
        ),
        sampling_pattern,
    )
