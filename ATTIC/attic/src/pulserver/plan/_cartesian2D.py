"""2D Cartesian sampling encoding plan generation."""

__all__ = ["cartesian2D"]


from types import SimpleNamespace


import numpy as np


from . import _ordering
from ._iterators import sampling2labels, Cartesian2DIterator
from ._sampling import grid_sampling2D, partial_fourier


def cartesian2D(
    g_slice_select: SimpleNamespace,
    slice_thickness: float,
    ny: int,
    n_slices: int,
    slice_gap: float = 0.0,
    Ry: int = 1,
    Rpf: float = 1.0,
    calib: int | None = None,
    view_order: str = "sequential",
    slice_order: str = "interleaved",
    view_loop_position: str = "inner",
    dummy_shots: int = 0,
):
    """
    Generate encoding table for 2D Cartesian imaging.

    This supports regular undersampling for Parallel Imaging
    and Partial Fourier acceleration.

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
    ny : int
        Number of phase encoding lines.
    n_slices : int
        Number of slices.
    slice_gap : float
        Slice gap in ``[mm]``.
    Ry : int, optional
        Parallel Imaging acceleration. The default is ``1`` (no acceleration).
    Rpf : float, optional
        Partial Fourier acceleration. The default is ``1.0`` (no acceleration).
        Must be > 0.5 (suggested > 0.7) and <= 1 (=1: no PF).
    calib : int | None = None, optional
        Image shape along phase encoding dim ``cy``.
        The default is ``None`` (no calibration).
    view_order : str, optional
        View ordering. Can be either ``"sequential"`` or ``"center-out"``.
        If ``"sequential"``, acquire all views in sequential order. If
        ``"center-out"``, acquisition ordering is symmetrical with respect to center.
        The default is ``"sequential"``.
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

    Returns
    -------
    NonCartesian2DIterator : object
        Iterator to keep trace of the shot index. Used to retrieve
        gradient amplitude and data labeling at a specific point during
        the scan loop.
    sampling_pattern : np.ndarray
        Cartesian sampling pattern of shape ``(ny,)``.

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

    # Compute phase encoding gradient scaling for each phase encoding step (from -0.5 to 0.5)
    phase_encoding_scaling = ((np.arange(ny)) - (ny // 2)) / ny

    # Compute sampling mask for phase encoding
    sampling_pattern = grid_sampling2D(ny, Ry, calib) * partial_fourier(ny, Rpf)

    # Apply undersampling
    phase_encoding_scaling = phase_encoding_scaling[sampling_pattern]
    phase_encoding_labels = sampling2labels(sampling_pattern)

    # Reorder views
    if view_order == "sequential":
        pass
    elif view_order == "center-out":
        phase_encoding_scaling = _ordering.center_out(phase_encoding_scaling)
        phase_encoding_labels = _ordering.center_out(phase_encoding_labels)
    else:
        raise ValueError(
            f"Unrecognized view order: {view_order} - must be either 'sequential' or 'center-out'."
        )

    # Generate encoding iterator
    return (
        Cartesian2DIterator(
            (phase_encoding_scaling, phase_encoding_labels.astype(int)),
            (slice_freq_offset, slice_labels.astype(int)),
            view_loop_position,
            dummy_shots,
        ),
        sampling_pattern,
    )
