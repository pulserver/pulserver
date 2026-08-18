"""Cartesian k-space masks and absolute view-ordering helpers.

The echo-train / segment view-ordering schemes (``linear``, ``radial``,
``radial_adaptive``, ``shuffling``) apply to any acquisition with an outer
loop and an inner echo train / segment — 2D and 3D FSE as well as
MPRAGE-style segmented GRE. They operate on a set of phase-encode locations
in the (ky, kz) plane (kz all-zero for a 2D single-partition acquisition)
and return a list of shots, each shot an ordered list of view indices (the
echo/segment index is the position within the shot).

References
----------
FSE linear / radial / radial-adaptive reordering follow Buonincontri et al.,
"Doubling the repetition time without paying the price: 3D TSE with
individually parameterized echo trains", ISMRM abstract 566-05-007 (Fig. 2),
in ``refcode/Abstract 566-05-007.pdf``. Random shuffling follows Tamir et
al., "T2 Shuffling", Magn Reson Med 2017;77:180-195, in
``refcode/nihms804984.pdf``.
"""

from __future__ import annotations

import numpy as np

from ._axes import EncodingAxis
from ._scanloop import ScanLoop
from ...pypulseq._masks import (  # noqa: F401
    calc_sampled_lines,
    make_caipirinha_mask,
    make_centric_order,
    make_linear_order,
    make_poisson_disc_mask,
    make_radial_adaptive_order,
    make_radial_order,
    make_random_mask,
    make_shuffling_order,
)
from ...pypulseq._ordering import calc_chunk_indices  # noqa: F401

#: Counter each encoded Cartesian axis is labelled with, in column order.
CARTESIAN_LABELS = ("LIN", "PAR")


def cartesian_axes(shape) -> tuple[EncodingAxis, ...]:
    """Return the ``LIN``/``PAR`` index axes of an encoded matrix ``shape``.

    The extents come from the *encoded matrix*, not from the sampled support,
    so an accelerated or partial-Fourier loop still scales its phase encodes
    against the full matrix.
    """
    return tuple(
        EncodingAxis(label, kind="index", size=int(size))
        for label, size in zip(CARTESIAN_LABELS, tuple(shape), strict=False)
    )




def calc_encoding_scales(coordinates, shape) -> np.ndarray:
    """Convert Cartesian encoding indices into phase-encode gradient scale factors.

    Index ``i`` of an axis with ``n`` steps encodes area ``(i - n/2) / fov``.
    The full-amplitude template :func:`~pulserver.design.make_phase_encoding`
    builds carries ``n / (2 * fov)``, so the factor
    :func:`pypulseq.scale_grad` needs is ``(i - n/2) / (n/2)`` — in ``[-1, 1)``,
    independent of fov. Keeping that arithmetic in one place is what lets a
    hand-written readout and a shipped module encode identical k-space.

    The factor depends only on the index and the matrix size, never on which
    views an acquisition happens to play: acceleration and partial Fourier
    drop views from the scan loop, they do not rescale the ones that remain.

    Works at any granularity: a whole loop's positions, one shot's
    coordinates, or a single point.

    Parameters
    ----------
    coordinates : array_like
        Integer indices: shape ``(..., D)``, ``(n,)`` for a single axis, or a
        bare scalar index.
    shape : int or tuple of int
        Encoded matrix extent per coordinate column.

    Returns
    -------
    numpy.ndarray
        Float factors in ``[-1, 1)``, same shape as ``coordinates``.

    Raises
    ------
    ValueError
        If ``shape`` does not match the coordinate columns or an extent is
        not a positive integer.

    Examples
    --------
    >>> import numpy as np
    >>> from pulserver.design import calc_encoding_scales
    >>> calc_encoding_scales(np.arange(8), 8).round(2).tolist()
    [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
    >>> calc_encoding_scales([[0, 4], [4, 0]], (8, 8)).tolist()
    [[-1.0, 0.0], [0.0, -1.0]]
    >>> float(calc_encoding_scales(0, 8))
    -1.0

    See Also
    --------
    pulserver.ScanLoop.to_scales : the same conversion for a whole scan loop.
    pulserver.design.make_phase_encoding : the template these scale.
    """
    values = np.asarray(coordinates, dtype=float)
    extents = np.atleast_1d(np.asarray(shape, dtype=float))
    if np.any(extents <= 0) or not np.all(extents == np.rint(extents)):
        raise ValueError("shape entries must be positive integers")
    columns = 1 if values.ndim < 2 else values.shape[-1]
    if len(extents) != columns:
        raise ValueError(f"shape has {len(extents)} entries but coordinates have {columns} columns")
    if values.ndim < 2:
        extents = extents[0]
    return (values - 0.5 * extents) / (0.5 * extents)




















def build_from_mask(
    mask: np.ndarray,
    *,
    train_length: int = 1,
    ordering: str = "linear",
    seed: int = 0,
    n_sections: int = 3,
) -> ScanLoop:
    """Implementation of :meth:`ScanLoop.from_mask`."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim not in (1, 2):
        raise ValueError("mask must be one- or two-dimensional")
    train_length = int(train_length)
    if train_length <= 0:
        raise ValueError("train_length must be positive")
    axes = cartesian_axes(mask.shape)
    support = np.argwhere(mask)
    if not len(support):
        return ScanLoop(support, (), mask, axes)
    coords = support[:, 0] if mask.ndim == 1 else support
    if ordering == "linear":
        shots = make_linear_order(coords, train_length)
    elif ordering in ("centric", "center_out"):
        shots = make_centric_order(coords, train_length)
    elif ordering == "radial":
        shots = make_radial_order(coords, train_length)
    elif ordering == "radial_adaptive":
        shots = make_radial_adaptive_order(coords, train_length, n_sections=n_sections)
    elif ordering == "shuffling":
        shots = make_shuffling_order(coords, train_length, seed=seed)
    else:
        raise ValueError("ordering must be linear, centric, radial, radial_adaptive, or shuffling")
    indices = tuple(np.asarray(shot, dtype=np.intp) for shot in shots)
    flattened = np.concatenate(indices) if indices else np.empty(0, dtype=np.intp)
    if len(flattened) != len(support) or len(np.unique(flattened)) != len(support):
        raise RuntimeError("Cartesian ordering did not cover the mask exactly once")
    return ScanLoop(support, indices, mask, axes)


