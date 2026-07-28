"""Sampling-pattern factories driven by UI-facing numeric attributes.

The factories here start from matrix sizes, acceleration factors, train
lengths and view counts, and stop at k-space: they return a
:class:`~pulserver.ScanLoop` and nothing else. Frames, slices,
contrasts, averages and preparations are outer loops the sequence writes
itself — see :func:`~pulserver.design.make_slice_loop` and
:func:`~pulserver.design._lowlevel.calc_traversal_order` for the pieces those loops
are built from.
"""

from __future__ import annotations

import math

import numpy as np

from .cartesian import (
    build_from_mask,
    make_caipirinha_mask,
    make_poisson_disc_mask,
    make_random_mask,
)
from .epi import _from_shots, make_skipped_caipi_order
from .noncartesian import (
    make_golden_means_3d_tilt,
    make_radial_tilt,
    make_rotated_segment_tilt,
    make_spiral_phyllotaxis_tilt,
)

__all__ = [
    "make_cartesian_sampling",
    "make_epi_sampling",
    "make_noncartesian_2d_sampling",
    "make_noncartesian_projection_sampling",
    "make_rotated_projection_sampling",
]


def _matrix(value, dimensions=None):
    matrix = tuple(int(item) for item in value)
    if len(matrix) not in (2, 3) or min(matrix) < 1 or (dimensions is not None and len(matrix) != dimensions):
        expected = "2D or 3D" if dimensions is None else f"{dimensions}D"
        raise ValueError(f"matrix must be a positive {expected} shape")
    return matrix


def _axes(value, dimensions, name, *, minimum=0):
    values = (int(value),) * dimensions if np.ndim(value) == 0 else tuple(int(item) for item in value)
    if len(values) != dimensions or min(values) < minimum:
        raise ValueError(f"{name} must contain {dimensions} values >= {minimum}")
    return values


def _calibration_mask(shape, calibration):
    mask = np.zeros(shape, dtype=bool)
    if any(calibration):
        slices = tuple(
            slice(size // 2 - width // 2, size // 2 - width // 2 + width)
            for size, width in zip(shape, calibration, strict=True)
        )
        mask[slices] = True
    return mask


def _uniform_mask(shape, acceleration, calibration):
    axes = np.ix_(*(np.arange(size) for size in shape))
    mask = np.ones(shape, dtype=bool)
    for grid, factor in zip(axes, acceleration, strict=True):
        mask &= grid % factor == 0
    return mask | _calibration_mask(shape, calibration)


def _cartesian_mask(shape, acceleration, calibration, mask, seed, caipi_shift):
    if mask == "uniform":
        return _uniform_mask(shape, acceleration, calibration)
    if len(shape) != 2:
        raise ValueError(f"mask={mask!r} requires a 3D matrix")
    if mask == "caipi":
        return make_caipirinha_mask(shape, *acceleration, delta=int(caipi_shift)) | _calibration_mask(
            shape, calibration
        )
    total_acceleration = float(np.prod(acceleration))
    if mask == "random":
        return make_random_mask(shape, total_acceleration, calib=calibration, seed=seed)
    if mask == "poisson":
        return make_poisson_disc_mask(shape, total_acceleration, calib=calibration, seed=seed)
    raise ValueError("mask must be uniform, caipi, random, or poisson")


def make_cartesian_sampling(
    matrix,
    *,
    acceleration=1,
    calibration=0,
    train_length=1,
    ordering="linear",
    mask="uniform",
    seed=0,
    caipi_shift=1,
):
    """Build a 2D/3D GRE, FSE, or segmented-GRE sampling pattern.

    ``matrix`` is the full image matrix ``(nx, ny[, nz])``; the pattern spans
    the encoded axes ``ny[, nz]``. A train length of one gives GRE — including
    multiecho GRE, whose echoes live inside the readout module and are not a
    sampling dimension. FSE and MPRAGE-style segmented GRE use their
    ETL/segment length, and ``ordering='centric'`` is the conventional MPRAGE
    choice. The ``mask`` mode selects uniform, CAIPI, random, or Poisson-disc
    support.

    Parameters
    ----------
    matrix : tuple of int
        Full image matrix ``(nx, ny)`` for 2D or ``(nx, ny, nz)`` for 3D.
    acceleration : int or tuple of int, optional
        Undersampling factor per encoded axis.
    calibration : int or tuple of int, optional
        Fully sampled centered calibration width per encoded axis.
    train_length : int, optional
        Echo-train or segment length; ``1`` gives GRE.
    ordering : {'linear', 'centric', 'radial', 'radial_adaptive', 'shuffling'}, optional
        Echo-train view order; see :meth:`~pulserver.ScanLoop.from_mask`.
    mask : {'uniform', 'caipi', 'random', 'poisson'}, optional
        Undersampling pattern; ``'caipi'``, ``'random'`` and ``'poisson'``
        require a 3D matrix.
    seed : int, optional
        Random seed for ``mask='random'`` and ``ordering='shuffling'``.
    caipi_shift : int, optional
        CAIPI shift used only when ``mask='caipi'``.

    Returns
    -------
    ScanLoop
        One shot per echo train; ``pattern[shot]`` holds its absolute
        ``(ky[, kz])`` indices and ``pattern.mask`` the sampled support.

    Raises
    ------
    ValueError
        If the matrix is not 2D/3D, the calibration exceeds the encoded
        matrix, or a mask mode is used with the wrong dimensionality.

    Examples
    --------
    The same factory covers GRE, FSE and MPRAGE — what differs is how many
    encoded locations occur inside one excitation:

    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> gre = design.make_cartesian_sampling((128, 128), acceleration=2)
    >>> fse = design.make_cartesian_sampling((128, 128, 64), train_length=32, ordering="radial")
    >>> gre.n_shots, gre.shot_lengths[:2], fse.shot_lengths[:2]
    (64, (1, 1), (32, 32))

    Nest it under whatever outer loops the sequence needs; nothing here
    assumes a frame, slice or contrast loop exists::

        for frame in range(n_frames):
            for group in slices:
                for shot in sampling:
                    readout.set_state(lin_idx=int(shot[0, 0])).add_to(seq)

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import numpy as np
       import pulserver.design as design
       import pulserver.pypulseq as pp
       cases = [
           ("2D GRE", design.make_cartesian_sampling((64, 48), acceleration=2)),
           ("3D FSE, ETL 8", design.make_cartesian_sampling((64, 48, 24), train_length=8, ordering="radial")),
           ("3D MPRAGE", design.make_cartesian_sampling((64, 48, 24), train_length=12, ordering="centric")),
       ]
       fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), squeeze=False)
       for axis, (title, pattern) in zip(axes[0], cases, strict=True):
           positions = pattern.positions
           if positions.shape[1] == 1:
               positions = np.column_stack((positions[:, 0], positions[:, 0] * 0))
           echo = positions[:, 0] * 0
           for shot in pattern.shots:
               echo[shot] = range(len(shot))
           axis.scatter(positions[:, 0], positions[:, 1], c=echo, s=8, cmap="viridis")
           axis.set(title=title, xlabel="ky", ylabel="kz", aspect="equal")
       fig.tight_layout()

    See Also
    --------
    pulserver.ScanLoop.to_scales : the gradient factors for these indices.
    pulserver.design.make_slice_loop : the separate slice loop.
    """
    matrix = _matrix(matrix)
    encoded_shape = matrix[1:]
    acceleration = _axes(acceleration, len(encoded_shape), "acceleration", minimum=1)
    calibration = _axes(calibration, len(encoded_shape), "calibration", minimum=0)
    if any(width > size for width, size in zip(calibration, encoded_shape, strict=True)):
        raise ValueError("calibration cannot exceed the encoded matrix")
    support_mask = _cartesian_mask(encoded_shape, acceleration, calibration, mask, seed, caipi_shift)
    return build_from_mask(
        support_mask,
        train_length=int(train_length),
        ordering=ordering,
        seed=int(seed),
    )


def make_epi_sampling(matrix, *, acceleration=1, segments=1, caipi_shift=0):
    """Build a 2D or 3D EPI sampling pattern.

    ``segments`` is the number of interleaves along ky. A 2D pattern samples
    every ``Ry``-th line and distributes those lines across the segments. A 3D
    pattern uses the skipped-CAIPI traversal (Stirnberg & Stöcker, MRM 2020,
    DOI ``10.1002/mrm.28486``) for ``(Ry, Rz)`` and ``caipi_shift``.

    One shot is one echo train. Turning a structural EPI into a dynamic or
    fMRI acquisition needs nothing from the pattern — wrap the loop in
    ``for frame in range(n_frames):``.

    Parameters
    ----------
    matrix : tuple of int
        Full image matrix ``(nx, ny)`` for 2D or ``(nx, ny, nz)`` for 3D.
    acceleration : int or tuple of int, optional
        ``Ry`` for a 2D pattern; ``(Ry, Rz)`` for a 3D one.
    segments : int, optional
        Number of interleaves the ky traversal is split into.
    caipi_shift : int, optional
        Partition shift per acquired line, 3D only (see
        :func:`~pulserver.design._lowlevel.make_skipped_caipi_order`).

    Returns
    -------
    ScanLoop
        One shot per interleave; ``pattern.relative(shot)`` gives the
        within-train traversal an EPI readout is designed from, and
        ``pattern[shot]`` the absolute coordinates it is labelled with.

    Raises
    ------
    ValueError
        If ``segments`` is not positive or the matrix dimensionality does not
        match ``acceleration``.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> epi2d = design.make_epi_sampling((96, 96), acceleration=2, segments=2)
    >>> epi3d = design.make_epi_sampling((96, 96, 32), acceleration=(2, 2), caipi_shift=1, segments=2)
    >>> epi2d.n_shots, epi3d.n_shots
    (2, 32)

    A pattern may hold more than one traversal, so build one readout per shot
    and replay them across frames::

        readouts = [
            design.make_epi_readout(system, fov, matrix, sampling.n_shots, sampling.relative(shot))
            for shot in range(sampling.n_shots)
        ]
        for frame in range(n_frames):
            for shot, readout in enumerate(readouts):
                start = sampling[shot][0]
                readout.set_state(lin_idx=int(start[0]), par_idx=int(start[1])).add_to(seq)

    Each interleave is a path through the shared lattice. Left is a 2D pattern
    at ``Ry=2`` with 2 segments — it has no kz to grid against, so its
    segments are drawn one per row; right is the 3D skipped-CAIPI traversal
    at ``R=(1, 8)`` with shift 2 and 2 segments:

    .. plot::
       :include-source: false

       import pulserver.design as design
       import pulserver.pypulseq as pp
       from _figures import epi_sampling_figure
       epi_sampling_figure([
           design.make_epi_sampling((64, 32), acceleration=2, segments=2),
           design.make_epi_sampling((64, 32, 16), acceleration=(1, 8), caipi_shift=2, segments=2),
       ])
    """
    matrix = _matrix(matrix)
    segments = int(segments)
    if segments < 1:
        raise ValueError("segments must be positive")
    if len(matrix) == 2:
        ry = _axes(acceleration, 1, "acceleration", minimum=1)[0]
        sampled = np.arange(0, matrix[1], ry, dtype=np.intp)
        shots = [sampled[segment::segments, None] for segment in range(segments)]
        return _from_shots([shot for shot in shots if len(shot)], (matrix[1],))
    return make_skipped_caipi_order(
        matrix[1:],
        acceleration=_axes(acceleration, 2, "acceleration", minimum=1),
        caipi_shift=int(caipi_shift),
        segments=segments,
    )


def _default_inplane_views(matrix):
    return math.ceil(np.pi * max(matrix[:2]) / 2.0)


def make_noncartesian_2d_sampling(
    matrix,
    *,
    views=None,
    scheme="golden",
    period=np.pi,
    tiny_index=1,
    approximation_order=13,
    segment_length=1,
):
    """Build an in-plane rotation pattern for any 2D non-Cartesian readout.

    Plans *rotations*, not waveform shape, so it applies equally to radial,
    spiral and rosette base readouts: the readout designs one base arm and
    each entry here replays it under a different in-plane angle.

    ``views`` is the total number of rotations in the chronology, not a
    per-frame count. For a golden-angle acquisition that is the point — ask
    for one long chronology spanning every frame and slice and consume it with
    ``iter()``/``next()``, and every outer loop position sees fresh angles.
    Re-``iter()`` inside a loop instead, and each outer position replays the
    same angular set.

    Parameters
    ----------
    matrix : tuple of int
        2D image matrix ``(nx, ny)``; sets the default view count only.
    views : int or None, optional
        Number of rotated views; defaults to the Nyquist-matched count for
        ``matrix``.
    scheme : str, optional
        Angular tilt scheme; forwarded to
        :func:`~pulserver.design._lowlevel.make_radial_tilt`.
    period : float, optional
        Angular period (rad); forwarded to ``make_radial_tilt``.
    tiny_index : int, optional
        Tiny-golden-angle index; forwarded to ``make_radial_tilt``.
    approximation_order : int, optional
        Golden-angle rational-approximation order; forwarded to
        ``make_radial_tilt``.
    segment_length : int, optional
        Views per shot. The default of one gives a shot per view; a larger
        value groups the chronology into segments, which is what a segmented
        acquisition — an inversion-prepared or magnetisation-prepared radial
        train — plays between successive preparations.

    Returns
    -------
    ScanLoop
        One shot per segment of ``segment_length`` views, each holding its
        rotation angles (rad); with the default this is one view per shot and
        :meth:`~pulserver.ScanLoop.to_rotations` converts them to the
        matrices a readout module takes.

    Raises
    ------
    ValueError
        If ``matrix`` is not 2D or ``views`` is not positive.

    Examples
    --------
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> pattern = design.make_noncartesian_2d_sampling((64, 64), views=8)
    >>> pattern.to_rotations().shape
    (8, 3, 3)

    Continuous golden-angle chronology across frames and slices — one
    iterator, consumed by the innermost loop::

        rotations = iter(
            design.make_noncartesian_2d_sampling(
                matrix, views=n_frames * n_slices * n_views, scheme="tiny_golden", tiny_index=2
            ).to_rotations()
        )
        for frame in range(n_frames):
            for group in slices:
                for view in range(n_views):
                    readout.set_state(lin_idx=view, rotation=next(rotations)).add_to(seq)

    Replaying the same angular set at every slice instead is the same loop
    with ``rotations`` rebuilt inside it.

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pulserver.design as design
       import pulserver.pypulseq as pp
       fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), subplot_kw={"projection": "polar"})
       for axis, scheme in zip(axes, ("linear", "golden"), strict=True):
           angles = design.make_noncartesian_2d_sampling((64, 64), views=24, scheme=scheme).flatten()[:, 0]
           axis.scatter(angles, [1] * len(angles), c=range(len(angles)), s=14, cmap="viridis")
           axis.set_yticks([]); axis.set_title(scheme)
       fig.tight_layout()

    See Also
    --------
    pulserver.design._lowlevel.calc_traversal_order : the partition loop of a stack.
    """
    matrix = _matrix(matrix, 2)
    views = _default_inplane_views(matrix) if views is None else int(views)
    if views < 1:
        raise ValueError("views must be positive")
    return make_radial_tilt(
        views,
        scheme=scheme,
        period=float(period),
        tiny_index=int(tiny_index),
        approximation_order=int(approximation_order),
        segment_length=int(segment_length),
    )


def make_noncartesian_projection_sampling(matrix, *, views=None, scheme="golden_means", interleaves=1):
    """Build a 3D projection direction pattern.

    Directions vary over both in-plane azimuth and through-plane polar angle.
    Golden means is stable in every temporal window and is the default for
    dynamic imaging; spiral phyllotaxis is available when smooth interleaved
    direction changes are preferred.

    As for the in-plane factory, ``views`` is the whole chronology: request
    enough for every frame and consume it with an iterator to get a continuous
    3D golden-means acquisition.

    Parameters
    ----------
    matrix : tuple of int
        3D image matrix ``(nx, ny, nz)``; sets the default view count only.
    views : int or None, optional
        Number of direction views; defaults to a Nyquist-matched spherical
        count for ``matrix``.
    scheme : {'golden_means', 'phyllotaxis'}, optional
        Direction-tilt scheme.
    interleaves : int, optional
        Number of interleaves; used only by ``scheme='phyllotaxis'``.

    Returns
    -------
    ScanLoop
        One shot per view, each holding a unit direction vector;
        :meth:`~pulserver.ScanLoop.to_rotations` maps them onto the
        readout's design axis.

    Raises
    ------
    ValueError
        If ``matrix`` is not 3D, ``views`` is not positive, or ``scheme`` is
        unknown.

    Examples
    --------
    >>> import numpy as np
    >>> import pulserver.design as design
    >>> import pulserver.pypulseq as pp
    >>> pattern = design.make_noncartesian_projection_sampling((64, 64, 64), views=128)
    >>> np.allclose(np.linalg.norm(pattern.flatten(), axis=1), 1.0)
    True

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pulserver.design as design
       import pulserver.pypulseq as pp
       directions = design.make_noncartesian_projection_sampling((64, 64, 64), views=256).flatten()
       fig = plt.figure(figsize=(4.5, 4.2))
       ax = fig.add_subplot(projection="3d")
       ax.scatter(directions[:, 0], directions[:, 1], directions[:, 2],
                  c=range(len(directions)), s=8, cmap="viridis")
       ax.set_box_aspect((1, 1, 1)); ax.set_title("3D projection: golden-means directions")
    """
    matrix = _matrix(matrix, 3)
    views = math.ceil(np.pi * max(matrix) ** 2) if views is None else int(views)
    if views < 1:
        raise ValueError("views must be positive")
    if scheme == "golden_means":
        return make_golden_means_3d_tilt(views)
    if scheme == "phyllotaxis":
        return make_spiral_phyllotaxis_tilt(views, int(interleaves))
    raise ValueError("scheme must be golden_means or phyllotaxis")


def make_rotated_projection_sampling(
    matrix, *, shots, views=None, scheme="spiral", step_deg=None, require_fibonacci=True
):
    """Build a 3D projection pattern as one base segment plus per-shot rotations.

    :func:`make_noncartesian_projection_sampling` gives every spoke its own
    direction, which a readout must encode with its own gradient waveform. A
    continuous-gradient readout — ZTE above all — cannot pay that: an
    interpreter holds a bounded number of waveform shapes per gradient
    definition, so a few thousand freely placed spokes are rejected outright.

    This factory covers the same sphere with a segment that is *designed once*
    and replayed rigidly rotated, so the waveform count is set by ``views``
    alone and does not grow with ``shots``. The rotation travels with each
    block as an extension, which is what the scanner applies at playout.

    Parameters
    ----------
    matrix : tuple of int
        3D image matrix ``(nx, ny, nz)``; sets the default view count only.
    shots : int
        Number of rotated replays of the base segment.
    views : int or None, optional
        Spokes in the base segment. Defaults to a Nyquist-matched spherical
        count for ``matrix``, divided among ``shots``.
    scheme : {'spiral', 'disk', 'phyllotaxis'}, optional
        Base-segment ordering; see
        :func:`~pulserver.design._lowlevel.make_rotated_segment_tilt`. The
        first two step by a constant angle and so can be played from a single
        designed transition; ``'phyllotaxis'`` cannot.
    step_deg : float, optional
        Angular step for ``'spiral'``; defaults to the smallest feasible.
    require_fibonacci : bool, optional
        Enforce a Fibonacci ``shots`` for ``'phyllotaxis'`` (default True).

    Returns
    -------
    base : ScanLoop
        One shot holding the ``(views, 3)`` unit base directions, ready to
        hand to :func:`~pulserver.design.make_zte_readout` as its view order.
    rotations : numpy.ndarray
        ``(shots, 3, 3)`` rotation matrices for ``set_state(rotation=...)``.

    Raises
    ------
    ValueError
        If ``matrix`` is not 3D, or either count is not positive.

    Examples
    --------
    >>> import pulserver.design as design
    >>> base, rotations = design.make_rotated_projection_sampling((64, 64, 64), shots=13, views=32)
    >>> base.positions.shape, rotations.shape
    ((32, 3), (13, 3, 3))

    The whole scan is then one designed segment and a rotation per shot::

        readout = design.make_zte_readout(system, fov, nx, base.flatten(), excitation, tr_s=tr)
        for shot, rotation in enumerate(rotations):
            readout.set_state(lin_idx=shot * len(rotations) + np.arange(readout.num_views),
                              rotation=rotation).add_to(seq)

    See Also
    --------
    make_noncartesian_projection_sampling : one free direction per spoke.
    """
    matrix = _matrix(matrix, 3)
    shots = int(shots)
    if shots < 1:
        raise ValueError("shots must be positive")
    if views is None:
        total = math.ceil(np.pi * max(matrix) ** 2)
        views = max(1, -(-total // shots))
    views = int(views)
    if views < 1:
        raise ValueError("views must be positive")
    return make_rotated_segment_tilt(
        views, shots, scheme=scheme, step_deg=step_deg, require_fibonacci=require_fibonacci
    )
