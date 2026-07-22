"""High-level acquisition plans built from UI-facing numeric attributes.

The factories in this module start from matrix sizes, acceleration factors,
train lengths, frame counts, and slice geometry.  They deliberately stop at
numeric acquisition state: sequence code remains responsible for deciding
which RF and readout modules to play for each entry.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import ordering as outer_ordering
from ._pattern import SamplingPattern
from .cartesian import (
    build_from_mask,
    make_caipirinha_mask,
    make_poisson_disc_mask,
    make_random_mask,
)
from .epi import make_skipped_caipi_order
from .noncartesian import (
    angles_to_rotations,
    make_golden_means_3d_tilt,
    make_radial_tilt,
    make_spiral_phyllotaxis_tilt,
)
from .slices import SliceGroup, SliceSampling, make_slice_sampling

_KINDS = {"cartesian", "epi", "noncartesian_2d", "noncartesian_stack", "noncartesian_projection"}


def _readonly(value, *, dtype=None):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _index_value(values):
    values = np.asarray(values, dtype=np.intp)
    if values.size == 1:
        return int(values[0])
    return values.copy()


@dataclass(frozen=True)
class Acquisition:
    """One outer-loop iteration from an :class:`AcquisitionPlan`.

    ``coordinates`` contains the complete inner readout train.  For
    Cartesian plans its columns are ``ky[, kz]`` indices.  For in-plane
    non-Cartesian plans it contains the angle of the base trajectory; for a
    projection it contains a unit direction vector.  ``rotation`` is already
    the corresponding 3x3 matrix when the acquisition is non-Cartesian.

    The convenience properties expose only numeric state.  This keeps the
    plan independent of any particular readout implementation while making
    the usual sequence loop direct: Cartesian modules use ``lin_idx`` and
    ``par_idx``; EPI additionally uses ``relative`` and ``labels``;
    non-Cartesian modules use ``view`` and ``rotation``.
    """

    kind: str
    frame: int
    shot: int
    coordinates: np.ndarray
    slice_group: SliceGroup | None = None
    view: int | None = None
    partition: int | None = None
    rotation: np.ndarray | None = None

    def __post_init__(self):
        if self.kind not in _KINDS:
            raise ValueError(f"unknown acquisition kind {self.kind!r}")
        coordinates = np.asarray(self.coordinates)
        if coordinates.ndim == 1:
            coordinates = coordinates.reshape(1, -1)
        if coordinates.ndim != 2 or coordinates.shape[0] < 1 or not np.all(np.isfinite(coordinates)):
            raise ValueError("coordinates must be a non-empty finite 2D array")
        object.__setattr__(self, "coordinates", _readonly(coordinates))
        if self.rotation is not None:
            rotation = np.asarray(self.rotation, dtype=float)
            if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
                raise ValueError("rotation must be a finite 3x3 matrix")
            object.__setattr__(self, "rotation", _readonly(rotation))

    @property
    def lin_idx(self):
        """Cartesian line index, or non-Cartesian view label."""
        if self.kind.startswith("noncartesian"):
            return self.view
        if self.kind == "epi":
            return int(self.coordinates[0, 0])
        return _index_value(self.coordinates[:, 0])

    @property
    def par_idx(self):
        """Cartesian partition index, or stack partition label."""
        if self.kind == "noncartesian_stack":
            return self.partition
        if self.kind.startswith("noncartesian") or self.coordinates.shape[1] < 2:
            return None
        if self.kind == "epi":
            return int(self.coordinates[0, 1])
        return _index_value(self.coordinates[:, 1])

    @property
    def labels(self) -> np.ndarray:
        """Absolute Cartesian coordinates for ADC labels."""
        return self.coordinates.copy()

    @property
    def relative(self) -> np.ndarray:
        """Inner coordinates relative to the first acquired location."""
        return self.coordinates - self.coordinates[0]

    @property
    def increments(self) -> np.ndarray:
        """Coordinate increments between consecutive inner-loop samples."""
        return np.diff(self.coordinates, axis=0)

    @property
    def slice_indices(self) -> tuple[int, ...]:
        """Physical slices excited in this iteration (empty for 3D plans)."""
        return () if self.slice_group is None else self.slice_group.slice_indices

    @property
    def slice_positions_m(self) -> np.ndarray:
        """Physical slice positions in metres (empty for 3D plans)."""
        if self.slice_group is None:
            return np.empty(0, dtype=float)
        return self.slice_group.positions_m.copy()

    def frequency_offsets_hz(self, gradient_hz_per_m: float) -> np.ndarray:
        """Return the per-band slice-frequency table for this iteration."""
        if self.slice_group is None:
            return np.empty(0, dtype=float)
        return self.slice_group.frequency_offsets_hz(gradient_hz_per_m)


@dataclass(frozen=True)
class AcquisitionPlan(Sequence[Acquisition]):
    """Acquisition-ready outer and inner loop plan.

    Unlike :class:`SamplingPattern`, this high-level type includes frames and
    optional slice/SMS groups.  Indexing returns an :class:`Acquisition`
    containing all numeric state for one sequence iteration.  The underlying
    ``sampling`` pattern remains available for reconstruction masks and for
    advanced users who need to replace only the ordering stage.
    """

    kind: str
    matrix: tuple[int, ...]
    sampling: SamplingPattern
    entries: tuple[Acquisition, ...]
    frames: int = 1
    slice_sampling: SliceSampling | None = None

    def __post_init__(self):
        matrix = tuple(int(value) for value in self.matrix)
        if self.kind not in _KINDS or len(matrix) not in (2, 3) or min(matrix) < 1:
            raise ValueError("kind and matrix dimensionality are invalid")
        if int(self.frames) < 1:
            raise ValueError("frames must be positive")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "entries", tuple(self.entries))
        if self.slice_sampling is not None and not isinstance(self.slice_sampling, SliceSampling):
            raise TypeError("slice_sampling must be a SliceSampling instance or None")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return self.entries[index]

    @property
    def mask(self):
        """Cartesian sampling mask, or ``None`` for non-Cartesian plans."""
        return self.sampling.mask

    @property
    def n_acquisitions(self) -> int:
        """Number of outer-loop iterations in the complete plan."""
        return len(self.entries)

    @property
    def n_frames(self) -> int:
        """Number of structural/dynamic volumes."""
        return int(self.frames)

    @property
    def acquisitions_per_frame(self) -> int:
        """Number of sequence iterations in each frame/volume."""
        return len(self.entries) // int(self.frames)

    @property
    def slice_groups(self) -> tuple[SliceGroup, ...]:
        """Scheduled conventional/SMS groups (empty for 3D acquisition)."""
        return () if self.slice_sampling is None else self.slice_sampling.groups

    def slice_frequency_table_hz(self, gradient_hz_per_m: float) -> np.ndarray:
        """Return one row of RF offsets per slice/SMS group.

        The table belongs to the plan because it follows slice order and SMS
        grouping.  Gradient amplitude is supplied later, after the sequence
        has designed its selective pulse.
        """
        if self.slice_sampling is None:
            return np.empty((0, 0), dtype=float)
        return self.slice_sampling.frequency_table_hz(gradient_hz_per_m)


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


def _slice_table(num_slices, slice_spacing_m, slice_order, sms_factor):
    if num_slices is None:
        return None
    num_slices = int(num_slices)
    if num_slices == 1:
        if int(sms_factor) != 1:
            raise ValueError("sms_factor must be 1 for a single slice")
        return SliceSampling(np.array([0.0]), (SliceGroup(0, (0,), np.array([0.0])),), 1)
    if slice_spacing_m is None:
        raise ValueError("slice_spacing_m is required when num_slices > 1")
    return make_slice_sampling(
        num_slices,
        float(slice_spacing_m),
        order=slice_order,
        sms_factor=sms_factor,
    )


def _uniform_mask(shape, acceleration, calibration):
    axes = np.ix_(*(np.arange(size) for size in shape))
    mask = np.ones(shape, dtype=bool)
    for grid, factor in zip(axes, acceleration, strict=True):
        mask &= grid % factor == 0
    if any(calibration):
        slices = tuple(
            slice(size // 2 - width // 2, size // 2 - width // 2 + width)
            for size, width in zip(shape, calibration, strict=True)
        )
        mask[slices] = True
    return mask


def _calibration_mask(shape, calibration):
    mask = np.zeros(shape, dtype=bool)
    if any(calibration):
        slices = tuple(
            slice(size // 2 - width // 2, size // 2 - width // 2 + width)
            for size, width in zip(shape, calibration, strict=True)
        )
        mask[slices] = True
    return mask


def _cartesian_mask(shape, acceleration, calibration, mask, seed, caipi_shift):
    if mask == "uniform":
        return _uniform_mask(shape, acceleration, calibration)
    if len(shape) != 2:
        raise ValueError(f"mask={mask!r} requires a 3D matrix")
    if mask == "caipi":
        mask = make_caipirinha_mask(shape, *acceleration, delta=int(caipi_shift))
        mask |= _calibration_mask(shape, calibration)
        return mask
    total_acceleration = float(np.prod(acceleration))
    if mask == "random":
        return make_random_mask(shape, total_acceleration, calib=calibration, seed=seed)
    if mask == "poisson":
        return make_poisson_disc_mask(shape, total_acceleration, calib=calibration, seed=seed)
    raise ValueError("mask must be uniform, caipi, random, or poisson")


def _repeat_sampling(matrix, sampling, *, kind, frames, slice_sampling):
    groups = tuple(slice_sampling) if slice_sampling is not None else (None,)
    entries = []
    for frame in range(int(frames)):
        for group in groups:
            for shot in range(sampling.n_shots):
                entries.append(
                    Acquisition(
                        kind=kind,
                        frame=frame,
                        shot=shot,
                        coordinates=sampling[shot],
                        slice_group=group,
                    )
                )
    return AcquisitionPlan(kind, matrix, sampling, tuple(entries), int(frames), slice_sampling)


def make_cartesian_sampling(
    matrix,
    *,
    acceleration=1,
    calibration=0,
    train_length=1,
    ordering="linear",
    mask="uniform",
    frames=1,
    num_slices=None,
    slice_spacing_m=None,
    slice_order="interleaved",
    sms_factor=1,
    seed=0,
    caipi_shift=1,
):
    """Build a 2D/3D GRE, FSE, or segmented-GRE acquisition plan.

    ``matrix`` is the full image matrix ``(nx, ny[, nz])``.  A train length
    of one gives GRE (including multiecho GRE, whose echoes live inside the
    readout module); FSE and MPRAGE-style segmented GRE use their ETL/segment
    length.  ``ordering='centric'`` is the conventional MPRAGE choice. The
    ``mask`` mode selects uniform, CAIPI, random, or Poisson-disc support.

    Two-dimensional plans optionally include the slice/SMS and frame loops.
    Three-dimensional plans span ``(ky, kz)`` directly and therefore reject a
    slice table.

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
        Echo-train view order; see :func:`build_from_mask`.
    mask : {'uniform', 'caipi', 'random', 'poisson'}, optional
        Undersampling pattern; ``'caipi'``, ``'random'`` and ``'poisson'``
        require a 3D matrix.
    frames : int, optional
        Number of dynamic volumes; repeats the same shots without altering
        them.
    num_slices : int or None, optional
        Number of slices, 2D only; omit for a 3D (slab-encoded) plan.
    slice_spacing_m : float or None, optional
        Center-to-center slice spacing (m); required when ``num_slices > 1``.
    slice_order : str, optional
        Slice traversal order; see :func:`make_slice_sampling`.
    sms_factor : int, optional
        Simultaneous-multislice acceleration, 2D only.
    seed : int, optional
        Random seed for ``mask='random'`` and ``ordering='shuffling'``.
    caipi_shift : int, optional
        CAIPI shift used only when ``mask='caipi'``.

    Returns
    -------
    AcquisitionPlan
        Plan of kind ``"cartesian"``.

    Examples
    --------
    ``train_length=1`` gives a 2D GRE schedule; FSE uses an ETL and MPRAGE
    uses centric segments::

        gre = make_cartesian_sampling((128, 128), acceleration=2)
        fse = make_cartesian_sampling((128, 128, 64), train_length=32,
                                      ordering="radial")
        mprage = make_cartesian_sampling((256, 192, 160), train_length=128,
                                         ordering="centric")

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import numpy as np
       import pulserver.pypulseq as pp
       cases = [
           ("2D GRE", pp.make_cartesian_sampling((64, 48), acceleration=2)),
           ("3D FSE, ETL 8", pp.make_cartesian_sampling((64, 48, 24), train_length=8, ordering="radial")),
           ("3D MPRAGE", pp.make_cartesian_sampling((64, 48, 24), train_length=12, ordering="centric")),
       ]
       fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), squeeze=False)
       for axis, (title, plan) in zip(axes[0], cases, strict=True):
           support = plan.sampling.support
           if support.shape[1] == 1:
               support = np.column_stack((support[:, 0], support[:, 0] * 0))
           echo = support[:, 0] * 0
           for shot in plan.sampling.order:
               echo[shot] = range(len(shot))
           axis.scatter(support[:, 0], support[:, 1], c=echo, s=8, cmap="viridis")
           axis.set(title=title, xlabel="ky", ylabel="kz", aspect="equal")
       fig.tight_layout()
    """
    matrix = _matrix(matrix)
    encoded_shape = matrix[1:]
    acceleration = _axes(acceleration, len(encoded_shape), "acceleration", minimum=1)
    calibration = _axes(calibration, len(encoded_shape), "calibration", minimum=0)
    if any(width > size for width, size in zip(calibration, encoded_shape, strict=True)):
        raise ValueError("calibration cannot exceed the encoded matrix")
    if len(matrix) == 3 and num_slices is not None:
        raise ValueError("num_slices belongs only to a 2D acquisition plan")
    slice_sampling = _slice_table(num_slices, slice_spacing_m, slice_order, sms_factor)
    support_mask = _cartesian_mask(encoded_shape, acceleration, calibration, mask, seed, caipi_shift)
    pattern = build_from_mask(
        support_mask,
        train_length=int(train_length),
        ordering=ordering,
        seed=int(seed),
    )
    return _repeat_sampling(matrix, pattern, kind="cartesian", frames=int(frames), slice_sampling=slice_sampling)


def _pattern_from_shots(shots, shape):
    normalized = [np.asarray(shot, dtype=np.intp).reshape(-1, len(shape)) for shot in shots]
    support = []
    lookup = {}
    order = []
    for shot in normalized:
        indices = []
        for point in shot:
            key = tuple(int(value) for value in point)
            if key not in lookup:
                lookup[key] = len(support)
                support.append(key)
            indices.append(lookup[key])
        order.append(np.asarray(indices, dtype=np.intp))
    support_array = np.asarray(support, dtype=np.intp).reshape(-1, len(shape))
    mask = np.zeros(shape, dtype=bool)
    if len(support_array):
        mask[tuple(support_array.T)] = True
    return SamplingPattern(support_array, tuple(order), mask)


def make_epi_sampling(
    matrix,
    *,
    acceleration=1,
    segments=1,
    caipi_shift=0,
    frames=1,
    num_slices=None,
    slice_spacing_m=None,
    slice_order="interleaved",
    sms_factor=1,
):
    """Build a 2D or 3D EPI plan, including structural/fMRI outer loops.

    ``segments`` is the number of interleaves along ky.  A 2D plan samples
    every ``Ry``-th line and distributes those lines across the segments.  A
    3D plan uses the skipped-CAIPI traversal (Stirnberg & Stöcker, MRM 2020,
    DOI ``10.1002/mrm.28486``) for ``(Ry, Rz)`` and ``caipi_shift``.
    ``frames`` turns either structural plan into a dynamic or fMRI plan
    without changing the within-shot traversal — ``plan.sampling`` describes
    one frame's shots; ``frames`` only repeats them across ``plan.entries``.

    Parameters
    ----------
    matrix : tuple of int
        Full image matrix ``(nx, ny)`` for 2D or ``(nx, ny, nz)`` for 3D.
    acceleration : int or tuple of int, optional
        ``Ry`` for a 2D plan; ``(Ry, Rz)`` for a 3D plan.
    segments : int, optional
        Number of interleaves the ky traversal is split into.
    caipi_shift : int, optional
        Partition shift per acquired line, 3D only (see
        :func:`make_skipped_caipi_order`).
    frames : int, optional
        Number of structural/dynamic volumes; repeats the same shots without
        altering them.
    num_slices : int or None, optional
        Number of slices, 2D only; omit for a 3D (slab-encoded) plan.
    slice_spacing_m : float or None, optional
        Center-to-center slice spacing (m); required when ``num_slices > 1``.
    slice_order : str, optional
        Slice traversal order; see :func:`make_slice_sampling`.
    sms_factor : int, optional
        Simultaneous-multislice acceleration, 2D only.

    Returns
    -------
    AcquisitionPlan
        Plan of kind ``"epi"``.

    Examples
    --------
    A single-volume 3D EPI and a multi-volume 2D fMRI schedule differ only in
    the matrix and outer-loop arguments::

        fmri = make_epi_sampling((96, 96), acceleration=2, frames=200,
                                 num_slices=48, slice_spacing_m=3e-3)
        structural = make_epi_sampling((96, 96, 32), acceleration=(2, 2),
                                       caipi_shift=1, segments=2)

    Each interleave is a path through the shared lattice.  Left is a 2D plan
    at ``Ry=2`` with 2 segments — it has no kz to grid against, so its
    segments are drawn one per row; right is the 3D skipped-CAIPI traversal
    at ``R=(1, 8)`` with shift 2 and 2 segments:

    .. plot::
       :include-source: false

       import pulserver.pypulseq as pp
       from _figures import epi_sampling_figure
       epi_sampling_figure([
           pp.make_epi_sampling((64, 32), acceleration=2, segments=2),
           pp.make_epi_sampling((64, 32, 16), acceleration=(1, 8), caipi_shift=2, segments=2),
       ])
    """
    matrix = _matrix(matrix)
    segments, frames = int(segments), int(frames)
    if segments < 1 or frames < 1:
        raise ValueError("segments and frames must be positive")
    if len(matrix) == 2:
        ry = _axes(acceleration, 1, "acceleration", minimum=1)[0]
        sampled = np.arange(0, matrix[1], ry, dtype=np.intp)
        shots = [sampled[segment::segments, None] for segment in range(segments)]
        shots = [shot for shot in shots if len(shot)]
        pattern = _pattern_from_shots(shots, (matrix[1],))
        slice_sampling = _slice_table(num_slices, slice_spacing_m, slice_order, sms_factor)
    else:
        if num_slices is not None:
            raise ValueError("num_slices belongs only to a 2D EPI plan")
        acceleration = _axes(acceleration, 2, "acceleration", minimum=1)
        pattern = make_skipped_caipi_order(
            matrix[1:],
            acceleration=acceleration,
            caipi_shift=int(caipi_shift),
            segments=segments,
        )
        slice_sampling = None
    return _repeat_sampling(matrix, pattern, kind="epi", frames=frames, slice_sampling=slice_sampling)


def _default_inplane_views(matrix):
    return int(math.ceil(np.pi * max(matrix[:2]) / 2.0))


def _angle_pattern(count, *, scheme, period, tiny_index, approximation_order):
    return make_radial_tilt(
        count,
        scheme=scheme,
        period=period,
        tiny_index=tiny_index,
        approximation_order=approximation_order,
        segment_length=1,
    )


def _angle_values(total, repeat, *, scheme, period, tiny_index, approximation_order):
    count = repeat if total != repeat else total
    pattern = _angle_pattern(
        count,
        scheme=scheme,
        period=period,
        tiny_index=tiny_index,
        approximation_order=approximation_order,
    )
    values = pattern.flatten()[:, 0]
    if total != count:
        values = np.tile(values, int(math.ceil(total / count)))[:total]
        pattern = SamplingPattern(values[:, None], tuple(np.array([idx]) for idx in range(total)))
    return pattern, values


def make_noncartesian_2d_sampling(
    matrix,
    *,
    views_per_frame=None,
    frames=1,
    num_slices=None,
    slice_spacing_m=None,
    slice_order="interleaved",
    sms_factor=1,
    scheme="golden",
    period=np.pi,
    tiny_index=1,
    approximation_order=13,
    continuous=True,
):
    """Build a structural or dynamic 2D non-Cartesian rotation plan.

    By default, a golden/RAGA chronology continues across both slice groups
    and frames, so angles vary along both outer dimensions. Set
    ``continuous=False`` to replay the same angular set for every slice and
    frame.  The result applies equally to radial, spiral, and rosette base
    readouts because it plans rotations rather than waveform shape.

    Parameters
    ----------
    matrix : tuple of int
        2D image matrix ``(nx, ny)``.
    views_per_frame : int or None, optional
        Number of rotated views per frame; defaults to the Nyquist-matched
        view count for ``matrix``.
    frames : int, optional
        Number of structural/dynamic volumes.
    num_slices : int or None, optional
        Number of slices; omit to skip the slice/SMS loop.
    slice_spacing_m : float or None, optional
        Center-to-center slice spacing (m); required when ``num_slices > 1``.
    slice_order : str, optional
        Slice traversal order; see :func:`make_slice_sampling`.
    sms_factor : int, optional
        Simultaneous-multislice acceleration.
    scheme : str, optional
        Angular tilt scheme; forwarded to :func:`make_radial_tilt`.
    period : float, optional
        Angular period (rad); forwarded to :func:`make_radial_tilt`.
    tiny_index : int, optional
        Tiny-golden-angle index; forwarded to :func:`make_radial_tilt`.
    approximation_order : int, optional
        Golden-angle rational-approximation order; forwarded to
        :func:`make_radial_tilt`.
    continuous : bool, optional
        If True (default), the angular chronology advances across slices and
        frames; if False, the same angular set replays at every one.

    Returns
    -------
    AcquisitionPlan
        Plan of kind ``"noncartesian_2d"``; each entry's ``coordinates`` is
        its rotation angle (rad).

    Examples
    --------
    ``continuous=True`` (default) advances a dynamic radial, spiral, or
    rosette trajectory through both slice and frame; ``False`` replays a
    structural angular set at each outer location.

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pulserver.pypulseq as pp
       continuous = pp.make_noncartesian_2d_sampling((64, 64), views_per_frame=12,
           frames=2, num_slices=2, slice_spacing_m=5e-3, scheme="golden")
       replayed = pp.make_noncartesian_2d_sampling((64, 64), views_per_frame=12,
           frames=2, num_slices=2, slice_spacing_m=5e-3, scheme="golden", continuous=False)
       fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), subplot_kw={"projection": "polar"})
       for axis, (title, plan) in zip(axes, (("continuous", continuous), ("replayed", replayed)), strict=True):
           angles = [entry.coordinates[0, 0] for entry in plan]
           axis.scatter(angles, [1] * len(angles), c=range(len(angles)), s=14, cmap="viridis")
           axis.set_yticks([]); axis.set_title(title)
       fig.tight_layout()
    """
    matrix = _matrix(matrix, 2)
    frames = int(frames)
    views = _default_inplane_views(matrix) if views_per_frame is None else int(views_per_frame)
    if frames < 1 or views < 1:
        raise ValueError("frames and views_per_frame must be positive")
    slice_sampling = _slice_table(num_slices, slice_spacing_m, slice_order, sms_factor)
    loop_groups = tuple(slice_sampling) if slice_sampling is not None else (None,)
    total = frames * len(loop_groups) * views
    pattern, angles = _angle_values(
        total,
        views if not continuous else total,
        scheme=scheme,
        period=float(period),
        tiny_index=int(tiny_index),
        approximation_order=int(approximation_order),
    )
    rotations = angles_to_rotations(angles)
    entries = []
    chronological = 0
    for frame in range(frames):
        for group in loop_groups:
            for view in range(views):
                entries.append(
                    Acquisition(
                        "noncartesian_2d",
                        frame,
                        view,
                        np.array([[angles[chronological]]]),
                        group,
                        view,
                        None,
                        rotations[chronological],
                    )
                )
                chronological += 1
    return AcquisitionPlan("noncartesian_2d", matrix, pattern, tuple(entries), frames, slice_sampling)


def make_noncartesian_stack_sampling(
    matrix,
    *,
    views_per_partition=None,
    frames=1,
    partition_order="center_out",
    scheme="golden",
    period=np.pi,
    tiny_index=1,
    approximation_order=13,
    continuous=True,
):
    """Build a structural or dynamic stack-of-trajectories plan.

    Each entry carries an in-plane rotation and a Cartesian ``par_idx``.
    With ``continuous=True`` the angular chronology advances across both
    partition and frame; false reuses the same angles at every partition.

    Parameters
    ----------
    matrix : tuple of int
        3D image matrix ``(nx, ny, nz)``.
    views_per_partition : int or None, optional
        Number of rotated views per partition; defaults to the
        Nyquist-matched view count for ``matrix``.
    frames : int, optional
        Number of structural/dynamic volumes.
    partition_order : {'sequential', 'reverse', 'interleaved', 'center_out', 'outside_in'}, optional
        Traversal order of the ``nz`` partitions.
    scheme : str, optional
        Angular tilt scheme; forwarded to :func:`make_radial_tilt`.
    period : float, optional
        Angular period (rad); forwarded to :func:`make_radial_tilt`.
    tiny_index : int, optional
        Tiny-golden-angle index; forwarded to :func:`make_radial_tilt`.
    approximation_order : int, optional
        Golden-angle rational-approximation order; forwarded to
        :func:`make_radial_tilt`.
    continuous : bool, optional
        If True (default), the angular chronology advances across partitions
        and frames; if False, the same angular set replays at every partition.

    Returns
    -------
    AcquisitionPlan
        Plan of kind ``"noncartesian_stack"``; each entry's ``coordinates``
        is its rotation angle (rad) and ``par_idx`` its kz partition.

    Examples
    --------
    A stack of rotated in-plane trajectories, tilted continuously across
    partition and frame::

        pp.make_noncartesian_stack_sampling((64, 64, 8), views_per_partition=8,
                                            frames=2, partition_order="center_out")

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import pulserver.pypulseq as pp
       plan = pp.make_noncartesian_stack_sampling((64, 64, 8), views_per_partition=8,
           frames=2, partition_order="center_out", scheme="golden")
       fig, ax = plt.subplots(figsize=(6, 3.2))
       ax.scatter(range(len(plan)), [entry.par_idx for entry in plan],
                  c=[entry.coordinates[0, 0] for entry in plan], s=16, cmap="twilight")
       ax.set(xlabel="acquisition chronology", ylabel="kz partition",
              title="stack: continuous in-plane tilt across partition and frame")
       fig.tight_layout()
    """
    matrix = _matrix(matrix, 3)
    frames = int(frames)
    views = _default_inplane_views(matrix) if views_per_partition is None else int(views_per_partition)
    if frames < 1 or views < 1:
        raise ValueError("frames and views_per_partition must be positive")
    try:
        partitions = getattr(outer_ordering, partition_order)(matrix[2])
    except AttributeError as error:
        raise ValueError("partition_order must be sequential, reverse, interleaved, center_out, or outside_in") from error
    total = frames * matrix[2] * views
    pattern, angles = _angle_values(
        total,
        views if not continuous else total,
        scheme=scheme,
        period=float(period),
        tiny_index=int(tiny_index),
        approximation_order=int(approximation_order),
    )
    rotations = angles_to_rotations(angles)
    entries = []
    chronological = 0
    for frame in range(frames):
        for partition in partitions:
            for view in range(views):
                entries.append(
                    Acquisition(
                        "noncartesian_stack",
                        frame,
                        view,
                        np.array([[angles[chronological]]]),
                        None,
                        view,
                        int(partition),
                        rotations[chronological],
                    )
                )
                chronological += 1
    return AcquisitionPlan("noncartesian_stack", matrix, pattern, tuple(entries), frames)


def make_noncartesian_projection_sampling(
    matrix,
    *,
    views_per_frame=None,
    frames=1,
    scheme="golden_means",
    interleaves=1,
    continuous=True,
):
    """Build a structural or dynamic 3D projection rotation plan.

    Directions vary over both in-plane azimuth and through-plane polar angle.
    Golden means is stable in every temporal window and is the default for
    dynamic imaging. Spiral phyllotaxis is available when smooth interleaved
    direction changes are preferred.

    Parameters
    ----------
    matrix : tuple of int
        3D image matrix ``(nx, ny, nz)``.
    views_per_frame : int or None, optional
        Number of direction views per frame; defaults to a Nyquist-matched
        spherical view count for ``matrix``.
    frames : int, optional
        Number of structural/dynamic volumes.
    scheme : {'golden_means', 'phyllotaxis'}, optional
        Direction-tilt scheme.
    interleaves : int, optional
        Number of interleaves; used only by ``scheme='phyllotaxis'``.
    continuous : bool, optional
        If True (default), the direction chronology advances across frames;
        if False, the same direction set replays at every frame.

    Returns
    -------
    AcquisitionPlan
        Plan of kind ``"noncartesian_projection"``; each entry's
        ``coordinates`` is its unit direction vector.

    Examples
    --------
    Dynamic golden-means directions spanning two frames::

        pp.make_noncartesian_projection_sampling((64, 64, 64),
                                                 views_per_frame=128, frames=2)

    .. plot::
       :include-source: false

       import matplotlib.pyplot as plt
       import numpy as np
       import pulserver.pypulseq as pp
       plan = pp.make_noncartesian_projection_sampling((64, 64, 64),
           views_per_frame=128, frames=2, scheme="golden_means")
       directions = np.concatenate([entry.coordinates for entry in plan])
       fig = plt.figure(figsize=(4.5, 4.2))
       ax = fig.add_subplot(projection="3d")
       ax.scatter(directions[:, 0], directions[:, 1], directions[:, 2],
                  c=range(len(directions)), s=8, cmap="viridis")
       ax.set_box_aspect((1, 1, 1)); ax.set_title("3D projection: dynamic golden-means directions")
    """
    matrix = _matrix(matrix, 3)
    frames = int(frames)
    views = int(math.ceil(np.pi * max(matrix) ** 2)) if views_per_frame is None else int(views_per_frame)
    if frames < 1 or views < 1:
        raise ValueError("frames and views_per_frame must be positive")
    count = frames * views if continuous else views
    if scheme == "golden_means":
        base = make_golden_means_3d_tilt(count)
    elif scheme == "phyllotaxis":
        base = make_spiral_phyllotaxis_tilt(count, int(interleaves))
    else:
        raise ValueError("scheme must be golden_means or phyllotaxis")
    directions = base.flatten()
    if not continuous:
        directions = np.tile(directions, (frames, 1))
        pattern = SamplingPattern(directions, tuple(np.array([idx]) for idx in range(len(directions))))
    else:
        pattern = SamplingPattern(directions, tuple(np.array([idx]) for idx in range(len(directions))))
    rotations = SamplingPattern(directions, (np.arange(len(directions)),)).to_rotations()
    entries = []
    chronological = 0
    for frame in range(frames):
        for view in range(views):
            entries.append(
                Acquisition(
                    "noncartesian_projection",
                    frame,
                    view,
                    directions[chronological][None, :],
                    None,
                    view,
                    None,
                    rotations[chronological],
                )
            )
            chronological += 1
    return AcquisitionPlan("noncartesian_projection", matrix, pattern, tuple(entries), frames)
