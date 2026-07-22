"""Core sampling-plan data model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _readonly_copy(value, *, dtype=None):
    out = np.array(value, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class SamplingPattern(Sequence[np.ndarray]):
    """Sampled support plus its shot-by-shot acquisition order.

    Every low-level sampling helper in :mod:`pulserver.pypulseq` returns one
    of these.
    *Where* k-space is sampled and *when* each location is acquired are kept
    separate: ``support`` lists the distinct sampled locations once, and each
    entry of ``order`` indexes into it for one shot. Trains of unequal length
    are supported by construction.

    A pattern is an immutable :class:`~collections.abc.Sequence` of shots:
    ``len(pattern)`` is the shot count and ``pattern[i]`` returns shot ``i``'s
    coordinates, already in acquisition order — so a sequence loop reads as
    ``for shot in pattern:``.

    Three alternate constructors build one without listing coordinates by
    hand: :meth:`from_mask` from a Cartesian sampling mask,
    :meth:`from_relative_shifts` from per-shot starts plus blip increments,
    and the ``make_*_sampling`` factories for analytic plans.

    Parameters
    ----------
    support : array_like
        Shape ``(n_points, D)``: Cartesian ``(ky[, kz])`` indices, spoke
        angles, or unit direction vectors. 1D input is treated as ``(n, 1)``.
    order : tuple of numpy.ndarray
        One integer index array per shot, indexing into ``support``.
    mask : numpy.ndarray, optional
        Cartesian boolean mask, when the support came from one. Its nonzero
        count must equal ``len(support)``.

    Examples
    --------
    >>> from pulserver.pypulseq import make_radial_tilt
    >>> pattern = make_radial_tilt(8, segment_length=4)
    >>> pattern.n_shots, pattern.inner_lengths
    (2, (4, 4))
    >>> pattern[0].shape
    (4, 1)

    Relative steps for a blipped train, instead of absolute coordinates:

    >>> from pulserver import SamplingPattern
    >>> plan = SamplingPattern.from_relative_shifts([[0, 0]], [[0, 0], [2, 1]], shape=(8, 4))
    >>> plan.relative(0)
    array([[0, 0],
           [2, 1]])
    >>> plan.increments(0)
    array([[2, 1]])
    """

    support: np.ndarray
    order: tuple[np.ndarray, ...]
    mask: np.ndarray | None = None

    def __post_init__(self):
        support = np.asarray(self.support)
        if support.ndim == 0:
            support = support.reshape(1, 1)
        elif support.ndim == 1:
            support = support.reshape(-1, 1)
        elif support.ndim != 2:
            raise ValueError("support must be scalar, 1D, or 2D")
        if not np.issubdtype(support.dtype, np.number):
            raise TypeError("support must be numeric")
        if not np.all(np.isfinite(support)):
            raise ValueError("support must contain only finite values")
        support = _readonly_copy(support)

        order = tuple(self.order)
        normalized = []
        for shot in order:
            indices = np.asarray(shot)
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                raise TypeError("each order entry must be a one-dimensional integer array")
            indices = _readonly_copy(indices, dtype=np.intp)
            if indices.size and (indices.min() < 0 or indices.max() >= len(support)):
                raise IndexError("order contains an out-of-range support index")
            normalized.append(indices)

        mask = None
        if self.mask is not None:
            mask = _readonly_copy(self.mask, dtype=bool)
            if int(np.count_nonzero(mask)) != len(support):
                raise ValueError("mask sample count must equal support length")

        object.__setattr__(self, "support", support)
        object.__setattr__(self, "order", tuple(normalized))
        object.__setattr__(self, "mask", mask)

    @classmethod
    def from_mask(
        cls,
        mask,
        *,
        train_length: int = 1,
        ordering: str = "linear",
        seed: int = 0,
        n_sections: int = 3,
    ) -> SamplingPattern:
        """Build a plan from a Cartesian mask by ordering its sampled lines.

        The bridge from *which* lines are sampled (a boolean mask) to *when*
        they are acquired (echo trains). Every sampled location is covered
        exactly once — the result is verified before returning — and the
        originating mask is retained on the pattern for the reconstruction.

        Use ``train_length=1`` for GRE and other acquisitions without an inner
        train; the result is then one shot per sampled line.

        Parameters
        ----------
        mask : array_like of bool
            1D ``(ny,)`` or 2D ``(ny, nz)`` sampling mask.
        train_length : int, optional
            Echo-train / segment length (default 1).
        ordering : {'linear', 'centric', 'radial', 'radial_adaptive', 'shuffling'}, optional
            Echo-train ordering applied to the sampled locations.
        seed : int, optional
            Seed for ``ordering='shuffling'``.
        n_sections : int, optional
            Radial section count for ``ordering='radial_adaptive'``.

        Returns
        -------
        SamplingPattern
            ``support`` of sampled ``(ky[, kz])`` indices, one order entry per
            shot, and the originating ``mask``.

        Raises
        ------
        RuntimeError
            If the chosen ordering does not cover the mask exactly once.

        Examples
        --------
        >>> import pulserver.pypulseq as pp
        >>> from pulserver import SamplingPattern
        >>> mask = pp.make_caipirinha_mask((32, 8), 2, 2, delta=1)
        >>> plan = SamplingPattern.from_mask(mask, train_length=8, ordering="radial")
        >>> plan.n_samples == int(mask.sum())
        True
        >>> plan.inner_lengths
        (8, 8, 8, 8, 8, 8, 8, 8)

        Which shot acquires each sampled line, and at which echo of its train:

        .. plot::
           :include-source: false

           import pulserver.pypulseq as pp
           from pulserver import SamplingPattern
           from _figures import pattern_figure
           mask = pp.make_poisson_disc_mask((48, 48), 4.0, calib=(10, 10), seed=0)
           pattern_figure(SamplingPattern.from_mask(mask, train_length=16, ordering="radial"),
                          title="Poisson-disc R=4, radial train of 16")

        See Also
        --------
        pulserver.pypulseq.make_random_mask : mask generators.
        pulserver.pypulseq.make_radial_order : one of the underlying orderings.
        """
        from .cartesian import build_from_mask

        return build_from_mask(
            mask,
            train_length=train_length,
            ordering=ordering,
            seed=seed,
            n_sections=n_sections,
        )

    @classmethod
    def from_relative_shifts(cls, starts, shifts, *, shape) -> SamplingPattern:
        """Build a plan from per-shot start points and blip increments.

        The natural description of any blipped train: a shot starts somewhere
        in (ky, kz) and each subsequent echo is reached by a *relative* step.
        Both views the sequence needs come out of the result —
        ``pattern[shot]`` gives the absolute coordinates to label,
        ``pattern.increments(shot)`` the blip areas to play.

        ``shifts`` may be one shared list of increments, or one list per shot
        when trains differ (EPTI, zigzag).

        Parameters
        ----------
        starts : array_like
            Shape ``(n_shots, D)`` integer start coordinates.
        shifts : array_like
            Shape ``(inner, D)`` shared by all shots, or
            ``(n_shots, inner, D)``. Entry 0 is normally all-zero so the first
            echo sits on ``starts``.
        shape : tuple of int
            Cartesian grid extent; every coordinate must fall inside it.

        Returns
        -------
        SamplingPattern
            Deduplicated ``support`` of absolute coordinates, one order entry
            per shot, plus the implied Cartesian ``mask``.

        Examples
        --------
        >>> from pulserver import SamplingPattern
        >>> plan = SamplingPattern.from_relative_shifts(
        ...     starts=[[0, 0], [1, 0]],
        ...     shifts=[[0, 0], [2, 1], [4, 2]],
        ...     shape=(8, 4),
        ... )
        >>> plan[0]
        array([[0, 0],
               [2, 1],
               [4, 2]])
        >>> plan.increments(0)
        array([[2, 1],
               [2, 1]])

        See Also
        --------
        pulserver.pypulseq.make_skipped_caipi_order : segmented blipped-CAIPI
            plan built on this representation.
        """
        from .epi import build_from_relative_shifts

        return build_from_relative_shifts(starts, shifts, shape=shape)

    def to_rotations(self, *, reference=(1.0, 0.0, 0.0)) -> np.ndarray:
        """Convert this pattern's support into per-shot rotation matrices.

        The bridge from a non-Cartesian plan to the sequence: the readout
        designs **one** base waveform along ``reference``, and each entry of
        ``support`` replays it under the matrix returned here. A one-column
        support is read as in-plane spoke angles (rotation about ``z``); a
        three-column support as unit directions, each mapped by the shortest
        rotation taking ``reference`` onto it (Rodrigues). Antipodal
        directions get a well-defined 180-degree rotation rather than a
        singular matrix.

        Feed the result to :func:`pulserver.pypulseq.make_rotation`, or pass a
        row to a readout module's ``rotation=`` argument.

        Parameters
        ----------
        reference : array_like, optional
            Direction the base waveform was designed along (default ``+x``).
            Ignored for angle-valued support.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(support), 3, 3)`` rotation matrices.

        Raises
        ------
        ValueError
            If ``support`` has neither one nor three columns.

        Examples
        --------
        >>> import numpy as np
        >>> import pulserver.pypulseq as pp
        >>> rotations = pp.make_golden_means_3d_tilt(4).to_rotations()
        >>> rotations.shape
        (4, 3, 3)
        >>> angles = pp.make_radial_tilt(4).to_rotations()
        >>> np.round(angles[1] @ np.array([1.0, 0.0, 0.0]), 6)
        array([0.707107, 0.707107, 0.      ])

        Rotate a 3D radial base waveform shot by shot::

            for rotation in pp.make_golden_means_3d_tilt(1000).to_rotations():
                readout(seq, rotation=rotation)
        """
        from .noncartesian import angles_to_rotations, directions_to_rotations

        if self.support.shape[1] == 1:
            return angles_to_rotations(self.support[:, 0])
        if self.support.shape[1] == 3:
            return directions_to_rotations(self.support, reference=reference)
        raise ValueError("to_rotations needs a support of spoke angles (1 column) or directions (3 columns)")

    @property
    def n_shots(self) -> int:
        """Number of shots in the plan."""
        return len(self.order)

    @property
    def n_samples(self) -> int:
        """Total number of acquired points, summed over shots."""
        return sum(len(shot) for shot in self.order)

    @property
    def inner_lengths(self) -> tuple[int, ...]:
        """Train length of each shot."""
        return tuple(len(shot) for shot in self.order)

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, shot):
        if isinstance(shot, slice):
            return tuple(self.support[idx] for idx in self.order[shot])
        return self.support[self.order[shot]]

    def shot_indices(self, shot: int) -> np.ndarray:
        """Return shot ``shot``'s indices into ``support`` (not the values)."""
        return self.order[shot]

    def flatten(self) -> np.ndarray:
        """Return every acquired coordinate, concatenated in acquisition order."""
        if not self.order:
            return np.empty((0, self.support.shape[1]), dtype=self.support.dtype)
        return self.support[np.concatenate(self.order)]

    def relative(self, shot: int) -> np.ndarray:
        """Return shot ``shot``'s coordinates relative to its own first point.

        The prewinder-relative view: what the gradients must reach once the
        shot's start has been played.
        """
        values = self[shot]
        if not len(values):
            return values.copy()
        return values - values[0]

    def increments(self, shot: int) -> np.ndarray:
        """Return the step between consecutive points of shot ``shot``.

        One entry shorter than the shot: these are the blip areas to play
        between echoes.
        """
        return np.diff(self[shot], axis=0)
