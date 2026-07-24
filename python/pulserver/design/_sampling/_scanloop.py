"""The scan-loop data model shared by k-space, slice and any other outer loop."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from ._axes import EncodingAxis, infer_axes


def _readonly_copy(value, *, dtype=None):
    out = np.array(value, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class ScanLoop(Sequence[np.ndarray]):
    """One ordered list of scan-loop iterations: what varies, and in what order.

    Every sampling and scheduling helper in :mod:`pulserver.design` returns
    one of these, because every outer loop in an MR sequence has the same
    shape: a table of **positions** the loop visits, a grouping of those
    positions into **shots** — one shot being one excitation's worth of the
    loop, in acquisition order — and one :class:`EncodingAxis` per position
    column saying what the numbers mean.

    A loop is *not* a k-space object. An encoding position is a set of
    frequency offsets, gradient scalings and rotations, and nothing about that
    restricts it to ``ky``/``kz``/``z``/interleave angle: a frame of a dynamic
    series, an inversion time, a b-value or a saturation offset is the same
    table under a different :attr:`axes` declaration.

    ==================== =============================== ====================
    Loop                 ``positions`` hold              feed a module as
    ==================== =============================== ====================
    Cartesian k-space    ``(ky[, kz])`` integer indices  ``lin_idx``/``par_idx``
    non-Cartesian        spoke angles or unit directions ``rotation``
    slices / SMS         positions along the axis (m)    ``freq_offset_hz``
    frames, contrasts    any physical parameter          a delay, an
                                                         amplitude, a flip
    ==================== =============================== ====================

    A loop is an immutable :class:`~collections.abc.Sequence` of shots:
    ``len(loop)`` is the shot count and ``loop[i]`` returns shot ``i``'s
    positions, already in acquisition order — so the sequence loop reads as
    ``for shot in loop:``. Trains of unequal length are supported by
    construction, so an echo train, a 3D partition and an SMS band group are
    all just shots of differing length.

    It deliberately carries **no** loop nesting. Frames outside slices outside
    shots, or the reverse, are `for` statements you write, which is what keeps
    a trigger, an inversion or a dummy TR insertable at any nesting level.

    It carries no gradients and no events — it carries the numbers those are
    derived from, plus the counters that let the reconstruction find the data
    again:

    - **Counters.** :meth:`labels` returns the ``SET <label>`` events for one
      shot, one per axis. Emitting them is not optional bookkeeping: the
      interpreter derives every ``FIRST_IN_*``/``LAST_IN_*`` MRD flag from the
      *observed range* of each counter, so a dimension that is looped but
      never labelled collapses to a single index and its flags fire on every
      acquisition. See :meth:`label_limits`.
    - **Amplitudes.** :meth:`to_scales` gives the ``[-1, 1]`` factors for
      :func:`pypulseq.scale_grad`, :meth:`to_rotations` the per-view rotation
      matrices, and :meth:`to_frequencies` the RF offsets that select a slice.
      Each applies to the position semantics it names and raises otherwise.

    Three alternate constructors build one without listing positions by hand:
    :meth:`from_mask` from a Cartesian sampling mask,
    :meth:`from_relative_shifts` from per-shot starts plus blip increments,
    and the ``make_*_sampling`` / ``make_slice_loop`` /
    :func:`~pulserver.design.make_counter_loop` factories.

    Parameters
    ----------
    positions : array_like
        Shape ``(n_positions, D)``. 1D input is treated as ``(n, 1)``.
    shots : tuple of numpy.ndarray
        One integer index array per shot, indexing into ``positions``.
    mask : numpy.ndarray, optional
        Cartesian support mask, when the positions came from one. Its nonzero
        count must equal ``len(positions)``.
    axes : tuple of EncodingAxis, optional
        One axis per position column (three columns for a ``'direction'``
        axis). Inferred when omitted: integer columns become ``LIN``/``PAR``
        encoding indices, three float columns a ``direction`` axis, anything
        else a neutral ``value`` axis.

    Examples
    --------
    >>> from pulserver.design._lowlevel import make_radial_tilt
    >>> loop = make_radial_tilt(8, segment_length=4)
    >>> loop.n_shots, loop.shot_lengths
    (2, (4, 4))
    >>> loop[0].shape
    (4, 1)

    The two views a Cartesian sequence loop needs, from one object:

    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> import pulserver.pypulseq as pp
    >>> loop = design.make_cartesian_sampling((128, 8), acceleration=2)
    >>> [int(shot[0, 0]) for shot in loop]              # LIN labels
    [0, 2, 4, 6]
    >>> loop.to_scales()[loop.shots[0]].round(2).tolist()   # gradient scaling
    [[-1.0]]

    The same object describes the slice loop, in metres:

    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> slices = design.make_slice_loop(4, 5e-3)
    >>> slices[0].ravel().round(4).tolist()
    [-0.0075]
    >>> slices.to_frequencies(42_000.0).round(1).tolist()
    [-315.0, -105.0, 105.0, 315.0]

    ...and the frame loop of a dynamic acquisition, which differs only in
    which counter it names:

    >>> import pulserver.design as design
    >>> from pulserver.design import _lowlevel
    >>> frames = design.make_counter_loop(40, label="REP")
    >>> [event.label for event in frames.labels(7)], frames.labels(7)[0].value
    (['REP'], 7)

    Relative steps for a blipped train, instead of absolute positions:

    >>> from pulserver import ScanLoop
    >>> loop = ScanLoop.from_relative_shifts([[0, 0]], [[0, 0], [2, 1]], shape=(8, 4))
    >>> loop.relative(0)
    array([[0, 0],
           [2, 1]])
    >>> loop.increments(0)
    array([[2, 1]])

    See Also
    --------
    pulserver.SequenceModule : consumes ``lin_idx``/``par_idx``/``rotation``/``freq_offset_hz``.
    pulserver.EncodingAxis : what one position column means.
    """

    positions: np.ndarray
    shots: tuple[np.ndarray, ...]
    mask: np.ndarray | None = None
    axes: tuple[EncodingAxis, ...] | None = None

    def __post_init__(self):
        positions = np.asarray(self.positions)
        if positions.ndim == 0:
            positions = positions.reshape(1, 1)
        elif positions.ndim == 1:
            positions = positions.reshape(-1, 1)
        elif positions.ndim != 2:
            raise ValueError("positions must be scalar, 1D, or 2D")
        if not np.issubdtype(positions.dtype, np.number):
            raise TypeError("positions must be numeric")
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions must contain only finite values")
        positions = _readonly_copy(positions)

        normalized = []
        for shot in tuple(self.shots):
            indices = np.asarray(shot)
            if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
                raise TypeError("each shot must be a one-dimensional integer array")
            indices = _readonly_copy(indices, dtype=np.intp)
            if indices.size and (indices.min() < 0 or indices.max() >= len(positions)):
                raise IndexError("a shot contains an out-of-range position index")
            normalized.append(indices)

        mask = None
        if self.mask is not None:
            mask = _readonly_copy(self.mask, dtype=bool)
            if int(np.count_nonzero(mask)) != len(positions):
                raise ValueError("mask sample count must equal the number of positions")

        if self.axes is None:
            axes = infer_axes(positions)
        else:
            axes = tuple(self.axes)
            if not all(isinstance(axis, EncodingAxis) for axis in axes):
                raise TypeError("axes must be EncodingAxis instances")
            declared = sum(axis.columns for axis in axes)
            if declared != positions.shape[1]:
                raise ValueError(f"axes describe {declared} position column(s), but positions has {positions.shape[1]}")

        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "shots", tuple(normalized))
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "axes", axes)

    # ------------------------------------------------------------------
    # axes and counters
    # ------------------------------------------------------------------
    def with_axes(self, *axes: EncodingAxis) -> ScanLoop:
        """Return the same loop under different axis declarations.

        The positions and the shot order are untouched; only what the columns
        *mean* changes. Use it to retarget a loop's counter — a rotation loop
        that indexes frames rather than views, a parameter loop that belongs
        on ``PHS`` rather than ``SET`` — without rebuilding it.

        Parameters
        ----------
        *axes : EncodingAxis
            One axis per position column, as for the constructor.

        Returns
        -------
        ScanLoop
            A new loop; this one is unchanged.

        Examples
        --------
        >>> import pulserver.design as design
        >>> from pulserver.design import _lowlevel
        >>> import pulserver.pypulseq as pp
        >>> from pulserver.pypulseq import EncodingAxis
        >>> views = design.make_noncartesian_2d_sampling((64, 64), views=8)
        >>> views.with_axes(EncodingAxis("PHS", kind="angle")).labels(3)[0].label
        'PHS'
        """
        return replace(self, axes=tuple(axes))

    def label_value(self, shot: int, *, position: int = 0) -> tuple[int, ...]:
        """Return one counter value per axis for a shot's ``position``-th entry.

        The value is the position *value* on an ``index`` axis (its numbers
        already are the encoding indices) and the position *index* on every
        other kind (angles, metres and inversion times are not counters, but
        their order is).

        Parameters
        ----------
        shot : int
            Shot index.
        position : int, optional
            Which entry within the shot; ``0`` (the default) is the shot's
            own first position, which is the labelled one for a slice, frame
            or SMS group. A readout labels the rest of a train itself.

        Returns
        -------
        tuple of int
            One value per entry in :attr:`axes`.
        """
        entry = int(self.shots[shot][position])
        row = self.positions[entry]
        values, column = [], 0
        for axis in self.axes:
            values.append(round(float(row[column])) if axis.kind == "index" else entry)
            column += axis.columns
        return tuple(values)

    def labels(self, shot: int, *, position: int = 0) -> tuple:
        """Return the ``SET`` label events for one shot, one per axis.

        This is how a loop reaches the reconstruction. Pass the result to
        :meth:`pulserver.SequenceModule.set_labels` — or spread it into
        ``seq.add_block`` — for every loop the readout does *not* consume
        through ``set_state``:

        .. code-block:: python

            excitation.set_labels(*slices.labels(s), *frames.labels(f))

        Emitting them is what makes the ``FIRST_IN_*``/``LAST_IN_*`` MRD flags
        correct. Those flags are not written by the sequence: the interpreter
        computes them by comparing each acquisition's counter against the
        *observed* minimum and maximum of that counter over the scan. A frame
        loop that never emits ``REP`` therefore reports ``min == max == 0``,
        and every acquisition comes back flagged both first and last in
        repetition — which is exactly the failure that makes a dynamic
        reconstruction fire once per line.

        Parameters
        ----------
        shot : int
            Shot index.
        position : int, optional
            Which entry within the shot to label; see :meth:`label_value`.

        Returns
        -------
        tuple
            Label events from :func:`~pulserver.pypulseq.make_label`.

        Examples
        --------
        >>> import pulserver.design as design
        >>> from pulserver.design import _lowlevel
        >>> import pulserver.pypulseq as pp
        >>> slices = design.make_slice_loop(6, 3e-3)
        >>> [(e.label, e.value) for e in slices.labels(1)]
        [('SLC', 2)]

        Every counter a scan varies gets one line in the loop body::

            for f in range(len(frames)):
                for s in range(len(slices)):
                    excitation.set_labels(*frames.labels(f), *slices.labels(s))
                    excitation.add_to(seq)

        See Also
        --------
        label_limits : the ranges those flags are derived from.
        pulserver.SequenceModule.set_labels : where the events go.
        """
        from ...pypulseq._make_label import make_label

        return tuple(
            make_label(label=axis.label, type="SET", value=value)
            for axis, value in zip(self.axes, self.label_value(shot, position=position), strict=True)
        )

    def label_limits(self) -> dict[str, tuple[int, int]]:
        """Return ``{label: (min, max)}`` over every position the loop visits.

        The interpreter derives ``FIRST_IN_<axis>`` and ``LAST_IN_<axis>`` by
        comparing each acquisition's counter against these same limits, so
        this is the loop's own prediction of where those flags will fire. A
        one-element range means both flags fire on every acquisition of the
        axis.

        Returns
        -------
        dict
            One entry per axis; empty when the loop visits no positions.

        Examples
        --------
        >>> import pulserver.design as design
        >>> from pulserver.design import _lowlevel
        >>> import pulserver.pypulseq as pp
        >>> design.make_cartesian_sampling((64, 32), acceleration=2).label_limits()
        {'LIN': (0, 30)}
        >>> import pulserver.design as design
        >>> from pulserver.design import _lowlevel
        >>> design.make_slice_loop(6, 3e-3).label_limits()
        {'SLC': (0, 5)}
        """
        limits: dict[str, tuple[int, int]] = {}
        for shot in range(len(self.shots)):
            for position in range(len(self.shots[shot])):
                for axis, value in zip(self.axes, self.label_value(shot, position=position), strict=True):
                    low, high = limits.get(axis.label, (value, value))
                    limits[axis.label] = (min(low, value), max(high, value))
        return limits

    @property
    def sizes(self) -> tuple[int, ...]:
        """Full encoded extent per axis, defaulting to the position count."""
        return tuple(axis.size if axis.size is not None else self.n_positions for axis in self.axes)

    @classmethod
    def from_mask(
        cls,
        mask,
        *,
        train_length: int = 1,
        ordering: str = "linear",
        seed: int = 0,
        n_sections: int = 3,
    ) -> ScanLoop:
        """Build a k-space loop from a Cartesian mask by ordering its sampled lines.

        The bridge from *which* lines are sampled (a boolean mask) to *when*
        they are acquired (echo trains). Every sampled position is covered
        exactly once — the result is verified before returning — and the
        originating mask is retained for the reconstruction.

        Use ``train_length=1`` for GRE and other acquisitions without an inner
        train; the result is then one shot per sampled line.

        Parameters
        ----------
        mask : array_like of bool
            1D ``(ny,)`` or 2D ``(ny, nz)`` sampling mask.
        train_length : int, optional
            Echo-train / segment length (default 1).
        ordering : {'linear', 'centric', 'radial', 'radial_adaptive', 'shuffling'}, optional
            Echo-train ordering applied to the sampled positions.
        seed : int, optional
            Seed for ``ordering='shuffling'``.
        n_sections : int, optional
            Radial section count for ``ordering='radial_adaptive'``.

        Returns
        -------
        ScanLoop
            ``positions`` of sampled ``(ky[, kz])`` indices, one entry in
            ``shots`` per shot, and the originating ``mask``.

        Raises
        ------
        RuntimeError
            If the chosen ordering does not cover the mask exactly once.

        Examples
        --------
        >>> import pulserver.design as design
        >>> from pulserver.design import _lowlevel
        >>> import pulserver.pypulseq as pp
        >>> from pulserver import ScanLoop
        >>> mask = _lowlevel.make_caipirinha_mask((32, 8), 2, 2, delta=1)
        >>> loop = ScanLoop.from_mask(mask, train_length=8, ordering="radial")
        >>> loop.n_samples == int(mask.sum())
        True
        >>> loop.shot_lengths
        (8, 8, 8, 8, 8, 8, 8, 8)

        Which shot acquires each sampled line, and at which echo of its train:

        .. plot::
           :include-source: false

           import pulserver.design as design
           from pulserver.design import _lowlevel
           import pulserver.pypulseq as pp
           from pulserver import ScanLoop
           from _figures import pattern_figure
           mask = _lowlevel.make_poisson_disc_mask((48, 48), 4.0, calib=(10, 10), seed=0)
           pattern_figure(ScanLoop.from_mask(mask, train_length=16, ordering="radial"),
                          title="Poisson-disc R=4, radial train of 16")

        See Also
        --------
        pulserver.design._lowlevel.make_random_mask : mask generators.
        pulserver.design._lowlevel.make_radial_order : one of the underlying orderings.
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
    def from_relative_shifts(cls, starts, shifts, *, shape) -> ScanLoop:
        """Build a k-space loop from per-shot start points and blip increments.

        The natural description of any blipped train: a shot starts somewhere
        in (ky, kz) and each subsequent echo is reached by a *relative* step.
        Both views the sequence needs come out of the result — ``loop[shot]``
        gives the absolute positions to label, ``loop.increments(shot)`` the
        blip areas to play.

        ``shifts`` may be one shared list of increments, or one list per shot
        when trains differ (EPTI, zigzag).

        Parameters
        ----------
        starts : array_like
            Shape ``(n_shots, D)`` integer start positions.
        shifts : array_like
            Shape ``(inner, D)`` shared by all shots, or
            ``(n_shots, inner, D)``. Entry 0 is normally all-zero so the first
            echo sits on ``starts``.
        shape : tuple of int
            Cartesian grid extent; every position must fall inside it.

        Returns
        -------
        ScanLoop
            Deduplicated ``positions``, one entry in ``shots`` per shot, plus
            the implied Cartesian ``mask``.

        Examples
        --------
        >>> from pulserver import ScanLoop
        >>> loop = ScanLoop.from_relative_shifts(
        ...     starts=[[0, 0], [1, 0]],
        ...     shifts=[[0, 0], [2, 1], [4, 2]],
        ...     shape=(8, 4),
        ... )
        >>> loop[0]
        array([[0, 0],
               [2, 1],
               [4, 2]])
        >>> loop.increments(0)
        array([[2, 1],
               [2, 1]])

        See Also
        --------
        pulserver.design._lowlevel.make_skipped_caipi_order : segmented blipped-CAIPI
            loop built on this representation.
        """
        from .epi import build_from_relative_shifts

        return build_from_relative_shifts(starts, shifts, shape=shape)

    def to_scales(self, shape=None) -> np.ndarray:
        """Convert Cartesian indices into phase-encode gradient scale factors.

        The bridge from a k-space loop to the gradients that realise it. Index
        ``i`` along an axis of extent ``n`` encodes area ``(i - n/2) / fov``,
        and the full-amplitude template
        :func:`~pulserver.design.make_phase_encoding` builds carries
        ``n / (2 * fov)``; the ratio is the ``[-1, 1)`` factor
        :func:`pypulseq.scale_grad` expects. This is exactly the convention
        every shipped readout module uses, so a hand-built readout encodes the
        same k-space as a shipped one.

        The factors depend only on the indices and the matrix size, never on
        which views this loop happens to contain — acceleration and partial
        Fourier drop views, they do not rescale the ones that remain.

        The result is indexed like ``positions``, **not** by k-space index: on
        an accelerated loop ``positions`` holds only the sampled lines, so
        ``scales[ky]`` is a bug waiting to happen. Take one shot's rows with
        ``scales[loop.shots[i]]``, or skip the table entirely and convert the
        shot's own coordinates with
        :func:`~pulserver.design.calc_encoding_scales`.

        Parameters
        ----------
        shape : tuple of int, optional
            Encoded matrix extent per column of ``positions`` — ``(ny,)`` for
            a 2D loop, ``(ny, nz)`` for a 3D one. Defaults to :attr:`sizes`,
            which every shipped Cartesian factory fills in from the matrix it
            was given, so passing it is only needed for a hand-built loop.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(positions), D)`` factors in ``[-1, 1)``.

        Raises
        ------
        ValueError
            If ``shape`` does not match the number of position columns.

        Examples
        --------
        >>> import pulserver.design as design
        >>> from pulserver.design import _lowlevel
        >>> import pulserver.pypulseq as pp
        >>> loop = design.make_cartesian_sampling((64, 8))
        >>> loop.to_scales()[:, 0].round(2).tolist()
        [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]

        Encode a shot by hand, with no readout module involved — converting
        each shot's own coordinates keeps the indexing correct however
        undersampled the loop is::

            gy = design.make_phase_encoding(system, "y", fov_y / ny)
            for shot in loop:
                ky = int(shot[0, 0])
                scale = float(design.calc_encoding_scales(ky, ny))
                seq.add_block(pp.scale_grad(gy, scale), gx_pre)
                seq.add_block(gx, adc, pp.make_label(type="SET", label="LIN", value=ky))

        See Also
        --------
        pulserver.design.calc_encoding_scales : the same conversion for
            bare position arrays.
        pulserver.design.make_phase_encoding : the template these scale.
        """
        from .cartesian import calc_encoding_scales

        if shape is None:
            if any(axis.kind != "index" for axis in self.axes):
                raise ValueError("to_scales needs index-valued positions, or an explicit shape")
            shape = self.sizes
        return calc_encoding_scales(self.positions, shape)

    def to_rotations(self, *, reference=(1.0, 0.0, 0.0)) -> np.ndarray:
        """Convert angle or direction positions into per-view rotation matrices.

        The bridge from a non-Cartesian loop to the sequence: the readout
        designs **one** base waveform along ``reference``, and each position
        replays it under the matrix returned here. A one-column loop is read
        as in-plane spoke angles (rotation about ``z``); a three-column one as
        unit directions, each mapped by the shortest rotation taking
        ``reference`` onto it (Rodrigues). Antipodal directions get a
        well-defined 180-degree rotation rather than a singular matrix.

        Feed the result to :func:`pulserver.pypulseq.make_rotation`, or pass a
        row to a readout module's ``rotation=`` argument.

        Parameters
        ----------
        reference : array_like, optional
            Direction the base waveform was designed along (default ``+x``).
            Ignored for angle-valued positions.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(positions), 3, 3)`` rotation matrices.

        Raises
        ------
        ValueError
            If ``positions`` has neither one nor three columns.

        Examples
        --------
        >>> import numpy as np
        >>> import pulserver.design as design
        >>> from pulserver.design import _lowlevel
        >>> import pulserver.pypulseq as pp
        >>> rotations = _lowlevel.make_golden_means_3d_tilt(4).to_rotations()
        >>> rotations.shape
        (4, 3, 3)
        >>> import pulserver.design as design
        >>> from pulserver.design import _lowlevel
        >>> angles = _lowlevel.make_radial_tilt(4).to_rotations()
        >>> np.round(angles[1] @ np.array([1.0, 0.0, 0.0]), 6)
        array([0.707107, 0.707107, 0.      ])

        Rotate a 3D radial base waveform view by view — the view number is
        also its ``LIN`` label::

            for view, rotation in enumerate(_lowlevel.make_golden_means_3d_tilt(1000).to_rotations()):
                readout.set_state(lin_idx=view, rotation=rotation).add_to(seq)
        """
        from .noncartesian import angles_to_rotations, directions_to_rotations

        kind = self._declared_kind()
        if kind == "angle" or (kind is None and self.positions.shape[1] == 1):
            return angles_to_rotations(self.positions[:, 0])
        if kind == "direction" or (kind is None and self.positions.shape[1] == 3):
            return directions_to_rotations(self.positions, reference=reference)
        raise ValueError("to_rotations needs spoke angles (1 column) or directions (3 columns)")

    def to_frequencies(self, gradient_hz_per_m: float) -> np.ndarray:
        """Convert positions in metres into the RF offsets (Hz) that select them.

        The bridge from a slice loop to the excitation: under a selection
        gradient of ``gradient_hz_per_m``, a spin at position ``z`` resonates
        at ``z * gradient_hz_per_m``. Pass the result to an excitation module
        as ``freq_offset_hz`` — one value for conventional imaging, one per
        band for an SMS multiband pulse.

        The result is indexed like ``positions``, so take one shot's offsets
        with ``offsets[loop.shots[i]]``.

        Parameters
        ----------
        gradient_hz_per_m : float
            Selection gradient amplitude in Pulseq units (Hz/m).

        Returns
        -------
        numpy.ndarray
            Shape ``(len(positions),)`` frequency offsets (Hz).

        Raises
        ------
        ValueError
            If ``positions`` is not one-dimensional, or the gradient is not
            finite.

        Examples
        --------
        >>> from pulserver.design import make_slice_loop
        >>> loop = make_slice_loop(6, 3e-3, sms_factor=2)
        >>> offsets = loop.to_frequencies(42_000.0)
        >>> offsets[loop.shots[0]].round(1).tolist()
        [-315.0, 63.0]

        Drive the slice loop, labelling by physical slice index::

            offsets = slices.to_frequencies(excitation.gradients[0].amplitude)
            for shot in slices.shots:
                excitation.set_state(freq_offset_hz=float(offsets[shot[0]]))
                excitation.set_labels(pp.make_label(type="SET", label="SLC", value=int(shot[0])))
                excitation.add_to(seq)

        See Also
        --------
        pulserver.design.make_slice_loop : builds the loop these offsets select.
        """
        gradient_hz_per_m = float(gradient_hz_per_m)
        if not np.isfinite(gradient_hz_per_m):
            raise ValueError("gradient_hz_per_m must be finite")
        if self._declared_kind() not in (None, "position") or self.positions.shape[1] != 1:
            raise ValueError("to_frequencies needs one-dimensional positions in metres")
        return self.positions[:, 0] * gradient_hz_per_m

    def _declared_kind(self) -> str | None:
        """Return this loop's single declared kind, ``None`` if it is neutral.

        A ``value`` axis carries no conversion semantics of its own, so it
        falls through to the historical column-count dispatch; a kind that is
        declared and wrong makes the converter refuse rather than guess.
        """
        kinds = {axis.kind for axis in self.axes}
        if kinds <= {"value"}:
            return None
        return next(iter(kinds)) if len(kinds) == 1 else "mixed"

    @property
    def n_shots(self) -> int:
        """Number of shots in the loop."""
        return len(self.shots)

    @property
    def n_positions(self) -> int:
        """Number of distinct positions the loop visits."""
        return len(self.positions)

    @property
    def n_samples(self) -> int:
        """Total number of positions visited, summed over shots."""
        return sum(len(shot) for shot in self.shots)

    @property
    def n_dims(self) -> int:
        """Number of position columns (1 for ky, an angle or metres; 2 for ky/kz; 3 for a direction)."""
        return int(self.positions.shape[1])

    @property
    def shot_lengths(self) -> tuple[int, ...]:
        """Number of positions in each shot."""
        return tuple(len(shot) for shot in self.shots)

    def __len__(self) -> int:
        return len(self.shots)

    def __getitem__(self, shot):
        if isinstance(shot, slice):
            return tuple(self.positions[idx] for idx in self.shots[shot])
        return self.positions[self.shots[shot]]

    def flatten(self) -> np.ndarray:
        """Return every visited position, concatenated in acquisition order."""
        if not self.shots:
            return np.empty((0, self.positions.shape[1]), dtype=self.positions.dtype)
        return self.positions[np.concatenate(self.shots)]

    def relative(self, shot: int) -> np.ndarray:
        """Return shot ``shot``'s positions relative to its own first point.

        The prewinder-relative view: what the gradients must reach once the
        shot's start has been played.
        """
        values = self[shot]
        if not len(values):
            return values.copy()
        return values - values[0]

    def increments(self, shot: int) -> np.ndarray:
        """Return the step between consecutive positions of shot ``shot``.

        One entry shorter than the shot: these are the blip areas to play
        between echoes.
        """
        return np.diff(self[shot], axis=0)
