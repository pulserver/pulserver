"""What each column of a scan loop's position table *means*."""

from __future__ import annotations

__all__ = ["EncodingAxis", "KINDS", "infer_axes"]

from dataclasses import dataclass

import numpy as np

from ...pypulseq._make_label import COUNTER_LABELS

#: Position semantics an axis can declare. The kind fixes two things: which
#: ``ScanLoop.to_*`` converter is legal, and whether the counter label is the
#: position *value* or the position *index*.
#:
#: ============= ======== ============================ =====================
#: kind          columns  positions hold               label value is
#: ============= ======== ============================ =====================
#: ``index``     1        integer Cartesian indices     the position value
#: ``angle``     1        in-plane rotation (rad)       the position index
#: ``direction`` 3        unit direction vectors        the position index
#: ``position``  1        physical location (m)         the position index
#: ``value``     1        any physical parameter        the position index
#: ============= ======== ============================ =====================
#:
#: ``value`` is the neutral kind: a frame number, an inversion time, a
#: b-value, a saturation offset. It carries no conversion of its own because
#: what the number drives — a delay, an amplitude, a frequency — is the
#: sequence's business, not the loop's.
KINDS = {"index": 1, "angle": 1, "direction": 3, "position": 1, "value": 1}

#: Labels the default inference assigns, in column order.
_DEFAULT_LABELS = ("LIN", "PAR")


@dataclass(frozen=True)
class EncodingAxis:
    """One named dimension of a :class:`~pulserver.ScanLoop`.

    A scan loop is a table of numbers; an axis says what one column of that
    table *is*. That is the whole difference between a loop that can only
    describe k-space and one that can equally describe frames, contrasts,
    averages or inversion times — the numbers were always general, only their
    interpretation was hard-coded.

    Declaring the axis buys three things:

    - **A counter label.** :meth:`~pulserver.ScanLoop.labels` emits
      ``SET <label>`` per shot, so the reconstruction sorts the data and the
      interpreter derives the ``FIRST_IN_*``/``LAST_IN_*`` MRD flags from the
      observed range. An unlabelled loop is invisible downstream.
    - **A conversion.** ``kind`` selects which of ``to_scales`` /
      ``to_rotations`` / ``to_frequencies`` applies, instead of guessing from
      the column count.
    - **An extent.** ``size`` is the full encoded extent, which is what
      :meth:`~pulserver.ScanLoop.to_scales` needs and what an accelerated loop
      cannot recover from its own (sparse) positions.

    Parameters
    ----------
    label : str
        Counter label emitted for this axis — any of
        :data:`~pulserver.pypulseq.COUNTER_LABELS`, or a custom string, which
        passes through unvalidated exactly as in
        :func:`~pulserver.pypulseq.make_label`.
    kind : str, optional
        Position semantics; one of :data:`KINDS` (default ``'index'``).
    size : int, optional
        Full encoded extent. Defaults to the number of distinct positions,
        which is only the same thing on a fully sampled ``index`` axis.

    Examples
    --------
    >>> from pulserver import EncodingAxis
    >>> EncodingAxis("LIN", size=128).columns
    1
    >>> EncodingAxis("LIN", kind="direction").columns
    3

    A frame axis of a dynamic acquisition is an ordinary axis:

    >>> EncodingAxis("REP", kind="value", size=40).label
    'REP'

    See Also
    --------
    pulserver.ScanLoop : the table these annotate.
    pulserver.design.make_counter_loop : builds a loop over one such axis.
    """

    label: str
    kind: str = "index"
    size: int | None = None

    def __post_init__(self):
        if not isinstance(self.label, str) or not self.label:
            raise TypeError("an encoding axis label must be a non-empty string")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {tuple(KINDS)}, got {self.kind!r}")
        if self.size is not None:
            size = int(self.size)
            if size <= 0:
                raise ValueError("size must be positive")
            object.__setattr__(self, "size", size)

    @property
    def columns(self) -> int:
        """Number of position columns this axis occupies."""
        return KINDS[self.kind]

    @property
    def is_counter(self) -> bool:
        """Whether ``label`` is one of the ten built-in Pulseq counters."""
        return self.label in COUNTER_LABELS


def infer_axes(positions: np.ndarray) -> tuple[EncodingAxis, ...]:
    """Return the axes a bare ``positions`` table implies.

    Integer columns are Cartesian encoding indices (``LIN``, then ``PAR``);
    anything else is a neutral ``value`` axis, which labels by position index
    and leaves every converter available through the legacy column-count
    dispatch. Three float columns are a single ``direction`` axis, matching
    what the 3D projection factories produce.
    """
    columns = int(positions.shape[1])
    integral = np.issubdtype(positions.dtype, np.integer)
    if not integral and columns == 3:
        return (EncodingAxis("LIN", kind="direction"),)
    kind = "index" if integral else "value"
    return tuple(
        EncodingAxis(
            _DEFAULT_LABELS[column] if column < len(_DEFAULT_LABELS) else f"AXIS{column}",
            kind=kind,
        )
        for column in range(columns)
    )
