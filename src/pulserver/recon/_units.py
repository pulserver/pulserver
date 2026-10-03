"""What identifies a reconstruction unit, and what closes it."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Protocol

from ..mrd._acquisitions import AcquisitionFlag
from ..mrd._header import LOOP_COUNTERS
from ..mrd._metadata import acquisition_label, has_acquisition_flag

if TYPE_CHECKING:
    from ._buffers import ReconUnit

#: ``(branch, encoding space, ((counter, value), ...))``: the counters are those
#: of :data:`~pulserver.mrd.LOOP_COUNTERS` that are not axes of the unit.
UnitKey = tuple[str, int, tuple[tuple[str, int], ...]]


def unit_key(
    branch: str, acquisition: Any, axes: Iterable[str], merge: Iterable[str] = ()
) -> UnitKey:
    """Return the key of the unit an acquisition of ``branch`` belongs to.

    ``segment`` and the user counters are not part of a key, so they never
    separate units; a counter listed in ``axes`` is placed along an axis of
    the unit, and one listed in ``merge`` along none, and neither separates
    them either.
    """
    apart = {*axes, *merge}
    counters = tuple(
        (name, int(acquisition_label(acquisition, name, 0) or 0))
        for name in LOOP_COUNTERS
        if name not in apart
    )
    space = int(acquisition_label(acquisition, "encoding_space_ref", 0) or 0)
    return branch, space, counters


def carries(acquisition: Any, flag: Any) -> bool:
    """Whether an acquisition carries ``flag``, or any member of a combined :class:`~pulserver.mrd.AcquisitionFlag`."""
    if isinstance(flag, AcquisitionFlag):
        return any(
            has_acquisition_flag(acquisition, member.flag)
            for member in AcquisitionFlag
            if member in flag
        )
    return has_acquisition_flag(acquisition, flag)


class _Closure(Protocol):
    """The rule that decides which units are complete."""

    def closed(
        self,
        units: Mapping[UnitKey, ReconUnit],
        acquisition: Any,
        branch: str,
    ) -> list[UnitKey]:
        """Return the keys of the units to reconstruct now.

        Called once for each acquisition a unit has taken in, after it was
        added to its unit, with every unit still open. Not called for an
        acquisition flagged ``LAST_IN_MEASUREMENT``, which closes every unit.
        """
        ...


class _FlagClosure:
    """Closes a unit when its branch's flag has arrived at every position along its axes and merged counters.

    A position is one combination of the counters in ``axes`` and ``merge``;
    with neither, the flag arriving once closes the unit. How many positions
    there are comes from the unit's encoding space.
    """

    def __init__(
        self,
        triggers: Mapping[str, Any],
        axes: Iterable[str],
        merge: Iterable[str] = (),
    ) -> None:
        self.triggers = triggers
        self.axes = tuple(axes)
        self.merge = tuple(merge)
        self._flagged: dict[UnitKey, set[tuple[int, ...]]] = {}

    def closed(
        self,
        units: Mapping[UnitKey, ReconUnit],
        acquisition: Any,
        branch: str,
    ) -> list[UnitKey]:
        flag = self.triggers.get(branch)
        if flag is None or not carries(acquisition, flag):
            return []
        key = unit_key(branch, acquisition, self.axes, self.merge)
        unit = units.get(key)
        if unit is None:
            return []
        flagged = self._flagged.setdefault(key, set())
        flagged.add(
            tuple(
                int(acquisition_label(acquisition, counter, 0) or 0)
                for counter in (*self.axes, *self.merge)
            )
        )
        if len(flagged) < unit.combinations:
            return []
        del self._flagged[key]
        return [key]
