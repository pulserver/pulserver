"""Small helpers shared by the sequence and the views built on it."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ._results import RF_USES

#: Upstream's default gradient-spectrum window, in seconds. Needed only to
#: tell a caller who chose a window from one who left it alone, since under
#: ``tr`` the window is the repetition time and nothing else.
_UPSTREAM_WINDOW_WIDTH = 0.05


def _span(time_range) -> tuple[float, float]:
    """``time_range`` as a pair of seconds, checked the way upstream checks it."""
    bounds = tuple(time_range)
    if len(bounds) != 2:
        raise ValueError("time_range must hold two elements")
    start, stop = float(bounds[0]), float(bounds[1])
    if start > stop:
        raise ValueError("time_range must end after it begins")
    return start, stop


def _per_axis(value, name: str) -> list[float]:
    """One number or three, always as three.

    Upstream takes either for ``trajectory_delay`` and ``gradient_offset``,
    and the C core takes three, so the widening happens once here rather than
    at each call site.
    """
    array = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
    if array.size == 1:
        return [float(array[0])] * 3
    if array.size == 3:
        return [float(v) for v in array]
    raise ValueError(f"{name} must be one value or three, got {array.size}")


#: The file format's trailing use character, as the name it stands for. A
#: block read back from the block table carries the character; one built by a
#: ``make_*`` call carries the name.
_USE_NAMES = dict(zip("erispo", RF_USES, strict=False))


def _rf_use(rf: SimpleNamespace) -> str:
    """One decoded RF pulse's use tag, as one of ``RF_USES``.

    A row with no use character reads back as ``"undefined"``, which is a
    distinct thing from ``"other"``: ``"other"`` was chosen by whoever wrote
    the sequence, ``"undefined"`` means nobody said. Upstream folds the latter
    in with excitation, and :meth:`Sequence.rf_times` reproduces that under
    ``compat=True`` -- but the tag itself is kept, so a caller who wants to
    know can find out.
    """
    use = getattr(rf, "use", None)
    if use in RF_USES:
        return use
    return _USE_NAMES.get(use, "undefined")
