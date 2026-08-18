"""Private grouping and filtering helpers for MRD acquisition streams.

They preserve input order and accept any acquisition-like object, making them
safe to use in a Gadgetron handler before data is converted to a framework
specific container.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator
from typing import Any

from .metadata import acquisition_label, has_acquisition_flag

__all__ = ["filter_acquisitions", "group_by_labels", "split_on_flag"]


def group_by_labels(
    acquisitions: Iterable[Any], labels: str | tuple[str, ...]
) -> OrderedDict[Any, list[Any]]:
    """Group acquisitions by one or more MRD labels, preserving order.

    Parameters
    ----------
    acquisitions : iterable
        Acquisition stream or collection.
    labels : str or tuple[str, ...]
        ``idx`` labels such as ``("repetition", "slice")``.  Header labels
        such as ``"encoding_space_ref"`` are also supported.

    Returns
    -------
    collections.OrderedDict
        A mapping from a scalar key (one label) or tuple key (many labels) to
        the corresponding acquisition list.
    """
    names = (labels,) if isinstance(labels, str) else labels
    if not names:
        raise ValueError("labels must not be empty")
    groups: OrderedDict[Any, list[Any]] = OrderedDict()
    for acquisition in acquisitions:
        values = tuple(acquisition_label(acquisition, name) for name in names)
        key: Any = values[0] if len(values) == 1 else values
        groups.setdefault(key, []).append(acquisition)
    return groups


def split_on_flag(acquisitions: Iterable[Any], flag: int | str) -> Iterator[list[Any]]:
    """Yield ordered acquisition groups ending at each occurrence of ``flag``.

    The terminating acquisition remains in its group.  A final unterminated
    group is yielded, which is useful for replaying incomplete saved data.
    """
    group: list[Any] = []
    for acquisition in acquisitions:
        group.append(acquisition)
        if has_acquisition_flag(acquisition, flag):
            yield group
            group = []
    if group:
        yield group


def filter_acquisitions(
    acquisitions: Iterable[Any],
    *,
    require_flags: Iterable[int | str] = (),
    reject_flags: Iterable[int | str] = (),
) -> Iterator[Any]:
    """Yield acquisitions satisfying all required and no rejected flags."""
    required = tuple(require_flags)
    rejected = tuple(reject_flags)
    for acquisition in acquisitions:
        if all(
            has_acquisition_flag(acquisition, flag) for flag in required
        ) and not any(has_acquisition_flag(acquisition, flag) for flag in rejected):
            yield acquisition
