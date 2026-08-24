"""Private generation-safe ownership of exam reconstruction artifacts."""

from __future__ import annotations

__all__ = ["ExamCacheManager", "resolve_exam_id"]

import contextlib
import itertools
from collections.abc import Callable, Hashable, Iterator
from dataclasses import dataclass
from threading import RLock
from typing import Any

from ..plugin import ExamCache
from ...mrd._metadata import user_parameter


@dataclass
class _Generation:
    cache: ExamCache
    leases: int = 0
    retired: bool = False


class ExamCacheManager:
    """Retain one current exam cache without invalidating active users.

    A header carrying a different exam identifier atomically retires the
    current generation. Its artifacts are released immediately when unused,
    or after the final reconstruction that already leased it completes.

    Parameters
    ----------
    resolver
        Function extracting a stable exam identifier from a parsed header.
        Headers without one receive a connection-local generation and are
        never accidentally shared.
    """

    def __init__(
        self,
        resolver: Callable[[Any], Hashable | None] | None = None,
    ) -> None:
        self._resolver = resolve_exam_id if resolver is None else resolver
        self._lock = RLock()
        self._current: _Generation | None = None
        self._anonymous_ids = itertools.count()
        self._closed = False

    @property
    def current_exam_id(self) -> Hashable | None:
        """Return the current exam identifier, if a header has been observed."""
        with self._lock:
            if self._current is None:
                return None
            return self._current.cache.exam_id

    @contextlib.contextmanager
    def lease(self, header: Any) -> Iterator[ExamCache]:
        """Lease the generation selected by ``header`` for one reconstruction."""
        close_now: ExamCache | None = None
        with self._lock:
            if self._closed:
                raise RuntimeError("exam cache manager is closed")
            exam_id = self._resolver(header)
            if exam_id is None:
                exam_id = ("connection", next(self._anonymous_ids))

            if self._current is None or self._current.cache.exam_id != exam_id:
                previous = self._current
                self._current = _Generation(ExamCache(exam_id))
                if previous is not None:
                    previous.retired = True
                    if previous.leases == 0:
                        close_now = previous.cache

            generation = self._current
            generation.leases += 1

        if close_now is not None:
            close_now.close()

        try:
            yield generation.cache
        finally:
            self._release(generation)

    def close(self) -> None:
        """Retire the current generation and prevent new leases."""
        close_now: ExamCache | None = None
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._current is not None:
                self._current.retired = True
                if self._current.leases == 0:
                    close_now = self._current.cache
                self._current = None
        if close_now is not None:
            close_now.close()

    def _release(self, generation: _Generation) -> None:
        close_now = False
        with self._lock:
            generation.leases -= 1
            if generation.leases < 0:
                raise RuntimeError("exam cache generation lease underflow")
            close_now = generation.retired and generation.leases == 0
        if close_now:
            generation.cache.close()


def resolve_exam_id(header: Any) -> Hashable | None:
    """Extract an exam identity without confusing it with a measurement.

    Vendor exporters commonly place ``ExamID`` in the typed user parameters.
    Standard MRD study identity is used as a fallback. ``measurementID`` is
    deliberately excluded because it normally changes between sequences in
    the same exam.
    """
    for name in ("ExamID", "exam_id", "examID"):
        value = user_parameter(header, name)
        if value not in (None, ""):
            return ("exam", str(value))

    study = getattr(header, "studyInformation", None)
    if study is None:
        return None
    instance_uid = getattr(study, "studyInstanceUID", None)
    if instance_uid not in (None, ""):
        return ("study-instance", str(instance_uid))
    study_id = getattr(study, "studyID", None)
    if study_id not in (None, ""):
        return ("study", str(study_id))
    return None
