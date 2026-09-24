"""Exam cache generations: one current exam, retired when a header names another."""

from __future__ import annotations

__all__ = ["ExamCacheManager", "resolve_exam_id"]

import contextlib
import hashlib
import itertools
import shutil
from collections.abc import Callable, Hashable, Iterator
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from ...mrd._metadata import user_parameter
from ..plugin import ExamCache


@dataclass
class _Generation:
    cache: ExamCache
    leases: int = 0
    retired: bool = False


class ExamCacheManager:
    """Holds the current exam's cache and retires it without breaking its users.

    Parameters
    ----------
    resolver
        Returns a header's exam identifier, or ``None``; :func:`resolve_exam_id`
        by default. A header without one gets a cache of its own that is never
        shared.
    directory
        Where each exam's cache gets a directory of its own (see
        :class:`~pulserver.recon.ExamCache`), removed when the cache closes;
        ``None`` keeps caches in memory.
    """

    def __init__(
        self,
        resolver: Callable[[Any], Hashable | None] | None = None,
        *,
        directory: Path | str | None = None,
    ) -> None:
        self._resolver = resolve_exam_id if resolver is None else resolver
        self._directory = None if directory is None else Path(directory)
        self._lock = RLock()
        self._current: _Generation | None = None
        self._anonymous_ids = itertools.count()
        self._closed = False

    @property
    def current_exam_id(self) -> Hashable | None:
        """Identifier of the current exam, or ``None`` before the first lease."""
        with self._lock:
            if self._current is None:
                return None
            return self._current.cache.exam_id

    @contextlib.contextmanager
    def lease(self, header: Any) -> Iterator[ExamCache]:
        """Yield the cache of the exam ``header`` names, for one reconstruction.

        A header naming a different exam retires the current cache, which closes
        when its last lease ends.

        Raises
        ------
        RuntimeError
            After :meth:`close`.
        """
        close_now: ExamCache | None = None
        with self._lock:
            if self._closed:
                raise RuntimeError("exam cache manager is closed")
            exam_id = self._resolver(header)
            if exam_id is None:
                exam_id = ("connection", next(self._anonymous_ids))

            if self._current is None or self._current.cache.exam_id != exam_id:
                previous = self._current
                self._current = _Generation(
                    ExamCache(exam_id, self._exam_directory(exam_id))
                )
                if previous is not None:
                    previous.retired = True
                    if previous.leases == 0:
                        close_now = previous.cache

            generation = self._current
            generation.leases += 1

        if close_now is not None:
            _discard(close_now)

        try:
            yield generation.cache
        finally:
            self._release(generation)

    def close(self) -> None:
        """Retire the current cache and refuse further leases."""
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
            _discard(close_now)

    def _release(self, generation: _Generation) -> None:
        close_now = False
        with self._lock:
            generation.leases -= 1
            if generation.leases < 0:
                raise RuntimeError("exam cache generation lease underflow")
            close_now = generation.retired and generation.leases == 0
        if close_now:
            _discard(generation.cache)

    def _exam_directory(self, exam_id: Hashable) -> Path | None:
        if self._directory is None:
            return None
        return self._directory / hashlib.sha256(repr(exam_id).encode()).hexdigest()


def _discard(cache: ExamCache) -> None:
    cache.close()
    if cache.directory is not None:
        shutil.rmtree(cache.directory, ignore_errors=True)


def resolve_exam_id(header: Any) -> Hashable | None:
    """Return a header's exam identity as a ``(source, value)`` tuple, or ``None``.

    Tries the ``ExamID``, ``exam_id`` and ``examID`` user parameters, then the
    study's ``studyInstanceUID``, then its ``studyID``. ``measurementID`` is not
    used: it changes between the series of one exam.
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
