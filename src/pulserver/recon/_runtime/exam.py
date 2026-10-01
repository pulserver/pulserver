"""Exam cache generations: one current exam, retired when a header names another."""

from __future__ import annotations

__all__ = [
    "DEFAULT_EXAM_DIRECTORY",
    "EXAM_DIRECTORY_MODE",
    "EXAM_IDLE",
    "ExamCacheManager",
    "resolve_exam_id",
]

import contextlib
import fcntl
import hashlib
import itertools
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Hashable, Iterator
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from ...mrd._metadata import user_parameter
from ..plugin import ExamCache

#: Where the proxies of a host keep the caches of the exams they reconstruct.
#: A reconstruction computer runs one proxy per acquisition, and a map measured
#: by a series on one is wanted by a series on another, so the location is the
#: host's rather than each process's own.
#:
#: Named for the user the proxies run as, and created so only that user can
#: open it: an exam's files are unpickled, so writing one is running code in a
#: reconstruction worker. A name another user could take first would be a name
#: another user could put a pickle under.
DEFAULT_EXAM_DIRECTORY = Path(tempfile.gettempdir()) / f"pulserver-exams-{os.getuid()}"

#: What an exam root is created as: the owner alone, for the reason above.
EXAM_DIRECTORY_MODE = 0o700

#: Seconds an exam nobody leases is kept before :meth:`ExamCacheManager.expire`
#: ends it. No message says an exam is over, and this is longer than the gap
#: between two series of one exam.
EXAM_IDLE = 2 * 3600.0


@dataclass
class _Generation:
    cache: ExamCache
    leases: int = 0
    retired: bool = False
    #: Descriptor whose shared lock states that this process is on this exam,
    #: so another proxy does not delete the directory under it. -1 in memory.
    held: int = -1
    #: When its last lease ended, as ``time.time()``.
    idle_since: float = 0.0


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
        close_now: _Generation | None = None
        with self._lock:
            if self._closed:
                raise RuntimeError("exam cache manager is closed")
            exam_id = self._resolver(header)
            if exam_id is None:
                exam_id = ("connection", next(self._anonymous_ids))

            if self._current is None or self._current.cache.exam_id != exam_id:
                previous = self._current
                directory = self._exam_directory(exam_id)
                self._current = _Generation(
                    ExamCache(exam_id, directory), held=self._hold(exam_id)
                )
                if previous is not None:
                    previous.retired = True
                    if previous.leases == 0:
                        close_now = previous

            generation = self._current
            generation.leases += 1

        if close_now is not None:
            _discard(close_now.cache, close_now.held)

        try:
            yield generation.cache
        finally:
            self._release(generation)

    def close(self) -> None:
        """Retire the current cache and refuse further leases."""
        close_now: _Generation | None = None
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._current is not None:
                self._current.retired = True
                if self._current.leases == 0:
                    close_now = self._current
                self._current = None
        if close_now is not None:
            _discard(close_now.cache, close_now.held)

    def expire(self, idle: float = EXAM_IDLE, now: float | None = None) -> None:
        """End every exam nothing has leased for ``idle`` seconds.

        The current exam is retired when none of its leases has been open for
        that long, and every exam under the directory no proxy is on is
        removed once it is as old: those of a proxy that died with them. An
        exam another proxy is on, idle or not, is left to that proxy.
        """
        now = time.time() if now is None else now
        close_now: _Generation | None = None
        with self._lock:
            current = self._current
            if (
                current is not None
                and current.leases == 0
                and now - current.idle_since >= idle
            ):
                current.retired = True
                self._current = None
                close_now = current
        if close_now is not None:
            _discard(close_now.cache, close_now.held)
        if self._directory is not None and self._directory.is_dir():
            for lock in self._directory.glob("*.lock"):
                directory = lock.with_name(lock.name[: -len(".lock")])
                with contextlib.suppress(OSError):
                    if (
                        max(p.stat().st_mtime for p in (lock, directory) if p.exists())
                        < now - idle
                    ):
                        _remove_unheld(directory)

    def _release(self, generation: _Generation) -> None:
        close_now = False
        with self._lock:
            generation.leases -= 1
            if generation.leases < 0:
                raise RuntimeError("exam cache generation lease underflow")
            if generation.leases == 0:
                generation.idle_since = time.time()
            close_now = generation.retired and generation.leases == 0
        if close_now:
            _discard(generation.cache, generation.held)

    def _exam_directory(self, exam_id: Hashable) -> Path | None:
        if self._directory is None:
            return None
        return self._directory / hashlib.sha256(repr(exam_id).encode()).hexdigest()

    def _hold(self, exam_id: Hashable) -> int:
        """Take the shared lock stating that this process is on this exam.

        A reader of the same exam in another process takes it too, and a
        descriptor is the kernel's, so a proxy that dies stops holding the exam
        without anything having to notice. -1 where caches are in memory, which
        no other process can read anyway.
        """
        directory = self._exam_directory(exam_id)
        if directory is None:
            return -1
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            fd = os.open(_lock_path(directory), os.O_CREAT | os.O_RDWR, 0o666)
        except OSError:
            return -1
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
        except OSError:
            os.close(fd)
            return -1
        return fd


def _discard(cache: ExamCache, held: int = -1) -> None:
    """Close a cache, and remove its directory where no other proxy is on it."""
    cache.close()
    if held >= 0:
        with contextlib.suppress(OSError):
            fcntl.flock(held, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(held)
    if cache.directory is None:
        return
    if held < 0:
        shutil.rmtree(cache.directory, ignore_errors=True)
        return
    _remove_unheld(cache.directory)


def _remove_unheld(directory: Path) -> None:
    """Remove an exam's directory and its lock unless a proxy is on the exam."""
    lock = _lock_path(directory)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o666)
    except OSError:
        return
    try:
        # Nothing else has this exam open, so its artifacts are nobody's.
        # Another proxy may be leasing it between this and the unlink, and
        # then finds the directory gone: a cache miss, not a wrong answer.
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return
    shutil.rmtree(directory, ignore_errors=True)
    with contextlib.suppress(OSError):
        lock.unlink()
    with contextlib.suppress(OSError):
        os.close(fd)


def _lock_path(directory: Path) -> Path:
    """Return the lock beside an exam's directory, so removing one leaves the other."""
    return directory.with_name(directory.name + ".lock")


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
