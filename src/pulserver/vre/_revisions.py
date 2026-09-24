"""Revisions of the host bucket, resolved from the header of an incoming stream."""

from __future__ import annotations

__all__ = [
    "REVISION_PARAMETER",
    "SESSION_PARAMETER",
    "Revision",
    "RevisionStore",
]

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..host import SessionKey
from ..mrd._metadata import user_parameter
from ._enrich import SequenceTable

#: Header user parameters naming the design the stream was played from.
SESSION_PARAMETER = "pulserver_session"
REVISION_PARAMETER = "pulserver_revision"

_ENTRY = "sequence.seq"
_RECORD = "meta.json"

#: Revisions a store keeps read.
_KEPT = 8


@dataclass(frozen=True)
class Revision:
    """A generated design, as the reconstruction side reads it.

    Attributes
    ----------
    directory
        ``bucket/<session>/rev/<n>/``.
    recon
        Reconstruction plugin the design names; empty when it names none, which
        leaves the choice to the client's config.
    table
        The readouts of the revision's sequence chain, in play order.
    """

    directory: Path
    recon: str
    table: SequenceTable

    @property
    def session_directory(self) -> Path:
        """``bucket/<session>/``, which the revision and the session's queue share."""
        return self.directory.parent.parent


class RevisionStore:
    """The revisions under ``<base>/bucket``, the most recently read kept.

    Reading a revision tabulates its sequence chain, which is the expensive
    part of accepting a series; concurrent streams of one revision wait for
    the first to finish rather than tabulating it again. A revision that is
    no longer among the most recently read is read again when next named.
    """

    def __init__(self, base: Path | str) -> None:
        self.bucket = Path(base) / "bucket"
        self._lock = threading.Lock()
        self._building: dict[Path, threading.Lock] = {}
        self._revisions: OrderedDict[Path, Revision] = OrderedDict()

    def locate(self, header: Any) -> Path:
        """Return the revision directory a header names.

        The session is a ``userParameterString``, ``<pid>-<day>``, and the
        revision a ``userParameterLong``, or a string holding an integer.

        Raises
        ------
        ValueError
            If the header carries no session or revision, or names them
            unreadably.
        FileNotFoundError
            If no such revision exists under the bucket.
        """
        session = user_parameter(header, SESSION_PARAMETER)
        revision = user_parameter(header, REVISION_PARAMETER)
        if session in (None, "") or revision in (None, ""):
            raise ValueError(
                f"the header carries no {SESSION_PARAMETER} and {REVISION_PARAMETER}"
            )
        try:
            key = SessionKey.parse(str(session))
            number = int(str(revision))
        except ValueError as error:
            raise ValueError(
                f"{SESSION_PARAMETER}={session!r} {REVISION_PARAMETER}={revision!r} "
                "is not a session and a revision"
            ) from error
        directory = self.bucket / str(key) / "rev" / str(number)
        if not directory.is_dir():
            raise FileNotFoundError(f"no revision {number} of session {key}")
        return directory

    def read(self, directory: Path) -> Revision:
        """Return a revision directory read, tabulating its chain unless kept."""
        directory = Path(directory)
        with self._lock:
            building = self._building.setdefault(directory, threading.Lock())
        with building:
            with self._lock:
                if directory in self._revisions:
                    self._revisions.move_to_end(directory)
                    return self._revisions[directory]
            revision = Revision(
                directory=directory,
                recon=str(_meta(directory).get("recon", "")),
                table=SequenceTable.read(directory / _ENTRY),
            )
            with self._lock:
                self._revisions[directory] = revision
                while len(self._revisions) > _KEPT:
                    forgotten, _ = self._revisions.popitem(last=False)
                    self._building.pop(forgotten, None)
            return revision

    def resolve(self, header: Any) -> Revision:
        """Return the revision a header names; see :meth:`locate` and :meth:`read`."""
        return self.read(self.locate(header))


def _meta(directory: Path) -> dict[str, Any]:
    path = directory / _RECORD
    return json.loads(path.read_text()) if path.is_file() else {}
