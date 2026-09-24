"""Session buckets: protocol, generated revisions and the revision to play."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_FLOAT32_EXACT = 2**24
_RECORD = "session.json"


@dataclass(frozen=True, order=True)
class SessionKey:
    """A host PSD process: its PID and the day it started.

    ``day`` counts days since 1970-01-01. Both fields must be representable
    exactly in a float32 raw-header slot.

    Raises
    ------
    ValueError
        If either field is negative or not below 2**24.
    """

    pid: int
    day: int

    def __post_init__(self) -> None:
        for name in ("pid", "day"):
            value = getattr(self, name)
            if not 0 <= value < _FLOAT32_EXACT:
                raise ValueError(f"{name}={value} does not fit a float32 slot")

    def __str__(self) -> str:
        return f"{self.pid}-{self.day}"

    @classmethod
    def parse(cls, text: str) -> SessionKey:
        """Read a key written as ``<pid>-<day>``."""
        pid, day = text.split("-")
        return cls(int(pid), int(day))


def revision_hash(
    plugin: str,
    limits: Mapping[str, Any],
    values: Mapping[str, Any],
    source: str = "",
) -> str:
    """Identify a design by plugin, scanner limits, resolved protocol and source.

    ``source`` is a digest of the code that designs, so that a plugin or
    package change yields a new revision for an unchanged protocol; empty for
    an imported revision, which ``values`` identifies by file contents.
    Independent of mapping order.
    """
    canonical = json.dumps(
        {
            "plugin": plugin,
            "limits": dict(limits),
            "values": dict(values),
            "source": source,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _write_atomic(path: Path, text: str) -> None:
    staged = path.with_name(f".{path.name}.tmp")
    staged.write_text(text)
    staged.replace(path)


class Session:
    """One session's state, persisted in ``bucket/<key>/``.

    ``rev/<n>/`` directories are immutable once committed; ``current`` is a
    relative symlink to the revision the target plays.
    """

    def __init__(self, directory: Path, record: dict[str, Any]) -> None:
        self.directory = directory
        self._record = record

    @property
    def key(self) -> SessionKey:
        return SessionKey.parse(self._record["key"])

    @property
    def plugin(self) -> str:
        return self._record["plugin"]

    @property
    def limits(self) -> dict[str, Any]:
        """``pypulseqpp.Opts`` keyword arguments the session was opened with."""
        return dict(self._record["limits"])

    @property
    def current(self) -> int | None:
        """Revision the target plays; ``None`` before the first generated one."""
        return self._record["current"]

    @property
    def closed(self) -> bool:
        """Whether the host process sent ``CLOSE``."""
        return self._record["closed"]

    def revision_directory(self, revision: int) -> Path:
        return self.directory / "rev" / str(revision)

    def find(self, digest: str) -> int | None:
        """Return the revision generated for a hash, ``None`` if there is none."""
        return self._record["revisions"].get(digest)

    def stage(self) -> Path:
        """Create an empty directory to generate a revision into."""
        rev = self.directory / "rev"
        rev.mkdir(exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=".stage-", dir=rev))

    def commit(self, digest: str, staged: Path) -> int:
        """Publish a staged directory as the next revision and make it current."""
        revision = max(self._record["revisions"].values(), default=0) + 1
        staged.replace(self.revision_directory(revision))
        self._record["revisions"][digest] = revision
        self.select(revision)
        return revision

    @staticmethod
    def discard(staged: Path) -> None:
        shutil.rmtree(staged, ignore_errors=True)

    def select(self, revision: int) -> None:
        """Point ``current`` at a committed revision."""
        link = self.directory / "current"
        staged = self.directory / ".current.tmp"
        staged.unlink(missing_ok=True)
        staged.symlink_to(Path("rev") / str(revision))
        staged.replace(link)
        self._record["current"] = revision
        self._save()

    def save_protocol(self, text: str) -> None:
        """Store the latest ``VALIDATE`` request block."""
        _write_atomic(self.directory / "protocol", text)

    def close(self) -> None:
        self._record["closed"] = True
        self._save()

    def _save(self) -> None:
        _write_atomic(self.directory / _RECORD, json.dumps(self._record, indent=2))


class SessionStore:
    """All sessions under ``<base>/bucket``, loaded from disk on first use."""

    def __init__(self, base: Path) -> None:
        self.bucket = Path(base) / "bucket"
        self.bucket.mkdir(parents=True, exist_ok=True)
        self._sessions: dict[SessionKey, Session] = {}

    def open(self, key: SessionKey, plugin: str, limits: Mapping[str, Any]) -> Session:
        """Create a session, or return it when it was opened with the same plugin and limits.

        Raises
        ------
        ValueError
            If the session exists with another plugin or other limits.
        """
        try:
            session = self.get(key)
        except KeyError:
            directory = self.bucket / str(key)
            directory.mkdir()
            record = {
                "key": str(key),
                "plugin": plugin,
                "limits": dict(limits),
                "revisions": {},
                "current": None,
                "closed": False,
            }
            session = Session(directory, record)
            session._save()
            self._sessions[key] = session
            return session
        if session.plugin != plugin or session.limits != dict(limits):
            raise ValueError(f"session {key} is open with another plugin or limits")
        return session

    def get(self, key: SessionKey) -> Session:
        """Return a session, loading it from disk on first use.

        Raises
        ------
        KeyError
            If no session with this key exists on disk.
        """
        if key not in self._sessions:
            path = self.bucket / str(key) / _RECORD
            if not path.is_file():
                raise KeyError(str(key))
            self._sessions[key] = Session(path.parent, json.loads(path.read_text()))
        return self._sessions[key]

    def __iter__(self) -> Iterator[Session]:
        for path in sorted(self.bucket.glob(f"*/{_RECORD}")):
            yield self.get(SessionKey.parse(path.parent.name))
