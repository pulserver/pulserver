"""What the calls of a warm design server keep for the calls after them.

A design call is a function of its inputs, and the server answers each in a
child that ends with it, so nothing a call works out in memory reaches the
next. What is worth keeping -- a reply, the RF a plugin lists, the slew rate a
request was lowered to, the effective TE of a solved echo train -- is written here, in a directory the server makes
when it starts serving and removes when it stops: every child reads what the
children before it wrote, and nothing outlives the server, which is restarted
to take up a changed package anyway. A process no server forked keeps
nothing here: :func:`read` finds nothing and :func:`write` does nothing.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

#: Entries of one kind kept before the least recently written are removed.
SIZE = 4096

_root: Path | None = None


def start() -> Path:
    """Make the directory the calls of this process and of its children keep their entries in; return it."""
    global _root
    _root = Path(tempfile.mkdtemp(prefix="pulserver-design-"))
    return _root


def stop() -> None:
    """Remove the directory :func:`start` made, and keep nothing from here on."""
    global _root
    if _root is not None:
        shutil.rmtree(_root, ignore_errors=True)
    _root = None


def kept() -> bool:
    """Whether entries are kept: a server started this process, or the one it was forked from."""
    return _root is not None


def _path(kind: str, key: str) -> Path:
    return _root / kind / f"{hashlib.sha256(key.encode()).hexdigest()}.json"


def read(kind: str, key: str) -> tuple[bool, Any]:
    """Return whether an entry of ``kind`` is kept under ``key``, and its value.

    An entry that cannot be read, one another call is still writing among
    them, is not found.
    """
    if _root is None:
        return False, None
    try:
        entry = json.loads(_path(kind, key).read_text())
    except (OSError, ValueError):
        return False, None
    if not isinstance(entry, dict) or entry.get("key") != key:
        return False, None
    return True, entry.get("value")


def write(kind: str, key: str, value: Any) -> None:
    """Keep ``value``, which JSON holds, under ``key`` among the entries of ``kind``.

    The entry replaces the one kept under ``key`` in one step, so a call
    reading meanwhile finds either. A directory that cannot be written keeps
    nothing: the calls after this one work the value out again.
    """
    if _root is None:
        return
    path = _path(kind, key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        scratch = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
        scratch.write_text(json.dumps({"key": key, "value": value}))
        scratch.replace(path)
        # One write in 64 looks at how many entries the kind holds.
        if path.name[:2] < "04":
            _trim(path.parent)
    except (OSError, TypeError, ValueError):
        return


def _trim(directory: Path) -> None:
    """Remove the least recently written entries of a kind past :data:`SIZE`."""
    entries = []
    for entry in directory.glob("*.json"):
        with contextlib.suppress(OSError):
            entries.append((entry.stat().st_mtime_ns, entry))
    entries.sort()
    for _, entry in entries[: max(0, len(entries) - SIZE)]:
        with contextlib.suppress(OSError):
            entry.unlink()
    # What a call that ended while writing left behind.
    stale = time.time() - 60.0
    for scratch in directory.glob("*.tmp"):
        with contextlib.suppress(OSError):
            if scratch.stat().st_mtime < stale:
                scratch.unlink()
