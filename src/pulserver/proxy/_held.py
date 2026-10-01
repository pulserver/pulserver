"""What a series sends its client, kept on disk until the client has had all of it.

A reconstruction can outlive the connection of the client that asked for it.
Everything sent to the client is also written, as the MRD bytes the client
reads, to a file named after the series' ``measurementID``; the file is removed
once the client has had the whole output, and kept when it has gone. A client
opening the same measurement again is sent the kept output in place of a new
reconstruction.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import socket
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from ..recon._runtime.connection import Connection

_log = logging.getLogger("pulserver.proxy")

#: Directory the kept outputs are held in, shared by the proxies of this host
#: so a client may return to any of them; readable by the owner alone.
DEFAULT_HELD_DIRECTORY = Path(tempfile.gettempdir()) / f"pulserver-held-{os.getuid()}"

#: Seconds an output nobody has taken up is kept.
KEPT_FOR = 24 * 3600.0

#: Suffix of an output still being written.
PART = ".part"

# How often a returning client looks at whether its output is complete.
_POLL = 0.5


def held_path(directory: Path, header: Any) -> tuple[Path, bool]:
    """Return where the output of the series ``header`` opens is held, and whether a client can claim it.

    The series is named by its ``measurementID``, which is what tells two
    concurrent reconstructions apart. A series without one is held under a
    name of its own, which no client can claim.
    """
    information = getattr(header, "measurementInformation", None)
    measurement = None if information is None else information.measurementID
    if measurement:
        return directory / hashlib.sha256(str(measurement).encode()).hexdigest(), True
    return directory / f"unclaimed-{uuid.uuid4().hex}", False


def take_up(client: Connection, path: Path) -> bool:
    """Send ``client`` the output held at ``path``; return whether there was one.

    An output still being written is waited for. The output is removed once
    sent.
    """
    part = path.with_name(path.name + PART)
    while part.exists() and not path.exists():
        time.sleep(_POLL)
    try:
        output = path.read_bytes()
    except FileNotFoundError:
        return False
    _log.info("sending the output held for %s", path.name)
    client.socket.write(output)
    path.unlink(missing_ok=True)
    return True


class HeldClient:
    """The client of one series, with everything sent to it also written to disk.

    Stands in for the client's :class:`Connection` where a reconstruction's
    output is sent. A send that fails, or finds the client's connection reset,
    leaves the client gone: what follows is written and no longer sent. On exit the file
    is removed when the client had everything, and kept at ``path`` otherwise.
    """

    def __init__(self, client: Connection, path: Path) -> None:
        self._client = client
        self.path = path
        self.gone = False
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _prune(path.parent)
        self._part = path.with_name(path.name + PART)
        self._file = self._part.open("wb")

    def send(self, item: Any) -> None:
        for accepts, writer in self._client.writers:
            if accepts(item):
                writer(self._file, item)
                break
        self._file.flush()
        if self.gone:
            return
        try:
            if _reset(self._client):
                raise ConnectionResetError("the client's connection was reset")
            self._client.send(item)
        except Exception as error:
            self._leave(error)

    def _leave(self, error: Exception) -> None:
        self.gone = True
        _log.warning(
            "the client of %s has gone (%s); its output is held",
            self.path.name,
            error,
        )

    def __enter__(self) -> HeldClient:
        return self

    def __exit__(self, *_: object) -> None:
        # The last items may have been written into the connection of a client
        # already gone, which only its reset says.
        if not self.gone and _reset(self._client):
            self._leave(ConnectionResetError("the client's connection was reset"))
        self._file.close()
        if self.gone:
            self._part.replace(self.path)
        else:
            self._part.unlink(missing_ok=True)


def _reset(connection: Connection) -> bool:
    """Whether the peer's connection was reset: what was sent since was not read.

    A peer that closes only its sending side may still be reading, so an end
    of stream does not count.
    """
    raw = connection.socket.socket
    try:
        raw.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
    except BlockingIOError:
        return False
    except OSError:
        return True
    return False


def _prune(directory: Path) -> None:
    """Remove the outputs held longer than :data:`KEPT_FOR`."""
    oldest = time.time() - KEPT_FOR
    for path in directory.iterdir():
        with contextlib.suppress(OSError):
            if path.stat().st_mtime < oldest:
                path.unlink()
