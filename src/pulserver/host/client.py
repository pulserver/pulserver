"""Blocking client for the host commands, speaking as a PSD host process does."""

from __future__ import annotations

import contextlib
import socket
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..protocol import (
    PROTOCOL_END,
    Parameter,
    Validation,
    format_values,
    parse_listing,
    parse_validation,
)
from ._blocks import format_import, format_limits
from ._sessions import SessionKey


class HostError(RuntimeError):
    """The daemon answered a command with ``ERROR``."""


class HostClient:
    """One session's connection to the host daemon.

    A dropped connection is reopened once per command, so a session survives a
    daemon restart.

    Raises
    ------
    HostError
        From any command the daemon refuses.
    """

    def __init__(self, socket_path: Path, session: SessionKey) -> None:
        self.socket_path = Path(socket_path)
        self.session = session
        self._listing: dict[str, Parameter] | None = None
        self._stream = None

    def open(self, plugin: str | None, limits: Mapping[str, Any]) -> None:
        """Open the session.

        ``limits`` are ``pypulseqpp.Opts`` keyword arguments plus optional ``ir_``
        conversion options. A session opened without a plugin only imports
        sequence files.
        """
        command = f"OPEN {self.session}" + (f" {plugin}" if plugin else "")
        self._command(command, format_limits(dict(limits)))

    def list_protocol(self) -> dict[str, Parameter]:
        """Return the protocol with its schema, kept for formatting later value blocks."""
        _, block = self._command(f"LIST_PROTOCOL {self.session}", until=PROTOCOL_END)
        self._listing = parse_listing(block)
        return self._listing

    def validate(self, values: Mapping[str, Any]) -> Validation:
        """Resolve values keyed by interpreter parameter names."""
        listing = self._listing or self.list_protocol()
        header, rest = self._command(
            f"VALIDATE {self.session}",
            format_values(values, listing),
            until=PROTOCOL_END,
        )
        return parse_validation(header + rest, listing)

    def generate(self, values: Mapping[str, Any]) -> int:
        """Return the revision generated, or found again, for the resolved protocol."""
        listing = self._listing or self.list_protocol()
        header, _ = self._command(
            f"GENERATE {self.session}", format_values(values, listing)
        )
        return int(header.split()[1])

    def import_sequence(self, path: Path | str) -> int:
        """Return the revision holding a sequence file, its chain and their cache."""
        header, _ = self._command(f"IMPORT {self.session}", format_import(path))
        return int(header.split()[1])

    def close(self) -> None:
        """Send ``CLOSE`` and drop the connection."""
        self._command(f"CLOSE {self.session}")
        self.disconnect()

    def disconnect(self) -> None:
        """Drop the connection; the next command reopens it."""
        stream, self._stream = self._stream, None
        if stream is not None:
            # A request still buffered for a dead daemon is dropped with the stream.
            with contextlib.suppress(OSError):
                stream.close()

    def _command(
        self, line: str, block: str = "", *, until: str | None = None
    ) -> tuple[str, str]:
        for attempt in (1, 2):
            try:
                return self._exchange(f"{line}\n{block}", until)
            except ConnectionError:
                self.disconnect()
                if attempt == 2:
                    raise
        raise AssertionError  # unreachable

    def _exchange(self, request: str, until: str | None) -> tuple[str, str]:
        if self._stream is None:
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.connect(str(self.socket_path))
            self._stream = connection.makefile("rwb")
        try:
            self._stream.write(request.encode())
            self._stream.flush()
        except OSError as error:
            raise ConnectionError(error) from error
        header = self._readline()
        if header.startswith("ERROR"):
            raise HostError(header.removeprefix("ERROR").strip())
        rest = []
        if until is not None:
            while True:
                line = self._readline()
                rest.append(line)
                if until in line:
                    break
        return header, "".join(rest)

    def _readline(self) -> str:
        line = self._stream.readline().decode()
        if not line:
            raise ConnectionError("the daemon closed the connection")
        return line
