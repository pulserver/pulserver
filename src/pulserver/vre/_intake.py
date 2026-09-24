"""The design intake: an HTTP endpoint the design calls push stored designs to."""

from __future__ import annotations

__all__ = ["DesignIntake"]

import logging
import os
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..host._store import BUNDLE_LIMIT, ID_DIGITS, MANIFEST, DesignStore

_log = logging.getLogger("pulserver.vre")
_PATH = re.compile(rf"/designs/([0-9a-f]{{{ID_DIGITS}}})")
# Seconds a connection may stay silent before the intake drops it.
_TIMEOUT = 60.0


class DesignIntake:
    """Receives stored designs over HTTP into the design store the proxy reads.

    ``HEAD /designs/<id>`` answers 200 when the store holds the design, which
    marks it as used for :meth:`~pulserver.host.DesignStore.prune`, and 404
    otherwise; ``PUT /designs/<id>`` stores a bundle of
    :meth:`~pulserver.host.DesignStore.pack` after checking every file against
    its manifest, and answers 201, or 200 for a design already stored. A
    bundle that is not the design its path names is refused with 400 and the
    reason. The endpoint is neither authenticated nor encrypted.

    Parameters
    ----------
    store
        Directory of designs; created when missing.
    host
        Address to listen on; the loopback interface by default.
    port
        TCP port; 0 takes a free one.
    """

    def __init__(
        self, store: Path | str, host: str = "127.0.0.1", port: int = 0
    ) -> None:
        self.store = DesignStore(store)
        intake = self

        class Handler(BaseHTTPRequestHandler):
            timeout = _TIMEOUT

            def do_HEAD(self) -> None:
                design = self._design()
                if design is None:
                    return
                try:
                    os.utime(intake.store.directory(design) / MANIFEST)
                except OSError:
                    self._answer(HTTPStatus.NOT_FOUND)
                    return
                self._answer(HTTPStatus.OK)

            def do_PUT(self) -> None:
                design = self._design()
                if design is None:
                    return
                try:
                    length = int(self.headers.get("Content-Length", ""))
                except ValueError:
                    self._answer(
                        HTTPStatus.LENGTH_REQUIRED, "a bundle states its length"
                    )
                    return
                if length > BUNDLE_LIMIT:
                    self._answer(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        f"a bundle takes at most {BUNDLE_LIMIT} bytes",
                    )
                    return
                bundle = self.rfile.read(length)
                held = (intake.store.directory(design) / MANIFEST).is_file()
                try:
                    received = intake.store.receive(bundle)
                except ValueError as error:
                    self._answer(HTTPStatus.BAD_REQUEST, str(error))
                    return
                if received != design:
                    self._answer(
                        HTTPStatus.BAD_REQUEST,
                        f"the bundle holds design {received}, not {design}",
                    )
                    return
                _log.info("received design %s", design)
                self._answer(HTTPStatus.OK if held else HTTPStatus.CREATED)

            def _design(self) -> str | None:
                match = _PATH.fullmatch(self.path)
                if match is None:
                    self._answer(
                        HTTPStatus.NOT_FOUND, "the intake serves /designs/<id>"
                    )
                    return None
                return match.group(1)

            def _answer(self, status: HTTPStatus, text: str = "") -> None:
                body = f"{text}\n".encode() if text else b""
                self.send_response(status)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if body and self.command != "HEAD":
                    self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                _log.info("intake %s", format % args)

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        """Port the intake listens on."""
        return int(self._server.server_address[1])

    def start(self) -> None:
        """Serve requests in a thread of its own until :meth:`close`."""
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="design-intake"
        )
        self._thread.start()

    def close(self) -> None:
        """Stop serving and release the port."""
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join()
        self._server.server_close()
