"""The design intake: an HTTP endpoint the design calls push stored designs to."""

from __future__ import annotations

__all__ = ["DesignIntake"]

import logging
import os
import re
import threading
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..host._store import BUNDLE_LIMIT, ID_DIGITS, MANIFEST, DesignStore, _flat

_log = logging.getLogger("pulserver.proxy")
_PATH = re.compile(rf"/designs/([^/]+)/([0-9a-f]{{{ID_DIGITS}}})")
# Seconds a connection may stay silent before the intake drops it.
_TIMEOUT = 60.0


class DesignIntake:
    """Receives stored designs over HTTP into the design store the proxy reads.

    ``HEAD /designs/<scan>/<id>`` answers 200 when the scan holds the design,
    which marks the scan as used for :func:`~pulserver.host.prune`, and 404
    otherwise; ``PUT /designs/<scan>/<id>`` stores a bundle of
    :meth:`~pulserver.host.DesignStore.pack` after checking every file against
    its manifest, and answers 201, or 200 for a design already stored. A
    bundle that is not the design its path names is refused with 400 and the
    reason. The endpoint is neither authenticated nor encrypted.

    Parameters
    ----------
    store : Path | str
        Directory of scans; created when missing.
    host : str, default='127.0.0.1'
        Address to listen on; the loopback interface by default.
    port : int, default=0
        TCP port; 0 takes a free one.
    received : Callable[[Path], object], default=None
        Called with the directory of each design stored or found stored, in
        a thread of its own once the request is answered: a proxy tabulates
        the design there, ahead of the scan that plays it.
    """

    def __init__(
        self,
        store: Path | str,
        host: str = "127.0.0.1",
        port: int = 0,
        received: Callable[[Path], object] | None = None,
    ) -> None:
        self.root = Path(store)
        self.root.mkdir(parents=True, exist_ok=True)
        intake = self

        class Handler(BaseHTTPRequestHandler):
            timeout = _TIMEOUT

            def do_HEAD(self) -> None:
                named = self._design()
                if named is None:
                    return
                store, design = named
                try:
                    os.utime(store.directory(design) / MANIFEST)
                except OSError:
                    self._answer(HTTPStatus.NOT_FOUND)
                    return
                self._answer(HTTPStatus.OK)
                self._prepare(store, design)

            def do_PUT(self) -> None:
                named = self._design()
                if named is None:
                    return
                store, design = named
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
                held = (store.directory(design) / MANIFEST).is_file()
                try:
                    store.receive(bundle, design)
                except ValueError as error:
                    self._answer(HTTPStatus.BAD_REQUEST, str(error))
                    return
                _log.info("received design %s of scan %s", design, store.scan)
                self._answer(HTTPStatus.OK if held else HTTPStatus.CREATED)
                self._prepare(store, design)

            def _prepare(self, store: DesignStore, design: str) -> None:
                if received is not None:
                    threading.Thread(
                        target=_prepare,
                        args=(received, store.directory(design)),
                        daemon=True,
                        name=f"prepare-{design}",
                    ).start()

            def _design(self) -> tuple[DesignStore, str] | None:
                match = _PATH.fullmatch(self.path)
                if match is None or not _flat(match.group(1)):
                    self._answer(
                        HTTPStatus.NOT_FOUND, "the intake serves /designs/<scan>/<id>"
                    )
                    return None
                return DesignStore(intake.root, match.group(1)), match.group(2)

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


def _prepare(received: Callable[[Path], object], directory: Path) -> None:
    try:
        received(directory)
    except Exception:
        _log.exception("preparing design %s failed", directory.name)
