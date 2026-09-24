"""The warm design server: one pre-imported process, one forked child per call."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import socket
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_log = logging.getLogger("pulserver.host")

# How often the accept loop reaps children and looks for a stop request.
_POLL = 0.2
# How long a stopping server waits for a terminated child.
_GRACE = 5.0
# A request line past this many bytes is refused.
_REQUEST_LIMIT = 64 << 20


def answer(request: Mapping[str, Any]) -> tuple[int, str]:
    """Answer a forwarded call as the command does in its own process.

    ``request`` carries ``call`` and, as the call takes them, ``plugins``,
    ``plugin``, ``limits`` (the text of a ``[Limits]`` block), ``store``,
    ``push`` and ``input`` (the block the command reads from standard input).
    """
    from . import _service
    from ._blocks import parse_limits
    from ._store import DesignStore

    inputs: dict[str, Any] = {}
    if request.get("plugin") is not None:
        inputs["plugins"] = Path(request["plugins"])
        inputs["plugin"] = str(request["plugin"])
    if request.get("limits") is not None:
        inputs["limits"] = parse_limits(str(request["limits"]))
    if request.get("store") is not None:
        inputs["store"] = DesignStore(request["store"])
    if request.get("call") in ("validate", "generate", "import"):
        inputs["block"] = str(request.get("input", ""))
    if request.get("push"):
        inputs["push"] = str(request["push"])
    return _service.call(str(request.get("call")), **inputs)


class DesignServer:
    """Answers design calls on a Unix socket, each in a child forked from a warm parent.

    The parent imports the design engine, designs, checks and converts one
    sequence of its own, and imports every plugin of ``plugins``; no plugin
    code designs in it. A call is then answered in a child forked from it, which
    inherits what the parent imported, answers as :func:`answer` does and
    exits: nothing a call does outlives its child. A child that ends without
    a reply, a crash of plugin code among them, is answered with an ``ERROR``
    line naming its exit status. Calls run concurrently, one child each.

    A request is one line of JSON, as :func:`answer` takes it; the reply is
    one line of JSON, ``{"status": <exit status>, "output": <reply text>}``.
    The socket is neither authenticated nor encrypted; its file permissions
    decide who may call.
    """

    def __init__(self, plugins: Path | str, socket_path: Path | str) -> None:
        self.plugins = Path(plugins)
        self.socket_path = Path(socket_path)
        self._stopping = False
        self._children: dict[int, socket.socket] = {}

    def warm(self) -> None:
        """Design, check and convert pypulseqpp's 2D gradient echo, and import every plugin.

        A plugin that cannot be imported is left cold; its calls are answered
        all the same.
        """
        from . import _service

        _warm_design()
        for path in sorted(self.plugins.glob("*.py")):
            try:
                _service.list_protocol(self.plugins, path.stem)
            except Exception as error:  # a plugin that fails to list stays cold
                _log.info("warming %s stopped: %s", path.stem, error)

    def serve(self) -> None:
        """Answer calls until :meth:`stop`, then end every running child."""
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket_path.unlink(missing_ok=True)
        listener.bind(str(self.socket_path))
        listener.listen(64)
        listener.settimeout(_POLL)
        _log.info("serving design calls on %s", self.socket_path)
        try:
            while not self._stopping:
                self._reap()
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                except OSError:
                    if self._stopping:
                        break
                    raise
                connection.settimeout(None)
                pid = os.fork()
                if pid == 0:
                    listener.close()
                    self._child(connection)
                self._children[pid] = connection
        finally:
            listener.close()
            self.socket_path.unlink(missing_ok=True)
            self._end_children()

    def stop(self) -> None:
        """Make :meth:`serve` return within one poll; safe to call from a signal handler."""
        self._stopping = True

    def _child(self, connection: socket.socket) -> None:
        for other in self._children.values():
            other.close()
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        status = 1
        try:
            with connection.makefile("rb") as reader:
                line = reader.readline(_REQUEST_LIMIT + 1)
            if len(line) > _REQUEST_LIMIT:
                code, output = 1, "ERROR the request exceeds the size a call takes\n"
            else:
                code, output = answer(json.loads(line))
            reply = json.dumps({"status": code, "output": output}) + "\n"
            connection.sendall(reply.encode())
            status = 0
        finally:
            os._exit(status)

    def _reap(self) -> None:
        while self._children:
            try:
                pid, wait_status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if pid == 0:
                return
            connection = self._children.pop(pid, None)
            if connection is None:
                continue
            code = os.waitstatus_to_exitcode(wait_status)
            if code != 0:
                reply = {
                    "status": 1,
                    "output": f"ERROR the design call ended with exit status {code}\n",
                }
                with contextlib.suppress(OSError):
                    connection.sendall((json.dumps(reply) + "\n").encode())
            connection.close()

    def _end_children(self) -> None:
        for pid in list(self._children):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGTERM)
        for pid, connection in list(self._children.items()):
            with contextlib.suppress(ChildProcessError):
                _wait(pid, _GRACE)
            connection.close()
        self._children.clear()


def _warm_design() -> None:
    import pypulseqpp as pp
    from pypulseqpp.sequences.sequence.gre2D_sequence import Gre2DApp

    from .. import ir

    system = pp.Opts()
    with tempfile.TemporaryDirectory(prefix="pulserver-warm-") as scratch:
        first = Gre2DApp(system).write(Path(scratch) / "sequence.seq", offline=False)[0]
        ir.check(first, system)
        ir.convert(first, system)


def _wait(pid: int, timeout: float) -> None:
    """Wait for a child to end, killing it once ``timeout`` has passed."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        done, _ = os.waitpid(pid, os.WNOHANG)
        if done:
            return
        time.sleep(0.05)
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)


def serve(plugins: Path | str, socket_path: Path | str) -> int:
    """Warm a server, then answer calls on ``socket_path`` until SIGTERM or SIGINT."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    server = DesignServer(plugins, socket_path)
    signal.signal(signal.SIGTERM, lambda *_: server.stop())
    signal.signal(signal.SIGINT, lambda *_: server.stop())
    server.warm()
    server.serve()
    return 0
