"""TCP MRD server between the scanner's reconstruction client and the workers."""

from __future__ import annotations

__all__ = ["ReconProxy"]

import contextlib
import itertools
import logging
import os
import re
import shutil
import socket
import tempfile
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import ismrmrd

from ..recon._runtime import constants
from ..recon._runtime.concurrency import compute_max_concurrent
from ..recon._runtime.connection import Connection
from ..recon._runtime.readers import deserialize_config, read_text
from ._enrich import enrich_acquisition, enrich_header, fov_offset_m
from ._queue import QueueFile
from ._revisions import Revision, RevisionStore
from ._workers import WorkerPool

_PLUGIN_NAME = re.compile(r"[A-Za-z0-9_\-]+")
_log = logging.getLogger("pulserver.vre")

# A worker spawns, imports its plugin and connects; past this it is not coming.
_CONNECT_TIMEOUT = 120.0
# How long closing the proxy waits for each running series.
_WORKER_TIMEOUT = 120.0
# How often the accept loop looks at whether the proxy is closing.
_ACCEPT_POLL = 0.5


class ReconProxy:
    """Routes each series of an MRD stream to a reconstruction worker.

    One thread per client connection. The header's ``pulserver_session`` and
    ``pulserver_revision`` name the design the series was played from; its
    readout table enriches the header and every acquisition, and its
    ``pulserver_fov_offset_mm`` demodulates them to the prescription centre.
    The reconstruction plugin is the revision's, falling back to the name in
    the client's config text.

    A series holds a slot for as long as it runs. Its worker's images, DICOM
    and text go back to the client as they arrive, the client's close closes
    the worker, and the worker's close closes the client.

    A series that finds every slot busy is queued instead: its enriched stream
    is written to ``bucket/<session>/queue/<id>.h5`` while it arrives, its
    client stays connected, and the file is replayed to a worker and deleted
    once a slot frees.

    Once the client's stream ends, the proxy waits for the worker to close,
    however long the reconstruction takes, unless ``recon_timeout`` caps it.

    Parameters
    ----------
    base
        Directory holding ``bucket/``, as the host daemon writes it.
    plugins
        Directory of reconstruction plugin files, ``<plugin>.py``.
    slots
        Series reconstructed at once; the memory-derived limit when ``None``.
    spares
        Warm worker processes waiting for a series.
    recon_timeout
        Seconds a worker may take after the stream ends before it is
        terminated and the client told so; ``None`` waits for its close.

    Attributes
    ----------
    workers : WorkerPool
        The spares assignments are taken from.
    """

    def __init__(
        self,
        base: Path | str,
        plugins: Path | str,
        *,
        slots: int | None = None,
        spares: int = 1,
        recon_timeout: float | None = None,
    ) -> None:
        self.revisions = RevisionStore(base)
        self.plugins = Path(plugins)
        self.workers = WorkerPool(spares=spares)
        self.recon_timeout = recon_timeout
        self._slots = threading.BoundedSemaphore(compute_max_concurrent(override=slots))
        self._server: socket.socket | None = None
        self._closing = threading.Event()
        self._stopping = False
        self._threads: list[threading.Thread] = []
        self._queued = itertools.count(1)

    def bind(self, port: int = 0, host: str = "127.0.0.1") -> int:
        """Listen on ``port`` and return the port bound; 0 takes a free one.

        ``host`` is the address to listen on, the loopback interface by
        default; ``"0.0.0.0"`` is every interface. The stream is neither
        authenticated nor encrypted.
        """
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(16)
        server.settimeout(_ACCEPT_POLL)
        self._server = server
        return int(server.getsockname()[1])

    def serve(self) -> None:
        """Accept clients until :meth:`stop` or :meth:`close`, one thread each."""
        if self._server is None:
            self.bind()
        _log.info("serving %s on port %d", self.revisions.bucket, self.port)
        while not (self._stopping or self._closing.is_set()):
            try:
                stream, _ = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            thread = threading.Thread(target=self._client, args=(stream,), daemon=True)
            thread.start()
            self._threads = [t for t in self._threads if t.is_alive()]
            self._threads.append(thread)

    @property
    def port(self) -> int:
        """Port the proxy listens on.

        Raises
        ------
        RuntimeError
            Before :meth:`bind`.
        """
        if self._server is None:
            raise RuntimeError("the proxy is not bound")
        return int(self._server.getsockname()[1])

    def stop(self) -> None:
        """Make :meth:`serve` return within one accept poll.

        Sets a flag and takes no lock, so a signal handler may call it.
        """
        self._stopping = True

    def close(self) -> None:
        """Stop accepting clients, release the spares and wait for running series."""
        self._closing.set()
        if self._server is not None:
            with contextlib.suppress(OSError):
                self._server.close()
        for thread in self._threads:
            thread.join(timeout=_WORKER_TIMEOUT)
        self.workers.close()

    # %% one client connection

    def _client(self, stream: socket.socket) -> None:
        connection = Connection(stream)
        # The config names a plugin, so the proxy keeps the text the client
        # wrote rather than the mapping a parser makes of it.
        connection.add_reader(constants.GADGET_MESSAGE_CONFIG, read_text)
        try:
            config = _first(connection)
            header = _first(connection)
            if header is None:
                raise ValueError("the stream carried no header")
            self._series(connection, str(config or ""), header)
        except Exception as error:
            _log.exception("series failed")
            with contextlib.suppress(Exception):
                connection.send(f"pulserver: {error}")
        finally:
            connection.shutdown_close()

    def _series(self, client: Connection, config: str, header: Any) -> None:
        revision = self.revisions.resolve(header)
        plugin = self._plugin_path(revision.recon or _config_plugin(config))
        enrich_header(header, revision.table)
        offset = fov_offset_m(header)
        _log.info(
            "series on %s: %s, %d readouts",
            revision.directory,
            plugin.stem,
            len(revision.table),
        )
        if self._slots.acquire(blocking=False):
            try:
                self._run(
                    client,
                    config,
                    header,
                    plugin,
                    lambda worker: _forward(client, worker, revision, offset),
                )
            finally:
                self._slots.release()
            return
        self._queue(client, config, header, revision, plugin, offset)

    def _queue(
        self,
        client: Connection,
        config: str,
        header: Any,
        revision: Revision,
        plugin: Path,
        offset: Any,
    ) -> None:
        """Hold the series on disk until a slot frees, then replay it to a worker."""
        queued = QueueFile(
            revision.session_directory
            / "queue"
            / f"{os.getpid()}-{next(self._queued)}.h5"
        )
        _log.info("queued %s on %s", queued.path.name, revision.directory)
        try:
            others = _record(client, header, revision, offset, queued)
            self._slots.acquire()
            try:
                self._run(
                    client,
                    config,
                    header,
                    plugin,
                    lambda worker: _replay(queued, others, worker),
                )
            finally:
                self._slots.release()
        finally:
            queued.unlink()

    def _run(
        self,
        client: Connection,
        config: str,
        header: Any,
        plugin: Path,
        feed: Callable[[Connection], None],
    ) -> None:
        """Give a worker the config, the header and whatever ``feed`` sends it."""
        channel = _WorkerChannel(self.workers, plugin)
        with channel as worker:
            worker.send_config(config)
            worker.send_header(header)
            relay = threading.Thread(
                target=_relay, args=(worker, client), daemon=True, name="relay"
            )
            relay.start()
            try:
                feed(worker)
            except BaseException:
                # A series that cannot be fed whole is not reconstructed.
                channel.terminate()
                relay.join()
                raise
            worker.send_close()
            relay.join(timeout=self.recon_timeout)
            if relay.is_alive():
                _log.warning(
                    "%s exceeded %s s; terminating it",
                    plugin.stem,
                    self.recon_timeout,
                )
                # The relay ends with the worker, so the notice cannot
                # interleave with an image it is still sending.
                channel.terminate()
                relay.join()
                with contextlib.suppress(Exception):
                    client.send(
                        f"pulserver: {plugin.stem} did not finish within "
                        f"{self.recon_timeout:g} s of the end of the series"
                    )

    def _plugin_path(self, plugin: str) -> Path:
        if not plugin:
            raise ValueError(
                "neither the revision nor the config names a reconstruction"
            )
        if not _PLUGIN_NAME.fullmatch(plugin):
            raise ValueError(f"invalid plugin name {plugin!r}")
        path = self.plugins / f"{plugin}.py"
        if not path.is_file():
            raise FileNotFoundError(f"no plugin {plugin!r} in {self.plugins}")
        return path


class _WorkerChannel:
    """A worker's end of one series: its Unix socket, its process, its stream."""

    def __init__(self, pool: WorkerPool, plugin: Path) -> None:
        self._directory = Path(tempfile.mkdtemp(prefix="pulserver-series-"))
        self._pool = pool
        self._plugin = plugin
        self._process = None
        self.connection: Connection | None = None

    def __enter__(self) -> Connection:
        path = self._directory / "worker.sock"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(path))
            listener.listen(1)
            listener.settimeout(_CONNECT_TIMEOUT)
            process = self._process = self._pool.assign(self._plugin, path)
            try:
                stream, _ = listener.accept()
            except TimeoutError:
                process.terminate()
                raise RuntimeError(
                    f"the worker for {self._plugin.stem} never connected"
                ) from None
        finally:
            listener.close()
        self.connection = Connection(stream)
        return self.connection

    def __exit__(self, *exception: object) -> None:
        if self.connection is not None:
            self.connection.shutdown_close()
        shutil.rmtree(self._directory, ignore_errors=True)

    def terminate(self) -> None:
        """Stop the worker process, whatever it is doing."""
        if self._process is not None and self._process.is_alive():
            self._process.terminate()


# %% private module subroutines


def _first(connection: Connection) -> Any:
    """Read the next item, or ``None`` when the stream ended first."""
    try:
        return connection.next()[1]
    except StopIteration:
        return None


def _config_plugin(config: str) -> str:
    """Return the plugin a config text names.

    A bare name is the plugin; anything else is parsed as the runtime parses a
    config and read from ``parameters.config``.
    """
    text = config.strip()
    if _PLUGIN_NAME.fullmatch(text):
        return text
    parsed = deserialize_config(text, "")
    parameters = parsed.get("parameters") if isinstance(parsed, dict) else None
    if not isinstance(parameters, dict):
        return ""
    return str(parameters.get("config", ""))


def _is_close_marker(item: Any) -> bool:
    """Whether an acquisition is the empty end marker a CLOSE message is read as."""
    return (
        isinstance(item, ismrmrd.Acquisition)
        and int(item.number_of_samples) == 0
        and item.isFlagSet(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
    )


def _enriched(client: Connection, revision: Revision, offset: Any) -> Iterator[Any]:
    """Yield the client's stream up to its close, acquisitions enriched in play order.

    Acquisitions are matched to readouts by position, so once the client numbers
    them, every ``scan_counter`` must follow the previous one by one: a gap or a
    repeat is a dropped or duplicated readout that would shift every later row.
    A stream whose counters stay 0 is not numbered and not checked.

    Raises
    ------
    ValueError
        If the stream carries more acquisitions than the sequence plays, or its
        scan counters skip or repeat one.
    """
    index = 0
    previous = None
    for item in client:
        if _is_close_marker(item):
            return
        if isinstance(item, ismrmrd.Acquisition):
            if index >= len(revision.table):
                raise ValueError(
                    f"the stream carries more than the {len(revision.table)} readouts "
                    f"{revision.directory} plays"
                )
            counter = int(item.scan_counter)
            numbered = previous is not None and (previous, counter) != (0, 0)
            if numbered and counter != previous + 1:
                raise ValueError(
                    f"acquisition {index} carries scan counter {counter} after "
                    f"{previous}: the stream no longer matches the readouts "
                    f"{revision.directory} plays"
                )
            previous = counter
            enrich_acquisition(item, revision.table, index, offset)
            index += 1
        yield item


def _forward(
    client: Connection, worker: Connection, revision: Revision, offset: Any
) -> None:
    """Send the client's stream to the worker as it arrives."""
    for item in _enriched(client, revision, offset):
        worker.send(item)


def _record(
    client: Connection,
    header: Any,
    revision: Revision,
    offset: Any,
    queued: QueueFile,
) -> list[Any]:
    """Store the client's stream until it closes; return what the file cannot hold."""
    queued.write_header(header)
    others: list[Any] = []
    try:
        for item in _enriched(client, revision, offset):
            if isinstance(item, (ismrmrd.Acquisition, ismrmrd.Waveform)):
                queued.append(item)
            else:
                others.append(item)
    finally:
        queued.close()
    return others


def _replay(queued: QueueFile, others: list[Any], worker: Connection) -> None:
    for item in others:
        worker.send(item)
    for item in queued:
        worker.send(item)


def _relay(worker: Connection, client: Connection) -> None:
    """Send everything the worker emits back to the client, until its close."""
    try:
        for item in worker:
            if _is_close_marker(item):
                break
            client.send(item)
    except Exception:
        _log.exception("relaying a worker's output failed")
