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
import ismrmrd.xsd as xsd

from ..recon._runtime import constants
from ..recon._runtime.concurrency import Slot, Slots, slot_devices
from ..recon._runtime.connection import Connection
from ..recon._runtime.exam import ExamCacheManager
from ..recon._runtime.readers import deserialize_config, read_text
from ._enrich import enrich_acquisition, enrich_header
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
# How long a refused client may send nothing before the proxy stops reading it.
_DRAIN_IDLE = 30.0


class ReconProxy:
    """Routes each series of an MRD stream to a reconstruction worker.

    One thread per client connection, which carries one series: an optional
    config text, the header, one acquisition per readout of the chain and a
    ``CLOSE``. The header's ``pulserver_session`` and ``pulserver_revision``
    name the design the series was played from; its readout table enriches the
    header and every acquisition. The readouts arrive demodulated to the
    prescribed field-of-view centre by the playout
    (:func:`pulserver.ir.prescribe`), and their samples are passed on as
    received. The reconstruction plugin is the revision's, falling back to
    the name in the client's config text. A series that breaks this is refused
    with a text naming the reason and a ``CLOSE``, and what the client still
    sends is discarded until it closes the connection.

    A series holds a slot for as long as it runs. On a host with GPUs, found
    through ``CUDA_VISIBLE_DEVICES`` or ``nvidia-smi``, each slot holds one,
    ``gpu_slots`` slots per GPU, and the reconstruction reads it as
    ``context.device``. Its worker's images, DICOM and text go back to the
    client as they arrive, the client's close closes the worker, and the
    worker's close closes the client.

    A series that finds every slot busy is queued instead: its enriched stream
    is written to ``bucket/<session>/queue/<id>.h5`` while it arrives, its
    client stays connected, and the file is replayed to a worker and deleted
    once a slot frees.

    Once the client's stream ends, the proxy waits for the worker to close,
    however long the reconstruction takes, unless ``recon_timeout`` caps it.

    The series of one exam share a directory under the system's temporary
    directory, through which a reconstruction reads what an earlier series of
    the exam stored in its :class:`~pulserver.recon.ExamCache`. It is deleted
    once a header names another exam and no series of the exam still runs.

    Parameters
    ----------
    base
        Directory holding ``bucket/``, as the host daemon writes it.
    plugins
        Directory of reconstruction plugin files, ``<plugin>.py``.
    slots
        Series reconstructed at once; derived from memory and the GPUs when
        ``None``.
    gpu_slots
        Series reconstructed at once on each GPU, when ``slots`` is ``None``.
    spares
        Warm worker processes waiting for a series.
    recon_timeout
        Seconds a worker may take after the stream ends before it is
        terminated and the client told so; ``None`` waits for its close.

    Attributes
    ----------
    workers : WorkerPool
        The spares assignments are taken from.
    exams : ExamCacheManager
        The current exam and its directory.
    """

    def __init__(
        self,
        base: Path | str,
        plugins: Path | str,
        *,
        slots: int | None = None,
        gpu_slots: int = 1,
        spares: int = 1,
        recon_timeout: float | None = None,
    ) -> None:
        self.revisions = RevisionStore(base)
        self.plugins = Path(plugins)
        self.workers = WorkerPool(spares=spares)
        self.recon_timeout = recon_timeout
        self._exam_root = Path(tempfile.mkdtemp(prefix="pulserver-exams-"))
        self.exams = ExamCacheManager(directory=self._exam_root)
        self._slots = Slots(slot_devices(slots, gpu_slots))
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
        self.exams.close()
        shutil.rmtree(self._exam_root, ignore_errors=True)

    # %% one client connection

    def _client(self, stream: socket.socket) -> None:
        connection = Connection(stream)
        # The config names a plugin, so the proxy keeps the text the client
        # wrote rather than the mapping a parser makes of it.
        connection.add_reader(constants.GADGET_MESSAGE_CONFIG, read_text)
        try:
            config, header = _opening(connection)
            self._series(connection, config, header)
        except Exception as error:
            _log.exception("series failed")
            _refuse(connection, error)
        finally:
            connection.shutdown_close()

    def _series(self, client: Connection, config: str, header: Any) -> None:
        revision = self.revisions.resolve(header)
        plugin = self._plugin_path(revision.recon or _config_plugin(config))
        enrich_header(header, revision.table)
        _log.info(
            "series on %s: %s, %d readouts",
            revision.directory,
            plugin.stem,
            len(revision.table),
        )
        slot = self._slots.take(wait=False)
        if slot is not None:
            try:
                self._run(
                    client,
                    config,
                    header,
                    plugin,
                    slot,
                    lambda worker: _forward(client, worker, revision),
                )
            finally:
                self._slots.release(slot)
            return
        self._queue(client, config, header, revision, plugin)

    def _queue(
        self,
        client: Connection,
        config: str,
        header: Any,
        revision: Revision,
        plugin: Path,
    ) -> None:
        """Hold the series on disk until a slot frees, then replay it to a worker."""
        queued = QueueFile(
            revision.session_directory
            / "queue"
            / f"{os.getpid()}-{next(self._queued)}.h5"
        )
        _log.info("queued %s on %s", queued.path.name, revision.directory)
        try:
            others = _record(client, header, revision, queued)
            slot = self._slots.take(wait=True)
            try:
                self._run(
                    client,
                    config,
                    header,
                    plugin,
                    slot,
                    lambda worker: _replay(queued, others, worker),
                )
            finally:
                self._slots.release(slot)
        finally:
            queued.unlink()

    def _run(
        self,
        client: Connection,
        config: str,
        header: Any,
        plugin: Path,
        slot: Slot,
        feed: Callable[[Connection], None],
    ) -> None:
        """Give a worker the config, the header and whatever ``feed`` sends it."""
        with self.exams.lease(header) as exam:
            channel = _WorkerChannel(self.workers, plugin, exam.directory, slot.device)
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

    def __init__(
        self,
        pool: WorkerPool,
        plugin: Path,
        exam_directory: Path | None,
        device: str | None,
    ) -> None:
        self._directory = Path(tempfile.mkdtemp(prefix="pulserver-series-"))
        self._pool = pool
        self._plugin = plugin
        self._exam_directory = exam_directory
        self._device = device
        self._process = None
        self.connection: Connection | None = None

    def __enter__(self) -> Connection:
        path = self._directory / "worker.sock"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(path))
            listener.listen(1)
            listener.settimeout(_CONNECT_TIMEOUT)
            process = self._process = self._pool.assign(
                self._plugin, path, self._exam_directory, self._device
            )
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


def _refuse(connection: Connection, error: Exception) -> None:
    """Send why a series failed and a CLOSE, then discard what the client still sends.

    The client reads the reason once it has sent its stream; closing the
    socket while it sends would end its stream with a reset instead.
    """
    with contextlib.suppress(Exception):
        connection.send(f"pulserver: {error}")
    connection.send_close()
    raw = connection.socket.socket
    with contextlib.suppress(OSError):
        raw.shutdown(socket.SHUT_WR)
        raw.settimeout(_DRAIN_IDLE)
        while raw.recv(1 << 16):
            pass


def _opening(connection: Connection) -> tuple[str, Any]:
    """Return the config text and the header a stream opens with.

    The config is optional: a stream may open with its header, which leaves
    the reconstruction to the one the revision names.

    Raises
    ------
    ValueError
        If the stream carries no header after at most a config.
    """
    config = _first(connection)
    if isinstance(config, xsd.ismrmrdHeader):
        return "", config
    header = _first(connection) if isinstance(config, str) else config
    if not isinstance(header, xsd.ismrmrdHeader):
        found = (
            "nothing"
            if header is None
            else f"a message of type {type(header).__name__}"
        )
        raise ValueError(f"the stream carries {found} where its MRD header belongs")
    return config, header


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


def _enriched(client: Connection, revision: Revision) -> Iterator[Any]:
    """Yield the client's stream up to its close, acquisitions enriched in play order.

    Acquisitions are matched to readouts by position, so the stream carries
    one per readout of the chain, and once the client numbers them, every
    ``scan_counter`` must follow the previous one by one: a gap or a repeat is
    a dropped or duplicated readout that would shift every later row. A stream
    whose counters stay 0 is not numbered and not checked. An acquisition
    flagged ``LAST_IN_MEASUREMENT`` ends the stream, so only the last may
    carry the flag.

    Raises
    ------
    ValueError
        If the stream carries more or fewer acquisitions than the sequence
        plays, a second header, an acquisition flagged
        ``LAST_IN_MEASUREMENT`` before the last, or scan counters that skip or
        repeat one.
    """
    index = 0
    previous = None
    readouts = len(revision.table)
    for item in client:
        if _is_close_marker(item):
            break
        if isinstance(item, xsd.ismrmrdHeader):
            raise ValueError("the stream carries a second MRD header")
        if isinstance(item, ismrmrd.Acquisition):
            if index >= readouts:
                raise ValueError(
                    f"the stream carries more than the {readouts} readouts "
                    f"{revision.directory} plays"
                )
            if item.isFlagSet(ismrmrd.ACQ_LAST_IN_MEASUREMENT) and index + 1 < readouts:
                raise ValueError(
                    f"acquisition {index} is flagged LAST_IN_MEASUREMENT, which ends "
                    f"the stream, and {revision.directory} plays {readouts} readouts"
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
            enrich_acquisition(item, revision.table, index)
            index += 1
        yield item
    if index < readouts:
        raise ValueError(
            f"the stream ended after {index} of the {readouts} readouts "
            f"{revision.directory} plays"
        )


def _forward(client: Connection, worker: Connection, revision: Revision) -> None:
    """Send the client's stream to the worker as it arrives."""
    for item in _enriched(client, revision):
        worker.send(item)


def _record(
    client: Connection,
    header: Any,
    revision: Revision,
    queued: QueueFile,
) -> list[Any]:
    """Store the client's stream until it closes; return what the file cannot hold."""
    queued.write_header(header)
    others: list[Any] = []
    try:
        for item in _enriched(client, revision):
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
