"""TCP MRD servers: the reconstruction proxy, and the reconstruction server a proxy forwards to."""

from __future__ import annotations

__all__ = ["ReconProxy", "ReconServer"]

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
from ..recon._runtime.mrd2dicom import MrdDicomBuilder
from ..recon._runtime.readers import deserialize_config, read_text
from ._designs import Design, DesignCache
from ._enrich import enrich_acquisition, enrich_header
from ._queue import QueueFile
from ._workers import WorkerPool

_PLUGIN_NAME = re.compile(r"[A-Za-z0-9_\-]+")
_log = logging.getLogger("pulserver.proxy")

# A worker spawns, imports its plugin and connects; past this it is not coming.
# Also how long a reconstruction server may take to accept a forwarded series.
_CONNECT_TIMEOUT = 120.0
# How long closing a server waits for each running series.
_WORKER_TIMEOUT = 120.0
# How often the accept loop looks at whether the server is closing.
_ACCEPT_POLL = 0.5
# How long a refused client may send nothing before the proxy stops reading it.
_DRAIN_IDLE = 30.0


class _Listener:
    """Accepts MRD clients on a TCP port, one series per connection and a thread each."""

    def __init__(self, serving: Path | str) -> None:
        self._serving = serving
        self._server: socket.socket | None = None
        self._closing = threading.Event()
        self._stopping = False
        self._threads: list[threading.Thread] = []

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
        _log.info("serving %s on port %d", self._serving, self.port)
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
        """Port the server listens on.

        Raises
        ------
        RuntimeError
            Before :meth:`bind`.
        """
        if self._server is None:
            raise RuntimeError("the server is not bound")
        return int(self._server.getsockname()[1])

    def stop(self) -> None:
        """Make :meth:`serve` return within one accept poll.

        Sets a flag and takes no lock, so a signal handler may call it.
        """
        self._stopping = True

    def _close_listener(self) -> None:
        """Stop accepting clients and wait for the running series."""
        self._closing.set()
        if self._server is not None:
            with contextlib.suppress(OSError):
                self._server.close()
        for thread in self._threads:
            thread.join(timeout=_WORKER_TIMEOUT)

    def _client(self, stream: socket.socket) -> None:
        connection = Connection(stream)
        # The config names a plugin, so the server keeps the text the client
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
        raise NotImplementedError


class ReconProxy(_Listener):
    """Routes each series of an MRD stream to a reconstruction.

    One thread per client connection, which carries one series: an optional
    config text, the header, one acquisition per readout of the chain and a
    ``CLOSE``. The header's ``pulserver_design`` names the design of the store
    the series was played from; its readout table enriches the header and
    every acquisition. The readouts arrive demodulated to the
    prescribed field-of-view centre by the playout
    (:func:`pulserver.ir.prescribe`), and their samples are passed on as
    received. The reconstruction plugin is the design's, falling back to
    the name in the client's config text. A series that breaks this is refused
    with a text naming the reason and a ``CLOSE``, and what the client still
    sends is discarded until it closes the connection.

    The enriched series is reconstructed by a local worker, or, with
    ``forward``, by the MRD server at that address. Whatever reconstructs it,
    its images, DICOM and text go back to the client as they arrive, the
    client's close closes the reconstruction's stream, and the
    reconstruction's close closes the client. A message the proxy has no
    reader for ends what goes back, with a text naming its type. Once the
    client's stream ends, the proxy waits for the reconstruction to close,
    however long it takes, unless ``recon_timeout`` caps it.

    A local series holds a slot for as long as it runs. On a host with GPUs,
    found through ``CUDA_VISIBLE_DEVICES`` or ``nvidia-smi``, each slot holds
    one, ``gpu_slots`` slots per GPU, and the reconstruction reads it as
    ``context.device``. A series that finds every slot busy is queued instead:
    its enriched stream is written to a file of ``queue`` while it arrives, its
    client stays connected, and the file is replayed to a worker and deleted
    once a slot frees. The series of one exam share a directory under the
    system's temporary directory, through which a reconstruction reads what an
    earlier series of the exam stored in its
    :class:`~pulserver.recon.ExamCache`. It is deleted once a header names
    another exam and no series of the exam still runs.

    A forwarded series is sent on as it arrives; the server's own slots and
    queue determine when it is reconstructed. The server is sent a config file message naming
    ``forward_config``, or else the series' reconstruction plugin; the client's
    config text itself is not forwarded.

    Parameters
    ----------
    store
        Directory of designs, as the design calls write it; read only.
    plugins
        Directory of reconstruction plugin files, ``<plugin>.py``; required
        unless the proxy forwards.
    slots
        Series reconstructed at once; derived from memory and the GPUs when
        ``None``.
    gpu_slots
        Series reconstructed at once on each GPU, when ``slots`` is ``None``.
    spares
        Warm worker processes waiting for a series.
    recon_timeout
        Seconds a reconstruction may take after the stream ends before it is
        stopped and the client told so; ``None`` waits for its close. A local
        worker is terminated; a forwarded series' connection is closed.
    queue
        Directory the queued series are written to; a temporary directory,
        removed on :meth:`close`, when ``None``.
    forward
        ``(host, port)`` of the MRD server that reconstructs every series;
        local workers when ``None``.
    forward_config
        Config name sent to that server instead of the reconstruction plugin.
    forward_dicom
        Convert each image the server sends back to DICOM, from the enriched
        header.

    Attributes
    ----------
    workers : WorkerPool | None
        The spares assignments are taken from; ``None`` when forwarding.
    exams : ExamCacheManager | None
        The current exam and its directory; ``None`` when forwarding.
    queue : Path | None
        The queue directory; ``None`` when forwarding.

    Raises
    ------
    ValueError
        If the proxy neither forwards nor has a plugin directory.
    """

    def __init__(
        self,
        store: Path | str,
        plugins: Path | str | None = None,
        *,
        slots: int | None = None,
        gpu_slots: int = 1,
        spares: int = 1,
        recon_timeout: float | None = None,
        queue: Path | str | None = None,
        forward: tuple[str, int] | None = None,
        forward_config: str | None = None,
        forward_dicom: bool = False,
    ) -> None:
        super().__init__(store)
        self.designs = DesignCache(store)
        self.workers = self.exams = self.queue = None
        if forward is not None:
            self._reconstruction: _Workers | _Remote = _Remote(
                forward, forward_config, forward_dicom, recon_timeout
            )
            return
        if plugins is None:
            raise ValueError("a proxy that does not forward needs a plugin directory")
        local = self._reconstruction = _Workers(
            plugins, slots, gpu_slots, spares, recon_timeout, queue
        )
        self.workers, self.exams, self.queue = local.workers, local.exams, local.queue

    def close(self) -> None:
        """Stop accepting clients, wait for running series and release the spares."""
        self._close_listener()
        self._reconstruction.close()

    def _series(self, client: Connection, config: str, header: Any) -> None:
        design = self.designs.resolve(header)
        plugin = design.recon or _config_plugin(config)
        enrich_header(header, design.table)
        _log.info(
            "series on %s: %s, %d readouts",
            design.directory,
            plugin,
            len(design.table),
        )
        self._reconstruction.run(
            client, config, header, plugin, _enriched(client, design)
        )


class ReconServer(_Listener):
    """Reconstructs each series of an MRD stream as it is received.

    The server a :class:`ReconProxy` forwards to: one thread per client
    connection, which carries one series, a config naming its reconstruction
    plugin, the header, the acquisitions and a ``CLOSE``. The config is a
    config file message, whose name is the plugin, or a config text naming it
    as a bare name or under ``parameters.config``. Nothing is enriched. Workers,
    slots, the queue and the exam directories are those of
    :class:`ReconProxy`.

    Parameters
    ----------
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
    queue
        Directory the queued series are written to; a temporary directory,
        removed on :meth:`close`, when ``None``.

    Attributes
    ----------
    workers : WorkerPool
        The spares assignments are taken from.
    exams : ExamCacheManager
        The current exam and its directory.
    queue : Path
        The queue directory.
    """

    def __init__(
        self,
        plugins: Path | str,
        *,
        slots: int | None = None,
        gpu_slots: int = 1,
        spares: int = 1,
        recon_timeout: float | None = None,
        queue: Path | str | None = None,
    ) -> None:
        super().__init__(plugins)
        local = self._reconstruction = _Workers(
            plugins, slots, gpu_slots, spares, recon_timeout, queue
        )
        self.workers, self.exams, self.queue = local.workers, local.exams, local.queue

    def close(self) -> None:
        """Stop accepting clients, wait for running series and release the spares."""
        self._close_listener()
        self._reconstruction.close()

    def _series(self, client: Connection, config: str, header: Any) -> None:
        plugin = _config_plugin(config)
        if not plugin:
            raise ValueError("the config names no reconstruction")
        _log.info("series: %s", plugin)
        self._reconstruction.run(client, config, header, plugin, _received(client))


class _Workers:
    """Local reconstruction: spare workers, slots, the exam directories and the queue."""

    def __init__(
        self,
        plugins: Path | str,
        slots: int | None,
        gpu_slots: int,
        spares: int,
        recon_timeout: float | None,
        queue: Path | str | None,
    ) -> None:
        self._owns_queue = queue is None
        self.queue = (
            Path(tempfile.mkdtemp(prefix="pulserver-queue-"))
            if queue is None
            else Path(queue)
        )
        self.queue.mkdir(parents=True, exist_ok=True)
        self.plugins = Path(plugins)
        self.workers = WorkerPool(spares=spares)
        self.recon_timeout = recon_timeout
        self._exam_root = Path(tempfile.mkdtemp(prefix="pulserver-exams-"))
        self.exams = ExamCacheManager(directory=self._exam_root)
        self._slots = Slots(slot_devices(slots, gpu_slots))
        self._queued = itertools.count(1)

    def run(
        self,
        client: Connection,
        config: str,
        header: Any,
        plugin: str,
        items: Iterator[Any],
    ) -> None:
        """Reconstruct the series ``items`` streams on a worker, in a slot or from the queue."""
        path = self._plugin_path(plugin)
        slot = self._slots.take(wait=False)
        if slot is not None:
            try:
                self._run(client, config, header, path, slot, lambda w: _send(items, w))
            finally:
                self._slots.release(slot)
            return
        queued = QueueFile(self.queue / f"{os.getpid()}-{next(self._queued)}.h5")
        _log.info("queued %s", queued.path.name)
        try:
            others = _record(items, header, queued)
            slot = self._slots.take(wait=True)
            try:
                self._run(
                    client,
                    config,
                    header,
                    path,
                    slot,
                    lambda worker: _replay(queued, others, worker),
                )
            finally:
                self._slots.release(slot)
        finally:
            queued.unlink()

    def close(self) -> None:
        self.workers.close()
        self.exams.close()
        shutil.rmtree(self._exam_root, ignore_errors=True)
        if self._owns_queue:
            shutil.rmtree(self.queue, ignore_errors=True)

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
                _drive(
                    channel,
                    worker,
                    client,
                    feed,
                    self.recon_timeout,
                    f"{plugin.stem} did not finish",
                )

    def _plugin_path(self, plugin: str) -> Path:
        if not plugin:
            raise ValueError("neither the design nor the config names a reconstruction")
        if not _PLUGIN_NAME.fullmatch(plugin):
            raise ValueError(f"invalid plugin name {plugin!r}")
        path = self.plugins / f"{plugin}.py"
        if not path.is_file():
            raise FileNotFoundError(f"no plugin {plugin!r} in {self.plugins}")
        return path


class _Remote:
    """Reconstruction on the MRD server at an address."""

    def __init__(
        self,
        address: tuple[str, int],
        config: str | None,
        dicom: bool,
        recon_timeout: float | None,
    ) -> None:
        self.address = (str(address[0]), int(address[1]))
        self.config = config
        self.dicom = dicom
        self.recon_timeout = recon_timeout

    def run(
        self,
        client: Connection,
        _config: str,
        header: Any,
        plugin: str,
        items: Iterator[Any],
    ) -> None:
        """Stream the series ``items`` carries to the server and its outputs back.

        The client's config text is not forwarded: the server is told the
        configured name, or else ``plugin``.
        """
        name = self.config or plugin
        if not name:
            raise ValueError("neither the design nor the config names a reconstruction")
        convert = MrdDicomBuilder(header) if self.dicom else None
        host, port = self.address
        try:
            stream = socket.create_connection(self.address, timeout=_CONNECT_TIMEOUT)
        except OSError as error:
            raise ConnectionError(
                f"the reconstruction server at {host}:{port} cannot be reached: {error}"
            ) from error
        channel = _RemoteChannel(stream)
        with channel as server:
            server.send_config_file(name)
            server.send_header(header)
            _drive(
                channel,
                server,
                client,
                lambda destination: _send(items, destination),
                self.recon_timeout,
                f"the reconstruction server at {host}:{port} did not finish",
                convert,
            )

    def close(self) -> None:
        pass


def _drive(
    channel: _WorkerChannel | _RemoteChannel,
    reconstruction: Connection,
    client: Connection,
    feed: Callable[[Connection], None],
    timeout: float | None,
    late: str,
    convert: Callable[[Any], Any] | None = None,
) -> None:
    """Feed a reconstruction its series while its outputs are relayed to the client.

    ``late`` opens the text the client receives when the reconstruction is
    still running ``timeout`` seconds after the series ended.
    """
    relay = threading.Thread(
        target=_relay,
        args=(reconstruction, client, convert),
        daemon=True,
        name="relay",
    )
    relay.start()
    try:
        feed(reconstruction)
    except BaseException:
        # A series that cannot be fed whole is not reconstructed.
        channel.terminate()
        relay.join()
        raise
    reconstruction.send_close()
    relay.join(timeout=timeout)
    if relay.is_alive():
        _log.warning("%s within %s s; stopping it", late, timeout)
        # The relay ends with the reconstruction, so the notice cannot
        # interleave with an image it is still sending.
        channel.terminate()
        relay.join()
        with contextlib.suppress(Exception):
            client.send(
                f"pulserver: {late} within {timeout:g} s of the end of the series"
            )


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


class _RemoteChannel:
    """A reconstruction server's end of one forwarded series."""

    def __init__(self, stream: socket.socket) -> None:
        self._stream = stream
        self.connection: Connection | None = None

    def __enter__(self) -> Connection:
        self.connection = Connection(self._stream)
        return self.connection

    def __exit__(self, *exception: object) -> None:
        if self.connection is not None:
            self.connection.shutdown_close()

    def terminate(self) -> None:
        """Close the connection, which ends the relay reading from it."""
        with contextlib.suppress(OSError):
            self._stream.shutdown(socket.SHUT_RDWR)


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
    the reconstruction to the one the design names.

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


def _enriched(client: Connection, design: Design) -> Iterator[Any]:
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
    readouts = len(design.table)
    for item in client:
        if _is_close_marker(item):
            break
        if isinstance(item, xsd.ismrmrdHeader):
            raise ValueError("the stream carries a second MRD header")
        if isinstance(item, ismrmrd.Acquisition):
            if index >= readouts:
                raise ValueError(
                    f"the stream carries more than the {readouts} readouts "
                    f"{design.directory} plays"
                )
            if item.isFlagSet(ismrmrd.ACQ_LAST_IN_MEASUREMENT) and index + 1 < readouts:
                raise ValueError(
                    f"acquisition {index} is flagged LAST_IN_MEASUREMENT, which ends "
                    f"the stream, and {design.directory} plays {readouts} readouts"
                )
            counter = int(item.scan_counter)
            numbered = previous is not None and (previous, counter) != (0, 0)
            if numbered and counter != previous + 1:
                raise ValueError(
                    f"acquisition {index} carries scan counter {counter} after "
                    f"{previous}: the stream no longer matches the readouts "
                    f"{design.directory} plays"
                )
            previous = counter
            enrich_acquisition(item, design.table, index)
            index += 1
        yield item
    if index < readouts:
        raise ValueError(
            f"the stream ended after {index} of the {readouts} readouts "
            f"{design.directory} plays"
        )


def _received(client: Connection) -> Iterator[Any]:
    """Yield the client's stream up to its close, as received."""
    for item in client:
        if _is_close_marker(item):
            break
        yield item


def _send(items: Iterator[Any], destination: Connection) -> None:
    """Send a series to the reconstruction as it arrives."""
    for item in items:
        destination.send(item)


def _record(items: Iterator[Any], header: Any, queued: QueueFile) -> list[Any]:
    """Store a series until it closes; return what the file cannot hold."""
    queued.write_header(header)
    others: list[Any] = []
    try:
        for item in items:
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


def _relay(
    source: Connection,
    client: Connection,
    convert: Callable[[Any], Any] | None = None,
) -> None:
    """Send everything a reconstruction emits back to the client, until its close.

    ``convert`` turns each image into what is sent in its place. A message
    ``source`` has no reader for ends the relay, and the client is told its
    type, because what followed it in the stream cannot be read.
    """
    try:
        for item in source:
            if _is_close_marker(item):
                break
            if convert is not None and isinstance(item, ismrmrd.Image):
                item = convert(item)
            client.send(item)
        if source.unreadable is not None:
            client.send(
                f"pulserver: the reconstruction sent a message of type "
                f"{source.unreadable}, which the proxy cannot read; the rest of "
                "its output is lost"
            )
    except Exception:
        _log.exception("relaying a reconstruction's output failed")
