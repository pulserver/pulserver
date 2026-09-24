"""One-shot reconstruction workers, each taken from a pre-warmed spare.

A worker serves one series and exits: importing a reconstruction stack costs
seconds and holds device memory for as long as the process lives. A spare pays
that cost before a series arrives and waits on a pipe for its assignment.
"""

from __future__ import annotations

__all__ = ["WorkerPool"]

import contextlib
import logging
import multiprocessing
import socket
import sys
import threading
from dataclasses import dataclass
from multiprocessing.connection import Connection as Pipe
from multiprocessing.context import SpawnProcess
from pathlib import Path

from ..recon import ExamCache, ReconContext, load_plugin
from ..recon._runtime.application import run_application
from ..recon._runtime.connection import Connection
from ..recon._runtime.exam import resolve_exam_id

_log = logging.getLogger("pulserver.vre")


@dataclass(frozen=True)
class _Spare:
    process: SpawnProcess
    pipe: Pipe


class WorkerPool:
    """Spawned reconstruction workers, kept warm until a series needs one.

    Workers are not daemonic, so a reconstruction may start processes of its
    own; :meth:`close` terminates the workers still running.

    Parameters
    ----------
    spares
        Warm processes waiting for an assignment. :meth:`assign` blocks while
        there is none.
    """

    def __init__(self, *, spares: int = 1) -> None:
        self._context = multiprocessing.get_context("spawn")
        self._condition = threading.Condition()
        self._spares: list[_Spare] = []
        self._assigned: list[SpawnProcess] = []
        self._closed = False
        for _ in range(max(1, spares)):
            self._start()

    def assign(
        self,
        plugin: Path | str,
        socket_path: Path | str,
        exam_directory: Path | str | None = None,
        device: str | None = None,
    ) -> SpawnProcess:
        """Hand a spare the plugin to run, the socket to reach the proxy on, its exam directory and device.

        Starts a replacement spare before returning, so the next series does not
        wait for an import. The series' :class:`~pulserver.recon.ExamCache`
        shares ``exam_directory`` with the exam's other series; ``None`` keeps
        it to this series. ``device`` is the series' ``context.device``.

        Raises
        ------
        RuntimeError
            After :meth:`close`.
        """
        with self._condition:
            while not self._spares and not self._closed:
                self._condition.wait()
            if self._closed:
                raise RuntimeError("the worker pool is closed")
            spare = self._spares.pop(0)
            self._assigned.append(spare.process)
        exam = None if exam_directory is None else str(exam_directory)
        spare.pipe.send((str(socket_path), str(plugin), exam, device))
        spare.pipe.close()
        self._start()
        return spare.process

    def spare_pids(self) -> tuple[int, ...]:
        """Process identifiers of the spares waiting for an assignment."""
        with self._condition:
            return tuple(spare.process.pid for spare in self._spares)

    def close(self) -> None:
        """Release every spare, terminate the workers still running, and refuse further assignments."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            spares, self._spares = self._spares, []
            assigned, self._assigned = self._assigned, []
            self._condition.notify_all()
        for spare in spares:
            with contextlib.suppress(OSError, ValueError):
                spare.pipe.send(None)
                spare.pipe.close()
            spare.process.join(timeout=10)
            if spare.process.is_alive():
                spare.process.terminate()
        for process in assigned:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

    def _start(self) -> None:
        # Reaps the workers that have finished their series.
        multiprocessing.active_children()
        with self._condition:
            self._assigned = [
                process for process in self._assigned if process.is_alive()
            ]
        parent, child = self._context.Pipe()
        process = self._context.Process(target=_warm, args=(child,), daemon=False)
        process.start()
        child.close()
        with self._condition:
            if self._closed:
                parent.close()
                process.terminate()
                return
            self._spares.append(_Spare(process, parent))
            self._condition.notify()


def _warm(pipe: Pipe) -> None:
    """Import the reconstruction engine, wait for an assignment, serve that one series, and exit.

    A host without bartorch reconstructs with whatever its plugins import.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    with contextlib.suppress(ImportError):
        import bartorch  # noqa: F401
    try:
        assignment = pipe.recv()
    except EOFError:
        return
    finally:
        pipe.close()
    if assignment is None:
        return
    sys.exit(_reconstruct(*assignment))


def _reconstruct(
    socket_path: str,
    plugin_path: str,
    exam_directory: str | None = None,
    device: str | None = None,
) -> int:
    """Drive one series over the proxy's socket; the exit status of the worker."""
    stream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stream.connect(socket_path)
    connection = Connection(stream, auto_read_config_header=True)
    exam = ExamCache(
        resolve_exam_id(connection.header) or ("series", socket_path), exam_directory
    )
    status = 0
    try:
        run_application(
            load_plugin(plugin_path),
            connection,
            ReconContext(
                header=connection.header,
                exam=exam,
                config=connection.config,
                device=device,
            ),
        )
    except Exception as error:
        _log.exception("reconstruction failed")
        status = 1
        with contextlib.suppress(Exception):
            connection.send(f"pulserver: {Path(plugin_path).stem} failed: {error}")
    finally:
        exam.close()
        connection.shutdown_close()
    return status
