"""Offline replay of queued ISMRMRD sessions.

When all recon slots are busy, pulserver.recon drains an incoming stream to an HDF5
file and writes a ``*.queued.json`` sidecar alongside it.  The
:class:`ReplayWorker` daemon thread picks these up one at a time, acquires a
recon slot, runs the originally requested handler through
:class:`ReplayConnection`, then renames the sidecar to ``*.processed.json``
(or ``*.failed.json`` on error).

Recovery after server crash:  ``*.processing.json`` files left by the previous
process are treated identically to ``*.queued.json`` on startup.
"""

__all__ = ["ReplayConnection", "ReplayWorker", "enqueue"]

import glob
import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

import ismrmrd
import ismrmrd.xsd


if TYPE_CHECKING:
    from .server import Server

# Where the VRE/PSD side keeps per-scan bucket directories.
# Overridable via PULSERVER_BASE environment variable.
_PULSERVER_BASE = os.environ.get("PULSERVER_BASE", "/export/home/sdc/pulserver")


# ---------------------------------------------------------------------------
# enqueue — atomic sidecar writer
# ---------------------------------------------------------------------------


def enqueue(mrd_path: str, handler_name: str, bucket_pid: str | None) -> str:
    """Write a ``*.queued.json`` sidecar next to *mrd_path*, atomically.

    Parameters
    ----------
    mrd_path : str
        Absolute path to the ISMRMRD HDF5 file being queued.
    handler_name : str
        Name of the reconstruction handler to run when a slot becomes free.
    bucket_pid : str or None
        Bucket PID extracted from the ISMRMRD header, for cross-referencing.

    Returns
    -------
    str
        Path of the sidecar file that was written.
    """
    sidecar = {
        "handler": handler_name,
        "mrd_file": mrd_path,
        "bucket_pid": bucket_pid,
        "received_at": datetime.now(timezone.utc).isoformat(),
    }
    base = os.path.splitext(mrd_path)[0]
    sidecar_path = base + ".queued.json"

    # Atomic write: write to a temp file in the same directory, then rename.
    dir_ = os.path.dirname(mrd_path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp.json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(sidecar, f, indent=2)
        os.replace(tmp_path, sidecar_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return sidecar_path


# ---------------------------------------------------------------------------
# ReplayConnection
# ---------------------------------------------------------------------------


class ReplayConnection:
    """Connection-like facade that reads from a saved ISMRMRD HDF5 file.

    Handlers see the same ``__iter__`` / ``send`` / ``socket.write``
    interface as a live :class:`~pulserver.recon.connection.Connection`.
    Outputs (DICOM images, MRD images) are written to *output_dir*.

    Parameters
    ----------
    mrd_path : str
        Path to the ISMRMRD HDF5 file to replay.
    output_dir : str
        Directory where handler outputs are written.
    group : str
        HDF5 dataset group name (default ``"dataset"``).
    """

    class _NoOpSocket:
        """Absorbs ``write()`` calls — replay has no TCP peer."""

        def write(self, data: bytes) -> None:  # noqa: ARG002
            pass

        def close(self) -> None:
            pass

    def __init__(self, mrd_path: str, output_dir: str, group: str = "dataset") -> None:
        self._mrd_path = mrd_path
        self._output_dir = output_dir
        self._group = group
        self.socket = self._NoOpSocket()
        self.is_exhausted = False
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Iteration — yield acquisitions from the HDF5 file
    # ------------------------------------------------------------------

    def __iter__(self):
        dset: ismrmrd.Dataset | None = None
        try:
            dset = ismrmrd.Dataset(self._mrd_path, self._group, create_if_needed=False)
            n = dset.number_of_acquisitions()
            for i in range(n):
                acq = dset.read_acquisition(i)
                yield acq
        except Exception as exc:
            logging.error("ReplayConnection: error reading %s: %s", self._mrd_path, exc)
        finally:
            if dset is not None:
                try:
                    dset.close()
                except Exception:
                    pass
            self.is_exhausted = True

    # ------------------------------------------------------------------
    # Output — write handler results to output_dir
    # ------------------------------------------------------------------

    def send(self, item: Any) -> None:
        """Route handler output to *output_dir* instead of a TCP socket."""
        try:
            from .mrd2dicom import DicomWithName

            if isinstance(item, DicomWithName):
                out_path = os.path.join(self._output_dir, item.filename)
                item.dataset.save_as(out_path)
                logging.info("ReplayConnection: saved DICOM → %s", out_path)
                return
        except Exception:
            pass

        if isinstance(item, ismrmrd.Image):
            out_path = os.path.join(
                self._output_dir, f"image_{item.image_series_index:04d}.h5"
            )
            out_dset = ismrmrd.Dataset(out_path, "dataset", create_if_needed=True)
            out_dset.append_image(f"image_{item.image_series_index}", item)
            out_dset.close()
            logging.info("ReplayConnection: saved MRD image → %s", out_path)
            return

        if isinstance(item, str):
            logging.info("ReplayConnection: handler message: %s", item)
            return

        logging.warning(
            "ReplayConnection: unhandled output type %s", type(item).__name__
        )

    def filter(self, predicate: Any) -> None:
        """No-op: replay does not filter items."""
        pass

    def shutdown_close(self) -> None:
        """No-op: no socket to close."""
        self.is_exhausted = True


# ---------------------------------------------------------------------------
# ReplayWorker
# ---------------------------------------------------------------------------


def _sidecar_suffix_re():
    import re

    return re.compile(r"\.(queued|processing)\.json$")


def _rename_sidecar(src: str, new_suffix: str) -> None:
    """Replace the ``.queued.json`` / ``.processing.json`` suffix atomically."""
    dst = _sidecar_suffix_re().sub(new_suffix, src)
    if dst == src:
        return
    try:
        os.replace(src, dst)
    except OSError as exc:
        logging.error("ReplayWorker: cannot rename %s → %s: %s", src, dst, exc)


class ReplayWorker(threading.Thread):
    """Background daemon thread that processes queued ISMRMRD sessions.

    Scans bucket and output directories for ``*.queued.json`` (and stale
    ``*.processing.json`` from a previous crash) sidecars, processes them
    one at a time through the requested handler, then renames the sidecar
    to ``*.processed.json`` (success) or ``*.failed.json`` (error).

    Parameters
    ----------
    server : Server
        The running :class:`~pulserver.recon.server.Server` instance.  The worker
        calls ``server._slots.acquire()`` / ``server._slots.release()``
        and ``server._resolve_handler()`` / ``server.output_dir``.
    """

    _POLL_INTERVAL = 5.0  # seconds between scans when queue is empty

    def __init__(self, server: "Server") -> None:
        super().__init__(daemon=True, name="pulserver-recon-replay-worker")
        self._server = server
        self._stop_event = threading.Event()

    def run(self) -> None:
        logging.info(
            "ReplayWorker started — polling every %.0fs for queued sessions",
            self._POLL_INTERVAL,
        )
        while not self._stop_event.is_set():
            try:
                processed = self._process_one()
            except Exception:
                logging.exception("ReplayWorker: unexpected error in main loop")
                processed = False
            # If we just processed something, check immediately for the next one;
            # otherwise wait before polling again to reduce idle CPU usage.
            if not processed:
                self._stop_event.wait(timeout=self._POLL_INTERVAL)

    def stop(self) -> None:
        """Signal the worker to exit after the current iteration."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_paths(self) -> list[str]:
        """Glob for all pending sidecar files across known bucket locations."""
        bucket_base = os.path.join(_PULSERVER_BASE, "bucket")
        patterns = [
            os.path.join(bucket_base, "*", "*.queued.json"),
            os.path.join(bucket_base, "*", "*.processing.json"),
        ]
        # Also scan server output_dir for dev / non-VRE environments
        if self._server.output_dir and os.path.isdir(self._server.output_dir):
            patterns += [
                os.path.join(self._server.output_dir, "*.queued.json"),
                os.path.join(self._server.output_dir, "*.processing.json"),
            ]
        found: list[str] = []
        for pat in patterns:
            found.extend(glob.glob(pat))
        return sorted(set(found))

    def _process_one(self) -> bool:
        """Process the oldest pending sidecar.  Returns True if one was found."""
        candidates = self._scan_paths()
        if not candidates:
            return False

        sidecar_path = candidates[0]

        # Load sidecar
        try:
            with open(sidecar_path) as f:
                sidecar = json.load(f)
        except Exception as exc:
            logging.error("ReplayWorker: cannot read sidecar %s: %s", sidecar_path, exc)
            _rename_sidecar(sidecar_path, ".skip.json")
            return True

        mrd_path: str | None = sidecar.get("mrd_file")
        handler_name: str = sidecar.get("handler") or self._server.default_handler
        bucket_pid = sidecar.get("bucket_pid")

        if not mrd_path or not os.path.isfile(mrd_path):
            logging.warning("ReplayWorker: MRD file missing: %s", mrd_path)
            _rename_sidecar(sidecar_path, ".failed.json")
            return True

        output_dir = os.path.join(os.path.dirname(mrd_path), "output")

        # Mark as in-progress (prevents double-processing on concurrent startup)
        _rename_sidecar(sidecar_path, ".processing.json")
        processing_path = _sidecar_suffix_re().sub(".processing.json", sidecar_path)

        logging.info(
            "ReplayWorker: processing queued session  handler=%s  bucket_pid=%s  mrd=%s",
            handler_name,
            bucket_pid,
            mrd_path,
        )

        acquired = False
        try:
            # Parse XML header for the handler
            dset = ismrmrd.Dataset(mrd_path, "dataset", create_if_needed=False)
            xml_header = dset.read_xml_header()
            dset.close()
            try:
                metadata = ismrmrd.xsd.CreateFromDocument(xml_header)
            except Exception:
                metadata = xml_header

            # Block until a recon slot is free
            self._server._slots.acquire()
            acquired = True
            logging.info("ReplayWorker: recon slot acquired  mrd=%s", mrd_path)

            module = self._server._resolve_handler(handler_name)
            replay_conn = ReplayConnection(mrd_path, output_dir)
            module.process(replay_conn, handler_name, metadata)

            _rename_sidecar(processing_path, ".processed.json")
            logging.info("ReplayWorker: completed  mrd=%s", mrd_path)

        except Exception:
            logging.exception("ReplayWorker: error processing %s", mrd_path)
            _rename_sidecar(processing_path, ".failed.json")

        finally:
            if acquired:
                self._server._slots.release()
                logging.info("ReplayWorker: recon slot released")

        return True
