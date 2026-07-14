"""MRD streaming server with dynamic handler loading."""

__all__ = ["Server"]

import importlib
import importlib.util
import logging
import os
import signal
import socket
import sys
import threading
import time
from types import ModuleType
from typing import Any

import ismrmrd
import ismrmrd.xsd

from . import constants
from .concurrency import compute_max_concurrent
from .connection import Connection, DataSaver, build_save_path
from .rtp_connection import RtpServer


class Server:
    """TCP server that accepts ISMRMRD/MRD streaming connections.

    Each incoming connection is handled in a separate thread.  The server
    reads the config message to determine which handler module to load,
    then delegates processing to ``module.process(connection, config, metadata)``.

    Parameters
    ----------
    host : str
        Bind address (default ``"0.0.0.0"``).
    port : int
        Bind port (default ``9002``).
    default_handler : str
        Fallback handler module name when config is empty or unknown.
    output_dir : str
        Directory for saved MRD data and per-connection logs.
    save_data : bool
        Whether to persist incoming MRD data to HDF5.
    handler_dirs : list[str] or None
        Additional directories to search for handler ``.py`` files.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9002,
        default_handler: str = "savedataonly",
        output_dir: str = "",
        save_data: bool = False,
        handler_dirs: list[str] | None = None,
        rtp_port: int | None = None,
        rtp_handler: str = "pmcrecon",
        max_concurrent_recons: int | None = None,
        per_recon_gb: float = 48.0,
        idle_timeout_s: int = 0,
    ) -> None:
        self.idle_timeout_s = idle_timeout_s
        self.host = host
        self.port = port
        self.default_handler = default_handler
        self.output_dir = output_dir
        self.save_data = save_data
        self.handler_dirs = handler_dirs or []
        self.rtp_port = rtp_port
        self.rtp_handler = rtp_handler
        self._rtp_server: RtpServer | None = None

        # Concurrency limit — auto-detected from available RAM or overridden
        self._max_slots = compute_max_concurrent(
            per_recon_gb=per_recon_gb,
            override=max_concurrent_recons,
        )
        self._slots = threading.BoundedSemaphore(self._max_slots)

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))

        logging.info(
            "MRD server listening on %s:%d  (default handler: %s, max recon slots: %d)",
            self.host,
            self.port,
            self.default_handler,
            self._max_slots,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_rtp_server(self) -> None:
        """Start the RTP PMC server in a background daemon thread (if rtp_port is set)."""
        if self.rtp_port is None:
            return
        handler_mod = self._try_import(self.rtp_handler)
        for d in self.handler_dirs:
            if handler_mod is not None:
                break
            import os

            path = os.path.join(d, self.rtp_handler + ".py")
            if os.path.isfile(path):
                handler_mod = self._load_from_file(self.rtp_handler, path)
        self._rtp_server = RtpServer(
            host=self.host,
            port=self.rtp_port,
            handler_module=handler_mod,
        )
        self._rtp_server.serve_in_thread()
        logging.info(
            "RTP PMC server started on port %d (handler: %s)",
            self.rtp_port,
            self.rtp_handler,
        )

    def serve(self) -> None:
        """Block and accept connections until interrupted."""
        self._socket.listen(5)

        # Graceful shutdown on SIGTERM / SIGINT
        def _shutdown(signum, frame):
            logging.info("Received signal %d — shutting down", signum)
            self._socket.close()
            sys.exit(0)

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

        # Start the offline-replay worker (daemon thread — exits when server exits)
        from .replay import ReplayWorker

        self._replay_worker = ReplayWorker(self)
        self._replay_worker.start()

        # Idle timeout: poll accept() every 30 s and exit if no connection
        # has arrived within idle_timeout_s seconds.
        idle_check_interval = 30  # seconds
        last_activity = time.monotonic()
        if self.idle_timeout_s > 0:
            self._socket.settimeout(idle_check_interval)

        while True:
            try:
                sock, (remote_addr, remote_port) = self._socket.accept()
                last_activity = time.monotonic()
            except TimeoutError:
                # socket.settimeout fires this on each poll interval
                if self.idle_timeout_s > 0:
                    idle = time.monotonic() - last_activity
                    if idle >= self.idle_timeout_s:
                        logging.info(
                            "Idle timeout reached (%.0f s) — shutting down", idle
                        )
                        self._socket.close()
                        sys.exit(0)
                continue
            except OSError:
                break

            logging.info("Accepted connection from %s:%d", remote_addr, remote_port)
            t = threading.Thread(
                target=self._handle_connection,
                args=(sock,),
                daemon=True,
            )
            t.start()

    # ------------------------------------------------------------------
    # Connection handler (runs in its own thread)
    # ------------------------------------------------------------------

    def _handle_connection(self, sock: socket.socket) -> None:
        # Create connection with savedata=False; the saver is configured explicitly
        # after metadata is available so that the bucket-based save path can be built.
        connection = Connection(
            sock,
            savedata=False,
            savedataGroup="dataset",
        )

        config: str = "<unknown>"
        acquired = False

        try:
            # 1) Config message (handler module name)
            _, config = next(connection)
            if config is None and connection.is_exhausted:
                logging.info("Connection closed without data")
                return

            # 2) XML header
            _, metadata_xml = next(connection)
            if metadata_xml is None and connection.is_exhausted:
                logging.info("Connection closed without MRD header")
                return

            # Parse header
            try:
                metadata = ismrmrd.xsd.CreateFromDocument(metadata_xml)
                if (
                    metadata.acquisitionSystemInformation.systemFieldStrength_T
                    is not None
                ):
                    logging.info(
                        "Data from %s %s at %1.1fT",
                        metadata.acquisitionSystemInformation.systemVendor,
                        metadata.acquisitionSystemInformation.systemModel,
                        metadata.acquisitionSystemInformation.systemFieldStrength_T,
                    )
            except Exception:
                logging.warning("Metadata is not valid MRD XML — passing as text")
                metadata = metadata_xml

            # 3) Try to acquire a recon slot (non-blocking)
            acquired = self._slots.acquire(blocking=False)
            if not acquired:
                logging.warning(
                    "All %d recon slot(s) busy — queuing connection to disk  handler=%s",
                    self._max_slots,
                    config,
                )
                self._drain_and_queue(connection, config, metadata, metadata_xml)
                return

            # 4) Configure data saver with the correct bucket-based path
            if self.save_data:
                save_path = build_save_path(metadata, self.output_dir)
                connection.saver = DataSaver(
                    savedataFile=os.path.basename(save_path),
                    savedataFolder=os.path.dirname(save_path),
                    savedataGroup="dataset",
                )
                connection.saver.create_save_file()
                if metadata_xml is not None and connection.saver.dset is not None:
                    try:
                        connection.saver.dset.write_xml_header(metadata_xml.toXML())
                    except Exception as exc:
                        logging.warning(
                            "Could not write XML header to save file: %s", exc
                        )

            # 5) Resolve and run handler
            module = self._resolve_handler(config)
            logging.info(
                "Starting handler '%s'  [slot %d/%d]",
                module.__name__,
                self._max_slots - self._slots._value,  # approximate occupancy
                self._max_slots,
            )
            module.process(connection, config, metadata)

        except Exception:
            logging.exception("Error handling connection  handler=%s", config)

        finally:
            if acquired:
                self._slots.release()
                logging.info("Recon slot released  handler=%s", config)
            connection.shutdown_close()
            if hasattr(connection.saver, "dset") and connection.saver.dset is not None:
                try:
                    connection.saver.dset.close()
                except Exception:
                    pass
            if (
                hasattr(connection.saver, "mrdFilePath")
                and connection.saver.mrdFilePath
            ):
                logging.info("Incoming data saved at %s", connection.saver.mrdFilePath)

    # ------------------------------------------------------------------
    # Overflow: drain to disk and queue for later replay
    # ------------------------------------------------------------------

    def _drain_and_queue(
        self,
        connection: Connection,
        config: str,
        metadata: Any,
        metadata_xml: str,
    ) -> None:
        """Drain *connection* to an HDF5 file and write a ``.queued.json`` sidecar.

        Called when all recon slots are busy.  The incoming stream is still
        fully consumed so the client (VRE) sees a clean connection lifecycle.
        A :class:`~pulserver.recon.replay.ReplayWorker` will pick up the sidecar
        and replay the session through *config* when a slot becomes available.
        """
        from .replay import enqueue

        save_path = build_save_path(metadata, self.output_dir)

        # Extract bucket_pid for the sidecar (informational)
        bucket_pid: str | None = None
        try:
            if (
                hasattr(metadata, "userParameters")
                and metadata.userParameters is not None
            ):
                for p in metadata.userParameters.userParameterString:
                    if p.name == "bucket_pid":
                        bucket_pid = p.value
                        break
        except Exception:
            pass

        saver = DataSaver(
            savedataFile=os.path.basename(save_path),
            savedataFolder=os.path.dirname(save_path),
            savedataGroup="dataset",
        )
        try:
            saver.create_save_file()
            # Write the XML header that was already read from the stream
            if metadata_xml is not None and saver.dset is not None:
                try:
                    saver.dset.write_xml_header(metadata_xml.toXML())
                except Exception as exc:
                    logging.warning(
                        "Could not write XML header to queued file: %s", exc
                    )
            # Drain all remaining acquisitions / waveforms from the stream
            for mid, item in connection.iter_with_mids():
                saver.save(mid, item)
        finally:
            if saver.dset is not None:
                try:
                    saver.dset.close()
                except Exception:
                    pass

        sidecar_path = enqueue(save_path, config, bucket_pid)
        logging.info(
            "Queued to disk  mrd=%s  handler=%s  bucket_pid=%s  sidecar=%s",
            save_path,
            config,
            bucket_pid,
            sidecar_path,
        )

    # ------------------------------------------------------------------
    # Handler resolution
    # ------------------------------------------------------------------

    def _resolve_handler(self, config: str) -> ModuleType:
        """Resolve a handler module from a *config* string.

        Resolution order:

        1. ``importlib.import_module(config)`` (installed packages / sys.path).
        2. Search ``handler_dirs`` for ``<config>.py`` with a ``process`` callable.
        3. Fall back to ``self.default_handler``.

        Parameters
        ----------
        config : str
            Handler name sent by the client in the CONFIG message.

        Returns
        -------
        ModuleType
            A module exposing a ``process(connection, config, metadata)`` callable.
        """
        # Direct absolute file path — load without the import system.
        # This handles custom recon scripts whose path is sent verbatim by the C++ client
        # (e.g. /workspace/.../recon/recon42.py).  Using _load_from_file avoids the
        # import-system which would mis-interpret a filesystem path as a module name.
        if config and os.path.isabs(config) and config.endswith('.py') and os.path.isfile(config):
            name = os.path.splitext(os.path.basename(config))[0]
            mod = self._load_from_file(name, config)
            if mod is not None:
                return mod
            logging.warning(
                "Handler file '%s' could not be loaded — falling back to '%s'",
                config,
                self.default_handler,
            )

        # Fast path: try standard import
        if config and config != "null":
            mod = self._try_import(config)
            if mod is not None:
                return mod

            # Search handler directories
            for d in self.handler_dirs:
                path = os.path.join(d, config + ".py")
                if os.path.isfile(path):
                    mod = self._load_from_file(config, path)
                    if mod is not None:
                        return mod

            logging.warning(
                "Handler '%s' not found — falling back to '%s'",
                config,
                self.default_handler,
            )

        # Fallback
        mod = self._try_import(self.default_handler)
        if mod is not None:
            return mod

        for d in self.handler_dirs:
            path = os.path.join(d, self.default_handler + ".py")
            if os.path.isfile(path):
                mod = self._load_from_file(self.default_handler, path)
                if mod is not None:
                    return mod

        # Last resort: inline drain
        return _NullHandler

    @staticmethod
    def _try_import(name: str) -> ModuleType | None:
        """Import *name* and return it if it has a ``process`` callable."""
        try:
            mod = importlib.import_module(name)
            if hasattr(mod, "process") and callable(mod.process):
                return mod
        except ImportError:
            pass
        return None

    @staticmethod
    def _load_from_file(name: str, path: str) -> ModuleType | None:
        """Load a module from *path* and return it if it has ``process``."""
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "process") and callable(mod.process):
                return mod
        except Exception as exc:
            logging.error("Failed to load '%s' from %s: %s", name, path, exc)
        return None


class _NullHandler:
    """Drain all messages without processing."""

    __name__ = "null"

    @staticmethod
    def process(connection, config, metadata):
        logging.info("Null handler — draining connection")
        try:
            for _msg in connection:
                pass
        finally:
            end = constants.GadgetMessageIdentifier.pack(constants.GADGET_MESSAGE_CLOSE)
            connection.socket.write(end)
