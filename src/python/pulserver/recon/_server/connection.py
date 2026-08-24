"""Private Gadgetron-compatible MRD connection handling.
Module for handling ISMRMRD connections.

This module provides the Connection class for managing connections to ISMRMRD clients,
including reading and writing various message types with logging and filtering capabilities.
"""

__all__ = ["Connection", "DataSaver", "DummySaver", "build_save_path"]

import contextlib
import logging
import os
import re
import socket
import threading
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any

import ismrmrd

from . import constants
from .mrd2dicom import DicomWithName
from .readers import (
    read,
    read_acquisition,
    read_config_file,
    read_config_text,
    read_dicom,
    read_header,
    read_image,
    read_text,
    read_waveform,
)
from .writers import (
    write_acquisition,
    write_dicom,
    write_image,
    write_text,
    write_waveform,
)


class MessageType(Enum):
    """Known ISMRMRD message types with logging metadata.

    Each member is a ``(mid, display_name, log_period)`` triple.  The
    ``log_period`` controls how often receive/send events are logged
    (every *N*-th message).

    Attributes
    ----------
    mid : int
        Numeric message identifier.
    display_name : str
        Human-readable constant name.
    period : int
        Log every *period*-th message of this type.
    """

    Acquisition = (
        constants.GADGET_MESSAGE_ISMRMRD_ACQUISITION,
        "GADGET_MESSAGE_ISMRMRD_ACQUISITION",
        100,
    )
    DicomWithName = (
        constants.GADGET_MESSAGE_DICOM_WITHNAME,
        "GADGET_MESSAGE_DICOM_WITHNAME",
        100,
    )
    Image = (constants.GADGET_MESSAGE_ISMRMRD_IMAGE, "GADGET_MESSAGE_ISMRMRD_IMAGE", 1)
    str = (
        constants.GADGET_MESSAGE_TEXT,
        "GADGET_MESSAGE_TEXT",
        1,
    )
    Waveform = (
        constants.GADGET_MESSAGE_ISMRMRD_WAVEFORM,
        "GADGET_MESSAGE_ISMRMRD_WAVEFORM",
        100,
    )

    def __init__(self, mid: int, name: str, period: int) -> None:
        self.mid = mid
        self.display_name = name
        self.period = period


# Helper dicts for fast lookup
MID_TO_TYPE = {mt.mid: mt for mt in MessageType}
NAME_TO_TYPE = {mt.name: mt for mt in MessageType}


# ---------------------------------------------------------------------------
# Save-path helper
# ---------------------------------------------------------------------------

_BUCKET_BASE = "/export/home/sdc/pulserver/bucket"


def build_save_path(metadata: Any, fallback_dir: str) -> str:
    """Return the full path for saving one scan's ISMRMRD data.

    Reads ``bucket_pid`` and ``stationName`` from the parsed ISMRMRD header
    when available, and constructs::

        /export/home/sdc/pulserver/bucket/<pid>/mrd_<scanner>_<YYYYMMDDTHHMMSS>.h5

    Falls back to ``fallback_dir/mrd_unknown_<ts>.h5`` when header fields are
    absent (e.g. in development / test environments).

    The directory is created with ``exist_ok=True`` before returning.
    """
    bucket_pid: str | None = None
    scanner_id = "unknown"
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    try:
        if hasattr(metadata, "userParameters") and metadata.userParameters is not None:
            for p in metadata.userParameters.userParameterString:
                if p.name == "bucket_pid":
                    bucket_pid = p.value
                    break
    except Exception:
        pass

    try:
        if (
            hasattr(metadata, "acquisitionSystemInformation")
            and metadata.acquisitionSystemInformation is not None
        ):
            station = metadata.acquisitionSystemInformation.stationName
            if station:
                # Keep only alphanumeric + underscore; strip leading/trailing underscores
                scanner_id = (
                    re.sub(r"[^a-zA-Z0-9_]", "_", station).strip("_") or "unknown"
                )
    except Exception:
        pass

    filename = f"mrd_{scanner_id}_{ts}.h5"

    folder = (
        os.path.join(_BUCKET_BASE, str(bucket_pid))
        if bucket_pid is not None
        else fallback_dir or "/tmp/mrdserver"
    )

    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)


# ---------------------------------------------------------------------------
# DataSaver
# ---------------------------------------------------------------------------


class DataSaver:
    """Persist incoming ISMRMRD data to an HDF5 file.

    Parameters
    ----------
    savedataFile : str
        Filename for the HDF5 file (auto-generated via :func:`build_save_path`
        if empty).
    savedataFolder : str
        Directory for saved files (used only when ``savedataFile`` is empty).
    savedataGroup : str
        HDF5 group name inside the file.
    """

    def __init__(
        self, savedataFile: str, savedataFolder: str, savedataGroup: str
    ) -> None:
        self.savedataFile = savedataFile
        self.savedataFolder = savedataFolder
        self.savedataGroup = savedataGroup
        self.mrdFilePath: str = ""
        self.dset: ismrmrd.Dataset | None = None

    def create_save_file(self) -> None:
        # Ensure save directory exists
        os.makedirs(self.savedataFolder, exist_ok=True)

        # Generate a fallback filename if none was supplied
        if not self.savedataFile:
            self.savedataFile = (
                "mrd_unknown_" + datetime.now().strftime("%Y%m%dT%H%M%S") + ".h5"
            )

        # Full path to the file
        self.mrdFilePath = os.path.join(self.savedataFolder, self.savedataFile)

        # Create HDF5 file to store incoming MRD data
        logging.info(
            "Incoming data will be saved to: '%s' in group '%s'",
            self.mrdFilePath,
            self.savedataGroup,
        )
        self.dset = ismrmrd.Dataset(
            self.mrdFilePath, self.savedataGroup, create_if_needed=True
        )

    def save(self, mid: int, item: Any) -> None:
        """Append *item* to the HDF5 dataset, keyed by message id *mid*."""
        if self.dset is None:
            self.create_save_file()
        self.dset._file.require_group("dataset")
        try:
            if mid == constants.GADGET_MESSAGE_HEADER:
                self.dset.write_xml_header(item.toxml())
            elif mid == constants.GADGET_MESSAGE_ISMRMRD_ACQUISITION:
                self.dset.append_acquisition(item)
            elif mid == constants.GADGET_MESSAGE_ISMRMRD_WAVEFORM:
                self.dset.append_waveform(item)
            elif mid == constants.GADGET_MESSAGE_ISMRMRD_IMAGE:
                self.dset.append_image(f"image_{item.image_series_index}", item)
            # Add more types as needed
            else:
                pass
        except Exception as e:
            logging.error("Failed to save item of type %s: %s", type(item), e)


class DummySaver:
    """No-op saver used when data saving is disabled."""

    def __init__(
        self,
        savedataFile: str = "",
        savedataFolder: str = "",
        savedataGroup: str = "dataset",
    ) -> None:
        pass

    def save(self, mid: int, item: Any) -> None:
        pass


class Connection:
    """
    Represents a connection to an ISMRMRD client.

    This class handles the communication with an ISMRMRD client, providing methods
    to read and write various message types, apply filters, and manage logging.
    """

    class SocketWrapper:
        """Thin wrapper providing ``read``/``write`` over a raw TCP socket."""

        def __init__(self, sock: socket.socket) -> None:
            self.socket = sock
            self.socket.settimeout(None)

        def read(self, nbytes: int) -> bytes:
            """Read exactly *nbytes* from the socket."""
            data = self.socket.recv(nbytes, socket.MSG_WAITALL)
            while len(data) < nbytes:
                data += self.socket.recv(nbytes - len(data), socket.MSG_WAITALL)
            return data

        def write(self, byte_array: bytes) -> None:
            """Send *byte_array* to the peer."""
            self.socket.sendall(byte_array)

        def close(self) -> None:
            """Send a CLOSE message and close the underlying socket."""
            end = constants.GadgetMessageIdentifier.pack(constants.GADGET_MESSAGE_CLOSE)
            self.socket.send(end)
            self.socket.close()

    class Struct:
        def __init__(self, **fields):
            self.__dict__.update(fields)

    def __init__(
        self,
        socket: socket.socket,
        savedata: bool = False,
        savedataFile: str = "",
        savedataFolder: str = "",
        savedataGroup: str = "dataset",
        auto_read_config_header: bool = False,
    ) -> None:
        self.socket = Connection.SocketWrapper(socket)
        self.lock = threading.Lock()

        # Data saving
        if savedata:
            self.saver = DataSaver(savedataFile, savedataFolder, savedataGroup)
        else:
            self.saver = DummySaver(savedataFile, savedataFolder, savedataGroup)

        self.readers = self._default_readers()
        self.writers = Connection._default_writers()
        self.raw_bytes = Connection.Struct(config=None, header=None)
        self.filters = []

        self._recv = {mt.mid: 0 for mt in MessageType}
        self._sent = {mt.name: 0 for mt in MessageType}

        # Connection state
        self.is_exhausted = False
        # Set to True only when the socket is physically closed (shutdown_close).
        # is_exhausted controls reading; _send_closed controls sending.
        # Receiving CLOSE from the client exhausts reading but must NOT prevent
        # the server from sending results back before the server-side CLOSE.
        self._send_closed = False

        # Auto-read config and header if enabled (for server-side)
        if auto_read_config_header:
            self._auto_read_config_header()

    def _auto_read_config_header(self):
        """Auto-read config and header for server connections."""
        try:
            _, self.config = self.next()  # Read config message
            self.raw_bytes.config = b""  # Placeholder
        except StopIteration:
            self.config = None
            logging.info("Connection closed without config")
            return

        try:
            _, self.header = self.next()  # Read header message
            self.raw_bytes.header = b""  # Placeholder
        except StopIteration:
            self.header = None
            logging.info("Connection closed without header")
            return

        # Break if no MRD header was received before a close message (e.g. Gadgetron dependency query)
        if self.header is None and self.is_exhausted:
            logging.info("Connection closed without an MRD header received")
            return

    def __next__(self):
        return self.next()

    def __enter__(self):
        return self

    def __exit__(self, *exception_info):
        self.socket.close()

    def __iter__(self):
        while not self.is_exhausted:
            try:
                _, item = self.next()
                yield item
            except StopIteration:
                return

    def iter_with_mids(self):
        """Iterate yielding ``(message_id, item)`` tuples."""
        while not self.is_exhausted:
            try:
                yield self.next()
            except StopIteration:
                return

    def add_reader(
        self, mid: int, reader: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        """
        Add a reader to the connection's readers.

        Parameters
        ----------
        mid : int
            The ISMRMRD Message ID for which the reader is called.
        reader : callable
            Reader function to be called when `mid` is encountered on the connection.
        *args : Any
            Additional arguments forwarded to the reader when it's called.
        **kwargs : Any
            Additional keyword-arguments forwarded to the reader when it's called.

        Notes
        -----
        Add (or overwrite) a reader to the connection's reader-set. Readers are used to deserialize
        binary ISMRMRD data into usable items.
        """
        self.readers[mid] = lambda readable: reader(readable, *args, **kwargs)

    def add_writer(
        self,
        accepts: Callable[[Any], bool],
        writer: Callable[..., None],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Add a writer to the connection's writers.

        Parameters
        ----------
        accepts : callable
            Predicate used to determine if a writer accepts an item.
        writer : callable
            Writer function to be called when `accepts` predicate returned truthy value.
        *args : Any
            Additional arguments forwarded to the writer when it's called.
        **kwargs : Any
            Additional keyword-arguments forwarded to the writer when it's called.

        Notes
        -----
        Add a writer to the connection's writer-set. Writers are used to serialize items into appropriate
        ISMRMRD binary data.
        """
        self.writers.insert(
            0, (accepts, lambda writable: writer(writable, *args, **kwargs))
        )

    def filter(self, predicate: Callable[[Any], bool] | type) -> None:
        """
        Filters the items that come through the Connection.

        Parameters
        ----------
        predicate : callable or type
            Predicate used when filtering items. Accepts types as well as function predicates.
            Supplying a type is shorthand for `isinstance` based filtering.

        Notes
        -----
        Filters the items returned by `next`, such that only items for which `predicate(item)`
        returns a truthy value is returned. Items not satisfying the predicate will be silently
        sent back to the client.
        """
        if isinstance(predicate, type):
            return self.filters.append(lambda o: isinstance(o, predicate))
        self.filters.append(predicate)

    def send(self, item: Any) -> None:
        """
        Send an item to the client.

        Parameters
        ----------
        item : Any
            Item to be sent.

        Raises
        ------
        TypeError
            If no appropriate writer is found.
        ValueError
            If the connection has been closed for sending.
        """
        if self._send_closed:
            error = ValueError("Cannot send on a closed connection.")
            logging.error(error)
            raise error
        with self.lock:
            for predicate, writer in self.writers:
                if predicate(item):
                    # Update counter and log
                    item_identifier = item.__class__.__name__
                    if item_identifier in NAME_TO_TYPE:
                        msg_type = NAME_TO_TYPE[item_identifier]
                        self._sent[item_identifier] += 1
                        if (self._sent[item_identifier] == 1) or (
                            self._sent[item_identifier] % msg_type.period == 0
                        ):
                            log_message = f"Sending {msg_type.display_name} (total: {self._sent[item_identifier]})"
                            logging.info(f"--> {log_message}")
                    else:
                        log_message = f"Sending item of type: {item_identifier}"
                        logging.info(f"--> {log_message}")

                    return writer(self.socket, item)

            raise TypeError(
                f"No appropriate writer found for item of type '{type(item)}'"
            )

    def peek(self) -> int | None:
        """
        Peek at the next message identifier without consuming it.

        Returns
        -------
        int or None
            The message identifier if available, or None if the connection is exhausted or an error occurs.
        """
        return self._peek_message_identifier()

    def next(self) -> tuple[int, Any]:
        """
        Retrieves the next item available on a connection.

        Returns
        -------
        tuple[int, Any]
            The next message from the connection (along with the corresponding MessageID).

        Raises
        ------
        StopIteration
            If no more items are available.
        TypeError
            If no appropriate reader is found.

        Notes
        -----
        When `next` is called, the next ISMRMRD message id is read from the connection.
        This message id is used to select an appropriate reader, which in turn reads
        the next item from the connection. The item is returned to the caller.

        If the connection is filtered, only items satisfying the predicate is
        returned. Any items not satisfying the predicate is silently returned to the
        client.
        """
        with self.lock:
            if self.is_exhausted:
                raise StopIteration
            try:
                mid, item = self._read_item()

                while not all(pred(item) for pred in self.filters):
                    self.send(item)
                    mid, item = self._read_item()

                # If it's the special acquisition, set exhausted for next call
                if isinstance(item, ismrmrd.Acquisition) and item.isFlagSet(
                    ismrmrd.ACQ_LAST_IN_MEASUREMENT
                ):
                    self.is_exhausted = True
                return mid, item
            except StopIteration:
                self.is_exhausted = True
                raise

    def shutdown_close(self) -> None:
        """
        Gracefully shuts down the socket and closes the connection.

        Sends the protocol-level CLOSE message first so the remote peer (e.g.
        GadgetronClientConnector on the C++ side) receives it before the TCP
        connection is torn down.  Without this ordering the remote connector.wait()
        loop never gets the close token and blocks indefinitely.
        """
        # 1. Send protocol CLOSE before any OS-level shutdown.
        with contextlib.suppress(OSError):
            end = constants.GadgetMessageIdentifier.pack(constants.GADGET_MESSAGE_CLOSE)
            self.socket.socket.sendall(end)
        # 2. Shut down the socket (both directions).
        with contextlib.suppress(OSError):
            self.socket.socket.shutdown(socket.SHUT_RDWR)
        # 3. Close the raw socket.
        with contextlib.suppress(OSError):
            self.socket.socket.close()
        self.is_exhausted = True
        self._send_closed = True
        logging.info("Socket closed")

    def _peek_message_identifier(self) -> int | None:
        try:
            peeked_bytes = self.socket.socket.recv(
                constants.GadgetMessageIdentifier.size, socket.MSG_PEEK
            )
            if len(peeked_bytes) == 0:
                self.is_exhausted = True
                return None
            return constants.GadgetMessageIdentifier.unpack(peeked_bytes)[0]
        except (OSError, ConnectionResetError):
            logging.error("Failed to peek message identifier")
            self.is_exhausted = True
            return None

    def _read_message_identifier(self):
        try:
            return read(self.socket, constants.GadgetMessageIdentifier)
        except ConnectionResetError as err:
            logging.error("Connection closed unexpectedly")
            self.is_exhausted = True
            raise StopIteration from err

    def _read_item(self):
        message_identifier = self._read_message_identifier()

        def unknown_message_identifier(*_):
            logging.error(
                f"Received message (id: {message_identifier}) with no registered readers."
            )
            raise StopIteration

        reader = self.readers.get(message_identifier, unknown_message_identifier)

        # Message handling
        if (
            message_identifier == constants.GADGET_MESSAGE_CLOSE
        ):  # Handle special close message
            item = reader(self.socket)
            self.is_exhausted = True
            return message_identifier, item
        else:  # Handle regular messages
            if message_identifier in MID_TO_TYPE:  # Update counter and log
                msg_type = MID_TO_TYPE[message_identifier]
                self._recv[message_identifier] += 1
                if (self._recv[message_identifier] == 1) or (
                    self._recv[message_identifier] % msg_type.period == 0
                ):
                    log_message = f"Received {msg_type.display_name} (total: {self._recv[message_identifier]})"
                    logging.info(f"<-- {log_message}")
            else:  # Unknown message type
                log_message = f"Received message id: {message_identifier}"
                logging.info(f"<-- {log_message}")
            item = reader(self.socket)
            self.saver.save(message_identifier, item)
            return message_identifier, item

    def stop_iteration(self, _readable: Any = None) -> ismrmrd.Acquisition:
        logging.info("<-- Received GADGET_MESSAGE_CLOSE")
        logging.info("------------------------------------------")
        for mt in MessageType:
            count = self._recv.get(mt.mid, 0)
            if count > 0:
                logging.info("    Total received %-20s: %5d", mt.display_name, count)
        for mt in MessageType:
            count = self._sent.get(mt.name, 0)
            if count > 0:
                logging.info("    Total sent     %-20s: %5d", mt.display_name, count)
        logging.info("------------------------------------------")

        if hasattr(self.saver, "dset") and self.saver.dset is not None:
            logging.debug("Closing file %s", self.saver.dset._file.filename)
            self.saver.dset.close()
            self.saver.dset = None

        # Return special acquisition (TODO: handle this on MRD file level, and return None instead)
        acq = ismrmrd.Acquisition()
        acq.setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
        return acq

    def _default_readers(self) -> dict[int, Callable[..., Any]]:
        """Return the built-in reader dispatch table."""
        return {
            constants.GADGET_MESSAGE_CLOSE: lambda readable: self.stop_iteration(
                readable
            ),
            constants.GADGET_MESSAGE_CONFIG: lambda readable: read_config_text(
                readable
            ),
            constants.GADGET_MESSAGE_FILENAME: lambda readable: read_config_file(
                readable
            ),
            constants.GADGET_MESSAGE_HEADER: lambda readable: read_header(readable),
            constants.GADGET_MESSAGE_ISMRMRD_ACQUISITION: read_acquisition,
            constants.GADGET_MESSAGE_DICOM_WITHNAME: read_dicom,
            constants.GADGET_MESSAGE_ISMRMRD_IMAGE: read_image,
            constants.GADGET_MESSAGE_ISMRMRD_WAVEFORM: read_waveform,
            constants.GADGET_MESSAGE_TEXT: read_text,
        }

    @staticmethod
    def _default_writers() -> list[tuple[Callable[[Any], bool], Callable[..., None]]]:
        """Return the built-in writer dispatch table."""
        return [
            (lambda item: isinstance(item, ismrmrd.Acquisition), write_acquisition),
            (lambda item: isinstance(item, ismrmrd.Waveform), write_waveform),
            (lambda item: isinstance(item, ismrmrd.Image), write_image),
            (lambda item: isinstance(item, DicomWithName), write_dicom),
            (lambda item: isinstance(item, str), write_text),
        ]
