"""MRD streaming connection over a TCP socket, with optional HDF5 capture.

Adapted from the ``Connection`` of
gadgetron-python (Copyright (c) 2019 Gadgetron; MIT, see ``LICENSES/gadgetron-python-MIT.txt``),
and the capture from
python-ismrmrd-server (Copyright (c) 2024 Kelvin Chow; MIT, see ``LICENSES/python-ismrmrd-server-MIT.txt``).
"""

__all__ = ["Connection", "DataSaver", "DummySaver"]

import builtins
import contextlib
import logging
import socket
import threading
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from pathlib import Path
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
    header_document,
    write_acquisition,
    write_config_text,
    write_dicom,
    write_header,
    write_image,
    write_text,
    write_waveform,
)


class MessageType(Enum):
    """Message types :class:`Connection` counts and logs.

    Members are named after the class of the item they carry, which is how
    :meth:`Connection.send` finds them. Each value is ``(mid, display_name,
    period)``: the first message and every ``period``-th one are logged.
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

    def __init__(self, mid: int, name: builtins.str, period: int) -> None:
        self.mid = mid
        self.display_name = name
        self.period = period


MID_TO_TYPE = {mt.mid: mt for mt in MessageType}
NAME_TO_TYPE = {mt.name: mt for mt in MessageType}


class DataSaver:
    """Appends received MRD items to an ISMRMRD HDF5 file, created on first save.

    Parameters
    ----------
    savedataFile
        File name inside ``savedataFolder``; empty generates
        ``mrd_unknown_<YYYYMMDDTHHMMSS>.h5``.
    savedataFolder
        Directory of the file, created when missing.
    savedataGroup
        HDF5 group of the dataset.
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
        """Create the folder and open the dataset for writing."""
        Path(self.savedataFolder).mkdir(parents=True, exist_ok=True)
        if not self.savedataFile:
            self.savedataFile = (
                "mrd_unknown_" + datetime.now().strftime("%Y%m%dT%H%M%S") + ".h5"
            )
        self.mrdFilePath = str(Path(self.savedataFolder) / self.savedataFile)
        logging.info(
            "Incoming data will be saved to: '%s' in group '%s'",
            self.mrdFilePath,
            self.savedataGroup,
        )
        self.dset = ismrmrd.Dataset(
            self.mrdFilePath, self.savedataGroup, create_if_needed=True
        )

    def save(self, mid: int, item: Any) -> None:
        """Append one item by message identifier.

        Headers, acquisitions, waveforms and images are written; other messages
        are ignored. Write errors are logged, not raised.
        """
        if self.dset is None:
            self.create_save_file()
        self.dset._file.require_group("dataset")
        try:
            if mid == constants.GADGET_MESSAGE_HEADER:
                self.dset.write_xml_header(header_document(item))
            elif mid == constants.GADGET_MESSAGE_ISMRMRD_ACQUISITION:
                self.dset.append_acquisition(item)
            elif mid == constants.GADGET_MESSAGE_ISMRMRD_WAVEFORM:
                self.dset.append_waveform(item)
            elif mid == constants.GADGET_MESSAGE_ISMRMRD_IMAGE:
                self.dset.append_image(f"image_{item.image_series_index}", item)
        except Exception as e:
            logging.error("Failed to save item of type %s: %s", type(item), e)


class DummySaver:
    """Saver that discards every item."""

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
    """MRD stream over a socket: iterate to read items, :meth:`send` to write them.

    Reading ends at a CLOSE message, at an acquisition flagged
    ``ACQ_LAST_IN_MEASUREMENT``, at a message with no reader, or when the peer
    resets. CLOSE is delivered as an empty acquisition flagged
    ``ACQ_LAST_IN_MEASUREMENT``. Sending stays open until :meth:`send_close`
    or :meth:`shutdown_close`, so outputs can follow the peer's CLOSE.

    Parameters
    ----------
    socket
        Connected socket; its timeout is cleared.
    savedata
        Append every received item to an ISMRMRD HDF5 file.
    savedataFile, savedataFolder, savedataGroup
        Target of ``savedata``; see :class:`DataSaver`.
    auto_read_config_header
        Read the config and header messages on construction into ``config``
        and ``header``, each ``None`` when the stream ends first.
    """

    class SocketWrapper:
        """Blocking ``read``/``write`` over a socket."""

        def __init__(self, sock: socket.socket) -> None:
            self.socket = sock
            self.socket.settimeout(None)

        def read(self, nbytes: int) -> bytes:
            """Read exactly ``nbytes``.

            Raises
            ------
            ConnectionResetError
                If the peer closes before the whole message arrives.
            """
            data = self.socket.recv(nbytes, socket.MSG_WAITALL)
            while len(data) < nbytes:
                received = self.socket.recv(nbytes - len(data), socket.MSG_WAITALL)
                if not received:
                    raise ConnectionResetError(
                        f"the peer closed {nbytes - len(data)} bytes into a message"
                    )
                data += received
            return data

        def write(self, byte_array: bytes) -> None:
            self.socket.sendall(byte_array)

        def close(self) -> None:
            """Send CLOSE and close the socket."""
            end = constants.GadgetMessageIdentifier.pack(constants.GADGET_MESSAGE_CLOSE)
            self.socket.send(end)
            self.socket.close()

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
        # Reading and sending lock separately: a proxy reads one peer while
        # another thread sends that peer the results of the first.
        self._read_lock = threading.Lock()
        self._send_lock = threading.Lock()

        if savedata:
            self.saver = DataSaver(savedataFile, savedataFolder, savedataGroup)
        else:
            self.saver = DummySaver(savedataFile, savedataFolder, savedataGroup)

        self.readers = self._default_readers()
        self.writers = Connection._default_writers()
        self.filters = []

        self._recv = {mt.mid: 0 for mt in MessageType}
        self._sent = {mt.name: 0 for mt in MessageType}

        # Reading and sending end separately: the peer's CLOSE ends reading,
        # and results still go out until shutdown_close.
        self.is_exhausted = False
        self._send_closed = False

        if auto_read_config_header:
            self._auto_read_config_header()

    def _auto_read_config_header(self):
        try:
            _, self.config = self.next()
        except StopIteration:
            self.config = None
            logging.info("Connection closed without config")
            return

        try:
            _, self.header = self.next()
        except StopIteration:
            self.header = None
            logging.info("Connection closed without header")
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

    def add_reader(
        self, mid: int, reader: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        """Register ``reader(source, *args, **kwargs)`` for message ``mid``, replacing any other."""
        self.readers[mid] = lambda readable: reader(readable, *args, **kwargs)

    def add_writer(
        self,
        accepts: Callable[[Any], bool],
        writer: Callable[..., None],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Register ``writer(destination, item, *args, **kwargs)`` ahead of the existing writers.

        The writer handles items for which ``accepts(item)`` is true.
        """
        self.writers.insert(
            0, (accepts, lambda writable: writer(writable, *args, **kwargs))
        )

    def filter(self, predicate: Callable[[Any], bool] | type) -> None:
        """Pass on only items ``predicate`` accepts; send the others back to the peer.

        A type is shorthand for an ``isinstance`` check.
        """
        if isinstance(predicate, type):
            return self.filters.append(lambda o: isinstance(o, predicate))
        self.filters.append(predicate)

    def send(self, item: Any) -> None:
        """Send one item with the first writer that accepts it.

        Raises
        ------
        TypeError
            If no writer accepts the item.
        ValueError
            After :meth:`shutdown_close`.
        """
        if self._send_closed:
            error = ValueError("Cannot send on a closed connection.")
            logging.error(error)
            raise error
        with self._send_lock:
            for predicate, writer in self.writers:
                if predicate(item):
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
        """Return the next message identifier without consuming it.

        Returns ``None``, and ends reading, when the peer closed or the read
        failed.
        """
        return self._peek_message_identifier()

    def next(self) -> tuple[int, Any]:
        """Return the next ``(message identifier, item)``.

        Items a filter rejects are sent back to the peer and skipped.

        Raises
        ------
        StopIteration
            When reading has ended.
        """
        with self._read_lock:
            if self.is_exhausted:
                raise StopIteration
            try:
                mid, item = self._read_item()

                while not all(pred(item) for pred in self.filters):
                    self.send(item)
                    mid, item = self._read_item()

                if isinstance(item, ismrmrd.Acquisition) and item.isFlagSet(
                    ismrmrd.ACQ_LAST_IN_MEASUREMENT
                ):
                    self.is_exhausted = True
                return mid, item
            except StopIteration:
                self.is_exhausted = True
                raise
            except ConnectionResetError as error:
                logging.error("Connection closed mid-message: %s", error)
                self.is_exhausted = True
                raise StopIteration from error

    def send_config(self, text: str) -> None:
        """Send a config text message, which a peer reads ahead of the header."""
        self._send_raw(write_config_text, text)

    def send_header(self, header: Any) -> None:
        """Send an MRD XML header message, from a parsed header or its document."""
        self._send_raw(write_header, header)

    def send_close(self) -> None:
        """Send CLOSE and stop sending; the socket stays open, so results still arrive.

        Ends what this side sends, as :meth:`shutdown_close` does, without
        ending what it reads.
        """
        with self._send_lock:
            if self._send_closed:
                return
            self._send_closed = True
            with contextlib.suppress(OSError):
                self.socket.write(
                    constants.GadgetMessageIdentifier.pack(
                        constants.GADGET_MESSAGE_CLOSE
                    )
                )
        logging.info("--> Sending GADGET_MESSAGE_CLOSE")

    def shutdown_close(self) -> None:
        """Send CLOSE, then shut down and close the socket; ends reading and sending.

        CLOSE goes first: the Orchestra client connector waits for it and does
        not return on a bare TCP close. A connection already closed for sending
        does not send a second CLOSE.
        """
        if not self._send_closed:
            with contextlib.suppress(OSError):
                end = constants.GadgetMessageIdentifier.pack(
                    constants.GADGET_MESSAGE_CLOSE
                )
                self.socket.socket.sendall(end)
        with contextlib.suppress(OSError):
            self.socket.socket.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(OSError):
            self.socket.socket.close()
        self.is_exhausted = True
        self._send_closed = True
        logging.info("Socket closed")

    def _send_raw(self, writer: Callable[..., None], item: Any) -> None:
        if self._send_closed:
            raise ValueError("Cannot send on a closed connection.")
        with self._send_lock:
            writer(self.socket, item)

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

        if message_identifier == constants.GADGET_MESSAGE_CLOSE:
            item = reader(self.socket)
            self.is_exhausted = True
            return message_identifier, item
        if message_identifier in MID_TO_TYPE:
            msg_type = MID_TO_TYPE[message_identifier]
            self._recv[message_identifier] += 1
            if (self._recv[message_identifier] == 1) or (
                self._recv[message_identifier] % msg_type.period == 0
            ):
                log_message = f"Received {msg_type.display_name} (total: {self._recv[message_identifier]})"
                logging.info(f"<-- {log_message}")
        else:
            logging.info(f"<-- Received message id: {message_identifier}")
        item = reader(self.socket)
        self.saver.save(message_identifier, item)
        return message_identifier, item

    def stop_iteration(self, _readable: Any = None) -> ismrmrd.Acquisition:
        """Read CLOSE: log the message totals, close the capture file, return the end marker.

        The marker is an empty acquisition flagged ``ACQ_LAST_IN_MEASUREMENT``.
        """
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

        acq = ismrmrd.Acquisition()
        acq.setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
        return acq

    def _default_readers(self) -> dict[int, Callable[..., Any]]:
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
        return [
            (lambda item: isinstance(item, ismrmrd.Acquisition), write_acquisition),
            (lambda item: isinstance(item, ismrmrd.Waveform), write_waveform),
            (lambda item: isinstance(item, ismrmrd.Image), write_image),
            (lambda item: isinstance(item, DicomWithName), write_dicom),
            (lambda item: isinstance(item, str), write_text),
        ]
