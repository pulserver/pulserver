"""RTP (Real-Time Processing) connection for PMC (Physiological Motion Correction).

The RtpServer listens on a separate TCP port (default 9003) and accepts a
single connection at a time.  The protocol is symmetric with the regular
MRD streaming protocol:

  Client → Server:
    GADGET_MESSAGE_HEADER          → XML header (navigator-only encoding spaces)
    GADGET_MESSAGE_ISMRMRD_ACQUISITION* (navigator readouts, one per TR)
    GADGET_MESSAGE_CLOSE

  Server → Client:
    GADGET_MESSAGE_PMC_PAYLOAD     → one per received acquisition

PmcPayload wire layout (52 bytes, little-endian):
    float32 × 3   shift      — additive translation delta [m] along scanner X/Y/Z
    float32 × 9   rotation   — 3×3 rotation correction matrix (row-major)
    int32         rescan     — non-zero → RTP app requests TR rescan
"""

__all__ = ["RtpServer", "PmcPayload", "write_pmc_payload"]

import logging
import socket
import struct
import threading
from dataclasses import dataclass

import ismrmrd
import ismrmrd.xsd

from . import constants
from .readers import read_acquisition, read_header

# ---------------------------------------------------------------------------
# PMC payload data type
# ---------------------------------------------------------------------------

_PMC_STRUCT = struct.Struct("<12fi")  # shift(3f) + rotation(9f) + rescan(i) = 52 bytes


@dataclass
class PmcPayload:
    """Rigid-body motion correction parameters returned to PSD."""

    shift: "list[float]" = None  # translation delta [m]   (3 values)
    rotation: "list[float]" = None  # 3×3 rotation matrix, row-major (9 values)
    rescan: int = 0  # non-zero → request TR rescan

    def __post_init__(self):
        if self.shift is None:
            self.shift = [0.0, 0.0, 0.0]
        if self.rotation is None:
            # identity matrix
            self.rotation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]


def write_pmc_payload(writable, payload: PmcPayload) -> None:
    """Serialise *payload* to *writable* as GADGET_MESSAGE_PMC_PAYLOAD."""
    mid = constants.GadgetMessageIdentifier.pack(constants.GADGET_MESSAGE_PMC_PAYLOAD)
    body = _PMC_STRUCT.pack(
        *payload.shift,
        *payload.rotation,
        payload.rescan,
    )
    size = constants.GadgetMessageLength.pack(len(body))
    writable.write(mid + size + body)


def _read_pmc_payload(readable) -> PmcPayload:
    """Deserialise a PMC_PAYLOAD message body from *readable*."""
    size_bytes = readable.read(constants.SIZEOF_GADGET_MESSAGE_LENGTH)
    (size,) = constants.GadgetMessageLength.unpack(size_bytes)
    body = readable.read(size)
    vals = _PMC_STRUCT.unpack(body)
    return PmcPayload(
        shift=list(vals[0:3]),
        rotation=list(vals[3:12]),
        rescan=vals[12],
    )


# ---------------------------------------------------------------------------
# RtpConnection wrapper (single accepted socket)
# ---------------------------------------------------------------------------


class RtpConnection:
    """Wraps a single RTP client socket.

    The server calls ``process(connection, header, metadata)`` on the registered
    handler module.  The connection object exposes:
    - ``__iter__``  → yields ISMRMRD Acquisition objects (navigator readouts)
    - ``send(payload)`` → sends a PmcPayload back

    Usage in a handler::

        def process(connection, config, metadata):
            for acq in connection:
                motion = compute_motion(acq)
                connection.send(PmcPayload(tx=motion[0], ...))
    """

    class _SocketWrapper:
        def __init__(self, sock: socket.socket) -> None:
            self._sock = sock
            self._sock.settimeout(None)

        def read(self, n: int) -> bytes:
            data = self._sock.recv(n, socket.MSG_WAITALL)
            while len(data) < n:
                data += self._sock.recv(n - len(data), socket.MSG_WAITALL)
            return data

        def write(self, b: bytes) -> None:
            self._sock.sendall(b)

        def close(self) -> None:
            self._sock.close()

    def __init__(self, sock: socket.socket) -> None:
        self._sw = RtpConnection._SocketWrapper(sock)
        self._closed = False

    # --- protocol helpers ---------------------------------------------------

    def _read_mid(self) -> int:
        raw = self._sw.read(constants.SIZEOF_GADGET_MESSAGE_IDENTIFIER)
        (mid,) = constants.GadgetMessageIdentifier.unpack(raw)
        return mid

    # --- public API ---------------------------------------------------------

    def __iter__(self):
        """Yield ISMRMRD Acquisition objects until CLOSE."""
        while not self._closed:
            mid = self._read_mid()
            if mid == constants.GADGET_MESSAGE_CLOSE:
                self._closed = True
                return
            elif mid == constants.GADGET_MESSAGE_ISMRMRD_ACQUISITION:
                acq = read_acquisition(self._sw)
                yield acq
            else:
                logging.warning("RtpConnection: unexpected MID %d, skipping", mid)

    def send(self, payload: PmcPayload) -> None:
        """Send a PMC payload to the RTP client."""
        write_pmc_payload(self._sw, payload)

    def close(self) -> None:
        self._sw.close()


# ---------------------------------------------------------------------------
# RtpServer — tcp listener, singleton connection
# ---------------------------------------------------------------------------


class RtpServer:
    """TCP server that accepts a single RTP client connection at a time.

    Parameters
    ----------
    host : str
        Bind address.
    port : int
        Bind port (default 9003).
    handler_module :
        Module with a ``process(connection, config, metadata)`` function.
        The handler receives an :class:`RtpConnection` as *connection*, the
        raw config string as *config*, and the parsed ISMRMRD header as
        *metadata*.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 9003,
        handler_module=None,
    ) -> None:
        self.host = host
        self.port = port
        self.handler_module = handler_module

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))

        logging.info("RTP server listening on %s:%d", self.host, self.port)

    def serve_forever(self) -> None:
        """Block, accepting one connection at a time, until the socket closes."""
        self._socket.listen(1)
        while True:
            try:
                sock, (addr, port) = self._socket.accept()
            except OSError:
                break
            logging.info("RTP client connected from %s:%d", addr, port)
            try:
                self._handle(sock)
            except Exception:
                logging.exception("Error in RTP connection handler")
            logging.info("RTP client disconnected")

    def serve_in_thread(self) -> threading.Thread:
        """Start :meth:`serve_forever` in a daemon thread and return it."""
        t = threading.Thread(target=self.serve_forever, daemon=True, name="RtpServer")
        t.start()
        return t

    def close(self) -> None:
        self._socket.close()

    # ------------------------------------------------------------------

    def _handle(self, sock: socket.socket) -> None:
        conn = RtpConnection(sock)
        try:
            # Read config (handler name) — same protocol as normal MRD stream
            mid = conn._read_mid()
            if mid == constants.GADGET_MESSAGE_CONFIG:
                size_bytes = conn._sw.read(constants.SIZEOF_GADGET_MESSAGE_LENGTH)
                (size,) = constants.GadgetMessageLength.unpack(size_bytes)
                config = (
                    conn._sw.read(size).decode("utf-8", errors="replace").rstrip("\x00")
                )
            else:
                config = "pmcrecon"

            # Read ISMRMRD XML header
            mid = conn._read_mid()
            if mid == constants.GADGET_MESSAGE_HEADER:
                header_xml = read_header(conn._sw)
                try:
                    metadata = ismrmrd.xsd.CreateFromDocument(header_xml)
                except Exception:
                    logging.warning("RTP header not valid MRD XML — passing raw")
                    metadata = header_xml
            else:
                logging.warning("RTP: expected header MID, got %d", mid)
                metadata = None

            if self.handler_module is not None:
                self.handler_module.process(conn, config, metadata)
            else:
                # Default: consume all acquisitions, echo zero-correction back
                for _acq in conn:
                    conn.send(PmcPayload())
        finally:
            conn.close()
