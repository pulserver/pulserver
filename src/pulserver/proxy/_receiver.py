"""The DICOM receiver a proxy stores a series' images to when their client has gone."""

from __future__ import annotations

__all__ = ["DicomReceiver"]

import io
import logging
import socket
import struct
from collections.abc import Sequence
from dataclasses import dataclass

from ..recon._runtime.mrd2dicom import DicomWithName

_log = logging.getLogger("pulserver.proxy")

# What a receiver answers for an image it stored: success, and the warnings of
# a store that kept the image (coerced, elements discarded, data set not
# matching its SOP class).
_STORED = {0x0000, 0xB000, 0xB006, 0xB007}


@dataclass(frozen=True)
class DicomReceiver:
    """A DICOM storage service: where images go by C-STORE.

    Parameters
    ----------
    host, port
        Address of the receiver.
    title
        Application entity title the receiver answers to.
    calling
        Application entity title the proxy calls as, which the receiver may
        have to be told to accept.
    timeout
        Seconds the association, and each store, may take.
    viewer_port
        Port on the receiver's host of a viewer that shows each stored image,
        sent to it as an ``AVR1`` message: the four bytes ``AVR1``, then -1
        three times and the file's length as big-endian 32-bit integers, then
        the image as a DICOM file. The viewer answers a big-endian 32-bit
        status, 0 when shown, and a line. ``None`` sends nothing.
    """

    host: str
    port: int
    title: str
    calling: str = "PULSERVER"
    timeout: float = 30.0
    viewer_port: int | None = None

    def __str__(self) -> str:
        return f"{self.title} at {self.host}:{self.port} called as {self.calling}"

    def store(self, images: Sequence[DicomWithName]) -> int:
        """Store ``images`` over one association; return how many the receiver kept.

        An image the receiver kept already is sent again: a receiver keeps one
        instance per ``SOPInstanceUID``. Nothing raises: a receiver that cannot
        be reached, or refuses, is logged and has stored none.
        """
        datasets = [image.dset for image in images if image.dset is not None]
        if not datasets:
            return 0
        try:
            from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian
            from pynetdicom import AE
        except ImportError:
            _log.error("pynetdicom is not installed: no image is stored to %s", self)
            return 0
        ae = AE(ae_title=self.calling)
        for sop_class in sorted({str(dataset.SOPClassUID) for dataset in datasets}):
            ae.add_requested_context(
                sop_class, [ExplicitVRLittleEndian, ImplicitVRLittleEndian]
            )
        ae.acse_timeout = ae.dimse_timeout = ae.network_timeout = self.timeout
        try:
            association = ae.associate(self.host, self.port, ae_title=self.title)
        except Exception as error:
            _log.error("%s cannot be reached: %s", self, error)
            return 0
        if not association.is_established:
            _log.error("%s refused the association", self)
            return 0
        stored = 0
        try:
            for dataset in datasets:
                try:
                    status = association.send_c_store(dataset)
                except Exception as error:
                    _log.error("storing an image to %s failed: %s", self, error)
                    break
                code = getattr(status, "Status", None)
                if code is not None and int(code) in _STORED:
                    stored += 1
                    self.show(dataset)
                else:
                    _log.error(
                        "%s did not store image %s: %s",
                        self,
                        dataset.get("InstanceNumber", "?"),
                        "no status" if code is None else f"status 0x{int(code):04X}",
                    )
        finally:
            association.release()
        return stored

    def show(self, dataset) -> bool:
        """Send ``dataset`` to the viewer at ``viewer_port``; return whether it was shown.

        Nothing raises: a viewer that cannot be reached, or refuses, is logged.
        """
        if self.viewer_port is None:
            return False
        buffer = io.BytesIO()
        dataset.save_as(buffer, enforce_file_format=True)
        image = buffer.getvalue()
        request = b"AVR1" + struct.pack(">iiiI", -1, -1, -1, len(image)) + image
        try:
            with socket.create_connection(
                (self.host, self.viewer_port), timeout=5.0
            ) as connection:
                connection.sendall(request)
                connection.shutdown(socket.SHUT_WR)
                reply = b""
                while chunk := connection.recv(512):
                    reply += chunk
        except OSError as error:
            _log.error(
                "the viewer at %s:%d cannot be reached: %s",
                self.host,
                self.viewer_port,
                error,
            )
            return False
        if len(reply) < 4:
            _log.error(
                "the viewer at %s:%d did not answer", self.host, self.viewer_port
            )
            return False
        status = struct.unpack(">I", reply[:4])[0]
        line = reply[4:].decode(errors="replace").strip()
        if status == 0:
            _log.info("viewer: %s", line)
        else:
            _log.error(
                "the viewer at %s:%d did not show image %s: %s",
                self.host,
                self.viewer_port,
                dataset.get("InstanceNumber", "?"),
                line,
            )
        return status == 0
