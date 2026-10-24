"""The DICOM receiver a proxy stores a series' images to when their client has gone."""

from __future__ import annotations

__all__ = ["DicomReceiver"]

import logging
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
    """

    host: str
    port: int
    title: str
    calling: str = "PULSERVER"
    timeout: float = 30.0

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
