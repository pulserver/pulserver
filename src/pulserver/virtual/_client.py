"""The virtual reconstruction client: one series sent to a reconstruction proxy as the scanner's client sends it, or recorded."""

from __future__ import annotations

__all__ = ["record", "send"]

import socket
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import ismrmrd
import ismrmrd.xsd
import numpy as np

from ..proxy._designs import DESIGN_PARAMETER
from ..recon._runtime.connection import Connection

# Placeholders for the schema's required encoding space, which the proxy
# replaces with the sequence's.
_PLACEHOLDER = 1


def send(
    address: tuple[str, int],
    design: str,
    readouts: Iterable[np.ndarray],
    *,
    frequency_hz: float = 123_200_000.0,
    position_mm: Sequence[float] = (0.0, 0.0, 0.0),
    rotation: np.ndarray | None = None,
    config: str | None = None,
    exam: str | None = None,
    timeout: float = 600.0,
) -> list[Any]:
    """Send one series to the proxy at ``address``; return what it sends back.

    The header carries the design identifier, the resonance frequency, the
    coil count and, with ``exam``, the ``ExamID``; the acquisitions carry the
    samples, scan counters from one, ``LAST_IN_MEASUREMENT`` on the last,
    the field-of-view centre ``position_mm``, in mm along the physical axes,
    and as ``read_dir``, ``phase_dir`` and ``slice_dir`` the columns of the
    prescription's ``rotation`` from logical to physical axes, the identity
    by default. Nothing else of the sequence is sent, as
    :doc:`/user-guide/reconstruction-client` specifies. The reply is the
    images, DICOM datasets and texts in the order they arrive.

    ``readouts`` may be an iterator: the header is sent once its first
    readout is taken, and each readout once the next one is, so that the
    last carries its flag.
    """
    header, acquisitions = _series(
        design, readouts, frequency_hz, position_mm, rotation, exam
    )
    stream = socket.create_connection(address, timeout=timeout)
    connection = Connection(stream)
    try:
        if config is not None:
            connection.send_config(config)
        connection.send_header(header)
        for acquisition in acquisitions:
            connection.send(acquisition)
        connection.send_close()
        return list(connection)
    finally:
        connection.shutdown_close()


def record(
    path: Path | str,
    design: str,
    readouts: Iterable[np.ndarray],
    *,
    frequency_hz: float = 123_200_000.0,
    position_mm: Sequence[float] = (0.0, 0.0, 0.0),
    rotation: np.ndarray | None = None,
    exam: str | None = None,
) -> int:
    """Write one series to an ISMRMRD file as :func:`send` sends it; return the readouts written.

    The header and the acquisitions are those :func:`send` sends, in the
    group ``dataset`` of the file, which is created where it does not exist.
    """
    import ismrmrd.hdf5

    header, acquisitions = _series(
        design, readouts, frequency_hz, position_mm, rotation, exam
    )
    dataset = ismrmrd.hdf5.Dataset(str(path), "dataset", create_if_needed=True)
    try:
        dataset.write_xml_header(header)
        written = 0
        for acquisition in acquisitions:
            dataset.append_acquisition(acquisition)
            written += 1
    finally:
        dataset.close()
    return written


def _series(
    design: str,
    readouts: Iterable[np.ndarray],
    frequency_hz: float,
    position_mm: Sequence[float],
    rotation: np.ndarray | None,
    exam: str | None,
) -> tuple[str, Iterator[ismrmrd.Acquisition]]:
    """Return the header of a series, its first readout taken, and its acquisitions."""
    pending = iter(readouts)
    current = next(pending, None)
    coils = 1 if current is None else int(current.shape[0])
    directions = np.eye(3) if rotation is None else np.asarray(rotation, float)

    def acquisitions() -> Iterator[ismrmrd.Acquisition]:
        counter, samples = 0, current
        while samples is not None:
            following = next(pending, None)
            counter += 1
            acquisition = ismrmrd.Acquisition.from_array(
                np.asarray(samples, np.complex64)
            )
            acquisition.scan_counter = counter
            acquisition.position[:] = position_mm
            acquisition.read_dir[:] = directions[:, 0]
            acquisition.phase_dir[:] = directions[:, 1]
            acquisition.slice_dir[:] = directions[:, 2]
            if following is None:
                acquisition.setFlag(ismrmrd.ACQ_LAST_IN_MEASUREMENT)
            yield acquisition
            samples = following

    return _header(design, coils, frequency_hz, exam), acquisitions()


def _header(design: str, coils: int, frequency_hz: float, exam: str | None) -> str:
    space = ismrmrd.xsd.encodingSpaceType(
        matrixSize=ismrmrd.xsd.matrixSizeType(
            x=_PLACEHOLDER, y=_PLACEHOLDER, z=_PLACEHOLDER
        ),
        fieldOfView_mm=ismrmrd.xsd.fieldOfViewMm(
            x=_PLACEHOLDER, y=_PLACEHOLDER, z=_PLACEHOLDER
        ),
    )
    parameters = [
        ismrmrd.xsd.userParameterStringType(name=DESIGN_PARAMETER, value=design)
    ]
    if exam is not None:
        parameters.append(
            ismrmrd.xsd.userParameterStringType(name="ExamID", value=exam)
        )
    header = ismrmrd.xsd.ismrmrdHeader(
        experimentalConditions=ismrmrd.xsd.experimentalConditionsType(
            H1resonanceFrequency_Hz=round(frequency_hz)
        ),
        acquisitionSystemInformation=ismrmrd.xsd.acquisitionSystemInformationType(
            receiverChannels=coils
        ),
        encoding=[
            ismrmrd.xsd.encodingType(
                encodedSpace=space,
                reconSpace=space,
                encodingLimits=ismrmrd.xsd.encodingLimitsType(),
                trajectory=ismrmrd.xsd.trajectoryType.CARTESIAN,
            )
        ],
        userParameters=ismrmrd.xsd.userParametersType(userParameterString=parameters),
    )
    return header.toXML("utf-8")
