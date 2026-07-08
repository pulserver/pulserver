"""Simple 2D FFT reconstruction handler.

Accumulates k-space lines per slice, applies 2D IFFT + root-sum-of-squares
coil combination, and sends back ISMRMRD images.
"""

import logging
from collections.abc import Generator, Iterator
from typing import Any

import ctypes
import ismrmrd
import numpy as np
import numpy.fft as fft

from .. import mrdhelper


def process(connection: Any, config: Any, metadata: Any) -> None:
    """Run simple 2D FFT reconstruction.

    Parameters
    ----------
    connection : Connection
        Active MRD connection yielding ``ismrmrd.Acquisition`` items.
    config : Any
        Configuration dict/string from the CONFIG message.
    metadata : Any
        Parsed ISMRMRD XML header (``ismrmrd.xsd`` object or raw text).
    """
    logging.info("simplefft handler — config: %s", config)

    for group in _conditional_groups(
        connection,
        accept=lambda acq: not acq.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA),
        finish=lambda acq: acq.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE),
    ):
        image = _reconstruct(group, metadata)
        if image is not None:
            connection.send(image)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _conditional_groups(
    iterable: Iterator[Any],
    accept: Any,
    finish: Any,
) -> Generator[list[ismrmrd.Acquisition], None, None]:
    """Yield groups of acquisitions accepted by *accept*, split on *finish*."""
    group: list[ismrmrd.Acquisition] = []
    try:
        for item in iterable:
            if item is None:
                break
            if accept(item):
                group.append(item)
            if finish(item):
                yield group
                group = []
    finally:
        from .. import constants

        end = constants.GadgetMessageIdentifier.pack(constants.GADGET_MESSAGE_CLOSE)
        iterable.socket.write(end)


def _reconstruct(
    group: list[ismrmrd.Acquisition], metadata: Any
) -> ismrmrd.Image | None:
    """Reconstruct a single-slice image from a group of readouts.

    Parameters
    ----------
    group : list[ismrmrd.Acquisition]
        Readout lines for one slice.
    metadata : Any
        Parsed ISMRMRD XML header.

    Returns
    -------
    ismrmrd.Image or None
        Reconstructed image, or ``None`` if *group* is empty.
    """
    if not group:
        return None

    logging.info("Reconstructing group of %d readouts", len(group))

    # Stack: [cha, RO, PE]
    data = np.stack([acq.data for acq in group], axis=-1)

    # 2D IFFT
    data = fft.fftshift(data, axes=(1, 2))
    data = fft.ifft2(data, axes=(1, 2))
    data = fft.ifftshift(data, axes=(1, 2))

    # Root-sum-of-squares coil combination
    data = np.sqrt(np.sum(np.abs(data) ** 2, axis=0))

    # Bit depth
    bits = mrdhelper.get_userParameterLong_value(metadata, "BitsStored") or 12
    max_val = 2**bits - 1
    data *= max_val / data.max()
    data = np.around(data).astype(np.int16)

    # Crop readout oversampling
    enc = metadata.encoding[0]
    if enc.reconSpace.matrixSize.x:
        off = (data.shape[0] - enc.reconSpace.matrixSize.x) // 2
        data = data[off : off + enc.reconSpace.matrixSize.x, :]
    if enc.reconSpace.matrixSize.y:
        off = (data.shape[1] - enc.reconSpace.matrixSize.y) // 2
        data = data[:, off : off + enc.reconSpace.matrixSize.y]

    # Build ISMRMRD Image
    image = ismrmrd.Image.from_array(
        data.transpose(), acquisition=group[0], transpose=False
    )
    image.image_index = 1
    image.field_of_view = (
        ctypes.c_float(enc.reconSpace.fieldOfView_mm.x),
        ctypes.c_float(enc.reconSpace.fieldOfView_mm.y),
        ctypes.c_float(enc.reconSpace.fieldOfView_mm.z),
    )

    meta = ismrmrd.Meta(
        {
            "DataRole": "Image",
            "ImageProcessingHistory": ["FIRE", "PYTHON"],
            "WindowCenter": str((max_val + 1) // 2),
            "WindowWidth": str(max_val + 1),
        }
    )
    image.attribute_string = meta.serialize()
    return image
