"""Multi-slice 2D FFT reconstruction with DICOM export.

Accumulates all readouts until ``ACQ_LAST_IN_MEASUREMENT``, reshapes into
``[cha, RO, PE, SLC]``, applies 2D IFFT + coil combine, and sends back
DICOM images via the ``mrd2dicom`` converter.
"""

import ctypes
import logging
from collections.abc import Generator, Iterator
from typing import Any

import ismrmrd
import numpy as np
import numpy.fft as fft

from .. import mrdhelper
from ..mrd2dicom import MrdDicomBuilder


def process(connection: Any, config: Any, metadata: Any) -> None:
    """Run multi-slice 2D FFT reconstruction with DICOM export.

    Parameters
    ----------
    connection : Connection
        Active MRD connection yielding ``ismrmrd.Acquisition`` items.
    config : Any
        Configuration dict/string from the CONFIG message.
    metadata : Any
        Parsed ISMRMRD XML header.
    """
    logging.info("fftrecon handler — config: %s", config)

    dicom_gen = MrdDicomBuilder(metadata)

    for group in _conditional_groups(
        connection,
        accept=lambda item: isinstance(item, ismrmrd.Acquisition),
        finish=lambda item: isinstance(item, ismrmrd.Acquisition)
        and item.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT),
    ):
        images = _reconstruct(group, metadata)
        for img_array in images:
            mrd_image = _array2image(img_array, group, metadata)
            named_dset = dicom_gen(mrd_image)
            connection.send(named_dset)


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


def _reconstruct(group: list[ismrmrd.Acquisition], metadata: Any) -> np.ndarray:
    """Reconstruct a stack of per-slice images.

    Parameters
    ----------
    group : list[ismrmrd.Acquisition]
        All readout lines in the measurement.
    metadata : Any
        Parsed ISMRMRD XML header.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_slices, RO, PE)``, dtype ``int16``.
    """
    if not group:
        return []

    logging.info("Reconstructing group of %d readouts", len(group))

    # Stack: [cha, RO, PE*SLC]
    data = np.stack([acq.data for acq in group], axis=-1)

    # Reshape to [cha, RO, PE, SLC]
    slices = [acq.idx.slice for acq in group]
    n_slices = max(slices) + 1
    data = data.reshape(data.shape[0], data.shape[1], -1, n_slices)
    slice_order = slices[:n_slices]
    data = data[..., np.argsort(slice_order)]

    # 2D IFFT
    data = fft.fftshift(data, axes=(1, 2))
    data = fft.ifft2(data, axes=(1, 2))
    data = fft.ifftshift(data, axes=(1, 2))

    # Root-sum-of-squares coil combine
    data = np.sqrt(np.sum(np.abs(data) ** 2, axis=0))

    # Normalize
    bits = mrdhelper.get_userParameterLong_value(metadata, "BitsStored") or 12
    max_val = 2**bits - 1
    data *= max_val / data.max()
    data = np.around(data).astype(np.int16)

    # Return per-slice images: list of [RO, PE]
    return data.transpose(2, 0, 1)  # [SLC, RO, PE]


def _array2image(
    data: np.ndarray,
    group: list[ismrmrd.Acquisition],
    metadata: Any,
) -> ismrmrd.Image:
    """Convert a 2-D pixel array to an ``ismrmrd.Image``."""
    enc = metadata.encoding[0]
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
        {"DataRole": "Image", "ImageProcessingHistory": ["FIRE", "PYTHON"]}
    )
    head = image.getHead()
    meta["ImageRowDir"] = [f"{head.read_dir[i]:.18f}" for i in range(3)]
    meta["ImageColumnDir"] = [f"{head.phase_dir[i]:.18f}" for i in range(3)]
    image.attribute_string = meta.serialize()
    return image
