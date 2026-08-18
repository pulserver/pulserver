"""Private Cartesian FFT reconstruction app returning MRD images."""

from __future__ import annotations

__all__ = ["PLUGIN", "SimpleFftRecon"]

import numpy as np

from ...postprocessing import coil_combine
import numpy.fft as fft

from ...plugin import (
    AcquisitionBucket,
    ReconContext,
    ReconPlugin,
    ReconResult,
    AcquisitionFlag,
)
from .. import mrdhelper


class SimpleFftRecon(ReconPlugin):
    """Reconstruct one Cartesian slice whenever its final line arrives."""

    def __init__(self) -> None:
        super().__init__(
            split_on=AcquisitionFlag.LAST_IN_SLICE,
            reject_flags=AcquisitionFlag.IS_NOISE_MEASUREMENT
            | AcquisitionFlag.IS_PHASECORR_DATA,
            # A generic handler takes streams from anywhere, and a header
            # that does not describe its encoding spaces cannot lay any
            # buffers out. This one sorts the bucket for itself.
            buffered=False,
        )

    def recon(
        self,
        bucket: AcquisitionBucket,
        context: ReconContext,
    ) -> ReconResult | None:
        """Apply a two-dimensional IFFT and root-sum-of-squares combine."""
        if not bucket.data:
            return None
        data = _reconstruct(bucket, context.header)
        return ReconResult(
            data.transpose(),
            attributes={
                "ImageProcessingHistory": ["PULSERVER", "PYTHON", "FFT"],
                "WindowCenter": str((_max_value(context.header) + 1) // 2),
                "WindowWidth": str(_max_value(context.header) + 1),
            },
        )


PLUGIN = SimpleFftRecon()


# %% private module subroutines


def _reconstruct(bucket: AcquisitionBucket, header) -> np.ndarray:
    data = np.stack([acquisition.data for acquisition in bucket.data], axis=-1)
    data = fft.fftshift(data, axes=(1, 2))
    data = fft.ifft2(data, axes=(1, 2))
    data = fft.ifftshift(data, axes=(1, 2))
    data = coil_combine(data, coil_axis=0)

    maximum = float(data.max(initial=0.0))
    if maximum > 0.0:
        data *= _max_value(header) / maximum
    data = np.around(data).astype(np.int16)

    encoding = header.encoding[0]
    target_x = int(encoding.reconSpace.matrixSize.x or data.shape[0])
    target_y = int(encoding.reconSpace.matrixSize.y or data.shape[1])
    data = _center_crop(data, target_x, axis=0)
    return _center_crop(data, target_y, axis=1)


def _max_value(header) -> int:
    bits = mrdhelper.get_userParameterLong_value(header, "BitsStored") or 12
    return 2 ** int(bits) - 1


def _center_crop(data: np.ndarray, size: int, *, axis: int) -> np.ndarray:
    if size >= data.shape[axis]:
        return data
    start = (data.shape[axis] - size) // 2
    selection = [slice(None)] * data.ndim
    selection[axis] = slice(start, start + size)
    return data[tuple(selection)]
