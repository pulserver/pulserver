"""Private Cartesian FFT reconstruction app returning MRD images."""

from __future__ import annotations

__all__ = ["PLUGIN", "SimpleFftRecon"]

import numpy as np
import numpy.fft as fft

from ...app import AcquisitionBucket, ReconApp, ReconContext, ReconResult
from .. import mrdhelper


class SimpleFftRecon(ReconApp):
    """Reconstruct one Cartesian slice whenever its final line arrives."""

    def __init__(self) -> None:
        super().__init__(
            split_on="ACQ_LAST_IN_SLICE",
            reject_flags=(
                "ACQ_IS_NOISE_MEASUREMENT",
                "ACQ_IS_PHASECORR_DATA",
            ),
        )

    def reconstruct(
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
    data = np.sqrt(np.sum(np.abs(data) ** 2, axis=0))

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
