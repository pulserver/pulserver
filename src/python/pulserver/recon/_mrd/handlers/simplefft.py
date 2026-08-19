"""Private Cartesian FFT reconstruction app returning MRD images."""

from __future__ import annotations

__all__ = ["PLUGIN", "SimpleFftRecon"]

import numpy as np

from ...plugin import (
    AcquisitionBucket,
    AcquisitionFlag,
    ReconContext,
    ReconPlugin,
    ReconResult,
)
from ...postprocessing import center_crop
from ..metadata import max_stored_value
from . import _fft_combine_scaled


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
                "WindowCenter": str((max_stored_value(context.header) + 1) // 2),
                "WindowWidth": str(max_stored_value(context.header) + 1),
            },
        )


PLUGIN = SimpleFftRecon()


# %% private module subroutines


def _reconstruct(bucket: AcquisitionBucket, header) -> np.ndarray:
    stacked = np.stack([acquisition.data for acquisition in bucket.data], axis=-1)
    data = _fft_combine_scaled(stacked, header)

    encoding = header.encoding[0]
    target_x = min(
        int(encoding.reconSpace.matrixSize.x or data.shape[0]), data.shape[0]
    )
    target_y = min(
        int(encoding.reconSpace.matrixSize.y or data.shape[1]), data.shape[1]
    )
    return center_crop(data, (target_x, target_y))
