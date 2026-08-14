"""Private multi-slice Cartesian FFT app with DICOM export."""

from __future__ import annotations

__all__ = ["PLUGIN", "FftRecon"]

import numpy as np

from ...postprocessing import coil_combine
import numpy.fft as fft

from ...plugin import AcquisitionBucket, ReconPlugin, ReconContext, ReconResult
from .. import mrdhelper


class FftRecon(ReconPlugin):
    """Reconstruct every Cartesian slice at the end of a measurement."""

    def __init__(self) -> None:
        super().__init__(
            reject_flags=(
                "ACQ_IS_NOISE_MEASUREMENT",
                "ACQ_IS_PHASECORR_DATA",
            )
        )

    def recon(
        self,
        bucket: AcquisitionBucket,
        context: ReconContext,
    ) -> list[ReconResult]:
        """Return one DICOM-bound result per slice."""
        results: list[ReconResult] = []
        slices = bucket.labels("slice")
        for slice_index in np.unique(slices):
            indices = np.flatnonzero(slices == slice_index)
            data = _reconstruct_slice(bucket, indices, context.header)
            results.append(
                ReconResult(
                    data.transpose(),
                    reference=int(indices[0]),
                    attributes={
                        "ImageProcessingHistory": ["PULSERVER", "PYTHON", "FFT"]
                    },
                    dicom=True,
                )
            )
        return results


PLUGIN = FftRecon()


# %% private module subroutines


def _reconstruct_slice(
    bucket: AcquisitionBucket,
    indices: np.ndarray,
    header,
) -> np.ndarray:
    data = np.stack(
        [bucket.data[int(index)].data for index in indices],
        axis=-1,
    )
    data = fft.fftshift(data, axes=(1, 2))
    data = fft.ifft2(data, axes=(1, 2))
    data = fft.ifftshift(data, axes=(1, 2))
    data = coil_combine(data, coil_axis=0)

    maximum = float(data.max(initial=0.0))
    if maximum > 0.0:
        data *= _max_value(header) / maximum
    return np.around(data).astype(np.int16)


def _max_value(header) -> int:
    bits = mrdhelper.get_userParameterLong_value(header, "BitsStored") or 12
    return 2 ** int(bits) - 1
