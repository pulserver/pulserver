"""Two-dimensional Cartesian FFT reconstruction, one image per slice.

Adapted from the ``simplefft`` example of
python-ismrmrd-server (Copyright (c) 2024 Kelvin Chow; MIT, see ``LICENSES/python-ismrmrd-server-MIT.txt``).
"""

from __future__ import annotations

__all__ = ["PLUGIN", "SimpleFftRecon"]

from typing import Any

import numpy as np
import numpy.fft as fft

from ...mrd._acquisitions import AcquisitionFlag
from ...mrd._images import center_crop, coil_combine
from ...mrd._metadata import max_stored_value
from ..plugin import ReconContext, ReconPlugin, ReconResult


class SimpleFftRecon(ReconPlugin):
    """Root-sum-of-squares FFT image of each slice, made at its ``LAST_IN_SLICE`` line.

    Lines are collected in arrival order rather than placed by their counters,
    so a header that does not describe the encoding is enough. Noise and
    phase-correction lines are rejected. Images are ``int16``, scaled so their
    maximum is the header's largest stored value, and cropped to the first
    encoding space's reconstruction matrix.
    """

    def __init__(self) -> None:
        super().__init__(
            branches={AcquisitionFlag.LAST_IN_SLICE: "imaging"},
            reject_flags=AcquisitionFlag.IS_NOISE_MEASUREMENT
            | AcquisitionFlag.IS_PHASECORR_DATA,
            buffered=False,
        )

    def startup(self, context: ReconContext) -> None:
        del context
        self.lines: list[Any] = []

    def receive(self, acquisition: Any, context: ReconContext) -> Any:
        self.lines.append(acquisition)
        return super().receive(acquisition, context)

    def recon(self, branch: str, context: ReconContext) -> ReconResult | None:
        del branch
        if not self.lines:
            return None
        data = _reconstruct(self.lines, context.header)
        self.lines = []
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


def _reconstruct(lines: list[Any], header: Any) -> np.ndarray:
    """Return the ``(x, y)`` image of lines stacked as ``(coils, x, y)``."""
    stacked = np.stack([acquisition.data for acquisition in lines], axis=-1)
    data = fft.ifftshift(stacked, axes=(1, 2))
    data = fft.ifft2(data, axes=(1, 2))
    data = fft.fftshift(data, axes=(1, 2))
    data = coil_combine(data, coil_axis=0)

    maximum = float(data.max(initial=0.0))
    if maximum > 0.0:
        data *= max_stored_value(header) / maximum
    data = np.around(data).astype(np.int16)

    encoding = header.encoding[0]
    target_x = min(
        int(encoding.reconSpace.matrixSize.x or data.shape[0]), data.shape[0]
    )
    target_y = min(
        int(encoding.reconSpace.matrixSize.y or data.shape[1]), data.shape[1]
    )
    return center_crop(data, (target_x, target_y))
