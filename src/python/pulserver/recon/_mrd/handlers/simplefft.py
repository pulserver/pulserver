"""Private Cartesian FFT reconstruction app returning MRD images."""

from __future__ import annotations

__all__ = ["PLUGIN", "SimpleFftRecon"]

from typing import Any

import numpy as np

from ...plugin import (
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
            # buffers out. This one collects the lines for itself.
            buffered=False,
        )

    def startup(self, context: ReconContext) -> None:
        """Start the slice empty; there are no buffers to lay out."""
        del context
        self.lines: list[Any] = []

    def receive(self, acquisition: Any, context: ReconContext) -> Any:
        """Keep the line, and reconstruct when it closes the slice."""
        self.lines.append(acquisition)
        return super().receive(acquisition, context)

    def recon(self, branch: str, context: ReconContext) -> ReconResult | None:
        """Apply a two-dimensional IFFT and root-sum-of-squares combine."""
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


def _reconstruct(lines: list[Any], header) -> np.ndarray:
    stacked = np.stack([acquisition.data for acquisition in lines], axis=-1)
    data = _fft_combine_scaled(stacked, header)

    encoding = header.encoding[0]
    target_x = min(
        int(encoding.reconSpace.matrixSize.x or data.shape[0]), data.shape[0]
    )
    target_y = min(
        int(encoding.reconSpace.matrixSize.y or data.shape[1]), data.shape[1]
    )
    return center_crop(data, (target_x, target_y))
