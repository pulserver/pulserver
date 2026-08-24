"""Private multi-slice Cartesian FFT app with DICOM export."""

from __future__ import annotations

__all__ = ["PLUGIN", "FftRecon"]

from typing import Any

import numpy as np

from ...plugin import (
    AcquisitionFlag,
    ReconContext,
    ReconPlugin,
    ReconResult,
)
from . import _fft_combine_scaled


class FftRecon(ReconPlugin):
    """Reconstruct every Cartesian slice at the end of a measurement."""

    def __init__(self) -> None:
        super().__init__(
            reject_flags=AcquisitionFlag.IS_NOISE_MEASUREMENT
            | AcquisitionFlag.IS_PHASECORR_DATA,
            # A generic handler takes streams from anywhere, and a header
            # that does not describe its encoding spaces cannot lay any
            # buffers out. This one collects the lines for itself.
            buffered=False,
        )

    def startup(self, context: ReconContext) -> None:
        """Start the measurement empty; there are no buffers to lay out."""
        del context
        self.lines: list[Any] = []

    def receive(self, acquisition: Any, context: ReconContext) -> Any:
        """Keep the line, and reconstruct when it closes the measurement."""
        self.lines.append(acquisition)
        return super().receive(acquisition, context)

    def recon(self, branch: str, context: ReconContext) -> list[ReconResult]:
        """Return one DICOM-bound result per slice."""
        del branch
        results: list[ReconResult] = []
        slices = np.asarray([int(line.idx.slice) for line in self.lines])
        for slice_index in np.unique(slices):
            indices = np.flatnonzero(slices == slice_index)
            stacked = np.stack(
                [self.lines[int(index)].data for index in indices],
                axis=-1,
            )
            data = _fft_combine_scaled(stacked, context.header)
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
