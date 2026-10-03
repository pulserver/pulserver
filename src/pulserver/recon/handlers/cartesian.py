"""Cartesian FFT reconstruction of readouts placed by their encoding counters."""

from __future__ import annotations

__all__ = ["PLUGIN", "CartesianRecon"]

import numpy as np

from ...mrd._acquisitions import AcquisitionFlag
from ...mrd._images import center_crop, coil_combine
from .._buffers import ReconBuffer, ReconData
from ..plugin import ReconContext, ReconPlugin, ReconResult


class CartesianRecon(ReconPlugin):
    """Root-sum-of-squares FFT image of each slice, contrast, cardiac phase, set and repetition, made as each closes.

    Readouts are placed by their encoding counters, so phase encodes,
    partitions and averages may arrive in any order. An image is made once
    ``LAST_IN_SLICE`` has arrived for each average of a slice, contrast,
    cardiac phase, set and repetition, its averages summed, and at the end of a
    measurement for any that never closed. Its partition axis, when its
    encoding space has one, its phase encodes and its readout are Fourier
    transformed, its coils combined as a root sum of squares, and it is cropped
    to the reconstruction matrix. The values are those of the transform,
    unscaled. Noise and phase-correction readouts are rejected, and a unit with
    no imaging readout makes no image.
    """

    def __init__(self) -> None:
        super().__init__(
            triggers={"imaging": AcquisitionFlag.LAST_IN_SLICE},
            axes=("average",),
            reject_flags=AcquisitionFlag.IS_NOISE_MEASUREMENT
            | AcquisitionFlag.IS_PHASECORR_DATA,
        )

    def recon(
        self, context: ReconContext, branch: str, data: ReconData
    ) -> ReconResult | None:
        del branch
        buffer = data.data
        if buffer is None:
            return None
        image = self.image(averaged(buffer), buffer.image_shape, context, data)
        return ReconResult(
            image, attributes={"ImageProcessingHistory": ["PULSERVER", "PYTHON", "FFT"]}
        )

    def image(
        self,
        kspace: np.ndarray,
        shape: tuple[int, ...],
        context: ReconContext,
        data: ReconData,
    ) -> np.ndarray:
        """Return the magnitude image of ``(coils, [partitions,] phase encodes, readout)`` k-space, cropped to ``shape``.

        ``context`` and ``data`` are those :meth:`recon` was given, for
        subclasses that need the device, the unit's counters or its
        calibration k-space.
        """
        del context, data
        return transformed(kspace, shape)


PLUGIN = CartesianRecon()


def averaged(buffer: ReconBuffer) -> np.ndarray:
    """Return the k-space of ``buffer``, ``(coils, *encoded, readout)``, its averages summed."""
    kspace = buffer.kspace
    if "average" in buffer.axes:
        kspace = kspace.sum(axis=buffer.axes.index("average"))
    return kspace


def transformed(kspace: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Return the root-sum-of-squares image of ``(coils, *encoded, readout)`` k-space, cropped to ``shape``."""
    axes = tuple(range(1, kspace.ndim))
    image = np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(kspace, axes=axes), axes=axes), axes=axes
    )
    combined = coil_combine(image, coil_axis=0)
    return np.array(center_crop(combined, shape[-combined.ndim :]))
