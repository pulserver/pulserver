"""Cartesian FFT reconstruction of readouts placed by their encoding counters."""

from __future__ import annotations

__all__ = ["PLUGIN", "CartesianRecon"]

from typing import Any

import numpy as np

from ...mrd._acquisitions import AcquisitionFlag
from ...mrd._images import center_crop, coil_combine
from ...mrd._metadata import acquisition_label, max_stored_value
from ..plugin import ReconContext, ReconPlugin, ReconResult

#: Placement axes that are Fourier transformed rather than looped over.
_ENCODED = ("partition", "phase_encode")


class CartesianRecon(ReconPlugin):
    """Root-sum-of-squares FFT image of each slice, contrast, cardiac phase, set and repetition, made as each closes.

    Readouts are placed by their encoding counters, so phase encodes,
    partitions and loops may arrive in any order. An image is made at each
    ``LAST_IN_SLICE`` readout of the last average, and at the end of a
    measurement that closes no slice, at the loop position of the readout that
    closed it, its averages summed. Its partition axis, when its encoding space has one, its phase
    encodes and its readout are Fourier transformed, its coils combined as a
    root sum of squares, and it is cropped to the reconstruction matrix and
    scaled to ``int16``, its maximum the header's largest stored value. Noise
    and phase-correction readouts are rejected.
    """

    def __init__(self) -> None:
        super().__init__(
            branches={AcquisitionFlag.LAST_IN_SLICE: "imaging"},
            reject_flags=AcquisitionFlag.IS_NOISE_MEASUREMENT
            | AcquisitionFlag.IS_PHASECORR_DATA,
        )

    def startup(self, context: ReconContext) -> None:
        super().startup(context)
        self.closing: Any = None

    def receive(self, acquisition: Any, context: ReconContext) -> Any:
        self.closing = acquisition
        return super().receive(acquisition, context)

    def recon(self, branch: str, context: ReconContext) -> ReconResult | None:
        del branch
        if self.closing is None:
            return None
        buffer = self.buffers[
            int(acquisition_label(self.closing, "encoding_space_ref", 0) or 0)
        ]
        if not last_average(buffer, self.closing):
            return None
        kspace = loop_position(buffer, self.closing)
        image = self.image(kspace, buffer.image_shape, context.device)
        peak = float(image.max(initial=0.0))
        if peak > 0.0:
            image *= max_stored_value(context.header) / peak
        return ReconResult(
            np.around(image).astype(np.int16),
            attributes={
                "ImageProcessingHistory": ["PULSERVER", "PYTHON", "FFT"],
                "WindowCenter": str((max_stored_value(context.header) + 1) // 2),
                "WindowWidth": str(max_stored_value(context.header) + 1),
            },
        )

    def image(
        self, kspace: np.ndarray, shape: tuple[int, ...], device: str | None
    ) -> np.ndarray:
        """Return the magnitude image of ``(coils, [partitions,] phase encodes, readout)`` k-space, cropped to ``shape``."""
        del device
        return transformed(kspace, shape)


PLUGIN = CartesianRecon()


def last_average(buffer: Any, acquisition: Any) -> bool:
    """Whether ``acquisition`` is of the last average ``buffer`` has room for."""
    if "average" not in buffer.axes:
        return True
    count = buffer.kspace.shape[buffer.axes.index("average")]
    return int(acquisition_label(acquisition, "average", 0) or 0) == count - 1


def loop_position(buffer: Any, acquisition: Any) -> np.ndarray:
    """Return the k-space of ``buffer`` at the loop position of ``acquisition``, averages summed.

    ``(coils, *encoded, readout)``, the encoded axes those of
    :data:`_ENCODED` the buffer has.
    """
    loops = [name for name in buffer.axes[1:-1] if name not in _ENCODED]
    where = {
        name: int(acquisition_label(acquisition, name, 0) or 0)
        for name in loops
        if name != "average"
    }
    kspace, _ = buffer.select(**where)
    kept = [name for name in buffer.axes if name not in where]
    if "average" in kept:
        kspace = kspace.sum(axis=kept.index("average"))
    return kspace


def transformed(kspace: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Return the root-sum-of-squares image of ``(coils, *encoded, readout)`` k-space, cropped to ``shape``."""
    axes = tuple(range(1, kspace.ndim))
    image = np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(kspace, axes=axes), axes=axes), axes=axes
    )
    combined = coil_combine(image, coil_axis=0)
    return np.array(center_crop(combined, shape[-combined.ndim :]))
