"""Any 2D Cartesian scan, one image per slice and echo.

What a reconstruction needs to know is how k-space was sampled, not which
contrast the sequence was after -- a spin echo, a balanced SSFP and a gradient
echo all leave one Cartesian grid per slice, and all of them come back through
this. So there is one plugin here rather than one per sequence.

A streamed reconstruction. The header describes the encoding space and each
acquisition carries its own counters, so every line is placed as it arrives and
``receive`` routes the two boundaries the sequence flags: a segment that closes
without closing the slice is the calibration block, and the slice itself is an
image. The calibration lines are imaging data too, so there is one buffer and
no second grid.

Which reconstruction runs is read off the sampling mask by
:func:`pulserver.recon.cartesian_recon`, not declared here.

Echoes are an axis, not a variant: a scan whose ``ECO`` label fills the
``contrast`` counter reconstructs each echo against the same sensitivities --
estimated from the first, whose calibration block has the most signal -- and a
single-echo scan is that loop run once.

One magnitude image per ``(slice, echo)``, the echo as the image series,
cropped to the prescribed matrix.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "Cartesian2DRecon"]

from typing import Any

import numpy as np

from pulserver import ReconContext, ReconPlugin, ReconResult
from pulserver.recon import (
    NLINV,
    AcquisitionFlag,
    cartesian_recon,
    center_crop,
    echo_count,
    has_acquisition_flag,
    recon_shape,
)


class Cartesian2DRecon(ReconPlugin):
    """Reconstruct a 2D Cartesian scan, one image per slice and echo.

    Parameters
    ----------
    regularization
        Tikhonov weight of the CG-SENSE solve.
    iterations
        Maximum CG iterations.
    pocs_iterations
        Partial-echo POCS iterations.
    device
        Torch device the reconstruction runs on. ``None`` is the CPU.
    """

    def __init__(
        self,
        *,
        regularization: float = 1e-3,
        iterations: int = 40,
        pocs_iterations: int = 12,
        device: Any = None,
    ) -> None:
        super().__init__(
            split_on=AcquisitionFlag.LAST_IN_SEGMENT | AcquisitionFlag.LAST_IN_SLICE,
            reject_flags=AcquisitionFlag.IS_NOISE_MEASUREMENT
            | AcquisitionFlag.IS_PHASECORR_DATA,
        )
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.pocs_iterations = int(pocs_iterations)
        self.device = device

    def startup(self, context: ReconContext) -> None:
        """Lay the scan's buffers out and note the matrix to crop to."""
        super().startup(context)
        self.n_echoes = echo_count(context.header)
        self.image_shape = recon_shape(context.header)
        self.coil_maps: dict[int, Any] = {}
        self.slice = 0

    def receive(self, acquisition: Any, context: ReconContext) -> Any:
        """Place the line, then route the boundary it closed.

        The slice is tested first: the last line of a slice closes its trailing
        segment as well, and only a segment that closed nothing larger is the
        calibration block.
        """
        self.buffers.add(acquisition)
        self.slice = int(acquisition.idx.slice)
        if has_acquisition_flag(acquisition, AcquisitionFlag.LAST_IN_SLICE):
            return self.recon("imaging", context)
        if has_acquisition_flag(acquisition, AcquisitionFlag.LAST_IN_SEGMENT):
            return self.recon("calibration", context)
        return None

    def recon(self, branch: str, context: ReconContext) -> list[ReconResult] | None:
        """Calibrate the slice, or reconstruct every echo of it."""
        del context
        buffer = self.buffers[0]
        index = self.slice

        if branch == "calibration":
            # The autocalibration block, from the first echo: it has the most
            # signal, and every echo is unaliased against the same maps.
            kspace, mask = buffer.select(slice=index, contrast=0)
            self.coil_maps[index] = NLINV(spatial_ndim=2)(
                kspace[None], mask=mask, device=self.device
            )
            return None

        results = []
        for echo in range(self.n_echoes):
            kspace, mask = buffer.select(slice=index, contrast=echo)
            image = cartesian_recon(
                kspace,
                mask,
                self.coil_maps.get(index),
                regularization=self.regularization,
                iterations=self.iterations,
                pocs_iterations=self.pocs_iterations,
                device=self.device,
            )
            results.append(
                ReconResult(
                    center_crop(np.abs(image), self.image_shape).transpose(),
                    reference=-1,
                    series_index=echo,
                    image_type="magnitude",
                    dicom=True,
                )
            )
        return results


PLUGIN = Cartesian2DRecon()
