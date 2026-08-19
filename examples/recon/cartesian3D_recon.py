"""Any 3D Cartesian scan, one volume per echo.

The same recipes as :mod:`pulserver.app.recon.cartesian2D_recon`, over a slab:
one buffer spanning ``(partition, line, readout)``, filled as the acquisitions
arrive, calibrated when the autocalibration rectangle's segment closes, and
reconstructed once at the end of the measurement -- a 3D scan has one slab, so
there is no per-slice boundary to reconstruct at. An MPRAGE, a fast spin echo
and a balanced SSFP all leave that same grid, so all of them come back through
this.

Which reconstruction runs is read off the sampling mask by
:func:`pulserver.recon.cartesian_recon`, not declared here.

Echoes are an axis, not a variant: each is unaliased and filled against the
same sensitivities, estimated from the first, and a single-echo scan is that
loop run once.

One magnitude volume per echo, the echo as the image series, cropped to the
prescribed matrix.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "Cartesian3DRecon"]

from typing import Any

import numpy as np

from pulserver import ReconContext, ReconPlugin, ReconResult
from pulserver.recon import (
    NLINV,
    AcquisitionFlag,
    cartesian_recon,
    center_crop,
    has_acquisition_flag,
)


class Cartesian3DRecon(ReconPlugin):
    """Reconstruct a 3D Cartesian scan, one volume per echo.

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
            split_on=AcquisitionFlag.LAST_IN_SEGMENT
            | AcquisitionFlag.LAST_IN_MEASUREMENT,
            reject_flags=AcquisitionFlag.IS_NOISE_MEASUREMENT
            | AcquisitionFlag.IS_PHASECORR_DATA,
        )
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.pocs_iterations = int(pocs_iterations)
        self.device = device

    def startup(self, context: ReconContext) -> None:
        """Lay the scan's buffers out, and start with no maps."""
        super().startup(context)
        self.coil_maps: Any = None

    def receive(self, acquisition: Any, context: ReconContext) -> Any:
        """Place the line, then route the boundary it closed.

        The measurement is tested first: its last line closes the trailing
        segment as well, and only a segment that closed nothing larger is the
        autocalibration rectangle.
        """
        self.buffers.add(acquisition)
        if has_acquisition_flag(acquisition, AcquisitionFlag.LAST_IN_MEASUREMENT):
            return self.recon("imaging", context)
        if has_acquisition_flag(acquisition, AcquisitionFlag.LAST_IN_SEGMENT):
            return self.recon("calibration", context)
        return None

    def recon(self, branch: str, context: ReconContext) -> list[ReconResult] | None:
        """Calibrate the slab, or reconstruct every echo of it."""
        del context
        buffer = self.buffers[0]

        if branch == "calibration":
            # The autocalibration rectangle, from the first echo: it has the
            # most signal, and every echo is unaliased against the same maps.
            kspace, mask = buffer.select(contrast=0)
            self.coil_maps = NLINV(spatial_ndim=3)(
                kspace[None], mask=mask, device=self.device
            )
            return None

        results = []
        for echo in range(buffer.extents.get("contrast", 1)):
            kspace, mask = buffer.select(contrast=echo)
            image = cartesian_recon(
                kspace,
                mask,
                self.coil_maps,
                regularization=self.regularization,
                iterations=self.iterations,
                pocs_iterations=self.pocs_iterations,
                device=self.device,
            )
            results.append(
                ReconResult(
                    center_crop(np.abs(image), buffer.image_shape).transpose(0, 2, 1),
                    reference=-1,
                    series_index=echo,
                    image_type="magnitude",
                    dicom=True,
                )
            )
        return results


PLUGIN = Cartesian3DRecon()
