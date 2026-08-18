"""Any 2D Cartesian scan, one image per slice and echo.

What a reconstruction needs to know is how k-space was sampled, not which
contrast the sequence was after -- a spin echo, a balanced SSFP and a gradient
echo all leave one Cartesian grid per slice, and all of them come back through
this. So there is one plugin here rather than one per sequence.

A streamed reconstruction in one hook. The header describes the encoding space
and each acquisition carries its own counters, so ``pulserver`` places every
line as it arrives; this implements ``recon`` alone, running at each boundary
the sequence flags -- estimating coil sensitivities when a calibration segment
closes, reconstructing when the slice does. The calibration lines are imaging
data too, so there is one buffer and no second grid.

Which of three reconstructions runs is read off the sampling mask rather than
declared: the coil-wise adjoint when everything is there, POCS when the readout
is truncated, CG-SENSE against NLINV maps when phase encodes are missing.
Parallel imaging comes first, because the phase constraint POCS relies on only
means something once the image is unaliased.

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

from pulserver import AcquisitionBucket, ReconContext, ReconPlugin, ReconResult
from pulserver.recon import (
    AcquisitionFlag,
    center_crop,
    coil_combine,
    coil_images,
    echo_count,
    fill_partial_echo,
    recon_shape,
    sense,
    sensitivities,
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

    def recon(
        self, bucket: AcquisitionBucket, context: ReconContext
    ) -> list[ReconResult] | None:
        """Calibrate or reconstruct, according to which boundary ended this bucket."""
        del context
        index = int(bucket.labels("slice")[-1])
        buffer = self.buffers[0]

        # A segment that ended without ending anything larger is the
        # calibration block: estimate the sensitivities now, from the first
        # echo, and let the rest of the slice keep arriving.
        if bucket.trigger is AcquisitionFlag.LAST_IN_SEGMENT:
            self.coil_maps[index] = sensitivities(
                *buffer.select(slice=index, contrast=0), device=self.device
            )
            return None

        results = []
        for echo in range(self.n_echoes):
            echo_kspace, echo_mask = buffer.select(slice=index, contrast=echo)
            lines = echo_mask.any(axis=-1)
            readout = echo_mask.any(axis=0)

            if lines.all():
                coils = (
                    coil_images(echo_kspace, echo_mask, device=self.device)
                    if readout.all()
                    else fill_partial_echo(
                        echo_kspace,
                        readout,
                        self.pocs_iterations,
                        dimension=2,
                        device=self.device,
                    )
                )
                image = coil_combine(coils, coil_axis=0)
            else:
                maps = self.coil_maps.get(index)
                if maps is None:
                    maps = sensitivities(echo_kspace, echo_mask, device=self.device)
                    self.coil_maps[index] = maps
                image = sense(
                    echo_kspace,
                    echo_mask,
                    maps,
                    readout,
                    regularization=self.regularization,
                    iterations=self.iterations,
                    pocs_iterations=self.pocs_iterations,
                    device=self.device,
                )

            image = center_crop(np.abs(image), self.image_shape)
            results.append(
                ReconResult(
                    image.transpose(),
                    reference=-1,
                    series_index=echo,
                    image_type="magnitude",
                    dicom=True,
                )
            )
        return results


PLUGIN = Cartesian2DRecon()
