"""Any 3D Cartesian scan, one volume per echo.

The same recipes as :mod:`pulserver.app.recon.cartesian2D_recon`, over a slab:
one buffer spanning ``(partition, line, readout)``, filled as the acquisitions
arrive, calibrated when the autocalibration rectangle's segment closes, and
reconstructed once at the end of the measurement -- a 3D scan has one slab, so
there is no per-slice boundary to reconstruct at. An MPRAGE, a fast spin echo
and a balanced SSFP all leave that same grid, so all of them come back through
this.

Which of three reconstructions runs is read off the sampling mask rather than
declared: the coil-wise adjoint when everything is there, POCS when the readout
is truncated, CG-SENSE against 3D NLINV maps when phase encodes are missing on
either axis. Parallel imaging comes first, because the phase constraint POCS
relies on only means something once the image is unaliased. Two and three
dimensions are not two pipelines: :func:`pulserver.recon.sense` and the rest
read which one is running off the sampling mask.

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

from pulserver import AcquisitionBucket, ReconContext, ReconPlugin, ReconResult
from pulserver.recon import (
    AcquisitionFlag,
    center_crop,
    coil_combine,
    coil_images,
    echo_count,
    fill_partial_echo,
    recon_volume,
    sense,
    sensitivities,
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
        """Lay the scan's buffers out and note the matrix to crop to."""
        super().startup(context)
        self.n_echoes = echo_count(context.header)
        self.image_shape = recon_volume(context.header)
        self.coil_maps: Any = None

    def recon(
        self, bucket: AcquisitionBucket, context: ReconContext
    ) -> list[ReconResult] | None:
        """Calibrate or reconstruct, according to which boundary ended this bucket."""
        del context
        buffer = self.buffers[0]

        if AcquisitionFlag.LAST_IN_MEASUREMENT not in bucket.trigger:
            self.coil_maps = sensitivities(
                *buffer.select(contrast=0), device=self.device
            )
            return None

        results = []
        for echo in range(self.n_echoes):
            echo_kspace, echo_mask = buffer.select(contrast=echo)
            encodes = echo_mask.any(axis=-1)
            readout = echo_mask.reshape(-1, echo_mask.shape[-1]).any(axis=0)

            if encodes.all():
                coils = (
                    coil_images(echo_kspace, echo_mask, device=self.device)
                    if readout.all()
                    else fill_partial_echo(
                        echo_kspace,
                        readout,
                        self.pocs_iterations,
                        dimension=3,
                        device=self.device,
                    )
                )
                image = coil_combine(coils, coil_axis=0)
            else:
                maps = self.coil_maps
                if maps is None:
                    maps = sensitivities(echo_kspace, echo_mask, device=self.device)
                    self.coil_maps = maps
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
                    image.transpose(0, 2, 1),
                    reference=-1,
                    series_index=echo,
                    image_type="magnitude",
                    dicom=True,
                )
            )
        return results


PLUGIN = Cartesian3DRecon()
