"""Reconstruction for :mod:`pulserver.app.sequence.gre_multiecho3D_sequence`.

The 3D Cartesian pipeline of :mod:`pulserver.app.recon.gre3D_recon`, once per echo:
the volume grid gains an echo axis fed by the ``contrast`` counter, the coil
sensitivities are estimated once from the first echo's calibration
rectangle, and each echo is then unaliased and filled against the same maps.
One magnitude volume per echo, the echo as the image series.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "GreMultiecho3DRecon"]

from typing import Any

import numpy as np

from pulserver import AcquisitionBucket, ReconContext, ReconPlugin, ReconResult
from pulserver.recon import has_acquisition_flag
from pulserver.recon.postprocessing import center_crop, coil_combine
from pulserver.recon.preprocessing import CartesianGridder, receiver_channels
from pulserver.app.recon.gre3D_recon import (
    coil_volumes,
    encoded_volume,
    fill_partial_echo,
    recon_volume,
    sense,
    sensitivities,
)
from pulserver.app.recon.gre_multiecho2D_recon import echo_count


class GreMultiecho3DRecon(ReconPlugin):
    """Reconstruct a multi-echo 3D Cartesian gradient echo, one volume per echo.

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
            split_on=("ACQ_LAST_IN_SEGMENT", "ACQ_LAST_IN_MEASUREMENT"),
            reject_flags=("ACQ_IS_NOISE_MEASUREMENT", "ACQ_IS_PHASECORR_DATA"),
        )
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.pocs_iterations = int(pocs_iterations)
        self.device = device

    def startup(self, context: ReconContext) -> None:
        """Allocate one grid per echo, from the header."""
        n_z, n_y, n_x = encoded_volume(context.header)
        self.n_echoes = echo_count(context.header)
        self.buffer = CartesianGridder(
            (self.n_echoes, n_z, n_y, n_x),
            coils=receiver_channels(context.header),
        )
        self.image_shape = recon_volume(context.header)
        self.coil_maps: Any = None

    def receive(self, acquisition: Any, context: ReconContext) -> None:
        """Place one arriving line at its echo, partition and phase encode."""
        del context
        self.buffer.add(
            acquisition.data,
            acquisition.idx.contrast,
            acquisition.idx.kspace_encode_step_2,
            acquisition.idx.kspace_encode_step_1,
        )

    def recon(
        self, bucket: AcquisitionBucket, context: ReconContext
    ) -> list[ReconResult] | None:
        """Calibrate or reconstruct, according to which boundary ended this bucket."""
        del context
        kspace, mask = self.buffer.kspace, self.buffer.mask

        last = bucket.acquisitions[-1]
        if not has_acquisition_flag(last, "ACQ_LAST_IN_MEASUREMENT"):
            self.coil_maps = sensitivities(kspace[:, 0], mask[0], device=self.device)
            return None

        results = []
        for echo in range(self.n_echoes):
            echo_kspace, echo_mask = kspace[:, echo], mask[echo]
            encodes = echo_mask.any(axis=-1)
            readout = echo_mask.reshape(-1, echo_mask.shape[-1]).any(axis=0)

            if encodes.all():
                coils = (
                    coil_volumes(echo_kspace, echo_mask, device=self.device)
                    if readout.all()
                    else fill_partial_echo(
                        echo_kspace, readout, self.pocs_iterations, device=self.device
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


PLUGIN = GreMultiecho3DRecon()
