"""Any stack of 2D non-Cartesian planes, one volume per measurement.

Stack of stars, stack of spirals: the in-plane shape is the trajectory's
business, and the reconstruction is the same either way.

A stack factorises: the partition axis is Cartesian, so an inverse FFT along z
turns the volume into independent planes, and each plane then goes through
:func:`pulserver.recon.reconstruct_plane` -- density compensation,
adjoint-derived sensitivities, CG-SENSE -- against the in-plane trajectory its
acquisitions carry, which ``pulserver`` buffered beside them. One magnitude
volume per measurement.

"""

from __future__ import annotations

__all__ = ["PLUGIN", "NonCartesianStackRecon"]

from typing import Any

import numpy as np

from pulserver import AcquisitionBucket, ReconContext, ReconPlugin, ReconResult
from pulserver.recon import (
    AcquisitionFlag,
    as_numpy,
    ifftc,
    recon_volume,
    reconstruct_plane,
)


class NonCartesianStackRecon(ReconPlugin):
    """Reconstruct a non-Cartesian stack, one volume per measurement.

    Parameters
    ----------
    mode
        ``"auto"`` finishes a fully sampled stack with the
        density-compensated adjoint and sends everything undersampled
        through the per-plane CG-SENSE solve; ``"direct"`` and ``"pics"``
        force one branch. Fully sampled means the in-plane view count
        reaches the radial Nyquist count, ``ceil(pi/2 * matrix)``; the
        partition axis is assumed Cartesian-complete.
    regularization
        Tikhonov weight of the per-plane CG-SENSE solve.
    iterations
        Maximum CG iterations per plane.
    device
        Torch device the reconstruction runs on. ``None`` is the CPU.
    """

    def __init__(
        self,
        *,
        mode: str = "auto",
        regularization: float = 1e-3,
        iterations: int = 30,
        device: Any = None,
    ) -> None:
        super().__init__(
            split_on=AcquisitionFlag.LAST_IN_MEASUREMENT,
            reject_flags=AcquisitionFlag.IS_NOISE_MEASUREMENT
            | AcquisitionFlag.IS_PHASECORR_DATA,
        )
        if mode not in ("auto", "direct", "pics"):
            raise ValueError(f"mode must be auto, direct or pics, got {mode!r}")
        self.mode = mode
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.device = device

    def startup(self, context: ReconContext) -> None:
        """Lay the scan's buffers out and size the volume from the header."""
        super().startup(context)
        self.volume_shape = recon_volume(context.header)

    def recon(
        self, bucket: AcquisitionBucket, context: ReconContext
    ) -> ReconResult | None:
        """Reconstruct once the measurement is complete."""
        del context
        if AcquisitionFlag.LAST_IN_MEASUREMENT not in bucket.trigger:
            return None

        buffer = self.buffers[0]
        extents = dict(zip(buffer.axes, buffer.kspace.shape, strict=True))
        n_z, n_views = extents.get("partition", 1), extents["phase_encode"]

        # A stack of 2D non-Cartesian planes decouples along the fully sampled
        # partition axis: a centered inverse FFT there turns the volume into
        # independent planes the in-plane recon handles one at a time.
        planes = ifftc(buffer.kspace.reshape(-1, n_z, n_views, buffer.readout), axes=1)

        # One trajectory per view, the same for every partition of the stack,
        # and in-plane: the two transverse components of the first partition's.
        trajectory = buffer.points(partition=0)[:2].transpose(1, 2, 0).reshape(-1, 2)
        nyquist = int(np.ceil(np.pi / 2 * max(self.volume_shape[1:])))
        direct = self.mode == "direct" or (self.mode == "auto" and n_views >= nyquist)
        volume = []
        for plane in range(n_z):
            data = planes[:, plane].reshape(planes.shape[0], -1)
            image = reconstruct_plane(
                data,
                trajectory,
                self.volume_shape[1:],
                direct=direct,
                regularization=self.regularization,
                iterations=self.iterations,
                device=self.device,
            )
            volume.append(np.abs(as_numpy(image)))

        return ReconResult(
            np.stack(volume).transpose(0, 2, 1),
            reference=-1,
            image_type="magnitude",
            dicom=True,
        )


PLUGIN = NonCartesianStackRecon()
