"""Any 3D non-Cartesian scan, one volume per measurement.

A reconstruction over the whole sampled ball -- ZTE, and any other sequence
whose readouts leave a genuinely three-dimensional trajectory rather than a
stack of planes: Pipe--Menon
density compensation and a coil-wise adjoint NUFFT through
:class:`pulserver.recon.physics.NonCartesian3D` -- the direct finish, valid
because a ZTE shell set is designed at Nyquist -- with a CG solve available
for anything undersampled, exactly as the 2D non-Cartesian plugin branches.
The dead-time gap's missing centre samples are declared by the sequence and
left to the density weighting here; a dedicated gap-filling refinement
starts by overriding this plugin.

"""

from __future__ import annotations

__all__ = ["PLUGIN", "NonCartesian3DRecon"]

from typing import Any

import numpy as np

from pulserver import ReconContext, ReconPlugin, ReconResult
from pulserver.recon import (
    NLINV,
    AcquisitionFlag,
    NonCartesian3D,
    as_numpy,
    coil_compress,
    has_acquisition_flag,
    noise_prewhiten,
    pics,
    pipe_menon_dcf,
)


class NonCartesian3DRecon(ReconPlugin):
    """Reconstruct a ZTE sphere, one volume per measurement.

    Parameters
    ----------
    mode
        ``"direct"`` (the default: the shell set satisfies Nyquist by
        design) or ``"pics"`` for an undersampled prescription.
    regularization, iterations
        The CG solve's Tikhonov weight and iteration ceiling, ``pics`` only.
    virtual_coils
        Channels to compress the array onto before the solve. A scan with fewer
        physical channels keeps them all.
    calibration_width
        Width of the centred cube NLINV solves the sensitivities over,
        ``pics`` only.
    device
        Torch device the reconstruction runs on. ``None`` is the CPU.
    """

    def __init__(
        self,
        *,
        mode: str = "direct",
        regularization: float = 1e-3,
        iterations: int = 20,
        virtual_coils: int = 8,
        calibration_width: int = 16,
        device: Any = None,
    ) -> None:
        super().__init__(
            split_on=AcquisitionFlag.LAST_IN_MEASUREMENT,
            reject_flags=AcquisitionFlag.IS_PHASECORR_DATA,
        )
        if mode not in ("direct", "pics"):
            raise ValueError(f"mode must be direct or pics, got {mode!r}")
        self.mode = mode
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.virtual_coils = int(virtual_coils)
        self.calibration_width = int(calibration_width)
        self.device = device

    def startup(self, context: ReconContext) -> None:
        """Lay the scan's buffers out, with no noise measured yet."""
        super().startup(context)
        self.noise: Any = None

    def receive(self, acquisition: Any, context: ReconContext) -> Any:
        """Whiten the readout and place it, and reconstruct at the end of the scan.

        A noise scan is not imaging data and never reaches a buffer: what it
        leaves behind is the covariance every readout that follows is whitened
        by.
        """
        line = np.asarray(acquisition.data)
        if has_acquisition_flag(acquisition, AcquisitionFlag.IS_NOISE_MEASUREMENT):
            self.noise = (
                line if self.noise is None else np.concatenate([self.noise, line], -1)
            )
            return None
        if self.noise is not None:
            line = noise_prewhiten(line, self.noise, coil_axis=0)

        self.buffers.add(acquisition, line)
        if has_acquisition_flag(acquisition, AcquisitionFlag.LAST_IN_MEASUREMENT):
            return self.recon("imaging", context)
        return None

    def recon(self, branch: str, context: ReconContext) -> ReconResult:
        """Reconstruct the volume, once the measurement is complete."""
        del branch, context
        # Spokes laid end to end, and the points they were taken at in the
        # same order: one non-Cartesian measurement of the volume.
        buffer = self.buffers[0]
        volume_shape = buffer.image_shape
        data = buffer.kspace.reshape(buffer.kspace.shape[0], -1)
        points = buffer.points()[:3].reshape(3, -1).T.astype(np.float32)

        data, _ = coil_compress(data, self.virtual_coils)
        density = pipe_menon_dcf(points, volume_shape)
        n_coils = int(data.shape[0])

        if self.mode == "direct":
            coil_wise = NonCartesian3D(
                points, volume_shape, density=density, n_coils=n_coils
            )
            # Native complex throughout: the measurement crosses the NumPy
            # boundary and the coil-wise adjoint answers with one complex
            # volume per coil, so there is no real/complex view juggling to
            # unpack the channel axis.
            coil_images = coil_wise.A_adjoint(data[None])[0]
            image = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0))
        else:
            maps = NLINV(spatial_ndim=3, calibration_width=self.calibration_width)(
                data,
                trajectory=points,
                image_shape=volume_shape,
                density=density,
                device=self.device,
            )
            sense = NonCartesian3D(
                points,
                volume_shape,
                coil_maps=maps,
                density=density,
                n_coils=n_coils,
            )
            # NumPy in, NumPy out: the unaliased complex volume comes straight
            # back from the solve, which keeps a singleton coil axis, so index
            # past both batch and channel to the volume.
            image = np.abs(
                pics(
                    data[None],
                    sense,
                    regularization=self.regularization,
                    iterations=self.iterations,
                )[0, 0]
            )

        return ReconResult(
            as_numpy(image).transpose(0, 2, 1),
            reference=-1,
            image_type="magnitude",
            dicom=True,
        )


PLUGIN = NonCartesian3DRecon()
