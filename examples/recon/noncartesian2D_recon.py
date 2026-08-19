"""Any 2D non-Cartesian scan, one image per slice.

Radial, spiral, PROPELLER: what a reconstruction needs is the trajectory each
acquisition carries, not which shape the sequence drew with it, so all of them
come back through this.

Model-based reconstruction, per slice: Pipe--Menon density compensation, NLINV
sensitivities calibrated from the samples inside the calibration radius --
selected before any gridding -- and a CG-SENSE solve against the
:class:`pulserver.recon.physics.NonCartesian2D` operator built from the same
trajectory. A fully sampled slice finishes with the density-compensated
adjoint instead, coil images root-sum-of-squares combined.

The trajectory arrives per acquisition, as MRD carries it, scaled to
MRI-NUFFT's ``[-0.5, 0.5)`` units -- what the LiveSDK's enrichment writes.

"""

from __future__ import annotations

__all__ = ["PLUGIN", "NonCartesian2DRecon"]

from typing import Any

import numpy as np

from pulserver import ReconContext, ReconPlugin, ReconResult
from pulserver.recon import (
    NLINV,
    AcquisitionFlag,
    NonCartesian2D,
    as_numpy,
    pics,
    pipe_menon_dcf,
)


class NonCartesian2DRecon(ReconPlugin):
    """Reconstruct a 2D non-Cartesian gradient echo, one image per slice.

    Parameters
    ----------
    mode
        ``"auto"`` finishes a fully sampled slice with the
        density-compensated adjoint and sends everything undersampled
        through CG-SENSE; ``"direct"`` and ``"pics"`` force one branch. A
        slice counts as fully sampled when its view count reaches the radial
        Nyquist count, ``ceil(pi/2 * matrix)``.
    regularization
        Tikhonov weight of the CG-SENSE solve.
    iterations
        Maximum CG iterations.
    calibration_width
        Width of the centred square NLINV solves the sensitivities over.
    device
        Torch device the reconstruction runs on. ``None`` is the CPU.
    """

    def __init__(
        self,
        *,
        mode: str = "auto",
        regularization: float = 1e-3,
        iterations: int = 30,
        calibration_width: int = 24,
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
        self.calibration_width = int(calibration_width)
        self.device = device

    def recon(self, branch: str, context: ReconContext) -> list[ReconResult]:
        """Reconstruct every slice, once the measurement is complete."""
        del branch, context
        buffer = self.buffers[0]
        image_shape = buffer.image_shape
        n_slices = buffer.extents.get("slice", 1)
        n_views = buffer.extents["phase_encode"]
        nyquist = int(np.ceil(np.pi / 2 * max(image_shape)))
        direct = self.mode == "direct" or (self.mode == "auto" and n_views >= nyquist)

        results = []
        for index in range(n_slices):
            # Views laid end to end, and the points they were taken at in the
            # same order: one non-Cartesian measurement of this slice.
            views, _ = buffer.select(slice=index)
            data = views.reshape(views.shape[0], -1)
            trajectory = (
                buffer.points(slice=index)[:2]
                .transpose(1, 2, 0)
                .reshape(-1, 2)
                .astype(np.float32)
            )
            density = pipe_menon_dcf(trajectory, image_shape)
            n_coils = int(data.shape[0])

            if direct:
                # The density-compensated adjoint is an inverse only at
                # Nyquist, and a coil-wise one at that: combine by
                # root-sum-of-squares.
                coil_wise = NonCartesian2D(
                    trajectory, image_shape, density=density, n_coils=n_coils
                )
                coils = coil_wise.A_adjoint(data[None])[0]
                image = np.sqrt(np.sum(np.abs(coils) ** 2, axis=0))
            else:
                maps = NLINV(spatial_ndim=2, calibration_width=self.calibration_width)(
                    data,
                    trajectory=trajectory,
                    image_shape=image_shape,
                    density=density,
                    device=self.device,
                )
                unaliasing = NonCartesian2D(
                    trajectory,
                    image_shape,
                    coil_maps=maps,
                    density=density,
                    n_coils=n_coils,
                )
                # The SENSE solve keeps a singleton coil axis, so index past
                # both batch and channel to reach the plane.
                image = pics(
                    data[None],
                    unaliasing,
                    regularization=self.regularization,
                    iterations=self.iterations,
                )[0, 0]
            results.append(
                ReconResult(
                    np.abs(as_numpy(image)).transpose(),
                    reference=-1,
                    image_index=index,
                    image_type="magnitude",
                    dicom=True,
                )
            )
        return results


PLUGIN = NonCartesian2DRecon()
