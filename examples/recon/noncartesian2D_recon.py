"""Any 2D non-Cartesian scan, one image per slice.

Radial, spiral, PROPELLER: what a reconstruction needs is the trajectory each
acquisition carries, not which shape the sequence drew with it, so all of them
come back through this.

Model-based reconstruction, per slice: Pipe--Menon density
compensation, a coil-wise adjoint NUFFT for the sensitivity estimate --
low-pass filtered and normalised to unit root-sum-of-squares, the classic
adaptive-combine approximation -- and a CG-SENSE solve against the
:class:`pulserver.recon.physics.NonCartesian2D` operator built from the
same trajectory.

The trajectory arrives per acquisition, as MRD carries it, scaled to
MRI-NUFFT's ``[-0.5, 0.5)`` units -- what the LiveSDK's enrichment writes.

"""

from __future__ import annotations

__all__ = ["PLUGIN", "NonCartesian2DRecon"]

from typing import Any

import numpy as np

from pulserver import AcquisitionBucket, ReconContext, ReconPlugin, ReconResult
from pulserver.recon import AcquisitionFlag, as_numpy, recon_shape, reconstruct_plane


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
        """Lay the scan's buffers out and size the image from the header."""
        super().startup(context)
        self.image_shape = recon_shape(context.header)

    def recon(
        self, bucket: AcquisitionBucket, context: ReconContext
    ) -> list[ReconResult] | None:
        """Reconstruct every slice once the measurement is complete."""
        del context
        if AcquisitionFlag.LAST_IN_MEASUREMENT not in bucket.trigger:
            return None

        buffer = self.buffers[0]
        extents = dict(zip(buffer.axes, buffer.kspace.shape, strict=True))
        n_slices, n_views = extents.get("slice", 1), extents["phase_encode"]
        nyquist = int(np.ceil(np.pi / 2 * max(self.image_shape)))
        direct = self.mode == "direct" or (self.mode == "auto" and n_views >= nyquist)

        results = []
        for index in range(n_slices):
            # Views laid end to end, and the points they were taken at in the
            # same order: one non-Cartesian measurement of this slice.
            views, _ = buffer.select(slice=index)
            data = views.reshape(views.shape[0], -1)
            trajectory = (
                buffer.points(slice=index)[:2].transpose(1, 2, 0).reshape(-1, 2)
            )
            image = reconstruct_plane(
                data,
                trajectory,
                self.image_shape,
                direct=direct,
                regularization=self.regularization,
                iterations=self.iterations,
                device=self.device,
            )
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
