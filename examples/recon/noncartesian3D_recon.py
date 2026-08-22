"""Any 3D non-Cartesian scan, one volume per measurement.

Calling this module reconstructs an MRD file; ``PLUGIN`` is the same
reconstruction behind the stream contract, driven live by the scanner.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "NonCartesian3DRecon"]

from typing import Any


from pulserver import ReconContext, ReconPlugin, ReconResult
from pulserver.recon import (
    AcquisitionFlag,
    NoiseAdjust,
    coil_compress,
    image_result,
    noncartesian_recon,
)


class NonCartesian3DRecon(ReconPlugin):
    """Reconstruct a ZTE sphere, one volume per measurement.

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
        Torch device the reconstruction runs on. ``"auto"`` is the host's
        GPU when it has one, and the CPU when it does not.

    Examples
    --------
    Calling the module reconstructs an MRD file, through the same hooks a
    scanner's live stream is driven through. Its settings are the arguments
    of that call:

    >>> import inspect
    >>> from pulserver.app import noncartesian3D_recon
    >>> "virtual_coils" in inspect.signature(noncartesian3D_recon).parameters
    True

    Reconstruct a fully 3D non-Cartesian scan::

        images = noncartesian3D_recon("scan.h5")
        images = noncartesian3D_recon("scan.h5", virtual_coils=4, mode="pics")

    A ball of diameters through k-space rather than a stack of planes, and the
    central partition of the volume it returns:

    .. plot::

       from pulserver.app import noncartesian3D_recon
       from _figures import images, koosh_spokes, sampling, volume_example

       size = 32
       measurement = volume_example(
           noncartesian3D_recon.PLUGIN, size=size, coils=4
       )
       sampling(
           [("the sampled ball", measurement.measured)],
           title="where a 3D radial scan sampled",
       )
       images(
           [
               ("object, central partition", measurement.truth),
               ("density-compensated adjoint", measurement.image),
           ],
           title="and what came back",
       )
    """

    def __init__(
        self,
        *,
        mode: str = "direct",
        regularization: float = 1e-3,
        iterations: int = 20,
        virtual_coils: int = 8,
        calibration_width: int = 16,
        device: Any = "auto",
    ) -> None:
        super().__init__(
            chain=[NoiseAdjust()],
            branches={AcquisitionFlag.LAST_IN_MEASUREMENT: "imaging"},
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
        """Lay the scan's buffers out."""
        super().startup(context)

    def recon(self, branch: str, context: ReconContext) -> ReconResult:
        """Reconstruct the volume, once the measurement is complete."""
        del branch, context
        # Spokes laid end to end, and the points they were taken at in the
        # same order: one non-Cartesian measurement of the volume.
        buffer = self.buffers[0]
        data, _ = coil_compress(
            buffer.kspace.reshape(buffer.kspace.shape[0], -1), self.virtual_coils
        )
        image = noncartesian_recon(
            data,
            buffer.points()[:3].reshape(3, -1).T,
            buffer.image_shape,
            mode=self.mode,
            regularization=self.regularization,
            iterations=self.iterations,
            calibration_width=self.calibration_width,
            device=self.device,
        )
        return image_result(image, buffer)


PLUGIN = NonCartesian3DRecon()
