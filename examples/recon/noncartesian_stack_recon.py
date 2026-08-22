"""Any stack of 2D non-Cartesian planes, one volume per measurement.

Calling this module reconstructs an MRD file; ``PLUGIN`` is the same
reconstruction behind the stream contract, driven live by the scanner.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "NonCartesianStackRecon"]

from typing import Any

import numpy as np

from pulserver import ReconContext, ReconPlugin, ReconResult
from pulserver.recon import (
    AcquisitionFlag,
    NoiseAdjust,
    coil_compress,
    ifftc,
    image_result,
    noncartesian_recon,
    pipe_menon_dcf,
)


class NonCartesianStackRecon(ReconPlugin):
    """Reconstruct a non-Cartesian stack, one volume per measurement.

    Stack of stars, stack of spirals: the in-plane shape is the trajectory's
    business, and the reconstruction is the same either way.

    A stack factorises: the partition axis is Cartesian, so an inverse FFT along z
    turns the volume into independent planes, and each plane then goes through the
    in-plane recipe -- Pipe--Menon density compensation, NLINV sensitivities from
    the samples inside the calibration radius, CG-SENSE -- against the trajectory
    its acquisitions carry, which ``pulserver`` buffered beside them. One
    magnitude volume per measurement.

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
    virtual_coils
        Channels to compress the array onto before the solve. A scan with fewer
        physical channels keeps them all.
    calibration_width
        Width of the centred square NLINV solves the sensitivities over.
    device
        Torch device the reconstruction runs on. ``"auto"`` is the host's
        GPU when it has one, and the CPU when it does not.

    Examples
    --------
    Calling the module reconstructs an MRD file, through the same hooks a
    scanner's live stream is driven through. Its settings are the arguments
    of that call:

    >>> import inspect
    >>> from pulserver.app import noncartesian_stack_recon
    >>> "virtual_coils" in inspect.signature(noncartesian_stack_recon).parameters
    True

    Reconstruct a stack whose partition axis is Cartesian::

        images = noncartesian_stack_recon("scan.h5")
        images = noncartesian_stack_recon("scan.h5", virtual_coils=4, mode="pics")

    Every partition repeats one in-plane trajectory, so the picture of where
    a stack sampled is one plane's, and the volume comes back a plane at a
    time:

    .. plot::

       from pulserver.app import noncartesian_stack_recon
       from _figures import images, sampling, stack_example

       measurement = stack_example(
           noncartesian_stack_recon.PLUGIN, size=48, coils=8, partitions=6
       )
       sampling(
           [("one plane's spokes", measurement.measured)],
           title="where a stack of stars sampled",
       )
       images(
           [
               ("object, central partition", measurement.truth),
               ("reconstructed", measurement.image),
           ],
           title="and what came back",
       )
    """

    def __init__(
        self,
        *,
        mode: str = "auto",
        regularization: float = 1e-3,
        iterations: int = 30,
        virtual_coils: int = 8,
        calibration_width: int = 24,
        device: Any = "auto",
    ) -> None:
        super().__init__(
            chain=[NoiseAdjust()],
            branches={AcquisitionFlag.LAST_IN_MEASUREMENT: "imaging"},
            reject_flags=AcquisitionFlag.IS_PHASECORR_DATA,
        )
        if mode not in ("auto", "direct", "pics"):
            raise ValueError(f"mode must be auto, direct or pics, got {mode!r}")
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
        """Reconstruct the stack, once the measurement is complete."""
        del branch, context
        buffer = self.buffers[0]
        n_z = buffer.extents.get("partition", 1)
        n_views = buffer.extents["phase_encode"]

        # A stack of 2D non-Cartesian planes decouples along the fully sampled
        # partition axis: a centered inverse FFT there turns the volume into
        # independent planes the in-plane recon handles one at a time.
        kspace, _ = coil_compress(
            buffer.kspace.reshape(buffer.kspace.shape[0], -1), self.virtual_coils
        )
        planes = ifftc(kspace.reshape(-1, n_z, n_views, buffer.readout), axes=1)
        n_coils = int(planes.shape[0])

        # One trajectory per view, the same for every partition of the stack,
        # and in-plane: the two transverse components of the first partition's.
        trajectory = buffer.points(partition=0)[:2].transpose(1, 2, 0).reshape(-1, 2)
        plane_shape = buffer.image_shape[1:]
        # The planes share a trajectory, so they share its density weighting.
        density = pipe_menon_dcf(trajectory, plane_shape)

        volume = [
            noncartesian_recon(
                planes[:, plane].reshape(n_coils, -1),
                trajectory,
                plane_shape,
                mode=self.mode,
                n_views=n_views,
                density=density,
                regularization=self.regularization,
                iterations=self.iterations,
                calibration_width=self.calibration_width,
                device=self.device,
            )
            for plane in range(n_z)
        ]
        return image_result(np.stack(volume), buffer)


PLUGIN = NonCartesianStackRecon()
