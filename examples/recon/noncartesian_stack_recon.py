"""Any stack of 2D non-Cartesian planes, one volume per measurement.

Stack of stars, stack of spirals: the in-plane shape is the trajectory's
business, and the reconstruction is the same either way.

A stack factorises: the partition axis is Cartesian, so an inverse FFT along z
turns the volume into independent planes, and each plane then goes through the
in-plane recipe -- Pipe--Menon density compensation, NLINV sensitivities from
the samples inside the calibration radius, CG-SENSE -- against the trajectory
its acquisitions carry, which ``pulserver`` buffered beside them. One
magnitude volume per measurement.


Examples
--------
Calling the module reconstructs an MRD file: the same three hooks an
inline reconstruction is driven through, fed from the file in this
process rather than over a socket.

>>> from pulserver import ReconPlugin
>>> from pulserver.app import noncartesian_stack_recon
>>> isinstance(noncartesian_stack_recon.PLUGIN, ReconPlugin)
True

The three hooks are the whole plugin, and nothing else is overridden:

>>> sorted(
...     hook for hook in ("startup", "receive", "recon")
...     if hook in vars(noncartesian_stack_recon.NonCartesianStackRecon)
... )
['receive', 'recon', 'startup']

Reconstruct a stack whose partition axis is Cartesian::

    images = noncartesian_stack_recon("scan.h5")

Or re-instantiate the plugin with different settings, and drive it the
same way::

    plugin = noncartesian_stack_recon.NonCartesianStackRecon(coil_compression=8)
    images = plugin.run("scan.h5")
"""

from __future__ import annotations

__all__ = ["PLUGIN", "NonCartesianStackRecon"]

from typing import Any

import numpy as np

from pulserver import ReconContext, ReconPlugin, ReconResult
from pulserver.recon import (
    NLINV,
    AcquisitionFlag,
    NonCartesian2D,
    as_numpy,
    coil_compress,
    has_acquisition_flag,
    noise_prewhiten,
    ifftc,
    pics,
    pipe_menon_dcf,
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
    virtual_coils
        Channels to compress the array onto before the solve. A scan with fewer
        physical channels keeps them all.
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
        virtual_coils: int = 8,
        calibration_width: int = 24,
        device: Any = None,
    ) -> None:
        super().__init__(
            split_on=AcquisitionFlag.LAST_IN_MEASUREMENT,
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
        """Reconstruct the stack, once the measurement is complete."""
        del branch, context
        buffer = self.buffers[0]
        n_z = buffer.extents.get("partition", 1)
        n_views = buffer.extents["phase_encode"]

        # A stack of 2D non-Cartesian planes decouples along the fully sampled
        # partition axis: a centered inverse FFT there turns the volume into
        # independent planes the in-plane recon handles one at a time.
        kspace = buffer.kspace.reshape(buffer.kspace.shape[0], -1)
        kspace, _ = coil_compress(kspace, self.virtual_coils)
        planes = ifftc(kspace.reshape(-1, n_z, n_views, buffer.readout), axes=1)

        # One trajectory per view, the same for every partition of the stack,
        # and in-plane: the two transverse components of the first partition's.
        trajectory = (
            buffer.points(partition=0)[:2]
            .transpose(1, 2, 0)
            .reshape(-1, 2)
            .astype(np.float32)
        )
        plane_shape = buffer.image_shape[1:]
        density = pipe_menon_dcf(trajectory, plane_shape)
        n_coils = int(planes.shape[0])
        nyquist = int(np.ceil(np.pi / 2 * max(plane_shape)))
        direct = self.mode == "direct" or (self.mode == "auto" and n_views >= nyquist)

        volume = []
        for plane in range(n_z):
            data = planes[:, plane].reshape(n_coils, -1)
            if direct:
                coil_wise = NonCartesian2D(
                    trajectory, plane_shape, density=density, n_coils=n_coils
                )
                coils = coil_wise.A_adjoint(data[None])[0]
                image = np.sqrt(np.sum(np.abs(coils) ** 2, axis=0))
            else:
                maps = NLINV(spatial_ndim=2, calibration_width=self.calibration_width)(
                    data,
                    trajectory=trajectory,
                    image_shape=plane_shape,
                    density=density,
                    device=self.device,
                )
                unaliasing = NonCartesian2D(
                    trajectory,
                    plane_shape,
                    coil_maps=maps,
                    density=density,
                    n_coils=n_coils,
                )
                image = pics(
                    data[None],
                    unaliasing,
                    regularization=self.regularization,
                    iterations=self.iterations,
                )[0, 0]
            volume.append(np.abs(as_numpy(image)))

        return ReconResult(
            np.stack(volume).transpose(0, 2, 1),
            reference=-1,
            image_type="magnitude",
            dicom=True,
        )


PLUGIN = NonCartesianStackRecon()
