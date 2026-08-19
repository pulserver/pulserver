"""Any 3D Cartesian scan, one volume per echo.

The same recipes as :mod:`pulserver.app.recon.cartesian2D_recon`, over a slab:
one buffer spanning ``(partition, line, readout)``, filled as the acquisitions
arrive, calibrated when the autocalibration rectangle's segment closes, and
reconstructed once at the end of the measurement -- a 3D scan has one slab, so
there is no per-slice boundary to reconstruct at. An MPRAGE, a fast spin echo
and a balanced SSFP all leave that same grid, so all of them come back through
this.

Which reconstruction runs is read off the sampling mask by
:func:`pulserver.recon.cartesian_recon`, not declared here.

The buffer holds every physical channel: an autocalibration rectangle is
imaging data on the imaging grid, so there is no point before the first line at
which a coil basis could exist, and the compression happens on the way into the
solve instead. A noise scan never reaches the buffer -- it whitens every line
that follows.

Echoes are an axis, not a variant: each is unaliased and filled against the
same sensitivities, estimated from the first, and a single-echo scan is that
loop run once.

One magnitude volume per echo, the echo as the image series, cropped to the
prescribed matrix.

Examples
--------
Calling the module reconstructs an MRD file: the same three hooks an
inline reconstruction is driven through, fed from the file in this
process rather than over a socket.

>>> from pulserver import ReconPlugin
>>> from pulserver.app import cartesian3D_recon
>>> isinstance(cartesian3D_recon.PLUGIN, ReconPlugin)
True

The three hooks are the whole plugin, and nothing else is overridden:

>>> sorted(
...     hook for hook in ("startup", "receive", "recon")
...     if hook in vars(cartesian3D_recon.Cartesian3DRecon)
... )
['receive', 'recon', 'startup']

Reconstruct any 3D Cartesian scan::

    images = cartesian3D_recon("scan.h5")

Or re-instantiate the plugin with different settings, and drive it the
same way::

    plugin = cartesian3D_recon.Cartesian3DRecon(coil_compression=8)
    images = plugin.run("scan.h5")
"""

from __future__ import annotations

__all__ = ["PLUGIN", "Cartesian3DRecon"]

from typing import Any

import numpy as np

from pulserver import ReconContext, ReconPlugin, ReconResult
from pulserver.recon import (
    NLINV,
    AcquisitionFlag,
    cartesian_recon,
    center_crop,
    coil_compress,
    has_acquisition_flag,
    noise_prewhiten,
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
    partial_fourier
        Which estimator fills a truncated readout, ``"pocs"`` or ``"homodyne"``.
    virtual_coils
        Channels to compress the array onto before the solve. A scan with fewer
        physical channels keeps them all.
    calibration_iterations
        Newton steps the sensitivity solve takes. More than the eight NLINV
        defaults to: those are for a whole imaging dataset, and an
        autocalibration block is small enough that the solve is still moving
        at eight.
    device
        Torch device the reconstruction runs on. ``None`` is the CPU.
    """

    def __init__(
        self,
        *,
        regularization: float = 1e-3,
        iterations: int = 40,
        pocs_iterations: int = 12,
        partial_fourier: str = "pocs",
        virtual_coils: int = 8,
        calibration_iterations: int = 16,
        device: Any = None,
    ) -> None:
        super().__init__(
            split_on=AcquisitionFlag.LAST_IN_SEGMENT
            | AcquisitionFlag.LAST_IN_MEASUREMENT,
            reject_flags=AcquisitionFlag.IS_PHASECORR_DATA,
        )
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.pocs_iterations = int(pocs_iterations)
        self.partial_fourier = partial_fourier
        self.virtual_coils = int(virtual_coils)
        self.calibration_iterations = int(calibration_iterations)
        self.device = device

    def startup(self, context: ReconContext) -> None:
        """Lay the scan's buffers out, and start with no maps."""
        super().startup(context)
        self.coil_maps: Any = None
        self.coil_basis: Any = None
        self.noise: Any = None

    def receive(self, acquisition: Any, context: ReconContext) -> Any:
        """Whiten the line, place it, then route the boundary it closed.

        The measurement is tested first: its last line closes the trailing
        segment as well, and only a segment that closed nothing larger is the
        autocalibration rectangle.
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
        if has_acquisition_flag(acquisition, AcquisitionFlag.LAST_IN_SEGMENT):
            return self.recon("calibration", context)
        return None

    def recon(self, branch: str, context: ReconContext) -> list[ReconResult] | None:
        """Calibrate the slab, or reconstruct every echo of it."""
        del context
        buffer = self.buffers[0]

        if branch == "calibration":
            # The autocalibration rectangle, from the first echo: it has the
            # most signal, and every echo is unaliased against the same maps.
            kspace, mask = buffer.select(contrast=0)
            lines = kspace[:, mask.any(axis=-1)].reshape(kspace.shape[0], -1)
            _, self.coil_basis = coil_compress(lines, self.virtual_coils)
            self.coil_maps = NLINV(
                spatial_ndim=3, max_iter=self.calibration_iterations
            )(
                np.einsum("vc,c...->v...", self.coil_basis, kspace)[None],
                mask=mask,
                device=self.device,
            )
            return None

        results = []
        for echo in range(buffer.extents.get("contrast", 1)):
            kspace, mask = buffer.select(contrast=echo)
            if self.coil_basis is not None:
                kspace = np.einsum("vc,c...->v...", self.coil_basis, kspace)
            image = cartesian_recon(
                kspace,
                mask,
                self.coil_maps,
                regularization=self.regularization,
                iterations=self.iterations,
                pocs_iterations=self.pocs_iterations,
                partial_fourier=self.partial_fourier,
                device=self.device,
            )
            results.append(
                ReconResult(
                    center_crop(np.abs(image), buffer.image_shape).transpose(0, 2, 1),
                    reference=-1,
                    series_index=echo,
                    image_type="magnitude",
                    dicom=True,
                )
            )
        return results


PLUGIN = Cartesian3DRecon()
