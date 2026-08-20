"""Any 2D Cartesian scan, one image per slice and echo.

What a reconstruction needs to know is how k-space was sampled, not which
contrast the sequence was after -- a spin echo, a balanced SSFP and a gradient
echo all leave one Cartesian grid per slice, and all of them come back through
this. So there is one plugin here rather than one per sequence.

A streamed reconstruction. The header describes the encoding space and each
acquisition carries its own counters, so every line is placed as it arrives and
``receive`` routes the two boundaries the sequence flags: a segment that closes
without closing the slice is the calibration block, and the slice itself is an
image. The calibration lines are imaging data too, so there is one buffer and
no second grid.

That is also why the buffer holds every physical channel. A calibration that is
imaging data arrives on the imaging grid, so there is no point before the first
line at which a coil basis could exist; the compression happens once the slice
is complete, on the way into the solve, against the basis the calibration
branch established. A noise scan, when the scanner sends one, is not imaging
data and never reaches the buffer -- it whitens every line that follows.

Which reconstruction runs is read off the sampling mask by
:func:`pulserver.recon.cartesian_recon`, not declared here.

Echoes are an axis, not a variant: a scan whose ``ECO`` label fills the
``contrast`` counter reconstructs each echo against the same sensitivities --
estimated from the first, whose calibration block has the most signal -- and a
single-echo scan is that loop run once.

One magnitude image per ``(slice, echo)``, the echo as the image series,
cropped to the prescribed matrix.

Examples
--------
Calling the module reconstructs an MRD file: the same three hooks an
inline reconstruction is driven through, fed from the file in this
process rather than over a socket.

>>> from pulserver import ReconPlugin
>>> from pulserver.app import cartesian2D_recon
>>> isinstance(cartesian2D_recon.PLUGIN, ReconPlugin)
True

The three hooks are the whole plugin, and nothing else is overridden:

>>> sorted(
...     hook for hook in ("startup", "receive", "recon")
...     if hook in vars(cartesian2D_recon.Cartesian2DRecon)
... )
['receive', 'recon', 'startup']

Reconstruct any 2D Cartesian scan::

    images = cartesian2D_recon("scan.h5")

Or re-instantiate the plugin with different settings, and drive it the
same way::

    plugin = cartesian2D_recon.Cartesian2DRecon(coil_compression=8)
    images = plugin.run("scan.h5")
"""

from __future__ import annotations

__all__ = ["PLUGIN", "Cartesian2DRecon"]

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


class Cartesian2DRecon(ReconPlugin):
    """Reconstruct a 2D Cartesian scan, one image per slice and echo.

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
            split_on=AcquisitionFlag.LAST_IN_SEGMENT | AcquisitionFlag.LAST_IN_SLICE,
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
        """Lay the scan's buffers out, and start with no maps and no slice."""
        super().startup(context)
        self.coil_maps: dict[int, Any] = {}
        self.coil_basis: Any = None
        self.noise: Any = None
        self.slice = 0

    def receive(self, acquisition: Any, context: ReconContext) -> Any:
        """Whiten the line, place it, then route the boundary it closed.

        The slice is tested first: the last line of a slice closes its trailing
        segment as well, and only a segment that closed nothing larger is the
        calibration block.
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
        self.slice = int(acquisition.idx.slice)
        if has_acquisition_flag(acquisition, AcquisitionFlag.LAST_IN_SLICE):
            return self.recon("imaging", context)
        if has_acquisition_flag(acquisition, AcquisitionFlag.LAST_IN_SEGMENT):
            return self.recon("calibration", context)
        return None

    def recon(self, branch: str, context: ReconContext) -> list[ReconResult] | None:
        """Calibrate the slice, or reconstruct every echo of it."""
        del context
        buffer = self.buffers[0]
        index = self.slice

        if branch == "calibration":
            # The autocalibration block, from the first echo: it has the most
            # signal, and every echo is unaliased against the same maps. The
            # array's principal channels come off the same block, so the maps
            # are estimated in the basis the imaging will be compressed onto.
            kspace, mask = buffer.select(slice=index, contrast=0)
            lines = kspace[:, mask.any(axis=-1)].reshape(kspace.shape[0], -1)
            _, self.coil_basis = coil_compress(lines, self.virtual_coils)
            self.coil_maps[index] = NLINV(
                spatial_ndim=2, max_iter=self.calibration_iterations
            )(
                np.einsum("vc,c...->v...", self.coil_basis, kspace)[None],
                mask=mask,
                device=self.device,
            )
            return None

        results = []
        for echo in range(buffer.extents.get("contrast", 1)):
            kspace, mask = buffer.select(slice=index, contrast=echo)
            if self.coil_basis is not None:
                kspace = np.einsum("vc,c...->v...", self.coil_basis, kspace)
            image = cartesian_recon(
                kspace,
                mask,
                self.coil_maps.get(index),
                regularization=self.regularization,
                iterations=self.iterations,
                pocs_iterations=self.pocs_iterations,
                partial_fourier=self.partial_fourier,
                device=self.device,
            )
            results.append(
                ReconResult(
                    center_crop(np.abs(image), buffer.image_shape).transpose(),
                    reference=-1,
                    series_index=echo,
                    image_type="magnitude",
                    dicom=True,
                )
            )
        return results


PLUGIN = Cartesian2DRecon()
