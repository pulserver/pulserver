"""Any 3D Cartesian scan, one volume per echo.

Calling this module reconstructs an MRD file; ``PLUGIN`` is the same
reconstruction behind the stream contract, driven live by the scanner.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "Cartesian3DRecon"]

from typing import Any

import numpy as np

from pulserver.recon import ReconContext, ReconPlugin, ReconResult
from pulserver.recon import (
    NLINV,
    NoiseAdjust,
    cartesian_recon,
    image_result,
)
from pulserver.mrd import (
    AcquisitionFlag,
    coil_compress,
)


class Cartesian3DRecon(ReconPlugin):
    """Reconstruct a 3D Cartesian scan, one volume per echo.

    The same recipes as :mod:`pulserver.app.cartesian2D_recon`, over a slab:
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
        Torch device the reconstruction runs on. ``"auto"`` is the host's
        GPU when it has one, and the CPU when it does not.

    Examples
    --------
    Calling the module reconstructs an MRD file, through the same hooks a
    scanner's live stream is driven through. Its settings are the arguments
    of that call:

    >>> import inspect
    >>> from pulserver.app import cartesian3D_recon
    >>> "virtual_coils" in inspect.signature(cartesian3D_recon).parameters
    True

    Reconstruct any 3D Cartesian scan::

        images = cartesian3D_recon("scan.h5")
        images = cartesian3D_recon("scan.h5", virtual_coils=4, partial_fourier="homodyne")

    A slab, fully sampled and then twofold accelerated with a calibration
    block at the centre of the phase-encode axis. Which reconstruction runs is
    read off the mask, so both are the same call; the panels are the volume's
    central partition.

    .. plot::

       from pulserver.app import cartesian3D_recon
       from _figures import images, slab_example

       plugin = cartesian3D_recon.PLUGIN
       full = slab_example(plugin, size=32, coils=8, partitions=8)
       fast = slab_example(
           plugin, size=32, coils=8, partitions=8, acceleration=2, n_acs=10
       )

       images(
           [
               ("object", full.truth),
               ("fully sampled", full.image),
               ("2x, CG-SENSE against NLINV maps", fast.image),
           ],
           title="one partition of the slab",
       )
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
        device: Any = "auto",
    ) -> None:
        super().__init__(
            chain=[NoiseAdjust()],
            branches={
                AcquisitionFlag.LAST_IN_MEASUREMENT: "imaging",
                AcquisitionFlag.LAST_IN_SEGMENT: "calibration",
            },
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
            results.append(image_result(image, buffer, series_index=echo))
        return results


PLUGIN = Cartesian3DRecon()
