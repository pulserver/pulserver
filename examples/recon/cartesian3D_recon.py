"""Any 3D Cartesian scan, one volume per echo.

The same recipes as :mod:`pulserver.app.recon.cartesian2D_recon`, over a slab:
one buffer spanning ``(partition, line, readout)``, filled as the acquisitions
arrive, calibrated when the autocalibration rectangle's segment closes, and
reconstructed once at the end of the measurement -- a 3D scan has one slab, so
there is no per-slice boundary to reconstruct at. An MPRAGE, a fast spin echo
and a balanced SSFP all leave that same grid, so all of them come back through
this.

Which of three reconstructions runs is read off the sampling mask rather than
declared: the coil-wise adjoint when everything is there, POCS when the readout
is truncated, CG-SENSE against 3D NLINV maps when phase encodes are missing on
either axis. Parallel imaging comes first, because the phase constraint POCS
relies on only means something once the image is unaliased. A slab whose
readout was fully sampled separates along it: a centered inverse FFT there
turns the three-dimensional solve into one two-dimensional solve per readout
position, all sharing the same phase-encode lattice, which is cheaper and
better conditioned.

Echoes are an axis, not a variant: each is unaliased and filled against the
same sensitivities, estimated from the first, and a single-echo scan is that
loop run once.

One magnitude volume per echo, the echo as the image series, cropped to the
prescribed matrix.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "Cartesian3DRecon"]

from typing import Any

import numpy as np

from pulserver import AcquisitionBucket, ReconContext, ReconPlugin, ReconResult
from pulserver.recon import (
    NLINV,
    AcquisitionFlag,
    Cartesian2D,
    Cartesian3D,
    center_crop,
    coil_combine,
    echo_count,
    fftc,
    fill_partial_echo,
    ifftc,
    pics,
    recon_volume,
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
    device
        Torch device the reconstruction runs on. ``None`` is the CPU.
    """

    def __init__(
        self,
        *,
        regularization: float = 1e-3,
        iterations: int = 40,
        pocs_iterations: int = 12,
        device: Any = None,
    ) -> None:
        super().__init__(
            split_on=AcquisitionFlag.LAST_IN_SEGMENT
            | AcquisitionFlag.LAST_IN_MEASUREMENT,
            reject_flags=AcquisitionFlag.IS_NOISE_MEASUREMENT
            | AcquisitionFlag.IS_PHASECORR_DATA,
        )
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.pocs_iterations = int(pocs_iterations)
        self.device = device

    def startup(self, context: ReconContext) -> None:
        """Lay the scan's buffers out and note the matrix to crop to."""
        super().startup(context)
        self.n_echoes = echo_count(context.header)
        self.image_shape = recon_volume(context.header)
        self.coil_maps: Any = None

    def recon(
        self, bucket: AcquisitionBucket, context: ReconContext
    ) -> list[ReconResult] | None:
        """Calibrate or reconstruct, according to which boundary ended this bucket."""
        del context
        buffer = self.buffers[0]

        if AcquisitionFlag.LAST_IN_MEASUREMENT not in bucket.trigger:
            kspace, mask = buffer.select(contrast=0)
            self.coil_maps = NLINV(spatial_ndim=3)(
                kspace[None], mask=mask, device=self.device
            )
            return None

        results = []
        for echo in range(self.n_echoes):
            echo_kspace, echo_mask = buffer.select(contrast=echo)
            encodes = echo_mask.any(axis=-1)
            readout = echo_mask.reshape(-1, echo_mask.shape[-1]).any(axis=0)

            if encodes.all():
                # Fully sampled k-space is zero outside the mask, so the
                # coil-wise adjoint is the centered inverse FFT itself.
                coils = (
                    ifftc(echo_kspace, axes=(-3, -2, -1))
                    if readout.all()
                    else fill_partial_echo(
                        echo_kspace, readout, self.pocs_iterations, dimension=3
                    )
                )
                image = coil_combine(coils, coil_axis=0)
            else:
                maps = self.coil_maps
                if maps is None:
                    maps = NLINV(spatial_ndim=3)(
                        echo_kspace[None], mask=echo_mask, device=self.device
                    )
                    self.coil_maps = maps
                if readout.all():
                    image = _sense_by_readout_plane(
                        echo_kspace,
                        echo_mask,
                        maps,
                        regularization=self.regularization,
                        iterations=self.iterations,
                        device=self.device,
                    )
                else:
                    # A partial echo couples the readout, so this case keeps
                    # the whole 3D solve and fills the echo last, once the
                    # aliasing its phase constraint depends on is gone.
                    image = pics(
                        echo_kspace[None],
                        Cartesian3D(echo_mask[None], maps, device=self.device),
                        regularization=self.regularization,
                        iterations=self.iterations,
                    )[0]
                    image = fill_partial_echo(
                        fftc(image, axes=(-3, -2, -1)),
                        readout,
                        self.pocs_iterations,
                        dimension=3,
                    )

            image = center_crop(np.abs(image), self.image_shape)
            results.append(
                ReconResult(
                    image.transpose(0, 2, 1),
                    reference=-1,
                    series_index=echo,
                    image_type="magnitude",
                    dicom=True,
                )
            )
        return results


def _sense_by_readout_plane(
    kspace: Any,
    mask: Any,
    coil_maps: Any,
    *,
    regularization: float,
    iterations: int,
    device: Any,
) -> np.ndarray:
    """CG-SENSE decoupled over a fully sampled readout: one 2D solve per column.

    The centered inverse FFT along the readout takes the volume to
    ``(coil, partition, line, x)`` hybrid space, where every readout position
    ``x`` is an independent 2D ``(partition, line)`` parallel-imaging problem
    sharing the one phase-encode lattice. Stacking the columns on the batch
    axis solves them together against a batched
    :class:`~pulserver.recon.physics.Cartesian2D`.
    """
    hybrid = ifftc(np.asarray(kspace), axes=-1)
    _, n_z, n_y, n_x = hybrid.shape
    data = np.moveaxis(hybrid, -1, 0)  # (x, coil, partition, line)

    maps = np.moveaxis(np.asarray(coil_maps)[0], -1, 0)  # (x, coil, partition, line)
    plane = np.asarray(mask).any(axis=-1).astype(np.float32)  # (partition, line)
    lattice = np.broadcast_to(plane, (n_x, 1, n_z, n_y)).copy()

    image = pics(
        data,
        Cartesian2D(lattice, np.ascontiguousarray(maps), device=device),
        regularization=regularization,
        iterations=iterations,
    )
    return np.moveaxis(image, 0, -1)


PLUGIN = Cartesian3DRecon()
