"""Reconstructing a filled Cartesian buffer, the sampling selecting how.

A Cartesian scan states what it sampled: the buffer's mask says which phase
encodes it took and which readout samples it kept, and those two facts decide
what has to happen to invert it. This is that decision and the three
reconstructions it chooses between, in one place, so a plugin composes it in a
hook rather than restating the branch per sampling.
"""

from __future__ import annotations

__all__ = ["cartesian_recon"]

from typing import Any

import numpy as np

from .postprocessing import coil_combine
from .preprocessing import fftc, fill_partial_echo, ifftc


def cartesian_recon(
    kspace: Any,
    mask: Any,
    coil_maps: Any = None,
    *,
    regularization: float = 1e-3,
    iterations: int = 40,
    pocs_iterations: int = 12,
    device: Any = None,
) -> np.ndarray:
    """Reconstruct one Cartesian k-space, the mask selecting which way.

    Three reconstructions, chosen by what the scan sampled rather than by what
    a caller declares. A phase encode with no samples was skipped, so parallel
    imaging is needed; a readout sample missing from every encode is echo that
    was never acquired, so the partial-Fourier constraint is needed; a scan
    with neither is inverted by the Fourier transform itself.

    Parallel imaging runs first when both are true, because the slowly varying
    phase the partial-Fourier constraint rests on only means something once the
    image is unaliased.

    A slab whose readout was fully sampled separates along it: a centered
    inverse Fourier transform there turns the three-dimensional solve into one
    two-dimensional solve per readout position, all sharing the same
    phase-encode lattice, which is cheaper and better conditioned than solving
    the volume whole. A partial echo couples the readout, so that case keeps
    the volume together.

    Parameters
    ----------
    kspace
        The buffer, ``(coil, phase_encode, readout)`` for a slice or
        ``(coil, partition, phase_encode, readout)`` for a slab.
    mask
        Where data landed: ``kspace`` without its coil axis.
    coil_maps
        Sensitivities to unalias against, ``(1, coil, *grid)`` as
        :class:`~pulserver.recon.NLINV` answers. ``None`` estimates them from
        the measurement itself, which needs a fully sampled centre.
    regularization
        Tikhonov weight of the CG-SENSE solve.
    iterations
        Maximum CG iterations.
    pocs_iterations
        Partial-echo POCS iterations.
    device
        Torch device the reconstruction runs on. ``None`` is the CPU.

    Returns
    -------
    numpy.ndarray
        One complex image, shaped like ``mask``.

    Raises
    ------
    ValueError
        If ``mask`` is neither a plane nor a volume.
    """
    from .calibration import NLINV
    from .optim import pics
    from .physics import Cartesian2D, Cartesian3D

    kspace = np.asarray(kspace)
    mask = np.asarray(mask)
    if mask.ndim not in (2, 3):
        raise ValueError(
            f"mask must be (phase_encode, readout) or (partition, phase_encode, "
            f"readout), got shape {mask.shape}"
        )
    spatial = tuple(range(-mask.ndim, 0))
    encodes = mask.any(axis=-1)
    readout = mask.reshape(-1, mask.shape[-1]).any(axis=0)

    if encodes.all():
        # Fully sampled k-space is zero outside the mask, so the coil-wise
        # adjoint is the centered inverse transform itself.
        coils = (
            ifftc(kspace, axes=spatial)
            if readout.all()
            else fill_partial_echo(
                kspace, readout, pocs_iterations, dimension=mask.ndim
            )
        )
        return coil_combine(coils, coil_axis=0)

    if coil_maps is None:
        coil_maps = NLINV(spatial_ndim=mask.ndim)(
            kspace[None], mask=mask, device=device
        )

    if mask.ndim == 3 and readout.all():
        image = _sense_by_readout_plane(
            kspace,
            mask,
            coil_maps,
            regularization=regularization,
            iterations=iterations,
            device=device,
        )
    else:
        physics = Cartesian2D if mask.ndim == 2 else Cartesian3D
        image = pics(
            kspace[None],
            physics(mask[None], coil_maps, device=device),
            regularization=regularization,
            iterations=iterations,
        )[0]

    if readout.all():
        return image
    return fill_partial_echo(
        fftc(image, axes=spatial), readout, pocs_iterations, dimension=mask.ndim
    )


# %% private module subroutines


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
    from .optim import pics
    from .physics import Cartesian2D

    hybrid = ifftc(kspace, axes=-1)
    _, n_z, n_y, n_x = hybrid.shape
    data = np.moveaxis(hybrid, -1, 0)  # (x, coil, partition, line)

    maps = np.moveaxis(np.asarray(coil_maps)[0], -1, 0)  # (x, coil, partition, line)
    plane = mask.any(axis=-1).astype(np.float32)  # (partition, line)
    lattice = np.broadcast_to(plane, (n_x, 1, n_z, n_y)).copy()

    image = pics(
        data,
        Cartesian2D(lattice, np.ascontiguousarray(maps), device=device),
        regularization=regularization,
        iterations=iterations,
    )
    return np.moveaxis(image, 0, -1)
