"""Reconstructing one image from one measurement of it.

The recipes a scan's k-space goes through once it is sorted: the coil-wise
adjoint of a fully sampled acquisition, the CG-SENSE solve of an accelerated
one, the POCS fill of a partial echo, and the density-compensated
non-Cartesian pair of the same two. Each composes a
:mod:`~pulserver.recon.physics` operator with a
:mod:`~pulserver.recon.optim` solver, which is the part every Cartesian
reconstruction repeats and none of them should have to write.

Two and three dimensions are the same recipe, and which one is running is read
off the sampling mask rather than declared: a mask over ``(phase encode,
readout)`` reconstructs a slice, one over ``(partition, phase encode,
readout)`` a slab.

Arrays cross the boundary in whatever namespace they arrive in -- NumPy in,
NumPy out -- so there is no real/complex view juggling around these.
"""

from __future__ import annotations

__all__ = [
    "coil_images",
    "fill_partial_echo",
    "reconstruct_plane",
    "sense",
]

from typing import Any

import numpy as np

from .preprocessing import POCS, ifftc, pipe_menon_dcf


def coil_images(kspace: Any, mask: Any, *, device: Any = None) -> Any:
    """Return one image per coil, from a coil-wise Cartesian adjoint.

    A Cartesian operator built with no sensitivity maps is coil-wise: its
    adjoint zero-fills, inverts and hands back one image per coil, without
    folding a coil combination into the transform.

    Parameters
    ----------
    kspace
        K-space, ``(coil, *spatial)``.
    mask
        Its sampling mask, ``(*spatial)``. Two axes reconstruct a slice, three
        a slab.
    device
        Torch device. ``None`` is the CPU.

    Returns
    -------
    numpy.ndarray
        Complex coil images, shaped like ``kspace``.
    """
    physics = _cartesian(mask, device, img_size=tuple(np.shape(kspace)[1:]))
    return physics.A_adjoint(kspace[None])[0]


def sense(
    kspace: Any,
    mask: Any,
    coil_maps: Any,
    readout: Any,
    *,
    regularization: float = 1e-3,
    iterations: int = 40,
    pocs_iterations: int = 12,
    device: Any = None,
) -> Any:
    """Unalias one measurement against its sensitivities, then fill its echo.

    A slab whose readout was fully sampled separates along it: a centered
    inverse FFT there turns the three-dimensional solve into one
    two-dimensional solve per readout position, all sharing the same
    phase-encode lattice, which is cheaper and better conditioned. A partial
    echo couples the readout, so that case keeps the whole solve.

    Parameters
    ----------
    kspace
        K-space, ``(coil, *spatial)``.
    mask
        Its sampling mask, ``(*spatial)``.
    coil_maps
        Sensitivities, as
        :func:`pulserver.recon.calibration.sensitivities` returns them.
    readout
        Which readout samples were acquired, over the full width.
    regularization, iterations
        Tikhonov weight and iteration ceiling of the CG solve.
    pocs_iterations
        POCS iterations, used only when the echo is partial.
    device
        Torch device. ``None`` is the CPU.

    Returns
    -------
    numpy.ndarray
        The complex image, ``(*spatial)``.
    """
    from . import pics

    dimension = int(np.ndim(mask))
    if dimension == 3 and np.all(readout):
        return _sense_by_readout_plane(
            kspace,
            mask,
            coil_maps,
            regularization=regularization,
            iterations=iterations,
            device=device,
        )

    physics = _cartesian(mask, device, coil_maps=coil_maps)
    image = pics(
        kspace[None],
        physics,
        regularization=regularization,
        iterations=iterations,
    )[0]
    if np.all(readout):
        return image
    # The same operator supplies the centered k-space POCS needs, and the phase
    # constraint it relies on means something only once the aliasing is gone.
    centered = physics.fft(_tensor(image, "complex64", device))
    return _asnumpy(
        fill_partial_echo(
            centered,
            readout,
            pocs_iterations,
            dimension=dimension,
            device=device,
        )
    )


def fill_partial_echo(
    kspace: Any,
    readout: Any,
    iterations: int = 12,
    *,
    dimension: int,
    device: Any = None,
) -> Any:
    """Recover the readout edge a partial echo never acquired.

    Parameters
    ----------
    kspace
        K-space over the full readout width, coil-wise or combined.
    readout
        Which readout samples were acquired.
    iterations
        POCS iterations.
    dimension
        How many trailing axes of ``kspace`` are spatial: 2 for a slice, 3 for
        a slab. Required rather than inferred, because whether a leading axis
        is coils or partitions is the caller's to know, and guessing it wrong
        fills the wrong axis and says nothing.
    device
        Torch device. ``None`` is the CPU.

    Returns
    -------
    array
        The filled k-space, in the namespace of ``kspace``.
    """
    return POCS(dimension=dimension, partial_axis=-1, iterations=iterations)(
        kspace, _tensor(readout, "bool", device)
    )


def reconstruct_plane(
    kspace: Any,
    trajectory: Any,
    image_shape: tuple[int, int],
    *,
    direct: bool = False,
    regularization: float = 1e-3,
    iterations: int = 30,
    device: Any = None,
) -> Any:
    """One non-Cartesian plane, from its measurements and trajectory.

    The density-compensated adjoint NUFFT is an inverse only when the
    acquisition satisfies Nyquist, so ``direct`` finishes with it -- coil
    images root-sum-of-squares combined -- and everything undersampled goes
    through CG-SENSE against adjoint-derived sensitivities, exactly as the
    Cartesian pipeline branches between its adjoint and its solve.

    Parameters
    ----------
    kspace
        Measurements, ``(coil, samples)`` over every view of the plane.
    trajectory
        K-space coordinates, ``(samples, 2)`` in ``[-0.5, 0.5)`` units.
    image_shape
        Reconstructed matrix, ``(h, w)``.
    direct
        Finish with the density-compensated adjoint. Only an inverse for a
        fully sampled plane.
    regularization, iterations
        Tikhonov weight and iteration ceiling of the CG solve.
    device
        Torch device. ``None`` is the CPU.

    Returns
    -------
    numpy.ndarray
        The image, ``(h, w)`` -- complex from the solve, magnitude from the
        direct path's coil combination.
    """
    del device

    from . import pics
    from .calibration import smooth_sensitivities
    from .physics import NonCartesian2D

    data = np.asarray(kspace)
    points = np.asarray(trajectory, dtype=np.float32)
    density = pipe_menon_dcf(points, image_shape)
    n_coils = int(data.shape[0])

    coil_wise = NonCartesian2D(points, image_shape, density=density, n_coils=n_coils)
    images = coil_wise.A_adjoint(data[None])[0]
    if direct:
        return np.sqrt(np.sum(np.abs(images) ** 2, axis=0))

    unaliasing = NonCartesian2D(
        points,
        image_shape,
        coil_maps=smooth_sensitivities(images),
        density=density,
        n_coils=n_coils,
    )
    # The SENSE solve keeps a singleton coil axis, so index past both batch and
    # channel to reach the plane.
    return pics(
        data[None],
        unaliasing,
        regularization=regularization,
        iterations=iterations,
    )[0, 0]


# %% private module subroutines


def _cartesian(mask: Any, device: Any, **kwargs: Any) -> Any:
    """The Cartesian operator this mask describes, of its own dimensionality.

    Raises
    ------
    ValueError
        If the mask is neither two- nor three-dimensional.
    """
    from .physics import Cartesian2D, Cartesian3D

    dimension = int(np.ndim(mask))
    if dimension not in (2, 3):
        raise ValueError(f"a sampling mask is 2D or 3D, got {dimension} dimensions")
    operator = Cartesian2D if dimension == 2 else Cartesian3D
    # A coil-wise adjoint carries no maps and needs one more leading axis than
    # a SENSE operator, whose channel axis the maps supply.
    lattice = _tensor(mask, "float32", device)
    lattice = lattice[None, None] if "img_size" in kwargs else lattice[None]
    return operator(lattice, **kwargs)


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
    import torch

    from . import pics
    from .physics import Cartesian2D

    hybrid = ifftc(np.asarray(kspace), axes=-1)
    _, n_z, n_y, n_x = hybrid.shape
    data = np.moveaxis(hybrid, -1, 0)  # (x, coil, partition, line)

    maps = torch.as_tensor(_asnumpy(coil_maps)[0])  # (coil, partition, line, x)
    maps = maps.movedim(-1, 0).contiguous()  # (x, coil, partition, line)
    plane = np.asarray(mask).any(axis=-1).astype(np.float32)  # (partition, line)
    lattice = _tensor(
        np.broadcast_to(plane, (n_x, 1, n_z, n_y)).copy(), "float32", device
    )
    if device is not None:
        maps = maps.to(device)

    image = pics(
        data,
        Cartesian2D(lattice, maps),
        regularization=regularization,
        iterations=iterations,
    )
    return np.moveaxis(_asnumpy(image), 0, -1)


def _tensor(array: Any, dtype: str, device: Any) -> Any:
    import torch

    return torch.as_tensor(
        array, dtype=getattr(torch, dtype), device="cpu" if device is None else device
    )


def _asnumpy(array: Any) -> np.ndarray:
    return array.cpu().numpy() if hasattr(array, "cpu") else np.asarray(array)
