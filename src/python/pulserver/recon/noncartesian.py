"""Reconstructing one non-Cartesian measurement, the sampling selecting how.

Radial, spiral, PROPELLER, a ZTE shell set: what a reconstruction needs is the
trajectory the acquisitions carry, not the shape the sequence drew with it. So
there is one recipe here, and the only question it answers per scan is whether
the views reach Nyquist -- a fully sampled measurement is finished by the
density-compensated adjoint, and everything else is solved.
"""

from __future__ import annotations

__all__ = ["noncartesian_recon"]

from typing import Any

import numpy as np


def noncartesian_recon(
    data: Any,
    trajectory: Any,
    image_shape: tuple[int, ...],
    *,
    mode: str = "auto",
    n_views: int | None = None,
    density: Any = None,
    regularization: float = 1e-3,
    iterations: int = 30,
    calibration_width: int = 24,
    device: Any = "auto",
) -> np.ndarray:
    """Reconstruct one non-Cartesian measurement, in the plane or the volume.

    Two reconstructions, chosen by how much of k-space the views covered. A
    measurement that reaches the radial Nyquist count is inverted by its own
    density-compensated adjoint, one image per coil, root-sum-of-squares
    combined. Anything below it is solved: NLINV sensitivities from the samples
    inside the calibration radius, then CG-SENSE against the same trajectory.

    Whether the operator is two- or three-dimensional follows from
    ``image_shape``, so a plane and a volume are the same call.

    Parameters
    ----------
    data
        The measurement, ``(coils, samples)`` -- the views laid end to end.
    trajectory
        Where those samples were taken, ``(samples, dimensions)``, in
        MRI-NUFFT's ``[-0.5, 0.5)`` units.
    image_shape
        The matrix to reconstruct, ``(h, w)`` or ``(d, h, w)``.
    mode
        ``"auto"`` reads the branch off the view count, ``"direct"`` and
        ``"pics"`` force one. A view count that cannot be checked -- ``"auto"``
        with no ``n_views`` -- is solved rather than assumed complete.
    n_views
        How many views the samples came from, which is what ``"auto"`` judges
        against ``ceil(pi / 2 * max(image_shape))``.
    density
        Density-compensation weights. ``None`` computes Pipe--Menon weights.
    regularization
        Tikhonov weight of the CG-SENSE solve.
    iterations
        Maximum CG iterations.
    calibration_width
        Width of the centred square or cube NLINV solves the sensitivities
        over.
    device
        Torch device the reconstruction runs on. ``"auto"`` is the host's
        GPU when it has one, and the CPU when it does not.

    Returns
    -------
    numpy.ndarray
        One image, shaped ``image_shape``.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the three, or ``image_shape`` is neither a
        plane nor a volume.

    Examples
    --------
    One call from a filled buffer to an image: the density-compensated adjoint
    when the views do not reach Nyquist, CG-SENSE when they do.

    .. plot::

       import numpy as np
       import pulserver.recon as recon
       from _figures import images, phantom, radial_spokes

       truth, coil_maps = phantom(64, coils=4)
       spokes = radial_spokes(64, 24)
       physics = recon.NonCartesian2D(spokes, (64, 64), coil_maps=coil_maps[0])

       measured = physics.A(truth)[0]
       image = recon.noncartesian_recon(
           measured, np.asarray(spokes).reshape(-1, 2), (64, 64), mode="direct"
       )

       images(
           [("truth", truth[0]), ("density-compensated adjoint", image)],
           title="noncartesian_recon over 24 radial spokes",
       )
    """
    from .calibration import NLINV
    from .execution import _resolve_device
    from .optim import pics
    from .physics import NonCartesian2D, NonCartesian3D
    from ..mrd._images import as_numpy
    from ..mrd._arrays import pipe_menon_dcf

    if mode not in ("auto", "direct", "pics"):
        raise ValueError(f"mode must be auto, direct or pics, got {mode!r}")
    if len(image_shape) not in (2, 3):
        raise ValueError(
            f"image_shape must be (h, w) or (d, h, w), got {image_shape!r}"
        )

    operator = NonCartesian2D if len(image_shape) == 2 else NonCartesian3D
    data = np.asarray(data)
    trajectory = np.asarray(trajectory, dtype=np.float32)
    if density is None:
        density = pipe_menon_dcf(trajectory, image_shape)
    n_coils = int(data.shape[0])

    device = _resolve_device(device)
    if device is not None:
        # A NUFFT plans on the device its trajectory is on, so placing the
        # samples places the whole reconstruction.
        import torch

        data = torch.as_tensor(data).to(device)
        trajectory = torch.as_tensor(trajectory).to(device)
        density = torch.as_tensor(density).to(device)

    if _direct(mode, n_views, image_shape):
        # The density-compensated adjoint is an inverse only at Nyquist, and a
        # coil-wise one at that: combine by root-sum-of-squares.
        coil_wise = operator(trajectory, image_shape, density=density, n_coils=n_coils)
        coils = as_numpy(coil_wise.A_adjoint(data[None])[0])
        return np.sqrt(np.sum(np.abs(coils) ** 2, axis=0))

    maps = NLINV(spatial_ndim=len(image_shape), calibration_width=calibration_width)(
        data,
        trajectory=trajectory,
        image_shape=image_shape,
        density=density,
        device=device,
    )
    unaliasing = operator(
        trajectory,
        image_shape,
        coil_maps=maps,
        density=density,
        n_coils=n_coils,
        # The whole kernel, not the locations the samples reached. This solve
        # is the regime where the normal operator has eigenvalues at zero, so
        # dropping any of the transfer carries the smallest ones negative and
        # CG stops on a non-positive recurrence. One plane's kernel is small
        # enough that keeping all of it costs nothing worth having.
        toeplitz={"compress": False},
    )
    # The SENSE solve keeps a singleton coil axis, so index past both batch and
    # channel to reach the image.
    return as_numpy(
        pics(
            data[None],
            unaliasing,
            regularization=regularization,
            iterations=iterations,
        )[0, 0]
    )


# %% private module subroutines


def _direct(mode: str, n_views: int | None, image_shape: tuple[int, ...]) -> bool:
    """Whether the adjoint finishes this measurement, or a solve has to."""
    if mode != "auto":
        return mode == "direct"
    nyquist = int(np.ceil(np.pi / 2 * max(image_shape)))
    return n_views is not None and n_views >= nyquist
