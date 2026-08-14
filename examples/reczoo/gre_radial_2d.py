"""Reconstruction for :mod:`pulserver.seqzoo.gre_radial_2d`.

Model-based non-Cartesian reconstruction, per slice: Pipe--Menon density
compensation, a coil-wise adjoint NUFFT for the sensitivity estimate --
low-pass filtered and normalised to unit root-sum-of-squares, the classic
adaptive-combine approximation -- and a CG-SENSE solve against the
:class:`pulserver.recon.physics.NonCartesian2D` operator built from the
same trajectory.

The trajectory arrives per acquisition, as MRD carries it, scaled to
MRI-NUFFT's ``[-0.5, 0.5)`` units -- what the LiveSDK's enrichment writes.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = [
    "PLUGIN",
    "GreRadial2DRecon",
    "reconstruct_plane",
    "smooth_sensitivities",
]

from typing import Any

import numpy as np

from pulserver import AcquisitionBucket, ReconContext, ReconPlugin, ReconResult
from pulserver.recon import has_acquisition_flag
from pulserver.recon.preprocessing import pipe_menon_dcf, recon_shape, receiver_channels


def smooth_sensitivities(coil_images: Any) -> Any:
    """Sensitivities from coil images, by Fourier low-pass and normalisation.

    Parameters
    ----------
    coil_images
        Complex coil images, ``(coils, h, w)``.

    Returns
    -------
    torch.Tensor
        Unit root-sum-of-squares maps of the same shape.
    """
    import torch

    images = torch.as_tensor(coil_images)
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(images, dim=(-2, -1))), dim=(-2, -1)
    )
    height, width = spectrum.shape[-2:]
    window_h = torch.hann_window(height, dtype=spectrum.real.dtype)
    window_w = torch.hann_window(width, dtype=spectrum.real.dtype)
    spectrum = spectrum * (window_h[:, None] * window_w[None, :]) ** 4
    smoothed = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(spectrum, dim=(-2, -1))), dim=(-2, -1)
    )
    norm = torch.sqrt(torch.sum(torch.abs(smoothed) ** 2, dim=0, keepdim=True))
    return smoothed / torch.clamp(norm, min=1e-12 * float(norm.max()))


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
    """One plane, from its measurements and trajectory.

    The density-compensated adjoint NUFFT is an inverse only when the
    acquisition satisfies Nyquist, so ``direct`` finishes with it -- coil
    images root-sum-of-squares combined -- and everything undersampled goes
    through CG-SENSE against adjoint-derived sensitivities, exactly as the
    Cartesian pipeline branches between its adjoint and its solve.

    Parameters
    ----------
    kspace
        Measurements, ``(coils, samples)`` over every view of the plane.
    trajectory
        K-space coordinates, ``(samples, 2)`` in ``[-0.5, 0.5)`` units.
    image_shape
        Reconstructed matrix, ``(h, w)``.
    direct
        Finish with the density-compensated adjoint. Only valid for a fully
        sampled plane.
    regularization, iterations
        Tikhonov weight and iteration ceiling of the CG solve.
    device
        Torch device. ``None`` is the CPU.

    Returns
    -------
    torch.Tensor
        The image, ``(h, w)`` -- complex from the solve, magnitude from the
        direct path's coil combination.
    """
    import torch

    from pulserver.recon import pics
    from pulserver.recon.physics import NonCartesian2D

    device = "cpu" if device is None else device
    data = torch.as_tensor(np.asarray(kspace), dtype=torch.complex64, device=device)
    points = np.asarray(trajectory, dtype=np.float32)

    density = pipe_menon_dcf(points, image_shape)

    n_coils = int(data.shape[0])
    coil_wise = NonCartesian2D(
        points,
        image_shape,
        density=density,
        n_coils=n_coils,
    )
    coils = coil_wise.A_adjoint(torch.view_as_real(data[None]))
    # Coil-wise images come back (batch, 2, coils, h, w) flattened into the
    # channel axis.
    coil_images = torch.view_as_complex(
        coils.reshape(1, 2, n_coils, *image_shape).movedim(1, -1).contiguous()
    )[0]

    if direct:
        return torch.sqrt(torch.sum(torch.abs(coil_images) ** 2, dim=0))

    maps = smooth_sensitivities(coil_images)
    sense = NonCartesian2D(
        points,
        image_shape,
        coil_maps=maps,
        density=density,
        n_coils=n_coils,
    )
    solution = pics(
        torch.view_as_real(data[None]),
        sense,
        regularization=regularization,
        iterations=iterations,
    )
    return torch.view_as_complex(solution.movedim(1, -1).contiguous())[0]


class GreRadial2DRecon(ReconPlugin):
    """Reconstruct a 2D non-Cartesian gradient echo, one image per slice.

    Parameters
    ----------
    mode
        ``"auto"`` finishes a fully sampled slice with the
        density-compensated adjoint and sends everything undersampled
        through CG-SENSE; ``"direct"`` and ``"pics"`` force one branch. A
        slice counts as fully sampled when its view count reaches the radial
        Nyquist count, ``ceil(pi/2 * matrix)``.
    regularization
        Tikhonov weight of the CG-SENSE solve.
    iterations
        Maximum CG iterations.
    device
        Torch device the reconstruction runs on. ``None`` is the CPU.
    """

    def __init__(
        self,
        *,
        mode: str = "auto",
        regularization: float = 1e-3,
        iterations: int = 30,
        device: Any = None,
    ) -> None:
        super().__init__(
            split_on=("ACQ_LAST_IN_MEASUREMENT",),
            reject_flags=("ACQ_IS_NOISE_MEASUREMENT", "ACQ_IS_PHASECORR_DATA"),
        )
        if mode not in ("auto", "direct", "pics"):
            raise ValueError(f"mode must be auto, direct or pics, got {mode!r}")
        self.mode = mode
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.device = device

    def startup(self, context: ReconContext) -> None:
        """Size the image from the header and open one bin per slice."""
        self.image_shape = recon_shape(context.header)
        self.coils = receiver_channels(context.header)
        self.views: dict[int, list[tuple[Any, Any]]] = {}

    def receive(self, acquisition: Any, context: ReconContext) -> None:
        """File one view's data and trajectory under its slice."""
        del context
        self.views.setdefault(int(acquisition.idx.slice), []).append(
            (np.asarray(acquisition.data), np.asarray(acquisition.traj))
        )

    def recon(
        self, bucket: AcquisitionBucket, context: ReconContext
    ) -> list[ReconResult] | None:
        """Reconstruct every slice once the measurement is complete."""
        del context
        last = bucket.acquisitions[-1]
        if not has_acquisition_flag(last, "ACQ_LAST_IN_MEASUREMENT"):
            return None

        nyquist = int(np.ceil(np.pi / 2 * max(self.image_shape)))
        results = []
        for index in sorted(self.views):
            data = np.concatenate([view for view, _ in self.views[index]], axis=-1)
            trajectory = np.concatenate(
                [points[:, :2] for _, points in self.views[index]], axis=0
            )
            direct = self.mode == "direct" or (
                self.mode == "auto" and len(self.views[index]) >= nyquist
            )
            image = reconstruct_plane(
                data,
                trajectory,
                self.image_shape,
                direct=direct,
                regularization=self.regularization,
                iterations=self.iterations,
                device=self.device,
            )
            results.append(
                ReconResult(
                    np.abs(_to_numpy(image)).transpose(),
                    reference=-1,
                    image_index=index,
                    image_type="magnitude",
                    dicom=True,
                )
            )
        return results


def _to_numpy(array: Any) -> np.ndarray:
    """Bring a reconstruction back to NumPy, whether it is Torch or already so."""
    return array.cpu().numpy() if hasattr(array, "cpu") else np.asarray(array)


PLUGIN = GreRadial2DRecon()
