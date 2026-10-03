"""Non-Cartesian parallel-imaging compressed-sensing reconstruction with bartorch, as ``bart nlinv -t`` and ``bart pics -t``."""

from __future__ import annotations

__all__ = ["PLUGIN", "NufftRecon"]

import numpy as np

from ...mrd._acquisitions import AcquisitionFlag
from ...mrd._images import center_crop
from .._buffers import ReconData
from ..gadgets import Prewhiten
from ..plugin import ReconContext, ReconPlugin, ReconResult
from .cartesian import averaged


class NufftRecon(ReconPlugin):
    """Image of each slice, contrast, cardiac phase, set and repetition of a non-Cartesian scan, a ``pics`` solve made as each closes.

    Each readout is whitened with the noise measurement of the stream
    (:class:`~pulserver.recon.Prewhiten`); noise readouts are consumed by the
    whitening and phase-correction readouts are rejected. The coil
    sensitivities are fitted by nonlinear inversion to the samples of the unit
    near the k-space centre (:func:`bartorch.apps.nlinv_maps`,
    ``bart nlinv -m 1 -t``), whatever calibration the stream or the exam
    holds: :func:`~pulserver.recon.coil_maps` estimates from Cartesian
    calibration k-space only. The image minimises
    ``|P F S x - y|^2 + lambda |W x|_1`` over them (``bart pics -t -R W``),
    ``F`` bartorch's NUFFT, so no density compensation is assumed of the
    trajectory; the step is the reciprocal of the largest eigenvalue of the
    normal operator (``bart pics -e``). The trajectory is the one the proxy's
    enrichment writes, which
    :meth:`~pulserver.recon.ReconBuffer.grid_trajectory` scales to the image
    grid. A stack of spokes or spirals is Fourier transformed along its
    partitions first and fitted partition by partition. Images close as
    :class:`~pulserver.recon.handlers.cartesian.CartesianRecon` makes them,
    their averages summed, and their values are those of the solve, unscaled.
    bartorch is imported when the first image is made, which the ``coils``
    extra installs.

    Parameters
    ----------
    wavelet
        ``lambda``, relative to the data scaling ``pics`` estimates.
    iterations
        Iterations of the solve.
    """

    def __init__(self, wavelet: float = 0.005, iterations: int = 30) -> None:
        super().__init__(
            gadgets=[Prewhiten()],
            triggers={"imaging": AcquisitionFlag.LAST_IN_SLICE},
            axes=("average",),
            reject_flags=AcquisitionFlag.IS_PHASECORR_DATA,
        )
        self.wavelet = wavelet
        self.iterations = iterations

    def recon(
        self, context: ReconContext, branch: str, data: ReconData
    ) -> ReconResult | None:
        del branch
        buffer = data.data
        if buffer is None:
            return None
        grid = buffer.grid_trajectory()
        if grid is None:
            raise ValueError(
                "a non-Cartesian reconstruction needs the readouts' trajectory"
            )
        # The trajectory of the first average stands for all of them.
        trajectory = grid[0] if "average" in buffer.axes else grid
        image = self._fitted(
            averaged(buffer), trajectory, buffer.image_shape, context.device
        )
        return ReconResult(
            image,
            attributes={"ImageProcessingHistory": ["PULSERVER", "PYTHON", "PICS"]},
        )

    def _fitted(
        self,
        kspace: np.ndarray,
        trajectory: np.ndarray,
        shape: tuple[int, ...],
        device: str | None,
    ) -> np.ndarray:
        """Return the magnitude image of ``(coils, [partitions,] shots, samples)`` k-space."""
        import torch
        from bartorch import apps, priors

        plane = shape[-2:]
        if kspace.ndim == 4:
            kspace = np.fft.fftshift(
                np.fft.ifft(np.fft.ifftshift(kspace, axes=1), axis=1), axes=1
            )
            stack = [
                (kspace[:, part], trajectory[part]) for part in range(kspace.shape[1])
            ]
        else:
            stack = [(kspace, trajectory)]
        images = []
        for samples, points in stack:
            samples = torch.from_numpy(
                np.ascontiguousarray(samples, dtype=np.complex64)
            ).to(device)
            points = torch.from_numpy(np.ascontiguousarray(points)).to(device)
            maps = apps.nlinv_maps(samples, traj=points)
            image = apps.pics(
                samples,
                maps,
                traj=points,
                regularizers=priors.Wavelet((-1, -2), self.wavelet),
                maxiter=self.iterations,
                eigen_step=True,
            )
            image = image.abs().cpu().numpy().reshape(maps.shape[-2:])
            images.append(np.array(center_crop(image, plane)))
        return np.stack(images) if len(images) > 1 else images[0]


PLUGIN = NufftRecon()
