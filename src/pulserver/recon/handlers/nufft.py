"""Non-Cartesian reconstruction with bartorch's NUFFT, one least-squares image per coil."""

from __future__ import annotations

__all__ = ["PLUGIN", "NufftRecon"]

from typing import Any

import numpy as np

from ...mrd._acquisitions import AcquisitionFlag
from ...mrd._images import coil_combine
from ...mrd._metadata import acquisition_label, max_stored_value
from ..plugin import ReconContext, ReconPlugin, ReconResult
from .cartesian import last_average, loop_position


class NufftRecon(ReconPlugin):
    """Root-sum-of-squares image of each slice, contrast, cardiac phase, set and repetition of a non-Cartesian scan, made as each closes.

    Each coil's image ``x`` minimises ``|A x - y|^2 + lambda |x|^2`` over its
    samples ``y``, with ``A`` bartorch's NUFFT, so no density compensation is
    assumed of the trajectory; ``lambda`` is ``damping`` times the largest
    eigenvalue of ``A^H A``, which :func:`bartorch.optim.maxeigen` estimates.
    The trajectory is the one the proxy's enrichment writes, which
    :meth:`~pulserver.recon.ReconBuffer.grid_trajectory` scales to the image
    grid. A stack of spokes or spirals is Fourier transformed along its
    partitions first and fitted partition by partition. Images close on the
    last average, sum their averages and are scaled as
    :class:`~pulserver.recon.handlers.cartesian.CartesianRecon` makes them.
    bartorch is imported when the first image is made, which the ``coils``
    extra installs.

    Parameters
    ----------
    iterations
        Conjugate-gradient steps of each fit, from zero.
    damping
        Tikhonov weight as a fraction of the largest eigenvalue of ``A^H A``.
    """

    def __init__(self, iterations: int = 30, damping: float = 0.05) -> None:
        super().__init__(
            branches={AcquisitionFlag.LAST_IN_SLICE: "imaging"},
            reject_flags=AcquisitionFlag.IS_NOISE_MEASUREMENT
            | AcquisitionFlag.IS_PHASECORR_DATA,
        )
        self.iterations = iterations
        self.damping = damping

    def startup(self, context: ReconContext) -> None:
        super().startup(context)
        self.closing: Any = None

    def receive(self, acquisition: Any, context: ReconContext) -> Any:
        self.closing = acquisition
        return super().receive(acquisition, context)

    def recon(self, branch: str, context: ReconContext) -> ReconResult | None:
        del branch
        if self.closing is None:
            return None
        index = int(acquisition_label(self.closing, "encoding_space_ref", 0) or 0)
        buffer = self.buffers[index]
        if not last_average(buffer, self.closing):
            return None
        grid = buffer.grid_trajectory()
        if grid is None:
            raise ValueError(
                "a non-Cartesian reconstruction needs the readouts' trajectory"
            )
        kspace = loop_position(buffer, self.closing)
        trajectory = grid[_loop_index(buffer, self.closing)]
        image = self._fitted(kspace, trajectory, buffer.image_shape, context.device)
        peak = float(image.max(initial=0.0))
        if peak > 0.0:
            image *= max_stored_value(context.header) / peak
        return ReconResult(
            np.around(image).astype(np.int16),
            attributes={
                "ImageProcessingHistory": ["PULSERVER", "PYTHON", "NUFFT"],
                "WindowCenter": str((max_stored_value(context.header) + 1) // 2),
                "WindowWidth": str(max_stored_value(context.header) + 1),
            },
        )

    def _fitted(
        self,
        kspace: np.ndarray,
        trajectory: np.ndarray,
        shape: tuple[int, ...],
        device: str | None,
    ) -> np.ndarray:
        """Return the root-sum-of-squares image of ``(coils, [partitions,] shots, samples)`` k-space."""
        import torch
        from bartorch import optim
        from bartorch.linop import NUFFT

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
            operator = NUFFT(
                torch.from_numpy(np.ascontiguousarray(points)).to(device),
                image_shape=(samples.shape[0], *plane),
            )
            coils = optim.cg(
                torch.from_numpy(np.ascontiguousarray(samples, dtype=np.complex64)).to(
                    device
                ),
                operator,
                self.damping * optim.maxeigen(operator),
                maxiter=self.iterations,
            )
            images.append(coil_combine(coils.cpu().numpy(), coil_axis=0))
        return np.stack(images) if len(images) > 1 else images[0]


PLUGIN = NufftRecon()


def _loop_index(buffer: Any, acquisition: Any) -> tuple[Any, ...]:
    """Index the placement axes of a grid trajectory at the loop position of ``acquisition``; the first average."""
    index: list[Any] = []
    for name in buffer.axes[1:-1]:
        if name in ("partition", "phase_encode"):
            index.append(slice(None))
        elif name == "average":
            index.append(0)
        else:
            index.append(int(acquisition_label(acquisition, name, 0) or 0))
    return tuple(index)
