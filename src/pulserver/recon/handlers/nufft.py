"""Non-Cartesian parallel-imaging compressed-sensing reconstruction with bartorch, as ``bart nlinv -t`` and ``bart pics -t``."""

from __future__ import annotations

__all__ = ["PLUGIN", "NufftRecon"]

from typing import Any

import numpy as np

from ...mrd._acquisitions import AcquisitionFlag
from ...mrd._images import center_crop
from .._buffers import ReconBuffer, ReconData
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
    ``F`` bartorch's NUFFT and ``W`` the wavelet transform over the spatial
    axes the trajectory encodes, so no density compensation is assumed of the
    trajectory; the step is the reciprocal of the largest eigenvalue of the
    normal operator (``bart pics -e``). The trajectory is the one the proxy's
    enrichment writes, which
    :meth:`~pulserver.recon.ReconBuffer.grid_trajectory` scales to the image
    grid. The readouts of a unit are its shots, whichever segment (a PROPELLER
    blade) and line they carry, and one solve takes them all; a shot at which
    no readout was placed is not a sample. A stack of blades, spokes or spirals
    is Fourier transformed along its partitions first and fitted partition by
    partition. A trajectory that encodes kz, with no partitions, is one
    three-dimensional solve, its maps fitted as ``bart nlinv -m 1 -t`` fits
    them over three spatial axes. A unit closes as that of
    :class:`~pulserver.recon.handlers.cartesian.CartesianRecon` does, and also
    once the last readout of each of its segments has arrived; an image is its
    averages summed, and its values are those of the solve, unscaled. bartorch
    is imported when the first image is made, which the ``coils`` extra
    installs.

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
            triggers={
                "imaging": AcquisitionFlag.LAST_IN_SLICE
                | AcquisitionFlag.LAST_IN_SEGMENT
            },
            axes=("average", "segment"),
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
        kspace, trajectory, played = _by_shots(buffer, grid)
        image = self._fitted(
            kspace, trajectory, played, buffer.image_shape, context.device
        )
        return ReconResult(
            image,
            attributes={"ImageProcessingHistory": ["PULSERVER", "PYTHON", "PICS"]},
        )

    def _fitted(
        self,
        kspace: np.ndarray,
        trajectory: np.ndarray,
        played: np.ndarray,
        shape: tuple[int, ...],
        device: str | None,
    ) -> np.ndarray:
        """Return the magnitude image of ``(coils, [partitions,] shots, samples)`` k-space.

        ``trajectory`` is ``([partitions,] shots, samples, 3)`` and ``played``
        ``([partitions,] shots)``; only the shots flagged in ``played`` are
        samples. The image is the last two axes of ``shape``, one per
        partition, or its last three where the trajectory encodes kz and there
        are no partitions: one volume.

        Raises
        ------
        ValueError
            If the trajectory encodes kz and ``shape`` has two axes.
        """
        volume = None
        if played.ndim == 1 and trajectory[..., 2].any():
            if len(shape) < 3:
                raise ValueError(
                    "a trajectory that encodes kz needs a three-dimensional image matrix"
                )
            volume = tuple(int(n) for n in shape[-3:])
        extent = shape[-2:] if volume is None else volume
        if played.ndim == 2:
            kspace = np.fft.fftshift(
                np.fft.ifft(np.fft.ifftshift(kspace, axes=1), axis=1), axes=1
            )
            stack = [
                (kspace[:, part], trajectory[part], played[part])
                for part in range(played.shape[0])
            ]
        else:
            stack = [(kspace, trajectory, played)]
        images = []
        for samples, points, shots in stack:
            if not shots.any():
                images.append(np.zeros(extent, dtype=np.float32))
                continue
            if not shots.all():
                samples, points = samples[:, shots], points[shots]
            image = self._solved(samples, points, device, volume)
            images.append(np.array(center_crop(image, extent)))
        return np.stack(images) if len(images) > 1 else images[0]

    def _solved(
        self,
        samples: np.ndarray,
        points: np.ndarray,
        device: str | None,
        volume: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Return the magnitude image of ``(coils, shots, samples)`` k-space and its ``(shots, samples, 3)`` trajectory.

        The image is a plane on the grid the trajectory spans, or the
        ``(z, y, x)`` matrix ``volume`` when it is given.
        """
        import torch
        from bartorch import apps, priors

        samples = torch.from_numpy(
            np.ascontiguousarray(samples, dtype=np.complex64)
        ).to(device)
        points = torch.from_numpy(np.ascontiguousarray(points)).to(device)
        axes = 2 if volume is None else 3
        maps = (
            apps.nlinv_maps(samples, traj=points)
            if volume is None
            else _volume_maps(samples, points, volume)
        )
        image = apps.pics(
            samples,
            maps,
            traj=points,
            regularizers=priors.Wavelet(tuple(range(-1, -axes - 1, -1)), self.wavelet),
            maxiter=self.iterations,
            eigen_step=True,
        )
        return image.abs().cpu().numpy().reshape(maps.shape[-axes:])


def _volume_maps(
    samples: Any, points: Any, shape: tuple[int, ...], radius: float = 12.0
) -> Any:
    """Return the coil maps ``(coils, z, y, x)`` on the grid ``shape`` of ``(coils, shots, samples)`` k-space along a three-dimensional trajectory.

    As :func:`bartorch.apps.nlinv_maps` fits them for a plane: ``bart nlinv
    -m 1`` over the samples within ``radius`` grid units of the k-space
    centre, normalised to unit root sum of squares over the coils, and unit
    for one coil.
    """
    import torch
    from bartorch import tools

    coils = samples.shape[0]
    if coils == 1:
        return torch.ones((1, *shape), dtype=samples.dtype, device=samples.device)
    centre = points.square().sum(dim=-1).sqrt() <= radius
    _, maps = tools.nlinv(
        (samples * centre)[..., None],
        traj=points,
        maps=1,
        return_sensitivities=True,
        x=tuple(reversed(shape)),
    )
    maps = maps.reshape(coils, *shape)
    return maps / maps.abs().square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-12)


def _by_shots(
    buffer: ReconBuffer, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the readouts of a unit with its segments and phase encodes folded into one shots axis.

    The k-space is ``(coils, [partitions,] shots, samples)``, its averages
    summed and its samples those the readouts placed; the trajectory,
    ``([partitions,] shots, samples, 3)``, is that of the first average; and
    the boolean ``([partitions,] shots)`` is true for each shot a readout was
    placed at. Shots run segment by segment, in the order of the buffer's axes.
    A partition axis along which readouts were placed at one position only is
    not a stack and is dropped: the header of a multi-slice 2D scan counts its
    slices as partitions.
    """
    axes = list(buffer.axes[1:-1])
    kspace, mask = averaged(buffer), buffer.mask
    if "average" in axes:
        average = axes.index("average")
        grid, mask = grid.take(0, axis=average), mask.any(axis=average)
        axes.pop(average)
    first, last = buffer.readout
    span = slice(first, last + 1)
    kspace, grid, mask = kspace[..., span], grid[..., span, :], mask[..., span]
    played = mask.any(axis=-1)
    lead: tuple[int, ...] = ()
    if "partition" in axes:
        partition = axes.index("partition")
        rest = tuple(axis for axis in range(played.ndim) if axis != partition)
        (planes,) = np.nonzero(played.any(axis=rest))
        if planes.size == 1:
            kspace = kspace.take(planes[0], axis=1 + partition)
            grid = grid.take(planes[0], axis=partition)
            played = played.take(planes[0], axis=partition)
        else:
            kspace = np.moveaxis(kspace, 1 + partition, 1)
            grid = np.moveaxis(grid, partition, 0)
            played = np.moveaxis(played, partition, 0)
            lead = (played.shape[0],)
    coils, samples = kspace.shape[0], kspace.shape[-1]
    return (
        kspace.reshape(coils, *lead, -1, samples),
        grid.reshape(*lead, -1, samples, 3),
        played.reshape(*lead, -1),
    )


PLUGIN = NufftRecon()
