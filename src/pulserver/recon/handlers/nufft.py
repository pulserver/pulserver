"""Non-Cartesian parallel-imaging compressed-sensing reconstruction with bartorch, as ``bart nlinv -t`` and ``bart pics -t``."""

from __future__ import annotations

__all__ = ["PLUGIN", "NufftRecon"]

from typing import Any

import numpy as np

from ...mrd._acquisitions import AcquisitionFlag
from ...mrd._images import center_crop
from .._buffers import ReconBuffer, ReconData
from .._calibration import _id, digest
from .._correct import Corrections
from ..gadgets import CompressChannels, Prewhiten
from ..plugin import ReconContext, ReconPlugin, ReconResult
from .pics import (
    CALIBRATION_LINES,
    VIRTUAL_COILS,
    averaged,
    compressed,
    nlinv_estimate,
    wavelet_axes,
)

#: Leading readouts whose channel covariance gives the basis the receive
#: channels are compressed by as they arrive
#: (:class:`~pulserver.recon.CompressChannels`); each crosses the k-space
#: centre.
TRAIN_READOUTS = 64

#: The step of the proximal-gradient iteration before it is divided by the
#: largest eigenvalue of the normal operator, as ``bart pics -e`` takes it.
FISTA_STEP = 0.95


class NufftRecon(ReconPlugin):
    """Image of each slice, contrast, cardiac phase, set and repetition of a non-Cartesian scan, a ``pics`` solve made as each closes.

    Each readout is whitened with the noise measurement of the stream
    (:class:`~pulserver.recon.Prewhiten`); noise readouts are consumed by the
    whitening and phase-correction readouts are rejected. Each readout is
    then compressed to ``virtual_coils`` virtual channels as it arrives, by
    the basis the first ``TRAIN_READOUTS`` readouts of the stream give
    (:class:`~pulserver.recon.CompressChannels`). The coil sensitivities of a
    slice and set are fitted by nonlinear inversion to the samples
    near the k-space centre of its first unit, on a grid of those samples
    (:func:`~pulserver.recon.handlers.pics.nlinv_estimate`,
    ``bart nlinv -m 1 --phase-pole 0 -t``), whatever calibration the stream or
    the exam holds: :func:`~pulserver.recon.coil_maps` estimates from
    Cartesian calibration k-space only. Its later units, other repetitions,
    contrasts and cardiac phases, reuse them.

    The image minimises ``|D (F S x - y)|^2 + lambda |W x|_1`` over ``x``
    (``bart pics -t -p D -R W``), ``F`` being bartorch's NUFFT, ``S`` the coil
    sensitivities, ``W`` the wavelet transform over the spatial axes the
    trajectory encodes of at least ``WAVELET_MIN_POINTS`` points, and ``D`` the
    square root of the trajectory's Pipe-Menon density compensation
    (:func:`bartorch.estimate_density`), which preconditions the solve (Baron
    et al., Magn Reson Med 2018). The step is ``FISTA_STEP`` over the largest
    eigenvalue of the normal operator (``bart pics -e``). The density and the
    eigenvalue of a trajectory are computed for the first unit that has them
    and reused by the rest of the stream, every slice included: the maps have
    unit root sum of squares, so the eigenvalue is that of the trajectory's
    weighted transform
    (:meth:`~pulserver.recon.ReconContext.memoized`). An image is its
    averages summed, and its values are those of the solve but for the
    corrections below. A unit of more receive channels than ``virtual_coils``,
    which the stream ended before compressing, is compressed to that many
    virtual channels first (:func:`~pulserver.recon.handlers.pics.compressed`).

    The sensitivities are kept as their k-space kernels of
    ``CALIBRATION_LINES`` points an axis, the region they were fitted on
    (:func:`coil_kernels`), and the encoding applies them as kernels, but for
    a single channel, whose map is applied as it is. The
    samples, the kernels, the density and the vectors of the solve stay on the
    host, where the maps and the density are estimated. The encoding is built
    on ``context.device``, the trajectory with it, and brings the rest over a
    coil at a time as it applies.

    The trajectory is the one the proxy's enrichment writes, which
    :meth:`~pulserver.recon.ReconBuffer.grid_trajectory` scales to the image
    grid. The readouts of a unit are its shots, whichever segment (a PROPELLER
    blade) and line they carry, and one solve takes them all. A shot at which
    no readout was placed is not a sample, nor is a sample past the image
    matrix, ``|k| > N / 2`` along an axis of ``N`` points, as the corners of a
    PROPELLER blade are: the solve is on the matrix's own grid. A stack of
    blades, spokes or spirals is one three-dimensional solve: its partitions
    are the kz encodes of the stack, its maps are fitted once to the
    low-resolution centre of its k-space in three dimensions, and the encoding
    decouples the partitions by a Fourier transform along them where they
    share their in-plane trajectory. A trajectory that encodes kz, with no
    partitions, is one three-dimensional solve too.

    A unit closes as that of
    :class:`~pulserver.recon.handlers.pics.PicsRecon` does, and also once the
    last readout of each of its segments has arrived. bartorch is imported when
    the first image is made.

    The image is corrected as that of
    :class:`~pulserver.recon.handlers.pics.PicsRecon` is
    (:class:`~pulserver.recon.Corrections`): the complex image of each solve
    is Fermi filtered over its axes before its magnitude is taken, a stack's
    partitions among them.

    Parameters
    ----------
    wavelet : float, default=0.005
        ``lambda``, relative to the data scaling ``pics`` estimates.
    iterations : int, default=30
        Iterations of the solve.
    virtual_coils : int | None, default=VIRTUAL_COILS
        Receive channels beyond this many are compressed to it; ``None``
        keeps every channel.
    kspace_filter, gradient_unwarp, intensity_correction : bool, default=True
        As :class:`~pulserver.recon.handlers.pics.PicsRecon`.
    """

    #: The corrections of the stream, made when it starts; ``None`` before.
    corrections: Corrections | None = None

    #: The context and location of the unit being solved; ``None`` outside one.
    _solving: tuple[ReconContext, tuple] | None = None

    def __init__(
        self,
        wavelet: float = 0.005,
        iterations: int = 30,
        virtual_coils: int | None = VIRTUAL_COILS,
        *,
        kspace_filter: bool = True,
        gradient_unwarp: bool = True,
        intensity_correction: bool = True,
    ) -> None:
        compression = (
            ()
            if virtual_coils is None
            else (CompressChannels(virtual_coils, train_readouts=TRAIN_READOUTS),)
        )
        super().__init__(
            gadgets=[Prewhiten(), *compression],
            triggers={
                "imaging": AcquisitionFlag.LAST_IN_SLICE
                | AcquisitionFlag.LAST_IN_SEGMENT
            },
            axes=("average", "segment"),
            reject_flags=AcquisitionFlag.IS_PHASECORR_DATA,
        )
        self.wavelet = wavelet
        self.iterations = iterations
        self.virtual_coils = virtual_coils
        self.kspace_filter = kspace_filter
        self.gradient_unwarp = gradient_unwarp
        self.intensity_correction = intensity_correction

    def startup(self, context: ReconContext) -> None:
        super().startup(context)
        self.corrections = Corrections(
            context.header,
            kspace_filter=self.kspace_filter,
            gradient_unwarp=self.gradient_unwarp,
            intensity_correction=self.intensity_correction,
        )

    def recon(
        self, context: ReconContext, branch: str, data: ReconData
    ) -> ReconResult | None:
        del branch
        if data.data is None:
            return None
        data = compressed(context, data, self.virtual_coils)
        buffer = data.data
        grid = buffer.grid_trajectory()
        if grid is None:
            raise ValueError(
                "a non-Cartesian reconstruction needs the readouts' trajectory"
            )
        kspace, trajectory, played = _by_shots(buffer, grid)
        self._solving = (
            context,
            (
                int(data.counters.get("slice", 0)),
                int(data.counters.get("set", 0)),
                _id(context.noise),
                _id(context.coil_compression),
            ),
        )
        try:
            image = self._fitted(
                kspace, trajectory, played, buffer.image_shape, context.device
            )
        finally:
            self._solving = None
        history = ["PULSERVER", "PYTHON", "PICS"]
        corrections = self.corrections
        if corrections is not None:
            image = corrections.unwarped(image, context, data)
            history += corrections.applied()
        result = ReconResult(image, attributes={"ImageProcessingHistory": history})
        if corrections is not None:
            result = corrections.intensity_corrected(result, data)
        return result

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
        samples. The image is the last two axes of ``shape``; a volume of one
        plane per partition for a stack, or the last three axes of ``shape``
        where the trajectory encodes kz and there are no partitions.

        A stack is one three-dimensional solve: partition ``p`` of ``n`` is
        sampled at ``kz = p - n // 2``, the centred convention of the image
        grid, so its maps are fitted to the low-resolution centre of the
        stack's k-space in three dimensions, and the encoding Fourier
        transforms along the partitions wherever every partition shares its
        in-plane trajectory (:class:`bartorch.linop.NoncartesianSense`).

        Raises
        ------
        ValueError
            If the trajectory encodes kz and ``shape`` has two axes.
        """
        stack = 1
        if played.ndim == 2:
            partitions = played.shape[0]
            if (played == played[:1]).all():
                stack = partitions
            extent = (partitions, *(int(n) for n in shape[-2:]))
            points = np.array(trajectory, dtype=np.float32)
            points[..., 2] = (np.arange(partitions) - partitions // 2)[:, None, None]
            kspace = kspace.reshape(kspace.shape[0], -1, kspace.shape[-1])
            points = points.reshape(-1, *points.shape[-2:])
            played = played.reshape(-1)
        elif trajectory[..., 2].any():
            if len(shape) < 3:
                raise ValueError(
                    "a trajectory that encodes kz needs a three-dimensional image matrix"
                )
            extent = tuple(int(n) for n in shape[-3:])
            points = trajectory
        else:
            extent = tuple(int(n) for n in shape[-2:])
            points = trajectory
        if not played.any():
            return np.zeros(extent, dtype=np.float32)
        if not played.all():
            kspace, points = kspace[:, played], points[played]
        image = self._solved(kspace, points, device, extent, partitions=stack)
        if image.shape != tuple(extent):
            raise ValueError(
                f"the solve's grid {image.shape} is not the image matrix {extent}"
            )
        return np.array(center_crop(image, extent))

    def _calibration(
        self, parts: tuple, compute: Any, what: str, *, per_location: bool = True
    ) -> Any:
        """Return the value of ``compute``, computed once per stream for ``parts``.

        A value ``per_location`` is computed once for each slice and set of
        the stream, with its prewhitening and compression; any other once for
        each value of ``parts``. Outside a unit it is computed every time.
        """
        if self._solving is None:
            return compute()
        context, location = self._solving
        where = location if per_location else ()
        named = f"{what} of slice {location[0]}, set {location[1]}"
        return context.memoized(
            ("nufft", *parts, *where), compute, what=named if per_location else what
        )

    def _solved(
        self,
        samples: np.ndarray,
        points: np.ndarray,
        device: str | None,
        shape: tuple[int, ...],
        partitions: int = 1,
    ) -> np.ndarray:
        """Return the magnitude image of ``(coils, shots, samples)`` k-space and its ``(shots, samples, 3)`` trajectory.

        The image is the ``(y, x)`` plane or the ``(z, y, x)`` volume ``shape``,
        from the samples the matrix holds (:func:`_within`), Fermi filtered as a
        complex image where the stream's :attr:`corrections` say so. The shots
        of a stack of ``partitions`` with one in-plane trajectory run
        partition after partition.
        """
        import bartorch
        import torch
        from bartorch import linop, optim, priors

        samples, points = _within(samples, points, shape, partitions)
        volume = tuple(shape) if len(shape) == 3 else None
        samples = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.complex64))
        points = torch.from_numpy(np.ascontiguousarray(points))
        axes = 2 if volume is None else 3
        trajectory = digest(points.numpy())
        kernels, grid = self._calibration(
            ("kernels",),
            lambda: coil_kernels(nlinv_estimate(samples, points, volume), axes),
            "coil sensitivity kernels",
        )
        encoded = wavelet_axes(grid, tuple(range(-1, -axes - 1, -1)))
        weights = self._calibration(
            ("density", trajectory, grid),
            lambda: _density(points, grid, partitions).sqrt(),
            "density compensation",
            per_location=False,
        )
        held_as_kernels = kernels.shape[0] > 1
        encoding = linop.NoncartesianSense(
            kernels
            if held_as_kernels
            else bartorch.kernels_to_maps(kernels, grid).reshape(1, *grid),
            grid,
            kernels=held_as_kernels,
            traj=points if device is None else points.to(device),
            weights=weights,
            coil_batch=1 if held_as_kernels else 0,
        )
        measured = samples * weights
        scaling = optim.data_scaling(measured[..., None], A=encoding) or 1.0
        eigenvalue = self._calibration(
            ("eigenvalue", trajectory, grid),
            lambda: optim.maxeigen(encoding),
            "largest eigenvalue of the normal operator",
            per_location=False,
        )
        solve = optim.FISTA(
            [priors.Wavelet(encoded, self.wavelet)],
            maxiter=self.iterations,
            step=FISTA_STEP / eigenvalue,
        )
        image = scaling * solve(measured * (1.0 / scaling), encoding)
        image = image.cpu().numpy().reshape(grid)
        if self.corrections is not None:
            image = self.corrections.filtered(image)
        return np.abs(image)


def _density(points: Any, grid: tuple[int, ...], partitions: int) -> Any:
    """Return the Pipe-Menon density of ``(shots, samples, 3)`` points on ``grid`` (:func:`bartorch.estimate_density`), shaped as the shots.

    The shots of a stack of ``partitions``, partition after partition, lie on
    whole kz, so each partition's density is that of its own in-plane
    trajectory. Where every partition plays the same in-plane trajectory the
    density is computed once and is the same at every partition, so the
    encoding decouples the partitions; a partition whose trajectory is shifted
    or rotated has the density of its own.
    """
    import bartorch
    import torch

    if partitions > 1:
        planes = points[..., :2].reshape(partitions, -1, *points.shape[1:-1], 2)
        if bool((planes == planes[:1]).all()):
            first = bartorch.estimate_density(planes[0].reshape(-1, 2), grid[-2:])
            density = first.reshape(1, -1).expand(partitions, -1)
        else:
            density = torch.stack(
                [
                    bartorch.estimate_density(plane.reshape(-1, 2), grid[-2:])
                    for plane in planes
                ]
            )
        return density.reshape(points.shape[:-1]).contiguous()
    axes = len(grid)
    density = bartorch.estimate_density(points[..., :axes].reshape(-1, axes), grid)
    return density.reshape(points.shape[:-1])


def coil_kernels(maps: Any, axes: int) -> tuple[Any, tuple[int, ...]]:
    """Return coil maps ``(coils, *grid)`` as their k-space kernels of ``CALIBRATION_LINES`` points an axis, with the grid.

    :func:`~pulserver.recon.handlers.pics.nlinv_estimate` fits the maps on
    that many points an axis around the k-space centre and zero-fills them to
    the grid, so the kernels (:func:`bartorch.maps_to_kernels`) hold them
    whole, and a SENSE operator built on them
    (:class:`bartorch.linop.NoncartesianSense` with ``kernels=True``) never
    holds a full-resolution coil image beyond the one it applies.
    """
    import bartorch

    grid = tuple(int(n) for n in maps.shape[-axes:])
    size = tuple(min(CALIBRATION_LINES, n) for n in grid)
    coils = maps.shape[0]
    kernels = bartorch.maps_to_kernels(maps.reshape(coils, *grid), size)
    return kernels.reshape(coils, *kernels.shape[-axes:]), grid


def _within(
    samples: np.ndarray,
    points: np.ndarray,
    extent: tuple[int, ...],
    partitions: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the samples of ``(coils, shots, samples)`` k-space inside the ellipse the image matrix holds, with their ``(shots, samples, 3)`` points; as one shot where any lies outside.

    Along an axis of ``N`` points the matrix holds ``|k| <= N / 2``. A sample
    beyond it, as the corners of a PROPELLER blade are, would grow the grid
    bartorch sizes to the farthest sample, and the image cropped from that
    grid to the matrix would be magnified by the ratio of the two.

    The shots of a stack of ``partitions``, partition after partition, each at
    its whole kz, are a cylinder: only their in-plane position is tested, and
    where every partition keeps the same number of samples they are returned
    as one shot per partition, so the encoding still decouples the partitions.
    """
    axes = 2 if partitions > 1 else len(extent)
    half = 0.5 * np.asarray(extent[::-1][:axes], dtype=np.float64)
    inside = np.square(points[..., :axes] / half).sum(axis=-1) <= 1.0 + 1e-6
    if inside.all():
        return samples, points
    kept = inside.reshape(partitions, -1).sum(axis=-1)
    if partitions > 1 and np.all(kept == kept[0]):
        return (
            samples[:, inside].reshape(samples.shape[0], partitions, -1),
            points[inside].reshape(partitions, -1, points.shape[-1]),
        )
    return samples[:, inside][:, None], points[inside][None]


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
    not a stack and is dropped: the header lays one out along the ``Matrix`` z
    of a trajectory that encodes kz itself and writes no ``PAR``.
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
