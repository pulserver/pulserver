"""Cartesian parallel-imaging compressed-sensing reconstruction with bartorch, as ``bart nlinv``, ``bart pics`` and ``bart homodyne``."""

from __future__ import annotations

__all__ = ["PLUGIN", "PicsRecon"]

import math
from typing import Any

import numpy as np

from ...mrd._acquisitions import AcquisitionFlag
from ...mrd._images import center_crop
from .._buffers import ReconBuffer, ReconData
from .._calibration import CoilCompression, coil_maps
from .._correct import gradient_unwarped, states_gradient_coefficients
from ..gadgets import AsymmetricEcho, Prewhiten, RemoveReadoutOversampling
from ..plugin import ReconContext, ReconPlugin, ReconResult

#: Fewest points along an axis that BART's wavelet, ``dau2`` as
#: :class:`bartorch.priors.Wavelet` takes it by default, decomposes: the
#: coarsest scale of a shorter axis is never reached, and the process dies.
WAVELET_MIN_POINTS = 4

#: Lines of every encoded axis, around the k-space centre, the coil maps are
#: fitted to, as :func:`bartorch.apps.nlinv_maps` takes them.
CALIBRATION_LINES = 24

#: Virtual channels the receive channels of a unit are compressed to before
#: its maps and its solve (:class:`~pulserver.recon.CoilCompression`).
VIRTUAL_COILS = 6


class PicsRecon(ReconPlugin):
    """Image of each slice, contrast, cardiac phase, set and repetition of a Cartesian scan, a ``pics`` solve made as each closes.

    Readouts are placed by their encoding counters, so phase encodes,
    partitions and averages may arrive in any order. An image is made once
    ``LAST_IN_SLICE`` has arrived for each average of a slice, contrast,
    cardiac phase, set and repetition, its averages summed, and at the end of a
    measurement for any that never closed.

    Each readout is whitened with the noise measurement of the stream
    (:class:`~pulserver.recon.Prewhiten`), completed to a full echo
    (:class:`~pulserver.recon.AsymmetricEcho`), has its readout oversampling
    removed (:class:`~pulserver.recon.RemoveReadoutOversampling`) and is placed
    by its echo along the readout and by the offset of its lines from the
    k-space centre, so the k-space is at the reconstruction matrix when its
    slice closes. Noise readouts are consumed by the whitening;
    phase-correction readouts are rejected.

    A unit of more receive channels than ``virtual_coils`` is compressed to
    that many virtual channels (:func:`compressed`) before its maps are
    estimated and its image solved.

    The coil sensitivities are those of :func:`~pulserver.recon.coil_maps`
    with :func:`nlinv_estimate` as the estimate: from the unit's calibration
    readouts, else the maps stored for its slice, else the exam's. A unit for
    which none of these has maps takes them from the low-resolution centre of
    its own k-space (``bart nlinv -m 1 --phase-pole 0``), with unit
    sensitivity for one coil. A unit that holds calibration readouts and no
    imaging readouts makes no image; its maps are stored for the later units
    of its slice. The image minimises
    ``|P F S x - y|^2 + lambda |W x|_1`` over the maps (``bart pics -R W``),
    ``W`` the wavelet transform over the encoded axes of at least
    ``WAVELET_MIN_POINTS`` points. An encoded axis is
    partial Fourier where the lines missing at one end outnumber the widest gap
    between sampled lines, and
    :func:`bartorch.apps.partial_fourier` completes the image along it
    (``bart homodyne -I -C``). bartorch is imported when the first unit is
    reconstructed.

    The k-space, the maps and the vectors of the solve stay on the host. The
    encoding is built on ``context.device`` and brings them over a coil at a
    time as it applies (:func:`bartorch.linop.CartesianSense`), so between
    applications the device holds the operator alone.

    A unit of a non-Cartesian encoding space is a wave-CAIPI volume, which the
    proxy makes of readouts whose k-space moves along the phase encodes:
    reconstructed by :meth:`wave_image`, with the readout oversampling the
    wave spreads the image into.

    Parameters
    ----------
    wavelet : float, default=0.005
        ``lambda``, relative to the data scaling ``pics`` estimates.
    iterations : int, default=30
        Iterations of the solve.
    virtual_coils : int | None, default=VIRTUAL_COILS
        Receive channels beyond this many are compressed to it; ``None``
        keeps every channel.
    """

    def __init__(
        self,
        wavelet: float = 0.005,
        iterations: int = 30,
        virtual_coils: int | None = VIRTUAL_COILS,
    ) -> None:
        super().__init__(
            triggers={"imaging": AcquisitionFlag.LAST_IN_SLICE},
            axes=("average",),
            reject_flags=AcquisitionFlag.IS_PHASECORR_DATA,
        )
        self.gadgets = (Prewhiten(), AsymmetricEcho(), RemoveReadoutOversampling())
        self.wavelet = wavelet
        self.iterations = iterations
        self.virtual_coils = virtual_coils

    def recon(
        self, context: ReconContext, branch: str, data: ReconData
    ) -> ReconResult | None:
        del branch
        data = compressed(context, data, self.virtual_coils)
        if data.data is None and data.ref is not None:
            coil_maps(context, data, estimate=nlinv_estimate)
        buffer = data.data
        if buffer is None:
            return None
        if buffer.space.cartesian:
            image = self.image(averaged(buffer), buffer.image_shape, context, data)
        else:
            image = self.wave_image(context, data)
        history = ["PULSERVER", "PYTHON", "PICS"]
        if states_gradient_coefficients(context.header):
            image = gradient_unwarped(image, context, data)
            history.append("GRADUNWARP")
        return ReconResult(image, attributes={"ImageProcessingHistory": history})

    def image(
        self,
        kspace: np.ndarray,
        shape: tuple[int, ...],
        context: ReconContext,
        data: ReconData,
    ) -> np.ndarray:
        """Return the magnitude image of ``(coils, [partitions,] phase encodes, readout)`` k-space, cropped to ``shape``."""
        import torch
        from bartorch import apps

        volume = kspace if kspace.ndim == 4 else kspace[:, None]
        volume = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.complex64))
        maps = coil_maps(context, data, estimate=nlinv_estimate, required=False)
        maps = nlinv_estimate(volume) if maps is None else maps.reshape(volume.shape)
        sampled = volume.abs().sum(dim=0) > 0
        image = pics_image(
            volume, maps, sampled, self.wavelet, self.iterations, context.device
        )
        image = apps.partial_fourier(image, sampled)
        image = image.abs().numpy()
        if kspace.ndim == 3:
            image = image[0]
        return np.array(center_crop(image, shape[-image.ndim :]))

    def wave_image(self, context: ReconContext, data: ReconData) -> np.ndarray:
        """Return the magnitude image ``(partitions, phase encodes, readout)`` of a wave-CAIPI volume.

        The wave-encoded readouts are ``data.data`` and the wave-free
        readouts of the calibration region ``data.ref``, both of one
        non-Cartesian encoding space whose ``LIN`` and ``PAR`` counters are
        their lines on the grid of the reconstruction matrix, the k-space
        centre at ``n // 2``, and whose readouts keep their samples and
        oversampling in acquisition order.

        The wave is the k-space trajectory along the phase encodes, in 1/m as
        the proxy stamps it, of a wave-encoded readout less that of the
        wave-free readout of its line. Its gradient scale and delay are fitted
        to the calibration region (:func:`bartorch.linop.wave_calibrate`), the
        point-spread function made from it with them
        (:func:`bartorch.linop.wave_psf`), and the image minimises
        ``|P F_yz Psi F_x S x - y|^2 + lambda |W x|_1``, ``Psi`` the
        point-spread function (:func:`bartorch.linop.WaveSense`), over the maps
        :func:`nlinv_estimate` estimates from the calibration region, cropped
        to the readout field of view.

        Raises
        ------
        ValueError
            If the unit is not one volume of wave-encoded and wave-free
            readouts, or no line was acquired both with and without the wave.
        """
        import torch
        from bartorch import linop, optim, priors

        buffer, ref = data.data, data.ref
        volume = ("coil", "partition", "phase_encode", "readout")
        if (
            ref is None
            or buffer.trajectory is None
            or {buffer.axes, ref.axes} != {volume}
        ):
            raise ValueError(
                f"pics reconstructs a non-Cartesian encoding space as one wave-CAIPI "
                f"volume {volume} of wave-encoded readouts with their trajectory and "
                f"wave-free calibration readouts; encoding space {buffer.space.index} "
                f"holds {buffer.axes}"
                + ("" if ref is not None else " and no calibration readouts")
            )
        shape = buffer.image_shape
        fov = buffer.space.recon_fov[:2]
        wave, reference = _volume(buffer, shape), _volume(ref, shape)
        k = _wave_trajectory(buffer, ref)
        scale, delay = linop.wave_calibrate(reference, wave, k, fov)
        psf = linop.wave_psf(k, shape, fov, scale=scale, delay=delay, centred=True)

        def tensor(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(array, dtype=np.complex64))

        maps = center_crop(nlinv_estimate(tensor(reference)), shape[-1:])
        encoding = linop.WaveSense(
            maps.contiguous(),
            shape,
            wave.shape[-1],
            tensor(np.abs(wave).sum(axis=0) > 0),
            centred=True,
            psf=tensor(psf.numpy()),
            device=context.device,
        )
        measured = tensor(wave)
        scaling = optim.data_scaling(measured, A=encoding)
        solve = optim.FISTA(
            priors.Wavelet(wavelet_axes(shape, (-1, -2, -3)), self.wavelet),
            maxiter=self.iterations,
        )
        image = scaling * solve(measured * (1.0 / scaling), encoding)
        return image.abs().numpy()


PLUGIN = PicsRecon()


def nlinv_estimate(
    kspace: Any, traj: Any = None, shape: tuple[int, ...] | None = None
) -> Any:
    """Return the coil maps :func:`bartorch.apps.nlinv_maps` fits to ``kspace``, with phase-pole correction.

    ``bart nlinv -m 1 --phase-pole 0`` on a calibration grid of
    ``CALIBRATION_LINES`` points an axis, capped at the axis: the central
    lines of Cartesian k-space, or the samples of a trajectory within half
    that of the centre, the maps of which are on ``shape``. After every
    Gauss-Newton step the phase poles the maps share are moved into the
    image, where they would otherwise drive every coil's sensitivity towards
    zero and leave a dark spot in the image. BART refuses the detection on a
    grid with an axis no longer than a twentieth of its longest, rounded up;
    k-space whose calibration grid has such an axis, a slab of fewer than 3
    partitions, is fitted without it.
    """
    from bartorch import apps

    image = kspace.shape[1:] if traj is None else (shape or ())
    grid = [min(CALIBRATION_LINES, n) for n in image if n > 1]
    detectable = not grid or min(grid) > math.ceil(max(grid) / 20)
    return apps.nlinv_maps(
        kspace,
        traj,
        size=CALIBRATION_LINES,
        radius=CALIBRATION_LINES / 2,
        shape=shape,
        phase_pole=0 if detectable else None,
    )


def compressed(
    context: ReconContext, data: ReconData, virtual_coils: int | None
) -> ReconData:
    """Return ``data`` with its receive channels compressed to ``virtual_coils`` (:class:`~pulserver.recon.CoilCompression`).

    ``data`` is returned as it is when it holds no more channels than that, or
    ``virtual_coils`` is ``None``.
    """
    buffer = data.data if data.data is not None else data.ref
    if virtual_coils is None or buffer is None or buffer.coils <= virtual_coils:
        return data
    return CoilCompression(virtual_coils)(context, data)


def pics_image(
    kspace: Any,
    maps: Any,
    sampled: Any,
    wavelet: float,
    iterations: int,
    device: str | None,
) -> Any:
    """Return the complex image ``(z, y, x)`` of ``bart pics -S -R W`` over host k-space ``(coils, z, y, x)``.

    The solve of :func:`bartorch.apps.pics` on a grid: ``sampled``, the
    ``(z, y, x)`` lines placed, as its pattern, the scaling estimated from the
    fully sampled centre, and FISTA over the wavelet term of weight
    ``wavelet`` along the axes of at least ``WAVELET_MIN_POINTS`` points. A
    volume of one partition is solved as a plane. The k-space, the maps and
    the iteration's vectors stay on the host, and the encoding computes on
    ``device``; ``None`` is the host. The image is on the host.
    """
    import bartorch
    import torch
    from bartorch import linop, optim, priors

    grid = tuple(kspace.shape[2:] if kspace.shape[1] == 1 else kspace.shape[1:])
    pattern = sampled.reshape(grid).to(torch.complex64)
    measured = kspace * sampled
    scaling = optim.data_scaling(
        bartorch.fftmod(measured, axes=(-1, -2, -3), inverse=True)
    )
    scaling = scaling or 1.0
    encoding = linop.CartesianSense(
        maps.reshape(maps.shape[0], *grid), grid, pattern=pattern, device=device
    )
    axes = wavelet_axes(grid, tuple(range(-1, -len(grid) - 1, -1)))
    solve = optim.FISTA(priors.Wavelet(axes, wavelet), maxiter=iterations)
    data = (measured * (1.0 / scaling)).reshape(kspace.shape[0], *grid)
    return (scaling * solve(data, encoding)).reshape(kspace.shape[1:])


def wavelet_axes(shape: tuple[int, ...], axes: tuple[int, ...]) -> tuple[int, ...]:
    """Return the ``axes`` of an image of ``shape`` with at least ``WAVELET_MIN_POINTS`` points."""
    return tuple(axis for axis in axes if shape[axis] >= WAVELET_MIN_POINTS)


def averaged(buffer: ReconBuffer) -> np.ndarray:
    """Return the k-space of ``buffer``, ``(coils, *encoded, readout)``, its averages summed."""
    kspace = buffer.kspace
    if "average" in buffer.axes:
        kspace = kspace.sum(axis=buffer.axes.index("average"))
    return kspace


def _volume(buffer: ReconBuffer, shape: tuple[int, ...]) -> np.ndarray:
    """Return the k-space of a wave-CAIPI buffer on the ``(partitions, lines)`` grid of ``shape``, zero where nothing was placed."""
    kspace = buffer.kspace
    grid = np.zeros((kspace.shape[0], *shape[:2], kspace.shape[-1]), kspace.dtype)
    window = tuple(
        slice(buffer.origin.get(name, 0), buffer.origin.get(name, 0) + extent)
        for name, extent in zip(
            ("partition", "phase_encode"), kspace.shape[1:3], strict=True
        )
    )
    grid[(slice(None), *window)] = kspace
    return grid


def _wave_trajectory(buffer: ReconBuffer, ref: ReconBuffer) -> np.ndarray:
    """Return the wave's k-space ``(readout, 2)`` along the partition and phase encodes, in 1/m.

    A wave-encoded readout's trajectory less that of the wave-free readout of
    the same line; the readouts' trajectories are ``(x, y, z)``.

    Raises
    ------
    ValueError
        If no line was acquired both with and without the wave.
    """
    origin = (ref.origin.get("partition", 0), ref.origin.get("phase_encode", 0))
    calibrated = ref.mask.any(axis=-1)
    for z, y in np.argwhere(buffer.mask.any(axis=-1)):
        rz, ry = z - origin[0], y - origin[1]
        inside = 0 <= rz < calibrated.shape[0] and 0 <= ry < calibrated.shape[1]
        if inside and calibrated[rz, ry]:
            moved = buffer.trajectory[:, z, y] - ref.trajectory[:, rz, ry]
            return np.stack([moved[2], moved[1]], axis=-1)
    raise ValueError(
        "no line was acquired both with and without the wave, so the wave's "
        "trajectory cannot be told from its line's"
    )
