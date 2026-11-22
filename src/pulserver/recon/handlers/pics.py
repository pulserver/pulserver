"""Cartesian parallel-imaging compressed-sensing reconstruction with bartorch, as ``bart nlinv`` and ``bart pics``, partial Fourier solved for a real image times its phase."""

from __future__ import annotations

__all__ = ["PLUGIN", "PicsRecon"]

import logging
import math
from typing import Any

import numpy as np

from ...mrd._acquisitions import AcquisitionFlag
from ...mrd._images import center_crop
from .._buffers import ReconBuffer, ReconData
from .._calibration import CoilCompression, MissingCalibration, coil_maps
from .._correct import RECEIVE_FIELDS, Corrections
from ..gadgets import (
    AsymmetricEcho,
    CompressChannels,
    Prewhiten,
    RemoveReadoutOversampling,
)
from ..plugin import ReconContext, ReconPlugin, ReconResult

#: Fewest points along an axis that BART's wavelet, ``dau2`` as
#: :class:`bartorch.priors.Wavelet` takes it by default, decomposes: the
#: coarsest scale of a shorter axis is never reached, and the process dies.
WAVELET_MIN_POINTS = 4

#: Lines of every encoded axis, around the k-space centre, the coil maps are
#: fitted to, as :func:`bartorch.apps.nlinv_maps` takes them. Maps fitted to
#: 24 lines left a ghost at half the field of view of a scan undersampled by
#: 2 along its partitions, about 5 % of the object; at 48 it is as faint as
#: GRAPPA leaves it.
CALIBRATION_LINES = 48

#: Virtual channels the receive channels are compressed to before the maps
#: and the solve (:class:`~pulserver.recon.CompressChannels`).
VIRTUAL_COILS = 6

#: Least Tikhonov weight of a conjugate-gradient SENSE solve, on data scaled
#: as ``pics`` scales them. Lines a unit leaves out, such as the ends of a
#: partial-Fourier phase encode, make the normal operator singular, and in
#: single precision the iterate grows without bound along them.
CG_MIN_L2 = 1e-6

#: Least Tikhonov weight of the real-image conjugate-gradient solve, on data
#: scaled as ``pics`` scales them. Two partial-Fourier axes leave out k-space
#: whose conjugate is not sampled either, and without a weight the solve
#: grows it without bound.
REAL_SOLVE_MIN_L2 = 1e-4

#: Standard deviation, in lines, of the in-plane Gaussian k-space window that
#: smooths that phase. At 7T the off-resonance phase near the sinuses wraps
#: within a few voxels by a BOLD echo time; a wavelet prior on that complex
#: image turns each wrap into a ring.
PHASE_SMOOTHING_LINES = 8.0

#: ``series_index`` of the magnitude, real and imaginary images.
MAGNITUDE_SERIES, REAL_SERIES, IMAGINARY_SERIES = 0, 1, 2

_log = logging.getLogger("pulserver.recon")


class PicsRecon(ReconPlugin):
    """Image of each slice, contrast, cardiac phase, set and repetition of a Cartesian scan, a ``pics`` solve made as each closes.

    Readouts are placed by their encoding counters, so phase encodes,
    partitions and averages may arrive in any order. An image is made once
    ``LAST_IN_SLICE`` has arrived for each average of a slice, contrast,
    cardiac phase, set and repetition, its averages summed, and at the end of a
    measurement for any that never closed.

    Each readout is whitened with the noise measurement of the stream
    (:class:`~pulserver.recon.Prewhiten`), compressed to ``virtual_coils``
    virtual channels by a basis from the calibration readouts that lead the
    scan (:class:`~pulserver.recon.CompressChannels`), completed to a full echo
    (:class:`~pulserver.recon.AsymmetricEcho`), has its readout oversampling
    removed (:class:`~pulserver.recon.RemoveReadoutOversampling`) and is placed
    by its echo along the readout and by the offset of its lines from the
    k-space centre, so the k-space is at the reconstruction matrix when its
    slice closes. Noise readouts are consumed by the whitening;
    phase-correction readouts are rejected.

    The buffers of a scan of many contrasts, which are open together, so
    hold ``virtual_coils`` channels each. A scan whose calibration readouts
    do not lead is compressed a unit at a time instead, as the unit closes
    and before its maps are estimated and its image solved
    (:func:`compressed`).

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
    ``WAVELET_MIN_POINTS`` points. An encoded axis is partial Fourier where
    the lines missing at one end outnumber the widest gap between sampled
    lines. A unit with such an axis is solved again for a real image times
    the phase of its first solve over the k-space sampled on both sides of
    the centre (:func:`symmetric_phase`), the maps
    carrying the phase (:func:`pics_image` with ``phase``), so the k-space it
    leaves unsampled is the conjugate of what it samples. bartorch is
    imported when the first unit is reconstructed.

    The magnitude image is series ``MAGNITUDE_SERIES``. A scan of more than
    one contrast, such as a multi-echo gradient echo, also sends the real and
    imaginary parts of each image as series ``REAL_SERIES`` and
    ``IMAGINARY_SERIES``; where an axis is partial Fourier their phase is
    that symmetric phase.

    The k-space, the maps and the vectors of the solve stay on the host. The
    encoding is built on ``context.device`` and brings them over a coil at a
    time as it applies (:func:`bartorch.linop.CartesianSense`), so between
    applications the device holds the operator alone.

    A unit of a non-Cartesian encoding space is a wave-CAIPI volume, which the
    proxy makes of readouts whose k-space moves along the phase encodes:
    reconstructed by :meth:`wave_image`, with the readout oversampling the
    wave spreads the image into.

    Each complex image the solve returns, before its magnitude is taken, is
    Fermi filtered in k-space over its encoded axes, then corrected for
    gradient nonlinearity, and its magnitude divided by the receive field
    (:class:`~pulserver.recon.Corrections`).

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
        The corrections of :class:`~pulserver.recon.Corrections` made where
        the header states no recon parameter for them.
    """

    #: The corrections of the stream, made when it starts; ``None`` before.
    corrections: Corrections | None = None

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
        super().__init__(
            triggers={"imaging": AcquisitionFlag.LAST_IN_SLICE},
            axes=("average",),
            reject_flags=AcquisitionFlag.IS_PHASECORR_DATA,
        )
        compression = (
            () if virtual_coils is None else (CompressChannels(virtual_coils),)
        )
        self.gadgets = (
            Prewhiten(),
            *compression,
            AsymmetricEcho(),
            RemoveReadoutOversampling(),
        )
        self.wavelet = wavelet
        self.iterations = iterations
        self.virtual_coils = virtual_coils
        self.kspace_filter = kspace_filter
        self.gradient_unwarp = gradient_unwarp
        self.intensity_correction = intensity_correction

    def startup(self, context: ReconContext) -> None:
        super().startup(context)
        self._calibration_logged: set[str] = set()
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
        unit = self.unit_image(context, data)
        if unit is None:
            return None
        buffer, data, image = unit
        corrections = self.corrections
        history = ["PULSERVER", "PYTHON", "PICS"]
        if corrections is not None:
            image = corrections.unwarped(image, context, data)
            history += corrections.applied()
        attributes = {"ImageProcessingHistory": history}
        results = [ReconResult(np.abs(image), attributes=attributes)]
        if np.iscomplexobj(image) and self._contrasts_are_images(context, buffer):
            results += [
                ReconResult(
                    np.real(image),
                    series_index=REAL_SERIES,
                    image_type="real",
                    attributes=attributes,
                ),
                ReconResult(
                    np.imag(image),
                    series_index=IMAGINARY_SERIES,
                    image_type="imaginary",
                    attributes=attributes,
                ),
            ]
        if corrections is not None:
            if corrections.intensity is not None:
                _prescribe_exam_field(context, data, corrections.intensity)
            results = corrections.intensity_corrected(results, data)
        return results if len(results) > 1 else results[0]

    def unit_image(
        self, context: ReconContext, data: ReconData
    ) -> tuple[ReconBuffer, ReconData, np.ndarray] | None:
        """Return the unit's buffer, its data and its image before the image corrections; ``None`` without a buffer.

        The data are compressed to the virtual coils, and a unit holding only
        calibration readouts stores their coil sensitivities and has no image.
        The image is Fermi filtered where the corrections say so, as
        :meth:`image` filters it; gradient nonlinearity and the receive field
        are left to :meth:`recon`.
        """
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
        return buffer, data, image

    def _contrasts_are_images(self, context: ReconContext, buffer: ReconBuffer) -> bool:
        """Return whether the encoding space of ``buffer`` states more than one contrast, each its own image."""
        if "contrast" in self.merge:
            return False
        encodings = getattr(context.header, "encoding", None) or ()
        index = buffer.space.index
        if index >= len(encodings):
            return False
        limits = getattr(encodings[index], "encodingLimits", None)
        contrast = getattr(limits, "contrast", None)
        return int(getattr(contrast, "maximum", 0) or 0) > 0

    def image(
        self,
        kspace: np.ndarray,
        shape: tuple[int, ...],
        context: ReconContext,
        data: ReconData,
    ) -> np.ndarray:
        """Return the complex image of ``(coils, [partitions,] phase encodes, readout)`` k-space, cropped to ``shape``.

        Fermi filtered over its axes of more than one point where the
        stream's :attr:`corrections` say so.
        """
        import torch

        volume = kspace if kspace.ndim == 4 else kspace[:, None]
        volume = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.complex64))
        try:
            maps = coil_maps(context, data, estimate=nlinv_estimate).reshape(
                volume.shape
            )
        except MissingCalibration as missing:
            self._self_calibrated(context, data, missing)
            maps = nlinv_estimate(volume)
        sampled = volume.abs().sum(dim=0) > 0
        image = pics_image(
            volume, maps, sampled, self.wavelet, self.iterations, context.device
        )
        if partial_fourier_axes(sampled.numpy()):
            phase = symmetric_phase(image, sampled)
            image = pics_image(
                volume,
                maps,
                sampled,
                self.wavelet,
                self.iterations,
                context.device,
                x0=(image * phase.conj()).real.to(image.dtype),
                phase=phase,
            )
        image = image.numpy()
        if kspace.ndim == 3:
            image = image[0]
        return self._filtered(np.array(center_crop(image, shape[-image.ndim :])))

    def _self_calibrated(
        self, context: ReconContext, data: ReconData, missing: MissingCalibration
    ) -> None:
        """Log that a unit's maps are fitted to its own k-space centre, and why.

        At ``WARNING`` where the stream stored maps for the unit's slice that
        do not serve it, such as a calibration on another grid, else at
        ``INFO``; a reason already logged in the stream is logged at ``DEBUG``.
        """
        location = int(data.counters.get("slice", 0))
        reason = str(missing).replace(":\n  ", ": ").replace("\n  ", "; ")
        logged = self._calibration_logged
        if reason in logged:
            level = logging.DEBUG
        elif location in context.coil_maps:
            level = logging.WARNING
        else:
            level = logging.INFO
        logged.add(reason)
        _log.log(
            level,
            "slice %d, set %d: coil sensitivities fitted to the unit's own k-space "
            "centre, which unfolds no undersampling, since %s",
            location,
            int(data.counters.get("set", 0)),
            reason,
        )

    def _filtered(self, image: np.ndarray) -> np.ndarray:
        if self.corrections is None:
            return image
        return self.corrections.filtered(image)

    def wave_image(self, context: ReconContext, data: ReconData) -> np.ndarray:
        """Return the complex image ``(partitions, phase encodes, readout)`` of a wave-CAIPI volume.

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
        to the readout field of view, and Fermi filtered as :meth:`image` is.

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
        return self._filtered(image.numpy())


PLUGIN = PicsRecon()


def _prescribe_exam_field(context: ReconContext, data: ReconData, intensity) -> None:
    """Hand the receive field an earlier calibration of the exam measured in the unit's slice to the intensity correction."""
    fields = context.exam.get(RECEIVE_FIELDS)
    if fields is None:
        return
    field = fields.on(context, data)
    if field is not None:
        intensity.prescribe(int(data.counters.get("slice", 0)), field)


def nlinv_estimate(
    kspace: Any,
    traj: Any = None,
    shape: tuple[int, ...] | None = None,
    *,
    sobolev: float | None = None,
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
    k-space whose calibration grid has such an axis, a slab of 3 partitions
    or fewer on a matrix of 48 or more, is fitted without it. ``sobolev`` is
    the Sobolev weight of the maps on the calibration grid
    (:func:`bartorch.apps.nlinv_maps`); ``None`` for BART's.
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
        **({} if sobolev is None else {"sobolev": sobolev}),
    )


def compressed(
    context: ReconContext,
    data: ReconData,
    virtual_coils: int | None,
    *,
    by_slice: bool = True,
) -> ReconData:
    """Return ``data`` with its receive channels compressed to ``virtual_coils`` (:class:`~pulserver.recon.CoilCompression`, ``by_slice`` as there).

    ``data`` is returned as it is when it holds no more channels than that, or
    ``virtual_coils`` is ``None``.
    """
    buffer = data.data if data.data is not None else data.ref
    if virtual_coils is None or buffer is None or buffer.coils <= virtual_coils:
        return data
    return CoilCompression(virtual_coils, by_slice=by_slice)(context, data)


def pics_image(
    kspace: Any,
    maps: Any,
    sampled: Any,
    wavelet: float,
    iterations: int,
    device: str | None,
    x0: Any = None,
    phase: Any = None,
    l2: float = 0.0,
) -> Any:
    """Return the complex image ``(z, y, x)`` of ``bart pics -S -R W`` over host k-space ``(coils, z, y, x)``.

    The solve of :func:`bartorch.apps.pics` on a grid: ``sampled``, the
    ``(z, y, x)`` lines placed, as its pattern, the scaling estimated from the
    fully sampled centre, or from the adjoint image where k-space has none,
    such as an accelerated EPI whose calibration is a unit of its own
    (:func:`bartorch.optim.data_scaling`), and FISTA over the wavelet term of
    weight
    ``wavelet`` along the axes of at least ``WAVELET_MIN_POINTS`` points. A
    volume of one partition is solved as a plane. The k-space, the maps and
    the iteration's vectors stay on the host, and the encoding computes on
    ``device``; ``None`` is the host. FISTA starts from the image ``x0``,
    ``(z, y, x)``, when one is given, and from zero otherwise. The image is
    on the host.

    With ``phase``, a unit-magnitude ``(z, y, x)`` image, the image is that
    phase times a real image ``m``, and the encoding is that of ``m``: the
    maps carry the phase and the unknown is real (``linop.Real``), so the
    k-space a partial-Fourier axis leaves unsampled is the conjugate of what
    it samples. ``x0`` is then the start of ``m``. With ``wavelet`` 0 the
    solve is conjugate gradients with Tikhonov weight ``l2``, at least
    ``REAL_SOLVE_MIN_L2`` with ``phase``.
    """
    import bartorch
    import torch
    from bartorch import linop, optim, priors

    grid = tuple(kspace.shape[2:] if kspace.shape[1] == 1 else kspace.shape[1:])
    pattern = sampled.reshape(grid).to(torch.complex64)
    measured = kspace * sampled
    shifted = bartorch.fftmod(measured, axes=(-1, -2, -3), inverse=True)
    if phase is not None:
        maps = maps * phase.reshape(1, *maps.shape[1:])
    encoding = linop.CartesianSense(
        maps.reshape(maps.shape[0], *grid), grid, pattern=pattern, device=device
    )
    if phase is not None:
        encoding = encoding @ linop.Real(grid)
    scaling = optim.data_scaling(shifted) or optim.data_scaling(
        shifted.reshape(kspace.shape[0], *grid), A=encoding
    )
    scaling = scaling or 1.0
    if wavelet > 0:
        axes = wavelet_axes(grid, tuple(range(-1, -len(grid) - 1, -1)))
        solve = optim.FISTA(priors.Wavelet(axes, wavelet), maxiter=iterations)
    elif phase is not None:
        solve = optim.CG(
            lambda_=max(float(l2), REAL_SOLVE_MIN_L2), maxiter=int(iterations)
        )
    else:
        solve = optim.CG(lambda_=max(float(l2), CG_MIN_L2), maxiter=int(iterations))
    data = (measured * (1.0 / scaling)).reshape(kspace.shape[0], *grid)
    start = {} if x0 is None else {"x0": (x0 * (1.0 / scaling)).reshape(grid)}
    image = (scaling * solve(data, encoding, **start)).reshape(kspace.shape[1:])
    if phase is None:
        return image
    return image.real.to(torch.complex64) * phase.reshape(image.shape)


def smooth_phase(image):
    """Return the unit-magnitude phase of ``image`` low-passed in plane.

    Parameters
    ----------
    image : torch.Tensor
        Complex ``(z, y, x)`` image, the transform centred.

    Returns
    -------
    torch.Tensor
        ``exp(i angle(g * image))``, complex64, ``g`` a Gaussian k-space
        window of ``PHASE_SMOOTHING_LINES`` lines over ``y`` and ``x``.
    """
    import bartorch
    import torch

    image = image.to(torch.complex64)
    ny, nx = image.shape[-2:]
    y = torch.arange(ny, dtype=torch.float32, device=image.device) - ny // 2
    x = torch.arange(nx, dtype=torch.float32, device=image.device) - nx // 2
    window = torch.exp(
        -(y[:, None] ** 2 + x[None, :] ** 2) / (2 * PHASE_SMOOTHING_LINES**2)
    )
    axes = (-1, -2)
    kspace = bartorch.fft(image, axes, unitary=True).reshape(image.shape)
    smooth = bartorch.fft(kspace * window, axes, inverse=True, unitary=True)
    return torch.exp(1j * torch.angle(smooth.reshape(image.shape))).to(torch.complex64)


def symmetric_phase(image: Any, sampled: Any) -> Any:
    """Return the unit-magnitude phase of ``image`` from the k-space a partial-Fourier ``sampled`` holds on both sides of the centre.

    Along each partial-Fourier axis (:func:`_acquired_range`) the k-space of
    ``image`` is kept where both a sample and its conjugate lie in the
    acquired range, the phase of a homodyne reconstruction.

    Parameters
    ----------
    image : torch.Tensor
        Complex ``(z, y, x)`` image, the transform centred.
    sampled : torch.Tensor
        Boolean mask of the acquired samples, with the shape of ``image``.

    Returns
    -------
    torch.Tensor
        ``exp(i angle(image_low))``, complex64.
    """
    import bartorch
    import torch

    mask = sampled.cpu().numpy()
    kept = np.ones(mask.shape, dtype=bool)
    for axis in range(mask.ndim):
        lines = _acquired_range(mask, axis)
        if lines is None:
            continue
        inside = np.zeros(mask.shape[axis], dtype=bool)
        inside[lines[0] : lines[1] + 1] = True
        shape = [1] * mask.ndim
        shape[axis] = mask.shape[axis]
        kept = kept & inside.reshape(shape)
    axes = tuple(range(-1, -mask.ndim - 1, -1))
    # Index i of a centred axis of n points holds the conjugate of index
    # 2 (n // 2) - i, the mirror image shifted by one point where n is even.
    mirrored = np.roll(np.flip(kept), [1 - n % 2 for n in mask.shape], axis=axes)
    symmetric = torch.from_numpy((kept & mirrored).astype(np.float32))
    kspace = bartorch.fft(image.to(torch.complex64), axes, unitary=True)
    low = bartorch.fft(
        kspace.reshape(image.shape) * symmetric.to(image.device),
        axes,
        inverse=True,
        unitary=True,
    ).reshape(image.shape)
    return torch.exp(1j * torch.angle(low)).to(torch.complex64)


def partial_fourier_axes(sampled: np.ndarray) -> bool:
    """Return whether any axis of the boolean mask ``sampled`` is partial Fourier (:func:`_acquired_range`)."""
    return any(
        _acquired_range(sampled, axis) is not None for axis in range(sampled.ndim)
    )


def _acquired_range(mask: np.ndarray, axis: int) -> tuple[int, int] | None:
    """Return the first and last line acquired along a partial-Fourier ``axis``; ``None`` where it is not one."""
    profile = mask.any(axis=tuple(a for a in range(mask.ndim) if a != axis))
    lines = np.flatnonzero(profile)
    if lines.size < 2:
        return None
    gap = int(np.diff(lines).max()) - 1
    low, high = int(lines[0]), profile.size - 1 - int(lines[-1])
    # Line 0 of an even axis has no conjugate on the grid: a scan that
    # reverses its phase encode leaves it out without being partial Fourier.
    unpaired = int(profile.size % 2 == 0 and low == 1 and high == 0)
    if max(low, high) - unpaired <= gap:
        return None
    return int(lines[0]), int(lines[-1])


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
