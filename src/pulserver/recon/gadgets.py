"""Gadgets applied to each readout on arrival: prewhitening, coil compression, asymmetric-echo filling and readout oversampling removal."""

from __future__ import annotations

__all__ = [
    "AsymmetricEcho",
    "CompressChannels",
    "Prewhiten",
    "RemoveReadoutOversampling",
]

import logging
from typing import Any

import numpy as np

from ..mrd._acquisitions import AcquisitionFlag
from ..mrd._metadata import acquisition_label, has_acquisition_flag
from ._buffers import discards, echo_centre
from ._calibration import (
    MissingCalibration,
    Whitening,
    basis_of,
    coil_labels,
    digest,
    projected_readout,
    rovir_basis,
)
from ._units import carries
from .plugin import NOISE_COVARIANCE, Gadget

#: The flags either of which marks a navigator readout: pypulseqpp's ``NAV``
#: label, or ``RTFEEDBACK``.
NAVIGATOR = AcquisitionFlag.IS_NAVIGATION_DATA | AcquisitionFlag.IS_RTFEEDBACK_DATA

#: Readouts that are not sampled on the imaging grid: the gadgets pass them on
#: as they are.
_OFF_GRID = NAVIGATOR | AcquisitionFlag.IS_NOISE_MEASUREMENT

#: Readouts that take the channels of the scan they are played in: navigators
#: and odd/even phase references.
_FOLLOWING = NAVIGATOR | AcquisitionFlag.IS_PHASECORR_DATA

#: The flags either of which marks a calibration readout: pypulseqpp's ``REF``
#: and ``IMA`` labels.
CALIBRATION = (
    AcquisitionFlag.IS_PARALLEL_CALIBRATION
    | AcquisitionFlag.IS_PARALLEL_CALIBRATION_AND_IMAGING
)


class Prewhiten(Gadget):
    """Whiten the channels of each readout with the stream's noise measurement.

    Readouts flagged ``IS_NOISE_MEASUREMENT`` are collected and consumed: no
    unit receives them. At the first readout that is not noise, the whitening
    matrix ``W`` is computed from the collected samples, with ``W Ψ W^H = I``
    for their channel covariance ``Ψ`` (:func:`bartorch.tools.whiten`), and is
    stored in ``context.noise``. Every readout of the stream, that one
    included, is then multiplied by ``W`` from the left, so the noise of the
    whitened channels has unit covariance. A readout whose ``sample_time_us``
    differs from that of the noise readouts is also multiplied by the square
    root of the ratio of the two, the noise variance of a sample being
    inversely proportional to its dwell time. Noise readouts after that first
    readout are consumed and not used.

    Without noise readouts, ``W`` is the exam's ``NOISE_COVARIANCE``, which
    :meth:`startup` loads when its coil labels are those of the header. With
    neither, readouts pass unchanged and a warning is logged once. A plugin
    that rejects noise readouts with ``reject_flags`` gives the gadget no noise
    to estimate from. bartorch is imported when ``W`` is computed.

    Corresponds to Gadgetron's ``NoiseAdjustGadget``.

    Parameters
    ----------
    required : bool, default=False
        Raise :class:`~pulserver.recon.MissingCalibration`, rather than log a
        warning, at the first readout that is not noise when there is no ``W``
        to apply.

    Raises
    ------
    ValueError
        From a call, if the collected noise has at most as many samples as
        channels, or a readout does not have the channels of ``W``.
    """

    def __init__(self, required: bool = False) -> None:
        self.required = required

    def startup(self, context):
        super().startup(context)
        self._noise: list[np.ndarray] = []
        self._dwell_us: float | None = None
        self._resolved = False
        context.noise = None
        stored = context.exam.get(NOISE_COVARIANCE)
        if isinstance(stored, Whitening) and stored.coils == coil_labels(
            context.header
        ):
            context.noise = stored

    def __call__(self, acquisition: Any, data: np.ndarray) -> np.ndarray | None:
        if has_acquisition_flag(acquisition, "ACQ_IS_NOISE_MEASUREMENT"):
            if not self._resolved:
                self._noise.append(np.array(data))
                self._dwell_us = self._dwell_us or _dwell(acquisition)
            return None
        if not self._resolved:
            self._resolve()
        whitening = self.context.noise
        return data if whitening is None else _whitened(whitening, acquisition, data)

    def publish(self) -> None:
        """Store the whitening of the stream as the exam's ``NOISE_COVARIANCE``.

        For the ``finish`` of a plugin that reconstructs a noise series, whose
        readouts are all noise, so that no readout resolves the whitening.

        Raises
        ------
        MissingCalibration
            If the stream has neither noise readouts nor a stored covariance.
        """
        if self._noise:
            self._resolve()
        if self.context.noise is None:
            raise MissingCalibration("the stream has no noise to publish")
        self.context.exam[NOISE_COVARIANCE] = self.context.noise

    def _resolve(self) -> None:
        context = self.context
        if self._noise:
            context.noise = _whitening(self._noise, self._dwell_us, context.header)
            self._noise = []
        elif context.noise is None:
            message = (
                "readouts are not prewhitened: the stream has no noise readouts "
                "before its first other readout, and the exam stores no noise "
                "covariance for its coil labels"
            )
            if self.required:
                raise MissingCalibration(message)
            logging.getLogger(__name__).warning(message)
        self._resolved = True


#: Signal fraction of the brightest pixel above which a pixel of the
#: calibration's channel images belongs to the head, for :class:`CompressChannels`
#: rejecting the scalp.
HEAD_THRESHOLD = 0.12
#: Thickness (m) of the shell inside the head's outline whose signal
#: :class:`CompressChannels` rejects: the scalp, brightest next to the coil
#: elements.
SCALP_SHELL_M = 0.010
#: Depth (m) inside the head's outline beyond which :class:`CompressChannels`
#: keeps the signal.
BRAIN_DEPTH_M = 0.016
#: ROVir directions :class:`CompressChannels` draws its virtual channels from,
#: per virtual channel.
ROVIR_DIRECTIONS_PER_CHANNEL = 2


class CompressChannels(Gadget):
    """Project the receive channels of each readout onto ``n_virtual`` virtual channels as it arrives.

    The calibration readouts (``REF`` or ``IMA``) that lead a scan are held
    until the first readout that is not one: the basis is the eigenvectors of
    the channel covariance of every sample of the held readouts of the first
    contrast, of all of them where none is of the first contrast
    (:func:`bartorch.tools.cc`), of which the ``n_virtual`` with the largest
    eigenvalues are kept. It is stored in ``context.coil_compression``, so the
    maps :func:`~pulserver.recon.coil_maps` estimates record it. The held
    readouts are then given on, projected, ahead of that readout, and every
    later readout is projected as it arrives, so a buffer holds
    ``n_virtual`` channels. Readouts held when the stream ends are given on
    by :meth:`release`.

    Navigator and odd/even phase-reference readouts are held, projected and
    given on with the readouts around them, so every readout of a buffer or a
    phase fit has the same channels; they give no basis. A Cartesian stream
    whose first readout of neither kind is not a calibration readout holds
    its readouts up to the k-space centre line of their encoding space, the
    header's ``center``, and they give the basis. A stream of an encoding
    space the header does not state as Cartesian that opens without
    calibration, or whose readouts have no more than ``n_virtual`` channels,
    passes unchanged, as do noise readouts. Place the gadget after
    :class:`Prewhiten`, so the basis is that of whitened channels, and before
    any gadget that changes the header of a readout: a held readout is given
    on with its acquisition as this gadget received it.

    Corresponds to Gadgetron's ``PCACoilGadget`` and
    ``CoilReductionGadget``.

    With ``reject_scalp`` the basis is region-optimized (ROVir,
    :func:`~pulserver.recon._calibration.rovir_basis`) instead: the held
    calibration lines of a single-slice Cartesian space are imaged, the head
    is the pixels above ``HEAD_THRESHOLD`` of the brightest, and the virtual
    channels keep the signal deeper than ``BRAIN_DEPTH_M`` inside its outline
    and reject that of the outer ``SCALP_SHELL_M``. Next to the coil elements
    the scalp is far brighter than the brain, and coil maps err most along
    it; in an accelerated scan those errors fold the scalp into the brain.
    The lines are put on the readout grid by :attr:`readout_grid`, a callable
    of an acquisition and its readout that returns the readout on the grid
    and leaves the acquisition as it is; without it, or without a Cartesian
    calibration, the basis is the channel covariance's.

    Parameters
    ----------
    n_virtual : int
        Virtual channels kept.
    reject_scalp : bool, default=False
        Whether the virtual channels reject the scalp's signal.
    """

    readout_grid = None

    def __init__(self, n_virtual: int, reject_scalp: bool = False) -> None:
        if n_virtual < 1:
            raise ValueError(f"n_virtual is {n_virtual}; it has to be at least 1")
        self.n_virtual = int(n_virtual)
        self.reject_scalp = bool(reject_scalp)

    def startup(self, context):
        super().startup(context)
        self._held: list[tuple[Any, np.ndarray]] = []
        self._passing = False
        self._centres = _kspace_centres(getattr(context, "header", None))
        self._planes = _planes(getattr(context, "header", None))

    def __call__(self, acquisition: Any, data: np.ndarray) -> Any:
        if self._passing or carries(acquisition, AcquisitionFlag.IS_NOISE_MEASUREMENT):
            return data
        basis = self.context.coil_compression
        if basis is not None:
            return projected_readout(data, basis)
        if data.shape[0] <= self.n_virtual:
            self._passing = True
            return self._unchanged(acquisition, data)
        self._held.append((acquisition, np.array(data)))
        if carries(acquisition, CALIBRATION | _FOLLOWING):
            return None
        if not any(carries(a, CALIBRATION) for a, _ in self._held):
            # A Cartesian scan that opens with no calibration gives the basis
            # from its own readouts, up to its k-space centre.
            centre = self._at_centre(acquisition)
            if centre is None:
                self._passing = True
                return self._unchanged(*self._held.pop())
            if not centre:
                return None
        return self.release()

    def _at_centre(self, acquisition: Any) -> bool | None:
        """Return whether ``acquisition`` reads the k-space centre line of its encoding space; ``None`` where the header states no Cartesian space for it."""
        space = int(acquisition_label(acquisition, "encoding_space_ref", 0) or 0)
        if space not in self._centres:
            return None
        line, partition = self._centres[space]
        if int(acquisition_label(acquisition, "kspace_encode_step_1", 0) or 0) != line:
            return False
        return (
            partition is None
            or int(acquisition_label(acquisition, "kspace_encode_step_2", 0) or 0)
            == partition
        )

    def _unchanged(self, acquisition: Any, data: np.ndarray) -> Any:
        """Return ``data``, after the readouts held as they arrived when there are any."""
        held, self._held = self._held, []
        return [*held, (acquisition, data)] if held else data

    def release(self) -> list[tuple[Any, np.ndarray]]:
        """Return the readouts held, projected onto the basis they give, in the order they arrived."""
        if not self._held:
            return []
        basis = self.context.coil_compression or self._basis()
        held, self._held = self._held, []
        return [
            (acquisition, projected_readout(data, basis)) for acquisition, data in held
        ]

    def _basis(self) -> Any:
        """Return the stream's basis, estimated from the held readouts when it has none."""
        if self.context.coil_compression is None:
            calibration = [
                (acquisition, data)
                for acquisition, data in self._held
                if carries(acquisition, CALIBRATION)
            ] or [
                (acquisition, data)
                for acquisition, data in self._held
                if not carries(acquisition, _FOLLOWING)
            ]
            first = [
                data
                for acquisition, data in calibration
                if int(acquisition_label(acquisition, "contrast", 0) or 0) == 0
            ]
            readouts = first or [data for _, data in calibration]
            regions = self._regions(calibration) if self.reject_scalp else None
            if regions is not None:
                images, wanted, rejected = regions
                keep = min(ROVIR_DIRECTIONS_PER_CHANNEL * self.n_virtual, len(images))
                self.context.coil_compression = rovir_basis(
                    images, wanted, rejected, self.n_virtual, keep
                )
            else:
                self.context.coil_compression = basis_of(
                    np.concatenate(readouts, axis=1), self.n_virtual
                )
        return self.context.coil_compression

    def _regions(
        self, calibration: list[tuple[Any, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return the channel images of the calibration lines, the brain and the scalp; ``None`` where they cannot be formed."""
        if self.readout_grid is None:
            return None
        spaces = [
            int(acquisition_label(acquisition, "encoding_space_ref", 0) or 0)
            for acquisition, _ in calibration
        ]
        planar = [space for space in spaces if space in self._planes]
        if not planar:
            return None
        space = max(set(planar), key=planar.count)
        lines_n, pixel_m = self._planes[space]
        # Each line from the readouts of one direction where it has them, so
        # the odd/even phase does not ghost the image.
        lines: dict[int, dict[bool, list[np.ndarray]]] = {}
        for (acquisition, data), own in zip(calibration, spaces, strict=True):
            line = int(acquisition_label(acquisition, "kspace_encode_step_1", 0) or 0)
            if own != space or not 0 <= line < lines_n:
                continue
            gridded = self.readout_grid(acquisition, data)
            if gridded is None:
                continue
            backwards = carries(acquisition, AcquisitionFlag.IS_REVERSE)
            if backwards:
                gridded = gridded[:, ::-1]
            lines.setdefault(line, {}).setdefault(backwards, []).append(gridded)
        if not lines:
            return None
        columns = {
            d.shape[1] for by in lines.values() for ds in by.values() for d in ds
        }
        if len(columns) != 1:
            return None
        kspace = np.zeros((calibration[0][1].shape[0], lines_n, columns.pop()), complex)
        for line, by_direction in lines.items():
            kspace[:, line] = np.mean(
                by_direction.get(False) or by_direction[True], axis=0
            )
        images = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1))), axes=(-2, -1)
        )
        wanted, rejected = head_regions(np.sqrt((np.abs(images) ** 2).sum(0)), pixel_m)
        if not wanted.any() or not rejected.any():
            return None
        return images, wanted, rejected


class AsymmetricEcho(Gadget):
    """Zero-fill a partial echo to the full echo that is symmetric about its ``center_sample``.

    A readout whose echo does not lie at ``samples // 2`` has more samples on
    one side of the echo than the other. Zeros are added on the shorter side,
    making a full echo of ``2 * max(center_sample, samples - center_sample)``
    samples. The zeros are recorded as discarded: ``discard_pre`` and
    ``discard_post`` grow by the zeros added at each end, and ``center_sample``
    becomes the centre of the full echo, so that the buffers place the acquired
    samples only. A readout already centred passes unchanged, as do navigator
    and noise readouts and a readout carrying a trajectory over more than one
    axis, which is not an echo along one line. An acquisition that states no
    ``center_sample`` is centred.

    This is Gadgetron's ``AsymmetricEchoAdjustROGadget``.
    """

    def __call__(self, acquisition: Any, data: np.ndarray) -> np.ndarray:
        if carries(acquisition, _OFF_GRID) or _off_line(acquisition):
            return data
        samples = data.shape[-1]
        centre = echo_centre(acquisition, samples)
        if centre == samples // 2:
            return data
        before = max(0, samples - 2 * centre)
        after = max(0, 2 * centre - samples)
        full = np.zeros((data.shape[0], before + samples + after), dtype=data.dtype)
        full[:, before : before + samples] = data
        pre, post = discards(acquisition)
        acquisition.discard_pre = pre + before
        acquisition.discard_post = post + after
        acquisition.center_sample = full.shape[-1] // 2
        return full


class RemoveReadoutOversampling(Gadget):
    """Crop each readout to the readout field of view of the reconstruction.

    The ratio of the field of view of the readout's encoding space to that of
    its reconstruction space is the oversampling. A readout of ``n`` samples is
    cropped to ``n / ratio`` samples in the image domain
    (:func:`bartorch.remove_readout_oversampling`); ``center_sample`` and the
    discards are divided by the ratio. The readout has to be a full echo with
    ``center_sample`` at ``samples // 2``, which :class:`AsymmetricEcho`
    makes of a partial one. Readouts of a space whose fields of view are equal
    or not stated pass unchanged, as do navigator and noise readouts.

    This is Gadgetron's ``RemoveROOversamplingGadget``. bartorch is imported
    when the first readout is cropped.

    Raises
    ------
    ValueError
        From a call, if a readout to be cropped is not a centred full echo.
    """

    def startup(self, context):
        super().startup(context)
        encodings = getattr(context.header, "encoding", None) or ()
        self.oversampling = [_oversampling(encoding) for encoding in encodings]

    def __call__(self, acquisition: Any, data: np.ndarray) -> np.ndarray:
        space = int(acquisition_label(acquisition, "encoding_space_ref", 0) or 0)
        ratio = self.oversampling[space] if space < len(self.oversampling) else 1.0
        if ratio <= 1.0 or carries(acquisition, _OFF_GRID):
            return data
        samples = data.shape[-1]
        centre = echo_centre(acquisition, samples)
        if centre != samples // 2:
            raise ValueError(
                f"a readout of {samples} samples with its echo at {centre} is not a "
                f"centred full echo, which removing the readout oversampling needs; "
                f"AsymmetricEcho completes a partial echo"
            )
        import torch
        from bartorch import remove_readout_oversampling

        target = max(1, round(samples / ratio))
        cropped = remove_readout_oversampling(
            torch.from_numpy(np.ascontiguousarray(data, dtype=np.complex64)),
            target,
            axis=-1,
        ).numpy()
        pre, post = discards(acquisition)
        acquisition.discard_pre = round(pre / ratio)
        acquisition.discard_post = round(post / ratio)
        acquisition.center_sample = target // 2
        return cropped


def _oversampling(encoding: Any) -> float:
    """Return the encoded to reconstructed field-of-view ratio along the readout; 1 where either is not stated."""

    def readout_fov(space: Any) -> float:
        fov = getattr(space, "fieldOfView_mm", None)
        return float(getattr(fov, "x", 0) or 0)

    encoded = readout_fov(getattr(encoding, "encodedSpace", None))
    recon = readout_fov(getattr(encoding, "reconSpace", None))
    return encoded / recon if encoded > 0 and recon > 0 else 1.0


def _kspace_centres(header: Any) -> dict[int, tuple[int | None, int | None]]:
    """Return the k-space centre line and, in a volume, partition of each Cartesian encoding space of ``header``."""
    if header is None or not getattr(header, "encoding", None):
        return {}
    from ..mrd._header import EncodingSpace

    centres = {}
    for space in EncodingSpace.all_from_header(header):
        if not space.cartesian:
            continue
        line = space.phase_center
        line = space.phase_encodes // 2 if line is None else line
        partition = None
        if space.partitions > 1:
            partition = space.partition_center
            partition = space.partitions // 2 if partition is None else partition
        centres[space.index] = (line, partition)
    return centres


def _planes(header: Any) -> dict[int, tuple[int, tuple[float, float]]]:
    """Return the phase-encode lines and the pixel size (m) of each single-slice Cartesian encoding space of ``header``."""
    if header is None or not getattr(header, "encoding", None):
        return {}
    from ..mrd._header import EncodingSpace

    planes = {}
    for space in EncodingSpace.all_from_header(header):
        if not space.cartesian or space.partitions > 1 or not space.recon_fov:
            continue
        if len(space.recon_matrix) != 2:
            continue
        (n_y, n_x), (fov_y, fov_x) = space.recon_matrix, space.recon_fov
        planes[space.index] = (space.phase_encodes, (fov_y / n_y, fov_x / n_x))
    return planes


def head_regions(
    rss: np.ndarray, pixel_m: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the brain and the scalp of a coil-combined image: the head deeper than ``BRAIN_DEPTH_M``, and its outer ``SCALP_SHELL_M``.

    The head is the pixels above ``HEAD_THRESHOLD`` of the brightest, closed
    and with its holes filled, so the dark skull and the ventricles stay in it.

    Parameters
    ----------
    rss : ndarray
        ``(n_y, n_x)`` coil-combined magnitude.
    pixel_m : tuple of float
        Pixel size (m) along y and x.

    Returns
    -------
    tuple of ndarray
        Boolean masks of the brain and of the scalp.
    """
    import torch

    head = torch.from_numpy(rss > HEAD_THRESHOLD * rss.max())[None, None].float()

    def dilated(mask, radius):
        size = (2 * radius[0] + 1, 2 * radius[1] + 1)
        return torch.nn.functional.max_pool2d(mask, size, 1, radius)

    def eroded(mask, radius):
        return 1 - dilated(1 - mask, radius)

    def radius(metres):
        return tuple(max(1, round(metres / p)) for p in pixel_m)

    closing = radius(SCALP_SHELL_M / 2)
    head = eroded(dilated(head, closing), closing)
    # Outside is what the border reaches through pixels outside the head.
    outside = torch.zeros_like(head)
    outside[..., 0, :] = outside[..., -1, :] = 1
    outside[..., :, 0] = outside[..., :, -1] = 1
    outside *= 1 - head
    while True:
        grown = dilated(outside, (1, 1)) * (1 - head)
        if torch.equal(grown, outside):
            break
        outside = grown
    head = 1 - outside
    brain = eroded(head, radius(BRAIN_DEPTH_M))
    scalp = head * (1 - eroded(head, radius(SCALP_SHELL_M)))
    return brain[0, 0].bool().numpy(), scalp[0, 0].bool().numpy()


def _off_line(acquisition: Any) -> bool:
    return int(acquisition_label(acquisition, "trajectory_dimensions", 0) or 0) > 1


def _dwell(acquisition: Any) -> float | None:
    """Return the dwell time of an acquisition in µs, ``None`` where it states none."""
    return float(acquisition_label(acquisition, "sample_time_us", 0) or 0) or None


def _whitening(
    readouts: list[np.ndarray], dwell_us: float | None, header: Any
) -> Whitening:
    """Return the whitening of noise readouts ``(coils, samples)`` sampled with ``dwell_us``."""
    import torch
    from bartorch import tools

    noise = np.concatenate(readouts, axis=1)
    coils, samples = noise.shape
    if samples <= coils:
        raise ValueError(
            f"{samples} noise samples cannot give the covariance of {coils} "
            f"channels; it takes more samples than channels"
        )
    noise = torch.from_numpy(np.ascontiguousarray(noise, dtype=np.complex64))
    noise = noise.reshape(coils, 1, 1, samples)
    _, matrix = tools.whiten(noise[..., :1], noise, return_matrix=True)
    matrix = matrix.numpy()[:, :, 0, 0, 0]
    labels = coil_labels(header)
    return Whitening(matrix, labels, dwell_us, digest(matrix, labels, dwell_us))


def _whitened(whitening: Whitening, acquisition: Any, data: np.ndarray) -> np.ndarray:
    """Return ``(coils, samples)`` readout ``data`` multiplied by the whitening matrix, scaled for the dwell time."""
    coils = whitening.matrix.shape[0]
    if data.shape[0] != coils:
        raise ValueError(
            f"a readout of {data.shape[0]} channels cannot be whitened by a "
            f"matrix of {coils}"
        )
    dwell = _dwell(acquisition)
    scale = 1.0
    if dwell is not None and whitening.dwell_us is not None:
        scale = float(np.sqrt(dwell / whitening.dwell_us))
    return (scale * whitening.matrix).astype(data.dtype, copy=False) @ data
