"""Gadgets applied to each readout on arrival: prewhitening, and the echo of a Cartesian readout brought to the reconstruction matrix's readout."""

from __future__ import annotations

__all__ = ["AsymmetricEcho", "Prewhiten", "RemoveReadoutOversampling"]

import logging
from typing import Any

import numpy as np

from ..mrd._acquisitions import AcquisitionFlag
from ..mrd._metadata import acquisition_label, has_acquisition_flag
from ._buffers import discards, echo_centre
from ._calibration import MissingCalibration, Whitening, coil_labels, digest
from ._units import carries
from .plugin import NOISE_COVARIANCE, Gadget

#: The flags either of which marks a navigator readout: pypulseqpp's ``NAV``
#: label, or ``RTFEEDBACK``.
NAVIGATOR = AcquisitionFlag.IS_NAVIGATION_DATA | AcquisitionFlag.IS_RTFEEDBACK_DATA

#: Readouts that are not sampled on the imaging grid: the gadgets pass them on
#: as they are.
_OFF_GRID = NAVIGATOR | AcquisitionFlag.IS_NOISE_MEASUREMENT


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
    to estimate from. bartorch is imported when ``W`` is computed; the
    ``coils`` extra installs it.

    Corresponds to Gadgetron's ``NoiseAdjustGadget``.

    Parameters
    ----------
    required
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
    when the first readout is cropped; the ``coils`` extra installs it.

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
    """Return the ratio of an encoding's encoded to reconstructed field of view along the readout, 1 where either is not stated."""

    def readout_fov(space: Any) -> float:
        fov = getattr(space, "fieldOfView_mm", None)
        return float(getattr(fov, "x", 0) or 0)

    encoded = readout_fov(getattr(encoding, "encodedSpace", None))
    recon = readout_fov(getattr(encoding, "reconSpace", None))
    return encoded / recon if encoded > 0 and recon > 0 else 1.0


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
