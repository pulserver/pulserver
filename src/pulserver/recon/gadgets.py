"""Gadgets that bring each Cartesian readout to a centred echo on the reconstruction matrix's readout."""

from __future__ import annotations

__all__ = ["AsymmetricEcho", "RemoveReadoutOversampling"]

from typing import Any

import numpy as np

from ..mrd._acquisitions import AcquisitionFlag
from ..mrd._metadata import acquisition_label
from ._buffers import discards, echo_centre
from ._units import carries
from .plugin import Gadget

#: The flags either of which marks a navigator readout: pypulseqpp's ``NAV``
#: label, or ``RTFEEDBACK``.
NAVIGATOR = AcquisitionFlag.IS_NAVIGATION_DATA | AcquisitionFlag.IS_RTFEEDBACK_DATA

#: Readouts that are not sampled on the imaging grid: the gadgets pass them on
#: as they are.
_OFF_GRID = NAVIGATOR | AcquisitionFlag.IS_NOISE_MEASUREMENT


class AsymmetricEcho(Gadget):
    """Zero-fill a partial echo to the full echo that is symmetric about its ``center_sample``.

    A readout whose echo does not lie at ``samples // 2`` has more samples on
    one side of the echo than the other. Zeros are added on the shorter side,
    making a full echo of ``2 * max(center_sample, samples - center_sample)``
    samples. The zeros are recorded as discarded: ``discard_pre`` and
    ``discard_post`` grow by the zeros added at each end, and ``center_sample``
    becomes the centre of the full echo, so that the buffers place the acquired
    samples only. A readout already centred passes unchanged, as do navigator
    and noise readouts. An acquisition that states no ``center_sample`` is
    centred.

    This is Gadgetron's ``AsymmetricEchoAdjustROGadget``.
    """

    def __call__(self, acquisition: Any, data: np.ndarray) -> np.ndarray:
        if carries(acquisition, _OFF_GRID):
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
