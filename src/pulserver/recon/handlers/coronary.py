"""Whole-heart coronary MRA: each segment's respiratory displacement read from its image navigator and undone in k-space before the pics solve."""

from __future__ import annotations

__all__ = ["PLUGIN", "CoronaryRecon", "SegmentMotion"]

import logging
from typing import Any

import numpy as np

from ...mrd._metadata import acquisition_label
from .._units import carries
from ..gadgets import NAVIGATOR, Prewhiten, _oversampling
from ..plugin import Gadget, ReconContext, ReconResult
from .pics import PicsRecon

LOGGER = logging.getLogger(__name__)


class SegmentMotion(Gadget):
    """Undo each segment's in-plane translation, measured by its image navigator, on its readouts.

    The navigator of a segment is the run of navigator readouts (``NAV``)
    that precede its imaging readouts: lines of the central partition, the
    projection of the slab along the partition axis. They are consumed. Its
    image is the root sum of squares over the coils of the lines' transform,
    the lines placed on the reconstruction's phase-encode grid, so that the
    few central lines give an image sinc-interpolated to its voxels. It is
    matched to the first segment's over the central ``region`` of the field
    of view along the readout and the phase encodes, a template that,
    narrowed to the heart, leaves out the chest wall and back that do not
    move with it: the displacement is the move of the image of largest
    normalised correlation with the template, refined to a fraction of a
    voxel. Each imaging readout of the segment is then multiplied by the
    linear phase that moves its image back by that displacement, along the
    readout and the phase encodes; the partition axis, which the navigator
    projects out, is left as it is. Readouts of a scan without navigators
    pass unchanged.

    The displacements from the first segment are kept in :attr:`shifts`, in
    mm along the readout and the phase encodes, by segment.

    Parameters
    ----------
    region : float, default=0.5
        Fraction of the field of view of the reconstruction, along the
        readout and the phase encodes, the navigator is registered over,
        about its centre.

    Raises
    ------
    ValueError
        If ``region`` is not in (0, 1].
    """

    def __init__(self, region: float = 0.5) -> None:
        if not 0.0 < region <= 1.0:
            raise ValueError(f"region is {region}; it has to be in (0, 1]")
        self.region = float(region)

    def startup(self, context: ReconContext) -> None:
        super().startup(context)
        encodings = getattr(context.header, "encoding", None) or ()
        self._oversampling = [_oversampling(encoding) for encoding in encodings]
        self._fov = [_encoded_fov(encoding) for encoding in encodings]
        self._lines: list[tuple[int, np.ndarray]] = []
        self._segment: int | None = None
        self._reference: np.ndarray | None = None
        self._pixels: dict[int, tuple[float, float, int]] = {}
        self.shifts: dict[int, tuple[float, float]] = {}

    def __call__(self, acquisition: Any, data: np.ndarray) -> np.ndarray | None:
        segment = int(acquisition_label(acquisition, "segment", 0) or 0)
        if carries(acquisition, NAVIGATOR):
            if segment != self._segment:
                self._lines, self._segment = [], segment
            line = int(acquisition_label(acquisition, "kspace_encode_step_1", 0) or 0)
            self._lines.append((line, np.array(data)))
            return None
        if segment == self._segment and self._lines:
            self._register(acquisition)
        if segment not in self._pixels:
            return data
        shift_x, shift_y, lines = self._pixels[segment]
        centre = int(acquisition_label(acquisition, "center_sample", 0) or 0)
        centre_line = _centre_line(self.context.header, acquisition)
        line = int(acquisition_label(acquisition, "kspace_encode_step_1", 0) or 0)
        samples = data.shape[-1]
        k_x = (np.arange(samples) - centre) / samples
        k_y = (line - centre_line) / lines
        # A displacement d multiplies the samples by exp(-2 pi i k d).
        phase = np.exp(2j * np.pi * (k_x * shift_x + k_y * shift_y))
        return (data * phase).astype(data.dtype, copy=False)

    def _register(self, acquisition: Any) -> None:
        """Measure the displacement of the navigator held, in voxels of the reconstruction, and forget its lines."""
        lines, self._lines = self._lines, []
        segment = self._segment
        space = int(acquisition_label(acquisition, "encoding_space_ref", 0) or 0)
        matrix = _phase_encodes(self.context.header, space)
        coils, samples = lines[0][1].shape
        # The navigator's lines on the reconstruction's phase-encode grid:
        # its image is sinc-interpolated to the voxels the displacement is
        # undone in.
        kspace = np.zeros((coils, matrix, samples), np.complex64)
        for line, data in lines:
            kspace[:, min(max(line, 0), matrix - 1)] = data
        import bartorch
        import torch

        images = bartorch.fft(torch.from_numpy(kspace), (1, 2), inverse=True)
        image = bartorch.rss(images.reshape(kspace.shape), 0).numpy()
        if self._reference is None:
            ratio = (
                self._oversampling[space] if space < len(self._oversampling) else 1.0
            )
            kept = max(2, round(samples * self.region / ratio))
            rows = max(3, round(matrix * self.region))
            self._window = (
                slice((matrix - rows) // 2, (matrix - rows) // 2 + rows),
                slice((samples - kept) // 2, (samples - kept) // 2 + kept),
            )
            self._reference = image
            shift_y, shift_x = 0.0, 0.0
        else:
            shift_y, shift_x = _matched(self._reference, image, self._window)
        self._pixels[segment] = (shift_x, shift_y, matrix)
        fov = self._fov[space] if space < len(self._fov) else (0.0, 0.0)
        self.shifts[segment] = (shift_x * fov[0] / samples, shift_y * fov[1] / matrix)
        LOGGER.debug(
            "segment %d displaced %.2f mm along the readout, %.2f mm along the phase encodes",
            segment,
            *self.shifts[segment],
        )


class CoronaryRecon(PicsRecon):
    """Whole-heart coronary MRA: each segment's translation undone (:class:`SegmentMotion`), then the volume solved as :class:`~pulserver.recon.handlers.pics.PicsRecon` solves it.

    Every segment is reconstructed, whatever its respiratory position: the
    scan efficiency is that of the acquisition, with no navigator gating.

    Parameters
    ----------
    region : float, default=0.5
        As :class:`SegmentMotion`.
    **kwargs
        As :class:`~pulserver.recon.handlers.pics.PicsRecon`.
    """

    def __init__(self, region: float = 0.5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        motion = SegmentMotion(region)
        self.gadgets = tuple(
            gadget
            for existing in self.gadgets
            for gadget in (
                (motion, existing) if isinstance(existing, Prewhiten) else (existing,)
            )
        )

    def recon(
        self, context: ReconContext, branch: str, data: Any
    ) -> ReconResult | list[ReconResult] | None:
        result = super().recon(context, branch, data)
        if result is None:
            return None
        results = result if isinstance(result, list) else [result]
        for r in results:
            history = r.attributes.get("ImageProcessingHistory")
            if history is not None and "INAV" not in history:
                history.insert(history.index("PICS"), "INAV")
        return result


def _matched(
    reference: np.ndarray, image: np.ndarray, window: tuple[slice, slice]
) -> tuple[float, float]:
    """Return the displacement ``(y, x)`` of ``image`` from ``reference``, in pixels, within the reference's ``window`` of rows and columns.

    The sub-pixel shift is the phase of the cross-spectrum of the two windows
    (:func:`bartorch.tools.estimate_shift`, BART's ``estshift``).
    """
    import torch
    from bartorch import tools

    shift = tools.estimate_shift(
        torch.from_numpy(np.ascontiguousarray(reference[window], dtype=np.complex64)),
        torch.from_numpy(np.ascontiguousarray(image[window], dtype=np.complex64)),
        (0, 1),
    )
    return -float(shift[0]), -float(shift[1])


def _encoded_fov(encoding: Any) -> tuple[float, float]:
    """Return the encoded field of view along the readout and the phase encodes in mm; 0 where not stated."""
    fov = getattr(getattr(encoding, "encodedSpace", None), "fieldOfView_mm", None)
    return float(getattr(fov, "x", 0) or 0), float(getattr(fov, "y", 0) or 0)


def _phase_encodes(header: Any, space: int) -> int:
    """Return the phase encodes of the encoding space's grid."""
    return int(header.encoding[space].encodedSpace.matrixSize.y)


def _centre_line(header: Any, acquisition: Any) -> int:
    """Return the k-space centre line of the acquisition's encoding space."""
    space = int(acquisition_label(acquisition, "encoding_space_ref", 0) or 0)
    encoding = header.encoding[space]
    limits = getattr(encoding.encodingLimits, "kspace_encoding_step_1", None)
    centre = getattr(limits, "center", None)
    if centre is None:
        return int(encoding.encodedSpace.matrixSize.y) // 2
    return int(centre)


PLUGIN = CoronaryRecon()
