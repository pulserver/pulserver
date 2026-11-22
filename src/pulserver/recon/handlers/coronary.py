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
    image, the root sum of squares over the coils of the lines' transform
    along the readout and the phase encodes, is matched to the first
    segment's over the central ``region`` of the readout field of view of the
    reconstruction, a template that, narrowed to the heart, leaves out the
    chest wall and back that do not move with it: the displacement is the
    move of the image of largest normalised correlation with the template,
    refined to a fraction of a pixel. Each imaging
    readout of the segment is then multiplied by the linear phase that moves
    its image back by that displacement, along the readout and the phase
    encodes; the partition axis, which the navigator projects out, is left
    as it is. Readouts of a scan without navigators pass unchanged.

    The displacements from the first segment are kept in :attr:`shifts`, in
    mm along the readout and the phase encodes, by segment.

    Parameters
    ----------
    region : float, default=0.5
        Fraction of the readout field of view of the reconstruction the
        navigator is registered over, about its centre.

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
        """Measure the displacement of the navigator held, in its pixels, and forget its lines."""
        lines, self._lines = self._lines, []
        segment = self._segment
        order = sorted(range(len(lines)), key=lambda i: lines[i][0])
        first = lines[order[0]][0]
        span = lines[order[-1]][0] - first + 1
        coils, samples = lines[0][1].shape
        kspace = np.zeros((coils, span, samples), np.complex64)
        for i in order:
            kspace[:, lines[i][0] - first] = lines[i][1]
        images = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(kspace, axes=(1, 2))), axes=(1, 2)
        )
        image = np.sqrt((np.abs(images) ** 2).sum(axis=0))
        space = int(acquisition_label(acquisition, "encoding_space_ref", 0) or 0)
        if self._reference is None:
            ratio = (
                self._oversampling[space] if space < len(self._oversampling) else 1.0
            )
            kept = max(2, round(samples * self.region / ratio))
            self._window = slice((samples - kept) // 2, (samples - kept) // 2 + kept)
            self._reference = image
            shift_y, shift_x = 0.0, 0.0
        else:
            shift_y, shift_x = _matched(self._reference, image, self._window)
        self._pixels[segment] = (shift_x, shift_y, span)
        fov = self._fov[space] if space < len(self._fov) else (0.0, 0.0)
        self.shifts[segment] = (shift_x * fov[0] / samples, shift_y * fov[1] / span)
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
    reference: np.ndarray, image: np.ndarray, window: slice
) -> tuple[float, float]:
    """Return the displacement ``(y, x)`` of ``image`` from ``reference``, in pixels, by matching the reference's ``window`` of columns.

    The template is the reference over ``window``; the image is moved by
    every whole pixel up to a quarter of the window along the columns and
    two pixels along the rows, and the move of the largest normalised
    correlation with the template is refined by a parabola through its
    neighbours along each axis.
    """
    template = reference[:, window]
    template = template - template.mean()
    reach_x = max(1, (window.stop - window.start) // 4)
    reach_y = min(2, (image.shape[0] - 1) // 2)
    moves_y = np.arange(-reach_y, reach_y + 1)
    moves_x = np.arange(-reach_x, reach_x + 1)
    score = np.empty((moves_y.size, moves_x.size))
    for i, dy in enumerate(moves_y):
        for j, dx in enumerate(moves_x):
            part = np.roll(image, (-dy, -dx), axis=(0, 1))[:, window]
            part = part - part.mean()
            norm = np.linalg.norm(part) * np.linalg.norm(template)
            score[i, j] = float(np.sum(part * template) / norm) if norm > 0 else 0.0
    i, j = np.unravel_index(int(np.argmax(score)), score.shape)

    def refined(values: np.ndarray, at: int) -> float:
        if at == 0 or at == values.size - 1:
            return 0.0
        left, centre, right = values[at - 1], values[at], values[at + 1]
        curvature = left - 2 * centre + right
        return 0.5 * (left - right) / curvature if curvature < 0 else 0.0

    return (
        float(moves_y[i] + refined(score[:, j], i)),
        float(moves_x[j] + refined(score[i, :], j)),
    )


def _encoded_fov(encoding: Any) -> tuple[float, float]:
    """Return the encoded field of view along the readout and the phase encodes in mm; 0 where not stated."""
    fov = getattr(getattr(encoding, "encodedSpace", None), "fieldOfView_mm", None)
    return float(getattr(fov, "x", 0) or 0), float(getattr(fov, "y", 0) or 0)


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
