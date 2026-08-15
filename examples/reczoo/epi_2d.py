"""Reconstruction for :mod:`pulserver.seqzoo.epi_2d`.

The preprocessing EPI cannot skip, then the Cartesian pipeline: the stream
is partitioned by
:func:`pulserver.recon._mrd.epi.partition_epi_acquisitions` into the
blip-nulled navigator, the opposite-polarity reference and the imaging
lines; the navigator's odd/even lines yield a linear phase fit that is
applied to every reversed line -- after the sample-order flip its ``REV``
polarity demands -- and the volume then reconstructs exactly as
:mod:`pulserver.reczoo.gre_2d` reconstructs. The opposite-polarity
reference is reconstructed alongside as its own series, so the pair a
distortion correction needs -- PyHySCO, through
:func:`pulserver.recon.postprocessing.run_pyhysco` on the two exported
volumes -- leaves the scanner together.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``; the
distortion step additionally needs the external ``recon-distortion`` extra.
"""

from __future__ import annotations

__all__ = [
    "PLUGIN",
    "Epi2DRecon",
    "correct_lines",
    "odd_even_fit",
]

from typing import Any

import numpy as np

from pulserver import AcquisitionBucket, ReconContext, ReconPlugin, ReconResult
from pulserver.recon import has_acquisition_flag
from pulserver.recon._mrd.epi import partition_epi_acquisitions
from pulserver.recon.postprocessing import center_crop, coil_combine
from pulserver.recon.preprocessing import (
    CartesianGridder,
    encoded_shape,
    receiver_channels,
    recon_shape,
)
from pulserver.reczoo.gre_2d import coil_images


def _hybrid(rows: Any) -> np.ndarray:
    """Rows into hybrid space: inverse FFT along the readout."""
    return np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(np.asarray(rows), axes=-1), axis=-1, norm="ortho"),
        axes=-1,
    )


def odd_even_fit(navigator_lines: list[np.ndarray]) -> tuple[float, float]:
    """Fit the odd/even linear phase from blip-nulled navigator lines.

    The middle line was read backwards; against the mean of its two
    like-polarity neighbours, its hybrid-space phase difference is the
    gradient-delay ramp plus a constant -- the two numbers every reversed
    line is corrected by.

    Parameters
    ----------
    navigator_lines : list of numpy.ndarray
        Three ``(coils, samples)`` lines, polarity ``+ - +``, reversed lines
        already flipped back into readout order.

    Returns
    -------
    tuple of float
        Phase slope (radians per sample) and intercept (radians).
    """
    forward = 0.5 * (_hybrid(navigator_lines[0]) + _hybrid(navigator_lines[2]))
    backward = _hybrid(navigator_lines[1])
    cross = np.sum(forward * np.conj(backward), axis=0)

    weights = np.abs(cross)
    phase = np.unwrap(np.angle(cross))
    samples = np.arange(phase.size)
    keep = weights > 0.1 * weights.max()
    slope, intercept = np.polyfit(samples[keep], phase[keep], 1, w=weights[keep])
    return float(slope), float(intercept)


def correct_lines(
    lines: list[tuple[np.ndarray, bool]], slope: float, intercept: float
) -> list[np.ndarray]:
    """Flip and phase-correct a train's lines into a consistent readout.

    Parameters
    ----------
    lines : list of tuple
        ``(data, reversed)`` per line, data ``(coils, samples)``.
    slope, intercept
        The odd/even fit of :func:`odd_even_fit`.

    Returns
    -------
    list of numpy.ndarray
        The corrected lines, all in forward readout order.
    """
    corrected = []
    for data, backwards in lines:
        row = np.asarray(data)
        if backwards:
            row = row[..., ::-1]
            hybrid = _hybrid(row)
            ramp = slope * np.arange(hybrid.shape[-1]) + intercept
            hybrid = hybrid * np.exp(1j * ramp)
            row = np.fft.fftshift(
                np.fft.fft(np.fft.ifftshift(hybrid, axes=-1), axis=-1, norm="ortho"),
                axes=-1,
            )
        corrected.append(row.astype(np.complex64))
    return corrected


class Epi2DRecon(ReconPlugin):
    """Reconstruct a 2D EPI time series, one image per slice and repetition.

    Parameters
    ----------
    device
        Torch device the reconstruction runs on. ``None`` is the CPU.
    """

    def __init__(self, *, device: Any = None) -> None:
        super().__init__(
            split_on=("ACQ_LAST_IN_MEASUREMENT",),
            reject_flags=("ACQ_IS_NOISE_MEASUREMENT",),
        )
        self.device = device

    def startup(self, context: ReconContext) -> None:
        """Size the grids from the header and collect the stream."""
        n_slices, n_y, n_x = encoded_shape(context.header)
        self.grid = (n_slices, n_y, n_x)
        self.coils = receiver_channels(context.header)
        self.image_shape = recon_shape(context.header)
        self.acquisitions: list[Any] = []

    def receive(self, acquisition: Any, context: ReconContext) -> None:
        """Keep the stream; the partitioning wants it whole."""
        del context
        self.acquisitions.append(acquisition)

    def recon(
        self, bucket: AcquisitionBucket, context: ReconContext
    ) -> list[ReconResult] | None:
        """Partition, phase-correct, and reconstruct at the end of the scan."""
        del context
        last = bucket.acquisitions[-1]
        if not has_acquisition_flag(last, "ACQ_LAST_IN_MEASUREMENT"):
            return None

        groups = partition_epi_acquisitions(self.acquisitions)

        # One odd/even fit per slice, from its navigator triplet.
        fits: dict[int, tuple[float, float]] = {}
        by_slice: dict[int, list[Any]] = {}
        for acquisition in groups.phase_correction:
            by_slice.setdefault(int(acquisition.idx.slice), []).append(acquisition)
        for index, triplet in by_slice.items():
            lines = [
                np.asarray(item.data)[..., :: (-1 if _reversed(item) else 1)]
                for item in triplet[:3]
            ]
            fits[index] = odd_even_fit(lines)

        results = []
        for series, group in enumerate(
            (groups.imaging, groups.reverse_polarity)
        ):
            if not group:
                continue
            repetitions = sorted(
                {int(item.idx.repetition) for item in group}
            )
            for repetition in repetitions:
                buffer = CartesianGridder(self.grid, coils=self.coils)
                for item in group:
                    if int(item.idx.repetition) != repetition:
                        continue
                    index = int(item.idx.slice)
                    slope, intercept = fits.get(index, (0.0, 0.0))
                    (row,) = correct_lines(
                        [(np.asarray(item.data), _reversed(item))],
                        slope,
                        intercept,
                    )
                    buffer.add(
                        row, index, int(item.idx.kspace_encode_step_1)
                    )
                for index in range(self.grid[0]):
                    kspace, _ = buffer[index]
                    image = coil_combine(
                        coil_images(kspace, np.ones(kspace.shape[1:], bool)),
                        coil_axis=0,
                    )
                    results.append(
                        ReconResult(
                            center_crop(
                                np.abs(image), self.image_shape
                            ).transpose(),
                            reference=-1,
                            series_index=series * 1000 + repetition,
                            image_index=index,
                            image_type="magnitude",
                            dicom=True,
                        )
                    )
        return results


def _reversed(acquisition: Any) -> bool:
    """Whether the line was read backwards, by its MRD flag."""
    return has_acquisition_flag(acquisition, "ACQ_IS_REVERSE")


PLUGIN = Epi2DRecon()
