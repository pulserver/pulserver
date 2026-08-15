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

When the sequence ran multiband (``SMS_EXCITATION``), the stream carries a
single-band reference (``ACQ_IS_PARALLEL_CALIBRATION``) and blipped-CAIPI
multiband shots instead. The reference gives each slice its coil sensitivities,
and a model-based solve (:class:`pulserver.recon.physics.SMS`) unfolds every
group back into its bands against the CAIPI phase the gz blips played.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``; the
distortion step additionally needs the external ``recon-distortion`` extra.
"""

from __future__ import annotations

__all__ = [
    "PLUGIN",
    "Epi2DRecon",
    "coil_maps_from_reference",
    "correct_lines",
    "odd_even_fit",
    "separate_slices",
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
from pulserver.reczoo.gre_2d import (
    coil_images,
    fill_partial_echo,
    sense,
    sensitivities,
)


def _ifft2c(kspace: Any) -> np.ndarray:
    """Centred 2D inverse FFT over the last two axes."""
    axes = (-2, -1)
    return np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(kspace, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


def coil_maps_from_reference(kspace: Any) -> np.ndarray:
    """Per-slice coil sensitivities from a fully sampled single-band reference.

    The reference slice is fully sampled, so its coil images are the
    sensitivities up to the object they share; dividing by the root
    sum-of-squares removes that common magnitude and leaves unit-norm maps a
    model-based separation can solve against.

    Parameters
    ----------
    kspace
        One slice's reference k-space, ``(coil, ky, kx)``.

    Returns
    -------
    numpy.ndarray
        Coil maps, ``(coil, ky, kx)``, root sum-of-squares one.
    """
    images = _ifft2c(np.asarray(kspace))
    rss = np.sqrt(np.sum(np.abs(images) ** 2, axis=0, keepdims=True))
    return (images / np.maximum(rss, 1e-8 * rss.max())).astype(np.complex64)


def separate_slices(
    collapsed: Any,
    coil_maps: Any,
    caipi_encoding: Any,
    *,
    regularization: float = 1e-3,
    iterations: int = 40,
    device: Any = None,
) -> np.ndarray:
    """Unfold one multiband group into its bands (model-based SMS).

    Parameters
    ----------
    collapsed
        The group's multiband k-space, ``(coil, ky, kx)``.
    coil_maps
        Per-band coil maps, ``(band, coil, ky, kx)``.
    caipi_encoding
        The CAIPI slice phase played, ``(band, ky, 1)``.
    regularization, iterations
        Tikhonov weight and iteration ceiling of the CG solve.
    device
        Torch device. ``None`` is the CPU.

    Returns
    -------
    numpy.ndarray
        One complex image per band, ``(band, ky, kx)``.
    """
    import torch

    from pulserver.recon import pics
    from pulserver.recon.physics import SMS, Cartesian2D

    device = "cpu" if device is None else device
    coil_maps = np.asarray(coil_maps)
    n_bands, _, n_y, n_x = coil_maps.shape
    mask = torch.ones((1, 1, n_y, n_x), dtype=torch.float32, device=device)
    per_band = [
        Cartesian2D(mask, torch.as_tensor(coil_maps[band], device=device)[None])
        for band in range(n_bands)
    ]
    physics = SMS(per_band, torch.as_tensor(np.asarray(caipi_encoding), device=device))
    image = pics(
        torch.as_tensor(np.asarray(collapsed), device=device)[None],
        physics,
        regularization=regularization,
        iterations=iterations,
    )[0]
    return image.cpu().numpy() if hasattr(image, "cpu") else np.asarray(image)


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
    regularization
        Tikhonov weight of the CG-SENSE solve.
    iterations
        Maximum CG iterations.
    pocs_iterations
        Partial-echo POCS iterations.
    device
        Torch device the reconstruction runs on. ``None`` is the CPU.
    """

    def __init__(
        self,
        *,
        regularization: float = 1e-3,
        iterations: int = 40,
        pocs_iterations: int = 12,
        device: Any = None,
    ) -> None:
        super().__init__(
            split_on=("ACQ_LAST_IN_MEASUREMENT",),
            reject_flags=("ACQ_IS_NOISE_MEASUREMENT",),
        )
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.pocs_iterations = int(pocs_iterations)
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

        # A single-band reference means multiband data: separate the collapsed
        # slices rather than reconstructing each slice on its own.
        if groups.single_band_reference:
            return self._reconstruct_sms(groups, fits)

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
                    kspace, mask = buffer[index]

                    # What the scan sampled selects the reconstruction: a phase
                    # encode with no samples was skipped, and a readout sample
                    # missing from every line is echo never acquired.
                    lines = mask.any(axis=-1)
                    readout = mask.any(axis=0)

                    if lines.all():
                        coils = (
                            coil_images(kspace, mask, device=self.device)
                            if readout.all()
                            else fill_partial_echo(
                                kspace,
                                readout,
                                self.pocs_iterations,
                                device=self.device,
                            )
                        )
                        image = coil_combine(coils, coil_axis=0)
                    else:
                        maps = sensitivities(kspace, mask, device=self.device)
                        image = sense(
                            kspace,
                            mask,
                            maps,
                            readout,
                            regularization=self.regularization,
                            iterations=self.iterations,
                            pocs_iterations=self.pocs_iterations,
                            device=self.device,
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

    def _grid_train(self, items: list[Any], fit: tuple[float, float]) -> Any:
        """Phase-correct a train's lines and grid them into one 2D k-space."""
        _, n_y, n_x = self.grid
        buffer = CartesianGridder((1, n_y, n_x), coils=self.coils)
        slope, intercept = fit
        for item in items:
            (row,) = correct_lines(
                [(np.asarray(item.data), _reversed(item))], slope, intercept
            )
            buffer.add(row, 0, int(item.idx.kspace_encode_step_1))
        return buffer[0]

    def _reconstruct_sms(
        self, groups: Any, fits: dict[int, tuple[float, float]]
    ) -> list[ReconResult]:
        """Separate the collapsed multiband slices against the reference maps.

        The single-band reference gives each slice its coil sensitivities; the
        blipped-CAIPI phase the imaging shots carry, together with those maps,
        is what a model-based solve unfolds a group's bands with. A group's
        bands are its slice and every ``n_groups``-th slice above it, matching
        how the sequence spaced the excited comb.
        """
        n_slices, n_y, _ = self.grid
        # One odd/even fit serves the whole multiband readout: the blip-nulled
        # navigator measured the readout, which every shot shares.
        fit = next(iter(fits.values()), (0.0, 0.0))

        reference: dict[int, list[Any]] = {}
        for item in groups.single_band_reference:
            reference.setdefault(int(item.idx.slice), []).append(item)
        coil_maps = {
            index: coil_maps_from_reference(self._grid_train(items, fit)[0])
            for index, items in reference.items()
        }

        group_ids = sorted({int(item.idx.slice) for item in groups.imaging})
        n_groups = len(group_ids)
        n_bands = n_slices // max(n_groups, 1)

        # The CAIPI slice phase played: band j shifted j / n_bands of the FOV,
        # a linear ramp along ky. The trailing unit axis lands the phase on the
        # phase-encode axis of the (coil, ky, kx) measurement.
        ky = np.arange(n_y)
        caipi = np.exp(
            1j * 2 * np.pi * (np.arange(n_bands)[:, None] / n_bands) * ky[None, :]
        )[..., None].astype(np.complex64)

        results: list[ReconResult] = []
        repetitions = sorted({int(item.idx.repetition) for item in groups.imaging})
        for repetition in repetitions:
            for group in group_ids:
                shots = [
                    item
                    for item in groups.imaging
                    if int(item.idx.repetition) == repetition
                    and int(item.idx.slice) == group
                ]
                collapsed, _ = self._grid_train(shots, fit)
                bands = [group + band * n_groups for band in range(n_bands)]
                maps = np.stack([coil_maps[index] for index in bands])
                images = separate_slices(
                    collapsed,
                    maps,
                    caipi,
                    regularization=self.regularization,
                    iterations=self.iterations,
                    device=self.device,
                )
                for band, slice_index in enumerate(bands):
                    results.append(
                        ReconResult(
                            center_crop(
                                np.abs(images[band]), self.image_shape
                            ).transpose(),
                            reference=-1,
                            series_index=repetition,
                            image_index=slice_index,
                            image_type="magnitude",
                            dicom=True,
                        )
                    )
        return results


def _reversed(acquisition: Any) -> bool:
    """Whether the line was read backwards, by its MRD flag."""
    return has_acquisition_flag(acquisition, "ACQ_IS_REVERSE")


PLUGIN = Epi2DRecon()
