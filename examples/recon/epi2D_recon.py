"""Reconstruction for :mod:`pulserver.app.sequence.epi2D_sequence`.

The preprocessing EPI cannot skip, then the Cartesian pipeline: the stream
is partitioned by
:func:`pulserver.recon.partition_epi_acquisitions` into the
blip-nulled navigator, the opposite-polarity reference and the imaging
lines; the navigator's odd/even lines yield a linear phase fit that is
applied to every reversed line -- after the sample-order flip its ``REV``
polarity demands -- and the volume then reconstructs exactly as
:mod:`pulserver.app.recon.cartesian2D_recon` reconstructs. The opposite-polarity
reference is reconstructed alongside as its own series, so the pair a
distortion correction needs -- PyHySCO, through
:func:`pulserver.recon.postprocessing.run_pyhysco` on the two exported
volumes -- leaves the scanner together.

When the scan is accelerated, its coil sensitivities come from a separate
low-resolution gradient echo (``ACQ_IS_PARALLEL_CALIBRATION``), estimated once
per slice and reused for every frame; the imaging file itself carries no
autocalibration lines, so an undersampled slice is unaliased against those maps
rather than a self-calibrated fit.

When the sequence ran multiband (``SMS_EXCITATION``), the stream carries the
same low-resolution GRE calibration (``ACQ_IS_PARALLEL_CALIBRATION``) plus its
phase navigator, then blipped-CAIPI multiband shots. The calibration gives each
slice its coil sensitivities, and a model-based solve
(:class:`pulserver.recon.physics.SMS`) unfolds every group back into its bands
against the CAIPI phase the gz blips played.

The distortion step needs the external ``pulserver[distortion]`` extra.
"""

from __future__ import annotations

__all__ = [
    "PLUGIN",
    "Epi2DRecon",
    "coil_maps_from_reference",
    "separate_slices",
]

from typing import Any

import numpy as np

from pulserver import AcquisitionBucket, ReconContext, ReconPlugin, ReconResult
from pulserver.recon import (
    NLINV,
    AcquisitionFlag,
    Cartesian2D,
    CartesianGridder,
    center_crop,
    coil_combine,
    correct_lines,
    encoded_shape,
    fftc,
    fill_partial_echo,
    has_acquisition_flag,
    ifftc,
    odd_even_fit,
    partition_epi_acquisitions,
    pics,
    receiver_channels,
    recon_shape,
)


def coil_maps_from_reference(kspace: Any) -> np.ndarray:
    """Per-slice coil sensitivities from a low-resolution reference k-space.

    The reference is a low-resolution gradient echo, so its coil images are
    smooth and, up to the object they share, are the sensitivities; dividing by
    the root sum-of-squares removes that common magnitude and leaves unit-norm
    maps a model-based separation can solve against. The unsampled outer k-space
    reads as zero, which band-limits the images -- exactly the smoothing a
    sensitivity map wants.

    Parameters
    ----------
    kspace
        One slice's reference k-space, ``(coil, ky, kx)``.

    Returns
    -------
    numpy.ndarray
        Coil maps, ``(coil, ky, kx)``, root sum-of-squares one.
    """
    images = ifftc(np.asarray(kspace), axes=(-2, -1))
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
            split_on=AcquisitionFlag.LAST_IN_MEASUREMENT,
            reject_flags=AcquisitionFlag.IS_NOISE_MEASUREMENT,
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
        """Keep the stream rather than placing it.

        Two things have to happen to an EPI line before it belongs anywhere:
        the stream is partitioned by flag into navigator, reverse-polarity
        reference and imaging, and every reversed line is flipped and phase
        corrected against a fit that only exists once its slice's navigator
        triplet has arrived. So this one plugin sorts for itself, which is what
        overriding the hook is for.
        """
        del context
        self.acquisitions.append(acquisition)

    def recon(
        self, bucket: AcquisitionBucket, context: ReconContext
    ) -> list[ReconResult] | None:
        """Partition, phase-correct, and reconstruct at the end of the scan."""
        del context
        if AcquisitionFlag.LAST_IN_MEASUREMENT not in bucket.trigger:
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

        # Multiband data collapses bands into fewer imaged groups than the
        # calibration has slices, and is separated model-based against the
        # single-band reference; a plain accelerated scan instead calibrates
        # slice-for-slice from the low-resolution gradient echo.
        if groups.single_band_reference and self._is_multiband(groups):
            return self._reconstruct_sms(groups, fits)

        # Coil sensitivities from the separate low-resolution GRE calibration
        # (ACQ_IS_PARALLEL_CALIBRATION), estimated once per slice and reused for
        # every frame; absent it, an undersampled slice falls back to a
        # self-calibrated NLINV solve.
        calibration_maps = self._calibration_maps(groups.single_band_reference)

        results = []
        for series, group in enumerate((groups.imaging, groups.reverse_polarity)):
            if not group:
                continue
            repetitions = sorted({int(item.idx.repetition) for item in group})
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
                    buffer.add(row, index, int(item.idx.kspace_encode_step_1))
                for index in range(self.grid[0]):
                    kspace, mask = buffer[index]

                    # What the scan sampled selects the reconstruction: a phase
                    # encode with no samples was skipped, and a readout sample
                    # missing from every line is echo never acquired.
                    lines = mask.any(axis=-1)
                    readout = mask.any(axis=0)

                    if lines.all():
                        # Fully sampled k-space is zero outside the mask, so
                        # the coil-wise adjoint is the centered inverse FFT.
                        coils = (
                            ifftc(kspace, axes=(-2, -1))
                            if readout.all()
                            else fill_partial_echo(
                                kspace, readout, self.pocs_iterations, dimension=2
                            )
                        )
                        image = coil_combine(coils, coil_axis=0)
                    else:
                        maps = calibration_maps.get(index)
                        if maps is None:
                            maps = NLINV(spatial_ndim=2)(
                                kspace[None], mask=mask, device=self.device
                            )
                        image = pics(
                            kspace[None],
                            Cartesian2D(mask[None], maps, device=self.device),
                            regularization=self.regularization,
                            iterations=self.iterations,
                        )[0]
                        if not readout.all():
                            image = fill_partial_echo(
                                fftc(image, axes=(-2, -1)),
                                readout,
                                self.pocs_iterations,
                                dimension=2,
                            )
                    results.append(
                        ReconResult(
                            center_crop(np.abs(image), self.image_shape).transpose(),
                            reference=-1,
                            series_index=series * 1000 + repetition,
                            image_index=index,
                            image_type="magnitude",
                            dicom=True,
                        )
                    )
        return results

    def _is_multiband(self, groups: Any) -> bool:
        """Whether the imaging collapses bands.

        The multiband imaging excites ``n_groups`` combs, so it carries fewer
        distinct slice labels than the single-band reference, which visits every
        slice on its own. A plain accelerated scan images and calibrates the
        same slices, so the two counts match and this is False.
        """
        imaged = {int(item.idx.slice) for item in groups.imaging}
        calibrated = {int(item.idx.slice) for item in groups.single_band_reference}
        return len(calibrated) > len(imaged)

    def _by_slice(self, items: list[Any]) -> dict[int, list[Any]]:
        """Group acquisitions by their slice counter."""
        grouped: dict[int, list[Any]] = {}
        for item in items:
            grouped.setdefault(int(item.idx.slice), []).append(item)
        return grouped

    def _grid_calibration(self, items: list[Any]) -> Any:
        """Grid one slice's GRE calibration into a k-space -- no phase correction.

        A plain gradient echo, so its lines carry no ``REV`` and want no odd/even
        correction; they grid straight into one 2D k-space with only the central
        block filled.
        """
        _, n_y, n_x = self.grid
        buffer = CartesianGridder((1, n_y, n_x), coils=self.coils)
        for item in items:
            buffer.add(np.asarray(item.data), 0, int(item.idx.kspace_encode_step_1))
        return buffer[0]

    def _calibration_maps(self, reference: list[Any]) -> dict[int, Any]:
        """Per-slice coil sensitivities from the low-resolution GRE calibration.

        NLINV reads the fully sampled centre off the mask and resamples the maps
        to the full matrix. Estimated once for the whole time series.
        """
        maps = {}
        for index, items in self._by_slice(reference).items():
            kspace, mask = self._grid_calibration(items)
            maps[index] = NLINV(spatial_ndim=2)(
                kspace[None], mask=mask, device=self.device
            )
        return maps

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
        """Separate the collapsed multiband slices against the calibration maps.

        The low-resolution GRE calibration gives each slice its coil
        sensitivities; the blipped-CAIPI phase the imaging shots carry, together
        with those maps, is what a model-based solve unfolds a group's bands
        with. A group's bands are its slice and every ``n_groups``-th slice above
        it, matching how the sequence spaced the excited comb.
        """
        n_slices, n_y, _ = self.grid
        # One odd/even fit serves the whole multiband readout: the blip-nulled
        # navigator measured the readout, which every shot shares.
        fit = next(iter(fits.values()), (0.0, 0.0))

        # The calibration is a plain gradient echo: grid it without the EPI
        # odd/even correction, then read the smooth per-slice maps off it.
        reference = self._by_slice(groups.single_band_reference)
        coil_maps = {
            index: coil_maps_from_reference(self._grid_calibration(items)[0])
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
    return has_acquisition_flag(acquisition, AcquisitionFlag.IS_REVERSE)


PLUGIN = Epi2DRecon()
