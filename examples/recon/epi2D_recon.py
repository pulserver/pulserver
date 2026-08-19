"""Reconstruction for :mod:`pulserver.app.sequence.epi2D_sequence`.

The preprocessing EPI cannot skip, then the Cartesian pipeline. A blip-nulled
navigator triplet gives the odd/even phase fit, and every reversed line
is flipped and corrected by it as it arrives -- before it is placed, because a
corrected line is what belongs on the grid. The volume then reconstructs
exactly as :mod:`pulserver.app.recon.cartesian2D_recon` reconstructs it.

The opposite-polarity reference is the scan's second ``SET``, so it is an axis
of the same buffer and comes back as its own series: the pair a distortion
correction needs -- PyHySCO, through
:func:`pulserver.recon.postprocessing.run_pyhysco` on the two exported volumes
-- leaves the scanner together. The distortion step needs the external
``pulserver[distortion]`` extra.

Coil sensitivities come from a separate low-resolution gradient echo
(``ACQ_IS_PARALLEL_CALIBRATION``). That prescan is a subsequence, so the header
gives it an encoding space of its own and its lines never touch the imaging
grid: they fill ``buffers[1]``, and each slice of it calibrates as it closes,
through :func:`pulserver.recon.coil_maps_from_reference`. A plain gradient echo
rather than an EPI train keeps EPI distortion out of the coil maps.

A noise scan, when the scanner sends one, is not imaging data and never reaches
a buffer: it whitens every readout that follows. The prescan is also where the
array's principal channels are read, and the basis goes into ``context.exam``
-- the prescan is its own sequence, so it may be its own stream, and the exam
cache is what carries an artifact from one to the next. Every imaging readout
is compressed onto that basis as it arrives, so the imaging buffer is allocated
at the virtual channel count and never holds the full array.

When the sequence ran multiband (``SMS_EXCITATION``), that same prescan visits
every slice while the imaging excites combs, so the calibration carries more
slices than the imaging does -- which is how the two are told apart here. A
group's bands land in one readout, so unfolding them is the ordinary
:func:`pulserver.recon.pics` solve against an operator that sums the bands
(:class:`pulserver.recon.physics.SMS`), each modulated by the CAIPI phase the
gz blips played. Nothing is undersampled in plane, so the sensitivities are the
only thing telling the bands apart, and the separation is correspondingly
sensitive to how well they were estimated.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "Epi2DRecon"]

from typing import Any

import numpy as np

from pulserver import ReconContext, ReconPlugin, ReconResult
from pulserver.recon import (
    SMS,
    AcquisitionFlag,
    Cartesian2D,
    cartesian_recon,
    center_crop,
    coil_compress,
    coil_maps_from_reference,
    correct_lines,
    estimate_epi_phase,
    has_acquisition_flag,
    noise_prewhiten,
    pics,
)

#: Where the coil basis the prescan established is left for the imaging that
#: follows it, which may arrive as a stream of its own.
_BASIS = "epi2D_coil_basis"


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
    partial_fourier
        Which estimator fills a truncated readout, ``"pocs"`` or ``"homodyne"``.
    virtual_coils
        Channels to compress the array onto. A scan with fewer physical
        channels keeps them all.
    phase_order
        Order of the odd/even phase fitted from the navigator. One is the
        gradient-delay ramp every product reconstruction corrects; raising it
        picks up what eddy currents leave beyond that.
    device
        Torch device the reconstruction runs on. ``None`` is the CPU.
    """

    def __init__(
        self,
        *,
        regularization: float = 1e-3,
        iterations: int = 40,
        pocs_iterations: int = 12,
        partial_fourier: str = "pocs",
        virtual_coils: int = 8,
        phase_order: int = 1,
        device: Any = None,
    ) -> None:
        super().__init__(split_on=AcquisitionFlag.LAST_IN_MEASUREMENT)
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.pocs_iterations = int(pocs_iterations)
        self.partial_fourier = partial_fourier
        self.virtual_coils = int(virtual_coils)
        self.phase_order = int(phase_order)
        self.device = device

    def startup(self, context: ReconContext) -> None:
        """Lay the scan's buffers out, and start with no maps and no fit."""
        super().startup(context)
        self.coil_maps: dict[int, Any] = {}
        self.navigator: list[Any] = []
        self.noise: Any = None
        self.phase: Any = None

    def receive(self, acquisition: Any, context: ReconContext) -> Any:
        """Whiten, correct and compress the line, place it, and route what it closed.

        The navigator never reaches a buffer: its three blip-nulled lines are a
        measurement of the readout, not of the object, and what they produce is
        the fit every reversed line that follows is corrected by.
        """
        line = np.asarray(acquisition.data)
        if has_acquisition_flag(acquisition, AcquisitionFlag.IS_NOISE_MEASUREMENT):
            self.noise = (
                line if self.noise is None else np.concatenate([self.noise, line], -1)
            )
            return None
        if self.noise is not None:
            line = noise_prewhiten(line, self.noise, coil_axis=0)

        backwards = has_acquisition_flag(acquisition, AcquisitionFlag.IS_REVERSE)
        if has_acquisition_flag(acquisition, AcquisitionFlag.IS_PHASECORR_DATA):
            self.navigator.append(line[..., :: -1 if backwards else 1])
            if len(self.navigator) == 3:
                self.phase = estimate_epi_phase(
                    self.navigator, polynomial_order=self.phase_order
                )
                self.navigator = []
            return None

        (line,) = correct_lines([(line, backwards)], self.phase)
        if has_acquisition_flag(acquisition, AcquisitionFlag.IS_PARALLEL_CALIBRATION):
            # The prescan fills its own space at full channel count: it is what
            # the basis is estimated from, so it cannot already be in it.
            self.buffers.add(acquisition, line)
            if has_acquisition_flag(acquisition, AcquisitionFlag.LAST_IN_SLICE):
                return self.recon("calibration", context)
            return None

        basis = context.exam.get(_BASIS)
        self.buffers.add(acquisition, line if basis is None else basis @ line)
        if has_acquisition_flag(acquisition, AcquisitionFlag.LAST_IN_MEASUREMENT):
            return self.recon("imaging", context)
        return None

    def recon(self, branch: str, context: ReconContext) -> list[ReconResult] | None:
        """Calibrate the prescan's slices, or reconstruct the time series."""
        calibration = self.buffers[1] if len(self.buffers.spaces) > 1 else None
        n_calibrated = 0 if calibration is None else calibration.extents.get("slice", 1)

        if branch == "calibration":
            for index in range(n_calibrated):
                kspace, mask = calibration.select(slice=index)
                if index in self.coil_maps or not mask.any():
                    continue
                # The first slice to close establishes the basis, so every
                # slice of the prescan and all the imaging share one array.
                basis = context.exam.get(_BASIS)
                if basis is None:
                    lines = kspace[:, mask.any(axis=-1)].reshape(kspace.shape[0], -1)
                    _, basis = coil_compress(lines, self.virtual_coils)
                    context.exam.set(_BASIS, basis)
                self.coil_maps[index] = coil_maps_from_reference(
                    np.einsum("vc,c...->v...", basis, kspace)[None]
                )
            return None

        buffer = self.buffers[0]
        n_groups = buffer.extents.get("slice", 1)
        n_repetitions = buffer.extents.get("repetition", 1)
        n_sets = buffer.extents.get("set", 1)

        # The multiband imaging excites combs, so it carries fewer slice labels
        # than the prescan, which visits every slice on its own. A plain
        # accelerated scan images and calibrates the same slices, so the two
        # counts match and there is nothing to unfold.
        n_bands = max(n_calibrated // n_groups, 1)

        # The CAIPI slice phase the gz blips played: band j shifted j / n_bands
        # of the FOV, a linear ramp along ky. The trailing unit axis lands the
        # phase on the phase-encode axis of the (coil, ky, kx) measurement.
        ky = np.arange(buffer.extents["phase_encode"])
        caipi = np.exp(
            1j * 2 * np.pi * (np.arange(n_bands)[:, None] / n_bands) * ky[None, :]
        )[..., None].astype(np.complex64)

        results: list[ReconResult] = []
        for set_index in range(n_sets):
            for repetition in range(n_repetitions):
                for group in range(n_groups):
                    kspace, mask = buffer.select(
                        slice=group, repetition=repetition, set=set_index
                    )
                    if not mask.any():
                        continue
                    series = set_index * 1000 + repetition

                    if n_bands > 1:
                        # A group's bands are its slice and every n_groups-th
                        # slice above it, matching the comb the sequence excited.
                        bands = [group + band * n_groups for band in range(n_bands)]
                        images = pics(
                            kspace[None],
                            SMS(
                                [
                                    Cartesian2D(
                                        mask[None],
                                        self.coil_maps[index],
                                        device=self.device,
                                    )
                                    for index in bands
                                ],
                                caipi,
                            ),
                            regularization=self.regularization,
                            iterations=self.iterations,
                        )[0]
                        for band, index in enumerate(bands):
                            results.append(
                                ReconResult(
                                    center_crop(
                                        np.abs(images[band]), buffer.image_shape
                                    ).transpose(),
                                    reference=-1,
                                    series_index=series,
                                    image_index=index,
                                    image_type="magnitude",
                                    dicom=True,
                                )
                            )
                        continue

                    image = cartesian_recon(
                        kspace,
                        mask,
                        self.coil_maps.get(group),
                        regularization=self.regularization,
                        iterations=self.iterations,
                        pocs_iterations=self.pocs_iterations,
                        partial_fourier=self.partial_fourier,
                        device=self.device,
                    )
                    results.append(
                        ReconResult(
                            center_crop(np.abs(image), buffer.image_shape).transpose(),
                            reference=-1,
                            series_index=series,
                            image_index=group,
                            image_type="magnitude",
                            dicom=True,
                        )
                    )
        return results


PLUGIN = Epi2DRecon()
