"""Reconstruction for :mod:`pulserver.app.sequence.epi2D_sequence`.

The preprocessing EPI cannot skip, then the Cartesian pipeline. A blip-nulled
navigator triplet gives the odd/even linear phase fit, and every reversed line
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

When the sequence ran multiband (``SMS_EXCITATION``), that same prescan visits
every slice while the imaging excites combs, so the calibration carries more
slices than the imaging does -- which is how the two are told apart here. Each
group is then unfolded back into its bands by a model-based solve
(:class:`pulserver.recon.physics.SMS`) against the CAIPI phase the gz blips
played.
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
    coil_maps_from_reference,
    correct_lines,
    has_acquisition_flag,
    odd_even_fit,
    pics,
    recon_shape,
)


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
        """Lay the scan's buffers out and note the matrix to crop to."""
        super().startup(context)
        self.image_shape = recon_shape(context.header)
        self.coil_maps: dict[int, Any] = {}
        self.navigator: list[Any] = []
        self.fit = (0.0, 0.0)

    def receive(self, acquisition: Any, context: ReconContext) -> Any:
        """Correct the line, place it, and route the boundary it closed.

        The navigator never reaches a buffer: its three blip-nulled lines are a
        measurement of the readout, not of the object, and what they produce is
        the fit every reversed line that follows is corrected by.
        """
        backwards = has_acquisition_flag(acquisition, AcquisitionFlag.IS_REVERSE)
        line = np.asarray(acquisition.data)
        if has_acquisition_flag(acquisition, AcquisitionFlag.IS_PHASECORR_DATA):
            self.navigator.append(line[..., :: -1 if backwards else 1])
            if len(self.navigator) == 3:
                self.fit = odd_even_fit(self.navigator)
                self.navigator = []
            return None

        (corrected,) = correct_lines([(line, backwards)], *self.fit)
        self.buffers.add(acquisition, corrected)

        if has_acquisition_flag(acquisition, AcquisitionFlag.IS_PARALLEL_CALIBRATION):
            if has_acquisition_flag(acquisition, AcquisitionFlag.LAST_IN_SLICE):
                return self.recon("calibration", context)
            return None
        if has_acquisition_flag(acquisition, AcquisitionFlag.LAST_IN_MEASUREMENT):
            return self.recon("imaging", context)
        return None

    def recon(self, branch: str, context: ReconContext) -> list[ReconResult] | None:
        """Calibrate the prescan's slices, or reconstruct the time series."""
        del context
        calibration = self.buffers[1] if len(self.buffers.spaces) > 1 else None
        n_calibrated = (
            0
            if calibration is None
            else dict(zip(calibration.axes, calibration.kspace.shape, strict=True)).get(
                "slice", 1
            )
        )

        if branch == "calibration":
            for index in range(n_calibrated):
                kspace, mask = calibration.select(slice=index)
                if index not in self.coil_maps and mask.any():
                    self.coil_maps[index] = coil_maps_from_reference(kspace[None])
            return None

        buffer = self.buffers[0]
        extents = dict(zip(buffer.axes, buffer.kspace.shape, strict=True))
        n_groups = extents.get("slice", 1)
        n_repetitions = extents.get("repetition", 1)
        n_sets = extents.get("set", 1)

        # The multiband imaging excites combs, so it carries fewer slice labels
        # than the prescan, which visits every slice on its own. A plain
        # accelerated scan images and calibrates the same slices, so the two
        # counts match and there is nothing to unfold.
        n_bands = max(n_calibrated // n_groups, 1)

        # The CAIPI slice phase the gz blips played: band j shifted j / n_bands
        # of the FOV, a linear ramp along ky. The trailing unit axis lands the
        # phase on the phase-encode axis of the (coil, ky, kx) measurement.
        ky = np.arange(extents["phase_encode"])
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
                                        np.abs(images[band]), self.image_shape
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
                        device=self.device,
                    )
                    results.append(
                        ReconResult(
                            center_crop(np.abs(image), self.image_shape).transpose(),
                            reference=-1,
                            series_index=series,
                            image_index=group,
                            image_type="magnitude",
                            dicom=True,
                        )
                    )
        return results


PLUGIN = Epi2DRecon()
