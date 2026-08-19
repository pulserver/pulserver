"""Reconstruction for :mod:`pulserver.app.sequence.epi3D_sequence`.

The 2D EPI preprocessing on a volume. A blip-nulled navigator triplet gives the
odd/even linear phase fit, every reversed line is flipped and corrected by it as
it arrives, and the slab fills over ``(partition, line, readout)`` before
:func:`pulserver.recon.cartesian_recon` inverts it -- the same three
reconstructions the Cartesian plugins choose between, selected by what the scan
sampled.

The opposite-polarity reference is the scan's second ``SET``, so it is an axis
of the same buffer and comes back as its own series: the pair PyHySCO corrects
leaves the scanner together.

Coil sensitivities come from a separate low-resolution calibration
(``ACQ_IS_PARALLEL_CALIBRATION``). That prescan is a subsequence, so the header
gives it an encoding space of its own, its lines never touch the imaging grid,
and it calibrates once for the whole time series, through
:func:`pulserver.recon.coil_maps_from_reference`.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "Epi3DRecon"]

from typing import Any

import numpy as np

from pulserver import ReconContext, ReconPlugin, ReconResult
from pulserver.recon import (
    AcquisitionFlag,
    cartesian_recon,
    center_crop,
    coil_maps_from_reference,
    correct_lines,
    has_acquisition_flag,
    odd_even_fit,
    recon_volume,
)


class Epi3DRecon(ReconPlugin):
    """Reconstruct a 3D EPI time series, one volume per repetition.

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
        self.image_shape = recon_volume(context.header)
        self.coil_maps: Any = None
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
        """Calibrate the prescan's slab, or reconstruct the time series."""
        del context
        if branch == "calibration":
            kspace, _ = self.buffers[1].select()
            self.coil_maps = coil_maps_from_reference(kspace[None], spatial_ndim=3)
            return None

        buffer = self.buffers[0]
        extents = dict(zip(buffer.axes, buffer.kspace.shape, strict=True))
        n_repetitions = extents.get("repetition", 1)
        n_sets = extents.get("set", 1)

        results = []
        for set_index in range(n_sets):
            for repetition in range(n_repetitions):
                kspace, mask = buffer.select(repetition=repetition, set=set_index)
                if not mask.any():
                    continue
                image = cartesian_recon(
                    kspace,
                    mask,
                    self.coil_maps,
                    regularization=self.regularization,
                    iterations=self.iterations,
                    pocs_iterations=self.pocs_iterations,
                    device=self.device,
                )
                results.append(
                    ReconResult(
                        center_crop(np.abs(image), self.image_shape).transpose(0, 2, 1),
                        reference=-1,
                        series_index=set_index * 1000 + repetition,
                        image_type="magnitude",
                        dicom=True,
                    )
                )
        return results


PLUGIN = Epi3DRecon()
