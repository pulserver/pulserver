"""Reconstruction for :mod:`pulserver.app.sequence.epi3D_sequence`.

The 2D EPI preprocessing on a volume: the stream is partitioned, the
navigator's odd/even fit corrects every reversed line, and the volume grids
over ``(partition, line, readout)`` before a coil-wise 3D Fourier
reconstruction. The opposite-polarity reference reconstructs alongside as
its own series, the pair PyHySCO corrects.

"""

from __future__ import annotations

__all__ = ["PLUGIN", "Epi3DRecon"]

from typing import Any

import numpy as np

from pulserver import AcquisitionBucket, ReconContext, ReconPlugin, ReconResult
from pulserver.recon import (
    AcquisitionFlag,
    CartesianGridder,
    center_crop,
    coil_combine,
    coil_images,
    correct_lines,
    encoded_volume,
    fill_partial_echo,
    has_acquisition_flag,
    odd_even_fit,
    partition_epi_acquisitions,
    receiver_channels,
    recon_volume,
    sense,
    sensitivities,
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
        """Size the volume from the header and collect the stream."""
        self.grid = encoded_volume(context.header)
        self.coils = receiver_channels(context.header)
        self.image_shape = recon_volume(context.header)
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

        navigator = [
            np.asarray(item.data)[
                ...,
                :: (
                    -1 if has_acquisition_flag(item, AcquisitionFlag.IS_REVERSE) else 1
                ),
            ]
            for item in groups.phase_correction[:3]
        ]
        slope, intercept = (
            odd_even_fit(navigator) if len(navigator) == 3 else (0.0, 0.0)
        )

        # Coil sensitivities come from the separate low-resolution calibration
        # (ACQ_IS_PARALLEL_CALIBRATION), estimated once and reused for every
        # frame; NLINV reads the fully sampled centre off its mask and resamples
        # to the full matrix. Absent it, an undersampled scan cannot be solved.
        coil_maps = None
        if groups.single_band_reference:
            calibration = CartesianGridder(self.grid, coils=self.coils)
            for item in groups.single_band_reference:
                calibration.add(
                    np.asarray(item.data),
                    int(item.idx.kspace_encode_step_2),
                    int(item.idx.kspace_encode_step_1),
                )
            coil_maps = sensitivities(
                calibration.kspace, calibration.mask, device=self.device
            )

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
                    (row,) = correct_lines(
                        [
                            (
                                np.asarray(item.data),
                                has_acquisition_flag(item, AcquisitionFlag.IS_REVERSE),
                            )
                        ],
                        slope,
                        intercept,
                    )
                    buffer.add(
                        row,
                        int(item.idx.kspace_encode_step_2),
                        int(item.idx.kspace_encode_step_1),
                    )
                kspace, mask = buffer.kspace, buffer.mask

                # What the scan sampled selects the reconstruction: a phase
                # encode with no samples was skipped, and a readout sample
                # missing from every line is echo never acquired.
                encodes = mask.any(axis=-1)
                readout = mask.reshape(-1, mask.shape[-1]).any(axis=0)

                if encodes.all():
                    coils = (
                        coil_images(kspace, mask, device=self.device)
                        if readout.all()
                        else fill_partial_echo(
                            kspace,
                            readout,
                            self.pocs_iterations,
                            dimension=3,
                            device=self.device,
                        )
                    )
                    image = coil_combine(coils, coil_axis=0)
                else:
                    maps = (
                        coil_maps
                        if coil_maps is not None
                        else sensitivities(kspace, mask, device=self.device)
                    )
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
                        center_crop(np.abs(image), self.image_shape).transpose(0, 2, 1),
                        reference=-1,
                        series_index=series * 1000 + repetition,
                        image_type="magnitude",
                        dicom=True,
                    )
                )
        return results


PLUGIN = Epi3DRecon()
