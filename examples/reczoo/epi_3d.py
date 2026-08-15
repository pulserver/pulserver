"""Reconstruction for :mod:`pulserver.seqzoo.epi_3d`.

The 2D EPI preprocessing on a volume: the stream is partitioned, the
navigator's odd/even fit corrects every reversed line, and the volume grids
over ``(partition, line, readout)`` before a coil-wise 3D Fourier
reconstruction. The opposite-polarity reference reconstructs alongside as
its own series, the pair PyHySCO corrects.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "Epi3DRecon"]

from typing import Any

import numpy as np

from pulserver import AcquisitionBucket, ReconContext, ReconPlugin, ReconResult
from pulserver.recon import has_acquisition_flag
from pulserver.recon._mrd.epi import partition_epi_acquisitions
from pulserver.recon.postprocessing import center_crop, coil_combine
from pulserver.recon.preprocessing import CartesianGridder, receiver_channels
from pulserver.reczoo.epi_2d import correct_lines, odd_even_fit
from pulserver.reczoo.gre_3d import coil_volumes, encoded_volume, recon_volume


class Epi3DRecon(ReconPlugin):
    """Reconstruct a 3D EPI time series, one volume per repetition.

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
        """Size the volume from the header and collect the stream."""
        self.grid = encoded_volume(context.header)
        self.coils = receiver_channels(context.header)
        self.image_shape = recon_volume(context.header)
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

        navigator = [
            np.asarray(item.data)[
                ..., :: (-1 if has_acquisition_flag(item, "ACQ_IS_REVERSE") else 1)
            ]
            for item in groups.phase_correction[:3]
        ]
        slope, intercept = odd_even_fit(navigator) if len(navigator) == 3 else (0.0, 0.0)

        results = []
        for series, group in enumerate(
            (groups.imaging, groups.reverse_polarity)
        ):
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
                                has_acquisition_flag(item, "ACQ_IS_REVERSE"),
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
                kspace = buffer.kspace
                image = coil_combine(
                    coil_volumes(kspace, np.ones(kspace.shape[1:], bool)),
                    coil_axis=0,
                )
                results.append(
                    ReconResult(
                        center_crop(
                            np.abs(image), self.image_shape
                        ).transpose(0, 2, 1),
                        reference=-1,
                        series_index=series * 1000 + repetition,
                        image_type="magnitude",
                        dicom=True,
                    )
                )
        return results


PLUGIN = Epi3DRecon()
