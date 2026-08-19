"""Reconstruction for :mod:`pulserver.app.sequence.epi3D_sequence`.

The 2D EPI preprocessing on a volume. A blip-nulled navigator triplet gives the
odd/even phase fit, every reversed line is flipped and corrected by it as
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

A noise scan, when the scanner sends one, is not imaging data and never reaches
a buffer: it whitens every readout that follows. The prescan is also where the
array's principal channels are read, and the basis goes into ``context.exam``
-- the prescan is its own sequence, so it may be its own stream, and the exam
cache is what carries an artifact from one to the next. Every imaging readout
is compressed onto that basis as it arrives, so the imaging buffer is allocated
at the virtual channel count and never holds the full array.

The readout is ramp-sampled -- an EPI train that waits for the plateau throws
away the time its ramps take -- so k does not advance at a constant rate across
a readout and the samples are not on the grid. ``receive`` resamples them onto
it, exactly: a readout is the transform of an object of known width, so where
its samples fell is a change of basis away from where they belong.

Where they fell is what the acquisition's trajectory says, which is what a
client attaches once it notices the gradient is still moving under the ADC --
normalised here onto the readout's own extent, so the units it was written in
do not matter, which holds for a readout that sweeps the prescribed width.
Failing that, the lobe timing the header declares (``rampUpTime``,
``flatTopTime``, ``rampDownTime``, ``acqDelayTime``, microseconds) says the
same thing. With neither, the readout is taken to be uniform already, which is
what a train that waits for its plateau is.
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
    coil_compress,
    coil_maps_from_reference,
    correct_lines,
    epi_ramp_operator,
    epi_ramp_positions,
    estimate_epi_phase,
    has_acquisition_flag,
    noise_prewhiten,
    user_parameter,
)

#: Where the coil basis the prescan established is left for the imaging that
#: follows it, which may arrive as a stream of its own.
_BASIS = "epi3D_coil_basis"

#: The read lobe's timing, as the header declares it, in microseconds.
_LOBE = ("rampUpTime", "flatTopTime", "rampDownTime", "acqDelayTime")


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
        self.coil_maps: Any = None
        self.navigator: list[Any] = []
        self.noise: Any = None
        self.phase: Any = None
        self.regrid: Any = None
        timing = [user_parameter(context.header, key) for key in _LOBE]
        self.lobe = (
            None
            if any(value is None for value in timing)
            else dict(
                zip(
                    ("ramp_up", "flat_top", "ramp_down", "delay"),
                    [float(value) * 1e-6 for value in timing],
                    strict=True,
                )
            )
        )

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
        # Where the samples fell: the trajectory the scanner attached is the
        # direct answer, and the read lobe's timing derives the same thing when
        # it did not. With neither, the readout is already on the grid.
        trajectory = getattr(acquisition, "traj", None)
        samples = line.shape[-1]
        if trajectory is not None and np.size(trajectory) >= samples:
            taken = np.asarray(trajectory).reshape(samples, -1)[:, 0]
            # Onto the readout's own extent, whatever units it was written in:
            # the resampling is against a pixel grid, so what matters is that a
            # full sweep spans one k width.
            taken = taken / (2.0 * np.abs(taken).max())
        elif self.lobe is not None:
            taken = epi_ramp_positions(
                samples, float(acquisition.sample_time_us) * 1e-6, **self.lobe
            )[0]
        else:
            taken = None

        if taken is not None:
            # One lobe is played for every readout of the train, so the change
            # of basis onto the grid is built once and applied to each.
            if self.regrid is None or self.regrid.shape[1] != taken.size:
                self.regrid = epi_ramp_operator(
                    taken,
                    np.linspace(taken[0], taken[-1], taken.size),
                    self.buffers[0].image_shape[-1],
                )
            line = line @ self.regrid.T

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
        """Calibrate the prescan's slab, or reconstruct the time series."""
        if branch == "calibration":
            kspace, mask = self.buffers[1].select()
            lines = kspace[:, mask.any(axis=-1)].reshape(kspace.shape[0], -1)
            _, basis = coil_compress(lines, self.virtual_coils)
            context.exam.set(_BASIS, basis)
            self.coil_maps = coil_maps_from_reference(
                np.einsum("vc,c...->v...", basis, kspace)[None], mask, spatial_ndim=3
            )
            return None

        del context
        buffer = self.buffers[0]
        n_repetitions = buffer.extents.get("repetition", 1)
        n_sets = buffer.extents.get("set", 1)

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
                    partial_fourier=self.partial_fourier,
                    device=self.device,
                )
                results.append(
                    ReconResult(
                        center_crop(np.abs(image), buffer.image_shape).transpose(
                            0, 2, 1
                        ),
                        reference=-1,
                        series_index=set_index * 1000 + repetition,
                        image_type="magnitude",
                        dicom=True,
                    )
                )
        return results


PLUGIN = Epi3DRecon()
