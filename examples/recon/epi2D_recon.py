"""2D echo-planar imaging, one image per slice and repetition.

Calling this module reconstructs an MRD file; ``PLUGIN`` is the same
reconstruction behind the stream contract, driven live by the scanner.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "Epi2DRecon"]

from typing import Any

import numpy as np

from pulserver.recon import ReconContext, ReconPlugin, ReconResult
from pulserver.recon import (
    Cartesian2D,
    CoilCompression,
    EpiPhaseCorrection,
    NoiseAdjust,
    RampSampling,
    SMS,
    cartesian_recon,
    coil_maps_from_reference,
    image_result,
    pics,
)
from pulserver.mrd import (
    AcquisitionFlag,
    has_acquisition_flag,
)

#: Where the coil basis the prescan established is left for the imaging that
#: follows it, which may arrive as a stream of its own.
_BASIS = "epi2D_coil_basis"


class Epi2DRecon(ReconPlugin):
    """Reconstruct a 2D EPI time series, one image per slice and repetition.

    The preprocessing EPI cannot skip, then the Cartesian pipeline. A blip-nulled
    navigator triplet gives the odd/even phase fit, and every reversed line
    is flipped and corrected by it as it arrives -- before it is placed, because a
    corrected line is what belongs on the grid. The volume then reconstructs
    exactly as :mod:`pulserver.app.cartesian2D_recon` reconstructs it.

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

    The readout is ramp-sampled -- an EPI train that waits for the plateau throws
    away the time its ramps take -- so k does not advance at a constant rate across
    a readout and the samples are not on the grid. ``receive`` resamples them onto
    it, exactly: a readout is the transform of an object of known width, so where
    its samples fell is a change of basis away from where they belong.

    Where they fell is what the acquisition's trajectory says, and only that: the
    client attaches one as soon as it notices the gradient is still moving under
    the ADC, and it is attached per readout, so it describes the lobe that was
    actually played rather than the one a header was told about. It is normalised
    here onto the readout's own extent, so the units it was written in do not
    matter -- which holds for a readout that sweeps the prescribed width. An
    acquisition carrying none was sampled uniformly, which is what a train that
    waits for its plateau is.

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
        Torch device the reconstruction runs on. ``"auto"`` is the host's
        GPU when it has one, and the CPU when it does not.

    Examples
    --------
    Calling the module reconstructs an MRD file, through the same hooks a
    scanner's live stream is driven through. Its settings are the arguments
    of that call:

    >>> import inspect
    >>> from pulserver.app import epi2D_recon
    >>> "virtual_coils" in inspect.signature(epi2D_recon).parameters
    True

    Reconstruct a 2D echo-planar scan and its calibration prescan::

        images = epi2D_recon("scan.h5")
        images = epi2D_recon("scan.h5", virtual_coils=4, phase_order=2)

    A single shot with a one-sample gradient delay, which is what a reversed
    line comes back displaced by. Without the navigator the displacement stays
    on the alternate lines and lands as a ghost at half the field of view;
    with it, the fit removes it.

    .. plot::

       from pulserver.app import epi2D_recon
       from _figures import epi_example, images

       plugin = epi2D_recon.PLUGIN
       ghost = epi_example(plugin, size=48, coils=8, corrected=False)
       clean = epi_example(plugin, size=48, coils=8)

       images(
           [
               ("object", clean.truth),
               ("no navigator", ghost.image),
               ("navigator fitted", clean.image),
           ],
           title="the odd/even phase, and the fit that removes it",
       )
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
        device: Any = "auto",
    ) -> None:
        super().__init__(
            chain=[
                NoiseAdjust(),
                RampSampling(),
                EpiPhaseCorrection(order=phase_order),
                CoilCompression(virtual_coils, key=_BASIS),
            ],
            branches={AcquisitionFlag.LAST_IN_MEASUREMENT: "imaging"},
        )
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.pocs_iterations = int(pocs_iterations)
        self.partial_fourier = partial_fourier
        self.virtual_coils = int(virtual_coils)
        self.phase_order = int(phase_order)
        self.device = device

    def startup(self, context: ReconContext) -> None:
        """Lay the scan's buffers out, and start with no maps."""
        super().startup(context)
        self.coil_maps: dict[int, Any] = {}

    def receive(self, acquisition: Any, context: ReconContext) -> Any:
        """Correct the line, place it, and route what it closed.

        The one thing the declaration cannot say: a prescan slice and an
        imaging slice both close ``LAST_IN_SLICE``, and only the prescan's
        calibrates. So the branch is chosen from the two flags together.
        """
        data = self.process(acquisition)
        if data is None:
            return None
        self.acquisition = acquisition
        self.buffers.add(acquisition, data)
        if has_acquisition_flag(acquisition, AcquisitionFlag.IS_PARALLEL_CALIBRATION):
            if has_acquisition_flag(acquisition, AcquisitionFlag.LAST_IN_SLICE):
                return self.recon("calibration", context)
            return None
        branch = self.branch_for(acquisition)
        return None if branch is None else self.recon(branch, context)

    def recon(self, branch: str, context: ReconContext) -> list[ReconResult] | None:
        """Calibrate the prescan's slices, or reconstruct the time series."""
        del context
        calibration = self.buffers[1] if len(self.buffers.spaces) > 1 else None
        n_calibrated = 0 if calibration is None else calibration.extents.get("slice", 1)

        if branch == "calibration":
            compression = self.gadget(CoilCompression)
            for index in range(n_calibrated):
                kspace, mask = calibration.select(slice=index)
                if index in self.coil_maps or not mask.any():
                    continue
                # The first slice to close establishes the basis, so every
                # slice of the prescan and all the imaging share one array.
                basis = compression.basis
                if basis is None:
                    basis = compression.learn(kspace, mask)
                self.coil_maps[index] = coil_maps_from_reference(
                    np.einsum("vc,c...->v...", basis, kspace)[None], mask
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
                                image_result(
                                    images[band],
                                    buffer,
                                    series_index=series,
                                    image_index=index,
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
                        image_result(
                            image, buffer, series_index=series, image_index=group
                        )
                    )
        return results


PLUGIN = Epi2DRecon()
