"""3D echo-planar imaging, one volume per repetition.

Calling this module reconstructs an MRD file; ``PLUGIN`` is the same
reconstruction behind the stream contract, driven live by the scanner.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "Epi3DRecon"]

from typing import Any

import numpy as np

from pulserver.recon import ReconContext, ReconPlugin, ReconResult
from pulserver.recon import (
    CoilCompression,
    EpiPhaseCorrection,
    NoiseAdjust,
    RampSampling,
    cartesian_recon,
    coil_maps_from_reference,
    image_result,
)
from pulserver.mrd import (
    AcquisitionFlag,
    has_acquisition_flag,
)

#: Where the coil basis the prescan established is left for the imaging that
#: follows it, which may arrive as a stream of its own.
_BASIS = "epi3D_coil_basis"


class Epi3DRecon(ReconPlugin):
    """Reconstruct a 3D EPI time series, one volume per repetition.

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
    >>> from pulserver.app import epi3D_recon
    >>> "virtual_coils" in inspect.signature(epi3D_recon).parameters
    True

    Reconstruct a 3D echo-planar scan and its calibration prescan::

        images = epi3D_recon("scan.h5")
        images = epi3D_recon("scan.h5", virtual_coils=4, phase_order=2)

    A single shot with a one-sample gradient delay, which is what a reversed
    line comes back displaced by. Without the navigator the displacement stays
    on the alternate lines and lands as a ghost at half the field of view;
    with it, the fit removes it.

    .. plot::

       from pulserver.app import epi3D_recon
       from _figures import epi_example, images

       plugin = epi3D_recon.PLUGIN
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
        self.coil_maps: Any = None

    def receive(self, acquisition: Any, context: ReconContext) -> Any:
        """Correct the line, place it, and route what it closed.

        The one thing the declaration cannot say: a prescan slab and the
        imaging both close their own boundaries, and only the prescan's
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
        """Calibrate the prescan's slab, or reconstruct the time series."""
        if branch == "calibration":
            kspace, mask = self.buffers[1].select()
            basis = self.gadget(CoilCompression).learn(kspace, mask)
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
                    image_result(
                        image, buffer, series_index=set_index * 1000 + repetition
                    )
                )
        return results


PLUGIN = Epi3DRecon()
