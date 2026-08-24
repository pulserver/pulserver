"""Any 2D non-Cartesian scan, one image per slice.

Calling this module reconstructs an MRD file; ``PLUGIN`` is the same
reconstruction behind the stream contract, driven live by the scanner.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "NonCartesian2DRecon"]

from typing import Any


from pulserver.recon import ReconContext, ReconPlugin, ReconResult
from pulserver.recon import (
    NoiseAdjust,
    image_result,
    noncartesian_recon,
)
from pulserver.mrd import (
    AcquisitionFlag,
    coil_compress,
)


class NonCartesian2DRecon(ReconPlugin):
    """Reconstruct a 2D non-Cartesian gradient echo, one image per slice.

    Radial, spiral, PROPELLER: what a reconstruction needs is the trajectory each
    acquisition carries, not which shape the sequence drew with it, so all of them
    come back through this.

    Model-based reconstruction, per slice: Pipe--Menon density compensation, NLINV
    sensitivities calibrated from the samples inside the calibration radius --
    selected before any gridding -- and a CG-SENSE solve against the
    :class:`pulserver.recon.physics.NonCartesian2D` operator built from the same
    trajectory. A fully sampled slice finishes with the density-compensated
    adjoint instead, coil images root-sum-of-squares combined.

    The trajectory arrives per acquisition, as MRD carries it, scaled to
    MRI-NUFFT's ``[-0.5, 0.5)`` units -- what the LiveSDK's enrichment writes.

    Parameters
    ----------
    mode
        ``"auto"`` finishes a fully sampled slice with the
        density-compensated adjoint and sends everything undersampled
        through CG-SENSE; ``"direct"`` and ``"pics"`` force one branch. A
        slice counts as fully sampled when its view count reaches the radial
        Nyquist count, ``ceil(pi/2 * matrix)``.
    regularization
        Tikhonov weight of the CG-SENSE solve.
    iterations
        Maximum CG iterations.
    virtual_coils
        Channels to compress the array onto before the solve. A scan with fewer
        physical channels keeps them all.
    calibration_width
        Width of the centred square NLINV solves the sensitivities over.
    device
        Torch device the reconstruction runs on. ``"auto"`` is the host's
        GPU when it has one, and the CPU when it does not.

    Examples
    --------
    Calling the module reconstructs an MRD file, through the same hooks a
    scanner's live stream is driven through. Its settings are the arguments
    of that call:

    >>> import inspect
    >>> from pulserver.app import noncartesian2D_recon
    >>> "virtual_coils" in inspect.signature(noncartesian2D_recon).parameters
    True

    Reconstruct a 2D scan whose acquisitions carry a trajectory::

        images = noncartesian2D_recon("scan.h5")
        images = noncartesian2D_recon("scan.h5", virtual_coils=4, mode="pics")

    Below, the spokes are the ones
    :func:`~pulserver.app.gre_radial2D_sequence` draws, read back out of the
    designed sequence: a scan at the radial Nyquist count, and one at a third
    of it.

    .. plot::

       from _figures import radial_spokes, sampling

       size = 48
       sampling(
           [
               ("76 spokes, Nyquist", radial_spokes(size, 76)),
               ("24 spokes", radial_spokes(size, 24)),
           ],
           title="where each scan sampled",
       )

    Which branch runs is read off the view count, so the same call answers
    both. The density-compensated adjoint inverts the Nyquist scan; below it
    the same adjoint is forced on the undersampled one, which is what the
    streaks are, and then the branch the plugin would actually have taken.

    .. plot::

       from pulserver.app import noncartesian2D_recon
       from _figures import images, noncartesian_example

       plugin = noncartesian2D_recon.PLUGIN
       size, coils = 48, 8
       full = noncartesian_example(plugin, size=size, coils=coils)
       streaky = noncartesian_example(
           noncartesian2D_recon.NonCartesian2DRecon(mode="direct"),
           size=size,
           coils=coils,
           spokes=24,
       )
       solved = noncartesian_example(plugin, size=size, coils=coils, spokes=24)

       images(
           [
               ("object", full.truth),
               ("76 spokes, adjoint", full.image),
               ("24 spokes, adjoint", streaky.image),
               ("24 spokes, CG-SENSE", solved.image),
           ],
           title="and what came back",
       )
    """

    def __init__(
        self,
        *,
        mode: str = "auto",
        regularization: float = 1e-3,
        iterations: int = 30,
        virtual_coils: int = 8,
        calibration_width: int = 24,
        device: Any = "auto",
    ) -> None:
        super().__init__(
            chain=[NoiseAdjust()],
            branches={AcquisitionFlag.LAST_IN_MEASUREMENT: "imaging"},
            reject_flags=AcquisitionFlag.IS_PHASECORR_DATA,
        )
        if mode not in ("auto", "direct", "pics"):
            raise ValueError(f"mode must be auto, direct or pics, got {mode!r}")
        self.mode = mode
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.virtual_coils = int(virtual_coils)
        self.calibration_width = int(calibration_width)
        self.device = device

    def startup(self, context: ReconContext) -> None:
        """Lay the scan's buffers out."""
        super().startup(context)

    def recon(self, branch: str, context: ReconContext) -> list[ReconResult]:
        """Reconstruct every slice, once the measurement is complete."""
        del branch, context
        buffer = self.buffers[0]
        n_views = buffer.extents["phase_encode"]

        results = []
        for index in range(buffer.extents.get("slice", 1)):
            # Views laid end to end, and the points they were taken at in the
            # same order: one non-Cartesian measurement of this slice.
            views, _ = buffer.select(slice=index)
            data, _ = coil_compress(
                views.reshape(views.shape[0], -1), self.virtual_coils
            )
            trajectory = (
                buffer.points(slice=index)[:2].transpose(1, 2, 0).reshape(-1, 2)
            )
            image = noncartesian_recon(
                data,
                trajectory,
                buffer.image_shape,
                mode=self.mode,
                n_views=n_views,
                regularization=self.regularization,
                iterations=self.iterations,
                calibration_width=self.calibration_width,
                device=self.device,
            )
            results.append(image_result(image, buffer, image_index=index))
        return results


PLUGIN = NonCartesian2DRecon()
