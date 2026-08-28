"""Any 3D wave-encoded Cartesian scan, one volume per echo.

Calling this module reconstructs an MRD file; ``PLUGIN`` is the same
reconstruction behind the stream contract, driven live by the scanner.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "Wave3DRecon"]

from typing import Any

import numpy as np

from pulserver.recon import ReconContext, ReconPlugin, ReconResult
from pulserver.recon import (
    NLINV,
    NoiseAdjust,
    WaveEncoding,
    WavePSF,
    image_result,
    pics,
)
from pulserver.mrd import (
    AcquisitionFlag,
    coil_compress,
    has_acquisition_flag,
)


class Wave3DRecon(ReconPlugin):
    """Reconstruct a 3D wave-encoded Cartesian scan, one volume per echo.

    A wave-encoded scan plays a corkscrew during the readout, so each voxel is
    smeared along it by an amount that grows with how far the voxel sits from
    the centre of the two encoded axes. The aliasing a parallel-imaging solve
    has to separate is spread the same way, which is what lets the same coils
    resolve a higher acceleration than they could on the Cartesian grid.

    Not an option on :mod:`pulserver.app.cartesian3D_recon` but a plugin of its
    own, because all three of the things a reconstruction is made of differ:
    the encoding operator carries the corkscrew, the sensitivities come from a
    pass acquired without it, and the sampling is a list of acquired lines
    rather than a mask over a grid.

    **Where the corkscrew comes from.** Nothing about the gradient is declared.
    With the wave on, ky and kz move *within* the readout, and that movement is
    the phase: it is read off one acquired readout's own trajectory by
    :meth:`pulserver.recon.WavePSF.phase_from_trajectory`, referred to the
    start of the readout, where a self-balanced wave has yet to accrue
    anything. So the waveform's exact shape is free -- ramped, windowed,
    slew-limited to whatever the system allowed on the day -- and the
    reconstruction undoes what was played rather than what was prescribed.

    **Where the sensitivities come from.** The autocalibration rectangle is
    acquired a second time with the wave scaled to zero, flagged calibration
    and nothing else. Those lines are Cartesian, so the maps are NLINV's over
    the rectangle exactly as
    :mod:`pulserver.app.cartesian3D_recon` estimates them -- and the imaging
    train that follows overwrites the same buffer positions with its
    wave-encoded lines, which is why the calibration is routed off the
    calibration flag rather than off a segment boundary: it has to run while
    the rectangle is still what the buffer holds.

    Echoes are an axis, not a variant: each is solved against the same
    sensitivities and the same corkscrew, and a single-echo scan is that loop
    run once.

    One magnitude volume per echo, the echo as the image series, cropped to the
    prescribed matrix.

    Parameters
    ----------
    regularization
        Tikhonov weight of the CG solve.
    iterations
        Maximum CG iterations.
    virtual_coils
        Channels to compress the array onto before the solve. A scan with fewer
        physical channels keeps them all.
    calibration_iterations
        Newton steps the sensitivity solve takes.
    coil_batch_size
        Channels the hybrid-space normal operator holds at once. The wave
        operator works on a grid the size of the volume times the readout, so
        this is the knob that trades reconstruction memory for speed.
    device
        Torch device the reconstruction runs on. ``"auto"`` is the host's GPU
        when it has one, and the CPU when it does not.

    Examples
    --------
    Calling the module reconstructs an MRD file, through the same hooks a
    scanner's live stream is driven through. Its settings are the arguments
    of that call:

    >>> import inspect
    >>> from pulserver.app import wave3D_recon
    >>> "virtual_coils" in inspect.signature(wave3D_recon).parameters
    True

    Reconstruct any 3D wave-encoded scan::

        images = wave3D_recon("scan.h5")
        images = wave3D_recon("scan.h5", virtual_coils=4, iterations=60)
    """

    def __init__(
        self,
        *,
        regularization: float = 1e-3,
        iterations: int = 40,
        virtual_coils: int = 8,
        calibration_iterations: int = 16,
        coil_batch_size: int = 1,
        device: Any = "auto",
    ) -> None:
        super().__init__(
            chain=[NoiseAdjust()],
            branches={AcquisitionFlag.LAST_IN_MEASUREMENT: "imaging"},
            reject_flags=AcquisitionFlag.IS_PHASECORR_DATA,
        )
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.virtual_coils = int(virtual_coils)
        self.calibration_iterations = int(calibration_iterations)
        self.coil_batch_size = int(coil_batch_size)
        self.device = device

    def startup(self, context: ReconContext) -> None:
        """Lay the scan's buffers out, and start with no maps."""
        super().startup(context)
        self.coil_maps: Any = None
        self.coil_basis: Any = None
        self.wave_lines: Any = None

    def receive(self, acquisition: Any, context: ReconContext) -> Any:
        """Place the line, and route what it closed.

        The one thing the declaration cannot say: the wave-free rectangle and
        the wave-encoded train are two acquisitions of one grid. So the
        calibration is routed off the calibration flag, which only the
        rectangle carries, rather than off the segment boundary the train
        closes too -- and which positions the corkscrew actually visited is
        recorded here, because a rectangle line the train never revisits is
        still in the buffer, and solving it against a corkscrew it never
        played puts a Cartesian readout in a wave operator.
        """
        data = self.process(acquisition)
        if data is None:
            return None
        self.buffers.add(acquisition, data)
        if has_acquisition_flag(acquisition, AcquisitionFlag.IS_PARALLEL_CALIBRATION):
            if has_acquisition_flag(acquisition, AcquisitionFlag.LAST_IN_SEGMENT):
                return self.recon("calibration", context)
            return None
        buffer = self.buffers[0]
        if self.wave_lines is None:
            self.wave_lines = np.zeros(buffer.mask.shape[:-1], dtype=bool)
        self.wave_lines[buffer.position(acquisition)] = True
        branch = self.branch_for(acquisition)
        return None if branch is None else self.recon(branch, context)

    def recon(self, branch: str, context: ReconContext) -> list[ReconResult] | None:
        """Calibrate off the wave-free rectangle, or solve every echo."""
        buffer = self.buffers[0]

        if branch == "calibration":
            # The rectangle, from the first echo: it has the most signal, and
            # every echo is unaliased against the same maps.
            kspace, mask = buffer.select(contrast=0)
            lines = kspace[:, mask.any(axis=-1)].reshape(kspace.shape[0], -1)
            _, self.coil_basis = coil_compress(lines, self.virtual_coils)
            self.coil_maps = NLINV(
                spatial_ndim=3, max_iter=self.calibration_iterations
            )(
                np.einsum("vc,c...->v...", self.coil_basis, kspace)[None],
                mask=mask,
                device=self.device,
            )
            return None

        if self.coil_maps is None:
            raise RuntimeError(
                "a wave-encoded scan is solved against sensitivities, and none "
                "were estimated: the scan sent no line flagged parallel "
                "calibration, so the wave-free rectangle never arrived"
            )

        points = buffer.points(contrast=0)
        if points is None:
            raise RuntimeError(
                "the wave point-spread function is read off the trajectory, "
                "and these acquisitions carry none"
            )
        placement = buffer.axes[1:-1]
        n_partition, n_phase = buffer.select(contrast=0)[1].shape[:2]

        # Voxel coordinates along the two encoded axes, in metres, over the
        # encoded field of view -- the grid the sensitivities are on.
        space = context.header.encoding[buffer.space.index].encodedSpace
        axes = [
            (np.arange(size) - size // 2) * (float(extent) * 1e-3 / size)
            for size, extent in (
                (n_phase, space.fieldOfView_mm.y),
                (n_partition, space.fieldOfView_mm.z),
            )
        ]

        # The corkscrew, off one readout: ky and kz are the only things that
        # move within a readout, and where they start is that line's own phase
        # encode. Every readout played the same one, so one is all it takes.
        # An axis a readout does not traverse is left off the trajectory it
        # carries, and is the zero it was.
        deviation = np.zeros((2, buffer.readout), dtype=np.float32)
        first = np.argwhere(
            self.wave_lines[
                tuple(0 if name == "contrast" else slice(None) for name in placement)
            ]
        )[0]
        traversed = points[1:3, first[0], first[1]]
        deviation[: traversed.shape[0]] = traversed
        psf = WavePSF(*axes)(WavePSF.phase_from_trajectory(deviation))

        # (coils, partition, phase, readout) as the buffer had them, laid out
        # (readout, phase, partition) for the wave operator and put back to be
        # cropped.
        maps = np.asarray(self.coil_maps)
        maps = maps.reshape(-1, n_partition, n_phase, buffer.readout)
        maps = maps.transpose(0, 3, 2, 1)

        results = []
        for echo in range(buffer.extents.get("contrast", 1)):
            kspace = buffer.select(contrast=echo)[0]
            if self.coil_basis is not None:
                kspace = np.einsum("vc,c...->v...", self.coil_basis, kspace)
            acquired = np.argwhere(
                self.wave_lines[
                    tuple(
                        echo if name == "contrast" else slice(None)
                        for name in placement
                    )
                ]
            )
            physics = WaveEncoding(
                np.stack([acquired[:, 1], acquired[:, 0]], axis=-1),
                maps,
                psf,
                coil_batch_size=self.coil_batch_size,
            )
            image = pics(
                kspace[:, acquired[:, 0], acquired[:, 1]][None],
                physics,
                regularization=self.regularization,
                iterations=self.iterations,
            )
            volume = np.asarray(image).reshape(buffer.readout, n_phase, n_partition)
            results.append(
                image_result(volume.transpose(2, 1, 0), buffer, series_index=echo)
            )
        return results


PLUGIN = Wave3DRecon()
