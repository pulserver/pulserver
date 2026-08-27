"""Prospective motion correction from a multi-plane navigator.

Calling this module tracks the head through a recorded navigator file;
``PLUGIN`` is the same tracking behind the stream contract, and ``process``
is what the scanner's real-time port drives it through while the scan runs.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "PmcRecon", "process"]

from typing import Any

import numpy as np

from pulserver.mrd import acquisition_label, coil_combine, pipe_menon_dcf
from pulserver.recon import (
    ExamCache,
    NavigatorMotionTracker,
    NonCartesian2D,
    PmcPayload,
    ReconContext,
    ReconPlugin,
    RigidRegistration,
)


class PmcRecon(ReconPlugin):
    """Track rigid head motion from a navigator and correct the scan for it.

    The scanner interrupts its own encoding every so often to play a
    navigator -- a handful of low-flip planes, cheap enough to fit in dead
    time -- and streams those readouts here while the scan is still running.
    Each navigator is turned into a pose, and the pose is returned to the
    sequence, which moves its gradients and its transmit frequency to follow
    the head. The imaging data is never touched: what this corrects is where
    the *next* readout is played, which is what makes it prospective.

    A navigator arrives as one readout per plane, in the same order every
    time. Each is reconstructed on its own -- the plane it lies in is read off
    the trajectory it carries, and its samples are gridded onto that plane by
    the density-compensated adjoint, root-sum-of-squares over the coils. A
    plane so cheaply acquired is a poor image and a perfectly good landmark,
    which is all a pose needs. The first complete navigator becomes the
    reference; every later one is registered against it and filtered by
    :class:`~pulserver.recon.NavigatorMotionTracker`.

    Nothing here is specific to three planes or to the orientations the
    sequence chose. Reading each plane off its own trajectory is what makes
    that true: a navigator of a different shape needs a different ``planes``
    and nothing else.

    What goes back on the wire is the *change* since the last correction, not
    the pose. The scanner accumulates what it receives -- it adds the shift to
    the one it is holding and left-multiplies its rotation -- so a pose sent
    twice would be applied twice. Every readout is answered, because the
    transport reads one reply for each it sends, and the readouts that
    complete no navigator are answered with the null correction: zero shift
    and the identity, which accumulates to nothing.

    Parameters
    ----------
    planes
        Readouts per navigator. They must arrive in the same order every
        navigator, and between them their normals must span the rotations and
        their in-plane axes the translations -- three orthogonal planes being
        the usual set.
    matrix
        Grid each plane is reconstructed on, holding the field of view, so a
        grid finer than the navigator was encoded at interpolates the plane
        rather than widening it. ``None`` is four times the encoded matrix,
        the ratio the method was validated at: the extra pixels claim no
        resolution the navigator did not acquire, they give the registration
        a smoother surface to descend, and rotation is what most feels it.
    iterations
        Optimizer iterations per plane, the setting that trades tracking
        latency against pose precision.
    navigator_tr
        Seconds between navigators, which is what the filter propagates its
        velocity over. ``None`` reads it off the acquisition timestamps.

    Examples
    --------
    Calling the module tracks a recorded navigator file, through the same
    hooks the scanner's stream is driven through, and returns one pose per
    navigator::

        from pulserver.app import pmc_recon

        poses = pmc_recon("navigators.h5")
        degrees = [np.rad2deg(pose.angles) for pose in poses]

    Its settings are the arguments of that call:

    >>> import inspect
    >>> from pulserver.app import pmc_recon
    >>> "navigator_tr" in inspect.signature(pmc_recon).parameters
    True

    Live, the real-time port drives ``process`` instead, and the poses go back
    to the sequence as they are measured rather than into a list.
    """

    def __init__(
        self,
        *,
        planes: int = 3,
        matrix: int | None = None,
        iterations: int = 10,
        navigator_tr: float | None = None,
    ) -> None:
        super().__init__(branches={}, buffered=False)
        if int(planes) < 2:
            raise ValueError(
                f"a pose needs at least two navigator planes, got {planes!r}"
            )
        self.planes = int(planes)
        self.matrix = None if matrix is None else int(matrix)
        self.iterations = int(iterations)
        self.navigator_tr = None if navigator_tr is None else float(navigator_tr)

    def startup(self, context: ReconContext) -> None:
        """Open the tracker and the state one navigator is assembled in."""
        super().startup(context)
        # One resolution, not a pyramid. A pyramid exists to catch a
        # displacement larger than the finest level can see, and a navigator
        # has neither problem: its planes are small enough that the coarse
        # levels are a handful of pixels, and the filter hands the
        # registration a prediction of where the head is before it starts. At
        # a navigator's matrix the levels cost time and accuracy both.
        self.tracker = NavigatorMotionTracker(
            registration=RigidRegistration(
                iterations=self.iterations,
                shrink_factors=(1,),
                smoothing_sigmas=(0.0,),
            )
        )
        self.navigator: list[Any] = []
        self.gridding: dict[int, NonCartesian2D] = {}
        # The pose the scanner is already holding, and the one it has not been
        # told about yet. The wire carries the difference between them.
        self.corrected = np.eye(4)
        self.pending: np.ndarray | None = None
        self.acquired: float | None = None

    def receive(self, acquisition: Any, context: ReconContext) -> Any:
        """Collect one plane, and track once the navigator is complete.

        A navigator is several readouts and closes on none of the flags a
        branch can be declared against, so what completes it is counted here.
        """
        data = self.process(acquisition)
        if data is None:
            return None
        self.acquisition = acquisition
        self.navigator.append((acquisition, data))
        if len(self.navigator) < self.planes:
            return None
        return self.recon("navigator", context)

    def recon(self, branch: str, context: ReconContext) -> Any:
        """Reconstruct one navigator's planes and measure the pose they show."""
        del branch
        encoding = context.header.encoding[
            int(acquisition_label(self.acquisition, "encoding_space_ref", 0) or 0)
        ]
        native = int(encoding.encodedSpace.matrixSize.x)
        matrix = 4 * native if self.matrix is None else self.matrix
        field_of_view = float(encoding.reconSpace.fieldOfView_mm.x)

        images, axes = [], []
        for index, (acquisition, data) in enumerate(self.navigator):
            trajectory = np.asarray(acquisition.traj, dtype=np.float64)
            # The plane a readout lies in is the one its own samples span, so
            # the normal is the direction they never left. Its in-plane pair
            # is then any orthonormal basis of the plane, built the same way
            # every time so that a plane is measured against itself.
            normal = np.linalg.svd(trajectory[:, :3], full_matrices=False)[2][2]
            row = np.cross(normal, np.eye(3)[int(np.argmin(np.abs(normal)))])
            row = row / np.linalg.norm(row)
            column = np.cross(normal, row)
            # Holding the field of view while the grid gets finer is what
            # interpolates the plane rather than widening it.
            plane = (trajectory[:, :3] @ np.stack([row, column]).T) * (native / matrix)
            if index not in self.gridding:
                # A navigator replays one trajectory for the whole scan, so
                # the weights and the transform are built once per plane and
                # not once per navigator. That is most of what keeps a train
                # inside the recovery period it has to be finished in.
                plane = plane.astype(np.float32)
                self.gridding[index] = NonCartesian2D(
                    plane,
                    (matrix, matrix),
                    density=pipe_menon_dcf(plane, (matrix, matrix)),
                    n_coils=int(np.shape(data)[0]),
                )
            # The density-compensated adjoint, one image per coil, combined
            # root-sum-of-squares: a landmark, not a picture.
            images.append(coil_combine(self.gridding[index].A_adjoint(data[None])[0]))
            axes.append((row, column))

        acquired = float(self.acquisition.acquisition_time_stamp) * 2.5e-3
        elapsed = self.navigator_tr
        if elapsed is None:
            elapsed = 1.0 if self.acquired is None else acquired - self.acquired
        self.acquired = acquired
        self.navigator = []

        pose = self.tracker.track(
            images,
            axes,
            dt=max(elapsed, 1e-3),
            spacing=field_of_view / matrix,
        )
        self.pending = pose.matrix
        return pose

    def payload(self, acquisition: Any) -> PmcPayload:
        """The correction this readout is answered with, in scanner axes.

        Every readout gets a reply and only the one that completed a navigator
        carries a correction, so the rest answer with the null one. The reply
        is the change since the last correction, because the scanner
        accumulates what it is sent.
        """
        if self.pending is None:
            return PmcPayload()
        # The pose is measured on the trajectory's own axes, which are the
        # sequence's logical ones; the scanner wants its own. Each readout
        # carries the prescription that relates them.
        prescription = np.stack(
            [
                np.asarray(acquisition.read_dir, dtype=np.float64),
                np.asarray(acquisition.phase_dir, dtype=np.float64),
                np.asarray(acquisition.slice_dir, dtype=np.float64),
            ]
        )
        if not np.isclose(abs(np.linalg.det(prescription)), 1.0):
            prescription = np.eye(3)
        step = self.pending @ np.linalg.inv(self.corrected)
        self.corrected = self.pending
        self.pending = None
        rotation = prescription.T @ step[:3, :3] @ prescription
        shift = (prescription.T @ step[:3, 3]) * 1e-3
        return PmcPayload(
            shift=[float(value) for value in shift],
            rotation=[float(value) for value in rotation.reshape(-1)],
        )


PLUGIN = PmcRecon()


def process(connection: Any, config: Any, metadata: Any) -> None:
    """Answer a real-time navigator stream with the corrections it earns.

    The transport reads one reply for every readout it sends, so this replies
    once per readout: the one that completed a navigator carries the pose
    change measured from it, and the rest carry the null correction.

    Parameters
    ----------
    connection
        The real-time connection: it yields navigator readouts and takes a
        :class:`~pulserver.recon.PmcPayload` for each.
    config
        Configuration payload naming this application.
    metadata
        The parsed MRD header, describing the navigator's encoding spaces.
    """
    context = ReconContext(header=metadata, exam=ExamCache("rtp"), config=config)
    plugin = PLUGIN.spawn()
    plugin.startup(context)
    for acquisition in connection:
        plugin.receive(acquisition, context)
        connection.send(plugin.payload(acquisition))
