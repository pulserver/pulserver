"""Prospective motion correction from three-plane spiral navigators with bartorch, alongside the pics reconstruction."""

from __future__ import annotations

__all__ = ["PLUGIN", "PmcRecon"]

from typing import Any

import numpy as np

from ...proxy._motion import Pose, pose_waveform
from .._buffers import ReconData
from .._units import carries
from ..gadgets import NAVIGATOR
from ..plugin import ReconContext
from .pics_train import PicsTrainRecon

#: Planes of one navigator, as pypulseqpp's ``SpiralNavigator`` plays them.
PLANES = 3


class PmcRecon(PicsTrainRecon):
    """Rigid pose from each navigator, stated to the scan; every other readout reconstructed as :class:`~pulserver.recon.handlers.pics_train.PicsTrainRecon`.

    Navigator readouts (``NAV`` or ``RTFEEDBACK``) belong to the ``navigator``
    branch, whose units are one readout each, and are collected a navigator at
    a time, one readout per plane. A plane is the readout as received, so the
    whitening of the stream is not applied to it. Each plane is reconstructed
    by a density-compensated adjoint NUFFT, coils combined by root sum of
    squares (:func:`bartorch.tools.reconstruct_navigator`), on one thread. The
    Pipe-Menon density (:func:`bartorch.estimate_density`) is computed once,
    from the first navigator's trajectory, which the proxy stamps from the
    sequence. The planes are registered against the first navigator's and the
    pose filtered by an extended Kalman filter
    (:class:`bartorch.tools.NavigatorMotionTracker`); the result is returned
    as a pose waveform, which the proxy publishes to the scan when the design
    sets ``EnablePmc``.

    The navigators of one repetition see different magnetisation: each comes
    a different time after the imaging readouts before it. So each position in
    the repetition, counted from the last imaging readout, has its own
    reference, taken from its second occurrence (the first comes before the
    steady state), and its own filter; until then the pose stated is the
    identity. On a still phantom (7 T, 2 Nov 2026) one reference for all five
    positions read up to 0.8 degrees and 0.4 mm apart, each position within
    0.05 degrees and 0.02 mm of itself.

    A plane's in-plane axes are the two physical axes its trajectory spans,
    so the pose is in the physical frame, relative to the first navigator.

    Parameters
    ----------
    matrix : int, default=32
        In-plane matrix each plane is reconstructed on.
    wavelet : float, default=0.005
        As :class:`~pulserver.recon.handlers.pics_train.PicsTrainRecon`.
    iterations : int, default=30
        As :class:`~pulserver.recon.handlers.pics_train.PicsTrainRecon`.
    """

    def __init__(
        self, matrix: int = 32, wavelet: float = 0.005, iterations: int = 30
    ) -> None:
        super().__init__(wavelet, iterations)
        self.triggers["navigator"] = NAVIGATOR
        self.matrix = matrix

    def startup(self, context: ReconContext) -> None:
        super().startup(context)
        self.planes: list[Any] = []
        self.geometry = None
        self.trackers: dict[int, Any] = {}
        self.seen: dict[int, int] = {}
        self.position = 0

    def branch_for(self, acquisition: Any) -> str | None:
        if carries(acquisition, NAVIGATOR):
            return "navigator"
        # An imaging readout: the next navigator is the first of its repetition.
        self.position = 0
        return super().branch_for(acquisition)

    def recon(self, context: ReconContext, branch: str, data: ReconData) -> Any:
        if branch != "navigator":
            return super().recon(context, branch, data)
        self.planes.extend(data.acquisitions)
        if len(self.planes) < PLANES:
            return None
        planes, self.planes = self.planes, []
        position, self.position = self.position, self.position + 1
        return pose_waveform(self.pose(planes, position))

    def pose(self, planes: list[Any], position: int = 0) -> Pose:
        """Return the filtered pose of one navigator's readouts, one per plane, at ``position`` in its repetition.

        The pose is computed on one thread, and the thread count of the process
        is restored after it, so the image units are reconstructed with the
        count they had.
        """
        import bartorch
        import torch
        from bartorch import tools

        # bartorch's count is process-wide and BART shares torch's OpenMP
        # runtime, so torch reports the count BART runs with.
        seen = self.seen.get(position, 0)
        self.seen[position] = seen + 1
        if seen == 0:
            return Pose(rotation=tuple(float(v) for v in np.eye(3).ravel()))
        threads = torch.get_num_threads()
        bartorch.set_num_threads(1)
        try:
            if self.geometry is None:
                self.geometry = _geometry(planes, self.matrix)
            tracker = self.trackers.get(position)
            if tracker is None:
                tracker = self.trackers[position] = tools.NavigatorMotionTracker()
            traj, axes, spacing_mm, density = self.geometry
            kspace = torch.from_numpy(
                np.stack(
                    [np.asarray(plane.data, dtype=np.complex64) for plane in planes]
                )
            )
            images = tools.reconstruct_navigator(
                kspace, traj, (self.matrix, self.matrix), density=density
            )
            estimate = tracker.track(list(images), axes, spacing=spacing_mm)
        finally:
            bartorch.set_num_threads(threads)
        rotation = estimate.matrix[:3, :3]
        return Pose(
            rotation=tuple(float(v) for v in rotation.ravel()),
            translation_m=tuple(1e-3 * float(v) for v in estimate.translation),
        )


PLUGIN = PmcRecon()


def _geometry(planes, matrix):
    """Return each plane's trajectory in grid units, its ``(row, column)`` axes, the pixel size in mm and the density."""
    import bartorch
    import torch

    k = []
    for plane in planes:
        traj = np.asarray(plane.traj, dtype=np.float64)
        k.append(np.pad(traj, ((0, 0), (0, 3 - traj.shape[1]))))
    k = np.stack(k)
    k_max = float(np.abs(k).max())
    traj, axes = [], []
    for plane in k:
        row, column = np.sort(np.argsort(np.ptp(plane, axis=0))[-2:])
        traj.append(plane[:, [column, row]])
        axes.append((np.eye(3)[row], np.eye(3)[column]))
    # Inside the grid's +-n/2 the transform takes, by a rounding's margin.
    traj = torch.from_numpy(np.stack(traj) * (0.999 * matrix / (2 * k_max))).float()
    density = torch.stack(
        [bartorch.estimate_density(plane, (matrix, matrix)) for plane in traj]
    )
    return traj, axes, 1e3 / (2 * k_max), density
