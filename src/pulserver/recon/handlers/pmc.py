"""Prospective motion correction from three-plane spiral navigators with bartorch, around an imaging reconstruction."""

from __future__ import annotations

__all__ = ["PLUGIN", "PmcRecon"]

from typing import Any

import numpy as np

from ...mrd._metadata import has_acquisition_flag
from ...proxy._motion import Pose, pose_waveform
from ..plugin import ReconPlugin
from .pics import PicsRecon

#: Flags either of which marks a navigator readout: pypulseqpp's ``NAV`` label,
#: or ``RTFEEDBACK``.
NAVIGATOR = ("ACQ_IS_NAVIGATION_DATA", "ACQ_IS_RTFEEDBACK_DATA")

#: Planes of one navigator, as pypulseqpp's ``SpiralNavigator`` plays them.
PLANES = 3


class PmcRecon(ReconPlugin):
    """Rigid pose from each navigator, stated to the scan; every other readout to ``imaging``.

    Navigator readouts (``NAV`` or ``RTFEEDBACK``) are collected a navigator at a time, one
    readout per plane. Each plane is reconstructed by a density-compensated
    adjoint NUFFT, coils combined by root sum of squares
    (:func:`bartorch.tools.reconstruct_navigator`), on one thread. The
    Pipe-Menon density (:func:`bartorch.estimate_density`) is computed once,
    from the first navigator's trajectory, which the proxy stamps from the
    sequence. The planes are registered against the first navigator's and the
    pose filtered by an extended Kalman filter
    (:class:`bartorch.tools.NavigatorMotionTracker`); the result is returned
    as a pose waveform, which the proxy publishes to the scan when the design
    sets ``EnablePmc``.

    A plane's in-plane axes are the two physical axes its trajectory spans,
    so the pose is in the physical frame, relative to the first navigator.

    Parameters
    ----------
    imaging
        The reconstruction of the readouts that are not navigators.
    matrix
        In-plane matrix each plane is reconstructed on.
    """

    def __init__(self, imaging: ReconPlugin | None = None, matrix: int = 32) -> None:
        imaging = PicsRecon() if imaging is None else imaging
        super().__init__(
            branches=imaging.branches,
            require_flags=imaging.require_flags,
            reject_flags=imaging.reject_flags,
            buffered=False,
        )
        self.imaging = imaging
        self.matrix = matrix

    def spawn(self) -> PmcRecon:
        plugin = super().spawn()
        plugin.imaging = self.imaging.spawn()
        return plugin

    def startup(self, context):
        self.imaging.startup(context)
        self.planes = []
        self.geometry = None
        self.tracker = None

    def receive(self, acquisition, context):
        if not any(has_acquisition_flag(acquisition, flag) for flag in NAVIGATOR):
            return self.imaging.receive(acquisition, context)
        self.planes.append(acquisition)
        if len(self.planes) < PLANES:
            return None
        planes, self.planes = self.planes, []
        return pose_waveform(self.pose(planes))

    def recon(self, branch, context):
        return self.imaging.recon(branch, context)

    def pose(self, planes: list[Any]) -> Pose:
        """Return the filtered pose of one navigator's readouts, one per plane."""
        import bartorch
        import torch
        from bartorch import tools

        bartorch.set_num_threads(1)
        if self.geometry is None:
            self.geometry = _geometry(planes, self.matrix)
            self.tracker = tools.NavigatorMotionTracker()
        traj, axes, spacing_mm, density = self.geometry
        kspace = torch.from_numpy(
            np.stack([np.asarray(plane.data, dtype=np.complex64) for plane in planes])
        )
        images = tools.reconstruct_navigator(
            kspace, traj, (self.matrix, self.matrix), density=density
        )
        estimate = self.tracker.track(list(images), axes, spacing=spacing_mm)
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
