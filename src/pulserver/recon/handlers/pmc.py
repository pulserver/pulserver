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
from .pics import PicsRecon

#: Planes of one navigator, as pypulseqpp's ``SpiralNavigator`` plays them.
PLANES = 3


class PmcRecon(PicsRecon):
    """Rigid pose from each navigator, stated to the scan; every other readout reconstructed as :class:`~pulserver.recon.handlers.pics.PicsRecon`.

    Navigator readouts (``NAV`` or ``RTFEEDBACK``) belong to the ``navigator``
    branch, whose units are one readout each, and are collected a navigator at
    a time, one readout per plane. Each plane is reconstructed by a
    density-compensated adjoint NUFFT, coils combined by root sum of squares
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
    matrix
        In-plane matrix each plane is reconstructed on.
    wavelet, iterations
        As :class:`~pulserver.recon.handlers.pics.PicsRecon`.
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
        self.tracker = None

    def branch_for(self, acquisition: Any) -> str | None:
        if carries(acquisition, NAVIGATOR):
            return "navigator"
        return super().branch_for(acquisition)

    def recon(self, context: ReconContext, branch: str, data: ReconData) -> Any:
        if branch != "navigator":
            return super().recon(context, branch, data)
        self.planes.extend(data.acquisitions)
        if len(self.planes) < PLANES:
            return None
        planes, self.planes = self.planes, []
        return pose_waveform(self.pose(planes))

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
