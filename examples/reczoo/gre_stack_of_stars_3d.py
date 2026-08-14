"""Reconstruction for :mod:`pulserver.seqzoo.gre_stack_of_stars_3d`.

A stack factorises: the partition axis is Cartesian, so an inverse FFT
along z turns the volume into independent planes, and each plane is then
the model-based non-Cartesian reconstruction of
:mod:`pulserver.reczoo.gre_radial_2d` -- density compensation,
adjoint-derived sensitivities, CG-SENSE -- against the in-plane trajectory
its acquisitions carry. One magnitude volume per measurement.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "GreStackOfStars3DRecon"]

from typing import Any

import numpy as np

from pulserver import AcquisitionBucket, ReconContext, ReconPlugin, ReconResult
from pulserver.recon import has_acquisition_flag
from pulserver.recon.preprocessing import receiver_channels
from pulserver.reczoo.gre_3d import recon_volume
from pulserver.reczoo.gre_radial_2d import reconstruct_plane


class GreStackOfStars3DRecon(ReconPlugin):
    """Reconstruct a non-Cartesian stack, one volume per measurement.

    Parameters
    ----------
    mode
        ``"auto"`` finishes a fully sampled stack with the
        density-compensated adjoint and sends everything undersampled
        through the per-plane CG-SENSE solve; ``"direct"`` and ``"pics"``
        force one branch. Fully sampled means the in-plane view count
        reaches the radial Nyquist count, ``ceil(pi/2 * matrix)``; the
        partition axis is assumed Cartesian-complete.
    regularization
        Tikhonov weight of the per-plane CG-SENSE solve.
    iterations
        Maximum CG iterations per plane.
    device
        Torch device the reconstruction runs on. ``None`` is the CPU.
    """

    def __init__(
        self,
        *,
        mode: str = "auto",
        regularization: float = 1e-3,
        iterations: int = 30,
        device: Any = None,
    ) -> None:
        super().__init__(
            split_on=("ACQ_LAST_IN_MEASUREMENT",),
            reject_flags=("ACQ_IS_NOISE_MEASUREMENT", "ACQ_IS_PHASECORR_DATA"),
        )
        if mode not in ("auto", "direct", "pics"):
            raise ValueError(f"mode must be auto, direct or pics, got {mode!r}")
        self.mode = mode
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.device = device

    def startup(self, context: ReconContext) -> None:
        """Size the volume from the header and open one bin per view."""
        self.volume_shape = recon_volume(context.header)
        self.coils = receiver_channels(context.header)
        #: ``(view key) -> {partition: data}`` plus the view's trajectory.
        self.views: dict[int, dict[int, Any]] = {}
        self.trajectories: dict[int, Any] = {}

    def receive(self, acquisition: Any, context: ReconContext) -> None:
        """File one acquisition under its view and partition.

        Views are keyed by arrival order within their partition: the loop
        plays partitions inside views, so an acquisition's view index is how
        many of its own partition arrived before it.
        """
        del context
        partition = int(acquisition.idx.kspace_encode_step_2)
        view = sum(
            1 for stored in self.views.values() if partition in stored
        )
        bin_ = self.views.setdefault(view, {})
        bin_[partition] = np.asarray(acquisition.data)
        self.trajectories.setdefault(view, np.asarray(acquisition.traj))

    def recon(
        self, bucket: AcquisitionBucket, context: ReconContext
    ) -> ReconResult | None:
        """Reconstruct once the measurement is complete."""
        del context
        last = bucket.acquisitions[-1]
        if not has_acquisition_flag(last, "ACQ_LAST_IN_MEASUREMENT"):
            return None

        n_z = self.volume_shape[0]
        views = sorted(self.views)
        # (views, coils, partitions, samples), inverse FFT along partitions.
        stacked = np.stack(
            [
                np.stack(
                    [self.views[view][partition] for partition in range(n_z)],
                    axis=1,
                )
                for view in views
            ]
        )
        planes = np.fft.fftshift(
            np.fft.ifft(
                np.fft.ifftshift(stacked, axes=2), axis=2, norm="ortho"
            ),
            axes=2,
        )

        trajectory = np.concatenate(
            [self.trajectories[view][:, :2] for view in views], axis=0
        )
        nyquist = int(np.ceil(np.pi / 2 * max(self.volume_shape[1:])))
        direct = self.mode == "direct" or (
            self.mode == "auto" and len(views) >= nyquist
        )
        volume = []
        for plane in range(n_z):
            data = np.concatenate(
                [planes[view_index, :, plane, :] for view_index in range(len(views))],
                axis=-1,
            )
            image = reconstruct_plane(
                data,
                trajectory,
                self.volume_shape[1:],
                direct=direct,
                regularization=self.regularization,
                iterations=self.iterations,
                device=self.device,
            )
            volume.append(np.abs(_asnumpy(image)))

        return ReconResult(
            np.stack(volume).transpose(0, 2, 1),
            reference=-1,
            image_type="magnitude",
            dicom=True,
        )


def _asnumpy(array: Any) -> np.ndarray:
    """Bring a reconstruction back to NumPy, whether it is Torch or already so."""
    return array.cpu().numpy() if hasattr(array, "cpu") else np.asarray(array)


PLUGIN = GreStackOfStars3DRecon()
