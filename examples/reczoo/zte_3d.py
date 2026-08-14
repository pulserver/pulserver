"""Reconstruction for :mod:`pulserver.seqzoo.zte_3d`.

A 3D non-Cartesian reconstruction over the whole koosh-ball: Pipe--Menon
density compensation and a coil-wise adjoint NUFFT through
:class:`pulserver.recon.physics.NonCartesian3D` -- the direct finish, valid
because a ZTE shell set is designed at Nyquist -- with a CG solve available
for anything undersampled, exactly as the 2D non-Cartesian plugin branches.
The dead-time gap's missing centre samples are declared by the sequence and
left to the density weighting here; a dedicated gap-filling refinement
starts by overriding this plugin.

Needs the numerical stack: ``pip install "pulserver[recon-cpu]"``.
"""

from __future__ import annotations

__all__ = ["PLUGIN", "Zte3DRecon"]

from typing import Any

import numpy as np

from pulserver import AcquisitionBucket, ReconContext, ReconPlugin, ReconResult
from pulserver.recon import has_acquisition_flag
from pulserver.recon.preprocessing import pipe_menon_dcf, receiver_channels
from pulserver.reczoo.gre_3d import recon_volume


class Zte3DRecon(ReconPlugin):
    """Reconstruct a ZTE sphere, one volume per measurement.

    Parameters
    ----------
    mode
        ``"direct"`` (the default: the shell set satisfies Nyquist by
        design) or ``"pics"`` for an undersampled prescription.
    regularization, iterations
        The CG solve's Tikhonov weight and iteration ceiling, ``pics`` only.
    device
        Torch device the reconstruction runs on. ``None`` is the CPU.
    """

    def __init__(
        self,
        *,
        mode: str = "direct",
        regularization: float = 1e-3,
        iterations: int = 20,
        device: Any = None,
    ) -> None:
        super().__init__(
            split_on=("ACQ_LAST_IN_MEASUREMENT",),
            reject_flags=("ACQ_IS_NOISE_MEASUREMENT", "ACQ_IS_PHASECORR_DATA"),
        )
        if mode not in ("direct", "pics"):
            raise ValueError(f"mode must be direct or pics, got {mode!r}")
        self.mode = mode
        self.regularization = float(regularization)
        self.iterations = int(iterations)
        self.device = device

    def startup(self, context: ReconContext) -> None:
        """Size the volume from the header and collect the spokes."""
        self.volume_shape = recon_volume(context.header)
        self.coils = receiver_channels(context.header)
        self.spokes: list[tuple[Any, Any]] = []

    def receive(self, acquisition: Any, context: ReconContext) -> None:
        """File one spoke's data and trajectory."""
        del context
        self.spokes.append(
            (np.asarray(acquisition.data), np.asarray(acquisition.traj))
        )

    def recon(
        self, bucket: AcquisitionBucket, context: ReconContext
    ) -> ReconResult | None:
        """Reconstruct once the measurement is complete."""
        del context
        last = bucket.acquisitions[-1]
        if not has_acquisition_flag(last, "ACQ_LAST_IN_MEASUREMENT"):
            return None

        import torch

        from pulserver.recon.physics import NonCartesian3D

        data = torch.as_tensor(
            np.concatenate([spoke for spoke, _ in self.spokes], axis=-1),
            dtype=torch.complex64,
            device="cpu" if self.device is None else self.device,
        )
        points = np.concatenate(
            [trajectory[:, :3] for _, trajectory in self.spokes], axis=0
        ).astype(np.float32)

        density = pipe_menon_dcf(points, self.volume_shape)
        n_coils = int(data.shape[0])
        coil_wise = NonCartesian3D(
            points, self.volume_shape, density=density, n_coils=n_coils
        )
        coils = coil_wise.A_adjoint(torch.view_as_real(data[None]))
        coil_images = torch.view_as_complex(
            coils.reshape(1, 2, n_coils, *self.volume_shape)
            .movedim(1, -1)
            .contiguous()
        )[0]

        if self.mode == "direct":
            image = torch.sqrt(torch.sum(torch.abs(coil_images) ** 2, dim=0))
        else:
            from pulserver.recon import pics
            from pulserver.reczoo.gre_radial_2d import smooth_sensitivities

            maps = smooth_sensitivities(coil_images)
            sense = NonCartesian3D(
                points,
                self.volume_shape,
                coil_maps=maps,
                density=density,
                n_coils=n_coils,
            )
            solution = pics(
                torch.view_as_real(data[None]),
                sense,
                regularization=self.regularization,
                iterations=self.iterations,
            )
            image = torch.abs(
                torch.view_as_complex(solution.movedim(1, -1).contiguous())[0]
            )

        return ReconResult(
            _to_numpy(image).transpose(0, 2, 1),
            reference=-1,
            image_type="magnitude",
            dicom=True,
        )


def _to_numpy(array: Any) -> np.ndarray:
    """Bring a reconstruction back to NumPy, whether it is Torch or already so."""
    return array.cpu().numpy() if hasattr(array, "cpu") else np.asarray(array)


PLUGIN = Zte3DRecon()
