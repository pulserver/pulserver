"""Physics for wave-encoded scans.

A corkscrew gradient during the readout spreads each voxel along it, so the
aliasing that parallel imaging has to separate is spread too."""

from __future__ import annotations

from typing import Any


from .._toeplitz import (
    as_torch,
)
from .._wave import _WaveLinearPhysics

from ._base import MRIPhysics


class WaveShuffling(MRIPhysics):
    """Three-dimensional Wave-Shuffling subspace physics.

    ``sampling`` contains ``(phase, partition, echo)`` indices and ``basis``
    follows Pulserver's ``(rank, echoes)`` convention. The forward and adjoint
    gather/scatter only acquired lines. The normal operator uses the exact
    packed temporal kernel in hybrid k-space and never materializes an echo
    train or a dense ``rank x rank`` field.

    Parameters
    ----------
    sampling
        Acquired ``(phase, partition, echo)`` indices.
    coil_maps
        Complex sensitivities over the reconstructed volume.
    wave_psf
        Wave point-spread function: a tensor, or a
        :class:`pulserver.recon.calibration.WavePSFResult`.
    basis
        Temporal basis shaped ``(rank, echoes)``.
    **kwargs
        Forwarded to the wave-shuffling operator.

    Examples
    --------
    Wave encoding and a temporal subspace at once: the corkscrew separates the
    aliasing, the basis carries the contrast change through the echo train, and
    one solve recovers the coefficient images both describe::

        import pulserver.recon as recon

        physics = recon.WaveShuffling(sampling, coil_maps, wave_psf, basis)
        coefficients = recon.pics(measured, physics)
    """

    def __init__(
        self,
        sampling: Any,
        coil_maps: Any,
        wave_psf: Any,
        basis: Any,
        *,
        line_weights: Any | None = None,
        viewed_as_real: bool = False,
        coil_batch_size: int = 1,
        cuda_transfer_precision: str = "auto",
        streaming: Any | None = None,
    ) -> None:
        operator = _WaveLinearPhysics(
            sampling,
            coil_maps,
            wave_psf,
            basis,
            line_weights=line_weights,
            viewed_as_real=viewed_as_real,
            coil_batch_size=coil_batch_size,
            cuda_transfer_precision=cuda_transfer_precision,
        )
        super().__init__(
            operator,
            native_operator=None,
            kind="wave-shuffling",
            spatial_ndim=3,
            viewed_as_real=viewed_as_real,
            modifiers=("wave", "subspace"),
        )
        if streaming is not None:
            self.enable_streaming(streaming)


class WaveEncoding(MRIPhysics):
    """Wave-CAIPI encoding: a corkscrew gradient spread along the readout.

    Parameters
    ----------
    sampling
        Acquired ``(phase, partition, echo)`` indices, or ``(phase,
        partition)`` for a single-echo scan.
    coil_maps
        Complex sensitivities over the reconstructed volume.
    wave_psf
        Wave point-spread function: a tensor, or a
        :class:`pulserver.recon.calibration.WavePSFResult` from
        :class:`pulserver.recon.calibration.WavePSFCalibration`.
    line_weights
        Optional per-line weights over the acquired samples.
    viewed_as_real
        Exchange images and measurements through real views.
    coil_batch_size
        Coils processed together by the hybrid-space normal operator.
    cuda_transfer_precision
        Precision of host-to-device transfers when streaming.
    streaming
        Optional :class:`pulserver.recon.execution.CudaStreaming` policy.

    Examples
    --------
    Wave encoding plays sinusoidal gradients during the readout, so each line
    is smeared along the encoded axes by a corkscrew point-spread function.
    Spreading the aliasing that way is what lets a higher acceleration still
    separate: at sixteen-fold the same coils and the same solver recover the
    object with the corkscrew and lose it without.

    .. plot::

       import torch
       import pulserver.recon as recon
       from _figures import images, volume, wave_gradients

       truth, coil_maps = volume(size=24, coils=4, depth=16)
       image, maps = truth[:, None], coil_maps[0]
       shape = tuple(truth.shape[1:])

       gradients, raster, times = wave_gradients(samples=2 * shape[0])
       phase = recon.WavePSF.phase_from_gradients(gradients, raster, times)
       axis_1 = torch.linspace(-0.10, 0.10, shape[1])
       axis_2 = torch.linspace(-0.10, 0.10, shape[2])
       corkscrew = recon.WavePSF(axis_1, axis_2)(phase)

       # Four-fold along each encoded axis, sixteen-fold overall.
       grid = torch.stack(
           torch.meshgrid(
               torch.arange(0, shape[1], 4),
               torch.arange(0, shape[2], 4),
               indexing="ij",
           ),
           -1,
       )
       sampling = grid.reshape(-1, 2)

       def solved(point_spread):
           physics = recon.WaveEncoding(sampling, maps, point_spread)
           return recon.pics(physics.A(image), physics, iterations=8)[
               0, 0, shape[0] // 2
           ]

       images(
           [
               ("object", truth[0, shape[0] // 2]),
               ("no wave gradients", solved(torch.ones_like(corkscrew))),
               ("wave encoded", solved(corkscrew)),
           ],
           title="sixteen-fold undersampled: spreading the aliasing separates it",
       )
    """

    def __init__(
        self,
        sampling: Any,
        coil_maps: Any,
        wave_psf: Any,
        *,
        line_weights: Any | None = None,
        viewed_as_real: bool = False,
        coil_batch_size: int = 1,
        cuda_transfer_precision: str = "auto",
        streaming: Any | None = None,
    ) -> None:
        sampling_tensor = as_torch(sampling)
        echoes = (
            int(sampling_tensor[:, 2].max()) + 1
            if sampling_tensor.ndim == 2
            and sampling_tensor.shape[1] == 3
            and sampling_tensor.shape[0]
            else 1
        )
        basis = as_torch(coil_maps).real.new_ones((1, echoes))
        operator = _WaveLinearPhysics(
            sampling,
            coil_maps,
            wave_psf,
            basis,
            line_weights=line_weights,
            viewed_as_real=viewed_as_real,
            coil_batch_size=coil_batch_size,
            cuda_transfer_precision=cuda_transfer_precision,
        )
        super().__init__(
            operator,
            native_operator=None,
            kind="wave",
            spatial_ndim=3,
            viewed_as_real=viewed_as_real,
            modifiers=("wave",),
        )
        if streaming is not None:
            self.enable_streaming(streaming)
