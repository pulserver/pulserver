"""Echo-resolved reconstruction of shuffled train scans in a temporal subspace, as BART's ``pics -B`` (T2 shuffling) and ``wshfl`` (wave-shuffling)."""

from __future__ import annotations

__all__ = [
    "PLUGIN",
    "FseSubspaceRecon",
    "MegreSubspaceRecon",
    "MprageSubspaceRecon",
    "SubspaceRecon",
]

import contextlib
import dataclasses
from types import SimpleNamespace
from typing import Any

import numpy as np

from ...mrd._images import center_crop
from .._buffers import ReconBuffer, ReconData, readout_roles
from .._calibration import MissingCalibration, coil_maps
from ..gadgets import AsymmetricEcho, Prewhiten, RemoveReadoutOversampling
from ..plugin import ReconContext, ReconResult
from .pics import PicsRecon, _volume, _wave_trajectory, compressed, nlinv_estimate

#: Temporal basis functions kept: the leading singular vectors of the dictionary.
RANK = 3

#: Edge, in voxels, of the blocks of the locally low-rank penalty.
BLOCK = 8

#: Weight of the calibration-only readouts beside the imaging readouts in the
#: solve: at one, their fully sampled centre at the train's first echoes pulls
#: the late echoes off the basis's span.
CALIBRATION_WEIGHT = 0.3

#: T1 values of the dictionary, in ms: white matter to CSF at 3 T and 7 T.
T1_MS = (400.0, 800.0, 1200.0, 1600.0, 2000.0, 2800.0, 4400.0)

#: Fewest and most T2, in ms, and how many log-spaced between them.
T2_MS = (20.0, 2000.0, 48)

#: Lowest and highest transmit field relative to nominal, and how many between
#: them: the range a head sees at 7 T.
B1 = (0.25, 1.25, 21)

#: TRs simulated before the train the basis is taken from, so the basis is of
#: the steady state the scan reaches.
REPETITIONS = 3


class SubspaceRecon(PicsRecon):
    """Echo images of a 3D train scan whose phase encodes are shuffled across the echoes of its trains.

    Each readout's ``contrast`` counter is its echo in the train. The images
    of all echoes are spanned by ``rank`` temporal basis functions, the
    leading singular vectors of a dictionary of echo trains that blochsim
    simulates over T1, T2 and the transmit field from the sequence
    description the stream carries (:func:`blochsim.description_from_waveforms`)
    as :meth:`interpret` reads it. The image coefficients minimise

    ``sum_t |P_t F Psi S sum_a Phi[a, t] x_a - y_t|^2 + lambda sum_b |X_b|_*``

    ``P_t`` the phase encodes acquired at echo ``t``, ``Psi`` the wave
    point-spread function of a wave-encoded scan and the identity otherwise,
    ``S`` the coil maps and ``X_b`` the coefficients of a block of ``block``
    voxels an edge, as a matrix of voxels by coefficients
    (:class:`bartorch.priors.LocallyLowRank`). The operator applies its normal
    in closed form (:class:`bartorch.linop.CartesianSense`,
    :class:`bartorch.linop.WaveSense`) and the solve is FISTA.

    Readouts are whitened and compressed to ``virtual_coils`` channels as the
    unit closes, as :class:`~pulserver.recon.handlers.pics_train.PicsTrainRecon`
    compresses them. The coil maps of a Cartesian scan are those
    :class:`~pulserver.recon.handlers.pics.PicsRecon` takes; those of a
    wave-encoded scan are fitted to its wave-free calibration readouts, and
    its wave point-spread function is that of :func:`_wave_psf`. A sequence
    that labels its echoes as contrasts (``axes=("contrast",)``) is laid out
    by contrast; one that fills each line at one echo of its train is laid
    out by line, with the echo of each line read from its contrast counter.

    The echo images, at the reconstruction matrix, are series 0, the echoes
    in train order, each a volume.

    Parameters
    ----------
    rank : int, default=RANK
        Temporal basis functions.
    llr : float, default=0.003
        ``lambda``, relative to the data scaling.
    iterations : int, default=40
        FISTA iterations.
    virtual_coils : int | None, default=6
        Receive channels beyond this many are compressed to it.
    block : int, default=BLOCK
        Block edge of the locally low-rank penalty.
    calibration_weight : float, default=CALIBRATION_WEIGHT
        Weight of the calibration-only readouts of a unit laid out by line,
        each solved in the frame of its echo without the wave, beside the
        imaging readouts; zero leaves them to the coil maps alone.
    kspace_filter, gradient_unwarp, intensity_correction : bool, default=True
        As :class:`~pulserver.recon.handlers.pics.PicsRecon`.
    """

    def __init__(
        self,
        rank: int = RANK,
        llr: float = 0.003,
        iterations: int = 40,
        virtual_coils: int | None = 6,
        block: int = BLOCK,
        calibration_weight: float = CALIBRATION_WEIGHT,
        *,
        kspace_filter: bool = True,
        gradient_unwarp: bool = True,
        intensity_correction: bool = True,
    ) -> None:
        super().__init__(
            llr,
            iterations,
            virtual_coils,
            kspace_filter=kspace_filter,
            gradient_unwarp=gradient_unwarp,
            intensity_correction=intensity_correction,
        )
        self.merge = ("contrast",)
        self.gadgets = (Prewhiten(), AsymmetricEcho(), RemoveReadoutOversampling())
        self.rank = rank
        self.block = block
        self.calibration_weight = calibration_weight
        self._described: tuple[Any, ...] = ()

    def interpret(self, description: Any) -> Any:
        """Return the sequence description with the gradient events the signal model of this sequence needs.

        The stream's description carries the RF pulses, readouts and their
        timing alone; what the gradients between them do to the
        magnetization is the reconstruction's model of the sequence.
        """
        raise NotImplementedError

    def tissue(self) -> Any:
        """Return the :class:`blochsim.sequence.TissueProperties` the dictionary spans."""
        import torch
        from blochsim.sequence import TissueProperties

        t2 = torch.logspace(np.log10(T2_MS[0]), np.log10(T2_MS[1]), T2_MS[2])
        t1, t2, b1 = torch.meshgrid(
            torch.tensor(T1_MS), t2, torch.linspace(*B1), indexing="ij"
        )
        return TissueProperties(
            t1_ms=t1.reshape(-1), t2_ms=t2.reshape(-1), b1=b1.reshape(-1)
        )

    def basis(self, context: ReconContext, waveforms: tuple[Any, ...]) -> Any:
        """Return the temporal basis ``(rank, echoes)`` of the sequence the waveforms describe.

        Raises
        ------
        ValueError
            If the waveforms carry no sequence description.
        """
        import blochsim
        from blochsim.sequence import simulate_subspace

        def simulated() -> Any:
            descriptions = blochsim.description_from_waveforms(waveforms)
            if not descriptions:
                raise ValueError(
                    "the stream carries no sequence description, which the "
                    "temporal basis of a subspace reconstruction is simulated from"
                )
            subspace = simulate_subspace(
                self.interpret(descriptions[0]),
                self.tissue(),
                rank=self.rank,
                repetitions=REPETITIONS,
            )
            return subspace.basis[:, : self.rank].T.contiguous()

        key = ("subspace basis", type(self).__name__, self.rank, len(waveforms))
        return context.memoized(key, simulated)

    def recon(
        self, context: ReconContext, branch: str, data: ReconData
    ) -> list[ReconResult] | None:
        del branch
        if data.waveforms:
            self._described = data.waveforms
        data = compressed(context, data, self.virtual_coils)
        if data.data is None:
            if data.ref is not None:
                coil_maps(context, data, estimate=nlinv_estimate)
            return None
        echoes = self.echo_images(context, data)
        attributes = {"ImageProcessingHistory": ["PULSERVER", "PYTHON", "SUBSPACE"]}
        return [
            ReconResult(
                np.abs(image), attributes={**attributes, "EchoNumber": echo + 1}
            )
            for echo, image in enumerate(echoes)
        ]

    def echo_images(self, context: ReconContext, data: ReconData) -> np.ndarray:
        """Return the complex echo images ``(echoes, z, y, x)`` of a unit.

        Raises
        ------
        ValueError
            If the unit is not one volume, or a readout's echo lies outside
            the trains the description simulates.
        """
        import torch
        from bartorch import linop, optim, priors

        buffer = data.data
        volume = ("coil", "partition", "phase_encode", "readout")
        by_contrast = buffer.axes == (volume[0], "contrast", *volume[1:])
        if buffer.axes != volume and not by_contrast:
            raise ValueError(
                f"subspace reconstruction takes one volume {volume}, or one per "
                f"contrast; encoding space {buffer.space.index} holds {buffer.axes}"
            )
        basis = self.basis(context, self._described)
        frames = basis.shape[1]
        if by_contrast:
            if buffer.mask.shape[0] > frames:
                raise ValueError(
                    f"{buffer.mask.shape[0]} contrasts for trains of {frames} echoes"
                )
            positions, kspace = _by_contrast(buffer, frames)
        else:
            echo_of = _echoes(buffer, data.acquisitions)
            if echo_of.max() >= frames:
                raise ValueError(
                    f"a readout lies at echo {int(echo_of.max())} of trains of "
                    f"{frames} echoes"
                )
            positions, kspace = _by_echo(buffer, echo_of, frames)
        calibration = None
        if not by_contrast and data.ref is not None and self.calibration_weight > 0:
            calibration = _calibration_lines(data, frames)
        shape = buffer.image_shape
        cartesian = buffer.space.cartesian
        if cartesian:
            maps = self._maps(context, data, buffer)
        else:
            reference = _volume(_first_contrast(data.ref), shape)
            maps = center_crop(
                nlinv_estimate(torch.from_numpy(reference.astype(np.complex64))),
                shape[-1:],
            )
        image_shape = (self.rank, *maps.shape[1:])
        common = {
            "positions": torch.from_numpy(positions),
            "basis": basis.to(torch.complex64),
            "device": context.device,
        }
        if cartesian:
            encoding = linop.CartesianSense(maps.contiguous(), image_shape, **common)
        else:
            encoding = linop.WaveSense(
                maps.contiguous(),
                image_shape,
                kspace.shape[-1],
                centred=True,
                psf=_wave_psf(data, shape).to(torch.complex64),
                **common,
            )
        measured = torch.from_numpy(kspace)
        if calibration is not None:
            weight = self.calibration_weight
            positions_ref, kspace_ref = calibration
            alone = linop.CartesianSense(
                maps.contiguous(),
                image_shape,
                **{**common, "positions": torch.from_numpy(positions_ref)},
            )
            if not cartesian:
                alone = linop.WaveSense(
                    maps.contiguous(),
                    image_shape,
                    kspace.shape[-1],
                    centred=True,
                    psf=torch.ones(
                        (*shape[:-1], kspace.shape[-1]), dtype=torch.complex64
                    ),
                    **{**common, "positions": torch.from_numpy(positions_ref)},
                )
            encoding = linop.concatenate([encoding, weight * alone], axis=-2)
            measured = torch.cat(
                [measured, weight * torch.from_numpy(kspace_ref)], dim=-2
            )
        scaling = optim.data_scaling(measured, A=encoding) or 1.0
        prior = priors.LocallyLowRank(
            (-3, -2, -1), self.wavelet, joint_axes=(0,), block=self.block
        )
        solve = optim.FISTA(prior, maxiter=self.iterations, eigen=True)
        coefficients = scaling * solve(measured * (1.0 / scaling), encoding)
        echoes = torch.einsum(
            "kf,kzyx->fzyx", basis.to(coefficients.dtype), coefficients
        )
        return np.stack([self._filtered(e) for e in echoes.numpy()])

    def _maps(self, context: ReconContext, data: ReconData, buffer: ReconBuffer) -> Any:
        """Return the coil maps of a Cartesian unit, fitted to its own k-space centre where the stream stored none."""
        import torch

        try:
            return coil_maps(context, data, estimate=nlinv_estimate).reshape(
                buffer.kspace.shape
            )
        except MissingCalibration as missing:
            self._self_calibrated(context, data, missing)
            return nlinv_estimate(torch.from_numpy(np.ascontiguousarray(buffer.kspace)))


class FseSubspaceRecon(SubspaceRecon):
    """Echo images of a shuffled 3D fast spin echo (T2 shuffling, wave-shuffling) by :class:`SubspaceRecon`.

    Every refocusing pulse sits between two crushers: the train is a CPMG
    train whose configuration states shift by one order on each side of each
    refocusing pulse.
    """

    def interpret(self, description: Any) -> Any:
        from blochsim.sequence import EventAction, EventType, RfUse

        crushed = EventAction.CRUSH_BEFORE | EventAction.CRUSH_AFTER
        events = tuple(
            dataclasses.replace(event, action=crushed)
            if event.type == EventType.RF and event.params[1] == RfUse.REFOCUSING
            else event
            for event in description.events
        )
        return dataclasses.replace(description, events=events)


def _with_actions(description: Any, action_of: Any) -> Any:
    """Return ``description`` with each event's action replaced by ``action_of(index, event)``; ``None`` keeps it."""
    events = []
    for index, event in enumerate(description.events):
        action = action_of(index, event)
        events.append(
            event if action is None else dataclasses.replace(event, action=action)
        )
    return dataclasses.replace(description, events=tuple(events))


class MprageSubspaceRecon(SubspaceRecon):
    """Images of each excitation of a shuffled 3D MPRAGE shot (T1 shuffling) by :class:`SubspaceRecon`.

    Each readout is spoiled ideally: the transverse magnetization is gone
    before the next excitation, as RF spoiling with a gradient spoiler leaves
    it. The dictionary spans T1 and the transmit field, which sets both the
    inversion and the excitation flip.
    """

    def interpret(self, description: Any) -> Any:
        from blochsim.sequence import EventAction, EventType

        return _with_actions(
            description,
            lambda _, event: (
                EventAction.SPOIL_AFTER if event.type == EventType.ADC else None
            ),
        )

    def tissue(self) -> Any:
        import torch
        from blochsim.sequence import TissueProperties

        t1, b1 = torch.meshgrid(
            torch.logspace(np.log10(300.0), np.log10(5000.0), 64),
            torch.linspace(*B1),
            indexing="ij",
        )
        return TissueProperties(
            t1_ms=t1.reshape(-1),
            t2_ms=torch.full((t1.numel(),), 45.0),
            b1=b1.reshape(-1),
        )


class MegreSubspaceRecon(SubspaceRecon):
    """Echo images of a 3D multi-echo spoiled gradient echo by :class:`SubspaceRecon`.

    The echoes of one excitation share its transverse magnetization, which
    decays with T2 and T2' and precesses at the voxel's frequency; the last
    readout of each excitation is spoiled ideally. The dictionary spans T2*,
    through T2', and the frequency, from ``-frequency_hz`` to
    ``frequency_hz``.

    Parameters
    ----------
    frequency_hz : float, default=150.0
        Largest off-resonance the dictionary spans, in Hz.
    **kwargs
        As :class:`SubspaceRecon`.
    """

    def __init__(self, *args: Any, frequency_hz: float = 150.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.merge = ()
        self.axes = ("contrast",)
        self.frequency_hz = frequency_hz

    def interpret(self, description: Any) -> Any:
        from blochsim.sequence import EventAction, EventType

        readouts = [
            i for i, e in enumerate(description.events) if e.type == EventType.ADC
        ]
        excitations = [
            i for i, e in enumerate(description.events) if e.type == EventType.RF
        ]
        last = {
            max(
                (
                    r
                    for r in readouts
                    if r > x and all(not (x < y < r) for y in excitations)
                ),
                default=None,
            )
            for x in excitations
        } - {None}
        return _with_actions(
            description,
            lambda index, _: EventAction.SPOIL_AFTER if index in last else None,
        )

    def tissue(self) -> Any:
        import torch
        from blochsim.sequence import TissueProperties

        t2_prime = torch.logspace(np.log10(5.0), np.log10(500.0), 24)
        frequency = torch.linspace(-self.frequency_hz, self.frequency_hz, 31)
        t1, t2p, b0 = torch.meshgrid(
            torch.tensor((1200.0, 2000.0, 4400.0)), t2_prime, frequency, indexing="ij"
        )
        return TissueProperties(
            t1_ms=t1.reshape(-1),
            t2_ms=torch.full((t1.numel(),), 45.0),
            t2_prime_ms=t2p.reshape(-1),
            b0_hz=b0.reshape(-1),
        )


def _echoes(buffer: ReconBuffer, acquisitions: list[Any]) -> np.ndarray:
    """Return the echo of each phase encode ``(z, y)`` of ``buffer``, -1 where none was placed."""
    echo_of = np.full(buffer.mask.shape[:2], -1, np.int64)
    for acquisition in acquisitions:
        try:
            z, y = buffer.position(acquisition)
        except ValueError:
            continue
        if buffer.mask[z, y].any():
            echo_of[z, y] = int(acquisition.idx.contrast)
    return echo_of


def _by_echo(
    buffer: ReconBuffer, echo_of: np.ndarray, frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the phase encodes ``(frames, shots, 2)`` of each echo, -1 padded, and their readouts ``(coils, frames, shots, readout)``."""
    lines = [np.argwhere(echo_of == frame) for frame in range(frames)]
    shots = max(1, max(len(line) for line in lines))
    positions = np.full((frames, shots, 2), -1, np.int64)
    kspace = np.zeros(
        (buffer.kspace.shape[0], frames, shots, buffer.kspace.shape[-1]), np.complex64
    )
    for frame, line in enumerate(lines):
        positions[frame, : len(line)] = line
        kspace[:, frame, : len(line)] = buffer.kspace[:, line[:, 0], line[:, 1]]
    return positions, kspace


def _calibration_lines(data: ReconData, frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the calibration-only readouts of a unit by echo, at their phase encodes on the imaging grid, as :func:`_by_echo` does."""
    ref = data.ref
    alone = [a for a in data.acquisitions if readout_roles(a) == (False, True)]
    echo_of = _echoes(ref, alone)
    echo_of[echo_of >= frames] = -1
    positions, kspace = _by_echo(ref, echo_of, frames)
    placed = positions[..., 0] >= 0
    origin = np.array(
        [ref.origin.get("partition", 0), ref.origin.get("phase_encode", 0)]
    )
    positions[placed] += origin
    return positions, kspace


def _by_contrast(buffer: ReconBuffer, frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the phase encodes ``(frames, shots, 2)`` of each contrast of a buffer laid out by contrast, -1 padded, and their readouts ``(coils, frames, shots, readout)``."""
    lines = [np.argwhere(placed.any(axis=-1)) for placed in buffer.mask]
    shots = max(1, max(len(line) for line in lines))
    positions = np.full((frames, shots, 2), -1, np.int64)
    kspace = np.zeros(
        (buffer.kspace.shape[0], frames, shots, buffer.kspace.shape[-1]), np.complex64
    )
    for frame, line in enumerate(lines):
        positions[frame, : len(line)] = line
        kspace[:, frame, : len(line)] = buffer.kspace[:, frame, line[:, 0], line[:, 1]]
    return positions, kspace


def _first_contrast(buffer: ReconBuffer) -> Any:
    """Return ``buffer``, or its first contrast where it is laid out by contrast, as one volume."""
    if "contrast" not in buffer.axes:
        return buffer
    axis = buffer.axes.index("contrast")
    trajectory = buffer.trajectory
    return SimpleNamespace(
        kspace=np.take(buffer.kspace, 0, axis=axis),
        mask=np.take(buffer.mask, 0, axis=axis - 1),
        trajectory=None if trajectory is None else np.take(trajectory, 0, axis=axis),
        origin={k: v for k, v in buffer.origin.items() if k != "contrast"},
    )


def _wave_psf(data: ReconData, shape: tuple[int, ...]) -> Any:
    """Return the wave point-spread function ``(z, y, readout)`` of a wave-encoded unit.

    The wave is a wave-encoded readout's trajectory along the partition and
    phase encodes less that of the wave-free readout of its line. Where the
    unit is laid out by contrast, the gradient scale and delay are fitted to
    the first contrast's calibration region
    (:func:`bartorch.linop.wave_calibrate`); a train that reads the region one
    line per echo, or a region narrower than the wave's excursion, keeps the
    nominal wave.
    """
    from bartorch import linop

    wave, ref = _first_contrast(data.data), _first_contrast(data.ref)
    k = _wave_trajectory(wave, ref)
    fov = data.data.space.recon_fov[:2]
    scale, delay = (1.0, 1.0), (0.0, 0.0)
    if "contrast" in data.data.axes:
        with contextlib.suppress(ValueError):
            scale, delay = linop.wave_calibrate(
                _volume(ref, shape), _volume(wave, shape), k, fov
            )
    return linop.wave_psf(k, shape, fov, scale=scale, delay=delay, centred=True)


PLUGIN = FseSubspaceRecon()
