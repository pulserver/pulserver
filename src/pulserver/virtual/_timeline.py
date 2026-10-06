"""The blocks a cache plays, as a timeline along the logical axes: gradient moments, pulses and readouts."""

from __future__ import annotations

__all__ = ["Timeline"]

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pypulseqpp as pp
import torch

from .. import ir
from .._accelerators import require
from ._scanner import _on_a_raster

_EXCITATION = 1
_REFOCUSING = 2

# Detunings, per 1/T of a pulse of duration T, its spectral profile is found
# at, and their reach either side of its frequency, in 1/T.
_PROFILE_POINTS = 16.0
_PROFILE_REACH = 64.0

# Fraction of a pulse's flip angle on resonance from which a field lies in
# the band it selects, side lobes included.
_IN_BAND = 1.0 / 128.0

# Largest change of a gradient during a pulse, relative to its largest axis,
# at which it is held, and the fractions of the pulse it is read at.
_HELD = 1e-6
_HELD_AT = (0.0, 0.25, 0.5, 0.75, 1.0)

# Pathways a readout may read: the free induction, and the echoes of the
# intervals one and two before.
_PATHWAYS = 3

#: Samples of a readout its echo, pathway and reach are first read off,
#: evenly spaced, before its echo is found among the samples around the
#: nearest of them.
_COARSE = 64


@dataclass(frozen=True)
class Pulses:
    """The RF pulses a timeline plays, in play order.

    Attributes
    ----------
    block
        ``(n,)`` block each plays in.
    time_us
        ``(n,)`` its centre, in µs from the start of the scan.
    use
        ``(n,)`` the cache's RF use.
    flip
        ``(n,)`` the angle it turns the magnetization through on resonance, at
        its nominal amplitude, in rad.
    phase
        ``(n,)`` the phase of the axis it turns about, at its centre, in rad.
    gradient
        ``(n, 3)`` the gradient it plays under, along the physical axes, in
        Hz/m; zero where it plays under none or under one that changes.
    band
        ``(n, 2)`` the lowest and highest field, in Hz, it selects: ``g . r +
        f`` for an isochromat at ``r`` precessing at ``f`` under the gradient
        ``g``.
    profile
        ``(n,)`` which of ``profiles`` is its flip angle across its band.
    profiles
        Per profile, ``(detuning, ratio)``: detunings from the pulse's
        frequency, in Hz, and the angle it turns the magnetization at rest
        through at each, over the one on resonance.
    """

    block: np.ndarray
    time_us: np.ndarray
    use: np.ndarray
    flip: np.ndarray
    phase: np.ndarray
    gradient: np.ndarray
    band: np.ndarray
    profile: np.ndarray
    profiles: tuple[tuple[np.ndarray, np.ndarray], ...]


@dataclass(frozen=True)
class Readouts:
    """The readouts a timeline plays, in play order.

    Attributes
    ----------
    block
        ``(n,)`` block each plays in.
    first
        ``(n + 1,)`` index of each one's first sample among all samples, the
        last entry their total.
    echo
        ``(n,)`` the sample at which the pathway it reads passes nearest the
        centre of k-space.
    echo_us
        ``(n,)`` that sample's time, in µs from the start of the scan.
    receiver
        ``(n,)`` the receiver phase at that sample, in rad.
    reach
        ``(n, 3)`` the largest magnitude of its k-space locations along each
        logical axis, in 1/m; zero before the first excitation.
    pathway
        ``(n,)`` the pathway it reads: how many times the moment ``winding``
        its excitation's interval winds lies between the k the excitation
        encodes and the one the pathway does. Zero for the free induction,
        one for the echo of the interval before, as an SSFP-echo readout
        reads.
    winding
        ``(n, 3)`` the moment from its excitation to the next pulse, in 1/m
        along the logical axes; zero after a refocusing pulse.
    """

    block: np.ndarray
    first: np.ndarray
    echo: np.ndarray
    echo_us: np.ndarray
    receiver: np.ndarray
    reach: np.ndarray
    pathway: np.ndarray
    winding: np.ndarray

    def __len__(self) -> int:
        return int(self.block.size)


class Timeline:
    """The blocks the cache beside a sequence file plays, along the logical axes of a prescription.

    A block's gradients, its own rotation in them, are along the logical axes,
    except in blocks labelled ``NOROT``, whose gradients are along the
    physical axes and are turned back by ``rotation``. The moment of the
    gradients is integrated without the resets of an excitation, from the
    start of the scan.

    Parameters
    ----------
    seq_path
        The first file of the sequence's chain, beside its cache.
    cache_ext
        The extension of the cache beside each file.
    rotation
        ``(3, 3)`` rotation of the prescription from logical to physical axes;
        the identity by default.
    device
        Where the moments of many samples are integrated.

    Attributes
    ----------
    played
        The playout's blocks, as :func:`pulserver.ir.playout` records them.
    starts_us
        ``(blocks + 1,)`` each block's start, in µs, and the scan's end.
    pulses
        The pulses that turn the magnetization.
    readouts
        The readouts.
    """

    def __init__(
        self,
        seq_path: Path | str,
        cache_ext: str = ".pseg",
        *,
        rotation: np.ndarray | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        playout = ir.playout(
            Path(seq_path), waveforms=True, pulses=False, cache_ext=cache_ext
        )
        self.played = played = playout["blocks"]
        self._prepared = prepared = playout["positions"]
        width = (
            int(
                max(
                    played["position"].max(initial=0),
                    prepared["position"].max(initial=0),
                )
            )
            + 1
        )
        self._position = np.searchsorted(
            prepared["segment"].astype(np.int64) * width + prepared["position"],
            played["segment"].astype(np.int64) * width + played["position"],
        )
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation, float)
        self.device = torch.device(device)
        durations = played["duration_us"].astype(np.float64)
        self.starts_us = np.concatenate([[0.0], np.cumsum(durations)])
        self._gradients = require("fourier").GradientTable(
            played["gradient_time_us"],
            played["gradient_waveform_hz_per_m"],
            played["gradient_span"],
            self.starts_us,
            played["rotate"] == 0,
            self.rotation,
        )
        self._start_moment = self._gradients.start_moments()

        def put(values, dtype=torch.float64):
            return torch.as_tensor(np.asarray(values), dtype=dtype, device=self.device)

        self._starts = put(self.starts_us)
        self._start_moment_t = put(self._start_moment)
        self._adc = tuple(
            put(played[name])
            for name in ("adc_dwell_ns", "adc_delay_us", "adc_phase_rad", "adc_freq_hz")
        )
        modulated = played["adc_modulation_span"]
        if np.any(modulated[:, 1] > modulated[:, 0]):
            self._modulated = (
                put(modulated[:, 0], torch.int64),
                put(modulated[:, 1] - modulated[:, 0], torch.int64),
            )
            self._modulation = put(played["adc_phase_modulation_rad"])
        else:
            self._modulated = None
        self.pulses = self._pulses()
        self._origins, self._precession = self._origin()
        self._origins_t = torch.as_tensor(self._origins, device=self.device)
        pulses = self.pulses
        self._pulse_moments = torch.as_tensor(
            self.moment(pulses.block, pulses.time_us - self.starts_us[pulses.block])
            if pulses.block.size
            else np.zeros((1, 3)),
            device=self.device,
        )
        self._intervals, self._interval_us = self._interval()
        self._pulse_times = torch.as_tensor(self.pulses.time_us, device=self.device)
        self._pulse_files = torch.as_tensor(
            played["subsequence"][self.pulses.block].astype(np.int64),
            device=self.device,
        )
        self._files = torch.as_tensor(
            played["subsequence"].astype(np.int64), device=self.device
        )
        self.readouts = self._readouts()

    @property
    def blocks(self) -> int:
        """Blocks in the scan."""
        return int(self.played["duration_us"].size)

    def moment(self, block: np.ndarray, since_us: np.ndarray) -> np.ndarray:
        """Return the moment of the gradients from the start of the scan to times within blocks, ``(n, 3)`` in 1/m along the logical axes.

        Parameters
        ----------
        block
            ``(n,)`` blocks.
        since_us
            ``(n,)`` times from their starts, in µs.
        """
        return self._gradients.moment(
            np.asarray(block, dtype=np.int64), np.asarray(since_us, dtype=np.float64)
        )

    def _moment(self, block: torch.Tensor, since_us: torch.Tensor) -> torch.Tensor:
        """Return :meth:`moment` on the device, from device tensors."""
        moment = self._gradients.moment(block.cpu().numpy(), since_us.cpu().numpy())
        return torch.as_tensor(moment, device=self.device)

    def samples(
        self, first: int, last: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the samples of the readouts from ``first`` to before ``last``.

        Returns
        -------
        times_us
            ``(n,)`` each sample's time, in µs from the start of the scan.
        moment
            ``(n, 3)`` the moment at it, as :meth:`moment` integrates it.
        receiver
            ``(n,)`` the receiver phase at it, in rad: the ADC's phase offset
            at its start, advancing at its frequency offset, plus its phase
            modulation.
        """
        times, moment, receiver = self._samples(first, last)
        return times.cpu().numpy(), moment.cpu().numpy(), receiver.cpu().numpy()

    def _samples(
        self, first: int, last: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return :meth:`samples` on the device."""
        counts = torch.as_tensor(
            np.diff(self.readouts.first[first : last + 1]), device=self.device
        )
        readout = torch.repeat_interleave(
            torch.arange(first, last, device=self.device), counts
        )
        offsets = torch.cumsum(counts, 0) - counts
        index = torch.arange(readout.numel(), device=self.device) - (
            torch.repeat_interleave(offsets, counts)
        )
        return self._samples_at(readout, index)

    def _samples_at(
        self, readout: torch.Tensor, index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return :meth:`samples` at sample ``index`` of each ``readout``, on the device."""
        block = self._readout_blocks[readout]
        dwell_us = 1e-3 * self._adc[0][block]
        since = dwell_us * (index.to(torch.float64) + 0.5)
        sampled_us = self._adc[1][block] + since
        receiver = (
            self._adc[2][block] + 2.0 * math.pi * 1e-6 * self._adc[3][block] * since
        )
        if self._modulated is not None:
            begin, length = self._modulated
            inside = index < length[block]
            at = (begin[block] + index).clamp_max(self._modulation.numel() - 1)
            receiver = receiver + torch.where(inside, self._modulation[at], 0.0)
        return (
            self._starts[block] + sampled_us,
            self._moment(block, sampled_us),
            receiver,
        )

    def _pulses(self) -> Pulses:
        played = self.played
        span = self._prepared["rf_span"][self._position]
        blocks = np.flatnonzero(
            (span[:, 1] > span[:, 0]) & (played["rf_amp_hz"] != 0.0)
        )
        # Positions that prepare the same samples play the same pulse.
        prepared = self._prepared
        shapes: dict[bytes, int] = {}
        shape_of = np.array(
            [
                shapes.setdefault(
                    prepared["rf_time_us"][start:stop].tobytes()
                    + prepared["rf_waveform_hz"][start:stop].tobytes(),
                    len(shapes),
                )
                for start, stop in prepared["rf_span"]
            ],
            dtype=np.int64,
        )
        keys = np.column_stack(
            [
                shape_of[self._position[blocks]],
                played["rf_channels"][blocks],
                played["rf_amp_hz"][blocks].astype(np.float64).view(np.int64),
            ]
        )
        _, first, kind_of = np.unique(
            keys, axis=0, return_index=True, return_inverse=True
        )
        turns = [
            _turn(
                self._prepared,
                int(self._position[blocks[at]]),
                float(played["rf_amp_hz"][blocks[at]]),
                int(played["rf_channels"][blocks[at]]),
                float(played["rf_delay_us"][blocks[at]]),
            )
            for at in first
        ]
        kind_of = kind_of.reshape(-1)
        flips = np.array([turn[0] for turn in turns])[kind_of]
        axes = np.array([turn[1] for turn in turns])[kind_of]
        # Pulses of one shape at one amplitude share a profile wherever in the
        # repetition they play.
        shared: dict[bytes, int] = {}
        profile_of = np.array(
            [
                shared.setdefault(
                    turn[2][0].tobytes() + turn[2][1].tobytes(), len(shared)
                )
                for turn in turns
            ]
        )
        profiles = tuple(
            turns[at][2] for at in np.unique(profile_of, return_index=True)[1]
        )
        kind_of = profile_of[kind_of]
        low = np.array([profile[0][0] for profile in profiles])[kind_of]
        high = np.array([profile[0][-1] for profile in profiles])[kind_of]
        offset = played["rf_freq_hz"][blocks].astype(np.float64)
        centre = played["rf_center_us"][blocks].astype(np.float64)
        phase = (
            played["rf_phase_rad"][blocks].astype(np.float64)
            + 2.0 * math.pi * 1e-6 * offset * (centre - played["rf_delay_us"][blocks])
            + axes
        )
        return Pulses(
            block=blocks,
            time_us=self.starts_us[blocks] + centre,
            use=played["rf_use"][blocks].astype(np.int64),
            flip=flips,
            phase=phase,
            gradient=self._held(blocks),
            band=np.column_stack([low + offset, high + offset]),
            profile=kind_of,
            profiles=profiles,
        )

    def _held(self, blocks: np.ndarray) -> np.ndarray:
        """Return the gradient each block's pulse plays under along the physical axes, in Hz/m, ``(n, 3)``; zero where it changes during the pulse."""
        played = self.played
        span = self._prepared["rf_span"][self._position[blocks]]
        channels = np.maximum(played["rf_channels"][blocks], 1)
        times = self._prepared["rf_time_us"].astype(np.float64)
        first = times[span[:, 0]]
        last = times[span[:, 0] + (span[:, 1] - span[:, 0]) // channels - 1]
        samples = np.stack(
            [first + fraction * (last - first) for fraction in _HELD_AT], axis=1
        )
        values = self._values(np.repeat(blocks, len(_HELD_AT)), samples.ravel())
        values = values.reshape(blocks.size, len(_HELD_AT), 3)
        largest = np.abs(values).max(axis=(1, 2))
        changing = np.ptp(values, axis=1).max(axis=1) > _HELD * largest
        held = np.where((changing | (largest == 0.0))[:, None], 0.0, values[:, 0])
        turned = played["rotate"][blocks] != 0
        held[turned] = held[turned] @ self.rotation.T
        return held

    def _values(self, block: np.ndarray, since_us: np.ndarray) -> np.ndarray:
        """Return each axis's gradient at times from block starts, in the frame each block plays in, ``(n, 3)`` in Hz/m."""
        return self._gradients.value(
            np.asarray(block, dtype=np.int64), np.asarray(since_us, dtype=np.float64)
        )

    def _readouts(self) -> Readouts:
        played = self.played
        blocks = np.flatnonzero(played["adc"] != 0)
        counts = played["adc_samples"][blocks].astype(np.int64)
        pulses = self.pulses
        count = pulses.block.size
        files = played["subsequence"].astype(np.int64)
        echo, echo_us, reach, pathway, winding = require("fourier").read(
            self._gradients,
            pulses.time_us,
            files[pulses.block],
            pulses.use == _REFOCUSING,
            self._intervals[:count].cpu().numpy(),
            self._origins,
            blocks,
            files[blocks],
            counts,
            1e-3 * played["adc_dwell_ns"][blocks].astype(np.float64),
            played["adc_delay_us"][blocks].astype(np.float64),
            _COARSE,
            _PATHWAYS,
        )
        self._readout_blocks = torch.as_tensor(blocks, device=self.device)
        self._offsets = torch.as_tensor(
            pathway[:, None] * winding, dtype=torch.float64, device=self.device
        )
        receiver = self._samples_at(
            torch.arange(blocks.size, device=self.device),
            torch.as_tensor(echo, device=self.device),
        )[2]
        return Readouts(
            blocks,
            np.concatenate([[0], np.cumsum(counts)]),
            echo,
            echo_us,
            receiver.cpu().numpy(),
            reach,
            pathway,
            winding,
        )

    def _interval(self) -> tuple[torch.Tensor, np.ndarray]:
        """Return the moment over each pulse's interval, from it to the next pulse, ``(pulses, 3)``, and the interval's duration in µs."""
        pulses = self.pulses
        count = pulses.block.size
        if not count:
            return torch.zeros(
                (1, 3), dtype=torch.float64, device=self.device
            ), np.zeros(1)
        ahead = torch.cat([self._pulse_moments[1:], self._start_moment_t[-1:]])
        wound = torch.nan_to_num(ahead - self._origins_t[1:], nan=0.0)
        lasting = np.diff(np.append(pulses.time_us, self.starts_us[-1]))
        place = (
            self.played["segment"][pulses.block],
            self.played["position"][pulses.block],
        )
        alike = pulses.use[:-1] == pulses.use[-1]
        earlier = np.flatnonzero(
            alike & (place[0][:-1] == place[0][-1]) & (place[1][:-1] == place[1][-1])
        )
        if not earlier.size:
            earlier = np.flatnonzero(alike)
        if earlier.size:
            wound[-1] = wound[int(earlier[-1])]
            lasting[-1] = lasting[int(earlier[-1])]
        return wound, lasting

    def read_free_induction(self, readouts: np.ndarray) -> None:
        """Read the free induction at ``readouts``, whose intervals wind too little across a voxel to part its pathways."""
        self.readouts.pathway[readouts] = 0
        self._offsets[torch.as_tensor(readouts, device=self.device)] = 0.0

    def readout_unrefocused_us(self) -> np.ndarray:
        """Return how long the pathway each readout reads has precessed unrefocused at its echo, in µs.

        As :meth:`unrefocused_us` gives it at the echo, less an interval for
        each order the pathway lies behind the free induction: the echo of an
        earlier interval dephased through that interval before the pulse
        turned it back.
        """
        echo = self.readouts.echo_us
        last = self._last(echo)
        behind = np.where(last >= 0, self._interval_us[np.maximum(last, 0)], 0.0)
        return self.unrefocused_us(echo) - self.readouts.pathway * behind

    def kspace(
        self, first: int, last: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the samples of the readouts from ``first`` to before ``last`` with their k-space locations.

        As :meth:`samples`, with the moment less the one at the last
        excitation, an excitation returning k to zero and a refocusing pulse
        negating it at its centre; NaN before the first excitation of each
        file of a chain.
        """
        times, k, receiver = self._kspace(first, last)
        return times.cpu().numpy(), k.cpu().numpy(), receiver.cpu().numpy()

    def _kspace(
        self, first: int, last: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return :meth:`kspace` on the device."""
        times, moment, receiver = self._samples(first, last)
        k = moment - self._origins_t[self._last_t(times) + 1]
        if hasattr(self, "_offsets"):
            counts = torch.as_tensor(
                np.diff(self.readouts.first[first : last + 1]), device=self.device
            )
            k = k - torch.repeat_interleave(self._offsets[first:last], counts, dim=0)
        return times, k, receiver

    def unrefocused_us(self, times_us: np.ndarray) -> np.ndarray:
        """Return how long the magnetization sampled at times has precessed unrefocused, in µs: from the last excitation, mirrored about each refocusing pulse since; NaN before the first excitation of each file of a chain."""
        return times_us - self._precession[self._last(times_us) + 1]

    def _last_t(self, times_us: torch.Tensor) -> torch.Tensor:
        """Return :meth:`_last` on the device."""
        if not self._pulse_times.numel():
            return torch.full_like(times_us, -1, dtype=torch.int64)
        last = torch.searchsorted(self._pulse_times, times_us, right=True) - 1
        block = torch.searchsorted(self._starts, times_us, right=True) - 1
        file = self._files[block.clamp(0, self.blocks - 1)]
        own = (last >= 0) & (self._pulse_files[last.clamp_min(0)] == file)
        return torch.where(own, last, -1)

    def _last(self, times_us: np.ndarray) -> np.ndarray:
        """Return the last pulse at or before each time within its file, -1 where none."""
        pulses = self.pulses
        last = np.searchsorted(pulses.time_us, times_us, side="right") - 1
        block = np.searchsorted(self.starts_us, times_us, side="right") - 1
        file = self.played["subsequence"][np.clip(block, 0, self.blocks - 1)]
        own = (last >= 0) & (
            self.played["subsequence"][pulses.block[np.clip(last, 0, None)]] == file
        )
        return np.where(own, last, -1)

    def _origin(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the moment k is measured from, and the time precession is, after each pulse, ``(pulses + 1, ...)``; NaN before the first excitation of a file."""
        pulses = self.pulses
        moment = self.moment(
            pulses.block, pulses.time_us - self.starts_us[pulses.block]
        )
        return require("fourier").origins(
            moment,
            pulses.time_us,
            pulses.use,
            self.played["subsequence"][pulses.block].astype(np.int64),
            _EXCITATION,
            _REFOCUSING,
        )


def _turn(
    prepared: dict, at: int, amplitude: float, channels: int, delay_us: float
) -> tuple[float, float, tuple[np.ndarray, np.ndarray]]:
    """Return what the pulse prepared position ``at`` plays does at ``amplitude`` on resonance, and its flip angle across the band it selects.

    Returns
    -------
    flip
        The angle it turns the magnetization through, in rad.
    axis
        The phase of the transverse axis it turns about, in rad, zero for a
        real, positive envelope.
    profile
        Detunings from its frequency, in Hz, from the first to the last at
        which it turns the magnetization at rest through at least
        ``_IN_BAND`` of the angle it does on resonance, side lobes included,
        and that angle's ratio at each.
    """
    start, stop = prepared["rf_span"][at]
    channels = max(channels, 1)
    times = prepared["rf_time_us"][start:stop].astype(float).reshape(channels, -1)[0]
    b1 = amplitude * (
        prepared["rf_waveform_hz"][start:stop]
        .astype(complex)
        .reshape(channels, -1)
        .sum(axis=0)
    )
    b1, step_us = _on_a_raster(times - delay_us, b1)
    dt = 1e-6 * step_us
    turned = pp.sim_bloch(b1, np.zeros((3, 1)), dt, initial=np.eye(3)).T
    flip = float(np.arccos(np.clip(0.5 * (np.trace(turned) - 1.0), -1.0, 1.0)))
    if flip > math.pi - 1e-3:
        # A half turn is 2 n n^T - 1: its axis is any nonzero column of R + 1.
        columns = 0.5 * (turned + np.eye(3))
        axis_vector = columns[:, int(np.argmax(np.linalg.norm(columns, axis=0)))]
    else:
        axis_vector = np.array(
            [
                turned[2, 1] - turned[1, 2],
                turned[0, 2] - turned[2, 0],
                turned[1, 0] - turned[0, 1],
            ]
        )
    axis = float(np.arctan2(axis_vector[1], axis_vector[0])) if flip > 1e-6 else 0.0
    duration = step_us * 1e-6 * b1.size
    step = 1.0 / (_PROFILE_POINTS * duration)
    reach = math.ceil(_PROFILE_REACH * _PROFILE_POINTS)
    detunings = step * np.arange(-reach, reach + 1)
    tipped = np.arccos(
        np.clip(pp.sim_bloch(b1, detunings[:, None], dt)[:, 2], -1.0, 1.0)
    )
    ratio = tipped / max(float(tipped[reach]), 1e-12)
    inside = np.flatnonzero(ratio >= _IN_BAND)
    low, high = (int(inside[0]), int(inside[-1])) if inside.size else (reach, reach)
    return flip, axis, (detunings[low : high + 1], ratio[low : high + 1])
