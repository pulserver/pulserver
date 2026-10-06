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

# Samples whose k-space locations are integrated at once.
_CHUNK = 1 << 22

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
        self._corners = [self._axis(axis) for axis in range(3)]
        blocks = durations.size
        every = np.arange(blocks)
        areas = self._areas(every, durations) - self._areas(every, np.zeros(blocks))
        self._start_moment = np.concatenate(
            [np.zeros((1, 3)), np.cumsum(self._logical(every, areas), axis=0)]
        )

        def put(values, dtype=torch.float64):
            return torch.as_tensor(np.asarray(values), dtype=dtype, device=self.device)

        self._starts = put(self.starts_us)
        self._start_moment_t = put(self._start_moment)
        self._norot = put(played["rotate"] == 0, torch.bool)
        self._rotation = put(self.rotation)
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
        self._refocusing = torch.as_tensor(
            pulses.use == _REFOCUSING if pulses.block.size else np.zeros(1, bool),
            device=self.device,
        )
        self._intervals = self._interval()
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
        block = torch.as_tensor(np.asarray(block, dtype=np.int64), device=self.device)
        since = torch.as_tensor(
            np.asarray(since_us, dtype=np.float64), device=self.device
        )
        return self._moment(block, since).cpu().numpy()

    def _moment(self, block: torch.Tensor, since_us: torch.Tensor) -> torch.Tensor:
        """Return :meth:`moment` on the device, from device tensors."""
        starts = self._starts[block]
        within = self._areas_at(starts + since_us) - self._areas_at(starts)
        physical = self._norot[block]
        if bool(physical.any()):
            within[physical] = within[physical] @ self._rotation
        return self._start_moment_t[block] + within

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

    def _axis(self, axis: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one axis's gradient over the scan: corner times in µs, values in Hz/m and the moment at each corner.

        Each block's corners are bracketed by corners of zero at its first and
        last, so that the waveform is zero between blocks.
        """
        played = self.played
        span = played["gradient_span"][:, axis]
        lengths = span[:, 1] - span[:, 0]
        held = np.flatnonzero(lengths > 0)
        lengths = lengths[held]
        if not held.size:
            empty = torch.zeros(1, dtype=torch.float64, device=self.device)
            return empty, empty, empty
        padded = lengths + 2
        total = int(padded.sum())
        owner = np.repeat(held, padded)
        place = np.arange(total) - np.repeat(np.cumsum(padded) - padded, padded)
        source = np.repeat(span[held, 0], padded) + np.clip(place - 1, 0, None)
        source = np.minimum(source, np.repeat(span[held, 1] - 1, padded))
        times = self.starts_us[owner] + played["gradient_time_us"][source].astype(
            np.float64
        )
        values = played["gradient_waveform_hz_per_m"][source].astype(np.float64)
        edge = (place == 0) | (place == np.repeat(padded - 1, padded))
        values[edge] = 0.0
        steps = np.diff(times)
        moment = np.concatenate(
            [[0.0], np.cumsum(0.5 * (values[1:] + values[:-1]) * steps * 1e-6)]
        )
        put = lambda a: torch.as_tensor(a, dtype=torch.float64, device=self.device)  # noqa: E731
        return put(times), put(values), put(moment)

    def _areas(self, block: np.ndarray, since_us: np.ndarray) -> np.ndarray:
        """Return the moment of each axis's gradient from the start of the scan, in the frame each block plays in, ``(n, 3)``."""
        at = torch.as_tensor(
            self.starts_us[block] + since_us, dtype=torch.float64, device=self.device
        )
        return self._areas_at(at).cpu().numpy()

    def _areas_at(self, at: torch.Tensor) -> torch.Tensor:
        """Return :meth:`_areas` at times from the start of the scan, in µs, on the device."""
        out = torch.zeros((at.numel(), 3), dtype=torch.float64, device=self.device)
        for axis, (times, values, moment) in enumerate(self._corners):
            if times.numel() < 2:
                continue
            j = torch.searchsorted(times, at, right=True) - 1
            inside = (j >= 0) & (j < times.numel() - 1)
            after = j >= times.numel() - 1
            k = j.clamp(0, times.numel() - 2)
            elapsed = at - times[k]
            width = times[k + 1] - times[k]
            slope = torch.where(
                width > 0, (values[k + 1] - values[k]) / width.clamp_min(1e-30), 0.0
            )
            area = moment[k] + 1e-6 * (values[k] * elapsed + 0.5 * slope * elapsed**2)
            out[:, axis] = torch.where(
                inside, area, torch.where(after, moment[-1], torch.zeros_like(area))
            )
        return out

    def _logical(self, block: np.ndarray, areas: np.ndarray) -> np.ndarray:
        """Turn moments in the frame each block plays in to the logical axes."""
        physical = self.played["rotate"][block] == 0
        if not np.any(physical):
            return areas
        turned = areas.copy()
        turned[physical] = areas[physical] @ self.rotation
        return turned

    def _pulses(self) -> Pulses:
        played = self.played
        span = self._prepared["rf_span"][self._position]
        blocks = np.flatnonzero(
            (span[:, 1] > span[:, 0]) & (played["rf_amp_hz"] != 0.0)
        )
        keys = np.column_stack(
            [
                self._position[blocks],
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
        at = torch.as_tensor(
            self.starts_us[block] + since_us, dtype=torch.float64, device=self.device
        )
        out = torch.zeros((at.numel(), 3), dtype=torch.float64, device=self.device)
        for axis, (times, values, _) in enumerate(self._corners):
            if times.numel() < 2:
                continue
            j = (torch.searchsorted(times, at, right=True) - 1).clamp(
                0, times.numel() - 2
            )
            width = times[j + 1] - times[j]
            fraction = torch.where(
                width > 0, (at - times[j]) / width.clamp_min(1e-30), 0.0
            ).clamp(0.0, 1.0)
            inside = (at >= times[0]) & (at <= times[-1])
            value = values[j] + fraction * (values[j + 1] - values[j])
            out[:, axis] = torch.where(inside, value, 0.0)
        return out.cpu().numpy()

    def _readouts(self) -> Readouts:
        played = self.played
        blocks = np.flatnonzero(played["adc"] != 0)
        counts = played["adc_samples"][blocks].astype(np.int64)
        first = np.concatenate([[0], np.cumsum(counts)])
        self.readouts = Readouts(
            blocks,
            first,
            np.zeros(blocks.size, dtype=np.int64),
            np.zeros(blocks.size),
            np.zeros(blocks.size),
            np.zeros((blocks.size, 3)),
            np.zeros(blocks.size, dtype=np.int64),
            np.zeros((blocks.size, 3)),
        )
        self._offsets = torch.zeros(
            (blocks.size, 3), dtype=torch.float64, device=self.device
        )
        self._readout_blocks = torch.as_tensor(blocks, device=self.device)
        steps = np.maximum(1, -(-counts // _COARSE))
        coarse = (counts - 1) // steps + 2
        read = coarse + 2 * steps + 1
        bounds = np.concatenate([[0], np.cumsum(read)])
        start = 0
        while start < blocks.size:
            stop = max(
                start + 1, int(np.searchsorted(bounds, bounds[start] + _CHUNK)) - 1
            )
            device = self.device
            readout = torch.arange(start, stop, device=device)
            count = torch.as_tensor(counts[start:stop], device=device)
            step = torch.as_tensor(steps[start:stop], device=device)
            # Evenly spaced samples, and the last.
            per = torch.as_tensor(coarse[start:stop], device=device)
            owner = torch.repeat_interleave(
                torch.arange(stop - start, device=device), per
            )
            place = torch.arange(
                owner.numel(), device=device
            ) - torch.repeat_interleave(torch.cumsum(per, 0) - per, per)
            index = torch.minimum(place * step[owner], count[owner] - 1)
            times, moment, _ = self._samples_at(readout[owner], index)
            k = torch.nan_to_num(moment - self._origins_t[self._last_t(times) + 1])
            winding = self._winding(times[torch.cumsum(per, 0) - per])
            readouts = stop - start
            nearest = torch.full(
                (_PATHWAYS, readouts), math.inf, dtype=k.dtype, device=device
            )
            for n in range(_PATHWAYS):
                norm = torch.linalg.vector_norm(k - n * winding[owner], dim=1)
                nearest[n].scatter_reduce_(0, owner, norm, "amin")
            # A later pathway is read only where it passes nearer the centre
            # than the free induction does by half what an interval winds.
            margin = 0.5 * torch.linalg.vector_norm(winding, dim=1)
            nearest[1:] = torch.where(
                nearest[1:] < nearest[0] - margin, nearest[1:], math.inf
            )
            pathway = torch.argmin(nearest, dim=0)
            offset = pathway[:, None].to(k.dtype) * winding
            k = k - offset[owner]
            reach = torch.zeros((readouts, 3), dtype=k.dtype, device=device)
            reach.scatter_reduce_(0, owner[:, None].expand(-1, 3), k.abs(), "amax")
            centre = index[self._least(k, owner, readouts)]
            # The samples within a step of the nearest of those.
            width = 2 * step + 1
            owner = torch.repeat_interleave(
                torch.arange(readouts, device=device), width
            )
            place = torch.arange(
                owner.numel(), device=device
            ) - torch.repeat_interleave(torch.cumsum(width, 0) - width, width)
            index = (centre[owner] - step[owner] + place).clamp(
                torch.zeros_like(owner), count[owner] - 1
            )
            times, moment, phases = self._samples_at(readout[owner], index)
            k = torch.nan_to_num(moment - self._origins_t[self._last_t(times) + 1])
            k = k - offset[owner]
            reach.scatter_reduce_(0, owner[:, None].expand(-1, 3), k.abs(), "amax")
            nearest_sample = self._least(k, owner, readouts)
            held = self.readouts
            held.echo[start:stop] = index[nearest_sample].cpu().numpy()
            held.echo_us[start:stop] = times[nearest_sample].cpu().numpy()
            held.receiver[start:stop] = phases[nearest_sample].cpu().numpy()
            held.reach[start:stop] = reach.cpu().numpy()
            held.pathway[start:stop] = pathway.cpu().numpy()
            held.winding[start:stop] = winding.cpu().numpy()
            self._offsets[start:stop] = offset
            start = stop
        return self.readouts

    @staticmethod
    def _least(k: torch.Tensor, owner: torch.Tensor, count: int) -> torch.Tensor:
        """Return, per owner, the first of its rows of ``k`` nearest the centre."""
        norm = torch.linalg.vector_norm(k, dim=1)
        least = torch.full((count,), math.inf, dtype=k.dtype, device=k.device)
        least.scatter_reduce_(0, owner, norm, "amin")
        place = torch.arange(norm.numel(), device=k.device)
        hit = torch.where(norm <= least[owner], place, norm.numel())
        first = torch.full((count,), norm.numel(), device=k.device)
        return first.scatter_reduce_(0, owner, hit, "amin")

    def _winding(self, times_us: torch.Tensor) -> torch.Tensor:
        """Return the moment from the pulse before each time to the next one, ``(n, 3)``; zero after a refocusing pulse and before the first excitation.

        After the last pulse, the moment over the interval of the last earlier
        pulse of its use, at its place in the repetition where one is; to the
        end of the scan without one.
        """
        last = self._last_t(times_us)
        kept = (last >= 0) & ~self._refocusing[last.clamp_min(0)]
        return torch.where(kept[:, None], self._intervals[last.clamp_min(0)], 0.0)

    def _interval(self) -> torch.Tensor:
        """Return the moment over each pulse's interval, from it to the next pulse, ``(pulses, 3)``."""
        pulses = self.pulses
        count = pulses.block.size
        if not count:
            return torch.zeros((1, 3), dtype=torch.float64, device=self.device)
        ahead = torch.cat([self._pulse_moments[1:], self._start_moment_t[-1:]])
        wound = torch.nan_to_num(ahead - self._origins_t[1:], nan=0.0)
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
        return wound

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
        files = self.played["subsequence"][pulses.block]
        origins = np.full((pulses.block.size + 1, 3), np.nan)
        precession = np.full(pulses.block.size + 1, np.nan)
        origin, since = origins[0], np.nan
        for at, use in enumerate(pulses.use):
            if at and files[at] != files[at - 1]:
                origin, since = origins[0], np.nan
            if use == _EXCITATION:
                origin, since = moment[at], pulses.time_us[at]
            elif use == _REFOCUSING:
                origin = 2.0 * moment[at] - origin
                since = 2.0 * pulses.time_us[at] - since
            origins[at + 1] = origin
            precession[at + 1] = since
        return origins, precession


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
