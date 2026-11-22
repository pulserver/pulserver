"""Cine reconstruction of a 2D Cartesian balanced SSFP: cardiac phases binned from the physiological record, or real-time frames, solved together with bartorch."""

from __future__ import annotations

__all__ = [
    "PLUGIN",
    "CineRecon",
    "binned",
    "cardiac_phases",
    "cine_image",
    "realtime",
]

import copy
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...mrd._images import center_crop
from ...mrd._metadata import user_parameter
from .._buffers import ReconBuffer, ReconData
from .._calibration import MissingCalibration, coil_maps
from ..plugin import ReconContext, ReconResult
from .pics import PicsRecon, compressed, nlinv_estimate, wavelet_axes

#: Physiological channel of each ``physiology_time_stamp`` entry, in the order
#: the scanner client writes them.
CHANNELS = ("ecg", "ppg", "respiratory")

#: Cardiac phases a retrospective or free-running cine is binned into when the
#: sequence states none.
CARDIAC_PHASES = 25

#: Fraction of the median beat-to-beat interval a beat may differ from it by
#: before its readouts are rejected as arrhythmic.
ARRHYTHMIA_REJECTION = 0.2

#: Undersampling of a real-time frame of a golden free-running series: a frame
#: is this fraction of the lines of a fully sampled frame, in consecutive
#: readouts.
REALTIME_ACCELERATION = 8
TEMPORAL = 0.01
REALTIME_TEMPORAL = 0.1

_log = logging.getLogger("pulserver.recon")


def cardiac_phases(
    times: np.ndarray,
    beats: np.ndarray,
    rejection: float = ARRHYTHMIA_REJECTION,
) -> tuple[np.ndarray, float]:
    """Return each readout's place in its beat, from 0 to 1, and the median beat-to-beat interval.

    Each readout's beat is the last trigger at or before it, as the scanner
    stamps it; its interval runs to the next trigger any readout was stamped
    with. A beat whose interval differs from the median by more than
    ``rejection`` of it, the last beat when its readouts lie further past it
    than the median allows, and readouts stamped before the first trigger, are
    rejected: their place is ``nan``. Times and triggers are in the same unit.

    Parameters
    ----------
    times : ndarray
        Acquisition time of each readout.
    beats : ndarray
        Last trigger at or before each readout; 0 where none had arrived.
    rejection : float, default=ARRHYTHMIA_REJECTION
        Fraction of the median interval a beat may differ from it by.

    Returns
    -------
    phases : ndarray
        Place of each readout in its beat, in ``[0, 1)``, ``nan`` where rejected.
    rr : float
        Median beat-to-beat interval, in the unit of ``times``.

    Raises
    ------
    ValueError
        If fewer than two distinct triggers were stamped.

    Examples
    --------
    >>> import numpy as np
    >>> times = np.array([100.0, 600.0, 1100.0, 1600.0, 1900.0, 2100.0])
    >>> beats = np.array([0.0, 500.0, 1000.0, 1500.0, 1500.0, 1500.0])
    >>> phases, rr = cardiac_phases(times, beats)
    >>> phases.round(2).tolist(), rr
    ([nan, 0.2, 0.2, 0.2, 0.8, nan], 500.0)
    """
    times = np.asarray(times, dtype=float)
    beats = np.asarray(beats, dtype=float)
    triggers = np.unique(beats[beats > 0])
    if triggers.size < 2:
        raise ValueError(
            f"{triggers.size} trigger(s) were stamped on the readouts; binning "
            "cardiac phases needs at least two"
        )
    intervals = np.diff(triggers)
    rr = float(np.median(intervals))
    which = np.searchsorted(triggers, beats)
    which = np.clip(which, 0, triggers.size - 1)
    known = beats > 0
    interval = np.full(times.shape, rr)
    inner = known & (which < intervals.size)
    interval[inner] = intervals[which[inner]]
    phases = (times - beats) / interval
    regular = np.abs(interval - rr) <= rejection * rr
    accepted = known & regular & (phases >= 0.0) & (phases < 1.0)
    phases = np.where(accepted, phases, np.nan)
    return phases, rr


def binned(
    phases: np.ndarray,
    lines: np.ndarray,
    n_lines: int,
    n_phases: int,
    *,
    interpolate: bool,
    reach: float | None = None,
) -> np.ndarray:
    """Return the weight of each readout in each cardiac phase and line, ``(n_phases, n_lines, readouts)``.

    Cardiac phase ``j`` is centred on ``j / n_phases`` of the beat. With
    ``interpolate``, a line takes the two of its readouts nearest the phase
    on either side, around the beat, weighted linearly by their distance,
    as a segmented retrospective cine interpolates its views, where they lie
    no further apart than ``reach``; across a wider gap, left by rejected
    beats, the nearest readout within one phase, else none. Without, it takes
    every readout of the line within one phase of the centre, weighted by a
    triangle of that half width; a line with none in reach stays empty, for
    the solve to fill. Readouts whose phase is ``nan`` weigh nothing.

    Parameters
    ----------
    phases : ndarray
        Place of each readout in its beat, from :func:`cardiac_phases`.
    lines : ndarray
        Line of each readout, from 0 to ``n_lines - 1``.
    n_lines : int
        Lines of the grid.
    n_phases : int
        Cardiac phases.
    interpolate : bool
        Interpolate between the nearest readouts of each line.
    reach : float, default=None
        Widest gap, as a fraction of the beat, interpolated across; ``None``
        is two phases.

    Returns
    -------
    ndarray
        Weights; those of a phase and line sum to 1 where any readout reaches.
    """
    phases = np.asarray(phases, dtype=float)
    lines = np.asarray(lines)
    weights = np.zeros((n_phases, n_lines, phases.size), dtype=np.float32)
    centres = np.arange(n_phases) / n_phases
    reach = 2.0 / n_phases if reach is None else reach
    for line in range(n_lines):
        taken = np.flatnonzero((lines == line) & np.isfinite(phases))
        if taken.size == 0:
            continue
        where = phases[taken]
        if interpolate:
            order = np.argsort(where)
            taken, where = taken[order], where[order]
            for j, centre in enumerate(centres):
                after = np.searchsorted(where, centre) % taken.size
                before = (after - 1) % taken.size
                gap = (where[after] - where[before]) % 1.0
                past = (centre - where[before]) % 1.0
                if taken.size == 1 or gap == 0.0 or gap > reach:
                    # No pair brackets the phase closely: the nearest
                    # readout within one phase, else none.
                    distance = np.abs((where - centre + 0.5) % 1.0 - 0.5)
                    nearest = int(np.argmin(distance))
                    if distance[nearest] <= 1.0 / n_phases:
                        weights[j, line, taken[nearest]] = 1.0
                    continue
                weights[j, line, taken[before]] += 1.0 - past / gap
                weights[j, line, taken[after]] += past / gap
            continue
        distance = np.abs((where[None, :] - centres[:, None] + 0.5) % 1.0 - 0.5)
        kernel = np.clip(1.0 - distance * n_phases, 0.0, None)
        total = kernel.sum(axis=1, keepdims=True)
        kernel = np.divide(kernel, total, out=np.zeros_like(kernel), where=total > 0)
        weights[:, line, taken] = kernel
    return weights


def realtime(order: np.ndarray, window: int) -> np.ndarray:
    """Return the real-time frame of each readout: consecutive runs of ``window`` readouts in acquisition order.

    Readouts past the last whole frame join it.

    Examples
    --------
    >>> realtime(np.arange(10), 4)
    array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    """
    order = np.asarray(order)
    frames = max(1, order.size // window)
    return np.minimum(order // window, frames - 1)


@dataclass
class Readouts:
    """The readouts of one slice of a cine, in acquisition order.

    Attributes
    ----------
    kspace : ndarray
        ``(readouts, coils, samples)``, placed along the readout as the
        buffer places them.
    lines : ndarray
        Phase-encode position of each readout on the buffer's grid.
    times : ndarray
        ``acquisition_time_stamp`` of each.
    physiology : ndarray
        ``(readouts, 3)`` ``physiology_time_stamp`` of each.
    phase : ndarray
        The ``PHS`` counter of each.
    headers : list
        The acquisitions, in acquisition order.
    """

    kspace: np.ndarray
    lines: np.ndarray
    times: np.ndarray
    physiology: np.ndarray
    phase: np.ndarray
    headers: list

    @classmethod
    def of(cls, buffer: ReconBuffer) -> Readouts:
        """Return the readouts ``buffer`` holds, each taken from where it was placed."""
        axes = buffer.axes
        line_axis = axes.index("phase_encode") - 1
        rows, lines, times, physiology, phase = [], [], [], [], []
        for header in buffer.headers:
            position = buffer.position(header)
            rows.append(buffer.kspace[(slice(None), *position)])
            lines.append(position[line_axis])
            times.append(float(header.acquisition_time_stamp))
            physiology.append([float(v) for v in header.physiology_time_stamp[:3]])
            phase.append(int(header.idx.phase))
        order = np.argsort(np.asarray(times), kind="stable")
        return cls(
            kspace=np.asarray(rows)[order],
            lines=np.asarray(lines)[order],
            times=np.asarray(times)[order],
            physiology=np.asarray(physiology).reshape(-1, 3)[order],
            phase=np.asarray(phase)[order],
            headers=[buffer.headers[i] for i in order],
        )


def cine_image(
    kspace: Any,
    maps: Any,
    spatial: float,
    temporal: float,
    iterations: int,
    device: str | None,
) -> Any:
    """Return the complex frames ``(frames, y, x)`` that fit k-space ``(coils, frames, y, x)`` over the maps.

    Minimises ``|P F S x - y|^2 + spatial |W x|_1 + temporal |D_t x|_1`` by
    ADMM, ``P`` the lines placed in each frame, ``W`` the wavelet transform
    over the image axes and ``D_t`` the finite difference between frames,
    with the weights relative to the data scaling estimated from the frames'
    mean k-space (:func:`bartorch.optim.data_scaling`). The frames share the
    coil maps of one :func:`bartorch.linop.CartesianSense`, each sampled on
    its own lines. The encoding computes on ``device``.
    """
    import bartorch
    import torch
    from bartorch import linop, optim, priors

    coils, frames, ny, nx = kspace.shape
    sampled = kspace.abs().sum(dim=(0, 3)) > 0
    pattern = sampled[:, None, :, None].to(torch.complex64)
    sense = linop.CartesianSense(
        maps.reshape(coils, ny, nx), (frames, ny, nx), device=device
    )
    encoding = linop.Diagonal(pattern, sense.oshape) @ sense
    measured = kspace.permute(1, 0, 2, 3) * pattern
    counts = sampled.sum(dim=0).clamp(min=1)
    mean = kspace.sum(dim=1) / counts[None, :, None].to(kspace.dtype)
    scaling = optim.data_scaling(
        bartorch.fftmod(mean[:, None], axes=(-1, -2), inverse=True)
    )
    scaling = scaling or 1.0
    terms = [priors.Wavelet(wavelet_axes((ny, nx), (-1, -2)), spatial)]
    if frames > 1 and temporal > 0:
        terms.append(priors.TotalVariation((0,), temporal))
    solve = optim.ADMM(terms, maxiter=iterations, cg_maxiter=10)
    return scaling * solve(measured * (1.0 / scaling), encoding)


class CineRecon(PicsRecon):
    """Cine of each slice of a 2D Cartesian scan: every frame of the slice solved together once the slice closes.

    The sequence states how the frames are told apart, in the recon
    parameters of the header (:func:`~pulserver.design.recon_parameters`):

    ``Gating`` ``prospective``, or ``none`` with ``Ordering`` ``sequential``
        Each frame is a ``PHS`` counter: a cardiac phase after the trigger,
        or a frame of a free-running series. Where the triggers were
        recorded, a heartbeat off the median is stretched to the median beat
        and its readouts after the next trigger dropped.
    ``Gating`` ``retrospective``
        Every readout is placed in its beat by the physiological triggers it
        is stamped with (:func:`cardiac_phases`), beats off the median
        rejected, and each line of each of ``CardiacPhases`` phases
        interpolated from its two nearest readouts (:func:`binned`). A phase
        of a line no regular beat reaches takes the rejected beats' readouts.
    ``Gating`` ``none`` with ``Ordering`` ``golden``
        Real-time frames of ``1 / realtime_acceleration`` of the lines in
        consecutive readouts (:func:`realtime`); with ``binning``, cardiac
        phases instead, each line taking the readouts within one phase of its
        centre.

    The readouts run through the gadgets of
    :class:`~pulserver.recon.handlers.pics.PicsRecon`, compressed to
    ``virtual_coils`` virtual channels. The coil sensitivities are those
    :func:`~pulserver.recon.coil_maps` stores for the slice, else fitted to the
    centre of the slice's k-space averaged over every readout
    (:func:`~pulserver.recon.handlers.pics.nlinv_estimate`). The frames are
    solved together by :func:`cine_image`, Fermi filtered, corrected for
    gradient nonlinearity and the receive field as ``pics`` corrects its
    images, and sent as the images of one slice, each frame its ``PHS``.
    The trigger time of a cardiac phase is its centre in the median beat.

    Parameters
    ----------
    spatial : float, default=0.002
        Weight of the wavelet term, relative to the data scaling.
    temporal : float or None, default=None
        Weight of the temporal total variation, relative to the data scaling;
        ``None`` takes ``REALTIME_TEMPORAL`` for real-time frames and
        ``TEMPORAL`` for cardiac phases.
    iterations : int, default=60
        Conjugate-gradient iterations the ADMM solve spends in all.
    binning : str or None, default=None
        Physiological channel, ``'ecg'``, ``'ppg'`` or ``'respiratory'``, a
        golden free-running series is binned by; ``None`` makes real-time
        frames. A retrospective cine takes the ECG, else the first channel
        stamped.
    realtime_acceleration : int, default=REALTIME_ACCELERATION
        Undersampling of a real-time frame.
    rejection : float, default=ARRHYTHMIA_REJECTION
        Fraction of the median beat a beat may differ from it by.
    virtual_coils : int | None, default=VIRTUAL_COILS
        Receive channels beyond this many are compressed to it.
    kspace_filter, gradient_unwarp, intensity_correction : bool, default=True
        The corrections of :class:`~pulserver.recon.Corrections`.
    """

    def __init__(
        self,
        spatial: float = 0.002,
        temporal: float | None = None,
        iterations: int = 60,
        binning: str | None = None,
        realtime_acceleration: int = REALTIME_ACCELERATION,
        rejection: float = ARRHYTHMIA_REJECTION,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.axes = ("phase",)
        if binning is not None and binning not in CHANNELS:
            raise ValueError(
                f"binning must be one of {CHANNELS} or None, got {binning!r}"
            )
        self.spatial = spatial
        self.temporal = temporal
        self.cine_iterations = iterations
        self.binning = binning
        self.realtime_acceleration = int(realtime_acceleration)
        self.rejection = rejection

    def recon(
        self, context: ReconContext, branch: str, data: ReconData
    ) -> list[ReconResult] | None:
        del branch
        data = compressed(context, data, self.virtual_coils)
        if data.data is None and data.ref is not None:
            coil_maps(context, data, estimate=nlinv_estimate)
        buffer = data.data
        if buffer is None:
            return None
        frames, references = self.frames(context, data)
        corrections = self.corrections
        history = ["PULSERVER", "PYTHON", "CINE"]
        results = []
        for frame, reference in zip(frames, references, strict=True):
            image = frame
            if corrections is not None:
                image = corrections.unwarped(image, context, data)
            results.append(
                ReconResult(
                    np.abs(image),
                    reference=reference,
                    image_index=None,
                    attributes={
                        "ImageProcessingHistory": history
                        + (corrections.applied() if corrections is not None else [])
                    },
                )
            )
        if corrections is not None:
            results = corrections.intensity_corrected(results, data)
        return results

    def frames(self, context: ReconContext, data: ReconData) -> tuple[np.ndarray, list]:
        """Return the slice's complex frames ``(frames, y, x)``, filtered, and the acquisition each takes its header from."""
        import torch

        buffer = data.data
        readouts = Readouts.of(buffer)
        gating = str(user_parameter(context.header, "Gating", "none"))
        ordering = str(user_parameter(context.header, "Ordering", "sequential"))
        n_phases = int(
            float(user_parameter(context.header, "CardiacPhases", CARDIAC_PHASES))
        )
        n_lines = buffer.kspace.shape[buffer.axes.index("phase_encode")]
        trigger_times = None
        temporal = TEMPORAL if self.temporal is None else self.temporal
        if gating == "retrospective" or (ordering == "golden" and self.binning):
            channel = self._channel(readouts.physiology)
            phases, rr = cardiac_phases(
                readouts.times, readouts.physiology[:, channel], self.rejection
            )
            kept = np.isfinite(phases).mean()
            _log.info(
                "slice %d: %s binned into %d phases of a %.0f beat, %.0f %% of "
                "the readouts kept",
                int(data.counters.get("slice", 0)),
                CHANNELS[channel],
                n_phases,
                rr,
                100.0 * kept,
            )
            interpolate = gating == "retrospective"
            weights = binned(
                phases, readouts.lines, n_lines, n_phases, interpolate=interpolate
            )
            # A cardiac phase of a line no regular beat reaches takes the
            # rejected beats' readouts, each placed in its own beat: a line
            # missing from every phase is beyond the temporal regulariser.
            every, _ = cardiac_phases(
                readouts.times, readouts.physiology[:, channel], np.inf
            )
            rejected = np.where(np.isnan(phases), every, np.nan)
            empty = weights.sum(axis=2) == 0
            if empty.any() and np.isfinite(rejected).any():
                fallback = binned(
                    rejected, readouts.lines, n_lines, n_phases, interpolate=interpolate
                )
                weights[empty] = fallback[empty]
            frame, line, readout = np.nonzero(weights)
            taken = (
                weights[frame, line, readout][:, None, None] * readouts.kspace[readout]
            )
            stacked = np.zeros(
                (n_phases, n_lines, *readouts.kspace.shape[1:]), dtype=np.complex64
            )
            np.add.at(stacked, (frame, line), taken)
            kspace = stacked.transpose(2, 0, 1, 3)
            trigger_times = (np.arange(n_phases) / n_phases) * rr
            labels = np.arange(n_phases)
        elif gating == "none" and ordering == "golden":
            if self.temporal is None:
                temporal = REALTIME_TEMPORAL
            window = max(1, n_lines // self.realtime_acceleration)
            frame = realtime(np.arange(readouts.lines.size), window)
            labels = np.arange(frame.max() + 1)
            kspace = np.zeros(
                (
                    readouts.kspace.shape[1],
                    labels.size,
                    n_lines,
                    readouts.kspace.shape[2],
                ),
                dtype=np.complex64,
            )
            counts = np.zeros((labels.size, n_lines), dtype=np.float32)
            np.add.at(counts, (frame, readouts.lines), 1.0)
            for r, (f, y) in enumerate(zip(frame, readouts.lines, strict=True)):
                kspace[:, f, y] += readouts.kspace[r]
            kspace /= np.maximum(counts, 1.0)[None, :, :, None]
        else:
            labels = np.unique(readouts.phase)
            if gating == "prospective":
                readouts = self._regular_beats(readouts, data)
            kspace = np.zeros(
                (
                    readouts.kspace.shape[1],
                    labels.size,
                    n_lines,
                    readouts.kspace.shape[2],
                ),
                dtype=np.complex64,
            )
            for r, (p, y) in enumerate(
                zip(readouts.phase, readouts.lines, strict=True)
            ):
                kspace[:, np.searchsorted(labels, p), y] = readouts.kspace[r]
        volume = torch.from_numpy(np.ascontiguousarray(kspace, dtype=np.complex64))
        maps = self._maps(context, data, volume)
        image = cine_image(
            volume,
            maps,
            self.spatial,
            temporal,
            self.cine_iterations,
            context.device,
        ).numpy()
        image = np.array(center_crop(image, buffer.image_shape[-2:]))
        image = np.stack([self._filtered(frame) for frame in image])
        references = [
            self._reference(
                readouts,
                int(label),
                None if trigger_times is None else trigger_times[i],
            )
            for i, label in enumerate(labels)
        ]
        return image, references

    def _regular_beats(self, readouts: Readouts, data: ReconData) -> Readouts:
        """Return the readouts of a prospective cine in the cardiac phase they image, where the triggers were recorded.

        A heartbeat's readouts run in rising ``PHS`` from the trigger that
        opened them. Readouts stamped with a later trigger lie in the next
        heartbeat and are dropped. A heartbeat whose interval differs from
        the median by more than ``rejection`` of it has its phases stretched
        to the median beat, as the retrospective cine places its readouts,
        and loses those stretched past the last phase.
        """
        stamped = [
            c
            for c in range(3)
            if np.unique(readouts.physiology[:, c][readouts.physiology[:, c] > 0]).size
            > 1
        ]
        if not stamped:
            return readouts
        channel = CHANNELS.index(self.binning) if self.binning else stamped[0]
        stamps = readouts.physiology[:, channel]
        opened = np.r_[True, np.diff(readouts.phase) < 0]
        opening = stamps[np.flatnonzero(opened)[np.cumsum(opened) - 1]]
        kept = stamps == opening
        beats = np.unique(stamps[stamps > 0])
        intervals = np.diff(beats)
        median = np.median(intervals)
        following = np.searchsorted(beats, opening, side="right")
        interval = np.full(stamps.shape, median)
        known = following < beats.size
        interval[known] = beats[following[known]] - opening[known]
        irregular = np.abs(interval - median) > self.rejection * median
        phase = readouts.phase.copy()
        phase[irregular] = np.rint(phase[irregular] * median / interval[irregular])
        kept &= phase <= readouts.phase.max()
        _log.info(
            "slice %d: %d of %d readouts in heartbeats off the median %s interval "
            "restretched, %d past the next trigger or the last phase dropped",
            int(data.counters.get("slice", 0)),
            int(irregular.sum()),
            kept.size,
            CHANNELS[channel],
            int((~kept).sum()),
        )
        return Readouts(
            readouts.kspace[kept],
            readouts.lines[kept],
            readouts.times[kept],
            readouts.physiology[kept],
            phase[kept],
            [h for h, k in zip(readouts.headers, kept, strict=True) if k],
        )

    def _channel(self, physiology: np.ndarray) -> int:
        """Return the physiological channel binned by: ``binning``, else the ECG, else the first stamped."""
        if self.binning is not None:
            return CHANNELS.index(self.binning)
        stamped = [
            c
            for c in range(3)
            if np.unique(physiology[:, c][physiology[:, c] > 0]).size > 1
        ]
        if not stamped:
            raise ValueError(
                "no physiological channel was stamped on the readouts; a retrospective "
                "cine needs the trigger record (opuser1)"
            )
        return stamped[0]

    def _maps(self, context: ReconContext, data: ReconData, volume: Any) -> Any:
        """Return the slice's coil maps: those stored for it, else fitted to its k-space averaged over the frames."""
        import torch

        averaged = volume.sum(dim=1)
        sampled = (volume.abs().sum(dim=(0, 3)) > 0).sum(dim=0).clamp(min=1)
        averaged = (averaged / sampled[None, :, None].to(averaged.dtype))[:, None]
        try:
            return coil_maps(context, data, estimate=nlinv_estimate).reshape(
                averaged.shape
            )
        except MissingCalibration:
            return nlinv_estimate(averaged.to(torch.complex64))

    @staticmethod
    def _reference(readouts: Readouts, phase: int, trigger_time: float | None) -> Any:
        """Return an acquisition header of the slice for frame ``phase``, with its trigger time."""
        header = readouts.headers[len(readouts.headers) // 2]
        reference = copy.deepcopy(header)
        reference.idx.phase = phase
        if trigger_time is not None:
            stamps = list(reference.physiology_time_stamp)
            stamps[0] = round(trigger_time * 2.5)
            reference.physiology_time_stamp[:] = stamps
        return reference


PLUGIN = CineRecon()
