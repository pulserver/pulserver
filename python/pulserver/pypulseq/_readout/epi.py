"""Echo-planar imaging (EPI) readout trains.

Four callable readout objects, built once and then called per shot to append a
blipped or flyback EPI echo train to a sequence:

* :class:`Epi2D` / :class:`Epi3D` -- blipped, ramp-sampled, split-gradient
  overlapped EPI (after the pypulseq ``write_epi_se_rs`` "high performance"
  demo). :class:`Epi3D` additionally blips the partition axis every line
  (a true 3D-EPI train, e.g. for blipped-CAIPI 3D acquisitions).
* :class:`Epi2DFlyback` / :class:`Epi3DFlyback` -- flyback EPI: every line
  has the *same* readout polarity, separated by a dedicated rewind gradient
  that carries the phase-encode/partition blip.

All four are *train only*: the excitation (and, for spin-echo EPI, the
refocusing pulse -- see :mod:`pulserver.pypulseq._readout.fse` for
``build_refocusing_pulse``) live in the caller. The train has no notion of
"slice" vs. "slab"; it is purely the gradient-echo readout following whatever
RF the caller played.

Sampling-mask contract
-----------------------
The constructor takes a mandatory ``sampling_mask``: the *within-shot* k-space
line-index pattern, in the exact order the train should visit it -- shape
``(etl,)`` (``ky`` index per line) for :class:`Epi2D`/:class:`Epi2DFlyback`,
or ``(etl, 2)`` (``(ky, kz)`` index pairs) for :class:`Epi3D`/
:class:`Epi3DFlyback`. Sorting the mask into the desired visiting order
(centric, interleaved, ...) is the caller's job -- the train just walks it.
Blip amplitudes are the *index deltas* between consecutive mask entries,
scaled from a single base blip built at the worst-case (largest) jump in each
direction, so every blip in the train shares identical ramp timing regardless
of its area -- required for the split-gradient overlap trick in the blipped
trains. ``__call__`` itself only takes the *absolute* starting index (a
scalar, or a ``(ky, kz)`` tuple for the 3D trains) -- the position of
``sampling_mask[0]`` -- which is folded into the same prewinder block that
positions ``kx`` at the start of the train (never a separate gradient).

Multi-shot interleaving (e.g. two odd/even sub-trains played as separate
shots) is the caller's job too: build one train object and derive the other
with :meth:`_EpiTrainBlipped.with_sampling_mask` /
:meth:`_EpiTrainFlyback.with_sampling_mask` rather than constructing a second
object from scratch, so both share numerically identical base gradients.

Every ADC block carries absolute ``SET LIN``/``SET PAR`` labels. By default
the labels are reconstructed from the starting index and relative sampling
mask; ``set_state(labels=...)`` accepts an explicit absolute schedule when
gradient shifts and reconstruction labels are supplied separately.

Rotation
--------
``__call__`` accepts an optional ``rotation`` (a scipy-``Rotation``-like
object, see :func:`pulserver.pypulseq.make_rotation`), attached to every block
the train emits. The readout gradients themselves stay axis-aligned; the
``ROTATIONS`` extension re-maps them downstream -- the same convention used
by the noncartesian zoo plugins (e.g. ``zte_3d.py``). This is what lets a
caller rotate a whole shot in-plane for e.g. a PROPELLER-EPI blade.
"""

from __future__ import annotations

__all__ = [
    "Epi2D",
    "Epi3D",
    "Epi2DFlyback",
    "Epi3DFlyback",
]

import copy
import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pypulseq as pp

from .._system import (
    DEFAULT_BANDWIDTH_HZ_PX,
    apply_system_derates,
    ceil_to_raster,
    quantize_readout_timing,
    round_to_raster,
)
from ._base import Readout, normalize_rotation

READOUT_GRAD_MARGIN = 0.95  # headroom below max_grad for the readout plateau


@dataclass(frozen=True)
class _EpiState:
    pe_start: int | tuple[int, int]
    rotation: object | None
    labels: np.ndarray | None = None


def _floor_to_raster(value_s: float, raster_s: float) -> float:
    """Round ``value_s`` down to the next multiple of ``raster_s``."""
    return math.floor(value_s / raster_s + 1e-10) * raster_s


def _axis_areas(fov_m, n_points):
    """delta_k (1/m) and the symmetric-about-center per-index area array."""
    delta_k = 1.0 / float(fov_m)
    areas = (np.arange(int(n_points)) - 0.5 * int(n_points)) * delta_k
    return delta_k, areas


def _make_axis(spec):
    """Build the ``(axis, delta_k, areas, label)`` bundle from a ``pe_spec``/``par_spec`` 4-tuple."""
    if spec is None:
        return None
    axis, fov_m, n_points, label = spec
    delta_k, areas = _axis_areas(fov_m, n_points)
    return SimpleNamespace(axis=axis, n=int(n_points), label=label, delta_k=delta_k, areas=areas)


def _parse_mask(sampling_mask, has_par):
    """Split ``sampling_mask`` into ``(mask_pe, mask_par)`` int index arrays (``mask_par`` is ``None`` in 2D)."""
    mask = np.asarray(sampling_mask, dtype=int)
    if has_par:
        if mask.ndim != 2 or mask.shape[1] != 2:
            raise ValueError(
                "sampling_mask must have shape (etl, 2) -- (ky_idx, kz_idx) per line -- "
                "when a partition axis is configured"
            )
        return mask[:, 0], mask[:, 1]
    if mask.ndim == 2 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 1:
        raise ValueError("sampling_mask must have shape (etl,) or (etl, 1) when no partition axis is configured")
    return mask, None


def _absolute_labels(labels, pe_start, mask_pe, mask_par):
    """Return validated per-ADC absolute LIN/PAR coordinates."""
    pe_idx, par_idx = _norm_pe_start(pe_start, mask_par is not None)
    etl = len(mask_pe)
    if labels is None:
        if mask_par is None:
            return (pe_idx + mask_pe)[:, None]
        return np.column_stack((pe_idx + mask_pe, par_idx + mask_par))
    raw = np.asarray(labels)
    if mask_par is None:
        if raw.ndim == 1:
            raw = raw[:, None]
        if raw.shape != (etl, 1):
            raise ValueError(f"labels must have shape ({etl},) or ({etl}, 1)")
    elif raw.shape != (etl, 2):
        raise ValueError(f"labels must have shape ({etl}, 2)")
    if not np.issubdtype(raw.dtype, np.number) or not np.all(np.isfinite(raw)) or not np.all(raw == np.rint(raw)):
        raise ValueError("labels must contain finite integers")
    return np.array(raw, dtype=int, copy=True)


def _validate_coordinates(pe_idx, par_idx, mask_pe, mask_par, labels, pe_axis, par_axis):
    physical_pe = pe_idx + mask_pe
    if np.any(physical_pe < 0) or np.any(physical_pe >= pe_axis.n):
        raise IndexError("physical LIN coordinates are outside the phase-encode matrix")
    if np.any(labels[:, 0] < 0) or np.any(labels[:, 0] >= pe_axis.n):
        raise IndexError("LIN labels are outside the phase-encode matrix")
    if par_axis is not None:
        physical_par = par_idx + mask_par
        if np.any(physical_par < 0) or np.any(physical_par >= par_axis.n):
            raise IndexError("physical PAR coordinates are outside the partition matrix")
        if np.any(labels[:, 1] < 0) or np.any(labels[:, 1] >= par_axis.n):
            raise IndexError("PAR labels are outside the partition matrix")


def _norm_pe_start(pe_start, has_par):
    """Normalize the ``__call__`` starting index into an ``(pe_idx, par_idx)`` int pair."""
    if has_par:
        pe_idx, par_idx = pe_start
        return int(pe_idx), int(par_idx)
    return int(pe_start), 0


def _min_blip_duration(opts, axis, delta_k, deltas):
    """Shortest trapezoid duration (s) for the worst-case (largest-magnitude) jump in ``deltas``.

    Returns 0.0 when there is no jump at all (``axis`` unused, ``deltas`` empty, or all zero).
    """
    if axis is None or deltas.size == 0:
        return 0.0
    max_delta = int(np.max(np.abs(deltas)))
    if max_delta == 0:
        return 0.0
    trial = pp.make_trapezoid(channel=axis, area=max_delta * delta_k, system=opts)
    return pp.calc_duration(trial)


def _combine_flanking_halves(before, after, gx_anchor, opts):
    """Combine the two half-blips flanking one readout line into that line's single event.

    ``before`` is the second half of the *previous* gap's blip (occurs during the line's
    elongated rise); ``after`` is the first half of the *next* gap's blip (occurs during the
    line's elongated fall). ``gx_anchor`` (any full-duration line gradient) anchors the
    common window duration for :func:`pypulseq.align`. Returns ``None`` if there is nothing
    to play on this channel for this line.
    """
    if before is None and after is None:
        return None
    if before is not None and after is not None:
        after_al, before_al, _ = pp.align(right=[after], left=[before, gx_anchor])
        return pp.add_gradients([before_al, after_al], system=opts)
    if before is not None:
        before_al, _ = pp.align(left=[before, gx_anchor])
        return before_al
    after_al, _ = pp.align(right=[after], left=[gx_anchor])
    return after_al


# ----------------------------------------------------------------------
# Blipped (ramp-sampled, gradient-overlapped) engine
# ----------------------------------------------------------------------


def _build_readout_x_blipped(opts, ro_axis, nx, fov_x_m, bandwidth_hz_px, oversamp, blip_duration_s, ramp_sample):
    """Elongated readout trapezoid (a blip rides its ramps) + ADC.

    Follows the pypulseq ``write_epi_se_rs`` high-performance EPI construction:
    the trapezoid's area is over-driven by ``extra_area`` and the amplitude is
    then rescaled so that, once a blip occupies the (now longer) ramps, the
    *net* k-space sweep is still exactly ``k_width``. ``blip_duration_s`` may
    be 0.0 (single-line train, no blips at all), in which case this reduces to
    a plain flat-top trapezoid.
    """
    raster = opts.grad_raster_time
    delta_k = 1.0 / (oversamp * fov_x_m)
    n_full = int(round(oversamp * nx))
    k_width = n_full * delta_k

    _, flat_time_s = quantize_readout_timing(
        nx_ro=n_full,
        target_bw_hz_px=bandwidth_hz_px,
        grad_raster_s=raster,
        adc_raster_s=opts.adc_raster_time,
        min_flat_time_s=k_width / (READOUT_GRAD_MARGIN * opts.max_grad),
    )

    # `flat_time_s` is only the *flat-top* target; the elongated trapezoid's
    # requested duration must instead extend the *natural* (pre-blip) total
    # duration D0 by `blip_duration_s`, or the extra_area correction below
    # (which assumes exactly `blip_duration_s / 2` is added to each ramp)
    # does not hold and the flat-top silently shrinks below `flat_time_s`.
    gx_natural = pp.make_trapezoid(channel=ro_axis, flat_area=k_width, flat_time=flat_time_s, system=opts)
    d0 = pp.calc_duration(gx_natural)

    extra_area = (blip_duration_s / 2.0) ** 2 * opts.max_slew
    gx = pp.make_trapezoid(channel=ro_axis, area=k_width + extra_area, duration=d0 + blip_duration_s, system=opts)
    actual_area = gx.area
    actual_area -= gx.amplitude / gx.rise_time * (blip_duration_s / 2.0) ** 2 / 2.0
    actual_area -= gx.amplitude / gx.fall_time * (blip_duration_s / 2.0) ** 2 / 2.0
    gx.amplitude = gx.amplitude / actual_area * k_width
    gx.area = gx.amplitude * (gx.flat_time + gx.rise_time / 2.0 + gx.fall_time / 2.0)
    gx.flat_area = gx.amplitude * gx.flat_time
    gx_neg = pp.scale_grad(gx, -1.0)

    if ramp_sample:
        adc_dwell_s = _floor_to_raster(delta_k / gx.amplitude, opts.adc_raster_time)
        n_samples = max(1, int(round(flat_time_s / adc_dwell_s)))
        adc = pp.make_adc(num_samples=n_samples, dwell=adc_dwell_s, delay=blip_duration_s / 2.0, system=opts)
        time_to_center = adc_dwell_s * ((n_samples - 1) / 2.0 + 0.5)
        # ADC start time must land on rf_raster_time (not adc_raster_time) -- pypulseq
        # samples the ADC itself on its own raster, but the *delay* shares the RF grid.
        adc.delay = round_to_raster(gx.rise_time + gx.flat_time / 2.0 - time_to_center, opts.rf_raster_time)
    else:
        adc_dwell_s, _ = quantize_readout_timing(
            nx_ro=n_full,
            target_bw_hz_px=bandwidth_hz_px,
            grad_raster_s=raster,
            adc_raster_s=opts.adc_raster_time,
            min_flat_time_s=k_width / (READOUT_GRAD_MARGIN * opts.max_grad),
        )
        n_samples = n_full
        # gx.rise_time is already the *elongated* ramp (natural rise + blip_duration_s / 2),
        # so sampling starts exactly when the flat-top begins -- no further blip offset here.
        adc = pp.make_adc(
            num_samples=n_samples,
            dwell=adc_dwell_s,
            delay=round_to_raster(gx.rise_time, opts.rf_raster_time),
            system=opts,
        )

    return SimpleNamespace(
        gx_pos=gx, gx_neg=gx_neg, adc=adc, k_width=k_width, delta_k=delta_k,
        n_samples=n_samples, dwell_s=adc_dwell_s, duration_s=pp.calc_duration(gx),
    )


class _EpiTrainBlipped(Readout):
    """Shared blipped/overlapped EPI readout train (see module docstring).

    Subclasses (:class:`Epi2D`, :class:`Epi3D`) configure the phase-encode
    axis (``pe_spec``) and, for 3D, a partition axis (``par_spec``) that is
    also blipped every line.
    """

    def __init__(
        self,
        system,
        fov_x_m,
        nx,
        num_shots,
        sampling_mask,
        *,
        pe_spec=None,
        par_spec=None,
        bandwidth_hz_px=DEFAULT_BANDWIDTH_HZ_PX,
        oversamp=1.0,
        ramp_sample=True,
        start_polarity_positive=True,
        ro_axis="x",
        derate=True,
    ):
        Readout.__init__(self, system)
        if derate:
            apply_system_derates(system)
        self._opts = system
        self.fov_x_m, self.nx = float(fov_x_m), int(nx)
        self.num_shots = int(num_shots)
        self.ro_axis = ro_axis
        self._start_pos = bool(start_polarity_positive)
        self._ramp_sample = bool(ramp_sample)
        self._bandwidth_hz_px = float(bandwidth_hz_px)
        self._oversamp = float(oversamp)

        self._pe = _make_axis(pe_spec)
        self._par = _make_axis(par_spec)

        self._configure(sampling_mask)

    # ------------------------------------------------------------------
    def _configure(self, sampling_mask):
        """(Re)build the per-shot geometry from ``sampling_mask``. Shared by ``__init__`` and ``with_sampling_mask``."""
        mask_pe, mask_par = _parse_mask(sampling_mask, self._par is not None)
        self.etl = int(mask_pe.size)
        if self.etl < 1:
            raise ValueError(f"sampling_mask must have at least one line, got {self.etl}")
        self._mask_pe, self._mask_par = mask_pe, mask_par

        deltas_pe = np.diff(mask_pe) if self.etl > 1 else np.zeros(0, dtype=int)
        deltas_par = np.diff(mask_par) if (self._par is not None and self.etl > 1) else np.zeros(0, dtype=int)

        raster = self._opts.grad_raster_time
        dur_pe = _min_blip_duration(self._opts, self._pe.axis if self._pe else None, getattr(self._pe, "delta_k", 0.0), deltas_pe)
        dur_par = _min_blip_duration(self._opts, self._par.axis if self._par else None, getattr(self._par, "delta_k", 0.0), deltas_par)
        worst = max(dur_pe, dur_par)
        blip_duration = ceil_to_raster(worst, 2.0 * raster) if worst > 0.0 else 0.0
        self._blip_duration = blip_duration

        self._ro = _build_readout_x_blipped(
            self._opts, self.ro_axis, self.nx, self.fov_x_m, self._bandwidth_hz_px,
            self._oversamp, blip_duration, self._ramp_sample,
        )
        self.n_samples = self._ro.n_samples
        self.adc_dwell_s = self._ro.dwell_s
        self.esp = self._ro.duration_s

        self._pe_events = self._axis_events(self._pe, deltas_pe, blip_duration)
        self._par_events = self._axis_events(self._par, deltas_par, blip_duration)

        gx_pre_dur = pp.calc_duration(self._make_prephase_x())
        self.duration = gx_pre_dur + self.etl * self.esp

    def _axis_events(self, axis, deltas, blip_duration):
        """Per-line finalized blip event for ``axis`` (length ``etl``, entries may be ``None``)."""
        if axis is None or blip_duration == 0.0 or deltas.size == 0:
            return [None] * self.etl
        max_delta = int(np.max(np.abs(deltas)))
        if max_delta == 0:
            return [None] * self.etl

        template = pp.make_trapezoid(channel=axis.axis, area=max_delta * axis.delta_k, duration=blip_duration, system=self._opts)
        parts = []
        for d in deltas:
            if d == 0:
                parts.append(None)
                continue
            scaled = pp.scale_grad(template, float(d) / max_delta)
            parts.append(pp.split_gradient_at(grad=scaled, time_point=blip_duration / 2.0, system=self._opts))

        gx_anchor = self._ro.gx_pos
        events = []
        for j in range(self.etl):
            before = parts[j - 1][1] if j > 0 and parts[j - 1] is not None else None
            after = parts[j][0] if j < self.etl - 1 and parts[j] is not None else None
            events.append(_combine_flanking_halves(before, after, gx_anchor, self._opts))
        return events

    def _make_prephase_x(self):
        area = -0.5 * self._ro.k_width if self._start_pos else 0.5 * self._ro.k_width
        return pp.make_trapezoid(channel=self.ro_axis, area=area, system=self._opts)

    def with_sampling_mask(self, sampling_mask):
        """Deep copy with a new in-shot ``sampling_mask`` -- same parent gradients.

        Use this (rather than constructing a second train from scratch) to
        interleave two within-shot patterns -- e.g. odd/even sub-trains played
        as separate shots -- so both instances share numerically identical
        base gradients from a pulseq event-library point of view.
        """
        new = copy.deepcopy(self)
        new._configure(sampling_mask)
        new._block_cache = None
        return new

    def _build_blocks(self):
        state = self._require_state()
        return self._collect_blocks(lambda seq: self._run(seq, state.pe_start, state.rotation, state.labels))

    def _run(self, seq, pe_start, rotation, labels=None):
        if seq is None:
            import pulserver.pypulseq as _ps

            seq = _ps.Sequence(self._opts)

        pe_idx, par_idx = _norm_pe_start(pe_start, self._par is not None)
        absolute_labels = _absolute_labels(labels, pe_start, self._mask_pe, self._mask_par)
        _validate_coordinates(
            pe_idx, par_idx, self._mask_pe, self._mask_par, absolute_labels, self._pe, self._par
        )

        pre = [self._make_prephase_x()]
        if self._pe is not None:
            area = self._pe.areas[pe_idx + int(self._mask_pe[0])]
            pre.append(pp.make_trapezoid(channel=self._pe.axis, area=area, system=self._opts))
        if self._par is not None:
            area = self._par.areas[par_idx + int(self._mask_par[0])]
            pre.append(pp.make_trapezoid(channel=self._par.axis, area=area, system=self._opts))
        pre_aligned = pp.align(right=pre)
        seq.add_block(*pre_aligned, rotation) if rotation is not None else seq.add_block(*pre_aligned)

        for j in range(self.etl):
            positive = (j % 2 == 0) == self._start_pos
            gx = self._ro.gx_pos if positive else self._ro.gx_neg
            adc = copy.deepcopy(self._ro.adc)
            line_labels = []
            if self._pe is not None:
                line_labels.append(pp.make_label(type="SET", label=self._pe.label, value=int(absolute_labels[j, 0])))
            if self._par is not None:
                line_labels.append(pp.make_label(type="SET", label=self._par.label, value=int(absolute_labels[j, 1])))

            events = [gx, adc, *line_labels]
            if self._pe_events[j] is not None:
                events.append(self._pe_events[j])
            if self._par_events[j] is not None:
                events.append(self._par_events[j])
            if rotation is not None:
                events.append(rotation)
            seq.add_block(*events)
        return seq


# ----------------------------------------------------------------------
# Flyback engine
# ----------------------------------------------------------------------


def _build_readout_x_flyback(opts, ro_axis, nx, fov_x_m, bandwidth_hz_px, oversamp, ramp_sample, positive_polarity):
    """Plain (non-elongated) flat-top readout trapezoid + ADC -- every line plays the same polarity."""
    raster = opts.grad_raster_time
    delta_k = 1.0 / (oversamp * fov_x_m)
    n_full = int(round(oversamp * nx))
    k_width = n_full * delta_k

    dwell_s, flat_time_s = quantize_readout_timing(
        nx_ro=n_full,
        target_bw_hz_px=bandwidth_hz_px,
        grad_raster_s=raster,
        adc_raster_s=opts.adc_raster_time,
        min_flat_time_s=k_width / (READOUT_GRAD_MARGIN * opts.max_grad),
    )
    sign = 1.0 if positive_polarity else -1.0
    gx = pp.make_trapezoid(channel=ro_axis, flat_area=sign * k_width, flat_time=flat_time_s, system=opts)

    if ramp_sample:
        n_samples = max(1, int(round(pp.calc_duration(gx) / dwell_s)))
        adc = pp.make_adc(num_samples=n_samples, dwell=dwell_s, delay=0.0, system=opts)
    else:
        n_samples = n_full
        adc = pp.make_adc(
            num_samples=n_samples, dwell=dwell_s, delay=round_to_raster(gx.rise_time, opts.rf_raster_time), system=opts
        )

    return SimpleNamespace(gx=gx, adc=adc, k_width=k_width, delta_k=delta_k, n_samples=n_samples, dwell_s=dwell_s)


def _flyback_axis_events(opts, axis, deltas, duration_s, n_gaps):
    """Per-gap finalized blip event for ``axis`` (length ``n_gaps == etl - 1``, entries may be ``None``)."""
    if axis is None or duration_s == 0.0 or deltas.size == 0:
        return [None] * n_gaps
    max_delta = int(np.max(np.abs(deltas)))
    if max_delta == 0:
        return [None] * n_gaps
    template = pp.make_trapezoid(channel=axis.axis, area=max_delta * axis.delta_k, duration=duration_s, system=opts)
    return [pp.scale_grad(template, float(d) / max_delta) if d != 0 else None for d in deltas]


class _EpiTrainFlyback(Readout):
    """Shared flyback EPI readout train (see module docstring).

    Every line has the same readout polarity; a dedicated rewind gradient
    between lines carries the phase-encode/partition blip(s). Kept as a
    separate engine from :class:`_EpiTrainBlipped` because flyback has no
    split-gradient overlap to perform -- the rewind already owns a full,
    dedicated time slot, so its construction is materially simpler.
    """

    def __init__(
        self,
        system,
        fov_x_m,
        nx,
        num_shots,
        sampling_mask,
        *,
        pe_spec=None,
        par_spec=None,
        bandwidth_hz_px=DEFAULT_BANDWIDTH_HZ_PX,
        oversamp=1.0,
        ramp_sample=True,
        positive_polarity=True,
        ro_axis="x",
        derate=True,
    ):
        Readout.__init__(self, system)
        if derate:
            apply_system_derates(system)
        self._opts = system
        self.fov_x_m, self.nx = float(fov_x_m), int(nx)
        self.num_shots = int(num_shots)
        self.ro_axis = ro_axis
        self._positive = bool(positive_polarity)
        self._ramp_sample = bool(ramp_sample)
        self._bandwidth_hz_px = float(bandwidth_hz_px)
        self._oversamp = float(oversamp)

        self._pe = _make_axis(pe_spec)
        self._par = _make_axis(par_spec)

        self._configure(sampling_mask)

    # ------------------------------------------------------------------
    def _configure(self, sampling_mask):
        mask_pe, mask_par = _parse_mask(sampling_mask, self._par is not None)
        self.etl = int(mask_pe.size)
        if self.etl < 1:
            raise ValueError(f"sampling_mask must have at least one line, got {self.etl}")
        self._mask_pe, self._mask_par = mask_pe, mask_par

        deltas_pe = np.diff(mask_pe) if self.etl > 1 else np.zeros(0, dtype=int)
        deltas_par = np.diff(mask_par) if (self._par is not None and self.etl > 1) else np.zeros(0, dtype=int)

        self._ro = _build_readout_x_flyback(
            self._opts, self.ro_axis, self.nx, self.fov_x_m, self._bandwidth_hz_px,
            self._oversamp, self._ramp_sample, self._positive,
        )
        self.n_samples = self._ro.n_samples
        self.adc_dwell_s = self._ro.dwell_s

        if self.etl > 1:
            dur_pe = _min_blip_duration(self._opts, self._pe.axis if self._pe else None, getattr(self._pe, "delta_k", 0.0), deltas_pe)
            dur_par = _min_blip_duration(self._opts, self._par.axis if self._par else None, getattr(self._par, "delta_k", 0.0), deltas_par)
            gx_line_area = self._ro.gx.area
            flyback_natural = pp.make_trapezoid(channel=self.ro_axis, area=-gx_line_area, system=self._opts)
            flyback_duration = ceil_to_raster(
                max(pp.calc_duration(flyback_natural), dur_pe, dur_par), self._opts.grad_raster_time
            )
            self._gx_flyback = pp.make_trapezoid(channel=self.ro_axis, area=-gx_line_area, duration=flyback_duration, system=self._opts)
        else:
            flyback_duration = 0.0
            self._gx_flyback = None
        self._flyback_duration = flyback_duration

        n_gaps = max(self.etl - 1, 0)
        self._pe_gap_events = _flyback_axis_events(self._opts, self._pe, deltas_pe, flyback_duration, n_gaps)
        self._par_gap_events = _flyback_axis_events(self._opts, self._par, deltas_par, flyback_duration, n_gaps)

        self.esp = pp.calc_duration(self._ro.gx) + self._flyback_duration
        gx_pre_dur = pp.calc_duration(self._make_prephase_x())
        self.duration = gx_pre_dur + self.etl * pp.calc_duration(self._ro.gx) + max(self.etl - 1, 0) * self._flyback_duration

    def _make_prephase_x(self):
        area = -0.5 * self._ro.k_width if self._positive else 0.5 * self._ro.k_width
        return pp.make_trapezoid(channel=self.ro_axis, area=area, system=self._opts)

    def with_sampling_mask(self, sampling_mask):
        """Deep copy with a new in-shot ``sampling_mask`` -- same parent gradients (see
        :meth:`_EpiTrainBlipped.with_sampling_mask`)."""
        new = copy.deepcopy(self)
        new._configure(sampling_mask)
        new._block_cache = None
        return new

    def _build_blocks(self):
        state = self._require_state()
        return self._collect_blocks(lambda seq: self._run(seq, state.pe_start, state.rotation, state.labels))

    def _run(self, seq, pe_start, rotation, labels=None):
        if seq is None:
            import pulserver.pypulseq as _ps

            seq = _ps.Sequence(self._opts)

        pe_idx, par_idx = _norm_pe_start(pe_start, self._par is not None)
        absolute_labels = _absolute_labels(labels, pe_start, self._mask_pe, self._mask_par)
        _validate_coordinates(
            pe_idx, par_idx, self._mask_pe, self._mask_par, absolute_labels, self._pe, self._par
        )

        pre = [self._make_prephase_x()]
        if self._pe is not None:
            area = self._pe.areas[pe_idx + int(self._mask_pe[0])]
            pre.append(pp.make_trapezoid(channel=self._pe.axis, area=area, system=self._opts))
        if self._par is not None:
            area = self._par.areas[par_idx + int(self._mask_par[0])]
            pre.append(pp.make_trapezoid(channel=self._par.axis, area=area, system=self._opts))
        pre_aligned = pp.align(right=pre)
        seq.add_block(*pre_aligned, rotation) if rotation is not None else seq.add_block(*pre_aligned)

        for j in range(self.etl):
            adc = copy.deepcopy(self._ro.adc)
            line_labels = []
            if self._pe is not None:
                line_labels.append(pp.make_label(type="SET", label=self._pe.label, value=int(absolute_labels[j, 0])))
            if self._par is not None:
                line_labels.append(pp.make_label(type="SET", label=self._par.label, value=int(absolute_labels[j, 1])))
            events = [self._ro.gx, adc, *line_labels]
            if rotation is not None:
                events.append(rotation)
            seq.add_block(*events)

            if j < self.etl - 1:
                gap_events = [self._gx_flyback]
                if self._pe_gap_events[j] is not None:
                    gap_events.append(self._pe_gap_events[j])
                if self._par_gap_events[j] is not None:
                    gap_events.append(self._par_gap_events[j])
                if rotation is not None:
                    gap_events.append(rotation)
                seq.add_block(*gap_events)
        return seq


# ----------------------------------------------------------------------
# Public callable readout objects
# ----------------------------------------------------------------------


class Epi2D(_EpiTrainBlipped):
    """2D blipped EPI: per-line ``ky`` blip (``LIN``), ramp-sampled, overlapped with the readout ramps.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits (derated in place unless ``derate=False``).
    fov : tuple[float, float]
        In-plane field of view ``(fov_x_m, fov_y_m)`` in metres.
    matrix : tuple[int, int]
        Matrix size ``(nx, ny)``.
    num_shots : int
        Number of shots the full ``ny`` is split across -- informational
        (the train length is ``len(sampling_mask)``; ``num_shots`` is stored
        as :attr:`num_shots` for the caller's bookkeeping/labels).
    sampling_mask : array-like of int, shape (etl,)
        Within-shot ``ky`` index visited at each line, in train order (see
        module docstring). ``etl = len(sampling_mask)``.
    bandwidth_hz_px, oversamp : float
        Readout receiver bandwidth (Hz/px) and frequency oversampling (>= 1).
    ramp_sample : bool
        ADC spans the whole readout trapezoid (rise+flat+fall) when True
        (default), else only the flat top.
    start_polarity_positive : bool
        Polarity of the first line's readout gradient.
    ro_axis, pe_axis : str
        Readout / phase-encode gradient channels.
    derate : bool
        Apply the standard system derates (default True).

    Attributes
    ----------
    etl : int
        Echo train length (``= len(sampling_mask)``).
    duration : float
        One shot's whole train duration (s), exact.
    esp : float
        Echo spacing (s) -- constant, one elongated readout trapezoid apart.
    n_samples, adc_dwell_s
        Readout sample count, ADC dwell (s).
    """

    def __init__(self, system, fov, matrix, num_shots, sampling_mask, *, pe_axis="y", **kw):
        super().__init__(
            system, fov[0], matrix[0], num_shots, sampling_mask,
            pe_spec=(pe_axis, fov[1], matrix[1], "LIN"), **kw,
        )
        self.pe_axis = pe_axis
        self.fov_y_m, self.ny = float(fov[1]), int(matrix[1])

    def set_state(self, lin_idx=0, rotation=None, *, labels=None):
        """Set the phase-encode offset and rotation for one EPI shot."""
        labels = _absolute_labels(labels, int(lin_idx), self._mask_pe, self._mask_par)
        self._replace_state(_EpiState(pe_start=int(lin_idx), rotation=normalize_rotation(rotation), labels=labels))
        return self

    def __call__(self, seq=None, pe_start=0, rotation=None, *, labels=None):
        """Append one shot. ``pe_start`` is the absolute ``LIN`` index of
        ``sampling_mask[0]`` -- the caller ``labelset``s ``LIN`` to this value
        before calling. ``rotation`` is an optional scipy ``Rotation``-like
        object (see :func:`pulserver.pypulseq.make_rotation`) applied to every
        block of the shot. Returns the sequence."""
        self.set_state(lin_idx=pe_start, rotation=rotation, labels=labels)
        return self._add_or_get(seq)


class Epi3D(_EpiTrainBlipped):
    """3D blipped EPI: per-line ``ky`` and ``kz`` blips (``LIN`` + ``PAR``).

    ``fov = (fov_x_m, fov_y_m, fov_z_m)`` and ``matrix = (nx, ny, nz)``. Both
    the phase-encode and partition axes are blipped every line (a true 3D-EPI
    train, e.g. for blipped-CAIPI); see :class:`Epi2D` for the shared
    parameters.
    """

    def __init__(self, system, fov, matrix, num_shots, sampling_mask, *, pe_axis="y", par_axis="z", **kw):
        super().__init__(
            system, fov[0], matrix[0], num_shots, sampling_mask,
            pe_spec=(pe_axis, fov[1], matrix[1], "LIN"),
            par_spec=(par_axis, fov[2], matrix[2], "PAR"), **kw,
        )
        self.pe_axis, self.par_axis = pe_axis, par_axis
        self.fov_y_m, self.ny = float(fov[1]), int(matrix[1])
        self.fov_z_m, self.nz = float(fov[2]), int(matrix[2])

    def set_state(self, lin_idx=0, par_idx=0, rotation=None, *, labels=None):
        """Set the phase/partition offsets and rotation for one EPI shot."""
        self._replace_state(
            _EpiState(
                pe_start=(int(lin_idx), int(par_idx)),
                rotation=normalize_rotation(rotation),
                labels=_absolute_labels(labels, (int(lin_idx), int(par_idx)), self._mask_pe, self._mask_par),
            )
        )
        return self

    def __call__(self, seq=None, pe_start=(0, 0), rotation=None, *, labels=None):
        """Append one shot. ``pe_start`` is the absolute ``(LIN, PAR)`` index
        pair of ``sampling_mask[0]``. Returns the sequence."""
        self.set_state(lin_idx=pe_start[0], par_idx=pe_start[1], rotation=rotation, labels=labels)
        return self._add_or_get(seq)


class Epi2DFlyback(_EpiTrainFlyback):
    """2D flyback EPI: every line the same readout polarity, ``ky`` blip on the rewind.

    See :class:`Epi2D` for the shared parameters; ``start_polarity_positive``
    is replaced by ``positive_polarity`` (every line, not just the first).
    """

    def __init__(self, system, fov, matrix, num_shots, sampling_mask, *, pe_axis="y", **kw):
        super().__init__(
            system, fov[0], matrix[0], num_shots, sampling_mask,
            pe_spec=(pe_axis, fov[1], matrix[1], "LIN"), **kw,
        )
        self.pe_axis = pe_axis
        self.fov_y_m, self.ny = float(fov[1]), int(matrix[1])

    def set_state(self, lin_idx=0, rotation=None, *, labels=None):
        """Set the phase-encode offset and rotation for one flyback EPI shot."""
        labels = _absolute_labels(labels, int(lin_idx), self._mask_pe, self._mask_par)
        self._replace_state(_EpiState(pe_start=int(lin_idx), rotation=normalize_rotation(rotation), labels=labels))
        return self

    def __call__(self, seq=None, pe_start=0, rotation=None, *, labels=None):
        """Append one shot. See :meth:`Epi2D.__call__`."""
        self.set_state(lin_idx=pe_start, rotation=rotation, labels=labels)
        return self._add_or_get(seq)


class Epi3DFlyback(_EpiTrainFlyback):
    """3D flyback EPI: every line the same readout polarity, ``ky``/``kz`` blips on the rewind.

    See :class:`Epi3D` for the shared parameters.
    """

    def __init__(self, system, fov, matrix, num_shots, sampling_mask, *, pe_axis="y", par_axis="z", **kw):
        super().__init__(
            system, fov[0], matrix[0], num_shots, sampling_mask,
            pe_spec=(pe_axis, fov[1], matrix[1], "LIN"),
            par_spec=(par_axis, fov[2], matrix[2], "PAR"), **kw,
        )
        self.pe_axis, self.par_axis = pe_axis, par_axis
        self.fov_y_m, self.ny = float(fov[1]), int(matrix[1])
        self.fov_z_m, self.nz = float(fov[2]), int(matrix[2])

    def set_state(self, lin_idx=0, par_idx=0, rotation=None, *, labels=None):
        """Set the phase/partition offsets and rotation for one flyback EPI shot."""
        self._replace_state(
            _EpiState(
                pe_start=(int(lin_idx), int(par_idx)),
                rotation=normalize_rotation(rotation),
                labels=_absolute_labels(labels, (int(lin_idx), int(par_idx)), self._mask_pe, self._mask_par),
            )
        )
        return self

    def __call__(self, seq=None, pe_start=(0, 0), rotation=None, *, labels=None):
        """Append one shot. See :meth:`Epi3D.__call__`."""
        self.set_state(lin_idx=pe_start[0], par_idx=pe_start[1], rotation=rotation, labels=labels)
        return self._add_or_get(seq)
