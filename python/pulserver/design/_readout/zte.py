"""Basic zero-echo-time (ZTE) readout segments.

``Zte`` builds an ordered segment of centre-out FID views.  The gradient ramps
from zero to the first point on the readout sphere only once.  Every excitation
and ADC then runs on a constant-gradient plateau; after the ADC, the gradient
slews directly to the next ordered sphere point without returning to zero.
Only the final view ramps down.  Smooth view ordering is therefore part of the
constructor, not an outer per-spoke loop.

The caller owns both excitation design and sampling design.  ``exc_rf`` is a
non-selective RF event (for example ``rf.make_hard_pulse(...).rf``), while
``view_order`` is either a sequence of in-plane angles or an ``(N, 2)``/
``(N, 3)`` array of directions.  ``set_state(rotation=...)`` applies one rigid
rotation to the complete segment.  The ordering is not phyllotaxis-specific:
for example, coplanar directions describe a radial disk which can be reused at
arbitrary 3D orientations.  A congruent base phyllotaxis interleaf is just one
other possible use.

The finite transmit/receive interval is rounded up to an integer number of ADC
dwells.  Those unavailable centre samples are omitted rather than compressing
the surviving samples into the nominal k-space extent.  No PETRA/HYFI gap
filling, BURST-ZTE, Looping Star, or multi-echo refocusing is implemented.
"""

from __future__ import annotations

__all__ = ["Zte", "DEFAULT_ZTE_BANDWIDTH_HZ", "DEFAULT_RADIAL_OVERSAMP"]

import copy
from dataclasses import dataclass

import numpy as np
import pypulseq as pp

from .._system import apply_system_derates, ceil_to_raster, quantize_readout_timing, round_to_raster
from ._base import Readout, normalize_rotation

DEFAULT_ZTE_BANDWIDTH_HZ = 62_500.0
DEFAULT_RADIAL_OVERSAMP = 2.0
READOUT_GRAD_MARGIN = 0.95
_CHANNELS = ("x", "y", "z")


@dataclass(frozen=True)
class _ZteState:
    lin_idx: tuple[int, ...]
    phase_offset_rad: tuple[float, ...]
    rotation: object | None


def _isotropic_float(value, name: str) -> float:
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{name} must contain positive finite values")
    if not np.allclose(values, values[0]):
        raise ValueError(f"ZTE requires isotropic {name}, got {tuple(values)}")
    return float(values[0])


def _isotropic_int(value, name: str) -> int:
    values = np.asarray(value).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name} must contain positive integers")
    integers = values.astype(int)
    if np.any(values != integers) or np.any(integers < 1):
        raise ValueError(f"{name} must contain positive integers")
    if np.any(integers != integers[0]):
        raise ValueError(f"ZTE requires isotropic {name}, got {tuple(integers)}")
    return int(integers[0])


def _directions(view_order) -> np.ndarray:
    order = np.asarray(view_order, dtype=float)
    if order.ndim == 1:
        if order.size == 0:
            raise ValueError("view_order must contain at least one view")
        directions = np.column_stack((np.cos(order), np.sin(order), np.zeros_like(order)))
    elif order.ndim == 2 and order.shape[1] in (2, 3) and order.shape[0] > 0:
        directions = order if order.shape[1] == 3 else np.column_stack((order, np.zeros(order.shape[0])))
    else:
        raise ValueError("view_order must be angles with shape (N,) or directions with shape (N, 2)/(N, 3)")
    if not np.all(np.isfinite(directions)):
        raise ValueError("view_order must contain only finite values")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("view_order directions must be non-zero")
    directions = np.array(directions / norms[:, None], copy=True)
    directions.setflags(write=False)
    return directions


def _schedule(value, length: int, name: str, *, integer: bool) -> tuple:
    array = np.asarray(value)
    if array.ndim == 0:
        array = np.repeat(array, length)
    elif array.shape != (length,):
        raise ValueError(f"{name} must be a scalar or have length {length}")
    if integer:
        integers = array.astype(int)
        if np.any(array != integers):
            raise ValueError(f"{name} must contain integers")
        return tuple(int(item) for item in integers)
    floats = array.astype(float)
    if not np.all(np.isfinite(floats)):
        raise ValueError(f"{name} must contain finite values")
    return tuple(float(item) for item in floats)


def _extended_gradients(system, times: np.ndarray, amplitudes: np.ndarray) -> tuple:
    events = []
    for axis, waveform in zip(_CHANNELS, amplitudes.T, strict=True):
        if np.any(np.abs(waveform) > 1e-12):
            events.append(
                pp.make_extended_trapezoid(
                    channel=axis,
                    times=times,
                    amplitudes=waveform,
                    system=system,
                )
            )
    return tuple(events)


class Zte(Readout):
    """An ordered, continuous-gradient basic-ZTE segment.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits (derated in place unless ``derate=False``).
    fov : float or sequence of float
        Isotropic encoded field of view in metres.
    matrix : int or sequence of int
        Isotropic nominal image matrix.
    view_order : array-like
        Ordered in-plane angles in radians, or ``(N, 2)``/``(N, 3)`` spoke
        directions. Directions are normalized by the constructor. Any ordering
        is accepted; coplanar directions form a disk, while non-coplanar
        directions can cover a spherical region or complete sphere.
    exc_rf
        Mandatory caller-supplied non-selective pypulseq RF event.
    bandwidth_hz_px : float
        Requested receiver bandwidth (``1 / dwell``), in Hz.
    oversamp : float
        Radial oversampling factor (>= 1).
    dead_time_s : float, optional
        Additional receiver dead time after the RF waveform. ``None`` uses
        ``system.adc_dead_time``; RF ringdown is included automatically.
    tr_s : float, optional
        RF-centre-to-RF-centre repetition time. ``None`` uses the shortest
        duration independently for every view. A specified TR adds a constant-
        gradient hold after each direction update and must fit the slowest
        transition in the supplied ordering.
    derate : bool
        Apply standard system gradient/slew derates.

    Attributes
    ----------
    duration, view_duration_s : float
        Whole segment duration and constant-gradient acquisition plateau (s).
    num_views : int
        Number of ordered views in this segment.
    n_samples, num_missing_samples, nominal_samples : int
        Per-view acquired, missing-centre, and nominal sample counts.
    adc_dwell_s, gap_s : float
        ADC dwell and raster-quantized RF-centre-to-ADC-window gap (s).
    directions : numpy.ndarray
        Read-only normalized base directions, shape ``(num_views, 3)``.
    transition_durations_s : tuple[float, ...]
        Slew-limited transition after each view; the last entry is the final
        ramp to zero.
    """

    def __init__(
        self,
        system,
        fov,
        matrix,
        view_order,
        exc_rf,
        *,
        bandwidth_hz_px=DEFAULT_ZTE_BANDWIDTH_HZ,
        oversamp=DEFAULT_RADIAL_OVERSAMP,
        dead_time_s=None,
        tr_s=None,
        derate=True,
    ):
        Readout.__init__(self, system)
        if exc_rf is None or getattr(exc_rf, "type", None) != "rf":
            raise ValueError("exc_rf must be a caller-supplied pypulseq RF event")
        if not np.isfinite(bandwidth_hz_px) or bandwidth_hz_px <= 0.0:
            raise ValueError(f"bandwidth_hz_px must be > 0, got {bandwidth_hz_px}")
        if not np.isfinite(oversamp) or oversamp < 1.0:
            raise ValueError(f"oversamp must be >= 1, got {oversamp}")

        self.fov_m = _isotropic_float(fov, "fov")
        self.matrix = _isotropic_int(matrix, "matrix")
        self.directions = _directions(view_order)
        self.num_views = len(self.directions)
        self.oversamp = float(oversamp)

        if derate:
            apply_system_derates(system)
        self._opts = system
        self._rf = copy.deepcopy(exc_rf)

        if dead_time_s is None:
            dead_time_s = system.adc_dead_time
        if not np.isfinite(dead_time_s) or dead_time_s < system.adc_dead_time:
            raise ValueError(
                f"dead_time_s must be finite and >= system.adc_dead_time ({system.adc_dead_time}), "
                f"got {dead_time_s}"
            )
        self.dead_time_s = float(dead_time_s)

        self.delta_k = 1.0 / (self.oversamp * self.fov_m)
        self.kmax = 0.5 * self.matrix / self.fov_m
        self.nominal_samples = max(1, int(round(self.kmax / self.delta_k)))
        k_extent = self.nominal_samples * self.delta_k
        dwell_s, _ = quantize_readout_timing(
            nx_ro=self.nominal_samples,
            target_bw_hz_px=float(bandwidth_hz_px),
            grad_raster_s=system.grad_raster_time,
            adc_raster_s=system.adc_raster_time,
            min_flat_time_s=k_extent / (READOUT_GRAD_MARGIN * system.max_grad),
        )
        self.adc_dwell_s = dwell_s
        self.bandwidth_hz = 1.0 / dwell_s

        rf_shape_s = float(self._rf.shape_dur)
        rf_center_local_s = float(pp.calc_rf_center(self._rf)[0])
        if rf_shape_s >= 2.0 * dwell_s - 1e-12:
            raise ValueError(
                "exc_rf is too long for uniform constant-gradient ZTE excitation: "
                f"duration={rf_shape_s * 1e6:.3f} us must be < 2*dwell={2.0 * dwell_s * 1e6:.3f} us"
            )

        minimum_gap_s = (
            rf_shape_s - rf_center_local_s
            + float(getattr(self._rf, "ringdown_time", 0.0))
            + self.dead_time_s
        )
        self.minimum_gap_s = minimum_gap_s
        self.num_missing_samples = max(0, int(np.ceil(minimum_gap_s / dwell_s - 1e-12)))
        if self.num_missing_samples >= self.nominal_samples:
            raise ValueError(
                "the RF/receiver dead-time gap consumes the entire ZTE half-spoke: "
                f"missing={self.num_missing_samples}, nominal={self.nominal_samples}"
            )
        self.n_samples = self.nominal_samples - self.num_missing_samples
        self.gap_s = self.num_missing_samples * dwell_s

        self._rf.delay = max(float(getattr(self._rf, "delay", 0.0)), float(system.rf_dead_time))
        self.rf_center_s = self._rf.delay + rf_center_local_s
        self.adc_start_s = self.rf_center_s + self.gap_s
        self.first_sample_s = self.adc_start_s + 0.5 * dwell_s
        self._adc = pp.make_adc(
            num_samples=self.n_samples,
            dwell=dwell_s,
            delay=self.adc_start_s,
            system=system,
        )
        self.view_duration_s = ceil_to_raster(
            self.rf_center_s + self.nominal_samples * dwell_s,
            system.grad_raster_time,
        )

        self.gradient_amplitude = self.delta_k / dwell_s
        endpoints = self.gradient_amplitude * self.directions
        ramp_up_s = ceil_to_raster(self.gradient_amplitude / system.max_slew, system.grad_raster_time)
        self.ramp_up_s = ramp_up_s
        self._ramp_up = _extended_gradients(
            system,
            np.array([0.0, ramp_up_s]),
            np.vstack((np.zeros(3), endpoints[0])),
        )

        transition_durations = []
        for index, start in enumerate(endpoints):
            end = endpoints[index + 1] if index + 1 < self.num_views else np.zeros(3)
            delta = end - start
            transition_s = ceil_to_raster(np.linalg.norm(delta) / system.max_slew, system.grad_raster_time)
            transition_s = max(system.grad_raster_time, transition_s)
            transition_durations.append(transition_s)

        if tr_s is None:
            self.tr = None
            block_durations = [self.view_duration_s + duration for duration in transition_durations]
        else:
            if not np.isfinite(tr_s) or tr_s <= 0.0:
                raise ValueError(f"tr_s must be positive and finite, got {tr_s}")
            self.tr = round_to_raster(float(tr_s), system.grad_raster_time)
            minimum_tr = self.view_duration_s + max(transition_durations)
            if self.tr < minimum_tr - 1e-12:
                raise ValueError(
                    f"tr_s={float(tr_s) * 1e3:.3f} ms is below the minimum required by this view order "
                    f"({minimum_tr * 1e3:.3f} ms)"
                )
            block_durations = [self.tr] * self.num_views

        view_gradients = []
        for index, start in enumerate(endpoints):
            end = endpoints[index + 1] if index + 1 < self.num_views else np.zeros(3)
            transition_s = transition_durations[index]
            transition_end_s = self.view_duration_s + transition_s
            times = [0.0, self.view_duration_s, transition_end_s]
            amplitudes = [start, start, end]
            if block_durations[index] > transition_end_s + 1e-12:
                times.append(block_durations[index])
                amplitudes.append(end)
            view_gradients.append(
                _extended_gradients(
                    system,
                    np.asarray(times),
                    np.asarray(amplitudes),
                )
            )
        self._view_gradients = tuple(view_gradients)
        self.transition_durations_s = tuple(transition_durations)
        self.block_durations_s = tuple(block_durations)
        self.duration = self.ramp_up_s + sum(self.block_durations_s)

    def set_state(self, lin_idx=0, phase_offset_rad=0.0, rotation=None):
        """Set per-view labels/phases and one rigid rotation for the segment.

        The rotation acts on the complete caller-defined direction set, such
        as a planar disk or a phyllotaxis interleaf; it does not impose or
        regenerate any sampling order.
        """
        labels = _schedule(lin_idx, self.num_views, "lin_idx", integer=True)
        phases = _schedule(phase_offset_rad, self.num_views, "phase_offset_rad", integer=False)
        self._replace_state(
            _ZteState(
                lin_idx=labels,
                phase_offset_rad=phases,
                rotation=normalize_rotation(rotation),
            )
        )
        return self

    def _build_blocks(self):
        state = self._require_state()
        blocks = []
        ramp = [*self._ramp_up]
        if state.rotation is not None:
            ramp.append(state.rotation)
        blocks.append(tuple(ramp))

        for index, gradients in enumerate(self._view_gradients):
            rf = copy.deepcopy(self._rf)
            adc = copy.deepcopy(self._adc)
            rf.phase_offset += state.phase_offset_rad[index]
            adc.phase_offset = rf.phase_offset
            events = [
                *gradients,
                rf,
                adc,
                pp.make_label(type="SET", label="LIN", value=state.lin_idx[index]),
            ]
            if state.rotation is not None:
                events.append(state.rotation)
            blocks.append(tuple(events))
        return tuple(blocks)
