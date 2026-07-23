"""Optimized balanced SSFP (TRUFI) Cartesian readout trains.

Unlike a balanced :class:`~pulserver.design._readout.line.Line2D` shot,
these modules reshape a caller-designed slice/slab excitation into a
continuous gradient waveform between consecutive phase encodes.  For 3D they
also accept a nonselective RF pulse, omitting selection-gradient surgery. The
result is the compact two-block TR from the Pulseq ``writeTrufi`` reference,
with alternating 0/pi RF and ADC phase.  A line readout with
``spoil_position="none"`` remains useful as a building block, but it
deliberately returns every gradient to zero between isolated shots and is not
this optimized steady-state sequence.

``Bssfp2D`` emits one complete 2D acquisition.  ``Bssfp3D`` partitions its
complete ``ky``/``kz`` grid into an integer number of equal-duration segments;
each segment repeats the same startup and ramp-down structure.  The startup
and ramp-down blocks carry ``SET ONCE=1`` and ``SET ONCE=2`` respectively, so
the interpreter can identify the bSSFP transient boundaries.
"""

from __future__ import annotations

__all__ = ["Bssfp2D", "Bssfp3D"]

import copy
from dataclasses import dataclass

import numpy as np
import pypulseq as pp

from .._system import quantize_readout_timing
from ._base import Readout

_READOUT_GRAD_MARGIN = 0.95


@dataclass(frozen=True)
class _BssfpState:
    segment_idx: int
    phase_offset_rad: float


def _duration(event) -> float:
    return float(pp.calc_duration(event)) if event is not None else 0.0


def _copy_at_start(grad):
    grad = copy.deepcopy(grad)
    grad.delay = 0.0
    return grad


def _phase_gradient(system, channel: str, area: float, duration_s: float):
    if abs(area) < 1e-12:
        return None
    return pp.make_trapezoid(channel=channel, area=float(area), duration=duration_s, system=system)


def _safe_phase_duration(system, initial_s: float, ky_areas, kz_areas, gz_1, gz_2) -> float:
    """Find one fixed PE duration that is feasible beside the selection lobes."""
    duration_s = initial_s
    z_areas = [float(value) for value in (np.min(kz_areas), np.max(kz_areas)) if abs(value) > 1e-12]
    max_ky = max(abs(float(np.min(ky_areas))), abs(float(np.max(ky_areas))))
    for _ in range(10_000):
        try:
            if max_ky > 1e-12:
                pp.make_trapezoid(channel="y", area=max_ky, duration=duration_s, system=system)
            combined_durations = [_duration(gz_1), _duration(gz_2)]
            for area in z_areas:
                partition = pp.make_trapezoid(channel="z", area=area, duration=duration_s, system=system)
                if gz_1 is not None:
                    combined_durations.append(_duration(pp.add_gradients([gz_1, partition], system=system)))
                if gz_2 is not None:
                    combined_durations.append(_duration(pp.add_gradients([gz_2, partition], system=system)))
            return max(duration_s, *combined_durations)
        except ValueError:
            duration_s += system.grad_raster_time
    raise ValueError("unable to place the 3D partition encode beside the bSSFP selection gradients")


class _Bssfp(Readout):
    """Shared continuous-gradient bSSFP implementation."""

    def __init__(
        self,
        system,
        fov,
        matrix,
        *,
        excitation_rf,
        excitation_gz,
        excitation_rephaser,
        segments: int,
        bandwidth_hz_px: float,
    ) -> None:
        super().__init__(system)
        self._opts = system
        self.fov = tuple(float(value) for value in fov)
        self.matrix = tuple(int(value) for value in matrix)
        self.nx, self.ny = self.matrix[:2]
        self.nz = self.matrix[2] if len(self.matrix) == 3 else 1
        if min(self.matrix) < 1 or min(self.fov) <= 0.0:
            raise ValueError("fov and matrix entries must be positive")
        if segments < 1:
            raise ValueError("segment must be >= 1")
        self.num_segments = int(segments)
        self.n_encoding_lines = self.ny * self.nz
        if self.n_encoding_lines % self.num_segments:
            raise ValueError(
                "segment must divide ny*nz so all 3D bSSFP shots have an identical structure"
            )
        self.lines_per_segment = self.n_encoding_lines // self.num_segments
        self.n_samples = self.nx

        # The reference uses a symmetric, full-echo readout.  The prephaser
        # and first ramp/flat part are merged, as are the final ramp and the
        # next prephaser, so adjacent TRs share their ramps.
        delta_kx = 1.0 / self.fov[0]
        dwell_s, flat_time_s = quantize_readout_timing(
            nx_ro=self.nx,
            target_bw_hz_px=float(bandwidth_hz_px),
            grad_raster_s=system.grad_raster_time,
            adc_raster_s=system.adc_raster_time,
            min_flat_time_s=(self.nx * delta_kx) / (_READOUT_GRAD_MARGIN * system.max_grad),
        )
        gx_full = pp.make_trapezoid(
            channel="x", flat_area=self.nx * delta_kx, flat_time=flat_time_s, system=system
        )
        adc = pp.make_adc(num_samples=self.nx, dwell=dwell_s, delay=gx_full.rise_time, system=system)
        gx_pre = pp.make_trapezoid(channel="x", area=-0.5 * gx_full.area, system=system)
        gx_head, gx_tail = pp.split_gradient_at(
            gx_full, gx_full.rise_time + gx_full.flat_time, system=system
        )
        gx_head = _copy_at_start(gx_head)
        gx_tail = _copy_at_start(gx_tail)
        gx_head.delay = _duration(gx_pre)
        gx_pre_tail = copy.deepcopy(gx_pre)
        gx_pre_tail.delay = _duration(gx_tail)
        self._gx_1 = pp.add_gradients([gx_pre, gx_head], system=system)
        self._gx_2 = pp.add_gradients([gx_tail, gx_pre_tail], system=system)
        self._adc = adc
        self._adc.delay += _duration(gx_pre)

        # Split the selection lobe at the end of RF and distribute its
        # rephaser over the two alternating blocks.  This is the key TRUFI
        # construction: gz_1 ends on the selection plateau and gz_2 starts
        # there, avoiding a return-to-zero between RF pulses.  Nonselective
        # 3D bSSFP has no z selection lobe, so its partition encode remains a
        # regular standalone z trapezoid.
        rf = copy.deepcopy(excitation_rf)
        if excitation_gz is not None:
            gz = copy.deepcopy(excitation_gz)
            gz_reph = copy.deepcopy(excitation_rephaser)
            gz_head, gz_tail = pp.split_gradient_at(gz, _duration(rf), system=system)
            gz_head = _copy_at_start(gz_head)
            gz_tail = _copy_at_start(gz_tail)
            gz_reph_tail = copy.deepcopy(gz_reph)
            gz_reph_tail.delay = _duration(gz_tail)
            self._gz_1 = pp.add_gradients([gz_reph, gz_head], system=system)
            self._gz_2 = pp.add_gradients([gz_tail, gz_reph_tail], system=system)
        else:
            self._gz_1 = None
            self._gz_2 = None

        # Right-align RF to the selection lobe when present; otherwise delay
        # it behind gx_2.  Both variants leave the RF and ADC centers
        # symmetrically placed within the steady-state TR.
        shift_s = max(_duration(self._gx_2) - rf.delay + rf.ringdown_time, 0.0)
        if self._gz_1 is not None:
            self._gz_1.delay = shift_s
        rf.delay += shift_s
        self._rf = rf
        self._rf_half = copy.deepcopy(rf)
        self._rf_half.signal *= 0.5

        self._ky_areas = (np.arange(self.ny) - 0.5 * self.ny) / self.fov[1]
        self._kz_areas = (
            (np.arange(self.nz) - 0.5 * self.nz) / self.fov[2] if self.nz > 1 else np.array([0.0])
        )
        self._pe_duration_s = _safe_phase_duration(
            system, _duration(self._gx_2), self._ky_areas, self._kz_areas, self._gz_1, self._gz_2
        )

        # A short readout (especially a small 3D matrix) and partition
        # encoding can make the ADC-side block shorter than the RF-side one.
        # Delay gx_1 and its ADC together until both halves have the same
        # duration; this is the minimum TR extension that preserves the
        # shared x-gradient boundary and TE = TR/2.
        z_phase_duration_s = self._pe_duration_s if self.nz > 1 else 0.0
        self._rf_side_duration_s = max(_duration(self._gz_1), _duration(self._rf))
        self._half_tr_s = max(self._rf_side_duration_s, _duration(self._gx_1), z_phase_duration_s)
        self._tr_pad_s = self._half_tr_s - _duration(self._gx_1)
        if self._tr_pad_s > 0.0:
            self._gx_1.delay += self._tr_pad_s
            self._adc.delay += self._tr_pad_s

        self.tr_s = 2.0 * self._half_tr_s
        self.te_s = 0.5 * self.tr_s

        # The half-flip preparation begins on the same x/z boundaries as the
        # repeating train.  Its x lobe ends at gx_2's nonzero first value so
        # the first normal RF block can continue without a discontinuity.
        gx_prep, _, _ = pp.make_extended_trapezoid_area(
            area=-self._gx_2.area,
            channel="x",
            grad_start=0.0,
            grad_end=float(self._gx_2.first),
            system=system,
        )
        self._prep_delay_s = max(0.5 * self.tr_s - self._rf_side_duration_s, 0.0)
        self._gx_prep = gx_prep

        self.set_state()

    def set_state(self, segment_idx: int = 0, phase_offset_rad: float = 0.0):
        segment_idx = int(segment_idx)
        if not 0 <= segment_idx < self.num_segments:
            raise IndexError(f"segment_idx must be in [0, {self.num_segments}), got {segment_idx}")
        self._replace_state(_BssfpState(segment_idx, float(phase_offset_rad)))
        return self

    def _phase_events(self, ky_idx: int, kz_idx: int, *, previous: bool):
        sign = -1.0 if previous else 1.0
        gy = _phase_gradient(self._opts, "y", sign * self._ky_areas[ky_idx], self._pe_duration_s)
        gz = _phase_gradient(self._opts, "z", sign * self._kz_areas[kz_idx], self._pe_duration_s)
        return gy, gz

    def _selection_with_partition(self, selection, partition):
        if selection is None:
            return partition
        if partition is None:
            return selection
        return pp.add_gradients([selection, partition], system=self._opts)

    def _build_blocks(self):
        state = self._require_state()
        start = state.segment_idx * self.lines_per_segment
        stop = start + self.lines_per_segment
        first_ky, first_kz = self._coordinates(start)
        last_ky, last_kz = self._coordinates(stop - 1)

        # alpha/2 preparation: the ONCE label identifies the entry transient.
        rf_half = copy.deepcopy(self._rf_half)
        rf_half.phase_offset = state.phase_offset_rad
        yield tuple(
            event for event in (rf_half, self._gz_1, pp.make_label(type="SET", label="ONCE", value=1)) if event is not None
        )

        gy_last, gz_last = self._phase_events(last_ky, last_kz, previous=False)
        gz_prep = self._selection_with_partition(self._gz_2, gz_last)
        prep_events = [gz_prep, gy_last, self._gx_prep]
        prep_duration_s = max(self._prep_delay_s, *(_duration(event) for event in prep_events if event is not None))
        gx_prep = copy.deepcopy(self._gx_prep)
        gx_prep.delay = prep_duration_s - _duration(gx_prep)
        yield tuple(
            event
            for event in (pp.make_delay(self._prep_delay_s), gz_prep, gy_last, gx_prep, pp.make_label(type="SET", label="ONCE", value=0))
            if event is not None
        )

        previous_ky, previous_kz = last_ky, last_kz
        for encoded_idx in range(start, stop):
            ky_idx, kz_idx = self._coordinates(encoded_idx)
            gy_prev, gz_prev = self._phase_events(previous_ky, previous_kz, previous=True)
            rf = copy.deepcopy(self._rf)
            adc = copy.deepcopy(self._adc)
            phase = state.phase_offset_rad + np.pi * (encoded_idx & 1)
            rf.phase_offset = phase
            adc.phase_offset = phase
            yield tuple(
                event
                for event in (rf, self._selection_with_partition(self._gz_1, gz_prev), gy_prev, self._gx_2)
                if event is not None
            )

            gy_next, gz_next = self._phase_events(ky_idx, kz_idx, previous=False)
            labels = [pp.make_label(type="SET", label="LIN", value=ky_idx)]
            if self.nz > 1:
                labels.append(pp.make_label(type="SET", label="PAR", value=kz_idx))
            yield tuple(
                event
                for event in (self._gx_1, gy_next, self._selection_with_partition(self._gz_2, gz_next), adc, *labels)
                if event is not None
            )
            previous_ky, previous_kz = ky_idx, kz_idx

        # gx_2 closes the final nonzero x boundary; ONCE=2 identifies the
        # bSSFP ramp-down independently for every structurally identical shot.
        yield (self._gx_2, pp.make_label(type="SET", label="ONCE", value=2))

    def _coordinates(self, encoded_idx: int) -> tuple[int, int]:
        return encoded_idx % self.ny, encoded_idx // self.ny


class Bssfp2D(_Bssfp):
    """One optimized, slice-selective 2D bSSFP/TRUFI acquisition."""

    def __init__(self, system, fov, matrix, **kwargs) -> None:
        super().__init__(system, fov, matrix, segments=1, **kwargs)


class Bssfp3D(_Bssfp):
    """One segment of an optimized slab-selective 3D bSSFP acquisition."""
