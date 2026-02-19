"""RF duty cycle and SAR evaluation for GEHC systems."""

__all__ = ["compute_max_energy"]

from types import SimpleNamespace

import numpy as np

from pypulseq import Opts

from .._core._ceq import PulseqRF, PulseqGrad, PulseqShapeTrap, Ceq

SEGMENT_RINGDOWN_TIME = 116 * 1e-6  # TODO: doesn't have to be hardcoded
GAMMA = 4.2575e3  # Hz/Gauss


def compute_max_energy(
    ceq: Ceq, system: Opts, window_width: float = 10.0, windows_stride: float = 5.0
) -> float:
    """
    Compute maximum 10s RF energy.

    Parameters
    ----------
    ceq : Ceq
        Ceq structure.
    system : pypulseq.Opts
        Structure containing system limits.
    window_width : float, optional
        Window width in ``[s]``. The default is 10.0.
    windows_stride : TYPE, optional
        Window stride in ``[s]``. The default is 5.0.

    Returns
    -------
    P : float
        Effective 10s RF power.
        Must be compared with max power of standard pulse.

    """
    # get block id
    block_id = np.ascontiguousarray(ceq.loop[:, 1].astype(int))

    # get RF blocks
    rf_blocks = [
        n for n in range(ceq.n_parent_blocks) if ceq.parent_blocks[n].rf is not None
    ]
    has_rf = np.stack([block_id == idx for idx in rf_blocks]).sum(axis=0).astype(bool)

    # get waveforms
    rf_pulses = [
        (
            _rfstat(ceq.parent_blocks[n].rf, system)
            if ceq.parent_blocks[n].rf is not None
            else None
        )
        for n in range(ceq.n_parent_blocks)
    ]

    # calculate energy for full waveform
    rf_energy = [_calc_rf_energy(rf) if rf is not None else None for rf in rf_pulses]

    # get energy scalings (= rf_amp**2)
    powscale = np.ascontiguousarray(ceq.loop[:, 2]) ** 2

    # get segment id
    segment_id = np.ascontiguousarray(ceq.loop[:, 0])
    seg_boundaries = np.diff(segment_id + 1) != 0
    seg_boundaries = np.concatenate((seg_boundaries, np.asarray([True])))

    # get block duration
    block_dur = np.ascontiguousarray(ceq.loop[:, 9])
    block_dur[seg_boundaries] += SEGMENT_RINGDOWN_TIME  # add segment ringdown

    # get absolute sequence time axis
    block_starts = np.cumsum(np.concatenate(([0.0], block_dur)))[:-1]
    block_ends = np.concatenate((block_starts[1:], [block_starts[-1] + block_dur[-1]]))

    # get sequence end
    sequence_end = block_ends[-1]

    # get windows starts
    window_starts = np.arange(
        0, sequence_end - window_width + windows_stride, windows_stride
    )
    if window_starts.size == 0:
        window_starts = np.asarray([0])
    window_ends = window_starts + window_width
    window_ends[-1] = min(window_ends[-1], sequence_end)

    # loop over windows
    energies = []
    n_windows = len(window_starts)
    for n in range(n_windows):
        first_block = np.argmin(abs(block_starts - window_starts[n]))
        last_block = np.argmin(abs(block_ends - window_ends[n]))

        if last_block == first_block:
            last_block += 1

        # get current blocks
        current_starts = block_starts[first_block:last_block][
            has_rf[first_block:last_block]
        ]
        current_ends = block_starts[first_block:last_block][
            has_rf[first_block:last_block]
        ]
        current_blocks = block_id[first_block:last_block][
            has_rf[first_block:last_block]
        ]
        current_powscale = powscale[first_block:last_block][
            has_rf[first_block:last_block]
        ]

        _energies = []

        # calculate energy for first element in block
        if current_starts[0] < window_starts[n]:
            first_rf_in_block = rf_pulses[current_blocks[0]]
            first_rf_idx_in_block = np.argmin(
                abs(first_rf_in_block.time + current_starts[0] - window_starts[n])
            )
            first_rf_in_block_energy = (
                _calc_rf_energy(first_rf_in_block, start=first_rf_idx_in_block)
                * current_powscale[0]
            )
        else:
            first_rf_in_block_energy = (
                rf_energy[current_blocks[0]] * current_powscale[0]
            )
        _energies.append(first_rf_in_block_energy)

        # calculate energies for second to last block
        _energies.extend(
            (
                np.asarray(rf_energy)[np.asarray(current_blocks[1:-1])]
                * np.asarray(current_powscale[1:-1])
            ).tolist()
        )

        # calculate energy for last element in block
        if current_ends[-1] > window_ends[n]:
            last_rf_in_block = rf_pulses[current_blocks[-1]]
            last_rf_idx_in_block = np.argmin(
                abs(last_rf_in_block.time + current_starts[-1] - window_ends[n])
            )
            last_rf_in_block_energy = (
                _calc_rf_energy(last_rf_in_block, stop=last_rf_idx_in_block)
                * current_powscale[-1]
            )
        else:
            last_rf_in_block_energy = (
                rf_energy[current_blocks[-1]] * current_powscale[-1]
            )
        _energies.append(last_rf_in_block_energy)

        # update energies list
        energies.append(np.sum(_energies))

    # find max 10s energy and corresponding power
    max_energy = np.max(energies)  # [G**2 * s]
    P = max_energy / window_width

    return P


def _calc_rf_energy(rf: np.ndarray, start: int = 0, stop: int | None = None) -> float:
    if stop is None:
        return sum(np.abs(rf.waveform[start:]) ** 2) * rf.raster
    return sum(np.abs(rf.waveform[start:]) ** 2) * rf.raster


def _rfstat(rf: PulseqRF, system: Opts) -> SimpleNamespace:
    # get waveform in physical units
    wave_max = abs(rf.wav.magnitude).max()
    waveform = rf.wav.amplitude * (rf.wav.magnitude / wave_max)

    # add phase
    if rf.wav.phase is not None:
        waveform = waveform * np.exp(1j * rf.wav.phase)

    # convert extended trapezoid to arbitrary
    if rf.wav.time is None:
        rf.wav.time = system.rf_raster_time * np.arange(waveform.shape[0])

    if rf.type == 1:
        waveform, time = waveform, rf.wav.time + rf.delay
    elif rf.type == 2:
        waveform, time = _extended2arb(
            waveform, rf.wav.time, system.rf_raster_time, rf.delay
        )

    return SimpleNamespace(
        waveform=waveform / GAMMA, time=time, raster=system.rf_raster_time
    )


def _gradstat(grad: PulseqGrad, system: Opts) -> SimpleNamespace:
    # get waveform in physical units
    if grad.type == 1:
        waveform = grad.trap
    else:
        wave_max = abs(grad.shape.magnitude).max()
        waveform = grad.shape.amplitude * (grad.shape.magnitude / wave_max)

    # convert extended trapezoid to arbitrary
    if grad.type == 1:
        waveform, time = _trap2arb(waveform, system.grad_raster_time, grad.delay)
    elif grad.type == 2:
        waveform, time = waveform, grad.shape.time + grad.delay
    elif grad.type == 3:
        waveform, time = _extended2arb(
            waveform, grad.shape.time, system.grad_raster_time, grad.delay
        )

    return SimpleNamespace(waveform=waveform, time=time, raster=system.grad_raster_time)


# %% local subroutines
def _trap2arb(
    trap: PulseqShapeTrap, dt: float, delay: float
) -> (np.ndarray, np.ndarray):
    waveform, time = _trap2extended(trap)
    return _extended2arb(waveform, time, dt, delay)


def _trap2extended(trap: PulseqShapeTrap) -> (np.ndarray, np.ndarray):
    if trap.flat_time > 0:
        waveform = np.asarray([0, 1, 1, 0]) * trap.amplitude
        time = np.asarray(
            [
                0,
                trap.rise_time,
                trap.rise_time + trap.flat_time,
                trap.rise_time + trap.flat_time + trap.fall_time,
            ]
        )
    else:
        waveform = np.asarray([0, 1, 0]) * trap.amplitude
        time = np.asarray([0, trap.rise_time, trap.rise_time + trap.fall_time])

    return waveform, time


def _extended2arb(
    waveform: np.ndarray, time: np.ndarray, dt: float, delay: float
) -> (np.ndarray, np.ndarray):
    _waveform = waveform
    _time = delay + time

    if delay > 0:
        _waveform = np.concatenate(([0], _waveform))
        _time = np.concatenate(([0], _time))

    time = _arange(0.5 * dt, _time[-1], dt)
    return np.interp(time, _time, _waveform, left=0, right=0), time


def _arange(start: int, stop: int, step: int = 1) -> np.ndarray:
    if stop is None:
        stop = step
        step = 1

    tol = 2.0 * np.finfo(float).eps * max(abs(start), abs(stop))
    sig = np.sign(step)

    # Exceptional cases
    if not np.isfinite(start) or not np.isfinite(step) or not np.isfinite(stop):
        return np.array([np.nan])
    elif step == 0 or (start < stop and step < 0) or (stop < start and step > 0):
        # Result is empty
        return np.zeros(0)

    # n = number of intervals = length(v) - 1
    if start == np.floor(start) and step == 1:
        # Consecutive integers
        n = int(np.floor(stop) - start)
    elif start == np.floor(start) and step == np.floor(step):
        # Integers with spacing > 1
        q = np.floor(start / step)
        r = start - q * step
        n = int(np.floor((stop - r) / step) - q)
    else:
        # General case
        n = round((stop - start) / step)
        if sig * (start + n * step - stop) > tol:
            n -= 1

    # last = right hand end point
    last = start + n * step
    if sig * (last - stop) > -tol:
        last = stop

    # out should be symmetric about the mid-point
    out = np.zeros(n + 1)
    k = np.arange(0, n // 2 + 1)
    out[k] = start + k * step
    out[n - k] = last - k * step
    if n % 2 == 0:
        out[n // 2 + 1] = (start + last) / 2

    return out
