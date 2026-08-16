"""Per-instance execution content, computed from the file.

The PulSeg SegmentInstance (spec 3.3) carries what varies per played block:
amplitudes, phase/frequency offsets, rotation, acquisition presence, and
the sticky safety labels. This module computes all of it directly from a
parsed ``pulserver.pypulseq.Sequence``, so a test can hold the C
interpreter's resolved tables against file-derived references without any
stored truth.

Conventions (the interpreter's, asserted where they are checked):

* RF instance amplitude is the peak of ``abs(signal)`` in Hz; the stored
  envelope is peak-normalized.
* A trapezoid's instance amplitude is its signed plateau. A sampled
  gradient's instance amplitude is the unsigned peak of the waveform; its
  sign lives in the shape, so equality is asserted on the reconstructed
  physical waveform, not on the amplitude alone.
* ``TRID``, ``NOROT`` and ``NOPOS`` are running label states: a ``SET``
  applies from its block onward until the next ``SET``.
"""

from __future__ import annotations

import numpy as np

_STICKY = ("TRID", "NOROT", "NOPOS")


def quaternion_matrix(q) -> np.ndarray:
    """Row-major 3x3 rotation from a ``[w, x, y, z]`` quaternion."""
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _grad_instance_amplitude(grad) -> float:
    if grad is None:
        return 0.0
    if grad.type == "trap":
        return float(grad.amplitude)
    return float(np.abs(np.asarray(grad.waveform)).max())


def _rf_offsets(rf, gamma_hz_per_t: float, b0_t: float) -> tuple[float, float]:
    mhz = gamma_hz_per_t * b0_t * 1e-6
    freq = float(rf.freq_offset) + float(getattr(rf, "freq_ppm", 0.0) or 0.0) * mhz
    phase = float(rf.phase_offset) + float(getattr(rf, "phase_ppm", 0.0) or 0.0) * mhz
    return freq, phase


def _adc_offsets(adc, gamma_hz_per_t: float, b0_t: float) -> tuple[float, float]:
    mhz = gamma_hz_per_t * b0_t * 1e-6
    freq = float(adc.freq_offset) + float(getattr(adc, "freq_ppm", 0.0) or 0.0) * mhz
    phase = float(adc.phase_offset) + float(getattr(adc, "phase_ppm", 0.0) or 0.0) * mhz
    return freq, phase


def _has_output_trigger(block) -> bool:
    for trig in getattr(block, "triggers", ()) or ():
        if getattr(trig, "channel", None) in ("ext1",) or getattr(trig, "type", "") in (
            "output",
            "trigger",
        ):
            return True
    return False


def instance_table(seq, gamma_hz_per_t: float, b0_t: float) -> dict[str, np.ndarray]:
    """One row per block of the stream, in file order.

    Returns the same columns the interpreter's instance walk exposes, so a
    test can compare them directly (sampled-gradient amplitudes compare
    unsigned, see the module docstring).
    """
    n = seq.num_blocks
    out = {
        "duration_us": np.zeros(n, dtype=int),
        "rf_amp_hz": np.zeros(n),
        "rf_freq_hz": np.zeros(n),
        "rf_phase_rad": np.zeros(n),
        "gx_amp_hz_per_m": np.zeros(n),
        "gy_amp_hz_per_m": np.zeros(n),
        "gz_amp_hz_per_m": np.zeros(n),
        "gx_is_trap": np.zeros(n, dtype=int),
        "gy_is_trap": np.zeros(n, dtype=int),
        "gz_is_trap": np.zeros(n, dtype=int),
        "adc_flag": np.zeros(n, dtype=int),
        "adc_freq_hz": np.zeros(n),
        "adc_phase_rad": np.zeros(n),
        "digitalout_flag": np.zeros(n, dtype=int),
        "rotmat": np.tile(np.eye(3).reshape(9), (n, 1)),
        "trid": np.zeros(n, dtype=int),
        "norot_flag": np.zeros(n, dtype=int),
        "nopos_flag": np.zeros(n, dtype=int),
    }

    sticky = {name: 0 for name in _STICKY}
    for i in range(n):
        block = seq.get_block(i + 1)
        for kind, name, value in block.labels:
            if kind == "SET" and name in sticky:
                sticky[name] = int(value)

        out["duration_us"][i] = round(float(block.block_duration) * 1e6)
        if block.rf is not None:
            out["rf_amp_hz"][i] = float(np.abs(np.asarray(block.rf.signal)).max())
            freq, phase = _rf_offsets(block.rf, gamma_hz_per_t, b0_t)
            out["rf_freq_hz"][i] = freq
            out["rf_phase_rad"][i] = phase
        for axis in ("gx", "gy", "gz"):
            grad = getattr(block, axis)
            out[f"{axis}_amp_hz_per_m"][i] = _grad_instance_amplitude(grad)
            out[f"{axis}_is_trap"][i] = int(grad is not None and grad.type == "trap")
        if block.adc is not None:
            out["adc_flag"][i] = 1
            freq, phase = _adc_offsets(block.adc, gamma_hz_per_t, b0_t)
            out["adc_freq_hz"][i] = freq
            out["adc_phase_rad"][i] = phase
        if _has_output_trigger(block):
            out["digitalout_flag"][i] = 1
        rotation = getattr(block, "rotation", None)
        if rotation is not None:
            out["rotmat"][i] = quaternion_matrix(rotation).reshape(9)
        out["trid"][i] = sticky["TRID"]
        out["norot_flag"][i] = sticky["NOROT"]
        out["nopos_flag"][i] = sticky["NOPOS"]
    return out


def adc_label_table(seq, columns: tuple[str, ...] = ("LIN", "SLC", "ECO")) -> np.ndarray:
    """The per-acquisition counter table, one row per ADC in stream order.

    Only labels the file WRITES count (SET assigns, INC accumulates): the
    instance table carries execution parameters as stated, never counters
    somebody could derive from the trajectory.
    """
    state = {name: 0 for name in columns}
    rows: list[list[int]] = []
    for i in range(1, seq.num_blocks + 1):
        block = seq.get_block(i)
        for kind, name, value in block.labels:
            if name not in state:
                continue
            if kind == "SET":
                state[name] = int(value)
            elif kind == "INC":
                state[name] += int(value)
        if block.adc is not None:
            rows.append([state[name] for name in columns])
    return np.asarray(rows, dtype=int).reshape(len(rows), len(columns))


def gradient_vertices(block, axis: str, t0_us: float) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear vertices (time in us, amplitude in Hz/m) of one
    block's gradient on `axis`, absolute-timed at block start `t0_us`."""
    grad = getattr(block, axis)
    if grad is None:
        return np.empty(0), np.empty(0)
    delay_us = float(grad.delay) * 1e6
    if grad.type == "trap":
        t = t0_us + delay_us + np.cumsum(
            [0.0, float(grad.rise_time) * 1e6, float(grad.flat_time) * 1e6,
             float(grad.fall_time) * 1e6]
        )
        v = np.array([0.0, grad.amplitude, grad.amplitude, 0.0])
        return t, v
    tt = np.asarray(grad.tt, dtype=float) * 1e6
    return t0_us + delay_us + tt, np.asarray(grad.waveform, dtype=float)


def block_boundaries(seq, start_block: int, num_blocks: int) -> np.ndarray:
    """Interior block-boundary times (us) of the window, relative to its start."""
    edges = np.cumsum(
        [
            round(float(seq.get_block(start_block + 1 + i).block_duration) * 1e6)
            for i in range(num_blocks)
        ]
    )
    return edges[:-1].astype(float)


def tr_window_waveforms(
    seq,
    start_block: int,
    num_blocks: int,
    times_us: np.ndarray,
    norot_flags: np.ndarray | None = None,
) -> np.ndarray:
    """The composed physical (rotated) gradients of blocks
    [start_block, +num_blocks), sampled at `times_us` relative to the window
    start. Returns a (3, len(times_us)) array in x/y/z order.

    Each block's per-axis events are sampled on the grid, rotated by the
    block's rotation (skipped where the sticky NOROT flag is set), and
    summed.
    """
    out = np.zeros((3, times_us.size))
    t0 = 0.0
    for i in range(num_blocks):
        block = seq.get_block(start_block + 1 + i)
        block_end = t0 + round(float(block.block_duration) * 1e6)
        vec = np.zeros((3, times_us.size))
        for a, axis in enumerate(("gx", "gy", "gz")):
            t, v = gradient_vertices(block, axis, t0)
            if t.size:
                sampled = np.interp(times_us, t, v, left=0.0, right=0.0)
                sampled[times_us < t[0]] = 0.0
                # An event ending off zero holds its last commanded
                # amplitude to the block boundary (continuous gradients:
                # the next block's event picks it up from there).
                sampled[times_us > t[-1]] = 0.0
                hold = (times_us > t[-1]) & (times_us <= block_end)
                sampled[hold] = v[-1]
                vec[a] = sampled
        rotation = getattr(block, "rotation", None)
        norot = bool(norot_flags[start_block + i]) if norot_flags is not None else False
        if rotation is not None and not norot:
            # Physical composition applies the TRANSPOSE of the quaternion's
            # matrix -- the interpreter's convention, validated end-to-end
            # against simulator playout by the SIM rotation suite.
            vec = quaternion_matrix(rotation).T @ vec
        out += vec
        t0 += round(float(block.block_duration) * 1e6)
    return out
