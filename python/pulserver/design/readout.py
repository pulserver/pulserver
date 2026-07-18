"""Readout families: Cartesian line/echo-train, EPI, FSE (CPMG), non-Cartesian, ZTE.

The builders here are moved verbatim from the per-family plugin helpers so
that the ``.seq`` output is byte- and k-space-identical to the pre-refactor
zoo; only their home and cross-references changed. The Cartesian single-echo
GRE view is the ``num_echoes=1`` case of the echo-train builder.
"""

from __future__ import annotations

import math

import numpy as np
import pypulseq as pp

from pulserver import arbgrad
from . import excitation
from .encoding import SPOIL_FACTOR_X, CRUSHER_CYCLES_Z
from .system import (
    DEFAULT_BANDWIDTH_HZ_PX,
    ceil_to_raster,
    copy_event,
    quantize_readout_timing,
    round_to_raster,
)

# --- FSE / CPMG constants (from the FSE refocusing-train helper) ---
RF_REFOCUS_TIME_S = 3.0e-3
RF_REFOCUS_APODIZATION = 0.5
RF_REFOCUS_TIME_BW_PRODUCT = 4.0
CPMG_PHASE_OFFSET_RAD = 1.5707963267948966  # pi/2, constant for all refocusing pulses
DEFAULT_REFOCUS_FLIP_DEG = 180.0
EXCITATION_FLIP_DEG = 90.0  # fixed, not user-selectable (TSEplus.m: flipex=90*pi/180)

_TRAPS_OPT_EX1 = 2.0
_TRAPS_OPT_CA = 0.4
_TRAPS_OPT_CB = 2.0
_TRAPS_OPT_CC = 2.0

# --- ZTE constants ---
ZTE_RF_DURATION_S = 20.0e-6

# --- non-Cartesian trajectory names ---
TRAJECTORY_SPIRAL = "spiral"
TRAJECTORY_ROSETTE = "rosette"


# ---------------------------------------------------------------------------
# Cartesian slice-axis crusher (Z-only, CPMG-safe)
# ---------------------------------------------------------------------------


def build_z_crusher(opts: pp.Opts, slice_thickness_m: float, cycles: float = CRUSHER_CYCLES_Z):
    """Build a slice-axis-only crusher for a slice-selective refocusing pulse.

    Deliberately Z-only (not a 3-axis spoiler): keeping crushers off the
    frequency/phase-encode axes avoids them accumulating rather than
    cancelling across a multi-refocusing CPMG/EPI train, where pypulseq's
    ``calculate_kspace`` models each refocusing pulse as an exact sign flip
    of the running k-space integral.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    slice_thickness_m : float
        Slice thickness (m); crusher area is ``cycles / slice_thickness_m``.
    cycles : float, optional
        Dephasing cycles per slice thickness.

    Returns
    -------
    SimpleNamespace
        The Z-channel crusher trapezoid.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design import readout
    >>> gz = readout.build_z_crusher(pp.Opts(), 5e-3)
    """
    return pp.make_trapezoid(channel="z", area=cycles / slice_thickness_m, system=opts)


# ---------------------------------------------------------------------------
# Cartesian readout / echo-train (single echo is num_echoes=1, flyback=True)
# ---------------------------------------------------------------------------


def compute_readout_and_echo_train(
    opts: pp.Opts,
    ro_axis: str,
    nx_ro: int,
    fov_ro_m: float,
    bandwidth_hz_px: float,
    slice_thickness_m: float,
    num_echoes: int,
    echo_spacing_s: float,
    flyback: bool,
    strict: bool,
    wave: bool = False,
    wave_cycles: int = 10,
    wave_axes: tuple[str, ...] = ("y",),
):
    """Design the readout trapezoid(s), ADC, prephaser, spoiler, and echo spacing.

    Returns ``None`` if infeasible under ``strict``. See :func:`line` for the
    single-echo convenience wrapper.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    ro_axis : str
        Readout gradient channel.
    nx_ro : int
        Readout samples.
    fov_ro_m : float
        Readout field of view (m).
    bandwidth_hz_px : float
        Receiver bandwidth per pixel (Hz).
    slice_thickness_m : float
        Slice thickness (m), used for the readout spoiler area.
    num_echoes : int
        Number of echoes in the train.
    echo_spacing_s : float
        Requested echo spacing (s); only validated for ``num_echoes > 1``.
    flyback : bool
        When True, insert a flyback rewinder between same-polarity echoes.
    strict : bool
        When True, return ``None`` on infeasible echo spacing.

    Returns
    -------
    dict or None
        Readout events and precomputed timings, or ``None`` when infeasible.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design import readout
    >>> echo = readout.compute_readout_and_echo_train(
    ...     pp.Opts(), "x", 128, 0.22, 125000.0, 5e-3, 1, 0.0, True, False)
    >>> echo["adc"].num_samples
    128
    """
    delta_k_ro = 1.0 / fov_ro_m
    readout_area = nx_ro * delta_k_ro
    dwell_s, flat_time_s = quantize_readout_timing(
        nx_ro=nx_ro,
        target_bw_hz_px=bandwidth_hz_px,
        grad_raster_s=opts.grad_raster_time,
        adc_raster_s=opts.adc_raster_time,
        min_flat_time_s=abs(readout_area) / (0.95 * opts.max_grad),
    )

    gx_echo = pp.make_trapezoid(channel=ro_axis, flat_area=readout_area, flat_time=flat_time_s, system=opts)
    gx_echo_neg = pp.scale_grad(gx_echo, -1.0)
    adc = pp.make_adc(num_samples=nx_ro, dwell=dwell_s, delay=gx_echo.rise_time, system=opts)

    gx_pre = pp.make_trapezoid(channel=ro_axis, area=-0.5 * gx_echo.area, system=opts)
    gx_spoil = pp.make_trapezoid(channel=ro_axis, area=SPOIL_FACTOR_X / slice_thickness_m, system=opts)

    gx_rewind = None
    gx_echo_duration_s = pp.calc_duration(gx_echo)
    if flyback:
        gx_rewind = pp.make_trapezoid(channel=ro_axis, area=-gx_echo.area, system=opts)
        min_echo_spacing_s = gx_echo_duration_s + pp.calc_duration(gx_rewind)
    else:
        min_echo_spacing_s = gx_echo_duration_s

    echo_pad_s = 0.0
    if num_echoes > 1:
        echo_pad_s = echo_spacing_s - min_echo_spacing_s
        if echo_pad_s < -1e-9 and strict:
            return None
        if echo_pad_s < 0.0:
            echo_pad_s = 0.0

    adc_center_s = adc.delay + 0.5 * gx_echo.flat_time
    echo_train_span_s = max(0, num_echoes - 1) * echo_spacing_s + gx_echo_duration_s

    waves = None
    if wave:
        waves = wave_gradients(
            opts, gx_echo.flat_time, gx_echo.rise_time, gx_echo.fall_time,
            n_cycles=wave_cycles, axes=wave_axes,
        )

    return {
        "gx_echo": gx_echo,
        "gx_echo_neg": gx_echo_neg,
        "gx_rewind": gx_rewind,
        "adc": adc,
        "gx_pre": gx_pre,
        "gx_spoil": gx_spoil,
        "gx_echo_duration_s": gx_echo_duration_s,
        "echo_pad_s": echo_pad_s,
        "echo_train_span_s": echo_train_span_s,
        "adc_center_s": adc_center_s,
        "wave_gradients": waves,
    }


def line(
    opts: pp.Opts,
    fov_m: float,
    nx: int,
    *,
    bandwidth_hz_px: float,
    axis: str = "x",
    wave: bool = False,
    wave_cycles: int = 10,
    wave_axes: tuple[str, ...] = ("y",),
):
    """Build a single Cartesian readout line (the ``num_echoes=1`` echo train).

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    fov_m : float
        Readout field of view (m).
    nx : int
        Readout samples.
    bandwidth_hz_px : float
        Receiver bandwidth per pixel (Hz).
    axis : str, optional
        Readout gradient channel.

    Returns
    -------
    dict
        Same structure as :func:`compute_readout_and_echo_train` with one echo.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design import readout
    >>> ro = readout.line(pp.Opts(), 0.22, 128, bandwidth_hz_px=125000.0)
    >>> ro["adc"].num_samples
    128
    """
    return compute_readout_and_echo_train(
        opts=opts,
        ro_axis=axis,
        nx_ro=nx,
        fov_ro_m=fov_m,
        bandwidth_hz_px=bandwidth_hz_px,
        slice_thickness_m=fov_m / nx,
        num_echoes=1,
        echo_spacing_s=0.0,
        flyback=True,
        strict=False,
        wave=wave,
        wave_cycles=wave_cycles,
        wave_axes=wave_axes,
    )


def add_echo_train_blocks(seq, echo: dict, num_echoes: int, flyback: bool, rf_phase_rad: float, emit_labels: bool = True) -> None:
    """Append the per-echo readout/ADC (+ inter-echo rewind/pad) blocks.

    Parameters
    ----------
    seq : pulserver.pulseq.Sequence
        Sequence to append blocks to.
    echo : dict
        Output of :func:`compute_readout_and_echo_train`.
    num_echoes : int
        Number of echoes.
    flyback : bool
        Flyback vs bipolar readout polarity.
    rf_phase_rad : float
        ADC phase offset (RF spoiling phase).
    emit_labels : bool, optional
        Emit ``ECO`` labels per echo.

    Returns
    -------
    None

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver import pulseq as ps
    >>> from pulserver.design import readout
    >>> opts = pp.Opts()
    >>> echo = readout.line(opts, 0.22, 64, bandwidth_hz_px=125000.0)
    >>> seq = ps.Sequence(opts)
    >>> readout.add_echo_train_blocks(seq, echo, 1, True, 0.0)
    """
    gx_echo = echo["gx_echo"]
    gx_echo_neg = echo["gx_echo_neg"]
    gx_rewind = echo["gx_rewind"]
    adc = echo["adc"]
    echo_pad = pp.make_delay(echo["echo_pad_s"]) if echo["echo_pad_s"] > 0.0 else None

    for echo_idx in range(num_echoes):
        gx_this = gx_echo if (flyback or echo_idx % 2 == 0) else gx_echo_neg
        adc_this = copy_event(adc)
        adc_this.phase_offset = rf_phase_rad
        if emit_labels:
            label_eco = pp.make_label(type="SET", label="ECO", value=echo_idx)
            seq.add_block(gx_this, adc_this, label_eco)
        else:
            seq.add_block(gx_this, adc_this)

        if echo_idx < num_echoes - 1:
            if flyback:
                seq.add_block(gx_rewind)
            if echo_pad is not None:
                seq.add_block(echo_pad)


# ---------------------------------------------------------------------------
# EPI train
# ---------------------------------------------------------------------------


def build_epi_readout(opts: pp.Opts, ro_axis: str, nx_ro: int, fov_ro_m: float, bandwidth_hz_px: float, ramp_sample: bool):
    """Build the (gx_pos, gx_neg, adc) EPI readout-line trio.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    ro_axis : str
        Readout channel.
    nx_ro : int
        Readout samples.
    fov_ro_m : float
        Readout field of view (m).
    bandwidth_hz_px : float
        Receiver bandwidth per pixel (Hz).
    ramp_sample : bool
        When True, the ADC spans the whole trapezoid (rise+flat+fall).

    Returns
    -------
    gx_pos, gx_neg, adc : SimpleNamespace
        Positive/negative readout gradients and the ADC.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design import readout
    >>> gx_pos, gx_neg, adc = readout.build_epi_readout(pp.Opts(), "x", 64, 0.22, 125000.0, False)
    """
    delta_k_ro = 1.0 / fov_ro_m
    readout_area = nx_ro * delta_k_ro
    dwell_s, flat_time_s = quantize_readout_timing(
        nx_ro=nx_ro,
        target_bw_hz_px=bandwidth_hz_px,
        grad_raster_s=opts.grad_raster_time,
        adc_raster_s=opts.adc_raster_time,
        min_flat_time_s=abs(readout_area) / (0.95 * opts.max_grad),
    )
    gx_pos = pp.make_trapezoid(channel=ro_axis, flat_area=readout_area, flat_time=flat_time_s, system=opts)
    gx_neg = pp.scale_grad(gx_pos, -1.0)

    if ramp_sample:
        total_duration_s = pp.calc_duration(gx_pos)
        n_samples = max(1, int(round(total_duration_s / dwell_s)))
        adc = pp.make_adc(num_samples=n_samples, dwell=dwell_s, delay=0.0, system=opts)
    else:
        adc = pp.make_adc(num_samples=nx_ro, dwell=dwell_s, delay=gx_pos.rise_time, system=opts)

    return gx_pos, gx_neg, adc


def build_pe_blip(opts: pp.Opts, pe_axis: str, fov_pe_m: float):
    """Build the small inter-line phase-encode blip (area = 1 delta_k_pe).

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    pe_axis : str
        Phase-encode channel.
    fov_pe_m : float
        Phase-encode field of view (m).

    Returns
    -------
    SimpleNamespace
        The blip trapezoid.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design import readout
    >>> blip = readout.build_pe_blip(pp.Opts(), "y", 0.22)
    """
    delta_k_pe = 1.0 / fov_pe_m
    return pp.make_trapezoid(channel=pe_axis, area=delta_k_pe, system=opts)


def pre_post_180_delays(half_te_s, d90_s, c90_s, d180_s, c180_s, pre180_fixed_s, post180_fixed_s, strict):
    """Compute the two spin-echo delays that keep the 180 at exactly TE/2.

    Parameters
    ----------
    half_te_s : float
        TE/2 (s).
    d90_s, c90_s : float
        90-pulse duration and center (s).
    d180_s, c180_s : float
        180-pulse duration and center (s).
    pre180_fixed_s, post180_fixed_s : float
        Fixed durations inserted before/after the 180 (crushers, prephasers).
    strict : bool
        When True, return ``None`` if either delay would be negative.

    Returns
    -------
    tuple of float or None
        ``(tau1_s, tau2_s)`` or ``None`` when infeasible under ``strict``.

    Examples
    --------
    >>> from pulserver.design import readout
    >>> readout.pre_post_180_delays(0.02, 1e-3, 5e-4, 1e-3, 5e-4, 1e-3, 1e-3, True)
    (0.0175, 0.0185)
    """
    tau1_s = round_to_raster(half_te_s - (d90_s - c90_s) - pre180_fixed_s - c180_s)
    tau2_s = round_to_raster(half_te_s - (d180_s - c180_s) - post180_fixed_s)
    if (tau1_s < -1e-9 or tau2_s < -1e-9) and strict:
        return None
    return max(tau1_s, 0.0), max(tau2_s, 0.0)


def add_epi_train_blocks(seq, gx_pos, gx_neg, gy_blip, adc, n_lines, start_polarity_positive=True, emit_lin_labels=True, lin_start=0) -> None:
    """Append ``n_lines`` alternating-polarity readout blocks with inter-line blips.

    Parameters
    ----------
    seq : pulserver.pulseq.Sequence
        Sequence to append to.
    gx_pos, gx_neg : SimpleNamespace
        Positive/negative readout gradients.
    gy_blip : SimpleNamespace
        Inter-line phase-encode blip.
    adc : SimpleNamespace
        ADC event.
    n_lines : int
        Number of EPI lines.
    start_polarity_positive : bool, optional
        Polarity of the first line.
    emit_lin_labels : bool, optional
        Emit ``LIN`` labels.
    lin_start : int, optional
        First ``LIN`` label value.

    Returns
    -------
    None

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver import pulseq as ps
    >>> from pulserver.design import readout
    >>> opts = pp.Opts()
    >>> gx_pos, gx_neg, adc = readout.build_epi_readout(opts, "x", 64, 0.22, 125000.0, False)
    >>> blip = readout.build_pe_blip(opts, "y", 0.22)
    >>> seq = ps.Sequence(opts)
    >>> readout.add_epi_train_blocks(seq, gx_pos, gx_neg, blip, adc, 4)
    """
    for i in range(n_lines):
        positive = start_polarity_positive == (i % 2 == 0)
        gx_this = gx_pos if positive else gx_neg
        adc_this = copy_event(adc)
        if emit_lin_labels:
            label_lin = pp.make_label(type="SET", label="LIN", value=lin_start + i)
            seq.add_block(gx_this, adc_this, label_lin)
        else:
            seq.add_block(gx_this, adc_this)
        if i < n_lines - 1:
            seq.add_block(gy_blip)


# ---------------------------------------------------------------------------
# FSE / CPMG refocusing train
# ---------------------------------------------------------------------------


def build_refocusing_pulse(opts: pp.Opts, slice_thickness_m: float):
    """Build the reference (rf180, gz180) CPMG refocusing pulse at 180 deg.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    slice_thickness_m : float
        Slice thickness (m).

    Returns
    -------
    rf180, gz180 : SimpleNamespace
        Refocusing RF (phase offset pi/2) and its slice-select gradient.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design import readout
    >>> rf180, gz180 = readout.build_refocusing_pulse(pp.Opts(), 5e-3)
    """
    rf180, gz180, _ = pp.make_sinc_pulse(
        flip_angle=3.14159265358979,
        duration=RF_REFOCUS_TIME_S,
        slice_thickness=slice_thickness_m,
        apodization=RF_REFOCUS_APODIZATION,
        time_bw_product=RF_REFOCUS_TIME_BW_PRODUCT,
        system=opts,
        use="refocusing",
        return_gz=True,
    )
    rf180.phase_offset = CPMG_PHASE_OFFSET_RAD
    return rf180, gz180


def scale_refocusing_pulse(rf180_ref, flip_deg: float):
    """Return a copy of the 180 deg reference pulse scaled to ``flip_deg``.

    Parameters
    ----------
    rf180_ref : SimpleNamespace
        Reference refocusing pulse.
    flip_deg : float
        Target refocusing flip angle (deg).

    Returns
    -------
    SimpleNamespace
        Envelope-scaled copy (timing/phase untouched).

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design import readout
    >>> rf180, _ = readout.build_refocusing_pulse(pp.Opts(), 5e-3)
    >>> rf120 = readout.scale_refocusing_pulse(rf180, 120.0)
    """
    rf_scaled = type(rf180_ref)()
    rf_scaled.__dict__.update(rf180_ref.__dict__)
    rf_scaled.signal = rf180_ref.signal * (flip_deg / 180.0)
    return rf_scaled


def build_refocus_flip_schedule(etl: int, refocus_flip_deg: float, variable: bool) -> list[float]:
    """Return the per-echo refocusing flip angle (deg), length ``etl``.

    ``variable=False`` gives a constant angle; ``variable=True`` gives a
    TRAPS 'opt' ramp decaying from near 180 deg toward ``refocus_flip_deg``.

    Parameters
    ----------
    etl : int
        Echo train length.
    refocus_flip_deg : float
        Target/plateau refocusing angle (deg).
    variable : bool
        Enable the TRAPS variable-angle ramp.

    Returns
    -------
    list of float
        Per-echo flip angles.

    Examples
    --------
    >>> from pulserver.design import readout
    >>> readout.build_refocus_flip_schedule(3, 120.0, False)
    [120.0, 120.0, 120.0]
    """
    if not variable or etl <= 1:
        return [refocus_flip_deg] * etl

    rf1 = (
        90.0
        + refocus_flip_deg / 2.0
        + _TRAPS_OPT_CA * ((_TRAPS_OPT_CC - 1.0) / _TRAPS_OPT_CC) ** _TRAPS_OPT_EX1 * (90.0 - refocus_flip_deg / 2.0)
    )
    dr = rf1 - refocus_flip_deg
    schedule = [rf1]
    for k in range(2, etl + 1):
        schedule.append(refocus_flip_deg + dr / (_TRAPS_OPT_CB ** (k - 0.5)))
    return schedule


def fse_timing(esp_s, d90_s, c90_s, d180_s, c180_s, crusher_dur_s, half_loop_fixed_s, strict):
    """Return ``(tau_first_s, tau_steady_s)`` for the CPMG refocusing train.

    Parameters
    ----------
    esp_s : float
        Echo spacing (s).
    d90_s, c90_s : float
        90-pulse duration and center (s).
    d180_s, c180_s : float
        180-pulse duration and center (s).
    crusher_dur_s : float
        Crusher duration on each side of a 180 (s).
    half_loop_fixed_s : float
        Fixed readout-side duration per ESP/2 half (prephase/rephase + ADC center).
    strict : bool
        When True, return ``None`` on infeasible timing.

    Returns
    -------
    tuple of float or None
        ``(tau_first_s, tau_steady_s)`` or ``None``.

    Examples
    --------
    >>> from pulserver.design import readout
    >>> readout.fse_timing(0.01, 1e-3, 5e-4, 3e-3, 1.5e-3, 5e-4, 1e-3, False)
    (0.0035, 0.001)
    """
    half_esp_s = esp_s / 2.0
    tau_first_s = round_to_raster(half_esp_s - (d90_s - c90_s) - crusher_dur_s - c180_s)
    tau_steady_s = round_to_raster(half_esp_s - (d180_s - c180_s) - crusher_dur_s - half_loop_fixed_s)
    if (tau_first_s < -1e-9 or tau_steady_s < -1e-9) and strict:
        return None
    return max(tau_first_s, 0.0), max(tau_steady_s, 0.0)


def surgery_ramp_time_s(opts: pp.Opts, amplitudes_hz_m) -> float:
    """Ramp time for a plain 0->amplitude (or reverse) transition, raster-ceiled.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    amplitudes_hz_m : iterable of float
        Amplitudes involved (Hz/m).

    Returns
    -------
    float
        Slew-feasible ramp time (s), rounded up to the gradient raster.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design import readout
    >>> t = readout.surgery_ramp_time_s(pp.Opts(), [1e6])
    """
    max_amp = max((abs(a) for a in amplitudes_hz_m), default=0.0)
    raster = opts.grad_raster_time
    if max_amp <= 0.0 or opts.max_slew <= 0.0:
        return raster
    dg = max_amp / opts.max_slew
    return math.ceil(dg / raster - 1e-9) * raster


def _fuse(channel: str, times, amplitudes):
    return pp.make_extended_trapezoid(channel=channel, times=np.asarray(times), amplitudes=np.asarray(amplitudes))


def _bridge(opts: pp.Opts, channel: str, area: float, grad_start: float, grad_end: float, target_duration_s: float):
    """Bridge ``grad_start``->``grad_end`` with exact net ``area`` over exact duration.

    See the FSE gradient-surgery design notes: the plateau is re-solved in
    closed form when ``target_duration_s`` exceeds the shortest feasible
    bridge, so padding never silently changes the net area.
    """
    _, times_min, amps_min = pp.make_extended_trapezoid_area(
        area=area, channel=channel, grad_start=grad_start, grad_end=grad_end, system=opts
    )
    times_min = [float(t) for t in times_min]
    min_duration = times_min[-1]
    if target_duration_s < min_duration - 1e-9:
        raise ValueError(
            f"target duration {target_duration_s * 1e6:.1f} us is shorter than the shortest feasible "
            f"{channel}-bridge ({min_duration * 1e6:.1f} us) for area={area:.3f}"
        )

    if len(times_min) == 3:
        rise_s = times_min[1] - times_min[0]
        fall_s = times_min[2] - times_min[1]
    else:
        rise_s = times_min[1] - times_min[0]
        fall_s = times_min[3] - times_min[2]

    flat_s = max(0.0, target_duration_s - rise_s - fall_s)
    denom = 0.5 * rise_s + flat_s + 0.5 * fall_s
    if denom <= 0.0:
        raise ValueError(f"degenerate {channel}-bridge duration for area={area:.3f}")
    plateau = (area - 0.5 * rise_s * grad_start - 0.5 * fall_s * grad_end) / denom

    if abs(plateau) > opts.max_grad + 1e-6:
        raise ValueError(f"re-solved {channel}-bridge plateau {plateau:.1f} exceeds max_grad")
    if rise_s > 0.0 and abs(plateau - grad_start) / rise_s > opts.max_slew * (1.0 + 1e-6):
        raise ValueError(f"re-solved {channel}-bridge rise leg exceeds max_slew")
    if fall_s > 0.0 and abs(grad_end - plateau) / fall_s > opts.max_slew * (1.0 + 1e-6):
        raise ValueError(f"re-solved {channel}-bridge fall leg exceeds max_slew")

    if flat_s > 0.0:
        times = [0.0, rise_s, rise_s + flat_s, rise_s + flat_s + fall_s]
        amplitudes = [grad_start, plateau, plateau, grad_end]
    else:
        times = [0.0, rise_s, rise_s + fall_s]
        amplitudes = [grad_start, plateau, grad_end]
    return _fuse(channel, times, amplitudes)


def build_slice_surgery_chain(opts, gz_ex_amp, gz_ref_amp, crusher_area, t_ex_wd_s, t_ref_wd_s, t_spex_s, t_sp_s, dg_s):
    """Z-channel slice-select + crusher gradient-surgery chain (GS1..GS7).

    Built once per shot; shape depends only on the fixed slice-select
    amplitudes and crusher area. See the FSE design notes for the role of
    each fused segment.

    Returns
    -------
    dict
        ``GS1``, ``GS2``, ``GS2_FALL``, ``GS3``, ``GS4``, ``GS5``, ``GS7``.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design import readout
    >>> opts = pp.Opts()
    >>> chain = readout.build_slice_surgery_chain(
    ...     opts, 1e6, 1e6, 400.0, 3e-3, 3e-3, 2e-3, 2e-3, 2e-4)
    >>> sorted(chain)
    ['GS1', 'GS2', 'GS2_FALL', 'GS3', 'GS4', 'GS5', 'GS7']
    """
    gs1 = _fuse("z", [0.0, dg_s], [0.0, gz_ex_amp])
    gs2 = _fuse("z", [0.0, t_ex_wd_s], [gz_ex_amp, gz_ex_amp])
    gs2_fall = _fuse("z", [0.0, dg_s], [gz_ex_amp, 0.0])
    gs3 = _bridge(opts, "z", crusher_area, 0.0, gz_ref_amp, t_spex_s)
    gs4 = _fuse("z", [0.0, t_ref_wd_s], [gz_ref_amp, gz_ref_amp])
    gs5 = _bridge(opts, "z", crusher_area, gz_ref_amp, 0.0, t_sp_s)
    gs7 = _bridge(opts, "z", crusher_area, 0.0, gz_ref_amp, t_sp_s)
    return {"GS1": gs1, "GS2": gs2, "GS2_FALL": gs2_fall, "GS3": gs3, "GS4": gs4, "GS5": gs5, "GS7": gs7}


def build_kz_crusher_bridge(opts, gz_ref_amp, crusher_area, kz_area, t_sp_s):
    """3D-only GS5/GS7 combining the crusher with a fixed partition (kz) encode.

    GS5 carries ``+kz_area``, GS7 carries ``-kz_area`` (net zero per cycle),
    while the gradient amplitude still returns to 0/``gz_ref_amp`` at the
    block boundary.

    Returns
    -------
    gs5, gs7 : SimpleNamespace
        The two fused Z bridges.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design import readout
    >>> gs5, gs7 = readout.build_kz_crusher_bridge(pp.Opts(), 1e6, 400.0, 50.0, 2e-3)
    """
    gs5 = _bridge(opts, "z", crusher_area + kz_area, gz_ref_amp, 0.0, t_sp_s)
    gs7 = _bridge(opts, "z", crusher_area - kz_area, 0.0, gz_ref_amp, t_sp_s)
    return gs5, gs7


def build_surgery_readout(opts, ro_axis, nx_ro, fov_ro_m, bandwidth_hz_px):
    """Readout plateau + ADC for the gradient-surgery FSE design.

    Returns the flat portion (amplitude + duration) plus the ADC; the ramps
    are fused into the prephase/rephase by :func:`build_readout_surgery_chain`.

    Returns
    -------
    dict
        ``adc``, ``readout_area``, ``gx_flat_amp``, ``flat_time_s``, ``adc_center_s``.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design import readout
    >>> ro = readout.build_surgery_readout(pp.Opts(), "x", 128, 0.22, 125000.0)
    >>> ro["adc"].num_samples
    128
    """
    delta_k_ro = 1.0 / fov_ro_m
    readout_area = nx_ro * delta_k_ro
    dwell_s, flat_time_s = quantize_readout_timing(
        nx_ro=nx_ro,
        target_bw_hz_px=bandwidth_hz_px,
        grad_raster_s=opts.grad_raster_time,
        adc_raster_s=opts.adc_raster_time,
        min_flat_time_s=abs(readout_area) / (0.95 * opts.max_grad),
    )
    gx_flat_amp = readout_area / flat_time_s
    adc = pp.make_adc(num_samples=nx_ro, dwell=dwell_s, delay=0.0, system=opts)
    return {
        "adc": adc,
        "readout_area": readout_area,
        "gx_flat_amp": gx_flat_amp,
        "flat_time_s": flat_time_s,
        "adc_center_s": 0.5 * flat_time_s,
    }


def build_readout_surgery_chain(opts, ro_axis, readout_area, gx_flat_amp, flat_time_s, t_sp_s):
    """X-channel readout prephase/rephase gradient-surgery chain (GR5/GR6/GR7).

    Fuses the ``-0.5 * readout_area`` prephase/rephase directly into the
    readout plateau ramp. Crushing stays z-only (no x/y spoil), preserving
    the k-space-symmetric CPMG design.

    Returns
    -------
    dict
        ``GR5``, ``GR6``, ``GR7``.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design import readout
    >>> opts = pp.Opts()
    >>> ro = readout.build_surgery_readout(opts, "x", 128, 0.22, 125000.0)
    >>> chain = readout.build_readout_surgery_chain(
    ...     opts, "x", ro["readout_area"], ro["gx_flat_amp"], ro["flat_time_s"], 2e-3)
    >>> sorted(chain)
    ['GR5', 'GR6', 'GR7']
    """
    prephase_area = -0.5 * readout_area
    gr5 = _bridge(opts, ro_axis, prephase_area, 0.0, gx_flat_amp, t_sp_s)
    gr6 = _fuse(ro_axis, [0.0, flat_time_s], [gx_flat_amp, gx_flat_amp])
    gr7 = _bridge(opts, ro_axis, prephase_area, gx_flat_amp, 0.0, t_sp_s)
    return {"GR5": gr5, "GR6": gr6, "GR7": gr7}


def surgery_timing(esp_s, t_ex_wd_s, c90_s, dg_s, d_gz_reph_s, t_ref_wd_s, c180_s, adc_center_s, strict):
    """Return ``(t_spex_s, t_sp_s)``, the fused segments' durations for FSE surgery.

    Replaces :func:`fse_timing`'s delay pair: the crusher/prephase segment's
    duration directly consumes the entire available half-ESP budget.

    Returns
    -------
    tuple of float or None
        ``(t_spex_s, t_sp_s)`` or ``None`` when infeasible under ``strict``.

    Examples
    --------
    >>> from pulserver.design import readout
    >>> readout.surgery_timing(0.01, 3e-3, 1.5e-3, 2e-4, 5e-4, 3e-3, 1.5e-3, 1e-3, False)
    (0.0018, 0.001)
    """
    half_esp_s = esp_s / 2.0
    t_spex_s = round_to_raster(half_esp_s - (t_ex_wd_s - c90_s) - dg_s - d_gz_reph_s - c180_s)
    t_sp_s = round_to_raster(half_esp_s - (t_ref_wd_s - c180_s) - adc_center_s)
    if (t_spex_s < -1e-9 or t_sp_s < -1e-9) and strict:
        return None
    return max(t_spex_s, 0.0), max(t_sp_s, 0.0)


# ---------------------------------------------------------------------------
# Non-Cartesian (arbgrad spiral/rosette + ZTE)
# ---------------------------------------------------------------------------


def trajectory_name(code: float) -> str:
    """Map a protocol code to ``"rosette"`` (>=0.5) or ``"spiral"``.

    Examples
    --------
    >>> from pulserver.design import readout
    >>> readout.trajectory_name(1.0)
    'rosette'
    """
    return TRAJECTORY_ROSETTE if code >= 0.5 else TRAJECTORY_SPIRAL


def order_mode_name(code: float) -> str:
    """Map a protocol code to ``"golden"`` (>=0.5) or ``"uniform"``.

    Examples
    --------
    >>> from pulserver.design import readout
    >>> readout.order_mode_name(0.0)
    'uniform'
    """
    return "golden" if code >= 0.5 else "uniform"


def build_base_waveform(opts: pp.Opts, trajectory: str, fov_m: float, n_pix: int):
    """Design the single base (x, y) non-Cartesian waveform in SI units (T/m).

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    trajectory : str
        ``"spiral"`` or ``"rosette"``.
    fov_m : float
        Field of view (m).
    n_pix : int
        Matrix size.

    Returns
    -------
    k0 : float
        Initial k-space offset.
    grad_si_xy : numpy.ndarray
        ``(n_samp, 2)`` gradient waveform (T/m).
    n_shots : int
        Number of shots to fully sample.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design import readout
    >>> opts = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> k0, grad, n = readout.build_base_waveform(opts, "spiral", 0.22, 64)
    >>> grad.shape[1]
    2
    """
    slew_limit_hz_pix_s = opts.max_slew * fov_m / n_pix
    grad_limit_hz_pix = opts.max_grad * fov_m / n_pix
    dt_s = opts.grad_raster_time

    if trajectory == TRAJECTORY_ROSETTE:
        wf = arbgrad.rosette(fov_m, n_pix, slew_limit_hz_pix_s, grad_limit_hz_pix, dt_s)
    elif trajectory == TRAJECTORY_SPIRAL:
        wf = arbgrad.spiral(fov_m, n_pix, slew_limit_hz_pix_s, grad_limit_hz_pix, dt_s)
    else:
        raise ValueError(f"Unknown trajectory '{trajectory}' (expected 'spiral' or 'rosette')")

    grad_si = arbgrad.to_gradient_tesla_per_meter(wf, fov_m, n_pix, gamma=opts.gamma)
    return wf.k0, grad_si[:, :2], wf.n_shots


def build_readout_gradients(opts: pp.Opts, grad_si_xy: np.ndarray):
    """Build the (gx, gy) arbitrary-gradient events for a base waveform.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    grad_si_xy : numpy.ndarray
        ``(n_samp, 2)`` gradient waveform (T/m).

    Returns
    -------
    gx, gy : SimpleNamespace
        Arbitrary-gradient events.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseq as pp
    >>> from pulserver.design import readout
    >>> opts = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> _, grad, _ = readout.build_base_waveform(opts, "spiral", 0.22, 64)
    >>> gx, gy = readout.build_readout_gradients(opts, grad)
    """
    gx = pp.make_arbitrary_grad(channel="x", waveform=np.ascontiguousarray(grad_si_xy[:, 0]), system=opts)
    gy = pp.make_arbitrary_grad(channel="y", waveform=np.ascontiguousarray(grad_si_xy[:, 1]), system=opts)
    return gx, gy


def build_readout_adc(opts: pp.Opts, n_samples: int):
    """Build an ADC spanning the whole non-Cartesian base waveform.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    n_samples : int
        Number of ADC samples (one per gradient raster step).

    Returns
    -------
    SimpleNamespace
        The ADC event (no delay; k-space center is the first sample).

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design import readout
    >>> adc = readout.build_readout_adc(pp.Opts(), 2048)
    >>> adc.num_samples
    2048
    """
    return pp.make_adc(num_samples=n_samples, dwell=opts.grad_raster_time, delay=0.0, system=opts)


def build_hard_pulse(opts: pp.Opts, flip_deg: float):
    """Build the short non-selective ZTE excitation pulse.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    flip_deg : float
        Flip angle (deg).

    Returns
    -------
    SimpleNamespace
        The block RF event (20 us).

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design import readout
    >>> rf = readout.build_hard_pulse(pp.Opts(), 3.0)
    """
    return excitation.nonselective(opts, flip_deg, duration_s=ZTE_RF_DURATION_S, use="excitation")


def build_half_echo_readout(opts: pp.Opts, ro_axis: str, nx_ro: int, fov_ro_m: float):
    """Build the ZTE (gx, adc, gap_s) half-echo (0 -> kmax) readout.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    ro_axis : str
        Readout channel.
    nx_ro : int
        Readout samples.
    fov_ro_m : float
        Readout field of view (m).

    Returns
    -------
    tuple or None
        ``(gx, adc, gap_s)`` or ``None`` if the readout is infeasible.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design import readout
    >>> opts = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> result = readout.build_half_echo_readout(opts, "x", 128, 0.22)
    """
    kmax = 0.5 * nx_ro * (1.0 / fov_ro_m)
    grad_hz_per_m = opts.max_grad

    gap_s = opts.rf_dead_time + opts.rf_ringdown_time + opts.adc_dead_time
    t_adc_start_s = 0.5 * ZTE_RF_DURATION_S + gap_s
    t_adc_end_s = kmax / grad_hz_per_m
    adc_duration_s = t_adc_end_s - t_adc_start_s
    if adc_duration_s <= 0.0:
        return None

    dwell_s = adc_duration_s / nx_ro
    dwell_s = max(opts.adc_raster_time, round(dwell_s / opts.adc_raster_time) * opts.adc_raster_time)
    adc_duration_s = dwell_s * nx_ro

    flat_time_s = t_adc_start_s + adc_duration_s
    flat_time_s = ceil_to_raster(flat_time_s, opts.grad_raster_time)

    gx = pp.make_trapezoid(channel=ro_axis, amplitude=grad_hz_per_m, flat_time=flat_time_s, system=opts)
    adc = pp.make_adc(num_samples=nx_ro, dwell=dwell_s, delay=gx.rise_time + t_adc_start_s, system=opts)

    return gx, adc, gap_s


def unbalanced_line(
    opts: pp.Opts,
    fov_m: float,
    nx: int,
    *,
    bandwidth_hz_px: float,
    slice_thickness_m: float,
    axis: str = "x",
    spoil_factor: float = SPOIL_FACTOR_X,
):
    """Build the bridged (split-gradient) Cartesian readout used by ``gre.py``.

    The readout trapezoid is split at the end of its flat top
    (``pp.split_gradient_at``): the returned ``gx`` carries rise + flat only,
    and the spoiler is an extended trapezoid (``make_extended_trapezoid_area``)
    that bridges directly from the readout plateau amplitude through the
    spoil area down to zero — no intermediate return to zero, matching the
    GE EPIC fixture's smooth crusher waveform.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    fov_m : float
        Readout field of view (m).
    nx : int
        Readout samples.
    bandwidth_hz_px : float
        Receiver bandwidth per pixel (Hz).
    slice_thickness_m : float
        Slice thickness (m); spoil area is ``spoil_factor / slice_thickness_m``.
    axis : str, optional
        Readout gradient channel.
    spoil_factor : float, optional
        Spoiling cycles per slice thickness on the readout axis.

    Returns
    -------
    dict
        ``gx`` (rise+flat split part), ``adc``, ``gx_pre``, ``gx_spoil``
        (plateau-bridged), and ``adc_center_s``.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design import readout
    >>> ro = readout.unbalanced_line(
    ...     pp.Opts(), 0.22, 64, bandwidth_hz_px=125000.0, slice_thickness_m=5e-3)
    >>> ro["adc"].num_samples
    64

    .. plot::
       :include-source: false

       import numpy as np
       import pypulseq as pp
       from pulserver.design import readout
       import matplotlib.pyplot as plt
       ro = readout.unbalanced_line(pp.Opts(), 0.22, 64,
                                    bandwidth_hz_px=125000.0, slice_thickness_m=5e-3)
       g = ro["gx_spoil"]
       plt.plot(g.tt * 1e3, g.waveform)
       plt.xlabel("t [ms]"); plt.ylabel("G [Hz/m]")
    """
    delta_k_ro = 1.0 / fov_m
    readout_area = nx * delta_k_ro
    dwell_s, flat_time_s = quantize_readout_timing(
        nx_ro=nx,
        target_bw_hz_px=bandwidth_hz_px,
        grad_raster_s=opts.grad_raster_time,
        adc_raster_s=opts.adc_raster_time,
        min_flat_time_s=abs(readout_area) / (0.95 * opts.max_grad),
    )

    gx_full = pp.make_trapezoid(channel=axis, flat_area=readout_area, flat_time=flat_time_s, system=opts)
    gx_parts = pp.split_gradient_at(gx_full, gx_full.rise_time + gx_full.flat_time, system=opts)
    gx = gx_parts[0]
    adc = pp.make_adc(num_samples=nx, dwell=dwell_s, delay=gx_full.rise_time, system=opts)

    gx_pre = pp.make_trapezoid(channel=axis, area=-0.5 * gx_full.area, system=opts)
    gx_spoil, _, _ = pp.make_extended_trapezoid_area(
        area=spoil_factor / slice_thickness_m,
        channel=axis,
        grad_start=gx_full.amplitude,
        grad_end=0.0,
        system=opts,
    )

    return {
        "gx": gx,
        "adc": adc,
        "gx_pre": gx_pre,
        "gx_spoil": gx_spoil,
        "adc_center_s": adc.delay + 0.5 * gx_full.flat_time,
    }


def _spiral_variant_waveform(opts: pp.Opts, grad_si_xy: np.ndarray, variant: str) -> np.ndarray:
    """Assemble the requested spiral variant from the base (spiral-out) arm.

    ``reversed`` and ``in_out`` join arms at the k-space center, where the
    gradient is near zero, so they need no bridge. ``out_in`` retraces the
    base arm inward after a per-axis max-slew linear bridge that reverses
    the gradient at the k-space edge (retraced out-in, e.g. for trajectory
    calibration); the bridge samples carry no ADC-relevant assumption — the
    actual k-locations always come from integrating the returned waveform.
    """
    if variant == "forward":
        return grad_si_xy
    if variant == "reversed":
        return -grad_si_xy[::-1]
    if variant == "in_out":
        return np.concatenate([grad_si_xy[::-1], grad_si_xy], axis=0)
    if variant == "out_in":
        g_end = grad_si_xy[-1]
        g_next = -g_end
        max_slew_si = opts.max_slew / opts.gamma
        dt = opts.grad_raster_time
        n_bridge = int(np.ceil(np.max(np.abs(g_next - g_end)) / (max_slew_si * dt)))
        bridge = np.linspace(g_end, g_next, n_bridge + 2, axis=0)[1:-1]
        return np.concatenate([grad_si_xy, bridge, -grad_si_xy[::-1]], axis=0)
    raise ValueError(f"Unknown spiral variant '{variant}' (expected 'forward', 'reversed', 'in_out', or 'out_in')")


def spiral(opts: pp.Opts, fov_m: float, n_pix: int, *, variant: str = "forward"):
    """Build a spiral readout (gradients + ADC) with trajectory variants.

    ``variant``:

    - ``"forward"`` — spiral-out, k-space center to edge (base arm).
    - ``"reversed"`` — spiral-in: the time-reversed, negated arm (edge to
      center; the sequence must prephase to the k-space edge first).
    - ``"in_out"`` — spiral-in on the mirrored arm followed by spiral-out
      (prephase to ``-k_edge``; the two arms join at the center where the
      gradient is near zero).
    - ``"out_in"`` — spiral-out then a max-slew reversal bridge and an
      inward retrace of the same arm (starts and ends at the center).

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    fov_m : float
        Field of view (m).
    n_pix : int
        Matrix size.
    variant : str, optional
        Trajectory variant (see above).

    Returns
    -------
    dict
        ``gx``, ``gy`` (arbitrary-gradient events), ``adc`` (spanning the
        whole waveform), ``n_shots``, ``k0``, ``n_samples``, and ``variant``.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design import readout
    >>> opts = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> ro = readout.spiral(opts, 0.22, 64, variant="in_out")
    >>> ro["n_samples"] > 0
    True

    .. plot::
       :include-source: false

       import numpy as np
       import pypulseq as pp
       from pulserver.design import readout
       import matplotlib.pyplot as plt
       opts = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       ro = readout.spiral(opts, 0.22, 64, variant="in_out")
       k = np.cumsum(np.stack([ro["gx"].waveform, ro["gy"].waveform], 1), 0)
       plt.plot(k[:, 0], k[:, 1]); plt.axis("equal")
       plt.xlabel("kx (a.u.)"); plt.ylabel("ky (a.u.)")
    """
    k0, grad_si_xy, n_shots = build_base_waveform(opts, TRAJECTORY_SPIRAL, fov_m, n_pix)
    waveform = _spiral_variant_waveform(opts, grad_si_xy, variant)
    gx, gy = build_readout_gradients(opts, waveform)
    adc = build_readout_adc(opts, waveform.shape[0])
    return {
        "gx": gx,
        "gy": gy,
        "adc": adc,
        "n_shots": n_shots,
        "k0": k0,
        "n_samples": waveform.shape[0],
        "variant": variant,
    }


def rosette(opts: pp.Opts, fov_m: float, n_pix: int):
    """Build a rosette readout (gradients + ADC).

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    fov_m : float
        Field of view (m).
    n_pix : int
        Matrix size.

    Returns
    -------
    dict
        ``gx``, ``gy``, ``adc``, ``n_shots``, ``k0``, ``n_samples``.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design import readout
    >>> opts = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> ro = readout.rosette(opts, 0.22, 64)
    >>> ro["n_shots"] >= 1
    True
    """
    k0, grad_si_xy, n_shots = build_base_waveform(opts, TRAJECTORY_ROSETTE, fov_m, n_pix)
    gx, gy = build_readout_gradients(opts, grad_si_xy)
    adc = build_readout_adc(opts, grad_si_xy.shape[0])
    return {
        "gx": gx,
        "gy": gy,
        "adc": adc,
        "n_shots": n_shots,
        "k0": k0,
        "n_samples": grad_si_xy.shape[0],
    }


def wave_gradients(
    opts: pp.Opts,
    flat_time_s: float,
    rise_time_s: float,
    fall_time_s: float,
    *,
    n_cycles: int = 10,
    axes: tuple[str, ...] = ("y",),
    max_wave_grad_hz_m: float | None = None,
):
    """Build wave-CAIPI sinusoidal gradients spanning a Cartesian readout.

    Follows ``refcode/wave-haste`` (``script_wave_haste_sequence.m``): a
    sinusoid with ``n_cycles`` periods plays during the readout flat top,
    zero-padded over the readout rise/fall ramps so the events span exactly
    the readout trapezoid and start/end at zero. With two ``axes`` the
    second axis plays the quadrature (cosine) waveform — its plateau value
    at the flat-top edges is reached/left via linear ramps over the readout
    rise/fall times, keeping block boundaries at zero amplitude. The
    amplitude is ``min(max_wave_grad, derated max_slew / omega)`` (slew
    limited when the cycles are fast).

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits.
    flat_time_s : float
        Readout flat-top duration the sinusoid spans (s).
    rise_time_s, fall_time_s : float
        Readout trapezoid ramp times used as zero/ramp padding (s).
    n_cycles : int, optional
        Number of sinusoid periods during the flat top.
    axes : tuple of str, optional
        Wave axes: first gets the sine, an optional second the cosine.
    max_wave_grad_hz_m : float or None, optional
        Wave amplitude cap (Hz/m); defaults to half the system limit.

    Returns
    -------
    list of SimpleNamespace
        One arbitrary-gradient event per axis, each spanning
        ``rise + flat + fall``.

    Examples
    --------
    >>> import pypulseq as pp
    >>> from pulserver.design import readout
    >>> opts = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> ro = readout.line(opts, 0.22, 64, bandwidth_hz_px=125000.0)
    >>> gx = ro["gx_echo"]
    >>> waves = readout.wave_gradients(opts, gx.flat_time, gx.rise_time, gx.fall_time,
    ...                                axes=("y", "z"))
    >>> len(waves)
    2

    .. plot::
       :include-source: false

       import numpy as np
       import pypulseq as pp
       from pulserver.design import readout
       import matplotlib.pyplot as plt
       opts = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
       ro = readout.line(opts, 0.22, 64, bandwidth_hz_px=125000.0)
       gx = ro["gx_echo"]
       gy, gz = readout.wave_gradients(opts, gx.flat_time, gx.rise_time, gx.fall_time,
                                       axes=("y", "z"))
       t = np.arange(gy.waveform.size) * opts.grad_raster_time * 1e3
       plt.plot(t, gy.waveform, label="sine")
       plt.plot(t, gz.waveform, label="cosine")
       plt.legend(); plt.xlabel("t [ms]"); plt.ylabel("G [Hz/m]")
    """
    if n_cycles < 1:
        raise ValueError("n_cycles must be >= 1")
    if not axes or len(axes) > 2:
        raise ValueError("axes must contain one or two channels")

    dt = opts.grad_raster_time
    n_flat = int(round(flat_time_s / dt))
    n_rise = int(round(rise_time_s / dt))
    n_fall = int(round(fall_time_s / dt))

    omega = 2.0 * np.pi * n_cycles / flat_time_s
    amp_cap = max_wave_grad_hz_m if max_wave_grad_hz_m is not None else 0.5 * opts.max_grad
    amp = min(amp_cap, 0.9 * opts.max_slew / omega)

    t_flat = np.arange(n_flat + 1) * dt
    events = []
    for i, axis in enumerate(axes):
        if i == 0:
            flat_wave = amp * np.sin(omega * t_flat)
            pre = np.zeros(max(n_rise - 1, 0))
            post = np.zeros(max(n_fall - 1, 0))
        else:
            flat_wave = amp * np.cos(omega * t_flat)
            pre = amp * np.linspace(0.0, 1.0, max(n_rise - 1, 0) + 2)[1:-1]
            post = amp * np.linspace(1.0, 0.0, max(n_fall - 1, 0) + 2)[1:-1]
        waveform = np.concatenate([[0.0], pre, flat_wave, post, [0.0]])
        events.append(pp.make_arbitrary_grad(channel=axis, waveform=waveform, system=opts))
    return events
