"""Readout families: Cartesian line/echo-train, EPI, non-Cartesian, ZTE.

The builders here are moved verbatim from the per-family plugin helpers so
that the ``.seq`` output is byte- and k-space-identical to the pre-refactor
zoo; only their home and cross-references changed. The Cartesian single-echo
GRE view is the ``num_echoes=1`` case of the echo-train builder.

FSE (CPMG) has moved to ``readout/fse.py`` — the other families here are
pending the same per-module split.
"""

from __future__ import annotations

import numpy as np
import pypulseq as pp

from ...pypulseq import _arbgrad as arbgrad
from .._gradients import SPOIL_FACTOR_X
from .._rf import _excitation_helpers as excitation
from .._system import (
    ceil_to_raster,
    copy_event,
    quantize_readout_timing,
    round_to_raster,
)

# --- ZTE constants ---
ZTE_RF_DURATION_S = 20.0e-6

# --- non-Cartesian trajectory names ---
TRAJECTORY_SPIRAL = "spiral"
TRAJECTORY_ROSETTE = "rosette"


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
    >>> from pulserver.design import _readout as readout
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
    >>> from pulserver.design import _readout as readout
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
    seq : pulserver.pypulseq.Sequence
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
    >>> from pulserver.design import _readout as readout
    >>> opts = pp.Opts()
    >>> echo = readout.line(opts, 0.22, 64, bandwidth_hz_px=125000.0)
    >>> seq = ps.Sequence._unstructured(opts)  # helper appends blocks one at a time
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
    >>> from pulserver.design import _readout as readout
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
        n_samples = max(1, round(total_duration_s / dwell_s))
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
    >>> from pulserver.design import _readout as readout
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
    >>> from pulserver.design import _readout as readout
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
    seq : pulserver.pypulseq.Sequence
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
    >>> from pulserver.design import _readout as readout
    >>> opts = pp.Opts()
    >>> gx_pos, gx_neg, adc = readout.build_epi_readout(opts, "x", 64, 0.22, 125000.0, False)
    >>> blip = readout.build_pe_blip(opts, "y", 0.22)
    >>> seq = ps.Sequence._unstructured(opts)  # helper appends blocks one at a time
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
# Non-Cartesian (arbgrad spiral/rosette + ZTE)
# ---------------------------------------------------------------------------


def trajectory_name(code: float) -> str:
    """Map a protocol code to ``"rosette"`` (>=0.5) or ``"spiral"``.

    Examples
    --------
    >>> from pulserver.design import _readout as readout
    >>> readout.trajectory_name(1.0)
    'rosette'
    """
    return TRAJECTORY_ROSETTE if code >= 0.5 else TRAJECTORY_SPIRAL


def order_mode_name(code: float) -> str:
    """Map a protocol code to ``"golden"`` (>=0.5) or ``"uniform"``.

    Examples
    --------
    >>> from pulserver.design import _readout as readout
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
    >>> from pulserver.design import _readout as readout
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
    >>> from pulserver.design import _readout as readout
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
    >>> from pulserver.design import _readout as readout
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
    >>> from pulserver.design import _readout as readout
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
    >>> from pulserver.design import _readout as readout
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
    """Build the bridged (split-gradient) Cartesian readout used by ``gre_2d.py``.

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
    >>> from pulserver.design import _readout as readout
    >>> ro = readout.unbalanced_line(
    ...     pp.Opts(), 0.22, 64, bandwidth_hz_px=125000.0, slice_thickness_m=5e-3)
    >>> ro["adc"].num_samples
    64
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
    >>> from pulserver.design import _readout as readout
    >>> opts = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> ro = readout.spiral(opts, 0.22, 64, variant="in_out")
    >>> ro["n_samples"] > 0
    True
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
    >>> from pulserver.design import _readout as readout
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
    >>> from pulserver.design import _readout as readout
    >>> opts = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> ro = readout.line(opts, 0.22, 64, bandwidth_hz_px=125000.0)
    >>> gx = ro["gx_echo"]
    >>> waves = readout.wave_gradients(opts, gx.flat_time, gx.rise_time, gx.fall_time,
    ...                                axes=("y", "z"))
    >>> len(waves)
    2
    """
    if n_cycles < 1:
        raise ValueError("n_cycles must be >= 1")
    if not axes or len(axes) > 2:
        raise ValueError("axes must contain one or two channels")

    dt = opts.grad_raster_time
    n_flat = round(flat_time_s / dt)
    n_rise = round(rise_time_s / dt)
    n_fall = round(fall_time_s / dt)

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
