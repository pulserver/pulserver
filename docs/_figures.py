"""Figure helpers for the documentation's ``.. plot::`` directives.

Not part of the shipped package. Everything here exists so that a docstring
can render an RF pulse *and the profile it actually produces* in three or four
lines, instead of carrying a simulator inline.

The simulator is a plain hard-pulse-approximation Bloch integrator: at each
raster step the magnetisation is rotated about the instantaneous effective
field. That is exact in the limit of small steps and, unlike the small-tip
(Fourier) approximation, stays correct for refocusing, inversion and adiabatic
pulses — which is the whole point of showing the profile.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

GAMMA_HZ_PER_T = 42.576e6


# --------------------------------------------------------------------------
# event sampling
# --------------------------------------------------------------------------


def _rf_grid(rf, dt=None):
    """Return ``(t, b1)``: uniform time base (s) and complex B1 (Hz)."""
    signal = np.asarray(rf.signal, dtype=complex)
    times = np.asarray(rf.t, dtype=float)
    dt = float(np.min(np.diff(times))) if dt is None else float(dt)
    grid = np.arange(times[0], times[-1] + 0.5 * dt, dt)
    b1 = np.interp(grid, times, signal.real) + 1j * np.interp(grid, times, signal.imag)
    b1 = b1 * np.exp(1j * (2.0 * np.pi * float(getattr(rf, "freq_offset", 0.0)) * grid + float(rf.phase_offset)))
    return grid + float(getattr(rf, "delay", 0.0)), b1


def sample_gradient(event, t):
    """Sample one Pulseq gradient event (Hz/m) on the absolute time base ``t``."""
    if event is None:
        return np.zeros_like(t)
    delay = float(getattr(event, "delay", 0.0))
    if getattr(event, "type", "grad") == "trap":
        rise, flat, fall = event.rise_time, event.flat_time, event.fall_time
        knots = delay + np.array([0.0, rise, rise + flat, rise + flat + fall])
        values = np.array([0.0, event.amplitude, event.amplitude, 0.0])
        return np.interp(t, knots, values, left=0.0, right=0.0)
    waveform = np.asarray(event.waveform, dtype=float)
    tt = delay + np.asarray(event.tt, dtype=float)
    return np.interp(t, tt, waveform, left=0.0, right=0.0)


def _channel(module, axis):
    for gradient in getattr(module, "gradients", ()) or ():
        if gradient is not None and gradient.channel == axis:
            return gradient
    return None


def _event_end(event):
    """Absolute end time (s) of one RF or gradient event."""
    delay = float(getattr(event, "delay", 0.0))
    kind = getattr(event, "type", None)
    if kind == "trap":
        return delay + event.rise_time + event.flat_time + event.fall_time
    if kind == "rf":
        return delay + float(np.asarray(event.t)[-1])
    return delay + float(np.asarray(event.tt)[-1])


def _events(module, name):
    return [event for event in (getattr(module, name, ()) or ()) if event is not None]


def module_gradients(module, dt=2e-6):
    """Sample every gradient the module plays, selection *and* rephasing.

    Rephasers live in the block after the RF, so they are shifted by the RF
    block's duration: what comes back is the module's whole gradient timeline,
    on one absolute time base.

    Returns
    -------
    tuple
        ``(t, {channel: waveform_hz_per_m})``; the mapping is empty for a
        non-selective pulse.
    """
    rf = getattr(module, "rf", None)
    selection, rephasers = _events(module, "gradients"), _events(module, "rephasers")
    if not selection and not rephasers:
        return np.zeros(0), {}
    block_end = max([_event_end(event) for event in selection] + ([_event_end(rf)] if rf is not None else [0.0]))
    total = block_end + max((_event_end(event) for event in rephasers), default=0.0)
    t = np.arange(0.0, total + dt, dt)
    traces = {}
    for channel in ("x", "y", "z"):
        trace = np.zeros_like(t)
        for event in selection:
            if event.channel == channel:
                trace += sample_gradient(event, t)
        for event in rephasers:
            if event.channel == channel:
                trace += sample_gradient(event, t - block_end)
        if np.any(trace):
            traces[channel] = trace
    return t, traces


# --------------------------------------------------------------------------
# Bloch integration
# --------------------------------------------------------------------------


def bloch(b1_hz, bz_hz, dt):
    """Integrate the Bloch equation, ignoring relaxation.

    Parameters
    ----------
    b1_hz : numpy.ndarray
        Complex transverse field per time step, shape ``(T,)``, in Hz.
    bz_hz : numpy.ndarray
        Longitudinal field, shape ``(P, T)`` or ``(P, 1)``, in Hz.
    dt : float
        Raster step (s).

    Returns
    -------
    numpy.ndarray
        Final magnetisation, shape ``(P, 3)``, from ``M = +z``.
    """
    b1_hz = np.asarray(b1_hz, dtype=complex)
    bz_hz = np.atleast_2d(np.asarray(bz_hz, dtype=float))
    n_pos, n_steps = bz_hz.shape[0], len(b1_hz)
    magnetisation = np.zeros((n_pos, 3))
    magnetisation[:, 2] = 1.0
    two_pi_dt = 2.0 * np.pi * dt
    for step in range(n_steps):
        omega = np.empty((n_pos, 3))
        omega[:, 0] = two_pi_dt * b1_hz[step].real
        omega[:, 1] = two_pi_dt * b1_hz[step].imag
        omega[:, 2] = two_pi_dt * bz_hz[:, step if bz_hz.shape[1] > 1 else 0]
        angle = np.linalg.norm(omega, axis=1)
        active = angle > 1e-15
        if not np.any(active):
            continue
        axis = np.zeros_like(omega)
        axis[active] = omega[active] / angle[active, None]
        cosine = np.cos(angle)[:, None]
        sine = np.sin(angle)[:, None]
        cross = np.cross(axis, magnetisation)
        dot = np.sum(axis * magnetisation, axis=1)[:, None]
        magnetisation = magnetisation * cosine + cross * sine + axis * dot * (1.0 - cosine)
    return magnetisation


# --------------------------------------------------------------------------
# profiles
# --------------------------------------------------------------------------


_GRADIENT_COLOURS = {"x": "tab:red", "y": "tab:green", "z": "tab:purple"}


def _envelope_axis(ax, t, b1, title):
    ax.plot(t * 1e3, b1.real, lw=1.2, label="real")
    ax.plot(t * 1e3, b1.imag, lw=1.2, label="imag")
    ax.plot(t * 1e3, np.abs(b1), lw=0.9, ls="--", color="0.4", label="|B1|")
    ax.set_ylabel("B1 [Hz]")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper right")


def _gradient_axis(ax, module):
    """Draw the module's gradient timeline; return False if it plays none."""
    t, traces = module_gradients(module)
    if not traces:
        return False
    for channel, trace in traces.items():
        ax.plot(t * 1e3, trace * 1e-3, lw=1.1, color=_GRADIENT_COLOURS[channel], label=f"G{channel}")
    ax.axhline(0.0, lw=0.6, color="0.7")
    ax.set_xlabel("t [ms]")
    ax.set_ylabel("G [kHz/m]")
    ax.legend(fontsize=7, loc="upper right")
    return True


def _pulse_panels(module, *, with_profile):
    """Lay out the RF row, an optional gradient row, and an optional profile.

    The gradient row appears only when the module actually carries gradients,
    so a non-selective pulse still renders as a single envelope panel.
    """
    has_gradients = bool(module_gradients(module)[1])
    if with_profile:
        width, ratios = 9.5, [1.15, 1.0]
    else:
        width, ratios = 5.6, [1.0]
    height = 4.2 if has_gradients else 3.3
    rows = [["rf", "profile"]] if with_profile else [["rf"]]
    if has_gradients:
        rows.append(["gradient", "profile"] if with_profile else ["gradient"])
    fig, axes = plt.subplot_mosaic(
        rows,
        figsize=(width, height),
        width_ratios=ratios,
        height_ratios=[1.0, 0.7][: len(rows)],
        layout="constrained",
    )
    if has_gradients:
        axes["rf"].tick_params(labelbottom=False)
    else:
        axes["rf"].set_xlabel("t [ms]")
    return fig, axes, has_gradients


def _finish_pulse_panels(module, axes, t, b1, title, has_gradients):
    _envelope_axis(axes["rf"], t, b1, title)
    if has_gradients:
        _gradient_axis(axes["gradient"], module)
        axes["gradient"].sharex(axes["rf"])


def _transverse(magnetisation):
    return np.hypot(magnetisation[:, 0], magnetisation[:, 1])


def rf_figure(module, kind="none", *, title=None, extent=None, n_points=201, span=None, dt=None):
    """Draw an RF pulse and the profile it actually produces.

    The left column is the pulse itself: the complex B1 envelope, and — when
    the module carries any — the gradient timeline underneath it, selection
    and rephasing lobes alike, on a shared time axis. The right column is the
    simulated profile.

    Parameters
    ----------
    module : SequenceModule
        RF module from any ``pulserver.pypulseq`` pulse factory.
    kind : {'none', 'slice', 'frequency', 'xy', 'zf'}
        Which profile to simulate. ``'none'`` draws the pulse only, which is
        all a non-selective pulse has to show.
    title : str, optional
        Title of the envelope panel.
    extent : float, optional
        Half-width of the simulated axis: metres for ``'slice'`` and
        ``'xy'``, Hz for ``'frequency'``.
    n_points : int, optional
        Samples along each simulated axis.
    span : float, optional
        Half-width of the frequency axis (Hz) for ``'zf'``.
    dt : float, optional
        Simulation raster (s); defaults to the RF event's own raster.
    """
    if kind not in ("none", "slice", "frequency", "xy", "zf"):
        raise ValueError("kind must be none, slice, frequency, xy, or zf")
    rf = getattr(module, "rf", module)
    t, b1 = _rf_grid(rf, dt)
    step = float(t[1] - t[0])
    title = title or type(module).__name__

    fig, axes, has_gradients = _pulse_panels(module, with_profile=kind != "none")
    _finish_pulse_panels(module, axes, t, b1, title, has_gradients)
    if kind == "none":
        return fig
    profile_axis = axes["profile"]

    if kind in ("xy", "zf"):
        if kind == "xy":
            extent = extent or 0.08
            first = np.linspace(-extent, extent, min(n_points, 81))
            second = first
            bz_first = sample_gradient(_channel(module, "x"), t)
            bz_second = sample_gradient(_channel(module, "y"), t)
            labels = ("x [mm]", "y [mm]")
            limits = [-extent * 1e3, extent * 1e3, -extent * 1e3, extent * 1e3]
            aspect = "equal"
        else:
            extent, span = extent or 0.02, span or 600.0
            first = np.linspace(-extent, extent, min(n_points, 81))
            second = np.linspace(-span, span, min(n_points, 81))
            bz_first = sample_gradient(_channel(module, "z"), t)
            bz_second = None
            labels = ("z [mm]", "off-resonance [Hz]")
            limits = [-extent * 1e3, extent * 1e3, -span, span]
            aspect = "auto"
        grid_a, grid_b = np.meshgrid(first, second, indexing="ij")
        bz = np.outer(grid_a.ravel(), bz_first)
        bz = bz + (np.outer(grid_b.ravel(), bz_second) if bz_second is not None else grid_b.ravel()[:, None])
        profile = _transverse(bloch(b1, bz, step)).reshape(grid_a.shape)
        image = profile_axis.imshow(profile.T, origin="lower", aspect=aspect, cmap="viridis", extent=limits)
        profile_axis.set_xlabel(labels[0])
        profile_axis.set_ylabel(labels[1])
        profile_axis.set_title("|Mxy| profile", fontsize=9)
        fig.colorbar(image, ax=profile_axis, shrink=0.85)
        return fig

    if kind == "slice":
        extent = extent or 0.02
        axis = np.linspace(-extent, extent, n_points)
        bz = np.outer(axis, sample_gradient(_channel(module, "z"), t))
        xlabel, scale = "z [mm]", 1e3
    else:
        extent = extent or 600.0
        axis = np.linspace(-extent, extent, n_points)
        bz = axis[:, None]
        xlabel, scale = "off-resonance [Hz]", 1.0

    magnetisation = bloch(b1, bz, step)
    profile_axis.plot(axis * scale, _transverse(magnetisation), lw=1.3, label="|Mxy|")
    profile_axis.plot(axis * scale, magnetisation[:, 2], lw=1.0, ls="--", label="Mz")
    profile_axis.set_xlabel(xlabel)
    profile_axis.set_ylabel("M / M0")
    profile_axis.set_ylim(-1.05, 1.05)
    profile_axis.set_title("simulated profile", fontsize=9)
    profile_axis.legend(fontsize=7)
    return fig


def rf_comparison(entries, kind="slice", *, extent=None, n_points=201, title=None):
    """Overlay the simulated profiles of several pulses on one axis.

    Parameters
    ----------
    entries : sequence of tuple
        ``(label, module)`` pairs.
    kind : {'slice', 'frequency'}
        Profile axis to simulate.
    extent : float, optional
        Half-width of the simulated axis (m, or Hz for ``'frequency'``).
    n_points : int, optional
        Samples along the axis.
    title : str, optional
        Axis title.
    """
    if kind == "slice":
        extent = extent or 0.02
        xlabel, scale = "z [mm]", 1e3
    else:
        extent = extent or 1500.0
        xlabel, scale = "off-resonance [Hz]", 1.0
    axis = np.linspace(-extent, extent, n_points)
    with_gradients = any(module_gradients(module)[1] for _, module in entries)

    rows = [["rf", "profile"]] + ([["gradient", "profile"]] if with_gradients else [])
    fig, axes = plt.subplot_mosaic(
        rows,
        figsize=(9.5, 4.2 if with_gradients else 3.3),
        width_ratios=[1.15, 1.0],
        height_ratios=[1.0, 0.7][: len(rows)],
        layout="constrained",
    )
    for index, (label, module) in enumerate(entries):
        colour = f"C{index}"
        rf = getattr(module, "rf", module)
        t, b1 = _rf_grid(rf)
        step = float(t[1] - t[0])
        bz = np.outer(axis, sample_gradient(_channel(module, "z"), t)) if kind == "slice" else axis[:, None]
        axes["rf"].plot((t - t[0]) * 1e3, np.abs(b1), lw=1.1, color=colour, label=label)
        axes["profile"].plot(axis * scale, _transverse(bloch(b1, bz, step)), lw=1.2, color=colour, label=label)
        if with_gradients:
            grad_t, traces = module_gradients(module)
            for trace in traces.values():
                axes["gradient"].plot(grad_t * 1e3, trace * 1e-3, lw=1.1, color=colour)
    axes["rf"].set_ylabel("|B1| [Hz]")
    axes["rf"].set_title(title or "envelopes", fontsize=9)
    axes["rf"].legend(fontsize=7)
    if with_gradients:
        axes["rf"].tick_params(labelbottom=False)
        axes["gradient"].axhline(0.0, lw=0.6, color="0.7")
        axes["gradient"].set_xlabel("t [ms]")
        axes["gradient"].set_ylabel("G [kHz/m]")
        axes["gradient"].sharex(axes["rf"])
    else:
        axes["rf"].set_xlabel("t [ms]")
    axes["profile"].set_xlabel(xlabel)
    axes["profile"].set_ylabel("|Mxy| / M0")
    axes["profile"].set_title("simulated profiles", fontsize=9)
    axes["profile"].legend(fontsize=7)
    return fig


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


def mask_figure(entries, *, cmap="viridis"):
    """Show one or more Cartesian sampling masks side by side."""
    fig, axes = plt.subplots(1, len(entries), figsize=(3.2 * len(entries), 3.3), squeeze=False)
    for ax, (title, mask) in zip(axes[0], entries, strict=True):
        ax.imshow(np.asarray(mask).T, cmap=cmap, origin="lower", interpolation="nearest", vmin=0, vmax=1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("ky")
        ax.set_ylabel("kz")
    fig.tight_layout()
    return fig


def order_figure(entries, coords, *, cmap="viridis", label="echo index"):
    """Colour a (ky, kz) point set by the echo index each ordering assigns it."""
    coords = np.asarray(coords, dtype=float)
    fig, axes = plt.subplots(1, len(entries), figsize=(3.1 * len(entries), 3.4), sharey=True, squeeze=False)
    image = None
    for ax, (title, shots) in zip(axes[0], entries, strict=True):
        echo = np.zeros(len(coords))
        for shot in shots:
            for index, point in enumerate(shot):
                echo[point] = index
        image = ax.scatter(coords[:, 0], coords[:, 1], c=echo, s=6, cmap=cmap)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("ky")
        ax.set_aspect("equal")
    axes[0][0].set_ylabel("kz")
    fig.colorbar(image, ax=axes[0], label=label, shrink=0.85)
    return fig


def pattern_figure(pattern, *, title="", cmap="viridis"):
    """Colour a Cartesian plan's support by shot index and by position in the shot."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    support = np.asarray(pattern.support, dtype=float)
    if support.shape[1] == 1:
        support = np.column_stack([support[:, 0], np.zeros(len(support))])
    shot_index = np.zeros(len(support))
    echo_index = np.zeros(len(support))
    for shot, indices in enumerate(pattern.order):
        for position, point in enumerate(indices):
            shot_index[point] = shot
            echo_index[point] = position
    for ax, values, name in ((left, shot_index, "shot index"), (right, echo_index, "echo index")):
        image = ax.scatter(support[:, 0], support[:, 1], c=values, s=10, cmap=cmap)
        ax.set_xlabel("ky")
        ax.set_aspect("equal")
        ax.set_title(f"{title} — {name}" if title else name, fontsize=9)
        fig.colorbar(image, ax=ax, shrink=0.85)
    left.set_ylabel("kz")
    fig.tight_layout()
    return fig
